"""ReasonSeg query rewriting with Qwen3-VL (image-grounded -> explicit referring expr).

ReasonSeg queries are *implicit*: they describe the target indirectly (a reasoning
question, a function, a property) and never name it -- and the answer is image
dependent ("the source of power for the ship" is a sail on one image, a funnel on
another). The Vision Banana paper puts Gemini 2.5 Pro in front as a single-turn
rewriter (image + implicit query -> one explicit referring expression) so the
segmenter receives a groundable phrase. We reproduce that with an open, offline
**Qwen3-VL-32B-Thinking** pass so our ReasonSeg gIoU is comparable. The same pass
also runs Qwen3-VL MoE (e.g. 30B-A3B) and Gemma 4 instruct models; Gemma 4 needs
transformers>=5 (requirements/reasonseg-rewrite-gemma.txt).

This mirrors the SA-Co/Gold gate (evaluation/instanceseg/gate_saco_gold_qwen.py):
a separate precompute pass (it CANNOT co-reside with FLUX on one GPU at 32B) whose
output is a cached JSON the refseg evaluator consumes. The rewrite depends only on
(image, query) -- NOT the checkpoint under test -- so it is computed once and reused
across every checkpoint evaluation.

Output cache (``--out``), keyed by the evaluator's ``output_stem`` so the keys line
up with inference_flux2_klein_refseg.py (build the rewrite with the SAME
--reasonseg_expr_index / --reasonseg_all_expr as the eval):
    {"<stem>": {"query": "<implicit query>", "target": "<referring expression>",
                "image": "<file.jpg>"}}

Sharding (array style, one GPU per shard): pass --num_shards/--shard_index (and mask
one GPU per task with CUDA_VISIBLE_DEVICES). Each shard writes
rewrite_shards/<out stem>/rank_XXXXX.json; shard 0 waits for the rest and merges
into --out.

Failure handling: an entry is unusable if its target is missing, is still the raw
implicit query, or is leaked reasoning prose (the <think> block overran
--max_new_tokens, so generation stopped before the JSON). Those are listed in
``<out>.failed.json`` and the script exits 1 -- pass --allow_fallback to accept
anyway, or re-run with --repair to retry ONLY those stems (at
--repair_max_new_tokens, since a truncated think block is the usual cause) and
merge them back into --out.

Usage:
    python evaluation/refseg/rewrite_reasonseg_qwen.py \
        --qwen_model Qwen/Qwen3-VL-32B-Thinking \
        --data_dir datasets_archives/refseg/reasonseg_val \
        --out datasets_archives/refseg/reasonseg_rewrites_qwen3vl32bthink.json

    # retry just the entries that failed above, in place
    python evaluation/refseg/rewrite_reasonseg_qwen.py --repair \
        --out datasets_archives/refseg/reasonseg_rewrites_qwen3vl32bthink.json

    # Gemma 4 (instruct, greedy), with transformers>=5 on PYTHONPATH
    python evaluation/refseg/rewrite_reasonseg_qwen.py --rewriter gemma4_31b \
        --data_dir datasets_archives/refseg/reasonseg_val \
        --out datasets_archives/refseg/reasonseg_rewrites_gemma4_31b.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Reuse the reasonseg sample builder + sharding/wait helpers from the evaluator.
import inference_flux2_klein_refseg as refseg  # noqa: E402

REFSEG_ROOT = refseg.REFSEG_ROOT
# --rewriter: model + decoding of each bundled cache (evaluation/assets/
# reasonseg_val_rewrites_<name>.json). The default output is
# REFSEG_ROOT/reasonseg_rewrites_<name>.json; explicit flags override any value.
REWRITERS = {
    # Qwen3 Thinking recommended sampling; greedy can loop. Room for <think> + JSON.
    "qwen3vl32bthink": dict(qwen_model="Qwen/Qwen3-VL-32B-Thinking", temperature=0.6,
                            top_p=0.95, top_k=20, max_new_tokens=1536),
    # Instruct model: greedy, answers are one short JSON object. Needs transformers>=5.
    "gemma4_31b": dict(qwen_model="google/gemma-4-31B-it", temperature=0.0,
                       top_p=0.95, top_k=20, max_new_tokens=256),
}

# --------------------------------------------------------------------------- #
# Prompts (verbatim contract: one image-grounded referring expression as JSON).
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """\
You are a query-rewriting module in a referring-segmentation pipeline.
You are given an image and a QUERY. The QUERY is implicit: it describes the
target object indirectly (a reasoning question, a function, a property) and
does NOT name it. Your job is to identify which visible object(s) in THIS
image the query points to, and rewrite it as one short, explicit referring
expression that a segmentation model can ground.

Rules:
- Look at the image first. Decide which actual, visible object satisfies the
  query in this specific image. Resolve the concept to the thing you see.
- Output ONE concrete referring expression: a noun phrase naming the object
  category plus only the visual attributes (color, material, shape, position,
  relative location) needed to uniquely pick it out in this image.
- Describe the visible object, not the abstract idea.
  "the sour food" -> name the fruit you actually see, e.g. "the lemon slices on the white plate".
- If several objects of the same kind all match, use a plural/collective phrase.
- Keep it under ~15 words. No full sentences, no questions, no explanation, no markdown.
- Never answer the query conversationally and never refuse. Always return the
  most plausible single target as a referring expression.

Return strictly this JSON and nothing else:
{"target": "<referring expression>"}"""

USER_TEMPLATE = (
    "QUERY: {query}\n"
    "Return the referring expression for the object to segment."
)


# --------------------------------------------------------------------------- #
# Tolerant output parsing (Thinking models emit a <think>...</think> block first).
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_CLOSED_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_target(text: str) -> tuple[str, bool]:
    """Pull the referring expression out of a (possibly Thinking) generation.

    Returns (target, parsed_ok). Strips the reasoning block, prefers a fenced /
    last JSON object's ``target`` field, and falls back to the last non-empty line
    -- but only when that line still *looks* like a referring expression. When the
    <think> block runs past --max_new_tokens the generation stops before the JSON
    and its last line is a slab of reasoning monologue; accepting that would put a
    paragraph in the segmenter's prompt, undetectably. Returning ('', False) sends
    the caller down the raw-query fallback instead, which the cache gate flags and
    --repair fixes.
    """
    t = text or ""
    low = t.lower()
    if "</think>" in low:
        # Keep only what follows the reasoning block (handles an unclosed <think>).
        t = t[low.rfind("</think>") + len("</think>"):]
    else:
        t = _CLOSED_THINK_RE.sub("", t)

    fences = _FENCE_RE.findall(t)
    for chunk in ([fences[-1]] if fences else []) + [t]:
        m = _OBJ_RE.search(chunk)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
            if isinstance(obj, dict) and str(obj.get("target", "")).strip():
                return str(obj["target"]).strip(), True

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines:
        cand = lines[-1].strip("`").strip('"').strip()
        if refseg.looks_like_referring_expr(cand):
            return cand, False
    return "", False


# --------------------------------------------------------------------------- #
# Model loading: Qwen3-VL (dense or MoE) and Gemma 4.
# --------------------------------------------------------------------------- #
def load_qwen(model_path: str, flash_attention: bool):
    import os

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not visible to this rewrite process, so Qwen3-VL cannot run. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}. "
            "This is usually a stale CUDA_VISIBLE_DEVICES (e.g. '0,1,2,3' on a 1-GPU node) "
            "or a transient driver/NVML error. Pin CUDA_VISIBLE_DEVICES to the allocated "
            "GPU(s) and resubmit."
        )

    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    # Auto class: dense Qwen3-VL, Qwen3-VL MoE (30B-A3B) and Gemma 4 (needs transformers>=5).
    # Weights download on first use unless HF_HUB_OFFLINE=1 (then a cached snapshot is needed).
    try:
        config = AutoConfig.from_pretrained(model_path)
    except (KeyError, ValueError) as e:
        import transformers
        raise RuntimeError(
            f"transformers {transformers.__version__} cannot load {model_path}: {e}. Gemma 4 needs "
            "transformers>=5: pip install --target <dir> -r requirements/reasonseg-rewrite-gemma.txt "
            "and put <dir> first on PYTHONPATH for this pass.") from e
    qwen = "qwen" in config.model_type
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if flash_attention and qwen:
        kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
    processor = AutoProcessor.from_pretrained(model_path)
    process_vision_info = None
    if qwen:
        from qwen_vl_utils import process_vision_info
    return model, processor, process_vision_info


def rewrite(model, processor, process_vision_info, query: str, image_path: Path,
            max_new_tokens: int, temperature: float, top_p: float, top_k: int,
            seed: int) -> str:
    import torch

    user_text = {"type": "text", "text": USER_TEMPLATE.format(query=query)}
    if process_vision_info is not None:  # Qwen3-VL
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image", "image": str(image_path)}, user_text]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
    else:  # Gemma 4 and other HF-native processors (same path as the SA-Co gate)
        from PIL import Image

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": Image.open(image_path).convert("RGB")}, user_text]},
        ]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt").to(model.device)
    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if temperature and temperature > 0:
        torch.manual_seed(seed)  # fixed seed -> reproducible despite sampling
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
    else:
        gen_kwargs.update(do_sample=False)  # greedy (discouraged for Thinking: may loop)
    generated = model.generate(**inputs, **gen_kwargs)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    return processor.batch_decode(trimmed, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rewriter", choices=REWRITERS, default="qwen3vl32bthink",
                   help="Model and decoding of a bundled cache (default qwen3vl32bthink, the "
                        "paper cache; gemma4_31b needs transformers>=5). Flags below override.")
    p.add_argument("--out", type=Path, default=None,
                   help="Final merged rewrites JSON (keyed by evaluator output_stem). "
                        "Default: <REFSEG_ROOT>/reasonseg_rewrites_<rewriter>.json.")
    p.add_argument("--data_dir", type=Path, default=refseg.REFSEG_ROOT / "reasonseg_val",
                   help="ReasonSeg val dir (holds <stem>.jpg/.json and masks/).")
    p.add_argument("--qwen_model", default=None,
                   help="VLM HF id or local snapshot (Qwen3-VL, Qwen3-VL MoE, Gemma 4). "
                        "Default: the --rewriter model.")
    # Must match the eval's sample-building so output_stem keys line up.
    p.add_argument("--reasonseg_expr_index", type=int, default=0)
    p.add_argument("--reasonseg_all_expr", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Cap samples (after stem sort).")
    p.add_argument("--max_new_tokens", type=int, default=None,
                   help="Generation budget (default: 1536 for Qwen3 Thinking, room for the "
                        "<think> block + JSON; 256 for Gemma 4).")
    p.add_argument("--repair", action="store_true",
                   help="Load the existing --out and re-run ONLY the unusable entries (missing / "
                        "still the raw query / leaked reasoning); merge back into --out. Uses "
                        "--repair_max_new_tokens, since truncated thinking is the usual cause.")
    p.add_argument("--repair_max_new_tokens", type=int, default=4096,
                   help="--repair only: generation budget for the retried stems (default 4096). "
                        "The entries that need repair are exactly the ones whose reasoning did "
                        "not fit the first pass, so retrying at the same budget just repeats it.")
    p.add_argument("--allow_fallback", action="store_true",
                   help="Exit 0 even if some entries are still unusable. Default: write the cache "
                        "but exit 1, so a degraded cache never reaches an eval unnoticed.")
    # Sampling defaults come from --rewriter. Greedy (temperature 0) is discouraged for
    # Thinking models -- it can loop. A fixed --seed keeps a sampled pass reproducible.
    p.add_argument("--temperature", type=float, default=None,
                   help="Sampling temperature, 0 = greedy (default: 0.6 Qwen3 Thinking, 0 Gemma 4).")
    p.add_argument("--top_p", type=float, default=None, help="Nucleus top_p when sampling (default 0.95).")
    p.add_argument("--top_k", type=int, default=None, help="top_k when sampling (default 20).")
    p.add_argument("--seed", type=int, default=0, help="Sampling seed when temperature>0.")
    p.add_argument("--no_flash_attention", action="store_true")
    p.add_argument("--save_review", action="store_true",
                   help="Also write <out>.review.tsv (stem, image, query, target) to eyeball.")
    # array-style sharding (mirrors the gate)
    p.add_argument("--num_shards", type=int, default=None)
    p.add_argument("--shard_index", type=int, default=None)
    p.add_argument("--local_rank", type=int, default=None)
    p.add_argument("--shard_aggregate_timeout", type=int, default=7200)
    p.add_argument("--merge_only", action="store_true",
                   help="Skip inference: just merge existing rewrite_shards/ into --out.")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    for key, value in REWRITERS[args.rewriter].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.out is None:
        args.out = REFSEG_ROOT / f"reasonseg_rewrites_{args.rewriter}.json"
    return args


def main() -> None:
    args = parse_args()
    shard = refseg.setup_sharding(args)
    # Per-output shard dir, so rewrite passes for different models can run at once.
    shard_dir = args.out.parent / "rewrite_shards" / args.out.stem
    shard_dir.mkdir(parents=True, exist_ok=True)

    # --repair keeps the good entries of an existing cache and re-runs only the rest.
    # Read up front (before any shard writes) so every array task sees the same cache.
    cached: dict[str, dict] = {}
    if args.repair:
        if not args.out.exists():
            raise FileNotFoundError(
                f"--repair needs an existing cache at {args.out} (run a full pass first).")
        cached = json.loads(args.out.read_text())

    def merge_and_write() -> dict[str, str]:
        """Merge the shard files (over the repaired-from cache) into --out and
        return {stem: problem} for every entry that is still unusable."""
        if shard.world_size > 1 and not shard.distributed:
            refseg.wait_for_shard_metrics(
                [shard_dir / f"rank_{r:05d}_of_{shard.world_size:05d}.json"
                 for r in range(shard.world_size)],
                args.shard_aggregate_timeout)
        # In repair mode the shards only cover the broken stems: start from the
        # existing cache so the entries we deliberately skipped survive the merge.
        merged: dict[str, dict] = dict(cached)
        n_fallback = 0
        for r in range(shard.world_size):
            data = json.loads((shard_dir / f"rank_{r:05d}_of_{shard.world_size:05d}.json").read_text())
            merged.update(data.get("rewrites", {}))
            n_fallback += int(data.get("n_fallback", 0))
        # Stable key order for a readable, diff-able cache.
        merged = {k: merged[k] for k in sorted(merged)}
        args.out.write_text(json.dumps(merged, indent=2))
        print(f"Wrote {args.out}: {len(merged)} rewrites "
              f"({n_fallback} fell back to raw-line/original).")

        # Fail-loud sidecar: entries that would silently poison a downstream gIoU.
        problems = {k: p for k, v in merged.items()
                    if (p := refseg.rewrite_entry_problem(v)) is not None}
        side = args.out.with_suffix(args.out.suffix + ".failed.json")
        if problems:
            side.write_text(json.dumps(problems, indent=2))
            print(f"  {len(problems)} entries still unusable -> {side}")
        elif side.exists():
            side.unlink()  # stale sidecar from an earlier degraded run
        if args.save_review:
            review = args.out.with_suffix(args.out.suffix + ".review.tsv")
            with review.open("w") as fh:
                fh.write("stem\timage\tquery\ttarget\n")
                for stem, e in merged.items():
                    q = e.get("query", "").replace("\t", " ").replace("\n", " ")
                    t = e.get("target", "").replace("\t", " ").replace("\n", " ")
                    fh.write(f"{stem}\t{e.get('image','')}\t{q}\t{t}\n")
            print(f"Wrote review TSV: {review}")
        return problems

    def finish(problems: dict[str, str]) -> None:
        """Tear down, then exit non-zero if the cache is still degraded."""
        refseg.cleanup_distributed(shard)
        if problems and not args.allow_fallback:
            sys.exit(
                f"ERROR: {len(problems)} entries in {args.out} are not usable referring "
                "expressions (see the .failed.json sidecar). A gIoU scored on this cache is "
                "not protocol-comparable. Re-run with --repair to retry just those stems "
                "(--allow_fallback to accept anyway).")

    if args.merge_only:
        problems = merge_and_write() if shard.rank == 0 else {}
        finish(problems)
        return

    all_samples = refseg.build_reasonseg_samples(
        args.data_dir, args.reasonseg_expr_index, args.reasonseg_all_expr, args.limit)

    # --repair: drop the stems that already hold a good rewrite. Filtered BEFORE
    # sharding so every shard slices the same global todo list.
    max_new_tokens = args.max_new_tokens
    if args.repair:
        todo = [(s, refseg.rewrite_entry_problem(cached.get(s.output_stem))) for s in all_samples]
        todo = [(s, p) for s, p in todo if p is not None]
        if shard.rank == 0:
            by_class = Counter(p for _, p in todo)
            summary = ", ".join(f"{n} {p}" for p, n in sorted(by_class.items())) or "none"
            print(f"--repair: {len(cached)} cached entries, "
                  f"{len(all_samples) - len(todo)} already good, re-running {len(todo)} ({summary}).")
            for s, p in todo[:10]:
                old = str(cached.get(s.output_stem, {}).get("target", ""))
                print(f"    [{p}] {s.output_stem}: {old[:70]!r}")
        all_samples = [s for s, _ in todo]
        max_new_tokens = args.repair_max_new_tokens

    samples = list(all_samples[shard.rank :: shard.world_size])

    if args.dry_run:
        print(f"ReasonSeg rewrite: {len(all_samples)} samples; shard "
              f"{shard.rank}/{shard.world_size} -> {len(samples)}. Model: {args.qwen_model}"
              f" | max_new_tokens: {max_new_tokens}")
        for s in samples[:5]:
            print(f"  [{s.output_stem}] query={s.expr!r}  img={s.image_path.name}")
        refseg.cleanup_distributed(shard)
        return

    if args.repair and not all_samples:
        # Nothing to re-run: leave --out untouched (merging here would fold in
        # shard files from an unrelated earlier pass).
        print("--repair: nothing to repair, every cached entry is a usable rewrite.")
        finish({})
        return

    model, processor, pvi = load_qwen(args.qwen_model, flash_attention=not args.no_flash_attention)
    rewrites: dict[str, dict] = {}
    n_fallback = 0
    for s in tqdm(samples, desc=f"Rewrite rank {shard.rank}/{shard.world_size}"):
        raw = rewrite(model, processor, pvi, s.expr, s.image_path,
                      max_new_tokens, args.temperature, args.top_p, args.top_k, args.seed)
        target, ok = extract_target(raw)
        if not target:                 # never drop a sample: fall back to the raw query
            target, ok = s.expr, False
        if not ok:
            n_fallback += 1
        rewrites[s.output_stem] = {
            "query": s.expr, "target": target, "image": s.image_path.name,
        }
    if n_fallback:
        print(f"  {n_fallback}/{len(samples)} generations unparsed -> raw-line/original fallback.")

    shard_path = shard_dir / f"rank_{shard.rank:05d}_of_{shard.world_size:05d}.json"
    shard_path.write_text(json.dumps({
        "rank": shard.rank, "world_size": shard.world_size,
        "qwen_model": args.qwen_model, "temperature": args.temperature,
        "max_new_tokens": max_new_tokens, "repair": args.repair,
        "n_fallback": n_fallback, "rewrites": rewrites,
    }))

    refseg.barrier(shard)
    problems = merge_and_write() if shard.rank == 0 else {}
    finish(problems)


if __name__ == "__main__":
    main()
