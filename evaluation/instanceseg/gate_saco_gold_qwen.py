"""Image-level presence gate for SA-Co/Gold, run locally with Qwen3-VL.

The Vision Banana paper gates each (image, noun-phrase) query with Gemini 3.1
Flash-Lite before generating masks -- SA-Co/Gold is ~83% negatives, and emitting
nothing for a negative is what drives the image-level IL_MCC term of CGF1. We
substitute the open-source **Qwen3-VL** run in-venv (mirroring the local Qwen-VL
ImgEdit judge in evaluation/imgedit/judge_imgedit_qwen.py) so the whole suite
stays offline -- no API key, no per-call network.

For each query we ask the paper's verbatim question:
    "Is there an instance of <NP> visible in this image? Choose your answer from
     the following options: (A): Yes, (B): No."
and parse (A)=positive / (B)=negative. The decisions are written to a cached
``gate_decisions.json`` ({subset: {image_id: bool}}) that
inference_flux2_klein_instanceseg.py consumes via --gate file. The gate depends
only on (image, NP) -- NOT on the checkpoint under test -- so it is computed once
and reused across every checkpoint evaluation.

--gate oracle derives presence from the GT (no model, no network) -- a debug
upper bound on gate quality, and a way to produce a valid gate file for testing.

Sharding (array style, one GPU per shard): pass --num_shards/--shard_index (and
mask one GPU per task with CUDA_VISIBLE_DEVICES). Each shard writes
gate_shards/rank_XXXXX.json; shard 0 waits for the rest and merges.

Use an **Instruct** variant (default Qwen3-VL-32B-Instruct; Qwen3-VL-8B-Instruct
is the lighter/faster alternative): the gate needs one deterministic single-letter
answer per query. Thinking variants are not supported: a <think> block routinely
exceeds the token budget on this yes/no task, and every truncated reply falls back
to the default verdict.

Usage:
    # one-time gate over all subsets (needs a cached Qwen3-VL snapshot)
    python evaluation/instanceseg/gate_saco_gold_qwen.py \
        --qwen_model Qwen/Qwen3-VL-32B-Instruct --out <dir>/gate_decisions.json
    # lighter/faster 8B Instruct gate (write to a distinct file to keep both)
    python evaluation/instanceseg/gate_saco_gold_qwen.py \
        --qwen_model Qwen/Qwen3-VL-8B-Instruct --out <dir>/gate_decisions_qwen8b.json
    # offline oracle gate file (no model)
    python evaluation/instanceseg/gate_saco_gold_qwen.py --gate oracle --out gate_oracle.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Reuse the sample builder + sharding/wait helpers from the inference evaluator.
import inference_flux2_klein_instanceseg as iseg  # noqa: E402
from eval_saco_gold import SACO_GOLD_SUBSETS  # noqa: E402

GATE_QUESTION = (
    'Is there an instance of "{np}" visible in this image? Choose your answer '
    "from the following options: (A): Yes, (B): No."
)


def parse_yes_no(text: str) -> Optional[bool]:
    """Parse a Qwen reply into True (yes / option A) or False (no / option B).

    Returns None if the answer is unintelligible (caller decides the default)."""
    t = text.strip().lower()
    # Prefer the explicit option letter.
    if re.search(r"\(?\s*a\s*\)|option a|:\s*a\b", t):
        return True
    if re.search(r"\(?\s*b\s*\)|option b|:\s*b\b", t):
        return False
    # Fall back to the yes/no word (whole-word to avoid 'no' inside other words).
    if re.search(r"\byes\b", t):
        return True
    if re.search(r"\bno\b|\bnot\b", t):
        return False
    return None


def save_gate_panel(image_path, np_text, verdict, gt_present, source, out_png):
    """Save the gated image with a caption (NP + verdict + GT presence) for review."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    v = "YES" if verdict else "NO"
    g = "present" if gt_present else "absent"
    title = f'NP: "{np_text}"   {source}: {v}   GT: {g}'
    panel = iseg.refseg.make_panel([(img, Image.LANCZOS)], labels=[v], title=title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_png)


def load_qwen(model_path: str, flash_attention: bool):
    import os
    import torch

    # device_map="auto" + flash_attention_2 lands on CPU and raises a cryptic
    # "FlashAttention2 not available on CPU" error when CUDA isn't visible (a
    # transient NVML/driver hiccup, or a stale CUDA_VISIBLE_DEVICES like "0,1,2,3"
    # leaking onto a 1-GPU node). Fail fast with an actionable message instead.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not visible to this gate process, so the Qwen3-VL gate cannot run. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}. "
            "This is usually a stale CUDA_VISIBLE_DEVICES (e.g. '0,1,2,3' on a 1-GPU node) "
            "or a transient driver/NVML error. Ensure the job pins CUDA_VISIBLE_DEVICES to "
            "its allocated GPU(s) and resubmit."
        )

    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto", "local_files_only": True}
    if flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    return model, processor, process_vision_info


def ask(model, processor, process_vision_info, question: str, image_path: Path,
        max_new_tokens: int) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": question},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    return processor.batch_decode(trimmed, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="Final gate_decisions.json path.")
    p.add_argument("--gate", choices=("qwen", "oracle"), default="qwen",
                   help="qwen: run Qwen3-VL; oracle: derive presence from GT (no model).")
    p.add_argument("--qwen_model", default="Qwen/Qwen3-VL-32B-Instruct",
                   help="Qwen3-VL Instruct HF id or local snapshot path (must be cached for "
                        "offline GPU). Pass Qwen/Qwen3-VL-8B-Instruct for a lighter, faster gate. "
                        "Thinking variants are unsupported (see module docstring).")
    p.add_argument("--saco_root", type=Path, default=iseg.SACO_ROOT_DEFAULT)
    p.add_argument("--gt_folder", type=Path, default=None)
    p.add_argument("--subsets", nargs="+", default=None,
                   choices=sorted(SACO_GOLD_SUBSETS.keys()))
    p.add_argument("--limit", type=int, default=None, help="Cap queries per subset (seeded).")
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--max_instances", type=int, default=None,
                   help="Gate only pairs with at most N GT instances (max over a/b/c). Default: all.")
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--no_flash_attention", action="store_true")
    p.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=False,
                   help="Save each gated image, captioned with NP + verdict + GT presence, "
                        "into <images_dir>/<subset>/{yes,no}/ for manual accuracy checks.")
    p.add_argument("--images_dir", type=Path, default=None,
                   help="Where to save --save_images panels (default: <out parent>/gate_images).")
    p.add_argument("--default_on_unparsed", action=argparse.BooleanOptionalAction, default=True,
                   help="When a reply can't be parsed, treat as positive (don't suppress).")
    # array-style sharding
    p.add_argument("--num_shards", type=int, default=None)
    p.add_argument("--shard_index", type=int, default=None)
    p.add_argument("--local_rank", type=int, default=None)
    p.add_argument("--shard_aggregate_timeout", type=int, default=7200)
    p.add_argument("--merge_only", action="store_true",
                   help="Skip inference: just merge existing gate_shards/ into --out.")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    shard = iseg.refseg.setup_sharding(args)
    gt_folder = args.gt_folder or (args.saco_root / "gt-annotations")
    subsets = args.subsets or list(SACO_GOLD_SUBSETS.keys())
    shard_dir = args.out.parent / "gate_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    def merge_and_write() -> None:
        if shard.world_size > 1 and not shard.distributed:
            iseg.refseg.wait_for_shard_metrics(
                [shard_dir / f"rank_{r:05d}_of_{shard.world_size:05d}.json"
                 for r in range(shard.world_size)],
                args.shard_aggregate_timeout)
        merged: dict[str, dict] = defaultdict(dict)
        for r in range(shard.world_size):
            data = json.loads((shard_dir / f"rank_{r:05d}_of_{shard.world_size:05d}.json").read_text())
            for subset, table in data.get("decisions", {}).items():
                merged[subset].update(table)
        args.out.write_text(json.dumps(merged, indent=2))
        npos = sum(v for t in merged.values() for v in t.values())
        ntot = sum(len(t) for t in merged.values())
        print(f"Wrote {args.out}: {ntot} decisions across {len(merged)} subset(s), "
              f"{npos} positive ({100*npos/max(ntot,1):.1f}%).")

    if args.merge_only:
        if shard.rank == 0:
            merge_and_write()
        iseg.refseg.cleanup_distributed(shard)
        return

    # Build the global deterministic sample list. Populate GT presence (oracle build
    # mode) when the verdict needs it (--gate oracle) or for the captioned save.
    build_mode = "oracle" if (args.gate == "oracle" or args.save_images) else "none"
    all_samples = []
    for subset in subsets:
        all_samples.extend(
            iseg.build_subset_samples(subset, gt_folder, args.saco_root,
                                      build_mode, args.limit, args.sample_seed, args.max_instances))
    samples = list(all_samples[shard.rank :: shard.world_size])
    images_dir = args.images_dir or (args.out.parent / "gate_images")

    if args.dry_run:
        print(f"Subsets {subsets}: {len(all_samples)} samples; shard "
              f"{shard.rank}/{shard.world_size} -> {len(samples)}. Gate mode: {args.gate}.")
        for s in samples[:5]:
            print(f"  [{s.subset}] id={s.image_id} np={s.np_text!r} img={s.image_path.name}")
        iseg.refseg.cleanup_distributed(shard)
        return

    def maybe_save(s, verdict):
        if args.save_images:
            out_png = images_dir / s.subset / ("yes" if verdict else "no") / f"{s.image_id}.png"
            save_gate_panel(s.image_path, s.np_text, verdict, s.gt_present, args.gate, out_png)

    decisions: dict[str, dict] = defaultdict(dict)
    if args.gate == "oracle":
        for s in samples:
            verdict = bool(s.gt_present)
            decisions[s.subset][str(s.image_id)] = verdict
            maybe_save(s, verdict)
    else:
        model, processor, pvi = load_qwen(args.qwen_model, flash_attention=not args.no_flash_attention)
        n_unparsed = 0
        for s in tqdm(samples, desc=f"Gate rank {shard.rank}/{shard.world_size}"):
            reply = ask(model, processor, pvi, GATE_QUESTION.format(np=s.np_text),
                        s.image_path, args.max_new_tokens)
            verdict = parse_yes_no(reply)
            if verdict is None:
                n_unparsed += 1
                verdict = args.default_on_unparsed
            decisions[s.subset][str(s.image_id)] = bool(verdict)
            maybe_save(s, bool(verdict))
        if n_unparsed:
            print(f"  {n_unparsed} replies unparsed -> defaulted to "
                  f"{'positive' if args.default_on_unparsed else 'negative'}.")
    if args.save_images:
        print(f"Saved captioned gate images under {images_dir}")

    shard_path = shard_dir / f"rank_{shard.rank:05d}_of_{shard.world_size:05d}.json"
    shard_path.write_text(json.dumps({
        "rank": shard.rank, "world_size": shard.world_size, "gate": args.gate,
        "qwen_model": args.qwen_model if args.gate == "qwen" else None,
        "decisions": {k: v for k, v in decisions.items()},
    }))

    iseg.refseg.barrier(shard)
    if shard.rank == 0:
        merge_and_write()
    iseg.refseg.cleanup_distributed(shard)


if __name__ == "__main__":
    main()
