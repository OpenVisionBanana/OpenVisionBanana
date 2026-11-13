#!/usr/bin/env python3
"""Legend pruning for grouped semantic segmentation (COCO-Object-80).

A VLM sees each image together with the dataset's class list and names the
classes it thinks are present. ``grouped_semseg.py --legend_pruning_json`` then
partitions and prompts only those classes, so an absent class can no longer be
painted onto a similar-looking object (bear -> dog, truck -> car). Seeds, colour
draws and decoding are otherwise unchanged; an image with no kept class is
predicted as all background without running FLUX.

The answer depends only on the image, not on the checkpoint under test, so it is
computed once per VLM and cached. The caches behind the reported scores ship in
``evaluation/assets/coco_object_80_val_legend_pruning_<vlm>.json``:

    {"dataset": "coco-object-80", "split": "val", "vlm": "<name>",
     "vlm_model": "<HF id>", "question": "<prompt template>", "max_new_tokens": 384,
     "limit": null, "classes": [...], "summary": {recall, precision, ...},
     "images": {"<output_stem>": {"present": [<class id>, ...], "reply": "<raw VLM reply>"}}}

``scripts/evaluate_semseg.py --mode legend-pruning --vlm-model NAME`` uses the
bundled cache, or creates a missing one with this script first. To run it
directly, start one process per GPU and merge afterwards:

    python evaluation/semseg/legend_pruning.py --dataset coco-object-80 --data_root DATA \\
        --out OUT.json --vlm gemma4_31b --num_shards 4 --shard_index 0   # ... 1, 2, 3
    python evaluation/semseg/legend_pruning.py --merge --out OUT.json --num_shards 4

Gemma 4 needs transformers>=5 while requirements.txt pins 4.x: install
requirements/reasonseg-rewrite-gemma.txt into a separate directory and pass it as
``--vlm_sys_path`` (scripts/evaluate_semseg.py: ``--vlm-pythonpath``). Only this
VLM pass sees it; FLUX keeps the pinned version. Batched greedy decoding is not
bit-reproducible, so a regenerated cache can differ slightly from the bundled one.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parents[1] / "evaluation" / "assets"

# --vlm-model short names -> HF ids. Any other value is used as an HF id or a
# local snapshot directory, named after its repository.
VLMS = {
    "qwen3vl8binstruct": "Qwen/Qwen3-VL-8B-Instruct",
    "gemma4_31b": "google/gemma-4-31B-it",
}
QUESTION = (
    "Here is a list of {n} semantic classes: {classes}.\n"
    "Which of these classes appear in the image? Include a class if any part of it is "
    "visible, even when it is small, distant, or partially occluded. Answer with only a "
    "JSON list of the class names that appear, spelled exactly as in the list, e.g. "
    '["class a", "class b"]. Answer [] if none of them appear.'
)
ALIASES = {
    "motorbike": "motorcycle", "bike": "bicycle", "sofa": "couch", "television": "tv",
    "tv monitor": "tv", "monitor": "tv", "cellphone": "cell phone", "mobile phone": "cell phone",
    "phone": "cell phone", "table": "dining table", "plant": "potted plant",
    "hair dryer": "hair drier", "hairdryer": "hair drier", "people": "person",
    "pedestrian": "person", "aeroplane": "airplane", "doughnut": "donut", "fridge": "refrigerator",
}


def resolve_vlm(value: str) -> tuple[str, str]:
    """Return (name, model) for a short name, HF id or local snapshot path."""
    if value in VLMS:
        return value, VLMS[value]
    match = re.search(r"models--[^/]+--([^/]+)", value)
    repository = match.group(1) if match else Path(value.rstrip("/")).name
    return re.sub(r"[^a-z0-9]", "", repository.lower()), value


def cache_path(directory: Path, dataset: str, split: str, name: str, limit: int | None) -> Path:
    """Expected cache location; a --limit cache never takes the full-set name."""
    suffix = f"_n{limit}" if limit else ""
    return directory / f"{dataset.replace('-', '_')}_{split}_legend_pruning_{name}{suffix}.json"


def normalize(text: str) -> str:
    text = re.sub(r"[_\-]", " ", text.lower())
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text).split())


def parse_reply(reply: str, by_name: dict[str, int]) -> tuple[set[int], list[str], bool]:
    """Return (kept class ids, unmatched items, parsed). Unparsed = no list found."""
    match = re.search(r"\[.*\]", reply, re.S)
    items = None
    if match:
        try:
            items = json.loads(match.group(0))
        except json.JSONDecodeError:
            items = None
    if not isinstance(items, list):
        items = re.split(r"[,\n;]", match.group(0)[1:-1] if match else reply)
    kept, unmatched = set(), []
    for item in items:
        key = normalize(str(item))
        if not key:
            continue
        for candidate in (key, ALIASES.get(key), key[:-1] if key.endswith("s") else None,
                          ALIASES.get(key[:-1]) if key.endswith("s") else None):
            if candidate in by_name:
                kept.add(by_name[candidate])
                break
        else:
            unmatched.append(key)
    return kept, unmatched, match is not None or bool(kept)


def shard_path(out: Path, num_shards: int, shard_index: int) -> Path:
    return out.with_name(out.name + ".shards") / f"rank_{shard_index:05d}_of_{num_shards:05d}.json"


def run_shard(a: argparse.Namespace) -> None:
    if a.vlm_sys_path:
        # Ahead of the venv, for this process only (Gemma 4 needs transformers>=5).
        sys.path.insert(0, a.vlm_sys_path)
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parents[1]))
    import numpy as np
    import torch
    from tqdm import tqdm
    import transformers
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    import inference_partial_flux2_klein_img2img as base

    layout = base.DATASETS[a.dataset]
    semantic = layout.semantic
    samples = base.build_samples(
        layout=layout, data_root=Path(a.data_root), split=a.split, condition=None, image_root=None,
        gt_root=None, image_suffixes=layout.image_suffixes, image_glob=None, limit=a.limit,
    )[a.shard_index::a.num_shards]
    candidates = [c for c in range(len(semantic.classes)) if c != semantic.implicit_background_id]
    cmap = (base.load_coco_category_map(Path(a.data_root), semantic.coco_mask_type, a.split)
            if semantic.coco_mask_type is not None else None)
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA not visible (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
    try:
        config = AutoConfig.from_pretrained(a.vlm_model, local_files_only=a.local_files_only)
    except (KeyError, ValueError) as error:
        raise SystemExit(
            f"transformers {transformers.__version__} cannot load {a.vlm_model} ({error}). Gemma 4 needs "
            "transformers>=5: install requirements/reasonseg-rewrite-gemma.txt into a separate directory "
            "and pass it with --vlm-pythonpath.") from error
    # Qwen3-VL runs with flash attention, Gemma 4 with the default sdpa.
    attention = ({"attn_implementation": "sdpa" if a.no_flash_attention else "flash_attention_2"}
                 if "qwen" in config.model_type else {})
    print(f"Legend pruning shard {a.shard_index}/{a.num_shards}: {a.vlm_model} "
          f"(transformers {transformers.__version__}), {len(samples)} images", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        a.vlm_model, dtype=torch.bfloat16, device_map="cuda", local_files_only=a.local_files_only, **attention,
    ).eval()
    processor = AutoProcessor.from_pretrained(a.vlm_model, local_files_only=a.local_files_only)
    processor.tokenizer.padding_side = "left"
    question = QUESTION.format(n=len(candidates), classes=", ".join(semantic.classes[c] for c in candidates))
    by_name = {normalize(semantic.classes[c]): c for c in candidates}

    records = {}
    for start in tqdm(range(0, len(samples), a.batch_size), desc=f"Legend pruning {a.shard_index}/{a.num_shards}"):
        batch = samples[start:start + a.batch_size]
        conversations = [[{"role": "user", "content": [
            {"type": "image", "image": str(s.image_path)}, {"type": "text", "text": question}]}]
            for s in batch]
        inputs = processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", padding=True,
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
        replies = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for s, reply in zip(batch, replies):
            kept, unmatched, parsed = parse_reply(reply, by_name)
            gt_ids = set(np.unique(base.load_gt_mask(s.gt_path, semantic, cmap)).tolist())
            # An unreadable reply prunes nothing rather than everything.
            records[s.output_stem] = {
                "present": sorted(kept) if parsed else list(candidates), "reply": reply,
                "unmatched": unmatched, "fallback": not parsed, "gt": [c for c in candidates if c in gt_ids],
            }
    path = shard_path(a.out, a.num_shards, a.shard_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": a.dataset, "split": a.split, "vlm": a.vlm, "vlm_model": a.vlm_model,
               "question": QUESTION, "max_new_tokens": a.max_new_tokens, "limit": a.limit,
               "classes": list(semantic.classes), "images": records}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(path)


def summarize(records: dict, classes: list[str]) -> dict:
    """Filter quality against the GT-present classes of each image."""
    tp = fp = fn = complete = 0
    errors = Counter()
    for record in records.values():
        pred, gt = set(record["present"]), set(record["gt"])
        tp += len(pred & gt); fp += len(pred - gt); fn += len(gt - pred)
        complete += gt <= pred
        errors.update(f"missed {classes[c]}" for c in gt - pred)
        errors.update(f"extra {classes[c]}" for c in pred - gt)
    n = max(len(records), 1)
    return {
        "num_images": len(records),
        "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
        "images_with_every_gt_class_kept": complete / n,
        "mean_kept": sum(len(r["present"]) for r in records.values()) / n,
        "mean_gt_present": sum(len(r["gt"]) for r in records.values()) / n,
        "fallback_unparsed": sum(r["fallback"] for r in records.values()),
        "unmatched_names": Counter(u for r in records.values() for u in r["unmatched"]).most_common(30),
        "top_errors": errors.most_common(30),
    }


def write_cache(path: Path, meta: dict, images: dict) -> None:
    """One image per line, so the bundled caches diff readably."""
    lines = [json.dumps(stem) + ": " + json.dumps(record) for stem, record in images.items()]
    text = json.dumps(meta, indent=1)[:-2] + ',\n "images": {\n' + ",\n".join(lines) + "\n }\n}\n"
    json.loads(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def merge_shards(out: Path, num_shards: int) -> dict:
    """Combine the shard files into ``out`` and return its summary."""
    shards = [json.loads(shard_path(out, num_shards, r).read_text()) for r in range(num_shards)]
    meta = {key: value for key, value in shards[0].items() if key != "images"}
    records = {}
    for shard in shards:
        if {key: value for key, value in shard.items() if key != "images"} != meta:
            raise ValueError(f"Legend pruning shards for {out} disagree on dataset/model/settings")
        records.update(shard["images"])
    summary = summarize(records, meta["classes"])
    images = {stem: {"present": r["present"], "reply": r["reply"]} for stem, r in sorted(records.items())}
    write_cache(out, {**meta, "summary": summary}, images)
    print(f"Legend pruning cache {out}: " + ", ".join(
        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items()
        if k not in ("top_errors", "unmatched_names")), flush=True)
    print("  most frequent errors: " + ", ".join(f"{k} x{v}" for k, v in summary["top_errors"][:12]), flush=True)
    return summary


def load_kept(paths: list[Path], classes: list[str], stems: list[str]) -> dict[str, set[int]]:
    """Per-image kept classes; several caches are united (a class survives if any VLM keeps it)."""
    kept: dict[str, set[int]] = {stem: set() for stem in stems}
    for path in paths:
        cache = json.loads(Path(path).read_text())
        if cache.get("classes") != classes:
            raise ValueError(f"Legend pruning cache {path} was built for a different class list")
        missing = [stem for stem in stems if stem not in cache["images"]]
        if missing:
            raise ValueError(f"Legend pruning cache {path} lacks {len(missing)} evaluated images, e.g. {missing[:3]}")
        for stem in stems:
            kept[stem].update(cache["images"][stem]["present"])
    return kept


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="Cache JSON to write.")
    p.add_argument("--merge", action="store_true", help="Merge the --num_shards shard files into --out and exit.")
    p.add_argument("--dataset", default="coco-object-80")
    p.add_argument("--data_root")
    p.add_argument("--split", default="val")
    p.add_argument("--vlm", default="qwen3vl8binstruct",
                   help=f"Short name ({', '.join(VLMS)}), HF id or local snapshot.")
    p.add_argument("--vlm_sys_path", help="Directory put first on sys.path for this pass (transformers>=5 for Gemma 4).")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=384, help="Reply budget; a list cut off here is still parsed.")
    p.add_argument("--no_flash_attention", action="store_true")
    p.add_argument("--allow_download", dest="local_files_only", action="store_false",
                   help="Let transformers download the VLM instead of reading only the local HF cache.")
    p.add_argument("--limit", type=int)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.vlm, a.vlm_model = resolve_vlm(a.vlm)
    return a


def main(argv=None) -> None:
    a = parse_args(argv)
    if a.merge:
        merge_shards(a.out, a.num_shards)
        return
    if a.data_root is None:
        raise SystemExit("--data_root is required for the VLM pass")
    run_shard(a)
    if a.num_shards == 1:
        merge_shards(a.out, 1)


if __name__ == "__main__":
    main()
