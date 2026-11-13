#!/usr/bin/env python3
"""Materialize a PD12M WebDataset shard subset into a canonical prepared member.

The PD12M shards (datasets_archives/image_generation/raw/pd12m/<preset>/shards/
pd12m-train-*.tar) hold text-to-image rows as <key>.jpg + <key>.txt (caption) +
<key>.json (id/task/source_dataset). The trainer cannot stream WebDataset, so
this script unpacks a seeded subset into the same layout the pool converters
write:

    <out>/target_images/<key>.jpg
    <out>/metadata.jsonl        rows: {"image": ..., "text": ..., "task": "text_to_image"}

Rows carry NO conditioning_image — this is a pure text-to-image member; the
downstream tooling (convert_metadata.py,
validate_dataset_specs.py) treats task == "text_to_image" rows as control-free.

Determinism: the shard list is shuffled with --seed and consumed in that order
until --limit rows are written, so re-runs with the same inputs pick the same
subset. metadata.jsonl is written last (via a .partial rename), so an
interrupted run is never mistaken for a complete member.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import tarfile
from pathlib import Path


def iter_shard_samples(shard: Path):
    """Yield (key, jpg_bytes, caption) per complete sample in one shard."""
    groups: dict[str, dict[str, bytes]] = {}
    with tarfile.open(shard, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue
            key, dot, suffix = member.name.rpartition(".")
            if not dot or suffix not in ("jpg", "txt", "json"):
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            groups.setdefault(key, {})[suffix] = fh.read()
            parts = groups[key]
            if len(parts) == 3:
                yield key, parts["jpg"], parts["txt"].decode("utf-8").strip()
                del groups[key]


def resize_jpg(data: bytes, max_side: int) -> bytes:
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        if max(w, h) <= max_side:
            return data
        scale = max_side / max(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards-dir", type=Path, required=True,
                   help="Directory holding pd12m-train-*.tar WebDataset shards.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output member directory (target_images/ + metadata.jsonl).")
    p.add_argument("--limit", type=int, default=50000,
                   help="Rows to materialize (default: 50000).")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the shard-order shuffle (default: 42).")
    p.add_argument("--require-count", action="store_true",
                   help="Fail unless --limit complete samples are available. Used by public setup.")
    p.add_argument("--max-side", type=int, default=0,
                   help="Resize so the longer side is at most this (0 = keep original bytes).")
    args = p.parse_args()
    if args.limit < 1:
        p.error("--limit must be positive")

    shards = sorted(args.shards_dir.glob("*.tar"))
    if not shards:
        print(f"ERROR: no *.tar shards under {args.shards_dir}", file=sys.stderr)
        return 1
    random.Random(args.seed).shuffle(shards)

    out = args.out
    images_dir = out / "target_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_partial = out / "metadata.jsonl.partial"

    written = 0
    with open(metadata_partial, "w", encoding="utf-8") as meta:
        for shard in shards:
            if written >= args.limit:
                break
            for key, jpg, caption in iter_shard_samples(shard):
                if written >= args.limit:
                    break
                if not caption:
                    continue
                if args.max_side > 0:
                    jpg = resize_jpg(jpg, args.max_side)
                (images_dir / f"{key}.jpg").write_bytes(jpg)
                meta.write(json.dumps({
                    "image": f"target_images/{key}.jpg",
                    "text": caption,
                    "task": "text_to_image",
                    "source_dataset": "pd12m",
                }, ensure_ascii=False) + "\n")
                written += 1
            print(f"  {shard.name}: total {written}/{args.limit}", flush=True)

    if written == 0:
        print("ERROR: no complete jpg/txt/json samples found in shards", file=sys.stderr)
        return 1
    if args.require_count and written != args.limit:
        print(f"ERROR: requested {args.limit} PD12M samples, but only {written} complete samples "
              "are available. Preparation is incomplete.", file=sys.stderr)
        return 1
    metadata_partial.rename(out / "metadata.jsonl")
    print(f"Materialized {written} pd12m rows into {out}")
    if written < args.limit:
        print(f"WARNING: shards exhausted at {written} rows (< limit {args.limit})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
