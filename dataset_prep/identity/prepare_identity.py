#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from pathlib import Path

from PIL import Image, ImageOps

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda iterable, **_: iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def safe_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._")
    return value or "image"


def iter_images(roots: list[Path]) -> list[Path]:
    images = []
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                images.append(path)
    return sorted(images)


def maybe_resize(image: Image.Image, max_side: int | None) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max_side is None:
        return image
    width, height = image.size
    largest = max(width, height)
    if largest <= max_side:
        return image
    scale = max_side / largest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def materialize(src: Path, dst: Path, copy_mode: str, max_side: int | None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        resized = maybe_resize(image, max_side)
        needs_reencode = (
            resized.size != ImageOps.exif_transpose(image).size
            or dst.suffix.lower() != src.suffix.lower()
            or src.suffix.lower() not in IMAGE_SUFFIXES
        )
        if needs_reencode:
            resized.save(dst.with_suffix(".png"))
            return

    if copy_mode == "copy":
        shutil.copy2(src, dst)
    elif copy_mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    elif copy_mode == "hardlink":
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported copy mode: {copy_mode}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare identity/no-op reconstruction rows.")
    parser.add_argument("--image-root", action="append", type=Path, required=True)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-side", type=int, default=1024, help="Resize larger images to this side; <=0 disables.")
    parser.add_argument("--copy-mode", choices=["copy", "symlink", "hardlink"], default="copy")
    parser.add_argument(
        "--prompt",
        default="[TASK: IDENTITY] Reproduce this image exactly without making any changes.",
    )
    parser.add_argument("--source-dataset", default="identity")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.max_side is not None and args.max_side <= 0:
        args.max_side = None
    image_roots = [path.expanduser().resolve() for path in args.image_root]
    for root in image_roots:
        if not root.exists():
            raise FileNotFoundError(f"Image root does not exist: {root}")

    images = iter_images(image_roots)
    rng = random.Random(args.seed)
    rng.shuffle(images)
    if args.limit is not None:
        images = images[: args.limit]

    out_root = args.out_root.expanduser().resolve()
    (out_root / "conditioning_images").mkdir(parents=True, exist_ok=True)
    (out_root / "target_images").mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = 0
    for index, image_path in enumerate(tqdm(images, desc="Preparing identity rows")):
        stem = f"id_{index:08d}_{safe_stem(image_path.stem)}"
        suffix = ".png" if args.max_side is not None else image_path.suffix.lower()
        cond_rel = Path("conditioning_images") / f"{stem}_source{suffix}"
        target_rel = Path("target_images") / f"{stem}_target{suffix}"
        try:
            materialize(image_path, out_root / cond_rel, args.copy_mode, args.max_side)
            materialize(image_path, out_root / target_rel, args.copy_mode, args.max_side)
        except Exception:
            skipped += 1
            continue

        if (out_root / cond_rel).with_suffix(".png").exists():
            cond_rel = cond_rel.with_suffix(".png")
        if (out_root / target_rel).with_suffix(".png").exists():
            target_rel = target_rel.with_suffix(".png")

        rows.append(
            {
                "conditioning_image": cond_rel.as_posix(),
                "image": target_rel.as_posix(),
                "text": args.prompt,
                "task": "identity",
                "source_dataset": args.source_dataset,
                "id": stem,
            }
        )

    with (out_root / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print("Identity preparation complete.")
    print(f"  read: {len(images):,}")
    print(f"  written: {len(rows):,}")
    print(f"  skipped_corrupt: {skipped:,}")
    print(f"  out_root: {out_root}")


if __name__ == "__main__":
    main()
