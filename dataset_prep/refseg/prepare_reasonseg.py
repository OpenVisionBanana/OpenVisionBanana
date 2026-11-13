#!/usr/bin/env python3
"""Prepare archive-native ReasonSeg train/test annotations as canonical RefSeg."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from dataset_prep.refseg.prepare_steervit_refseg import (  # noqa: E402
    DEFAULT_COLOR_SPECS, TASK_NAME, build_prompt, color_binary_mask,
    parse_color_list, safe_name, save_rgb_png,
)

DEFAULT_ROOT = REPO_ROOT / "datasets_archives" / "refseg" / "reasonseg_train_test"
DEFAULT_VAL = REPO_ROOT / "datasets_archives" / "refseg" / "reasonseg_val"


@dataclass(frozen=True)
class Sample:
    split: str
    member: str
    image_name: str
    image_member: str
    extracted_image: Path | None
    texts: tuple[str, ...]
    polygons: tuple[tuple[tuple[float, float], ...], ...]


def selected_splits(raw: str) -> list[str]:
    values = [v.strip().lower() for v in raw.split(",") if v.strip()]
    if not values or any(v not in {"train", "test"} for v in values):
        raise ValueError("--splits must be a comma-separated subset of train,test")
    return list(dict.fromkeys(values))


def target_polygons(annotation: dict) -> tuple[tuple[tuple[float, float], ...], ...]:
    result = []
    for shape in annotation.get("shapes") or []:
        labels = set(shape.get("labels") or [shape.get("label")])
        points = shape.get("points") or []
        if "target" not in labels or len(points) < 3:
            continue
        result.append(tuple((float(x), float(y)) for x, y in points))
    return tuple(result)


def archive_samples(archive: Path, split: str) -> list[Sample]:
    samples: list[Sample] = []
    with tarfile.open(archive) as tar:
        members = {member.name: member for member in tar.getmembers() if member.isfile()}
        images = {Path(name).stem: name for name in members
                  if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}}
        for name, member in sorted(members.items()):
            if not name.endswith(".json"):
                continue
            annotation = json.load(tar.extractfile(member))
            polygons = target_polygons(annotation)
            image_name = next((str(s.get("image_name")) for s in annotation.get("shapes") or []
                               if s.get("image_name")), "")
            image_member = images.get(Path(image_name).stem, "")
            texts = tuple(dict.fromkeys(str(t).strip() for t in annotation.get("text") or [] if str(t).strip()))
            if image_member:
                samples.append(Sample(split, name, image_name, image_member, None, texts, polygons))
    return samples


def directory_samples(directory: Path, split: str) -> list[Sample]:
    samples: list[Sample] = []
    for annotation_path in sorted(directory.glob("*.json")):
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        polygons = target_polygons(annotation)
        image_name = next((str(s.get("image_name")) for s in annotation.get("shapes") or []
                           if s.get("image_name")), "")
        image_path = directory / image_name
        texts = tuple(dict.fromkeys(str(t).strip() for t in annotation.get("text") or [] if str(t).strip()))
        if image_path.is_file():
            samples.append(Sample(split, annotation_path.name, image_name, "", image_path, texts, polygons))
    return samples


def val_image_stems(directory: Path) -> set[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"ReasonSeg validation directory not found: {directory}")
    return {path.stem for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}}


def rasterize(polygons, width: int, height: int) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        draw.polygon(polygon, fill=255)
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reasonseg-root", type=Path, default=DEFAULT_ROOT,
                        help="Directory containing train.tar/test.tar or extracted train/test directories.")
    parser.add_argument("--extract-root", type=Path, default=None,
                        help="Extract selected tar split(s) here first, then prepare from the extracted files.")
    parser.add_argument("--keep-extracted", action="store_true",
                        help="Retain --extract-root after successful preparation (default removes it).")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--splits", default="train", help="Comma-separated subset of train,test.")
    parser.add_argument("--all-expressions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expressions-per-image", type=int, default=1,
                        help="Used only with --no-all-expressions; deterministic count per image.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--colors", default=";".join(DEFAULT_COLOR_SPECS))
    parser.add_argument("--no-random-colors", action="store_true")
    parser.add_argument("--reasonseg-val-dir", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--exclude-val-overlap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.expressions_per_image < 1:
        raise ValueError("--expressions-per-image must be positive")
    root, out = args.reasonseg_root.expanduser().resolve(), args.out_root.expanduser().resolve()
    source_root = root
    staging_root: Path | None = None
    splits = selected_splits(args.splits)
    if args.extract_root is not None:
        extract_root = args.extract_root.expanduser().resolve()
        if extract_root == source_root:
            raise ValueError("--extract-root must differ from --reasonseg-root")
        staging_root = extract_root
        extract_root.mkdir(parents=True, exist_ok=True)
        for split in splits:
            archive = root / f"{split}.tar"
            if not archive.is_file():
                raise FileNotFoundError(f"ReasonSeg archive not found: {archive}")
            split_dir = extract_root / split
            if not split_dir.is_dir():
                print(f"Extracting {archive} -> {extract_root}")
                with tarfile.open(archive) as tar:
                    tar.extractall(extract_root)
        root = extract_root
    colors = parse_color_list(args.colors)
    val_stems = val_image_stems(args.reasonseg_val_dir.expanduser().resolve()) if args.exclude_val_overlap else set()
    samples, rejected = [], []
    for split in splits:
        split_dir = root / split
        archive = root / f"{split}.tar"
        if split_dir.is_dir():
            split_samples = directory_samples(split_dir, split)
        elif archive.is_file():
            split_samples = archive_samples(archive, split)
        else:
            raise FileNotFoundError(f"ReasonSeg split not found as directory or tar: {split_dir} / {archive}")
        for sample in split_samples:
            if Path(sample.image_name).stem in val_stems:
                rejected.append({"split": split, "member": sample.member, "reason": "reasonseg_val_image_overlap"})
            elif not sample.polygons:
                rejected.append({"split": split, "member": sample.member, "reason": "missing_target_polygon"})
            elif not sample.texts:
                rejected.append({"split": split, "member": sample.member, "reason": "missing_expression"})
            else:
                samples.append(sample)
    manifest = {"reasonseg_root": str(root), "splits": splits, "all_expressions": args.all_expressions,
                "expressions_per_image": args.expressions_per_image, "exclude_val_overlap": args.exclude_val_overlap,
                "reasonseg_val_dir": str(args.reasonseg_val_dir), "selected_images": len(samples),
                "rejection_counts": dict(Counter(row["reason"] for row in rejected))}
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True)); return
    out.mkdir(parents=True, exist_ok=True)
    cond_dir, target_dir = out / "conditioning_images", out / "target_images"
    cond_dir.mkdir(exist_ok=True); target_dir.mkdir(exist_ok=True)
    rows = []
    # Keep each archive open; reopening a TAR per image is quadratic in archive scans.
    for split in splits:
        split_samples = [item for item in samples if item.split == split]
        archive = root / f"{split}.tar"
        tar = tarfile.open(archive) if archive.is_file() else None
        try:
            for sample in split_samples:
                if sample.extracted_image is not None:
                    with Image.open(sample.extracted_image) as source:
                        image = source.convert("RGB")
                        image.load()
                else:
                    assert tar is not None
                    with tar.extractfile(sample.image_member) as stream:
                        image = Image.open(stream).convert("RGB")
                        image.load()
                mask = rasterize(sample.polygons, image.width, image.height)
                if not mask.getbbox():
                    rejected.append({"split": sample.split, "member": sample.member, "reason": "empty_target_polygon"})
                    continue
                input_name = safe_name("reasonseg", sample.split, Path(sample.image_name).stem) + "_input.png"
                save_rgb_png(image, cond_dir / input_name)
                texts = sample.texts if args.all_expressions else random.Random(f"{args.seed}:{sample.member}").sample(sample.texts, min(args.expressions_per_image, len(sample.texts)))
                for index, expression in enumerate(texts):
                    rng = random.Random(f"{args.seed}:{sample.split}:{sample.member}:{index}")
                    color = colors[0] if args.no_random_colors else colors[rng.randrange(len(colors))]
                    stem = safe_name("reasonseg", sample.split, Path(sample.image_name).stem, index)
                    target_name = f"{stem}_mask.png"
                    color_binary_mask(mask, color.rgb).save(target_dir / target_name)
                    rows.append({"conditioning_image": f"conditioning_images/{input_name}", "image": f"target_images/{target_name}",
                                 "text": build_prompt(expression, color, rng), "task": TASK_NAME, "source_dataset": "reasonseg",
                                 "preprocess_target": "nearest", "id": stem, "split": sample.split,
                                 "metadata": {"reasonseg_member": sample.member, "source_image": sample.image_name,
                                              "expression_index": index, "expression": expression,
                                              "foreground_color": {"name": color.name, "rgb": list(color.rgb)},
                                              "background_color": {"name": "black", "rgb": [0, 0, 0]}}})
        finally:
            if tar is not None:
                tar.close()
    if not rows:
        raise RuntimeError("ReasonSeg preparation wrote zero rows")
    manifest["written_rows"] = len(rows); manifest["rejection_counts"] = dict(Counter(row["reason"] for row in rejected))
    (out / "metadata.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    (out / "selection_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "rejected_rows.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rejected), encoding="utf-8")
    if staging_root is not None and not args.keep_extracted:
        shutil.rmtree(staging_root)
        print(f"Removed transient ReasonSeg extraction: {staging_root}")
    print(f"ReasonSeg preparation complete: {len(rows):,} rows -> {out}")


if __name__ == "__main__":
    main()
