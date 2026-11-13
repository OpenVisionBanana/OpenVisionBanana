#!/usr/bin/env python3
"""Preview prepared COCONut-XL semantic- and instance-segmentation samples.

COCONut-XL is already materialized as a coco directory. This tool reads its
final merged manifest, copies only a few selected RGB/panoptic pairs into a
small coco source tree, runs the production converters, and writes HTML
galleries containing the conditioning image, prompt, and prepared target.

Example:

    python scripts/dataset_prep/preview_coconut_xlarge.py \
        --coconut-xlarge-root datasets_archives/shared_seg/coconut_xlarge/coco \
        --out-root /scratch/coconut_xlarge_preview --num-samples 4

Use --sample-ids with image IDs from selected_samples.json to reproduce a
preview. The large manifest is read, but only selected image/mask files are
copied below the output directory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import html
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.semseg import taxonomy  # noqa: E402


@dataclass(frozen=True)
class Sample:
    image_id: str
    stem: str
    image_file: str
    panoptic_file: str


@dataclass
class SourceIndex:
    root: Path
    categories: list[dict]
    images: dict[str, dict]
    annotations: dict[str, dict]
    candidate_ids: list[str]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def discover_samples(root: Path) -> SourceIndex:
    """Load IDs with a non-crowd thing instance, suitable for both previews."""
    resolved = taxonomy.resolve_coconut_xlarge_root(root)
    manifest_path = taxonomy.coconut_xlarge_annotation_path(resolved)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    categories = list(data.get("categories", ()))
    thing_ids = {
        int(category["id"])
        for category in categories
        if int(category.get("isthing", 0)) == 1
    }
    images = {str(image.get("id")): image for image in data.get("images", ())}
    annotations = {
        str(annotation.get("image_id")): annotation
        for annotation in data.get("annotations", ())
    }
    candidate_ids = sorted(
        image_id for image_id, annotation in annotations.items()
        if (image_id in images and images[image_id].get("file_name")
            and annotation.get("file_name")
            and any(int(segment.get("category_id", -1)) in thing_ids
                    and int(segment.get("iscrowd", 0)) == 0
                    for segment in annotation.get("segments_info", ())))
    )
    if not candidate_ids:
        raise ValueError(f"No non-crowd thing-annotated COCONut-XL images in {manifest_path}")
    return SourceIndex(resolved, categories, images, annotations, candidate_ids)


def sample_for_id(index: SourceIndex, image_id: str) -> Sample:
    image = index.images[image_id]
    annotation = index.annotations[image_id]
    image_file = str(image.get("file_name") or "")
    panoptic_file = str(annotation.get("file_name") or "")
    if not image_file or not panoptic_file:
        raise ValueError(f"COCONut-XL image {image_id} has no paired RGB/panoptic filename")
    return Sample(
        image_id=image_id,
        stem=str(image.get("object365_name") or Path(image_file).stem),
        image_file=image_file,
        panoptic_file=panoptic_file,
    )


def select_samples(index: SourceIndex, sample_ids: str | None, *, num_samples: int,
                   seed: int) -> list[Sample]:
    candidates = set(index.candidate_ids)
    if sample_ids is not None:
        requested = parse_csv(sample_ids)
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("--sample-ids must be non-empty and contain no duplicates")
        by_stem = {
            str(image.get("object365_name") or Path(str(image.get("file_name") or "")).stem): image_id
            for image_id, image in index.images.items()
        }
        selected: list[Sample] = []
        for value in requested:
            image_id = value if value in candidates else by_stem.get(value)
            if image_id not in candidates:
                raise ValueError(
                    f"Unknown or unsuitable COCONut-XL sample ID: {value}. "
                    "A preview sample must have a non-crowd thing instance."
                )
            selected.append(sample_for_id(index, image_id))
        return selected
    if not 1 <= num_samples <= len(index.candidate_ids):
        raise ValueError(f"--num-samples must be in [1, {len(index.candidate_ids)}]")
    return [
        sample_for_id(index, image_id)
        for image_id in sorted(random.Random(seed).sample(index.candidate_ids, num_samples))
    ]


def copy_selected_pairs(index: SourceIndex, samples: list[Sample], source_root: Path) -> None:
    """Write selected source files and a tiny valid COCONut panoptic manifest."""
    for directory in ("train2017", "panoptic_train2017", "annotations"):
        (source_root / directory).mkdir(parents=True, exist_ok=True)
    selected_images, selected_annotations = [], []
    for sample in samples:
        image_source = index.root / "train2017" / sample.image_file
        panoptic_source = index.root / "panoptic_train2017" / sample.panoptic_file
        if not image_source.is_file() or not panoptic_source.is_file():
            raise FileNotFoundError(f"Missing RGB or panoptic mask for COCONut-XL image {sample.image_id}")
        shutil.copy2(image_source, source_root / "train2017" / sample.image_file)
        shutil.copy2(panoptic_source, source_root / "panoptic_train2017" / sample.panoptic_file)
        selected_images.append(index.images[sample.image_id])
        selected_annotations.append(index.annotations[sample.image_id])
    (source_root / "annotations" / "panoptic_train2017.json").write_text(
        json.dumps({
            "categories": index.categories,
            "images": selected_images,
            "annotations": selected_annotations,
        }) + "\n",
        encoding="utf-8",
    )


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def build_prepared_previews(source_root: Path, out_root: Path, args: argparse.Namespace) -> None:
    shared = ["--coconut-xlarge-root", str(source_root), "--seed", str(args.seed),
              "--num-workers", "1"]
    run([
        sys.executable, str(ROOT / "dataset_prep/semseg/dataset_prep.py"),
        "--out-root", str(out_root / "semseg"), *shared,
        "--variants-per-image", str(args.semseg_variants_per_image),
    ])
    run([
        sys.executable, str(ROOT / "dataset_prep/instanceseg/dataset_prep.py"),
        "--out-root", str(out_root / "instanceseg"), *shared,
        "--variants-per-image", str(args.instanceseg_variants_per_image),
        "--negative-ratio", str(args.instanceseg_negative_ratio),
        "--draw-mode", args.draw_mode,
    ])


def write_gallery(task_root: Path, title: str) -> int:
    rows = [
        json.loads(line)
        for line in (task_root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cards = []
    for row in rows:
        metadata = json.dumps(row.get("metadata", {}), indent=2, sort_keys=True)
        cards.append(
            "<article>"
            f"<h2>{html.escape(str(row.get('id', 'unknown')))}</h2>"
            f"<p><strong>split:</strong> {html.escape(str(row.get('split', 'unknown')))}</p>"
            "<div class=images>"
            f"<figure><img src={html.escape(str(row['conditioning_image']), quote=True)}><figcaption>conditioning image</figcaption></figure>"
            f"<figure><img src={html.escape(str(row['image']), quote=True)}><figcaption>prepared target</figcaption></figure>"
            "</div>"
            f"<h3>Prompt</h3><pre>{html.escape(str(row['text']))}</pre>"
            f"<details><summary>row metadata</summary><pre>{html.escape(metadata)}</pre></details>"
            "</article>"
        )
    page = f"""<!doctype html>
<html><head><meta charset=utf-8><title>{html.escape(title)}</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; background: #161616; color: #eee; }}
article {{ margin: 1.5rem 0; padding: 1rem; background: #242424; border-radius: .5rem; }}
.images {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
figure {{ margin: 0; max-width: 48%; }} img {{ max-width: 100%; height: auto; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; }} summary {{ cursor: pointer; }}
</style></head><body><h1>{html.escape(title)}</h1><p>{len(rows)} prepared row(s).</p>
{''.join(cards)}</body></html>"""
    (task_root / "index.html").write_text(page, encoding="utf-8")
    return len(rows)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coconut-xlarge-root", type=Path, required=True,
                        help="Materialized COCONut-XL coco root.")
    parser.add_argument("--out-root", type=Path, required=True,
                        help="New output directory for the mini source tree and both galleries.")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of pairs to sample when --sample-ids is unset (default: 4).")
    parser.add_argument("--sample-ids", default=None,
                        help="Optional comma-separated image IDs from selected_samples.json; unique filename stems also work.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--semseg-variants-per-image", type=int, default=1)
    parser.add_argument("--instanceseg-variants-per-image", type=int, default=1)
    parser.add_argument("--instanceseg-negative-ratio", type=float, default=0.0,
                        help="Fraction of all-black absent-class previews (default: 0, preserving positive samples).")
    parser.add_argument("--draw-mode", default="fixeddraw_32_lr",
                        help="Instance target/prompt draw mode (default: fixeddraw_32_lr).")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    out_root = args.out_root.expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"Preview output directory must be empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    print("Reading COCONut-XL final merged manifest...")
    index = discover_samples(args.coconut_xlarge_root)
    selected = select_samples(index, args.sample_ids, num_samples=args.num_samples, seed=args.seed)
    print("Selected COCONut-XL samples:")
    for sample in selected:
        print(f"  {sample.image_id}: {sample.stem}")
    source_root = out_root / "source" / "coco"
    print("Copying selected RGB/panoptic pairs into a miniature COCONut tree...")
    copy_selected_pairs(index, selected, source_root)
    build_prepared_previews(source_root, out_root, args)
    semantic_rows = write_gallery(out_root / "semseg", "COCONut-XL semantic-segmentation preview")
    instance_rows = write_gallery(out_root / "instanceseg", "COCONut-XL instance-segmentation preview")
    (out_root / "selected_samples.json").write_text(
        json.dumps({"samples": [asdict(sample) for sample in selected]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Preview complete.")
    print(f"  semantic rows: {semantic_rows} -> {out_root / 'semseg/index.html'}")
    print(f"  instance rows: {instance_rows} -> {out_root / 'instanceseg/index.html'}")


if __name__ == "__main__":
    main()
