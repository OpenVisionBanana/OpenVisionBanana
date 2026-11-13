#!/usr/bin/env python3
"""Preview prepared semantic- and instance-segmentation IDD examples.

The tool accepts the outer ``idd_1_2.tar`` directly. It streams its two nested
release archives, writes only selected RGB and polygon-annotation pairs into a
tiny temporary source tree, and invokes the production converters. The output
is a browser-ready gallery for both tasks, including the exact conditioning
image, prompt, and prepared target.

Example:

    python scripts/dataset_prep/preview_idd_tar.py \\
        --idd-tar /data/shared_seg/idd_1_2.tar \\
        --out-root /scratch/idd_preview --num-samples 4

Use ``--sample-keys`` with a key printed in ``selected_samples.json`` to select
specific frames on a later run. The raw release is large and nested, so this
may scan both inner archives; only requested pairs are written below
``--out-root/source/``.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import html
import json
from pathlib import Path, PurePosixPath
import random
import shutil
import subprocess
import sys
import tarfile
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
INNER_ARCHIVES = {
    "idd-segmentation.tar.gz": "IDD_Segmentation",
    "idd-20k-II.tar.gz": "idd20kII",
}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


@dataclass(frozen=True)
class Sample:
    part: str
    split: str
    drive: str
    stem: str
    image_member: str
    annotation_member: str

    @property
    def key(self) -> str:
        return f"{self.part}/{self.split}/{self.drive}/{self.stem}"


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_splits(value: str) -> list[str]:
    selected = []
    for split in parse_csv(value):
        normalized = SPLIT_ALIASES.get(split.lower())
        if normalized is None:
            raise ValueError("--splits must contain train, val, and/or test")
        if normalized not in selected:
            selected.append(normalized)
    if not selected:
        raise ValueError("--splits must not be empty")
    return selected


def _canonical_member_name(name: str, release_part: str) -> str:
    """Map an inner-tar member to a safe source-tree-relative path."""
    parts = [part for part in PurePosixPath(name).parts if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise ValueError(f"Unsafe IDD archive member: {name}")
    for index, part in enumerate(parts):
        if part.lower() in {"idd_segmentation", "idd20kii"}:
            return "/".join([release_part, *parts[index + 1:]])
    return "/".join([release_part, *parts])


def _inner_members(archive_path: Path) -> Iterable[tuple[str, tarfile.TarFile]]:
    """Yield each expected nested release archive as a streaming tar reader."""
    with tarfile.open(archive_path, "r:*") as outer:
        found: set[str] = set()
        for outer_member in outer:
            release_part = INNER_ARCHIVES.get(PurePosixPath(outer_member.name).name)
            if release_part is None or not outer_member.isfile():
                continue
            found.add(release_part)
            payload = outer.extractfile(outer_member)
            if payload is None:
                raise OSError(f"Could not read nested IDD archive: {outer_member.name}")
            with payload, tarfile.open(fileobj=payload, mode="r|*") as inner:
                yield release_part, inner
        missing = set(INNER_ARCHIVES.values()) - found
        if missing:
            raise FileNotFoundError(
                f"{archive_path} lacks expected nested IDD archive(s) for: "
                + ", ".join(sorted(missing))
            )


def _sample_from_annotation(annotation_member: str) -> Sample | None:
    path = PurePosixPath(annotation_member)
    parts = path.parts
    try:
        gt_index = parts.index("gtFine")
    except ValueError:
        return None
    if gt_index + 2 >= len(parts) or not path.name.endswith("_gtFine_polygons.json"):
        return None
    part = parts[0]
    split = parts[gt_index + 1]
    drive = parts[gt_index + 2]
    stem = path.name.removesuffix("_gtFine_polygons.json")
    image_member = "/".join([
        part,
        *parts[1:gt_index],
        "leftImg8bit",
        split,
        drive,
        f"{stem}_leftImg8bit.png",
    ])
    return Sample(part, split, drive, stem, image_member, annotation_member)


def discover_samples(archive_path: Path, splits: Iterable[str]) -> list[Sample]:
    """List valid image/annotation pairs while scanning the nested archives."""
    selected = set(splits)
    samples: list[Sample] = []
    for release_part, inner in _inner_members(archive_path):
        names: set[str] = set()
        annotations: list[str] = []
        for member in inner:
            if not member.isfile():
                continue
            name = _canonical_member_name(member.name, release_part)
            names.add(name)
            if "gtFine" in PurePosixPath(name).parts and name.endswith("_gtFine_polygons.json"):
                annotations.append(name)
        for annotation in annotations:
            sample = _sample_from_annotation(annotation)
            if sample is not None and sample.split in selected and sample.image_member in names:
                samples.append(sample)
    if not samples:
        raise FileNotFoundError(
            "No IDD RGB/polygon pairs were found for selected splits: "
            + ", ".join(sorted(selected))
        )
    return sorted(samples, key=lambda sample: sample.key)


def select_samples(samples: list[Sample], sample_keys: str | None, *, num_samples: int,
                   seed: int) -> list[Sample]:
    by_key = {sample.key: sample for sample in samples}
    if sample_keys is not None:
        keys = parse_csv(sample_keys)
        if not keys:
            raise ValueError("--sample-keys must not be empty")
        if len(set(keys)) != len(keys):
            raise ValueError("--sample-keys must not contain duplicates")
        missing = [key for key in keys if key not in by_key]
        if missing:
            raise ValueError("Unknown IDD sample key(s): " + ", ".join(missing))
        return [by_key[key] for key in keys]
    if num_samples < 1:
        raise ValueError("--num-samples must be at least one")
    if num_samples > len(samples):
        raise ValueError(f"Requested {num_samples} samples, but only {len(samples)} are available")
    return sorted(random.Random(seed).sample(samples, num_samples), key=lambda sample: sample.key)


def extract_selected_members(archive_path: Path, samples: Iterable[Sample], source_root: Path) -> None:
    """Stream only selected RGB and annotation members into a source tree."""
    wanted = {
        member
        for sample in samples
        for member in (sample.image_member, sample.annotation_member)
    }
    remaining = set(wanted)
    for release_part, inner in _inner_members(archive_path):
        for member in inner:
            if not member.isfile():
                continue
            name = _canonical_member_name(member.name, release_part)
            if name not in remaining:
                continue
            source = inner.extractfile(member)
            if source is None:
                raise OSError(f"Could not read IDD tar member: {member.name}")
            destination = source_root / PurePosixPath(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            remaining.remove(name)
        if not remaining:
            break
    if remaining:
        raise FileNotFoundError(
            "The IDD archive did not contain all requested source files:\n  "
            + "\n  ".join(sorted(remaining))
        )


def _run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def build_prepared_previews(source_root: Path, out_root: Path, args: argparse.Namespace) -> None:
    shared = ["--idd-root", str(source_root), "--idd-splits", args.splits,
              "--seed", str(args.seed), "--num-workers", "1"]
    _run([
        sys.executable, str(ROOT / "dataset_prep/semseg/dataset_prep.py"),
        "--out-root", str(out_root / "semseg"), *shared,
        "--variants-per-image", str(args.semseg_variants_per_image),
    ])
    _run([
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--idd-tar", type=Path, required=True,
                        help="Outer idd_1_2.tar containing idd-segmentation.tar.gz and idd-20k-II.tar.gz.")
    parser.add_argument("--out-root", type=Path, required=True,
                        help="New output directory for the mini source tree and both galleries.")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of image/polygon pairs to sample when --sample-keys is unset (default: 4).")
    parser.add_argument("--sample-keys", default=None,
                        help="Optional comma-separated keys from selected_samples.json.")
    parser.add_argument("--splits", default="train,val",
                        help="IDD splits from which to sample (default: train,val).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--semseg-variants-per-image", type=int, default=1)
    parser.add_argument("--instanceseg-variants-per-image", type=int, default=1)
    parser.add_argument("--instanceseg-negative-ratio", type=float, default=0.2)
    parser.add_argument("--draw-mode", default="fixeddraw_32_lr",
                        help="Instance target/prompt draw mode (default: fixeddraw_32_lr).")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    archive = args.idd_tar.expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"IDD tar does not exist: {archive}")
    out_root = args.out_root.expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"Preview output directory must be empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    splits = normalize_splits(args.splits)
    samples = discover_samples(archive, splits)
    selected = select_samples(samples, args.sample_keys, num_samples=args.num_samples, seed=args.seed)
    print("Selected IDD samples:")
    for sample in selected:
        print(f"  {sample.key}")
    print("Streaming selected RGB/polygon pairs from the nested IDD archives...")
    source_root = out_root / "source"
    extract_selected_members(archive, selected, source_root)
    build_prepared_previews(source_root, out_root, args)
    semantic_rows = write_gallery(out_root / "semseg", "IDD semantic-segmentation preview")
    instance_rows = write_gallery(out_root / "instanceseg", "IDD instance-segmentation preview")
    (out_root / "selected_samples.json").write_text(
        json.dumps({"splits": splits, "samples": [asdict(sample) | {"key": sample.key} for sample in selected]}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print("Preview complete.")
    print(f"  semantic rows: {semantic_rows} -> {out_root / 'semseg/index.html'}")
    print(f"  instance rows: {instance_rows} -> {out_root / 'instanceseg/index.html'}")


if __name__ == "__main__":
    main()
