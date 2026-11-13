#!/usr/bin/env python3
"""Preview prepared semantic- and instance-segmentation WildDash 2 examples.

The tool accepts the raw ``wd_public_v2p.tar`` directly. It writes the small
panoptic index/split metadata plus only selected anonymized RGB and panoptic PNG
pairs into a temporary source tree, then invokes the production converters.
Both output galleries show the exact conditioning image, prompt, and target.

Example:

    python scripts/dataset_prep/preview_wilddash2_tar.py \\
        --wilddash2-tar /data/shared_seg/wd_public_v2p.tar \\
        --out-root /scratch/wilddash2_preview --num-samples 4

Use ``--sample-ids`` with a frame stem printed in ``selected_samples.json`` to
repeat a particular example. The archive is large and has no central directory,
so it is scanned to locate selected files; only requested sample files are
written below ``--out-root/source/``.
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
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
}
SPLIT_FILES = {"train": "training.txt", "val": "validation.txt"}
METADATA_MEMBERS = frozenset({
    "panoptic.json",
    "wilddash2-meta.json",
    "random_split/training.txt",
    "random_split/validation.txt",
})


@dataclass(frozen=True)
class Sample:
    stem: str
    split: str
    image_member: str
    panoptic_member: str


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_splits(value: str) -> list[str]:
    selected: list[str] = []
    for split in parse_csv(value):
        normalized = SPLIT_ALIASES.get(split.lower())
        if normalized is None:
            raise ValueError("--splits must contain train and/or val")
        if normalized not in selected:
            selected.append(normalized)
    if not selected:
        raise ValueError("--splits must not be empty")
    return selected


def _member_name(name: str) -> str:
    parts = [part for part in PurePosixPath(name).parts if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise ValueError(f"Unsafe WildDash 2 archive member: {name}")
    return "/".join(parts)


def extract_metadata_members(archive_path: Path, source_root: Path) -> None:
    """Copy only index, label metadata, and official random split lists."""
    remaining = set(METADATA_MEMBERS)
    with tarfile.open(archive_path, "r|*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = _member_name(member.name)
            if name not in remaining:
                continue
            source = archive.extractfile(member)
            if source is None:
                raise OSError(f"Could not read WildDash 2 member: {member.name}")
            destination = source_root / PurePosixPath(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            remaining.remove(name)
            if not remaining:
                break
    if remaining:
        raise FileNotFoundError(
            "The WildDash 2 tar did not contain required metadata:\n  "
            + "\n  ".join(sorted(remaining))
        )


def discover_samples(source_root: Path, splits: Iterable[str]) -> list[Sample]:
    """Discover image/panoptic pairs from the extracted official index/splits."""
    selected = set(splits)
    index = json.loads((source_root / "panoptic.json").read_text(encoding="utf-8"))
    images = {Path(str(image["file_name"])).stem: image for image in index.get("images", ())}
    annotations = {str(ann.get("image_id")): ann for ann in index.get("annotations", ())}
    samples: list[Sample] = []
    for split in sorted(selected):
        stems = (source_root / "random_split" / SPLIT_FILES[split]).read_text(encoding="utf-8").split()
        for stem in stems:
            image = images.get(stem)
            if image is None:
                continue
            annotation = annotations.get(str(image.get("id")))
            if annotation is None:
                continue
            samples.append(Sample(
                stem=stem,
                split=split,
                image_member=f"anonymized/{image['file_name']}",
                panoptic_member=f"panoptic/{annotation['file_name']}",
            ))
    if not samples:
        raise FileNotFoundError(
            "No WildDash 2 image/panoptic pairs were found for selected splits: "
            + ", ".join(sorted(selected))
        )
    return sorted(samples, key=lambda sample: (sample.split, sample.stem))


def select_samples(samples: list[Sample], sample_ids: str | None, *, num_samples: int,
                   seed: int) -> list[Sample]:
    by_stem = {sample.stem: sample for sample in samples}
    if sample_ids is not None:
        stems = parse_csv(sample_ids)
        if not stems:
            raise ValueError("--sample-ids must not be empty")
        if len(set(stems)) != len(stems):
            raise ValueError("--sample-ids must not contain duplicates")
        missing = [stem for stem in stems if stem not in by_stem]
        if missing:
            raise ValueError("Unknown or unselected WildDash 2 sample ID(s): " + ", ".join(missing))
        return [by_stem[stem] for stem in stems]
    if num_samples < 1:
        raise ValueError("--num-samples must be at least one")
    if num_samples > len(samples):
        raise ValueError(f"Requested {num_samples} samples, but only {len(samples)} are available")
    return sorted(random.Random(seed).sample(samples, num_samples), key=lambda sample: sample.stem)


def extract_selected_members(archive_path: Path, samples: Iterable[Sample], source_root: Path) -> None:
    """Stream selected anonymized RGB + panoptic PNG members into the source tree."""
    remaining = {
        member
        for sample in samples
        for member in (sample.image_member, sample.panoptic_member)
    }
    with tarfile.open(archive_path, "r|*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = _member_name(member.name)
            if name not in remaining:
                continue
            source = archive.extractfile(member)
            if source is None:
                raise OSError(f"Could not read WildDash 2 member: {member.name}")
            destination = source_root / PurePosixPath(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            remaining.remove(name)
            if not remaining:
                break
    if remaining:
        raise FileNotFoundError(
            "The WildDash 2 tar did not contain all requested source files:\n  "
            + "\n  ".join(sorted(remaining))
        )


def _run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def build_prepared_previews(source_root: Path, out_root: Path, args: argparse.Namespace) -> None:
    shared = ["--wilddash2-root", str(source_root), "--wilddash2-splits", args.splits,
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
    parser.add_argument("--wilddash2-tar", type=Path, required=True,
                        help="Raw wd_public_v2p.tar carrying anonymized/, panoptic/, and metadata.")
    parser.add_argument("--out-root", type=Path, required=True,
                        help="New output directory for the mini source tree and both galleries.")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of image/panoptic pairs to sample when --sample-ids is unset (default: 4).")
    parser.add_argument("--sample-ids", default=None,
                        help="Optional comma-separated WildDash 2 frame stems, e.g. ae0002_100000.")
    parser.add_argument("--splits", default="train,val",
                        help="WildDash 2 random splits from which to sample (default: train,val).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--semseg-variants-per-image", type=int, default=1)
    parser.add_argument("--instanceseg-variants-per-image", type=int, default=1)
    parser.add_argument("--instanceseg-negative-ratio", type=float, default=0.2)
    parser.add_argument("--draw-mode", default="fixeddraw_32_lr",
                        help="Instance target/prompt draw mode (default: fixeddraw_32_lr).")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    archive = args.wilddash2_tar.expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"WildDash 2 tar does not exist: {archive}")
    out_root = args.out_root.expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"Preview output directory must be empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    splits = normalize_splits(args.splits)
    source_root = out_root / "source"
    print("Extracting WildDash 2 index, label metadata, and official random split lists...")
    extract_metadata_members(archive, source_root)
    samples = discover_samples(source_root, splits)
    selected = select_samples(samples, args.sample_ids, num_samples=args.num_samples, seed=args.seed)
    print("Selected WildDash 2 samples:")
    for sample in selected:
        print(f"  {sample.split}: {sample.stem}")
    print("Streaming selected anonymized RGB/panoptic pairs from the raw tar...")
    extract_selected_members(archive, selected, source_root)
    build_prepared_previews(source_root, out_root, args)
    semantic_rows = write_gallery(out_root / "semseg", "WildDash 2 semantic-segmentation preview")
    instance_rows = write_gallery(out_root / "instanceseg", "WildDash 2 instance-segmentation preview")
    (out_root / "selected_samples.json").write_text(
        json.dumps({"splits": splits, "samples": [asdict(sample) for sample in selected]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Preview complete.")
    print(f"  semantic rows: {semantic_rows} -> {out_root / 'semseg/index.html'}")
    print(f"  instance rows: {instance_rows} -> {out_root / 'instanceseg/index.html'}")


if __name__ == "__main__":
    main()
