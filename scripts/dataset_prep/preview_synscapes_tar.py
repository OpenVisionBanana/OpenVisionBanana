#!/usr/bin/env python3
"""Preview prepared semantic- and instance-segmentation Synscapes examples.

The tool accepts the *raw* Synscapes tar directly.  It streams only the selected
frames' RGB, semantic, instance, and metadata files into a tiny temporary source
tree, then invokes the production converters on that tree.  The result is a
browser-ready gallery for both tasks, including the exact conditioning image,
prompt, and generated target.

Example:

    python scripts/dataset_prep/preview_synscapes_tar.py \
        --synscapes-tar /data/shared_seg/synscapes.tar \
        --out-root /scratch/synscapes_preview --num-samples 4

The raw release is large.  A tar has no central directory, so locating selected
members may scan through part (or all) of the archive; only the requested frames
are written to ``--out-root/source/``.
"""
from __future__ import annotations

import argparse
import html
import json
import random
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
SYNSCAPES_FIRST_INDEX = 1
SYNSCAPES_TRAIN_LAST_INDEX = 24_000
SYNSCAPES_LAST_INDEX = 25_000


def split_for_index(index: int) -> str | None:
    if SYNSCAPES_FIRST_INDEX <= index <= SYNSCAPES_TRAIN_LAST_INDEX:
        return "train"
    if SYNSCAPES_TRAIN_LAST_INDEX < index <= SYNSCAPES_LAST_INDEX:
        return "val"
    return None


def parse_csv(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_sample_ids(value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        ids = [int(item) for item in parse_csv(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--sample-ids must be a comma-separated list of integer frame IDs"
        ) from exc
    if not ids:
        raise argparse.ArgumentTypeError("--sample-ids must not be empty")
    if len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError("--sample-ids must not contain duplicates")
    return ids


def select_indices(sample_ids: list[int] | None, *, num_samples: int,
                   splits: Iterable[str], seed: int) -> list[int]:
    selected_splits = set(splits)
    if not selected_splits or not selected_splits <= {"train", "val"}:
        raise ValueError("--splits must contain train and/or val")
    if sample_ids is not None:
        invalid = [index for index in sample_ids if split_for_index(index) not in selected_splits]
        if invalid:
            raise ValueError(
                "Requested sample IDs are outside --splits or the 1..25000 release: "
                f"{invalid}"
            )
        return sorted(sample_ids)

    candidates = [
        index for index in range(SYNSCAPES_FIRST_INDEX, SYNSCAPES_LAST_INDEX + 1)
        if split_for_index(index) in selected_splits
    ]
    if num_samples < 1:
        raise ValueError("--num-samples must be at least one")
    if num_samples > len(candidates):
        raise ValueError(f"Requested {num_samples} samples, but only {len(candidates)} are available")
    return sorted(random.Random(seed).sample(candidates, num_samples))


def _needed_members(indices: Iterable[int], source_root: Path) -> dict[str, Path]:
    targets: dict[str, Path] = {}
    for index in indices:
        for relative in (
            f"img/rgb/{index}.png",
            f"img/class/{index}.png",
            f"img/instance/{index}.png",
            f"meta/{index}.json",
        ):
            targets[f"synscapes/{relative}".lower()] = source_root / "Synscapes" / relative
    return targets


def extract_selected_members(archive_path: Path, indices: Iterable[int], source_root: Path) -> None:
    """Stream only the source files required for the selected frame IDs."""
    wanted = _needed_members(indices, source_root)
    remaining = dict(wanted)
    with tarfile.open(archive_path, "r|*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            normalized = member.name.removeprefix("./").lower()
            destination = remaining.pop(normalized, None)
            if destination is None:
                continue
            source = archive.extractfile(member)
            if source is None:
                raise OSError(f"Could not read tar member: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            if not remaining:
                break
    if remaining:
        missing = "\n  ".join(sorted(remaining))
        raise FileNotFoundError(
            "The Synscapes tar did not contain all required files:\n  " + missing
        )


def _run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def build_prepared_previews(source_root: Path, out_root: Path, args: argparse.Namespace) -> None:
    semseg_out = out_root / "semseg"
    instanceseg_out = out_root / "instanceseg"
    shared = ["--synscapes-root", str(source_root), "--synscapes-splits", args.splits,
              "--seed", str(args.seed), "--num-workers", "1"]
    _run([
        sys.executable, str(ROOT / "dataset_prep/semseg/dataset_prep.py"),
        "--out-root", str(semseg_out), *shared,
        "--variants-per-image", str(args.semseg_variants_per_image),
    ])
    _run([
        sys.executable, str(ROOT / "dataset_prep/instanceseg/dataset_prep.py"),
        "--out-root", str(instanceseg_out), *shared,
        "--variants-per-image", str(args.instanceseg_variants_per_image),
        "--negative-ratio", str(args.instanceseg_negative_ratio),
    ])


def write_gallery(task_root: Path, title: str) -> int:
    rows = [
        json.loads(line) for line in (task_root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
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
    parser.add_argument("--synscapes-tar", type=Path, required=True,
                        help="Raw Synscapes tar containing Synscapes/img/{rgb,class,instance} and meta/.")
    parser.add_argument("--out-root", type=Path, required=True,
                        help="New output directory for the mini source tree and both galleries.")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of raw frames to sample when --sample-ids is unset (default: 4).")
    parser.add_argument("--sample-ids", default=None,
                        help="Optional comma-separated Synscapes frame IDs, e.g. 1,24000,24001.")
    parser.add_argument("--splits", default="train,val",
                        help="Logical Synscapes splits from which to sample (default: train,val).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--semseg-variants-per-image", type=int, default=1)
    parser.add_argument("--instanceseg-variants-per-image", type=int, default=1)
    parser.add_argument("--instanceseg-negative-ratio", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    archive = args.synscapes_tar.expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Synscapes tar does not exist: {archive}")
    out_root = args.out_root.expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"Preview output directory must be empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    splits = parse_csv(args.splits)
    indices = select_indices(parse_sample_ids(args.sample_ids), num_samples=args.num_samples,
                             splits=splits, seed=args.seed)
    source_root = out_root / "source"
    print(f"Selected Synscapes frames: {', '.join(map(str, indices))}")
    print("Streaming selected RGB/class/instance/meta files from the raw tar...")
    extract_selected_members(archive, indices, source_root)
    build_prepared_previews(source_root, out_root, args)
    semantic_rows = write_gallery(out_root / "semseg", "Synscapes semantic-segmentation preview")
    instance_rows = write_gallery(out_root / "instanceseg", "Synscapes instance-segmentation preview")
    (out_root / "selected_frames.json").write_text(
        json.dumps({"frames": indices, "splits": splits}, indent=2) + "\n", encoding="utf-8")
    print("Preview complete.")
    print(f"  semantic rows: {semantic_rows} -> {out_root / 'semseg/index.html'}")
    print(f"  instance rows: {instance_rows} -> {out_root / 'instanceseg/index.html'}")


if __name__ == "__main__":
    main()
