#!/usr/bin/env python3
"""Preview prepared PACO-LVIS instance-segmentation examples from its raw tar.

The tool reads ``paco-lvis-DatasetNinja.tar`` directly, copies only ``meta.json``
and selected ``<split>2017/{img,ann}`` JPEG/JSON pairs to a tiny source tree, then
runs the production instance-segmentation converter. The HTML gallery shows the
exact conditioning image, prompt, and prepared target.

Example:

    python scripts/dataset_prep/preview_paco_lvis_tar.py \\
        --paco-lvis-tar datasets_archives/instanceseg/paco-lvis-DatasetNinja.tar \\
        --out-root /scratch/paco_lvis_preview --num-samples 4

Use ``--sample-ids`` with the ``split/stem`` keys in ``selected_samples.json`` to
reproduce particular examples. The tar is indexed to find candidates, but only
its metadata and requested JPEG/annotation pairs are written below ``--out-root``.
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
    "test": "test",
    "testing": "test",
}


@dataclass(frozen=True)
class Sample:
    stem: str
    split: str
    image_member: str
    annotation_member: str

    @property
    def key(self) -> str:
        return f"{self.split}/{self.stem}"


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_splits(value: str) -> list[str]:
    selected: list[str] = []
    for split in parse_csv(value):
        normalized = SPLIT_ALIASES.get(split.lower())
        if normalized is None:
            raise ValueError("--splits must contain train, val, and/or test")
        if normalized not in selected:
            selected.append(normalized)
    if not selected:
        raise ValueError("--splits must not be empty")
    return selected

def safe_member_name(name: str) -> str:
    path = PurePosixPath(name)
    parts = [part for part in path.parts if part not in ("", ".")]
    if path.is_absolute() or not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe PACO-LVIS archive member: {name}")
    return "/".join(parts)
    return "/".join(parts)


def archive_members(archive_path: Path) -> dict[str, tarfile.TarInfo]:
    """Index file members without extracting the high-file-count source tree."""
    members: dict[str, tarfile.TarInfo] = {}
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            if member.isfile():
                members[safe_member_name(member.name)] = member
    return members


def copy_member(archive: tarfile.TarFile, member: tarfile.TarInfo, destination: Path) -> None:
    source = archive.extractfile(member)
    if source is None:
        raise OSError(f"Could not read PACO-LVIS member: {member.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source, destination.open("wb") as handle:
        shutil.copyfileobj(source, handle)


def extract_metadata(archive_path: Path, source_root: Path) -> None:
    """Copy PACO's class table, which the production converter requires."""
    with tarfile.open(archive_path, "r:*") as archive:
        members = {safe_member_name(member.name): member for member in archive.getmembers() if member.isfile()}
        member = members.get("meta.json")
        if member is None:
            raise FileNotFoundError("PACO-LVIS tar is missing meta.json")
        copy_member(archive, member, source_root / "meta.json")


def discover_samples(archive_path: Path, splits: Iterable[str]) -> list[Sample]:
    """Find JPEG/annotation pairs for the requested native PACO splits."""
    selected = set(splits)
    members = archive_members(archive_path)
    samples: list[Sample] = []
    for name in members:
        path = PurePosixPath(name)
        parts = path.parts
        if len(parts) != 3 or not parts[0].endswith("2017") or parts[1] != "ann":
            continue
        split = parts[0].removesuffix("2017")
        if split not in selected or not path.name.endswith(".jpg.json"):
            continue
        image_member = f"{parts[0]}/img/{path.name.removesuffix('.json')}"
        if image_member not in members:
            continue
        samples.append(Sample(
            stem=path.name.removesuffix(".jpg.json"), split=split,
            image_member=image_member, annotation_member=name,
        ))
    if not samples:
        raise FileNotFoundError(
            "No PACO-LVIS JPEG/annotation pairs were found for selected splits: "
            + ", ".join(sorted(selected))
        )
    return sorted(samples, key=lambda sample: (sample.split, sample.stem))


def select_samples(samples: list[Sample], sample_ids: str | None, *, num_samples: int,
                   seed: int) -> list[Sample]:
    by_key = {sample.key: sample for sample in samples}
    by_stem: dict[str, list[Sample]] = {}
    for sample in samples:
        by_stem.setdefault(sample.stem, []).append(sample)
    if sample_ids is not None:
        requested = parse_csv(sample_ids)
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("--sample-ids must be non-empty and contain no duplicates")
        selected: list[Sample] = []
        for sample_id in requested:
            sample = by_key.get(sample_id)
            if sample is None and "/" not in sample_id:
                candidates = by_stem.get(sample_id, [])
                if len(candidates) == 1:
                    sample = candidates[0]
                elif len(candidates) > 1:
                    raise ValueError(
                        f"Ambiguous PACO-LVIS sample ID {sample_id!r}; use split/stem instead")
            if sample is None:
                raise ValueError(f"Unknown or unselected PACO-LVIS sample ID: {sample_id}")
            selected.append(sample)
        return selected
    if not 1 <= num_samples <= len(samples):
        raise ValueError(f"--num-samples must be in [1, {len(samples)}]")
    return sorted(random.Random(seed).sample(samples, num_samples), key=lambda sample: sample.key)


def extract_selected_members(archive_path: Path, samples: Iterable[Sample], source_root: Path) -> None:
    """Copy only the selected RGB JPEG and Supervisely JSON annotation members."""
    remaining = {
        member
        for sample in samples
        for member in (sample.image_member, sample.annotation_member)
    }
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            name = safe_member_name(member.name)
            if name not in remaining:
                continue
            copy_member(archive, member, source_root / PurePosixPath(name))
            remaining.remove(name)
            if not remaining:
                break
    if remaining:
        raise FileNotFoundError(
            "The PACO-LVIS tar did not contain all requested source files:\n  "
            + "\n  ".join(sorted(remaining))
        )


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def build_prepared_preview(source_root: Path, out_root: Path, args: argparse.Namespace) -> None:
    run([
        sys.executable, str(ROOT / "dataset_prep/instanceseg/dataset_prep.py"),
        "--out-root", str(out_root / "instanceseg"),
        "--paco-lvis-root", str(source_root),
        "--paco-lvis-splits", args.splits,
        "--seed", str(args.seed), "--num-workers", "1",
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
    parser.add_argument("--paco-lvis-tar", type=Path, required=True,
                        help="Raw paco-lvis-DatasetNinja.tar from DatasetNinja.")
    parser.add_argument("--out-root", type=Path, required=True,
                        help="New output directory for the mini source tree and HTML gallery.")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of JPEG/annotation pairs to sample when --sample-ids is unset (default: 4).")
    parser.add_argument("--sample-ids", default=None,
                        help="Optional comma-separated PACO IDs as split/stem (or an unambiguous bare stem).")
    parser.add_argument("--splits", default="train,val,test",
                        help="PACO-LVIS splits to sample: train,val,test (default: train,val,test).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instanceseg-variants-per-image", type=int, default=1)
    parser.add_argument("--instanceseg-negative-ratio", type=float, default=0.2)
    parser.add_argument("--draw-mode", default="fixeddraw_32_lr",
                        help="Instance target/prompt draw mode (default: fixeddraw_32_lr).")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    archive = args.paco_lvis_tar.expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"PACO-LVIS tar does not exist: {archive}")
    splits = normalize_splits(args.splits)
    out_root = args.out_root.expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"Preview output directory must be empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    source_root = out_root / "source"

    print("Indexing PACO-LVIS image/annotation pairs in the raw tar...")
    samples = discover_samples(archive, splits)
    selected = select_samples(samples, args.sample_ids, num_samples=args.num_samples, seed=args.seed)
    print("Selected PACO-LVIS samples:")
    for sample in selected:
        print(f"  {sample.key}")
    print("Extracting PACO-LVIS class metadata and selected JPEG/JSON pairs...")
    extract_metadata(archive, source_root)
    extract_selected_members(archive, selected, source_root)
    build_prepared_preview(source_root, out_root, args)
    rows = write_gallery(out_root / "instanceseg", "PACO-LVIS instance-segmentation preview")
    (out_root / "selected_samples.json").write_text(json.dumps({
        "splits": splits, "samples": [asdict(sample) | {"key": sample.key} for sample in selected],
    }, indent=2) + "\n", encoding="utf-8")
    print("Preview complete.")
    print(f"  instance rows: {rows} -> {out_root / 'instanceseg/index.html'}")


if __name__ == "__main__":
    main()
