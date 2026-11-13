#!/usr/bin/env python3
"""Preview prepared semantic- and instance-segmentation NuImages examples.

The tool reads the nested ``nuimages.tar`` archive directly. It streams only
the train/val metadata tables and the selected annotated-keyframe JPEGs into a
small temporary source tree, then invokes the production converters. Both HTML
galleries show the exact conditioning image, prompt, and target mask.

Example:

    python scripts/dataset_prep/preview_nuimages_tar.py \
        --nuimages-tar datasets_archives/shared_seg/nuimages.tar \
        --out-root /scratch/nuimages_preview --num-samples 4

The preview follows the production default: first mask-bearing keyframe in each
driving log, then one no sooner than every 10 seconds. Use ``--sample-ids`` with
a sample token written to ``selected_samples.json`` to repeat a frame.
"""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT))
from common.semseg import taxonomy  # noqa: E402


OUTER_METADATA_MEMBER = "nuimages-v1.0-all-metadata.tgz"
OUTER_SAMPLES_MEMBER = "nuimages-v1.0-all-samples.tgz"
METADATA_MEMBERS = frozenset({
    "v1.0-train/category.json",
    "v1.0-train/sample.json",
    "v1.0-train/sample_data.json",
    "v1.0-train/object_ann.json",
    "v1.0-train/surface_ann.json",
    "v1.0-val/sample.json",
    "v1.0-val/sample_data.json",
    "v1.0-val/object_ann.json",
    "v1.0-val/surface_ann.json",
})


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_member_name(name: str) -> str:
    parts = [part for part in PurePosixPath(name).parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe NuImages archive member: {name}")
    return "/".join(parts)


def _open_nested(archive_path: Path, outer_member: str) -> tarfile.TarFile:
    outer = tarfile.open(archive_path, "r:*")
    try:
        member = outer.getmember(outer_member)
    except KeyError:
        outer.close()
        raise FileNotFoundError(f"NuImages outer archive is missing {outer_member}") from None
    source = outer.extractfile(member)
    if source is None:
        outer.close()
        raise OSError(f"Could not read {outer_member} from {archive_path}")
    # The inner TarFile owns the source only indirectly; keep the outer archive
    # alive on it until the caller closes the stream.
    inner = tarfile.open(fileobj=source, mode="r|gz")
    inner._nuimages_outer_archive = outer  # type: ignore[attr-defined]
    return inner


def close_nested(archive: tarfile.TarFile) -> None:
    outer = getattr(archive, "_nuimages_outer_archive", None)
    archive.close()
    if outer is not None:
        outer.close()


def copy_member(source, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        shutil.copyfileobj(source, handle)


def extract_metadata(archive_path: Path, source_root: Path) -> None:
    """Extract only JSON tables required by the production NuImages loader."""
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "samples").mkdir(exist_ok=True)
    remaining = set(METADATA_MEMBERS)
    archive = _open_nested(archive_path, OUTER_METADATA_MEMBER)
    try:
        for member in archive:
            if not member.isfile():
                continue
            name = safe_member_name(member.name)
            if name not in remaining:
                continue
            source = archive.extractfile(member)
            if source is None:
                raise OSError(f"Could not read NuImages metadata member: {member.name}")
            with source:
                copy_member(source, source_root / PurePosixPath(name))
            remaining.remove(name)
            if not remaining:
                break
    finally:
        close_nested(archive)
    if remaining:
        raise FileNotFoundError(
            "NuImages metadata package is missing required table(s):\n  "
            + "\n  ".join(sorted(remaining))
        )


def select_frames(frames: list[taxonomy.NuImagesFrame], sample_ids: str | None,
                  *, num_samples: int, seed: int) -> list[taxonomy.NuImagesFrame]:
    by_token = {frame.token: frame for frame in frames}
    if sample_ids is not None:
        tokens = parse_csv(sample_ids)
        if not tokens or len(tokens) != len(set(tokens)):
            raise ValueError("--sample-ids must be non-empty and contain no duplicates")
        missing = [token for token in tokens if token not in by_token]
        if missing:
            raise ValueError("Unknown or unsampled NuImages sample token(s): " + ", ".join(missing))
        return [by_token[token] for token in tokens]
    if not 1 <= num_samples <= len(frames):
        raise ValueError(f"--num-samples must be in [1, {len(frames)}]")
    return sorted(random.Random(seed).sample(frames, num_samples),
                  key=lambda frame: (frame.split, frame.log_token, frame.timestamp, frame.token))


def extract_selected_images(archive_path: Path, frames: Iterable[taxonomy.NuImagesFrame],
                            source_root: Path) -> None:
    """Stream only selected annotated key-camera JPEGs from the nested tar."""
    remaining = {
        frame.image_path.relative_to(source_root).as_posix()
        for frame in frames
    }
    archive = _open_nested(archive_path, OUTER_SAMPLES_MEMBER)
    try:
        for member in archive:
            if not member.isfile():
                continue
            name = safe_member_name(member.name)
            if name not in remaining:
                continue
            source = archive.extractfile(member)
            if source is None:
                raise OSError(f"Could not read NuImages image member: {member.name}")
            with source:
                copy_member(source, source_root / PurePosixPath(name))
            remaining.remove(name)
            if not remaining:
                break
    finally:
        close_nested(archive)
    if remaining:
        raise FileNotFoundError(
            "NuImages samples package is missing selected image(s):\n  "
            + "\n  ".join(sorted(remaining))
        )


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def build_prepared_previews(source_root: Path, out_root: Path, args: argparse.Namespace) -> None:
    shared = [
        "--nuimages-root", str(source_root),
        "--nuimages-splits", args.splits,
        "--nuimages-sample-interval-seconds", str(args.sample_interval_seconds),
        "--seed", str(args.seed), "--num-workers", "1",
    ]
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
    parser.add_argument("--nuimages-tar", type=Path, required=True,
                        help="Raw nuimages.tar containing the two nested release tgz members.")
    parser.add_argument("--out-root", type=Path, required=True,
                        help="New output directory for the mini source tree and both galleries.")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--sample-ids", default=None,
                        help="Optional comma-separated NuImages sample tokens from selected_samples.json.")
    parser.add_argument("--splits", default="train,val",
                        help="NuImages labelled splits: train and/or val (default: train,val).")
    parser.add_argument("--sample-interval-seconds", type=float,
                        default=taxonomy.NUIMAGES_DEFAULT_SAMPLE_INTERVAL_SECONDS,
                        help="Per-log keyframe spacing in seconds (default: 10; 0 keeps all mask-bearing keyframes).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--semseg-variants-per-image", type=int, default=1)
    parser.add_argument("--instanceseg-variants-per-image", type=int, default=1)
    parser.add_argument("--instanceseg-negative-ratio", type=float, default=0.2)
    parser.add_argument("--draw-mode", default="fixeddraw_32_lr",
                        help="Instance target/prompt draw mode (default: fixeddraw_32_lr).")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    archive = args.nuimages_tar.expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"NuImages tar does not exist: {archive}")
    splits = taxonomy.normalize_nuimages_splits(parse_csv(args.splits))
    out_root = args.out_root.expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"Preview output directory must be empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    source_root = out_root / "source"

    print("Extracting NuImages train/val metadata tables...")
    extract_metadata(archive, source_root)
    frames, skipped = taxonomy.discover_nuimages_frames(
        source_root, sorted(splits), sample_interval_seconds=args.sample_interval_seconds,
        require_image_files=False)
    if not frames:
        raise RuntimeError("No mask-bearing NuImages keyframes remained after split/sampling selection")
    selected = select_frames(frames, args.sample_ids, num_samples=args.num_samples, seed=args.seed)
    print(f"Selected {len(selected)} NuImages frame(s); skipped unusable keyframes: {skipped}")
    for frame in selected:
        print(f"  {frame.split}: {frame.token} ({frame.log_token}, {frame.timestamp})")
    print("Streaming selected annotated key-camera JPEGs...")
    extract_selected_images(archive, selected, source_root)
    build_prepared_previews(source_root, out_root, args)
    semantic_rows = write_gallery(out_root / "semseg", "NuImages semantic-segmentation preview")
    instance_rows = write_gallery(out_root / "instanceseg", "NuImages instance-segmentation preview")
    (out_root / "selected_samples.json").write_text(json.dumps({
        "splits": sorted(splits), "sample_interval_seconds": args.sample_interval_seconds,
        "samples": [{
            "token": frame.token, "split": frame.split, "log_token": frame.log_token,
            "timestamp": frame.timestamp,
            "image_member": frame.image_path.relative_to(source_root).as_posix(),
        } for frame in selected],
    }, indent=2) + "\n", encoding="utf-8")
    print("Preview complete.")
    print(f"  semantic rows: {semantic_rows} -> {out_root / 'semseg/index.html'}")
    print(f"  instance rows: {instance_rows} -> {out_root / 'instanceseg/index.html'}")


if __name__ == "__main__":
    main()
