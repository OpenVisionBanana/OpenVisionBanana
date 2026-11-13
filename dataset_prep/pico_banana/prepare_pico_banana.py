#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - convenience fallback
    tqdm = lambda iterable, **_: iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def safe_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._")
    return value or "row"


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"Line {line_number} in {path} is not a JSON object.")
                records.append(record)
        return records

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "records", "annotations", "examples"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Could not find a list of records in {path}.")


def candidate_values(value: str) -> Iterable[str]:
    yield value
    prefixes = ("nano-banana-300k/", "pico-banana-400k/")
    for prefix in prefixes:
        if value.startswith(prefix):
            yield value[len(prefix) :]


def resolve_path(value: Any, roots: list[Path]) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path if path.exists() else None
    for candidate_value in candidate_values(str(value)):
        for root in roots:
            candidate = root / candidate_value
            if candidate.exists():
                return candidate
    return None


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


def valid_geometry(image: Image.Image, min_side: int | None, max_aspect_ratio: float | None) -> bool:
    width, height = image.size
    if min_side is not None and min(width, height) < min_side:
        return False
    if max_aspect_ratio is not None:
        aspect = max(width / height, height / width)
        if aspect > max_aspect_ratio:
            return False
    return True


def materialize_image(
    src: Path,
    dst: Path,
    copy_mode: str,
    max_side: int | None,
    min_side: int | None,
    max_aspect_ratio: float | None,
) -> bool:
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image)
        if not valid_geometry(image, min_side, max_aspect_ratio):
            return False
        resized = maybe_resize(image, max_side)
        needs_reencode = (
            resized.size != image.size
            or dst.suffix.lower() != src.suffix.lower()
            or src.suffix.lower() not in IMAGE_SUFFIXES
        )

        dst.parent.mkdir(parents=True, exist_ok=True)
        if needs_reencode:
            resized.save(dst.with_suffix(".png"))
            return True

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
    return True


def output_relpath(kind: str, stem: str, src: Path, force_png: bool) -> Path:
    suffix = ".png" if force_png else src.suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        suffix = ".png"
    return Path(kind) / f"{stem}{suffix}"


def add_task_prefix(text: str, prefix: str, no_task_prefix: bool) -> str:
    text = text.strip()
    if no_task_prefix or text.startswith("[TASK:"):
        return text
    return f"{prefix} {text}"


def process_record(
    source_index: int,
    record: dict[str, Any],
    roots: list[str],
    out_root: str,
    source_image_key: str,
    target_image_key: str,
    instruction_key: str,
    fallback_instruction_key: str,
    copy_mode: str,
    max_side: int | None,
    min_side: int | None,
    max_aspect_ratio: float | None,
    instruction_prefix: str,
    no_task_prefix: bool,
) -> tuple[str, dict[str, Any] | None]:
    root_paths = [Path(root) for root in roots]
    out_root_path = Path(out_root)

    source = resolve_path(record.get(source_image_key), root_paths)
    target = resolve_path(record.get(target_image_key), root_paths)
    instruction = record.get(instruction_key) or record.get(fallback_instruction_key)
    if source is None or target is None or not instruction:
        return "skipped_missing", None

    raw_id = record.get("id") or Path(str(record.get(target_image_key))).stem or source_index
    stem = f"pico_{source_index:08d}_{safe_stem(str(raw_id))}"
    force_png = max_side is not None
    source_rel = output_relpath("conditioning_images", f"{stem}_source", source, force_png)
    target_rel = output_relpath("target_images", f"{stem}_target", target, force_png)
    source_dst = out_root_path / source_rel
    target_dst = out_root_path / target_rel

    try:
        source_ok = materialize_image(
            source,
            source_dst,
            copy_mode,
            max_side,
            min_side,
            max_aspect_ratio,
        )
        target_ok = materialize_image(
            target,
            target_dst,
            copy_mode,
            max_side,
            min_side,
            max_aspect_ratio,
        )
    except Exception:
        return "skipped_corrupt", None

    if not source_ok or not target_ok:
        return "skipped_geometry", None

    if source_dst.with_suffix(".png").exists():
        source_rel = source_rel.with_suffix(".png")
    if target_dst.with_suffix(".png").exists():
        target_rel = target_rel.with_suffix(".png")

    return (
        "written",
        {
            "conditioning_image": source_rel.as_posix(),
            "image": target_rel.as_posix(),
            "text": add_task_prefix(str(instruction), instruction_prefix, no_task_prefix),
            "task": "image_editing",
            "source_dataset": "pico_banana_400k",
            "id": stem,
            "metadata": {
                "edit_type": record.get("edit_type"),
                "source_metadata": record,
            },
        },
    )


def process_record_task(task: tuple[Any, ...]) -> tuple[str, dict[str, Any] | None]:
    return process_record(*task)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Pico-Banana rows in the canonical image-to-image format.")
    parser.add_argument("--pico-root", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--source-image-key", default="local_input_image")
    parser.add_argument("--target-image-key", default="output_image")
    parser.add_argument("--instruction-key", default="text")
    parser.add_argument("--fallback-instruction-key", default="summarized_text")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-side", type=int, default=1024, help="Resize larger images to this side; <=0 disables.")
    parser.add_argument("--min-side", type=int, default=None)
    parser.add_argument("--max-aspect-ratio", type=float, default=None)
    parser.add_argument("--copy-mode", choices=["copy", "symlink", "hardlink"], default="copy")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel workers for materializing image pairs.")
    parser.add_argument("--instruction-prefix", default="[TASK: IMAGE_EDITING]")
    parser.add_argument("--no-task-prefix", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.max_side is not None and args.max_side <= 0:
        args.max_side = None
    pico_root = args.pico_root.expanduser().resolve()
    metadata = args.metadata.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    roots = [pico_root, metadata.parent, pico_root / "openimages"]

    records = read_records(metadata)
    rng = random.Random(args.seed)
    rng.shuffle(records)

    out_conditioning = out_root / "conditioning_images"
    out_target = out_root / "target_images"
    out_conditioning.mkdir(parents=True, exist_ok=True)
    out_target.mkdir(parents=True, exist_ok=True)

    counts = {
        "read": len(records),
        "written": 0,
        "skipped_missing": 0,
        "skipped_corrupt": 0,
        "skipped_geometry": 0,
    }
    metadata_rows = []
    progress = tqdm(total=len(records), desc="Preparing Pico-Banana")

    def handle_result(status: str, metadata_row: dict[str, Any] | None) -> None:
        if status == "written":
            metadata_rows.append(metadata_row)
            counts["written"] += 1
            return
        counts[status] += 1

    common_task_args = (
        [str(root) for root in roots],
        str(out_root),
        args.source_image_key,
        args.target_image_key,
        args.instruction_key,
        args.fallback_instruction_key,
        args.copy_mode,
        args.max_side,
        args.min_side,
        args.max_aspect_ratio,
        args.instruction_prefix,
        args.no_task_prefix,
    )

    if args.num_workers <= 1:
        for source_index, record in enumerate(records):
            if args.limit is not None and counts["written"] >= args.limit:
                break
            status, metadata_row = process_record(source_index, record, *common_task_args)
            handle_result(status, metadata_row)
            progress.update(1)
    else:
        batch_size = max(args.num_workers * 32, 512)
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            for batch_start in range(0, len(records), batch_size):
                if args.limit is not None and counts["written"] >= args.limit:
                    break
                batch = records[batch_start : batch_start + batch_size]
                tasks = [
                    (batch_start + offset, record, *common_task_args)
                    for offset, record in enumerate(batch)
                ]
                chunksize = max(1, len(tasks) // max(1, args.num_workers * 4))
                for status, metadata_row in executor.map(process_record_task, tasks, chunksize=chunksize):
                    handle_result(status, metadata_row)
                    progress.update(1)

    progress.close()

    if counts["written"] == 0:
        raise RuntimeError(
            "Pico-Banana preparation produced zero rows. "
            "Check that the mapped source images exist under "
            f"{pico_root / 'openimages/openimage_source_images'} and that edited target images exist under "
            f"{pico_root / 'openimages/images/positive-edit'}. "
            f"Counts: read={counts['read']}, skipped_missing={counts['skipped_missing']}, "
            f"skipped_corrupt={counts['skipped_corrupt']}, skipped_geometry={counts['skipped_geometry']}."
        )

    with (out_root / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print("Pico-Banana preparation complete.")
    for key in ("read", "written", "skipped_missing", "skipped_corrupt", "skipped_geometry"):
        print(f"  {key}: {counts[key]:,}")
    print(f"  out_root: {out_root}")


if __name__ == "__main__":
    main()
