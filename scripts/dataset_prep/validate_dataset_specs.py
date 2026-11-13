#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from common.indexed_tar import validate_archive_fingerprint

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover - optional validation mode
    Image = None
    ImageOps = None


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_number, json.loads(line)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def validate_indexed_prepared_tar(
    name: str,
    spec: dict,
    weight: float,
    config_root: Path,
    check_images: int,
    check_decode: bool,
    allow_missing_loss_masks: bool,
) -> None:
    archive = resolve(config_root, spec["path"]).expanduser().resolve()
    index = resolve(config_root, spec["index_path"]).expanduser().resolve()
    if not archive.is_file() or not index.is_file():
        raise FileNotFoundError(f"{name}: missing indexed TAR or index: {archive}, {index}")
    with sqlite3.connect(f"file:{index}?mode=ro", uri=True) as connection:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if int(metadata.get("schema_version", 0)) != 1:
            raise ValueError(f"{name}: unsupported indexed-tar schema")
        validate_archive_fingerprint(archive, json.loads(metadata["archive_fingerprint"]))
        rows = int(connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0])
        if rows <= 0 or rows != int(metadata.get("num_rows", -1)):
            raise ValueError(f"{name}: invalid indexed row count {rows}")
        limit = rows if check_images < 0 else min(rows, max(0, check_images))
        payloads = [
            json.loads(payload)
            for (payload,) in connection.execute(
                "SELECT payload FROM rows ORDER BY row_number LIMIT ?", (limit,)
            )
        ]
        members = {
            member: (int(offset), int(size))
            for member, offset, size in connection.execute("SELECT name,offset,size FROM members")
        }
    fd = os.open(archive, os.O_RDONLY) if check_decode and payloads else None
    try:
        task_counts: dict[str, int] = {}
        for row_number, row in enumerate(payloads, 1):
            task = str(row.get("task") or "unknown").replace("-", "_")
            task_counts[task] = task_counts.get(task, 0) + 1
            required = ["image", "text"]
            if task not in {"text_to_image", "text2image", "image_generation"}:
                required.append("conditioning_image")
                geometry = task in {
                    "depth", "depth_estimation", "surface_normal",
                    "surface_normals", "surface_normal_estimation",
                }
                # Geometry normally requires an explicit loss mask.  A small set
                # of archived legacy sources predates that contract and is
                # explicitly opted in by its dataset spec; keep all current
                # geometry sources strict.
                allow_spec_missing_loss_mask = bool(spec.get("allow_missing_loss_mask", False))
                if (geometry and not allow_spec_missing_loss_mask) or (
                    not geometry and not allow_missing_loss_masks
                ):
                    required.append("loss_mask")
            for field in required:
                value = row.get(field)
                if value in (None, ""):
                    raise ValueError(f"{name}: indexed row {row_number} missing {field}")
                if field == "text":
                    continue
                if value not in members:
                    raise ValueError(f"{name}: indexed row references missing member {value!r}")
                if fd is not None:
                    offset, size = members[value]
                    data = os.pread(fd, size, offset)
                    if len(data) != size:
                        raise OSError(f"{name}: short read for {value!r}")
                    with Image.open(io.BytesIO(data)) as image:
                        image.load()
    finally:
        if fd is not None:
            os.close(fd)
    summary = ", ".join(f"{task}={count}" for task, count in sorted(task_counts.items()))
    print(
        f"{name}: type=indexed_prepared_tar, rows={rows:,}, checked={len(payloads):,}, "
        f"weight={weight}, tasks=({summary})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate canonical prepared dataset specs.")
    parser.add_argument("dataset_specs", type=Path)
    parser.add_argument(
        "--check-images",
        type=int,
        default=20,
        help="Number of rows per dataset to check for image existence. Use 0 to skip or -1 to check every row.",
    )
    parser.add_argument(
        "--check-decode",
        action="store_true",
        help="Open checked images with Pillow to catch corrupt files.",
    )
    parser.add_argument(
        "--allow-missing-loss-masks",
        action="store_true",
        help="Allow non-geometry rows without loss_mask; geometry requires an explicit per-spec opt-in.",
    )
    args = parser.parse_args()

    specs_path = args.dataset_specs.expanduser().resolve()
    config_root = specs_path.parent
    config = json.loads(specs_path.read_text(encoding="utf-8"))
    samples_per_epoch = config.get("samples_per_epoch")
    if samples_per_epoch is None or int(samples_per_epoch) <= 0:
        raise ValueError("dataset_specs must set a positive samples_per_epoch.")
    datasets = config.get("datasets", [])
    if not datasets:
        raise ValueError("dataset_specs has no datasets.")

    if args.check_decode and Image is None:
        raise RuntimeError("--check-decode requires Pillow in the current Python environment.")

    seen_names = set()
    print(f"samples_per_epoch={samples_per_epoch}")
    for spec in datasets:
        name = spec["name"]
        if name in seen_names:
            raise ValueError(f"duplicate dataset name: {name}")
        seen_names.add(name)
        weight = float(spec.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"{name}: weight must be positive, got {weight}")
        dataset_type = str(spec.get("type", "prepared_folder"))
        if dataset_type == "indexed_prepared_tar":
            validate_indexed_prepared_tar(
                name,
                spec,
                weight,
                config_root,
                args.check_images,
                args.check_decode,
                args.allow_missing_loss_masks,
            )
            continue
        if dataset_type == "webdataset_text_to_image":
            shards = spec.get("shards")
            patterns = [shards] if isinstance(shards, str) else shards
            if not patterns:
                raise ValueError(f"{name}: webdataset_text_to_image requires shards")
            resolved_patterns = [
                str(resolve(config_root, pattern).expanduser()) for pattern in patterns
            ]
            matches = [
                match for pattern in resolved_patterns for match in glob.glob(pattern)
            ]
            if not matches and not any("{" in str(pattern) for pattern in patterns):
                raise FileNotFoundError(f"{name}: no local shards match {shards!r}")
            print(f"{name}: type={dataset_type}, shards={len(set(matches)):,}, weight={weight}")
            continue
        if dataset_type != "prepared_folder":
            raise ValueError(f"{name}: unsupported type={dataset_type!r}")
        root = resolve(config_root, spec["path"]).expanduser()
        metadata = root / "metadata.jsonl"
        if not metadata.exists():
            raise FileNotFoundError(f"{name}: missing {metadata}")

        rows = 0
        missing_required = 0
        missing_images = 0
        task_counts = {}
        for line_number, row in read_jsonl(metadata):
            rows += 1
            task = str(row.get("task") or "unknown").replace("-", "_")
            required_fields = ["image", "text"]
            if task not in {"text_to_image", "text2image", "image_generation"}:
                required_fields.append("conditioning_image")
            for field in required_fields:
                if not row.get(field):
                    missing_required += 1
                    print(f"{name}: line {line_number} missing {field}")
            task_counts[task] = task_counts.get(task, 0) + 1
            should_check_image = args.check_images < 0 or (args.check_images > 0 and rows <= args.check_images)
            if should_check_image:
                for field in ("conditioning_image", "image"):
                    if not row.get(field):
                        continue
                    image_path = resolve(root, row[field])
                    if not image_path.exists():
                        missing_images += 1
                        print(f"{name}: line {line_number} missing {field}: {row[field]}")
                        continue
                    if args.check_decode:
                        try:
                            with Image.open(image_path) as image:
                                ImageOps.exif_transpose(image).convert("RGB")
                        except Exception as exc:
                            missing_images += 1
                            print(f"{name}: line {line_number} cannot decode {field}: {row[field]} ({exc})")

        if rows == 0:
            raise RuntimeError(f"{name}: metadata.jsonl has no rows.")
        if missing_required or missing_images:
            raise RuntimeError(
                f"{name}: validation failed with {missing_required} missing required fields and "
                f"{missing_images} missing checked images."
            )
        task_summary = ", ".join(f"{task}={count}" for task, count in sorted(task_counts.items()))
        print(f"{name}: rows={rows:,}, weight={weight}, tasks=({task_summary})")

    print("dataset_specs validation ok")


if __name__ == "__main__":
    main()
