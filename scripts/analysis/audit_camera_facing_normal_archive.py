#!/usr/bin/env python3
"""Fail-closed audit for camera-facing surface-normal training TARs."""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from common.indexed_tar import validate_archive_fingerprint
from common.surface_normal_encoding import (
    VISION_BANANA_NORMAL_ENCODING,
    decode_surface_normals_rgb,
)
from dataset_prep.geometry_conventions import (
    CAMERA_FACING_GEOMETRY_DATA_VERSION,
    CANONICAL_NORMAL_CONVENTION,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-ids-from", type=Path)
    parser.add_argument("--forbidden-scenes", type=Path)
    parser.add_argument("--sample-limit", type=int, default=512)
    return parser.parse_args()


def member_bytes(handle: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    stream = handle.extractfile(member)
    if stream is None:
        raise FileNotFoundError(member.name)
    return stream.read()


def archive_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    with tarfile.open(path, "r:") as handle:
        members = {item.name: item for item in handle.getmembers()}
        metadata_names = sorted(name for name in members if name.endswith("/metadata.jsonl"))
        info_names = sorted(name for name in members if name.endswith("/dataset_info.json"))
        if len(metadata_names) != 1 or len(info_names) != 1:
            raise ValueError("archive must contain exactly one metadata.jsonl and dataset_info.json")
        metadata_name = metadata_names[0]
        root = metadata_name.rsplit("/", 1)[0]
        rows = [
            json.loads(line)
            for line in member_bytes(handle, members[metadata_name]).decode("utf-8").splitlines()
            if line.strip()
        ]
        info = json.loads(member_bytes(handle, members[info_names[0]]).decode("utf-8"))
    return info, rows, root


def evenly_spaced(count: int, limit: int) -> list[int]:
    if count <= limit:
        return list(range(count))
    return sorted({int(round(value)) for value in np.linspace(0, count - 1, limit)})


def row_scene(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata") or {}
    value = metadata.get("scene") or row.get("scene")
    return str(value) if value is not None else None


def validate_contract(info: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    expected = {
        "geometry_version": CAMERA_FACING_GEOMETRY_DATA_VERSION,
        "normal_orientation": "face_camera_ray",
        "stored_normal_convention": CANONICAL_NORMAL_CONVENTION,
        "normal_encoding": VISION_BANANA_NORMAL_ENCODING,
    }
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(f"dataset_info {key}={info.get(key)!r}, expected {value!r}")
    if info.get("gt_to_canonical_permutation") != [0, 1, 2]:
        raise ValueError("dataset_info permutation is not identity")
    if info.get("gt_to_canonical_signs") != [1, 1, 1]:
        raise ValueError("dataset_info signs are not identity")
    if info.get("loss_masks_required") is not True:
        raise ValueError("dataset_info does not require loss masks")
    ids = [str(row.get("id", "")).strip() for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("metadata IDs are empty or duplicated")


def main() -> None:
    args = parse_args()
    archive = args.archive.expanduser().resolve()
    index = args.index.expanduser().resolve()
    with sqlite3.connect(f"file:{index}?mode=ro", uri=True) as connection:
        index_metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        indexed_rows = int(connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0])
    validate_archive_fingerprint(
        archive,
        json.loads(index_metadata["archive_fingerprint"]),
    )
    info, rows, root = archive_rows(archive)
    validate_contract(info, rows)
    if indexed_rows != len(rows):
        raise ValueError(f"indexed row count {indexed_rows} != metadata row count {len(rows)}")
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise ValueError(f"row count {len(rows)} != expected {args.expected_rows}")

    membership: dict[str, Any] = {"checked": False}
    if args.expected_ids_from:
        _, old_rows, _ = archive_rows(args.expected_ids_from.expanduser().resolve())
        current_ids = {str(row["id"]) for row in rows}
        old_ids = {str(row["id"]) for row in old_rows}
        missing = sorted(old_ids - current_ids)
        added = sorted(current_ids - old_ids)
        membership = {
            "checked": True,
            "reference": str(args.expected_ids_from.expanduser().resolve()),
            "missing_count": len(missing),
            "added_count": len(added),
            "missing_first": missing[:20],
            "added_first": added[:20],
        }
        if missing or added:
            raise ValueError(
                f"membership differs: missing={len(missing)}, added={len(added)}"
            )

    leakage: dict[str, Any] = {"checked": False}
    if args.forbidden_scenes:
        forbidden = {
            line.strip()
            for line in args.forbidden_scenes.expanduser().read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
        overlap = sorted({scene for row in rows if (scene := row_scene(row)) in forbidden})
        leakage = {
            "checked": True,
            "forbidden_scene_count": len(forbidden),
            "overlap": overlap,
        }
        if overlap:
            raise ValueError(f"forbidden scenes in training archive: {overlap[:20]}")

    sample_indices = evenly_spaced(len(rows), max(1, args.sample_limit))
    coverages: list[float] = []
    with tarfile.open(archive, "r:") as handle:
        members = {item.name: item for item in handle.getmembers()}
        for index_value in sample_indices:
            row = rows[index_value]
            for key in ("conditioning_image", "image", "loss_mask"):
                if not row.get(key):
                    raise ValueError(f"row {index_value}: missing {key}")
            target_name = f"{root}/{row['image']}"
            mask_name = f"{root}/{row['loss_mask']}"
            condition_name = f"{root}/{row['conditioning_image']}"
            for name in (target_name, mask_name, condition_name):
                if name not in members:
                    raise FileNotFoundError(name)
            with Image.open(io.BytesIO(member_bytes(handle, members[target_name]))) as image:
                target = np.asarray(image.convert("RGB"))
            with Image.open(io.BytesIO(member_bytes(handle, members[mask_name]))) as image:
                mask = np.asarray(image.convert("L")) > 0
            with Image.open(io.BytesIO(member_bytes(handle, members[condition_name]))) as image:
                condition_size = image.size
            if target.shape[:2] != mask.shape or condition_size != (target.shape[1], target.shape[0]):
                raise ValueError(f"row {index_value}: conditioning/target/mask shape mismatch")
            normals, encoded_valid = decode_surface_normals_rgb(
                target, encoding=VISION_BANANA_NORMAL_ENCODING
            )
            valid = mask & encoded_valid & np.isfinite(normals).all(axis=-1)
            if not np.any(valid):
                raise ValueError(f"row {index_value}: zero valid pixels")
            lengths = np.linalg.norm(normals[valid], axis=-1)
            if float(np.max(np.abs(lengths - 1.0))) > 1e-4:
                raise ValueError(f"row {index_value}: decoded normals are not unit length")
            metadata = row.get("metadata") or {}
            if metadata.get("geometry_version") != CAMERA_FACING_GEOMETRY_DATA_VERSION:
                raise ValueError(f"row {index_value}: incorrect geometry version")
            if metadata.get("normal_orientation") != "face_camera_ray":
                raise ValueError(f"row {index_value}: incorrect normal orientation")
            if metadata.get("normal_axis_permutation") != [0, 1, 2]:
                raise ValueError(f"row {index_value}: non-identity permutation")
            if metadata.get("normal_axis_signs") != [1, 1, 1]:
                raise ValueError(f"row {index_value}: non-identity signs")
            coverages.append(float(valid.mean()))

    report = {
        "status": "PASS",
        "archive": str(archive),
        "index": str(index),
        "rows": len(rows),
        "sampled_rows": len(sample_indices),
        "valid_fraction_mean": float(np.mean(coverages)),
        "valid_fraction_min": float(np.min(coverages)),
        "membership": membership,
        "leakage": leakage,
        "contract": {
            "geometry_version": CAMERA_FACING_GEOMETRY_DATA_VERSION,
            "normal_orientation": "face_camera_ray",
            "stored_normal_convention": CANONICAL_NORMAL_CONVENTION,
            "normal_encoding": VISION_BANANA_NORMAL_ENCODING,
        },
    }
    output = args.report.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
