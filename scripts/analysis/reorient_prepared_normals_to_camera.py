#!/usr/bin/env python3
"""Reorient prepared canonical normal PNGs to face the camera ray.

This creates a new diagnostic archive and never modifies the source archive.
It is intended for auditing depth-derived targets whose old preparation used a
positive-z heuristic, which is ambiguous on surfaces nearly tangent to z.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from common.surface_normal_encoding import (
    VISION_BANANA_NORMAL_ENCODING,
    decode_surface_normals_rgb,
    encode_surface_normals_rgb,
    normal_encoding_formula,
)
from dataset_prep.geometry_conventions import (
    CAMERA_FACING_GEOMETRY_DATA_VERSION,
    CANONICAL_NORMAL_CONVENTION,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoding", default=VISION_BANANA_NORMAL_ENCODING)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="normal-facing-audit-") as tmp_name:
        tmp = Path(tmp_name)
        with tarfile.open(args.input, "r:*") as source:
            source.extractall(tmp, filter="data")
        root = tmp / args.dataset_root
        split = root / args.split_file
        total_valid = 0
        total_flipped = 0
        per_sample = []
        for line in split.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"Expected image/normal/mask/intrinsics fields: {line}")
            normal_path = root / fields[1].lstrip("/")
            mask_path = root / fields[2].lstrip("/")
            intrinsics_path = root / fields[3].lstrip("/")
            rgb = np.asarray(Image.open(normal_path).convert("RGB"))
            normals, decoded_valid = decode_surface_normals_rgb(
                rgb, encoding=args.encoding, min_norm=0.01
            )
            mask = np.asarray(Image.open(mask_path).convert("L")) > 0
            valid = mask & decoded_valid
            intrinsics = np.loadtxt(intrinsics_path, dtype=np.float64).reshape(3, 3)
            height, width = valid.shape
            u = np.arange(width, dtype=np.float32)[None, :]
            v = np.arange(height, dtype=np.float32)[:, None]
            ray = np.empty((height, width, 3), dtype=np.float32)
            ray[..., 0] = (u - float(intrinsics[0, 2])) / float(intrinsics[0, 0])
            ray[..., 1] = -(v - float(intrinsics[1, 2])) / float(intrinsics[1, 1])
            ray[..., 2] = -1.0
            flip = valid & (np.sum(normals * ray, axis=-1) > 0)
            normals[flip] *= -1.0
            Image.fromarray(
                encode_surface_normals_rgb(
                    normals,
                    valid,
                    encoding=args.encoding,
                    invalid_rgb=(128, 128, 128),
                ),
                mode="RGB",
            ).save(normal_path)
            valid_count = int(valid.sum())
            flipped_count = int(flip.sum())
            total_valid += valid_count
            total_flipped += flipped_count
            per_sample.append(
                {
                    "normal": fields[1],
                    "valid_pixels": valid_count,
                    "flipped_pixels": flipped_count,
                    "flipped_fraction": flipped_count / max(valid_count, 1),
                }
            )

        audit = {
            "source_archive": str(args.input),
            "method": "flip canonical normal when dot(normal, camera_to_surface_ray) > 0",
            "encoding": args.encoding,
            "valid_pixels": total_valid,
            "flipped_pixels": total_flipped,
            "flipped_fraction": total_flipped / max(total_valid, 1),
            "per_sample": per_sample,
        }
        (root / "camera_facing_reorientation.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
        info_path = root / "dataset_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
        info.update(
            {
                "normal_orientation": "face_camera_ray",
                "normal_orientation_audit": "camera_facing_reorientation.json",
                "diagnostic_reorientation_only": True,
                "normal_encoding": args.encoding,
                "normal_encoding_formula": normal_encoding_formula(args.encoding),
                "geometry_version": CAMERA_FACING_GEOMETRY_DATA_VERSION,
                "stored_normal_convention": CANONICAL_NORMAL_CONVENTION,
                "gt_to_canonical_permutation": [0, 1, 2],
                "gt_to_canonical_signs": [1, 1, 1],
            }
        )
        info_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = args.output.with_suffix(args.output.suffix + ".partial")
        with tarfile.open(partial, "w") as output:
            output.add(root, arcname=root.name)
        with tarfile.open(partial, "r:") as check:
            check.getmembers()
        partial.replace(args.output)
        print(json.dumps({key: value for key, value in audit.items() if key != "per_sample"}, indent=2))


if __name__ == "__main__":
    main()
