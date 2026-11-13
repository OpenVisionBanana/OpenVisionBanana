#!/usr/bin/env python3
"""Prepare the official DSINE/GeoNet NYUv2 test asset for OVB evaluation.

The source zip contains the complete DSINE evaluation bundle.  This converter
copies only the 654 NYUv2 test RGB/normal pairs, derives the official validity
mask (encoded normal RGB sum > 0), and writes the existing OVB NYU split-file
layout.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_prep.geometry_conventions import (
    CANONICAL_NORMAL_CONVENTION,
    GEOMETRY_DATA_VERSION,
    IMAGE_Y_DOWN_Z_AWAY_TO_CANONICAL,
)


DSINE_SOURCE_COMMIT = "ef0c2afa32b4dd19cb8ca4567c652802cd92591c"
EXPECTED_SOURCE_SHA256 = "c60b9670e9f47648175f5752d6634a0672b203f5ec5a8b86c9ae07f8653fa8d5"
EXPECTED_TEST_SAMPLES = 654
DEFAULT_FOCAL_LENGTH = 518.8579011745019


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source-sha256",
        default=EXPECTED_SOURCE_SHA256,
        help="Pinned source digest. Pass an empty string only for an intentional unpinned audit.",
    )
    parser.add_argument("--focal-length", type=float, default=DEFAULT_FOCAL_LENGTH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_zip = args.source_zip.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not source_zip.is_file():
        raise FileNotFoundError(source_zip)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {output_root}")

    source_sha256 = sha256_file(source_zip)
    if args.source_sha256 and source_sha256 != args.source_sha256:
        raise ValueError(
            f"DSINE source SHA256 mismatch: expected {args.source_sha256}, got {source_sha256}"
        )

    images_dir = output_root / "images"
    normals_dir = output_root / "normals"
    masks_dir = output_root / "masks"
    for directory in (images_dir, normals_dir, masks_dir):
        directory.mkdir(parents=True, exist_ok=False)

    split_lines: list[str] = []
    sample_manifest: list[dict[str, object]] = []
    with zipfile.ZipFile(source_zip) as archive:
        image_members = sorted(
            name
            for name in archive.namelist()
            if name.startswith("dsine_eval/nyuv2/test/") and name.endswith("_img.png")
        )
        if len(image_members) != EXPECTED_TEST_SAMPLES:
            raise ValueError(
                f"Expected {EXPECTED_TEST_SAMPLES} DSINE NYU test images, got {len(image_members)}"
            )

        for image_member in image_members:
            source_id = int(Path(image_member).name.removesuffix("_img.png"))
            source_prefix = f"dsine_eval/nyuv2/test/{source_id:06d}"
            normal_member = f"{source_prefix}_normal.png"
            if normal_member not in archive.namelist():
                raise FileNotFoundError(normal_member)

            image_bytes = archive.read(image_member)
            normal_bytes = archive.read(normal_member)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            native_normal = Image.open(io.BytesIO(normal_bytes)).convert("RGB")
            if image.size != native_normal.size:
                raise ValueError(
                    f"RGB/normal size mismatch for source ID {source_id}: "
                    f"{image.size} != {native_normal.size}"
                )
            native_normal_rgb = np.asarray(native_normal, dtype=np.uint8)
            valid = np.sum(native_normal_rgb, axis=2, dtype=np.uint16) > 0
            if not np.any(valid):
                raise ValueError(f"No valid normal pixels for source ID {source_id}")
            native_normals = native_normal_rgb.astype(np.float32) / 255.0 * 2.0 - 1.0
            native_norm = np.linalg.norm(native_normals, axis=2, keepdims=True)
            native_normals = np.divide(
                native_normals,
                native_norm,
                out=np.zeros_like(native_normals),
                where=native_norm > 1e-8,
            )
            canonical_normals = IMAGE_Y_DOWN_Z_AWAY_TO_CANONICAL.apply(
                native_normals,
                valid,
            )
            canonical_rgb = np.zeros_like(native_normal_rgb)
            encoded = np.rint(
                (np.clip(canonical_normals, -1.0, 1.0) * 0.5 + 0.5) * 255.0
            ).astype(np.uint8)
            canonical_rgb[valid] = encoded[valid]
            canonical_normal = Image.fromarray(canonical_rgb, mode="RGB")

            stem = f"nyu_{source_id + 1:04d}"
            image_rel = Path("images") / f"{stem}_rgb.png"
            normal_rel = Path("normals") / f"{stem}_normal.png"
            mask_rel = Path("masks") / f"{stem}_mask.png"
            image.save(output_root / image_rel, format="PNG")
            canonical_normal.save(output_root / normal_rel, format="PNG")
            Image.fromarray(valid.astype(np.uint8) * 255, mode="L").save(
                output_root / mask_rel,
                format="PNG",
            )
            split_lines.append(
                f"/{image_rel} /{normal_rel} /{mask_rel} "
                f"{args.focal_length:.10f} {source_id + 1}"
            )
            sample_manifest.append(
                {
                    "source_id_0based": source_id,
                    "source_index_1based": source_id + 1,
                    "image_size_wh": list(image.size),
                    "valid_pixels": int(valid.sum()),
                    "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                    "source_native_normal_sha256": hashlib.sha256(normal_bytes).hexdigest(),
                    "prepared_canonical_normal_sha256": hashlib.sha256(
                        canonical_normal.tobytes()
                    ).hexdigest(),
                }
            )

    (output_root / "nyu_test.txt").write_text(
        "\n".join(split_lines) + "\n",
        encoding="utf-8",
    )
    info = {
        "dataset": "nyu_surface_normals",
        "geometry_version": GEOMETRY_DATA_VERSION,
        "split": "test",
        "count": len(sample_manifest),
        "ground_truth_source": "DSINE evaluation bundle; NYUv2 normals from GeoNet",
        "source_zip": str(source_zip),
        "source_zip_sha256": source_sha256,
        "source_code_repository": "https://github.com/baegwangbin/DSINE",
        "source_code_commit": DSINE_SOURCE_COMMIT,
        "source_normal_decode": "(rgb / 255) * 2 - 1, then unit normalization",
        "prepared_normal_encoding": (
            "rgb = round((canonical_normal_xyz * 0.5 + 0.5) * 255); "
            "invalid pixels remain black and are excluded by the explicit mask"
        ),
        "validity_mask": "sum(normal_rgb, axis=channels) > 0",
        "normal_convention": {
            "source_native_coordinates": "camera space",
            "source_native_channel_order": ["x", "y", "z"],
            "source_native_axis_directions": [
                "image right",
                "image down",
                "away from camera; labels predominantly +z",
            ],
            "prepared_convention": CANONICAL_NORMAL_CONVENTION,
            "transform_to_ovb_canonical": {
                "permutation": [0, 1, 2],
                "signs": [-1, 1, 1],
            },
            "evidence": (
                "DSINE native labels matched the legacy x-right/y-down/z-away, "
                "z-positive normal field under a matched-ID 48-convention audit "
                "(identity best at 25.06 degrees; next candidate 47.88 degrees on "
                "50 images). The established tilted-plane native-to-canonical audit "
                "then requires (-x,+y,+z); see depth_derived_normal_transform.json."
            ),
        },
        "aggregation_note": (
            "DSINE benchmark-standard reporting pools valid pixels globally. "
            "Per-image diagnostic reporting must remain separate from globally pooled metrics."
        ),
        "license_note": (
            "The DSINE README instructs users to inspect each underlying dataset license. "
            "This prepared asset does not grant or replace NYUv2/GeoNet dataset terms."
        ),
        "samples": sample_manifest,
    }
    (output_root / "dataset_info.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: info[key] for key in ("dataset", "split", "count", "source_zip_sha256")}))


if __name__ == "__main__":
    main()
