#!/usr/bin/env python3
"""Audit DIODE depth quantity and official-normal axes from paired GT.

The audit treats the official normal map as the unknown field.  For both
possible depth interpretations (projective camera-z and Euclidean camera
range), it derives a canonical camera-facing normal reference from depth and
scores all 48 axis permutation/sign transforms of the official normals.
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import tarfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image

from dataset_prep.geometry_conventions import (
    IMAGE_Y_DOWN_Z_AWAY_TO_CANONICAL,
    range_to_projective_z,
)


def read_bytes(tar: tarfile.TarFile, member: str) -> bytes:
    handle = tar.extractfile(member)
    if handle is None:
        raise FileNotFoundError(f"Missing tar member: {member}")
    return handle.read()


def read_jsonl(tar: tarfile.TarFile, member: str) -> list[dict]:
    return [
        json.loads(line)
        for line in read_bytes(tar, member).decode("utf-8").splitlines()
        if line.strip()
    ]


def decode_png(data: bytes, mode: str | None = None) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        if mode is not None:
            image = image.convert(mode)
        return np.asarray(image)


def derive_canonical_normals(
    depth: np.ndarray,
    valid: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    radius: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    u = np.arange(width, dtype=np.float32)[None, :]
    v = np.arange(height, dtype=np.float32)[:, None]
    z = np.asarray(depth, dtype=np.float32)
    points = np.stack(
        (
            (u - cx) * z / fx,
            (v - cy) * z / fy,
            z,
        ),
        axis=-1,
    )
    normals = np.zeros((height, width, 3), dtype=np.float32)
    score_mask = np.zeros((height, width), dtype=bool)
    ys = slice(radius, height - radius)
    xs = slice(radius, width - radius)
    dx = points[ys, 2 * radius :] - points[ys, : width - 2 * radius]
    dy = points[2 * radius :, xs] - points[: height - 2 * radius, xs]
    native = np.cross(dx, dy)
    norm = np.linalg.norm(native, axis=-1)
    native = np.divide(
        native,
        norm[..., None],
        out=np.zeros_like(native),
        where=np.isfinite(norm[..., None]) & (norm[..., None] > 1e-8),
    )
    native[native[..., 2] < 0] *= -1
    normals[ys, xs] = native
    neighbor_valid = (
        valid[ys, 2 * radius :]
        & valid[ys, : width - 2 * radius]
        & valid[2 * radius :, xs]
        & valid[: height - 2 * radius, xs]
    )
    score_mask[ys, xs] = neighbor_valid & np.isfinite(norm) & (norm > 1e-8)
    return IMAGE_Y_DOWN_Z_AWAY_TO_CANONICAL.apply(normals, score_mask), score_mask


def transformations() -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    return [
        (permutation, signs)
        for permutation in itertools.permutations(range(3))
        for signs in itertools.product((1, -1), repeat=3)
    ]


def choose_rows(rows: list[dict], per_case: int) -> list[dict]:
    selected: list[dict] = []
    for case in ("indoor", "outdoor"):
        case_rows = [row for row in rows if row["case"] == case]
        indices = np.linspace(
            0, len(case_rows) - 1, min(per_case, len(case_rows)), dtype=int
        )
        selected.extend(case_rows[int(index)] for index in indices)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-archive", type=Path, required=True)
    parser.add_argument("--normal-archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples-per-case", type=int, default=8)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--fx", type=float, default=886.81)
    parser.add_argument("--fy", type=float, default=927.06)
    parser.add_argument("--cx", type=float, default=512.0)
    parser.add_argument("--cy", type=float, default=384.0)
    args = parser.parse_args()

    transforms = transformations()
    angle_sums = {
        quantity: np.zeros(len(transforms), dtype=np.float64)
        for quantity in ("projective_camera_z", "euclidean_camera_range")
    }
    pixel_counts = {quantity: 0 for quantity in angle_sums}
    sample_records: list[dict] = []

    with tarfile.open(args.depth_archive, "r:") as depth_tar, tarfile.open(
        args.normal_archive, "r:"
    ) as normal_tar:
        depth_rows = read_jsonl(depth_tar, "diode_depth/metadata.jsonl")
        normal_rows = {
            row["id"]: row
            for row in read_jsonl(
                normal_tar, "diode_surface_normals/metadata.jsonl"
            )
        }
        selected = choose_rows(depth_rows, args.samples_per_case)

        for depth_row in selected:
            sample_id = depth_row["id"]
            normal_row = normal_rows[sample_id]
            depth = np.load(
                io.BytesIO(
                    read_bytes(
                        depth_tar, f"diode_depth/{depth_row['depth_raw']}"
                    )
                )
            ).astype(np.float32)
            depth_mask = (
                decode_png(
                    read_bytes(depth_tar, f"diode_depth/{depth_row['mask']}"),
                    "L",
                )
                > 0
            )
            normal_rgb = decode_png(
                read_bytes(
                    normal_tar,
                    f"diode_surface_normals/{normal_row['image']}",
                ),
                "RGB",
            )
            normal_mask = (
                decode_png(
                    read_bytes(
                        normal_tar,
                        f"diode_surface_normals/{normal_row['mask']}",
                    ),
                    "L",
                )
                > 0
            )
            native = normal_rgb.astype(np.float32) / 127.5 - 1.0
            native_norm = np.linalg.norm(native, axis=-1)
            native = np.divide(
                native,
                native_norm[..., None],
                out=np.zeros_like(native),
                where=native_norm[..., None] > 1e-8,
            )

            sample_record = {"id": sample_id, "case": depth_row["case"]}
            for quantity in angle_sums:
                z = (
                    depth
                    if quantity == "projective_camera_z"
                    else range_to_projective_z(
                        depth,
                        fx=args.fx,
                        fy=args.fy,
                        cx=args.cx,
                        cy=args.cy,
                    )
                )
                reference, reference_valid = derive_canonical_normals(
                    z,
                    depth_mask,
                    fx=args.fx,
                    fy=args.fy,
                    cx=args.cx,
                    cy=args.cy,
                )
                valid = reference_valid & normal_mask
                valid[:: args.pixel_stride, :: args.pixel_stride] &= True
                stride_mask = np.zeros_like(valid)
                stride_mask[:: args.pixel_stride, :: args.pixel_stride] = True
                valid &= stride_mask
                ref = reference[valid]
                raw = native[valid]
                if ref.shape[0] == 0:
                    raise RuntimeError(f"No jointly valid pixels for {sample_id}")
                for index, (permutation, signs) in enumerate(transforms):
                    transformed = raw[:, permutation] * np.asarray(
                        signs, dtype=np.float32
                    )
                    dots = np.clip(np.sum(transformed * ref, axis=-1), -1.0, 1.0)
                    angle_sums[quantity][index] += float(
                        np.degrees(np.arccos(dots)).sum(dtype=np.float64)
                    )
                pixel_counts[quantity] += int(ref.shape[0])
                sample_record[f"{quantity}_pixels"] = int(ref.shape[0])
            sample_records.append(sample_record)

    results = {}
    for quantity, sums in angle_sums.items():
        means = sums / pixel_counts[quantity]
        order = np.argsort(means)
        best_index = int(order[0])
        runner_up_index = int(order[1])
        best = transforms[best_index]
        runner_up = transforms[runner_up_index]
        results[quantity] = {
            "pixels": pixel_counts[quantity],
            "best": {
                "permutation": list(best[0]),
                "signs": list(best[1]),
                "mean_angle_degrees": float(means[best_index]),
            },
            "runner_up": {
                "permutation": list(runner_up[0]),
                "signs": list(runner_up[1]),
                "mean_angle_degrees": float(means[runner_up_index]),
            },
            "margin_degrees": float(means[runner_up_index] - means[best_index]),
        }

    best_quantity = min(
        results,
        key=lambda quantity: results[quantity]["best"]["mean_angle_degrees"],
    )
    other_quantity = next(
        quantity for quantity in results if quantity != best_quantity
    )
    quantity_margin = (
        results[other_quantity]["best"]["mean_angle_degrees"]
        - results[best_quantity]["best"]["mean_angle_degrees"]
    )
    selected_best = results[best_quantity]["best"]
    selected_transform_is_identity = (
        selected_best["permutation"] == [0, 1, 2]
        and selected_best["signs"] == [1, 1, 1]
    )
    report = {
        "schema_version": 1,
        "status": "pass"
        if selected_best["mean_angle_degrees"] < 35.0
        and results[best_quantity]["margin_degrees"] > 10.0
        and quantity_margin > 3.0
        and selected_transform_is_identity
        else "review_required",
        "depth_archive": str(args.depth_archive.resolve()),
        "normal_archive": str(args.normal_archive.resolve()),
        "samples_per_case": args.samples_per_case,
        "pixel_stride": args.pixel_stride,
        "intrinsics": {
            "fx": args.fx,
            "fy": args.fy,
            "cx": args.cx,
            "cy": args.cy,
        },
        "selected_depth_quantity": best_quantity,
        "selected_normal_transform_is_identity": selected_transform_is_identity,
        "depth_quantity_margin_degrees": float(quantity_margin),
        "acceptance": {
            "mean_angle_below_degrees": 35.0,
            "normal_transform_margin_above_degrees": 10.0,
            "depth_quantity_margin_above_degrees": 3.0,
            "require_identity_normal_transform": True,
            "note": (
                "The absolute paired angle includes differentiation of sparse/noisy "
                "laser depth; transform and depth-quantity margins are the convention "
                "identification criteria."
            ),
        },
        "results": results,
        "samples": sample_records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
