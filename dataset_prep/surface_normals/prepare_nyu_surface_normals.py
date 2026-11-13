#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.surface_normal_encoding import (
    LEGACY_DIRECT_NORMAL_ENCODING,
    normal_encoding_formula,
)
from dataset_prep.geometry_conventions import (
    CAMERA_FACING_GEOMETRY_DATA_VERSION,
    CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL,
    CANONICAL_NORMAL_CONVENTION,
    orient_normals_toward_camera,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda iterable, **_: iterable


NORMAL_PROMPT = (
    "[TASK: SURFACE_NORMAL_ESTIMATION] Generate a dense camera-space surface normal map "
    "of the input indoor scene. Output only a clean RGB normal map with no text, labels, "
    "legends, or extra objects. Encode normal vector components x, y, z in red, green, "
    "blue by mapping [-1, 1] to [0, 255]; front-facing surfaces should be mostly blue. "
    "Keep object and geometry boundaries sharp while keeping planar surfaces smooth."
)

DEFAULT_LABELED_URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
DEFAULT_SPLITS_URL = "http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat"

# NYUv2 RGB camera intrinsics from camera_params.m in the official toolbox.
DEFAULT_FX = 518.8579011745019
DEFAULT_FY = 519.4696111212749
DEFAULT_CX = 325.58244941119034
DEFAULT_CY = 253.73616633400465


@dataclass(frozen=True)
class SampleRecord:
    index_1based: int
    split: str
    scene: str
    scene_type: str
    raw_rgb_filename: str
    raw_depth_filename: str
    image_rel: Path
    normal_rel: Path
    mask_rel: Path


def require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "ERROR: h5py is required to read nyu_depth_v2_labeled.mat. "
            "Install h5py or build the Singularity definition that includes it."
        ) from exc
    return h5py


def read_matlab_char_ref(handle: Any, ref: Any) -> str:
    if ref is None:
        return ""
    try:
        obj = handle[ref]
    except Exception:
        return ""
    arr = np.asarray(obj[()])
    chars = []
    for value in arr.flat:
        code = int(value)
        if code != 0:
            chars.append(chr(code))
    return "".join(chars)


def read_string_vector(handle: Any, key: str, count: int) -> list[str]:
    if key not in handle:
        return [""] * count
    refs = handle[key][0]
    return [read_matlab_char_ref(handle, refs[i]) for i in range(min(count, len(refs)))] + [
        ""
    ] * max(0, count - len(refs))


def as_rgb_image(image_cwh: np.ndarray) -> Image.Image:
    if image_cwh.ndim != 3 or image_cwh.shape[0] != 3:
        raise ValueError(f"Expected NYU image with shape (3, W, H), got {image_cwh.shape}")
    rgb = np.transpose(image_cwh, (2, 1, 0))
    return Image.fromarray(rgb.astype(np.uint8, copy=False), mode="RGB")


def as_depth_hw(depth_wh: np.ndarray) -> np.ndarray:
    if depth_wh.ndim != 2:
        raise ValueError(f"Expected NYU depth with shape (W, H), got {depth_wh.shape}")
    return np.asarray(depth_wh, dtype=np.float32).T


def load_split_indices(splits_mat: Path) -> dict[str, list[int]]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "ERROR: scipy is required to read splits.mat. Install scipy, build the "
            "NYU-normal Singularity definition, or use --split-policy sequential."
        ) from exc

    split = loadmat(splits_mat)
    if "trainNdxs" not in split or "testNdxs" not in split:
        raise KeyError(f"Expected trainNdxs and testNdxs in {splits_mat}")
    return {
        "train": [int(value) - 1 for value in np.asarray(split["trainNdxs"]).reshape(-1)],
        "test": [int(value) - 1 for value in np.asarray(split["testNdxs"]).reshape(-1)],
    }


def build_indices(count: int, splits_mat: Path | None, split_policy: str) -> dict[str, list[int]]:
    if splits_mat is not None:
        indices = load_split_indices(splits_mat.expanduser().resolve())
    elif split_policy == "sequential":
        if count < 1449:
            raise ValueError(f"Sequential NYUv2 split expects at least 1449 samples, got {count}")
        indices = {"train": list(range(795)), "test": list(range(795, 1449))}
    else:
        raise ValueError(
            "Missing --splits-mat. Download the official splits.mat or pass "
            "--split-policy sequential for the simple 795/654 fallback."
        )

    seen = set()
    for split_name, values in indices.items():
        for index in values:
            if index < 0 or index >= count:
                raise ValueError(f"{split_name} index {index + 1} is out of range for {count} samples")
            if index in seen:
                raise ValueError(f"Index {index + 1} appears in more than one split")
            seen.add(index)
    return indices


def make_point_cloud(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    height, width = depth.shape
    u = np.arange(width, dtype=np.float32)[None, :]
    v = np.arange(height, dtype=np.float32)[:, None]
    z = depth.astype(np.float32, copy=False)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack((x, y, z), axis=-1)


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool, copy=True)

    eroded = mask.astype(bool, copy=True)
    height, width = eroded.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.zeros_like(mask, dtype=bool)
            src_y0 = max(0, -dy)
            src_y1 = min(height, height - dy)
            src_x0 = max(0, -dx)
            src_x1 = min(width, width - dx)
            dst_y0 = max(0, dy)
            dst_y1 = min(height, height + dy)
            dst_x0 = max(0, dx)
            dst_x1 = min(width, width + dx)
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
            eroded &= shifted
    return eroded


def compute_normals(
    depth: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    if radius < 1:
        raise ValueError("--gradient-radius must be >= 1")

    points = make_point_cloud(depth, fx=fx, fy=fy, cx=cx, cy=cy)
    height, width = depth.shape
    normals = np.zeros((height, width, 3), dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)

    y_slice = slice(radius, height - radius)
    x_slice = slice(radius, width - radius)
    if height <= 2 * radius or width <= 2 * radius:
        return normals, np.zeros((height, width), dtype=bool)

    dx = points[y_slice, slice(2 * radius, width)] - points[y_slice, slice(0, width - 2 * radius)]
    dy = points[slice(2 * radius, height), x_slice] - points[slice(0, height - 2 * radius), x_slice]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    good = np.isfinite(norm[..., 0]) & (norm[..., 0] > 1e-8)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 1e-8)
    n = orient_normals_toward_camera(n, points[y_slice, x_slice], good)
    n = CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL.apply(n, good)

    normals[y_slice, x_slice] = n

    neighbor_valid = (
        valid[y_slice, slice(2 * radius, width)]
        & valid[y_slice, slice(0, width - 2 * radius)]
        & valid[slice(2 * radius, height), x_slice]
        & valid[slice(0, height - 2 * radius), x_slice]
    )
    normal_valid = np.zeros((height, width), dtype=bool)
    normal_valid[y_slice, x_slice] = good & neighbor_valid
    return normals, normal_valid


def normals_to_rgb(normals: np.ndarray, valid: np.ndarray, invalid_rgb: tuple[int, int, int]) -> np.ndarray:
    rgb = np.zeros((*normals.shape[:2], 3), dtype=np.uint8)
    rgb[...] = np.asarray(invalid_rgb, dtype=np.uint8)
    encoded = np.rint((np.clip(normals, -1.0, 1.0) * 0.5 + 0.5) * 255.0)
    rgb[valid] = np.clip(encoded[valid], 0, 255).astype(np.uint8)
    return rgb


def mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255)


def depth_to_uint16_mm(depth_m: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros(depth_m.shape, dtype=np.uint16)
    good = valid & np.isfinite(depth_m) & (depth_m > 0)
    out[good] = np.clip(np.rint(depth_m[good] * 1000.0), 0, 65535).astype(np.uint16)
    return out


def selected_splits(all_splits: dict[str, list[int]], split: str) -> dict[str, list[int]]:
    if split == "all":
        return all_splits
    if split not in all_splits:
        raise ValueError(f"Unsupported split: {split}")
    return {split: all_splits[split]}


def maybe_limited(values: Iterable[int], limit: int | None) -> list[int]:
    values = list(values)
    if limit is None:
        return values
    return values[:limit]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def write_split_file(out_root: Path, split: str, records: list[SampleRecord], focal_length: float) -> None:
    split_path = out_root / f"nyu_{split}.txt"
    with split_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                f"/{record.image_rel.as_posix()} "
                f"/{record.normal_rel.as_posix()} "
                f"/{record.mask_rel.as_posix()} "
                f"{focal_length:.10f} "
                f"{record.index_1based}\n"
            )


def write_prepared_tar(out_root: Path, tar_path: Path) -> None:
    tar_path = tar_path.expanduser().resolve()
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = tar_path.with_suffix(tar_path.suffix + ".partial")
    if tmp_path.exists():
        tmp_path.unlink()
    with tarfile.open(tmp_path, "w") as archive:
        archive.add(out_root, arcname=out_root.name)
    tmp_path.replace(tar_path)


def parse_rgb(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("RGB value must have the form R,G,B")
    rgb = tuple(int(part) for part in parts)
    if any(part < 0 or part > 255 for part in rgb):
        raise argparse.ArgumentTypeError("RGB channels must be in [0, 255]")
    return rgb


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an NYUv2 RGB-to-surface-normal evaluation dataset from nyu_depth_v2_labeled.mat."
    )
    parser.add_argument("--mat-path", type=Path, required=True, help=f"Downloaded labeled MAT file. URL: {DEFAULT_LABELED_URL}")
    parser.add_argument("--splits-mat", type=Path, default=None, help=f"Official splits.mat. URL: {DEFAULT_SPLITS_URL}")
    parser.add_argument(
        "--split-policy",
        choices=("require_splits", "sequential"),
        default="require_splits",
        help="Fallback when --splits-mat is omitted. sequential uses first 795 train and last 654 test.",
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--tar-output", type=Path, default=None, help="Optional tar containing the generated out-root folder.")
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-split sample limit for quick checks.")
    parser.add_argument("--depth-source", choices=("depths", "rawDepths"), default="depths")
    parser.add_argument("--mask-source", choices=("rawDepths", "depths", "both"), default="rawDepths")
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--gradient-radius", type=int, default=2)
    parser.add_argument("--mask-erosion-radius", type=int, default=1)
    parser.add_argument("--fx", type=float, default=DEFAULT_FX)
    parser.add_argument("--fy", type=float, default=DEFAULT_FY)
    parser.add_argument("--cx", type=float, default=DEFAULT_CX)
    parser.add_argument("--cy", type=float, default=DEFAULT_CY)
    parser.add_argument("--prompt", default=NORMAL_PROMPT)
    parser.add_argument("--invalid-normal-rgb", type=parse_rgb, default=(128, 128, 128))
    parser.add_argument("--write-depth-mm", action="store_true", help="Also write metric depth PNGs for debugging/provenance.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty output directory.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.min_depth < 0 or args.max_depth <= args.min_depth:
        raise ValueError("--max-depth must be greater than --min-depth")

    mat_path = args.mat_path.expanduser().resolve()
    if not mat_path.is_file():
        raise FileNotFoundError(f"Missing NYUv2 labeled MAT file: {mat_path}")
    splits_mat = args.splits_mat.expanduser().resolve() if args.splits_mat is not None else None
    if splits_mat is not None and not splits_mat.is_file():
        raise FileNotFoundError(f"Missing NYUv2 splits MAT file: {splits_mat}")

    out_root = args.out_root.expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {out_root}. Use --overwrite to rebuild.")
    for rel in ("images", "normals", "masks"):
        (out_root / rel).mkdir(parents=True, exist_ok=True)
    if args.write_depth_mm:
        (out_root / "depth_mm").mkdir(parents=True, exist_ok=True)

    h5py = require_h5py()
    records_by_split: dict[str, list[SampleRecord]] = {}

    with h5py.File(mat_path, "r") as handle:
        images = handle["images"]
        depths = handle[args.depth_source]
        raw_depths = handle["rawDepths"]
        count = int(images.shape[0])
        all_indices = selected_splits(build_indices(count, splits_mat, args.split_policy), args.split)

        scenes = read_string_vector(handle, "scenes", count)
        scene_types = read_string_vector(handle, "sceneTypes", count)
        raw_rgb_filenames = read_string_vector(handle, "rawRgbFilenames", count)
        raw_depth_filenames = read_string_vector(handle, "rawDepthFilenames", count)

        for split_name, split_indices in all_indices.items():
            records: list[SampleRecord] = []
            for index in tqdm(maybe_limited(split_indices, args.limit), desc=f"Preparing NYU normals {split_name}"):
                index_1based = index + 1
                stem = f"nyu_{index_1based:04d}"
                image_rel = Path("images") / f"{stem}_rgb.png"
                normal_rel = Path("normals") / f"{stem}_normal.png"
                mask_rel = Path("masks") / f"{stem}_mask.png"

                rgb = as_rgb_image(np.asarray(images[index]))
                depth = as_depth_hw(np.asarray(depths[index]))
                raw_depth = as_depth_hw(np.asarray(raw_depths[index]))

                normals, normal_valid = compute_normals(
                    depth,
                    fx=args.fx,
                    fy=args.fy,
                    cx=args.cx,
                    cy=args.cy,
                    radius=args.gradient_radius,
                )
                depth_valid = np.isfinite(depth) & (depth > args.min_depth) & (depth < args.max_depth)
                raw_valid = np.isfinite(raw_depth) & (raw_depth > args.min_depth) & (raw_depth < args.max_depth)
                if args.mask_source == "rawDepths":
                    mask = raw_valid
                elif args.mask_source == "depths":
                    mask = depth_valid
                else:
                    mask = raw_valid & depth_valid
                mask = erode_mask(mask & normal_valid, args.mask_erosion_radius)

                Image.fromarray(np.asarray(rgb), mode="RGB").save(out_root / image_rel)
                Image.fromarray(normals_to_rgb(normals, mask, args.invalid_normal_rgb), mode="RGB").save(
                    out_root / normal_rel
                )
                Image.fromarray(mask_to_uint8(mask), mode="L").save(out_root / mask_rel)
                if args.write_depth_mm:
                    Image.fromarray(depth_to_uint16_mm(depth, depth_valid), mode="I;16").save(
                        out_root / "depth_mm" / f"{stem}_depth_mm.png"
                    )

                records.append(
                    SampleRecord(
                        index_1based=index_1based,
                        split=split_name,
                        scene=scenes[index],
                        scene_type=scene_types[index],
                        raw_rgb_filename=raw_rgb_filenames[index],
                        raw_depth_filename=raw_depth_filenames[index],
                        image_rel=image_rel,
                        normal_rel=normal_rel,
                        mask_rel=mask_rel,
                    )
                )
            records_by_split[split_name] = records
            write_split_file(out_root, split_name, records, focal_length=args.fx)

    manifest_rows = []
    for split_name, records in sorted(records_by_split.items()):
        for record in records:
            manifest_rows.append(
                {
                    "id": f"nyu_{record.index_1based:04d}",
                    "index_1based": record.index_1based,
                    "split": split_name,
                    "conditioning_image": record.image_rel.as_posix(),
                    "image": record.normal_rel.as_posix(),
                    "mask": record.mask_rel.as_posix(),
                    "text": args.prompt,
                    "task": "surface_normal_estimation",
                    "source_dataset": "nyu_depth_v2_labeled",
                    "scene": record.scene,
                    "scene_type": record.scene_type,
                    "raw_rgb_filename": record.raw_rgb_filename,
                    "raw_depth_filename": record.raw_depth_filename,
                }
            )
    with (out_root / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    counts = {split_name: len(records) for split_name, records in sorted(records_by_split.items())}
    write_json(
        out_root / "dataset_info.json",
        {
            "dataset": "nyu_surface_normals",
            "source_mat": str(mat_path),
            "source_splits_mat": str(splits_mat) if splits_mat is not None else None,
            "source_urls": {
                "nyu_depth_v2_labeled.mat": DEFAULT_LABELED_URL,
                "splits.mat": DEFAULT_SPLITS_URL,
            },
            "counts": counts,
            "prompt": args.prompt,
            "normal_encoding": LEGACY_DIRECT_NORMAL_ENCODING,
            "normal_encoding_formula": normal_encoding_formula(LEGACY_DIRECT_NORMAL_ENCODING),
            "normal_orientation": "face_camera_ray",
            "geometry_version": CAMERA_FACING_GEOMETRY_DATA_VERSION,
            "stored_normal_convention": CANONICAL_NORMAL_CONVENTION,
            "gt_to_canonical_permutation": [0, 1, 2],
            "gt_to_canonical_signs": [1, 1, 1],
            "normal_transform": CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL.metadata(),
            "intrinsics": {
                "fx": args.fx,
                "fy": args.fy,
                "cx": args.cx,
                "cy": args.cy,
            },
            "preprocessing": {
                "depth_source": args.depth_source,
                "mask_source": args.mask_source,
                "min_depth": args.min_depth,
                "max_depth": args.max_depth,
                "gradient_radius": args.gradient_radius,
                "mask_erosion_radius": args.mask_erosion_radius,
                "invalid_normal_rgb": list(args.invalid_normal_rgb),
            },
        },
    )

    if args.tar_output is not None:
        write_prepared_tar(out_root, args.tar_output)

    print("NYUv2 surface-normal dataset preparation complete.")
    print(f"  out_root: {out_root}")
    print(f"  rows: {sum(counts.values()):,}")
    for split_name, count in counts.items():
        print(f"  {split_name}: {count:,}")
    if args.tar_output is not None:
        print(f"  tar_output: {args.tar_output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
