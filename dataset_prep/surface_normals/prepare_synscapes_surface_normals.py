#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from common.geometry_protocol_prompts import VISION_BANANA_NORMAL_PROMPT
from common.surface_normal_encoding import (
    SUPPORTED_NORMAL_ENCODINGS,
    VISION_BANANA_NORMAL_ENCODING,
    encode_surface_normals_rgb,
    normal_encoding_formula,
)
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda iterable, **_: iterable

from dataset_prep.depth.prepare_synscapes_depth import load_exr_depth
from dataset_prep.geometry_conventions import (
    CAMERA_FACING_GEOMETRY_DATA_VERSION,
    CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL,
    CANONICAL_NORMAL_CONVENTION,
    GEOMETRY_DATA_VERSION,
    IMAGE_Y_DOWN_Z_AWAY_TO_CANONICAL,
    orient_normals_toward_camera,
    resize_normal_field,
    geometry_data_version_for_normal_encoding,
)


NORMAL_PROMPT = VISION_BANANA_NORMAL_PROMPT

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
_RESAMPLING = getattr(Image, "Resampling", Image)


@dataclass(frozen=True)
class SynscapesNormalPair:
    stem: str
    rgb_ref: Any
    depth_ref: Any
    meta_ref: Any | None


def resolve_synscapes_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    if (root / "img" / "depth").is_dir() and (root / "img").is_dir():
        return root
    nested = root / "Synscapes"
    if (nested / "img" / "depth").is_dir() and (nested / "img").is_dir():
        return nested
    raise FileNotFoundError(
        "Could not resolve Synscapes root. Expected either <root>/img/depth "
        f"or <root>/Synscapes/img/depth under: {root}"
    )


def sort_key(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def list_root_pairs(root: Path, rgb_subdir: str) -> list[SynscapesNormalPair]:
    root = resolve_synscapes_root(root)
    rgb_root = root / "img" / rgb_subdir
    depth_root = root / "img" / "depth"
    meta_root = root / "meta"
    if not rgb_root.is_dir():
        raise FileNotFoundError(f"Missing Synscapes RGB directory: {rgb_root}")
    if not depth_root.is_dir():
        raise FileNotFoundError(f"Missing Synscapes depth directory: {depth_root}")

    rgb_by_stem = {
        path.stem: path
        for path in rgb_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    depth_by_stem = {
        path.stem: path
        for path in depth_root.iterdir()
        if path.is_file() and path.suffix.lower() == ".exr"
    }
    meta_by_stem = {}
    if meta_root.is_dir():
        meta_by_stem = {
            path.stem: path
            for path in meta_root.iterdir()
            if path.is_file() and path.suffix.lower() == ".json"
        }
    stems = sorted(set(rgb_by_stem) & set(depth_by_stem), key=sort_key)
    return [
        SynscapesNormalPair(stem, rgb_by_stem[stem], depth_by_stem[stem], meta_by_stem.get(stem))
        for stem in stems
    ]


def list_tar_pairs(tar_path: Path, rgb_subdir: str) -> list[SynscapesNormalPair]:
    tar_path = tar_path.expanduser().resolve()
    if not tar_path.is_file():
        raise FileNotFoundError(f"Synscapes tar does not exist: {tar_path}")

    rgb_prefix = f"Synscapes/img/{rgb_subdir}/"
    depth_prefix = "Synscapes/img/depth/"
    meta_prefix = "Synscapes/meta/"
    with tarfile.open(tar_path, "r:*") as archive:
        rgb_by_stem: dict[str, str] = {}
        depth_by_stem: dict[str, str] = {}
        meta_by_stem: dict[str, str] = {}
        for member in archive.getmembers():
            if not member.isfile():
                continue
            name = member.name
            path = Path(name)
            if name.startswith(rgb_prefix) and path.suffix.lower() in IMAGE_SUFFIXES:
                rgb_by_stem[path.stem] = name
            elif name.startswith(depth_prefix) and path.suffix.lower() == ".exr":
                depth_by_stem[path.stem] = name
            elif name.startswith(meta_prefix) and path.suffix.lower() == ".json":
                meta_by_stem[path.stem] = name

    stems = sorted(set(rgb_by_stem) & set(depth_by_stem), key=sort_key)
    return [
        SynscapesNormalPair(stem, rgb_by_stem[stem], depth_by_stem[stem], meta_by_stem.get(stem))
        for stem in stems
    ]


def sample_pairs(pairs: list[SynscapesNormalPair], limit: int | None, seed: int) -> list[SynscapesNormalPair]:
    if limit is None or limit >= len(pairs):
        return pairs
    rng = random.Random(seed)
    sampled = list(pairs)
    rng.shuffle(sampled)
    return sorted(sampled[:limit], key=lambda pair: sort_key(pair.stem))


def load_rgb_from_root(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def load_rgb_from_tar(archive: tarfile.TarFile, member_name: str) -> Image.Image:
    extracted = archive.extractfile(member_name)
    if extracted is None:
        raise FileNotFoundError(f"Could not extract tar member: {member_name}")
    with Image.open(io.BytesIO(extracted.read())) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def read_member_bytes(archive: tarfile.TarFile, member_name: str) -> bytes:
    extracted = archive.extractfile(member_name)
    if extracted is None:
        raise FileNotFoundError(f"Could not extract tar member: {member_name}")
    return extracted.read()


def load_json_from_root(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json_from_tar(archive: tarfile.TarFile, member_name: str | None) -> dict[str, Any]:
    if member_name is None:
        return {}
    data = read_member_bytes(archive, member_name)
    return json.loads(data.decode("utf-8"))


def resize_to_max_side(image: Image.Image, max_side: int | None, resample: int) -> Image.Image:
    if max_side is None or max_side <= 0:
        return image
    width, height = image.size
    largest = max(width, height)
    if largest <= max_side:
        return image
    scale = max_side / largest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, resample=resample)


def parse_intrinsics(meta: dict[str, Any], depth_shape: tuple[int, int]) -> dict[str, float | bool]:
    height, width = depth_shape
    intrinsic = meta.get("camera", {}).get("intrinsic", {}) if isinstance(meta, dict) else {}
    required = ("fx", "fy", "u0", "v0")
    if not all(key in intrinsic for key in required):
        raise ValueError("Synscapes metadata is missing camera.intrinsic fx/fy/u0/v0.")

    fx = float(intrinsic["fx"])
    fy = float(intrinsic["fy"])
    cx = float(intrinsic["u0"])
    cy = float(intrinsic["v0"])
    resx = float(intrinsic.get("resx", width))
    resy = float(intrinsic.get("resy", height))
    scaled = False
    if resx > 0 and resy > 0 and (int(round(resx)) != width or int(round(resy)) != height):
        sx = width / resx
        sy = height / resy
        fx *= sx
        fy *= sy
        cx *= sx
        cy *= sy
        scaled = True
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "scaled_to_depth_shape": scaled}


def range_depth_to_z_depth(depth: np.ndarray, *, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    height, width = depth.shape
    u = np.arange(width, dtype=np.float32)[None, :]
    v = np.arange(height, dtype=np.float32)[:, None]
    x = (u - cx) / fx
    y = (v - cy) / fy
    ray_norm = np.sqrt(x * x + y * y + 1.0)
    return depth / ray_norm


def make_point_cloud(z_depth: np.ndarray, *, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    height, width = z_depth.shape
    u = np.arange(width, dtype=np.float32)[None, :]
    v = np.arange(height, dtype=np.float32)[:, None]
    z = z_depth.astype(np.float32, copy=False)
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
    z_depth: np.ndarray,
    valid_depth: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    radius: int,
    orient_z_positive: bool,
    max_depth_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    if radius < 1:
        raise ValueError("--gradient-radius must be >= 1")

    points = make_point_cloud(z_depth, fx=fx, fy=fy, cx=cx, cy=cy)
    height, width = z_depth.shape
    normals = np.zeros((height, width, 3), dtype=np.float32)

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
    if orient_z_positive:
        n = orient_normals_toward_camera(n, points[y_slice, x_slice], good)

    center_valid = valid_depth[y_slice, x_slice]
    neighbor_valid = (
        valid_depth[y_slice, slice(2 * radius, width)]
        & valid_depth[y_slice, slice(0, width - 2 * radius)]
        & valid_depth[slice(2 * radius, height), x_slice]
        & valid_depth[slice(0, height - 2 * radius), x_slice]
    )
    if max_depth_delta > 0:
        center = z_depth[y_slice, x_slice]
        smooth_enough = (
            (np.abs(z_depth[y_slice, slice(2 * radius, width)] - center) <= max_depth_delta)
            & (np.abs(z_depth[y_slice, slice(0, width - 2 * radius)] - center) <= max_depth_delta)
            & (np.abs(z_depth[slice(2 * radius, height), x_slice] - center) <= max_depth_delta)
            & (np.abs(z_depth[slice(0, height - 2 * radius), x_slice] - center) <= max_depth_delta)
        )
    else:
        smooth_enough = np.ones_like(center_valid, dtype=bool)

    normal_valid = np.zeros((height, width), dtype=bool)
    normal_valid[y_slice, x_slice] = good & center_valid & neighbor_valid & smooth_enough
    normals[y_slice, x_slice] = n
    return normals, normal_valid


def normals_to_rgb(
    normals: np.ndarray,
    valid: np.ndarray,
    invalid_rgb: tuple[int, int, int],
    encoding: str = VISION_BANANA_NORMAL_ENCODING,
) -> np.ndarray:
    return encode_surface_normals_rgb(
        normals, valid, encoding=encoding, invalid_rgb=invalid_rgb
    )


def mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    return mask.astype(np.uint8) * 255


def save_raw_normals(path: Path, normals: np.ndarray, mask: np.ndarray, dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dtype == "float16":
        normals_to_write = normals.astype(np.float16)
    elif dtype == "float32":
        normals_to_write = normals.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported raw normal dtype: {dtype}")
    np.savez_compressed(path, normals=normals_to_write, valid_mask=mask.astype(np.bool_))


def parse_rgb(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("RGB value must have the form R,G,B")
    rgb = tuple(int(part) for part in parts)
    if any(part < 0 or part > 255 for part in rgb):
        raise argparse.ArgumentTypeError("RGB channels must be in [0, 255]")
    return rgb


def write_prepared_tar(out_root: Path, tar_path: Path) -> None:
    tar_path = tar_path.expanduser().resolve()
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = tar_path.with_suffix(tar_path.suffix + ".partial")
    if tmp_path.exists():
        tmp_path.unlink()
    with tarfile.open(tmp_path, "w") as archive:
        archive.add(out_root, arcname=out_root.name)
    tmp_path.replace(tar_path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_sample(
    pair: SynscapesNormalPair,
    *,
    out_root: Path,
    rgb_loader: Callable[[Any], Image.Image],
    depth_loader: Callable[[Any], np.ndarray],
    meta_loader: Callable[[Any], dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    basename = f"synscapes_{int(pair.stem):05d}" if pair.stem.isdigit() else f"synscapes_{pair.stem}"
    cond_rel = Path("conditioning_images") / f"{basename}_input.png"
    target_rel = Path("target_images") / f"{basename}_normal.png"
    mask_rel = Path("masks") / f"{basename}_mask.png"
    raw_rel = Path("raw_normals") / f"{basename}_normals.npz"

    rgb = rgb_loader(pair.rgb_ref)
    depth_raw = depth_loader(pair.depth_ref)
    if depth_raw.ndim != 2:
        raise ValueError(f"Expected 2D depth after channel selection, got shape={depth_raw.shape}")

    meta = meta_loader(pair.meta_ref)
    intrinsics = parse_intrinsics(meta, depth_raw.shape)
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    z_depth = (
        range_depth_to_z_depth(depth_raw, fx=fx, fy=fy, cx=cx, cy=cy)
        if args.depth_kind == "ray"
        else depth_raw.astype(np.float32, copy=False)
    )
    valid_depth = np.isfinite(z_depth) & (z_depth >= args.min_depth) & (z_depth <= args.max_depth)
    normals, normal_valid = compute_normals(
        z_depth,
        valid_depth,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        radius=args.gradient_radius,
        orient_z_positive=args.orient_z_positive,
        max_depth_delta=args.max_depth_delta,
    )
    normal_transform = (
        CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL
        if args.orient_z_positive
        else IMAGE_Y_DOWN_Z_AWAY_TO_CANONICAL
    )
    normals = normal_transform.apply(normals, normal_valid)
    mask = erode_mask(normal_valid, args.mask_erosion_radius)
    valid_fraction = float(mask.mean())
    if valid_fraction < args.min_valid_fraction:
        return None

    max_side = args.max_side if args.max_side > 0 else None
    rgb = resize_to_max_side(rgb, max_side, _RESAMPLING.LANCZOS)
    if normals.shape[:2] != (rgb.height, rgb.width):
        normals, mask = resize_normal_field(normals, mask, rgb.size)
    target = Image.fromarray(
        normals_to_rgb(
            normals, mask, args.invalid_normal_rgb, args.normal_encoding
        ),
        mode="RGB",
    )
    mask_image = Image.fromarray(mask_to_uint8(mask), mode="L")

    (out_root / cond_rel).parent.mkdir(parents=True, exist_ok=True)
    (out_root / target_rel).parent.mkdir(parents=True, exist_ok=True)
    (out_root / mask_rel).parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out_root / cond_rel)
    target.save(out_root / target_rel)
    mask_image.save(out_root / mask_rel)
    raw_rel_str = None
    if args.write_raw_normals:
        save_raw_normals(out_root / raw_rel, normals, mask, args.raw_normal_dtype)
        raw_rel_str = raw_rel.as_posix()

    metadata = {
        "stem": pair.stem,
        "rgb_ref": str(pair.rgb_ref),
        "depth_ref": str(pair.depth_ref),
        "meta_ref": str(pair.meta_ref) if pair.meta_ref is not None else None,
        "valid_fraction": valid_fraction,
        "normal_encoding": args.normal_encoding,
        "normal_encoding_formula": normal_encoding_formula(args.normal_encoding),
        "coordinate_convention": CANONICAL_NORMAL_CONVENTION,
        **normal_transform.metadata(),
        "geometry_version": (
            CAMERA_FACING_GEOMETRY_DATA_VERSION
            if args.orient_z_positive
            else geometry_data_version_for_normal_encoding(args.normal_encoding)
        ),
        "intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "scaled_to_depth_shape": bool(intrinsics["scaled_to_depth_shape"]),
        },
        "depth_kind": args.depth_kind,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "gradient_radius": args.gradient_radius,
        "mask_erosion_radius": args.mask_erosion_radius,
        "max_depth_delta": args.max_depth_delta,
        "normal_orientation": "face_camera_ray" if args.orient_z_positive else "legacy_z_positive_disabled",
    }
    row = {
        "conditioning_image": cond_rel.as_posix(),
        "image": target_rel.as_posix(),
        "loss_mask": mask_rel.as_posix(),
        "mask": mask_rel.as_posix(),
        "text": args.prompt,
        "task": "surface_normal_estimation",
        "source_dataset": "synscapes",
        "id": basename,
        "preprocess_target": "resize_xyz_then_renormalize_then_encode",
        "metadata": metadata,
    }
    if raw_rel_str is not None:
        row["raw_normals"] = raw_rel_str
    return row


def prepare_from_root(pairs: list[SynscapesNormalPair], out_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    def process(pair: SynscapesNormalPair) -> dict[str, Any] | None:
        try:
            return build_sample(
                pair,
                out_root=out_root,
                rgb_loader=load_rgb_from_root,
                depth_loader=lambda path: load_exr_depth(path=path, channel=args.depth_channel),
                meta_loader=load_json_from_root,
                args=args,
            )
        except Exception as exc:
            if args.strict:
                raise
            print(f"WARNING: skipped {pair.stem}: {exc}", flush=True)
            return None

    if args.num_workers <= 1:
        rows = [process(pair) for pair in tqdm(pairs, desc="Preparing Synscapes surface normals")]
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            rows = list(
                tqdm(
                    executor.map(process, pairs),
                    total=len(pairs),
                    desc=f"Preparing Synscapes surface normals ({args.num_workers} workers)",
                )
            )
    return [row for row in rows if row is not None]


def prepare_from_tar(tar_path: Path, pairs: list[SynscapesNormalPair], out_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    with tarfile.open(tar_path, "r:*") as archive:
        for pair in tqdm(pairs, desc="Preparing Synscapes surface normals from tar"):
            try:
                row = build_sample(
                    pair,
                    out_root=out_root,
                    rgb_loader=lambda member_name: load_rgb_from_tar(archive, member_name),
                    depth_loader=lambda member_name: load_exr_depth(
                        data=read_member_bytes(archive, member_name),
                        channel=args.depth_channel,
                    ),
                    meta_loader=lambda member_name: load_json_from_tar(archive, member_name),
                    args=args,
                )
            except Exception as exc:
                if args.strict:
                    raise
                print(f"WARNING: skipped {pair.stem}: {exc}", flush=True)
                continue
            if row is not None:
                rows.append(row)
    return rows


def write_metadata(out_root: Path, rows: list[dict[str, Any]]) -> None:
    with (out_root / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Synscapes RGB-to-surface-normal training pairs.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synscapes-root", type=Path, help="Extracted Synscapes root or parent directory.")
    source.add_argument("--synscapes-tar", type=Path, help="Synscapes tar containing Synscapes/img/{rgb,depth} and Synscapes/meta.")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--prepared-tar", type=Path, default=None, help="Optional tar of the prepared dataset folder.")
    parser.add_argument("--rgb-subdir", default="rgb", choices=["rgb", "rgb-2k"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=1, help="Used for extracted-folder input.")
    parser.add_argument("--max-side", type=int, default=0, help="Resize larger RGB/normal targets to this side; <=0 preserves.")
    parser.add_argument("--prompt", default=NORMAL_PROMPT)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=250.0)
    parser.add_argument("--depth-channel", choices=["first", "last", "mean", "red", "green", "blue"], default="first")
    parser.add_argument(
        "--depth-kind",
        choices=["z", "ray"],
        default="z",
        help="Interpret EXR depth as z-depth or Euclidean ray distance before back-projecting.",
    )
    parser.add_argument("--gradient-radius", type=int, default=2)
    parser.add_argument("--mask-erosion-radius", type=int, default=1)
    parser.add_argument(
        "--max-depth-delta",
        type=float,
        default=0.0,
        help="Optional absolute depth-jump filter in meters for normal validity. <=0 disables.",
    )
    parser.add_argument("--min-valid-fraction", type=float, default=0.05)
    parser.add_argument("--invalid-normal-rgb", type=parse_rgb, default=(128, 128, 128))
    parser.add_argument(
        "--normal-encoding",
        choices=SUPPORTED_NORMAL_ENCODINGS,
        default=VISION_BANANA_NORMAL_ENCODING,
    )
    parser.add_argument(
        "--no-orient-z-positive",
        dest="orient_z_positive",
        action="store_false",
        help="Disable camera-ray-facing orientation (legacy reproduction only).",
    )
    parser.set_defaults(orient_z_positive=True)
    parser.add_argument("--write-raw-normals", action="store_true", help="Also write raw_normals/*.npz with normals and valid masks.")
    parser.add_argument("--raw-normal-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty output directory.")
    parser.add_argument("--strict", action="store_true", help="Fail on the first corrupt/unreadable sample.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.min_depth < 0 or args.max_depth <= args.min_depth:
        raise ValueError("--max-depth must be greater than --min-depth")
    if args.gradient_radius < 1:
        raise ValueError("--gradient-radius must be >= 1")
    if args.mask_erosion_radius < 0:
        raise ValueError("--mask-erosion-radius must be >= 0")
    if not 0.0 <= args.min_valid_fraction <= 1.0:
        raise ValueError("--min-valid-fraction must be in [0, 1]")

    out_root = args.out_root.expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {out_root}. Use --overwrite to rebuild.")
    for rel in ("conditioning_images", "target_images", "masks"):
        (out_root / rel).mkdir(parents=True, exist_ok=True)
    if args.write_raw_normals:
        (out_root / "raw_normals").mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    if args.synscapes_root is not None:
        source_path = args.synscapes_root.expanduser().resolve()
        pairs = list_root_pairs(source_path, args.rgb_subdir)
        pairs = sample_pairs(pairs, args.limit, args.seed)
        rows = prepare_from_root(pairs, out_root, args)
    else:
        source_path = args.synscapes_tar.expanduser().resolve()
        pairs = list_tar_pairs(source_path, args.rgb_subdir)
        pairs = sample_pairs(pairs, args.limit, args.seed)
        rows = prepare_from_tar(source_path, pairs, out_root, args)

    if not rows:
        raise RuntimeError("No Synscapes surface-normal rows were written.")
    write_metadata(out_root, rows)
    write_json(
        out_root / "dataset_info.json",
        {
            "dataset": "synscapes_surface_normals",
            "source": str(source_path),
            "source_kind": "root" if args.synscapes_root is not None else "tar",
            "rows_written": len(rows),
            "pairs_found": len(pairs),
            "prompt": args.prompt,
            "normal_encoding": args.normal_encoding,
            "normal_encoding_formula": normal_encoding_formula(args.normal_encoding),
            "coordinate_convention": CANONICAL_NORMAL_CONVENTION,
            "geometry_version": (
                CAMERA_FACING_GEOMETRY_DATA_VERSION
                if args.orient_z_positive
                else geometry_data_version_for_normal_encoding(args.normal_encoding)
            ),
            "normal_transform": (
                CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL.metadata()
                if args.orient_z_positive
                else IMAGE_Y_DOWN_Z_AWAY_TO_CANONICAL.metadata()
            ),
            "normal_orientation": "face_camera_ray" if args.orient_z_positive else "legacy_unoriented",
            "target_preparation": "resize_xyz_then_renormalize_then_encode",
            "preprocessing": {
                "rgb_subdir": args.rgb_subdir,
                "max_side": args.max_side,
                "min_depth": args.min_depth,
                "max_depth": args.max_depth,
                "depth_channel": args.depth_channel,
                "depth_kind": args.depth_kind,
                "gradient_radius": args.gradient_radius,
                "mask_erosion_radius": args.mask_erosion_radius,
                "max_depth_delta": args.max_depth_delta,
                "min_valid_fraction": args.min_valid_fraction,
                "invalid_normal_rgb": list(args.invalid_normal_rgb),
                "orient_camera_facing": args.orient_z_positive,
                "write_raw_normals": args.write_raw_normals,
                "raw_normal_dtype": args.raw_normal_dtype if args.write_raw_normals else None,
            },
        },
    )
    if args.prepared_tar is not None:
        write_prepared_tar(out_root, args.prepared_tar)

    skipped = len(pairs) - len(rows)
    print("Synscapes surface-normal preparation complete.")
    print(f"  pairs_found: {len(pairs):,}")
    print(f"  rows_written: {len(rows):,}")
    print(f"  skipped: {skipped:,}")
    print(f"  out_root: {out_root}")
    if args.prepared_tar is not None:
        print(f"  prepared_tar: {args.prepared_tar.expanduser().resolve()}")


if __name__ == "__main__":
    main()
