#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.surface_normal_encoding import (
    SUPPORTED_NORMAL_ENCODINGS,
    VISION_BANANA_NORMAL_ENCODING,
    encode_surface_normals_rgb,
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


SPLIT_FRAME_COUNTS = {
    "debug": 1,
    "small": 5,
    "standard": 10,
    "large": 20,
    "giant": 50,
}

DEFAULT_INVALID_NORMAL_RGB = (128, 128, 128)


@dataclass(frozen=True)
class RawFrame:
    scene_id: str
    frame_id: str
    color_path: str
    depth_path: str
    intrinsic_path: str | None
    zip_path: Path | None
    fx: float
    fy: float
    cx: float
    cy: float
    valid_fraction: float


@dataclass(frozen=True)
class PreparedFrame:
    raw: RawFrame
    image_rel: Path
    depth_rel: Path
    normal_rel: Path
    mask_rel: Path
    intrinsic_rel: Path
    sample_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a scene-balanced ScanNet depth and depth-derived normal evaluation dataset."
    )
    parser.add_argument("--source-root", type=Path, required=True, help="ScanNet root containing tasks/scannet_frames_25k.zip or extracted frames.")
    parser.add_argument("--frames-root", type=Path, default=None, help="Optional extracted scannet_frames_25k root.")
    parser.add_argument(
        "--scene-split-file",
        type=Path,
        default=None,
        help="Official ScanNet scene list for the requested split.",
    )
    parser.add_argument(
        "--val-scenes-file",
        type=Path,
        default=None,
        help="Deprecated alias for --scene-split-file (kept for release compatibility).",
    )
    parser.add_argument(
        "--split-name",
        choices=("train", "val", "test"),
        default="val",
        help="Split label recorded in filenames and manifest metadata.",
    )
    parser.add_argument("--out-root", type=Path, required=True, help="Prepared output directory.")
    parser.add_argument("--prepared-tar", type=Path, default=None, help="Optional tar path to write.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory/tar.")
    parser.add_argument("--extract-zip", action="store_true", help="Extract tasks/scannet_frames_25k.zip when frames are not already extracted.")
    parser.add_argument("--min-depth", type=float, default=0.001)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.05)
    parser.add_argument("--gradient-radius", type=int, default=2)
    parser.add_argument("--mask-erosion-radius", type=int, default=1)
    parser.add_argument("--invalid-normal-rgb", default="128,128,128")
    parser.add_argument(
        "--normal-encoding",
        choices=SUPPORTED_NORMAL_ENCODINGS,
        default=VISION_BANANA_NORMAL_ENCODING,
    )
    parser.add_argument("--strict", action="store_true", help="Fail on malformed frames instead of skipping them.")
    return parser.parse_args()


def parse_rgb_triplet(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected R,G,B triplet, got {value!r}")
    rgb = tuple(int(part) for part in parts)
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise ValueError(f"Invalid RGB triplet: {value!r}")
    return rgb  # type: ignore[return-value]


def find_frames_root(source_root: Path, explicit: Path | None, *, extract_zip: bool) -> Path:
    if explicit is not None:
        explicit = explicit.expanduser()
        if explicit.is_dir() or (explicit.is_file() and explicit.suffix.lower() == ".zip"):
            return explicit

    candidates = [
        source_root / "scannet_frames_25k",
        source_root / "tasks" / "scannet_frames_25k",
    ]

    zip_candidates = [
        source_root / "tasks" / "scannet_frames_25k.zip",
        source_root / "scannet_frames_25k.zip",
    ]
    archive = next((path for path in zip_candidates if path.is_file()), None)
    if archive is not None:
        return archive

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    if archive is not None and extract_zip:
        out_dir = source_root / "tasks"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {archive} into {out_dir}")
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(out_dir)
        for candidate in candidates:
            if candidate.is_dir():
                return candidate

    expected = "\n".join(f"  {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not find extracted scannet_frames_25k. Expected one of:\n"
        f"{expected}\n"
        "or pass --extract-zip when scannet_frames_25k.zip is present."
    )


def find_scene_split_file(
    source_root: Path,
    explicit: Path | None,
    *,
    split_name: str,
) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            source_root / "tasks" / f"scannetv2_{split_name}.txt",
            source_root / f"scannetv2_{split_name}.txt",
            source_root / "Tasks" / "Benchmark" / f"scannetv2_{split_name}.txt",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    expected = "\n".join(f"  {path}" for path in candidates)
    raise FileNotFoundError(
        f"Could not find scannetv2_{split_name}.txt. Download the official ScanNet split file and place it at one of:\n"
        f"{expected}"
    )


def read_scenes(path: Path) -> list[str]:
    scenes = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            scene = line.strip()
            if scene and not scene.startswith("#"):
                scenes.append(scene)
    if not scenes:
        raise ValueError(f"No scenes found in {path}")
    return scenes


def sorted_frame_paths(directory: Path, suffixes: tuple[str, ...]) -> list[Path]:
    paths = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes]
    return sorted(paths, key=lambda path: frame_sort_key(path.stem))


def frame_sort_key(stem: str) -> tuple[int, str]:
    try:
        return (int(stem), stem)
    except ValueError:
        return (10**12, stem)


def intrinsics_from_matrix(matrix: np.ndarray, width: int, height: int) -> tuple[float, float, float, float]:
    if matrix.ndim == 2 and matrix.shape[0] >= 3 and matrix.shape[1] >= 3:
        fx = float(matrix[0, 0])
        fy = float(matrix[1, 1])
        cx = float(matrix[0, 2])
        cy = float(matrix[1, 2])
        if all(math.isfinite(value) and value > 0 for value in (fx, fy)):
            return fx, fy, cx, cy
    return float(width), float(height), float(width - 1) / 2.0, float(height - 1) / 2.0


def load_intrinsics(path: Path | None, width: int, height: int) -> tuple[float, float, float, float]:
    if path is not None and path.is_file():
        matrix = np.loadtxt(path, dtype=np.float64)
        return intrinsics_from_matrix(matrix, width=width, height=height)
    return float(width), float(height), float(width - 1) / 2.0, float(height - 1) / 2.0


def load_zip_intrinsics(handle: zipfile.ZipFile, key: str | None, width: int, height: int) -> tuple[float, float, float, float]:
    if key is not None:
        with handle.open(key) as member:
            text = member.read().decode("utf-8")
        matrix = np.loadtxt(io.StringIO(text), dtype=np.float64)
        return intrinsics_from_matrix(matrix, width=width, height=height)
    return float(width), float(height), float(width - 1) / 2.0, float(height - 1) / 2.0


def read_depth_m(path: Path) -> np.ndarray:
    depth = np.asarray(Image.open(path), dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth / 1000.0


def read_zip_depth_m(handle: zipfile.ZipFile, key: str) -> np.ndarray:
    with handle.open(key) as member:
        depth = np.asarray(Image.open(member), dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth / 1000.0


def valid_depth_mask(depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    return np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)


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
    height, width = depth.shape
    normals = np.zeros((height, width, 3), dtype=np.float32)
    normal_valid = np.zeros((height, width), dtype=bool)
    if height <= 2 * radius or width <= 2 * radius:
        return normals, normal_valid

    points = make_point_cloud(depth, fx=fx, fy=fy, cx=cx, cy=cy)
    valid = np.isfinite(depth) & (depth > 0)
    y_slice = slice(radius, height - radius)
    x_slice = slice(radius, width - radius)

    dx = points[y_slice, slice(2 * radius, width)] - points[y_slice, slice(0, width - 2 * radius)]
    dy = points[slice(2 * radius, height), x_slice] - points[slice(0, height - 2 * radius), x_slice]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    good = np.isfinite(norm[..., 0]) & (norm[..., 0] > 1e-8)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 1e-8)
    n = orient_normals_toward_camera(n, points[y_slice, x_slice], good)

    neighbor_valid = (
        valid[y_slice, slice(2 * radius, width)]
        & valid[y_slice, slice(0, width - 2 * radius)]
        & valid[slice(2 * radius, height), x_slice]
        & valid[slice(0, height - 2 * radius), x_slice]
    )
    normals[y_slice, x_slice] = n
    normal_valid[y_slice, x_slice] = good & neighbor_valid
    return CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL.apply(normals, normal_valid), normal_valid


def normals_to_rgb(
    normals: np.ndarray,
    valid: np.ndarray,
    invalid_rgb: tuple[int, int, int],
    encoding: str = VISION_BANANA_NORMAL_ENCODING,
) -> np.ndarray:
    return encode_surface_normals_rgb(
        normals, valid, encoding=encoding, invalid_rgb=invalid_rgb
    )


def depth_to_uint16_mm(depth_m: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros(depth_m.shape, dtype=np.uint16)
    out[valid] = np.clip(np.rint(depth_m[valid] * 1000.0), 0, 65535).astype(np.uint16)
    return out


def write_intrinsics(path: Path, fx: float, fy: float, cx: float, cy: float) -> None:
    matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    np.savetxt(path, matrix, fmt="%.10f")


def discover_raw_frames(
    frames_root: Path,
    val_scenes: Iterable[str],
    *,
    min_depth: float,
    max_depth: float,
    min_valid_fraction: float,
    strict: bool,
) -> list[RawFrame]:
    if frames_root.is_file() and frames_root.suffix.lower() == ".zip":
        return discover_zip_raw_frames(
            frames_root,
            val_scenes,
            min_depth=min_depth,
            max_depth=max_depth,
            min_valid_fraction=min_valid_fraction,
            strict=strict,
        )

    frames: list[RawFrame] = []
    for scene_id in tqdm(list(val_scenes), desc="Scanning ScanNet scenes"):
        scene_dir = frames_root / scene_id
        color_dir = scene_dir / "color"
        depth_dir = scene_dir / "depth"
        intrinsic_dir = scene_dir / "intrinsic"
        if not color_dir.is_dir() or not depth_dir.is_dir():
            if strict:
                raise FileNotFoundError(f"Missing color/depth directories for {scene_id}")
            continue

        color_paths = sorted_frame_paths(color_dir, (".jpg", ".jpeg", ".png"))
        depth_by_stem = {path.stem: path for path in sorted_frame_paths(depth_dir, (".png",))}
        intrinsic_path = first_existing(
            [
                intrinsic_dir / "intrinsic_depth.txt",
                intrinsic_dir / "intrinsic_color.txt",
                scene_dir / "intrinsics_depth.txt",
                scene_dir / "intrinsics_color.txt",
            ]
        )

        for color_path in color_paths:
            depth_path = depth_by_stem.get(color_path.stem)
            if depth_path is None:
                if strict:
                    raise FileNotFoundError(f"Missing depth for {color_path}")
                continue
            try:
                depth = read_depth_m(depth_path)
                valid = valid_depth_mask(depth, min_depth=min_depth, max_depth=max_depth)
                valid_fraction = float(valid.mean())
                if valid_fraction < min_valid_fraction:
                    continue
                fx, fy, cx, cy = load_intrinsics(intrinsic_path, width=depth.shape[1], height=depth.shape[0])
            except Exception:
                if strict:
                    raise
                continue
            frames.append(
                RawFrame(
                    scene_id=scene_id,
                    frame_id=color_path.stem,
                    color_path=str(color_path),
                    depth_path=str(depth_path),
                    intrinsic_path=str(intrinsic_path) if intrinsic_path is not None else None,
                    zip_path=None,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    valid_fraction=valid_fraction,
                )
            )
    return frames


def discover_zip_raw_frames(
    archive: Path,
    val_scenes: Iterable[str],
    *,
    min_depth: float,
    max_depth: float,
    min_valid_fraction: float,
    strict: bool,
) -> list[RawFrame]:
    frames: list[RawFrame] = []
    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
        for scene_id in tqdm(list(val_scenes), desc="Scanning ScanNet scenes from zip"):
            scene_prefix = f"scannet_frames_25k/{scene_id}/"
            color_prefix = f"{scene_prefix}color/"
            depth_prefix = f"{scene_prefix}depth/"
            color_keys = sorted(
                (
                    name
                    for name in names
                    if name.startswith(color_prefix) and Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
                ),
                key=lambda name: frame_sort_key(Path(name).stem),
            )
            depth_by_stem = {
                Path(name).stem: name
                for name in names
                if name.startswith(depth_prefix) and Path(name).suffix.lower() == ".png"
            }
            if not color_keys or not depth_by_stem:
                if strict:
                    raise FileNotFoundError(f"Missing color/depth entries for {scene_id} in {archive}")
                continue
            intrinsic_key = next(
                (
                    key
                    for key in (
                        f"{scene_prefix}intrinsics_depth.txt",
                        f"{scene_prefix}intrinsics_color.txt",
                        f"{scene_prefix}intrinsic/intrinsic_depth.txt",
                        f"{scene_prefix}intrinsic/intrinsic_color.txt",
                    )
                    if key in names
                ),
                None,
            )

            for color_key in color_keys:
                frame_id = Path(color_key).stem
                depth_key = depth_by_stem.get(frame_id)
                if depth_key is None:
                    if strict:
                        raise FileNotFoundError(f"Missing depth for {color_key}")
                    continue
                try:
                    depth = read_zip_depth_m(handle, depth_key)
                    valid = valid_depth_mask(depth, min_depth=min_depth, max_depth=max_depth)
                    valid_fraction = float(valid.mean())
                    if valid_fraction < min_valid_fraction:
                        continue
                    fx, fy, cx, cy = load_zip_intrinsics(
                        handle,
                        intrinsic_key,
                        width=depth.shape[1],
                        height=depth.shape[0],
                    )
                except Exception:
                    if strict:
                        raise
                    continue
                frames.append(
                    RawFrame(
                        scene_id=scene_id,
                        frame_id=frame_id,
                        color_path=color_key,
                        depth_path=depth_key,
                        intrinsic_path=intrinsic_key,
                        zip_path=archive,
                        fx=fx,
                        fy=fy,
                        cx=cx,
                        cy=cy,
                        valid_fraction=valid_fraction,
                    )
                )
    return frames


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def raw_color_suffix(raw: RawFrame) -> str:
    return Path(raw.color_path).suffix.lower()


def copy_raw_color(raw: RawFrame, out_path: Path) -> None:
    if raw.zip_path is None:
        shutil.copy2(Path(raw.color_path), out_path)
        return
    with zipfile.ZipFile(raw.zip_path) as handle:
        with handle.open(raw.color_path) as source, out_path.open("wb") as target:
            shutil.copyfileobj(source, target)


def read_raw_depth_m(raw: RawFrame) -> np.ndarray:
    if raw.zip_path is None:
        return read_depth_m(Path(raw.depth_path))
    with zipfile.ZipFile(raw.zip_path) as handle:
        return read_zip_depth_m(handle, raw.depth_path)


def sample_scene_frames(frames: list[PreparedFrame], count: int) -> list[PreparedFrame]:
    if len(frames) <= count:
        return frames
    indices = np.linspace(0, len(frames) - 1, num=count)
    chosen = sorted({int(round(index)) for index in indices})
    while len(chosen) < count:
        for index in range(len(frames)):
            if index not in chosen:
                chosen.append(index)
                if len(chosen) == count:
                    break
    return [frames[index] for index in sorted(chosen)]


def sample_raw_frames_for_largest_split(frames: list[RawFrame]) -> list[RawFrame]:
    max_per_scene = max(SPLIT_FRAME_COUNTS.values())
    by_scene: dict[str, list[RawFrame]] = {}
    for frame in frames:
        by_scene.setdefault(frame.scene_id, []).append(frame)
    sampled: list[RawFrame] = []
    for scene_id in sorted(by_scene):
        scene_frames = sorted(by_scene[scene_id], key=lambda frame: frame_sort_key(frame.frame_id))
        if len(scene_frames) <= max_per_scene:
            sampled.extend(scene_frames)
            continue
        indices = np.linspace(0, len(scene_frames) - 1, num=max_per_scene)
        chosen = sorted({int(round(index)) for index in indices})
        while len(chosen) < max_per_scene:
            for index in range(len(scene_frames)):
                if index not in chosen:
                    chosen.append(index)
                    if len(chosen) == max_per_scene:
                        break
        sampled.extend(scene_frames[index] for index in sorted(chosen))
    return sampled


def prepare_frames(
    raw_frames: list[RawFrame],
    out_root: Path,
    *,
    min_depth: float,
    max_depth: float,
    gradient_radius: int,
    mask_erosion_radius: int,
    invalid_normal_rgb: tuple[int, int, int],
    normal_encoding: str,
) -> list[PreparedFrame]:
    prepared: list[PreparedFrame] = []
    for raw in tqdm(raw_frames, desc="Preparing ScanNet frames"):
        scene_id = raw.scene_id
        frame_id = raw.frame_id
        sample_id = f"{scene_id}_{frame_id}"
        image_rel = Path("images") / scene_id / f"{frame_id}{raw_color_suffix(raw)}"
        depth_rel = Path("depths") / scene_id / f"{frame_id}.png"
        normal_rel = Path("normals_depth") / scene_id / f"{frame_id}.png"
        mask_rel = Path("masks") / scene_id / f"{frame_id}.png"
        intrinsic_rel = Path("intrinsics") / scene_id / f"{frame_id}.txt"

        for rel in (image_rel, depth_rel, normal_rel, mask_rel, intrinsic_rel):
            (out_root / rel).parent.mkdir(parents=True, exist_ok=True)

        copy_raw_color(raw, out_root / image_rel)
        depth = read_raw_depth_m(raw)
        depth_valid = valid_depth_mask(depth, min_depth=min_depth, max_depth=max_depth)
        normals, normal_valid = compute_normals(
            depth,
            fx=raw.fx,
            fy=raw.fy,
            cx=raw.cx,
            cy=raw.cy,
            radius=gradient_radius,
        )
        eval_mask = erode_mask(depth_valid & normal_valid, mask_erosion_radius)

        Image.fromarray(depth_to_uint16_mm(depth, depth_valid), mode="I;16").save(out_root / depth_rel)
        Image.fromarray(
            normals_to_rgb(
                normals, eval_mask, invalid_normal_rgb, normal_encoding
            ),
            mode="RGB",
        ).save(out_root / normal_rel)
        Image.fromarray((eval_mask.astype(np.uint8) * 255), mode="L").save(out_root / mask_rel)
        write_intrinsics(out_root / intrinsic_rel, raw.fx, raw.fy, raw.cx, raw.cy)

        prepared.append(
            PreparedFrame(
                raw=raw,
                image_rel=image_rel,
                depth_rel=depth_rel,
                normal_rel=normal_rel,
                mask_rel=mask_rel,
                intrinsic_rel=intrinsic_rel,
                sample_id=sample_id,
            )
        )
    return prepared


def write_split_files(
    out_root: Path,
    prepared: list[PreparedFrame],
    *,
    split_name: str,
) -> dict[str, int]:
    by_scene: dict[str, list[PreparedFrame]] = {}
    for frame in prepared:
        by_scene.setdefault(frame.raw.scene_id, []).append(frame)
    for frames in by_scene.values():
        frames.sort(key=lambda frame: frame_sort_key(frame.raw.frame_id))

    counts = {}
    for profile_name, per_scene in SPLIT_FRAME_COUNTS.items():
        rows = []
        for scene_id in sorted(by_scene):
            for frame in sample_scene_frames(by_scene[scene_id], per_scene):
                rows.append(
                    " ".join(
                        [
                            str(frame.image_rel),
                            str(frame.depth_rel),
                            str(frame.normal_rel),
                            str(frame.mask_rel),
                            str(frame.intrinsic_rel),
                            f"{frame.raw.fx:.10f}",
                            frame.sample_id,
                            frame.raw.scene_id,
                            frame.raw.frame_id,
                        ]
                    )
                )
        split_path = out_root / f"scannet_{split_name}_{profile_name}.txt"
        split_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        counts[profile_name] = len(rows)
    return counts


def write_manifest(
    out_root: Path,
    *,
    source_root: Path,
    frames_root: Path,
    scene_split_file: Path,
    scene_count: int,
    split_name: str,
    raw_frame_count: int,
    prepared_count: int,
    split_counts: dict[str, int],
    args: argparse.Namespace,
) -> None:
    manifest = {
        "dataset": "scannet",
        "prepared_root_name": out_root.name,
        "normal_target": "depth-derived camera-space normals",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "frames_root": str(frames_root),
        "scene_split_file": str(scene_split_file),
        "scene_count": scene_count,
        "split": split_name,
        "raw_valid_frame_count": raw_frame_count,
        "prepared_frame_count": prepared_count,
        "split_counts": split_counts,
        "depth_units": "uint16 millimeters; 0 is invalid",
        "normal_encoding": args.normal_encoding,
        "normal_encoding_formula": normal_encoding_formula(args.normal_encoding),
        "geometry_version": CAMERA_FACING_GEOMETRY_DATA_VERSION,
        "normal_transform": CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL.metadata(),
        "normal_orientation": "face_camera_ray",
        "stored_normal_convention": CANONICAL_NORMAL_CONVENTION,
        "gt_to_canonical_permutation": [0, 1, 2],
        "gt_to_canonical_signs": [1, 1, 1],
        "mask_encoding": "uint8 255 valid, 0 invalid",
        "parameters": {
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "min_valid_fraction": args.min_valid_fraction,
            "gradient_radius": args.gradient_radius,
            "mask_erosion_radius": args.mask_erosion_radius,
            "invalid_normal_rgb": args.invalid_normal_rgb,
            "splits": SPLIT_FRAME_COUNTS,
        },
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tar(out_root: Path, tar_path: Path, *, force: bool) -> None:
    if tar_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing tar without --force: {tar_path}")
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    partial = tar_path.with_suffix(tar_path.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    with tarfile.open(partial, "w") as handle:
        handle.add(out_root, arcname=out_root.name)
    partial.replace(tar_path)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    out_root = args.out_root.expanduser()
    invalid_normal_rgb = parse_rgb_triplet(args.invalid_normal_rgb)

    if out_root.exists():
        if not args.force:
            raise FileExistsError(f"Output root already exists; pass --force to rebuild: {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    frames_root = find_frames_root(source_root, args.frames_root, extract_zip=args.extract_zip).resolve()
    if args.scene_split_file is not None and args.val_scenes_file is not None:
        raise ValueError("Pass only one of --scene-split-file and --val-scenes-file")
    explicit_split = args.scene_split_file or args.val_scenes_file
    scene_split_file = find_scene_split_file(
        source_root,
        explicit_split,
        split_name=args.split_name,
    ).resolve()
    scenes = read_scenes(scene_split_file)
    raw_frames = discover_raw_frames(
        frames_root,
        scenes,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        min_valid_fraction=args.min_valid_fraction,
        strict=args.strict,
    )
    if not raw_frames:
        raise FileNotFoundError(f"No valid ScanNet {args.split_name} frames were found.")

    selected_raw_frames = sample_raw_frames_for_largest_split(raw_frames)
    prepared = prepare_frames(
        selected_raw_frames,
        out_root,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        gradient_radius=args.gradient_radius,
        mask_erosion_radius=args.mask_erosion_radius,
        invalid_normal_rgb=invalid_normal_rgb,
        normal_encoding=args.normal_encoding,
    )
    split_counts = write_split_files(out_root, prepared, split_name=args.split_name)
    write_manifest(
        out_root,
        source_root=source_root,
        frames_root=frames_root,
        scene_split_file=scene_split_file,
        scene_count=len(scenes),
        split_name=args.split_name,
        raw_frame_count=len(raw_frames),
        prepared_count=len(prepared),
        split_counts=split_counts,
        args=args,
    )

    if args.prepared_tar is not None:
        write_tar(out_root, args.prepared_tar.expanduser(), force=args.force)

    print("ScanNet preparation finished.")
    print(f"  prepared root: {out_root}")
    if args.prepared_tar is not None:
        print(f"  prepared tar:  {args.prepared_tar.expanduser()}")
    for split_name, count in split_counts.items():
        print(f"  {args.split_name}_{split_name}: {count}")


if __name__ == "__main__":
    main()
