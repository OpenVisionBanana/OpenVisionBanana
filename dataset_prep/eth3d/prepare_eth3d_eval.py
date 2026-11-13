#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.geometry_protocol_prompts import VISION_BANANA_NORMAL_PROMPT
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


DEPTH_PROMPT = (
    "[TASK: DEPTH_ESTIMATION] Generate a metric depth visualization of the input natural scene "
    "using the Vision Banana RGB depth encoding. Output only the RGB depth visualization with no "
    "text, labels, legends, or extra objects."
)
NORMAL_PROMPT = VISION_BANANA_NORMAL_PROMPT


@dataclass(frozen=True)
class Eth3dScene:
    scene: str
    image_path: Path
    right_image_path: Path
    calib_path: Path
    disparity_path: Path
    mask_path: Path


@dataclass(frozen=True)
class Calibration:
    intrinsics: np.ndarray
    baseline_raw: float
    baseline_m: float
    baseline_unit: str


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_prepared_tar(out_root: Path, tar_path: Path) -> None:
    tar_path = tar_path.expanduser().resolve()
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = tar_path.with_suffix(tar_path.suffix + ".partial")
    if tmp_path.exists():
        tmp_path.unlink()
    mode = "w:gz" if tar_path.name.endswith((".tar.gz", ".tgz")) else "w"
    with tarfile.open(tmp_path, mode) as archive:
        archive.add(out_root, arcname=out_root.name)
    tmp_path.replace(tar_path)


def reset_out_root(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite to rebuild.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    return mask.astype(bool).astype(np.uint8) * 255


def normals_to_rgb(
    normals: np.ndarray,
    valid: np.ndarray,
    invalid_rgb: tuple[int, int, int],
    encoding: str = VISION_BANANA_NORMAL_ENCODING,
) -> np.ndarray:
    return encode_surface_normals_rgb(
        normals, valid, encoding=encoding, invalid_rgb=invalid_rgb
    )


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool, copy=True)
    eroded = mask.astype(bool, copy=True)
    height, width = eroded.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.zeros_like(eroded)
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


def extract_7z_if_needed(archive: Path, output_root: Path, expected_dir: str, force: bool) -> None:
    if (output_root / expected_dir).is_dir() and not force:
        return
    if not archive.is_file():
        raise FileNotFoundError(f"Missing ETH3D archive: {archive}")
    seven_zip = shutil.which("7z") or shutil.which("7za") or shutil.which("7zr")
    if seven_zip is None:
        try:
            import py7zr  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "ETH3D raw archives are .7z files, but neither `7z` nor Python package `py7zr` "
                "is available. Install p7zip-full, install py7zr, or extract the archives manually "
                "before running preparation."
            ) from exc
        if force and (output_root / expected_dir).exists():
            shutil.rmtree(output_root / expected_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        with py7zr.SevenZipFile(archive, mode="r") as handle:
            handle.extractall(path=output_root)
        return

    if force and (output_root / expected_dir).exists():
        shutil.rmtree(output_root / expected_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    subprocess.run([seven_zip, "x", "-y", f"-o{output_root}", str(archive)], check=True)


def ensure_raw_layout(args: argparse.Namespace) -> Path:
    raw_root = args.eth3d_root.expanduser().resolve()
    if (raw_root / "two_view_training").is_dir() and (raw_root / "two_view_training_gt").is_dir():
        return raw_root
    if any((path / "im0.png").is_file() and (path / "disp0GT.pfm").is_file() for path in raw_root.iterdir() if path.is_dir()):
        return raw_root

    extract_7z_if_needed(raw_root / "two_view_training.7z", raw_root, "two_view_training", args.force_extract)
    extract_7z_if_needed(raw_root / "two_view_training_gt.7z", raw_root, "two_view_training_gt", args.force_extract)
    if (raw_root / "two_view_training").is_dir() and (raw_root / "two_view_training_gt").is_dir():
        return raw_root
    if any((path / "im0.png").is_file() and (path / "disp0GT.pfm").is_file() for path in raw_root.iterdir() if path.is_dir()):
        return raw_root
    if not (raw_root / "two_view_training").is_dir() or not (raw_root / "two_view_training_gt").is_dir():
        raise FileNotFoundError(f"Could not resolve ETH3D raw layout under {raw_root}")
    return raw_root


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = handle.readline().decode("ascii").strip()
        if header not in {"PF", "Pf"}:
            raise ValueError(f"Unsupported PFM header in {path}: {header!r}")
        color = header == "PF"

        line = handle.readline().decode("ascii").strip()
        while line.startswith("#"):
            line = handle.readline().decode("ascii").strip()
        width, height = [int(value) for value in line.split()]

        scale = float(handle.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        channels = 3 if color else 1
        data = np.frombuffer(handle.read(), dtype=endian + "f4")
        expected = width * height * channels
        if data.size != expected:
            raise ValueError(f"PFM payload size mismatch in {path}: expected {expected}, got {data.size}")
        shape = (height, width, channels) if color else (height, width)
        return np.flipud(data.reshape(shape)).astype(np.float32, copy=False)


def parse_matrix(value: str) -> np.ndarray:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    rows = [row.strip() for row in value.split(";") if row.strip()]
    matrix = [[float(part) for part in row.split()] for row in rows]
    return np.asarray(matrix, dtype=np.float32)


def parse_calibration(path: Path, baseline_unit: str) -> Calibration:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()

    if "cam0" not in fields:
        raise ValueError(f"Missing cam0 calibration in {path}")
    if "baseline" not in fields:
        raise ValueError(f"Missing baseline calibration in {path}")

    intrinsics = parse_matrix(fields["cam0"])
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Expected 3x3 cam0 matrix in {path}, got {intrinsics.shape}")
    baseline_raw = float(fields["baseline"])
    if baseline_unit == "auto":
        resolved_unit = "millimeters" if baseline_raw > 10.0 else "meters"
    else:
        resolved_unit = baseline_unit
    baseline_m = baseline_raw / 1000.0 if resolved_unit == "millimeters" else baseline_raw
    if baseline_m <= 0:
        raise ValueError(f"Invalid baseline in {path}: {baseline_raw}")
    return Calibration(
        intrinsics=intrinsics.astype(np.float32, copy=False),
        baseline_raw=baseline_raw,
        baseline_m=float(baseline_m),
        baseline_unit=resolved_unit,
    )


def scene_slug(scene: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", scene).strip("_")


def load_scenes(raw_root: Path) -> list[Eth3dScene]:
    scenes: list[Eth3dScene] = []
    if (raw_root / "two_view_training").is_dir() and (raw_root / "two_view_training_gt").is_dir():
        image_root = raw_root / "two_view_training"
        gt_root = raw_root / "two_view_training_gt"
        for scene_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
            gt_dir = gt_root / scene_dir.name
            row = Eth3dScene(
                scene=scene_dir.name,
                image_path=scene_dir / "im0.png",
                right_image_path=scene_dir / "im1.png",
                calib_path=scene_dir / "calib.txt",
                disparity_path=gt_dir / "disp0GT.pfm",
                mask_path=gt_dir / "mask0nocc.png",
            )
            missing = [path for path in (row.image_path, row.right_image_path, row.calib_path, row.disparity_path, row.mask_path) if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing ETH3D files for scene {scene_dir.name}: {missing}")
            scenes.append(row)
    else:
        for scene_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
            row = Eth3dScene(
                scene=scene_dir.name,
                image_path=scene_dir / "im0.png",
                right_image_path=scene_dir / "im1.png",
                calib_path=scene_dir / "calib.txt",
                disparity_path=scene_dir / "disp0GT.pfm",
                mask_path=scene_dir / "mask0nocc.png",
            )
            if all(path.is_file() for path in (row.image_path, row.right_image_path, row.calib_path, row.disparity_path, row.mask_path)):
                scenes.append(row)
    if not scenes:
        raise FileNotFoundError(f"No ETH3D scenes found under {raw_root}")
    return scenes


def derive_normals_from_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    radius: int,
    max_neighbor_depth_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    if radius < 1:
        raise ValueError("--gradient-radius must be >= 1")

    height, width = depth.shape
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    u = np.arange(width, dtype=np.float32)[None, :]
    v = np.arange(height, dtype=np.float32)[:, None]
    z = depth.astype(np.float32, copy=False)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack((x, y, z), axis=-1)

    normals = np.zeros((height, width, 3), dtype=np.float32)
    valid_depth = np.isfinite(depth) & (depth > 0)
    if height <= 2 * radius or width <= 2 * radius:
        return normals, np.zeros((height, width), dtype=bool)

    y_slice = slice(radius, height - radius)
    x_slice = slice(radius, width - radius)
    left = points[y_slice, slice(0, width - 2 * radius)]
    right = points[y_slice, slice(2 * radius, width)]
    up = points[slice(0, height - 2 * radius), x_slice]
    down = points[slice(2 * radius, height), x_slice]

    dx = right - left
    dy = down - up
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    good = np.isfinite(norm[..., 0]) & (norm[..., 0] > 1e-8)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 1e-8)
    n = orient_normals_toward_camera(n, points[y_slice, x_slice], good)
    normals[y_slice, x_slice] = n

    neighbor_valid = (
        valid_depth[y_slice, slice(0, width - 2 * radius)]
        & valid_depth[y_slice, slice(2 * radius, width)]
        & valid_depth[slice(0, height - 2 * radius), x_slice]
        & valid_depth[slice(2 * radius, height), x_slice]
        & valid_depth[y_slice, x_slice]
    )
    if max_neighbor_depth_ratio > 0:
        center = depth[y_slice, x_slice]
        neighbors = np.stack(
            (
                depth[y_slice, slice(0, width - 2 * radius)],
                depth[y_slice, slice(2 * radius, width)],
                depth[slice(0, height - 2 * radius), x_slice],
                depth[slice(2 * radius, height), x_slice],
            ),
            axis=0,
        )
        rel_jump = np.max(np.abs(neighbors - center[None, ...]) / np.maximum(center[None, ...], 1e-6), axis=0)
        neighbor_valid &= rel_jump <= max_neighbor_depth_ratio

    normal_valid = np.zeros((height, width), dtype=bool)
    normal_valid[y_slice, x_slice] = good & neighbor_valid
    return CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL.apply(normals, normal_valid), normal_valid


def prepare_depth(args: argparse.Namespace, scenes: list[Eth3dScene]) -> None:
    out_root = args.depth_out_root.expanduser().resolve()
    reset_out_root(out_root, args.overwrite)
    for rel in ("images/low_res_two_view", "depth", "masks", "intrinsics", "metadata", "splits"):
        (out_root / rel).mkdir(parents=True, exist_ok=True)

    split_lines: list[str] = []
    rows: list[dict] = []
    for scene in tqdm(scenes, desc="Preparing ETH3D depth"):
        slug = scene_slug(scene.scene)
        calib = parse_calibration(scene.calib_path, args.baseline_unit)
        disp = read_pfm(scene.disparity_path)
        if disp.ndim == 3:
            disp = disp[..., 0]
        mask_occ = np.asarray(Image.open(scene.mask_path).convert("L"), dtype=np.uint8) > 0
        if mask_occ.shape != disp.shape:
            raise ValueError(f"Mask/disparity shape mismatch for {scene.scene}: {mask_occ.shape} vs {disp.shape}")

        valid_disp = np.isfinite(disp) & (disp > 0)
        depth_m = np.full(disp.shape, np.nan, dtype=np.float32)
        depth_m[valid_disp] = float(calib.intrinsics[0, 0]) * calib.baseline_m / disp[valid_disp]
        valid_depth = mask_occ & valid_disp & np.isfinite(depth_m) & (depth_m > args.min_depth) & (depth_m < args.max_depth)

        image_rel = Path("images/low_res_two_view") / f"{slug}.png"
        depth_rel = Path("depth") / f"{slug}_depth_m.npy"
        mask_rel = Path("masks") / f"{slug}_depth_valid.png"
        intrinsics_rel = Path("intrinsics") / f"{slug}.txt"
        metadata_rel = Path("metadata") / f"{slug}.json"

        shutil.copy2(scene.image_path, out_root / image_rel)
        np.save(out_root / depth_rel, depth_m.astype(np.float32, copy=False))
        Image.fromarray(mask_to_uint8(valid_depth), mode="L").save(out_root / mask_rel)
        np.savetxt(out_root / intrinsics_rel, calib.intrinsics, fmt="%.10f")

        metadata = {
            "id": slug,
            "scene": scene.scene,
            "split": "test",
            "case": "low_res_two_view",
            "conditioning_image": image_rel.as_posix(),
            "depth": depth_rel.as_posix(),
            "mask": mask_rel.as_posix(),
            "intrinsics": intrinsics_rel.as_posix(),
            "source_image": str(scene.image_path),
            "source_right_image": str(scene.right_image_path),
            "source_disparity": str(scene.disparity_path),
            "source_mask": str(scene.mask_path),
            "source_calib": str(scene.calib_path),
            "baseline_raw": calib.baseline_raw,
            "baseline_unit": calib.baseline_unit,
            "baseline_m": calib.baseline_m,
            "depth_unit": "meters",
            "valid_depth_pixels": int(valid_depth.sum()),
            "task": "depth_estimation",
            "source_dataset": "eth3d_low_res_two_view",
            "text": args.depth_prompt,
        }
        write_json(out_root / metadata_rel, metadata)
        rows.append(metadata)
        split_lines.append(
            f"/{image_rel.as_posix()} /{depth_rel.as_posix()} /{mask_rel.as_posix()} "
            f"/{intrinsics_rel.as_posix()} {float(calib.intrinsics[0, 0]):.10f} {slug} low_res_two_view\n"
        )

    (out_root / "eth3d_test.txt").write_text("".join(split_lines), encoding="utf-8")
    (out_root / "eth3d_low_res_two_view.txt").write_text("".join(split_lines), encoding="utf-8")
    write_jsonl(out_root / "metadata.jsonl", rows)
    write_dataset_info(out_root, args, "eth3d_low_res_two_view_depth", len(rows), "stereo_derived_depth")
    if args.depth_tar_output is not None:
        write_prepared_tar(out_root, args.depth_tar_output)


def prepare_normals(args: argparse.Namespace, scenes: list[Eth3dScene]) -> None:
    out_root = args.normals_out_root.expanduser().resolve()
    reset_out_root(out_root, args.overwrite)
    for rel in ("images/low_res_two_view", "normals", "masks", "depth", "intrinsics", "metadata", "splits"):
        (out_root / rel).mkdir(parents=True, exist_ok=True)

    split_lines: list[str] = []
    rows: list[dict] = []
    for source_index, scene in enumerate(tqdm(scenes, desc="Preparing ETH3D derived normals"), start=1):
        slug = scene_slug(scene.scene)
        calib = parse_calibration(scene.calib_path, args.baseline_unit)
        disp = read_pfm(scene.disparity_path)
        if disp.ndim == 3:
            disp = disp[..., 0]
        mask_occ = np.asarray(Image.open(scene.mask_path).convert("L"), dtype=np.uint8) > 0
        valid_disp = np.isfinite(disp) & (disp > 0)
        depth_m = np.full(disp.shape, np.nan, dtype=np.float32)
        depth_m[valid_disp] = float(calib.intrinsics[0, 0]) * calib.baseline_m / disp[valid_disp]
        valid_depth = mask_occ & valid_disp & np.isfinite(depth_m) & (depth_m > args.min_depth) & (depth_m < args.max_depth)
        normals, normal_valid = derive_normals_from_depth(
            depth=depth_m,
            intrinsics=calib.intrinsics,
            radius=args.gradient_radius,
            max_neighbor_depth_ratio=args.max_neighbor_depth_ratio,
        )
        normal_valid = erode_mask(valid_depth & normal_valid, args.mask_erosion_radius)
        valid_normal_pixels = int(normal_valid.sum())
        if valid_normal_pixels < args.min_normal_valid_pixels:
            continue

        normal_rgb = normals_to_rgb(
            normals, normal_valid, args.invalid_normal_rgb, args.normal_encoding
        )
        image_rel = Path("images/low_res_two_view") / f"{slug}.png"
        normal_rel = Path("normals") / f"{slug}_normal.png"
        mask_rel = Path("masks") / f"{slug}_normal_valid.png"
        depth_rel = Path("depth") / f"{slug}_depth_m.npy"
        intrinsics_rel = Path("intrinsics") / f"{slug}.txt"
        metadata_rel = Path("metadata") / f"{slug}.json"

        shutil.copy2(scene.image_path, out_root / image_rel)
        Image.fromarray(normal_rgb, mode="RGB").save(out_root / normal_rel)
        Image.fromarray(mask_to_uint8(normal_valid), mode="L").save(out_root / mask_rel)
        np.save(out_root / depth_rel, depth_m.astype(np.float32, copy=False))
        np.savetxt(out_root / intrinsics_rel, calib.intrinsics, fmt="%.10f")

        metadata = {
            "id": slug,
            "scene": scene.scene,
            "split": "test",
            "case": "low_res_two_view",
            "conditioning_image": image_rel.as_posix(),
            "image": normal_rel.as_posix(),
            "mask": mask_rel.as_posix(),
            "depth": depth_rel.as_posix(),
            "intrinsics": intrinsics_rel.as_posix(),
            "source_index": source_index,
            "source_image": str(scene.image_path),
            "source_disparity": str(scene.disparity_path),
            "source_mask": str(scene.mask_path),
            "source_calib": str(scene.calib_path),
            "baseline_raw": calib.baseline_raw,
            "baseline_unit": calib.baseline_unit,
            "baseline_m": calib.baseline_m,
            "normal_ground_truth": "derived_from_stereo_disparity_depth",
            "normal_encoding": args.normal_encoding,
            "normal_encoding_formula": normal_encoding_formula(args.normal_encoding),
            **CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL.metadata(),
            "geometry_version": CAMERA_FACING_GEOMETRY_DATA_VERSION,
            "valid_normal_pixels": valid_normal_pixels,
            "task": "surface_normal_estimation",
            "source_dataset": "eth3d_low_res_two_view",
            "text": args.normal_prompt,
        }
        write_json(out_root / metadata_rel, metadata)
        rows.append(metadata)
        split_lines.append(
            f"/{image_rel.as_posix()} /{normal_rel.as_posix()} /{mask_rel.as_posix()} "
            f"/{intrinsics_rel.as_posix()} {float(calib.intrinsics[0, 0]):.10f} {source_index} {slug} low_res_two_view\n"
        )

    (out_root / "eth3d_test.txt").write_text("".join(split_lines), encoding="utf-8")
    (out_root / "eth3d_low_res_two_view.txt").write_text("".join(split_lines), encoding="utf-8")
    write_jsonl(out_root / "metadata.jsonl", rows)
    write_dataset_info(out_root, args, "eth3d_low_res_two_view_surface_normals", len(rows), "derived_surface_normals")
    if args.normals_tar_output is not None:
        write_prepared_tar(out_root, args.normals_tar_output)


def write_dataset_info(out_root: Path, args: argparse.Namespace, dataset_name: str, count: int, target_type: str) -> None:
    is_normals = target_type == "derived_surface_normals"
    write_json(
        out_root / "dataset_info.json",
        {
            "dataset": dataset_name,
            "source_dataset": "ETH3D low-res two-view training data",
            "split": "test",
            "case": "low_res_two_view",
            "count": count,
            "target_type": target_type,
            "depth_unit": "meters",
            "baseline_unit_mode": args.baseline_unit,
            "normal_encoding": args.normal_encoding if is_normals else None,
            "normal_encoding_formula": normal_encoding_formula(args.normal_encoding) if is_normals else None,
            "geometry_version": CAMERA_FACING_GEOMETRY_DATA_VERSION if is_normals else None,
            "normal_orientation": "face_camera_ray" if is_normals else None,
            "stored_normal_convention": CANONICAL_NORMAL_CONVENTION if is_normals else None,
            "gt_to_canonical_permutation": [0, 1, 2] if is_normals else None,
            "gt_to_canonical_signs": [1, 1, 1] if is_normals else None,
            "normal_transform": (
                CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL.metadata()
                if is_normals else None
            ),
            "notes": (
                "Use only im0.png as model input. Depth is converted from ETH3D stereo disparity. "
                "Surface normals, when present, are derived from that converted depth and are not "
                "official ETH3D normal ground truth."
            ),
        },
    )


def parse_rgb(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise argparse.ArgumentTypeError("Expected RGB triplet like 128,128,128")
    return parts[0], parts[1], parts[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ETH3D low-res two-view depth and derived-normal eval tars.")
    parser.add_argument("--eth3d-root", type=Path, required=True, help="Directory containing ETH3D raw folders or .7z archives.")
    parser.add_argument("--task", choices=("depth", "surface_normals", "both"), default="both")
    parser.add_argument("--depth-out-root", type=Path, required=True)
    parser.add_argument("--normals-out-root", type=Path, required=True)
    parser.add_argument("--depth-tar-output", type=Path)
    parser.add_argument("--normals-tar-output", type=Path)
    parser.add_argument("--baseline-unit", choices=("auto", "meters", "millimeters"), default="auto")
    parser.add_argument("--min-depth", type=float, default=0.001)
    parser.add_argument("--max-depth", type=float, default=1000.0)
    parser.add_argument("--gradient-radius", type=int, default=2)
    parser.add_argument("--mask-erosion-radius", type=int, default=1)
    parser.add_argument("--max-neighbor-depth-ratio", type=float, default=0.1)
    parser.add_argument("--min-normal-valid-pixels", type=int, default=4096)
    parser.add_argument("--invalid-normal-rgb", type=parse_rgb, default=(128, 128, 128))
    parser.add_argument(
        "--normal-encoding",
        choices=SUPPORTED_NORMAL_ENCODINGS,
        default=VISION_BANANA_NORMAL_ENCODING,
    )
    parser.add_argument("--depth-prompt", default=DEPTH_PROMPT)
    parser.add_argument("--normal-prompt", default=NORMAL_PROMPT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = ensure_raw_layout(args)
    scenes = load_scenes(raw_root)
    if args.task in {"depth", "both"}:
        prepare_depth(args, scenes)
    if args.task in {"surface_normals", "both"}:
        prepare_normals(args, scenes)


if __name__ == "__main__":
    main()
