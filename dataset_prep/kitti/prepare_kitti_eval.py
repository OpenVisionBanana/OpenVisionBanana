#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

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
    "[TASK: DEPTH_ESTIMATION] Generate a metric depth visualization of the input outdoor driving scene "
    "using the Vision Banana RGB depth encoding. Output only the RGB depth visualization with no text, "
    "labels, legends, or extra objects."
)
NORMAL_PROMPT = VISION_BANANA_NORMAL_PROMPT


@dataclass(frozen=True)
class KittiSelectionRow:
    sample_id: str
    image_member: str
    depth_member: str
    intrinsics_member: str


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
    with tarfile.open(tmp_path, "w") as archive:
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


def load_kitti_depth_png(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr / 256.0


def read_intrinsics(path: Path) -> np.ndarray:
    values = [float(part) for part in path.read_text(encoding="utf-8").split()]
    if len(values) != 9:
        raise ValueError(f"Expected 9 KITTI intrinsics values in {path}, got {len(values)}")
    return np.asarray(values, dtype=np.float32).reshape(3, 3)


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


def normals_to_rgb(
    normals: np.ndarray,
    valid: np.ndarray,
    invalid_rgb: tuple[int, int, int],
    encoding: str = VISION_BANANA_NORMAL_ENCODING,
) -> np.ndarray:
    return encode_surface_normals_rgb(
        normals, valid, encoding=encoding, invalid_rgb=invalid_rgb
    )


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

    center = depth[y_slice, x_slice]
    if max_neighbor_depth_ratio > 0:
        eps = 1e-6
        neighbor_stack = np.stack(
            (
                depth[y_slice, slice(0, width - 2 * radius)],
                depth[y_slice, slice(2 * radius, width)],
                depth[slice(0, height - 2 * radius), x_slice],
                depth[slice(2 * radius, height), x_slice],
            ),
            axis=0,
        )
        rel_jump = np.max(np.abs(neighbor_stack - center[None, ...]) / np.maximum(center[None, ...], eps), axis=0)
        neighbor_valid &= rel_jump <= max_neighbor_depth_ratio

    normal_valid = np.zeros((height, width), dtype=bool)
    normal_valid[y_slice, x_slice] = good & neighbor_valid
    return CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL.apply(normals, normal_valid), normal_valid


def copy_member(archive: ZipFile, member: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as src, output_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def sample_id_from_image_member(member: str) -> str:
    return Path(member).stem.replace("_image_", "_frame_")


def expected_depth_member(image_member: str) -> str:
    name = Path(image_member).name.replace("_image_", "_groundtruth_depth_", 1)
    return "depth_selection/val_selection_cropped/groundtruth_depth/" + name


def expected_intrinsics_member(image_member: str) -> str:
    name = Path(image_member).with_suffix(".txt").name
    return "depth_selection/val_selection_cropped/intrinsics/" + name


def load_selection_rows(selection_zip: Path) -> list[KittiSelectionRow]:
    with ZipFile(selection_zip) as archive:
        names = set(archive.namelist())
        image_members = sorted(
            name
            for name in names
            if name.startswith("depth_selection/val_selection_cropped/image/") and name.endswith(".png")
        )
        rows: list[KittiSelectionRow] = []
        for image_member in image_members:
            depth_member = expected_depth_member(image_member)
            intrinsics_member = expected_intrinsics_member(image_member)
            missing = [member for member in (depth_member, intrinsics_member) if member not in names]
            if missing:
                raise FileNotFoundError(f"Missing KITTI selection pair member(s) for {image_member}: {missing}")
            rows.append(
                KittiSelectionRow(
                    sample_id=sample_id_from_image_member(image_member),
                    image_member=image_member,
                    depth_member=depth_member,
                    intrinsics_member=intrinsics_member,
                )
            )
    if not rows:
        raise FileNotFoundError(f"No KITTI val_selection_cropped images found in {selection_zip}")
    return rows


def write_dataset_info(out_root: Path, args: argparse.Namespace, dataset_name: str, count: int) -> None:
    is_normals = dataset_name == "kitti_depth_selection_surface_normals"
    write_json(
        out_root / "dataset_info.json",
        {
            "dataset": dataset_name,
            "source_dataset": "KITTI Depth Completion / Depth Prediction benchmark",
            "source_archive": str(args.selection_zip.expanduser().resolve()),
            "split": "test",
            "case": "depth_selection",
            "count": count,
            "depth_unit": "meters",
            "depth_png_scale": "meters = uint16_png / 256",
            "normal_ground_truth": "derived_from_annotated_depth" if is_normals else None,
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
                "KITTI does not provide official surface-normal ground truth; normal targets in this "
                "prepared dataset are derived from annotated depth and intrinsics."
                if is_normals
                else "Prepared from depth_selection/val_selection_cropped."
            ),
        },
    )


def prepare_depth(args: argparse.Namespace, rows: list[KittiSelectionRow]) -> None:
    out_root = args.depth_out_root.expanduser().resolve()
    reset_out_root(out_root, args.overwrite)
    for rel in ("images/depth_selection", "depth", "masks", "intrinsics", "splits"):
        (out_root / rel).mkdir(parents=True, exist_ok=True)

    split_lines: list[str] = []
    metadata_rows: list[dict] = []
    with ZipFile(args.selection_zip) as archive, tempfile.TemporaryDirectory(prefix="kitti-depth-") as tmp:
        tmp_root = Path(tmp)
        for row in tqdm(rows, desc="Preparing KITTI depth-selection depth"):
            depth_tmp = tmp_root / f"{row.sample_id}_depth.png"
            intrinsics_tmp = tmp_root / f"{row.sample_id}_K.txt"
            copy_member(archive, row.depth_member, depth_tmp)
            copy_member(archive, row.intrinsics_member, intrinsics_tmp)

            depth_m = load_kitti_depth_png(depth_tmp)
            valid = np.isfinite(depth_m) & (depth_m > args.min_depth) & (depth_m < args.max_depth)
            intrinsics = read_intrinsics(intrinsics_tmp)
            fx = float(intrinsics[0, 0])

            image_rel = Path("images/depth_selection") / f"{row.sample_id}.png"
            depth_rel = Path("depth") / f"{row.sample_id}_depth.png"
            mask_rel = Path("masks") / f"{row.sample_id}_valid.png"
            intrinsics_rel = Path("intrinsics") / f"{row.sample_id}.txt"

            copy_member(archive, row.image_member, out_root / image_rel)
            shutil.copy2(depth_tmp, out_root / depth_rel)
            shutil.copy2(intrinsics_tmp, out_root / intrinsics_rel)
            Image.fromarray(mask_to_uint8(valid), mode="L").save(out_root / mask_rel)

            split_lines.append(
                f"/{image_rel.as_posix()} /{depth_rel.as_posix()} /{mask_rel.as_posix()} "
                f"/{intrinsics_rel.as_posix()} {fx:.10f} {row.sample_id} depth_selection\n"
            )
            metadata_rows.append(
                {
                    "id": row.sample_id,
                    "split": "test",
                    "case": "depth_selection",
                    "conditioning_image": image_rel.as_posix(),
                    "depth": depth_rel.as_posix(),
                    "mask": mask_rel.as_posix(),
                    "intrinsics": intrinsics_rel.as_posix(),
                    "task": "depth_estimation",
                    "source_dataset": "kitti_depth_selection",
                    "depth_unit": "meters",
                    "text": args.depth_prompt,
                }
            )

    (out_root / "kitti_test.txt").write_text("".join(split_lines), encoding="utf-8")
    (out_root / "kitti_depth_selection.txt").write_text("".join(split_lines), encoding="utf-8")
    write_jsonl(out_root / "metadata.jsonl", metadata_rows)
    write_dataset_info(out_root, args, "kitti_depth_selection", len(rows))
    if args.depth_tar_output is not None:
        write_prepared_tar(out_root, args.depth_tar_output)


def prepare_normals(args: argparse.Namespace, rows: list[KittiSelectionRow]) -> None:
    out_root = args.normals_out_root.expanduser().resolve()
    reset_out_root(out_root, args.overwrite)
    for rel in ("images/depth_selection", "normals", "masks", "depth", "intrinsics", "splits"):
        (out_root / rel).mkdir(parents=True, exist_ok=True)

    split_lines: list[str] = []
    metadata_rows: list[dict] = []
    with ZipFile(args.selection_zip) as archive, tempfile.TemporaryDirectory(prefix="kitti-normals-") as tmp:
        tmp_root = Path(tmp)
        for source_index, row in enumerate(tqdm(rows, desc="Preparing KITTI derived normals"), start=1):
            depth_tmp = tmp_root / f"{row.sample_id}_depth.png"
            intrinsics_tmp = tmp_root / f"{row.sample_id}_K.txt"
            copy_member(archive, row.depth_member, depth_tmp)
            copy_member(archive, row.intrinsics_member, intrinsics_tmp)

            depth_m = load_kitti_depth_png(depth_tmp)
            intrinsics = read_intrinsics(intrinsics_tmp)
            valid_depth = np.isfinite(depth_m) & (depth_m > args.min_depth) & (depth_m < args.max_depth)
            normals, normal_valid = derive_normals_from_depth(
                depth=depth_m,
                intrinsics=intrinsics,
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
            fx = float(intrinsics[0, 0])

            image_rel = Path("images/depth_selection") / f"{row.sample_id}.png"
            normal_rel = Path("normals") / f"{row.sample_id}_normal.png"
            mask_rel = Path("masks") / f"{row.sample_id}_normal_mask.png"
            depth_rel = Path("depth") / f"{row.sample_id}_depth.png"
            intrinsics_rel = Path("intrinsics") / f"{row.sample_id}.txt"

            copy_member(archive, row.image_member, out_root / image_rel)
            Image.fromarray(normal_rgb, mode="RGB").save(out_root / normal_rel)
            Image.fromarray(mask_to_uint8(normal_valid), mode="L").save(out_root / mask_rel)
            shutil.copy2(depth_tmp, out_root / depth_rel)
            shutil.copy2(intrinsics_tmp, out_root / intrinsics_rel)

            split_lines.append(
                f"/{image_rel.as_posix()} /{normal_rel.as_posix()} /{mask_rel.as_posix()} "
                f"/{intrinsics_rel.as_posix()} {fx:.10f} {source_index} {row.sample_id} depth_selection\n"
            )
            metadata_rows.append(
                {
                    "id": row.sample_id,
                    "split": "test",
                    "case": "depth_selection",
                    "conditioning_image": image_rel.as_posix(),
                    "image": normal_rel.as_posix(),
                    "mask": mask_rel.as_posix(),
                    "depth": depth_rel.as_posix(),
                    "intrinsics": intrinsics_rel.as_posix(),
                    "task": "surface_normal_estimation",
                    "source_dataset": "kitti_depth_selection",
                    "normal_ground_truth": "derived_from_annotated_depth",
                    "valid_normal_pixels": valid_normal_pixels,
                    "normal_encoding": args.normal_encoding,
                    "normal_encoding_formula": normal_encoding_formula(args.normal_encoding),
                    **CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL.metadata(),
                    "geometry_version": CAMERA_FACING_GEOMETRY_DATA_VERSION,
                    "text": args.normal_prompt,
                }
            )

    (out_root / "kitti_test.txt").write_text("".join(split_lines), encoding="utf-8")
    (out_root / "kitti_depth_selection.txt").write_text("".join(split_lines), encoding="utf-8")
    write_jsonl(out_root / "metadata.jsonl", metadata_rows)
    write_dataset_info(out_root, args, "kitti_depth_selection_surface_normals", len(metadata_rows))
    if args.normals_tar_output is not None:
        write_prepared_tar(out_root, args.normals_tar_output)


def parse_rgb(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise argparse.ArgumentTypeError("Expected RGB triplet like 128,128,128")
    return parts[0], parts[1], parts[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare KITTI depth-selection depth and derived-normal eval tars.")
    parser.add_argument("--selection-zip", type=Path, required=True)
    parser.add_argument("--task", choices=("depth", "surface_normals", "both"), default="both")
    parser.add_argument("--depth-out-root", type=Path, required=True)
    parser.add_argument("--normals-out-root", type=Path, required=True)
    parser.add_argument("--depth-tar-output", type=Path)
    parser.add_argument("--normals-tar-output", type=Path)
    parser.add_argument("--min-depth", type=float, default=0.001)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--gradient-radius", type=int, default=2)
    parser.add_argument("--mask-erosion-radius", type=int, default=0)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.selection_zip = args.selection_zip.expanduser().resolve()
    if not args.selection_zip.is_file():
        raise FileNotFoundError(f"Missing KITTI depth-selection archive: {args.selection_zip}")

    rows = load_selection_rows(args.selection_zip)
    if args.task in {"depth", "both"}:
        prepare_depth(args, rows)
    if args.task in {"surface_normals", "both"}:
        prepare_normals(args, rows)


if __name__ == "__main__":
    main()
