#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
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
    "[TASK: DEPTH_ESTIMATION] Generate a metric depth visualization of the input synthetic outdoor "
    "driving scene using the Vision Banana RGB depth encoding. Output only the RGB depth visualization "
    "with no text, labels, legends, or extra objects."
)
NORMAL_PROMPT = VISION_BANANA_NORMAL_PROMPT


@dataclass(frozen=True)
class Vkitti2Row:
    scene: str
    variant: str
    camera: str
    frame: int
    sample_id: str
    image_member: str
    depth_member: str
    intrinsics: np.ndarray

    @property
    def case(self) -> str:
        return f"{self.variant}_{self.camera.lower()}"


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


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def parse_list(value: str) -> tuple[str, ...]:
    parts = [part.strip() for chunk in value.split(",") for part in chunk.split()]
    return tuple(part for part in parts if part)


def camera_index(camera: str) -> int:
    match = re.fullmatch(r"Camera_(\d+)", camera)
    if not match:
        raise ValueError(f"Expected camera name like Camera_0, got {camera!r}")
    return int(match.group(1))


def mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    return mask.astype(bool).astype(np.uint8) * 255


def load_depth_png_from_bytes(data: bytes, scale: float) -> np.ndarray:
    arr = np.asarray(Image.open(io.BytesIO(data)), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr / scale


def read_tar_member(archive: tarfile.TarFile, member: str) -> bytes:
    handle = archive.extractfile(member)
    if handle is None:
        raise FileNotFoundError(f"Could not read tar member: {member}")
    return handle.read()


def write_tar_member(archive: tarfile.TarFile, member: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = read_tar_member(archive, member)
    output_path.write_bytes(data)


def read_text_member(archive: tarfile.TarFile, member: str) -> str:
    return read_tar_member(archive, member).decode("utf-8")


def parse_intrinsics_table(text: str, camera: str) -> dict[int, np.ndarray]:
    cam_idx = camera_index(camera)
    rows: dict[int, np.ndarray] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("frame "):
            continue
        parts = line.split()
        if len(parts) < 6:
            raise ValueError(f"Malformed VKITTI2 intrinsic row at line {line_number}: {line}")
        frame = int(parts[0])
        row_camera = int(parts[1])
        if row_camera != cam_idx:
            continue
        fx, fy, cx, cy = [float(value) for value in parts[2:6]]
        rows[frame] = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    if not rows:
        raise ValueError(f"No intrinsic rows found for {camera}")
    return rows


def load_intrinsics(
    text_archive: tarfile.TarFile,
    scenes: tuple[str, ...],
    variants: tuple[str, ...],
    cameras: tuple[str, ...],
) -> dict[tuple[str, str, str, int], np.ndarray]:
    intrinsics: dict[tuple[str, str, str, int], np.ndarray] = {}
    for scene in scenes:
        for variant in variants:
            member = f"{scene}/{variant}/intrinsic.txt"
            text = read_text_member(text_archive, member)
            for camera in cameras:
                for frame, matrix in parse_intrinsics_table(text, camera).items():
                    intrinsics[(scene, variant, camera, frame)] = matrix
    return intrinsics


def discover_rows(
    rgb_archive: tarfile.TarFile,
    depth_archive: tarfile.TarFile,
    text_archive: tarfile.TarFile,
    scenes: tuple[str, ...],
    variants: tuple[str, ...],
    cameras: tuple[str, ...],
    limit: int | None,
) -> list[Vkitti2Row]:
    depth_members = set(depth_archive.getnames())
    intrinsics = load_intrinsics(text_archive, scenes, variants, cameras)
    wanted = {(scene, variant, camera) for scene in scenes for variant in variants for camera in cameras}

    rows: list[Vkitti2Row] = []
    pattern = re.compile(
        r"^(?P<scene>[^/]+)/(?P<variant>[^/]+)/frames/rgb/(?P<camera>Camera_\d+)/rgb_(?P<frame>\d+)\.jpg$"
    )
    for member in sorted(rgb_archive.getnames()):
        match = pattern.match(member)
        if not match:
            continue
        scene = match.group("scene")
        variant = match.group("variant")
        camera = match.group("camera")
        if (scene, variant, camera) not in wanted:
            continue
        frame = int(match.group("frame"))
        depth_member = f"{scene}/{variant}/frames/depth/{camera}/depth_{frame:05d}.png"
        if depth_member not in depth_members:
            raise FileNotFoundError(f"Missing VKITTI2 depth member for {member}: {depth_member}")
        matrix = intrinsics.get((scene, variant, camera, frame))
        if matrix is None:
            raise FileNotFoundError(f"Missing VKITTI2 intrinsics for {scene}/{variant}/{camera}/frame {frame}")
        sample_id = f"{scene}_{slug(variant)}_{camera}_{frame:05d}"
        rows.append(
            Vkitti2Row(
                scene=scene,
                variant=variant,
                camera=camera,
                frame=frame,
                sample_id=sample_id,
                image_member=member,
                depth_member=depth_member,
                intrinsics=matrix,
            )
        )
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise FileNotFoundError("No VKITTI2 RGB/depth pairs matched the requested scenes, variants, and cameras.")
    return rows


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
    points = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=-1)

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

    valid = np.zeros((height, width), dtype=bool)
    valid[y_slice, x_slice] = good & neighbor_valid
    return CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL.apply(normals, valid), valid


def normals_to_rgb(
    normals: np.ndarray,
    valid: np.ndarray,
    invalid_rgb: tuple[int, int, int],
    encoding: str = VISION_BANANA_NORMAL_ENCODING,
) -> np.ndarray:
    return encode_surface_normals_rgb(
        normals, valid, encoding=encoding, invalid_rgb=invalid_rgb
    )


def write_intrinsics(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = matrix.reshape(-1)
    path.write_text(" ".join(f"{float(value):.10g}" for value in values) + "\n", encoding="utf-8")


def write_dataset_info(out_root: Path, args: argparse.Namespace, dataset_name: str, count: int, task: str) -> None:
    write_json(
        out_root / "dataset_info.json",
        {
            "dataset": dataset_name,
            "source_dataset": "Virtual KITTI 2",
            "source_archives": {
                "rgb": str(args.rgb_tar.expanduser().resolve()),
                "depth": str(args.depth_tar.expanduser().resolve()),
                "textgt": str(args.textgt_tar.expanduser().resolve()),
            },
            "split": "test",
            "cases": sorted({f"{variant}_{camera.lower()}" for variant in parse_list(args.variants) for camera in parse_list(args.cameras)}),
            "scenes": list(parse_list(args.scenes)),
            "count": count,
            "task": task,
            "depth_unit": "meters",
            "depth_png_scale": f"meters = uint16_png / {args.depth_png_scale:g}",
            "normal_ground_truth": "derived_from_synthetic_depth_and_intrinsics" if task == "surface_normals" else None,
            "normal_encoding": args.normal_encoding if task == "surface_normals" else None,
            "normal_encoding_formula": (
                normal_encoding_formula(args.normal_encoding)
                if task == "surface_normals"
                else None
            ),
            "geometry_version": (
                CAMERA_FACING_GEOMETRY_DATA_VERSION
                if task == "surface_normals"
                else None
            ),
            "normal_orientation": "face_camera_ray" if task == "surface_normals" else None,
            "stored_normal_convention": CANONICAL_NORMAL_CONVENTION if task == "surface_normals" else None,
            "gt_to_canonical_permutation": [0, 1, 2] if task == "surface_normals" else None,
            "gt_to_canonical_signs": [1, 1, 1] if task == "surface_normals" else None,
            "normal_transform": (
                CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL.metadata()
                if task == "surface_normals" else None
            ),
            "notes": (
                "Virtual KITTI 2 is synthetic. Depth targets are dense synthetic ground truth. "
                "Surface-normal targets are derived during preprocessing, not official VKITTI2 normal maps."
            ),
        },
    )


def prepare_depth(args: argparse.Namespace, rows: list[Vkitti2Row]) -> None:
    out_root = args.depth_out_root.expanduser().resolve()
    reset_out_root(out_root, args.overwrite)
    for rel in ("images", "depth", "masks", "intrinsics", "splits"):
        (out_root / rel).mkdir(parents=True, exist_ok=True)

    split_lines: list[str] = []
    metadata_rows: list[dict] = []
    with tarfile.open(args.rgb_tar) as rgb_archive, tarfile.open(args.depth_tar) as depth_archive:
        for row in tqdm(rows, desc="Preparing VKITTI2 depth"):
            depth_bytes = read_tar_member(depth_archive, row.depth_member)
            depth_m = load_depth_png_from_bytes(depth_bytes, args.depth_png_scale)
            valid = np.isfinite(depth_m) & (depth_m > args.min_depth) & (depth_m < args.max_depth)
            fx = float(row.intrinsics[0, 0])

            image_rel = Path("images") / row.case / f"{row.sample_id}.jpg"
            depth_rel = Path("depth") / row.case / f"{row.sample_id}_depth.png"
            mask_rel = Path("masks") / row.case / f"{row.sample_id}_valid.png"
            intrinsics_rel = Path("intrinsics") / row.case / f"{row.sample_id}.txt"

            write_tar_member(rgb_archive, row.image_member, out_root / image_rel)
            (out_root / depth_rel).parent.mkdir(parents=True, exist_ok=True)
            (out_root / depth_rel).write_bytes(depth_bytes)
            (out_root / mask_rel).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask_to_uint8(valid), mode="L").save(out_root / mask_rel)
            write_intrinsics(out_root / intrinsics_rel, row.intrinsics)

            split_lines.append(
                f"/{image_rel.as_posix()} /{depth_rel.as_posix()} /{mask_rel.as_posix()} "
                f"/{intrinsics_rel.as_posix()} {fx:.10f} {row.sample_id} {row.case}\n"
            )
            metadata_rows.append(
                {
                    "id": row.sample_id,
                    "split": "test",
                    "case": row.case,
                    "scene": row.scene,
                    "variant": row.variant,
                    "camera": row.camera,
                    "frame": row.frame,
                    "conditioning_image": image_rel.as_posix(),
                    "depth": depth_rel.as_posix(),
                    "mask": mask_rel.as_posix(),
                    "intrinsics": intrinsics_rel.as_posix(),
                    "task": "depth_estimation",
                    "source_dataset": "virtual_kitti_2",
                    "depth_unit": "meters",
                    "depth_png_scale": args.depth_png_scale,
                    "text": args.depth_prompt,
                }
            )

    (out_root / "vkitti2_test.txt").write_text("".join(split_lines), encoding="utf-8")
    (out_root / "vkitti2.txt").write_text("".join(split_lines), encoding="utf-8")
    write_jsonl(out_root / "metadata.jsonl", metadata_rows)
    write_dataset_info(out_root, args, out_root.name, len(rows), "depth")
    if args.depth_tar_output is not None:
        write_prepared_tar(out_root, args.depth_tar_output)


def prepare_normals(args: argparse.Namespace, rows: list[Vkitti2Row]) -> None:
    out_root = args.normals_out_root.expanduser().resolve()
    reset_out_root(out_root, args.overwrite)
    for rel in ("images", "normals", "masks", "depth", "intrinsics", "splits"):
        (out_root / rel).mkdir(parents=True, exist_ok=True)

    split_lines: list[str] = []
    metadata_rows: list[dict] = []
    with tarfile.open(args.rgb_tar) as rgb_archive, tarfile.open(args.depth_tar) as depth_archive:
        for source_index, row in enumerate(tqdm(rows, desc="Preparing VKITTI2 derived normals"), start=1):
            depth_bytes = read_tar_member(depth_archive, row.depth_member)
            depth_m = load_depth_png_from_bytes(depth_bytes, args.depth_png_scale)
            valid_depth = np.isfinite(depth_m) & (depth_m > args.min_depth) & (depth_m < args.max_depth)
            normals, normal_valid = derive_normals_from_depth(
                depth_m,
                row.intrinsics,
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
            fx = float(row.intrinsics[0, 0])

            image_rel = Path("images") / row.case / f"{row.sample_id}.jpg"
            normal_rel = Path("normals") / row.case / f"{row.sample_id}_normal.png"
            mask_rel = Path("masks") / row.case / f"{row.sample_id}_normal_mask.png"
            depth_rel = Path("depth") / row.case / f"{row.sample_id}_depth.png"
            intrinsics_rel = Path("intrinsics") / row.case / f"{row.sample_id}.txt"

            write_tar_member(rgb_archive, row.image_member, out_root / image_rel)
            (out_root / normal_rel).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(normal_rgb, mode="RGB").save(out_root / normal_rel)
            (out_root / mask_rel).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask_to_uint8(normal_valid), mode="L").save(out_root / mask_rel)
            (out_root / depth_rel).parent.mkdir(parents=True, exist_ok=True)
            (out_root / depth_rel).write_bytes(depth_bytes)
            write_intrinsics(out_root / intrinsics_rel, row.intrinsics)

            split_lines.append(
                f"/{image_rel.as_posix()} /{normal_rel.as_posix()} /{mask_rel.as_posix()} "
                f"/{intrinsics_rel.as_posix()} {fx:.10f} {source_index} {row.sample_id} {row.case}\n"
            )
            metadata_rows.append(
                {
                    "id": row.sample_id,
                    "split": "test",
                    "case": row.case,
                    "scene": row.scene,
                    "variant": row.variant,
                    "camera": row.camera,
                    "frame": row.frame,
                    "conditioning_image": image_rel.as_posix(),
                    "image": normal_rel.as_posix(),
                    "mask": mask_rel.as_posix(),
                    "depth": depth_rel.as_posix(),
                    "intrinsics": intrinsics_rel.as_posix(),
                    "task": "surface_normal_estimation",
                    "source_dataset": "virtual_kitti_2",
                    "normal_ground_truth": "derived_from_synthetic_depth_and_intrinsics",
                    "valid_normal_pixels": valid_normal_pixels,
                    "normal_encoding": args.normal_encoding,
                    "normal_encoding_formula": normal_encoding_formula(args.normal_encoding),
                    **CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL.metadata(),
                    "geometry_version": CAMERA_FACING_GEOMETRY_DATA_VERSION,
                    "normal_orientation": "face_camera_ray",
                    "coordinate_convention": CANONICAL_NORMAL_CONVENTION,
                    "text": args.normal_prompt,
                }
            )

    (out_root / "vkitti2_test.txt").write_text("".join(split_lines), encoding="utf-8")
    (out_root / "vkitti2.txt").write_text("".join(split_lines), encoding="utf-8")
    write_jsonl(out_root / "metadata.jsonl", metadata_rows)
    write_dataset_info(out_root, args, out_root.name, len(metadata_rows), "surface_normals")
    if args.normals_tar_output is not None:
        write_prepared_tar(out_root, args.normals_tar_output)


def parse_rgb(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise argparse.ArgumentTypeError("Expected RGB triplet like 128,128,128")
    return parts[0], parts[1], parts[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Virtual KITTI 2 depth and derived-normal eval tars.")
    parser.add_argument("--rgb-tar", type=Path, required=True)
    parser.add_argument("--depth-tar", type=Path, required=True)
    parser.add_argument("--textgt-tar", type=Path, required=True)
    parser.add_argument("--task", choices=("depth", "surface_normals", "both"), default="both")
    parser.add_argument("--scenes", default="Scene01 Scene02 Scene06 Scene18 Scene20")
    parser.add_argument("--variants", default="clone")
    parser.add_argument("--cameras", default="Camera_0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--depth-out-root", type=Path, required=True)
    parser.add_argument("--normals-out-root", type=Path, required=True)
    parser.add_argument("--depth-tar-output", type=Path)
    parser.add_argument("--normals-tar-output", type=Path)
    parser.add_argument("--depth-png-scale", type=float, default=100.0)
    parser.add_argument("--min-depth", type=float, default=0.001)
    parser.add_argument("--max-depth", type=float, default=655.35)
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
    args.rgb_tar = args.rgb_tar.expanduser().resolve()
    args.depth_tar = args.depth_tar.expanduser().resolve()
    args.textgt_tar = args.textgt_tar.expanduser().resolve()
    for path in (args.rgb_tar, args.depth_tar, args.textgt_tar):
        if not path.is_file():
            raise FileNotFoundError(f"Missing VKITTI2 source archive: {path}")

    scenes = parse_list(args.scenes)
    variants = parse_list(args.variants)
    cameras = parse_list(args.cameras)
    with tarfile.open(args.rgb_tar) as rgb_archive, tarfile.open(args.depth_tar) as depth_archive, tarfile.open(args.textgt_tar) as text_archive:
        rows = discover_rows(rgb_archive, depth_archive, text_archive, scenes, variants, cameras, args.limit)

    if args.task in {"depth", "both"}:
        prepare_depth(args, rows)
    if args.task in {"surface_normals", "both"}:
        prepare_normals(args, rows)


if __name__ == "__main__":
    main()
