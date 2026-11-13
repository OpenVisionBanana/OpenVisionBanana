#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import zipfile

import h5py
import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from common.geometry_protocol_prompts import VISION_BANANA_NORMAL_PROMPT
from common.hypersim_zip import ZipMember, zip_frames
from common.surface_normal_encoding import (
    SUPPORTED_NORMAL_ENCODINGS,
    VISION_BANANA_NORMAL_ENCODING,
    encode_surface_normals_rgb,
    normal_encoding_formula,
)
from dataset_prep.depth.encoding import (
    DEPTH_ENCODING_GRAYSCALE,
    DEPTH_ENCODING_VISION_BANANA_RGB,
    DEPTH_ENCODINGS,
    encode_depth_to_vision_banana_rgb,
    valid_depth_mask,
)
from dataset_prep.geometry_conventions import (
    CANONICAL_DEPTH_QUANTITY,
    CANONICAL_NORMAL_CONVENTION,
    GEOMETRY_DATA_VERSION,
    HYPERSIM_NATIVE_TO_CANONICAL,
    geometry_data_version_for_normal_encoding,
    range_to_projective_z_from_ray_matrix,
    resize_normal_field,
    resize_scalar_field,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda iterable, **_: iterable


DEPTH_PROMPT = (
    "[TASK: DEPTH_ESTIMATION] Generate a metric depth visualization of the input indoor scene "
    "using the Vision Banana RGB depth encoding. Output only the RGB depth visualization with no "
    "text, labels, legends, or extra objects."
)
NORMAL_PROMPT = VISION_BANANA_NORMAL_PROMPT
IMAGE_RESAMPLING = getattr(Image, "Resampling", Image)


@dataclass(frozen=True)
class HypersimFrame:
    sample_id: str
    scene: str
    camera: str
    frame: str
    scene_dir: Path
    rgb_path: Path
    depth_path: Path
    normal_path: Path
    split: str = "train"


def first_hdf5_dataset(handle: h5py.File) -> np.ndarray:
    datasets: list[h5py.Dataset] = []

    def collect(_: str, obj: h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets.append(obj)

    handle.visititems(collect)
    if not datasets:
        raise ValueError("HDF5 file contains no datasets")
    return np.asarray(datasets[0])


def read_hdf5_array(path: Path) -> np.ndarray:
    if isinstance(path, ZipMember):
        # At most one HDF5 channel per worker is decompressed into memory.
        with zipfile.ZipFile(path.archive) as archive:
            with io.BytesIO(archive.read(path.member)) as stream, h5py.File(stream, "r") as handle:
                return np.asarray(handle["dataset"]) if "dataset" in handle else first_hdf5_dataset(handle)
    with h5py.File(path, "r") as handle:
        if "dataset" in handle:
            return np.asarray(handle["dataset"])
        return first_hdf5_dataset(handle)


def sort_key(path: Path) -> tuple[str, str]:
    return (path.parent.as_posix(), path.name)


def frame_token_from_color(path: Path) -> str:
    name = path.name
    suffix = ".color.hdf5"
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected Hypersim color filename: {path}")
    return name[: -len(suffix)]


def scene_dir_from_final_dir(final_dir: Path) -> Path:
    parent = final_dir.parent
    if parent.name == "images":
        return parent.parent
    return parent


def discover_frames(raw_root: Path) -> list[HypersimFrame]:
    raw_root = raw_root.expanduser().resolve()
    color_paths = sorted(raw_root.glob("**/scene_cam_*_final_hdf5/frame.*.color.hdf5"), key=sort_key)
    frames: list[HypersimFrame] = []
    for rgb_path in color_paths:
        final_dir = rgb_path.parent
        camera = final_dir.name.replace("_final_hdf5", "")
        geometry_dir = final_dir.parent / f"{camera}_geometry_hdf5"
        frame = frame_token_from_color(rgb_path)
        depth_path = geometry_dir / f"{frame}.depth_meters.hdf5"
        normal_path = geometry_dir / f"{frame}.normal_cam.hdf5"
        if not depth_path.is_file() or not normal_path.is_file():
            continue
        scene_dir = scene_dir_from_final_dir(final_dir)
        scene = scene_dir.name
        frame_id = frame.replace(".", "_")
        sample_id = f"hypersim_{scene}_{camera}_{frame_id}"
        frames.append(
            HypersimFrame(
                sample_id=sample_id,
                scene=scene,
                camera=camera,
                frame=frame,
                scene_dir=scene_dir,
                rgb_path=rgb_path,
                depth_path=depth_path,
                normal_path=normal_path,
            )
        )
    existing = {f.sample_id for f in frames}
    for scene_dir, camera, frame, paths in zip_frames(raw_root, (".depth_meters.hdf5", ".normal_cam.hdf5")):
        scene = scene_dir.name
        sample_id = f"hypersim_{scene}_{camera}_{frame.replace('.', '_')}"
        if sample_id in existing:
            raise ValueError(f"Duplicate extracted/archived Hypersim frame: {sample_id}")
        existing.add(sample_id)
        frames.append(HypersimFrame(sample_id, scene, camera, frame, raw_root / str(scene_dir), *paths))
    return sorted(frames, key=lambda f: (f.scene, f.camera, f.frame))


def allocate_counts(
    frames_by_scene: dict[str, list[HypersimFrame]],
    *,
    samples_per_scene: int | None,
    target_samples: int | None,
    all_frames: bool,
    seed: int,
) -> dict[str, int]:
    scenes = sorted(frames_by_scene)
    if all_frames:
        counts = {scene: len(frames_by_scene[scene]) for scene in scenes}
    elif target_samples is not None:
        base = target_samples // max(1, len(scenes))
        remainder = target_samples % max(1, len(scenes))
        counts = {scene: base for scene in scenes}
        rng = random.Random(seed)
        for scene in rng.sample(scenes, remainder):
            counts[scene] += 1
    else:
        per_scene = samples_per_scene if samples_per_scene is not None else 25
        counts = {scene: per_scene for scene in scenes}

    changed = True
    while changed:
        changed = False
        shortage = 0
        for scene in scenes:
            available = len(frames_by_scene[scene])
            if counts[scene] > available:
                shortage += counts[scene] - available
                counts[scene] = available
                changed = True
        if shortage <= 0:
            break
        for scene in scenes:
            if shortage <= 0:
                break
            available = len(frames_by_scene[scene])
            spare = available - counts[scene]
            if spare <= 0:
                continue
            add = min(spare, shortage)
            counts[scene] += add
            shortage -= add
            changed = True
    return counts


def sample_frames(
    frames: list[HypersimFrame],
    *,
    samples_per_scene: int | None,
    target_samples: int | None,
    all_frames: bool,
    seed: int,
    val_ratio: float,
) -> list[HypersimFrame]:
    frames_by_scene: dict[str, list[HypersimFrame]] = {}
    for frame in frames:
        frames_by_scene.setdefault(frame.scene, []).append(frame)
    for scene_frames in frames_by_scene.values():
        scene_frames.sort(key=lambda item: (item.camera, item.frame))

    counts = allocate_counts(
        frames_by_scene,
        samples_per_scene=samples_per_scene,
        target_samples=target_samples,
        all_frames=all_frames,
        seed=seed,
    )
    rng = random.Random(seed)
    sampled: list[HypersimFrame] = []
    for scene in sorted(frames_by_scene):
        scene_frames = list(frames_by_scene[scene])
        rng.shuffle(scene_frames)
        selected = sorted(scene_frames[: counts[scene]], key=lambda item: (item.camera, item.frame))
        sampled.extend(selected)

    scenes = sorted({frame.scene for frame in sampled})
    rng = random.Random(seed)
    rng.shuffle(scenes)
    num_val = max(1, int(round(len(scenes) * val_ratio))) if scenes and val_ratio > 0 else 0
    if num_val >= len(scenes):
        num_val = max(0, len(scenes) - 1)
    val_scenes = set(scenes[:num_val])
    # type(frame) rather than HypersimFrame: the segmentation prep reuses this
    # sampler with its own frame dataclass (same scene/camera/frame fields, different
    # channel paths), so the split must be stamped back onto whatever type came in.
    return [
        type(frame)(**{**frame.__dict__, "split": "val" if frame.scene in val_scenes else "train"})
        for frame in sorted(sampled, key=lambda item: (item.scene, item.camera, item.frame))
    ]


def rgb_array_to_image(
    rgb: np.ndarray, valid_mask: np.ndarray | None = None
) -> Image.Image:
    """Apply Hypersim's official percentile/gamma tone-mapping recipe."""
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[0] in {3, 4} and rgb.shape[-1] not in {3, 4}:
        rgb = np.moveaxis(rgb, 0, -1)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=-1)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise ValueError(f"Expected RGB-like array, got shape={rgb.shape}")
    rgb = rgb[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.asarray(rgb, dtype=np.float32)
        valid = np.isfinite(rgb).all(axis=-1)
        if valid_mask is not None:
            if np.asarray(valid_mask).shape != rgb.shape[:2]:
                raise ValueError(
                    f"RGB tone-map mask shape mismatch: {np.asarray(valid_mask).shape} "
                    f"vs {rgb.shape[:2]}"
                )
            valid &= np.asarray(valid_mask, dtype=bool)
        brightness = (
            0.30 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]
        )
        if np.any(valid):
            percentile = float(np.percentile(brightness[valid], 90))
            scale = 0.0 if percentile < 1e-4 else 0.8**2.2 / percentile
        else:
            scale = 1.0
        mapped = np.power(np.maximum(scale * rgb, 0.0), 1.0 / 2.2)
        rgb_u8 = np.rint(np.clip(mapped, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        rgb_u8 = np.clip(
            np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0), 0, 255
        ).astype(np.uint8)
    return Image.fromarray(rgb_u8, mode="RGB")


def resize_to_max_side(image: Image.Image, max_side: int, resample: int) -> Image.Image:
    if max_side <= 0:
        return image
    width, height = image.size
    largest = max(width, height)
    if largest <= max_side:
        return image
    scale = max_side / largest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, resample=resample)


def resize_float_array(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if array.shape[:2] == (size[1], size[0]):
        return array.astype(np.float32, copy=False)
    values = np.where(np.isfinite(array), array, 0.0).astype(np.float32, copy=False)
    weights = np.isfinite(array).astype(np.float32, copy=False)
    value_img = Image.fromarray(values, mode="F").resize(size, resample=IMAGE_RESAMPLING.BILINEAR)
    weight_img = Image.fromarray(weights, mode="F").resize(size, resample=IMAGE_RESAMPLING.BILINEAR)
    resized_values = np.asarray(value_img, dtype=np.float32)
    resized_weights = np.asarray(weight_img, dtype=np.float32)
    out = np.full(resized_values.shape, np.nan, dtype=np.float32)
    valid = resized_weights > 1e-6
    out[valid] = resized_values[valid] / resized_weights[valid]
    return out


def resize_normal_array(normals: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if normals.shape[:2] == (size[1], size[0]):
        return normals.astype(np.float32, copy=False)
    channels = [resize_float_array(normals[..., idx], size) for idx in range(3)]
    resized = np.stack(channels, axis=-1).astype(np.float32, copy=False)
    norm = np.linalg.norm(resized, axis=-1, keepdims=True)
    return np.divide(resized, norm, out=np.zeros_like(resized), where=norm > 1e-8)


def parse_rgb(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("RGB value must have the form R,G,B")
    rgb = tuple(int(part) for part in parts)
    if any(part < 0 or part > 255 for part in rgb):
        raise argparse.ArgumentTypeError("RGB channels must be in [0, 255]")
    return rgb


def depth_to_grayscale(
    depth: np.ndarray,
    valid: np.ndarray,
    *,
    min_depth: float,
    max_depth: float,
    invalid_value: int,
) -> np.ndarray:
    clipped = np.clip(depth, min_depth, max_depth)
    norm = np.log(clipped / min_depth) / np.log(max_depth / min_depth)
    gray = np.rint((1.0 - norm) * 255.0).clip(0, 255).astype(np.uint8)
    gray[~valid] = np.uint8(invalid_value)
    return gray


def normals_to_rgb(
    normals: np.ndarray,
    valid: np.ndarray,
    invalid_rgb: tuple[int, int, int],
    encoding: str = VISION_BANANA_NORMAL_ENCODING,
) -> np.ndarray:
    return encode_surface_normals_rgb(
        normals, valid, encoding=encoding, invalid_rgb=invalid_rgb
    )


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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


def load_camera_parameter_matrices(path: Path) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    with path.expanduser().open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scene = row.get("scene_name")
            if not scene:
                continue
            matrices[scene] = np.asarray(
                [
                    [float(row[f"M_cam_from_uv_{axis}{column}"]) for column in range(3)]
                    for axis in range(3)
                ],
                dtype=np.float64,
            )
    if not matrices:
        raise ValueError(f"No Hypersim M_cam_from_uv matrices found in {path}")
    return matrices


def canonicalize_hypersim_depth(
    depth_range_m: np.ndarray, frame: HypersimFrame, args: argparse.Namespace
) -> np.ndarray:
    if args.legacy_euclidean_depth:
        return depth_range_m
    matrix = args.camera_parameter_matrices.get(frame.scene)
    if matrix is None:
        raise KeyError(
            f"Missing Hypersim camera parameters for scene {frame.scene!r}; "
            "canonical range-to-z conversion cannot proceed"
        )
    return range_to_projective_z_from_ray_matrix(depth_range_m, matrix)


def write_depth_sample(frame: HypersimFrame, args: argparse.Namespace) -> dict[str, Any] | None:
    rgb_array = read_hdf5_array(frame.rgb_path)
    depth = np.asarray(read_hdf5_array(frame.depth_path), dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth, got shape={depth.shape}")
    rgb = resize_to_max_side(
        rgb_array_to_image(rgb_array, np.isfinite(depth) & (depth > 0)),
        args.max_side,
        IMAGE_RESAMPLING.LANCZOS,
    )
    depth = canonicalize_hypersim_depth(depth, frame, args)
    native_valid = np.isfinite(depth)
    depth, _ = resize_scalar_field(depth, native_valid, rgb.size)
    valid = valid_depth_mask(depth, min_depth=args.min_depth, max_depth=args.max_depth)
    if float(valid.mean()) < args.min_valid_fraction:
        return None

    if args.depth_encoding == DEPTH_ENCODING_VISION_BANANA_RGB:
        target_rgb = encode_depth_to_vision_banana_rgb(depth, valid_mask=valid, invalid_color=args.depth_invalid_rgb)
        target = Image.fromarray(target_rgb, mode="RGB")
        target_name = f"{frame.sample_id}_depth_vb_rgb.png"
        preprocess_target = "raw_depth_resize_then_encode"
    elif args.depth_encoding == DEPTH_ENCODING_GRAYSCALE:
        gray = depth_to_grayscale(
            depth,
            valid,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            invalid_value=args.depth_invalid_rgb[0],
        )
        target = Image.fromarray(gray, mode="L").convert("RGB")
        target_name = f"{frame.sample_id}_depth.png"
        preprocess_target = "bilinear"
    else:
        raise ValueError(f"Unsupported depth encoding: {args.depth_encoding}")

    cond_rel = Path("conditioning_images") / f"{frame.sample_id}_input.png"
    target_rel = Path("target_images") / target_name
    depth_rel = Path("depth_raw") / f"{frame.sample_id}.npy"
    mask_rel = Path("valid_masks") / f"{frame.sample_id}.png"

    (args.depth_out_root / cond_rel).parent.mkdir(parents=True, exist_ok=True)
    (args.depth_out_root / target_rel).parent.mkdir(parents=True, exist_ok=True)
    (args.depth_out_root / depth_rel).parent.mkdir(parents=True, exist_ok=True)
    (args.depth_out_root / mask_rel).parent.mkdir(parents=True, exist_ok=True)
    rgb.save(args.depth_out_root / cond_rel)
    target.save(args.depth_out_root / target_rel)
    depth_to_save = depth.astype(np.float32, copy=True)
    depth_to_save[~valid] = np.nan
    np.save(args.depth_out_root / depth_rel, depth_to_save)
    Image.fromarray(valid.astype(np.uint8) * 255, mode="L").save(args.depth_out_root / mask_rel)

    return {
        "conditioning_image": cond_rel.as_posix(),
        "image": target_rel.as_posix(),
        "depth_raw": depth_rel.as_posix(),
        "valid_mask": mask_rel.as_posix(),
        "text": args.depth_prompt,
        "task": "depth_estimation",
        "source_dataset": "hypersim",
        "depth_encoding": args.depth_encoding,
        "id": frame.sample_id,
        "preprocess_target": preprocess_target,
        "metadata": {
            "scene": frame.scene,
            "camera": frame.camera,
            "frame": frame.frame,
            "split": frame.split,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "raw_depth_units": "meters",
            "native_depth_quantity": "euclidean_camera_range_meters",
            "canonical_depth_quantity": (
                "legacy_euclidean_camera_range_meters"
                if args.legacy_euclidean_depth
                else CANONICAL_DEPTH_QUANTITY
            ),
            "geometry_version": (
                "legacy" if args.legacy_euclidean_depth else GEOMETRY_DATA_VERSION
            ),
            "target_preparation": preprocess_target,
            "conditioning_tone_mapping": "hypersim_official_p90_0.8_gamma_2.2",
        },
    }


def write_normal_sample(frame: HypersimFrame, args: argparse.Namespace) -> dict[str, Any] | None:
    rgb_array = read_hdf5_array(frame.rgb_path)
    depth = np.asarray(read_hdf5_array(frame.depth_path), dtype=np.float32)
    normals = np.asarray(read_hdf5_array(frame.normal_path), dtype=np.float32)
    if normals.ndim == 3 and normals.shape[0] == 3 and normals.shape[-1] != 3:
        normals = np.moveaxis(normals, 0, -1)
    if depth.ndim != 2 or normals.ndim != 3 or normals.shape[-1] != 3:
        raise ValueError(f"Expected depth HxW and normals HxWx3, got depth={depth.shape}, normals={normals.shape}")
    rgb = resize_to_max_side(
        rgb_array_to_image(rgb_array, np.isfinite(depth) & (depth > 0)),
        args.max_side,
        IMAGE_RESAMPLING.LANCZOS,
    )

    depth = canonicalize_hypersim_depth(depth, frame, args)
    native_normal_norm = np.linalg.norm(normals, axis=-1)
    native_normal_valid = (
        np.isfinite(normals).all(axis=-1)
        & (native_normal_norm > 0.5)
        & (native_normal_norm < 1.5)
    )
    normals = HYPERSIM_NATIVE_TO_CANONICAL.apply(normals, native_normal_valid)
    depth, _ = resize_scalar_field(depth, np.isfinite(depth), rgb.size)
    normals, resized_normal_valid = resize_normal_field(
        normals, native_normal_valid, rgb.size
    )
    valid_depth = valid_depth_mask(depth, min_depth=args.min_depth, max_depth=args.max_depth)
    normal_norm = np.linalg.norm(normals, axis=-1)
    valid_normal = (
        resized_normal_valid
        & np.isfinite(normals).all(axis=-1)
        & (normal_norm > 0.5)
        & (normal_norm < 1.5)
    )
    valid = valid_depth & valid_normal
    if float(valid.mean()) < args.min_valid_fraction:
        return None

    cond_rel = Path("conditioning_images") / f"{frame.sample_id}_input.png"
    target_rel = Path("target_images") / f"{frame.sample_id}_normal.png"
    mask_rel = Path("masks") / f"{frame.sample_id}_mask.png"
    raw_rel = Path("raw_normals") / f"{frame.sample_id}_normals.npz"

    (args.normals_out_root / cond_rel).parent.mkdir(parents=True, exist_ok=True)
    (args.normals_out_root / target_rel).parent.mkdir(parents=True, exist_ok=True)
    (args.normals_out_root / mask_rel).parent.mkdir(parents=True, exist_ok=True)
    rgb.save(args.normals_out_root / cond_rel)
    Image.fromarray(
        normals_to_rgb(
            normals, valid, args.invalid_normal_rgb, args.normal_encoding
        ),
        mode="RGB",
    ).save(args.normals_out_root / target_rel)
    Image.fromarray(valid.astype(np.uint8) * 255, mode="L").save(args.normals_out_root / mask_rel)
    raw_rel_str = None
    if args.write_raw_normals:
        (args.normals_out_root / raw_rel).parent.mkdir(parents=True, exist_ok=True)
        raw_normals = normals.astype(np.float16 if args.raw_normal_dtype == "float16" else np.float32)
        np.savez_compressed(args.normals_out_root / raw_rel, normals=raw_normals, valid_mask=valid.astype(np.bool_))
        raw_rel_str = raw_rel.as_posix()

    row = {
        "conditioning_image": cond_rel.as_posix(),
        "image": target_rel.as_posix(),
        "loss_mask": mask_rel.as_posix(),
        "mask": mask_rel.as_posix(),
        "text": args.normals_prompt,
        "task": "surface_normal_estimation",
        "source_dataset": "hypersim",
        "id": frame.sample_id,
        "preprocess_target": "resize_xyz_then_renormalize_then_encode",
        "metadata": {
            "scene": frame.scene,
            "camera": frame.camera,
            "frame": frame.frame,
            "split": frame.split,
            "normal_encoding": args.normal_encoding,
            "normal_encoding_formula": normal_encoding_formula(args.normal_encoding),
            "coordinate_convention": CANONICAL_NORMAL_CONVENTION,
            **HYPERSIM_NATIVE_TO_CANONICAL.metadata(),
            "geometry_version": geometry_data_version_for_normal_encoding(
                args.normal_encoding
            ),
            "native_depth_quantity": "euclidean_camera_range_meters",
            "canonical_depth_quantity": (
                "legacy_euclidean_camera_range_meters"
                if args.legacy_euclidean_depth
                else CANONICAL_DEPTH_QUANTITY
            ),
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "conditioning_tone_mapping": "hypersim_official_p90_0.8_gamma_2.2",
        },
    }
    if raw_rel_str is not None:
        row["raw_normals"] = raw_rel_str
    return row


def prepare_task(
    frames: list[HypersimFrame],
    *,
    task: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    writer = write_depth_sample if task == "depth" else write_normal_sample

    def process(frame: HypersimFrame) -> dict[str, Any] | None:
        try:
            return writer(frame, args)
        except Exception as exc:
            if args.strict:
                raise
            print(f"WARNING: skipped {task} {frame.sample_id}: {exc}", flush=True)
            return None

    if args.num_workers <= 1:
        rows = [process(frame) for frame in tqdm(frames, desc=f"Preparing Hypersim {task}")]
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            rows = list(
                tqdm(
                    executor.map(process, frames),
                    total=len(frames),
                    desc=f"Preparing Hypersim {task} ({args.num_workers} workers)",
                )
            )
    return [row for row in rows if row is not None]


def write_task_outputs(
    out_root: Path,
    tar_path: Path | None,
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    source_root: Path,
    args: argparse.Namespace,
) -> None:
    train_rows = [row for row in rows if row["metadata"].get("split") == "train"]
    val_rows = [row for row in rows if row["metadata"].get("split") == "val"]
    metadata_rows = rows if args.include_val_in_metadata else train_rows
    write_jsonl(out_root / "metadata.jsonl", metadata_rows)
    write_jsonl(out_root / "manifests" / "train.jsonl", train_rows)
    write_jsonl(out_root / "manifests" / "val.jsonl", val_rows)
    write_json(
        out_root / "dataset_info.json",
        {
            "dataset": dataset_name,
            "source": str(source_root),
            "rows_written": len(rows),
            "metadata_rows": len(metadata_rows),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "metadata_contains_val": bool(args.include_val_in_metadata),
            "samples_per_scene": args.samples_per_scene,
            "target_samples": args.target_samples,
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "max_side": args.max_side,
            "geometry_version": (
                "legacy" if args.legacy_euclidean_depth else GEOMETRY_DATA_VERSION
            ),
            "canonical_depth_quantity": (
                "legacy_euclidean_camera_range_meters"
                if args.legacy_euclidean_depth
                else CANONICAL_DEPTH_QUANTITY
            ),
            "canonical_normal_convention": CANONICAL_NORMAL_CONVENTION,
            "normal_encoding": args.normal_encoding,
            "normal_geometry_version": geometry_data_version_for_normal_encoding(
                args.normal_encoding
            ),
        },
    )
    if tar_path is not None:
        write_prepared_tar(out_root, tar_path)


def clean_output(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite to rebuild.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def maybe_delete_raw_scenes(frames: list[HypersimFrame], raw_root: Path) -> None:
    if any(isinstance(frame.rgb_path, ZipMember) for frame in frames):
        raise ValueError("Deleting archived Hypersim sources is prohibited")
    raw_root = raw_root.resolve()
    scene_dirs = sorted({frame.scene_dir.resolve() for frame in frames}, key=lambda path: path.as_posix())
    for scene_dir in scene_dirs:
        if scene_dir == raw_root or raw_root not in scene_dir.parents:
            print(f"WARNING: refusing to delete scene outside raw root: {scene_dir}", flush=True)
            continue
        shutil.rmtree(scene_dir)
        print(f"Deleted raw Hypersim scene: {scene_dir}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Hypersim depth and surface-normal training pairs.")
    parser.add_argument("--raw-root", type=Path, required=True, help="Extracted Hypersim root containing scene folders.")
    parser.add_argument("--depth-out-root", type=Path, required=True)
    parser.add_argument("--normals-out-root", type=Path, required=True)
    parser.add_argument("--depth-prepared-tar", type=Path, default=None)
    parser.add_argument("--normals-prepared-tar", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", choices=["depth", "normals"], default=["depth", "normals"])
    parser.add_argument("--samples-per-scene", type=int, default=25)
    parser.add_argument("--target-samples", type=int, default=None)
    parser.add_argument("--all-frames", action="store_true", help="Use every complete RGB/depth/normal frame.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.80)
    parser.add_argument("--max-side", type=int, default=0)
    parser.add_argument(
        "--camera-parameters-csv",
        type=Path,
        default=None,
        help=(
            "Official Hypersim metadata_camera_parameters.csv containing "
            "M_cam_from_uv. Required for canonical projective-z depth."
        ),
    )
    parser.add_argument(
        "--legacy-euclidean-depth",
        action="store_true",
        help="Historical-only opt-out: retain Hypersim Euclidean range instead of canonical z.",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--depth-encoding", choices=DEPTH_ENCODINGS, default=DEPTH_ENCODING_VISION_BANANA_RGB)
    parser.add_argument("--depth-invalid-rgb", type=parse_rgb, default=(0, 0, 0))
    parser.add_argument("--invalid-normal-rgb", type=parse_rgb, default=(128, 128, 128))
    parser.add_argument(
        "--normal-encoding",
        choices=SUPPORTED_NORMAL_ENCODINGS,
        default=VISION_BANANA_NORMAL_ENCODING,
    )
    parser.add_argument("--depth-prompt", default=DEPTH_PROMPT)
    parser.add_argument("--normals-prompt", default=NORMAL_PROMPT)
    parser.add_argument("--write-raw-normals", action="store_true")
    parser.add_argument("--raw-normal-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--include-val-in-metadata", action="store_true")
    parser.add_argument("--allow-empty", action="store_true", help="Write empty metadata/tars instead of failing when no rows pass filters.")
    parser.add_argument("--delete-raw-scenes-after-convert", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.all_frames:
        args.target_samples = None
        args.samples_per_scene = None
    elif args.target_samples is not None and args.samples_per_scene is not None:
        args.samples_per_scene = None
    if args.min_depth <= 0 or args.max_depth <= args.min_depth:
        raise ValueError("--max-depth must be greater than --min-depth > 0")
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1)")
    if not 0.0 <= args.min_valid_fraction <= 1.0:
        raise ValueError("--min-valid-fraction must be in [0, 1]")
    if args.legacy_euclidean_depth:
        args.camera_parameter_matrices = {}
    else:
        if args.camera_parameters_csv is None:
            raise ValueError(
                "--camera-parameters-csv is required unless --legacy-euclidean-depth is set"
            )
        args.camera_parameter_matrices = load_camera_parameter_matrices(
            args.camera_parameters_csv
        )

    args.raw_root = args.raw_root.expanduser().resolve()
    args.depth_out_root = args.depth_out_root.expanduser().resolve()
    args.normals_out_root = args.normals_out_root.expanduser().resolve()
    if "depth" in args.tasks:
        clean_output(args.depth_out_root, args.overwrite)
    if "normals" in args.tasks:
        clean_output(args.normals_out_root, args.overwrite)

    frames = discover_frames(args.raw_root)
    if not frames:
        raise RuntimeError(f"No complete Hypersim RGB/depth/normal frames found under: {args.raw_root}")
    sampled = sample_frames(
        frames,
        samples_per_scene=args.samples_per_scene,
        target_samples=args.target_samples,
        all_frames=args.all_frames,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )
    if not sampled:
        raise RuntimeError("No Hypersim frames selected after sampling.")

    if "depth" in args.tasks:
        depth_rows = prepare_task(sampled, task="depth", args=args)
        if not depth_rows and not args.allow_empty:
            raise RuntimeError("No Hypersim depth rows were written.")
        write_task_outputs(
            args.depth_out_root,
            args.depth_prepared_tar,
            depth_rows,
            dataset_name="hypersim_depth",
            source_root=args.raw_root,
            args=args,
        )

    if "normals" in args.tasks:
        normal_rows = prepare_task(sampled, task="normals", args=args)
        if not normal_rows and not args.allow_empty:
            raise RuntimeError("No Hypersim surface-normal rows were written.")
        write_task_outputs(
            args.normals_out_root,
            args.normals_prepared_tar,
            normal_rows,
            dataset_name="hypersim_surface_normals",
            source_root=args.raw_root,
            args=args,
        )

    if args.delete_raw_scenes_after_convert:
        maybe_delete_raw_scenes(sampled, args.raw_root)


if __name__ == "__main__":
    main()
