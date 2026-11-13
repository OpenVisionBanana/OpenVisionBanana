#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from common.geometry_protocol_prompts import (
    CANONICAL_DEPTH_PROMPT,
    VISION_BANANA_NORMAL_PROMPT,
)
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
    VISION_BANANA_GEOMETRY_DATA_VERSION,
    resize_normal_field,
    resize_scalar_field,
    taskonomy_normal_transform,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda iterable, **_: iterable


DEPTH_PROMPT = CANONICAL_DEPTH_PROMPT
NORMAL_PROMPT = VISION_BANANA_NORMAL_PROMPT

IMAGE_RESAMPLING = getattr(Image, "Resampling", Image)
TASKONOMY_DEPTH_SCALE = 512.0
TASKONOMY_DEPTH_MAX_METERS = 128.0


@dataclass(frozen=True)
class TaskonomyFrame:
    building: str
    frame_key: str
    sample_id: str
    point_id: str
    view_id: str
    rgb_member: str
    depth_member: str
    normal_member: str
    split: str


def parse_rgb(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("RGB value must have the form R,G,B")
    rgb = tuple(int(part) for part in parts)
    if any(part < 0 or part > 255 for part in rgb):
        raise argparse.ArgumentTypeError("RGB channels must be in [0, 255]")
    return rgb


def stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def sanitize_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def frame_key_from_member(member_name: str, suffix: str) -> str | None:
    path = Path(member_name)
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        return None
    stem = path.stem
    if not stem.endswith(suffix):
        return None
    return stem[: -len(suffix)]


def split_point_view(frame_key: str) -> tuple[str, str]:
    match = re.match(r"^(point_[^_]+)_?(view_[^_]+)$", frame_key)
    if match:
        return match.group(1), match.group(2)
    parts = frame_key.split("_")
    if len(parts) >= 4 and parts[0] == "point" and "view" in parts:
        view_index = parts.index("view")
        return "_".join(parts[:view_index]), "_".join(parts[view_index:])
    return frame_key, "view_unknown"


def list_tar_members(tar_path: Path, suffix: str) -> dict[str, str]:
    members: dict[str, str] = {}
    with tarfile.open(tar_path, "r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            frame_key = frame_key_from_member(member.name, suffix)
            if frame_key is not None:
                members[frame_key] = member.name
    return members


def discover_buildings(raw_tar_root: Path, depth_type: str) -> list[str]:
    rgb_buildings = {path.name[: -len("_rgb.tar")] for path in raw_tar_root.glob("*_rgb.tar")}
    depth_suffix = f"_{depth_type}.tar"
    depth_buildings = {path.name[: -len(depth_suffix)] for path in raw_tar_root.glob(f"*_{depth_type}.tar")}
    normal_buildings = {path.name[: -len("_normal.tar")] for path in raw_tar_root.glob("*_normal.tar")}
    return sorted(rgb_buildings & depth_buildings & normal_buildings)


def read_building_list(path: Path) -> list[str]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]


def select_buildings(all_buildings: list[str], args: argparse.Namespace) -> list[str]:
    if args.building_list is not None:
        requested = read_building_list(args.building_list)
        missing = sorted(set(requested) - set(all_buildings))
        if missing:
            raise FileNotFoundError(f"Missing Taskonomy tars for buildings: {', '.join(missing[:20])}")
        return requested
    if args.num_buildings == "all":
        return all_buildings
    count = int(args.num_buildings)
    if count <= 0:
        raise ValueError("--num-buildings must be 'all' or a positive integer")
    if count >= len(all_buildings):
        return all_buildings
    rng = random.Random(args.seed)
    return sorted(rng.sample(all_buildings, count))


def allocate_counts(
    available: dict[str, int],
    *,
    samples_per_building: int | None,
    target_samples: int | None,
    all_frames: bool,
    seed: int,
) -> dict[str, int]:
    buildings = sorted(available)
    if all_frames:
        counts = dict(available)
    elif target_samples is not None:
        base = target_samples // max(1, len(buildings))
        remainder = target_samples % max(1, len(buildings))
        counts = {building: base for building in buildings}
        rng = random.Random(seed)
        for building in rng.sample(buildings, remainder):
            counts[building] += 1
    else:
        per_building = samples_per_building if samples_per_building is not None else 50
        counts = {building: per_building for building in buildings}

    shortage = 0
    for building in buildings:
        if counts[building] > available[building]:
            shortage += counts[building] - available[building]
            counts[building] = available[building]
    for building in buildings:
        if shortage <= 0:
            break
        spare = available[building] - counts[building]
        if spare <= 0:
            continue
        add = min(spare, shortage)
        counts[building] += add
        shortage -= add
    return counts


def assign_split(seed: int, building: str, frame_key: str, train_ratio: float) -> str:
    if train_ratio >= 1.0:
        return "train"
    rng = random.Random(stable_seed(seed, f"{building}:{frame_key}:split"))
    return "train" if rng.random() < train_ratio else "val"


def discover_frames(args: argparse.Namespace) -> list[TaskonomyFrame]:
    all_buildings = discover_buildings(args.raw_tar_root, args.depth_type)
    if not all_buildings:
        raise RuntimeError(f"No complete Taskonomy rgb/{args.depth_type}/normal tar triplets found in {args.raw_tar_root}")
    selected_buildings = select_buildings(all_buildings, args)

    candidates_by_building: dict[str, list[TaskonomyFrame]] = {}
    for building in tqdm(selected_buildings, desc="Indexing Taskonomy building tars"):
        rgb_tar = args.raw_tar_root / f"{building}_rgb.tar"
        depth_tar = args.raw_tar_root / f"{building}_{args.depth_type}.tar"
        normal_tar = args.raw_tar_root / f"{building}_normal.tar"
        rgb_members = list_tar_members(rgb_tar, "_domain_rgb")
        depth_members = list_tar_members(depth_tar, f"_domain_{args.depth_type}")
        normal_members = list_tar_members(normal_tar, "_domain_normal")
        keys = sorted(set(rgb_members) & set(depth_members) & set(normal_members))
        frames: list[TaskonomyFrame] = []
        for frame_key in keys:
            point_id, view_id = split_point_view(frame_key)
            sample_id = f"taskonomy_{sanitize_token(building)}_{sanitize_token(point_id)}_{sanitize_token(view_id)}"
            frames.append(
                TaskonomyFrame(
                    building=building,
                    frame_key=frame_key,
                    sample_id=sample_id,
                    point_id=point_id,
                    view_id=view_id,
                    rgb_member=rgb_members[frame_key],
                    depth_member=depth_members[frame_key],
                    normal_member=normal_members[frame_key],
                    split=assign_split(args.seed, building, frame_key, args.train_ratio),
                )
            )
        if frames:
            candidates_by_building[building] = frames

    if not candidates_by_building:
        raise RuntimeError("No complete Taskonomy RGB/depth/normal frame triplets found.")

    counts = allocate_counts(
        {building: len(frames) for building, frames in candidates_by_building.items()},
        samples_per_building=args.samples_per_building,
        target_samples=args.target_samples,
        all_frames=args.all_frames,
        seed=args.seed,
    )
    sampled: list[TaskonomyFrame] = []
    for building in sorted(candidates_by_building):
        frames = list(candidates_by_building[building])
        rng = random.Random(stable_seed(args.seed, building))
        rng.shuffle(frames)
        selected = sorted(frames[: counts[building]], key=lambda frame: frame.frame_key)
        sampled.extend(selected)
    return sampled


def load_image_from_tar(archive: tarfile.TarFile, member_name: str) -> Image.Image:
    extracted = archive.extractfile(member_name)
    if extracted is None:
        raise FileNotFoundError(f"Could not extract tar member: {member_name}")
    with Image.open(io.BytesIO(extracted.read())) as image:
        return ImageOps.exif_transpose(image).copy()


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
    finite = np.isfinite(array)
    values = np.where(finite, array, 0.0).astype(np.float32, copy=False)
    weights = finite.astype(np.float32, copy=False)
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
        out = normals.astype(np.float32, copy=False)
    else:
        channels = [resize_float_array(normals[..., idx], size) for idx in range(3)]
        out = np.stack(channels, axis=-1).astype(np.float32, copy=False)
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    return np.divide(out, norm, out=np.zeros_like(out), where=norm > 1e-8)


def decode_taskonomy_depth(image: Image.Image) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    depth = array.astype(np.float32) / TASKONOMY_DEPTH_SCALE
    depth[depth <= 0] = np.nan
    depth[depth > TASKONOMY_DEPTH_MAX_METERS] = np.nan
    return depth


def decode_taskonomy_normals(image: Image.Image) -> np.ndarray:
    rgb_image = image.convert("RGB")
    rgb = np.asarray(rgb_image, dtype=np.float32)
    normals = (rgb - 127.0) / 127.0
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    source_valid = (np.asarray(rgb_image) != 0).any(axis=-1, keepdims=True)
    normals = np.divide(normals, norm, out=np.zeros_like(normals), where=norm > 1e-8)
    normals[~np.broadcast_to(source_valid, normals.shape)] = 0.0
    return normals.astype(np.float32, copy=False)


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


def write_depth_sample(
    frame: TaskonomyFrame,
    rgb: Image.Image,
    depth_image: Image.Image,
    normal_image: Image.Image,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    rgb = resize_to_max_side(rgb.convert("RGB"), args.max_side, IMAGE_RESAMPLING.LANCZOS)
    native_depth = decode_taskonomy_depth(depth_image)
    native_valid = valid_depth_mask(
        native_depth, min_depth=args.min_depth, max_depth=args.max_depth
    )
    depth, resized_valid = resize_scalar_field(native_depth, native_valid, rgb.size)
    valid = resized_valid & valid_depth_mask(
        depth, min_depth=args.min_depth, max_depth=args.max_depth
    )
    if float(valid.mean()) < args.min_valid_fraction:
        return None

    if args.depth_encoding == DEPTH_ENCODING_VISION_BANANA_RGB:
        target = Image.fromarray(
            encode_depth_to_vision_banana_rgb(depth, valid_mask=valid, invalid_color=args.depth_invalid_rgb),
            mode="RGB",
        )
        target_name = f"{frame.sample_id}_depth_vb_rgb.png"
        target_preparation = "taskonomy_depth_decode_metric_resize_then_vision_banana_rgb_encode"
    elif args.depth_encoding == DEPTH_ENCODING_GRAYSCALE:
        target = Image.fromarray(
            depth_to_grayscale(
                depth,
                valid,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                invalid_value=args.depth_invalid_rgb[0],
            ),
            mode="L",
        ).convert("RGB")
        target_name = f"{frame.sample_id}_depth.png"
        target_preparation = "taskonomy_depth_decode_metric_resize_then_grayscale_log_encode"
    else:
        raise ValueError(f"Unsupported depth encoding: {args.depth_encoding}")

    cond_rel = Path("conditioning_images") / f"{frame.sample_id}_input.png"
    target_rel = Path("target_images") / target_name
    depth_rel = Path("depth_raw") / f"{frame.sample_id}.npy"
    mask_rel = Path("valid_masks") / f"{frame.sample_id}.png"
    for rel in (cond_rel, target_rel, depth_rel, mask_rel):
        (args.depth_out_root / rel).parent.mkdir(parents=True, exist_ok=True)

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
        "loss_mask": mask_rel.as_posix(),
        "valid_mask": mask_rel.as_posix(),
        "text": args.depth_prompt,
        "task": "depth_estimation",
        "source_dataset": "taskonomy",
        "depth_encoding": args.depth_encoding,
        "id": frame.sample_id,
        "scene": frame.building,
        "preprocess_target": "bilinear",
        "metadata": {
            "building": frame.building,
            "scene": frame.building,
            "point_id": frame.point_id,
            "view_id": frame.view_id,
            "frame_key": frame.frame_key,
            "split": frame.split,
            "depth_type": args.depth_type,
            "source_depth_encoding": "Taskonomy uint16 PNG in 1/512 meter units",
            "source_depth_scale": TASKONOMY_DEPTH_SCALE,
            "source_depth_max_meters": TASKONOMY_DEPTH_MAX_METERS,
            "raw_depth_units": "meters",
            "geometry_version": (
                "legacy_unverified" if args.legacy_geometry else GEOMETRY_DATA_VERSION
            ),
            "canonical_depth_quantity": (
                CANONICAL_DEPTH_QUANTITY
                if args.depth_type == "depth_zbuffer"
                else "legacy_taskonomy_euclidean_depth"
            ),
            "target_depth_transform": "vision_banana_power_rgb",
            "target_preparation": target_preparation,
            "resize_policy": "metric_field_then_encode_with_hard_mask",
            "mask_policy": "hard_nearest_neighbor",
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
        },
    }


def write_normal_sample(
    frame: TaskonomyFrame,
    rgb: Image.Image,
    depth_image: Image.Image,
    normal_image: Image.Image,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    rgb = resize_to_max_side(rgb.convert("RGB"), args.max_side, IMAGE_RESAMPLING.LANCZOS)
    native_depth = decode_taskonomy_depth(depth_image)
    native_depth_valid = valid_depth_mask(
        native_depth, min_depth=args.min_depth, max_depth=args.max_depth
    )
    depth, resized_depth_valid = resize_scalar_field(
        native_depth, native_depth_valid, rgb.size
    )
    native_normals = decode_taskonomy_normals(normal_image)
    native_normal_norm = np.linalg.norm(native_normals, axis=-1)
    native_normal_valid = (
        np.isfinite(native_normals).all(axis=-1)
        & (native_normal_norm > 0.5)
        & (native_normal_norm < 1.5)
    )
    transform = taskonomy_normal_transform()
    canonical_normals = transform.apply(native_normals, native_normal_valid)
    normals, resized_normal_valid = resize_normal_field(
        canonical_normals, native_normal_valid, rgb.size
    )
    valid = (
        resized_depth_valid
        & valid_depth_mask(depth, min_depth=args.min_depth, max_depth=args.max_depth)
        & resized_normal_valid
    )
    if float(valid.mean()) < args.min_valid_fraction:
        return None

    cond_rel = Path("conditioning_images") / f"{frame.sample_id}_input.png"
    target_rel = Path("target_images") / f"{frame.sample_id}_normal.png"
    mask_rel = Path("masks") / f"{frame.sample_id}_mask.png"
    raw_rel = Path("raw_normals") / f"{frame.sample_id}_normals.npz"
    for rel in (cond_rel, target_rel, mask_rel):
        (args.normals_out_root / rel).parent.mkdir(parents=True, exist_ok=True)

    rgb.save(args.normals_out_root / cond_rel)
    Image.fromarray(normals_to_rgb(
        normals,
        valid,
        args.invalid_normal_rgb,
        args.normal_encoding,
    ), mode="RGB").save(
        args.normals_out_root / target_rel
    )
    Image.fromarray(valid.astype(np.uint8) * 255, mode="L").save(args.normals_out_root / mask_rel)

    row = {
        "conditioning_image": cond_rel.as_posix(),
        "image": target_rel.as_posix(),
        "loss_mask": mask_rel.as_posix(),
        "mask": mask_rel.as_posix(),
        "text": args.normals_prompt,
        "task": "surface_normal_estimation",
        "source_dataset": "taskonomy",
        "id": frame.sample_id,
        "scene": frame.building,
        "preprocess_target": "bilinear",
        "metadata": {
            "building": frame.building,
            "scene": frame.building,
            "point_id": frame.point_id,
            "view_id": frame.view_id,
            "frame_key": frame.frame_key,
            "split": frame.split,
            "normal_source": "native_taskonomy",
            "source_normal_encoding": "Taskonomy RGB normal PNG decoded as (rgb - 127) / 127",
            "normal_encoding": args.normal_encoding,
            "normal_encoding_formula": normal_encoding_formula(args.normal_encoding),
            "coordinate_convention": CANONICAL_NORMAL_CONVENTION,
            "canonical_normal_convention": CANONICAL_NORMAL_CONVENTION,
            "geometry_version": (
                VISION_BANANA_GEOMETRY_DATA_VERSION
                if args.normal_encoding == VISION_BANANA_NORMAL_ENCODING
                else GEOMETRY_DATA_VERSION
            ),
            "normal_axis_permutation": list(transform.permutation),
            "normal_axis_signs": list(transform.signs),
            "native_normal_convention": transform.native_convention,
            "normal_transform_evidence": transform.evidence,
            "quarantine_status": "released_by_paired_depth_48_transform_audit",
            "target_preparation": "canonical_xyz_resize_renormalize_then_encode",
            "resize_policy": "vector_field_renormalize_then_encode_with_hard_mask",
            "mask_policy": "hard_nearest_neighbor",
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
        },
    }
    if args.write_raw_normals:
        (args.normals_out_root / raw_rel).parent.mkdir(parents=True, exist_ok=True)
        raw_normals = normals.astype(np.float16 if args.raw_normal_dtype == "float16" else np.float32)
        np.savez_compressed(args.normals_out_root / raw_rel, normals=raw_normals, valid_mask=valid.astype(np.bool_))
        row["raw_normals"] = raw_rel.as_posix()
    return row


def prepare_frames(frames: list[TaskonomyFrame], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frames_by_building: dict[str, list[TaskonomyFrame]] = {}
    for frame in frames:
        frames_by_building.setdefault(frame.building, []).append(frame)

    depth_rows: list[dict[str, Any]] = []
    normal_rows: list[dict[str, Any]] = []
    for building in tqdm(sorted(frames_by_building), desc="Preparing Taskonomy buildings"):
        rgb_tar = args.raw_tar_root / f"{building}_rgb.tar"
        depth_tar = args.raw_tar_root / f"{building}_{args.depth_type}.tar"
        normal_tar = args.raw_tar_root / f"{building}_normal.tar"
        with tarfile.open(rgb_tar, "r:*") as rgb_archive, tarfile.open(depth_tar, "r:*") as depth_archive, tarfile.open(
            normal_tar, "r:*"
        ) as normal_archive:
            for frame in tqdm(frames_by_building[building], desc=building, leave=False):
                try:
                    rgb = load_image_from_tar(rgb_archive, frame.rgb_member)
                    depth_image = load_image_from_tar(depth_archive, frame.depth_member)
                    normal_image = load_image_from_tar(normal_archive, frame.normal_member)
                    if "depth" in args.tasks:
                        row = write_depth_sample(frame, rgb, depth_image, normal_image, args)
                        if row is not None:
                            depth_rows.append(row)
                    if "normals" in args.tasks:
                        row = write_normal_sample(frame, rgb, depth_image, normal_image, args)
                        if row is not None:
                            normal_rows.append(row)
                except Exception as exc:
                    if args.strict:
                        raise
                    print(f"WARNING: skipped {frame.sample_id}: {exc}", flush=True)
    return depth_rows, normal_rows


def write_task_outputs(
    out_root: Path,
    tar_path: Path | None,
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    args: argparse.Namespace,
) -> None:
    train_rows = [row for row in rows if row["metadata"].get("split") == "train"]
    val_rows = [row for row in rows if row["metadata"].get("split") == "val"]
    metadata_rows = rows if args.include_val_in_metadata else train_rows
    write_jsonl(out_root / "metadata.jsonl", metadata_rows)
    write_jsonl(out_root / "manifests" / "train.jsonl", train_rows)
    write_jsonl(out_root / "manifests" / "val.jsonl", val_rows)
    output_geometry_version = (
        "legacy_unverified"
        if args.legacy_geometry
        else (
            VISION_BANANA_GEOMETRY_DATA_VERSION
            if dataset_name == "taskonomy_surface_normals"
            and args.normal_encoding == VISION_BANANA_NORMAL_ENCODING
            else GEOMETRY_DATA_VERSION
        )
    )
    write_json(
        out_root / "dataset_info.json",
        {
            "dataset": dataset_name,
            "source": str(args.raw_tar_root),
            "rows_written": len(rows),
            "metadata_rows": len(metadata_rows),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "metadata_contains_val": bool(args.include_val_in_metadata),
            "depth_type": args.depth_type,
            "num_buildings": args.num_buildings,
            "samples_per_building": args.samples_per_building,
            "target_samples": args.target_samples,
            "all_frames": args.all_frames,
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "max_side": args.max_side,
            "source_depth_encoding": "uint16_png_1_over_512_meter",
            "target_depth_encoding": args.depth_encoding,
            "geometry_version": output_geometry_version,
            "canonical_depth_quantity": (
                CANONICAL_DEPTH_QUANTITY
                if dataset_name == "taskonomy_depth"
                else None
            ),
            "canonical_normal_convention": (
                CANONICAL_NORMAL_CONVENTION
                if dataset_name == "taskonomy_surface_normals"
                else None
            ),
            "normal_encoding": (
                args.normal_encoding
                if dataset_name == "taskonomy_surface_normals"
                else None
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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Taskonomy depth and surface-normal training pairs.")
    parser.add_argument("--raw-tar-root", type=Path, required=True, help="Directory containing Taskonomy modality tars.")
    parser.add_argument("--depth-out-root", type=Path, required=True)
    parser.add_argument("--normals-out-root", type=Path, required=True)
    parser.add_argument("--depth-prepared-tar", type=Path, default=None)
    parser.add_argument("--normals-prepared-tar", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", choices=["depth", "normals"], default=["depth", "normals"])
    parser.add_argument("--depth-type", choices=["depth_zbuffer", "depth_euclidean"], default="depth_zbuffer")
    parser.add_argument("--num-buildings", default="all", help="'all' or a positive integer sampled by seed.")
    parser.add_argument("--building-list", type=Path, default=None)
    parser.add_argument("--samples-per-building", type=int, default=50)
    parser.add_argument("--target-samples", type=int, default=None)
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=1.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.70)
    parser.add_argument("--max-side", type=int, default=0)
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--legacy-geometry",
        action="store_true",
        help=(
            "Historical-only escape hatch for unverified Taskonomy normals or "
            "Euclidean depth. Outputs are labeled legacy_unverified."
        ),
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.all_frames:
        args.target_samples = None
        args.samples_per_building = None
    elif args.target_samples is not None:
        args.samples_per_building = None
    if args.min_depth <= 0 or args.max_depth <= args.min_depth:
        raise ValueError("--max-depth must be greater than --min-depth > 0")
    if not 0.0 < args.train_ratio <= 1.0:
        raise ValueError("--train-ratio must be in (0, 1]")
    if not 0.0 <= args.min_valid_fraction <= 1.0:
        raise ValueError("--min-valid-fraction must be in [0, 1]")
    if "normals" in args.tasks and not args.legacy_geometry:
        taskonomy_normal_transform()
    if args.depth_type == "depth_euclidean" and not args.legacy_geometry:
        raise RuntimeError(
            "Taskonomy depth_euclidean is not canonical projective z and camera "
            "intrinsics are not available in this preparer. Use depth_zbuffer for "
            "geometry_canonical_v2 or --legacy-geometry for historical reproduction."
        )

    args.raw_tar_root = args.raw_tar_root.expanduser().resolve()
    args.depth_out_root = args.depth_out_root.expanduser().resolve()
    args.normals_out_root = args.normals_out_root.expanduser().resolve()
    if "depth" in args.tasks:
        clean_output(args.depth_out_root, args.overwrite)
    if "normals" in args.tasks:
        clean_output(args.normals_out_root, args.overwrite)

    frames = discover_frames(args)
    if not frames:
        raise RuntimeError("No Taskonomy frames selected after sampling.")
    depth_rows, normal_rows = prepare_frames(frames, args)

    if "depth" in args.tasks:
        if not depth_rows:
            raise RuntimeError("No Taskonomy depth rows were written.")
        write_task_outputs(
            args.depth_out_root,
            args.depth_prepared_tar,
            depth_rows,
            dataset_name="taskonomy_depth",
            args=args,
        )
    if "normals" in args.tasks:
        if not normal_rows:
            raise RuntimeError("No Taskonomy surface-normal rows were written.")
        write_task_outputs(
            args.normals_out_root,
            args.normals_prepared_tar,
            normal_rows,
            dataset_name="taskonomy_surface_normals",
            args=args,
        )

    print("Taskonomy preparation finished.")
    print(f"  selected frames: {len(frames)}")
    if "depth" in args.tasks:
        print(f"  depth rows:      {len(depth_rows)}")
        print(f"  depth root:      {args.depth_out_root}")
        if args.depth_prepared_tar:
            print(f"  depth tar:       {args.depth_prepared_tar}")
    if "normals" in args.tasks:
        print(f"  normal rows:     {len(normal_rows)}")
        print(f"  normal root:     {args.normals_out_root}")
        if args.normals_prepared_tar:
            print(f"  normal tar:      {args.normals_prepared_tar}")


if __name__ == "__main__":
    main()
