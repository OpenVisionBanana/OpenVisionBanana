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
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.geometry_protocol_prompts import (
    CANONICAL_DEPTH_PROMPT,
    CANONICAL_NORMAL_PROMPT,
)
from common.surface_normal_encoding import (
    LEGACY_DIRECT_NORMAL_ENCODING,
    normal_encoding_formula,
)
from dataset_prep.geometry_conventions import (
    CAMERA_FACING_GEOMETRY_DATA_VERSION,
    CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL,
    CANONICAL_NORMAL_CONVENTION,
    GEOMETRY_DATA_VERSION,
    IMAGE_Y_DOWN_Z_AWAY_TO_CANONICAL,
    orient_normals_toward_camera,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda iterable, **_: iterable


DEPTH_PROMPT = CANONICAL_DEPTH_PROMPT
NORMAL_PROMPT = CANONICAL_NORMAL_PROMPT

APPEARANCE_AUG_PREFIXES = (
    "blur_gaussian_",
    "contrast_norm",
    "gamma_",
    "hue_",
    "noise_gaussian_",
    "noise_sp_",
    "saturation_",
    "swap_channels_",
)
GEOMETRIC_AUGMENTATIONS = ("flip_lr", "flip_ud")


@dataclass(frozen=True)
class CoreSample:
    index_1based: int
    sample_id: str
    rgb: np.ndarray
    depth_m: np.ndarray
    valid_mask: np.ndarray
    calib: np.ndarray
    edges: np.ndarray
    mask_floor: np.ndarray
    mask_table: np.ndarray
    mask_wall: np.ndarray
    mask_floor_paras: np.ndarray
    mask_table_paras: np.ndarray
    mask_wall_paras: np.ndarray
    normal_rgb: np.ndarray
    normal_mask: np.ndarray


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


def depth_to_uint16_mm(depth_m: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros(depth_m.shape, dtype=np.uint16)
    good = valid & np.isfinite(depth_m) & (depth_m > 0)
    out[good] = np.clip(np.rint(depth_m[good] * 1000.0), 0, 65535).astype(np.uint16)
    return out


def mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    return mask.astype(np.uint8) * 255


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


def depth_upper_bound(args: argparse.Namespace) -> float | None:
    if args.depth_validity_protocol == "ibims_official" or args.max_depth <= 0:
        return None
    return float(args.max_depth)


def build_valid_depth_mask(depth: np.ndarray, mask_invalid: np.ndarray, mask_transp: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    valid = mask_invalid & mask_transp & np.isfinite(depth) & (depth > args.min_depth)
    upper_bound = depth_upper_bound(args)
    if upper_bound is not None:
        valid &= depth < upper_bound
    return valid


def plane_parameter_rows(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 7), dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr.reshape(arr.shape[0], -1)


def normals_to_rgb(normals: np.ndarray, valid: np.ndarray, invalid_rgb: tuple[int, int, int]) -> np.ndarray:
    rgb = np.zeros((*normals.shape[:2], 3), dtype=np.uint8)
    rgb[...] = np.asarray(invalid_rgb, dtype=np.uint8)
    encoded = np.rint((np.clip(normals, -1.0, 1.0) * 0.5 + 0.5) * 255.0)
    rgb[valid] = np.clip(encoded[valid], 0, 255).astype(np.uint8)
    return rgb


def derive_normals_from_depth(
    depth: np.ndarray,
    calib: np.ndarray,
    radius: int,
    *,
    camera_facing: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if radius < 1:
        raise ValueError("--gradient-radius must be >= 1")

    height, width = depth.shape
    fx = float(calib[0, 0])
    fy = float(calib[1, 1])
    cx = float(calib[2, 0])
    cy = float(calib[2, 1])

    u = np.arange(1, width + 1, dtype=np.float32)[None, :]
    v = np.arange(1, height + 1, dtype=np.float32)[:, None]
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
    dx = points[y_slice, slice(2 * radius, width)] - points[y_slice, slice(0, width - 2 * radius)]
    dy = points[slice(2 * radius, height), x_slice] - points[slice(0, height - 2 * radius), x_slice]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    good = np.isfinite(norm[..., 0]) & (norm[..., 0] > 1e-8)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 1e-8)
    if camera_facing:
        n = orient_normals_toward_camera(n, points[y_slice, x_slice], good)
    else:
        n[n[..., 2] < 0] *= -1.0
    normals[y_slice, x_slice] = n

    neighbor_valid = (
        valid_depth[y_slice, slice(2 * radius, width)]
        & valid_depth[y_slice, slice(0, width - 2 * radius)]
        & valid_depth[slice(2 * radius, height), x_slice]
        & valid_depth[slice(0, height - 2 * radius), x_slice]
    )
    normal_valid = np.zeros((height, width), dtype=bool)
    normal_valid[y_slice, x_slice] = good & neighbor_valid
    return normals, normal_valid


def load_image_names(source_root: Path) -> list[str]:
    imagelist = source_root / "imagelist.txt"
    if not imagelist.is_file():
        raise FileNotFoundError(f"Missing iBims-1 imagelist: {imagelist}")
    return [line.strip() for line in imagelist.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_core_samples(args: argparse.Namespace) -> list[CoreSample]:
    source_root = args.source_root.expanduser().resolve()
    core_zip = source_root / "ibims1_core_mat.zip"
    if not core_zip.is_file():
        raise FileNotFoundError(f"Missing core archive: {core_zip}")

    names = load_image_names(source_root)
    samples: list[CoreSample] = []
    with tempfile.TemporaryDirectory(prefix="ibims1-core-") as tmp:
        tmp_root = Path(tmp)
        with ZipFile(core_zip) as archive:
            for index, sample_id in enumerate(tqdm(names, desc="Reading iBims-1 core MAT files"), start=1):
                member = f"ibims1_core_mat/{sample_id}.mat"
                archive.extract(member, tmp_root)
                mat = loadmat(tmp_root / member, squeeze_me=True, struct_as_record=False)
                data = mat["data"]
                rgb = np.asarray(data.rgb, dtype=np.uint8)
                depth = np.asarray(data.depth, dtype=np.float32)
                mask_invalid = np.asarray(data.mask_invalid).astype(bool)
                mask_transp = np.asarray(data.mask_transp).astype(bool)
                valid = build_valid_depth_mask(depth, mask_invalid, mask_transp, args)
                normals, normal_valid = derive_normals_from_depth(
                    depth,
                    np.asarray(data.calib),
                    radius=args.gradient_radius,
                    camera_facing=(
                        args.geometry_version == CAMERA_FACING_GEOMETRY_DATA_VERSION
                    ),
                )
                normal_mask = erode_mask(valid & normal_valid, args.mask_erosion_radius)
                if args.geometry_version == GEOMETRY_DATA_VERSION:
                    normals = IMAGE_Y_DOWN_Z_AWAY_TO_CANONICAL.apply(
                        normals, normal_mask
                    )
                elif args.geometry_version == CAMERA_FACING_GEOMETRY_DATA_VERSION:
                    normals = CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL.apply(
                        normals, normal_mask
                    )
                normal_rgb = normals_to_rgb(normals, normal_mask, args.invalid_normal_rgb)
                samples.append(
                    CoreSample(
                        index_1based=index,
                        sample_id=sample_id,
                        rgb=rgb,
                        depth_m=depth,
                        valid_mask=valid,
                        calib=np.asarray(data.calib, dtype=np.float32),
                        edges=np.asarray(data.edges, dtype=np.uint8),
                        mask_floor=np.asarray(data.mask_floor, dtype=np.uint8),
                        mask_table=np.asarray(data.mask_table, dtype=np.uint8),
                        mask_wall=np.asarray(data.mask_wall, dtype=np.uint8),
                        mask_floor_paras=plane_parameter_rows(data.mask_floor_paras),
                        mask_table_paras=plane_parameter_rows(data.mask_table_paras),
                        mask_wall_paras=plane_parameter_rows(data.mask_wall_paras),
                        normal_rgb=normal_rgb,
                        normal_mask=normal_mask,
                    )
                )
    return samples


def reset_out_root(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite to rebuild.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_depth_geometry(out_root: Path, rel_path: Path, sample: CoreSample) -> None:
    out_path = out_root / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        calib=sample.calib.astype(np.float32, copy=False),
        edges=sample.edges.astype(np.uint8, copy=False),
        mask_floor=sample.mask_floor.astype(np.uint8, copy=False),
        mask_table=sample.mask_table.astype(np.uint8, copy=False),
        mask_wall=sample.mask_wall.astype(np.uint8, copy=False),
        mask_floor_paras=sample.mask_floor_paras.astype(np.float32, copy=False),
        mask_table_paras=sample.mask_table_paras.astype(np.float32, copy=False),
        mask_wall_paras=sample.mask_wall_paras.astype(np.float32, copy=False),
    )


def save_standard_depth(out_root: Path, samples: list[CoreSample], prompt: str) -> tuple[list[str], list[dict]]:
    for rel in ("images/standard", "depth_mm", "masks", "geometry", "splits"):
        (out_root / rel).mkdir(parents=True, exist_ok=True)
    split_lines: list[str] = []
    rows: list[dict] = []
    for sample in samples:
        image_rel = Path("images/standard") / f"{sample.sample_id}_rgb.png"
        depth_rel = Path("depth_mm") / f"{sample.sample_id}_depth_mm.png"
        mask_rel = Path("masks") / f"{sample.sample_id}_valid.png"
        geometry_rel = Path("geometry") / f"{sample.sample_id}_geometry.npz"
        Image.fromarray(sample.rgb, mode="RGB").save(out_root / image_rel)
        Image.fromarray(depth_to_uint16_mm(sample.depth_m, sample.valid_mask), mode="I;16").save(out_root / depth_rel)
        Image.fromarray(mask_to_uint8(sample.valid_mask), mode="L").save(out_root / mask_rel)
        save_depth_geometry(out_root, geometry_rel, sample)
        fx = float(sample.calib[0, 0])
        split_lines.append(
            f"/{image_rel.as_posix()} /{depth_rel.as_posix()} /{mask_rel.as_posix()} "
            f"{fx:.10f} {sample.sample_id} standard - /{geometry_rel.as_posix()}\n"
        )
        rows.append(
            {
                "id": sample.sample_id,
                "split": "test",
                "case": "standard",
                "conditioning_image": image_rel.as_posix(),
                "depth_mm": depth_rel.as_posix(),
                "mask": mask_rel.as_posix(),
                "geometry": geometry_rel.as_posix(),
                "text": prompt,
                "task": "depth_estimation",
                "source_dataset": "ibims1",
            }
        )
    (out_root / "ibims1_standard.txt").write_text("".join(split_lines), encoding="utf-8")
    return split_lines, rows


def save_standard_normals(out_root: Path, samples: list[CoreSample], prompt: str) -> tuple[list[str], list[dict]]:
    for rel in ("images/standard", "normals", "masks", "depth_mm", "splits"):
        (out_root / rel).mkdir(parents=True, exist_ok=True)
    split_lines: list[str] = []
    rows: list[dict] = []
    for sample in samples:
        image_rel = Path("images/standard") / f"{sample.sample_id}_rgb.png"
        normal_rel = Path("normals") / f"{sample.sample_id}_normal.png"
        mask_rel = Path("masks") / f"{sample.sample_id}_normal_mask.png"
        depth_rel = Path("depth_mm") / f"{sample.sample_id}_depth_mm.png"
        Image.fromarray(sample.rgb, mode="RGB").save(out_root / image_rel)
        Image.fromarray(sample.normal_rgb, mode="RGB").save(out_root / normal_rel)
        Image.fromarray(mask_to_uint8(sample.normal_mask), mode="L").save(out_root / mask_rel)
        Image.fromarray(depth_to_uint16_mm(sample.depth_m, sample.valid_mask), mode="I;16").save(out_root / depth_rel)
        fx = float(sample.calib[0, 0])
        split_lines.append(f"/{image_rel.as_posix()} /{normal_rel.as_posix()} /{mask_rel.as_posix()} {fx:.10f} {sample.index_1based} {sample.sample_id} standard -\n")
        rows.append(
            {
                "id": sample.sample_id,
                "split": "test",
                "case": "standard",
                "conditioning_image": image_rel.as_posix(),
                "image": normal_rel.as_posix(),
                "mask": mask_rel.as_posix(),
                "depth_mm": depth_rel.as_posix(),
                "text": prompt,
                "task": "surface_normal_estimation",
                "source_dataset": "ibims1",
                "normal_ground_truth": "derived_from_depth",
            }
        )
    (out_root / "ibims1_standard.txt").write_text("".join(split_lines), encoding="utf-8")
    return split_lines, rows


def all_augmented_groups(source_root: Path) -> list[str]:
    aux_zip = source_root / "ibims1_aux_augmented.zip"
    if not aux_zip.is_file():
        return []
    with ZipFile(aux_zip) as archive:
        groups = {
            name.split("/")[1]
            for name in archive.namelist()
            if name.startswith("ibims1_aux_augmented/")
            and len(name.split("/")) > 2
            and not name.endswith("/")
            and not name.endswith("Thumbs.db")
        }
    return sorted(groups)


def classify_augmented_groups(groups: list[str], requested: list[str]) -> list[str]:
    requested_set = set(requested)
    if "all" in requested_set:
        return groups
    expanded: set[str] = set()
    if "appearance" in requested_set:
        expanded.update(group for group in groups if group.startswith(APPEARANCE_AUG_PREFIXES))
    if "geometric" in requested_set:
        expanded.update(group for group in groups if group in GEOMETRIC_AUGMENTATIONS)
    expanded.update(group for group in groups if group in requested_set)
    missing = requested_set - {"appearance", "geometric"} - set(groups)
    if missing:
        raise ValueError(f"Unknown iBims-1 augmentation group(s): {', '.join(sorted(missing))}")
    return sorted(expanded)


def flipped_depth_and_mask(sample: CoreSample, augmentation: str) -> tuple[np.ndarray, np.ndarray]:
    if augmentation == "flip_lr":
        return np.fliplr(sample.depth_m), np.fliplr(sample.valid_mask)
    if augmentation == "flip_ud":
        return np.flipud(sample.depth_m), np.flipud(sample.valid_mask)
    return sample.depth_m, sample.valid_mask


def flipped_normal_and_mask(sample: CoreSample, augmentation: str) -> tuple[np.ndarray, np.ndarray]:
    rgb = sample.normal_rgb.copy()
    valid = sample.normal_mask.copy()
    normals = rgb.astype(np.float32) / 255.0 * 2.0 - 1.0
    if augmentation == "flip_lr":
        normals = np.fliplr(normals)
        normals[..., 0] *= -1.0
        valid = np.fliplr(valid)
    elif augmentation == "flip_ud":
        normals = np.flipud(normals)
        normals[..., 1] *= -1.0
        valid = np.flipud(valid)
    else:
        return sample.normal_rgb, sample.normal_mask
    return normals_to_rgb(normals, valid, (128, 128, 128)), valid


def save_aux_augmented(
    *,
    task: str,
    out_root: Path,
    source_root: Path,
    samples: list[CoreSample],
    requested: list[str],
    prompt: str,
) -> list[dict]:
    groups = classify_augmented_groups(all_augmented_groups(source_root), requested)
    if not groups:
        return []
    sample_by_id = {sample.sample_id: sample for sample in samples}
    aux_zip = source_root / "ibims1_aux_augmented.zip"
    rows: list[dict] = []
    with ZipFile(aux_zip) as archive:
        for group in tqdm(groups, desc=f"Preparing iBims-1 {task} aux_augmented"):
            split_lines: list[str] = []
            image_dir = out_root / "images/aux_augmented" / group
            image_dir.mkdir(parents=True, exist_ok=True)
            if group in GEOMETRIC_AUGMENTATIONS:
                (out_root / f"aux_augmented/{group}/depth_mm").mkdir(parents=True, exist_ok=True)
                (out_root / f"aux_augmented/{group}/masks").mkdir(parents=True, exist_ok=True)
                (out_root / f"aux_augmented/{group}/normals").mkdir(parents=True, exist_ok=True)

            for sample_id, sample in sample_by_id.items():
                member = f"ibims1_aux_augmented/{group}/{sample_id}_{group}.png"
                if member not in archive.namelist():
                    raise FileNotFoundError(f"Missing augmented image in archive: {member}")
                image_rel = Path("images/aux_augmented") / group / f"{sample_id}_{group}.png"
                with archive.open(member) as src, (out_root / image_rel).open("wb") as dst:
                    shutil.copyfileobj(src, dst)

                fx = float(sample.calib[0, 0])
                if task == "depth":
                    if group in GEOMETRIC_AUGMENTATIONS:
                        depth, mask = flipped_depth_and_mask(sample, group)
                        depth_rel = Path("aux_augmented") / group / "depth_mm" / f"{sample_id}_depth_mm.png"
                        mask_rel = Path("aux_augmented") / group / "masks" / f"{sample_id}_valid.png"
                        Image.fromarray(depth_to_uint16_mm(depth, mask), mode="I;16").save(out_root / depth_rel)
                        Image.fromarray(mask_to_uint8(mask), mode="L").save(out_root / mask_rel)
                    else:
                        depth_rel = Path("depth_mm") / f"{sample_id}_depth_mm.png"
                        mask_rel = Path("masks") / f"{sample_id}_valid.png"
                    split_lines.append(f"/{image_rel.as_posix()} /{depth_rel.as_posix()} /{mask_rel.as_posix()} {fx:.10f} {sample_id} aux_augmented {group}\n")
                    rows.append(
                        {
                            "id": f"{sample_id}_{group}",
                            "source_id": sample_id,
                            "split": "test",
                            "case": "aux_augmented",
                            "augmentation": group,
                            "conditioning_image": image_rel.as_posix(),
                            "depth_mm": depth_rel.as_posix(),
                            "mask": mask_rel.as_posix(),
                            "text": prompt,
                            "task": "depth_estimation",
                        }
                    )
                else:
                    if group in GEOMETRIC_AUGMENTATIONS:
                        normal_rgb, normal_mask = flipped_normal_and_mask(sample, group)
                        normal_rel = Path("aux_augmented") / group / "normals" / f"{sample_id}_normal.png"
                        mask_rel = Path("aux_augmented") / group / "masks" / f"{sample_id}_normal_mask.png"
                        Image.fromarray(normal_rgb, mode="RGB").save(out_root / normal_rel)
                        Image.fromarray(mask_to_uint8(normal_mask), mode="L").save(out_root / mask_rel)
                    else:
                        normal_rel = Path("normals") / f"{sample_id}_normal.png"
                        mask_rel = Path("masks") / f"{sample_id}_normal_mask.png"
                    split_lines.append(f"/{image_rel.as_posix()} /{normal_rel.as_posix()} /{mask_rel.as_posix()} {fx:.10f} {sample.index_1based} {sample_id} aux_augmented {group}\n")
                    rows.append(
                        {
                            "id": f"{sample_id}_{group}",
                            "source_id": sample_id,
                            "split": "test",
                            "case": "aux_augmented",
                            "augmentation": group,
                            "conditioning_image": image_rel.as_posix(),
                            "image": normal_rel.as_posix(),
                            "mask": mask_rel.as_posix(),
                            "text": prompt,
                            "task": "surface_normal_estimation",
                            "normal_ground_truth": "derived_from_depth",
                        }
                    )
            split_path = out_root / "splits" / f"ibims1_aux_augmented_{group}.txt"
            split_path.parent.mkdir(parents=True, exist_ok=True)
            split_path.write_text("".join(split_lines), encoding="utf-8")

    group_files = [out_root / "splits" / f"ibims1_aux_augmented_{group}.txt" for group in groups]
    combined = "".join(path.read_text(encoding="utf-8") for path in group_files)
    (out_root / "splits" / "ibims1_aux_augmented_all.txt").write_text(combined, encoding="utf-8")
    appearance_groups = [group for group in groups if group.startswith(APPEARANCE_AUG_PREFIXES)]
    if appearance_groups:
        combined_appearance = "".join(
            (out_root / "splits" / f"ibims1_aux_augmented_{group}.txt").read_text(encoding="utf-8")
            for group in appearance_groups
        )
        (out_root / "splits" / "ibims1_aux_augmented_appearance.txt").write_text(combined_appearance, encoding="utf-8")
    geometric_groups = [group for group in groups if group in GEOMETRIC_AUGMENTATIONS]
    if geometric_groups:
        combined_geometric = "".join(
            (out_root / "splits" / f"ibims1_aux_augmented_{group}.txt").read_text(encoding="utf-8")
            for group in geometric_groups
        )
        (out_root / "splits" / "ibims1_aux_augmented_geometric.txt").write_text(combined_geometric, encoding="utf-8")
    return rows


def prepare_depth(args: argparse.Namespace, samples: list[CoreSample]) -> None:
    out_root = args.depth_out_root.expanduser().resolve()
    reset_out_root(out_root, args.overwrite)
    _, rows = save_standard_depth(out_root, samples, args.depth_prompt)
    aux_rows = []
    if args.include_aux_augmented:
        aux_rows = save_aux_augmented(
            task="depth",
            out_root=out_root,
            source_root=args.source_root.expanduser().resolve(),
            samples=samples,
            requested=args.aux_augmentations,
            prompt=args.depth_prompt,
        )
    write_jsonl(out_root / "metadata.jsonl", rows + aux_rows)
    write_dataset_info(out_root, args, "ibims1_depth", samples, aux_rows)
    if args.depth_tar_output is not None:
        write_prepared_tar(out_root, args.depth_tar_output)


def prepare_normals(args: argparse.Namespace, samples: list[CoreSample]) -> None:
    out_root = args.normals_out_root.expanduser().resolve()
    reset_out_root(out_root, args.overwrite)
    _, rows = save_standard_normals(out_root, samples, args.normal_prompt)
    aux_rows = []
    if args.include_aux_augmented:
        aux_rows = save_aux_augmented(
            task="surface_normals",
            out_root=out_root,
            source_root=args.source_root.expanduser().resolve(),
            samples=samples,
            requested=args.aux_augmentations,
            prompt=args.normal_prompt,
        )
    write_jsonl(out_root / "metadata.jsonl", rows + aux_rows)
    write_dataset_info(out_root, args, "ibims1_surface_normals", samples, aux_rows)
    if args.normals_tar_output is not None:
        write_prepared_tar(out_root, args.normals_tar_output)


def write_dataset_info(
    out_root: Path,
    args: argparse.Namespace,
    dataset_name: str,
    samples: list[CoreSample],
    aux_rows: list[dict],
) -> None:
    source_root = args.source_root.expanduser().resolve()
    normal_transform = (
        CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL
        if args.geometry_version == CAMERA_FACING_GEOMETRY_DATA_VERSION
        else IMAGE_Y_DOWN_Z_AWAY_TO_CANONICAL
    )
    is_canonical = args.geometry_version in {
        GEOMETRY_DATA_VERSION,
        CAMERA_FACING_GEOMETRY_DATA_VERSION,
    }
    write_json(
        out_root / "dataset_info.json",
        {
            "dataset": dataset_name,
            "geometry_version": (
                args.geometry_version
                if dataset_name == "ibims1_surface_normals"
                else None
            ),
            "canonical_normal_convention": (
                CANONICAL_NORMAL_CONVENTION
                if dataset_name == "ibims1_surface_normals"
                and is_canonical
                else None
            ),
            "normal_axis_permutation": (
                list(normal_transform.permutation)
                if dataset_name == "ibims1_surface_normals"
                and is_canonical
                else None
            ),
            "normal_axis_signs": (
                list(normal_transform.signs)
                if dataset_name == "ibims1_surface_normals"
                and is_canonical
                else None
            ),
            "normal_encoding": (
                LEGACY_DIRECT_NORMAL_ENCODING
                if dataset_name == "ibims1_surface_normals"
                else None
            ),
            "normal_encoding_formula": (
                normal_encoding_formula(LEGACY_DIRECT_NORMAL_ENCODING)
                if dataset_name == "ibims1_surface_normals"
                else None
            ),
            "normal_orientation": (
                "face_camera_ray"
                if dataset_name == "ibims1_surface_normals"
                and args.geometry_version == CAMERA_FACING_GEOMETRY_DATA_VERSION
                else None
            ),
            "stored_normal_convention": (
                CANONICAL_NORMAL_CONVENTION
                if dataset_name == "ibims1_surface_normals" and is_canonical
                else None
            ),
            "gt_to_canonical_permutation": (
                [0, 1, 2]
                if dataset_name == "ibims1_surface_normals" and is_canonical
                else None
            ),
            "gt_to_canonical_signs": (
                [1, 1, 1]
                if dataset_name == "ibims1_surface_normals" and is_canonical
                else None
            ),
            "source_root": str(source_root),
            "source_archives": {
                "core": str(source_root / "ibims1_core_mat.zip"),
                "aux_augmented": str(source_root / "ibims1_aux_augmented.zip"),
                "aux_mvs": str(source_root / "ibims1_aux_mvs.zip"),
                "aux_patterns": str(source_root / "ibims1_aux_patterns.zip"),
            },
            "counts": {
                "standard": len(samples),
                "aux_augmented": len(aux_rows),
            },
            "auxiliary": {
                "included_aux_augmented": bool(args.include_aux_augmented),
                "requested_aux_augmentations": args.aux_augmentations,
                "note": "aux_mvs and aux_patterns are not scored here because this prepared evaluation requires aligned GT depth or derived normals.",
            },
            "depth_range_m": {
                "min_depth": args.min_depth,
                "max_depth": depth_upper_bound(args),
                "validity_protocol": args.depth_validity_protocol,
            },
            "ibims_geometry": {
                "depth_standard_geometry_npz": dataset_name == "ibims1_depth",
                "metrics_enabled_when_split_has_geometry": ["DDE@3m", "DBE", "PE_flat", "PE_ori"],
            },
            "normal_derivation": {
                "ground_truth": "derived from iBims-1 metric depth and calibration",
                "gradient_radius": args.gradient_radius,
                "mask_erosion_radius": args.mask_erosion_radius,
                "encoding": "rgb = round((normal_xyz * 0.5 + 0.5) * 255)",
                "geometry_version": args.geometry_version,
                "canonical_normal_convention": (
                    CANONICAL_NORMAL_CONVENTION
                    if is_canonical
                    else None
                ),
                "native_normal_convention": (
                    normal_transform.native_convention
                ),
                "normal_axis_permutation": (
                    list(normal_transform.permutation)
                    if is_canonical
                    else None
                ),
                "normal_axis_signs": (
                    list(normal_transform.signs)
                    if is_canonical
                    else None
                ),
                "normal_transform_evidence": (
                    normal_transform.evidence
                    if is_canonical
                    else "legacy_untransformed"
                ),
            },
        },
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare iBims-1 depth and derived-normal evaluation datasets.")
    parser.add_argument("--source-root", type=Path, required=True, help="Directory holding ibims1_core_mat.zip and imagelist.txt.")
    parser.add_argument("--task", choices=("depth", "surface_normals", "both"), default="both")
    parser.add_argument("--depth-out-root", type=Path, help="Default: next to --source-root.")
    parser.add_argument("--normals-out-root", type=Path, help="Default: next to --source-root.")
    parser.add_argument("--depth-tar-output", type=Path, help="Default: next to --source-root.")
    parser.add_argument("--normals-tar-output", type=Path, help="Default: next to --source-root.")
    parser.add_argument("--include-aux-augmented", action="store_true", help="Prepare optional aux_augmented split files and images.")
    parser.add_argument(
        "--aux-augmentations",
        nargs="+",
        default=["appearance"],
        help="Aux groups to prepare: appearance, geometric, all, or explicit archive group names.",
    )
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument(
        "--depth-validity-protocol",
        choices=("nyu_10m", "ibims_official"),
        default="nyu_10m",
        help=(
            "nyu_10m keeps the historical 10 m upper cutoff. ibims_official uses "
            "the iBims masks and finite positive depths without an upper-depth cutoff."
        ),
    )
    parser.add_argument(
        "--geometry-version",
        choices=("legacy", GEOMETRY_DATA_VERSION, CAMERA_FACING_GEOMETRY_DATA_VERSION),
        default=CAMERA_FACING_GEOMETRY_DATA_VERSION,
        help=(
            "The default camera-facing v4 contract orients every derived normal "
            "against its camera ray before transforming it into the frozen "
            "x-right/y-up/z-toward-camera convention. Older modes are retained "
            "only to reproduce historical assets."
        ),
    )
    parser.add_argument("--gradient-radius", type=int, default=2)
    parser.add_argument("--mask-erosion-radius", type=int, default=1)
    parser.add_argument("--invalid-normal-rgb", type=lambda value: tuple(int(part) for part in value.split(",")), default=(128, 128, 128))
    parser.add_argument("--depth-prompt", default=DEPTH_PROMPT)
    parser.add_argument("--normal-prompt", default=NORMAL_PROMPT)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    parent = args.source_root.parent
    args.depth_out_root = args.depth_out_root or parent / "ibims1_depth"
    args.normals_out_root = args.normals_out_root or parent / "ibims1_surface_normals"
    args.depth_tar_output = args.depth_tar_output or parent / "ibims1_depth.tar"
    args.normals_tar_output = args.normals_tar_output or parent / "ibims1_surface_normals.tar"
    if len(args.invalid_normal_rgb) != 3 or any(channel < 0 or channel > 255 for channel in args.invalid_normal_rgb):
        raise ValueError("--invalid-normal-rgb must have the form R,G,B with channels in [0, 255]")
    samples = load_core_samples(args)
    if args.task in {"depth", "both"}:
        prepare_depth(args, samples)
    if args.task in {"surface_normals", "both"}:
        prepare_normals(args, samples)
    print("iBims-1 preparation complete.")
    if args.task in {"depth", "both"}:
        print(f"  depth out root: {args.depth_out_root.expanduser().resolve()}")
        print(f"  depth tar:      {args.depth_tar_output.expanduser().resolve() if args.depth_tar_output else '<none>'}")
    if args.task in {"surface_normals", "both"}:
        print(f"  normals root:   {args.normals_out_root.expanduser().resolve()}")
        print(f"  normals tar:    {args.normals_tar_output.expanduser().resolve() if args.normals_tar_output else '<none>'}")


if __name__ == "__main__":
    main()
