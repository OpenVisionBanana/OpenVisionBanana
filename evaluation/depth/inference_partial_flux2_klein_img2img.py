# ===============================================================================
# Flux2-Klein partial-transformer depth-estimation inference and NYU evaluation.
# ===============================================================================

"""
1. Reads NYU train/test split files from an extracted nyu/ root.
2. Runs each RGB image through Flux2-Klein with either a Vision-Banana RGB
   depth prompt or the legacy grayscale depth prompt.
3. Decodes generated RGB either through the Vision Banana RGB path back to
   metric depth, or through legacy luminance for grayscale.
4. Resizes the prediction to the GT frame, reports direct metric-depth metrics
   for Vision Banana RGB, and also performs the existing small translation
   search plus per-image affine alignment for development diagnostics.
5. Saves raw RGB generations, decoded depth maps, aligned depth maps, and
   per-sample / aggregate metrics.

Typical usage:
    python evaluation/depth/inference_partial_flux2_klein_img2img.py \
        --data_root /path/to/nyu \
        --out_dir outputs/nyu_depth \
        --transformer_path /path/to/open-visionbanana-partial-ft
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from common.evaluation_resume import (
    DurableEvaluationJournal,
    summarize_sharded_journals,
)
from common.evaluation_subset import apply_sample_manifest
from common.evaluation_sharding import (
    round_robin_shard,
    sharding_run_identity,
    synchronize_shard_records,
    validate_shard_parameters,
)
from dataset_prep.depth.encoding import (
    DEPTH_ENCODING_BARRON_SCALAR_GRAYSCALE,
    DEPTH_ENCODING_GRAYSCALE,
    DEPTH_ENCODING_RELATIVE_INVERSE_GRAYSCALE,
    DEPTH_ENCODING_RELATIVE_LINEAR_GRAYSCALE,
    DEPTH_ENCODING_VISION_BANANA_RGB,
    DEPTH_ENCODINGS,
    decode_barron_scalar_grayscale_to_depth,
    decode_vision_banana_rgb_to_depth,
    project_rgb_to_vision_banana_path,
)
from evaluation.depth import datasets as depth_datasets
from evaluation.depth.datasets import marigold_ppd as marigold_ppd_datasets
from evaluation.depth.ppd_protocol import (
    PROTOCOL_ID as MARIGOLD_V2_PPD_PROTOCOL,
    relative_grayscale_to_log_depth_signal,
    resolve_ppd_alignment,
    score_ppd_ransac_log,
    score_ppd_split_disparity_affine,
)
from evaluation.geometry_diagnostics import palette_diagnostics
from evaluation.geometry_protocols import (
    PROTOCOL_VERSION,
    DEPTH_DIRECT_METRIC,
    DEPTH_DISPARITY_AFFINE,
    DEPTH_HELDOUT_SELECTED_AFFINE,
    DEPTH_LINEAR_AFFINE,
    DEPTH_REFERENCE_DISPARITY_AFFINE,
    DEPTH_REFERENCE_LINEAR_AFFINE,
    DEPTH_VISION_BANANA_RGB_DIRECT,
    depth_configuration_identity,
    depth_protocol_mask,
    depth_range_for_dataset,
    disparity_affine_depth,
    linear_affine_depth,
    load_depth_affine_calibration_record,
    reference_luminance,
    standard_depth_crop,
)
from common.geometry_protocol_prompts import (
    BARRON_SCALAR_DEPTH_PROMPT,
    CANONICAL_DEPTH_PROMPT,
    CONCISE_INVERSE_DEPTH_PROMPT,
    CONCISE_LINEAR_DEPTH_PROMPT,
    prompt_sha256,
)

GRAYSCALE_PROMPT = (
    "[TASK: DEPTH_ESTIMATION] Generate a dense monocular depth map of the input indoor scene. "
    "Output only a clean grayscale depth image with no text, labels, legends, or extra colors. "
    "Use smooth depth gradients, keep object boundaries sharp, and make nearer surfaces brighter "
    "while farther surfaces are darker."
)
RELATIVE_DEPTH_ENCODINGS = {
    DEPTH_ENCODING_GRAYSCALE,
    DEPTH_ENCODING_RELATIVE_INVERSE_GRAYSCALE,
    DEPTH_ENCODING_RELATIVE_LINEAR_GRAYSCALE,
}
VISION_BANANA_PROMPT = CANONICAL_DEPTH_PROMPT
LEGACY_UNSEEDED_EVALUATOR_SHA256 = (
    "a441e219802ee9a0f532e37f06764854bad9df902f10e14ba6a7ed7ec91d6fdc"
)

ALIGNMENT_MODES = ("linear", "log", "inverse")
DEPTH_METRIC_NAMES = (
    "abs_rel",
    "sq_rel",
    "rmse",
    "rmse_log",
    "log10",
    "silog",
    "delta1",
    "delta2",
    "delta3",
)
IBIMS_GEOMETRY_METRIC_NAMES = (
    "ibims_dde_0",
    "ibims_dde_m",
    "ibims_dde_p",
    "ibims_dbe_acc",
    "ibims_dbe_com",
    "ibims_pe_flat",
    "ibims_pe_ori",
)

_RESAMPLING = getattr(Image, "Resampling", Image)
BILINEAR = _RESAMPLING.BILINEAR


@dataclass(frozen=True)
class NYUSample:
    split: str
    image_path: Path
    depth_path: Path
    focal_length: float
    rel_image_path: Path
    output_stem: str


@dataclass(frozen=True)
class AlignmentConfig:
    mode: str
    invert_signal: bool
    dy: int
    dx: int
    scale: float
    shift: float
    score: float
    overlap_pixels: int


def canonicalize_split_name(split: str) -> str:
    split = split.lower()
    if split == "val":
        return "test"
    return split


def expand_requested_splits(split: str) -> list[str]:
    split = canonicalize_split_name(split)
    if split == "all":
        return ["test", "train"]
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    return [split]


def resolve_dataset_root(data_root: Path) -> Path:
    data_root = data_root.expanduser()
    if (data_root / "nyu_train.txt").is_file() and (data_root / "nyu_test.txt").is_file():
        return data_root
    nested = data_root / "nyu"
    if (nested / "nyu_train.txt").is_file() and (nested / "nyu_test.txt").is_file():
        return nested
    raise FileNotFoundError(
        "Could not find NYU split files. Expected either:\n"
        f"  {data_root / 'nyu_train.txt'} and {data_root / 'nyu_test.txt'}\n"
        "or:\n"
        f"  {nested / 'nyu_train.txt'} and {nested / 'nyu_test.txt'}"
    )


def output_stem_from_rel_path(split: str, rel_path: Path) -> str:
    parts = [split, *rel_path.with_suffix("").parts]
    return "__".join(part for part in parts if part not in ("", "."))


def load_split_file(dataset_root: Path, split: str) -> list[NYUSample]:
    split_file = dataset_root / f"nyu_{split}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing NYU split file: {split_file}")

    samples = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed split entry in {split_file}:{line_number}: {line}")

            rel_image_path = Path(parts[0].lstrip("/"))
            rel_depth_path = Path(parts[1].lstrip("/"))
            focal_length = float(parts[2]) if len(parts) >= 3 else float("nan")

            image_path = dataset_root / rel_image_path
            depth_path = dataset_root / rel_depth_path
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing RGB image from split file: {image_path}")
            if not depth_path.is_file():
                raise FileNotFoundError(f"Missing depth image from split file: {depth_path}")

            samples.append(
                NYUSample(
                    split=split,
                    image_path=image_path,
                    depth_path=depth_path,
                    focal_length=focal_length,
                    rel_image_path=rel_image_path,
                    output_stem=output_stem_from_rel_path(split, rel_image_path),
                )
            )
    return samples


def build_samples(dataset_root: Path, split: str, limit: Optional[int]) -> list[NYUSample]:
    samples = []
    for split_name in expand_requested_splits(split):
        samples.extend(load_split_file(dataset_root, split_name))
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise FileNotFoundError(f"No NYU samples resolved for split={split!r}")
    return samples


def load_gt_depth(depth_path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(depth_path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr / 1000.0


def valid_depth_mask(depth_m: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    return np.isfinite(depth_m) & (depth_m > min_depth) & (depth_m < max_depth)


def decode_depth_signal(pred_rgb: Image.Image) -> np.ndarray:
    return np.asarray(pred_rgb.convert("L"), dtype=np.float32) / 255.0


def resize_scalar_map(values: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if values.shape == target_shape:
        return values.astype(np.float32, copy=False)
    image = Image.fromarray(values.astype(np.float32, copy=False), mode="F")
    resized = image.resize((target_shape[1], target_shape[0]), resample=BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def reference_resize_scalar_map(
    values: np.ndarray, target_shape: tuple[int, int]
) -> np.ndarray:
    """Apply the frozen scipy order-1 reference resize."""
    if values.shape == target_shape:
        return values.astype(np.float32, copy=False)
    from scipy.ndimage import zoom

    factors = (target_shape[0] / values.shape[0], target_shape[1] / values.shape[1])
    return zoom(values.astype(np.float32), factors, order=1)


def reference_standard_crop(dataset: str, shape: tuple[int, int]) -> np.ndarray:
    """Apply the frozen reference crop policy to every supported dataset."""
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    normalized = dataset.lower().replace("-", "_")
    if normalized in {"nyu", "nyuv2"}:
        if shape != (480, 640):
            raise ValueError(f"Reference NYUv2 crop requires 480x640 GT, got {shape}")
        mask[45:471, 41:601] = True
    elif normalized in {"kitti_eigen", "kittieigen"}:
        mask[
            int(0.40810811 * height) : int(0.99189189 * height),
            int(0.03594771 * width) : int(0.96405229 * width),
        ] = True
    else:
        mask[:] = True
    return mask


def heldout_affine_crop(dataset: str, shape: tuple[int, int]) -> np.ndarray:
    """Frozen benchmark crop for OVB held-out affine scoring."""
    normalized = dataset.lower().replace("-", "_")
    if normalized == "kitti":
        # Depth-selection-1000 is already spatially cropped and is not Eigen-652.
        return np.ones(shape, dtype=bool)
    return standard_depth_crop(dataset, shape)


def iter_shifts(max_shift_px: int, shift_step: int) -> list[tuple[int, int]]:
    if max_shift_px <= 0:
        return [(0, 0)]

    step = max(1, shift_step)
    axis = list(range(-max_shift_px, max_shift_px + 1, step))
    if 0 not in axis:
        axis.append(0)
    axis = sorted(set(axis))
    return [(dy, dx) for dy in axis for dx in axis]


def overlap_slices(
    height: int,
    width: int,
    dy: int,
    dx: int,
) -> Optional[tuple[slice, slice, slice, slice]]:
    if abs(dy) >= height or abs(dx) >= width:
        return None

    if dy >= 0:
        pred_y = slice(0, height - dy)
        gt_y = slice(dy, height)
    else:
        pred_y = slice(-dy, height)
        gt_y = slice(0, height + dy)

    if dx >= 0:
        pred_x = slice(0, width - dx)
        gt_x = slice(dx, width)
    else:
        pred_x = slice(-dx, width)
        gt_x = slice(0, width + dx)

    if pred_y.stop <= pred_y.start or pred_x.stop <= pred_x.start:
        return None
    return pred_y, pred_x, gt_y, gt_x


def fit_affine_1d(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    x_centered = x - x_mean
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 1e-12:
        return 0.0, y_mean
    scale = float(np.dot(x_centered, y - y_mean) / denom)
    shift = float(y_mean - scale * x_mean)
    return scale, shift


def fit_alignment(
    signal_values: np.ndarray,
    gt_depth_values: np.ndarray,
    mode: str,
) -> tuple[float, float]:
    if mode == "linear":
        target = gt_depth_values
    elif mode == "log":
        target = np.log(gt_depth_values)
    elif mode == "inverse":
        target = 1.0 / gt_depth_values
    else:
        raise ValueError(f"Unsupported alignment mode: {mode}")

    return fit_affine_1d(signal_values, target)


def apply_alignment(
    signal: np.ndarray,
    mode: str,
    scale: float,
    shift: float,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    aligned = scale * signal + shift
    if mode == "linear":
        depth = aligned
    elif mode == "log":
        depth = np.exp(aligned)
    elif mode == "inverse":
        min_inv = 1.0 / max_depth
        depth = 1.0 / np.clip(aligned, min_inv, None)
    else:
        raise ValueError(f"Unsupported alignment mode: {mode}")

    return np.clip(depth, min_depth, max_depth).astype(np.float32, copy=False)


def find_best_alignment(
    signal: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: np.ndarray,
    alignment_mode: str,
    min_depth: float,
    max_depth: float,
    max_shift_px: int,
    shift_step: int,
    min_overlap_pixels: int,
    allow_invert_signal: bool = True,
) -> AlignmentConfig:
    height, width = gt_depth.shape
    allowed_modes = ALIGNMENT_MODES if alignment_mode == "auto" else (alignment_mode,)
    best: Optional[AlignmentConfig] = None

    for dy, dx in iter_shifts(max_shift_px=max_shift_px, shift_step=shift_step):
        slices = overlap_slices(height, width, dy, dx)
        if slices is None:
            continue
        pred_y, pred_x, gt_y, gt_x = slices
        gt_patch = gt_depth[gt_y, gt_x]
        valid_patch = valid_mask[gt_y, gt_x]
        overlap_pixels = int(valid_patch.sum())
        if overlap_pixels < min_overlap_pixels:
            continue

        signal_patch = signal[pred_y, pred_x]
        gt_values = gt_patch[valid_patch]
        base_signal_values = signal_patch[valid_patch]

        invert_options = (False, True) if allow_invert_signal else (False,)
        for invert_signal in invert_options:
            source_values = 1.0 - base_signal_values if invert_signal else base_signal_values
            source_patch = 1.0 - signal_patch if invert_signal else signal_patch

            for mode in allowed_modes:
                scale, shift = fit_alignment(source_values, gt_values, mode)
                aligned_patch = apply_alignment(
                    signal=source_patch,
                    mode=mode,
                    scale=scale,
                    shift=shift,
                    min_depth=min_depth,
                    max_depth=max_depth,
                )
                score = float(np.mean(np.abs(aligned_patch[valid_patch] - gt_values) / gt_values))

                candidate = AlignmentConfig(
                    mode=mode,
                    invert_signal=invert_signal,
                    dy=dy,
                    dx=dx,
                    scale=scale,
                    shift=shift,
                    score=score,
                    overlap_pixels=overlap_pixels,
                )
                if best is None or candidate.score < best.score:
                    best = candidate

    if best is None:
        raise RuntimeError(
            "No valid alignment candidates were found. Try lowering --min_overlap_pixels or --max_shift_px."
        )
    return best


def materialize_aligned_depth(
    signal: np.ndarray,
    target_shape: tuple[int, int],
    config: AlignmentConfig,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    slices = overlap_slices(target_shape[0], target_shape[1], config.dy, config.dx)
    if slices is None:
        raise RuntimeError("Chosen alignment shift does not overlap the GT frame.")

    pred_y, pred_x, gt_y, gt_x = slices
    source_signal = 1.0 - signal if config.invert_signal else signal
    aligned_source = apply_alignment(
        signal=source_signal,
        mode=config.mode,
        scale=config.scale,
        shift=config.shift,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    aligned_canvas = np.full(target_shape, np.nan, dtype=np.float32)
    aligned_canvas[gt_y, gt_x] = aligned_source[pred_y, pred_x]
    return aligned_canvas


def compute_depth_metrics(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> tuple[dict[str, float], int]:
    mask = valid_mask & np.isfinite(pred_depth)
    valid_pixels = int(mask.sum())
    if valid_pixels == 0:
        raise RuntimeError("Aligned prediction has no valid overlap with GT.")

    pred = np.clip(pred_depth[mask], min_depth, max_depth)
    gt = np.clip(gt_depth[mask], min_depth, max_depth)
    diff = pred - gt

    abs_rel = float(np.mean(np.abs(diff) / gt))
    sq_rel = float(np.mean((diff ** 2) / gt))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    log_diff = np.log(pred) - np.log(gt)
    rmse_log = float(np.sqrt(np.mean(log_diff ** 2)))
    log10 = float(np.mean(np.abs(np.log10(pred) - np.log10(gt))))

    silog_var = max(0.0, float(np.mean(log_diff ** 2) - np.mean(log_diff) ** 2))
    silog = float(np.sqrt(silog_var) * 100.0)

    thresh = np.maximum(gt / pred, pred / gt)
    delta1 = float(np.mean(thresh < 1.25))
    delta2 = float(np.mean(thresh < 1.25 ** 2))
    delta3 = float(np.mean(thresh < 1.25 ** 3))

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "log10": log10,
        "silog": silog,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }, valid_pixels


def compute_ibims_dde(gt_depth: np.ndarray, pred_depth: np.ndarray, valid_mask: np.ndarray, threshold_m: float = 3.0) -> dict[str, float]:
    mask = valid_mask & np.isfinite(gt_depth) & np.isfinite(pred_depth) & (gt_depth > 0) & (pred_depth > 0)
    valid_pixels = int(mask.sum())
    if valid_pixels == 0:
        return {"ibims_dde_0": float("nan"), "ibims_dde_m": float("nan"), "ibims_dde_p": float("nan")}

    gt_close = gt_depth[mask] <= threshold_m
    pred_close = pred_depth[mask] <= threshold_m
    diff = pred_close.astype(np.int8) - gt_close.astype(np.int8)
    return {
        "ibims_dde_0": float(np.mean(diff == 0)),
        "ibims_dde_m": float(np.mean(diff == 1)),
        "ibims_dde_p": float(np.mean(diff == -1)),
    }


def compute_ibims_dbe(edges: np.ndarray, pred_depth: np.ndarray) -> dict[str, float]:
    if int(np.sum(edges > 0)) == 0:
        return {"ibims_dbe_acc": float("nan"), "ibims_dbe_com": float("nan")}

    try:
        from scipy import ndimage
        from skimage import feature
    except ImportError:
        return {"ibims_dbe_acc": float("nan"), "ibims_dbe_com": float("nan")}

    pred = pred_depth.astype(np.float32, copy=True)
    pred[~np.isfinite(pred) | (pred <= 0)] = np.nan
    if not np.any(np.isfinite(pred)):
        return {"ibims_dbe_acc": 10.0, "ibims_dbe_com": 10.0}

    pred = pred - np.nanmin(pred)
    denom = float(np.nanmax(pred))
    if not np.isfinite(denom) or denom <= 1e-12:
        return {"ibims_dbe_acc": 10.0, "ibims_dbe_com": 10.0}

    pred_normalized = np.nan_to_num(pred / denom, nan=0.0, posinf=1.0, neginf=0.0)
    edges_est = feature.canny(
        pred_normalized,
        sigma=np.sqrt(2.0),
        low_threshold=0.1,
        high_threshold=0.2,
    )

    gt_edges = edges > 0
    d_gt = ndimage.distance_transform_edt(~gt_edges)
    d_est = ndimage.distance_transform_edt(~edges_est)
    max_dist_threshold = 10.0
    local_gt_mask = d_gt < max_dist_threshold
    filtered_est_edges = edges_est & local_gt_mask
    if int(np.sum(filtered_est_edges)) == 0:
        return {"ibims_dbe_acc": max_dist_threshold, "ibims_dbe_com": max_dist_threshold}

    return {
        "ibims_dbe_acc": float(np.nansum(d_gt * filtered_est_edges) / np.nansum(filtered_est_edges)),
        "ibims_dbe_com": float(np.nansum(d_est * gt_edges) / np.nansum(gt_edges)),
    }


def as_plane_parameter_rows(paras: np.ndarray) -> np.ndarray:
    paras = np.asarray(paras, dtype=np.float32)
    if paras.size == 0:
        return np.zeros((0, 7), dtype=np.float32)
    if paras.ndim == 1:
        return paras.reshape(1, -1)
    return paras.reshape(paras.shape[0], -1)


def compute_ibims_planarity_for_type(
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    valid_mask: np.ndarray,
    plane_mask: np.ndarray,
    paras: np.ndarray,
    calib: np.ndarray,
) -> tuple[list[float], list[float]]:
    paras = as_plane_parameter_rows(paras)
    if paras.shape[0] == 0:
        return [], []

    height, width = gt_depth.shape
    fx = float(calib[0, 0])
    fy = float(calib[1, 1])
    cx = float(calib[2, 0])
    cy = float(calib[2, 1])
    c, r = np.meshgrid(
        np.arange(1, width + 1, dtype=np.float32),
        np.arange(1, height + 1, dtype=np.float32),
    )

    flatness: list[float] = []
    orientation: list[float] = []
    base_valid = valid_mask & np.isfinite(gt_depth) & np.isfinite(pred_depth) & (gt_depth > 0) & (pred_depth > 0)
    for plane_index in range(paras.shape[0]):
        current_mask = (plane_mask == (plane_index + 1)) & base_valid
        if float(current_mask.sum()) / float(height * width) < 0.05:
            continue

        gt_values = gt_depth[current_mask]
        pred_values = pred_depth[current_mask]
        median_pred = float(np.median(pred_values))
        median_gt = float(np.median(gt_values))
        if median_pred <= 0 or median_gt <= 0 or not np.isfinite(median_pred) or not np.isfinite(median_gt):
            continue

        scaled_pred = pred_depth / (median_pred / median_gt)
        scaled_pred = np.where(current_mask, scaled_pred, np.nan)
        x = ((c - cx) * scaled_pred / fx).ravel()
        y = scaled_pred.ravel()
        z = (-(r - cy) * scaled_pred / fy).ravel()
        points = np.stack((x, y, z), axis=1)
        points = points[np.all(np.isfinite(points), axis=1)]
        if points.shape[0] < 3:
            continue

        centered = points - points.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = -vh[-1]
        point = points.mean(axis=0)
        offset = -float(np.dot(normal, point))
        flatness.append(float(np.std(np.dot(points, normal) + offset) * 100.0))

        gt_normal = paras[plane_index, 4:7]
        gt_norm = float(np.linalg.norm(gt_normal))
        normal_norm = float(np.linalg.norm(normal))
        if gt_norm <= 1e-12 or normal_norm <= 1e-12:
            continue
        gt_normal = gt_normal / gt_norm
        normal = normal / normal_norm
        if float(np.dot(normal, gt_normal)) < 0:
            normal = -normal
        angle = np.degrees(np.arctan2(np.linalg.norm(np.cross(gt_normal, normal)), np.dot(gt_normal, normal)))
        orientation.append(float(angle))

    return flatness, orientation


def compute_ibims_planarity(
    geometry: dict[str, np.ndarray],
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float | int]:
    flatness: list[float] = []
    orientation: list[float] = []
    calib = geometry["calib"]
    for plane_type in ("wall", "table", "floor"):
        plane_flatness, plane_orientation = compute_ibims_planarity_for_type(
            gt_depth=gt_depth,
            pred_depth=pred_depth,
            valid_mask=valid_mask,
            plane_mask=geometry[f"mask_{plane_type}"],
            paras=geometry[f"mask_{plane_type}_paras"],
            calib=calib,
        )
        flatness.extend(plane_flatness)
        orientation.extend(plane_orientation)

    flatness = [value for value in flatness if np.isfinite(value)]
    orientation = [value for value in orientation if np.isfinite(value)]
    return {
        "ibims_pe_flat": float(np.mean(flatness)) if flatness else float("nan"),
        "ibims_pe_ori": float(np.mean(orientation)) if orientation else float("nan"),
        "ibims_pe_flat_sum": float(np.sum(flatness)) if flatness else 0.0,
        "ibims_pe_ori_sum": float(np.sum(orientation)) if orientation else 0.0,
        "ibims_pe_flat_count": len(flatness),
        "ibims_pe_ori_count": len(orientation),
    }


def load_ibims_geometry(sample: object) -> Optional[dict[str, np.ndarray]]:
    metadata = getattr(sample, "metadata", {})
    geometry_path_value = metadata.get("geometry_path") if isinstance(metadata, dict) else None
    if not geometry_path_value:
        return None

    geometry_path = Path(str(geometry_path_value))
    if not geometry_path.is_file():
        return None

    with np.load(geometry_path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def compute_ibims_geometry_metrics(
    sample: object,
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: np.ndarray,
    geometry: Optional[dict[str, np.ndarray]],
) -> dict[str, float | int]:
    if getattr(sample, "dataset", None) != "ibims1" or geometry is None:
        return {}

    metrics: dict[str, float | int] = {}
    metrics.update(compute_ibims_dde(gt_depth=gt_depth, pred_depth=pred_depth, valid_mask=valid_mask))
    metrics.update(compute_ibims_dbe(edges=geometry["edges"], pred_depth=pred_depth))
    metrics.update(
        compute_ibims_planarity(
            geometry=geometry,
            gt_depth=gt_depth,
            pred_depth=pred_depth,
            valid_mask=valid_mask,
        )
    )
    return metrics


def aggregate_metrics(records: Sequence[dict[str, float]]) -> dict[str, float]:
    return {
        name: float(np.mean([record[name] for record in records]))
        for name in DEPTH_METRIC_NAMES
    }


def aggregate_diagnostic_fields(records, prefix="representation_"):
    keys = sorted({key for record in records for key in record if key.startswith(prefix)})
    return {key: float(np.nanmean([float(record[key]) for record in records if key in record])) for key in keys}


def aggregate_prefixed_metrics(records: Sequence[dict[str, float | int | bool | str]], prefix: str) -> dict[str, float]:
    return {
        name: float(np.mean([float(record[f"{prefix}_{name}"]) for record in records]))
        for name in DEPTH_METRIC_NAMES
    }


def aggregate_optional_prefixed_metrics(
    records: Sequence[dict[str, float | int | bool | str | None]], prefix: str
) -> Optional[dict[str, float]]:
    selected = [
        record
        for record in records
        if all(f"{prefix}_{name}" in record for name in DEPTH_METRIC_NAMES)
    ]
    # A benchmark lane is complete only when every evaluated sample contributes.
    # Silently averaging the subset whose affine fit happened to succeed would
    # make sparse datasets look valid while changing their effective sample set.
    if not selected or len(selected) != len(records):
        return None
    return aggregate_prefixed_metrics(selected, prefix)


def nanmean_or_none(values: Sequence[float]) -> Optional[float]:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    if not finite:
        return None
    return float(np.mean(finite))


def aggregate_prefixed_geometry_metrics(
    records: Sequence[dict[str, float | int | bool | str | None]],
    prefix: str,
) -> Optional[dict[str, float | None]]:
    if not any(f"{prefix}_ibims_dde_0" in record for record in records):
        return None

    summary: dict[str, float | None] = {}
    for name in ("ibims_dde_0", "ibims_dde_m", "ibims_dde_p", "ibims_dbe_acc", "ibims_dbe_com"):
        key = f"{prefix}_{name}"
        summary[name] = nanmean_or_none([float(record[key]) for record in records if key in record])

    for name in ("ibims_pe_flat", "ibims_pe_ori"):
        sum_key = f"{prefix}_{name}_sum"
        count_key = f"{prefix}_{name}_count"
        total = 0.0
        count = 0
        for record in records:
            if sum_key not in record or count_key not in record:
                continue
            record_count = int(record[count_key])
            if record_count <= 0:
                continue
            total += float(record[sum_key])
            count += record_count
        summary[name] = float(total / count) if count > 0 else None

    return summary


def records_by_case(records: Sequence[dict[str, float | int | bool | str | None]]) -> dict[str, list[dict[str, float | int | bool | str | None]]]:
    grouped: dict[str, list[dict[str, float | int | bool | str | None]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("case", "standard")), []).append(record)
    return grouped


def depth_to_uint16_mm(depth_m: np.ndarray) -> np.ndarray:
    depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
    valid = np.isfinite(depth_m)
    depth_mm[valid] = np.clip(np.rint(depth_m[valid] * 1000.0), 0, 65535).astype(np.uint16)
    return depth_mm


def depth_to_vis(depth_m: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    vis = np.zeros(depth_m.shape, dtype=np.uint8)
    valid = np.isfinite(depth_m)
    if not np.any(valid):
        return vis

    denom = max(max_depth - min_depth, 1e-6)
    norm = (np.clip(depth_m[valid], min_depth, max_depth) - min_depth) / denom
    vis[valid] = np.clip(np.rint((1.0 - norm) * 255.0), 0, 255).astype(np.uint8)
    return vis


def save_grayscale_png(values: np.ndarray, out_path: Path) -> None:
    array = np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(array, mode="L").save(out_path)


def select_prompt(depth_encoding: str, prompt: Optional[str]) -> str:
    if prompt is not None:
        return prompt
    if depth_encoding == DEPTH_ENCODING_RELATIVE_INVERSE_GRAYSCALE:
        return CONCISE_INVERSE_DEPTH_PROMPT
    if depth_encoding == DEPTH_ENCODING_RELATIVE_LINEAR_GRAYSCALE:
        return CONCISE_LINEAR_DEPTH_PROMPT
    if depth_encoding == DEPTH_ENCODING_GRAYSCALE:
        return GRAYSCALE_PROMPT
    if depth_encoding == DEPTH_ENCODING_VISION_BANANA_RGB:
        return VISION_BANANA_PROMPT
    if depth_encoding == DEPTH_ENCODING_BARRON_SCALAR_GRAYSCALE:
        return BARRON_SCALAR_DEPTH_PROMPT
    raise ValueError(f"Unsupported depth encoding: {depth_encoding}")


def resolve_transformer_load_args(transformer_path: str) -> tuple[str, Optional[str]]:
    path = Path(transformer_path).expanduser()
    if (path / "transformer").is_dir():
        return str(path), "transformer"
    return str(path), None


def load_pipeline(model_name: str, transformer_path: Optional[str], model_family="klein", lora_path=None, offload=False):
    from common.flux2_runtime import load_flux2_pipeline
    pipe, torch, _metadata = load_flux2_pipeline(
        model_name, model_family=model_family, transformer_path=transformer_path,
        lora_path=lora_path, device="cuda", offload=offload,
    )
    return pipe, torch


def make_generation_generator(torch_module, seed: int, legacy_unseeded: bool = False):
    """Return a freshly seeded CUDA generator for one independent sample."""
    if legacy_unseeded:
        return None
    return torch_module.Generator(device="cuda").manual_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-base-9B")
    parser.add_argument("--model_family", choices=("klein", "dev", "qwen_edit_2511"), default="klein")
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--model_offload", action="store_true")
    parser.add_argument(
        "--transformer_path",
        default=None,
        help=(
            "Optional path to the trained output dir, checkpoint dir, or transformer dir. "
            "Both open-visionbanana-partial-ft and open-visionbanana-partial-ft/transformer are accepted."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=depth_datasets.SUPPORTED_DATASETS,
        default="nyu",
        help="Evaluation dataset adapter to use. Defaults to the existing NYU behavior.",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Extracted dataset root, or a parent directory containing the prepared dataset folder.",
    )
    parser.add_argument(
        "--benchmark_protocol",
        choices=("geometry_v5", "marigold_v2_ppd"),
        default="geometry_v5",
        help=(
            "Dataset and scoring contract. marigold_v2_ppd uses the exact five "
            "Marigold V2 table splits and adds the PPD RANSAC-log result lane."
        ),
    )
    parser.add_argument(
        "--benchmark_split_root",
        default=None,
        help="Root containing Marigold's evaluation/data_split tree.",
    )
    parser.add_argument(
        "--ppd_primary_alignment",
        choices=("auto", "disparity_affine", "log_depth_ransac"),
        default="auto",
        help="PPD splits: auto selects disparity-affine for inverse grayscale; "
             "log_depth_ransac retains the separately labelled legacy log lane.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "test", "val", "all", "debug", "small", "standard", "large", "giant"),
        help="Default is the held-out NYU test split. 'val' is accepted as an alias for 'test'.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=("standard",),
        help=(
            "Dataset cases to evaluate. For iBims-1, default is standard. "
            "Use 'aux_augmented' or 'all' to include optional auxiliary data. "
            "For DIODE, use indoor, outdoor, or all. For nuScenes, use cam_front or all."
        ),
    )
    parser.add_argument(
        "--aux_augmentations",
        "--aux-augmentations",
        nargs="+",
        default=("appearance",),
        help=(
            "iBims-1 auxiliary augmentation groups. Defaults to appearance, which is the safe "
            "non-geometric group. Use all or specific group names for diagnostic runs."
        ),
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Generation seed reset independently for every sample. Resetting at "
            "the sample boundary makes retries and durable resume deterministic."
        ),
    )
    parser.add_argument(
        "--legacy_unseeded",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--depth_encoding",
        "--depth-encoding",
        choices=DEPTH_ENCODINGS,
        default=DEPTH_ENCODING_VISION_BANANA_RGB,
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sample_manifest",
        default=None,
        help=(
            "Frozen ordered JSONL sample manifest. Mutually exclusive with --limit; "
            "used for versioned geometry ablation subsets."
        ),
    )
    parser.add_argument(
        "--primary_case",
        default=None,
        help=(
            "Optional case selected for top-level primary metrics while all "
            "evaluated cases remain under *_metrics_by_case."
        ),
    )
    parser.add_argument(
        "--protocol_version",
        default=PROTOCOL_VERSION,
        help="Versioned evaluation contract recorded in the result JSON.",
    )
    parser.add_argument(
        "--dataset_role",
        default="unspecified",
        choices=(
            "vision_banana_primary",
            "extended_diagnostic",
            "affine_diagnostic_only",
            "heldout_calibration",
            "unspecified",
        ),
    )
    parser.add_argument(
        "--prediction_dir",
        default=None,
        help=(
            "CPU rescoring mode: load existing raw prediction PNGs named "
            "<sample.output_stem>.png from this directory instead of loading a model."
        ),
    )
    parser.add_argument(
        "--metrics_only",
        action="store_true",
        help="Write only JSON/JSONL metrics; do not duplicate per-image diagnostic files.",
    )
    parser.add_argument(
        "--archived_generation_result",
        default=None,
        help=(
            "Original result JSON whose generation block produced --prediction_dir. "
            "Required by archival rescoring workflows to preserve prompt/model provenance."
        ),
    )
    parser.add_argument(
        "--resume_state_dir",
        default=None,
        help="Persistent state directory used to resume a terminated evaluation safely.",
    )
    parser.add_argument("--resume_chunk_size", type=int, default=50)
    parser.add_argument(
        "--resume_data_identity",
        default=None,
        help="Stable archive identity for resume validation (avoids job-local extraction paths).",
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument(
        "--shard_sync_dir",
        default=None,
        help="Shared node-local directory used to merge multi-process records.",
    )
    parser.add_argument("--shard_sync_timeout_seconds", type=float, default=3600.0)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Resolve NYU image/depth pairs and exit before loading the model.",
    )
    parser.add_argument(
        "--native_resolution",
        action="store_true",
        help="Ask Flux2-Klein to generate at the input image resolution.",
    )
    parser.add_argument("--min_depth", type=float, default=1e-3)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument(
        "--alignment_mode",
        choices=("auto", *ALIGNMENT_MODES),
        default="auto",
        help="Depth-space used for per-image affine calibration. 'auto' tries linear, log, and inverse depth.",
    )
    parser.add_argument(
        "--max_shift_px",
        type=int,
        default=8,
        help="Maximum integer translation searched in each direction before scoring.",
    )
    parser.add_argument(
        "--shift_step",
        type=int,
        default=4,
        help="Stride used when enumerating translation candidates.",
    )
    parser.add_argument(
        "--min_overlap_pixels",
        type=int,
        default=4096,
        help="Minimum valid overlap required for an alignment candidate.",
    )
    parser.add_argument(
        "--disable_affine_diagnostics",
        action="store_true",
        help=(
            "Do not report the fixed disparity-affine and linear-affine diagnostic "
            "lanes. These lanes never use spatial shifts "
            "or GT-based protocol/polarity selection."
        ),
    )
    parser.add_argument(
        "--heldout_depth_calibration_record",
        default=None,
        help=(
            "Immutable record selecting depth- or disparity-space affine fitting on "
            "disjoint calibration data. Required for the complete depth protocol."
        ),
    )
    parser.add_argument(
        "--depth_calibration_candidates_only",
        action="store_true",
        help=(
            "Generate both fixed affine candidates without requiring a prior selection. "
            "This mode is only for constructing a held-out calibration record."
        ),
    )
    parser.add_argument(
        "--vision_banana_direct_rescore_only",
        action="store_true",
        help=(
            "CPU-only recovery mode for archived raw RGB predictions. It emits "
            "the fixed Vision Banana direct lane without requiring a held-out "
            "affine calibration record; callers must merge only that named lane."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_shard_parameters(args.num_shards, args.shard_index)
    if args.num_shards > 1 and args.legacy_unseeded:
        raise ValueError(
            "multi-process evaluation requires fixed per-sample seeds; "
            "legacy process RNG is intentionally single-process only"
        )
    if args.num_shards > 1 and not args.shard_sync_dir:
        raise ValueError("--shard_sync_dir is required when --num_shards is greater than 1")
    if args.metrics_only and args.resume_state_dir:
        raise ValueError("--metrics_only cannot be combined with --resume_state_dir")
    if args.sample_manifest and args.limit is not None:
        raise ValueError("--sample_manifest and --limit are mutually exclusive")
    if args.vision_banana_direct_rescore_only and not (
        args.metrics_only and args.prediction_dir and args.archived_generation_result
    ):
        raise ValueError(
            "--vision_banana_direct_rescore_only requires --metrics_only, "
            "--prediction_dir, and --archived_generation_result"
        )
    if args.disable_affine_diagnostics and not args.vision_banana_direct_rescore_only:
        raise ValueError(
            "--disable_affine_diagnostics is incompatible with the mandatory "
            "complete depth protocol"
        )
    prediction_dir = (
        Path(args.prediction_dir).expanduser().resolve()
        if args.prediction_dir
        else None
    )
    archived_generation_result_path = (
        Path(args.archived_generation_result).expanduser().resolve()
        if args.archived_generation_result
        else None
    )
    archived_generation = None
    if archived_generation_result_path is not None:
        if prediction_dir is None:
            raise ValueError("--archived_generation_result requires --prediction_dir")
        source_result = json.loads(
            archived_generation_result_path.read_text(encoding="utf-8")
        )
        archived_generation = source_result.get("generation")
        if not isinstance(archived_generation, dict):
            raise ValueError(
                f"Original result lacks a generation object: "
                f"{archived_generation_result_path}"
            )
        if not args.prompt and not isinstance(archived_generation.get("prompt"), str):
            raise ValueError(
                f"Original generation metadata lacks a prompt string: "
                f"{archived_generation_result_path}"
            )
    prompt = select_prompt(
        args.depth_encoding,
        args.prompt
        or (archived_generation["prompt"] if archived_generation else None),
    )
    qwen_negative_prompt = (
        os.environ.get("QWEN_NEGATIVE_PROMPT", " ")
        if args.model_family == "qwen_edit_2511"
        else None
    )
    identity_generation = archived_generation or {}
    depth_configuration = {
        "model": identity_generation.get("model", args.model),
        "model_family": identity_generation.get("model_family", args.model_family),
        "transformer_path": identity_generation.get(
            "transformer_path", args.transformer_path
        ),
        "lora_path": identity_generation.get("lora_path", args.lora_path),
        "prompt_sha256": prompt_sha256(prompt),
        "negative_prompt_sha256": identity_generation.get(
            "negative_prompt_sha256",
            prompt_sha256(qwen_negative_prompt)
            if qwen_negative_prompt is not None
            else None,
        ),
        "steps": identity_generation.get("steps", args.steps),
        "guidance_scale": identity_generation.get(
            "guidance_scale", args.guidance_scale
        ),
        "seed": identity_generation.get(
            "seed", None if args.legacy_unseeded else args.seed
        ),
        "seed_mode": identity_generation.get(
            "seed_mode",
            "legacy_process_rng" if args.legacy_unseeded else "fixed_per_sample_reset",
        ),
        "native_resolution": identity_generation.get(
            "native_resolution", bool(args.native_resolution)
        ),
        "depth_encoding": args.depth_encoding,
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_backend_revision": identity_generation.get(
            "runtime_backend_revision",
            os.environ.get("QWEN_MUSUBI_REV")
            if args.model_family == "qwen_edit_2511"
            else "diffusers_environment",
        ),
        "attention_mode": identity_generation.get(
            "attention_mode", os.environ.get("QWEN_ATTN_MODE")
        ),
    }
    normalized_depth_configuration, depth_configuration_sha256 = (
        depth_configuration_identity(depth_configuration)
    )
    heldout_depth_calibration = None
    if (
        not args.depth_calibration_candidates_only
        and not args.vision_banana_direct_rescore_only
        and not args.dry_run
        and args.benchmark_protocol != "marigold_v2_ppd"
    ):
        if not args.heldout_depth_calibration_record:
            raise ValueError(
                "The complete depth protocol requires "
                "--heldout_depth_calibration_record. Use "
                "--depth_calibration_candidates_only only on disjoint calibration data."
            )
        heldout_depth_calibration = load_depth_affine_calibration_record(
            Path(args.heldout_depth_calibration_record), normalized_depth_configuration
        )

    if args.benchmark_protocol == "marigold_v2_ppd":
        ppd_primary_alignment = resolve_ppd_alignment(
            args.depth_encoding, args.ppd_primary_alignment
        )
        if args.depth_encoding in RELATIVE_DEPTH_ENCODINGS:
            print("Note: direct relative grayscale is not log depth. "
                  "The RANSAC-log lane is a legacy diagnostic; "
                  f"selected primary alignment: {ppd_primary_alignment}.")
        if not args.benchmark_split_root:
            raise ValueError(
                "--benchmark_split_root is required for --benchmark_protocol "
                "marigold_v2_ppd"
            )
        dataset_root = marigold_ppd_datasets.resolve_dataset_root(
            args.dataset, Path(args.data_root)
        )
    else:
        dataset_root = depth_datasets.resolve_dataset_root(
            args.dataset, Path(args.data_root)
        )
    out_dir = Path(args.out_dir)
    if args.benchmark_protocol == "marigold_v2_ppd":
        resolved_full_samples = marigold_ppd_datasets.build_samples(
            dataset=args.dataset,
            dataset_root=dataset_root,
            split_root=Path(args.benchmark_split_root),
            limit=None if args.sample_manifest else args.limit,
        )
    else:
        resolved_full_samples = depth_datasets.build_samples(
            dataset=args.dataset,
            dataset_root=dataset_root,
            split=args.split,
            limit=None if args.sample_manifest else args.limit,
            cases=tuple(args.cases),
            aux_augmentations=tuple(args.aux_augmentations),
        )
    full_samples, subset_metadata = apply_sample_manifest(
        resolved_full_samples,
        args.sample_manifest,
        dataset=args.dataset,
    )
    resolved_splits = sorted({sample.split for sample in full_samples})
    resolved_cases = sorted({sample.case for sample in full_samples})
    samples = round_robin_shard(full_samples, args.num_shards, args.shard_index)

    if args.dry_run:
        print(f"Dataset root: {dataset_root}")
        print(f"Requested split: {args.split}")
        print(f"Resolved splits: {', '.join(resolved_splits)}")
        print(f"Resolved cases: {', '.join(resolved_cases)}")
        print(f"Resolved image/depth pairs: {len(full_samples)}")
        if subset_metadata is not None:
            print(
                f"Sample subset: {subset_metadata['protocol_id']} "
                f"({subset_metadata['sample_count']}/"
                f"{subset_metadata['full_resolved_sample_count']})"
            )
        if args.num_shards > 1:
            print(
                f"Shard: {args.shard_index}/{args.num_shards} "
                f"({len(samples)} assigned pairs)"
            )
        print("First pairs:")
        for sample in samples[:5]:
            print(f"  split:  {sample.split}")
            print(f"  case:   {sample.case}")
            print(f"  image:  {sample.image_path}")
            print(f"  depth:  {sample.depth_path}")
            if sample.mask_path is not None:
                print(f"  mask:   {sample.mask_path}")
            print(f"  focal:  {sample.focal_length:.4f}")
        return

    raw_rgb_dir = out_dir / "pred_rgb_raw"
    gray_dir = out_dir / "pred_depth_gray"
    direct_mm_dir = out_dir / "pred_depth_direct_mm"
    direct_vis_dir = out_dir / "pred_depth_direct_vis"
    aligned_mm_dir = out_dir / "pred_depth_aligned_mm"
    aligned_vis_dir = out_dir / "pred_depth_aligned_vis"
    noshift_mm_dir = out_dir / "pred_depth_aligned_noshift_mm"
    noshift_vis_dir = out_dir / "pred_depth_aligned_noshift_vis"
    path_distance_dir = out_dir / "pred_vb_rgb_path_distance"
    directories = [raw_rgb_dir, aligned_mm_dir, aligned_vis_dir, noshift_mm_dir, noshift_vis_dir]
    if args.depth_encoding in RELATIVE_DEPTH_ENCODINGS:
        directories.append(gray_dir)
    elif args.depth_encoding == DEPTH_ENCODING_BARRON_SCALAR_GRAYSCALE:
        directories.extend([gray_dir, direct_mm_dir, direct_vis_dir])
    else:
        directories.extend([direct_mm_dir, direct_vis_dir, path_distance_dir])
    if args.metrics_only:
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    resume = None
    if args.resume_state_dir:
        evaluator_sha256 = (
            LEGACY_UNSEEDED_EVALUATOR_SHA256
            if args.legacy_unseeded
            else hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        )
        resume_config = {
            "evaluator": "depth_flux2_klein",
            "evaluator_sha256": evaluator_sha256,
            "dataset": args.dataset,
            "data_identity": args.resume_data_identity or str(dataset_root.resolve()),
            "split": args.split,
            "cases": list(args.cases),
            "aux_augmentations": list(args.aux_augmentations),
            "model": args.model,
            "model_family": args.model_family,
            "transformer_path": args.transformer_path,
            "lora_path": args.lora_path,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "prompt": prompt,
            "prompt_sha256": prompt_sha256(prompt),
            "qwen_negative_prompt": qwen_negative_prompt,
            "qwen_negative_prompt_sha256": (
                prompt_sha256(qwen_negative_prompt)
                if qwen_negative_prompt is not None
                else None
            ),
            "primary_case": args.primary_case,
            "protocol_version": args.protocol_version,
            "benchmark_protocol": args.benchmark_protocol,
            "ppd_primary_alignment": (
                ppd_primary_alignment if args.benchmark_protocol == "marigold_v2_ppd" else None
            ),
            "benchmark_split_root": args.benchmark_split_root,
            "sample_subset": subset_metadata,
            "native_resolution": bool(args.native_resolution),
            "depth_encoding": args.depth_encoding,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "alignment_mode": args.alignment_mode,
            "max_shift_px": args.max_shift_px,
            "shift_step": args.shift_step,
            "min_overlap_pixels": args.min_overlap_pixels,
            "affine_diagnostics_enabled": not args.disable_affine_diagnostics,
            "depth_configuration_sha256": depth_configuration_sha256,
            "heldout_depth_calibration_record": (
                str(Path(args.heldout_depth_calibration_record).resolve())
                if args.heldout_depth_calibration_record
                else None
            ),
            "heldout_depth_affine_space": (
                heldout_depth_calibration.get("selected_affine_space")
                if heldout_depth_calibration
                else None
            ),
            "depth_calibration_candidates_only": bool(
                args.depth_calibration_candidates_only
            ),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        }
        if not args.legacy_unseeded:
            resume_config["generation_seed"] = args.seed
            resume_config["generation_seed_mode"] = "fixed_per_sample_reset"
        resume_state_dir = Path(args.resume_state_dir)
        if args.num_shards > 1:
            resume_state_dir = (
                resume_state_dir
                / f"shard-{args.shard_index:04d}-of-{args.num_shards:04d}"
            )
        resume = DurableEvaluationJournal(
            state_dir=resume_state_dir,
            config=resume_config,
            sample_keys=[sample.output_stem for sample in samples],
            output_dir=out_dir,
            chunk_size=args.resume_chunk_size,
        )
        print(
            f"Durable resume: {len(resume.completed_keys)}/{len(samples)} samples already committed "
            f"in {resume_state_dir}"
        )

    remaining = [sample for sample in samples if resume is None or sample.output_stem not in resume.completed_keys]
    pipe = torch = None
    if remaining and prediction_dir is None:
        pipe, torch = load_pipeline(
            model_name=args.model, transformer_path=args.transformer_path,
            model_family=args.model_family, lora_path=args.lora_path, offload=args.model_offload,
        )

    per_sample_records: list[dict[str, float | int | bool | str | None]] = []
    per_sample_record_keys: list[str] = []
    for sample in tqdm(samples):
        if resume is not None and sample.output_stem in resume.completed_keys:
            continue
        img = Image.open(sample.image_path).convert("RGB")
        if args.benchmark_protocol == "marigold_v2_ppd":
            img = marigold_ppd_datasets.prepare_input_image(sample, img)
        width, height = img.size

        save_name = f"{sample.output_stem}.png"
        if prediction_dir is not None:
            prediction_path = prediction_dir / save_name
            if not prediction_path.is_file():
                raise FileNotFoundError(
                    f"Missing archived raw prediction for {sample.output_stem}: "
                    f"{prediction_path}"
                )
            pred_rgb = Image.open(prediction_path).convert("RGB")
        else:
            generation_kwargs = {
                "prompt": prompt,
                "image": img,
                "num_inference_steps": args.steps,
                "guidance_scale": args.guidance_scale,
            }
            generator = make_generation_generator(
                torch, args.seed, legacy_unseeded=args.legacy_unseeded
            )
            if generator is not None:
                generation_kwargs["generator"] = generator
            if args.native_resolution:
                generation_kwargs["width"] = width
                generation_kwargs["height"] = height
            with torch.inference_mode():
                pred_rgb = pipe(**generation_kwargs).images[0]
            if not args.metrics_only:
                pred_rgb.save(raw_rgb_dir / save_name)

        if args.benchmark_protocol == "marigold_v2_ppd":
            gt_depth = marigold_ppd_datasets.load_gt_depth(sample)
        else:
            gt_depth = depth_datasets.load_gt_depth(sample)
        if args.benchmark_protocol == "marigold_v2_ppd":
            valid_mask = marigold_ppd_datasets.valid_depth_mask(sample, gt_depth)
            pred_rgb_array = np.asarray(pred_rgb.convert("RGB"), dtype=np.uint8)
            if args.depth_encoding in RELATIVE_DEPTH_ENCODINGS:
                ppd_pred_log = relative_grayscale_to_log_depth_signal(
                    decode_depth_signal(pred_rgb),
                    inverse=args.depth_encoding == DEPTH_ENCODING_RELATIVE_INVERSE_GRAYSCALE,
                )
                ppd_input_signal = (
                    "negative_generated_inverse_grayscale_as_log_depth"
                    if args.depth_encoding == DEPTH_ENCODING_RELATIVE_INVERSE_GRAYSCALE
                    else "generated_relative_grayscale_as_log_depth"
                )
            elif args.depth_encoding == DEPTH_ENCODING_BARRON_SCALAR_GRAYSCALE:
                ppd_depth = decode_barron_scalar_grayscale_to_depth(
                    decode_depth_signal(pred_rgb)
                )
                ppd_pred_log = np.log(np.clip(ppd_depth, 1e-6, None))
                ppd_input_signal = "log_decoded_barron_metric_depth"
            else:
                ppd_depth = decode_vision_banana_rgb_to_depth(pred_rgb_array)
                ppd_pred_log = np.log(np.clip(ppd_depth, 1e-6, None))
                ppd_input_signal = "log_decoded_vision_banana_metric_depth"
            ppd_metrics, ppd_metadata, _ = score_ppd_ransac_log(
                dataset=sample.dataset,
                pred_log_depth=ppd_pred_log,
                gt_depth=gt_depth,
                valid_mask=valid_mask,
            )
            record: dict[str, float | int | bool | str | None] = {
                "split": sample.split,
                "dataset": sample.dataset,
                "case": sample.case,
                "sample_id": sample.sample_id,
                "image_path": str(sample.image_path),
                "depth_path": str(sample.depth_path),
                "mask_path": (
                    str(sample.mask_path) if sample.mask_path is not None else None
                ),
            }
            record.update(
                {
                    f"marigold_v2_ppd_{name}": value
                    for name, value in ppd_metrics.items()
                }
            )
            record.update(
                {
                    f"marigold_v2_ppd_{name}": value
                    for name, value in ppd_metadata.items()
                    if name != "protocol"
                }
            )
            record["marigold_v2_ppd_input_signal"] = ppd_input_signal
            if args.depth_encoding == DEPTH_ENCODING_RELATIVE_INVERSE_GRAYSCALE:
                affine_metrics, affine_metadata = score_ppd_split_disparity_affine(
                    dataset=sample.dataset,
                    pred_inverse_signal=decode_depth_signal(pred_rgb),
                    gt_depth=gt_depth,
                    valid_mask=valid_mask,
                )
                record.update({f"ppd_split_disparity_affine_{name}": value
                               for name, value in affine_metrics.items()})
                record.update({f"ppd_split_disparity_affine_{name}": value
                               for name, value in affine_metadata.items()})
            if resume is not None:
                resume.add(
                    "record",
                    sample.output_stem,
                    record,
                    (raw_rgb_dir / save_name,),
                )
            else:
                per_sample_records.append(record)
                per_sample_record_keys.append(sample.output_stem)
            continue
        reference_rgb = np.asarray(
            pred_rgb.convert("RGB").resize(img.size, resample=BILINEAR),
            dtype=np.uint8,
        )
        reference_signal = reference_resize_scalar_map(
            reference_luminance(reference_rgb), gt_depth.shape
        )
        # Relative-depth editors generate one grayscale signal. Keep that
        # signal available for grayscale models. Vision Banana-encoded
        # outputs instead use their declared decoded depth (linear lane) and
        # reciprocal decoded depth (disparity lane); using RGB luminance here
        # would silently define a third, unreported representation.
        affine_generated_signal = resize_scalar_map(
            decode_depth_signal(pred_rgb), gt_depth.shape
        )
        valid_mask = depth_datasets.valid_depth_mask(
            sample,
            gt_depth,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
        ibims_geometry = load_ibims_geometry(sample)
        geometry_path_value = sample.metadata.get("geometry_path") if hasattr(sample, "metadata") else None

        # The Vision Banana direct lane decodes the raw RGB output regardless of
        # the prompted representation and is computed for every prompt/encoding.
        # ``direct_metrics`` below is the representation-native unaligned lane.
        pred_rgb_array = np.asarray(pred_rgb.convert("RGB"), dtype=np.uint8)
        vision_banana_direct_depth = decode_vision_banana_rgb_to_depth(pred_rgb_array)
        vision_banana_direct_depth = resize_scalar_map(
            vision_banana_direct_depth, gt_depth.shape
        )
        vision_banana_direct_depth = np.clip(
            vision_banana_direct_depth, args.min_depth, args.max_depth
        )

        if args.depth_encoding in RELATIVE_DEPTH_ENCODINGS:
            pred_signal = decode_depth_signal(pred_rgb)
            pred_signal = resize_scalar_map(pred_signal, gt_depth.shape)
            if not args.metrics_only:
                save_grayscale_png(pred_signal, gray_dir / save_name)
            direct_depth = None
            path_distance_stats = None
            alignment_signal = pred_signal
            allow_invert_signal = True
        elif args.depth_encoding == DEPTH_ENCODING_BARRON_SCALAR_GRAYSCALE:
            pred_signal = decode_depth_signal(pred_rgb)
            pred_signal = resize_scalar_map(pred_signal, gt_depth.shape)
            if not args.metrics_only:
                save_grayscale_png(pred_signal, gray_dir / save_name)
            direct_depth = decode_barron_scalar_grayscale_to_depth(pred_signal)
            direct_depth = np.clip(direct_depth, args.min_depth, args.max_depth)
            path_distance_stats = None
            if not args.metrics_only:
                Image.fromarray(depth_to_uint16_mm(direct_depth), mode="I;16").save(
                    direct_mm_dir / save_name
                )
                Image.fromarray(
                    depth_to_vis(direct_depth, args.min_depth, args.max_depth), mode="L"
                ).save(direct_vis_dir / save_name)
            alignment_signal = direct_depth
            allow_invert_signal = False
        else:
            direct_depth = decode_vision_banana_rgb_to_depth(pred_rgb_array)
            direct_depth = resize_scalar_map(direct_depth, gt_depth.shape)
            direct_depth = np.clip(direct_depth, args.min_depth, args.max_depth)

            projection = project_rgb_to_vision_banana_path(pred_rgb_array)
            path_distance = resize_scalar_map(projection.distance, gt_depth.shape)
            selected_path_distance = path_distance[valid_mask & np.isfinite(path_distance)]
            path_distance_stats = {
                "path_distance_mean": float(np.mean(selected_path_distance)),
                "path_distance_median": float(np.median(selected_path_distance)),
                "path_distance_p95": float(np.percentile(selected_path_distance, 95)),
            }
            diagnostic_rgb = np.asarray(
                pred_rgb.convert("RGB").resize((gt_depth.shape[1], gt_depth.shape[0]), resample=BILINEAR), dtype=np.uint8
            )
            path_distance_stats.update(
                {f"representation_{key}": value for key, value in palette_diagnostics(diagnostic_rgb, valid_mask).items()}
            )

            if not args.metrics_only:
                Image.fromarray(depth_to_uint16_mm(direct_depth), mode="I;16").save(direct_mm_dir / save_name)
                Image.fromarray(depth_to_vis(direct_depth, args.min_depth, args.max_depth), mode="L").save(
                    direct_vis_dir / save_name
                )
                save_grayscale_png(np.clip(path_distance / max(float(path_distance.max()), 1e-6), 0.0, 1.0), path_distance_dir / save_name)

            alignment_signal = direct_depth
            allow_invert_signal = False

        if direct_depth is None:
            disparity_affine_signal = affine_generated_signal
            linear_affine_signal = affine_generated_signal
            disparity_affine_signal_kind = "generated_grayscale_relative_signal"
            linear_affine_signal_kind = "generated_grayscale_relative_signal"
        else:
            linear_affine_signal = np.asarray(direct_depth, dtype=np.float64)
            disparity_affine_signal = 1.0 / np.maximum(linear_affine_signal, 1e-6)
            disparity_affine_signal_kind = "reciprocal_decoded_metric_depth"
            linear_affine_signal_kind = "decoded_metric_depth"

        alignment = find_best_alignment(
            signal=alignment_signal,
            gt_depth=gt_depth,
            valid_mask=valid_mask,
            alignment_mode=args.alignment_mode,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            max_shift_px=args.max_shift_px,
            shift_step=args.shift_step,
            min_overlap_pixels=args.min_overlap_pixels,
            allow_invert_signal=allow_invert_signal,
        )
        noshift_alignment = find_best_alignment(
            signal=alignment_signal,
            gt_depth=gt_depth,
            valid_mask=valid_mask,
            alignment_mode=args.alignment_mode,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            max_shift_px=0,
            shift_step=1,
            min_overlap_pixels=args.min_overlap_pixels,
            allow_invert_signal=allow_invert_signal,
        )
        aligned_depth = materialize_aligned_depth(
            signal=alignment_signal,
            target_shape=gt_depth.shape,
            config=alignment,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
        noshift_depth = materialize_aligned_depth(
            signal=alignment_signal,
            target_shape=gt_depth.shape,
            config=noshift_alignment,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
        aligned_metrics, valid_pixels = compute_depth_metrics(
            pred_depth=aligned_depth,
            gt_depth=gt_depth,
            valid_mask=valid_mask,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
        noshift_metrics, noshift_valid_pixels = compute_depth_metrics(
            pred_depth=noshift_depth,
            gt_depth=gt_depth,
            valid_mask=valid_mask,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
        aligned_geometry_metrics = compute_ibims_geometry_metrics(
            sample=sample,
            pred_depth=aligned_depth,
            gt_depth=gt_depth,
            valid_mask=valid_mask,
            geometry=ibims_geometry,
        )
        noshift_geometry_metrics = compute_ibims_geometry_metrics(
            sample=sample,
            pred_depth=noshift_depth,
            gt_depth=gt_depth,
            valid_mask=valid_mask,
            geometry=ibims_geometry,
        )
        if direct_depth is not None:
            direct_metrics, direct_valid_pixels = compute_depth_metrics(
                pred_depth=direct_depth,
                gt_depth=gt_depth,
                valid_mask=valid_mask,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
            )
            direct_geometry_metrics = compute_ibims_geometry_metrics(
                sample=sample,
                pred_depth=direct_depth,
                gt_depth=gt_depth,
                valid_mask=valid_mask,
                geometry=ibims_geometry,
            )
        else:
            direct_metrics = None
            direct_valid_pixels = valid_pixels
            direct_geometry_metrics = {}

        vision_banana_direct_metrics, vision_banana_direct_valid_pixels = (
            compute_depth_metrics(
                pred_depth=vision_banana_direct_depth,
                gt_depth=gt_depth,
                valid_mask=valid_mask,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
            )
        )
        vision_banana_direct_geometry_metrics = compute_ibims_geometry_metrics(
            sample=sample,
            pred_depth=vision_banana_direct_depth,
            gt_depth=gt_depth,
            valid_mask=valid_mask,
            geometry=ibims_geometry,
        )

        disparity_affine_metrics = None
        linear_affine_metrics = None
        reference_disparity_metrics = None
        reference_linear_metrics = None
        heldout_selected_metrics = None
        disparity_affine_metadata: dict[str, float | int | str] = {}
        linear_affine_metadata: dict[str, float | int | str] = {}
        affine_unavailable_reason = None
        if not args.disable_affine_diagnostics:
            try:
                affine_min_depth, affine_max_depth = depth_range_for_dataset(
                    sample.dataset, sample.case
                )
                affine_crop = heldout_affine_crop(sample.dataset, gt_depth.shape)
                affine_dataset_mask = depth_datasets.valid_depth_mask(
                    sample,
                    gt_depth,
                    min_depth=affine_min_depth,
                    max_depth=affine_max_depth,
                )
                affine_valid_mask = depth_protocol_mask(
                    prediction_signal=disparity_affine_signal,
                    gt_depth=gt_depth,
                    dataset_mask=affine_dataset_mask,
                    crop_mask=affine_crop,
                    min_depth=affine_min_depth,
                    max_depth=affine_max_depth,
                )
                disparity_affine_prediction, disparity_affine_metadata = (
                    disparity_affine_depth(
                        disparity_affine_signal,
                        gt_depth,
                        affine_valid_mask,
                        affine_min_depth,
                        affine_max_depth,
                    )
                )
                linear_affine_prediction, linear_affine_metadata = linear_affine_depth(
                    linear_affine_signal,
                    gt_depth,
                    affine_valid_mask,
                    affine_min_depth,
                    affine_max_depth,
                )
                disparity_affine_metrics, _ = compute_depth_metrics(
                    disparity_affine_prediction,
                    gt_depth,
                    affine_valid_mask,
                    affine_min_depth,
                    affine_max_depth,
                )
                linear_affine_metrics, _ = compute_depth_metrics(
                    linear_affine_prediction,
                    gt_depth,
                    affine_valid_mask,
                    affine_min_depth,
                    affine_max_depth,
                )
                disparity_affine_metadata.update(
                    {
                        "min_depth": affine_min_depth,
                        "max_depth": affine_max_depth,
                        "input_signal": disparity_affine_signal_kind,
                    }
                )
                linear_affine_metadata.update(
                    {
                        "min_depth": affine_min_depth,
                        "max_depth": affine_max_depth,
                        "input_signal": linear_affine_signal_kind,
                    }
                )
                if heldout_depth_calibration is not None:
                    selected_affine_space = str(
                        heldout_depth_calibration["selected_affine_space"]
                    )
                    heldout_selected_metrics = (
                        disparity_affine_metrics
                        if selected_affine_space == "disparity"
                        else linear_affine_metrics
                    )
            except (ValueError, FileNotFoundError) as exc:
                affine_unavailable_reason = str(exc)

            try:
                reference_min_depth, reference_max_depth = depth_range_for_dataset(
                    sample.dataset, sample.case
                )
                reference_crop = reference_standard_crop(sample.dataset, gt_depth.shape)
                reference_dataset_mask = depth_datasets.valid_depth_mask(
                    sample,
                    gt_depth,
                    min_depth=reference_min_depth,
                    max_depth=reference_max_depth,
                )
                reference_valid_mask = depth_protocol_mask(
                    prediction_signal=reference_signal,
                    gt_depth=gt_depth,
                    dataset_mask=reference_dataset_mask,
                    crop_mask=reference_crop,
                    min_depth=reference_min_depth,
                    max_depth=reference_max_depth,
                )
                reference_disparity_prediction, _ = disparity_affine_depth(
                    reference_signal,
                    gt_depth,
                    reference_valid_mask,
                    reference_min_depth,
                    reference_max_depth,
                )
                reference_linear_prediction, _ = linear_affine_depth(
                    reference_signal,
                    gt_depth,
                    reference_valid_mask,
                    reference_min_depth,
                    reference_max_depth,
                )
                reference_disparity_metrics, _ = compute_depth_metrics(
                    reference_disparity_prediction,
                    gt_depth,
                    reference_valid_mask,
                    reference_min_depth,
                    reference_max_depth,
                )
                reference_linear_metrics, _ = compute_depth_metrics(
                    reference_linear_prediction,
                    gt_depth,
                    reference_valid_mask,
                    reference_min_depth,
                    reference_max_depth,
                )
            except (ValueError, FileNotFoundError) as exc:
                affine_unavailable_reason = str(exc)

        if not args.metrics_only:
            Image.fromarray(depth_to_uint16_mm(aligned_depth), mode="I;16").save(aligned_mm_dir / save_name)
            Image.fromarray(depth_to_vis(aligned_depth, args.min_depth, args.max_depth), mode="L").save(
                aligned_vis_dir / save_name
            )
            Image.fromarray(depth_to_uint16_mm(noshift_depth), mode="I;16").save(noshift_mm_dir / save_name)
            Image.fromarray(depth_to_vis(noshift_depth, args.min_depth, args.max_depth), mode="L").save(
                noshift_vis_dir / save_name
            )

        record = {
            "split": sample.split,
            "dataset": sample.dataset,
            "case": sample.case,
            "sample_id": sample.sample_id,
            "image_path": str(sample.image_path),
            "depth_path": str(sample.depth_path),
            "mask_path": str(sample.mask_path) if sample.mask_path is not None else None,
            "geometry_path": str(geometry_path_value) if geometry_path_value else None,
            "focal_length": float(sample.focal_length),
            "valid_pixels": direct_valid_pixels,
            "vision_banana_direct_valid_pixels": vision_banana_direct_valid_pixels,
            "aligned_valid_pixels": valid_pixels,
            "noshift_valid_pixels": noshift_valid_pixels,
            "alignment_mode": alignment.mode,
            "signal_inverted": bool(alignment.invert_signal),
            "shift_y_px": int(alignment.dy),
            "shift_x_px": int(alignment.dx),
            "alignment_scale": float(alignment.scale),
            "alignment_shift": float(alignment.shift),
            "alignment_score_abs_rel": float(alignment.score),
            "alignment_overlap_pixels": int(alignment.overlap_pixels),
            "noshift_alignment_mode": noshift_alignment.mode,
            "noshift_signal_inverted": bool(noshift_alignment.invert_signal),
            "noshift_shift_y_px": int(noshift_alignment.dy),
            "noshift_shift_x_px": int(noshift_alignment.dx),
            "noshift_alignment_scale": float(noshift_alignment.scale),
            "noshift_alignment_shift": float(noshift_alignment.shift),
            "noshift_alignment_score_abs_rel": float(noshift_alignment.score),
            "noshift_alignment_overlap_pixels": int(noshift_alignment.overlap_pixels),
        }
        if direct_metrics is not None:
            record.update({f"direct_{name}": value for name, value in direct_metrics.items()})
            record.update({f"direct_{name}": value for name, value in direct_geometry_metrics.items()})
            record.update(path_distance_stats or {})
        record.update(
            {
                f"vision_banana_direct_{name}": value
                for name, value in vision_banana_direct_metrics.items()
            }
        )
        record.update(
            {
                f"vision_banana_direct_{name}": value
                for name, value in vision_banana_direct_geometry_metrics.items()
            }
        )
        if disparity_affine_metrics is not None and linear_affine_metrics is not None:
            record.update(
                {
                    f"disparity_affine_{name}": value
                    for name, value in disparity_affine_metrics.items()
                }
            )
            record.update(
                {f"linear_affine_{name}": value for name, value in linear_affine_metrics.items()}
            )
            record.update(
                {
                    f"disparity_affine_{name}": value
                    for name, value in disparity_affine_metadata.items()
                    if name != "protocol"
                }
            )
            record.update(
                {
                    f"linear_affine_{name}": value
                    for name, value in linear_affine_metadata.items()
                    if name != "protocol"
                }
            )
            if heldout_selected_metrics is not None:
                record.update(
                    {
                        f"heldout_selected_affine_{name}": value
                        for name, value in heldout_selected_metrics.items()
                    }
                )
                record["heldout_selected_affine_space"] = str(
                    heldout_depth_calibration["selected_affine_space"]
                )
        elif affine_unavailable_reason is not None:
            record["affine_diagnostics_unavailable_reason"] = affine_unavailable_reason
        if reference_disparity_metrics is not None:
            record.update(
                {
                    f"reference_disparity_affine_{name}": value
                    for name, value in reference_disparity_metrics.items()
                }
            )
        if reference_linear_metrics is not None:
            record.update(
                {
                    f"reference_linear_affine_{name}": value
                    for name, value in reference_linear_metrics.items()
                }
            )
        record.update({f"aligned_{name}": value for name, value in aligned_metrics.items()})
        record.update({f"aligned_{name}": value for name, value in aligned_geometry_metrics.items()})
        record.update({f"noshift_{name}": value for name, value in noshift_metrics.items()})
        record.update({f"noshift_{name}": value for name, value in noshift_geometry_metrics.items()})

        primary_metrics = (
            direct_metrics
            if direct_metrics is not None
            else (reference_disparity_metrics or aligned_metrics)
        )
        record.update(primary_metrics)
        if resume is not None:
            resume.add(
                "record",
                sample.output_stem,
                record,
                (directory / save_name for directory in directories),
            )
        else:
            per_sample_records.append(record)
            per_sample_record_keys.append(sample.output_stem)

    if resume is not None:
        resume.flush()
        record_items = resume.records_with_keys_in_sample_order("record")
        per_sample_record_keys = [key for key, _ in record_items]
        per_sample_records = [record for _, record in record_items]

    if args.num_shards > 1:
        full_sample_keys = [sample.output_stem for sample in full_samples]
        identity_args = {
            key: value
            for key, value in vars(args).items()
            if key
            not in {
                "shard_index",
                "shard_sync_dir",
                "shard_sync_timeout_seconds",
                "resume_state_dir",
            }
        }
        run_identity = sharding_run_identity(
            identity_args, full_sample_keys, args.num_shards
        )
        merged = synchronize_shard_records(
            sync_dir=args.shard_sync_dir,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            run_identity=run_identity,
            full_sample_keys=full_sample_keys,
            record_items=list(zip(per_sample_record_keys, per_sample_records)),
            skipped_items=(),
            timeout_seconds=args.shard_sync_timeout_seconds,
        )
        if merged is None:
            if resume is not None:
                resume.close()
            print(
                f"Evaluation shard {args.shard_index}/{args.num_shards} published; "
                "aggregate is written by shard zero."
            )
            return
        per_sample_records, unexpected_skips = merged
        if unexpected_skips:
            raise RuntimeError("depth evaluation shards unexpectedly contained skipped records")
        samples = full_samples

    if args.benchmark_protocol == "marigold_v2_ppd":
        metric_names = ("abs_rel", "delta1", "delta2", "delta3")
        ppd_metrics = {
            name: float(
                np.mean(
                    [float(record[f"marigold_v2_ppd_{name}"]) for record in per_sample_records]
                )
            )
            for name in metric_names
        }
        ppd_metrics["abs_rel_percent"] = ppd_metrics["abs_rel"] * 100.0
        ppd_metrics["delta1_percent"] = ppd_metrics["delta1"] * 100.0
        affine_metrics_summary = None
        if args.depth_encoding == DEPTH_ENCODING_RELATIVE_INVERSE_GRAYSCALE:
            affine_metrics_summary = {
                name: float(np.mean([float(record[f"ppd_split_disparity_affine_{name}"])
                                     for record in per_sample_records]))
                for name in metric_names
            }
            affine_metrics_summary["abs_rel_percent"] = affine_metrics_summary["abs_rel"] * 100.0
            affine_metrics_summary["delta1_percent"] = affine_metrics_summary["delta1"] * 100.0
        metrics_by_case = {}
        primary_prefix = (
            "ppd_split_disparity_affine" if ppd_primary_alignment == "disparity_affine"
            else "marigold_v2_ppd"
        )
        primary_metrics = (
            affine_metrics_summary if ppd_primary_alignment == "disparity_affine"
            else ppd_metrics
        )
        primary_metrics_by_case = {}
        for case, records in sorted(records_by_case(per_sample_records).items()):
            primary_metrics_by_case[case] = {
                name: float(np.mean([float(record[f"{primary_prefix}_{name}"])
                                     for record in records]))
                for name in metric_names
            }
            metrics_by_case[case] = {
                name: float(
                    np.mean(
                        [float(record[f"marigold_v2_ppd_{name}"]) for record in records]
                    )
                )
                for name in metric_names
            }
        generation_summary = {
            "source": (
                "archived_raw_predictions"
                if prediction_dir is not None
                else "model_inference"
            ),
            "prediction_dir": str(prediction_dir) if prediction_dir else None,
            "model": args.model,
            "model_family": args.model_family,
            "transformer_path": args.transformer_path,
            "lora_path": args.lora_path,
            "prompt": prompt,
            "prompt_sha256": prompt_sha256(prompt),
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "seed": None if args.legacy_unseeded else args.seed,
            "seed_mode": (
                "legacy_process_rng"
                if args.legacy_unseeded
                else "fixed_per_sample_reset"
            ),
            "native_resolution": True,
            "eth3d_inference_resolution": [1008, 672],
        }
        summary = {
            "dataset": args.dataset,
            "dataset_root": str(dataset_root),
            "requested_split": args.split,
            "resolved_splits": resolved_splits,
            "resolved_cases": resolved_cases,
            "images_evaluated": len(per_sample_records),
            "depth_encoding": args.depth_encoding,
            "benchmark_protocol": MARIGOLD_V2_PPD_PROTOCOL,
            "sample_subset": subset_metadata,
            "benchmark_split_root": str(Path(args.benchmark_split_root).resolve()),
            "marigold_v2_ppd_metrics": ppd_metrics,
            "ppd_split_disparity_affine_metrics": affine_metrics_summary,
            "primary_metrics": primary_metrics,
            "primary_metrics_by_case": primary_metrics_by_case,
            "marigold_v2_ppd_metrics_by_case": metrics_by_case,
            "depth_evaluation": {
                "primary_lane": (
                    "ppd_split_disparity_affine_v1" if ppd_primary_alignment == "disparity_affine"
                    else "marigold_v2_ppd_ransac_log"
                ),
                "primary_alignment": (
                    "per-image least-squares scale+shift in inverse depth"
                    if ppd_primary_alignment == "disparity_affine"
                    else "per-image seeded RANSAC scale+shift in log-depth"
                ),
                "requested_alignment": args.ppd_primary_alignment,
                "log_lane_encoding_compatible": args.depth_encoding not in RELATIVE_DEPTH_ENCODINGS,
                "log_lane_note": (
                    "Orientation only, not inverse/linear-depth to log-depth conversion; legacy diagnostic."
                    if args.depth_encoding in RELATIVE_DEPTH_ENCODINGS else None
                ),
                "comparison_note": "Shared PPD splits/masks do not make disparity-affine identical to Marigold V2 log scoring.",
                "aggregation": "arithmetic mean of per-image metrics",
                "input_signal": (
                    "generated_inverse_grayscale_as_disparity"
                    if ppd_primary_alignment == "disparity_affine" else (
                    "negative_generated_inverse_grayscale_as_log_depth"
                    if args.depth_encoding == DEPTH_ENCODING_RELATIVE_INVERSE_GRAYSCALE
                    else (
                        "generated_relative_grayscale_as_log_depth"
                        if args.depth_encoding in RELATIVE_DEPTH_ENCODINGS
                        else "log of fixed decoded metric-depth representation"
                    ))
                ),
                "spatial_shift_search": False,
                "polarity_search": False,
                "representation_oracle": False,
                "secondary_lane": (
                    "marigold_v2_ppd_ransac_log (legacy direct-gray diagnostic)"
                    if ppd_primary_alignment == "disparity_affine" else (
                    "ppd_split_disparity_affine_v1 (same split/mask; not exact RINO protocol)"
                    if affine_metrics_summary is not None else None)
                ),
            },
            "generation": generation_summary,
        }
        if args.num_shards > 1:
            summary["execution_sharding"] = {
                "strategy": "round_robin_process_per_gpu",
                "num_shards": args.num_shards,
                "aggregation": "global_sample_order_before_metric_aggregation",
            }
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / f"{args.dataset}_test_results.json"
        per_sample_path = out_dir / f"{args.dataset}_test_per_sample.jsonl"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with per_sample_path.open("w", encoding="utf-8") as handle:
            for record in per_sample_records:
                handle.write(json.dumps(record) + "\n")
        print(f"\nPPD benchmark splits / {ppd_primary_alignment} Evaluation Results")
        print("=" * 50)
        print(f"Images evaluated: {len(per_sample_records)}")
        print(f"AbsRel (%): {primary_metrics['abs_rel_percent']:.6f}")
        print(f"delta1 (%): {primary_metrics['delta1_percent']:.6f}")
        print(f"Saved summary to: {summary_path}")
        return

    grouped_records = records_by_case(per_sample_records)
    if args.primary_case:
        if args.primary_case not in grouped_records:
            raise ValueError(
                f"Requested primary case {args.primary_case!r} is absent; "
                f"resolved cases: {', '.join(sorted(grouped_records))}"
            )
        primary_records = grouped_records[args.primary_case]
    elif args.dataset == "ibims1":
        primary_records = grouped_records.get("standard", per_sample_records)
    else:
        primary_records = per_sample_records
    metrics_summary = aggregate_metrics(primary_records)
    combined_metrics_summary = aggregate_metrics(per_sample_records)
    metrics_by_case = {
        case: aggregate_metrics(case_records)
        for case, case_records in sorted(grouped_records.items())
    }
    aligned_metrics_summary = aggregate_prefixed_metrics(primary_records, "aligned")
    aligned_metrics_by_case = {
        case: aggregate_prefixed_metrics(case_records, "aligned")
        for case, case_records in sorted(grouped_records.items())
    }
    noshift_metrics_summary = aggregate_prefixed_metrics(primary_records, "noshift")
    noshift_metrics_by_case = {
        case: aggregate_prefixed_metrics(case_records, "noshift")
        for case, case_records in sorted(grouped_records.items())
    }
    direct_metrics_summary = (
        aggregate_prefixed_metrics(primary_records, "direct")
        if args.depth_encoding in {
            DEPTH_ENCODING_VISION_BANANA_RGB,
            DEPTH_ENCODING_BARRON_SCALAR_GRAYSCALE,
        }
        else None
    )
    direct_metrics_by_case = (
        {
            case: aggregate_prefixed_metrics(case_records, "direct")
            for case, case_records in sorted(grouped_records.items())
        }
        if args.depth_encoding in {
            DEPTH_ENCODING_VISION_BANANA_RGB,
            DEPTH_ENCODING_BARRON_SCALAR_GRAYSCALE,
        }
        else None
    )
    vision_banana_direct_metrics_summary = aggregate_prefixed_metrics(
        primary_records, "vision_banana_direct"
    )
    vision_banana_direct_metrics_by_case = {
        case: aggregate_prefixed_metrics(case_records, "vision_banana_direct")
        for case, case_records in sorted(grouped_records.items())
    }
    disparity_affine_metrics_summary = aggregate_optional_prefixed_metrics(
        primary_records, "disparity_affine"
    )
    disparity_affine_metrics_by_case = {
        case: case_metrics
        for case, case_metrics in (
            (
                case,
                aggregate_optional_prefixed_metrics(case_records, "disparity_affine"),
            )
            for case, case_records in sorted(grouped_records.items())
        )
        if case_metrics is not None
    }
    linear_affine_metrics_summary = aggregate_optional_prefixed_metrics(
        primary_records, "linear_affine"
    )
    linear_affine_metrics_by_case = {
        case: case_metrics
        for case, case_metrics in (
            (case, aggregate_optional_prefixed_metrics(case_records, "linear_affine"))
            for case, case_records in sorted(grouped_records.items())
        )
        if case_metrics is not None
    }
    reference_disparity_affine_metrics_summary = aggregate_optional_prefixed_metrics(
        primary_records, "reference_disparity_affine"
    )
    reference_disparity_affine_metrics_by_case = {
        case: case_metrics
        for case, case_metrics in (
            (
                case,
                aggregate_optional_prefixed_metrics(
                    case_records, "reference_disparity_affine"
                ),
            )
            for case, case_records in sorted(grouped_records.items())
        )
        if case_metrics is not None
    }
    reference_linear_affine_metrics_summary = aggregate_optional_prefixed_metrics(
        primary_records, "reference_linear_affine"
    )
    reference_linear_affine_metrics_by_case = {
        case: case_metrics
        for case, case_metrics in (
            (
                case,
                aggregate_optional_prefixed_metrics(
                    case_records, "reference_linear_affine"
                ),
            )
            for case, case_records in sorted(grouped_records.items())
        )
        if case_metrics is not None
    }
    heldout_selected_affine_metrics_summary = aggregate_optional_prefixed_metrics(
        primary_records, "heldout_selected_affine"
    )
    heldout_selected_affine_metrics_by_case = {
        case: case_metrics
        for case, case_metrics in (
            (
                case,
                aggregate_optional_prefixed_metrics(
                    case_records, "heldout_selected_affine"
                ),
            )
            for case, case_records in sorted(grouped_records.items())
        )
        if case_metrics is not None
    }
    aligned_geometry_metrics_summary = aggregate_prefixed_geometry_metrics(primary_records, "aligned")
    aligned_geometry_metrics_by_case = {
        case: case_geometry_metrics
        for case, case_geometry_metrics in (
            (case, aggregate_prefixed_geometry_metrics(case_records, "aligned"))
            for case, case_records in sorted(grouped_records.items())
        )
        if case_geometry_metrics is not None
    }
    noshift_geometry_metrics_summary = aggregate_prefixed_geometry_metrics(primary_records, "noshift")
    noshift_geometry_metrics_by_case = {
        case: case_geometry_metrics
        for case, case_geometry_metrics in (
            (case, aggregate_prefixed_geometry_metrics(case_records, "noshift"))
            for case, case_records in sorted(grouped_records.items())
        )
        if case_geometry_metrics is not None
    }
    direct_geometry_metrics_summary = (
        aggregate_prefixed_geometry_metrics(primary_records, "direct")
        if args.depth_encoding in {
            DEPTH_ENCODING_VISION_BANANA_RGB,
            DEPTH_ENCODING_BARRON_SCALAR_GRAYSCALE,
        }
        else None
    )
    direct_geometry_metrics_by_case = (
        {
            case: case_geometry_metrics
            for case, case_geometry_metrics in (
                (case, aggregate_prefixed_geometry_metrics(case_records, "direct"))
                for case, case_records in sorted(grouped_records.items())
            )
            if case_geometry_metrics is not None
        }
        if args.depth_encoding in {
            DEPTH_ENCODING_VISION_BANANA_RGB,
            DEPTH_ENCODING_BARRON_SCALAR_GRAYSCALE,
        }
        else None
    )
    vision_banana_direct_geometry_metrics_summary = (
        aggregate_prefixed_geometry_metrics(primary_records, "vision_banana_direct")
    )
    vision_banana_direct_geometry_metrics_by_case = {
        case: case_geometry_metrics
        for case, case_geometry_metrics in (
            (
                case,
                aggregate_prefixed_geometry_metrics(
                    case_records, "vision_banana_direct"
                ),
            )
            for case, case_records in sorted(grouped_records.items())
        )
        if case_geometry_metrics is not None
    }
    mode_counts = Counter(str(record["alignment_mode"]) for record in per_sample_records)
    inverted_count = sum(int(bool(record["signal_inverted"])) for record in per_sample_records)
    noshift_mode_counts = Counter(str(record["noshift_alignment_mode"]) for record in per_sample_records)
    noshift_inverted_count = sum(int(bool(record["noshift_signal_inverted"])) for record in per_sample_records)
    representation_diagnostics = aggregate_diagnostic_fields(primary_records)

    if archived_generation is not None:
        generation_summary = dict(archived_generation)
        generation_summary["source"] = "archived_raw_predictions"
        generation_summary["original_result_json"] = str(
            archived_generation_result_path
        )
        generation_summary["original_prediction_dir"] = generation_summary.get(
            "prediction_dir"
        )
        generation_summary["prediction_dir"] = str(prediction_dir)
        generation_summary["prompt"] = prompt
        generation_summary["prompt_sha256"] = prompt_sha256(prompt)
    else:
        generation_summary = {
            "source": "archived_raw_predictions" if prediction_dir is not None else "model_inference",
            "prediction_dir": str(prediction_dir) if prediction_dir is not None else None,
            "model": args.model,
            "model_family": args.model_family,
            "transformer_path": args.transformer_path,
            "lora_path": args.lora_path,
            "prompt": prompt,
            "prompt_sha256": prompt_sha256(prompt),
            "negative_prompt": qwen_negative_prompt,
            "negative_prompt_sha256": (
                prompt_sha256(qwen_negative_prompt)
                if qwen_negative_prompt is not None
                else None
            ),
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "seed": None if args.legacy_unseeded else args.seed,
            "seed_mode": (
                "legacy_process_rng"
                if args.legacy_unseeded
                else "fixed_per_sample_reset"
            ),
            "native_resolution": bool(args.native_resolution),
        }

    durable_resume_summary = (
        resume.summary() if resume is not None else {"enabled": False}
    )
    if args.num_shards > 1 and resume is not None:
        durable_resume_summary = summarize_sharded_journals(
            args.resume_state_dir,
            num_shards=args.num_shards,
            committed_samples=len(per_sample_records),
            chunk_size=args.resume_chunk_size,
        )

    summary = {
        "dataset": args.dataset,
        "dataset_root": str(dataset_root),
        "requested_split": args.split,
        "resolved_splits": resolved_splits,
        "resolved_cases": resolved_cases,
        "primary_case": args.primary_case,
        "images_evaluated": len(per_sample_records),
        "dataset_role": args.dataset_role,
        "sample_subset": subset_metadata,
        "depth_encoding": args.depth_encoding,
        "metrics": metrics_summary,
        "combined_metrics": combined_metrics_summary,
        "metrics_by_case": metrics_by_case,
        "direct_metrics": direct_metrics_summary,
        "direct_metrics_by_case": direct_metrics_by_case,
        "vision_banana_direct_metrics": vision_banana_direct_metrics_summary,
        "vision_banana_direct_metrics_by_case": (
            vision_banana_direct_metrics_by_case
        ),
        "disparity_affine_metrics": disparity_affine_metrics_summary,
        "disparity_affine_metrics_by_case": disparity_affine_metrics_by_case,
        "linear_affine_metrics": linear_affine_metrics_summary,
        "linear_affine_metrics_by_case": linear_affine_metrics_by_case,
        "reference_disparity_affine_metrics": reference_disparity_affine_metrics_summary,
        "reference_disparity_affine_metrics_by_case": (
            reference_disparity_affine_metrics_by_case
        ),
        "reference_linear_affine_metrics": reference_linear_affine_metrics_summary,
        "reference_linear_affine_metrics_by_case": (
            reference_linear_affine_metrics_by_case
        ),
        "heldout_selected_affine_metrics": heldout_selected_affine_metrics_summary,
        "heldout_selected_affine_metrics_by_case": (
            heldout_selected_affine_metrics_by_case
        ),
        "aligned_metrics": aligned_metrics_summary,
        "aligned_metrics_by_case": aligned_metrics_by_case,
        "noshift_aligned_metrics": noshift_metrics_summary,
        "noshift_aligned_metrics_by_case": noshift_metrics_by_case,
        "direct_ibims_geometry_metrics": direct_geometry_metrics_summary,
        "direct_ibims_geometry_metrics_by_case": direct_geometry_metrics_by_case,
        "vision_banana_direct_ibims_geometry_metrics": (
            vision_banana_direct_geometry_metrics_summary
        ),
        "vision_banana_direct_ibims_geometry_metrics_by_case": (
            vision_banana_direct_geometry_metrics_by_case
        ),
        "representation_diagnostics": representation_diagnostics,
        "aligned_ibims_geometry_metrics": aligned_geometry_metrics_summary,
        "aligned_ibims_geometry_metrics_by_case": aligned_geometry_metrics_by_case,
        "noshift_ibims_geometry_metrics": noshift_geometry_metrics_summary,
        "noshift_ibims_geometry_metrics_by_case": noshift_geometry_metrics_by_case,
        "metrics_protocol": (
            f"Top-level metrics select case {args.primary_case!r}; all evaluated "
            "cases are retained under *_metrics_by_case."
            if args.primary_case
            else (
            "For iBims-1 multi-case runs, top-level metrics are computed on the standard case only; "
            "auxiliary results are reported under metrics_by_case."
            if args.dataset == "ibims1" and len(metrics_by_case) > 1
            else (
                "For DIODE multi-case runs, top-level metrics are combined over evaluated samples; "
                "indoor and outdoor results are reported under metrics_by_case."
                if args.dataset == "diode" and len(metrics_by_case) > 1
                else (
                    "nuScenes metrics are computed only on sparse projected LiDAR valid pixels; "
                    "report as sparse LiDAR-derived outdoor-driving evaluation."
                    if args.dataset == "nuscenes"
                    else "Top-level metrics are computed over all evaluated samples."
                )
            )
            )
        ),
        "alignment": {
            "selection_metric": "abs_rel",
            "mode_requested": args.alignment_mode,
            "mode_counts": dict(mode_counts),
            "signal_inverted_count": inverted_count,
            "max_shift_px": args.max_shift_px,
            "shift_step": args.shift_step,
            "min_overlap_pixels": args.min_overlap_pixels,
        },
        "noshift_alignment": {
            "selection_metric": "abs_rel",
            "mode_requested": args.alignment_mode,
            "mode_counts": dict(noshift_mode_counts),
            "signal_inverted_count": noshift_inverted_count,
            "max_shift_px": 0,
            "shift_step": 1,
            "min_overlap_pixels": args.min_overlap_pixels,
        },
        "depth_evaluation": {
            "protocol_version": args.protocol_version,
            "dataset_role": args.dataset_role,
            "primary_case": args.primary_case,
            "physical_quantity": "projective_camera_z_meters",
            "primary_lane": (
                "direct_metric"
                if direct_metrics_summary is not None
                else "reference_disparity_affine"
            ),
            "primary_alignment": (
                "none"
                if direct_metrics_summary is not None
                else "per_image_disparity_scale_shift"
            ),
            "configuration": normalized_depth_configuration,
            "configuration_sha256": depth_configuration_sha256,
            "calibration_candidates_only": bool(
                args.depth_calibration_candidates_only
            ),
            "vision_banana_direct_rescore_only": bool(
                args.vision_banana_direct_rescore_only
            ),
            "lanes": {
                "vision_banana_rgb_direct": {
                    "protocol": DEPTH_VISION_BANANA_RGB_DIRECT,
                    "metrics_key": "vision_banana_direct_metrics",
                    "status": "available",
                    "decoder": "fixed Vision Banana RGB path projection and inverse Barron transform",
                    "alignment": "none",
                    "input": "raw generated RGB for every prompt and declared representation",
                },
                "direct_metric": {
                    "protocol": DEPTH_DIRECT_METRIC,
                    "metrics_key": "direct_metrics",
                    "status": (
                        "available" if direct_metrics_summary is not None else "not_applicable"
                    ),
                    "reason": (
                        None
                        if direct_metrics_summary is not None
                        else "The declared output representation is relative grayscale, not metric depth."
                    ),
                },
                "representation_disparity_affine": {
                    "protocol": DEPTH_DISPARITY_AFFINE,
                    "metrics_key": "disparity_affine_metrics",
                    "status": (
                        "available"
                        if disparity_affine_metrics_summary is not None
                        else "unavailable"
                    ),
                    "input_signal": (
                        "generated_grayscale_relative_signal"
                        if args.depth_encoding in RELATIVE_DEPTH_ENCODINGS
                        else "reciprocal_decoded_metric_depth"
                    ),
                    "alignment": "per-image least-squares disparity scale+shift",
                },
                "representation_linear_affine": {
                    "protocol": DEPTH_LINEAR_AFFINE,
                    "metrics_key": "linear_affine_metrics",
                    "status": (
                        "available"
                        if linear_affine_metrics_summary is not None
                        else "unavailable"
                    ),
                    "input_signal": (
                        "generated_grayscale_relative_signal"
                        if args.depth_encoding in RELATIVE_DEPTH_ENCODINGS
                        else "decoded_metric_depth"
                    ),
                    "alignment": "per-image least-squares depth scale+shift",
                },
                "reference_disparity_affine": {
                    "protocol": DEPTH_REFERENCE_DISPARITY_AFFINE,
                    "metrics_key": "reference_disparity_affine_metrics",
                    "status": (
                        "available"
                        if reference_disparity_affine_metrics_summary is not None
                        else "unavailable"
                    ),
                    "decoder": "ITU-R BT.709 luminance",
                    "resize": "scipy.ndimage.zoom(order=1)",
                    "alignment": "per-image least-squares disparity scale+shift",
                    "aggregation": "arithmetic mean of per-image metrics",
                    "dataset_scope": "frozen_reference_math_all_supported_datasets",
                },
                "reference_linear_affine": {
                    "protocol": DEPTH_REFERENCE_LINEAR_AFFINE,
                    "metrics_key": "reference_linear_affine_metrics",
                    "status": (
                        "available"
                        if reference_linear_affine_metrics_summary is not None
                        else "unavailable"
                    ),
                    "decoder": "ITU-R BT.709 luminance",
                    "resize": "scipy.ndimage.zoom(order=1)",
                    "alignment": "per-image least-squares depth scale+shift",
                    "aggregation": "arithmetic mean of per-image metrics",
                    "dataset_scope": "frozen_reference_math_all_supported_datasets",
                },
                "heldout_selected_affine": {
                    "protocol": DEPTH_HELDOUT_SELECTED_AFFINE,
                    "metrics_key": "heldout_selected_affine_metrics",
                    "status": (
                        "calibration_candidates_only"
                        if args.depth_calibration_candidates_only
                        else (
                            "available"
                            if heldout_selected_affine_metrics_summary is not None
                            else "unavailable"
                        )
                    ),
                    "selected_affine_space": (
                        heldout_depth_calibration.get("selected_affine_space")
                        if heldout_depth_calibration
                        else None
                    ),
                    "calibration_record": (
                        str(Path(args.heldout_depth_calibration_record).resolve())
                        if args.heldout_depth_calibration_record
                        else None
                    ),
                    "calibration_record_sha256": (
                        hashlib.sha256(
                            Path(args.heldout_depth_calibration_record).read_bytes()
                        ).hexdigest()
                        if args.heldout_depth_calibration_record
                        else None
                    ),
                },
            },
        },
        "relative_depth_protocols": {
            "protocol_version": args.protocol_version,
            "vision_banana_direct_metrics": {
                "lane": "fixed_vision_banana_rgb_direct",
                "decoder": "vision_banana_rgb",
                "comparison_scope": "vision_banana_direct_protocol",
                "description": (
                    "Raw generated RGB is always decoded with the fixed Vision "
                    "Banana RGB path and scored without per-image alignment, "
                    "irrespective of the prompt's requested representation."
                ),
            },
            "direct_metrics": {
                "lane": "unaligned_metric_depth",
                "decoder": args.depth_encoding,
                "comparison_scope": (
                    "vision_banana_direct_protocol"
                    if args.depth_encoding == DEPTH_ENCODING_VISION_BANANA_RGB
                    else (
                        "bounded_metric_grayscale_ablation"
                        if args.depth_encoding == DEPTH_ENCODING_BARRON_SCALAR_GRAYSCALE
                        else "not_applicable"
                    )
                ),
                "description": (
                    "The declared invertible representation is decoded to metric "
                    "camera-z depth and scored with no per-image calibration."
                ),
            },
            "disparity_affine_metrics": {
                "lane": "relative_depth_affine_diagnostic",
                "protocol": DEPTH_DISPARITY_AFFINE,
                "input_signal": (
                    "generated_grayscale_relative_signal"
                    if args.depth_encoding in RELATIVE_DEPTH_ENCODINGS
                    else "reciprocal_decoded_metric_depth"
                ),
                "description": (
                    "The declared disparity input signal is scored with fixed "
                    "least-squares disparity-affine post-processing, dataset crop/cap, and no "
                    "shift, mode search, or polarity enumeration. For Vision Banana "
                    "metric RGB output this signal is reciprocal decoded metric depth. "
                    "This lane measures relative geometry, not raw metric scale."
                ),
            },
            "linear_affine_metrics": {
                "lane": "relative_depth_affine_diagnostic",
                "protocol": DEPTH_LINEAR_AFFINE,
                "input_signal": (
                    "generated_grayscale_relative_signal"
                    if args.depth_encoding in RELATIVE_DEPTH_ENCODINGS
                    else "decoded_metric_depth"
                ),
                "description": (
                    "The declared linear input signal is scored with fixed least-squares "
                    "depth-affine post-processing. For Vision Banana metric RGB output "
                    "this signal is decoded metric depth. This lane measures relative "
                    "geometry, not uncalibrated metric scale."
                ),
            },
            "noshift_aligned_metrics": {
                "lane": "oracle_diagnostic",
                "description": (
                    "Legacy GT-selected best-of linear/log/inverse calibration without translation."
                ),
            },
            "aligned_metrics": {
                "lane": "oracle_diagnostic",
                "description": (
                    "Legacy GT-selected best-of modes, polarity, and configured translation search."
                ),
            },
        },
        "generation": generation_summary,
        "depth_range_m": {
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
        },
        "durable_resume": durable_resume_summary,
    }
    if args.num_shards > 1:
        summary["execution_sharding"] = {
            "strategy": "round_robin_process_per_gpu",
            "num_shards": args.num_shards,
            "aggregation": "global_sample_order_before_metric_aggregation",
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    split_name = depth_datasets.canonicalize_split_name(args.dataset, args.split)
    summary_path = out_dir / f"{args.dataset}_{split_name}_results.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    per_sample_path = out_dir / f"{args.dataset}_{split_name}_per_sample.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as handle:
        for record in per_sample_records:
            handle.write(json.dumps(record) + "\n")

    print("\nEvaluation Results")
    print("=" * 50)
    print(f"Dataset root: {dataset_root}")
    print(f"Resolved splits: {', '.join(resolved_splits)}")
    print(f"Resolved cases: {', '.join(resolved_cases)}")
    print(f"Images evaluated: {len(per_sample_records)}")
    print(f"Depth encoding: {args.depth_encoding}")
    for name, value in metrics_summary.items():
        print(f"{name}: {value:.6f}")
    print("=" * 50)
    print(f"Saved summary to: {summary_path}")
    print(f"Saved per-sample metrics to: {per_sample_path}")
    if not args.metrics_only:
        print(f"Raw RGB predictions: {raw_rgb_dir}")
    if not args.metrics_only:
        if args.depth_encoding in RELATIVE_DEPTH_ENCODINGS:
            print(f"Decoded grayscale predictions: {gray_dir}")
        else:
            print(f"Direct metric-depth PNGs: {direct_mm_dir}")
            print(f"Direct depth visualizations: {direct_vis_dir}")
            print(f"RGB path-distance maps: {path_distance_dir}")
        print(f"Aligned metric-depth PNGs: {aligned_mm_dir}")
        print(f"Aligned depth visualizations: {aligned_vis_dir}")
        print(f"No-shift aligned metric-depth PNGs: {noshift_mm_dir}")
        print(f"No-shift aligned depth visualizations: {noshift_vis_dir}")


if __name__ == "__main__":
    main()
