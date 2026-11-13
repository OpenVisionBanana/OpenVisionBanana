"""Shared, side-effect-free diagnostics for encoded geometry predictions.

These metrics deliberately separate *representation validity* (is an RGB value
a legal encoding?) from *physical accuracy* (does it describe the right depth
or normal?).  They are diagnostics, not replacements for benchmark metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_prep.depth.encoding import project_rgb_to_vision_banana_path


def _finite(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    return values[valid]


def distribution(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | int]:
    selected = _finite(values, mask)
    if selected.size == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "p95": float("nan")}
    return {
        "count": int(selected.size),
        "mean": float(np.mean(selected)),
        "median": float(np.median(selected)),
        "p95": float(np.quantile(selected, 0.95)),
    }


def palette_diagnostics(
    rgb: np.ndarray,
    valid_mask: np.ndarray | None = None,
    tolerances: tuple[float, ...] = (1.0 / 255.0, 2.0 / 255.0, 5.0 / 255.0),
) -> dict[str, float | int]:
    """Measure Euclidean RGB distance to the Vision Banana polyline.

    ``project_rgb_to_vision_banana_path`` stores squared distance; reporting
    Euclidean distance keeps thresholds interpretable in normalized RGB units.
    """
    projection = project_rgb_to_vision_banana_path(rgb)
    distance = np.sqrt(np.maximum(projection.distance, 0.0))
    selected = _finite(distance, valid_mask)
    result = {f"distance_{key}": value for key, value in distribution(distance, valid_mask).items()}
    for tolerance in tolerances:
        label = f"fraction_gt_{int(round(tolerance * 255))}rgb_step"
        result[label] = float(np.mean(selected > tolerance)) if selected.size else float("nan")
    return result


def decode_raw_normals(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(rgb, dtype=np.float32)
    if array.size and float(np.nanmax(array)) > 1.0:
        array = array / 255.0
    raw = array * 2.0 - 1.0
    length = np.linalg.norm(raw, axis=-1)
    unit = np.divide(raw, length[..., None], out=np.zeros_like(raw), where=length[..., None] > 1e-8)
    return raw.astype(np.float32), unit.astype(np.float32), length.astype(np.float32)


def normal_representation_diagnostics(
    rgb: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Report how far decoded RGB normals are from the physical unit sphere."""
    _, _, length = decode_raw_normals(rgb)
    requested = (
        np.ones(length.shape, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    valid = requested & np.isfinite(length) & (length > 1e-8)
    error = np.abs(length - 1.0)
    selected = error[valid]
    result = {f"unit_error_{key}": value for key, value in distribution(error, valid).items()}
    requested_count = int(requested.sum())
    invalid_count = int((requested & ~valid).sum())
    result["fraction_invalid_zero_length"] = (
        float(invalid_count / requested_count) if requested_count else float("nan")
    )
    result["requested_pixels"] = requested_count
    result["invalid_zero_length_pixels"] = invalid_count
    for tolerance in (0.01, 0.05, 0.10):
        result[f"fraction_unit_error_gt_{str(tolerance).replace('.', '_')}"] = (
            float(np.mean(selected > tolerance)) if selected.size else float("nan")
        )
    return result


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square-kernel binary erosion implemented with NumPy only."""
    source = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return source.copy()
    height, width = source.shape
    result = source.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.zeros_like(source)
            sy0, sy1 = max(0, -dy), min(height, height - dy)
            sx0, sx1 = max(0, -dx), min(width, width - dx)
            dy0, dy1 = max(0, dy), min(height, height + dy)
            dx0, dx1 = max(0, dx), min(width, width + dx)
            shifted[dy0:dy1, dx0:dx1] = source[sy0:sy1, sx0:sx1]
            result &= shifted
    return result


def mask_coverage_diagnostics(
    mask: np.ndarray,
    latent_shape: tuple[int, int],
    thresholds: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9, 0.999),
) -> dict[str, float | int]:
    """Compare hard and soft latent-mask policies using area coverage."""
    from PIL import Image

    source = np.asarray(mask, dtype=np.float32)
    coverage = np.asarray(
        Image.fromarray(source, mode="F").resize((latent_shape[1], latent_shape[0]), Image.Resampling.BOX),
        dtype=np.float32,
    )
    result: dict[str, float | int] = {
        "image_valid_fraction": float(source.mean()),
        "latent_soft_weight_mean": float(coverage.mean()),
        "latent_cells": int(coverage.size),
        "latent_partial_fraction": float(np.mean((coverage > 0) & (coverage < 1))),
    }
    for threshold in thresholds:
        key = str(threshold).replace(".", "_")
        result[f"latent_hard_{key}_valid_fraction"] = float(np.mean(coverage >= threshold))
    return result


def angular_difference_degrees(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    angle = np.full(mask.shape, np.nan, dtype=np.float32)
    dot = np.clip(np.sum(first * second, axis=-1), -1.0, 1.0)
    angle[mask] = np.degrees(np.arccos(dot[mask]))
    return angle
