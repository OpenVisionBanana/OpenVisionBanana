from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEPTH_ENCODING_GRAYSCALE = "grayscale"
DEPTH_ENCODING_BARRON_SCALAR_GRAYSCALE = "barron_scalar_grayscale"
DEPTH_ENCODING_VISION_BANANA_RGB = "vision_banana_rgb"
DEPTH_ENCODING_RELATIVE_INVERSE_GRAYSCALE = "relative_inverse_grayscale"
DEPTH_ENCODING_RELATIVE_LINEAR_GRAYSCALE = "relative_linear_grayscale"
DEPTH_ENCODINGS = (
    DEPTH_ENCODING_VISION_BANANA_RGB,
    DEPTH_ENCODING_BARRON_SCALAR_GRAYSCALE,
    DEPTH_ENCODING_RELATIVE_INVERSE_GRAYSCALE,
    DEPTH_ENCODING_RELATIVE_LINEAR_GRAYSCALE,
    DEPTH_ENCODING_GRAYSCALE,
)

VISION_BANANA_LAMBDA = -3.0
VISION_BANANA_C = 10.0 / 3.0
VISION_BANANA_EPS = 1e-6

# Figure-5-derived Vision Banana path:
# black -> red -> yellow -> green -> cyan -> blue -> magenta -> white.
VISION_BANANA_RGB_PATH = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class RGBPathProjection:
    t: np.ndarray
    distance: np.ndarray


def valid_depth_mask(depth_meters: np.ndarray, min_depth: float = 0.0, max_depth: float = 0.0) -> np.ndarray:
    depth = np.asarray(depth_meters)
    mask = np.isfinite(depth) & (depth > 0)
    if min_depth > 0:
        mask &= depth >= min_depth
    if max_depth > 0:
        mask &= depth <= max_depth
    return mask


def depth_to_vision_banana_t(depth_meters: np.ndarray, eps: float = VISION_BANANA_EPS) -> np.ndarray:
    """Map metric depth in meters to Vision Banana's bounded scalar t."""
    depth = np.asarray(depth_meters, dtype=np.float32)
    safe_depth = np.where(np.isfinite(depth), np.maximum(depth, 0.0), 0.0)
    t = 1.0 - np.power(1.0 + safe_depth / 10.0, -2.0)
    return np.clip(t, 0.0, 1.0 - eps).astype(np.float32, copy=False)


def vision_banana_t_to_depth(t: np.ndarray, eps: float = VISION_BANANA_EPS) -> np.ndarray:
    """Invert Vision Banana's bounded scalar t back to metric depth in meters."""
    t = np.asarray(t, dtype=np.float32)
    t = np.clip(t, 0.0, 1.0 - eps)
    depth = 10.0 * (np.power(1.0 - t, -0.5) - 1.0)
    return depth.astype(np.float32, copy=False)


def vision_banana_t_to_rgb_float(t: np.ndarray, eps: float = VISION_BANANA_EPS) -> np.ndarray:
    t = np.asarray(t, dtype=np.float32)
    t = np.clip(t, 0.0, 1.0 - eps)

    num_segments = VISION_BANANA_RGB_PATH.shape[0] - 1
    position = t * num_segments
    segment_idx = np.floor(position).astype(np.int32)
    segment_idx = np.clip(segment_idx, 0, num_segments - 1)
    alpha = (position - segment_idx)[..., None]

    start = VISION_BANANA_RGB_PATH[segment_idx]
    end = VISION_BANANA_RGB_PATH[segment_idx + 1]
    rgb = (1.0 - alpha) * start + alpha * end
    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)


def encode_depth_to_vision_banana_rgb(
    depth_meters: np.ndarray,
    valid_mask: np.ndarray | None = None,
    invalid_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    t = depth_to_vision_banana_t(depth_meters)
    rgb = vision_banana_t_to_rgb_float(t)
    rgb_uint8 = np.rint(rgb * 255.0).astype(np.uint8)

    if valid_mask is not None:
        invalid = ~np.asarray(valid_mask, dtype=bool)
        rgb_uint8[invalid] = np.asarray(invalid_color, dtype=np.uint8)
    return rgb_uint8


def encode_depth_to_barron_scalar_grayscale(
    depth_meters: np.ndarray,
    valid_mask: np.ndarray | None = None,
    invalid_value: int = 0,
) -> np.ndarray:
    """Encode the bounded Vision Banana depth scalar directly as grayscale."""
    gray = np.rint(depth_to_vision_banana_t(depth_meters) * 255.0).astype(np.uint8)
    if valid_mask is not None:
        gray = gray.copy()
        gray[~np.asarray(valid_mask, dtype=bool)] = np.uint8(invalid_value)
    return gray


def decode_barron_scalar_grayscale_to_depth(
    grayscale: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Invert a grayscale bounded Vision Banana scalar into metric depth."""
    signal = np.asarray(grayscale, dtype=np.float32)
    if signal.ndim == 3:
        signal = signal[..., :3].mean(axis=-1)
    if signal.size and float(np.nanmax(signal)) > 1.0:
        signal = signal / 255.0
    depth = vision_banana_t_to_depth(signal)
    if valid_mask is not None:
        depth = depth.copy()
        depth[~np.asarray(valid_mask, dtype=bool)] = np.nan
    return depth.astype(np.float32, copy=False)


def encode_depth_to_relative_grayscale(
    depth_meters: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    space: str,
    invalid_value: int = 0,
) -> np.ndarray:
    """Encode one metric field as deterministic per-image relative grayscale.

    ``space='inverse'`` maps nearer valid pixels to brighter values. ``space='linear'``
    maps smaller depth to darker and larger depth to brighter values. Only valid pixels
    determine the per-image extrema; invalid pixels are excluded from the training loss.
    """
    depth = np.asarray(depth_meters, dtype=np.float32)
    valid = valid_depth_mask(depth)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    if not np.any(valid):
        return np.full(depth.shape, invalid_value, dtype=np.uint8)
    if space == "inverse":
        signal = 1.0 / np.maximum(depth, VISION_BANANA_EPS)
    elif space == "linear":
        signal = depth
    else:
        raise ValueError(f"Unsupported relative depth space: {space!r}")
    lower = float(np.min(signal[valid]))
    upper = float(np.max(signal[valid]))
    normalized = np.zeros(depth.shape, dtype=np.float32)
    if upper > lower:
        normalized[valid] = (signal[valid] - lower) / (upper - lower)
    else:
        normalized[valid] = 0.5
    encoded = np.rint(np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
    encoded[~valid] = np.uint8(invalid_value)
    return encoded


def project_rgb_to_vision_banana_path(rgb: np.ndarray, eps: float = VISION_BANANA_EPS) -> RGBPathProjection:
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.size and float(np.nanmax(rgb)) > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)

    shape = rgb.shape[:-1]
    num_segments = VISION_BANANA_RGB_PATH.shape[0] - 1
    best_t = np.zeros(shape, dtype=np.float32)
    best_dist = np.full(shape, np.inf, dtype=np.float32)

    for segment_idx in range(num_segments):
        start = VISION_BANANA_RGB_PATH[segment_idx]
        end = VISION_BANANA_RGB_PATH[segment_idx + 1]
        direction = end - start
        denom = float(np.dot(direction, direction))
        alpha = np.sum((rgb - start) * direction, axis=-1) / denom
        alpha = np.clip(alpha, 0.0, 1.0)
        projection = start + alpha[..., None] * direction
        distance = np.sum((rgb - projection) ** 2, axis=-1)

        update = distance < best_dist
        best_dist[update] = distance[update]
        best_t[update] = (segment_idx + alpha[update]) / num_segments

    return RGBPathProjection(
        t=np.clip(best_t, 0.0, 1.0 - eps).astype(np.float32, copy=False),
        distance=best_dist.astype(np.float32, copy=False),
    )


def decode_vision_banana_rgb_to_depth(
    rgb: np.ndarray,
    valid_mask: np.ndarray | None = None,
    eps: float = VISION_BANANA_EPS,
) -> np.ndarray:
    projection = project_rgb_to_vision_banana_path(rgb, eps=eps)
    depth = vision_banana_t_to_depth(projection.t, eps=eps)
    if valid_mask is not None:
        depth = depth.copy()
        depth[~np.asarray(valid_mask, dtype=bool)] = np.nan
    return depth.astype(np.float32, copy=False)
