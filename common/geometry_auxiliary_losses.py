"""Differentiable geometry losses for decoded FLUX clean-image predictions."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from common.surface_normal_encoding import (
    VISION_BANANA_NORMAL_ENCODING,
    decode_surface_normals_tensor,
)

VISION_BANANA_RGB_PATH = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 1.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
)


def recover_clean_latents(
    noisy_latents: torch.Tensor,
    predicted_velocity: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    """Recover x0 from z_t=(1-sigma)x0+sigma*z1 and v=z1-x0."""
    return noisy_latents - sigmas * predicted_velocity


def _path_projection(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return differentiable nearest-path scalar t and squared RGB distance."""
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"Expected RGB tensor [B,3,H,W], received {tuple(rgb.shape)}")
    points = rgb.float().permute(0, 2, 3, 1)
    vertices = torch.as_tensor(
        VISION_BANANA_RGB_PATH,
        dtype=points.dtype,
        device=points.device,
    )
    starts = vertices[:-1]
    directions = vertices[1:] - starts
    relative = points.unsqueeze(-2) - starts
    alpha = (relative * directions).sum(-1) / directions.square().sum(-1)
    alpha = alpha.clamp(0.0, 1.0)
    projections = starts + alpha.unsqueeze(-1) * directions
    squared_distance = (points.unsqueeze(-2) - projections).square().sum(-1)
    best_distance, best_segment = squared_distance.min(dim=-1)
    best_alpha = alpha.gather(-1, best_segment.unsqueeze(-1)).squeeze(-1)
    t = (best_segment.to(points.dtype) + best_alpha) / float(len(VISION_BANANA_RGB_PATH) - 1)
    return t.clamp(0.0, 1.0 - 1e-6), best_distance


def _weighted_mean(values: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return values.mean()
    mask = valid_mask.float()
    if mask.ndim == 4:
        mask = mask[:, 0]
    if mask.shape[-2:] != values.shape[-2:]:
        mask = F.interpolate(mask.unsqueeze(1), size=values.shape[-2:], mode="area")[:, 0]
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def palette_manifold_loss(
    predicted_rgb: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Squared Euclidean distance to the nearest Vision Banana path segment."""
    _, squared_distance = _path_projection(predicted_rgb)
    return _weighted_mean(squared_distance, valid_mask)


def unit_normal_loss(
    predicted_rgb: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    normal_encoding: str = VISION_BANANA_NORMAL_ENCODING,
) -> torch.Tensor:
    """Smooth-L1 penalty for decoded normal-vector length deviating from one."""
    predicted = decode_surface_normals_tensor(predicted_rgb, normal_encoding)
    error = (torch.linalg.vector_norm(predicted, dim=1) - 1.0).abs()
    penalty = F.smooth_l1_loss(error, torch.zeros_like(error), beta=0.05, reduction="none")
    return _weighted_mean(penalty, valid_mask)


def angular_normal_loss(
    predicted_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    normal_encoding: str = VISION_BANANA_NORMAL_ENCODING,
) -> torch.Tensor:
    """Cosine-direction loss between decoded predicted and target normals."""
    predicted = F.normalize(
        decode_surface_normals_tensor(predicted_rgb, normal_encoding),
        dim=1,
        eps=1e-6,
    )
    target = F.normalize(
        decode_surface_normals_tensor(target_rgb, normal_encoding),
        dim=1,
        eps=1e-6,
    )
    penalty = 1.0 - (predicted * target).sum(dim=1).clamp(-1.0, 1.0)
    return _weighted_mean(penalty, valid_mask)


def _depth_from_t(t: torch.Tensor, max_depth: float) -> torch.Tensor:
    depth = 10.0 * ((1.0 - t.clamp(0.0, 1.0 - 1e-6)).rsqrt() - 1.0)
    return depth.clamp(0.0, float(max_depth))


def robust_metric_depth_loss(
    predicted_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    max_depth: float = 80.0,
) -> torch.Tensor:
    """Robust log-metric loss after differentiable Vision Banana projection."""
    predicted_t, _ = _path_projection(predicted_rgb)
    target_t, _ = _path_projection(target_rgb)
    normalizer = math.log1p(float(max_depth))
    predicted_log = torch.log1p(_depth_from_t(predicted_t, max_depth)) / normalizer
    target_log = torch.log1p(_depth_from_t(target_t, max_depth)) / normalizer
    penalty = F.smooth_l1_loss(predicted_log, target_log, beta=0.02, reduction="none")
    return _weighted_mean(penalty, valid_mask)
