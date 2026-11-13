"""Pixel-Perfect Depth / Marigold V2 affine-depth scoring.

This is an independent evaluation lane. It intentionally mirrors the public
Marigold V2 evaluator: per-image RANSAC scale-and-shift fitting in log-depth,
PPD validity bounds, DIODE edge filtering, and ETH3D scoring at 2048x1360.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


PROTOCOL_ID = "marigold_v2_ppd_ransac_log_v1"
SUPPORTED_DATASETS = ("nyu", "kitti_eigen", "eth3d", "scannet", "diode")


def resolve_ppd_alignment(depth_encoding: str, requested: str = "auto") -> str:
    """Select by declared encoding, never by which benchmark score is higher."""
    if requested not in {"auto", "disparity_affine", "log_depth_ransac"}:
        raise ValueError(f"Unknown PPD alignment: {requested}")
    if requested == "disparity_affine":
        if depth_encoding != "relative_inverse_grayscale":
            raise ValueError("disparity_affine requires relative_inverse_grayscale")
        return requested
    if requested == "log_depth_ransac":
        return requested  # Explicit legacy diagnostic for relative grayscale.
    if depth_encoding == "relative_inverse_grayscale":
        return "disparity_affine"
    if depth_encoding in {"vision_banana_rgb", "barron_scalar_grayscale"}:
        return "log_depth_ransac"
    raise ValueError(
        f"No encoding-aware PPD alignment for {depth_encoding!r}; "
        "log_depth_ransac must be explicitly selected as a diagnostic."
    )


def relative_grayscale_to_log_depth_signal(
    grayscale: np.ndarray, *, inverse: bool
) -> np.ndarray:
    """Orient a relative grayscale signal so larger values mean farther depth.

    This is an orientation-only legacy diagnostic, NOT a conversion to log
    depth. Inverse/linear grayscale is not generally affine in log depth.
    PPD fits a positive scale in log-depth space. Inverse-depth targets encode
    nearer pixels as brighter, so passing raw grayscale to the fitter reverses
    the geometry and triggers its non-positive-scale fallback.
    """
    signal = np.asarray(grayscale, dtype=np.float32)
    return -signal if inverse else signal


def canonical_dataset_name(dataset: str) -> str:
    normalized = dataset.lower().replace("-", "_")
    aliases = {
        "nyuv2": "nyu",
        "kitti": "kitti_eigen",
        "kittieigen": "kitti_eigen",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_DATASETS:
        raise ValueError(
            f"{PROTOCOL_ID} supports only: {', '.join(SUPPORTED_DATASETS)}; "
            f"got {dataset!r}"
        )
    return normalized


def depth_bounds(dataset: str) -> tuple[float, float | None]:
    dataset = canonical_dataset_name(dataset)
    return {
        "nyu": (0.1, 10.0),
        "kitti_eigen": (0.1, 80.0),
        "eth3d": (0.01, None),
        "scannet": (0.01, 10.0),
        "diode": (0.6, 350.0),
    }[dataset]


def _resize_nearest(values: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = torch.from_numpy(values.astype(np.float32, copy=False))[None, None]
    resized = F.interpolate(tensor, size=(height, width), mode="nearest")
    return resized[0, 0].cpu().numpy().astype(np.float32, copy=False)


def prepare_protocol_arrays(
    dataset: str,
    gt_depth: np.ndarray,
    valid_mask: np.ndarray,
    pred_log_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the exact PPD dataset resizing and validity-mask policy."""
    dataset = canonical_dataset_name(dataset)
    gt = np.asarray(gt_depth, dtype=np.float32).squeeze()
    mask = np.asarray(valid_mask, dtype=bool).squeeze()
    pred_log = np.asarray(pred_log_depth, dtype=np.float32).squeeze()

    if gt.ndim != 2 or mask.shape != gt.shape or pred_log.ndim != 2:
        raise ValueError(
            f"Expected 2D GT/mask/prediction; got {gt.shape}, {mask.shape}, "
            f"{pred_log.shape}"
        )

    if dataset == "eth3d":
        gt = _resize_nearest(gt, 1360, 2048)
        mask = _resize_nearest(mask.astype(np.float32), 1360, 2048) > 0.5
        pred_log = _resize_nearest(pred_log, 1360, 2048)
    elif pred_log.shape != gt.shape:
        pred_log = _resize_nearest(pred_log, gt.shape[0], gt.shape[1])

    depth_min, depth_max = depth_bounds(dataset)
    mask = mask & np.isfinite(gt) & (gt > depth_min)
    if depth_max is not None and math.isfinite(depth_max):
        mask &= gt < depth_max

    if dataset == "diode":
        gradient = np.abs(ndimage.sobel(gt, 0)) + np.abs(ndimage.sobel(gt, 1))
        mask &= gradient <= 0.3

    mask &= np.isfinite(pred_log)
    if not np.any(mask):
        raise RuntimeError("PPD protocol has no valid pixels for this sample")
    return gt, mask, pred_log


def align_log_depth_ransac(
    pred_log_depth: np.ndarray,
    gt_log_depth: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Mirror Marigold V2's seeded sklearn RANSAC alignment exactly."""
    try:
        from sklearn.linear_model import RANSACRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures
    except ImportError as exc:
        raise ImportError(
            "The Marigold V2 / PPD lane requires scikit-learn==1.7.2. "
            "Install requirements/geometry.txt."
        ) from exc

    mask_gt = gt_log_depth[valid_mask].astype(np.float32)
    mask_pred = pred_log_depth[valid_mask].astype(np.float32)
    model = make_pipeline(
        PolynomialFeatures(degree=1, include_bias=False),
        RANSACRegressor(max_trials=1000, random_state=42),
    )
    try:
        model.fit(mask_pred[:, None], mask_gt[:, None])
        estimator = model.named_steps["ransacregressor"].estimator_
        scale = estimator.coef_.item()
        shift = estimator.intercept_.item()
    except Exception:
        scale, shift = 1.0, 0.0

    if scale > 0:
        aligned = scale * pred_log_depth + shift
    else:
        pred_mean = float(np.mean(mask_pred))
        gt_mean = float(np.mean(mask_gt))
        scale = gt_mean / pred_mean if abs(pred_mean) > 1e-8 else 1.0
        shift = 0.0
        aligned = scale * pred_log_depth
    return aligned, float(scale), float(shift)


def score_ppd_ransac_log(
    dataset: str,
    pred_log_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[dict[str, float], dict[str, float | int | str], np.ndarray]:
    """Return Marigold-compatible per-image depth metrics and aligned depth."""
    gt, mask, pred_log = prepare_protocol_arrays(
        dataset, gt_depth, valid_mask, pred_log_depth
    )
    gt_log = np.log(np.clip(gt, 1e-6, None))
    aligned_log, scale, shift = align_log_depth_ransac(pred_log, gt_log, mask)
    pred = np.exp(aligned_log)
    pred = np.clip(pred, a_min=1e-3, a_max=float(np.max(gt[mask])))
    pred = np.clip(pred, a_min=1e-6, a_max=None)

    pred_valid = pred[mask]
    gt_valid = gt[mask]
    ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    metrics = {
        "abs_rel": float(np.mean(np.abs(pred_valid - gt_valid) / gt_valid)),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
    }
    metadata: dict[str, float | int | str] = {
        "protocol": PROTOCOL_ID,
        "ransac_scale": scale,
        "ransac_shift": shift,
        "valid_pixels": int(mask.sum()),
    }
    return metrics, metadata, pred.astype(np.float32, copy=False)


def score_ppd_split_disparity_affine(
    dataset: str,
    pred_inverse_signal: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[dict[str, float], dict[str, float | int | str]]:
    """Score a fixed inverse-depth signal with disparity-affine alignment.

    This encoding-compatible lane uses the *same PPD split and masks* but
    differs from native PPD log-depth alignment. It must not be presented as
    an exact reproduction of another paper's full protocol.
    """
    gt, mask, signal = prepare_protocol_arrays(
        dataset, gt_depth, valid_mask, pred_inverse_signal
    )
    x = signal[mask].astype(np.float64)
    y = (1.0 / gt[mask]).astype(np.float64)
    scale, shift = np.linalg.lstsq(
        np.stack((x, np.ones_like(x)), axis=-1), y, rcond=None
    )[0]
    disparity = np.clip(scale * signal.astype(np.float64) + shift, 1e-6, None)
    depth_min, depth_max = depth_bounds(dataset)
    upper = float(depth_max) if depth_max is not None else float(np.max(gt[mask]))
    pred = np.clip(1.0 / disparity, depth_min, upper)
    pred_valid = pred[mask]
    gt_valid = gt[mask]
    ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    metrics = {
        "abs_rel": float(np.mean(np.abs(pred_valid - gt_valid) / gt_valid)),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
    }
    metadata: dict[str, float | int | str] = {
        "protocol": "ppd_split_disparity_affine_v1",
        "scale": float(scale),
        "shift": float(shift),
        "valid_pixels": int(mask.sum()),
    }
    return metrics, metadata
