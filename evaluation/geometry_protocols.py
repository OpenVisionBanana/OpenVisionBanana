"""Versioned geometry evaluation protocols.

This module intentionally contains no model/runtime imports.  It is shared by
GPU inference and CPU-only rescoring so a saved prediction cannot receive a
different metric merely because it was scored in a different launcher.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

PROTOCOL_VERSION = "geometry-eval-v2"
VISION_BANANA_PROTOCOL_VERSION = "geometry-eval-v3"
NEUTRAL_GEOMETRY_PROTOCOL_VERSION = "geometry-eval-v4"
DEPTH_DISPARITY_AFFINE = "depth-disparity-affine-lstsq-v1"
DEPTH_LINEAR_AFFINE = "depth-linear-affine-lstsq-v1"
DEPTH_DIRECT_METRIC = "depth-direct-metric-v1"
DEPTH_VISION_BANANA_RGB_DIRECT = "depth-vision-banana-rgb-direct-v1"
DEPTH_REFERENCE_DISPARITY_AFFINE = "depth-reference-disparity-affine-v1"
DEPTH_REFERENCE_LINEAR_AFFINE = "depth-reference-linear-affine-v1"
DEPTH_HELDOUT_SELECTED_AFFINE = "depth-heldout-selected-affine-v1"
NORMAL_STRICT = "normal-signed-full-gt-mask-v1"
NORMAL_VALID_ONLY_DIAGNOSTIC = "normal-signed-valid-pred-only-diagnostic-v1"
NORMAL_BIDIRECTIONAL_DIAGNOSTIC = "normal-bidirectional-full-gt-mask-diagnostic-v1"
NORMAL_HELDOUT_CALIBRATED = "normal-heldout-48-axis-calibrated-v1"
NORMAL_HELDOUT_SO3_CALIBRATED = "normal-heldout-so3-calibrated-v1"
NORMAL_TEST5_ORACLE = "normal-test5-48-axis-oracle-v1"

NORMAL_METRIC_NAMES = (
    "mean_angle",
    "median_angle",
    "rmse_angle",
    "angle_11_25",
    "angle_22_5",
    "angle_30",
)


def fit_affine(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Least-squares ``scale * source + shift ~= target`` in float64."""
    x = np.asarray(source, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size == 0:
        raise ValueError("source and target must have the same non-zero size")
    design = np.stack((x, np.ones_like(x)), axis=-1)
    scale, shift = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(scale), float(shift)


def depth_protocol_mask(
    prediction_signal: np.ndarray,
    gt_depth: np.ndarray,
    dataset_mask: np.ndarray | None,
    crop_mask: np.ndarray | None,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    mask = (
        np.isfinite(prediction_signal)
        & np.isfinite(gt_depth)
        & (gt_depth > min_depth)
        & (gt_depth < max_depth)
    )
    if dataset_mask is not None:
        mask &= np.asarray(dataset_mask, dtype=bool)
    if crop_mask is not None:
        mask &= np.asarray(crop_mask, dtype=bool)
    return mask


def disparity_affine_depth(
    relative_signal: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Fit one affine transform from a relative signal to GT disparity.

    There is deliberately no polarity enumeration, mode selection, or spatial
    translation.  A negative fitted scale naturally recovers reversed polarity.
    """
    mask = np.asarray(valid_mask, dtype=bool)
    if int(mask.sum()) < 10:
        raise ValueError("disparity-affine scoring requires at least 10 valid pixels")
    scale, shift = fit_affine(relative_signal[mask], 1.0 / gt_depth[mask])
    aligned_disparity = scale * np.asarray(relative_signal, dtype=np.float64) + shift
    aligned_disparity = np.clip(aligned_disparity, 1e-6, None)
    depth = np.clip(1.0 / aligned_disparity, min_depth, max_depth).astype(np.float32)
    return depth, {
        "protocol": DEPTH_DISPARITY_AFFINE,
        "scale": scale,
        "shift": shift,
        "fit_pixels": int(mask.sum()),
    }


def linear_affine_depth(
    relative_signal: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Fit one affine transform from a relative signal to metric depth."""
    mask = np.asarray(valid_mask, dtype=bool)
    if int(mask.sum()) < 10:
        raise ValueError("linear depth-affine scoring requires at least 10 valid pixels")
    scale, shift = fit_affine(relative_signal[mask], gt_depth[mask])
    depth = np.clip(
        scale * np.asarray(relative_signal, dtype=np.float64) + shift,
        min_depth,
        max_depth,
    ).astype(np.float32)
    return depth, {
        "protocol": DEPTH_LINEAR_AFFINE,
        "scale": scale,
        "shift": shift,
        "fit_pixels": int(mask.sum()),
    }


def reference_luminance(image_rgb: np.ndarray) -> np.ndarray:
    """Decode RGB using the frozen ITU-R BT.709 reference convention."""
    rgb = np.asarray(image_rgb, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Reference luminance expects HxWx3 RGB, got {rgb.shape}")
    if np.issubdtype(np.asarray(image_rgb).dtype, np.integer):
        rgb = rgb / 255.0
    return (rgb @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)).astype(
        np.float32
    )


def select_depth_affine_protocol(
    disparity_abs_rel: Sequence[float],
    linear_abs_rel: Sequence[float],
) -> tuple[str, dict[str, float | int | str]]:
    """Select one affine space on held-out samples using mean per-image AbsRel.

    Ties deterministically prefer disparity, the established relative-depth
    convention.  The returned choice must be frozen before benchmark scoring.
    """
    disparity = np.asarray(disparity_abs_rel, dtype=np.float64)
    linear = np.asarray(linear_abs_rel, dtype=np.float64)
    if disparity.size == 0 or disparity.size != linear.size:
        raise ValueError("held-out depth candidates must have equal non-zero lengths")
    if not np.all(np.isfinite(disparity)) or not np.all(np.isfinite(linear)):
        raise ValueError("held-out depth candidate scores must all be finite")
    disparity_mean = float(np.mean(disparity))
    linear_mean = float(np.mean(linear))
    selected = "disparity" if disparity_mean <= linear_mean else "depth"
    return selected, {
        "selection_metric": "mean_per_image_abs_rel",
        "selected_affine_space": selected,
        "disparity_abs_rel": disparity_mean,
        "depth_abs_rel": linear_mean,
        "sample_count": int(disparity.size),
    }


def depth_configuration_identity(configuration: dict[str, object]) -> tuple[dict[str, object], str]:
    """Return the frozen model/prompt/runtime identity used by depth calibration."""
    keys = (
        "model",
        "model_family",
        "transformer_path",
        "lora_path",
        "prompt_sha256",
        "negative_prompt_sha256",
        "steps",
        "guidance_scale",
        "seed",
        "seed_mode",
        "native_resolution",
        "depth_encoding",
        "evaluator_sha256",
        "runtime_backend_revision",
        "attention_mode",
    )
    normalized = {key: configuration.get(key) for key in keys}
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return normalized, hashlib.sha256(encoded).hexdigest()


def normal_configuration_identity(
    configuration: dict[str, object],
) -> tuple[dict[str, object], str]:
    """Return the frozen model/prompt/runtime identity used by normal calibration."""
    keys = (
        "model",
        "model_family",
        "transformer_path",
        "lora_path",
        "prompt_sha256",
        "negative_prompt_sha256",
        "steps",
        "guidance_scale",
        "seed",
        "seed_mode",
        "native_resolution",
        "prediction_normal_encoding",
        "evaluator_sha256",
        "runtime_backend_revision",
        "attention_mode",
    )
    normalized = {key: configuration.get(key) for key in keys}
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return normalized, hashlib.sha256(encoded).hexdigest()


def load_normal_calibration_record(
    path: str | Path,
    configuration: dict[str, object],
    *,
    expected_protocol: str,
) -> tuple[dict[str, object], "NormalConvention"]:
    """Load a normal convention record and fail closed on identity/subset drift."""
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("protocol") != expected_protocol:
        raise ValueError(
            f"Expected normal calibration protocol {expected_protocol!r} in {target}"
        )
    _, expected_sha256 = normal_configuration_identity(configuration)
    if payload.get("configuration_sha256") != expected_sha256:
        raise ValueError(
            "Normal calibration configuration mismatch: "
            f"record={payload.get('configuration_sha256')} evaluation={expected_sha256}"
        )
    if int(payload.get("sample_count", 0)) <= 0:
        raise ValueError(f"Normal calibration has no samples: {target}")
    sample_ids = payload.get("sample_ids")
    prediction_hashes = payload.get("prediction_hashes")
    if (
        not isinstance(sample_ids, list)
        or not isinstance(prediction_hashes, list)
        or len(sample_ids) != int(payload["sample_count"])
        or len(prediction_hashes) != int(payload["sample_count"])
        or len(sample_ids) != len(set(str(value) for value in sample_ids))
    ):
        raise ValueError(f"Malformed normal calibration sample identity: {target}")
    expected_calibration_id = calibration_identity(
        [str(value) for value in sample_ids],
        [str(value) for value in prediction_hashes],
        protocol=expected_protocol,
        metadata=(
            payload.get("metadata")
            if isinstance(payload.get("metadata"), dict)
            else None
        ),
    )
    if payload.get("calibration_id") != expected_calibration_id:
        raise ValueError(f"Normal calibration identity checksum mismatch: {target}")
    if expected_protocol == NORMAL_HELDOUT_CALIBRATED:
        subset = payload.get("calibration_subset")
        if (
            not isinstance(subset, dict)
            or subset.get("protocol_id") != "heldout_nyu_normals_train32_v1"
        ):
            raise ValueError(
                "Held-out normal calibration does not use the frozen NYU normals "
                f"train-32 manifest: {target}"
            )
        if payload.get("calibration_dataset") != "nyu" or payload.get(
            "calibration_split"
        ) != "train":
            raise ValueError(
                f"Held-out normal calibration must use NYU train data: {target}"
            )
    elif expected_protocol == NORMAL_TEST5_ORACLE and int(
        payload.get("sample_count", 0)
    ) != 5:
        raise ValueError(f"Test-five normal convention record must contain five samples: {target}")
    permutation = tuple(int(value) for value in payload["permutation"])
    signs = tuple(int(value) for value in payload["signs"])
    if len(permutation) != 3 or sorted(permutation) != [0, 1, 2]:
        raise ValueError(f"Invalid normal calibration axis permutation: {target}")
    if len(signs) != 3 or any(value not in (-1, 1) for value in signs):
        raise ValueError(f"Invalid normal calibration axis signs: {target}")
    convention = NormalConvention(permutation, signs)
    return payload, convention


def load_normal_so3_calibration_record(
    path: str | Path,
    configuration: dict[str, object],
) -> tuple[dict[str, object], "NormalRotation"]:
    """Load a held-out proper-rotation record and fail closed on drift."""
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("protocol") != NORMAL_HELDOUT_SO3_CALIBRATED:
        raise ValueError(
            f"Expected normal calibration protocol {NORMAL_HELDOUT_SO3_CALIBRATED!r} "
            f"in {target}"
        )
    _, expected_sha256 = normal_configuration_identity(configuration)
    if payload.get("configuration_sha256") != expected_sha256:
        raise ValueError(
            "Normal SO(3) calibration configuration mismatch: "
            f"record={payload.get('configuration_sha256')} evaluation={expected_sha256}"
        )
    sample_count = int(payload.get("sample_count", 0))
    sample_ids = payload.get("sample_ids")
    prediction_hashes = payload.get("prediction_hashes")
    if (
        sample_count <= 0
        or not isinstance(sample_ids, list)
        or not isinstance(prediction_hashes, list)
        or len(sample_ids) != sample_count
        or len(prediction_hashes) != sample_count
        or len(sample_ids) != len(set(str(value) for value in sample_ids))
    ):
        raise ValueError(f"Malformed normal SO(3) calibration sample identity: {target}")
    expected_calibration_id = calibration_identity(
        [str(value) for value in sample_ids],
        [str(value) for value in prediction_hashes],
        protocol=NORMAL_HELDOUT_SO3_CALIBRATED,
        metadata=(
            payload.get("metadata")
            if isinstance(payload.get("metadata"), dict)
            else None
        ),
    )
    if payload.get("calibration_id") != expected_calibration_id:
        raise ValueError(f"Normal SO(3) calibration identity checksum mismatch: {target}")
    subset = payload.get("calibration_subset")
    if (
        not isinstance(subset, dict)
        or subset.get("protocol_id") != "heldout_nyu_normals_train32_v1"
        or payload.get("calibration_dataset") != "nyu"
        or payload.get("calibration_split") != "train"
    ):
        raise ValueError(
            "Held-out normal SO(3) calibration must use the frozen NYU normals "
            f"train-32 manifest: {target}"
        )
    rotation = NormalRotation.from_array(payload.get("rotation_matrix"))
    metadata = payload.get("metadata") or {}
    if metadata.get("rotation_matrix") != payload.get("rotation_matrix"):
        raise ValueError(f"Normal SO(3) rotation is not checksum-bound: {target}")
    if metadata.get("singular_values") != payload.get("singular_values"):
        raise ValueError(f"Normal SO(3) fit summary is not checksum-bound: {target}")
    if not np.isclose(float(payload.get("determinant", np.nan)), rotation.determinant):
        raise ValueError(f"Normal SO(3) determinant metadata mismatch: {target}")
    if not np.isclose(
        float(payload.get("orthogonality_error", np.nan)),
        rotation.orthogonality_error,
        atol=1e-12,
    ):
        raise ValueError(f"Normal SO(3) orthogonality metadata mismatch: {target}")
    return payload, rotation


def write_depth_affine_calibration_record(
    path: Path,
    *,
    selected_affine_space: str,
    candidate_summary: dict[str, float | int | str],
    configuration: dict[str, object],
    calibration_dataset: str,
    calibration_split: str,
    sample_ids: Sequence[str],
    source_sha256: str,
    calibration_subset: dict[str, object] | None = None,
) -> dict[str, object]:
    """Atomically write an immutable held-out depth-selection record."""
    if selected_affine_space not in {"depth", "disparity"}:
        raise ValueError(f"Unsupported affine space: {selected_affine_space!r}")
    normalized_configuration, configuration_sha256 = depth_configuration_identity(
        configuration
    )
    payload: dict[str, object] = {
        "protocol": DEPTH_HELDOUT_SELECTED_AFFINE,
        "selected_affine_space": selected_affine_space,
        "selection_metric": "mean_per_image_abs_rel",
        "candidate_metrics": dict(candidate_summary),
        "configuration": normalized_configuration,
        "configuration_sha256": configuration_sha256,
        "calibration_dataset": calibration_dataset,
        "calibration_split": calibration_split,
        "sample_ids": list(sample_ids),
        "sample_count": len(sample_ids),
        "source_result_sha256": source_sha256,
        "calibration_subset": calibration_subset,
    }
    canonical = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path = Path(path)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != canonical:
            raise RuntimeError(f"Depth calibration record conflict: {path}")
        return payload
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(canonical)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return payload


def load_depth_affine_calibration_record(
    path: Path, configuration: dict[str, object]
) -> dict[str, object]:
    """Load a held-out record and fail closed on protocol or identity drift."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != DEPTH_HELDOUT_SELECTED_AFFINE:
        raise ValueError(f"Unsupported held-out depth calibration protocol in {path}")
    selected = payload.get("selected_affine_space")
    if selected not in {"depth", "disparity"}:
        raise ValueError(f"Invalid held-out depth affine selection in {path}: {selected!r}")
    _, expected_sha256 = depth_configuration_identity(configuration)
    if payload.get("configuration_sha256") != expected_sha256:
        raise ValueError(
            "Held-out depth calibration configuration mismatch: "
            f"record={payload.get('configuration_sha256')} evaluation={expected_sha256}"
        )
    if int(payload.get("sample_count", 0)) <= 0:
        raise ValueError(f"Held-out depth calibration has no samples: {path}")
    subset = payload.get("calibration_subset")
    if not isinstance(subset, dict) or subset.get("protocol_id") != "heldout_nyu_train32_v1":
        raise ValueError(
            f"Held-out depth calibration does not use the frozen NYU train-32 manifest: {path}"
        )
    return payload


def standard_depth_crop(dataset: str, shape: tuple[int, int]) -> np.ndarray:
    """Return the standard benchmark crop for affine depth evaluation."""
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    normalized = dataset.lower().replace("-", "_")
    if normalized in {"nyu", "nyuv2"}:
        # The established NYUv2 crop expects the native 480x640 evaluation asset.
        if shape != (480, 640):
            raise ValueError(f"NYUv2 standard crop requires 480x640 GT, got {shape}")
        mask[45:471, 41:601] = True
    elif normalized in {"kitti_eigen", "kittieigen"}:
        mask[
            int(0.40810811 * height) : int(0.99189189 * height),
            int(0.03594771 * width) : int(0.96405229 * width),
        ] = True
    elif normalized == "kitti":
        raise ValueError(
            "The KITTI depth-selection-1000 asset is not the Eigen-652 asset; "
            "it cannot use the Eigen evaluation crop."
        )
    else:
        mask[:] = True
    return mask


def depth_range_for_dataset(dataset: str, case: str = "standard") -> tuple[float, float]:
    normalized = dataset.lower().replace("-", "_")
    normalized_case = case.lower()
    if normalized in {"nyu", "nyuv2"}:
        return 1e-3, 10.0
    if normalized in {"ibims", "ibims1"}:
        return 1e-3, 50.0
    if normalized == "diode":
        return (1e-3, 10.0) if normalized_case == "indoor" else (1e-3, 350.0)
    if normalized in {"kitti", "kitti_eigen", "kittieigen", "nuscenes", "vkitti2"}:
        return 1e-3, 80.0
    if normalized == "eth3d":
        return 1e-3, 80.0
    if normalized == "scannet":
        return 1e-3, 10.0
    raise ValueError(f"No frozen affine depth range for dataset={dataset!r}, case={case!r}")


def _normal_stats(angles: np.ndarray) -> dict[str, float]:
    return {
        "mean_angle": float(np.mean(angles)),
        "median_angle": float(np.median(angles)),
        "rmse_angle": float(np.sqrt(np.mean(angles**2))),
        "angle_11_25": float(np.mean(angles < 11.25)),
        "angle_22_5": float(np.mean(angles < 22.5)),
        "angle_30": float(np.mean(angles < 30.0)),
    }


def normal_metrics(
    pred_normals: np.ndarray,
    pred_valid: np.ndarray,
    gt_normals: np.ndarray,
    gt_valid: np.ndarray,
    eval_mask: np.ndarray,
    *,
    bidirectional: bool = False,
    invalid_prediction_angle: float = 90.0,
    valid_prediction_only: bool = False,
    min_gt_pixels: int = 10,
) -> tuple[dict[str, float], np.ndarray]:
    """Score normals over the full GT-valid mask.

    Invalid predictions remain in the denominator and receive 90 degrees.
    ``valid_prediction_only`` exists only to reproduce the legacy diagnostic.
    """
    denominator = np.asarray(eval_mask, dtype=bool) & np.asarray(gt_valid, dtype=bool)
    denominator_pixels = int(denominator.sum())
    if denominator_pixels < min_gt_pixels:
        raise ValueError(
            f"normal scoring requires {min_gt_pixels} GT-valid pixels; got {denominator_pixels}"
        )
    prediction_valid = denominator & np.asarray(pred_valid, dtype=bool)
    score_mask = prediction_valid if valid_prediction_only else denominator
    if valid_prediction_only and int(score_mask.sum()) < min_gt_pixels:
        raise ValueError(
            f"valid-only normal diagnostic requires {min_gt_pixels} pixels; got {int(score_mask.sum())}"
        )

    dot = np.sum(
        np.asarray(pred_normals, dtype=np.float64)
        * np.asarray(gt_normals, dtype=np.float64),
        axis=-1,
    )
    dot = np.clip(dot, -1.0, 1.0)
    if bidirectional:
        dot = np.abs(dot)
    angle_map = np.full(denominator.shape, np.nan, dtype=np.float32)
    angle_map[prediction_valid] = np.degrees(np.arccos(dot[prediction_valid])).astype(np.float32)
    angle_map[denominator & ~prediction_valid] = float(invalid_prediction_angle)
    angles = angle_map[score_mask].astype(np.float64)

    metrics = _normal_stats(angles)
    metrics.update(
        {
            "gt_valid_pixels": denominator_pixels,
            "pred_valid_pixels": int(prediction_valid.sum()),
            "coverage": float(prediction_valid.sum() / denominator_pixels),
            "invalid_prediction_angle": float(invalid_prediction_angle),
        }
    )
    return metrics, angle_map


@dataclass(frozen=True, order=True)
class NormalConvention:
    permutation: tuple[int, int, int]
    signs: tuple[int, int, int]

    @property
    def convention_id(self) -> str:
        axes = "".join(str(value) for value in self.permutation)
        signs = "".join("p" if value > 0 else "m" for value in self.signs)
        return f"perm-{axes}_sign-{signs}"

    def apply(self, normals: np.ndarray) -> np.ndarray:
        return np.asarray(normals)[..., list(self.permutation)] * np.asarray(self.signs)


@dataclass(frozen=True)
class NormalRotation:
    """One frozen proper 3D rotation mapping predicted normals to GT frame."""

    rotation_matrix: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.rotation_matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("normal SO(3) rotation must be a finite 3x3 matrix")
        orthogonality_error = float(np.linalg.norm(matrix.T @ matrix - np.eye(3), ord="fro"))
        determinant = float(np.linalg.det(matrix))
        if orthogonality_error > 1e-6 or abs(determinant - 1.0) > 1e-6:
            raise ValueError(
                "normal SO(3) calibration must be a proper rotation: "
                f"orthogonality_error={orthogonality_error:.3e}, det={determinant:.9f}"
            )

    @classmethod
    def from_array(cls, value: object) -> "NormalRotation":
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"normal SO(3) rotation must be 3x3, got {matrix.shape}")
        return cls(tuple(tuple(float(item) for item in row) for row in matrix))

    @property
    def determinant(self) -> float:
        return float(np.linalg.det(np.asarray(self.rotation_matrix, dtype=np.float64)))

    @property
    def orthogonality_error(self) -> float:
        matrix = np.asarray(self.rotation_matrix, dtype=np.float64)
        return float(np.linalg.norm(matrix.T @ matrix - np.eye(3), ord="fro"))

    def apply(self, normals: np.ndarray) -> np.ndarray:
        """Apply the column-vector rotation to an array storing row vectors."""
        matrix = np.asarray(self.rotation_matrix, dtype=np.float64)
        return np.asarray(normals, dtype=np.float64) @ matrix.T


def all_normal_conventions() -> tuple[NormalConvention, ...]:
    return tuple(
        NormalConvention(permutation=perm, signs=signs)
        for perm in itertools.permutations(range(3))
        for signs in itertools.product((1, -1), repeat=3)
    )


def calibration_identity(
    sample_ids: Sequence[str],
    prediction_hashes: Sequence[str],
    *,
    protocol: str = NORMAL_HELDOUT_CALIBRATED,
    metadata: dict[str, object] | None = None,
) -> str:
    payload = {
        "protocol": protocol,
        "sample_ids": list(sample_ids),
        "prediction_hashes": list(prediction_hashes),
    }
    if metadata:
        payload["metadata"] = metadata
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()


def calibrate_normal_convention(
    pairs: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[NormalConvention, float]:
    """Deterministically select one of all 48 axis/sign conventions.

    Each tuple is ``pred, pred_valid, gt, gt_valid, eval_mask``.  Ties are
    resolved by the stable convention ID, not by shard completion order.
    """
    materialized = list(pairs)
    if not materialized:
        raise ValueError("normal convention calibration requires at least one pair")
    candidates: list[tuple[float, str, NormalConvention]] = []
    for convention in all_normal_conventions():
        per_image = []
        for pred, pred_valid, gt, gt_valid, eval_mask in materialized:
            metrics, _ = normal_metrics(
                convention.apply(pred),
                pred_valid,
                gt,
                gt_valid,
                eval_mask,
                min_gt_pixels=1,
            )
            per_image.append(metrics["mean_angle"])
        candidates.append((float(np.mean(per_image)), convention.convention_id, convention))
    mean_error, _, best = min(candidates)
    return best, mean_error


def calibrate_normal_so3(
    pairs: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[NormalRotation, float, tuple[float, float, float]]:
    """Fit one proper rotation on held-out data with orthogonal Procrustes.

    Each calibration image contributes equally to the 3x3 cross-covariance,
    independent of its resolution or valid-pixel count. Only pixels valid in
    the prediction, GT, and evaluation mask participate in the fit. The fitted
    transform is constrained to SO(3), so reflections and axis scaling are
    impossible.
    """
    materialized = list(pairs)
    if not materialized:
        raise ValueError("normal SO(3) calibration requires at least one pair")

    covariance = np.zeros((3, 3), dtype=np.float64)
    normalized_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for pred, pred_valid, gt, gt_valid, eval_mask in materialized:
        prediction = np.asarray(pred, dtype=np.float64)
        target = np.asarray(gt, dtype=np.float64)
        if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
            raise ValueError("normal SO(3) pairs must contain matching HxWx3 arrays")
        valid = (
            np.asarray(pred_valid, dtype=bool)
            & np.asarray(gt_valid, dtype=bool)
            & np.asarray(eval_mask, dtype=bool)
            & np.all(np.isfinite(prediction), axis=-1)
            & np.all(np.isfinite(target), axis=-1)
        )
        pred_vectors = prediction[valid]
        gt_vectors = target[valid]
        if pred_vectors.size == 0:
            raise ValueError("every normal SO(3) calibration image needs valid overlap")
        pred_norm = np.linalg.norm(pred_vectors, axis=-1, keepdims=True)
        gt_norm = np.linalg.norm(gt_vectors, axis=-1, keepdims=True)
        nonzero = (pred_norm[:, 0] > 1e-12) & (gt_norm[:, 0] > 1e-12)
        pred_vectors = pred_vectors[nonzero] / pred_norm[nonzero]
        gt_vectors = gt_vectors[nonzero] / gt_norm[nonzero]
        if pred_vectors.size == 0:
            raise ValueError("normal SO(3) calibration image has no nonzero overlap")
        covariance += (pred_vectors.T @ gt_vectors) / float(pred_vectors.shape[0])
        normalized_pairs.append((pred_vectors, gt_vectors))

    covariance /= float(len(normalized_pairs))
    left, singular_values, right_t = np.linalg.svd(covariance)
    determinant_fix = np.eye(3, dtype=np.float64)
    determinant_fix[-1, -1] = 1.0 if np.linalg.det(right_t.T @ left.T) >= 0 else -1.0
    rotation_matrix = right_t.T @ determinant_fix @ left.T
    rotation = NormalRotation.from_array(rotation_matrix)

    per_image_mean_angles = []
    for pred_vectors, gt_vectors in normalized_pairs:
        aligned = rotation.apply(pred_vectors)
        dot = np.clip(np.sum(aligned * gt_vectors, axis=-1), -1.0, 1.0)
        per_image_mean_angles.append(float(np.degrees(np.arccos(dot)).mean()))
    return (
        rotation,
        float(np.mean(per_image_mean_angles)),
        tuple(float(value) for value in singular_values),
    )


def convention_oracle_normal_metrics(
    pred_normals: np.ndarray,
    gt_normals: np.ndarray,
    valid_mask: np.ndarray,
    convention: NormalConvention,
    *,
    min_valid_pixels: int = 10,
) -> dict[str, float]:
    """Score normalized predictions after applying a selected axis convention.

    Unlike the primary fixed-frame lane, the convention-oracle diagnostic has
    no prediction-validity mask: every decoded RGB pixel is normalized and
    scored wherever the GT mask is valid.
    """
    valid = np.asarray(valid_mask, dtype=bool)
    if int(valid.sum()) < min_valid_pixels:
        raise ValueError(
            f"normal convention-oracle scoring requires {min_valid_pixels} valid pixels; "
            f"got {int(valid.sum())}"
        )
    transformed = convention.apply(np.asarray(pred_normals, dtype=np.float64))
    gt = np.asarray(gt_normals, dtype=np.float64)
    dot = np.clip(np.sum(transformed * gt, axis=-1), -1.0, 1.0)
    angles = np.degrees(np.arccos(dot))[valid]
    return _normal_stats(angles)


def calibrate_test5_axis_oracle(
    pairs: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[NormalConvention, float]:
    """Select an axis convention on five test samples using per-image means."""
    materialized = list(pairs)
    if not materialized:
        raise ValueError("test-five normal convention selection requires at least one pair")
    scores: dict[NormalConvention, list[float]] = {}
    # Preserve lexicographic permutation/sign order for deterministic ties.
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            convention = NormalConvention(permutation, signs)
            per_image = []
            for pred, gt, valid in materialized:
                metrics = convention_oracle_normal_metrics(
                    pred, gt, valid, convention, min_valid_pixels=1
                )
                per_image.append(metrics["mean_angle"])
            scores[convention] = per_image
    best = min(scores, key=lambda item: float(np.mean(scores[item])))
    return best, float(np.mean(scores[best]))


def write_calibration_record(
    path: str | Path,
    convention: NormalConvention,
    mean_error: float,
    sample_ids: Sequence[str],
    prediction_hashes: Sequence[str],
    *,
    protocol: str = NORMAL_HELDOUT_CALIBRATED,
    metadata: dict[str, object] | None = None,
    configuration: dict[str, object] | None = None,
    calibration_dataset: str | None = None,
    calibration_split: str | None = None,
    calibration_subset: dict[str, object] | None = None,
) -> dict[str, object]:
    """Atomically create or verify an immutable calibration record."""
    if protocol not in {NORMAL_HELDOUT_CALIBRATED, NORMAL_TEST5_ORACLE}:
        raise ValueError(f"unsupported normal calibration protocol: {protocol}")
    target = Path(path)
    normalized_configuration = None
    configuration_sha256 = None
    identity_metadata = dict(metadata or {})
    if configuration is not None:
        normalized_configuration, configuration_sha256 = normal_configuration_identity(
            configuration
        )
        identity_metadata["configuration_sha256"] = configuration_sha256
    record: dict[str, object] = {
        "protocol": protocol,
        "calibration_id": calibration_identity(
            sample_ids,
            prediction_hashes,
            protocol=protocol,
            metadata=identity_metadata,
        ),
        "convention_id": convention.convention_id,
        **asdict(convention),
        "mean_error": float(mean_error),
        "sample_ids": list(sample_ids),
        "prediction_hashes": list(prediction_hashes),
        "sample_count": len(sample_ids),
    }
    if configuration is not None:
        record["configuration"] = normalized_configuration
        record["configuration_sha256"] = configuration_sha256
    if calibration_dataset is not None:
        record["calibration_dataset"] = calibration_dataset
    if calibration_split is not None:
        record["calibration_split"] = calibration_split
    if calibration_subset is not None:
        record["calibration_subset"] = calibration_subset
    if identity_metadata:
        record["metadata"] = identity_metadata
    # JSON has no tuple; normalize before comparing an existing record.
    record["permutation"] = list(convention.permutation)
    record["signs"] = list(convention.signs)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != record:
            raise RuntimeError(f"immutable calibration record conflict: {target}")
        return existing

    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Refuse silent replacement if another shard won the race.
        try:
            os.link(temporary_name, target)
        except FileExistsError:
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing != record:
                raise RuntimeError(f"immutable calibration record conflict: {target}")
            return existing
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return record


def write_so3_calibration_record(
    path: str | Path,
    rotation: NormalRotation,
    mean_error: float,
    singular_values: Sequence[float],
    sample_ids: Sequence[str],
    prediction_hashes: Sequence[str],
    *,
    metadata: dict[str, object] | None = None,
    configuration: dict[str, object],
    calibration_dataset: str,
    calibration_split: str,
    calibration_subset: dict[str, object],
) -> dict[str, object]:
    """Atomically create or verify an immutable held-out SO(3) record."""
    if len(sample_ids) == 0 or len(sample_ids) != len(prediction_hashes):
        raise ValueError("SO(3) calibration requires matching non-empty sample identities")
    values = tuple(float(value) for value in singular_values)
    if len(values) != 3 or not np.all(np.isfinite(values)):
        raise ValueError("SO(3) calibration requires three finite singular values")
    normalized_configuration, configuration_sha256 = normal_configuration_identity(
        configuration
    )
    identity_metadata = dict(metadata or {})
    identity_metadata["configuration_sha256"] = configuration_sha256
    identity_metadata["fit_method"] = "equal-image-weighted-orthogonal-procrustes"
    identity_metadata["rotation_matrix"] = [
        list(row) for row in rotation.rotation_matrix
    ]
    identity_metadata["singular_values"] = list(values)
    record: dict[str, object] = {
        "protocol": NORMAL_HELDOUT_SO3_CALIBRATED,
        "calibration_id": calibration_identity(
            sample_ids,
            prediction_hashes,
            protocol=NORMAL_HELDOUT_SO3_CALIBRATED,
            metadata=identity_metadata,
        ),
        "rotation_matrix": [list(row) for row in rotation.rotation_matrix],
        "determinant": rotation.determinant,
        "orthogonality_error": rotation.orthogonality_error,
        "fit_method": "equal-image-weighted-orthogonal-procrustes",
        "fit_objective": "maximize mean per-image cosine similarity",
        "mean_error": float(mean_error),
        "singular_values": list(values),
        "sample_ids": list(sample_ids),
        "prediction_hashes": list(prediction_hashes),
        "sample_count": len(sample_ids),
        "configuration": normalized_configuration,
        "configuration_sha256": configuration_sha256,
        "calibration_dataset": calibration_dataset,
        "calibration_split": calibration_split,
        "calibration_subset": calibration_subset,
        "metadata": identity_metadata,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != record:
            raise RuntimeError(f"immutable SO(3) calibration record conflict: {target}")
        return existing

    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, target)
        except FileExistsError:
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing != record:
                raise RuntimeError(f"immutable SO(3) calibration record conflict: {target}")
            return existing
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return record
