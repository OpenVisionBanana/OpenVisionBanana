"""Canonical geometry contracts used by training-data preparation.

Canonical surface-normal coordinates are camera-relative:

* +x points right in the image;
* +y points up in the image;
* +z points from the surface toward the camera.

Canonical depth is projective camera-z in metres.  Physical scalar/vector
fields are resized before visualization encoding and masks are resized with
nearest-neighbour sampling.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from PIL import Image

from common.surface_normal_encoding import (
    LEGACY_DIRECT_NORMAL_ENCODING,
    VISION_BANANA_NORMAL_ENCODING,
)

GEOMETRY_DATA_VERSION = "geometry_canonical_v2"
VISION_BANANA_GEOMETRY_DATA_VERSION = "geometry_vision_banana_v3"
CAMERA_FACING_GEOMETRY_DATA_VERSION = "geometry_camera_facing_v4"
CANONICAL_NORMAL_CONVENTION = "camera_x_right_y_up_z_toward_camera"
CANONICAL_DEPTH_QUANTITY = "projective_camera_z_meters"
_RESAMPLING = getattr(Image, "Resampling", Image)


def geometry_data_version_for_normal_encoding(encoding: str) -> str:
    """Return the immutable dataset-contract version for an RGB normal encoding."""
    if encoding == VISION_BANANA_NORMAL_ENCODING:
        return VISION_BANANA_GEOMETRY_DATA_VERSION
    if encoding == LEGACY_DIRECT_NORMAL_ENCODING:
        return GEOMETRY_DATA_VERSION
    raise ValueError(f"Unsupported surface-normal encoding: {encoding!r}")


@dataclass(frozen=True)
class NormalTransform:
    permutation: tuple[int, int, int]
    signs: tuple[int, int, int]
    native_convention: str
    evidence: str
    geometry_version: str = GEOMETRY_DATA_VERSION

    def apply(self, normals: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
        transformed = np.asarray(normals, dtype=np.float32)[
            ..., list(self.permutation)
        ] * np.asarray(self.signs, dtype=np.float32)
        norm = np.linalg.norm(transformed, axis=-1, keepdims=True)
        normalized = np.divide(
            transformed,
            norm,
            out=np.zeros_like(transformed),
            where=np.isfinite(norm) & (norm > 1e-8),
        )
        if valid is not None:
            normalized[~np.asarray(valid, dtype=bool)] = 0
        return normalized

    def metadata(self) -> dict[str, object]:
        return {
            "geometry_version": self.geometry_version,
            "native_normal_convention": self.native_convention,
            "canonical_normal_convention": CANONICAL_NORMAL_CONVENTION,
            "normal_axis_permutation": list(self.permutation),
            "normal_axis_signs": list(self.signs),
            "normal_transform_evidence": self.evidence,
        }


IDENTITY_CANONICAL = NormalTransform(
    (0, 1, 2),
    (1, 1, 1),
    CANONICAL_NORMAL_CONVENTION,
    "already canonical",
)

# Point clouds built as x=(u-cx)z/fx, y=(v-cy)z/fy, z=depth have +y down and
# +z away.  The legacy preparers then force the derived normal itself to have
# positive native z (away from the camera).  Canonical labels must first reverse
# that surface orientation, then change from the native basis to x-right,
# y-up, z-toward-camera.  The composed component transform is (-x, +y, +z).
IMAGE_Y_DOWN_Z_AWAY_TO_CANONICAL = NormalTransform(
    (0, 1, 2),
    (-1, 1, 1),
    "camera_x_right_y_down_z_away_from_camera_normal_forced_z_positive",
    (
        "Analytic surface-orientation reversal composed with camera-basis "
        "conversion; 48-transform tilted-plane oracle selected (-x,+y,+z) at "
        "0.00025 degree mean over 49152 pixels with a 24.03 degree margin. "
        "Evidence: open_visionbanana_experiments/geometry_pipeline_repair/"
        "audit/depth_derived_normal_transform.json"
    ),
)

# nuScenes PCA normals are first expressed in the standard camera basis
# (+x right, +y down, +z away) and then explicitly oriented toward the camera.
# Once their orientation is fixed, only the basis conversion remains.
CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL = NormalTransform(
    (0, 1, 2),
    (1, -1, -1),
    "camera_x_right_y_down_z_away_from_camera_normal_oriented_toward_camera",
    (
        "Analytic camera-basis conversion after the preparer orients each PCA "
        "normal against its camera-space point: (+x,+y,+z)_native maps to "
        "(+x,-y,-z)_canonical."
    ),
    CAMERA_FACING_GEOMETRY_DATA_VERSION,
)


def orient_normals_toward_camera(
    normals: np.ndarray,
    camera_points: np.ndarray,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Orient native camera-space normals against camera-to-surface rays.

    Both arrays use the conventional camera basis (+x right, +y down, +z
    away). A camera-facing normal has a non-positive dot product with the
    camera-to-surface point. Unlike forcing a component sign, this remains
    well-defined for road/floor planes whose z component is near zero.
    """
    oriented = np.asarray(normals, dtype=np.float32).copy()
    points = np.asarray(camera_points, dtype=np.float32)
    if oriented.shape != points.shape or oriented.shape[-1] != 3:
        raise ValueError(
            f"Expected matching (...,3) normals and points, got "
            f"{oriented.shape} and {points.shape}"
        )
    finite = np.isfinite(oriented).all(axis=-1) & np.isfinite(points).all(axis=-1)
    if valid is not None:
        finite &= np.asarray(valid, dtype=bool)
    flip = finite & (np.sum(oriented * points, axis=-1) > 0)
    oriented[flip] *= -1.0
    if valid is not None:
        oriented[~np.asarray(valid, dtype=bool)] = 0.0
    return oriented

HYPERSIM_NATIVE_TO_CANONICAL = NormalTransform(
    (0, 1, 2),
    (1, 1, 1),
    "hypersim_camera_x_right_y_up_z_toward_camera",
    (
        "Five-pair native-normal versus official-ray depth-derived 48-transform "
        "audit: identity was best (8.43 degree mean over 3914260 jointly valid "
        "pixels; 41.45 degree margin). Evidence: "
        "open_visionbanana_experiments/geometry_pipeline_repair/audit/"
        "hypersim_normal_transform.json"
    ),
)

MATTERPORT_NATIVE_TO_CANONICAL = NormalTransform(
    (0, 1, 2),
    (1, 1, 1),
    "matterport_undistorted_camera_x_right_y_up_z_toward_camera",
    (
        "Paired native-normal versus calibrated depth-derived audit on "
        "17DRP5sb8fy/c429b363fd3145fe8a7631bbe9644066_d0_0: identity was the "
        "best of 48 transforms (7.49 degree mean, 6.19 degree median over "
        "300090 jointly valid pixels)."
    ),
)

TASKONOMY_NATIVE_TO_CANONICAL = NormalTransform(
    (0, 1, 2),
    (1, -1, -1),
    "taskonomy_camera_x_right_y_down_z_away_normal",
    (
        "Paired Taskonomy native-normal versus depth_zbuffer-derived canonical "
        "audit over 12 buildings, 24 frames, four assumed pinhole FOVs, and "
        "23,208,308 jointly valid pixels selected (+x,-y,-z) for all 96 pairs "
        "(8.31 degree mean; 40.59 degree margin; pair consensus 1.0). "
        "Evidence SHA-256 7ed243c196c2ec5614f9f8d9d9f9fc7e67a12c92eabb30eb001bdf778801120e: "
        "open_visionbanana_experiments/geometry_pipeline_repair/audit/taskonomy/"
        "taskonomy_normal_transform.json"
    ),
)


def taskonomy_normal_transform() -> NormalTransform:
    return TASKONOMY_NATIVE_TO_CANONICAL


def all_axis_sign_transforms() -> tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]:
    return tuple(
        (permutation, signs)
        for permutation in itertools.permutations(range(3))
        for signs in itertools.product((1, -1), repeat=3)
    )


def range_to_projective_z(
    range_m: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Convert Euclidean camera range to projective z using pinhole rays."""
    if fx <= 0 or fy <= 0:
        raise ValueError("fx and fy must be positive")
    height, width = range_m.shape
    u = np.arange(width, dtype=np.float32)[None, :]
    v = np.arange(height, dtype=np.float32)[:, None]
    ray_norm = np.sqrt(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2 + 1.0)
    result = np.asarray(range_m, dtype=np.float32) / ray_norm
    result[~np.isfinite(range_m)] = np.nan
    return result.astype(np.float32, copy=False)


def range_to_projective_z_from_ray_matrix(
    range_m: np.ndarray,
    camera_from_uv: np.ndarray,
) -> np.ndarray:
    """Hypersim range→z conversion using its published ``M_cam_from_uv``.

    Hypersim pixel centres span u,v in [-1,1], with v reversed.  Projective z
    is the magnitude of the camera-ray z component times Euclidean range.
    """
    height, width = range_m.shape
    half_du = 1.0 / width
    half_dv = 1.0 / height
    u = np.linspace(-1.0 + half_du, 1.0 - half_du, width, dtype=np.float64)
    v = np.linspace(-1.0 + half_dv, 1.0 - half_dv, height, dtype=np.float64)[::-1]
    uu, vv = np.meshgrid(u, v)
    uv = np.stack((uu, vv, np.ones_like(uu)), axis=-1)
    rays = uv @ np.asarray(camera_from_uv, dtype=np.float64).T
    ray_norm = np.linalg.norm(rays, axis=-1)
    z_fraction = np.abs(rays[..., 2]) / np.clip(ray_norm, 1e-12, None)
    result = np.asarray(range_m, dtype=np.float64) * z_fraction
    result[~np.isfinite(range_m)] = np.nan
    return result.astype(np.float32)


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    return np.asarray(image.resize(size, resample=_RESAMPLING.NEAREST)) > 0


def resize_scalar_field(
    values: np.ndarray,
    valid: np.ndarray,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Resize a physical scalar without blending invalid values into it."""
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if values.shape == (size[1], size[0]):
        return values.copy(), valid.copy()
    numerator = np.where(valid, values, 0).astype(np.float32)
    weights = valid.astype(np.float32)
    numerator_image = Image.fromarray(numerator, mode="F").resize(
        size, resample=_RESAMPLING.BILINEAR
    )
    weight_image = Image.fromarray(weights, mode="F").resize(
        size, resample=_RESAMPLING.BILINEAR
    )
    numerator_resized = np.asarray(numerator_image, dtype=np.float32)
    weights_resized = np.asarray(weight_image, dtype=np.float32)
    hard_valid = resize_mask(valid, size)
    output = np.full((size[1], size[0]), np.nan, dtype=np.float32)
    usable = hard_valid & (weights_resized > 1e-6)
    output[usable] = numerator_resized[usable] / weights_resized[usable]
    return output, usable


def resize_normal_field(
    normals: np.ndarray,
    valid: np.ndarray,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Resize XYZ components, renormalize, and retain a hard validity mask."""
    channels = [
        resize_scalar_field(normals[..., channel], valid, size)[0]
        for channel in range(3)
    ]
    resized = np.stack(channels, axis=-1)
    hard_valid = resize_mask(valid, size)
    norm = np.linalg.norm(resized, axis=-1, keepdims=True)
    hard_valid &= np.isfinite(norm[..., 0]) & (norm[..., 0] > 1e-8)
    output = np.divide(
        resized,
        norm,
        out=np.zeros_like(resized),
        where=norm > 1e-8,
    ).astype(np.float32)
    output[~hard_valid] = 0
    return output, hard_valid


def horizontal_flip_normals(normals: np.ndarray) -> np.ndarray:
    result = np.asarray(normals)[..., ::-1, :].copy()
    result[..., 0] *= -1
    return result


def vertical_flip_normals(normals: np.ndarray) -> np.ndarray:
    result = np.asarray(normals)[..., ::-1, :, :].copy()
    result[..., 1] *= -1
    return result
