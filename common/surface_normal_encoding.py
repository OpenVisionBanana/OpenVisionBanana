"""Versioned RGB encodings for canonical camera-space surface normals.

The physical convention is always ``(+x right, +y up, +z toward camera)``.
Only the red-channel storage convention differs:

``vision_banana_x_left_red``
    Vision Banana's published encoding: ``R=(1-x)/2``, ``G=(1+y)/2``,
    ``B=(1+z)/2``. A left-facing vector is red.

``legacy_direct_xyz``
    Historical direct-XYZ storage:
    ``RGB=((x,y,z)+1)/2``. Kept to decode older assets.
"""

from __future__ import annotations

from typing import Final

import numpy as np


VISION_BANANA_NORMAL_ENCODING: Final = "vision_banana_x_left_red"
LEGACY_DIRECT_NORMAL_ENCODING: Final = "legacy_direct_xyz"
SUPPORTED_NORMAL_ENCODINGS: Final = (
    VISION_BANANA_NORMAL_ENCODING,
    LEGACY_DIRECT_NORMAL_ENCODING,
)
VISION_BANANA_NORMAL_ENCODING_FORMULA: Final = (
    "R=clip((1-x)/2,0,1)*255; "
    "G=clip((1+y)/2,0,1)*255; "
    "B=clip((1+z)/2,0,1)*255"
)
LEGACY_DIRECT_NORMAL_ENCODING_FORMULA: Final = (
    "RGB=clip((normal_xyz+1)/2,0,1)*255"
)


def validate_normal_encoding(encoding: str) -> str:
    if encoding not in SUPPORTED_NORMAL_ENCODINGS:
        raise ValueError(
            f"Unsupported normal encoding {encoding!r}; expected one of "
            f"{', '.join(SUPPORTED_NORMAL_ENCODINGS)}"
        )
    return encoding


def normal_encoding_formula(encoding: str) -> str:
    validate_normal_encoding(encoding)
    if encoding == VISION_BANANA_NORMAL_ENCODING:
        return VISION_BANANA_NORMAL_ENCODING_FORMULA
    return LEGACY_DIRECT_NORMAL_ENCODING_FORMULA


def physical_normals_to_storage_components(
    normals_xyz: np.ndarray,
    encoding: str,
) -> np.ndarray:
    """Map physical canonical XYZ to the signed components stored in RGB."""
    validate_normal_encoding(encoding)
    components = np.asarray(normals_xyz)
    if components.shape[-1] != 3:
        raise ValueError(f"Expected (...,3) normals, received {components.shape}")
    if encoding == VISION_BANANA_NORMAL_ENCODING:
        components = components.copy()
        components[..., 0] *= -1
    return components


def storage_components_to_physical_normals(
    components: np.ndarray,
    encoding: str,
) -> np.ndarray:
    """Map signed RGB storage components back to physical canonical XYZ."""
    # Both supported transforms are self-inverse.
    return physical_normals_to_storage_components(components, encoding)


def transform_surface_normals(
    normals_xyz: np.ndarray,
    permutation: tuple[int, int, int] = (0, 1, 2),
    signs: tuple[int, int, int] = (1, 1, 1),
) -> np.ndarray:
    """Apply a declared axis permutation/sign transform and renormalize."""
    if sorted(permutation) != [0, 1, 2]:
        raise ValueError(f"Invalid normal-axis permutation: {permutation!r}")
    if len(signs) != 3 or any(sign not in (-1, 1) for sign in signs):
        raise ValueError(f"Invalid normal-axis signs: {signs!r}")
    normals = np.asarray(normals_xyz, dtype=np.float32)[..., list(permutation)]
    normals = normals * np.asarray(signs, dtype=np.float32)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    return np.divide(
        normals,
        norm,
        out=np.zeros_like(normals),
        where=np.isfinite(norm) & (norm > 1e-8),
    )


def encode_surface_normals_rgb(
    normals_xyz: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    encoding: str = VISION_BANANA_NORMAL_ENCODING,
    invalid_rgb: tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """Encode canonical physical unit normals as uint8 RGB."""
    normals = np.asarray(normals_xyz, dtype=np.float32)
    if normals.ndim < 2 or normals.shape[-1] != 3:
        raise ValueError(f"Expected (...,3) normals, received {normals.shape}")
    stored = physical_normals_to_storage_components(normals, encoding)
    encoded = np.rint((np.clip(stored, -1.0, 1.0) * 0.5 + 0.5) * 255.0)
    rgb = np.clip(encoded, 0, 255).astype(np.uint8)
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != normals.shape[:-1]:
            raise ValueError(
                f"Validity shape {mask.shape} does not match normals "
                f"{normals.shape[:-1]}"
            )
        rgb = rgb.copy()
        rgb[~mask] = np.asarray(invalid_rgb, dtype=np.uint8)
    return rgb


def decode_surface_normals_rgb(
    rgb: np.ndarray,
    *,
    encoding: str = VISION_BANANA_NORMAL_ENCODING,
    min_norm: float = 0.05,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode RGB to canonical physical unit normals and a validity mask."""
    array = np.asarray(rgb, dtype=dtype)
    if array.ndim < 2 or array.shape[-1] != 3:
        raise ValueError(f"Expected (...,3) RGB, received {array.shape}")
    stored = array / 255.0 * 2.0 - 1.0
    normals = storage_components_to_physical_normals(stored, encoding)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    valid = np.isfinite(norm[..., 0]) & (norm[..., 0] >= float(min_norm))
    unit = np.divide(
        normals,
        norm,
        out=np.zeros_like(normals),
        where=norm >= float(min_norm),
    )
    unit[~np.isfinite(unit)] = 0
    return unit.astype(dtype, copy=False), valid


def decode_surface_normals_tensor(rgb, encoding: str):
    """Decode a ``[B,3,H,W]`` normalized RGB tensor without normalizing it.

    Torch is intentionally imported lazily so dataset-preparation environments
    do not require it.
    """
    validate_normal_encoding(encoding)
    components = rgb.float() * 2.0 - 1.0
    if encoding == VISION_BANANA_NORMAL_ENCODING:
        components = components.clone()
        components[:, 0] *= -1
    return components
