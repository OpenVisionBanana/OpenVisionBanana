from __future__ import annotations

import numpy as np
import pytest

from dataset_prep.geometry_conventions import orient_normals_toward_camera
from evaluation.surface_normals.target_protocol import resolve_ground_truth_protocol


def test_float_npy_targets_are_identity_physical_xyz() -> None:
    protocol = resolve_ground_truth_protocol(
        dataset_info={"normal_encoding": "float_xyz_npy after upstream conversion"},
        storage_formats=["npy"],
    )
    assert protocol.encoding == "float_xyz_npy"
    assert protocol.permutation == (0, 1, 2)
    assert protocol.signs == (1, 1, 1)


def test_metadata_resolves_vision_banana_rgb_without_axis_guess() -> None:
    protocol = resolve_ground_truth_protocol(
        dataset_info={
            "normal_encoding": "vision_banana_x_left_red",
            "stored_normal_convention": "camera_x_right_y_up_z_toward_camera",
            "gt_to_canonical_permutation": [0, 1, 2],
            "gt_to_canonical_signs": [1, 1, 1],
        },
        storage_formats=["png"],
    )
    assert protocol.encoding == "vision_banana_x_left_red"
    assert protocol.signs == (1, 1, 1)


def test_ambiguous_rgb_archive_fails_closed() -> None:
    with pytest.raises(ValueError, match="no unambiguous normal encoding"):
        resolve_ground_truth_protocol(dataset_info={}, storage_formats=["png"])


def test_rgb_encoding_without_axis_contract_fails_closed() -> None:
    with pytest.raises(ValueError, match="not declared canonical"):
        resolve_ground_truth_protocol(
            dataset_info={"normal_encoding": "legacy_direct_xyz"},
            storage_formats=["png"],
        )


def test_camera_ray_orientation_handles_floor_normal_with_zero_z() -> None:
    points = np.array([[[0.0, 1.0, 5.0], [1.0, 1.0, 5.0]]], dtype=np.float32)
    # Positive-z orientation cannot distinguish the two signs when z == 0.
    normals = np.array([[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]], dtype=np.float32)
    oriented = orient_normals_toward_camera(normals, points)
    dots = np.sum(oriented * points, axis=-1)
    assert np.all(dots <= 0.0)
    np.testing.assert_allclose(oriented[0, 0], [0.0, -1.0, 0.0])
    np.testing.assert_allclose(oriented[0, 1], [0.0, -1.0, 0.0])
