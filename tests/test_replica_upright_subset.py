import io
import tarfile

import numpy as np
from PIL import Image
import pytest

from scripts.dataset_prep.rebuild_replica_upright_subset import add_member, png_bytes, upright_rotation


def test_upright_rotation_aligns_camera_up_with_scene_z():
    pytest.importorskip('habitat_sim')
    quaternion = pytest.importorskip('quaternion')
    for yaw in (0, .7, 2.1, 5.2):
        for pitch in (-.15, 0, .15):
            matrix = quaternion.as_rotation_matrix(upright_rotation(yaw, pitch))
            assert matrix[2, 1] == pytest.approx(np.cos(pitch))
            assert -matrix[2, 2] == pytest.approx(np.sin(pitch))
            assert np.linalg.det(matrix) == pytest.approx(1.)


def test_png_and_tar_roundtrip_preserve_exact_pixels_without_loose_files():
    image = Image.fromarray(np.arange(12, dtype='uint8').reshape(2, 2, 3))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        add_member(archive, 'conditioning_images/test.png', png_bytes(image))
    buffer.seek(0)
    with tarfile.open(fileobj=buffer, mode='r:') as archive:
        entry = archive.getmember('conditioning_images/test.png')
        assert entry.mtime == 0
        with Image.open(archive.extractfile(entry)) as decoded:
            assert np.array_equal(np.asarray(decoded), np.asarray(image))
