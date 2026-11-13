from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.surface_normals.datasets import ibims1
from evaluation.surface_normals.inference_partial_flux2_klein_img2img import (
    load_gt_normals,
)


def test_marigold_reference_layout_and_implicit_mask(tmp_path: Path):
    Image.new("RGB", (3, 2), color=(10, 20, 30)).save(tmp_path / "room_01_img.png")
    normals = np.zeros((2, 3, 3), dtype=np.float32)
    normals[0, 0] = (0.0, 0.0, 2.0)
    normals[0, 1] = (1.0, 0.0, 0.0)
    np.save(tmp_path / "room_01_normal.npy", normals)

    root = ibims1.resolve_dataset_root(tmp_path)
    samples = ibims1.build_samples(root, "test", None, ("standard",), ("appearance",))
    assert len(samples) == 1
    assert samples[0].mask_path is None
    assert samples[0].output_stem == "ibims1__standard__room_01"

    loaded, valid = load_gt_normals(
        samples[0].normal_path,
        min_norm=0.01,
        rgb_encoding="legacy_direct_xyz",
    )
    np.testing.assert_allclose(loaded[0, 0], (0.0, 0.0, 1.0))
    np.testing.assert_allclose(loaded[0, 1], (1.0, 0.0, 0.0))
    assert valid.tolist() == [[True, True, False], [False, False, False]]

