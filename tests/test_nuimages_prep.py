from __future__ import annotations

import json
import base64
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from common.semseg import taxonomy
from dataset_prep.instanceseg import instance_taxonomy
from dataset_prep.instanceseg.dataset_prep import (
    DatasetInfo,
    build_nuimages_pairs as build_instance_pairs,
    read_masks,
)
from dataset_prep.semseg.dataset_prep import build_nuimages_pairs as build_semantic_pairs


def rle_full(height: int, width: int) -> dict:
    return {"size": [height, width], "counts": f"0{height * width}"}


class NuImagesPrepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "nuimages"
        (self.root / "samples" / "CAM_FRONT").mkdir(parents=True)
        self._write_split("train", [
            ("sample_a", "sd_a", "log_train", 0, "a.jpg"),
            ("sample_b", "sd_b", "log_train", 2_000_000, "b.jpg"),
            ("sample_c", "sd_c", "log_train", 10_000_000, "c.jpg"),
            ("sample_no_mask", "sd_no_mask", "log_train", 12_000_000, "none.jpg"),
        ])
        self._write_split("val", [
            ("sample_v", "sd_v", "log_val", 0, "v.jpg"),
        ])
        categories = [
            {"token": "surface", "name": "flat.driveable_surface"},
            {"token": "car", "name": "vehicle.car"},
            {"token": "ego", "name": "vehicle.ego"},
            {"token": "rack", "name": "static_object.bicycle_rack"},
        ]
        for split in ("train", "val"):
            (self.root / f"v1.0-{split}" / "category.json").write_text(
                json.dumps(categories), encoding="utf-8")

        # Base64-wrapped RLE "123" means column-major counts [1, 2, 3] in a 2x3 image.
        car = {"size": [2, 3], "counts": base64.b64encode(b"123").decode("ascii")}
        train = self.root / "v1.0-train"
        (train / "surface_ann.json").write_text(json.dumps([
            {"sample_data_token": "sd_a", "category_token": "surface", "mask": rle_full(2, 3)},
            {"sample_data_token": "sd_c", "category_token": "surface", "mask": rle_full(2, 3)},
        ]), encoding="utf-8")
        (train / "object_ann.json").write_text(json.dumps([
            {"sample_data_token": "sd_a", "category_token": "car", "mask": car},
            {"sample_data_token": "sd_b", "category_token": "car", "mask": car},
            {"sample_data_token": "sd_c", "category_token": "ego", "mask": rle_full(2, 3)},
            {"sample_data_token": "sd_c", "category_token": "rack", "mask": rle_full(2, 3)},
            {"sample_data_token": "sd_no_mask", "category_token": "car", "mask": None},
        ]), encoding="utf-8")
        val = self.root / "v1.0-val"
        (val / "surface_ann.json").write_text(json.dumps([]), encoding="utf-8")
        (val / "object_ann.json").write_text(json.dumps([
            {"sample_data_token": "sd_v", "category_token": "car", "mask": car},
        ]), encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_split(self, split: str, frames: list[tuple[str, str, str, int, str]]) -> None:
        table = self.root / f"v1.0-{split}"
        table.mkdir(parents=True, exist_ok=True)
        samples, sample_data = [], []
        for token, data_token, log_token, timestamp, filename in frames:
            Image.new("RGB", (3, 2), (20, 30, 40)).save(self.root / "samples" / "CAM_FRONT" / filename)
            samples.append({
                "token": token, "key_camera_token": data_token,
                "log_token": log_token, "timestamp": timestamp,
            })
            sample_data.append({
                "token": data_token, "is_key_frame": True,
                "filename": f"samples/CAM_FRONT/{filename}", "timestamp": timestamp,
            })
        (table / "sample.json").write_text(json.dumps(samples), encoding="utf-8")
        (table / "sample_data.json").write_text(json.dumps(sample_data), encoding="utf-8")

    def test_compressed_rle_uses_coco_column_major_order(self) -> None:
        mask = taxonomy.decode_coco_rle({"size": [2, 3], "counts": "123"})
        np.testing.assert_array_equal(mask, np.array([
            [False, True, False],
            [True, False, False],
        ]))

    def test_semantic_pairs_sample_per_log_and_render_surface_then_objects(self) -> None:
        pairs, skipped = build_semantic_pairs(self.root, ["train", "val"], 10.0)
        self.assertEqual(skipped, 1)
        self.assertEqual([(pair.stem, pair.split) for pair in pairs], [
            ("sample_a", "train"), ("sample_c", "train"), ("sample_v", "val"),
        ])
        self.assertEqual(taxonomy.class_names("nuimages"), (
            "flat driveable surface", "vehicle car", "vehicle ego",
        ))
        self.assertNotIn("static object bicycle rack", taxonomy.class_names("nuimages"))
        ignored = taxonomy.read_label_array(
            pairs[1].mask_path, "nuimages", annotations=pairs[1].annotations)
        labels = taxonomy.read_label_array(
            pairs[0].mask_path, "nuimages", annotations=pairs[0].annotations)
        np.testing.assert_array_equal(labels, np.array([
            [0, 1, 0],
            [1, 0, 0],
        ], dtype=np.int64))

    def test_instances_keep_foreground_masks_but_exclude_ego_vehicle(self) -> None:
        pairs = build_instance_pairs(self.root, ["train", "val"], 10.0)
        self.assertEqual([(pair.stem, pair.split) for pair in pairs], [
            ("sample_a", "train"), ("sample_c", "train"), ("sample_v", "val"),
        ])
        names, instanceable, token_to_index = instance_taxonomy.nuimages_categories(self.root)
        self.assertEqual(names[1], "vehicle car")
        self.assertEqual(instanceable, {1})
        masks = read_masks(pairs[0], DatasetInfo(names, instanceable, token_to_index))
        self.assertEqual(set(masks), {1})
        np.testing.assert_array_equal(masks[1][0], np.array([
            [False, True, False],
            [True, False, False],
        ]))
        self.assertEqual(read_masks(pairs[1], DatasetInfo(names, instanceable, token_to_index)), {})


if __name__ == "__main__":
    unittest.main()
