from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from common.semseg import taxonomy
from dataset_prep.instanceseg import instance_taxonomy
from dataset_prep.instanceseg.dataset_prep import (
    build_wilddash2_pairs as build_instance_pairs,
    read_masks,
    DatasetInfo,
)
from dataset_prep.semseg.dataset_prep import build_wilddash2_pairs as build_semantic_pairs


def panoptic_rgb(ids: np.ndarray) -> np.ndarray:
    return np.stack((ids % 256, (ids // 256) % 256, (ids // 65_536) % 256), axis=-1).astype(np.uint8)


class WildDash2PrepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "wilddash2"
        (self.root / "anonymized").mkdir(parents=True)
        (self.root / "panoptic").mkdir()
        (self.root / "random_split").mkdir()
        self._write_source()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_source(self) -> None:
        labels = [
            {"id": 0, "name": "unlabeled", "readable": "unlabeled(void)",
             "supercategory": "void", "instances": False},
            {"id": 7, "name": "road", "readable": "road(flat)",
             "supercategory": "flat", "instances": False},
            {"id": 26, "name": "car", "readable": "car(vehicle)",
             "supercategory": "vehicle", "instances": True},
            {"id": 36, "name": "billboard", "readable": "billboard(object)",
             "supercategory": "object", "instances": True},
        ]
        (self.root / "wilddash2-meta.json").write_text(json.dumps({"labels": labels}), encoding="utf-8")
        (self.root / "random_split/training.txt").write_text("aa0001_100000\n", encoding="utf-8")
        (self.root / "random_split/validation.txt").write_text("bb0002_100000\n", encoding="utf-8")

        annotations = []
        images = []
        for image_id, stem, split in ((1, "aa0001_100000", "train"), (2, "bb0002_100000", "val")):
            Image.new("RGB", (3, 2), (20, 30, 40)).save(self.root / "anonymized" / f"{stem}.jpg")
            ids = np.array([[100, 200, 200], [100, 300, 300]], dtype=np.int64)
            Image.fromarray(panoptic_rgb(ids)).save(self.root / "panoptic" / f"{stem}.png")
            images.append({"id": image_id, "file_name": f"{stem}.jpg"})
            annotations.append({
                "image_id": image_id,
                "file_name": f"{stem}.png",
                "segments_info": [
                    {"id": 100, "category_id": 7, "iscrowd": 0},
                    {"id": 200, "category_id": 26, "iscrowd": 0},
                    {"id": 300, "category_id": 36, "iscrowd": 1},
                ],
            })
        (self.root / "panoptic.json").write_text(
            json.dumps({"images": images, "annotations": annotations}), encoding="utf-8")

    def test_semantic_pairs_and_compact_nonvoid_panoptic_labels(self) -> None:
        pairs, skipped = build_semantic_pairs(self.root.parent, ["train", "validation"])
        self.assertEqual(skipped, 0)
        self.assertEqual([(pair.stem, pair.split) for pair in pairs], [
            ("aa0001_100000", "train"), ("bb0002_100000", "val"),
        ])
        self.assertEqual(taxonomy.class_names("wilddash2"), ("road", "car", "billboard"))
        labels = taxonomy.read_label_array(
            pairs[0].mask_path, "wilddash2", pairs[0].segments_info)
        np.testing.assert_array_equal(labels, np.array([[0, 1, 1], [0, 2, 2]], dtype=np.int64))

    def test_instance_pairs_keep_non_crowd_things(self) -> None:
        pairs = build_instance_pairs(self.root.parent, ["train", "val"])
        self.assertEqual(len(pairs), 2)
        names, instanceable = instance_taxonomy.wilddash2_categories(self.root)
        self.assertEqual(names[26], "car")
        self.assertIn(36, instanceable)
        masks = read_masks(pairs[0], DatasetInfo(names, instanceable))
        self.assertEqual(set(masks), {26})
        np.testing.assert_array_equal(
            masks[26][0], np.array([[False, True, True], [False, False, False]]))


if __name__ == "__main__":
    unittest.main()
