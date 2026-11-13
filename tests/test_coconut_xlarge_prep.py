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
    DatasetInfo,
    build_argparser as build_instance_argparser,
    build_coconut_xlarge_pairs as build_instance_pairs,
    read_masks,
)
from dataset_prep.semseg.dataset_prep import (
    build_argparser as build_semantic_argparser,
    build_coconut_xlarge_pairs as build_semantic_pairs,
)


def panoptic_rgb(ids: np.ndarray) -> np.ndarray:
    return np.stack((ids % 256, (ids // 256) % 256, ids // (256 * 256)), axis=-1).astype(np.uint8)


class CoconutXLargePrepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "coconut_xlarge" / "coco"
        (self.root / "train2017").mkdir(parents=True)
        (self.root / "panoptic_train2017").mkdir()
        (self.root / "annotations").mkdir()
        ids = np.array([[7, 7, 9], [7, 3, 3]], dtype=np.int64)
        Image.new("RGB", (3, 2), (20, 30, 40)).save(self.root / "train2017" / "sample.jpg")
        Image.fromarray(panoptic_rgb(ids)).save(self.root / "panoptic_train2017" / "sample.png")
        Image.new("RGB", (3, 2), (50, 60, 70)).save(self.root / "train2017" / "objects365.jpg")
        Image.fromarray(panoptic_rgb(ids)).save(self.root / "panoptic_train2017" / "objects365.png")
        (self.root / "annotations" / "panoptic_train2017.json").write_text(json.dumps({
            "categories": [
                {"id": 1, "name": "person", "isthing": 1},
                {"id": 92, "name": "sky", "isthing": 0},
                {"id": 199, "name": "wall-other-merged", "isthing": 0},
            ],
            "images": [
                {"id": 10, "file_name": "sample.jpg", "width": 3, "height": 2},
                {"id": 11, "file_name": "objects365.jpg", "object365_name": "objects365_demo", "width": 3, "height": 2},
            ],
            "annotations": [
                {
                    "image_id": 10,
                    "file_name": "sample.png",
                    "segments_info": [
                        {"id": 7, "category_id": 1, "iscrowd": 0},
                        {"id": 9, "category_id": 92, "iscrowd": 0},
                        {"id": 3, "category_id": 1, "iscrowd": 1},
                    ],
                },
                {
                    "image_id": 11,
                    "file_name": "objects365.png",
                    "segments_info": [{"id": 7, "category_id": 1, "iscrowd": 0}],
                },
            ],
        }), encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_semantic_and_instance_panoptic_adapters(self) -> None:
        parent = self.root.parent
        self.assertEqual(taxonomy.resolve_coconut_xlarge_root(parent), self.root.resolve())
        semantic_pairs, skipped = build_semantic_pairs(parent)
        self.assertEqual(skipped, 0)
        self.assertEqual([(pair.stem, pair.split, pair.group) for pair in semantic_pairs], [
            ("sample", "train", "coco"),
            ("objects365_demo", "train", "objects365"),
        ])
        self.assertEqual(taxonomy.class_names("coconut_xlarge"), ("person", "sky", "wall other"))
        labels = taxonomy.read_label_array(
            semantic_pairs[0].mask_path, "coconut_xlarge", semantic_pairs[0].segments_info)
        np.testing.assert_array_equal(labels, np.array([[0, 0, 1], [0, 0, 0]], dtype=np.int64))

        instance_pairs = build_instance_pairs(parent)
        names, instanceable = instance_taxonomy.coconut_xlarge_categories(parent)
        self.assertEqual(names, {1: "person", 92: "sky", 199: "wall other"})
        self.assertEqual(instanceable, {1})
        masks = read_masks(instance_pairs[0], DatasetInfo(names, instanceable))
        self.assertEqual({class_id: len(items) for class_id, items in masks.items()}, {1: 1})
        np.testing.assert_array_equal(masks[1][0], np.array([
            [True, True, False], [True, False, False],
        ]))

    def test_objects365_group_selection(self) -> None:
        parent = self.root.parent
        semantic_pairs, skipped = build_semantic_pairs(parent, groups="objects365")
        self.assertEqual(skipped, 0)
        self.assertEqual([(pair.stem, pair.group) for pair in semantic_pairs], [
            ("objects365_demo", "objects365"),
        ])
        instance_pairs = build_instance_pairs(parent, groups="objects365")
        self.assertEqual([pair.stem for pair in instance_pairs], ["objects365_demo"])
        with self.assertRaisesRegex(ValueError, "Unknown COCONut-XL source group"):
            build_semantic_pairs(parent, groups="other")

    def test_both_converters_accept_the_coco_root(self) -> None:
        semantic = build_semantic_argparser().parse_args([
            "--out-root", "out", "--coconut-xlarge-root", str(self.root),
            "--coconut-xlarge-groups", "objects365",
        ])
        instance = build_instance_argparser().parse_args([
            "--out-root", "out", "--coconut-xlarge-root", str(self.root),
            "--coconut-xlarge-groups", "objects365",
        ])
        self.assertEqual(semantic.coconut_xlarge_root, str(self.root))
        self.assertEqual(instance.coconut_xlarge_root, str(self.root))
        self.assertEqual(semantic.coconut_xlarge_groups, "objects365")
        self.assertEqual(instance.coconut_xlarge_groups, "objects365")


if __name__ == "__main__":
    unittest.main()
