from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from dataset_prep.instanceseg import instance_taxonomy
from dataset_prep.instanceseg.dataset_prep import (
    DatasetInfo,
    build_argparser,
    build_paco_lvis_pairs,
    read_masks,
)


class PacoLvisPrepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "paco-lvis-DatasetNinja"
        image_dir = self.root / "train2017" / "img"
        annotation_dir = self.root / "train2017" / "ann"
        image_dir.mkdir(parents=True)
        annotation_dir.mkdir(parents=True)
        Image.new("RGB", (7, 6), (20, 30, 40)).save(image_dir / "000000000001.jpg")
        (self.root / "meta.json").write_text(json.dumps({"classes": [
            {"id": 1, "title": "car_(automobile)", "shape": "any"},
            {"id": 2, "title": "strap", "shape": "any"},
            {"id": 3, "title": "book", "shape": "any"},
        ]}), encoding="utf-8")

        bitmap = Image.fromarray(np.array([[255, 0], [0, 255]], dtype=np.uint8))
        payload = io.BytesIO()
        bitmap.save(payload, format="PNG")
        bitmap_data = base64.b64encode(zlib.compress(payload.getvalue())).decode("ascii")
        (annotation_dir / "000000000001.jpg.json").write_text(json.dumps({
            "size": {"height": 6, "width": 7},
            "objects": [
                {"classId": 1, "geometryType": "polygon", "points": {
                    "exterior": [[0, 0], [2, 0], [2, 2], [0, 2]], "interior": []}},
                # DatasetNinja includes a rectangle duplicate of detailed geometry.
                {"classId": 1, "geometryType": "rectangle", "points": {
                    "exterior": [[0, 0], [2, 2]]}},
                {"classId": 2, "geometryType": "bitmap", "bitmap": {
                    "origin": [3, 1], "data": bitmap_data}},
                {"classId": 2, "geometryType": "rectangle", "points": {
                    "exterior": [[3, 1], [4, 2]]}},
                # A rectangle without a detailed counterpart remains a valid instance.
                {"classId": 3, "geometryType": "rectangle", "points": {
                    "exterior": [[5, 3], [6, 4]]}},
            ],
        }), encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_pairs_decode_all_native_geometries_without_bbox_duplicates(self) -> None:
        self.assertEqual(
            instance_taxonomy.resolve_paco_lvis_root(self.root.parent), self.root.resolve())
        names, instanceable = instance_taxonomy.paco_lvis_classes(self.root.parent)
        self.assertEqual(names, {1: "car", 2: "strap", 3: "book"})
        self.assertEqual(instanceable, {1, 2, 3})

        pairs = build_paco_lvis_pairs(self.root.parent, ["training"])
        self.assertEqual([(pair.stem, pair.split) for pair in pairs], [
            ("000000000001", "train"),
        ])
        masks = read_masks(pairs[0], DatasetInfo(names, instanceable))
        self.assertEqual({class_id: len(items) for class_id, items in masks.items()}, {
            1: 1, 2: 1, 3: 1,
        })
        self.assertEqual(int(masks[1][0].sum()), 9)
        np.testing.assert_array_equal(masks[2][0], np.array([
            [False, False, False, False, False, False, False],
            [False, False, False, True, False, False, False],
            [False, False, False, False, True, False, False],
            [False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False],
        ]))
        self.assertEqual(int(masks[3][0].sum()), 4)


    def test_default_converter_splits_include_all_released_annotations(self) -> None:
        parsed = build_argparser().parse_args(["--out-root", "out", "--paco-lvis-root", "source"])
        self.assertEqual(parsed.paco_lvis_splits, "train,val,test")


if __name__ == "__main__":
    unittest.main()
