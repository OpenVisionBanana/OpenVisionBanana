import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from common.semseg import taxonomy
from dataset_prep.instanceseg import instance_taxonomy
from dataset_prep.instanceseg.dataset_prep import build_idd_pairs as build_instance_pairs
from dataset_prep.semseg.dataset_prep import build_idd_pairs as build_semantic_pairs


class IDDPrepTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "idd"
        self._write_example("IDD_Segmentation", "train", "101", "frame0001")
        self._write_example("idd20kII", "val", "202", "frame0001")

    def tearDown(self):
        self.tmp.cleanup()

    def _write_example(self, part: str, split: str, drive: str, stem: str) -> Path:
        source = self.root / part
        image_dir = source / "leftImg8bit" / split / drive
        annotation_dir = source / "gtFine" / split / drive
        image_dir.mkdir(parents=True, exist_ok=True)
        annotation_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (5, 4), (20, 30, 40)).save(
            image_dir / f"{stem}_leftImg8bit.png")
        annotation_path = annotation_dir / f"{stem}_gtFine_polygons.json"
        annotation_path.write_text(json.dumps({
            "imgWidth": 5,
            "imgHeight": 4,
            "objects": [
                {"label": "road", "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]]},
                {"label": "car", "polygon": [[1, 1], [3, 1], [3, 2], [1, 2]]},
                # Later polygons obscure earlier ones, as in the official script.
                {"label": "person", "polygon": [[2, 1], [4, 1], [4, 3], [2, 3]]},
                {"label": "truck", "deleted": True,
                 "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]},
            ],
        }), encoding="utf-8")
        return annotation_path

    def test_two_part_root_discovers_unique_semantic_and_instance_pairs(self):
        roots = taxonomy.resolve_idd_roots(self.root)
        self.assertEqual([path.name for path in roots], ["IDD_Segmentation", "idd20kII"])

        semantic_pairs, skipped = build_semantic_pairs(self.root, ["train", "val"])
        self.assertEqual(skipped, 0)
        self.assertEqual(len(semantic_pairs), 2)
        self.assertEqual(len({pair.stem for pair in semantic_pairs}), 2)

        instance_pairs = build_instance_pairs(self.root, ["train", "val"])
        self.assertEqual(len(instance_pairs), 2)
        self.assertEqual(len({pair.stem for pair in instance_pairs}), 2)

    def test_level3_semantics_and_visible_raw_id_instances(self):
        annotation = (
            self.root / "IDD_Segmentation" / "gtFine" / "train" / "101" /
            "frame0001_gtFine_polygons.json"
        )
        decoded = taxonomy.read_label_array(annotation, "idd")
        self.assertEqual(len(taxonomy.class_names("idd")), 26)
        self.assertEqual(decoded[0, 0], 0)   # road level3Id
        self.assertEqual(decoded[1, 1], 9)   # car level3Id
        self.assertEqual(decoded[1, 2], 4)   # later person obscures car
        self.assertEqual(decoded[3, 4], 4)

        names, instanceable = instance_taxonomy.idd_classes()
        self.assertEqual(names[11], "autorickshaw")
        masks = instance_taxonomy.read_idd(annotation, instanceable)
        self.assertEqual(set(masks), {6, 12})
        car, person = masks[12][0], masks[6][0]
        self.assertTrue(car.any())
        self.assertTrue(person.any())
        self.assertFalse(np.any(car & person))
        self.assertFalse(car[1, 2])
        self.assertTrue(person[1, 2])


if __name__ == "__main__":
    unittest.main()
