import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from common.semseg import taxonomy
from dataset_prep.instanceseg import instance_taxonomy
from dataset_prep.instanceseg.dataset_prep import build_synscapes_pairs as build_instance_pairs
from dataset_prep.semseg.dataset_prep import build_synscapes_pairs as build_semantic_pairs


class SynscapesPrepTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "download" / "Synscapes"
        for relative in ("img/rgb", "img/class", "img/instance", "meta"):
            (self.root / relative).mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_frame(self, index: int) -> None:
        Image.new("RGB", (3, 2), (20, 30, 40)).save(
            self.root / "img" / "rgb" / f"{index}.png")
        semantic = np.array([[7, 24, 26], [0, 25, 33]], dtype=np.uint8)
        Image.fromarray(semantic).save(
            self.root / "img" / "class" / f"{index}.png")
        Image.new("RGB", (3, 2), (0, 0, 0)).save(
            self.root / "img" / "instance" / f"{index}.png")

    def test_root_resolution_and_numbered_train_val_split(self):
        self._write_frame(24_000)
        self._write_frame(24_001)

        parent = self.root.parent
        self.assertEqual(taxonomy.resolve_synscapes_root(parent), self.root.resolve())
        self.assertEqual(taxonomy.synscapes_split_for_index(24_000), "train")
        self.assertEqual(taxonomy.synscapes_split_for_index(24_001), "val")

        train_pairs, skipped = build_semantic_pairs(parent, ["train"])
        self.assertEqual(skipped, 0)
        self.assertEqual([(pair.stem, pair.split) for pair in train_pairs],
                         [("24000", "train")])

        val_pairs = build_instance_pairs(parent, ["validation"])
        self.assertEqual([(pair.stem, pair.split) for pair in val_pairs],
                         [("24001", "val")])

    def test_semantic_raw_ids_map_to_cityscapes_train_ids(self):
        self._write_frame(1)
        decoded = taxonomy.read_label_array(
            self.root / "img" / "class" / "1.png", "synscapes")
        expected = np.array([[0, 11, 13], [255, 12, 18]], dtype=np.int64)
        np.testing.assert_array_equal(decoded, expected)

    def test_rgb_instances_use_metadata_classes_then_semantic_fallback(self):
        index = 1
        self._write_frame(index)
        instance_ids = np.array([[1, 1, 258], [0, 258, 258]], dtype=np.int64)
        rgb = np.stack((
            instance_ids % 256,
            (instance_ids // 256) % 256,
            (instance_ids // (256 * 256)) % 256,
        ), axis=-1).astype(np.uint8)
        instance_path = self.root / "img" / "instance" / f"{index}.png"
        Image.fromarray(rgb, "RGB").save(instance_path)
        semantic = np.array([[24, 24, 26], [0, 26, 26]], dtype=np.uint8)
        Image.fromarray(semantic).save(
            self.root / "img" / "class" / f"{index}.png")

        metadata_path = self.root / "meta" / f"{index}.json"
        metadata_path.write_text(json.dumps({
            "instance": {"class": {"1": 24, "258": 25}}
        }), encoding="utf-8")

        names, instanceable = instance_taxonomy.synscapes_classes()
        self.assertEqual(names[24], "person")
        self.assertEqual(names[25], "rider")
        masks = instance_taxonomy.read_synscapes(
            instance_path,
            self.root / "img" / "class" / f"{index}.png",
            metadata_path,
            instanceable,
        )
        self.assertEqual(set(masks), {24, 25})
        np.testing.assert_array_equal(masks[24][0], instance_ids == 1)
        np.testing.assert_array_equal(masks[25][0], instance_ids == 258)

        fallback = instance_taxonomy.read_synscapes(
            instance_path,
            self.root / "img" / "class" / f"{index}.png",
            None,
            instanceable,
        )
        self.assertIn(26, fallback)


if __name__ == "__main__":
    unittest.main()
