from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dataset_prep" / "preview_coconut_xlarge.py"
SPEC = importlib.util.spec_from_file_location("preview_coconut_xlarge", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
preview = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preview
SPEC.loader.exec_module(preview)


def panoptic_rgb(ids: np.ndarray) -> np.ndarray:
    return np.stack((ids % 256, (ids // 256) % 256, ids // (256 * 256)), axis=-1).astype(np.uint8)


class PreviewCoconutXLargeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "coconut_xlarge" / "coco"
        (self.root / "train2017").mkdir(parents=True)
        (self.root / "panoptic_train2017").mkdir()
        (self.root / "annotations").mkdir()
        for name, sid in (("one.jpg", 7), ("two.jpg", 9)):
            Image.new("RGB", (2, 2), (20, 30, 40)).save(self.root / "train2017" / name)
            Image.fromarray(panoptic_rgb(np.full((2, 2), sid))).save(
                self.root / "panoptic_train2017" / f"{Path(name).stem}.png")
        (self.root / "annotations" / "panoptic_train2017.json").write_text(json.dumps({
            "categories": [
                {"id": 1, "name": "person", "isthing": 1},
                {"id": 92, "name": "sky", "isthing": 0},
            ],
            "images": [
                {"id": 10, "file_name": "one.jpg"},
                {"id": 11, "file_name": "two.jpg", "object365_name": "objects365_v2_11"},
            ],
            "annotations": [
                {"image_id": 10, "file_name": "one.png",
                 "segments_info": [{"id": 7, "category_id": 1, "iscrowd": 1}]},
                {"image_id": 11, "file_name": "two.png",
                 "segments_info": [{"id": 9, "category_id": 1, "iscrowd": 0}]},
            ],
        }), encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_selects_non_crowd_thing_pairs_and_materializes_only_them(self) -> None:
        index = preview.discover_samples(self.root.parent)
        self.assertEqual(index.candidate_ids, ["11"])
        selected = preview.select_samples(index, "objects365_v2_11", num_samples=99, seed=0)
        self.assertEqual(selected[0].image_id, "11")
        source = Path(self.tmp.name) / "preview" / "source" / "coco"
        preview.copy_selected_pairs(index, selected, source)
        self.assertTrue((source / "train2017" / "two.jpg").is_file())
        self.assertTrue((source / "panoptic_train2017" / "two.png").is_file())
        self.assertFalse((source / "train2017" / "one.jpg").exists())
        mini = json.loads((source / "annotations" / "panoptic_train2017.json").read_text())
        self.assertEqual([image["id"] for image in mini["images"]], [11])
        self.assertEqual([annotation["image_id"] for annotation in mini["annotations"]], [11])

    def test_fixed_lr_preview_invokes_both_converters(self) -> None:
        commands: list[list[str]] = []
        original_run = preview.run
        preview.run = commands.append
        try:
            preview.build_prepared_previews(
                Path("source/coco"), Path("out"), SimpleNamespace(
                    seed=42, semseg_variants_per_image=1,
                    instanceseg_variants_per_image=1,
                    instanceseg_negative_ratio=0.0,
                    draw_mode="fixeddraw_32_lr",
                ))
        finally:
            preview.run = original_run
        self.assertIn("--coconut-xlarge-root", commands[0])
        self.assertIn("--coconut-xlarge-root", commands[1])
        self.assertEqual(commands[1][-2:], ["--draw-mode", "fixeddraw_32_lr"])
        parsed = preview.build_argparser().parse_args([
            "--coconut-xlarge-root", "source/coco", "--out-root", "out",
        ])
        self.assertEqual(parsed.draw_mode, "fixeddraw_32_lr")


if __name__ == "__main__":
    unittest.main()
