from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from common import coco_eval_holdouts as holdouts
from dataset_prep.instanceseg.dataset_prep import (
    build_coconut_xlarge_pairs as build_instance_coconut_pairs,
)
from dataset_prep.instanceseg.dataset_prep import build_paco_lvis_pairs
from dataset_prep.refseg.prepare_steervit_refseg import item_evaluation_holdout
from dataset_prep.semseg.dataset_prep import (
    build_coconut_xlarge_pairs as build_semantic_coconut_pairs,
)


class CocoEvaluationHoldoutTest(unittest.TestCase):
    def test_missing_refcocog_annotations_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "Evaluation-image exclusion"):
                holdouts.load_coco_evaluation_holdouts(Path(temporary))
    def test_authenticated_holdouts_cover_every_evaluation_protocol(self) -> None:
        sets = holdouts.load_coco_evaluation_holdouts()
        self.assertEqual(len(sets["coco_val2017"]), 5000)
        self.assertEqual(len(sets["refcocog_umd_val"]), 1300)
        self.assertEqual(
            holdouts.holdout_reasons("COCO_train2014_000000000309.jpg", sets),
            ("refcocog_umd_val",),
        )
        self.assertEqual(
            holdouts.coco_image_id("COCO_train2017_000000000309.jpg"),
            "000000000309",
        )
        self.assertEqual(
            holdouts.coco_image_id(309, allow_unpadded_numeric_id=True),
            "000000000309",
        )
        self.assertEqual(holdouts.holdout_reasons("objects365_v2_00000309.jpg", sets), ())

    def test_refseg_and_audit_canonicalize_coco_ids_without_touching_vg_ids(self) -> None:
        exclusion = {"refcocog_umd_val": frozenset(("000000000309",))}
        reference, image_id, reasons = item_evaluation_holdout(
            {"dataset": "gRefCOCO", "image_id": 309, "id": "gref-row"}, exclusion
        )
        self.assertEqual((reference, image_id, reasons), ("309", "000000000309", ("refcocog_umd_val",)))
        self.assertEqual(
            item_evaluation_holdout(
                {"dataset": "visual genome", "image_id": 309, "id": "vg-row"}, exclusion
            ),
            (None, None, ()),
        )
        audit = holdouts.CocoEvaluationHoldoutAudit(exclusion)
        audit.record(
            "COCO_train2014_000000000309.jpg", reasons,
            source="refseg_grefcoco", reference="gref-row",
        )
        audit.record(
            "COCO_train2017_000000000309.jpg", reasons,
            source="refseg_grefcoco", reference="gref-row-duplicate",
        )
        with tempfile.TemporaryDirectory() as temporary:
            payload = json.loads(audit.write(temporary).read_text())
        self.assertEqual(payload["excluded_candidates"], 2)
        self.assertEqual(payload["excluded_distinct_source_images"], 1)
        self.assertEqual(payload["excluded_images"][0]["canonical_coco_image_id"], "000000000309")

    def test_both_coconut_builders_exclude_coco_eval_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "coco"
            (root / "train2017").mkdir(parents=True)
            (root / "panoptic_train2017").mkdir()
            (root / "annotations").mkdir()
            names = ("000000000139", "000000000309", "objects365_v2_00000309")
            for name in names:
                Image.new("RGB", (1, 1)).save(root / "train2017" / f"{name}.jpg")
                Image.new("RGB", (1, 1)).save(root / "panoptic_train2017" / f"{name}.png")
            images = [
                {"id": index, "file_name": f"{name}.jpg", "width": 1, "height": 1,
                 **({"object365_name": name} if name.startswith("objects365") else {})}
                for index, name in enumerate(names, start=1)
            ]
            annotations = [
                {"image_id": image["id"], "file_name": f"{Path(image['file_name']).stem}.png",
                 "segments_info": []}
                for image in images
            ]
            (root / "annotations" / "panoptic_train2017.json").write_text(json.dumps({
                "categories": [{"id": 1, "name": "person", "isthing": 1}],
                "images": images,
                "annotations": annotations,
            }))

            exclusion = {
                "coco_val2017": frozenset(("000000000139",)),
                "refcocog_umd_val": frozenset(("000000000309",)),
            }
            semantic_pairs, skipped = build_semantic_coconut_pairs(root, exclusion)
            instance_pairs = build_instance_coconut_pairs(root, exclusion)
            self.assertEqual(skipped, 0)
            self.assertEqual([pair.stem for pair in semantic_pairs], ["objects365_v2_00000309"])
            self.assertEqual([pair.stem for pair in instance_pairs], ["objects365_v2_00000309"])

    def test_paco_builder_excludes_coco_eval_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "paco_lvis"
            image_dir = root / "train2017" / "img"
            annotation_dir = root / "train2017" / "ann"
            image_dir.mkdir(parents=True)
            annotation_dir.mkdir()
            (root / "meta.json").write_text(json.dumps({"classes": []}))
            for image_id in ("000000000139", "000000000200"):
                Image.new("RGB", (1, 1)).save(image_dir / f"{image_id}.jpg")
                (annotation_dir / f"{image_id}.jpg.json").write_text(
                    '{"size":{"height":1,"width":1},"objects":[]}'
                )
            pairs = build_paco_lvis_pairs(
                root,
                ("train",),
                {"coco_val2017": frozenset(("000000000139",))},
            )
            self.assertEqual([pair.stem for pair in pairs], ["000000000200"])


if __name__ == "__main__":
    unittest.main()
