import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

from common.instanceseg.prompts import (
    BASE_TEMPLATES,
    DRAW_MODES,
    ORDER_CONVENTION,
    ORDER_NUMBERING,
    build_draw_prompt,
    draw_prompt_templates,
    select_draw_plan,
)


class InstanceSegUnlistedDrawModesTest(unittest.TestCase):
    def test_random_unlisted_reuses_randomdraw_32_target_contract(self):
        parent = DRAW_MODES["randomdraw_32"]
        mode = DRAW_MODES["randomdraw_32_unlisted"]
        for field in ("k", "palette", "style", "randomize", "scan"):
            self.assertEqual(mode[field], parent[field])
        self.assertEqual(mode["prompt_contract"], "unlisted")

        permutation = list(reversed(range(32)))
        template = BASE_TEMPLATES[0]
        prompt, colors = build_draw_prompt(
            "randomdraw_32_unlisted", "cars", template, permutation=permutation)
        _, parent_colors = build_draw_prompt(
            "randomdraw_32", "cars", template, permutation=permutation)

        self.assertEqual(colors, parent_colors)
        self.assertEqual(prompt, template.format(np="cars"))
        self.assertNotIn(ORDER_NUMBERING, prompt)
        self.assertNotIn("The 1st cars is", prompt)

    def test_random_unlisted_keeps_parent_eval_draw_plan(self):
        fields = ("sa1b_nps", 1234, "traffic lights", 42)
        self.assertEqual(
            select_draw_plan(*fields, "randomdraw_32_unlisted"),
            select_draw_plan(*fields, "randomdraw_32"),
        )

    def test_ordered_unlisted_reuses_fixeddraw_32_target_contract(self):
        parent = DRAW_MODES["fixeddraw_32"]
        mode = DRAW_MODES["fixeddraw_32_ordered_unlisted"]
        for field in ("k", "palette", "style", "randomize", "scan"):
            self.assertEqual(mode[field], parent[field])
        self.assertEqual(mode["prompt_contract"], "ordered_unlisted")

        template = BASE_TEMPLATES[0]
        prompt, colors = build_draw_prompt(
            "fixeddraw_32_ordered_unlisted", "cars", template)
        _, parent_colors = build_draw_prompt("fixeddraw_32", "cars", template)

        self.assertEqual(colors, parent_colors)
        self.assertEqual(prompt, template.format(np="cars") + ORDER_CONVENTION)
        self.assertIn("i-th instance always gets the i-th palette color", prompt)
        self.assertNotIn("The 1st cars is", prompt)

    def test_static_templates_match_the_built_ordered_contract(self):
        expected = [template + ORDER_CONVENTION for template in BASE_TEMPLATES]
        self.assertEqual(
            draw_prompt_templates("fixeddraw_32_ordered_unlisted"), expected)
        self.assertEqual(draw_prompt_templates("randomdraw_32_unlisted"), BASE_TEMPLATES)

    def test_random_unlisted_still_requires_a_valid_target_permutation(self):
        with self.assertRaisesRegex(ValueError, "requires a permutation"):
            build_draw_prompt("randomdraw_32_unlisted", "cars", BASE_TEMPLATES[0])
        with self.assertRaisesRegex(ValueError, "fixed-order"):
            build_draw_prompt(
                "fixeddraw_32_ordered_unlisted",
                "cars",
                BASE_TEMPLATES[0],
                permutation=list(range(32)),
            )


@unittest.skipIf(Image is None, "Pillow is not installed")
class InstanceSegUnlistedDatasetPrepTest(unittest.TestCase):
    def _materialize(self, root: Path, mode: str):
        from dataset_prep.instanceseg.dataset_prep import (
            Config,
            DatasetInfo,
            Pair,
            write_image_variants,
        )

        source = root / "source"
        source.mkdir(exist_ok=True)
        image_path = source / "sample.png"
        ann_path = source / "sample_instances.png"
        Image.new("RGB", (8, 8), (90, 100, 110)).save(image_path)

        annotation = np.zeros((8, 8, 3), dtype=np.uint8)
        annotation[0:3, 0:3, 0] = 1
        annotation[0:3, 0:3, 1] = 1
        annotation[4:8, 4:8, 0] = 1
        annotation[4:8, 4:8, 1] = 2
        Image.fromarray(annotation, "RGB").save(ann_path)

        output = root / mode
        conditioning = output / "conditioning_images"
        targets = output / "target_images"
        conditioning.mkdir(parents=True)
        targets.mkdir(parents=True)
        pair = Pair("ade20k_inst", "train", "sample", image_path, ann_path)
        cfg = Config(
            variants_per_image=1,
            negative_ratio=0.0,
            draw_mode=mode,
            seed=42,
            datasets={
                "ade20k_inst": DatasetInfo(names={1: "car"}, instanceable={1}),
            },
        )
        row = write_image_variants(pair, conditioning, targets, cfg)[0]
        return row, (output / row["image"]).read_bytes()

    def test_unlisted_modes_change_prompts_not_parent_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixed, fixed_png = self._materialize(root, "fixeddraw_32")
            fixed_unlisted, fixed_unlisted_png = self._materialize(
                root, "fixeddraw_32_ordered_unlisted")
            random, random_png = self._materialize(root, "randomdraw_32")
            random_unlisted, random_unlisted_png = self._materialize(
                root, "randomdraw_32_unlisted")

        self.assertEqual(fixed_unlisted_png, fixed_png)
        self.assertEqual(random_unlisted_png, random_png)
        self.assertEqual(
            random_unlisted["metadata"]["permutation"],
            random["metadata"]["permutation"],
        )
        for row, contract in (
            (fixed_unlisted, "ordered_unlisted"),
            (random_unlisted, "unlisted"),
        ):
            metadata = row["metadata"]
            self.assertEqual(metadata["prompt_contract"], contract)
            self.assertEqual(metadata["palette_size"], 32)
            self.assertEqual(metadata["legend_len"], 0)
            self.assertIsNone(metadata["notation_style"])


if __name__ == "__main__":
    unittest.main()
