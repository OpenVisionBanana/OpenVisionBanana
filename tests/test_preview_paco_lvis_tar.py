from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dataset_prep" / "preview_paco_lvis_tar.py"
SPEC = importlib.util.spec_from_file_location("preview_paco_lvis_tar", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
preview = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preview
SPEC.loader.exec_module(preview)


def write_tar(path: Path) -> None:
    members = {
        "meta.json": json.dumps({"classes": [{"id": 1, "title": "car", "shape": "any"}]}).encode(),
        "train2017/img/000000000001.jpg": b"train-image",
        "train2017/ann/000000000001.jpg.json": b'{"size":{"height":1,"width":1},"objects":[]}',
        "val2017/img/000000000002.jpg": b"val-image",
        "val2017/ann/000000000002.jpg.json": b'{"size":{"height":1,"width":1},"objects":[]}',
        "test2017/img/000000000003.jpg": b"test-image",
        "test2017/ann/000000000003.jpg.json": b'{"size":{"height":1,"width":1},"objects":[]}',
    }
    with tarfile.open(path, "w") as archive:
        for name, contents in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))


class PreviewPacoLvisTarTests(unittest.TestCase):
    def test_discovery_and_selected_pair_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "paco-lvis-DatasetNinja.tar"
            write_tar(archive)
            source = root / "source"

            samples = preview.discover_samples(archive, ["train", "val"])
            self.assertEqual([sample.key for sample in samples], [
                "train/000000000001", "val/000000000002",
            ])
            selected = preview.select_samples(samples, "val/000000000002", num_samples=99, seed=0)
            preview.extract_metadata(archive, source)
            preview.extract_selected_members(archive, selected, source)

            self.assertTrue((source / "meta.json").is_file())
            self.assertEqual((source / "val2017/img/000000000002.jpg").read_bytes(), b"val-image")
            self.assertEqual(
                (source / "val2017/ann/000000000002.jpg.json").read_bytes(),
                b'{"size":{"height":1,"width":1},"objects":[]}',
            )
            self.assertFalse((source / "train2017/img/000000000001.jpg").exists())

    def test_aliases_and_fixed_lr_draw_mode(self) -> None:
        self.assertEqual(preview.normalize_splits("training,validation,testing"), ["train", "val", "test"])
        parsed = preview.build_argparser().parse_args([
            "--paco-lvis-tar", "source.tar", "--out-root", "out",
        ])
        self.assertEqual(parsed.splits, "train,val,test")
        commands: list[list[str]] = []
        original_run = preview.run
        preview.run = commands.append
        try:
            preview.build_prepared_preview(
                Path("source"), Path("out"), SimpleNamespace(
                    splits="train,val", seed=42, instanceseg_variants_per_image=1,
                    instanceseg_negative_ratio=0.0, draw_mode="fixeddraw_32_lr",
                ))
        finally:
            preview.run = original_run
        self.assertEqual(commands[0][-2:], ["--draw-mode", "fixeddraw_32_lr"])


if __name__ == "__main__":
    unittest.main()
