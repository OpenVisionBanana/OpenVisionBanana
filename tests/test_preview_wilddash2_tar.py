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
SCRIPT = ROOT / "scripts" / "dataset_prep" / "preview_wilddash2_tar.py"
SPEC = importlib.util.spec_from_file_location("preview_wilddash2_tar", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
preview = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preview
SPEC.loader.exec_module(preview)


def write_tar(path: Path) -> None:
    index = {
        "images": [
            {"id": 1, "file_name": "aa0001_100000.jpg"},
            {"id": 2, "file_name": "bb0002_100000.jpg"},
        ],
        "annotations": [
            {"image_id": 1, "file_name": "aa0001_100000.png", "segments_info": []},
            {"image_id": 2, "file_name": "bb0002_100000.png", "segments_info": []},
        ],
    }
    members = {
        "anonymized/aa0001_100000.jpg": b"train-image",
        "anonymized/bb0002_100000.jpg": b"val-image",
        "panoptic/aa0001_100000.png": b"train-panoptic",
        "panoptic/bb0002_100000.png": b"val-panoptic",
        "panoptic.json": json.dumps(index).encode("utf-8"),
        "wilddash2-meta.json": b'{"labels": []}',
        "random_split/training.txt": b"aa0001_100000\n",
        "random_split/validation.txt": b"bb0002_100000\n",
    }
    with tarfile.open(path, "w") as archive:
        for name, contents in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))


class PreviewWildDash2TarTests(unittest.TestCase):
    def test_metadata_discovery_and_selected_pair_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "wd_public_v2p.tar"
            write_tar(archive)
            source = root / "source"

            preview.extract_metadata_members(archive, source)
            samples = preview.discover_samples(source, ["train", "val"])
            self.assertEqual([sample.stem for sample in samples], ["aa0001_100000", "bb0002_100000"])
            selected = preview.select_samples(samples, "bb0002_100000", num_samples=99, seed=0)
            preview.extract_selected_members(archive, selected, source)

            self.assertEqual((source / "anonymized/bb0002_100000.jpg").read_bytes(), b"val-image")
            self.assertEqual((source / "panoptic/bb0002_100000.png").read_bytes(), b"val-panoptic")
            self.assertFalse((source / "anonymized/aa0001_100000.jpg").exists())

    def test_split_aliases_and_fixed_lr_draw_mode(self) -> None:
        self.assertEqual(preview.normalize_splits("training,validation"), ["train", "val"])
        commands: list[list[str]] = []
        original_run = preview._run
        preview._run = commands.append
        try:
            preview.build_prepared_previews(
                Path("source"), Path("out"), SimpleNamespace(
                    splits="train,val", seed=42, semseg_variants_per_image=1,
                    instanceseg_variants_per_image=1, instanceseg_negative_ratio=0.2,
                    draw_mode="fixeddraw_32_lr",
                ))
        finally:
            preview._run = original_run
        self.assertEqual(commands[1][-2:], ["--draw-mode", "fixeddraw_32_lr"])


if __name__ == "__main__":
    unittest.main()
