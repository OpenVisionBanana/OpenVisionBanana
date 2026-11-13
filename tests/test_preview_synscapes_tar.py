from __future__ import annotations

import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dataset_prep" / "preview_synscapes_tar.py"
SPEC = importlib.util.spec_from_file_location("preview_synscapes_tar", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
preview = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preview)


class PreviewSynscapesTarTests(unittest.TestCase):
    def test_select_indices_honors_splits_and_explicit_ids(self) -> None:
        self.assertEqual(preview.select_indices([1, 24_001], num_samples=99,
                                                splits=["train", "val"], seed=0),
                         [1, 24_001])
        self.assertEqual(preview.select_indices(None, num_samples=3, splits=["val"], seed=7),
                         [24_155, 24_332, 24_971])
        with self.assertRaises(ValueError):
            preview.select_indices([24_001], num_samples=1, splits=["train"], seed=0)

    def test_extracts_only_requested_frame_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "synscapes.tar"
            names = (
                "Synscapes/img/rgb/1.png",
                "Synscapes/img/class/1.png",
                "Synscapes/img/instance/1.png",
                "Synscapes/meta/1.json",
                "Synscapes/img/rgb/2.png",
            )
            with tarfile.open(archive_path, "w") as archive:
                for name in names:
                    payload = name.encode("utf-8")
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))

            source_root = root / "source"
            preview.extract_selected_members(archive_path, [1], source_root)
            for name in names[:4]:
                relative = Path(*Path(name).parts[1:])
                self.assertEqual((source_root / "Synscapes" / relative).read_bytes(), name.encode("utf-8"))
            self.assertFalse((source_root / "Synscapes/img/rgb/2.png").exists())
