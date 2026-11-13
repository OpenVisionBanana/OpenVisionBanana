from __future__ import annotations

import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dataset_prep" / "preview_idd_tar.py"
SPEC = importlib.util.spec_from_file_location("preview_idd_tar", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
preview = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preview
SPEC.loader.exec_module(preview)


def nested_tar_bytes(members: dict[str, bytes]) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for name, contents in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))
    return payload.getvalue()


def write_outer_tar(path: Path) -> None:
    part_one = nested_tar_bytes({
        "IDD_Segmentation/leftImg8bit/train/101/frame0001_leftImg8bit.png": b"train-image",
        "IDD_Segmentation/gtFine/train/101/frame0001_gtFine_polygons.json": b"train-json",
        "IDD_Segmentation/leftImg8bit/train/101/unpaired_leftImg8bit.png": b"unused",
    })
    part_two = nested_tar_bytes({
        "idd20kII/leftImg8bit/val/202/frame0002_leftImg8bit.png": b"val-image",
        "idd20kII/gtFine/val/202/frame0002_gtFine_polygons.json": b"val-json",
    })
    with tarfile.open(path, "w") as archive:
        for name, contents in {
            "idd-segmentation.tar.gz": part_one,
            "idd-20k-II.tar.gz": part_two,
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))


class PreviewIDDTarTests(unittest.TestCase):
    def test_discovers_selects_and_extracts_only_requested_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "idd_1_2.tar"
            write_outer_tar(archive)

            samples = preview.discover_samples(archive, ["train", "val"])
            self.assertEqual(
                [sample.key for sample in samples],
                [
                    "IDD_Segmentation/train/101/frame0001",
                    "idd20kII/val/202/frame0002",
                ],
            )
            selected = preview.select_samples(
                samples, "idd20kII/val/202/frame0002", num_samples=99, seed=0)
            source = root / "source"
            preview.extract_selected_members(archive, selected, source)

            self.assertEqual(
                (source / "idd20kII/leftImg8bit/val/202/frame0002_leftImg8bit.png").read_bytes(),
                b"val-image",
            )
            self.assertEqual(
                (source / "idd20kII/gtFine/val/202/frame0002_gtFine_polygons.json").read_bytes(),
                b"val-json",
            )
            self.assertFalse((source / "IDD_Segmentation").exists())

    def test_split_aliases_and_sample_key_validation(self) -> None:
        self.assertEqual(preview.normalize_splits("training,validation"), ["train", "val"])
        sample = preview.Sample("IDD_Segmentation", "train", "101", "frame0001", "image", "annotation")
        with self.assertRaises(ValueError):
            preview.select_samples([sample], "missing", num_samples=1, seed=0)

    def test_instance_preview_uses_fixed_32_lr_draw_mode(self) -> None:
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
