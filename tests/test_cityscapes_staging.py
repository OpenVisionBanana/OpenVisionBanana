"""Reuse the existing raw Cityscapes route without treating prepared rows as raw."""
import io
import tarfile

from common.official_sources import complete_cityscapes_raw_archive, missing_manual_assets
from common.release_data import BY_NAME, raw_directory, staged_source_status


def fixture(path, counts, missing=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for split, count in counts.items():
            for index in range(count):
                stem = f"city_000000_{index:06d}"
                names = [f"cityscapes/leftImg8bit/{split}/city/{stem}_leftImg8bit.png"]
                names += [f"cityscapes/gtFine/{split}/city/{stem}_gtFine_{suffix}"
                          for suffix in ("labelIds.png", "labelTrainIds.png", "polygons.json")]
                for name in names[:-1] if missing else names:
                    member = tarfile.TarInfo(name)
                    member.size = 1
                    archive.addfile(member, io.BytesIO(b"x"))


def test_archive_requires_complete_pairs_and_original_raw_layout(tmp_path):
    path = tmp_path / "cityscapes.tar"
    fixture(path, {"train": 2, "val": 1})
    assert complete_cityscapes_raw_archive(path, {"train": 2, "val": 1})
    assert not complete_cityscapes_raw_archive(path)  # Small fixtures are not full sources.
    fixture(path, {"train": 2, "val": 1}, missing=True)
    assert not complete_cityscapes_raw_archive(path, {"train": 2, "val": 1})
    path.write_bytes(b"not a tar")
    assert not complete_cityscapes_raw_archive(path)


def test_complete_raw_tar_clears_inputs_but_never_forges_acquisition(tmp_path):
    item = BY_NAME["cityscapes"]
    folder = raw_directory(tmp_path, item)
    fixture(folder / "cityscapes.tar", {"train": 2975, "val": 500})
    assert missing_manual_assets(tmp_path, [item]) == []
    assert not staged_source_status(tmp_path, item)[0]
    assert not (folder / ".ovb-download-complete").exists()


def test_raw_archive_rejects_unsafe_extra_members(tmp_path):
    path = tmp_path / "cityscapes.tar"
    fixture(path, {"train": 2, "val": 1})
    with tarfile.open(path, "a") as archive:
        member = tarfile.TarInfo("../outside.txt")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    assert not complete_cityscapes_raw_archive(path, {"train": 2, "val": 1})
