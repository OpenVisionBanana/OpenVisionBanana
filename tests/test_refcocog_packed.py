"""Packed source safety and acquisition contracts, without network or masks."""
import hashlib
import json
from pathlib import Path
import pickle
import stat
import sys
import zipfile

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.refcocog_packed import zip_entries, validation_images
from scripts.dataset_prep import acquire_refcocog_packed as acquisition


def annotation_files(raw):
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "instances.json").write_text(json.dumps(dict(images=[dict(id=1, file_name="photo.jpg")],
        annotations=[dict(id=2, image_id=1)])))
    with (raw / "refs(umd).p").open("wb") as f:
        pickle.dump([dict(ref_id=3, ann_id=2, image_id=1, split="val")], f)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a\\escape", "a/./b"])
def test_rejects_unsafe_zip_names(tmp_path, name):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name, b"bad")
    with zipfile.ZipFile(path) as z, pytest.raises(ValueError, match="Unsafe"):
        zip_entries(z)


def test_rejects_zip_symlink(tmp_path):
    path = tmp_path / "source.zip"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(info, "../source")
    with zipfile.ZipFile(path) as z, pytest.raises(ValueError, match="Unsafe"):
        zip_entries(z)


def test_acquisition_reuses_packed_original_and_keeps_holdout_guard(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    annotation_files(raw)
    shared = tmp_path / "shared/train2014.zip"
    shared.parent.mkdir()
    with zipfile.ZipFile(shared, "w") as z:
        z.writestr("train2014/photo.jpg", b"fixture photo")
    before = hashlib.sha256(shared.read_bytes()).hexdigest()
    calls = []
    monkeypatch.setattr(acquisition, "_read_refcocog_umd_val_ids", lambda p: calls.append(p))
    def forbidden(*args):
        pytest.fail("Existing official assets must not be downloaded again")
    acquisition.acquire(raw, shared, fetch=forbidden)
    acquisition.acquire(raw, shared, fetch=forbidden)
    assert len(calls) == 2
    assert (raw / "train2014.zip").resolve() == shared.resolve()
    assert not (raw / "train2014").exists()
    assert hashlib.sha256(shared.read_bytes()).hexdigest() == before


def test_val_only_extraction_and_cleanup(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    annotation_files(raw)
    node = tmp_path / "node"
    node.mkdir()
    monkeypatch.setenv("OVB_SCRATCH_ROOT", str(node))
    with zipfile.ZipFile(raw / "train2014.zip", "w") as z:
        z.writestr("train2014/photo.jpg", b"fixture photo")
        z.writestr("train2014/extra.jpg", b"not selected")
    with validation_images(raw) as (_, prepared):
        assert (prepared / "train2014/photo.jpg").read_bytes() == b"fixture photo"
        assert not (prepared / "train2014/extra.jpg").exists()
    assert list(node.iterdir()) == []


def test_real_holdout_guard_rejects_tiny_annotations_before_fetch(tmp_path):
    raw = tmp_path / "raw"
    annotation_files(raw)
    with pytest.raises(ValueError, match="Unexpected RefCOCOg"):
        acquisition.acquire(raw, tmp_path / "never-downloaded.zip", fetch=lambda *a: pytest.fail("fetch"))


def test_annotation_acquisition_reads_only_required_original_members(tmp_path, monkeypatch):
    source = tmp_path / "source"
    annotation_files(source)
    original = tmp_path / "original.zip"
    with zipfile.ZipFile(original, "w") as z:
        for name in ("refs(umd).p", "instances.json"):
            z.write(source / name, "refcocog/" + name)
        z.writestr("refcocog/unneeded.p", b"unused")
    raw = tmp_path / "raw"
    def fetch(url, destination):
        import shutil
        shutil.copyfile(original, destination)
    calls = []
    monkeypatch.setattr(acquisition, "_read_refcocog_umd_val_ids", lambda p: calls.append(p))
    assert acquisition.acquire_annotations(raw, fetch=fetch) == raw
    assert len(calls) == 1 and not (raw / "unneeded.p").exists()
    assert not (raw / "train2014.zip").exists()


def test_missing_val_image_does_not_leave_scratch(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    annotation_files(raw)
    node = tmp_path / "node"
    node.mkdir()
    monkeypatch.setenv("OVB_SCRATCH_ROOT", str(node))
    with zipfile.ZipFile(raw / "train2014.zip", "w") as z:
        z.writestr("train2014/extra.jpg", b"not selected")
    with pytest.raises(KeyError):
        with validation_images(raw):
            pytest.fail("Missing requested images must fail")
    assert list(node.iterdir()) == []
