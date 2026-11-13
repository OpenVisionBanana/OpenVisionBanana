"""Public preparation must respect temporary storage and requested corpus sizes."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import pytest
from scripts.dataset_prep.prepare_creative_training import prepare as creative
from scripts.dataset_prep import prepare_geometry_training as geometry


def add(archive, name, value):
    member = tarfile.TarInfo(name)
    member.size = len(value)
    archive.addfile(member, io.BytesIO(value))


def generation_source(raw, count):
    (raw / "shards").mkdir(parents=True)
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buf, format="JPEG")
    with tarfile.open(raw / "shards/000000.tar", "w") as archive:
        for index in range(count):
            add(archive, f"{index}.jpg", buf.getvalue())
            add(archive, f"{index}.txt", b"A red image")
            add(archive, f"{index}.json", b"{}")


@pytest.mark.parametrize("available", [1, 3])
@pytest.mark.parametrize("previous", [False, True])
def test_public_pd12m_requires_count_and_preserves_old_archive(tmp_path, available, previous):
    raw = tmp_path / "raw"
    generation_source(raw, available)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    output = tmp_path / "prepared/train/ImgGeneration/pd12m_250000.tar"
    if previous:
        output.parent.mkdir(parents=True)
        output.write_bytes(b"previous completed archive")
        Path(str(output) + ".sha256").write_text("previous checksum")
    args = SimpleNamespace(dataset="pd12m", raw_dir=raw, root=tmp_path / "prepared",
                           limit=3, seed=42, workers=1, overwrite=previous)
    if available < 3:
        with pytest.raises(subprocess.CalledProcessError):
            creative(args, scratch)
        if previous:
            assert output.read_bytes() == b"previous completed archive"
            assert Path(str(output) + ".sha256").read_text() == "previous checksum"
        else:
            assert not output.exists() and not Path(str(output) + ".sha256").exists()
        assert not (scratch / "pd12m/metadata.jsonl").exists()
    else:
        creative(args, scratch)
        with tarfile.open(output) as archive:
            member = next(m for m in archive if m.name.endswith("metadata.jsonl"))
            assert len(archive.extractfile(member).read().splitlines()) == 3
        assert Path(str(output) + ".ovb-index.sqlite").is_file()


def test_legacy_pd12m_partial_selection_remains_explicitly_supported(tmp_path):
    raw = tmp_path / "raw"
    generation_source(raw, 1)
    result = subprocess.run([sys.executable, str(ROOT / "scripts/dataset_prep/materialize_pd12m_canonical.py"),
        "--shards-dir", str(raw / "shards"), "--out", str(tmp_path / "out"), "--limit", "3"], capture_output=True, text=True)
    assert result.returncode == 0 and "WARNING" in result.stderr
    assert len((tmp_path / "out/metadata.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("available", [1, 3])
@pytest.mark.parametrize("previous", [False, True])
def test_training_pool_uses_strict_pd12m_adapter_without_unpublishing_old_output(tmp_path, available, previous):
    raw = tmp_path / "raw"
    generation_source(raw / "ImgGeneration/pd12m", available)
    root = tmp_path / "prepared"
    output = root / "train/ImgGeneration/pd12m_250000.tar"
    if previous:
        output.parent.mkdir(parents=True)
        output.write_bytes(b"previous completed archive")
        Path(str(output) + ".sha256").write_text("previous checksum")
    command = [sys.executable, str(ROOT / "scripts/dataset_prep/build_training_pool.py"),
        "--pool", "strict_count_test", "--sources", "pd12m", "--raw-root", str(raw),
        "--root", str(root), "--limit", "3", "--workers", "1"]
    if previous:
        command.append("--overwrite")
    result = subprocess.run(command, capture_output=True, text=True)
    if available < 3:
        assert result.returncode != 0 and "requested 3 PD12M samples" in result.stderr
        assert not (root / "manifests/strict_count_test_pool.json").exists()
        if previous:
            assert output.read_bytes() == b"previous completed archive"
            assert Path(str(output) + ".sha256").read_text() == "previous checksum"
        else:
            assert not output.exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        manifest = json.loads((root / "manifests/strict_count_test_pool.json").read_text())
        assert len(manifest["datasets"]) == 1
        with tarfile.open(output) as archive:
            member = next(m for m in archive if m.name.endswith("metadata.jsonl"))
            assert len(archive.extractfile(member).read().splitlines()) == 3


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_pd12m_count_has_no_output(tmp_path, limit):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/dataset_prep/materialize_pd12m_canonical.py"),
        "--shards-dir", str(tmp_path / "missing"), "--out", str(tmp_path / "out"), "--limit", str(limit)], capture_output=True)
    assert result.returncode == 2 and not (tmp_path / "out").exists()
    with pytest.raises(ValueError, match="positive"):
        creative(SimpleNamespace(limit=limit), tmp_path)


@pytest.mark.parametrize("failure", [False, True])
def test_diode_spool_uses_selected_workspace_and_cleans_up(tmp_path, monkeypatch, failure):
    raw, work, output = tmp_path / "raw", tmp_path / "work", tmp_path / "persistent/train"
    raw.mkdir()
    work.mkdir()
    rgb = np.full((32, 32, 3), 80, np.uint8)
    depth = np.ones((32, 32), np.float32) * 2
    mask = np.ones((32, 32), np.uint8)
    normal = np.broadcast_to(np.array([0, 0, 1], np.float32), (32, 32, 3))
    key = "train/indoors/scene/frame"
    with tarfile.open(raw / "train.tar.gz", "w:gz") as archive:
        for suffix, value in ((".png", geometry.png(rgb)), ("_depth.npy", geometry.npy(depth)),
                              ("_depth_mask.npy", geometry.npy(mask))):
            add(archive, key + suffix, value)
    with tarfile.open(raw / "train_normals.tar.gz", "w:gz") as archive:
        add(archive, key + "_normal.npy", geometry.npy(normal))
    monkeypatch.setenv("OVB_SCRATCH_ROOT", str(work))
    real = geometry.tempfile.TemporaryFile
    spools, handles = [], []
    def trace(*args, **kwargs):
        spools.append(Path(kwargs["dir"]))
        handle = real(*args, **kwargs)
        handles.append(handle)
        return handle
    monkeypatch.setattr(geometry.tempfile, "TemporaryFile", trace)
    if failure:
        def fail(*args):
            raise RuntimeError("injected conversion failure")
        monkeypatch.setattr(geometry, "add_bytes", fail)
        with pytest.raises(RuntimeError, match="injected"):
            geometry.prepare("diode", raw, output, 1, None)
    else:
        geometry.prepare("diode", raw, output, 1, None)
        with tarfile.open(output / "Normals/diode.tar") as archive:
            row = json.loads(archive.extractfile("metadata.jsonl").read())
            actual = np.load(io.BytesIO(archive.extractfile(row["raw_normals"]).read()))
            np.testing.assert_array_equal(actual, normal)
    assert len(spools) == 1 and spools[0].parent == work
    assert all(handle.closed for handle in handles)
    assert list(work.iterdir()) == []
    if failure:
        assert not (output / "Depth/diode.tar").exists()
        assert not (output / "Normals/diode.tar").exists()
    else:
        other_work = tmp_path / "other_work"
        other_work.mkdir()
        monkeypatch.setenv("OVB_SCRATCH_ROOT", str(other_work))
        other_output = tmp_path / "other_prepared"
        geometry.prepare("diode", raw, other_output, 1, None)
        for task in ("Depth", "Normals"):
            assert (output / task / "diode.tar").read_bytes() == (other_output / task / "diode.tar").read_bytes()
        assert list(other_work.iterdir()) == []
        assert all(handle.closed for handle in handles)
