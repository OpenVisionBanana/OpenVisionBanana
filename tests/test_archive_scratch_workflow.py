"""Archive-first conversion preserves sources and bounds loose files to scratch."""
import io
import json
import sys
from pathlib import Path
import shutil
from types import SimpleNamespace
import tarfile
import zipfile

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import node_scratch
from common.hypersim_zip import ZipMember
from common.release_eval import locate_dataset
from dataset_prep.hypersim import prepare_hypersim_depth_normals as geometry
from dataset_prep.hypersim import prepare_hypersim_segmentation as segmentation
from scripts.dataset_prep import acquire_cityscapes_packed as city


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    root = tmp_path / "node"
    root.mkdir()
    monkeypatch.setenv("TMPDIR", str(root))
    monkeypatch.delenv("OVB_SCRATCH_ROOT", raising=False)
    monkeypatch.setenv("OVB_SCRATCH_RESERVE_BYTES", "0")
    return root


def test_owned_workspace_cleanup_and_source_preservation(tmp_path, scratch):
    source = tmp_path / "original.tar"
    source.write_bytes(b"original")
    with pytest.raises(RuntimeError):
        with node_scratch.workspace("failure") as work:
            (work / "input.tar").symlink_to(source)
            (work / "loose").write_bytes(b"temporary")
            raise RuntimeError("injected interruption")
    assert list(scratch.iterdir()) == []
    assert source.read_bytes() == b"original"


def test_capacity_fails_before_workspace_creation(scratch, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(OSError, match="Insufficient scratch bytes"):
        with node_scratch.workspace("too-large", bytes_needed=2):
            pytest.fail("Must reject before conversion")
    assert list(scratch.iterdir()) == []


def test_capacity_checks_inodes(scratch, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=10**9))
    monkeypatch.setattr(node_scratch.os, "statvfs", lambda _: SimpleNamespace(f_files=100, f_favail=1))
    with pytest.raises(OSError, match="Insufficient scratch entries"):
        node_scratch.require_capacity(scratch, entries_needed=2)


def test_system_temporary_fallback_without_environment(tmp_path, monkeypatch):
    import tempfile
    for key in ("TMPDIR", "TMP", "TEMP", "OVB_SCRATCH_ROOT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(tempfile, "tempdir", None)
    root = node_scratch.scratch_root()
    assert root.is_dir()
    with node_scratch.workspace("portable-default") as work:
        assert work.parent == root
        (work / "sample").write_text("temporary")
    assert not work.exists()


def test_explicit_work_dir_and_missing_root(tmp_path, monkeypatch):
    monkeypatch.setenv("OVB_SCRATCH_ROOT", str(tmp_path / "unavailable"))
    node_scratch.configure_work_dir(tmp_path)
    with node_scratch.workspace("explicit") as work:
        assert work.parent == tmp_path
    with pytest.raises(ValueError, match="already exist"):
        node_scratch.configure_work_dir(tmp_path / "missing")
    assert list(tmp_path.iterdir()) == []


def test_redirects_only_loose_geometry_outputs(scratch):
    command = ["converter.py", "--raw-root", "/original", "--depth-out-root", "/persistent/depth",
               "--normals-out-root", "/persistent/normals", "--depth-prepared-tar", "/persistent/depth.tar",
               "--scratch-root", "/persistent/scratch", "--shard-root", "/persistent/shards"]
    changed = node_scratch.conversion_command(command, scratch)
    assert changed[changed.index("--depth-prepared-tar") + 1] == "/persistent/depth.tar"
    assert changed[changed.index("--raw-root") + 1] == "/original"
    for flag in ("--depth-out-root", "--normals-out-root", "--scratch-root", "--shard-root"):
        assert Path(changed[changed.index(flag) + 1]).is_relative_to(scratch)
    assert command[command.index("--depth-out-root") + 1] == "/persistent/depth"


def test_acquisition_redirects_work_not_inputs_or_archives(scratch):
    env = {"WORK": "/raw/goose/_download", "ARCHIVE_DIR": "/raw/goose",
           "GOOSE_BUILD_ROOT": "/raw/goose/_build", "ACDC_DIR": "/raw/acdc"}
    changed = node_scratch.acquisition_environment(env, scratch, "goose")
    assert changed["ARCHIVE_DIR"] == env["ARCHIVE_DIR"]
    assert changed["ACDC_DIR"] == env["ACDC_DIR"]
    for key in ("WORK", "GOOSE_BUILD_ROOT", "TMPDIR"):
        assert Path(changed[key]).is_relative_to(scratch)
    assert Path(node_scratch.acquisition_environment({}, scratch, "acdc")["WORK"]).is_relative_to(scratch)


def hypersim_fixture(root):
    scene = root / "ai_001_001/images"
    base = "frame.0000"
    arrays = {"color": np.ones((8, 8, 3), dtype=np.float32),
              "depth_meters": np.full((8, 8), 2, dtype=np.float32),
              "normal_cam": np.tile([0, 0, -1.0], (8, 8, 1)).astype(np.float32),
              "semantic": np.ones((8, 8), dtype=np.int16),
              "semantic_instance": np.ones((8, 8), dtype=np.int16),
              "render_entity_id": np.ones((8, 8), dtype=np.int16)}
    for channel, array in arrays.items():
        directory = scene / ("scene_cam_00_final_hdf5" if channel == "color" else "scene_cam_00_geometry_hdf5")
        directory.mkdir(parents=True, exist_ok=True)
        with h5py.File(directory / (base + "." + channel + ".hdf5"), "w") as handle:
            handle["dataset"] = array
    return arrays


def archive_hypersim(source, destination):
    folder = destination / "archives"
    folder.mkdir(parents=True)
    archive = folder / "ai_001_001.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for path in sorted(source.rglob("*.hdf5")):
            target.write(path, path.relative_to(source).as_posix())
    return archive


def test_hypersim_zip_arrays_and_sample_selection_match_extracted(tmp_path):
    extracted, packed = tmp_path / "extracted", tmp_path / "packed"
    hypersim_fixture(extracted)
    archive = archive_hypersim(extracted, packed)
    before = archive.read_bytes()
    for module, fields in ((geometry, ("rgb_path", "depth_path", "normal_path")),
                           (segmentation, ("rgb_path", "semantic_path", "instance_path", "entity_path"))):
        plain, zipped = module.discover_frames(extracted), module.discover_frames(packed)
        assert len(plain) == len(zipped) == 1
        assert plain[0].sample_id == zipped[0].sample_id
        for field in fields:
            a, b = getattr(plain[0], field), getattr(zipped[0], field)
            assert isinstance(b, ZipMember)
            np.testing.assert_array_equal(geometry.read_hdf5_array(a), geometry.read_hdf5_array(b))
        kwargs = dict(samples_per_scene=None, target_samples=None, all_frames=True, seed=42, val_ratio=0)
        assert [f.sample_id for f in geometry.sample_frames(plain, **kwargs)] == [f.sample_id for f in geometry.sample_frames(zipped, **kwargs)]
    assert archive.read_bytes() == before
    assert not (packed / "ai_001_001").exists()
    with pytest.raises(ValueError, match="Deleting archived"):
        geometry.maybe_delete_raw_scenes(geometry.discover_frames(packed), packed)


def test_hypersim_zip_and_extracted_targets_match(tmp_path):
    extracted, packed = tmp_path / "extracted", tmp_path / "packed"
    hypersim_fixture(extracted)
    archive_hypersim(extracted, packed)
    for source in (extracted, packed):
        args = geometry.build_argparser().parse_args(["--raw-root", str(source),
            "--depth-out-root", str(tmp_path / source.name / "outputs/depth"),
            "--normals-out-root", str(tmp_path / source.name / "outputs/normals"),
            "--legacy-euclidean-depth"])
        frame = geometry.discover_frames(source)[0]
        geometry.write_depth_sample(frame, args)
        geometry.write_normal_sample(frame, args)
    left, right = extracted / "outputs", packed / "outputs"
    for path in left.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (right / path.relative_to(left)).read_bytes()


def test_unsafe_hypersim_zip_is_rejected(tmp_path):
    folder = tmp_path / "archives"
    folder.mkdir()
    with zipfile.ZipFile(folder / "bad.zip", "w") as archive:
        archive.writestr("../outside", b"x")
    with pytest.raises(ValueError, match="Unsafe Hypersim"):
        geometry.discover_frames(tmp_path)


def write_city_tar(path):
    with tarfile.open(path, "w") as archive:
        for split, count in (("train", 2975), ("val", 500)):
            for index in range(count):
                stem = f"city_000000_{index:06d}"
                names = [f"cityscapes/leftImg8bit/{split}/city/{stem}_leftImg8bit.png"]
                names += [f"cityscapes/gtFine/{split}/city/{stem}_gtFine_{suffix}"
                          for suffix in ("labelIds.png", "labelTrainIds.png", "polygons.json")]
                for name in names:
                    member = tarfile.TarInfo(name)
                    member.size = 1
                    archive.addfile(member, io.BytesIO(b"x"))


def test_existing_city_tar_needs_no_extraction(tmp_path, scratch):
    write_city_tar(tmp_path / "cityscapes.tar")
    city.acquire(tmp_path, runner=lambda *a, **k: pytest.fail("No packer needed"))
    assert not (tmp_path / "gtFine").exists()
    assert list(scratch.iterdir()) == []


def test_city_failure_recovers_without_mutating_original_zips(tmp_path, scratch):
    inputs = []
    for name in ("gtFine_trainvaltest.zip", "leftImg8bit_trainvaltest.zip"):
        path = tmp_path / name
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("fixture", b"x")
        inputs.append((path, path.read_bytes()))
    def fail(*args, **kwargs):
        Path(kwargs["env"]["CITYSCAPES_DIR"], "incomplete").write_bytes(b"loose")
        raise RuntimeError("injected helper interruption")
    with pytest.raises(RuntimeError):
        city.acquire(tmp_path, runner=fail)
    assert not (tmp_path / "cityscapes.tar").exists()
    assert list(scratch.iterdir()) == []
    def succeed(*args, **kwargs):
        write_city_tar(Path(kwargs["env"]["ARCHIVE_DIR"]) / "cityscapes.tar")
    city.acquire(tmp_path, runner=succeed)
    assert city.complete_cityscapes_raw_archive(tmp_path / "cityscapes.tar")
    assert list(scratch.iterdir()) == []
    for path, original in inputs:
        assert path.read_bytes() == original
    assert {p.name for p in tmp_path.iterdir() if p != scratch} == {p.name for p, _ in inputs} | {"cityscapes.tar"}


def test_eval_tar_cache_is_node_local_even_on_non_main_node(tmp_path, scratch):
    source = tmp_path / "cityscapes.tar"
    with tarfile.open(source, "w") as archive:
        member = tarfile.TarInfo("cityscapes/gtFine/val/example.png")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    nodes = SimpleNamespace(main=False, barrier=lambda _: None)
    results = tmp_path / "results"
    located = locate_dataset(source, results, "Semseg", "cityscapes", nodes=nodes)
    assert located.is_relative_to(scratch)
    assert (located / "gtFine/val/example.png").read_bytes() == b"x"
    assert not results.exists()
    assert locate_dataset(source, results, "Semseg", "cityscapes", nodes=nodes) == located


def test_packed_city_eval_does_not_restore_source_tree(tmp_path, scratch, monkeypatch):
    import sys
    from scripts.dataset_prep import prepare_task_eval
    raw = tmp_path / "raw"
    raw.mkdir()
    write_city_tar(raw / "cityscapes.tar")
    output = tmp_path / "prepared/eval/Semseg/cityscapes"
    monkeypatch.setattr(sys, "argv", ["prepare_task_eval", "--dataset", "cityscapes", "--raw-dir", str(raw), "--out-dir", str(output)])
    assert prepare_task_eval.main() == 0
    assert not output.exists()
    assert not (raw / "gtFine").exists()
    assert output.with_suffix(".tar").samefile(raw / "cityscapes.tar")
    assert prepare_task_eval.main() == 0


def test_unsafe_city_zip_never_runs_helper(tmp_path, scratch):
    for name in ("gtFine_trainvaltest.zip", "leftImg8bit_trainvaltest.zip"):
        with zipfile.ZipFile(tmp_path / name, "w") as archive:
            archive.writestr("../outside", b"x")
    with pytest.raises(ValueError, match="Unsafe original ZIP"):
        city.acquire(tmp_path, runner=lambda *a, **k: pytest.fail("Must reject unsafe input"))
    assert list(scratch.iterdir()) == []


def test_synscapes_packed_source_requires_matching_native_modalities(tmp_path):
    from common.official_sources import complete_synscapes_raw_archive
    archive_path = tmp_path / "synscapes.tar"
    with tarfile.open(archive_path, "w") as archive:
        for stem in ("1", "2"):
            for group in ("rgb", "depth", "class", "instance", "meta"):
                prefix = "Synscapes/meta" if group == "meta" else f"Synscapes/img/{group}"
                suffix = ".json" if group == "meta" else ".exr" if group == "depth" else ".png"
                member = tarfile.TarInfo(f"{prefix}/{stem}{suffix}")
                member.size = 1
                archive.addfile(member, io.BytesIO(b"x"))
    assert complete_synscapes_raw_archive(archive_path, expected_frames=2)
    assert not complete_synscapes_raw_archive(archive_path)
    with tarfile.open(archive_path, "a") as archive:
        member = tarfile.TarInfo("../unsafe")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    assert not complete_synscapes_raw_archive(archive_path, expected_frames=2)


def test_synscapes_geometry_selects_existing_tar_without_changing_estimators(tmp_path, scratch, monkeypatch):
    from scripts.dataset_prep import prepare_synscapes_geometry
    archive = tmp_path / "synscapes.tar"
    archive.write_bytes(b"fixture; command construction only")
    commands = []
    monkeypatch.setattr(prepare_synscapes_geometry.subprocess, "run", lambda command, **kwargs: commands.append(command))
    args = SimpleNamespace(raw_dir=tmp_path, output=tmp_path / "prepared", seed=42, workers=1,
                           max_side=0, limit=2, overwrite=False)
    prepare_synscapes_geometry.prepare(args, scratch)
    assert len(commands) == 2
    for command in commands:
        assert command[command.index("--synscapes-tar") + 1] == str(archive)
        assert "--synscapes-root" not in command
        assert Path(command[command.index("--out-root") + 1]).is_relative_to(scratch)
    assert commands[0][commands[0].index("--depth-encoding") + 1] == "vision_banana_rgb"
    assert commands[1][commands[1].index("--normal-encoding") + 1] == "vision_banana_x_left_red"
