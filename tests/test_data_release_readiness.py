"""Regression checks for preparation dependencies, integrity and continuation."""
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import indexed_tar
from common.data_preparation_state import run_stage, stage_lock, input_identity
from common.official_sources import scannet_exclusions, SCANNET_SPLITS
from common.release_data import BY_NAME, acquisition_items
from scripts import setup_data, prepare_data


def test_release_inventory_includes_diode_guard_regression():
    inventory = json.loads((ROOT / "DATA_RELEASE_FILES.json").read_text())
    assert "tests/test_geometry_release_correctness.py" in inventory["overlay"]


def test_snapshot_rejects_duplicate_inventory_before_writes(tmp_path, monkeypatch):
    from scripts.dataset_prep import check_data_release
    (tmp_path / "DATA_RELEASE_FILES.json").write_text(json.dumps(dict(
        overlay=["same.py"], additional=["same.py"])))
    monkeypatch.setattr(check_data_release, "REPO", tmp_path)
    destination = tmp_path / "candidate"
    with pytest.raises(ValueError, match="Duplicate release inventory"):
        check_data_release.snapshot(destination)
    assert not destination.exists()


def test_selected_release_tests_cannot_be_omitted():
    from scripts.dataset_prep.check_data_release import CHECK_TESTS, validate_check_inventory
    names = ["tests/"+name for name in CHECK_TESTS]
    validate_check_inventory(names, [])
    with pytest.raises(ValueError, match="absent from snapshot inventory"):
        validate_check_inventory(names[:-1], [])


def test_train_only_nzs_acquires_exclusions_without_benchmark_images():
    args = setup_data.parser().parse_args(["--config", str(ROOT / "configs/data_nzs.yaml"), "--splits", "train"])
    _, selected = setup_data.resolve(args)
    names = [item.name for item in acquisition_items(selected)]
    assert "geometry_splits" in names and "scannet" in names
    assert names.index("geometry_splits") < names.index("scannet")
    assert "marigold_depth" not in names and "marigold_normals" not in names
    assert len(names) == len(set(names))


def test_additive_dependency_cycles_fail(monkeypatch):
    monkeypatch.setitem(BY_NAME, "scannet", replace(BY_NAME["scannet"], dependencies=("geometry_splits",)))
    monkeypatch.setitem(BY_NAME, "geometry_splits", replace(BY_NAME["geometry_splits"], dependencies=("scannet",)))
    with pytest.raises(ValueError, match="Cyclic"):
        acquisition_items([BY_NAME["scannet"]])


def test_exclusions_validate_lists_and_use_physical_location(tmp_path):
    for filename, count in zip(SCANNET_SPLITS, (800, 300)):
        path = tmp_path / filename
        path.parent.mkdir(parents=True)
        path.write_text("".join(f"scene0123_01/color/{i}.jpg scene0123_01/depth/{i}.png\n" for i in range(count)))
    assert scannet_exclusions(tmp_path) == {"scene0123"}
    (tmp_path / SCANNET_SPLITS[0]).write_text("scene0123_00/a.jpg a.png\n")
    with pytest.raises(ValueError, match="800 unique"):
        scannet_exclusions(tmp_path)


@pytest.mark.parametrize("override", [None, "custom renderer/ReplicaRenderer", "/opt/replica/renderer"])
def test_renderer_follows_data_root_from_any_cwd(tmp_path, monkeypatch, override):
    arguments = ["--config", str(ROOT / "configs/data_zs.yaml"), "--data-root", str(tmp_path / "my data")]
    if override:
        arguments += ["--set", "replica.ptex-renderer=" + override]
    expected = Path(override or "tools/replica/renderer/ReplicaSDK/ReplicaRenderer")
    expected = (tmp_path / "my data" / expected).resolve()
    monkeypatch.chdir(tmp_path)
    cfg, _ = setup_data.resolve(setup_data.parser().parse_args(arguments))
    assert cfg["preparation"]["replica"]["ptex-renderer"] == str(expected)


@pytest.mark.parametrize("replace_file", [False, True])
def test_same_size_archive_change_ignores_stale_sidecar(tmp_path, replace_file):
    path = tmp_path / "data.tar"
    path.write_bytes(b"original")
    expected = indexed_tar.archive_fingerprint(path)
    Path(str(path) + ".sha256").write_text(expected["sha256"] + "  data.tar\n")
    original_stat = path.stat()
    if replace_file:
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"modified")
        replacement.replace(path)
    else:
        path.write_bytes(b"modified")
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    with pytest.raises(ValueError, match="SHA-256 changed"):
        indexed_tar.validate_archive_fingerprint(path, expected)


def test_unchanged_file_hash_cache_is_process_local_and_movable(tmp_path, monkeypatch):
    source = tmp_path / "data.tar"
    source.write_bytes(b"data")
    real_now = indexed_tar.time.time_ns()
    monkeypatch.setattr(indexed_tar.time, "time_ns", lambda: real_now + 3_000_000_000)
    original = indexed_tar.sha256_file
    calls = []
    def counted(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(indexed_tar, "sha256_file", counted)
    expected = indexed_tar.archive_fingerprint(source)
    indexed_tar.validate_archive_fingerprint(source, expected)
    assert len(calls) == 1
    moved = tmp_path / "moved.tar"
    source.rename(moved)
    indexed_tar.validate_archive_fingerprint(moved, expected)
    assert len(calls) == 2


def write_tar(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        data = json.dumps({"image": "test.txt", "text": "sample", "task": "text_to_image"}).encode() + b"\n"
        for name, content in (("metadata.jsonl", data), ("test.txt", b"sample")):
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def test_stage_reuse_rejects_changed_settings_or_corruption(tmp_path):
    output = tmp_path / "train/data.tar"
    calls = []
    def action(force):
        calls.append(force)
        write_tar(output)
    with stage_lock(tmp_path, "sample", "train"):
        run_stage(tmp_path, "sample_train_convert", {"seed": 42}, lambda: [output], action)
        run_stage(tmp_path, "sample_train_convert", {"seed": 42}, lambda: [output], action)
        assert calls == [False]
        with pytest.raises(ValueError, match="settings, source bytes or converter changed"):
            run_stage(tmp_path, "sample_train_convert", {"seed": 43}, lambda: [output], action)
        with output.open("r+b") as handle:
            handle.write(b"x")
        with pytest.raises(ValueError, match="corrupt"):
            run_stage(tmp_path, "sample_train_convert", {"seed": 42}, lambda: [output], action)


def test_existing_outputs_require_provenance(tmp_path):
    output = tmp_path / "train/data.tar"
    write_tar(output)
    with pytest.raises(ValueError, match="no verified preparation receipt"):
        run_stage(tmp_path, "sample_train_convert", {}, lambda: [output], lambda _: pytest.fail("Must not rebuild"))


def test_filter_failure_resumes_after_completed_native_conversion(tmp_path, monkeypatch):
    raw, prepared = tmp_path / "raw", tmp_path / "prepared"
    source = raw / "Shared_geo/hypersim"
    source.mkdir(parents=True)
    (source / "official_fixture").write_bytes(b"source")
    args = SimpleNamespace(dry_run=False, geometry_policy="base-clean", overwrite=False,
                           raw_root=raw, seed=42, max_samples=1)
    calls = []
    def fake_run(script, arguments, dry_run):
        calls.append((script.name, list(arguments)))
        if script.name == "materialize_geometry_source.py":
            if sum(name == script.name for name, _ in calls) == 1:
                raise RuntimeError("injected filter failure")
            output = prepared
            (output / "manifests/hypersim_filtering.json").write_text("{}")
        else:
            output = prepared / "work/native_geometry"
        for task in ("Depth", "Normals"):
            write_tar(output / "train" / task / "hypersim.tar")
    monkeypatch.setattr(prepare_data, "run_python", fake_run)
    job = ("hypersim", "train", [str(ROOT / "dataset_prep/hypersim/prepare_hypersim_depth_normals.py")])
    with pytest.raises(RuntimeError, match="injected"):
        prepare_data.execute_preparation(job, prepared, args)
    prepare_data.execute_preparation(job, prepared, args)
    assert [name for name, _ in calls].count("prepare_hypersim_depth_normals.py") == 1
    assert "--overwrite" in calls[-1][1]
    assert not list((prepared / "manifests/preparation_stages").glob("*.pending.json"))
    prepare_data.execute_preparation(job, prepared, args)
    assert len(calls) == 3


def test_missing_index_is_rebuilt_without_conversion(tmp_path, monkeypatch):
    archive = tmp_path / "train/ImgGeneration/pd12m_250000.tar"
    write_tar(archive)
    args = SimpleNamespace(splits="train", no_build_indexes=False, overwrite=False, samples_per_epoch=10, seed=42)
    actual = prepare_data.run_python
    calls = []
    def track(script, arguments, dry_run):
        calls.append(script.name)
        return actual(script, arguments, dry_run)
    monkeypatch.setattr(prepare_data, "run_python", track)
    prepare_data.finalize_outputs(tmp_path, args, [])
    prepare_data.finalize_outputs(tmp_path, args, [])
    assert calls == ["build_prepared_tar_index.py"]
    Path(str(archive) + ".ovb-index.sqlite").unlink()
    prepare_data.finalize_outputs(tmp_path, args, [])
    assert calls == ["build_prepared_tar_index.py", "build_prepared_tar_index.py"]


def test_source_identity_follows_staged_symlink(tmp_path):
    source = tmp_path / "official"
    source.mkdir()
    (source / "sample").write_bytes(b"raw")
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "source").symlink_to(source, target_is_directory=True)
    before = input_identity([("source", staged)])
    (source / "sample").write_bytes(b"new")
    assert before["digest"] != input_identity([("source", staged)])["digest"]


@pytest.mark.parametrize("registered", [True, False])
def test_replica_ptex_registration_requires_matching_depth(tmp_path, monkeypatch, registered):
    import numpy as np
    from PIL import Image
    from scipy.ndimage import binary_erosion  # Load SciPy before replacing subprocess.run.
    from dataset_prep.replica.prepare_replica_depth_normals import apply_ptex_rgb
    scene = tmp_path / "scene"
    (scene / "textures").mkdir(parents=True)
    (scene / "mesh.ply").write_text("mesh fixture")
    sample = SimpleNamespace(scene=SimpleNamespace(root=scene), sample_id="test",
        yaw_degrees=0, pitch_degrees=0, camera_position=(0, 0, 0),
        valid_depth=np.ones((512, 512), bool), depth=np.ones((512, 512), np.float32))
    def renderer(arguments, check):
        output = Path(arguments[-1])
        Image.new("RGB", (512, 512), (12, 34, 56)).save(output / "test.ppm")
        depth = np.full((512, 512), 1 if registered else 2, dtype="<f4")
        depth.tofile(output / "test.f32")
    monkeypatch.setattr(subprocess, "run", renderer)
    args = SimpleNamespace(ptex_renderer=tmp_path / "renderer", width=512, height=512,
                           hfov=90., camera_world_up_axis="z", min_depth=.001)
    if registered:
        apply_ptex_rgb([sample], args)
        assert sample.rgb.getpixel((0, 0)) == (12, 34, 56)
    else:
        with pytest.raises(ValueError, match="registration failed"):
            apply_ptex_rgb([sample], args)


def test_matterport_strict_accepts_official_half_resolution_normals():
    import numpy as np
    from PIL import Image
    from dataset_prep.matterport3d.prepare_matterport3d_depth_normals import align_official_modalities
    depth = np.arange(48, dtype=np.uint16).reshape(6, 8)
    rgb, resized = align_official_modalities(Image.new("RGB", (8, 6)), depth, (3, 4), True)
    assert rgb.size == (4, 3) and resized.shape == (3, 4)
    assert np.isin(resized, depth).all()


@pytest.mark.parametrize("rgb_size,normal_shape", [((8, 6), (3, 3)), ((9, 6), (3, 4))])
def test_matterport_strict_rejects_unexplained_registration(rgb_size, normal_shape):
    import numpy as np
    from PIL import Image
    from dataset_prep.matterport3d.prepare_matterport3d_depth_normals import align_official_modalities
    with pytest.raises(ValueError, match="dimensions"):
        align_official_modalities(Image.new("RGB", rgb_size), np.ones((6, 8), np.uint16), normal_shape, True)
