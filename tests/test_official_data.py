"""Official-source recipes must preserve task contracts and reject bad data."""
import argparse
from dataclasses import replace
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.release_data import BY_NAME, acquisition_items, staged_source_status
from common.official_sources import missing_manual_assets
from scripts import setup_data as setup, prepare_data
from scripts.dataset_prep.prepare_geometry_training import add_bytes, png, npy
from scripts.dataset_prep.materialize_geometry_source import materialize, source_rows
from scripts.dataset_prep.materialize_geometry_source import crop_metadata
from scripts.dataset_prep.finalize_training_recipe import combine, ZS_SOURCES, NZS_SOURCES
import tarfile


def test_export_crop_updates_intrinsics_without_changing_source_metadata():
    row = {"metadata": {"intrinsics": [[100., 0., 40.], [0., 100., 30.], [0., 0., 1.]]}}
    metadata = crop_metadata(row, (8, 6, 72, 54))
    assert metadata["intrinsics"][0][2] == 32.
    assert metadata["intrinsics"][1][2] == 24.
    assert row["metadata"]["intrinsics"][0][2] == 40.
    assert metadata["source_intrinsics"] == row["metadata"]["intrinsics"]


def test_depth_benchmark_smoke_cannot_masquerade_as_full_archive(tmp_path):
    from scripts.dataset_prep.prepare_geometry_benchmarks import prepare
    from common.official_sources import DEPTH_FILES
    raw, output = tmp_path / "raw", tmp_path / "prepared"
    split = raw / "data_split" / DEPTH_FILES["nyuv2"][1]
    split.parent.mkdir(parents=True)
    split.write_text("".join(f"rgb/{i}.png depth/{i}.png\n" for i in range(654)))
    source = raw / "archives/nyuv2" / DEPTH_FILES["nyuv2"][0]
    source.parent.mkdir(parents=True)
    with tarfile.open(source, "w") as archive:
        for i in range(2):
            for field in ("rgb", "depth"):
                add_bytes(archive, f"{field}/{i}.png", b"test member")
    (raw / "sources.json").write_text(json.dumps({"settings": {"datasets": "nyuv2"}}))
    prepare("marigold_depth", raw, output, limit=2)
    with pytest.raises(ValueError, match="Missing author depth benchmark members"):
        prepare("marigold_depth", raw, output, overwrite=True)


@pytest.mark.parametrize("dataset,task", [("pico_banana", "ImgEdit"), ("pd12m", "ImgGeneration")])
def test_standalone_creative_preparation_does_not_enter_segmentation_pool(tmp_path, dataset, task):
    args = SimpleNamespace(workers=1, seed=42, overwrite=False)
    command = prepare_data.pool_train_command(dataset, tmp_path / "raw", tmp_path / "prepared", args, {})
    assert Path(command[0]).name == "prepare_creative_training.py"
    assert command[command.index("--dataset")+1] == dataset


@pytest.mark.parametrize("pool", ["zs", "nzs"])
def test_complete_recipes_share_model_data_and_have_split_specific_support(pool):
    cfg, selected = setup.resolve(setup.parser().parse_args(["--config", str(ROOT / f"configs/data_{pool}.yaml")]))
    expected = ZS_SOURCES if pool == "zs" else NZS_SOURCES
    assert set(cfg["selections"]["train"]) & prepare_data.GEOMETRY_TRAIN == expected
    assert cfg["training_pool"] == pool and cfg["geometry_policy"] == "base-clean"
    assert {"marigold_depth", "marigold_normals", "dsine", "diode_normals"} <= set(cfg["selections"]["eval"])
    commands = setup.preparation_commands(cfg)
    assert len(commands) == 2
    assert "--splits" in commands[0] and "train" in commands[0]
    assert "eval" in commands[1]
    assert "ovb26" not in json.dumps(commands).lower()


def test_all_expands_and_rejects_unsupported_pairs_before_output(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump(dict(selections={"train": ["all"]})))
    with pytest.raises(ValueError, match="Unsupported"):
        setup.resolve(setup.parser().parse_args(["--config", str(recipe)]))


def test_missing_scripted_manual_input_allows_download_but_stops_preparation(tmp_path, monkeypatch, capsys):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump(dict(datasets=["cityscapes", "genai_bench"], splits=["eval"], data_root=str(tmp_path / "data"))))
    commands = []
    monkeypatch.setattr(setup, "run", commands.append)
    assert setup.main(["--config", str(recipe)]) == 2
    text = capsys.readouterr().err
    assert "gtFine_trainvaltest.zip" in text and str(tmp_path / "data/raw/Semseg/cityscapes") in text
    assert len(commands) == 1 and commands[0][1].endswith("download_data.py")
    assert not (tmp_path / "data").exists()


def test_manual_input_zips_are_sufficient_for_acquisition(tmp_path):
    folder = tmp_path / "Semseg/cityscapes"
    folder.mkdir(parents=True)
    for name in ("gtFine_trainvaltest.zip", "leftImg8bit_trainvaltest.zip"):
        (folder / name).write_bytes(b"official input fixture")
    assert missing_manual_assets(tmp_path, [BY_NAME["cityscapes"]]) == []
    # Official inputs are not yet prepared/staged output.
    assert not staged_source_status(tmp_path, BY_NAME["cityscapes"])[0]


def test_official_geometry_sources_are_automatic():
    for name in ("hypersim", "taskonomy", "replica", "diode", "vkitti2", "marigold_depth", "marigold_normals"):
        assert BY_NAME[name].access == "automatic"
    assert BY_NAME["matterport3d"].access == BY_NAME["scannet"].access == "manual"


def fixture_archives(root, bad_normal=False, misalign=False):
    for task in ("Depth", "Normals"):
        path = root / task / "nyuv2.tar"
        path.parent.mkdir(parents=True)
        rows = []
        with tarfile.open(path, "w") as archive:
            for number in range(2):
                key = str(number)
                rgb = np.full((64, 80, 3), 100, np.uint8)
                if misalign and task == "Normals":
                    rgb += 1
                target = np.full_like(rgb, (128, 128, 255))
                if bad_normal and task == "Normals" and number == 1:
                    target[:] = 128
                mask = np.full((64, 80), 255, np.uint8)
                row = dict(id=key, conditioning_image=f"{key}/rgb.png", image=f"{key}/target.png",
                    valid_mask=f"{key}/mask.png", task="depth_estimation" if task == "Depth" else "surface_normals",
                    depth_encoding="vision_banana_rgb", normal_encoding="vision_banana_x_left_red", metadata={})
                for field, array in (("conditioning_image", rgb), ("image", target), ("valid_mask", mask)):
                    add_bytes(archive, row[field], png(array))
                rows.append(row)
            add_bytes(archive, "metadata.jsonl", ("\n".join(json.dumps(r) for r in rows)+"\n").encode())


def test_nzs_filter_preserves_depth_when_only_normal_gt_is_invalid(tmp_path):
    source, output = tmp_path / "native", tmp_path / "prepared/train"
    fixture_archives(source, bad_normal=True)
    materialize("nyuv2", source, output)
    counts = []
    for task in ("Depth", "Normals"):
        with tarfile.open(output / task / "nyuv2.tar") as archive:
            rows = source_rows(archive)
            counts.append(len(rows))
            for row in rows.values():
                assert row["loss_mask"] and "base-clean" in row["metadata"]["preparation_policy"]
                assert "training_crop_box" not in row
    assert counts == [2, 1]
    with pytest.raises(FileExistsError):
        materialize("nyuv2", source, output)


def test_zs_filter_rejects_unregistered_pairs_without_publishing(tmp_path):
    source = tmp_path / "native"
    fixture_archives(source, misalign=True)
    for task in ("Depth", "Normals"):
        (source / task / "nyuv2.tar").rename(source / task / "hypersim.tar")
    with pytest.raises(ValueError, match="registration"):
        materialize("hypersim", source, tmp_path / "prepared/train")
    assert not list((tmp_path / "prepared/train").glob("*/*.tar"))


@pytest.mark.parametrize("pool", ["zs", "nzs"])
def test_final_mix_preserves_masses_and_equal_geometry_sources(tmp_path, pool, monkeypatch):
    monkeypatch.setattr("scripts.dataset_prep.finalize_training_recipe.ensure_diode_convention_verified", lambda: None)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    others = dict(datasets=[dict(name="editing", path="edit.tar", weight=.7)])
    name = "zeroshot_pool.json" if pool == "zs" else "nonzeroshot_pool.json"
    (manifests / name).write_text(json.dumps(others))
    sources = ZS_SOURCES if pool == "zs" else NZS_SOURCES
    geometry = []
    for task in ("depth_estimation", "surface_normals"):
        for source in sorted(sources):
            archive = tmp_path / f"{task}_{source}.tar"
            archive.write_bytes(b"fixture archive")
            index = tmp_path / f"{task}_{source}.sqlite"
            with sqlite3.connect(index) as c:
                from common.indexed_tar import archive_fingerprint, INDEX_SCHEMA_VERSION
                c.execute("create table metadata(key text, value text)")
                c.executemany("insert into metadata values (?, ?)", [
                    ("schema_version", str(INDEX_SCHEMA_VERSION)),
                    ("archive_fingerprint", json.dumps(archive_fingerprint(archive)))])
                c.execute("create table members(name text, offset integer, size integer)")
                c.execute("insert into members values ('mask.png', 0, 1)")
                c.execute("create table rows(row_number integer, payload text)")
                c.execute("insert into rows values (0, ?)", (json.dumps(dict(loss_mask="mask.png")),))
            geometry.append(dict(name=archive.stem, path=str(archive), index_path=str(index),
                                 task=task, source_dataset=source, weight=1.))
    (manifests / "geometry_training_specs.json").write_text(json.dumps(dict(datasets=geometry)))
    path = combine(tmp_path, pool)
    data = json.loads(path.read_text())
    assert data["data_shared_by_model_sizes"] == ["4B", "9B"]
    assert data["datasets"][0] == others["datasets"][0]
    if pool == "nzs":
        target = next(e for e in geometry if e["source_dataset"] == "diode" and e["task"] == "surface_normals")
        with sqlite3.connect(target["index_path"]) as c:
            original = c.execute("select payload from rows where row_number=0").fetchone()[0]
            row = json.loads(original)
            row["metadata"] = {"geometry_version": "diode_identity_candidate_unverified_v1"}
            c.execute("update rows set payload=? where row_number=0", (json.dumps(row),))
        with pytest.raises(ValueError, match="DIODE normal convention remains unverified"):
            combine(tmp_path, pool)
        with sqlite3.connect(target["index_path"]) as c:
            c.execute("update rows set payload=? where row_number=0", (original,))
    for task, mass in (("depth_estimation", .1), ("surface_normals", .2)):
        weights = [r["weight"] for r in data["datasets"] if r.get("task") == task]
        assert sum(weights) == pytest.approx(mass) and len(set(weights)) == 1
    original_bytes = archive.read_bytes()
    archive.write_bytes(original_bytes + b"changed")
    with pytest.raises(ValueError, match="size changed"):
        combine(tmp_path, pool)
    archive.write_bytes(original_bytes)
    data = json.loads((manifests / "geometry_training_specs.json").read_text())
    data["datasets"].pop()
    (manifests / "geometry_training_specs.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Incomplete"):
        combine(tmp_path, pool)


def test_nzs_source_gate_rejects_old_cache_without_overwriting_manifest(tmp_path, monkeypatch):
    from dataclasses import replace
    from scripts.dataset_prep import prepare_geometry_training as geometry
    monkeypatch.setattr(geometry, "DIODE_NORMAL_TO_CANONICAL", replace(
        geometry.DIODE_NORMAL_TO_CANONICAL, geometry_version="diode_identity_candidate_unverified_v1"))
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    previous = manifests / "nzs_training.json"
    previous.write_text('{"historical": true}\n')
    with pytest.raises(ValueError, match="Older cached geometry metadata"):
        combine(tmp_path, "nzs")
    assert previous.read_text() == '{"historical": true}\n'
    assert not (manifests / "nzs_training.json.partial").exists()


def test_index_failure_does_not_publish_preparation_completion(tmp_path, monkeypatch):
    root = tmp_path / "prepared"
    (root / "train/Depth").mkdir(parents=True)
    (root / "train/Depth/test.tar").write_bytes(b"fixture")
    args = SimpleNamespace(splits="train", no_build_indexes=False, overwrite=False,
                           samples_per_epoch=10, seed=42)
    monkeypatch.setattr(prepare_data, "run_python", lambda *a: (_ for _ in ()).throw(RuntimeError("index failure")))
    with pytest.raises(RuntimeError, match="index failure"):
        prepare_data.finalize_outputs(root, args, [dict(dataset="test")])
    assert not (root / "manifests/preparation_manifest.json").exists()


@pytest.mark.parametrize("entry", ["download_data.py", "prepare_data.py", "setup_data.py"])
def test_three_entrypoints_accept_same_recipe_outside_checkout(tmp_path, entry):
    command = [sys.executable, str(ROOT / "scripts" / entry), "--config", str(ROOT / "configs/data_normals_smoke.yaml"),
               "--data-root", str(tmp_path / "data"), "--dry-run"]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "data").exists()


def test_small_nyu_conversion_uses_official_train_indices(tmp_path):
    import h5py
    from scipy.io import savemat
    from scripts.dataset_prep.prepare_geometry_training import prepare
    raw = tmp_path / "raw"
    raw.mkdir()
    savemat(raw / "splits.mat", dict(trainNdxs=np.array([[1]]), testNdxs=np.array([[2]])))
    with h5py.File(raw / "nyu_depth_v2_labeled.mat", "w") as h:
        h["images"] = np.full((2, 3, 80, 64), 90, np.uint8)
        h["depths"] = np.ones((2, 80, 64), np.float32)*2
        sparse = np.ones((2, 80, 64), np.float32)*2
        sparse[0, 0, 0] = 0
        h["rawDepths"] = sparse
    output = tmp_path / "native"
    import zipfile
    from scripts.dataset_prep.prepare_geometry_training import png
    with zipfile.ZipFile(raw / "dsine_eval.zip", "w") as labels:
        labels.writestr("dsine_eval/nyuv2/train/000000_img.png", png(np.full((64, 80, 3), 90, np.uint8)))
        labels.writestr("dsine_eval/nyuv2/train/000000_normal.png", png(np.full((64, 80, 3), [127, 127, 255], np.uint8)))
    prepare("nyuv2", raw, output, None, None)
    for task in ("Depth", "Normals"):
        with tarfile.open(output / task / "nyuv2.tar") as archive:
            rows = source_rows(archive)
            assert set(rows) == {"nyu_0001"}
            assert rows["nyu_0001"]["loss_mask"]
            if task == "Normals":
                row = rows["nyu_0001"]
                assert row["metadata"]["normal_ground_truth"] == "GeoNet labels distributed by DSINE"
                mask = np.asarray(Image.open(archive.extractfile(row["loss_mask"])))
                assert mask[0, 0] == 255  # Supplied normal validity is independent of sparse depth.


def test_nyu_missing_author_labels_fails_without_depth_fallback(tmp_path):
    from scipy.io import savemat
    from scripts.dataset_prep.prepare_geometry_training import nyu_samples
    savemat(tmp_path / "splits.mat", dict(trainNdxs=np.array([[1]]), testNdxs=np.array([[2]])))
    with pytest.raises(ValueError, match="Depth-derived normals are not substituted"):
        list(nyu_samples(tmp_path, 1))


def test_nyu_acquisition_includes_author_labels_and_pico_uses_packed_route():
    from common.release_data import acquisition_items
    assert "dsine" in {item.name for item in acquisition_items([BY_NAME["nyuv2"]])}
    assert all(step.script.endswith("acquire_geometry_sources.py") for step in BY_NAME["pico_banana"].steps)
