"""The combined CLI must never download or prepare past a failed preflight."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import setup_data as setup
from common.release_data import BY_NAME


@pytest.fixture
def recipe(tmp_path, monkeypatch):
    # Exercise manual-stage behavior with a catalog fixture.
    monkeypatch.setitem(BY_NAME, "taskonomy", replace(BY_NAME["taskonomy"], access="manual", steps=()))
    path = tmp_path / "recipe.yaml"
    cfg = dict(setup.DEFAULTS, data_root=str(tmp_path / "data"), datasets=["taskonomy"], preparation={})
    path.write_text(yaml.safe_dump(cfg))
    return path


def stage(recipe, dataset="taskonomy"):
    root = Path(yaml.safe_load(recipe.read_text())["data_root"])
    folder = root / "raw/Shared_geo" / dataset
    folder.mkdir(parents=True)
    if dataset == "taskonomy":
        for modality in ("rgb", "depth_zbuffer", "normal"):
            (folder / f"building_{modality}.tar").write_bytes(b"fixture")
    return root


def change(recipe, **values):
    cfg = yaml.safe_load(recipe.read_text())
    cfg.update(values)
    recipe.write_text(yaml.safe_dump(cfg))


def invocation(recipe, *arguments):
    return ["--config", str(recipe), *arguments]


def test_default_recipe_and_cli_precedence(recipe, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg, selected = setup.resolve(setup.parser().parse_args(invocation(
        recipe, "--data-root", "relative", "--seed", "7", "--set", "taskonomy.samples-per-building=9")))
    assert cfg["data_root"] == str(tmp_path / "relative")
    assert cfg["seed"] == 7
    assert cfg["preparation"] == {"taskonomy": {"samples-per-building": "9"}}
    assert [item.name for item in selected] == ["taskonomy"]
    assert yaml.safe_load(recipe.read_text())["seed"] == 42
    default, _ = setup.resolve(setup.parser().parse_args([]))
    assert default["training_pool"] == "zs"
    assert default["max_samples"] is None


def test_work_directory_reaches_both_commands_without_creating_data(recipe, tmp_path):
    cfg, _ = setup.resolve(setup.parser().parse_args(invocation(recipe, "--work-dir", str(tmp_path))))
    for command in setup.commands(cfg):
        assert command[command.index("--work-dir") + 1] == str(tmp_path)
    assert not (tmp_path / "data").exists()
    with pytest.raises(ValueError, match="already exist"):
        setup.resolve(setup.parser().parse_args(invocation(recipe, "--work-dir", str(tmp_path / "missing"))))


def test_unverified_nzs_refuses_before_acquisition(monkeypatch, capsys):
    from dataclasses import replace
    from scripts.dataset_prep import prepare_geometry_training as geometry
    monkeypatch.setattr(geometry, "DIODE_NORMAL_TO_CANONICAL", replace(
        geometry.DIODE_NORMAL_TO_CANONICAL, geometry_version="diode_identity_candidate_unverified_v1"))
    from common import official_sources
    monkeypatch.setattr(official_sources, "missing_manual_assets", lambda *args: [])
    monkeypatch.setattr(setup, "run", lambda _: pytest.fail("Must reject before acquisition"))
    assert setup.main(["--config", str(ROOT / "configs/data_nzs.yaml")]) == 1
    assert "DIODE normal convention remains unverified" in capsys.readouterr().err


@pytest.mark.parametrize("values", [
    {"download_workers": 0}, {"seed": True}, {"overwrite": "false"},
    {"version": "../escape"}, {"splits": ["test"]}, {"datasets": []},
    {"unknown": 4}, {"preparation": {"taskonomy": {"bad-key": 2}}},
    {"preparation": {"hypersim": {"delete-raw-scenes-after-convert": True}}},
])
def test_invalid_config_is_nonmutating(recipe, monkeypatch, values):
    change(recipe, **values)
    monkeypatch.setattr(setup, "run", lambda _: pytest.fail("Must not run"))
    assert setup.main(invocation(recipe)) == 1
    assert not (recipe.parent / "data").exists()


def test_malformed_yaml(recipe, capsys):
    recipe.write_text("datasets: [\n")
    assert setup.main(invocation(recipe)) == 1
    assert "Invalid YAML" in capsys.readouterr().err


@pytest.mark.parametrize("datasets,splits", [("kitti_eigen", "train"), ("taskonomy", "eval"), ("all", "train")])
def test_unsupported_pairs_stop_before_download(recipe, monkeypatch, datasets, splits):
    monkeypatch.setattr(setup, "run", lambda _: pytest.fail("Must not run"))
    assert setup.main(invocation(recipe, "--datasets", datasets, "--splits", splits)) == 1


def test_all_missing_manual_sources_reported(recipe, monkeypatch, capsys):
    monkeypatch.setitem(BY_NAME, "hypersim", replace(BY_NAME["hypersim"], access="manual", steps=()))
    monkeypatch.setattr(setup, "run", lambda _: pytest.fail("Must not run"))
    assert setup.main(invocation(recipe, "--datasets", "taskonomy,hypersim")) == 2
    output = capsys.readouterr().err
    assert "Missing manual assets: taskonomy" in output
    assert "Missing manual assets: hypersim" in output
    assert "official EPFL" in output or "Official EPFL" in output
    assert str(recipe.parent / "data/raw/Shared_geo/taskonomy") in output
    assert "rerun:" in output
    assert not (recipe.parent / "data").exists()


def test_staged_manual_assets_and_provenance(recipe, monkeypatch):
    root = stage(recipe)
    commands = []
    def fake(command):
        commands.append(command)
        if command[1].endswith("prepare_data.py"):
            prepared = root / "prepared/V1_prepared_seg_geo"
            for split in ("train", "eval"):
                for task in ("Semseg", "Instanceseg", "Refseg", "ImgEdit", "ImgGeneration", "Depth", "Normals"):
                    (prepared / split / task).mkdir(parents=True)
            (prepared / "SHA256SUMS").write_text("existing checksums\n")
    monkeypatch.setattr(setup, "run", fake)
    original = recipe.read_bytes()
    assert setup.main(invocation(recipe, "--overwrite")) == 0
    assert [Path(command[1]).name for command in commands] == ["download_data.py", "prepare_data.py"]
    assert "--overwrite" in commands[1]
    assert recipe.read_bytes() == original
    prepared = root / "prepared/V1_prepared_seg_geo"
    record, = (prepared / "manifests").glob("setup_data-*.json")
    payload = json.loads(record.read_text())
    assert payload["status"] == "preparation_succeeded"
    assert payload["resolved"]["datasets"] == ["taskonomy"]
    assert Path(str(record) + ".sha256").read_text().split()[0] == hashlib.sha256(record.read_bytes()).hexdigest()
    assert (prepared / "SHA256SUMS").read_text() == "existing checksums\n"


def test_public_download_precedes_staging_and_preparation(recipe, monkeypatch):
    public = replace(BY_NAME["taskonomy"], access="automatic", files=(("https://example.invalid/rgb.tar", "rgb.tar"),), expected=("rgb.tar",))
    monkeypatch.setattr(setup, "parse_selection", lambda _: [public])
    root = recipe.parent / "data"
    commands = []
    def fake(command):
        commands.append(command)
        if command[1].endswith("download_data.py"):
            path = root / "raw/Shared_geo/taskonomy/rgb.tar"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"fixture")
    monkeypatch.setattr(setup, "run", fake)
    assert setup.main(invocation(recipe)) == 0
    assert len(commands) == 2


@pytest.mark.parametrize("failed_stage", ["download_data.py", "prepare_data.py"])
def test_failure_never_publishes_completion(recipe, monkeypatch, failed_stage):
    root = stage(recipe)
    commands = []
    def fake(command):
        commands.append(command)
        if command[1].endswith(failed_stage):
            raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(setup, "run", fake)
    assert setup.main(invocation(recipe)) == 1
    assert not list(root.rglob("setup_data-*.json"))
    assert len(commands) == (1 if failed_stage == "download_data.py" else 2)


def test_download_success_with_missing_files_stops(recipe, monkeypatch):
    public = replace(BY_NAME["taskonomy"], access="automatic", files=(("https://example.invalid/rgb.tar", "rgb.tar"),))
    monkeypatch.setattr(setup, "parse_selection", lambda _: [public])
    calls = []
    monkeypatch.setattr(setup, "run", calls.append)
    assert setup.main(invocation(recipe)) == 1
    assert len(calls) == 1


@pytest.mark.parametrize("staged", [False, True])
def test_dry_run_does_not_download_or_publish(recipe, monkeypatch, staged):
    if staged:
        stage(recipe)
    commands = []
    monkeypatch.setattr(setup, "run", commands.append)
    assert setup.main(invocation(recipe, "--dry-run")) == (0 if staged else 2)
    assert len(commands) == int(staged)
    if staged:
        assert commands[0][1].endswith("prepare_data.py") and "--dry-run" in commands[0]
    assert not (recipe.parent / "data/prepared").exists()


def test_replica_guard_not_bypassed(recipe, monkeypatch):
    change(recipe, datasets=["replica"])
    monkeypatch.setattr(setup, "run", lambda _: pytest.fail("Must not run"))
    assert setup.main(invocation(recipe)) == 1


def test_outside_repository_and_existing_prepare_interface(recipe, tmp_path):
    stage(recipe)
    result = subprocess.run([sys.executable, str(ROOT / "scripts/setup_data.py"), *invocation(recipe, "--dry-run")],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "prepare_taskonomy_depth_normals.py" in result.stdout
    assert not (tmp_path / "data/prepared").exists()
    result = subprocess.run([sys.executable, str(ROOT / "scripts/setup_data.py"), "--list"],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0 and "taskonomy" in result.stdout
