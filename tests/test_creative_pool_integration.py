"""Creative pool preparation, packed identity and failed rebuild recovery."""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import pico_packed, creative_publication
from common.indexed_tar import load_index, read_reference_bytes
from scripts.dataset_prep import build_training_pool as pool
from scripts.dataset_prep.prepare_creative_training import prepare, archive_relative
from scripts import download_data, setup_data
from common.release_data import BY_NAME


def source_data(tmp_path):
    raw, work = tmp_path / "raw", tmp_path / "work"
    work.mkdir()
    rows = [dict(open_image_input_url=f"https://example.org/source{i}.jpg",
                 output_image=f"images/target{i}.png", text=f"Edit image {i}") for i in range(3)]
    def fetch(url, path):
        if url == pico_packed.METADATA_URL:
            path.write_text("\n".join(json.dumps(row) for row in rows))
        else:
            Image.new("RGB", (32, 40), "red" if "source" in url else "blue").save(path, format="PNG")
    pico_packed.acquire(raw / "ImgEdit/pico_banana", {}, work, fetch)
    generation = raw / "ImgGeneration/pd12m/shards"
    generation.mkdir(parents=True)
    buffer = io.BytesIO()
    Image.new("RGB", (32, 40), "green").save(buffer, format="JPEG")
    with tarfile.open(generation / "000000.tar", "w") as archive:
        for index in range(3):
            for suffix, payload in (("jpg", buffer.getvalue()), ("txt", b"A green image"), ("json", b"{}")):
                pico_packed.add(archive, f"{index}.{suffix}", payload)
    return raw, work


def build(monkeypatch, raw, root, name="creative", sources="pico_banana,identity,pd12m", overwrite=False):
    argv = ["build", "--pool", name, "--raw-root", str(raw), "--root", str(root), "--limit", "3", "--workers", "1"]
    if name not in pool.POOLS:
        argv += ["--sources", sources]
    if overwrite:
        argv.append("--overwrite")
    monkeypatch.setattr(sys, "argv", argv)
    return pool.main()


def payloads(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file() and p.suffix != ".lock"}


@pytest.mark.parametrize("name", ["zeroshot", "nonzeroshot"])
def test_both_pool_paths_prepare_packed_creative_members(tmp_path, monkeypatch, name):
    raw, _ = source_data(tmp_path)
    monkeypatch.setitem(pool.POOLS, name, ([], [], ["pico_banana", "identity", "pd12m"]))
    root = tmp_path / "prepared"
    assert build(monkeypatch, raw, root, name) == 0
    for dataset in ("pico_banana", "identity", "pd12m"):
        archive = root / archive_relative(dataset)
        rows, _ = load_index(Path(str(archive) + ".ovb-index.sqlite"), archive)
        assert len(rows) == 3
        if dataset == "identity":
            for row in rows:
                assert row["task"] == "identity"
                assert read_reference_bytes(row["image"]) == read_reference_bytes(row["conditioning_image"])
                image = Image.open(io.BytesIO(read_reference_bytes(row["image"])))
                assert image.getpixel((0, 0)) == (255, 0, 0)  # Original, never blue edited target.
    before = payloads(root)
    assert build(monkeypatch, raw, root, name) == 0
    assert payloads(root) == before
    assert build(monkeypatch, raw, root, name, overwrite=True) == 0
    for dataset in ("pico_banana", "identity", "pd12m"):
        archive = root / archive_relative(dataset)
        assert len(load_index(Path(str(archive) + ".ovb-index.sqlite"), archive)[0]) == 3
    assert not list(root.glob(".creative-*"))


def test_identity_reuses_existing_pico_without_raw_rebuild(tmp_path, monkeypatch):
    raw, _ = source_data(tmp_path)
    root = tmp_path / "prepared"
    build(monkeypatch, raw, root, sources="pico_banana")
    pico = root / archive_relative("pico_banana")
    before = pico.stat().st_mtime_ns
    build(monkeypatch, tmp_path / "absent_raw", root, name="identity_only", sources="identity")
    assert pico.stat().st_mtime_ns == before
    assert (root / archive_relative("identity")).is_file()


def test_failed_pool_conversion_preserves_every_creative_artifact_and_manifest(tmp_path, monkeypatch):
    raw, _ = source_data(tmp_path)
    root = tmp_path / "prepared"
    build(monkeypatch, raw, root)
    before = payloads(root)
    original = pool.run
    def fail_generation(command, env, dry_run):
        if "--dataset" in command and command[command.index("--dataset") + 1] == "pd12m":
            raise subprocess.CalledProcessError(1, command)
        return original(command, env, dry_run)
    monkeypatch.setattr(pool, "run", fail_generation)
    with pytest.raises(subprocess.CalledProcessError):
        build(monkeypatch, raw, root, overwrite=True)
    assert payloads(root) == before
    assert not list(root.glob(".creative-*"))


def test_index_failure_preserves_standalone_archive_and_index(tmp_path, monkeypatch):
    raw, work = source_data(tmp_path)
    root = tmp_path / "prepared"
    args = SimpleNamespace(dataset="pico_banana", raw_dir=raw / "ImgEdit/pico_banana", root=root,
                           workers=1, seed=42, limit=3, overwrite=False)
    prepare(args, work)
    before = payloads(root)
    from scripts.dataset_prep import prepare_creative_training as creative
    monkeypatch.setattr(creative, "finish", lambda _: (_ for _ in ()).throw(ValueError("index failure")))
    args.overwrite = True
    with pytest.raises(ValueError, match="index failure"):
        prepare(args, work)
    assert payloads(root) == before


def test_publication_replacement_failure_rolls_back(tmp_path, monkeypatch):
    root, staged = tmp_path / "root", tmp_path / "stage"
    name = Path("train/ImgEdit/example.tar")
    for parent, value in ((root, b"previous"), (staged, b"replacement")):
        for suffix in ("", ".sha256", ".ovb-index.sqlite"):
            path = parent / (str(name) + suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
    before = payloads(root)
    original = creative_publication.os.replace
    failed = False
    def fail_once(source, destination):
        nonlocal failed
        if str(source).endswith(".sha256") and not failed:
            failed = True
            raise OSError("injected replacement failure")
        original(source, destination)
    monkeypatch.setattr(creative_publication.os, "replace", fail_once)
    with pytest.raises(OSError, match="injected"):
        creative_publication.publish(staged, root, [name], True)
    assert payloads(root) == before


def test_missing_gate_allows_public_download_and_skips_dependent(tmp_path, monkeypatch, capsys):
    gate = BY_NAME["matterport3d"]
    public = replace(BY_NAME["vkitti2"], files=(("https://example.org/public.tar", "public.tar"),), dependencies=())
    dependent = replace(BY_NAME["dsine"], dependencies=(gate.name,), steps=(),
                        files=(("https://example.org/dependent.zip", "dependent.zip"),))
    monkeypatch.setitem(BY_NAME, public.name, public)
    monkeypatch.setitem(BY_NAME, dependent.name, dependent)
    calls = []
    def fetch(url, path):
        calls.append(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"downloaded")
    monkeypatch.setattr(download_data, "download_with_resume", fetch)
    monkeypatch.setattr(sys, "argv", ["download", "--datasets", "matterport3d,vkitti2,dsine", "--destination", str(tmp_path)])
    assert download_data.main() == 2
    assert calls == ["https://example.org/public.tar"]
    assert "incomplete" in capsys.readouterr().err
    assert download_data.main() == 2
    assert len(calls) == 1  # Completed public input is reused.


@pytest.mark.parametrize("mode,expected_calls", [(None, 1), ("--download-only", 1), ("--prepare-only", 0), ("--dry-run", 0)])
def test_combined_setup_with_missing_gate_never_prepares(tmp_path, monkeypatch, mode, expected_calls):
    import yaml
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump(dict(data_root=str(tmp_path / "data"),
                                        datasets=["matterport3d", "pd12m"], splits=["train"])))
    calls = []
    def run(command):
        assert command[1].endswith("download_data.py")
        calls.append(command)
        raise subprocess.CalledProcessError(2, command)
    monkeypatch.setattr(setup_data, "run", run)
    assert setup_data.main(["--config", str(recipe)] + ([mode] if mode else [])) == 2
    assert len(calls) == expected_calls
    assert not (tmp_path / "data/prepared").exists()
