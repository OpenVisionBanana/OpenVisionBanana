"""The supported dataset route never consumes temporary project uploads."""
from pathlib import Path
import importlib.util
import sys

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.dataset_source_policy import validate_dataset_source


@pytest.mark.parametrize("source", [
    "ovb26/OVB_Data", "FunAILab/test-data", "ArpitJadon/example", "OVB-testing/depth",
    "https://huggingface.co/datasets/ovb26/OVB_Data/resolve/main/a.tar",
    "https://hf.co/datasets/FunAILab/test-data/resolve/main/a.tar",
    "https://huggingface.co/api/datasets/ovb26/OVB_Data/tree/main",
    "https://huggingface.co/spaces/ArpitJadon/temp/resolve/main/a.tar",
])
def test_test_account_sources_are_refused(source):
    with pytest.raises(ValueError, match="prohibited"):
        validate_dataset_source(source)


@pytest.mark.parametrize("source", ["Spawning/PD12M", "prs-eth/marigold-eval-data",
    "BaiqiL/GenAI-Bench-1600", "https://huggingface.co/datasets/Spawning/PD12M",
    "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/jsonl/sft.jsonl"])
def test_independent_public_and_author_sources_remain_allowed(source):
    assert validate_dataset_source(source) == source


def test_direct_download_rejects_test_account_before_network_or_output(tmp_path, monkeypatch):
    from common import release_data
    monkeypatch.setattr(release_data.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("network"))
    with pytest.raises(ValueError, match="prohibited"):
        release_data.download_with_resume("https://huggingface.co/datasets/ovb26/OVB_Data/resolve/main/a", tmp_path / "out/a")
    assert list(tmp_path.iterdir()) == []


def test_pinned_revision_does_not_bypass_pd12m_source_check():
    from dataset_prep.pd12m.prepare_pd12m import resolve_hub_revision
    with pytest.raises(ValueError, match="prohibited"):
        resolve_hub_revision("ovb26/OVB_Data", "0123456789")


def test_legacy_downloader_is_inert(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/dataset_prep/download_and_prepare_ovb_geometry.py"
    spec = importlib.util.spec_from_file_location("retired_geometry_download", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not module.REPO_ID
    with pytest.raises(ValueError, match="retired"):
        module.remote_inventory()
    with pytest.raises(ValueError, match="retired"):
        module.main()


def test_supported_four_task_entrypoints_have_no_cluster_paths():
    root = Path(__file__).resolve().parents[1]
    paths = ["scripts/setup_data.py", "scripts/download_data.py", "scripts/prepare_data.py",
             "common/node_scratch.py", "common/release_data.py", "common/official_sources.py",
             "scripts/dataset_prep/acquire_geometry_sources.py",
             "scripts/dataset_prep/prepare_geometry_training.py",
             "scripts/dataset_prep/prepare_creative_training.py",
             "scripts/dataset_prep/materialize_geometry_source.py",
             "scripts/dataset_prep/finalize_training_recipe.py",
             "configs/data_zs.yaml", "configs/data_nzs.yaml"]
    for name in paths:
        text = (root / name).read_text()
        for forbidden in ("/hnvme/", "/home/hpc/", "SLURM_", "apptainer exec"):
            assert forbidden not in text, (name, forbidden)
