"""The retired temporary-data route must never acquire assets."""
from pathlib import Path
import sys
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import setup_data as setup


@pytest.mark.parametrize("dry", [False, True])
def test_temporary_package_route_is_rejected_before_network_or_outputs(tmp_path, monkeypatch, dry):
    recipe = tmp_path / "old.yaml"
    recipe.write_text(yaml.safe_dump(dict(route="published", data_root=str(tmp_path / "data"))))
    monkeypatch.setattr(setup, "run", lambda _: pytest.fail("Retired route must not run commands"))
    args = ["--config", str(recipe)] + (["--dry-run"] if dry else [])
    assert setup.main(args) == 1
    assert not (tmp_path / "data").exists()
