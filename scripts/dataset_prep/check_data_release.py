#!/usr/bin/env python3
"""Build and check an isolated release candidate from an explicit inventory."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

REPO = Path(__file__).resolve().parents[2]
CHECK_TESTS = (
    "test_creative_pool_integration.py",
    "test_download_recovery.py",
    "test_setup_failure_contracts.py",
    "test_pico_packed.py",
    "test_restored_geometry_contracts.py",
    "test_dataset_source_policy.py",
    "test_refcocog_packed.py",
    "test_archive_scratch_workflow.py",
    "test_setup_data.py",
    "test_setup_data_published.py",
    "test_official_data.py",
    "test_ibims_marigold_reference.py",
    "test_geometry_release_correctness.py",
    "test_replica_rgb_correction.py",
    "test_depth_ppd_protocol.py",
    "test_surface_normal_target_protocol.py",
    "test_data_release_readiness.py",
    "test_indexed_tar.py",
    "test_synscapes_prep.py",
    "test_cityscapes_staging.py",
)


def validate_check_inventory(included, base_files):
    available = set(included) | set(base_files)
    missing = ["tests/"+name for name in CHECK_TESTS if "tests/"+name not in available]
    if missing:
        raise ValueError("Selected release tests absent from snapshot inventory: " + ", ".join(missing))


def snapshot(destination):
    inventory = json.loads((REPO / "DATA_RELEASE_FILES.json").read_text())
    included = inventory["overlay"] + inventory["additional"]
    excluded = set(inventory.get("exclude", []))
    if len(included) != len(set(included)):
        raise ValueError("Duplicate release inventory entry")
    if excluded & set(included):
        raise ValueError("Release inventory includes an excluded path")
    for name in excluded:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Unsafe release exclusion: {name}")
    if destination.exists():
        raise FileExistsError(f"Choose a new candidate directory: {destination}")
    base_files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=REPO, text=True).splitlines()
    validate_check_inventory(included, set(base_files) - excluded)
    for name in included:
        path = (REPO / name).resolve()
        if not path.is_relative_to(REPO) or not path.is_file():
            raise ValueError(f"Missing or external release dependency: {name}")
    with tempfile.TemporaryDirectory(prefix="ovb-release-base-") as temporary:
        archive = Path(temporary) / "base.tar"
        subprocess.run(["git", "archive", "--format=tar", "--output", str(archive), "HEAD"], cwd=REPO, check=True)
        destination.mkdir(parents=True)
        with tarfile.open(archive) as source:
            members = [member for member in source.getmembers() if member.name not in excluded]
            for member in members:
                target = (destination / member.name).resolve()
                if not target.is_relative_to(destination) or not (member.isfile() or member.isdir()):
                    raise ValueError(f"Unexpected release member: {member.name}")
            source.extractall(destination, members=members)
    for name in included:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)
    # Identify the exact candidate without staging changes in the source repo.
    from hashlib import sha256
    records = {str(p.relative_to(destination)): sha256(p.read_bytes()).hexdigest()
               for p in sorted(destination.rglob("*")) if p.is_file()}
    (destination / "RELEASE_SNAPSHOT.json").write_text(json.dumps(dict(
        base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        artifacts=records), indent=2) + "\n")
    return destination


def check(candidate, python):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTEST_ADDOPTS", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    def run(arguments):
        subprocess.run([python, *map(str, arguments)], cwd=candidate.parent, env=environment, check=True)
    for script in ("download_data.py", "prepare_data.py", "setup_data.py"):
        run([candidate / "scripts" / script, "--list"])
        run([candidate / "scripts" / script, "--config", candidate / "configs/data_smoke.yaml",
             "--data-root", candidate.parent / "dry_run_data", "--dry-run"])
    run(["-m", "pytest", "-p", "no:cacheprovider", "-q", *[candidate / "tests" / name for name in CHECK_TESTS]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--snapshot-only", action="store_true")
    args = parser.parse_args()
    candidate = snapshot(args.destination.expanduser().resolve())
    print(f"Release candidate: {candidate}", flush=True)
    if not args.snapshot_only:
        check(candidate, args.python)
        print("Release candidate checks passed.", flush=True)
