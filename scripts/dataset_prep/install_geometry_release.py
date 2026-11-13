#!/usr/bin/env python3
"""Install downloadable geometry-v6 ZIPs into the portable repository layout.

The release intentionally preserves authoritative TAR payload names.  The
portable launchers use a stable ``depth/<dataset>.tar`` and
``surface_normals/<dataset>.tar`` layout, so this installer verifies every
member against ``PACKAGE_MANIFEST.json`` and publishes it atomically under the
portable name.  It never rewrites an inner TAR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


RELEASE_ID = "geometry_v6_20260917"

# Packages must carry their own checksummed manifest; unmanifested legacy
# archives are rejected rather than given synthesized metadata.
LEGACY_PACKAGE_FILENAMES: dict[str, str] = {}


@dataclass(frozen=True)
class AssetTarget:
    member: str
    relative: str


PACKAGE_TARGETS: dict[str, tuple[AssetTarget, ...]] = {
    "eval_nyu_full": (
        AssetTarget("nyu.tar", "depth/nyu.tar"),
        AssetTarget(
            "nyu_surface_normals_marigold_geonet_exact.tar",
            "surface_normals/nyu.tar",
        ),
        AssetTarget("nyu_surface_normals.tar", "calibration/nyu_surface_normals.tar"),
        AssetTarget("heldout_nyu_train32_v1.jsonl", "calibration/heldout_nyu_train32_v1.jsonl"),
        AssetTarget(
            "heldout_nyu_normals_train32_v1.jsonl",
            "calibration/heldout_nyu_normals_train32_v1.jsonl",
        ),
    ),
    "eval_ibims1_full": (
        AssetTarget("ibims1_depth_official.tar", "depth/ibims1_official.tar"),
        AssetTarget(
            "ibims1_surface_normals_marigold_dsine_exact.tar",
            "surface_normals/ibims1_official.tar",
        ),
    ),
    "eval_diode_full": (
        AssetTarget("diode_depth_max350.geometry_canonical_v2.tar", "depth/diode.tar"),
        AssetTarget(
            "diode_surface_normals_official_float.tar",
            "surface_normals/diode.tar",
        ),
    ),
    "eval_kitti_eigen_full": (
        AssetTarget(
            "kitti_eigen_split_test.marigold_2024-01-09.tar",
            "depth/kitti_eigen.tar",
        ),
    ),
    "eval_eth3d_unchanged": (
        AssetTarget("eth3d_low_res_two_view_depth.tar", "depth/eth3d.tar"),
        AssetTarget(
            "eth3d_low_res_two_view_surface_normals.tar",
            "surface_normals/eth3d.tar",
        ),
    ),
    "eval_kitti_unchanged": (
        AssetTarget("kitti_depth_selection.tar", "depth/kitti.tar"),
        AssetTarget(
            "kitti_depth_selection_surface_normals.tar",
            "surface_normals/kitti.tar",
        ),
    ),
    "eval_nuscenes_unchanged": (
        AssetTarget("nuscenes_trainval_val_cam_front_depth.tar", "depth/nuscenes.tar"),
        AssetTarget(
            "nuscenes_trainval_val_cam_front_surface_normals.tar",
            "surface_normals/nuscenes.tar",
        ),
    ),
    "eval_scannet_full": (
        AssetTarget("scannet_depth.tar", "depth/scannet.tar"),
        AssetTarget(
            "scannet_surface_normals_framenet_exact.tar",
            "surface_normals/scannet.tar",
        ),
    ),
    "eval_vkitti2_full": (
        AssetTarget("vkitti2_clone_camera_0_depth.tar", "depth/vkitti2.tar"),
        AssetTarget(
            "vkitti2_surface_normals_dsine_exact.tar",
            "surface_normals/vkitti2.tar",
        ),
    ),
    "eval_sintel_full": (
        AssetTarget(
            "sintel_surface_normals_marigold_dsine_exact.tar",
            "surface_normals/sintel.tar",
        ),
    ),
    "eval_nuscenes_ablation750": (
        AssetTarget("nuscenes_trainval_val_cam_front_depth.subset.tar", "depth/nuscenes.tar"),
        AssetTarget(
            "nuscenes_trainval_val_cam_front_surface_normals.subset.tar",
            "surface_normals/nuscenes.tar",
        ),
        AssetTarget("nuscenes.jsonl", "manifests/nuscenes.jsonl"),
    ),
    "eval_scannet_ablation600": (
        AssetTarget("scannet_depth.subset.tar", "depth/scannet.tar"),
        AssetTarget(
            "scannet_surface_normals_framenet_exact.subset.tar",
            "surface_normals/scannet.tar",
        ),
        AssetTarget("scannet.jsonl", "manifests/scannet.jsonl"),
    ),
    "eval_vkitti2_ablation500": (
        AssetTarget("vkitti2_clone_camera_0_depth.subset.tar", "depth/vkitti2.tar"),
        AssetTarget(
            "vkitti2_surface_normals_dsine_exact.subset.tar",
            "surface_normals/vkitti2.tar",
        ),
        AssetTarget("vkitti2.jsonl", "manifests/vkitti2.jsonl"),
    ),
}


TRAINING_PACKAGE_SOURCES = {
    "training_synscapes_geometry": "synscapes",
    "training_replica_geometry": "replica",
    "training_matterport3d_geometry": "matterport3d",
    "training_hypersim_geometry": "hypersim",
    "training_taskonomy_geometry": "taskonomy",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_zip_member(archive: zipfile.ZipFile, member: str) -> str:
    digest = hashlib.sha256()
    with archive.open(member) as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(archive: zipfile.ZipFile) -> dict:
    try:
        manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
    except KeyError as error:
        raise ValueError(f"{archive.filename}: missing PACKAGE_MANIFEST.json") from error
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{archive.filename}: unsupported package schema")
    if manifest.get("release_id") != RELEASE_ID:
        raise ValueError(
            f"{archive.filename}: expected release {RELEASE_ID}, got {manifest.get('release_id')!r}"
        )
    return manifest


def load_or_synthesize_manifest(path: Path, archive: zipfile.ZipFile) -> dict:
    """Read a geometry-v6 package manifest; unmanifested legacy ZIPs are rejected."""
    if "PACKAGE_MANIFEST.json" in archive.namelist():
        return load_manifest(archive)
    package_id = LEGACY_PACKAGE_FILENAMES.get(path.name)
    if package_id is None:
        raise ValueError(f"{path}: missing PACKAGE_MANIFEST.json")
    return {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "package_id": package_id,
        "integrity_policy": "release_manifest_plus_zip_crc",
        "assets": [
            {"name": row.filename, "bytes": row.file_size, "sha256": None}
            for row in archive.infolist()
            if not row.is_dir()
        ],
    }


def training_targets(manifest: dict) -> tuple[AssetTarget, ...]:
    package_id = manifest["package_id"]
    source = TRAINING_PACKAGE_SOURCES[package_id]
    targets: list[AssetTarget] = []
    for asset in manifest["assets"]:
        member = asset["name"]
        base = Path(member).name
        if member.endswith(".tar"):
            task = "surface_normals" if "surface_normals" in base else "depth"
            targets.append(AssetTarget(member, f"training/{source}_{task}.tar"))
        elif member.endswith(".tar.ovb-index.sqlite"):
            task = "surface_normals" if "surface_normals" in base else "depth"
            targets.append(
                AssetTarget(member, f"training/{source}_{task}.tar.ovb-index.sqlite")
            )
        elif member.startswith("validation/"):
            member_path = PurePosixPath(member)
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or "\\" in member
                or member_path.parts[:1] != ("validation",)
            ):
                raise ValueError(
                    f"Unsafe validation asset path in {package_id}: {member!r}"
                )
            targets.append(
                AssetTarget(member, str(PurePosixPath("training") / member_path))
            )
    relative_targets = [target.relative for target in targets]
    if len(relative_targets) != len(set(relative_targets)):
        raise ValueError(f"Package {package_id} maps multiple assets to one destination")
    return tuple(targets)


def target_destination(root: Path, target: AssetTarget) -> Path:
    destination = root / target.relative
    resolved_root = root.resolve()
    resolved_destination = destination.resolve(strict=False)
    if not resolved_destination.is_relative_to(resolved_root):
        raise ValueError(
            f"Refusing asset destination outside profile root: "
            f"{target.relative!r} -> {resolved_destination}"
        )
    return destination


def copy_verified_member(
    archive: zipfile.ZipFile,
    member: str,
    destination: Path,
    expected: dict,
    *,
    force: bool,
    verify_only: bool,
) -> dict:
    expected_size = int(expected["bytes"])
    expected_sha = expected.get("sha256")
    expected_sha = str(expected_sha) if expected_sha else None
    destination_is_file = destination.is_file()
    destination_exists = destination.exists() or destination.is_symlink()
    if destination_is_file:
        actual_size = destination.stat().st_size
        if actual_size == expected_size:
            actual = sha256_file(destination)
            reference_sha = expected_sha or sha256_zip_member(archive, member)
            if actual == reference_sha:
                return {
                    "path": str(destination),
                    "bytes": expected_size,
                    "sha256": actual,
                    "action": "verified",
                }
            if verify_only:
                raise ValueError(
                    f"Verification failed for existing asset: {destination}; "
                    f"sha256={actual}/{reference_sha}"
                )
            if not force:
                raise ValueError(
                    f"Refusing to replace mismatched existing asset without --force: {destination}"
                )
        elif verify_only:
            raise ValueError(
                f"Verification failed for existing asset: {destination}; "
                f"bytes={actual_size}/{expected_size}"
            )
        elif not force:
            raise ValueError(
                f"Refusing to replace mismatched existing asset without --force: {destination}"
            )
    elif verify_only:
        if destination_exists:
            raise ValueError(f"Expected an installed regular file: {destination}")
        raise FileNotFoundError(destination)
    elif destination_exists:
        if not destination.is_symlink():
            raise ValueError(f"Refusing to replace a non-file asset path: {destination}")
        if not force:
            raise ValueError(
                f"Refusing to replace existing symlink without --force: {destination}"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.partial-", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    copied = 0
    try:
        with archive.open(member) as source, temporary.open("wb") as target:
            while True:
                chunk = source.read(8 * 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
            target.flush()
            os.fsync(target.fileno())
        actual_sha = digest.hexdigest()
        if copied != expected_size or (expected_sha is not None and actual_sha != expected_sha):
            raise ValueError(
                f"Verification failed for {member}: bytes={copied}/{expected_size}, "
                f"sha256={actual_sha}/{expected_sha or 'release-level-integrity'}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(destination), "bytes": copied, "sha256": actual_sha, "action": "installed"}


def publish_shared_scannet(
    profile_root: Path, *, force: bool, verify_only: bool
) -> None:
    shared = profile_root / "shared/scannet.tar"
    if not shared.is_file():
        return
    for relative in ("depth/scannet.tar", "surface_normals/scannet.tar"):
        destination = target_destination(
            profile_root, AssetTarget("scannet_alias", relative)
        )
        destination_exists = destination.exists() or destination.is_symlink()
        if destination.is_file():
            if os.path.samefile(shared, destination):
                continue
            if (
                shared.stat().st_size == destination.stat().st_size
                and sha256_file(shared) == sha256_file(destination)
            ):
                continue
            if verify_only:
                raise ValueError(f"ScanNet alias does not match {shared}: {destination}")
            if not force:
                raise ValueError(
                    f"Refusing to replace mismatched ScanNet alias without --force: {destination}"
                )
            destination.unlink()
        elif verify_only:
            if destination_exists:
                raise ValueError(f"Expected a regular ScanNet alias: {destination}")
            raise FileNotFoundError(destination)
        elif destination_exists:
            if not destination.is_symlink():
                raise ValueError(f"Refusing to replace a non-file ScanNet alias: {destination}")
            if not force:
                raise ValueError(
                    f"Refusing to replace existing ScanNet symlink without --force: {destination}"
                )
            destination.unlink()
        if verify_only:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(shared, destination)
        except OSError:
            destination.symlink_to(os.path.relpath(shared, destination.parent))


def load_release_records(release_root: Path) -> dict[str, dict]:
    path = release_root / "RELEASE_MANIFEST.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("release_id") != RELEASE_ID:
        raise ValueError(f"{path}: incompatible release manifest")
    records: dict[str, dict] = {}
    for row in payload.get("packages", []):
        records[str((release_root / row["path"]).resolve())] = row
    return records


def find_packages(release_root: Path) -> tuple[dict[str, Path], dict[str, dict]]:
    release_records = load_release_records(release_root)
    packages: dict[str, Path] = {}
    for path in sorted(release_root.rglob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            if (
                "PACKAGE_MANIFEST.json" not in archive.namelist()
                and path.name not in LEGACY_PACKAGE_FILENAMES
            ):
                continue
            manifest = load_or_synthesize_manifest(path, archive)
        package_id = manifest["package_id"]
        if package_id in packages:
            raise ValueError(f"Duplicate package_id {package_id}: {packages[package_id]} and {path}")
        packages[package_id] = path
    return packages, release_records


def selected_package_ids(profile: str) -> Iterable[str]:
    full = (
        "eval_nyu_full",
        "eval_ibims1_full",
        "eval_diode_full",
        "eval_kitti_eigen_full",
        "eval_eth3d_unchanged",
        "eval_kitti_unchanged",
        "eval_nuscenes_unchanged",
        "eval_scannet_full",
        "eval_vkitti2_full",
        "eval_sintel_full",
    )
    ablation = (
        "eval_nuscenes_ablation750",
        "eval_scannet_ablation600",
        "eval_vkitti2_ablation500",
    )
    training = tuple(TRAINING_PACKAGE_SOURCES)
    if profile == "full":
        return full
    if profile == "ablation":
        return ablation
    if profile == "training":
        return training
    return (*full, *ablation, *training)


def profile_destination(destination_root: Path, package_id: str) -> Path:
    if "_ablation" in package_id:
        return destination_root / "profiles/ablation"
    return destination_root


def publish_training_specs(
    destination_root: Path,
    installation: list[dict],
    *,
    verify_only: bool,
) -> Path:
    """Validate shipped indexes and publish a trainer-ready portable mix."""
    installed_hashes = {
        str(Path(asset["path"]).resolve()): asset["sha256"]
        for package in installation
        for asset in package["assets"]
    }
    datasets = []
    for archive in sorted((destination_root / "training").glob("*_depth.tar")) + sorted(
        (destination_root / "training").glob("*_surface_normals.tar")
    ):
        index = archive.with_name(archive.name + ".ovb-index.sqlite")
        if not index.is_file():
            raise FileNotFoundError(f"Missing shipped index for {archive}")
        with sqlite3.connect(f"file:{index}?mode=ro", uri=True) as connection:
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            rows = int(connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0])
        if int(metadata.get("schema_version", 0)) != 1 or rows <= 0:
            raise ValueError(f"Invalid shipped index: {index}")
        fingerprint = json.loads(metadata["archive_fingerprint"])
        if archive.stat().st_size != int(fingerprint["size"]):
            raise ValueError(f"Index/archive size mismatch: {archive}")
        expected_sha = fingerprint.get("sha256")
        installed_sha = installed_hashes.get(str(archive.resolve()))
        if expected_sha and installed_sha != expected_sha:
            raise ValueError(f"Index/archive SHA-256 mismatch: {archive}")
        if rows != int(metadata.get("num_rows", -1)):
            raise ValueError(f"Index row-count mismatch: {index}")
        datasets.append(
            {
                "name": archive.name.removesuffix(".tar"),
                "type": "indexed_prepared_tar",
                "path": str(archive.resolve()),
                "index_path": str(index.resolve()),
                "weight": 1.0,
            }
        )
    if not datasets:
        raise FileNotFoundError(destination_root / "training")
    output = target_destination(
        destination_root,
        AssetTarget("geometry_mix", "training/geometry_mix.json"),
    )
    payload = {"samples_per_epoch": 50000, "seed": 42, "shuffle": True, "datasets": datasets}
    if verify_only:
        if not output.is_file():
            raise FileNotFoundError(output)
        installed_payload = json.loads(output.read_text(encoding="utf-8"))
        if installed_payload != payload:
            raise ValueError(f"Installed training dataset spec does not match: {output}")
        return output
    temporary = output.with_name(f".{output.name}.partial-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=("full", "ablation", "training", "all"), default="all"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing installation without modifying any files.",
    )
    parser.add_argument("--plan", action="store_true", help="Validate package discovery and print the install mapping without copying data.")
    args = parser.parse_args()

    release_root = args.release_root.expanduser().resolve()
    destination_root = args.destination_root.expanduser().resolve()
    packages, release_records = find_packages(release_root)
    required = tuple(selected_package_ids(args.profile))
    missing = [package_id for package_id in required if package_id not in packages]
    if missing:
        raise FileNotFoundError(
            "The release directory is missing required packages: " + ", ".join(missing)
        )

    if args.plan:
        for package_id in required:
            package_path = packages[package_id]
            with zipfile.ZipFile(package_path) as archive:
                manifest = load_or_synthesize_manifest(package_path, archive)
                declared = {row["name"] for row in manifest["assets"]}
                targets = (
                    training_targets(manifest)
                    if package_id in TRAINING_PACKAGE_SOURCES
                    else PACKAGE_TARGETS[package_id]
                )
                missing_members = [target.member for target in targets if target.member not in declared]
                if missing_members:
                    raise ValueError(f"{package_path}: missing declared members {missing_members}")
                root = profile_destination(destination_root, package_id)
                print(f"{package_id}: {package_path}")
                for target in targets:
                    print(f"  {target.member} -> {target_destination(root, target)}")
        return

    installation: list[dict] = []
    for package_id in required:
        package_path = packages[package_id]
        release_record = release_records.get(str(package_path.resolve()))
        if release_record is None or release_record.get("status") != "PASS":
            raise ValueError(f"{package_path}: missing PASS record in RELEASE_MANIFEST.json")
        if package_path.stat().st_size != int(release_record["bytes"]):
            raise ValueError(f"{package_path}: outer ZIP byte count differs from release manifest")
        outer_sha = release_record.get("sha256")
        if outer_sha and sha256_file(package_path) != outer_sha:
            raise ValueError(f"{package_path}: outer ZIP SHA-256 differs from release manifest")
        with zipfile.ZipFile(package_path) as archive:
            corrupt = archive.testzip()
            if corrupt:
                raise ValueError(f"ZIP CRC check failed for {package_path}: {corrupt}")
            manifest = load_or_synthesize_manifest(package_path, archive)
            expected_assets = {row["name"]: row for row in manifest["assets"]}
            targets = (
                training_targets(manifest)
                if package_id in TRAINING_PACKAGE_SOURCES
                else PACKAGE_TARGETS[package_id]
            )
            root = profile_destination(destination_root, package_id)
            package_record = {
                "package_id": package_id,
                "source_zip": str(package_path),
                "profile_root": str(root),
                "assets": [],
            }
            for target in targets:
                if target.member not in expected_assets:
                    raise ValueError(f"{package_path}: manifest does not declare {target.member}")
                destination = target_destination(root, target)
                record = copy_verified_member(
                    archive,
                    target.member,
                    destination,
                    expected_assets[target.member],
                    force=args.force,
                    verify_only=args.verify_only,
                )
                package_record["assets"].append(record)
            manifest_path = target_destination(
                root,
                AssetTarget(
                    "PACKAGE_MANIFEST.json",
                    f"package_manifests/{package_id}.json",
                ),
            )
            manifest_dir = manifest_path.parent
            if args.verify_only:
                if not manifest_path.is_file():
                    raise FileNotFoundError(manifest_path)
                installed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if installed_manifest != manifest:
                    raise ValueError(f"Installed package manifest does not match: {manifest_path}")
            else:
                manifest_dir.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )
            installation.append(package_record)

    training_specs = None
    if any(package_id in TRAINING_PACKAGE_SOURCES for package_id in required):
        training_specs = publish_training_specs(
            destination_root, installation, verify_only=args.verify_only
        )

    if args.verify_only:
        print(f"Verified {len(installation)} installed packages under {destination_root}")
        return

    destination_root.mkdir(parents=True, exist_ok=True)
    output = target_destination(
        destination_root,
        AssetTarget("installation_record", "geometry_release_installation.json"),
    )
    payload = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "source_root": str(release_root),
        "destination_root": str(destination_root),
        "profile": args.profile,
        "packages": installation,
        "training_dataset_specs": str(training_specs) if training_specs else None,
    }
    temporary = output.with_name(f".{output.name}.partial-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(f"Installed and verified {len(installation)} packages under {destination_root}")
    print(f"Installation record: {output}")


if __name__ == "__main__":
    main()
