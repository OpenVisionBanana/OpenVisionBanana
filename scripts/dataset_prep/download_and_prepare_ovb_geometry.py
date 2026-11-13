#!/usr/bin/env python3
"""Retired temporary-data downloader. Network entry points refuse to run.

Use scripts/setup_data.py with the original-source ZS or NZS recipes instead.
Local verification helpers remain for interpreting historical artifacts only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


REPO_ID = ""  # Retired temporary research-upload route. No download target.
RELEASE = "ovb_geometry_training_subsets_v1"
# Immutable repository revision.
REVISION = "5fb7da3782443dc4cc4be9d28a591800e83e6ce3"
MANIFEST_SHA256 = "6ab069e8f69b3cdc0fa9384534c56526a3d0f1cd34884bcf3ae041f72fa30e97"
SHA256SUMS_SHA256 = "f7e09954a92076d32d400bfa0c3cfe6183c2f218c51a9d10c258cb02b02a56c9"
LOGICAL_SHA256 = {
    "hypersim_geometry_train_v1.zip": "12bec7cdf24724d00743ec210a2dd392c8dade9f87d73b0ebe552f65869a852d",
    "matterport3d_geometry_train_v1.zip": "d01fce3c07398ed492d91f9af7cd6624b458e096f9567e9debbd6d714c027327",
    "replica_geometry_train_v1.zip": "cd236b08b8407c02f4eeb2476b8de59acdcc03b8debe0d85d00c7c31fa3d3a91",
    "synscapes_geometry_train_v1.zip": "6331b61b6e103c692b31f7b945dfdfa2d567763b704ff58955e8acfd3db39cf6",
    "taskonomy_geometry_train_v1.zip": "7f272588b9126b456ace0e22f026977f7aa5c72a606f4c4dac497266a89d520a",
}
CHUNK = 16 * 1024 * 1024
def existing_download_path(download_root: Path, remote_path: str) -> Path:
    """Return a downloaded file from either supported local HF layout."""
    canonical = download_root / remote_path
    if canonical.is_file():
        return canonical

    # Some huggingface-cli invocations preserve the complete repository path
    # below the selected part directory, for example:
    #   .../synscapes...parts/ovb_geometry.../synscapes...parts/part-0000
    nested = canonical.parent / remote_path
    if nested.is_file():
        return nested
    return canonical


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def remote_inventory() -> list[dict]:
    raise ValueError("Temporary project-data downloads are retired. Use scripts/setup_data.py with official sources.")
    revision = urllib.parse.quote(REVISION, safe="")
    release = urllib.parse.quote(RELEASE, safe="/")
    url = (
        f"https://huggingface.co/api/datasets/{REPO_ID}/tree/{revision}/{release}"
        "?recursive=true&expand=false&limit=1000"
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        entries = json.load(response)
    files = [entry for entry in entries if entry.get("type") == "file"]
    if len(files) != 59:
        raise RuntimeError(f"Pinned release should contain 59 files, found {len(files)}")
    return files


def download_one(entry: dict, download_root: Path, retries: int) -> Path:
    raise ValueError("Temporary project-data downloads are retired. Use scripts/setup_data.py with official sources.")
    remote_path = entry["path"]
    expected_size = int(entry["size"])
    destination = existing_download_path(download_root, remote_path)
    partial = destination.with_name(destination.name + ".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == expected_size:
        print(f"[download:skip] {remote_path}")
        return destination
    if destination.exists():
        destination.unlink()

    quoted_path = urllib.parse.quote(remote_path, safe="/")
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{quoted_path}?download=true"
    for attempt in range(1, retries + 1):
        try:
            current = partial.stat().st_size if partial.exists() else 0
            if current > expected_size:
                partial.unlink()
                current = 0
            if current == expected_size:
                os.replace(partial, destination)
                return destination
            headers = {"User-Agent": "ovb-geometry-release-downloader/1.0"}
            if current:
                headers["Range"] = f"bytes={current}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                resumed = current > 0 and getattr(response, "status", None) == 206
                mode = "ab" if resumed else "wb"
                if current and not resumed:
                    current = 0
                with partial.open(mode) as output:
                    while block := response.read(CHUNK):
                        output.write(block)
            actual_size = partial.stat().st_size
            if actual_size != expected_size:
                raise IOError(
                    f"size mismatch after transfer: expected {expected_size}, got {actual_size}"
                )
            os.replace(partial, destination)
            print(f"[download:done] {remote_path} ({expected_size} bytes)")
            return destination
        except Exception as error:
            if attempt == retries:
                raise RuntimeError(f"Failed downloading {remote_path}: {error}") from error
            wait = min(60, 2 ** attempt)
            print(f"[download:retry {attempt}/{retries}] {remote_path}: {error}; waiting {wait}s")
            time.sleep(wait)
    raise AssertionError("unreachable")


def verify_release_metadata(release_dir: Path) -> dict:
    manifest_path = release_dir / "MULTIPART_MANIFEST.json"
    sums_path = release_dir / "SHA256SUMS"
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("MULTIPART_MANIFEST.json does not match the audited release")
    if sha256_file(sums_path) != SHA256SUMS_SHA256:
        raise RuntimeError("SHA256SUMS does not match the audited release")
    return json.loads(manifest_path.read_text())


def verify_parts(release_dir: Path, manifest: dict, workers: int) -> None:
    checks: list[tuple[Path, int, str]] = []
    for archive in manifest["archives"]:
        part_dir = release_dir / (archive["logical_path"] + ".parts")
        for part in archive["parts"]:
            relative_path = Path(part_dir.name) / part["path"]
            path = existing_download_path(release_dir.parent, str(Path(RELEASE) / relative_path))
            checks.append((path, int(part["bytes"]), part["sha256"]))

    def check(item: tuple[Path, int, str]) -> str:
        path, expected_size, expected_hash = item
        if path.stat().st_size != expected_size:
            raise RuntimeError(f"Part has wrong size: {path}")
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"Part has wrong SHA-256: {path}")
        return path.name

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for name in pool.map(check, checks):
            print(f"[part:verified] {name}")


def assemble_archives(release_dir: Path, archive_dir: Path, manifest: dict) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    split_archives = {entry["logical_path"]: entry for entry in manifest["archives"]}
    for name, expected_hash in LOGICAL_SHA256.items():
        output = archive_dir / name
        if output.is_file() and sha256_file(output) == expected_hash:
            print(f"[archive:skip] {name}")
            continue
        partial = output.with_name(output.name + ".partial")
        if partial.exists():
            partial.unlink()
        if name in split_archives:
            entry = split_archives[name]
            part_dir = release_dir / (name + ".parts")
            with partial.open("wb") as destination:
                for part in entry["parts"]:
                    relative_path = Path(part_dir.name) / part["path"]
                    source = existing_download_path(
                        release_dir.parent, str(Path(RELEASE) / relative_path)
                    )
                    print(f"[archive:append] {source.name} -> {name}")
                    with source.open("rb") as stream:
                        shutil.copyfileobj(stream, destination, length=CHUNK)
        else:
            source = release_dir / name
            print(f"[archive:copy] {source.name}")
            with source.open("rb") as stream, partial.open("wb") as destination:
                shutil.copyfileobj(stream, destination, length=CHUNK)
        actual_hash = sha256_file(partial)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Reassembled {name} has SHA-256 {actual_hash}, expected {expected_hash}"
            )
        os.replace(partial, output)
        print(f"[archive:verified] {name}")


def safe_extract(archive: Path, destination: Path) -> None:
    marker_dir = destination / ".ovb_extraction_state"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / (archive.name + ".json")
    expected_hash = LOGICAL_SHA256[archive.name]
    if marker.is_file():
        try:
            if json.loads(marker.read_text()).get("sha256") == expected_hash:
                print(f"[extract:skip] {archive.name}")
                return
        except (OSError, json.JSONDecodeError):
            pass
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            member_path = (destination / member.filename).resolve()
            if destination_resolved not in member_path.parents and member_path != destination_resolved:
                raise RuntimeError(f"Unsafe ZIP path in {archive}: {member.filename}")
        print(f"[extract:start] {archive.name}")
        bundle.extractall(destination)
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps({"archive": archive.name, "sha256": expected_hash}, indent=2) + "\n")
    os.replace(temporary, marker)
    print(f"[extract:done] {archive.name}")


def copy_release_metadata(release_dir: Path, ready_dir: Path) -> list[Path]:
    copied: list[Path] = []
    for directory in ("configs", "manifests"):
        source = release_dir / directory
        if source.is_dir():
            shutil.copytree(source, ready_dir / directory, dirs_exist_ok=True)
            copied.append(ready_dir / directory)
    for name in ("README.md", "SHA256SUMS", "MULTIPART_MANIFEST.json"):
        source = release_dir / name
        if source.is_file():
            shutil.copy2(source, ready_dir / name)
            copied.append(ready_dir / name)
    return copied


def print_ready_paths(ready_dir: Path) -> None:
    print(f"READY_DATA_ROOT={ready_dir}")
    training_config = ready_dir / "configs/geometry_pool_epoch450k_relative.json"
    if training_config.is_file():
        print(f"TRAINING_CONFIG={training_config}")
    else:
        print(f"[metadata:missing] Training config not present: {training_config}")


def extract_archives(archive_dir: Path, ready_dir: Path, workers: int) -> None:
    archives = [archive_dir / name for name in LOGICAL_SHA256]
    missing = [str(path) for path in archives if not path.is_file()]
    if missing:
        raise RuntimeError("Missing prepared archive(s):\n  " + "\n  ".join(missing))

    ready_dir.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(safe_extract, archive, ready_dir) for archive in archives]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def main() -> None:
    raise ValueError("Temporary project-data downloads are retired. Use configs/data_zs.yaml or configs/data_nzs.yaml.")
    parser = argparse.ArgumentParser(
        description="Download and prepare the pinned OVB geometry training release."
    )
    parser.add_argument(
        "--root",
        "--output",
        dest="root",
        type=Path,
        required=True,
        help=(
            "Preparation root; download/, archives/, and ready/ are resolved "
            "relative to it"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent downloads, hash checks, and archive extractions",
    )
    parser.add_argument("--retries", type=int, default=8, help="Attempts per downloaded file")
    parser.add_argument("--no-extract", action="store_true", help="Stop after verified ZIP assembly")
    parser.add_argument(
        "--ready-from-archives",
        action="store_true",
        help=(
            "Skip download, part verification, and assembly; extract the existing "
            "verified ZIPs from <root>/archives directly into <root>/ready"
        ),
    )
    args = parser.parse_args()

    if args.ready_from_archives and args.no_extract:
        parser.error("--ready-from-archives cannot be combined with --no-extract")

    root = args.root.resolve()
    download_root = root / "download"
    archive_dir = root / "archives"
    ready_dir = root / "ready"
    root.mkdir(parents=True, exist_ok=True)

    release_dir = download_root / RELEASE
    if args.ready_from_archives:
        print(f"Preparation root: {root}")
        print(f"Using prepared archives from: {archive_dir}")
        extract_archives(archive_dir, ready_dir, args.workers)
        if release_dir.is_dir():
            copied = copy_release_metadata(release_dir, ready_dir)
            if not copied:
                print(f"[metadata:skip] No release metadata found in: {release_dir}")
        else:
            print(f"[metadata:skip] Release metadata directory not found: {release_dir}")
        print_ready_paths(ready_dir)
        return

    inventory = remote_inventory()
    total = sum(int(item["size"]) for item in inventory)
    print(f"Pinned revision: {REVISION}")
    print(f"Remote payload: {len(inventory)} files, {total / 1e9:.2f} GB")
    print(f"Preparation root: {root}")
    free = shutil.disk_usage(root).free
    print(f"Free space before download: {free / 1e9:.2f} GB")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(download_one, item, download_root, args.retries) for item in inventory]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    manifest = verify_release_metadata(release_dir)
    verify_parts(release_dir, manifest, args.workers)
    assemble_archives(release_dir, archive_dir, manifest)

    if not args.no_extract:
        extract_archives(archive_dir, ready_dir, args.workers)
        copy_release_metadata(release_dir, ready_dir)
        print_ready_paths(ready_dir)
    else:
        print(f"VERIFIED_ARCHIVE_ROOT={archive_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; rerun the same command to resume downloads.", file=sys.stderr)
        raise SystemExit(130)
