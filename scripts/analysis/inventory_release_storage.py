#!/usr/bin/env python3
"""Bounded, read-only storage inventory of the sources selected by release recipes.

ZIP central directories and uncompressed TAR headers are inspected, never extracted.
Compressed TARs are deliberately not decompressed. Counts describe staged objects,
not source integrity, complete acquisition, or the eventual conversion working set.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tarfile
import zipfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def archive_inventory(path, limit):
    name = path.name.lower()
    if name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar.zst")):
        return {"status": "UNKNOWN_COMPRESSED_TAR", "expanded_bytes": None,
                "reason": "Full decompression excluded from metadata-only inventory"}
    if not name.endswith((".zip", ".tar")):
        return None
    result = dict(status="HEADERS_ONLY", regular_files=0, directories=0,
                  other_members=0, expanded_bytes=0, nested_archives=0)
    try:
        if name.endswith(".zip"):
            # zipfile loads the central directory; never read member bodies.
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                if len(members) > limit:
                    return {"status": "UNKNOWN_MEMBER_LIMIT", "expanded_bytes": None}
                for member in members:
                    mode = member.external_attr >> 16
                    kind = ("directories" if member.is_dir() else
                            "other_members" if stat.S_ISLNK(mode) else "regular_files")
                    result[kind] += 1
                    if kind == "regular_files":
                        result["expanded_bytes"] += member.file_size
                        result["nested_archives"] += is_archive(member.filename)
        else:
            # r: prohibits implicit gzip/bzip2/xz decompression.
            with tarfile.open(path, "r:") as archive:
                for index, member in enumerate(archive):
                    if index >= limit:
                        return {"status": "UNKNOWN_MEMBER_LIMIT", "expanded_bytes": None}
                    kind = ("regular_files" if member.isfile() else
                            "directories" if member.isdir() else "other_members")
                    result[kind] += 1
                    if member.isfile():
                        result["expanded_bytes"] += member.size
                        result["nested_archives"] += is_archive(member.name)
        return result
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        return {"status": "ERROR", "expanded_bytes": None, "error": str(exc)}


def is_archive(name):
    return name.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz", ".tar.xz", ".tar.bz2", ".tar.zst"))


def scan_source(root, entry_limit=10000, member_limit=200000):
    result = dict(path=str(root), status="SCANNED", entries=0, directories=0,
                  symlink_directories_skipped=[], errors=[], files=[])
    if not root.exists():
        result["status"] = "NOT_STAGED"
        return result
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if result["entries"] >= entry_limit:
                        result["status"] = "TRUNCATED_ENTRY_LIMIT"
                        return result
                    result["entries"] += 1
                    path = Path(entry.path)
                    try:
                        if entry.is_symlink() and entry.is_dir():
                            result["symlink_directories_skipped"].append(str(path))
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            result["directories"] += 1
                            pending.append(path)
                            continue
                        info = entry.stat()
                        if not stat.S_ISREG(info.st_mode):
                            result["errors"].append(f"Non-regular or broken entry: {path}")
                            continue
                        row = dict(path=str(path), resolved_path=str(path.resolve()),
                                   bytes=info.st_size, device=info.st_dev, inode=info.st_ino,
                                   symlink=entry.is_symlink())
                        inventory = archive_inventory(path, member_limit)
                        if inventory is not None:
                            row["archive"] = inventory
                        result["files"].append(row)
                    except OSError as exc:
                        result["errors"].append(f"{path}: {exc}")
        except OSError as exc:
            result["errors"].append(f"{directory}: {exc}")
    if result["errors"] or result["symlink_directories_skipped"]:
        result["status"] = "INCOMPLETE"
    result["files"].sort(key=lambda row: row["path"])
    return result


def unique_bytes(sources):
    objects = {}
    for source in sources:
        for row in source["staging"]["files"]:
            objects[(row["device"], row["inode"])] = row["bytes"]
    return sum(objects.values()), len(objects)


def main():
    from common.release_data import acquisition_items, raw_directory
    from scripts import setup_data
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--entry-limit", type=int, default=10000)
    parser.add_argument("--member-limit", type=int, default=200000)
    args = parser.parse_args()
    if min(args.entry_limit, args.member_limit) < 1:
        parser.error("Inventory limits must be positive")
    selected = {}
    recipes = []
    for config in args.config:
        cfg, datasets = setup_data.resolve(setup_data.parser().parse_args(
            ["--config", str(config), "--data-root", str(args.data_root)]))
        recipes.append(dict(path=str(config.resolve()), sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                            acquisition=cfg.get("acquisition"), preparation=cfg.get("preparation")))
        for item in acquisition_items(datasets):
            selected.setdefault(item.name, (item, []))[1].append(str(config))
    rows = []
    for name, (item, configs) in sorted(selected.items()):
        rows.append(dict(dataset=name, recipes=configs, access=item.access, source=item.source,
                         catalog_files=[dict(url=url, filename=filename) for url, filename in item.files],
                         acquisition_scripts=[step.script for step in item.steps],
                         staging=scan_source(raw_directory(args.data_root / "raw", item),
                                             args.entry_limit, args.member_limit),
                         missing_download_bytes=None, prepared_bytes=None, scratch_peak_bytes=None))
    byte_count, count = unique_bytes(rows)
    print(json.dumps(dict(status="PARTIAL_STAGED_METADATA_NOT_FULL_RECIPE_BUDGET",
                         timestamp_utc=datetime.now(timezone.utc).isoformat(), recipes=recipes,
                         data_root=str(args.data_root.resolve()), sources=rows,
                         known_unique_staged_bytes=byte_count, known_unique_staged_files=count,
                         scope="Only selected source directories below data_root/raw; file symlinks resolved, directory symlinks skipped",
                         limitations=["No downloads, extraction, source hashes or integrity verification",
                                      "Compressed TAR expansion and nested archive expansion unknown",
                                      "Scripted upstream object sizes not queried; missing bytes unknown",
                                      "Logical bytes, not allocated disk usage; no full quota or scratch estimate",
                                      "ZIP central-directory loading may use memory proportional to member count"]), indent=2))


if __name__ == "__main__":
    main()
