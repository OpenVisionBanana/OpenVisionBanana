#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import tarfile
from pathlib import Path


TAR_BLOCK_SIZE = 512
TAR_END_SIZE = TAR_BLOCK_SIZE * 2


def md5_base64(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - matching the source-provided checksum
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def find_incomplete_final_member(path: Path) -> tarfile.TarInfo:
    last_member: tarfile.TarInfo | None = None
    try:
        with tarfile.open(path, "r:*") as archive:
            while True:
                member = archive.next()
                if member is None:
                    raise ValueError("archive ended cleanly; repair is not needed")
                last_member = member
    except tarfile.ReadError as exc:
        if last_member is None:
            raise ValueError("archive has no readable members") from exc
        declared_end = last_member.offset_data + last_member.size
        if declared_end <= path.stat().st_size:
            raise ValueError("tar error is not an incomplete final member") from exc
        return last_member


def repair(path: Path, expected_md5_base64: str, manifest: Path) -> dict[str, object]:
    actual_md5 = md5_base64(path)
    if actual_md5 != expected_md5_base64:
        raise ValueError("local file does not match the source-provided MD5")

    original_size = path.stat().st_size
    incomplete = find_incomplete_final_member(path)
    truncate_at = incomplete.offset
    if truncate_at <= 0 or truncate_at % TAR_BLOCK_SIZE != 0:
        raise ValueError(f"unsafe tar truncation offset: {truncate_at}")

    with path.open("r+b") as handle:
        handle.truncate(truncate_at)
        handle.seek(truncate_at)
        handle.write(b"\0" * TAR_END_SIZE)
        handle.flush()
        os.fsync(handle.fileno())

    with tarfile.open(path, "r:*") as archive:
        member_count = len(archive.getmembers())

    record: dict[str, object] = {
        "archive": path.name,
        "source_md5_base64": expected_md5_base64,
        "source_size": original_size,
        "repaired_size": path.stat().st_size,
        "dropped_member": incomplete.name,
        "dropped_declared_size": incomplete.size,
        "remaining_members": member_count,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair an official Taskonomy tar with one truncated final member.")
    parser.add_argument("tar_path", type=Path)
    parser.add_argument("--expected-md5-base64", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    record = repair(args.tar_path, args.expected_md5_base64, args.manifest)
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
