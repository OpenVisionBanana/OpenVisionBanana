#!/usr/bin/env python3
"""Build an immutable SQLite byte-offset index for an uncompressed prepared tar.

Regular TAR members and in-archive hardlinks are both supported.  A hardlink
gets the byte range of its referenced regular member, so the training loader
can read it with ``pread`` exactly like a normal image member.
"""

from __future__ import annotations

import argparse
import json
import stat
import sqlite3
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.indexed_tar import INDEX_SCHEMA_VERSION, archive_fingerprint, sha256_file


FIELDS = ("image", "conditioning_image", "loss_mask")


def resolve_member(value: str, metadata_member: str, members: set[str]) -> str:
    value = value.removeprefix("./")
    candidates = [value]
    parent = str(PurePosixPath(metadata_member).parent)
    if parent != ".":
        candidates.append(str(PurePosixPath(parent) / value))
    for candidate in candidates:
        if candidate in members:
            return candidate
    for candidate in members:
        if candidate.endswith("/" + value):
            return candidate
    raise KeyError(f"Could not resolve {value!r} relative to {metadata_member!r}")


def build_member_offsets(infos: list[tarfile.TarInfo]) -> dict[str, tuple[int, int]]:
    """Return byte ranges for regular TAR members and TAR hardlinks.

    ``TarInfo.offset_data`` for an ``islnk`` entry is not image data: the
    member has size zero and points at another name in the same archive.  Keep
    its logical name in the index but recursively reuse the target range.
    """
    entries = {item.name: item for item in infos if item.isfile() or item.islnk()}
    names = set(entries)
    resolved: dict[str, tuple[int, int]] = {}
    resolving: set[str] = set()

    def resolve(name: str) -> tuple[int, int]:
        if name in resolved:
            return resolved[name]
        if name in resolving:
            raise ValueError(f"Cyclic TAR hardlink involving {name!r}")
        item = entries[name]
        if item.isfile():
            value = (item.offset_data, item.size)
        else:
            resolving.add(name)
            try:
                target = resolve_member(item.linkname, item.name, names)
                value = resolve(target)
            finally:
                resolving.remove(name)
        resolved[name] = value
        return value

    for name in entries:
        resolve(name)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-sha256", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    archive = args.archive.expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    output = (args.output or archive.with_name(archive.name + ".ovb-index.sqlite")).expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --force to rebuild it")

    fingerprint = archive_fingerprint(archive, include_sha256=False)
    if not args.no_sha256:
        actual_sha256 = sha256_file(archive)
        sidecar = archive.with_name(archive.name + ".sha256")
        if sidecar.exists():
            declared = sidecar.read_text(encoding="utf-8").split()[0]
            if declared != actual_sha256:
                raise ValueError(f"SHA-256 sidecar does not match archive bytes: {archive}")
        else:
            sidecar.write_text(f"{actual_sha256}  {archive.name}\n", encoding="utf-8")
        fingerprint["sha256"] = actual_sha256

    with tarfile.open(archive, "r:") as tar:
        infos = list(tar)
        offsets = build_member_offsets(infos)
        member_names = set(offsets)
        metadata_infos = [
            item for item in infos if item.isfile() and PurePosixPath(item.name).name == "metadata.jsonl"
        ]
        if not metadata_infos:
            raise ValueError(f"No metadata.jsonl found in {archive}")
        rows = []
        for metadata_info in sorted(metadata_infos, key=lambda item: item.name):
            extracted = tar.extractfile(metadata_info)
            if extracted is None:
                raise OSError(f"Could not read {metadata_info.name}")
            for line_number, raw_line in enumerate(extracted, 1):
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line)
                for field in FIELDS:
                    if row.get(field) not in (None, ""):
                        row[field] = resolve_member(str(row[field]), metadata_info.name, member_names)
                row["_metadata_member"] = metadata_info.name
                row["_metadata_line"] = line_number
                rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        connection = sqlite3.connect(temporary)
        connection.executescript(
            "PRAGMA journal_mode=OFF;"
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            "CREATE TABLE members (name TEXT PRIMARY KEY, offset INTEGER NOT NULL, size INTEGER NOT NULL);"
            "CREATE TABLE rows (row_number INTEGER PRIMARY KEY, payload TEXT NOT NULL);"
        )
        connection.executemany("INSERT INTO members VALUES (?, ?, ?)", ((name, *value) for name, value in offsets.items()))
        connection.executemany(
            "INSERT INTO rows VALUES (?, ?)",
            ((number, json.dumps(row, separators=(",", ":"), sort_keys=True)) for number, row in enumerate(rows)),
        )
        metadata = {
            "schema_version": str(INDEX_SCHEMA_VERSION),
            "archive_fingerprint": json.dumps(fingerprint, sort_keys=True),
            "num_rows": str(len(rows)),
            "num_members": str(len(offsets)),
            "metadata_members": json.dumps([item.name for item in metadata_infos]),
        }
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", metadata.items())
        connection.commit()
        connection.close()
        temporary.replace(output)
        # SQLite may create the file under a restrictive umask; match the
        # archive's read permissions while retaining owner read/write.
        archive_mode = stat.S_IMODE(archive.stat().st_mode)
        output.chmod(0o600 | (archive_mode & 0o066))
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"archive": str(archive), "index": str(output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
