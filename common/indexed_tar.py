"""Random-access helpers for uncompressed prepared-dataset tar archives."""

from __future__ import annotations

import hashlib
import atexit
import io
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from PIL import Image


INDEX_SCHEMA_VERSION = 1
_FD_CACHE: dict[tuple[int, str], int] = {}
_HASH_CACHE: dict[tuple, str] = {}
_PNG_TEXT_CHUNK_RETRY_LIMIT = 64 * 1024 * 1024
_PNG_TEXT_CHUNK_ERROR = "PngImagePlugin.MAX_TEXT_CHUNK"
_PNG_TEXT_CHUNK_LOCK = threading.Lock()


@dataclass(frozen=True)
class IndexedTarReference:
    archive: str
    member: str
    offset: int
    size: int


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verified_sha256(path: Path) -> str:
    """Hash current bytes once per process and unchanged file identity."""
    path = path.resolve()
    def identity():
        stat = path.stat()
        return (os.getpid(), str(path), stat.st_dev, stat.st_ino, stat.st_size,
                stat.st_mtime_ns, stat.st_ctime_ns)
    before = identity()
    # Some shared filesystems round timestamps. Rapid same-size writes can
    # retain both timestamps, so never cache recently created/modified files.
    stable = time.time_ns() - max(before[-2:]) > 2_000_000_000
    if not stable or before not in _HASH_CACHE:
        digest = sha256_file(path)
        if identity() != before:
            raise ValueError(f"Archive changed during verification: {path}")
        # Keep only the current identity for each path in this process.
        for key in list(_HASH_CACHE):
            if key[:2] == before[:2]:
                del _HASH_CACHE[key]
        if not stable:
            return digest
        _HASH_CACHE[before] = digest
    return _HASH_CACHE[before]


def archive_fingerprint(path: Path, include_sha256: bool = True) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        result["sha256"] = verified_sha256(path)
    return result


def validate_archive_fingerprint(archive: Path, expected: dict[str, Any]) -> None:
    actual = archive.stat()
    if actual.st_size != int(expected["size"]):
        raise ValueError(f"Indexed tar size changed: {archive}")
    expected_sha = expected.get("sha256")
    if expected_sha:
        actual_sha = verified_sha256(archive)
        if actual_sha != expected_sha:
            raise ValueError(f"Indexed tar SHA-256 changed: {archive}")
    elif actual.st_mtime_ns != int(expected["mtime_ns"]):
        raise ValueError(
            f"Indexed tar modification time changed and the index has no SHA-256: {archive}"
        )


def read_reference_bytes(reference: IndexedTarReference) -> bytes:
    key = (os.getpid(), reference.archive)
    fd = _FD_CACHE.get(key)
    if fd is None:
        fd = os.open(reference.archive, os.O_RDONLY)
        _FD_CACHE[key] = fd
    data = os.pread(fd, reference.size, reference.offset)
    if len(data) != reference.size:
        raise OSError(
            f"Short read for {reference.member}: expected {reference.size}, got {len(data)}"
        )
    return data


@atexit.register
def _close_cached_descriptors() -> None:
    for fd in _FD_CACHE.values():
        try:
            os.close(fd)
        except OSError:
            pass
    _FD_CACHE.clear()


def _open_image_with_png_metadata_retry(factory):
    """Open an image, allowing a bounded retry for legitimate large PNG ICC chunks.

    Pillow's default 1 MiB text-chunk limit also applies to an ``iCCP`` chunk.
    Some source PNGs have a valid ICC profile slightly above that limit. Keep
    Pillow's normal decompression-bomb protection for all other data, and only
    retry this one specific parser rejection with a bounded 64 MiB limit.
    """
    from PIL import PngImagePlugin

    try:
        return factory()
    except ValueError as error:
        if _PNG_TEXT_CHUNK_ERROR not in str(error):
            raise

    # PngImagePlugin reads this module-global setting; header indexing may run
    # in several threads, so the override is lock-guarded and ``factory``
    # supplies a fresh BytesIO for the retry.
    with _PNG_TEXT_CHUNK_LOCK:
        original_limit = PngImagePlugin.MAX_TEXT_CHUNK
        try:
            PngImagePlugin.MAX_TEXT_CHUNK = max(original_limit, _PNG_TEXT_CHUNK_RETRY_LIMIT)
            return factory()
        finally:
            PngImagePlugin.MAX_TEXT_CHUNK = original_limit


def open_reference_image(reference: IndexedTarReference) -> "Image.Image":
    from PIL import Image

    image_bytes = read_reference_bytes(reference)
    return _open_image_with_png_metadata_retry(lambda: Image.open(io.BytesIO(image_bytes)))


def load_index(index_path: Path, archive_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load normalized rows and replace image fields with byte-offset references."""
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        if int(metadata.get("schema_version", 0)) != INDEX_SCHEMA_VERSION:
            raise ValueError(f"Unsupported indexed-tar schema in {index_path}")
        fingerprint = json.loads(metadata["archive_fingerprint"])
        validate_archive_fingerprint(archive_path, fingerprint)
        members = {
            name: (int(offset), int(size))
            for name, offset, size in connection.execute("SELECT name, offset, size FROM members")
        }
        rows = []
        for (payload,) in connection.execute("SELECT payload FROM rows ORDER BY row_number"):
            row = json.loads(payload)
            # Older archives name the supervision mask `valid_mask` (paths relative
            # to the metadata member's directory); current archives use `loss_mask`
            # with tar-rooted paths. Normalize before converting to references.
            if row.get("loss_mask") in (None, "") and row.get("valid_mask") not in (None, ""):
                mask_member = str(row["valid_mask"])
                if mask_member not in members:
                    metadata_member = str(row.get("_metadata_member", ""))
                    prefix = metadata_member.rsplit("/", 1)[0] if "/" in metadata_member else ""
                    prefixed = f"{prefix}/{mask_member}" if prefix else mask_member
                    if prefixed in members:
                        mask_member = prefixed
                row["loss_mask"] = mask_member
            for field in ("image", "conditioning_image", "loss_mask"):
                member = row.get(field)
                if member in (None, ""):
                    continue
                if member not in members:
                    raise ValueError(f"Index row references missing tar member {member!r}")
                offset, size = members[member]
                row[field] = IndexedTarReference(str(archive_path), member, offset, size)
            rows.append(row)
        return rows, fingerprint
    finally:
        connection.close()


def iter_index_rows(index_path: Path) -> Iterator[dict[str, Any]]:
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        for (payload,) in connection.execute("SELECT payload FROM rows ORDER BY row_number"):
            yield json.loads(payload)
    finally:
        connection.close()
