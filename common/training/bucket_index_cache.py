"""Durable cache for deterministic aspect-ratio bucket assignments.

The training dataset reads every target-image header once to map it to an
aspect-ratio bucket. For prepared pools this is deterministic but expensive,
and performing it independently on every rank makes a restart unnecessarily
slow. This module stores the compact ``row_index -> bucket_index`` array under
an immutable configuration fingerprint. A filesystem lock lets one process
build the entry while other ranks and later Slurm links reuse it.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _fingerprint(metadata: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(metadata)).hexdigest()


def _paths(cache_dir: Path, fingerprint: str) -> tuple[Path, Path, Path]:
    stem = f"bucket-index-v{SCHEMA_VERSION}-{fingerprint}"
    return (
        cache_dir / f"{stem}.npy",
        cache_dir / f"{stem}.json",
        cache_dir / f"{stem}.lock",
    )


def _expected_manifest(metadata: dict[str, Any], row_count: int, bucket_count: int) -> dict[str, Any]:
    fingerprint = _fingerprint(metadata)
    return {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "metadata": metadata,
        "row_count": int(row_count),
        "bucket_count": int(bucket_count),
    }


def _load_if_complete(data_path: Path, manifest_path: Path, expected: dict[str, Any]) -> np.ndarray | None:
    """Return a verified entry, or ``None`` for absent, stale, or corrupt data."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest != expected:
        return None
    try:
        with data_path.open("rb") as handle:
            bucket_indices = np.load(handle, allow_pickle=False)
    except (OSError, ValueError):
        return None

    row_count = int(expected["row_count"])
    bucket_count = int(expected["bucket_count"])
    if (
        bucket_indices.ndim != 1
        or len(bucket_indices) != row_count
        or bucket_indices.dtype.kind not in {"u", "i"}
        or (len(bucket_indices) and (int(bucket_indices.min()) < 0 or int(bucket_indices.max()) >= bucket_count))
    ):
        return None
    return np.ascontiguousarray(bucket_indices, dtype=np.uint16)


def _load_compatible_entry(
    root: Path,
    expected: dict[str, Any],
    metadata_compatible: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> np.ndarray | None:
    """Load a valid older entry whose metadata is explicitly compatible.

    The callback must establish that the existing assignment still addresses
    the same ordered rows.
    """
    for candidate_manifest_path in root.glob(f"bucket-index-v{SCHEMA_VERSION}-*.json"):
        try:
            candidate = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        candidate_metadata = candidate.get("metadata")
        if not isinstance(candidate_metadata, dict):
            continue
        if (
            candidate.get("schema_version") != SCHEMA_VERSION
            or candidate.get("fingerprint") != _fingerprint(candidate_metadata)
            or candidate.get("row_count") != expected["row_count"]
            or candidate.get("bucket_count") != expected["bucket_count"]
            or not metadata_compatible(candidate_metadata, expected["metadata"])
        ):
            continue
        cached = _load_if_complete(
            candidate_manifest_path.with_suffix(".npy"), candidate_manifest_path, candidate
        )
        if cached is not None:
            return cached
    return None


def _atomic_write_bytes(path: Path, writer: Callable[[Any], None]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("xb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_entry(data_path: Path, manifest_path: Path, bucket_indices: np.ndarray, manifest: dict[str, Any]) -> None:
    _atomic_write_bytes(data_path, lambda handle: np.save(handle, bucket_indices, allow_pickle=False))
    _atomic_write_bytes(
        manifest_path,
        lambda handle: handle.write(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"),
    )


def load_or_build_bucket_indices(
    cache_dir: str | os.PathLike,
    metadata: dict[str, Any],
    row_count: int,
    bucket_count: int,
    build_indices: Callable[[], np.ndarray],
    log_fn: Callable[[str], None] | None = None,
    metadata_compatible: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None,
) -> tuple[np.ndarray, bool]:
    """Load a complete bucket index or build it exactly once under a file lock.

    Returns ``(bucket_indices, reused)``. An entry is valid only once both the
    array and its manifest are written. IO failures propagate to the caller.
    """
    if row_count < 0:
        raise ValueError(f"row_count must be non-negative, got {row_count}")
    if not 1 <= bucket_count <= np.iinfo(np.uint16).max:
        raise ValueError(f"bucket_count must be in [1, {np.iinfo(np.uint16).max}], got {bucket_count}")

    root = Path(cache_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    expected = _expected_manifest(metadata, row_count, bucket_count)
    data_path, manifest_path, lock_path = _paths(root, expected["fingerprint"])

    cached = _load_if_complete(data_path, manifest_path, expected)
    if cached is not None:
        return cached, True

    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            cached = _load_if_complete(data_path, manifest_path, expected)
            if cached is not None:
                return cached, True
            if metadata_compatible is not None:
                compatible = _load_compatible_entry(root, expected, metadata_compatible)
                if compatible is not None:
                    try:
                        _write_entry(data_path, manifest_path, compatible, expected)
                    except OSError as error:
                        if log_fn is not None:
                            log_fn(
                                "Reusing compatible persistent bucket index without migration "
                                f"because writing {data_path} failed: {error}"
                            )
                    else:
                        if log_fn is not None:
                            log_fn(f"Migrated compatible persistent bucket index to {data_path}.")
                    return compatible, True
            if log_fn is not None:
                log_fn(
                    f"Building persistent bucket index ({row_count:,} rows) at {data_path}; "
                    "later ranks and chain links will reuse it."
                )
            bucket_indices = np.asarray(build_indices())
            if (
                bucket_indices.ndim != 1
                or len(bucket_indices) != row_count
                or bucket_indices.dtype.kind not in {"u", "i"}
                or (
                    len(bucket_indices)
                    and (int(bucket_indices.min()) < 0 or int(bucket_indices.max()) >= bucket_count)
                )
            ):
                raise ValueError("Bucket-index builder returned invalid assignments.")
            bucket_indices = np.ascontiguousarray(bucket_indices, dtype=np.uint16)
            _write_entry(data_path, manifest_path, bucket_indices, expected)
            return bucket_indices, False
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
