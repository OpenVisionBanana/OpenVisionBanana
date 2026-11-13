"""Deterministic process-level sharding for image evaluation.

Each process evaluates a round-robin slice of the frozen sample list.  Shards
publish their records atomically and shard zero restores the original global
sample order before the existing aggregation code runs.  No metric is reduced
per shard, which keeps multi-GPU results numerically identical to one process.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence, TypeVar


T = TypeVar("T")
SCHEMA_VERSION = 1


def validate_shard_parameters(num_shards: int, shard_index: int) -> None:
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards}), got {shard_index}"
        )


def round_robin_shard(values: Sequence[T], num_shards: int, shard_index: int) -> list[T]:
    """Return the stable round-robin slice assigned to one evaluator process."""

    validate_shard_parameters(num_shards, shard_index)
    return list(values[shard_index::num_shards])


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sharding_run_identity(identity: Any, sample_keys: Sequence[str], num_shards: int) -> str:
    payload = {
        "identity": identity,
        "sample_keys": list(sample_keys),
        "num_shards": num_shards,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        # Preserve each metric record's insertion order so the final JSONL is
        # byte-for-byte identical to a single-process evaluator run.
        json.dump(value, handle, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def synchronize_shard_records(
    *,
    sync_dir: str | Path,
    num_shards: int,
    shard_index: int,
    run_identity: str,
    full_sample_keys: Sequence[str],
    record_items: Sequence[tuple[str, dict[str, Any]]],
    skipped_items: Sequence[tuple[str, dict[str, Any]]],
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Publish one shard and merge all records on shard zero.

    Non-zero shards return ``None`` after their atomic publication.  Shard zero
    waits for every matching payload, checks exact one-to-one sample coverage,
    and returns records/skips in the original unsharded order.
    """

    validate_shard_parameters(num_shards, shard_index)
    if num_shards == 1:
        return (
            [record for _, record in record_items],
            [record for _, record in skipped_items],
        )
    if timeout_seconds <= 0:
        raise ValueError("shard synchronization timeout must be positive")

    root = Path(sync_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "run_identity": run_identity,
        "records": [
            {"sample_key": key, "record": record} for key, record in record_items
        ],
        "skipped": [
            {"sample_key": key, "record": record} for key, record in skipped_items
        ],
    }
    _atomic_json(root / f"shard-{shard_index:04d}.json", payload)
    if shard_index != 0:
        return None

    paths = [root / f"shard-{index:04d}.json" for index in range(num_shards)]
    deadline = time.monotonic() + timeout_seconds
    while not all(path.is_file() for path in paths):
        if time.monotonic() >= deadline:
            missing = [path.name for path in paths if not path.is_file()]
            raise TimeoutError(
                "timed out waiting for evaluation shards: " + ", ".join(missing)
            )
        time.sleep(min(1.0, max(0.01, deadline - time.monotonic())))

    by_key: dict[str, tuple[str, dict[str, Any]]] = {}
    for expected_index, path in enumerate(paths):
        with path.open("r", encoding="utf-8") as handle:
            shard = json.load(handle)
        expected_header = {
            "schema_version": SCHEMA_VERSION,
            "num_shards": num_shards,
            "shard_index": expected_index,
            "run_identity": run_identity,
        }
        actual_header = {key: shard.get(key) for key in expected_header}
        if actual_header != expected_header:
            raise RuntimeError(
                f"evaluation shard identity mismatch in {path}: "
                f"expected {expected_header}, found {actual_header}"
            )
        for kind in ("records", "skipped"):
            entries = shard.get(kind)
            if not isinstance(entries, list):
                raise RuntimeError(f"invalid {kind} list in evaluation shard {path}")
            for entry in entries:
                key = entry.get("sample_key") if isinstance(entry, dict) else None
                record = entry.get("record") if isinstance(entry, dict) else None
                if not isinstance(key, str) or not isinstance(record, dict):
                    raise RuntimeError(f"invalid {kind} entry in evaluation shard {path}")
                if key in by_key:
                    raise RuntimeError(f"duplicate evaluation sample across shards: {key}")
                by_key[key] = (kind, record)

    expected_keys = list(full_sample_keys)
    if len(expected_keys) != len(set(expected_keys)):
        raise ValueError("full evaluation sample keys must be unique")
    missing = [key for key in expected_keys if key not in by_key]
    unexpected = sorted(set(by_key).difference(expected_keys))
    if missing or unexpected:
        raise RuntimeError(
            "evaluation shard coverage mismatch: "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )

    records = [by_key[key][1] for key in expected_keys if by_key[key][0] == "records"]
    skipped = [by_key[key][1] for key in expected_keys if by_key[key][0] == "skipped"]
    return records, skipped
