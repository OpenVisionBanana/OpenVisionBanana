"""Frozen sample-manifest support shared by geometry evaluators."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence, TypeVar


T = TypeVar("T")
SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_sample_manifest(
    samples: Sequence[T],
    manifest_path: str | Path | None,
    *,
    dataset: str,
) -> tuple[list[T], dict[str, Any] | None]:
    """Filter samples to one exact ordered, versioned JSONL manifest.

    Sample objects must expose ``sample_id`` and ``output_stem``. The manifest
    order is authoritative, and every selected identifier must occur exactly
    once in the resolved full dataset.
    """

    if manifest_path is None:
        return list(samples), None
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Geometry sample manifest not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Manifest row is not an object at {path}:{line_number}")
            if row.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported geometry manifest schema at {path}:{line_number}: "
                    f"{row.get('schema_version')!r}"
                )
            if row.get("dataset") != dataset:
                raise ValueError(
                    f"Manifest dataset mismatch at {path}:{line_number}: "
                    f"expected {dataset!r}, found {row.get('dataset')!r}"
                )
            if not isinstance(row.get("sample_id"), str):
                raise ValueError(f"Missing sample_id at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Geometry sample manifest is empty: {path}")

    manifest_ids = [row["sample_id"] for row in rows]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError(f"Duplicate sample IDs in geometry manifest: {path}")

    by_id: dict[str, T] = {}
    for sample in samples:
        sample_id = str(getattr(sample, "sample_id"))
        if sample_id in by_id:
            raise ValueError(f"Resolved dataset contains duplicate sample ID: {sample_id}")
        by_id[sample_id] = sample
    missing = [sample_id for sample_id in manifest_ids if sample_id not in by_id]
    if missing:
        raise ValueError(
            f"Geometry manifest contains {len(missing)} IDs absent from {dataset}: "
            + ", ".join(missing[:10])
        )

    selected = [by_id[sample_id] for sample_id in manifest_ids]
    for row, sample in zip(rows, selected):
        expected_stem = row.get("output_stem")
        actual_stem = str(getattr(sample, "output_stem"))
        if expected_stem is not None and expected_stem != actual_stem:
            raise ValueError(
                f"Geometry manifest output_stem mismatch for {row['sample_id']}: "
                f"expected {expected_stem!r}, resolved {actual_stem!r}"
            )

    protocol_ids = {str(row.get("protocol_id", "")) for row in rows}
    if len(protocol_ids) != 1 or not next(iter(protocol_ids)):
        raise ValueError(f"Manifest must contain one non-empty protocol_id: {path}")
    metadata = {
        "enabled": True,
        "protocol_id": next(iter(protocol_ids)),
        "manifest_path": str(path),
        "manifest_sha256": _sha256(path),
        "sample_count": len(selected),
        "full_resolved_sample_count": len(samples),
        "ordered_sample_ids_sha256": hashlib.sha256(
            "".join(f"{sample_id}\n" for sample_id in manifest_ids).encode("utf-8")
        ).hexdigest(),
    }
    return selected, metadata
