#!/usr/bin/env python3
"""Validate PD12M WebDataset shards and their manifests."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from .prepare_pd12m import DEFAULT_MAX_MEMBER_BYTES, file_sha256, iter_jsonl
except ImportError:  # direct script execution
    from prepare_pd12m import DEFAULT_MAX_MEMBER_BYTES, file_sha256, iter_jsonl


def validate_shard(path: Path, expected: dict[str, Any], max_member_bytes: int) -> int:
    groups: dict[str, dict[str, bytes]] = {}
    with tarfile.open(path, "r") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if member.size > max_member_bytes:
                raise ValueError(f"{path}: oversized member {member.name}: {member.size}")
            key, dot, suffix = member.name.rpartition(".")
            if not dot or suffix not in {"jpg", "txt", "json"}:
                raise ValueError(f"{path}: unexpected member {member.name}")
            if suffix in groups.setdefault(key, {}):
                raise ValueError(f"{path}: duplicate {suffix} member for key {key}")
            extracted = tar.extractfile(member)
            if extracted is None:
                raise ValueError(f"{path}: cannot extract {member.name}")
            groups[key][suffix] = extracted.read()
    for key, parts in groups.items():
        if set(parts) != {"jpg", "txt", "json"}:
            raise ValueError(f"{path}: key {key} has {sorted(parts)}, expected jpg/txt/json")
        caption = parts["txt"].decode("utf-8").strip()
        if not caption:
            raise ValueError(f"{path}: key {key} has an empty caption")
        metadata = json.loads(parts["json"].decode("utf-8"))
        if metadata.get("id") != key or metadata.get("task") != "text_to_image" or metadata.get("source_dataset") != "pd12m":
            raise ValueError(f"{path}: key {key} has inconsistent metadata")
        with Image.open(io.BytesIO(parts["jpg"])) as image:
            image.load()
            if image.mode != "RGB" or image.width <= 0 or image.height <= 0:
                raise ValueError(f"{path}: key {key} is not a valid RGB image")
    if len(groups) != int(expected["sample_count"]):
        raise ValueError(f"{path}: manifest count={expected['sample_count']} actual={len(groups)}")
    if path.stat().st_size != int(expected["byte_size"]):
        raise ValueError(f"{path}: byte-size mismatch")
    actual_checksum = file_sha256(path)
    if actual_checksum != expected["sha256"]:
        raise ValueError(f"{path}: SHA-256 mismatch")
    return len(groups)


def validate_dataset(root: Path, max_member_bytes: int | None = None) -> dict[str, int]:
    manifest_path = root / "shards.jsonl"
    info_path = root / "dataset_info.json"
    if not manifest_path.is_file() or not info_path.is_file():
        raise FileNotFoundError(f"Missing shards.jsonl or dataset_info.json under {root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if max_member_bytes is None:
        max_member_bytes = int(info.get("max_member_bytes", DEFAULT_MAX_MEMBER_BYTES))
    if max_member_bytes <= 0:
        raise ValueError("max_member_bytes must be positive")
    manifests = list(iter_jsonl(manifest_path))
    if not manifests:
        raise ValueError("shards.jsonl is empty")
    seen_shards: set[str] = set()
    total = 0
    for expected in manifests:
        shard_name = expected["shard"]
        if shard_name in seen_shards:
            raise ValueError(f"duplicate shard manifest entry: {shard_name}")
        seen_shards.add(shard_name)
        shard_path = root / "shards" / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        total += validate_shard(shard_path, expected, max_member_bytes)
    if total != int(info["successful_samples"]):
        raise ValueError(f"dataset_info successful_samples={info['successful_samples']} actual={total}")
    if len(manifests) != int(info["num_shards"]):
        raise ValueError(f"dataset_info num_shards={info['num_shards']} actual={len(manifests)}")
    return {"shards": len(manifests), "samples": total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--max-member-bytes",
        type=int,
        default=None,
        help=f"Maximum tar member size (default: dataset metadata or {DEFAULT_MAX_MEMBER_BYTES} bytes)",
    )
    args = parser.parse_args()
    summary = validate_dataset(args.root, args.max_member_bytes)
    print(f"PD12M validation ok: shards={summary['shards']:,}, samples={summary['samples']:,}")


if __name__ == "__main__":
    main()
