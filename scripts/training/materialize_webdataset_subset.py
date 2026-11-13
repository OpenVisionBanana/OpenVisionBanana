#!/usr/bin/env python3
"""Materialize a small image/text subset from a local WebDataset tar."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def materialize(shard: Path, output: Path, limit: int) -> Path:
    if limit < 1:
        raise ValueError("limit must be positive")
    shard = shard.expanduser().resolve()
    if not shard.is_file():
        raise FileNotFoundError(f"WebDataset shard does not exist: {shard}")
    output = output.expanduser().resolve()
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)

    samples: dict[str, dict[str, tarfile.TarInfo]] = {}
    with tarfile.open(shard, mode="r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            path = Path(member.name)
            suffix = path.suffix.lower()
            if suffix not in IMAGE_SUFFIXES | {".txt", ".json"}:
                continue
            samples.setdefault(path.stem, {})[suffix] = member
            complete = [key for key, parts in samples.items() if ".txt" in parts and IMAGE_SUFFIXES & parts.keys()]
            if len(complete) >= limit:
                break

        complete = sorted(key for key, parts in samples.items() if ".txt" in parts and IMAGE_SUFFIXES & parts.keys())[
            :limit
        ]
        if len(complete) < limit:
            raise RuntimeError(f"Requested {limit} samples but found only {len(complete)} complete image/text pairs")

        rows = []
        for key in complete:
            parts = samples[key]
            image_suffix = next(suffix for suffix in sorted(IMAGE_SUFFIXES) if suffix in parts)
            image_name = f"images/{key}{image_suffix}"
            image_stream = archive.extractfile(parts[image_suffix])
            text_stream = archive.extractfile(parts[".txt"])
            if image_stream is None or text_stream is None:
                raise RuntimeError(f"Could not read WebDataset sample {key}")
            (output / image_name).write_bytes(image_stream.read())
            prompt = text_stream.read().decode("utf-8", errors="replace").strip()
            row = {
                "id": key,
                "image": image_name,
                "text": prompt,
                "task": "text_to_image",
                "source_dataset": "pd12m",
            }
            if ".json" in parts:
                metadata_stream = archive.extractfile(parts[".json"])
                if metadata_stream is not None:
                    source_metadata = json.loads(metadata_stream.read().decode("utf-8"))
                    row["source_metadata"] = source_metadata
            rows.append(row)

    metadata = output / "metadata.jsonl"
    temporary = metadata.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()
    print(materialize(args.shard, args.output_dir, args.limit))


if __name__ == "__main__":
    main()
