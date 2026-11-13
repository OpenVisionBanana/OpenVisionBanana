#!/usr/bin/env python3
"""Normalize a prepared member's metadata.jsonl to the canonical pool schema.

Some converters emit the diffusers-style dialect

    {"file_name": ..., "edited_file_name": ..., "mask_file_name": ..., "prompt": ...}

while the prepared pool (and validate_dataset_specs.py / convert_metadata.py /
convert_metadata.py) expects

    {"conditioning_image": ..., "image": ..., "valid_mask": ..., "text": ...}

plus top-level "task"/"source_dataset". This rewrites the former into the
latter in place. Only key names change; the images on disk are untouched.

Idempotent: already-canonical rows pass through unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RENAMES = {
    "file_name": "conditioning_image",
    "edited_file_name": "image",
    "mask_file_name": "valid_mask",
    "prompt": "text",
}


def normalize_row(row: dict) -> tuple[dict, bool]:
    changed = False
    for old, new in RENAMES.items():
        if old in row and new not in row:
            row[new] = row.pop(old)
            changed = True
    meta = row.get("metadata") or {}
    # Lift the fields the pool reads from the top level.
    if not row.get("task") and meta.get("task"):
        row["task"] = meta["task"]
        changed = True
    if not row.get("source_dataset") and meta.get("dataset"):
        row["source_dataset"] = meta["dataset"]
        changed = True
    if not row.get("id") and row.get("conditioning_image"):
        row["id"] = Path(row["conditioning_image"]).stem
        changed = True
    return row, changed


def normalize_file(path: Path, dry_run: bool = False) -> tuple[int, int]:
    rows, changed_rows = [], 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row, changed = normalize_row(json.loads(line))
            rows.append(row)
            changed_rows += int(changed)
    if changed_rows and not dry_run:
        # Write via a temp file so an interrupted run cannot truncate metadata.
        tmp = path.with_suffix(path.suffix + ".partial")
        tmp.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
        )
        tmp.replace(path)
    return len(rows), changed_rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("members", nargs="+", type=Path,
                   help="Member dirs (or metadata.jsonl paths) to normalize.")
    p.add_argument("--dry-run", action="store_true", help="Report without rewriting.")
    args = p.parse_args()

    status = 0
    for member in args.members:
        path = member if member.is_file() else member / "metadata.jsonl"
        if not path.is_file():
            print(f"  {member}: no metadata.jsonl, skipped", file=sys.stderr)
            status = 1
            continue
        total, changed = normalize_file(path, args.dry_run)
        verb = "would rewrite" if args.dry_run else "rewrote"
        note = f"{verb} {changed}/{total} rows" if changed else f"already canonical ({total} rows)"
        print(f"  {path.parent.name}: {note}")
    return status


if __name__ == "__main__":
    sys.exit(main())