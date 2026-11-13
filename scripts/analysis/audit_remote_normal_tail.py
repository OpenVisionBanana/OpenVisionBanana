#!/usr/bin/env python3
"""Compare a frozen benchmark-list tail against the upstream Marigold files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def identical(reference: Path, prepared: Path) -> bool:
    if reference.suffix == ".npy":
        left = np.load(reference, allow_pickle=False)
        right = np.load(prepared, allow_pickle=False)
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    left = np.asarray(Image.open(reference).convert("RGB"))
    right = np.asarray(Image.open(prepared).convert("RGB"))
    return left.shape == right.shape and np.array_equal(left, right)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--pair-list", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [line.split() for line in args.pair_list.read_text().splitlines() if line.strip()]
    if args.start < 0 or args.count <= 0 or len(rows[args.start:args.start + args.count]) != args.count:
        parser.error("invalid or incomplete requested tail window")
    mismatches = []
    for rgb, normal, *_ in rows[args.start:args.start + args.count]:
        for kind, relative in (("images", rgb), ("normals", normal)):
            reference = args.reference_root / relative
            prepared = args.prepared_root / kind / relative
            if not reference.is_file() or not prepared.is_file():
                mismatches.append({"path": relative, "reason": "missing", "reference": str(reference), "prepared": str(prepared)})
            elif not identical(reference, prepared):
                mismatches.append({"path": relative, "reason": "content_mismatch"})
    report = {
        "pass": not mismatches,
        "pair_list": str(args.pair_list),
        "start": args.start,
        "count": args.count,
        "files_checked": 2 * args.count,
        "mismatches": mismatches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if mismatches:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
