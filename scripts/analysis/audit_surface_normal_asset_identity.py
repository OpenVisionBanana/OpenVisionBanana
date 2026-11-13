#!/usr/bin/env python3
"""Compare two prepared surface-normal dataset trees byte-for-byte."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def semantic_equal(left: Path, right: Path, relative: str) -> tuple[bool, str]:
    if left.suffix == right.suffix == ".npy":
        a = np.load(left, allow_pickle=False)
        b = np.load(right, allow_pickle=False)
        return (a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), "npy_array")
    if left.suffix.lower() in {".png", ".jpg", ".jpeg"} and right.suffix.lower() == left.suffix.lower():
        mode = "L" if relative.startswith("masks/") else "RGB"
        a = np.asarray(Image.open(left).convert(mode))
        b = np.asarray(Image.open(right).convert(mode))
        return (a.shape == b.shape and np.array_equal(a, b), "image_pixels")
    if left.stat().st_size != right.stat().st_size:
        return False, "size"
    return digest(left) == digest(right), "sha256"


def members(root: Path, *, ignore_manifests: bool) -> dict[str, Path]:
    selected = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        is_flat_reference = path.name.endswith(("_img.png", "_normal.npy"))
        if (
            relative.parts[0] in {"images", "normals", "masks"}
            or is_flat_reference
            or (path.suffix == ".txt" and not ignore_manifests)
        ):
            selected[str(relative)] = path
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-right-superset",
        action="store_true",
        help="Pass when every left file is an identical member of the right tree.",
    )
    parser.add_argument(
        "--ignore-manifests",
        action="store_true",
        help="Compare data members only; full archives necessarily have longer split manifests than prefix references.",
    )
    args = parser.parse_args()

    left = members(Path(args.left), ignore_manifests=args.ignore_manifests)
    right = members(Path(args.right), ignore_manifests=args.ignore_manifests)
    common = sorted(left.keys() & right.keys())
    mismatches = []
    for name in common:
        equal, comparison = semantic_equal(left[name], right[name], name)
        if not equal:
            mismatches.append({
                "path": name,
                "reason": comparison,
                "left_sha256": digest(left[name]),
                "right_sha256": digest(right[name]),
            })
    report = {
        "left": str(Path(args.left).resolve()),
        "right": str(Path(args.right).resolve()),
        "left_files": len(left),
        "right_files": len(right),
        "common_files": len(common),
        "identical_common_files": len(common) - len(mismatches),
        "mismatches": mismatches,
        "left_only": sorted(left.keys() - right.keys()),
        "right_only": sorted(right.keys() - left.keys()),
        "pass": not mismatches and (
            set(left) <= set(right) if args.allow_right_superset else set(left) == set(right)
        ),
        "comparison_mode": "left_subset" if args.allow_right_superset else "exact",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
