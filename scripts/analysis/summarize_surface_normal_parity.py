#!/usr/bin/env python3
"""Summarize all completed surface-normal parity reports below one result root."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def nested(data: dict, *keys: str):
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.rglob("rino_parity_report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        rel = path.relative_to(args.root)
        stability = report.get("calibration_stability", {})
        rows.append({
            "run": str(rel.parent),
            "dataset": report.get("dataset"),
            "samples": report.get("sample_count"),
            "fixed_mean_angle": nested(report, "fixed_release", "per_image_mean", "mean_angle"),
            "rino_exact_mean_angle": nested(report, "rino_exact", "per_image_mean", "mean_angle"),
            "rino_convention": nested(report, "rino_exact", "convention"),
            "direct_test5_mean_angle": nested(
                report, "shared_direct_resize_test5", "per_image_mean", "mean_angle"
            ),
            "per_image_oracle_mean_angle": nested(
                report, "per_image_axis_oracle", "metrics", "mean_angle"
            ),
            "erosion3_mean_angle": nested(report, "mask_erosion", "3", "mean_angle"),
            "calibration5_spread": nested(stability, "5", "full_set_mean_angle_spread"),
            "calibration20_spread": nested(stability, "20", "full_set_mean_angle_spread"),
        })

    output = args.output or args.root / "surface_normal_parity_summary.tsv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["run", "dataset", "samples"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
