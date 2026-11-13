#!/usr/bin/env python3
"""Partition RINO-parity DIODE error by official scene/scan identifiers."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["dataset"] == "diode"
    rows = report["rino_exact"]["per_sample"]
    assert len(rows) == report["sample_count"] == 325
    by_scan: dict[str, list[float]] = defaultdict(list)
    by_scene: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        parts = row["sample_id"].split("_")
        assert len(parts) >= 4, row["sample_id"]
        value = float(row["mean_angle"])
        by_scene[parts[0]].append(value)
        by_scan["_".join(parts[:2])].append(value)
    full = statistics.fmean(float(row["mean_angle"]) for row in rows)
    scenes = {key: {"images": len(values), "mean_angle": statistics.fmean(values),
                    "contribution_to_full_mean": sum(values) / len(rows)}
              for key, values in sorted(by_scene.items())}
    scans = {key: {"images": len(values), "mean_angle": statistics.fmean(values),
                   "contribution_to_full_mean": sum(values) / len(rows)}
             for key, values in sorted(by_scan.items())}
    payload = {"status": "PASS", "full_mean_angle": full, "scenes": scenes, "scans": scans,
               "source": str(args.report)}
    assert abs(full - report["rino_exact"]["per_image_mean"]["mean_angle"]) < 1e-6
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# DIODE Qwen/RINO scene error", "",
             f"Full-set mean angle: {full:.3f}° across {len(rows)} images.", "",
             "| Scene/scan | Images | Mean angle | Contribution to full mean |",
             "|---|---:|---:|---:|"]
    for key, value in sorted(scans.items(), key=lambda item: item[1]["mean_angle"], reverse=True):
        lines.append(f"| {key} | {value['images']} | {value['mean_angle']:.3f}° | "
                     f"{value['contribution_to_full_mean']:.3f}° |")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"PASS scans={len(scans)} full_mean_angle={full:.3f}")


if __name__ == "__main__":
    main()
