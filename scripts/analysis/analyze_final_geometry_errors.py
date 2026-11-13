#!/usr/bin/env python3
"""Paired, sample-level error audit for final OVB evaluations.

Diagnostic only: never selects a checkpoint or changes primary metrics.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


PAIRS = (
    ("4b_nonzero_checkpoint100000", "4b_zeroshot_checkpoint87500_95000_w25"),
    ("9b_nonzero_checkpoint75000", "9b_zeroshot_checkpoint100000"),
)
LANES = {
    "normals": ("surface_normals_geometry_v6", "mean_angle"),
    "depth": ("depth_marigold_v2_ppd", "marigold_v2_ppd_abs_rel"),
}


def records(path: Path) -> dict[str, dict]:
    matches = glob.glob(str(path / "*_per_sample.jsonl"))
    if len(matches) != 1:
        raise ValueError(f"Expected one per-sample file in {path}, got {matches}")
    rows = [json.loads(line) for line in Path(matches[0]).read_text().splitlines() if line]
    indexed = {str(row["sample_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate sample IDs in {matches[0]}")
    return indexed


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    pos = fraction * (len(ordered) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    return ordered[low] + (pos - low) * (ordered[high] - ordered[low])


def scene(row: dict) -> str:
    """Recover a useful scene/drive group from either benchmark's sample row."""
    sample_id = str(row["sample_id"])
    parts = sample_id.split("__")
    if len(parts) >= 3:
        return parts[1]
    image_path = str(row.get("image_path", ""))
    for pattern in (r"(2011_\d\d_\d\d_drive_\d+_sync)", r"(scene\d+_\d+)",
                    r"(Scene\d+)"):
        match = re.search(pattern, image_path)
        if match:
            return match.group(1)
    return "ungrouped"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    findings: list[dict] = []
    lines = ["# Corrected final-model paired error audit", "",
             "Positive mean delta means the non-zero variant is worse than its zero-shot counterpart.", "",
             "| Lane | Family | Dataset | Paired images | Mean delta | P90 delta | Non-zero worse |",
             "|---|---|---|---:|---:|---:|---:|"]
    for lane, (folder, metric) in LANES.items():
        lane_root = args.result_root / folder
        for nonzero, zeroshot in PAIRS:
            datasets = sorted(p.name for p in (lane_root / nonzero).iterdir() if p.is_dir())
            for dataset in datasets:
                left = records(lane_root / nonzero / dataset)
                right = records(lane_root / zeroshot / dataset)
                if left.keys() != right.keys():
                    raise ValueError(f"Sample mismatch: {lane} {dataset} {nonzero} vs {zeroshot}")
                paired = []
                by_scene: dict[str, list[float]] = defaultdict(list)
                for sample_id in sorted(left):
                    a, b = float(left[sample_id][metric]), float(right[sample_id][metric])
                    if not math.isfinite(a) or not math.isfinite(b):
                        raise ValueError(f"Non-finite metric: {lane} {dataset} {sample_id}")
                    delta = a - b
                    paired.append({"sample_id": sample_id, "nonzero": a, "zeroshot": b,
                                   "delta_nonzero_minus_zeroshot": delta})
                    by_scene[scene(left[sample_id])].append(delta)
                deltas = [row["delta_nonzero_minus_zeroshot"] for row in paired]
                unit = "degrees" if lane == "normals" else "AbsRel fraction"
                result = {
                    "lane": lane, "dataset": dataset, "nonzero": nonzero, "zeroshot": zeroshot,
                    "metric": metric, "unit": unit, "paired_images": len(paired),
                    "mean_delta": statistics.fmean(deltas), "p90_delta": quantile(deltas, 0.90),
                    "fraction_nonzero_worse": sum(value > 0 for value in deltas) / len(deltas),
                    "worst_nonzero_regressions": sorted(
                        paired, key=lambda row: row["delta_nonzero_minus_zeroshot"], reverse=True
                    )[:20],
                    "best_nonzero_improvements": sorted(
                        paired, key=lambda row: row["delta_nonzero_minus_zeroshot"]
                    )[:20],
                    "scene_mean_deltas": {
                        key: {"images": len(values), "mean_delta": statistics.fmean(values)}
                        for key, values in sorted(by_scene.items())
                    },
                }
                findings.append(result)
                lines.append(
                    f"| {lane} | {nonzero[:2]} | {dataset} | {len(paired)} | "
                    f"{result['mean_delta']:.4f} | {result['p90_delta']:.4f} | "
                    f"{100 * result['fraction_nonzero_worse']:.1f}% |"
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "PASS", "findings": findings}, indent=2) + "\n")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"PASS paired audits={len(findings)} output={args.output}")


if __name__ == "__main__":
    main()
