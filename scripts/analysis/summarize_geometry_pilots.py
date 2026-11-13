#!/usr/bin/env python3
"""Fail-closed, paired first-32 summary for the geometry training pilots."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


NORMAL_ARMS = (
    "original_unmasked",
    "camera_facing_v4_unmasked",
    "camera_facing_v4_masked",
    "camera_facing_v4_masked_angular005",
    "camera_facing_v4_masked_unit005",
    "camera_facing_v4_masked_real60",
    "camera_facing_v4_masked_angular005_real60",
)
NORMAL_DATASETS = ("nyu", "ibims1", "scannet", "vkitti2", "diode", "sintel")
DEPTH_ARMS = ("rgb_baseline", "rgb_logmetric_aux002", "relative_inverse_target")
DEPTH_DATASETS = ("kitti_eigen", "eth3d", "diode")


def one(root: Path, pattern: str) -> Path:
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one {pattern} in {root}, got {len(paths)}")
    return paths[0]


def collect(root: Path, arms: tuple[str, ...], datasets: tuple[str, ...],
            folder: str, metric: str, reported_path: tuple[str, ...]) -> dict:
    out = {}
    for arm in arms:
        out[arm] = {}
        for dataset in datasets:
            directory = root / arm / folder / dataset
            result = json.loads(one(directory, "*_results.json").read_text())
            if result["images_evaluated"] != 32 or result.get("images_skipped", 0) != 0:
                raise ValueError(f"Incomplete evaluation: {directory}")
            rows = [json.loads(line) for line in one(directory, "*_per_sample.jsonl").read_text().splitlines() if line]
            by_id = {str(row["sample_id"]): float(row[metric]) for row in rows}
            if len(rows) != 32 or len(by_id) != 32:
                raise ValueError(f"Expected 32 unique samples: {directory}")
            reported = result
            for key in reported_path:
                reported = reported[key]
            mean = statistics.fmean(by_id.values())
            if abs(mean - float(reported)) > 1e-4:
                raise ValueError(f"Reported/sample mean mismatch: {directory}: {mean} vs {reported}")
            out[arm][dataset] = {"mean": mean, "per_sample": by_id,
                                 "result": str(one(directory, "*_results.json"))}
    for dataset in datasets:
        expected = set(out[arms[0]][dataset]["per_sample"])
        for arm in arms[1:]:
            if set(out[arm][dataset]["per_sample"]) != expected:
                raise ValueError(f"Non-paired samples: {dataset}, {arm}")
    return out


def summarize(values: dict, arms: tuple[str, ...], datasets: tuple[str, ...]) -> dict:
    baseline = arms[0]
    return {
        arm: {
            "macro_mean": statistics.fmean(values[arm][d]["mean"] for d in datasets),
            "datasets": {
                dataset: {
                    "mean": values[arm][dataset]["mean"],
                    "delta_vs_baseline": statistics.fmean(
                        values[arm][dataset]["per_sample"][sid] - values[baseline][dataset]["per_sample"][sid]
                        for sid in values[baseline][dataset]["per_sample"]
                    ),
                    "result": values[arm][dataset]["result"],
                }
                for dataset in datasets
            },
        }
        for arm in arms
    }


def table(summary: dict, arms: tuple[str, ...], datasets: tuple[str, ...], title: str) -> list[str]:
    lines = [f"## {title}", "", "Lower is better. Values are first-32 per-image means; deltas are paired to the first arm.", "",
             "| Arm | Macro | " + " | ".join(datasets) + " |",
             "|---|---:|" + "---:|" * len(datasets)]
    for arm in arms:
        row = summary[arm]
        lines.append("| " + arm + f" | {row['macro_mean']:.4f} | " + " | ".join(
            f"{row['datasets'][d]['mean']:.4f} ({row['datasets'][d]['delta_vs_baseline']:+.4f})"
            for d in datasets
        ) + " |")
    return lines + [""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--depth-folder", default="evaluation_marigold_ppd_first32_training_aligned")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    normal = summarize(collect(args.normal_root, NORMAL_ARMS, NORMAL_DATASETS,
                               "evaluation_geometry_v6_first32", "mean_angle",
                               ("metrics", "mean_angle")), NORMAL_ARMS, NORMAL_DATASETS)
    depth = summarize(collect(args.depth_root, DEPTH_ARMS, DEPTH_DATASETS,
                              args.depth_folder, "marigold_v2_ppd_abs_rel",
                              ("marigold_v2_ppd_metrics", "abs_rel")), DEPTH_ARMS, DEPTH_DATASETS)
    payload = {"status": "PASS", "scope": "paired first-32 diagnostic only",
               "normal_mean_angle_degrees": normal, "depth_ppd_abs_rel_fraction": depth}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Geometry pilot summary", "",
             "These are bounded diagnostic subsets, not full-benchmark or publication results.", ""]
    lines += table(normal, NORMAL_ARMS, NORMAL_DATASETS, "Surface normals (mean angle, degrees)")
    lines += table(depth, DEPTH_ARMS, DEPTH_DATASETS, "Depth (Marigold-like PPD AbsRel fraction)")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"PASS normal_arms={len(NORMAL_ARMS)} depth_arms={len(DEPTH_ARMS)} output={args.output}")


if __name__ == "__main__":
    main()
