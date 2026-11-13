#!/usr/bin/env python3
"""Fail-closed paired summaries for full-split geometry training pilots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NORMAL_ARMS = ("camera_facing_v4_masked_angular005", "camera_facing_v4_masked_angular005_real60")
NORMAL_DATASETS = {"nyu": 654, "ibims1": 100, "scannet": 300, "vkitti2": 1000, "diode": 325, "sintel": 627}
DEPTH_ARMS = ("rgb_baseline", "rgb_logmetric_aux002", "relative_inverse_target")
DEPTH_DATASETS = {"kitti_eigen": 652, "eth3d": 454, "diode": 771}


def one(directory: Path, pattern: str) -> Path:
    files = list(directory.glob(pattern))
    if len(files) != 1:
        raise ValueError(f"Expected exactly one {pattern} in {directory}; found {len(files)}")
    return files[0]


def collect(root: Path, arm: str, dataset: str, lane: str, expected: int) -> dict:
    folder = "evaluation_geometry_v6_full" if lane == "normals" else "evaluation_marigold_ppd_full_training_aligned"
    directory = root / arm / folder / dataset
    result = json.loads(one(directory, "*_results.json").read_text())
    if result["images_evaluated"] != expected or result.get("images_skipped", 0) != 0:
        raise ValueError(f"Incomplete full evaluation: {directory}")
    if lane == "depth":
        required_encoding = "relative_inverse_grayscale" if arm == "relative_inverse_target" else "vision_banana_rgb"
        if result["depth_encoding"] != required_encoding:
            raise ValueError(f"Depth representation mismatch: {directory}")
        from common.geometry_protocol_prompts import CANONICAL_DEPTH_PROMPT, CONCISE_INVERSE_DEPTH_PROMPT
        required_prompt = CONCISE_INVERSE_DEPTH_PROMPT if arm == "relative_inverse_target" else CANONICAL_DEPTH_PROMPT
        if result["generation"]["prompt"] != required_prompt:
            raise ValueError(f"Depth prompt mismatch: {directory}")
    metric = "mean_angle" if lane == "normals" else "marigold_v2_ppd_abs_rel"
    reported = result["metrics"]["mean_angle"] if lane == "normals" else result["marigold_v2_ppd_metrics"]["abs_rel"]
    rows = [json.loads(line) for line in one(directory, "*_per_sample.jsonl").read_text().splitlines() if line]
    by_id = {str(row["sample_id"]): float(row[metric]) for row in rows}
    if len(rows) != expected or len(by_id) != expected:
        raise ValueError(f"Incomplete or duplicate per-sample rows: {directory}")
    mean = float(np.mean(list(by_id.values())))
    if abs(mean - float(reported)) > 1e-4:
        raise ValueError(f"Reported/sample mismatch: {directory}: {mean} vs {reported}")
    return {"mean": mean, "per_sample": by_id, "result": str(one(directory, "*_results.json"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("normals", "depth"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = NORMAL_ARMS if args.lane == "normals" else DEPTH_ARMS
    datasets = NORMAL_DATASETS if args.lane == "normals" else DEPTH_DATASETS
    rows = []
    for dataset, expected in datasets.items():
        data = {arm: collect(args.root, arm, dataset, args.lane, expected) for arm in arms}
        baseline_ids = set(data[arms[0]]["per_sample"])
        if any(set(data[arm]["per_sample"]) != baseline_ids for arm in arms[1:]):
            raise ValueError(f"Non-paired full samples: {dataset}")
        for arm in arms:
            deltas = np.asarray([
                data[arm]["per_sample"][sid] - data[arms[0]]["per_sample"][sid]
                for sid in sorted(baseline_ids)
            ], dtype=np.float64)
            rng = np.random.default_rng(20260923)
            bootstrap = np.mean(deltas[rng.integers(0, len(deltas), (2000, len(deltas)))], axis=1)
            rows.append({
                "dataset": dataset,
                "arm": arm,
                "samples": expected,
                "mean": data[arm]["mean"],
                "delta_vs_first_arm": float(np.mean(deltas)),
                "descriptive_image_bootstrap_95pct_delta": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
                "fraction_worse_than_first_arm": float(np.mean(deltas > 0)),
                "result": data[arm]["result"],
            })
    macro = {arm: float(np.mean([row["mean"] for row in rows if row["arm"] == arm])) for arm in arms}
    payload = {
        "status": "PASS", "lane": args.lane, "scope": "full prepared benchmark splits",
        "note": "Image bootstrap is descriptive; scenes are correlated. Short-trained pilots, not frozen final checkpoint results.",
        "macro_mean": macro, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"PASS {args.lane}: {len(rows)} full-split rows -> {args.output}")


if __name__ == "__main__":
    main()
