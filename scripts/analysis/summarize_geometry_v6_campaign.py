#!/usr/bin/env python3
"""Validate and summarize the geometry full and training-ablation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "mean_angular_error", "median_angular_error", "sub11_25_error",
    "sub22_5_error", "sub30_error",
)


def load_result(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["normal_evaluation"]["protocol_version"] != "geometry-eval-v6":
        raise ValueError(f"Wrong protocol in {path}")
    if int(payload["images_evaluated"]) <= 0 or int(payload["images_skipped"]) != 0:
        raise ValueError(f"Incomplete evaluation in {path}")
    return payload


def result_in(directory: Path) -> Path:
    matches = sorted(directory.glob("*_results.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected one result JSON in {directory}, found {len(matches)}")
    return matches[0]


def row(label: str, dataset: str, path: Path) -> dict:
    payload = load_result(path)
    metrics = payload["marigold_v1_1_metrics"]
    return {
        "experiment": label,
        "dataset": dataset,
        "images": int(payload["images_evaluated"]),
        **{name: float(metrics[name]) for name in METRICS},
        "result": str(path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    datasets = ("nyu", "ibims1", "scannet", "vkitti2", "diode", "sintel")
    full_layout = {
        "4b_step100000": args.full_root / "4b_step100000/training_campaign_short_guidance3p5",
        "9b_step75000": args.full_root / "9b_step75000/training_campaign_short_guidance3p5",
        "qwen_rino_prompt_seed0": args.full_root / "qwen_rino_prompt_seed0",
    }
    ablations = (
        "original_unmasked", "camera_facing_v4_unmasked",
        "camera_facing_v4_masked", "camera_facing_v4_masked_angular005",
    )
    rows = []
    for label, root in full_layout.items():
        for dataset in datasets:
            rows.append(row(label, dataset, result_in(root / dataset)))
    for label in ablations:
        for dataset in datasets:
            root = args.ablation_root / label / "evaluation_geometry_v6_first32" / dataset
            rows.append(row(f"ablation:{label}", dataset, result_in(root)))
    if len(rows) != 42:
        raise AssertionError(f"Expected 42 result rows, got {len(rows)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "geometry_v6_campaign_results.json").write_text(
        json.dumps({"status": "PASS", "rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Geometry-v6 campaign results", "",
        "| Experiment | Dataset | N | Mean ↓ | Median ↓ | <11.25° ↑ | <22.5° ↑ | <30° ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in rows:
        lines.append(
            f"| {item['experiment']} | {item['dataset']} | {item['images']} | "
            f"{item['mean_angular_error']:.3f} | {item['median_angular_error']:.3f} | "
            f"{item['sub11_25_error']:.2f} | {item['sub22_5_error']:.2f} | {item['sub30_error']:.2f} |"
        )
    (args.output_root / "geometry_v6_campaign_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"PASS rows={len(rows)} output={args.output_root}")


if __name__ == "__main__":
    main()
