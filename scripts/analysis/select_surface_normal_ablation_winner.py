#!/usr/bin/env python3
"""Select the best normal-contract ablation with paired bootstrap diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LABELS = (
    "original_unmasked",
    "camera_facing_v4_unmasked",
    "camera_facing_v4_masked",
    "camera_facing_v4_masked_angular005",
)
DATASETS = ("nyu", "ibims1", "scannet", "vkitti2", "diode", "sintel")


def one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern} in {directory}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    per_sample: dict[tuple[str, str], dict[str, float]] = {}
    rows = []
    for label in LABELS:
        for dataset in DATASETS:
            root = args.ablation_root / label / "evaluation_geometry_v6_first32" / dataset
            result_path = one(root, "*_results.json")
            record_path = one(root, "*_per_sample.jsonl")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result["normal_evaluation"]["protocol_version"] != "geometry-eval-v6":
                raise ValueError(f"Wrong protocol in {result_path}")
            records = [json.loads(line) for line in record_path.read_text().splitlines() if line.strip()]
            by_id = {str(record["sample_id"]): float(record["mean_angle"]) for record in records}
            if len(by_id) != 32 or len(records) != 32:
                raise ValueError(f"Expected 32 unique samples in {record_path}")
            per_sample[label, dataset] = by_id
            rows.append({
                "label": label,
                "dataset": dataset,
                "mean_angular_error": float(result["marigold_v1_1_metrics"]["mean_angular_error"]),
                "result": str(result_path.resolve()),
            })
    baseline_ids = {dataset: tuple(per_sample["original_unmasked", dataset]) for dataset in DATASETS}
    for label in LABELS[1:]:
        for dataset in DATASETS:
            if tuple(per_sample[label, dataset]) != baseline_ids[dataset]:
                raise ValueError(f"Paired sample IDs/order differ for {label}/{dataset}")

    macro = {
        label: float(np.mean([row["mean_angular_error"] for row in rows if row["label"] == label]))
        for label in LABELS
    }
    winner = min(macro, key=macro.get)
    rng = np.random.default_rng(42)
    paired = []
    for label in LABELS[1:]:
        differences = []
        for dataset in DATASETS:
            ids = baseline_ids[dataset]
            differences.append(np.asarray([
                per_sample[label, dataset][sample_id] - per_sample["original_unmasked", dataset][sample_id]
                for sample_id in ids
            ], dtype=np.float64))
        boot = np.empty(args.bootstrap_replicates, dtype=np.float64)
        for index in range(args.bootstrap_replicates):
            boot[index] = np.mean([
                values[rng.integers(0, len(values), size=len(values))].mean()
                for values in differences
            ])
        observed = float(np.mean([values.mean() for values in differences]))
        paired.append({
            "label": label,
            "delta_vs_original_degrees": observed,
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "probability_better_than_original": float(np.mean(boot < 0.0)),
        })

    transformer = args.ablation_root / winner / "transformer"
    if not (transformer / "config.json").is_file():
        raise FileNotFoundError(transformer / "config.json")
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "PASS", "selection_rule": "lowest unweighted six-dataset macro mean angular error",
        "winner": winner, "winner_transformer": str(transformer.resolve()),
        "macro_mean_angular_error": macro, "paired_bootstrap": paired, "rows": rows,
    }
    (args.output_root / "ablation_winner.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Surface-normal ablation selection", "", f"Winner: `{winner}`", "", "| Arm | Macro mean angle ↓ |", "|---|---:|"]
    lines += [f"| {label} | {macro[label]:.3f} |" for label in LABELS]
    by_dataset = {(row["label"], row["dataset"]): row["mean_angular_error"] for row in rows}
    lines += ["", "| Dataset | Original | Camera-facing | Camera-facing + mask | Camera-facing + mask + angular |",
              "|---|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        lines.append("| " + dataset + " | " + " | ".join(
            f"{by_dataset[label, dataset]:.3f}" for label in LABELS
        ) + " |")
    lines += ["", "| Arm | Δ vs original ↓ | 95% paired bootstrap CI | P(better) |", "|---|---:|---:|---:|"]
    for item in paired:
        lines.append(f"| {item['label']} | {item['delta_vs_original_degrees']:.3f} | [{item['ci95'][0]:.3f}, {item['ci95'][1]:.3f}] | {item['probability_better_than_original']:.3f} |")
    (args.output_root / "ablation_winner.md").write_text("\n".join(lines) + "\n")
    print(f"PASS winner={winner} output={args.output_root}")


if __name__ == "__main__":
    main()
