#!/usr/bin/env python3
"""Summarize fixed and RINO-exact Qwen subset results across seeds/backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    datasets = ("nyu", "ibims1", "scannet", "vkitti2", "diode", "sintel")
    conditions = tuple(f"diffusers_seed{seed}" for seed in range(5)) + ("musubi_seed0",)
    rows = []
    for condition in conditions:
        for dataset in datasets:
            path = args.root / condition / dataset / "rino_parity_report.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            report = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "condition": condition,
                "dataset": dataset,
                "fixed_mean": float(report["fixed_release"]["per_image_mean"]["mean_angle"]),
                "rino_exact_mean": float(report["rino_exact"]["per_image_mean"]["mean_angle"]),
                "rino_convention": report["rino_exact"]["convention"],
                "per_image_oracle_mean": float(report["per_image_axis_oracle"]["metrics"]["mean_angle"]),
                "report": str(path.resolve()),
            })
    summary = []
    for dataset in datasets:
        values = [row["fixed_mean"] for row in rows if row["dataset"] == dataset and row["condition"].startswith("diffusers_seed")]
        rino_values = [row["rino_exact_mean"] for row in rows if row["dataset"] == dataset and row["condition"].startswith("diffusers_seed")]
        summary.append({
            "dataset": dataset,
            "diffusers_fixed_mean": float(np.mean(values)),
            "diffusers_fixed_std": float(np.std(values, ddof=1)),
            "diffusers_fixed_min": float(np.min(values)),
            "diffusers_fixed_max": float(np.max(values)),
            "rino_exact_mean": float(np.mean(rino_values)),
            "rino_exact_std": float(np.std(rino_values, ddof=1)),
        })
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {"status": "PASS", "rows": rows, "summary": summary}
    (args.output_root / "qwen_normal_robustness.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Qwen surface-normal robustness", "",
        "| Dataset | Fixed mean ± seed std ↓ | Range ↓ | RINO-exact mean ± seed std ↓ |",
        "|---|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {row['diffusers_fixed_mean']:.3f} ± {row['diffusers_fixed_std']:.3f} | "
            f"{row['diffusers_fixed_min']:.3f}–{row['diffusers_fixed_max']:.3f} | "
            f"{row['rino_exact_mean']:.3f} ± {row['rino_exact_std']:.3f} |"
        )
    (args.output_root / "qwen_normal_robustness.md").write_text("\n".join(lines) + "\n")
    print(f"PASS rows={len(rows)} output={args.output_root}")


if __name__ == "__main__":
    main()
