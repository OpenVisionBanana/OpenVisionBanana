#!/usr/bin/env python3
"""Rank surface-normal failures and freeze top-tail sample manifests for review."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worst", type=int, default=20)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    results = sorted(args.result_root.glob("**/*_results.json"))
    if not results:
        raise FileNotFoundError(f"No result JSON files below {args.result_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    lines = [
        "# Surface-normal failure review", "",
        "| Experiment | Dataset | N | Median image | P90 | P95 | P99 | Worst |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result_path in results:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        record_paths = sorted(result_path.parent.glob("*_per_sample.jsonl"))
        if len(record_paths) != 1:
            if args.allow_incomplete:
                continue
            raise ValueError(f"Expected one per-sample file beside {result_path}")
        records = [json.loads(line) for line in record_paths[0].read_text().splitlines() if line.strip()]
        if not records:
            raise ValueError(f"No records in {record_paths[0]}")
        records.sort(key=lambda row: float(row["mean_angle"]), reverse=True)
        rel = result_path.parent.relative_to(args.result_root)
        dataset = str(result["dataset"])
        experiment = str(rel.parent) if rel.name == dataset else str(rel)
        angles = np.asarray([float(record["mean_angle"]) for record in records])
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "__", f"{experiment}__{dataset}").strip("_")
        manifest_path = args.output_root / f"worst_{args.worst}_{safe}.jsonl"
        manifest_rows = []
        for rank, record in enumerate(records[: args.worst], start=1):
            item = {
                "schema_version": 1,
                "protocol_id": f"failure-tail-v1:{safe}:worst{args.worst}",
                "dataset": dataset,
                "sample_id": record["sample_id"],
                "output_stem": record.get("output_stem"),
                "rank": rank,
                "mean_angle": float(record["mean_angle"]),
                "median_angle": float(record["median_angle"]),
                "coverage": float(record["coverage"]),
            }
            manifest_rows.append(item)
            rows.append({"experiment": experiment, **item})
        manifest_path.write_text("".join(json.dumps(row) + "\n" for row in manifest_rows))
        quantiles = np.quantile(angles, (0.5, 0.9, 0.95, 0.99))
        lines.append(
            f"| {experiment} | {dataset} | {len(records)} | {quantiles[0]:.3f} | "
            f"{quantiles[1]:.3f} | {quantiles[2]:.3f} | {quantiles[3]:.3f} | {angles.max():.3f} |"
        )
    (args.output_root / "failure_review.json").write_text(json.dumps({"status": "PASS", "rows": rows}, indent=2) + "\n")
    (args.output_root / "failure_review.md").write_text("\n".join(lines) + "\n")
    print(f"PASS experiments={len(lines) - 4} ranked_rows={len(rows)} output={args.output_root}")


if __name__ == "__main__":
    main()
