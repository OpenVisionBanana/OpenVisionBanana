#!/usr/bin/env python3
"""Write a backward-compatible dataset spec with an opt-in PD12M source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pd12m-shards", required=True)
    parser.add_argument("--pd12m-weight", type=float, default=0.15)
    parser.add_argument(
        "--prepared",
        action="append",
        default=[],
        metavar="NAME:PATH:WEIGHT",
        help="Conditioned prepared-folder source; repeat as needed.",
    )
    parser.add_argument("--samples-per-epoch", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-shuffle-buffer", type=int, default=10000)
    args = parser.parse_args()
    if args.pd12m_weight <= 0:
        parser.error("--pd12m-weight must be positive")
    datasets = []
    for value in args.prepared:
        try:
            name, path, weight = value.rsplit(":", 2)
        except ValueError:
            parser.error(f"Invalid --prepared value: {value!r}")
        root = Path(path).expanduser()
        if not (root / "metadata.jsonl").is_file():
            parser.error(f"Prepared source has no metadata.jsonl: {root}")
        datasets.append({"name": name, "path": str(root.resolve()), "weight": float(weight)})
    datasets.append(
        {
            "name": "pd12m_generation",
            "type": "webdataset_text_to_image",
            "shards": args.pd12m_shards,
            "task": "text_to_image",
            "source_dataset": "pd12m",
            "weight": args.pd12m_weight,
            "streaming": {
                "resampled": True,
                "shard_shuffle": True,
                "sample_shuffle_buffer": args.sample_shuffle_buffer,
                "error_policy": "warn_and_continue",
            },
        }
    )
    config = {
        "samples_per_epoch": args.samples_per_epoch,
        "seed": args.seed,
        "shuffle": True,
        "datasets": datasets,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
