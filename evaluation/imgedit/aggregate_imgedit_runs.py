#!/usr/bin/env python3
"""Aggregate N repeat ImgEdit runs of one checkpoint into a single summary.

Repeat runs live as run1/, run2/, ... inside a suite dir. Each carries the
judge's own per-run outputs:

  score_summary.json  {"suite", "num_scored", "average_score", ...}  <- headline
  typescore.json      {"add": 4.81, "remove": 4.29, ...}             <- basic only
  average_score.json  {"<key>": 3.67, ...}                           <- per sample

This walks those, and writes <suite_dir>/average_across_runs.json holding, for the
headline score and for every edit type, the mean / std / min / max / n over runs
plus the per-run values that produced them.

std is the SAMPLE standard deviation (ddof=1) -- an estimate of run-to-run spread,
not of the mean's precision. With the usual N=3 it is itself a very noisy estimate
(a 3-sample std has ~40% relative error), so treat it as an order-of-magnitude
guide to whether two checkpoints differ, not as a significance test. stderr is
reported alongside it as std/sqrt(n) for the same reason.

Runs whose score_summary.json is missing or unreadable are skipped and listed
under "missing_runs" rather than silently dropped -- a partial aggregate that
looks complete is worse than a loud one.

Usage:
  python evaluation/imgedit/aggregate_imgedit_runs.py <suite_dir> [<suite_dir> ...]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path


def _mean_std(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    # ddof=1: sample std. Undefined for n==1, which is reported as None rather
    # than 0.0 -- a single run has unknown spread, not zero spread.
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(var)
        stderr = std / math.sqrt(n)
    else:
        std = stderr = None
    return {
        "mean": round(mean, 4),
        "std": None if std is None else round(std, 4),
        "stderr": None if stderr is None else round(stderr, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "n": n,
        "per_run": [round(v, 4) for v in values],
    }


def _run_dirs(suite_dir: Path) -> list[Path]:
    """run1/, run2/, ... in numeric order (run10 after run9, not after run1)."""
    runs = [p for p in suite_dir.iterdir()
            if p.is_dir() and re.fullmatch(r"run\d+", p.name)]
    return sorted(runs, key=lambda p: int(p.name[3:]))


def aggregate(suite_dir: Path) -> dict | None:
    runs = _run_dirs(suite_dir)
    if not runs:
        print(f"  {suite_dir}: no run*/ subdirectories -- nothing to aggregate")
        return None

    headline: list[float] = []
    per_type: dict[str, list[float]] = {}
    used: list[str] = []
    missing: list[str] = []
    num_scored: list[int] = []

    for run in runs:
        summary_path = run / "score_summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            score = float(summary["average_score"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            missing.append(f"{run.name}: {type(exc).__name__} on {summary_path.name}")
            continue

        headline.append(score)
        used.append(run.name)
        if isinstance(summary.get("num_scored"), int):
            num_scored.append(summary["num_scored"])

        # typescore.json is written for the basic suite only; uge has no edit types.
        type_path = run / "typescore.json"
        if type_path.exists():
            try:
                for edit_type, value in json.loads(type_path.read_text(encoding="utf-8")).items():
                    per_type.setdefault(edit_type, []).append(float(value))
            except (OSError, ValueError, TypeError) as exc:
                missing.append(f"{run.name}: {type(exc).__name__} on typescore.json")

    if not headline:
        print(f"  {suite_dir}: found {len(runs)} run dir(s) but no readable score_summary.json")
        return None

    # Only aggregate a type present in EVERY used run; a type scored in 2 of 3 runs
    # would otherwise get a mean over a different denominator than the headline.
    complete_types = {k: v for k, v in per_type.items() if len(v) == len(headline)}
    partial_types = sorted(set(per_type) - set(complete_types))

    out = {
        "suite_dir": str(suite_dir),
        "num_runs_found": len(runs),
        "num_runs_used": len(headline),
        "runs_used": used,
        "missing_runs": missing,
        "partial_edit_types_excluded": partial_types,
        "num_scored_per_run": num_scored,
        "average_score": _mean_std(headline),
        "typescore": {k: _mean_std(v) for k, v in sorted(complete_types.items())},
    }
    if num_scored and len(set(num_scored)) > 1:
        # Different denominators mean the runs are not directly comparable.
        out["warning"] = (
            f"runs scored different sample counts {num_scored}; the mean is over "
            "runs of unequal size"
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suite_dirs", nargs="+", type=Path,
                        help="Suite dir(s) containing run1/, run2/, ... subdirectories")
    parser.add_argument("--out_name", default="average_across_runs.json",
                        help="Filename written inside each suite dir (default: %(default)s)")
    args = parser.parse_args()

    wrote = 0
    for suite_dir in args.suite_dirs:
        if not suite_dir.is_dir():
            print(f"  {suite_dir}: not a directory -- skipped")
            continue
        result = aggregate(suite_dir)
        if result is None:
            continue
        out_path = suite_dir / args.out_name
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        agg = result["average_score"]
        std = "n/a" if agg["std"] is None else f"{agg['std']:.4f}"
        print(f"  {out_path}: average_score {agg['mean']:.4f} +/- {std} "
              f"over {agg['n']} run(s) {result['runs_used']}")
        if result["missing_runs"]:
            print(f"    WARNING skipped: {result['missing_runs']}")
        if result.get("warning"):
            print(f"    WARNING {result['warning']}")
        wrote += 1

    return 0 if wrote else 1


if __name__ == "__main__":
    sys.exit(main())