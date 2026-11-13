#!/usr/bin/env python3
"""Compare a base and a candidate model on GenAI-Bench.

Each model may be given several score directories, one per generation seed;
seeds are averaged per prompt before the models are differenced.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base_scores",
        required=True,
        action="append",
        metavar="DIR",
        help="Directory containing per_prompt.jsonl for the base model. Repeat once per generation seed.",
    )
    parser.add_argument(
        "--candidate_scores",
        required=True,
        action="append",
        metavar="DIR",
        help="Directory containing per_prompt.jsonl for the candidate model. Repeat once per generation seed.",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="Paired bootstrap resamples for the delta CI. 0 disables.",
    )
    parser.add_argument("--ci", type=float, default=95.0, help="Confidence level for the bootstrap CI.")
    parser.add_argument("--bootstrap_seed", type=int, default=0)
    return parser.parse_args()


def bootstrap_ci(
    deltas: list[float], resamples: int, ci: float, seed: int
) -> dict[str, Any] | None:
    """Percentile CI for the mean paired delta.

    The pairing is per prompt (same prompt, both models), so resampling prompts
    with replacement keeps base and candidate together and captures the
    prompt-sampling variance the raw mean hides. It does NOT capture seed
    variance: both runs use one image per prompt, so a CI excluding 0 means the
    difference is stable across THIS prompt set, not across regenerations.
    """
    if resamples <= 0 or len(deltas) < 2:
        return None
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(resamples):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    tail = (100.0 - ci) / 200.0
    lo = means[max(0, int(tail * resamples) - 1)]
    hi = means[min(resamples - 1, int((1.0 - tail) * resamples))]
    return {"ci_level": ci, "resamples": resamples, "lo": lo, "hi": hi, "excludes_zero": lo > 0 or hi < 0}


def group_stats(rows: list[dict[str, Any]], resamples: int, ci: float, seed: int) -> dict[str, Any]:
    deltas = [row["delta"] for row in rows]
    base_values = [row["base_score"] for row in rows]
    candidate_values = [row["candidate_score"] for row in rows]
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    return {
        "count": len(rows),
        "base_mean_score": mean(base_values),
        "candidate_mean_score": mean(candidate_values),
        "mean_delta": mean(deltas),
        # Ratio of means, not mean of ratios: VQAScore lands near 0 on failed
        # prompts, where per-prompt ratios explode and an exact 0 is dropped.
        "preservation_ratio": (mean(candidate_values) / mean(base_values)) if mean(base_values) else None,
        "win_rate": wins / len(rows) if rows else None,
        "wins": wins,
        "losses": losses,
        "ties": len(rows) - wins - losses,
        "delta_ci": bootstrap_ci(deltas, resamples, ci, seed),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_model_scores(dirs: list[str], label: str) -> tuple[dict[str, dict[str, Any]], list[float]]:
    """Collapse one model's generation seeds into a single score per prompt.

    Averaging over seeds here, rather than emitting one row per (prompt, seed),
    is deliberate: a prompt's seeds share its difficulty, so treating them as
    independent rows would inflate n by num_seeds and shrink the bootstrap CI by
    roughly sqrt(num_seeds) -- a confidently wrong error bar. The prompt is the
    sampling unit, so it stays the bootstrap unit.

    Only prompts scored in every seed directory are kept, so each average is
    taken over the same number of seeds.
    """
    per_seed: list[dict[str, dict[str, Any]]] = []
    for directory in dirs:
        path = Path(directory).expanduser() / "per_prompt.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"{label}: no per_prompt.jsonl at {path}")
        per_seed.append({str(row["id"]): row for row in read_jsonl(path)})

    common = set(per_seed[0])
    for rows in per_seed[1:]:
        common &= set(rows)
    if not common:
        raise ValueError(f"{label}: no prompt id is present in all {len(per_seed)} score directories.")
    dropped = sum(len(rows) for rows in per_seed) - len(common) * len(per_seed)
    if dropped:
        print(f"WARNING: {label}: dropped {dropped} rows not scored in all {len(per_seed)} seeds")

    merged: dict[str, dict[str, Any]] = {}
    for prompt_id in common:
        scores = [float(rows[prompt_id]["score"]) for rows in per_seed]
        reference = per_seed[0][prompt_id]
        merged[prompt_id] = {
            "scores": scores,
            "score": mean(scores),
            "prompt": reference.get("prompt"),
            "tags": reference.get("tags", []),
            "evaluator": reference.get("evaluator"),
        }
    seed_means = [mean(float(rows[pid]["score"]) for pid in common) for rows in per_seed]
    return merged, seed_means


def seed_spread(dirs: list[str], seed_means: list[float]) -> dict[str, Any]:
    """Run-to-run spread of one model's headline score across generation seeds.

    This describes seed stability; it is NOT an error bar on the delta. Both
    models draw identical noise for a given prompt, so most of this spread is
    common-mode and cancels on subtraction -- quoting it against a paired
    comparison overstates the uncertainty. Use delta_ci for that.
    """
    return {
        "num_seeds": len(seed_means),
        "score_dirs": list(dirs),
        "per_seed_mean_score": seed_means,
        "mean_score": mean(seed_means),
        "std_score": stdev(seed_means) if len(seed_means) > 1 else None,
    }


def main() -> None:
    args = parse_args()
    base_rows, base_seed_means = load_model_scores(args.base_scores, "base")
    candidate_rows, candidate_seed_means = load_model_scores(args.candidate_scores, "candidate")
    if len(args.base_scores) != len(args.candidate_scores):
        # The means stay valid, but the seed-for-seed noise cancellation that makes
        # the paired delta tight only holds when both sides ran the same seeds.
        print(
            f"WARNING: unequal seed counts (base {len(args.base_scores)}, "
            f"candidate {len(args.candidate_scores)}); the delta CI will be wider than a matched run"
        )
    common_ids = sorted(set(base_rows) & set(candidate_rows))
    if not common_ids:
        raise ValueError("No shared prompt ids between the two score directories.")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows = []
    for prompt_id in common_ids:
        base = base_rows[prompt_id]
        candidate = candidate_rows[prompt_id]
        base_score = float(base["score"])
        candidate_score = float(candidate["score"])
        comparison_rows.append(
            {
                "id": prompt_id,
                "prompt": candidate["prompt"],
                "tags": candidate.get("tags", []),
                "evaluator": candidate.get("evaluator"),
                "base_score": base_score,
                "candidate_score": candidate_score,
                "base_scores_per_seed": base["scores"],
                "candidate_scores_per_seed": candidate["scores"],
                "delta": candidate_score - base_score,
                "preservation_ratio": candidate_score / base_score if base_score != 0 else None,
            }
        )

    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comparison_rows:
        for tag in row.get("tags", []):
            by_tag[str(tag)].append(row)

    overall = group_stats(comparison_rows, args.bootstrap, args.ci, args.bootstrap_seed)
    summary = {
        "num_prompts": len(comparison_rows),
        "evaluator": comparison_rows[0].get("evaluator"),
        "base_seeds": seed_spread(args.base_scores, base_seed_means),
        "candidate_seeds": seed_spread(args.candidate_scores, candidate_seed_means),
        **overall,
        # Tags overlap (a prompt is in basic, attribute and scene at once), so
        # these groups are not independent and do not sum to num_prompts.
        "per_tag": {
            tag: group_stats(tag_rows, args.bootstrap, args.ci, args.bootstrap_seed)
            for tag, tag_rows in sorted(by_tag.items())
        },
    }

    per_prompt_path = out_dir / "comparison_per_prompt.jsonl"
    with per_prompt_path.open("w", encoding="utf-8") as handle:
        for row in comparison_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (out_dir / "comparison_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = [
        "group", "count", "base_mean_score", "candidate_mean_score", "mean_delta",
        "delta_ci_lo", "delta_ci_hi", "significant", "win_rate", "preservation_ratio",
    ]

    def csv_row(group: str, stats: dict[str, Any]) -> dict[str, Any]:
        ci = stats.get("delta_ci") or {}
        return {
            "group": group,
            "count": stats["count"],
            "base_mean_score": stats["base_mean_score"],
            "candidate_mean_score": stats["candidate_mean_score"],
            "mean_delta": stats["mean_delta"],
            "delta_ci_lo": ci.get("lo"),
            "delta_ci_hi": ci.get("hi"),
            "significant": ci.get("excludes_zero"),
            "win_rate": stats["win_rate"],
            "preservation_ratio": stats["preservation_ratio"],
        }

    with (out_dir / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(csv_row("all", overall))
        for tag, stats in summary["per_tag"].items():
            writer.writerow(csv_row(tag, stats))

    def seed_line(label: str, spread: dict[str, Any]) -> str:
        value = f"{spread['mean_score']:.4f}"
        if spread["std_score"] is not None:
            value += f" +/- {spread['std_score']:.4f} SD"
        return f"{label:<18}{value}  ({spread['num_seeds']} seed(s))"

    ci = overall.get("delta_ci") or {}
    print(f"Compared prompts: {summary['num_prompts']}")
    print(seed_line("Base mean:", summary["base_seeds"]))
    print(seed_line("Candidate mean:", summary["candidate_seeds"]))
    if ci:
        print(
            f"Mean delta:       {overall['mean_delta']:+.4f}  "
            f"[{ci['lo']:+.4f}, {ci['hi']:+.4f}] {ci['ci_level']:.0f}% CI"
            f"{'' if ci['excludes_zero'] else '  (includes 0)'}"
        )
    else:
        print(f"Mean delta:       {overall['mean_delta']:+.4f}")
    print(f"Win rate:         {overall['win_rate']:.3f}")
    # The SD above is seed-to-seed spread of each model's own score; the CI is the
    # uncertainty of the difference. They answer different questions -- the SD is
    # largely common-mode and cancels in the delta.
    print()
    print(f"{'group':<18}{'n':>6}{'delta':>10}{'  95% CI':>20}{'  sig':>6}")
    for tag, stats in summary["per_tag"].items():
        tci = stats.get("delta_ci") or {}
        span = f"[{tci['lo']:+.4f}, {tci['hi']:+.4f}]" if tci else "-"
        sig = ("yes" if tci.get("excludes_zero") else "no") if tci else "-"
        print(f"{tag:<18}{stats['count']:>6}{stats['mean_delta']:>+10.4f}{span:>20}{sig:>6}")
    print()
    print(f"Saved comparison: {out_dir}")


if __name__ == "__main__":
    main()
