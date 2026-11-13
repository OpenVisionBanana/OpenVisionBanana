#!/usr/bin/env python3
"""Score generated GenAI-Bench images with t2v_metrics VQAScore."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation_dir", required=True)
    parser.add_argument("--out_dir", help="Defaults to generation_dir/scores/<evaluator>.")
    parser.add_argument("--evaluator", default="qwen2.5-vl-7b")
    parser.add_argument("--checkpoint", help="Optional local evaluator-model snapshot.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache_dir")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--question")
    parser.add_argument("--answer")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
    return rows


def load_manifest(generation_dir: Path) -> list[dict[str, Any]]:
    manifest_path = generation_dir / "manifest.jsonl"
    if manifest_path.is_file():
        rows = read_jsonl(manifest_path)
    else:
        shard_paths = sorted((generation_dir / "shard_manifests").glob("rank_*_of_*.jsonl"))
        if not shard_paths:
            raise FileNotFoundError(f"No manifest.jsonl or shard manifests found under {generation_dir}")
        rows = []
        seen = set()
        for path in shard_paths:
            for row in read_jsonl(path):
                key = str(row["id"])
                if key in seen:
                    continue
                rows.append(row)
                seen.add(key)
    rows.sort(key=lambda row: str(row["id"]))
    for row in rows:
        image_path = Path(row["image"])
        if not image_path.is_absolute():
            image_path = generation_dir / image_path
        row["_resolved_image"] = str(image_path)
    missing = [row for row in rows if not Path(row["_resolved_image"]).is_file()]
    if missing:
        example = missing[0]
        raise FileNotFoundError(f"{len(missing)} generated images are missing; first missing: {example['image']}")
    return rows


def scalar_score(value: Any) -> float:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().reshape(-1)[0].item())
    except ModuleNotFoundError:
        pass
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return float(value.reshape(-1)[0])
    except ModuleNotFoundError:
        pass
    if isinstance(value, (list, tuple)):
        return scalar_score(value[0])
    return float(value)


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["score"]) for row in rows if math.isfinite(float(row["score"]))]
    by_tag: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        score = float(row["score"])
        for tag in row.get("tags", []):
            by_tag[str(tag)].append(score)
    return {
        "num_prompts": len(rows),
        "num_scored": len(values),
        "mean_score": mean(values) if values else None,
        "median_score": median(values) if values else None,
        "std_score": pstdev(values) if len(values) > 1 else 0.0,
        "min_score": min(values) if values else None,
        "max_score": max(values) if values else None,
        "per_tag": {
            tag: {
                "count": len(tag_values),
                "mean_score": mean(tag_values),
                "median_score": median(tag_values),
                "std_score": pstdev(tag_values) if len(tag_values) > 1 else 0.0,
            }
            for tag, tag_values in sorted(by_tag.items())
            if tag_values
        },
    }


def write_summary_csv(path: Path, metrics: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "count", "mean_score", "median_score", "std_score"])
        writer.writeheader()
        writer.writerow(
            {
                "group": "all",
                "count": metrics["num_scored"],
                "mean_score": metrics["mean_score"],
                "median_score": metrics["median_score"],
                "std_score": metrics["std_score"],
            }
        )
        for tag, values in metrics["per_tag"].items():
            writer.writerow(
                {
                    "group": tag,
                    "count": values["count"],
                    "mean_score": values["mean_score"],
                    "median_score": values["median_score"],
                    "std_score": values["std_score"],
                }
            )


def main() -> None:
    args = parse_args()
    generation_dir = Path(args.generation_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else generation_dir / "scores" / args.evaluator.replace("/", "_")
    per_prompt_path = out_dir / "per_prompt.jsonl"
    metrics_path = out_dir / "metrics.json"
    scores_path = out_dir / "scores.pt"

    if metrics_path.exists() and per_prompt_path.exists() and not args.overwrite:
        print(f"Using existing score outputs: {out_dir}")
        return

    rows = load_manifest(generation_dir)
    dataset = [{"images": [row["_resolved_image"]], "texts": [row["prompt"]]} for row in rows]

    try:
        import t2v_metrics
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "GenAI-Bench scoring requires the pinned t2v_metrics checkout. "
            "See evaluation/image_generation/README.md for the installation command."
        ) from exc
    import torch
    from t2v_metrics.constants import HF_CACHE_DIR

    cache_dir = args.cache_dir or HF_CACHE_DIR
    if args.checkpoint and args.evaluator.startswith(("qwen2-vl-", "qwen2.5-vl-")):
        # The pinned t2v_metrics Qwen adapter forwards its checkpoint to the
        # model but otherwise keeps the processor's Hub ID. Point both at the
        # same standalone snapshot so offline scoring never reaches the Hub.
        from t2v_metrics.models.vqascore_models import qwen2vl_model

        model_info = qwen2vl_model.QWEN2_VL_MODELS.get(args.evaluator)
        if model_info is None:
            raise ValueError(f"Unsupported local Qwen evaluator: {args.evaluator}")
        model_info["tokenizer"]["path"] = args.checkpoint
    model_kwargs = {"checkpoint": args.checkpoint} if args.checkpoint else {}
    score_func = t2v_metrics.get_score_model(
        model=args.evaluator,
        device=args.device,
        cache_dir=cache_dir,
        **model_kwargs,
    )
    kwargs = {}
    if args.question is not None:
        kwargs["question_template"] = args.question
    if args.answer is not None:
        kwargs["answer_template"] = args.answer

    out_dir.mkdir(parents=True, exist_ok=True)
    scores = score_func.batch_forward(dataset, batch_size=args.batch_size, **kwargs).detach().cpu()
    torch.save(scores, scores_path)

    scored_rows = []
    with per_prompt_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            scored = {
                "id": row["id"],
                "prompt": row["prompt"],
                "tags": row.get("tags", []),
                "image": row["image"],
                "model_tag": row.get("model_tag"),
                "checkpoint": row.get("checkpoint"),
                "evaluator": args.evaluator,
                "score": scalar_score(scores[index]),
            }
            scored_rows.append(scored)
            handle.write(json.dumps(scored, ensure_ascii=False, sort_keys=True) + "\n")

    metrics = summarize_scores(scored_rows)
    metrics.update(
        {
            "generation_dir": str(generation_dir),
            "evaluator": args.evaluator,
            "batch_size": args.batch_size,
            "score_tensor": str(scores_path),
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_csv(out_dir / "summary.csv", metrics)

    print(f"Scored prompts: {metrics['num_scored']}")
    print(f"Mean score: {metrics['mean_score']}")
    print(f"Saved per-prompt scores: {per_prompt_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
