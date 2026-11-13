#!/usr/bin/env python3
"""Compare Marigold V2's original PPD functions with saved OVB depth scores."""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
import torch
import torch.nn.functional as F
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.depth.datasets import marigold_ppd as data
from evaluation.depth.ppd_protocol import score_ppd_ransac_log


def original_functions(path: Path, names: set[str], scope: dict) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in found} != names:
        raise RuntimeError(f"Missing required upstream functions from {path}")
    exec(compile(ast.Module(body=found, type_ignores=[]), str(path), "exec"), scope)
    return scope


def upstream_score(dataset: str, pred_log: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                   src: dict) -> tuple[dict[str, float], dict[str, float]]:
    gt, mask, pred_log = src["_resize_for_ppd_protocol"](gt, mask, pred_log, dataset)
    mask = src["_apply_ppd_mask"](gt, mask, dataset)
    mask = src["_apply_diode_gradient_filter"](gt, mask, dataset)
    gt_log, positive = src["depth2log_space"](gt)
    fitted, scale, shift = src["align_log_space_ransac_ppd"](pred_log, gt_log, mask & positive)
    pred = src["log_space2depth"](fitted)
    pred = src["_clip_depth_like_ppd"](pred, gt, mask)
    pred = np.clip(pred, a_min=1e-6, a_max=None)
    valid_pred = pred[mask]
    valid_gt = gt[mask]
    ratio = np.maximum(valid_pred / valid_gt, valid_gt / valid_pred)
    score = {
        "abs_rel": float(np.mean(np.abs(valid_pred - valid_gt) / valid_gt)),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
    }
    return score, {"scale": float(scale), "shift": float(shift), "valid_pixels": int(mask.sum())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=data.EXPECTED_COUNTS)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--split-root", required=True, type=Path)
    p.add_argument("--prediction-dir", required=True, type=Path)
    p.add_argument("--record-jsonl", required=True, type=Path)
    p.add_argument("--marigold-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--stress", action="store_true",
                   help="score first N, worst N, and all non-positive fit-scale cases")
    args = p.parse_args()
    if args.limit < 1:
        raise ValueError("limit must be positive")

    scope = {"math": math, "ndimage": ndimage, "np": np, "torch": torch, "F": F,
             "PolynomialFeatures": PolynomialFeatures, "RANSACRegressor": RANSACRegressor,
             "make_pipeline": make_pipeline,
             "_PPD_RANSAC_MODEL": make_pipeline(
                 PolynomialFeatures(degree=1, include_bias=False),
                 RANSACRegressor(max_trials=1000, random_state=42)),
             "_PPD_RANSAC_SEED": 42}
    src = original_functions(
        args.marigold_root / "evaluation/depth/eval.py",
        {"_ppd_depth_bounds_from_disp_name", "_apply_ppd_mask",
         "_apply_diode_gradient_filter", "_ppd_target_hw_from_disp_name",
         "_resize_2d_float", "_resize_for_ppd_protocol", "_clip_depth_like_ppd",
         "align_log_space_ransac_ppd"}, scope)
    src = original_functions(
        args.marigold_root / "evaluation/src/util/alignment.py",
        {"depth2log_space", "log_space2depth"}, src)
    recorded = {}
    for line in args.record_jsonl.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        recorded[row["sample_id"]] = row
    dataset_root = data.resolve_dataset_root(args.dataset, args.data_root)
    all_samples = data.build_samples(args.dataset, dataset_root, args.split_root, None)
    selected_ids = {sample.sample_id for sample in all_samples[:args.limit]}
    if args.stress:
        ranked = sorted(
            all_samples,
            key=lambda sample: float(recorded[sample.sample_id]["marigold_v2_ppd_abs_rel"]),
            reverse=True,
        )
        selected_ids.update(sample.sample_id for sample in ranked[:args.limit])
        selected_ids.update(
            sample.sample_id for sample in all_samples
            if float(recorded[sample.sample_id]["marigold_v2_ppd_ransac_scale"]) <= 0
        )
    samples = [sample for sample in all_samples if sample.sample_id in selected_ids]
    rows = []
    for sample in samples:
        old = recorded[sample.sample_id]
        if old["marigold_v2_ppd_input_signal"] != "negative_generated_inverse_grayscale_as_log_depth":
            raise ValueError(f"Unexpected saved signal for {sample.sample_id}")
        path = args.prediction_dir / f"{sample.output_stem}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        pred_log = -np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        gt = data.load_gt_depth(sample)
        mask = data.valid_depth_mask(sample, gt)
        expected = {key: float(old[f"marigold_v2_ppd_{key}"])
                    for key in ("abs_rel", "delta1", "delta2", "delta3")}
        upstream, meta = upstream_score(args.dataset, pred_log, gt, mask, src)
        ours, ours_meta, _ = score_ppd_ransac_log(args.dataset, pred_log, gt, mask)
        rows.append({"sample_id": sample.sample_id, "recorded": expected,
                     "upstream": upstream, "current": ours, "upstream_fit": meta,
                     "current_fit": ours_meta,
                     "max_abs_metric_delta": max(abs(upstream[k] - ours[k]) for k in upstream)})
    max_delta = max(row["max_abs_metric_delta"] for row in rows)
    report = {"dataset": args.dataset, "sample_count": len(rows), "stress": args.stress,
              "max_abs_metric_delta": max_delta, "pass_tolerance_1e5": max_delta <= 1e-5,
              "per_sample": rows,
              "source": str(args.marigold_root / "evaluation/depth/eval.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temp.replace(args.output)
    if not report["pass_tolerance_1e5"]:
        raise SystemExit(f"FAIL {args.dataset}: max score delta {max_delta:.8f}")
    print(f"PASS {args.dataset}: {len(rows)} samples, max score delta {max_delta:.8f}")


if __name__ == "__main__":
    main()
