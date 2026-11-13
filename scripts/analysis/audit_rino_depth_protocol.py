#!/usr/bin/env python3
"""Score saved RGB depth outputs with RINO's original disparity-affine code.

This is a paired-output diagnostic. It does not claim that OVB inference,
prompts, or every input crop is identical to the RINO paper setup.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.depth.datasets import marigold_ppd as data


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("nyu", "kitti_eigen", "diode"), required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split-root", type=Path, required=True)
    p.add_argument("--prediction-dir", type=Path, required=True)
    p.add_argument("--rino-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    src = args.rino_root / "tasks/depth_estimation/evaluate"
    # RINO metrics.py imports its copied official functions from this directory.
    sys.path.insert(0, str(src))
    codec = load_module("audit_rino_depth_codec", src / "codec.py")
    metrics = load_module("audit_rino_depth_metrics", src / "metrics.py")

    root = data.resolve_dataset_root(args.dataset, args.data_root)
    samples = data.build_samples(args.dataset, root, args.split_root, args.limit)
    rows = []
    for sample in samples:
        path = args.prediction_dir / f"{sample.output_stem}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        gt = data.load_gt_depth(sample)
        official_mask = data.valid_depth_mask(sample, gt)
        # Match RINO's generated-image resize then BT.709 luminance decode.
        image = data.prepare_input_image(sample, Image.open(sample.image_path).convert("RGB"))
        raw = Image.open(path).convert("RGB").resize(image.size, Image.Resampling.BILINEAR)
        pred = codec.decode_depth(raw, mode="luminance")
        if pred.shape != gt.shape:
            pred = zoom(pred.astype(np.float32),
                        (gt.shape[0] / pred.shape[0], gt.shape[1] / pred.shape[1]), order=1)
        if pred.shape != gt.shape:
            raise ValueError(f"Prediction/GT shape mismatch: {sample.sample_id}")
        kind = "nyuv2" if args.dataset == "nyu" else args.dataset
        crop = np.ones(gt.shape, dtype=bool)
        if kind == "nyuv2":
            crop[:] = False
            crop[45:471, 41:601] = True
        elif kind == "kitti_eigen":
            crop[:] = False
            h, w = gt.shape
            crop[int(0.40810811 * h):int(0.99189189 * h),
                 int(0.03594771 * w):int(0.96405229 * w)] = True
        max_depth = 350.0 if args.dataset == "diode" and sample.case == "outdoor" else (
            80.0 if args.dataset == "kitti_eigen" else 10.0)
        result = metrics.compute_official(
            pred, gt, mask=official_mask, crop=crop, min_depth=1e-3,
            max_depth=max_depth)
        if not result:
            raise ValueError(f"RINO scorer skipped {sample.sample_id}")
        rows.append({"sample_id": sample.sample_id, "case": sample.case, **result})

    by_case = {}
    for case in sorted({row["case"] for row in rows}):
        subset = [row for row in rows if row["case"] == case]
        by_case[case] = {"count": len(subset), "metrics": metrics.average([
            {key: value for key, value in row.items() if key not in ("sample_id", "case")}
            for row in subset])}
    report = {
        "dataset": args.dataset, "sample_count": len(rows),
        "protocol": "rino_original_compute_official_on_cached_ovb_rgb",
        "comparability_note": "Same saved OVB outputs; RINO model, prompt and KITTI input crop are not reproduced.",
        "by_case": by_case, "per_sample": rows,
        "source": {"codec": str(src / "codec.py"), "metrics": str(src / "metrics.py")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temp.replace(args.output)
    print(f"PASS {args.dataset}: {len(rows)} samples; "
          + ", ".join(f"{k} AbsRel={v['metrics']['abs_rel']:.4f}" for k, v in by_case.items()))


if __name__ == "__main__":
    main()
