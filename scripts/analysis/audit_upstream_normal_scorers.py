#!/usr/bin/env python3
"""Score cached RGB normal maps with the original RINO and Marigold functions.

This is a scorer/protocol audit, not a claim that OVB inference matches either
baseline's own model or preprocessing. The first five samples select RINO's
single axis convention, exactly as in its evaluator.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.surface_normal_encoding import decode_surface_normals_rgb
from evaluation.surface_normals import datasets


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_marigold_metric_functions(path: Path):
    """Load unchanged source definitions without unrelated training imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_compute_cosine_error", "_metric_values"}
    ]
    if {node.name for node in selected} != {"_compute_cosine_error", "_metric_values"}:
        raise RuntimeError(f"Marigold metric functions not found in {path}")
    scope = {"np": np, "torch": torch, "F": F,
             "METRIC_NAMES": ("mean_angular_error", "median_angular_error",
                              "sub5_error", "sub7_5_error", "sub11_25_error",
                              "sub22_5_error", "sub30_error")}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return scope


def raw_gt(path: Path) -> np.ndarray:
    array = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Invalid normal array: {path}: {array.shape}")
    if array.shape[0] == 3:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        raise ValueError(f"Invalid normal channels: {path}: {array.shape}")
    return array


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=("nyu", "ibims1", "scannet", "sintel", "diode"))
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--prediction-dir", required=True, type=Path)
    p.add_argument("--prediction-encoding", required=True,
                   choices=("legacy_direct_xyz", "vision_banana_x_left_red"))
    p.add_argument("--rino-root", required=True, type=Path)
    p.add_argument("--marigold-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    rino_src = args.rino_root / "tasks/surface_normals/evaluate"
    codec = load_module("audit_rino_codec", rino_src / "codec.py")
    rino = load_module("audit_rino_metrics", rino_src / "metrics.py")
    marigold = load_marigold_metric_functions(
        args.marigold_root / "evaluation/normals/eval_normals.py"
    )
    case = "indoor" if args.dataset == "diode" else "standard"
    root = datasets.resolve_dataset_root(args.dataset, args.data_root)
    samples = datasets.build_samples(args.dataset, root, "test", args.limit, (case,), ("appearance",))
    if len(samples) < 5:
        raise ValueError("RINO calibration requires at least five samples")

    rows = []
    calibration = []
    for sample in samples:
        rgb_path = args.prediction_dir / f"{sample.output_stem}.png"
        if not rgb_path.is_file():
            raise FileNotFoundError(rgb_path)
        raw = Image.open(rgb_path).convert("RGB")
        source_size = Image.open(sample.image_path).size
        gt = raw_gt(sample.normal_path)
        hw = gt.shape[:2]

        # RINO run.py: image-space bilinear resize, RGB decode and unitize,
        # vector-space zoom to the GT shape, then unitize again.
        rino_pred = codec.decode_normals(raw.resize(source_size, Image.Resampling.BILINEAR))
        if rino_pred.shape[:2] != hw:
            rino_pred = zoom(rino_pred, (hw[0] / rino_pred.shape[0], hw[1] / rino_pred.shape[1], 1), order=1)
            rino_pred, _ = codec.to_unit(rino_pred)
        rino_gt, rino_valid = codec.to_unit(gt)
        if sample.mask_path is not None:
            if sample.mask_path.suffix.lower() == ".npy":
                mask = np.asarray(np.load(sample.mask_path, allow_pickle=False)) > 0
            else:
                mask = np.asarray(Image.open(sample.mask_path).convert("L")) > 0
            if mask.shape != hw:
                raise ValueError(f"Mask shape mismatch: {sample.mask_path}")
            rino_valid &= mask
        if len(calibration) < 5:
            calibration.append((rino_pred, rino_gt, rino_valid))

        # Marigold eval_normals.py's metric on OVB's fixed camera-space RGB
        # prediction. This audits the scorer, not Marigold's model output path.
        direct = np.asarray(raw.resize((hw[1], hw[0]), Image.Resampling.BILINEAR))
        fixed_pred, _ = decode_surface_normals_rgb(
            direct, encoding=args.prediction_encoding, min_norm=0.01
        )
        pred_chw = torch.from_numpy(np.moveaxis(fixed_pred.astype(np.float32), -1, 0))[None]
        gt_chw = torch.from_numpy(np.moveaxis(gt, -1, 0))[None]
        angles = marigold["_compute_cosine_error"](pred_chw, gt_chw)
        marigold_metrics = marigold["_metric_values"](angles)
        rows.append((sample.sample_id, rino_pred, rino_gt, rino_valid, marigold_metrics))

    perm, signs, calib_error = rino.calibrate(calibration)
    rino_rows = []
    marigold_rows = []
    for sample_id, pred, gt, valid, mg in rows:
        score = rino.metrics(pred, gt, valid, perm, signs)
        if not score:
            raise ValueError(f"RINO scorer skipped {sample_id}")
        rino_rows.append({"sample_id": sample_id, **score})
        marigold_rows.append({"sample_id": sample_id, **mg})

    report = {
        "dataset": args.dataset,
        "prediction_encoding": args.prediction_encoding,
        "sample_count": len(rows),
        "rino": {"axis_perm": perm, "axis_signs": signs, "calibration_mean": calib_error,
                 "mean": rino.average([{k: v for k, v in row.items() if k != "sample_id"} for row in rino_rows])},
        "marigold_fixed_rgb": {key: float(np.mean([row[key] for row in marigold_rows]))
                               for key in marigold["METRIC_NAMES"]},
        "per_sample": [{"sample_id": a["sample_id"], "rino": a, "marigold_fixed_rgb": b}
                       for a, b in zip(rino_rows, marigold_rows, strict=True)],
        "source": {"rino_metrics": str(rino_src / "metrics.py"),
                   "marigold_metrics": str(args.marigold_root / "evaluation/normals/eval_normals.py")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temp.replace(args.output)
    print(f"PASS {args.dataset}: {len(rows)} samples; RINO mean {report['rino']['mean']['mean']:.3f}")


if __name__ == "__main__":
    main()
