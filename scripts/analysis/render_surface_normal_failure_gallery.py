#!/usr/bin/env python3
"""Render compact input/prediction/GT/error galleries for ranked failures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.surface_normal_encoding import VISION_BANANA_NORMAL_ENCODING, encode_surface_normals_rgb, transform_surface_normals
from evaluation.surface_normals import datasets
from evaluation.surface_normals.inference_partial_flux2_klein_img2img import load_gt_normals, load_mask, resize_rgb
from evaluation.surface_normals.target_protocol import resolve_ground_truth_protocol
from common.surface_normal_encoding import decode_surface_normals_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=datasets.SUPPORTED_DATASETS)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--cases", nargs="+", default=("standard",))
    parser.add_argument("--prediction-encoding", default=VISION_BANANA_NORMAL_ENCODING)
    args = parser.parse_args()

    root = datasets.resolve_dataset_root(args.dataset, args.data_root)
    samples = datasets.build_samples(args.dataset, root, args.split, None, tuple(args.cases), ("appearance",))
    by_id = {sample.sample_id: sample for sample in samples}
    selected = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    info_path = root / "dataset_info.json"
    if not info_path.is_file() and (root / "manifest.json").is_file():
        info_path = root / "manifest.json"
    info = json.loads(info_path.read_text()) if info_path.is_file() else {}
    formats = sorted({by_id[row["sample_id"]].normal_path.suffix for row in selected})
    protocol = resolve_ground_truth_protocol(dataset_info=info, storage_formats=formats)
    gt_encoding = "legacy_direct_xyz" if protocol.encoding.startswith("float_xyz") else protocol.encoding

    panel_size = (256, 192)
    header = 24
    rows = []
    for item in selected:
        sample = by_id[item["sample_id"]]
        gt, gt_valid = load_gt_normals(sample.normal_path, min_norm=0.01, rgb_encoding=gt_encoding)
        gt = transform_surface_normals(gt, protocol.permutation, protocol.signs)
        mask = load_mask(sample.mask_path, gt.shape[:2], gt_valid) & gt_valid
        raw = Image.open(args.prediction_dir / f"{sample.output_stem}.png").convert("RGB")
        pred, pred_valid = decode_surface_normals_rgb(np.asarray(resize_rgb(raw, gt.shape[:2])), encoding=args.prediction_encoding)
        dot = np.clip(np.sum(pred * gt, axis=-1), -1.0, 1.0)
        angle = np.degrees(np.arccos(dot))
        angle[mask & ~pred_valid] = 90.0
        scale = np.clip(angle / 90.0, 0.0, 1.0)
        heat = np.zeros((*angle.shape, 3), dtype=np.uint8)
        heat[..., 0] = np.rint(255 * scale).astype(np.uint8)
        heat[..., 1] = np.rint(255 * (1.0 - np.abs(scale - 0.5) * 2.0)).astype(np.uint8)
        heat[..., 2] = np.rint(255 * (1.0 - scale)).astype(np.uint8)
        heat[~mask] = 0
        gt_rgb = encode_surface_normals_rgb(gt, mask, encoding=VISION_BANANA_NORMAL_ENCODING)
        pred_rgb = encode_surface_normals_rgb(pred, pred_valid, encoding=VISION_BANANA_NORMAL_ENCODING)
        panels = [
            Image.open(sample.image_path).convert("RGB"),
            Image.fromarray(pred_rgb), Image.fromarray(gt_rgb), Image.fromarray(heat),
        ]
        row = Image.new("RGB", (panel_size[0] * 4, panel_size[1] + header), "white")
        draw = ImageDraw.Draw(row)
        for index, (panel, title) in enumerate(zip(panels, ("input", "prediction", "ground truth", "angle error"))):
            row.paste(panel.resize(panel_size, Image.Resampling.BILINEAR), (index * panel_size[0], header))
            draw.text((index * panel_size[0] + 4, 4), title, fill="black")
        draw.text((520, 4), f"rank {item['rank']}  mean {item['mean_angle']:.2f}°  {sample.sample_id}", fill="black")
        rows.append(row)
    gallery = Image.new("RGB", (panel_size[0] * 4, (panel_size[1] + header) * len(rows)), "white")
    for index, row in enumerate(rows):
        gallery.paste(row, (0, index * row.height))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gallery.save(args.output, quality=90, optimize=True)
    print(f"PASS rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
