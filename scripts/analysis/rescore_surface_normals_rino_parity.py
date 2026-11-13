#!/usr/bin/env python3
"""Cross-score raw normal-map PNGs under release and exact RINO protocols."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.surface_normal_encoding import decode_surface_normals_rgb, transform_surface_normals
from evaluation.geometry_protocols import (
    all_normal_conventions,
    calibrate_test5_axis_oracle,
    convention_oracle_normal_metrics,
)
from evaluation.surface_normals import datasets
from evaluation.surface_normals.target_protocol import resolve_ground_truth_protocol
from evaluation.surface_normals.inference_partial_flux2_klein_img2img import (
    aggregate_metrics,
    aggregate_pooled_normal_statistics,
    compute_normal_metrics,
    erode_mask,
    load_gt_normals,
    load_mask,
    pooled_normal_statistics,
    rgb_to_oracle_unit_normals,
    resize_rgb,
)

RESAMPLING = getattr(Image, "Resampling", Image)
METRICS = ("mean_angle", "median_angle", "rmse_angle", "angle_11_25", "angle_22_5", "angle_30")


def triplet(value: str, signs: bool = False) -> tuple[int, int, int]:
    result = tuple(int(part) for part in value.split(","))
    valid = len(result) == 3 and (
        all(item in (-1, 1) for item in result) if signs else sorted(result) == [0, 1, 2]
    )
    if not valid:
        raise argparse.ArgumentTypeError(f"invalid triplet: {value}")
    return result


def rino_resize(raw: Image.Image, image_size: tuple[int, int], hw: tuple[int, int]) -> np.ndarray:
    """RINO: RGB -> input size -> decode -> bilinear vectors to GT -> normalize."""
    from scipy.ndimage import zoom

    unit = rgb_to_oracle_unit_normals(raw.convert("RGB").resize(image_size, RESAMPLING.BILINEAR))
    if unit.shape[:2] != hw:
        unit = zoom(unit, (hw[0] / unit.shape[0], hw[1] / unit.shape[1], 1), order=1)
        unit /= np.clip(np.linalg.norm(unit, axis=-1, keepdims=True), 1e-8, None)
    return unit


def average(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def numeric(row: dict, excluded: set[str]) -> dict[str, float]:
    return {key: float(value) for key, value in row.items() if key not in excluded}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=datasets.SUPPORTED_DATASETS)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--cases", nargs="+", default=("standard",))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prediction-encoding", default="legacy_direct_xyz")
    parser.add_argument("--gt-encoding", default="auto")
    parser.add_argument("--gt-permutation", type=triplet)
    parser.add_argument("--gt-signs", type=lambda value: triplet(value, True))
    parser.add_argument("--min-normal-norm", type=float, default=0.01)
    args = parser.parse_args()

    root = datasets.resolve_dataset_root(args.dataset, Path(args.data_root))
    samples = datasets.build_samples(
        args.dataset, root, args.split, args.limit, tuple(args.cases), ("appearance",)
    )
    info_path = root / "dataset_info.json"
    if not info_path.is_file() and (root / "manifest.json").is_file():
        info_path = root / "manifest.json"
    dataset_info = json.loads(info_path.read_text()) if info_path.is_file() else {}
    gt_protocol = resolve_ground_truth_protocol(
        dataset_info=dataset_info,
        storage_formats=sorted({sample.normal_path.suffix for sample in samples}),
        requested_encoding=args.gt_encoding,
        requested_permutation=args.gt_permutation,
        requested_signs=args.gt_signs,
    )
    gt_rgb_encoding = (
        "legacy_direct_xyz"
        if gt_protocol.encoding.startswith("float_xyz")
        else gt_protocol.encoding
    )
    prediction_dir = Path(args.prediction_dir)
    rows = []
    for sample in samples:
        raw_path = prediction_dir / f"{sample.output_stem}.png"
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        raw = Image.open(raw_path).convert("RGB")
        gt, gt_valid = load_gt_normals(
            sample.normal_path,
            min_norm=args.min_normal_norm,
            rgb_encoding=gt_rgb_encoding,
        )
        gt = transform_surface_normals(gt, gt_protocol.permutation, gt_protocol.signs)
        mask = load_mask(sample.mask_path, gt.shape[:2], gt_valid) & gt_valid
        if sample.normal_path.suffix.lower() == ".npy":
            raw_gt = np.load(sample.normal_path, allow_pickle=False)
            if raw_gt.ndim == 3 and raw_gt.shape[0] == 3:
                raw_gt = np.moveaxis(raw_gt, 0, -1)
            # RINO's codec.to_unit() uses original vector length > 0.5 for
            # validity, whereas the release evaluator uses min_normal_norm.
            rino_gt_valid = np.linalg.norm(raw_gt, axis=-1) > 0.5
        else:
            rino_gt_valid = gt_valid
        rino_mask = load_mask(sample.mask_path, gt.shape[:2], rino_gt_valid) & rino_gt_valid
        direct_rgb = resize_rgb(raw, gt.shape[:2])
        fixed, fixed_valid = decode_surface_normals_rgb(
            np.asarray(direct_rgb), encoding=args.prediction_encoding, min_norm=args.min_normal_norm
        )
        rows.append((
            sample, fixed, fixed_valid, rgb_to_oracle_unit_normals(direct_rgb),
            rino_resize(raw, Image.open(sample.image_path).size, gt.shape[:2]),
            gt, gt_valid, mask, rino_mask,
        ))

    rino_convention, rino_calibration = calibrate_test5_axis_oracle(
        (row[4], row[5], row[8]) for row in rows[:5]
    )
    # RINO's eight-GPU launcher starts shards concurrently, each of which may
    # reach calibration before the shared convention.json exists. Simulate
    # those shard-local first-five calibration sets from the same cached maps.
    shard_calibrations = []
    if len(rows) >= 40:
        for shard in range(8):
            calibration_rows = rows[shard::8][:5]
            convention, calibration_mae = calibrate_test5_axis_oracle(
                (row[4], row[5], row[8]) for row in calibration_rows
            )
            shard_calibrations.append({
                "shard": shard,
                "sample_ids": [row[0].sample_id for row in calibration_rows],
                "convention": convention.convention_id,
                "calibration_mean_angle": float(calibration_mae),
            })
    shard_score_by_convention = {}
    for convention_id in sorted({item["convention"] for item in shard_calibrations}):
        convention = next(
            c for c in all_normal_conventions() if c.convention_id == convention_id
        )
        shard_score_by_convention[convention_id] = float(np.mean([
            convention_oracle_normal_metrics(row[4], row[5], row[8], convention)["mean_angle"]
            for row in rows
        ]))
    direct_convention, direct_calibration = calibrate_test5_axis_oracle(
        (row[3], row[5], row[8]) for row in rows[:5]
    )
    fixed_rows, rino_rows, direct_rows, oracle_rows = [], [], [], []
    erosion_rows: dict[int, list[dict[str, float]]] = {1: [], 3: [], 5: []}
    convention_counts: dict[str, int] = {}
    convention_mean_by_image: dict[str, list[float]] = {}
    for sample, fixed, fixed_valid, direct, rino, gt, gt_valid, mask, rino_mask in rows:
        metrics, _, angle_map = compute_normal_metrics(
            fixed, fixed_valid, gt, gt_valid, mask,
            bidirectional_normals=False, min_valid_pixels=10,
        )
        fixed_rows.append({
            "sample_id": sample.sample_id, "output_stem": sample.output_stem,
            **metrics, **pooled_normal_statistics(angle_map, mask),
        })
        rino_metrics = convention_oracle_normal_metrics(rino, gt, rino_mask, rino_convention)
        direct_metrics = convention_oracle_normal_metrics(direct, gt, rino_mask, direct_convention)
        rino_rows.append({"sample_id": sample.sample_id, "output_stem": sample.output_stem, **rino_metrics})
        direct_rows.append({"sample_id": sample.sample_id, "output_stem": sample.output_stem, **direct_metrics})
        candidates = []
        for convention in all_normal_conventions():
            candidate = convention_oracle_normal_metrics(direct, gt, rino_mask, convention)
            candidates.append((candidate["mean_angle"], convention.convention_id, candidate))
            convention_mean_by_image.setdefault(convention.convention_id, []).append(
                float(candidate["mean_angle"])
            )
        _, convention_id, best = min(candidates)
        convention_counts[convention_id] = convention_counts.get(convention_id, 0) + 1
        oracle_rows.append({
            "sample_id": sample.sample_id, "output_stem": sample.output_stem,
            "convention": convention_id, **best,
        })
        for radius in erosion_rows:
            try:
                erosion_rows[radius].append(convention_oracle_normal_metrics(
                    direct, gt, erode_mask(mask, radius), direct_convention
                ))
            except ValueError as exc:
                if "requires 10 valid pixels" not in str(exc):
                    raise

    fixed_metric_rows = [{key: float(row[key]) for key in METRICS} for row in fixed_rows]
    direct_numeric = [numeric(row, {"sample_id", "output_stem"}) for row in direct_rows]
    means = np.asarray([row["mean_angle"] for row in direct_numeric])
    calibration_stability = {}
    for group_size in (5, 20):
        groups = []
        for start in range(0, min(len(samples), group_size * 5), group_size):
            stop = min(start + group_size, len(samples))
            if stop - start < group_size:
                break
            chosen = min(
                convention_mean_by_image,
                key=lambda key: float(np.mean(convention_mean_by_image[key][start:stop])),
            )
            groups.append({
                "start": start,
                "stop": stop,
                "convention": chosen,
                "calibration_mean_angle": float(
                    np.mean(convention_mean_by_image[chosen][start:stop])
                ),
                "full_set_mean_angle": float(np.mean(convention_mean_by_image[chosen])),
            })
        if groups:
            full_means = [group["full_set_mean_angle"] for group in groups]
            calibration_stability[str(group_size)] = {
                "groups": groups,
                "unique_conventions": sorted({group["convention"] for group in groups}),
                "full_set_mean_angle_range": [min(full_means), max(full_means)],
                "full_set_mean_angle_spread": max(full_means) - min(full_means),
            }
    report = {
        "dataset": args.dataset,
        "sample_count": len(samples),
        "ground_truth_protocol": gt_protocol.metadata(),
        "paths": {"data_root": str(root), "prediction_dir": str(prediction_dir)},
        "fixed_release": {
            "per_image_mean": aggregate_metrics(fixed_metric_rows),
            "global_pixel_pool": aggregate_pooled_normal_statistics(fixed_rows),
        },
        "rino_exact": {
            "ground_truth_min_original_vector_norm_exclusive": 0.5,
            "convention": rino_convention.convention_id,
            "calibration_mean_angle": rino_calibration,
            "calibration_sample_ids": [row[0].sample_id for row in rows[:5]],
            "per_image_mean": average([numeric(row, {"sample_id", "output_stem"}) for row in rino_rows]),
            "per_sample": rino_rows,
        },
        "rino_eight_shard_calibration_race_diagnostic": {
            "interpretation": "Independent shard-local calibrations if all eight workers race before the shared convention file exists; not proof of a particular upstream run.",
            "shards": shard_calibrations,
            "full_set_mean_angle_by_convention": shard_score_by_convention,
        },
        "shared_direct_resize_test5": {
            "convention": direct_convention.convention_id,
            "calibration_mean_angle": direct_calibration,
            "per_image_mean": average(direct_numeric),
        },
        "mask_erosion": {
            str(radius): average(values)
            for radius, values in erosion_rows.items()
            if values
        },
        "calibration_stability": calibration_stability,
        "per_image_axis_oracle": {
            "metrics": average([numeric(row, {"sample_id", "output_stem", "convention"}) for row in oracle_rows]),
            "convention_counts": dict(sorted(convention_counts.items(), key=lambda item: (-item[1], item[0]))),
        },
        "tail": {
            "image_mean_angle_quantiles": {
                str(q): float(np.quantile(means, q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)
            },
            "worst_20": sorted(direct_rows, key=lambda row: row["mean_angle"], reverse=True)[:20],
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
