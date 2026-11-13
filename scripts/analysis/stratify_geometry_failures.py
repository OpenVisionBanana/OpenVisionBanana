#!/usr/bin/env python3
"""Paired final-model errors by dataset subdomain, with fixed-seed intervals."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


PAIRS = {
    "4b": ("4b_nonzero_checkpoint100000", "4b_zeroshot_checkpoint87500_95000_w25"),
    "9b": ("9b_nonzero_checkpoint75000", "9b_zeroshot_checkpoint100000"),
}


def rows(path: Path) -> dict[str, dict]:
    matches = list(path.glob("*_per_sample.jsonl"))
    if len(matches) != 1:
        raise ValueError(f"Expected one per-sample file in {path}; found {matches}")
    items = [json.loads(line) for line in matches[0].read_text(encoding="utf-8").splitlines()]
    result = {item["sample_id"]: item for item in items}
    if len(result) != len(items):
        raise ValueError(f"Duplicate sample IDs in {matches[0]}")
    return result


def group(lane: str, dataset: str, row: dict, overlap: dict) -> str:
    image_path = row["image_path"]
    sample_id = row["sample_id"]
    if lane == "normals":
        if dataset == "scannet":
            match = re.search(r"/(scene\d{4}_\d{2})/(\d+)_img", image_path)
            if not match:
                raise ValueError(image_path)
            frame = "_".join(match.groups())
            if frame in overlap["exact"]:
                return "exact_training_frame"
            return "trained_scene_other_frame" if match.group(1) in overlap["scenes"] else "clean_scene"
        if dataset == "diode":
            match = re.search(r"/scene_(\d+)/scan_(\d+)/", image_path)
            return f"scene_{match.group(1)}" if match else "unknown_scene"
        if dataset == "sintel":
            match = re.search(r"/images/([^/]+)/", image_path)
            return match.group(1) if match else "unknown_sequence"
        if dataset == "vkitti2":
            match = re.search(r"/images/(Scene\d+)/", image_path)
            return match.group(1) if match else "unknown_scene"
        if dataset == "ibims1":
            return sample_id.split("_", 1)[0]
        return "all_nyu"
    if dataset == "diode":
        match = re.search(r"/(indoors|outdoor)/scene_(\d+)/", image_path)
        return f"{match.group(1)}_scene_{match.group(2)}" if match else row["case"]
    if dataset == "kitti_eigen":
        match = re.search(r"/(\d{4}_\d{2}_\d{2}_drive_\d+_sync)/", image_path)
        return match.group(1) if match else "unknown_drive"
    if dataset == "eth3d":
        match = re.search(r"/rgb/([^/]+)/", image_path)
        return match.group(1) if match else "unknown_scene"
    if dataset == "scannet":
        match = re.search(r"/(scene\d{4}_\d{2})/color/", image_path)
        return match.group(1) if match else "unknown_scene"
    if dataset == "nyu":
        match = re.search(r"/test/([^/]+)/", image_path)
        return match.group(1) if match else "unknown_room"
    return "unknown"


def summary(values: list[float], seed: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    output = {"count": int(array.size), "mean_delta": float(array.mean()),
              "median_delta": float(np.median(array)),
              "fraction_nonzero_worse": float(np.mean(array > 0))}
    if len(array) >= 10:
        rng = np.random.default_rng(seed)
        sampled = array[rng.integers(len(array), size=(1000, len(array)))].mean(axis=1)
        output["bootstrap_95pct_mean_delta"] = [float(x) for x in np.quantile(sampled, [0.025, 0.975])]
    return output


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--overlap-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    overlap_report = json.loads(args.overlap_report.read_text(encoding="utf-8"))
    if overlap_report["dataset"] != "scannet":
        raise ValueError("Not a ScanNet overlap report")
    overlap = {"exact": set(overlap_report["exact_source_ids"]),
               "scenes": {key.rsplit("_", 1)[0] for key in overlap_report["exact_source_ids"]}}
    # The exact IDs represent only a subset of trained scenes. Reconstruct all
    # train scenes from a separate supplied list if available; absent that,
    # fail closed rather than mislabel unseen frames as clean.
    if "all_training_scene_ids" not in overlap_report:
        raise ValueError("Overlap report lacks all_training_scene_ids")
    overlap["scenes"] = set(overlap_report["all_training_scene_ids"])

    findings = []
    for family, (nonzero, zeroshot) in PAIRS.items():
        for lane, datasets, folder, metric in (
            ("normals", ("nyu", "ibims1", "scannet", "vkitti2", "diode", "sintel"),
             "surface_normals_geometry_v6", "mean_angle"),
            ("depth", ("nyu", "kitti_eigen", "eth3d", "scannet", "diode"),
             "depth_marigold_v2_ppd", "marigold_v2_ppd_abs_rel"),
        ):
            for dataset in datasets:
                a = rows(args.results_root / folder / nonzero / dataset)
                b = rows(args.results_root / folder / zeroshot / dataset)
                if set(a) != set(b):
                    raise ValueError(f"Non-paired samples: {family}/{lane}/{dataset}")
                strata = defaultdict(list)
                for sample_id in sorted(a):
                    stratum = group(lane, dataset, a[sample_id], overlap)
                    strata[stratum].append(float(a[sample_id][metric]) - float(b[sample_id][metric]))
                for stratum, values in sorted(strata.items()):
                    findings.append({"family": family, "lane": lane,
                                     "dataset": dataset, "stratum": stratum,
                                     "metric": metric, **summary(values, seed=42)})

    report = {"status": "complete", "positive_delta_means_nonzero_worse": True,
              "bootstrap_note": "Descriptive image-level bootstrap; scenes are correlated and this is not a publication CI.",
              "findings": findings}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temp.replace(args.output)
    print(f"PASS: {len(findings)} strata across paired normal and depth results")


if __name__ == "__main__":
    main()
