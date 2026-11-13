#!/usr/bin/env python3
"""Pair full native Marigold scores with frozen OVB scores by exact image ID."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np


MODELS = (
    "4b_nonzero_checkpoint100000",
    "4b_zeroshot_checkpoint87500_95000_w25",
    "9b_nonzero_checkpoint75000",
    "9b_zeroshot_checkpoint100000",
)
NORMAL_DATASETS = {"nyuv2": "nyu", "ibims": "ibims1", "scannet": "scannet", "sintel": "sintel"}
DEPTH_DATASETS = {"nyuv2": "nyu", "kitti": "kitti_eigen", "eth3d": "eth3d", "scannet": "scannet", "diode": "diode"}
DEPTH_PATH = re.compile(r"/data/(nyuv2|kitti|eth3d|scannet|diode)/(.+)$")


def native_rows(archive: Path) -> dict[str, dict]:
    with tarfile.open(archive) as handle:
        members = [m for m in handle.getmembers() if m.name.endswith("/per_sample_metrics.csv")]
        if len(members) != 1:
            raise ValueError(f"Expected one per-sample CSV in {archive}, got {len(members)}")
        source = handle.extractfile(members[0])
        if source is None:
            raise ValueError(f"Cannot read {members[0].name}")
        rows = list(csv.DictReader(io.StringIO(source.read().decode())))
    indexed = {row["filename"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate native filenames in {archive}")
    return indexed


def normal_sample_id(dataset: str, filename: str) -> str:
    stem = filename.removesuffix(".png").replace("/", "__")
    if dataset == "ibims":
        return stem.removesuffix("_img")
    return ("images__" if dataset == "scannet" else "test__images__") + stem


def depth_native_filename(image_path: str) -> str:
    match = DEPTH_PATH.search(image_path)
    if not match:
        raise ValueError(f"Unrecognized OVB depth source path: {image_path}")
    source = Path(match.group(2))
    stem = source.stem
    if match.group(1) == "nyuv2":
        stem = stem.removeprefix("rgb_")
    return str(source.with_name("pred_" + stem + ".npy"))


def ovb_rows(base: Path, model: str, dataset: str, lane: str) -> dict[str, dict]:
    directory = base / ("surface_normals_geometry_v6" if lane == "normals" else "depth_marigold_v2_ppd") / model / dataset
    files = list(directory.glob("*_per_sample.jsonl"))
    if len(files) != 1:
        raise ValueError(f"Expected one OVB JSONL in {directory}, got {len(files)}")
    rows = [json.loads(line) for line in files[0].read_text().splitlines() if line]
    key = (lambda row: row["sample_id"]) if lane == "normals" else (lambda row: depth_native_filename(row["image_path"]))
    indexed = {key(row): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate OVB IDs in {files[0]}")
    return indexed


def cluster(dataset: str, filename: str, lane: str) -> str:
    parts = filename.split("/")
    if dataset in {"scannet", "sintel"}:
        return parts[0]
    if dataset == "nyuv2" and lane == "depth":
        return parts[1]
    if dataset == "kitti":
        return next((part for part in parts if "_drive_" in part), parts[0])
    if dataset == "diode":
        return "/".join(parts[1:3])
    if dataset == "eth3d":
        return parts[1]
    if dataset == "ibims":
        return filename.split("_")[0]
    return filename.split("_")[0]


def cluster_interval(deltas: list[tuple[str, float]], seed: int = 20260923) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    for name, value in deltas:
        groups[name].append(value)
    if len(groups) < 3:
        return {"clusters": len(groups), "descriptive_95pct_cluster_bootstrap": None}
    values = list(groups.values())
    rng = np.random.default_rng(seed)
    replicates = []
    for _ in range(2000):
        picked = rng.integers(0, len(values), len(values))
        selected = [value for index in picked for value in values[index]]
        replicates.append(float(np.mean(selected)))
    return {
        "clusters": len(groups),
        "descriptive_95pct_cluster_bootstrap": [float(x) for x in np.quantile(replicates, [0.025, 0.975])],
    }


def scannet_stratum(filename: str, training_scenes: set[str], exact_ids: set[str]) -> str:
    scene, image = filename.split("/", 1)
    frame = Path(image).stem.removesuffix("_img")
    if f"{scene}_{frame}" in exact_ids:
        return "exact_training_image"
    if scene in training_scenes:
        return "seen_scene_other_image"
    return "unseen_scene"


def scannet_strata(pairs: list[tuple[str, float, float]], training_scenes: set[str], exact_ids: set[str]) -> dict:
    grouped: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for row in pairs:
        grouped[scannet_stratum(row[0], training_scenes, exact_ids)].append(row)
    result = {}
    for name in ("exact_training_image", "seen_scene_other_image", "unseen_scene"):
        rows = grouped.get(name, [])
        if not rows:
            result[name] = {"samples": 0}
            continue
        differences = [(cluster("scannet", row[0], "normals"), row[2] - row[1]) for row in rows]
        result[name] = {
            "samples": len(rows),
            "native_marigold_mean": float(np.mean([row[1] for row in rows])),
            "ovb_mean_on_same_samples": float(np.mean([row[2] for row in rows])),
            "ovb_minus_marigold": float(np.mean([row[2] - row[1] for row in rows])),
            **cluster_interval(differences),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("first32", "full"), default="full")
    parser.add_argument("--lane", choices=("all", "normals", "depth"), default="all")
    args = parser.parse_args()
    overlap_path = args.base / "audits" / "train_eval_overlap" / "scannet_source_overlap_v2.json"
    overlap = json.loads(overlap_path.read_text())
    training_scenes = set(overlap["all_training_scene_ids"])
    exact_ids = set(overlap["exact_source_ids"])
    results = []
    for lane, datasets in (("normals", NORMAL_DATASETS), ("depth", DEPTH_DATASETS)):
        if args.lane != "all" and lane != args.lane:
            continue
        for native_dataset, ovb_dataset in datasets.items():
            archive = args.base / "audits" / f"native_marigold_{lane}_{args.scope}" / f"{native_dataset}_{args.scope}.tar"
            native = native_rows(archive)
            native_metric = "mean_angular_error" if lane == "normals" else "abs_relative_difference"
            ovb_metric = "mean_angle" if lane == "normals" else "marigold_v2_ppd_abs_rel"
            for model in MODELS:
                ovb = ovb_rows(args.base, model, ovb_dataset, lane)
                pairs = []
                for filename, native_row in native.items():
                    key = normal_sample_id(native_dataset, filename) if lane == "normals" else filename
                    if key not in ovb:
                        raise ValueError(f"Missing OVB pair: {lane}/{native_dataset}/{model}/{key}")
                    pairs.append((filename, float(native_row[native_metric]), float(ovb[key][ovb_metric])))
                if len(pairs) != len(native):
                    raise ValueError(f"Incomplete pairing: {lane}/{native_dataset}/{model}")
                native_values = [row[1] for row in pairs]
                ovb_values = [row[2] for row in pairs]
                differences = [(cluster(native_dataset, row[0], lane), row[2] - row[1]) for row in pairs]
                result = {
                    "lane": lane,
                    "dataset": native_dataset,
                    "model": model,
                    "samples": len(pairs),
                    "native_marigold_mean": float(np.mean(native_values)),
                    "ovb_mean_on_same_samples": float(np.mean(ovb_values)),
                    "ovb_minus_marigold": float(np.mean([value for _, value in differences])),
                    "fraction_ovb_worse": float(np.mean(np.asarray(ovb_values) > np.asarray(native_values))),
                    **cluster_interval(differences),
                }
                if lane == "normals" and native_dataset == "scannet":
                    result["training_overlap_strata"] = scannet_strata(pairs, training_scenes, exact_ids)
                results.append(result)
    payload = {
        "status": "PASS",
        "scope": args.scope,
        "lane": args.lane,
        "note": "Paired image IDs; native Marigold and frozen OVB scorer paths remain separately labelled. Bootstrap is descriptive and clusters may be few.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"PASS paired rows={len(results)} output={args.output}")


if __name__ == "__main__":
    main()
