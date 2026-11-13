#!/usr/bin/env python3
"""Audit source-image overlap between prepared geometry training and evaluation.

Exact IDs are definitive only when source naming preserves dataset IDs. The
NYU lane instead returns image-similarity candidates for manual provenance
review; it does not declare train/test leakage from a nearest-neighbour score.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image


def rgb(tar: tarfile.TarFile, member: tarfile.TarInfo) -> np.ndarray:
    handle = tar.extractfile(member)
    if handle is None:
        raise ValueError(f"Cannot read {member.name}")
    with Image.open(io.BytesIO(handle.read())) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def feature(image: np.ndarray) -> np.ndarray:
    small = Image.fromarray(image).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    vector = np.asarray(small, dtype=np.float32).reshape(-1)
    vector -= vector.mean()
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def train_id(dataset: str, name: str) -> str | None:
    if "/conditioning_images/" not in name:
        return None
    stem = Path(name).stem.split("__", 1)[0]
    if dataset == "scannet" and re.fullmatch(r"scene\d{4}_\d{2}_\d+", stem):
        return stem
    if dataset == "diode" and re.fullmatch(r"\d{5}_\d{5}_(?:indoors|outdoor)_\d+_\d+", stem):
        return stem
    if dataset == "nyu" and re.fullmatch(r"nyu_\d+", stem):
        return stem
    return None


def eval_id(dataset: str, name: str) -> str | None:
    if "/images/" not in name:
        return None
    if dataset == "scannet":
        match = re.search(r"/images/(scene\d{4}_\d{2})/(\d+)_img\.png$", name)
        return "_".join(match.groups()) if match else None
    if dataset == "diode":
        match = re.search(r"/images/.*/(\d{5}_\d{5}_(?:indoors|outdoor)_\d+_\d+)\.png$", name)
        return match.group(1) if match else None
    if dataset == "nyu":
        match = re.search(r"/images/(\d+)_img\.png$", name)
        return match.group(1) if match else None
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("scannet", "diode", "nyu"), required=True)
    p.add_argument("--train-archive", type=Path, required=True)
    p.add_argument("--eval-archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    for path in (args.train_archive, args.eval_archive):
        if not path.is_file():
            raise FileNotFoundError(path)

    with tarfile.open(args.train_archive) as training, tarfile.open(args.eval_archive) as evaluation:
        train = {key: member for member in training.getmembers()
                 if member.isfile() and (key := train_id(args.dataset, member.name)) is not None}
        test = {key: member for member in evaluation.getmembers()
                if member.isfile() and (key := eval_id(args.dataset, member.name)) is not None}
        if not train or not test:
            raise ValueError(f"No images found for {args.dataset}: train={len(train)}, eval={len(test)}")
        if args.dataset == "nyu":
            train_images = {key: rgb(training, member) for key, member in train.items()}
            eval_images = {key: rgb(evaluation, member) for key, member in test.items()}
            train_keys = sorted(train)
            train_matrix = np.stack([feature(train_images[key]) for key in train_keys])
            candidates = []
            for eval_key, image in eval_images.items():
                similarities = train_matrix @ feature(image)
                index = int(np.argmax(similarities))
                train_key = train_keys[index]
                other = train_images[train_key]
                if image.shape == other.shape:
                    mae = float(np.abs(image.astype(np.float32) - other).mean())
                    full_corr = float(np.corrcoef(image.reshape(-1), other.reshape(-1))[0, 1])
                else:
                    mae, full_corr = None, None
                if float(similarities[index]) >= 0.97:
                    candidates.append({"eval_id": eval_key, "train_id": train_key,
                                       "low_res_similarity": float(similarities[index]),
                                       "full_image_correlation": full_corr, "mean_absolute_rgb_difference": mae})
            result = {"dataset": args.dataset, "train_images": len(train),
                      "eval_images": len(test), "method": "nearest_low_res_luma_then_full_rgb",
                      "candidate_threshold": 0.97,
                      "candidates_not_proven_overlap": sorted(candidates, key=lambda row: -row["low_res_similarity"])}
        else:
            shared = sorted(set(train) & set(test))
            result = {"dataset": args.dataset, "train_images": len(train),
                      "eval_images": len(test), "method": "exact_source_id",
                      "exact_source_id_overlap_count": len(shared), "exact_source_ids": shared}
            if args.dataset == "scannet":
                train_scenes = {key.rsplit("_", 1)[0] for key in train}
                eval_scenes = {key.rsplit("_", 1)[0] for key in test}
                result["eval_scene_count"] = len(eval_scenes)
                result["overlapping_scene_count"] = len(eval_scenes & train_scenes)
                result["all_training_scene_ids"] = sorted(train_scenes)
                result["eval_frames_from_trained_scenes"] = sum(
                    key.rsplit("_", 1)[0] in train_scenes for key in test)
                pairs = []
                for key in shared:
                    a, b = rgb(training, train[key]), rgb(evaluation, test[key])
                    if a.shape != b.shape:
                        raise ValueError(f"Resolution mismatch for {key}: {a.shape}, {b.shape}")
                    pairs.append({"source_id": key,
                                  "mean_absolute_rgb_difference": float(np.abs(a.astype(np.float32) - b).mean()),
                                  "rgb_correlation": float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])})
                result["matched_image_pairs"] = pairs

    result["training_archive"] = str(args.train_archive)
    result["evaluation_archive"] = str(args.eval_archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(args.output)
    print(f"PASS {args.dataset}: {len(train)} train images, {len(test)} eval images")


if __name__ == "__main__":
    main()
