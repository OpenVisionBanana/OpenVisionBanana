from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .common import DepthSample


def canonicalize_split_name(split: str) -> str:
    split = split.lower()
    if split == "val":
        return "test"
    return split


def expand_requested_splits(split: str) -> list[str]:
    split = canonicalize_split_name(split)
    if split == "all":
        return ["test", "train"]
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    return [split]


def resolve_dataset_root(data_root: Path) -> Path:
    data_root = data_root.expanduser()
    if (data_root / "nyu_train.txt").is_file() and (data_root / "nyu_test.txt").is_file():
        return data_root
    nested = data_root / "nyu"
    if (nested / "nyu_train.txt").is_file() and (nested / "nyu_test.txt").is_file():
        return nested
    raise FileNotFoundError(
        "Could not find NYU split files. Expected either:\n"
        f"  {data_root / 'nyu_train.txt'} and {data_root / 'nyu_test.txt'}\n"
        "or:\n"
        f"  {nested / 'nyu_train.txt'} and {nested / 'nyu_test.txt'}"
    )


def output_stem_from_rel_path(split: str, rel_path: Path) -> str:
    parts = [split, *rel_path.with_suffix("").parts]
    return "__".join(part for part in parts if part not in ("", "."))


def load_split_file(dataset_root: Path, split: str) -> list[DepthSample]:
    split_file = dataset_root / f"nyu_{split}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing NYU split file: {split_file}")

    samples = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed split entry in {split_file}:{line_number}: {line}")

            rel_image_path = Path(parts[0].lstrip("/"))
            rel_depth_path = Path(parts[1].lstrip("/"))
            focal_length = float(parts[2]) if len(parts) >= 3 else float("nan")

            image_path = dataset_root / rel_image_path
            depth_path = dataset_root / rel_depth_path
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing RGB image from split file: {image_path}")
            if not depth_path.is_file():
                raise FileNotFoundError(f"Missing depth image from split file: {depth_path}")

            samples.append(
                DepthSample(
                    dataset="nyu",
                    split=split,
                    image_path=image_path,
                    depth_path=depth_path,
                    focal_length=focal_length,
                    rel_image_path=rel_image_path,
                    output_stem=output_stem_from_rel_path(split, rel_image_path),
                    sample_id=output_stem_from_rel_path(split, rel_image_path),
                )
            )
    return samples


def build_samples(dataset_root: Path, split: str, limit: Optional[int]) -> list[DepthSample]:
    samples = []
    for split_name in expand_requested_splits(split):
        samples.extend(load_split_file(dataset_root, split_name))
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise FileNotFoundError(f"No NYU samples resolved for split={split!r}")
    return samples


def load_gt_depth(depth_path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(depth_path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr / 1000.0


def valid_depth_mask(depth_m: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    return np.isfinite(depth_m) & (depth_m > min_depth) & (depth_m < max_depth)
