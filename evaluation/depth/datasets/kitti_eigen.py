"""KITTI Eigen-652 adapter for the frozen Marigold evaluation archive.

This is deliberately separate from :mod:`kitti`, which is the KITTI depth
selection benchmark with 1,000 samples.  The two assets must never share a
benchmark label.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from evaluation.depth.datasets.common import DepthSample
else:
    from .common import DepthSample

EXPECTED_SAMPLE_COUNT = 652
DEFAULT_FOCAL_LENGTH = 721.5377


def canonicalize_split_name(split: str) -> str:
    if split.lower() in {"val", "test", "standard", "all"}:
        return "test"
    raise ValueError(f"Unsupported KITTI Eigen split: {split}")


def _looks_like_marigold_root(root: Path) -> bool:
    return any(root.glob("*_drive_*_sync/proj_depth/groundtruth/image_02/*.png"))


def resolve_dataset_root(data_root: Path) -> Path:
    data_root = data_root.expanduser()
    candidates = (
        data_root,
        data_root / "kitti_eigen_652_garg",
        data_root / "kitti_eigen_split_test",
    )
    for candidate in candidates:
        if candidate.is_dir() and _looks_like_marigold_root(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find the frozen Marigold KITTI Eigen-652 layout under "
        f"{data_root}. Expected *_drive_*_sync/proj_depth/groundtruth/image_02/*.png."
    )


def parse_cases(cases: tuple[str, ...]) -> tuple[str, ...]:
    requested = tuple(case.lower() for case in cases if case)
    if not requested or "all" in requested or "standard" in requested:
        return ("eigen_652",)
    unknown = set(requested) - {"eigen_652"}
    if unknown:
        raise ValueError(
            f"Unsupported KITTI Eigen case(s): {', '.join(sorted(unknown))}"
        )
    return requested


def build_samples(
    dataset_root: Path,
    split: str,
    limit: Optional[int],
    cases: tuple[str, ...],
    aux_augmentations: tuple[str, ...],
) -> list[DepthSample]:
    del aux_augmentations
    canonicalize_split_name(split)
    if "eigen_652" not in parse_cases(cases):
        return []

    samples: list[DepthSample] = []
    gt_paths = sorted(
        dataset_root.glob("*_drive_*_sync/proj_depth/groundtruth/image_02/*.png")
    )
    for depth_path in gt_paths:
        drive = depth_path.parents[3].name
        date = drive[:10]
        image_path = (
            dataset_root
            / date
            / drive
            / "image_02"
            / "data"
            / depth_path.name
        )
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Missing KITTI Eigen RGB pair for {depth_path}: {image_path}"
            )
        sample_id = f"{drive}_image_02_{depth_path.stem}"
        samples.append(
            DepthSample(
                dataset="kitti_eigen",
                split="test",
                image_path=image_path,
                depth_path=depth_path,
                mask_path=None,
                focal_length=DEFAULT_FOCAL_LENGTH,
                rel_image_path=image_path.relative_to(dataset_root),
                output_stem=f"kitti_eigen__eigen_652__{sample_id}",
                sample_id=sample_id,
                case="eigen_652",
                metadata={
                    "asset_family": "marigold_kitti_eigen_split_test",
                    "crop": "garg",
                    "max_depth_m": "80",
                },
            )
        )

    if len(samples) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            "KITTI Eigen asset identity failure: expected "
            f"{EXPECTED_SAMPLE_COUNT} paired samples, found {len(samples)}"
        )
    if limit is not None:
        samples = samples[:limit]
    return samples


def load_gt_depth(sample: DepthSample) -> np.ndarray:
    depth = np.asarray(Image.open(sample.depth_path), dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth / 256.0


def valid_depth_mask(
    sample: DepthSample,
    depth_m: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    del sample
    return (
        np.isfinite(depth_m)
        & (depth_m > min_depth)
        & (depth_m < max_depth)
    )
