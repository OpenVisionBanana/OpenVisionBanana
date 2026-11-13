"""Exact dataset adapters for the Marigold V2 / PPD depth table."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from evaluation.depth.ppd_protocol import canonical_dataset_name

from .common import DepthSample


EXPECTED_COUNTS = {
    "nyu": 654,
    "kitti_eigen": 652,
    "eth3d": 454,
    "scannet": 800,
    "diode": 771,
}

SPLIT_FILES = {
    "nyu": "nyu_depth/labeled/filename_list_test.txt",
    "kitti_eigen": "kitti_depth/eigen_test_files_with_gt.txt",
    "eth3d": "eth3d_depth/eth3d_filename_list.txt",
    "scannet": "scannet_depth/scannet_val_sampled_list_800_1.txt",
    "diode": "diode_depth/diode_val_all_filename_list.txt",
}

DATA_DIRS = {
    "nyu": "nyuv2",
    "kitti_eigen": "kitti",
    "eth3d": "eth3d",
    "scannet": "scannet",
    "diode": "diode",
}


def resolve_dataset_root(dataset: str, data_root: Path) -> Path:
    dataset = canonical_dataset_name(dataset)
    root = data_root.expanduser()
    candidates = (root / DATA_DIRS[dataset], root)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing Marigold depth dataset {DATA_DIRS[dataset]!r} under {root}"
    )


def build_samples(
    dataset: str,
    dataset_root: Path,
    split_root: Path,
    limit: Optional[int],
) -> list[DepthSample]:
    dataset = canonical_dataset_name(dataset)
    split_file = split_root / SPLIT_FILES[dataset]
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing Marigold split file: {split_file}")

    lines = split_file.read_text(encoding="utf-8").splitlines()
    valid_rows = [line.split() for line in lines if len(line.split()) >= 2 and line.split()[1] != "None"]
    expected = EXPECTED_COUNTS[dataset]
    if len(valid_rows) != expected or len({row[0] for row in valid_rows}) != expected:
        raise ValueError(f"Marigold {dataset} split identity failure: expected {expected} unique samples")
    samples: list[DepthSample] = []
    for line_number, line in enumerate(lines, start=1):
        columns = line.split()
        if len(columns) < 2 or columns[1] == "None":
            continue
        if limit is not None and len(samples) >= limit:
            continue
        rel_image = Path(columns[0])
        rel_depth = Path(columns[1])
        rel_mask = Path(columns[2]) if dataset == "diode" else None
        image_path = dataset_root / rel_image
        depth_path = dataset_root / rel_depth
        mask_path = dataset_root / rel_mask if rel_mask is not None else None
        required = [image_path, depth_path]
        if mask_path is not None:
            required.append(mask_path)
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing Marigold benchmark member from {split_file}:{line_number}: "
                + ", ".join(str(path) for path in missing)
            )

        case = "standard"
        if dataset == "diode":
            case = "indoor" if columns[0].startswith("indoors/") else "outdoor"
        sample_id = f"{dataset}_{line_number:06d}_{rel_image.stem}"
        samples.append(
            DepthSample(
                dataset=dataset,
                split="test",
                image_path=image_path,
                depth_path=depth_path,
                mask_path=mask_path,
                focal_length=float(columns[2])
                if dataset == "kitti_eigen" and len(columns) > 2
                else float("nan"),
                rel_image_path=rel_image,
                output_stem=f"marigold_ppd__{sample_id}",
                sample_id=sample_id,
                case=case,
                metadata={
                    "benchmark_protocol": "marigold_v2_ppd_ransac_log_v1",
                    "split_file": str(split_file),
                    "split_line": str(line_number),
                },
            )
        )

    if len(samples) != (expected if limit is None else min(expected, limit)):
        raise ValueError(
            f"Marigold {dataset} split identity failure: expected {expected} valid "
            f"samples, resolved {len(samples)}"
        )
    return samples if limit is None else samples[:limit]


def prepare_input_image(sample: DepthSample, image: Image.Image) -> Image.Image:
    dataset = canonical_dataset_name(sample.dataset)
    if dataset == "kitti_eigen":
        width, height = image.size
        if width < 1216 or height < 352:
            raise ValueError(f"KITTI image is too small for benchmark crop: {image.size}")
        left = int((width - 1216) / 2)
        top = height - 352
        return image.crop((left, top, left + 1216, top + 352))
    if dataset == "eth3d":
        return image.resize((1008, 672), resample=Image.Resampling.LANCZOS)
    return image


def load_gt_depth(sample: DepthSample) -> np.ndarray:
    dataset = canonical_dataset_name(sample.dataset)
    if dataset == "diode":
        return np.asarray(np.load(sample.depth_path), dtype=np.float32).squeeze()
    if dataset == "eth3d":
        depth = np.fromfile(sample.depth_path, dtype=np.float32).copy()
        depth[~np.isfinite(depth)] = 0.0
        return depth.reshape(4032, 6048)

    depth = np.asarray(Image.open(sample.depth_path), dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if dataset == "kitti_eigen":
        depth = depth / 256.0
        height, width = depth.shape
        left = int((width - 1216) / 2)
        top = height - 352
        return depth[top : top + 352, left : left + 1216]
    return depth / 1000.0


def valid_depth_mask(sample: DepthSample, depth: np.ndarray) -> np.ndarray:
    dataset = canonical_dataset_name(sample.dataset)
    if dataset == "diode":
        if sample.mask_path is None:
            raise ValueError("DIODE PPD sample is missing its official mask")
        return np.asarray(np.load(sample.mask_path), dtype=bool).squeeze()

    mask = np.isfinite(depth) & (depth > 0)
    if dataset == "nyu":
        crop = np.zeros(depth.shape, dtype=bool)
        crop[45:471, 41:601] = True
        mask &= crop
    elif dataset == "kitti_eigen":
        height, width = depth.shape
        crop = np.zeros(depth.shape, dtype=bool)
        crop[
            int(0.3324324 * height) : int(0.91351351 * height),
            int(0.0359477 * width) : int(0.96405229 * width),
        ] = True
        mask &= crop
    return mask
