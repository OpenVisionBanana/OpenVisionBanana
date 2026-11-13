from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .common import DepthSample


SPLIT_ALIASES = {
    "val": "val_standard",
    "test": "val_standard",
    "standard": "val_standard",
    "debug": "val_debug",
    "small": "val_small",
    "large": "val_large",
    "giant": "val_giant",
}


def canonicalize_split_name(split: str) -> str:
    split = split.lower()
    return SPLIT_ALIASES.get(split, split)


def resolve_dataset_root(data_root: Path) -> Path:
    data_root = data_root.expanduser()
    if (data_root / "scannet_val_standard.txt").is_file():
        return data_root
    nested = data_root / "scannet_depth_normals"
    if (nested / "scannet_val_standard.txt").is_file():
        return nested
    raise FileNotFoundError(
        "Could not find prepared ScanNet split files. Expected either:\n"
        f"  {data_root / 'scannet_val_standard.txt'}\n"
        "or:\n"
        f"  {nested / 'scannet_val_standard.txt'}"
    )


def split_file_for(dataset_root: Path, split: str) -> Path:
    split_name = canonicalize_split_name(split)
    if split_name == "all":
        raise ValueError("ScanNet does not support split='all'; choose debug, small, standard, large, or giant.")
    path = dataset_root / f"scannet_{split_name}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing ScanNet split file: {path}")
    return path


def output_stem(sample_id: str) -> str:
    return f"scannet__{sample_id}"


def load_split_file(dataset_root: Path, split: str) -> list[DepthSample]:
    split_file = split_file_for(dataset_root, split)
    split_name = canonicalize_split_name(split)
    samples: list[DepthSample] = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 9:
                raise ValueError(f"Malformed ScanNet depth split entry in {split_file}:{line_number}: {line}")

            rel_image = Path(parts[0].lstrip("/"))
            rel_depth = Path(parts[1].lstrip("/"))
            rel_normal = Path(parts[2].lstrip("/"))
            rel_mask = Path(parts[3].lstrip("/"))
            rel_intrinsics = Path(parts[4].lstrip("/"))
            focal_length = float(parts[5])
            sample_id = parts[6]
            scene_id = parts[7]
            frame_id = parts[8]

            image_path = dataset_root / rel_image
            depth_path = dataset_root / rel_depth
            mask_path = dataset_root / rel_mask
            intrinsics_path = dataset_root / rel_intrinsics
            for label, path in (
                ("RGB image", image_path),
                ("depth image", depth_path),
                ("mask image", mask_path),
                ("intrinsics file", intrinsics_path),
            ):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing ScanNet {label} from split file: {path}")

            samples.append(
                DepthSample(
                    dataset="scannet",
                    split=split_name,
                    image_path=image_path,
                    depth_path=depth_path,
                    mask_path=mask_path,
                    focal_length=focal_length,
                    rel_image_path=rel_image,
                    output_stem=output_stem(sample_id),
                    sample_id=sample_id,
                    case=split_name.removeprefix("val_"),
                    metadata={
                        "normal_path": str(dataset_root / rel_normal),
                        "intrinsics_path": str(intrinsics_path),
                        "scene_id": scene_id,
                        "frame_id": frame_id,
                    },
                )
            )
    return samples


def build_samples(
    dataset_root: Path,
    split: str,
    limit: Optional[int],
    cases: tuple[str, ...],
    aux_augmentations: tuple[str, ...],
) -> list[DepthSample]:
    samples = load_split_file(dataset_root, split)
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise FileNotFoundError(f"No ScanNet depth samples resolved for split={split!r}")
    return samples


def load_gt_depth(sample: DepthSample) -> np.ndarray:
    arr = np.asarray(Image.open(sample.depth_path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr / 1000.0


def valid_depth_mask(
    sample: DepthSample,
    depth_m: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    if sample.mask_path is None:
        mask = np.ones(depth_m.shape, dtype=bool)
    else:
        mask_arr = np.asarray(Image.open(sample.mask_path).convert("L"), dtype=np.uint8)
        if mask_arr.shape != depth_m.shape:
            mask_arr = np.asarray(
                Image.fromarray(mask_arr, mode="L").resize((depth_m.shape[1], depth_m.shape[0]), resample=Image.Resampling.NEAREST),
                dtype=np.uint8,
            )
        mask = mask_arr > 0
    return mask & np.isfinite(depth_m) & (depth_m > min_depth) & (depth_m < max_depth)
