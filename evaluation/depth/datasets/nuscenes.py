from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .common import DepthSample


def canonicalize_split_name(split: str) -> str:
    split = split.lower()
    if split == "val":
        return "val"
    if split in {"test", "standard"}:
        return "test"
    if split == "all":
        return "all"
    raise ValueError(f"Unsupported nuScenes split: {split}")


def prepared_split(dataset_root: Path) -> Optional[str]:
    info_path = dataset_root / "dataset_info.json"
    if not info_path.is_file():
        return None
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    split = str(info.get("split", "")).lower()
    if split in {"test", "val"}:
        return split
    if info.get("version") == "v1.0-mini":
        return "test"
    return None


def validate_requested_split(dataset_root: Path, split: str) -> None:
    actual = prepared_split(dataset_root)
    if split != "all" and actual is not None and split != actual:
        raise ValueError(
            f"Requested nuScenes split {split!r}, but the prepared archive contains split {actual!r}: {dataset_root}"
        )


def resolve_dataset_root(data_root: Path) -> Path:
    data_root = data_root.expanduser()
    if any((data_root / name).is_file() for name in ("nuscenes_test.txt", "nuscenes_val.txt", "nuscenes_all.txt")):
        return data_root
    candidates = sorted(data_root.glob("nuscenes_*_depth"))
    for candidate in candidates:
        if any((candidate / name).is_file() for name in ("nuscenes_test.txt", "nuscenes_val.txt", "nuscenes_all.txt")):
            return candidate
    nested = data_root / "nuscenes_trainval_val_cam_front_depth"
    if (nested / "nuscenes_val.txt").is_file() or (nested / "nuscenes_all.txt").is_file():
        return nested
    raise FileNotFoundError(
        "Could not find prepared nuScenes depth split files. Expected either:\n"
        f"  {data_root / 'nuscenes_val.txt'}\n"
        "or a nested prepared folder such as:\n"
        f"  {nested / 'nuscenes_val.txt'}"
    )


def parse_cases(cases: tuple[str, ...]) -> tuple[str, ...]:
    requested = tuple(case.lower() for case in cases if case)
    if not requested or "all" in requested or "standard" in requested:
        return ("all",)
    return requested


def split_files_for(dataset_root: Path, split: str, cases: tuple[str, ...]) -> list[Path]:
    requested = parse_cases(cases)
    if requested == ("all",):
        split_file = dataset_root / f"nuscenes_{split}.txt"
        return [split_file if split_file.is_file() else dataset_root / "nuscenes_all.txt"]
    return [dataset_root / f"nuscenes_{case}.txt" for case in requested]


def output_stem(case: str, sample_id: str) -> str:
    return f"nuscenes__{case}__{sample_id}"


def load_split_file(dataset_root: Path, split_file: Path, split: str) -> list[DepthSample]:
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing nuScenes depth split file: {split_file}")

    samples: list[DepthSample] = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                raise ValueError(f"Malformed nuScenes depth split entry in {split_file}:{line_number}: {line}")

            rel_image = Path(parts[0].lstrip("/"))
            rel_depth = Path(parts[1].lstrip("/"))
            rel_mask = Path(parts[2].lstrip("/"))
            rel_intrinsics = Path(parts[3].lstrip("/"))
            focal_length = float(parts[4])
            sample_id = parts[5]
            case = parts[6]

            image_path = dataset_root / rel_image
            depth_path = dataset_root / rel_depth
            mask_path = dataset_root / rel_mask
            intrinsics_path = dataset_root / rel_intrinsics
            for label, path in (
                ("RGB image", image_path),
                ("depth npy", depth_path),
                ("mask image", mask_path),
                ("intrinsics file", intrinsics_path),
            ):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing nuScenes {label} from split file: {path}")

            samples.append(
                DepthSample(
                    dataset="nuscenes",
                    split=split,
                    image_path=image_path,
                    depth_path=depth_path,
                    mask_path=mask_path,
                    focal_length=focal_length,
                    rel_image_path=rel_image,
                    output_stem=output_stem(case, sample_id),
                    sample_id=sample_id,
                    case=case,
                    metadata={"intrinsics_path": str(intrinsics_path)},
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
    split = canonicalize_split_name(split)
    validate_requested_split(dataset_root, split)
    samples: list[DepthSample] = []
    for split_file in split_files_for(dataset_root, split=split, cases=cases):
        samples.extend(load_split_file(dataset_root, split_file, split=split))
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise FileNotFoundError("No nuScenes depth samples resolved.")
    return samples


def load_gt_depth(sample: DepthSample) -> np.ndarray:
    return np.asarray(np.load(sample.depth_path), dtype=np.float32).squeeze()


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
