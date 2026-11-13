from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .common import DepthSample


def canonicalize_split_name(split: str) -> str:
    split = split.lower()
    if split in {"val", "test", "standard", "depth_selection", "all"}:
        return "test"
    raise ValueError(f"Unsupported KITTI split: {split}")


def resolve_dataset_root(data_root: Path) -> Path:
    data_root = data_root.expanduser()
    if (data_root / "kitti_test.txt").is_file():
        return data_root
    nested = data_root / "kitti_depth_selection"
    if (nested / "kitti_test.txt").is_file():
        return nested
    raise FileNotFoundError(
        "Could not find prepared KITTI depth split file. Expected either:\n"
        f"  {data_root / 'kitti_test.txt'}\n"
        "or:\n"
        f"  {nested / 'kitti_test.txt'}"
    )


def parse_cases(cases: tuple[str, ...]) -> tuple[str, ...]:
    requested = tuple(case.lower() for case in cases if case)
    if not requested or "all" in requested or "standard" in requested:
        return ("depth_selection",)
    unknown = set(requested) - {"depth_selection"}
    if unknown:
        raise ValueError(f"Unsupported KITTI case(s): {', '.join(sorted(unknown))}")
    return requested


def output_stem(case: str, sample_id: str) -> str:
    return f"kitti__{case}__{sample_id}"


def load_split_file(dataset_root: Path, split_file: Path, cases: tuple[str, ...]) -> list[DepthSample]:
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing KITTI split file: {split_file}")
    requested_cases = set(parse_cases(cases))

    samples: list[DepthSample] = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                raise ValueError(f"Malformed KITTI depth split entry in {split_file}:{line_number}: {line}")

            rel_image = Path(parts[0].lstrip("/"))
            rel_depth = Path(parts[1].lstrip("/"))
            rel_mask = Path(parts[2].lstrip("/"))
            rel_intrinsics = Path(parts[3].lstrip("/"))
            focal_length = float(parts[4])
            sample_id = parts[5]
            case = parts[6]
            if case not in requested_cases:
                continue

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
                    raise FileNotFoundError(f"Missing KITTI {label} from split file: {path}")

            samples.append(
                DepthSample(
                    dataset="kitti",
                    split="test",
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
    canonicalize_split_name(split)
    samples = load_split_file(dataset_root, dataset_root / "kitti_test.txt", cases=cases)
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise FileNotFoundError("No KITTI depth samples resolved.")
    return samples


def load_gt_depth(sample: DepthSample) -> np.ndarray:
    arr = np.asarray(Image.open(sample.depth_path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr / 256.0


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
