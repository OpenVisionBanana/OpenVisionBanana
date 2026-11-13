from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .common import DepthSample


def canonicalize_split_name(split: str) -> str:
    split = split.lower()
    if split in {"val", "test", "standard"}:
        return "test"
    if split == "all":
        return "all"
    raise ValueError(f"Unsupported DIODE split: {split}")


def resolve_dataset_root(data_root: Path) -> Path:
    data_root = data_root.expanduser()
    if (data_root / "diode_indoor.txt").is_file() and (data_root / "diode_outdoor.txt").is_file():
        return data_root
    nested = data_root / "diode_depth"
    if (nested / "diode_indoor.txt").is_file() and (nested / "diode_outdoor.txt").is_file():
        return nested
    raise FileNotFoundError(
        "Could not find prepared DIODE depth split files. Expected either:\n"
        f"  {data_root / 'diode_indoor.txt'} and {data_root / 'diode_outdoor.txt'}\n"
        "or:\n"
        f"  {nested / 'diode_indoor.txt'} and {nested / 'diode_outdoor.txt'}"
    )


def parse_cases(cases: tuple[str, ...]) -> tuple[str, ...]:
    requested = tuple(case.lower() for case in cases if case)
    if not requested or "all" in requested or "standard" in requested:
        return ("indoor", "outdoor")
    unknown = set(requested) - {"indoor", "outdoor"}
    if unknown:
        raise ValueError(f"Unsupported DIODE case(s): {', '.join(sorted(unknown))}")
    return requested


def output_stem(case: str, sample_id: str) -> str:
    return f"diode__{case}__{sample_id}"


def split_files_for(dataset_root: Path, cases: tuple[str, ...]) -> list[Path]:
    return [dataset_root / f"diode_{case}.txt" for case in parse_cases(cases)]


def load_split_file(dataset_root: Path, split_file: Path) -> list[DepthSample]:
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing DIODE split file: {split_file}")

    samples: list[DepthSample] = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Malformed DIODE depth split entry in {split_file}:{line_number}: {line}")

            rel_image = Path(parts[0].lstrip("/"))
            rel_depth = Path(parts[1].lstrip("/"))
            rel_mask = Path(parts[2].lstrip("/"))
            focal_length = float(parts[3])
            sample_id = parts[4]
            case = parts[5]

            image_path = dataset_root / rel_image
            depth_path = dataset_root / rel_depth
            mask_path = dataset_root / rel_mask
            for label, path in (("RGB image", image_path), ("depth npy", depth_path), ("mask image", mask_path)):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing DIODE {label} from split file: {path}")

            samples.append(
                DepthSample(
                    dataset="diode",
                    split="test",
                    image_path=image_path,
                    depth_path=depth_path,
                    mask_path=mask_path,
                    focal_length=focal_length,
                    rel_image_path=rel_image,
                    output_stem=output_stem(case, sample_id),
                    sample_id=sample_id,
                    case=case,
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
    samples: list[DepthSample] = []
    for split_file in split_files_for(dataset_root, cases=cases):
        samples.extend(load_split_file(dataset_root, split_file))
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise FileNotFoundError("No DIODE depth samples resolved.")
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
