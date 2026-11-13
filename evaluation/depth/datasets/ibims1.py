from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .common import DepthSample


DEFAULT_AUX_AUGMENTATION_GROUPS = ("appearance",)


def canonicalize_split_name(split: str) -> str:
    split = split.lower()
    if split in {"val", "test", "standard"}:
        return "test"
    if split == "all":
        return "all"
    raise ValueError(f"Unsupported iBims-1 split: {split}")


def resolve_dataset_root(data_root: Path) -> Path:
    data_root = data_root.expanduser()
    if (data_root / "ibims1_standard.txt").is_file():
        return data_root
    nested = data_root / "ibims1_depth"
    if (nested / "ibims1_standard.txt").is_file():
        return nested
    raise FileNotFoundError(
        "Could not find prepared iBims-1 depth split files. Expected either:\n"
        f"  {data_root / 'ibims1_standard.txt'}\n"
        "or:\n"
        f"  {nested / 'ibims1_standard.txt'}"
    )


def parse_cases(cases: tuple[str, ...]) -> tuple[str, ...]:
    if not cases:
        return ("standard",)
    if "all" in cases:
        return ("standard", "aux_augmented")
    return cases


def expand_aux_augmentations(dataset_root: Path, aux_augmentations: tuple[str, ...]) -> set[str]:
    requested = set(aux_augmentations or DEFAULT_AUX_AUGMENTATION_GROUPS)
    if "all" in requested:
        return {"all"}
    return requested


def output_stem(case: str, sample_id: str, augmentation: str | None) -> str:
    parts = ["ibims1", case]
    if augmentation:
        parts.append(augmentation)
    parts.append(sample_id)
    return "__".join(parts)


def split_files_for(
    dataset_root: Path,
    cases: tuple[str, ...],
    aux_augmentations: tuple[str, ...],
) -> list[Path]:
    files: list[Path] = []
    parsed_cases = parse_cases(cases)
    if "standard" in parsed_cases:
        files.append(dataset_root / "ibims1_standard.txt")
    if "aux_augmented" in parsed_cases:
        for group in sorted(expand_aux_augmentations(dataset_root, aux_augmentations)):
            files.append(dataset_root / "splits" / f"ibims1_aux_augmented_{group}.txt")
    return files


def load_split_file(dataset_root: Path, split_file: Path) -> list[DepthSample]:
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing iBims-1 split file: {split_file}")

    samples: list[DepthSample] = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                raise ValueError(f"Malformed iBims-1 split entry in {split_file}:{line_number}: {line}")

            rel_image = Path(parts[0].lstrip("/"))
            rel_depth = Path(parts[1].lstrip("/"))
            rel_mask = Path(parts[2].lstrip("/"))
            focal_length = float(parts[3])
            sample_id = parts[4]
            case = parts[5]
            augmentation = parts[6]
            rel_geometry = Path(parts[7].lstrip("/")) if len(parts) >= 8 and parts[7] != "-" else None

            image_path = dataset_root / rel_image
            depth_path = dataset_root / rel_depth
            mask_path = dataset_root / rel_mask
            for label, path in (("RGB image", image_path), ("depth image", depth_path), ("mask image", mask_path)):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing iBims-1 {label} from split file: {path}")
            geometry_path = dataset_root / rel_geometry if rel_geometry is not None else None
            if geometry_path is not None and not geometry_path.is_file():
                raise FileNotFoundError(f"Missing iBims-1 geometry metadata from split file: {geometry_path}")

            aug_value = None if augmentation == "-" else augmentation
            metadata = {"augmentation": augmentation}
            if geometry_path is not None:
                metadata["geometry_path"] = str(geometry_path)
            samples.append(
                DepthSample(
                    dataset="ibims1",
                    split="test",
                    image_path=image_path,
                    depth_path=depth_path,
                    mask_path=mask_path,
                    focal_length=focal_length,
                    rel_image_path=rel_image,
                    output_stem=output_stem(case, sample_id, aug_value),
                    sample_id=sample_id,
                    case=case,
                    metadata=metadata,
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
    for split_file in split_files_for(dataset_root, cases=cases, aux_augmentations=aux_augmentations):
        samples.extend(load_split_file(dataset_root, split_file))
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise FileNotFoundError("No iBims-1 samples resolved.")
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
