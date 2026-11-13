from __future__ import annotations

from pathlib import Path
from typing import Optional

from .common import NormalSample


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


def load_split_file(dataset_root: Path, split: str) -> list[NormalSample]:
    split_file = split_file_for(dataset_root, split)
    split_name = canonicalize_split_name(split)
    samples: list[NormalSample] = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) not in {2} and len(parts) < 9:
                raise ValueError(f"Malformed ScanNet normal split entry in {split_file}:{line_number}: {line}")

            if len(parts) == 2:
                rel_image = Path(parts[0].lstrip("/"))
                rel_normal = Path(parts[1].lstrip("/"))
                image_path = dataset_root / rel_image
                normal_path = dataset_root / rel_normal
                for label, path in (("RGB image", image_path), ("normal array", normal_path)):
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing ScanNet {label} from split file: {path}")
                sample_id = "__".join(rel_image.with_suffix("").parts)
                samples.append(
                    NormalSample(
                        dataset="scannet",
                        split=split_name,
                        image_path=image_path,
                        normal_path=normal_path,
                        mask_path=None,
                        focal_length=float("nan"),
                        source_index=line_number - 1,
                        rel_image_path=rel_image,
                        output_stem=output_stem(sample_id),
                        sample_id=sample_id,
                        case="standard",
                        metadata={"ground_truth_protocol": "marigold_framenet_float_npy"},
                    )
                )
                continue

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
            normal_path = dataset_root / rel_normal
            mask_path = dataset_root / rel_mask
            intrinsics_path = dataset_root / rel_intrinsics
            for label, path in (
                ("RGB image", image_path),
                ("normal image", normal_path),
                ("mask image", mask_path),
                ("intrinsics file", intrinsics_path),
            ):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing ScanNet {label} from split file: {path}")

            samples.append(
                NormalSample(
                    dataset="scannet",
                    split=split_name,
                    image_path=image_path,
                    normal_path=normal_path,
                    mask_path=mask_path,
                    focal_length=focal_length,
                    source_index=None,
                    rel_image_path=rel_image,
                    output_stem=output_stem(sample_id),
                    sample_id=sample_id,
                    case=split_name.removeprefix("val_"),
                    metadata={
                        "depth_path": str(dataset_root / rel_depth),
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
) -> list[NormalSample]:
    samples = load_split_file(dataset_root, split)
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise FileNotFoundError(f"No ScanNet surface-normal samples resolved for split={split!r}")
    return samples
