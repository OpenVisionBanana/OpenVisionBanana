from __future__ import annotations

from pathlib import Path
from typing import Optional

from .common import NormalSample


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
    if (data_root / "nyu_train.txt").is_file() or (data_root / "nyu_test.txt").is_file():
        return data_root
    nested = data_root / "nyu_surface_normals"
    if (nested / "nyu_train.txt").is_file() or (nested / "nyu_test.txt").is_file():
        return nested
    raise FileNotFoundError(
        "Could not find NYU surface-normal split files. Expected either:\n"
        f"  {data_root / 'nyu_train.txt'} or {data_root / 'nyu_test.txt'}\n"
        "or:\n"
        f"  {nested / 'nyu_train.txt'} or {nested / 'nyu_test.txt'}"
    )


def output_stem_from_rel_path(split: str, rel_path: Path) -> str:
    parts = [split, *rel_path.with_suffix("").parts]
    return "__".join(part for part in parts if part not in ("", "."))


def looks_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def load_split_file(dataset_root: Path, split: str) -> list[NormalSample]:
    split_file = dataset_root / f"nyu_{split}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing NYU surface-normal split file: {split_file}")

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
            rel_normal_path = Path(parts[1].lstrip("/"))
            rel_mask_path: Optional[Path] = None
            focal_length = float("nan")
            source_index: Optional[int] = None

            cursor = 2
            if cursor < len(parts) and not looks_float(parts[cursor]):
                rel_mask_path = Path(parts[cursor].lstrip("/"))
                cursor += 1
            if cursor < len(parts) and looks_float(parts[cursor]):
                focal_length = float(parts[cursor])
                cursor += 1
            if cursor < len(parts):
                try:
                    source_index = int(parts[cursor])
                except ValueError:
                    source_index = None

            image_path = dataset_root / rel_image_path
            normal_path = dataset_root / rel_normal_path
            mask_path = dataset_root / rel_mask_path if rel_mask_path is not None else None
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing RGB image from split file: {image_path}")
            if not normal_path.is_file():
                raise FileNotFoundError(f"Missing normal image from split file: {normal_path}")
            if mask_path is not None and not mask_path.is_file():
                raise FileNotFoundError(f"Missing normal mask from split file: {mask_path}")

            samples.append(
                NormalSample(
                    dataset="nyu",
                    split=split,
                    image_path=image_path,
                    normal_path=normal_path,
                    mask_path=mask_path,
                    focal_length=focal_length,
                    source_index=source_index,
                    rel_image_path=rel_image_path,
                    output_stem=output_stem_from_rel_path(split, rel_image_path),
                    sample_id=output_stem_from_rel_path(split, rel_image_path),
                )
            )
    return samples


def build_samples(dataset_root: Path, split: str, limit: Optional[int]) -> list[NormalSample]:
    samples = []
    for split_name in expand_requested_splits(split):
        samples.extend(load_split_file(dataset_root, split_name))
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise FileNotFoundError(f"No NYU surface-normal samples resolved for split={split!r}")
    return samples
