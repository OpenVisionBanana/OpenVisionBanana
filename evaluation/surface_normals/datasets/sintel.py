from __future__ import annotations

from pathlib import Path
from typing import Optional

from .common import NormalSample


def canonicalize_split_name(split: str) -> str:
    split = split.lower()
    if split in {"test", "val", "standard", "all"}:
        return "test"
    raise ValueError(f"Unsupported Sintel split: {split}")


def resolve_dataset_root(data_root: Path) -> Path:
    data_root = data_root.expanduser()
    if (data_root / "sintel_test.txt").is_file():
        return data_root
    nested = data_root / "sintel_surface_normals"
    if (nested / "sintel_test.txt").is_file():
        return nested
    raise FileNotFoundError(
        "Could not find Sintel surface-normal split file. Expected either:\n"
        f"  {data_root / 'sintel_test.txt'}\n"
        "or:\n"
        f"  {nested / 'sintel_test.txt'}"
    )


def output_stem(rel_path: Path) -> str:
    return "__".join(("test", *rel_path.with_suffix("").parts))


def build_samples(
    dataset_root: Path,
    split: str,
    limit: Optional[int],
) -> list[NormalSample]:
    canonicalize_split_name(split)
    split_file = dataset_root / "sintel_test.txt"
    samples: list[NormalSample] = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Malformed Sintel split entry in {split_file}:{line_number}: {line}"
                )
            rel_image = Path(parts[0].lstrip("/"))
            rel_normal = Path(parts[1].lstrip("/"))
            image_path = dataset_root / rel_image
            normal_path = dataset_root / rel_normal
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing Sintel RGB image: {image_path}")
            if not normal_path.is_file():
                raise FileNotFoundError(f"Missing Sintel normal target: {normal_path}")
            sample_id = output_stem(rel_image)
            samples.append(
                NormalSample(
                    dataset="sintel",
                    split="test",
                    image_path=image_path,
                    normal_path=normal_path,
                    mask_path=None,
                    focal_length=float("nan"),
                    source_index=len(samples),
                    rel_image_path=rel_image,
                    output_stem=sample_id,
                    sample_id=sample_id,
                    case="standard",
                )
            )
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise FileNotFoundError("No Sintel surface-normal samples resolved.")
    return samples
