from __future__ import annotations

from pathlib import Path
from typing import Optional

from .common import NormalSample


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
    if any(data_root.glob("*_img.png")) and any(data_root.glob("*_normal.npy")):
        return data_root
    nested = data_root / "ibims1_surface_normals"
    if (nested / "ibims1_standard.txt").is_file():
        return nested
    marigold_nested = data_root / "ibims"
    if any(marigold_nested.glob("*_img.png")) and any(
        marigold_nested.glob("*_normal.npy")
    ):
        return marigold_nested
    raise FileNotFoundError(
        "Could not find prepared iBims-1 surface-normal split files. Expected either:\n"
        f"  {data_root / 'ibims1_standard.txt'}\n"
        "or:\n"
        f"  {nested / 'ibims1_standard.txt'}\n"
        "or a Marigold reference directory containing paired *_img.png and "
        "*_normal.npy files."
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


def output_stem(case: str, sample_id: str, augmentation: str | None) -> str:
    parts = ["ibims1", case]
    if augmentation:
        parts.append(augmentation)
    parts.append(sample_id)
    return "__".join(parts)


def load_split_file(dataset_root: Path, split_file: Path) -> list[NormalSample]:
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing iBims-1 split file: {split_file}")

    samples: list[NormalSample] = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 8:
                raise ValueError(f"Malformed iBims-1 normal split entry in {split_file}:{line_number}: {line}")

            rel_image = Path(parts[0].lstrip("/"))
            rel_normal = Path(parts[1].lstrip("/"))
            rel_mask = Path(parts[2].lstrip("/"))
            focal_length = float(parts[3])
            source_index = int(parts[4])
            sample_id = parts[5]
            case = parts[6]
            augmentation = parts[7]

            image_path = dataset_root / rel_image
            normal_path = dataset_root / rel_normal
            mask_path = dataset_root / rel_mask
            for label, path in (("RGB image", image_path), ("normal image", normal_path), ("mask image", mask_path)):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing iBims-1 {label} from split file: {path}")

            aug_value = None if augmentation == "-" else augmentation
            samples.append(
                NormalSample(
                    dataset="ibims1",
                    split="test",
                    image_path=image_path,
                    normal_path=normal_path,
                    mask_path=mask_path,
                    focal_length=focal_length,
                    source_index=source_index,
                    rel_image_path=rel_image,
                    output_stem=output_stem(case, sample_id, aug_value),
                    sample_id=sample_id,
                    case=case,
                    metadata={"augmentation": augmentation},
                )
            )
    return samples


def load_marigold_reference(dataset_root: Path) -> list[NormalSample]:
    """Load Marigold/DSINE's published iBims-1 float-normal layout.

    The validity mask is implicit in each float array: finite normals with
    non-zero norm are valid.  No TUM depth-validity or transparency mask is
    applied, matching the Marigold evaluator.
    """
    samples: list[NormalSample] = []
    for source_index, image_path in enumerate(sorted(dataset_root.glob("*_img.png"))):
        sample_id = image_path.name.removesuffix("_img.png")
        normal_path = dataset_root / f"{sample_id}_normal.npy"
        if not normal_path.is_file():
            raise FileNotFoundError(
                f"Missing Marigold iBims-1 normal for {image_path}: {normal_path}"
            )
        samples.append(
            NormalSample(
                dataset="ibims1",
                split="test",
                image_path=image_path,
                normal_path=normal_path,
                mask_path=None,
                focal_length=float("nan"),
                source_index=source_index,
                rel_image_path=Path(image_path.name),
                output_stem=output_stem("standard", sample_id, None),
                sample_id=sample_id,
                case="standard",
                metadata={"ground_truth_protocol": "marigold_dsine_float_npy"},
            )
        )
    if not samples:
        raise FileNotFoundError(
            f"No Marigold iBims-1 *_img.png files found in {dataset_root}"
        )
    return samples


def build_samples(
    dataset_root: Path,
    split: str,
    limit: Optional[int],
    cases: tuple[str, ...],
    aux_augmentations: tuple[str, ...],
) -> list[NormalSample]:
    canonicalize_split_name(split)
    if not (dataset_root / "ibims1_standard.txt").is_file():
        parsed_cases = parse_cases(cases)
        if parsed_cases != ("standard",):
            raise ValueError(
                "The Marigold iBims-1 reference layout only contains the standard case"
            )
        samples = load_marigold_reference(dataset_root)
        if limit is not None:
            samples = samples[:limit]
        return samples
    samples: list[NormalSample] = []
    for split_file in split_files_for(dataset_root, cases=cases, aux_augmentations=aux_augmentations):
        samples.extend(load_split_file(dataset_root, split_file))
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise FileNotFoundError("No iBims-1 surface-normal samples resolved.")
    return samples
