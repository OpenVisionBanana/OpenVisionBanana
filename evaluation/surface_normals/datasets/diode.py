from __future__ import annotations

from pathlib import Path
from typing import Optional

from .common import NormalSample


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
    nested = data_root / "diode_surface_normals"
    if (nested / "diode_indoor.txt").is_file() and (nested / "diode_outdoor.txt").is_file():
        return nested
    raise FileNotFoundError(
        "Could not find prepared DIODE surface-normal split files. Expected either:\n"
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


def split_files_for(dataset_root: Path, cases: tuple[str, ...]) -> list[Path]:
    return [dataset_root / f"diode_{case}.txt" for case in parse_cases(cases)]


def output_stem(case: str, sample_id: str) -> str:
    return f"diode__{case}__{sample_id}"


def load_split_file(dataset_root: Path, split_file: Path) -> list[NormalSample]:
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing DIODE split file: {split_file}")

    samples: list[NormalSample] = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) not in {2} and len(parts) < 7:
                raise ValueError(f"Malformed DIODE normal split entry in {split_file}:{line_number}: {line}")

            if len(parts) == 2:
                rel_image = Path(parts[0].lstrip("/"))
                rel_normal = Path(parts[1].lstrip("/"))
                image_path = dataset_root / rel_image
                normal_path = dataset_root / rel_normal
                for label, path in (("RGB image", image_path), ("normal array", normal_path)):
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing DIODE {label} from split file: {path}")
                sample_id = rel_image.stem
                case = split_file.stem.removeprefix("diode_")
                samples.append(
                    NormalSample(
                        dataset="diode",
                        split="test",
                        image_path=image_path,
                        normal_path=normal_path,
                        mask_path=None,
                        focal_length=float("nan"),
                        source_index=line_number - 1,
                        rel_image_path=rel_image,
                        output_stem=output_stem(case, sample_id),
                        sample_id=sample_id,
                        case=case,
                        metadata={"ground_truth_protocol": "official_diode_float_npy"},
                    )
                )
                continue

            rel_image = Path(parts[0].lstrip("/"))
            rel_normal = Path(parts[1].lstrip("/"))
            rel_mask = Path(parts[2].lstrip("/"))
            focal_length = float(parts[3])
            source_index = int(parts[4])
            sample_id = parts[5]
            case = parts[6]

            image_path = dataset_root / rel_image
            normal_path = dataset_root / rel_normal
            mask_path = dataset_root / rel_mask
            for label, path in (("RGB image", image_path), ("normal image", normal_path), ("mask image", mask_path)):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing DIODE {label} from split file: {path}")

            samples.append(
                NormalSample(
                    dataset="diode",
                    split="test",
                    image_path=image_path,
                    normal_path=normal_path,
                    mask_path=mask_path,
                    focal_length=focal_length,
                    source_index=source_index,
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
) -> list[NormalSample]:
    canonicalize_split_name(split)
    samples: list[NormalSample] = []
    for split_file in split_files_for(dataset_root, cases=cases):
        samples.extend(load_split_file(dataset_root, split_file))
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise FileNotFoundError("No DIODE surface-normal samples resolved.")
    return samples
