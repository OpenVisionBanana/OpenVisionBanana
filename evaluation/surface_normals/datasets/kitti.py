from __future__ import annotations

from pathlib import Path
from typing import Optional

from .common import NormalSample


def canonicalize_split_name(split: str) -> str:
    split = split.lower()
    if split in {"val", "test", "standard", "depth_selection", "all"}:
        return "test"
    raise ValueError(f"Unsupported KITTI split: {split}")


def resolve_dataset_root(data_root: Path) -> Path:
    data_root = data_root.expanduser()
    if (data_root / "kitti_test.txt").is_file():
        return data_root
    nested = data_root / "kitti_depth_selection_surface_normals"
    if (nested / "kitti_test.txt").is_file():
        return nested
    raise FileNotFoundError(
        "Could not find prepared KITTI surface-normal split file. Expected either:\n"
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


def load_split_file(dataset_root: Path, split_file: Path, cases: tuple[str, ...]) -> list[NormalSample]:
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing KITTI split file: {split_file}")
    requested_cases = set(parse_cases(cases))

    samples: list[NormalSample] = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 8:
                raise ValueError(f"Malformed KITTI normal split entry in {split_file}:{line_number}: {line}")

            rel_image = Path(parts[0].lstrip("/"))
            rel_normal = Path(parts[1].lstrip("/"))
            rel_mask = Path(parts[2].lstrip("/"))
            rel_intrinsics = Path(parts[3].lstrip("/"))
            focal_length = float(parts[4])
            source_index = int(parts[5])
            sample_id = parts[6]
            case = parts[7]
            if case not in requested_cases:
                continue

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
                    raise FileNotFoundError(f"Missing KITTI {label} from split file: {path}")

            samples.append(
                NormalSample(
                    dataset="kitti",
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
) -> list[NormalSample]:
    canonicalize_split_name(split)
    samples = load_split_file(dataset_root, dataset_root / "kitti_test.txt", cases=cases)
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise FileNotFoundError("No KITTI surface-normal samples resolved.")
    return samples
