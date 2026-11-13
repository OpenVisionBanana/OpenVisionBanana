from __future__ import annotations

from pathlib import Path
from typing import Optional

from .common import NormalSample


def canonicalize_split_name(split: str) -> str:
    split = split.lower()
    if split in {"val", "test", "standard", "clone_camera_0", "clone_camera0", "all"}:
        return "test"
    raise ValueError(f"Unsupported VKITTI2 split: {split}")


def resolve_dataset_root(data_root: Path) -> Path:
    data_root = data_root.expanduser()
    if (data_root / "vkitti2_test.txt").is_file():
        return data_root
    for nested_name in ("vkitti2_clone_camera_0_surface_normals", "vkitti2_clone_camera0_surface_normals"):
        nested = data_root / nested_name
        if (nested / "vkitti2_test.txt").is_file():
            return nested
    matches = sorted(path for path in data_root.glob("vkitti2*_surface_normals") if (path / "vkitti2_test.txt").is_file())
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "Could not find prepared VKITTI2 surface-normal split file. Expected either:\n"
        f"  {data_root / 'vkitti2_test.txt'}\n"
        "or a nested vkitti2*_surface_normals directory containing vkitti2_test.txt."
    )


def normalize_case(case: str) -> str:
    return case.lower().replace("camera0", "camera_0").replace("camera1", "camera_1")


def parse_cases(cases: tuple[str, ...]) -> tuple[str, ...] | None:
    requested = tuple(normalize_case(case) for case in cases if case)
    if "all" in requested:
        return None
    if not requested or "standard" in requested:
        return ("clone_camera_0",)
    return requested


def output_stem(case: str, sample_id: str) -> str:
    return f"vkitti2__{case}__{sample_id}"


def load_split_file(dataset_root: Path, split_file: Path, cases: tuple[str, ...]) -> list[NormalSample]:
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing VKITTI2 split file: {split_file}")
    requested = parse_cases(cases)
    requested_cases = set(requested) if requested is not None else None

    samples: list[NormalSample] = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) not in {2} and len(parts) < 8:
                raise ValueError(f"Malformed VKITTI2 normal split entry in {split_file}:{line_number}: {line}")

            if len(parts) == 2:
                rel_image = Path(parts[0].lstrip("/"))
                rel_normal = Path(parts[1].lstrip("/"))
                image_path = dataset_root / rel_image
                normal_path = dataset_root / rel_normal
                for label, path in (("RGB image", image_path), ("normal array", normal_path)):
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing VKITTI2 {label} from split file: {path}")
                sample_id = "__".join(rel_image.with_suffix("").parts)
                samples.append(
                    NormalSample(
                        dataset="vkitti2",
                        split="test",
                        image_path=image_path,
                        normal_path=normal_path,
                        mask_path=None,
                        focal_length=float("nan"),
                        source_index=line_number - 1,
                        rel_image_path=rel_image,
                        output_stem=output_stem("dsine_reference", sample_id),
                        sample_id=sample_id,
                        case="dsine_reference",
                        metadata={"ground_truth_protocol": "dsine_plane_svd_float_npy"},
                    )
                )
                continue

            rel_image = Path(parts[0].lstrip("/"))
            rel_normal = Path(parts[1].lstrip("/"))
            rel_mask = Path(parts[2].lstrip("/"))
            rel_intrinsics = Path(parts[3].lstrip("/"))
            focal_length = float(parts[4])
            source_index = int(parts[5])
            sample_id = parts[6]
            case = normalize_case(parts[7])
            if requested_cases is not None and case not in requested_cases:
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
                    raise FileNotFoundError(f"Missing VKITTI2 {label} from split file: {path}")

            samples.append(
                NormalSample(
                    dataset="vkitti2",
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
    samples = load_split_file(dataset_root, dataset_root / "vkitti2_test.txt", cases=cases)
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise FileNotFoundError("No VKITTI2 surface-normal samples resolved.")
    return samples
