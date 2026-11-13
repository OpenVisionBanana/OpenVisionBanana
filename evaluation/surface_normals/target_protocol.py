"""Resolve a frozen surface-normal target archive into a scoring contract.

The evaluator must not guess an axis transform from the dataset name. Exact
benchmark ``.npy`` targets already contain physical XYZ vectors. RGB targets
must declare both their encoding and whether their stored vectors are already
in the canonical camera convention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from common.surface_normal_encoding import (
    LEGACY_DIRECT_NORMAL_ENCODING,
    SUPPORTED_NORMAL_ENCODINGS,
    VISION_BANANA_NORMAL_ENCODING,
)

AUTO_NORMAL_ENCODING = "auto"
CANONICAL_NORMAL_CONVENTION = "camera_x_right_y_up_z_toward_camera"
IDENTITY_PERMUTATION = (0, 1, 2)
IDENTITY_SIGNS = (1, 1, 1)


@dataclass(frozen=True)
class GroundTruthNormalProtocol:
    encoding: str
    permutation: tuple[int, int, int]
    signs: tuple[int, int, int]
    storage_formats: tuple[str, ...]
    source: str

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


def _triplet(value: Any, *, signs: bool, field: str) -> tuple[int, int, int]:
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",")]
    try:
        result = tuple(int(part) for part in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if len(result) != 3:
        raise ValueError(f"Invalid {field}: expected three values, got {value!r}")
    if signs and any(part not in (-1, 1) for part in result):
        raise ValueError(f"Invalid {field}: signs must be -1 or 1, got {value!r}")
    if not signs and sorted(result) != [0, 1, 2]:
        raise ValueError(f"Invalid {field}: expected a permutation of 0,1,2, got {value!r}")
    return result  # type: ignore[return-value]


def _metadata_encoding(info: Mapping[str, Any]) -> str | None:
    declared = info.get("stored_normal_encoding", info.get("normal_encoding"))
    if declared in SUPPORTED_NORMAL_ENCODINGS:
        return str(declared)
    text = " ".join(
        str(info.get(key, ""))
        for key in ("normal_encoding", "normal_encoding_formula", "prepared_normal_encoding")
    ).lower().replace(" ", "")
    if "(1-x)/2" in text or "x_left_red" in text:
        return VISION_BANANA_NORMAL_ENCODING
    if "normal_xyz*0.5+0.5" in text or "(normal_xyz+1)/2" in text or "direct_xyz" in text:
        return LEGACY_DIRECT_NORMAL_ENCODING
    return None


def resolve_ground_truth_protocol(
    *,
    dataset_info: Mapping[str, Any],
    storage_formats: Sequence[str],
    requested_encoding: str = AUTO_NORMAL_ENCODING,
    requested_permutation: tuple[int, int, int] | None = None,
    requested_signs: tuple[int, int, int] | None = None,
) -> GroundTruthNormalProtocol:
    """Resolve decoding and stored-to-canonical axes without dataset-name guesses."""
    formats = tuple(sorted({str(value).lower().lstrip(".") for value in storage_formats}))
    if not formats:
        raise ValueError("Cannot resolve a normal protocol without target storage formats")

    is_float_xyz = set(formats) <= {"npy", "npz", "exr"}
    if is_float_xyz:
        encoding = "float_xyz_npy" if set(formats) <= {"npy", "npz"} else "float_xyz"
        source = "float XYZ target; RGB decoding is not applicable"
    else:
        if requested_encoding != AUTO_NORMAL_ENCODING:
            if requested_encoding not in SUPPORTED_NORMAL_ENCODINGS:
                raise ValueError(f"Unsupported requested GT normal encoding: {requested_encoding!r}")
            encoding = requested_encoding
            source = "explicit command-line override"
        else:
            encoding = _metadata_encoding(dataset_info) or ""
            if not encoding:
                raise ValueError(
                    "RGB surface-normal targets have no unambiguous normal encoding in "
                    "dataset_info.json/manifest.json. Rebuild the archive with release metadata "
                    "or pass --gt_normal_encoding explicitly for a diagnostic legacy run."
                )
            source = "dataset metadata"

    if requested_permutation is not None:
        permutation = _triplet(requested_permutation, signs=False, field="GT normal permutation")
        axes_source = "explicit command-line override"
    elif "gt_to_canonical_permutation" in dataset_info:
        permutation = _triplet(
            dataset_info["gt_to_canonical_permutation"], signs=False,
            field="gt_to_canonical_permutation",
        )
        axes_source = "dataset metadata"
    else:
        permutation = IDENTITY_PERMUTATION
        axes_source = "physical XYZ/canonical stored target"

    if requested_signs is not None:
        signs_value = _triplet(requested_signs, signs=True, field="GT normal signs")
        axes_source = "explicit command-line override"
    elif "gt_to_canonical_signs" in dataset_info:
        signs_value = _triplet(
            dataset_info["gt_to_canonical_signs"], signs=True,
            field="gt_to_canonical_signs",
        )
        axes_source = "dataset metadata"
    else:
        signs_value = IDENTITY_SIGNS

    stored_convention = dataset_info.get(
        "stored_normal_convention", dataset_info.get("canonical_normal_convention")
    )
    if (
        not is_float_xyz
        and requested_encoding == AUTO_NORMAL_ENCODING
        and requested_permutation is None
        and requested_signs is None
        and "gt_to_canonical_permutation" not in dataset_info
        and "gt_to_canonical_signs" not in dataset_info
        and stored_convention != CANONICAL_NORMAL_CONVENTION
    ):
        raise ValueError(
            f"RGB target convention {stored_convention!r} is not declared canonical and "
            "does not declare gt_to_canonical_permutation/signs. Rebuild the archive with "
            "release metadata or pass explicit GT encoding and axes for a diagnostic run."
        )

    return GroundTruthNormalProtocol(
        encoding=encoding,
        permutation=permutation,
        signs=signs_value,
        storage_formats=formats,
        source=f"{source}; axes: {axes_source}",
    )
