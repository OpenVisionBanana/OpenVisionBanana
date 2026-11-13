"""Shared source-group handling for the merged COCONut-XL manifest.

COCONut-XL stores COCO and Objects365 rows in one panoptic manifest.  Rows from
Objects365 carry ``object365_name``; native COCO rows do not.  Both segmentation
preparers use this module so source-selection ablations cannot diverge.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

COCONUT_XL_SOURCE_GROUPS = frozenset(("coco", "objects365"))
DEFAULT_COCONUT_XL_GROUPS = frozenset(COCONUT_XL_SOURCE_GROUPS)


def coconut_xlarge_source_group(image: Mapping[str, object]) -> str:
    """Return the canonical provenance group for one COCONut-XL image record."""
    return "objects365" if image.get("object365_name") else "coco"


def normalize_coconut_xlarge_groups(
    groups: str | Iterable[str] | None = None,
) -> frozenset[str]:
    """Normalize a comma-separated or iterable COCONut-XL provenance filter."""
    if groups is None:
        selected = DEFAULT_COCONUT_XL_GROUPS
    elif isinstance(groups, str):
        selected = frozenset(part.strip().lower() for part in groups.split(",") if part.strip())
    else:
        selected = frozenset(str(part).strip().lower() for part in groups if str(part).strip())
    if not selected:
        raise ValueError("COCONut-XL source groups cannot be empty")
    unknown = selected - COCONUT_XL_SOURCE_GROUPS
    if unknown:
        choices = ", ".join(sorted(COCONUT_XL_SOURCE_GROUPS))
        raise ValueError(f"Unknown COCONut-XL source group(s): {sorted(unknown)}; choose from {choices}")
    return selected
