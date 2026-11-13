"""Canonical COCO image holdouts for OpenVisionBanana evaluation.

COCONut-XL and PACO-LVIS contain COCO images. Their source image IDs can
therefore overlap with the held-out COCO val2017 and RefCOCOg UMD evaluation
images even though their annotations and tasks differ. Dataset preparation
uses this module to exclude those images before any training rows are made.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
# IDs extracted from the official ``annotations/instances_val2017.json`` image
# table. The committed text manifest lets preparation enforce the complete
# 5,000-image holdout without requiring the full COCO evaluation download.
COCO_VAL2017_IMAGE_IDS = REPO_ROOT / "common" / "coco_val2017_image_ids.txt"
COCO_VAL2017_IMAGE_COUNT = 5000
COCO_VAL2017_IMAGE_IDS_SHA256 = (
    "e1cc220cde76f9a1fc39aff5b5f8c0b128de8b071602e6d57f262697116e2906"
)
REFCOCOG_UMD_VAL_IMAGE_COUNT = 1300
REFCOCOG_UMD_VAL_IMAGE_IDS_SHA256 = (
    "a08ad8fb64cc7a8706896d8e9892f704e547b06f8688e30a51f4d835202fdd94"
)
DEFAULT_REFCOCOG_UMD_ROOT = REPO_ROOT / "datasets_archives" / "refseg" / "refcocog"

_COCO_IMAGE_NAME = re.compile(
    r"^(?:COCO_(?:train|val|test)\d{4}_)?(?P<image_id>\d{12})$",
    re.IGNORECASE,
)


def coco_image_id(
    value: object,
    *,
    allow_unpadded_numeric_id: bool = False,
) -> str | None:
    """Return a canonical 12-digit COCO ID from a standard COCO basename.

    COCONut stores bare ``000000123456.jpg`` names, while PACO-LVIS may retain
    the conventional ``COCO_train2014_000000123456.jpg`` prefix. Other source
    names (notably Objects365) deliberately do not match. A caller that knows
    an annotation field is a COCO ``image_id`` can opt into unpadded integer
    IDs.
    """
    if isinstance(value, bool):
        return None
    if allow_unpadded_numeric_id and isinstance(value, int) and 0 <= value < 10**12:
        return f"{value:012d}"
    stem = Path(str(value)).stem
    match = _COCO_IMAGE_NAME.fullmatch(stem)
    if match:
        return match.group("image_id")
    if allow_unpadded_numeric_id and stem.isdecimal() and len(stem) <= 12:
        return stem.zfill(12)
    return None


def _read_coco_val2017_ids(path: Path) -> frozenset[str]:
    """Load the full COCO 2017 validation population."""
    raw = path.read_bytes()
    ids = tuple(line.strip() for line in raw.decode("utf-8").splitlines() if line.strip())
    digest = hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()
    if (len(ids) != COCO_VAL2017_IMAGE_COUNT
            or len(set(ids)) != COCO_VAL2017_IMAGE_COUNT
            or ids != tuple(sorted(ids))
            or any(not image_id.isdigit() or len(image_id) != 12 for image_id in ids)
            or digest != COCO_VAL2017_IMAGE_IDS_SHA256):
        raise ValueError(
            "Invalid COCO 2017 validation ID list: expected "
            f"{COCO_VAL2017_IMAGE_COUNT} sorted IDs with SHA256 "
            f"{COCO_VAL2017_IMAGE_IDS_SHA256}, got {len(ids)} / {digest}: {path}"
        )
    return frozenset(ids)


def _read_refcocog_umd_val_ids(root: Path) -> frozenset[str]:
    refs_path = Path(root) / "refs(umd).p"
    if not refs_path.is_file():
        raise FileNotFoundError(
            "Evaluation-image exclusion requires the official RefCOCOg UMD "
            f"annotations at {refs_path}. Pass --refcocog-umd-root with the "
            "directory containing refs(umd).p, or use --allow-eval-image-overlap "
            "only for an explicitly non-evaluation training run."
        )
    with refs_path.open("rb") as handle:
        try:
            refs = pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0)
            refs = pickle.load(handle, encoding="latin1")

    ids = frozenset(
        str(ref["image_id"]).zfill(12)
        for ref in refs
        if str(ref.get("split", "")).lower() == "val" and ref.get("image_id") is not None
    )
    payload = ("\n".join(sorted(ids)) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if len(ids) != REFCOCOG_UMD_VAL_IMAGE_COUNT or digest != REFCOCOG_UMD_VAL_IMAGE_IDS_SHA256:
        raise ValueError(
            "Unexpected RefCOCOg UMD val image set: expected "
            f"{REFCOCOG_UMD_VAL_IMAGE_COUNT} IDs with SHA256 "
            f"{REFCOCOG_UMD_VAL_IMAGE_IDS_SHA256}, got {len(ids)} / {digest}: {refs_path}"
        )
    return ids


def load_coco_evaluation_holdouts(
    refcocog_umd_root: Path | str = DEFAULT_REFCOCOG_UMD_ROOT,
) -> dict[str, frozenset[str]]:
    """Load authenticated COCO image-ID sets that must not enter training."""
    return {
        "coco_val2017": _read_coco_val2017_ids(COCO_VAL2017_IMAGE_IDS),
        "refcocog_umd_val": _read_refcocog_umd_val_ids(Path(refcocog_umd_root)),
    }


def holdout_reasons(
    image_name: object,
    holdouts: Mapping[str, Iterable[str]],
    *,
    allow_unpadded_numeric_id: bool = False,
) -> tuple[str, ...]:
    """Return the evaluation sets containing this COCO source image, if any."""
    image_id = coco_image_id(
        image_name, allow_unpadded_numeric_id=allow_unpadded_numeric_id
    )
    if image_id is None:
        return ()
    return tuple(name for name, image_ids in holdouts.items() if image_id in image_ids)

@dataclass
class CocoEvaluationHoldoutAudit:
    """Track every source image excluded by the shared COCO evaluation policy.

    ``excluded_candidates`` counts every rejected annotation candidate. The
    serialized ``excluded_images`` list is deduplicated by source and canonical
    image ID, which keeps it compact for multi-expression data.
    """

    holdouts: Mapping[str, Iterable[str]]
    excluded_candidates: int = 0
    _reason_counts: Counter[str] = field(default_factory=Counter, init=False)
    _images: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict, init=False)

    def record(
        self,
        image_name: object,
        reasons: Iterable[str],
        *,
        source: str,
        reference: str | None = None,
        canonical_id: str | None = None,
    ) -> None:
        reason_tuple = tuple(sorted(set(reasons)))
        if not reason_tuple:
            return
        image_id = canonical_id or coco_image_id(image_name)
        if image_id is None:
            raise ValueError(
                "Cannot audit a COCO evaluation exclusion without a canonical image ID: "
                f"{image_name!r}"
            )
        self.excluded_candidates += 1
        self._reason_counts.update(reason_tuple)
        key = (str(source), image_id)
        self._images.setdefault(
            key,
            {
                "source": str(source),
                "canonical_coco_image_id": image_id,
                "image_reference": str(image_name),
                "reference": reference,
                "reasons": list(reason_tuple),
            },
        )

    def summary(self) -> dict[str, Any]:
        return {
            "protected_holdouts": {
                name: len(set(image_ids)) for name, image_ids in sorted(self.holdouts.items())
            },
            "excluded_candidates": self.excluded_candidates,
            "excluded_distinct_source_images": len(self._images),
            "excluded_by_reason": dict(sorted(self._reason_counts.items())),
            "excluded_images": [self._images[key] for key in sorted(self._images)],
        }

    def write(self, out_root: Path | str) -> Path:
        """Write the machine-readable exclusion audit before variants are made."""
        path = Path(out_root) / "coco_evaluation_holdout_exclusions.json"
        path.write_text(
            json.dumps(self.summary(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path
