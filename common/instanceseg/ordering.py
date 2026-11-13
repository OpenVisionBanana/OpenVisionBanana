"""Canonical (deterministic) instance ordering for instance-segmentation targets.

The training target must be a pure function of (image, noun-phrase): the eval
decoder (evaluation/instanceseg/decode.py) is permutation-invariant, but the
flow-matching MSE loss is not -- if the instance->color mapping is arbitrary, the
loss punishes valid outputs and the model regresses toward the only consistent
part of the target (the black background). So instances are ranked by a fixed,
image-derived rule and painted with the committed ordered palette
(common/instanceseg/instance_palette.json): canonical instance #i always gets
palette[i].

The rule: sort by the mask's *anchor*, the raster-first foreground pixel
(topmost row, then leftmost column in that row). Integer-valued and tie-free --
instance masks of one class are pairwise disjoint (all three readers in
dataset_prep/instanceseg/instance_taxonomy.py partition pixel values), so no two
masks can share an anchor.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def mask_anchor(mask: np.ndarray, scan: str = "tb") -> Tuple[int, int]:
    """The scan-first True pixel of a 2-D bool mask, as a sortable tuple.

    scan="tb" (default): (row, col) -- lexicographic min over (row, col): topmost
    row, then leftmost column within it (top-to-bottom scan, ties left-to-right).
    scan="lr": (col, row) -- leftmost column containing the mask, then topmost
    row within that column (left-to-right scan, ties top-to-bottom). Both are
    deterministic on the pixel grid (no float ties), and disjoint masks can never
    share an anchor under either scan.
    """
    if scan == "tb":
        rows = np.flatnonzero(mask.any(axis=1))
        if len(rows) == 0:
            raise ValueError("mask_anchor: empty mask")
        r = int(rows[0])
        return r, int(np.flatnonzero(mask[r])[0])
    if scan == "lr":
        cols = np.flatnonzero(mask.any(axis=0))
        if len(cols) == 0:
            raise ValueError("mask_anchor: empty mask")
        c = int(cols[0])
        return c, int(np.flatnonzero(mask[:, c])[0])
    raise ValueError(f"unknown scan direction: {scan!r} (expected 'tb' or 'lr')")


def canonical_instance_order(masks: Sequence[np.ndarray], scan: str = "tb") -> List[int]:
    """Indices of ``masks`` sorted by anchor under the given scan direction.

    scan="tb": top-to-bottom, ties left-to-right (the default convention).
    scan="lr": left-to-right, ties top-to-bottom (the fixeddraw_32_lr variant).
    Disjoint masks guarantee unique anchors, so the order is total and stable.
    """
    return sorted(range(len(masks)), key=lambda i: mask_anchor(masks[i], scan))


def select_instances(
    masks: Sequence[np.ndarray], max_instances: Optional[int]
) -> List[np.ndarray]:
    """Deterministic cap: keep the ``max_instances`` largest masks by pixel area.

    Ties broken by anchor so the selection is byte-reproducible. Large instances
    are the ones actually renderable/learnable at training resolution (same
    rationale as the min_instance_area filter); a raster-first cap would instead
    bias toward tiny top-of-frame objects. The returned list is in the original
    sequence order -- callers re-derive canonical_instance_order on the survivors.
    """
    if max_instances is None or len(masks) <= max_instances:
        return list(masks)
    ranked = sorted(
        range(len(masks)),
        key=lambda i: (-int(masks[i].sum()), mask_anchor(masks[i])),
    )
    keep = sorted(ranked[:max_instances])  # original order, selection is what matters
    return [masks[i] for i in keep]
