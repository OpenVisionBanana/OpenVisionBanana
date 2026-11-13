"""Decode a generated instance-segmentation RGB image into per-instance masks.

Faithful implementation of the multi-stage clustering algorithm from the Vision
Banana paper (arXiv:2604.20329v3, Appendix A). The model paints each instance of
the queried noun-phrase a distinct color on a black background; this turns that
RGB image back into a list of binary instance masks.

Five stages (all distances are squared L2 in RGB space):
  1. Background:   pixel p is background if  ||C(p) - C_bg||^2 <  tau^2.
  2. Region grow:  group foreground pixels into connected blobs of near-equal
                   color. The paper's seed-relative within-tau grow is approximated
                   by quantizing colors into ~tau-wide bins and connected-component
                   labelling each bin (8-connectivity) -- an exact per-component
                   seed-grow is O(#components x N) and intractable on real outputs.
  3. Noise prune:  drop a component whose pixel area <  theta_size * H * W.
  4. Boundary prune: drop a component whose area ratio after a 3x3 erosion
                   |S (-) K| / |S|  <  theta_erosion (kills thin anti-aliasing rings).
  5. Spatial merge: merge two surviving components if their mean colors are within
                   tau^2 AND bbox_area(S_a u S_b) <= gamma * (bbox_area(S_a) +
                   bbox_area(S_b)) -- i.e. same-color fragments that are spatially
                   close (an occluded instance split in two), but not scattered
                   across the whole image.

Defaults are the paper's hyperparameters: tau=14, theta_size=2e-4,
theta_erosion=0.1, gamma=5.0, C_bg=(0,0,0).

This module is pure numpy + scipy (no torch / pycocotools) so it can be unit
tested without a GPU. RLE encoding for COCO predictions lives in `masks_to_coco`,
which imports pycocotools lazily.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from scipy import ndimage

# Paper (Appendix A) defaults.
DEFAULT_TAU = 14.0
DEFAULT_THETA_SIZE = 2e-4
DEFAULT_THETA_EROSION = 0.1
DEFAULT_GAMMA = 5.0
DEFAULT_BG_COLOR = (0, 0, 0)
DEFAULT_CONSENSUS_MODE = "inclusive_majority"
CONSENSUS_MODES = (DEFAULT_CONSENSUS_MODE, "strict_majority")


def consensus_vote_threshold(repeats: int, consensus_mode: str = DEFAULT_CONSENSUS_MODE) -> int:
    """Votes required for both instance support and pixels; empty repeats count."""
    if consensus_mode not in CONSENSUS_MODES:
        raise ValueError(f"consensus_mode must be one of {CONSENSUS_MODES}, got {consensus_mode!r}")
    if repeats < 1:
        raise ValueError("Consensus needs at least one repeat")
    return (repeats + 1) // 2 if consensus_mode == "inclusive_majority" else repeats // 2 + 1


def _as_rgb_array(image) -> np.ndarray:
    """Accept a PIL.Image or HxWx{3,4} array; return an HxWx3 float64 RGB array."""
    arr = np.asarray(image)
    if arr.ndim == 2:  # grayscale -> 3 channels
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:  # drop alpha
        arr = arr[..., :3]
    return arr.astype(np.float64)


def _connectivity_structure(connectivity: int) -> np.ndarray:
    if connectivity == 4:
        return ndimage.generate_binary_structure(2, 1)
    if connectivity == 8:
        return ndimage.generate_binary_structure(2, 2)
    raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")


def decode_instances(
    image,
    bg_color: Sequence[int] = DEFAULT_BG_COLOR,
    tau: float = DEFAULT_TAU,
    theta_size: float = DEFAULT_THETA_SIZE,
    theta_erosion: float = DEFAULT_THETA_EROSION,
    gamma: float = DEFAULT_GAMMA,
    connectivity: int = 8,
    max_components: Optional[int] = None,
) -> List[np.ndarray]:
    """Return a list of boolean HxW instance masks decoded from a painted RGB image.

    See the module docstring for the five-stage algorithm. ``connectivity`` is the
    pixel adjacency for the connected components (the paper says 16; in a 2D grid we
    use 8-connectivity, the densest standard neighbourhood). ``max_components`` caps
    how many components survive noise-pruning (largest by area kept) -- a safety
    bound for degenerate (non-paint) outputs.

    Implementation note: rather than a per-component seed-grow (O(#components x N),
    which blows up on noisy/photo-like outputs), the seed-relative within-tau grow
    is approximated by quantizing colors into ~tau-wide bins and connected-component
    labelling each bin. Stages 3-5 use bincount + find_objects so the whole decode
    is ~O(N) + O(k^2) over k surviving components (k is small).
    """
    arr = _as_rgb_array(image)
    h, w = arr.shape[:2]
    tau2 = float(tau) * float(tau)
    structure = _connectivity_structure(connectivity)
    min_area = theta_size * h * w

    # --- Stage 1: background mask ---------------------------------------------
    bg = np.asarray(bg_color, dtype=np.float64)
    foreground = ((arr - bg) ** 2).sum(axis=-1) >= tau2  # background where < tau^2

    # --- Stage 2: quantized-color connected components -------------------------
    # Bin colors to ~tau and label connected components within each bin. Bins with
    # fewer than min_area pixels can't survive noise-pruning, so skip them -- this
    # bounds the loop and makes noisy/photo outputs return [] fast.
    bin_w = max(1.0, float(tau))
    q = np.floor(arr / bin_w).astype(np.int64)
    qid = (q[..., 0] * 512 + q[..., 1]) * 512 + q[..., 2]  # 0..255/tau per chan -> fits
    labels = np.zeros((h, w), dtype=np.int64)
    offset = 0
    if foreground.any():
        uniq, counts = np.unique(qid[foreground], return_counts=True)
        for v in uniq[counts >= max(1.0, min_area)]:
            comp, ncomp = ndimage.label(foreground & (qid == v), structure=structure)
            if ncomp:
                m = comp > 0
                labels[m] = comp[m] + offset
                offset += ncomp
    if offset == 0:
        return []

    # --- Stage 3: noise pruning (areas via one bincount) ----------------------
    areas = np.bincount(labels.ravel(), minlength=offset + 1)
    keep = [l for l in range(1, offset + 1) if areas[l] >= min_area]
    if max_components is not None and len(keep) > max_components:
        keep = sorted(keep, key=lambda l: areas[l], reverse=True)[:max_components]
    if not keep:
        return []

    # Per-label bounding boxes (one pass) + mean colors (one bincount per channel).
    boxes = ndimage.find_objects(labels)
    chan_sums = [np.bincount(labels.ravel(), weights=arr[..., c].ravel(), minlength=offset + 1)
                 for c in range(3)]

    # --- Stage 4: boundary-artifact elimination (erode within each bbox) ------
    erosion_k = np.ones((3, 3), dtype=bool)
    survivors = []
    for l in keep:
        sub = labels[boxes[l - 1]] == l
        area = int(sub.sum())
        if area and int(ndimage.binary_erosion(sub, structure=erosion_k).sum()) / area >= theta_erosion:
            survivors.append(l)
    if not survivors:
        return []

    # --- Stage 5: spatially constrained merging (O(k^2), O(1) per pair) -------
    def box_tuple(l):
        r, c = boxes[l - 1]
        return r.start, r.stop, c.start, c.stop

    def box_area(b):
        return (b[1] - b[0]) * (b[3] - b[2])

    bx = {l: box_tuple(l) for l in survivors}
    barea = {l: box_area(bx[l]) for l in survivors}
    mean = {l: np.array([chan_sums[c][l] / areas[l] for c in range(3)]) for l in survivors}
    parent = {l: l for l in survivors}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(survivors)):
        a = survivors[i]
        for j in range(i + 1, len(survivors)):
            b = survivors[j]
            if float(((mean[a] - mean[b]) ** 2).sum()) > tau2:  # color far -> no merge
                continue
            ba, bb = bx[a], bx[b]
            union = (min(ba[0], bb[0]), max(ba[1], bb[1]), min(ba[2], bb[2]), max(ba[3], bb[3]))
            if box_area(union) <= gamma * (barea[a] + barea[b]):  # spatially close
                parent[find(a)] = find(b)

    groups: dict[int, list] = {}
    for l in survivors:
        groups.setdefault(find(l), []).append(l)
    return [np.isin(labels, ids) for ids in groups.values()]


def consensus_instance_masks(
    repeat_masks: Sequence[Sequence[np.ndarray]],
    match_iou: float = 0.5,
    consensus_mode: str = DEFAULT_CONSENSUS_MODE,
) -> List[np.ndarray]:
    """Majority-consensus instance masks across repeated random-draw outputs.

    Instance masks have no fixed class/instance id across generations, unlike
    semantic masks. We therefore first join masks from *different* repeats when
    their IoU is at least ``match_iou``. By default, inclusive majority retains
    objects and pixels supported by at least half the repeats (50/50 ties keep
    foreground). ``consensus_mode="strict_majority"`` requires more than half,
    discarding tied objects and pixels. Empty repeats count in the denominator;
    multiple masks from one repeat contribute only one vote per component.

    ``len(repeat_masks)==1`` returns the original masks unchanged.
    """
    if not repeat_masks:
        raise ValueError("consensus_instance_masks needs at least one repeat")
    if not 0.0 < match_iou <= 1.0:
        raise ValueError(f"match_iou must be in (0, 1], got {match_iou}")
    required = consensus_vote_threshold(len(repeat_masks), consensus_mode)
    if len(repeat_masks) == 1:
        return list(repeat_masks[0])

    shape = None
    nodes: list[tuple[int, np.ndarray]] = []
    for repeat, masks in enumerate(repeat_masks):
        for mask in masks:
            mask = np.asarray(mask, dtype=bool)
            if shape is None:
                shape = mask.shape
            elif mask.shape != shape:
                raise ValueError(f"vote over mismatched shapes: {shape} vs {mask.shape}")
            nodes.append((repeat, mask))
    if not nodes:
        return []

    parent = list(range(len(nodes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i, (repeat_i, mask_i) in enumerate(nodes):
        area_i = int(mask_i.sum())
        if not area_i:
            continue
        for j in range(i + 1, len(nodes)):
            repeat_j, mask_j = nodes[j]
            if repeat_i == repeat_j:
                continue
            area_j = int(mask_j.sum())
            if not area_j:
                continue
            intersection = int(np.logical_and(mask_i, mask_j).sum())
            union_area = area_i + area_j - intersection
            if union_area and intersection / union_area >= match_iou:
                union(i, j)

    components: dict[int, list[tuple[int, np.ndarray]]] = {}
    for i, node in enumerate(nodes):
        components.setdefault(find(i), []).append(node)

    output: list[np.ndarray] = []
    for component in components.values():
        by_repeat: dict[int, list[np.ndarray]] = {}
        for repeat, mask in component:
            by_repeat.setdefault(repeat, []).append(mask)
        if len(by_repeat) < required:
            continue

        votes = np.zeros(shape, dtype=np.uint16)
        # Multiple masks from one repeat can enter a transitive component. They
        # collectively count as one repeat vote rather than double-counting it.
        for masks in by_repeat.values():
            votes += np.logical_or.reduce(masks)
        consensus = votes >= required
        if consensus.any():
            output.append(consensus)

    return sorted(output, key=lambda mask: int(mask.sum()), reverse=True)


def masks_to_coco(
    masks: Sequence[np.ndarray],
    image_id,
    score: float = 1.0,
    category_id: int = 1,
) -> List[dict]:
    """Encode binary masks as COCO segm-result dicts (compressed RLE).

    pycocotools is imported lazily so the clustering core stays dependency-light.
    """
    import pycocotools.mask as mask_utils

    results = []
    for m in masks:
        rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
        # `counts` is bytes from pycocotools; COCO json wants a str.
        if isinstance(rle.get("counts"), bytes):
            rle["counts"] = rle["counts"].decode("ascii")
        # Emit a pixel-space bbox as a plain list. CGF1Evaluator.loadRes is called
        # once per oracle GT on the SAME prediction list; with a bbox present it
        # takes the bbox branch (which preserves our RLE segmentation) instead of
        # the segmentation branch, which would write back a numpy bbox and crash
        # the next oracle pass.
        results.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": rle,
                "bbox": [float(v) for v in mask_utils.toBbox(rle).tolist()],
                "score": float(score),
            }
        )
    return results
