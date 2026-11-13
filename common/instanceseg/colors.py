"""Fixed ORDERED instance palette for instance-segmentation targets.

The model must paint each instance of the queried noun-phrase a *distinct* color,
and the eval decoder (evaluation/instanceseg/decode.py) recovers instances by
clustering colors within tau=14 RGB of each other. The instance->color mapping
is deterministic (an arbitrary mapping is an ambiguous target for the
element-wise flow-matching loss):

  * ``instance_palette.json`` (committed; regenerate only deliberately via
    build_instance_palette.py) holds K colors in **farthest-point-sampling
    order** over the candidate grid below, so every prefix palette[:k] is itself
    a maximally-separated k-color set -- exactly what typical instance counts
    need for the decoder's clustering.
  * Canonical instance #i (raster order of mask anchors, see
    common/instanceseg/ordering.py) is ALWAYS painted palette[i].

Candidates exclude colors close to black so they never collide with the
background (C_bg = (0,0,0)); they reuse common.semseg.build_palette.srgb_to_lab
for CIELAB.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from common.semseg.build_palette import srgb_to_lab

BACKGROUND_COLOR = (0, 0, 0)
# Minimum CIELAB distance any instance color must keep from black, so a dark color
# is never snapped to background by the decoder's tau=14 background test.
MIN_BG_DELTA_E = 40.0

INSTANCE_PALETTE_PATH = Path(__file__).resolve().parent / "instance_palette.json"
# Named variant (legend-prompt scheme): 36 everyday-named colors, blue/green/red
# first, used for BOTH notation styles whenever an image has <= 36 instances. The
# 64-color grid palette above remains the fallback for k > 36 (rgb notation only).
NAMED_INSTANCE_PALETTE_PATH = Path(__file__).resolve().parent / "instance_palette_named.json"


def _build_candidates() -> tuple[np.ndarray, np.ndarray]:
    """A dense, evenly-spaced RGB grid (minus near-black), with its CIELAB coords."""
    levels = np.array([0, 36, 72, 109, 145, 182, 218, 255], dtype=np.int64)  # 8^3 = 512
    grid = np.array(np.meshgrid(levels, levels, levels, indexing="ij")).reshape(3, -1).T
    lab = srgb_to_lab(grid)
    black_lab = srgb_to_lab(np.array([BACKGROUND_COLOR]))[0]
    far_from_black = np.sqrt(((lab - black_lab) ** 2).sum(1)) >= MIN_BG_DELTA_E
    return grid[far_from_black], lab[far_from_black]


CANDIDATE_RGB, CANDIDATE_LAB = _build_candidates()

_PALETTE_CACHE: List[Tuple[int, int, int]] | None = None
_NAMED_PALETTE_CACHE: List[dict] | None = None


def load_instance_palette(path: Path = INSTANCE_PALETTE_PATH) -> List[Tuple[int, int, int]]:
    """The committed ordered palette as [(r, g, b), ...] (cached for the default path)."""
    global _PALETTE_CACHE
    if path == INSTANCE_PALETTE_PATH and _PALETTE_CACHE is not None:
        return _PALETTE_CACHE
    entries = json.loads(path.read_text(encoding="utf-8"))
    palette = [tuple(int(v) for v in e["rgb"]) for e in entries]
    if path == INSTANCE_PALETTE_PATH:
        _PALETTE_CACHE = palette
    return palette


def load_named_instance_palette() -> List[dict]:
    """The committed ordered NAMED palette as [{"name", "hex", "rgb"}, ...] (cached).

    palette[i] is the color of canonical instance #i whenever the image has at most
    len(palette) (=36) instances, regardless of prompt notation (name or rgb) -- the
    notation only changes how the color is written in the prompt, never the pixels.
    """
    global _NAMED_PALETTE_CACHE
    if _NAMED_PALETTE_CACHE is None:
        entries = json.loads(NAMED_INSTANCE_PALETTE_PATH.read_text(encoding="utf-8"))
        _NAMED_PALETTE_CACHE = [
            {"name": e["name"], "hex": e["hex"], "rgb": tuple(int(v) for v in e["rgb"])}
            for e in entries
        ]
    return _NAMED_PALETTE_CACHE


_NAMED_SEP_ORDER_CACHE: List[int] | None = None


def named_separation_order() -> List[int]:
    """Pure farthest-point order over the named palette (no pinning).

    The committed instance_palette_named.json pins blue/green/red first (the
    legend convention) before its FPS tail. The fixed-K draw modes instead want a
    strict most->least-separated enumeration, so this recomputes the greedy
    max-min CIELAB order from scratch, seeded from the most chromatic entry --
    the same convention as common/semseg/diversify._separation_order and
    build_instance_palette.py. Deterministic (no RNG); cached. The committed
    file itself is untouched, so the legend scheme keeps its convention.
    """
    global _NAMED_SEP_ORDER_CACHE
    if _NAMED_SEP_ORDER_CACHE is None:
        lab = srgb_to_lab(np.array([e["rgb"] for e in load_named_instance_palette()],
                                   dtype=np.int64))
        chroma = np.hypot(lab[:, 1], lab[:, 2])
        chosen = [int(np.argmax(chroma))]
        d = ((lab - lab[chosen[0]]) ** 2).sum(axis=1)
        d[chosen[0]] = -1.0
        while len(chosen) < len(lab):
            nxt = int(np.argmax(d))
            chosen.append(nxt)
            d = np.minimum(d, ((lab - lab[nxt]) ** 2).sum(axis=1))
            d[nxt] = -1.0
        _NAMED_SEP_ORDER_CACHE = chosen
    return _NAMED_SEP_ORDER_CACHE


def ordered_instance_colors(k: int) -> List[Tuple[int, int, int]]:
    """First ``k`` colors of the committed ordered palette.

    palette[i] is always the color of canonical instance #i. ``k`` beyond the
    palette cycles (with duplicate colors) -- dataset prep clamps max_instances
    to the palette size, so this is a never-expected safety net.
    """
    if k <= 0:
        return []
    palette = load_instance_palette()
    if k > len(palette):
        import warnings

        warnings.warn(
            f"ordered_instance_colors: k={k} exceeds palette size {len(palette)}; "
            "cycling colors (duplicate instance colors!)"
        )
        palette = palette * (k // len(palette) + 1)
    return palette[:k]
