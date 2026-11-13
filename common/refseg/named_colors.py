"""Palette of common, well-separated NAMED colors for referring segmentation.

Everyday color names (red, blue, teal, ...) curated to sit far apart in CIELAB,
so the generator can reliably tell them apart and the decoder can snap a
prediction back to the intended one. The refseg evaluator
(evaluation/refseg/inference_flux2_klein_refseg.py) uses this to paint the
referred region a random, well-separated named color instead of always white.

Data: common/refseg/palette_named.json (same curated set as the semseg name
palette; regenerate both with common/semseg/build_named_palette.py if changed).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
NAMED_COLORS: list[dict] = json.loads((HERE / "palette_named.json").read_text(encoding="utf-8"))
NAMES: list[str] = [c["name"] for c in NAMED_COLORS]
RGB = np.array([c["rgb"] for c in NAMED_COLORS], dtype=np.int64)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Vectorized sRGB(0-255) -> CIELAB (D65). Mirrors common/semseg/build_palette.py."""
    srgb = rgb.astype(np.float64) / 255.0
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    m = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = (lin @ m.T) / np.array([0.95047, 1.0, 1.08883])
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


LAB = srgb_to_lab(RGB.astype(np.float64))


def sample(rng: random.Random, k: int = 1, min_delta_e: float = 25.0
           ) -> list[tuple[str, tuple[int, int, int]]]:
    """Pick ``k`` distinct named colors, greedily keeping each at least
    ``min_delta_e`` (CIELAB) from those already chosen; relax if the palette
    cannot satisfy it. Returns (name, rgb) pairs. ``k=1`` is just a uniform pick.
    """
    order = list(range(len(RGB)))
    rng.shuffle(order)
    chosen: list[int] = []
    chosen_lab: list[np.ndarray] = []
    leftover: list[int] = []
    min_sq = min_delta_e * min_delta_e
    for i in order:
        if chosen_lab and min(((np.stack(chosen_lab) - LAB[i]) ** 2).sum(1)) < min_sq:
            leftover.append(i)
            continue
        chosen.append(i)
        chosen_lab.append(LAB[i])
        if len(chosen) == k:
            break
    for i in leftover:  # relax: allow closer colors rather than return fewer than k
        if len(chosen) == k:
            break
        chosen.append(i)
    return [(NAMED_COLORS[i]["name"], tuple(int(v) for v in RGB[i])) for i in chosen[:k]]


def sample_one(rng: random.Random) -> tuple[str, tuple[int, int, int]]:
    """Convenience: a single random named color (name, rgb)."""
    return sample(rng, 1)[0]


if __name__ == "__main__":
    rng = random.Random(0)
    print(f"{len(NAMED_COLORS)} named colors; sample of 5:", sample(rng, 5))
