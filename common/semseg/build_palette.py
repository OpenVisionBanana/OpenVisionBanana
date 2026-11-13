#!/usr/bin/env python3
"""Generate the 200-color semantic-segmentation palette (run once, commit output).

This is the FULL palette, used for the rgb/hex notation styles and for painting the
corresponding masks (names don't matter there -- the model just matches a number).
The natural-language ("name") notation instead uses the small, clean, everyday set
in build_named_palette.py. 200 colors gives the per-image sampler headroom even for
ADE20k images, which can contain a large fraction of the 150 ADE classes.

Goal: 200 colors that are (a) mutually well separated so Flux can paint and we can
decode them, and (b) clear of near-black, which is reserved for background/ignore.
We draw from matplotlib's 949 XKCD survey colors (each has a name + hex), drop
near-black entries, and greedily farthest-point-sample 200 in CIELAB (perceptual)
so the selection is as separable as the gamut allows. The result is written
name/hex/rgb in selection order, so the first K entries are themselves a
well-separated K-color sub-palette.

    python common/semseg/build_palette.py                  # writes palette_200.json
    python common/semseg/build_palette.py --n 200 --min-brightness 60

The committed palette_200.json is the single source of truth shared by dataset prep
and evaluation; regenerate only deliberately (it changes the train/eval color set).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "palette_200.json"


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Vectorized sRGB(0-255) -> CIELAB (D65). rgb: (...,3) -> (...,3)."""
    srgb = rgb.astype(np.float64) / 255.0
    # sRGB -> linear
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    # linear RGB -> XYZ (D65)
    m = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = lin @ m.T
    # normalize by D65 white
    white = np.array([0.95047, 1.0, 1.08883])
    xyz = xyz / white
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def load_named_colors() -> list[tuple[str, str, tuple[int, int, int]]]:
    """(name, hex, rgb) from matplotlib's XKCD survey colors, deduped by rgb."""
    import matplotlib.colors as mc

    out: dict[tuple[int, int, int], tuple[str, str]] = {}
    for raw_name, hexstr in mc.XKCD_COLORS.items():
        name = raw_name.split(":", 1)[1] if ":" in raw_name else raw_name
        rgb = tuple(int(round(c * 255)) for c in mc.to_rgb(hexstr))
        out.setdefault(rgb, (name, hexstr.upper()))  # first name wins on dup rgb
    return [(nm, hx, rgb) for rgb, (nm, hx) in out.items()]


def farthest_point_sample(lab: np.ndarray, n: int, seed_idx: int) -> list[int]:
    """Greedy max-min (farthest-point) selection of n indices in LAB space."""
    chosen = [seed_idx]
    # min squared distance of every point to the chosen set, updated incrementally
    d = ((lab - lab[seed_idx]) ** 2).sum(axis=1)
    while len(chosen) < min(n, len(lab)):
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, ((lab - lab[nxt]) ** 2).sum(axis=1))
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="Number of colors to select.")
    ap.add_argument("--min-brightness", type=int, default=60,
                    help="Drop colors with R+G+B below this (keeps black unambiguous for background/ignore).")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args()

    named = load_named_colors()
    rgb = np.array([c[2] for c in named], dtype=np.int64)
    keep = rgb.sum(axis=1) >= args.min_brightness
    named = [c for c, k in zip(named, keep) if k]
    rgb = rgb[keep]
    print(f"Candidate named colors after dropping near-black (<{args.min_brightness}): {len(named)}")

    lab = srgb_to_lab(rgb)
    # Seed from the most chromatic color so the first picks are vivid/distinct.
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    seed = int(np.argmax(chroma))
    idx = farthest_point_sample(lab, args.n, seed)

    palette = [
        {"name": named[i][0], "hex": named[i][1], "rgb": list(map(int, rgb[i]))}
        for i in idx
    ]

    # Report the worst-case separation actually achieved (sanity on decodability).
    sel_lab = lab[idx]
    dmat = np.sqrt(((sel_lab[:, None, :] - sel_lab[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dmat, np.inf)
    print(f"Selected {len(palette)} colors. Min pairwise LAB distance (deltaE): {dmat.min():.2f}")
    print("First 8:", [(p["name"], p["hex"]) for p in palette[:8]])

    args.out.write_text(json.dumps(palette, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
