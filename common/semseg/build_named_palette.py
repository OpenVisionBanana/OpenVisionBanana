#!/usr/bin/env python3
"""Build the curated everyday-name color palette (run once, commit output).

The full palette_150.json is well separated but xkcd-named ("baby shit green",
"snot", bare "dark"), which is fine for the rgb/hex notations (the model just
matches a number) but useless for the natural-language notation. This file is a
hand-authored set of genuinely common, sayable color names with canonical RGBs,
in priority order (most basic first). We greedily drop any color that lands within
--min-distance (CIELAB deltaE) of an already-kept higher-priority color, so the
survivors are both common AND well separated. Used only when a prompt's notation
style is "name".

    python common/semseg/build_named_palette.py        # writes palette_named.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .build_palette import srgb_to_lab
except ImportError:
    from build_palette import srgb_to_lab

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "palette_named.json"

# Everyday color names in priority order (basics first). Canonical-ish RGBs.
# Black is intentionally absent: it is reserved for background/ignore.
NAMED_COLORS: list[tuple[str, tuple[int, int, int]]] = [
    ("red", (220, 30, 30)),
    ("blue", (30, 60, 220)),
    ("green", (40, 160, 50)),
    ("yellow", (250, 225, 30)),
    ("orange", (245, 130, 30)),
    ("purple", (140, 30, 170)),
    ("pink", (250, 130, 190)),
    ("brown", (130, 75, 40)),
    ("gray", (140, 140, 140)),
    ("white", (250, 250, 250)),
    ("cyan", (40, 210, 210)),
    ("magenta", (230, 40, 200)),
    ("teal", (20, 130, 130)),
    ("navy", (20, 25, 110)),
    ("maroon", (125, 25, 30)),
    ("olive", (120, 120, 25)),
    ("beige", (235, 220, 170)),
    ("tan", (210, 180, 140)),
    ("gold", (240, 195, 20)),
    ("silver", (200, 200, 205)),
    ("turquoise", (50, 210, 190)),
    ("lavender", (200, 170, 235)),
    ("violet", (170, 70, 210)),
    ("indigo", (75, 30, 140)),
    ("crimson", (200, 20, 60)),
    ("salmon", (250, 150, 130)),
    ("coral", (255, 120, 85)),
    ("khaki", (190, 175, 110)),
    ("mint", (150, 240, 180)),
    ("peach", (255, 200, 160)),
    ("mustard", (205, 170, 30)),
    ("plum", (150, 70, 140)),
    ("orchid", (215, 120, 205)),
    ("charcoal", (70, 70, 75)),
    ("sky blue", (120, 200, 235)),
    ("dark green", (20, 90, 35)),
    ("hot pink", (255, 40, 150)),
    ("rust", (180, 70, 25)),
    ("forest green", (30, 115, 50)),
    ("royal blue", (55, 90, 210)),
    ("lime", (170, 235, 40)),
    ("burgundy", (110, 20, 50)),
    ("mauve", (200, 150, 170)),
    ("light blue", (150, 200, 255)),
    ("dark red", (140, 20, 20)),
    ("light green", (150, 230, 130)),
    ("light gray", (205, 205, 205)),
    ("dark gray", (90, 90, 90)),
    ("sea green", (45, 170, 120)),
    ("sand", (225, 205, 150)),
    ("apricot", (255, 180, 120)),
    ("emerald", (20, 180, 100)),
    ("periwinkle", (140, 150, 235)),
    ("chocolate", (110, 60, 30)),
    ("ruby", (190, 30, 70)),
    ("amber", (255, 190, 30)),
    ("jade", (40, 170, 130)),
    ("rose", (230, 110, 140)),
    ("steel blue", (70, 130, 180)),
    ("lemon", (250, 240, 100)),
    ("aqua", (90, 220, 220)),
    ("wine", (110, 30, 60)),
    ("slate", (110, 120, 140)),
    ("ivory", (250, 245, 220)),
    ("fuchsia", (240, 50, 200)),
    ("lilac", (200, 160, 220)),
    ("cream", (255, 245, 200)),
    ("scarlet", (230, 30, 30)),
    ("chartreuse", (170, 230, 40)),
    ("cobalt", (30, 70, 180)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-distance", type=float, default=20.0,
                    help="Drop a color within this CIELAB deltaE of an already-kept one.")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args()

    kept: list[tuple[str, tuple[int, int, int]]] = []
    kept_lab: list[np.ndarray] = []
    dropped: list[tuple[str, str, float]] = []
    for name, rgb in NAMED_COLORS:
        lab = srgb_to_lab(np.array(rgb))
        if kept_lab:
            dists = np.sqrt(((np.stack(kept_lab) - lab) ** 2).sum(1))
            j = int(np.argmin(dists))
            if dists[j] < args.min_distance:
                dropped.append((name, kept[j][0], float(dists[j])))
                continue
        kept.append((name, rgb))
        kept_lab.append(lab)

    palette = [{"name": n, "rgb": list(rgb),
                "hex": "#{:02X}{:02X}{:02X}".format(*rgb)} for n, rgb in kept]
    # worst-case separation among survivors
    L = np.stack(kept_lab)
    D = np.sqrt(((L[:, None] - L[None]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
    print(f"Kept {len(palette)}/{len(NAMED_COLORS)} named colors. Min pairwise deltaE: {D.min():.1f}")
    if dropped:
        print("Dropped (too close to a kept color):")
        for n, near, d in dropped:
            print(f"  {n:14s} ~ {near} (deltaE {d:.1f})")
    args.out.write_text(json.dumps(palette, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}  ({[p['name'] for p in palette]})")


if __name__ == "__main__":
    main()
