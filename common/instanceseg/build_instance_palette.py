#!/usr/bin/env python3
"""Generate the fixed ORDERED instance palette (run once, commit output).

palette[i] is ALWAYS the color of the i-th instance in canonical raster order
(common/instanceseg/ordering.py). Selection is farthest-point sampling in CIELAB
over the near-black-free candidate grid from colors.py, seeded from the most
chromatic candidate (deterministic -- no RNG anywhere), so every prefix
palette[:k] is itself a maximally-separated k-color set. That prefix property is
what keeps targets decodable at any instance count under the eval decoder's
tau=14 color clustering.

    python3 common/instanceseg/build_instance_palette.py            # writes instance_palette.json
    python3 common/instanceseg/build_instance_palette.py --k 64

The committed instance_palette.json is the single source of truth shared by
dataset prep (and any future eval tooling); regenerate only deliberately -- it
changes the train-time color convention.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.instanceseg.colors import (  # noqa: E402
    CANDIDATE_LAB,
    CANDIDATE_RGB,
    INSTANCE_PALETTE_PATH,
)
from common.semseg.build_palette import farthest_point_sample  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=64, help="Number of palette colors.")
    ap.add_argument("--out", type=Path, default=INSTANCE_PALETTE_PATH)
    args = ap.parse_args()

    rgb, lab = CANDIDATE_RGB, CANDIDATE_LAB
    print(f"Candidate grid colors (near-black excluded): {len(rgb)}")

    # Seed from the most chromatic candidate so the first picks are vivid/distinct
    # (same convention as common/semseg/build_palette.py). argmax over the fixed
    # grid is tie-deterministic (first index wins).
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    seed = int(np.argmax(chroma))
    idx = farthest_point_sample(lab, args.k, seed)

    palette = [
        {"hex": "#{:02X}{:02X}{:02X}".format(*rgb[i]), "rgb": list(map(int, rgb[i]))}
        for i in idx
    ]

    # Sanity: min pairwise deltaE of each prefix = worst-case separation the
    # decoder sees when an image has that many instances.
    sel_lab = lab[idx]
    for k in (2, 4, 8, 16, 32, len(palette)):
        if k > len(palette):
            continue
        d = np.sqrt(((sel_lab[:k, None, :] - sel_lab[None, :k, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        print(f"  prefix k={k:<3} min pairwise deltaE: {d.min():.2f}")
    print("First 8:", [p["hex"] for p in palette[:8]])

    args.out.write_text(json.dumps(palette, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
