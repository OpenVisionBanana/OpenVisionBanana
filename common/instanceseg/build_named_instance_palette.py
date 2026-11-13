#!/usr/bin/env python3
"""Generate the fixed ordered NAMED instance palette (run once, commit output).

Companion to build_instance_palette.py for the legend-prompt scheme: prompts now
spell out the per-instance color assignment ("the 1st is blue, the 2nd is green"),
so every palette entry needs an everyday, sayable name. Candidates are the curated
named colors from common/semseg/palette_named.json (canonical name->RGB pairs,
already mutually well separated), minus any entry closer than MIN_BG_DELTA_E to
black (drops "charcoal"), so no instance color ever collides with the background.

Ordering is deterministic -- no RNG anywhere:
  * palette[0..2] are pinned to blue, green, red (the stated convention: 1st
    instance blue, 2nd green, 3rd red),
  * the rest follow by greedy max-min (farthest-point) selection in CIELAB, so
    every prefix palette[:k] stays a well-separated k-color set for the eval
    decoder's tau=14 clustering.

    python3 common/instanceseg/build_named_instance_palette.py   # writes instance_palette_named.json

The committed instance_palette_named.json is the single source of truth shared by
dataset prep and eval prompt building; regenerate only deliberately -- it changes
the train-time color convention.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.instanceseg.colors import (  # noqa: E402
    MIN_BG_DELTA_E,
    NAMED_INSTANCE_PALETTE_PATH,
)
from common.semseg.build_palette import srgb_to_lab  # noqa: E402

SEMSEG_NAMED_PALETTE = Path(__file__).resolve().parents[1] / "semseg" / "palette_named.json"
PINNED_FIRST = ("blue", "green", "red")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=NAMED_INSTANCE_PALETTE_PATH)
    args = ap.parse_args()

    entries = json.loads(SEMSEG_NAMED_PALETTE.read_text(encoding="utf-8"))
    rgb = np.array([e["rgb"] for e in entries], dtype=np.int64)
    lab = srgb_to_lab(rgb)
    black_lab = srgb_to_lab(np.array([[0, 0, 0]]))[0]
    far = np.sqrt(((lab - black_lab) ** 2).sum(1)) >= MIN_BG_DELTA_E
    dropped = [e["name"] for e, keep in zip(entries, far) if not keep]
    entries = [e for e, keep in zip(entries, far) if keep]
    rgb, lab = rgb[far], lab[far]
    print(f"Named candidates: {len(entries)} (dropped near-black: {dropped})")

    by_name = {e["name"]: i for i, e in enumerate(entries)}
    chosen = [by_name[n] for n in PINNED_FIRST]
    # Greedy max-min over the remainder (multi-seed farthest-point; the semseg
    # helper takes a single seed index, so the loop lives here).
    d = np.min([((lab - lab[i]) ** 2).sum(axis=1) for i in chosen], axis=0)
    d[chosen] = -1.0
    while len(chosen) < len(entries):
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, ((lab - lab[nxt]) ** 2).sum(axis=1))
        d[nxt] = -1.0

    palette = [
        {"name": entries[i]["name"], "hex": entries[i]["hex"], "rgb": list(map(int, rgb[i]))}
        for i in chosen
    ]

    sel_lab = lab[chosen]
    for k in (2, 4, 8, 16, len(palette)):
        d2 = np.sqrt(((sel_lab[:k, None, :] - sel_lab[None, :k, :]) ** 2).sum(-1))
        np.fill_diagonal(d2, np.inf)
        print(f"  prefix k={k:<3} min pairwise deltaE: {d2.min():.2f}")
    print("First 10:", [p["name"] for p in palette[:10]])

    args.out.write_text(json.dumps(palette, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} ({len(palette)} colors)")


if __name__ == "__main__":
    main()
