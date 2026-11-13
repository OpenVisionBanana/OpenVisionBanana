#!/usr/bin/env python3
"""Experimental randomdraw eval with MORE SEPARATED per-image colors.

Hypothesis under test: randomdraw mIoU is low partly because the 19 per-image
colors sit close together (the production greedy sampler only reaches ~26 deltaE
min separation for 19 colors), so a small color drift in the model output snaps
to a neighboring class. This script re-runs the exact same randomdraw evaluation
but draws the 19 colors with MAX separation, to see if drift tolerance lifts mIoU.

It does NOT modify the production evaluator. It imports
inference_partial_flux2_klein_img2img and monkeypatches only its color sampler,
then calls its main() unchanged. Pass all the usual CLI args (--prompt_style
randomdraw, --out_dir <fresh dir>, etc.) -- they are parsed by the base script.

Sampler is chosen via env vars (kept out of the base argparse):
    RANDOMDRAW_COLOR_SELECT = farthest (default) | greedy
    RANDOMDRAW_MIN_DELTA_E  = greedy floor, default 35   (ignored by farthest)

farthest:  max-min (farthest-point) selection from a random per-image start;
           maximizes the minimum pairwise CIELAB separation (~40-43 deltaE for
           19 colors vs ~26 greedy), and -- unlike raising the greedy floor --
           does not collapse to near-duplicate colors at high separation.
greedy:    the production sampler but with a configurable (higher) deltaE floor.

Colors stay deterministic per image (same seed scheme as production), so the
legends/ and metrics are reproducible.

Example:
    RANDOMDRAW_COLOR_SELECT=farthest \
    python evaluation/semseg/inference_randomdraw_separated.py \
        --prompt_style randomdraw --steps 28 --guidance_scale 3.5 \
        --model <ckpt> --dataset cityscapes --split val \
        --cityscapes_root <...> \
        --out_dir <...>/semseg_cityscapes-val_steps28_randomdraw_farthest
"""
from __future__ import annotations

import importlib.util
import os
import random
import sys
from pathlib import Path

import numpy as np

# Load the production evaluator as a module (by path, cwd-independent). It must be
# registered in sys.modules before exec so its @dataclass definitions resolve.
_BASE_PATH = Path(__file__).resolve().with_name("inference_partial_flux2_klein_img2img.py")
_spec = importlib.util.spec_from_file_location("ovb_semseg_base", _BASE_PATH)
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)

diversify = base.diversify
NUM_CLASSES = len(base.CITYSCAPES_CLASSES)

_SELECT = os.environ.get("RANDOMDRAW_COLOR_SELECT", "farthest").lower()
_MIN_DELTA_E = float(os.environ.get("RANDOMDRAW_MIN_DELTA_E", "35"))
# Palette draw for the greedy branch (diversify.COLOR_MAPPING_MODES); match the
# checkpoint's training color_mapping. The farthest branch spans the full palette.
_COLOR_MAPPING = os.environ.get("RANDOMDRAW_COLOR_MAPPING", "p200_randomnegdraw")


def _farthest_point_colors(rng: random.Random, k: int) -> list[tuple[int, int, int]]:
    """Max-min selection of k palette colors: random start, then repeatedly add
    the palette color whose nearest already-chosen color (CIELAB) is farthest."""
    lab = diversify.FULL_LAB.astype(np.float64)
    rgb_pal = diversify.FULL_RGB
    start = rng.randrange(len(lab))
    chosen = [start]
    min_d = ((lab - lab[start]) ** 2).sum(1)
    for _ in range(k - 1):
        nxt = int(np.argmax(min_d))
        chosen.append(nxt)
        min_d = np.minimum(min_d, ((lab - lab[nxt]) ** 2).sum(1))
    return [tuple(int(v) for v in rgb_pal[i]) for i in chosen]


def sample_randomdraw_colors(rng: random.Random) -> list[tuple[int, int, int]]:
    """Drop-in replacement for base.sample_randomdraw_colors with the chosen
    separation strategy. Same (rng) signature so base.main() calls it unchanged."""
    if _SELECT == "farthest":
        return _farthest_point_colors(rng, NUM_CLASSES)
    sampled = diversify.sample_colors(rng, NUM_CLASSES, style="rgb", min_delta_e=_MIN_DELTA_E,
                                      color_mapping=_COLOR_MAPPING)
    return [rgb for rgb, _ in sampled]


# Swap only the sampler; everything else (prompt build, decode, metric) is base's.
base.sample_randomdraw_colors = sample_randomdraw_colors


if __name__ == "__main__":
    print(f"[separated-eval] color_select={_SELECT}"
          + (f" min_delta_e={_MIN_DELTA_E} color_mapping={_COLOR_MAPPING}" if _SELECT == "greedy" else ""))
    base.main()
