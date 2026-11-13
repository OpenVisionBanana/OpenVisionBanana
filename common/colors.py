#!/usr/bin/env python3
"""Canonical RGB spelling for every prompt, in every task.

Semantic seg, referring seg and instance seg all name colors in their prompts. The
notation STYLE still varies (a semseg prompt may use everyday names or hex instead
of a triple), but whenever a raw RGB triple is written it must look identical
across tasks and across train/eval:

    [52,28,2]     square brackets, comma-separated, no spaces

Shared by the dataset-prep converters and the evaluators. Black background pixels
are spelled via BLACK_TOKEN.
"""
from __future__ import annotations

from typing import Sequence


def format_rgb(rgb: Sequence[int]) -> str:
    """Render one RGB triple in the canonical prompt notation: ``[52,28,2]``."""
    r, g, b = (int(v) for v in rgb)
    return f"[{r},{g},{b}]"


BLACK_TOKEN = format_rgb((0, 0, 0))  # "[0,0,0]" -- the reserved background color


if __name__ == "__main__":
    assert format_rgb((52, 28, 2)) == "[52,28,2]"
    assert format_rgb([255, 0, 0]) == "[255,0,0]"
    assert BLACK_TOKEN == "[0,0,0]"
    print("ok:", format_rgb((52, 28, 2)), BLACK_TOKEN)
