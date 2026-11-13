# ===============================================================================
# Flux2-Klein partial-transformer semantic-segmentation inference and mIoU evaluation.
# ===============================================================================

"""
1. Builds (image,ground-truth) pairs for Cityscapes, ACDC, Dark Zurich, or BDD.
2. Sends each RGB image through a Flux2-Klein image-to-image pipeline using the
   Cityscapes segmentation prompt below.
3. Decodes the raw RGB diffusion output into Cityscapes trainIds by assigning
   every pixel to the nearest color in the 19-class Cityscapes palette. This
   happens in rgb_to_trainids(): it computes squared RGB distance from each
   output pixel to every palette color, picks the closest class, and can mark
   pixels as ignore_index=255 when --max_palette_dist is used. The snapped
   trainId mask is then rendered back to exact Cityscapes colors by
   trainids_to_rgb().
4. Saves three prediction formats under --out_dir:
   - pred_rgb_raw/: the raw RGB image returned by Flux2-Klein.
   - pred_trainids/: the raw output snapped to Cityscapes trainIds.
   - pred_rgb_snapped/: the snapped trainIds rendered back into the palette.
5. Computes mmseg mIoU, mDice, and mFscore against the dataset GT masks.

Typical usage:
    CUDA_VISIBLE_DEVICES=0 python3 evaluation/semseg/inference_partial_flux2_klein_img2img.py \
        --dataset acdc \
        --data_root /path/to/ACDC \
        --condition night \
        --transformer_path /path/to/open-visionbanana-partial-ft \
        --out_dir outputs/acdc_night

Useful options:
    --dry_run
        Only resolve image/GT pairs and print the first few matches. Use this
        before loading the model when testing a new dataset path.
    --image_root /path/to/edited/images
        Evaluate edited RGB variants while still resolving GT masks from
        --data_root or --gt_root.
    --gt_root /path/to/gt
        Override the dataset's normal GT directory.
    --image_suffixes .png,.jpg
        Override allowed input suffixes. This is also used to strip edited
        image suffixes when matching GT masks.
    --image_glob '**/*_edited.png'
        Restrict which images are evaluated under the image root(s).
    --native_resolution
        Ask Flux2-Klein to generate at the input image resolution.

Dataset layout notes:
    The dataset-specific path rules live in DatasetLayout entries below. When
    --image_root is omitted, images are found from --data_root using the chosen
    layout. When --image_root is provided, only images come from there; GT masks
    are still taken from --data_root unless --gt_root is set.

"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
# diffusers is imported lazily inside main() so the lightweight helpers (palette
# rendering, panels, dataset pairing) can be imported without the pipeline.

# Shared semseg vocabulary (palette, fixed prompt, decoder) -- single source of
# truth with dataset prep, so eval can't drift from training.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.semseg import diversify  # noqa: E402

# Cityscapes vocabulary comes from the shared module.
PROMPT = diversify.FIXED_CITYSCAPES_PROMPT          # eval "fixed" style
CITYSCAPES_CLASSES = diversify._CS_CLASSES
PALETTE = np.asarray(diversify._CS_PALETTE, dtype=np.uint8)
BACKGROUND_COLOR = diversify.BACKGROUND_COLOR

COMMON_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
ACDC_CONDITIONS = ("fog", "night", "rain", "snow")


# How randomdraw legend colors are drawn (one of diversify.COLOR_MAPPING_MODES);
# set from --color_mapping in main() so it matches the checkpoint's training pool.
COLOR_MAPPING = "p200_randomnegdraw"


def sample_randomdraw_colors(rng: random.Random, num_classes: int = len(CITYSCAPES_CLASSES)) -> list[tuple[int, int, int]]:
    """One color per class for the "randomdraw" style: ``num_classes`` distinct,
    well-separated colors drawn from the shared palette under COLOR_MAPPING (the
    exact draw training used), so the model is probed on following per-image
    colors it has actually seen."""
    sampled = diversify.sample_colors(rng, num_classes, style="rgb", color_mapping=COLOR_MAPPING)
    return [rgb for rgb, _ in sampled]


def sample_randomdraw_colors_with_names(
    rng: random.Random,
    num_classes: int = len(CITYSCAPES_CLASSES),
) -> tuple[list[tuple[int, int, int]], list[str]]:
    """Return the same RGB draw as sample_randomdraw_colors, plus its names."""
    sampled = diversify.sample_colors(
        rng, num_classes, style="rgb", color_mapping=COLOR_MAPPING
    )
    return [rgb for rgb, _ in sampled], [name for _, name in sampled]


# --- COCO super-category taxonomies ------------------------------------------
# Classes + their fixed ("fixed" style) colors are read from the CSVs next to this
# file (coco_{stuff,thing}_label_color_mapping.csv), in label_id order -- which is
# the alphabetical super-category order that load_coco_category_map produces, so
# CSV row i == remapped trainId i.
def load_coco_label_csv(name: str) -> tuple[tuple[str, ...], np.ndarray]:
    import csv

    path = Path(__file__).resolve().parent / name
    rows: list[tuple[int, str, tuple[int, int, int]]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                (int(row["label_id"]), row["class_name"], (int(row["r"]), int(row["g"]), int(row["b"])))
            )
    rows.sort(key=lambda item: item[0])
    classes = tuple(name for _, name, _ in rows)
    palette = np.array([rgb for _, _, rgb in rows], dtype=np.uint8)
    return classes, palette


COCO_THING_CLASSES, COCO_THING_PALETTE = load_coco_label_csv("coco_thing_label_color_mapping.csv")
COCO_STUFF_CLASSES, COCO_STUFF_PALETTE = load_coco_label_csv("coco_stuff_label_color_mapping.csv")


# Canonical COCO-Object semantic-segmentation taxonomy used by GroupViT and
# follow-up open-vocabulary segmentation work: the 80 COCO thing classes in
# category-id order, plus an explicit background class at trainId 0. The source
# stuffthingmaps use sparse COCO category ids minus one; load_coco_category_map()
# below converts those ids to this contiguous 1..80 foreground range.
COCO_OBJECT80_FOREGROUND_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)
COCO_OBJECT80_CLASSES = ("background",) + COCO_OBJECT80_FOREGROUND_CLASSES
COCO_OBJECT80_PALETTE = np.vstack(
    [np.asarray([[0, 0, 0]], dtype=np.uint8), diversify.FULL_RGB[:80].astype(np.uint8)]
)

def make_selective_prompt(classes: Sequence[str], colors: Sequence[Sequence[int]]) -> str:
    """A SELECTIVE semseg prompt (used for COCO): paint only the listed classes
    with their colors; everything else -> black. Unlike the Cityscapes "fixed"
    prompt (full coverage), this fits COCO images that contain lots of content
    outside the chosen super-categories."""
    color_map = ", ".join(
        f"'{name}': {list(colors[idx])}" for idx, name in enumerate(classes)
    )
    return (
        "[TASK: SEMANTIC_SEGMENTATION] Generate a semantic segmentation visualization of the input image, "
        f"using this color mapping for the listed classes: {{ {color_map} }}. "
        "Paint pixels belonging to a listed class with exactly its RGB color. "
        "Pixels that do not belong to one of these listed classes must be black [0, 0, 0] "
        "and must not be forced into a listed class. Output only the segmentation visualization image."
    )

def rgb_to_trainids(
    rgb: Image.Image,
    max_palette_dist: Optional[int] = None,
) -> np.ndarray:
    """
    Decode an RGB segmentation visualization into Cityscapes trainIds.

    Diffusion outputs are continuous/anti-aliased, so exact color matching is
    brittle. Each RGB pixel is mapped to the nearest Cityscapes palette color.
    """
    arr = np.array(rgb.convert("RGB"), dtype=np.int32)
    h, w, _ = arr.shape

    flat = arr.reshape(-1, 3)
    palette = PALETTE.astype(np.int32)

    dist = ((flat[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    nearest = dist.argmin(axis=1).astype(np.uint8)

    if max_palette_dist is not None:
        min_dist = dist.min(axis=1)
        nearest[min_dist > max_palette_dist] = 255

    return nearest.reshape(h, w)


# Decode against an explicit per-trainId legend (black -> ignore). Shared with
# dataset prep so the snapping rule is identical at train and eval time.
rgb_to_trainids_via_legend = diversify.rgb_to_trainids_via_legend


def trainids_to_rgb(
    trainids: np.ndarray,
    palette: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render a trainId mask to RGB. Defaults to the fixed Cityscapes palette,
    but ``palette`` can override it (e.g. the per-image randomdraw legend) so the
    visualization reflects the colors the model was actually asked to produce."""
    pal = PALETTE if palette is None else np.asarray(palette, dtype=np.uint8)
    vis = np.zeros((*trainids.shape, 3), dtype=np.uint8)

    valid = (trainids >= 0) & (trainids < len(pal))
    vis[valid] = pal[trainids[valid]]
    vis[~valid] = np.array([0, 0, 0], dtype=np.uint8)

    return vis


PANEL_LABELS = ("input", "ground truth", "raw prediction", "snapped prediction")


def render_visual_color_legend(
    classes: Sequence[str],
    colors: Sequence[tuple[int, int, int]],
    color_names: Optional[Sequence[str]],
    mode: str,
    size: int = 1024,
) -> Image.Image:
    """Render the second image used by the visual-legend B/C/D ablation."""
    from PIL import ImageDraw, ImageFont

    if mode not in {"swatches", "class_labels", "full_labels"}:
        raise ValueError(f"Unsupported visual legend mode: {mode}")
    if len(classes) != len(colors):
        raise ValueError("Visual legend needs exactly one color per class.")

    columns = 4
    rows = (len(classes) + columns - 1) // columns
    margin, gap = 28, 14
    cell_w = (size - 2 * margin - (columns - 1) * gap) // columns
    cell_h = (size - 2 * margin - (rows - 1) * gap) // rows
    label_h = 0 if mode == "swatches" else (42 if mode == "class_labels" else 64)
    canvas = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        font = ImageFont.truetype(font_path, 22)
        small_font = ImageFont.truetype(font_path, 15)
    except OSError:
        font = small_font = ImageFont.load_default()

    for index, (class_name, color) in enumerate(zip(classes, colors)):
        row, column = divmod(index, columns)
        x0 = margin + column * (cell_w + gap)
        y0 = margin + row * (cell_h + gap)
        x1, y1 = x0 + cell_w, y0 + cell_h
        swatch_y1 = y1 - label_h
        draw.rectangle(
            (x0, y0, x1, swatch_y1),
            fill=tuple(color),
            outline=(30, 30, 30),
            width=2,
        )
        if mode == "class_labels":
            draw.text((x0 + 4, swatch_y1 + 8), class_name, fill=(10, 10, 10), font=font)
        elif mode == "full_labels":
            color_name = color_names[index] if color_names else "unnamed"
            draw.text((x0 + 4, swatch_y1 + 4), class_name, fill=(10, 10, 10), font=font)
            draw.text(
                (x0 + 4, swatch_y1 + 34),
                f"{color_name}  RGB{tuple(color)}",
                fill=(10, 10, 10),
                font=small_font,
            )
    return canvas


def add_visual_legend_instruction(prompt: str, mode: str) -> str:
    label_description = {
        "swatches": "unlabelled swatches in the same row-major order as the class mapping below",
        "class_labels": "swatches labelled with semantic class names",
        "full_labels": "swatches labelled with class name, color name, and RGB value",
    }[mode]
    return (
        "You receive two input images. Image 1 is the street scene to segment. "
        f"Image 2 is the authoritative color legend ({label_description}). "
        "Use the exact solid color shown in Image 2 for every class; do not copy the "
        "legend layout, grid, swatches, or text into the result. Return only the semantic "
        "segmentation mask for Image 1.\n\n"
        + prompt
    )

# 255 is the ignore label: it means "this pixel snapped to nothing" (black void,
# or beyond max_palette_dist), not "class 255".
IGNORE_LABEL = 255

# Diverging vote-stability palette used by 05_confidence.png. Keep the
# orientation explicit: unstable pixels are red and stable pixels are blue.
# The midpoint is deliberately off-white rather than pure white.
VOTE_AGREEMENT_LOW_RGB = np.asarray((178, 24, 43), dtype=np.float32)  # #B2182B
VOTE_AGREEMENT_MID_RGB = np.asarray((247, 247, 247), dtype=np.float32)  # #F7F7F7
VOTE_AGREEMENT_HIGH_RGB = np.asarray((33, 102, 172), dtype=np.float32)  # #2166AC


def cumulative_majority_vote_trainids(
    masks: Sequence[np.ndarray],
    num_classes: int,
) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    """Yield the class-space majority vote and agreement for every prefix.

    The yielded integer is the number of random color assignments included in
    that prefix. Agreement is the winning class's share among non-abstaining
    votes; it is a vote-stability diagnostic, not a calibrated correctness
    probability. Ties retain the evaluator's deterministic lowest-class-id
    rule.
    """
    if not masks:
        raise ValueError("cumulative_majority_vote_trainids needs at least one mask")
    shape = masks[0].shape
    for mask in masks:
        if mask.shape != shape:
            raise ValueError(f"vote over mismatched shapes: {shape} vs {mask.shape}")

    counts = np.zeros((num_classes, *shape), dtype=np.uint16)
    voted_total = np.zeros(shape, dtype=np.uint16)
    for repeat_count, mask in enumerate(masks, start=1):
        valid = mask < num_classes
        voted_total[valid] += 1
        for class_id in np.unique(mask[valid]):
            counts[class_id][valid & (mask == class_id)] += 1

        winner = counts.argmax(axis=0).astype(np.uint8)
        top = counts.max(axis=0)
        voted = np.where(top > 0, winner, IGNORE_LABEL).astype(np.uint8)
        agreement = np.zeros(shape, dtype=np.float32)
        np.divide(top, voted_total, out=agreement, where=voted_total > 0)
        yield repeat_count, voted, agreement


def majority_vote_trainids(
    masks: Sequence[np.ndarray],
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel majority vote over repeated randomdraw predictions.

    Each mask is the same image decoded against a DIFFERENT random legend, so the
    masks are directly comparable in label space even though their RGB outputs
    are not. Voting there rather than in color space is what makes this valid.

    ``IGNORE_LABEL`` votes are abstentions, not a class: a pixel that snapped to
    nothing in one repeat should not outvote the repeats that did decode it. A
    pixel is only ignored in the result if EVERY repeat abstained. Ties go to the
    lowest class id, which is arbitrary but deterministic -- and with an odd
    number of repeats exact ties are rare.

    Returns (voted mask, agreement) where agreement is the winning class's vote
    share among non-abstaining repeats, 0 where every repeat abstained.
    """
    final = None
    for _repeat_count, voted, agreement in cumulative_majority_vote_trainids(
        masks, num_classes
    ):
        final = voted, agreement
    if final is None:  # guarded by cumulative_majority_vote_trainids; helps typing
        raise ValueError("majority_vote_trainids needs at least one mask")
    return final


def vote_agreement_to_rgb(agreement: np.ndarray) -> np.ndarray:
    """Map 0..1 vote agreement to red -> off-white -> blue RGB."""
    values = np.nan_to_num(
        np.asarray(agreement, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0
    )
    values = np.clip(values, 0.0, 1.0)
    rgb = np.empty((*values.shape, 3), dtype=np.float32)
    low_half = values <= 0.5
    low_mix = (values[low_half] * 2.0)[:, None]
    high_mix = ((values[~low_half] - 0.5) * 2.0)[:, None]
    rgb[low_half] = (
        VOTE_AGREEMENT_LOW_RGB
        + low_mix * (VOTE_AGREEMENT_MID_RGB - VOTE_AGREEMENT_LOW_RGB)
    )
    rgb[~low_half] = (
        VOTE_AGREEMENT_MID_RGB
        + high_mix * (VOTE_AGREEMENT_HIGH_RGB - VOTE_AGREEMENT_MID_RGB)
    )
    return np.floor(rgb + 0.5).astype(np.uint8)


def render_vote_agreement_visualization(
    agreement: np.ndarray,
    num_assignments: int,
) -> Image.Image:
    """Colorize agreement and append a labeled 0--100% horizontal colorbar."""
    from PIL import ImageDraw, ImageFont

    if num_assignments < 2:
        raise ValueError(
            "Vote-agreement visualizations require at least two assignments."
        )
    colorized = Image.fromarray(vote_agreement_to_rgb(agreement))
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font_size = max(12, min(24, colorized.width // 70))
    try:
        font = ImageFont.truetype(font_path, font_size)
        small_font = ImageFont.truetype(font_path, max(10, font_size - 2))
    except OSError:
        font = small_font = ImageFont.load_default()

    label = f"Vote agreement across {num_assignments} random color assignments."
    note = "Vote stability, not calibrated correctness probability."
    measuring_canvas = Image.new("RGB", (1, 1), "white")
    measuring_draw = ImageDraw.Draw(measuring_canvas)
    label_width = measuring_draw.textbbox((0, 0), label, font=font)[2]
    note_width = measuring_draw.textbbox((0, 0), note, font=small_font)[2]
    canvas_width = max(colorized.width, label_width + 40, note_width + 40)
    side_margin = max(24, canvas_width // 12)
    bar_width = canvas_width - 2 * side_margin
    bar_height = max(12, font_size // 2)
    label_y = colorized.height + 12
    bar_y = label_y + font_size + 10
    tick_label_y = bar_y + bar_height + 7
    note_y = tick_label_y + font_size + 8
    canvas_height = note_y + font_size + 14

    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    canvas.paste(colorized, ((canvas_width - colorized.width) // 2, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (canvas_width // 2, label_y),
        label,
        fill=(25, 25, 25),
        font=font,
        anchor="ma",
    )

    gradient_values = np.linspace(0.0, 1.0, bar_width, dtype=np.float32)[None, :]
    gradient_rgb = vote_agreement_to_rgb(gradient_values)
    gradient = Image.fromarray(np.repeat(gradient_rgb, bar_height, axis=0))
    canvas.paste(gradient, (side_margin, bar_y))
    draw.rectangle(
        (side_margin - 1, bar_y - 1, side_margin + bar_width, bar_y + bar_height),
        outline=(60, 60, 60),
        width=1,
    )
    for fraction, tick_label in zip(
        (0.0, 0.25, 0.5, 0.75, 1.0),
        ("0%", "25%", "50%", "75%", "100%"),
    ):
        x = side_margin + round(fraction * (bar_width - 1))
        draw.line(
            (x, bar_y + bar_height, x, bar_y + bar_height + 4),
            fill=(40, 40, 40),
        )
        anchor = "la" if fraction == 0.0 else ("ra" if fraction == 1.0 else "ma")
        draw.text(
            (x, tick_label_y),
            tick_label,
            fill=(25, 25, 25),
            font=small_font,
            anchor=anchor,
        )
    draw.text(
        (canvas_width // 2, note_y),
        note,
        fill=(70, 70, 70),
        font=small_font,
        anchor="ma",
    )
    return canvas


def _wrap_text(draw, text: str, font, max_width: int) -> list[str]:
    """Greedily wrap ``text`` so each line fits within ``max_width`` pixels."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split(" ")
        current = ""
        for word in words:
            trial = word if not current else f"{current} {word}"
            if draw.textlength(trial, font=font) <= max_width or not current:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def make_panel(
    cells: Sequence[tuple[Image.Image, int]],
    labels: Optional[Sequence[str]] = None,
    header: Optional[str] = None,
    height: int = 512,
    pad: int = 6,
    bg: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Horizontally tile images (each scaled to a common height) into one panel.

    ``cells`` is a list of (image, PIL resample) pairs so photos can use a smooth
    filter while discrete masks stay crisp under nearest-neighbor. Optional
    ``labels`` are drawn at the top-left of each cell. Optional ``header`` text
    (e.g. the prompt) is drawn as black text on a white strip above the row,
    wrapped to the panel width.
    """
    from PIL import ImageDraw, ImageFont

    resized = []
    for im, resample in cells:
        w, h = im.size
        new_w = max(1, round(w * height / h))
        resized.append(im.convert("RGB").resize((new_w, height), resample))

    total_w = sum(im.width for im in resized) + pad * (len(resized) + 1)
    row = Image.new("RGB", (total_w, height + 2 * pad), bg)

    draw = ImageDraw.Draw(row)
    font = ImageFont.load_default()

    x = pad
    for idx, im in enumerate(resized):
        row.paste(im, (x, pad))
        if labels and idx < len(labels):
            text = labels[idx]
            tw = draw.textlength(text, font=font) if hasattr(draw, "textlength") else len(text) * 6
            draw.rectangle([x, pad, x + tw + 6, pad + 14], fill=(0, 0, 0))
            draw.text((x + 3, pad + 2), text, fill=(255, 255, 255), font=font)
        x += im.width + pad

    if not header:
        return row

    # White header strip with the wrapped prompt text above the tiled row.
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    line_h = (bbox[3] - bbox[1]) + 4
    lines = _wrap_text(draw, header, font, total_w - 2 * pad)
    header_h = pad + line_h * len(lines) + pad

    panel = Image.new("RGB", (total_w, header_h + row.height), (255, 255, 255))
    hdr_draw = ImageDraw.Draw(panel)
    y = pad
    for line in lines:
        hdr_draw.text((pad, y), line, fill=(0, 0, 0), font=font)
        y += line_h
    panel.paste(row, (0, header_h))
    return panel


def load_trainid_mask(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path)
    arr = np.array(mask)
    if arr.ndim == 3:
        return rgb_to_trainids(mask, max_palette_dist=0)
    return arr.astype(np.uint8)


def strip_known_suffix(filename: str, suffixes: Sequence[str]) -> str:
    suffixes_by_length = sorted(suffixes, key=len, reverse=True)
    for suffix in suffixes_by_length:
        if filename.lower().endswith(suffix.lower()):
            return filename[:-len(suffix)]
    return Path(filename).stem


def merge_suffixes(*suffix_groups: Sequence[str]) -> tuple[str, ...]:
    merged = []
    seen = set()
    for suffixes in suffix_groups:
        for suffix in suffixes:
            lowered = suffix.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            merged.append(suffix)
    return tuple(merged)


def safe_output_stem(rel_path: Path, image_suffixes: Sequence[str]) -> str:
    stem = strip_known_suffix(rel_path.name, image_suffixes)
    parent_parts = list(rel_path.parent.parts)
    if parent_parts and stem.startswith(f"{parent_parts[-1]}_"):
        parent_parts = parent_parts[:-1]
    parts = [*parent_parts, stem]
    return "__".join(part for part in parts if part not in ("", "."))


def unique_existing_dirs(paths: Iterable[Path]) -> list[Path]:
    seen = set()
    unique = []
    for path in paths:
        path = path.expanduser()
        key = path.resolve() if path.exists() else path.absolute()
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            unique.append(path)
    return unique


@dataclass(frozen=True)
class DatasetSample:
    image_path: Path
    gt_path: Path
    rel_path: Path
    output_stem: str


@dataclass(frozen=True)
class SemanticSpec:
    """Per-dataset taxonomy: class names, the fixed palette, and how to build the
    prompt for the fixed/randomdraw styles. ``coco_mask_type``
    (thing|stuff|object80) routes GT decoding through the corresponding
    COCO remap instead of a trainId PNG. ``implicit_background_id`` identifies a
    class that grouped evaluation must not prompt and assigns to pixels not
    claimed by any foreground group."""
    classes: tuple[str, ...]
    palette: np.ndarray
    fixed_prompt: str
    prompt_builder: Callable[[Sequence[Sequence[int]]], str]
    selective: bool = False
    coco_mask_type: Optional[str] = None
    implicit_background_id: Optional[int] = None


@dataclass(frozen=True)
class DatasetLayout:
    name: str
    semantic: SemanticSpec
    image_suffixes: tuple[str, ...]
    gt_suffix: str
    image_dirs: Callable[[Path, str, Optional[str]], list[Path]]
    gt_dirs: Callable[[Path, str, Optional[str]], list[Path]]
    inferred_subdir: Optional[Callable[[str], Optional[str]]] = None

    def stem_from_image(
        self,
        image_path: Path,
        image_suffixes: Sequence[str],
    ) -> str:
        return strip_known_suffix(image_path.name, image_suffixes)

    def gt_name_from_image(
        self,
        image_path: Path,
        image_suffixes: Sequence[str],
    ) -> str:
        return f"{self.stem_from_image(image_path, image_suffixes)}{self.gt_suffix}"

    def direct_gt_candidates(
        self,
        image_path: Path,
        rel_path: Path,
        gt_dirs: Sequence[Path],
        image_suffixes: Sequence[str],
    ) -> list[Path]:
        image_stem = self.stem_from_image(image_path, image_suffixes)
        gt_name = f"{image_stem}{self.gt_suffix}"
        candidates = []

        for gt_dir in gt_dirs:
            candidates.append(gt_dir / rel_path.parent / gt_name)
            candidates.append(gt_dir / gt_name)

            if self.inferred_subdir is not None:
                subdir = self.inferred_subdir(image_stem)
                if subdir is not None:
                    candidates.append(gt_dir / subdir / gt_name)

        return candidates


def cityscapes_image_dirs(root: Path, split: str, condition: Optional[str]) -> list[Path]:
    del condition
    return [root / "leftImg8bit" / split]


def cityscapes_gt_dirs(root: Path, split: str, condition: Optional[str]) -> list[Path]:
    del condition
    return [root / "gtFine" / split]


def city_from_cityscapes_stem(stem: str) -> Optional[str]:
    if "_" not in stem:
        return None
    return stem.split("_", 1)[0]


def acdc_image_dirs(root: Path, split: str, condition: Optional[str]) -> list[Path]:
    if condition:
        return [root / "rgb_anon" / condition / split]

    flat_dir = root / "rgb_anon" / split
    if flat_dir.exists():
        return [flat_dir]

    return [root / "rgb_anon" / cond / split for cond in ACDC_CONDITIONS]


def acdc_gt_dirs(root: Path, split: str, condition: Optional[str]) -> list[Path]:
    if condition:
        return [root / "gt" / condition / split]

    flat_dir = root / "gt" / split
    if flat_dir.exists():
        return [flat_dir]

    return [root / "gt" / cond / split for cond in ACDC_CONDITIONS]


def darkzurich_image_dirs(root: Path, split: str, condition: Optional[str]) -> list[Path]:
    del condition
    return [root / "rgb_anon" / split]


def darkzurich_gt_dirs(root: Path, split: str, condition: Optional[str]) -> list[Path]:
    del condition
    return [root / "gt" / split]


def bdd_image_dirs(root: Path, split: str, condition: Optional[str]) -> list[Path]:
    del condition
    return [
        root / "images" / "10k" / split,
        root / "images" / "100k" / split,
        root / "images" / split,
    ]


def bdd_gt_dirs(root: Path, split: str, condition: Optional[str]) -> list[Path]:
    del condition
    return [
        root / "labels" / "sem_seg" / "masks" / split,
        root / "labels" / "sem_seg" / split,
        root / "labels" / split,
    ]


# --- COCO (super-category) directory resolvers -------------------------------
# DATA_ROOT holds the staged `dataset/` tree (coco.tar), so the `dataset/` and
# `coco/dataset/` fallbacks cover both extracted layouts.
def coco_image_dirs(root: Path, split: str, condition: Optional[str]) -> list[Path]:
    del condition
    return [
        root / "images" / f"{split}2017",
        root / "dataset" / "images" / f"{split}2017",
        root / "coco" / "dataset" / "images" / f"{split}2017",
    ]


def coco_thing_gt_dirs(root: Path, split: str, condition: Optional[str]) -> list[Path]:
    del condition
    return [
        root / "annotations" / f"{split}2017",
        root / "dataset" / "annotations" / f"{split}2017",
        root / "coco" / "dataset" / "annotations" / f"{split}2017",
    ]


def coco_stuff_gt_dirs(root: Path, split: str, condition: Optional[str]) -> list[Path]:
    del condition
    rel = Path("annotations") / "stuff_annotations" / f"stuff_{split}2017_pixelmaps"
    return [root / rel, root / "dataset" / rel, root / "coco" / "dataset" / rel]


def coco_annotation_json(root: Path, mask_type: str, split: str) -> Path:
    if mask_type in ("thing", "object80"):
        rel = Path("annotations") / "panoptic_annotations" / f"panoptic_{split}2017.json"
    elif mask_type == "stuff":
        rel = Path("annotations") / "stuff_annotations" / f"stuff_{split}2017.json"
    else:
        raise ValueError(f"Unsupported COCO mask type: {mask_type}")
    for candidate in (root / rel, root / "dataset" / rel, root / "coco" / "dataset" / rel):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find COCO {mask_type} category JSON under {root} ({rel})")


def load_coco_category_map(root: Path, mask_type: str, split: str) -> dict[int, int]:
    """Map raw COCO category id -> super-category trainId (255 = ignore). The
    super-category index order is sorted() alphabetical, which matches the CSV
    row order used for COCO_*_CLASSES / COCO_*_PALETTE."""
    with open(coco_annotation_json(root, mask_type, split), encoding="utf-8") as handle:
        all_categories = json.load(handle)["categories"]

    if mask_type == "object80":
        things = [category for category in all_categories if category.get("isthing", 0) == 1]
        things.sort(key=lambda category: int(category["id"]))
        names = tuple(str(category["name"]) for category in things)
        if names != COCO_OBJECT80_FOREGROUND_CLASSES:
            raise ValueError(
                "COCO panoptic category JSON does not contain the canonical 80 thing "
                "classes in category-id order."
            )
        return {int(category["id"]): train_id for train_id, category in enumerate(things, 1)}

    if mask_type == "thing":
        thing_supercats = sorted({c["supercategory"] for c in all_categories if c.get("isthing", 0) == 1})
        supercat_to_id = {name: idx for idx, name in enumerate(thing_supercats)}
        return {
            int(c["id"]): (supercat_to_id[c["supercategory"]] if c.get("isthing", 0) == 1 else 255)
            for c in all_categories
        }

    supercats = sorted({c["supercategory"] for c in all_categories if c["supercategory"] != "other"})
    supercat_to_id = {name: idx for idx, name in enumerate(supercats)}
    supercat_to_id["other"] = 255
    return {int(c["id"]): supercat_to_id.get(c["supercategory"], 255) for c in all_categories}


def remap_coco_mask(mask_path: Path, mask_type: str, category_map: dict[int, int]) -> np.ndarray:
    """Decode a COCO GT PNG into super-category trainIds (255 = ignore)."""
    arr = np.array(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    mask = arr.astype(np.int64)

    if mask_type == "stuff":
        mask[mask == 0] = 183  # pixelmaps: 0 = not-stuff -> "other" id (183 -> 255)
    else:
        # thing stuffthingmaps are 0-indexed (label L <-> cocostuff id L+1); 255 = void.
        void = mask == 255
        mask = mask + 1
        mask[void] = 255

    # COCO-Object scores the explicit background class: stuff, void, and any
    # non-thing fine category all map to trainId 0. Other COCO protocols retain
    # the established ignore=255 behavior.
    fill_value = 0 if mask_type == "object80" else 255
    remapped = np.full(mask.shape, fill_value, dtype=np.uint8)
    for cat_id in np.unique(mask):
        remapped[mask == cat_id] = category_map.get(int(cat_id), fill_value)
    return remapped


# --- Per-dataset semantic specs ----------------------------------------------
CITYSCAPES_SEMANTIC = SemanticSpec(
    classes=tuple(CITYSCAPES_CLASSES),
    palette=PALETTE,
    fixed_prompt=PROMPT,
    prompt_builder=diversify.cityscapes_fixed_prompt,
    selective=False,
    coco_mask_type=None,
)

COCO_THING_SEMANTIC = SemanticSpec(
    classes=COCO_THING_CLASSES,
    palette=COCO_THING_PALETTE,
    fixed_prompt=make_selective_prompt(COCO_THING_CLASSES, COCO_THING_PALETTE),
    prompt_builder=lambda colors: make_selective_prompt(COCO_THING_CLASSES, colors),
    selective=True,
    coco_mask_type="thing",
)

COCO_STUFF_SEMANTIC = SemanticSpec(
    classes=COCO_STUFF_CLASSES,
    palette=COCO_STUFF_PALETTE,
    fixed_prompt=make_selective_prompt(COCO_STUFF_CLASSES, COCO_STUFF_PALETTE),
    prompt_builder=lambda colors: make_selective_prompt(COCO_STUFF_CLASSES, colors),
    selective=True,
    coco_mask_type="stuff",
)

COCO_OBJECT80_SEMANTIC = SemanticSpec(
    classes=COCO_OBJECT80_CLASSES,
    palette=COCO_OBJECT80_PALETTE,
    # COCO-Object-80 is intentionally evaluated through grouped_semseg.py. This
    # fixed prompt remains metadata only; grouped prompts cover trainIds 1..80,
    # while trainId 0 is handled as implicit background.
    fixed_prompt=make_selective_prompt(
        COCO_OBJECT80_FOREGROUND_CLASSES, COCO_OBJECT80_PALETTE[1:].tolist()
    ),
    prompt_builder=lambda colors: make_selective_prompt(COCO_OBJECT80_FOREGROUND_CLASSES, colors),
    selective=True,
    coco_mask_type="object80",
    implicit_background_id=0,
)


def load_gt_mask(gt_path: Path, semantic: SemanticSpec, coco_category_map: Optional[dict[int, int]]) -> np.ndarray:
    """Dataset-agnostic GT -> trainId loader: COCO uses the super-category remap,
    everything else loads a trainId PNG (or snaps an RGB GT to the palette)."""
    if semantic.coco_mask_type is not None:
        assert coco_category_map is not None, "COCO category map not loaded"
        return remap_coco_mask(gt_path, semantic.coco_mask_type, coco_category_map)
    return load_trainid_mask(gt_path)


DATASETS = {
    "cityscapes": DatasetLayout(
        name="cityscapes",
        semantic=CITYSCAPES_SEMANTIC,
        image_suffixes=("_leftImg8bit.png", ".png", ".jpg", ".jpeg"),
        gt_suffix="_gtFine_labelTrainIds.png",
        image_dirs=cityscapes_image_dirs,
        gt_dirs=cityscapes_gt_dirs,
        inferred_subdir=city_from_cityscapes_stem,
    ),
    "acdc": DatasetLayout(
        name="acdc",
        semantic=CITYSCAPES_SEMANTIC,
        image_suffixes=("_rgb_anon.png", ".png", ".jpg", ".jpeg"),
        gt_suffix="_gt_labelTrainIds.png",
        image_dirs=acdc_image_dirs,
        gt_dirs=acdc_gt_dirs,
    ),
    "darkzurich": DatasetLayout(
        name="darkzurich",
        semantic=CITYSCAPES_SEMANTIC,
        image_suffixes=("_rgb_anon.png", ".png", ".jpg", ".jpeg"),
        gt_suffix="_gt_labelTrainIds.png",
        image_dirs=darkzurich_image_dirs,
        gt_dirs=darkzurich_gt_dirs,
    ),
    "bdd": DatasetLayout(
        name="bdd",
        semantic=CITYSCAPES_SEMANTIC,
        image_suffixes=(".jpg", ".jpeg", ".png"),
        gt_suffix=".png",
        image_dirs=bdd_image_dirs,
        gt_dirs=bdd_gt_dirs,
    ),
    "coco-thing": DatasetLayout(
        name="coco-thing",
        semantic=COCO_THING_SEMANTIC,
        image_suffixes=(".jpg", ".jpeg", ".png"),
        gt_suffix=".png",
        image_dirs=coco_image_dirs,
        gt_dirs=coco_thing_gt_dirs,
    ),
    "coco-object-80": DatasetLayout(
        name="coco-object-80",
        semantic=COCO_OBJECT80_SEMANTIC,
        image_suffixes=(".jpg", ".jpeg", ".png"),
        gt_suffix=".png",
        image_dirs=coco_image_dirs,
        # Combined stuffthingmaps are remapped to 80 contiguous thing ids plus
        # explicit background.
        gt_dirs=coco_thing_gt_dirs,
    ),
    "coco-stuff": DatasetLayout(
        name="coco-stuff",
        semantic=COCO_STUFF_SEMANTIC,
        image_suffixes=(".jpg", ".jpeg", ".png"),
        gt_suffix=".png",
        image_dirs=coco_image_dirs,
        gt_dirs=coco_stuff_gt_dirs,
    ),
}


def parse_suffixes(value: Optional[str]) -> Optional[tuple[str, ...]]:
    if value is None:
        return None
    suffixes = tuple(s.strip() for s in value.split(",") if s.strip())
    return suffixes or None


def list_image_paths(
    image_roots: Sequence[Path],
    image_suffixes: Sequence[str],
    image_glob: Optional[str],
) -> list[tuple[Path, Path]]:
    items = []
    seen = set()

    for image_root in image_roots:
        patterns = [image_glob] if image_glob else [
            f"**/*{suffix}" for suffix in image_suffixes
        ]

        for pattern in patterns:
            for image_path in sorted(image_root.glob(pattern)):
                if not image_path.is_file():
                    continue
                key = image_path.resolve()
                if key in seen:
                    continue
                seen.add(key)
                items.append((image_path, image_root))

    return sorted(items, key=lambda item: str(item[0]))


def build_gt_index(gt_dirs: Sequence[Path]) -> dict[str, Path]:
    index = {}
    for gt_dir in gt_dirs:
        if not gt_dir.exists():
            continue
        for gt_path in sorted(gt_dir.rglob("*.png")):
            index.setdefault(gt_path.name, gt_path)
    return index


def resolve_gt_path(
    layout: DatasetLayout,
    image_path: Path,
    rel_path: Path,
    gt_dirs: Sequence[Path],
    gt_index: dict[str, Path],
    image_suffixes: Sequence[str],
) -> Path:
    for candidate in layout.direct_gt_candidates(
        image_path,
        rel_path,
        gt_dirs,
        image_suffixes,
    ):
        if candidate.exists():
            return candidate

    gt_name = layout.gt_name_from_image(image_path, image_suffixes)
    if gt_name in gt_index:
        return gt_index[gt_name]

    searched = "\n  ".join(str(path) for path in gt_dirs)
    raise FileNotFoundError(
        f"Could not find GT mask for {image_path.name} ({gt_name}). "
        f"Searched under:\n  {searched}"
    )


def build_samples(
    layout: DatasetLayout,
    data_root: Path,
    split: str,
    condition: Optional[str],
    image_root: Optional[Path],
    gt_root: Optional[Path],
    image_suffixes: Sequence[str],
    image_glob: Optional[str],
    limit: Optional[int],
) -> list[DatasetSample]:
    if image_root is not None:
        image_roots = [image_root.expanduser()]
    else:
        image_roots = unique_existing_dirs(layout.image_dirs(data_root, split, condition))

    if gt_root is not None:
        gt_dirs = [gt_root.expanduser()]
    else:
        gt_dirs = unique_existing_dirs(layout.gt_dirs(data_root, split, condition))

    if not image_roots:
        candidates = "\n  ".join(
            str(path) for path in layout.image_dirs(data_root, split, condition)
        )
        raise FileNotFoundError(f"No image directories found. Tried:\n  {candidates}")

    if not gt_dirs:
        candidates = "\n  ".join(
            str(path) for path in layout.gt_dirs(data_root, split, condition)
        )
        raise FileNotFoundError(f"No GT directories found. Tried:\n  {candidates}")

    gt_index = build_gt_index(gt_dirs)
    image_items = list_image_paths(image_roots, image_suffixes, image_glob)
    if limit is not None:
        image_items = image_items[:limit]

    stem_suffixes = merge_suffixes(layout.image_suffixes, image_suffixes)
    samples = []
    for image_path, source_root in image_items:
        rel_path = image_path.relative_to(source_root)
        gt_path = resolve_gt_path(
            layout,
            image_path,
            rel_path,
            gt_dirs,
            gt_index,
            stem_suffixes,
        )
        samples.append(
            DatasetSample(
                image_path=image_path,
                gt_path=gt_path,
                rel_path=rel_path,
                output_stem=safe_output_stem(rel_path, stem_suffixes),
            )
        )

    if not samples:
        suffix_hint = ", ".join(image_suffixes)
        raise FileNotFoundError(
            f"No images found under {', '.join(str(p) for p in image_roots)} "
            f"with suffixes: {suffix_hint}"
        )

    return samples


@dataclass(frozen=True)
class PipelineBatchRequest:
    """One independent FLUX generation that may share a pipeline call with peers.

    ``generator`` is intentionally one object per request. Diffusers accepts a
    generator list for batched calls; keeping that list preserves the evaluator's
    image-keyed seed contract when batching changes the call grouping/order.
    """

    prompt: str
    image: object
    generator: object | None = None
    height: int | None = None
    width: int | None = None


def run_pipeline_batch(
    pipe,
    requests: Sequence[PipelineBatchRequest],
    *,
    num_inference_steps: int,
    guidance_scale: float,
) -> list[Image.Image]:
    """Run compatible semantic-segmentation requests in one FLUX call.

    The single-request path uses scalar arguments. FLUX.2 Klein normally interprets an
    image list as *multiple references shared by every prompt*, rather than a
    one-image-per-prompt batch. For a real batch, temporarily replace only
    ``prepare_image_latents`` so its image list is encoded as one conditioning
    image per prompt. The rest of the installed pipeline (prompt encoding,
    denoising, scheduler, and decode) remains its native implementation.
    """
    if not requests:
        return []

    first = requests[0]
    if any((request.height, request.width) != (first.height, first.width) for request in requests):
        raise ValueError("A FLUX batch must use one shared output height and width.")
    if len(requests) > 1 and any(isinstance(request.image, (list, tuple)) for request in requests):
        raise ValueError(
            "Batched semantic segmentation currently supports one conditioning image per sample; "
            "visual-legend inputs require batch_size=1."
        )

    kwargs = {
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
    }
    if first.height is not None:
        kwargs.update({"height": first.height, "width": first.width})

    if len(requests) == 1:
        kwargs.update({"prompt": first.prompt, "image": first.image})
        if first.generator is not None:
            kwargs["generator"] = first.generator
        output_images = list(pipe(**kwargs).images)
    else:
        generators = [request.generator for request in requests]
        if any(generator is None for generator in generators) and any(
            generator is not None for generator in generators
        ):
            raise ValueError("A FLUX batch cannot mix seeded and unseeded requests.")

        required_private_api = (
            "prepare_image_latents",
            "_encode_vae_image",
            "_prepare_image_ids",
            "_pack_latents",
        )
        missing_private_api = [name for name in required_private_api if not hasattr(pipe, name)]
        if missing_private_api:
            raise RuntimeError(
                "The installed FLUX.2 Klein pipeline cannot perform one-to-one image batching; "
                "missing private API: " + ", ".join(missing_private_api) + ". Use --batch_size 1."
            )

        kwargs.update(
            {
                "prompt": [request.prompt for request in requests],
                "image": [request.image for request in requests],
            }
        )
        if generators[0] is not None:
            kwargs["generator"] = generators

        had_instance_prepare_image_latents = "prepare_image_latents" in vars(pipe)
        original_prepare_image_latents = pipe.prepare_image_latents

        def prepare_one_image_per_prompt(
            pipeline,
            images,
            batch_size,
            generator,
            device,
            dtype,
        ):
            """Adapt Klein image-reference helper to independent images."""
            if len(images) != batch_size:
                raise ValueError(
                    "One-to-one image batching requires exactly one prepared image per prompt."
                )
            image_shape = images[0].shape
            if any(image.shape != image_shape for image in images[1:]):
                raise ValueError(
                    "One-to-one image batching requires conditioning images with the same shape."
                )

            # Klein conditions on the VAE posterior mode, so the conditioning
            # VAE encode is deterministic. The generator list is still passed
            # to the native diffusion call below for per-sample noise.
            vae_generator = generator[0] if isinstance(generator, list) else generator
            image_batch = torch.cat(
                [image.to(device=device, dtype=dtype) for image in images], dim=0
            )
            image_latents = pipeline._encode_vae_image(
                image=image_batch,
                generator=vae_generator,
            )
            image_latent_ids = pipeline._prepare_image_ids([image_latents[:1]])
            image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1).to(device)
            return pipeline._pack_latents(image_latents), image_latent_ids

        pipe.prepare_image_latents = types.MethodType(prepare_one_image_per_prompt, pipe)
        try:
            output_images = list(pipe(**kwargs).images)
        finally:
            if had_instance_prepare_image_latents:
                pipe.prepare_image_latents = original_prepare_image_latents
            else:
                delattr(pipe, "prepare_image_latents")

    if len(output_images) != len(requests):
        raise RuntimeError(
            f"FLUX returned {len(output_images)} images for a batch of {len(requests)} requests."
        )
    return output_images


@dataclass
class PendingSemsegSample:
    """All state retained for one sample while its repeat generations are queued."""

    sample: DatasetSample
    image: Image.Image
    gt: np.ndarray
    save_name: str
    repeat_trainids: list[Optional[np.ndarray]]
    repeat_legends: list[Optional[list[tuple[int, int, int]]]]
    repeat_color_names: list[Optional[list[str]]]
    repeat_prompts: list[Optional[str]]
    panel_pred_rgb: Optional[Image.Image] = None
    panel_prompt: Optional[str] = None
    panel_visual_legend: Optional[Image.Image] = None


@dataclass(frozen=True)
class PendingSemsegGeneration:
    state: PendingSemsegSample
    repeat: int
    request: PipelineBatchRequest
    legend_colors: Optional[list[tuple[int, int, int]]]
    visual_legend: Optional[Image.Image]


def stable_generation_seed(base_seed: int, output_stem: str, repeat: int = 0) -> int:
    """A shard/order-independent diffusion seed for one image (repeat is optional)."""
    key = f"{base_seed}:{output_stem}:{repeat}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63 - 1)


def repeat_output_filename(output_stem: str, repeat: int) -> str:
    """Filename shared by a raw repeat and its prompt-palette ground truth."""
    if repeat < 0:
        raise ValueError(f"Repeat index must be non-negative, got {repeat}.")
    suffix = "" if repeat == 0 else f"_r{repeat}"
    return f"{output_stem}{suffix}.png"


def normalized_model_reference(value: Optional[str]) -> Optional[str]:
    """Canonicalize local model references while preserving remote model IDs."""
    if value is None:
        return None
    path = Path(value).expanduser()
    return str(path.resolve()) if path.exists() else value


def save_repeat_trainids_cache(path: Path, repeat_trainids: Sequence[np.ndarray], *, metadata=None) -> None:
    """Atomically save all decoded repeats for one image in a compact cache."""
    if not repeat_trainids:
        raise ValueError("Cannot save an empty repeat cache.")
    stacked = np.stack(repeat_trainids).astype(np.uint8, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(temporary, "wb") as handle:
        extra = {} if metadata is None else {"metadata": np.asarray(json.dumps(metadata, sort_keys=True))}
        np.savez_compressed(handle, repeat_trainids=stacked, **extra)
    os.replace(temporary, path)


def load_repeat_trainids_cache(
    path: Path,
    expected_repeats: int,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    """Load and strictly validate one incremental randomdraw cache file."""
    if not path.is_file():
        raise FileNotFoundError(f"Incremental repeat cache not found: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if "repeat_trainids" not in payload:
            raise ValueError(f"Repeat cache lacks 'repeat_trainids': {path}")
        repeat_trainids = np.asarray(payload["repeat_trainids"])
        if repeat_trainids.dtype != np.uint8:
            raise ValueError(f"Repeat cache {path} must contain uint8 train IDs")
    expected = (expected_repeats, *expected_shape)
    if repeat_trainids.shape != expected:
        raise ValueError(
            f"Repeat cache {path} has shape {repeat_trainids.shape}, expected {expected}."
        )
    return repeat_trainids


def validate_incremental_repeat_source(
    source_dir: Path,
    target_dir: Path,
    target_repeats: int,
    expected_config: dict[str, object],
    expected_num_images: int,
) -> tuple[Path, int]:
    """Validate an earlier complete randomdraw run before reusing its repeats."""
    source_dir = source_dir.expanduser().resolve()
    target_dir = target_dir.expanduser().resolve()
    if source_dir == target_dir:
        raise ValueError("--resume_from_out_dir must differ from --out_dir.")
    config_path = source_dir / "eval_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Incremental source lacks eval_config.json: {source_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_repeats = config.get("randomdraw_repeats")
    if not isinstance(source_repeats, int) or source_repeats < 1:
        raise ValueError(
            f"Incremental source records invalid randomdraw_repeats={source_repeats!r}."
        )
    if source_repeats >= target_repeats:
        raise ValueError(
            f"Incremental source has {source_repeats} repeats; target must request more "
            f"than that, got {target_repeats}."
        )
    for field, expected in expected_config.items():
        if field not in config:
            raise ValueError(
                f"Incremental source lacks required provenance field {field!r}."
            )
        if config[field] != expected:
            raise ValueError(
                f"Incremental source {field}={config[field]!r}, expected {expected!r}."
            )
    if config.get("num_images") != expected_num_images:
        raise ValueError(
            f"Incremental source contains {config.get('num_images')!r} images, "
            f"expected {expected_num_images}."
        )
    result_path = source_dir / (
        f"{expected_config['dataset']}_{expected_config['split']}_results.json"
    )
    if not result_path.is_file():
        raise FileNotFoundError(f"Incremental source lacks final results: {result_path}")
    cache_dir = source_dir / "resume_cache"
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"Incremental source lacks resume_cache/: {source_dir}. "
            "Only runs produced after incremental-repeat caching was enabled can "
            "be extended without regenerating their earlier repeats."
        )
    return source_dir, source_repeats


def make_metric():
    try:
        from mmseg.evaluation import IoUMetric

        metric = IoUMetric(
            iou_metrics=["mIoU", "mDice", "mFscore"],
            ignore_index=255,
            nan_to_num=None,
        )
    except ImportError as exc:
        print(f"Falling back to local IoU metric because mmseg import failed: {exc}")
        metric = LocalIoUMetric(ignore_index=255)

    metric.dataset_meta = {
        "classes": CITYSCAPES_CLASSES,
        "palette": PALETTE.tolist(),
    }
    return metric


class LocalIoUMetric:
    def __init__(self, ignore_index: int = 255):
        self.ignore_index = ignore_index
        self.results = []
        self.dataset_meta = {}

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        del data_batch
        num_classes = len(self.dataset_meta["classes"])
        for data_sample in data_samples:
            pred = data_sample["pred_sem_seg"]["data"].squeeze().cpu().numpy()
            label = data_sample["gt_sem_seg"]["data"].squeeze().cpu().numpy()
            self.results.append(
                self.intersect_and_union(
                    pred,
                    label,
                    num_classes=num_classes,
                    ignore_index=self.ignore_index,
                )
            )

    @staticmethod
    def intersect_and_union(
        pred: np.ndarray,
        label: np.ndarray,
        num_classes: int,
        ignore_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        valid = label != ignore_index
        pred = pred[valid]
        label = label[valid]

        class_range = np.arange(num_classes)
        intersect = pred[pred == label]
        area_intersect = np.bincount(
            intersect[(intersect >= 0) & (intersect < num_classes)],
            minlength=num_classes,
        )[class_range]
        area_pred = np.bincount(
            pred[(pred >= 0) & (pred < num_classes)],
            minlength=num_classes,
        )[class_range]
        area_label = np.bincount(
            label[(label >= 0) & (label < num_classes)],
            minlength=num_classes,
        )[class_range]
        area_union = area_pred + area_label - area_intersect
        return area_intersect, area_union, area_pred, area_label

    def compute_metrics(self, results: list) -> dict[str, float]:
        total_intersect = sum(result[0] for result in results).astype(np.float64)
        total_union = sum(result[1] for result in results).astype(np.float64)
        total_pred = sum(result[2] for result in results).astype(np.float64)
        total_label = sum(result[3] for result in results).astype(np.float64)

        if total_label.sum() <= 0:
            raise ValueError("Semantic-segmentation evaluation contains no valid ground-truth pixels.")

        with np.errstate(divide="ignore", invalid="ignore"):
            all_acc = total_intersect.sum() / total_label.sum()
            iou = total_intersect / total_union
            acc = total_intersect / total_label
            dice = 2 * total_intersect / (total_pred + total_label)
            recall = total_intersect / total_label
        evaluated = total_label > 0
        precision = np.full_like(total_intersect, np.nan)
        precision[evaluated] = 0.0
        np.divide(total_intersect, total_pred, out=precision, where=evaluated & (total_pred > 0))
        fscore = np.full_like(total_intersect, np.nan)
        fscore[evaluated] = 0.0
        np.divide(
            2 * precision * recall,
            precision + recall,
            out=fscore,
            where=evaluated & ((precision + recall) > 0),
        )

        return {
            "aAcc": float(np.round(np.nanmean(all_acc) * 100, 2)),
            "mIoU": float(np.round(np.nanmean(iou) * 100, 2)),
            "mAcc": float(np.round(np.nanmean(acc) * 100, 2)),
            "mDice": float(np.round(np.nanmean(dice) * 100, 2)),
            "mFscore": float(np.round(np.nanmean(fscore) * 100, 2)),
            "mPrecision": float(np.round(np.nanmean(precision) * 100, 2)),
            "mRecall": float(np.round(np.nanmean(recall) * 100, 2)),
        }


class SegmentationAccumulator:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.total_intersect = np.zeros(num_classes, dtype=np.float64)
        self.total_union = np.zeros(num_classes, dtype=np.float64)
        self.total_pred = np.zeros(num_classes, dtype=np.float64)
        self.total_label = np.zeros(num_classes, dtype=np.float64)
        self.num_samples = 0

    def update(self, pred: np.ndarray, label: np.ndarray) -> None:
        intersect, union, area_pred, area_label = LocalIoUMetric.intersect_and_union(
            pred,
            label,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )
        self.total_intersect += intersect
        self.total_union += union
        self.total_pred += area_pred
        self.total_label += area_label
        self.num_samples += 1

    def merge(self, other: "SegmentationAccumulator") -> None:
        self.total_intersect += other.total_intersect
        self.total_union += other.total_union
        self.total_pred += other.total_pred
        self.total_label += other.total_label
        self.num_samples += other.num_samples

    def compute(self) -> dict[str, float]:
        if self.total_label.sum() <= 0:
            raise ValueError("Semantic-segmentation evaluation contains no valid ground-truth pixels.")

        with np.errstate(divide="ignore", invalid="ignore"):
            all_acc = self.total_intersect.sum() / self.total_label.sum()
            iou = self.total_intersect / self.total_union
            acc = self.total_intersect / self.total_label
            dice = 2 * self.total_intersect / (self.total_pred + self.total_label)
            recall = self.total_intersect / self.total_label
        evaluated = self.total_label > 0
        precision = np.full_like(self.total_intersect, np.nan)
        precision[evaluated] = 0.0
        np.divide(self.total_intersect, self.total_pred, out=precision, where=evaluated & (self.total_pred > 0))
        fscore = np.full_like(self.total_intersect, np.nan)
        fscore[evaluated] = 0.0
        np.divide(
            2 * precision * recall,
            precision + recall,
            out=fscore,
            where=evaluated & ((precision + recall) > 0),
        )

        return {
            "aAcc": float(np.round(np.nanmean(all_acc) * 100, 2)),
            "mIoU": float(np.round(np.nanmean(iou) * 100, 2)),
            "mAcc": float(np.round(np.nanmean(acc) * 100, 2)),
            "mDice": float(np.round(np.nanmean(dice) * 100, 2)),
            "mFscore": float(np.round(np.nanmean(fscore) * 100, 2)),
            "mPrecision": float(np.round(np.nanmean(precision) * 100, 2)),
            "mRecall": float(np.round(np.nanmean(recall) * 100, 2)),
        }

    def per_class_iou(
        self,
        class_names: Sequence[str],
    ) -> dict[str, Optional[float]]:
        """Return dataset-level class IoUs as percentages keyed by class name.

        A class with no pixels in either predictions or ground truth has zero
        union and therefore no defined IoU. Represent it as ``None`` so result
        files contain standards-compliant JSON ``null`` rather than ``NaN``.
        """
        if len(class_names) != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} class names, got {len(class_names)}."
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = self.total_intersect / self.total_union
        return {
            str(class_name): (
                None if np.isnan(value) else float(np.round(value * 100, 2))
            )
            for class_name, value in zip(class_names, iou)
        }

    def to_dict(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "ignore_index": self.ignore_index,
            "num_samples": self.num_samples,
            "total_intersect": self.total_intersect.tolist(),
            "total_union": self.total_union.tolist(),
            "total_pred": self.total_pred.tolist(),
            "total_label": self.total_label.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SegmentationAccumulator":
        accumulator = cls(
            num_classes=int(data["num_classes"]),
            ignore_index=int(data.get("ignore_index", 255)),
        )
        accumulator.num_samples = int(data.get("num_samples", 0))
        accumulator.total_intersect = np.array(data["total_intersect"], dtype=np.float64)
        accumulator.total_union = np.array(data["total_union"], dtype=np.float64)
        accumulator.total_pred = np.array(data["total_pred"], dtype=np.float64)
        accumulator.total_label = np.array(data["total_label"], dtype=np.float64)
        return accumulator



@dataclass(frozen=True)
class ShardInfo:
    rank: int
    world_size: int
    local_rank: int
    device: str
    distributed: bool


def setup_sharding(args: argparse.Namespace) -> ShardInfo:
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", str(env_rank)))

    if args.num_shards is not None or args.shard_index is not None:
        world_size = args.num_shards if args.num_shards is not None else 1
        rank = args.shard_index if args.shard_index is not None else 0
        distributed = False
    else:
        world_size = env_world_size
        rank = env_rank
        distributed = world_size > 1

    if world_size < 1:
        raise ValueError(f"--num_shards/WORLD_SIZE must be positive, got {world_size}.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Shard/rank index {rank} is out of range for {world_size} shards.")

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        local_rank = args.local_rank if args.local_rank is not None else env_local_rank
        local_rank = int(local_rank) % max(num_devices, 1)
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        local_rank = 0
        device = "cpu"

    if (
        distributed
        and torch.distributed.is_available()
        and not torch.distributed.is_initialized()
    ):
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)

    return ShardInfo(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        distributed=distributed and torch.distributed.is_initialized(),
    )


def barrier(shard: ShardInfo) -> None:
    if shard.distributed:
        torch.distributed.barrier()


def cleanup_distributed(shard: ShardInfo) -> None:
    if shard.distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def shard_samples(samples: Sequence[DatasetSample], shard: ShardInfo) -> list[DatasetSample]:
    return list(samples[shard.rank :: shard.world_size])


def wait_for_shard_metrics(paths: Sequence[Path], timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    missing = [path for path in paths if not path.exists()]
    while missing and time.time() < deadline:
        time.sleep(5)
        missing = [path for path in paths if not path.exists()]

    if missing:
        missing_text = "\n  ".join(str(path) for path in missing)
        raise TimeoutError(f"Timed out waiting for shard metric files:\n  {missing_text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-base-9B")
    parser.add_argument("--model_family", choices=("klein", "dev"), default="klein")
    parser.add_argument("--model_offload", action="store_true")
    parser.add_argument(
        "--offload_mode",
        choices=("none", "model", "group", "sequential"),
        default=None,
        help="CPU offload granularity; group/sequential can run models larger than GPU memory.",
    )
    parser.add_argument(
        "--transformer_path",
        default=None,
        help=(
            "Optional path to the trained output dir, checkpoint dir, or transformer dir. "
            "Both open-visionbanana-partial-ft and open-visionbanana-partial-ft/transformer are accepted."
            " If omitted, the base --model pipeline is loaded without a fine-tuned transformer."
        ),
    )
    parser.add_argument(
        "--lora_path",
        default=None,
        help=(
            "Optional standard Diffusers LoRA artifact directory (holds pytorch_lora_weights.safetensors) "
            "applied on top of --model. Mutually exclusive with --transformer_path."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS.keys()),
        default="cityscapes",
        help="Dataset layout to use for RGB/GT pairing.",
    )
    parser.add_argument(
        "--data_root",
        help="Dataset root containing the canonical RGB and GT folders.",
    )
    parser.add_argument(
        "--cityscapes_root",
        help="Backward-compatible alias for --data_root with --dataset cityscapes.",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--condition",
        choices=ACDC_CONDITIONS,
        default=None,
        help="Optional ACDC condition. If omitted, all available ACDC conditions are scanned.",
    )
    parser.add_argument(
        "--image_root",
        help=(
            "Optional RGB root to evaluate instead of the dataset RGB folder. "
            "Use this for image-to-image edited variants; GT still comes from "
            "--data_root/--gt_root."
        ),
    )
    parser.add_argument(
        "--gt_root",
        help="Optional exact GT mask root. Defaults to the selected dataset layout.",
    )
    parser.add_argument(
        "--image_suffixes",
        default=None,
        help=(
            "Comma-separated image suffix override, e.g. .png,.jpg. "
            "Also used when stripping edited-variant suffixes for GT pairing."
        ),
    )
    parser.add_argument(
        "--image_glob",
        default=None,
        help="Optional glob under image root(s), e.g. '**/*_edited.png'.",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help=(
            "Per-GPU FLUX inference batch size for fixed/randomdraw semantic segmentation. "
            "1 preserves the historical one-sample-per-call behavior. Batches are "
            "formed only from samples with compatible model geometry."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=0,
        help=(
            "Optional base diffusion seed. Each image receives a stable derived "
            "seed, independent of shard count and ordering; randomdraw repeats "
            "reuse it to isolate color-assignment effects."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Resolve image/GT pairs and exit before loading the model.",
    )

    # Kept for compatibility. Generated predictions are always snapped to the
    # nearest of the 19 Cityscapes classes so snapped RGB outputs have no extra
    # ignore/background color.
    parser.add_argument("--max_palette_dist", type=int, default=None)

    parser.add_argument(
        "--prompt_style",
        choices=("fixed", "randomdraw"),
        default="randomdraw",
        help=(
            "fixed: the original Cityscapes 19-class palette prompt. "
            "randomdraw: same fixed 19-class prompt template, but each image gets 19 "
            "per-image random colors from the shared 200-color palette; predictions are "
            "decoded against that image's own legend (tests per-image color following)."
        ),
    )
    parser.add_argument(
        "--legend_seed",
        type=int,
        default=0,
        help="Base seed for the per-image random colors of --prompt_style randomdraw.",
    )
    parser.add_argument(
        "--visual_legend_mode",
        choices=("none", "swatches", "class_labels", "full_labels"),
        default="none",
        help=(
            "Evaluation-only multi-image ablation. Add a second input image containing "
            "B: swatches, C: class-labelled swatches, or D: class/color/RGB-labelled "
            "swatches. The original text mapping remains in the prompt."
        ),
    )
    parser.add_argument(
        "--randomdraw_repeats",
        type=int,
        default=1,
        help=(
            "randomdraw only. Generate the same image this many times, each with an "
            "independent color draw, and majority-vote the decoded trainId masks into "
            "one prediction before scoring. 1 (default) is the original single-pass "
            "behavior, bit-for-bit. Voting cancels unlucky draws -- a class the model "
            "only mislabels under one particular color assignment is outvoted by the "
            "repeats that drew different colors -- so it measures the model's semantic "
            "ability separated from per-draw color luck. Saves independent and every "
            "cumulative 1-to-K metric/agreement output. Costs N x the generation time."
        ),
    )
    parser.add_argument(
        "--resume_from_out_dir",
        type=Path,
        default=None,
        help=(
            "randomdraw only. Extend a completed smaller repeat run into this "
            "larger --randomdraw_repeats target. Reuses decoded repeats from the "
            "source resume_cache, generates only the new suffix, and recomputes "
            "all per-repeat and cumulative 1-to-K outputs. The source must have "
            "matching evaluation/model provenance and remain unchanged."
        ),
    )
    parser.add_argument(
        "--color_mapping",
        choices=diversify.COLOR_MAPPING_MODES,
        default="best64_maxnegdraw",
        help="How the randomdraw per-image colors are drawn from the palette (same modes "
             "as dataset prep); match the checkpoint's training color_mapping so the eval "
             "legend colors come from the pool the model was trained on.",
    )

    # Keep generation at native input resolution if set.
    parser.add_argument("--native_resolution", action="store_true")
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help=(
            "Manual sharding: total number of evaluator shards. If omitted, "
            "WORLD_SIZE from torchrun is used when available."
        ),
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=None,
        help=(
            "Manual sharding: index of this evaluator shard. If omitted, "
            "RANK from torchrun is used when available."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=None,
        help="CUDA device index for this process. Usually set automatically by torchrun.",
    )
    parser.add_argument(
        "--shard_aggregate_timeout",
        type=int,
        default=3600,
        help="Seconds rank 0 waits for all shard metric files before aggregating.",
    )
    parser.add_argument(
        "--no_aggregate",
        action="store_true",
        help=(
            "Each rank writes only its own shard_metrics/ file and exits without "
            "merging. Use for cross-job array sharding, where a separate aggregate "
            "job (aggregate_semseg_shards.py) merges after all shards finish, so no "
            "shard sits idle holding a GPU while waiting for the others."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse images an earlier run already finished instead of regenerating "
            "them: a sample whose panels/ file exists is read back from "
            "pred_trainids/ and fed to the metrics accumulator unchanged. OFF by "
            "default, so a plain rerun always regenerates everything. Use after a "
            "run was killed mid-way (walltime, preemption, a cancelled step) to pay "
            "only for the samples that never finished. The shard's mIoU is identical "
            "either way -- resumed samples still contribute to the confusion matrix. "
            "Do NOT use it to 'top up' an output dir written with different weights, "
            "steps, colors or repeats: the guard checks only that a file exists, not "
            "what produced it. To extend N repeats to a larger target, use "
            "--resume_from_out_dir instead."
        ),
    )

    return parser.parse_args()


def resolve_transformer_load_args(transformer_path: str) -> tuple[str, str | None]:
    path = Path(transformer_path).expanduser()
    if (path / "transformer").is_dir():
        return str(path), "transformer"
    return str(path), None


def main():
    from common.flux2_runtime import load_flux2_pipeline

    args = parse_args()
    global COLOR_MAPPING
    COLOR_MAPPING = args.color_mapping

    if args.dataset == "coco-object-80":
        raise ValueError(
            "COCO-Object-80 requires grouped_semseg.py / PROMPT_STYLE=grouped: "
            "the 80 foreground classes are evaluated in three short-legend passes."
        )

    if args.limit is not None and args.limit <= 0:
        raise ValueError(f"--limit must be positive, got {args.limit}.")
    if args.randomdraw_repeats <= 0:
        raise ValueError(
            f"--randomdraw_repeats must be positive, got {args.randomdraw_repeats}."
        )
    if args.resume and args.resume_from_out_dir is not None:
        raise ValueError("Do not combine --resume with --resume_from_out_dir.")
    if args.resume_from_out_dir is not None and args.prompt_style != "randomdraw":
        raise ValueError("--resume_from_out_dir requires --prompt_style randomdraw.")
    if args.batch_size <= 0:
        raise ValueError(f"--batch_size must be positive, got {args.batch_size}.")
    if args.visual_legend_mode != "none" and args.prompt_style != "randomdraw":
        raise ValueError("--visual_legend_mode requires --prompt_style randomdraw.")

    shard = setup_sharding(args)

    data_root_arg = args.data_root or args.cityscapes_root
    if data_root_arg is None:
        raise ValueError("Provide --data_root, or --cityscapes_root for old commands.")

    if args.cityscapes_root and args.dataset != "cityscapes" and not args.data_root:
        raise ValueError("--cityscapes_root can only be used with --dataset cityscapes.")

    layout = DATASETS[args.dataset]
    semantic = layout.semantic
    num_classes = len(semantic.classes)
    data_root = Path(data_root_arg)
    out_dir = Path(args.out_dir)
    image_root = Path(args.image_root) if args.image_root else None
    gt_root = Path(args.gt_root) if args.gt_root else None
    image_suffixes = parse_suffixes(args.image_suffixes)
    if image_suffixes is None:
        image_suffixes = COMMON_IMAGE_SUFFIXES if image_root else layout.image_suffixes

    raw_rgb_dir = out_dir / "pred_rgb_raw"
    snapped_rgb_dir = out_dir / "pred_rgb_snapped"
    trainid_dir = out_dir / "pred_trainids"
    repeat_snapped_rgb_dir = out_dir / "pred_rgb_snapped_repeats"
    cumulative_trainid_dir = out_dir / "pred_trainids_cumulative"
    cumulative_snapped_rgb_dir = out_dir / "pred_rgb_snapped_cumulative"
    ground_truth_rgb_dir = out_dir / "ground_truth_rgb"
    prompt_palette_gt_dir = out_dir / "ground_truth_rgb_prompt_palette"
    resume_cache_dir = out_dir / "resume_cache"
    panel_dir = out_dir / "panels"
    # For every cumulative 1->K vote (K >= 2), save the winning class's vote
    # share as 0..255 grayscale and as a labeled red-white-blue visualization.
    # The root-level grayscale file is the full 1->N agreement.
    agreement_dir = out_dir / "vote_agreement"
    agreement_color_dir = out_dir / "vote_agreement_color"

    all_samples = build_samples(
        layout=layout,
        data_root=data_root,
        split=args.split,
        condition=args.condition,
        image_root=image_root,
        gt_root=gt_root,
        image_suffixes=image_suffixes,
        image_glob=args.image_glob,
        limit=args.limit,
    )

    samples = shard_samples(all_samples, shard)

    resume_source_dir: Optional[Path] = None
    resume_source_repeats = 0
    if args.resume_from_out_dir is not None:
        expected_resume_config: dict[str, object] = {
            "dataset": args.dataset,
            "split": args.split,
            "prompt_style": args.prompt_style,
            "visual_legend_mode": args.visual_legend_mode,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "limit": args.limit,
            "legend_seed": args.legend_seed,
            "color_mapping": args.color_mapping,
            "generation_seed": args.generation_seed,
            "native_resolution": bool(args.native_resolution),
            "classes": list(semantic.classes),
            "palette": semantic.palette.tolist(),
            "model": normalized_model_reference(args.model),
            "transformer_path": normalized_model_reference(args.transformer_path),
            "lora_path": normalized_model_reference(args.lora_path) if args.lora_path else None,
        }
        resume_source_dir, resume_source_repeats = validate_incremental_repeat_source(
            args.resume_from_out_dir,
            out_dir,
            args.randomdraw_repeats,
            expected_resume_config,
            len(all_samples),
        )

    if args.dry_run:
        print(f"Dataset: {args.dataset}")
        print(f"Split: {args.split}")
        print(
            f"Shard: {shard.rank}/{shard.world_size} "
            f"({len(samples)} of {len(all_samples)} resolved image/GT pairs)"
        )
        if resume_source_dir is not None:
            print(
                f"Incremental source: {resume_source_dir} "
                f"({resume_source_repeats} cached -> {args.randomdraw_repeats} target repeats)"
            )
        print("First shard pairs:")
        for sample in samples[:5]:
            print(f"  image: {sample.image_path}")
            print(f"  gt:    {sample.gt_path}")
        cleanup_distributed(shard)
        return

    raw_rgb_dir.mkdir(parents=True, exist_ok=True)
    snapped_rgb_dir.mkdir(parents=True, exist_ok=True)
    trainid_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    if args.prompt_style == "randomdraw":
        prompt_palette_gt_dir.mkdir(parents=True, exist_ok=True)
        resume_cache_dir.mkdir(parents=True, exist_ok=True)
    if args.prompt_style == "randomdraw" and args.randomdraw_repeats > 1:
        agreement_dir.mkdir(parents=True, exist_ok=True)
        agreement_color_dir.mkdir(parents=True, exist_ok=True)
        repeat_snapped_rgb_dir.mkdir(parents=True, exist_ok=True)
        cumulative_trainid_dir.mkdir(parents=True, exist_ok=True)
        cumulative_snapped_rgb_dir.mkdir(parents=True, exist_ok=True)
        ground_truth_rgb_dir.mkdir(parents=True, exist_ok=True)
        for repeat_count in range(1, args.randomdraw_repeats + 1):
            (repeat_snapped_rgb_dir / f"repeat{repeat_count:02d}").mkdir(
                parents=True, exist_ok=True
            )
        for vote_count in range(2, args.randomdraw_repeats + 1):
            prefix_dir = f"1to{vote_count:02d}"
            (agreement_dir / prefix_dir).mkdir(parents=True, exist_ok=True)
            (agreement_color_dir / prefix_dir).mkdir(parents=True, exist_ok=True)
            (cumulative_trainid_dir / prefix_dir).mkdir(
                parents=True, exist_ok=True
            )
            (cumulative_snapped_rgb_dir / prefix_dir).mkdir(
                parents=True, exist_ok=True
            )
    shard_metric_dir = out_dir / "shard_metrics"
    shard_metric_dir.mkdir(parents=True, exist_ok=True)

    pipe, _torch, _metadata = load_flux2_pipeline(
        args.model,
        model_family=args.model_family,
        transformer_path=args.transformer_path,
        lora_path=args.lora_path,
        device=shard.device,
        offload=args.model_offload,
        offload_mode=args.offload_mode,
        torch_dtype=torch.bfloat16,
    )

    # COCO datasets decode GT through a category->super-category map loaded once.
    coco_category_map = (
        load_coco_category_map(data_root, semantic.coco_mask_type, args.split)
        if semantic.coco_mask_type is not None
        else None
    )

    accumulator = SegmentationAccumulator(num_classes=num_classes, ignore_index=255)
    # Keep a metric accumulator for every random color draw as well as the
    # existing accumulator for the final majority-voted prediction.
    repeats = args.randomdraw_repeats if args.prompt_style == "randomdraw" else 1
    repeat_accumulators = [
        SegmentationAccumulator(num_classes=num_classes, ignore_index=255)
        for _ in range(repeats)
    ]
    cumulative_vote_accumulators = [
        SegmentationAccumulator(num_classes=num_classes, ignore_index=255)
        for _ in range(max(0, repeats - 1))
    ]

    # Resolve the evaluation prompt. "fixed" uses the dataset's canonical prompt for
    # every image; "randomdraw" keeps that prompt template but gives each image one
    # per-image random color per class (built per-sample below).
    run_prompt = semantic.fixed_prompt
    legends_dir = out_dir / "legends"
    visual_legends_dir = out_dir / "visual_legends"
    if args.prompt_style == "randomdraw":
        legends_dir.mkdir(parents=True, exist_ok=True)
    if args.visual_legend_mode != "none":
        visual_legends_dir.mkdir(parents=True, exist_ok=True)

    progress_desc = f"Rank {shard.rank}/{shard.world_size}"
    if args.batch_size > 1 and args.visual_legend_mode != "none":
        raise ValueError(
            "--batch_size > 1 does not support --visual_legend_mode yet; use --batch_size 1. "
            "A visual legend is a second conditioning image, whereas batched FLUX inputs "
            "currently require one conditioning image per sample."
        )
    if args.batch_size > 1 and args.generation_seed is None:
        print(
            "WARNING: batched inference without --generation_seed is stochastic and will not "
            "reproduce a batch_size=1 run. Set --generation_seed for a comparable protocol."
        )

    resumed = 0  # samples reused from a previous run (--resume only)
    pending_by_geometry: dict[tuple[object, ...], list[PendingSemsegGeneration]] = {}

    def resize_to_gt(mask: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """Resize a decoded prediction to GT size without interpolating label ids."""
        if mask.shape == gt.shape:
            return mask
        return np.array(
            Image.fromarray(mask).resize(
                (gt.shape[1], gt.shape[0]),
                resample=Image.NEAREST,
            ),
            dtype=np.uint8,
        )

    def finish_sample(state: PendingSemsegSample) -> None:
        if any(item is None for item in state.repeat_trainids):
            return
        repeat_trainids = [item for item in state.repeat_trainids if item is not None]
        if len(repeat_trainids) != repeats:
            raise RuntimeError(f"Incomplete repeat state for {state.sample.output_stem}.")

        if args.prompt_style == "randomdraw":
            save_repeat_trainids_cache(
                resume_cache_dir / f"{state.sample.output_stem}.npz",
                repeat_trainids,
            )

        if args.prompt_style == "randomdraw":
            if any(item is None for item in state.repeat_legends):
                raise RuntimeError(f"Missing randomdraw legend for {state.sample.output_stem}.")
            repeat_legends = [item for item in state.repeat_legends if item is not None]
            repeat_prompts = [item for item in state.repeat_prompts if item is not None]
            if len(repeat_legends) != repeats or len(repeat_prompts) != repeats:
                raise RuntimeError(f"Incomplete randomdraw prompt state for {state.sample.output_stem}.")
            legend_payload = {
                "classes": {name: list(color) for name, color in zip(semantic.classes, repeat_legends[0])},
                "prompt": (
                    repeat_prompts[0]
                    if repeats == 1
                    else semantic.prompt_builder(repeat_legends[0])
                ),
                "repeats": repeats,
                "visual_legend_mode": args.visual_legend_mode,
                "repeat_legends": [
                    {name: list(color) for name, color in zip(semantic.classes, legend)}
                    for legend in repeat_legends
                ],
            }
            with open(legends_dir / f"{state.sample.output_stem}.json", "w") as handle:
                json.dump(legend_payload, handle, indent=2)

        for rep, repeat_trainids_item in enumerate(repeat_trainids):
            repeat_accumulators[rep].update(repeat_trainids_item, state.gt)

        if args.prompt_style == "randomdraw":
            # Mirror pred_rgb_raw's filenames so every generated repeat has an
            # exact ground-truth counterpart rendered with that repeat's prompt
            # palette. The unsuffixed repeat-1 GT also matches the root-level
            # final pred_rgb_snapped mask, which uses repeat 1's palette.
            for rep, repeat_legend in enumerate(repeat_legends):
                prompt_palette_gt_rgb = trainids_to_rgb(
                    state.gt, palette=repeat_legend
                )
                Image.fromarray(prompt_palette_gt_rgb).save(
                    prompt_palette_gt_dir
                    / repeat_output_filename(state.sample.output_stem, rep)
                )

        if args.prompt_style == "randomdraw" and repeats > 1:
            # Use one stable, canonical color language for paper comparisons.
            # The GT is identical for every repeat, so save it only once.
            canonical_gt_rgb = trainids_to_rgb(state.gt, palette=semantic.palette)
            Image.fromarray(canonical_gt_rgb).save(
                ground_truth_rgb_dir / state.save_name
            )
            for rep, repeat_trainids_item in enumerate(repeat_trainids, start=1):
                repeat_snapped_rgb = trainids_to_rgb(
                    repeat_trainids_item, palette=semantic.palette
                )
                Image.fromarray(repeat_snapped_rgb).save(
                    repeat_snapped_rgb_dir
                    / f"repeat{rep:02d}"
                    / state.save_name
                )

        if repeats == 1:
            pred_trainids = repeat_trainids[0]
        else:
            pred_trainids = None
            for vote_count, cumulative_pred, agreement in cumulative_majority_vote_trainids(
                repeat_trainids, num_classes
            ):
                if vote_count < 2:
                    continue
                cumulative_vote_accumulators[vote_count - 2].update(
                    cumulative_pred, state.gt
                )
                prefix_dir = f"1to{vote_count:02d}"
                Image.fromarray(cumulative_pred).save(
                    cumulative_trainid_dir / prefix_dir / state.save_name
                )
                cumulative_snapped_rgb = trainids_to_rgb(
                    cumulative_pred, palette=semantic.palette
                )
                Image.fromarray(cumulative_snapped_rgb).save(
                    cumulative_snapped_rgb_dir / prefix_dir / state.save_name
                )
                grayscale = Image.fromarray((agreement * 255).astype(np.uint8))
                grayscale.save(agreement_dir / prefix_dir / state.save_name)
                colorized_agreement = render_vote_agreement_visualization(
                    agreement, vote_count
                )
                colorized_agreement.save(
                    agreement_color_dir / prefix_dir / state.save_name
                )
                if vote_count == repeats:
                    pred_trainids = cumulative_pred
                    # Root-level aliases for the final 1->N maps.
                    grayscale.save(agreement_dir / state.save_name)
                    colorized_agreement.save(agreement_color_dir / state.save_name)
            if pred_trainids is None:
                raise RuntimeError(f"No cumulative vote produced for {state.sample.output_stem}.")

        Image.fromarray(pred_trainids).save(trainid_dir / state.save_name)
        vis_palette = (
            repeat_legends[0] if args.prompt_style == "randomdraw" else semantic.palette
        )
        snapped_rgb = trainids_to_rgb(pred_trainids, palette=vis_palette)
        Image.fromarray(snapped_rgb).save(snapped_rgb_dir / state.save_name)

        if state.panel_pred_rgb is None or state.panel_prompt is None:
            raise RuntimeError(f"Missing panel state for {state.sample.output_stem}.")
        gt_rgb = trainids_to_rgb(state.gt, palette=vis_palette)
        panel_cells = [(state.image, Image.LANCZOS)]
        panel_labels = ["input"]
        if state.panel_visual_legend is not None:
            panel_cells.append((state.panel_visual_legend, Image.NEAREST))
            panel_labels.append("visual legend")
        panel_cells.extend(
            [
                (Image.fromarray(gt_rgb), Image.NEAREST),
                (state.panel_pred_rgb, Image.LANCZOS),
                (Image.fromarray(snapped_rgb), Image.NEAREST),
            ]
        )
        panel_labels.extend(PANEL_LABELS[1:])
        make_panel(panel_cells, labels=panel_labels, header=state.panel_prompt).save(
            panel_dir / state.save_name
        )
        accumulator.update(pred_trainids, state.gt)

    def process_generation_batch(tasks: Sequence[PendingSemsegGeneration]) -> None:
        if not tasks:
            return
        with torch.inference_mode():
            pred_rgbs = run_pipeline_batch(
                pipe,
                [task.request for task in tasks],
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
            )
        for task, pred_rgb in zip(tasks, pred_rgbs):
            state = task.state
            pred_rgb.save(
                raw_rgb_dir
                / repeat_output_filename(state.sample.output_stem, task.repeat)
            )
            if task.repeat == 0:
                state.panel_pred_rgb = pred_rgb
                state.panel_prompt = task.request.prompt
                state.panel_visual_legend = task.visual_legend

            if args.prompt_style == "fixed":
                decoded = rgb_to_trainids_via_legend(pred_rgb, semantic.palette.tolist())
            else:
                if task.legend_colors is None:
                    raise RuntimeError(f"Missing randomdraw colors for {state.sample.output_stem}.")
                decoded = rgb_to_trainids_via_legend(pred_rgb, task.legend_colors)
            state.repeat_trainids[task.repeat] = resize_to_gt(decoded, state.gt)
            finish_sample(state)

    def enqueue_generation(task: PendingSemsegGeneration) -> None:
        geometry_key = (
            task.state.image.size,
            task.request.height,
            task.request.width,
            isinstance(task.request.image, (list, tuple)),
        )
        queue = pending_by_geometry.setdefault(geometry_key, [])
        queue.append(task)
        if len(queue) >= args.batch_size:
            to_process = list(queue)
            queue.clear()
            process_generation_batch(to_process)

    for sample in tqdm(samples, desc=progress_desc):
        source_img = Image.open(sample.image_path).convert("RGB")
        source_gt = load_gt_mask(sample.gt_path, semantic, coco_category_map)
        img = source_img
        gt = source_gt
        generation_height = img.height if args.native_resolution else None
        generation_width = img.width if args.native_resolution else None
        save_name = f"{sample.output_stem}.png"

        # The panel is written last, so it marks a fully materialized sample.
        if args.resume and repeats == 1 and (panel_dir / save_name).exists():
            cached_path = trainid_dir / save_name
            if cached_path.exists():
                cached = np.array(Image.open(cached_path), dtype=np.uint8)
                if cached.shape == gt.shape:
                    accumulator.update(cached, gt)
                    repeat_accumulators[0].update(cached, gt)
                    resumed += 1
                    continue

        state = PendingSemsegSample(
            sample=sample,
            image=img,
            gt=gt,
            save_name=save_name,
            repeat_trainids=[None] * repeats,
            repeat_legends=[None] * repeats,
            repeat_color_names=[None] * repeats,
            repeat_prompts=[None] * repeats,
        )
        cached_repeat_trainids: Optional[np.ndarray] = None
        if resume_source_dir is not None:
            cached_repeat_trainids = load_repeat_trainids_cache(
                resume_source_dir / "resume_cache" / f"{sample.output_stem}.npz",
                resume_source_repeats,
                gt.shape,
            )
            for rep in range(resume_source_repeats):
                state.repeat_trainids[rep] = cached_repeat_trainids[rep]

        for rep in range(repeats):
            prompt = run_prompt
            legend_colors: Optional[list[tuple[int, int, int]]] = None
            legend_color_names: Optional[list[str]] = None
            if args.prompt_style == "randomdraw":
                seed_key = f"{args.legend_seed}:{sample.output_stem}"
                if rep:
                    seed_key = f"{seed_key}:{rep}"
                rng = random.Random(seed_key)
                legend_colors, legend_color_names = sample_randomdraw_colors_with_names(
                    rng, num_classes
                )
                prompt = semantic.prompt_builder(legend_colors)
                state.repeat_legends[rep] = legend_colors
                state.repeat_color_names[rep] = legend_color_names
                state.repeat_prompts[rep] = prompt

            visual_legend = None
            generation_prompt = prompt
            generation_image: object = img
            if args.visual_legend_mode != "none":
                if legend_colors is None:
                    raise RuntimeError("Visual legends require a randomdraw legend.")
                visual_legend = render_visual_color_legend(
                    semantic.classes,
                    legend_colors,
                    legend_color_names,
                    args.visual_legend_mode,
                )
                generation_prompt = add_visual_legend_instruction(prompt, args.visual_legend_mode)
                generation_image = [img, visual_legend]
                visual_legend.save(
                    visual_legends_dir
                    / (save_name if rep == 0 else f"{sample.output_stem}_r{rep}.png")
                )

            if cached_repeat_trainids is not None and rep < resume_source_repeats:
                repeat_name = repeat_output_filename(sample.output_stem, rep)
                target_raw_path = raw_rgb_dir / repeat_name
                source_raw_path = resume_source_dir / "pred_rgb_raw" / repeat_name
                if not target_raw_path.is_file() and source_raw_path.is_file():
                    shutil.copy2(source_raw_path, target_raw_path)
                if not target_raw_path.is_file():
                    raise FileNotFoundError(
                        f"Cached repeat {rep + 1} for {sample.output_stem} lacks its "
                        f"raw RGB counterpart. Expected {source_raw_path} or "
                        f"pre-materialized {target_raw_path}. If the source was "
                        "archived, extract pred_rgb_raw/ from artifacts.tar first."
                    )
                if rep == 0:
                    with Image.open(target_raw_path) as cached_raw:
                        state.panel_pred_rgb = cached_raw.convert("RGB").copy()
                    state.panel_prompt = generation_prompt
                    state.panel_visual_legend = visual_legend
                continue

            generator = None
            if args.generation_seed is not None:
                # Keep noise fixed across randomdraw repeats; only the legend changes.
                generator = torch.Generator(device=shard.device).manual_seed(
                    stable_generation_seed(args.generation_seed, sample.output_stem)
                )
            request = PipelineBatchRequest(
                prompt=generation_prompt,
                image=generation_image,
                generator=generator,
                height=generation_height,
                width=generation_width,
            )
            enqueue_generation(
                PendingSemsegGeneration(
                    state=state,
                    repeat=rep,
                    request=request,
                    legend_colors=legend_colors,
                    visual_legend=visual_legend,
                )
            )

    for queued in pending_by_geometry.values():
        if queued:
            process_generation_batch(queued)

    if args.resume:
        print(
            f"Rank {shard.rank}/{shard.world_size}: reused {resumed}/{len(samples)} "
            f"samples from a previous run, generated {len(samples) - resumed}."
        )
    if resume_source_dir is not None:
        reused_predictions = len(samples) * resume_source_repeats
        generated_predictions = len(samples) * (repeats - resume_source_repeats)
        print(
            f"Rank {shard.rank}/{shard.world_size}: incrementally reused "
            f"{reused_predictions} repeat predictions (1-{resume_source_repeats}) "
            f"and generated {generated_predictions} "
            f"({resume_source_repeats + 1}-{repeats})."
        )

    shard_metric_path = shard_metric_dir / f"rank_{shard.rank:05d}_of_{shard.world_size:05d}.json"
    run_shard_metadata = {
        "model": normalized_model_reference(args.model),
        "transformer_path": normalized_model_reference(args.transformer_path),
            "lora_path": normalized_model_reference(args.lora_path) if args.lora_path else None,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "limit": args.limit,
        "legend_seed": args.legend_seed if args.prompt_style == "randomdraw" else None,
        "color_mapping": args.color_mapping if args.prompt_style == "randomdraw" else None,
        # The aggregator refuses to merge shards whose repeat protocol is unrecorded.
        "randomdraw_repeats": (
            args.randomdraw_repeats if args.prompt_style == "randomdraw" else None
        ),
        "generation_seed": args.generation_seed,
        "native_resolution": bool(args.native_resolution),
        "classes": list(semantic.classes),
        "palette": semantic.palette.tolist(),
        "resume_from_out_dir": (
            str(resume_source_dir) if resume_source_dir is not None else None
        ),
        "resume_from_repeats": resume_source_repeats,
    }
    with open(shard_metric_path, "w") as f:
        json.dump(
            {
                "rank": shard.rank,
                "world_size": shard.world_size,
                "dataset": args.dataset,
                "split": args.split,
                "prompt_style": args.prompt_style,
                "visual_legend_mode": args.visual_legend_mode,
                "batch_size": args.batch_size,
                "num_total_samples": len(all_samples),
                # 0 (or absent) means every sample was generated by this run.
                "resumed_samples": resumed,
                "reused_repeat_predictions": len(samples) * resume_source_repeats,
                **run_shard_metadata,
                "repeat_accumulators": [
                    repeat_accumulator.to_dict()
                    for repeat_accumulator in repeat_accumulators
                ],
                "cumulative_vote_accumulators": [
                    cumulative_accumulator.to_dict()
                    for cumulative_accumulator in cumulative_vote_accumulators
                ],
                **accumulator.to_dict(),
            },
            f,
            indent=2,
        )

    barrier(shard)

    # Cross-job array sharding: every shard just persists its metrics; a separate
    # aggregate job merges them later (afterok), so no shard idles on a GPU.
    if args.no_aggregate:
        print(f"Shard {shard.rank}/{shard.world_size} wrote metrics; aggregation deferred (--no_aggregate).")
        cleanup_distributed(shard)
        return

    if shard.rank != 0:
        cleanup_distributed(shard)
        return

    if shard.world_size > 1 and not shard.distributed:
        expected_metric_paths = [
            shard_metric_dir / f"rank_{rank:05d}_of_{shard.world_size:05d}.json"
            for rank in range(shard.world_size)
        ]
        wait_for_shard_metrics(expected_metric_paths, args.shard_aggregate_timeout)

    merged_accumulator = SegmentationAccumulator(num_classes=num_classes, ignore_index=255)
    merged_repeat_accumulators = [
        SegmentationAccumulator(num_classes=num_classes, ignore_index=255)
        for _ in range(repeats)
    ]
    merged_cumulative_vote_accumulators = [
        SegmentationAccumulator(num_classes=num_classes, ignore_index=255)
        for _ in range(max(0, repeats - 1))
    ]
    for rank in range(shard.world_size):
        metric_path = shard_metric_dir / f"rank_{rank:05d}_of_{shard.world_size:05d}.json"
        with open(metric_path) as f:
            shard_payload = json.load(f)
        for key, expected in run_shard_metadata.items():
            if shard_payload.get(key) != expected:
                raise ValueError(
                    f"Shard provenance mismatch in {metric_path}: {key}="
                    f"{shard_payload.get(key)!r}, expected {expected!r}."
                )
        merged_accumulator.merge(SegmentationAccumulator.from_dict(shard_payload))
        shard_repeat_accumulators = shard_payload.get("repeat_accumulators")
        if not isinstance(shard_repeat_accumulators, list) or len(shard_repeat_accumulators) != repeats:
            raise ValueError(
                f"Shard {metric_path} has invalid per-repeat metrics: expected {repeats} accumulators."
            )
        for rep, repeat_payload in enumerate(shard_repeat_accumulators):
            merged_repeat_accumulators[rep].merge(
                SegmentationAccumulator.from_dict(repeat_payload)
            )
        shard_cumulative_accumulators = shard_payload.get(
            "cumulative_vote_accumulators", []
        )
        expected_cumulative = max(0, repeats - 1)
        if (
            not isinstance(shard_cumulative_accumulators, list)
            or len(shard_cumulative_accumulators) != expected_cumulative
        ):
            raise ValueError(
                f"Shard {metric_path} has invalid cumulative-vote metrics: "
                f"expected {expected_cumulative} accumulators."
            )
        for prefix_index, cumulative_payload in enumerate(
            shard_cumulative_accumulators
        ):
            merged_cumulative_vote_accumulators[prefix_index].merge(
                SegmentationAccumulator.from_dict(cumulative_payload)
            )

    results = merged_accumulator.compute()
    results_serializable = {k: float(v) for k, v in results.items()}
    results_serializable["per_class_iou"] = merged_accumulator.per_class_iou(
        semantic.classes
    )
    results_serializable["prompt_style"] = args.prompt_style
    results_serializable["visual_legend_mode"] = args.visual_legend_mode
    results_serializable["batch_size"] = args.batch_size
    if args.prompt_style == "randomdraw":
        results_serializable.update(
            {
                "randomdraw_repeats": repeats,
                "randomdraw_repeats_used": repeats,
                "randomdraw_repeat_range": [1, repeats],
                "aggregation": (
                    "single_randomdraw_pass"
                    if repeats == 1
                    else "cumulative_majority_vote"
                ),
            }
        )
    out_json = out_dir / f"{args.dataset}_{args.split}_results.json"
    with open(out_json, "w") as f:
        json.dump(results_serializable, f, indent=2)

    if args.prompt_style == "randomdraw":
        for rep, repeat_accumulator in enumerate(merged_repeat_accumulators, start=1):
            repeat_results = {
                key: float(value) for key, value in repeat_accumulator.compute().items()
            }
            repeat_results["per_class_iou"] = repeat_accumulator.per_class_iou(
                semantic.classes
            )
            repeat_results.update(
                {
                    "prompt_style": args.prompt_style,
                    "visual_legend_mode": args.visual_legend_mode,
                    "batch_size": args.batch_size,
                    "randomdraw_repeat": rep,
                    "randomdraw_repeats": repeats,
                    "aggregation": "single_randomdraw_pass",
                }
            )
            repeat_json = out_dir / f"{args.dataset}_{args.split}_results_repeat{rep:02d}.json"
            with open(repeat_json, "w") as f:
                json.dump(repeat_results, f, indent=2)

        for vote_count, cumulative_accumulator in enumerate(
            merged_cumulative_vote_accumulators, start=2
        ):
            cumulative_results = {
                key: float(value)
                for key, value in cumulative_accumulator.compute().items()
            }
            cumulative_results["per_class_iou"] = (
                cumulative_accumulator.per_class_iou(semantic.classes)
            )
            cumulative_results.update(
                {
                    "prompt_style": args.prompt_style,
                    "visual_legend_mode": args.visual_legend_mode,
                    "batch_size": args.batch_size,
                    "randomdraw_repeats": repeats,
                    "randomdraw_repeats_used": vote_count,
                    "randomdraw_repeat_range": [1, vote_count],
                    "aggregation": "cumulative_majority_vote",
                    "agreement_definition": (
                        "winning-class vote share among non-abstaining random "
                        "color assignments; vote stability, not calibrated "
                        "correctness probability"
                    ),
                }
            )
            cumulative_json = (
                out_dir
                / f"{args.dataset}_{args.split}_results_1to{vote_count:02d}.json"
            )
            with open(cumulative_json, "w") as f:
                json.dump(cumulative_results, f, indent=2)

    eval_config = {
        "dataset": args.dataset,
        "split": args.split,
        "prompt_style": args.prompt_style,
        "visual_legend_mode": args.visual_legend_mode,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "limit": args.limit,
        "num_images": merged_accumulator.num_samples,
        "model": normalized_model_reference(args.model),
        "transformer_path": normalized_model_reference(args.transformer_path),
            "lora_path": normalized_model_reference(args.lora_path) if args.lora_path else None,
        "resume_from_out_dir": (
            str(resume_source_dir) if resume_source_dir is not None else None
        ),
        "resume_from_repeats": resume_source_repeats,
        "reused_repeat_predictions": (
            merged_accumulator.num_samples * resume_source_repeats
        ),
        "generated_repeat_range": (
            [resume_source_repeats + 1, repeats]
            if resume_source_dir is not None
            else [1, repeats]
        ),
        "resume_cache": (
            "resume_cache/<image>.npz; retained unpacked for incremental extension"
            if args.prompt_style == "randomdraw"
            else None
        ),
        "legend_seed": args.legend_seed if args.prompt_style == "randomdraw" else None,
        "palette_size": (
            len(diversify.FULL_PALETTE) if args.prompt_style == "randomdraw" else None
        ),
        "color_mapping": args.color_mapping if args.prompt_style == "randomdraw" else None,
        # >1 means the score is over majority-voted masks, not single passes;
        # such a run is NOT comparable with a repeats=1 run.
        "randomdraw_repeats": args.randomdraw_repeats if args.prompt_style == "randomdraw" else None,
        # randomdraw prompts are per-image (see legends/); record the shared one otherwise.
        "prompt": run_prompt if args.prompt_style != "randomdraw" else None,
        "classes": list(semantic.classes),
        "palette": semantic.palette.tolist(),
        "native_resolution": args.native_resolution,
        "generation_seed": args.generation_seed,
        "paper_qualitative_masks": (
            {
                "palette": "canonical_dataset_palette",
                "ground_truth_rgb": "ground_truth_rgb/<image>.png (saved once)",
                "per_repeat_prompt_palette_ground_truth": (
                    "ground_truth_rgb_prompt_palette/<image>[_rK].png; filenames "
                    "and prompt palettes mirror pred_rgb_raw exactly"
                ),
                "final_prompt_palette_ground_truth": (
                    "ground_truth_rgb_prompt_palette/<image>.png; repeat-1 palette "
                    "matching pred_rgb_snapped/<image>.png"
                ),
                "per_repeat_snapped_rgb": (
                    "pred_rgb_snapped_repeats/repeatKK/<image>.png"
                ),
                "cumulative_trainids": (
                    "pred_trainids_cumulative/1toKK/<image>.png"
                ),
                "cumulative_snapped_rgb": (
                    "pred_rgb_snapped_cumulative/1toKK/<image>.png"
                ),
                "raw_rgb_note": (
                    "Raw model RGB exists only per independent repeat in "
                    "pred_rgb_raw; cumulative votes are formed in class-id space."
                ),
            }
            if args.prompt_style == "randomdraw" and repeats > 1
            else None
        ),
        "cumulative_vote_prefixes": (
            list(range(2, repeats + 1))
            if args.prompt_style == "randomdraw" and repeats > 1
            else []
        ),
        "vote_agreement_definition": (
            "winning-class vote share among non-abstaining random color "
            "assignments; vote stability, not calibrated correctness probability"
            if args.prompt_style == "randomdraw" and repeats > 1
            else None
        ),
        "vote_agreement_color_scale": (
            {
                "orientation": "red_low_blue_high",
                "low_0": "#B2182B",
                "mid_0_5": "#F7F7F7",
                "high_1": "#2166AC",
                "ticks_percent": [0, 25, 50, 75, 100],
            }
            if args.prompt_style == "randomdraw" and repeats > 1
            else None
        ),
    }
    with open(out_dir / "eval_config.json", "w") as f:
        json.dump(eval_config, f, indent=2)

    print("\nEvaluation Results")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Prompt style: {args.prompt_style}")
    print(f"Visual legend mode: {args.visual_legend_mode}")
    print(f"Images evaluated: {merged_accumulator.num_samples}")
    print(f"Evaluator shards: {shard.world_size}")
    for k, v in results.items():
        print(f"{k}: {v:.2f}")
    print("=" * 50)
    print(f"Saved results to: {out_json}")
    print(f"Raw RGB predictions: {raw_rgb_dir}")
    print(f"Snapped RGB predictions: {snapped_rgb_dir}")
    print(f"TrainId predictions: {trainid_dir}")
    print(f"Panels: {panel_dir}")
    if args.prompt_style == "randomdraw" and repeats > 1:
        print(
            "Cumulative vote results: "
            f"{out_dir / f'{args.dataset}_{args.split}_results_1toKK.json'}"
        )
        print(f"Per-repeat snapped RGB: {repeat_snapped_rgb_dir}")
        print(f"Cumulative TrainId masks: {cumulative_trainid_dir}")
        print(f"Cumulative snapped RGB: {cumulative_snapped_rgb_dir}")
        print(f"Canonical ground-truth RGB: {ground_truth_rgb_dir}")
        print(f"Prompt-palette ground-truth RGB: {prompt_palette_gt_dir}")
        print(f"Grayscale vote agreement: {agreement_dir}")
        print(f"Color vote agreement: {agreement_color_dir}")
    cleanup_distributed(shard)


if __name__ == "__main__":
    main()