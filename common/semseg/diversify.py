#!/usr/bin/env python3
"""Shared semseg diversification vocabulary: palettes, color notations, prompt
templates, the per-image color/notation sampler, and the prediction decoder.

This is the single source of truth imported by BOTH dataset preparation
(dataset_prep/semseg/dataset_prep.py) and evaluation (evaluation/semseg/*), so the
randomized (prompt, mask) construction used at train time and the prompt/decoding
used at eval time can never drift apart.

Design:
  * Two committed palettes (see build_palette.py / build_named_palette.py):
      - palette_200.json : 200 well-separated colors, used for the rgb/hex notation
        styles and for painting those masks (names irrelevant -- exact match).
      - palette_named.json: 37 everyday, clearly-separated NAMED colors, used only
        for the natural-language ("name") notation.
  * Three color-notation styles, chosen once per prompt and uniform within it:
      name  -> "red"        (only when the prompt needs <= len(named palette) colors;
                             otherwise we fall back to rgb/hex)
      rgb   -> "[255,0,0]"  (common.colors.format_rgb -- the one spelling used by
                             semseg, refseg and instanceseg alike)
      hex   -> "#FF0000"
  * Many base templates, dict-style ({"cat": red, ...}) and sentence-style
    ("The cat is red."), so the phrasing varies per prompt.
  * Per-image sampler supports several color_mapping modes (see
    COLOR_MAPPING_MODES / sample_colors): the legacy greedy-with-relaxation draw
    over the full palette, fixed best-separated pools (top 64 / top 32) with
    rank-weighted farthest backfill on overflow, and a per-image pool sized by the
    number of classes present in the mask. All of them keep every legend decodable.
  * The decoder always treats black [0,0,0] as background -> ignore(255), so black
    voids never snap to a dark class.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    from .build_palette import srgb_to_lab          # imported as common.semseg.diversify
except ImportError:
    from build_palette import srgb_to_lab            # run as a script from this dir

try:
    from common.colors import BLACK_TOKEN, format_rgb
except ImportError:                                  # run as a script from this dir
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from common.colors import BLACK_TOKEN, format_rgb

HERE = Path(__file__).resolve().parent

IGNORE_INDEX = 255
BACKGROUND_COLOR = (0, 0, 0)
NOTATION_STYLES = ("name", "rgb", "hex")
# Every semantic-segmentation prompt must start with this task tag.
SEMSEG_TASK_TAG = "[TASK: SEMANTIC_SEGMENTATION]"


def _ensure_task_tag(text: str) -> str:
    """Guarantee a prompt begins with SEMSEG_TASK_TAG (no double-tagging)."""
    return text if text.startswith(SEMSEG_TASK_TAG) else f"{SEMSEG_TASK_TAG} {text}"

try:
    from .taxonomy import CITYSCAPES_CLASSES as _CS_CLASSES, CITYSCAPES_PALETTE as _CS_PALETTE
except ImportError:
    from taxonomy import CITYSCAPES_CLASSES as _CS_CLASSES, CITYSCAPES_PALETTE as _CS_PALETTE


def cityscapes_fixed_prompt(colors: Optional[Sequence[Sequence[int]]] = None) -> str:
    """The original fixed Cityscapes-19 prompt. With ``colors`` (19 RGBs) it keeps
    the exact same template/phrasing but swaps in those colors -- this is the eval
    "randomdraw" style: identical prompt structure, per-image random colors."""
    cols = _CS_PALETTE if colors is None else colors
    mapping = ", ".join(f"'{n}': {format_rgb(c)}" for n, c in zip(_CS_CLASSES, cols))
    return (
        "[TASK: SEMANTIC_SEGMENTATION] Generate a semantic segmentation visualization of the input image, "
        "using exactly these 19 Cityscapes classes and this color mapping: { " + mapping + " "
        "}. Each pixel must be colored as exactly one of these semantic classes. "
        "Output only the segmentation visualization image."
    )


# Kept verbatim for the eval "fixed" style and the optional canonical-anchor pair.
FIXED_CITYSCAPES_PROMPT = cityscapes_fixed_prompt()


def _load_palette(filename: str) -> list[dict]:
    return json.loads((HERE / filename).read_text(encoding="utf-8"))


FULL_PALETTE = _load_palette("palette_200.json")          # rgb/hex styles
NAMED_PALETTE = _load_palette("palette_named.json")        # name style
FULL_RGB = np.array([p["rgb"] for p in FULL_PALETTE], dtype=np.int64)
NAMED_RGB = np.array([p["rgb"] for p in NAMED_PALETTE], dtype=np.int64)
FULL_LAB = srgb_to_lab(FULL_RGB)
NAMED_LAB = srgb_to_lab(NAMED_RGB)
NAME_CAPACITY = len(NAMED_PALETTE)  # 37; name notation unavailable beyond this


def _separation_order(lab: np.ndarray) -> list[int]:
    """Farthest-point ordering of a palette in CIELAB: prefix of any length K is
    the greedily most-separated K-color sub-palette (seeded from the most
    chromatic entry, matching build_palette.py)."""
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    chosen = [int(np.argmax(chroma))]
    d = ((lab - lab[chosen[0]]) ** 2).sum(axis=1)
    while len(chosen) < len(lab):
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, ((lab - lab[nxt]) ** 2).sum(axis=1))
    return chosen


# palette_200.json is committed in farthest-point selection order already; the
# named palette is in priority order (most basic name first), so its separation
# order is computed here once for the pool-based color_mapping modes.
FULL_SEP_ORDER = list(range(len(FULL_PALETTE)))
NAMED_SEP_ORDER = _separation_order(NAMED_LAB)


# ----------------------------------------------------------------------------- #
# Color notation                                                                #
# ----------------------------------------------------------------------------- #
def render_color(rgb: Sequence[int], style: str, name: Optional[str] = None) -> str:
    """Render one color as its token in the chosen notation style."""
    r, g, b = (int(v) for v in rgb)
    if style == "name":
        if name is None:
            raise ValueError("name style requires a color name")
        return name
    if style == "hex":
        return f"#{r:02X}{g:02X}{b:02X}"
    if style == "rgb":
        return format_rgb(rgb)
    raise ValueError(f"unknown notation style: {style}")


# ----------------------------------------------------------------------------- #
# Legend rendering + templates                                                  #
# ----------------------------------------------------------------------------- #
# Dict-style base templates: the rendered "{legend}" is a JSON-ish mapping.
_DICT_TEMPLATES = [
    "[TASK: SEMANTIC_SEGMENTATION] Generate a semantic segmentation visualization of "
    "the input image, using this color mapping: {legend} Color each pixel of a listed "
    "class with its color and leave everything else black. Output only the "
    "segmentation visualization image.",
    "[TASK: SEMANTIC_SEGMENTATION] Segment only the following classes, using exactly "
    "these colors: {legend} All other pixels must be black. Output only the "
    "segmentation image.",
    "[TASK: SEMANTIC_SEGMENTATION] Produce a segmentation mask that highlights only "
    "these classes with the given colors: {legend} Everything not listed must be "
    "black. Output just the mask image.",
    "[TASK: SEMANTIC_SEGMENTATION] Using the color legend below, paint each named "
    "class in its color and leave the rest black: {legend} Output only the "
    "segmentation visualization image.",
    "Generate a semantic segmentation visualization image, using this color mapping: "
    "{legend}.",
    "Generate a visualization image of semantic segmentation, using this color "
    "mapping: {legend}.",
]

# Sentence-style templates: (lead-in, per-class clause). One clause per class.
_SENTENCE_TEMPLATES = [
    ("This image is a per-pixel class labeling of the input.", "The {cls} are represented by {tok}."),
    ("Generate a semantic segmentation visualization of the input.", "The {cls} is {tok}."),
    ("Produce a per-pixel semantic segmentation of the image.", "The {cls} should be colored {tok}."),
    ("Color the input image by semantic class.", "Paint the {cls} {tok}."),
]


def _render_legend_dict(pairs: list[tuple[str, str]], quote_value: bool) -> str:
    """{"cat": red, "lock": pink} (values quoted only for the name style)."""
    items = []
    for cls, tok in pairs:
        val = f'"{tok}"' if quote_value else tok
        items.append(f'"{cls}": {val}')
    return "{" + ", ".join(items) + "}"


def build_prompt(legend_entries: Sequence[tuple[str, Sequence[int], str]], style: str, rng: random.Random,
                 background_token: Optional[str] = None) -> str:
    """Build one randomized prompt for the given legend.

    ``legend_entries`` is a list of (class_name, rgb, color_name); ``color_name`` is
    the everyday name of the assigned color, used as the token for the name style.
    ``style`` selects the color notation; the colors are exactly those painted into
    the mask, so prompt and target match. ``background_token`` (when given) appends
    an explicit background entry; by default unlisted pixels are simply black.
    """
    pairs = [
        (cls, render_color(rgb, style, name=color_name))
        for cls, rgb, color_name in legend_entries
    ]

    use_sentence = rng.random() < len(_SENTENCE_TEMPLATES) / (len(_SENTENCE_TEMPLATES) + len(_DICT_TEMPLATES))
    if use_sentence:
        lead, clause = rng.choice(_SENTENCE_TEMPLATES)
        body = " ".join(clause.format(cls=cls, tok=tok) for cls, tok in pairs)
        text = f"{lead} {body}"
        if background_token is not None:
            text += f" Everything else is {background_token}."
        return _ensure_task_tag(text)

    template = rng.choice(_DICT_TEMPLATES)
    legend_pairs = list(pairs)
    if background_token is not None:
        legend_pairs.append(("background", background_token))
    legend = _render_legend_dict(legend_pairs, quote_value=(style == "name"))
    return _ensure_task_tag(template.format(legend=legend))


# ----------------------------------------------------------------------------- #
# Per-image sampling                                                            #
# ----------------------------------------------------------------------------- #
def choose_notation_style(rng: random.Random, n_colors: int) -> str:
    """Pick a notation style; exclude 'name' when more colors are needed than the
    named palette can supply (then only the exact rgb/hex styles work)."""
    styles = list(NOTATION_STYLES) if n_colors <= NAME_CAPACITY else ["rgb", "hex"]
    return rng.choice(styles)


# How a variant's colors are drawn from the palette (see sample_colors):
#   p200_randomnegdraw -> legacy: shuffle the whole palette, greedy >= min_delta_e,
#                         silently relax when the floor can't be met (backfill is a
#                         random rejected color, unchecked against the chosen set).
#   p100_randomnegdraw -> exactly the p200 scheme, but the candidate set is only
#                         the 100 most-separated colors.
#   best64_maxnegdraw  -> uniform draw from the 64 most-separated colors (pairwise
#                         deltaE >= ~24); if a legend needs more, each extra color is
#                         drawn from outside the pool with probability 1/rank, ranked
#                         farthest-first by min deltaE to the already-selected set.
#   best32_maxnegdraw  -> same with a 32-color pool (pairwise deltaE >= ~31).
#   bestdrawperimg     -> pool sized per image: the N most-separated colors for the
#                         N classes present in the mask, randomly assigned; subset
#                         and negative variants of that image draw from the same
#                         pool (grown to the legend size when negatives exceed it).
COLOR_MAPPING_MODES = (
    "p200_randomnegdraw",
    "p100_randomnegdraw",
    "best64_maxnegdraw",
    "best32_maxnegdraw",
    "bestdrawperimg",
)
_POOL_SIZES = {"best64_maxnegdraw": 64, "best32_maxnegdraw": 32}


def _sample_from_pool(rng: random.Random, k: int, pool_size: int,
                      lab_pal: np.ndarray, sep_order: Sequence[int]) -> list[int]:
    """Uniformly draw from the ``pool_size`` most-separated colors (a prefix of
    ``sep_order``); when ``k`` exceeds the pool, draw each extra color from the
    remaining palette with probability proportional to 1/rank, ranked
    farthest-first by min deltaE to everything selected so far."""
    pool = list(sep_order[:min(pool_size, len(sep_order))])
    rng.shuffle(pool)
    chosen = pool[:k]
    remaining = list(sep_order[len(pool):])
    while len(chosen) < k and remaining:
        d = np.sqrt(
            ((lab_pal[np.array(remaining)][:, None, :] - lab_pal[np.array(chosen)][None, :, :]) ** 2)
            .sum(-1)).min(axis=1)
        ranked = [remaining[j] for j in np.argsort(-d)]
        pick = rng.choices(ranked, weights=[1.0 / (r + 1) for r in range(len(ranked))], k=1)[0]
        remaining.remove(pick)
        chosen.append(pick)
    return chosen


def sample_colors(rng: random.Random, k: int, style: str, min_delta_e: float = 25.0,
                  color_mapping: str = "p200_randomnegdraw", n_present: Optional[int] = None,
                  ) -> list[tuple[tuple[int, int, int], str]]:
    """Pick ``k`` distinct colors from the style's palette according to
    ``color_mapping`` (see COLOR_MAPPING_MODES). ``min_delta_e`` only applies to the
    greedy p200/p100_randomnegdraw modes; ``n_present`` (the number of classes present
    in the image's mask) is required by bestdrawperimg and ignored elsewhere. Returns
    (rgb, color_name) pairs."""
    pal, rgb_pal, lab_pal, sep_order = (
        (NAMED_PALETTE, NAMED_RGB, NAMED_LAB, NAMED_SEP_ORDER) if style == "name"
        else (FULL_PALETTE, FULL_RGB, FULL_LAB, FULL_SEP_ORDER)
    )
    if color_mapping in _POOL_SIZES:
        chosen_idx = _sample_from_pool(rng, k, _POOL_SIZES[color_mapping], lab_pal, sep_order)
        return [(tuple(int(v) for v in rgb_pal[i]), pal[i]["name"]) for i in chosen_idx[:k]]
    if color_mapping == "bestdrawperimg":
        pool_size = max(n_present if n_present is not None else k, k)
        chosen_idx = _sample_from_pool(rng, k, pool_size, lab_pal, sep_order)
        return [(tuple(int(v) for v in rgb_pal[i]), pal[i]["name"]) for i in chosen_idx[:k]]
    if color_mapping == "p100_randomnegdraw":
        order = list(sep_order[:100])
    elif color_mapping == "p200_randomnegdraw":
        order = list(range(len(rgb_pal)))
    else:
        raise ValueError(f"unknown color_mapping: {color_mapping!r} (options: {COLOR_MAPPING_MODES})")
    rng.shuffle(order)
    chosen_idx: list[int] = []
    chosen_lab: list[np.ndarray] = []
    leftover: list[int] = []
    min_sq = min_delta_e * min_delta_e
    for i in order:
        if chosen_lab and min(((np.stack(chosen_lab) - lab_pal[i]) ** 2).sum(1)) < min_sq:
            leftover.append(i)
            continue
        chosen_idx.append(i)
        chosen_lab.append(lab_pal[i])
        if len(chosen_idx) == k:
            break
    for i in leftover:  # relax: take distinct entries even if closer than the floor
        if len(chosen_idx) == k:
            break
        chosen_idx.append(i)
    if color_mapping == "p100_randomnegdraw" and len(chosen_idx) < k:
        # A legend larger than the 100-color pool (huge negative-set draws only):
        # extend from the rest of the palette rather than return short, which would
        # desync the legend from the classes.
        for i in sep_order[100:]:
            if len(chosen_idx) == k:
                break
            chosen_idx.append(i)
    return [(tuple(int(v) for v in rgb_pal[i]), pal[i]["name"]) for i in chosen_idx[:k]]


def build_legend(class_names: Sequence[str], rng: random.Random,
                 background: bool = False, min_delta_e: float = 25.0,
                 color_mapping: str = "p200_randomnegdraw", n_present: Optional[int] = None,
                 ) -> tuple[str, list[tuple[int, int, int]], str, Optional[tuple[int, int, int]]]:
    """Top-level helper used by prep and eval: choose a notation style + colors for
    these classes and return (prompt, colors_in_class_order, style, background_color).

    ``colors_in_class_order[i]`` is the RGB assigned to ``class_names[i]`` -- the
    caller paints the mask with exactly these, so prompt and target stay consistent.
    When ``background`` is True an extra color is sampled, named "background" in the
    legend, and returned as ``background_color`` so the caller can paint all unselected
    pixels that color (instead of black); otherwise ``background_color`` is None.
    ``color_mapping``/``n_present`` select how colors are drawn (COLOR_MAPPING_MODES);
    the defaults keep the legacy behavior (and RNG stream) exactly.
    """
    n = len(class_names)
    k = n + (1 if background else 0)
    style = choose_notation_style(rng, k)
    sampled = sample_colors(rng, k, style, min_delta_e=min_delta_e,
                            color_mapping=color_mapping, n_present=n_present)
    colors = [rgb for rgb, _ in sampled[:n]]
    legend_entries = [(cls, rgb, cname) for cls, (rgb, cname) in zip(class_names, sampled[:n])]
    bg_color = None
    bg_token = None
    if background:
        bg_rgb, bg_name = sampled[n]
        bg_color = bg_rgb
        bg_token = render_color(bg_rgb, style, name=bg_name)
    prompt = build_prompt(legend_entries, style, rng, background_token=bg_token)
    return prompt, colors, style, bg_color


# ----------------------------------------------------------------------------- #
# Decoding (eval): nearest legend color, black -> ignore                        #
# ----------------------------------------------------------------------------- #
def rgb_to_trainids_via_legend(rgb_img, colors: Sequence[Sequence[int]]) -> np.ndarray:
    """Decode an RGB prediction against a per-image legend. ``colors[i]`` is the RGB
    the prompt assigned to label ``i``. Black is added as an extra snap target that
    decodes to ignore(255), so model-produced background voids don't snap to the
    nearest (often dark) class and pollute its IoU."""
    return rgb_to_trainids_via_legend_with_dist(rgb_img, colors)[0]


def rgb_to_trainids_via_legend_with_dist(
    rgb_img, colors: Sequence[Sequence[int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Like rgb_to_trainids_via_legend, but also returns the per-pixel squared-RGB
    distance to the snapped target (float32 HxW; black->ignore included). The
    grouped multi-pass eval merges K single-group passes by this snap confidence."""
    from PIL import Image
    arr = np.array(rgb_img.convert("RGB") if isinstance(rgb_img, Image.Image) else rgb_img, dtype=np.int32)
    h, w, _ = arr.shape
    palette = np.array(list(colors) + [list(BACKGROUND_COLOR)], dtype=np.int32)
    mapping = np.array(list(range(len(colors))) + [IGNORE_INDEX], dtype=np.uint8)
    flat = arr.reshape(-1, 3)
    dist = ((flat[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    idx = dist.argmin(axis=1)
    ids = mapping[idx].reshape(h, w)
    min_dist = dist[np.arange(dist.shape[0]), idx].reshape(h, w).astype(np.float32)
    return ids, min_dist


def merge_grouped_predictions(
    ids_maps: Sequence[np.ndarray], dist_maps: Sequence[np.ndarray],
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """Merge K single-group decodes (already mapped to FULL-taxonomy ids, with
    void=``ignore_index``) into one label map. Per pixel: among the passes that
    decoded a real class, take the one with the smallest snap distance; ties go to
    the lowest pass index; a pixel void in every pass stays void.

    Void entries are masked to +inf so they never win; on an all-inf pixel argmin
    returns pass 0, whose id there is ignore_index by construction."""
    ids = np.stack([np.asarray(m) for m in ids_maps])
    dist = np.stack([np.asarray(d, dtype=np.float64) for d in dist_maps])
    if ids.shape != dist.shape:
        raise ValueError(f"ids/dist shape mismatch: {ids.shape} vs {dist.shape}")
    dist = np.where(ids == ignore_index, np.inf, dist)
    best = dist.argmin(axis=0)  # first minimum -> lowest pass index on ties
    return np.take_along_axis(ids, best[None], axis=0)[0]


# ----------------------------------------------------------------------------- #
# Self-test / demo                                                              #
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    print(f"FULL_PALETTE: {len(FULL_PALETTE)}  NAMED_PALETTE: {len(NAMED_PALETTE)} (name capacity)")
    rng = random.Random(0)
    demo_classes = ["car", "road", "person", "tree", "sky"]
    print("\n--- 6 randomized prompts for the same 5 classes (style varies) ---")
    for _ in range(6):
        prompt, colors, style, bg = build_legend(demo_classes, rng, background=rng.random() < 0.5)
        print(f"\n[style={style}] colors={colors} bg={bg}")
        print(prompt)

    # Round-trip: paint a fake mask with the chosen colors, decode, check recovery.
    print("\n--- decode round-trip (perfect paint should recover labels) ---")
    prompt, colors, style, _ = build_legend(demo_classes, rng)
    h = w = 32
    label = np.random.default_rng(1).integers(0, len(demo_classes), size=(h, w))
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in enumerate(colors):
        rgb[label == i] = c
    decoded = rgb_to_trainids_via_legend(rgb, colors)
    ok = bool((decoded == label).all())
    print(f"style={style} colors={colors}")
    print("perfect round-trip recovers labels:", ok)

    # min separation actually handed out for a big legend (forces relaxation)
    big = [rgb for rgb, _ in sample_colors(random.Random(2), 60, "hex", min_delta_e=25.0)]
    lab = srgb_to_lab(np.array(big))
    D = np.sqrt(((lab[:, None] - lab[None]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
    print(f"\n60-color legend min pairwise deltaE (relaxed from 25): {D.min():.1f}")

    # color_mapping modes: distinctness, pool membership, and separation floors.
    print("\n--- color_mapping modes ---")
    ok_modes = True
    def _min_de(picks):
        l = srgb_to_lab(np.array([rgb for rgb, _ in picks]))
        M = np.sqrt(((l[:, None] - l[None]) ** 2).sum(-1)); np.fill_diagonal(M, np.inf)
        return M.min()
    for mode, pool_n in (("best64_maxnegdraw", 64), ("best32_maxnegdraw", 32)):
        pool_rgbs = {tuple(FULL_PALETTE[i]["rgb"]) for i in FULL_SEP_ORDER[:pool_n]}
        r = random.Random(3)
        small = sample_colors(r, min(pool_n, 20), "hex", color_mapping=mode)
        in_pool = all(rgb in pool_rgbs for rgb, _ in small)
        over = sample_colors(r, pool_n + 10, "hex", color_mapping=mode)
        distinct = len({rgb for rgb, _ in over}) == pool_n + 10
        ok = in_pool and distinct
        ok_modes = ok_modes and ok
        print(f"{mode}: within-pool draw min dE={_min_de(small):.1f}, "
              f"overflow k={pool_n + 10} distinct={distinct}, in_pool={in_pool}")
    for n_present, k in ((10, 4), (10, 10), (3, 12)):  # subset / full / negatives beyond pool
        picks = sample_colors(random.Random(4), k, "rgb",
                              color_mapping="bestdrawperimg", n_present=n_present)
        pool_rgbs = {tuple(FULL_PALETTE[i]["rgb"]) for i in FULL_SEP_ORDER[:max(n_present, k)]}
        ok = len({rgb for rgb, _ in picks}) == k and all(rgb in pool_rgbs for rgb, _ in picks)
        ok_modes = ok_modes and ok
        print(f"bestdrawperimg n_present={n_present} k={k}: min dE={_min_de(picks):.1f} ok={ok}")
    # p100_randomnegdraw: greedy stays inside the 100-color pool; huge k extends past it.
    p100_pool = {tuple(FULL_PALETTE[i]["rgb"]) for i in FULL_SEP_ORDER[:100]}
    p100_small = sample_colors(random.Random(6), 30, "hex", color_mapping="p100_randomnegdraw")
    p100_big = sample_colors(random.Random(6), 110, "hex", color_mapping="p100_randomnegdraw")
    ok_p100 = (all(rgb in p100_pool for rgb, _ in p100_small)
               and len({rgb for rgb, _ in p100_big}) == 110
               and sum(rgb in p100_pool for rgb, _ in p100_big) == 100)
    ok_modes = ok_modes and ok_p100
    print(f"p100_randomnegdraw: k=30 min dE={_min_de(p100_small):.1f} in_pool=True, "
          f"k=110 distinct with 100 in-pool: {ok_p100}")
    # name style must draw from the named palette's computed separation order.
    named_picks = sample_colors(random.Random(5), 8, "name", color_mapping="best32_maxnegdraw")
    named_pool = {tuple(NAMED_PALETTE[i]["rgb"]) for i in NAMED_SEP_ORDER[:32]}
    ok_named = all(rgb in named_pool for rgb, _ in named_picks)
    ok_modes = ok_modes and ok_named
    print(f"name style best32 pool draw ok: {ok_named}")

    # with-dist decode: perfectly painted pixels must snap at distance 0.
    print("\n--- with-dist decode + grouped merge ---")
    ids_d, dist_d = rgb_to_trainids_via_legend_with_dist(rgb, colors)
    ok_dist = bool((ids_d == decoded).all() and (dist_d[label >= 0] == 0).all())
    print("with-dist decode matches + painted dist==0:", ok_dist)

    # merge rule: smaller dist wins / tie -> lower pass / void never wins / all-void -> 255.
    ids0 = np.array([[1, 255, 3, 255]], dtype=np.uint8)
    ids1 = np.array([[2, 4, 5, 255]], dtype=np.uint8)
    d0 = np.array([[10.0, 0.0, 5.0, 0.0]], dtype=np.float32)
    d1 = np.array([[3.0, 7.0, 5.0, 0.0]], dtype=np.float32)
    merged_map = merge_grouped_predictions([ids0, ids1], [d0, d1])
    want = np.array([[2, 4, 3, 255]], dtype=np.uint8)  # min-dist / void-loses / tie->pass0 / all-void
    ok_merge = bool((merged_map == want).all())
    print(f"merge rule ok: {ok_merge}  (got {merged_map.tolist()}, want {want.tolist()})")

    sys.exit(0 if (ok and ok_dist and ok_merge and ok_modes) else 1)
