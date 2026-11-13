"""Shared instance-segmentation generation prompts (legend scheme).

Single source of truth for the (image, noun-phrase) -> "paint each instance a
distinct color on black" prompt, imported by BOTH the training-data converter
(dataset_prep/instanceseg/dataset_prep.py) and the SA-Co/Gold evaluator
(evaluation/instanceseg/inference_flux2_klein_instanceseg.py), so the wording the
model is trained on and the wording it is prompted with at eval time can never
drift apart -- the same design as common/semseg/diversify.py for semseg.

Legend scheme -- build_instance_prompt():
  * base template (BASE_TEMPLATES, natural caption styles with the
    [TASK: INSTANCE_SEGMENTATION] tag and a BLACK background so the decoder's
    C_bg=(0,0,0) stays valid),
  * + ORDER_NUMBERING: instances are numbered by raster order of their mask
    anchors (common/instanceseg/ordering.py),
  * + an explicit per-instance color assignment list ("The 1st {np} is blue, the
    2nd is green, ..."), rendered in ONE notation style per prompt: "name"
    (everyday color names) or "rgb" ("[30, 60, 220]"). Both notations describe
    the SAME fixed colors -- the ordered named palette
    (common/instanceseg/instance_palette_named.json, 36 entries); notation only
    changes how a color is written, never which RGBs are painted. Only when an
    image has more instances than the named palette holds does the 64-color grid
    palette (instance_palette.json) take over, with rgb notation forced and the
    list truncated to fit the text-encoder budget.
    The list states assignments, not a count: at eval time (instance count
    unknown) a constant-length prefix of the same convention is emitted and the
    model paints however many instances it finds.

Experimental fixed-K draw schemes (DRAW_MODES / build_draw_prompt): fixed vs.
randomized palette assignment, with either an explicit K-entry legend or an
unlisted prompt contract, plus scan-direction and notation variants -- see the
DRAW_MODES comment block for the design.

Kept for evaluating older checkpoints (the evaluator's --prompt_set flag):
  * ORDERED_PROMPT_TEMPLATES -- the fixed-palette-order wording (no explicit
    colors). PROMPT_TEMPLATES aliases it.
  * LEGACY_PROMPT_TEMPLATES -- the pre-convention wording.
"""
from __future__ import annotations

import hashlib
import random
from typing import List, Optional, Sequence, Tuple

from common.instanceseg.colors import (
    load_instance_palette,
    load_named_instance_palette,
    named_separation_order,
)

NOTATION_STYLES = ("name", "rgb")

# --- fixed-K draw schemes ----------------------------------------------------- #
# Every mode uses the SAME K-entry color pool; the arms differ in the
# instance->color assignment, prompt contract, and instance-numbering scan:
#   fixeddraw_K   pool colors always in their max-separated (FPS) order -- the
#                 instance-seg analog of semseg's best{32,64}_maxnegdraw.
#   randomdraw_K  same pool, but a fresh (seeded) permutation per sample; the
#                 legend states the shuffled assignment, so the target stays a
#                 pure function of (image, prompt): the mapping is random but stated.
#   randomdraw_32_unlisted  exactly the same randomized targets as
#                 randomdraw_32, but the prompt only asks for a different/unique
#                 color per instance. It states neither the scan convention nor
#                 any instance->color assignments.
#   fixeddraw_32_ordered_unlisted  exactly the same targets as fixeddraw_32, but
#                 uses the fixed-palette-order wording: it states the
#                 top-to-bottom scan and i-th-instance -> i-th-palette-color
#                 convention without enumerating the 32 colors.
#   fixeddraw_32_lr  exactly fixeddraw_32, but instances are numbered by a
#                 left-to-right scan (ties top-to-bottom) instead of the default
#                 top-to-bottom scan (ties left-to-right); the numbering sentence
#                 in the prompt states the lr rule. Isolates scan-direction
#                 sensitivity with everything else held fixed.
#   fixeddraw_32_mixedrepresent  exactly fixeddraw_32 targets, but the NOTATION
#                 varies per sample (50/50, seeded/hash-stable): half the prompts
#                 write the colors as names, half as hex values. Painted pixels
#                 are identical either way -- notation only changes the wording.
#                 Both notations enumerate all 32 entries whenever the prompt
#                 fits the 512-token budget (the cascade below falls back through
#                 shorter templates); only the ultra-long-NP tail at eval
#                 (~0.16% of SA-Co pairs) is truncated.
# Pools are enumerated in PURE most->least-separated (farthest-point) order:
# the grid palette's committed order is already that; the named palette's
# blue/green/red pinning is bypassed via colors.named_separation_order().
# K=32 uses the named palette in name notation (~359 worst-case tokens, fits
# the Flux2-Klein 512 budget). K=64 must enumerate the 64-color grid palette;
# only hex notation fits a full 64-entry legend (~928 tokens), which exceeds
# Klein's budget -- those modes need a longer text window.
# Fixed-K prompts are emitted for negatives too (all-black target): at eval the
# instance count is unknown, so every query carries the mode's constant prompt
# contract -- the training distribution must include "prompt + absent class ->
# paint nothing".
DRAW_MODES: dict[str, dict] = {
    "fixeddraw_32": {"k": 32, "palette": "named", "style": "name", "randomize": False, "scan": "tb", "prompt_contract": "explicit"},
    "randomdraw_32": {"k": 32, "palette": "named", "style": "name", "randomize": True, "scan": "tb", "prompt_contract": "explicit"},
    "randomdraw_32_unlisted": {"k": 32, "palette": "named", "style": "name", "randomize": True, "scan": "tb", "prompt_contract": "unlisted"},
    "fixeddraw_32_ordered_unlisted": {"k": 32, "palette": "named", "style": "name", "randomize": False, "scan": "tb", "prompt_contract": "ordered_unlisted"},
    "fixeddraw_64": {"k": 64, "palette": "grid", "style": "hex", "randomize": False, "scan": "tb", "prompt_contract": "explicit"},
    "randomdraw_64": {"k": 64, "palette": "grid", "style": "hex", "randomize": True, "scan": "tb", "prompt_contract": "explicit"},
    "fixeddraw_32_lr": {"k": 32, "palette": "named", "style": "name", "randomize": False, "scan": "lr", "prompt_contract": "explicit"},
    # fixeddraw_32_lr's left-to-right numbering with the full 64-colour grid
    # legend in hex (~900-940 tokens); needs a raised text window.
    "fixeddraw_64_lr": {"k": 64, "palette": "grid", "style": "hex", "randomize": False, "scan": "lr", "prompt_contract": "explicit"},
    "fixeddraw_32_mixedrepresent": {"k": 32, "palette": "named", "style": "mixed", "randomize": False, "scan": "tb", "prompt_contract": "explicit"},
}

# --- mixed-mode budget cascade ------------------------------------------------ #
# fixeddraw_32_mixedrepresent ALWAYS enumerates all 32 entries; only the base
# template gives way when the prompt would exceed the 512-token text-encoder
# budget (drawn template -> remaining templates shortest-first). Whether a
# candidate "fits" is decided by a pluggable predicate:
#   * The training converter uses the character budgets below, calibrated so
#     every source-taxonomy noun phrase fits the full-32 legend within 512 Qwen
#     tokens; if no template fits, overflow="raise" fails loud.
#   * The evaluator passes an exact token-count predicate from the pipeline
#     tokenizer plus overflow="accept": for the ultra-long tail (~0.16% of SA-Co
#     pairs) the full-32 prompt on the shortest template is sent and truncated.
HEX_FULL_PROMPT_CHAR_BUDGET = 1045
NAME_FULL_PROMPT_CHAR_BUDGET = 1150

# Exact continuation wording shared with build_legend_suffix -- appended when a
# legend enumerates fewer entries than the convention covers. Fixed-order only:
# a shuffled (randomdraw) legend cannot be continued and always enumerates fully.
CONTINUATION_CLAUSE = ", and later instances continue in the same fixed palette order"

# Base caption phrasings; every final prompt starts from one of these. The
# "masks-natural" style is listed twice (~50% weight); edit/reweight freely.
# Each must contain "{np}".
BASE_TEMPLATES: list[str] = [
    # masks-natural, detailed (primary)
    "[TASK: INSTANCE_SEGMENTATION] This image shows segmentation masks for {np} from the input image. The "
    "background is set to black color [0, 0, 0]. Each {np} instance is represented "
    "by a solid mask, and a different color is used for each mask.",
    # masks-natural, short (primary)
    "[TASK: INSTANCE_SEGMENTATION] This image shows segmentation masks for {np} from the input image. The "
    "background is set to black color [0, 0, 0]. Each {np} is represented by a "
    "different color.",
    # paper verbatim instruction
    '[TASK: INSTANCE_SEGMENTATION] Generate an instance segmentation visualization of this image. Each "{np}" is '
    "colored differently. Leave everything else black [0, 0, 0].",
    # segmentation-task style
    '[TASK: INSTANCE_SEGMENTATION] This image is a segmentation task derived from the input. The "{np}" instances '
    "are each represented by a unique, solid color on a black [0, 0, 0] background.",
]

# Defines what "the i-th instance" means; precedes the assignment list (and is all
# a negative prompt carries -- no list when no instances are present).
ORDER_NUMBERING = (
    " Instances are numbered by where their mask first appears in a top-to-bottom "
    "scan of the image (ties broken left to right)."
)

# The fixeddraw_32_lr numbering rule: column-major scan. Same wording shape as
# ORDER_NUMBERING so the only difference the model sees is the scan direction.
ORDER_NUMBERING_LR = (
    " Instances are numbered by where their mask first appears in a left-to-right "
    "scan of the image (ties broken top to bottom)."
)

# --- pre-legend wording, kept verbatim for --prompt_set ordered ---------------- #
ORDER_CONVENTION = (
    " Masks are colored in a fixed palette order: instances are numbered by "
    "where their mask first appears in a top-to-bottom scan of the image "
    "(ties broken left to right), and the i-th instance always gets the i-th "
    "palette color."
)

# Pre-convention wording (--prompt_set legacy).
LEGACY_PROMPT_TEMPLATES: list[str] = list(BASE_TEMPLATES)

ORDERED_PROMPT_TEMPLATES: list[str] = [t + ORDER_CONVENTION for t in LEGACY_PROMPT_TEMPLATES]

# Deprecated alias for ORDERED_PROMPT_TEMPLATES.
PROMPT_TEMPLATES: list[str] = ORDERED_PROMPT_TEMPLATES


def draw_prompt_templates(mode: str,
                          templates: Sequence[str] = BASE_TEMPLATES) -> List[str]:
    """Return the static prompt templates actually used by a draw mode.

    Explicit modes add their dynamic assignment list in ``build_draw_prompt``,
    so their static templates remain the base captions. The ordered-unlisted
    mode has a static convention suffix and returns it here as well as using it
    in the builder. This helper keeps evaluation provenance synchronized with
    the training prompt contract.
    """
    contract = DRAW_MODES[mode]["prompt_contract"]
    if contract == "ordered_unlisted":
        return [t + ORDER_CONVENTION for t in templates]
    if contract in ("explicit", "unlisted"):
        return list(templates)
    raise ValueError(f"{mode}: unknown prompt contract {contract!r}")


def _ordinal(i: int) -> str:
    """1 -> '1st', 2 -> '2nd', 11 -> '11th', 23 -> '23rd'."""
    if 10 <= i % 100 <= 13:
        return f"{i}th"
    return f"{i}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(i % 10, 'th') }"


def _color_token(rgb: Sequence[int], style: str, name: str | None) -> str:
    """One color as its prompt token: 'blue' (name), '[30, 60, 220]' (rgb), '#1e3cdc' (hex)."""
    if style == "name":
        if name is None:
            raise ValueError("name notation requires a named palette entry")
        return name
    if style == "rgb":
        r, g, b = (int(v) for v in rgb)
        return f"[{r}, {g}, {b}]"
    if style == "hex":
        r, g, b = (int(v) for v in rgb)
        return f"#{r:02x}{g:02x}{b:02x}"
    raise ValueError(f"unknown notation style: {style}")


# How many assignments are spelled out per notation before the continuation
# clause takes over. Sized so the WORST-case prompt (longest template, long noun
# phrase -- the NP appears up to 3x) stays under the 512-token text-encoder
# budget with headroom: an rgb clause ("the 15th is [245, 130, 30],") costs ~20
# tokens vs ~8 for a name clause, so rgb enumerates fewer. Counts beyond these
# caps are ~1% of rows.
LEGEND_ENUM_CAP = {"name": 36, "rgb": 16}


def build_legend_suffix(k: int, style: str, np_text: str, open_ended: bool = True
                        ) -> Tuple[str, List[Tuple[int, int, int]]]:
    """The per-instance color assignment sentence + the exact RGBs to paint.

    Returns ("", []) for k <= 0 (negatives). For 0 < k <= 36 the colors are the
    named-palette prefix -- the SAME RGBs regardless of notation, which only
    changes how they are written. For k > 36 the 64-color grid palette takes over
    (its colors have no names, so rgb notation is forced). At most
    LEGEND_ENUM_CAP[style] assignments are spelled out (text-encoder budget); a
    continuation clause covers any instances beyond that.

    ``open_ended`` (default True, the EVAL behavior): always append the
    continuation clause, so a fixed-length eval legend reads as a statement of
    the convention rather than an instance count -- the model paints however
    many instances it finds. Same clause wording the truncated training rows
    carry, so it is not novel vocabulary at eval time. The TRAINING converter is
    the one caller that opts out (open_ended=False): its legends are exact-count
    by design, with the clause only on truncated rows.
    Single source of truth: callers must paint exactly the returned colors, so
    prompt text and target pixels cannot drift.
    """
    if k <= 0:
        return "", []
    named = load_named_instance_palette()
    if k <= len(named):
        colors = [e["rgb"] for e in named[:k]]
        names = [e["name"] for e in named[:k]]
    else:
        if style != "rgb":
            raise ValueError(f"k={k} exceeds the named palette ({len(named)}); rgb notation required")
        palette = load_instance_palette()
        colors = [tuple(c) for c in palette[:k]]
        names = [None] * k
    enum_n = min(k, LEGEND_ENUM_CAP[style])
    tokens = [_color_token(colors[i], style, names[i]) for i in range(enum_n)]
    tail = CONTINUATION_CLAUSE if (k > enum_n or open_ended) else ""
    clauses = [f"The {_ordinal(1)} {np_text} is {tokens[0]}"]
    clauses += [f"the {_ordinal(i + 1)} is {tok}" for i, tok in enumerate(tokens) if i > 0]
    return " " + ", ".join(clauses) + tail + ".", colors


def build_instance_prompt(np_text: str, k: int, style: str, template: str,
                          open_ended: bool = True
                          ) -> Tuple[str, List[Tuple[int, int, int]]]:
    """Full prompt (template + numbering rule + assignment list) and paint colors.

    ``k`` is the number of instances the target paints (0 for negatives -- then no
    assignment list and no colors). ``colors[rank]`` is the RGB of canonical
    instance #rank; the caller paints exactly these. ``open_ended`` defaults to
    the eval behavior; training passes False (see build_legend_suffix).
    """
    suffix, colors = build_legend_suffix(k, style, np_text, open_ended=open_ended)
    return template.format(np=np_text) + ORDER_NUMBERING + suffix, colors


def _draw_pool(mode: str) -> Tuple[List[Tuple[int, int, int]], List[Optional[str]]]:
    """The mode's color pool in pure most->least-separated order, as (colors, names).

    Grid palette: the committed order is already the pure FPS order. Named
    palette: the committed file pins
    blue/green/red first, so the draw modes re-derive the unpinned separation
    order via colors.named_separation_order().
    """
    cfg = DRAW_MODES[mode]
    k = cfg["k"]
    if cfg["palette"] == "named":
        named = load_named_instance_palette()
        if k > len(named):
            raise ValueError(f"{mode}: k={k} exceeds the named palette ({len(named)})")
        sel = [named[i] for i in named_separation_order()[:k]]
        return [e["rgb"] for e in sel], [e["name"] for e in sel]
    return [tuple(c) for c in load_instance_palette()[:k]], [None] * k


def _render_draw_prompt(np_text: str, template: str, numbering: str,
                        tokens: Sequence[str], truncated: bool) -> str:
    clauses = [f"The {_ordinal(1)} {np_text} is {tokens[0]}"]
    clauses += [f"the {_ordinal(i + 1)} is {tok}" for i, tok in enumerate(tokens) if i > 0]
    tail = CONTINUATION_CLAUSE if truncated else ""
    return template.format(np=np_text) + numbering + " " + ", ".join(clauses) + tail + "."


def default_prompt_fits(prompt: str, style: str) -> bool:
    """Tokenizer-free budget predicate: the taxonomy-calibrated char budgets."""
    budget = HEX_FULL_PROMPT_CHAR_BUDGET if style == "hex" else NAME_FULL_PROMPT_CHAR_BUDGET
    return len(prompt) <= budget


def build_draw_prompt(mode: str, np_text: str, template: str,
                      permutation: Optional[Sequence[int]] = None,
                      notation: Optional[str] = None,
                      fits=None,
                      overflow: str = "raise",
                      randomize_palette: bool = False,
                      ) -> Tuple[str, List[Tuple[int, int, int]]]:
    """Prompt + paint colors for a fixed-K draw mode (see DRAW_MODES).

    ``colors[rank]`` is the RGB of canonical instance #rank UNDER THE MODE'S
    SCAN DIRECTION (order instances with canonical_instance_order(masks,
    scan=DRAW_MODES[mode]["scan"])) and callers paint at most K instances.
    ``permutation`` is required for randomdraw_* (a permutation of range(K))
    and forbidden for the fixed-order modes, except when
    ``randomize_palette=True``. That explicit evaluation-only override preserves
    a fixed mode's scan direction, prompt contract, and notation while sampling
    a new stated instance->color mapping for a repeat evaluation. ``notation``
    is required for the style="mixed" mode ("name" or "hex", the per-sample
    draw) and forbidden elsewhere.

    Explicit contracts ALWAYS enumerate all K entries. The unlisted contract
    emits only the chosen base template; ordered_unlisted appends the fixed-order
    scan convention but no color list. The style="mixed" explicit mode
    additionally runs the template cascade (see the comment above
    HEX_FULL_PROMPT_CHAR_BUDGET): if the render fails the ``fits`` predicate
    (default: the char budgets; the evaluator passes an exact token counter),
    it deterministically tries the other templates shortest-first. When even
    the shortest template does not fit: overflow="raise" (default, the
    converter -- a truncated training row would corrupt supervision) fails
    loud; overflow="accept" (the evaluator) returns the shortest-template
    full-K prompt anyway, knowingly letting the text encoder truncate its tail.
    """
    cfg = DRAW_MODES[mode]
    k = cfg["k"]
    style = cfg["style"]
    if style == "mixed":
        if notation not in ("name", "hex"):
            raise ValueError(f"{mode} requires notation='name' or 'hex'")
        style = notation
    elif notation is not None:
        raise ValueError(f"{mode} has fixed notation; notation param not allowed")
    uses_random_permutation = cfg["randomize"] or randomize_palette
    if uses_random_permutation:
        if permutation is None or sorted(permutation) != list(range(k)):
            raise ValueError(f"{mode} requires a permutation of range({k})")
        order = list(permutation)
    else:
        if permutation is not None:
            raise ValueError(f"{mode} is fixed-order; permutation not allowed")
        order = list(range(k))
    pool_colors, pool_names = _draw_pool(mode)
    colors = [pool_colors[i] for i in order]
    names = [pool_names[i] for i in order]
    contract = cfg["prompt_contract"]
    if contract == "unlisted":
        return template.format(np=np_text), colors
    if contract == "ordered_unlisted":
        return draw_prompt_templates(mode, [template])[0].format(np=np_text), colors
    if contract != "explicit":
        raise ValueError(f"{mode}: unknown prompt contract {contract!r}")

    numbering = ORDER_NUMBERING_LR if cfg["scan"] == "lr" else ORDER_NUMBERING
    all_tokens = [_color_token(colors[i], style, names[i]) for i in range(k)]

    if cfg["style"] != "mixed":
        return _render_draw_prompt(np_text, template, numbering, all_tokens, False), colors

    # Template cascade: the full enumeration is non-negotiable; only the
    # template gives way (drawn first, then the others shortest-first).
    check = (lambda p: fits(p)) if fits is not None else (lambda p: default_prompt_fits(p, style))
    fallback_templates = [template] + [t for t in sorted(BASE_TEMPLATES, key=len) if t != template]
    for tmpl in fallback_templates:
        prompt = _render_draw_prompt(np_text, tmpl, numbering, all_tokens, False)
        if check(prompt):
            return prompt, colors
    if overflow == "accept":
        shortest = min(BASE_TEMPLATES, key=len)
        return _render_draw_prompt(np_text, shortest, numbering, all_tokens, False), colors
    raise ValueError(
        f"{mode}: full-{k} legend fits no template for noun phrase {np_text!r} "
        f"({len(np_text)} chars); pass overflow='accept' to send it anyway "
        "(text-encoder truncation).")


def select_draw_plan(subset: str, image_id, np_text: str, seed: int, mode: str,
                     repeat_index: int = 0,
                     randomize_palette: bool = False,
                     ) -> Tuple[str, Optional[List[int]], Optional[str]]:
    """Deterministic (template, permutation, notation) for a pair at eval time.

    Shard-stable like select_prompt_plan: hashed from stable fields, so identical
    for any --num_shards. ``repeat_index=0`` is the single-pass plan.
    For a random draw mode (or with ``randomize_palette=True``), later repeat
    indices rotate that same random permutation. This guarantees a different
    instance->palette assignment while holding the prompt template and notation
    fixed, so repeat evaluation isolates palette assignment rather than changing
    other prompt variables. permutation is None for fixed-order modes unless the
    explicit evaluation override is enabled; notation is None except for style="mixed"
    (then a 50/50 name/hex pick, mirroring the training converter's per-sample
    draw).
    """
    cfg = DRAW_MODES[mode]
    h = _stable_hash(subset, image_id, np_text, seed)
    template = BASE_TEMPLATES[h % len(BASE_TEMPLATES)]
    notation = None
    if cfg["style"] == "mixed":
        notation = ("name", "hex")[(h // len(BASE_TEMPLATES)) % 2]
    if not cfg["randomize"] and not randomize_palette:
        return template, None, notation
    perm = list(range(cfg["k"]))
    random.Random(h // len(BASE_TEMPLATES)).shuffle(perm)
    if repeat_index < 0:
        raise ValueError(f"repeat_index must be non-negative, got {repeat_index}")
    if repeat_index:
        # A cyclic shift keeps a true permutation and cannot equal repeat 0 for
        # 0 < repeat_index < K. The evaluator caps repeats at K accordingly.
        shift = repeat_index % cfg["k"]
        perm = perm[shift:] + perm[:shift]
    return template, perm, notation


def validate_templates(templates: Sequence[str] = BASE_TEMPLATES) -> None:
    """Raise if any template is missing the required "{np}" placeholder."""
    for t in templates:
        if "{np}" not in t:
            raise ValueError(f"prompt template missing '{{np}}': {t!r}")


def _stable_hash(subset: str, image_id, np_text: str, seed: int) -> int:
    key = f"{seed}:{subset}:{image_id}:{np_text}".encode()
    return int(hashlib.md5(key).hexdigest(), 16)


def select_prompt_template(subset: str, image_id, np_text: str, seed: int,
                           templates: Sequence[str] = ORDERED_PROMPT_TEMPLATES) -> str:
    """Pick a template for a (image, NP) pair deterministically.

    Hashes stable fields (not a running RNG), so the choice is identical regardless
    of how samples are sharded across GPUs -- the eval reproduces for any
    --num_shards. Same seed + same pair -> same template. Used by the evaluator for
    the ordered/legacy prompt sets; the training converter samples templates with
    its own RNG for variety.
    """
    return templates[_stable_hash(subset, image_id, np_text, seed) % len(templates)]


def select_prompt_plan(subset: str, image_id, np_text: str, seed: int
                       ) -> Tuple[str, str]:
    """Deterministic (base_template, notation_style) for a (image, NP) pair.

    Same shard-stable hashing as select_prompt_template; the style bits are taken
    above the template bits so the two choices are independent. Mirrors the
    training distribution (uniform over BASE_TEMPLATES x NOTATION_STYLES).
    """
    h = _stable_hash(subset, image_id, np_text, seed)
    template = BASE_TEMPLATES[h % len(BASE_TEMPLATES)]
    style = NOTATION_STYLES[(h // len(BASE_TEMPLATES)) % len(NOTATION_STYLES)]
    return template, style
