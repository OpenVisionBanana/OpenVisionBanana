# ===============================================================================
# Flux2-Klein SA-Co/Gold instance-segmentation inference + CGF1 evaluation.
# ===============================================================================
"""Probe the base or partial-fine-tuned Flux2-Klein generator on SA-Co/Gold
open-vocabulary instance segmentation, scored with SAM3's CGF1 metric.

Pipeline (Vision Banana paper arXiv:2604.20329v3, Table 8 + Appendix A):
  1. Build (image, noun-phrase) query samples from each subset's primary GT.
  2. Gate each query for image-level presence (Qwen3-VL via a cached
     gate_decisions.json, or --gate oracle/none). Negatives emit NO prediction --
     this drives the image-level IL_MCC term of CGF1.
  3. For positive queries, prompt Flux2-Klein img2img to paint each instance of
     the noun-phrase a distinct color on black (paper template, overridable).
  4. Snap to the prompted palette, resize to GT size, decode, fill painted gaps,
     then refine with GrabCut. The winning settings are defaults; each step has
     an independent --no-* switch. Refinement precedes repeat consensus.
  5. Encode masks as COCO segm results (constant score, the Gemini/Qwen gate having
     already decided presence) and dump <out_dir>/<subset>/coco_predictions_segm.json.
  6. Optionally (--run_eval) score every subset with the (vendored) CGF1Evaluator
     over its 3 oracle GTs and report per-subset + aggregate cgF1/IL_MCC/pmF1.

Sharding mirrors the refseg/semseg evaluators: one process per shard handles
samples[rank::world]; rank 0 concatenates the shard predictions and (optionally)
runs the evaluator.

Typical usage (single GPU, one subset, oracle gate, quick check):
    CUDA_VISIBLE_DEVICES=0 python evaluation/instanceseg/inference_flux2_klein_instanceseg.py \
        --transformer_path /path/to/checkpoint/transformer \
        --out_dir outputs/sacogold --subsets sa1b_nps --limit 8 --gate oracle \
        --run_eval --save_panels
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))
# Reuse the refseg evaluator's sharding / model-load / panel helpers.
sys.path.insert(0, str(REPO_ROOT / "evaluation/refseg"))
import inference_flux2_klein_refseg as refseg  # noqa: E402
from decode import CONSENSUS_MODES, DEFAULT_CONSENSUS_MODE, consensus_instance_masks, decode_instances, masks_to_coco  # noqa: E402
# Prompts are shared with the training-data converter so train/eval can't drift.
from common.instanceseg.prompts import (  # noqa: E402
    BASE_TEMPLATES,
    DRAW_MODES,
    LEGACY_PROMPT_TEMPLATES,
    ORDERED_PROMPT_TEMPLATES,
    build_draw_prompt,
    build_instance_prompt,
    draw_prompt_templates,
    select_draw_plan,
    select_prompt_plan,
    select_prompt_template,
    validate_templates,
)
from common.instanceseg.colors import (  # noqa: E402
    load_named_instance_palette,
    ordered_instance_colors,
)
from common.instanceseg.ordering import canonical_instance_order, select_instances  # noqa: E402
from eval_saco_gold import (  # noqa: E402
    DEFAULT_GT_FOLDER,
    GT_FILE_TMPL,
    GT_PARTS,
    SACO_GOLD_SUBSETS,
    run_eval,
)

from evaluation.semseg.inference_partial_flux2_klein_img2img import (  # noqa: E402
    PipelineBatchRequest,
    run_pipeline_batch,
)
from evaluation.instanceseg.repeat_analysis import (  # noqa: E402
    AGREEMENT_DEFINITION,
    COLOR_SCALES,
    analyze_repeat_outputs,
    lock_aggregation,
    repeat_protocol,
    validate_repeat_shard,
    write_json_atomic,
)
from evaluation.instanceseg.raw_recovery import (  # noqa: E402
    lock_raw_shard, raw_prediction_path, raw_protocol, read_saved_rgb,
    verify_raw_protocol,
)
from evaluation.instanceseg.artifact_writer import ArtifactWriter  # noqa: E402
from evaluation.instanceseg.postprocess import (  # noqa: E402
    add_postprocess_arguments, decode_and_refine, postprocess_protocol,
    reproducibility_manifest, validate_postprocess_arguments,
)

SACO_ROOT_DEFAULT = REPO_ROOT / "datasets_archives/instanceseg/SA-Co-Gold"

# Generation prompts live in common/instanceseg/prompts.py, shared with the
# training-data converter so the wording the model is trained on and prompted with
# at eval time stay identical. --prompt_set picks the scheme generation a checkpoint
# was trained on:
#   legend  template + numbering rule + explicit color-assignment list
#           ("The 1st {np} is blue, the 2nd is green, ..."). The count is unknown at
#           eval, so a CONSTANT --legend_len-entry prefix of the fixed convention is
#           emitted for every pair; the model paints as many instances as it finds.
#           --notation mix hash-picks name/rgb per pair (training distribution).
#   fixeddraw_32/64, randomdraw_32/64, randomdraw_32_unlisted,
#   fixeddraw_32_ordered_unlisted, fixeddraw_32_lr,
#   fixeddraw_32_mixedrepresent
#           the fixed-K draw schemes (common/instanceseg/prompts.py DRAW_MODES):
#           fixed pool order vs. per-pair hash-stable permutation. Most carry a
#           constant K-entry legend; _unlisted variants intentionally omit the
#           mapping (the ordered one retains the fixed-order scan convention).
#           The _lr variant numbers instances by a left-to-right scan (ties
#           top-to-bottom); _mixedrepresent hash-picks name/hex notation per
#           pair. Notation and K are fixed by the mode; --legend_len/--notation
#           are ignored.
#   ordered fixed-palette-order wording, no explicit colors.
#   legacy  pre-convention wording.
# Template (and notation/permutation) choice is hash-per-pair (shard-stable); pass
# --prompt_template to force a single prompt instead.


# --------------------------------------------------------------------------- #
# Samples
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IsegSample:
    subset: str
    image_id: int
    image_path: Path
    np_text: str       # the queried noun-phrase
    gt_h: int
    gt_w: int
    gt_present: bool    # only meaningful for --gate oracle (NP annotated in any GT)


@dataclass(frozen=True)
class PendingIsegGeneration:
    """One positive query waiting for a compatible FLUX batch."""

    sample: IsegSample
    image: Image.Image
    prompt: str
    prompt_colors: Optional[Sequence[Sequence[int]]]
    prompt_scan: str
    prompt_style: Optional[str]
    legend_template: Optional[str]
    evaluated_index: int
    generator: object | None
    height: int | None
    width: int | None
    repeat_index: int = 0


def resolve_image_path(file_name: str, saco_root: Path) -> Path:
    """sa1b file_names start with 'sa_' (-> sa1b-images/), else metaclip-images/."""
    sub = "sa1b-images" if file_name.startswith("sa_") else "metaclip-images"
    return saco_root / sub / file_name


def instance_counts(gt_path: Path) -> Counter:
    """Per-(image,NP) GT instance count for one annotator file."""
    data = json.loads(gt_path.read_text())
    return Counter(a["image_id"] for a in data.get("annotations", []))


def build_subset_samples(
    subset: str,
    gt_folder: Path,
    saco_root: Path,
    gate_mode: str,
    limit: Optional[int],
    sample_seed: int,
    max_instances: Optional[int] = None,
) -> list[IsegSample]:
    """Samples from a subset's primary GT (a), restricted to instance-exhaustive
    queries (eval_img_ids are a subset of these).

    ``max_instances`` keeps only queries with at most N instances; the count is the
    **max over the 3 oracle annotators** (a/b/c), so the bound holds whichever GT
    the oracle eval ends up scoring. Negatives (0 instances) always pass. Default
    None = no limit. For --gate oracle, presence is count>0 (same union as before).
    """
    stem = SACO_GOLD_SUBSETS[subset]
    data = json.loads((gt_folder / GT_FILE_TMPL.format(stem=stem, part="a")).read_text())

    # Compute per-id instance counts (max over a/b/c) only when needed -- for the
    # oracle presence flag or the --max_instances filter.
    max_count: dict[int, int] = {}
    if gate_mode == "oracle" or max_instances is not None:
        for part in GT_PARTS:
            for iid, n in instance_counts(gt_folder / GT_FILE_TMPL.format(stem=stem, part=part)).items():
                max_count[iid] = max(max_count.get(iid, 0), n)

    samples = []
    for im in data["images"]:
        if not im.get("is_instance_exhaustive", 1):
            continue
        cnt = max_count.get(im["id"], 0)
        if max_instances is not None and cnt > max_instances:
            continue
        samples.append(IsegSample(
            subset=subset,
            image_id=im["id"],
            image_path=resolve_image_path(im["file_name"], saco_root),
            np_text=im["text_input"],
            gt_h=int(im["height"]),
            gt_w=int(im["width"]),
            gt_present=cnt > 0,
        ))
    samples.sort(key=lambda s: s.image_id)
    if limit is not None and limit < len(samples):
        samples = sorted(random.Random(f"{sample_seed}:{subset}").sample(samples, limit),
                         key=lambda s: s.image_id)
    return samples


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #
def load_gate_file(gate_file: Optional[Path]) -> dict:
    if gate_file is None:
        return {}
    p = Path(gate_file)
    if not p.exists():
        # Missing gate file: warn and run ungated.
        print(f"WARNING: gate file {p} not found; treating all queries as positive (ungated).")
        return {}
    # accept either {subset: {image_id: bool}} or a flat {image_id: bool}
    return json.loads(p.read_text())


def is_positive(sample: IsegSample, gate_mode: str, gate: dict) -> bool:
    if gate_mode == "none":
        return True
    if gate_mode == "oracle":
        return sample.gt_present
    # gate_mode == "file": missing entries default to positive (don't suppress).
    table = gate.get(sample.subset, gate)
    val = table.get(str(sample.image_id), table.get(sample.image_id))
    return True if val is None else bool(val)


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def masks_to_prompt_target(
    masks: Sequence[np.ndarray],
    h: int,
    w: int,
    colors: Sequence[Sequence[int]],
    scan: str = "tb",
) -> tuple[Image.Image, int]:
    """Paint GT masks with the training target's scan-order color contract.

    SA-Co/Gold provides uncolored binary RLEs. For a meaningful qualitative
    comparison, panels recreate the RGB target convention used in training:
    retain at most the palette capacity (largest masks first), rank retained
    masks by their canonical scan anchor, and paint rank ``i`` with ``colors[i]``.

    Some SA-Co/Gold RLEs disagree with the image JSON dimensions. Normalize to
    the panel canvas before selection and ordering so the displayed target has
    the same geometry as the decoded prediction.
    """
    normalized = []
    for m in masks:
        if m.shape[:2] != (h, w):
            m = np.asarray(
                Image.fromarray(m.astype(np.uint8)).resize((w, h), Image.NEAREST),
                dtype=bool,
            )
        # Empty masks (or masks lost in canvas normalization) have no scan anchor;
        # exclude them from the panel only.
        if m.any():
            normalized.append(m)

    # Same deterministic cap as dataset preparation (only matters beyond the pool size).
    kept = select_instances(normalized, len(colors))
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for rank, idx in enumerate(canonical_instance_order(kept, scan=scan)):
        vis[kept[idx]] = colors[rank]
    return Image.fromarray(vis), len(kept)


def _decode_rle(seg, h: int, w: int) -> np.ndarray:
    """Decode a COCO GT segmentation (compressed RLE / uncompressed / polygon) to bool."""
    import pycocotools.mask as mu
    if isinstance(seg, list):                      # polygon
        rle = mu.merge(mu.frPyObjects(seg, h, w))
    elif isinstance(seg.get("counts"), list):      # uncompressed RLE
        rle = mu.frPyObjects(seg, h, w)
    else:                                          # compressed RLE (SA-Co/Gold)
        counts = seg["counts"]
        rle = {"size": seg["size"],
               "counts": counts.encode("ascii") if isinstance(counts, str) else counts}
    return mu.decode(rle).astype(bool)


def load_gt_masks(subset: str, gt_folder: Path, image_ids) -> dict:
    """{image_id: [segmentation, ...]} from a subset's primary GT (a) for the given ids
    (the GT instance masks of that (image, NP) pair, used to draw the panel's GT cell)."""
    stem = SACO_GOLD_SUBSETS[subset]
    data = json.loads((gt_folder / GT_FILE_TMPL.format(stem=stem, part="a")).read_text())
    want = set(image_ids)
    out: dict = defaultdict(list)
    for a in data.get("annotations", []):
        if a["image_id"] in want:
            out[a["image_id"]].append(a["segmentation"])
    return out


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="black-forest-labs/FLUX.2-klein-base-9B")
    p.add_argument("--transformer_path", default=None,
                   help="Trained output / checkpoint / transformer dir; omit for base model.")
    p.add_argument("--lora_path", default=None,
                   help="Standard Diffusers LoRA directory (holds pytorch_lora_weights.safetensors) applied on top of --model.")
    p.add_argument("--out_dir", required=True)

    p.add_argument("--saco_root", type=Path, default=SACO_ROOT_DEFAULT,
                   help="SA-Co-Gold root (holds sa1b-images/, metaclip-images/, gt-annotations/).")
    p.add_argument("--gt_folder", type=Path, default=None,
                   help="GT json dir (default: <saco_root>/gt-annotations).")
    p.add_argument("--subsets", nargs="+", default=None,
                   choices=sorted(SACO_GOLD_SUBSETS.keys()),
                   help="Subset(s) to run (default: all 7).")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap queries PER SUBSET (seeded random sample; default: full set).")
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--max_instances", type=int, default=None,
                   help="Process only (image, NP) pairs with at most N GT instances "
                        "(max over the 3 oracle annotators; negatives always pass). "
                        "Default: no limit (all pairs).")

    # gate
    p.add_argument("--gate", choices=("file", "oracle", "none"), default="file",
                   help="Image-level presence gate. file: read --gate_file; oracle: "
                        "GT presence (debug upper bound); none: generate for all.")
    p.add_argument("--gate_file", type=Path,
                   default=Path(__file__).resolve().parents[1] / "assets" / "saco_gold_gate_decisions_qwen3vl8b.json",
                   help="gate_decisions.json from gate_saco_gold_qwen.py (for --gate file). "
                        "Defaults to the bundled Qwen3-VL-8B decisions used for the paper.")
    p.add_argument(
        "--require_complete_gate",
        action="store_true",
        help="Fail before loading the diffusion model when --gate file is missing "
             "a decision for any selected query. Recommended for headline runs.",
    )

    # generation / decode
    p.add_argument("--prompt_set", default="fixeddraw_32_lr",
                   choices=("legend", "ordered", "legacy", *DRAW_MODES),
                   help="Prompt scheme generation the checkpoint was trained on: fixeddraw_32_lr "
                        "(default), legend "
                        "(explicit per-instance color assignments, current), ordered (v1 "
                        "fixed-palette-order wording, isegordpal run), legacy (pre-fix), "
                        "or a fixed-K draw mode (fixeddraw_32/64, randomdraw_32/64, "
                        "randomdraw_32_unlisted, fixeddraw_32_ordered_unlisted, "
                        "fixeddraw_32_lr, fixeddraw_32_mixedrepresent) matching the "
                        "converter's --draw-mode.")
    p.add_argument("--legend_len", type=int, default=10,
                   help="prompt_set=legend: how many entries of the fixed color convention "
                        "are spelled out. A constant per run -- identical for every pair, "
                        "encodes nothing about the image. Default 10 (~95%% of positive "
                        "training rows have <= 10 instances).")
    p.add_argument("--notation", choices=("mix", "name", "rgb"), default="mix",
                   help="prompt_set=legend: color notation. mix hash-picks name/rgb per "
                        "pair, reproducing the 50/50 training distribution.")
    p.add_argument("--prompt_template", default=None,
                   help='Force a single prompt for all pairs (must contain "{np}"). '
                        "Default: per-pair hash over the --prompt_set scheme.")
    p.add_argument(
        "--prompt_template_index",
        type=int,
        choices=range(len(BASE_TEMPLATES)),
        default=None,
        help=(
            "Force one canonical BASE_TEMPLATES entry for every pair while retaining "
            "the selected draw mode's color legend, color order, and decoder contract. "
            "Intended for a controlled base-caption ablation; incompatible with "
            "--prompt_template."
        ),
    )
    p.add_argument("--prompt_seed", type=int, default=None,
                   help="Seed for the per-pair template sampling (default: --sample_seed).")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help=(
            "Per-GPU FLUX inference batch size; batch_size=1 preserves historical behavior."
        ),
    )
    p.add_argument("--generation_seed", type=int, default=0,
                   help="Base diffusion seed; derived per query for shard-independent outputs.")
    p.add_argument(
        "--randomdraw_repeats",
        type=int,
        default=1,
        help=(
            "Explicit randomdraw modes, or fixed draw modes with --randomize_palette. Generate each positive query this many times "
            "with distinct deterministic palette assignments. When combined with "
            "--generation_seed_repeats, evaluates their Cartesian product; otherwise uses the same per-query diffusion "
            "seed. Match decoded instances across all members at IoU 0.5, then vote using --consensus_mode before "
            "CGF1 scoring. 1 preserves the historic one-pass result exactly; N costs "
            "N x generation time."
        ),
    )
    p.add_argument(
        "--repeat_index_offset",
        type=int,
        default=0,
        help=(
            "Start this run's palette draws at this sweep index instead of 0, to extend a "
            "finished sweep without regenerating it: a run with offset P and "
            "--randomdraw_repeats N covers palette indices P..P+N-1. Each index is a "
            "deterministic per-query permutation independent of how many repeats a run "
            "holds, so the earlier run's members stay valid. This run's shard files hold "
            "only its own N members (and a protocol naming N); stitch them onto the "
            "earlier shard files to score the combined sweep."
        ),
    )
    p.add_argument(
        "--randomize_palette",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Evaluation-only override for a fixed explicit draw mode: use the deterministic "
            "randomdraw palette permutation while preserving that mode's scan direction, "
            "legend wording, and notation."
        ),
    )
    p.add_argument(
        "--generation_seed_repeats",
        type=int,
        default=1,
        help=(
            "Generate each positive query this many times with distinct deterministic diffusion "
            "seeds. The prompt (including palette assignment) stays fixed unless combined "
            "with --randomdraw_repeats; in that case their Cartesian product is evaluated. "
            "Match decoded instances at IoU 0.5 and vote using --consensus_mode. "
            "Requires --generation_seed."
        ),
    )
    p.add_argument(
        "--consensus_mode", choices=CONSENSUS_MODES, default=DEFAULT_CONSENSUS_MODE,
        help=("Consensus for instance support and pixels across all palette/seed repeats and every prefix. "
              "inclusive_majority (default) retains foreground on 50/50 ties (at least half the votes); "
              "strict_majority requires more than half. Single-pass results are unchanged."),
    )
    p.add_argument("--native_resolution", action="store_true",
                   help="Generate at the input image resolution.")
    p.add_argument("--instance_score", type=float, default=1.0,
                   help="Constant confidence per decoded instance (CGF1 thresholds at 0.5).")
    p.add_argument("--bg_color", type=int, nargs=3, default=(0, 0, 0),
                   help="Background color C_bg for the decoder (default black).")
    # Decoder defaults are the tuned SA-Co/Gold values; the paper's Appendix-A
    # values (tau 14, theta_size 2e-4, gamma 5) stay in decode.py.
    p.add_argument("--tau", type=float, default=58.0,
                   help="Decoder RGB tolerance; 58 was selected by the SA-Co/Gold tau ablation.")
    p.add_argument("--theta_size", type=float, default=8e-4)
    p.add_argument("--theta_erosion", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=1.0)
    add_postprocess_arguments(p)

    # outputs
    p.add_argument("--save_rgb", action=argparse.BooleanOptionalAction, default=False,
                   help="Save the raw generated RGB per query (heavy on full runs).")
    p.add_argument("--resume_from_rgb", action="store_true",
                   help="Replay verified saved raw PNGs; generate missing batches in their original "
                        "grouping, retaining existing PNGs. Requires --save_rgb and --generation_seed. "
                        "Do not run concurrently with an old process owning the same shard.")
    p.add_argument("--artifact_stage_dir", type=Path,
                   help="Node-local PNG spool; bounded background copies keep normal shared output paths.")
    p.add_argument("--postprocess_only", action="store_true",
                   help="CPU-only: decode/refine --raw_rgb_root into a NEW --out_dir, then score. "
                        "Missing positive-query RGBs are errors; generation is never attempted.")
    p.add_argument("--raw_rgb_root", type=Path,
                   help="Source evaluation containing <subset>/pred_rgb_raw/*.png, for --postprocess_only.")
    p.add_argument("--artifact_sync_dir", type=Path,
                   help="Shared lock directory for limiting concurrent artifact copies across runs.")
    p.add_argument("--artifact_sync_slots", type=int, default=2)
    p.add_argument("--log_batch_timing", action="store_true",
                   help="Log generation, raw-save, decode/consensus, panels and checkpoint wait separately.")
    p.add_argument("--save_panels", action=argparse.BooleanOptionalAction, default=False,
                   help="Save input | prompt-colored GT target | raw generated panels (heavy on full runs).")
    p.add_argument("--panels_every_repeat", action=argparse.BooleanOptionalAction, default=False,
                   help="With --save_panels, panel every palette draw this run generates instead of "
                        "only its first, so a query's panels show how the draw changed its "
                        "prediction. Each panel takes the filename of the raw image it renders "
                        "(<image_id>.png, <image_id>_r<draw>.png), so draws never overwrite one "
                        "another; the panel count multiplies by this run's repeats.")
    p.add_argument("--repeat_analysis", action=argparse.BooleanOptionalAction, default=True,
                   help="For repeated runs, save every member's decoded masks and compute standalone "
                        "and all cumulative scores during aggregation. With --save_panels, also "
                        "save sparse progression, change, support, and grayscale/red-blue agreement maps.")
    p.add_argument("--panel_every", type=int, default=500,
                   help="With --save_panels, save a panel every Nth evaluated (positive) "
                        "sample within each subset across all shards (plus the first), so full "
                        "runs produce a sparse qualitative sample from every subset instead of "
                        "one PNG per query. Use 1 for every sample.")
    p.add_argument("--run_eval", action="store_true",
                   help="After dumping predictions, run CGF1 on rank 0 and write saco_gold_results.json.")
    p.add_argument("--iou_type", choices=("segm", "bbox"), default="segm")
    p.add_argument("--dry_run", action="store_true",
                   help="Resolve samples + gate and print stats, then exit before model load.")
    # sharding (same scheme as the refseg evaluator)
    p.add_argument("--num_shards", type=int, default=None)
    p.add_argument("--shard_index", type=int, default=None)
    p.add_argument("--local_rank", type=int, default=None)
    p.add_argument("--shard_aggregate_timeout", type=int, default=7200)
    p.add_argument("--no_aggregate", action="store_true",
                   help="Each shard writes its rank_*_of_N.json and returns; skip the "
                        "in-process rank-0 merge. Pair with a separate --merge_only pass "
                        "(cross-job sharding) so the merge is not bound to one shard's "
                        "wait window / held GPU.")
    p.add_argument("--merge_only", action="store_true",
                   help="Skip model load + generation; read all rank_*_of_N.json shard "
                        "predictions, concatenate per subset, and (with --run_eval) score "
                        "CGF1. Needs --num_shards and the same --out_dir as the shards. "
                        "No GPU required.")
    return p.parse_args()


def stable_generation_seed(base_seed: int, sample: IsegSample, generation_repeat_index: int = 0) -> int:
    """Shard/order-independent diffusion seed for one query.

    Random-draw palette repeats reuse the index-0 seed; generation-seed repeats
    receive a distinct index-specific seed with the prompt held fixed.
    """
    key = f"{base_seed}:{sample.subset}:{sample.image_id}:{sample.np_text}"
    if generation_repeat_index:
        key += f":generation_repeat:{generation_repeat_index}"
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def merge_and_eval(out_dir, gt_folder, subsets, shard_pred_dir, all_samples, shard, args) -> None:
    with lock_aggregation(out_dir):
        _merge_and_eval(out_dir, gt_folder, subsets, shard_pred_dir, all_samples, shard, args)


def _merge_and_eval(out_dir, gt_folder, subsets, shard_pred_dir, all_samples, shard, args) -> None:
    """Concatenate the per-shard rank_*_of_N.json predictions into per-subset
    coco_predictions_segm.json, write inference_config.json, and (with --run_eval)
    score CGF1 -> saco_gold_results.json. Shared by the in-process rank-0 merge and
    the decoupled --merge_only aggregate pass."""
    # Cross-job sharding has no live barrier: wait for every shard's file first.
    if shard.world_size > 1 and not shard.distributed:
        refseg.wait_for_shard_metrics(
            [shard_pred_dir / f"rank_{r:05d}_of_{shard.world_size:05d}.json"
             for r in range(shard.world_size)],
            args.shard_aggregate_timeout,
        )

    merged: dict[str, list] = defaultdict(list)
    total_gated = 0
    repeat_available = []
    for r in range(shard.world_size):
        data = json.loads((shard_pred_dir / f"rank_{r:05d}_of_{shard.world_size:05d}.json").read_text())
        shard_batch_size = data.get("batch_size", 1)
        if shard_batch_size != args.batch_size:
            raise RuntimeError(
                f"Shard {r} used batch_size={shard_batch_size}, expected {args.batch_size}. "
                "Refusing to merge mixed batch settings."
            )
        if data.get("postprocessing") != postprocess_protocol(args):
            raise ValueError(f"Shard {r} postprocessing differs from aggregation settings. "
                             "Pass the same refinement switches and parameters to generation and merge.")
        expected_decoder = {"bg_color": list(args.bg_color), "tau": args.tau,
                            "theta_size": args.theta_size, "theta_erosion": args.theta_erosion, "gamma": args.gamma}
        if "decode" in data and data["decode"] != expected_decoder:
            raise ValueError(f"Shard {r} decoder settings differ from aggregation settings")
        repeat_available.append(validate_repeat_shard(data, args, all_samples[r::shard.world_size]))
        if repeat_available[-1] and (data.get("rank") != r or data.get("world_size") != shard.world_size):
            raise ValueError(f"Repeat shard {r} has incorrect rank/world_size metadata")
        total_gated += data.get("num_gated_out", 0)
        for subset, preds in data.get("preds", {}).items():
            merged[subset].extend(preds)

    if any(repeat_available) and not all(repeat_available):
        raise RuntimeError("Cannot merge mixed legacy and repeat-aware shards; repeat scores would be incomplete.")
    repeat_status = (
        "disabled" if not args.repeat_analysis else
        "pending" if all(repeat_available) else
        "unavailable_legacy_shards" if args.randomdraw_repeats * args.generation_seed_repeats > 1 else
        "not_applicable"
    )
    if repeat_status == "unavailable_legacy_shards":
        print("WARNING: legacy shards contain only final consensus masks; standalone/prefix "
              "scores and vote maps cannot be recovered. Merging final results only.")

    for subset in subsets:
        d = out_dir / subset
        d.mkdir(parents=True, exist_ok=True)
        write_json_atomic(d / "coco_predictions_segm.json", merged.get(subset, []))

    prompt_seed = args.prompt_seed if args.prompt_seed is not None else args.sample_seed
    config = {
        "subsets": subsets, "limit": args.limit, "sample_seed": args.sample_seed,
        "generation_seed": args.generation_seed,
        "max_instances": args.max_instances,
        "gate": args.gate, "gate_file": str(args.gate_file) if args.gate_file else None,
        "num_gated_out": total_gated, "steps": args.steps, "guidance_scale": args.guidance_scale,
        "prompt_sampling": (
            args.prompt_template is None and args.prompt_template_index is None
        ),
        "prompt_seed": prompt_seed,
        "prompt_set": args.prompt_set,
        "prompt_template_index": args.prompt_template_index,
        "randomize_palette": args.randomize_palette,
        "randomdraw_repeats": args.randomdraw_repeats,
        "consensus_mode": args.consensus_mode,
        "randomdraw_consensus": (
            {"match_iou": 0.5, "mask_vote": args.consensus_mode}
            if args.randomdraw_repeats > 1 else None
        ),
        "generation_seed_repeats": args.generation_seed_repeats,
        "generation_seed_consensus": (
            {"match_iou": 0.5, "mask_vote": args.consensus_mode}
            if args.generation_seed_repeats > 1 else None
        ),
        "joint_repeat_consensus": (
            {
                "palette_members": args.randomdraw_repeats,
                "generation_seed_members": args.generation_seed_repeats,
                "total_members": args.randomdraw_repeats * args.generation_seed_repeats,
                "match_iou": 0.5,
                "mask_vote": args.consensus_mode,
            }
            if args.randomdraw_repeats > 1 and args.generation_seed_repeats > 1 else None
        ),
        "legend_len": args.legend_len if args.prompt_set == "legend" else None,
        "notation": args.notation if args.prompt_set == "legend" else None,
        "draw_mode_cfg": DRAW_MODES.get(args.prompt_set),
        "prompt_templates": ([args.prompt_template] if args.prompt_template
                             else [BASE_TEMPLATES[args.prompt_template_index]]
                             if args.prompt_template_index is not None
                             else ORDERED_PROMPT_TEMPLATES if args.prompt_set == "ordered"
                             else LEGACY_PROMPT_TEMPLATES if args.prompt_set == "legacy"
                             else draw_prompt_templates(args.prompt_set)
                             if args.prompt_set in DRAW_MODES
                             else BASE_TEMPLATES),
        "instance_score": args.instance_score,
        "native_resolution": args.native_resolution,
        "save_rgb": args.save_rgb,
        "save_panels": args.save_panels,
        "repeat_analysis": {
            "status": repeat_status,
            "per_repeat_masks_saved": all(repeat_available),
            "standalone_repeats": list(range(1, args.randomdraw_repeats * args.generation_seed_repeats + 1)) if all(repeat_available) else [],
            "cumulative_prefixes": list(range(1, args.randomdraw_repeats * args.generation_seed_repeats + 1)) if all(repeat_available) else [],
            "agreement_definition": AGREEMENT_DEFINITION,
            "color_scales": COLOR_SCALES,
        },
        "panel_every": args.panel_every if args.save_panels else None,
        "panel_selection": (
            "first_and_every_Nth_positive_query_per_subset_across_all_shards"
            if args.save_panels else None
        ),
        "decode": {"bg_color": list(args.bg_color), "tau": args.tau, "theta_size": args.theta_size,
                   "theta_erosion": args.theta_erosion, "gamma": args.gamma},
        "postprocessing": postprocess_protocol(args),
        "postprocess_only": args.postprocess_only,
        "raw_rgb_root": str(args.raw_rgb_root.resolve()) if args.raw_rgb_root else None,
        "world_size": shard.world_size,
        "batch_size": args.batch_size,
        "reproducibility": reproducibility_manifest(args, gt_folder, subsets, all_samples),
    }
    write_json_atomic(out_dir / "inference_config.json", config)
    if args.randomdraw_repeats * args.generation_seed_repeats > 1:
        write_json_atomic(out_dir / "repeat_analysis_results.json", {
            "status": repeat_status, "repeat_protocol": repeat_protocol(args),
            "standalone": [], "cumulative": [],
        })
    print(f"\nDumped predictions for {len(subsets)} subset(s) to {out_dir} "
          f"(gated out {total_gated} negatives).")

    results = None
    if args.run_eval:
        # Score only the queries this run covered (sampled ids incl. gated-out
        # negatives), so with --limit non-sampled GT ids are not counted as missed.
        eval_ids_by_subset: dict[str, set] = defaultdict(set)
        for s in all_samples:
            eval_ids_by_subset[s.subset].add(s.image_id)
        results = run_eval(
            pred_root=out_dir, gt_folder=gt_folder, subsets=subsets,
            iou_type=args.iou_type, verbose=True,
            out_json=out_dir / "saco_gold_results.json",
            eval_ids_by_subset=eval_ids_by_subset,
            # IL_MCC/cgF1 depend on the presence gate -> keep results attributable.
            provenance={"gate": args.gate,
                        "gate_file": str(args.gate_file) if args.gate_file else None,
                        "batch_size": args.batch_size,
                        "consensus_mode": args.consensus_mode},
        )

    if repeat_status == "pending":
        status = analyze_repeat_outputs(
            out_dir=out_dir, gt_folder=gt_folder, subsets=subsets,
            shard_paths=[shard_pred_dir / f"rank_{r:05d}_of_{shard.world_size:05d}.json" for r in range(shard.world_size)],
            all_samples=all_samples, args=args, run_eval_fn=run_eval,
            load_gt_masks_fn=load_gt_masks, decode_rle_fn=_decode_rle, final_results=results,
        )
        config["repeat_analysis"]["status"] = status
        write_json_atomic(out_dir / "inference_config.json", config)


def main() -> None:
    with ExitStack() as resources:
        _main(resources)


def _main(resources) -> None:
    args = parse_args()
    validate_postprocess_arguments(args)
    if args.postprocess_only:
        if args.raw_rgb_root is None:
            raise ValueError("--postprocess_only requires --raw_rgb_root")
        if args.raw_rgb_root.resolve() == Path(args.out_dir).resolve():
            raise ValueError("--postprocess_only requires a new --out_dir; source outputs stay intact")
        if args.resume_from_rgb or args.save_rgb or args.merge_only:
            raise ValueError("--postprocess_only cannot be combined with --resume_from_rgb, --save_rgb or --merge_only")
    elif args.raw_rgb_root is not None:
        raise ValueError("--raw_rgb_root requires --postprocess_only")
    if args.resume_from_rgb and (not args.save_rgb or args.generation_seed is None):
        raise ValueError("--resume_from_rgb requires --save_rgb and an explicit --generation_seed")
    if args.batch_size <= 0:
        raise ValueError(f"--batch_size must be positive, got {args.batch_size}.")
    if args.prompt_template is not None and "{np}" not in args.prompt_template:
        raise ValueError('--prompt_template must contain "{np}".')
    if args.prompt_template_index is not None:
        if args.prompt_template is not None:
            raise ValueError(
                "--prompt_template_index and --prompt_template are mutually exclusive."
            )
        if args.prompt_set not in DRAW_MODES:
            raise ValueError(
                "--prompt_template_index requires a fixed-K draw-mode --prompt_set "
                "so the color legend and decoder contract remain intact."
            )
        if args.prompt_set == "fixeddraw_32_mixedrepresent":
            raise ValueError(
                "--prompt_template_index is not supported with fixeddraw_32_mixedrepresent "
                "because that mode can fall back to another template for token budget."
            )
    if args.prompt_set == "legend":
        name_cap = len(load_named_instance_palette())
        if args.legend_len > name_cap and args.notation != "rgb":
            raise ValueError(f"--legend_len {args.legend_len} exceeds the named palette "
                             f"({name_cap}); pass --notation rgb or a smaller length.")
    if args.prompt_set in ("fixeddraw_64", "randomdraw_64"):
        print(f"WARNING: --prompt_set {args.prompt_set} enumerates 64 hex entries "
              "(~915 tokens). The Flux2-Klein text encoder truncates silently at 512 "
              "tokens -- these modes are meant for longer-budget backbones "
              "(Qwen-Image-Edit 2511). Proceeding anyway.")
    if args.randomdraw_repeats <= 0:
        raise ValueError(
            f"--randomdraw_repeats must be positive, got {args.randomdraw_repeats}."
        )
    if args.generation_seed_repeats <= 0:
        raise ValueError(
            f"--generation_seed_repeats must be positive, got {args.generation_seed_repeats}."
        )
    if args.panel_every <= 0:
        raise ValueError(f"--panel_every must be positive, got {args.panel_every}.")
    if args.generation_seed_repeats > 1 and args.generation_seed is None:
        raise ValueError("--generation_seed_repeats > 1 requires an explicit --generation_seed.")
    if args.randomize_palette:
        draw_cfg = DRAW_MODES.get(args.prompt_set)
        if draw_cfg is None or draw_cfg["prompt_contract"] != "explicit":
            raise ValueError(
                "--randomize_palette requires a fixed-K draw mode with an explicit color mapping; "
                f"got {args.prompt_set!r}."
            )
    if args.randomdraw_repeats > 1:
        draw_cfg = DRAW_MODES.get(args.prompt_set)
        if draw_cfg is None or not (draw_cfg["randomize"] or args.randomize_palette):
            raise ValueError(
                "--randomdraw_repeats > 1 requires a randomized palette or --randomize_palette; "
                f"got {args.prompt_set!r}."
            )
        if draw_cfg["prompt_contract"] != "explicit":
            raise ValueError(
                "--randomdraw_repeats > 1 requires an explicit palette mapping in "
                f"the prompt; {args.prompt_set!r} has contract "
                f"{draw_cfg['prompt_contract']!r}."
            )
        if args.randomdraw_repeats > draw_cfg["k"]:
            raise ValueError(
                f"--randomdraw_repeats={args.randomdraw_repeats} exceeds "
                f"{args.prompt_set}'s {draw_cfg['k']}-color palette."
            )
        if args.prompt_template is not None:
            raise ValueError(
                "--randomdraw_repeats > 1 cannot be combined with --prompt_template; "
                "the evaluator must build each draw mode's own palette-aware prompt."
            )
    if args.repeat_index_offset < 0:
        raise ValueError(
            f"--repeat_index_offset must be non-negative, got {args.repeat_index_offset}."
        )
    if args.repeat_index_offset:
        # The offset walks the palette axis only; refuse when a second axis is present.
        if args.generation_seed_repeats > 1:
            raise ValueError(
                "--repeat_index_offset covers the palette axis only; it cannot be "
                "combined with --generation_seed_repeats > 1."
            )
        draw_cfg = DRAW_MODES.get(args.prompt_set)
        if draw_cfg is None or not (draw_cfg["randomize"] or args.randomize_palette):
            raise ValueError(
                "--repeat_index_offset only means anything for a randomized palette; "
                f"got {args.prompt_set!r} without --randomize_palette."
            )
        total_repeats = args.repeat_index_offset + args.randomdraw_repeats
        if total_repeats > draw_cfg["k"]:
            raise ValueError(
                f"--repeat_index_offset={args.repeat_index_offset} plus "
                f"--randomdraw_repeats={args.randomdraw_repeats} needs {total_repeats} "
                f"palette draws, exceeding {args.prompt_set}'s {draw_cfg['k']}-color palette."
            )
    validate_templates()
    prompt_seed = args.prompt_seed if args.prompt_seed is not None else args.sample_seed
    if args.gate == "file" and args.gate_file is None:
        print("WARNING: --gate file but no --gate_file given; nothing will be suppressed "
              "(equivalent to --gate none). IL_MCC will be poor without a real gate.")

    shard = refseg.setup_sharding(args)
    out_dir = Path(args.out_dir)
    gt_folder = args.gt_folder or (args.saco_root / "gt-annotations")
    subsets = args.subsets or list(SACO_GOLD_SUBSETS.keys())
    gate = load_gate_file(args.gate_file)

    # Build the global, deterministic sample list (concat over subsets), then shard.
    all_samples: list[IsegSample] = []
    for subset in subsets:
        all_samples.extend(
            build_subset_samples(subset, gt_folder, args.saco_root, args.gate,
                                 args.limit, args.sample_seed, args.max_instances)
        )
    if args.gate == "file" and args.require_complete_gate:
        missing = []
        for sample in all_samples:
            table = gate.get(sample.subset, gate)
            if str(sample.image_id) not in table and sample.image_id not in table:
                missing.append((sample.subset, sample.image_id))
        if missing:
            preview = ", ".join(f"{subset}:{image_id}" for subset, image_id in missing[:10])
            raise ValueError(
                f"gate file {args.gate_file} is missing {len(missing)}/{len(all_samples)} "
                f"selected queries (first: {preview}). Refusing a mixed gated/ungated run."
            )
    legacy_samples = list(all_samples[shard.rank :: shard.world_size])

    # Choose panel queries before sharding: every subset contributes its first
    # gate-positive query and then every Nth positive query globally.
    panel_sample_keys: set[tuple[str, int, str]] = set()
    if args.save_panels:
        positive_index_by_subset: dict[str, int] = defaultdict(int)
        for sample in all_samples:
            if not is_positive(sample, args.gate, gate):
                continue
            positive_index_by_subset[sample.subset] += 1
            positive_index = positive_index_by_subset[sample.subset]
            if positive_index == 1 or positive_index % args.panel_every == 0:
                panel_sample_keys.add((sample.subset, sample.image_id, sample.np_text))
        panel_counts = {
            subset: sum(key[0] == subset for key in panel_sample_keys)
            for subset in subsets
        }
        print(f"Panel sampling: first + every {args.panel_every}th positive query per subset "
              f"across all shards: {panel_counts}")

    n_pos = sum(is_positive(s, args.gate, gate) for s in legacy_samples)
    if args.dry_run:
        per_subset = defaultdict(int)
        for s in all_samples:
            per_subset[s.subset] += 1
        print(f"Subsets: {subsets}  total samples: {len(all_samples)}  "
              f"(per subset: {dict(per_subset)})  max_instances: {args.max_instances}")
        print(f"Shard {shard.rank}/{shard.world_size}: {len(legacy_samples)} samples, "
              f"{n_pos} positive after gate '{args.gate}'.")
        for s in legacy_samples[:5]:
            print(f"  [{s.subset}] id={s.image_id} np={s.np_text!r} "
                  f"present={is_positive(s, args.gate, gate)} img={s.image_path}")
        refseg.cleanup_distributed(shard)
        return

    shard_pred_dir = out_dir / "shard_preds"
    shard_pred_dir.mkdir(parents=True, exist_ok=True)

    # Decoupled aggregate pass: merge + score the written shard predictions, no model.
    if args.merge_only:
        merge_and_eval(out_dir, gt_folder, subsets, shard_pred_dir, all_samples, shard, args)
        refseg.cleanup_distributed(shard)
        return

    samples = legacy_samples
    n_gated_out = 0

    if args.save_rgb:
        resources.enter_context(lock_raw_shard(out_dir, shard.rank))
        protocol = raw_protocol(args, shard, samples, lambda s: is_positive(s, args.gate, gate))
        protocol_path = verify_raw_protocol(args, shard, protocol, samples,
                                            lambda s: is_positive(s, args.gate, gate))
        write_json_atomic(protocol_path, protocol)

    artifact_writer = resources.enter_context(ArtifactWriter(
        out_dir, args.artifact_stage_dir, args.artifact_sync_dir, args.artifact_sync_slots))

    # Load the generator (same as refseg/set_semseg).
    # Lazy on replay: fully cached fixed-draw runs require no model/GPU work.
    pipe = None
    def get_pipeline():
        nonlocal pipe
        if pipe is not None:
            return pipe
        model_load_started = time.monotonic()
        if args.transformer_path:
            from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
            root, subfolder = refseg.resolve_transformer_load_args(args.transformer_path)
            tkw = {"torch_dtype": torch.bfloat16}
            if subfolder is not None:
                tkw["subfolder"] = subfolder
            transformer = Flux2Transformer2DModel.from_pretrained(root, **tkw)
            pipe = Flux2KleinPipeline.from_pretrained(args.model, transformer=transformer,
                                                      torch_dtype=torch.bfloat16)
        else:
            from diffusers import Flux2KleinPipeline
            pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        from common.flux2_runtime import load_lora_into_pipeline
        load_lora_into_pipeline(pipe, args.lora_path)
        pipe.to(shard.device)
        pipe.set_progress_bar_config(disable=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize(shard.device)
        print(f"Rank {shard.rank}/{shard.world_size}: pipeline load+GPU transfer took "
              f"{time.monotonic() - model_load_started:.1f}s.")
        return pipe

    if not args.resume_from_rgb and not args.postprocess_only:
        get_pipeline()

    # Exact token-budget predicate for the mixed draw mode's cascade: count real
    # tokens with the pipeline tokenizer (same chat template as _get_qwen3_prompt_embeds).
    replay_tokenizer = None
    def token_budget_fits(prompt: str, _cap: int = 512) -> bool:
        nonlocal replay_tokenizer
        # Saved-RGB refinement needs only the palette, never a model tokenizer.
        # Token truncation statistics describe generation, not this CPU replay.
        if args.postprocess_only:
            return True
        if (args.resume_from_rgb or args.postprocess_only) and pipe is None:
            if replay_tokenizer is None:
                from transformers import AutoTokenizer
                replay_tokenizer = AutoTokenizer.from_pretrained(
                    args.model, subfolder="tokenizer", local_files_only=True)
            tokenizer = replay_tokenizer
        else:
            tokenizer = get_pipeline().tokenizer
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return len(tokenizer(text).input_ids) <= _cap

    if args.batch_size > 1 and args.generation_seed is None:
        print(
            "WARNING: batched inference without --generation_seed is stochastic and will not "
            "reproduce a batch_size=1 run. Set --generation_seed for a comparable protocol."
        )

    # GT instance masks (for the panel's GT cell), loaded once per subset for this
    # shard's sampled image ids. Only needed when saving panels.
    gt_masks: dict[str, dict] = {}
    if args.save_panels:
        ids_by_subset: dict[str, set] = defaultdict(set)
        for s in samples:
            ids_by_subset[s.subset].add(s.image_id)
        for subset, ids in ids_by_subset.items():
            gt_masks[subset] = load_gt_masks(subset, gt_folder, ids)

    preds_by_subset: dict[str, list] = defaultdict(list)
    repeats = args.randomdraw_repeats * args.generation_seed_repeats
    save_repeats = args.repeat_analysis and repeats > 1
    per_repeat_preds = [defaultdict(list) for _ in range(repeats)] if save_repeats else None
    n_evaluated = 0
    n_prompt_truncated = 0  # mixed-mode ultra-long-NP tail (accepted noise)
    pending_by_geometry: dict[tuple[object, ...], list[PendingIsegGeneration]] = {}
    repeat_masks_by_sample: dict[IsegSample, dict[int, list[np.ndarray]]] = {}
    raw_recovery_counts = {"reused_predictions": 0, "new_predictions": 0,
                           "generated_batches": 0, "regenerated_cached_batch_members": 0}

    def process_generation_batch(tasks: Sequence[PendingIsegGeneration]) -> None:
        batch_started = time.monotonic()
        decode_seconds = panel_seconds = 0.0
        if not tasks:
            return
        # Raw PNGs are named by the absolute sweep index, so an offset run writes
        # the same filenames as the unsplit run.
        sweep_palette_repeats = args.repeat_index_offset + args.randomdraw_repeats
        paths = [raw_prediction_path(args.raw_rgb_root if args.postprocess_only else out_dir, task.sample,
                                     args.repeat_index_offset * args.generation_seed_repeats
                                     + task.repeat_index,
                                     sweep_palette_repeats, args.generation_seed_repeats)
                 for task in tasks]
        cached = [read_saved_rgb(path) if args.resume_from_rgb or args.postprocess_only else None for path in paths]
        if args.postprocess_only and any(image is None for image in cached):
            missing = [str(path) for path, image in zip(paths, cached) if image is None]
            raise FileNotFoundError(f"Missing positive-query raw RGBs ({len(missing)} in batch): {missing[:5]}")
        cache_done = time.monotonic()
        if all(image is not None for image in cached):
            pred_rgbs = cached
        else:
            with torch.inference_mode():
                pred_rgbs = run_pipeline_batch(
                    get_pipeline(),
                    [
                        PipelineBatchRequest(
                            prompt=task.prompt,
                            image=task.image,
                            generator=task.generator,
                            height=task.height,
                            width=task.width,
                        )
                        for task in tasks
                    ],
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                )
            if len(pred_rgbs) != len(tasks):
                raise RuntimeError("Pipeline returned an incomplete generation batch")
            raw_recovery_counts["generated_batches"] += 1
            raw_recovery_counts["regenerated_cached_batch_members"] += sum(image is not None for image in cached)
            pred_rgbs = [old if old is not None else new for old, new in zip(cached, pred_rgbs)]
        generation_done = time.monotonic()

        # Persist the whole batch before decoding/panel rendering.
        for path, previous, pred_rgb in zip(paths, cached, pred_rgbs):
            if previous is not None:
                raw_recovery_counts["reused_predictions"] += 1
            else:
                raw_recovery_counts["new_predictions"] += 1
                if args.save_rgb:
                    artifact_writer.save(pred_rgb, path)
        raw_done = time.monotonic()

        for task, pred_rgb, previous in zip(tasks, pred_rgbs, cached):
            decode_started = time.monotonic()
            sample = task.sample
            # Fixed pool order reproduces nearest-colour ties from the search,
            # independently of each pass's permuted prompt legend.
            colors = task.prompt_colors
            if args.snapping and args.prompt_set in DRAW_MODES:
                from common.instanceseg.prompts import _draw_pool
                colors = _draw_pool(args.prompt_set)[0]
            masks = decode_and_refine(
                pred_rgb, task.image, (sample.gt_w, sample.gt_h), args,
                colors=colors, subset=sample.subset, image_id=sample.image_id,
                member_index=args.repeat_index_offset * args.generation_seed_repeats + task.repeat_index,
                decode_fn=decode_instances,
            )
            if save_repeats:
                per_repeat_preds[task.repeat_index][sample.subset].extend(
                    masks_to_coco(masks, image_id=sample.image_id, score=args.instance_score)
                )
            # A query's repeats can span batches; emit only its complete consensus.
            members = repeat_masks_by_sample.setdefault(sample, {})
            members[task.repeat_index] = masks
            if len(members) == repeats:
                masks = consensus_instance_masks([members[i] for i in range(repeats)],
                                                 consensus_mode=args.consensus_mode)
                preds_by_subset[sample.subset].extend(
                    masks_to_coco(masks, image_id=sample.image_id, score=args.instance_score)
                )
                del repeat_masks_by_sample[sample]
            decode_seconds += time.monotonic() - decode_started

            # Panels depict the sweep's first palette draw (none in an offset run)
            # unless --panels_every_repeat.
            first_member = args.repeat_index_offset == 0 and task.repeat_index == 0
            if args.save_panels and (args.panels_every_repeat or first_member) and (
                sample.subset, sample.image_id, sample.np_text
            ) in panel_sample_keys:
                panel_started = time.monotonic()
                d = out_dir / sample.subset / "panels"
                panel_path = d / raw_prediction_path(
                    out_dir, sample,
                    args.repeat_index_offset * args.generation_seed_repeats + task.repeat_index,
                    sweep_palette_repeats, args.generation_seed_repeats).name
                if args.resume_from_rgb and previous is not None:
                    try:
                        already_saved = read_saved_rgb(panel_path) is not None
                    except ValueError:
                        already_saved = False  # Repair only derived, corrupt panels.
                    if already_saved:
                        panel_seconds += time.monotonic() - panel_started
                        continue
                gt_rles = gt_masks.get(sample.subset, {}).get(sample.image_id, [])
                gt_bin = [_decode_rle(r, sample.gt_h, sample.gt_w) for r in gt_rles]
                # Recreate the training target colours: the legend names a fixed
                # prefix, its continuation clause defines the full palette.
                if args.prompt_set == "legend" and args.prompt_template is None:
                    if task.legend_template is None:
                        raise RuntimeError("Missing selected legend template for panel rendering.")
                    gt_style = (
                        task.prompt_style
                        if len(gt_bin) <= len(load_named_instance_palette())
                        else "rgb"
                    )
                    _, gt_colors = build_instance_prompt(
                        sample.np_text, len(gt_bin), gt_style, task.legend_template
                    )
                elif task.prompt_colors is not None:
                    gt_colors = task.prompt_colors
                else:
                    # Custom/legacy prompts carry no color contract; use a deterministic palette.
                    gt_colors = ordered_instance_colors(len(gt_bin))
                gt_target, n_gt_rendered = masks_to_prompt_target(
                    gt_bin, sample.gt_h, sample.gt_w, gt_colors, scan=task.prompt_scan
                )
                panel = refseg.make_panel(
                    [(task.image, Image.LANCZOS), (gt_target, Image.NEAREST),
                     (pred_rgb, Image.NEAREST)],
                    labels=(
                        "input",
                        f"GT target ({n_gt_rendered}/{len(gt_bin)})",
                        "generated (raw RGB)",
                    ),
                    title=task.prompt,
                )
                # Panel shares the raw PNG's name; the first draw keeps the bare <image_id>.png.
                artifact_writer.save(panel, panel_path)
                panel_seconds += time.monotonic() - panel_started

        checkpoint_started = time.monotonic()
        artifact_writer.checkpoint()
        if args.log_batch_timing:
            print(f"Batch timing rank {shard.rank}: members={len(tasks)} "
                  f"cached={sum(image is not None for image in cached)} "
                  f"read={cache_done - batch_started:.2f}s "
                  f"generate={generation_done - cache_done:.2f}s "
                  f"raw_save={raw_done - generation_done:.2f}s "
                  f"decode_consensus={decode_seconds:.2f}s panels={panel_seconds:.2f}s "
                  f"checkpoint_wait={time.monotonic() - checkpoint_started:.2f}s "
                  f"total={time.monotonic() - batch_started:.2f}s "
                  f"shared_files_committed={artifact_writer.copied_files}", flush=True)

        if args.resume_from_rgb:
            print(f"Raw replay rank {shard.rank}: {raw_recovery_counts}", flush=True)

    def enqueue_generation(task: PendingIsegGeneration) -> None:
        geometry_key = (task.image.size, task.height, task.width)
        queue = pending_by_geometry.setdefault(geometry_key, [])
        queue.append(task)
        if len(queue) >= args.batch_size:
            to_process = list(queue)
            queue.clear()
            process_generation_batch(to_process)

    # Cartesian product of G diffusion seeds x P palette assignments; the flat
    # index is palette-major when G=1 and generation-major when P=1.
    for sample in tqdm(samples, desc=f"Rank {shard.rank}/{shard.world_size}"):
        if not is_positive(sample, args.gate, gate):
            n_gated_out += 1
            continue  # negative query -> no prediction (correct image-level negative)
        n_evaluated += 1

        img = Image.open(sample.image_path).convert("RGB")
        w, h = img.size
        for repeat in range(repeats):
            # The palette index is absolute (an offset run continues at OFFSET);
            # `repeat` is local to this run's members and consensus.
            palette_repeat_index, generation_repeat_index = divmod(
                args.repeat_index_offset * args.generation_seed_repeats + repeat,
                args.generation_seed_repeats)
            prompt_colors = None
            prompt_scan = "tb"
            prompt_style = None
            legend_template = None
            if args.prompt_template is not None:
                prompt = args.prompt_template.format(np=sample.np_text)
            elif args.prompt_set == "legend":
                template, style = select_prompt_plan(
                    sample.subset, sample.image_id, sample.np_text, prompt_seed
                )
                legend_template = template
                if args.notation != "mix":
                    style = args.notation
                # default open_ended=True: the fixed-length list ends with the
                # continuation clause, stating the color convention, not a count.
                prompt, prompt_colors = build_instance_prompt(
                    sample.np_text, args.legend_len, style, template
                )
                prompt_style = style
            elif args.prompt_set in DRAW_MODES:
                # Draw-mode prompt: template + permutation + notation hashed per pair,
                # as in the converter. overflow="accept" sends the full-K prompt even
                # for ultra-long NPs (~0.16% of pairs), truncated; tallied below.
                template, perm, notation = select_draw_plan(
                    sample.subset, sample.image_id, sample.np_text, prompt_seed, args.prompt_set,
                    repeat_index=palette_repeat_index, randomize_palette=args.randomize_palette
                )
                if args.prompt_template_index is not None:
                    # Replace only the caption. The selected draw mode still supplies
                    # its exact color pool, scan direction, legend, and decoder colors.
                    template = BASE_TEMPLATES[args.prompt_template_index]
                prompt, prompt_colors = build_draw_prompt(
                    args.prompt_set, sample.np_text, template, perm, notation,
                    fits=token_budget_fits, overflow="accept",
                    randomize_palette=args.randomize_palette
                )
                prompt_scan = DRAW_MODES[args.prompt_set]["scan"]
                n_prompt_truncated += not token_budget_fits(prompt)
            else:
                templates = (ORDERED_PROMPT_TEMPLATES if args.prompt_set == "ordered"
                             else LEGACY_PROMPT_TEMPLATES)
                tmpl = select_prompt_template(
                    sample.subset, sample.image_id, sample.np_text, prompt_seed, templates
                )
                prompt = tmpl.format(np=sample.np_text)

            generator = None
            if args.generation_seed is not None and not args.postprocess_only:
                generator = torch.Generator(device=shard.device).manual_seed(
                    stable_generation_seed(args.generation_seed, sample, generation_repeat_index)
                )
            generation_height = h if args.native_resolution else None
            generation_width = w if args.native_resolution else None
            enqueue_generation(
                PendingIsegGeneration(
                    sample=sample,
                    image=img,
                    prompt=prompt,
                    prompt_colors=prompt_colors,
                    prompt_scan=prompt_scan,
                    prompt_style=prompt_style,
                    legend_template=legend_template,
                    evaluated_index=n_evaluated,
                    generator=generator,
                    height=generation_height,
                    width=generation_width,
                    repeat_index=repeat,
                )
            )

    for queued in pending_by_geometry.values():
        if queued:
            process_generation_batch(queued)

    if repeat_masks_by_sample:
        raise RuntimeError("Incomplete repeat predictions after flushing generation batches.")

    if n_prompt_truncated:
        print(f"Rank {shard.rank}: {n_prompt_truncated}/{n_evaluated} prompts exceeded the "
              "512-token budget and were truncated by the text encoder (accepted "
              "ultra-long-NP noise floor of the mixed draw mode).")

    shard_pred_path = shard_pred_dir / f"rank_{shard.rank:05d}_of_{shard.world_size:05d}.json"
    shard_payload = {
        "rank": shard.rank, "world_size": shard.world_size,
        "postprocessing": postprocess_protocol(args),
        "decode": {"bg_color": list(args.bg_color), "tau": args.tau,
                   "theta_size": args.theta_size, "theta_erosion": args.theta_erosion, "gamma": args.gamma},
        "num_samples": len(samples), "num_gated_out": n_gated_out,
        "batch_size": args.batch_size,
        "consensus_mode": args.consensus_mode,
        "gate": args.gate, "preds": {k: v for k, v in preds_by_subset.items()},
    }
    if args.resume_from_rgb:
        shard_payload["raw_recovery"] = raw_recovery_counts
    if save_repeats:
        shard_payload.update({
            "repeat_protocol": repeat_protocol(args),
            "repeat_preds": per_repeat_preds,
            "repeat_queries": [
                {"subset": sample.subset, "image_id": sample.image_id, "np_text": sample.np_text,
                 "positive": is_positive(sample, args.gate, gate)} for sample in samples
            ],
        })
    # Never publish a completed shard before every raw RGB and panel is durable.
    artifact_writer.flush()
    if args.artifact_stage_dir:
        print(f"Artifact copies complete: files={artifact_writer.copied_files}, "
              f"copy={artifact_writer.copy_seconds:.2f}s, "
              f"slot_wait={artifact_writer.wait_seconds:.2f}s", flush=True)
    write_json_atomic(shard_pred_path, shard_payload)

    refseg.barrier(shard)
    # Cross-job sharding (--no_aggregate): every shard just writes its file; a
    # separate --merge_only pass scores, so the merge isn't bound to one shard's
    # wait window. Otherwise only rank 0 does the in-process merge.
    if args.no_aggregate or shard.rank != 0:
        if args.no_aggregate and shard.rank == 0:
            print(f"--no_aggregate: wrote shard predictions to {shard_pred_dir}; "
                  f"run a --merge_only pass to score.")
        refseg.cleanup_distributed(shard)
        return

    merge_and_eval(out_dir, gt_folder, subsets, shard_pred_dir, all_samples, shard, args)
    refseg.cleanup_distributed(shard)


if __name__ == "__main__":
    main()
