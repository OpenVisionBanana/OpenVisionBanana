# ===============================================================================
# Flux2-Klein referring-image-segmentation inference + cIoU / gIoU evaluation.
# ===============================================================================
"""
Evaluates the base or partial-fine-tuned Flux2-Klein image generator on two
referring-segmentation benchmarks, prepared by
``dataset_prep/refseg/prepare_refseg_eval.py``:

  * ``refcocog``  RefCOCOg UMD val  (reported metric: **cIoU**)
  * ``reasonseg`` ReasonSeg val     (reported metric: **gIoU**)

Pipeline (mirrors evaluation/semseg/inference_partial_flux2_klein_img2img.py):
  1. Build (image, expression, GT-mask) samples for the chosen dataset.
  2. For each sample, prompt Flux2-Klein img2img with the referring expression
     to paint the referred region a randomly sampled common, well-separated NAMED
     color (common/named_colors.py) and the rest black -- so the model can't
     overfit to a single foreground color. (Drop "{color}" from the prompt
     template to fall back to plain white-on-black.)
  3. Decode the RGB output to a binary foreground mask (nearest of fg-color/black).
  4. Score with LISA's metric (K=2, ignore_index=255): two foreground/background
     accumulators give
       - cIoU = sum(fg_intersection) / sum(fg_union)        (cumulative / overall)
       - gIoU = mean over samples of per-sample fg IoU       (empty union -> 1.0)
     We always report BOTH; cIoU is the RefCOCOg headline, gIoU the ReasonSeg one.

ReasonSeg GT masks carry value 255 = ignore (don't-care) which is excluded from
both intersection and union, exactly as in LISA. RefCOCOg masks are binary {0,1}.

Typical usage:
    CUDA_VISIBLE_DEVICES=0 python3 inference_flux2_klein_refseg.py \
        --dataset refcocog \
        --index ~/open_visionbanana/datasets_archives/refseg/refcocog/val_umd/refcocog_val_umd.json \
        --transformer_path /path/to/open-visionbanana-partial-ft \
        --out_dir outputs/refcocog_val

    CUDA_VISIBLE_DEVICES=0 python3 inference_flux2_klein_refseg.py \
        --dataset reasonseg \
        --data_dir ~/open_visionbanana/datasets_archives/refseg/reasonseg_val \
        --transformer_path /path/to/open-visionbanana-partial-ft \
        --out_dir outputs/reasonseg_val

Sharding (multi-GPU) works like the semseg evaluator: launch one process per
shard with torchrun, or pass --num_shards/--shard_index manually. Rank 0
aggregates the per-shard metric JSONs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel

# Repo root, derived from this file; the index's image paths are repo-relative.
REPO_ROOT = Path(__file__).resolve().parents[2]
REFSEG_ROOT = REPO_ROOT / "datasets_archives/refseg"

# Palette of common, well-separated NAMED colors for painting the referred region.
sys.path.insert(0, str(REPO_ROOT))
from common.refseg import named_colors  # noqa: E402

# Referring-segmentation prompt (semseg style: [TASK: ...] prefix + painting
# instruction). The referred region is painted a randomly sampled named color via
# "{color}" (rendered as e.g. "red [220, 30, 30]"); the decoder snaps against it.
# Drop "{color}" from a custom --prompt_template for plain white-on-black.
DEFAULT_PROMPT_TEMPLATE = (
    "[TASK: REFERRING_SEGMENTATION] Segment the region that best matches this "
    'description: "{expr}". Paint the matching region {color} and '
    "leave everything else black [0, 0, 0]. Output only the segmentation mask image."
)

IGNORE_INDEX = 255


# --------------------------------------------------------------------------- #
# Samples
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RefSample:
    image_path: Path
    mask_path: Path
    expr: str
    output_stem: str  # unique per (object/expression)
    orig_expr: Optional[str] = None  # implicit query before a reasonseg rewrite (None = not rewritten)


def build_refcocog_samples(
    index_path: Path,
    images_root: Optional[Path],
    per_sentence: bool,
    limit: Optional[int],
) -> list[RefSample]:
    """Samples from the RefCOCOg UMD val index produced by prepare_refseg_eval.py.

    ``per_sentence`` (default) emits one sample per referring sentence (the COCO
    object mask is reused across its sentences) -- the standard RefCOCO* protocol
    of ~4,896 val samples. Otherwise one sample per object (ref_id), using the
    first sentence.
    """
    index = json.loads(index_path.read_text())
    base = index_path.parent
    samples: list[RefSample] = []
    for entry in index:
        mask_path = base / entry["mask"]
        if images_root is not None:
            image_path = images_root / entry["file_name"]
        else:
            ip = Path(entry["image_path"])
            image_path = ip if ip.is_absolute() else REPO_ROOT / ip
        sents = entry["sentences"] if per_sentence else entry["sentences"][:1]
        for si, sent in enumerate(sents):
            samples.append(
                RefSample(
                    image_path=image_path,
                    mask_path=mask_path,
                    expr=sent,
                    output_stem=f"{entry['ref_id']}_{si}",
                )
            )
    samples.sort(key=lambda s: s.output_stem)
    if limit is not None:
        samples = samples[:limit]
    return samples


def build_reasonseg_samples(
    data_dir: Path,
    expr_index: int,
    all_expr: bool,
    limit: Optional[int],
) -> list[RefSample]:
    """Samples from the ReasonSeg val dir (masks/ produced by prepare_refseg_eval.py).

    Each <stem>.json holds a list of paraphrased queries describing the *same*
    target, so the GT mask is shared. Default: one sample per image using
    query ``expr_index`` (the 200-sample ReasonSeg val convention). ``all_expr``
    emits one sample per paraphrase.
    """
    mask_dir = data_dir / "masks"
    samples: list[RefSample] = []
    for json_path in sorted(data_dir.glob("*.json")):
        stem = json_path.stem
        mask_path = mask_dir / f"{stem}.png"
        image_path = data_dir / f"{stem}.jpg"
        if not mask_path.exists() or not image_path.exists():
            continue
        texts = json.loads(json_path.read_text())["text"]
        if not texts:
            continue
        chosen = texts if all_expr else [texts[expr_index % len(texts)]]
        for ei, expr in enumerate(chosen):
            stem_id = f"{stem}_{ei}" if all_expr else stem
            samples.append(RefSample(image_path, mask_path, expr, stem_id))
    samples.sort(key=lambda s: s.output_stem)
    if limit is not None:
        samples = samples[:limit]
    return samples


# --------------------------------------------------------------------------- #
# Mask decode + LISA metric
# --------------------------------------------------------------------------- #
def rgb_to_foreground(rgb: Image.Image, fg_color: Sequence[int] = (255, 255, 255)) -> np.ndarray:
    """Decode an RGB mask output to a binary foreground map.

    The referred region is painted ``fg_color`` and everything else black, so
    snap each (anti-aliased) pixel to the nearer of fg_color / black in RGB:
    foreground = pixels closer to fg_color. For white fg this is exactly a
    luminance threshold at ~127.
    """
    arr = np.asarray(rgb.convert("RGB"), dtype=np.int64)
    fg = np.asarray(fg_color, dtype=np.int64)
    d_fg = ((arr - fg) ** 2).sum(-1)
    d_bg = (arr ** 2).sum(-1)  # distance to black
    return (d_fg < d_bg).astype(np.uint8)


def load_gt(mask_path: Path) -> np.ndarray:
    """ReasonSeg {0,1,255} / RefCOCOg {0,1} label map (255 = ignore)."""
    return np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)


def gt_to_rgb(gt: np.ndarray) -> np.ndarray:
    """Amplify a {0,1,255} label map into a visible RGB image.

    target(1) -> white, ignore(255) -> gray, background(0) -> black (the raw
    label values 0/1 are nearly black and invisible to the eye on their own).
    """
    vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
    vis[gt == 1] = (255, 255, 255)
    vis[gt == IGNORE_INDEX] = (128, 128, 128)
    return vis


def _wrap_text(draw, text, font, max_width):
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if not cur or draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def make_panel(cells, labels, title, title2=None, row_height=512, pad=6, bg=(255, 255, 255)):
    """Tile (image, resample) cells side by side with per-cell labels and a
    wrapped header band across the top. ``title`` is the first line; ``title2``
    (reasonseg: the rewritten prompt) is wrapped directly below it when given."""
    from PIL import ImageDraw, ImageFont

    font = ImageFont.load_default()
    resized = []
    for im, resample in cells:
        w, h = im.size
        resized.append(im.convert("RGB").resize((max(1, round(w * row_height / h)), row_height), resample))
    body_w = sum(im.width for im in resized) + pad * (len(resized) + 1)

    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    line_h = (font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) + 4
    title_lines = _wrap_text(probe, title, font, body_w - 2 * pad)
    if title2 is not None:
        title_lines += _wrap_text(probe, title2, font, body_w - 2 * pad)
    title_h = pad + line_h * max(len(title_lines), 1) + pad

    panel = Image.new("RGB", (body_w, title_h + row_height + 2 * pad), bg)
    draw = ImageDraw.Draw(panel)
    y = pad
    for line in title_lines:
        draw.text((pad, y), line, fill=(0, 0, 0), font=font)
        y += line_h

    x, top = pad, title_h + pad
    for idx, im in enumerate(resized):
        panel.paste(im, (x, top))
        if labels and idx < len(labels):
            tw = draw.textlength(labels[idx], font=font)
            draw.rectangle([x, top, x + tw + 6, top + 14], fill=(0, 0, 0))
            draw.text((x + 3, top + 2), labels[idx], fill=(255, 255, 255), font=font)
        x += im.width + pad
    return panel


def intersection_and_union(pred: np.ndarray, gt: np.ndarray, K: int = 2,
                           ignore_index: int = IGNORE_INDEX):
    """Per-class intersection/union/target areas following LISA's metric definition (K classes).

    Returns per-class (area_intersection, area_union, area_target) over the K
    classes; ignore_index pixels are excluded from all three.
    """
    pred = pred.reshape(-1).copy()
    gt = gt.reshape(-1)
    pred[gt == ignore_index] = ignore_index
    inter = pred[pred == gt]
    area_i = np.histogram(inter, bins=K, range=(0, K - 1))[0]
    area_p = np.histogram(pred, bins=K, range=(0, K - 1))[0]
    area_t = np.histogram(gt, bins=K, range=(0, K - 1))[0]
    area_u = area_p + area_t - area_i
    return area_i.astype(np.float64), area_u.astype(np.float64), area_t.astype(np.float64)


class RefSegAccumulator:
    """Accumulate LISA cIoU/gIoU over 2 classes [background, foreground]."""

    def __init__(self):
        self.cum_intersection = np.zeros(2, dtype=np.float64)
        self.cum_union = np.zeros(2, dtype=np.float64)
        self.acc_iou = np.zeros(2, dtype=np.float64)  # sum of per-sample IoU
        self.num_samples = 0

    def update(self, pred_fg: np.ndarray, gt: np.ndarray) -> None:
        inter, union, _ = intersection_and_union(pred_fg, gt, K=2)
        self.cum_intersection += inter
        self.cum_union += union
        per_iou = inter / (union + 1e-5)
        per_iou[union == 0] = 1.0  # no-object class scores perfect (LISA)
        self.acc_iou += per_iou
        self.num_samples += 1

    def merge(self, other: "RefSegAccumulator") -> None:
        self.cum_intersection += other.cum_intersection
        self.cum_union += other.cum_union
        self.acc_iou += other.acc_iou
        self.num_samples += other.num_samples

    def compute(self) -> dict[str, float]:
        iou_class = self.cum_intersection / (self.cum_union + 1e-10)
        giou_class = self.acc_iou / max(self.num_samples, 1)
        return {
            "cIoU": float(round(iou_class[1] * 100, 2)),   # foreground, cumulative
            "gIoU": float(round(giou_class[1] * 100, 2)),  # foreground, per-sample mean
            "num_samples": self.num_samples,
        }

    def to_dict(self) -> dict:
        return {
            "cum_intersection": self.cum_intersection.tolist(),
            "cum_union": self.cum_union.tolist(),
            "acc_iou": self.acc_iou.tolist(),
            "num_samples": self.num_samples,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RefSegAccumulator":
        acc = cls()
        acc.cum_intersection = np.array(data["cum_intersection"], dtype=np.float64)
        acc.cum_union = np.array(data["cum_union"], dtype=np.float64)
        acc.acc_iou = np.array(data["acc_iou"], dtype=np.float64)
        acc.num_samples = int(data["num_samples"])
        return acc


# --------------------------------------------------------------------------- #
# Sharding (same scheme as the semseg evaluator)
# --------------------------------------------------------------------------- #
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
        world_size, rank = env_world_size, env_rank
        distributed = world_size > 1

    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} out of range for {world_size} shards.")

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        local_rank = args.local_rank if args.local_rank is not None else env_local_rank
        local_rank = int(local_rank) % max(num_devices, 1)
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        local_rank, device = 0, "cpu"

    if distributed and torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    return ShardInfo(rank, world_size, local_rank, device,
                     distributed and torch.distributed.is_initialized())


def barrier(shard: ShardInfo) -> None:
    if shard.distributed:
        torch.distributed.barrier()


def cleanup_distributed(shard: ShardInfo) -> None:
    if shard.distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def wait_for_shard_metrics(paths: Sequence[Path], timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    missing = [p for p in paths if not p.exists()]
    while missing and time.time() < deadline:
        time.sleep(5)
        missing = [p for p in paths if not p.exists()]
    if missing:
        raise TimeoutError("Timed out waiting for shard metrics:\n  "
                           + "\n  ".join(str(p) for p in missing))


def resolve_transformer_load_args(transformer_path: str) -> tuple[str, str | None]:
    path = Path(transformer_path).expanduser()
    if (path / "transformer").is_dir():
        return str(path), "transformer"
    return str(path), None


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("refcocog", "reasonseg"), required=True)
    p.add_argument("--model", default="black-forest-labs/FLUX.2-klein-base-9B")
    p.add_argument("--transformer_path", default=None,
                   help="Trained output / checkpoint / transformer dir; omit for base model.")
    p.add_argument("--lora_path", default=None,
                   help="Standard Diffusers LoRA directory (holds pytorch_lora_weights.safetensors) applied on top of --model.")
    p.add_argument("--out_dir", required=True)

    # refcocog
    p.add_argument("--index", type=Path,
                   default=REFSEG_ROOT / "refcocog/val_umd/refcocog_val_umd.json",
                   help="refcocog: path to refcocog_val_umd.json from prepare_refseg_eval.py.")
    p.add_argument("--images_root", type=Path, default=None,
                   help="refcocog: override the COCO train2014 image dir (else index's image_path).")
    p.add_argument("--per_object", action="store_true",
                   help="refcocog: one sample per object (first sentence) instead of per sentence.")
    # reasonseg
    p.add_argument("--data_dir", type=Path, default=REFSEG_ROOT / "reasonseg_val",
                   help="reasonseg: ReasonSeg val dir (holds <stem>.jpg/.json and masks/).")
    p.add_argument("--reasonseg_expr_index", type=int, default=0,
                   help="reasonseg: which paraphrase to use when one-per-image (default 0).")
    p.add_argument("--reasonseg_all_expr", action="store_true",
                   help="reasonseg: emit one sample per paraphrase instead of one per image.")
    p.add_argument("--reasonseg_rewrite", action=argparse.BooleanOptionalAction, default=True,
                   help="reasonseg: rewrite each implicit query into an explicit referring "
                        "expression from a cached Qwen3-VL pass (default ON, to match the "
                        "Vision Banana protocol). --no-reasonseg-rewrite scores the raw "
                        "implicit queries (non-comparable baseline / ablation).")
    p.add_argument("--reasonseg_rewrite_file", type=Path,
                   default=REPO_ROOT / "evaluation/assets/reasonseg_val_rewrites_qwen3vl32bthink.json",
                   help="reasonseg: cached rewrites JSON ({stem: {target}}) produced by "
                        "evaluation/refseg/rewrite_reasonseg_qwen.py.")

    p.add_argument("--prompt_template", default=DEFAULT_PROMPT_TEMPLATE,
                   help='Must contain "{expr}". May contain "{color}" to paint the '
                        "referred region a random common named color (decoder snaps "
                        "to it); omit it to keep plain white-on-black.")
    p.add_argument("--color_seed", type=int, default=0,
                   help="Seed for the per-sample named-color draw (reproducible).")
    p.add_argument("--generation_seed", type=int, default=0,
                   help="Base diffusion seed; derived per sample for shard-independent outputs.")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--native_resolution", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save_masks", action="store_true",
                   help="Also save decoded binary masks (raw RGB is always saved).")
    p.add_argument("--save_panels", "--save-panels", action=argparse.BooleanOptionalAction, default=True,
                   help="Save input | ground-truth | output qualitative panels (default on). "
                        "Use --no-save-panels to skip (e.g. on a full 4,896-sample run).")
    p.add_argument("--dry_run", action="store_true",
                   help="Resolve samples and print the first few, then exit before model load.")

    p.add_argument("--num_shards", type=int, default=None)
    p.add_argument("--shard_index", type=int, default=None)
    p.add_argument("--local_rank", type=int, default=None)
    p.add_argument("--shard_aggregate_timeout", type=int, default=3600)
    p.add_argument("--no_aggregate", action="store_true",
                   help="Write this shard's metrics and exit. Use a dependent CPU "
                        "aggregate job for independently scheduled array shards.")
    return p.parse_args()


def build_samples(args: argparse.Namespace) -> list[RefSample]:
    if args.dataset == "refcocog":
        if args.index is None:
            raise ValueError("--index is required for --dataset refcocog.")
        return build_refcocog_samples(args.index, args.images_root,
                                      per_sentence=not args.per_object, limit=args.limit)
    if args.data_dir is None:
        raise ValueError("--data_dir is required for --dataset reasonseg.")
    return build_reasonseg_samples(args.data_dir, args.reasonseg_expr_index,
                                   args.reasonseg_all_expr, args.limit)


# --------------------------------------------------------------------------- #
# ReasonSeg rewrite-cache quality gate
#
# A cached rewrite is unusable when it is
#   "unrewritten"  the target is missing or still the raw implicit query;
#   "leaked"       a Thinking model's <think> block hit max_new_tokens, so the
#                  parser's last-line fallback returned reasoning prose.
# Both are fixed by rewrite_reasonseg_qwen.py --repair. The predicates live here
# so the cache writer and this reader agree on what a good entry is.
# --------------------------------------------------------------------------- #
MAX_REFERRING_EXPR_WORDS = 20  # the rewrite system prompt asks for "under ~15 words"

_REASONING_LEAD_RE = re.compile(
    r"^(wait|but|so|however|alternatively|maybe|perhaps|let|hmm|okay|actually|looking|"
    r"since|therefore|thus|first|now|well|note|also|because|then|given|"
    r"i\s|the\s+(answer|question|query|user|problem|instructions?|correct)\b)",
    re.IGNORECASE)
_PROSE_RE = re.compile(r"[.!?][\"')\]]?\s+\S")  # a sentence break -> prose, not a phrase


def looks_like_referring_expr(text: str) -> bool:
    """Cheap sanity gate: a groundable noun phrase, or leaked model prose?

    A rewrite is a short noun phrase ("the open car door on the right side of the
    silver car"). Reasoning leakage is long, multi-sentence, and usually opens on a
    discourse connective. Rejecting it at parse time degrades the sample to the raw
    query, which this gate reports and --repair fixes.
    """
    t = (text or "").strip()
    if not t or "<think>" in t.lower():
        return False
    if len(t.split()) > MAX_REFERRING_EXPR_WORDS:
        return False
    if t.endswith("?"):  # a question is an implicit query, never a rewrite of one
        return False
    return not (_REASONING_LEAD_RE.match(t) or _PROSE_RE.search(t))


def rewrite_entry_problem(entry) -> Optional[str]:
    """Why a rewrite-cache entry is unusable ("missing"/"unrewritten"/"leaked"),
    or None when it is a good rewrite."""
    if not isinstance(entry, dict):
        return "missing"
    target = str(entry.get("target", "")).strip()
    if not target:
        return "missing"
    if target.lower() == str(entry.get("query", "")).strip().lower():
        return "unrewritten"
    if not looks_like_referring_expr(target):
        return "leaked"
    return None


def apply_reasonseg_rewrites(
    samples: list[RefSample], rewrite_file: Path
) -> tuple[list[RefSample], int, int]:
    """Replace each reasonseg sample's implicit query with its cached explicit
    referring expression (image-grounded Qwen3-VL rewrite). Keyed by output_stem so
    keys line up with the rewrite pass. Missing stems keep the original query (and
    are counted as fallbacks) so a partial cache never drops a sample.

    Fails loud if the cache file is absent (pass --no-reasonseg_rewrite to score
    raw queries on purpose). The cache is used as-is; unusable entries are reported
    by class.
    """
    if not rewrite_file.exists():
        raise FileNotFoundError(
            f"reasonseg rewrite cache not found: {rewrite_file}\n"
            "Generate it once (checkpoint-independent, reused across checkpoints) with\n"
            "  python evaluation/refseg/rewrite_reasonseg_qwen.py\n"
            "To score the raw implicit queries instead (baseline/ablation), pass "
            "--no-reasonseg_rewrite."
        )
    cache = json.loads(rewrite_file.read_text())
    rewritten, n_rw, n_fb = [], 0, 0
    problems: dict[str, list[str]] = {}
    for s in samples:
        entry = cache.get(s.output_stem)
        problem = rewrite_entry_problem(entry)
        if problem is not None:
            problems.setdefault(problem, []).append(s.output_stem)
        target = entry.get("target") if isinstance(entry, dict) else None
        if target:
            rewritten.append(replace(s, expr=str(target), orig_expr=s.expr))
            n_rw += 1
        else:
            rewritten.append(s)
            n_fb += 1
    if problems:
        n_bad = sum(len(v) for v in problems.values())
        detail = ", ".join(f"{len(v)} {k}" for k, v in sorted(problems.items()))
        examples = [stem for _, v in sorted(problems.items()) for stem in v][:5]
        print(f"WARNING: {n_bad}/{len(samples)} entries in {rewrite_file.name} are not usable "
              f"referring expressions ({detail}); e.g. {examples}. Those samples are prompted "
              "with the raw implicit query or a truncated reasoning monologue, so this gIoU is "
              "NOT protocol-comparable. Repair the cache first (re-runs only the bad stems):\n"
              f"  python evaluation/refseg/rewrite_reasonseg_qwen.py --repair --out {rewrite_file}",
              file=sys.stderr)
    return rewritten, n_rw, n_fb


def main() -> None:
    args = parse_args()
    if "{expr}" not in args.prompt_template:
        raise ValueError('--prompt_template must contain "{expr}".')
    shard = setup_sharding(args)

    out_dir = Path(args.out_dir)
    all_samples = build_samples(args)

    # reasonseg: swap each implicit query for its cached explicit referring
    # expression (image-grounded Qwen3-VL rewrite) BEFORE sharding, so every shard
    # sees the same global mapping. ON by default; --no-reasonseg-rewrite opts out.
    use_rewrite = args.dataset == "reasonseg" and args.reasonseg_rewrite
    n_rewritten = n_rewrite_fallback = 0
    if use_rewrite:
        all_samples, n_rewritten, n_rewrite_fallback = apply_reasonseg_rewrites(
            all_samples, args.reasonseg_rewrite_file)

    samples = list(all_samples[shard.rank :: shard.world_size])

    use_color = "{color}" in args.prompt_template
    if args.dry_run:
        print(f"Dataset: {args.dataset} | total samples: {len(all_samples)}")
        print(f"Shard {shard.rank}/{shard.world_size}: {len(samples)} samples"
              f" | fg color: {'random named' if use_color else 'white'}")
        for s in samples[:5]:
            if use_color:
                name, rgb = named_colors.sample_one(random.Random(f"{args.color_seed}:{s.output_stem}"))
                color = f"{name} {list(rgb)}"
            else:
                color = "white [255, 255, 255]"
            print(f"  expr: {s.expr!r}  color: {color}\n    image: {s.image_path}\n    mask:  {s.mask_path}")
        cleanup_distributed(shard)
        return

    raw_dir = out_dir / "pred_rgb_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out_dir / "pred_masks"
    if args.save_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = out_dir / "panels"
    if args.save_panels:
        panel_dir.mkdir(parents=True, exist_ok=True)
    shard_metric_dir = out_dir / "shard_metrics"
    shard_metric_dir.mkdir(parents=True, exist_ok=True)

    if args.transformer_path:
        root, subfolder = resolve_transformer_load_args(args.transformer_path)
        tkw = {"torch_dtype": torch.bfloat16}
        if subfolder is not None:
            tkw["subfolder"] = subfolder
        transformer = Flux2Transformer2DModel.from_pretrained(root, **tkw)
        pipe = Flux2KleinPipeline.from_pretrained(args.model, transformer=transformer,
                                                  torch_dtype=torch.bfloat16)
    else:
        pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    from common.flux2_runtime import load_lora_into_pipeline
    load_lora_into_pipeline(pipe, args.lora_path)
    pipe.to(shard.device)
    pipe.set_progress_bar_config(disable=True)

    acc = RefSegAccumulator()
    for sample in tqdm(samples, desc=f"Rank {shard.rank}/{shard.world_size}"):
        img = Image.open(sample.image_path).convert("RGB")
        w, h = img.size

        # Paint the referred region a randomly sampled common named color (and
        # decode against that exact color); fall back to white if the template
        # has no "{color}" field. Per-sample rng keeps the draw reproducible.
        if use_color:
            rng = random.Random(f"{args.color_seed}:{sample.output_stem}")
            color_name, fg_color = named_colors.sample_one(rng)
        else:
            color_name, fg_color = "white", (255, 255, 255)
        r, g, b = fg_color
        color_token = f"{color_name} [{r}, {g}, {b}]"
        prompt = args.prompt_template.format(expr=sample.expr, color=color_token)

        gen_kwargs = dict(prompt=prompt, image=img,
                          num_inference_steps=args.steps, guidance_scale=args.guidance_scale)
        if args.generation_seed is not None:
            digest = hashlib.sha256(f"{args.generation_seed}:{sample.output_stem}".encode()).digest()
            gen_kwargs["generator"] = torch.Generator(device=shard.device).manual_seed(
                int.from_bytes(digest[:8], "big") % (2**63 - 1)
            )
        if args.native_resolution:
            gen_kwargs.update({"height": h, "width": w})
        with torch.inference_mode():
            pred_rgb = pipe(**gen_kwargs).images[0]
        pred_rgb.save(raw_dir / f"{sample.output_stem}.png")

        gt = load_gt(sample.mask_path)
        pred_fg = rgb_to_foreground(pred_rgb, fg_color)
        if pred_fg.shape != gt.shape:
            pred_fg = np.array(
                Image.fromarray(pred_fg * 255).resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
            ) > 127
            pred_fg = pred_fg.astype(np.uint8)
        if args.save_masks:
            Image.fromarray((pred_fg * 255).astype(np.uint8)).save(mask_dir / f"{sample.output_stem}.png")

        # Qualitative panel: input | ground truth (amplified) | model output,
        # with the prompt as a header. Photos use LANCZOS; the discrete GT mask
        # uses NEAREST so its edges stay crisp.
        if args.save_panels:
            # reasonseg rewrite: show the original implicit query on top and the
            # rewritten explicit prompt (what's actually fed) below it. Otherwise
            # (refcocog, or a fallback sample the cache missed) show the one prompt.
            if sample.orig_expr is not None:
                title, title2 = f"original: {sample.orig_expr}", f"rewritten: {prompt}"
            else:
                title, title2 = f"prompt: {prompt}", None
            panel = make_panel(
                [(img, Image.LANCZOS),
                 (Image.fromarray(gt_to_rgb(gt)), Image.NEAREST),
                 (pred_rgb, Image.LANCZOS)],
                labels=("input", "ground truth", "output"),
                title=title,
                title2=title2,
            )
            panel.save(panel_dir / f"{sample.output_stem}.png")

        acc.update(pred_fg, gt)

    shard_metric_path = shard_metric_dir / f"rank_{shard.rank:05d}_of_{shard.world_size:05d}.json"
    shard_metric_path.write_text(json.dumps(
        {"rank": shard.rank, "world_size": shard.world_size,
         "num_total_samples": len(all_samples), **acc.to_dict()}, indent=2))

    barrier(shard)
    if args.no_aggregate:
        cleanup_distributed(shard)
        return
    if shard.rank != 0:
        cleanup_distributed(shard)
        return

    if shard.world_size > 1 and not shard.distributed:
        wait_for_shard_metrics(
            [shard_metric_dir / f"rank_{r:05d}_of_{shard.world_size:05d}.json"
             for r in range(shard.world_size)],
            args.shard_aggregate_timeout,
        )

    merged = RefSegAccumulator()
    for r in range(shard.world_size):
        merged.merge(RefSegAccumulator.from_dict(
            json.loads((shard_metric_dir / f"rank_{r:05d}_of_{shard.world_size:05d}.json").read_text())))

    results = merged.compute()
    headline = "cIoU" if args.dataset == "refcocog" else "gIoU"
    out_json = out_dir / f"{args.dataset}_results.json"
    out_json.write_text(json.dumps(
        {**results, "dataset": args.dataset, "headline_metric": headline,
         "prompt_template": args.prompt_template, "steps": args.steps,
         "guidance_scale": args.guidance_scale,
         "reasonseg_rewrite": use_rewrite,
         "reasonseg_rewrite_file": str(args.reasonseg_rewrite_file) if use_rewrite else None,
         "n_rewritten": n_rewritten, "n_rewrite_fallback": n_rewrite_fallback}, indent=2))

    print("\nReferring-segmentation results")
    print("=" * 50)
    print(f"Dataset: {args.dataset}  (headline: {headline})")
    print(f"Samples: {results['num_samples']}  | shards: {shard.world_size}")
    print(f"cIoU: {results['cIoU']:.2f}")
    print(f"gIoU: {results['gIoU']:.2f}")
    print("=" * 50)
    print(f"Saved results to: {out_json}")
    cleanup_distributed(shard)


if __name__ == "__main__":
    main()
