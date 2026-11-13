#!/usr/bin/env python3
"""Grouped multi-pass semseg evaluation: full-taxonomy mIoU with in-distribution prompts.


  1. Per image, partition the explicitly prompted taxonomy classes into
     K = ceil(N / group_size) disjoint groups with a seeded per-image RNG
     (19 classes @ size 5 -> 5,5,5,4; COCO-Object has 80 foreground classes
     @ size 27 -> 27,27,26 and keeps background implicit).
  2. Run one img2img pass per group with a randomized-template legend. Name
     notation uses the legacy 37-color named palette and optional farthest-point
     selection. RGB/hex notation uses the exact shared training sampler; e.g.
     rgb + best64_maxnegdraw uniformly draws distinct colors from the 64-color
     max-separated pool independently for every pass.
  3. Decode each pass against its own legend keeping the per-pixel snap distance
     (diversify.rgb_to_trainids_via_legend_with_dist), map to global class ids.
  4. Merge the K maps by minimum snap distance (diversify.merge_grouped_predictions)
     into ONE full-taxonomy label map. For an implicit-background taxonomy,
     pixels unclaimed by all foreground passes become background. Update the
     accumulator exactly once per image (per-pass updates would double-count area).

Unlike the "set" style (which masks unselected classes in both pred and GT and thus
scores an easier, non-comparable subset metric), the merged prediction covers every
class, so the resulting mIoU is directly comparable to prompt_style=fixed/randomdraw.

The partition + legends are a pure function of (subset_seed, stem, group_size,
color_min_delta_e, color_mapping, notation, farthest_colors) -- GT-independent -- so shards and reruns
re-derive identically with no shared eval_set file; legends/<stem>.json is the
reproducibility artifact. Cost remains K diffusion generations per image.
``group_batch_size`` may place compatible prompts for the same image in one FLUX
call for better throughput; decoding and merging remain independent, while peak
latent/activation memory scales with that batch size.

``--legend_pruning_json`` (legend_pruning.py) restricts each image's partition to
the classes a VLM named as present. The kept ids enter derive_entry in taxonomy
order, so seeds and colour draws follow the same rules; an image with no kept class
is scored as all background without a FLUX call.

Reuses the pipeline / sharding / metric machinery from the main evaluator.
"""
from __future__ import annotations

import argparse
from itertools import chain, islice
import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))  # repo root for common.*
import inference_partial_flux2_klein_img2img as base  # noqa: E402
import legend_pruning  # noqa: E402
from common.semseg import diversify  # noqa: E402
from common.semseg.build_palette import srgb_to_lab  # noqa: E402


_NAMED_LAB = diversify.NAMED_LAB.astype(np.float64)
_BLACK_LAB = srgb_to_lab(np.array([diversify.BACKGROUND_COLOR], dtype=np.int64))[0]
_D_BLACK = ((_NAMED_LAB - _BLACK_LAB) ** 2).sum(1)


def _named_pool(color_mapping: str, num_classes: int) -> Optional[set[int]]:
    """NAMED-palette indices the training draw under ``color_mapping`` could use
    for name-style prompts: best32 restricts to the 32 most-separated names;
    bestdrawperimg's per-image pools are prefixes of the same separation order, so
    the num_classes-sized prefix is its full-taxonomy analogue; every other mode
    reaches the whole named palette (None = unrestricted)."""
    if color_mapping == "best32_maxnegdraw":
        return set(diversify.NAMED_SEP_ORDER[:32])
    if color_mapping == "bestdrawperimg":
        return set(diversify.NAMED_SEP_ORDER[:num_classes])
    return None


def _farthest_point_colors(rng: random.Random, k: int, exclude: set[int],
                           allowed: Optional[set[int]] = None) -> list[int]:
    """Max-min selection of k NAMED palette colors (indices): random start, then
    repeatedly add the color whose nearest already-chosen color (CIELAB) is
    farthest. Two twists vs inference_randomdraw_separated.py's sampler: BLACK is
    an implicit pre-chosen anchor (it is the void snap target in decoding, so every
    color keeps a margin to it), and ``exclude`` optionally removes colors already
    assigned to earlier groups when the full prompted taxonomy fits the pool."""
    avail = np.array([i for i in range(len(_NAMED_LAB))
                      if i not in exclude and (allowed is None or i in allowed)])
    # Random start for per-group variety, restricted to colors comfortably off-black.
    eligible = avail[_D_BLACK[avail] >= 25.0 ** 2]
    start = int(eligible[rng.randrange(len(eligible))])
    chosen = [start]
    min_d = np.minimum(((_NAMED_LAB[avail] - _NAMED_LAB[start]) ** 2).sum(1), _D_BLACK[avail])
    for _ in range(k - 1):
        min_d[np.isin(avail, chosen)] = -1.0
        nxt = int(avail[np.argmax(min_d)])
        chosen.append(nxt)
        min_d = np.minimum(min_d, ((_NAMED_LAB[avail] - _NAMED_LAB[nxt]) ** 2).sum(1))
    return chosen


def _sample_named_colors(rng: random.Random, k: int, min_delta_e: float,
                         exclude: set[int], allowed: Optional[set[int]] = None) -> list[int]:
    """diversify.sample_colors' greedy dE-floor draw, restricted to the NAMED
    palette minus ``exclude``. The caller enforces cross-group distinctness when
    the full prompted taxonomy fits the named pool."""
    avail = [i for i in range(len(_NAMED_LAB))
             if i not in exclude and (allowed is None or i in allowed)]
    rng.shuffle(avail)
    min_sq = min_delta_e * min_delta_e
    chosen: list[int] = []
    leftover: list[int] = []
    for i in avail:
        if chosen and min(((_NAMED_LAB[chosen] - _NAMED_LAB[i]) ** 2).sum(1)) < min_sq:
            leftover.append(i)
            continue
        chosen.append(i)
        if len(chosen) == k:
            break
    for i in leftover:  # relax: take distinct entries even if closer than the floor
        if len(chosen) == k:
            break
        chosen.append(i)
    return chosen


def image_palette(entry: Sequence[dict], num_classes: int) -> list[tuple[int, int, int]]:
    """Build a visualization palette from the per-pass legends.

    Prompted classes get their group color and an implicit background remains
    black. Large taxonomies may reuse a color between independently decoded
    passes, which is harmless for decoding but can make the union render visually
    ambiguous.
    """
    palette: list[tuple[int, int, int]] = [(0, 0, 0)] * num_classes
    for g in entry:
        for i, c in enumerate(g["class_ids"]):
            palette[c] = tuple(g["colors"][i])
    return palette


def prompted_class_ids(semantic: base.SemanticSpec) -> tuple[int, ...]:
    """Return global trainIds that receive explicit colored prompts."""
    return tuple(
        class_id
        for class_id in range(len(semantic.classes))
        if class_id != semantic.implicit_background_id
    )


def no_class_entry(notation: str) -> list[dict]:
    """Legend pruning kept no class: one placeholder group with a black legend
    colour and no class id, so decoding, panels and caching run unchanged and
    every pixel becomes background. iter_repeat_predictions skips its FLUX call."""
    return [{"class_ids": [], "colors": [[0, 0, 0]], "color_names": ["black"], "style": notation,
             "prompt": "<legend pruning kept no class: FLUX skipped, predicted all background>"}]


def finalize_grouped_prediction(
    merged: np.ndarray, implicit_background_id: Optional[int]
) -> np.ndarray:
    """Turn pixels unclaimed by every foreground pass into explicit background."""
    if implicit_background_id is None:
        return merged
    finalized = merged.copy()
    finalized[finalized == 255] = implicit_background_id
    return finalized


def derive_entry(stem: str, seed: int, classes: Sequence[str], group_size: int,
                 min_delta_e: float, farthest: bool = True,
                 color_mapping: str = "p200_randomnegdraw",
                 class_ids: Optional[Sequence[int]] = None,
                 notation: str = "name", repeat_index: int = 0) -> list[dict]:
    """Deterministically derive one image's K group legends (GT-independent).

    ``class_ids`` defaults to the full taxonomy. A taxonomy larger than the named
    palette reuses colors *between* passes while retaining unique colors within a
    pass; each pass is decoded independently, so cross-pass reuse is unambiguous.
    """
    ids = list(range(len(classes))) if class_ids is None else list(class_ids)
    if not ids:
        raise ValueError("grouped evaluation needs at least one prompted class")
    if len(ids) != len(set(ids)) or any(c < 0 or c >= len(classes) for c in ids):
        raise ValueError("class_ids must contain unique valid taxonomy ids")

    if notation not in diversify.NOTATION_STYLES:
        raise ValueError(
            f"unknown notation {notation!r}; expected one of {diversify.NOTATION_STYLES}"
        )
    allowed = _named_pool(color_mapping, len(ids)) if notation == "name" else None
    color_capacity = (
        len(allowed) if allowed is not None
        else diversify.NAME_CAPACITY if notation == "name"
        else len(diversify.FULL_PALETTE)
    )
    if group_size > color_capacity:
        raise ValueError(
            f"group_size={group_size} exceeds the {color_capacity}-color {notation} prompt pool "
            f"for color_mapping={color_mapping!r}"
        )

    seed_key = f"{seed}:{stem}"
    if repeat_index:
        seed_key = f"{seed_key}:{repeat_index}"
    rng = random.Random(seed_key)
    rng.shuffle(ids)
    groups = [ids[i:i + group_size] for i in range(0, len(ids), group_size)]
    entries = []
    reuse_across_groups = len(ids) > color_capacity
    used: set[int] = set()
    for g in groups:
        names = [classes[c] for c in g]
        if notation != "name":
            sampled = diversify.sample_colors(
                rng, len(names), style=notation, min_delta_e=min_delta_e,
                color_mapping=color_mapping, n_present=len(names),
            )
        else:
            excluded = set() if reuse_across_groups else used
            if farthest:
                idxs = _farthest_point_colors(
                    rng, len(names), exclude=excluded, allowed=allowed
                )
            else:
                idxs = _sample_named_colors(
                    rng, len(names), min_delta_e, exclude=excluded, allowed=allowed
                )
            if not reuse_across_groups:
                used.update(idxs)
            sampled = [(tuple(int(v) for v in diversify.NAMED_RGB[i]),
                        diversify.NAMED_PALETTE[i]["name"]) for i in idxs]
        colors = [rgb for rgb, _ in sampled]
        color_names = [cname for _, cname in sampled]
        prompt = diversify.build_prompt(
            [(cls, rgb, cname) for cls, (rgb, cname) in zip(names, sampled)], notation, rng)
        entries.append({"class_ids": list(g), "colors": [list(c) for c in colors],
                        "color_names": color_names, "style": notation, "prompt": prompt})
    return entries


def load_pipeline(model, transformer_path, device, lora_path=None):
    """Shared production/benchmark loader; no offload or precision changes."""
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
    from common.flux2_runtime import load_lora_into_pipeline
    kwargs = {"torch_dtype": torch.bfloat16}
    if transformer_path:
        root, sub = base.resolve_transformer_load_args(transformer_path)
        tk = {"torch_dtype": torch.bfloat16}
        if sub is not None:
            tk["subfolder"] = sub
        kwargs["transformer"] = Flux2Transformer2DModel.from_pretrained(root, **tk)
    pipe = Flux2KleinPipeline.from_pretrained(model, **kwargs)
    load_lora_into_pipeline(pipe, lora_path)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def iter_repeat_predictions(get_pipeline, entries, image, *, batch_size, device,
                            generation_seed, steps, guidance_scale,
                            native_resolution=False):
    """Batch a repeat-major request stream, then restore independent repeats.

    Every request owns a fresh identically seeded generator. Only compatible
    prompts for THIS image share a call; no legends or masks merge across
    repeats here. Keep at most one inference batch plus one repeat's RGB
    results in memory.
    """
    if batch_size < 1 or any(not entry for entry in entries):
        raise ValueError("Need positive batch size and nonempty repeat entries")
    w, h = image.size
    requests = [
        base.PipelineBatchRequest(
            prompt=group["prompt"], image=image,
            generator=torch.Generator(device=device).manual_seed(generation_seed),
            height=h if native_resolution else None,
            width=w if native_resolution else None,
        )
        for entry in entries for group in entry if group["class_ids"]
    ]

    def predictions():
        for offset in range(0, len(requests), batch_size):
            batch = requests[offset:offset + batch_size]
            with torch.inference_mode():
                outputs = base.run_pipeline_batch(
                    get_pipeline(), batch, num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                )
            if len(outputs) != len(batch):
                raise RuntimeError("Pipeline returned a different number of images than requests")
            yield from outputs

    stream = predictions()
    for entry in entries:
        if all(group["class_ids"] for group in entry):
            yield entry, list(islice(stream, len(entry)))
        else:  # legend pruning kept no class (no_class_entry)
            yield entry, [next(stream) if group["class_ids"] else Image.new("RGB", image.size) for group in entry]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="black-forest-labs/FLUX.2-klein-base-9B")
    p.add_argument("--transformer_path", default=None)
    p.add_argument("--lora_path", default=None,
                   help="Optional Diffusers LoRA directory (pytorch_lora_weights.safetensors) applied on top of --model.")
    p.add_argument("--dataset", choices=sorted(base.DATASETS.keys()), default="cityscapes")
    p.add_argument("--data_root")
    p.add_argument("--cityscapes_root")
    p.add_argument("--split", default="val")
    p.add_argument("--condition", choices=base.ACDC_CONDITIONS, default=None)
    p.add_argument("--image_root")
    p.add_argument("--gt_root")
    p.add_argument("--image_suffixes", default=None)
    p.add_argument("--image_glob", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                   help="Validate and reuse complete single-repeat samples in out_dir; rebuild all metrics.")
    p.add_argument("--resume_from_out_dir", type=Path, default=None,
                   help="Completed grouped run to extend; reuse its cached repeats and generate only the suffix.")
    p.add_argument("--native_resolution", action="store_true")
    p.add_argument(
        "--generation_seed", type=int, default=0,
        help="Shard/order-independent diffusion seed; the same noise is reused across groups.",
    )
    # grouped multi-pass controls
    p.add_argument("--subset_seed", type=int, default=0,
                   help="Seed for the deterministic per-image partition + colors + prompts.")
    p.add_argument("--group_size", type=int, default=20,
                   help="Classes per group/pass; K = ceil(num_classes / group_size).")
    p.add_argument(
        "--group_batch_size", type=int, default=4,
        help="Number of same-image group prompts per FLUX call, spanning repeats when needed.",
    )
    p.add_argument(
        "--randomdraw_repeats", type=int, default=1,
        help="Independent grouped legend/partition repeats; cumulative predictions "
             "are majority-voted in train-ID space.",
    )
    p.add_argument("--color_min_delta_e", type=float, default=25.0)
    p.add_argument("--color_mapping", choices=diversify.COLOR_MAPPING_MODES, default="best64_maxnegdraw",
                   help="Select the shared training-time color pool/sampler.")
    p.add_argument(
        "--notation", choices=diversify.NOTATION_STYLES, default="rgb",
        help="Legend color notation. rgb/hex use the exact shared training palette sampler; "
             "name preserves the legacy grouped named-palette protocol.",
    )
    p.add_argument("--farthest_colors", action=argparse.BooleanOptionalAction, default=True,
                   help="Black-anchored max-min selection over the named palette "
                        "(default). --no-farthest_colors falls back to the "
                        "training-matched random dE>=25 draw (folder tag _rndcol).")
    p.add_argument("--legend_pruning_json", type=Path, nargs="+", default=None,
                   help="Legend pruning cache(s) from legend_pruning.py: prompt only the classes kept for "
                        "each image; several caches are united.")
    p.add_argument("--no_panels", action="store_true",
                   help="Skip the per-image input|complete GT|raw/snapped g0..gK-1|merged panel.")
    p.add_argument("--minimal_artifacts", action="store_true",
                   help=(
                       "Write only what cannot be rebuilt: raw group predictions, resume_cache, "
                       "legends, the final prediction and GT maps, and shard metrics. Skips the "
                       "per-group snapped maps, per-repeat and cumulative maps, vote-agreement "
                       "maps and per-repeat panels; the final per-image panel (input | GT | vote) "
                       "is still written unless --no_panels. Scores are unchanged, since every "
                       "metric is accumulated in memory; rebuild_grouped_artifacts.py regenerates "
                       "the skipped maps from resume_cache and the raw predictions."
                   ))
    p.add_argument("--verify_replayed_raws", action="store_true",
                   help=(
                       "Continuation only. Re-read and re-decode the source's raw group "
                       "predictions for the reused repeats and assert they match "
                       "resume_cache. Off by default: the cache is the authoritative "
                       "decode and is already digest-validated against the legend, image "
                       "and ground truth, so this only re-does work. Enabling it requires "
                       "the source's pred_rgb_raw/ to be unpacked into the target first."
                   ))
    p.add_argument("--dry_run", action="store_true",
                   help="Print resolved pairs + the first sample's derived groups and exit.")
    # sharding (same as the main evaluator)
    p.add_argument("--num_shards", type=int, default=None)
    p.add_argument("--shard_index", type=int, default=None)
    p.add_argument("--local_rank", type=int, default=None)
    p.add_argument("--shard_aggregate_timeout", type=int, default=3600)
    p.add_argument("--no_aggregate", action="store_true",
                   help="Write only this shard's metrics and exit (cross-job array sharding); "
                        "a separate aggregate job merges afterwards.")
    return p.parse_args()


def validate_resume_run(out_dir, args, semantic, num_samples):
    """Require matching run provenance, including legacy single-repeat shards.

    Shard metrics are the preferred record, but a run killed mid-flight commits
    none. Such a run still has the eval_config.json written before generation
    starts, so recovery does not require the dead run to have reached
    aggregation.
    """
    paths = sorted((out_dir / "shard_metrics").glob("rank_*.json"))
    sources = [(str(path), json.loads(path.read_text())) for path in paths]
    config_path = out_dir / "eval_config.json"
    if not sources and config_path.is_file():
        sources = [(str(config_path), json.loads(config_path.read_text()))]
    if not sources:
        raise ValueError(
            "--resume requires shard_metrics/ or eval_config.json for run provenance"
        )
    expected = {
        "dataset": args.dataset, "split": args.split, "prompt_style": "grouped",
        "steps": args.steps, "guidance_scale": args.guidance_scale,
        "limit": args.limit, "subset_seed": args.subset_seed,
        "group_size": args.group_size, "group_batch_size": args.group_batch_size,
        "color_mapping": args.color_mapping, "notation": args.notation,
        "generation_seed": args.generation_seed, "native_resolution": args.native_resolution,
        "num_total_samples": num_samples, "classes": list(semantic.classes),
        "palette": semantic.palette.tolist(),
    }
    for source, data in sources:
        for field, value in expected.items():
            if data.get(field) != value:
                raise ValueError(f"Cannot resume {source}: {field} differs from requested run")
        # Absent means a legacy single-repeat shard, which predates the field.
        if data.get("randomdraw_repeats", 1) != args.randomdraw_repeats:
            raise ValueError(
                f"Cannot resume {source}: randomdraw_repeats differs from requested run"
            )
        for field in ("model", "transformer_path", "lora_path"):
            if base.normalized_model_reference(data.get(field)) != base.normalized_model_reference(getattr(args, field)):
                raise ValueError(f"Cannot resume {source}: {field} differs from requested run")
        # Absent (every run before legend pruning) means unpruned.
        recorded = data.get("legend_pruning", (data.get("grouped_cache_provenance") or {}).get("legend_pruning"))
        if recorded != args.legend_pruning_provenance:
            raise ValueError(f"Cannot resume {source}: legend pruning differs from requested run")


def load_completed_prediction(out_dir, stem, gt, entry, semantic, args):
    """Reuse only complete samples with the exact derived legend and valid IDs.

    Missing/truncated output is regenerated. A valid but different legend or GT
    indicates another protocol/dataset and is an error, not a cache miss.
    """
    name = f"{stem}.png"
    legend_path = out_dir / "legends" / f"{stem}.json"
    required = [out_dir / folder / name for folder in (
        "pred_trainids", "pred_rgb_snapped", "ground_truth_trainids", "ground_truth_rgb"
    )]
    group_folders = ("pred_rgb_raw",) if args.minimal_artifacts else ("pred_rgb_raw", "pred_rgb_snapped")
    required += [out_dir / folder / f"g{i}" / name
                 for folder in group_folders for i in range(len(entry))]
    if not args.no_panels and not args.minimal_artifacts:
        required.append(out_dir / "panels" / name)
    if not legend_path.is_file() or any(not p.is_file() or p.stat().st_size == 0 for p in required):
        return None
    try:
        legend = json.loads(legend_path.read_text())
        with Image.open(out_dir / "pred_trainids" / name) as image:
            pred = np.array(image)
        with Image.open(out_dir / "ground_truth_trainids" / name) as image:
            saved_gt = np.array(image)
    except (OSError, ValueError):
        return None
    expected_groups = [dict(g, classes=[semantic.classes[c] for c in g["class_ids"]]) for g in entry]
    if (legend.get("groups") != expected_groups
            or legend.get("subset_seed") != args.subset_seed
            or legend.get("group_size") != args.group_size
            or legend.get("notation") != args.notation
            or legend.get("implicit_background_id") != semantic.implicit_background_id
            or legend.get("randomdraw_repeats", 1) != 1):
        raise ValueError(f"Cannot resume {stem}: saved legend differs from requested run")
    if not np.array_equal(saved_gt, gt):
        raise ValueError(f"Cannot resume {stem}: saved ground truth differs from current dataset")
    valid = pred < len(semantic.classes)
    if semantic.implicit_background_id is None:
        valid |= pred == 255
    if pred.dtype != np.uint8 or pred.shape != gt.shape or not valid.all():
        return None
    return pred


def write_json_atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def array_digest(value):
    value = np.asarray(value)
    return hashlib.sha256(str((value.shape, value.dtype.str)).encode() + value.tobytes()).hexdigest()


def grouped_cache_provenance(args, all_samples):
    payload = {
        "version": 1,
        "sample_ids_sha256": json_digest([sample.output_stem for sample in all_samples]),
        "condition": args.condition,
        "color_min_delta_e": args.color_min_delta_e,
        "farthest_colors": args.farthest_colors,
        "batching": "same_image_repeat_major_v1",
    }
    # Only pruned runs carry the key; unpruned provenance is unchanged.
    if getattr(args, "legend_pruning_provenance", None) is not None:
        payload["legend_pruning"] = args.legend_pruning_provenance
    return payload


def cache_metadata(entries, image, gt):
    return {"version": 1, "entries_sha256": json_digest(entries),
            "image_sha256": array_digest(np.asarray(image)), "gt_sha256": array_digest(gt)}


def validate_grouped_incremental_source(args, semantic, all_samples):
    source, target = args.resume_from_out_dir.resolve(), Path(args.out_dir).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Continuation source and target must be separate, non-nested directories")
    expected = {
        "dataset": args.dataset, "split": args.split, "prompt_style": "grouped",
        "steps": args.steps, "guidance_scale": args.guidance_scale, "limit": args.limit,
        "subset_seed": args.subset_seed, "group_size": args.group_size,
        "group_batch_size": args.group_batch_size, "color_mapping": args.color_mapping,
        "notation": args.notation, "native_resolution": args.native_resolution,
        "generation_seed": args.generation_seed,
        "classes": list(semantic.classes), "palette": semantic.palette.tolist(),
        "model": base.normalized_model_reference(args.model),
        "transformer_path": base.normalized_model_reference(args.transformer_path),
        "grouped_cache_provenance": grouped_cache_provenance(args, all_samples),
    }
    return base.validate_incremental_repeat_source(
        source, target, args.randomdraw_repeats, expected, len(all_samples),
    )


def load_grouped_repeat_cache(source, stem, entries, image, gt, semantic):
    path = source / "resume_cache" / f"{stem}.npz"
    predictions = base.load_repeat_trainids_cache(path, len(entries), gt.shape)
    with np.load(path, allow_pickle=False) as payload:
        if "metadata" not in payload or json.loads(str(payload["metadata"].item())) != cache_metadata(entries, image, gt):
            raise ValueError(f"Grouped cache image/GT/legend provenance differs: {path}")
    valid = predictions < len(semantic.classes)
    if semantic.implicit_background_id is None:
        valid |= predictions == 255
    if not valid.all():
        raise ValueError(f"Grouped cache contains invalid train IDs: {path}")
    return predictions


def repeat_record(rep, entry, classes):
    """Legend provenance for one repeat, identical on the replay and generate paths."""
    return {
        "repeat": rep + 1,
        "groups": [
            {
                "class_ids": group["class_ids"],
                "classes": [classes[class_id] for class_id in group["class_ids"]],
                "colors": group["colors"],
                "color_names": group["color_names"],
                "style": group["style"],
                "prompt": group["prompt"],
            }
            for group in entry
        ],
    }


def load_finished_repeats(out_dir, stem, entries, image, gt, semantic):
    """Return all cached repeats for one image, or None if it was never finished.

    The cache commit is the last write of a sample, so its presence means every
    artifact for that image is already on disk. A missing file is an ordinary
    miss. A file that disagrees with the derived legend, image or ground truth
    is an error, not a miss: that means another protocol wrote this directory.
    """
    if not (out_dir / "resume_cache" / f"{stem}.npz").is_file():
        return None
    return load_grouped_repeat_cache(out_dir, stem, entries, image, gt, semantic)


def replay_finished_repeats(cached, gt, num_classes, accumulator,
                            repeat_accumulators, cumulative_vote_accumulators):
    """Score one fully cached image exactly as generating it would have scored it.

    Voting is a pure function of the per-repeat train IDs held in the cache, so
    this reproduces the generation path's accumulator updates rather than
    approximating them.
    """
    for repeat_index, prediction in enumerate(cached):
        repeat_accumulators[repeat_index].update(prediction, gt)
    final_pred = cached[0]
    for vote_count, cumulative_pred, _agreement in base.cumulative_majority_vote_trainids(
        list(cached), num_classes
    ):
        final_pred = cumulative_pred
        if vote_count < 2:
            continue
        cumulative_vote_accumulators[vote_count - 2].update(cumulative_pred, gt)
    accumulator.update(final_pred, gt)


def prior_raw_path(source, target, stem, rep, group, source_repeats):
    relative = Path("pred_rgb_raw")
    if source_repeats > 1:
        relative /= f"repeat{rep + 1:02d}"
    relative /= f"g{group}/{stem}.png"
    # The sharded launcher restores archived raw predictions into the target.
    for path in (target / relative, source / relative):
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Prior grouped raw prediction missing: {relative}. Use the sharded launcher "
        "to restore pred_rgb_raw/ from artifacts.tar, or unpack that directory first."
    )


def replay_prior_predictions(source, target, stem, entries):
    for rep, entry in enumerate(entries):
        images = []
        for gi in range(len(entry)):
            path = prior_raw_path(source, target, stem, rep, gi, len(entries))
            destination = target / "pred_rgb_raw" / f"repeat{rep + 1:02d}" / f"g{gi}" / f"{stem}.png"
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(path, destination)
                except OSError:
                    shutil.copy2(path, destination)
            # Never save over a reused raw file: it may be hard-linked to source.
            with Image.open(destination) as raw:
                images.append(raw.convert("RGB"))
        yield entry, images


def main():
    args = parse_args()
    # --resume reuses images this target already finished at the full repeat
    # count; --resume_from_out_dir supplies repeats 1..N for the rest. A
    # target-cache hit short-circuits first, so no image is served by both.
    if args.dataset == "coco-object-80" and args.split != "val":
        raise ValueError("COCO-Object-80 evaluation is defined only for --split val.")
    if args.group_size < 1:
        raise ValueError(f"--group_size must be >= 1, got {args.group_size}")
    if args.group_batch_size < 1:
        raise ValueError(
            f"--group_batch_size must be >= 1, got {args.group_batch_size}"
        )
    if args.randomdraw_repeats < 1:
        raise ValueError(
            f"--randomdraw_repeats must be >= 1, got {args.randomdraw_repeats}"
        )
    shard = base.setup_sharding(args)

    data_root_arg = args.data_root or args.cityscapes_root
    if data_root_arg is None:
        raise ValueError("Provide --data_root (or --cityscapes_root for cityscapes).")
    layout = base.DATASETS[args.dataset]
    semantic = layout.semantic
    classes = semantic.classes
    num_classes = len(classes)
    prompt_ids = prompted_class_ids(semantic)
    num_groups = math.ceil(len(prompt_ids) / args.group_size)
    # COCO datasets decode GT through the super-category map; others use trainId PNGs.
    coco_category_map = (
        base.load_coco_category_map(Path(data_root_arg), semantic.coco_mask_type, args.split)
        if semantic.coco_mask_type is not None
        else None
    )

    def load_gt(path):
        return base.load_gt_mask(path, semantic, coco_category_map)

    out_dir = Path(args.out_dir)
    image_suffixes = base.parse_suffixes(args.image_suffixes)
    if image_suffixes is None:
        image_suffixes = base.COMMON_IMAGE_SUFFIXES if args.image_root else layout.image_suffixes

    all_samples = base.build_samples(
        layout=layout, data_root=Path(data_root_arg), split=args.split, condition=args.condition,
        image_root=Path(args.image_root) if args.image_root else None,
        gt_root=Path(args.gt_root) if args.gt_root else None,
        image_suffixes=image_suffixes, image_glob=args.image_glob, limit=args.limit,
    )
    samples = base.shard_samples(all_samples, shard)
    kept_by_stem, args.legend_pruning_provenance = None, None
    if args.legend_pruning_json:
        stems = [sample.output_stem for sample in all_samples]
        kept_by_stem = legend_pruning.load_kept(args.legend_pruning_json, list(classes), stems)
        args.legend_pruning_provenance = {
            "version": 1,
            "caches": sorted(path.name for path in args.legend_pruning_json),
            "kept_sha256": json_digest({stem: sorted(kept_by_stem[stem]) for stem in stems}),
        }
        print(f"Rank {shard.rank}: legend pruning keeps "
              f"{sum(len(kept_by_stem[s] & set(prompt_ids)) for s in stems) / max(len(stems), 1):.2f} "
              f"of {len(prompt_ids)} prompted classes per image ({', '.join(args.legend_pruning_provenance['caches'])})",
              flush=True)

    def derive(stem, repeat_index=0):
        """derive_entry for one image, over its legend-pruned classes if any."""
        ids = prompt_ids if kept_by_stem is None else tuple(c for c in prompt_ids if c in kept_by_stem[stem])
        if not ids:
            return no_class_entry(args.notation)
        return derive_entry(stem, args.subset_seed, classes, args.group_size, args.color_min_delta_e,
                            args.farthest_colors, color_mapping=args.color_mapping, class_ids=ids,
                            notation=args.notation, repeat_index=repeat_index)

    cache_provenance = grouped_cache_provenance(args, all_samples)
    resume_source, resume_repeats = None, 0
    if args.resume_from_out_dir is not None:
        resume_source, resume_repeats = validate_grouped_incremental_source(args, semantic, all_samples)
        print(f"Rank {shard.rank}: continuing from {resume_source}; reuse repeats "
              f"1-{resume_repeats}, generate {resume_repeats + 1}-{args.randomdraw_repeats} "
              f"for {len(samples)} images", flush=True)
    if args.resume:
        validate_resume_run(out_dir, args, semantic, len(all_samples))

    if args.dry_run:
        print(f"Dataset: {args.dataset}  Split: {args.split}")
        print(f"Groups: {num_groups} x <= {args.group_size} prompted classes over "
              f"{num_classes} scored classes (group_batch_size={args.group_batch_size}, "
              f"notation={args.notation}, repeats={args.randomdraw_repeats}, "
              f"farthest_colors={args.farthest_colors})")
        print(f"Shard: {shard.rank}/{shard.world_size} "
              f"({len(samples)} of {len(all_samples)} resolved image/GT pairs)")
        if resume_source is not None:
            for sample in tqdm(samples, desc="Validating grouped continuation"):
                with Image.open(sample.image_path) as source_image:
                    image = source_image.convert("RGB")
                entries = [derive(sample.output_stem, rep) for rep in range(resume_repeats)]
                load_grouped_repeat_cache(resume_source, sample.output_stem, entries,
                                          image, load_gt(sample.gt_path), semantic)
                if args.verify_replayed_raws:
                    for rep, entry in enumerate(entries):
                        for gi in range(len(entry)):
                            prior_raw_path(resume_source, out_dir, sample.output_stem, rep, gi, resume_repeats)
            print(f"Continuation: reuse repeats 1-{resume_repeats}; generate "
                  f"{resume_repeats + 1}-{args.randomdraw_repeats} for {len(samples)} images")
        if args.resume:
            reusable = 0
            for sample in tqdm(samples, desc="Validating cached predictions"):
                gt = load_gt(sample.gt_path)
                if args.randomdraw_repeats > 1:
                    with Image.open(sample.image_path) as source_image:
                        image = source_image.convert("RGB")
                    cached = load_finished_repeats(
                        out_dir, sample.output_stem,
                        [derive(sample.output_stem, rep) for rep in range(args.randomdraw_repeats)],
                        image, gt, semantic,
                    )
                else:
                    entry = derive(sample.output_stem)
                    cached = load_completed_prediction(out_dir, sample.output_stem,
                                                        gt, entry, semantic, args)
                reusable += cached is not None
            print(f"Resume audit: reuse={reusable}, generate={len(samples) - reusable}")
        print("First shard pairs:")
        for sample in samples[:5]:
            print(f"  image: {sample.image_path}")
            print(f"  gt:    {sample.gt_path}")
        if samples:
            entry = derive(samples[0].output_stem)
            print(f"Derived groups for {samples[0].output_stem}:")
            for gi, g in enumerate(entry):
                print(f"  g{gi}: {[classes[c] for c in g['class_ids']]}  style={g['style']}")
                print(f"      {g['prompt']}")
        base.cleanup_distributed(shard)
        return

    repeats = args.randomdraw_repeats

    # Keep every per-group directory below the scratch soft file-count limit.
    # A repeated run adds repeatNN between the artifact type and gK.
    def repeat_group_dirs(root: Path, repeat_index: int) -> list[Path]:
        repeat_root = root if repeats == 1 else root / f"repeat{repeat_index + 1:02d}"
        return [repeat_root / f"g{i}" for i in range(num_groups)]

    raw_dirs = [
        repeat_group_dirs(out_dir / "pred_rgb_raw", rep) for rep in range(repeats)
    ]
    snap_g_dirs = [
        repeat_group_dirs(out_dir / "pred_rgb_snapped", rep)
        for rep in range(repeats)
    ]
    snap_dir, tid_dir = out_dir / "pred_rgb_snapped", out_dir / "pred_trainids"
    gt_tid_dir = out_dir / "ground_truth_trainids"
    gt_rgb_dir = out_dir / "ground_truth_rgb"
    panel_dir, legends_dir = out_dir / "panels", out_dir / "legends"
    shard_metric_dir = out_dir / "shard_metrics"
    repeat_tid_dirs = [
        out_dir / "pred_trainids_repeats" / f"repeat{rep + 1:02d}"
        for rep in range(repeats)
    ] if repeats > 1 else []
    repeat_snap_dirs = [
        out_dir / "pred_rgb_snapped_repeats" / f"repeat{rep + 1:02d}"
        for rep in range(repeats)
    ] if repeats > 1 else []
    cumulative_tid_dirs = {
        count: out_dir / "pred_trainids_cumulative" / f"1to{count:02d}"
        for count in range(2, repeats + 1)
    }
    cumulative_snap_dirs = {
        count: out_dir / "pred_rgb_snapped_cumulative" / f"1to{count:02d}"
        for count in range(2, repeats + 1)
    }
    agreement_dirs = {
        count: out_dir / "vote_agreement" / f"1to{count:02d}"
        for count in range(2, repeats + 1)
    }
    agreement_color_dirs = {
        count: out_dir / "vote_agreement_color" / f"1to{count:02d}"
        for count in range(2, repeats + 1)
    }
    repeat_panel_dirs = [
        panel_dir if repeats == 1 else panel_dir / f"repeat{rep + 1:02d}"
        for rep in range(repeats)
    ]
    dirs = [
        *(d for per_repeat in raw_dirs for d in per_repeat),
        *(d for per_repeat in snap_g_dirs for d in per_repeat),
        snap_dir, tid_dir, gt_tid_dir, gt_rgb_dir, panel_dir, legends_dir,
        shard_metric_dir, out_dir / "resume_cache", *repeat_tid_dirs, *repeat_snap_dirs,
        *cumulative_tid_dirs.values(), *cumulative_snap_dirs.values(),
        *agreement_dirs.values(), *agreement_color_dirs.values(),
        *repeat_panel_dirs,
    ]
    if args.minimal_artifacts:
        skipped = {
            *(d for per_repeat in snap_g_dirs for d in per_repeat),
            *repeat_tid_dirs, *repeat_snap_dirs,
            *cumulative_tid_dirs.values(), *cumulative_snap_dirs.values(),
            *agreement_dirs.values(), *agreement_color_dirs.values(),
            *(d for d in repeat_panel_dirs if d != panel_dir),
        }
        dirs = [d for d in dirs if d not in skipped]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)

    if shard.rank == 0:
        # Committed before any generation so a run killed mid-flight stays
        # resumable. Aggregation rewrites this file with the full config once
        # the run finishes; the fields below are the ones --resume validates.
        write_json_atomic(out_dir / "eval_config.json", {
            "dataset": args.dataset,
            "split": args.split,
            "prompt_style": "grouped",
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "limit": args.limit,
            "subset_seed": args.subset_seed,
            "group_size": args.group_size,
            "group_batch_size": args.group_batch_size,
            "color_mapping": args.color_mapping,
            "notation": args.notation,
            "generation_seed": args.generation_seed,
            "native_resolution": args.native_resolution,
            "num_total_samples": len(all_samples),
            "classes": list(classes),
            "palette": semantic.palette.tolist(),
            "randomdraw_repeats": repeats,
            "model": str(args.model),
            "transformer_path": str(args.transformer_path),
            "lora_path": base.normalized_model_reference(args.lora_path),
            "minimal_artifacts": args.minimal_artifacts,
            "legend_pruning": args.legend_pruning_provenance,
        })

    # A shard with all samples cached needs no GPU model at all.
    pipe = None
    def get_pipeline():
        nonlocal pipe
        if pipe is None:
            pipe = load_pipeline(args.model, args.transformer_path, shard.device, args.lora_path)
        return pipe

    acc = base.SegmentationAccumulator(num_classes=num_classes, ignore_index=255)
    repeat_accumulators = [
        base.SegmentationAccumulator(num_classes=num_classes, ignore_index=255)
        for _ in range(repeats)
    ]
    cumulative_vote_accumulators = [
        base.SegmentationAccumulator(num_classes=num_classes, ignore_index=255)
        for _ in range(max(0, repeats - 1))
    ]

    resumed_samples = 0
    for sample in tqdm(samples, desc=f"Rank {shard.rank}/{shard.world_size}"):
        img = Image.open(sample.image_path).convert("RGB")
        w, h = img.size
        gt = load_gt(sample.gt_path)
        name = f"{sample.output_stem}.png"
        if args.resume and repeats > 1:
            # Multi-repeat samples are reused whole: the compact cache holds
            # every repeat, so nothing has to be re-decoded from the artifacts.
            cached_repeats = load_finished_repeats(
                out_dir, sample.output_stem,
                [derive(sample.output_stem, rep) for rep in range(repeats)],
                img, gt, semantic,
            )
            if cached_repeats is not None:
                replay_finished_repeats(
                    cached_repeats, gt, num_classes, acc,
                    repeat_accumulators, cumulative_vote_accumulators,
                )
                resumed_samples += 1
                continue
        elif args.resume:
            entry = derive(sample.output_stem)
            cached = load_completed_prediction(out_dir, sample.output_stem, gt, entry, semantic, args)
            if cached is not None:
                # A crash may have occurred after the artifacts but before the
                # compact cache commit. Recovery must leave this run extendable.
                base.save_repeat_trainids_cache(
                    out_dir / "resume_cache" / f"{sample.output_stem}.npz", [cached],
                    metadata=cache_metadata([entry], img, gt),
                )
                acc.update(cached, gt)
                repeat_accumulators[0].update(cached, gt)
                resumed_samples += 1
                continue

        repeat_predictions: list[np.ndarray] = []
        repeat_palettes: list[list[tuple[int, int, int]]] = []
        repeat_records: list[dict] = []

        entries = [derive(sample.output_stem, rep) for rep in range(repeats)]
        cached_repeats = None
        if resume_source is not None:
            cached_repeats = load_grouped_repeat_cache(
                resume_source, sample.output_stem, entries[:resume_repeats], img, gt, semantic,
            )
        new_predictions = iter_repeat_predictions(
            get_pipeline, entries[resume_repeats:], img, batch_size=args.group_batch_size,
            device=shard.device,
            generation_seed=base.stable_generation_seed(args.generation_seed, sample.output_stem),
            steps=args.steps, guidance_scale=args.guidance_scale,
            native_resolution=args.native_resolution,
        )
        # Reused repeats carry no images unless their raws are being verified, so
        # the loop below takes them straight from the cache.
        if args.verify_replayed_raws:
            prior_predictions = replay_prior_predictions(
                resume_source, out_dir, sample.output_stem, entries[:resume_repeats]
            )
        else:
            prior_predictions = ((entry, None) for entry in entries[:resume_repeats])
        prediction_stream = chain(prior_predictions, new_predictions)
        for rep, (entry, pred_rgbs) in enumerate(prediction_stream):
            img_pal = image_palette(entry, num_classes)
            repeat_palettes.append(img_pal)
            replayed = rep < resume_repeats

            if replayed and pred_rgbs is None:
                # Cache-only replay: resume_cache holds this repeat's validated
                # train IDs, and its per-repeat artifacts already exist in the source.
                merged_repeat = cached_repeats[rep]
                repeat_predictions.append(merged_repeat)
                repeat_accumulators[rep].update(merged_repeat, gt)
                repeat_records.append(repeat_record(rep, entry, classes))
                continue

            ids_maps: list[np.ndarray] = []
            dist_maps: list[np.ndarray] = []
            raws: list[Image.Image] = []
            group_snappeds: list[Image.Image] = []
            for gi, (group, pred_rgb) in enumerate(zip(entry, pred_rgbs)):
                if not replayed:
                    pred_rgb.save(raw_dirs[rep][gi] / name)
                raws.append(pred_rgb)

                dec, dist = diversify.rgb_to_trainids_via_legend_with_dist(
                    pred_rgb, [tuple(color) for color in group["colors"]]
                )
                if dec.shape != gt.shape:
                    size = (gt.shape[1], gt.shape[0])
                    dec = np.array(
                        Image.fromarray(dec).resize(size, Image.NEAREST),
                        dtype=np.uint8,
                    )
                    dist = np.array(
                        Image.fromarray(dist).resize(size, Image.NEAREST),
                        dtype=np.float32,
                    )
                group_snapped = Image.fromarray(
                    base.trainids_to_rgb(dec, palette=group["colors"])
                )
                if not replayed and not args.minimal_artifacts:
                    group_snapped.save(snap_g_dirs[rep][gi] / name)
                group_snappeds.append(group_snapped)

                full = np.full(gt.shape, 255, np.uint8)
                for local_id, class_id in enumerate(group["class_ids"]):
                    full[dec == local_id] = class_id
                ids_maps.append(full)
                dist_maps.append(dist)

            merged_repeat = finalize_grouped_prediction(
                diversify.merge_grouped_predictions(ids_maps, dist_maps),
                semantic.implicit_background_id,
            )
            if rep < resume_repeats:
                if not np.array_equal(merged_repeat, cached_repeats[rep]):
                    raise ValueError(f"Prior raw predictions disagree with cached repeat {rep + 1}: {sample.output_stem}")
                merged_repeat = cached_repeats[rep]
            repeat_predictions.append(merged_repeat)
            repeat_accumulators[rep].update(merged_repeat, gt)

            repeat_snapped = base.trainids_to_rgb(
                merged_repeat, palette=img_pal
            )
            if repeats > 1 and not replayed and not args.minimal_artifacts:
                Image.fromarray(merged_repeat).save(repeat_tid_dirs[rep] / name)
                Image.fromarray(repeat_snapped).save(repeat_snap_dirs[rep] / name)

            if not args.no_panels and not args.minimal_artifacts and not replayed:
                gt_repeat_rgb = base.trainids_to_rgb(gt, palette=img_pal)
                cells = [
                    (img, Image.LANCZOS),
                    (Image.fromarray(gt_repeat_rgb), Image.NEAREST),
                ]
                labels = ["input", "complete ground truth"]
                for gi, (raw, group_snapped) in enumerate(
                    zip(raws, group_snappeds)
                ):
                    cells += [
                        (raw, Image.LANCZOS),
                        (group_snapped, Image.NEAREST),
                    ]
                    labels += [f"raw g{gi}", f"snapped g{gi}"]
                cells += [(Image.fromarray(repeat_snapped), Image.NEAREST)]
                labels += [f"merged repeat {rep + 1}"]
                header = "\n".join(
                    f"[g{gi}] {group['prompt']}"
                    for gi, group in enumerate(entry)
                )
                base.make_panel(cells, labels=labels, header=header).save(
                    repeat_panel_dirs[rep] / name
                )

            repeat_records.append(repeat_record(rep, entry, classes))

        final_pred = repeat_predictions[0]
        for vote_count, cumulative_pred, agreement in (
            base.cumulative_majority_vote_trainids(
                repeat_predictions, num_classes
            )
        ):
            final_pred = cumulative_pred
            if vote_count < 2:
                continue
            # Every vote count is scored; visual artifacts are written only for
            # the vote counts this continuation extends (the rest exist in the source).
            cumulative_vote_accumulators[vote_count - 2].update(
                cumulative_pred, gt
            )
            if vote_count <= resume_repeats or args.minimal_artifacts:
                continue
            Image.fromarray(cumulative_pred).save(
                cumulative_tid_dirs[vote_count] / name
            )
            cumulative_rgb = base.trainids_to_rgb(
                cumulative_pred, palette=repeat_palettes[0]
            )
            Image.fromarray(cumulative_rgb).save(
                cumulative_snap_dirs[vote_count] / name
            )
            agreement_gray = np.rint(
                np.clip(agreement, 0.0, 1.0) * 255.0
            ).astype(np.uint8)
            Image.fromarray(agreement_gray).save(
                agreement_dirs[vote_count] / name
            )
            base.render_vote_agreement_visualization(
                agreement, vote_count
            ).save(agreement_color_dirs[vote_count] / name)

        # The top-level prediction is the final cumulative vote. Render it with
        # repeat 1's palette only for visualization; scoring uses train IDs.
        Image.fromarray(final_pred).save(tid_dir / name)
        final_rgb = base.trainids_to_rgb(
            final_pred, palette=repeat_palettes[0]
        )
        Image.fromarray(final_rgb).save(snap_dir / name)
        gt_rgb = base.trainids_to_rgb(gt, palette=repeat_palettes[0])
        Image.fromarray(gt).save(gt_tid_dir / name)
        Image.fromarray(gt_rgb).save(gt_rgb_dir / name)

        if not args.no_panels and (repeats > 1 or args.minimal_artifacts):
            base.make_panel(
                [
                    (img, Image.LANCZOS),
                    (Image.fromarray(gt_rgb), Image.NEAREST),
                    (Image.fromarray(final_rgb), Image.NEAREST),
                ],
                labels=[
                    "input",
                    "complete ground truth",
                    f"majority vote 1-{repeats}" if repeats > 1 else "prediction",
                ],
                header=(
                    f"Grouped random-repeat evaluation: {repeats} complete "
                    f"{num_groups}-pass predictions"
                ),
            ).save(panel_dir / name)

        legend_payload = {
            "subset_seed": args.subset_seed,
            "group_size": args.group_size,
            "notation": args.notation,
            "randomdraw_repeats": repeats,
            "implicit_background_id": semantic.implicit_background_id,
            "groups": repeat_records[0]["groups"],
        }
        if repeats > 1:
            legend_payload["repeats"] = repeat_records
        write_json_atomic(legends_dir / f"{sample.output_stem}.json", legend_payload)
        base.save_repeat_trainids_cache(
            out_dir / "resume_cache" / f"{sample.output_stem}.npz", repeat_predictions,
            metadata=cache_metadata(entries, img, gt),
        )

        # Exactly one update per image, using the final majority-voted map.
        acc.update(final_pred, gt)

    # write + aggregate shard metrics (same protocol as the main evaluator)
    metric_path = shard_metric_dir / f"rank_{shard.rank:05d}_of_{shard.world_size:05d}.json"
    metric_payload = {
        "rank": shard.rank,
        "world_size": shard.world_size,
        "dataset": args.dataset,
        "split": args.split,
        "prompt_style": "grouped",
        "visual_legend_mode": "none",
        "batch_size": args.group_batch_size,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "limit": args.limit,
        "subset_seed": args.subset_seed,
        "group_size": args.group_size,
        "group_batch_size": args.group_batch_size,
        "color_mapping": args.color_mapping,
        "notation": args.notation,
        "generation_seed": args.generation_seed,
        "native_resolution": args.native_resolution,
        "model": base.normalized_model_reference(args.model),
        "transformer_path": base.normalized_model_reference(args.transformer_path),
        "lora_path": base.normalized_model_reference(args.lora_path),
        "classes": list(classes),
        "palette": semantic.palette.tolist(),
        "num_total_samples": len(all_samples),
        "resumed_samples": resumed_samples,
        "generated_samples": len(samples) - resumed_samples,
        "grouped_cache_provenance": cache_provenance,
        "legend_pruning": args.legend_pruning_provenance,
        "minimal_artifacts": args.minimal_artifacts,
        "minimal_final_panels": args.minimal_artifacts and not args.no_panels,
        "resume_from_out_dir": str(resume_source) if resume_source is not None else None,
        "resume_from_repeats": resume_repeats,
        # Mutually exclusive: resume_repeats is 0 under --resume, resumed_samples
        # is 0 under a continuation.
        "reused_repeat_predictions": len(samples) * resume_repeats + resumed_samples * repeats,
        "randomdraw_repeats": repeats,
        "repeat_accumulators": [
            repeat_accumulator.to_dict()
            for repeat_accumulator in repeat_accumulators
        ],
        "cumulative_vote_accumulators": [
            cumulative_accumulator.to_dict()
            for cumulative_accumulator in cumulative_vote_accumulators
        ],
        **acc.to_dict(),
    }
    write_json_atomic(metric_path, metric_payload)
    print(f"Rank {shard.rank}: reused {resumed_samples}, generated {len(samples) - resumed_samples}")
    base.barrier(shard)
    if args.no_aggregate:
        print(f"Shard {shard.rank}/{shard.world_size} wrote metrics; aggregation deferred (--no_aggregate).")
        base.cleanup_distributed(shard)
        return
    if shard.rank != 0:
        base.cleanup_distributed(shard)
        return
    if shard.world_size > 1 and not shard.distributed:
        base.wait_for_shard_metrics(
            [shard_metric_dir / f"rank_{r:05d}_of_{shard.world_size:05d}.json" for r in range(shard.world_size)],
            args.shard_aggregate_timeout)
    # Direct and sharded runs must emit the same continuation-ready provenance
    # and every per-repeat/cumulative metric, not only the final voted score.
    import aggregate_semseg_shards as aggregate
    aggregate_argv = [
        "--out_dir", str(out_dir), "--dataset", args.dataset, "--split", args.split,
        "--prompt_style", "grouped", "--num_shards", str(shard.world_size),
        "--steps", str(args.steps), "--guidance_scale", str(args.guidance_scale),
        "--batch_size", str(args.group_batch_size), "--group_size", str(args.group_size),
        "--group_notation", args.notation, "--subset_seed", str(args.subset_seed),
        "--color_mapping", args.color_mapping, "--generation_seed", str(args.generation_seed),
        "--randomdraw_repeats", str(repeats),
    ]
    if args.limit is not None:
        aggregate_argv += ["--limit", str(args.limit)]
    if args.native_resolution:
        aggregate_argv += ["--native_resolution"]
    aggregate.main(aggregate.parse_args(aggregate_argv))
    base.cleanup_distributed(shard)


if __name__ == "__main__":
    main()
