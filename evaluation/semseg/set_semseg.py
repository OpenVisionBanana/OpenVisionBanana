#!/usr/bin/env python3
"""Dynamic-subset semseg evaluation (the eval analogue of training diversification).

Where inference_partial_flux2_klein_img2img.py evaluates the full 19-class task
(styles: fixed / randomdraw), this evaluates the *diversified* task exactly as the
model was trained: for each image we look at the classes present in its GT,
subsample M of them (the same M-rule as dataset_prep: N==1 -> that one; N>1 -> a
strict proper subset), assign each a random color from the shared 200-color
palette, and build a randomized prompt (random notation + template) carrying that
legend. The prediction is decoded against the image's own legend and scored only
over the selected classes (everything else is ignore), accumulated per Cityscapes
class across images.

Reproducibility: the per-image (selected classes, colors, prompt) "eval set" is
derived deterministically from --subset-seed + the image id, and written to
eval_set.json. Pass --eval-set <path> to REUSE a previously generated set (so a
metric is exactly reproducible across runs / models); --regenerate forces a rebuild.

Reuses the full pipeline / sharding / metric machinery from the main evaluator.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
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
from common.semseg import diversify, taxonomy  # noqa: E402

def derive_entry(gt: np.ndarray, stem: str, seed: int, min_delta_e: float, classes: Sequence[str],
                 color_mapping: str = "p200_randomnegdraw",
                 classes_per_image: Optional[int] = None):
    """Deterministically derive one image's dynamic-subset eval spec from its GT.

    Returns {selected (dataset class indices), colors, style, prompt}, or None if
    the image has no foreground class to evaluate. ``color_mapping`` selects the
    palette draw (diversify.COLOR_MAPPING_MODES) -- match the checkpoint's training
    mode so the legend colors come from the pool the model was trained on.

    ``classes_per_image`` (None by default) forces a fixed subset size instead of
    the training M-rule; None keeps the M-rule and its RNG draw order."""
    rng = random.Random(f"{seed}:{stem}")
    num_classes = len(classes)
    present = sorted(int(v) for v in np.unique(gt) if 0 <= int(v) < num_classes)
    if not present:
        return None
    if classes_per_image is None:
        selected = taxonomy.select_subset(present, rng)    # same M-rule as training
    else:
        # Fixed K, clamped to the classes present: K >= len(present) degenerates
        # to the full present set (which the M-rule never produces).
        selected = rng.sample(present, min(int(classes_per_image), len(present)))
    names = [classes[c] for c in selected]
    prompt, colors, style, _ = diversify.build_legend(names, rng, min_delta_e=min_delta_e,
                                                      color_mapping=color_mapping,
                                                      n_present=len(present))
    return {"selected": list(selected), "colors": [list(c) for c in colors],
            "style": style, "prompt": prompt}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="black-forest-labs/FLUX.2-klein-base-9B")
    p.add_argument("--transformer_path", default=None)
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
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--native_resolution", action="store_true")
    p.add_argument("--generation_seed", type=int, default=0,
                   help="Base generation seed (default 0, the fixed value the random-draw evaluation "
                        "uses). Each image receives a stable derived seed (base.stable_generation_seed), "
                        "independent of shard count and ordering, so a run reproduces itself. A negative "
                        "value disables seeding (unseeded generation, recorded as generation_seed null).")
    # dynamic-subset controls
    p.add_argument("--subset_seed", type=int, default=0,
                   help="Seed for the deterministic per-image subset + colors + prompt.")
    p.add_argument("--classes_per_image", type=int, default=None,
                   help="Force exactly this many classes per image (clamped to the number "
                        "present in the GT). Default None keeps the training M-rule: a "
                        "random strict subset of the present classes, so the count varies "
                        "per image. A fixed-K run is a DIFFERENT protocol and its scores "
                        "are not comparable with default-rule scores.")
    p.add_argument("--color_min_delta_e", type=float, default=25.0)
    p.add_argument("--color_mapping", choices=diversify.COLOR_MAPPING_MODES, default="p200_randomnegdraw",
                   help="Palette draw for the per-image legend (same modes as dataset prep); match the "
                        "checkpoint's training color_mapping.")
    p.add_argument("--eval-set", dest="eval_set", default=None,
                   help="Path to an eval_set.json to REUSE (reproducible). Default: <out_dir>/eval_set.json.")
    p.add_argument("--regenerate", action="store_true",
                   help="Re-derive the eval set even if the --eval-set file exists.")
    # sharding (same as the main evaluator)
    p.add_argument("--num_shards", type=int, default=None)
    p.add_argument("--shard_index", type=int, default=None)
    p.add_argument("--local_rank", type=int, default=None)
    p.add_argument("--shard_aggregate_timeout", type=int, default=3600)
    p.add_argument("--no_aggregate", action="store_true",
                   help="Write only this shard's metrics and exit (cross-job array sharding); "
                        "a separate aggregate job merges afterwards.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.generation_seed is not None and args.generation_seed < 0:
        args.generation_seed = None   # explicit opt-out: unseeded generation
    if args.dataset == "coco-object-80":
        raise ValueError(
            "COCO-Object-80 requires grouped_semseg.py / PROMPT_STYLE=grouped; "
            "set mode uses a GT-present-class oracle and is not the benchmark protocol."
        )
    shard = base.setup_sharding(args)

    data_root_arg = args.data_root or args.cityscapes_root
    if data_root_arg is None:
        raise ValueError("Provide --data_root (or --cityscapes_root for cityscapes).")
    layout = base.DATASETS[args.dataset]
    semantic = layout.semantic
    classes = semantic.classes
    num_classes = len(classes)
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

    raw_dir, snap_dir, tid_dir = out_dir / "pred_rgb_raw", out_dir / "pred_rgb_snapped", out_dir / "pred_trainids"
    panel_dir, legends_dir = out_dir / "panels", out_dir / "legends"
    shard_metric_dir = out_dir / "shard_metrics"
    for d in (raw_dir, snap_dir, tid_dir, panel_dir, legends_dir, shard_metric_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Cross-job shards have no barrier, so each derives its specs on the fly;
    # derive_entry is deterministic in (subset_seed, stem).
    sharded_xjob = args.num_shards is not None or args.shard_index is not None
    eval_set_path = Path(args.eval_set) if args.eval_set else (out_dir / "eval_set.json")
    entries: dict[str, dict] = {}
    reuse = False
    if sharded_xjob:
        print(f"Cross-job shard {shard.rank}/{shard.world_size}: deriving subsets on the fly (no shared eval_set).")
    else:
        # Reuse a previously generated eval set if asked; else derive deterministically.
        reuse = eval_set_path.exists() and not args.regenerate
        if reuse:
            entries = json.loads(eval_set_path.read_text()).get("entries", {})
            print(f"Reusing eval set: {eval_set_path} ({len(entries)} entries)")
        elif shard.rank == 0:
            for s in tqdm(all_samples, desc="Deriving eval set"):
                e = derive_entry(load_gt(s.gt_path), s.output_stem, args.subset_seed, args.color_min_delta_e,
                                 classes, color_mapping=args.color_mapping,
                                 classes_per_image=args.classes_per_image)
                if e is not None:
                    entries[s.output_stem] = e
            eval_set_path.write_text(json.dumps(
                {"subset_seed": args.subset_seed, "dataset": args.dataset, "split": args.split,
                 "classes_per_image": args.classes_per_image,
                 "color_min_delta_e": args.color_min_delta_e, "palette_size": len(diversify.FULL_PALETTE),
                 "color_mapping": args.color_mapping,
                 "entries": entries}, indent=2))
            print(f"Wrote eval set: {eval_set_path} ({len(entries)} entries)")
        base.barrier(shard)
        if not entries:  # non-rank0 freshly-generated: load what rank0 wrote
            entries = json.loads(eval_set_path.read_text()).get("entries", {})

    # pipeline (diffusers imported here, mirroring base.main(); base imports it
    # lazily inside its own main(), so it is NOT a base module attribute).
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
    if args.transformer_path:
        root, sub = base.resolve_transformer_load_args(args.transformer_path)
        tk = {"torch_dtype": torch.bfloat16}
        if sub is not None:
            tk["subfolder"] = sub
        transformer = Flux2Transformer2DModel.from_pretrained(root, **tk)
        pipe = Flux2KleinPipeline.from_pretrained(args.model, transformer=transformer, torch_dtype=torch.bfloat16)
    else:
        pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.to(shard.device)
    pipe.set_progress_bar_config(disable=True)

    acc = base.SegmentationAccumulator(num_classes=num_classes, ignore_index=255)
    for sample in tqdm(samples, desc=f"Rank {shard.rank}/{shard.world_size}"):
        entry = entries.get(sample.output_stem) or derive_entry(
            load_gt(sample.gt_path), sample.output_stem, args.subset_seed, args.color_min_delta_e,
            classes, color_mapping=args.color_mapping,
            classes_per_image=args.classes_per_image)
        if entry is None:
            continue
        selected, colors, prompt = entry["selected"], [tuple(c) for c in entry["colors"]], entry["prompt"]

        img = Image.open(sample.image_path).convert("RGB")
        w, h = img.size
        gkwargs = dict(prompt=prompt, image=img, num_inference_steps=args.steps, guidance_scale=args.guidance_scale)
        if args.native_resolution:
            gkwargs.update({"height": h, "width": w})
        if args.generation_seed is not None:
            gkwargs["generator"] = torch.Generator(device=shard.device).manual_seed(
                base.stable_generation_seed(args.generation_seed, sample.output_stem))
        with torch.inference_mode():
            pred_rgb = pipe(**gkwargs).images[0]

        name = f"{sample.output_stem}.png"
        pred_rgb.save(raw_dir / name)

        # decode against this image's legend (-> 0..M-1 / 255), map back to cityscapes ids
        dec = base.rgb_to_trainids_via_legend(pred_rgb, colors)
        gt = load_gt(sample.gt_path)
        if dec.shape != gt.shape:
            dec = np.array(Image.fromarray(dec).resize((gt.shape[1], gt.shape[0]), Image.NEAREST), dtype=np.uint8)
        pred_ds = np.full(gt.shape, 255, np.uint8)
        for i, c in enumerate(selected):
            pred_ds[dec == i] = c
        gt_ds = gt.copy()
        gt_ds[~np.isin(gt, selected)] = 255   # score only the selected classes

        Image.fromarray(pred_ds).save(tid_dir / name)
        # Visualize GT / snapped in this image's random legend colors (not the
        # fixed Cityscapes palette) so all tiles share one color language.
        img_pal = np.zeros((num_classes, 3), dtype=np.uint8)
        for i, c in enumerate(selected):
            img_pal[c] = colors[i]
        snapped = base.trainids_to_rgb(pred_ds, palette=img_pal)
        Image.fromarray(snapped).save(snap_dir / name)
        panel = base.make_panel(
            [(img, Image.LANCZOS),
             (Image.fromarray(base.trainids_to_rgb(gt_ds, palette=img_pal)), Image.NEAREST),
             (pred_rgb, Image.LANCZOS),
             (Image.fromarray(snapped), Image.NEAREST)],
            labels=base.PANEL_LABELS, header=prompt)
        panel.save(panel_dir / name)
        with open(legends_dir / f"{sample.output_stem}.json", "w") as f:
            json.dump({"selected": [classes[c] for c in selected],
                       "colors": entry["colors"], "style": entry["style"], "prompt": prompt}, f, indent=2)
        acc.update(pred_ds, gt_ds)

    # write + aggregate shard metrics (same protocol as the main evaluator)
    metric_path = shard_metric_dir / f"rank_{shard.rank:05d}_of_{shard.world_size:05d}.json"
    # Provenance aggregate_semseg_shards.py cross-checks against its own flags (only
    # keys it expects for prompt_style=set: no legend_seed / color_mapping here).
    provenance = {
        "dataset": args.dataset, "split": args.split, "prompt_style": "set",
        "model": args.model, "transformer_path": args.transformer_path,
        "steps": args.steps, "guidance_scale": args.guidance_scale, "limit": args.limit,
        "subset_seed": args.subset_seed, "classes_per_image": args.classes_per_image,
        "generation_seed": args.generation_seed, "native_resolution": bool(args.native_resolution),
    }
    metric_path.write_text(json.dumps({"rank": shard.rank, "world_size": shard.world_size,
                                       **provenance, **acc.to_dict()}, indent=2))
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
    merged = base.SegmentationAccumulator(num_classes=num_classes, ignore_index=255)
    for r in range(shard.world_size):
        merged.merge(base.SegmentationAccumulator.from_dict(
            json.loads((shard_metric_dir / f"rank_{r:05d}_of_{shard.world_size:05d}.json").read_text())))
    results = merged.compute()

    out_json = out_dir / f"{args.dataset}_{args.split}_results.json"
    out_json.write_text(json.dumps({**{k: float(v) for k, v in results.items()}, "prompt_style": "set"}, indent=2))
    (out_dir / "eval_config.json").write_text(json.dumps({
        "prompt_style": "set", "steps": args.steps, "guidance_scale": args.guidance_scale,
        "limit": args.limit, "num_images": merged.num_samples, "subset_seed": args.subset_seed,
        "classes_per_image": args.classes_per_image, "generation_seed": args.generation_seed,
        "color_mapping": args.color_mapping,
        "color_min_delta_e": args.color_min_delta_e, "palette_size": len(diversify.FULL_PALETTE),
        "eval_set": str(eval_set_path), "reused_eval_set": reuse,
    }, indent=2))

    print("\nSet (dynamic-subset) Evaluation Results")
    print("=" * 50)
    print(f"Dataset: {args.dataset}  Split: {args.split}  Images: {merged.num_samples}  Shards: {shard.world_size}")
    for k, v in results.items():
        print(f"{k}: {v:.2f}")
    print(f"Saved results to: {out_json}\nEval set: {eval_set_path}")
    base.cleanup_distributed(shard)


if __name__ == "__main__":
    main()
