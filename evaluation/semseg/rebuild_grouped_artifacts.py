#!/usr/bin/env python3
"""Rebuild the visualization maps a grouped run skipped with --minimal_artifacts.

grouped_semseg.py --minimal_artifacts keeps only what cannot be rebuilt: the raw
group predictions, resume_cache/<image>.npz (every repeat's decoded train IDs),
legends/<image>.json, the final maps, the final per-image panel and the shard
metrics. Everything else is a pure function of those files, and this script
regenerates it on CPU with the same calls the evaluator uses, into the same layout:

  group_snapped  pred_rgb_snapped[/repeatKK]/gG/<image>.png  (raw + legend colors)
  repeats        pred_trainids_repeats/repeatKK/, pred_rgb_snapped_repeats/repeatKK/
  cumulative     pred_trainids_cumulative/1toKK/, pred_rgb_snapped_cumulative/1toKK/
  agreement      vote_agreement/1toKK/, vote_agreement_color/1toKK/

Per-repeat panels are not rebuilt (they need the input image). Restrict the work with
--stems/--limit, --what and --counts; by default every image, kind and prefix is
written. Existing files are overwritten with identical content.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))  # repo root for common.*
import inference_partial_flux2_klein_img2img as base  # noqa: E402
from common.semseg import diversify  # noqa: E402
from grouped_semseg import image_palette  # noqa: E402

KINDS = ("group_snapped", "repeats", "cumulative", "agreement")


def repeat_groups(legend: dict, repeats: int) -> list[list[dict]]:
    """Per-repeat group legends (class_ids, colors) as the evaluator recorded them."""
    if repeats == 1:
        return [legend["groups"]]
    records = sorted(legend["repeats"], key=lambda record: record["repeat"])
    if [record["repeat"] for record in records] != list(range(1, repeats + 1)):
        raise ValueError(f"legend lists repeats {[r['repeat'] for r in records]}, expected 1..{repeats}")
    return [record["groups"] for record in records]


def rebuild_one(task) -> tuple[str, int]:
    run, sources, dest, stem, repeats, num_classes, kinds, counts = task
    name = f"{stem}.png"
    legend = json.loads((run / "legends" / f"{stem}.json").read_text())
    groups = repeat_groups(legend, repeats)
    with Image.open(run / "ground_truth_trainids" / name) as image:
        gt_shape = np.array(image).shape
    preds = list(base.load_repeat_trainids_cache(run / "resume_cache" / f"{stem}.npz", repeats, gt_shape))

    def save(array_or_image, *parts):
        path = dest.joinpath(*parts, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        image = array_or_image if isinstance(array_or_image, Image.Image) else Image.fromarray(array_or_image)
        image.save(path)

    def repeat_dir(rep):  # repeatKK level exists only for multi-repeat runs
        return () if repeats == 1 else (f"repeat{rep + 1:02d}",)

    missing_raws = 0
    if "group_snapped" in kinds:
        for rep, entry in enumerate(groups):
            for gi, group in enumerate(entry):
                # A continuation stores raws only for the repeats it generated; the
                # reused ones stay in the run it continued from (--sources).
                raw_path = next((d / "pred_rgb_raw" / Path(*repeat_dir(rep)) / f"g{gi}" / name
                                 for d in (run, *sources)
                                 if (d / "pred_rgb_raw" / Path(*repeat_dir(rep)) / f"g{gi}" / name).is_file()), None)
                if raw_path is None:
                    missing_raws += 1
                    continue
                with Image.open(raw_path) as raw:
                    dec, _ = diversify.rgb_to_trainids_via_legend_with_dist(
                        raw.convert("RGB"), [tuple(color) for color in group["colors"]]
                    )
                if dec.shape != gt_shape:
                    dec = np.array(Image.fromarray(dec).resize((gt_shape[1], gt_shape[0]), Image.NEAREST),
                                   dtype=np.uint8)
                save(base.trainids_to_rgb(dec, palette=group["colors"]), "pred_rgb_snapped", *repeat_dir(rep), f"g{gi}")

    if "repeats" in kinds and repeats > 1:
        for rep, (pred, entry) in enumerate(zip(preds, groups)):
            save(pred, "pred_trainids_repeats", f"repeat{rep + 1:02d}")
            save(base.trainids_to_rgb(pred, palette=image_palette(entry, num_classes)),
                 "pred_rgb_snapped_repeats", f"repeat{rep + 1:02d}")

    if {"cumulative", "agreement"} & set(kinds) and repeats > 1:
        first_palette = image_palette(groups[0], num_classes)
        for vote_count, cumulative_pred, agreement in base.cumulative_majority_vote_trainids(preds, num_classes):
            if vote_count < 2 or (counts and vote_count not in counts):
                continue
            prefix = f"1to{vote_count:02d}"
            if "cumulative" in kinds:
                save(cumulative_pred, "pred_trainids_cumulative", prefix)
                save(base.trainids_to_rgb(cumulative_pred, palette=first_palette), "pred_rgb_snapped_cumulative", prefix)
            if "agreement" in kinds:
                save(np.rint(np.clip(agreement, 0.0, 1.0) * 255.0).astype(np.uint8), "vote_agreement", prefix)
                save(base.render_vote_agreement_visualization(agreement, vote_count), "vote_agreement_color", prefix)
    return stem, missing_raws


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", type=Path, required=True, help="Grouped run directory (holds eval_config.json).")
    p.add_argument("--dest", type=Path, default=None, help="Write here instead of --out_dir (same layout).")
    p.add_argument("--sources", type=Path, nargs="*", default=[],
                   help="Earlier runs of a continuation chain, searched for raws of reused repeats "
                        "(extract their pred_rgb_raw/ from artifacts.tar first).")
    p.add_argument("--stems", default=None, help="Comma-separated image stems, or @file with one stem per line.")
    p.add_argument("--limit", type=int, default=None, help="First N images (sorted) when --stems is not given.")
    p.add_argument("--what", default=",".join(KINDS), help=f"Comma-separated subset of {', '.join(KINDS)}.")
    p.add_argument("--counts", default=None, help="Comma-separated vote counts to write (default: 2..K).")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    run = args.out_dir.resolve()
    config = json.loads((run / "eval_config.json").read_text())
    if config.get("prompt_style") != "grouped":
        raise SystemExit(f"{run} is not a grouped run (prompt_style={config.get('prompt_style')!r})")
    repeats = int(config.get("randomdraw_repeats", 1))
    num_classes = len(config["classes"])
    kinds = [kind for kind in args.what.split(",") if kind]
    if unknown := set(kinds) - set(KINDS):
        raise SystemExit(f"unknown --what {sorted(unknown)}; choose from {KINDS}")
    counts = {int(c) for c in args.counts.split(",")} if args.counts else None

    if args.stems:
        text = Path(args.stems[1:]).read_text() if args.stems.startswith("@") else args.stems.replace(",", "\n")
        stems = [s.strip() for s in text.splitlines() if s.strip()]
    else:
        stems = sorted(path.stem for path in (run / "resume_cache").glob("*.npz"))[: args.limit]
    dest = (args.dest or run).resolve()
    sources = tuple(source.resolve() for source in args.sources)
    tasks = [(run, sources, dest, stem, repeats, num_classes, kinds, counts) for stem in stems]
    missing = 0
    with Pool(args.workers) as pool:
        for done, (_, missing_raws) in enumerate(pool.imap_unordered(rebuild_one, tasks, chunksize=4), 1):
            missing += missing_raws
            if done % 500 == 0 or done == len(tasks):
                print(f"rebuilt {done}/{len(tasks)}", flush=True)
    if missing:
        print(f"skipped {missing} per-group snapped maps whose raw prediction was not found; "
              f"pass the chain's earlier runs with --sources", flush=True)


if __name__ == "__main__":
    main()
