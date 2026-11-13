#!/usr/bin/env python3
"""Merge per-shard semseg metrics into the final mIoU result.

Used with cross-job array sharding: the shard array runs the evaluator with
--no_aggregate, so each shard only writes shard_metrics/rank_<i>_of_<N>.json into
the shared OUT_DIR. This (CPU-only, afterok) job sums those SegmentationAccumulator
dumps and writes <dataset>_<split>_results.json + eval_config.json -- exactly the
files the non-sharded evaluator would have produced.

Usage:
  python aggregate_semseg_shards.py --out_dir <dir> --dataset coco-stuff \
      --split val --prompt_style fixed [--num_shards 8]
  python aggregate_semseg_shards.py --out_dir <dir> --dataset cityscapes \
      --split val --prompt_style randomdraw --randomdraw_repeats 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))  # repo root for common.*
import inference_partial_flux2_klein_img2img as base  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True, help="Shared eval output dir holding shard_metrics/.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--prompt_style", default="fixed")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=None,
                   help="Expected shard count; if set, wait for all rank files (afterok makes this instant).")
    p.add_argument("--shard_aggregate_timeout", type=int, default=1800)
    # Recorded into eval_config.json for provenance (optional).
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-GPU generation batch size recorded in the shard metrics.",
    )
    p.add_argument("--legend_seed", type=int, default=0)
    p.add_argument(
        "--color_mapping",
        choices=base.diversify.COLOR_MAPPING_MODES,
        default="p200_randomnegdraw",
    )
    p.add_argument("--randomdraw_repeats", type=int, default=1)
    p.add_argument("--subset_seed", type=int, default=None)
    p.add_argument("--group_size", type=int, default=None)
    p.add_argument(
        "--group_notation",
        choices=base.diversify.NOTATION_STYLES,
        default="name",
        help="Grouped legend notation expected in every shard.",
    )
    p.add_argument("--generation_seed", type=int, default=0)
    p.add_argument(
        "--native_resolution",
        action="store_true",
        help="Record/validate that generation used native input resolution.",
    )
    p.add_argument(
        "--visual_legend_mode",
        choices=("none", "swatches", "class_labels", "full_labels"),
        default="none",
    )
    return p.parse_args(argv)


def main(args=None) -> None:
    args = parse_args() if args is None else args
    # set_semseg.py: a negative --generation_seed means unseeded generation, recorded null.
    if args.generation_seed is not None and args.generation_seed < 0:
        args.generation_seed = None
    out_dir = Path(args.out_dir)
    shard_metric_dir = out_dir / "shard_metrics"

    if args.dataset == "coco-object-80" and args.prompt_style != "grouped":
        raise SystemExit("COCO-Object-80 requires --prompt_style grouped.")
    if args.dataset == "coco-object-80" and args.split != "val":
        raise SystemExit("COCO-Object-80 evaluation is defined only for --split val.")
    if args.prompt_style == "grouped" and args.group_size is None:
        raise SystemExit("Grouped aggregation requires --group_size.")
    uses_repeat_payloads = (
        args.prompt_style == "randomdraw"
        or (args.prompt_style == "grouped" and args.randomdraw_repeats > 1)
    )

    if args.limit is not None and args.limit <= 0:
        raise SystemExit(f"--limit must be positive, got {args.limit}.")
    if args.randomdraw_repeats <= 0:
        raise SystemExit(
            f"--randomdraw_repeats must be positive, got {args.randomdraw_repeats}."
        )
    if args.batch_size <= 0:
        raise SystemExit(f"--batch_size must be positive, got {args.batch_size}.")

    if args.num_shards:
        expected = [shard_metric_dir / f"rank_{r:05d}_of_{args.num_shards:05d}.json" for r in range(args.num_shards)]
        base.wait_for_shard_metrics(expected, args.shard_aggregate_timeout)
        paths = expected
    else:
        paths = sorted(shard_metric_dir.glob("rank_*_of_*.json"))
    if not paths:
        raise SystemExit(f"No shard metric files under {shard_metric_dir}")

    merged = None
    merged_repeats = None
    merged_cumulative_votes = None
    num_images = 0
    shard_total_counts = set()
    merged_run_provenance = None
    minimal_artifact_shards = 0
    minimal_final_panel_shards = 0
    reused_repeat_predictions = 0
    resumed_samples = 0
    generated_samples = 0
    classes_per_image_values = set()   # set style: fixed K recorded by set_semseg.py shards
    for path in paths:
        data = json.loads(Path(path).read_text())
        for field, expected_value in (
            ("dataset", args.dataset),
            ("split", args.split),
            ("prompt_style", args.prompt_style),
            ("visual_legend_mode", args.visual_legend_mode),
        ):
            if data.get(field) is not None and data[field] != expected_value:
                raise SystemExit(
                    f"Shard {path.name} records {field}={data[field]!r}, expected {expected_value!r}."
                )
        shard_batch_size = data.get("batch_size", 1)
        if shard_batch_size != args.batch_size:
            raise SystemExit(
                f"Shard {path.name} records batch_size={shard_batch_size!r}, "
                f"expected {args.batch_size!r}."
            )
        if args.prompt_style == "grouped":
            shard_notation = data.get("notation", "name")
            if shard_notation != args.group_notation:
                raise SystemExit(
                    f"Shard {path.name} records notation={shard_notation!r}, "
                    f"expected {args.group_notation!r}."
                )
        if uses_repeat_payloads:
            shard_repeat_count = data.get("randomdraw_repeats")
            if shard_repeat_count != args.randomdraw_repeats:
                raise SystemExit(
                    f"Shard {path.name} records randomdraw_repeats="
                    f"{shard_repeat_count!r}, expected {args.randomdraw_repeats!r}."
                )
        for field, expected_value in (
            ("steps", args.steps),
            ("guidance_scale", args.guidance_scale),
            ("limit", args.limit),
            ("legend_seed", args.legend_seed if args.prompt_style == "randomdraw" else None),
            ("color_mapping", args.color_mapping if args.prompt_style in ("randomdraw", "grouped") else None),
            ("subset_seed", args.subset_seed if args.prompt_style in ("set", "grouped") else None),
            ("group_size", args.group_size if args.prompt_style == "grouped" else None),
            ("group_batch_size", args.batch_size if args.prompt_style == "grouped" else None),
            ("generation_seed", args.generation_seed),
            ("native_resolution", bool(args.native_resolution)),
        ):
            if field in data and data[field] != expected_value:
                raise SystemExit(
                    f"Shard {path.name} records {field}={data[field]!r}, "
                    f"expected {expected_value!r}."
                )
        shard_run_provenance = {
            "model": data.get("model"),
            "transformer_path": data.get("transformer_path"),
            "native_resolution": data.get(
                "native_resolution", bool(args.native_resolution)
            ),
            "resume_from_out_dir": data.get("resume_from_out_dir"),
            "resume_from_repeats": int(data.get("resume_from_repeats", 0)),
        }
        if args.prompt_style == "grouped":
            shard_run_provenance["grouped_cache_provenance"] = data.get("grouped_cache_provenance")
        if merged_run_provenance is None:
            merged_run_provenance = shard_run_provenance
        elif shard_run_provenance != merged_run_provenance:
            raise SystemExit(
                f"Shard {path.name} has inconsistent model/resume provenance: "
                f"{shard_run_provenance!r}, expected {merged_run_provenance!r}."
            )
        minimal_artifact_shards += bool(data.get("minimal_artifacts", False))
        minimal_final_panel_shards += bool(data.get("minimal_final_panels", False))
        reused_repeat_predictions += int(data.get("reused_repeat_predictions", 0))
        resumed_samples += int(data.get("resumed_samples", 0))
        generated_samples += int(data.get("generated_samples", data.get("num_samples", 0)))
        if "classes_per_image" in data:
            classes_per_image_values.add(data["classes_per_image"])
            if len(classes_per_image_values) > 1:
                raise SystemExit(
                    f"Shard {path.name} records classes_per_image={data['classes_per_image']!r}, "
                    f"inconsistent with other shards: {sorted(classes_per_image_values, key=str)!r}."
                )

        shard_classes = data.get("classes")
        shard_palette = data.get("palette")
        if shard_classes is not None or shard_palette is not None:
            if not isinstance(shard_classes, list) or not isinstance(shard_palette, list):
                raise SystemExit(f"Shard {path.name} has incomplete taxonomy/palette provenance.")
            expected_classes = list(base.DATASETS[args.dataset].semantic.classes)
            expected_palette = base.DATASETS[args.dataset].semantic.palette.tolist()
            if shard_classes != expected_classes or shard_palette != expected_palette:
                raise SystemExit(f"Shard {path.name} uses a different dataset taxonomy/palette.")
        acc = base.SegmentationAccumulator.from_dict(data)
        shard_repeats = None
        shard_cumulative_votes = None
        if uses_repeat_payloads:
            shard_repeats = data.get("repeat_accumulators")
            if not isinstance(shard_repeats, list) or len(shard_repeats) != args.randomdraw_repeats:
                raise SystemExit(
                    f"Shard {path.name} has invalid per-repeat metrics: expected "
                    f"{args.randomdraw_repeats} accumulators."
                )
            shard_cumulative_votes = data.get("cumulative_vote_accumulators", [])
            expected_cumulative = max(0, args.randomdraw_repeats - 1)
            if (
                not isinstance(shard_cumulative_votes, list)
                or len(shard_cumulative_votes) != expected_cumulative
            ):
                raise SystemExit(
                    f"Shard {path.name} has invalid cumulative-vote metrics: expected "
                    f"{expected_cumulative} accumulators."
                )
            for repeat_payload in [*shard_repeats, *shard_cumulative_votes]:
                if repeat_payload.get("num_samples") != acc.num_samples:
                    raise SystemExit(f"Shard {path.name} has incomplete per-repeat/cumulative sample counts.")

        num_images += int(data.get("num_samples", acc.num_samples))
        if data.get("num_total_samples") is not None:
            shard_total_counts.add(int(data["num_total_samples"]))
        if merged is None:
            merged = base.SegmentationAccumulator(
                num_classes=acc.num_classes, ignore_index=acc.ignore_index
            )
            if uses_repeat_payloads:
                merged_repeats = [
                    base.SegmentationAccumulator(
                        num_classes=acc.num_classes, ignore_index=acc.ignore_index
                    )
                    for _ in range(args.randomdraw_repeats)
                ]
                merged_cumulative_votes = [
                    base.SegmentationAccumulator(
                        num_classes=acc.num_classes, ignore_index=acc.ignore_index
                    )
                    for _ in range(max(0, args.randomdraw_repeats - 1))
                ]
        merged.merge(acc)
        if uses_repeat_payloads:
            for rep, repeat_payload in enumerate(shard_repeats):
                merged_repeats[rep].merge(
                    base.SegmentationAccumulator.from_dict(repeat_payload)
                )
            for prefix_index, cumulative_payload in enumerate(shard_cumulative_votes):
                merged_cumulative_votes[prefix_index].merge(
                    base.SegmentationAccumulator.from_dict(cumulative_payload)
                )

    if len(shard_total_counts) > 1:
        raise SystemExit(
            "Shard metrics disagree on num_total_samples: "
            + ", ".join(str(value) for value in sorted(shard_total_counts))
        )
    if args.prompt_style == "grouped" and shard_total_counts != {merged.num_samples}:
        raise SystemExit("Grouped shard metrics do not cover the complete selected dataset.")

    try:
        class_names = list(base.DATASETS[args.dataset].semantic.classes)
    except KeyError as exc:
        raise SystemExit(f"Unsupported dataset for class IoUs: {args.dataset}") from exc
    if len(class_names) != merged.num_classes:
        raise SystemExit(
            f"Dataset {args.dataset} defines {len(class_names)} class names, but "
            f"the merged metrics contain {merged.num_classes} classes."
        )

    results = merged.compute()
    results_serializable = {k: float(v) for k, v in results.items()}
    results_serializable["per_class_iou"] = merged.per_class_iou(class_names)
    results_serializable["prompt_style"] = args.prompt_style
    results_serializable["visual_legend_mode"] = args.visual_legend_mode
    results_serializable["batch_size"] = args.batch_size
    if uses_repeat_payloads:
        results_serializable.update(
            {
                "randomdraw_repeats": args.randomdraw_repeats,
                "randomdraw_repeats_used": args.randomdraw_repeats,
                "randomdraw_repeat_range": [1, args.randomdraw_repeats],
                "aggregation": (
                    "single_randomdraw_pass"
                    if args.randomdraw_repeats == 1
                    else "cumulative_majority_vote"
                ),
            }
        )
    out_json = out_dir / f"{args.dataset}_{args.split}_results.json"
    out_json.write_text(json.dumps(results_serializable, indent=2))

    if uses_repeat_payloads:
        for rep, repeat_accumulator in enumerate(merged_repeats, start=1):
            repeat_results = {
                key: float(value) for key, value in repeat_accumulator.compute().items()
            }
            repeat_results["per_class_iou"] = repeat_accumulator.per_class_iou(
                class_names
            )
            repeat_results.update(
                {
                    "prompt_style": args.prompt_style,
                    "visual_legend_mode": args.visual_legend_mode,
                    "batch_size": args.batch_size,
                    "randomdraw_repeat": rep,
                    "randomdraw_repeats": args.randomdraw_repeats,
                    "aggregation": (
                        "single_grouped_prediction"
                        if args.prompt_style == "grouped"
                        else "single_randomdraw_pass"
                    ),
                }
            )
            repeat_json = out_dir / f"{args.dataset}_{args.split}_results_repeat{rep:02d}.json"
            repeat_json.write_text(json.dumps(repeat_results, indent=2))

        for vote_count, cumulative_accumulator in enumerate(
            merged_cumulative_votes, start=2
        ):
            cumulative_results = {
                key: float(value)
                for key, value in cumulative_accumulator.compute().items()
            }
            cumulative_results["per_class_iou"] = (
                cumulative_accumulator.per_class_iou(class_names)
            )
            cumulative_results.update(
                {
                    "prompt_style": args.prompt_style,
                    "visual_legend_mode": args.visual_legend_mode,
                    "batch_size": args.batch_size,
                    "randomdraw_repeats": args.randomdraw_repeats,
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
            cumulative_json.write_text(json.dumps(cumulative_results, indent=2))

    eval_config = {
        "dataset": args.dataset,
        "split": args.split,
        "prompt_style": args.prompt_style,
        "visual_legend_mode": args.visual_legend_mode,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "limit": args.limit,
        "num_images": num_images,
        "num_shards": len(paths),
        "model": merged_run_provenance["model"],
        "transformer_path": merged_run_provenance["transformer_path"],
        "resume_from_out_dir": merged_run_provenance["resume_from_out_dir"],
        "resume_from_repeats": merged_run_provenance["resume_from_repeats"],
        "generated_repeat_range": (
            [merged_run_provenance["resume_from_repeats"] + 1, args.randomdraw_repeats]
            if merged_run_provenance["resume_from_out_dir"] is not None
            else [1, args.randomdraw_repeats]
        ) if args.prompt_style in ("randomdraw", "grouped") else None,
        "reused_repeat_predictions": reused_repeat_predictions,
        "resumed_samples": resumed_samples,
        "generated_samples": generated_samples,
        "resume_cache": (
            "resume_cache/<image>.npz; retained unpacked for incremental extension"
            if args.prompt_style in ("randomdraw", "grouped")
            else None
        ),
        "legend_seed": args.legend_seed if args.prompt_style == "randomdraw" else None,
        "palette_size": (
            len(base.diversify.FULL_PALETTE) if args.prompt_style == "randomdraw" else None
        ),
        "color_mapping": args.color_mapping if args.prompt_style in ("randomdraw", "grouped") else None,
        "randomdraw_repeats": (
            args.randomdraw_repeats if uses_repeat_payloads or args.prompt_style == "grouped" else None
        ),
        "repeat_unit": (
            "complete grouped prediction"
            if args.prompt_style == "grouped" and uses_repeat_payloads
            else "single randomdraw prediction"
            if args.prompt_style == "randomdraw"
            else None
        ),
        "subset_seed": args.subset_seed if args.prompt_style in ("set", "grouped") else None,
        "classes_per_image": (
            next(iter(classes_per_image_values))
            if args.prompt_style == "set" and len(classes_per_image_values) == 1 else None
        ),
        "group_size": args.group_size if args.prompt_style == "grouped" else None,
        "group_batch_size": args.batch_size if args.prompt_style == "grouped" else None,
        "notation": args.group_notation if args.prompt_style == "grouped" else None,
        "implicit_background_id": (
            base.DATASETS[args.dataset].semantic.implicit_background_id
            if args.prompt_style == "grouped" else None
        ),
        "num_prompted_classes": (
            len(class_names) - (base.DATASETS[args.dataset].semantic.implicit_background_id is not None)
            if args.prompt_style == "grouped" else None
        ),
        "prompt": base.DATASETS[args.dataset].semantic.fixed_prompt if args.prompt_style == "fixed" else None,
        "classes": class_names,
        "palette": base.DATASETS[args.dataset].semantic.palette.tolist(),
        "native_resolution": bool(merged_run_provenance["native_resolution"]),
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
            if args.prompt_style == "randomdraw" and args.randomdraw_repeats > 1
            else None
        ),
        "cumulative_vote_prefixes": (
            list(range(2, args.randomdraw_repeats + 1))
            if uses_repeat_payloads and args.randomdraw_repeats > 1
            else []
        ),
        "vote_agreement_definition": (
            "winning-class vote share among non-abstaining random color "
            "assignments; vote stability, not calibrated correctness probability"
            if uses_repeat_payloads and args.randomdraw_repeats > 1
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
            if uses_repeat_payloads and args.randomdraw_repeats > 1
            else None
        ),
        "aggregated_from_shards": True,
    }
    if args.prompt_style == "grouped":
        eval_config["grouped_cache_provenance"] = merged_run_provenance["grouped_cache_provenance"]
        cache_provenance = eval_config["grouped_cache_provenance"] or {}
        if cache_provenance.get("legend_pruning") is not None:
            eval_config["legend_pruning"] = cache_provenance["legend_pruning"]
        eval_config.update(
            color_min_delta_e=cache_provenance.get("color_min_delta_e"),
            farthest_colors=cache_provenance.get("farthest_colors"),
            num_groups=(eval_config["num_prompted_classes"] + args.group_size - 1) // args.group_size,
            palette_size=len(base.diversify.NAMED_PALETTE if args.group_notation == "name" else base.diversify.FULL_PALETTE),
            artifacts={
                "per_group_raw_rgb": "pred_rgb_raw/" + ("repeatKK/" if args.randomdraw_repeats > 1 else "") + "gK/<image>.png",
                "per_group_snapped_rgb": "pred_rgb_snapped/" + ("repeatKK/" if args.randomdraw_repeats > 1 else "") + "gK/<image>.png",
                "per_repeat_trainids": "pred_trainids_repeats/repeatKK/<image>.png" if args.randomdraw_repeats > 1 else None,
                "cumulative_trainids": "pred_trainids_cumulative/1toKK/<image>.png" if args.randomdraw_repeats > 1 else None,
                "merged_trainids": "pred_trainids/<image>.png",
                "merged_snapped_rgb": "pred_rgb_snapped/<image>.png",
                "ground_truth_trainids": "ground_truth_trainids/<image>.png",
                "ground_truth_rgb": "ground_truth_rgb/<image>.png",
                "panel": "panels/<image>.png",
            },
        )
        if minimal_artifact_shards:
            # --minimal_artifacts shards skip every map that resume_cache and the
            # raw predictions can rebuild, plus the per-repeat panels; scores are
            # unaffected.
            final_panels = minimal_final_panel_shards == minimal_artifact_shards
            eval_config["minimal_artifacts"] = {
                "shards": minimal_artifact_shards,
                "skipped": ["per_group_snapped_rgb", "per_repeat_trainids", "per_repeat_snapped_rgb",
                            "cumulative_trainids", "cumulative_snapped_rgb", "vote_agreement",
                            "vote_agreement_color", "per_repeat_panels"] + ([] if final_panels else ["panel"]),
                "rebuild": "python evaluation/semseg/rebuild_grouped_artifacts.py --out_dir <this directory>",
            }
            for key in ("per_group_snapped_rgb", "per_repeat_trainids", "cumulative_trainids") + (() if final_panels else ("panel",)):
                eval_config["artifacts"][key] = None
    (out_dir / "eval_config.json").write_text(json.dumps(eval_config, indent=2))

    print(f"Aggregated {len(paths)} shards ({num_images} images) -> {out_json}")
    for k, v in results.items():
        print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
