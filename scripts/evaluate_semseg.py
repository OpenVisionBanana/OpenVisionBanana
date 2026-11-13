#!/usr/bin/env python3
"""Evaluate FLUX.2-Klein semantic segmentation with the paper protocol.

By default every image gets one colour draw (``--repeats 1``): a random-draw
legend with the best-64 colour pool (Cityscapes), and grouped RGB legends of 27
classes, three groups per FLUX call (COCO-Object-80). ``--mode`` adds one of two
evaluations; seeds, colour draws and grouping stay the same:

  legend-ensembling  Test-time scaling: N independent colour draws per image scored
                     by cumulative majority vote (``--repeats N``, default 5 for
                     Cityscapes and 10 for COCO-Object-80). Every prefix 1..N is
                     written as ``*_results_1toK.json``.
  legend-pruning     COCO-Object-80 only. A VLM (``--vlm-model``) first names the
                     classes present in each image and the grouped legends cover only
                     those. Add ``--repeats N`` to also ensemble N draws of the pruned
                     legends. The VLM answers are cached per model as
                     evaluation/assets/coco_object_80_val_legend_pruning_<vlm>.json
                     (bundled for qwen3vl8binstruct and gemma4_31b); a missing cache
                     is created first with the given model. Several models keep the
                     union of their classes.

``--repeats`` without ``--mode`` keeps its earlier meaning (legend-ensembling).
"""
import argparse, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "evaluation" / "semseg")]
from common.release_eval import add_common_arguments, add_node_arguments, append_if, evaluator_python, forward_extra, launch_sharded, locate_dataset, node_group, selected_datasets, prepare_outputs, finish_outputs
import legend_pruning

# release name -> (evaluator --dataset, extra data directory names)
DATASETS = {
    "cityscapes": ("cityscapes", ()),
    "coco_object_80": ("coco-object-80", ("coco",)),
    "acdc": ("acdc", ()),
    "dark_zurich": ("darkzurich", ("darkzurich",)),
    "bdd": ("bdd", ("bdd100k",)),
    "coco_stuff": ("coco-stuff", ("coco",)),
    "coco_thing": ("coco-thing", ("coco",)),
}
PAPER_DATASETS = "cityscapes,coco_object_80"
GROUPED = {"coco_object_80"}
DEFAULT_REPEATS = {"cityscapes": 5, "coco_object_80": 10}
MODES = ("legend-ensembling", "legend-pruning")


def pruning_caches(args, name, data, nodes):
    """Legend pruning cache per VLM: the full-set cache if present, else one for
    this --limit; a missing cache is created with the VLM over every shard first."""
    paths = []
    for value in args.vlm_model:
        vlm, _ = legend_pruning.resolve_vlm(value)
        path = legend_pruning.cache_path(args.legend_pruning_dir, name, args.split, vlm, None)
        if not path.is_file() and args.limit:
            path = legend_pruning.cache_path(args.legend_pruning_dir, name, args.split, vlm, args.limit)
        if path.is_file():
            print(f"Legend pruning ({vlm}): {path}")
        elif args.dry_run:
            print(f"Legend pruning ({vlm}): would create {path}"); continue
        else:
            print(f"Legend pruning ({vlm}): {path} missing; creating it")
            shards = path.with_name(path.name + ".shards")
            cmd = evaluator_python("evaluation/semseg/legend_pruning.py") + [
                "--dataset", name, "--data_root", str(data), "--split", args.split, "--out", str(path),
                "--vlm", value, "--batch_size", str(args.vlm_batch_size)]
            append_if(cmd, "--limit", args.limit); append_if(cmd, "--vlm_sys_path", args.vlm_pythonpath)
            launch_sharded(cmd, args.num_gpus, shards, lambda n, r, s: ["--num_shards", str(n), "--shard_index", str(r)], nodes)
            nodes.barrier(f"legend-pruning-{vlm}")
            if nodes.main:
                if not path.is_file(): legend_pruning.merge_shards(path, args.num_gpus * nodes.count)
                shutil.rmtree(shards, ignore_errors=True)
            nodes.barrier(f"legend-pruning-merged-{vlm}")
        paths.append(path)
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(p, tuple(DATASETS)); add_node_arguments(p); p.set_defaults(datasets=None)
    p.add_argument("--split", default="val")
    p.add_argument("--mode", choices=MODES, help="Default: one colour draw per image (legend-ensembling if --repeats N is given).")
    p.add_argument("--repeats", type=int, nargs="?", const=0, default=1, metavar="N",
                   help="Random colour draws per image, majority-voted. Bare --repeats uses 5 (Cityscapes) or 10 (COCO-Object-80).")
    p.add_argument("--vlm-model", nargs="+", metavar="VLM",
                   help=f"Legend pruning VLM: {', '.join(legend_pruning.VLMS)}, an HF id or a local snapshot. "
                        "Several VLMs keep the union of their classes.")
    p.add_argument("--legend-pruning-dir", type=Path, default=legend_pruning.ASSETS,
                   help="Where legend pruning caches are looked up and created (default: evaluation/assets).")
    p.add_argument("--vlm-pythonpath", help="Directory put first on sys.path of the legend pruning VLM pass only "
                                            "(transformers>=5 for Gemma 4, see requirements/reasonseg-rewrite-gemma.txt).")
    p.add_argument("--vlm-batch-size", type=int, default=8, help="Images per VLM generate call when creating a cache.")
    p.add_argument("--color-mapping", default="best64_maxnegdraw")
    p.add_argument("--prompt-style", choices=("fixed", "randomdraw"), default="randomdraw", help="Non-grouped datasets only.")
    p.add_argument("--group-size", type=int, default=27, help="COCO-Object-80: classes per grouped legend (paper: 27, i.e. 3 legends).")
    p.add_argument("--group-batch-size", type=int, default=3, help="COCO-Object-80: group prompts per FLUX call (paper: 3).")
    p.add_argument("--group-notation", choices=("rgb", "hex", "name"), default="rgb", help="COCO-Object-80 legend notation.")
    p.add_argument("--model-offload", action="store_true"); p.add_argument("--resume", action="store_true"); p.add_argument("--native-resolution", action="store_true")
    args = p.parse_args()
    mode = args.mode or ("legend-ensembling" if args.repeats != 1 else None)
    if mode == "legend-ensembling" and args.repeats == 1: args.repeats = 0  # dataset default count
    pruning = mode == "legend-pruning"
    if pruning and not args.vlm_model: p.error(f"--mode legend-pruning needs --vlm-model ({' or '.join(legend_pruning.VLMS)})")
    if args.vlm_model and not pruning: p.error("--vlm-model applies only to --mode legend-pruning")
    if pruning:  # one entry per VLM, in a stable order for the output folder name
        args.vlm_model = sorted({legend_pruning.resolve_vlm(v)[0]: v for v in args.vlm_model}.items())
        prune_tag = "_legendpruning-" + "-".join(vlm for vlm, _ in args.vlm_model)
        args.vlm_model = [v for _, v in args.vlm_model]
    datasets = selected_datasets(args.datasets or (",".join(sorted(GROUPED)) if pruning else PAPER_DATASETS), tuple(DATASETS))
    if pruning and set(datasets) - GROUPED:
        p.error(f"--mode legend-pruning supports {', '.join(sorted(GROUPED))} only, got {', '.join(sorted(set(datasets) - GROUPED))}")
    args.output_dir.mkdir(parents=True, exist_ok=True); nodes = node_group(args)
    for ds in datasets:
        name, aliases = DATASETS[ds]
        repeats = args.repeats or DEFAULT_REPEATS.get(ds, 5)
        if repeats < 1: raise ValueError("--repeats must be positive")
        data = locate_dataset(args.data_root, args.output_dir, "Semseg", ds, names=aliases, task_dirs=("sem_segm",), nodes=nodes)
        caches = pruning_caches(args, name, data, nodes) if pruning else []
        if pruning and len(caches) < len(args.vlm_model): continue  # dry run without a cache
        out = args.output_dir / (f"{ds}{'' if repeats == 1 else f'_x{repeats}'}{prune_tag if pruning else ''}"); out.mkdir(parents=True, exist_ok=True)
        common = ["--model", args.model, "--dataset", name, "--data_root", str(data), "--out_dir", str(out), "--split", args.split,
                  "--steps", str(args.steps), "--guidance_scale", str(args.guidance), "--generation_seed", str(args.seed),
                  "--color_mapping", args.color_mapping, "--randomdraw_repeats", str(repeats)]
        if ds in GROUPED:
            cmd = evaluator_python("evaluation/semseg/grouped_semseg.py") + common + [
                "--subset_seed", "0", "--group_size", str(args.group_size), "--group_batch_size", str(args.group_batch_size),
                "--notation", args.group_notation, "--color_min_delta_e", "25.0", "--farthest_colors"]
            if caches: cmd += ["--legend_pruning_json", *map(str, caches)]
        else:
            cmd = evaluator_python("evaluation/semseg/inference_partial_flux2_klein_img2img.py") + common + [
                "--model_family", "klein", "--prompt_style", args.prompt_style, "--legend_seed", "0", "--batch_size", "1",
                "--visual_legend_mode", "none"]
            if args.model_offload: cmd.append("--model_offload")
        append_if(cmd, "--transformer_path", args.transformer_path); append_if(cmd, "--limit", args.limit)
        if args.native_resolution: cmd.append("--native_resolution")
        if args.resume: cmd.append("--resume")
        if args.dry_run: cmd.append("--dry_run")
        forward_extra(cmd, args.extra_arg); prepare_outputs(out, args, nodes, ds)
        launch_sharded(cmd, args.num_gpus, out, lambda n, r, s: ["--num_shards", str(n), "--shard_index", str(r), "--local_rank", "0"], nodes)
        finish_outputs(out, args, nodes, ds, keep=("resume_cache",))
if __name__ == "__main__": main()
