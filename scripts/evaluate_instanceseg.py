#!/usr/bin/env python3
"""Evaluate FLUX.2-Klein instance segmentation on SA-Co/Gold with the paper protocol.

Defaults reproduce the paper: the 14k seed-0 subset (2000 images per subset), the
bundled Qwen3-VL-8B presence-gate decisions, the fixed-draw 32-colour
left-to-right prompt contract, and the refined decode (palette snap, decoder tau 58 /
gamma 1 / theta_size 8e-4, gap fill, GrabCut; tuned on the 14k seed-0 subset).
The image ids of that subset are listed in evaluation/assets/saco_gold_14k_seed0_subset.json.
``--full`` scores all 167,522 images with a bundled full-set gate (``--full-gate``). ``--repeats N`` generates N palette
draws per query and votes their masks. ``--raw-rgb-root`` rescores saved raw outputs
on CPU without loading the model.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.release_eval import add_common_arguments, add_node_arguments, append_if, evaluator_python, forward_extra, launch_sharded, locate_dataset, node_group, selected_datasets, prepare_outputs, finish_outputs
DATASETS=("saco_gold",)
ASSETS=Path(__file__).resolve().parents[1]/"evaluation"/"assets"
PAPER_GATE=ASSETS/"saco_gold_gate_decisions_qwen3vl8b.json"
FULL_GATES={"qwen3vl8b":ASSETS/"saco_gold_gate_decisions_qwen3vl8b_full.json","qwen3vl30ba3b":ASSETS/"saco_gold_gate_decisions_qwen3vl30ba3b_full.json"}
SUBSET_LIST=ASSETS/"saco_gold_14k_seed0_subset.json"
def main():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); add_common_arguments(p,DATASETS); add_node_arguments(p)
    p.add_argument("--gate",choices=("file","oracle","none"),default="file"); p.add_argument("--gate-file",type=Path,help=f"Default: {PAPER_GATE.name} (14k seed-0 subset).")
    p.add_argument("--prompt-set",default="fixeddraw_32_lr"); p.add_argument("--tau",type=float,default=58.0); p.add_argument("--subsets",nargs="+")
    p.add_argument("--per-subset-limit",type=int,default=2000,help="Images per subset (paper: 2000 x 7 subsets = 14k)."); p.add_argument("--sample-seed",type=int,default=0)
    p.add_argument("--full",action="store_true",help="Score every SA-Co/Gold image instead of the 14k subset.")
    p.add_argument("--full-gate",choices=sorted(FULL_GATES),default="qwen3vl8b",help="Bundled full-set presence gate used with --full when no --gate-file is given (Qwen3-VL-8B-Instruct or Qwen3-VL-30B-A3B decisions).")
    p.add_argument("--batch-size",type=int,help="Queries per FLUX call; default 8 for a 4B base model, else 4 (as in the paper runs).")
    p.add_argument("--native-resolution",action="store_true")
    p.add_argument("--repeats",type=int,default=1,metavar="N",help="Random palette draws per query, matched and majority-voted (default 1).")
    p.add_argument("--raw-rgb-root",type=Path,help="Earlier output dir with <subset>/pred_rgb_raw/*.png: decode, refine and score those saved outputs on CPU instead of generating. Use the source run's sample, gate and --repeats settings.")
    args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True); nodes=node_group(args)
    if args.limit is not None: raise SystemExit("Use --per-subset-limit for SA-Co/Gold.")
    gate_file=args.gate_file or ((FULL_GATES[args.full_gate] if args.full else PAPER_GATE) if args.gate=="file" else None)
    if args.repeats<1: raise SystemExit("--repeats must be positive")
    batch_size=args.batch_size or (8 if "4B" in str(args.model) else 4)
    for ds in selected_datasets(args.datasets,DATASETS):
        data=locate_dataset(args.data_root,args.output_dir,"Instanceseg",ds,names=("SA-Co-Gold","sa_co_gold"),task_dirs=("instanceseg",),nodes=nodes); out=args.output_dir/(ds if args.repeats==1 else f"{ds}_x{args.repeats}"); out.mkdir(parents=True,exist_ok=True)
        cmd=evaluator_python("evaluation/instanceseg/inference_flux2_klein_instanceseg.py")+["--model",args.model,"--saco_root",str(data),"--out_dir",str(out),"--steps",str(args.steps),"--guidance_scale",str(args.guidance),"--generation_seed",str(args.seed),"--gate",args.gate,"--prompt_set",args.prompt_set,"--tau",str(args.tau),"--sample_seed",str(args.sample_seed),"--batch_size",str(batch_size),"--instance_score","1.0","--iou_type","segm","--run_eval"]
        for flag,value in (("--transformer_path",args.transformer_path),("--gate_file",gate_file)): append_if(cmd,flag,value)
        if not args.full: cmd += ["--limit",str(args.per_subset_limit)]
        if args.gate=="file": cmd.append("--require_complete_gate")
        if args.subsets: cmd += ["--subsets",*args.subsets]
        if args.native_resolution: cmd.append("--native_resolution")
        if args.repeats>1: cmd += ["--randomize_palette","--randomdraw_repeats",str(args.repeats)]
        if args.raw_rgb_root: cmd += ["--postprocess_only","--raw_rgb_root",str(args.raw_rgb_root)]
        if args.dry_run: cmd.append("--dry_run")
        forward_extra(cmd,args.extra_arg); prepare_outputs(out,args,nodes,ds)
        launch_sharded(cmd,args.num_gpus,out,lambda n,r,s:["--num_shards",str(n),"--shard_index",str(r),"--local_rank","0"],nodes)
        finish_outputs(out,args,nodes,ds)
if __name__=="__main__": main()
