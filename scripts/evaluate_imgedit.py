#!/usr/bin/env python3
"""Generate, judge, and average FLUX.2-Klein predictions on ImgEdit-Bench.

Defaults reproduce the paper: Basic and UGE suites, three runs with seeds
1001-1003, each generated as 16 logical shards (ImgEdit seeds one generator per
shard, so the shard layout is part of the protocol; the 16 shards are queued over
however many GPUs are given), scored by the ImgEdit judge and averaged over runs
into <suite>/average_across_runs.json.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.release_eval import add_common_arguments, add_node_arguments, append_if, evaluator_python, forward_extra, launch_shard_queue, launch_sharded, locate_dataset, node_group, selected_datasets, prepare_outputs, finish_outputs
SUITES=("basic","uge","multiturn")
def merge_manifests(out):
    shards=sorted((out/"shard_manifests").glob("rank_*_of_*.jsonl"))
    if shards:
        with (out/"manifest.jsonl").open("w",encoding="utf-8") as target:
            for shard in shards: target.write(shard.read_text(encoding="utf-8"))
def main():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); add_common_arguments(p,SUITES); add_node_arguments(p); p.set_defaults(datasets="basic,uge")
    p.add_argument("--prompt-prefix",default="[TASK: IMAGE_EDITING]"); p.add_argument("--runs",type=int,default=3); p.add_argument("--seed-base",type=int,default=1000,help="Run i uses seed seed_base+i.")
    p.add_argument("--logical-shards",type=int,default=16,help="Generation shard layout (paper: 16).")
    p.add_argument("--judge-model",type=Path,help="ImgEdit_Judge directory (huggingface.co/datasets/sysuyy/ImgEdit). Default: ImgEdit_Judge next to the benchmark.")
    p.add_argument("--skip-judge",action="store_true"); p.add_argument("--overwrite",action="store_true"); p.add_argument("--native-resolution",action="store_true")
    args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True); nodes=node_group(args)
    data=locate_dataset(args.data_root,args.output_dir,"ImgEdit","imgedit_bench",names=("Benchmark",),task_dirs=("imgedit","image_editing/imgedit_bench"),nodes=nodes)
    if not (data/"singleturn").is_dir() and (data/"Benchmark"/"singleturn").is_dir(): data=data/"Benchmark"  # Benchmark.tar unpacks to Benchmark/
    judge=args.judge_model or next((c for c in (data/"ImgEdit_Judge",data.parent/"ImgEdit_Judge") if c.is_dir()),None)
    if not args.skip_judge and not args.dry_run and judge is None: raise SystemExit("Pass --judge-model (ImgEdit_Judge) or --skip-judge.")
    suites=selected_datasets(args.datasets,SUITES)
    for suite in suites:
        for run in range(1,args.runs+1):
            out=args.output_dir/suite/f"run{run}"; out.mkdir(parents=True,exist_ok=True)
            cmd=evaluator_python("evaluation/imgedit/inference_flux2_klein_imgedit.py")+["--model",args.model,"--benchmark_root",str(data),"--suite",suite,"--out_dir",str(out),"--steps",str(args.steps),"--guidance_scale",str(args.guidance),"--seed",str(args.seed_base+run),"--prompt_prefix",args.prompt_prefix]
            append_if(cmd,"--transformer_path",args.transformer_path); append_if(cmd,"--limit",args.limit)
            if args.native_resolution: cmd.append("--native_resolution")
            if args.overwrite: cmd.append("--overwrite")
            if args.dry_run: cmd.append("--dry_run")
            forward_extra(cmd,args.extra_arg); prepare_outputs(out,args,nodes,f"{suite}-run{run}")
            launch_shard_queue([cmd+["--num_shards",str(args.logical_shards),"--shard_index",str(i),"--local_rank","0"] for i in range(args.logical_shards)],args.num_gpus,nodes)
            nodes.barrier(f"{suite}-run{run}-generated")
            if nodes.main: merge_manifests(out)
            if suite=="multiturn" and nodes.main: print("multiturn: the official Multi-Turn bench has no automatic judge; outputs are generated only.")
            if args.skip_judge or args.dry_run or suite=="multiturn": finish_outputs(out,args,nodes,f"{suite}-run{run}"); continue
            judge_cmd=evaluator_python("evaluation/imgedit/judge_imgedit_qwen.py")+["--suite",suite,"--benchmark_root",str(data),"--judge_model",str(judge),"--result_img_folder",str(out)]
            append_if(judge_cmd,"--limit",args.limit)
            launch_sharded(judge_cmd,args.num_gpus,out,lambda n,r,s:["--num_shards",str(n),"--shard_index",str(r),"--local_rank","0"],nodes)
            nodes.barrier(f"{suite}-run{run}-judged")
            if nodes.main: launch_sharded(judge_cmd+["--merge_shards"],1,out,lambda n,r,s:[])
            finish_outputs(out,args,nodes,f"{suite}-run{run}")
    scored=[suite for suite in suites if suite!="multiturn"]
    if args.runs>1 and scored and not (args.skip_judge or args.dry_run) and nodes.main:
        launch_sharded(evaluator_python("evaluation/imgedit/aggregate_imgedit_runs.py")+[str(args.output_dir/suite) for suite in scored],1,args.output_dir,lambda n,r,s:[])
if __name__=="__main__": main()
