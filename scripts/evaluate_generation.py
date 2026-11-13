#!/usr/bin/env python3
"""Generate and score FLUX.2-Klein images on GenAI-Bench (1600 prompts).

Defaults reproduce the paper: seeds 0, 1 and 2 (per-prompt noise derived from
seed and prompt id, so results do not depend on the GPU count), 28 steps,
guidance 3.5, scored with the CLIP-FlanT5-XXL VQAScore judge (t2v_metrics, see
evaluation/image_generation/README.md for the scoring environment). The headline is the mean over prompts of each
prompt's seed-averaged score, written to genai_bench/summary.json.
``--compare-to`` adds a paired comparison against another output of this script
(e.g. the base model).
"""
import argparse, os, json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.release_eval import add_common_arguments, add_node_arguments, append_if, evaluator_python, forward_extra, launch_sharded, locate_dataset, node_group, selected_datasets, prepare_outputs, finish_outputs
DATASETS=("genai_bench",)
def summarize(score_dirs, out):
    per_prompt={}; per_seed={}
    for seed,score_dir in score_dirs.items():
        rows=[json.loads(line) for line in (score_dir/"per_prompt.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        per_seed[str(seed)]=round(statistics.mean(row["score"] for row in rows),4)
        for row in rows: per_prompt.setdefault(row["id"],[]).append(row["score"])
    summary={"mean_score":round(statistics.mean(statistics.mean(v) for v in per_prompt.values()),4),"per_seed_mean":per_seed,"num_prompts":len(per_prompt),"score_dirs":{str(k):str(v) for k,v in score_dirs.items()}}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8"); print(json.dumps(summary,indent=2))
def main():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); add_common_arguments(p,DATASETS); add_node_arguments(p)
    p.add_argument("--seeds",type=int,nargs="+",default=[0,1,2]); p.add_argument("--prompts-jsonl"); p.add_argument("--height",type=int); p.add_argument("--width",type=int); p.add_argument("--prompt-prefix",default="")
    p.add_argument("--evaluator",default="clip-flant5-xxl",help="VQAScore judge (paper: clip-flant5-xxl; qwen2.5-vl-7b is also supported)."); p.add_argument("--score-batch-size",type=int,default=1); p.add_argument("--score-model-path",help="Local judge snapshot (optional).")
    p.add_argument("--score-python",default=sys.executable,help="Python of the GenAI scoring environment.")
    p.add_argument("--score-pythonpath",nargs="+",metavar="DIR",help="Directories put first on PYTHONPATH of the scoring process only, e.g. the CLIP-FlanT5 compatibility tree and the transformers 4.40.2 overlay (see evaluation/image_generation/README.md).")
    p.add_argument("--skip-score",action="store_true"); p.add_argument("--compare-to",type=Path,help="Another genai_bench output dir from this script (e.g. the base model).")
    p.add_argument("--overwrite",action="store_true"); p.add_argument("--model-offload",action="store_true")
    args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True); nodes=node_group(args)
    if args.seed!=0: raise SystemExit("Use --seeds for GenAI-Bench.")
    for ds in selected_datasets(args.datasets,DATASETS):
        prompts=args.prompts_jsonl
        if not prompts:
            data=locate_dataset(args.data_root,args.output_dir,"ImgGeneration",ds,names=("genai-bench",),task_dirs=("image_generation/prepared","image_generation"),nodes=nodes)
            matches=list(data.rglob("prompts.jsonl"))
            if len(matches)!=1: raise FileNotFoundError(f"Expected one prompts.jsonl under {data}; found {len(matches)}")
            prompts=str(matches[0])
        root=args.output_dir/ds; score_dirs={}; outs={}
        for seed in args.seeds:
            out=outs[seed]=root/f"seed{seed}"; out.mkdir(parents=True,exist_ok=True)
            cmd=evaluator_python("evaluation/image_generation/genai_bench/generate_flux2_klein.py")+["--prompts_jsonl",str(prompts),"--out_dir",str(out),"--model",args.model,"--model_family","klein","--steps",str(args.steps),"--guidance_scale",str(args.guidance),"--seed",str(seed),"--prompt_prefix",args.prompt_prefix]
            for flag,value in (("--transformer_path",args.transformer_path),("--limit",args.limit),("--height",args.height),("--width",args.width)): append_if(cmd,flag,value)
            if args.model_offload: cmd.append("--model_offload")
            if args.overwrite: cmd.append("--overwrite")
            if args.dry_run: cmd.append("--dry_run")
            forward_extra(cmd,args.extra_arg); prepare_outputs(out,args,nodes,f"{ds}-seed{seed}")
            launch_sharded(cmd,args.num_gpus,out,lambda n,r,s:["--num_shards",str(n),"--shard_index",str(r),"--local_rank","0"],nodes)
        # Every seed is generated on all nodes first; seed i is then merged and
        # scored on node i mod num-nodes, and node 0 writes the summary.
        nodes.barrier(f"{ds}-generated")
        for index,(seed,out) in enumerate(outs.items()):
            if index%nodes.count!=nodes.rank:
                if not (args.skip_score or args.dry_run): score_dirs[seed]=out/"scores"/args.evaluator
                continue
            shards=sorted((out/"shard_manifests").glob("rank_*_of_*.jsonl"))
            if shards:
                with (out/"manifest.jsonl").open("w",encoding="utf-8") as target:
                    for shard in shards: target.write(shard.read_text(encoding="utf-8"))
            if args.skip_score or args.dry_run: continue
            score=[args.score_python,str(Path(__file__).resolve().parents[1]/"evaluation/image_generation/genai_bench/score_t2v_metrics.py"),"--generation_dir",str(out),"--evaluator",args.evaluator,"--batch_size",str(args.score_batch_size)]
            append_if(score,"--checkpoint",args.score_model_path)
            if args.overwrite: score.append("--overwrite")
            previous=os.environ.get("PYTHONPATH")
            if args.score_pythonpath: os.environ["PYTHONPATH"]=os.pathsep.join([*args.score_pythonpath,*([previous] if previous else [])])
            try: launch_sharded(score,1,out,lambda n,r,s:[])
            finally:
                if args.score_pythonpath:
                    if previous is None: os.environ.pop("PYTHONPATH",None)
                    else: os.environ["PYTHONPATH"]=previous
            score_dirs[seed]=out/"scores"/args.evaluator
        if not score_dirs:
            for seed,out in outs.items(): finish_outputs(out,args,nodes,f"{ds}-seed{seed}")
            continue
        nodes.barrier(f"{ds}-scored")
        for seed,out in outs.items(): finish_outputs(out,args,nodes,f"{ds}-seed{seed}")
        if not nodes.main: continue
        summarize(score_dirs,root)
        if args.compare_to:
            base=[str(args.compare_to/f"seed{seed}"/"scores"/args.evaluator) for seed in args.seeds]
            compare=evaluator_python("evaluation/image_generation/genai_bench/compare_scores.py")+[x for b in base for x in ("--base_scores",b)]+[x for c in score_dirs.values() for x in ("--candidate_scores",str(c))]+["--out_dir",str(root/"comparison")]
            launch_sharded(compare,1,root,lambda n,r,s:[])
if __name__=="__main__": main()
