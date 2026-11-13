#!/usr/bin/env python3
"""Evaluate FLUX.2-Klein on RefCOCOg (UMD val, cIoU) and/or ReasonSeg (val, gIoU).

Defaults reproduce the paper: ReasonSeg's implicit queries are replaced by the
bundled Gemma 4 31B explicit rewrites (output ``reasonseg_gemma4_31b/``).
``--reasonseg-rewrites qwen3vl32bthink`` uses the bundled Qwen3-VL-32B-Thinking rewrites
instead (output ``reasonseg_qwen3vl32bthink/``), ``--reasonseg-rewrite-file`` a custom
cache (output ``reasonseg/``), and ``--no-rewrite`` the raw ReasonSeg queries (output
``reasonseg_raw/``).
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.release_eval import add_common_arguments, add_node_arguments, append_if, evaluator_python, forward_extra, launch_sharded, locate_dataset, node_group, selected_datasets, prepare_outputs, finish_outputs
DATASETS=("refcocog","reasonseg")
DATA_NAMES={"refcocog":(),"reasonseg":("reasonseg_val",)}
ASSETS=Path(__file__).resolve().parents[1]/"evaluation"/"assets"
# Bundled ReasonSeg rewrite caches; DEFAULT_REWRITES is the paper protocol.
REWRITES={"gemma4_31b":ASSETS/"reasonseg_val_rewrites_gemma4_31b.json","qwen3vl32bthink":ASSETS/"reasonseg_val_rewrites_qwen3vl32bthink.json"}
DEFAULT_REWRITES="gemma4_31b"
def main():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); add_common_arguments(p,DATASETS); add_node_arguments(p)
    p.add_argument("--reasonseg-rewrites",choices=REWRITES,default=DEFAULT_REWRITES,help=f"Bundled ReasonSeg query rewrites (default: {DEFAULT_REWRITES}, the paper protocol).")
    p.add_argument("--reasonseg-rewrite-file",help="Custom rewrite cache instead of a bundled one; writes to reasonseg/.")
    p.add_argument("--no-rewrite","--raw-reasonseg-query",dest="no_rewrite",action="store_true",help="Use the raw ReasonSeg queries.")
    p.add_argument("--images-root",type=Path,help="RefCOCOg train2014 image root; default <refcocog>/train2014 when present. Overrides paths stored in the index.")
    p.add_argument("--native-resolution",action="store_true")
    args=p.parse_args()
    alt=args.reasonseg_rewrites!=DEFAULT_REWRITES
    if alt and (args.no_rewrite or args.reasonseg_rewrite_file): p.error("--reasonseg-rewrites cannot be combined with --no-rewrite or --reasonseg-rewrite-file")
    # Always pass the bundled cache explicitly: the evaluator's own default is the Qwen file.
    bundled=not (args.no_rewrite or args.reasonseg_rewrite_file)
    if bundled: args.reasonseg_rewrite_file=REWRITES[args.reasonseg_rewrites]
    args.output_dir.mkdir(parents=True,exist_ok=True); nodes=node_group(args)
    for ds in selected_datasets(args.datasets,DATASETS):
        data=locate_dataset(args.data_root,args.output_dir,"Refseg",ds,names=DATA_NAMES[ds],task_dirs=("refseg",),nodes=nodes); out=args.output_dir/(f"{ds}_raw" if ds=="reasonseg" and args.no_rewrite else f"{ds}_{args.reasonseg_rewrites}" if ds=="reasonseg" and bundled else ds); out.mkdir(parents=True,exist_ok=True)
        cmd=evaluator_python("evaluation/refseg/inference_flux2_klein_refseg.py")+["--dataset",ds,"--model",args.model,"--out_dir",str(out),"--steps",str(args.steps),"--guidance_scale",str(args.guidance),"--generation_seed",str(args.seed),"--color_seed","0"]
        if ds=="refcocog":
            matches=list(data.rglob("refcocog_val_umd.json"))
            if len(matches)!=1: raise FileNotFoundError(f"Expected one prepared refcocog_val_umd.json under {data}; found {len(matches)}")
            cmd += ["--index",str(matches[0])]
            images_root=args.images_root or (data/"train2014" if (data/"train2014").is_dir() else None)
            append_if(cmd,"--images_root",images_root)
        else:
            cmd += ["--data_dir",str(data)]
            if args.no_rewrite: cmd.append("--no-reasonseg_rewrite")
            else: append_if(cmd,"--reasonseg_rewrite_file",args.reasonseg_rewrite_file)
        append_if(cmd,"--transformer_path",args.transformer_path); append_if(cmd,"--limit",args.limit)
        if args.native_resolution: cmd.append("--native_resolution")
        if args.dry_run: cmd.append("--dry_run")
        forward_extra(cmd,args.extra_arg); prepare_outputs(out,args,nodes,ds)
        launch_sharded(cmd,args.num_gpus,out,lambda n,r,s:["--num_shards",str(n),"--shard_index",str(r),"--local_rank","0"],nodes)
        finish_outputs(out,args,nodes,ds)
if __name__=="__main__": main()
