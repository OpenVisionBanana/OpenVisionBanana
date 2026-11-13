#!/usr/bin/env python3
"""Evaluate FLUX.2-Klein on one, several, or all supported normal benchmarks."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.geometry_protocol_prompts import CONCISE_NORMAL_PROMPT
from common.release_eval import add_common_arguments, append_if, evaluator_python, forward_extra, launch_sharded, locate_dataset, selected_datasets
SUPPORTED_DATASETS=("nyu","ibims1","diode","kitti","eth3d","nuscenes","vkitti2","scannet","sintel")
PAPER_DATASETS="nyu,ibims1,diode,scannet,vkitti2,sintel"

def main():
    p=argparse.ArgumentParser(description=__doc__); add_common_arguments(p,SUPPORTED_DATASETS)
    p.set_defaults(datasets=PAPER_DATASETS)
    p.add_argument("--split",default="test"); p.add_argument("--cases",nargs="+",default=["standard"]); p.add_argument("--prompt",default=CONCISE_NORMAL_PROMPT)
    p.add_argument("--model-offload",action="store_true"); p.add_argument("--resume",action="store_true")
    p.add_argument("--native-resolution",action=argparse.BooleanOptionalAction,default=True,help="Generate at the input resolution (paper setting). --no-native-resolution uses the pipeline default size.")
    p.add_argument("--prediction-encoding",default="vision_banana_x_left_red"); p.add_argument("--gt-encoding",default="auto")
    p.add_argument("--protocol-version", default="geometry-eval-v6", help="Corrected normal target/scoring contract recorded in outputs.")
    p.add_argument("--heldout-calibration-record"); p.add_argument("--heldout-so3-calibration-record"); p.add_argument("--sample-manifest")
    args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    for dataset in selected_datasets(args.datasets,SUPPORTED_DATASETS):
        data=locate_dataset(args.data_root,args.output_dir,"Normals",dataset); out=args.output_dir/dataset; out.mkdir(parents=True,exist_ok=True)
        cmd=evaluator_python("evaluation/surface_normals/inference_partial_flux2_klein_img2img.py")+["--model",args.model,"--model_family","klein","--dataset",dataset,"--data_root",str(data),"--out_dir",str(out),"--split",args.split,"--steps",str(args.steps),"--guidance_scale",str(args.guidance),"--seed",str(args.seed),"--prediction_normal_encoding",args.prediction_encoding,"--gt_normal_encoding",args.gt_encoding,"--cases",*args.cases]
        for flag,value in (("--transformer_path",args.transformer_path),("--limit",args.limit),("--prompt",args.prompt),("--heldout_calibration_record",args.heldout_calibration_record),("--heldout_so3_calibration_record",args.heldout_so3_calibration_record),("--sample_manifest",args.sample_manifest)): append_if(cmd,flag,value)
        cmd.extend(["--protocol_version",args.protocol_version])
        if args.native_resolution: cmd.append("--native_resolution")
        if args.model_offload: cmd.append("--model_offload")
        if args.dry_run: cmd.append("--dry_run")
        forward_extra(cmd,args.extra_arg)
        launch_sharded(cmd,args.num_gpus,out,lambda n,r,s:["--num_shards",str(n),"--shard_index",str(r),"--shard_sync_dir",str(s)]+(["--resume_state_dir",str(out/"resume")] if args.resume else []))
if __name__=="__main__": main()
