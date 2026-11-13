#!/usr/bin/env python3
"""Evaluate FLUX.2-Klein on one, several, or all supported depth benchmarks."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.geometry_protocol_prompts import DETAILED_INVERSE_DEPTH_PROMPT
from common.release_eval import add_common_arguments, append_if, evaluator_python, forward_extra, launch_sharded, locate_dataset, selected_datasets
SUPPORTED_DATASETS=("nyu","ibims1","diode","kitti","kitti_eigen","eth3d","nuscenes","vkitti2","scannet")
PPD_DATASETS=("nyu","kitti_eigen","eth3d","scannet","diode")

def main():
    p=argparse.ArgumentParser(description=__doc__); add_common_arguments(p, SUPPORTED_DATASETS)
    p.add_argument("--split", default="test"); p.add_argument("--cases", nargs="+", default=["standard"])
    p.add_argument("--model-offload", action="store_true"); p.add_argument("--resume", action="store_true")
    p.add_argument("--native-resolution", action=argparse.BooleanOptionalAction, default=True, help="Generate at the input resolution (paper setting). --no-native-resolution uses the pipeline default size.")
    p.add_argument("--depth-encoding", choices=("relative_inverse_grayscale","relative_linear_grayscale","vision_banana_rgb","barron_scalar_grayscale","grayscale"), default="relative_inverse_grayscale")
    p.add_argument("--prompt", default=DETAILED_INVERSE_DEPTH_PROMPT); p.add_argument("--heldout-calibration-record"); p.add_argument("--sample-manifest")
    p.add_argument("--benchmark-protocol", choices=("geometry_v5","marigold_v2_ppd"), default="marigold_v2_ppd", help="Paper protocol: marigold_v2_ppd (Marigold V2 splits and alignment).")
    p.add_argument("--benchmark-split-root", type=Path, help="Marigold V2 evaluation/data_split directory for the PPD lane. Default: <data-root>/prepared/V1_prepared_seg_geo/manifests/benchmark_splits when it exists.")
    p.add_argument("--ppd-primary-alignment", choices=("auto","disparity_affine","log_depth_ransac"), default="auto", help="Select by encoding; inverse grayscale defaults to disparity-affine on PPD splits.")
    p.add_argument("--protocol-version", help="Optional recorded contract version; PPD defaults to marigold-v2-ppd-v1.")
    args=p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    supported=PPD_DATASETS if args.benchmark_protocol=="marigold_v2_ppd" else SUPPORTED_DATASETS
    if args.benchmark_protocol=="marigold_v2_ppd" and args.benchmark_split_root is None:
        candidates=[args.data_root/"prepared"/f"V1_prepared_seg_geo{suffix}"/"manifests"/"benchmark_splits" for suffix in ("","_nzs")]
        candidates+=[args.data_root/"manifests"/"benchmark_splits", args.data_root/"benchmark_splits", args.data_root/"data_split"]
        args.benchmark_split_root=next((c for c in candidates if c.is_dir()),None)
        if args.benchmark_split_root is None:
            p.error("--benchmark-split-root is required for marigold_v2_ppd (no manifests/benchmark_splits found under --data-root)")
    for dataset in selected_datasets(args.datasets, supported):
        data=locate_dataset(args.data_root,args.output_dir,"Depth",dataset); out=args.output_dir/dataset; out.mkdir(parents=True,exist_ok=True)
        cmd=evaluator_python("evaluation/depth/inference_partial_flux2_klein_img2img.py")+["--model",args.model,"--model_family","klein","--dataset",dataset,"--data_root",str(data),"--out_dir",str(out),"--split",args.split,"--steps",str(args.steps),"--guidance_scale",str(args.guidance),"--seed",str(args.seed),"--depth_encoding",args.depth_encoding,"--cases",*args.cases]
        append_if(cmd,"--transformer_path",args.transformer_path); append_if(cmd,"--limit",args.limit); append_if(cmd,"--prompt",args.prompt); append_if(cmd,"--heldout_depth_calibration_record",args.heldout_calibration_record); append_if(cmd,"--sample_manifest",args.sample_manifest)
        cmd.extend(["--benchmark_protocol",args.benchmark_protocol])
        append_if(cmd,"--benchmark_split_root",args.benchmark_split_root)
        cmd.extend(["--ppd_primary_alignment",args.ppd_primary_alignment])
        append_if(cmd,"--protocol_version",args.protocol_version or ("marigold-v2-ppd-v1" if args.benchmark_protocol=="marigold_v2_ppd" else None))
        if args.native_resolution: cmd.append("--native_resolution")
        if args.model_offload: cmd.append("--model_offload")
        if args.dry_run: cmd.append("--dry_run")
        forward_extra(cmd,args.extra_arg)
        launch_sharded(cmd,args.num_gpus,out,lambda n,r,s:["--num_shards",str(n),"--shard_index",str(r),"--shard_sync_dir",str(s)]+(["--resume_state_dir",str(out/"resume")] if args.resume else []))
if __name__=="__main__": main()
