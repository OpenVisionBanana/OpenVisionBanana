#!/usr/bin/env python3
"""Small upstream-scorer regression checks, not dataset or inference parity."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def audit(marigold):
    import numpy as np
    from scipy import ndimage
    import torch
    import torch.nn.functional as F
    from sklearn.linear_model import RANSACRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from evaluation.depth.ppd_protocol import score_ppd_ransac_log, SUPPORTED_DATASETS
    from scripts.analysis.audit_upstream_ppd_scorer import original_functions, upstream_score
    scope=dict(np=np, torch=torch, F=F, math=math, ndimage=ndimage,
               RANSACRegressor=RANSACRegressor, make_pipeline=make_pipeline, PolynomialFeatures=PolynomialFeatures,
               _PPD_RANSAC_MODEL=make_pipeline(PolynomialFeatures(degree=1, include_bias=False),
                                             RANSACRegressor(max_trials=1000, random_state=42)), _PPD_RANSAC_SEED=42)
    source=marigold/'evaluation/depth/eval.py'
    alignment=marigold/'evaluation/src/util/alignment.py'
    src=original_functions(source, {'_ppd_depth_bounds_from_disp_name','_apply_ppd_mask',
        '_apply_diode_gradient_filter','_ppd_target_hw_from_disp_name','_resize_2d_float',
        '_resize_for_ppd_protocol','_clip_depth_like_ppd','align_log_space_ransac_ppd'},scope)
    src=original_functions(alignment, {'depth2log_space','log_space2depth'},src)
    yy,xx=np.mgrid[:32,:48].astype(np.float32)
    gt=1.+xx*.002+yy*.001
    mask=np.ones(gt.shape,bool);mask[:2]=False;mask[10:12,15:17]=False
    rows=[]
    for dataset in SUPPORTED_DATASETS:
        # Sparse mask on ETH3D keeps the prescribed full-size resize but makes
        # fitting cheap. The resizing itself is still compared upstream.
        valid=mask.copy()
        if dataset=='eth3d':valid[:]=False;valid[::8,::8]=True
        for name,pred in [('positive_fit',np.log(gt)*.7-.3),
                          ('nonpositive_fallback',-np.log(gt)*.7-.3)]:
            ours, _, _=score_ppd_ransac_log(dataset,pred,gt,valid)
            upstream, _=upstream_score(dataset,pred,gt,valid,src)
            delta=max(abs(ours[k]-upstream[k]) for k in upstream)
            rows.append(dict(dataset=dataset,case=name,max_metric_delta=delta,passed=delta<=1e-5))
    from evaluation.geometry_protocols import normal_metrics
    from scripts.analysis.audit_upstream_normal_scorers import load_marigold_metric_functions, load_module
    normal_source=marigold/'evaluation/normals/eval_normals.py'
    mg=load_marigold_metric_functions(normal_source)
    rino_source=marigold.parent/'RINO/tasks/surface_normals/evaluate/metrics.py'
    rino=load_module('synthetic_rino_metrics',rino_source)
    normal_rows=[]
    for invalid in (False,True):
        angles=np.resize(np.array([3.,10.,15.,25.,45.,90.],np.float32),(8,12))
        gt=np.zeros((8,12,3),np.float32);gt[...,2]=-1
        pred=np.zeros_like(gt);pred[...,0]=np.sin(np.deg2rad(angles));pred[...,2]=-np.cos(np.deg2rad(angles))
        valid=np.ones((8,12),bool);valid[0]=False;gt[~valid]=0
        pv=np.ones_like(valid)
        if invalid:pred[3:5,4:6]=0;pv[3:5,4:6]=False
        ours,_=normal_metrics(pred,pv,gt,valid,valid)
        error=mg['_compute_cosine_error'](torch.from_numpy(pred.transpose(2,0,1))[None],
                                          torch.from_numpy(gt.transpose(2,0,1))[None])
        upstream=mg['_metric_values'](error)
        scores=rino.metrics(pred,gt,valid,(0,1,2),(1,1,1))
        deltas=[abs(ours['mean_angle']-upstream['mean_angular_error']),
                abs(ours['median_angle']-upstream['median_angular_error']),
                abs(ours['mean_angle']-scores['mean']),abs(ours['median_angle']-scores['median'])]
        for ours_key,mg_key,rino_key in [('angle_11_25','sub11_25_error','acc_11.25'),
                                        ('angle_22_5','sub22_5_error','acc_22.5'),('angle_30','sub30_error','acc_30')]:
            deltas.extend((abs(ours[ours_key]*100-upstream[mg_key]),abs(ours[ours_key]-scores[rino_key])))
        normal_rows.append(dict(case='invalid_prediction_90deg' if invalid else 'fixed_axes',
                               max_metric_delta=max(deltas),passed=max(deltas)<=1e-3))
    return dict(status='PASS' if all(r['passed'] for r in rows+normal_rows) else 'FAIL',
                scope='Synthetic PPD fits/masks/resizing and fixed-axis angular scorers; not full dataset, inference or RINO calibration parity',
                upstream_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,alignment,normal_source,rino_source)},
                cases=rows,normal_cases=normal_rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--marigold-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.marigold_root)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if result['status']!='PASS':raise SystemExit('Upstream synthetic protocol regression failed')
    print('PASS: ten PPD and two Marigold/RINO angular-scorer regression cases')


if __name__=='__main__':main()
