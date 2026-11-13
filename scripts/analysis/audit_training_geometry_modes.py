#!/usr/bin/env python3
"""Exercise actual targets through both loss modes; report label-quality signals."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import train_dreambooth_partial_flux2_klein_img2img as trainer
from common.surface_normal_encoding import decode_surface_normals_rgb
from common.training.asset_lock import digest
from common.training.benchmark_exclusions import exclusion_reason, load_registry
from dataset_prep.surface_normals.edge_aware_plane import target_statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--split-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--samples-per-source',type=int,default=16)
    p.add_argument('--source-name',action='append',default=None,
                   help='Audit only these named sources (repeatable); default audits every geometry source')
    p.add_argument('--depth-target-encoding',default='vision_banana_rgb')
    p.add_argument('--depth-prompt-contract',default='canonical_v2')
    p.add_argument('--normal-prompt-contract',default='vision_banana_v3')
    a=p.parse_args();specs,_,_,_=trainer.load_dataset_specs(str(a.manifest))
    if a.source_name and set(a.source_name)-{spec.name for spec in specs}:
        raise ValueError('Unknown requested source names in manifest')
    registry,_=load_registry(a.split_root);results=[]
    for spec in specs:
        if not spec.name.startswith(('depth_','surface_normals_')):continue
        if a.source_name and spec.name not in a.source_name:continue
        index=trainer.PreparedDatasetIndex(spec,'image','conditioning_image','text')
        # The manifest builder separately validates every source ID/exclusion.
        chosen=np.linspace(0,len(index.rows)-1,min(a.samples_per_source,len(index.rows))).round().astype(int)
        data=trainer.DreamBoothDataset.__new__(trainer.DreamBoothDataset)
        data.instance_prompt='fallback';data.dataset_pool=None;data.rows=index.rows
        data.args=SimpleNamespace(image_column='image',cond_image_column='conditioning_image',
            loss_mask_policy='auto',allow_missing_geometry_masks=False,
            depth_target_resize_policy='metric_field_then_encode',normal_target_resize_policy='vector_field_renormalize_then_encode',
            normal_target_encoding='vision_banana_x_left_red',depth_target_encoding=a.depth_target_encoding,
            depth_prompt_contract=a.depth_prompt_contract,normal_prompt_contract=a.normal_prompt_contract,
            center_crop=False,random_flip=False)
        data._validate_loss_mask_policy()
        for ordinal in chosen:
            row=index.rows[ordinal]
            source_image=data._load_row_image(row,row.image,'image')
            source_condition=data._load_row_image(row,row.conditioning_image,'conditioning_image')
            source_mask=data._load_row_loss_mask(row)
            if source_image.size!=source_condition.size or source_image.size!=source_mask.size:
                raise ValueError(f'Unregistered source dimensions: {spec.name}/{row.row_id}')
            width,height=source_image.size
            size=(max(32,round(height/width*512/32)*32),512)
            data.args.loss_mask_policy='auto';masked=data._load_example(row,0,0,size)
            data.args.loss_mask_policy='disabled';unmasked=data._load_example(row,0,0,size)
            if not np.array_equal(masked['instance_images'].numpy(),unmasked['instance_images'].numpy()):
                raise ValueError('Masking changed geometry preprocessing')
            if 'loss_mask' not in masked or 'loss_mask' in unmasked:raise ValueError('Wrong loss-mask routing')
            for key in ('instance_images','cond_images'):
                if not np.isfinite(masked[key].numpy()).all():raise ValueError('Nonfinite training input')
            mask=masked['loss_mask'].numpy()[0]>0
            if not mask.any():raise ValueError('Empty transformed validity mask')
            result={'dataset':spec.name,'row_id':row.row_id,'size':list(size),
                    'mask_fraction':float(mask.mean()),'both_modes_identical_preprocessing':True}
            if trainer.normalize_task_name(row.task) in trainer.SURFACE_NORMAL_TASKS:
                image=((masked['instance_images'].permute(1,2,0).numpy()+1)*127.5).round().clip(0,255).astype('uint8')
                normals,valid=decode_surface_normals_rgb(image,encoding='vision_banana_x_left_red')
                result.update(target_statistics(normals,valid & mask))
                length=np.linalg.norm(normals[valid & mask],axis=-1)
                if not np.allclose(length,1.,atol=1e-4):raise ValueError('Normal normalization failed')
            results.append(result)
        print(f'PASS both modes {spec.name}: {len(index.rows)} rows, {len(chosen)} decoded samples',flush=True)
    output={'status':'PASS','validation_scope':'selected-source metadata/mask presence; evenly spaced decoded samples through real loader',
            'selected_source_names':a.source_name,
            'manifest':str(a.manifest.resolve()),'manifest_sha256':digest(a.manifest),
            'semantic_quality_certified':False,'samples':results,'total_samples':len(results)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
