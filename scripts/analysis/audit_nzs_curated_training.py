#!/usr/bin/env python3
"""Real-loader acceptance for additive curated NZS manifests in both loss modes."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import train_dreambooth_partial_flux2_klein_img2img as trainer
from scripts.dataset_prep.build_geometry_readiness_manifests import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads((args.campaign/'plan.json').read_text())
    prepared=json.loads((args.campaign/'PREPARED.json').read_text())
    added={e['name'] for e in plan['sources']}
    results=[]
    for variant in prepared['variants']:
        if variant in ('original_no_scannet','original_reference'):
            continue  # Preserved reference, not a claim that curation repaired it.
        manifest=args.campaign/'manifests'/f'nonzero_{variant}_geometry.json'
        specs,_,_,_=trainer.load_dataset_specs(str(manifest))
        for spec in specs:
            if spec.name not in added: continue
            index=trainer.PreparedDatasetIndex(spec,'image','conditioning_image','text')
            data=trainer.DreamBoothDataset.__new__(trainer.DreamBoothDataset)
            data.instance_prompt='fallback'; data.dataset_pool=None; data.rows=index.rows
            data.args=SimpleNamespace(image_column='image',cond_image_column='conditioning_image',
                loss_mask_policy='auto',allow_missing_geometry_masks=False,
                depth_target_resize_policy='metric_field_then_encode',normal_target_resize_policy='vector_field_renormalize_then_encode',
                normal_target_encoding='vision_banana_x_left_red',depth_target_encoding='vision_banana_rgb',
                depth_prompt_contract='canonical_v2',center_crop=False,random_flip=False)
            data._validate_loss_mask_policy()
            chosen=np.linspace(0,len(index.rows)-1,min(4,len(index.rows))).round().astype(int)
            for ordinal in chosen:
                row=index.rows[ordinal]
                target=data._load_row_image(row,row.image,'image')
                condition=data._load_row_image(row,row.conditioning_image,'conditioning_image')
                mask=data._load_row_loss_mask(row)
                if not target.size==condition.size==mask.size:
                    raise ValueError('Curated triplet lost registration')
                if data._load_row_image_size(row,row.image,'image') != target.size:
                    raise ValueError('Bucket dimensions ignored source crop')
                width,height=target.size
                size=(max(32,round(height/width*512/32)*32),512)
                data.args.loss_mask_policy='auto'; masked=data._load_example(row,0,0,size)
                data.args.loss_mask_policy='disabled'; unmasked=data._load_example(row,0,0,size)
                for key in ('instance_images','cond_images'):
                    a,b=masked[key].numpy(),unmasked[key].numpy()
                    if not np.array_equal(a,b) or not np.isfinite(a).all():
                        raise ValueError('Loss policy changed cropped preprocessing or produced nonfinite data')
                if 'loss_mask' not in masked or 'loss_mask' in unmasked:
                    raise ValueError('Incorrect loss-mask routing')
                results.append({'variant':variant,'source':spec.name,'row_id':row.row_id,
                                'crop_box':row.training_crop_box,'size':list(target.size)})
        print(f'PASS real-loader masked/unmasked {variant}',flush=True)
    write_json(args.campaign/'reports'/'loader_acceptance.json',{
        'status':'PASS','samples':results,'scope':'four evenly spaced retained rows/source/variant; not exhaustive semantic certification'})
    write_json(args.campaign/'READY.json',{'status':'subset_and_loader_checks_passed',
        'variants':prepared['variants'],'model_quality':'not tested; no training launched',
        'production_asset_lock':'freeze the selected manifest and current crop-capable code before training'})


if __name__=='__main__': main()
