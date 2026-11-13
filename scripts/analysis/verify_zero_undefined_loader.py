#!/usr/bin/env python3
"""Exhaustively check one zero-hole source through every training bucket.

Resumable SQLite reports bind every row to the immutable manifest/index and
loader code. A source-zero mask alone is never considered final certification.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import sys
import threading
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from common.training.asset_lock import digest
from common.surface_normal_encoding import decode_surface_normals_rgb
from scripts.dataset_prep.build_geometry_readiness_manifests import write_json

BUCKETS=((512,512),(448,576),(576,448),(416,608),(608,416))


def main():
    import train_dreambooth_partial_flux2_klein_img2img as trainer
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--source',type=int,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--report-directory',type=Path)
    a=p.parse_args();manifest=a.campaign/'subsets/manifests/zero_undefined_geometry.json'
    if not manifest.exists():
        print('GATED: zero-undefined manifest unavailable; no domain silently removed');return
    specs,_,_,_=trainer.load_dataset_specs(str(manifest));spec=specs[a.source]
    index=trainer.PreparedDatasetIndex(spec,'image','conditioning_image','text')
    identity=json.dumps(dict(manifest_sha256=digest(manifest),index_sha256=digest(Path(spec.index_path)),
        trainer_sha256=digest(ROOT/'train_dreambooth_partial_flux2_klein_img2img.py'),
        verifier_sha256=digest(Path(__file__)),
        source=spec.name,buckets=BUCKETS),sort_keys=True)
    reports=a.report_directory or a.campaign/'reports';reports.mkdir(parents=True,exist_ok=True)
    target=reports/f'zero_final_{a.source}.sqlite'
    db=sqlite3.connect(target);db.execute('create table if not exists metadata (key text primary key,value text)')
    db.execute('create table if not exists checks (row_number integer primary key,id text,minimum_validity real)')
    db.execute('create table if not exists invalid (row_number integer primary key,id text,reasons text)')
    old=db.execute("select value from metadata where key='identity'").fetchone()
    if old and old[0]!=identity:raise ValueError('Cannot reuse different zero-hole checks')
    db.execute('insert or replace into metadata values (?,?)',('identity',identity));db.commit()
    done={n for n, in db.execute('select row_number from checks')};local=threading.local()
    def check(ordinal):
        if not hasattr(local,'data'):
            data=trainer.DreamBoothDataset.__new__(trainer.DreamBoothDataset)
            data.instance_prompt='fallback';data.dataset_pool=None;data.rows=index.rows
            data.args=SimpleNamespace(image_column='image',cond_image_column='conditioning_image',
                loss_mask_policy='auto',allow_missing_geometry_masks=False,
                depth_target_resize_policy='metric_field_then_encode',normal_target_resize_policy='vector_field_renormalize_then_encode',
                normal_target_encoding='vision_banana_x_left_red',depth_target_encoding='relative_inverse_grayscale',
                depth_prompt_contract='campaign_detailed_inverse',normal_prompt_contract='campaign_short',center_crop=False,random_flip=False)
            local.data=data
        minimum=1.; reasons=[]
        for size in BUCKETS:
            sample=local.data._load_example(index.rows[ordinal],0,ordinal,size)
            mask=sample['loss_mask'].numpy()
            if not np.isfinite(mask).all():raise ValueError('Nonfinite validity')
            for key in ('instance_images','cond_images'):
                if not np.isfinite(sample[key].numpy()).all():raise ValueError('Nonfinite training input')
            if trainer.normalize_task_name(index.rows[ordinal].task) in trainer.SURFACE_NORMAL_TASKS:
                rgb=((sample['instance_images'].permute(1,2,0).numpy()+1)*127.5).round().clip(0,255).astype('uint8')
                _,vector_valid=decode_surface_normals_rgb(rgb,encoding='vision_banana_x_left_red')
                if not vector_valid.all():
                    reasons.append(dict(bucket=size,reason='undefined_normal_vector_after_resize',pixels=int((~vector_valid).sum())))
            minimum=min(minimum,float(mask.min()))
            if float(mask.min())<1-1e-6:
                reasons.append(dict(bucket=size,reason='undefined_mask_after_preprocessing',pixels=int((mask<1-1e-6).sum())))
        return ordinal,index.rows[ordinal].row_id,minimum,reasons
    pending=[i for i in range(len(index.rows)) if i not in done]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        count=0
        for start in range(0,len(pending),256):
            for result in ex.map(check,pending[start:start+256]):
                ordinal,row_id,minimum,reasons=result;count+=1
                db.execute('insert into checks values (?,?,?)',(ordinal,row_id,minimum))
                if reasons:db.execute('insert into invalid values (?,?,?)',(ordinal,row_id,json.dumps(reasons)))
            db.commit();print(f'ZERO FINAL {spec.name}: {len(done)+count}/{len(index.rows)} x {len(BUCKETS)} buckets',flush=True)
    db.commit();actual=db.execute('select count(*) from checks').fetchone()[0]
    if actual!=len(index.rows):raise ValueError('Incomplete final-loader coverage')
    invalid=[dict(row_number=n,id=k,reasons=json.loads(reasons)) for n,k,reasons in db.execute('select * from invalid order by row_number')]
    db.execute('insert or replace into metadata values (?,?)',('complete','true'));db.commit();db.close()
    write_json(reports/f'zero_final_{a.source}.json',dict(status='INVALID_ROWS_FOUND' if invalid else 'PASS',source=spec.name,rows=actual,
        buckets=len(BUCKETS),all_transformed_pixels_valid=not invalid,invalid_rows=invalid,identity=json.loads(identity),
        scope='Every source row and bucket; seed42 campaign with no flips and no validity erosion. No augmentation-invariance claim for other configurations.'))


if __name__=='__main__':main()
