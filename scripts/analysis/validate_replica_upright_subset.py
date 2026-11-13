#!/usr/bin/env python3
"""Accept an upright-render Replica candidate only after paired/loader checks."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from common.training.asset_lock import digest
from scripts.analysis.audit_zs_geometry_quality import Reader, paired_geometry
from scripts.dataset_prep.curate_nzs_geometry import readonly
from scripts.dataset_prep.build_geometry_readiness_manifests import write_json
from scripts.dataset_prep.curate_zs_geometry_pairs import POLICIES


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args(); root=a.campaign/'data'
    sources=[]; rows=[]
    for task, basename in [('depth','replica_depth_upright.tar'),('normals','replica_normals_upright.tar')]:
        path=root/'archives'/basename; index=Path(str(path)+'.ovb-index.sqlite')
        entry=dict(name='depth_replica_upright' if task=='depth' else 'surface_normals_replica_upright',
                   type='indexed_prepared_tar',path=str(path.resolve()),index_path=str(index.resolve()),
                   weight=.5,offset_base=0)
        sources.append(entry)
        with readonly(index) as db:rows.append({r['id']:r for payload, in db.execute('select payload from rows') if (r:=json.loads(payload))})
    if rows[0].keys()!=rows[1].keys():raise ValueError('Rebuilt depth/normal IDs differ')
    readers=list(map(Reader,sources));geometry_checks=[]; selected=[]
    scenes=sorted({r['metadata']['scene'] for r in rows[0].values()})
    if len(scenes)!=18:raise ValueError('Rebuild lost original scene coverage')
    for scene in scenes:
        keys=sorted(k for k,r in rows[0].items() if r['metadata']['scene']==scene)
        chosen=[keys[i] for i in np.linspace(0,len(keys)-1,min(16,len(keys))).round().astype(int)]
        for key in chosen:
            dep,nor=rows[0][key],rows[1][key]
            rgb=readers[0].image(dep,dep['conditioning_image'])
            if rgb.tobytes()!=readers[1].image(nor,nor['conditioning_image']).tobytes():raise ValueError('RGB registration differs')
            valid=(np.asarray(readers[0].image(dep,dep['loss_mask'],'L'))>0)&(np.asarray(readers[1].image(nor,nor['loss_mask'],'L'))>0)
            if valid.mean()<.99:raise ValueError('Candidate exceeds 1% joint invalid coverage')
            depth=readers[0].array(dep,dep['depth_raw'])
            result=paired_geometry(np.asarray(readers[1].image(nor,nor['image'])),nor,depth,dep,valid,'vision_banana_x_left_red')
            if result['status']!='checked' or result['mean_angle']>1:raise ValueError('Depth/normal basis acceptance failed')
            geometry_checks.append(dict(id=key,**result))
        selected.append(keys[len(keys)//2])
    loader=json.loads((root/'reports/loader.json').read_text())
    if loader.get('status')!='PASS' or loader.get('total_samples',0)<2:raise ValueError('Production loader acceptance missing')
    canvas=Image.new('RGB',(768,len(selected)*210),'white');draw=ImageDraw.Draw(canvas)
    for i,key in enumerate(selected):
        dep,nor=rows[0][key],rows[1][key]
        pictures=[readers[0].image(dep,dep['conditioning_image']),readers[0].image(dep,dep['image']),readers[1].image(nor,nor['image'])]
        for j,image in enumerate(pictures):image.thumbnail((250,180));canvas.paste(image,(j*256,i*210))
        draw.text((2,i*210+185),key+' | upright RGB / depth / normals',fill='black')
    canvas.save(root/'reports/retained_upright.jpg',quality=92)
    write_json(root/'reports/acceptance.json',dict(status='PASS',frames=len(rows[0]),scenes=len(scenes),
        sampled_geometry_checks=geometry_checks,both_loss_modes_checked=True,semantic_quality_certified=False))
    write_json(root/'READY.json',dict(status='upright_candidate_and_loader_checks_passed',frames=len(rows[0]),scenes=len(scenes),
        rgb_texturing='Original mesh vertex colours; not a Ptex texture restoration',
        note='Separate training candidate; original/manifests/running jobs untouched',
        before_training='Select an explicit fresh-training manifest and freeze the model/runtime/code asset lock'))
    print(f'PASS upright candidate: {len(rows[0])} pairs/{len(scenes)} scenes',flush=True)


if __name__=='__main__':main()
