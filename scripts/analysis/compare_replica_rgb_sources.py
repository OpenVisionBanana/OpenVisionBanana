#!/usr/bin/env python3
"""Diagnostic only: packaged vertex RGB, fixed-pose Ptex and NICE-SLAM inputs.

NICE-SLAM frames are different poses/calibration. Only the first two columns
are a controlled same-camera comparison. No training assets are changed.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.analysis.audit_zs_geometry_quality import Reader
from scripts.analysis.fetch_remote_zip_subset import read_members, extract_member


def rows(entry):
    with sqlite3.connect('file:'+entry['index_path']+'?mode=ro',uri=True) as c:
        return {r['id']:r for (payload,) in c.execute('select payload from rows') if (r:=json.loads(payload))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError('Choose a fresh diagnostic output')
    a.output.mkdir(parents=True)
    url='https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip'
    members=read_members(url)
    publication=Path('data/geometry_training_publication_20261004/upload')
    current=dict(path=str(publication/'archives/depth_legacy_replica-00000.tar'),
                 index_path=str(publication/'indices/depth_legacy_replica-00000.tar.ovb-index.sqlite'),offset_base=0)
    candidate_root=Path('data/zs_quality_candidates_20261003/ptex_full_candidates/official')
    manifest=json.loads((candidate_root/'manifests/ptex_official_full.json').read_text())
    ptex={**next(e for e in manifest['datasets'] if e['name']=='depth_legacy_replica'),'offset_base':0}
    cr,pr=Reader(current),Reader(ptex)
    cd,pd=rows(current),rows(ptex)
    cases=[('office0','replica_office_0_000000_upright'),
           ('office1','replica_office_1_000245_upright'),
           ('room0','replica_room_0_000265_upright')]
    try: font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    except OSError: font=ImageFont.load_default()
    canvas=Image.new('RGB',(1536,70+len(cases)*558),'white')
    draw=ImageDraw.Draw(canvas)
    headers=['Current training RGB (vertex colours)', 'Ptex candidate (same camera; not adopted)', 'NICE-SLAM input (different camera pose)']
    for i,title in enumerate(headers): draw.text((i*512+6,15),title,fill='black',font=font)
    report=dict(url=url,zip_members=len(members),current_archive=current['path'],ptex_candidate_archive=ptex['path'],
                comparison='Columns 1 and 2 have the same sample ID and pose; column 3 is a different trajectory/view',
                production_data_unchanged=True,examples=[])
    for i,(scene,sample) in enumerate(cases):
        name=f'Replica/{scene}/results/frame000000.jpg'
        blob=extract_member(url,members[name]); (a.output/(scene+'_nice_frame000000.jpg')).write_bytes(blob)
        nice=Image.open(io.BytesIO(blob)).convert('RGB')
        depname=f'Replica/{scene}/results/depth000000.png'
        db=extract_member(url,members[depname]); (a.output/(scene+'_nice_depth000000.png')).write_bytes(db)
        di=Image.open(io.BytesIO(db)); da=np.asarray(di)
        vertex=cr.image(cd[sample],cd[sample]['conditioning_image'])
        textured=pr.image(pd[sample],pd[sample]['conditioning_image'])
        for field in ('image','loss_mask'):
            mode='L' if field=='loss_mask' else 'RGB'
            assert np.array_equal(np.asarray(cr.image(cd[sample],cd[sample][field],mode)),
                                  np.asarray(pr.image(pd[sample],pd[sample][field],mode))), 'Candidate changed GT'
        vertex.save(a.output/(scene+'_current_vertex.png'))
        textured.save(a.output/(scene+'_same_pose_ptex.png'))
        y=70+i*558
        for j,image in enumerate((vertex,textured,nice)):
            canvas.paste(ImageOps.pad(image,(512,512),method=Image.Resampling.LANCZOS,color=(235,235,235)),(512*j,y))
        draw.text((6,y+518),scene+' | '+sample+' | NICE frame000000: different viewpoint, not registered',fill='black',font=font)
        report['examples'].append(dict(scene=scene,sample=sample,nice_member=name,nice_sha256=hashlib.sha256(blob).hexdigest(),
            nice_size=list(nice.size),nice_depth_size=list(di.size),nice_depth_dtype=str(da.dtype),
            nice_depth_min=int(da.min()),nice_depth_max=int(da.max()),current_size=list(vertex.size),
            ptex_metadata=pd[sample]['metadata'],candidate_targets_and_masks_unchanged=True))
    metadata='Replica/cam_params.json'
    if metadata in members:
        report['nice_camera_parameters']=json.loads(extract_member(url,members[metadata]))
    canvas.save(a.output/'rgb_source_comparison.jpg',quality=95)
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='examples'},indent=2))
    for e in report['examples']: print(e['scene'],e['nice_size'],e['nice_sha256'])


if __name__=='__main__':main()
