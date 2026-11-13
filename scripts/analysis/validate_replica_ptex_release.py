#!/usr/bin/env python3
"""All-frame Ptex validation and colour review; no production files are mutated."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.analysis.audit_zs_geometry_quality import Reader
from common.indexed_tar import read_reference_bytes
from common.training.asset_lock import digest
from scripts.dataset_prep.export_geometry_training_release import write

VARIANTS=('official','half_exposure_neutral_saturation')
CAMERA_FIELDS=('camera_position','camera_world_up_axis','yaw_degrees','pitch_degrees','hfov_degrees','width','height')


def load(entry):
    with sqlite3.connect('file:'+entry['index_path']+'?mode=ro',uri=True) as c:
        assert c.execute('pragma integrity_check').fetchone()[0]=='ok'
        result=[json.loads(p) for (p,) in c.execute('select payload from rows order by row_number')]
    assert len({r['id'] for r in result})==len(result)
    return {r['id']:r for r in result},Reader({**entry,'offset_base':0})


def choose_entries(manifest):
    return [next(e for e in manifest['datasets'] if e['name'].startswith(prefix) and 'replica' in e['name'])
            for prefix in ('depth_','surface_normals_')]


def colour_stats(rgb):
    gray=rgb.astype(np.float32).mean(-1)
    return dict(mean=float(gray.mean()),std=float(gray.std()),
                clipped_channels=float((rgb>=254).mean()),
                fully_white_pixels=float((rgb>=254).all(-1).mean()),
                fully_black_pixels=float((rgb<=1).all(-1).mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,default=Path('data/zs_quality_candidates_20261003'))
    p.add_argument('--control-manifest',type=Path,default=Path('data/zs_upright_curated_20261002/manifests/zeroshot_masked_full.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a fresh diagnostic directory')
    a.output.mkdir(parents=True)
    controls=choose_entries(json.loads(a.control_manifest.read_text()))
    crow,creaders=list(zip(*[load(e) for e in controls]))
    assert set(crow[0])==set(crow[1])
    candidates={}; proofs={}; assets=[]
    for variant in VARIANTS:
        r=a.campaign/'ptex_full_candidates'/variant
        proof=json.loads((r/'REGISTRATION.json').read_text())
        assert proof['status']=='PASS' and proof['frames']==len(crow[0]) and proof['scenes']==18
        assert proof['targets_and_masks_copied_unchanged'] and proof['depth_p95_relative_max']<.02
        entries=choose_entries(json.loads((r/'manifests'/f'ptex_{variant}_full.json').read_text()))
        rows,readers=list(zip(*[load(e) for e in entries]))
        for e,rs in zip(entries,rows):
            assert set(rs)==set(crow[0])
            sha=digest(Path(e['path']))
            assert sha==Path(e['path']+'.sha256').read_text().split()[0]
            assets.append(dict(path=e['path'],sha256=sha,index_path=e['index_path'],index_sha256=digest(Path(e['index_path']))))
        candidates[variant]=(rows,readers,entries)
        proofs[variant]=dict(path=str(r/'REGISTRATION.json'),sha256=digest(r/'REGISTRATION.json'),**proof)

    def inspect(sample):
        result=dict(id=sample,scene=crow[0][sample]['metadata']['scene'],errors=[],colours={})
        try:
            control=crow[0][sample]
            original=np.asarray(creaders[0].image(control,control['conditioning_image']))
            result['colours']['vertex']=colour_stats(original)
            for variant,(rows,readers,_) in candidates.items():
                rgb=[]
                for task in (0,1):
                    new,old=rows[task][sample],crow[task][sample]
                    assert new['text']==old['text'] and new['task']==old['task']
                    assert new.get('normal_encoding')==old.get('normal_encoding')
                    for field in CAMERA_FIELDS:
                        assert new['metadata'][field]==old['metadata'][field],f'Changed calibration: {field}'
                    assert new['metadata']['rgb_source']=='official_replica_ptex_fixed_pose'
                    assert new['metadata']['targets_and_masks_copied_unchanged']
                    for field in ('image','loss_mask','depth_raw','raw_normals'):
                        if old.get(field):
                            assert read_reference_bytes(readers[task].ref(new,new[field]))==read_reference_bytes(creaders[task].ref(old,old[field])),f'Changed {variant}/{field}'
                    im=readers[task].image(new,new['conditioning_image'])
                    assert im.size==(512,512)
                    rgb.append(np.asarray(im))
                assert np.array_equal(*rgb),'Depth and normal conditioning differ'
                assert not np.array_equal(rgb[0],original),'Ptex did not replace the vertex RGB'
                stats=colour_stats(rgb[0]); assert stats['std']>0 and stats['fully_black_pixels']<.99
                result['colours'][variant]=stats
        except Exception as e:result['errors']=[f'{type(e).__name__}: {e}']
        return result

    frames=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i,r in enumerate(pool.map(inspect,sorted(crow[0])),1):
            frames.append(r)
            if i%200==0 or i==len(crow[0]):print(f'PTEX QA {i}/{len(crow[0])}; failures={sum(bool(x["errors"]) for x in frames)}',flush=True)
    errors=[r for r in frames if r['errors']]
    summary=dict(status='PASS_RGB_REPLACEMENT_CONTRACT' if not errors else 'FAIL',frames=len(frames),
        scenes=dict(Counter(f['scene'] for f in frames)),assets=assets,registration_proofs=proofs,errors=errors,
        targets_masks_depth_and_selection_unchanged=not errors,production_adopted=False,
        training_quality_improvement_proven=False,visual_review_required=True,statistics={})
    if not errors:
        for variant in ('vertex',)+VARIANTS:
            summary['statistics'][variant]={}
            for field in frames[0]['colours'][variant]:
                values=np.array([f['colours'][variant][field] for f in frames])
                summary['statistics'][variant][field]=dict(mean=float(values.mean()),p95=float(np.quantile(values,.95)),max=float(values.max()),min=float(values.min()))
        for variant in VARIANTS:
            summary['statistics'][variant]['scenes']={s:{field:float(np.mean([f['colours'][variant][field] for f in frames if f['scene']==s]))
                for field in frames[0]['colours'][variant]} for s in summary['scenes']}
    write(a.output/'QA.json',summary)
    (a.output/'frames.jsonl').write_text(''.join(json.dumps(f,sort_keys=True)+'\n' for f in frames))
    if errors:raise SystemExit(1)
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    except OSError:font=ImageFont.load_default()
    selected=[]
    for scene in summary['scenes']:
        ids=sorted(f['id'] for f in frames if f['scene']==scene)
        selected.append(ids[len(ids)//2])
    extreme=[max(frames,key=lambda f:f['colours'][VARIANTS[1]][field])['id']
             for field in ('fully_white_pixels','fully_black_pixels','clipped_channels')]
    pages=[selected[i:i+6] for i in range(0,len(selected),6)]+[list(dict.fromkeys(extreme))]
    for n,ids in enumerate(pages):
        tile=256;canvas=Image.new('RGB',(3*tile,48+len(ids)*290),'white');draw=ImageDraw.Draw(canvas)
        for k,title in enumerate(('Vertex colours (current)','Ptex / SDK colour defaults','Ptex / half exposure, neutral')):
            draw.text((tile*k+3,8),title,fill='black',font=font)
        for i,sample in enumerate(ids):
            row=crow[0][sample];im=creaders[0].image(row,row['conditioning_image'])
            images=[im]+[candidates[v][1][0].image(candidates[v][0][0][sample],candidates[v][0][0][sample]['conditioning_image']) for v in VARIANTS]
            y=48+i*290
            for k,image in enumerate(images):canvas.paste(image.resize((256,256),Image.Resampling.LANCZOS),(k*tile,y))
            draw.text((3,y+260),sample+' (same camera in all columns)',fill='black',font=font)
        canvas.save(a.output/f'colour_review_{n+1}.jpg',quality=95)
    print('PASS full-frame Ptex contract; colour galleries still require visual review',flush=True)


if __name__=='__main__':main()
