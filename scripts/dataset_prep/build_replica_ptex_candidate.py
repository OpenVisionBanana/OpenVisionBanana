#!/usr/bin/env python3
"""Versioned Ptex RGB-only candidate at every retained control camera pose.

All target/mask/raw-depth bytes and frame selections are preserved. Every pose
must pass depth registration before either replacement archive becomes eligible.
No production dataset or public upload is replaced.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from common.replica_geometry import camera_rotation
from common.training.asset_lock import digest
from common.indexed_tar import read_reference_bytes
from scripts.analysis.audit_zs_candidate_safety import rows
from scripts.analysis.audit_zs_geometry_quality import Reader
from scripts.dataset_prep.build_geometry_readiness_manifests import write_json
from scripts.dataset_prep.rebuild_replica_upright_subset import add_member,png_bytes

CONTROL=Path('data/zs_upright_curated_20261002')
ASSETS=Path('data/3d_geo/replica/replica_v1_scene_assets.tar')


def unchanged_target_members(row):
    members=list(dict.fromkeys(row[k] for k in ('image','loss_mask','valid_mask','mask','depth_raw','raw_normals')
        if row.get(k) is not None))
    if any(not isinstance(m,str) or Path(m).is_absolute() or '..' in Path(m).parts for m in members):
        raise ValueError('Target members must be safe relative archive paths')
    return members


def build(campaign,scratch,renderer,variant):
    probe=campaign/'ptex_colour_candidates'/variant/'acceptance.json'
    proof=json.loads(probe.read_text())
    if not proof.get('registration_passed') or len(proof['frames'])!=12:
        raise ValueError('Fixed-pose colour probe must register all 12 control views')
    mesh=json.loads((campaign/'reports/independent_mesh.json').read_text())
    if mesh['status']!='PASS' or len(mesh['frames'])!=72:raise ValueError('Independent all-scene mesh verification incomplete')
    output=campaign/'ptex_full_candidates'/variant
    if output.exists():raise FileExistsError('Use a new candidate version; do not overwrite partial or accepted output')
    output.mkdir(parents=True);(output/'archives').mkdir();(output/'manifests').mkdir();(output/'reports').mkdir()
    scratch.mkdir(parents=True,exist_ok=True)
    full=json.loads((CONTROL/'manifests/zeroshot_masked_full.json').read_text())
    dep=next(e for e in full['datasets'] if e['name'].startswith('depth_') and 'replica' in e['name'])
    nor=next(e for e in full['datasets'] if e['name'].startswith('surface_normals_') and 'replica' in e['name'])
    dr,nr=rows(dep),rows(nor)
    if dr.keys()!=nr.keys():raise ValueError('Unpaired control IDs')
    readers=[Reader({**e,'offset_base':0}) for e in (dep,nor)]
    archives=[output/'archives'/f'replica_{task}_{variant}.tar' for task in ('depth','normals')]
    payloads=[[],[]];frames=[]
    scales=proof['fixed_colour_scales'];render_hash=digest(renderer)
    with tarfile.open(ASSETS) as source,tarfile.open(archives[0].with_suffix('.partial'),'w') as td,tarfile.open(archives[1].with_suffix('.partial'),'w') as tn:
        members=source.getmembers()
        for scene in sorted({r['metadata']['scene'] for _,r in dr.values()}):
            if Path(scene).name!=scene or scene in ('.','..'):raise ValueError('Unsafe scene name')
            stage=scratch/scene;stage.mkdir();prefix='replica_v1/'+scene+'/'
            for member in members:
                if member.isfile() and member.name.startswith(prefix) and (member.name==prefix+'mesh.ply' or '/textures/' in member.name):
                    relative=Path(member.name[len(prefix):])
                    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe source member')
                    destination=stage/relative;destination.parent.mkdir(parents=True,exist_ok=True)
                    with source.extractfile(member) as src,destination.open('wb') as dst:shutil.copyfileobj(src,dst)
            keys=sorted(k for k,(_,r) in dr.items() if r['metadata']['scene']==scene)
            rendered=stage/'rendered';rendered.mkdir();poses=[]
            for key in keys:
                if Path(key).name!=key:raise ValueError('Unsafe camera ID')
                m=dr[key][1]['metadata']
                if (m['width'],m['height'],m['hfov_degrees'],m['camera_world_up_axis'])!=(512,512,90.,'z'):
                    raise ValueError('Fixed renderer requires the exact verified control camera')
                rotation=np.diag([1,-1,-1])@camera_rotation(np.radians(m['yaw_degrees']),np.radians(m['pitch_degrees'])).T
                matrix=np.eye(4);matrix[:3,:3]=rotation;matrix[:3,3]=-rotation@np.array(m['camera_position'])
                poses.append(key+'\t'+'\t'.join(map(str,matrix.ravel())))
            posefile=stage/'poses.tsv';posefile.write_text('\n'.join(poses)+'\n')
            subprocess.run([str(renderer),str(stage/'mesh.ply'),str(stage/'textures'),str(posefile),str(rendered),
                str(scales['exposure']),str(scales['saturation'])],check=True)
            settings=(rendered/'sdk_colour_settings.txt').read_text().strip()
            scene_frames=[]
            for key in keys:
                row=dr[key][1];truth=readers[0].array(row,row['depth_raw'])
                valid=np.asarray(readers[0].image(row,row['loss_mask'],'L'))>0
                z=np.fromfile(rendered/(key+'.f32'),dtype='<f4').reshape(512,512)
                support=binary_erosion(valid,iterations=3)&np.isfinite(truth)&(truth>.1)
                hit=support&np.isfinite(z)&(z>.1)
                fraction=float(hit.sum()/max(1,support.sum()))
                error=float(np.quantile(np.abs(z[hit]-truth[hit])/truth[hit],.95)) if hit.any() else float('inf')
                if hit.sum()<10000 or fraction<.99 or error>=.02:
                    raise ValueError(f'Ptex full pose failed exact registration: {key}; coverage={fraction}, p95={error}')
                with Image.open(rendered/(key+'.ppm')) as image:rgb=image.convert('RGB');rgb_bytes=png_bytes(rgb)
                stats=dict(id=key,scene=scene,depth_p95_relative=error,compared_pixels=int(hit.sum()),coverage=fraction,
                    highlight_channel_fraction=float((np.asarray(rgb)>=254).mean()),sdk_settings=settings)
                frames.append(stats);scene_frames.append(stats)
                for index,(tar,reader,original) in enumerate(((td,readers[0],dr[key][1]),(tn,readers[1],nr[key][1]))):
                    item=copy.deepcopy(original)
                    item.pop('_metadata_member',None);item.pop('_metadata_line',None)
                    for member in unchanged_target_members(original):
                        # Copy GT bytes, not regenerated labels or validity masks.
                        add_member(tar,member,read_reference_bytes(reader.ref(original,member)))
                    item['conditioning_image']=f'conditioning_images/{key}.png'
                    add_member(tar,item['conditioning_image'],rgb_bytes)
                    item['metadata'].update(rgb_source='official_replica_ptex_fixed_pose',rgb_quality_candidate=variant,
                        ptex_fixed_colour_scales=scales,ptex_sdk_settings=settings,ptex_renderer_sha256=render_hash,
                        targets_and_masks_copied_unchanged=True,mirror_policy='no_reflection_rendering_same_as_control')
                    payloads[index].append(item)
            write_json(output/'reports'/f'{scene}_registration.json',dict(scene=scene,frames=scene_frames,status='PASS'))
            print(f'PTEX FULL {variant}: {scene} {len(keys)} poses PASS; targets/masks unchanged',flush=True)
            # Only this newly created node-local scene copy, never source assets.
            shutil.rmtree(stage)
        for tar,payload in zip((td,tn),payloads):add_member(tar,'metadata.jsonl',('\n'.join(json.dumps(r,sort_keys=True) for r in payload)+'\n').encode())
    for path in archives:
        path.with_suffix('.partial').replace(path)
        subprocess.run([sys.executable,str(ROOT/'scripts/dataset_prep/build_prepared_tar_index.py'),str(path)],check=True)
        path.with_suffix('.tar.sha256').write_text(digest(path)+'  '+str(path)+'\n')
    candidate=copy.deepcopy(full)
    for entry in candidate['datasets']:
        if entry['name'] in (dep['name'],nor['name']):
            path=archives[0 if entry['name']==dep['name'] else 1]
            entry.update(path=str(path),index_path=str(path)+'.ovb-index.sqlite')
    name='ptex_'+variant;manifest=output/'manifests'/f'{name}_full.json'
    candidate['quality_candidate']=dict(name=name,training_only=True,source_weights_unchanged=True,selection_and_gt_unchanged=True)
    write_json(manifest,candidate)
    write_json(output/'REGISTRATION.json',dict(status='PASS',frames=len(frames),scenes=len({f['scene'] for f in frames}),
        depth_p95_relative_max=max(f['depth_p95_relative'] for f in frames),renderer_sha256=render_hash,
        colour_probe_sha256=digest(probe),production_adopted=False,targets_and_masks_copied_unchanged=True))
    write_json(output/'CANDIDATES.json',{name:dict(manifest=str(manifest),sha256=digest(manifest),
        status='CANDIDATE_REQUIRES_PRODUCTION_LOADER_ACCEPTANCE',frames=len(frames))})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','scratch','renderer'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--variant',choices=('official','half_exposure_neutral_saturation'),required=True)
    a=p.parse_args();build(a.campaign,a.scratch,a.renderer,a.variant)


if __name__=='__main__':main()
