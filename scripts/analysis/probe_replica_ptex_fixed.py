#!/usr/bin/env python3
"""Fixed-pose Ptex RGB probe: require depth registration, retain vertex control."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from common.replica_geometry import camera_rotation
from scripts.analysis.audit_zs_candidate_safety import rows
from scripts.analysis.audit_zs_geometry_quality import Reader
from scripts.dataset_prep.build_geometry_readiness_manifests import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('pool','assets','scratch','output','renderer'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--exposure-scale',type=float,default=1.);p.add_argument('--saturation-scale',type=float,default=1.)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);a.scratch.mkdir(parents=True,exist_ok=True)
    template=json.loads((a.pool/'replica_geometry.json').read_text());dep=template['datasets'][0]
    dr=rows(dep);reader=Reader({**dep,'offset_base':0});reports=[];panels=[]
    with tarfile.open(a.assets) as archive:
        for scene in ('apartment_0','office_0','frl_apartment_4'):
            scene_root=a.scratch/scene;scene_root.mkdir()
            prefix='replica_v1/'+scene+'/'
            for member in archive.getmembers():
                if member.isfile() and member.name.startswith(prefix) and (member.name==prefix+'mesh.ply' or '/textures/' in member.name):
                    relative=Path(member.name[len(prefix):])
                    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe source member')
                    target=scene_root/relative;target.parent.mkdir(parents=True,exist_ok=True)
                    with archive.extractfile(member) as src,target.open('wb') as dst:
                        import shutil;shutil.copyfileobj(src,dst)
            keys=sorted(k for k,(_,r) in dr.items() if r['metadata']['scene']==scene)
            chosen=[keys[i] for i in np.linspace(0,len(keys)-1,4).round().astype(int)]
            posefile=scene_root/'poses.tsv';rendered=scene_root/'rendered';rendered.mkdir()
            poses=[]
            for key in chosen:
                m=dr[key][1]['metadata'];r=camera_rotation(np.radians(m['yaw_degrees']),np.radians(m['pitch_degrees']))
                rotation=np.diag([1,-1,-1])@r.T
                matrix=np.eye(4);matrix[:3,:3]=rotation;matrix[:3,3]=-rotation@np.array(m['camera_position'])
                poses.append(key+'\t'+'\t'.join(map(str,matrix.ravel())))
            posefile.write_text('\n'.join(poses)+'\n')
            subprocess.run([str(a.renderer),str(scene_root/'mesh.ply'),str(scene_root/'textures'),str(posefile),str(rendered),
                str(a.exposure_scale),str(a.saturation_scale)],check=True)
            for key in chosen:
                row=dr[key][1];truth=reader.array(row,row['depth_raw']);mask=np.asarray(reader.image(row,row['loss_mask'],'L'))>0
                candidate=np.fromfile(rendered/(key+'.f32'),dtype='<f4').reshape(512,512)
                rgb=Image.open(rendered/(key+'.ppm')).convert('RGB');original=reader.image(row,row['conditioning_image'])
                good=binary_erosion(mask,iterations=3)&np.isfinite(truth)&(truth>.1)&(candidate>.1)
                relative=np.abs(candidate[good]-truth[good])/truth[good]
                passed=len(relative)>10000 and np.quantile(relative,.95)<.02
                stats=dict(id=key,scene=scene,registration_passed=bool(passed),depth_p95_relative=float(np.quantile(relative,.95)),
                    compared_pixels=int(good.sum()),vertex_rgb_std=float(np.asarray(original).std()),ptex_rgb_std=float(np.asarray(rgb).std()),
                    vertex_extreme_fraction=float((np.asarray(original)>=254).mean()),ptex_extreme_fraction=float((np.asarray(rgb)>=254).mean()),
                    ptex_dark_fraction=float((np.asarray(rgb)<=1).mean()),
                    sdk_colour_settings=(rendered/'sdk_colour_settings.txt').read_text().strip())
                # Diagnostics only: a better flip/transpose never bypasses the
                # fixed-pose registration gate or silently transforms GT.
                orientations={}
                for k in range(4):
                    for flip in (False,True):
                        transformed=np.rot90(candidate,k)
                        if flip:transformed=np.fliplr(transformed)
                        supported=binary_erosion(mask,iterations=3)&np.isfinite(truth)&(truth>.1)&(transformed>.1)
                        orientations[f'rot90_{k}_flip_{flip}']=float(np.quantile(np.abs(transformed[supported]-truth[supported])/truth[supported],.95)) if supported.any() else None
                stats['image_orientation_diagnostics_not_adopted']=orientations
                reports.append(stats)
                rgb.save(a.output/(key+'_ptex.png'));original.save(a.output/(key+'_vertex.png'))
                panels.append((key,original,rgb))
    canvas=Image.new('RGB',(768,len(panels)*208),'white');draw=ImageDraw.Draw(canvas)
    for i,(key,left,right) in enumerate(panels):
        for j,image in enumerate((left,right)):
            image.thumbnail((380,180));canvas.paste(image,(384*j,208*i))
        draw.text((2,208*i+185),key+' | vertex colours / Ptex',fill='black')
    canvas.save(a.output/'ptex_fixed_pose_gallery.jpg',quality=92)
    passed=all(r['registration_passed'] for r in reports)
    write_json(a.output/'acceptance.json',dict(status='REGISTERED_REQUIRES_VISUAL_REVIEW' if passed else 'REVIEW_REQUIRED',
        registration_passed=passed,frames=reports,production_adopted=False,
        fixed_colour_scales=dict(exposure=a.exposure_scale,saturation=a.saturation_scale,gamma='pinned SDK default'),
        limitations=['Mirror reflections omitted in probe; no mirror-containing candidate adopted',
            'RGB contrast is not proof of better training performance','No full Ptex dataset generated until registration and colour review pass']))


if __name__=='__main__':main()
