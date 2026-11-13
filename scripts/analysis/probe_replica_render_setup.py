#!/usr/bin/env python3
"""Paired Replica rendering probe: original versus official dataset configuration.

Uses identical stored camera poses and saves new diagnostics only. This is not
a production dataset rebuild and never edits the existing source archives.
"""
import argparse
import io
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import habitat_sim
from dataset_prep.replica.prepare_replica_depth_normals import make_rotation, rgb_observation_to_image
from common.indexed_tar import IndexedTarReference, read_reference_bytes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replica-root',type=Path,required=True)
    p.add_argument('--index',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True,help='Stored RGB/depth archive for legacy-replay verification')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--test-z-up-camera',action='store_true',
                   help='Also diagnose an upright camera in the raw mesh Z-up frame; not a production replacement')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    config=a.replica_root/'replica.scene_dataset_config.json'
    if not config.is_file():raise FileNotFoundError(config)
    with sqlite3.connect(f'file:{a.index}?mode=ro',uri=True) as db:
        rows=[json.loads(r) for r, in db.execute('select payload from rows order by row_number')]
        members={n:(o,s) for n,o,s in db.execute('select name,offset,size from members')}
    def original_image(row):
        name=row['conditioning_image']
        if name not in members:
            name=str(row.get('_metadata_member','')).rsplit('/',1)[0]+'/'+name
        offset,size=members[name]
        raw=read_reference_bytes(IndexedTarReference(str(a.archive),name,offset,size))
        with Image.open(io.BytesIO(raw)) as image:return image.convert('RGB')
    selected=[]
    for scene in ('apartment_0','office_3','room_2'):
        options=[r for r in rows if r['metadata']['scene']==scene]
        selected.extend(options[i] for i in np.linspace(0,len(options)-1,4).round().astype(int))
    stats=[];panels={}
    for row in selected:panels[(row['id'],'stored_original')]=original_image(row)
    modes=['legacy_default','official_dataset_config','official_registered_scene']
    if a.test_z_up_camera:modes.append('upright_z_camera_diagnostic')
    for scene in sorted({r['metadata']['scene'] for r in selected}):
        scene_root=a.replica_root/scene
        for mode in modes:
            backend=habitat_sim.SimulatorConfiguration()
            backend.scene_id=str(scene_root/'habitat/replica_stage.stage_config.json')
            backend.enable_physics=False;backend.gpu_device_id=0
            if mode!='legacy_default':backend.scene_dataset_config_file=str(config)
            if mode in ('official_registered_scene','upright_z_camera_diagnostic'):backend.scene_id=scene
            sensors=[]
            for uuid,kind in (('color_sensor',habitat_sim.SensorType.COLOR),('depth_sensor',habitat_sim.SensorType.DEPTH)):
                spec=habitat_sim.CameraSensorSpec();spec.uuid=uuid;spec.sensor_type=kind
                spec.sensor_subtype=habitat_sim.SensorSubType.PINHOLE;spec.resolution=[512,512]
                spec.hfov=90.;spec.position=[0.,0.,0.];sensors.append(spec)
            agent=habitat_sim.AgentConfiguration();agent.sensor_specifications=sensors
            sim=habitat_sim.Simulator(habitat_sim.Configuration(backend,[agent]))
            try:
                stage=sim.get_stage_initialization_template()
                stage_attributes=dict(orient_up=list(stage.orient_up), orient_front=list(stage.orient_front),
                                      shader_type=str(stage.shader_type), handle=stage.handle)
                for row in selected:
                    if row['metadata']['scene']!=scene:continue
                    meta=row['metadata'];state=habitat_sim.AgentState()
                    state.position=np.array(meta['camera_position'],dtype=np.float32)
                    state.rotation=make_rotation(np.radians(meta['yaw_degrees']),np.radians(meta['pitch_degrees']))
                    if mode=='upright_z_camera_diagnostic':
                        from habitat_sim.utils.common import quat_from_angle_axis
                        state.rotation=(quat_from_angle_axis(np.radians(meta['yaw_degrees']), np.array([0.,0.,1.]))
                                        * quat_from_angle_axis(np.pi/2+np.radians(meta['pitch_degrees']), np.array([1.,0.,0.])))
                    sim.initialize_agent(0).set_state(state)
                    obs=sim.get_sensor_observations();image=rgb_observation_to_image(obs['color_sensor'])
                    out=a.output/mode;out.mkdir(exist_ok=True);image.save(out/(row['id']+'.png'))
                    depth=np.asarray(obs['depth_sensor']);valid=np.isfinite(depth)&(depth>.1)&(depth<=20)
                    np.save(out/(row['id']+'.depth.npy'),depth.astype(np.float32))
                    panels[(row['id'],mode)]=image
                    original=np.asarray(panels[(row['id'],'stored_original')],dtype=np.float32)
                    rendered=np.asarray(image,dtype=np.float32)
                    stats.append(dict(id=row['id'],mode=mode,valid_fraction=float(valid.mean()),
                                      median_depth=float(np.median(depth[valid])) if valid.any() else None,
                                      rgb_rmse_vs_stored=float(np.sqrt(np.mean((original-rendered)**2))),
                                      rgb_exact_match=bool(np.array_equal(original,rendered)),
                                      stage_attributes=stage_attributes,
                                      camera_pose_unchanged=mode!='upright_z_camera_diagnostic'))
            finally:sim.close()
    canvas=Image.new('RGB',(384*(1+len(modes)),len(selected)*270),'white');draw=ImageDraw.Draw(canvas)
    for i,row in enumerate(selected):
        for j,mode in enumerate(['stored_original']+modes):
            image=panels[(row['id'],mode)].copy();image.thumbnail((382,245));canvas.paste(image,(j*384,i*270))
        draw.text((2,i*270+247),row['id']+' | original / legacy replay / config + path / config + registered scene / optional Z-up camera',fill='black')
    canvas.save(a.output/'paired_render_config.jpg',quality=92)
    (a.output/'render_probe.json').write_text(json.dumps(dict(status='PASS',habitat_version=habitat_sim.__version__,
        official_dataset_config=json.loads(config.read_text()),samples=stats,
        caution='Same numerical poses in all modes; any scene-alignment change needs freshly sampled pose validation'),indent=2)+'\n')


if __name__=='__main__':main()
