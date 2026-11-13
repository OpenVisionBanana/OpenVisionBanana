#!/usr/bin/env python3
"""Regenerate a separate upright Replica candidate at recorded positions.

RGB, metric depth, normals and masks are regenerated jointly. Reusing positions
does not certify camera clearance; every new view passes strict validity/close
view checks. Mesh vertex colours are retained: this is not a Ptex texture fix.
No existing archives, indices, training jobs or Hub files are changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import io
import json
from pathlib import Path
import sqlite3
import sys
import tarfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.surface_normal_encoding import encode_surface_normals_rgb
from dataset_prep.geometry_conventions import CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL
from dataset_prep.depth.encoding import encode_depth_to_vision_banana_rgb
from dataset_prep.replica.prepare_replica_depth_normals import (
    derive_normals_from_depth, rgb_observation_to_image,
)
from scripts.dataset_prep.curate_zs_geometry_pairs import REPLICA, select_diverse, pose_cell, difference_hash
from scripts.dataset_prep.build_geometry_readiness_manifests import write_json
from common.training.asset_lock import digest


def upright_rotation(yaw, pitch):
    from habitat_sim.utils.common import quat_from_angle_axis
    return (quat_from_angle_axis(yaw, np.array([0., 0., 1.]))
            * quat_from_angle_axis(np.pi/2+pitch, np.array([1., 0., 0.])))


def png_bytes(image):
    stream = io.BytesIO(); image.save(stream, format='PNG'); return stream.getvalue()


def add_member(archive, name, content):
    info = tarfile.TarInfo(name); info.size = len(content); info.mtime = 0; info.mode = 0o644
    archive.addfile(info, io.BytesIO(content))


def index_rows(path):
    with sqlite3.connect('file:'+str(path)+'?mode=ro', uri=True) as db:
        return {r['id']:r for payload, in db.execute('select payload from rows') if (r := json.loads(payload))}


def render(replica_root, depth_index, normal_index, output):
    import habitat_sim
    if output.exists(): raise FileExistsError('Use a new version; an existing candidate is never overwritten')
    original_depth, original_normal = index_rows(depth_index), index_rows(normal_index)
    if original_depth.keys() != original_normal.keys(): raise ValueError('Original Replica task IDs differ')
    config = replica_root/'replica.scene_dataset_config.json'
    if not config.exists(): raise FileNotFoundError(config)
    scenes = sorted({r['metadata']['scene'] for r in original_depth.values()})
    if len(scenes) != 18: raise ValueError('Expected all 18 original Replica training scenes')
    output.mkdir(parents=True); (output/'archives').mkdir(); (output/'reports').mkdir()
    row_lists = [[], []]; scene_reports = {}; accepted_checks = []
    targets = [output/'archives/replica_depth_upright.tar', output/'archives/replica_normals_upright.tar']
    partials = [p.with_name(p.name+'.partial') for p in targets]
    with tarfile.open(partials[0], 'w') as depth_tar, tarfile.open(partials[1], 'w') as normal_tar:
        for scene in scenes:
            backend = habitat_sim.SimulatorConfiguration()
            backend.scene_dataset_config_file = str(config)
            backend.scene_id = str(replica_root/scene/'habitat/replica_stage.stage_config.json')
            backend.enable_physics = False; backend.gpu_device_id = 0
            sensors = []
            for uuid, kind in [('color_sensor', habitat_sim.SensorType.COLOR), ('depth_sensor', habitat_sim.SensorType.DEPTH)]:
                sensor = habitat_sim.CameraSensorSpec(); sensor.uuid = uuid; sensor.sensor_type = kind
                sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
                sensor.resolution = [512, 512]; sensor.hfov = 90.; sensor.position = [0., 0., 0.]
                sensors.append(sensor)
            agent = habitat_sim.AgentConfiguration(); agent.sensor_specifications = sensors
            simulator = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
            candidates = []; arrays = {}; reasons = Counter()
            originals = sorted(r for r in original_depth if original_depth[r]['metadata']['scene'] == scene)
            try:
                for number, key in enumerate(originals, 1):
                    row = original_depth[key]; meta = row['metadata']
                    state = habitat_sim.AgentState(); state.position = np.asarray(meta['camera_position'], dtype=np.float32)
                    state.rotation = upright_rotation(np.radians(meta['yaw_degrees']), np.radians(meta['pitch_degrees']))
                    simulator.initialize_agent(0).set_state(state)
                    obs = simulator.get_sensor_observations(); rgb = rgb_observation_to_image(obs['color_sensor'])
                    depth = np.asarray(obs['depth_sensor'], dtype=np.float32).copy()
                    valid = np.isfinite(depth) & (depth > .1) & (depth <= 20)
                    if valid.mean() < .99: reasons['invalid_depth_coverage'] += 1; continue
                    normal, nvalid = derive_normals_from_depth(depth, valid, 90.)
                    normal = CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL.apply(normal, nvalid)
                    joint = valid & nvalid
                    if joint.mean() < .99: reasons['invalid_joint_coverage'] += 1; continue
                    close = float((depth[joint] < .5).mean()); median = float(np.median(depth[joint]))
                    if close > REPLICA['max_close_fraction'] or median < REPLICA['min_median_depth']:
                        reasons['close_view'] += 1; continue
                    gray = np.asarray(rgb, dtype=np.float32).mean(-1)
                    lap = gray[1:-1,:-2]+gray[1:-1,2:]+gray[:-2,1:-1]+gray[2:,1:-1]-4*gray[1:-1,1:-1]
                    contrast, sharpness = float(gray.std()), float(lap.var())
                    if contrast < 5 and sharpness < 1: reasons['featureless_view'] += 1; continue
                    new_id = key+'_upright'
                    candidates.append(dict(id=new_id, scene=scene, original_id=key, dhash=difference_hash(rgb),
                        rgb_mean=np.asarray(rgb).mean((0, 1)).tolist(), pose_cell=pose_cell(row),
                        quality_score=float(10*joint.mean()+min(contrast, 40)/40+min(np.log1p(sharpness), 5)/5-close),
                        invalid_after=float(1-joint.mean()), median_depth=median,
                        depth_below_0_5m_fraction=close))
                    arrays[new_id] = (rgb, depth, valid, normal, nvalid)
                    if number % 100 == 0: print(f'RENDER {scene}: {number}/{len(originals)}; candidates={len(candidates)}', flush=True)
            finally: simulator.close()
            kept, excluded = select_diverse(candidates)
            reasons.update(reason for rs in excluded.values() for reason in rs)
            if not kept: raise ValueError(f'No acceptable upright views in {scene}; do not substitute this candidate')
            scene_reports[scene] = dict(original=len(originals), quality_candidates=len(candidates), kept=len(kept), reasons=dict(reasons))
            for selected in kept:
                sample_id = selected['id']; old_id = selected['original_id']
                rgb, depth, valid, normal, nvalid = arrays[sample_id]
                dep = copy.deepcopy(original_depth[old_id]); nor = copy.deepcopy(original_normal[old_id])
                for task_row in (dep, nor):
                    for obsolete in ('_metadata_member', '_metadata_line', 'training_crop_box', 'training_crop_contract',
                                     'raw_normals', 'loss_mask', 'valid_mask', 'mask'):
                        task_row.pop(obsolete, None)
                    task_row['id'] = sample_id
                    task_row['conditioning_image'] = f'conditioning_images/{sample_id}.png'
                    task_row['loss_mask'] = f'loss_masks/{sample_id}.png'
                    task_row['metadata'].update(camera_world_up_axis='z', render_revision='replica-upright-recorded-positions-v1',
                        original_sample_id=old_id, camera_position_reused=True,
                        yaw_axis='world_z', rgb_source='mesh_vertex_colours_not_Ptex', invalid_joint_fraction=selected['invalid_after'])
                    task_row['metadata'].pop('training_curation', None)
                dep.update(image=f'target_images/{sample_id}.png', depth_raw=f'depth_raw/{sample_id}.npy',
                           preprocess_target='raw_depth_resize_then_encode')
                nor.update(image=f'target_images/{sample_id}.png', normal_encoding='vision_banana_x_left_red',
                           preprocess_target='resize_normal_vector_then_encode', required_normal_prompt_contract='campaign_short')
                nor['metadata'].update(normal_encoding='vision_banana_x_left_red', normal_orientation='face_camera_ray',
                    normal_source='derived_from_depth', coordinate_convention='camera_x_right_y_up_z_toward_camera')
                rgb_data = png_bytes(rgb)
                for archive, row, mask in [(depth_tar, dep, valid), (normal_tar, nor, nvalid)]:
                    add_member(archive, row['conditioning_image'], rgb_data)
                    add_member(archive, row['loss_mask'], png_bytes(Image.fromarray(mask.astype('uint8')*255)))
                encoded_depth = encode_depth_to_vision_banana_rgb(depth, valid)
                encoded_normal = encode_surface_normals_rgb(normal, nvalid, encoding='vision_banana_x_left_red')
                add_member(depth_tar, dep['image'], png_bytes(Image.fromarray(encoded_depth)))
                add_member(normal_tar, nor['image'], png_bytes(Image.fromarray(encoded_normal)))
                raw = io.BytesIO(); depth[~valid] = np.nan; np.save(raw, depth)
                add_member(depth_tar, dep['depth_raw'], raw.getvalue())
                row_lists[0].append(dep); row_lists[1].append(nor)
                accepted_checks.append({k:v for k,v in selected.items() if k not in ('dhash', 'pose_cell')})
            print(f'KEPT {scene}: {len(kept)} upright paired views', flush=True)
        for archive, rows in zip((depth_tar, normal_tar), row_lists):
            add_member(archive, 'metadata.jsonl', ('\n'.join(json.dumps(r, sort_keys=True) for r in rows)+'\n').encode())
    for temporary, final in zip(partials, targets): temporary.replace(final)
    write_json(output/'reports/render_summary.json', dict(status='RENDERED_CANDIDATE_REQUIRES_ACCEPTANCE',
        frames=len(row_lists[0]), scenes=scene_reports, renderer='Habitat-Sim with official config and explicit Z-up camera',
        code_sha256=digest(Path(__file__)), original_depth_index_sha256=digest(depth_index),
        original_normal_index_sha256=digest(normal_index), positions='Reused; rejected if the new view has insufficient valid geometry',
        limitations=['Mesh vertex colours unchanged; no Ptex texture restoration',
                    'This does not independently certify mesh-normal accuracy',
                    'No training manifests are automatically substituted'], originals_unchanged=True))
    write_json(output/'reports/selected_frames.json', dict(frames=accepted_checks))
    print(f'Prepared {len(row_lists[0])} new upright pairs; index and loader acceptance still required', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replica-root', type=Path, required=True)
    p.add_argument('--depth-index', type=Path, required=True)
    p.add_argument('--normal-index', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); render(a.replica_root, a.depth_index, a.normal_index, a.output)


if __name__ == '__main__': main()
