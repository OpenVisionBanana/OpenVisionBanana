#!/usr/bin/env python3
"""Read-only QA of materialized Replica training archives, with review galleries.

Compare every exported RGB/target/mask against the accepted upright source, using
that source's raw metric depth/calibration for geometry checks. Texture/roughness
statistics are review proxies, not automatic rejection rules.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.audit_zs_geometry_quality import Reader, paired_geometry
from common.indexed_tar import read_reference_bytes
from common.surface_normal_encoding import decode_surface_normals_rgb
from dataset_prep.depth.encoding import encode_depth_to_vision_banana_rgb

PUBLICATION = Path('data/geometry_training_publication_20261004/upload')
SOURCE = Path('data/replica_upright_quality_20261002/data/archives')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(archive, index):
    with sqlite3.connect('file:' + str(index) + '?mode=ro', uri=True) as c:
        assert c.execute('pragma integrity_check').fetchone()[0] == 'ok'
        sequence = [json.loads(p) for (p,) in c.execute('select payload from rows order by rowid')]
        assert len({r['id'] for r in sequence}) == len(sequence), 'Duplicate row ID'
        for name, offset, size in c.execute('select name,offset,size from members'):
            assert offset >= 0 and size >= 0 and offset + size <= archive.stat().st_size, name
    entry = dict(path=str(archive), index_path=str(index), offset_base=0)
    return {r['id']: r for r in sequence}, Reader(entry)


def gallery(ids, results, readers, rows, path, title):
    # No masking or smoothing is applied to RGB or normal labels in this gallery.
    # Depth colour is only a display choice; the adjacent column is the exact GT.
    from matplotlib import colormaps
    cell, left, top, row_height = 230, 280, 72, 262
    canvas = Image.new('RGB', (left + 5 * cell, top + len(ids) * row_height), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 8), title, fill='black', font=font)
    for k, label in enumerate(('Input RGB', 'Metric depth (display only)', 'Stored depth GT (VB RGB)', 'Stored normal GT (VB RGB)', 'Joint mask (white = valid)')):
        draw.text((left + k * cell, 42), label, fill='black', font=font)
    pd, pn, sd, sn = readers
    dr, nr, ds, ns = rows
    for i, sample in enumerate(ids):
        dep, nor, source = dr[sample], nr[sample], ds[sample]
        rgb = pd.image(dep, dep['conditioning_image'])
        depth = sd.array(source, source['depth_raw'])
        valid = np.asarray(pd.image(dep, dep['loss_mask'], 'L')) > 0
        values = depth[valid]
        lo, hi = float(values.min()), float(values.max())
        colour = (colormaps['Spectral'](np.clip((np.nan_to_num(depth, nan=lo)-lo)/max(hi-lo, 1e-8), 0, 1))[..., :3]*255).astype('uint8')
        colour[~valid] = (128,128,128)
        normal_valid = np.asarray(pn.image(nor, nor['loss_mask'], 'L')) > 0
        joint_mask = Image.fromarray((valid & normal_valid).astype('uint8')*255).convert('RGB')
        images = [rgb, Image.fromarray(colour), pd.image(dep, dep['image']), pn.image(nor, nor['image']), joint_mask]
        y = top + i * row_height
        stats = results[sample]
        scene = source['metadata']['scene']
        label = '\n'.join([scene, sample.replace('replica_'+scene+'_', ''),
                           f'Invalid depth: {100*stats["depth_invalid"]:.3f}%',
                           f'Invalid normal: {100*stats["normal_invalid"]:.3f}%',
                           f'Depth: {lo:.2f} - {hi:.2f} m',
                           f'Normal neighbour >45 deg: {100*stats["normal_jump_fraction"]:.2f}%'])
        draw.multiline_text((10,y+12), label, fill='black', font=font, spacing=8)
        for k, im in enumerate(images):
            sampling = Image.Resampling.LANCZOS if k == 0 else Image.Resampling.NEAREST
            canvas.paste(im.resize((cell-6,cell-6), sampling), (left+k*cell,y))
        if path.stem == 'examples':
            native = Image.new('RGB',(3*512,548),'white')
            labels = ('Input RGB (unaltered)', 'Stored depth target (unaltered)', 'Stored normal target (unaltered)')
            nd = ImageDraw.Draw(native)
            for k, (im,label) in enumerate(zip((images[0],images[2],images[3]), labels)):
                nd.text((k*512+8,8),label,fill='black',font=font)
                native.paste(im,(k*512,36))
            native.save(path.parent/(sample+'.png'))
    canvas.save(path, quality=95)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--publication', type=Path, default=PUBLICATION)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--limit', type=int, default=0, help='Pilot only; zero scans every row')
    p.add_argument('--skip-checksums', action='store_true', help='Pilot only')
    p.add_argument('--galleries-only', action='store_true', help='Regenerate diagnostic galleries from a completed scan, without modifying data or scan results')
    args = p.parse_args()
    if args.galleries_only:
        summary = json.loads((args.output/'summary.json').read_text())
        with (args.output/'frame_checks.csv').open() as f:
            result_map = {}
            for row in csv.DictReader(f):
                if row['errors']:
                    continue
                for key in ('depth_invalid','normal_invalid','normal_jump_fraction'):
                    row[key] = float(row[key])
                result_map[row['id']] = row
        exported = [(args.publication/'archives'/(n+'.tar'), args.publication/'indices'/(n+'.tar.ovb-index.sqlite'))
                    for n in ('depth_legacy_replica-00000','surface_normals_legacy_replica_camera_facing_v4-00000')]
        sources = [(args.source/n,args.source/(n+'.ovb-index.sqlite')) for n in ('replica_depth_upright.tar','replica_normals_upright.tar')]
        rows,readers = list(zip(*[load(a,b) for a,b in exported+sources]))
        gallery(summary['galleries']['representative_ids'],result_map,readers,rows,args.output/'examples.jpg','Replica: representative exported training examples (GT, not model predictions)')
        gallery(summary['galleries']['review_ids'],result_map,readers,rows,args.output/'review_cases.jpg','Replica: review cases (most invalid / most normal jumps / darkest / lowest contrast)')
        return
    if args.output.exists():
        raise FileExistsError('Use a fresh diagnostic directory; existing acceptance results are never overwritten')
    args.output.mkdir(parents=True)
    start = time.monotonic()
    names = ['depth_legacy_replica-00000', 'surface_normals_legacy_replica_camera_facing_v4-00000']
    exported = [(args.publication/'archives'/(n+'.tar'), args.publication/'indices'/(n+'.tar.ovb-index.sqlite')) for n in names]
    source_names = ['replica_depth_upright.tar', 'replica_normals_upright.tar']
    source_files = [(args.source/n, args.source/(n+'.ovb-index.sqlite')) for n in source_names]
    loaded = [load(a,b) for a,b in exported+source_files]
    rows, readers = list(zip(*loaded))
    dr, nr, ds, ns = rows
    pd, pn, sd, sn = readers
    assert set(dr) == set(nr) == set(ds) == set(ns), 'Depth/normal/source ID sets differ'
    checksums = []
    if not args.skip_checksums:
        for archive,index in exported:
            for f in (archive,index):
                actual = digest(f)
                expected = Path(str(f)+'.sha256').read_text().split()[0]
                assert actual == expected, f'SHA-256 mismatch: {f}'
                checksums.append(dict(path=str(f), sha256=actual, matched=True))
    groups = {}
    for sample in sorted(dr):
        groups.setdefault(ds[sample]['metadata']['scene'], []).append(sample)
    geometry_ids = {ids[int(i)] for ids in groups.values() for i in np.linspace(0,len(ids)-1,4)}
    ids = sorted(dr)
    if args.limit:
        ids = ids[:args.limit]

    def inspect(sample):
        result = dict(id=sample, scene=ds[sample]['metadata']['scene'], errors=[])
        try:
            dep,nor,original_depth,original_normal = (r[sample] for r in rows)
            assert dep['task'] == 'depth_estimation' and nor['task'] == 'surface_normal_estimation'
            assert dep['canonical_depth_quantity'] == 'projective_camera_z_meters'
            assert nor['normal_encoding'] == 'vision_banana_x_left_red'
            for row in (dep,nor):
                assert row['metadata']['materialized_crop_box'] is None
                assert row['metadata']['source_id'] == sample
            for row in (original_depth, original_normal):
                assert row['metadata']['camera_world_up_axis'] == 'z'
                assert row['metadata']['yaw_axis'] == 'world_z'
                assert row['metadata']['rgb_source'] == 'mesh_vertex_colours_not_Ptex'
            images = {}
            for key,new,old,reader,source_reader in (('depth',dep,original_depth,pd,sd), ('normal',nor,original_normal,pn,sn)):
                images[key] = {}
                for field in ('conditioning_image','image','loss_mask'):
                    mode = 'L' if field == 'loss_mask' else 'RGB'
                    im = reader.image(new,new[field],mode)
                    a = np.asarray(im)
                    b = np.asarray(source_reader.image(old,old[field],mode))
                    assert a.shape == b.shape and np.array_equal(a,b), f'Export changed {key}/{field}'
                    assert im.size == (512,512), f'Unexpected {key}/{field} dimensions'
                    images[key][field] = a
            rgb = images['depth']['conditioning_image']
            assert np.array_equal(rgb,images['normal']['conditioning_image']), 'Paired RGB differs'
            masks = {}
            for key,row in (('depth',dep),('normal',nor)):
                mask = images[key]['loss_mask']
                assert np.isin(mask,[0,255]).all() and mask.any(), 'Bad mask'
                masks[key] = mask > 0
                invalid = float(1-masks[key].mean())
                assert abs(invalid-row['metadata']['invalid_fraction']) < 1e-12, 'Mask metadata differs'
                result[key+'_invalid'] = invalid
            joint = masks['depth'] & masks['normal']
            result['joint_invalid'] = float(1-joint.mean())
            assert result['joint_invalid'] <= .0100001, 'Joint validity below selection threshold'
            depth = sd.array(original_depth,original_depth['depth_raw'])
            assert depth.shape == joint.shape
            assert np.isfinite(depth[masks['depth']]).all() and (depth[masks['depth']] > .1).all()
            assert (depth[masks['depth']] <= 20).all()
            assert np.array_equal(encode_depth_to_vision_banana_rgb(depth,masks['depth']),images['depth']['image']), 'Depth GT encoding differs from raw metric depth'
            units,valid_normals = decode_surface_normals_rgb(images['normal']['image'],encoding=nor['normal_encoding'])
            assert (valid_normals | ~masks['normal']).all(), 'Undefined normal inside valid mask'
            stored = images['normal']['image'].astype(np.float32)/127.5-1
            result['nonunit_normal_fraction'] = float((np.abs(np.linalg.norm(stored,axis=-1)[masks['normal']]-1)>.025).mean())
            assert result['nonunit_normal_fraction'] == 0, 'Non-unit normal labels'
            yy,xx = np.mgrid[:512,:512]
            x=(xx-255.5)/256; y=(yy-255.5)/256
            dot=(-units[...,0]*x + units[...,1]*y + units[...,2])/np.sqrt(x*x+y*y+1)
            result['away_normal_fraction'] = float((dot[masks['normal']] < -.015).mean())
            assert result['away_normal_fraction'] == 0, 'Camera-facing convention failed'
            stats = []
            for a,b,m in ((units[:,:-1],units[:,1:],masks['normal'][:,:-1]&masks['normal'][:,1:]),
                          (units[:-1],units[1:],masks['normal'][:-1]&masks['normal'][1:])):
                stats.append(np.sum((a*b).sum(-1)[m] < np.cos(np.pi/4)))
            count = int((masks['normal'][:,:-1]&masks['normal'][:,1:]).sum()+(masks['normal'][:-1]&masks['normal'][1:]).sum())
            result['normal_jump_fraction'] = float(sum(stats)/max(1,count))
            gray = rgb.astype(np.float32).mean(-1)
            result.update(rgb_mean=float(gray.mean()),rgb_std=float(gray.std()),
                          rgb_exact_hash=hashlib.sha256(rgb.tobytes()).hexdigest(),
                          depth_median=float(np.median(depth[masks['depth']])),
                          joint_border_invalid=float(1-np.concatenate((joint[0],joint[-1],joint[:,0],joint[:,-1])).mean()))
            if sample in geometry_ids:
                result['geometry'] = paired_geometry(images['normal']['image'],original_normal,depth,original_depth,joint,nor['normal_encoding'])
                assert result['geometry']['status'] == 'checked', 'Geometry check not possible'
                assert result['geometry']['mean_angle'] < 1, 'Smooth surface depth/normal mismatch'
        except Exception as e:
            result['errors'].append(f'{type(e).__name__}: {e}')
        return result

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,result in enumerate(pool.map(inspect,ids),1):
            results.append(result)
            if i%100 == 0 or i==len(ids):
                print(f'CHECKED {i}/{len(ids)}; errors={sum(bool(r["errors"]) for r in results)}; elapsed={time.monotonic()-start:.1f}s',flush=True)
    good = [r for r in results if not r['errors']]
    fields = sorted({k for r in results for k in r if k not in ('geometry','errors')})
    with (args.output/'frame_checks.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields+['errors']); writer.writeheader()
        for r in results:
            writer.writerow({**{k:r.get(k) for k in fields},'errors':'; '.join(r['errors'])})
    geom = [r['geometry'] for r in good if r.get('geometry')]
    summary = dict(status='PASS_FILE_PAIRING_AND_GEOMETRY' if len(good)==len(results) else 'FAIL',
                   full_scan=len(ids)==len(dr), rows_checked=len(ids), total_pairs=len(dr),
                   scenes=dict(Counter(r['scene'] for r in results)), errors=[r for r in results if r['errors']],
                   checksum_checks=checksums, source_archives=[str(a) for a,_ in source_files],
                   geometry_sampled_frames=len(geom), source_equivalence='Every decoded exported RGB, target and mask compared pixel-for-pixel against the accepted upright source',
                   exact_duplicate_rgb_groups=sum(v>1 for v in Counter(r['rgb_exact_hash'] for r in good).values()),
                   semantic_quality_certified=False,
                   limitations=['RGB still uses mesh vertex colours, not official Ptex textures.',
                                'Central-difference normals can mix surfaces across depth discontinuities.',
                                'Matching depth-derived normals is a consistency test, not independent proof of GT.',
                                'Colour/depth scales in diagnostic displays are not changes to training data.'])
    if good:
        for key in ('depth_invalid','normal_invalid','joint_invalid','normal_jump_fraction','rgb_mean','rgb_std'):
            values=np.array([r[key] for r in good])
            summary[key]=dict(mean=float(values.mean()),p95=float(np.percentile(values,95)),max=float(values.max()),min=float(values.min()))
        summary['zero_undefined_pairs']=sum(r['joint_invalid']==0 for r in good)
    if geom:
        summary['smooth_depth_normal_mean_degrees']=float(np.mean([g['mean_angle'] for g in geom]))
        summary['smooth_depth_normal_max_frame_mean_degrees']=max(g['mean_angle'] for g in geom)
        summary['axis_improvement_max_degrees']=max(g['axis_improvement'] for g in geom)
    result_map={r['id']:r for r in good}
    representative=[]
    for scene in ('apartment_0','apartment_2','office_1','office_3','hotel_0','room_0'):
        candidates=[s for s in groups.get(scene,[]) if s in result_map]
        if candidates: representative.append(candidates[len(candidates)//2])
    if representative:
        gallery(representative,result_map,readers,rows,args.output/'examples.jpg','Replica: representative exported training examples (GT, not model predictions)')
    review=[]
    for key,reverse in (('joint_invalid',True),('normal_jump_fraction',True),('rgb_mean',False),('rgb_std',False)):
        for r in sorted(good,key=lambda r:r[key],reverse=reverse):
            if r['id'] not in review:
                review.append(r['id']); break
    if review:
        gallery(review,result_map,readers,rows,args.output/'review_cases.jpg','Replica: review cases (most invalid / most normal jumps / darkest / lowest contrast)')
    summary['galleries']={'representative_ids':representative,'review_ids':review}
    summary['elapsed_seconds']=time.monotonic()-start
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output/'frames.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in results))
    print(json.dumps(summary,indent=2),flush=True)
    if summary['status']=='FAIL':
        raise SystemExit(1)


if __name__=='__main__':
    main()
