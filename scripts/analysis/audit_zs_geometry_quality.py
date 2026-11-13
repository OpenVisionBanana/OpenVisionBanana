#!/usr/bin/env python3
"""Training-only ZS geometry audit and additive, provenance-locked index views.

Current TARs and the pinned public ZIP payloads are audited separately. Stored
ZIP/TAR data are read by offset: no giant archive extraction or source edits.
Quality proxies are not a certificate that a label is correct or wrong.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import sqlite3
import struct
import sys
import time
import zipfile

import numpy as np
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.indexed_tar import IndexedTarReference, read_reference_bytes, open_reference_image
from common.surface_normal_encoding import decode_surface_normals_rgb
from common.training.asset_lock import digest
from common.training.geometry_curation import CONTRACT, edge_trim, quality_signals
from dataset_prep.depth.encoding import encode_depth_to_vision_banana_rgb, decode_vision_banana_rgb_to_depth
from scripts.dataset_prep.build_geometry_readiness_manifests import normalize, task_group, write_json
from scripts.dataset_prep.curate_nzs_geometry import readonly, write_view
from scripts.dataset_prep.download_and_prepare_ovb_geometry import REPO_ID, REVISION, LOGICAL_SHA256

DEFAULT_SOURCE = Path('data/geometry_training_readiness_20261002/plane/zeroshot_full.json')
DEFAULT_RELEASE = Path('data/ovb_geometry_training_subsets_v1')
LIMITS = dict(max_edge_fraction=.25, min_retained_area=.6, max_strip_valid_fraction=.5, min_side=32)
POLICIES = {
    'base': dict(max_invalid=.25, min_area=.6, reject_quality=False),
    'high': dict(max_invalid=.10, min_area=.75, reject_quality=True),
}
VERSION = 'zs-geometry-quality-v1'


def domain(e):
    for name in ('hypersim', 'matterport3d', 'replica', 'synscapes', 'taskonomy'):
        if name in e['name']: return name
    raise ValueError(f'Unexpected ZS geometry source: {e["name"]}')


def code_identity():
    names = ['scripts/analysis/audit_zs_geometry_quality.py', 'common/training/geometry_curation.py',
             'scripts/dataset_prep/curate_nzs_geometry.py', 'dataset_prep/depth/encoding.py',
             'common/surface_normal_encoding.py', 'train_dreambooth_partial_flux2_klein_img2img.py']
    return {n: digest(ROOT / n) for n in names}


def zip_offset(path, info):
    if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1:
        raise ValueError('Random-access audit requires an uncompressed, unencrypted ZIP member')
    with path.open('rb') as f:
        f.seek(info.header_offset); header = f.read(30)
    if header[:4] != b'PK\x03\x04': raise ValueError('Invalid ZIP local header')
    name_size, extra_size = struct.unpack_from('<HH', header, 26)
    return info.header_offset + 30 + name_size + extra_size


def initialize(campaign, source, release, dataset=None):
    if (campaign / 'plan.json').exists(): raise FileExistsError('Existing campaign: resume scans, do not reinitialize')
    expected = Path(str(source) + '.sha256').read_text().split()[0]
    if digest(source) != expected: raise ValueError('Current source manifest checksum mismatch')
    payload = json.loads(source.read_text())
    entries = [e for e in payload['datasets'] if task_group(e) != 'other']
    if len(entries) != 10: raise ValueError('Expected the five ZS depth and five ZS normal sources')
    if dataset is not None:
        entries = [e for e in entries if domain(e) == dataset]
        if len(entries) != 2: raise ValueError('Expected a matched depth/normal pair for the selected dataset')
    campaign.mkdir(parents=True, exist_ok=True)
    for folder in ('scans', 'indices', 'manifests', 'reports', 'published_indices', 'logs', 'verification', 'galleries'):
        (campaign / folder).mkdir(exist_ok=True)
    sources, zip_assets = [], []
    asset_lock = json.loads(Path(str(source) + '.lock.json').read_text())
    if not asset_lock.get('content_verified') or asset_lock['manifest_sha256'] != expected:
        raise ValueError('Current training asset lock is not content-verified for this manifest')
    locked = {str(Path(a['path']).resolve()): a['sha256'] for a in asset_lock['assets']}
    for e in entries:
        archive_sha = locked.get(str(Path(e['path']).resolve()))
        if archive_sha is None: raise ValueError('Source archive missing from verified asset lock')
        sources.append({**e, 'lane': 'current', 'dataset': domain(e), 'offset_base': 0,
                        'source_index_sha256': digest(Path(e['index_path'])),
                        'archive_sha256': archive_sha})
    for dataset in sorted({domain(e) for e in entries}):
        path = release / f'{dataset}_geometry_train_v1.zip'
        with zipfile.ZipFile(path) as z:
            package = json.loads(z.read('PACKAGE_MANIFEST.json'))
            for asset in package['assets']:
                if not asset['name'].endswith('.tar'): continue
                info = z.getinfo(asset['name']); offset = zip_offset(path, info)
                normal = 'surface_normals' in asset['name']
                index_name = asset['name'] + '.ovb-index.sqlite'
                index_asset = next(a for a in package['assets'] if a['name'] == index_name)
                data = z.read(index_name)
                if hashlib.sha256(data).hexdigest() != index_asset['sha256']:
                    raise ValueError('Published index differs from embedded package manifest')
                index = campaign / 'published_indices' / index_name
                if index.exists(): raise FileExistsError(index)
                index.write_bytes(data)
                sources.append(dict(name=f'published_{"normals" if normal else "depth"}_{dataset}',
                    lane='published', dataset=dataset, task='normals' if normal else 'depth',
                    path=str(path.resolve()), index_path=str(index.resolve()), offset_base=offset,
                    tar_member=asset['name'], tar_size=info.file_size, archive_sha256=asset['sha256'],
                    source_index_sha256=digest(index)))
            zip_assets.append(dict(path=str(path.resolve()), sha256=LOGICAL_SHA256[path.name], size=path.stat().st_size))
    # Preserve the current manifest and source weights.
    write_json(campaign / 'plan.json', dict(contract=VERSION, code_sha256=code_identity(),
        source_manifest=str(source.resolve()), source_manifest_sha256=expected,
        public_repo=REPO_ID, public_revision=REVISION, edge_limits=LIMITS, policies=POLICIES,
        sources=sources, public_zip_assets=zip_assets,
        originals_untouched=True, benchmark_exclusions_inherited=True,
        warnings=['RGB blur/low texture alone is not a rejection.',
                  'Coverage and texture screens are not semantic GT certification.',
                  'No training jobs, existing manifests, or Hugging Face files are changed.']))
    print(f'Prepared {len(sources)} audit sources; current and published versions kept separate', flush=True)


def checked_plan(campaign):
    plan = json.loads((campaign / 'plan.json').read_text())
    if plan['code_sha256'] != code_identity(): raise ValueError('Audit implementation changed; use a new campaign')
    if digest(Path(plan['source_manifest'])) != plan['source_manifest_sha256']:
        raise ValueError('Training source manifest changed')
    return plan


class Reader:
    def __init__(self, entry):
        self.entry = entry
        with readonly(Path(entry['index_path'])) as c:
            self.members = {n:(o,s) for n,o,s in c.execute('select name,offset,size from members')}
    def resolve(self, row, member):
        if member in self.members: return member
        candidate = str(row.get('_metadata_member', '')).rsplit('/', 1)[0] + '/' + str(member)
        if candidate in self.members: return candidate
        raise KeyError(f'Missing archive member: {member}')
    def ref(self, row, member):
        member = self.resolve(row, member); offset, size = self.members[member]
        return IndexedTarReference(self.entry['path'], member, self.entry['offset_base'] + offset, size)
    def image(self, row, member, mode='RGB'):
        with open_reference_image(self.ref(row, member)) as image:
            image.load(); return ImageOps.exif_transpose(image).convert(mode)
    def array(self, row, member):
        value = np.load(io.BytesIO(read_reference_bytes(self.ref(row, member))), allow_pickle=False)
        if isinstance(value, np.lib.npyio.NpzFile):
            try: return value['depth'].copy()
            finally: value.close()
        return value


def is_normal(entry):
    return entry.get('task') == 'normals' or entry['name'].startswith('surface_normals_')


def metric_depth(reader, row, valid):
    if row.get('depth_raw'):
        d = np.asarray(reader.array(row, row['depth_raw']), dtype=np.float32)
        if d.shape != valid.shape: raise ValueError('Raw depth and mask dimensions differ')
        return d, 'raw_metric'
    if row.get('depth_encoding', row.get('metadata', {}).get('depth_encoding')) == 'vision_banana_rgb':
        return decode_vision_banana_rgb_to_depth(np.asarray(reader.image(row, row['image']))), 'decoded_quantized_rgb'
    return None, 'unavailable'


def camera_intrinsics(row, width, height):
    meta = row.get('metadata', {})
    if 'hfov_degrees' in meta:
        f = width / (2 * math.tan(math.radians(meta['hfov_degrees']) / 2))
        return f, f, (width - 1) / 2, (height - 1) / 2
    # Arbitrary guessed intrinsics are never used to claim a convention error.
    return None


def paired_geometry(normal_rgb, normal_row, depth, depth_row, valid, encoding):
    intrinsics = camera_intrinsics(depth_row, depth.shape[1], depth.shape[0])
    if intrinsics is None: return {'status': 'not_checked_no_verified_intrinsics'}
    if normal_rgb.shape[:2] != depth.shape: return {'status': 'not_checked_unregistered_dimensions'}
    fx, fy, cx, cy = intrinsics
    h, w = depth.shape
    yy, xx = np.mgrid[:h, :w]
    points = np.stack(((xx-cx)*depth/fx, (yy-cy)*depth/fy, depth), axis=-1)
    derived = np.cross(points[1:-1, 2:] - points[1:-1, :-2], points[2:, 1:-1] - points[:-2, 1:-1])
    lengths = np.linalg.norm(derived, axis=-1)
    center = points[1:-1, 1:-1]
    derived[(derived*center).sum(-1) > 0] *= -1
    derived = derived / np.maximum(lengths[..., None], 1e-8)
    derived *= np.array([1, -1, -1], dtype=np.float32)
    units, nvalid = decode_surface_normals_rgb(normal_rgb, encoding=encoding)
    m = valid[1:-1, 1:-1] & nvalid[1:-1, 1:-1] & (lengths > 1e-8)
    d = depth[1:-1, 1:-1]
    for neighbor, ok in ((depth[1:-1,:-2],valid[1:-1,:-2]), (depth[1:-1,2:],valid[1:-1,2:]),
                         (depth[:-2,1:-1],valid[:-2,1:-1]), (depth[2:,1:-1],valid[2:,1:-1])):
        m &= ok & np.isfinite(neighbor) & (np.abs(neighbor-d) < .03*np.maximum(d, .1))
    # Bound work for 48-transform diagnostics, without using benchmark/model predictions.
    candidates = np.flatnonzero(m)
    if candidates.size < 100: return {'status': 'not_checked_insufficient_smooth_pixels'}
    chosen = candidates[::max(1, len(candidates)//4096)]
    a = units[1:-1,1:-1].reshape(-1,3)[chosen]
    b = derived.reshape(-1,3)[chosen]
    angles = np.degrees(np.arccos(np.clip((a*b).sum(-1), -1, 1)))
    scores = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1,1), repeat=3):
            value = a[:,perm] * np.array(signs)
            score = np.degrees(np.arccos(np.clip((value*b).sum(-1), -1, 1))).mean()
            scores.append((float(score), list(perm), list(signs)))
    best = min(scores, key=lambda s:s[0])
    return dict(status='checked', smooth_pixels=len(candidates), sampled_pixels=len(chosen),
                mean_angle=float(angles.mean()), median_angle=float(np.median(angles)),
                best_axis_mean=best[0], best_permutation=best[1], best_signs=best[2],
                axis_improvement=float(angles.mean()-best[0]),
                caution='Local derivatives versus stored normals; not a direct GT correctness certificate')


def inspect(item, entry, reader, pair=None):
    ordinal, payload = item; row = json.loads(payload)
    result = dict(row_number=ordinal, id=row.get('id'), hard_reasons=[], review_flags=[],
                  scene=row.get('scene', row.get('metadata', {}).get('scene', row.get('metadata', {}).get('building', 'unknown'))))
    try:
        rgb = reader.image(row, row['conditioning_image'])
        gt = reader.image(row, row['image'])
        mask_ref = row.get('loss_mask') or row.get('valid_mask') or row.get('mask')
        if not mask_ref: raise ValueError('No declared validity mask; grey RGB is not a substitute')
        mask = reader.image(row, mask_ref, 'L')
        if not rgb.size == gt.size == mask.size: raise ValueError('RGB/GT/mask dimensions differ')
        values = np.asarray(mask); valid = values > 0
        if not set(np.unique(values)).issubset({0,255}): raise ValueError('Non-binary validity mask')
        box, warnings = edge_trim(valid, **LIMITS)
        l,t,r,b = box; mv = valid[t:b,l:r]
        result.update(original_size=list(rgb.size), crop_box=list(box), edge_warnings=warnings,
                      invalid_before=float(1-valid.mean()), invalid_after=float(1-mv.mean()),
                      retained_area=float(mv.size/valid.size),
                      invalid_edge_fraction=float(1-np.concatenate((valid[0],valid[-1],valid[:,0],valid[:,-1])).mean()))
        size = tuple(max(3, round(x*min(1,320/max(r-l,b-t)))) for x in (r-l,b-t))
        small_rgb = rgb.crop(box).resize(size, Image.Resampling.BILINEAR)
        small_gt = gt.crop(box).resize(size, Image.Resampling.NEAREST)
        small_valid = np.asarray(mask.crop(box).resize(size, Image.Resampling.NEAREST))>0
        signals, flags, hard = quality_signals(small_rgb, small_gt, small_valid, normals=is_normal(entry))
        result.update(signals=signals, review_flags=flags, hard_reasons=hard)
        result['rgb_hash_32'] = hashlib.sha256(rgb.resize((32,32), Image.Resampling.BILINEAR).tobytes()).hexdigest()
        gray = np.asarray(small_rgb, dtype=np.float32).mean(-1)
        result['signals']['flat_patch_fraction'] = float(np.mean([
            gray[y:y+16,x:x+16].std()<2 for y in range(0,gray.shape[0]-15,16)
            for x in range(0,gray.shape[1]-15,16)]))
        if is_normal(entry):
            # Resolve legacy storage exactly as the production loader does.
            from train_dreambooth_partial_flux2_klein_img2img import prepared_row_normal_encoding
            encoding = prepared_row_normal_encoding(row)
            if not encoding:
                result['review_flags'].append('normal_encoding_missing')
            else:
                units, unit_valid = decode_surface_normals_rgb(np.asarray(gt), encoding=encoding)
                result['signals']['invalid_vector_inside_mask_fraction'] = float((~unit_valid[valid]).mean()) if valid.any() else 1.
                if valid.any(): result['signals']['negative_z_fraction'] = float((units[...,2][valid] < 0).mean())
                if pair is not None and ordinal in pair['chosen'] and row.get('id') in pair['rows']:
                    depthrow = pair['rows'][row['id']]; depthreader = pair['reader']
                    depthmask = np.asarray(depthreader.image(depthrow, depthrow.get('loss_mask') or depthrow.get('valid_mask') or depthrow.get('mask'), 'L'))>0
                    depth, method = metric_depth(depthreader, depthrow, depthmask)
                    same_rgb = depthreader.image(depthrow, depthrow['conditioning_image']).tobytes() == rgb.tobytes()
                    result['paired_rgb_identical'] = same_rgb
                    if depth is not None and same_rgb:
                        result['paired_geometry'] = paired_geometry(np.asarray(gt), row, depth, depthrow,
                            valid & depthmask & np.isfinite(depth) & (depth>0), encoding)
                    else:
                        result['paired_geometry'] = {'status':'not_checked_RGB_or_raw_depth_unavailable'}
        else:
            depth, method = metric_depth(reader, row, valid); result['depth_audit_method'] = method
            if depth is not None:
                finite = np.isfinite(depth) & (depth > 0)
                bad = float((~finite[valid]).mean()) if valid.any() else 1.
                result['signals']['bad_depth_inside_mask_fraction'] = bad
                if bad > 0: result['hard_reasons'].append('invalid_metric_depth_inside_mask')
                d = depth[valid & finite]
                if d.size:
                    qs = np.quantile(d,[.01,.1,.5,.9,.99]).tolist()
                    result['signals'].update(depth_quantiles=qs, depth_below_0_5m_fraction=float((d<.5).mean()),
                        depth_below_1m_fraction=float((d<1).mean()), relative_depth_spread=float((qs[-1]-qs[0])/max(qs[2],.001)))
                    if (d<.5).mean() > .75: result['review_flags'].append('mostly_very_close_view')
                    if qs[2] < 1.0 and result['signals']['flat_patch_fraction'] > .85:
                        result['review_flags'].append('mostly_close_flat_view')
                if method == 'raw_metric' and row.get('depth_encoding') == 'vision_banana_rgb':
                    expected = encode_depth_to_vision_banana_rgb(depth, valid)
                    error = np.max(np.abs(expected.astype(float)-np.asarray(gt).astype(float)),axis=-1)
                    result['signals']['raw_vs_stored_rgb_mismatch_fraction'] = float((error[valid]>2).mean()) if valid.any() else 1.
                    if result['signals']['raw_vs_stored_rgb_mismatch_fraction'] > .05:
                        result['review_flags'].append('raw_depth_vs_stored_encoding_disagreement')
        if not valid.any() and 'empty_validity' not in result['hard_reasons']:
            result['hard_reasons'].append('empty_validity')
    except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
        result['hard_reasons'] = ['unreadable_or_misaligned_triplet']
        result['error'] = f'{type(error).__name__}: {error}'
    return ordinal, json.dumps(result,sort_keys=True,allow_nan=False), payload


def rejection(result, variant):
    policy = POLICIES[variant]
    reasons = list(result['hard_reasons'])
    if result.get('invalid_after',1) > policy['max_invalid']+1e-12: reasons.append('remaining_invalid_coverage')
    if result.get('retained_area',0) < policy['min_area']: reasons.append('retained_area')
    if result.get('edge_warnings'): reasons.append('unsafe_edge_crop')
    flags = result['review_flags']
    # Only explicit degenerate/ambiguous signals are auto-filtered. No rejection
    # based on generic blur, textureless walls or GT surface roughness alone.
    if policy['reject_quality']:
        reasons += [f'quality:{f}' for f in ('near_constant_rgb','near_uniform_black_or_white_rgb',
                    'mostly_very_close_view','mostly_close_flat_view') if f in flags]
    return reasons


def scan(campaign, number, workers, limit):
    plan = checked_plan(campaign); entry=plan['sources'][number]
    if digest(Path(entry['index_path'])) != entry['source_index_sha256']: raise ValueError('Source index changed')
    reader = Reader(entry)
    with readonly(Path(entry['index_path'])) as source:
        total = source.execute('select count(*) from rows').fetchone()[0]
        # Pilots are stratified across the entire index, not the first scene.
        ordinals = [n for n, in source.execute('select row_number from rows order by row_number')]
        chosen = {ordinals[i] for i in np.linspace(0,total-1,min(limit,total)).round().astype(int)} if limit else None
        pair = None
        if is_normal(entry):
            sibling = next(e for e in plan['sources'] if e['dataset']==entry['dataset'] and e['lane']==entry['lane'] and not is_normal(e))
            with readonly(Path(sibling['index_path'])) as db:
                depthrows = {r['id']:r for _,p in db.execute('select row_number,payload from rows') if (r:=json.loads(p)).get('id')}
            pairchosen = chosen if limit else {ordinals[i] for i in np.linspace(0,len(ordinals)-1,min(512,len(ordinals))).round().astype(int)}
            pair = dict(reader=Reader(sibling), rows=depthrows, chosen=pairchosen)
        identity = json.dumps(dict(entry=entry,limit=limit,code=plan['code_sha256']),sort_keys=True)
        output = campaign/'scans'/f'{entry["name"]}{".pilot" if limit else ""}.sqlite'
        with sqlite3.connect(output) as out:
            out.execute('create table if not exists metadata(key text primary key,value text)')
            out.execute('create table if not exists records(row_number integer primary key,result text,payload text)')
            existing = dict(out.execute('select key,value from metadata'))
            if existing and existing.get('identity')!=identity: raise ValueError('Scan belongs to another implementation or source')
            out.execute('insert or replace into metadata values (?,?)',('identity',identity));out.commit()
            done = {n for n, in out.execute('select row_number from records')}
            cursor=source.execute('select row_number,payload from rows order by row_number')
            start=time.monotonic()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                while batch:=cursor.fetchmany(128):
                    batch=[item for item in batch if item[0] not in done and (chosen is None or item[0] in chosen)]
                    out.executemany('insert into records values (?,?,?)',executor.map(lambda item:inspect(item,entry,reader,pair),batch))
                    out.commit()
                    count=out.execute('select count(*) from records').fetchone()[0]
                    if batch and (count%2048<128 or count == (len(chosen) if chosen is not None else total)):
                        print(f'{entry["name"]}: {count}/{len(chosen) if chosen is not None else total}; {time.monotonic()-start:.1f}s',flush=True)
            count=out.execute('select count(*) from records').fetchone()[0]
            if count!=(len(chosen) if chosen is not None else total):raise ValueError('Incomplete scan')
            out.execute('insert or replace into metadata values (?,?)',('complete','true'))
            out.execute('insert or replace into metadata values (?,?)',('full_source',json.dumps(not limit)));out.commit()
    print(f'PASS full_source={not limit} {entry["name"]}: {count}',flush=True)


def verify(campaign, number):
    plan=checked_plan(campaign); asset=plan['public_zip_assets'][number]
    if digest(Path(asset['path']))!=asset['sha256']:raise ValueError('ZIP payload differs from pinned public release')
    entries=[e for e in plan['sources'] if e['lane']=='current' and e['dataset']==Path(asset['path']).name.split('_geometry')[0]]
    for e in entries:
        if digest(Path(e['path']))!=e['archive_sha256']:raise ValueError(f'Current archive content checksum failed: {e["name"]}')
    write_json(campaign/'verification'/f'{number}.json',dict(status='PASS',asset=asset,current_sources=[e['name'] for e in entries]))


def quantiles(values):
    return np.quantile(values,[0,.25,.5,.75,.9,.99,1]).tolist() if values else []


def summary(records):
    results=[r for _,r,_ in records]
    return dict(rows=len(results),hard_reasons=dict(Counter(f for r in results for f in r['hard_reasons'])),
        review_flags=dict(Counter(f for r in results for f in r['review_flags'])),
        scenes=dict(Counter(r['scene'] for r in results)),
        cropped_rows=sum(r.get('retained_area',1)<1 for r in results),
        invalid_before_quantiles=quantiles([r['invalid_before'] for r in results if 'invalid_before' in r]),
        invalid_after_quantiles=quantiles([r['invalid_after'] for r in results if 'invalid_after' in r]),
        exact_thumbnail_duplicate_rows=sum('rgb_hash_32' in r for r in results)-len({r['rgb_hash_32'] for r in results if 'rgb_hash_32' in r}),
        paired_geometry=[dict(id=r['id'],**r['paired_geometry']) for r in results if 'paired_geometry' in r],
        provisional_kept={v:sum(not rejection(r,v) for r in results) for v in POLICIES})


def gallery(campaign, entry, records):
    good=[r for r in records if 'crop_box' in r[1]]
    if not good:return
    # Deterministic random control plus largest invalid fractions and quality flags.
    invalid=sorted(good,key=lambda r:r[1]['invalid_before'],reverse=True)[:4]
    quality=sorted(good,key=lambda r:(len(r[1]['review_flags']),r[1]['signals'].get('depth_below_0_5m_fraction',0)),reverse=True)[:4]
    control=[good[i] for i in np.linspace(0,len(good)-1,min(4,len(good))).round().astype(int)]
    chosen=[];seen=set()
    for item in invalid+quality+control:
        if item[0] not in seen:chosen.append(item);seen.add(item[0])
    reader=Reader(entry); canvas=Image.new('RGB',(768,len(chosen)*210),'white');draw=ImageDraw.Draw(canvas)
    for i,(_,r,row) in enumerate(chosen):
        rgb=reader.image(row,row['conditioning_image']);gt=reader.image(row,row['image'])
        mask=reader.image(row,row.get('loss_mask')or row.get('valid_mask')or row.get('mask'),'L')
        for j,image in enumerate((rgb,gt,mask.convert('RGB'))):
            image.thumbnail((254,175));canvas.paste(image,(j*256,i*210))
        draw.text((2,i*210+177),f'{r["id"]} invalid={r["invalid_before"]:.3f}->{r["invalid_after"]:.3f}',fill='black')
        draw.text((2,i*210+192),', '.join(r['review_flags'])[:120],fill='black')
    canvas.save(campaign/'galleries'/f'{entry["name"]}.jpg',quality=90)


def finalize(campaign, pilot, report_only=False):
    plan=checked_plan(campaign); reports={};all_records={}
    for entry in plan['sources']:
        p=campaign/'scans'/f'{entry["name"]}{".pilot" if pilot else ""}.sqlite'
        if not p.exists():
            if pilot:continue
            raise ValueError(f'Missing scan: {p}')
        with readonly(p) as db:
            meta=dict(db.execute('select key,value from metadata'))
            if meta.get('complete')!='true' or (not pilot and meta.get('full_source')!='true'):
                raise ValueError('Incomplete or sampled scan cannot produce full-source conclusions')
            records=[(n,json.loads(r),json.loads(payload)) for n,r,payload in db.execute('select * from records order by row_number')]
        reports[entry['name']]=summary(records);all_records[entry['name']]=records
        gallery(campaign,entry,records)
    if not pilot:
        for i in range(len(plan['public_zip_assets'])):
            if json.loads((campaign/'verification'/f'{i}.json').read_text())['status']!='PASS':raise ValueError('Content verification missing')
    write_json(campaign/'reports'/('pilot_summary.json' if pilot else 'quality_summary.json'),dict(
        status='SAMPLED_DIAGNOSTIC' if pilot else 'FULL_SCAN', sources=reports,
        quantile_order=[0,.25,.5,.75,.9,.99,1],policies=POLICIES,
        warning='Quality flags and paired gradients are review evidence, not proof of wrong labels or model improvement.'))
    if pilot or report_only:return
    template=json.loads(Path(plan['source_manifest']).read_text())
    names={e['name'] for e in plan['sources'] if e['lane']=='current'}
    masses=Counter()
    for e in template['datasets']:masses[task_group(e)]+=e['weight']
    decisions={}
    for variant in POLICIES:
        selected=[copy.deepcopy(e) for e in template['datasets'] if e['name'] not in names]
        counts={}
        for entry in plan['sources']:
            if entry['lane']!='current':continue
            source_records=all_records[entry['name']];kept=[];reasons=Counter();scenes=Counter()
            # Widespread decode failures indicate a pipeline issue; fail instead of filtering.
            if reports[entry['name']]['hard_reasons'].get('unreadable_or_misaligned_triplet',0)>max(10,len(source_records)*.05):
                raise ValueError(f'Widespread structural failures: {entry["name"]}; inspect before filtering')
            for ordinal,result,row in source_records:
                excluded=rejection(result,variant)
                if excluded:reasons.update(excluded);continue
                row=copy.deepcopy(row);row['training_crop_box']=result['crop_box'];row['training_crop_contract']=CONTRACT
                row.setdefault('metadata',{})['training_curation']=dict(variant='zs_'+variant,contract=VERSION,
                    invalid_before=result['invalid_before'],invalid_after=result['invalid_after'])
                kept.append((ordinal,json.dumps(row,sort_keys=True)));scenes[result['scene']]+=1
            counts[entry['name']]=dict(original=len(source_records),kept=len(kept),reasons_nonexclusive=dict(reasons),retained_scenes=dict(scenes))
            if kept:
                clean={k:v for k,v in entry.items() if k in ('name','type','path','index_path','weight','allow_missing_loss_mask')}
                selected.append(write_view(clean,campaign/'indices'/f'{entry["name"]}.{variant}.sqlite',kept,'zs_'+variant))
        if any(not any(task_group(e)==g for e in selected) for g in masses):raise ValueError('Filtering removed an entire task')
        decisions[variant]=counts
        for scope in ('full','geometry'):
            rows=[e for e in selected if scope=='full' or task_group(e)!='other']
            write_json(campaign/'manifests'/f'zeroshot_{variant}_{scope}.json',dict(
                **template,datasets=normalize(rows,masses),curation_contract=dict(contract=VERSION,variant=variant,
                scope='ZS geometry only; non-geometry sources unchanged',policy=POLICIES[variant],
                originals_untouched=True,interior_holes='unchanged',benchmarks_unchanged=True,
                provenance_plan_sha256=digest(campaign/'plan.json'),model_quality_certified=False)))
    write_json(campaign/'reports'/'selection_summary.json',dict(variants=decisions,task_masses=dict(masses),
        counts_are_task_entries_not_unique_images=True,scene_distribution_may_change=True))
    write_json(campaign/'PREPARED.json',dict(status='indices_and_manifests_prepared',variants=list(POLICIES),
        real_training_loader_acceptance='must pass before READY',public_upload='not performed'))
    print('Prepared additive ZS base/high views; real-loader acceptance still required',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--source',type=Path,default=DEFAULT_SOURCE);p.add_argument('--release',type=Path,default=DEFAULT_RELEASE)
    p.add_argument('--dataset',choices=['replica','hypersim','matterport3d','synscapes','taskonomy'],
                   help='Limit a new audit to one dataset, preserving current/published depth and normal lanes.')
    action=p.add_mutually_exclusive_group(required=True)
    action.add_argument('--initialize',action='store_true');action.add_argument('--scan',type=int)
    action.add_argument('--verify',type=int);action.add_argument('--finalize',action='store_true')
    p.add_argument('--workers',type=int,default=8);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--report-only',action='store_true',help='Report full-scan evidence without creating filtered training views.')
    a=p.parse_args()
    if a.initialize:initialize(a.campaign,a.source,a.release,a.dataset)
    elif a.scan is not None:scan(a.campaign,a.scan,a.workers,a.limit)
    elif a.verify is not None:verify(a.campaign,a.verify)
    else:finalize(a.campaign,a.pilot,a.report_only)


if __name__=='__main__':main()
