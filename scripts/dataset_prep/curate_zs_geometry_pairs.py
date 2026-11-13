#!/usr/bin/env python3
"""Build deterministic, paired ZS geometry views from a completed quality audit.

Original archives and indices are immutable. Coverage is measured from declared
GT masks, not grey RGB, and selection never uses model or benchmark predictions.
Replica receives additional close-view and within-scene diversity selection.
Other domains are retained unless an explicit coverage/structural check fails.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.training.asset_lock import digest
from common.training.geometry_curation import CONTRACT, edge_trim
from scripts.analysis.audit_zs_geometry_quality import Reader, checked_plan, is_normal
from scripts.dataset_prep.curate_nzs_geometry import readonly, write_view
from scripts.dataset_prep.build_geometry_readiness_manifests import normalize, task_group, write_json

VERSION = 'zs-paired-quality-v1'
POLICIES = {
    'masked': dict(max_invalid=.10, min_area=.75),
    'unmasked': dict(max_invalid=.01, min_area=.75),
}
REPLICA = dict(max_invalid=.01, max_close_fraction=.35, min_median_depth=.75,
               max_per_scene=250, spatial_cell_m=.75, yaw_bin_degrees=45,
               pitch_bin_degrees=30, max_per_pose_cell=2, near_duplicate_hamming=4)


def load_records(audit, entry):
    with readonly(audit/'scans'/f'{entry["name"]}.sqlite') as db:
        meta = dict(db.execute('select key,value from metadata'))
        if meta.get('complete') != 'true' or meta.get('full_source') != 'true':
            raise ValueError(f'Incomplete source audit: {entry["name"]}')
        records = [(n, json.loads(result), json.loads(payload))
                   for n, result, payload in db.execute('select * from records order by row_number')]
    with readonly(Path(entry['index_path'])) as db:
        expected = db.execute('select count(*) from rows').fetchone()[0]
    if len(records) != expected or digest(Path(entry['index_path'])) != entry['source_index_sha256']:
        raise ValueError('Source index identity or full scan coverage changed')
    ids = [r[2]['id'] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate frame IDs in source index')
    return {r[2]['id']: r for r in records}


def difference_hash(rgb):
    gray = np.asarray(rgb.convert('L').resize((9, 8), Image.Resampling.BILINEAR))
    return int.from_bytes(np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes(), 'big')


def pose_cell(row, policy=REPLICA):
    meta = row.get('metadata', {})
    position = meta.get('camera_position')
    yaw, pitch = meta.get('yaw_degrees'), meta.get('pitch_degrees')
    if position is None or yaw is None or pitch is None:
        raise ValueError('Replica diversity needs recorded camera position/yaw/pitch')
    if len(position) != 3 or not np.isfinite([*position, yaw, pitch]).all():
        raise ValueError('Invalid recorded Replica pose')
    return (*[math.floor(x/policy['spatial_cell_m']) for x in position],
            math.floor((yaw % 360)/policy['yaw_bin_degrees']),
            math.floor((pitch+90)/policy['pitch_bin_degrees']))


def select_diverse(candidates, policy=REPLICA):
    """Quality-first deterministic scene selection, with pose and appearance caps."""
    scenes = defaultdict(list)
    for candidate in candidates:
        scenes[candidate['scene']].append(candidate)
    selected, rejected = [], {}
    for scene in sorted(scenes):
        cells = Counter(); hashes = []
        for item in sorted(scenes[scene], key=lambda r: (-r['quality_score'], r['id'])):
            if len(hashes) >= policy['max_per_scene']:
                rejected[item['id']] = ['scene_cap']; continue
            if cells[item['pose_cell']] >= policy['max_per_pose_cell']:
                rejected[item['id']] = ['similar_pose']; continue
            # A low-detail wall can share dHash with a different view: require
            # similar coarse colour before treating it as an appearance duplicate.
            if any(bin(item['dhash'] ^ h).count('1') <= policy['near_duplicate_hamming']
                   and np.linalg.norm(np.asarray(item['rgb_mean'])-color) < 12
                   for h, color in hashes):
                rejected[item['id']] = ['similar_appearance']; continue
            selected.append(item); cells[item['pose_cell']] += 1
            hashes.append((item['dhash'], np.asarray(item['rgb_mean'])))
    return sorted(selected, key=lambda r: r['id']), rejected


def joint_features(sample_id, depth_item, normal_item, readers, dataset):
    _, depth_result, depth_row = depth_item
    _, normal_result, normal_row = normal_item
    reasons = list(dict.fromkeys(depth_result['hard_reasons'] + normal_result['hard_reasons']))
    if reasons: return dict(id=sample_id, reasons=reasons)
    if depth_result['original_size'] != normal_result['original_size']:
        return dict(id=sample_id, reasons=['depth_normal_dimensions_differ'])
    if depth_result['rgb_hash_32'] != normal_result['rgb_hash_32']:
        return dict(id=sample_id, reasons=['depth_normal_RGB_differ'])
    masks = []
    for row, reader in ((depth_row, readers[0]), (normal_row, readers[1])):
        name = row.get('loss_mask') or row.get('valid_mask') or row.get('mask')
        masks.append(np.asarray(reader.image(row, name, 'L')) > 0)
    valid = masks[0] & masks[1]
    box, warnings = edge_trim(valid, max_edge_fraction=.25, min_retained_area=.75,
                              max_strip_valid_fraction=.5, min_side=64)
    left, top, right, bottom = box
    crop_valid = valid[top:bottom, left:right]
    if warnings: reasons.append('unsafe_edge_crop')
    if not crop_valid.any(): reasons.append('empty_joint_validity')
    flags = set(depth_result['review_flags'] + normal_result['review_flags'])
    for flag in ('near_constant_rgb', 'near_uniform_black_or_white_rgb',
                 'raw_depth_vs_stored_encoding_disagreement', 'normal_encoding_missing'):
        if flag in flags: reasons.append(flag)
    # A deliberately conservative training selection, not a claim that complex
    # native geometry is mislabeled. Keep originals and record the exclusion.
    if 'rough_normal_field_at_review_resolution' in flags:
        reasons.append('rough_normal_field_conservative_quality_filter')
    paired = normal_result.get('paired_geometry', {})
    if (paired.get('status') == 'checked' and paired['mean_angle'] > 20
            and paired['axis_improvement'] > 10):
        reasons.append('normal_basis_requires_source_review')
    signals = depth_result['signals']
    close = signals.get('depth_below_0_5m_fraction')
    median_depth = signals.get('depth_quantiles', [None]*5)[2]
    invalid = float(1-crop_valid.mean())
    if dataset == 'replica':
        if depth_result.get('depth_audit_method') != 'raw_metric':
            raise ValueError('Replica close-view screening requires raw metric depth')
        if close > REPLICA['max_close_fraction']: reasons.append('close_surface_dominates')
        if median_depth < REPLICA['min_median_depth']: reasons.append('median_depth_too_close')
    rgb = readers[0].image(depth_row, depth_row['conditioning_image']).crop(box)
    gray = np.asarray(rgb, dtype=np.float32).mean(-1)
    contrast = float(gray.std())
    laplacian = signals.get('laplacian_variance', 0)
    # Reject only the conjunction, not a generic blur/textureless-wall flag.
    if contrast < 5 and laplacian < 1: reasons.append('near_featureless_low_contrast_view')
    result = dict(id=sample_id, scene=depth_result['scene'], reasons=reasons,
        crop_box=list(box), invalid_after=invalid, retained_area=float(crop_valid.size/valid.size),
        dhash=difference_hash(rgb), rgb_mean=np.asarray(rgb).mean((0, 1)).tolist(),
        quality_score=float(10*(1-invalid) + min(contrast, 40)/40
                            + min(math.log1p(laplacian), 5)/5 - (close or 0)),
        review_flags=sorted(flags), depth_below_0_5m_fraction=close,
        normal_geometry_check=paired.get('status', 'not_sampled'))
    if dataset == 'replica': result['pose_cell'] = pose_cell(depth_row)
    return result


def clean_entry(entry):
    return {k:v for k,v in entry.items() if k in
            ('name', 'type', 'path', 'index_path', 'weight', 'allow_missing_loss_mask')}


def build(audit, output, dataset=None, nonzero=None, workers=8):
    if output.exists(): raise FileExistsError('Use a new output version; do not overwrite a prepared subset')
    plan = checked_plan(audit)
    for i in range(len(plan['public_zip_assets'])):
        verified = json.loads((audit/'verification'/f'{i}.json').read_text())
        if verified.get('status') != 'PASS': raise ValueError('Full source content verification missing')
    entries = [e for e in plan['sources'] if e['lane'] == 'current'
               and (dataset is None or e['dataset'] == dataset)]
    domains = sorted({e['dataset'] for e in entries})
    pairs = {}; features = {}; records = {}; unmatched = {}
    # Complete checks before publishing any derivative index.
    for domain in domains:
        depth = next(e for e in entries if e['dataset'] == domain and not is_normal(e))
        normal = next(e for e in entries if e['dataset'] == domain and is_normal(e))
        drows, nrows = load_records(audit, depth), load_records(audit, normal)
        missing = set(drows) ^ set(nrows)
        if len(missing) > max(10, max(len(drows), len(nrows))*.01):
            raise ValueError(f'{domain}: depth/normal ID sets differ ({len(missing)}); investigate source registration')
        unmatched[domain] = dict(depth_only=sorted(set(drows)-set(nrows)),
                                 normals_only=sorted(set(nrows)-set(drows)))
        pairs[domain] = (depth, normal); records[domain] = (drows, nrows)
        readers = (Reader(depth), Reader(normal))
        features[domain] = []
        identifiers = sorted(set(drows)&set(nrows))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for number, feature in enumerate(executor.map(
                    lambda k: joint_features(k, drows[k], nrows[k], readers, domain), identifiers), 1):
                features[domain].append(feature)
                if number % 2048 == 0 or number == len(identifiers):
                    print(f'PAIR QUALITY {domain}: {number}/{len(identifiers)}', flush=True)
        failures = sum(any(reason in f['reasons'] for reason in ('unreadable_or_misaligned_triplet',
                       'depth_normal_dimensions_differ', 'depth_normal_RGB_differ')) for f in features[domain])
        if failures > max(10, len(drows)*.05):
            raise ValueError(f'{domain}: widespread decode failures need source repair, not filtering')
        print(f'CHECKED {domain}: {len(drows)} paired frames', flush=True)
    output.mkdir(parents=True)
    for folder in ('indices', 'manifests', 'reports'): (output/folder).mkdir()
    write_json(output/'plan.json', dict(contract=VERSION, audit=str(audit.resolve()),
        audit_plan_sha256=digest(audit/'plan.json'), sources=entries, policies=POLICIES,
        replica_policy=REPLICA, selection_code_sha256=digest(Path(__file__)),
        training_only=True, evaluation_unchanged=True, originals_unchanged=True,
        quality_caution='Coverage/contrast/diversity screens are not semantic quality or model-performance certification'))
    template = json.loads(Path(plan['source_manifest']).read_text())
    templates = {'zeroshot': template}
    if nonzero is not None:
        expected = Path(str(nonzero)+'.sha256').read_text().split()[0]
        if digest(nonzero) != expected: raise ValueError('NZS manifest checksum mismatch')
        templates['nonzero'] = json.loads(nonzero.read_text())
    reports = {}; replacement_names = {e['name'] for e in entries}
    for variant, policy in POLICIES.items():
        replacements = {}; decisions = {}
        for domain in domains:
            candidates = []; reason_ids = {}
            for feature in features[domain]:
                reasons = list(feature['reasons'])
                limit = min(policy['max_invalid'], REPLICA['max_invalid']) if domain == 'replica' else policy['max_invalid']
                if feature.get('invalid_after', 1) > limit + 1e-12: reasons.append('joint_invalid_coverage')
                if feature.get('retained_area', 0) < policy['min_area']: reasons.append('retained_area')
                if reasons: reason_ids[feature['id']] = reasons
                else: candidates.append(feature)
            if domain == 'replica':
                kept, diverse_rejections = select_diverse(candidates)
                reason_ids.update(diverse_rejections)
            else:
                # Exact thumbnail duplicates only; do not aggressively remove
                # different outdoor views or change domain balance implicitly.
                kept = []; seen = set()
                for feature in sorted(candidates, key=lambda f: (-f['quality_score'], f['id'])):
                    rgb_hash = records[domain][0][feature['id']][1]['rgb_hash_32']
                    if rgb_hash in seen: reason_ids[feature['id']] = ['exact_rgb_thumbnail_duplicate']
                    else: kept.append(feature); seen.add(rgb_hash)
            if not kept: raise ValueError(f'{domain}/{variant}: no safe paired frames; source needs repair or explicit exclusion')
            original_scenes = {r[1]['scene'] for r in records[domain][0].values()}
            retained_scenes = Counter(f['scene'] for f in kept)
            if domain == 'replica' and set(retained_scenes) != original_scenes:
                raise ValueError('Replica curation lost an entire scene; review before publishing')
            reasons = Counter(reason for rs in reason_ids.values() for reason in rs)
            decisions[domain] = dict(original_paired=len(features[domain]), original_depth=len(records[domain][0]),
                original_normals=len(records[domain][1]), unmatched_frame_ids=unmatched[domain], quality_candidates=len(candidates),
                kept=len(kept), removed=len(reason_ids), retained_scenes=dict(retained_scenes),
                reason_counts_nonexclusive=dict(reasons),
                max_invalid_kept=max(f['invalid_after'] for f in kept),
                semantic_quality_certified=False)
            write_json(output/'reports'/f'{domain}_{variant}_decisions.json', dict(
                kept=[{k:v for k,v in f.items() if k != 'dhash'} for f in kept], rejected=reason_ids))
            for entry, source_records in zip(pairs[domain], records[domain]):
                selected = []
                for f in kept:
                    ordinal, _, original = source_records[f['id']]
                    row = copy.deepcopy(original)
                    row['training_crop_box'] = f['crop_box']; row['training_crop_contract'] = CONTRACT
                    row.setdefault('metadata', {})['training_curation'] = dict(contract=VERSION, variant=variant,
                        joint_depth_normal_frame=True, invalid_after=f['invalid_after'])
                    selected.append((ordinal, json.dumps(row, sort_keys=True)))
                selected.sort()
                name = entry['name']
                replacements[name] = write_view(clean_entry(entry), output/'indices'/f'{name}.{variant}.sqlite',
                                                selected, 'zs_paired_'+variant)
        for mode, source in templates.items():
            masses = Counter()
            for e in source['datasets']: masses[task_group(e)] += e['weight']
            selected = []
            for entry in source['datasets']:
                if entry['name'] in replacement_names:
                    replacement = {**replacements[entry['name']], 'weight':entry['weight']}
                    if entry['path'] != replacement['path'] or entry['index_path'] != next(
                            e['index_path'] for e in entries if e['name'] == entry['name']):
                        raise ValueError('NZS/ZS source identity mismatch; cannot silently substitute')
                    selected.append(replacement)
                else: selected.append(copy.deepcopy(entry))
            for scope in ('full', 'geometry'):
                rows = [e for e in selected if scope == 'full' or task_group(e) != 'other']
                write_json(output/'manifests'/f'{mode}_{variant}_{scope}.json', {**source,
                    'datasets':normalize(rows, masses), 'curation_contract':dict(contract=VERSION,
                    variant=variant, domains=domains, unchanged_non_ZS_sources=True,
                    evaluation_unchanged=True, originals_unchanged=True,
                    holes='Retained holes remain invalid; no fabricated GT',
                    plan_sha256=digest(output/'plan.json'))})
        reports[variant] = decisions
    write_json(output/'reports/selection_summary.json', dict(variants=reports, domains=domains,
        mode='Matched training-only frame selection; dataset/task weights preserved',
        limitation='RGB texture/rendering defects cannot be repaired by filtering or label reorientation'))
    write_json(output/'PREPARED.json', dict(status='paired_indices_and_manifests_prepared',
        variants=list(POLICIES), manifests=[str(p.resolve()) for p in sorted((output/'manifests').glob('*.json'))],
        acceptance_required='Production loader masked/unmasked, benchmark exclusions and asset checks',
        originals_unchanged=True, training_submitted=False, public_upload='not performed'))
    print(json.dumps(reports, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset', choices=['replica', 'hypersim', 'matterport3d', 'synscapes', 'taskonomy'])
    p.add_argument('--nonzero', type=Path)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args(); build(a.audit, a.output, a.dataset, a.nonzero, a.workers)


if __name__ == '__main__': main()
