#!/usr/bin/env python3
"""Build additive NZS training index views; never rewrite source archives/indices."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tarfile
import time

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.indexed_tar import IndexedTarReference, open_reference_image, validate_archive_fingerprint
from common.training.asset_lock import digest
from common.training.geometry_curation import CONTRACT, edge_trim, quality_signals
from scripts.dataset_prep.build_geometry_readiness_manifests import normalize, task_group, write_json


def readonly(path):
    return sqlite3.connect(f'file:{path}?mode=ro', uri=True)


def initialize(campaign, source, include_scannet=False, reuse_campaign=None):
    full_path = source / 'nonzero_full.json'
    zs_path = source / 'zeroshot_full.json'
    full = json.loads(full_path.read_text())
    zs_names = {e['name'] for e in json.loads(zs_path.read_text())['datasets']}
    additions = [e for e in full['datasets'] if task_group(e) != 'other'
                 and e['name'] not in zs_names
                 and (include_scannet or 'scannet' not in e['name'].lower())]
    for entry in additions:
        if entry['type'] != 'indexed_prepared_tar':
            raise ValueError('Curation requires indexed prepared archives')
    plan = {'contract': CONTRACT, 'source_full_manifest': str(full_path.resolve()),
            'source_full_sha256': digest(full_path), 'source_zs_manifest': str(zs_path.resolve()),
            'source_zs_sha256': digest(zs_path), 'scannet_excluded_in_variants': not include_scannet,
            'edge_limits': {'max_edge_fraction': 0.25, 'min_retained_area': 0.6,
                            'max_strip_valid_fraction': 0.5, 'min_side': 32},
            'sources': [{**e, 'source_index_sha256': digest(Path(e['index_path']))} for e in additions]}
    if reuse_campaign is not None:
        previous = json.loads((reuse_campaign/'plan.json').read_text())
        # Byte-identical inputs are not sufficient if the quality/trimming logic
        # changed. Check the frozen implementation before allowing scan reuse.
        snapshot = reuse_campaign/'code_snapshot.tar'
        expected = (reuse_campaign/'code_snapshot.tar.sha256').read_text().split()[0]
        if digest(snapshot) != expected:
            raise ValueError('Reuse campaign code snapshot checksum mismatch')
        with tarfile.open(snapshot) as archive:
            old_builder = archive.extractfile('scripts/dataset_prep/curate_nzs_geometry.py').read().decode()
            old_quality = archive.extractfile('common/training/geometry_curation.py').read()
        def scan_ast(source_text):
            return ast.dump(next(n for n in ast.parse(source_text).body
                                 if isinstance(n,ast.FunctionDef) and n.name=='inspect_row'))
        if (scan_ast(old_builder) != scan_ast(Path(__file__).read_text())
                or old_quality != (ROOT/'common/training/geometry_curation.py').read_bytes()
                or previous['contract'] != plan['contract']
                or previous['edge_limits'] != plan['edge_limits']):
            raise ValueError('Reuse campaign scan implementation/limits differ')
        originals = {e['name']: e for e in previous['sources']}
        plan['reuse_campaign'] = str(reuse_campaign.resolve())
        plan['reuse_scan_sources'] = [e['name'] for e in plan['sources'] if originals.get(e['name'])==e]
    write_json(campaign / 'plan.json', plan)
    for directory in ('scans', 'indices', 'manifests', 'reports'):
        (campaign / directory).mkdir(exist_ok=True)
    # This is a pointer to the original full manifest, not an altered replacement.
    write_json(campaign / 'original_full_pool.json', {
        'manifest': str(full_path.resolve()), 'sha256': digest(full_path),
        'unchanged': True, 'contains_scannet': any('scannet' in e['name'].lower() for e in full['datasets'])})
    print(f'Initialized {len(additions)} additional NZS sources; original full pool untouched', flush=True)


def inspect_row(item, entry, members, limits):
    ordinal, payload = item
    row = json.loads(payload)
    result = {'row_number': ordinal, 'id': row.get('id'), 'hard_reasons': [], 'review_flags': []}
    try:
        mask = row.get('loss_mask') or row.get('valid_mask')
        if not row.get('loss_mask') and mask not in members:
            prefix = str(row.get('_metadata_member', '')).rsplit('/', 1)[0]
            candidate = f'{prefix}/{mask}'
            if candidate in members:
                mask = candidate

        def image(member, mode):
            offset, length = members[member]
            with open_reference_image(IndexedTarReference(entry['path'], member, offset, length)) as opened:
                opened.load()  # Reject truncated/corrupt files, not just bad headers.
                return ImageOps.exif_transpose(opened).convert(mode)

        mask_image = image(mask, 'L')
        target = image(row['image'], 'RGB')
        rgb = image(row['conditioning_image'], 'RGB')
        if not (rgb.size == target.size == mask_image.size):
            raise ValueError('RGB/GT/mask dimensions differ')
        values = np.asarray(mask_image)
        if not set(np.unique(values)).issubset({0, 255}):
            raise ValueError('Validity mask is not binary 0/255')
        valid = values > 0
        box, warnings = edge_trim(valid, **limits)
        left, top, right, bottom = box
        cropped_valid = valid[top:bottom, left:right]
        result.update(original_size=list(rgb.size), crop_box=list(box),
                      invalid_before=float(1-valid.mean()),
                      invalid_after=float(1-cropped_valid.mean()),
                      retained_area=float(cropped_valid.size/valid.size), edge_warnings=warnings)
        small_size = (max(3, round((right-left)*min(1,320/max(right-left,bottom-top)))),
                      max(3, round((bottom-top)*min(1,320/max(right-left,bottom-top)))))
        small_rgb = rgb.crop(box).resize(small_size, Image.Resampling.BILINEAR)
        small_target = target.crop(box).resize(small_size, Image.Resampling.NEAREST)
        small_mask = mask_image.crop(box).resize(small_size, Image.Resampling.NEAREST)
        signals, flags, hard = quality_signals(small_rgb, small_target, np.asarray(small_mask)>0,
                                              normals=task_group(entry)=='normals')
        if cropped_valid.any() and 'empty_validity' in hard:
            hard.remove('empty_validity')
            flags.append('sparse_validity_not_seen_at_review_resolution')
        result.update(signals=signals, review_flags=flags, hard_reasons=hard)
        if not cropped_valid.any() and 'empty_validity' not in result['hard_reasons']:
            result['hard_reasons'].append('empty_validity')
    except (OSError, ValueError, KeyError, TypeError) as error:
        result['hard_reasons'] = ['unreadable_or_misaligned_triplet']
        result['error'] = f'{type(error).__name__}: {error}'
    return ordinal, json.dumps(result, sort_keys=True), payload


def scan(campaign, source_number, workers, limit):
    plan = json.loads((campaign/'plan.json').read_text())
    entry = plan['sources'][source_number]
    source = Path(entry['index_path'])
    if digest(source) != entry['source_index_sha256']:
        raise ValueError('Source index changed since initialization')
    output = campaign/'scans'/f'{entry["name"]}.sqlite'
    identity = json.dumps({'entry': entry, 'edge_limits': plan['edge_limits'], 'limit': limit}, sort_keys=True)
    with readonly(source) as connection:
        info = dict(connection.execute('select key,value from metadata'))
        validate_archive_fingerprint(Path(entry['path']), json.loads(info['archive_fingerprint']))
        members = {name: (offset, size) for name,offset,size in connection.execute('select name,offset,size from members')}
        total = connection.execute('select count(*) from rows').fetchone()[0]
        expected = min(total, limit) if limit else total
        with sqlite3.connect(output) as destination:
            destination.execute('create table if not exists metadata(key text primary key,value text)')
            destination.execute('create table if not exists records(row_number integer primary key,result text,payload text)')
            existing = dict(destination.execute('select key,value from metadata'))
            if existing and existing.get('identity') != identity:
                raise ValueError('Refusing to reuse a scan from a different configuration')
            destination.execute('insert or replace into metadata values (?,?)', ('identity', identity))
            destination.commit()
            completed = {n for n, in destination.execute('select row_number from records')}
            cursor = connection.execute('select row_number,payload from rows order by row_number limit ?', (expected,))
            start = time.monotonic()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                while batch := cursor.fetchmany(256):
                    batch = [item for item in batch if item[0] not in completed]
                    results = executor.map(lambda item: inspect_row(item, entry, members, plan['edge_limits']), batch)
                    destination.executemany('insert into records values (?,?,?)', results)
                    destination.commit()
                    count = destination.execute('select count(*) from records').fetchone()[0]
                    if count % 4096 < 256 or count == expected:
                        print(f'{entry["name"]}: {count}/{expected}; elapsed {time.monotonic()-start:.1f}s', flush=True)
            count = destination.execute('select count(*) from records').fetchone()[0]
            if count != expected:
                raise ValueError('Incomplete scan')
            destination.execute('insert or replace into metadata values (?,?)', ('complete', 'true'))
            destination.execute('insert or replace into metadata values (?,?)', ('full_source', json.dumps(limit == 0)))
            destination.commit()
    print(f'PASS source scan {entry["name"]}, {expected} triplets; full_source={not limit}', flush=True)


def write_view(source_entry, destination, records, variant):
    source = Path(source_entry['index_path'])
    if source.resolve() == destination.resolve():
        raise ValueError('Cannot replace a source index')
    if destination.exists():
        with readonly(destination) as existing:
            metadata = dict(existing.execute('select key,value from metadata'))
            actual = list(existing.execute('select row_number,payload from rows order by row_number'))
        if (actual != records or metadata.get('view_contract') != CONTRACT
                or metadata.get('source_index_sha256') != digest(source)
                or metadata.get('curation_variant') != variant):
            raise FileExistsError(f'Refusing to overwrite incompatible index: {destination}')
        return {**source_entry, 'index_path': str(destination.resolve())}
    temporary = destination.with_suffix('.partial')
    if temporary.exists():
        raise FileExistsError(f'Incomplete output needs inspection: {temporary}')
    with readonly(source) as source_db, sqlite3.connect(temporary) as target_db:
        source_db.backup(target_db)
        target_db.execute('delete from rows')  # Only the newly created derivative copy.
        target_db.executemany('insert into rows values (?,?)', records)
        target_db.execute('insert or replace into metadata values (?,?)', ('num_rows', str(len(records))))
        target_db.execute('insert or replace into metadata values (?,?)', ('view_contract', CONTRACT))
        target_db.execute('insert or replace into metadata values (?,?)', ('curation_variant', variant))
        target_db.execute('insert or replace into metadata values (?,?)', ('source_index_sha256', digest(source)))
        target_db.commit()
    temporary.replace(destination)
    destination.with_suffix(destination.suffix+'.sha256').write_text(f'{digest(destination)}  {destination.name}\n')
    return {**source_entry, 'index_path': str(destination.resolve())}


def reuse_scans(campaign, plan):
    """Copy completed equivalent scan DBs only; never mutate previous scans."""
    if not plan.get('reuse_scan_sources'):
        return
    previous_root = Path(plan['reuse_campaign'])
    previous = json.loads((previous_root/'plan.json').read_text())
    originals = {e['name']:e for e in previous['sources']}
    for entry in plan['sources']:
        name = entry['name']
        if name not in plan['reuse_scan_sources']:
            continue
        if (originals.get(name) != entry or previous['edge_limits'] != plan['edge_limits']
                or digest(Path(entry['index_path'])) != entry['source_index_sha256']):
            raise ValueError('Reused scan source identity changed')
        source = previous_root/'scans'/f'{name}.sqlite'
        destination = campaign/'scans'/f'{name}.sqlite'
        expected = json.dumps({'entry':entry,'edge_limits':plan['edge_limits'],'limit':0},sort_keys=True)
        with readonly(source) as original:
            metadata = dict(original.execute('select key,value from metadata'))
            if (metadata.get('identity') != expected or metadata.get('complete') != 'true'
                    or metadata.get('full_source') != 'true'):
                raise ValueError(f'Reuse scan not fully complete: {name}')
            if not destination.exists():
                temporary = destination.with_suffix('.reuse.partial')
                if temporary.exists():
                    raise FileExistsError(f'Inspect incomplete reused scan: {temporary}')
                with sqlite3.connect(temporary) as target:
                    original.backup(target)
                temporary.replace(destination)
            else:
                with readonly(destination) as target:
                    actual = dict(target.execute('select key,value from metadata'))
                if actual != metadata:
                    raise ValueError(f'Existing reused scan differs: {destination}')
        print(f'PASS reused immutable scan {name}',flush=True)


def finalize(campaign):
    plan = json.loads((campaign/'plan.json').read_text())
    reuse_scans(campaign,plan)
    source_path = Path(plan['source_full_manifest'])
    if digest(source_path) != plan['source_full_sha256']:
        raise ValueError('Source manifest changed')
    if digest(Path(plan['source_zs_manifest'])) != plan['source_zs_sha256']:
        raise ValueError('ZS manifest changed')
    template = json.loads(source_path.read_text())
    exclude_scannet = plan['scannet_excluded_in_variants']
    entries = [e for e in template['datasets'] if not exclude_scannet or 'scannet' not in e['name'].lower()]
    task_masses = Counter()
    for e in template['datasets']:
        task_masses[task_group(e)] += e['weight']
    scans, summaries = {}, {}
    for e in plan['sources']:
        if digest(Path(e['index_path'])) != e['source_index_sha256']:
            raise ValueError('Source index changed')
        with readonly(campaign/'scans'/f'{e["name"]}.sqlite') as connection:
            metadata = dict(connection.execute('select key,value from metadata'))
            if metadata.get('complete') != 'true' or metadata.get('full_source') != 'true':
                raise ValueError('Incomplete or sampled scan cannot produce training manifests')
            expected_identity = json.dumps({'entry': e, 'edge_limits': plan['edge_limits'], 'limit': 0}, sort_keys=True)
            if metadata.get('identity') != expected_identity:
                raise ValueError('Scan provenance does not match plan')
            records = [(n, json.loads(r), json.loads(p)) for n,r,p in connection.execute('select * from records order by row_number')]
        scans[e['name']] = records
        hard = Counter(reason for _,r,_ in records for reason in r['hard_reasons'])
        # Large structural failure rates indicate a broken preparation/configuration,
        # not permission to silently discard most of a source.
        if hard['unreadable_or_misaligned_triplet'] > max(10, len(records)*0.05):
            raise ValueError(f'Too many structural failures in {e["name"]}; inspect scan before selection')
        fractions = [r['invalid_after'] for _,r,_ in records if 'invalid_after' in r]
        summaries[e['name']] = {'rows': len(records), 'hard_reasons': dict(hard),
            'review_flags': dict(Counter(f for _,r,_ in records for f in r['review_flags'])),
            'cropped_rows': sum(r.get('retained_area', 1) < 1 for _,r,_ in records),
            'invalid_after_quantiles': [float(x) for x in np.quantile(fractions,[0,.25,.5,.75,.9,1])] if fractions else []}

    # Thresholds are experimental, not an assertion of optimal label quality.
    reference = 'original_no_scannet' if exclude_scannet else 'original_reference'
    variants = [(reference, None, False, None, False),
                ('trim_only', 1.0, True, None, False),
                ('trim_invalid50', .50, True, None, False),
                ('trim_invalid25', .25, True, None, False),
                ('trim_invalid10', .10, True, None, False),
                ('trim_fully_valid', 0.0, True, None, False),
                ('trim_invalid25_review_screened', .25, True, None, True),
                ('trim_invalid25_without_diode_normals', .25, True, 'diode_normals', False),
                ('trim_invalid25_without_kitti_depth', .25, True, 'kitti_depth', False)]
    outputs = {}
    for variant, threshold, trim, excluded, screen in variants:
        selected, counts = [], {}
        for entry in entries:
            name = entry['name']
            if name not in scans or not trim:
                selected.append(dict(entry))
                continue
            exclusion = (excluded == 'diode_normals' and 'diode' in name and task_group(entry)=='normals') or (
                excluded == 'kitti_depth' and 'kitti' in name and 'vkitti' not in name and task_group(entry)=='depth')
            keep, reasons = [], Counter()
            for ordinal, result, row in scans[name]:
                reason = ('source_excluded' if exclusion else
                          'hard_quality_failure' if result['hard_reasons'] else
                          'remaining_invalid_coverage' if result['invalid_after'] > threshold+1e-12 else
                          'extreme_rgb_review_flag' if screen and
                          any(f in result['review_flags'] for f in ('near_constant_rgb','near_uniform_black_or_white_rgb')) else None)
                if reason:
                    reasons[reason] += 1
                    continue
                row = copy.deepcopy(row)
                row['training_crop_box'] = result['crop_box']
                row['training_crop_contract'] = CONTRACT
                row.setdefault('metadata', {})['training_curation'] = {
                    'variant': variant, 'invalid_after': result['invalid_after'],
                    'retained_area': result['retained_area'], 'review_flags': result['review_flags']}
                keep.append((ordinal,json.dumps(row,sort_keys=True)))
            counts[name] = {'retained': len(keep), 'excluded': dict(reasons)}
            if keep:
                view = write_view(entry, campaign/'indices'/f'{name}.{variant}.sqlite', keep, variant)
                selected.append(view)
        if any(not any(task_group(e)==group for e in selected) for group in task_masses):
            raise ValueError(f'Variant removes an entire task: {variant}')
        contract = {'version': CONTRACT, 'variant': variant, 'plan_sha256': digest(campaign/'plan.json'),
                    'scope': 'added NZS geometry only; shared ZS/other sources unchanged',
                    'scannet_included': not exclude_scannet,
                    'scannet_targets': template.get('mix_contract',{}).get('normal_targets'),
                    'max_remaining_invalid_fraction': threshold, 'edge_limits': plan['edge_limits'],
                    'loss_mask_policy': 'disabled is supported; masks still required for preprocessing',
                    'quality_certification': False, 'review_screened_may_remove_legitimate_low_texture_scenes': screen,
                    'camera_intrinsics': 'not consumed by RGB-target trainer; stored source calibration is not a cropped calibration'}
        for scope in ('full','geometry'):
            scoped = [e for e in selected if scope=='full' or task_group(e) != 'other']
            payload = {**template, 'datasets': normalize(scoped,task_masses), 'curation_contract': contract}
            write_json(campaign/'manifests'/f'nonzero_{variant}_{scope}.json', payload)
        outputs[variant] = counts
        print(f'PASS manifests {variant}', flush=True)
    write_json(campaign/'reports'/'selection_summary.json', {'contract': plan, 'sources': summaries, 'variants': outputs})
    write_json(campaign/'PREPARED.json', {'contract': CONTRACT, 'status': 'indices_and_manifests_prepared',
                                       'training_loader_acceptance': 'must pass separately', 'variants': list(outputs)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--initialize', type=Path, metavar='READINESS_LANE')
    group.add_argument('--scan-source', type=int)
    group.add_argument('--finalize', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0, help='Smoke scan only; cannot finalize training manifests')
    parser.add_argument('--include-scannet',action='store_true',help='Include the selected lane ScanNet sources; default still excludes them')
    parser.add_argument('--reuse-campaign',type=Path,help='Initialization only: reuse completed byte-identical inputs/scan logic')
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0:
        parser.error('Invalid workers/limit')
    if args.scan_source is not None and args.scan_source < 0:
        parser.error('Source number must be nonnegative')
    args.campaign.mkdir(parents=True, exist_ok=True)
    if args.initialize:
        initialize(args.campaign, args.initialize,args.include_scannet,args.reuse_campaign)
    elif args.scan_source is not None:
        scan(args.campaign, args.scan_source,args.workers,args.limit)
    else:
        finalize(args.campaign)


if __name__ == '__main__':
    main()
