#!/usr/bin/env python3
"""Training-only base-clean views: retain detail, reject demonstrably broken labels.

No parent archive/index is modified. Suspicious texture/roughness is reported,
not used to remove genuine geometry. Full native-resolution scans are required.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.indexed_tar import IndexedTarReference, open_reference_image, validate_archive_fingerprint
from common.training.asset_lock import digest
from scripts.dataset_prep.build_geometry_readiness_manifests import write_json, task_group, normalize

VERSION = 'geometry-base-clean-native-v1'
DEFAULT = Path('data/geometry_base_clean_20261004')
CONTROL = Path('data/zs_upright_curated_20261002/manifests/zeroshot_masked_full.json')
BASE = Path('data/scannet_location_correction_20261003/manifests/curated/nonzero_trim_invalid25_full.json')


def read(path):
    return json.loads(Path(path).read_text())


def readonly(path):
    return sqlite3.connect(f'file:{path}?mode=ro', uri=True)


def checked(path):
    if digest(path) != Path(str(path)+'.sha256').read_text().split()[0]:
        raise ValueError(f'Checksum mismatch: {path}')
    return read(path)


def initialize(root):
    zs, nzs = checked(CONTROL), checked(BASE)
    replacements = {e['name']: e for e in zs['datasets']}
    if not replacements.keys() <= {e['name'] for e in nzs['datasets']}:
        raise ValueError('Base does not contain all shared ZS sources')
    # Keep base task/source probabilities while replacing stale shared ZS data.
    nzs = copy.deepcopy(nzs)
    nzs['datasets'] = [{**copy.deepcopy(replacements[e['name']]), 'weight': e['weight']}
                       if e['name'] in replacements else e for e in nzs['datasets']]
    entries = {e['name']: e for e in zs['datasets']+nzs['datasets'] if task_group(e) != 'other'}
    plan = dict(version=VERSION, parent_manifests={str(p): digest(p) for p in (CONTROL, BASE)},
                templates={'zeroshot': zs, 'base': nzs},
                sources=[{**e, 'source_index_sha256': digest(Path(e['index_path']))}
                         for e in sorted(entries.values(), key=lambda e: e['name'])],
                policy={'normal_zero_vector_threshold': .05, 'roughness_exclusion': False,
                        'additional_coverage_filter': False, 'fill_holes': False},
                originals_unchanged=True, evaluation_unchanged=True)
    root.mkdir(parents=True, exist_ok=True)
    for directory in ('scans', 'reports', 'indices', 'manifests', 'logs'):
        (root/directory).mkdir(exist_ok=True)
    write_json(root/'plan.json', plan)
    print(f'Initialized {len(entries)} native-resolution source audits; no training submitted', flush=True)


def normal_signals(target, valid):
    import numpy as np
    vectors = np.asarray(target, dtype=np.float32)/127.5-1
    lengths = np.linalg.norm(vectors, axis=-1)
    zero_count = int(((lengths < .05) & valid).sum())
    return {'undefined_vectors_marked_valid': zero_count,
            'nonunit_fraction': float((np.abs(lengths[valid]-1) > .03).mean()) if valid.any() else 0.}


def inspect(item, entry, members):
    import numpy as np
    from PIL import Image, ImageOps
    from common.training.geometry_curation import source_crop_from_row, cropped_size
    number, payload = item
    row = json.loads(payload)
    result = dict(id=row.get('id'), hard_reasons=[], review_flags=[])
    try:
        mask = row.get('loss_mask') or row.get('valid_mask')
        if mask not in members:
            prefix = row.get('_metadata_member', '').rsplit('/', 1)[0]
            mask = f'{prefix}/{mask}'
        def image(member, mode):
            offset, size = members[member]
            with open_reference_image(IndexedTarReference(entry['path'], member, offset, size)) as im:
                im.load()
                return ImageOps.exif_transpose(im).convert(mode)
        rgb, target, mask_im = (image(row['conditioning_image'], 'RGB'),
                               image(row['image'], 'RGB'), image(mask, 'L'))
        if rgb.size != target.size or rgb.size != mask_im.size:
            raise ValueError('RGB/target/mask dimensions differ')
        box = source_crop_from_row(row)
        cropped_size(rgb.size, box)
        if box is not None:
            rgb, target, mask_im = rgb.crop(box), target.crop(box), mask_im.crop(box)
        mask_values = np.asarray(mask_im)
        if not np.isin(mask_values, [0, 255]).all():
            raise ValueError('Validity mask is not binary')
        valid = mask_values > 0
        result['invalid_fraction'] = float(1-valid.mean())
        if not valid.any():
            result['hard_reasons'].append('empty_validity')
        if task_group(entry) == 'normals':
            result.update(normal_signals(target, valid))
            if result['undefined_vectors_marked_valid']:
                result['hard_reasons'].append('undefined_normal_vectors_marked_valid')
            if result['nonunit_fraction'] > .05:
                result['review_flags'].append('nonunit_vectors_review_only')
        # Do not confuse textureless walls, real discontinuities or foliage with
        # bad labels. Existing roughness flags stay review-only in this tier.
        result['review_flags'] += row.get('metadata', {}).get('training_curation', {}).get('review_flags', [])
    except (OSError, ValueError, KeyError, TypeError, Image.DecompressionBombError) as error:
        result['hard_reasons'] = ['unreadable_or_misaligned_triplet']
        result['error'] = f'{type(error).__name__}: {error}'
    return number, json.dumps(result, sort_keys=True), payload


def scan(root, source_number, workers=8):
    plan = checked(root/'plan.json'); entry = plan['sources'][source_number]
    source = Path(entry['index_path'])
    if digest(source) != entry['source_index_sha256']:
        raise ValueError('Parent index changed')
    identity = json.dumps({'version': VERSION, 'source': entry, 'policy': plan['policy']}, sort_keys=True)
    with readonly(source) as src, sqlite3.connect(root/'scans'/f'{entry["name"]}.sqlite') as dst:
        metadata = dict(src.execute('select key,value from metadata'))
        validate_archive_fingerprint(Path(entry['path']), json.loads(metadata['archive_fingerprint']))
        members = {n: (o, s) for n, o, s in src.execute('select name,offset,size from members')}
        dst.execute('create table if not exists metadata(key text primary key,value text)')
        dst.execute('create table if not exists records(row_number integer primary key,result text,payload text)')
        old = dict(dst.execute('select key,value from metadata'))
        if old and old.get('identity') != identity:
            raise ValueError('Refusing to reuse an incompatible scan')
        dst.execute('insert or replace into metadata values (?,?)', ('identity', identity)); dst.commit()
        done = {n for n, in dst.execute('select row_number from records')}
        total = src.execute('select count(*) from rows').fetchone()[0]
        cursor = src.execute('select row_number,payload from rows order by row_number')
        with ThreadPoolExecutor(max_workers=workers) as executor:
            while batch := cursor.fetchmany(128):
                pending = [r for r in batch if r[0] not in done]
                dst.executemany('insert into records values (?,?,?)',
                                executor.map(lambda r: inspect(r, entry, members), pending))
                dst.commit()
                if batch[-1][0] % 2048 < 128:
                    print(entry['name'], dst.execute('select count(*) from records').fetchone()[0], '/', total, flush=True)
        if dst.execute('select count(*) from records').fetchone()[0] != total:
            raise ValueError('Incomplete native scan')
        dst.execute('insert or replace into metadata values (?,?)', ('complete', 'true'))
        dst.commit()
    print('PASS full native scan', entry['name'], total, flush=True)


def validate_pools(paths):
    payloads = {name: checked(Path(path)) for name, path in paths.items()}
    masses = {}
    for name, p in payloads.items():
        total = Counter()
        for e in p['datasets']:
            total[task_group(e)] += e['weight']
        if any(abs(total[g]-v) > 1e-10 for g, v in {'depth': .1, 'normals': .2, 'other': .7}.items()):
            raise ValueError(f'Changed task masses: {name}/{total}')
        masses[name] = total
    z = {e['name']: e for e in payloads['zeroshot']['datasets']}
    b = {e['name']: e for e in payloads['base']['datasets']}
    if not z.keys() < b.keys():
        raise ValueError('NZS base-clean must extend ZS base-clean')
    for name, e in z.items():
        if {k:v for k,v in e.items() if k != 'weight'} != {k:v for k,v in b[name].items() if k != 'weight'}:
            raise ValueError(f'Shared source changed: {name}')
        if task_group(e) == 'other' and abs(e['weight']-b[name]['weight']) > 1e-10:
            raise ValueError('Other-task sampling changed')
    if 'added_only' in payloads:
        expected = {name for name in b if name not in z or task_group(b[name]) == 'other'}
        actual = {e['name']: e for e in payloads['added_only']['datasets']}
        if actual.keys() != expected:
            raise ValueError('Added-only view has an unexpected source set')
        for name, e in actual.items():
            if {k:v for k,v in e.items() if k != 'weight'} != {k:v for k,v in b[name].items() if k != 'weight'}:
                raise ValueError('Added-only view changed source content')
    return masses


def finalize(root):
    from scripts.dataset_prep.curate_nzs_geometry import write_view
    plan = checked(root/'plan.json')
    for path, expected in plan['parent_manifests'].items():
        if digest(Path(path)) != expected:
            raise ValueError('Parent manifest changed')
    entries, reports = {}, {}
    for source in plan['sources']:
        if digest(Path(source['index_path'])) != source['source_index_sha256']:
            raise ValueError('Parent index changed')
        scan_path = root/'scans'/f'{source["name"]}.sqlite'
        if not scan_path.is_file():
            raise ValueError(f'Native scan missing: {source["name"]}; retry data preparation, not training')
        with readonly(scan_path) as db, readonly(Path(source['index_path'])) as parent:
            expected_identity = json.dumps({'version': VERSION, 'source': source, 'policy': plan['policy']}, sort_keys=True)
            meta = dict(db.execute('select key,value from metadata'))
            if meta.get('complete') != 'true' or meta.get('identity') != expected_identity:
                raise ValueError('Incomplete or incompatible scan')
            actual = list(db.execute('select row_number,result,payload from records order by row_number'))
            originals = list(parent.execute('select row_number,payload from rows order by row_number'))
        if [(n,p) for n,_,p in actual] != originals:
            raise ValueError('Full audit row coverage/content differs from parent')
        keep, removed, flags, reasons = [], [], Counter(), Counter()
        for n, result, payload in actual:
            r = json.loads(result)
            if r['hard_reasons']:
                reasons.update(r['hard_reasons']); removed.append(r)
            else:
                keep.append((n,payload)); flags.update(r['review_flags'])
        if not keep:
            raise ValueError(f'All rows rejected: {source["name"]}; review rather than silently renormalize')
        entry = {k:v for k,v in source.items() if k != 'source_index_sha256'}
        if removed:
            entry = write_view(entry, root/'indices'/f'{source["name"]}.base_clean.sqlite', keep, VERSION)
        entries[source['name']] = entry
        reports[source['name']] = dict(before=len(actual), after=len(keep), rejected=len(removed),
            hard_reasons=dict(reasons), retained_review_flags=dict(flags), source_index_sha256=source['source_index_sha256'])
        write_json(root/'reports'/f'{source["name"]}_removed.json', removed)
    paths = {}
    for name, template in plan['templates'].items():
        out = copy.deepcopy(template)
        out['datasets'] = [{**entries[e['name']], 'weight': e['weight']} if e['name'] in entries else e
                           for e in out['datasets']]
        out['base_clean_contract'] = dict(version=VERSION, plan_sha256=digest(root/'plan.json'),
            roughness_exclusion=False, coverage_unchanged=True, originals_unchanged=True)
        paths[name] = root/'manifests'/f'{name}_full.json'; write_json(paths[name], out)
    added = copy.deepcopy(read(paths['base']))
    zs_names = {e['name'] for e in read(paths['zeroshot'])['datasets']}
    # This optional view excludes ZS GEOMETRY, not the 70% non-geometry tasks.
    added['datasets'] = normalize([e for e in added['datasets'] if e['name'] not in zs_names or task_group(e) == 'other'],
                                  {'depth': .1, 'normals': .2, 'other': .7})
    added['base_clean_contract']['zs_geometry_excluded'] = True
    paths['added_only'] = root/'manifests/added_only_full.json'; write_json(paths['added_only'], added)
    validate_pools(paths)
    from scripts.analysis.audit_selected_geometry_training import audit
    from common.training.benchmark_exclusions import PINNED_SPLIT_ROOT
    for name, path in paths.items():
        evidence = audit(path, PINNED_SPLIT_ROOT)
        write_json(root/'reports'/f'{name}_selected_rows.json', evidence)
        if evidence['status'] != 'PASS':
            raise ValueError('Selected-row benchmark/location holdout audit failed')
    write_json(root/'reports/selection.json', dict(sources=reports, policy=plan['policy']))
    write_json(root/'READY.json', dict(status='full_native_structural_and_selected_row_checks_passed',
        plan_sha256=digest(root/'plan.json'), manifests={str(p):digest(p) for p in paths.values()},
        originals_unchanged=True, semantic_quality_certified=False, training_quality_improvement_proven=False))
    print('PASS base-clean full native/selected-row checks; real loader remains campaign-CPU gated', flush=True)


def verify(root, paths):
    if not (root/'READY.json').is_file():
        raise ValueError('Base-clean native CPU audits are not finished. Run data.sh status; do not submit training yet.')
    ready = checked(root/'READY.json')
    if ready['status'] != 'full_native_structural_and_selected_row_checks_passed' or ready['plan_sha256'] != digest(root/'plan.json'):
        raise ValueError('Base-clean preparation not accepted')
    for path, expected in ready['manifests'].items():
        if digest(Path(path)) != expected:
            raise ValueError('Accepted base-clean manifest changed')
    # Frozen copies must match the accepted source manifests exactly.
    for name, path in paths.items():
        if digest(Path(path)) != digest(root/'manifests'/f'{name}_full.json'):
            raise ValueError('Campaign manifest does not match accepted base-clean view')
    # A manifest hash alone cannot prove that its mutable SQLite view still
    # contains the audited rows. Verify unchanged index hashes, and compare any
    # derivative against the full native audit (not a caller-updated sidecar).
    plan = checked(root/'plan.json')
    accepted = {e['name']: e for e in read(root/'manifests/base_full.json')['datasets']}
    import itertools
    for source in plan['sources']:
        index = Path(accepted[source['name']]['index_path'])
        if index.resolve() == Path(source['index_path']).resolve():
            if digest(index) != source['source_index_sha256']:
                raise ValueError(f'Audited parent index changed: {source["name"]}')
        else:
            with readonly(root/'scans'/f'{source["name"]}.sqlite') as scan_db, readonly(index) as view:
                expected = scan_db.execute("select row_number,payload from records where json_array_length(json_extract(result,'$.hard_reasons'))=0 order by row_number")
                actual = view.execute('select row_number,payload from rows order by row_number')
                if any(a != b for a, b in itertools.zip_longest(expected, actual)):
                    raise ValueError(f'Clean index no longer matches audited retained rows: {source["name"]}')
    masses = validate_pools({name: root/'manifests'/f'{name}_full.json'
                             for name in ('zeroshot', 'base', 'added_only')})
    return {name: masses[name] for name in paths if name in masses}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['initialize', 'scan', 'finalize'])
    p.add_argument('--root', type=Path, default=DEFAULT)
    p.add_argument('--source', type=int)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    if a.action == 'initialize': initialize(a.root)
    elif a.action == 'scan':
        if a.source is None or a.workers < 1: p.error('scan requires --source and positive --workers')
        scan(a.root, a.source, a.workers)
    else: finalize(a.root)


if __name__ == '__main__':
    main()
