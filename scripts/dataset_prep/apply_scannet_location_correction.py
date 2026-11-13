#!/usr/bin/env python3
"""Add physical-location-heldout ScanNet views; never mutate parent pools/TARs.

Only ScanNet entries change. Weights and row payloads (including target encoding,
masks and edge-strip crops) are preserved exactly. A corrected view is suitable
for a NEW training run, not for declaring an exposed checkpoint held-out.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.training.asset_lock import digest
from common.training.benchmark_exclusions import domain, load_registry, source_identity
from scripts.dataset_prep.build_geometry_readiness_manifests import clean_index, write_json

DEFAULT_SPLITS = Path('data/marigold_v2_depth_eval_20260920/data_split')
DEFAULT_OUTPUT = Path('data/scannet_location_correction_20261003')
DATA = Path('data')


def default_manifests():
    """Version existing pools, including diagnostic filtered variants."""
    result = {}
    sources = {
        'plane': DATA/'geometry_training_readiness_20261002/plane',
        'baseline_v4': DATA/'geometry_training_readiness_20261002/baseline',
        'curated': DATA/'nzs_geometry_curation_scannet_20261002/manifests',
        'quality_tiers': DATA/'nzs_geometry_quality_tiers_20261002/manifests',
        'upright': DATA/'zs_upright_curated_20261002/manifests',
        'staged': DATA/'zs_quality_staged_20261003_h100_seed42/manifests',
    }
    for label, folder in sources.items():
        selected = sorted(p for p in folder.glob('nzs_*.json' if label == 'staged' else 'nonzero_*.json')
                          if p.name.endswith(('_full.json', '_geometry.json')))
        if not selected:
            raise FileNotFoundError(f'Missing parent manifests: {folder}')
        for path in selected:
            result[f'{label}/{path.name}'] = path
    return result


def independent_locations(split_root):
    """Second parser: audit the output independently of the filtering helper."""
    result, files, captures, counts = set(), [], set(), {}
    for lane, relative in (
        ('depth', 'scannet_depth/scannet_val_sampled_list_800_1.txt'),
        ('normals', 'scannet_normals/scannet_test.txt'),
    ):
        path = split_root/relative
        files.append(path)
        lines = [line for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
        counts[lane] = len(lines)
        if not lines:
            raise ValueError(f'Empty ScanNet {lane} evaluation split: {path}')
        identities = set()
        for line in lines:
            components = line.split()[0].split('/')
            found = [x for x in components if re.fullmatch(r'scene[0-9]{4}_[0-9]{2}', x)]
            if len(found) != 1:
                raise ValueError(f'Ambiguous ScanNet split entry: {line}')
            frame = re.fullmatch(r'(\d+)(?:_(?:rgb|img))?', Path(components[-1]).stem)
            if frame is None:
                raise ValueError(f'Ambiguous ScanNet split frame: {line}')
            identity = (found[0], int(frame[1]))
            if identity in identities:
                raise ValueError(f'Duplicate ScanNet split frame: {identity}')
            identities.add(identity)
            captures.add(found[0])
            result.add(found[0].rsplit('_', 1)[0])
    return result, captures, counts, files


def inventory(entry, blocked):
    ids, locations, payloads = set(), set(), {}
    with sqlite3.connect(f'file:{entry["index_path"]}?mode=ro', uri=True) as conn:
        for ordinal, payload in conn.execute('select row_number,payload from rows order by row_number'):
            row = json.loads(payload)
            if domain(row) != 'scannet':
                raise ValueError(f'Non-ScanNet row in ScanNet view: {entry["name"]}/{row.get("id")}')
            _, identifier, capture = source_identity(row)
            # Split the original capture instead of using scannet_location_id.
            location = capture.rsplit('_', 1)[0]
            if identifier in ids:
                raise ValueError(f'Duplicate ScanNet source ID: {identifier}')
            ids.add(identifier)
            locations.add(location)
            payloads[ordinal] = (payload, identifier, location)
    return ids, locations, payloads


def correct_manifest(parent, target, indices, registry, blocked, split_hashes, cache):
    template = json.loads(parent.read_text())
    payload = copy.deepcopy(template)
    reports = {}
    unchanged = []
    for i, entry in enumerate(template['datasets']):
        is_scannet = ('scannet' in entry['name'].lower() or domain(entry) == 'scannet')
        if not is_scannet:
            unchanged.append(entry['name'])
            continue
        if entry.get('type') != 'indexed_prepared_tar':
            raise ValueError(f'ScanNet correction requires an indexed archive: {entry["name"]}')
        resolved = copy.deepcopy(entry)
        for key in ('path', 'index_path'):
            path = Path(entry[key])
            resolved[key] = str((path if path.is_absolute() else parent.parent/path).resolve())
        key = (resolved['path'], resolved['index_path'])
        if key not in cache:
            source_hash = digest(Path(resolved['index_path']))
            before_ids, before_locations, before_rows = inventory(resolved, blocked)
            tag = hashlib.sha256(json.dumps({'source': source_hash, 'splits': split_hashes}, sort_keys=True).encode()).hexdigest()[:24]
            view_dir = indices/tag
            view_dir.mkdir(parents=True, exist_ok=True)
            corrected, report = clean_index(resolved, view_dir, registry)
            if corrected is None:
                raise ValueError(f'No ScanNet rows remain in {entry["name"]}; review mixture explicitly')
            after_ids, after_locations, after_rows = inventory(corrected, blocked)
            expected = {ordinal: row for ordinal, row in before_rows.items() if row[2] not in blocked}
            if after_rows != expected or after_locations & blocked:
                raise ValueError(f'Independent location/payload audit failed: {entry["name"]}')
            if digest(Path(resolved['index_path'])) != source_hash:
                raise ValueError('Source index changed during correction')
            report.update({
                'source_index': resolved['index_path'], 'source_index_sha256': source_hash,
                'corrected_index': corrected['index_path'], 'corrected_index_sha256': digest(Path(corrected['index_path'])),
                'archive_unchanged': resolved['path'],
                'locations_before': len(before_locations), 'locations_after': len(after_locations),
                'overlap_locations_before': sorted(before_locations & blocked),
                'overlap_locations_after': [], 'row_payloads_preserved': True,
                'retained_ids_sha256': hashlib.sha256('\n'.join(sorted(after_ids)).encode()).hexdigest(),
                'retained_frame_ids': sorted(after_ids),
                'exclusion_policy': 'all recordings at union of depth/normal evaluation physical locations',
            })
            cache[key] = (corrected['index_path'], report)
        corrected_index, report = cache[key]
        # Other entry fields, including archive paths and weights, remain exact.
        payload['datasets'][i]['index_path'] = corrected_index
        reports[entry['name']] = {k: v for k, v in report.items() if k != 'retained_frame_ids'}
    if not reports:
        raise ValueError(f'No ScanNet source found in requested NZS pool: {parent}')
    contract = {
        'version': 'scannet-location-heldout-v1', 'name': 'THE SCANNET CORRECTION',
        'parent_manifest': str(parent.resolve()), 'parent_manifest_sha256': digest(parent),
        'split_sha256': split_hashes, 'heldout_locations': sorted(blocked),
        'policy': 'exclude every ScanNet recording at either depth or normal evaluation locations, from both tasks',
        'changed': 'ScanNet row selection only; exact source weights, crops, encodings, masks, RGB and GT preserved',
        'evaluation_unchanged': True, 'valid_with_masked_or_unmasked_loss': True,
        'requires_fresh_training_for_heldout_claim': True,
        'semantic_quality_certified': False,
    }
    payload['scannet_split_contract'] = contract
    if 'mix_contract' in payload:
        payload['mix_contract']['nzs_policy'] = (
            'ScanNet physical evaluation locations excluded across all recordings and both geometry tasks; '
            'other parent exclusions unchanged; VKITTI2 remains a variant-heldout NZS diagnostic')
    # Do not normalize: dataset and task sampling probabilities remain exact.
    if [e['weight'] for e in payload['datasets']] != [e['weight'] for e in template['datasets']]:
        raise AssertionError('Sampling weights changed')
    write_json(target, payload)
    return {'parent': str(parent), 'parent_sha256': digest(parent), 'manifest': str(target),
            'manifest_sha256': digest(target), 'unchanged_datasets': unchanged,
            'scannet': reports, 'weights_unchanged': True}


def build(manifests, split_root, output):
    registry, _ = load_registry(split_root)
    blocked, captures, counts, split_files = independent_locations(split_root)
    if blocked != registry['scannet']['locations']:
        raise ValueError('Independent evaluation-location parsers disagree')
    hashes = {str(p.resolve()): digest(p) for p in split_files}
    output.mkdir(parents=True, exist_ok=True)
    cache, reports = {}, {}
    for label, parent in manifests.items():
        relative = Path(label)
        if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.json':
            raise ValueError(f'Unsafe output label: {label}')
        reports[label] = correct_manifest(parent, output/'manifests'/relative, output/'indices', registry, blocked, hashes, cache)
        print(f'PASS location-heldout {label}', flush=True)
    for p in split_files:
        if digest(p) != hashes[str(p.resolve())]:
            raise ValueError('Evaluation split changed during correction')
    # Retention intersections are diagnostic: base/high may filter the two GT
    # tasks independently. Neither needs fabricated supervision to align IDs.
    for label, report in reports.items():
        entries = json.loads(Path(report['manifest']).read_text())['datasets']
        ids = {}
        for e in entries:
            if e['name'] in report['scannet']:
                ids[e['name']] = inventory(e, blocked)[0]
        if len(ids) == 2:
            a, b = ids.values()
            report['shared_depth_normal_frames'] = len(a & b)
            report['depth_normal_frame_selection_identical'] = a == b
    summary = {
        'status': 'PASS', 'correction': 'THE SCANNET CORRECTION', 'heldout_location_count': len(blocked),
        'heldout_capture_count': len(captures), 'evaluation_split_images': counts, 'evaluation_split_sha256': hashes,
        'manifests': reports, 'independent_zero_location_overlap_audit': True,
        'verification_scope': 'all ScanNet row identities/payloads and indexed archive fingerprints; loader validation is separate',
        'originals_and_existing_jobs_unchanged': True, 'evaluation_unchanged': True,
        'historical_checkpoint_exposure': 'not inferred from these replacement manifests; audit original training provenance',
    }
    write_json(output/'REPORT.json', summary)
    write_json(output/'MANIFESTS_READY.json', {'status': 'location_split_and_payload_checks_passed',
        'report_sha256': digest(output/'REPORT.json'), 'manifest_count': len(reports),
        'loader_checks_required': True, 'requires_fresh_training_for_heldout_claim': True})
    return summary


def finalize_loader_checks(output):
    report_path = output/'REPORT.json'
    report = json.loads(report_path.read_text())
    if report.get('status') != 'PASS' or not report.get('independent_zero_location_overlap_audit'):
        raise ValueError('Location/payload audit has not passed')
    checked_indices = set()
    for info in report['manifests'].values():
        if digest(Path(info['parent'])) != info['parent_sha256'] or digest(Path(info['manifest'])) != info['manifest_sha256']:
            raise ValueError('Parent/corrected manifest changed after split audit')
        for view in info['scannet'].values():
            for key in ('source_index', 'corrected_index'):
                path = Path(view[key])
                if str(path) not in checked_indices:
                    if digest(path) != view[key+'_sha256']:
                        raise ValueError('Source/corrected index changed after split audit')
                    checked_indices.add(str(path))
    for path, expected in report['evaluation_split_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError('Evaluation split changed after split audit')
    checks = {}
    for label, relative in (
        ('plane', 'plane/nonzero_geometry.json'),
        ('base', 'curated/nonzero_trim_invalid25_geometry.json'),
        ('high', 'quality_tiers/nonzero_high_geometry.json'),
        ('baseline_v4', 'baseline_v4/nonzero_geometry.json'),
    ):
        manifest = output/'manifests'/relative
        evidence = output/'loader'/f'{label}.json'
        check = json.loads(evidence.read_text())
        expected_names = set(report['manifests'][relative]['scannet'])
        if (check.get('status') != 'PASS' or check.get('manifest_sha256') != digest(manifest)
                or {sample['dataset'] for sample in check.get('samples', [])} != expected_names
                or not all(sample.get('both_modes_identical_preprocessing') for sample in check['samples'])):
            raise ValueError(f'Bound real-loader validation failed/missing: {label}')
        checks[label] = {'path': str(evidence), 'sha256': digest(evidence), 'decoded_samples': check['total_samples']}
    ready = {'status': 'location_split_and_primary_loader_checks_passed',
             'report_sha256': digest(report_path), 'loader_checks': checks,
             'manifest_count': len(report['manifests']),
             'loader_scope': 'full plane/base/high/v4 ScanNet; both loss modes, sampled RGB/GT/masks',
             'all_variant_location_and_payload_checks': True, 'evaluation_unchanged': True,
             'semantic_quality_certified': False, 'requires_fresh_training_for_heldout_claim': True}
    write_json(output/'READY.json', ready)
    return ready


def verify_corrected_pools(output, paths):
    """Fail closed on changed manifest/index/split/evidence before a new run."""
    ready = json.loads((output/'READY.json').read_text())
    if ready.get('status') != 'location_split_and_primary_loader_checks_passed':
        raise ValueError('ScanNet physical-location correction is not loader validated')
    if digest(output/'REPORT.json') != ready['report_sha256']:
        raise ValueError('ScanNet correction report changed')
    report = json.loads((output/'REPORT.json').read_text())
    proven = {str(Path(info['manifest']).resolve()): info for info in report['manifests'].values()}
    checked = set()
    for key, path in paths.items():
        if key == 'zeroshot':
            continue  # Original ZS excludes ScanNet; validate_pools checks shared sources.
        info = proven.get(str(Path(path).resolve()))
        if info is None or digest(Path(path)) != info['manifest_sha256']:
            raise ValueError(f'NZS manifest is not the verified ScanNet correction: {path}')
        for view in info['scannet'].values():
            index = Path(view['corrected_index'])
            if str(index) not in checked and digest(index) != view['corrected_index_sha256']:
                raise ValueError('Corrected ScanNet index changed')
            checked.add(str(index))
    for path, expected in report['evaluation_split_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError('ScanNet evaluation split changed')
    for evidence in ready['loader_checks'].values():
        if digest(Path(evidence['path'])) != evidence['sha256']:
            raise ValueError('ScanNet correction loader evidence changed')
    return ready


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', action='append', default=[], metavar='LABEL.json=PATH',
                   help='Repeat for explicit parent pools; otherwise correct all known NZS versions')
    p.add_argument('--split-root', type=Path, default=DEFAULT_SPLITS)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--finalize', action='store_true', help='Verify the four bound real-loader reports and freeze READY.json')
    a = p.parse_args()
    if a.finalize:
        if a.manifest:
            raise ValueError('--finalize does not rebuild manifests')
        print(json.dumps(finalize_loader_checks(a.output), indent=2))
        return
    manifests = {}
    for value in a.manifest:
        label, path = value.split('=', 1)
        if label in manifests:
            raise ValueError(f'Duplicate manifest label: {label}')
        manifests[label] = Path(path)
    summary = build(manifests or default_manifests(), a.split_root, a.output)
    print(json.dumps({'status': summary['status'], 'manifests': len(summary['manifests']),
                      'heldout_locations': summary['heldout_location_count'], 'report': str(a.output/'REPORT.json')}, indent=2))


if __name__ == '__main__':
    main()
