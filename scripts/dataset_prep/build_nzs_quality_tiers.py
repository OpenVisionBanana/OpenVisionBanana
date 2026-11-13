#!/usr/bin/env python3
"""Name the existing NZS subset base and build a stricter, additive high view."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.indexed_tar import validate_archive_fingerprint
from common.training.asset_lock import digest
from scripts.dataset_prep.build_geometry_readiness_manifests import normalize, task_group, write_json
from scripts.dataset_prep.curate_nzs_geometry import readonly, write_view

HIGH_POLICY = {
    'max_remaining_invalid_fraction': 0.10,
    'min_retained_area': 0.75,
    'reject_edge_warnings': True,
    'rejected_review_flags': [
        'rough_normal_field_at_review_resolution', 'nonunit_encoded_normal_vectors',
        'near_constant_rgb', 'near_uniform_black_or_white_rgb'],
    'generic_low_texture_or_blur_is_not_a_rejection': True,
    'scope': 'added NZS geometry only; shared ZS and other tasks unchanged',
    'quality_certification': False,
}


def high_rejection_reasons(result):
    """Coverage plus conservative quality proxies, not semantic GT certification."""
    reasons = []
    if result.get('hard_reasons'):
        reasons.append('hard_quality_failure')
    invalid, area = result.get('invalid_after'), result.get('retained_area')
    if invalid is None or area is None or not all(math.isfinite(x) for x in (invalid, area)):
        return reasons + ['missing_or_nonfinite_coverage']
    if not 0 <= invalid <= 1 or not 0 < area <= 1:
        return reasons + ['out_of_range_coverage']
    if invalid > HIGH_POLICY['max_remaining_invalid_fraction'] + 1e-12:
        reasons.append('remaining_invalid_coverage')
    if area + 1e-12 < HIGH_POLICY['min_retained_area']:
        reasons.append('retained_area_below_75_percent')
    if result.get('edge_warnings'):
        reasons.append('edge_crop_warning')
    reasons.extend(f'quality_flag:{flag}' for flag in HIGH_POLICY['rejected_review_flags']
                   if flag in result.get('review_flags', []))
    return reasons


def checked_json(path):
    expected = path.with_suffix(path.suffix + '.sha256').read_text().split()[0]
    if digest(path) != expected:
        raise ValueError(f'Checksum mismatch: {path}')
    return json.loads(path.read_text())


def build(parent, campaign):
    parent, campaign = parent.resolve(), campaign.resolve()
    if parent == campaign or parent in campaign.parents:
        raise ValueError('Use a separate output campaign, not the parent or its subtree')
    ready = checked_json(parent / 'READY.json')
    if ready['status'] != 'subset_and_loader_checks_passed':
        raise ValueError('Parent subset has not passed loader acceptance')
    plan = checked_json(parent / 'plan.json')
    bases = {scope: checked_json(parent / 'manifests' / f'nonzero_trim_invalid25_{scope}.json')
             for scope in ('full', 'geometry')}
    for payload in bases.values():
        contract = payload['curation_contract']
        if contract['variant'] != 'trim_invalid25' or contract['plan_sha256'] != digest(parent / 'plan.json'):
            raise ValueError('Expected the completed 25%-invalid base with matching provenance')
    for path_key, hash_key in (('source_full_manifest', 'source_full_sha256'),
                               ('source_zs_manifest', 'source_zs_sha256')):
        if digest(Path(plan[path_key])) != plan[hash_key]:
            raise ValueError('Parent training source manifest changed')
    campaign.mkdir(parents=True, exist_ok=True)
    for directory in ('indices', 'manifests', 'reports'):
        (campaign / directory).mkdir(exist_ok=True)
    write_json(campaign / 'plan.json', {
        **plan, 'parent_campaign': str(parent), 'parent_plan_sha256': digest(parent / 'plan.json'),
        'parent_ready_sha256': digest(parent / 'READY.json'), 'quality_tiers': HIGH_POLICY,
        'base_alias': 'trim_invalid25; byte-equivalent manifest contents and identical indices',
    })
    entries = {e['name']: e for e in bases['full']['datasets']}
    added = {e['name'] for e in plan['sources']}
    selected = [copy.deepcopy(e) for e in bases['full']['datasets'] if e['name'] not in added]
    summaries, decisions, input_hashes = {}, {}, {}
    for original in plan['sources']:
        name = original['name']
        if digest(Path(original['index_path'])) != original['source_index_sha256']:
            raise ValueError(f'Original index changed: {name}')
        scan_path = parent / 'scans' / f'{name}.sqlite'
        input_hashes[name] = {'scan_sha256': digest(scan_path),
                              'source_index_sha256': original['source_index_sha256']}
        with readonly(scan_path) as scan:
            metadata = dict(scan.execute('select key,value from metadata'))
            identity = json.dumps({'entry': original, 'edge_limits': plan['edge_limits'], 'limit': 0}, sort_keys=True)
            if (metadata.get('complete') != 'true' or metadata.get('full_source') != 'true'
                    or metadata.get('identity') != identity):
                raise ValueError(f'Incomplete or mismatched scan: {name}')
            records = {n: (json.loads(r), json.loads(p)) for n, r, p in scan.execute('select * from records')}
        base = entries.get(name)
        keep, rejected, reasons = [], [], Counter()
        base_count = 0
        if base:
            checked_index = Path(base['index_path'])
            expected = checked_index.with_suffix(checked_index.suffix + '.sha256').read_text().split()[0]
            if digest(checked_index) != expected:
                raise ValueError(f'Base index checksum mismatch: {name}')
            input_hashes[name]['base_index_sha256'] = expected
            with readonly(checked_index) as index:
                archive_metadata = dict(index.execute('select key,value from metadata'))
                validate_archive_fingerprint(Path(base['path']), json.loads(archive_metadata['archive_fingerprint']))
                for ordinal, payload in index.execute('select row_number,payload from rows order by row_number'):
                    base_count += 1
                    row = json.loads(payload)
                    result, raw = records[ordinal]
                    # Verify the base view still represents exactly the scanned row,
                    # with only the documented paired crop and curation annotation.
                    uncropped = copy.deepcopy(row)
                    uncropped.pop('training_crop_box', None)
                    uncropped.pop('training_crop_contract', None)
                    uncropped.get('metadata', {}).pop('training_curation', None)
                    if not uncropped.get('metadata') and 'metadata' not in raw:
                        uncropped.pop('metadata', None)
                    if uncropped != raw or row.get('training_crop_box') != result.get('crop_box'):
                        raise ValueError(f'Base row does not match full scan: {name}/{ordinal}')
                    excluded = high_rejection_reasons(result)
                    if excluded:
                        reasons.update(excluded)
                        rejected.append({'row_number': ordinal, 'id': row.get('id'), 'reasons': excluded})
                    else:
                        row['metadata']['training_curation'].update(variant='high', parent_variant='trim_invalid25')
                        keep.append((ordinal, json.dumps(row, sort_keys=True)))
            if keep:
                selected.append(write_view(base, campaign / 'indices' / f'{name}.high.sqlite', keep, 'high'))
        summaries[name] = {'original': len(records), 'base': base_count, 'high': len(keep),
                           'excluded_from_base': base_count - len(keep), 'reasons_nonexclusive': dict(reasons)}
        decisions[name] = rejected
        print(f'PASS {name}: base={base_count} high={len(keep)}', flush=True)
    masses = Counter()
    for entry in bases['full']['datasets']:
        masses[task_group(entry)] += entry['weight']
    if any(not any(task_group(e) == group for e in selected) for group in masses):
        raise ValueError('High tier would remove an entire task')
    for scope in ('full', 'geometry'):
        # Base is an alias, not another round of filtering or reweighting.
        write_json(campaign / 'manifests' / f'nonzero_base_{scope}.json', bases[scope])
        scoped = [e for e in selected if scope == 'full' or task_group(e) != 'other']
        contract = {**bases[scope]['curation_contract'], 'variant': 'high',
                    'plan_sha256': digest(campaign / 'plan.json'), **HIGH_POLICY,
                    'parent_base_sha256': digest(parent / 'manifests' / f'nonzero_trim_invalid25_{scope}.json'),
                    'interior_holes': 'unchanged; still contribute to loss when masks are disabled',
                    'same_paired_crops_as_base': True,
                    'review_screened_may_remove_legitimate_low_texture_scenes': True}
        write_json(campaign / 'manifests' / f'nonzero_high_{scope}.json', {
            **bases[scope], 'datasets': normalize(scoped, masses), 'curation_contract': contract})
    totals = {key: sum(s[key] for s in summaries.values())
              for key in ('original', 'base', 'high', 'excluded_from_base')}
    write_json(campaign / 'reports' / 'selection_summary.json', {
        'sources': summaries, 'totals_added_geometry': totals, 'policy': HIGH_POLICY,
        'input_hashes': input_hashes, 'high_is_subset_of_base': True,
        'counts_are_task_entries_not_unique_rgb_images': True,
        'limitations': ['Quality signals are proxies, not proof that GT is wrong.',
                        'VKITTI2 normal coverage is substantially reduced.',
                        'Shared ZS geometry is unchanged and may contain invalid regions.',
                        'No model-quality or training-performance claim has been tested.']})
    write_json(campaign / 'reports' / 'excluded_from_base.json', decisions)
    write_json(campaign / 'PREPARED.json', {
        'status': 'indices_and_manifests_prepared', 'variants': ['base', 'high'],
        'training_loader_acceptance': 'must pass separately'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    build(args.parent, args.campaign)


if __name__ == '__main__':
    main()
