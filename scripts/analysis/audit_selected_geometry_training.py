#!/usr/bin/env python3
"""Audit selected SQLite rows, not unused members of a parent archive.

Dataset-selection audit only. ScanNet physical-location checks use a second split parser.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.training.asset_lock import digest
from common.training.benchmark_exclusions import exclusion_reason, load_registry, source_identity, validate_scannet_splits
from scripts.dataset_prep.apply_scannet_location_correction import independent_locations


def audit(manifest, split_root):
    manifest, split_root = Path(manifest), Path(split_root)
    registry, split_files = load_registry(split_root)
    blocked, _, _, _ = independent_locations(split_root)
    sources = []
    for entry in json.loads(manifest.read_text())['datasets']:
        if not entry['name'].startswith(('depth_', 'surface_normals_')):
            continue
        if entry.get('type') != 'indexed_prepared_tar':
            raise ValueError(f'Unsupported geometry source type: {entry["name"]}')
        index = Path(entry['index_path'])
        seen, failures, missing_masks = set(), Counter(), 0
        example_failures = []
        with sqlite3.connect(f'file:{index}?mode=ro', uri=True) as conn:
            for (payload,) in conn.execute('select payload from rows order by row_number'):
                row = json.loads(payload)
                source, identifier, scene = source_identity(row)
                if source not in {'hypersim','matterport3d','replica','synscapes','taskonomy',
                                  'nyu','scannet','kitti','diode','vkitti2','eth3d','ibims1','sintel'}:
                    raise ValueError(f'Unverified geometry source identity: {source!r}/{entry["name"]}')
                identity = (source, identifier)
                if identity in seen:
                    raise ValueError(f'Duplicate selected identity: {entry["name"]}/{identity}')
                seen.add(identity)
                # The real loader canonicalizes legacy valid_mask to loss_mask.
                missing_masks += not bool(row.get('loss_mask') or row.get('valid_mask'))
                reason = exclusion_reason(row, registry)
                # Deliberately independent of the shared location-ID helper.
                independent_overlap = source == 'scannet' and scene.rsplit('_', 1)[0] in blocked
                if independent_overlap and reason is None:
                    raise ValueError('Shared and independent ScanNet exclusion checks disagree')
                if reason:
                    failures[reason] += 1
                    if len(example_failures) < 8:
                        example_failures.append({'id': identifier, 'reason': reason})
        if not seen:
            raise ValueError(f'Empty selected geometry source: {entry["name"]}')
        sources.append({'name': entry['name'], 'selected_rows': len(seen),
            'index_path': str(index), 'index_sha256': digest(index),
            'archive': entry['path'], 'missing_mask_rows': missing_masks,
            'overlap_rows': sum(failures.values()), 'overlap_reasons': dict(failures),
            'overlap_examples': example_failures})
    if not sources:
        raise ValueError('No geometry sources in manifest')
    return {'status': 'PASS' if all(not s['overlap_rows'] and not s['missing_mask_rows'] for s in sources) else 'FAIL',
        'scope': 'all selected SQLite geometry rows; independent ScanNet physical-location audit; no GT quality certification',
        'manifest': str(manifest.resolve()), 'manifest_sha256': digest(manifest),
        'split_sha256': {str(p): digest(p) for p in split_files},
        'scannet_split_identity': validate_scannet_splits(split_root), 'sources': sources,
        'requires_original_training_provenance_for_checkpoint_claim': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--split-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report-overlap', action='store_true', help='Report known historical exposure without a PASS exit')
    a = p.parse_args()
    result = audit(a.manifest, a.split_root)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(result['status'], a.output)
    if result['status'] != 'PASS' and not a.report_overlap:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
