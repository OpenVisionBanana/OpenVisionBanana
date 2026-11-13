#!/usr/bin/env python3
"""Conservative historical exposure classification from original, pinned assets.

Input catalog: [{label, checkpoint, original_manifest?, original_manifest_sha256?,
original_index_sha256?: {absolute_index_path: sha256}}]. A filename or present-day
replacement manifest alone does not verify which rows a checkpoint saw.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.training.asset_lock import digest
from scripts.analysis.audit_selected_geometry_training import audit


def classify(item, splits):
    result = {'label': item['label'], 'checkpoint': item['checkpoint'], 'status': 'unverified'}
    try:
        checkpoint = Path(item['checkpoint'])
        if not checkpoint.exists():
            raise ValueError('Checkpoint unavailable locally')
        manifest = Path(item['original_manifest'])
        if digest(manifest) != item['original_manifest_sha256']:
            raise ValueError('Original manifest identity mismatch')
        evidence = item.get('original_index_sha256') or {}
        entries = json.loads(manifest.read_text())['datasets']
        for entry in entries:
            if not entry['name'].startswith(('depth_', 'surface_normals_')):
                continue
            index = Path(entry['index_path'])
            if evidence.get(str(index.resolve())) != digest(index):
                raise ValueError(f'Original selected-row identity missing/mismatched: {index}')
        # The catalog must point to a training-time artifact linking the actual
        # checkpoint/run to these manifest and index hashes, not an assertion.
        link = Path(item['training_time_evidence'])
        if digest(link) != item['training_time_evidence_sha256']:
            raise ValueError('Training-time evidence identity mismatch')
        provenance = json.loads(link.read_text())
        if provenance.get('manifest_sha256') != item['original_manifest_sha256'] or provenance.get('index_sha256') != evidence:
            raise ValueError('Training-time evidence does not bind the original selection')
        if Path(provenance.get('checkpoint', '')).resolve() != checkpoint.resolve():
            raise ValueError('Training-time evidence does not bind this checkpoint')
        review = audit(manifest, splits)
        overlaps = sum(s['overlap_rows'] for s in review['sources'])
        if not overlaps and review['status'] != 'PASS':
            raise ValueError('Selected-row audit failed; cannot certify held-out training')
        result.update(status='verified_exposed' if overlaps else 'verified_heldout',
                      overlap_rows=overlaps, audit=review, training_time_evidence_sha256=digest(link))
    except (KeyError, ValueError, OSError) as error:
        result['reason'] = str(error)
    return result


def inventory_model(model, staged_model_root):
    """Discover optional training metadata; missing or unreadable evidence stays unverified."""
    candidates = [Path(staged_model_root)/model['label']/'transformer']
    if model.get('local_source') not in (None, '-', ''):
        candidates.insert(0, Path(model['local_source']))
    errors=[]
    checkpoint=candidates[0]
    for candidate in candidates:
        try:
            if candidate.exists():
                checkpoint=candidate;break
        except OSError as error:
            errors.append({'path':str(candidate),'reason':str(error)})
    item = {'label':model['label'],'checkpoint':str(checkpoint),
            'original_training_metadata_files':[], 'discovery_errors':errors}
    for folder in (checkpoint, checkpoint.parent, checkpoint.parent.parent):
        for filename in ('training_args.json','runtime_provenance.json','training_asset_lock.json'):
            path=folder/filename
            try:
                if path.is_file():
                    item['original_training_metadata_files'].append({'path':str(path),'sha256':digest(path)})
            except OSError as error:
                errors.append({'path':str(path),'reason':str(error)})
    return item


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sources = p.add_mutually_exclusive_group(required=True)
    sources.add_argument('--catalog', type=Path)
    sources.add_argument('--models-tsv', type=Path, help='Inventory local final weights; missing original training evidence stays unverified')
    p.add_argument('--staged-model-root', type=Path, default=Path('data/ovb_final_models_20260920'))
    p.add_argument('--split-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.catalog:
        items = json.loads(a.catalog.read_text())
    else:
        contents = a.models_tsv.read_text()
        if '\t' not in contents: contents = contents.replace('\\t','\t')
        items = [inventory_model(model,a.staged_model_root)
                 for model in csv.DictReader(io.StringIO(contents), delimiter='\t')]
    if not items or len({i['label'] for i in items}) != len(items):
        raise ValueError('Empty or duplicate checkpoint catalog')
    reports = [dict(classify(i, a.split_root), metadata_files=i.get('original_training_metadata_files', []),
                    discovery_errors=i.get('discovery_errors', [])) for i in items]
    output = {'catalog_sha256': digest(a.catalog or a.models_tsv), 'checkpoints': reports,
              'warning': 'Unverified is not clean. No inference from replacement training pools or evaluation scores.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(a.output)


if __name__ == '__main__':
    main()
