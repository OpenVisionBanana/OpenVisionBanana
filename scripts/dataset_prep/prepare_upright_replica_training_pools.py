#!/usr/bin/env python3
"""Substitute an accepted upright Replica candidate in separate curated pools.

This is an additive preparation step, never a change to running training or to
the original manifests. Task and dataset sampling weights remain unchanged.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.training.asset_lock import digest, identity
from scripts.dataset_prep.build_geometry_readiness_manifests import write_json


def replace_replica(payload, replacements):
    result = copy.deepcopy(payload)
    counts = dict(depth=0, surface_normals=0)
    for entry in result['datasets']:
        for task in counts:
            if entry['name'].startswith(task + '_') and 'replica' in entry['name'].split('_'):
                replacement = replacements[task]
                for key in ('type', 'path', 'index_path'):
                    entry[key] = replacement[key]
                entry.pop('offset_base', None)
                counts[task] += 1
    if any(count != 1 for count in counts.values()):
        raise ValueError(f'Expected exactly one Replica source per task: {counts}')
    result['replica_render_contract'] = dict(
        revision='replica-upright-recorded-positions-v1',
        joint_rgb_depth_normal_rebuild=True, max_joint_invalid_fraction=.01,
        world_up_axis='z', rgb_source='mesh_vertex_colours_not_Ptex',
        sampling_weights_unchanged=True, other_sources_unchanged=True,
        training_only=True, original_manifests_unchanged=True,
        warning='Selected candidate; no semantic-noise-free or performance certification',
    )
    return result


def build(curated, upright, output):
    if output.exists():
        raise FileExistsError('Use a new version; existing training pools are immutable')
    for root in (curated, upright):
        for relative in ('READY.json', 'reports/acceptance.json'):
            path = root / relative
            if digest(path) != Path(str(path) + '.sha256').read_text().split()[0]:
                raise ValueError(f'Acceptance provenance changed: {path}')
        if json.loads((root/'reports/acceptance.json').read_text())['status'] != 'PASS':
            raise ValueError('Both source pools must have passed acceptance')
    original = json.loads((upright/'replica_geometry.json').read_text())
    replacements = {
        task: next(e for e in original['datasets'] if e['name'] == task + '_replica_upright')
        for task in ('depth', 'surface_normals')
    }
    parent_manifest = upright/'replica_geometry.json'
    if digest(parent_manifest) != Path(str(parent_manifest)+'.sha256').read_text().split()[0]:
        raise ValueError('Upright manifest checksum changed')
    manifests = []
    assets = [identity(Path(e['index_path'])) for e in replacements.values()]
    for mode in ('zeroshot', 'nonzero'):
        for variant in ('masked', 'unmasked'):
            for scope in ('full', 'geometry'):
                name = f'{mode}_{variant}_{scope}.json'
                source = curated/'manifests'/name
                if digest(source) != Path(str(source)+'.sha256').read_text().split()[0]:
                    raise ValueError(f'Curated manifest checksum changed: {source}')
                payload = replace_replica(json.loads(source.read_text()), replacements)
                payload['replica_render_contract']['accepted_candidate'] = str(upright.resolve())
                target = output/'manifests'/name
                write_json(target, payload)
                assets.append(identity(target)); manifests.append(str(target.resolve()))
    write_json(output/'ASSETS.json', dict(
        schema='upright-replica-pool-dataset-provenance-v1',
        curated_parent=str(curated.resolve()), upright_parent=str(upright.resolve()),
        parent_acceptance_sha256={str(r/'reports/acceptance.json'):digest(r/'reports/acceptance.json')
                                 for r in (curated, upright)},
        new_assets=assets,
        caution='Dataset provenance only; freeze chosen manifest, executing code, model and container before fresh training',
    ))
    write_json(output/'reports/acceptance.json', dict(
        status='PASS', scope='Manifest composition with accepted parent datasets',
        parent_curated_pool=str(curated.resolve()), parent_upright_replica=str(upright.resolve()),
        source_loader_checks='Inherited sampled checks from both accepted parents',
        paired_replica_checks='Inherited all-ID/scene and sampled geometry checks from upright parent',
        exactly_one_replica_source_per_task=True, sampling_weights_unchanged=True,
        other_sources_unchanged=True, semantic_quality_certified=False,
    ))
    write_json(output/'READY.json', dict(
        status='accepted_upright_replica_substituted_in_curated_training_pools',
        manifests=manifests, sampling_weights_unchanged=True,
        unchanged_non_ZS_sources=True, running_training_unchanged=True,
        public_upload='not performed',
        limitations=['Original mesh vertex colours; Ptex texturing not restored',
                     'Unmasked views may still have up to 1% undefined pixels',
                     'NZS-added sources are unchanged, not the earlier NZS base/high filters'],
        before_training='Use a fresh experiment and a new model/runtime/code-specific training asset lock',
    ))
    print(f'Prepared {len(manifests)} additive upright Replica training manifests', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curated', type=Path, required=True)
    parser.add_argument('--upright', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.curated, args.upright, args.output)


if __name__ == '__main__':
    main()
