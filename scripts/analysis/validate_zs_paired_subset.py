#!/usr/bin/env python3
"""Fail-closed acceptance of paired training-only ZS quality index views."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.training.asset_lock import digest, identity
from common.training.benchmark_exclusions import exclusion_reason, load_registry, domain, BENCHMARK_DOMAINS
from scripts.analysis.audit_zs_geometry_quality import checked_plan, is_normal
from scripts.dataset_prep.curate_nzs_geometry import readonly
from scripts.dataset_prep.build_geometry_readiness_manifests import write_json, task_group
from PIL import Image, ImageDraw
from scripts.analysis.audit_zs_geometry_quality import Reader


def rows_at(path):
    with readonly(Path(path)) as db:
        meta = dict(db.execute('select key,value from metadata'))
        rows = [json.loads(p) for p, in db.execute('select payload from rows order by row_number')]
    if int(meta['num_rows']) != len(rows): raise ValueError('Index row count metadata disagrees')
    return meta, {r['id']:r for r in rows}


def validate(output, split_root):
    plan = json.loads((output/'plan.json').read_text())
    audit = Path(plan['audit']); audit_plan = checked_plan(audit)
    if digest(audit/'plan.json') != plan['audit_plan_sha256']: raise ValueError('Audit provenance changed')
    prepared = json.loads((output/'PREPARED.json').read_text())
    registry, split_files = load_registry(split_root)
    summary = json.loads((output/'reports/selection_summary.json').read_text())
    source_assets = json.loads(Path(audit_plan['source_manifest']+'.lock.json').read_text())
    if not source_assets.get('content_verified'): raise ValueError('Parent asset content verification absent')
    if source_assets['manifest_sha256'] != audit_plan['source_manifest_sha256']:
        raise ValueError('Parent asset lock manifest mismatch')
    # Preserve the parent's content proof; reject any later stat mutation.
    for asset in source_assets['assets']:
        path = Path(asset['path']); stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (asset['size'], asset['mtime_ns']):
            raise ValueError(f'Parent training asset changed: {path}')
    validated = []; new_assets = {}
    for variant in prepared['variants']:
        loader = json.loads((output/'reports'/f'{variant}_loader.json').read_text())
        if loader.get('status') != 'PASS' or not loader.get('total_samples'):
            raise ValueError('Production loader acceptance missing or empty')
        for dataset in summary['domains']:
            pair = [e for e in plan['sources'] if e['dataset'] == dataset]
            pair.sort(key=is_normal)
            indexed = []
            for entry in pair:
                path = output/'indices'/f'{entry["name"]}.{variant}.sqlite'
                meta, rows = rows_at(path)
                if meta['source_index_sha256'] != entry['source_index_sha256']:
                    raise ValueError('Filtered index source identity mismatch')
                if digest(Path(entry['index_path'])) != entry['source_index_sha256']:
                    raise ValueError('Original source index changed')
                for row in rows.values():
                    if exclusion_reason(row, registry) or domain(row) in BENCHMARK_DOMAINS:
                        raise ValueError('Known geometry benchmark domain/frame entered the ZS subset')
                indexed.append(rows)
                new_assets[str(path.resolve())] = identity(path)
            if indexed[0].keys() != indexed[1].keys(): raise ValueError('Curated depth/normal IDs differ')
            for key in indexed[0]:
                a, b = indexed[0][key], indexed[1][key]
                if a['training_crop_box'] != b['training_crop_box']:
                    raise ValueError('Curated depth/normal crop boxes differ')
                for row in (a, b):
                    c = row['metadata']['training_curation']
                    if c['contract'] != plan['contract'] or c['variant'] != variant:
                        raise ValueError('Unknown paired quality provenance')
            decision = summary['variants'][variant][dataset]
            if len(indexed[0]) != decision['kept']: raise ValueError('Selection report count differs from index')
            validated.append(dict(dataset=dataset, variant=variant, matched_frames=len(indexed[0]),
                                  max_invalid_kept=decision['max_invalid_kept']))
            if dataset == 'replica':
                # One control frame per scene, taken from the actual retained IDs.
                seen = set(); chosen = []
                for key in sorted(indexed[0]):
                    scene = indexed[0][key]['metadata']['scene']
                    if scene not in seen: seen.add(scene); chosen.append(key)
                readers = [Reader(e) for e in pair]
                canvas = Image.new('RGB', (768, len(chosen)*210), 'white'); draw = ImageDraw.Draw(canvas)
                for i, key in enumerate(chosen):
                    dep, nor = indexed[0][key], indexed[1][key]
                    box = dep['training_crop_box']
                    images = [readers[0].image(dep, dep['conditioning_image']),
                              readers[0].image(dep, dep['image']), readers[1].image(nor, nor['image'])]
                    for j, image in enumerate(images):
                        image = image.crop(box); image.thumbnail((250, 180)); canvas.paste(image, (j*256, i*210))
                    draw.text((2, i*210+185), key+' | RGB / depth / corrected normals', fill='black')
                canvas.save(output/'reports'/f'replica_{variant}_retained.jpg', quality=92)
    for manifest in map(Path, prepared['manifests']):
        if digest(manifest) != Path(str(manifest)+'.sha256').read_text().split()[0]:
            raise ValueError('Manifest checksum differs')
        payload = json.loads(manifest.read_text())
        if 'geometry' not in manifest.name:
            mode = manifest.name.split('_')[0]
            original = Path(audit_plan['source_manifest']) if mode == 'zeroshot' else Path(
                'data/geometry_training_readiness_20261002/plane/nonzero_full.json')
            before = json.loads(original.read_text())
            weights = {e['name']:e['weight'] for e in before['datasets']}
            if any(abs(e['weight']-weights[e['name']]) > 1e-12 for e in payload['datasets']):
                raise ValueError('Dataset sampling weights changed')
        new_assets[str(manifest.resolve())] = identity(manifest)
    write_json(output/'ASSETS.json', dict(schema='paired-subset-content-provenance-v1',
        parent_asset_lock=str(Path(audit_plan['source_manifest']+'.lock.json').resolve()),
        parent_asset_lock_sha256=digest(Path(audit_plan['source_manifest']+'.lock.json')),
        inherited_assets_stat_validated=True, new_assets=list(new_assets.values()),
        split_sha256={str(p):digest(p) for p in split_files},
        caution='Dataset provenance, not a replacement for a model/runtime/code-specific production training asset lock'))
    write_json(output/'reports/acceptance.json', dict(status='PASS', subsets=validated,
        all_retained_IDs_and_crop_boxes_checked=True, loader_sampled=True,
        both_loss_modes_checked=True, known_benchmark_domains_excluded=True,
        scene_diversity_checked=True, semantic_quality_certified=False))
    write_json(output/'READY.json', dict(status='paired_subset_and_loader_checks_passed',
        domains=summary['domains'], variants=prepared['variants'],
        original_data_and_running_jobs_unchanged=True, public_upload='not performed',
        warning='Filtered candidates, not guaranteed noise-free or proven to improve trained models',
        before_training='Freeze the chosen manifest, current execution code, base model and container with freeze_training_assets.py'))
    print('PASS paired IDs/crops, source identities, benchmark exclusions, weights and masked/unmasked loader checks', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split-root', type=Path, default=Path('data/marigold_v2_depth_eval_20260920/data_split'))
    a = p.parse_args(); validate(a.output, a.split_root)


if __name__ == '__main__': main()
