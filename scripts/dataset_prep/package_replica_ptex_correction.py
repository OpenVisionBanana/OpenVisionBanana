#!/usr/bin/env python3
"""Replace only Replica conditioning RGB in a new, filter-free geometry package.

Never edit the parent package. Other archives are hard-linked (not copied or
rewritten); Replica targets and masks are copied byte-for-byte. READY is issued
only after both working and public-release loaders pass the separate smoke test.
"""
from __future__ import annotations
import argparse
import copy
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.training.asset_lock import digest
from common.indexed_tar import read_reference_bytes
from scripts.analysis.audit_zs_geometry_quality import Reader
from scripts.dataset_prep.export_geometry_training_release import ShardWriter, write

VARIANT = 'half_exposure_neutral_saturation'
SOURCES = ('depth_legacy_replica', 'surface_normals_legacy_replica_camera_facing_v4')


def checked(path):
    path = Path(path)
    if digest(path) != Path(str(path)+'.sha256').read_text().split()[0]:
        raise ValueError(f'Checksum mismatch: {path}')
    return json.loads(path.read_text())


def indexed(entry):
    with sqlite3.connect('file:'+str(entry['index_path'])+'?mode=ro', uri=True) as db:
        if db.execute('pragma integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Corrupt source index')
        rows = [json.loads(p) for (p,) in db.execute('select payload from rows order by row_number')]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source IDs')
    return rows, Reader({**entry, 'offset_base': 0})


def localize_manifests(parent, out):
    for path in (parent.parent/'local_manifests').glob('*_full.json'):
        payload = json.loads(path.read_text())
        for entry in payload['datasets']:
            if entry['name'].startswith(('depth_', 'surface_normals_')):
                for field in ('path', 'index_path'):
                    if entry.get(field):
                        relative = Path(entry[field]).relative_to(parent)
                        entry[field] = str(out/relative)
        output = out.parent/'local_manifests'/path.name
        if output.exists():
            if checked(output) != payload:
                raise ValueError('Existing local manifest differs; refusing to replace it')
        else:
            write(output, payload)


def prepare(args):
    parent = args.parent.resolve()
    ready, package = checked(parent/'READY.json'), checked(parent/'PACKAGE.json')
    if ready['status'] != 'materialized_export_complete' or ready['package_sha256'] != digest(parent/'PACKAGE.json'):
        raise ValueError('Parent is not a complete package')
    qa = checked(args.qa/'QA.json')
    if qa['status'] != 'PASS_RGB_REPLACEMENT_CONTRACT' or not qa['targets_masks_depth_and_selection_unchanged']:
        raise ValueError('Full-frame colour/registration contract has not passed')
    candidate = args.campaign/'ptex_full_candidates'/VARIANT
    manifest = json.loads((candidate/'manifests'/f'ptex_{VARIANT}_full.json').read_text())
    candidates = [next(e for e in manifest['datasets'] if 'replica' in e['name'] and e['name'].startswith(prefix))
                  for prefix in ('depth_', 'surface_normals_')]
    locks = {a['path']: a for a in qa['assets']}
    for entry in candidates:
        proof = locks[entry['path']]
        if digest(Path(entry['path'])) != proof['sha256'] or digest(Path(entry['index_path'])) != proof['index_sha256']:
            raise ValueError('Candidate changed after full-frame validation')
    if args.output.exists():
        raise FileExistsError('Use a new output directory; original releases are never overwritten')
    out = args.output.resolve()
    for folder in ('archives', 'indices', 'manifests', 'reports'):
        (out/folder).mkdir(parents=True, exist_ok=True)
    newpackage = copy.deepcopy(package)
    newpackage['parent_package_sha256'] = digest(parent/'PACKAGE.json')
    newpackage['version'] = 'ovb-materialized-geometry-replica-ptex-v2'
    # Do not inherit old acceptance as evidence for the newly changed RGB.
    newpackage['inherited_non_replica_sample_acceptance'] = newpackage.pop('sample_acceptance', None)
    excluded = {s[k] for name in SOURCES for s in package['sources'][name]['shards'] for k in ('archive', 'index')}
    for folder in ('archives', 'indices'):
        for path in (parent/folder).iterdir():
            if path.name.removesuffix('.sha256') in excluded:
                continue
            if not path.is_file():
                raise ValueError(f'Unexpected asset: {path}')
            os.link(path, out/folder/path.name)
    # Preserve the portable installer and correctness patch; deliberately do not
    # copy old experiment/training overrides into this RGB-only data correction.
    shutil.copyfile(parent/'use_geometry_training_release.py', out/'use_geometry_training_release.py')
    for folder in ('compatibility',):
        if (parent/folder).exists():
            shutil.copytree(parent/folder, out/folder)
    for name, entry in zip(SOURCES, candidates):
        report = package['sources'][name]
        if len(report['shards']) != 1:
            raise ValueError('Review correction for a multi-shard Replica parent')
        oldshard = report['shards'][0]
        oldentry = dict(path=str(parent/'archives'/oldshard['archive']), index_path=str(parent/'indices'/oldshard['index']))
        if digest(Path(oldentry['path'])) != oldshard['archive_sha256'] or digest(Path(oldentry['index_path'])) != oldshard['index_sha256']:
            raise ValueError('Parent Replica asset changed')
        oldrows, oldreader = indexed(oldentry)
        newrows, newreader = indexed(entry)
        byid = {r['id']: r for r in newrows}
        if set(byid) != {r['id'] for r in oldrows} or len(oldrows) != qa['frames']:
            raise ValueError('Selection changed')
        writer = ShardWriter(out/'archives', name, 0)
        for ordinal, old in enumerate(oldrows):
            candidate_row = byid[old['id']]
            if candidate_row['task'] != old['task'] or candidate_row['text'] != old['text']:
                raise ValueError('Task or prompt changed')
            assets = []
            for field, mode in (('conditioning_image', 'RGB'), ('image', 'RGB'), ('loss_mask', 'L')):
                oldblob = read_reference_bytes(oldreader.ref(old, old[field]))
                newblob = read_reference_bytes(newreader.ref(candidate_row, candidate_row[field]))
                with Image.open(io.BytesIO(newblob)) as im:
                    newpixels = np.asarray(im.convert(mode))
                with Image.open(io.BytesIO(oldblob)) as im:
                    oldpixels = np.asarray(im.convert(mode))
                if newpixels.shape != oldpixels.shape:
                    raise ValueError(f'Unpaired dimensions: {old["id"]}')
                if field != 'conditioning_image' and not np.array_equal(oldpixels, newpixels):
                    raise ValueError(f'Changed {field}: {old["id"]}')
                if field == 'conditioning_image' and np.array_equal(oldpixels, newpixels):
                    raise ValueError('Vertex colours were not replaced')
                # Preserve exact published target/mask bytes, not just pixels.
                assets.append((old[field], newblob if field == 'conditioning_image' else oldblob))
            row = copy.deepcopy(old)
            row['metadata']['rgb_source'] = 'official_replica_ptex_fixed_pose'
            row['metadata']['rgb_render_variant'] = VARIANT
            row['metadata']['rgb_correction_version'] = 'replica-ptex-v2'
            row['metadata']['ptex_sdk_settings'] = candidate_row['metadata']['ptex_sdk_settings']
            provenance = dict(id=old['id'], ordinal=ordinal, original_row=candidate_row,
                              parent_archive=oldshard['archive'], parent_archive_sha256=oldshard['archive_sha256'],
                              correction='Conditioning RGB only; target/mask bytes and row order preserved')
            writer.row(row, assets, provenance)
            if (ordinal+1) % 500 == 0:
                print(f'PTEX EXPORT {name}: {ordinal+1}/{len(oldrows)}', flush=True)
        shard = writer.finish(out/'indices')
        newpackage['sources'][name] = dict(source=name, rows=len(oldrows), shards=[shard],
            sample_only=False, no_runtime_crops_or_filters=True,
            target_and_mask_bytes_unchanged=True, row_order_and_prompts_unchanged=True,
            rgb_source='official_replica_ptex_fixed_pose', rgb_render_variant=VARIANT)
    for path in (parent/'manifests').glob('*_geometry_*.json'):
        payload = checked(path)
        # Names, paths, weights and sampling settings stay identical; files in
        # this NEW package carry the corrected RGB and their own checksums.
        write(out/'manifests'/path.name, payload)
    localize_manifests(parent, out)
    correction = dict(status='PASS_RGB_REPLACEMENT_CONTRACT', scenes=qa['scenes'], frames_per_task=qa['frames'],
        variant=VARIANT, sdk_settings='exposure=0.0125, gamma=1.6969, saturation=1.0',
        qa_sha256=digest(args.qa/'QA.json'), parent_package_sha256=digest(parent/'PACKAGE.json'),
        target_mask_bytes_row_order_prompts_weights_unchanged=True,
        inherited_non_replica_archives='Exact parent files, hard-linked; verify SHA-256 with the installer before training',
        remaining_colour_limits=qa['statistics'][VARIANT], training_gain_proven=False,
        nice_slam_identical=False, uploaded=False)
    newpackage['rgb_correction'] = correction
    write(out/'PACKAGE.json', newpackage)
    write(out/'reports/RGB_CORRECTION.json', correction)
    write(out/'reports/full_frame_qa.json', qa)
    print(f'Prepared {out}; loader acceptance required before READY', flush=True)


def finalize(args):
    out = args.output.resolve()
    package = checked(out/'PACKAGE.json')
    reports = [checked(p) for p in sorted((out/'reports').glob('loader_*.json'))]
    if len(reports) != 2 or {r['repo_role'] for r in reports} != {'working', 'public_release'}:
        raise ValueError('Require both working and public-release loader reports')
    if any(r['status'] != 'PASS' or r['package_sha256'] != digest(out/'PACKAGE.json') for r in reports):
        raise ValueError('Missing or stale actual-loader acceptance')
    if (out/'READY.json').exists():
        raise FileExistsError('Already finalized')
    localize_manifests(args.parent.resolve(), out)
    write(out/'VALIDATION.json', dict(status='PASS', rgb=checked(out/'reports/RGB_CORRECTION.json'),
          loader_reports=reports, gpu_training='Not run: RGB-only correction; training gains remain unproven'))
    write(out/'READY.json', dict(status='materialized_export_complete', rows=package['rows'],
          package_sha256=digest(out/'PACKAGE.json'), validation_sha256=digest(out/'VALIDATION.json'),
          uploaded=False, defaults='Keep release_colorexp; toggle only loss_mask_policy for the intended comparison'))
    print('PASS: versioned Ptex geometry package ready; nothing uploaded or restarted')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'finalize'])
    p.add_argument('--parent', type=Path, default=Path('data/geometry_training_publication_20261004/upload'))
    p.add_argument('--campaign', type=Path, default=Path('data/zs_quality_candidates_20261003'))
    p.add_argument('--qa', type=Path, default=Path('data/replica_ptex_fix_20261004/qa'))
    p.add_argument('--output', type=Path, default=Path('data/replica_ptex_fix_20261004/upload'))
    a = p.parse_args()
    (prepare if a.action == 'prepare' else finalize)(a)
