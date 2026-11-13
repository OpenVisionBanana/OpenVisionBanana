#!/usr/bin/env python3
"""Materialize accepted geometry selections into portable, filter-free prepared TARs.

This is an export, not another curation pass. It preserves every accepted row,
prompt and mask, physically applies its existing paired crop, and transcodes
legacy normal storage into the declared Vision Banana representation once.
Original archives, indices and evaluation data are never modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import difflib
import hashlib
import io
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.indexed_tar import INDEX_SCHEMA_VERSION, IndexedTarReference, open_reference_image
from common.training.asset_lock import digest
from scripts.dataset_prep.build_geometry_base_clean import DEFAULT as INPUT, checked, readonly, verify

DEFAULT = Path('data/geometry_training_portable_20261004')
PINNED_REPO = Path('.')
VERSION = 'ovb-materialized-geometry-v1'
TRAINER = 'train_dreambooth_partial_flux2_klein_img2img.py'


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)
    Path(str(path)+'.sha256').write_text(f'{digest(path)}  {path.name}\n')


def preprocessing_patch(text):
    """A non-filtering upstream compatibility fix; never edit its checkout."""
    old = '''        loss_mask = (
            self._load_row_loss_mask(row)
            if use_loss_mask and row.loss_mask is not None
            else None
        )'''
    new = '''        # Validity guides geometry resizing even when loss weighting is disabled.
        preprocessing_mask = (
            self._load_row_loss_mask(row) if row.loss_mask is not None else None
        )
        loss_mask = preprocessing_mask if use_loss_mask else None'''
    substitutions = [(old, new),
        ('instance_image, size, loss_mask\n            )', 'instance_image, size, preprocessing_mask\n            )'),
        ('self.args.normal_target_encoding,\n                loss_mask,\n            )',
         'self.args.normal_target_encoding,\n                preprocessing_mask,\n            )')]
    for before, after in substitutions:
        if text.count(before) != 1:
            raise ValueError('Pinned upstream preprocessing changed; review compatibility patch')
        text = text.replace(before, after)
    return text


def initialize(root, source_root, repo, rows_per_shard=5000):
    if (root/'PLAN.json').exists():
        raise FileExistsError('Export already initialized; use check/status/submit, not initialize')
    paths = {name: source_root/'manifests'/f'{name}_full.json'
             for name in ('zeroshot', 'base', 'added_only')}
    verify(source_root, paths)
    templates = {name: checked(path) for name, path in paths.items()}
    entries = {e['name']: e for e in templates['base']['datasets']
               if e['name'].startswith(('depth_', 'surface_normals_'))}
    for folder in ('archives', 'indices', 'reports', 'logs', 'compatibility', 'source_indices'):
        (root/folder).mkdir(parents=True, exist_ok=True)
    sources = []
    for name, entry in sorted(entries.items()):
        e = copy.deepcopy(entry)
        frozen = root/'source_indices'/f'{name}.sqlite'
        # Freeze selection independently of mutable source paths, not image data.
        __import__('shutil').copyfile(e['index_path'], frozen)
        with readonly(frozen) as db:
            count = db.execute('select count(*) from rows').fetchone()[0]
        sources.append(dict(entry=e, frozen_index=str(frozen), index_sha256=digest(frozen), rows=count))
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'origin/release'], text=True).strip()
    text = subprocess.check_output(['git', '-C', str(repo), 'show', f'{revision}:{TRAINER}'], text=True)
    patched = preprocessing_patch(text)
    patch = ''.join(difflib.unified_diff(text.splitlines(True), patched.splitlines(True),
                                        fromfile='a/'+TRAINER, tofile='b/'+TRAINER))
    (root/'compatibility/unmasked_validity.patch').write_text(patch)
    with (root/'compatibility/upstream_release.tar').open('wb') as out:
        subprocess.run(['git', '-C', str(repo), 'archive', revision], stdout=out, check=True)
    plan = dict(version=VERSION, source_root=str(source_root),
        accepted_ready_sha256=digest(source_root/'READY.json'), templates=templates,
        upstream_revision=revision, rows_per_shard=rows_per_shard, max_shard_bytes=10*1024**3,
        sources=sources, source_rows=sum(s['rows'] for s in sources),
        source_archive_fingerprints={e['name']: json.loads(readonly_fingerprint(e['index_path'])) for e in entries.values()},
        upstream_bundle_sha256=digest(root/'compatibility/upstream_release.tar'),
        compatibility_patch_sha256=digest(root/'compatibility/unmasked_validity.patch'),
        selection='Accepted corrected ZS + base-clean NZS; no additional filtering',
        normals='Canonical physical camera frame; Vision Banana x-left-red RGB storage',
        depth='Existing quantized Vision Banana metric-depth RGB; not raw floating-point depth',
        scope='Geometry only. Other-task data are NOT included.',
        public_upload_authorized=False, dataset_redistribution_terms_reviewed=False)
    write(root/'PLAN.json', plan)
    print(json.dumps({'sources':len(sources), 'rows':plan['source_rows'], 'revision':revision}, indent=2))


def readonly_fingerprint(index):
    with readonly(index) as db:
        return db.execute("select value from metadata where key='archive_fingerprint'").fetchone()[0]


def materialize(item, entry, members):
    import numpy as np
    from PIL import Image, ImageOps
    from common.training.geometry_curation import source_crop_from_row, cropped_size
    from common.surface_normal_encoding import (LEGACY_DIRECT_NORMAL_ENCODING,
        VISION_BANANA_NORMAL_ENCODING, decode_surface_normals_rgb, encode_surface_normals_rgb)
    ordinal, payload = item
    raw = json.loads(payload)
    metadata = raw.get('metadata') or {}
    def value(key): return raw.get(key) or metadata.get(key)
    version = value('geometry_version')
    if version not in {'geometry_canonical_v2','geometry_vision_banana_v3','geometry_camera_facing_v4'}:
        raise ValueError(f'Unverified geometry version: {version}')
    box = source_crop_from_row(raw)
    def image(field, mode):
        member = raw.get(field) or (raw.get('valid_mask') if field == 'loss_mask' else None)
        if member not in members:
            parent = str(PurePosixPath(raw.get('_metadata_member', 'metadata.jsonl')).parent)
            member = str(PurePosixPath(parent)/str(member))
        offset, size = members[member]
        with open_reference_image(IndexedTarReference(entry['path'], member, offset, size)) as im:
            im.load()
            result = ImageOps.exif_transpose(im).convert(mode)
        cropped_size(result.size, box)
        return result
    rgb, target, mask = image('conditioning_image','RGB'), image('image','RGB'), image('loss_mask','L')
    native_size = rgb.size
    if target.size != native_size or mask.size != native_size:
        raise ValueError('Misaligned RGB / target / mask')
    if box is not None:
        rgb, target, mask = rgb.crop(box), target.crop(box), mask.crop(box)
    valid = np.asarray(mask)
    if not np.isin(valid, [0,255]).all() or not valid.any():
        raise ValueError('Missing, empty or nonbinary validity mask')
    normals = raw['task'] == 'surface_normal_estimation'
    source_encoding = value('normal_encoding')
    if normals:
        source_encoding = source_encoding if source_encoding in {LEGACY_DIRECT_NORMAL_ENCODING,VISION_BANANA_NORMAL_ENCODING} else (
            VISION_BANANA_NORMAL_ENCODING if version == 'geometry_vision_banana_v3' else LEGACY_DIRECT_NORMAL_ENCODING)
        # This is the same one-time transcode as our loader. Already-canonical
        # VB pixels are copied without a second normalization/quantization.
        if source_encoding != VISION_BANANA_NORMAL_ENCODING:
            physical, encoded_valid = decode_surface_normals_rgb(np.asarray(target), encoding=source_encoding)
            target = Image.fromarray(encode_surface_normals_rgb(physical, encoded_valid, encoding=VISION_BANANA_NORMAL_ENCODING))
        _, encoded_valid = decode_surface_normals_rgb(np.asarray(target), encoding=VISION_BANANA_NORMAL_ENCODING)
        if ((valid > 0) & ~encoded_valid).any():
            raise ValueError('Undefined normal marked valid')
    elif value('canonical_depth_quantity') != 'projective_camera_z_meters':
        raise ValueError('Depth is not verified projective camera-z')
    fields = {'conditioning_image':f'conditioning_images/{ordinal:08d}.png',
              'image':f'target_images/{ordinal:08d}.png', 'loss_mask':f'loss_masks/{ordinal:08d}.png'}
    assets = []
    for field, im in zip(fields, (rgb, target, mask)):
        out = io.BytesIO(); im.save(out, format='PNG', compress_level=3)
        assets.append((fields[field], out.getvalue()))
    row = {k:raw[k] for k in ('id','task','source_dataset','text','scene','referential_expression') if k in raw}
    row.update(fields, preprocess_target='bilinear', geometry_version='geometry_vision_banana_v3',
               canonical_depth_quantity=value('canonical_depth_quantity'),
               metadata={'export_contract':VERSION, 'source_geometry_version':version,
                   'source_normal_encoding':source_encoding, 'source_id':raw.get('id'),
                   'original_size':list(native_size), 'materialized_crop_box':list(box) if box else None,
                   'size':list(rgb.size), 'invalid_fraction':float((valid == 0).mean()),
                   'source_metadata_is_provenance_only':True})
    if normals: row['normal_encoding'] = VISION_BANANA_NORMAL_ENCODING
    # Keep original conventions, calibration, references and estimator details
    # as provenance, not dangling dependencies in training-row metadata.
    provenance = dict(ordinal=ordinal, id=raw.get('id'), original_row=raw,
        source_archive=Path(entry['path']).name,
        calibration_note='Original calibration is provenance; cropped pixels have a translated principal point. No calibration file is required by this RGB/target/mask loader.')
    return row, assets, provenance


class ShardWriter:
    def __init__(self, directory, source, number):
        self.path = directory/f'{source}-{number:05d}.tar'
        self.temporary = self.path.with_name(self.path.name+'.partial')
        self.tar = tarfile.open(self.temporary, 'w', format=tarfile.USTAR_FORMAT)
        self.rows, self.members, self.provenance = [], [], []

    def add(self, name, data):
        info = tarfile.TarInfo(name); info.size=len(data); info.mode=0o644; info.mtime=0
        self.members.append((name,self.tar.offset+512,len(data)))
        self.tar.addfile(info, io.BytesIO(data))

    def row(self, row, assets, provenance):
        for name, data in assets: self.add(name, data)
        self.rows.append(row); self.provenance.append(provenance)

    def finish(self, index_dir):
        self.add('metadata.jsonl', ''.join(json.dumps(r,sort_keys=True)+'\n' for r in self.rows).encode())
        self.add('provenance.jsonl', ''.join(json.dumps(r,sort_keys=True)+'\n' for r in self.provenance).encode())
        self.tar.close()
        self.temporary.replace(self.path); self.path.chmod(0o644)
        sha = digest(self.path)
        Path(str(self.path)+'.sha256').write_text(f'{sha}  {self.path.name}\n')
        index = index_dir/(self.path.name+'.ovb-index.sqlite')
        temporary = index.with_name(index.name+'.partial')
        if temporary.exists(): temporary.unlink()
        with sqlite3.connect(temporary) as db:
            db.executescript('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);'
                'CREATE TABLE members(name TEXT PRIMARY KEY,offset INTEGER NOT NULL,size INTEGER NOT NULL);'
                'CREATE TABLE rows(row_number INTEGER PRIMARY KEY,payload TEXT NOT NULL);')
            db.executemany('INSERT INTO members VALUES(?,?,?)',self.members)
            db.executemany('INSERT INTO rows VALUES(?,?)',[(n,json.dumps(r,sort_keys=True)) for n,r in enumerate(self.rows)])
            fingerprint=dict(path=self.path.name, size=self.path.stat().st_size, sha256=sha)
            db.executemany('INSERT INTO metadata VALUES(?,?)', dict(schema_version=str(INDEX_SCHEMA_VERSION),
                archive_fingerprint=json.dumps(fingerprint), num_rows=str(len(self.rows)),
                num_members=str(len(self.members)),metadata_members='["metadata.jsonl"]').items())
        temporary.replace(index); index.chmod(0o644)
        Path(str(index)+'.sha256').write_text(f'{digest(index)}  {index.name}\n')
        return dict(archive=self.path.name, archive_sha256=sha, index=index.name,
                    index_sha256=digest(index), rows=len(self.rows), bytes=self.path.stat().st_size)


def export_source(root, source_number, workers=8, sample=False):
    from common.indexed_tar import validate_archive_fingerprint
    plan = checked(root/'PLAN.json'); source = plan['sources'][source_number]; e = source['entry']
    if digest(Path(source['frozen_index'])) != source['index_sha256']:
        raise ValueError('Frozen selection changed')
    validate_archive_fingerprint(Path(e['path']), plan['source_archive_fingerprints'][e['name']])
    output = root/'samples' if sample else root
    for folder in ('archives','indices','reports'): (output/folder).mkdir(parents=True,exist_ok=True)
    report = output/'reports'/f'{e["name"]}.json'
    if report.exists():
        done = checked(report)
        if done['plan_sha256'] != digest(root/'PLAN.json'): raise ValueError('Completed source belongs to another plan')
        for s in done['shards']:
            if digest(output/'archives'/s['archive']) != s['archive_sha256'] or digest(output/'indices'/s['index']) != s['index_sha256']:
                raise ValueError('Completed package changed')
        print('Already exported and checked:',e['name']); return
    shards=[]; writer=None; count=0
    with readonly(source['frozen_index']) as db, ThreadPoolExecutor(max_workers=workers) as pool:
        members={name:(offset,size) for name,offset,size in db.execute('select name,offset,size from members')}
        rows=db.execute('select row_number,payload from rows order by row_number')
        if sample:
            # First/middle/last: includes source-order and boundary cases, not a
            # quality selection. This miniature is never a training manifest.
            ordinals=[r[0] for r in db.execute('select row_number from rows order by row_number')]
            take=set(ordinals[:2]+ordinals[len(ordinals)//2:len(ordinals)//2+2]+ordinals[-2:])
            rows=(r for r in rows if r[0] in take)
        while batch:=list(itertools.islice(rows, max(1,workers)*2)):
            for row,assets,provenance in pool.map(lambda item:materialize(item,e,members), batch):
                asset_bytes=sum(len(data)+1024 for _,data in assets)
                if writer is not None and writer.rows and (len(writer.rows)>=plan['rows_per_shard'] or writer.tar.offset+asset_bytes>plan['max_shard_bytes']):
                    shards.append(writer.finish(output/'indices'));writer=None
                if writer is None: writer=ShardWriter(output/'archives',e['name'],len(shards))
                writer.row(row,assets,provenance);count+=1
            if count % 1000 < len(batch): print(e['name'],count,'rows',flush=True)
    if writer is not None: shards.append(writer.finish(output/'indices'))
    if not sample and count != source['rows']: raise ValueError('Row count changed during export')
    write(report,dict(source=e['name'], rows=count, shards=shards, sample_only=sample,
                      plan_sha256=digest(root/'PLAN.json'), no_runtime_crops_or_filters=True))
    print(e['name'],count,'rows exported',flush=True)


def portable_manifest(template, reports, folder=False):
    payload={k:copy.deepcopy(v) for k,v in template.items() if k in ('samples_per_epoch','sampling_strategy','seed','shuffle')}
    payload['datasets']=[]
    for e in template['datasets']:
        if e['name'] not in reports: continue
        report=reports[e['name']]
        for s in report['shards']:
            item=dict(name=s['archive'][:-4], weight=e['weight']*s['rows']/report['rows'],
                      type='prepared_folder' if folder else 'indexed_prepared_tar',
                      path=('folders/'+s['archive'][:-4]) if folder else ('archives/'+s['archive']),
                      allow_missing_loss_mask=False)
            if not folder: item['index_path']='indices/'+s['index']
            payload['datasets'].append(item)
    return payload


def finalize(root):
    plan=checked(root/'PLAN.json'); reports={}
    acceptance=checked(root/'SAMPLE_ACCEPTANCE.json')
    if acceptance['status']!='pass' or acceptance['plan_sha256']!=digest(root/'PLAN.json'):
        raise ValueError('Actual pinned-loader sample acceptance is missing')
    for source in plan['sources']:
        report=checked(root/'reports'/f'{source["entry"]["name"]}.json')
        if report['rows']!=source['rows'] or report['sample_only'] or report['plan_sha256']!=digest(root/'PLAN.json'):
            raise ValueError('Incomplete or mismatched source report')
        for s in report['shards']:
            for field,directory in [('archive','archives'),('index','indices')]:
                if digest(root/directory/s[field])!=s[field+'_sha256']:raise ValueError('Output checksum mismatch')
        reports[report['source']]=report
    for name, template in plan['templates'].items():
        for folder in (False,True):
            write(root/'manifests'/f'{name}_geometry_{"folders" if folder else "indexed"}.json',portable_manifest(template,reports,folder))
    write(root/'PACKAGE.json',dict(version=VERSION, sources=reports, rows=plan['source_rows'],
         scope=plan['scope'], upstream_revision=plan['upstream_revision'],
         upstream_unmodified_masked_loadable=True, upstream_unmasked_requires_validity_patch=True,
         sample_acceptance=acceptance,
         redistribution_terms_reviewed=False))
    write(root/'READY.json',dict(status='materialized_export_complete',package_sha256=digest(root/'PACKAGE.json'),
         plan_sha256=digest(root/'PLAN.json'),rows=plan['source_rows'],
         compatibility_note='Masked: unmodified pinned release. Unmasked: apply supplied non-filtering validity preprocessing patch.',
         public_upload_authorized=False))
    print('All materialized geometry archives complete; no upload or training submitted')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['initialize','export','sample','finalize'])
    p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--source-root',type=Path,default=INPUT)
    p.add_argument('--release-repo',type=Path,default=PINNED_REPO)
    p.add_argument('--source',type=int)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--rows-per-shard',type=int,default=5000)
    a=p.parse_args()
    if a.workers<1 or a.rows_per_shard<1: p.error('Positive worker/shard sizes required')
    if a.action=='initialize': initialize(a.root,a.source_root,a.release_repo,a.rows_per_shard)
    elif a.action=='finalize': finalize(a.root)
    else:
        if a.source is None:p.error('--source is required')
        export_source(a.root,a.source,a.workers,a.action=='sample')


if __name__=='__main__':main()
