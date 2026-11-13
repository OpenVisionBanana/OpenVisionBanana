#!/usr/bin/env python3
"""Verify a downloaded portable geometry release and resolve local loader paths.

No filtering, selection, target conversion or curation is performed here.
Optional extraction is intended for node-local scratch.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(16*1024**2):h.update(block)
    return h.hexdigest()


def checked(path):
    if sha(path)!=Path(str(path)+'.sha256').read_text().split()[0]:
        raise ValueError(f'Checksum mismatch: {path}')
    return json.loads(path.read_text())


def resolve(payload, root, folder=False, extracted=None):
    for e in payload['datasets']:
        e['path']=str((extracted/e['name']) if folder else root/e['path'])
        if 'index_path' in e:e['index_path']=str(root/e['index_path'])
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pool',choices=['zeroshot','base','added_only'],default='base')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--extract-to',type=Path,help='Optional node-local destination for a folder loader')
    p.add_argument('--other-task-manifest',type=Path,help='Existing local pool; replaces only its geometry entries')
    a=p.parse_args();root=a.root.resolve()
    ready=checked(root/'READY.json');package=checked(root/'PACKAGE.json')
    if ready['status']!='materialized_export_complete' or ready['package_sha256']!=sha(root/'PACKAGE.json'):
        raise ValueError('Package is not complete')
    folder=a.extract_to is not None
    manifest=checked(root/'manifests'/f'{a.pool}_geometry_{"folders" if folder else "indexed"}.json')
    wanted={e['name'] for e in manifest['datasets']}
    shards=[s for report in package['sources'].values() for s in report['shards'] if s['archive'][:-4] in wanted]
    if len(wanted)!=len(manifest['datasets']) or len(shards)!=len(wanted) or {s['archive'][:-4] for s in shards}!=wanted:
        raise ValueError('Manifest entries must have unique, complete shard records')
    for s in shards:
        if sha(root/'archives'/s['archive'])!=s['archive_sha256'] or sha(root/'indices'/s['index'])!=s['index_sha256']:
            raise ValueError('Downloaded archive or index checksum mismatch')
    extracted=a.extract_to.resolve() if folder else None
    if folder:
        for s in shards:
            destination=extracted/s['archive'][:-4]
            stamp=destination/'.ovb-extracted-sha256'
            if stamp.exists() and stamp.read_text().strip()==s['archive_sha256']:continue
            if destination.exists():raise FileExistsError(f'Refusing to overwrite extraction: {destination}')
            destination.mkdir(parents=True)
            with tarfile.open(root/'archives'/s['archive'],'r:') as tar:
                # Archives contain only relative regular files. Refuse link,
                # device and path traversal members even on older Python.
                for item in tar:
                    path=Path(item.name)
                    if not item.isfile() or path.is_absolute() or '..' in path.parts:
                        raise ValueError(f'Unsafe archive member: {item.name}')
                    target=destination/path;target.parent.mkdir(parents=True,exist_ok=True)
                    with tar.extractfile(item) as src,target.open('xb') as dst:
                        __import__('shutil').copyfileobj(src,dst)
            stamp.write_text(s['archive_sha256']+'\n')
    payload=resolve(manifest,root,folder,extracted)
    if a.other_task_manifest:
        existing=json.loads(a.other_task_manifest.read_text())
        others=[e for e in existing['datasets'] if not e['name'].startswith(('depth_','surface_normals_'))]
        if abs(sum(e['weight'] for e in others)-.7)>1e-8 or abs(sum(e['weight'] for e in payload['datasets'])-.3)>1e-8:
            raise ValueError('Expected original 0.7 other-task / 0.3 geometry masses')
        for e in others:
            for field in ('path','index_path'):
                if e.get(field):e[field]=str((a.other_task_manifest.resolve().parent/Path(e[field]).expanduser()).resolve())
            if e.get('shards'):
                def local_shard(value):
                    return str((a.other_task_manifest.resolve().parent/Path(value).expanduser()).resolve())
                e['shards']=([local_shard(value) for value in e['shards']]
                             if isinstance(e['shards'],list) else local_shard(e['shards']))
        payload['datasets']+=others
    if a.output.exists():raise FileExistsError(f'Refusing to replace manifest: {a.output}')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,indent=2)+'\n')
    print(f'Verified {len(shards)} shards; wrote {a.output}. '+(
        'Preserved 0.7 other-task / 0.3 geometry masses.' if a.other_task_manifest else
        'Geometry-only task weights retained (sum=0.3); not a full multitask manifest.'))


if __name__=='__main__':main()
