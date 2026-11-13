#!/usr/bin/env python3
"""Hash all selected archive/index/model/runtime bytes before allocating GPUs."""
from __future__ import annotations
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from common.training.asset_lock import digest, identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--container',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    manifest=json.loads(a.manifest.read_text());assets={a.manifest.resolve(),a.container.resolve()}
    for entry in manifest['datasets']:
        for key in ('path','index_path'):
            if key in entry:
                path=Path(entry[key]);path=(path if path.is_absolute() else a.manifest.parent/path).resolve()
                if not path.is_file():raise ValueError(f'Lock requires indexed immutable files: {path}')
                assets.add(path)
                sidecar=path.with_name(path.name+'.sha256')
                if sidecar.exists():assets.add(sidecar)
    assets.update(p.resolve() for p in a.model.rglob('*') if p.is_file() and '.cache' not in p.parts)
    code_paths=set(ROOT.glob('*.py'))
    for folder in ('common','dataset_prep','evaluation','scripts','configs','requirements'):
        code_paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in {'.py','.sh','.job','.json','.yaml','.yml','.txt'} and '__pycache__' not in p.parts)
    print(f'verifying {len(assets)} assets and {len(code_paths)} code/config files',flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as executor:verified=list(executor.map(identity,sorted(assets)))
    by_path={x['path']:x for x in verified}
    for item in verified:
        sidecar=Path(item['path']+'.sha256')
        if sidecar.exists() and sidecar.read_text().split()[0]!=item['sha256']:
            raise ValueError(f'SHA sidecar does not match actual bytes: {item["path"]}')
    payload={'schema_version':1,'content_verified':True,'manifest_sha256':digest(a.manifest),
             'model_root':str(a.model.resolve()),'container_path':str(a.container.resolve()),
             'code_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sorted(code_paths)},'assets':verified}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(payload,sort_keys=True,indent=2)+'\n'
    if a.output.exists() and a.output.read_text()!=data:raise FileExistsError(f'Frozen lock differs: {a.output}')
    temporary=a.output.with_suffix('.partial');temporary.write_text(data);temporary.replace(a.output)
    print(f'PASS content-verified lock {a.output}',flush=True)


if __name__=='__main__':main()
