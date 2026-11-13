#!/usr/bin/env python3
"""Make immutable, benchmark-excluded index views without duplicating big TARs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.indexed_tar import validate_archive_fingerprint
from common.training.asset_lock import digest
from common.training.benchmark_exclusions import BENCHMARK_DOMAINS, domain, exclusion_reason, load_registry


def write_json(path, payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    data = json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if path.exists():
        if path.read_text() != data: raise FileExistsError(f"Refusing to change frozen file: {path}")
        return
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(data)
    temporary.replace(path)
    path.with_suffix(path.suffix+'.sha256').write_text(f'{digest(path)}  {path.name}\n')


def clean_index(entry, output, registry):
    source = Path(entry['index_path'])
    with sqlite3.connect(f'file:{source}?mode=ro',uri=True) as conn:
        metadata = dict(conn.execute('select key,value from metadata'))
        validate_archive_fingerprint(Path(entry['path']),json.loads(metadata['archive_fingerprint']))
        keep, reasons, sources = [], Counter(), set()
        for ordinal,payload in conn.execute('select row_number,payload from rows order by row_number'):
            row = json.loads(payload)
            if not (row.get('loss_mask') or row.get('valid_mask')):
                raise ValueError(f'Geometry row lacks source validity mask: {entry["name"]}/{row.get("id")}')
            sources.add(domain(row))
            reason = exclusion_reason(row,registry)
            if reason: reasons[reason]+=1
            else: keep.append(ordinal)
        total = conn.execute('select count(*) from rows').fetchone()[0]
        if not keep: return None, {'rows_before':total,'rows_after':0,'excluded':dict(reasons),'sources':sorted(sources)}
        if reasons:
            target = output / (entry['name']+'.benchmark_excluded.sqlite')
            view_identity = hashlib.sha256(json.dumps({
                'source_index_sha256':digest(source),'retained_ordinals':keep,
            },sort_keys=True).encode()).hexdigest()
            if target.exists():
                with sqlite3.connect(f'file:{target}?mode=ro',uri=True) as existing:
                    recorded=existing.execute('select value from metadata where key=?',('view_sha256',)).fetchone()
                    actual=[n for n, in existing.execute('select row_number from rows order by row_number')]
                if recorded != (view_identity,) or actual != keep:
                    raise ValueError(f'Existing index view does not match requested exclusions: {target}')
            if not target.exists():
                tmp = target.with_suffix('.partial')
                tmp.unlink(missing_ok=True)
                with sqlite3.connect(tmp) as dst:
                    conn.backup(dst)
                    dst.execute('create temp table retained(id integer primary key)')
                    dst.executemany('insert into retained values (?)',[(x,) for x in keep])
                    dst.execute('delete from rows where row_number not in (select id from retained)')
                    dst.execute('insert or replace into metadata values (?,?)',('num_rows',str(len(keep))))
                    dst.execute('insert or replace into metadata values (?,?)',('view_contract','geometry-readiness-v2 benchmark-excluded; ScanNet physical-location heldout'))
                    dst.execute('insert or replace into metadata values (?,?)',('view_sha256',view_identity))
                    dst.commit()
                tmp.replace(target)
            result = {**entry,'index_path':str(target)}
        else: result = dict(entry)
        return result, {'rows_before':total,'rows_after':len(keep),'excluded':dict(reasons),'sources':sorted(sources)}


def task_group(entry):
    if entry['name'].startswith('depth_'): return 'depth'
    if entry['name'].startswith('surface_normals_'): return 'normals'
    return 'other'


def normalize(entries, task_masses=None):
    if task_masses:
        totals=Counter()
        for entry in entries:totals[task_group(entry)]+=entry['weight']
        entries=[{**entry,'weight':entry['weight']/totals[task_group(entry)]*task_masses[task_group(entry)]} for entry in entries]
    total = sum(x['weight'] for x in entries)
    if total <= 0: raise ValueError('No training sources remain')
    return [{**x,'weight':x['weight']/total} for x in entries]


def build(base, split_root, output, replacement=None):
    output.mkdir(parents=True,exist_ok=True)
    (output/'indices').mkdir(exist_ok=True)
    template = json.loads(base.read_text())
    task_masses=Counter()
    for entry in template['datasets']:task_masses[task_group(entry)]+=entry['weight']
    registry,files = load_registry(split_root)
    cleaned, zs, reports = [], [], {}
    for entry in template['datasets']:
        entry = copy.deepcopy(entry)
        for key in ('path','index_path'):
            if key in entry:
                p=Path(entry[key]);entry[key]=str((p if p.is_absolute() else base.parent/p).resolve())
        if replacement and entry['name'].startswith('surface_normals_scannetv2_'):
            entry['path']=str(replacement.resolve())
            entry['index_path']=str(replacement.resolve())+'.ovb-index.sqlite'
        geometry=entry['name'].startswith(('depth_','surface_normals_'))
        if geometry:
            entry, report = clean_index(entry,output/'indices',registry)
            reports[entry['name'] if entry else 'fully_excluded']=report
            if entry is None: continue
            source=report['sources']
            if len(source)!=1 or not source[0]: raise ValueError('Unidentified geometry source')
            if source[0] not in BENCHMARK_DOMAINS: zs.append(entry)
        else:
            # Preserve other-task data in NZS; strict ZS removes named geometry
            # benchmark domains even when they occur under another task.
            tokens=set(entry['name'].lower().split('_'))
            if not tokens & (BENCHMARK_DOMAINS|{'nyuv2','scannetv2','kitti_eigen'}): zs.append(entry)
        cleaned.append(entry)
    contract={'version':'geometry-readiness-v2','source_manifest_sha256':digest(base),
              'split_sha256':{str(p):digest(p) for p in files},
              'nzs_policy':'exact IDs + ScanNet physical evaluation locations (all recordings, union of depth/normals) + KITTI/DIODE evaluation scenes excluded; VKITTI2 non-clone variant diagnostic',
              'zs_policy':'named geometry benchmark domains excluded across prepared tasks; no claim about unknown foundation/PD12M pretraining',
              'validity':'source masks required for preprocessing in both loss modes',
              'sampling':'preserve depth/normal/other task mass across ZS and NZS; normalize source weights within task',
              'normal_targets':'edge-aware ScanNet candidate' if replacement else 'camera-facing v4 baseline',
              'gate':'pilot validation pending; not a model-quality certification'}
    report={'contract':contract,'datasets':reports}
    write_json(output/'exclusion_report.json',report)
    for name,entries in (('nonzero',cleaned),('zeroshot',zs)):
        for scope in ('full','geometry'):
            selected=[x for x in entries if scope=='full' or x['name'].startswith(('depth_','surface_normals_'))]
            payload={**template,'datasets':normalize(selected,task_masses),'mix_contract':contract,
                     'samples_per_epoch':100000,'sampling_strategy':'with_replacement'}
            write_json(output/f'{name}_{scope}.json',payload)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--split-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scannet-replacement',type=Path)
    args=parser.parse_args()
    report=build(args.base,args.split_root,args.output,args.scannet_replacement)
    print(json.dumps(report,indent=2))
