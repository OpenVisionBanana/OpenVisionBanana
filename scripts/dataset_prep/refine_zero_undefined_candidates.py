#!/usr/bin/env python3
"""Drop paired rows that fail exhaustive final-loader zero-validity checks.

Only immutable index views are created. Certification is transferred solely for
identical retained payloads already checked in every training bucket. Thin
domains still require a sampling/diversity decision before any training.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from common.training.asset_lock import digest
from scripts.analysis.audit_zs_candidate_safety import rows,domain,candidate_manifest
from scripts.dataset_prep.build_geometry_readiness_manifests import write_json
from scripts.dataset_prep.curate_nzs_geometry import readonly


def refine(campaign,reports=None):
    reports=reports or campaign/'reports'
    manifest=campaign/'subsets/manifests/zero_undefined_geometry.json'
    full=campaign/'subsets/manifests/zero_undefined_full.json'
    output=campaign/'zero_refined'
    if output.exists():raise FileExistsError('Use a new output version; refinement is immutable')
    entries=json.loads(manifest.read_text())['datasets'];proofs=[];invalid={domain(e):set() for e in entries}
    for i,entry in enumerate(entries):
        path=reports/f'zero_final_{i}.json'
        if not path.exists():
            print('GATED: exhaustive checks incomplete; no refined zero pool created');return
        proof=json.loads(path.read_text());identity=proof['identity']
        if proof['status'] not in ('PASS','INVALID_ROWS_FOUND'):raise ValueError('Uncompleted or incompatible zero check')
        if (identity['manifest_sha256']!=digest(manifest) or identity['index_sha256']!=digest(Path(entry['index_path']))
            or identity['source']!=entry['name']):raise ValueError('Exhaustive check source changed')
        if identity['trainer_sha256']!=digest(ROOT/'train_dreambooth_partial_flux2_klein_img2img.py'):
            raise ValueError('Final loader changed; cannot transfer zero certification')
        source_rows=rows(entry)
        if proof['rows']!=len(source_rows) or proof['buckets']!=5:raise ValueError('Incomplete coverage')
        failed={p['id'] for p in proof.get('invalid_rows',[])}
        if proof['status']=='INVALID_ROWS_FOUND' and not failed:raise ValueError('Missing invalid row evidence')
        if not failed<=source_rows.keys():raise ValueError('Unknown failed row IDs')
        with readonly(reports/f'zero_final_{i}.sqlite') as db:
            metadata=dict(db.execute('select key,value from metadata'))
            checked={k for k, in db.execute('select id from checks')}
            if metadata.get('complete')!='true' or json.loads(metadata['identity'])!=identity or checked!=source_rows.keys():
                raise ValueError('SQLite row coverage does not match report')
        invalid[domain(entry)]|=failed;proofs.append((path,proof))
    full_template=json.loads(full.read_text())
    full_entries={e['name']:e for e in full_template['datasets']}
    pairs=[];selected={}
    for d in sorted(invalid):
        dep=next(e for e in entries if domain(e)==d and e['name'].startswith('depth_'))
        normal=next(e for e in entries if domain(e)==d and e['name'].startswith('surface_normals_'))
        if rows(dep).keys()!=rows(normal).keys():raise ValueError('Source pairs differ')
        pairs.append((full_entries[dep['name']],full_entries[normal['name']]));selected[d]=set(rows(dep))-invalid[d]
        if not selected[d]:
            print(f'GATED: no all-bucket-valid pairs in {d}; no silent domain exclusion');return
    output.mkdir();(output/'indices').mkdir();(output/'manifests').mkdir();(output/'reports').mkdir()
    item=candidate_manifest(full_template,pairs,selected,output,'zero_undefined')
    refined=json.loads((output/'manifests/zero_undefined_geometry.json').read_text())['datasets']
    new_manifest_hash=digest(output/'manifests/zero_undefined_geometry.json')
    for i,old in enumerate(entries):
        new=next(e for e in refined if e['name']==old['name'])
        old_rows=rows(old);new_rows=rows(new)
        if any(new_rows[k][1]!=old_rows[k][1] for k in new_rows):raise ValueError('Retained row content changed')
        path,proof=proofs[i]
        write_json(output/'reports'/f'zero_final_{i}.json',dict(status='PASS',source=new['name'],rows=len(new_rows),buckets=5,
            all_transformed_pixels_valid=True,identity={**proof['identity'],
                'manifest_sha256':new_manifest_hash,'index_sha256':digest(Path(new['index_path']))},
            certification_transfer=dict(original_report=str(path),original_report_sha256=digest(path),
                original_sqlite_sha256=digest(reports/f'zero_final_{i}.sqlite'),
                method='Retained payloads are byte-equivalent as parsed JSON; every retained ID was checked in every bucket; paired invalid IDs dropped, no new labels or transforms.')))
    item['exhaustive_report_directory']=str(output/'reports')
    item['geometry_manifest']=str(output/'manifests/zero_undefined_geometry.json')
    write_json(output/'CANDIDATES.json',{'zero_undefined':item})
    write_json(output/'REFINEMENT.json',dict(status='PASS',removed_pair_ids={d:sorted(v) for d,v in invalid.items()},
        retained_pairs=item['pairs'],training_approved=False,
        note='No originals changed. Very small domains remain blocked at unchanged sampling weight.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--reports',type=Path);a=p.parse_args();refine(a.campaign,a.reports)


if __name__=='__main__':main()
