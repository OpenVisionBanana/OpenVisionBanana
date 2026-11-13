import json
from pathlib import Path
import sqlite3

from common.training.asset_lock import digest
from scripts.dataset_prep.refine_zero_undefined_candidates import refine, ROOT


def test_final_zero_refinement_drops_both_tasks_and_preserves_full_weights(tmp_path):
    subsets=tmp_path/'subsets';(subsets/'manifests').mkdir(parents=True)
    (tmp_path/'reports').mkdir()
    names=['hypersim','matterport3d','replica','synscapes','taskonomy']
    full=[]
    for task in ('depth','surface_normals'):
        for d in names:
            name=f'{task}_{d}';index=tmp_path/f'{name}.sqlite'
            with sqlite3.connect(index) as db:
                db.execute('create table metadata(key text primary key,value text)')
                db.execute('create table rows(row_number integer primary key,payload text)')
                db.executemany('insert into rows values (?,?)',[(i,json.dumps({'id':f'{d}_{i}','metadata':{}})) for i in range(2)])
            full.append(dict(name=name,index_path=str(index),path=str(tmp_path/'untouched.tar'),weight=.03))
    geometry=[{**e,'weight':.1} for e in full]
    non_geometry=dict(name='other_task',weight=.7)
    (subsets/'manifests/zero_undefined_full.json').write_text(json.dumps({'datasets':full+[non_geometry]}))
    manifest=subsets/'manifests/zero_undefined_geometry.json';manifest.write_text(json.dumps({'datasets':geometry}))
    for i,entry in enumerate(geometry):
        failed=i==5
        identity=dict(manifest_sha256=digest(manifest),index_sha256=digest(Path(entry['index_path'])),
                      trainer_sha256=digest(ROOT/'train_dreambooth_partial_flux2_klein_img2img.py'),source=entry['name'])
        proof=dict(status='INVALID_ROWS_FOUND' if failed else 'PASS',rows=2,buckets=5,identity=identity,
                   invalid_rows=[{'id':'hypersim_1'}] if failed else [])
        (tmp_path/'reports'/f'zero_final_{i}.json').write_text(json.dumps(proof))
        d=names[i%5]
        with sqlite3.connect(tmp_path/'reports'/f'zero_final_{i}.sqlite') as db:
            db.execute('create table metadata(key text primary key,value text)')
            db.executemany('insert into metadata values (?,?)',[('complete','true'),('identity',json.dumps(identity))])
            db.execute('create table checks(id text)')
            db.executemany('insert into checks values (?)',[(f'{d}_{j}',) for j in range(2)])
    refine(tmp_path)
    output=tmp_path/'zero_refined'
    rebuilt=json.loads((output/'manifests/zero_undefined_full.json').read_text())['datasets']
    assert [e['weight'] for e in rebuilt]==[e['weight'] for e in full+[non_geometry]]
    assert rebuilt[-1]==non_geometry
    for entry in rebuilt[:-1]:
        with sqlite3.connect(entry['index_path']) as db:
            ids=[json.loads(p)['id'] for p, in db.execute('select payload from rows')]
        assert len(ids)==(1 if 'hypersim' in entry['name'] else 2)
    registry=json.loads((output/'CANDIDATES.json').read_text())['zero_undefined']
    assert registry['pairs']['hypersim']==1
    for i in range(10):
        proof=json.loads((output/'reports'/f'zero_final_{i}.json').read_text())
        assert proof['status']=='PASS'
        assert proof['identity']['manifest_sha256']==digest(output/'manifests/zero_undefined_geometry.json')
