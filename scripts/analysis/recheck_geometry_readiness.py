#!/usr/bin/env python3
"""Read-only checks of current training pools and frozen evaluation archives.

All training geometry metadata/member references are checked. Pixel checks are
sampled; passing them is deliberately not a certification of every label.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import train_dreambooth_partial_flux2_klein_img2img as trainer
from common.indexed_tar import validate_archive_fingerprint
from common.training.asset_lock import digest
from common.training.benchmark_exclusions import BENCHMARK_DOMAINS, domain, exclusion_reason, load_registry
from evaluation.depth.datasets import marigold_ppd
from evaluation.surface_normals import datasets as normal_datasets
from evaluation.surface_normals.inference_partial_flux2_klein_img2img import load_gt_normals
from evaluation.surface_normals.target_protocol import resolve_ground_truth_protocol


def geometry_members(row: dict, members: set[str]) -> dict:
    """Mirror the production loader's legacy metadata-relative mask alias."""
    mask = row.get('loss_mask') or row.get('valid_mask')
    if not row.get('loss_mask') and mask and mask not in members:
        metadata_member = str(row.get('_metadata_member', ''))
        prefix = metadata_member.rsplit('/', 1)[0] if '/' in metadata_member else ''
        prefixed = f'{prefix}/{mask}' if prefix else mask
        if prefixed in members:
            mask = prefixed
    return {'image': row.get('image'), 'conditioning_image': row.get('conditioning_image'), 'mask': mask}


def check_training(campaign: Path, split_root: Path, sample_count: int) -> dict:
    registry, _ = load_registry(split_root)
    nzs = json.loads((campaign/'baseline/nonzero_full.json').read_text())
    zs = json.loads((campaign/'baseline/zeroshot_full.json').read_text())
    nzs_entries = {e['name']: e for e in nzs['datasets']}
    zs_entries = {e['name']: e for e in zs['datasets']}
    for name, entry in zs_entries.items():
        original = nzs_entries[name]
        for key in ('path', 'index_path'):
            if entry[key] != original[key]:
                raise ValueError(f'ZS silently changes shared source bytes: {name}')
    task_mass = {}
    for mode, entries in (('zeroshot', zs_entries), ('nonzero', nzs_entries)):
        weights = Counter()
        for e in entries.values():
            group = 'depth' if e['name'].startswith('depth_') else 'normals' if e['name'].startswith('surface_normals_') else 'other'
            weights[group] += e['weight']
        if not np.isclose(sum(weights.values()), 1.):
            raise ValueError('Unnormalized task sampling weights')
        task_mass[mode] = dict(weights)
    if any(not np.isclose(task_mass['zeroshot'][key], value) for key, value in task_mass['nonzero'].items()):
        raise ValueError('ZS/NZS task probabilities differ')
    specs, _, _, _ = trainer.load_dataset_specs(str(campaign/'baseline/nonzero_geometry.json'))
    by_name = {s.name: s for s in specs}
    reports = []
    for name, entry in nzs_entries.items():
        geometry = name in by_name
        with sqlite3.connect(f'file:{entry["index_path"]}?mode=ro', uri=True) as conn:
            info = dict(conn.execute('select key,value from metadata'))
            validate_archive_fingerprint(Path(entry['path']), json.loads(info['archive_fingerprint']))
            if not geometry:
                declared = [source for source, in conn.execute("select distinct json_extract(payload,'$.source_dataset') from rows")]
                normalized = {domain({'source_dataset': source}) for source in declared}
                if name in zs_entries and normalized & BENCHMARK_DOMAINS:
                    raise ValueError(f'Benchmark domain retained in another ZS task: {name}')
                continue
            members = {n for n, in conn.execute('select name from members')}
            rows = 0
            sources, contracts = Counter(), Counter()
            ids = set()
            for payload, in conn.execute('select payload from rows order by row_number'):
                row = json.loads(payload); rows += 1
                source = domain(row); sources[source] += 1
                if row['id'] in ids:
                    raise ValueError(f'Duplicate geometry identity: {name}/{row["id"]}')
                ids.add(row['id'])
                if exclusion_reason(row, registry):
                    raise ValueError(f'Benchmark overlap remains: {name}/{row["id"]}')
                if name in zs_entries and source in BENCHMARK_DOMAINS:
                    raise ValueError(f'Benchmark geometry domain retained in ZS: {name}')
                for key, reference in geometry_members(row, members).items():
                    if not isinstance(reference, str) or reference not in members:
                        raise ValueError(f'Missing TAR {key} member: {name}/{row["id"]}')
                contract = trainer.prepared_row_geometry_contract(row)
                task = trainer.normalize_task_name(row.get('task'))
                expected = trainer.DEPTH_TASKS if name.startswith('depth_') else trainer.SURFACE_NORMAL_TASKS
                if task not in expected:
                    raise ValueError(f'Wrong task routing: {name}/{row["id"]}')
                if task in trainer.DEPTH_TASKS:
                    if contract['canonical_depth_quantity'] != 'projective_camera_z_meters':
                        raise ValueError(f'Non-projective depth contract: {name}/{row["id"]}')
                else:
                    if contract['normal_contract_status'] == 'unverified':
                        raise ValueError(f'Unverified normal axes: {name}/{row["id"]}')
                    contracts[contract['normal_contract_status']] += 1
        index = trainer.PreparedDatasetIndex(by_name[name], 'image', 'conditioning_image', 'text')
        chosen = np.unique(np.linspace(0, len(index.rows)-1, min(sample_count, len(index.rows))).round().astype(int))
        loader = trainer.DreamBoothDataset.__new__(trainer.DreamBoothDataset)
        bad_vectors = nonunit = valid_pixels = 0
        mask_values = set()
        for number in chosen:
            row = index.rows[number]
            target = loader._load_row_image(row, row.image, 'image')
            condition = loader._load_row_image(row, row.conditioning_image, 'conditioning_image')
            mask = loader._load_row_loss_mask(row)
            if target.size != condition.size or target.size != mask.size:
                raise ValueError(f'Unregistered source dimensions: {name}/{row.row_id}')
            raw_mask = np.asarray(mask.convert('L'))
            mask_values.update(map(int, np.unique(raw_mask)))
            valid = raw_mask > 127
            if not valid.any():
                raise ValueError(f'Empty binary source mask: {name}/{row.row_id}')
            if name.startswith('surface_normals_'):
                components = np.asarray(target.convert('RGB'), dtype=np.float32)/255.*2.-1.
                length = np.linalg.norm(components, axis=-1)
                bad_vectors += int(((length < .05) & valid).sum())
                nonunit += int(((np.abs(length-1.) > .025) & valid).sum())
                valid_pixels += int(valid.sum())
        if not mask_values <= {0,255}:
            raise ValueError(f'Non-binary source validity mask: {name}: {mask_values}')
        if bad_vectors:
            raise ValueError(f'Validity mask supervises {bad_vectors} grey/zero vectors in {name}')
        report = {'name':name, 'rows':rows, 'sources':dict(sources), 'in_zs':name in zs_entries,
                  'normal_contracts':dict(contracts), 'decoded_samples':len(chosen),
                  'mask_values':sorted(mask_values), 'masked_invalid_normal_vectors':bad_vectors,
                  'nonunit_normal_fraction_before_decoder':nonunit/max(valid_pixels,1)}
        reports.append(report)
        print(f'PASS metadata/members {name}: {rows} rows; {len(chosen)} decoded samples', flush=True)
        del index, loader
    locks = {}
    for mode in ('nonzero','zeroshot'):
        path = campaign/f'baseline/{mode}_geometry.json.lock.json'
        payload = json.loads(path.read_text())
        changed = []
        for asset in payload['assets']:
            stat = Path(asset['path']).stat()
            if (stat.st_size,stat.st_mtime_ns) != (asset['size'],asset['mtime_ns']):
                changed.append(asset['path'])
        if changed:
            raise ValueError(f'Assets changed since CPU byte verification: {changed}')
        locks[mode] = {'sha256':digest(path),'assets':len(payload['assets']),'unchanged_since_byte_verification':True}
    return {'status':'PASS','scope':'all geometry metadata/member references; sampled source pixels; no exhaustive semantic label certification',
            'full_manifest_sources':{'zeroshot':len(zs_entries),'nonzero':len(nzs_entries)},
            'task_probability':task_mass,'datasets':reports,'asset_locks':locks}


def stage_archive(archive: Path, stage: Path) -> str:
    expected = Path(str(archive)+'.sha256').read_text().split()[0]
    actual = digest(archive)
    if actual != expected:
        raise ValueError(f'Archive checksum mismatch: {archive}')
    subprocess.run(['tar','--no-same-owner','--no-same-permissions','-xf',str(archive),'-C',str(stage)],check=True)
    return actual


def check_evaluation(normal_root: Path, depth_root: Path, sample_count: int) -> dict:
    normals = []; depth = []
    for dataset, archive_name in [('nyu','nyu.tar'),('scannet','scannet.tar'),('diode','diode.tar'),('ibims1','ibims1_official.tar'),('vkitti2','vkitti2.tar'),('sintel','sintel.tar')]:
        with tempfile.TemporaryDirectory(prefix='ovb-normals-recheck-',dir=os.environ['TMPDIR']) as temporary:
            stage = Path(temporary)
            sha = stage_archive(normal_root/archive_name, stage)
            root = normal_datasets.resolve_dataset_root(dataset, stage)
            info = json.loads((root/'dataset_info.json').read_text())
            samples = normal_datasets.build_samples(dataset,root,'test',None,('indoor',) if dataset=='diode' else ('standard',),('appearance',))
            if len(samples) != info['count']:
                raise ValueError(f'Normal benchmark count differs: {dataset}: {len(samples)} vs {info["count"]}')
            protocol = resolve_ground_truth_protocol(dataset_info=info,storage_formats=[s.normal_path.suffix for s in samples])
            if protocol.permutation != (0,1,2) or protocol.signs != (1,1,1) or not protocol.encoding.startswith('float_xyz'):
                raise ValueError(f'Unexpected reference-target transform: {dataset}: {protocol}')
            chosen = np.unique(np.linspace(0,len(samples)-1,min(sample_count,len(samples))).round().astype(int))
            fractions = []
            for number in chosen:
                sample = samples[number]
                unit, valid = load_gt_normals(sample.normal_path,min_norm=.01)
                with Image.open(sample.image_path) as image:
                    if image.size != (unit.shape[1],unit.shape[0]):
                        raise ValueError(f'Evaluation RGB/normal registration differs: {dataset}/{sample.sample_id}')
                if not valid.any():
                    raise ValueError(f'Empty reference normal: {dataset}/{sample.sample_id}')
                if not np.isfinite(unit).all():
                    raise ValueError('Nonfinite normal after reference decoding')
                fractions.append(float(valid.mean()))
            normals.append({'dataset':dataset,'archive_sha256':sha,'images':len(samples),'decoded_samples':len(chosen),
                            'gt_protocol':protocol.metadata(),'median_valid_fraction':float(np.median(fractions))})
            print(f'PASS normal evaluation {dataset}: {len(samples)} images',flush=True)
    paths = {'nyu':'nyuv2/nyu_labeled_extracted.tar','kitti_eigen':'kitti/kitti_eigen_split_test.tar',
             'eth3d':'eth3d/eth3d.tar','scannet':'scannet/scannet_val_sampled_800_1.tar','diode':'diode/diode_val.tar'}
    for dataset, relative in paths.items():
        with tempfile.TemporaryDirectory(prefix='ovb-depth-recheck-',dir=os.environ['TMPDIR']) as temporary:
            stage = Path(temporary)
            sha = stage_archive(depth_root/'archives'/relative,stage)
            samples = marigold_ppd.build_samples(dataset,stage,depth_root/'data_split',None)
            chosen = np.unique(np.linspace(0,len(samples)-1,min(sample_count,len(samples))).round().astype(int))
            for number in chosen:
                sample = samples[number]
                gt = marigold_ppd.load_gt_depth(sample)
                valid = marigold_ppd.valid_depth_mask(sample,gt)
                if valid.shape != gt.shape or not (valid & np.isfinite(gt) & (gt>0)).any():
                    raise ValueError(f'Empty/misaligned reference depth mask: {dataset}/{sample.sample_id}')
                with Image.open(sample.image_path) as image:
                    prepared = marigold_ppd.prepare_input_image(sample,image)
                    if dataset=='eth3d' and prepared.size != (1008,672):
                        raise ValueError('ETH3D input size differs from PPD')
                    if dataset=='kitti_eigen' and prepared.size != (gt.shape[1],gt.shape[0]):
                        raise ValueError('KITTI benchmark crops are not registered')
            depth.append({'dataset':dataset,'archive_sha256':sha,'images':len(samples),'decoded_samples':len(chosen)})
            print(f'PASS PPD depth evaluation {dataset}: {len(samples)} images',flush=True)
    return {'status':'PASS','scope':'complete benchmark split/member resolution and archive hashes; sampled decoded GT/masks',
            'normal_datasets':normals,'depth_datasets':depth,
            'not_certified':['depth-derived/sparse KITTI, ETH3D, nuScenes normal diagnostics','every possible legacy launcher/override','exhaustive upstream byte equality of every GT array']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--section',required=True,choices=['training','evaluation'])
    p.add_argument('--campaign',type=Path,default=Path('data/geometry_training_readiness_20261002'))
    p.add_argument('--depth-root',type=Path,default=Path('data/marigold_v2_depth_eval_20260920'))
    p.add_argument('--normal-root',type=Path,default=Path('data/geometry_evaluation_v6/surface_normals'))
    p.add_argument('--samples',type=int,default=32);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.samples<1:raise ValueError('samples must be positive')
    report = check_training(a.campaign,a.depth_root/'data_split',a.samples) if a.section=='training' else check_evaluation(a.normal_root,a.depth_root,a.samples)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temp = a.output.with_suffix('.partial');temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(a.output)
    print(f'PASS {a.section}: {a.output}',flush=True)


if __name__=='__main__':main()
