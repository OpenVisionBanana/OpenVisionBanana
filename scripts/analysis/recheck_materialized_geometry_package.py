#!/usr/bin/env python3
"""Read-only full-index audit and sampled actual-release-loader acceptance."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(4*1024**2):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    sys.path.insert(0, str(a.repo.resolve()))
    import numpy as np
    import torch
    import train_dreambooth_partial_flux2_klein_img2img as t
    from common.indexed_tar import load_index
    from common.surface_normal_encoding import decode_surface_normals_rgb

    package = json.loads((a.package/'PACKAGE.json').read_text())
    ready = json.loads((a.package/'READY.json').read_text())
    errors, issues, sources, weights = [], [], {}, {}
    assert ready['status'] == 'materialized_export_complete'
    assert ready['package_sha256'] == sha(a.package/'PACKAGE.json')
    for filename in ('PACKAGE.json', 'READY.json', 'VALIDATION.json'):
        assert sha(a.package/filename) == Path(str(a.package/filename)+'.sha256').read_text().split()[0]
    for pool in ('zeroshot', 'base', 'added_only'):
        path = a.package/'manifests'/f'{pool}_geometry_indexed.json'
        assert sha(path) == Path(str(path)+'.sha256').read_text().split()[0]
        spec = json.loads(path.read_text())
        masses = defaultdict(float)
        for e in spec['datasets']:
            assert math.isfinite(e['weight']) and e['weight'] > 0
            source = e['name'].rsplit('-', 1)[0]
            assert source in package['sources']
            masses[source] += e['weight']
            assert (a.package/e['path']).is_file() and (a.package/e['index_path']).is_file()
        depth = sum(w for n,w in masses.items() if n.startswith('depth_'))
        normals = sum(w for n,w in masses.items() if n.startswith('surface_normals_'))
        assert abs(depth-.1) < 1e-8 and abs(normals-.2) < 1e-8
        for prefix in ('depth_', 'surface_normals_'):
            values = [w for n,w in masses.items() if n.startswith(prefix)]
            assert max(values)-min(values) < 1e-8, (pool, prefix, values)
        weights[pool] = dict(source_masses=dict(masses), depth=depth, normals=normals)

    data = t.DreamBoothDataset.__new__(t.DreamBoothDataset)
    data.instance_prompt = 'unused'
    data.args = SimpleNamespace(image_column='image', cond_image_column='conditioning_image',
        loss_mask_policy='auto', depth_target_resize_policy='metric_field_then_encode',
        normal_target_resize_policy='vector_field_renormalize_then_encode',
        normal_target_encoding='vision_banana_x_left_red', depth_target_encoding='relative_inverse_grayscale',
        depth_prompt_contract='campaign_detailed_inverse', normal_prompt_contract='campaign_short',
        center_crop=False, random_flip=False, seed=42, allow_missing_geometry_masks=False,
        allow_unverified_normal_sources=False, allow_legacy_hypersim_range_depth=False)
    row_total, sample_total = 0, 0
    for name, source in sorted(package['sources'].items()):
        seen, counters, examples = set(), Counter(), []
        count = 0
        for shard in source['shards']:
            archive, ipath = a.package/'archives'/shard['archive'], a.package/'indices'/shard['index']
            assert archive.stat().st_size == shard['bytes']
            assert sha(ipath) == shard['index_sha256']
            assert Path(str(archive)+'.sha256').read_text().split()[0] == shard['archive_sha256']
            with sqlite3.connect('file:'+str(ipath)+'?mode=ro', uri=True) as c:
                assert c.execute('pragma integrity_check').fetchone()[0] == 'ok'
                bad_offsets = c.execute('select count(*) from members where offset<0 or size<=0 or offset+size>?',
                    (archive.stat().st_size,)).fetchone()[0]
                assert bad_offsets == 0
            # The real loader checks index fingerprints and resolves every
            # referenced RGB/GT/mask member, not just the sampled positions.
            raw, _ = load_index(ipath, archive)
            assert len(raw) == shard['rows']
            for r in raw:
                assert all(r.get(k) is not None for k in ('id','image','conditioning_image','loss_mask','text','task','source_dataset'))
                assert r['id'] not in seen, (name, r['id'])
                seen.add(r['id'])
                assert r['text'].strip()
                assert r.get('geometry_version') == 'geometry_vision_banana_v3'
                assert not any(k in r for k in ('training_crop_box','training_quality_filter'))
                meta = r.get('metadata', {})
                invalid = float(meta['invalid_fraction'])
                assert math.isfinite(invalid) and 0 <= invalid < 1
                counters['invalid_over_10pct'] += invalid > .1
                counters['invalid_over_20pct'] += invalid > .2
                if r['task'] == 'surface_normal_estimation':
                    assert r['normal_encoding'] == 'vision_banana_x_left_red'
                if r['source_dataset'] == 'hypersim' and r['task'] == 'depth_estimation':
                    assert r['canonical_depth_quantity'] == 'projective_camera_z_meters'
                t.get_target_interpolation(r['task'], r.get('preprocess_target'))
            del raw
            spec = t.DatasetSpec(name, archive, 1., type='indexed_prepared_tar', index_path=ipath)
            index = t.PreparedDatasetIndex(spec, 'image', 'conditioning_image', 'text')
            data.rows = index.rows
            data._apply_geometry_contract()
            data._validate_loss_mask_policy()
            # Exercise first/middle/last of every shard at a non-square size.
            positions = sorted({0, len(index.rows)//2, len(index.rows)-1})
            for position in positions:
                row = index.rows[position]
                rgb = data._load_row_image(row, row.conditioning_image, 'conditioning_image')
                gt = data._load_row_image(row, row.image, 'image')
                mask = data._load_row_loss_mask(row)
                assert rgb.size == gt.size == mask.size
                m = np.asarray(mask.convert('L'))
                assert set(np.unique(m)).issubset({0,255}) and (m>0).any()
                warning = None
                if row.task == 'surface_normal_estimation':
                    _, vector_valid = decode_surface_normals_rgb(np.asarray(gt), encoding=row.normal_encoding)
                    fraction = float(((m>0) & ~vector_valid).sum()/max(1,(m>0).sum()))
                    if fraction > 0:
                        warning = dict(id=row.row_id, source=name, valid_mask_neutral_normal_fraction=fraction)
                        issues.append(warning)
                results = {}
                for policy in ('auto', 'disabled'):
                    data.args.loss_mask_policy = policy
                    result = data._load_example(row, position, 0, (64,96))
                    for field in ('instance_images','cond_images'):
                        assert result[field].shape == (3,64,96) and torch.isfinite(result[field]).all()
                        assert result[field].abs().max() <= 1
                    if policy == 'auto':
                        assert result['loss_mask'].shape == (1,64,96)
                        assert set(result['loss_mask'].unique().tolist()).issubset({0.,1.})
                        assert result['loss_mask'].sum() > 0
                    else:
                        assert 'loss_mask' not in result
                    results[policy] = result
                for field in ('instance_images','cond_images'):
                    assert torch.equal(results['auto'][field], results['disabled'][field])
                examples.append(dict(shard=shard['archive'], position=position, id=row.row_id,
                    invalid_fraction=float((m==0).mean()), masked_unmasked_targets_identical=True))
                sample_total += 1
            count += len(index.rows)
        assert count == source['rows']
        sources[name] = dict(rows=count, counters=dict(counters), samples=examples)
        row_total += count
        print(f'PASS {name}: {count} metadata rows; {len(examples)} actual-loader samples', flush=True)
    assert row_total == package['rows']
    result = dict(status='PASS', metadata_rows=row_total, shards=sum(len(s['shards']) for s in package['sources'].values()),
        actual_loader_samples=sample_total, sources=sources, weights=weights, review_notes=issues,
        trainer_sha256=sha(a.repo/'train_dreambooth_partial_flux2_klein_img2img.py'),
        package_sha256=sha(a.package/'PACKAGE.json'), full_archive_rehash=False,
        full_image_payload_scan=False, training_forward_backward_tested=False,
        scope='All metadata/index rows, index SHA-256, TAR sizes and declared hashes; sampled actual loader with current inverse-grayscale/vector-field recipe. Does not establish GT physical accuracy for every pixel.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(f'PASS {row_total} rows; {sample_total} loader samples; {len(issues)} normal-mask review notes')


if __name__ == '__main__':
    main()
