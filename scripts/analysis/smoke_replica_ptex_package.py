#!/usr/bin/env python3
"""Exercise actual release loaders on the corrected RGB package, without GPUs."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(16*1024**2):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--role', choices=['working', 'public_release'], required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.repo.resolve()))
    import torch
    import train_dreambooth_partial_flux2_klein_img2img as trainer
    root = a.package.resolve()
    package = json.loads((root/'PACKAGE.json').read_text())
    data = trainer.DreamBoothDataset.__new__(trainer.DreamBoothDataset)
    data.instance_prompt = 'unused'
    data.args = SimpleNamespace(image_column='image', cond_image_column='conditioning_image',
        loss_mask_policy='auto', depth_target_resize_policy='metric_field_then_encode',
        normal_target_resize_policy='vector_field_renormalize_then_encode',
        normal_target_encoding='vision_banana_x_left_red', depth_target_encoding='vision_banana_rgb',
        depth_prompt_contract='canonical_v2', normal_prompt_contract='vision_banana_v3',
        center_crop=False, random_flip=False, seed=42, allow_missing_geometry_masks=False)
    cases = []
    for name in ('depth_legacy_replica', 'surface_normals_legacy_replica_camera_facing_v4'):
        shard = package['sources'][name]['shards'][0]
        archive, indexpath = root/'archives'/shard['archive'], root/'indices'/shard['index']
        assert digest(archive) == shard['archive_sha256'] and digest(indexpath) == shard['index_sha256']
        spec = trainer.DatasetSpec(name, archive, 1., type='indexed_prepared_tar', index_path=indexpath)
        index = trainer.PreparedDatasetIndex(spec, 'image', 'conditioning_image', 'text')
        data.rows = index.rows
        for position in (0, len(index.rows)//2, len(index.rows)-1):
            row = index.rows[position]
            results = {}
            for policy in ('auto', 'disabled'):
                data.args.loss_mask_policy = policy
                data._validate_loss_mask_policy()
                result = data._load_example(row, 0, 0, (64, 96))
                for field in ('instance_images', 'cond_images'):
                    assert result[field].shape == (3, 64, 96)
                    assert torch.isfinite(result[field]).all() and result[field].abs().max() <= 1
                if policy == 'auto':
                    assert set(result['loss_mask'].unique().tolist()) <= {0., 1.}
                    assert result['loss_mask'].sum() > 0
                else:
                    assert 'loss_mask' not in result
                assert result['sample_valid'] and result['instance_prompt']
                results[policy] = result
            for field in ('instance_images', 'cond_images'):
                assert torch.equal(results['auto'][field], results['disabled'][field])
            # Exercise the plain-folder path too, using the very same pixels.
            with sqlite3.connect('file:'+str(indexpath)+'?mode=ro', uri=True) as db:
                raw = json.loads(db.execute('select payload from rows where row_number=?', (position,)).fetchone()[0])
                with tempfile.TemporaryDirectory(prefix='ovb-ptex-loader-') as tmp:
                    folder = Path(tmp)
                    with archive.open('rb') as f:
                        for field in ('conditioning_image', 'image', 'loss_mask'):
                            member = raw[field]
                            path = Path(member)
                            assert not path.is_absolute() and '..' not in path.parts
                            offset, length = db.execute('select offset,size from members where name=?', (member,)).fetchone()
                            f.seek(offset); blob = f.read(length); assert len(blob) == length
                            destination = folder/path
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            destination.write_bytes(blob)
                    (folder/'metadata.jsonl').write_text(json.dumps(raw)+'\n')
                    findex = trainer.PreparedDatasetIndex(trainer.DatasetSpec(name, folder, 1.),
                        'image', 'conditioning_image', 'text')
                    for policy in ('auto', 'disabled'):
                        data.args.loss_mask_policy = policy
                        folderresult = data._load_example(findex.rows[0], 0, 0, (64, 96))
                        for field in ('instance_images', 'cond_images'):
                            assert torch.equal(results[policy][field], folderresult[field])
            cases.append(dict(source=name, row=position, masked_unmasked_and_folder_indexed_equal=True))
        print(f'PASS {a.role}: {name}', flush=True)
    result = dict(status='PASS', repo_role=a.role, repo=str(a.repo.resolve()),
        trainer_sha256=digest(a.repo/'train_dreambooth_partial_flux2_klein_img2img.py'),
        package_sha256=digest(root/'PACKAGE.json'), cases=cases,
        geometry_settings=dict(depth_encoding='vision_banana_rgb', normal_encoding='vision_banana_x_left_red'),
        training_forward_backward_tested=False)
    output = root/'reports'/f'loader_{a.role}.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2)+'\n')
    Path(str(output)+'.sha256').write_text(f'{digest(output)}  {output.name}\n')


if __name__ == '__main__':
    main()
