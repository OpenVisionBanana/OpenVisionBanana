"""Portable export regression tests: selection stays fixed, loader needs no crops."""
import copy
import io
import json
from pathlib import Path
import sqlite3
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from common.indexed_tar import load_index, open_reference_image
from scripts.dataset_prep.export_geometry_training_release import materialize, portable_manifest, ShardWriter
from scripts.dataset_prep.use_geometry_training_release import resolve


class PortableTest(unittest.TestCase):
    def make(self, task='surface_normal_estimation', version='geometry_camera_facing_v4', encoding='vision_banana_x_left_red'):
        rgb=np.arange(6*8*3,dtype=np.uint8).reshape(6,8,3)
        gt=np.full_like(rgb,[10,128,250]);mask=np.full((6,8),255,np.uint8);mask[2,3]=0
        gt[2,3]=128
        row=dict(id='scene/f',text='Original prompt',task=task,source_dataset='replica',
            conditioning_image='rgb',image='gt',loss_mask='mask',geometry_version=version,
            normal_encoding=encoding,canonical_depth_quantity='projective_camera_z_meters',
            training_crop_box=[1,1,7,5],training_crop_contract='nzs-edge-curation-v1')
        images=dict(rgb=Image.fromarray(rgb),gt=Image.fromarray(gt),mask=Image.fromarray(mask))
        with patch('scripts.dataset_prep.export_geometry_training_release.open_reference_image',side_effect=lambda ref:images[ref.member].copy()):
            output=materialize((31,json.dumps(row)),dict(path='source.tar'),{k:(0,1) for k in images})
        return output,rgb,gt,mask,row

    def test_bakes_identical_paired_crop_without_changing_v4_pixels(self):
        (row,assets,_),rgb,gt,mask,_=self.make()
        for (_,data),expected in zip(assets,(rgb[1:5,1:7],gt[1:5,1:7],mask[1:5,1:7])):
            np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(data))),expected)
        self.assertNotIn('training_crop_box',row)
        self.assertNotIn('filters',row)
        self.assertEqual(row['metadata']['source_geometry_version'],'geometry_camera_facing_v4')
        self.assertEqual(row['geometry_version'],'geometry_vision_banana_v3')
        self.assertEqual(row['text'],'Original prompt')

    def test_legacy_normal_transcode_retains_physical_direction(self):
        (row,assets,_),_,gt,mask,_=self.make(version='geometry_canonical_v2',encoding='legacy_direct_xyz')
        from common.surface_normal_encoding import decode_surface_normals_rgb
        expected,_=decode_surface_normals_rgb(gt[1:5,1:7],encoding='legacy_direct_xyz')
        actual,_=decode_surface_normals_rgb(np.asarray(Image.open(io.BytesIO(assets[1][1]))),encoding=row['normal_encoding'])
        valid=mask[1:5,1:7]>0
        self.assertLess(float(np.abs(actual[valid]-expected[valid]).max()),.01)

    def test_depth_is_not_silently_reencoded_as_inverse(self):
        (_,assets,_),_,gt,_,_=self.make(task='depth_estimation',version='geometry_canonical_v2')
        np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(assets[1][1]))),gt[1:5,1:7])

    def test_portable_byte_offsets_and_loss_masks(self):
        (row,assets,provenance),*_=self.make()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);writer=ShardWriter(root,'example',0);writer.row(row,assets,provenance)
            s=writer.finish(root)
            relocated=root/'relocated';relocated.mkdir()
            for name in (s['archive'],s['index'],s['archive']+'.sha256',s['index']+'.sha256'):
                (root/name).rename(relocated/name)
            loaded,_=load_index(relocated/s['index'],relocated/s['archive'])
            self.assertEqual(len(loaded),1)
            with open_reference_image(loaded[0]['loss_mask']) as im:
                self.assertEqual(im.size,(6,4));self.assertEqual(np.sum(np.asarray(im)==0),1)
            with tarfile.open(relocated/s['archive']) as tar:
                self.assertEqual(len(tar.getmembers()),5)
                self.assertTrue(all(x.isfile() for x in tar))

    def test_shard_weights_preserve_source_not_shard_uniformity(self):
        template=dict(datasets=[dict(name='a',weight=.1),dict(name='other',weight=.7)],seed=42)
        reports=dict(a=dict(rows=100,shards=[dict(rows=30,archive='a-0.tar',index='a-0.sqlite'),dict(rows=70,archive='a-1.tar',index='a-1.sqlite')]))
        out=portable_manifest(template,reports)
        self.assertEqual(len(out['datasets']),2)
        self.assertAlmostEqual(out['datasets'][0]['weight'],.03)
        self.assertAlmostEqual(sum(e['weight'] for e in out['datasets']),.1)
        self.assertTrue(all('filters' not in e for e in out['datasets']))
        relocated=resolve(copy.deepcopy(out),Path('/other/machine'))
        self.assertEqual(relocated['datasets'][0]['path'],'/other/machine/archives/a-0.tar')

    def test_folder_manifests_have_no_index_or_filter(self):
        reports=dict(a=dict(rows=1,shards=[dict(rows=1,archive='a-0.tar',index='x.sqlite')]))
        e=portable_manifest(dict(datasets=[dict(name='a',weight=1)]),reports,folder=True)['datasets'][0]
        self.assertEqual(e['type'],'prepared_folder');self.assertNotIn('index_path',e)


if __name__=='__main__':unittest.main()
