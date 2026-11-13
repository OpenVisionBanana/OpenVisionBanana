import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/analysis/audit_zs_geometry_quality.py'
spec = importlib.util.spec_from_file_location('zs_quality', path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_zip_offset_reads_exact_stored_payload(tmp_path):
    path = tmp_path / 'example.zip'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as z:
        z.writestr('a.tar', b'zero-copy payload')
    with zipfile.ZipFile(path) as z:
        offset = audit.zip_offset(path, z.getinfo('a.tar'))
    with path.open('rb') as f:
        f.seek(offset)
        assert f.read(len(b'zero-copy payload')) == b'zero-copy payload'


def test_compressed_zip_is_not_silently_random_accessed(tmp_path):
    path = tmp_path / 'compressed.zip'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('a.tar', b'payload')
    with zipfile.ZipFile(path) as z:
        with pytest.raises(ValueError, match='uncompressed'):
            audit.zip_offset(path, z.getinfo('a.tar'))


def test_gray_pixels_are_not_used_as_mask():
    class Reader:
        def image(self, *args):
            from PIL import Image
            return Image.new('RGB', (64,64), (128,128,128))
    _, result, _ = audit.inspect((0,json.dumps(dict(id='a', image='gt', conditioning_image='rgb'))),
                                 dict(name='depth_replica'), Reader())
    result = json.loads(result)
    assert result['hard_reasons'] == ['unreadable_or_misaligned_triplet']
    assert 'No declared validity mask' in result['error']


def test_textureless_scene_is_not_automatically_removed():
    result = dict(hard_reasons=[], review_flags=['low_texture_or_blur'],
                  invalid_after=0.03, retained_area=1., edge_warnings=[])
    assert audit.rejection(result,'base') == []
    assert audit.rejection(result,'high') == []
    result['review_flags'].append('mostly_very_close_view')
    assert audit.rejection(result,'base') == []
    assert audit.rejection(result,'high') == ['quality:mostly_very_close_view']


def test_high_filters_coverage_and_never_fills_interior_holes():
    result = dict(hard_reasons=[], review_flags=[], invalid_after=.2, retained_area=.8,edge_warnings=[])
    assert audit.rejection(result,'base') == []
    assert audit.rejection(result,'high') == ['remaining_invalid_coverage']


def test_paired_plane_checks_camera_basis_and_exposes_axis_flip():
    from common.surface_normal_encoding import encode_surface_normals_rgb
    depth = np.full((64,64), 2., dtype=np.float32)
    valid = np.ones_like(depth, dtype=bool)
    normals = np.zeros((64,64,3),dtype=np.float32); normals[...,2] = 1.
    row = dict(metadata=dict(hfov_degrees=90.))
    rgb = encode_surface_normals_rgb(normals,valid)
    stats = audit.paired_geometry(rgb,{},depth,row,valid,'vision_banana_x_left_red')
    assert stats['status']=='checked' and stats['mean_angle'] < 1.
    normals[...,2] = -1.
    rgb = encode_surface_normals_rgb(normals,valid)
    stats = audit.paired_geometry(rgb,{},depth,row,valid,'vision_banana_x_left_red')
    assert stats['mean_angle'] > 179. and stats['best_axis_mean'] < 1.


def test_unknown_intrinsics_are_not_guessed():
    assert audit.camera_intrinsics({},512,512) is None


def test_replica_revision_distinguishes_reversed_and_camera_facing_normals():
    from scripts.analysis.audit_replica_normal_revision import compare_fields
    new = np.zeros((8, 8, 3), dtype=np.float32)
    new[..., 2] = 1.
    valid = np.ones((8, 8), dtype=bool)
    stats = compare_fields(-new, new, valid, (8., 8., 3.5, 3.5))
    assert stats['mean_angle'] == 180.
    assert stats['reversed_pixel_fraction'] == 1.
    assert stats['published_away_from_camera_fraction'] == 1.
    assert stats['current_v4_away_from_camera_fraction'] == 0.
    stats = compare_fields(new, new, valid, None)
    assert stats['mean_angle'] == 0.
    assert stats['reversed_pixel_fraction'] == 0.


def test_report_only_never_creates_filtered_training_views(tmp_path, monkeypatch):
    import sqlite3
    (tmp_path/'scans').mkdir(); (tmp_path/'reports').mkdir(); (tmp_path/'verification').mkdir()
    (tmp_path/'verification/0.json').write_text('{"status":"PASS"}')
    entry = dict(name='published_depth_replica')
    monkeypatch.setattr(audit, 'checked_plan', lambda path: dict(sources=[entry], public_zip_assets=[{}]))
    monkeypatch.setattr(audit, 'gallery', lambda *args: None)
    result = dict(id='a', scene='s', hard_reasons=[], review_flags=[], invalid_before=0., invalid_after=0., retained_area=1.)
    with sqlite3.connect(tmp_path/'scans/published_depth_replica.sqlite') as db:
        db.execute('create table metadata(key text,value text)')
        db.executemany('insert into metadata values (?,?)', [('complete','true'),('full_source','true')])
        db.execute('create table records(row_number integer,result text,payload text)')
        db.execute('insert into records values (?,?,?)', (0,json.dumps(result),'{}'))
    audit.finalize(tmp_path, False, report_only=True)
    assert (tmp_path/'reports/quality_summary.json').exists()
    assert not (tmp_path/'manifests').exists()
    assert not (tmp_path/'indices').exists()
