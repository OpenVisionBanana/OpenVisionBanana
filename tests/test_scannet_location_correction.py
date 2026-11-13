from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from common.indexed_tar import archive_fingerprint
from common.training.asset_lock import digest
from common.training.benchmark_exclusions import exclusion_reason, load_registry, scannet_location_id, source_identity
from scripts.dataset_prep.apply_scannet_location_correction import (
    build, independent_locations, finalize_loader_checks, verify_corrected_pools,
)


def row(capture, frame=120):
    return {'id': f'{capture}_{frame:06d}', 'scene': capture, 'source_dataset': 'scannetv2',
            'task': 'depth_estimation', 'loss_mask': 'mask.png',
            'metadata': {'scene': capture}, 'training_crop_box': [0, 8, 640, 472]}


def splits(tmp_path):
    root = tmp_path/'splits'
    for relative in (
        'scannet_normals/scannet_test.txt', 'scannet_depth/scannet_val_sampled_list_800_1.txt',
        'nyu_normals/nyuv2_test.txt', 'nyu_depth/labeled/filename_list_test.txt',
        'kitti_depth/eigen_test_files_with_gt.txt', 'diode_depth/diode_val_all_filename_list.txt',
        'eth3d_depth/eth3d_filename_list.txt', 'ibims_normals/ibims_test.txt', 'sintel_normals/sintel_filtered.txt',
    ):
        p = root/relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('')
    (root/'scannet_normals/scannet_test.txt').write_text('scene0001_01/000120_rgb.png normal.npy mask.png\n')
    (root/'scannet_depth/scannet_val_sampled_list_800_1.txt').write_text('scene0002_02/000240_rgb.png depth.png\n')
    return root


def test_repeat_recording_and_exact_frame_reasons():
    registry = {'scannet': {'ids': {'scene0001_01_000120'}, 'scenes': {'scene0001_01'}}}
    assert exclusion_reason(row('scene0001_01'), registry) == 'exact_evaluation_frame'
    assert exclusion_reason(row('scene0001_01', 121), registry) == 'evaluation_scene'
    assert exclusion_reason(row('scene0001_00'), registry) == 'evaluation_location'
    assert exclusion_reason(row('scene0001_02'), registry) == 'evaluation_location'
    assert exclusion_reason(row('scene0003_00'), registry) is None


def test_union_of_normal_and_depth_locations(tmp_path):
    root = splits(tmp_path)
    registry, _ = load_registry(root)
    assert registry['scannet']['locations'] == {'scene0001', 'scene0002'}
    assert independent_locations(root)[0] == registry['scannet']['locations']
    assert exclusion_reason(row('scene0002_00'), registry) == 'evaluation_location'
    assert exclusion_reason(row('scene0001_00'), registry) == 'evaluation_location'


@pytest.mark.parametrize('bad', ['', 'scene1_00', 'scene0001_00_123', 'unknown', 'scene0001_0'])
def test_malformed_location_fails_closed(bad):
    with pytest.raises(ValueError, match='Ambiguous ScanNet'):
        scannet_location_id(bad)


def test_conflicting_or_missing_source_identity_fails_closed():
    inconsistent = row('scene0001_00')
    inconsistent['metadata']['scene'] = 'scene0002_00'
    with pytest.raises(ValueError, match='Conflicting ScanNet'):
        source_identity(inconsistent)
    ambiguous = row('scene0001_00')
    ambiguous['id'] = 'unknown'
    with pytest.raises(ValueError, match='Ambiguous ScanNet'):
        source_identity(ambiguous)


def test_non_scannet_exclusions_unchanged():
    registry = {'nyu': {'ids': {'nyu_0003'}, 'scenes': set()},
                'kitti': {'ids': set(), 'scenes': {'drive'}},
                'diode': {'ids': set(), 'scenes': {'outdoor/scan'}}}
    assert exclusion_reason({'id': 'nyu_0003', 'source_dataset': 'nyuv2'}, registry) == 'exact_evaluation_frame'
    assert exclusion_reason({'id': 'frame', 'scene': 'drive', 'source_dataset': 'kitti'}, registry) == 'evaluation_scene'
    assert exclusion_reason({'id': 'frame', 'scene': 'outdoor/scan', 'source_dataset': 'diode'}, registry) == 'evaluation_scene'


def test_empty_or_malformed_scannet_split_fails_closed(tmp_path):
    root = splits(tmp_path)
    p = root/'scannet_depth/scannet_val_sampled_list_800_1.txt'
    p.write_text('')
    with pytest.raises(ValueError, match='Empty ScanNet'):
        independent_locations(root)
    p.write_text('unknown/000240_rgb.png depth.png\n')
    with pytest.raises(ValueError, match='Ambiguous ScanNet'):
        load_registry(root)


def fixture_parent(tmp_path, captures):
    archive = tmp_path/'original.tar'
    archive.write_bytes(b'fixture unchanged archive')
    index = tmp_path/'original.sqlite'
    with sqlite3.connect(index) as conn:
        conn.execute('create table metadata(key text primary key, value text)')
        conn.execute('create table rows(row_number integer primary key, payload text)')
        conn.execute('create table members(name text, offset integer, size integer)')
        conn.execute('insert into members values (?,?,?)', ('mask.png', 0, 1))
        conn.executemany('insert into rows values (?,?)', [(n, json.dumps(row(c))) for n, c in enumerate(captures)])
        conn.executemany('insert into metadata values (?,?)', [
            ('archive_fingerprint', json.dumps(archive_fingerprint(archive, include_sha256=False))),
            ('num_rows', str(len(captures))), ('schema_version', '1'),
        ])
    parent = tmp_path/'parent.json'
    parent.write_text(json.dumps({'seed': 42, 'mix_contract': {'nzs_policy': 'recording exclusion'}, 'datasets': [
        {'name': 'depth_scannetv2_train', 'type': 'indexed_prepared_tar', 'path': str(archive),
         'index_path': str(index), 'weight': 0.123456789},
        {'name': 'depth_replica_train', 'type': 'indexed_prepared_tar', 'path': 'unchanged.tar',
         'index_path': 'unchanged.sqlite', 'weight': 0.876543211},
    ]}))
    return parent, index, archive


def test_additive_views_preserve_sources_weights_and_crops(tmp_path):
    root = splits(tmp_path)
    parent, source, archive = fixture_parent(tmp_path, ['scene0001_00', 'scene0002_00', 'scene0003_00'])
    hashes = [digest(p) for p in (parent, source, archive)]
    output = tmp_path/'corrected'
    report = build({'full.json': parent}, root, output)
    assert report['independent_zero_location_overlap_audit']
    stats = report['manifests']['full.json']['scannet']['depth_scannetv2_train']
    assert stats['rows_before'] == 3 and stats['rows_after'] == 1
    assert stats['excluded'] == {'evaluation_location': 2}
    assert [digest(p) for p in (parent, source, archive)] == hashes
    original = json.loads(parent.read_text())
    corrected = json.loads((output/'manifests/full.json').read_text())
    assert corrected['datasets'][1] == original['datasets'][1]
    assert corrected['datasets'][0]['weight'] == original['datasets'][0]['weight']
    with sqlite3.connect(corrected['datasets'][0]['index_path']) as conn:
        kept = json.loads(conn.execute('select payload from rows').fetchone()[0])
        assert kept == row('scene0003_00')
        assert conn.execute('select * from members').fetchall() == [('mask.png', 0, 1)]
    # Idempotent frozen rerun; cannot replace an existing manifest/report.
    assert build({'full.json': parent}, root, output) == report


def test_correction_refuses_empty_training_source(tmp_path):
    root = splits(tmp_path)
    parent, _, _ = fixture_parent(tmp_path, ['scene0001_00'])
    with pytest.raises(ValueError, match='No ScanNet rows remain'):
        build({'full.json': parent}, root, tmp_path/'corrected')


def finalized_fixture(tmp_path):
    root = splits(tmp_path)
    parent, _, _ = fixture_parent(tmp_path, ['scene0001_00', 'scene0003_00'])
    output = tmp_path/'corrected'
    labels = {'plane': 'plane/nonzero_geometry.json',
              'base': 'curated/nonzero_trim_invalid25_geometry.json',
              'high': 'quality_tiers/nonzero_high_geometry.json',
              'baseline_v4': 'baseline_v4/nonzero_geometry.json'}
    report = build({label: parent for label in labels.values()}, root, output)
    (output/'loader').mkdir()
    for name, label in labels.items():
        path = output/'loader'/f'{name}.json'
        path.write_text(json.dumps({'status': 'PASS', 'manifest_sha256': digest(output/'manifests'/label),
            'total_samples': 1, 'samples': [{'dataset': 'depth_scannetv2_train',
                                           'both_modes_identical_preprocessing': True}]}))
    return output, report


def test_finalize_binds_loader_evidence_and_manifest(tmp_path):
    output, report = finalized_fixture(tmp_path)
    evidence = output/'loader/base.json'
    original = evidence.read_text()
    payload = json.loads(original)
    payload['manifest_sha256'] = 'wrong'
    evidence.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='Bound real-loader'):
        finalize_loader_checks(output)
    assert not (output/'READY.json').exists()
    evidence.write_text(original)
    ready = finalize_loader_checks(output)
    assert ready['requires_fresh_training_for_heldout_claim']
    manifest = Path(report['manifests']['plane/nonzero_geometry.json']['manifest'])
    assert verify_corrected_pools(output, {'nonzero': manifest}) == ready
    evidence.write_text(original+' ')
    with pytest.raises(ValueError, match='loader evidence changed'):
        verify_corrected_pools(output, {'nonzero': manifest})


def test_fresh_preflight_rejects_unproven_manifest_and_mutated_index(tmp_path):
    output, report = finalized_fixture(tmp_path)
    finalize_loader_checks(output)
    info = report['manifests']['plane/nonzero_geometry.json']
    with pytest.raises(ValueError, match='not the verified ScanNet correction'):
        verify_corrected_pools(output, {'nonzero': Path(info['parent'])})
    view = Path(info['scannet']['depth_scannetv2_train']['corrected_index'])
    with sqlite3.connect(view) as conn:
        conn.execute('delete from rows')
    with pytest.raises(ValueError, match='index changed'):
        verify_corrected_pools(output, {'nonzero': Path(info['manifest'])})
