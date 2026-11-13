import copy
import json

import numpy as np
from PIL import Image
import pytest

from scripts.dataset_prep.curate_zs_geometry_pairs import (
    REPLICA, difference_hash, joint_features, pose_cell, select_diverse,
)


def candidate(name, cell, score=1., color=(80., 100., 130.), dhash=0):
    return dict(id=name, scene='room', pose_cell=cell, quality_score=score,
                rgb_mean=color, dhash=dhash)


def test_selection_reproducible_and_scene_cap_is_not_a_fill_quota():
    policy = {**REPLICA, 'max_per_scene': 2}
    rows = [candidate('b', (1,), color=(20, 30, 40)),
            candidate('a', (0,), score=2), candidate('c', (2,), color=(150, 160, 170))]
    selected, excluded = select_diverse(rows, policy)
    again, _ = select_diverse(list(reversed(rows)), policy)
    assert selected == again
    assert [r['id'] for r in selected] == ['a', 'b']
    assert excluded == {'c': ['scene_cap']}
    assert len(select_diverse(rows[:1], policy)[0]) == 1


def test_diversity_requires_colour_similarity_for_appearance_duplicate():
    rows = [candidate('a', (0,), score=3), candidate('b', (1,), score=2),
            candidate('c', (2,), color=(180., 190., 200.))]
    selected, excluded = select_diverse(rows)
    assert [r['id'] for r in selected] == ['a', 'c']
    assert excluded == {'b': ['similar_appearance']}


def test_pose_bin_handles_yaw_wrap_and_negative_positions():
    row = dict(metadata=dict(camera_position=[-.1, 0, .8], yaw_degrees=360, pitch_degrees=0))
    assert pose_cell(row) == (-1, 0, 1, 0, 3)
    with pytest.raises(ValueError, match='recorded camera'):
        pose_cell({})


def test_difference_hash_is_deterministic():
    image = Image.fromarray(np.arange(64, dtype='uint8').reshape(8, 8)).convert('RGB')
    assert difference_hash(image) == difference_hash(image.copy())


def test_joint_crop_keeps_interior_hole_and_uses_both_masks():
    row = dict(id='a', image='gt', conditioning_image='rgb', loss_mask='mask')
    result = dict(hard_reasons=[], review_flags=[], original_size=[100, 100],
                  rgb_hash_32='same', scene='s', signals=dict(laplacian_variance=10))
    mask_depth = np.ones((100, 100), dtype='uint8')*255
    mask_normal = mask_depth.copy(); mask_depth[:, :5] = 0
    mask_normal[:5, :] = 0; mask_normal[50, 50] = 0

    class Reader:
        def __init__(self, mask): self.mask = mask
        def image(self, row, name, mode='RGB'):
            if name == 'mask': return Image.fromarray(self.mask)
            return Image.fromarray(np.tile(np.arange(100, dtype='uint8')[None, :, None], (100, 1, 3)))

    item = (0, result, row)
    features = joint_features('a', item, item, (Reader(mask_depth), Reader(mask_normal)), 'hypersim')
    assert features['crop_box'] == [5, 5, 100, 100]
    assert features['invalid_after'] == pytest.approx(1/(95*95))
    assert features['reasons'] == []
    bad = copy.deepcopy(result); bad['rgb_hash_32'] = 'different'
    assert joint_features('a', item, (0, bad, row), (), 'hypersim')['reasons'] == ['depth_normal_RGB_differ']
    rough = copy.deepcopy(result); rough['review_flags'] = ['rough_normal_field_at_review_resolution']
    flagged = joint_features('a', item, (0, rough, row), (Reader(mask_depth), Reader(mask_normal)), 'hypersim')
    assert flagged['reasons'] == ['rough_normal_field_conservative_quality_filter']


def test_build_replaces_manifest_datasets_without_overwriting_originals(tmp_path, monkeypatch):
    from scripts.dataset_prep import curate_zs_geometry_pairs as module
    audit = tmp_path/'audit'; audit.mkdir()
    (audit/'plan.json').write_text('{}')
    depth = dict(name='depth_hypersim', type='indexed_prepared_tar', path='depth.tar',
                 index_path='depth.sqlite', weight=.3, dataset='hypersim', lane='current')
    normal = dict(name='surface_normals_hypersim', type='indexed_prepared_tar', path='normal.tar',
                  index_path='normal.sqlite', weight=.4, dataset='hypersim', lane='current')
    other = dict(name='identity', type='indexed_prepared_tar', path='identity.tar',
                 index_path='identity.sqlite', weight=.3)
    source = tmp_path/'source.json'
    source.write_text(json.dumps(dict(datasets=[module.clean_entry(depth), module.clean_entry(normal), other])))
    original = source.read_bytes()
    plan = dict(sources=[depth, normal], public_zip_assets=[], source_manifest=str(source))
    monkeypatch.setattr(module, 'checked_plan', lambda path: plan)
    row = dict(id='frame', metadata={})
    records = {'frame':(0, dict(scene='s'), row)}
    monkeypatch.setattr(module, 'load_records', lambda *args: records)
    monkeypatch.setattr(module, 'Reader', lambda *args: None)
    monkeypatch.setattr(module, 'joint_features', lambda *args: dict(id='frame', scene='s', reasons=[],
        invalid_after=0., retained_area=1., quality_score=1., crop_box=[0, 0, 100, 100]))
    records['frame'][1]['rgb_hash_32'] = 'hash'
    views = []
    def write_view(entry, destination, selected, variant):
        views.append((entry, selected)); return {**entry, 'index_path':str(destination)}
    monkeypatch.setattr(module, 'write_view', write_view)
    output = tmp_path/'new_subset'
    module.build(audit, output, workers=1)
    assert source.read_bytes() == original
    manifest = json.loads((output/'manifests/zeroshot_unmasked_full.json').read_text())
    assert [e['weight'] for e in manifest['datasets']] == pytest.approx([.3, .4, .3])
    assert manifest['datasets'][2] == other
    assert len(views) == 4
    for entry, selected in views:
        selected_row = json.loads(selected[0][1])
        assert selected_row['training_crop_box'] == [0, 0, 100, 100]
        assert selected_row['metadata']['training_curation']['joint_depth_normal_frame']
    assert (output/'PREPARED.json').exists()
    assert not (output/'READY.json').exists()
