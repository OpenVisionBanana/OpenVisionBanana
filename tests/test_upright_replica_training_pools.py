import copy

import pytest

from scripts.dataset_prep.prepare_upright_replica_training_pools import replace_replica


def test_upright_replacement_preserves_other_sources_weights_and_original():
    original = dict(datasets=[
        dict(name='depth_legacy_replica', path='/old-depth', index_path='/old-depth-index', weight=.12),
        dict(name='surface_normals_legacy_replica_camera_facing_v4', path='/old-normal', index_path='/old-normal-index', weight=.13),
        dict(name='depth_hypersim', path='/unchanged', index_path='/unchanged-index', weight=.75),
    ], depth_target_encoding='relative_inverse_grayscale')
    before = copy.deepcopy(original)
    replacements = {task: dict(type='indexed_prepared_tar', path='/new-'+task,
                              index_path='/new-'+task+'-index')
                    for task in ('depth', 'surface_normals')}
    result = replace_replica(original, replacements)
    assert original == before
    assert result['datasets'][2] == original['datasets'][2]
    assert [e['weight'] for e in result['datasets']] == [e['weight'] for e in original['datasets']]
    assert result['datasets'][0]['path'] == '/new-depth'
    assert result['datasets'][1]['path'] == '/new-surface_normals'
    assert result['depth_target_encoding'] == original['depth_target_encoding']


def test_upright_replacement_rejects_missing_or_duplicate_sources():
    replacements = {task: dict(type='indexed_prepared_tar', path='/new', index_path='/index')
                    for task in ('depth', 'surface_normals')}
    with pytest.raises(ValueError, match='exactly one'):
        replace_replica(dict(datasets=[]), replacements)
