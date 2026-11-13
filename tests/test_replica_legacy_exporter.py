"""Local legacy-package test. Deliberately outside the official-source release inventory."""
import json
import pytest

from scripts.dataset_prep.package_replica_ptex_correction import localize_manifests, write


def test_new_manifest_changes_geometry_paths_only_and_preserves_weights(tmp_path):
    parent = tmp_path/'old'/'upload'; parent.mkdir(parents=True)
    old = parent.parent/'local_manifests'; old.mkdir()
    payload = {'seed': 42, 'datasets': [
        {'name': 'depth_replica', 'path': str(parent/'archives/depth.tar'),
         'index_path': str(parent/'indices/depth.sqlite'), 'weight': .1},
        {'name': 'surface_normals_replica', 'path': str(parent/'archives/normals.tar'), 'weight': .2},
        {'name': 'semantic_segmentation', 'path': '/unchanged/seg.tar', 'weight': .7}]}
    write(old/'base_full.json', payload)
    out = tmp_path/'new'/'upload'
    localize_manifests(parent, out)
    result = json.loads((out.parent/'local_manifests/base_full.json').read_text())
    assert result['datasets'][0]['path'] == str(out/'archives/depth.tar')
    assert result['datasets'][0]['index_path'] == str(out/'indices/depth.sqlite')
    assert result['datasets'][2] == payload['datasets'][2]
    assert [e['weight'] for e in result['datasets']] == [.1, .2, .7]
    localize_manifests(parent, out)
    result['seed'] = 123
    write(out.parent/'local_manifests/base_full.json', result)
    with pytest.raises(ValueError, match='refusing to replace'):
        localize_manifests(parent, out)
