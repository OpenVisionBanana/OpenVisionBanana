import pytest
from scripts.dataset_prep.build_replica_ptex_candidate import unchanged_target_members


def test_rgb_only_candidate_copies_every_gt_asset_once():
    row=dict(image='target/a.png',loss_mask='mask/a.png',valid_mask='mask/a.png',depth_raw='raw/a.npy',
             raw_normals='raw/a_normal.npy',conditioning_image='old_rgb/a.png')
    assert unchanged_target_members(row)==['target/a.png','mask/a.png','raw/a.npy','raw/a_normal.npy']


def test_rgb_only_candidate_rejects_unsafe_target_names():
    with pytest.raises(ValueError,match='safe relative'):
        unchanged_target_members(dict(image='../outside.png'))
