from scripts.analysis.recheck_geometry_readiness import geometry_members


def test_legacy_valid_mask_is_relative_to_metadata_directory():
    row = {'image': 'hypersim/target.png', 'conditioning_image': 'hypersim/input.png',
           'valid_mask': 'valid_masks/0.png', '_metadata_member': 'hypersim/metadata.jsonl'}
    assert geometry_members(row, {'hypersim/valid_masks/0.png'})['mask'] == 'hypersim/valid_masks/0.png'


def test_current_loss_mask_is_not_silently_rebased():
    row = {'loss_mask': 'missing.png', 'valid_mask': 'valid.png',
           '_metadata_member': 'dataset/metadata.jsonl'}
    assert geometry_members(row, {'dataset/missing.png', 'dataset/valid.png'})['mask'] == 'missing.png'


def test_tar_rooted_legacy_mask_is_preserved():
    row = {'valid_mask': 'valid_masks/0.png', '_metadata_member': 'dataset/metadata.jsonl'}
    assert geometry_members(row, {'valid_masks/0.png'})['mask'] == 'valid_masks/0.png'


def test_missing_legacy_mask_stays_missing():
    row = {'valid_mask': 'missing.png', '_metadata_member': 'dataset/metadata.jsonl'}
    assert geometry_members(row, set())['mask'] == 'missing.png'
