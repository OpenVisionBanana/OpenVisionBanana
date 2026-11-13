"""Upstream scope compatibility must not undo the geometry release safeguards."""
from types import SimpleNamespace

import pytest
import torch

import train_dreambooth_partial_flux2_klein_img2img as trainer


class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.single_transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        for name in ('norm_out', 'proj_out', 'pos_embed', 'x_embedder',
                     'context_embedder', 'time_guidance_embed',
                     'double_stream_modulation_img', 'double_stream_modulation_txt',
                     'single_stream_modulation'):
            setattr(self, name, torch.nn.Linear(2, 2))


def scope_args(**overrides):
    values = dict(train_last_transformer_blocks=0, train_last_single_transformer_blocks=1,
                  train_output_layers=True, train_input_layers=False,
                  train_non_block_layers=False, train_all_transformer_blocks=False,
                  train_full_transformer=False)
    return SimpleNamespace(**(values | overrides))


@pytest.mark.parametrize('contract', ['separate_embeddings', 'release_colorexp'])
@pytest.mark.parametrize('inputs', [False, True])
@pytest.mark.parametrize('non_block', [False, True])
def test_parameter_scope_contract(contract, inputs, non_block):
    model = TinyTransformer()
    selected, trainable, total = trainer.configure_partial_transformer_training(
        model, scope_args(trainable_scope_contract=contract,
                          train_input_layers=inputs, train_non_block_layers=non_block))
    expected = {'single_transformer_blocks.0', 'norm_out', 'proj_out'}
    if non_block:
        expected |= {'time_guidance_embed', 'double_stream_modulation_img',
                     'double_stream_modulation_txt', 'single_stream_modulation'}
        if contract == 'release_colorexp':
            expected |= {'x_embedder', 'context_embedder'}
    if inputs:
        expected |= {'pos_embed', 'x_embedder', 'context_embedder'}
        if contract == 'release_colorexp':
            expected.add('time_guidance_embed')
    assert set(selected) == expected
    assert len(selected) == len(expected)
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == (name.rsplit('.', 1)[0] in expected)
    assert trainable == len(expected) * 6
    assert total == 11 * 6


def test_existing_args_preserve_release_embeddings():
    model = TinyTransformer()
    trainer.configure_partial_transformer_training(model, scope_args(train_non_block_layers=True))
    assert model.x_embedder.weight.requires_grad
    assert model.context_embedder.weight.requires_grad
    assert model.time_guidance_embed.weight.requires_grad


@pytest.mark.parametrize('contract', ['separate_embeddings', 'release_colorexp'])
def test_literal_full_scope_is_independent_of_contract(contract):
    model = TinyTransformer()
    selected, trainable, total = trainer.configure_partial_transformer_training(
        model, scope_args(trainable_scope_contract=contract, train_full_transformer=True))
    assert selected == ['<all transformer parameters>']
    assert trainable == total
    assert all(p.requires_grad for p in model.parameters())


def test_scope_parser_defaults_and_override(tmp_path):
    required = ['--pretrained_model_name_or_path', 'local-model',
                '--instance_data_dir', str(tmp_path)]
    assert trainer.parse_args(required).trainable_scope_contract == 'release_colorexp'
    assert trainer.parse_args(required + ['--trainable_scope_contract', 'release_colorexp']).trainable_scope_contract == 'release_colorexp'


def test_unknown_scope_fails_before_parameter_mutation():
    model = TinyTransformer()
    with pytest.raises(ValueError, match='scope contract'):
        trainer.configure_partial_transformer_training(model, scope_args(trainable_scope_contract='unknown'))
    assert all(p.requires_grad for p in model.parameters())


def test_local_model_card_never_loads_hub_metadata(tmp_path, monkeypatch):
    def unexpected_hub_call(*args, **kwargs):
        raise AssertionError('Local training tried to access Hub metadata')
    monkeypatch.setattr(trainer, 'load_or_create_model_card', unexpected_hub_call)
    trainer.save_model_card('local-' + 'long-name-' * 30, base_model='local-base',
                            instance_prompt='Predict geometry.', repo_folder=str(tmp_path))
    assert (tmp_path / 'README.md').is_file()


def test_legacy_valid_mask_retained_for_prepared_folders(tmp_path):
    import json
    (tmp_path / 'metadata.jsonl').write_text(json.dumps({
        'id': 'row', 'image': 'target.png', 'conditioning_image': 'input.png',
        'text': 'Predict geometry.', 'task': 'depth_estimation', 'valid_mask': 'mask.png',
    }) + '\n')
    index = trainer.PreparedDatasetIndex(trainer.DatasetSpec(name='legacy', path=tmp_path, weight=1.),
                                         'image', 'conditioning_image', 'text')
    assert index.rows[0].loss_mask == 'mask.png'


def test_manifest_relative_paths_retained(tmp_path, monkeypatch):
    import json
    data = tmp_path / 'data'; data.mkdir()
    (data / 'metadata.jsonl').write_text('{}\n')
    manifest = tmp_path / 'mix.json'
    manifest.write_text(json.dumps({'samples_per_epoch': 1,
                                   'datasets': [{'name': 'relative', 'path': 'data', 'weight': 1.}]}))
    elsewhere = tmp_path / 'elsewhere'; elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    specs, _, _, _ = trainer.load_dataset_specs(str(manifest))
    assert specs[0].path == data.resolve()
