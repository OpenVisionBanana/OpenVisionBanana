"""Portable regression checks for geometry loading and distributed loss."""
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

import train_dreambooth_partial_flux2_klein_img2img as trainer
from common.replica_geometry import camera_rotation, raised_position
from common.training.masked_reduction import normalized_distributed_loss


def test_diode_candidate_metadata_does_not_bypass_normal_verification():
    from scripts.dataset_prep.prepare_geometry_training import DIODE_NORMAL_TO_CANONICAL
    candidate = dict(DIODE_NORMAL_TO_CANONICAL.metadata(), geometry_version="diode_identity_candidate_unverified_v1")
    contract = trainer.prepared_row_geometry_contract(dict(
        task="surface_normals", source_dataset="diode",
        metadata=candidate))
    assert contract["normal_contract_status"] == "unverified"


@pytest.mark.parametrize('policy', ['auto', 'enabled', 'disabled'])
def test_geometry_requires_validity_even_for_unmasked_loss(policy):
    data = trainer.DreamBoothDataset.__new__(trainer.DreamBoothDataset)
    data.args = SimpleNamespace(loss_mask_policy=policy, allow_missing_geometry_masks=False)
    data.rows = [trainer.PreparedRow('test', None, 'row', 'rgb', 'gt', 'prompt', task='surface_normal_estimation')]
    with pytest.raises(ValueError, match='BOTH'):
        data._validate_loss_mask_policy()
    data.rows[0].loss_mask = 'mask.png'
    data._validate_loss_mask_policy()


def test_unsampled_source_rows_are_checked():
    data = trainer.DreamBoothDataset.__new__(trainer.DreamBoothDataset)
    data.args = SimpleNamespace(loss_mask_policy='auto', allow_missing_geometry_masks=False)
    data.rows = []
    data.dataset_pool = SimpleNamespace(rows=[trainer.PreparedRow('test', None, 'row', 'rgb', 'gt', 'prompt', task='depth_estimation')])
    with pytest.raises(ValueError):
        data._validate_loss_mask_policy()


def test_auto_does_not_require_masks_for_other_tasks():
    data = trainer.DreamBoothDataset.__new__(trainer.DreamBoothDataset)
    data.args = SimpleNamespace(loss_mask_policy='auto', allow_missing_geometry_masks=False)
    data.rows = [trainer.PreparedRow('test', None, 'row', 'rgb', 'gt', 'prompt', task='semseg')]
    data._validate_loss_mask_policy()
    data.args.loss_mask_policy = 'enabled'
    with pytest.raises(ValueError):
        data._validate_loss_mask_policy()


def test_legacy_missing_geometry_mask_is_explicit():
    data = trainer.DreamBoothDataset.__new__(trainer.DreamBoothDataset)
    data.args = SimpleNamespace(loss_mask_policy='disabled', allow_missing_geometry_masks=True)
    data.rows = [trainer.PreparedRow('test', None, 'row', 'rgb', 'gt', 'prompt', task='depth_estimation')]
    data._validate_loss_mask_policy()


def test_auxiliary_requires_task_homogeneous_batches(tmp_path):
    with pytest.raises(ValueError, match='task_homogeneous'):
        trainer.parse_args(['--pretrained_model_name_or_path', 'local-model',
                            '--instance_data_dir', str(tmp_path), '--angular_normal_aux_weight', '.05'])


def test_release_mask_default_is_preserved(tmp_path):
    args = trainer.parse_args(['--pretrained_model_name_or_path', 'local-model', '--instance_data_dir', str(tmp_path)])
    assert args.loss_mask_policy == 'disabled'


def test_empty_mask_is_differentiable_zero():
    x = torch.tensor(3., requires_grad=True)
    loss = normalized_distributed_loss(x * 0, torch.tensor(0.))
    loss.backward()
    assert loss.item() == x.grad.item() == 0


def test_raw_replica_camera_basis_and_height():
    matrix = camera_rotation(0., 0.)
    np.testing.assert_allclose(matrix @ [0., 0., -1.], [0., 1., 0.], atol=1e-7)
    np.testing.assert_allclose(matrix @ [0., 1., 0.], [0., 0., 1.], atol=1e-7)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-7)
    np.testing.assert_allclose(raised_position([1., 2., 3.], 1.5), [1., 2., 4.5])
    np.testing.assert_allclose(raised_position([1., 2., 3.], 1.5, 'y'), [1., 3.5, 3.])


def test_replica_rgb_guard_precedes_output_changes(monkeypatch):
    from dataset_prep.replica import prepare_replica_depth_normals as replica
    parser = SimpleNamespace(parse_args=lambda: SimpleNamespace(allow_vertex_color_rgb=False))
    monkeypatch.setattr(replica, 'build_argparser', lambda: parser)
    with pytest.raises(RuntimeError, match='Ptex'):
        replica.main()


def test_replica_legacy_rgb_requires_explicit_acknowledgement(monkeypatch):
    from dataset_prep.replica import prepare_replica_depth_normals as replica
    parser = SimpleNamespace(parse_args=lambda: SimpleNamespace(
        allow_vertex_color_rgb=True, samples_per_scene=0))
    monkeypatch.setattr(replica, 'build_argparser', lambda: parser)
    with pytest.warns(RuntimeWarning, match='vertex colours'):
        with pytest.raises(ValueError, match='samples-per-scene'):
            replica.main()


class Scalar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, coefficient):
        return self.weight * coefficient


def _distributed_worker(rank, rendezvous, output, empty_rank):
    torch.distributed.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        model = torch.nn.parallel.DistributedDataParallel(Scalar())
        count = 0. if empty_rank and rank == 0 else (2. if rank == 0 else 6.)
        coefficient = count * (1. if rank == 0 else 3.)
        loss = normalized_distributed_loss(model(torch.tensor(coefficient)), torch.tensor(count))
        loss.backward()
        expected = 3. if empty_rank else 2.5
        torch.testing.assert_close(model.module.weight.grad, torch.tensor(expected))
        Path(output + str(rank)).write_text('pass')
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize('empty_rank', [False, True])
def test_two_rank_masked_gradient_matches_global_reference(tmp_path, empty_rank):
    torch.multiprocessing.spawn(_distributed_worker,
        args=(str(tmp_path / 'rendezvous'), str(tmp_path / 'rank'), empty_rank), nprocs=2, join=True)
    assert all((tmp_path / ('rank' + str(rank))).read_text() == 'pass' for rank in range(2))


def test_ppd_frontend_selects_only_its_five_benchmarks(tmp_path, monkeypatch):
    import runpy
    import sys
    from common import release_eval
    calls = []
    monkeypatch.setattr(release_eval, 'locate_dataset', lambda *args: tmp_path)
    monkeypatch.setattr(release_eval, 'launch_sharded', lambda command, *args: calls.append(command))
    monkeypatch.setattr(sys, 'argv', ['evaluate_depth.py', '--data-root', str(tmp_path),
        '--output-dir', str(tmp_path / 'out'), '--benchmark-protocol', 'marigold_v2_ppd',
        '--benchmark-split-root', str(tmp_path), '--datasets', 'all'])
    runpy.run_path(str(Path(trainer.__file__).parent / 'scripts/evaluate_depth.py'), run_name='__main__')
    assert {cmd[cmd.index('--dataset') + 1] for cmd in calls} == {'nyu', 'kitti_eigen', 'eth3d', 'scannet', 'diode'}
    assert all(cmd[cmd.index('--protocol_version') + 1] == 'marigold-v2-ppd-v1' for cmd in calls)


def test_normal_frontend_defaults_to_v6_and_auto_gt(tmp_path, monkeypatch):
    import runpy
    import sys
    from common import release_eval
    calls = []
    monkeypatch.setattr(release_eval, 'locate_dataset', lambda *args: tmp_path)
    monkeypatch.setattr(release_eval, 'launch_sharded', lambda command, *args: calls.append(command))
    monkeypatch.setattr(sys, 'argv', ['evaluate_normals.py', '--data-root', str(tmp_path),
                                   '--output-dir', str(tmp_path / 'out'), '--datasets', 'nyu,sintel'])
    runpy.run_path(str(Path(trainer.__file__).parent / 'scripts/evaluate_normals.py'), run_name='__main__')
    assert len(calls) == 2
    assert all(cmd[cmd.index('--protocol_version') + 1] == 'geometry-eval-v6' for cmd in calls)
    assert all(cmd[cmd.index('--gt_normal_encoding') + 1] == 'auto' for cmd in calls)
