from __future__ import annotations

import numpy as np
import pytest

from evaluation.depth.ppd_protocol import (
    align_log_depth_ransac,
    prepare_protocol_arrays,
    relative_grayscale_to_log_depth_signal,
    resolve_ppd_alignment,
    score_ppd_split_disparity_affine,
    score_ppd_ransac_log,
)


def test_inverse_grayscale_is_oriented_toward_farther_depth() -> None:
    gray = np.array([[1.0, 0.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        relative_grayscale_to_log_depth_signal(gray, inverse=True),
        np.array([[-1.0, 0.0]], dtype=np.float32),
    )


def test_ransac_log_recovers_positive_affine_transform() -> None:
    y, x = np.mgrid[0:24, 0:32]
    pred_log = (x / 31.0 + y / 23.0).astype(np.float32)
    gt_log = 1.7 * pred_log - 0.4
    mask = np.ones(pred_log.shape, dtype=bool)

    aligned, scale, shift = align_log_depth_ransac(pred_log, gt_log, mask)

    np.testing.assert_allclose(aligned, gt_log, atol=1e-5)
    assert abs(scale - 1.7) < 1e-5
    assert abs(shift + 0.4) < 1e-5


def test_ppd_score_is_perfect_for_affine_log_depth() -> None:
    gt = np.linspace(0.2, 9.0, 20 * 30, dtype=np.float32).reshape(20, 30)
    pred_log = (np.log(gt) - 0.7) / 1.3
    metrics, metadata, _ = score_ppd_ransac_log(
        "nyu", pred_log, gt, np.ones(gt.shape, dtype=bool)
    )

    assert metrics["abs_rel"] < 1e-5
    assert metrics["delta1"] == 1.0
    assert metadata["valid_pixels"] == gt.size


def test_ppd_split_disparity_affine_recovers_inverse_target() -> None:
    gt = np.linspace(0.2, 9.0, 20 * 30, dtype=np.float32).reshape(20, 30)
    inverse_signal = (1.0 / gt - 0.1) / 1.7
    metrics, metadata = score_ppd_split_disparity_affine(
        "nyu", inverse_signal, gt, np.ones(gt.shape, dtype=bool)
    )
    assert metrics["abs_rel"] < 1e-5
    assert metrics["delta1"] == 1.0
    assert metadata["protocol"] == "ppd_split_disparity_affine_v1"


def test_diode_bounds_and_gradient_filter_are_applied() -> None:
    gt = np.ones((8, 8), dtype=np.float32)
    gt[:, 4:] = 4.0
    pred = np.zeros_like(gt)
    _, mask, _ = prepare_protocol_arrays("diode", gt, np.ones_like(gt), pred)

    assert not mask[:, 3:5].any()
    assert mask[:, :2].all()


@pytest.mark.parametrize('encoding, expected', [
    ('relative_inverse_grayscale', 'disparity_affine'),
    ('vision_banana_rgb', 'log_depth_ransac'),
    ('barron_scalar_grayscale', 'log_depth_ransac'),
])
def test_primary_alignment_follows_declared_encoding(encoding, expected):
    assert resolve_ppd_alignment(encoding) == expected


def test_legacy_log_alignment_is_explicitly_available():
    assert resolve_ppd_alignment('relative_inverse_grayscale', 'log_depth_ransac') == 'log_depth_ransac'


@pytest.mark.parametrize('encoding', ['relative_linear_grayscale', 'grayscale'])
def test_unknown_log_interpretation_is_not_automatic(encoding):
    with pytest.raises(ValueError, match='No encoding-aware'):
        resolve_ppd_alignment(encoding)
    with pytest.raises(ValueError, match='requires relative_inverse'):
        resolve_ppd_alignment(encoding, 'disparity_affine')


def test_perfect_inverse_target_must_not_be_interpreted_as_log_depth():
    gt = np.linspace(2., 79., 4096, dtype=np.float32).reshape(64, 64)
    inv = 1. / gt
    gray = (inv - inv.min()) / (inv.max() - inv.min())
    mask = np.ones(gt.shape, bool)
    wrong, _, _ = score_ppd_ransac_log('kitti_eigen', -gray, gt, mask)
    correct, _ = score_ppd_split_disparity_affine('kitti_eigen', gray, gt, mask)
    assert wrong['delta1'] < .95
    assert correct['delta1'] == 1.
    assert correct['abs_rel'] < 1e-5


@pytest.mark.parametrize('alignment', ['auto', 'log_depth_ransac', 'disparity_affine'])
def test_frontend_forwards_primary_alignment(tmp_path, monkeypatch, alignment):
    import runpy
    import sys
    from common import release_eval
    calls = []
    monkeypatch.setattr(release_eval, 'locate_dataset', lambda *args: tmp_path)
    monkeypatch.setattr(release_eval, 'launch_sharded', lambda cmd, *args: calls.append(cmd))
    monkeypatch.setattr(sys, 'argv', ['evaluate_depth.py', '--model', 'local-model',
        '--data-root', str(tmp_path), '--output-dir', str(tmp_path/'out'),
        '--datasets', 'nyu', '--benchmark-protocol', 'marigold_v2_ppd',
        '--benchmark-split-root', str(tmp_path), '--ppd-primary-alignment', alignment])
    from pathlib import Path
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/evaluate_depth.py'), run_name='__main__')
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[cmd.index('--ppd_primary_alignment')+1] == alignment


@pytest.mark.parametrize('alignment, expected_delta1', [
    ('disparity_affine', .98), ('log_depth_ransac', .85),
])
def test_summary_primary_metrics_and_labels_match_selection(tmp_path, alignment, expected_delta1):
    # Execute the real summary branch without importing GPU inference packages.
    import ast
    import json
    from pathlib import Path
    from types import SimpleNamespace
    source = Path(__file__).resolve().parents[1]/'evaluation/depth/inference_partial_flux2_klein_img2img.py'
    main = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'main')
    branch = [n for n in main.body if isinstance(n, ast.If)
              and ast.unparse(n.test) == "args.benchmark_protocol == 'marigold_v2_ppd'"][-1]
    function = ast.FunctionDef(name='write_summary', args=ast.arguments(
        posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=branch.body, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    record = {}
    for prefix, delta in [('marigold_v2_ppd', .85), ('ppd_split_disparity_affine', .98)]:
        record.update({f'{prefix}_{name}': value for name,value in
                       [('abs_rel', .1), ('delta1', delta), ('delta2', .99), ('delta3', 1.)]})
    args = SimpleNamespace(dataset='nyu', split='test', depth_encoding='relative_inverse_grayscale',
        model='local', model_family='klein', transformer_path=None, lora_path=None,
        steps=28, guidance_scale=3.5, legacy_unseeded=False, seed=0,
        benchmark_split_root=str(tmp_path), num_shards=1, ppd_primary_alignment=alignment)
    env = dict(np=np, json=json, Path=Path, args=args, per_sample_records=[record],
        ppd_primary_alignment=alignment, records_by_case=lambda rows: {'standard': rows},
        prediction_dir=None, prompt='depth', prompt_sha256=lambda value: 'test-hash',
        dataset_root=tmp_path, resolved_splits=['test'], resolved_cases=['standard'],
        subset_metadata=None, out_dir=tmp_path, MARIGOLD_V2_PPD_PROTOCOL='marigold_v2_ppd_ransac_log_v1',
        RELATIVE_DEPTH_ENCODINGS={'relative_inverse_grayscale', 'relative_linear_grayscale'},
        DEPTH_ENCODING_RELATIVE_INVERSE_GRAYSCALE='relative_inverse_grayscale')
    exec(compile(module, str(source), 'exec'), env)
    env['write_summary']()
    summary = json.loads((tmp_path/'nyu_test_results.json').read_text())
    assert summary['primary_metrics']['delta1'] == expected_delta1
    assert summary['primary_metrics_by_case']['standard']['delta1'] == expected_delta1
    assert summary['marigold_v2_ppd_metrics']['delta1'] == .85
    assert summary['ppd_split_disparity_affine_metrics']['delta1'] == .98
    assert not summary['depth_evaluation']['log_lane_encoding_compatible']
    assert ('disparity' in summary['depth_evaluation']['primary_lane']) == (alignment == 'disparity_affine')
