"""Do not silently regenerate the low-fidelity Replica RGB release."""
import pytest

from dataset_prep.replica.prepare_replica_depth_normals import build_argparser, main


def test_vertex_render_is_explicit_opt_in_before_output_cleanup(monkeypatch, tmp_path):
    depth, normals = tmp_path/'depth', tmp_path/'normals'
    depth.mkdir(); (depth/'keep.txt').write_text('original')
    monkeypatch.setattr('sys.argv', ['prepare_replica', '--replica-root', str(tmp_path),
        '--depth-out-root', str(depth), '--normals-out-root', str(normals), '--overwrite'])
    with pytest.raises(RuntimeError, match='vertex-colour RGB'):
        main()
    assert (depth/'keep.txt').read_text() == 'original'
    assert not normals.exists()


def test_legacy_option_is_available_without_changing_training_defaults():
    parser = build_argparser()
    args = parser.parse_args(['--replica-root', '/unused', '--depth-out-root', '/unused/depth',
                              '--normals-out-root', '/unused/normals', '--allow-vertex-color-rgb'])
    assert args.allow_vertex_color_rgb

