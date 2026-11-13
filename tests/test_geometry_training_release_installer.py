"""Verify portable installation without source datasets or a GPU."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checked_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    Path(str(path) + '.sha256').write_text(_digest(path) + '  ' + path.name + '\n')


def _package(tmp_path, member='metadata.jsonl'):
    root = tmp_path / 'package'
    for directory in ('archives', 'indices', 'manifests'):
        (root / directory).mkdir(parents=True)
    archive = root / 'archives/depth_test.tar'
    with tarfile.open(archive, 'w') as tar:
        payload = b'{}\n'
        item = tarfile.TarInfo(member)
        item.size = len(payload)
        tar.addfile(item, io.BytesIO(payload))
    index = root / 'indices/depth_test.sqlite'
    index.write_bytes(b'checksum fixture; actual loader is tested separately')
    shard = dict(archive=archive.name, index=index.name,
                 archive_sha256=_digest(archive), index_sha256=_digest(index))
    _checked_json(root / 'PACKAGE.json', {'sources': {'depth_test': {'shards': [shard]}}})
    _checked_json(root / 'READY.json', {'status': 'materialized_export_complete', 'package_sha256': _digest(root / 'PACKAGE.json')})
    dataset = dict(name='depth_test', type='indexed_prepared_tar', path='archives/depth_test.tar',
                   index_path='indices/depth_test.sqlite', weight=.3)
    _checked_json(root / 'manifests/base_geometry_indexed.json', {'samples_per_epoch': 1, 'datasets': [dataset]})
    _checked_json(root / 'manifests/base_geometry_folders.json', {'samples_per_epoch': 1, 'datasets': [dict(name='depth_test', path='depth_test', weight=.3)]})
    return root


def _run(root, output, *args):
    return subprocess.run([sys.executable, str(ROOT / 'scripts/dataset_prep/use_geometry_training_release.py'),
                           '--root', str(root), '--output', str(output), *args], capture_output=True, text=True)


def test_verified_package_preserves_other_tasks(tmp_path):
    root = _package(tmp_path)
    mix = tmp_path / 'other.json'
    mix.write_text(json.dumps({'datasets': [dict(name='semseg', path='images', weight=.7),
                                         dict(name='depth_old', path='obsolete', weight=.3)]}))
    output = tmp_path / 'local.json'
    result = _run(root, output, '--other-task-manifest', str(mix))
    assert result.returncode == 0, result.stderr
    entries = json.loads(output.read_text())['datasets']
    assert {entry['name'] for entry in entries} == {'depth_test', 'semseg'}
    assert entries[0]['path'] == str(root / 'archives/depth_test.tar')
    assert entries[0]['index_path'] == str(root / 'indices/depth_test.sqlite')
    assert entries[1]['path'] == str(tmp_path / 'images')
    assert sum(entry['weight'] for entry in entries) == 1.
    assert _run(root, output).returncode != 0  # Existing manifests are not overwritten.


def test_modified_archive_is_rejected_before_writing_manifest(tmp_path):
    root = _package(tmp_path)
    (root / 'archives/depth_test.tar').write_bytes(b'corrupted')
    output = tmp_path / 'local.json'
    result = _run(root, output)
    assert result.returncode != 0 and 'checksum mismatch' in result.stderr
    assert not output.exists()


def test_relative_stream_shards_resolve_against_other_task_manifest(tmp_path):
    root = _package(tmp_path)
    mix = tmp_path / 'other.json'
    mix.write_text(json.dumps({'datasets': [dict(name='generation', type='webdataset', shards='images-{000..003}.tar', weight=.7)]}))
    output = tmp_path / 'local.json'
    result = _run(root, output, '--other-task-manifest', str(mix))
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())['datasets'][1]['shards'] == str(tmp_path / 'images-{000..003}.tar')


def test_missing_shard_record_is_rejected(tmp_path):
    root = _package(tmp_path)
    manifest = {'samples_per_epoch': 1, 'datasets': [dict(name='depth_missing', path='missing.tar', weight=.3)]}
    _checked_json(root / 'manifests/base_geometry_indexed.json', manifest)
    output = tmp_path / 'local.json'
    result = _run(root, output)
    assert result.returncode != 0 and 'complete shard records' in result.stderr
    assert not output.exists()


def test_incomplete_package_is_rejected(tmp_path):
    root = _package(tmp_path)
    _checked_json(root / 'READY.json', {'status': 'in_progress', 'package_sha256': _digest(root / 'PACKAGE.json')})
    output = tmp_path / 'local.json'
    result = _run(root, output)
    assert result.returncode != 0 and 'not complete' in result.stderr
    assert not output.exists()


def test_folder_extraction_rejects_traversal(tmp_path):
    root = _package(tmp_path, '../escape.json')
    output = tmp_path / 'local.json'
    extraction = tmp_path / 'extracted'
    result = _run(root, output, '--extract-to', str(extraction))
    assert result.returncode != 0 and 'Unsafe archive member' in result.stderr
    assert not output.exists()
    assert not (extraction / 'escape.json').exists()
