"""Verified, per-stage receipts for resumable dataset preparation."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tarfile

from common.indexed_tar import verified_sha256


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def files_below(path):
    """Visit staged symlinks too, while excluding cycles and transient files."""
    path = Path(path)
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(path)
    seen = set()
    for directory, dirs, files in os.walk(path, followlinks=True):
        real = Path(directory).resolve()
        if real in seen:
            dirs[:] = []
            continue
        seen.add(real)
        dirs.sort()
        for name in sorted(files):
            if name.endswith((".partial", ".lock")) or name == ".ovb-download-complete":
                continue
            candidate = Path(directory) / name
            if candidate.is_file():
                yield candidate


def input_identity(paths):
    records = []
    for label, path in paths:
        path = Path(path)
        files = list(files_below(path))
        if not files:
            raise ValueError(f"Empty preparation input: {path}")
        for file in files:
            relative = file.relative_to(path).as_posix() if path.is_dir() else file.name
            records.append([label, relative, verified_sha256(file)])
    return dict(digest=digest(records), files=records)


def converter_identity(repo):
    # Shared target/geometry helpers also affect conversion; documentation
    # and tests do not.
    inventory = json.loads((repo / "DATA_RELEASE_FILES.json").read_text())
    paths = [repo / name for name in inventory["conversion_files"]]
    return digest([(str(p.relative_to(repo)), verified_sha256(p)) for p in sorted(set(paths))])


def artifact(path, root):
    if not path.is_file():
        raise FileNotFoundError(path)
    record = dict(path=str(path.relative_to(root)), sha256=verified_sha256(path), bytes=path.stat().st_size)
    if path.suffix == ".tar":
        rows, members = 0, 0
        with tarfile.open(path, "r:") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                members += 1
                if Path(member.name).name == "metadata.jsonl":
                    rows += sum(1 for line in archive.extractfile(member) if line.strip())
        if not members:
            raise ValueError(f"Empty prepared archive: {path}")
        record.update(members=members, rows=rows)
    return record


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@contextmanager
def stage_lock(root, dataset, split):
    folder = root / "manifests/preparation_stages"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{dataset}_{split}.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def run_stage(root, name, identity, outputs, action, overwrite=False):
    """Run under stage_lock. action(force) must write only this stage's outputs."""
    folder = root / "manifests/preparation_stages"
    folder.mkdir(parents=True, exist_ok=True)
    receipt, pending = folder / f"{name}.json", folder / f"{name}.pending.json"
    previous = json.loads(receipt.read_text()) if receipt.is_file() else None
    interrupted = json.loads(pending.read_text()) if pending.is_file() else None
    expected = digest(identity)
    if not overwrite:
        for record in (previous, interrupted):
            if record and record.get("identity_digest") != expected:
                changed = sorted(key for key in set(record.get("identity", {})) | set(identity)
                                 if record.get("identity", {}).get(key) != identity.get(key))
                raise ValueError(f"Preparation settings, source bytes or converter changed for {name}. "
                                 f"Changed fields: {', '.join(changed)}. "
                                 "Use --overwrite to rebuild or choose another prepared version.")
        if previous and not interrupted:
            for record in previous["artifacts"]:
                path = root / record["path"]
                if not path.is_file() or verified_sha256(path) != record["sha256"]:
                    raise ValueError(f"Prepared output is missing or corrupt: {path}. Use --overwrite to rebuild.")
            print(f"REUSE {name}: verified {len(previous['artifacts'])} artifacts", flush=True)
            return previous
        existing = [str(p) for p in outputs() if p.exists()]
        if existing and not interrupted:
            raise ValueError(f"Outputs have no verified preparation receipt for {name}: {existing[:3]}. "
                             "Use --overwrite to rebuild or choose another prepared version.")
    atomic_json(pending, dict(identity_digest=expected, identity=identity))
    action(bool(overwrite or interrupted))
    paths = sorted(set(outputs()))
    if not paths:
        raise ValueError(f"Preparation produced no artifacts: {name}")
    records = [artifact(path, root) for path in paths]
    if "_train_" in name and any(r.get("rows") == 0 for r in records if "rows" in r):
        raise ValueError(f"Prepared training archive has no rows: {name}")
    result = dict(schema_version=1, identity=identity, identity_digest=expected, artifacts=records)
    atomic_json(receipt, result)
    pending.unlink()
    print(f"COMPLETE {name}: verified {len(records)} artifacts", flush=True)
    return result
