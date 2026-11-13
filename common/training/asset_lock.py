"""Content identities verified on CPU; cheap, strict checks before GPU training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": digest(path)}


def verify_asset_lock(lock: Path, code_root: Path, manifest: Path, model: Path) -> str:
    payload = json.loads(lock.read_text())
    if payload.get("schema_version") != 1 or not payload.get("content_verified"):
        raise ValueError("Training asset lock was not content-verified")
    if str(model.resolve()) != payload["model_root"]:
        raise ValueError("Base model differs from the training asset lock")
    if manifest is None or digest(manifest) != payload["manifest_sha256"]:
        raise ValueError("Training manifest differs from the training asset lock")
    for relative, expected in payload["code_sha256"].items():
        if digest(code_root / relative) != expected:
            raise ValueError(f"Code differs from the training asset lock: {relative}")
    for expected in payload["assets"]:
        path = Path(expected["path"])
        stat = path.stat()
        # Full byte hashes are computed in the CPU preflight. Detect mutation
        # cheaply on every GPU rank; indices/manifests are small enough to rehash.
        if (stat.st_size, stat.st_mtime_ns) != (expected["size"], expected["mtime_ns"]):
            raise ValueError(f"Asset changed after content verification: {path}")
        if path.suffix in {".json", ".sqlite", ".txt", ".sha256"} and digest(path) != expected["sha256"]:
            raise ValueError(f"Asset content differs from lock: {path}")
    return digest(lock)


def verify_code_only_lock_migration(previous: Path, current: Path, previous_digest: str) -> dict:
    """Allow a new code lock, never a different data/model/container identity.

    Both files are retained. The caller must also verify the current lock against
    the executing code and assets, and explicitly permit a code-only resume.
    """
    if digest(previous) != previous_digest:
        raise ValueError("Previous asset lock does not match checkpoint provenance")
    before = json.loads(previous.read_text())
    after = json.loads(current.read_text())
    if not before.get("content_verified") or not after.get("content_verified"):
        raise ValueError("Both migration locks must be content-verified")
    if {k: v for k, v in before.items() if k != "code_sha256"} != {
        k: v for k, v in after.items() if k != "code_sha256"
    }:
        raise ValueError("Asset-lock migration changes more than code identity")
    return {"previous_lock_sha256": previous_digest, "new_lock_sha256": digest(current),
            "previous_code_sha256": before["code_sha256"],
            "new_code_sha256": after["code_sha256"]}
