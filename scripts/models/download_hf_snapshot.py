#!/usr/bin/env python3
"""Secure, resumable acquisition of a generic Hugging Face model snapshot."""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
except ImportError:
    HfApi = hf_hub_download = snapshot_download = None


def optional_token() -> str | None:
    path = Path(
        os.environ.get("HF_TOKEN_PATH", Path.home() / ".cache" / "huggingface" / "token")
    ).expanduser()
    if not path.exists():
        return None
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise RuntimeError(f"Token file permissions are too broad ({mode:o}): {path}; run chmod 600")
    token = path.read_text(encoding="utf-8").strip()
    return token or None


def indexed_shards(index: Path) -> set[str]:
    data = json.loads(index.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map", {})
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"invalid or empty weight_map: {index}")
    return {str(value) for value in weight_map.values()}


def verify(root: Path, repo_id: str) -> dict[str, object]:
    if not root.is_dir():
        raise RuntimeError(f"snapshot directory not found: {root}")
    if not any((root / name).is_file() for name in ("config.json", "model_index.json")):
        raise RuntimeError(f"model config is missing below {root} (config.json or model_index.json)")
    indexes = sorted(root.rglob("*.index.json"))
    missing = []
    shard_count = 0
    for index in indexes:
        for relative in indexed_shards(index):
            shard_count += 1
            if not (index.parent / relative).is_file():
                missing.append(str(index.parent / relative))
    if missing:
        raise RuntimeError("snapshot has missing indexed shards:\n  " + "\n  ".join(missing))
    if not indexes and not any(any(root.rglob(pattern)) for pattern in ("*.safetensors", "*.bin")):
        raise RuntimeError(f"no model weights found below {root}")
    result = {
        "repo_id": repo_id,
        "root": str(root.resolve()),
        "weight_indexes": len(indexes),
        "indexed_shards": shard_count,
        "status": "complete",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "download", "verify"), required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    dest = args.dest.expanduser()

    if args.stage == "verify":
        verify(dest, args.repo_id)
        return

    if HfApi is None or hf_hub_download is None or snapshot_download is None:
        raise RuntimeError("huggingface_hub is required for preflight/download")

    token = optional_token()
    info = HfApi(token=token).model_info(args.repo_id, revision=args.revision, files_metadata=False)
    siblings = {sibling.rfilename for sibling in info.siblings}
    metadata_file = "config.json" if "config.json" in siblings else "model_index.json"
    if metadata_file not in siblings:
        raise RuntimeError(f"{args.repo_id} exposes neither config.json nor model_index.json")
    hf_hub_download(args.repo_id, metadata_file, revision=info.sha, token=token)
    print(f"Access preflight passed for {args.repo_id} at revision {info.sha}")
    if args.stage == "preflight":
        return

    if dest.exists():
        verify(dest, args.repo_id)
        print(f"Verified existing snapshot; nothing to download: {dest}")
        return
    partial = dest.with_name(dest.name + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir(parents=True, exist_ok=True)
    snapshot_download(args.repo_id, revision=info.sha, local_dir=partial, token=token)
    manifest = verify(partial, args.repo_id)
    manifest["resolved_revision"] = info.sha
    (partial / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, dest)
    print(f"Snapshot installed atomically at {dest}")


if __name__ == "__main__":
    main()
