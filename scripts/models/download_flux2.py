#!/usr/bin/env python3
"""Secure, resumable acquisition and structural verification of a FLUX.2 snapshot."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
except ImportError:  # Keep local snapshot verification usable in the old container.
    HfApi = hf_hub_download = snapshot_download = None
    GatedRepoError = HfHubHTTPError = ()


REPO_ID = "black-forest-labs/FLUX.2-klein-base-9B"
REQUIRED_COMPONENTS = ("scheduler", "tokenizer", "text_encoder", "transformer", "vae")


def read_token() -> str:
    token_path = Path(
        os.environ.get("HF_TOKEN_PATH", Path.home() / ".cache" / "huggingface" / "token")
    ).expanduser()
    if not token_path.is_file():
        raise RuntimeError(
            f"Hugging Face token file not found: {token_path}. Accept the model gate if required, "
            "then create this file with mode 600."
        )
    mode = stat.S_IMODE(token_path.stat().st_mode)
    if mode & 0o077:
        raise RuntimeError(f"Token file permissions are too broad ({mode:o}): {token_path}; run chmod 600.")
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(f"Hugging Face token file is empty: {token_path}")
    return token


def classify_hub_error(exc: BaseException) -> str:
    if isinstance(exc, GatedRepoError):
        return "MODEL_GATE_NOT_ACCEPTED"
    if isinstance(exc, HfHubHTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 401:
            return "INVALID_OR_EXPIRED_TOKEN"
        if status == 403:
            return "MODEL_GATE_NOT_ACCEPTED"
        return f"HUGGINGFACE_HTTP_{status or 'ERROR'}"
    message = str(exc).lower()
    if "certificate" in message or "ssl" in message:
        return "TLS_CERTIFICATE_ERROR"
    if any(term in message for term in ("name resolution", "network is unreachable", "connection")):
        return "NETWORK_UNAVAILABLE"
    return "HUGGINGFACE_ACCESS_ERROR"


def preflight(token: str) -> str:
    if HfApi is None or hf_hub_download is None:
        raise RuntimeError(
            "MISSING_HUGGINGFACE_HUB: install the repository requirements or rebuild "
            "open_visionbanana.sif before using preflight/download."
        )
    try:
        info = HfApi(token=token).model_info(REPO_ID, files_metadata=False)
        # Metadata for a public gated repository may be visible before its terms
        # are accepted. Fetch a tiny gated file to prove actual weight access.
        hf_hub_download(REPO_ID, filename="model_index.json", revision=info.sha, token=token)
    except Exception as exc:
        code = classify_hub_error(exc)
        raise RuntimeError(
            f"{code}: cannot access {REPO_ID}. If the token is valid, accept the model terms at "
            f"https://huggingface.co/{REPO_ID}. Details: {exc}"
        ) from exc
    print(f"Access preflight passed for {REPO_ID} at revision {info.sha}")
    return info.sha


def indexed_shards(index_path: Path) -> list[str]:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid or empty weight_map in {index_path}")
    return sorted(set(str(value) for value in weight_map.values()))


def verify_weights(component: Path) -> list[str]:
    indexes = sorted(component.glob("*.index.json"))
    missing: list[str] = []
    if indexes:
        for index in indexes:
            for shard in indexed_shards(index):
                if not (component / shard).is_file():
                    missing.append(str(component / shard))
    elif not any(component.glob(pattern) for pattern in ("*.safetensors", "*.bin")):
        missing.append(f"{component}/*.safetensors (or *.bin)")
    return missing


def verify_snapshot(root: Path) -> dict[str, object]:
    if not root.is_dir():
        raise FileNotFoundError(f"Snapshot directory not found: {root}")
    missing: list[str] = []
    if not (root / "model_index.json").is_file():
        missing.append(str(root / "model_index.json"))
    for name in REQUIRED_COMPONENTS:
        component = root / name
        if not component.is_dir():
            missing.append(str(component))
            continue
        if name in {"text_encoder", "transformer", "vae"}:
            missing.extend(verify_weights(component))
    if not any((root / "tokenizer" / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        missing.append(str(root / "tokenizer" / "tokenizer.json|tokenizer.model"))
    if missing:
        raise RuntimeError(f"Incomplete {REPO_ID} snapshot; missing:\n  " + "\n  ".join(missing))

    indexes = sorted(root.glob("**/*.index.json"))
    shards = sum((len(indexed_shards(path)) for path in indexes), 0)
    result = {
        "repo_id": REPO_ID,
        "root": str(root.resolve()),
        "required_components": list(REQUIRED_COMPONENTS),
        "weight_indexes": len(indexes),
        "indexed_shards": shards,
        "status": "complete",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def download(dest: Path, token: str, revision: str) -> Path:
    if dest.exists():
        verify_snapshot(dest)
        print(f"Verified existing snapshot; nothing to download: {dest}")
        return dest

    partial = dest.with_name(dest.name + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir(parents=True, exist_ok=True)
    if snapshot_download is None:
        raise RuntimeError(
            "MISSING_HUGGINGFACE_HUB: install the repository requirements or rebuild "
            "open_visionbanana.sif before downloading."
        )
    snapshot_download(
        repo_id=REPO_ID,
        revision=revision,
        local_dir=partial,
        token=token,
    )
    manifest = verify_snapshot(partial)
    manifest["resolved_revision"] = revision
    (partial / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, dest)
    print(f"{REPO_ID} snapshot installed atomically at {dest}")
    return dest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "download", "verify"), default="preflight")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--revision", default="main")
    return parser.parse_args()


def main() -> int:
    global REPO_ID
    args = parse_args()
    REPO_ID = args.repo_id
    try:
        if args.stage == "verify":
            verify_snapshot(args.dest.expanduser())
            return 0
        token = read_token()
        resolved = preflight(token)
        if args.stage == "download":
            download(args.dest.expanduser(), token, resolved if args.revision == "main" else args.revision)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
