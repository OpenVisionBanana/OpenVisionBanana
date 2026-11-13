#!/usr/bin/env python3
"""Materialize the last CLIP-FlanT5-compatible t2v_metrics implementation.

The current t2v_metrics release supports the Qwen judges used by this project,
but removed CLIP-FlanT5 from its model registry.  GenAI-Bench protocol v1.1
reports both judges, so scoring jobs need the older CLIP implementation without
replacing the current package used by Qwen.

This script exports only the required files from a pinned commit in the local
t2v_metrics checkout and installs a deliberately minimal model registry.  The
result is intended to be placed first on PYTHONPATH for the CLIP scoring
process only.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path


DEFAULT_REVISION = "495706caa27fac440796e73c12439a35c47fceef"
TOKENIZER_REPO = "zhiqiulin/clip-flant5-xxl"

ARCHIVE_PATHS = (
    "t2v_metrics/__init__.py",
    "t2v_metrics/constants.py",
    "t2v_metrics/score.py",
    "t2v_metrics/vqascore.py",
    "t2v_metrics/models/__init__.py",
    "t2v_metrics/models/model.py",
    "t2v_metrics/models/vqascore_models/clip_t5",
    "t2v_metrics/models/vqascore_models/clip_t5_model.py",
    "t2v_metrics/models/vqascore_models/mm_utils.py",
    "t2v_metrics/models/vqascore_models/vqa_model.py",
)

MINIMAL_REGISTRY = """\
from .clip_t5_model import CLIP_T5_MODELS, CLIPT5Model
from ...constants import HF_CACHE_DIR

ALL_VQA_MODELS = [CLIP_T5_MODELS]


def list_all_vqascore_models():
    return [model for models in ALL_VQA_MODELS for model in models]


def get_vqascore_model(model_name, device="cuda", cache_dir=HF_CACHE_DIR, **kwargs):
    if model_name in CLIP_T5_MODELS:
        return CLIPT5Model(
            model_name,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    raise NotImplementedError(model_name)
"""

MINIMAL_PACKAGE_INIT = """\
from .constants import HF_CACHE_DIR
from .vqascore import VQAScore, list_all_vqascore_models


def list_all_models():
    return list_all_vqascore_models()


def get_score_model(model="clip-flant5-xxl", device="cuda", cache_dir=HF_CACHE_DIR, **kwargs):
    if model in list_all_vqascore_models():
        return VQAScore(model, device=device, cache_dir=cache_dir, **kwargs)
    raise NotImplementedError(model)
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency_repo", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    return parser.parse_args()


def git_output(repo: Path, *arguments: str) -> bytes:
    command = [
        "git",
        "-c",
        f"safe.directory={repo.resolve()}",
        "-C",
        str(repo),
        *arguments,
    ]
    return subprocess.run(command, check=True, stdout=subprocess.PIPE).stdout


def safe_extract(archive_bytes: bytes, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        for member in archive.getmembers():
            member_path = (destination / member.name).resolve()
            if destination_resolved not in member_path.parents and member_path != destination_resolved:
                raise RuntimeError(f"Unsafe path in compatibility archive: {member.name}")
        archive.extractall(destination)


def main() -> None:
    args = parse_args()
    repo = args.dependency_repo.resolve()
    out_dir = args.out_dir.resolve()
    if not (repo / ".git").exists():
        raise FileNotFoundError(f"Not a git checkout: {repo}")

    resolved_revision = git_output(repo, "rev-parse", args.revision).decode().strip()
    if resolved_revision != args.revision:
        raise RuntimeError(
            f"CLIP compatibility revision mismatch: expected {args.revision}, "
            f"resolved {resolved_revision}"
        )

    package_dir = out_dir / "t2v_metrics"
    package_init_path = package_dir / "__init__.py"
    registry_path = package_dir / "models/vqascore_models/__init__.py"
    clip_model_path = package_dir / "models/vqascore_models/clip_t5_model.py"
    provenance_path = out_dir / "CLIP_FLANT5_COMPAT.json"

    expected_provenance = {
        "schema_version": 1,
        "transformers_runtime": "4.40.2",
        "t2v_metrics_revision": resolved_revision,
        "tokenizer_repo": TOKENIZER_REPO,
        "package_init_sha256": hashlib.sha256(MINIMAL_PACKAGE_INIT.encode()).hexdigest(),
        "registry_sha256": hashlib.sha256(MINIMAL_REGISTRY.encode()).hexdigest(),
    }
    if provenance_path.is_file():
        actual = json.loads(provenance_path.read_text(encoding="utf-8"))
        if actual != expected_provenance:
            raise RuntimeError(f"Existing compatibility tree has different provenance: {out_dir}")
        print(out_dir)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    archive = git_output(repo, "archive", resolved_revision, "--", *ARCHIVE_PATHS)
    safe_extract(archive, out_dir)

    package_init_path.write_text(MINIMAL_PACKAGE_INIT, encoding="utf-8")
    registry_path.write_text(MINIMAL_REGISTRY, encoding="utf-8")
    clip_source = clip_model_path.read_text(encoding="utf-8")
    old_tokenizer = "'path': 'google/flan-t5-xxl'"
    new_tokenizer = f"'path': '{TOKENIZER_REPO}'"
    replacements = clip_source.count(old_tokenizer)
    if replacements != 8:
        raise RuntimeError(
            f"Expected eight FlanT5-XXL tokenizer references, found {replacements}"
        )
    clip_model_path.write_text(
        clip_source.replace(old_tokenizer, new_tokenizer),
        encoding="utf-8",
    )
    provenance_path.write_text(
        json.dumps(expected_provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(out_dir)


if __name__ == "__main__":
    main()
