#!/usr/bin/env python3
"""Atomically publish one prepared member as an indexed, uncompressed TAR.

The training data loader can random-access ``indexed_prepared_tar`` dataset
specs, so after this succeeds the source member directory can be removed to
release its many image inodes.  The TAR deliberately remains uncompressed:
the SQLite sidecar stores byte offsets into it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INDEX_BUILDER = ROOT / "scripts" / "dataset_prep" / "build_prepared_tar_index.py"


def has_metadata(member: Path) -> bool:
    metadata = member / "metadata.jsonl"
    return metadata.is_file() and metadata.stat().st_size > 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member", type=Path, required=True)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--name", required=True, help="Dataset-spec name to replace.")
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--remove-member", action="store_true")
    args = parser.parse_args()

    member = args.member.expanduser().resolve()
    specs_path = args.specs.expanduser().resolve()
    if not specs_path.is_file():
        raise SystemExit(f"Dataset specs not found: {specs_path}")

    archive = (args.archive or member.with_suffix(".tar")).expanduser().resolve()
    index = archive.with_name(archive.name + ".ovb-index.sqlite")
    config = json.loads(specs_path.read_text(encoding="utf-8"))
    matches = [item for item in config.get("datasets", []) if item.get("name") == args.name]
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one dataset named {args.name!r} in {specs_path}, found {len(matches)}")

    if archive.exists() or index.exists():
        if not archive.is_file() or not index.is_file():
            raise SystemExit(f"Incomplete prior publication: expected both {archive} and {index}")
        # A job can be interrupted after one member is published. Reuse only an
        # archive that passes normal validation, then discard any rebuilt folder.
        spec = matches[0]
        spec["type"] = "indexed_prepared_tar"
        spec["path"] = str(archive)
        spec["index_path"] = str(index)
        temporary_specs = specs_path.with_name(f".{specs_path.name}.{os.getpid()}.tmp")
        temporary_specs.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "dataset_prep" / "validate_dataset_specs.py"),
                 str(temporary_specs), "--check-images", "20", "--allow-missing-loss-masks"],
                check=True,
            )
        except Exception:
            temporary_specs.unlink(missing_ok=True)
            raise
        temporary_specs.replace(specs_path)
        if args.remove_member and member.is_dir():
            subprocess.run(["rm", "-rf", "--", str(member)], check=True)
        print(json.dumps({"name": args.name, "archive": str(archive), "index": str(index), "reused": True}))
        return 0
    if not has_metadata(member):
        raise SystemExit(f"Prepared member has no nonempty metadata.jsonl: {member}")
    archive.parent.mkdir(parents=True, exist_ok=True)

    # Write both artifacts under final names only when their complete temporary
    # counterparts have been created.  A killed job therefore never publishes a
    # valid-looking partial TAR to the dataset specs.
    with tempfile.TemporaryDirectory(dir=archive.parent, prefix=f".{archive.name}.") as tmpdir:
        tmpdir_path = Path(tmpdir)
        temp_archive = tmpdir_path / archive.name
        temp_index = tmpdir_path / index.name
        subprocess.run(
            ["tar", "-cf", str(temp_archive), "-C", str(member.parent), member.name],
            check=True,
        )
        subprocess.run(
            [sys.executable, str(INDEX_BUILDER), str(temp_archive), "--output", str(temp_index)],
            check=True,
        )
        temp_archive.replace(archive)
        temp_index.replace(index)
        sha_sidecar = temp_archive.with_name(temp_archive.name + ".sha256")
        if sha_sidecar.exists():
            sha_sidecar.replace(archive.with_name(archive.name + ".sha256"))

    config = json.loads(specs_path.read_text(encoding="utf-8"))
    matches = [item for item in config.get("datasets", []) if item.get("name") == args.name]
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one dataset named {args.name!r} in {specs_path}, found {len(matches)}")
    spec = matches[0]
    spec["type"] = "indexed_prepared_tar"
    spec["path"] = str(archive)
    spec["index_path"] = str(index)
    temporary_specs = specs_path.with_name(f".{specs_path.name}.{os.getpid()}.tmp")
    temporary_specs.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_specs.replace(specs_path)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "dataset_prep" / "validate_dataset_specs.py"),
            str(specs_path),
            "--check-images",
            "20",
            # Members such as Pico, identity, PD12M and refseg carry no loss
            # masks; the paper pools validate with the same allowance.
            "--allow-missing-loss-masks",
        ],
        check=True,
    )
    if args.remove_member:
        subprocess.run(["rm", "-rf", "--", str(member)], check=True)
    print(json.dumps({"name": args.name, "archive": str(archive), "index": str(index), "removed_member": args.remove_member}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
