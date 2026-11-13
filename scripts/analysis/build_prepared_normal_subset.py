#!/usr/bin/env python3
"""Build an inode-efficient subset tar from a prepared normal-evaluation tar."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path


def normalized(name: str) -> str:
    return name.lstrip("./")


def read_member(archive: tarfile.TarFile, name: str) -> bytes:
    member = archive.getmember(name)
    handle = archive.extractfile(member)
    if handle is None:
        raise FileNotFoundError(f"Archive member is not a regular file: {name}")
    return handle.read()


def add_bytes(output: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mode = 0o644
    output.addfile(info, io.BytesIO(payload))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    root = args.dataset_root.strip("/")
    split_name = normalized(f"{root}/{args.split_file}")

    with tarfile.open(args.input, "r:*") as source:
        member_by_name = {normalized(member.name): member for member in source.getmembers()}
        if split_name not in member_by_name:
            raise FileNotFoundError(f"Missing split member {split_name} in {args.input}")
        split_bytes = read_member(source, member_by_name[split_name].name)
        lines = [line for line in split_bytes.decode("utf-8").splitlines() if line.strip()]
        selected = lines[: args.limit]
        if len(selected) < args.limit:
            print(f"warning: requested {args.limit}, found {len(selected)} samples")

        required = {split_name}
        for line in selected:
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"Malformed split line: {line}")
            for field in fields[:4]:
                required.add(normalized(f"{root}/{field.lstrip('/')}"))
        for optional in ("dataset_info.json", "provenance.json"):
            name = normalized(f"{root}/{optional}")
            if name in member_by_name:
                required.add(name)

        missing = sorted(name for name in required if name not in member_by_name)
        if missing:
            raise FileNotFoundError("Missing referenced members:\n" + "\n".join(missing))

        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + ".partial")
        with tarfile.open(tmp, "w") as output:
            for name in sorted(required - {split_name}):
                member = member_by_name[name]
                handle = source.extractfile(member)
                if handle is None:
                    raise FileNotFoundError(f"Archive member is not a regular file: {name}")
                copied = tarfile.TarInfo(name=name)
                copied.size = member.size
                copied.mode = member.mode
                copied.mtime = member.mtime
                output.addfile(copied, handle)
            add_bytes(output, split_name, ("\n".join(selected) + "\n").encode("utf-8"))
            manifest = {
                "source_archive": str(args.input),
                "dataset_root": root,
                "source_split_member": split_name,
                "selection": "first_n_in_source_split",
                "count": len(selected),
            }
            add_bytes(
                output,
                f"{root}/subset_info.json",
                (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
        with tarfile.open(tmp, "r:") as check:
            check.getmembers()
        tmp.replace(args.output)
        print(f"wrote {args.output} with {len(selected)} samples and {len(required) + 1} files")


if __name__ == "__main__":
    main()
