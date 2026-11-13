#!/usr/bin/env python3
"""Fetch selected members from a large remote ZIP using HTTP byte ranges."""

from __future__ import annotations

import argparse
import binascii
import struct
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


EOCD = b"PK\x05\x06"
ZIP64_LOCATOR = b"PK\x06\x07"
ZIP64_EOCD = b"PK\x06\x06"
CENTRAL = b"PK\x01\x02"
LOCAL = b"PK\x03\x04"


@dataclass(frozen=True)
class Member:
    name: str
    method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_offset: int


def request_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request) as response:
        data = response.read()
        if len(data) != end - start + 1:
            raise RuntimeError(
                f"Range {start}-{end} returned {len(data)} bytes, expected {end-start+1}"
            )
        return data


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request) as response:
        return int(response.headers["Content-Length"])


def zip64_value(extra: bytes, wanted: list[str], values: dict[str, int]) -> None:
    cursor = 0
    while cursor + 4 <= len(extra):
        kind, size = struct.unpack_from("<HH", extra, cursor)
        payload = extra[cursor + 4 : cursor + 4 + size]
        cursor += 4 + size
        if kind != 0x0001:
            continue
        offset = 0
        for field in wanted:
            values[field] = struct.unpack_from("<Q", payload, offset)[0]
            offset += 8
        return
    raise RuntimeError("ZIP64 member is missing its ZIP64 extended information")


def central_directory(url: str) -> tuple[int, int]:
    size = remote_size(url)
    tail_start = max(0, size - 131072)
    tail = request_range(url, tail_start, size - 1)
    eocd_at = tail.rfind(EOCD)
    if eocd_at < 0:
        raise RuntimeError("ZIP end-of-central-directory record not found")
    _, _, _, _, _, central_size32, central_offset32, _ = struct.unpack_from(
        "<4s4H2LH", tail, eocd_at
    )
    if central_size32 != 0xFFFFFFFF and central_offset32 != 0xFFFFFFFF:
        return central_offset32, central_size32

    locator_at = tail.rfind(ZIP64_LOCATOR, 0, eocd_at)
    if locator_at < 0:
        raise RuntimeError("ZIP64 locator not found")
    _, _, zip64_offset, _ = struct.unpack_from("<4sLQL", tail, locator_at)
    record = request_range(url, zip64_offset, zip64_offset + 55)
    if record[:4] != ZIP64_EOCD:
        raise RuntimeError("Invalid ZIP64 end-of-central-directory signature")
    central_size = struct.unpack_from("<Q", record, 40)[0]
    central_offset = struct.unpack_from("<Q", record, 48)[0]
    return central_offset, central_size


def read_members(url: str) -> dict[str, Member]:
    offset, size = central_directory(url)
    data = request_range(url, offset, offset + size - 1)
    members: dict[str, Member] = {}
    cursor = 0
    while cursor < len(data):
        if data[cursor : cursor + 4] != CENTRAL:
            raise RuntimeError(f"Invalid central-directory record at byte {cursor}")
        method = struct.unpack_from("<H", data, cursor + 10)[0]
        crc32, compressed32, uncompressed32 = struct.unpack_from(
            "<LLL", data, cursor + 16
        )
        name_size, extra_size, comment_size = struct.unpack_from(
            "<HHH", data, cursor + 28
        )
        local32 = struct.unpack_from("<L", data, cursor + 42)[0]
        name_start = cursor + 46
        name = data[name_start : name_start + name_size].decode("utf-8")
        extra = data[name_start + name_size : name_start + name_size + extra_size]
        values = {
            "uncompressed": uncompressed32,
            "compressed": compressed32,
            "local": local32,
        }
        wanted = []
        if uncompressed32 == 0xFFFFFFFF:
            wanted.append("uncompressed")
        if compressed32 == 0xFFFFFFFF:
            wanted.append("compressed")
        if local32 == 0xFFFFFFFF:
            wanted.append("local")
        if wanted:
            zip64_value(extra, wanted, values)
        members[name] = Member(
            name=name,
            method=method,
            crc32=crc32,
            compressed_size=values["compressed"],
            uncompressed_size=values["uncompressed"],
            local_offset=values["local"],
        )
        cursor = name_start + name_size + extra_size + comment_size
    return members


def extract_member(url: str, member: Member) -> bytes:
    header = request_range(url, member.local_offset, member.local_offset + 29)
    if header[:4] != LOCAL:
        raise RuntimeError(f"Invalid local header for {member.name}")
    name_size, extra_size = struct.unpack_from("<HH", header, 26)
    data_offset = member.local_offset + 30 + name_size + extra_size
    compressed = request_range(
        url, data_offset, data_offset + member.compressed_size - 1
    )
    if member.method == 0:
        result = compressed
    elif member.method == 8:
        result = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError(f"Unsupported compression method {member.method}: {member.name}")
    if len(result) != member.uncompressed_size:
        raise RuntimeError(f"Size mismatch for {member.name}")
    if binascii.crc32(result) & 0xFFFFFFFF != member.crc32:
        raise RuntimeError(f"CRC mismatch for {member.name}")
    return result


def safe_destination(root: Path, member_name: str) -> Path:
    relative = PurePosixPath(member_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe ZIP member path: {member_name}")
    return root.joinpath(*relative.parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--member-file", type=Path)
    parser.add_argument(
        "--pair-list",
        type=Path,
        help="Two-column RGB/normal list whose paths are relative to --pair-prefix.",
    )
    parser.add_argument("--pair-prefix", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0,
                        help="Skip this many pair-list rows before applying --limit.")
    parser.add_argument("--prefix", action="append", default=[])
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    members = read_members(args.url)
    selected_names: set[str] = set()
    if args.member_file:
        selected_names.update(
            line.strip()
            for line in args.member_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if args.pair_list:
        rows = [
            line.split()
            for line in args.pair_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if args.start < 0 or (args.limit is not None and args.limit < 0):
            parser.error("--start and --limit must be non-negative")
        rows = rows[args.start:]
        if args.limit is not None:
            rows = rows[: args.limit]
        for row in rows:
            if len(row) < 2:
                raise ValueError(f"Malformed pair-list row: {row}")
            selected_names.update(
                f"{args.pair_prefix.rstrip('/')}/{name.lstrip('/')}".lstrip("/")
                for name in row[:2]
            )
    for prefix in args.prefix:
        selected_names.update(name for name in members if name.startswith(prefix))
    if args.list:
        for name in sorted(selected_names or members):
            print(name)
        return
    if not selected_names or args.output is None:
        parser.error("extraction requires --output and at least one --member-file/--prefix")

    missing = sorted(selected_names - members.keys())
    if missing:
        raise FileNotFoundError(f"Remote ZIP members not found: {missing[:10]}")
    args.output.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(sorted(selected_names), start=1):
        if name.endswith("/"):
            continue
        destination = safe_destination(args.output, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(extract_member(args.url, members[name]))
        print(f"[{index}/{len(selected_names)}] {name}", flush=True)


if __name__ == "__main__":
    main()
