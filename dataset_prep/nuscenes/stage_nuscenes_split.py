#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath


REQUIRED_TABLES = (
    "calibrated_sensor.json",
    "ego_pose.json",
    "sample.json",
    "sample_data.json",
    "scene.json",
    "sensor.json",
)


def normalized_member_name(name: str) -> str:
    while name.startswith("./"):
        name = name[2:]
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member path: {name!r}")
    return path.as_posix()


def copy_member(archive: tarfile.TarFile, member: tarfile.TarInfo, destination: Path, suffix: str) -> None:
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"Could not read archive member: {member.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial-{suffix}")
    try:
        with source, partial.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def extract_exact_members(archive_path: Path, output_root: Path, wanted: set[str]) -> set[str]:
    found: set[str] = set()
    with tarfile.open(archive_path, mode="r|*") as archive:
        for member in archive:
            name = normalized_member_name(member.name)
            if name not in wanted:
                continue
            if name in found:
                raise ValueError(f"Duplicate requested member in {archive_path}: {name}")
            if not member.isfile():
                raise ValueError(f"Requested archive member is not a regular file: {name}")
            copy_member(archive, member, output_root / name, str(os.getpid()))
            found.add(name)
    return found


def extract_metadata(metadata_archive: Path, output_root: Path, version: str) -> None:
    wanted = {f"{version}/{name}" for name in REQUIRED_TABLES}
    found = extract_exact_members(metadata_archive, output_root, wanted)
    missing = sorted(wanted - found)
    if missing:
        raise FileNotFoundError(f"Metadata archive is missing required files: {', '.join(missing)}")


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_scene_names(path: Path) -> set[str]:
    names = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not names:
        raise ValueError(f"Scene list is empty: {path}")
    return names


def required_sensor_files(
    metadata_root: Path,
    scene_list: Path,
    cameras: tuple[str, ...],
    limit: int | None,
) -> tuple[set[str], int, int]:
    scenes = load_json(metadata_root / "scene.json")
    samples = load_json(metadata_root / "sample.json")
    sample_data = load_json(metadata_root / "sample_data.json")
    calibrated = {row["token"]: row for row in load_json(metadata_root / "calibrated_sensor.json")}
    sensors = {row["token"]: row for row in load_json(metadata_root / "sensor.json")}

    requested_scene_names = load_scene_names(scene_list)
    scene_tokens = {row["token"] for row in scenes if row["name"] in requested_scene_names}
    found_scene_names = {row["name"] for row in scenes if row["token"] in scene_tokens}
    missing_scenes = sorted(requested_scene_names - found_scene_names)
    if missing_scenes:
        raise ValueError(f"Scene list contains names absent from metadata: {', '.join(missing_scenes)}")

    selected_samples = [row for row in samples if row["scene_token"] in scene_tokens]
    if limit is not None:
        selected_samples = selected_samples[:limit]
    selected_tokens = {row["token"] for row in selected_samples}
    requested_channels = {"LIDAR_TOP", *cameras}
    files_by_sample: dict[str, dict[str, str]] = {}
    for row in sample_data:
        if row["sample_token"] not in selected_tokens or not row.get("is_key_frame", False):
            continue
        calibration = calibrated[row["calibrated_sensor_token"]]
        channel = sensors[calibration["sensor_token"]]["channel"]
        if channel in requested_channels:
            files_by_sample.setdefault(row["sample_token"], {})[channel] = row["filename"]

    required: set[str] = set()
    for sample in selected_samples:
        channels = files_by_sample.get(sample["token"], {})
        missing_channels = sorted(requested_channels - channels.keys())
        if missing_channels:
            raise ValueError(f"Sample {sample['token']} is missing keyframe channels: {', '.join(missing_channels)}")
        required.update(channels[channel] for channel in requested_channels)
    return required, len(scene_tokens), len(selected_samples)


def stage_blob_archives(archives: list[Path], output_root: Path, wanted: set[str], workers: int) -> None:
    claimed: set[str] = set()
    lock = threading.Lock()

    def stage_one(index_and_path: tuple[int, Path]) -> tuple[Path, set[str]]:
        index, archive_path = index_and_path
        local_found: set[str] = set()
        with tarfile.open(archive_path, mode="r|*") as archive:
            for member in archive:
                name = normalized_member_name(member.name)
                if name not in wanted:
                    continue
                with lock:
                    if name in claimed:
                        raise ValueError(f"Requested member appears in more than one blob archive: {name}")
                    claimed.add(name)
                if not member.isfile():
                    raise ValueError(f"Requested archive member is not a regular file: {name}")
                copy_member(archive, member, output_root / name, f"{index}-{threading.get_ident()}")
                local_found.add(name)
        return archive_path, local_found

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for archive_path, found in executor.map(stage_one, enumerate(archives)):
            print(f"Scanned {archive_path.name}: extracted {len(found)} requested files", flush=True)

    missing = sorted(wanted - claimed)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... and {len(missing) - 10} more"
        raise FileNotFoundError(f"Blob archives are missing {len(missing)} requested files: {preview}{suffix}")


def parse_cameras(value: str) -> tuple[str, ...]:
    cameras = tuple(part.strip().upper() for part in value.replace(",", " ").split() if part.strip())
    if not cameras:
        raise argparse.ArgumentTypeError("At least one camera is required")
    return cameras


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Selectively stage a nuScenes scene split from blob archives.")
    parser.add_argument("--metadata-archive", type=Path, required=True)
    parser.add_argument("--blob", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--cameras", type=parse_cameras, default=("CAM_FRONT",))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--expected-archives", type=int, default=10)
    parser.add_argument("--expected-scenes", type=int)
    parser.add_argument("--expected-samples", type=int)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    archives = [path.expanduser().resolve() for path in args.blob]
    if len(archives) != args.expected_archives:
        raise ValueError(f"Expected {args.expected_archives} blob archives, found {len(archives)}")
    missing_archives = [path for path in archives if not path.is_file()]
    if missing_archives:
        raise FileNotFoundError(f"Missing blob archives: {', '.join(map(str, missing_archives))}")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    extract_metadata(args.metadata_archive.expanduser().resolve(), output_root, args.version)
    wanted, scene_count, sample_count = required_sensor_files(
        output_root / args.version,
        args.scene_list.expanduser().resolve(),
        args.cameras,
        args.limit,
    )
    if args.expected_scenes is not None and scene_count != args.expected_scenes:
        raise ValueError(f"Expected {args.expected_scenes} scenes, resolved {scene_count}")
    if args.expected_samples is not None and sample_count != args.expected_samples:
        raise ValueError(f"Expected {args.expected_samples} samples, resolved {sample_count}")
    print(f"Resolved {scene_count} scenes, {sample_count} samples, and {len(wanted)} sensor files", flush=True)
    stage_blob_archives(archives, output_root, wanted, args.workers)

    manifest = {
        "version": args.version,
        "scene_list": str(args.scene_list.expanduser().resolve()),
        "scene_count": scene_count,
        "sample_count": sample_count,
        "cameras": list(args.cameras),
        "sensor_file_count": len(wanted),
        "blob_archives": [str(path) for path in archives],
    }
    (output_root / "staging_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Selective nuScenes staging complete: {output_root}")


if __name__ == "__main__":
    main()
