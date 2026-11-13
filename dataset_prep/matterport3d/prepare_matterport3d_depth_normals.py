#!/usr/bin/env python3
"""Convert archived Matterport3D undistorted views to canonical OVB training tars."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import shutil
import sys
import tarfile
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from common.geometry_protocol_prompts import VISION_BANANA_NORMAL_PROMPT
from common.surface_normal_encoding import (
    SUPPORTED_NORMAL_ENCODINGS,
    VISION_BANANA_NORMAL_ENCODING,
    encode_surface_normals_rgb,
    normal_encoding_formula,
)
from dataset_prep.depth.encoding import encode_depth_to_vision_banana_rgb
from dataset_prep.geometry_conventions import (
    CANONICAL_DEPTH_QUANTITY,
    CANONICAL_NORMAL_CONVENTION,
    GEOMETRY_DATA_VERSION,
    MATTERPORT_NATIVE_TO_CANONICAL,
    geometry_data_version_for_normal_encoding,
)

DEPTH_PROMPT = "[TASK: DEPTH_ESTIMATION] Generate a metric depth visualization of the input indoor scene using the Vision Banana RGB depth encoding. Output only the RGB depth visualization."
NORMAL_PROMPT = VISION_BANANA_NORMAL_PROMPT
MODALITIES = ("undistorted_color_images", "undistorted_depth_images", "undistorted_normal_images", "undistorted_camera_parameters")


def align_official_modalities(rgb, depth_raw, normal_shape, strict):
    """Map aligned official full-resolution RGB/depth to the normal grid."""
    height, width = normal_shape
    if depth_raw.shape != (rgb.height, rgb.width):
        if strict:
            raise ValueError(f"RGB/depth dimensions differ: {rgb.size}, {depth_raw.shape}")
        rgb = rgb.resize((depth_raw.shape[1], depth_raw.shape[0]), Image.Resampling.BILINEAR)
    if depth_raw.shape != (height, width):
        if strict and depth_raw.shape != (2 * height, 2 * width):
            raise ValueError(f"Unexplained Matterport modality dimensions: depth {depth_raw.shape}, normals {normal_shape}")
        depth_raw = np.asarray(Image.fromarray(depth_raw).resize((width, height), Image.Resampling.NEAREST), dtype=np.uint16)
        rgb = rgb.resize((width, height), Image.Resampling.BILINEAR)
    return rgb, depth_raw


def safe_extract(tf: tarfile.TarFile, root: Path) -> None:
    root_abs = root.resolve()
    for member in tf.getmembers():
        destination = (root / member.name).resolve()
        if root_abs not in destination.parents and destination != root_abs:
            raise ValueError(f"unsafe tar member: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"links are not allowed in source tar: {member.name}")
    tf.extractall(root)


def frame_key(name: str) -> str | None:
    stem = PurePosixPath(name).stem
    # Official names use <panorama>_[id]<yaw>_<camera>; color uses i and
    # depth/normal uses d for the same physical frame.
    stem = re.sub(r"_([id])(\d+)_(\d+)$", r"_\2_\3", stem)
    stem = re.sub(r"_(?:i|d)$", "", stem)
    return stem or None


def index_zip(path: Path, kind: str) -> dict[str, str]:
    result: dict[str, str] = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            base = PurePosixPath(name).name
            if kind == "rgb" and base.lower().endswith((".jpg", ".jpeg")):
                key = frame_key(base)
            elif kind == "depth" and base.lower().endswith(".png"):
                key = frame_key(base)
            else:
                key = None
            if key:
                result[key] = name
    return result


def index_normals(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    pattern = re.compile(r"^(.*)_(n[xyz])\.png$", re.I)
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            match = pattern.match(PurePosixPath(name).name)
            if match:
                result.setdefault(frame_key(match.group(1)) or match.group(1), {})[match.group(2).lower()] = name
    return {key: value for key, value in result.items() if set(value) == {"nx", "ny", "nz"}}


def panorama_orientation(key: str) -> tuple[str, int, int]:
    match = re.match(r"^(.*)_([0-5])_([0-2])$", key)
    if not match:
        return key, -1, -1
    return match.group(1), int(match.group(2)), int(match.group(3))


def parse_camera_positions(text: str) -> dict[str, tuple[float, float, float]]:
    positions: dict[str, tuple[float, float, float]] = {}
    for line in text.splitlines():
        tokens = line.split()
        if not tokens or tokens[0] != "scan" or len(tokens) < 19:
            continue
        key = frame_key(PurePosixPath(tokens[1]).name)
        try:
            matrix = [float(value) for value in tokens[-16:]]
        except ValueError:
            continue
        if key:
            positions[key] = (matrix[3], matrix[7], matrix[11])
    return positions


def farthest_panorama_order(panoramas: list[str], positions: dict[str, tuple[float, float, float]], seed: int) -> list[str]:
    if not panoramas:
        return []
    fallback = sorted(panoramas, key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).hexdigest())
    located = {p: np.asarray(positions[p], dtype=np.float64) for p in panoramas if p in positions}
    if len(located) != len(panoramas):
        return fallback
    selected = [fallback[0]]
    remaining = set(panoramas) - set(selected)
    while remaining:
        choice = max(remaining, key=lambda p: (min(float(np.linalg.norm(located[p] - located[q])) for q in selected), p))
        selected.append(choice); remaining.remove(choice)
    return selected


def diverse_candidate_order(keys: list[str], positions: dict[str, tuple[float, float, float]], seed: int) -> list[str]:
    by_panorama: dict[str, dict[tuple[int, int], str]] = {}
    panorama_positions: dict[str, tuple[float, float, float]] = {}
    for key in keys:
        panorama, yaw, elevation = panorama_orientation(key)
        by_panorama.setdefault(panorama, {})[(yaw, elevation)] = key
        if key in positions:
            panorama_positions[panorama] = positions[key]
    panorama_order = farthest_panorama_order(list(by_panorama), panorama_positions, seed)
    per_panorama: dict[str, list[str]] = {}
    for index, panorama in enumerate(panorama_order):
        available = by_panorama[panorama]
        preferred: list[tuple[int, int]] = []
        # First three rounds cover up/level/down with rotating yaw bins.
        for elevation in range(3):
            preferred.append(((index + 2 * elevation) % 6, elevation))
        remaining = sorted(set(available) - set(preferred), key=lambda value: ((value[0] - index) % 6, value[1]))
        per_panorama[panorama] = [available[o] for o in preferred + remaining if o in available]
    result: list[str] = []
    for round_index in range(max(map(len, per_panorama.values()), default=0)):
        for panorama in panorama_order:
            values = per_panorama[panorama]
            if round_index < len(values):
                result.append(values[round_index])
    return result


def read_image(zf: zipfile.ZipFile, member: str) -> Image.Image:
    with zf.open(member) as handle:
        image = Image.open(io.BytesIO(handle.read()))
        image.load()
    return image


def normal_array(zf: zipfile.ZipFile, members: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    components = [np.asarray(read_image(zf, members[c]), dtype=np.float32) / 32768.0 - 1.0 for c in ("nx", "ny", "nz")]
    normal = np.stack(components, axis=-1)
    lengths = np.linalg.norm(normal, axis=-1)
    valid = np.isfinite(normal).all(axis=-1) & (lengths > 0.5) & (lengths < 1.5)
    normal = MATTERPORT_NATIVE_TO_CANONICAL.apply(normal, valid)
    return normal, valid


def write_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def tar_directory(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with tarfile.open(partial, "w") as tf:
        tf.add(source, arcname=source.name)
    os.replace(partial, target)


def process_scene(source_tar: Path, scan: str, args: argparse.Namespace, shard_root: Path) -> tuple[Path, Path]:
    depth_tar = shard_root / "depth" / f"{scan}.tar"
    normal_tar = shard_root / "normals" / f"{scan}.tar"
    if depth_tar.is_file() and normal_tar.is_file() and not args.force_scenes:
        return depth_tar, normal_tar
    scene_work = args.scratch_root / f"scene_{scan}"
    shutil.rmtree(scene_work, ignore_errors=True)
    scene_work.mkdir(parents=True)
    try:
        with tarfile.open(source_tar) as tf:
            safe_extract(tf, scene_work)
        paths = {m: scene_work / f"{m}.zip" for m in MODALITIES}
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        rgb_idx = index_zip(paths["undistorted_color_images"], "rgb")
        depth_idx = index_zip(paths["undistorted_depth_images"], "depth")
        normal_idx = index_normals(paths["undistorted_normal_images"])
        keys = sorted(set(rgb_idx) & set(depth_idx) & set(normal_idx))
        if not keys:
            raise RuntimeError(f"{scan}: no aligned RGB/depth/normal frames")
        camera_text = ""
        with zipfile.ZipFile(paths["undistorted_camera_parameters"]) as camera_zip:
            text_names = [n for n in camera_zip.namelist() if not n.endswith("/")]
            for name in text_names:
                try:
                    camera_text += camera_zip.read(name).decode("utf-8", "ignore")
                except Exception:
                    pass
        calibrated = [k for k in keys if PurePosixPath(rgb_idx[k]).name in camera_text and PurePosixPath(depth_idx[k]).name in camera_text]
        positions = parse_camera_positions(camera_text)
        if calibrated:
            keys = calibrated
        elif args.strict_calibration:
            raise RuntimeError(f"{scan}: camera parameter package did not reference aligned frames")
        candidate_order = diverse_candidate_order(keys, positions, args.seed)
        target_count = len(candidate_order) if args.samples_per_scene == "all" else min(int(args.samples_per_scene), len(candidate_order))
        depth_root = scene_work / "matterport3d_depth"
        normal_root = scene_work / "matterport3d_surface_normals"
        depth_rows, normal_rows = [], []
        with zipfile.ZipFile(paths["undistorted_color_images"]) as rz, zipfile.ZipFile(paths["undistorted_depth_images"]) as dz, zipfile.ZipFile(paths["undistorted_normal_images"]) as nz:
            for key in candidate_order:
                if len(depth_rows) >= target_count:
                    break
                rgb = read_image(rz, rgb_idx[key]).convert("RGB")
                depth_raw = np.asarray(read_image(dz, depth_idx[key]), dtype=np.uint16)
                normal, normal_valid = normal_array(nz, normal_idx[key])
                native_sizes = dict(rgb=list(rgb.size), depth=list(depth_raw.shape), normals=list(normal.shape[:2]))
                rgb, depth_raw = align_official_modalities(rgb, depth_raw, normal.shape[:2], args.strict)
                depth = depth_raw.astype(np.float32) / 4000.0
                depth_valid = (depth_raw > 0) & (depth >= args.min_depth) & (depth <= args.max_depth)
                valid = depth_valid & normal_valid
                if float(valid.mean()) < args.min_valid_fraction:
                    continue
                sample = f"matterport3d_{scan}_{re.sub('[^A-Za-z0-9]+', '_', key)}"
                condition_rel = f"conditioning_images/{sample}.png"
                target_rel = f"target_images/{sample}.png"
                mask_rel = f"valid_masks/{sample}.png"
                depth_rgb = encode_depth_to_vision_banana_rgb(depth, valid_mask=depth_valid, invalid_color=(0, 0, 0))
                normal_rgb = encode_surface_normals_rgb(
                    normal,
                    valid,
                    encoding=args.normal_encoding,
                )
                mask_img = Image.fromarray(valid.astype(np.uint8) * 255)
                for root in (depth_root, normal_root): write_png(rgb, root / condition_rel)
                write_png(Image.fromarray(depth_rgb), depth_root / target_rel)
                write_png(mask_img, depth_root / mask_rel)
                write_png(Image.fromarray(normal_rgb), normal_root / target_rel)
                write_png(mask_img, normal_root / mask_rel)
                if args.write_raw_depth:
                    raw_path = depth_root / "depth_raw" / f"{sample}.npy"; raw_path.parent.mkdir(parents=True, exist_ok=True); np.save(raw_path, depth.astype(np.float32))
                if args.write_raw_normals:
                    raw_path = normal_root / "raw_normals" / f"{sample}.npy"; raw_path.parent.mkdir(parents=True, exist_ok=True); np.save(raw_path, normal.astype(np.float16))
                panorama, yaw, elevation = panorama_orientation(key)
                metadata = {"dataset": "matterport3d", "scene": scan, "frame": key, "panorama": panorama, "yaw_index": yaw, "elevation_index": elevation, "camera_position": positions.get(key), "split": "train", "sampling_profile": args.profile, "sampling_policy": "spatial_panorama_orientation_balanced_v1", "calibration_source": "undistorted_camera_parameters", "valid_fraction": float(valid.mean()), "geometry_version": GEOMETRY_DATA_VERSION,
                            "native_modality_sizes": native_sizes, "target_grid": "official_normal_grid", "depth_resize": "nearest"}
                depth_rows.append({"id": sample, "conditioning_image": condition_rel, "image": target_rel, "valid_mask": mask_rel, "text": DEPTH_PROMPT, "task": "depth_estimation", "source_dataset": "matterport3d", "depth_encoding": "vision_banana_rgb", "metadata": {**metadata, "depth_units": "meters", "native_depth_scale": 4000.0, "depth_encoding": "vision_banana_rgb", "canonical_depth_quantity": CANONICAL_DEPTH_QUANTITY, "target_preparation": "resize_metric_depth_then_encode", "min_depth": args.min_depth, "max_depth": args.max_depth}})
                normal_rows.append({"id": sample, "conditioning_image": condition_rel, "image": target_rel, "valid_mask": mask_rel, "preprocess_target": "native_xyz_then_normalize_then_encode", "text": NORMAL_PROMPT, "task": "surface_normal_estimation", "source_dataset": "matterport3d", "metadata": {**metadata, "normal_source": "matterport3d_native_undistorted", "native_normal_decode": "value/32768-1", "coordinate_convention": CANONICAL_NORMAL_CONVENTION, **MATTERPORT_NATIVE_TO_CANONICAL.metadata(), "normal_encoding": args.normal_encoding, "normal_encoding_formula": normal_encoding_formula(args.normal_encoding), "geometry_version": geometry_data_version_for_normal_encoding(args.normal_encoding)}})
                if args.write_raw_depth:
                    depth_rows[-1]["depth_raw"] = f"depth_raw/{sample}.npy"
                if args.write_raw_normals:
                    normal_rows[-1]["raw_normals"] = f"raw_normals/{sample}.npy"
        if not depth_rows:
            raise RuntimeError(f"{scan}: all sampled frames failed validation")
        for root, rows in ((depth_root, depth_rows), (normal_root, normal_rows)):
            (root / "metadata.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        tar_directory(depth_root, depth_tar)
        tar_directory(normal_root, normal_tar)
        return depth_tar, normal_tar
    finally:
        shutil.rmtree(scene_work, ignore_errors=True)


def process_scene_worker(payload: tuple[Path, str, argparse.Namespace, Path]) -> tuple[Path, Path]:
    source, scan, args, shard_root = payload
    print(f"[{scan}] preparing", flush=True)
    return process_scene(source, scan, args, shard_root)


def merge_shards(shards: list[Path], dataset_name: str, scratch: Path, final_tar: Path) -> None:
    merge = scratch / f"merge_{dataset_name}"
    shutil.rmtree(merge, ignore_errors=True)
    root = merge / dataset_name
    root.mkdir(parents=True)
    rows: list[str] = []
    for shard in shards:
        extract = merge / "extract"
        shutil.rmtree(extract, ignore_errors=True); extract.mkdir()
        with tarfile.open(shard) as tf: safe_extract(tf, extract)
        source = extract / dataset_name
        rows.extend((source / "metadata.jsonl").read_text().splitlines())
        for subdir in ("conditioning_images", "target_images", "valid_masks", "depth_raw", "raw_normals"):
            if not (source / subdir).is_dir(): continue
            (root / subdir).mkdir(exist_ok=True)
            for item in (source / subdir).iterdir(): shutil.move(str(item), root / subdir / item.name)
    parsed = [json.loads(row) for row in rows]
    (root / "metadata.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in parsed))
    manifests = root / "manifests"; manifests.mkdir()
    (manifests / "train.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in parsed))
    tar_directory(root, final_tar)
    shutil.rmtree(merge)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-tar-root", type=Path, required=True)
    p.add_argument("--scratch-root", type=Path, required=True)
    p.add_argument("--shard-root", type=Path, required=True)
    p.add_argument("--depth-tar", type=Path, required=True)
    p.add_argument("--normals-tar", type=Path, required=True)
    p.add_argument("--scans-file", type=Path)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-scenes", type=int)
    p.add_argument("--profile", choices=("smoke", "small", "standard", "large", "full", "custom"), default="standard")
    p.add_argument("--samples-per-scene", default=None, help="Positive integer or 'all'; overrides the profile.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-depth", type=float, default=.1)
    p.add_argument("--max-depth", type=float, default=20.)
    p.add_argument("--min-valid-fraction", type=float, default=.05)
    p.add_argument("--force-scenes", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--strict-calibration", action="store_true")
    p.add_argument("--write-raw-depth", action="store_true")
    p.add_argument("--write-raw-normals", action="store_true")
    p.add_argument(
        "--normal-encoding",
        choices=SUPPORTED_NORMAL_ENCODINGS,
        default=VISION_BANANA_NORMAL_ENCODING,
    )
    p.add_argument(
        "--scene-workers",
        type=int,
        default=1,
        help="Independent source scenes to prepare concurrently.",
    )
    args = p.parse_args()
    profile_samples = {"smoke": "10", "small": "100", "standard": "360", "large": "720", "full": "all"}
    args.samples_per_scene = args.samples_per_scene or profile_samples.get(args.profile)
    if args.samples_per_scene is None:
        raise ValueError("custom profile requires --samples-per-scene")
    if args.samples_per_scene != "all" and int(args.samples_per_scene) <= 0:
        raise ValueError("--samples-per-scene must be positive or 'all'")
    if args.scene_workers <= 0:
        raise ValueError("--scene-workers must be positive")
    args.scratch_root.mkdir(parents=True, exist_ok=True); args.shard_root.mkdir(parents=True, exist_ok=True)
    if args.scans_file:
        scans = [x.strip() for x in args.scans_file.read_text().splitlines() if x.strip() and not x.startswith("#")]
        sources = [(scan, args.source_tar_root / f"{scan}.tar") for scan in scans]
    else:
        sources = [(p.stem, p) for p in sorted(args.source_tar_root.glob("*.tar"))]
    sources = sources[args.start_index:]
    if args.max_scenes is not None: sources = sources[:args.max_scenes]
    missing = [str(path) for _, path in sources if not path.is_file()]
    if missing: raise FileNotFoundError(f"missing source scene tars (first 10): {missing[:10]}")
    payloads = [
        (source, scan, args, args.shard_root)
        for scan, source in sources
    ]
    if args.scene_workers == 1:
        prepared = map(process_scene_worker, payloads)
        worker_count = 1
    else:
        worker_count = min(args.scene_workers, len(payloads))
        executor = ProcessPoolExecutor(max_workers=worker_count)
        prepared = executor.map(process_scene_worker, payloads, chunksize=1)
    print(
        f"Preparing {len(payloads)} Matterport3D scenes with "
        f"{worker_count} worker(s)",
        flush=True,
    )
    try:
        pairs = list(prepared)
    finally:
        if args.scene_workers != 1:
            executor.shutdown(wait=True, cancel_futures=True)
    depth_shards = [pair[0] for pair in pairs]
    normal_shards = [pair[1] for pair in pairs]
    if not depth_shards: raise RuntimeError("no scenes selected")
    merge_shards(depth_shards, "matterport3d_depth", args.scratch_root, args.depth_tar)
    merge_shards(normal_shards, "matterport3d_surface_normals", args.scratch_root, args.normals_tar)
    print(f"wrote {args.depth_tar}\nwrote {args.normals_tar}")


if __name__ == "__main__": main()
