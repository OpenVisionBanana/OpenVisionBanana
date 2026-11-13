#!/usr/bin/env python3
"""Prepare NZS geometry training samples from official training splits.

Dataset readers and normal estimators come from the existing converters.
This adapter writes the shared training archive contract for both tasks.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import io
import json
from pathlib import Path
import re
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from common.geometry_protocol_prompts import DETAILED_INVERSE_DEPTH_PROMPT, CONCISE_NORMAL_PROMPT
from common.surface_normal_encoding import encode_surface_normals_rgb, VISION_BANANA_NORMAL_ENCODING
from dataset_prep.depth.encoding import encode_depth_to_vision_banana_rgb
from dataset_prep.surface_normals.prepare_nyu_surface_normals import compute_normals
from dataset_prep.geometry_conventions import (
    CAMERA_FACING_GEOMETRY_DATA_VERSION, CANONICAL_NORMAL_CONVENTION,
    CANONICAL_DEPTH_QUANTITY, CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL,
    NormalTransform,
    IMAGE_Y_DOWN_Z_AWAY_TO_CANONICAL,
)


DIODE_NORMAL_TO_CANONICAL = NormalTransform(
    (0, 1, 2), (1, 1, 1), CANONICAL_NORMAL_CONVENTION,
    "Official DIODE float XYZ targets retained without axis changes. "
    "Historical fixed geometry audit reproduced on 16 frames and independently "
    "expanded to 80 indoor/outdoor frames: identity wins globally with a 24.82-degree "
    "axis margin, and projective camera-z wins over ray range by 4.78 degrees. "
    "No per-image axis selection or calibration adjustment. "
    "See validation/data_setup/20261029/diode_convention_reconciled.json. "
    "This is a reproduced empirical contract, not an author certification.",
    "diode_identity_reproduced_v2",
)


def add_bytes(archive, name, data):
    info = tarfile.TarInfo(name)
    info.size, info.mode = len(data), 0o644
    archive.addfile(info, io.BytesIO(data))


def png(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def npy(array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def nyu_samples(raw, limit):
    import h5py
    from dataset_prep.surface_normals.prepare_nyu_surface_normals import load_split_indices, as_rgb_image, as_depth_hw
    indices = load_split_indices(raw / "splits.mat")
    if set(indices["train"]) & set(indices["test"]):
        raise ValueError("NYU train/test split overlap")
    bundles = list(raw.glob("dsine_eval.zip")) or list((raw.parent / "dsine").rglob("dsine_eval.zip"))
    if len(bundles) != 1:
        raise ValueError("NYUv2 normal supervision requires the author's dsine_eval.zip. "
                         "Run download_data.py --datasets nyuv2 or stage it under Shared_geo/dsine/. "
                         "Depth-derived normals are not substituted.")
    with h5py.File(raw / "nyu_depth_v2_labeled.mat") as source, zipfile.ZipFile(bundles[0]) as labels:
        for index in sorted(indices["train"])[:limit]:
            rgb = np.asarray(as_rgb_image(source["images"][index]))
            depth = as_depth_hw(source["depths"][index])
            sparse = as_depth_hw(source["rawDepths"][index])
            valid = np.isfinite(sparse) & (sparse > .001) & (depth > .001) & (depth < 10)
            k = np.array([[518.8579, 0, 325.5824], [0, 519.4696, 253.7362], [0, 0, 1]])
            prefix = f"dsine_eval/nyuv2/train/{index:06d}"
            encoded = np.asarray(Image.open(io.BytesIO(labels.read(prefix + "_normal.png"))).convert("RGB"))
            source_rgb = np.asarray(Image.open(io.BytesIO(labels.read(prefix + "_img.png"))).convert("RGB"))
            if source_rgb.shape != rgb.shape:
                raise ValueError(f"DSINE/MAT RGB dimensions mismatch for NYUv2 index {index+1}")
            if encoded.shape != rgb.shape:
                raise ValueError(f"Unregistered DSINE normal for NYUv2 index {index+1}")
            normal_valid = encoded.astype(np.uint16).sum(-1) > 0
            native = encoded.astype(np.float32) / 127.5 - 1.
            normals = IMAGE_Y_DOWN_Z_AWAY_TO_CANONICAL.apply(native, normal_valid)
            yield f"nyu_{index+1:04d}", rgb, depth, valid, normals, normal_valid, k, dict(
                source_index_1based=index+1, _normal_rgb=source_rgb,
                normal_ground_truth="GeoNet labels distributed by DSINE",
                normal_transform=IMAGE_Y_DOWN_Z_AWAY_TO_CANONICAL.metadata(),
                geometry_version="nyuv2_geonet_dsine_v1", canonical_normal_convention=CANONICAL_NORMAL_CONVENTION)


def diode_row(stem, rgb, depth, mask, normals):
    depth = np.asarray(depth).squeeze().astype(np.float32)
    valid = np.asarray(mask).squeeze().astype(bool) & np.isfinite(depth) & (depth > .001) & (depth < 350)
    normals = np.asarray(normals).squeeze().astype(np.float32)
    nvalid = valid & np.isfinite(normals).all(-1) & (np.linalg.norm(normals, axis=-1) > .01)
    k = np.array([[886.81, 0, 512], [0, 927.06, 384], [0, 0, 1]])
    return stem, rgb, depth, valid, normals, nvalid, k, dict(
        DIODE_NORMAL_TO_CANONICAL.metadata(), split="train",
        normal_ground_truth="official_diode_normal_npy",
        normal_transform=DIODE_NORMAL_TO_CANONICAL.metadata())


def diode_archive_samples(raw, limit, scratch):
    """Stream original compressed archives with one bounded-inode sample spool."""
    records = {}
    with tempfile.TemporaryFile(dir=scratch) as spool:
        with tarfile.open(raw / "train.tar.gz", "r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = member.name.removeprefix("./")
                if not name.startswith("train/"):
                    continue
                suffix = next((s for s in ("_depth_mask.npy", "_depth.npy", ".png") if name.endswith(s)), None)
                if suffix is None:
                    continue
                key = name[:-len(suffix)]
                if key not in records and limit is not None and len(records) >= limit:
                    continue
                fields = records.setdefault(key, {})
                if suffix in fields:
                    raise ValueError(f"Duplicate official DIODE member: {name}")
                data = archive.extractfile(member).read()
                fields[suffix] = (spool.tell(), len(data))
                spool.write(data)
                if limit is not None and len(records) == limit and all(len(v) == 3 for v in records.values()):
                    break
        if not records or any(len(v) != 3 for v in records.values()):
            raise ValueError("Incomplete official DIODE RGB/depth/mask samples")
        spool.flush()
        missing = set(records)
        with tarfile.open(raw / "train_normals.tar.gz", "r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = member.name.removeprefix("./")
                if not name.endswith("_normal.npy"):
                    continue
                key = name.removesuffix("_normal.npy")
                if key not in records:
                    continue
                if key not in missing:
                    raise ValueError(f"Duplicate official DIODE normal: {key}")
                def read(suffix):
                    offset, length = records[key][suffix]
                    spool.seek(offset)
                    return io.BytesIO(spool.read(length))
                rgb = np.asarray(Image.open(read(".png")).convert("RGB"))
                yield diode_row(Path(key).name, rgb, np.load(read("_depth.npy")),
                                np.load(read("_depth_mask.npy")), np.load(io.BytesIO(archive.extractfile(member).read())))
                missing.remove(key)
                if not missing:
                    break
        if missing:
            raise ValueError(f"Missing official DIODE training normals: {sorted(missing)[:3]}")


def diode_samples(raw, limit, scratch=None):
    # RGB/depth and normals live in separate official archives with matching IDs.
    paths = sorted((raw / "depth_sources/train").rglob("*_depth.npy"))
    if not paths:
        if (raw / "train.tar.gz").is_file() and (raw / "train_normals.tar.gz").is_file():
            yield from diode_archive_samples(raw, limit, scratch)
            return
        raise FileNotFoundError("Expected original DIODE training archives or extracted training sources under " + str(raw))
    for path in paths[:limit]:
        relative = path.relative_to(raw / "depth_sources")
        stem = path.name.removesuffix("_depth.npy")
        rgb_path = path.with_name(stem + ".png")
        mask_path = path.with_name(stem + "_depth_mask.npy")
        normal_path = raw / "normal_sources" / relative.parent / (stem + "_normal.npy")
        if not normal_path.exists():
            raise FileNotFoundError(normal_path)
        # Official DIODE normals already use the canonical camera basis.
        yield diode_row(stem, np.asarray(Image.open(rgb_path).convert("RGB")),
                        np.load(path), np.load(mask_path), np.load(normal_path))


def vkitti_samples(raw, limit):
    from dataset_prep.vkitti2.prepare_vkitti2_eval import discover_rows, read_tar_member, load_depth_png_from_bytes, derive_normals_from_depth
    with ExitStack() as stack:
        rgb, dep, text = [stack.enter_context(tarfile.open(next(raw.glob(pattern))))
                          for pattern in ("*rgb*.tar*", "*depth*.tar*", "*textgt*.tar*")]
        variants = ("15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right", "fog", "morning", "overcast", "rain", "sunset")
        rows = discover_rows(rgb, dep, text, ("Scene01", "Scene02", "Scene06", "Scene18", "Scene20"), variants, ("Camera_0",), limit)
        for row in rows:
            depth = load_depth_png_from_bytes(read_tar_member(dep, row.depth_member), 100.)
            valid = np.isfinite(depth) & (depth > .1) & (depth < 80)
            normal, nvalid = derive_normals_from_depth(np.where(valid, depth, np.nan), row.intrinsics, 2, .05)
            image = np.asarray(Image.open(io.BytesIO(read_tar_member(rgb, row.image_member))).convert("RGB"))
            yield row.sample_id, image, depth, valid, normal, nvalid, row.intrinsics, dict(scene=row.scene, variant=row.variant)


def scannet_samples(raw, split_root, limit):
    from dataset_prep.scannet.prepare_scannet_depth_normals import find_frames_root, discover_raw_frames, read_raw_depth_m, copy_raw_color
    from dataset_prep.surface_normals.edge_aware_plane import plane_normals
    from common.official_sources import scannet_exclusions
    heldout = scannet_exclusions(split_root)
    scene_file = next(raw.rglob("scannetv2_train.txt"))
    scenes = [s for s in scene_file.read_text().split() if s.split("_")[0] not in heldout]
    frames = discover_raw_frames(find_frames_root(raw, None, extract_zip=False), scenes,
        min_depth=.001, max_depth=10., min_valid_fraction=.05, strict=True)
    for frame in frames[:limit]:
        if frame.scene_id.split("_")[0] in heldout:
            raise ValueError("ScanNet benchmark-location overlap")
        if frame.zip_path:
            with zipfile.ZipFile(frame.zip_path) as archive:
                image = np.asarray(Image.open(io.BytesIO(archive.read(frame.color_path))).convert("RGB"))
        else:
            image = np.asarray(Image.open(frame.color_path).convert("RGB"))
        depth = read_raw_depth_m(frame)
        valid = np.isfinite(depth) & (depth > .001) & (depth < 10.)
        if image.shape[:2] != depth.shape:
            # The existing ScanNet converter uses the registered depth grid.
            image = np.asarray(Image.fromarray(image).resize((depth.shape[1], depth.shape[0]), Image.Resampling.LANCZOS))
        normal, nvalid = plane_normals(np.where(valid, depth, np.nan), fx=frame.fx, fy=frame.fy, cx=frame.cx, cy=frame.cy)
        k = np.array([[frame.fx, 0, frame.cx], [0, frame.fy, frame.cy], [0, 0, 1]])
        yield f"{frame.scene_id}_{int(frame.frame_id):06d}", image, depth, valid, normal, nvalid, k, dict(scene=frame.scene_id, normal_method="edge_aware_plane")


def prepare(name, raw, output, limit, split_root, overwrite=False):
    if name == "scannet":
        from common.official_sources import scannet_exclusions
        scannet_exclusions(split_root)
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / task / f"{name}.tar" for task in ("Depth", "Normals")]
    if any(p.exists() for p in paths) and not overwrite:
        raise FileExistsError("Training converter output already exists: " + str(paths))
    partials = [p.with_suffix(".tar.partial") for p in paths]
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
    rows = [[], []]
    with ExitStack() as stack:
        from common.node_scratch import workspace
        spool_root = stack.enter_context(workspace("diode-spool")) if name == "diode" else None
        generators = {"nyuv2": lambda: nyu_samples(raw, limit), "diode": lambda: diode_samples(raw, limit, spool_root),
                      "vkitti2": lambda: vkitti_samples(raw, limit), "scannet": lambda: scannet_samples(raw, split_root, limit)}
        archives = [stack.enter_context(tarfile.open(p, "w")) for p in partials]
        for key, rgb, depth, valid, normals, nvalid, k, metadata in generators[name]():
            normal_rgb = metadata.pop("_normal_rgb", rgb)
            if rgb.shape[:2] != depth.shape or valid.shape != depth.shape:
                raise ValueError(f"Unregistered geometry sample: {key}")
            if normals is None:
                if name == "nyuv2":
                    raise ValueError("Missing supplied GeoNet/DSINE normals; depth fallback is disabled")
                normals, nvalid = compute_normals(np.where(valid, depth, np.nan), fx=k[0,0], fy=k[1,1], cx=k[0,2], cy=k[1,2], radius=2)
            length = np.linalg.norm(normals, axis=-1)
            nvalid = nvalid & (True if name == "nyuv2" else valid) & np.isfinite(length) & (length > .05)
            normals = np.divide(normals, length[..., None], out=np.zeros_like(normals), where=length[..., None] > 1e-8)
            for index, archive in enumerate(archives):
                mask = valid if index == 0 else nvalid
                if not mask.any():
                    continue
                target = (encode_depth_to_vision_banana_rgb(depth, mask)
                          if index == 0 else encode_surface_normals_rgb(normals, mask, encoding=VISION_BANANA_NORMAL_ENCODING))
                prefix = f"samples/{key}"
                values = {"conditioning_image": prefix+"/rgb.png", "image": prefix+"/target.png", "loss_mask": prefix+"/mask.png"}
                for field, array in (("conditioning_image", rgb if index == 0 else normal_rgb), ("image", target), ("loss_mask", mask.astype(np.uint8)*255)):
                    add_bytes(archive, values[field], png(array))
                raw_key = "depth_raw" if index == 0 else "raw_normals"
                values[raw_key] = prefix + "/raw.npy"
                add_bytes(archive, values[raw_key], npy(depth if index == 0 else normals))
                rows[index].append(dict(values, id=key, source_dataset=name,
                    task="depth_estimation" if index == 0 else "surface_normals",
                    text=DETAILED_INVERSE_DEPTH_PROMPT if index == 0 else CONCISE_NORMAL_PROMPT,
                    depth_encoding="vision_banana_rgb" if index == 0 else None,
                    normal_encoding=VISION_BANANA_NORMAL_ENCODING if index else None,
                    metadata=dict(metadata, **({
                        "geometry_version": CAMERA_FACING_GEOMETRY_DATA_VERSION,
                        "canonical_normal_convention": CANONICAL_NORMAL_CONVENTION,
                        "normal_transform": CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL.metadata(),
                    } if name in {"vkitti2", "scannet"} else {}),
                                  canonical_depth_quantity=CANONICAL_DEPTH_QUANTITY,
                                  split="train", intrinsics=k.tolist(), normal_orientation="face_camera_ray",
                                  training_depth_encoding="relative_inverse_grayscale", depth_units="meters")))
        for index, archive in enumerate(archives):
            if not rows[index]:
                raise ValueError(f"Empty training dataset: {name}/{index}")
            add_bytes(archive, "metadata.jsonl", ("\n".join(json.dumps(r) for r in rows[index])+"\n").encode())
    for partial, path in zip(partials, paths):
        partial.replace(path)


if __name__ == "__main__":
    import zipfile
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("nyuv2", "diode", "vkitti2", "scannet"), required=True)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--split-root", type=Path)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    prepare(args.dataset, args.raw_dir, args.output, args.limit, args.split_root, args.overwrite)
