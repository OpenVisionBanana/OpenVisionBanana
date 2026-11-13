#!/usr/bin/env python3
"""Export filtered geometry archives with masks and no runtime curation.

Only training data enter this step. RGB, targets, masks and raw fields receive
the same crop. Invalid interior pixels remain excluded from supervision.
"""
from __future__ import annotations
import argparse
from collections import Counter
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from common.geometry_protocol_prompts import DETAILED_INVERSE_DEPTH_PROMPT, CONCISE_NORMAL_PROMPT
from common.surface_normal_encoding import decode_surface_normals_rgb
from dataset_prep.depth.encoding import encode_depth_to_vision_banana_rgb
from common.training.geometry_curation import edge_trim, quality_signals
from scripts.dataset_prep.build_prepared_tar_index import resolve_member
from scripts.dataset_prep.prepare_geometry_training import add_bytes, png, npy
from common.release_data import sha256


def crop_metadata(row, box):
    metadata = dict(row.get("metadata", {}))
    intrinsics = metadata.get("intrinsics")
    if isinstance(intrinsics, dict):
        camera = dict(intrinsics)
        values = np.asarray([camera[key] for key in ("fx", "fy", "cx", "cy")], dtype=float)
        if not np.isfinite(values).all() or min(values[:2]) <= 0:
            raise ValueError("Expected finite camera intrinsics with positive focal lengths")
        metadata["source_intrinsics"] = dict(camera)
        camera["cx"], camera["cy"] = values[2] - box[0], values[3] - box[1]
        metadata["intrinsics"] = camera
    elif intrinsics is not None:
        matrix = np.asarray(intrinsics, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError("Expected 3x3 camera intrinsics")
        metadata["source_intrinsics"] = matrix.tolist()
        matrix[0, 2] -= box[0]
        matrix[1, 2] -= box[1]
        metadata["intrinsics"] = matrix.tolist()
    metadata.update(preparation_policy="official-base-clean-v1", source_crop_box=list(box))
    return metadata


def publish(partial, output):
    partial.replace(output)
    receipt = Path(str(output) + ".sha256")
    temporary = receipt.with_suffix(receipt.suffix + ".partial")
    temporary.write_text(sha256(output) + "  " + output.name + "\n")
    temporary.replace(receipt)


def source_rows(handle):
    members = {m.name for m in handle.getmembers() if m.isfile()}
    rows = {}
    for name in sorted(n for n in members if PurePosixPath(n).name == "metadata.jsonl"):
        for line in handle.extractfile(name):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("id", ""))
            if not key or key in rows:
                raise ValueError(f"Missing or duplicate sample ID: {key}")
            for field in ("image", "conditioning_image", "loss_mask", "valid_mask", "mask", "depth_raw", "raw_normals"):
                if row.get(field):
                    row[field] = resolve_member(row[field], name, members)
            rows[key] = row
    if not rows:
        raise ValueError("Source archive contains no training metadata")
    return rows


def image(handle, member, mode):
    with Image.open(handle.extractfile(member)) as im:
        return im.convert(mode)


def replica_features(row, source):
    rgb = image(source, row["conditioning_image"], "RGB")
    gray = np.asarray(rgb, dtype=np.float32).mean(-1)
    lap = gray[1:-1,:-2]+gray[1:-1,2:]+gray[:-2,1:-1]+gray[2:,1:-1]-4*gray[1:-1,1:-1]
    valid = np.asarray(image(source, row.get("loss_mask") or row["valid_mask"], "L")) > 0
    if not row.get("depth_raw"):
        raise ValueError("Replica quality selection requires raw metric depth")
    depth = np.load(io.BytesIO(source.extractfile(row["depth_raw"]).read()))
    close = float((depth[valid] < .5).mean()) if valid.any() else 1.
    score = 10*valid.mean()+min(gray.std(),40)/40+min(math.log1p(lap.var()),5)/5-close
    meta = row["metadata"]
    cell = tuple(math.floor(float(v)/.75) for v in meta["camera_position"]) + (
        math.floor((float(meta["yaw_degrees"]) % 360)/45), math.floor((float(meta["pitch_degrees"])+90)/30))
    pixels = np.asarray(rgb.convert("L").resize((9,8), Image.Resampling.BILINEAR))
    dhash = int.from_bytes(np.packbits(pixels[:,1:] > pixels[:,:-1]).tobytes(), "big")
    return dict(score=float(score), cell=cell, dhash=dhash, color=np.asarray(rgb).mean((0,1)))


def materialize(dataset, source_root, output_root, overwrite=False):
    if dataset in {"nyuv2", "diode", "scannet", "vkitti2"}:
        return materialize_independent(dataset, source_root, output_root, overwrite)
    from contextlib import ExitStack
    outputs = [output_root / t / f"{dataset}.tar" for t in ("Depth", "Normals")]
    if any(p.exists() for p in outputs) and not overwrite:
        raise FileExistsError("Filtered geometry already exists: " + str(outputs))
    with ExitStack() as stack:
        sources = [stack.enter_context(tarfile.open(source_root / t / f"{dataset}.tar")) for t in ("Depth", "Normals")]
        rows = [source_rows(s) for s in sources]
        common = sorted(rows[0].keys() & rows[1].keys())
        if not common:
            raise ValueError("Depth/normal sources have no paired training IDs")
        if len(rows[0].keys() ^ rows[1].keys()) > max(10, max(len(r) for r in rows)*.01):
            raise ValueError(f"Widespread depth/normal ID mismatch: {dataset}")
        replica = {key: replica_features(rows[0][key], sources[0]) for key in common} if dataset == "replica" else {}
        if replica:
            common.sort(key=lambda key: (-replica[key]["score"], key))
        reports = []
        metadata = [[], []]
        for p in outputs:
            p.parent.mkdir(parents=True, exist_ok=True)
        partials = [p.with_suffix(".tar.partial") for p in outputs]
        archives = [stack.enter_context(tarfile.open(p, "w")) for p in partials]
        scene_counts = Counter()
        pose_counts = Counter()
        appearances = {}
        for key in common:
            items = [r[key] for r in rows]
            triplets = []
            for source, row in zip(sources, items):
                mask = row.get("loss_mask") or row.get("valid_mask") or row.get("mask")
                if not mask:
                    raise ValueError(f"Missing validity mask: {dataset}/{key}")
                triplet = [image(source, row["conditioning_image"], "RGB"), image(source, row["image"], "RGB"), image(source, mask, "L")]
                if len({im.size for im in triplet}) != 1:
                    raise ValueError(f"RGB/target/mask dimensions differ: {dataset}/{key}")
                values = np.asarray(triplet[2])
                if not np.isin(values, (0, 255)).all():
                    raise ValueError(f"Nonbinary validity mask: {dataset}/{key}")
                triplets.append(triplet)
            if triplets[0][0].size != triplets[1][0].size or not np.array_equal(triplets[0][0], triplets[1][0]):
                raise ValueError(f"Depth/normal RGB registration mismatch: {dataset}/{key}")
            valid = (np.asarray(triplets[0][2]) > 0) & (np.asarray(triplets[1][2]) > 0)
            box, _ = edge_trim(valid, max_edge_fraction=.25, min_retained_area=.75,
                               max_strip_valid_fraction=.5, min_side=32)
            left, top, right, bottom = box
            retained = valid[top:bottom, left:right]
            max_invalid = .01 if dataset == "replica" else .25 if dataset in {"nyuv2", "diode", "scannet", "vkitti2"} else .1
            reason = None
            if not retained.any() or 1-retained.mean() > max_invalid:
                reason = "insufficient_valid_geometry"
            normal_rgb = np.asarray(triplets[1][1])[top:bottom, left:right]
            _, good_normals = decode_surface_normals_rgb(normal_rgb, encoding=items[1].get("normal_encoding") or
                items[1].get("metadata", {}).get("normal_encoding", "vision_banana_x_left_red"), min_norm=.05)
            normal_mask = np.asarray(triplets[1][2])[top:bottom, left:right] > 0
            if (normal_mask & ~good_normals).any():
                reason = "undefined_normals_marked_valid"
            # Preserve the shared-source quality screen. The NZS base-clean
            # export below intentionally does not filter by surface roughness.
            view_size = (128, 128)
            review_rgb = triplets[0][0].crop(box).resize(view_size, Image.Resampling.BILINEAR)
            review_normal = triplets[1][1].crop(box).resize(view_size, Image.Resampling.NEAREST)
            review_valid = np.asarray(triplets[1][2].crop(box).resize(view_size, Image.Resampling.NEAREST)) > 0
            signals, flags, _ = quality_signals(review_rgb, review_normal, review_valid, normals=True)
            if "rough_normal_field_at_review_resolution" in flags:
                reason = "rough_normal_field_conservative_quality_filter"
            if signals["rgb_std"] < 5 and signals["laplacian_variance"] < 1:
                reason = "near_featureless_low_contrast_view"
            scene = items[0].get("metadata", {}).get("scene", "")
            if dataset == "replica":
                if items[0].get("metadata", {}).get("rgb_source") != "replica_sdk_ptex":
                    raise ValueError("Default Replica export requires registered Ptex RGB")
                depth_name = items[0].get("depth_raw")
                if not depth_name:
                    raise ValueError("Replica filtering requires raw metric depth")
                depth = np.load(io.BytesIO(sources[0].extractfile(depth_name).read()))[top:bottom, left:right]
                values = depth[retained]
                if len(values) and (np.median(values) < .75 or (values < .5).mean() > .35):
                    reason = "close_surface_dominates"
                if scene_counts[scene] >= 250:
                    reason = "replica_scene_cap"
                feature = replica[key]
                if pose_counts[(scene, feature["cell"])] >= 2:
                    reason = "similar_pose"
                if any(bin(feature["dhash"] ^ old_hash).count("1") <= 4 and np.linalg.norm(feature["color"]-color) < 12
                       for old_hash, color in appearances.get(scene, [])):
                    reason = "similar_appearance"
            reports.append(dict(id=key, retained=reason is None, reason=reason, crop=list(box)))
            if reason:
                continue
            scene_counts[scene] += 1
            if dataset == "replica":
                pose_counts[(scene, replica[key]["cell"])] += 1
                appearances.setdefault(scene, []).append((replica[key]["dhash"], replica[key]["color"]))
            for task_index, (source, row, triplet, archive) in enumerate(zip(sources, items, triplets, archives)):
                out = {k: v for k, v in row.items() if k not in {"valid_mask", "mask", "_metadata_member", "_metadata_line"}}
                if task_index == 0 and row.get("depth_encoding") != "vision_banana_rgb":
                    if not row.get("depth_raw"):
                        raise ValueError(f"Metric depth is required to correct the training encoding: {dataset}/{key}")
                    depth = np.load(io.BytesIO(source.extractfile(row["depth_raw"]).read()))
                    encoded = encode_depth_to_vision_banana_rgb(depth, np.asarray(triplet[2]) > 0)
                    triplet[1] = Image.fromarray(encoded)
                    out["depth_encoding"] = "vision_banana_rgb"
                for field, im in zip(("conditioning_image", "image", "loss_mask"), triplet):
                    out[field] = f"samples/{key}/{field}.png"
                    add_bytes(archive, out[field], png(np.asarray(im.crop(box))))
                for field in ("depth_raw", "raw_normals"):
                    if row.get(field):
                        raw = np.load(io.BytesIO(source.extractfile(row[field]).read()))
                        if isinstance(raw, np.lib.npyio.NpzFile):
                            buffer = io.BytesIO()
                            np.savez_compressed(buffer, **{n: raw[n][top:bottom, left:right] for n in raw.files})
                            data, suffix = buffer.getvalue(), ".npz"
                        else:
                            data, suffix = npy(raw[top:bottom, left:right]), ".npy"
                        out[field] = f"samples/{key}/{field}{suffix}"
                        add_bytes(archive, out[field], data)
                out["text"] = DETAILED_INVERSE_DEPTH_PROMPT if task_index == 0 else CONCISE_NORMAL_PROMPT
                out["metadata"] = crop_metadata(row, box)
                metadata[task_index].append(out)
        if not metadata[0] or not metadata[1]:
            raise ValueError(f"Filtering retained no paired geometry: {dataset}")
        for archive, values in zip(archives, metadata):
            add_bytes(archive, "metadata.jsonl", ("\n".join(json.dumps(r) for r in values)+"\n").encode())
    for partial, output in zip(partials, outputs):
        publish(partial, output)
    report = output_root.parent / "manifests" / f"{dataset}_filtering.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(dict(policy="official-base-clean-v1", source_counts=[len(x) for x in rows],
        retained=len(metadata[0]), unpaired_ids=sorted(rows[0].keys() ^ rows[1].keys()), samples=reports), indent=2)+"\n")


def materialize_independent(dataset, source_root, output_root, overwrite=False):
    """Keep NZS depth and normal eligibility independent, as in base-clean."""
    outputs = [output_root / t / f"{dataset}.tar" for t in ("Depth", "Normals")]
    if any(p.exists() for p in outputs) and not overwrite:
        raise FileExistsError(f"Filtered {dataset} outputs already exist")
    reports = {}
    partials = []
    for task_index, (task, output) in enumerate(zip(("Depth", "Normals"), outputs)):
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_suffix(".tar.partial")
        partials.append(partial)
        rows_out, decisions = [], []
        with tarfile.open(source_root / task / f"{dataset}.tar") as source, tarfile.open(partial, "w") as target:
            for key, row in source_rows(source).items():
                mask_name = row.get("loss_mask") or row.get("valid_mask") or row.get("mask")
                if not mask_name:
                    raise ValueError(f"Missing validity mask: {dataset}/{task}/{key}")
                rgb, gt, mask = image(source, row["conditioning_image"], "RGB"), image(source, row["image"], "RGB"), image(source, mask_name, "L")
                if len({rgb.size, gt.size, mask.size}) != 1 or not np.isin(np.asarray(mask), (0, 255)).all():
                    raise ValueError(f"Invalid geometry triplet: {dataset}/{task}/{key}")
                valid = np.asarray(mask) > 0
                box, _ = edge_trim(valid, max_edge_fraction=.25, min_retained_area=.6, max_strip_valid_fraction=.5, min_side=32)
                left, top, right, bottom = box
                kept = valid[top:bottom, left:right]
                reason = None
                if not kept.any() or 1-kept.mean() > .25:
                    reason = "insufficient_valid_geometry"
                if task == "Normals":
                    _, defined = decode_surface_normals_rgb(np.asarray(gt)[top:bottom,left:right], encoding="vision_banana_x_left_red", min_norm=.05)
                    if (kept & ~defined).any():
                        reason = "undefined_normals_marked_valid"
                decisions.append(dict(id=key, retained=reason is None, reason=reason, crop=list(box)))
                if reason:
                    continue
                out = {k:v for k,v in row.items() if k not in {"valid_mask", "mask", "_metadata_member", "_metadata_line"}}
                for field, im in (("conditioning_image", rgb), ("image", gt), ("loss_mask", mask)):
                    out[field] = f"samples/{key}/{field}.png"
                    add_bytes(target, out[field], png(np.asarray(im.crop(box))))
                for field in ("depth_raw", "raw_normals"):
                    if row.get(field):
                        array = np.load(io.BytesIO(source.extractfile(row[field]).read()))
                        out[field] = f"samples/{key}/{field}.npy"
                        add_bytes(target, out[field], npy(array[top:bottom,left:right]))
                out["text"] = DETAILED_INVERSE_DEPTH_PROMPT if task == "Depth" else CONCISE_NORMAL_PROMPT
                out["metadata"] = crop_metadata(row, box)
                rows_out.append(out)
            if not rows_out:
                raise ValueError(f"No eligible {dataset}/{task} rows")
            add_bytes(target, "metadata.jsonl", ("\n".join(json.dumps(r) for r in rows_out)+"\n").encode())
        reports[task] = dict(retained=len(rows_out), samples=decisions)
    for partial, output in zip(partials, outputs):
        publish(partial, output)
    path = output_root.parent / "manifests" / f"{dataset}_filtering.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(policy="official-base-clean-v1", tasks=reports), indent=2)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    materialize(a.dataset, a.source_root, a.output_root, a.overwrite)
