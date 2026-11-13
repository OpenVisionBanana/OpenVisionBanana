#!/usr/bin/env python3
"""Materialize Hypersim's HDF5 segmentation channels into the PNG layout the
semseg / instance-seg converters already read.

Hypersim ships every channel as per-frame HDF5. This script normalizes one scene
tree into the same layout download_datasets_waymo.sh writes, which both
converters read through their panoptic code path:

  images/{train,val}/<scene>_<cam>_<frame>.jpg     tonemapped RGB
  panoptic/{train,val}/<scene>_<cam>_<frame>.png   uint16, class*1000 + instance
  id2label.json                                    raw class names + divisor
  things.json                                      observed instance-eligible ids

Source channels per frame (see the layout in prepare_hypersim_depth_normals.py):
  scene_cam_<NN>_final_hdf5/frame.<NNNN>.color.hdf5             HDR linear float16
  scene_cam_<NN>_geometry_hdf5/frame.<NNNN>.semantic.hdf5       int16 NYU40, -1 unlabeled
  scene_cam_<NN>_geometry_hdf5/frame.<NNNN>.semantic_instance.hdf5  int16, -1 = none
  scene_cam_<NN>_geometry_hdf5/frame.<NNNN>.render_entity_id.hdf5   int16, -1 = no geometry

``color.hdf5`` is linear HDR radiance in float16 with values well above 1.0, so it
is tonemapped with Hypersim's reference recipe (CCIR601-brightness percentile scale
plus gamma, the one behind the released ``*.tonemap.jpg`` previews). See ``tonemap``.

Instance ids are compacted per (image, class) to 1..N so the packed value stays in
uint16 (40 * 1000 + 999 = 40999), as in the Waymo encoding. Instance 0 means "no
instance" (stuff pixel); packed value 0 means unlabeled (NYU40 raw ids start at 1).

Frame discovery, per-scene balanced sampling and the scene-disjoint train/val split
are shared with prepare_hypersim_depth_normals.py.

Usage:
  python -m dataset_prep.hypersim.prepare_hypersim_segmentation \\
      --raw-root  <hypersim>/work/raw_extracted \\
      --out-root  <staging>/hypersim_seg \\
      --all-frames --num-workers 32
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

from common.semseg import taxonomy
from common.hypersim_zip import zip_frames
from dataset_prep.hypersim.prepare_hypersim_depth_normals import (
    IMAGE_RESAMPLING,
    clean_output,
    frame_token_from_color,
    read_hdf5_array,
    sample_frames,
    scene_dir_from_final_dir,
    sort_key,
    write_json,
    write_prepared_tar,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda iterable, **_: iterable  # noqa: E731


DIVISOR = taxonomy.HYPERSIM_PANOPTIC_DIVISOR


@dataclass(frozen=True)
class HypersimSegFrame:
    """Mirrors HypersimFrame's scene/camera/frame fields so sample_frames() works."""

    sample_id: str
    scene: str
    camera: str
    frame: str
    scene_dir: Path
    rgb_path: Path
    semantic_path: Path
    instance_path: Path
    entity_path: Path
    split: str = "train"


def discover_frames(raw_root: Path) -> list[HypersimSegFrame]:
    """Every frame carrying all four channels this converter needs.

    Sibling of prepare_hypersim_depth_normals.discover_frames; that one gates on
    depth+normals, this one on semantic+instance+entity (the entity map is what the
    reference tonemapper uses to mask out no-geometry pixels).
    """
    raw_root = Path(raw_root).expanduser().resolve()
    frames: list[HypersimSegFrame] = []
    for rgb_path in sorted(raw_root.glob("**/scene_cam_*_final_hdf5/frame.*.color.hdf5"), key=sort_key):
        final_dir = rgb_path.parent
        camera = final_dir.name.replace("_final_hdf5", "")
        geometry_dir = final_dir.parent / f"{camera}_geometry_hdf5"
        frame = frame_token_from_color(rgb_path)
        semantic_path = geometry_dir / f"{frame}.semantic.hdf5"
        instance_path = geometry_dir / f"{frame}.semantic_instance.hdf5"
        entity_path = geometry_dir / f"{frame}.render_entity_id.hdf5"
        if not (semantic_path.is_file() and instance_path.is_file() and entity_path.is_file()):
            continue
        scene_dir = scene_dir_from_final_dir(final_dir)
        scene = scene_dir.name
        # The camera dir is "scene_cam_NN"; drop the constant "scene_" so the stem
        # reads <scene>_cam_NN_frame_NNNN and the converters can recover the scene
        # by splitting on "_cam_" (they group by scene to sample evenly across rooms).
        camera_label = camera[len("scene_"):] if camera.startswith("scene_") else camera
        frames.append(
            HypersimSegFrame(
                sample_id=f"{scene}_{camera_label}_{frame.replace('.', '_')}",
                scene=scene,
                camera=camera,
                frame=frame,
                scene_dir=scene_dir,
                rgb_path=rgb_path,
                semantic_path=semantic_path,
                instance_path=instance_path,
                entity_path=entity_path,
            )
        )
    existing = {f.sample_id for f in frames}
    for scene_dir, camera, frame, paths in zip_frames(raw_root, (".semantic.hdf5", ".semantic_instance.hdf5", ".render_entity_id.hdf5")):
        scene = scene_dir.name
        camera_label = camera.removeprefix("scene_")
        sample_id = f"{scene}_{camera_label}_{frame.replace('.', '_')}"
        if sample_id in existing:
            raise ValueError(f"Duplicate extracted/archived Hypersim frame: {sample_id}")
        existing.add(sample_id)
        frames.append(HypersimSegFrame(sample_id, scene, camera, frame, raw_root / str(scene_dir), *paths))
    return sorted(frames, key=lambda f: (f.scene, f.camera, f.frame))


# --------------------------------------------------------------------------- #
# tonemapping                                                                  #
# --------------------------------------------------------------------------- #
TONEMAP_GAMMA = 1.0 / 2.2
TONEMAP_PERCENTILE = 90
TONEMAP_BRIGHTNESS_TARGET = 0.8  # desired post-scale brightness at that percentile
TONEMAP_EPS = 1e-4


def tonemap(rgb_hdr: np.ndarray, render_entity_id: np.ndarray) -> np.ndarray:
    """Hypersim's reference tonemapper -> uint8 RGB.

    Hypersim's official recipe: scale the image so that the 90th-percentile CCIR601
    brightness over pixels with geometry lands at 0.8 after gamma, then gamma-encode.
    Output matches the released *.tonemap.jpg previews.
    """
    rgb = np.asarray(rgb_hdr, dtype=np.float32)
    if rgb.ndim == 3 and rgb.shape[0] in (3, 4) and rgb.shape[-1] not in (3, 4):
        rgb = np.moveaxis(rgb, 0, -1)
    rgb = np.nan_to_num(rgb[..., :3], nan=0.0, posinf=0.0, neginf=0.0)

    valid = np.asarray(render_entity_id) != -1
    brightness = 0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]
    if not valid.any():
        scale = 1.0
    else:
        current = float(np.percentile(brightness[valid], TONEMAP_PERCENTILE))
        # Below eps the frame is essentially black; upstream sets scale 0 rather
        # than dividing by ~0 and blowing the image out to white.
        scale = 0.0 if current < TONEMAP_EPS else float(
            np.power(TONEMAP_BRIGHTNESS_TARGET, 1.0 / TONEMAP_GAMMA) / current)

    out = np.power(np.maximum(scale * rgb, 0.0), TONEMAP_GAMMA)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# panoptic packing                                                             #
# --------------------------------------------------------------------------- #
def pack_panoptic(semantic: np.ndarray, instance: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """(semantic, semantic_instance) -> uint16 ``class * DIVISOR + instance``.

    Instance ids are compacted to 1..N per (image, class) -- Hypersim numbers them
    per scene, which would overflow the divisor. Pixels with no instance keep
    instance 0 (stuff), and unlabeled pixels (raw class <= 0) pack to 0, which both
    readers treat as ignore since NYU40 raw ids start at 1.

    Also returns {raw_class: instance_count} for the caller's things.json tally.
    """
    sem = np.asarray(semantic).astype(np.int32)
    inst = np.asarray(instance).astype(np.int32)
    if sem.ndim == 3:
        sem = sem[..., 0]
    if inst.ndim == 3:
        inst = inst[..., 0]
    n_classes = len(taxonomy.HYPERSIM_CLASSES)

    labeled = (sem >= 1) & (sem <= n_classes)
    packed = np.zeros(sem.shape, dtype=np.uint16)
    packed[labeled] = sem[labeled].astype(np.uint16) * DIVISOR

    counts: dict[int, int] = {}
    has_instance = labeled & (inst >= 0)
    if has_instance.any():
        # Group by the (class, raw instance id) pair: an instance is only meaningful
        # within its class, and pairing also survives any id reused across classes.
        pairs = np.stack([sem[has_instance], inst[has_instance]], axis=1)
        unique_pairs = np.unique(pairs, axis=0)
        next_id: dict[int, int] = {}
        for raw_cls, raw_inst in unique_pairs.tolist():
            compact = next_id.get(raw_cls, 0) + 1
            if compact >= DIVISOR:
                raise ValueError(
                    f"class {raw_cls} has >= {DIVISOR} instances in one frame; "
                    f"the {DIVISOR} panoptic divisor cannot encode it")
            next_id[raw_cls] = compact
            sel = has_instance & (sem == raw_cls) & (inst == raw_inst)
            packed[sel] = np.uint16(raw_cls * DIVISOR + compact)
        counts = dict(next_id)
    return packed, counts


def convert_frame(frame: HypersimSegFrame, args: argparse.Namespace) -> dict[str, Any] | None:
    semantic = read_hdf5_array(frame.semantic_path)
    if semantic.ndim == 3:
        semantic = semantic[..., 0]
    n_classes = len(taxonomy.HYPERSIM_CLASSES)
    labeled_fraction = float(((semantic >= 1) & (semantic <= n_classes)).mean())
    if labeled_fraction < args.min_labeled_fraction:
        return None

    instance = read_hdf5_array(frame.instance_path)
    entity = read_hdf5_array(frame.entity_path)
    if entity.ndim == 3:
        entity = entity[..., 0]
    rgb = tonemap(read_hdf5_array(frame.rgb_path), entity)
    packed, per_class_instances = pack_panoptic(semantic, instance)

    if rgb.shape[:2] != packed.shape:
        raise ValueError(
            f"channel size mismatch for {frame.sample_id}: rgb={rgb.shape[:2]} panoptic={packed.shape}")

    image = Image.fromarray(rgb, mode="RGB")
    panoptic = Image.fromarray(packed)  # uint16 -> mode I;16, the encoding both readers expect
    if args.max_side > 0 and max(image.size) > args.max_side:
        scale = args.max_side / max(image.size)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, resample=IMAGE_RESAMPLING.LANCZOS)
        # NEAREST only: any interpolation would invent packed values that decode to
        # nonsense classes/instances.
        panoptic = panoptic.resize(size, resample=IMAGE_RESAMPLING.NEAREST)

    image_path = args.out_root / "images" / frame.split / f"{frame.sample_id}.jpg"
    panoptic_path = args.out_root / "panoptic" / frame.split / f"{frame.sample_id}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    panoptic_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path, quality=args.jpeg_quality)
    panoptic.save(panoptic_path)

    return {
        "sample_id": frame.sample_id,
        "split": frame.split,
        "scene": frame.scene,
        "camera": frame.camera,
        "frame": frame.frame,
        "labeled_fraction": round(labeled_fraction, 6),
        "classes": sorted(int(c) for c in np.unique(semantic) if 1 <= int(c) <= n_classes),
        "instances_per_class": {str(k): int(v) for k, v in sorted(per_class_instances.items())},
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-root", type=Path, required=True,
                   help="Extracted Hypersim root holding the ai_* scene folders "
                        "(e.g. <hypersim>/work/raw_extracted).")
    p.add_argument("--out-root", type=Path, required=True,
                   help="Destination for images/, panoptic/, id2label.json, things.json.")
    p.add_argument("--prepared-tar", type=Path, default=None, help="Also pack --out-root into this tar.")
    p.add_argument("--samples-per-scene", type=int, default=25)
    p.add_argument("--target-samples", type=int, default=None)
    p.add_argument("--all-frames", action="store_true", help="Convert every frame with all four channels.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.0,
                   help="Scene-disjoint val fraction (frames within a scene are near-duplicates, "
                        "so the split is by scene, matching the depth/normals prep).")
    p.add_argument("--min-labeled-fraction", type=float, default=0.10,
                   help="Skip frames whose semantic map is mostly unlabeled.")
    p.add_argument("--max-side", type=int, default=0, help="Downscale the long side (0 = native 1024x768).")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="Debug: cap converted frames.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--strict", action="store_true", help="Raise instead of skipping a failing frame.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.all_frames:
        args.target_samples = None
        args.samples_per_scene = None
    elif args.target_samples is not None and args.samples_per_scene is not None:
        args.samples_per_scene = None
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1)")
    args.raw_root = args.raw_root.expanduser().resolve()
    args.out_root = args.out_root.expanduser().resolve()
    clean_output(args.out_root, args.overwrite)

    frames = discover_frames(args.raw_root)
    if not frames:
        raise RuntimeError(
            f"No Hypersim frames with color + semantic + semantic_instance + render_entity_id "
            f"under: {args.raw_root}")
    sampled = sample_frames(
        frames,
        samples_per_scene=args.samples_per_scene,
        target_samples=args.target_samples,
        all_frames=args.all_frames,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )
    if args.limit:
        sampled = sampled[: args.limit]
    print(f"Converting {len(sampled)} frames from {len({f.scene for f in sampled})} scenes")

    def process(frame: HypersimSegFrame) -> dict[str, Any] | None:
        try:
            return convert_frame(frame, args)
        except Exception as exc:  # noqa: BLE001
            if args.strict:
                raise
            print(f"WARNING: skipped {frame.sample_id}: {exc}", flush=True)
            return None

    desc = f"Converting Hypersim segmentation ({args.num_workers} workers)"
    if args.num_workers > 1:
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            records = list(tqdm(ex.map(process, sampled), total=len(sampled), desc=desc))
    else:
        records = [process(f) for f in tqdm(sampled, desc=desc)]
    records = [r for r in records if r is not None]
    if not records:
        raise RuntimeError("No Hypersim frames converted; every frame failed or was filtered out.")

    # things.json records the classes that actually carried instances, intersected
    # with the taxonomy's thing set -- same override hook read_waymo's loader uses.
    observed: set[int] = set()
    for record in records:
        observed.update(int(c) for c in record["instances_per_class"])
    thing_ids = sorted(observed & set(taxonomy.HYPERSIM_THING_RAW_IDS))

    write_json(args.out_root / "id2label.json", {
        "classes": {str(i + 1): name for i, name in enumerate(taxonomy.HYPERSIM_RAW_LABEL_NAMES)},
        "panoptic_label_divisor": DIVISOR,
        "source": "hypersim work/semantic_label_descs.csv (NYU40)",
    })
    write_json(args.out_root / "things.json", {
        "thing_ids": thing_ids,
        "observed_with_instances": sorted(observed),
        "note": "thing_ids = taxonomy.HYPERSIM_THING_RAW_IDS restricted to classes "
                "that actually carried an instance id in this build",
    })
    with (args.out_root / "frames.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    by_split: dict[str, int] = {}
    for record in records:
        by_split[record["split"]] = by_split.get(record["split"], 0) + 1
    print(f"Wrote {len(records)} frames to {args.out_root} ({by_split}); "
          f"{len(thing_ids)} instance-eligible classes observed")
    if args.prepared_tar is not None:
        write_prepared_tar(args.out_root, args.prepared_tar)


if __name__ == "__main__":
    main()
