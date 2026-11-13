"""SA-Co Gold refinement, independent of search scripts and model loading.

Winning order: snap at generation resolution -> nearest resize -> decode ->
gapfill using the UNSNAPPED RGB -> GrabCut using the source photograph.
Refine each member before instance matching and consensus voting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from evaluation.instanceseg.decode import decode_instances

K3 = np.ones((3, 3), np.uint8)


def add_postprocess_arguments(parser):
    group = parser.add_argument_group("SA-Co Gold refinement (winning defaults)")
    for name, help_text in (
        ("snapping", "Snap generated RGB to the prompted palette plus background, before resizing."),
        ("gapfill", "Assign painted, unassigned pixels to a nearby decoded instance."),
        ("grabcut", "Refine boundaries with source-image OpenCV GrabCut."),
    ):
        group.add_argument("--" + name, action=argparse.BooleanOptionalAction,
                           default=True, help=help_text)
    group.add_argument("--gapfill_radius", type=float, default=3.0,
                       help="Maximum assignment distance in GT-resolution pixels (default 3).")
    group.add_argument("--gapfill_threshold", type=float, default=58.0,
                       help="Minimum unsnapped RGB distance from background considered painted (default 58).")
    group.add_argument("--refinement_order", choices=("gapfill_grabcut", "grabcut_gapfill"),
                       default="gapfill_grabcut")
    group.add_argument("--grabcut_iterations", type=int, default=2)
    group.add_argument("--grabcut_band_mode", choices=("fixed", "adaptive"), default="fixed")
    group.add_argument("--grabcut_radius", type=int, default=3,
                       help="Fixed core erosion radius in GT pixels (default 3).")
    group.add_argument("--grabcut_band_scale", type=float, default=0.05,
                       help="Adaptive radius = max(min_radius, floor(scale * sqrt(mask area))).")
    group.add_argument("--grabcut_min_radius", type=int, default=3)
    group.add_argument("--grabcut_ring_scale", type=float, default=1.0,
                       help="Outer dilation radius = max(1, round(core radius * ring_scale)).")
    group.add_argument("--grabcut_crop_margin", type=int, default=5,
                       help="Crop padding = this multiplier * max(core radius, outer radius).")
    group.add_argument("--grabcut_min_area_ratio", type=float, default=0.9,
                       help="Keep the input mask if refined area is below this fraction (default .9).")


def postprocess_protocol(args):
    """None denotes the legacy decoder, including old namespaces without switches.

    Only active parameters enter the protocol, so unused knobs do not prevent
    a legacy merge. Active refinements must match even for single-pass shards.
    """
    enabled = {key: bool(getattr(args, key, False)) for key in ("snapping", "gapfill", "grabcut")}
    if not any(enabled.values()):
        return None
    result = {"version": 1, **enabled,
              "order": getattr(args, "refinement_order", "gapfill_grabcut"),
              "snap_resolution": "generation_before_nearest_resize",
              "gapfill_rgb": "unsnapped_nearest_resize",
              "grabcut_photo": "RGB_to_BGR_after_bilinear_resize",
              "grabcut_seed": "sha256(subset:image_id:absolute_zero_based_member)[:4]_big_mod_2**31",
              "opencv_threads": 1}
    if enabled["gapfill"]:
        result["gapfill_parameters"] = {"radius": args.gapfill_radius,
                                        "painted_threshold": args.gapfill_threshold}
    if enabled["grabcut"]:
        band = ({"mode": "fixed", "radius": args.grabcut_radius}
                if args.grabcut_band_mode == "fixed" else
                {"mode": "adaptive", "scale": args.grabcut_band_scale,
                 "min_radius": args.grabcut_min_radius})
        result["grabcut_parameters"] = {
            "iterations": args.grabcut_iterations, "band": band,
            "ring_scale": args.grabcut_ring_scale, "crop_margin": args.grabcut_crop_margin,
            "min_area_ratio": args.grabcut_min_area_ratio,
        }
    return result


def validate_postprocess_arguments(args):
    if not math.isfinite(args.tau) or args.tau <= 0:
        raise ValueError("--tau must be finite and positive")
    if not 0 <= args.theta_size <= 1 or not 0 <= args.theta_erosion <= 1:
        raise ValueError("--theta_size and --theta_erosion must be in [0, 1]")
    if math.isnan(args.gamma) or args.gamma <= 0:
        raise ValueError("--gamma must be positive (infinity is allowed)")
    if any(not 0 <= c <= 255 for c in args.bg_color):
        raise ValueError("--bg_color must contain values in [0, 255]")
    for name in ("gapfill_radius", "gapfill_threshold"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name} must be finite and nonnegative")
    for name in ("grabcut_iterations", "grabcut_radius", "grabcut_min_radius", "grabcut_crop_margin"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be at least 1")
    for name in ("grabcut_ring_scale", "grabcut_band_scale"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name} must be finite and positive")
    if not 0 <= args.grabcut_min_area_ratio <= 1:
        raise ValueError("--grabcut_min_area_ratio must be in [0, 1]")
    if args.snapping and (args.prompt_template is not None or args.prompt_set in ("ordered", "legacy")):
        raise ValueError("Snapping needs an explicit prompted palette; use --no-snapping for this prompt mode")


def snap_rgb(raw, colors, bg_color=(0, 0, 0)):
    if colors is None or not len(colors):
        raise ValueError("Snapping requires the actual prompted palette")
    # Caller supplies the fixed draw pool order, independent of prompt permutation.
    palette = np.asarray([bg_color, *colors], dtype=np.float64)
    _, nearest = cKDTree(palette).query(np.asarray(raw, np.float64).reshape(-1, 3))
    return Image.fromarray(palette[nearest].reshape(np.asarray(raw).shape).astype(np.uint8))


def fill_gaps(raw_up, masks, radius, threshold, bg_color=(0, 0, 0)):
    if not masks:
        return masks
    labels = np.zeros(raw_up.shape[:2], np.int32)
    for index, mask in enumerate(masks, 1):
        labels[mask] = index
    painted = ((raw_up.astype(np.float64) - np.asarray(bg_color)) ** 2).sum(-1) >= threshold ** 2
    distance, (iy, ix) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    nearest = labels[iy, ix]
    add = painted & (labels == 0) & (distance <= radius)
    labels[add] = nearest[add]
    return [labels == index for index in range(1, len(masks) + 1)]


def refine_grabcut(photo_bgr, masks, config):
    """Same initialization, fallbacks and collapse guard as the parameter search."""
    height, width = photo_bgr.shape[:2]
    output = []
    for mask in masks:
        area = int(mask.sum())
        if not area:
            output.append(mask)
            continue
        band = config["band"]
        radius = (band["radius"] if band["mode"] == "fixed" else
                  max(band["min_radius"], int(band["scale"] * np.sqrt(area))))
        ring = max(1, int(round(radius * config["ring_scale"])))
        margin = config["crop_margin"] * max(radius, ring)
        ys, xs = np.where(mask)
        y0, y1 = max(0, ys.min() - margin), min(height, ys.max() + margin + 1)
        x0, x1 = max(0, xs.min() - margin), min(width, xs.max() + margin + 1)
        sub = mask[y0:y1, x0:x1].astype(np.uint8)
        gc = np.full(sub.shape, cv2.GC_BGD, np.uint8)
        gc[cv2.dilate(sub, K3, iterations=ring) > 0] = cv2.GC_PR_BGD
        gc[sub > 0] = cv2.GC_PR_FGD
        gc[cv2.erode(sub, K3, iterations=radius) > 0] = cv2.GC_FGD
        if not (gc == cv2.GC_FGD).any() or sub.size < 64:
            output.append(mask)
            continue
        try:
            cv2.grabCut(np.ascontiguousarray(photo_bgr[y0:y1, x0:x1]), gc, None,
                        np.zeros((1, 65)), np.zeros((1, 65)), config["iterations"], cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            output.append(mask)
            continue
        refined = np.zeros_like(mask)
        refined[y0:y1, x0:x1] = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
        output.append(refined if refined.sum() >= config["min_area_ratio"] * area else mask)
    return output


def decode_and_refine(raw, photo, size, args, *, colors, subset, image_id, member_index,
                      decode_fn=decode_instances):
    """Decode/refine a single member; size is (GT width, GT height)."""
    raw = raw.convert("RGB")
    protocol = postprocess_protocol(args)
    decoded_rgb = snap_rgb(raw, colors, args.bg_color) if protocol and protocol["snapping"] else raw
    decoded_up = np.asarray(decoded_rgb.resize(size, Image.Resampling.NEAREST))
    masks = decode_fn(decoded_up, bg_color=tuple(args.bg_color), tau=args.tau,
                      theta_size=args.theta_size, theta_erosion=args.theta_erosion, gamma=args.gamma)
    if protocol is None:
        return masks
    raw_up = np.asarray(raw.resize(size, Image.Resampling.NEAREST))
    for operation in protocol["order"].split("_"):
        if operation == "gapfill" and protocol["gapfill"]:
            masks = fill_gaps(raw_up, masks, args.gapfill_radius, args.gapfill_threshold, args.bg_color)
        elif operation == "grabcut" and protocol["grabcut"] and masks:
            cv2.setNumThreads(1)
            seed = int.from_bytes(hashlib.sha256(
                f"{subset}:{image_id}:{member_index}".encode()).digest()[:4], "big") % (2**31)
            cv2.setRNGSeed(seed)
            rgb = np.asarray(photo.convert("RGB").resize(size, Image.Resampling.BILINEAR))
            masks = refine_grabcut(rgb[:, :, ::-1], masks, protocol["grabcut_parameters"])
    return masks


def reproducibility_manifest(args, gt_folder, subsets, samples):
    """Record portable content hashes as well as machine-specific input paths."""
    import importlib.metadata
    import platform
    import sys
    from evaluation.instanceseg.eval_saco_gold import gt_paths_for

    def sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    root = Path(__file__).resolve().parents[2]
    names = ["evaluation/instanceseg/" + name for name in (
        "inference_flux2_klein_instanceseg.py", "postprocess.py", "decode.py",
        "repeat_analysis.py", "raw_recovery.py", "artifact_writer.py", "eval_saco_gold.py", "cgf1_eval.py")]
    names += ["common/instanceseg/" + name for name in (
        "prompts.py", "colors.py", "ordering.py", "instance_palette_named.json", "instance_palette.json")]
    names += ["evaluation/refseg/inference_flux2_klein_refseg.py",
              "evaluation/semseg/inference_partial_flux2_klein_img2img.py",
              "common/refseg/named_colors.py", "common/semseg/diversify.py"]
    versions = {}
    for package in ("numpy", "scipy", "Pillow", "pycocotools", "opencv-python", "opencv-python-headless",
                    "torch", "diffusers", "transformers", "accelerate", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        import cv2
        versions["cv2_runtime"] = cv2.__version__
    except ImportError:
        pass
    gate_file = getattr(args, "gate_file", None)
    source_root = getattr(args, "raw_rgb_root", None)
    source_config = source_root / "inference_config.json" if source_root else None
    selected = [[s.subset, s.image_id, s.np_text] for s in samples]
    return {
        "python": platform.python_version(), "packages": versions,
        "argv": sys.argv[1:], "source_sha256": {name: sha256(root / name) for name in names},
        "gate_sha256": sha256(gate_file) if gate_file else None,
        "gt_sha256": {Path(path).name: sha256(path) for subset in subsets
                      for path in gt_paths_for(subset, Path(gt_folder))},
        "selected_queries": len(selected),
        "selected_query_sha256": hashlib.sha256(json.dumps(selected, ensure_ascii=True).encode()).hexdigest(),
        "source_inference_config_sha256": sha256(source_config)
        if source_config and source_config.is_file() else None,
    }
