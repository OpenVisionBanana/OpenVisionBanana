#!/usr/bin/env python3
"""Build canonical instance-segmentation training pairs for Open Vision Banana.

Mirrors dataset_prep/semseg/dataset_prep.py, but colors per **instance** instead of
per class, so the model learns the SA-Co/Gold task: given (image, noun-phrase),
paint every instance of that noun-phrase a distinct color on black.

Per image we emit one (or more) randomized (prompt, target) variants. Each variant
picks **one** thing class present in the image (the noun phrase) and paints its
instances deterministically: canonical instance #i (raster order of mask anchors,
common/instanceseg/ordering.py) always gets color #i of the committed ordered
named palette (common/instanceseg/instance_palette_named.json, 36 everyday-named
colors; the 64-color grid palette instance_palette.json takes over only when an
image has more instances than that). The prompt spells the assignment out as an
explicit legend ("The 1st car is blue, the 2nd is green, ..."), written per row in
one notation style -- color names or raw RGB values -- which changes only the
wording, never the painted pixels. The target is therefore a pure function of
(image, noun-phrase); the per-image RNG only picks which variant to emit (class,
negative-or-not, template, notation). A configurable fraction of variants are
**negatives**: the prompt names a class that has no instance in the image and the
target is all-black. Negatives carry no color legend.

Prompts come from common/instanceseg/prompts.py (build_instance_prompt), the same
builder the SA-Co/Gold evaluator uses.

--draw-mode swaps the default exact-count legend scheme for one of the fixed-K
draw schemes (explicit, unlisted, scan-order and notation variants) defined in
common/instanceseg/prompts.py DRAW_MODES.

Sources (instance encodings decoded in instance_taxonomy.py):
  ade20k_inst  ADE20K instance release  (R=class 1..100, G=instance)
  mapillary    Mapillary Vistas v2.0    (uint16 class*256+instance, config_v2.0.json)
  voc_inst     Pascal VOC 2012          (SegmentationObject ∩ SegmentationClass)
  muses        MUSES panoptic            (COCO panoptic RGB ids + segments_info)
  acdc         ACDC panoptic             (same COCO panoptic encoding as MUSES;
                                          Cityscapes label ids, 8 thing classes.
                                          Crowd/group regions carry iscrowd=1 and
                                          are skipped, so they stay unpainted)
  synscapes    Synscapes                  (RGB instance ids + per-frame metadata;
                                          Cityscapes raw class ids, 8 things)
  idd          India Driving Dataset       (JSON polygons; raw IDD thing labels)
  nuimages     NuImages                    (train/val key-camera foreground
                                          COCO-RLE masks; 10-second log sampling)
  paco_lvis    PACO-LVIS DatasetNinja       (Supervisely polygon/bitmap masks;
                                          all object and object-part classes)
  coconut_xlarge COCONut-XL                 (COCO panoptic RGB IDs + final merged manifest;
                                          80 thing categories, crowd regions skipped)
  wilddash2    WildDash 2                  (COCO panoptic RGB IDs + official
                                          random train/validation split)
  goose        GOOSE 2D                  (labelids + instanceids)
  vipseg       VIPSeg native panomasks   (semantic_id*100+instance_id for things)
  ovis         OVIS video instance seg   (YouTube-VOS compressed RLE per frame,
                                          sampled to 1 fps by default)
  waymo        Waymo camera panoptic     (uint16 raw_class*1000+instance; the same
                                          tree also feeds the semseg converter)
  hypersim     Hypersim indoor scenes    (NYU40; the same packed uint16 encoding,
                                          materialized from the release's HDF5 by
                                          dataset_prep/hypersim/prepare_hypersim_segmentation.py)

Output is the standard pool layout: metadata.jsonl + conditioning_images/ +
target_images/, with rows {conditioning_image, image, text, task, source_dataset,
id, split, metadata}.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - convenience for minimal prep envs
    def tqdm(iterable, **_kwargs):
        return iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import coco_eval_holdouts, coconut_xlarge  # noqa: E402
from common.instanceseg import prompts as iseg_prompts  # noqa: E402
from common.instanceseg.colors import load_instance_palette, load_named_instance_palette  # noqa: E402
from common.instanceseg.ordering import canonical_instance_order, select_instances  # noqa: E402
from common.semseg import taxonomy  # noqa: E402
from dataset_prep.instanceseg import instance_taxonomy as it  # noqa: E402

BACKGROUND_COLOR = (0, 0, 0)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
# Logical split -> source-specific directory / list name.
ADE_SPLIT = {"train": "training", "val": "validation"}
MAPI_SPLIT = {"train": "training", "val": "validation"}


# --------------------------------------------------------------------------- #
# small helpers (kept local to avoid importing the semseg converter)          #
# --------------------------------------------------------------------------- #
def safe_name(*parts: str) -> str:
    joined = "__".join(str(p) for p in parts if p)
    return joined.replace("/", "_").replace("\\", "_").replace(" ", "_")


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".png":
        shutil.copy2(src, dst)
    else:
        with Image.open(src) as im:
            im.convert("RGB").save(dst)


def normalize_np(name: str) -> str:
    """Class name -> noun phrase used in the prompt (lowercased, spaces tidy)."""
    return " ".join(name.strip().lower().replace("_", " ").split())


# --------------------------------------------------------------------------- #
# datasets                                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Pair:
    dataset: str
    split: str
    stem: str
    image_path: Path
    ann_primary: Path                 # ADE ann / Mapillary instances / VOC SegObject
    ann_secondary: Optional[Path] = None  # VOC SegmentationClass
    metadata_path: Optional[Path] = None  # Synscapes per-frame metadata JSON
    segments_info: tuple[dict, ...] = ()  # COCO-panoptic segment metadata
    annotations: tuple[dict, ...] = ()    # NuImages surface/object RLE metadata
    frame_index: Optional[int] = None      # OVIS frame index in video
    frame_height: Optional[int] = None     # OVIS video/image height
    frame_width: Optional[int] = None      # OVIS video/image width
    frame_annotations: tuple[dict[str, Any], ...] = ()  # OVIS per-video anns


@dataclass
class DatasetInfo:
    names: Dict[int, str]             # class id -> raw name
    instanceable: Set[int]            # class ids eligible as a noun phrase / negative
    raw_to_index: Dict[object, int] = field(default_factory=dict)  # source raw id/token -> canonical id


def build_ade_pairs(root: Path, splits: Sequence[str]) -> List[Pair]:
    pairs: List[Pair] = []
    for sp in splits:
        d = ADE_SPLIT.get(sp, sp)
        ann_dir, img_dir = root / "annotations_instance" / d, root / "images" / d
        if not ann_dir.is_dir():
            continue
        for ann in sorted(ann_dir.glob("*.png")):
            img = img_dir / f"{ann.stem}.jpg"
            if img.exists():
                pairs.append(Pair("ade20k_inst", sp, ann.stem, img, ann))
    return pairs


def build_mapillary_pairs(root: Path, splits: Sequence[str]) -> List[Pair]:
    pairs: List[Pair] = []
    for sp in splits:
        d = MAPI_SPLIT.get(sp, sp)
        ann_dir, img_dir = root / d / "v2.0" / "instances", root / d / "images"
        if not ann_dir.is_dir():
            continue
        for ann in sorted(ann_dir.glob("*.png")):
            img = img_dir / f"{ann.stem}.jpg"
            if img.exists():
                pairs.append(Pair("mapillary", sp, ann.stem, img, ann))
    return pairs


def build_synscapes_pairs(root: Path, splits: Sequence[str]) -> List[Pair]:
    resolved = taxonomy.resolve_synscapes_root(root)
    selected = taxonomy.normalize_synscapes_splits(splits)
    instance_dir = resolved / "img" / "instance"
    pairs: List[Pair] = []
    if not instance_dir.is_dir():
        return pairs
    for index, instance_path in taxonomy.synscapes_numbered_pngs(instance_dir):
        split = taxonomy.synscapes_split_for_index(index)
        if split not in selected:
            continue
        image_path = resolved / "img" / "rgb" / f"{index}.png"
        class_path = resolved / "img" / "class" / f"{index}.png"
        metadata_path = resolved / "meta" / f"{index}.json"
        if image_path.exists() and class_path.exists():
            pairs.append(Pair(
                "synscapes", split, str(index), image_path, instance_path,
                ann_secondary=class_path,
                metadata_path=metadata_path if metadata_path.exists() else None,
            ))
    return pairs


def build_idd_pairs(root: Path, splits: Sequence[str]) -> List[Pair]:
    """Discover IDD polygon annotations across either/both parts of idd_1_2.tar."""
    pairs: List[Pair] = []
    selected = taxonomy.normalize_idd_splits(splits)
    for source_root in taxonomy.resolve_idd_roots(root):
        part = taxonomy.idd_source_key(source_root)
        for split in sorted(selected):
            annotation_root = source_root / "gtFine" / split
            for annotation_path in sorted(annotation_root.glob("*/*_gtFine_polygons.json")):
                drive = annotation_path.parent.name
                image_stem = annotation_path.name.removesuffix("_gtFine_polygons.json")
                image_path = (
                    source_root / "leftImg8bit" / split / drive /
                    f"{image_stem}_leftImg8bit.png"
                )
                if image_path.exists():
                    pairs.append(Pair(
                        "idd", split, f"{part}__{drive}__{image_stem}", image_path,
                        annotation_path,
                    ))
    return pairs


def build_wilddash2_pairs(root: Path, splits: Sequence[str]) -> List[Pair]:
    """Build WildDash 2 pairs from its official random train/validation split."""
    resolved = taxonomy.resolve_wilddash2_root(root)
    selected = taxonomy.normalize_wilddash2_splits(splits)
    index = json.loads((resolved / "panoptic.json").read_text(encoding="utf-8"))
    images = {Path(str(image["file_name"])).stem: image for image in index.get("images", ())}
    annotations = {str(ann.get("image_id")): ann for ann in index.get("annotations", ())}
    split_files = {"train": "training.txt", "val": "validation.txt"}
    pairs: List[Pair] = []
    for split in sorted(selected):
        stems = (resolved / "random_split" / split_files[split]).read_text(encoding="utf-8").split()
        for stem in stems:
            image = images.get(stem)
            if image is None:
                continue
            annotation = annotations.get(str(image.get("id")))
            if annotation is None:
                continue
            image_path = resolved / "anonymized" / str(image["file_name"])
            panoptic_path = resolved / "panoptic" / str(annotation["file_name"])
            if image_path.exists() and panoptic_path.exists():
                pairs.append(Pair(
                    "wilddash2", split, stem, image_path, panoptic_path,
                    segments_info=tuple(annotation.get("segments_info", ())),
                ))
    return pairs



def build_coconut_xlarge_pairs(
    root: Path,
    holdouts: dict[str, frozenset[str]] | None = None,
    audit: coco_eval_holdouts.CocoEvaluationHoldoutAudit | None = None,
    groups: str | Iterable[str] | None = None,
) -> List[Pair]:
    """Build selected materialized COCONut-XL panoptic pairs.

    Its final merged manifest is a single training set, so prepared rows use
    the logical ``train`` split. ``groups`` selects COCO, Objects365, or both.
    """
    selected_groups = coconut_xlarge.normalize_coconut_xlarge_groups(groups)
    resolved = taxonomy.resolve_coconut_xlarge_root(root)
    index = json.loads(
        taxonomy.coconut_xlarge_annotation_path(resolved).read_text(encoding="utf-8"))
    images = {str(image.get("id")): image for image in index.get("images", ())}
    pairs: List[Pair] = []
    excluded: dict[str, int] = {}
    excluded_images = 0
    excluded_groups: dict[str, int] = {}
    for annotation in index.get("annotations", ()):
        image = images.get(str(annotation.get("image_id")))
        if image is None:
            continue
        group = coconut_xlarge.coconut_xlarge_source_group(image)
        if group not in selected_groups:
            excluded_groups[group] = excluded_groups.get(group, 0) + 1
            continue
        image_name = str(image.get("file_name") or "")
        reasons = coco_eval_holdouts.holdout_reasons(image_name, holdouts or {})
        if reasons:
            excluded_images += 1
            for reason in reasons:
                excluded[reason] = excluded.get(reason, 0) + 1
            if audit is not None:
                audit.record(
                    image_name, reasons, source="coconut_xlarge_instanceseg",
                    reference=str(annotation.get("image_id") or image.get("id") or ""),
                )
            continue
        image_path = resolved / "train2017" / str(image.get("file_name") or "")
        panoptic_path = resolved / "panoptic_train2017" / str(annotation.get("file_name") or "")
        if not image_path.is_file() or not panoptic_path.is_file():
            continue
        stem = str(image.get("object365_name") or Path(str(image["file_name"])).stem)
        pairs.append(Pair(
            "coconut_xlarge", "train", stem, image_path, panoptic_path,
            segments_info=tuple(annotation.get("segments_info", ())),
        ))
    if excluded_groups:
        print(f"Excluded COCONut-XL source groups outside {sorted(selected_groups)}: {dict(sorted(excluded_groups.items()))}")
    if excluded:
        print(f"Excluded {excluded_images} COCONut-XL evaluation images: {dict(sorted(excluded.items()))}")
    return pairs

def build_nuimages_pairs(
    root: Path,
    splits: Sequence[str],
    sample_interval_seconds: float = taxonomy.NUIMAGES_DEFAULT_SAMPLE_INTERVAL_SECONDS,
) -> List[Pair]:
    """Build NuImages foreground-instance pairs using shared log sampling."""
    resolved = taxonomy.resolve_nuimages_root(root)
    frames, _skipped = taxonomy.discover_nuimages_frames(
        resolved, splits, sample_interval_seconds=sample_interval_seconds)
    return [
        Pair(
            "nuimages", frame.split, frame.token, frame.image_path,
            resolved / f"v1.0-{frame.split}" / "object_ann.json",
            annotations=frame.annotations,
        )
        for frame in frames
    ]



def build_paco_lvis_pairs(
    root: Path,
    splits: Sequence[str],
    holdouts: dict[str, frozenset[str]] | None = None,
    audit: coco_eval_holdouts.CocoEvaluationHoldoutAudit | None = None,
) -> List[Pair]:
    """Build PACO-LVIS pairs from its native Supervisely export."""
    resolved = it.resolve_paco_lvis_root(root)
    selected = it.normalize_paco_lvis_splits(splits)
    pairs: List[Pair] = []
    excluded: dict[str, int] = {}
    excluded_images = 0
    for split in selected:
        split_root = resolved / f"{split}2017"
        annotation_dir = split_root / "ann"
        image_dir = split_root / "img"
        if not annotation_dir.is_dir() or not image_dir.is_dir():
            continue
        for annotation_path in sorted(annotation_dir.glob("*.jpg.json")):
            image_name = annotation_path.name.removesuffix(".json")
            reasons = coco_eval_holdouts.holdout_reasons(image_name, holdouts or {})
            if reasons:
                excluded_images += 1
                for reason in reasons:
                    excluded[reason] = excluded.get(reason, 0) + 1
                if audit is not None:
                    audit.record(
                        image_name, reasons, source="paco_lvis_instanceseg",
                        reference=str(annotation_path.relative_to(resolved)),
                    )
                continue
            image_path = image_dir / image_name
            if image_path.is_file():
                pairs.append(Pair(
                    "paco_lvis", split, image_name.removesuffix(".jpg"), image_path,
                    annotation_path,
                ))
    if excluded:
        print(f"Excluded {excluded_images} PACO-LVIS evaluation images: {dict(sorted(excluded.items()))}")
    return pairs

def build_voc_pairs(root: Path, splits: Sequence[str]) -> List[Pair]:
    pairs: List[Pair] = []
    img_dir = root / "JPEGImages"
    obj_dir, cls_dir = root / "SegmentationObject", root / "SegmentationClass"
    for sp in splits:
        listf = root / "ImageSets" / "Segmentation" / f"{sp}.txt"
        if not listf.exists():
            continue
        for stem in listf.read_text().split():
            img, obj, cls = img_dir / f"{stem}.jpg", obj_dir / f"{stem}.png", cls_dir / f"{stem}.png"
            if img.exists() and obj.exists() and cls.exists():
                pairs.append(Pair("voc_inst", sp, stem, img, obj, cls))
    return pairs


def build_muses_pairs(root: Path, splits: Sequence[str]) -> List[Pair]:
    pairs: List[Pair] = []
    for sp in splits:
        index_path = root / "gt_panoptic" / f"{sp}.json"
        if not index_path.exists():
            continue
        data = json.loads(index_path.read_text(encoding="utf-8"))
        images = {str(image["id"]): image for image in data.get("images", [])}
        for ann in data.get("annotations", []):
            image_id = str(ann.get("image_id", ""))
            image = images.get(image_id)
            if image is None:
                continue
            img_path = root / "frame_camera" / image["file_name"]
            pan_path = root / "gt_panoptic" / ann["file_name"]
            if img_path.exists() and pan_path.exists():
                pairs.append(Pair(
                    "muses",
                    sp,
                    image_id,
                    img_path,
                    pan_path,
                    segments_info=tuple(ann.get("segments_info", ())),
                ))
    return pairs


def build_acdc_pairs(root: Path, splits: Sequence[str],
                     conditions: Optional[Sequence[str]] = None) -> List[Pair]:
    """ACDC panoptic -- the same COCO-panoptic encoding as MUSES, different paths.

    The index is gt_panoptic/<split>_gt_panoptic.json (the merged one; the
    per-condition <cond>/<cond>_<split>_gt_panoptic.json files are exactly its
    partition, so reading the merged file avoids stitching four indices). Both
    file_name fields are condition-relative -- images "fog/train/GOPR0475/
    <stem>_rgb_anon.png" under rgb_anon/, panoptic PNGs "<...>_gt_panoptic.png"
    under gt_panoptic/ -- so the optional condition filter reads the leading path
    component. image["id"] is the frame stem and is unique across all conditions
    and both splits, so it is safe as the output filename component.
    """
    keep = {c.strip().lower() for c in conditions if c.strip()} if conditions else None
    pairs: List[Pair] = []
    for sp in splits:
        index_path = root / "gt_panoptic" / f"{sp}_gt_panoptic.json"
        if not index_path.exists():
            continue
        data = json.loads(index_path.read_text(encoding="utf-8"))
        images = {str(image["id"]): image for image in data.get("images", [])}
        for ann in data.get("annotations", []):
            image = images.get(str(ann.get("image_id", "")))
            if image is None:
                continue
            rel = str(image["file_name"])
            if keep is not None and rel.split("/")[0].lower() not in keep:
                continue
            img_path = root / "rgb_anon" / rel
            pan_path = root / "gt_panoptic" / ann["file_name"]
            if img_path.exists() and pan_path.exists():
                pairs.append(Pair(
                    "acdc",
                    sp,
                    str(image["id"]),
                    img_path,
                    pan_path,
                    segments_info=tuple(ann.get("segments_info", ())),
                ))
    return pairs


def _goose_key(path: Path, suffix: str) -> str:
    stem = path.name.replace(suffix, "")
    while "__" in stem:
        stem = stem.replace("__", "_")
    return stem


def build_goose_pairs(root: Path, splits: Sequence[str]) -> List[Pair]:
    pairs: List[Pair] = []
    for sp in splits:
        img_root, label_root = root / "images" / sp, root / "labels" / sp
        if not img_root.is_dir() or not label_root.is_dir():
            continue
        images_by_key: Dict[tuple[str, str], Path] = {}
        for img in sorted(img_root.glob("*/*_windshield_vis.png")):
            images_by_key[(img.parent.name, _goose_key(img, "_windshield_vis.png"))] = img
        for label in sorted(label_root.glob("*/*_labelids.png")):
            group = label.parent.name
            key = _goose_key(label, "_labelids.png")
            img = images_by_key.get((group, key))
            inst = label.with_name(label.name.replace("_labelids.png", "_instanceids.png"))
            if img is not None and inst.exists():
                pairs.append(Pair("goose", sp, f"{group}_{key}", img, label, inst))
    return pairs


def _vipseg_split_video_ids(root: Path, split: str) -> List[str]:
    candidates = [root / f"{split}.txt", root / "VIPSeg_720P" / f"{split}.txt"]
    split_path = next((path for path in candidates if path.exists()), None)
    if split_path is None:
        return []
    return [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_vipseg_pairs(root: Path, splits: Sequence[str]) -> List[Pair]:
    pairs: List[Pair] = []
    img_root, mask_root = root / "imgs", root / "panomasks"
    if not img_root.is_dir() or not mask_root.is_dir():
        return pairs
    for sp in splits:
        for video in _vipseg_split_video_ids(root, sp):
            video_img_dir = img_root / video
            video_mask_dir = mask_root / video
            if not video_img_dir.is_dir() or not video_mask_dir.is_dir():
                continue
            for ann in sorted(video_mask_dir.glob("*.png")):
                img = None
                for suffix in IMAGE_SUFFIXES:
                    candidate = video_img_dir / f"{ann.stem}{suffix}"
                    if candidate.exists():
                        img = candidate
                        break
                if img is not None:
                    pairs.append(Pair("vipseg", sp, f"{video}_{ann.stem}", img, ann))
    return pairs


def _ovis_split_files(root: Path, split: str) -> tuple[Path, Path]:
    normalized = "valid" if split in {"val", "validation", "valid"} else split
    ann_name = "annotations_valid.json" if normalized == "valid" else f"annotations_{normalized}.json"
    if normalized == "valid":
        image_dir = root / "valid" if (root / "valid").is_dir() or not (root / "val").is_dir() else root / "val"
    else:
        image_dir = root / normalized
    return root / ann_name, image_dir


def _ovis_sample_step(source_fps: float, sample_fps: float) -> int:
    if source_fps <= 0 or sample_fps <= 0:
        raise ValueError("--ovis-source-fps and --ovis-fps must be positive.")
    return max(1, int(round(source_fps / sample_fps)))


def build_ovis_pairs(root: Path, splits: Sequence[str], sample_fps: float, source_fps: float) -> List[Pair]:
    pairs: List[Pair] = []
    step = _ovis_sample_step(source_fps, sample_fps)
    for sp in splits:
        ann_path, img_dir = _ovis_split_files(root, sp)
        if not ann_path.exists() or not img_dir.is_dir():
            continue
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        anns_by_video: Dict[int, List[dict[str, Any]]] = {}
        # OVIS withholds masks on non-train splits: annotations_valid.json carries
        # videos but a null "annotations" (so `.get(..., [])` returns None, not []).
        # Treat null as empty -> that split just yields no instance-seg pairs.
        for ann in data.get("annotations") or []:
            anns_by_video.setdefault(int(ann["video_id"]), []).append(ann)
        logical_split = "val" if sp in {"validation", "valid"} else sp
        for video in data.get("videos") or []:
            video_id = int(video["id"])
            file_names = list(video.get("file_names", []))
            height, width = int(video["height"]), int(video["width"])
            video_anns = tuple(anns_by_video.get(video_id, ()))
            for frame_index in range(0, len(file_names), step):
                frame_anns = tuple(
                    ann for ann in video_anns
                    if frame_index < len(ann.get("segmentations") or [])
                    and ann["segmentations"][frame_index] is not None
                )
                if not frame_anns:
                    continue
                image_path = img_dir / file_names[frame_index]
                if image_path.exists():
                    stem = f"video{video_id:06d}_frame{frame_index:06d}"
                    pairs.append(Pair(
                        "ovis",
                        logical_split,
                        stem,
                        image_path,
                        ann_path,
                        frame_index=frame_index,
                        frame_height=height,
                        frame_width=width,
                        frame_annotations=frame_anns,
                    ))
    return pairs


def build_waymo_pairs(root: Path, splits: Sequence[str]) -> List[Pair]:
    """Waymo camera panoptic labels, read for their instance half.

    Layout from download_datasets_waymo.sh (shared with the semseg converter):
      images/{train,val}/<stem>.jpg + panoptic/{train,val}/<stem>.png
    """
    pairs: List[Pair] = []
    alias = {"training": "train", "validation": "val"}
    for sp in splits:
        d = alias.get(sp, sp)
        ann_dir, img_dir = root / "panoptic" / d, root / "images" / d
        if not ann_dir.is_dir() or not img_dir.is_dir():
            continue
        for ann in sorted(ann_dir.glob("*.png")):
            img = img_dir / f"{ann.stem}.jpg"
            if img.exists():
                pairs.append(Pair("waymo", d, ann.stem, img, ann))
    return pairs


def build_hypersim_pairs(root: Path, splits: Sequence[str]) -> List[Pair]:
    """Hypersim packed panoptic labels, read for their instance half.

    Layout from prepare_hypersim_segmentation.py (shared with the semseg converter):
      images/{train,val}/<stem>.jpg + panoptic/{train,val}/<stem>.png
    """
    pairs: List[Pair] = []
    alias = {"training": "train", "validation": "val"}
    for sp in splits:
        d = alias.get(sp, sp)
        ann_dir, img_dir = root / "panoptic" / d, root / "images" / d
        if not ann_dir.is_dir() or not img_dir.is_dir():
            continue
        for ann in sorted(ann_dir.glob("*.png")):
            img = img_dir / f"{ann.stem}.jpg"
            if img.exists():
                pairs.append(Pair("hypersim", d, ann.stem, img, ann))
    return pairs


def build_dataset_info(args: argparse.Namespace) -> Dict[str, DatasetInfo]:
    info: Dict[str, DatasetInfo] = {}
    if args.ade_root:
        names = it.ade_names(Path(args.ade_root) / "instanceInfo100_train.txt")
        info["ade20k_inst"] = DatasetInfo(names, set(names))
    if args.mapillary_root:
        names, inst_ok = it.mapillary_classes(Path(args.mapillary_root) / "config_v2.0.json")
        info["mapillary"] = DatasetInfo(names, inst_ok)
    if args.synscapes_root:
        names, inst_ok = it.synscapes_classes()
        info["synscapes"] = DatasetInfo(names, inst_ok)
    if args.idd_root:
        names, inst_ok = it.idd_classes()
        info["idd"] = DatasetInfo(names, inst_ok)
    if args.wilddash2_root:
        names, inst_ok = it.wilddash2_categories(Path(args.wilddash2_root))
        info["wilddash2"] = DatasetInfo(names, inst_ok)
    if args.nuimages_root:
        names, inst_ok, token_to_index = it.nuimages_categories(Path(args.nuimages_root))
        info["nuimages"] = DatasetInfo(names, inst_ok, token_to_index)
    if args.coconut_xlarge_root:
        names, inst_ok = it.coconut_xlarge_categories(Path(args.coconut_xlarge_root))
        info["coconut_xlarge"] = DatasetInfo(names, inst_ok)
    if args.paco_lvis_root:
        names, inst_ok = it.paco_lvis_classes(Path(args.paco_lvis_root))
        info["paco_lvis"] = DatasetInfo(names, inst_ok)
    if args.voc_root:
        names = it.voc_names()
        info["voc_inst"] = DatasetInfo(names, set(names))
    if args.muses_root:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        names, inst_ok = it.muses_categories(Path(args.muses_root), splits)
        info["muses"] = DatasetInfo(names, inst_ok)
    if args.acdc_root:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        names, inst_ok = it.acdc_categories(Path(args.acdc_root), splits)
        info["acdc"] = DatasetInfo(names, inst_ok)
    if args.goose_root:
        names, inst_ok, raw_to_index = it.goose_names(Path(args.goose_root))
        info["goose"] = DatasetInfo(names, inst_ok, raw_to_index)
    if args.vipseg_root:
        names, inst_ok = it.vipseg_categories(Path(args.vipseg_root))
        info["vipseg"] = DatasetInfo(names, inst_ok)
    if args.ovis_root:
        root = Path(args.ovis_root)
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        names, inst_ok = it.ovis_categories(_ovis_split_files(root, sp)[0] for sp in splits)
        info["ovis"] = DatasetInfo(names, inst_ok)
    if args.waymo_root:
        names, inst_ok = it.waymo_classes(Path(args.waymo_root))
        info["waymo"] = DatasetInfo(names, inst_ok)
    if args.hypersim_root:
        names, inst_ok = it.hypersim_classes(Path(args.hypersim_root))
        info["hypersim"] = DatasetInfo(names, inst_ok)
    return info


def read_masks(pair: Pair, info: DatasetInfo) -> it.Masks:
    if pair.dataset == "ade20k_inst":
        return it.read_ade(pair.ann_primary)
    if pair.dataset == "mapillary":
        return it.read_mapillary(pair.ann_primary, info.instanceable)
    if pair.dataset == "synscapes":
        if pair.ann_secondary is None:
            raise ValueError(f"Synscapes pair is missing its class mask: {pair.stem}")
        return it.read_synscapes(
            pair.ann_primary, pair.ann_secondary, pair.metadata_path, info.instanceable)
    if pair.dataset == "idd":
        return it.read_idd(pair.ann_primary, info.instanceable)
    if pair.dataset == "nuimages":
        return it.read_nuimages(pair.annotations, info.raw_to_index, info.instanceable)
    if pair.dataset == "paco_lvis":
        return it.read_paco_lvis(pair.ann_primary, info.instanceable)
    if pair.dataset == "voc_inst":
        return it.read_voc(pair.ann_primary, pair.ann_secondary)
    if pair.dataset in ("muses", "acdc", "wilddash2", "coconut_xlarge"):
        return it.read_coco_panoptic(pair.ann_primary, pair.segments_info, info.instanceable)
    if pair.dataset == "goose":
        return it.read_goose(pair.ann_primary, pair.ann_secondary, info.raw_to_index)
    if pair.dataset == "vipseg":
        return it.read_vipseg(pair.ann_primary, info.instanceable)
    if pair.dataset == "waymo":
        return it.read_waymo(pair.ann_primary, info.instanceable)
    if pair.dataset == "hypersim":
        return it.read_hypersim(pair.ann_primary, info.instanceable)
    if pair.dataset == "ovis":
        if pair.frame_index is None or pair.frame_height is None or pair.frame_width is None:
            raise ValueError(f"OVIS pair is missing frame metadata: {pair.stem}")
        return it.read_ovis_frame(
            pair.frame_annotations,
            pair.frame_index,
            pair.frame_height,
            pair.frame_width,
            info.instanceable,
        )
    raise ValueError(f"unknown dataset {pair.dataset}")


# --------------------------------------------------------------------------- #
# per-image variant writing                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    variants_per_image: int = 1
    negative_ratio: float = 0.2       # fraction of variants that are absent-class negatives
    min_instance_area: float = 0.0    # drop instances smaller than this fraction of the image
    max_instances: Optional[int] = None  # cap painted instances per class (keep largest, deterministic)
    legend_slack_prob: float = 0.0    # prob. of listing a few extra legend colors beyond the count
    draw_mode: str = "legend"         # "legend" (exact-count) or a DRAW_MODES key
    seed: int = 42
    datasets: Dict[str, DatasetInfo] = field(default_factory=dict)


def _filter_small(masks: it.Masks, min_area: float, h: int, w: int) -> it.Masks:
    if min_area <= 0:
        return masks
    thr = min_area * h * w
    out: it.Masks = {}
    for cls, ms in masks.items():
        kept = [m for m in ms if m.sum() >= thr]
        if kept:
            out[cls] = kept
    return out


def write_image_variants(pair: Pair, out_input: Path, out_target: Path, cfg: Config) -> List[dict]:
    info = cfg.datasets[pair.dataset]
    basename = safe_name(pair.dataset, pair.split, pair.stem)
    input_name = f"{basename}_input.png"

    masks = read_masks(pair, info)
    if not masks:
        return []
    h, w = next(iter(next(iter(masks.values())))).shape
    masks = _filter_small(masks, cfg.min_instance_area, h, w)
    present = sorted(masks.keys())
    if not present:
        return []

    # Conditioning images are independent of the color/prompt knobs, so an
    # existing file is reused as-is (rebuilds may hardlink conditioning_images/
    # from a sibling pool; interrupted runs are resumable).
    if not (out_input / input_name).exists():
        copy_image(pair.image_path, out_input / input_name)
    rng = random.Random(f"{cfg.seed}:{basename}")
    present_set = set(present)
    absent = sorted(c for c in info.instanceable if c not in present_set)
    rows: List[dict] = []

    mode_cfg = iseg_prompts.DRAW_MODES.get(cfg.draw_mode)  # None -> exact-count legend
    draw_k = mode_cfg["k"] if mode_cfg else None
    scan = mode_cfg["scan"] if mode_cfg else "tb"  # instance-numbering scan direction

    for v in range(cfg.variants_per_image):
        is_negative = bool(absent) and rng.random() < cfg.negative_ratio
        rgb = np.zeros((h, w, 3), dtype=np.uint8)  # black background

        if is_negative:
            cls = rng.choice(absent)
            inst_masks, order, n_inst = [], [], 0
        else:
            cls = rng.choice(present)
            inst_masks = list(masks[cls])
            # Deterministic target: cap to the largest instances (never more than
            # the color pool has colors -- the mode's K, else the fallback grid
            # palette), then canonical scan rank -> color rank.
            palette_k = draw_k if draw_k is not None else len(load_instance_palette())
            cap = min(cfg.max_instances or palette_k, palette_k)
            inst_masks = select_instances(inst_masks, cap)
            order = canonical_instance_order(inst_masks, scan)
            n_inst = len(inst_masks)

        np_text = normalize_np(info.names[cls])
        permutation = None
        if mode_cfg is not None:
            # Fixed-K draw schemes: constant-length prompts, mode-fixed notation.
            # The legend is emitted for negatives too -- eval sends the constant
            # legend for every query (count unknown), so training must cover
            # "legend + absent class -> all-black".
            if mode_cfg["randomize"]:
                permutation = rng.sample(range(draw_k), draw_k)
            notation = rng.choice(("name", "hex")) if mode_cfg["style"] == "mixed" else None
            template = rng.choice(iseg_prompts.BASE_TEMPLATES)
            prompt, legend_colors = iseg_prompts.build_draw_prompt(
                cfg.draw_mode, np_text, template, permutation, notation)
            prompt_contract = mode_cfg["prompt_contract"]
            # Unlisted modes still paint from the full K-color pool, but the
            # prompt renders no legend entries and no color notation; record both.
            if prompt_contract == "explicit":
                style, legend_k = notation or mode_cfg["style"], draw_k
            else:
                style, legend_k = None, 0
            scheme = cfg.draw_mode
        else:
            # Legend length = instance count (0 -> no legend); optional slack lists a
            # few extra convention entries so "listed color goes unused" is also seen.
            name_cap = len(load_named_instance_palette())
            legend_k = n_inst
            if 0 < n_inst <= name_cap and rng.random() < cfg.legend_slack_prob:
                legend_k = min(n_inst + rng.randint(1, 4), name_cap)
            style = "rgb" if legend_k > name_cap else rng.choice(list(iseg_prompts.NOTATION_STYLES))
            template = rng.choice(iseg_prompts.BASE_TEMPLATES)
            # open_ended=False: training legends are exact-count (the continuation
            # clause appears only on truncated rows); the open-ended clause is the
            # evaluator's fixed-length-legend behavior.
            prompt, legend_colors = iseg_prompts.build_instance_prompt(
                np_text, legend_k, style, template, open_ended=False)
            scheme = "named_palette_legend_v2"
        for rank, idx in enumerate(order):
            rgb[inst_masks[idx]] = legend_colors[rank]  # canonical inst #rank -> color #rank
        target_name = f"{basename}_seg{v}.png"
        Image.fromarray(rgb, "RGB").save(out_target / target_name)
        rows.append({
            "conditioning_image": f"conditioning_images/{input_name}",
            "image": f"target_images/{target_name}",
            "text": prompt,
            "task": "instance_segmentation",
            "source_dataset": pair.dataset,
            "id": f"{basename}_v{v}",
            "split": pair.split,
            "metadata": {
                "stem": pair.stem,
                "image_path": str(pair.image_path),
                "ann_path": str(pair.ann_primary),
                "noun_phrase": np_text,
                "class_id": int(cls),
                "num_instances": n_inst,
                "variant_type": "negative" if is_negative else "positive",
                "color_scheme": scheme,
                "notation_style": style,
                "legend_len": legend_k,
                **({
                    "prompt_contract": mode_cfg["prompt_contract"],
                    "palette_size": draw_k,
                } if mode_cfg is not None else {}),
                # randomdraw_* only: legend entry i names pool color permutation[i];
                # unlisted modes keep it as target-generation provenance.
                **({"permutation": permutation} if permutation is not None else {}),
            },
        })
    return rows


def write_dataset(pairs: Iterable[Pair], out_root: Path, cfg: Config, num_workers: int = 1) -> int:
    pairs = list(pairs)
    out_input, out_target = out_root / "conditioning_images", out_root / "target_images"
    out_input.mkdir(parents=True, exist_ok=True)
    out_target.mkdir(parents=True, exist_ok=True)
    desc = f"Writing instance variants ({num_workers} workers)"
    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            results = list(tqdm(ex.map(write_image_variants, pairs, repeat(out_input),
                                       repeat(out_target), repeat(cfg)),
                                total=len(pairs), desc=desc))
    else:
        results = [write_image_variants(p, out_input, out_target, cfg)
                   for p in tqdm(pairs, desc=desc)]

    rows = [r for rs in results for r in rs]
    (out_root / "metadata.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--ade-root", default=None, help="ade20k_instanceseg/ root.")
    p.add_argument("--mapillary-root", default=None, help="mapillary_vistasv2/ root.")
    p.add_argument("--synscapes-root", default=None,
                   help="Synscapes root carrying img/{rgb,class,instance} and meta/ (or its parent).")
    p.add_argument("--synscapes-splits", default="train,val",
                   help="Comma-separated Synscapes splits (default train,val).")
    p.add_argument("--idd-root", default=None,
                   help="IDD root carrying gtFine/ + leftImg8bit/, or a parent containing "
                        "IDD_Segmentation/ and idd20kII/ from idd_1_2.tar.")
    p.add_argument("--idd-splits", default="train,val",
                   help="Comma-separated IDD splits (default train,val).")
    p.add_argument("--wilddash2-root", default=None,
                   help="WildDash 2 root carrying anonymized/, panoptic/, panoptic.json, and random_split/.")
    p.add_argument("--wilddash2-splits", default="train,val",
                   help="Comma-separated WildDash 2 random splits (default train,val).")
    p.add_argument("--nuimages-root", default=None,
                   help="NuImages root with samples/ and v1.0-{train,val}/ JSON tables.")
    p.add_argument("--nuimages-splits", default="train,val",
                   help="NuImages labelled splits: train and/or val (default train,val).")
    p.add_argument("--nuimages-sample-interval-seconds", type=float,
                   default=taxonomy.NUIMAGES_DEFAULT_SAMPLE_INTERVAL_SECONDS,
                   help="Keep the first mask-bearing keyframe in each driving log and then one at least this many seconds later (default: 10). Use 0 to retain all mask-bearing keyframes.")
    p.add_argument("--coconut-xlarge-root", default=None,
                   help="Materialized COCONut-XL coco/ root (train2017/, panoptic_train2017/, annotations/panoptic_train2017.json); its merged manifest is emitted as train.")
    p.add_argument("--coconut-xlarge-groups", default="coco,objects365",
                   help="Comma-separated merged COCONut-XL source groups to retain: coco, objects365, or both (default both).")
    p.add_argument("--paco-lvis-root", default=None,
                   help="PACO-LVIS DatasetNinja root with meta.json and {train,val,test}2017/{img,ann}/.")
    p.add_argument("--paco-lvis-splits", default="train,val,test",
                   help="PACO-LVIS splits to use (default train,val,test).")
    p.add_argument("--refcocog-umd-root", default=str(coco_eval_holdouts.DEFAULT_REFCOCOG_UMD_ROOT),
                   help="Official RefCOCOg UMD annotation directory containing refs(umd).p; used to prevent held-out val images entering COCONut-XL and PACO-LVIS training.")
    p.add_argument("--allow-eval-image-overlap", action="store_true",
                   help="Disable the COCO val2017 and RefCOCOg UMD validation image holdout. Not suitable for evaluation training runs.")
    p.add_argument("--voc-root", default=None, help="VOCdevkit/VOC2012/ root.")
    p.add_argument("--muses-root", default=None, help="MUSES root with frame_camera/ and gt_panoptic/.")
    p.add_argument("--acdc-root", default=None, help="ACDC root with rgb_anon/ and gt_panoptic/.")
    p.add_argument("--acdc-conditions", default=None,
                   help="Comma-separated ACDC conditions to keep, e.g. fog,night (default: all).")
    p.add_argument("--goose-root", default=None, help="GOOSE 2D root with images/, labels/, goose_label_mapping.csv.")
    p.add_argument("--vipseg-root", default=None, help="VIPSeg root with imgs/, panomasks/, and VIPSeg_720P/panoVIPSeg_categories.json.")
    p.add_argument("--ovis-root", default=None, help="OVIS root with train[/valid] frame dirs and annotations_{train,valid}.json.")
    p.add_argument("--waymo-root", default=None,
                   help="Waymo root with images/{train,val}/ and panoptic/{train,val}/; "
                        "see scripts/dataset_prep/data_download_scripts/download_datasets_waymo.sh.")
    p.add_argument("--hypersim-root", default=None,
                   help="Hypersim root with images/{train,val}/, panoptic/{train,val}/ and "
                        "id2label.json; build it from the raw HDF5 release with "
                        "dataset_prep/hypersim/prepare_hypersim_segmentation.py.")
    p.add_argument("--ovis-fps", type=float, default=1.0,
                   help="Frame sampling rate for OVIS. Default: 1 fps.")
    p.add_argument("--ovis-source-fps", type=float, default=5.0,
                   help="Approximate FPS of the released OVIS annotated frame sequence. "
                        "Default: 5, matching the dataset note that videos are annotated every 5 frames.")
    p.add_argument("--splits", default="train,val", help="Comma-separated logical splits.")
    p.add_argument("--variants-per-image", type=int, default=1)
    p.add_argument("--negative-ratio", type=float, default=0.2,
                   help="Fraction of variants naming an absent class -> all-black target.")
    p.add_argument("--min-instance-area", type=float, default=0.0,
                   help="Drop instances smaller than this fraction of the image area.")
    p.add_argument("--max-instances", type=int, default=None,
                   help="Cap painted instances per class (keeps the largest by area, "
                        "deterministic). Always clamped to the palette size. Default: all.")
    p.add_argument("--legend-slack-prob", type=float, default=0.0,
                   help="Probability of listing 1-4 extra legend colors beyond the "
                        "instance count (teaches that listed colors may go unused, "
                        "matching eval's fixed-length legend). Default: exact count.")
    p.add_argument("--draw-mode", default="legend",
                   choices=["legend", *iseg_prompts.DRAW_MODES],
                   help="Color/prompt scheme. legend (default): v2 exact-count "
                        "legends. fixeddraw_{32,64}: constant K-entry legend in "
                        "fixed max-separated pool order. randomdraw_{32,64}: same "
                        "pool, per-sample random permutation stated in the legend. "
                        "randomdraw_32_unlisted: same randomized targets as "
                        "randomdraw_32, but asks only for a different/unique color "
                        "per instance (no mapping or scan rule). "
                        "fixeddraw_32_ordered_unlisted: same fixed targets as "
                        "fixeddraw_32, but states the top-to-bottom fixed palette "
                        "order without enumerating instance-to-color mappings. "
                        "fixeddraw_32_lr: fixeddraw_32 with left-to-right instance "
                        "numbering (ties top-to-bottom). fixeddraw_32_mixedrepresent: "
                        "fixeddraw_32 with per-sample 50/50 name/hex notation (all 32 "
                        "enumerated; long noun phrases fall back to shorter templates "
                        "via the budget cascade). 32 = named palette / name "
                        "notation (fits Flux2-Klein's 512-token budget); 64 = grid "
                        "palette / hex notation (~928 tokens -- longer-budget "
                        "backbones only, e.g. Qwen-Image-Edit 2511). Draw modes "
                        "ignore --legend-slack-prob and clamp --max-instances to K.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="Debug: cap total pairs.")
    return p


def main(args: argparse.Namespace) -> None:
    iseg_prompts.validate_templates()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    holdout_audit: coco_eval_holdouts.CocoEvaluationHoldoutAudit | None = None
    if (args.coconut_xlarge_root or args.paco_lvis_root) and not args.allow_eval_image_overlap:
        holdout_audit = coco_eval_holdouts.CocoEvaluationHoldoutAudit(
            coco_eval_holdouts.load_coco_evaluation_holdouts(Path(args.refcocog_umd_root))
        )
        print(
            "Protecting evaluation images:",
            {name: len(ids) for name, ids in holdout_audit.holdouts.items()},
        )

    datasets = build_dataset_info(args)
    if not datasets:
        raise SystemExit("No sources given. Set at least one dataset root, including --synscapes-root, --nuimages-root, or --paco-lvis-root, or --coconut-xlarge-root.")

    pairs: List[Pair] = []
    coco_holdouts = holdout_audit.holdouts if holdout_audit is not None else {}
    if args.ade_root:
        pairs += build_ade_pairs(Path(args.ade_root), splits)
    if args.mapillary_root:
        pairs += build_mapillary_pairs(Path(args.mapillary_root), splits)
    if args.synscapes_root:
        synscapes_splits_value = args.synscapes_splits or args.splits
        synscapes_splits = [s.strip() for s in synscapes_splits_value.split(",") if s.strip()]
        pairs += build_synscapes_pairs(Path(args.synscapes_root), synscapes_splits)
    if args.idd_root:
        idd_splits_value = args.idd_splits or args.splits
        idd_splits = [s.strip() for s in idd_splits_value.split(",") if s.strip()]
        pairs += build_idd_pairs(Path(args.idd_root), idd_splits)
    if args.wilddash2_root:
        wilddash2_splits_value = args.wilddash2_splits or args.splits
        wilddash2_splits = [s.strip() for s in wilddash2_splits_value.split(",") if s.strip()]
        pairs += build_wilddash2_pairs(Path(args.wilddash2_root), wilddash2_splits)
    if args.nuimages_root:
        nuimages_splits_value = args.nuimages_splits or args.splits
        nuimages_splits = [s.strip() for s in nuimages_splits_value.split(",") if s.strip()]
        pairs += build_nuimages_pairs(
            Path(args.nuimages_root), nuimages_splits, args.nuimages_sample_interval_seconds)
    if args.coconut_xlarge_root:
        pairs += build_coconut_xlarge_pairs(
            Path(args.coconut_xlarge_root), coco_holdouts, holdout_audit,
            groups=args.coconut_xlarge_groups,
        )
    if args.paco_lvis_root:
        paco_lvis_splits_value = args.paco_lvis_splits or args.splits
        paco_lvis_splits = [s.strip() for s in paco_lvis_splits_value.split(",") if s.strip()]
        pairs += build_paco_lvis_pairs(Path(args.paco_lvis_root), paco_lvis_splits, coco_holdouts, holdout_audit)
    if args.voc_root:
        pairs += build_voc_pairs(Path(args.voc_root), splits)
    if args.muses_root:
        pairs += build_muses_pairs(Path(args.muses_root), splits)
    if args.acdc_root:
        conds = [c for c in (args.acdc_conditions or "").split(",") if c.strip()] or None
        pairs += build_acdc_pairs(Path(args.acdc_root), splits, conds)
    if args.goose_root:
        pairs += build_goose_pairs(Path(args.goose_root), splits)
    if args.vipseg_root:
        pairs += build_vipseg_pairs(Path(args.vipseg_root), splits)
    if args.ovis_root:
        pairs += build_ovis_pairs(Path(args.ovis_root), splits, args.ovis_fps, args.ovis_source_fps)
    if args.waymo_root:
        pairs += build_waymo_pairs(Path(args.waymo_root), splits)
    if args.hypersim_root:
        pairs += build_hypersim_pairs(Path(args.hypersim_root), splits)
    if args.limit:
        rng = random.Random(args.seed)
        rng.shuffle(pairs)
        pairs = pairs[: args.limit]

    by_ds: Dict[str, int] = {}
    for pr in pairs:
        by_ds[pr.dataset] = by_ds.get(pr.dataset, 0) + 1
    print(f"Discovered {len(pairs)} image pairs: {by_ds}")

    cfg = Config(
        variants_per_image=args.variants_per_image,
        negative_ratio=args.negative_ratio,
        min_instance_area=args.min_instance_area,
        max_instances=args.max_instances,
        legend_slack_prob=args.legend_slack_prob,
        draw_mode=args.draw_mode,
        seed=args.seed,
        datasets=datasets,
    )
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if holdout_audit is not None:
        audit_path = holdout_audit.write(out_root)
        print(f"Wrote evaluation-image exclusion audit: {audit_path}")
    n = write_dataset(pairs, out_root, cfg, num_workers=args.num_workers)
    print(f"Wrote {n} instance-seg variants to {out_root}/metadata.jsonl")


if __name__ == "__main__":
    main(build_argparser().parse_args())
