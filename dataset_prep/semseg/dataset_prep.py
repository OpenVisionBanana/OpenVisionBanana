#!/usr/bin/env python3
"""Build the image-to-image semantic-segmentation training pool.

For every (image, ground-truth mask) pair across Cityscapes / ACDC / BDD /
Synscapes / IDD / WildDash 2 / NuImages / COCONut-XL / MUSES / GOOSE / VIPSeg / Mapillary-Vistas / ADE20k / Pascal-VOC /
PASCAL-Context / FoodSeg103 / Trans10K-v2 / RailSem19 / Waymo / Hypersim, this emits one or more
randomized (prompt, colored-mask) variants:

  1. read the GT into a canonical class-index array (common/semseg/taxonomy.py),
  2. take the N foreground classes present, subsample M of them (N==1 -> that one;
     N>1 -> a strict proper subset),
  3. assign each selected class a color drawn from the shared 200-color palette
     (or the 37 everyday-named colors when the prompt uses the natural-language
     notation), well separated within the image,
  4. paint only those classes (everything else black, or a random background color
     with --background-color-prob), and
  5. write a prompt carrying the matching color legend, in a per-prompt-random
     notation (name / rgb tuple / hex) and base template.

The color/notation/template/decoding logic lives in common/semseg, shared with
evaluation.

Layouts are configured per dataset via the --*-root flags (see argparse below).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import Iterable, Optional, Sequence
# Make repository-level common/ importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import numpy as np
from PIL import Image
from tqdm import tqdm

# Shared semseg vocabulary (palettes, notations, templates, sampler) + taxonomies.
from common import coco_eval_holdouts, coconut_xlarge  # noqa: E402
from common.semseg import diversify, taxonomy  # noqa: E402

ACDC_CONDITIONS = ("fog", "night", "rain", "snow")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class Pair:
    dataset: str          # taxonomy key: cityscapes/acdc/bdd/ade20k/voc/pascal_context/
                          # foodseg103/trans10k/vipseg/mapillary
    split: str
    stem: str
    image_path: Path
    mask_path: Path
    group: str = "default"
    segments_info: tuple[dict, ...] = ()  # WildDash 2 COCO-panoptic metadata
    annotations: tuple[dict, ...] = ()    # NuImages surface/object COCO RLEs


# --------------------------------------------------------------------------- #
# small helpers                                                               #
# --------------------------------------------------------------------------- #
def split_csv(value: Optional[str], default: Sequence[str]) -> list[str]:
    if value is None:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_name(*parts: str) -> str:
    joined = "__".join(str(part) for part in parts if part)
    return joined.replace("/", "_").replace("\\", "_").replace(" ", "_")


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".png":
        shutil.copy2(src, dst)
    else:
        with Image.open(src) as image:
            image.convert("RGB").save(dst)


def find_image_by_stem(image_dir: Path, stem: str, suffixes: Sequence[str] = IMAGE_SUFFIXES) -> Optional[Path]:
    for suffix in suffixes:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


# --------------------------------------------------------------------------- #
# dataset pair discovery                                                      #
# --------------------------------------------------------------------------- #
def build_cityscapes_pairs(root: Path, splits: Sequence[str], mask_root: Optional[Path] = None,
                           trainid_suffixes: Sequence[str] = ("_gtFine_labelTrainIds.png", "_labelTrainIds.png")
                           ) -> tuple[list[Pair], int]:
    pairs, skipped = [], 0
    for split in splits:
        image_root = root / "leftImg8bit" / split
        split_mask_root = (mask_root / split) if mask_root else (root / "gtFine" / split)
        for image_path in sorted(image_root.glob("*/*_leftImg8bit.png")):
            city = image_path.parent.name
            stem = image_path.name.replace("_leftImg8bit.png", "")
            mask = next((split_mask_root / city / f"{stem}{s}" for s in trainid_suffixes
                         if (split_mask_root / city / f"{stem}{s}").exists()), None)
            if mask is not None:
                pairs.append(Pair("cityscapes", split, stem, image_path, mask, city))
            else:
                skipped += 1
    return pairs, skipped


def build_synscapes_pairs(root: Path, splits: Sequence[str]) -> tuple[list[Pair], int]:
    """Discover native-resolution RGB and raw Cityscapes-ID semantic masks."""
    resolved = taxonomy.resolve_synscapes_root(root)
    selected = taxonomy.normalize_synscapes_splits(splits)
    pairs, skipped = [], 0
    for index, mask_path in taxonomy.synscapes_numbered_pngs(resolved / "img" / "class"):
        split = taxonomy.synscapes_split_for_index(index)
        if split not in selected:
            continue
        image_path = resolved / "img" / "rgb" / f"{index}.png"
        if not image_path.exists():
            skipped += 1
            continue
        pairs.append(Pair(
            "synscapes", split, str(index), image_path, mask_path, "numbered"))
    return pairs, skipped


def build_idd_pairs(root: Path, splits: Sequence[str]) -> tuple[list[Pair], int]:
    """Discover IDD RGB/JSON pairs across either or both release archives.

    ``idd_1_2.tar`` expands to two source directories. Prefixing output stems
    with the part and drive avoids collisions from generic frame names such as
    ``frame0149`` that recur between drives.
    """
    pairs, skipped = [], 0
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
                if not image_path.exists():
                    skipped += 1
                    continue
                pairs.append(Pair(
                    "idd", split, f"{part}__{drive}__{image_stem}", image_path,
                    annotation_path, f"{part}_{drive}",
                ))
    return pairs, skipped


def build_wilddash2_pairs(root: Path, splits: Sequence[str]) -> tuple[list[Pair], int]:
    """Build WildDash 2 RGB/panoptic pairs using its official random split."""
    resolved = taxonomy.resolve_wilddash2_root(root)
    taxonomy.register_wilddash2_metadata(resolved)
    selected = taxonomy.normalize_wilddash2_splits(splits)
    index = json.loads((resolved / "panoptic.json").read_text(encoding="utf-8"))
    images = {Path(str(image["file_name"])).stem: image for image in index.get("images", ())}
    annotations = {str(ann.get("image_id")): ann for ann in index.get("annotations", ())}
    split_files = {"train": "training.txt", "val": "validation.txt"}
    pairs: list[Pair] = []
    skipped = 0
    for split in sorted(selected):
        stems = (resolved / "random_split" / split_files[split]).read_text(encoding="utf-8").split()
        for stem in stems:
            image = images.get(stem)
            if image is None:
                skipped += 1
                continue
            annotation = annotations.get(str(image.get("id")))
            if annotation is None:
                skipped += 1
                continue
            image_path = resolved / "anonymized" / str(image["file_name"])
            mask_path = resolved / "panoptic" / str(annotation["file_name"])
            if not image_path.exists() or not mask_path.exists():
                skipped += 1
                continue
            pairs.append(Pair(
                "wilddash2", split, stem, image_path, mask_path,
                group=f"country_{stem[:2]}",
                segments_info=tuple(annotation.get("segments_info", ())),
            ))
    return pairs, skipped


def build_coconut_xlarge_pairs(
    root: Path,
    holdouts: dict[str, frozenset[str]] | None = None,
    audit: coco_eval_holdouts.CocoEvaluationHoldoutAudit | None = None,
    groups: str | Iterable[str] | None = None,
) -> tuple[list[Pair], int]:
    """Build selected materialized COCONut-XL COCO-panoptic pairs.

    The final COCONut-XL release exposes one merged train manifest rather than
    separate train/validation partitions, so all rows retain the logical
    ``train`` split. ``groups`` selects the merged manifest provenance:
    ``coco``, ``objects365``, or both.
    """
    selected_groups = coconut_xlarge.normalize_coconut_xlarge_groups(groups)
    resolved = taxonomy.resolve_coconut_xlarge_root(root)
    taxonomy.register_coconut_xlarge_metadata(resolved)
    index_path = taxonomy.coconut_xlarge_annotation_path(resolved)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    images = {str(image.get("id")): image for image in index.get("images", ())}
    pairs: list[Pair] = []
    skipped = 0
    excluded = Counter()
    excluded_images = 0
    excluded_groups = Counter()
    for annotation in index.get("annotations", ()):
        image = images.get(str(annotation.get("image_id")))
        if image is None:
            skipped += 1
            continue
        group = coconut_xlarge.coconut_xlarge_source_group(image)
        if group not in selected_groups:
            excluded_groups[group] += 1
            continue
        image_name = str(image.get("file_name") or "")
        reasons = coco_eval_holdouts.holdout_reasons(image_name, holdouts or {})
        if reasons:
            excluded.update(reasons)
            excluded_images += 1
            if audit is not None:
                audit.record(
                    image_name, reasons, source="coconut_xlarge_semseg",
                    reference=str(annotation.get("image_id") or image.get("id") or ""),
                )
            continue
        image_path = resolved / "train2017" / str(image.get("file_name") or "")
        mask_path = resolved / "panoptic_train2017" / str(annotation.get("file_name") or "")
        if not image_path.is_file() or not mask_path.is_file():
            skipped += 1
            continue
        stem = str(image.get("object365_name") or Path(str(image["file_name"])).stem)
        pairs.append(Pair(
            "coconut_xlarge", "train", stem, image_path, mask_path, group,
            segments_info=tuple(annotation.get("segments_info", ())),
        ))
    if excluded_groups:
        print(f"Excluded COCONut-XL source groups outside {sorted(selected_groups)}: {dict(sorted(excluded_groups.items()))}")
    if excluded:
        print(f"Excluded {excluded_images} COCONut-XL evaluation images: {dict(sorted(excluded.items()))}")
    return pairs, skipped

def build_nuimages_pairs(
    root: Path,
    splits: Sequence[str],
    sample_interval_seconds: float = taxonomy.NUIMAGES_DEFAULT_SAMPLE_INTERVAL_SECONDS,
) -> tuple[list[Pair], int]:
    """Build mask-bearing NuImages key-camera pairs, sampled per driving log."""
    resolved = taxonomy.resolve_nuimages_root(root)
    frames, skipped = taxonomy.discover_nuimages_frames(
        resolved, splits, sample_interval_seconds=sample_interval_seconds)
    return [
        Pair(
            "nuimages", frame.split, frame.token, frame.image_path,
            resolved / f"v1.0-{frame.split}" / "object_ann.json",
            group=f"log_{frame.log_token}", annotations=frame.annotations,
        )
        for frame in frames
    ], skipped


def acdc_dirs(root: Path, split: str, condition: Optional[str]) -> tuple[list[Path], list[Path]]:
    if condition is not None:
        return [root / "rgb_anon" / condition / split], [root / "gt" / condition / split]
    flat_image, flat_gt = root / "rgb_anon" / split, root / "gt" / split
    if flat_image.exists() or flat_gt.exists():
        return [flat_image], [flat_gt]
    return ([root / "rgb_anon" / c / split for c in ACDC_CONDITIONS],
            [root / "gt" / c / split for c in ACDC_CONDITIONS])


def build_acdc_pairs(root: Path, splits: Sequence[str], conditions: Optional[Sequence[str]] = None
                     ) -> tuple[list[Pair], int]:
    pairs, skipped = [], 0
    selected = list(conditions) if conditions else [None]
    for split in splits:
        for condition in selected:
            image_dirs, gt_dirs = acdc_dirs(root, split, condition)
            for image_dir, gt_dir in zip(image_dirs, gt_dirs):
                if not image_dir.exists():
                    continue
                group = condition or (image_dir.parent.name if image_dir.parent.name in ACDC_CONDITIONS else "flat")
                for image_path in sorted(image_dir.rglob("*_rgb_anon.png")):
                    rel = image_path.parent.relative_to(image_dir)
                    stem = image_path.name.replace("_rgb_anon.png", "")
                    mask = gt_dir / rel / f"{stem}_gt_labelTrainIds.png"
                    if mask.exists():
                        pairs.append(Pair("acdc", split, stem, image_path, mask, group))
                    else:
                        skipped += 1
    return pairs, skipped


def build_bdd_pairs(root: Path, splits: Sequence[str]) -> tuple[list[Pair], int]:
    pairs, skipped = [], 0
    for split in splits:
        image_dir, mask_dir = root / "images" / split, root / "labels" / split
        if not mask_dir.exists():
            continue
        for mask_path in sorted(mask_dir.glob("*_train_id.png")):
            stem = mask_path.name.replace("_train_id.png", "")
            image_path = find_image_by_stem(image_dir, stem)
            if image_path is None:
                skipped += 1
                continue
            pairs.append(Pair("bdd", split, stem, image_path, mask_path, split))
    return pairs, skipped


def build_muses_pairs(root: Path, splits: Sequence[str]) -> tuple[list[Pair], int]:
    """MUSES frame-camera images + Cityscapes trainId semantic masks.

    Layout:
      frame_camera/<split>/<weather>/<time>/<stem>_frame_camera.png
      gt_semantic/<split>/<weather>/<time>/<stem>_gt_labelTrainIds.png
    """
    pairs, skipped = [], 0
    for split in splits:
        image_root = root / "frame_camera" / split
        mask_root = root / "gt_semantic" / split
        if not image_root.exists() or not mask_root.exists():
            continue
        for image_path in sorted(image_root.glob("*/*/*_frame_camera.png")):
            rel_parent = image_path.parent.relative_to(image_root)
            stem = image_path.name.replace("_frame_camera.png", "")
            mask = mask_root / rel_parent / f"{stem}_gt_labelTrainIds.png"
            if mask.exists():
                weather = rel_parent.parts[0] if len(rel_parent.parts) >= 1 else "unknown"
                time_of_day = rel_parent.parts[1] if len(rel_parent.parts) >= 2 else "unknown"
                pairs.append(Pair("muses", split, stem, image_path, mask, f"{weather}_{time_of_day}"))
            else:
                skipped += 1
    return pairs, skipped


def _goose_key(path: Path, suffix: str) -> str:
    stem = path.name.replace(suffix, "")
    while "__" in stem:
        stem = stem.replace("__", "_")
    return stem


def build_goose_pairs(root: Path, splits: Sequence[str]) -> tuple[list[Pair], int]:
    """GOOSE 2D RGB images + fine-grained labelids masks.

    Layout after the official setup:
      images/<split>/<sequence>/*_windshield_vis.png
      labels/<split>/<sequence>/*_labelids.png
      labels/<split>/<sequence>/*_instanceids.png
      goose_label_mapping.csv
    """
    taxonomy.register_goose_mapping(root)
    pairs, skipped = [], 0
    for split in splits:
        image_root = root / "images" / split
        mask_root = root / "labels" / split
        if not image_root.exists() or not mask_root.exists():
            continue
        images_by_key: dict[tuple[str, str], Path] = {}
        for image_path in sorted(image_root.glob("*/*_windshield_vis.png")):
            group = image_path.parent.name
            images_by_key[(group, _goose_key(image_path, "_windshield_vis.png"))] = image_path
        for mask_path in sorted(mask_root.glob("*/*_labelids.png")):
            group = mask_path.parent.name
            key = (group, _goose_key(mask_path, "_labelids.png"))
            image_path = images_by_key.get(key)
            if image_path is None:
                skipped += 1
                continue
            pairs.append(Pair("goose", split, key[1], image_path, mask_path, group))
    return pairs, skipped


def build_mapillary_pairs(root: Path, splits: Sequence[str], version: str = "v2.0") -> tuple[list[Pair], int]:
    """Mapillary Vistas v2.0 RGB images + paletted label masks.

    Official layout:
      config_v2.0.json
      {training,validation,testing}/images/*.jpg
      {training,validation,testing}/v2.0/labels/*.png
    """
    taxonomy.register_mapillary_config(root)
    pairs, skipped = [], 0
    alias = {"train": "training", "training": "training",
             "val": "validation", "validation": "validation",
             "test": "testing", "testing": "testing"}
    for split in splits:
        mv_split = alias.get(split, split)
        image_dir = root / mv_split / "images"
        mask_dir = root / mv_split / version / "labels"
        if not image_dir.exists() or not mask_dir.exists():
            continue
        for mask_path in sorted(mask_dir.glob("*.png")):
            image_path = find_image_by_stem(image_dir, mask_path.stem, (".jpg", ".jpeg", ".png"))
            if image_path is not None:
                pairs.append(Pair("mapillary", split, mask_path.stem, image_path, mask_path, mv_split))
            else:
                skipped += 1
    return pairs, skipped


def build_ade20k_pairs(root: Path, splits: Sequence[str]) -> tuple[list[Pair], int]:
    pairs, skipped = [], 0
    alias = {"train": "training", "training": "training", "val": "validation", "validation": "validation"}
    for split in splits:
        ade_split = alias.get(split, split)
        image_dir, mask_dir = root / "images" / ade_split, root / "annotations" / ade_split
        if not image_dir.exists() or not mask_dir.exists():
            continue
        for mask_path in sorted(mask_dir.glob("*.png")):
            stem = mask_path.stem
            image_path = find_image_by_stem(image_dir, stem, (".jpg", ".jpeg", ".png"))
            if image_path is not None:
                pairs.append(Pair("ade20k", split, stem, image_path, mask_path, ade_split))
            else:
                skipped += 1
    return pairs, skipped


def build_voc_pairs(root: Path, splits: Sequence[str], mask_dir_name: str = "SegmentationClass"
                    ) -> tuple[list[Pair], int]:
    pairs, skipped = [], 0
    image_dir, sets_dir = root / "images", root / "sets"
    for split in splits:
        split_file = sets_dir / f"{split}.txt"
        if split_file.exists():
            stems = [ln.strip() for ln in split_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        elif sets_dir.exists():
            continue
        else:
            md = root / mask_dir_name
            stems = [p.stem for p in sorted(md.glob("*.png"))] if md.exists() else []
        for stem in stems:
            image_path = find_image_by_stem(image_dir, stem, (".jpg", ".jpeg", ".png"))
            candidates = []
            if split.startswith("trainaug"):
                candidates.append(root / "SegmentationClassAug" / f"{stem}.png")
            candidates += [root / mask_dir_name / f"{stem}.png",
                           root / "SegmentationClass" / f"{stem}.png",
                           root / "SegmentationClassAug" / f"{stem}.png"]
            mask_path = next((c for c in candidates if c.exists()), None)
            if image_path is not None and mask_path is not None:
                pairs.append(Pair("voc", split, stem, image_path, mask_path, split))
            else:
                skipped += 1
    return pairs, skipped


PASCAL_CONTEXT_DIR_NAME = "pascal_context_59classlabels"


def build_pascal_context_pairs(root: Path, splits: Sequence[str], image_root: Optional[Path] = None
                               ) -> tuple[list[Pair], int]:
    """PASCAL-Context 59-class labels over the VOC images they annotate.

    Layout written by download_datasets_pascal_context.sh (packed inside voc.tar):
      voc/images/*.jpg
      voc/pascal_context_59classlabels/labels/*.png    # 0 = background, 1..59
      voc/pascal_context_59classlabels/sets/{train,val}.txt

    ``root`` accepts either the VOC root or the pascal_context_59classlabels dir;
    images default to <voc-root>/images and can be overridden with ``image_root``.
    """
    root = Path(root)
    labels_root = root if (root / "labels").is_dir() else root / PASCAL_CONTEXT_DIR_NAME
    voc_root = labels_root.parent
    images_dir = Path(image_root) if image_root else voc_root / "images"
    taxonomy.verify_pascal_context_labels(labels_root)

    pairs, skipped = [], 0
    alias = {"training": "train", "validation": "val"}
    label_dir, sets_dir = labels_root / "labels", labels_root / "sets"
    if not label_dir.is_dir():
        return pairs, skipped
    for split in splits:
        pc_split = alias.get(split, split)
        split_file = sets_dir / f"{pc_split}.txt"
        if split_file.exists():
            stems = [ln.strip() for ln in split_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        elif sets_dir.exists():
            continue  # split lists exist but not this split -> nothing to do
        else:
            stems = [p.stem for p in sorted(label_dir.glob("*.png"))]
        for stem in stems:
            mask_path = label_dir / f"{stem}.png"
            image_path = find_image_by_stem(images_dir, stem, (".jpg", ".jpeg", ".png"))
            if image_path is not None and mask_path.exists():
                pairs.append(Pair("pascal_context", pc_split, stem, image_path, mask_path, pc_split))
            else:
                skipped += 1
    return pairs, skipped


def build_foodseg103_pairs(root: Path, splits: Sequence[str]) -> tuple[list[Pair], int]:
    """FoodSeg103 materialized from Hugging Face parquet rows.

    Normalized layout produced by download_datasets_foodseg103.sh:
      images/{train,val}/*.png
      labels/{train,val}/*.png   # grayscale uint8, raw 0..103 class ids
    """
    pairs, skipped = [], 0
    alias = {"validation": "val"}
    for split in splits:
        food_split = alias.get(split, split)
        image_dir = root / "images" / food_split
        mask_dir = root / "labels" / food_split
        if not image_dir.exists() or not mask_dir.exists():
            continue
        for mask_path in sorted(mask_dir.glob("*.png")):
            stem = mask_path.stem
            image_path = find_image_by_stem(image_dir, stem, (".png", ".jpg", ".jpeg"))
            if image_path is not None:
                pairs.append(Pair("foodseg103", food_split, stem, image_path, mask_path, food_split))
            else:
                skipped += 1
    return pairs, skipped


def build_trans10k_pairs(root: Path, splits: Sequence[str]) -> tuple[list[Pair], int]:
    """Trans10K-v2 materialized from Hugging Face parquet rows.

    Normalized layout produced by download_datasets_trans10k.sh:
      images/{train,val,test}/*.jpg
      labels/{train,val,test}/*.png   # grayscale uint8, 0 = background, 1..11
    """
    root = Path(root)
    taxonomy.verify_trans10k_labels(root)
    pairs, skipped = [], 0
    alias = {"validation": "val", "training": "train", "testing": "test"}
    for split in splits:
        t10k_split = alias.get(split, split)
        image_dir = root / "images" / t10k_split
        mask_dir = root / "labels" / t10k_split
        if not image_dir.exists() or not mask_dir.exists():
            continue
        for mask_path in sorted(mask_dir.glob("*.png")):
            stem = mask_path.stem
            image_path = find_image_by_stem(image_dir, stem, (".jpg", ".jpeg", ".png"))
            if image_path is not None:
                pairs.append(Pair("trans10k", t10k_split, stem, image_path, mask_path, t10k_split))
            else:
                skipped += 1
    return pairs, skipped


def resolve_railsem19_root(root: Path) -> Optional[Path]:
    """Find the directory holding jpgs/ + uint8/ inside a RailSem19 checkout.

    The release nests them under versions/<n>/, and the staged tar keeps that
    native layout, so accept either the folder itself or any parent of it.
    """
    root = Path(root)
    if (root / "jpgs").is_dir() and (root / "uint8").is_dir():
        return root
    for candidate in sorted(root.glob("**/uint8")):
        if candidate.is_dir() and (candidate.parent / "jpgs").is_dir():
            return candidate.parent
    return None


def build_railsem19_pairs(root: Path, splits: Sequence[str]) -> tuple[list[Pair], int]:
    """RailSem19 rail-scene semantic segmentation (https://www.wilddash.cc/railsem19).

    Native release layout (kept by the staged tar):
      jpgs/rs19_val/*.jpg
      uint8/rs19_val/*.png    # grayscale uint8, 0..18 = classes, 255 = void

    RailSem19 ships all 8,500 frames as one annotated pool with no official split
    (the folder is called rs19_val regardless), so every frame is emitted under
    "train" and nothing is held out.
    """
    resolved = resolve_railsem19_root(root)
    if resolved is None:
        return [], 0
    taxonomy.verify_railsem19_labels(resolved)

    alias = {"training": "train"}
    if "train" not in {alias.get(split, split) for split in splits}:
        return [], 0
    pairs, skipped = [], 0
    for mask_dir in sorted(p for p in (resolved / "uint8").iterdir() if p.is_dir()):
        image_dir = resolved / "jpgs" / mask_dir.name
        if not image_dir.is_dir():
            continue
        for mask_path in sorted(mask_dir.glob("*.png")):
            stem = mask_path.stem
            image_path = find_image_by_stem(image_dir, stem, (".jpg", ".jpeg", ".png"))
            if image_path is not None:
                pairs.append(Pair("railsem19", "train", stem, image_path, mask_path, mask_dir.name))
            else:
                skipped += 1
    return pairs, skipped


def build_waymo_pairs(root: Path, splits: Sequence[str]) -> tuple[list[Pair], int]:
    """Waymo Open Dataset camera panoptic labels, read for their semantic half.

    Normalized layout produced by download_datasets_waymo.sh:
      images/{train,val}/<context>_<timestamp>_<CAMERA>.jpg
      panoptic/{train,val}/<context>_<timestamp>_<CAMERA>.png  # uint16 packed

    The same tree feeds dataset_prep/instanceseg/dataset_prep.py, which reads the
    instance half of those pixels.
    """
    root = Path(root)
    taxonomy.verify_waymo_labels(root)
    pairs, skipped = [], 0
    alias = {"training": "train", "validation": "val"}
    for split in splits:
        waymo_split = alias.get(split, split)
        image_dir = root / "images" / waymo_split
        mask_dir = root / "panoptic" / waymo_split
        if not image_dir.exists() or not mask_dir.exists():
            continue
        for mask_path in sorted(mask_dir.glob("*.png")):
            stem = mask_path.stem
            image_path = find_image_by_stem(image_dir, stem, (".jpg", ".jpeg", ".png"))
            if image_path is not None:
                # Group by camera so --num-images samples evenly across viewpoints.
                camera = stem.rsplit("_", 1)[-1]
                pairs.append(Pair("waymo", waymo_split, stem, image_path, mask_path, camera))
            else:
                skipped += 1
    return pairs, skipped


def build_hypersim_pairs(root: Path, splits: Sequence[str]) -> tuple[list[Pair], int]:
    """Hypersim photorealistic synthetic indoor scenes, NYU40 semantics.

    Layout produced by dataset_prep/hypersim/prepare_hypersim_segmentation.py, which
    normalizes the release's per-frame HDF5 channels into the same shape the Waymo
    download script writes:
      images/{train,val}/<scene>_<cam>_<frame>.jpg
      panoptic/{train,val}/<scene>_<cam>_<frame>.png   # uint16 packed

    The same tree feeds dataset_prep/instanceseg/dataset_prep.py, which reads the
    instance half of those pixels.
    """
    root = Path(root)
    taxonomy.verify_hypersim_labels(root)
    pairs, skipped = [], 0
    alias = {"training": "train", "validation": "val"}
    for split in splits:
        hs_split = alias.get(split, split)
        image_dir = root / "images" / hs_split
        mask_dir = root / "panoptic" / hs_split
        if not image_dir.exists() or not mask_dir.exists():
            continue
        for mask_path in sorted(mask_dir.glob("*.png")):
            stem = mask_path.stem
            image_path = find_image_by_stem(image_dir, stem, (".jpg", ".jpeg", ".png"))
            if image_path is not None:
                # Group by scene so --num-images samples evenly across rooms rather
                # than piling onto whichever scene has the most rendered frames.
                scene = stem.split("_cam_")[0]
                pairs.append(Pair("hypersim", hs_split, stem, image_path, mask_path, scene))
            else:
                skipped += 1
    return pairs, skipped


def _split_video_ids(root: Path, split: str) -> list[str]:
    candidates = [root / f"{split}.txt", root / "VIPSeg_720P" / f"{split}.txt"]
    split_path = next((path for path in candidates if path.exists()), None)
    if split_path is None:
        return []
    return [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_vipseg_pairs(root: Path, splits: Sequence[str]) -> tuple[list[Pair], int]:
    """VIPSeg native video panoptic masks.

    Layout from the official tar:
      imgs/<video>/*.jpg
      panomasks/<video>/*.png
      {train,val,test}.txt or VIPSeg_720P/{train,val,test}.txt  # video ids
      VIPSeg_720P/panoVIPSeg_categories.json
    """
    taxonomy.register_vipseg_categories(root)
    pairs, skipped = [], 0
    image_root = root / "imgs"
    mask_root = root / "panomasks"
    if not image_root.exists() or not mask_root.exists():
        return pairs, skipped
    for split in splits:
        for video in _split_video_ids(root, split):
            video_mask_dir = mask_root / video
            video_image_dir = image_root / video
            if not video_mask_dir.exists() or not video_image_dir.exists():
                skipped += 1
                continue
            for mask_path in sorted(video_mask_dir.glob("*.png")):
                image_path = find_image_by_stem(video_image_dir, mask_path.stem, (".jpg", ".jpeg", ".png"))
                if image_path is None:
                    skipped += 1
                    continue
                pairs.append(Pair("vipseg", split, mask_path.stem, image_path, mask_path, video))
    return pairs, skipped


def build_all_pairs(
    args: argparse.Namespace,
    holdout_audit: coco_eval_holdouts.CocoEvaluationHoldoutAudit | None = None,
) -> tuple[list[Pair], dict[str, int]]:
    splits = split_csv(args.split or args.splits, ("train", "val"))
    skipped: dict[str, int] = defaultdict(int)
    all_pairs: list[Pair] = []
    holdouts: Mapping[str, Iterable[str]] = {}
    if holdout_audit is not None:
        holdouts = holdout_audit.holdouts
    elif args.coconut_xlarge_root and not args.allow_eval_image_overlap:
        holdouts = coco_eval_holdouts.load_coco_evaluation_holdouts(
            Path(args.refcocog_umd_root))
        print(f"Protecting evaluation images: { {name: len(ids) for name, ids in holdouts.items()} }")
    if args.cityscapes_root:
        cityscapes_splits = split_csv(args.cityscapes_splits, splits)
        p, s = build_cityscapes_pairs(
            Path(args.cityscapes_root), cityscapes_splits,
            mask_root=Path(args.cityscapes_mask_root) if args.cityscapes_mask_root else None,
            trainid_suffixes=split_csv(args.cityscapes_trainid_suffixes,
                                       ("_gtFine_labelTrainIds.png", "_labelTrainIds.png")))
        all_pairs += p; skipped["cityscapes"] += s
    if args.synscapes_root:
        synscapes_splits = split_csv(args.synscapes_splits, splits)
        p, s = build_synscapes_pairs(Path(args.synscapes_root), synscapes_splits)
        all_pairs += p; skipped["synscapes"] += s
    if args.idd_root:
        idd_splits = split_csv(args.idd_splits, splits)
        p, s = build_idd_pairs(Path(args.idd_root), idd_splits)
        all_pairs += p; skipped["idd"] += s
    if args.wilddash2_root:
        wilddash2_splits = split_csv(args.wilddash2_splits, splits)
        p, s = build_wilddash2_pairs(Path(args.wilddash2_root), wilddash2_splits)
        all_pairs += p; skipped["wilddash2"] += s
    if args.nuimages_root:
        nuimages_splits = split_csv(args.nuimages_splits, splits)
        p, s = build_nuimages_pairs(
            Path(args.nuimages_root), nuimages_splits,
            args.nuimages_sample_interval_seconds)
        all_pairs += p; skipped["nuimages"] += s
    if args.coconut_xlarge_root:
        p, s = build_coconut_xlarge_pairs(
            Path(args.coconut_xlarge_root), holdouts, holdout_audit,
            groups=args.coconut_xlarge_groups,
        )
        all_pairs += p; skipped["coconut_xlarge"] += s
    if args.acdc_root:
        conds = split_csv(args.acdc_conditions, ()) if args.acdc_conditions else None
        p, s = build_acdc_pairs(Path(args.acdc_root), splits, conds); all_pairs += p; skipped["acdc"] += s
    if args.bdd_root:
        p, s = build_bdd_pairs(Path(args.bdd_root), splits); all_pairs += p; skipped["bdd"] += s
    if args.muses_root:
        p, s = build_muses_pairs(Path(args.muses_root), splits); all_pairs += p; skipped["muses"] += s
    if args.goose_root:
        p, s = build_goose_pairs(Path(args.goose_root), splits); all_pairs += p; skipped["goose"] += s
    if args.vipseg_root:
        p, s = build_vipseg_pairs(Path(args.vipseg_root), splits); all_pairs += p; skipped["vipseg"] += s
    if args.mapillary_root:
        p, s = build_mapillary_pairs(Path(args.mapillary_root), splits, args.mapillary_version)
        all_pairs += p; skipped["mapillary"] += s
    if args.ade20k_root:
        p, s = build_ade20k_pairs(Path(args.ade20k_root), splits); all_pairs += p; skipped["ade20k"] += s
    if args.voc_root:
        p, s = build_voc_pairs(Path(args.voc_root), splits, args.voc_mask_dir); all_pairs += p; skipped["voc"] += s
    if args.pascal_context_root:
        p, s = build_pascal_context_pairs(
            Path(args.pascal_context_root), splits,
            image_root=Path(args.pascal_context_image_root) if args.pascal_context_image_root else None)
        all_pairs += p; skipped["pascal_context"] += s
    if args.foodseg103_root:
        p, s = build_foodseg103_pairs(Path(args.foodseg103_root), splits); all_pairs += p; skipped["foodseg103"] += s
    if args.trans10k_root:
        p, s = build_trans10k_pairs(Path(args.trans10k_root), splits); all_pairs += p; skipped["trans10k"] += s
    if args.railsem19_root:
        p, s = build_railsem19_pairs(Path(args.railsem19_root), splits)
        all_pairs += p; skipped["railsem19"] += s
    if args.waymo_root:
        p, s = build_waymo_pairs(Path(args.waymo_root), splits); all_pairs += p; skipped["waymo"] += s
    if args.hypersim_root:
        p, s = build_hypersim_pairs(Path(args.hypersim_root), splits); all_pairs += p; skipped["hypersim"] += s
    return all_pairs, dict(skipped)


def sample_balanced(pairs: list[Pair], num_images: Optional[int] = None, seed: int = 42) -> list[Pair]:
    if num_images is None or num_images >= len(pairs):
        return pairs
    rng = random.Random(seed)
    by_group: dict[tuple, list[Pair]] = defaultdict(list)
    for p in pairs:
        by_group[(p.dataset, p.split, p.group)].append(p)
    for g in by_group.values():
        rng.shuffle(g)
    order = sorted(by_group)
    rng.shuffle(order)
    selected: list[Pair] = []
    while len(selected) < num_images:
        progressed = False
        for g in order:
            if by_group[g]:
                selected.append(by_group[g].pop())
                progressed = True
                if len(selected) == num_images:
                    break
        if not progressed:
            break
    return sorted(selected, key=lambda p: (p.dataset, p.split, p.group, p.stem))


# --------------------------------------------------------------------------- #
# inverse-frequency weights (per dataset taxonomy)                            #
# --------------------------------------------------------------------------- #
def _present_for_pair(pair: Pair) -> tuple[str, list[int]]:
    try:
        arr = taxonomy.read_label_array(
            pair.mask_path, pair.dataset, pair.segments_info, pair.annotations)
        return pair.dataset, taxonomy.present_classes(arr, pair.dataset)
    except Exception:
        return pair.dataset, []


def compute_weights(pairs: list[Pair], power: float, num_workers: int) -> dict[str, list[float]]:
    """Per-dataset inverse-frequency class weights: w[c] = 1 / freq[c]**power."""
    counts: dict[str, list[int]] = {
        ds: [0] * len(taxonomy.class_names(ds)) for ds in {p.dataset for p in pairs}
    }
    desc = "Counting class frequencies"
    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            results = list(tqdm(ex.map(_present_for_pair, pairs), total=len(pairs), desc=desc))
    else:
        results = [_present_for_pair(p) for p in tqdm(pairs, total=len(pairs), desc=desc)]
    for ds, present in results:
        for c in present:
            counts[ds][c] += 1
    return {ds: [1.0 / (max(f, 1) ** power) for f in cnt] for ds, cnt in counts.items()}


# --------------------------------------------------------------------------- #
# per-image variant writing                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class DiversifyConfig:
    variants_per_image: int = 1
    # Dataset composition by variant category (must sum to 1.0):
    #   subset       -> random proper subset of the classes present in the mask
    #   full-set     -> all classes present in the mask
    #   negative-set -> proper subset of present + K absent classes that are named
    #                   in the prompt with a color but never painted
    #   full-dataset -> every class of the source's taxonomy: all present classes
    #                   painted plus every absent class named but never painted
    #                   (no max_negative_classes cap; legends reach 150 classes)
    subset_ratio: float = 1.0
    full_set_ratio: float = 0.0
    negative_set_ratio: float = 0.0
    full_dataset_ratio: float = 0.0
    # Cap on K for negative-set: K ~ uniform(1, min(#absent, max_negative_classes)).
    max_negative_classes: int = 30
    background_color_prob: float = 0.0
    canonical_anchor: bool = False
    min_class_area: float = 0.0
    color_min_delta_e: float = 25.0
    # How legend colors are drawn from the palette; one of
    # diversify.COLOR_MAPPING_MODES (see there for the semantics of each mode).
    color_mapping: str = "p200_randomnegdraw"
    seed: int = 42
    weights: Optional[dict[str, list[float]]] = None


def _paint_canonical_cityscapes(label_arr: np.ndarray) -> Image.Image:
    pal = taxonomy.CITYSCAPES_PALETTE
    rgb = np.zeros((*label_arr.shape, 3), dtype=np.uint8)
    valid = (label_arr >= 0) & (label_arr < len(pal))
    rgb[valid] = pal[label_arr[valid]]
    return Image.fromarray(rgb, "RGB")


def write_image_variants(pair: Pair, out_input: Path, out_target: Path, cfg: DiversifyConfig) -> list[dict]:
    basename = safe_name(pair.dataset, pair.split, pair.group, pair.stem)
    input_name = f"{basename}_input.png"
    # Conditioning images are independent of the color/variant knobs, so an
    # existing file is reused as-is (rebuilds may hardlink conditioning_images/
    # from a sibling pool; interrupted runs are resumable).
    if not (out_input / input_name).exists():
        copy_image(pair.image_path, out_input / input_name)

    label_arr = taxonomy.read_label_array(
        pair.mask_path, pair.dataset, pair.segments_info, pair.annotations)
    names = taxonomy.class_names(pair.dataset)
    present = taxonomy.present_classes(label_arr, pair.dataset)
    if cfg.min_class_area > 0 and present:
        total = label_arr.size
        present = [c for c in present if (label_arr == c).sum() / total >= cfg.min_class_area]

    rows: list[dict] = []

    # Optional canonical anchor (Cityscapes taxonomy only: full 19-class palette).
    if cfg.canonical_anchor and pair.dataset in ("cityscapes", "acdc", "bdd", "synscapes"):
        target_name = f"{basename}_canonical.png"
        _paint_canonical_cityscapes(label_arr).save(out_target / target_name)
        rows.append(_row(pair, input_name, target_name, diversify.FIXED_CITYSCAPES_PROMPT,
                         {"variant": "canonical", "style": "fixed"}))

    if not present:
        return rows

    rng = random.Random(f"{cfg.seed}:{basename}")
    weights = cfg.weights.get(pair.dataset) if cfg.weights else None
    present_set = set(present)
    bg_index = taxonomy.background_index(pair.dataset)
    absent = [c for c in range(len(names)) if c not in present_set and c != bg_index]
    # Only draw a per-variant category when full/negative are actually requested, so
    # pure-subset runs leave the RNG stream where the legacy single-category code did.
    use_categories = cfg.full_set_ratio > 0 or cfg.negative_set_ratio > 0 or cfg.full_dataset_ratio > 0
    for v in range(cfg.variants_per_image):
        if cfg.full_dataset_ratio > 0:
            category = rng.choices(
                ("subset", "full", "negative", "full_dataset"),
                weights=(cfg.subset_ratio, cfg.full_set_ratio, cfg.negative_set_ratio,
                         cfg.full_dataset_ratio),
            )[0]
        elif use_categories:
            # Unchanged three-way draw, so pools built without full-dataset
            # variants reproduce row for row.
            category = rng.choices(
                ("subset", "full", "negative"),
                weights=(cfg.subset_ratio, cfg.full_set_ratio, cfg.negative_set_ratio),
            )[0]
        else:
            category = "subset"

        negatives: list[int] = []
        if category == "full_dataset":
            selected = list(present)
            negatives = list(absent)
        elif category == "full":
            selected = list(present)
        else:  # subset and negative both take a random proper subset of present
            selected = taxonomy.select_subset(present, rng, weights=weights, strict_subset=True)
            if category == "negative" and absent:
                k = rng.randint(1, min(len(absent), cfg.max_negative_classes))
                negatives = rng.sample(absent, k)
        if not selected and not negatives:
            continue

        requested = selected + negatives
        rng.shuffle(requested)
        req_names = [names[c] for c in requested]
        use_bg = cfg.background_color_prob > 0 and rng.random() < cfg.background_color_prob
        prompt, colors, style, bg_color = diversify.build_legend(
            req_names, rng, background=use_bg, min_delta_e=cfg.color_min_delta_e,
            color_mapping=cfg.color_mapping, n_present=len(present))

        fill = bg_color if (use_bg and bg_color is not None) else diversify.BACKGROUND_COLOR
        rgb = np.empty((*label_arr.shape, 3), dtype=np.uint8)
        rgb[:] = np.array(fill, dtype=np.uint8)
        selected_set = set(selected)
        for i, c in enumerate(requested):
            if c in selected_set:  # negatives are named in the prompt but never painted
                rgb[label_arr == c] = colors[i]

        target_name = f"{basename}_seg{v}.png"
        Image.fromarray(rgb, "RGB").save(out_target / target_name)
        rows.append(_row(pair, input_name, target_name, prompt, {
            "variant": f"v{v}", "style": style, "variant_type": category,
            "selected_classes": [names[c] for c in selected],
            "negative_classes": [names[c] for c in negatives],
            "class_colors": {names[c]: list(colors[i]) for i, c in enumerate(requested) if c in selected_set},
            "negative_class_colors": {names[c]: list(colors[i]) for i, c in enumerate(requested)
                                      if c not in selected_set},
            "background_color": list(bg_color) if (use_bg and bg_color is not None) else None,
        }))
    return rows


def _row(pair: Pair, input_name: str, target_name: str, prompt: str, extra: dict) -> dict:
    return {
        "conditioning_image": f"conditioning_images/{input_name}",
        "image": f"target_images/{target_name}",
        "text": prompt,
        "task": "semantic_segmentation",
        "source_dataset": pair.dataset,
        "id": f"{safe_name(pair.dataset, pair.split, pair.group, pair.stem)}_{extra['variant']}",
        "split": pair.split,
        "metadata": {
            "group": pair.group, "stem": pair.stem,
            "image_path": str(pair.image_path), "mask_path": str(pair.mask_path),
            **extra,
        },
    }


def write_dataset(pairs: Iterable[Pair], out_root: Path, cfg: DiversifyConfig, num_workers: int = 1) -> int:
    pairs = list(pairs)
    out_input, out_target = out_root / "conditioning_images", out_root / "target_images"
    out_input.mkdir(parents=True, exist_ok=True)
    out_target.mkdir(parents=True, exist_ok=True)
    desc = f"Writing variants ({num_workers} workers)" if num_workers > 1 else "Writing variants"
    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            results = list(tqdm(ex.map(write_image_variants, pairs, repeat(out_input), repeat(out_target),
                                       repeat(cfg)), total=len(pairs), desc=desc))
    else:
        results = [write_image_variants(p, out_input, out_target, cfg)
                   for p in tqdm(pairs, total=len(pairs), desc=desc)]
    metadata = [row for rows in results for row in rows]
    with open(out_root / "metadata.jsonl", "w", encoding="utf-8") as f:
        for row in metadata:
            f.write(json.dumps(row) + "\n")
    write_legends_summary(out_root, metadata, cfg.color_mapping)
    return len(metadata)


def write_legends_summary(out_root: Path, metadata: list[dict], color_mapping: str) -> None:
    """Write legends_used.json: every color the prompts named (painted classes,
    negative classes, background), per notation style with usage counts -- the
    effective palette this run actually exposed to the model."""
    full_names = {tuple(p["rgb"]): p["name"] for p in diversify.FULL_PALETTE}
    named_names = {tuple(p["rgb"]): p["name"] for p in diversify.NAMED_PALETTE}
    styles: dict[str, Counter] = defaultdict(Counter)
    vtypes: Counter = Counter()
    max_prompt_colors = 0
    rows_missing_negatives = 0  # rows written before negative_class_colors existed
    n_rows = 0
    for row in metadata:
        m = row["metadata"]
        style = m.get("style")
        if style not in diversify.NOTATION_STYLES:  # canonical-anchor rows etc.
            continue
        colors = list((m.get("class_colors") or {}).values())
        if "negative_class_colors" in m:
            colors += list(m["negative_class_colors"].values())
        elif m.get("negative_classes"):
            rows_missing_negatives += 1
        if m.get("background_color"):
            colors.append(m["background_color"])
        styles[style].update(tuple(c) for c in colors)
        vtypes[m.get("variant_type", "?")] += 1
        max_prompt_colors = max(max_prompt_colors, len(colors))
        n_rows += 1
    record = {
        "color_mapping": color_mapping,
        "rows": n_rows,
        "max_prompt_colors": max_prompt_colors,
        "variant_types": dict(sorted(vtypes.items())),
        "styles": {
            st: {
                "distinct_colors": len(counts),
                "colors": [
                    {"rgb": list(rgb),
                     "name": (named_names if st == "name" else full_names).get(rgb),
                     "count": count}
                    for rgb, count in counts.most_common()
                ],
            }
            for st, counts in sorted(styles.items())
        },
    }
    if rows_missing_negatives:
        record["rows_missing_negative_colors"] = rows_missing_negatives
    (Path(out_root) / "legends_used.json").write_text(json.dumps(record, indent=2) + "\n",
                                                      encoding="utf-8")


def save_diversify_config(out_root: Path, cfg: DiversifyConfig) -> None:
    record = {
        "variants_per_image": cfg.variants_per_image,
        "subset_ratio": cfg.subset_ratio,
        "full_set_ratio": cfg.full_set_ratio,
        "negative_set_ratio": cfg.negative_set_ratio,
        "full_dataset_ratio": cfg.full_dataset_ratio,
        "max_negative_classes": cfg.max_negative_classes,
        "background_color_prob": cfg.background_color_prob,
        "canonical_anchor": cfg.canonical_anchor,
        "min_class_area": cfg.min_class_area,
        "color_min_delta_e": cfg.color_min_delta_e,
        "color_mapping": cfg.color_mapping,
        "seed": cfg.seed,
        "full_palette_size": len(diversify.FULL_PALETTE),
        "named_palette_size": len(diversify.NAMED_PALETTE),
        "notation_styles": list(diversify.NOTATION_STYLES),
    }
    (Path(out_root) / "diversify_config.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                                                          encoding="utf-8")


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    holdout_audit: coco_eval_holdouts.CocoEvaluationHoldoutAudit | None = None
    if args.coconut_xlarge_root and not args.allow_eval_image_overlap:
        holdout_audit = coco_eval_holdouts.CocoEvaluationHoldoutAudit(
            coco_eval_holdouts.load_coco_evaluation_holdouts(Path(args.refcocog_umd_root))
        )
        print(
            "Protecting evaluation images:",
            {name: len(ids) for name, ids in holdout_audit.holdouts.items()},
        )
    all_pairs, skipped = build_all_pairs(args, holdout_audit)
    if not all_pairs:
        raise RuntimeError("No image/mask pairs found. Check dataset roots and --splits.")
    all_pairs = sample_balanced(all_pairs, args.num_images, args.seed)

    if args.dry_run:
        print(f"Resolved {len(all_pairs)} image/mask pairs.")
        if holdout_audit is not None:
            print("Evaluation-image exclusions:", holdout_audit.summary()["excluded_by_reason"])
        counts: dict[tuple, int] = defaultdict(int)
        for p in all_pairs:
            counts[(p.dataset, p.split)] += 1
        for k, c in sorted(counts.items()):
            print(f"  {k[0]} {k[1]}: {c}")
        for p in all_pairs[:10]:
            print(f"  {p.dataset}/{p.split}: {p.image_path} -> {p.mask_path}")
        for ds, s in sorted(skipped.items()):
            if s:
                print(f"Skipped {s} {ds} images (no matching mask/image).")
        return

    ratio_sum = args.subset_ratio + args.full_set_ratio + args.negative_set_ratio + args.full_dataset_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise SystemExit(
            "--subset-ratio + --full-set-ratio + --negative-set-ratio + --full-dataset-ratio must sum to 1.0 "
            f"(got {ratio_sum:.6g})"
        )
    weights = None
    if args.inverse_freq_power > 0:
        weights = compute_weights(all_pairs, args.inverse_freq_power, args.num_workers)

    cfg = DiversifyConfig(
        variants_per_image=args.variants_per_image,
        subset_ratio=args.subset_ratio,
        full_set_ratio=args.full_set_ratio,
        negative_set_ratio=args.negative_set_ratio,
        full_dataset_ratio=args.full_dataset_ratio,
        max_negative_classes=args.max_negative_classes,
        background_color_prob=args.background_color_prob,
        canonical_anchor=args.canonical_anchor,
        min_class_area=args.min_class_area,
        color_min_delta_e=args.color_min_delta_e,
        color_mapping=args.color_mapping,
        seed=args.seed,
        weights=weights,
    )
    out_root.mkdir(parents=True, exist_ok=True)
    if holdout_audit is not None:
        audit_path = holdout_audit.write(out_root)
        print(f"Wrote evaluation-image exclusion audit: {audit_path}")
    save_diversify_config(out_root, cfg)
    written = write_dataset(all_pairs, out_root, cfg, args.num_workers)
    print(f"Saved {written} rows from {len(all_pairs)} images to {out_root}")
    for ds, s in sorted(skipped.items()):
        if s:
            print(f"Skipped {s} {ds} images (no matching mask/image).")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", required=True)
    p.add_argument("--splits", default="train,val", help="Comma-separated splits (default train,val).")
    p.add_argument("--split", default=None, help="Single split; overrides --splits when set.")
    # dataset roots
    p.add_argument("--cityscapes-root", default=None)
    p.add_argument("--cityscapes-splits", default=None,
                   help="Comma-separated Cityscapes splits. Defaults to the global --splits/--split selection.")
    p.add_argument("--synscapes-root", default=None,
                   help="Synscapes root carrying img/{rgb,class} (or its parent).")
    p.add_argument("--synscapes-splits", default="train,val",
                   help="Comma-separated Synscapes splits (default train,val).")
    p.add_argument("--refcocog-umd-root", default=str(coco_eval_holdouts.DEFAULT_REFCOCOG_UMD_ROOT),
                   help="Official RefCOCOg UMD annotation directory containing refs(umd).p; used to prevent held-out val images entering COCONut-XL training.")
    p.add_argument("--allow-eval-image-overlap", action="store_true",
                   help="Disable the COCO val2017 and RefCOCOg UMD validation image holdout. Not suitable for evaluation training runs.")
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
    p.add_argument("--coconut-xlarge-root", default=None,
                   help="Materialized COCONut-XL coco/ root (train2017/, panoptic_train2017/, annotations/panoptic_train2017.json); its merged manifest is emitted as train.")
    p.add_argument("--coconut-xlarge-groups", default="coco,objects365",
                   help="Comma-separated merged COCONut-XL source groups to retain: coco, objects365, or both (default both).")
    p.add_argument("--nuimages-sample-interval-seconds", type=float,
                   default=taxonomy.NUIMAGES_DEFAULT_SAMPLE_INTERVAL_SECONDS,
                   help="Keep the first mask-bearing keyframe in each driving log and then one at least this many seconds later (default: 10). Use 0 to retain all mask-bearing keyframes.")
    p.add_argument("--cityscapes-mask-root", "--mask-root", dest="cityscapes_mask_root", default=None,
                   help="Cityscapes masks split by city. Defaults to <cityscapes-root>/gtFine.")
    p.add_argument("--cityscapes-trainid-suffixes", default="_gtFine_labelTrainIds.png,_labelTrainIds.png")
    p.add_argument("--acdc-root", default=None)
    p.add_argument("--acdc-conditions", default=None, help="Comma-separated ACDC conditions, e.g. fog,night.")
    p.add_argument("--bdd-root", default=None, help="Root like /path/to/BDD/bdd_10k.")
    p.add_argument("--muses-root", default=None, help="Root like /path/to/muses (frame_camera/, gt_semantic/).")
    p.add_argument("--goose-root", default=None, help="Root like /path/to/goose-dataset (images/, labels/, goose_label_mapping.csv).")
    p.add_argument("--vipseg-root", default=None, help="Root like /path/to/VIPSeg (imgs/, panomasks/, VIPSeg_720P/panoVIPSeg_categories.json).")
    p.add_argument("--mapillary-root", default=None,
                   help="Root like /path/to/mapillary-vistas (config_v2.0.json, training/, validation/).")
    p.add_argument("--mapillary-version", default="v2.0",
                   help="Mapillary label subdirectory under each split (default v2.0).")
    p.add_argument("--ade20k-root", default=None, help="Root like /path/to/ade20k (images/, annotations/).")
    p.add_argument("--voc-root", default=None, help="Root like /path/to/VOCSegmentation (images/, sets/, SegmentationClass/).")
    p.add_argument("--voc-mask-dir", default="SegmentationClass",
                   help="Preferred VOC mask dir; falls back to SegmentationClass(Aug).")
    p.add_argument("--pascal-context-root", default=None,
                   help="VOC root carrying pascal_context_59classlabels/ (or that dir itself); "
                        "see scripts/dataset_prep/data_download_scripts/download_datasets_pascal_context.sh.")
    p.add_argument("--pascal-context-image-root", default=None,
                   help="Override the PASCAL-Context image dir (defaults to <voc-root>/images).")
    p.add_argument("--foodseg103-root", default=None,
                   help="Root like /path/to/foodseg103 (images/{train,val}/, labels/{train,val}/).")
    p.add_argument("--trans10k-root", default=None,
                   help="Root like /path/to/trans10k (images/{train,val,test}/, labels/{train,val,test}/); "
                        "see scripts/dataset_prep/data_download_scripts/download_datasets_trans10k.sh.")
    p.add_argument("--railsem19-root", default=None,
                   help="RailSem19 root carrying jpgs/ + uint8/ (or any parent, e.g. the "
                        "release's versions/1/). All 8,500 annotated frames are emitted "
                        "under the train split; RailSem19 defines no val split. See "
                        "scripts/dataset_prep/data_download_scripts/download_datasets_railsem19.sh.")
    p.add_argument("--waymo-root", default=None,
                   help="Root like /path/to/waymo (images/{train,val}/, panoptic/{train,val}/); "
                        "see scripts/dataset_prep/data_download_scripts/download_datasets_waymo.sh.")
    p.add_argument("--hypersim-root", default=None,
                   help="Root like /path/to/hypersim_seg (images/{train,val}/, panoptic/{train,val}/, "
                        "id2label.json); build it from the raw HDF5 release with "
                        "dataset_prep/hypersim/prepare_hypersim_segmentation.py.")
    # sampling
    p.add_argument("--num-images", type=int, default=None, help="Total pairs to sample (balanced across dataset/split/group).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    # diversification knobs
    p.add_argument("--variants-per-image", type=int, default=1, help="Randomized (prompt, mask) variants per image.")
    # Dataset composition by variant category (must sum to 1.0). Each variant is
    # independently assigned a category by these ratios (per-variant random draw).
    p.add_argument("--subset-ratio", type=float, default=1.0,
                   help="Fraction of variants using a random PROPER subset of present classes (default 1.0).")
    p.add_argument("--full-set-ratio", type=float, default=0.0,
                   help="Fraction of variants using ALL present classes (random colors).")
    p.add_argument("--negative-set-ratio", type=float, default=0.0,
                   help="Fraction of variants that add absent classes to the prompt (named, colored, never painted).")
    p.add_argument("--full-dataset-ratio", type=float, default=0.0,
                   help="Fraction of variants listing EVERY class of the source taxonomy: present classes painted, "
                        "all absent classes named but never painted (no --max-negative-classes cap).")
    p.add_argument("--max-negative-classes", type=int, default=30,
                   help="Cap on negatives per negative-set variant: K ~ uniform(1, min(#absent, this)).")
    p.add_argument("--background-color-prob", type=float, default=0.0,
                   help="Prob. a variant paints unselected pixels a random named background color instead of black.")
    p.add_argument("--canonical-anchor", action="store_true",
                   help="Also emit the canonical full 19-class Cityscapes-palette pair + fixed prompt (trainids datasets only).")
    p.add_argument("--inverse-freq-power", type=float, default=1.0,
                   help="Exponent on inverse class frequency for subset sampling (0 = uniform, skips the frequency pass).")
    p.add_argument("--min-class-area", type=float, default=0.0,
                   help="Drop classes covering less than this pixel fraction from sampling (0 = keep all).")
    p.add_argument("--color-min-delta-e", type=float, default=25.0,
                   help="Target minimum CIELAB deltaE between a variant's colors (relaxed if the palette can't satisfy it). "
                        "Only applies to the p200/p100_randomnegdraw color mappings.")
    p.add_argument("--color-mapping", default="p200_randomnegdraw", choices=diversify.COLOR_MAPPING_MODES,
                   help="How legend colors are drawn: p200_randomnegdraw = legacy greedy draw over the full 200 palette "
                        "with silent relaxation; p100_randomnegdraw = the same greedy draw restricted to the 100 "
                        "most-separated colors; best64_maxnegdraw / best32_maxnegdraw = uniform draw from the 64/32 "
                        "most-separated colors, overflow colors drawn 1/rank-weighted farthest-first from the rest; "
                        "bestdrawperimg = per-image pool of the N most-separated colors for the N present classes, "
                        "shared by that image's subset/full/negative variants (grown only if negatives exceed it).")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
