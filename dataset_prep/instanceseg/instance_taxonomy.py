"""Per-source instance decoders + class-name tables for instance-seg dataset prep.

Three sources, three native encodings -> a uniform output: per image, a dict
``{class_id: [bool HxW mask, ...]}`` grouping the instances of each thing class,
plus a ``{class_id: name}`` table for the noun-phrase used in the prompt.

  ADE20K instance  annotations_instance/<split>/<stem>.png : RGB uint8,
                   R = thing class id (1..100, 0=bg), G = instance id within image.
                   Names from instanceInfo100_train.txt (Idx -> Object Name).
  Mapillary v2.0   <split>/v2.0/instances/<stem>.png : uint16, value = class*256 +
                   instance. Keep classes with "instances": true in config_v2.0.json;
                   names from that config's "readable" field.
  Pascal VOC 2012  SegmentationObject/<stem>.png (palette: 0=bg, 255=void, 1..N =
                   instance id) gives instance ids; the class of each instance is the
                   majority SegmentationClass value over its pixels. Names = VOC_CLASSES.
  MUSES            gt_panoptic/<split>.json + RGB panoptic PNGs in COCO panoptic
                   format. Decode segment id as R + 256*G + 256^2*B and keep
                   non-crowd categories with isthing=1.
  ACDC             gt_panoptic/<split>_gt_panoptic.json + RGB panoptic PNGs, the
                   same COCO panoptic format and Cityscapes category ids as MUSES
                   (8 thing classes); images live under rgb_anon/.
  Synscapes        img/instance/<id>.png uses RGB ids R + 256*G + 256^2*B;
                   meta/<id>.json maps each id to one of the 8 Cityscapes thing
                   class ids, with img/class/<id>.png as a defensive fallback.
  IDD              gtFine/<split>/<drive>/*_gtFine_polygons.json. Polygons are
                   rasterized in release order; raw IDD thing labels are classes.
  WildDash 2       COCO-panoptic RGB PNGs + panoptic.json. Shared metadata
                   supplies all non-void labels and its thing subset.
  NuImages         train/val key-camera metadata stores foreground object
                   masks as COCO RLE. Surface RLEs are semantic-only; the ego
                   vehicle is intentionally excluded from instance prompts.
  PACO-LVIS        DatasetNinja/Supervisely export: polygon, rectangle, and
                   zlib-compressed PNG bitmap geometries. Its emitted paired
                   bounding rectangles are suppressed when detailed geometry
                   of the same class has matching bounds.
  GOOSE            labels/<split>/<sequence>/*_labelids.png +
                   *_instanceids.png. Names from goose_label_mapping.csv.
  VIPSeg           panomasks/<video>/*.png. Native value 0 is void; stuff is
                   semantic id 1..124; things are semantic_id*100+instance_id.
                   Names and thing flags from VIPSeg_720P/panoVIPSeg_categories.json.
  OVIS             annotations_{train,valid}.json in YouTube-VOS / COCO-video
                   format. Decode each selected frame's compressed RLE
                   segmentation into one instance mask.
  Waymo            panoptic/{split}/*.png : uint16, value = raw_class_id *
                   WAYMO_PANOPTIC_DIVISOR + instance_id (0 = stuff), written by
                   download_datasets_waymo.sh. Names and thing flags come from the
                   shared semseg taxonomy (deeplab2's WOD-PVPS eval config).
  Hypersim         panoptic/{split}/*.png : the same packed uint16 encoding at
                   HYPERSIM_PANOPTIC_DIVISOR, written by
                   dataset_prep/hypersim/prepare_hypersim_segmentation.py from the
                   release's semantic/semantic_instance HDF5s. NYU40 names and the
                   thing subset come from the shared semseg taxonomy.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import re
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw

# VOC class names + the canonical VOC SegmentationClass reader live in the shared
# semseg taxonomy (index 0 = background).
from common.semseg import taxonomy

Masks = Dict[int, List[np.ndarray]]


# --------------------------------------------------------------------------- #
# ADE20K instance (100 thing classes)
# --------------------------------------------------------------------------- #
def ade_names(info_path: Path) -> Dict[int, str]:
    """Parse instanceInfo100_{train,val}.txt -> {class_id(1..100): name}.

    Tab-separated; row 0 is a header. Field 0 = Idx, field 1 = Object Name (which
    carries trailing padding spaces and may itself contain ", "-separated synonyms;
    we keep the first synonym as the noun phrase)."""
    names: Dict[int, str] = {}
    for line in Path(info_path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) < 2 or not parts[0].strip().isdigit():
            continue
        idx = int(parts[0].strip())
        name = parts[1].strip().split(",")[0].strip()
        if name:
            names[idx] = name
    return names


def read_ade(ann_path: Path) -> Masks:
    arr = np.asarray(Image.open(ann_path).convert("RGB"))
    cls_map, inst_map = arr[..., 0], arr[..., 1]
    out: Masks = {}
    for cls in np.unique(cls_map):
        if cls == 0:
            continue
        cm = cls_map == cls
        for inst in np.unique(inst_map[cm]):
            if inst == 0:
                continue
            out.setdefault(int(cls), []).append(cm & (inst_map == inst))
    return out


# --------------------------------------------------------------------------- #
# Mapillary Vistas v2.0
# --------------------------------------------------------------------------- #
def mapillary_classes(config_path: Path) -> Tuple[Dict[int, str], Set[int]]:
    """-> ({label_index: readable_name}, {instanceable label indices}).

    label_index is the position in config["labels"], which equals value//256 in the
    instance PNGs."""
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    names: Dict[int, str] = {}
    instanceable: Set[int] = set()
    for i, lab in enumerate(cfg["labels"]):
        names[i] = lab.get("readable") or lab["name"].split("--")[-1].replace("-", " ")
        if lab.get("instances"):
            instanceable.add(i)
    return names, instanceable


def read_mapillary(ann_path: Path, instanceable: Set[int]) -> Masks:
    arr = np.asarray(Image.open(ann_path))  # uint16 (mode I;16)
    out: Masks = {}
    for val in np.unique(arr):
        cls = int(val) // 256
        if cls not in instanceable:
            continue
        out.setdefault(cls, []).append(arr == val)
    return out


# --------------------------------------------------------------------------- #
# COCO-panoptic sources: MUSES, ACDC (both Cityscapes thing categories)
# --------------------------------------------------------------------------- #
def coco_panoptic_categories(index_paths: Iterable[Path]) -> Tuple[Dict[int, str], Set[int]]:
    """Read category metadata from COCO-panoptic index JSONs.

    Shared by every panoptic source; only the index PATHS differ per dataset
    (MUSES: gt_panoptic/<split>.json, ACDC: gt_panoptic/<split>_gt_panoptic.json),
    so callers pass them in. Category ids follow Cityscapes label ids (e.g.
    car=26), not trainIds. Only categories marked isthing are eligible as
    instance noun phrases.
    """
    names: Dict[int, str] = {}
    instanceable: Set[int] = set()
    for json_path in index_paths:
        json_path = Path(json_path)
        if not json_path.exists():
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for category in data.get("categories", []):
            cid = int(category["id"])
            names[cid] = str(category.get("name") or cid)
            if int(category.get("isthing", 0)) == 1:
                instanceable.add(cid)
    return names, instanceable


def muses_categories(root: Path, splits: Iterable[str]) -> Tuple[Dict[int, str], Set[int]]:
    return coco_panoptic_categories(
        Path(root) / "gt_panoptic" / f"{split}.json" for split in splits)


def acdc_categories(root: Path, splits: Iterable[str]) -> Tuple[Dict[int, str], Set[int]]:
    return coco_panoptic_categories(
        Path(root) / "gt_panoptic" / f"{split}_gt_panoptic.json" for split in splits)


def rgb_to_panoptic_id(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.int64)
    return arr[..., 0] + 256 * arr[..., 1] + 256 * 256 * arr[..., 2]


def read_coco_panoptic(ann_path: Path, segments_info: Iterable[dict], instanceable: Set[int]) -> Masks:
    id_map = rgb_to_panoptic_id(np.asarray(Image.open(ann_path).convert("RGB")))
    out: Masks = {}
    for segment in segments_info:
        cls = int(segment.get("category_id", -1))
        if cls not in instanceable:
            continue
        if int(segment.get("iscrowd", 0)) != 0:
            continue
        sid = int(segment["id"])
        mask = id_map == sid
        if mask.any():
            out.setdefault(cls, []).append(mask)
    return out


# --------------------------------------------------------------------------- #
# Synscapes RGB instance IDs + Cityscapes raw class IDs
# --------------------------------------------------------------------------- #
def synscapes_classes() -> Tuple[Dict[int, str], Set[int]]:
    names = {
        24: "person",
        25: "rider",
        26: "car",
        27: "truck",
        28: "bus",
        31: "train",
        32: "motorcyclist",
        33: "bicyclist",
    }
    return names, set(names)


def read_synscapes(instance_path: Path, class_path: Path,
                   metadata_path: Path | None, instanceable: Set[int]) -> Masks:
    """Decode Synscapes RGB instance ids, assigning classes from per-frame metadata."""
    instance_ids = rgb_to_panoptic_id(
        np.asarray(Image.open(instance_path).convert("RGB")))
    class_ids = np.asarray(Image.open(class_path))
    if class_ids.ndim == 3:
        if not (np.array_equal(class_ids[..., 0], class_ids[..., 1])
                and np.array_equal(class_ids[..., 0], class_ids[..., 2])):
            raise ValueError(f"Expected a grayscale Synscapes class mask: {class_path}")
        class_ids = class_ids[..., 0]
    if instance_ids.shape != class_ids.shape:
        raise ValueError(
            f"Synscapes instance/class dimensions differ: {instance_path} "
            f"{instance_ids.shape} vs {class_path} {class_ids.shape}")

    metadata_classes: Dict[int, int] = {}
    if metadata_path is not None and Path(metadata_path).is_file():
        data = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        raw_classes = data.get("instance", {}).get("class", {})
        if isinstance(raw_classes, dict):
            for instance_id, class_id in raw_classes.items():
                try:
                    metadata_classes[int(instance_id)] = int(class_id)
                except (TypeError, ValueError):
                    continue

    out: Masks = {}
    for instance_id in np.unique(instance_ids):
        instance_id = int(instance_id)
        if instance_id == 0:
            continue
        mask = instance_ids == instance_id
        class_id = metadata_classes.get(instance_id)
        if class_id not in instanceable:
            values, counts = np.unique(class_ids[mask], return_counts=True)
            candidates = [
                (int(value), int(count))
                for value, count in zip(values, counts)
                if int(value) in instanceable
            ]
            class_id = max(candidates, key=lambda item: (item[1], -item[0]))[0] if candidates else None
        if class_id in instanceable:
            out.setdefault(class_id, []).append(mask)
    return out


# --------------------------------------------------------------------------- #
# IDD polygon instances
# --------------------------------------------------------------------------- #
def idd_classes() -> Tuple[Dict[int, str], Set[int]]:
    """Raw-ID thing labels from AutoNUE's official IDD taxonomy.

    These are the classes for which the release declares ``hasInstances``. We
    retain the raw ID rather than level3Id so, for example, ``person`` and
    ``animal`` stay separately nameable even though they share a level-3 class.
    """
    raw_ids = set(taxonomy.IDD_THING_RAW_IDS)
    names = {
        raw_id: name
        for name, raw_id in taxonomy.IDD_RAW_LABEL_IDS.items()
        if raw_id in raw_ids
    }
    return names, set(names)


def read_idd(ann_path: Path, instanceable: Set[int]) -> Masks:
    return taxonomy.read_idd_instance_masks(ann_path, instanceable)


def wilddash2_categories(root: Path) -> Tuple[Dict[int, str], Set[int]]:
    """Return WildDash 2 raw category names and eligible thing IDs."""
    names, instanceable = taxonomy.wilddash2_instance_categories(
        taxonomy.resolve_wilddash2_root(root))
    return names, instanceable



def coconut_xlarge_categories(root: Path) -> Tuple[Dict[int, str], Set[int]]:
    """Return COCONut-XL COCO-panoptic names and eligible thing IDs."""
    return taxonomy.coconut_xlarge_instance_categories(
        taxonomy.resolve_coconut_xlarge_root(root))

def nuimages_categories(root: Path) -> Tuple[Dict[int, str], Set[int], Dict[str, int]]:
    """Return compact NuImages names, thing IDs, and metadata-token mapping."""
    return taxonomy.nuimages_instance_categories(taxonomy.resolve_nuimages_root(root))


def read_nuimages(annotations: Iterable[dict], token_to_index: Dict[str, int],
                  instanceable: Set[int]) -> Masks:
    """Decode foreground NuImages RLE masks into per-class instance lists."""
    out: Masks = {}
    for annotation in annotations:
        if annotation.get("_nuimages_annotation_type") != "object":
            continue
        class_id = token_to_index.get(str(annotation.get("category_token") or ""))
        if class_id not in instanceable or annotation.get("mask") is None:
            continue
        mask = taxonomy.decode_coco_rle(annotation["mask"])
        if mask.any():
            out.setdefault(class_id, []).append(mask)
    return out


# --------------------------------------------------------------------------- #
# PACO-LVIS (Supervisely export)
# --------------------------------------------------------------------------- #
PACO_LVIS_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


def resolve_paco_lvis_root(root: Path) -> Path:
    """Resolve a PACO-LVIS DatasetNinja/Supervisely export tree."""
    root = Path(root).expanduser().resolve()
    for candidate in (root, root / "paco_lvis", root / "paco-lvis-DatasetNinja"):
        if ((candidate / "meta.json").is_file()
                and any((candidate / f"{split}2017" / "ann").is_dir()
                        for split in PACO_LVIS_SPLIT_ALIASES.values())):
            return candidate
    raise FileNotFoundError(
        "Could not resolve PACO-LVIS root. Expected meta.json plus at least one "
        "{train,val,test}2017/ann/ directory under "
        f"{root} or a paco_lvis/ child."
    )


def normalize_paco_lvis_splits(splits: Iterable[str]) -> list[str]:
    selected: list[str] = []
    for split in splits:
        normalized = PACO_LVIS_SPLIT_ALIASES.get(str(split).strip().lower())
        if normalized is None:
            raise ValueError("PACO-LVIS splits must be train, val, and/or test")
        if normalized not in selected:
            selected.append(normalized)
    if not selected:
        raise ValueError("PACO-LVIS split selection must not be empty")
    return selected


def _clean_paco_lvis_title(title: str) -> str:
    """Turn DatasetNinja's LVIS disambiguators into prompt-ready nouns."""
    return " ".join(re.sub(r"_\([^)]*\)$", "", str(title)).replace("_", " ").split())


def paco_lvis_classes(root: Path) -> Tuple[Dict[int, str], Set[int]]:
    """Read all PACO object and object-part classes from Supervisely metadata."""
    meta = json.loads((resolve_paco_lvis_root(root) / "meta.json").read_text(encoding="utf-8"))
    names: Dict[int, str] = {}
    for category in meta.get("classes", []):
        try:
            class_id = int(category["id"])
        except (KeyError, TypeError, ValueError):
            continue
        title = _clean_paco_lvis_title(category.get("title") or "")
        if title:
            names[class_id] = title
    if not names:
        raise ValueError("PACO-LVIS meta.json contains no valid classes")
    return names, set(names)


def _paco_polygon_mask(points: dict, height: int, width: int) -> np.ndarray:
    mask = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    exterior = points.get("exterior") or []
    if len(exterior) >= 3:
        draw.polygon(exterior, fill=1)
    for interior in points.get("interior") or []:
        if len(interior) >= 3:
            draw.polygon(interior, fill=0)
    return np.asarray(mask, dtype=bool)


def _paco_rectangle_mask(points: dict, height: int, width: int) -> np.ndarray:
    corners = points.get("exterior") or []
    if len(corners) < 2:
        return np.zeros((height, width), dtype=bool)
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    mask = Image.new("1", (width, height), 0)
    ImageDraw.Draw(mask).rectangle((min(xs), min(ys), max(xs), max(ys)), fill=1)
    return np.asarray(mask, dtype=bool)


def _paco_bitmap_mask(bitmap: dict, height: int, width: int) -> np.ndarray:
    try:
        compressed = base64.b64decode(str(bitmap["data"]))
        payload = zlib.decompress(compressed)
        origin_x, origin_y = (int(value) for value in bitmap["origin"])
    except (KeyError, TypeError, ValueError, zlib.error) as exc:
        raise ValueError("Invalid PACO-LVIS Supervisely bitmap") from exc
    with Image.open(io.BytesIO(payload)) as image:
        crop = np.asarray(image.convert("1"), dtype=bool)
    full = np.zeros((height, width), dtype=bool)
    y0, x0 = max(origin_y, 0), max(origin_x, 0)
    y1, x1 = min(origin_y + crop.shape[0], height), min(origin_x + crop.shape[1], width)
    if y0 < y1 and x0 < x1:
        crop_y0, crop_x0 = y0 - origin_y, x0 - origin_x
        full[y0:y1, x0:x1] = crop[crop_y0:crop_y0 + y1 - y0, crop_x0:crop_x0 + x1 - x0]
    return full


def _paco_object_mask(object_data: dict, height: int, width: int) -> tuple[str, np.ndarray]:
    geometry_type = str(object_data.get("geometryType") or "")
    if geometry_type == "polygon":
        return geometry_type, _paco_polygon_mask(object_data.get("points") or {}, height, width)
    if geometry_type == "rectangle":
        return geometry_type, _paco_rectangle_mask(object_data.get("points") or {}, height, width)
    if geometry_type == "bitmap":
        return geometry_type, _paco_bitmap_mask(object_data.get("bitmap") or {}, height, width)
    raise ValueError(f"Unsupported PACO-LVIS Supervisely geometry: {geometry_type or '<missing>'}")


def _paco_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _paco_duplicate_rectangle(rectangle: np.ndarray, detailed_masks: Iterable[np.ndarray]) -> bool:
    rect_bounds = _paco_bounds(rectangle)
    if rect_bounds is None:
        return True
    for mask in detailed_masks:
        bounds = _paco_bounds(mask)
        if bounds is not None and all(abs(a - b) <= 1 for a, b in zip(rect_bounds, bounds)):
            return True
    return False


def read_paco_lvis(ann_path: Path, instanceable: Set[int]) -> Masks:
    """Decode PACO's Supervisely masks, omitting paired bbox-only duplicates.

    DatasetNinja exports each detailed polygon/bitmap together with a rectangle
    carrying the same bounding box. Detailed masks are retained; a rectangle is
    kept only when no same-class detailed mask has the same bounds.
    """
    data = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    try:
        height = int(data["size"]["height"])
        width = int(data["size"]["width"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid PACO-LVIS annotation size: {ann_path}") from exc

    detailed: Dict[int, List[np.ndarray]] = {}
    rectangles: Dict[int, List[np.ndarray]] = {}
    for object_data in data.get("objects", []):
        try:
            class_id = int(object_data.get("classId"))
        except (TypeError, ValueError):
            continue
        if class_id not in instanceable:
            continue
        geometry_type, mask = _paco_object_mask(object_data, height, width)
        if not mask.any():
            continue
        (rectangles if geometry_type == "rectangle" else detailed).setdefault(class_id, []).append(mask)

    out: Masks = {class_id: list(masks) for class_id, masks in detailed.items()}
    for class_id, masks in rectangles.items():
        for mask in masks:
            if not _paco_duplicate_rectangle(mask, detailed.get(class_id, ())):
                out.setdefault(class_id, []).append(mask)
    return out


# --------------------------------------------------------------------------- #
# GOOSE 2D instance masks
# --------------------------------------------------------------------------- #
def _norm_key(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _clean_name(value: str) -> str:
    return " ".join(str(value).strip().replace("_", " ").replace("-", " ").split())


def _row_value(row: dict[str, str], candidates: Iterable[str]) -> str:
    normalized = {_norm_key(k): v for k, v in row.items()}
    for key in candidates:
        value = normalized.get(_norm_key(key))
        if value is not None:
            return value
    return ""


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _goose_instanceable_from_csv(root: Path, raw_to_index: Dict[int, int]) -> Set[int]:
    mapping_path = Path(root) / "goose_label_mapping.csv"
    text = mapping_path.read_text(encoding="utf-8", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    instanceable: Set[int] = set()
    saw_flag = False
    for row in csv.DictReader(text.splitlines(), dialect=dialect):
        flag = _row_value(row, ("has_instance", "hasinstance", "instances", "isthing"))
        if flag == "":
            continue
        saw_flag = True
        raw_id = _row_value(row, ("id", "label_key", "labelkey", "label_id", "labelid", "class_id", "classid"))
        try:
            canonical = raw_to_index.get(int(raw_id))
        except ValueError:
            canonical = None
        if canonical is not None and _truthy(flag):
            instanceable.add(canonical)
    return instanceable if saw_flag else set(raw_to_index.values())


def goose_names(root: Path) -> Tuple[Dict[int, str], Set[int], Dict[int, int]]:
    """Return canonical GOOSE class ids, names, and raw-label -> canonical map.

    ``has_instance`` in goose_label_mapping.csv marks classes eligible for
    instance segmentation. If the column is absent in a future metadata version,
    every non-void semantic class is treated as eligible and read_goose() still
    only emits classes that actually have non-zero instance ids in an image.
    """
    names, raw_to_index, _raw_to_name = taxonomy.register_goose_mapping(root)
    canonical_names = {idx: name for idx, name in enumerate(names)}
    instanceable = _goose_instanceable_from_csv(root, raw_to_index)
    return canonical_names, instanceable, raw_to_index


def read_goose(label_path: Path, instance_path: Path, raw_to_index: Dict[int, int]) -> Masks:
    label = np.asarray(Image.open(label_path))
    if label.ndim == 3:
        label = label[..., 0]
    inst = np.asarray(Image.open(instance_path))
    if inst.ndim == 3:
        inst = inst[..., 0]
    out: Masks = {}
    for raw_cls in np.unique(label):
        cls = raw_to_index.get(int(raw_cls))
        if cls is None:
            continue
        cls_mask = label == raw_cls
        for inst_id in np.unique(inst[cls_mask]):
            if int(inst_id) == 0:
                continue
            mask = cls_mask & (inst == inst_id)
            if mask.any():
                out.setdefault(cls, []).append(mask)
    return out


# --------------------------------------------------------------------------- #
# VIPSeg native panomasks
# --------------------------------------------------------------------------- #
def vipseg_categories(root: Path) -> Tuple[Dict[int, str], Set[int]]:
    """Read zero-based VIPSeg categories and return instance-eligible ids."""
    categories_path = Path(root) / "VIPSeg_720P" / "panoVIPSeg_categories.json"
    if not categories_path.exists():
        categories_path = Path(root) / "panoVIPSeg_categories.json"
    categories = json.loads(categories_path.read_text(encoding="utf-8"))
    names: Dict[int, str] = {}
    instanceable: Set[int] = set()
    for category in categories:
        cid = int(category["id"])
        names[cid] = _clean_name(str(category.get("name") or cid))
        if int(category.get("isthing", 0)) == 1:
            instanceable.add(cid)
    return names, instanceable


def read_vipseg(ann_path: Path, instanceable: Set[int]) -> Masks:
    arr = np.asarray(Image.open(ann_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.int64)
    out: Masks = {}
    for val in np.unique(arr):
        raw = int(val)
        if raw <= 124:
            continue
        cls = raw // 100 - 1
        if cls not in instanceable:
            continue
        mask = arr == raw
        if mask.any():
            out.setdefault(cls, []).append(mask)
    return out


# --------------------------------------------------------------------------- #
# Waymo Open Dataset camera panoptic
# --------------------------------------------------------------------------- #
def waymo_classes(root: Path) -> Tuple[Dict[int, str], Set[int]]:
    """Waymo raw class ids -> name, plus the instance-eligible ids.

    Keyed by RAW CameraSegmentation.Type id (0..28) so it lines up with the
    semantic half of the panoptic pixels. ``things.json`` beside the labels wins
    when present, so a rebuild that observed a different thing set is honored.
    """
    taxonomy.verify_waymo_labels(root)
    names = {i: name for i, name in enumerate(taxonomy.WAYMO_RAW_CLASS_NAMES)}
    instanceable = set(taxonomy.WAYMO_THING_RAW_IDS)
    things_path = Path(root) / "things.json"
    if things_path.exists():
        observed = json.loads(things_path.read_text(encoding="utf-8")).get("thing_ids")
        if observed:
            instanceable = {int(i) for i in observed}
    return names, instanceable


def read_waymo(ann_path: Path, instanceable: Set[int]) -> Masks:
    arr = np.asarray(Image.open(ann_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.int64)
    divisor = taxonomy.WAYMO_PANOPTIC_DIVISOR
    out: Masks = {}
    for val in np.unique(arr):
        raw = int(val)
        cls, instance = divmod(raw, divisor)
        if instance == 0 or cls not in instanceable:  # instance 0 = stuff
            continue
        mask = arr == raw
        if mask.any():
            out.setdefault(cls, []).append(mask)
    return out


# --------------------------------------------------------------------------- #
# Hypersim (NYU40 indoor scenes)
# --------------------------------------------------------------------------- #
def hypersim_classes(root: Path) -> Tuple[Dict[int, str], Set[int]]:
    """Hypersim raw NYU40 ids -> name, plus the instance-eligible ids.

    Keyed by RAW id (1..40) so it lines up with the semantic half of the packed
    panoptic pixels. Names are the prompt-facing spellings (HYPERSIM_CLASSES, which
    respaces the five run-together NYU40 names); the raw CSV spellings are what
    verify_hypersim_labels checks the pool against. ``things.json`` beside the
    labels wins when present, so a build that observed a different thing set is
    honored -- same override hook as Waymo.
    """
    taxonomy.verify_hypersim_labels(root)
    names = {i + taxonomy.HYPERSIM_RAW_OFFSET: name
             for i, name in enumerate(taxonomy.HYPERSIM_CLASSES)}
    instanceable = set(taxonomy.HYPERSIM_THING_RAW_IDS)
    things_path = Path(root) / "things.json"
    if things_path.exists():
        observed = json.loads(things_path.read_text(encoding="utf-8")).get("thing_ids")
        if observed:
            instanceable = {int(i) for i in observed}
    return names, instanceable


def read_hypersim(ann_path: Path, instanceable: Set[int]) -> Masks:
    """Packed uint16 panoptic PNG -> per-class instance masks.

    Identical encoding to Waymo (class * divisor + instance, instance 0 = stuff),
    only the divisor constant differs, so the decode is the same walk over unique
    packed values.
    """
    arr = np.asarray(Image.open(ann_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.int64)
    divisor = taxonomy.HYPERSIM_PANOPTIC_DIVISOR
    out: Masks = {}
    for val in np.unique(arr):
        raw = int(val)
        cls, instance = divmod(raw, divisor)
        if instance == 0 or cls not in instanceable:  # instance 0 = stuff, 0 = unlabeled
            continue
        mask = arr == raw
        if mask.any():
            out.setdefault(cls, []).append(mask)
    return out


# --------------------------------------------------------------------------- #
# OVIS / YouTube-VOS-style compressed RLE
# --------------------------------------------------------------------------- #
def ovis_categories(annotation_paths: Iterable[Path]) -> Tuple[Dict[int, str], Set[int]]:
    """Read OVIS categories from one or more annotations_{train,valid}.json files."""
    names: Dict[int, str] = {}
    for path in annotation_paths:
        if not Path(path).exists():
            continue
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for category in data.get("categories", []):
            cid = int(category["id"])
            name = _clean_name(str(category.get("name") or cid))
            names[cid] = "Vehicle" if name.lower() == "vehical" else name
    return names, set(names)


def _decode_compressed_rle_counts(counts: str) -> List[int]:
    """Decode COCO's compact ASCII RLE counts string.

    This mirrors the compressed RLE used by pycocotools/youtubevos. The decoded
    counts alternate background and foreground runs in column-major order.
    """
    out: List[int] = []
    p = 0
    while p < len(counts):
        x = 0
        k = 0
        more = True
        while more:
            c = ord(counts[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = (c & 0x20) != 0
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(out) > 2:
            x += out[-2]
        out.append(int(x))
    return out


def _rle_to_mask(segm: dict[str, Any], height: int, width: int) -> np.ndarray:
    counts_raw = segm["counts"]
    counts = _decode_compressed_rle_counts(counts_raw) if isinstance(counts_raw, str) else list(counts_raw)
    total = height * width
    flat = np.zeros(total, dtype=np.bool_)
    pos = 0
    value = 0
    for run in counts:
        run = int(run)
        end = min(pos + run, total)
        if value == 1 and end > pos:
            flat[pos:end] = True
        pos = end
        value = 1 - value
        if pos >= total:
            break
    return flat.reshape((height, width), order="F")


def _polygons_to_mask(polygons: list, height: int, width: int) -> np.ndarray:
    img = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(img)
    for poly in polygons:
        if len(poly) >= 6:
            coords = list(zip(poly[0::2], poly[1::2]))
            draw.polygon(coords, outline=1, fill=1)
    return np.asarray(img, dtype=np.bool_)


def decode_ovis_segmentation(segm: Any, height: int, width: int) -> np.ndarray:
    """Decode one frame's OVIS segmentation entry into a boolean mask."""
    if segm is None:
        return np.zeros((height, width), dtype=np.bool_)
    if isinstance(segm, list):
        return _polygons_to_mask(segm, height, width)
    if isinstance(segm, dict):
        size = segm.get("size") or [height, width]
        return _rle_to_mask(segm, int(size[0]), int(size[1]))
    raise TypeError(f"unsupported OVIS segmentation type: {type(segm).__name__}")


def read_ovis_frame(
    frame_annotations: Iterable[dict[str, Any]],
    frame_index: int,
    height: int,
    width: int,
    instanceable: Set[int],
) -> Masks:
    out: Masks = {}
    for ann in frame_annotations:
        cls = int(ann.get("category_id", -1))
        if cls not in instanceable:
            continue
        if int(ann.get("iscrowd", 0)) != 0:
            continue
        segmentations = ann.get("segmentations") or []
        if frame_index >= len(segmentations) or segmentations[frame_index] is None:
            continue
        mask = decode_ovis_segmentation(segmentations[frame_index], height, width)
        if mask.any():
            out.setdefault(cls, []).append(mask)
    return out


# --------------------------------------------------------------------------- #
# Pascal VOC 2012 instance
# --------------------------------------------------------------------------- #
def voc_names() -> Dict[int, str]:
    """{class_id(1..20): name} from VOC_CLASSES (index 0 = background, excluded)."""
    return {i: n for i, n in enumerate(taxonomy.VOC_CLASSES) if i != 0}


def read_voc(obj_path: Path, cls_path: Path) -> Masks:
    obj = np.asarray(Image.open(obj_path))          # palette idx: 0 bg, 255 void, 1..N inst
    cls_arr = taxonomy.read_label_array(cls_path, "voc")  # 0..20, 255 ignore
    out: Masks = {}
    for iid in np.unique(obj):
        if iid in (0, 255):
            continue
        m = obj == iid
        vals, counts = np.unique(cls_arr[m], return_counts=True)
        # the instance's class = most common non-ignore, non-background class label
        best_cls, best_cnt = 0, 0
        for v, c in zip(vals.tolist(), counts.tolist()):
            if v in (0, taxonomy.IGNORE_INDEX):
                continue
            if c > best_cnt:
                best_cls, best_cnt = int(v), c
        if best_cls > 0:
            out.setdefault(best_cls, []).append(m)
    return out
