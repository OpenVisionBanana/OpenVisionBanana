#!/usr/bin/env python3
"""Per-dataset semseg taxonomies + mask readers, shared by prep and eval.

Each supported dataset is read into a canonical, 0-based class-index array with a
single ignore value (255). The diversification logic (diversify.py) only ever sees
"class indices present in this image" + the matching class-name list, so it works
identically across Cityscapes/ACDC/BDD/Synscapes (19), IDD level-3 (26), ADE20k (150), VOC (21), PASCAL-Context
(60), FoodSeg103 (104), Trans10K-v2 (12), VIPSeg (124), and Mapillary Vistas v2.0
(124 raw labels) without any dataset-specific special-casing downstream.

  read_label_array(path, dataset) -> HxW int array, class idx in [0, n_classes),
                                     ignore/void/background-as-ignore = 255.
  present_classes(arr, dataset)   -> sorted selectable foreground indices present.
  class_names(dataset)            -> tuple of names indexed by class index.
  select_subset(present, rng, ...)-> the M-rule subset (N==1 -> [that one];
                                     N>1 -> a strict proper subset, 1..N-1).
"""
from __future__ import annotations

import csv
import base64
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

IGNORE_INDEX = 255

# --- class names (0-based, in canonical index order) ------------------------- #
CITYSCAPES_CLASSES = (
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
)

# IDD's official level3Id taxonomy. The polygon JSON stores raw labels; the
# reader below rasterizes and folds them to these compact 0..25 IDs.
IDD_LEVEL3_CLASSES = (
    "road", "parking", "sidewalk", "rail track", "person", "rider",
    "motorcycle", "bicycle", "autorickshaw", "car", "truck", "bus",
    "large vehicle", "curb", "wall", "fence", "guard rail", "billboard",
    "traffic sign", "traffic light", "pole", "obstacle", "building",
    "bridge or tunnel", "vegetation", "sky",
)

# Synscapes ships one numbered sequence instead of physical split directories.
SYNSCAPES_FIRST_INDEX = 1
SYNSCAPES_TRAIN_LAST_INDEX = 24_000
SYNSCAPES_LAST_INDEX = 25_000
SYNSCAPES_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
}

# IDD raw IDs and level-3 IDs from AutoNUE helpers/anue_labels.py. The 26
# level-3 IDs are already compact and zero based.
IDD_RAW_ID_TO_LEVEL3_ID = {
    0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4, 8: 5, 9: 6,
    10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 15: 12, 16: 12, 17: 12,
    18: 12, 19: 13, 20: 14, 21: 15, 22: 16, 23: 17, 24: 18, 25: 19,
    26: 20, 27: 20, 28: 21, 29: 22, 30: 23, 31: 23, 32: 24, 33: 25,
    34: 25, 35: IGNORE_INDEX, 36: IGNORE_INDEX, 37: IGNORE_INDEX,
    38: IGNORE_INDEX, 39: IGNORE_INDEX,
}

IDD_RAW_LABEL_IDS = {
    "road": 0,
    "parking": 1,
    "drivable fallback": 2,
    "sidewalk": 3,
    "rail track": 4,
    "non-drivable fallback": 5,
    "person": 6,
    "animal": 7,
    "rider": 8,
    "motorcycle": 9,
    "bicycle": 10,
    "autorickshaw": 11,
    "car": 12,
    "truck": 13,
    "bus": 14,
    "caravan": 15,
    "trailer": 16,
    "train": 17,
    "vehicle fallback": 18,
    "curb": 19,
    "wall": 20,
    "fence": 21,
    "guard rail": 22,
    "billboard": 23,
    "traffic sign": 24,
    "traffic light": 25,
    "pole": 26,
    "polegroup": 27,
    "obs-str-bar-fallback": 28,
    "building": 29,
    "bridge": 30,
    "tunnel": 31,
    "vegetation": 32,
    "sky": 33,
    "fallback background": 34,
    "unlabeled": 35,
    "ego vehicle": 36,
    "rectification border": 37,
    "out of roi": 38,
    "license plate": 39,
}

IDD_THING_RAW_IDS = frozenset(range(6, 19))
IDD_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}

WILDDASH2_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
}

# NuImages has released masks only for its train and validation key-camera
# samples. ``sample_data`` also contains temporal sweeps, but they are not
# annotated and must never be added to either segmentation pool.
NUIMAGES_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
}
NUIMAGES_DEFAULT_SAMPLE_INTERVAL_SECONDS = 10.0
NUIMAGES_SURFACE_CATEGORY_NAMES = frozenset({"flat.driveable_surface"})
NUIMAGES_EGO_CATEGORY_NAME = "vehicle.ego"
NUIMAGES_IGNORE_CATEGORY_NAMES = frozenset({"static_object.bicycle_rack"})


@dataclass(frozen=True)
class IDDPolygonObject:
    raw_id: int
    polygon: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class NuImagesFrame:
    """One mask-bearing NuImages key-camera frame selected for preparation."""

    split: str
    token: str
    log_token: str
    timestamp: int
    image_path: Path
    annotations: tuple[dict, ...]  # surface RLEs first, then object RLEs


# The canonical Cityscapes 19-class colors (train-id order). Used by the optional
# canonical-anchor pair in prep and the eval "fixed" prompt style.
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
    [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],
], dtype=np.uint8)

# Raw ``labelIds`` used by Cityscapes and Synscapes -> canonical train ids.
CITYSCAPES_RAW_ID_TO_TRAIN_ID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
    22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
    31: 16, 32: 17, 33: 18,
}

# Cityscapes classes that have instance identities. Synscapes uses the same raw
# class-id convention for its semantic and instance metadata.
CITYSCAPES_THING_RAW_IDS = frozenset({24, 25, 26, 27, 28, 31, 32, 33})

# ADE20k 150 (mmseg order). Index i here == raw mask value (i + 1); raw 0 = ignore.
ADE20K_CLASSES = (
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door",
    "table", "mountain", "plant", "curtain", "chair", "car", "water", "painting",
    "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat",
    "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion",
    "base", "box", "column", "signboard", "chest of drawers", "counter", "sand",
    "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path",
    "stairs", "runway", "case", "pool table", "pillow", "screen door", "stairway",
    "river", "bridge", "bookcase", "blind", "coffee table", "toilet", "flower",
    "book", "hill", "bench", "countertop", "stove", "palm", "kitchen island",
    "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus",
    "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight",
    "booth", "television receiver", "airplane", "dirt track", "apparel", "pole",
    "land", "bannister", "escalator", "ottoman", "bottle", "buffet", "poster",
    "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer",
    "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall", "tent",
    "bag", "minibike", "cradle", "oven", "ball", "food", "step", "tank",
    "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light",
    "tray", "ashcan", "fan", "pier", "crt screen", "plate", "monitor",
    "bulletin board", "shower", "radiator", "glass", "clock", "flag",
)

# Pascal VOC 21 (index 0 == background; mask palette index == class index).
VOC_CLASSES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)

# FoodSeg103 ingredient labels. Raw mask value == class index; 0 is background.
FOODSEG103_CLASSES = (
    "background", "candy", "egg tart", "french fries", "chocolate", "biscuit",
    "popcorn", "pudding", "ice cream", "cheese butter", "cake", "wine",
    "milkshake", "coffee", "juice", "milk", "tea", "almond", "red beans",
    "cashew", "dried cranberries", "soy", "walnut", "peanut", "egg", "apple",
    "date", "apricot", "avocado", "banana", "strawberry", "cherry",
    "blueberry", "raspberry", "mango", "olives", "peach", "lemon", "pear",
    "fig", "pineapple", "grape", "kiwi", "melon", "orange", "watermelon",
    "steak", "pork", "chicken duck", "sausage", "fried meat", "lamb", "sauce",
    "crab", "fish", "shellfish", "shrimp", "soup", "bread", "corn", "hamburg",
    "pizza", "hanamaki baozi", "wonton dumplings", "pasta", "noodles", "rice",
    "pie", "tofu", "eggplant", "potato", "garlic", "cauliflower", "tomato",
    "kelp", "seaweed", "spring onion", "rape", "ginger", "okra", "lettuce",
    "pumpkin", "cucumber", "white radish", "carrot", "asparagus",
    "bamboo shoots", "broccoli", "celery stick", "cilantro mint", "snow peas",
    "cabbage", "bean sprouts", "onion", "pepper", "green beans",
    "French beans", "king oyster mushroom", "shiitake", "enoki mushroom",
    "oyster mushroom", "white button mushroom", "salad", "other ingredients",
)


# PASCAL-Context 59 (+ background at index 0), in mmsegmentation's METAINFO order.
# Raw mask value == class index; masks are built by
# scripts/dataset_downloads/download_datasets_pascal_context.sh, which folds every
# class outside these 59 into background, per the standard 60-way protocol.
PASCAL_CONTEXT_CLASSES = (
    "background", "aeroplane", "bag", "bed", "bedclothes", "bench", "bicycle", "bird",
    "boat", "book", "bottle", "building", "bus", "cabinet", "car", "cat", "ceiling",
    "chair", "cloth", "computer", "cow", "cup", "curtain", "dog", "door", "fence",
    "floor", "flower", "food", "grass", "ground", "horse", "keyboard", "light",
    "motorbike", "mountain", "mouse", "person", "plate", "platform", "pottedplant",
    "road", "rock", "sheep", "shelves", "sidewalk", "sign", "sky", "snow", "sofa",
    "table", "track", "train", "tree", "truck", "tvmonitor", "wall", "water", "window",
    "wood",
)


# Waymo Open Dataset camera (2D panoptic video) segmentation.
#
# Raw ids 0..28 are CameraSegmentation.Type in dataset.proto, and the names below
# are deeplab2's WOD-PVPS eval config (data/waymo_constants.py), which is the
# reference protocol for this benchmark.
#
# Raw 0 ("unknown") is deeplab2's IGNORE_LABEL. Raw 1 ("sdc", the Waymo car's own
# hood at the bottom of every frame) is read as ignore too, matching how GOOSE
# treats ego-vehicle pixels here -- otherwise the same region would be a target
# class in one pool member and ignored in another.
WAYMO_RAW_CLASS_NAMES = (
    "unknown", "sdc", "car", "truck", "bus", "other large vehicle", "bicycle",
    "motorcycle", "trailer", "person", "cyclist", "motorcyclist", "bird",
    "ground animal", "construction cone or pole", "pole", "pedestrian object",
    "sign", "traffic light", "building", "road", "lane marker", "road marker",
    "sidewalk", "vegetation", "sky", "ground", "dynamic", "static",
)
# Two upstream names are reworded for prompt readability; everything else is the
# upstream name with underscores turned into spaces:
#   construction_cone_pole  -> "construction cone or pole"
#   other_pedestrian_object -> "pedestrian object"

WAYMO_IGNORE_RAW_IDS = (0, 1)  # unknown, sdc/ego-vehicle
WAYMO_RAW_OFFSET = len(WAYMO_IGNORE_RAW_IDS)  # canonical index = raw id - 2
WAYMO_CLASSES = WAYMO_RAW_CLASS_NAMES[WAYMO_RAW_OFFSET:]

# Instance-eligible raw ids, from deeplab2's _IS_THINGS. Keyed by RAW id because
# the instance-seg converter works in the raw id space (see instance_taxonomy.py).
WAYMO_THING_RAW_IDS = frozenset({
    2,   # car
    3,   # truck
    4,   # bus
    5,   # other large vehicle
    8,   # trailer
    9,   # person
    10,  # cyclist
    11,  # motorcyclist
})

# Storage encoding for the labels the download script writes: a uint16 PNG holding
# ``raw_class_id * WAYMO_PANOPTIC_DIVISOR + instance_id``, instance 0 = "no
# instance" (stuff). This mirrors the release's own panoptic encoding, but with a
# fixed divisor and per-image compacted instance ids so semantic and instance
# labels stay in one file. 28 * 1000 + 999 = 28999 fits uint16 with room to spare.
WAYMO_PANOPTIC_DIVISOR = 1000

# Trans10K-v2 transparent-object classes (background + 11), in the official index
# order taken from the Trans2Seg reference implementation
# (segmentron/data/dataloader/transparent11.py in xieenze/Trans2Seg). Lowercased to
# match every other taxonomy here, and the repo's "Glass Bow" typo is spelled out.
#
# NOTE: the Hugging Face mirror's README numbers the classes differently and calls
# index 8 "wall"; the official name is "floor glass". The names below follow the
# reference implementation, whose palette (TRANS10K_PALETTE) is index-aligned with it.
TRANS10K_CLASSES = (
    "background", "shelf", "jar or tank", "freezer", "window", "glass door",
    "eyeglass", "cup", "floor glass", "glass bowl", "water bottle", "storage box",
)

# trans10kv2pallete from segmentron/utils/visualize.py, index-aligned with
# TRANS10K_CLASSES. The raw release stores masks as these RGB colors; the download
# script folds them to class ids, and _read_trans10k_labelids falls back to matching
# against this palette when handed a color mask.
TRANS10K_PALETTE = np.array([
    [0, 0, 0], [120, 120, 70], [235, 255, 7], [6, 230, 230], [204, 255, 4],
    [120, 120, 120], [140, 140, 140], [255, 51, 7], [224, 5, 255], [204, 5, 255],
    [150, 5, 61], [4, 250, 7],
], dtype=np.uint8)


# RailSem19 (https://www.wilddash.cc/railsem19), index-aligned with the "labels"
# array in the release's rs19-config.json. Masks are uint8 with 0..18 = classes
# and 255 = void; there is no background class, so index 0 is a real class (road).
# The first eleven names overlap the Cityscapes taxonomy; the rail-specific
# supervision is tram-track / rail-track / trackbed / on-rails / rail-raised /
# rail-embedded.
RAILSEM19_CLASSES = (
    "road", "sidewalk", "construction", "tram-track", "fence", "pole",
    "traffic-light", "traffic-sign", "vegetation", "terrain", "sky", "human",
    "rail-track", "car", "truck", "trackbed", "on-rails", "rail-raised",
    "rail-embedded",
)


# Hypersim (photorealistic synthetic indoor scenes), labeled with the NYU40
# taxonomy. Names are transcribed from the release's own
# ``work/semantic_label_descs.csv`` (identical to the copy the ml-hypersim repo
# ships at code/cpp/tools/scene_annotation_tool/semantic_label_descs.csv), whose
# ``semantic_id`` column runs 1..40 -- raw id 0 does not exist and the semantic
# HDF5s use -1 for unlabeled pixels. Canonical index = raw id - 1.
HYPERSIM_RAW_LABEL_NAMES = (
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window",
    "bookshelf", "picture", "counter", "blinds", "desk", "shelves", "curtain",
    "dresser", "pillow", "mirror", "floormat", "clothes", "ceiling", "books",
    "refrigerator", "television", "paper", "towel", "showercurtain", "box",
    "whiteboard", "person", "nightstand", "toilet", "sink", "lamp", "bathtub",
    "bag", "otherstructure", "otherfurniture", "otherprop",
)
HYPERSIM_RAW_OFFSET = 1  # canonical index = raw id - 1

# Five run-together NYU40 names are respaced for prompt readability; every other
# name is the CSV spelling unchanged. verify_hypersim_labels() cross-checks the
# RAW names above against the built pool, so this rewording cannot mask a
# taxonomy that drifted upstream.
_HYPERSIM_READABLE = {
    "floormat": "floor mat",
    "showercurtain": "shower curtain",
    "otherstructure": "other structure",
    "otherfurniture": "other furniture",
    "otherprop": "other prop",
}
HYPERSIM_CLASSES = tuple(_HYPERSIM_READABLE.get(name, name) for name in HYPERSIM_RAW_LABEL_NAMES)

# Instance-eligible (countable) NYU40 classes, keyed by RAW id because the
# instance-seg converter works in the raw id space (see instance_taxonomy.py).
# Excluded as stuff: wall (1), floor (2), floormat (20), ceiling (22) -- unbounded
# surfaces -- plus the three catch-all buckets otherstructure (38),
# otherfurniture (39) and otherprop (40), which are not nameable noun phrases
# ("the 1st other prop" is not a query any evaluator would issue).
HYPERSIM_THING_RAW_IDS = frozenset(
    set(range(3, 20)) | {21} | set(range(23, 38))
)

# Storage encoding written by dataset_prep/hypersim/prepare_hypersim_segmentation.py:
# a uint16 PNG holding ``raw_class_id * HYPERSIM_PANOPTIC_DIVISOR + instance_id``,
# instance 0 = "no instance" (stuff), packed value 0 = unlabeled. Mirrors the Waymo
# encoding so both converters share one panoptic code path. 40*1000+999 fits uint16.
HYPERSIM_PANOPTIC_DIVISOR = 1000


def _pascal_colormap(n: int = 256) -> np.ndarray:
    """Standard PASCAL VOC colormap (bit-spread); index -> RGB."""
    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        cmap[i] = (r, g, b)
    return cmap


VOC_COLORMAP = _pascal_colormap()
VOC_VOID_RGB = (224, 224, 192)  # the cream boundary color in VOC SegmentationClass

# --- dataset registry -------------------------------------------------------- #
# trainids datasets share the Cityscapes taxonomy; background_index is excluded
# from selectable foreground (only VOC has an explicit background class).
_REGISTRY = {
    "cityscapes": dict(classes=CITYSCAPES_CLASSES, kind="trainids", background_index=None),
    "coconut_xlarge": dict(classes=(), kind="coconut_xlarge_panoptic", background_index=None,
                            raw_to_index={}, raw_names={}, instanceable_raw_ids=frozenset()),
    "acdc": dict(classes=CITYSCAPES_CLASSES, kind="trainids", background_index=None),
    "bdd": dict(classes=CITYSCAPES_CLASSES, kind="trainids", background_index=None),
    "synscapes": dict(classes=CITYSCAPES_CLASSES, kind="cityscapes_labelids", background_index=None),
    "idd": dict(classes=IDD_LEVEL3_CLASSES, kind="idd_level3_polygons", background_index=None),
    "wilddash2": dict(classes=(), kind="wilddash2_panoptic", background_index=None,
                       raw_to_index={}, raw_names={}, instanceable_raw_ids=frozenset()),
    "nuimages": dict(classes=(), kind="nuimages_rles", background_index=None,
                     token_to_index={}, instanceable_indices=frozenset(),
                     ignore_tokens=frozenset()),
    "muses": dict(classes=CITYSCAPES_CLASSES, kind="trainids", background_index=None),
    "goose": dict(classes=(), kind="goose_labelids", background_index=None, raw_to_index={}),
    "ade20k": dict(classes=ADE20K_CLASSES, kind="labelids", background_index=None),
    "voc": dict(classes=VOC_CLASSES, kind="voc_rgb", background_index=0),
    "pascal_context": dict(classes=PASCAL_CONTEXT_CLASSES, kind="pascal_context_labelids",
                           background_index=0),
    "foodseg103": dict(classes=FOODSEG103_CLASSES, kind="foodseg103_labelids", background_index=0),
    "trans10k": dict(classes=TRANS10K_CLASSES, kind="trans10k_labelids", background_index=0),
    "railsem19": dict(classes=RAILSEM19_CLASSES, kind="railsem19_labelids", background_index=None),
    "waymo": dict(classes=WAYMO_CLASSES, kind="waymo_panoptic", background_index=None),
    "hypersim": dict(classes=HYPERSIM_CLASSES, kind="hypersim_panoptic", background_index=None),
    "vipseg": dict(classes=(), kind="vipseg_panomask", background_index=None),
    "mapillary": dict(classes=(), kind="mapillary_labels", background_index=None,
                      raw_to_index={}, colors=()),
}


def class_names(dataset: str) -> tuple[str, ...]:
    return _REGISTRY[dataset]["classes"]


def background_index(dataset: str) -> Optional[int]:
    return _REGISTRY[dataset]["background_index"]


def _clean_mapping_name(value: str) -> str:
    return " ".join(str(value).strip().replace("_", " ").replace("-", " ").split())


def _norm_key(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _row_value(row: dict[str, str], candidates: Sequence[str]) -> str:
    normalized = {_norm_key(k): v for k, v in row.items()}
    for key in candidates:
        if _norm_key(key) in normalized:
            return normalized[_norm_key(key)]
    return ""


def _is_goose_void(row: dict[str, str], name: str) -> bool:
    group = _row_value(row, ("group", "category", "supercategory", "class_group"))
    text = f"{group} {name}".lower()
    return any(token in text for token in ("void", "undefined", "outlier", "ego vehicle", "ego_vehicle"))


def read_goose_mapping(mapping_path: Path) -> tuple[tuple[str, ...], dict[int, int], dict[int, str]]:
    """Parse GOOSE goose_label_mapping.csv.

    Returns canonical class names, raw-label-id -> canonical-index, and
    raw-label-id -> class name. Void/outlier/ego labels are treated as ignore.
    The parser accepts several likely column spellings so it remains robust
    across minor metadata revisions.
    """
    text = Path(mapping_path).read_text(encoding="utf-8", errors="replace")
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.DictReader(text.splitlines(), dialect=dialect))
    parsed: list[tuple[int, str, dict[str, str]]] = []
    for row in rows:
        raw_id = _row_value(row, ("id", "label_key", "labelkey", "label_id", "labelid", "class_id", "classid"))
        name = _row_value(row, ("name", "class_name", "classname", "class", "label", "label_name"))
        if not raw_id or not name:
            continue
        try:
            label_id = int(raw_id)
        except ValueError:
            continue
        clean_name = _clean_mapping_name(name)
        if not clean_name or _is_goose_void(row, clean_name):
            continue
        parsed.append((label_id, clean_name, row))
    if not parsed:
        raise ValueError(f"Could not parse any usable GOOSE classes from {mapping_path}")

    names: list[str] = []
    raw_to_index: dict[int, int] = {}
    raw_to_name: dict[int, str] = {}
    for raw_id, name, _row in sorted(parsed, key=lambda item: item[0]):
        if raw_id in raw_to_index:
            continue
        raw_to_index[raw_id] = len(names)
        raw_to_name[raw_id] = name
        names.append(name)
    return tuple(names), raw_to_index, raw_to_name


def register_goose_mapping(root_or_csv: Path) -> tuple[tuple[str, ...], dict[int, int], dict[int, str]]:
    path = Path(root_or_csv)
    mapping_path = path if path.is_file() else path / "goose_label_mapping.csv"
    names, raw_to_index, raw_to_name = read_goose_mapping(mapping_path)
    _REGISTRY["goose"]["classes"] = names
    _REGISTRY["goose"]["raw_to_index"] = raw_to_index
    _REGISTRY["goose"]["raw_to_name"] = raw_to_name
    return names, raw_to_index, raw_to_name


def read_vipseg_categories(categories_path: Path) -> tuple[str, ...]:
    """Parse VIPSeg's zero-based panoVIPSeg_categories.json.

    Native panomask pixels are one-based semantic ids for stuff classes and
    ``semantic_id * 100 + instance_id`` for thing classes, while this JSON stores
    category ids as 0..123. The mask reader below subtracts one from the native
    semantic id so the output indexes this name tuple directly.
    """
    categories = json.loads(Path(categories_path).read_text(encoding="utf-8"))
    names_by_id: dict[int, str] = {}
    for category in categories:
        cid = int(category["id"])
        name = _clean_mapping_name(str(category.get("name") or cid))
        if name:
            names_by_id[cid] = name
    if not names_by_id:
        raise ValueError(f"Could not parse any VIPSeg categories from {categories_path}")
    max_id = max(names_by_id)
    missing = [idx for idx in range(max_id + 1) if idx not in names_by_id]
    if missing:
        raise ValueError(f"VIPSeg categories are not contiguous in {categories_path}; missing ids {missing[:10]}")
    return tuple(names_by_id[idx] for idx in range(max_id + 1))


def register_vipseg_categories(root_or_json: Path) -> tuple[str, ...]:
    path = Path(root_or_json)
    categories_path = path if path.is_file() else path / "VIPSeg_720P" / "panoVIPSeg_categories.json"
    if not categories_path.exists():
        categories_path = path / "panoVIPSeg_categories.json"
    names = read_vipseg_categories(categories_path)
    _REGISTRY["vipseg"]["classes"] = names
    return names


def read_mapillary_config(config_path: Path
                          ) -> tuple[tuple[str, ...], dict[int, int], tuple[tuple[int, int, int], ...]]:
    """Parse Mapillary Vistas ``config_v2.0.json``.

    The label list is index-aligned with the pixel values in ``v2.0/labels/*.png``.
    Entries with ``evaluate == false`` (void / unlabeled) are dropped and read as
    ignore, so the returned names are compact and every one is selectable. Readable
    names are made unique by qualifying collisions with their taxonomy group
    ("Water" -> "nature water" / "marking water").

    Returns class names, raw-pixel-value -> canonical-index, and the raw palette
    colors (aligned with the raw pixel values, for RGB-encoded label fallbacks).
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    labels = config.get("labels") or []
    if not labels:
        raise ValueError(f"Could not parse any Mapillary labels from {config_path}")

    # Lowercased to match every other taxonomy here (prompts read "the road", not "the Road").
    readable = [_clean_mapping_name(str(lab.get("readable") or lab.get("name") or idx)).lower()
                for idx, lab in enumerate(labels)]
    duplicated = {name for name in readable if readable.count(name) > 1}

    names: list[str] = []
    raw_to_index: dict[int, int] = {}
    colors: list[tuple[int, int, int]] = []
    for raw_id, (label, name) in enumerate(zip(labels, readable)):
        colors.append(tuple(int(c) for c in (label.get("color") or (0, 0, 0))[:3]))
        if not label.get("evaluate", True) or not name:
            continue
        if name in duplicated:
            group = _clean_mapping_name(str(label.get("name", "")).split("--")[0])
            name = f"{group} {name}".strip()
        raw_to_index[raw_id] = len(names)
        names.append(name)
    if not names:
        raise ValueError(f"Every Mapillary label in {config_path} is marked evaluate=false")
    return tuple(names), raw_to_index, tuple(colors)


def register_mapillary_config(root_or_json: Path
                              ) -> tuple[tuple[str, ...], dict[int, int], tuple[tuple[int, int, int], ...]]:
    path = Path(root_or_json)
    config_path = path if path.is_file() else path / "config_v2.0.json"
    if not config_path.exists():
        config_path = path / "config.json"
    names, raw_to_index, colors = read_mapillary_config(config_path)
    _REGISTRY["mapillary"]["classes"] = names
    _REGISTRY["mapillary"]["raw_to_index"] = raw_to_index
    _REGISTRY["mapillary"]["colors"] = colors
    return names, raw_to_index, colors


def resolve_wilddash2_root(root: Path) -> Path:
    """Resolve a WildDash 2 source tree or a parent containing one."""
    root = Path(root).expanduser().resolve()
    for candidate in (root, root / "wilddash2", root / "wd_public_v2p"):
        if ((candidate / "anonymized").is_dir() and (candidate / "panoptic").is_dir()
                and (candidate / "panoptic.json").is_file()
                and (candidate / "wilddash2-meta.json").is_file()
                and (candidate / "random_split").is_dir()):
            return candidate
    raise FileNotFoundError(
        "Could not resolve WildDash 2 root. Expected anonymized/, panoptic/, "
        "panoptic.json, wilddash2-meta.json, and random_split/ under "
        f"{root} or a wilddash2/ child."
    )


def normalize_wilddash2_splits(splits: Sequence[str]) -> set[str]:
    selected = {
        WILDDASH2_SPLIT_ALIASES[str(split).strip().lower()]
        for split in splits
        if str(split).strip().lower() in WILDDASH2_SPLIT_ALIASES
    }
    if not selected:
        raise ValueError(
            f"WildDash 2 splits must contain train and/or val; received {list(splits)!r}"
        )
    return selected


def _wilddash2_display_name(label: dict) -> str:
    raw_name = _clean_mapping_name(str(label.get("name") or "unknown"))
    readable = str(label.get("readable") or "").strip()
    friendly = readable.split("(", 1)[0].split(";", 1)[0].strip()
    if not friendly or friendly.lower().startswith("only used") or len(friendly) > 48:
        return raw_name
    return _clean_mapping_name(friendly)


def read_wilddash2_metadata(
    root_or_json: Path,
) -> tuple[tuple[str, ...], dict[int, int], dict[int, str], frozenset[int]]:
    """Read public WildDash 2 labels into a compact non-void taxonomy.

    The public release defines 81 raw labels.  We retain every non-void label,
    including categories not scored by the benchmark, because they are valid
    public semantic annotations and carry useful instance labels.
    """
    path = Path(root_or_json)
    metadata_path = path if path.is_file() else path / "wilddash2-meta.json"
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    labels = data.get("labels", ())
    classes: list[str] = []
    raw_to_index: dict[int, int] = {}
    raw_names: dict[int, str] = {}
    instanceable: set[int] = set()
    used_names: set[str] = set()
    for label in sorted(labels, key=lambda item: int(item["id"])):
        if str(label.get("supercategory", "")).strip().lower() == "void":
            continue
        raw_id = int(label["id"])
        name = _wilddash2_display_name(label)
        if name in used_names:
            name = f"{name} ({_clean_mapping_name(str(label.get('name') or raw_id))})"
        used_names.add(name)
        raw_to_index[raw_id] = len(classes)
        raw_names[raw_id] = name
        classes.append(name)
        if bool(label.get("instances", False)):
            instanceable.add(raw_id)
    if not classes:
        raise ValueError(f"Could not parse non-void WildDash 2 labels from {metadata_path}")
    return tuple(classes), raw_to_index, raw_names, frozenset(instanceable)


def register_wilddash2_metadata(
    root_or_json: Path,
) -> tuple[tuple[str, ...], dict[int, int], dict[int, str], frozenset[int]]:
    classes, raw_to_index, raw_names, instanceable = read_wilddash2_metadata(root_or_json)
    entry = _REGISTRY["wilddash2"]
    entry["classes"] = classes
    entry["raw_to_index"] = raw_to_index
    entry["raw_names"] = raw_names
    entry["instanceable_raw_ids"] = instanceable
    return classes, raw_to_index, raw_names, instanceable


def wilddash2_instance_categories(root_or_json: Path) -> tuple[dict[int, str], set[int]]:
    _classes, _raw_to_index, raw_names, instanceable = register_wilddash2_metadata(root_or_json)
    return raw_names, set(instanceable)



def resolve_coconut_xlarge_root(root: Path) -> Path:
    """Resolve COCONut-XL's materialized ``coco/`` directory."""
    root = Path(root).expanduser().resolve()
    for candidate in (root, root / "coco", root / "coconut_xlarge" / "coco"):
        if ((candidate / "train2017").is_dir()
                and (candidate / "panoptic_train2017").is_dir()
                and (candidate / "annotations" / "panoptic_train2017.json").is_file()):
            return candidate
    raise FileNotFoundError(
        "Could not resolve COCONut-XL root. Expected train2017/, panoptic_train2017/, "
        "and annotations/panoptic_train2017.json under "
        f"{root}, its coco/ child, or its coconut_xlarge/coco/ child."
    )


def coconut_xlarge_annotation_path(root_or_json: Path) -> Path:
    """Return the final merged COCONut-XL COCO-panoptic manifest path."""
    path = Path(root_or_json)
    return path if path.is_file() else (
        resolve_coconut_xlarge_root(path) / "annotations" / "panoptic_train2017.json")



def _coconut_xlarge_display_name(raw_name: str) -> str:
    """Make COCONut's COCO-panoptic provenance suffix prompt-friendly."""
    return _clean_mapping_name(str(raw_name).removesuffix("-merged"))


def read_coconut_xlarge_metadata(
    root_or_json: Path,
) -> tuple[tuple[str, ...], dict[int, int], dict[int, str], frozenset[int]]:
    """Read COCONut-XL's COCO-panoptic categories into a compact taxonomy."""
    metadata_path = coconut_xlarge_annotation_path(root_or_json)
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    classes: list[str] = []
    raw_to_index: dict[int, int] = {}
    raw_names: dict[int, str] = {}
    instanceable: set[int] = set()
    used_names: set[str] = set()
    for category in sorted(data.get("categories", ()), key=lambda item: int(item["id"])):
        raw_id = int(category["id"])
        name = _coconut_xlarge_display_name(str(category.get("name") or raw_id))
        if name in used_names:
            name = f"{name} ({raw_id})"
        used_names.add(name)
        raw_to_index[raw_id] = len(classes)
        raw_names[raw_id] = name
        classes.append(name)
        if int(category.get("isthing", 0)) == 1:
            instanceable.add(raw_id)
    if not classes:
        raise ValueError(f"Could not parse COCONut-XL categories from {metadata_path}")
    return tuple(classes), raw_to_index, raw_names, frozenset(instanceable)


def register_coconut_xlarge_metadata(
    root_or_json: Path,
) -> tuple[tuple[str, ...], dict[int, int], dict[int, str], frozenset[int]]:
    """Register COCONut-XL metadata for semantic and instance preparation."""
    classes, raw_to_index, raw_names, instanceable = read_coconut_xlarge_metadata(root_or_json)
    entry = _REGISTRY["coconut_xlarge"]
    entry["classes"] = classes
    entry["raw_to_index"] = raw_to_index
    entry["raw_names"] = raw_names
    entry["instanceable_raw_ids"] = instanceable
    return classes, raw_to_index, raw_names, instanceable


def coconut_xlarge_instance_categories(root_or_json: Path) -> tuple[dict[int, str], set[int]]:
    """Return COCONut-XL's panoptic category names and thing-category IDs."""
    _classes, _raw_to_index, raw_names, instanceable = register_coconut_xlarge_metadata(root_or_json)
    return raw_names, set(instanceable)

def resolve_nuimages_root(root: Path) -> Path:
    """Resolve a NuImages tree containing key images and v1.0 JSON tables."""
    root = Path(root).expanduser().resolve()
    for candidate in (root, root / "nuimages"):
        if ((candidate / "samples").is_dir()
                and (candidate / "v1.0-train" / "category.json").is_file()
                and (candidate / "v1.0-train" / "sample.json").is_file()
                and (candidate / "v1.0-train" / "sample_data.json").is_file()):
            return candidate
    raise FileNotFoundError(
        "Could not resolve NuImages root. Expected samples/ plus v1.0-train/"
        "{category,sample,sample_data}.json under "
        f"{root} or a nuimages/ child."
    )


def normalize_nuimages_splits(splits: Sequence[str]) -> set[str]:
    selected = {
        NUIMAGES_SPLIT_ALIASES[str(split).strip().lower()]
        for split in splits
        if str(split).strip().lower() in NUIMAGES_SPLIT_ALIASES
    }
    if not selected:
        raise ValueError(
            "NuImages supplies masks only for train and/or val; "
            f"received {list(splits)!r}"
        )
    return selected


def _nuimages_display_name(raw_name: str) -> str:
    return _clean_mapping_name(raw_name.replace(".", " "))


def read_nuimages_metadata(
    root_or_json: Path,
) -> tuple[tuple[str, ...], dict[str, int], frozenset[int], frozenset[str]]:
    """Read NuImages categories into a compact semantic/instance taxonomy.

    ``object_ann`` marks foreground object masks. ``surface_ann`` carries the
    driveable-surface masks. The ego vehicle is useful semantic context but is
    excluded from the instance pool, like other ego-car labels in this project.
    Official NuImages guidance designates bicycle racks as ignore labels, so they
    remain void rather than becoming semantic or instance prompt categories.
    """
    path = Path(root_or_json)
    category_path = path if path.is_file() else path / "v1.0-train" / "category.json"
    categories = json.loads(category_path.read_text(encoding="utf-8"))
    classes: list[str] = []
    token_to_index: dict[str, int] = {}
    instanceable: set[int] = set()
    ignore_tokens: set[str] = set()
    used_names: set[str] = set()
    for category in categories:
        token = str(category.get("token") or "")
        raw_name = str(category.get("name") or "")
        if not token or not raw_name:
            continue
        if raw_name in NUIMAGES_IGNORE_CATEGORY_NAMES:
            ignore_tokens.add(token)
            continue
        if token in token_to_index:
            continue
        name = _nuimages_display_name(raw_name)
        if name in used_names:
            name = f"{name} ({token[:8]})"
        used_names.add(name)
        index = len(classes)
        token_to_index[token] = index
        classes.append(name)
        if raw_name not in NUIMAGES_SURFACE_CATEGORY_NAMES | {NUIMAGES_EGO_CATEGORY_NAME}:
            instanceable.add(index)
    if not classes:
        raise ValueError(f"Could not parse NuImages categories from {category_path}")
    return tuple(classes), token_to_index, frozenset(instanceable), frozenset(ignore_tokens)


def register_nuimages_metadata(
    root_or_json: Path,
) -> tuple[tuple[str, ...], dict[str, int], frozenset[int]]:
    classes, token_to_index, instanceable, ignore_tokens = read_nuimages_metadata(root_or_json)
    entry = _REGISTRY["nuimages"]
    entry["classes"] = classes
    entry["token_to_index"] = token_to_index
    entry["instanceable_indices"] = instanceable
    entry["ignore_tokens"] = ignore_tokens
    return classes, token_to_index, instanceable


def nuimages_instance_categories(root_or_json: Path) -> tuple[dict[int, str], set[int], dict[str, int]]:
    """Return canonical NuImages noun phrases, thing IDs, and token mapping."""
    classes, token_to_index, instanceable = register_nuimages_metadata(root_or_json)
    return {index: name for index, name in enumerate(classes)}, set(instanceable), token_to_index


def _read_nuimages_table(root: Path, split: str, table: str) -> list[dict]:
    path = root / f"v1.0-{split}" / f"{table}.json"
    if not path.is_file():
        raise FileNotFoundError(f"NuImages {split} metadata table is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"NuImages metadata table is not a JSON list: {path}")
    return data


def discover_nuimages_frames(
    root: Path,
    splits: Sequence[str],
    *,
    sample_interval_seconds: float = NUIMAGES_DEFAULT_SAMPLE_INTERVAL_SECONDS,
    require_image_files: bool = True,
) -> tuple[list[NuImagesFrame], int]:
    """Discover mask-bearing keyframes and sample every N seconds per log.

    The first usable keyframe in every log is retained, then the next one at
    least ``sample_interval_seconds`` later. This is deterministic, preserves
    all logs, and avoids the dense near-duplicate samples in a driving record.
    ``require_image_files=False`` supports the raw-tar preview helper, which
    reads metadata before streaming its few selected JPEGs.
    """
    resolved = resolve_nuimages_root(root)
    register_nuimages_metadata(resolved)
    selected_splits = normalize_nuimages_splits(splits)
    if sample_interval_seconds < 0:
        raise ValueError("NuImages sample interval must be non-negative")
    interval_us = int(round(sample_interval_seconds * 1_000_000))
    frames: list[NuImagesFrame] = []
    skipped = 0

    for split in sorted(selected_splits):
        samples = _read_nuimages_table(resolved, split, "sample")
        sample_keys = {str(sample.get("key_camera_token") or "") for sample in samples}
        sample_keys.discard("")
        sample_data = {
            str(record.get("token")): record
            for record in _read_nuimages_table(resolved, split, "sample_data")
            if str(record.get("token") or "") in sample_keys and bool(record.get("is_key_frame"))
        }
        surface_by_key: dict[str, list[dict]] = {}
        object_by_key: dict[str, list[dict]] = {}
        for ann in _read_nuimages_table(resolved, split, "surface_ann"):
            key = str(ann.get("sample_data_token") or "")
            if key in sample_keys and ann.get("mask") is not None:
                surface_by_key.setdefault(key, []).append(ann)
        for ann in _read_nuimages_table(resolved, split, "object_ann"):
            key = str(ann.get("sample_data_token") or "")
            if key in sample_keys and ann.get("mask") is not None:
                object_by_key.setdefault(key, []).append(ann)

        by_log: dict[str, list[NuImagesFrame]] = {}
        for sample in samples:
            key = str(sample.get("key_camera_token") or "")
            source_annotations = surface_by_key.get(key, []) + object_by_key.get(key, [])
            record = sample_data.get(key)
            if not source_annotations or record is None:
                skipped += 1
                continue
            filename = str(record.get("filename") or "")
            image_path = resolved / filename
            if not filename or (require_image_files and not image_path.is_file()):
                skipped += 1
                continue
            annotations = tuple(
                [dict(ann, _nuimages_annotation_type="surface") for ann in surface_by_key.get(key, [])]
                + [dict(ann, _nuimages_annotation_type="object") for ann in object_by_key.get(key, [])]
            )
            frame = NuImagesFrame(
                split=split,
                token=str(sample.get("token") or key),
                log_token=str(sample.get("log_token") or "unknown"),
                timestamp=int(sample.get("timestamp") or record.get("timestamp") or 0),
                image_path=image_path,
                annotations=annotations,
            )
            by_log.setdefault(frame.log_token, []).append(frame)

        for log_token in sorted(by_log):
            next_timestamp: int | None = None
            for frame in sorted(by_log[log_token], key=lambda item: (item.timestamp, item.token)):
                if next_timestamp is None or interval_us == 0 or frame.timestamp >= next_timestamp:
                    frames.append(frame)
                    next_timestamp = frame.timestamp + interval_us

    return sorted(frames, key=lambda item: (item.split, item.log_token, item.timestamp, item.token)), skipped


# --- source layout helpers shared by semantic and instance preparation ------- #
def resolve_synscapes_root(root: Path) -> Path:
    """Resolve the dataset root or a parent containing ``Synscapes/``."""
    root = Path(root).expanduser().resolve()
    for candidate in (root, root / "Synscapes", root / "synscapes"):
        image_root = candidate / "img"
        if (image_root / "rgb").is_dir() and (image_root / "class").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not resolve Synscapes root. Expected img/{rgb,class} under "
        f"{root} or a Synscapes/ child."
    )


def normalize_synscapes_splits(splits: Sequence[str]) -> set[str]:
    selected = {
        SYNSCAPES_SPLIT_ALIASES[str(split).strip().lower()]
        for split in splits
        if str(split).strip().lower() in SYNSCAPES_SPLIT_ALIASES
    }
    if not selected:
        raise ValueError(
            f"Synscapes splits must contain train and/or val; received {list(splits)!r}"
        )
    return selected


def synscapes_split_for_index(index: int) -> str | None:
    if SYNSCAPES_FIRST_INDEX <= index <= SYNSCAPES_TRAIN_LAST_INDEX:
        return "train"
    if SYNSCAPES_TRAIN_LAST_INDEX < index <= SYNSCAPES_LAST_INDEX:
        return "val"
    return None


def synscapes_numbered_pngs(directory: Path) -> Iterable[tuple[int, Path]]:
    """Yield numeric-stem PNGs in numeric rather than lexicographic order."""
    indexed: list[tuple[int, Path]] = []
    for path in Path(directory).glob("*.png"):
        try:
            index = int(path.stem)
        except ValueError:
            continue
        indexed.append((index, path))
    yield from sorted(indexed, key=lambda item: item[0])


def normalize_idd_splits(splits: Sequence[str]) -> set[str]:
    selected = {
        IDD_SPLIT_ALIASES[str(split).strip().lower()]
        for split in splits
        if str(split).strip().lower() in IDD_SPLIT_ALIASES
    }
    if not selected:
        raise ValueError(
            "IDD splits must contain train, val, and/or test; "
            f"received {list(splits)!r}"
        )
    return selected


def resolve_idd_roots(root: Path) -> tuple[Path, ...]:
    """Resolve one IDD part or a parent containing both archive parts."""
    root = Path(root).expanduser().resolve()
    candidates = (root, root / "IDD_Segmentation", root / "idd20kII")
    roots: list[Path] = []
    for candidate in candidates:
        if (candidate / "gtFine").is_dir() and candidate not in roots:
            roots.append(candidate)
    if not roots:
        raise FileNotFoundError(
            "Could not resolve an IDD source root. Expected gtFine/ under "
            f"{root}, {root / 'IDD_Segmentation'}, or {root / 'idd20kII'}."
        )
    return tuple(roots)


def idd_source_key(root: Path) -> str:
    """Return a stable, output-safe part name for pair IDs and balancing."""
    return Path(root).name.lower().replace("-", "_")


def _idd_label_id(label: object) -> int | None:
    name = str(label).strip()
    raw_id = IDD_RAW_LABEL_IDS.get(name)
    if raw_id is not None:
        return raw_id
    if name.endswith("group"):
        return IDD_RAW_LABEL_IDS.get(name[: -len("group")])
    return None


def read_idd_objects(path: Path) -> tuple[int, int, tuple[IDDPolygonObject, ...]]:
    """Read valid, recognized IDD polygons in release draw order."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    width, height = int(data["imgWidth"]), int(data["imgHeight"])
    objects: list[IDDPolygonObject] = []
    for obj in data.get("objects", []):
        if obj.get("deleted", False):
            continue
        raw_id = _idd_label_id(obj.get("label", ""))
        if raw_id is None:
            continue
        polygon = tuple((float(x), float(y)) for x, y in obj.get("polygon", []))
        if len(polygon) >= 3:
            objects.append(IDDPolygonObject(raw_id, polygon))
    return width, height, tuple(objects)


def read_idd_level3_labels(path: Path) -> np.ndarray:
    """Rasterize IDD polygon JSON to official compact level-3 label IDs."""
    width, height, objects = read_idd_objects(path)
    image = Image.new("L", (width, height), IGNORE_INDEX)
    drawer = ImageDraw.Draw(image)
    for obj in objects:
        drawer.polygon(obj.polygon, fill=IDD_RAW_ID_TO_LEVEL3_ID[obj.raw_id])
    return np.asarray(image, dtype=np.int64)


def read_idd_instance_masks(
    path: Path,
    instanceable: set[int],
) -> dict[int, list[np.ndarray]]:
    """Return visible thing polygons grouped by raw IDD instance class ID."""
    width, height, objects = read_idd_objects(path)
    owner = Image.new("I", (width, height), 0)
    drawer = ImageDraw.Draw(owner)
    for index, obj in enumerate(objects, start=1):
        drawer.polygon(obj.polygon, fill=index)
    owner_array = np.asarray(owner)

    output: dict[int, list[np.ndarray]] = {}
    for index, obj in enumerate(objects, start=1):
        if obj.raw_id not in instanceable:
            continue
        mask = owner_array == index
        if mask.any():
            output.setdefault(obj.raw_id, []).append(mask)
    return output


# --- mask readers ------------------------------------------------------------ #
def _read_trainids(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:  # gray train-id stored as RGB -> collapse
        if not (np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2])):
            raise ValueError(f"Expected a train-id mask, got a color image: {path}")
        arr = arr[..., 0]
    return arr.astype(np.int64)


def _read_cityscapes_labelids(path: Path) -> np.ndarray:
    """Map raw Cityscapes label IDs to canonical train IDs; all void IDs ignore."""
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        if not (np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2])):
            raise ValueError(f"Expected a grayscale Cityscapes label-id mask: {path}")
        arr = arr[..., 0]
    out = np.full(arr.shape, IGNORE_INDEX, dtype=np.int64)
    for raw_id, train_id in CITYSCAPES_RAW_ID_TO_TRAIN_ID.items():
        out[arr == raw_id] = train_id
    return out


def _read_labelids(path: Path) -> np.ndarray:
    """ADE20k: raw value 0 = ignore, 1..150 = classes -> canonical 0..149 / 255."""
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.int64)
    out = arr - 1
    out[arr == 0] = IGNORE_INDEX
    out[arr > len(ADE20K_CLASSES)] = IGNORE_INDEX
    return out


def _read_voc(path: Path) -> np.ndarray:
    """VOC: palette-index PNG gives class ids directly; RGB falls back to colormap."""
    im = Image.open(path)
    if im.mode in ("P", "L"):
        arr = np.asarray(im).astype(np.int64)
        arr[arr > len(VOC_CLASSES) - 1] = IGNORE_INDEX  # 255 boundary stays ignore
        return arr
    rgb = np.asarray(im.convert("RGB")).astype(np.int64)
    targets = np.array([list(VOC_COLORMAP[i]) for i in range(len(VOC_CLASSES))] + [list(VOC_VOID_RGB)])
    mapping = np.array(list(range(len(VOC_CLASSES))) + [IGNORE_INDEX], dtype=np.int64)
    flat = rgb.reshape(-1, 3)
    nearest = ((flat[:, None, :] - targets[None, :, :]) ** 2).sum(2).argmin(1)
    return mapping[nearest].reshape(rgb.shape[:2])


def _read_goose_labelids(path: Path) -> np.ndarray:
    raw_to_index = _REGISTRY["goose"].get("raw_to_index") or {}
    if not raw_to_index:
        raise RuntimeError("GOOSE mapping is not registered. Call register_goose_mapping(<goose-root>) first.")
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.int64)
    out = np.full(arr.shape, IGNORE_INDEX, dtype=np.int64)
    for raw_id, idx in raw_to_index.items():
        out[arr == int(raw_id)] = int(idx)
    return out


def verify_pascal_context_labels(labels_root: Path) -> tuple[str, ...]:
    """Cross-check PASCAL_CONTEXT_CLASSES against the id2label.json the download
    script writes beside the labels, so a regenerated dataset whose class order
    changed upstream fails loudly instead of mislabeling every mask."""
    id2label = Path(labels_root) / "id2label.json"
    if not id2label.exists():
        return PASCAL_CONTEXT_CLASSES
    mapping = json.loads(id2label.read_text(encoding="utf-8"))
    names = tuple(mapping[str(i)] for i in range(len(mapping)))
    if names != PASCAL_CONTEXT_CLASSES:
        differing = [(i, a, b) for i, (a, b) in enumerate(zip(names, PASCAL_CONTEXT_CLASSES)) if a != b]
        raise ValueError(
            f"{id2label} disagrees with PASCAL_CONTEXT_CLASSES "
            f"({len(names)} vs {len(PASCAL_CONTEXT_CLASSES)} names; first mismatch: {differing[:1]})"
        )
    return PASCAL_CONTEXT_CLASSES


def _read_pascal_context_labelids(path: Path) -> np.ndarray:
    """PASCAL-Context masks: raw value 0 = background, 1..59 = classes."""
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        if not (np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2])):
            raise ValueError(f"Expected a raw class-id mask, got a color image: {path}")
        arr = arr[..., 0]
    arr = arr.astype(np.int64)
    n = len(PASCAL_CONTEXT_CLASSES)
    arr[(arr < 0) | (arr >= n)] = IGNORE_INDEX
    return arr


def _read_foodseg103_labelids(path: Path) -> np.ndarray:
    """FoodSeg103 grayscale masks: raw value 0=background, 1..103=ingredients."""
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        if not (np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2])):
            raise ValueError(f"Expected a raw class-id mask, got a color image: {path}")
        arr = arr[..., 0]
    arr = arr.astype(np.int64)
    n = len(FOODSEG103_CLASSES)
    arr[(arr < 0) | (arr >= n)] = IGNORE_INDEX
    return arr


def verify_trans10k_labels(root: Path) -> tuple[str, ...]:
    """Cross-check TRANS10K_CLASSES against the id2label.json the download script
    writes beside the labels, so a regenerated dataset whose class order changed
    upstream fails loudly instead of mislabeling every mask."""
    id2label = Path(root) / "id2label.json"
    if not id2label.exists():
        return TRANS10K_CLASSES
    mapping = json.loads(id2label.read_text(encoding="utf-8"))
    names = tuple(mapping[str(i)] for i in range(len(mapping)))
    if names != TRANS10K_CLASSES:
        differing = [(i, a, b) for i, (a, b) in enumerate(zip(names, TRANS10K_CLASSES)) if a != b]
        raise ValueError(
            f"{id2label} disagrees with TRANS10K_CLASSES "
            f"({len(names)} vs {len(TRANS10K_CLASSES)} names; first mismatch: {differing[:1]})"
        )
    return TRANS10K_CLASSES


def verify_railsem19_labels(root: Path) -> tuple[str, ...]:
    """Cross-check RAILSEM19_CLASSES against the rs19-config.json the release ships.

    Mirrors verify_trans10k_labels: if the config is present anywhere under the
    dataset root, a class list that drifted upstream fails loudly here instead of
    silently mislabeling every mask.
    """
    root = Path(root)
    candidates = [root / "rs19-config.json", *root.glob("*/rs19-config.json")]
    config = next((c for c in candidates if c.exists()), None)
    if config is None:
        return RAILSEM19_CLASSES
    labels = json.loads(config.read_text(encoding="utf-8")).get("labels") or []
    names = tuple(entry["name"] for entry in labels if "name" in entry)
    if names != RAILSEM19_CLASSES:
        differing = [(i, a, b) for i, (a, b) in enumerate(zip(names, RAILSEM19_CLASSES)) if a != b]
        raise ValueError(
            f"{config} disagrees with RAILSEM19_CLASSES "
            f"({len(names)} vs {len(RAILSEM19_CLASSES)} names; first mismatch: {differing[:1]})"
        )
    return RAILSEM19_CLASSES


def _read_railsem19_labelids(path: Path) -> np.ndarray:
    """RailSem19 uint8 masks: 0..18 = classes, 255 = void -> IGNORE_INDEX.

    Unlike FoodSeg103/Trans10K there is no background class -- index 0 is `road`,
    so nothing is remapped; only out-of-range values become ignore.
    """
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        if not (np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2])):
            raise ValueError(f"Expected a raw class-id mask, got a color image: {path}")
        arr = arr[..., 0]
    arr = arr.astype(np.int64)
    arr[(arr < 0) | (arr >= len(RAILSEM19_CLASSES))] = IGNORE_INDEX
    return arr


def _read_trans10k_labelids(path: Path) -> np.ndarray:
    """Trans10K-v2 masks: raw value 0 = background, 1..11 = transparent classes.

    download_datasets_trans10k.sh writes class ids, but the raw release ships RGB
    masks painted with TRANS10K_PALETTE, so a color image is matched back against
    that palette instead (nearest color, as a handful of pixels sit slightly off it).
    """
    im = Image.open(path)
    arr = np.asarray(im)
    n = len(TRANS10K_CLASSES)
    if arr.ndim == 3:
        arr = arr[..., :3]  # drop alpha; the release stores masks as RGBA
        if not (np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2])):
            targets = TRANS10K_PALETTE.astype(np.int64)
            flat = arr.reshape(-1, 3).astype(np.int64)
            nearest = ((flat[:, None, :] - targets[None, :, :]) ** 2).sum(2).argmin(1)
            return nearest.reshape(arr.shape[:2])
        arr = arr[..., 0]
    arr = arr.astype(np.int64)
    arr[(arr < 0) | (arr >= n)] = IGNORE_INDEX
    return arr


def verify_waymo_labels(root: Path) -> tuple[str, ...]:
    """Cross-check WAYMO_RAW_CLASS_NAMES + the panoptic divisor against the
    id2label.json the download script writes, so a regenerated dataset built with a
    different class order or divisor fails loudly instead of mislabeling every mask."""
    meta_path = Path(root) / "id2label.json"
    if not meta_path.exists():
        return WAYMO_CLASSES
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    names = tuple(meta["classes"][str(i)] for i in range(len(meta["classes"])))
    if names != WAYMO_RAW_CLASS_NAMES:
        differing = [(i, a, b) for i, (a, b) in enumerate(zip(names, WAYMO_RAW_CLASS_NAMES)) if a != b]
        raise ValueError(
            f"{meta_path} disagrees with WAYMO_RAW_CLASS_NAMES "
            f"({len(names)} vs {len(WAYMO_RAW_CLASS_NAMES)} names; first mismatch: {differing[:1]})"
        )
    divisor = int(meta.get("panoptic_label_divisor", WAYMO_PANOPTIC_DIVISOR))
    if divisor != WAYMO_PANOPTIC_DIVISOR:
        raise ValueError(
            f"{meta_path} was written with panoptic_label_divisor={divisor}, but this "
            f"code decodes with {WAYMO_PANOPTIC_DIVISOR}. Rebuild the dataset or fix the constant."
        )
    return WAYMO_CLASSES


def _read_waymo_panoptic(path: Path) -> np.ndarray:
    """Waymo panoptic PNG -> semantic class index / ignore.

    Pixels are ``raw_class_id * WAYMO_PANOPTIC_DIVISOR + instance_id``; the semantic
    half is the raw CameraSegmentation.Type, shifted down by the two leading ignore
    classes (unknown, sdc).
    """
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    raw = arr.astype(np.int64) // WAYMO_PANOPTIC_DIVISOR
    out = raw - WAYMO_RAW_OFFSET
    out[raw < WAYMO_RAW_OFFSET] = IGNORE_INDEX          # unknown / sdc
    out[raw >= len(WAYMO_RAW_CLASS_NAMES)] = IGNORE_INDEX
    return out


def verify_hypersim_labels(root: Path) -> tuple[str, ...]:
    """Cross-check HYPERSIM_RAW_LABEL_NAMES + the divisor against the id2label.json
    prepare_hypersim_segmentation.py writes, so a pool rebuilt from a different
    NYU40 spelling or divisor fails loudly instead of mislabeling every mask.

    Mirrors verify_waymo_labels; the ids in that file are the raw 1..40 CSV ids.
    """
    meta_path = Path(root) / "id2label.json"
    if not meta_path.exists():
        return HYPERSIM_CLASSES
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    classes = meta["classes"]
    names = tuple(classes[str(i + HYPERSIM_RAW_OFFSET)] for i in range(len(classes)))
    if names != HYPERSIM_RAW_LABEL_NAMES:
        differing = [(i, a, b) for i, (a, b) in enumerate(zip(names, HYPERSIM_RAW_LABEL_NAMES)) if a != b]
        raise ValueError(
            f"{meta_path} disagrees with HYPERSIM_RAW_LABEL_NAMES "
            f"({len(names)} vs {len(HYPERSIM_RAW_LABEL_NAMES)} names; first mismatch: {differing[:1]})"
        )
    divisor = int(meta.get("panoptic_label_divisor", HYPERSIM_PANOPTIC_DIVISOR))
    if divisor != HYPERSIM_PANOPTIC_DIVISOR:
        raise ValueError(
            f"{meta_path} was written with panoptic_label_divisor={divisor}, but this "
            f"code decodes with {HYPERSIM_PANOPTIC_DIVISOR}. Rebuild the pool or fix the constant."
        )
    return HYPERSIM_CLASSES


def _read_hypersim_panoptic(path: Path) -> np.ndarray:
    """Hypersim packed panoptic PNG -> semantic class index / ignore.

    Pixels are ``raw_class_id * HYPERSIM_PANOPTIC_DIVISOR + instance_id`` with NYU40
    raw ids running 1..40, so packed value 0 (unlabeled) falls below the offset and
    reads as ignore.
    """
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    raw = arr.astype(np.int64) // HYPERSIM_PANOPTIC_DIVISOR
    out = raw - HYPERSIM_RAW_OFFSET
    out[raw < HYPERSIM_RAW_OFFSET] = IGNORE_INDEX
    out[out >= len(HYPERSIM_CLASSES)] = IGNORE_INDEX
    return out


def _read_vipseg_panomask(path: Path) -> np.ndarray:
    """VIPSeg native uint16 panomask -> zero-based category index / ignore.

    Official encoding:
      0 = void/other
      1..124 = stuff semantic id
      class_id * 100 + instance_id = thing instance, with class_id in 1..124
    """
    if not class_names("vipseg"):
        raise RuntimeError("VIPSeg categories are not registered. Call register_vipseg_categories(<vipseg-root>) first.")
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.int64)
    sem = np.full(arr.shape, IGNORE_INDEX, dtype=np.int64)

    stuff = (arr > 0) & (arr < 125)
    sem[stuff] = arr[stuff] - 1

    things = arr > 124
    sem[things] = (arr[things] // 100) - 1

    n = len(class_names("vipseg"))
    sem[(sem < 0) | (sem >= n)] = IGNORE_INDEX
    return sem


def _read_mapillary_labels(path: Path) -> np.ndarray:
    """Mapillary Vistas v2.0 labels -> compact class index / ignore.

    ``v2.0/labels/*.png`` is a paletted PNG whose palette index IS the label id in
    config_v2.0.json. Some redistributions flatten it to RGB, so a color image is
    matched back against the config palette instead.
    """
    raw_to_index = _REGISTRY["mapillary"].get("raw_to_index") or {}
    if not raw_to_index:
        raise RuntimeError("Mapillary config is not registered. Call register_mapillary_config(<mapillary-root>) first.")
    im = Image.open(path)
    if im.mode in ("P", "L", "I", "I;16"):
        arr = np.asarray(im).astype(np.int64)
    else:
        palette = np.array(_REGISTRY["mapillary"]["colors"], dtype=np.int64)
        rgb = np.asarray(im.convert("RGB")).astype(np.int64).reshape(-1, 3)
        arr = ((rgb[:, None, :] - palette[None, :, :]) ** 2).sum(2).argmin(1)
        arr = arr.reshape(np.asarray(im).shape[:2])
    out = np.full(arr.shape, IGNORE_INDEX, dtype=np.int64)
    for raw_id, idx in raw_to_index.items():
        out[arr == int(raw_id)] = int(idx)
    return out


def _read_idd_level3_polygons(path: Path) -> np.ndarray:
    """IDD polygon JSON -> its official compact level-3 labels / ignore."""
    return read_idd_level3_labels(path)


def _read_wilddash2_panoptic(path: Path, segments_info: Sequence[dict]) -> np.ndarray:
    """Decode a COCO-panoptic RGB PNG to compact WildDash 2 semantic labels."""
    raw_to_index = _REGISTRY["wilddash2"]["raw_to_index"]
    if not raw_to_index:
        raise RuntimeError(
            "WildDash 2 taxonomy is unregistered. Call register_wilddash2_metadata() first."
        )
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int64)
    segment_ids = rgb[..., 0] + 256 * rgb[..., 1] + 256 * 256 * rgb[..., 2]
    out = np.full(segment_ids.shape, IGNORE_INDEX, dtype=np.int64)
    for segment in segments_info:
        raw_id = int(segment.get("category_id", -1))
        index = raw_to_index.get(raw_id)
        if index is not None:
            out[segment_ids == int(segment["id"])] = index
    return out



def _read_coconut_xlarge_panoptic(path: Path, segments_info: Sequence[dict]) -> np.ndarray:
    """Decode a COCONut-XL COCO-panoptic RGB PNG to compact semantic labels."""
    raw_to_index = _REGISTRY["coconut_xlarge"]["raw_to_index"]
    if not raw_to_index:
        raise RuntimeError(
            "COCONut-XL taxonomy is unregistered. Call register_coconut_xlarge_metadata() first."
        )
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int64)
    segment_ids = rgb[..., 0] + 256 * rgb[..., 1] + 256 * 256 * rgb[..., 2]
    out = np.full(segment_ids.shape, IGNORE_INDEX, dtype=np.int64)
    for segment in segments_info:
        raw_id = int(segment.get("category_id", -1))
        index = raw_to_index.get(raw_id)
        if index is not None:
            out[segment_ids == int(segment["id"])] = index
    return out

def _decode_compressed_coco_counts(data: bytes) -> list[int]:
    """Decode pycocotools' compact signed/delta RLE count string."""
    counts: list[int] = []
    position = 0
    while position < len(data):
        value = 0
        shift = 0
        while True:
            if position >= len(data):
                raise ValueError("Truncated compressed COCO RLE")
            code = data[position] - 48
            position += 1
            if not 0 <= code <= 0x3F:
                raise ValueError("Invalid compressed COCO RLE character")
            value |= (code & 0x1F) << shift
            shift += 5
            if not (code & 0x20):
                if code & 0x10:
                    value |= -1 << shift
                break
        if len(counts) > 2:
            value += counts[-2]
        counts.append(value)
    return counts


def decode_coco_rle(rle: dict) -> np.ndarray:
    """Decode a COCO RLE (including NuImages' compressed-count strings)."""
    try:
        height, width = (int(value) for value in rle["size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid COCO RLE size: {rle!r}") from exc
    encoded = rle.get("counts")
    if isinstance(encoded, (list, tuple)):
        counts = [int(value) for value in encoded]
    elif isinstance(encoded, (str, bytes)):
        raw = encoded.encode("ascii") if isinstance(encoded, str) else encoded
        candidates = [raw]
        try:
            unwrapped = base64.b64decode(raw + b"=" * (-len(raw) % 4), validate=True)
        except ValueError:
            pass
        else:
            if unwrapped:
                candidates.append(unwrapped)
        counts = None
        for data in candidates:
            try:
                candidate_counts = _decode_compressed_coco_counts(data)
            except ValueError:
                continue
            if not any(count < 0 for count in candidate_counts) and sum(candidate_counts) == height * width:
                counts = candidate_counts
                break
        if counts is None:
            raise ValueError(f"Invalid COCO RLE run lengths for {height}x{width} mask")
    else:
        raise ValueError("COCO RLE counts must be a list, string, or bytes")

    if any(count < 0 for count in counts) or sum(counts) != height * width:
        raise ValueError(f"Invalid COCO RLE run lengths for {height}x{width} mask")
    flat = np.zeros(height * width, dtype=bool)
    start = 0
    for run_index, count in enumerate(counts):
        end = start + count
        if run_index % 2 == 1:
            flat[start:end] = True
        start = end
    return flat.reshape((height, width), order="F")


def _read_nuimages_rles(annotations: Sequence[dict]) -> np.ndarray:
    """Render NuImages surface RLEs first, then foreground object RLEs."""
    token_to_index = _REGISTRY["nuimages"].get("token_to_index") or {}
    ignore_tokens = _REGISTRY["nuimages"].get("ignore_tokens") or frozenset()
    if not token_to_index:
        raise RuntimeError("NuImages categories are not registered. Call register_nuimages_metadata() first.")
    out: np.ndarray | None = None
    for annotation_type in ("surface", "object"):
        for annotation in annotations:
            if annotation.get("_nuimages_annotation_type") != annotation_type:
                continue
            rle = annotation.get("mask")
            if rle is None:
                continue
            mask = decode_coco_rle(rle)
            if out is None:
                out = np.full(mask.shape, IGNORE_INDEX, dtype=np.int64)
            elif out.shape != mask.shape:
                raise ValueError("NuImages annotations for one image have inconsistent mask dimensions")
            category_token = str(annotation.get("category_token") or "")
            class_index = token_to_index.get(category_token)
            if class_index is not None:
                out[mask] = class_index
            elif category_token in ignore_tokens:
                out[mask] = IGNORE_INDEX
    if out is None:
        raise ValueError("NuImages frame has no decodable segmentation mask")
    return out


_READERS = {
    "trainids": _read_trainids,
    "cityscapes_labelids": _read_cityscapes_labelids,
    "labelids": _read_labelids,
    "voc_rgb": _read_voc,
    "goose_labelids": _read_goose_labelids,
    "foodseg103_labelids": _read_foodseg103_labelids,
    "trans10k_labelids": _read_trans10k_labelids,
    "railsem19_labelids": _read_railsem19_labelids,
    "waymo_panoptic": _read_waymo_panoptic,
    "hypersim_panoptic": _read_hypersim_panoptic,
    "pascal_context_labelids": _read_pascal_context_labelids,
    "vipseg_panomask": _read_vipseg_panomask,
    "mapillary_labels": _read_mapillary_labels,
    "idd_level3_polygons": _read_idd_level3_polygons,
}


def read_label_array(
    path: Path,
    dataset: str,
    segments_info: Sequence[dict] | None = None,
    annotations: Sequence[dict] | None = None,
) -> np.ndarray:
    if _REGISTRY[dataset]["kind"] == "wilddash2_panoptic":
        if segments_info is None:
            raise ValueError("WildDash 2 panoptic decoding requires segments_info from panoptic.json")
        return _read_wilddash2_panoptic(Path(path), segments_info)
    if _REGISTRY[dataset]["kind"] == "coconut_xlarge_panoptic":
        if segments_info is None:
            raise ValueError("COCONut-XL panoptic decoding requires segments_info from panoptic_train2017.json")
        return _read_coconut_xlarge_panoptic(Path(path), segments_info)
    if _REGISTRY[dataset]["kind"] == "nuimages_rles":
        if annotations is None:
            raise ValueError("NuImages RLE decoding requires per-frame object/surface annotations")
        return _read_nuimages_rles(annotations)
    return _READERS[_REGISTRY[dataset]["kind"]](Path(path))


def present_classes(arr: np.ndarray, dataset: str) -> list[int]:
    """Selectable foreground class indices present (excl. ignore + background)."""
    n = len(class_names(dataset))
    bg = background_index(dataset)
    return sorted(
        int(v) for v in np.unique(arr)
        if 0 <= int(v) < n and int(v) != IGNORE_INDEX and int(v) != bg
    )


def select_subset(present: Sequence[int], rng: random.Random,
                  weights: Optional[Sequence[float]] = None,
                  strict_subset: bool = True) -> list[int]:
    """The M-rule. N==1 -> that one class. N>1 -> a proper subset of size
    1..N-1 (strict_subset), else 1..N. Optional per-index ``weights`` bias which
    classes are chosen (e.g. inverse-frequency to favor rare classes)."""
    present = list(present)
    n = len(present)
    if n == 0:
        return []
    if n == 1:
        return present
    hi = n - 1 if strict_subset else n
    m = rng.randint(1, hi)
    if weights is None:
        return rng.sample(present, m)
    return _weighted_sample(present, [weights[c] for c in present], m, rng)


def _weighted_sample(items, weights, k, rng):
    """k items without replacement, prob proportional to weights."""
    items, weights = list(items), [max(float(w), 1e-12) for w in weights]
    chosen = []
    for _ in range(min(k, len(items))):
        thresh = rng.uniform(0, sum(weights))
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if thresh <= acc:
                chosen.append(items.pop(i))
                weights.pop(i)
                break
    return chosen


if __name__ == "__main__":
    # Synthetic round-trip checks (no datasets on disk needed).
    print("datasets:", list(_REGISTRY))
    for ds in _REGISTRY:
        print(f"  {ds}: {len(class_names(ds))} classes, bg={background_index(ds)}")

    # trainids
    ct = np.array([[0, 1, 255], [13, 18, 0]], dtype=np.int64)
    print("\ncityscapes present:", present_classes(ct, "cityscapes"),
          "->", [CITYSCAPES_CLASSES[i] for i in present_classes(ct, "cityscapes")])

    # ade: raw 0 ignore, 1->idx0 wall, 21->idx20 car
    ade = np.array([[0, 1, 21], [150, 13, 0]], dtype=np.int64)
    em = ade - 1; em[ade == 0] = 255  # same transform _read_labelids applies
    print("ade present:", present_classes(em, "ade20k"),
          "->", [ADE20K_CLASSES[i] for i in present_classes(em, "ade20k")])

    # voc colormap invert: build an rgb mask from known ids then read back
    ids = np.array([[0, 7, 15], [255, 20, 0]], dtype=np.int64)  # bg, car, person, void, train, bg
    rgb = np.zeros((*ids.shape, 3), dtype=np.uint8)
    for i in range(len(VOC_CLASSES)):
        rgb[ids == i] = VOC_COLORMAP[i]
    rgb[ids == 255] = VOC_VOID_RGB
    from PIL import Image as _I
    import tempfile, os
    p = Path(tempfile.gettempdir()) / "_voc_test.png"
    _I.fromarray(rgb, "RGB").save(p)
    dec = _read_voc(p)
    os.remove(p)
    print("voc decode matches ids:", bool((dec == ids).all()),
          "present:", present_classes(dec, "voc"),
          "->", [VOC_CLASSES[i] for i in present_classes(dec, "voc")], "(background 0 excluded)")

    # subset M-rule
    rng = random.Random(0)
    print("\nM-rule samples:")
    for present in ([5], [3, 7], list(range(10))):
        picks = [len(select_subset(present, rng)) for _ in range(200)]
        print(f"  present N={len(present)}: M in [{min(picks)}, {max(picks)}]"
              + ("  (N==1 -> always 1)" if len(present) == 1 else "  (strict: max should be N-1)"))
