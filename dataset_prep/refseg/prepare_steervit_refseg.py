"""Prepare SteerViT referential-segmentation data in the training JSONL format.

The output mirrors the prepared image-to-image datasets used by the trainer:

  OUT_ROOT/
    metadata.jsonl
    conditioning_images/
    target_images/

Each metadata row points at the RGB input image and a prompt-colored binary
target mask. The foreground color is either fixed or sampled per row, and the
prompt carries the matching color name/RGB value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm


DEFAULT_HF_DATASETS = ("JonaRuthardt/SteerViT",)
REFCOCOG_UMD_SPLITS = frozenset(("train", "val", "test"))

# Source-image roots default to the repo's datasets_archives/refseg (same
# convention as prepare_refseg_eval.py): visual_genome.tar and the images of the
# shared Vistas ZIP (datasets_archives/shared_seg/mapillary.tar) extracted there.
REPO_ROOT = Path(__file__).resolve().parents[2]
REFSEG_ROOT = REPO_ROOT / "datasets_archives/refseg"

sys.path.insert(0, str(REPO_ROOT))
from common import coco_eval_holdouts  # noqa: E402
from common.colors import BLACK_TOKEN, format_rgb  # noqa: E402

# Fallback if the shared palette cannot be loaded.
_FALLBACK_COLOR_SPECS = (
    "white:255,255,255",
    "red:255,0,0",
    "green:0,255,0",
    "blue:0,0,255",
    "yellow:255,255,0",
    "cyan:0,255,255",
    "magenta:255,0,255",
    "orange:255,128,0",
)


def _load_default_color_specs() -> tuple[str, ...]:
    """Default foreground colors = the shared 37 well-separated named colors
    (common/refseg/palette_named.json), so training and evaluation paint the
    referred region from the same palette. Falls back to a small fixed set."""
    palette = Path(__file__).resolve().parents[2] / "common" / "refseg" / "palette_named.json"
    try:
        entries = json.loads(palette.read_text(encoding="utf-8"))
        specs = tuple(
            f"{c['name']}:{c['rgb'][0]},{c['rgb'][1]},{c['rgb'][2]}"
            for c in entries
            if sum(c["rgb"]) > 0  # black is reserved for background
        )
        if specs:
            return specs
    except (OSError, ValueError, KeyError, IndexError):
        pass
    return _FALLBACK_COLOR_SPECS


DEFAULT_COLOR_SPECS = _load_default_color_specs()

TASK_NAME = "referring_segmentation"
TASK_TAG = "REFERRING_SEGMENTATION"

REFSEG_PROMPT_TEMPLATES = (
    f"[TASK: {TASK_TAG}] Segment the region referred to by: {{expression}} "
    "Output only a binary segmentation mask with the referred region in {color_name} "
    f"RGB {{foreground_rgb}} and all background pixels black RGB {BLACK_TOKEN}.",
    f"[TASK: {TASK_TAG}] Given the referring expression {{expression}}, "
    "produce only the mask image: paint the target region {color_name} "
    f"({{foreground_rgb}}), and leave everything else black ({BLACK_TOKEN}).",
    f"[TASK: {TASK_TAG}] Create a binary mask for {{expression}} "
    "Use {color_name} for the referred object or region and "
    "black for the background. Output only the mask image.",
)


@dataclass(frozen=True)
class ColorSpec:
    name: str
    rgb: tuple[int, int, int]


@dataclass(frozen=True)
class ImageRoots:
    mscoco: Path | None = None
    visual_genome: Path | None = None
    mapillary: Path | None = None
    lisa: Path | None = None


def split_csv(value: str | None, default: Sequence[str] = ()) -> list[str]:
    if value is None:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_refcocog_umd_splits(value: str | None) -> set[str] | None:
    """Parse an optional comma-separated set of native RefCOCOg UMD splits."""
    if value is None:
        return None
    splits = {split.lower() for split in split_csv(value)}
    if not splits:
        raise ValueError("--refcocog-umd-splits must name at least one split.")
    invalid = sorted(splits - REFCOCOG_UMD_SPLITS)
    if invalid:
        valid = ", ".join(sorted(REFCOCOG_UMD_SPLITS))
        raise ValueError(
            f"Unsupported RefCOCOg UMD split(s): {', '.join(invalid)}. Expected: {valid}."
        )
    return splits


def load_refcocog_umd_splits(data_dir: Path) -> dict[str, str]:
    """Return ``ref_id -> native UMD split`` from the official refer release.

    The SteerViT HF record's ``id`` is the RefCOCOg reference ID. Keeping the
    mapping separate from the HF train/validation partitions lets callers avoid
    train/eval leakage with an explicit native UMD-split filter.
    """
    refs_path = data_dir / "refs(umd).p"
    if not refs_path.is_file():
        raise FileNotFoundError(
            f"RefCOCOg UMD split filtering needs {refs_path}. "
            "Download/extract the official refcocog annotations or pass "
            "--refcocog-umd-dir."
        )

    with refs_path.open("rb") as handle:
        try:
            refs = pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0)
            refs = pickle.load(handle, encoding="latin1")

    split_by_ref_id: dict[str, str] = {}
    for ref in refs:
        ref_id = ref.get("ref_id")
        split = str(ref.get("split", "")).lower()
        if ref_id is None or split not in REFCOCOG_UMD_SPLITS:
            raise ValueError(f"Invalid RefCOCOg UMD reference in {refs_path}: {ref!r}")
        key = str(ref_id)
        previous = split_by_ref_id.setdefault(key, split)
        if previous != split:
            raise ValueError(f"RefCOCOg UMD ref_id {key!r} has conflicting splits: {previous!r}, {split!r}")
    if not split_by_ref_id:
        raise ValueError(f"No RefCOCOg UMD references found in {refs_path}.")
    return split_by_ref_id


def refcocog_umd_split_for_item(item: Mapping[str, Any], split_by_ref_id: Mapping[str, str] | None) -> str | None:
    """Look up a SteerViT RefCOCOg row's native UMD split, if available."""
    if split_by_ref_id is None or str(item.get("dataset", "")).lower() != "refcocog":
        return None
    for field in ("ref_id", "id"):
        value = item.get(field)
        if value is not None:
            return split_by_ref_id.get(str(value))
    return None


def safe_name(*parts: Any, max_len: int = 180) -> str:
    raw = "__".join(str(part) for part in parts if part is not None and str(part))
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in raw)
    safe = "_".join(part for part in safe.split("_") if part)
    if len(safe) <= max_len:
        return safe or "sample"
    digest = hashlib.sha1(safe.encode("utf-8")).hexdigest()[:12]
    return f"{safe[: max_len - 13]}_{digest}"


def parse_color_spec(value: str) -> ColorSpec:
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid color {value!r}. Expected NAME:R,G,B, e.g. red:255,0,0."
        )
    name, rgb_text = value.split(":", 1)
    try:
        rgb = tuple(int(part.strip()) for part in rgb_text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid RGB value in {value!r}") from exc
    if len(rgb) != 3 or any(channel < 0 or channel > 255 for channel in rgb):
        raise argparse.ArgumentTypeError(f"RGB values must be three integers in [0, 255]: {value!r}")
    if sum(rgb) == 0:
        raise argparse.ArgumentTypeError("Foreground color cannot be black; black is reserved for background.")
    return ColorSpec(name.strip().lower(), rgb)  # type: ignore[arg-type]


def parse_color_specs(values: Iterable[str]) -> list[ColorSpec]:
    colors = [parse_color_spec(value) for value in values]
    if not colors:
        raise ValueError("At least one foreground color is required.")
    return colors


def parse_color_list(value: str) -> list[ColorSpec]:
    return parse_color_specs(part.strip() for part in value.split(";") if part.strip())


def get_coco_path(coco_path: Path, img_id: str) -> Path:
    if not img_id.endswith(".jpg"):
        img_id += ".jpg"
    image_path = coco_path / img_id.split("_")[1] / img_id
    if not image_path.is_file():
        raise FileNotFoundError(f"COCO image not found: {image_path}")
    return image_path


def get_visual_genome_path(vg_path: Path, img_id: str) -> Path:
    if not img_id.endswith(".jpg"):
        img_id += ".jpg"
    candidates = (vg_path / "VG_100K" / img_id, vg_path / "VG_100K_2" / img_id)
    for image_path in candidates:
        if image_path.is_file():
            return image_path
    raise FileNotFoundError(f"Visual Genome image not found: {candidates[0]} or {candidates[1]}")


def get_mapillary_path(mapillary_path: Path, img_id: str) -> Path:
    if not img_id.endswith(".jpg"):
        img_id += ".jpg"
    # Resolve against either a flat dir (<root>/<id>.jpg) or the native Mapillary
    # Vistas tree (<root>[/mapillary_vistasv2]/{training,validation,testing}/images/).
    candidates = [mapillary_path / img_id]
    for base in (mapillary_path, mapillary_path / "mapillary_vistasv2"):
        for split_dir in ("training", "validation", "testing"):
            candidates.append(base / split_dir / "images" / img_id)
    for image_path in candidates:
        if image_path.is_file():
            return image_path
    raise FileNotFoundError(
        f"Mapillary image {img_id!r} not found under {mapillary_path} "
        "(tried flat + Vistas split layouts)."
    )


def get_lisa_path(lisa_path: Path, split: str, img_id: str) -> Path:
    split_name = "val" if split == "validation" else split
    candidates = (lisa_path / split_name / img_id, lisa_path / img_id)
    for image_path in candidates:
        if image_path.is_file():
            return image_path
    raise FileNotFoundError(f"LISA image not found: {list(candidates)}")


def resolve_local_image_path(item: dict[str, Any], split: str, roots: ImageRoots) -> Path:
    dataset_name = str(item.get("dataset", ""))
    image_id = str(item.get("image_id", ""))
    if not image_id:
        raise FileNotFoundError("Row has no image_id and no embedded image.")

    if image_id.startswith("COCO"):
        if roots.mscoco is None:
            raise FileNotFoundError("COCO row requires --mscoco.")
        return get_coco_path(roots.mscoco, image_id)
    if "visual genome" in dataset_name.lower():
        if roots.visual_genome is None:
            raise FileNotFoundError("Visual Genome row requires --visual-genome.")
        return get_visual_genome_path(roots.visual_genome, image_id)
    if "mapillary" in dataset_name.lower():
        if roots.mapillary is None:
            raise FileNotFoundError("Mapillary row requires --mapillary.")
        return get_mapillary_path(roots.mapillary, image_id)
    if "lisa" in dataset_name.lower():
        if roots.lisa is None:
            raise FileNotFoundError("LISA row requires --lisa.")
        return get_lisa_path(roots.lisa, split, image_id)

    raise NotImplementedError(f"Unknown image source for dataset={dataset_name!r}, image_id={image_id!r}")


def image_from_hf_value(value: Any, mode: str = "RGB") -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert(mode)
    if isinstance(value, np.ndarray):
        return Image.fromarray(value).convert(mode)
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        if path.is_file():
            return Image.open(path).convert(mode)
        return None
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"])).convert(mode)
        if value.get("path"):
            path = Path(value["path"]).expanduser()
            if path.is_file():
                return Image.open(path).convert(mode)
    return None


def load_source_image(item: dict[str, Any], split: str, roots: ImageRoots) -> Image.Image:
    for key in ("image", "img", "rgb"):
        if key in item:
            image = image_from_hf_value(item.get(key), mode="RGB")
            if image is not None:
                return image

    for key in ("image_path", "path", "file_name"):
        raw_path = item.get(key)
        if raw_path:
            image = image_from_hf_value(raw_path, mode="RGB")
            if image is not None:
                return image

    image_path = resolve_local_image_path(item, split, roots)
    return Image.open(image_path).convert("RGB")


def load_mask(item: dict[str, Any]) -> Image.Image:
    for key in ("mask", "segmentation_mask", "target_mask"):
        if key in item:
            mask = image_from_hf_value(item.get(key), mode="L")
            if mask is not None:
                return mask
    raise KeyError("Row has no readable mask column. Tried: mask, segmentation_mask, target_mask.")


def normalize_expression(item: dict[str, Any]) -> str:
    expression = raw_expression(item)
    if not expression:
        raise ValueError("Row has no referential expression.")
    if "lisa" in str(item.get("dataset", "")).lower():
        expression = expression.split("?")[0].strip() + "?"
    return expression


def raw_expression(item: dict[str, Any]) -> str:
    return str(
        item.get("referential_expression")
        or item.get("referring_expression")
        or item.get("expression")
        or item.get("prompt")
        or ""
    ).strip()


def color_binary_mask(mask: Image.Image, foreground_rgb: tuple[int, int, int]) -> Image.Image:
    label = np.asarray(mask.convert("L"))
    foreground = label > 0
    rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    rgb[foreground] = np.asarray(foreground_rgb, dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def save_rgb_png(image: Image.Image, dst: Path) -> None:
    # Atomic write: a shared conditioning image may be written by several worker
    # threads racing on the same path (many rows reference one source image).
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp.{os.getpid()}.{threading.get_ident()}.png")
    image.convert("RGB").save(tmp, format="PNG")
    os.replace(tmp, dst)


def build_prompt(expression: str, color: ColorSpec, rng: random.Random) -> str:
    quoted_expression = json.dumps(expression, ensure_ascii=True)
    return rng.choice(REFSEG_PROMPT_TEMPLATES).format(
        expression=quoted_expression,
        color_name=color.name,
        foreground_rgb=format_rgb(color.rgb),
    )


def load_steervit_dataset(dataset_names: Sequence[str], splits: Sequence[str]):
    try:
        from datasets import Image as HFImage
        from datasets import concatenate_datasets, load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The HuggingFace 'datasets' package is required. Install the project requirements "
            "or run in the environment that contains datasets."
        ) from exc

    datasets = []
    for dataset_name in dataset_names:
        for split in splits:
            dataset = load_dataset(dataset_name, split=split)
            dataset = dataset.add_column("_steervit_split", [split] * len(dataset))
            datasets.append(dataset)
    dataset = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
    for column in ("image", "img", "rgb", "mask", "segmentation_mask", "target_mask"):
        if column in dataset.column_names:
            try:
                dataset = dataset.cast_column(column, HFImage(decode=False))
            except (TypeError, ValueError):
                pass
    return dataset


def row_source_split(item: dict[str, Any], default_split: str) -> str:
    return str(item.get("_steervit_split") or item.get("split") or default_split)


# Mask override: masks of selected source datasets (values of the `dataset`
# column) come from a second HF release that re-annotates them; other sources
# keep the base release's masks. Rows, order, ids, colors and templates still
# come from the base release. Override rows are matched on (split, dataset, id)
# and must agree on these fields; a base row with no override row is dropped.
OVERRIDE_MATCH_FIELDS = ("image_id", "referential_expression", "bbox", "height", "width")


@dataclass
class MaskOverride:
    hf_dataset: str
    sources: frozenset[str]
    dataset: Any
    index_by_key: dict[tuple[str, str, str], int]
    meta: dict[str, list[Any]]
    revisions: list[str]


def dataset_revisions(dataset) -> list[str]:
    """Snapshot directories of the cache files backing a loaded HF dataset."""
    return sorted({str(Path(entry["filename"]).parent) for entry in getattr(dataset, "cache_files", [])})


def load_mask_override(hf_dataset: str, sources: Iterable[str], splits: Sequence[str]) -> MaskOverride:
    sources = frozenset(sources)
    if not sources:
        raise ValueError("--mask-override-sources must name at least one source dataset.")
    dataset = load_steervit_dataset([hf_dataset], splits)
    columns = ("_steervit_split", "dataset", "id", *OVERRIDE_MATCH_FIELDS)
    meta = {column: list(dataset[column]) for column in columns}
    present = set(meta["dataset"])
    missing = sorted(sources - present)
    if missing:
        raise ValueError(f"{hf_dataset} has no rows for override source(s) {missing}; it has {sorted(present)}.")
    index_by_key: dict[tuple[str, str, str], int] = {}
    for index, (split, name, row_id) in enumerate(zip(meta["_steervit_split"], meta["dataset"], meta["id"])):
        if name not in sources:
            continue
        key = (str(split), str(name), str(row_id))
        if key in index_by_key:
            raise ValueError(f"{hf_dataset} has duplicate rows for {key}.")
        index_by_key[key] = index
    return MaskOverride(hf_dataset, sources, dataset, index_by_key, meta, dataset_revisions(dataset))


def override_index_for_item(item: Mapping[str, Any], split: str, override: MaskOverride) -> int | None:
    """Index of this base row's override row, or None if the override release has none.

    Raises if the override row's metadata differs from the base row: a mismatch
    means the two releases are not aligned and masks would land on wrong rows.
    """
    key = (split, str(item.get("dataset", "")), str(item.get("id", "")))
    index = override.index_by_key.get(key)
    if index is None:
        return None
    for field in OVERRIDE_MATCH_FIELDS:
        base_value, override_value = item.get(field), override.meta[field][index]
        if field == "bbox":
            base_value = [float(value) for value in base_value or ()]
            override_value = [float(value) for value in override_value or ()]
        if base_value != override_value:
            raise ValueError(
                f"Mask override row {key} differs from the base row in {field}: "
                f"{base_value!r} vs {override_value!r}."
            )
    return index

COCO_BACKED_DATASET_NAMES = frozenset(("refcoco", "refcoco+", "refcocog", "grefcoco", "lvis"))


def item_evaluation_holdout(
    item: Mapping[str, Any], holdouts: Mapping[str, Iterable[str]]
) -> tuple[str | None, str | None, tuple[str, ...]]:
    """Return this row's canonical COCO source image and evaluation matches.

    A bare numeric ID is treated as COCO only for a known COCO-backed source;
    a Visual Genome integer image ID must not be mistaken for a COCO image.
    """
    dataset_name = str(item.get("dataset", "")).lower()
    known_coco_source = dataset_name in COCO_BACKED_DATASET_NAMES
    for field in ("image_id", "file_name", "image_path", "path"):
        value = item.get(field)
        if value is None or value == "":
            continue
        image_id = coco_eval_holdouts.coco_image_id(
            value, allow_unpadded_numeric_id=known_coco_source
        )
        if image_id is None:
            continue
        if known_coco_source or str(value).upper().startswith("COCO_"):
            reasons = tuple(name for name, ids in holdouts.items() if image_id in ids)
            return str(value), image_id, reasons
    return None, None, ()



def row_matches_filters(
    item: dict[str, Any],
    include_datasets: set[str] | None,
    min_width: int | None,
    min_height: int | None,
    max_prompt_length: int | None,
    ignore_coco: bool,
    refcocog_umd_splits: set[str] | None = None,
    refcocog_umd_split_by_sample: Mapping[str, str] | None = None,
) -> bool:
    dataset_name = str(item.get("dataset", ""))
    if include_datasets is not None and dataset_name not in include_datasets:
        return False
    if ignore_coco and str(item.get("image_id", "")).startswith("COCO"):
        return False
    if refcocog_umd_splits is not None and dataset_name.lower() == "refcocog":
        native_split = refcocog_umd_split_for_item(item, refcocog_umd_split_by_sample)
        if native_split not in refcocog_umd_splits:
            return False
    if min_width is not None and item.get("width") is not None and int(item["width"]) < min_width:
        return False
    if min_height is not None and item.get("height") is not None and int(item["height"]) < min_height:
        return False
    if max_prompt_length is not None and len(raw_expression(item)) > max_prompt_length:
        return False
    return True


def choose_row_color(colors: Sequence[ColorSpec], randomize: bool, rng: random.Random, index: int) -> ColorSpec:
    if not randomize:
        return colors[0]
    return colors[rng.randrange(len(colors))]


# One source image carries many referring expressions, so each source image is
# written once under a stable per-image name and shared by its rows. The cache
# maps that name -> source (w, h); lock-guarded because rows are processed by a
# ThreadPoolExecutor.
_COND_LOCK = threading.Lock()
_COND_SIZES: dict[str, tuple[int, int]] = {}


def conditioning_name(item: dict[str, Any], row_id: str) -> str:
    """Stable conditioning filename keyed by the source image (dataset+image_id),
    so all rows referencing one image share a single file. Falls back to the
    per-row id when no image_id is available (no dedup possible)."""
    image_id = str(item.get("image_id", "")).strip()
    if image_id:
        return f"{safe_name('steervit', item.get('dataset'), image_id)}_input.png"
    return f"{row_id}_input.png"


def make_row_id(item: dict[str, Any], split: str, index: int) -> str:
    raw_id = (
        item.get("id")
        or item.get("ref_id")
        or item.get("ann_id")
        or item.get("image_id")
        or f"row_{index:08d}"
    )
    dataset_name = item.get("dataset", "steervit")
    return safe_name("steervit", split, dataset_name, raw_id, index)


def process_item(
    item: dict[str, Any],
    index: int,
    split: str,
    roots: ImageRoots,
    out_root: Path,
    colors: Sequence[ColorSpec],
    randomize_colors: bool,
    seed: int,
    refcocog_umd_split_by_sample: Mapping[str, str] | None,
    mask_item: Mapping[str, Any] | None = None,
    mask_source: str | None = None,
) -> tuple[str, dict[str, Any] | None]:
    rng = random.Random(f"{seed}:{split}:{index}:{item.get('image_id', '')}:{item.get('referential_expression', '')}")
    row_id = make_row_id(item, split, index)
    color = choose_row_color(colors, randomize_colors, rng, index)
    native_refcocog_umd_split = refcocog_umd_split_for_item(item, refcocog_umd_split_by_sample)

    input_name = conditioning_name(item, row_id)
    cond_path = out_root / "conditioning_images" / input_name
    try:
        expression = normalize_expression(item)
        mask = load_mask(dict(mask_item) if mask_item is not None else item)
        # Resolve the source image size, writing the shared conditioning image once.
        with _COND_LOCK:
            src_size = _COND_SIZES.get(input_name)
        if src_size is None:
            if cond_path.exists():  # already written (sibling row or earlier run)
                with Image.open(cond_path) as existing:
                    src_size = existing.size
            else:
                image = load_source_image(item, split, roots)
                src_size = image.size
                save_rgb_png(image, cond_path)
            with _COND_LOCK:
                _COND_SIZES[input_name] = src_size
    except (FileNotFoundError, KeyError, NotImplementedError, OSError, ValueError) as exc:
        return f"skipped:{type(exc).__name__}", None

    if src_size != mask.size:
        return "skipped:geometry", None

    target = color_binary_mask(mask, color.rgb)
    target_name = f"{row_id}_mask.png"
    target_path = out_root / "target_images" / target_name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target.save(target_path)

    height, width = target.height, target.width
    bbox = item.get("bbox") if isinstance(item.get("bbox"), list) and len(item["bbox"]) == 4 else [-1, -1, -1, -1]
    row = {
        "conditioning_image": f"conditioning_images/{input_name}",
        "image": f"target_images/{target_name}",
        "text": build_prompt(expression, color, rng),
        "task": TASK_NAME,
        "source_dataset": mask_source or "JonaRuthardt/SteerViT",
        "preprocess_target": "nearest",
        "id": row_id,
        "split": split,
        "metadata": {
            "dataset": item.get("dataset"),
            "image_id": item.get("image_id"),
            "referential_expression": expression,
            "bbox": bbox,
            "height": int(item.get("height") or height),
            "width": int(item.get("width") or width),
            "foreground_color": {"name": color.name, "rgb": list(color.rgb)},
            "background_color": {"name": "black", "rgb": [0, 0, 0]},
        },
    }
    if native_refcocog_umd_split is not None:
        row["metadata"]["refcocog_umd_split"] = native_refcocog_umd_split
    return "written", row


def process_index(
    index: int,
    dataset,
    default_split: str,
    roots: ImageRoots,
    out_root: Path,
    colors: Sequence[ColorSpec],
    randomize_colors: bool,
    seed: int,
    refcocog_umd_split_by_sample: Mapping[str, str] | None,
    override: MaskOverride | None = None,
    override_index_by_row: Mapping[int, int] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    item = dict(dataset[index])
    mask_item = mask_source = None
    if override is not None and str(item.get("dataset", "")) in override.sources:
        # Filtering already dropped rows without an override row, so a miss here is a bug.
        mask_item = override.dataset[override_index_by_row[index]]
        mask_source = override.hf_dataset
    return process_item(
        item,
        index,
        row_source_split(item, default_split),
        roots,
        out_root,
        colors,
        randomize_colors,
        seed,
        refcocog_umd_split_by_sample,
        mask_item,
        mask_source,
    )


def write_rows(
    dataset,
    indices: Sequence[int],
    default_split: str,
    roots: ImageRoots,
    out_root: Path,
    colors: Sequence[ColorSpec],
    randomize_colors: bool,
    seed: int,
    num_workers: int,
    refcocog_umd_split_by_sample: Mapping[str, str] | None,
    override: MaskOverride | None = None,
    override_index_by_row: Mapping[int, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    metadata_rows = []
    skipped: dict[str, int] = {}

    def handle_result(status: str, row: dict[str, Any] | None) -> None:
        if row is not None:
            metadata_rows.append(row)
            return
        skipped[status] = skipped.get(status, 0) + 1

    if num_workers <= 1:
        for index in tqdm(indices, desc="Writing SteerViT refseg"):
            handle_result(
                *process_index(
                    index,
                    dataset,
                    default_split,
                    roots,
                    out_root,
                    colors,
                    randomize_colors,
                    seed,
                    refcocog_umd_split_by_sample,
                    override,
                    override_index_by_row,
                )
            )
        return metadata_rows, skipped

    batch_size = max(num_workers * 64, 512)
    progress = tqdm(total=len(indices), desc=f"Writing SteerViT refseg ({num_workers} workers)")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for batch_start in range(0, len(indices), batch_size):
            batch = indices[batch_start : batch_start + batch_size]
            results = executor.map(
                process_index,
                batch,
                [dataset] * len(batch),
                [default_split] * len(batch),
                [roots] * len(batch),
                [out_root] * len(batch),
                [colors] * len(batch),
                [randomize_colors] * len(batch),
                [seed] * len(batch),
                [refcocog_umd_split_by_sample] * len(batch),
                [override] * len(batch),
                [override_index_by_row] * len(batch),
            )
            for status, row in results:
                handle_result(status, row)
                progress.update(1)
    progress.close()
    return metadata_rows, skipped


def write_config(
    args: argparse.Namespace,
    colors: Sequence[ColorSpec],
    out_root: Path,
    mask_override: dict[str, Any] | None = None,
) -> None:
    splits = split_csv(args.split)
    config = {
        "hf_datasets": split_csv(args.hf_datasets, DEFAULT_HF_DATASETS),
        "split": args.split,
        "splits": splits,
        "include_datasets": split_csv(args.include_datasets) if args.include_datasets else None,
        "min_width": args.min_width,
        "min_height": args.min_height,
        "max_prompt_length": args.max_prompt_length,
        "ignore_coco": args.ignore_coco,
        "refcocog_umd_splits": sorted(parse_refcocog_umd_splits(args.refcocog_umd_splits) or ()),
        "evaluation_image_holdout": {
            "enabled": not args.allow_eval_image_overlap,
            "refcocog_umd_dir": str(args.refcocog_umd_dir),
        },
        "refcocog_umd_dir": str(args.refcocog_umd_dir) if args.refcocog_umd_splits else None,
        "limit": args.limit,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "randomize_colors": not args.no_random_colors,
        "colors": [{"name": color.name, "rgb": list(color.rgb)} for color in colors],
        "prompt_templates": list(REFSEG_PROMPT_TEMPLATES),
    }
    if mask_override is not None:  # absent otherwise, so base-only configs keep their keys
        config["mask_override"] = mask_override
    with (out_root / "steervit_refseg_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument(
        "--hf-datasets",
        default=",".join(DEFAULT_HF_DATASETS),
        help="Comma-separated HuggingFace dataset names to load and concatenate.",
    )
    parser.add_argument(
        "--split",
        "--splits",
        dest="split",
        default="train",
        help="Comma-separated HF split names to load and merge, e.g. train or train,validation.",
    )
    parser.add_argument(
        "--include-datasets",
        default=None,
        help="Comma-separated values from the HF 'dataset' column to keep. Defaults to all.",
    )
    parser.add_argument("--min-width", type=int, default=None, help="Keep rows with width >= this value.")
    parser.add_argument("--min-height", type=int, default=None, help="Keep rows with height >= this value.")
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=None,
        help="Keep rows whose referring expression is at most this many characters long.",
    )
    parser.add_argument(
        "--ignore-coco",
        action="store_true",
        help="Skip rows whose HF image_id starts with 'COCO'.",
    )
    parser.add_argument(
        "--refcocog-umd-splits",
        default=None,
        help=(
            "Comma-separated native RefCOCOg UMD split(s) to retain, e.g. train. "
            "This filters only dataset=refcocog rows and requires refs(umd).p."
        ),
    )
    parser.add_argument(
        "--refcocog-umd-dir",
        type=Path,
        default=REFSEG_ROOT / "refcocog",
        help="Directory containing official RefCOCOg refs(umd).p (used with --refcocog-umd-splits).",
    )
    parser.add_argument(
        "--allow-eval-image-overlap",
        action="store_true",
        help="Disable COCO val2017 and RefCOCOg UMD validation image exclusion. Only use for an explicitly non-evaluation run.",
    )

    parser.add_argument("--limit", type=int, default=None, help="Maximum number of written rows after filtering.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of writer threads for image/mask decode and PNG output. Increase for IO-bound preparation.",
    )
    parser.add_argument(
        "--colors",
        default=";".join(DEFAULT_COLOR_SPECS),
        help="Semicolon-separated foreground colors as NAME:R,G,B. Background is always black.",
    )
    parser.add_argument(
        "--no-random-colors",
        action="store_true",
        help="Use only the first configured color instead of sampling a color per row.",
    )
    parser.add_argument(
        "--mask-override-hf-dataset",
        default=None,
        help=(
            "HF dataset that re-annotates some sources of the base release (same rows, new masks), "
            "e.g. JonaRuthardt/SteerViT-SAM3-bboxprompt. Rows of --mask-override-sources take its mask; "
            "rows it lacks are dropped. Row ids, colors and prompts stay those of the base release."
        ),
    )
    parser.add_argument(
        "--mask-override-sources",
        default=None,
        help=(
            "Comma-separated values of the 'dataset' column whose masks come from --mask-override-hf-dataset, "
            "e.g. 'visual genome,refcoco,refcoco+,refcocog'."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Load/filter the dataset and print counts without writing files.")

    parser.add_argument("--mscoco", type=Path, default=REFSEG_ROOT / "coco")
    parser.add_argument("--visual-genome", type=Path, default=REFSEG_ROOT / "visual_genome")
    parser.add_argument("--mapillary", type=Path, default=REFSEG_ROOT / "mapillary")
    parser.add_argument("--lisa", type=Path, default=REFSEG_ROOT / "lisa")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    out_root = args.out_root.expanduser().resolve()
    refcocog_umd_dir = Path(os.path.expandvars(str(args.refcocog_umd_dir))).expanduser()
    holdout_audit: coco_eval_holdouts.CocoEvaluationHoldoutAudit | None
    if args.allow_eval_image_overlap:
        holdout_audit = None
        print("WARNING: evaluation-image exclusion disabled by --allow-eval-image-overlap.")
    else:
        holdout_audit = coco_eval_holdouts.CocoEvaluationHoldoutAudit(
            coco_eval_holdouts.load_coco_evaluation_holdouts(refcocog_umd_dir)
        )
        print(
            "Protecting evaluation images: "
            f"{ {name: len(ids) for name, ids in holdout_audit.holdouts.items()} }"
        )
    dataset_names = split_csv(args.hf_datasets, DEFAULT_HF_DATASETS)
    splits = split_csv(args.split)
    if not splits:
        raise ValueError("At least one split is required.")
    include_datasets = set(split_csv(args.include_datasets)) if args.include_datasets else None
    refcocog_umd_splits = parse_refcocog_umd_splits(args.refcocog_umd_splits)
    refcocog_umd_split_by_sample = (
        load_refcocog_umd_splits(refcocog_umd_dir)
        if refcocog_umd_splits is not None
        else None
    )
    colors = parse_color_list(args.colors)
    roots = ImageRoots(
        mscoco=Path(os.path.expandvars(str(args.mscoco))).expanduser() if args.mscoco else None,
        visual_genome=Path(os.path.expandvars(str(args.visual_genome))).expanduser() if args.visual_genome else None,
        mapillary=Path(os.path.expandvars(str(args.mapillary))).expanduser() if args.mapillary else None,
        lisa=Path(os.path.expandvars(str(args.lisa))).expanduser() if args.lisa else None,
    )

    if bool(args.mask_override_hf_dataset) != bool(args.mask_override_sources):
        raise ValueError("--mask-override-hf-dataset and --mask-override-sources must be given together.")
    override: MaskOverride | None = None
    if args.mask_override_hf_dataset:
        override = load_mask_override(
            args.mask_override_hf_dataset, split_csv(args.mask_override_sources), splits
        )
        print(
            f"Mask override: {sorted(override.sources)} from {override.hf_dataset} "
            f"({len(override.index_by_key):,} rows; {', '.join(override.revisions) or 'no cache files'})"
        )
    override_index_by_row: dict[int, int] = {}
    override_missing: dict[str, int] = {}

    dataset = load_steervit_dataset(dataset_names, splits)
    filtered_indices = []
    unmatched_refcocog_ids: set[str] = set()
    for index, item in enumerate(tqdm(dataset, desc="Filtering SteerViT")):
        if holdout_audit is not None:
            image_reference, canonical_id, reasons = item_evaluation_holdout(item, holdout_audit.holdouts)
            if reasons:
                holdout_audit.record(
                    image_reference or canonical_id or "<unknown>",
                    reasons,
                    source=str(item.get("dataset") or "steervit"),
                    reference=str(item.get("ref_id") or item.get("id") or index),
                    canonical_id=canonical_id,
                )
                continue
        if (
            refcocog_umd_splits is not None
            and str(item.get("dataset", "")).lower() == "refcocog"
            and refcocog_umd_split_for_item(item, refcocog_umd_split_by_sample) is None
        ):
            unmatched_refcocog_ids.add(str(item.get("ref_id") or item.get("id") or "<missing>"))
        if row_matches_filters(
            item,
            include_datasets,
            args.min_width,
            args.min_height,
            args.max_prompt_length,
            args.ignore_coco,
            refcocog_umd_splits,
            refcocog_umd_split_by_sample,
        ):
            if override is not None and str(item.get("dataset", "")) in override.sources:
                # After the other filters, so the drop counts cover only rows that would be written.
                split = row_source_split(item, splits[0])
                override_index = override_index_for_item(item, split, override)
                if override_index is None:
                    key = f"{split}/{item.get('dataset')}"
                    override_missing[key] = override_missing.get(key, 0) + 1
                    continue
                override_index_by_row[index] = override_index
            filtered_indices.append(index)
    if unmatched_refcocog_ids:
        examples = ", ".join(sorted(unmatched_refcocog_ids)[:5])
        raise ValueError(
            "Could not match SteerViT RefCOCOg row id(s) to refs(umd).p "
            f"under {args.refcocog_umd_dir}: {examples}. "
            "Refuse to apply a non-verifiable UMD split filter."
        )
    if args.limit is not None and len(filtered_indices) > args.limit:
        # Seeded random subset rather than a prefix (rows are grouped by split and
        # source); sorted back so writing stays in dataset order.
        rng = random.Random(args.seed)
        filtered_indices = sorted(rng.sample(filtered_indices, args.limit))

    if override is not None:
        print(
            f"Mask override: {len(override_index_by_row):,} base rows matched; "
            f"dropped base rows without an override row: {dict(sorted(override_missing.items()))}"
        )

    if args.dry_run:
        counts: dict[str, int] = {}
        split_counts: dict[str, int] = {}
        umd_split_counts: dict[str, int] = {}
        for index in filtered_indices:
            item = dict(dataset[index])
            dataset_name = str(item.get("dataset", ""))
            split = row_source_split(item, splits[0])
            counts[dataset_name] = counts.get(dataset_name, 0) + 1
            split_counts[split] = split_counts.get(split, 0) + 1
            native_split = refcocog_umd_split_for_item(item, refcocog_umd_split_by_sample)
            if native_split is not None:
                umd_split_counts[native_split] = umd_split_counts.get(native_split, 0) + 1
        print(f"Matched {len(filtered_indices):,} rows from {len(dataset):,} loaded rows.")
        print("By split:")
        for split, count in sorted(split_counts.items()):
            print(f"  {split}: {count:,}")
        print("By source dataset:")
        for dataset_name, count in sorted(counts.items()):
            print(f"  {dataset_name}: {count:,}")
        if holdout_audit is not None:
            summary = holdout_audit.summary()
            print(f"Excluded {summary['excluded_candidates']:,} RefSeg evaluation candidates: {summary['excluded_by_reason']}")
        if refcocog_umd_splits is not None:
            print("By native RefCOCOg UMD split:")
            for split, count in sorted(umd_split_counts.items()):
                print(f"  {split}: {count:,}")
        return

    (out_root / "conditioning_images").mkdir(parents=True, exist_ok=True)
    (out_root / "target_images").mkdir(parents=True, exist_ok=True)
    if holdout_audit is not None:
        audit_path = holdout_audit.write(out_root)
        print(f"Wrote evaluation-image exclusion audit: {audit_path}")

    metadata_rows, skipped = write_rows(
        dataset,
        filtered_indices,
        splits[0],
        roots,
        out_root,
        colors,
        not args.no_random_colors,
        args.seed,
        args.num_workers,
        refcocog_umd_split_by_sample,
        override,
        override_index_by_row,
    )

    if not metadata_rows:
        raise RuntimeError(
            "SteerViT preparation produced zero rows. Check --include-datasets, image roots, and mask columns."
        )

    with (out_root / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    mask_override_config = None
    if override is not None:
        mask_source_counts: dict[str, int] = {}
        for row in metadata_rows:
            key = f"{row['source_dataset']}/{row['metadata']['dataset']}"
            mask_source_counts[key] = mask_source_counts.get(key, 0) + 1
        mask_override_config = {
            "hf_dataset": override.hf_dataset,
            "sources": sorted(override.sources),
            "match_fields": ["split", "dataset", "id", *OVERRIDE_MATCH_FIELDS],
            "base_cache_dirs": dataset_revisions(dataset),
            "override_cache_dirs": override.revisions,
            "dropped_without_override_row": dict(sorted(override_missing.items())),
            "written_rows_by_mask_source": dict(sorted(mask_source_counts.items())),
        }
        print(f"  rows by mask source: {mask_override_config['written_rows_by_mask_source']}")
    write_config(args, colors, out_root, mask_override_config)

    print("SteerViT refseg preparation complete.")
    print(f"  splits: {', '.join(splits)}")
    print(f"  loaded rows: {len(dataset):,}")
    print(f"  matched rows: {len(filtered_indices):,}")
    print(f"  written rows: {len(metadata_rows):,}")
    for key, count in sorted(skipped.items()):
        print(f"  {key}: {count:,}")
    print(f"  out_root: {out_root}")


if __name__ == "__main__":
    main()
