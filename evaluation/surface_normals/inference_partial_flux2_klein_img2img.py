# ===============================================================================
# Flux2-Klein partial-transformer NYUv2 surface-normal inference and evaluation.
# ===============================================================================

"""
1. Reads NYUv2 surface-normal split files from an extracted nyu_surface_normals/ root.
2. Runs each RGB image through Flux2-Klein with a surface-normal prompt.
3. Decodes the generated RGB output into camera-space unit normals.
4. Compares predictions against encoded GT normals with angular-error metrics.
5. Saves raw generations, normalized encoded predictions, angular-error maps,
   and per-sample / aggregate metrics.

Typical usage:
    python evaluation/surface_normals/inference_partial_flux2_klein_img2img.py \
        --data_root /path/to/nyu_surface_normals \
        --out_dir outputs/nyu_surface_normals \
        --transformer_path /path/to/open-visionbanana-partial-ft
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from common.evaluation_resume import (
    DurableEvaluationJournal,
    summarize_sharded_journals,
)
from common.evaluation_subset import apply_sample_manifest
from common.evaluation_sharding import (
    round_robin_shard,
    sharding_run_identity,
    synchronize_shard_records,
    validate_shard_parameters,
)
from evaluation.geometry_diagnostics import normal_representation_diagnostics
from evaluation.geometry_protocols import (
    NORMAL_BIDIRECTIONAL_DIAGNOSTIC,
    NORMAL_HELDOUT_CALIBRATED,
    NORMAL_HELDOUT_SO3_CALIBRATED,
    NORMAL_TEST5_ORACLE,
    NORMAL_STRICT,
    NORMAL_VALID_ONLY_DIAGNOSTIC,
    convention_oracle_normal_metrics,
    load_normal_calibration_record,
    load_normal_so3_calibration_record,
    normal_configuration_identity,
    PROTOCOL_VERSION,
    normal_metrics as geometry_normal_metrics,
)
from common.geometry_protocol_prompts import CANONICAL_NORMAL_PROMPT, prompt_sha256
from common.surface_normal_encoding import (
    LEGACY_DIRECT_NORMAL_ENCODING,
    SUPPORTED_NORMAL_ENCODINGS,
    VISION_BANANA_NORMAL_ENCODING,
    decode_surface_normals_rgb,
    encode_surface_normals_rgb,
    normal_encoding_formula,
    transform_surface_normals,
)
from evaluation.surface_normals import datasets as normal_datasets
from evaluation.surface_normals.target_protocol import (
    AUTO_NORMAL_ENCODING,
    resolve_ground_truth_protocol,
)

PROMPT = CANONICAL_NORMAL_PROMPT
RINO_NORMAL_PROMPT = (
    "Convert this image into a high-quality normal map that faithfully represents "
    "surface orientation across the scene. Capture both large-scale object geometry "
    "and fine-scale structural details through consistent color-coded normal vectors."
)
RINO_QWEN_NEGATIVE_PROMPT = (
    "photo, photorealistic, road paint, lane marking, white line, yellow line, "
    "crosswalk, text, license plate, shadow, reflection, gradient, texture, "
    "fine detail, brick texture, window frame"
)
LEGACY_UNSEEDED_EVALUATOR_SHA256 = (
    "401322edb537f066b9a571008b700e117d320c1b554e885e3efd816c074c5dd0"
)
# Calibration compatibility identifies the frozen primary decoding/masking/
# scoring semantics, not optional diagnostic lanes. Adding the opt-in SO(3)
# lane must not invalidate existing immutable 48-axis calibration records.
CALIBRATION_COMPATIBLE_EVALUATOR_SHA256 = (
    "079f626a4af1ba959a8b124e9f490ed7e63eb99d515eae8a26818891a699f7f8"
)
NORMAL_METRIC_NAMES = (
    "mean_angle",
    "median_angle",
    "rmse_angle",
    "angle_11_25",
    "angle_22_5",
    "angle_30",
)

_RESAMPLING = getattr(Image, "Resampling", Image)
BILINEAR = _RESAMPLING.BILINEAR
POOLED_ANGLE_HISTOGRAM_BIN_WIDTH_DEG = 0.01
POOLED_ANGLE_HISTOGRAM_BINS = int(180.0 / POOLED_ANGLE_HISTOGRAM_BIN_WIDTH_DEG)


@dataclass(frozen=True)
class NYUNormalSample:
    split: str
    image_path: Path
    normal_path: Path
    mask_path: Optional[Path]
    focal_length: float
    source_index: Optional[int]
    rel_image_path: Path
    output_stem: str


class TooFewValidNormalPixelsError(RuntimeError):
    def __init__(self, valid_pixels: int, min_valid_pixels: int):
        super().__init__(
            f"Only {valid_pixels} valid normal pixels found; required at least {min_valid_pixels}."
        )
        self.valid_pixels = valid_pixels
        self.min_valid_pixels = min_valid_pixels


def canonicalize_split_name(split: str) -> str:
    split = split.lower()
    if split == "val":
        return "test"
    return split


def expand_requested_splits(split: str) -> list[str]:
    split = canonicalize_split_name(split)
    if split == "all":
        return ["test", "train"]
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    return [split]


def resolve_dataset_root(data_root: Path) -> Path:
    data_root = data_root.expanduser()
    if (data_root / "nyu_train.txt").is_file() or (data_root / "nyu_test.txt").is_file():
        return data_root
    nested = data_root / "nyu_surface_normals"
    if (nested / "nyu_train.txt").is_file() or (nested / "nyu_test.txt").is_file():
        return nested
    raise FileNotFoundError(
        "Could not find NYU surface-normal split files. Expected either:\n"
        f"  {data_root / 'nyu_train.txt'} or {data_root / 'nyu_test.txt'}\n"
        "or:\n"
        f"  {nested / 'nyu_train.txt'} or {nested / 'nyu_test.txt'}"
    )


def output_stem_from_rel_path(split: str, rel_path: Path) -> str:
    parts = [split, *rel_path.with_suffix("").parts]
    return "__".join(part for part in parts if part not in ("", "."))


def looks_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def load_split_file(dataset_root: Path, split: str) -> list[NYUNormalSample]:
    split_file = dataset_root / f"nyu_{split}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing NYU surface-normal split file: {split_file}")

    samples = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed split entry in {split_file}:{line_number}: {line}")

            rel_image_path = Path(parts[0].lstrip("/"))
            rel_normal_path = Path(parts[1].lstrip("/"))
            rel_mask_path: Optional[Path] = None
            focal_length = float("nan")
            source_index: Optional[int] = None

            cursor = 2
            if cursor < len(parts) and not looks_float(parts[cursor]):
                rel_mask_path = Path(parts[cursor].lstrip("/"))
                cursor += 1
            if cursor < len(parts) and looks_float(parts[cursor]):
                focal_length = float(parts[cursor])
                cursor += 1
            if cursor < len(parts):
                try:
                    source_index = int(parts[cursor])
                except ValueError:
                    source_index = None

            image_path = dataset_root / rel_image_path
            normal_path = dataset_root / rel_normal_path
            mask_path = dataset_root / rel_mask_path if rel_mask_path is not None else None
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing RGB image from split file: {image_path}")
            if not normal_path.is_file():
                raise FileNotFoundError(f"Missing normal image from split file: {normal_path}")
            if mask_path is not None and not mask_path.is_file():
                raise FileNotFoundError(f"Missing normal mask from split file: {mask_path}")

            samples.append(
                NYUNormalSample(
                    split=split,
                    image_path=image_path,
                    normal_path=normal_path,
                    mask_path=mask_path,
                    focal_length=focal_length,
                    source_index=source_index,
                    rel_image_path=rel_image_path,
                    output_stem=output_stem_from_rel_path(split, rel_image_path),
                )
            )
    return samples


def build_samples(dataset_root: Path, split: str, limit: Optional[int]) -> list[NYUNormalSample]:
    samples = []
    for split_name in expand_requested_splits(split):
        samples.extend(load_split_file(dataset_root, split_name))
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise FileNotFoundError(f"No NYU surface-normal samples resolved for split={split!r}")
    return samples


def rgb_to_unit_normals(
    rgb: Image.Image,
    min_norm: float,
    encoding: str = LEGACY_DIRECT_NORMAL_ENCODING,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a declared RGB storage convention into physical canonical XYZ.

    The default is the legacy RGB convention. Launchers select Vision Banana
    encoding for model predictions and separately declare the encoding of each
    frozen GT archive.
    """
    return decode_surface_normals_rgb(
        np.asarray(rgb.convert("RGB"), dtype=np.uint8),
        encoding=encoding,
        min_norm=min_norm,
    )


def load_gt_normals(
    normal_path: Path,
    *,
    min_norm: float,
    rgb_encoding: str = LEGACY_DIRECT_NORMAL_ENCODING,
) -> tuple[np.ndarray, np.ndarray]:
    """Load float NPY benchmark normals or legacy RGB-encoded normals.

    Marigold distributes iBims-1 ground truth as HWC float arrays with invalid
    pixels encoded as zero vectors.  Keeping this path in float avoids the
    angular error and validity ambiguity introduced by uint8 RGB quantization.
    """
    if normal_path.suffix.lower() != ".npy":
        return rgb_to_unit_normals(
            Image.open(normal_path).convert("RGB"),
            min_norm=min_norm,
            encoding=rgb_encoding,
        )

    normals = np.asarray(np.load(normal_path, allow_pickle=False), dtype=np.float32)
    if normals.ndim != 3:
        raise ValueError(f"Expected a 3-D normal array in {normal_path}, got {normals.shape}")
    if normals.shape[-1] == 3:
        pass
    elif normals.shape[0] == 3:
        normals = np.transpose(normals, (1, 2, 0))
    else:
        raise ValueError(f"Expected three normal channels in {normal_path}, got {normals.shape}")
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    valid = np.all(np.isfinite(normals), axis=-1) & (norm[..., 0] > float(min_norm))
    unit = np.divide(
        normals,
        norm,
        out=np.zeros_like(normals),
        where=np.isfinite(norm) & (norm > float(min_norm)),
    )
    unit[~valid] = 0
    return unit, valid


def rgb_to_oracle_unit_normals(rgb: Image.Image) -> np.ndarray:
    """Decode and normalize every RGB vector for convention diagnostics.

    RGB quantization means even neutral 127/128 pixels have a small non-zero
    vector. The oracle diagnostic normalizes and scores them; the primary protocol
    does not and instead records zero coverage plus a fixed 90-degree penalty.
    """
    normals = np.asarray(rgb.convert("RGB"), dtype=np.float64) / 255.0 * 2.0 - 1.0
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.clip(norm, 1e-8, None)


def unit_normals_to_rgb(
    normals: np.ndarray,
    valid: np.ndarray,
    encoding: str = LEGACY_DIRECT_NORMAL_ENCODING,
) -> np.ndarray:
    return encode_surface_normals_rgb(
        normals,
        valid,
        encoding=encoding,
        invalid_rgb=(128, 128, 128),
    )


def load_mask(mask_path: Optional[Path], shape: tuple[int, int], gt_valid: np.ndarray) -> np.ndarray:
    if mask_path is None:
        return gt_valid.astype(bool, copy=True)
    arr = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    if arr.shape != shape:
        arr = np.asarray(
            Image.fromarray(arr, mode="L").resize((shape[1], shape[0]), resample=_RESAMPLING.NEAREST),
            dtype=np.uint8,
        )
    return arr > 0


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool, copy=True)

    eroded = mask.astype(bool, copy=True)
    height, width = eroded.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.zeros_like(eroded)
            src_y0 = max(0, -dy)
            src_y1 = min(height, height - dy)
            src_x0 = max(0, -dx)
            src_x1 = min(width, width - dx)
            dst_y0 = max(0, dy)
            dst_y1 = min(height, height + dy)
            dst_x0 = max(0, dx)
            dst_x1 = min(width, width + dx)
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
            eroded &= shifted
    return eroded


def resize_rgb(image: Image.Image, target_shape: tuple[int, int]) -> Image.Image:
    if image.size == (target_shape[1], target_shape[0]):
        return image.convert("RGB")
    return image.convert("RGB").resize((target_shape[1], target_shape[0]), resample=BILINEAR)


def compute_normal_metrics(
    pred_normals: np.ndarray,
    pred_valid: np.ndarray,
    gt_normals: np.ndarray,
    gt_valid: np.ndarray,
    eval_mask: np.ndarray,
    *,
    bidirectional_normals: bool,
    min_valid_pixels: int,
    valid_prediction_only: bool = False,
) -> tuple[dict[str, float], int, np.ndarray]:
    gt_pixels = int((eval_mask & gt_valid).sum())
    try:
        metrics, angle_map = geometry_normal_metrics(
            pred_normals=pred_normals,
            pred_valid=pred_valid,
            gt_normals=gt_normals,
            gt_valid=gt_valid,
            eval_mask=eval_mask,
            bidirectional=bidirectional_normals,
            valid_prediction_only=valid_prediction_only,
            min_gt_pixels=min_valid_pixels,
        )
    except ValueError as exc:
        valid_pixels = int((eval_mask & gt_valid & pred_valid).sum()) if valid_prediction_only else gt_pixels
        raise TooFewValidNormalPixelsError(valid_pixels, min_valid_pixels) from exc
    scored_pixels = int(metrics["pred_valid_pixels"]) if valid_prediction_only else gt_pixels
    return metrics, scored_pixels, angle_map


def aggregate_metrics(records: Sequence[dict[str, float]]) -> dict[str, float]:
    return {name: float(np.mean([record[name] for record in records])) for name in NORMAL_METRIC_NAMES}


def marigold_v1_1_metrics(
    metrics: dict[str, float], records: Sequence[dict[str, float]]
) -> dict[str, float]:
    """Expose the upstream Marigold table schema and percentage units."""
    def per_image_threshold(record: dict[str, float], key: str, count_key: str) -> float:
        if key in record:
            return float(record[key])
        # Compatibility with durable records written before the named
        # Marigold lane was added; their mergeable pooled counts are sufficient.
        return float(record[count_key]) / float(record["_pooled_angle_count"])

    return {
        "mean_angular_error": metrics["mean_angle"],
        "median_angular_error": metrics["median_angle"],
        "sub5_error": float(np.mean([
            per_image_threshold(record, "marigold_angle_5", "_pooled_angle_lt_5_count")
            for record in records
        ])) * 100.0,
        "sub7_5_error": float(np.mean([
            per_image_threshold(record, "marigold_angle_7_5", "_pooled_angle_lt_7_5_count")
            for record in records
        ])) * 100.0,
        "sub11_25_error": metrics["angle_11_25"] * 100.0,
        "sub22_5_error": metrics["angle_22_5"] * 100.0,
        "sub30_error": metrics["angle_30"] * 100.0,
    }


def pooled_normal_statistics(angle_map: np.ndarray, score_mask: np.ndarray) -> dict[str, object]:
    """Return mergeable sufficient statistics for DSINE-style pixel pooling.

    The 0.01-degree histogram makes the pooled median durable across 24-hour
    resume chunks while keeping the journal compact. All other pooled metrics
    are exact up to the float64 angular-error calculation.
    """
    angles = np.asarray(angle_map, dtype=np.float64)[np.asarray(score_mask, dtype=bool)]
    angles = angles[np.isfinite(angles)]
    if not angles.size:
        raise ValueError("cannot pool an empty normal-angle array")
    histogram, _ = np.histogram(
        angles,
        bins=POOLED_ANGLE_HISTOGRAM_BINS,
        range=(0.0, 180.0),
    )
    packed_histogram = base64.b64encode(
        zlib.compress(histogram.astype("<u4", copy=False).tobytes(), level=9)
    ).decode("ascii")
    return {
        "_pooled_angle_count": int(angles.size),
        "_pooled_angle_sum": float(np.sum(angles, dtype=np.float64)),
        "_pooled_angle_squared_sum": float(np.sum(angles * angles, dtype=np.float64)),
        "_pooled_angle_lt_5_count": int(np.sum(angles < 5.0)),
        "_pooled_angle_lt_7_5_count": int(np.sum(angles < 7.5)),
        "_pooled_angle_lt_11_25_count": int(np.sum(angles < 11.25)),
        "_pooled_angle_lt_22_5_count": int(np.sum(angles < 22.5)),
        "_pooled_angle_lt_30_count": int(np.sum(angles < 30.0)),
        "_pooled_angle_histogram_0p01deg_zlib_base64": packed_histogram,
    }


def aggregate_pooled_normal_statistics(
    records: Sequence[dict[str, object]],
) -> dict[str, float | int | str]:
    count = sum(int(record["_pooled_angle_count"]) for record in records)
    if count <= 0:
        raise ValueError("cannot aggregate zero pooled normal pixels")
    histogram = np.zeros(POOLED_ANGLE_HISTOGRAM_BINS, dtype=np.uint64)
    for record in records:
        packed = base64.b64decode(
            str(record["_pooled_angle_histogram_0p01deg_zlib_base64"])
        )
        decoded = np.frombuffer(zlib.decompress(packed), dtype="<u4")
        if decoded.size != POOLED_ANGLE_HISTOGRAM_BINS:
            raise ValueError(
                f"invalid pooled angle histogram: expected "
                f"{POOLED_ANGLE_HISTOGRAM_BINS} bins, got {decoded.size}"
            )
        histogram += decoded.astype(np.uint64)
    if int(histogram.sum()) != count:
        raise ValueError(
            f"pooled angle histogram count mismatch: {int(histogram.sum())} != {count}"
        )
    median_bin = int(np.searchsorted(np.cumsum(histogram), (count + 1) // 2))
    median_angle = (median_bin + 0.5) * POOLED_ANGLE_HISTOGRAM_BIN_WIDTH_DEG
    angle_sum = sum(float(record["_pooled_angle_sum"]) for record in records)
    squared_sum = sum(float(record["_pooled_angle_squared_sum"]) for record in records)
    return {
        "mean_angle": angle_sum / count,
        "median_angle": median_angle,
        "rmse_angle": float(np.sqrt(squared_sum / count)),
        "angle_5": sum(int(record["_pooled_angle_lt_5_count"]) for record in records) / count,
        "angle_7_5": sum(int(record["_pooled_angle_lt_7_5_count"]) for record in records) / count,
        "angle_11_25": sum(int(record["_pooled_angle_lt_11_25_count"]) for record in records) / count,
        "angle_22_5": sum(int(record["_pooled_angle_lt_22_5_count"]) for record in records) / count,
        "angle_30": sum(int(record["_pooled_angle_lt_30_count"]) for record in records) / count,
        "pixels": count,
        "aggregation": "global_valid_pixel_pool",
        "median_precision_degrees": POOLED_ANGLE_HISTOGRAM_BIN_WIDTH_DEG,
    }


def aggregate_prefixed_metrics(
    records: Sequence[dict[str, float | int | bool | str | None]],
    prefix: str,
) -> Optional[dict[str, float]]:
    if not any(f"{prefix}_{NORMAL_METRIC_NAMES[0]}" in record for record in records):
        return None
    return {
        name: float(np.mean([float(record[f"{prefix}_{name}"]) for record in records if f"{prefix}_{name}" in record]))
        for name in NORMAL_METRIC_NAMES
    }


def aggregate_diagnostic_fields(records, prefix="representation_"):
    keys = sorted({key for record in records for key in record if key.startswith(prefix)})
    return {key: float(np.nanmean([float(record[key]) for record in records if key in record])) for key in keys}


def records_by_case(records: Sequence[dict[str, float | int | bool | str | None]]) -> dict[str, list[dict[str, float | int | bool | str | None]]]:
    grouped: dict[str, list[dict[str, float | int | bool | str | None]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("case", "standard")), []).append(record)
    return grouped


def angle_to_gray(angle_map: np.ndarray, max_angle: float) -> np.ndarray:
    vis = np.zeros(angle_map.shape, dtype=np.uint8)
    valid = np.isfinite(angle_map)
    if not np.any(valid):
        return vis
    scaled = np.clip(angle_map[valid] / max(max_angle, 1e-6), 0.0, 1.0)
    vis[valid] = np.rint(scaled * 255.0).astype(np.uint8)
    return vis


def resolve_transformer_load_args(transformer_path: str) -> tuple[str, Optional[str]]:
    path = Path(transformer_path).expanduser()
    if (path / "transformer").is_dir():
        return str(path), "transformer"
    return str(path), None


def load_pipeline(model_name: str, transformer_path: Optional[str], model_family="klein", lora_path=None, offload=False):
    if model_family == "qwen_musubi_2511":
        if lora_path:
            raise ValueError("The base Musubi parity lane does not accept --lora_path")
        import importlib.util

        runtime_path = Path(os.environ["QWEN_RUNTIME_FILE"]).expanduser().resolve()
        spec = importlib.util.spec_from_file_location("ovb_qwen_edit_runtime", runtime_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load Qwen Musubi runtime: {runtime_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        pipe = module.QwenEditPipeline.load(
            transformer_path or model_name,
            models_dir=os.environ.get("QWEN_MODELS_DIR"),
            device="cuda",
            negative_prompt=os.environ.get("QWEN_NEGATIVE_PROMPT", " "),
            blocks_to_swap=int(os.environ.get("QWEN_BLOCKS_TO_SWAP", "0")),
            text_encoder_cpu=os.environ.get("QWEN_TEXT_ENCODER_CPU", "0").lower()
            in {"1", "true", "yes", "on"},
            attn_mode=os.environ.get("QWEN_ATTN_MODE") or None,
        )
        import torch

        return pipe, torch
    if model_family == "qwen_edit_2511":
        if transformer_path or lora_path:
            raise ValueError(
                "The official Qwen-Image-Edit parity backend does not accept "
                "--transformer_path or --lora_path."
            )
        import torch
        from diffusers import QwenImageEditPlusPipeline

        local_only = Path(model_name).expanduser().exists() or os.environ.get("HF_HUB_OFFLINE") == "1"
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            local_files_only=local_only,
        )
        if offload:
            # Match RINO exactly; Accelerate selects the current CUDA device.
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        return pipe, torch
    from common.flux2_runtime import load_flux2_pipeline
    pipe, torch, _metadata = load_flux2_pipeline(
        model_name, model_family=model_family, transformer_path=transformer_path,
        lora_path=lora_path, device="cuda", offload=offload,
    )
    return pipe, torch


def make_generation_generator(
    torch_module,
    seed: int,
    legacy_unseeded: bool = False,
    model_family: str = "klein",
):
    """Return a freshly seeded CUDA generator for one independent sample."""
    if legacy_unseeded:
        return None
    # RINO calls torch.manual_seed(), which returns the default CPU generator.
    # Preserve that exact behavior in the Qwen parity lane; Flux pipelines use
    # a CUDA generator.
    if model_family == "qwen_edit_2511":
        return torch_module.manual_seed(seed)
    return torch_module.Generator(device="cuda").manual_seed(seed)


def parse_axis_permutation(value: str) -> tuple[int, int, int]:
    try:
        parsed = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected three comma-separated integers") from exc
    if len(parsed) != 3 or sorted(parsed) != [0, 1, 2]:
        raise argparse.ArgumentTypeError("normal permutation must contain 0,1,2 once")
    return parsed


def parse_axis_signs(value: str) -> tuple[int, int, int]:
    try:
        parsed = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected three comma-separated signs") from exc
    if len(parsed) != 3 or any(sign not in (-1, 1) for sign in parsed):
        raise argparse.ArgumentTypeError("normal signs must be three values from {-1,+1}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-base-9B")
    parser.add_argument(
        "--model_family",
        choices=("klein", "dev", "qwen_edit_2511", "qwen_musubi_2511"),
        default="klein",
    )
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--model_offload", action="store_true")
    parser.add_argument(
        "--transformer_path",
        default=None,
        help=(
            "Optional path to the trained output dir, checkpoint dir, or transformer dir. "
            "Both open-visionbanana-partial-ft and open-visionbanana-partial-ft/transformer are accepted."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=normal_datasets.SUPPORTED_DATASETS,
        default="nyu",
        help="Evaluation dataset adapter to use. Defaults to the existing NYU behavior.",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Extracted dataset root, or a parent directory containing the prepared dataset folder.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "test", "val", "all", "debug", "small", "standard", "large", "giant"),
        help="Default is the held-out NYUv2 test split. 'val' is accepted as an alias for 'test'.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=("standard",),
        help=(
            "Dataset cases to evaluate. For iBims-1, default is standard. "
            "Use 'aux_augmented' or 'all' to include optional auxiliary data. "
            "For DIODE, use indoor, outdoor, or all. For nuScenes, use cam_front or all."
        ),
    )
    parser.add_argument(
        "--aux_augmentations",
        "--aux-augmentations",
        nargs="+",
        default=("appearance",),
        help=(
            "iBims-1 auxiliary augmentation groups. Defaults to appearance, which is the safe "
            "non-geometric group. Use all or specific group names for diagnostic runs."
        ),
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument(
        "--true_cfg_scale",
        type=float,
        default=4.0,
        help="Qwen-Image-Edit true CFG scale. RINO uses 4.0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Generation seed reset independently for every sample. Resetting at "
            "the sample boundary makes retries and durable resume deterministic."
        ),
    )
    parser.add_argument(
        "--legacy_unseeded",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--rino_prompt",
        action="store_true",
        help="Use RINO's exact published surface-normal prompt.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sample_manifest",
        default=None,
        help=(
            "Frozen ordered JSONL sample manifest. Mutually exclusive with --limit; "
            "used for versioned geometry ablation subsets."
        ),
    )
    parser.add_argument(
        "--prediction_normal_encoding",
        choices=SUPPORTED_NORMAL_ENCODINGS,
        default=VISION_BANANA_NORMAL_ENCODING,
        help=(
            "RGB convention emitted by the release model. Default: "
            "vision_banana_x_left_red; pass legacy_direct_xyz only to reproduce v2."
        ),
    )
    parser.add_argument(
        "--gt_normal_encoding",
        choices=(AUTO_NORMAL_ENCODING, *SUPPORTED_NORMAL_ENCODINGS),
        default=AUTO_NORMAL_ENCODING,
        help=(
            "RGB storage convention used by the frozen GT archive. Default: auto, "
            "resolved from archive metadata; float NPY targets are physical XYZ."
        ),
    )
    parser.add_argument(
        "--gt_normal_permutation",
        type=parse_axis_permutation,
        default=None,
        help="Optional stored-GT to canonical XYZ override, e.g. 0,1,2.",
    )
    parser.add_argument(
        "--gt_normal_signs",
        type=parse_axis_signs,
        default=None,
        help="Optional stored-GT to canonical XYZ override, e.g. -1,1,1.",
    )
    parser.add_argument(
        "--primary_case",
        default=None,
        help=(
            "Optional case selected for top-level primary metrics while all "
            "evaluated cases remain available under *_metrics_by_case."
        ),
    )
    parser.add_argument(
        "--protocol_version",
        default=PROTOCOL_VERSION,
        help="Versioned evaluation contract recorded in the result JSON.",
    )
    parser.add_argument(
        "--dataset_role",
        default="unspecified",
        choices=(
            "vision_banana_primary",
            "extended_diagnostic",
            "axis_oracle_diagnostic_only",
            "heldout_calibration",
            "unspecified",
        ),
    )
    parser.add_argument(
        "--resume_state_dir",
        default=None,
        help="Persistent state directory used to resume a terminated evaluation safely.",
    )
    parser.add_argument("--resume_chunk_size", type=int, default=50)
    parser.add_argument(
        "--resume_data_identity",
        default=None,
        help="Stable archive identity for resume validation (avoids job-local extraction paths).",
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument(
        "--shard_sync_dir",
        default=None,
        help="Shared node-local directory used to merge multi-process records.",
    )
    parser.add_argument("--shard_sync_timeout_seconds", type=float, default=3600.0)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Resolve NYU image/normal/mask pairs and exit before loading the model.",
    )
    parser.add_argument(
        "--native_resolution",
        action="store_true",
        help="Ask Flux2-Klein to generate at the input image resolution.",
    )
    parser.add_argument("--min_normal_norm", type=float, default=0.01)
    parser.add_argument("--min_valid_pixels", type=int, default=4096)
    parser.add_argument(
        "--eval_mask_erosion_radius",
        type=int,
        default=0,
        help="Erode the primary evaluation mask by this pixel radius before scoring. Default keeps existing masks unchanged.",
    )
    parser.add_argument(
        "--diagnostic_mask_erosion_radius",
        type=int,
        default=0,
        help="Also report diagnostic metrics on this eroded mask without changing primary metrics.",
    )
    parser.add_argument(
        "--bidirectional_normals",
        action="store_true",
        help=(
            "Deprecated alias for --report_bidirectional_diagnostic. The primary "
            "metric is always signed canonical camera-space angular error."
        ),
    )
    parser.add_argument(
        "--report_bidirectional_diagnostic",
        action="store_true",
        help="Also report sign-insensitive abs(dot) metrics without changing primary strict metrics.",
    )
    parser.add_argument(
        "--angle_vis_max",
        type=float,
        default=90.0,
        help="Angular error in degrees mapped to white in angular_error_gray outputs.",
    )
    parser.add_argument(
        "--heldout_calibration_record",
        default=None,
        help=(
            "Immutable calibration JSON produced from a fixed, disjoint calibration "
            "subset. When supplied, report the held-out 48-convention calibration lane."
        ),
    )
    parser.add_argument(
        "--heldout_so3_calibration_record",
        default=None,
        help=(
            "Optional immutable record containing one proper global rotation "
            "fitted on the fixed, disjoint calibration subset. When supplied, "
            "report a separate heldout_so3_calibrated lane."
        ),
    )
    parser.add_argument(
        "--test5_oracle_record",
        default=None,
        help=(
            "Immutable record selecting one of 48 signed axis conventions from "
            "five test samples. This GT-assisted lane is always diagnostic."
        ),
    )
    parser.add_argument(
        "--prediction_dir",
        default=None,
        help=(
            "CPU rescoring mode: load existing raw prediction PNGs named "
            "<sample.output_stem>.png from this directory instead of loading a model."
        ),
    )
    parser.add_argument(
        "--metrics_only",
        action="store_true",
        help="Write only JSON/JSONL metrics; do not duplicate per-image diagnostic files.",
    )
    parser.add_argument(
        "--raw_predictions_only",
        action="store_true",
        help=(
            "Persist generated raw PNGs and aggregate JSON, but skip encoded-normal "
            "and angular-error images. Useful for low-inode cross-scoring runs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_shard_parameters(args.num_shards, args.shard_index)
    if args.num_shards > 1 and args.legacy_unseeded:
        raise ValueError(
            "multi-process evaluation requires fixed per-sample seeds; "
            "legacy process RNG is intentionally single-process only"
        )
    if args.num_shards > 1 and not args.shard_sync_dir:
        raise ValueError("--shard_sync_dir is required when --num_shards is greater than 1")
    if args.sample_manifest and args.limit is not None:
        raise ValueError("--sample_manifest and --limit are mutually exclusive")
    if args.metrics_only and args.resume_state_dir:
        raise ValueError("--metrics_only cannot be combined with --resume_state_dir")
    if args.metrics_only and args.raw_predictions_only:
        raise ValueError("--metrics_only and --raw_predictions_only are mutually exclusive")
    prediction_dir = (
        Path(args.prediction_dir).expanduser().resolve()
        if args.prediction_dir
        else None
    )
    report_bidirectional_diagnostic = bool(
        args.report_bidirectional_diagnostic or args.bidirectional_normals
    )
    if args.prompt and args.rino_prompt:
        raise ValueError("--prompt and --rino_prompt are mutually exclusive")
    prompt = args.prompt or (RINO_NORMAL_PROMPT if args.rino_prompt else PROMPT)
    qwen_negative_prompt = (
        os.environ.get("QWEN_NEGATIVE_PROMPT", RINO_QWEN_NEGATIVE_PROMPT)
        if args.model_family in {"qwen_edit_2511", "qwen_musubi_2511"}
        else None
    )
    normal_configuration = {
        "model": args.model,
        "model_family": args.model_family,
        "transformer_path": args.transformer_path,
        "lora_path": args.lora_path,
        "prompt_sha256": prompt_sha256(prompt),
        "negative_prompt_sha256": (
            prompt_sha256(qwen_negative_prompt)
            if qwen_negative_prompt is not None
            else None
        ),
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "true_cfg_scale": (
            args.true_cfg_scale if args.model_family == "qwen_edit_2511" else None
        ),
        "seed": None if args.legacy_unseeded else args.seed,
        "seed_mode": (
            "legacy_process_rng"
            if args.legacy_unseeded
            else "fixed_per_sample_reset"
        ),
        "native_resolution": bool(args.native_resolution),
        "prediction_normal_encoding": args.prediction_normal_encoding,
        "evaluator_sha256": (
            LEGACY_UNSEEDED_EVALUATOR_SHA256
            if args.legacy_unseeded
            else CALIBRATION_COMPATIBLE_EVALUATOR_SHA256
        ),
        "runtime_backend_revision": (
            os.environ.get("QWEN_MUSUBI_REV")
            if args.model_family == "qwen_musubi_2511"
            else "diffusers_environment"
        ),
        "attention_mode": os.environ.get("QWEN_ATTN_MODE"),
    }
    normalized_normal_configuration, normal_configuration_sha256 = (
        normal_configuration_identity(normal_configuration)
    )

    def load_convention_record(path_value: str | None, expected_protocol: str):
        if not path_value:
            return None, None
        return load_normal_calibration_record(
            Path(path_value).expanduser().resolve(),
            normalized_normal_configuration,
            expected_protocol=expected_protocol,
        )

    heldout_calibration, heldout_convention = load_convention_record(
        args.heldout_calibration_record, NORMAL_HELDOUT_CALIBRATED
    )
    if args.heldout_so3_calibration_record:
        heldout_so3_calibration, heldout_so3_rotation = (
            load_normal_so3_calibration_record(
                Path(args.heldout_so3_calibration_record).expanduser().resolve(),
                normalized_normal_configuration,
            )
        )
    else:
        heldout_so3_calibration, heldout_so3_rotation = None, None
    test5_oracle, test5_convention = load_convention_record(
        args.test5_oracle_record, NORMAL_TEST5_ORACLE
    )
    if test5_oracle is not None and (
        test5_oracle.get("calibration_dataset") != args.dataset
        or test5_oracle.get("calibration_split") != args.split
    ):
        raise ValueError(
            "Test-five normal convention record dataset/split does not match evaluation"
        )

    dataset_root = normal_datasets.resolve_dataset_root(args.dataset, Path(args.data_root))
    dataset_info_path = dataset_root / "dataset_info.json"
    if not dataset_info_path.is_file() and (dataset_root / "manifest.json").is_file():
        dataset_info_path = dataset_root / "manifest.json"
    dataset_info = (
        json.loads(dataset_info_path.read_text(encoding="utf-8"))
        if dataset_info_path.is_file()
        else {}
    )
    dataset_provenance = {
        key: dataset_info[key]
        for key in (
            "ground_truth_source",
            "source_zip_sha256",
            "source_code_repository",
            "source_code_commit",
            "geometry_version",
            "source_normal_decode",
            "prepared_normal_encoding",
            "validity_mask",
            "normal_convention",
            "license_note",
        )
        if key in dataset_info
    }
    out_dir = Path(args.out_dir)
    resolved_full_samples = normal_datasets.build_samples(
        dataset=args.dataset,
        dataset_root=dataset_root,
        split=args.split,
        limit=None if args.sample_manifest else args.limit,
        cases=tuple(args.cases),
        aux_augmentations=tuple(args.aux_augmentations),
    )
    full_samples, subset_metadata = apply_sample_manifest(
        resolved_full_samples,
        args.sample_manifest,
        dataset=args.dataset,
    )
    resolved_splits = sorted({sample.split for sample in full_samples})
    resolved_cases = sorted({sample.case for sample in full_samples})
    gt_storage_formats = sorted({sample.normal_path.suffix.lower().lstrip(".") for sample in full_samples})
    gt_protocol = resolve_ground_truth_protocol(
        dataset_info=dataset_info,
        storage_formats=gt_storage_formats,
        requested_encoding=args.gt_normal_encoding,
        requested_permutation=args.gt_normal_permutation,
        requested_signs=args.gt_normal_signs,
    )
    resolved_gt_rgb_encoding = (
        LEGACY_DIRECT_NORMAL_ENCODING
        if gt_protocol.encoding.startswith("float_xyz")
        else gt_protocol.encoding
    )
    if args.dataset == "ibims1" and gt_storage_formats == ["npy"]:
        dataset_provenance.update(
            {
                "ground_truth_source": (
                    "Marigold official normals evaluation archive; iBims-1 normals "
                    "provided by DSINE and derived from TUM laser-scanner depth"
                ),
                "ground_truth_kind": "depth-derived discontinuity-aware plane-fit normals",
                "validity_mask": "implicit: finite ground-truth normal norm > min_normal_norm",
                "normal_convention": "Marigold float XYZ (DSINE EXR with x sign reflected)",
                "reference_url": (
                    "https://share.phys.ethz.ch/~pf/bingkedata/marigold/"
                    "marigold_normals/evaluation_dataset.zip"
                ),
            }
        )
    samples = round_robin_shard(full_samples, args.num_shards, args.shard_index)

    if args.dry_run:
        print(f"Dataset root: {dataset_root}")
        print(f"Requested split: {args.split}")
        print(f"Resolved splits: {', '.join(resolved_splits)}")
        print(f"Resolved cases: {', '.join(resolved_cases)}")
        print(f"Resolved image/normal pairs: {len(full_samples)}")
        print(
            "GT protocol: "
            f"encoding={gt_protocol.encoding} permutation={gt_protocol.permutation} "
            f"signs={gt_protocol.signs} ({gt_protocol.source})"
        )
        if subset_metadata is not None:
            print(
                f"Sample subset: {subset_metadata['protocol_id']} "
                f"({subset_metadata['sample_count']}/"
                f"{subset_metadata['full_resolved_sample_count']})"
            )
        if args.num_shards > 1:
            print(
                f"Shard: {args.shard_index}/{args.num_shards} "
                f"({len(samples)} assigned pairs)"
            )
        print("First pairs:")
        for sample in samples[:5]:
            print(f"  split:  {sample.split}")
            print(f"  case:   {sample.case}")
            print(f"  image:  {sample.image_path}")
            print(f"  normal: {sample.normal_path}")
            print(f"  mask:   {sample.mask_path if sample.mask_path is not None else '<none>'}")
            print(f"  focal:  {sample.focal_length:.4f}")
        return

    raw_rgb_dir = out_dir / "pred_rgb_raw"
    encoded_dir = out_dir / "pred_normals_encoded"
    angle_dir = out_dir / "angular_error_gray"
    if args.metrics_only:
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        directories = (raw_rgb_dir,) if args.raw_predictions_only else (raw_rgb_dir, encoded_dir, angle_dir)
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    resume = None
    if args.resume_state_dir:
        evaluator_sha256 = (
            LEGACY_UNSEEDED_EVALUATOR_SHA256
            if args.legacy_unseeded
            else hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        )
        resume_config = {
            "evaluator": "surface_normals_flux2_klein",
            "evaluator_sha256": evaluator_sha256,
            "dataset": args.dataset,
            "data_identity": args.resume_data_identity or str(dataset_root.resolve()),
            "split": args.split,
            "cases": list(args.cases),
            "aux_augmentations": list(args.aux_augmentations),
            "model": args.model,
            "model_family": args.model_family,
            "transformer_path": args.transformer_path,
            "lora_path": args.lora_path,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "true_cfg_scale": (
                args.true_cfg_scale if args.model_family == "qwen_edit_2511" else None
            ),
            "prompt": prompt,
            "prompt_sha256": prompt_sha256(prompt),
            "qwen_negative_prompt": qwen_negative_prompt,
            "qwen_negative_prompt_sha256": (
                prompt_sha256(qwen_negative_prompt)
                if qwen_negative_prompt is not None
                else None
            ),
            "prediction_normal_encoding": args.prediction_normal_encoding,
            "gt_normal_encoding": gt_protocol.encoding,
            "gt_normal_permutation": list(gt_protocol.permutation),
            "gt_normal_signs": list(gt_protocol.signs),
            "primary_case": args.primary_case,
            "protocol_version": args.protocol_version,
            "sample_subset": subset_metadata,
            "native_resolution": bool(args.native_resolution),
            "min_normal_norm": args.min_normal_norm,
            "min_valid_pixels": args.min_valid_pixels,
            "eval_mask_erosion_radius": args.eval_mask_erosion_radius,
            "diagnostic_mask_erosion_radius": args.diagnostic_mask_erosion_radius,
            "primary_bidirectional_normals": False,
            "report_bidirectional_diagnostic": report_bidirectional_diagnostic,
            "heldout_calibration_id": (
                heldout_calibration.get("calibration_id") if heldout_calibration else None
            ),
            "test5_oracle_id": (
                test5_oracle.get("calibration_id") if test5_oracle else None
            ),
            "angle_vis_max": args.angle_vis_max,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        }
        if not args.legacy_unseeded:
            resume_config["generation_seed"] = args.seed
            resume_config["generation_seed_mode"] = "fixed_per_sample_reset"
        resume_state_dir = Path(args.resume_state_dir)
        if args.num_shards > 1:
            resume_state_dir = (
                resume_state_dir
                / f"shard-{args.shard_index:04d}-of-{args.num_shards:04d}"
            )
        resume = DurableEvaluationJournal(
            state_dir=resume_state_dir,
            config=resume_config,
            sample_keys=[sample.output_stem for sample in samples],
            output_dir=out_dir,
            chunk_size=args.resume_chunk_size,
        )
        print(
            f"Durable resume: {len(resume.completed_keys)}/{len(samples)} samples already committed "
            f"in {resume_state_dir}"
        )

    remaining = [sample for sample in samples if resume is None or sample.output_stem not in resume.completed_keys]
    pipe = torch = None
    if remaining and prediction_dir is None:
        pipe, torch = load_pipeline(
            model_name=args.model, transformer_path=args.transformer_path,
            model_family=args.model_family, lora_path=args.lora_path, offload=args.model_offload,
        )

    per_sample_records: list[dict[str, float | int | bool | str | None]] = []
    skipped_records: list[dict[str, float | int | bool | str | None]] = []
    per_sample_record_keys: list[str] = []
    skipped_record_keys: list[str] = []
    for sample in tqdm(samples):
        if resume is not None and sample.output_stem in resume.completed_keys:
            continue
        img = Image.open(sample.image_path).convert("RGB")
        width, height = img.size

        save_name = f"{sample.output_stem}.png"
        if prediction_dir is not None:
            prediction_path = prediction_dir / save_name
            if not prediction_path.is_file():
                raise FileNotFoundError(
                    f"Missing archived raw prediction for {sample.output_stem}: "
                    f"{prediction_path}"
                )
            pred_rgb = Image.open(prediction_path).convert("RGB")
        else:
            generation_kwargs = {
                "prompt": prompt,
                "image": img,
                "num_inference_steps": args.steps,
                "guidance_scale": args.guidance_scale,
            }
            generator = make_generation_generator(
                torch,
                args.seed,
                legacy_unseeded=args.legacy_unseeded,
                model_family=args.model_family,
            )
            if generator is not None:
                generation_kwargs["generator"] = generator
            if args.model_family == "qwen_edit_2511":
                generation_kwargs["negative_prompt"] = qwen_negative_prompt
                generation_kwargs["true_cfg_scale"] = args.true_cfg_scale
            if args.native_resolution:
                generation_kwargs["width"] = width
                generation_kwargs["height"] = height
            with torch.inference_mode():
                pred_rgb = pipe(**generation_kwargs).images[0]
            if not args.metrics_only:
                pred_rgb.save(raw_rgb_dir / save_name)

        gt_normals, gt_valid = load_gt_normals(
            sample.normal_path,
            min_norm=args.min_normal_norm,
            rgb_encoding=resolved_gt_rgb_encoding,
        )
        gt_normals = transform_surface_normals(
            gt_normals,
            gt_protocol.permutation,
            gt_protocol.signs,
        )
        raw_eval_mask = load_mask(sample.mask_path, gt_normals.shape[:2], gt_valid)
        eval_mask = erode_mask(raw_eval_mask, args.eval_mask_erosion_radius)

        pred_resized = resize_rgb(pred_rgb, gt_normals.shape[:2])
        representation_diagnostics = normal_representation_diagnostics(
            np.asarray(pred_resized, dtype=np.uint8), raw_eval_mask & gt_valid
        )
        pred_normals, pred_valid = rgb_to_unit_normals(
            pred_resized,
            min_norm=args.min_normal_norm,
            encoding=args.prediction_normal_encoding,
        )
        oracle_pred_normals = rgb_to_oracle_unit_normals(pred_resized)
        if not args.metrics_only and not args.raw_predictions_only:
            Image.fromarray(
                unit_normals_to_rgb(
                    pred_normals,
                    pred_valid,
                    encoding=args.prediction_normal_encoding,
                ),
                mode="RGB",
            ).save(encoded_dir / save_name)

        try:
            metrics, valid_pixels, angle_map = compute_normal_metrics(
                pred_normals=pred_normals,
                pred_valid=pred_valid,
                gt_normals=gt_normals,
                gt_valid=gt_valid,
                eval_mask=eval_mask,
                bidirectional_normals=False,
                min_valid_pixels=args.min_valid_pixels,
            )
        except TooFewValidNormalPixelsError as exc:
            skipped_record = {
                    "split": sample.split,
                    "dataset": sample.dataset,
                    "case": sample.case,
                    "sample_id": sample.sample_id,
                    "image_path": str(sample.image_path),
                    "normal_path": str(sample.normal_path),
                    "mask_path": str(sample.mask_path) if sample.mask_path is not None else None,
                    "focal_length": float(sample.focal_length),
                    "source_index": int(sample.source_index) if sample.source_index is not None else None,
                    "raw_eval_mask_pixels": int(raw_eval_mask.sum()),
                    "eval_mask_pixels": int(eval_mask.sum()),
                    "valid_pixels": exc.valid_pixels,
                    "min_valid_pixels": exc.min_valid_pixels,
                    "skip_reason": "too_few_valid_normal_pixels",
                }
            if resume is not None:
                resume.add(
                    "skipped",
                    sample.output_stem,
                    skipped_record,
                    (raw_rgb_dir / save_name, encoded_dir / save_name),
                )
            else:
                skipped_records.append(skipped_record)
                skipped_record_keys.append(sample.output_stem)
            continue

        if not args.metrics_only and not args.raw_predictions_only:
            Image.fromarray(angle_to_gray(angle_map, args.angle_vis_max), mode="L").save(angle_dir / save_name)

        record = {
            "split": sample.split,
            "dataset": sample.dataset,
            "case": sample.case,
            "sample_id": sample.sample_id,
            "image_path": str(sample.image_path),
            "normal_path": str(sample.normal_path),
            "mask_path": str(sample.mask_path) if sample.mask_path is not None else None,
            "focal_length": float(sample.focal_length),
            "source_index": int(sample.source_index) if sample.source_index is not None else None,
            "raw_eval_mask_pixels": int(raw_eval_mask.sum()),
            "eval_mask_pixels": int(eval_mask.sum()),
            "valid_pixels": valid_pixels,
            **{f"representation_{key}": value for key, value in representation_diagnostics.items()},
            **metrics,
            **pooled_normal_statistics(angle_map, eval_mask & gt_valid),
        }
        marigold_angles = np.asarray(angle_map)[eval_mask & gt_valid]
        record["marigold_angle_5"] = float(np.mean(marigold_angles < 5.0))
        record["marigold_angle_7_5"] = float(np.mean(marigold_angles < 7.5))
        # Preserve the former evaluator behavior only as an explicitly named
        # diagnostic.  It drops invalid predictions and therefore can look
        # artificially better when a model emits neutral/zero normal pixels.
        try:
            valid_only_metrics, valid_only_pixels, _ = compute_normal_metrics(
                pred_normals=pred_normals,
                pred_valid=pred_valid,
                gt_normals=gt_normals,
                gt_valid=gt_valid,
                eval_mask=eval_mask,
                bidirectional_normals=False,
                min_valid_pixels=args.min_valid_pixels,
                valid_prediction_only=True,
            )
            record["valid_only_diagnostic_pixels"] = valid_only_pixels
            record.update(
                {
                    f"valid_only_diagnostic_{name}": value
                    for name, value in valid_only_metrics.items()
                }
            )
        except TooFewValidNormalPixelsError as exc:
            record["valid_only_diagnostic_pixels"] = exc.valid_pixels
            record["valid_only_diagnostic_skip_reason"] = "too_few_valid_normal_pixels"
        if report_bidirectional_diagnostic:
            try:
                bidirectional_metrics, bidirectional_valid_pixels, _ = compute_normal_metrics(
                    pred_normals=pred_normals,
                    pred_valid=pred_valid,
                    gt_normals=gt_normals,
                    gt_valid=gt_valid,
                    eval_mask=eval_mask,
                    bidirectional_normals=True,
                    min_valid_pixels=args.min_valid_pixels,
                )
                record["bidirectional_valid_pixels"] = bidirectional_valid_pixels
                record.update({f"bidirectional_{name}": value for name, value in bidirectional_metrics.items()})
            except TooFewValidNormalPixelsError as exc:
                record["bidirectional_valid_pixels"] = exc.valid_pixels
                record["bidirectional_skip_reason"] = "too_few_valid_normal_pixels"

        if test5_convention is not None:
            oracle_metrics = convention_oracle_normal_metrics(
                    oracle_pred_normals,
                    gt_normals,
                    eval_mask & gt_valid,
                    test5_convention,
                    # The frozen test-five compatibility lane follows its
                    # reference scorer's ten-pixel inclusion threshold. It is
                    # intentionally independent of the stricter primary-lane
                    # dataset quality gate configured by the launcher.
                    min_valid_pixels=10,
                )
            record["test5_axis_oracle_valid_pixels"] = int((eval_mask & gt_valid).sum())
            record.update(
                {
                    f"test5_axis_oracle_{name}": value
                    for name, value in oracle_metrics.items()
                }
            )
        if heldout_convention is not None:
            calibrated_normals = heldout_convention.apply(pred_normals)
            frozen_metrics, frozen_valid_pixels, _ = compute_normal_metrics(
                pred_normals=calibrated_normals,
                pred_valid=pred_valid,
                gt_normals=gt_normals,
                gt_valid=gt_valid,
                eval_mask=eval_mask,
                bidirectional_normals=False,
                min_valid_pixels=args.min_valid_pixels,
            )
            record["heldout_calibrated_valid_pixels"] = frozen_valid_pixels
            record.update(
                {
                    f"heldout_calibrated_{name}": value
                    for name, value in frozen_metrics.items()
                }
            )
        if heldout_so3_rotation is not None:
            so3_calibrated_normals = heldout_so3_rotation.apply(pred_normals)
            so3_metrics, so3_valid_pixels, _ = compute_normal_metrics(
                pred_normals=so3_calibrated_normals,
                pred_valid=pred_valid,
                gt_normals=gt_normals,
                gt_valid=gt_valid,
                eval_mask=eval_mask,
                bidirectional_normals=False,
                min_valid_pixels=args.min_valid_pixels,
            )
            record["heldout_so3_calibrated_valid_pixels"] = so3_valid_pixels
            record.update(
                {
                    f"heldout_so3_calibrated_{name}": value
                    for name, value in so3_metrics.items()
                }
            )

        if args.diagnostic_mask_erosion_radius > 0:
            diagnostic_mask = erode_mask(raw_eval_mask, args.diagnostic_mask_erosion_radius)
            record["diagnostic_mask_erosion_radius"] = int(args.diagnostic_mask_erosion_radius)
            record["diagnostic_eval_mask_pixels"] = int(diagnostic_mask.sum())
            try:
                diagnostic_metrics, diagnostic_valid_pixels, _ = compute_normal_metrics(
                    pred_normals=pred_normals,
                    pred_valid=pred_valid,
                    gt_normals=gt_normals,
                    gt_valid=gt_valid,
                    eval_mask=diagnostic_mask,
                    bidirectional_normals=False,
                    min_valid_pixels=args.min_valid_pixels,
                )
                record["diagnostic_valid_pixels"] = diagnostic_valid_pixels
                record.update({f"diagnostic_mask_{name}": value for name, value in diagnostic_metrics.items()})
            except TooFewValidNormalPixelsError as exc:
                record["diagnostic_valid_pixels"] = exc.valid_pixels
                record["diagnostic_mask_skip_reason"] = "too_few_valid_normal_pixels"
        if resume is not None:
            resume.add(
                "record",
                sample.output_stem,
                record,
                (raw_rgb_dir / save_name, encoded_dir / save_name, angle_dir / save_name),
            )
        else:
            per_sample_records.append(record)
            per_sample_record_keys.append(sample.output_stem)

    if resume is not None:
        resume.flush()
        record_items = resume.records_with_keys_in_sample_order("record")
        skipped_items = resume.records_with_keys_in_sample_order("skipped")
        per_sample_record_keys = [key for key, _ in record_items]
        per_sample_records = [record for _, record in record_items]
        skipped_record_keys = [key for key, _ in skipped_items]
        skipped_records = [record for _, record in skipped_items]

    if args.num_shards > 1:
        full_sample_keys = [sample.output_stem for sample in full_samples]
        identity_args = {
            key: value
            for key, value in vars(args).items()
            if key
            not in {
                "shard_index",
                "shard_sync_dir",
                "shard_sync_timeout_seconds",
                "resume_state_dir",
            }
        }
        run_identity = sharding_run_identity(
            identity_args, full_sample_keys, args.num_shards
        )
        merged = synchronize_shard_records(
            sync_dir=args.shard_sync_dir,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            run_identity=run_identity,
            full_sample_keys=full_sample_keys,
            record_items=list(zip(per_sample_record_keys, per_sample_records)),
            skipped_items=list(zip(skipped_record_keys, skipped_records)),
            timeout_seconds=args.shard_sync_timeout_seconds,
        )
        if merged is None:
            if resume is not None:
                resume.close()
            print(
                f"Evaluation shard {args.shard_index}/{args.num_shards} published; "
                "aggregate is written by shard zero."
            )
            return
        per_sample_records, skipped_records = merged
        samples = full_samples

    if not per_sample_records:
        raise RuntimeError(
            f"No surface-normal samples were evaluated successfully; skipped {len(skipped_records)} samples."
        )

    grouped_records = records_by_case(per_sample_records)
    if args.primary_case:
        if args.primary_case not in grouped_records:
            raise ValueError(
                f"Requested primary case {args.primary_case!r} is absent; "
                f"resolved cases: {', '.join(sorted(grouped_records))}"
            )
        primary_records = grouped_records[args.primary_case]
    elif args.dataset == "ibims1":
        primary_records = grouped_records.get("standard", per_sample_records)
    else:
        primary_records = per_sample_records
    metrics_summary = aggregate_metrics(primary_records)
    combined_metrics_summary = aggregate_metrics(per_sample_records)
    metrics_by_case = {
        case: aggregate_metrics(case_records)
        for case, case_records in sorted(grouped_records.items())
    }
    marigold_metrics_summary = marigold_v1_1_metrics(metrics_summary, primary_records)
    marigold_metrics_by_case = {
        case: marigold_v1_1_metrics(case_metrics, grouped_records[case])
        for case, case_metrics in metrics_by_case.items()
    }
    benchmark_pooled_metrics = aggregate_pooled_normal_statistics(primary_records)
    benchmark_pooled_metrics_by_case = {
        case: aggregate_pooled_normal_statistics(case_records)
        for case, case_records in sorted(grouped_records.items())
    }
    bidirectional_diagnostic_metrics = aggregate_prefixed_metrics(primary_records, "bidirectional")
    bidirectional_diagnostic_metrics_by_case = {
        case: case_metrics
        for case, case_metrics in (
            (case, aggregate_prefixed_metrics(case_records, "bidirectional"))
            for case, case_records in sorted(grouped_records.items())
        )
        if case_metrics is not None
    }
    diagnostic_mask_metrics = aggregate_prefixed_metrics(primary_records, "diagnostic_mask")
    diagnostic_mask_metrics_by_case = {
        case: case_metrics
        for case, case_metrics in (
            (case, aggregate_prefixed_metrics(case_records, "diagnostic_mask"))
            for case, case_records in sorted(grouped_records.items())
        )
        if case_metrics is not None
    }
    representation_diagnostics = aggregate_diagnostic_fields(primary_records)
    valid_only_diagnostic_metrics = aggregate_prefixed_metrics(
        primary_records, "valid_only_diagnostic"
    )
    valid_only_diagnostic_metrics_by_case = {
        case: case_metrics
        for case, case_metrics in (
            (case, aggregate_prefixed_metrics(case_records, "valid_only_diagnostic"))
            for case, case_records in sorted(grouped_records.items())
        )
        if case_metrics is not None
    }
    coverage = float(np.mean([float(record["coverage"]) for record in primary_records]))
    coverage_by_case = {
        case: float(np.mean([float(record["coverage"]) for record in case_records]))
        for case, case_records in sorted(grouped_records.items())
    }
    heldout_calibrated_metrics = aggregate_prefixed_metrics(
        primary_records, "heldout_calibrated"
    )
    heldout_calibrated_metrics_by_case = {
        case: case_metrics
        for case, case_metrics in (
            (case, aggregate_prefixed_metrics(case_records, "heldout_calibrated"))
            for case, case_records in sorted(grouped_records.items())
        )
        if case_metrics is not None
    }
    heldout_so3_calibrated_metrics = aggregate_prefixed_metrics(
        primary_records, "heldout_so3_calibrated"
    )
    heldout_so3_calibrated_metrics_by_case = {
        case: case_metrics
        for case, case_metrics in (
            (case, aggregate_prefixed_metrics(case_records, "heldout_so3_calibrated"))
            for case, case_records in sorted(grouped_records.items())
        )
        if case_metrics is not None
    }
    test5_axis_oracle_metrics = aggregate_prefixed_metrics(
        primary_records, "test5_axis_oracle"
    )
    test5_axis_oracle_metrics_by_case = {
        case: case_metrics
        for case, case_metrics in (
            (case, aggregate_prefixed_metrics(case_records, "test5_axis_oracle"))
            for case, case_records in sorted(grouped_records.items())
        )
        if case_metrics is not None
    }

    durable_resume_summary = (
        resume.summary() if resume is not None else {"enabled": False}
    )
    if args.num_shards > 1 and resume is not None:
        durable_resume_summary = summarize_sharded_journals(
            args.resume_state_dir,
            num_shards=args.num_shards,
            committed_samples=len(per_sample_records) + len(skipped_records),
            chunk_size=args.resume_chunk_size,
        )

    summary = {
        "dataset": args.dataset,
        "dataset_root": str(dataset_root),
        "dataset_provenance": dataset_provenance,
        "requested_split": args.split,
        "resolved_splits": resolved_splits,
        "resolved_cases": resolved_cases,
        "primary_case": args.primary_case,
        "images_requested": len(samples),
        "images_evaluated": len(per_sample_records),
        "images_skipped": len(skipped_records),
        "sample_subset": subset_metadata,
        "metrics": metrics_summary,
        "marigold_v1_1_metrics": marigold_metrics_summary,
        "marigold_v1_1_metrics_by_case": marigold_metrics_by_case,
        "benchmark_standard_pooled_metrics": benchmark_pooled_metrics,
        "benchmark_standard_pooled_metrics_by_case": benchmark_pooled_metrics_by_case,
        "combined_metrics": combined_metrics_summary,
        "metrics_by_case": metrics_by_case,
        "coverage": coverage,
        "coverage_by_case": coverage_by_case,
        "valid_only_diagnostic_metrics": valid_only_diagnostic_metrics,
        "valid_only_diagnostic_metrics_by_case": valid_only_diagnostic_metrics_by_case,
        "heldout_calibrated_metrics": heldout_calibrated_metrics,
        "heldout_calibrated_metrics_by_case": heldout_calibrated_metrics_by_case,
        "test5_axis_oracle_metrics": test5_axis_oracle_metrics,
        "test5_axis_oracle_metrics_by_case": test5_axis_oracle_metrics_by_case,
        "bidirectional_diagnostic_metrics": bidirectional_diagnostic_metrics,
        "bidirectional_diagnostic_metrics_by_case": bidirectional_diagnostic_metrics_by_case,
        "diagnostic_mask_metrics": diagnostic_mask_metrics,
        "diagnostic_mask_metrics_by_case": diagnostic_mask_metrics_by_case,
        "representation_diagnostics": representation_diagnostics,
        "metrics_protocol": (
            f"Top-level metrics select case {args.primary_case!r}; all evaluated "
            "cases are retained under *_metrics_by_case."
            if args.primary_case
            else (
            "For iBims-1 multi-case runs, top-level metrics are computed on the standard case only; "
            "auxiliary results are reported under metrics_by_case."
            if args.dataset == "ibims1" and len(metrics_by_case) > 1
            else (
                "For DIODE multi-case runs, top-level metrics are combined over evaluated samples; "
                "indoor and outdoor results are reported under metrics_by_case."
                if args.dataset == "diode" and len(metrics_by_case) > 1
                else (
                    "nuScenes metrics are computed only on sparse projected LiDAR-derived normal valid pixels; "
                    "report as sparse LiDAR-derived outdoor-driving evaluation."
                    if args.dataset == "nuscenes"
                    else "Top-level metrics are computed over all evaluated samples."
                )
            )
            )
        ),
        "generation": {
            "source": "archived_raw_predictions" if prediction_dir is not None else "model_inference",
            "prediction_dir": str(prediction_dir) if prediction_dir is not None else None,
            "model": args.model,
            "model_family": args.model_family,
            "transformer_path": args.transformer_path,
            "lora_path": args.lora_path,
            "prompt": prompt,
            "prompt_sha256": prompt_sha256(prompt),
            "negative_prompt": qwen_negative_prompt,
            "negative_prompt_sha256": (
                prompt_sha256(qwen_negative_prompt)
                if qwen_negative_prompt is not None
                else None
            ),
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "true_cfg_scale": (
                args.true_cfg_scale if args.model_family == "qwen_edit_2511" else None
            ),
            "seed": None if args.legacy_unseeded else args.seed,
            "seed_mode": (
                "legacy_process_rng"
                if args.legacy_unseeded
                else "fixed_per_sample_reset"
            ),
            "native_resolution": bool(args.native_resolution),
        },
        "normal_evaluation": {
            "protocol_version": args.protocol_version,
            "dataset_role": args.dataset_role,
            "sample_subset": subset_metadata,
            "configuration": normalized_normal_configuration,
            "configuration_sha256": normal_configuration_sha256,
            "primary_protocol": NORMAL_STRICT,
            "primary_denominator": "full eval_mask AND GT-valid mask",
            "invalid_prediction_policy": "90 degree angular error; retained in denominator",
            "valid_only_diagnostic_protocol": NORMAL_VALID_ONLY_DIAGNOSTIC,
            "bidirectional_diagnostic_protocol": NORMAL_BIDIRECTIONAL_DIAGNOSTIC,
            "heldout_calibration_protocol": NORMAL_HELDOUT_CALIBRATED,
            "heldout_calibration": heldout_calibration,
            "test5_oracle_protocol": NORMAL_TEST5_ORACLE,
            "test5_oracle": test5_oracle,
            "prediction_encoding": args.prediction_normal_encoding,
            "prediction_encoding_formula": normal_encoding_formula(
                args.prediction_normal_encoding
            ),
            "gt_encoding": (
                gt_protocol.encoding
            ),
            "gt_encoding_formula": (
                "HWC/CHW float XYZ; finite nonzero vectors normalized before scoring"
                if gt_storage_formats == ["npy"]
                else normal_encoding_formula(resolved_gt_rgb_encoding)
            ),
            "gt_storage_formats": gt_storage_formats,
            "gt_to_canonical_permutation": list(gt_protocol.permutation),
            "gt_to_canonical_signs": list(gt_protocol.signs),
            "gt_protocol_resolution": gt_protocol.source,
            "min_normal_norm": args.min_normal_norm,
            "min_valid_pixels": args.min_valid_pixels,
            "eval_mask_erosion_radius": args.eval_mask_erosion_radius,
            "diagnostic_mask_erosion_radius": args.diagnostic_mask_erosion_radius,
            "primary_bidirectional_normals": False,
            "aggregation_lanes": {
                "metrics": (
                    "per-image arithmetic mean, matching Marigold V1.1 aggregation; "
                    "angle threshold fields use fractions in [0,1]"
                ),
                "marigold_v1_1_metrics": (
                    "same per-image aggregation under Marigold's published field names; "
                    "sub-threshold fields use percentages in [0,100]"
                ),
                "benchmark_standard_pooled_metrics": (
                    "global valid-pixel pooling retained as a separately labeled diagnostic; "
                    "do not substitute it for Marigold V1.1 per-image table values"
                ),
                "benchmark_standard_pooled_median": (
                    "0.01-degree durable histogram approximation; maximum binning "
                    "error is 0.005 degrees"
                ),
            },
            "report_bidirectional_diagnostic": report_bidirectional_diagnostic,
            "angle_vis_max": args.angle_vis_max,
        },
        "durable_resume": durable_resume_summary,
    }
    # Preserve the historical result schema byte-for-byte unless this new lane
    # is explicitly requested. In particular, absent/None placeholder fields
    # are deliberately not written into existing or currently running results.
    if heldout_so3_rotation is not None:
        summary["heldout_so3_calibrated_metrics"] = heldout_so3_calibrated_metrics
        summary["heldout_so3_calibrated_metrics_by_case"] = (
            heldout_so3_calibrated_metrics_by_case
        )
        summary["normal_evaluation"]["heldout_so3_calibration_protocol"] = (
            NORMAL_HELDOUT_SO3_CALIBRATED
        )
        summary["normal_evaluation"]["heldout_so3_calibration"] = (
            heldout_so3_calibration
        )
    if args.num_shards > 1:
        summary["execution_sharding"] = {
            "strategy": "round_robin_process_per_gpu",
            "num_shards": args.num_shards,
            "aggregation": "global_sample_order_before_metric_aggregation",
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    split_name = normal_datasets.canonicalize_split_name(args.dataset, args.split)
    summary_path = out_dir / f"{args.dataset}_surface_normals_{split_name}_results.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    per_sample_path = out_dir / f"{args.dataset}_surface_normals_{split_name}_per_sample.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as handle:
        for record in per_sample_records:
            handle.write(json.dumps(record) + "\n")

    skipped_path = out_dir / f"{args.dataset}_surface_normals_{split_name}_skipped.jsonl"
    if skipped_records:
        with skipped_path.open("w", encoding="utf-8") as handle:
            for record in skipped_records:
                handle.write(json.dumps(record) + "\n")

    print("\nEvaluation Results")
    print("=" * 50)
    print(f"Dataset root: {dataset_root}")
    print(f"Resolved splits: {', '.join(resolved_splits)}")
    print(f"Resolved cases: {', '.join(resolved_cases)}")
    print(f"Images requested: {len(samples)}")
    print(f"Images evaluated: {len(per_sample_records)}")
    print(f"Images skipped: {len(skipped_records)}")
    for name, value in metrics_summary.items():
        print(f"{name}: {value:.6f}")
    print("=" * 50)
    print(f"Saved summary to: {summary_path}")
    print(f"Saved per-sample metrics to: {per_sample_path}")
    if skipped_records:
        print(f"Saved skipped samples to: {skipped_path}")
    if not args.metrics_only:
        print(f"Raw RGB predictions: {raw_rgb_dir}")
        print(f"Encoded normal predictions: {encoded_dir}")
        print(f"Angular error visualizations: {angle_dir}")


if __name__ == "__main__":
    main()
