#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from common.geometry_protocol_prompts import CANONICAL_DEPTH_PROMPT
from dataset_prep.depth.encoding import (
    DEPTH_ENCODING_GRAYSCALE,
    DEPTH_ENCODING_VISION_BANANA_RGB,
    DEPTH_ENCODINGS,
    encode_depth_to_vision_banana_rgb,
    valid_depth_mask,
)
from dataset_prep.geometry_conventions import (
    CANONICAL_DEPTH_QUANTITY,
    GEOMETRY_DATA_VERSION,
    resize_scalar_field,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda iterable, **_: iterable


GRAYSCALE_DEPTH_PROMPT = (
    "[TASK: DEPTH_ESTIMATION] Generate a dense monocular depth map of the input driving scene. "
    "Output only a clean grayscale depth image with no text, labels, legends, or extra colors. "
    "Use smooth depth gradients, keep object boundaries sharp, and make nearer surfaces brighter "
    "while farther surfaces are darker."
)
VISION_BANANA_DEPTH_PROMPT = CANONICAL_DEPTH_PROMPT

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
_RESAMPLING = getattr(Image, "Resampling", Image)


@dataclass(frozen=True)
class SynscapesPair:
    stem: str
    rgb_ref: Any
    depth_ref: Any


def resolve_synscapes_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    if (root / "img" / "depth").is_dir() and (root / "img").is_dir():
        return root
    nested = root / "Synscapes"
    if (nested / "img" / "depth").is_dir() and (nested / "img").is_dir():
        return nested
    raise FileNotFoundError(
        "Could not resolve Synscapes root. Expected either <root>/img/depth "
        f"or <root>/Synscapes/img/depth under: {root}"
    )


def list_root_pairs(root: Path, rgb_subdir: str) -> list[SynscapesPair]:
    root = resolve_synscapes_root(root)
    rgb_root = root / "img" / rgb_subdir
    depth_root = root / "img" / "depth"
    if not rgb_root.is_dir():
        raise FileNotFoundError(f"Missing Synscapes RGB directory: {rgb_root}")
    if not depth_root.is_dir():
        raise FileNotFoundError(f"Missing Synscapes depth directory: {depth_root}")

    rgb_by_stem = {
        path.stem: path
        for path in rgb_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    depth_by_stem = {
        path.stem: path
        for path in depth_root.iterdir()
        if path.is_file() and path.suffix.lower() == ".exr"
    }
    stems = sorted(set(rgb_by_stem) & set(depth_by_stem), key=lambda value: int(value) if value.isdigit() else value)
    return [SynscapesPair(stem, rgb_by_stem[stem], depth_by_stem[stem]) for stem in stems]


def list_tar_pairs(tar_path: Path, rgb_subdir: str) -> list[SynscapesPair]:
    tar_path = tar_path.expanduser().resolve()
    if not tar_path.is_file():
        raise FileNotFoundError(f"Synscapes tar does not exist: {tar_path}")

    rgb_prefix = f"Synscapes/img/{rgb_subdir}/"
    depth_prefix = "Synscapes/img/depth/"
    with tarfile.open(tar_path, "r:*") as archive:
        rgb_by_stem = {}
        depth_by_stem = {}
        for member in archive.getmembers():
            if not member.isfile():
                continue
            name = member.name
            path = Path(name)
            if name.startswith(rgb_prefix) and path.suffix.lower() in IMAGE_SUFFIXES:
                rgb_by_stem[path.stem] = name
            elif name.startswith(depth_prefix) and path.suffix.lower() == ".exr":
                depth_by_stem[path.stem] = name

    stems = sorted(set(rgb_by_stem) & set(depth_by_stem), key=lambda value: int(value) if value.isdigit() else value)
    return [SynscapesPair(stem, rgb_by_stem[stem], depth_by_stem[stem]) for stem in stems]


def sample_pairs(pairs: list[SynscapesPair], limit: int | None, seed: int) -> list[SynscapesPair]:
    if limit is None or limit >= len(pairs):
        return pairs
    rng = random.Random(seed)
    sampled = list(pairs)
    rng.shuffle(sampled)
    return sorted(sampled[:limit], key=lambda pair: int(pair.stem) if pair.stem.isdigit() else pair.stem)


def resize_to_max_side(image: Image.Image, max_side: int | None, resample: int) -> Image.Image:
    if max_side is None:
        return image
    width, height = image.size
    largest = max(width, height)
    if largest <= max_side:
        return image
    scale = max_side / largest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, resample=resample)


def resize_depth_to_size(depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if depth.shape == (size[1], size[0]):
        return depth.astype(np.float32, copy=False)

    finite = np.isfinite(depth)
    values = np.where(finite, depth, 0.0).astype(np.float32, copy=False)
    weights = finite.astype(np.float32, copy=False)

    value_image = Image.fromarray(values, mode="F").resize(size, resample=_RESAMPLING.BILINEAR)
    weight_image = Image.fromarray(weights, mode="F").resize(size, resample=_RESAMPLING.BILINEAR)
    resized_values = np.asarray(value_image, dtype=np.float32)
    resized_weights = np.asarray(weight_image, dtype=np.float32)

    out = np.full(resized_values.shape, np.nan, dtype=np.float32)
    valid = resized_weights > 1e-6
    out[valid] = resized_values[valid] / resized_weights[valid]
    return out


def select_depth_channel(depth: np.ndarray, channel: str) -> np.ndarray:
    if depth.ndim == 2:
        return depth
    if depth.ndim != 3:
        raise ValueError(f"Expected 2D or 3D EXR depth array, got shape={depth.shape}")
    if channel == "mean":
        return np.nanmean(depth, axis=2)
    channel_map = {
        "first": 0,
        "blue": 0,
        "green": 1,
        "red": 2,
        "last": depth.shape[2] - 1,
    }
    index = channel_map[channel]
    if index >= depth.shape[2]:
        raise ValueError(f"Depth channel {channel!r} is unavailable for EXR shape={depth.shape}")
    return depth[..., index]


def read_exr_with_cv2(path: Path | None, data: bytes | None) -> np.ndarray | None:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    try:
        import cv2
    except ImportError:
        return None

    if data is not None:
        encoded = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    return np.asarray(image, dtype=np.float32)


def read_exr_with_imageio(path: Path | None, data: bytes | None) -> np.ndarray | None:
    try:
        import imageio.v3 as iio
    except ImportError:
        return None

    if data is not None:
        with tempfile.NamedTemporaryFile(suffix=".exr") as handle:
            handle.write(data)
            handle.flush()
            return np.asarray(iio.imread(handle.name), dtype=np.float32)
    return np.asarray(iio.imread(path), dtype=np.float32)


def read_exr_with_openexr(path: Path | None, data: bytes | None) -> np.ndarray | None:
    try:
        import Imath
        import OpenEXR
    except ImportError:
        return None

    temp_name = None
    if data is not None:
        with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as handle:
            handle.write(data)
            temp_name = handle.name
        path = Path(temp_name)

    try:
        exr = OpenEXR.InputFile(str(path))
        header = exr.header()
        window = header["dataWindow"]
        width = window.max.x - window.min.x + 1
        height = window.max.y - window.min.y + 1
        channels = list(header["channels"])
        if not channels:
            raise ValueError(f"EXR file has no channels: {path}")
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        arrays = [
            np.frombuffer(exr.channel(channel, pixel_type), dtype=np.float32).reshape(height, width)
            for channel in channels[:3]
        ]
        if len(arrays) == 1:
            return arrays[0]
        return np.stack(arrays, axis=2)
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except OSError:
                pass


def load_exr_depth(path: Path | None = None, data: bytes | None = None, channel: str = "first") -> np.ndarray:
    readers: tuple[Callable[[Path | None, bytes | None], np.ndarray | None], ...] = (
        read_exr_with_cv2,
        read_exr_with_imageio,
        read_exr_with_openexr,
    )
    errors = []
    for reader in readers:
        try:
            depth = reader(path, data)
        except Exception as exc:
            errors.append(f"{reader.__name__}: {exc}")
            continue
        if depth is not None:
            return select_depth_channel(depth, channel).astype(np.float32, copy=False)
    detail = f" Reader errors: {'; '.join(errors)}" if errors else ""
    raise RuntimeError(
        "Could not read Synscapes EXR depth. Install one EXR-capable reader "
        "(opencv-python-headless with OPENCV_IO_ENABLE_OPENEXR=1, imageio, or OpenEXR)."
        f"{detail}"
    )


def depth_to_uint8(
    depth: np.ndarray,
    *,
    mode: str,
    min_depth: float,
    max_depth: float,
    low_percentile: float,
    high_percentile: float,
    invalid_value: int,
) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if min_depth > 0:
        valid &= depth >= min_depth
    if max_depth > 0:
        valid &= depth <= max_depth
    if not np.any(valid):
        return np.full(depth.shape, invalid_value, dtype=np.uint8)

    clipped = depth.astype(np.float32, copy=True)
    if min_depth > 0 or max_depth > 0:
        lower = min_depth if min_depth > 0 else float(np.nanmin(clipped[valid]))
        upper = max_depth if max_depth > 0 else float(np.nanmax(clipped[valid]))
        clipped = np.clip(clipped, lower, upper)

    if mode == "inverse":
        signal = 1.0 / np.maximum(clipped, 1e-6)
    elif mode == "log":
        signal = -np.log(np.maximum(clipped, 1e-6))
    elif mode == "linear":
        signal = -clipped
    else:
        raise ValueError(f"Unsupported depth normalization mode: {mode}")

    signal_valid = signal[valid]
    lo = float(np.percentile(signal_valid, low_percentile))
    hi = float(np.percentile(signal_valid, high_percentile))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo = float(signal_valid.min())
        hi = float(signal_valid.max())
    if hi <= lo:
        out = np.full(depth.shape, invalid_value, dtype=np.uint8)
        out[valid] = 127
        return out

    normalized = (signal - lo) / (hi - lo)
    normalized = np.clip(normalized, 0.0, 1.0)
    out = np.rint(normalized * 255.0).astype(np.uint8)
    out[~valid] = np.uint8(np.clip(invalid_value, 0, 255))
    return out


def save_depth_raw_and_mask(
    *,
    out_root: Path,
    depth_rel: Path,
    mask_rel: Path,
    depth_m: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    raw_depth = depth_m.astype(np.float32, copy=True)
    raw_depth[~valid_mask] = np.nan
    (out_root / depth_rel).parent.mkdir(parents=True, exist_ok=True)
    (out_root / mask_rel).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_root / depth_rel, raw_depth)
    Image.fromarray((valid_mask.astype(np.uint8) * 255), mode="L").save(out_root / mask_rel)


def load_rgb_from_root(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def load_rgb_from_tar(archive: tarfile.TarFile, member_name: str) -> Image.Image:
    extracted = archive.extractfile(member_name)
    if extracted is None:
        raise FileNotFoundError(f"Could not extract tar member: {member_name}")
    with Image.open(io.BytesIO(extracted.read())) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def read_member_bytes(archive: tarfile.TarFile, member_name: str) -> bytes:
    extracted = archive.extractfile(member_name)
    if extracted is None:
        raise FileNotFoundError(f"Could not extract tar member: {member_name}")
    return extracted.read()


def write_pair(
    pair: SynscapesPair,
    *,
    out_root: Path,
    rgb_loader: Callable[[Any], Image.Image],
    depth_loader: Callable[[Any], np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    basename = f"synscapes_{int(pair.stem):05d}" if pair.stem.isdigit() else f"synscapes_{pair.stem}"
    cond_rel = Path("conditioning_images") / f"{basename}_input.png"
    target_name = (
        f"{basename}_depth.png"
        if args.depth_encoding == DEPTH_ENCODING_GRAYSCALE
        else f"{basename}_depth_vb_rgb.png"
    )
    target_rel = Path("target_images") / target_name
    depth_raw_rel = Path("depth_raw") / f"{basename}.npy"
    valid_mask_rel = Path("valid_masks") / f"{basename}.png"

    rgb = rgb_loader(pair.rgb_ref)
    depth = depth_loader(pair.depth_ref)

    max_side = args.max_side if args.max_side > 0 else None
    rgb = resize_to_max_side(rgb, max_side, _RESAMPLING.LANCZOS)
    native_valid = valid_depth_mask(
        depth, min_depth=args.min_depth, max_depth=args.max_depth
    )
    depth_for_metadata, resized_valid = resize_scalar_field(
        depth, native_valid, rgb.size
    )
    metadata_valid = resized_valid & valid_depth_mask(
        depth_for_metadata, min_depth=args.min_depth, max_depth=args.max_depth
    )

    if args.depth_encoding == DEPTH_ENCODING_GRAYSCALE:
        # Preserve the legacy grayscale target behavior: normalize the native
        # depth map first, then resize the generated target image.
        depth_u8 = depth_to_uint8(
            depth,
            mode=args.normalization,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            low_percentile=args.low_percentile,
            high_percentile=args.high_percentile,
            invalid_value=args.invalid_value,
        )
        target = Image.fromarray(depth_u8, mode="L").convert("RGB")
        target = resize_to_max_side(target, max_side, _RESAMPLING.BILINEAR)
        if target.size != rgb.size:
            target = target.resize(rgb.size, resample=_RESAMPLING.BILINEAR)
    elif args.depth_encoding == DEPTH_ENCODING_VISION_BANANA_RGB:
        target_rgb = encode_depth_to_vision_banana_rgb(
            depth_for_metadata,
            valid_mask=metadata_valid,
            invalid_color=args.invalid_rgb,
        )
        target = Image.fromarray(target_rgb, mode="RGB")
    else:
        raise ValueError(f"Unsupported depth encoding: {args.depth_encoding}")

    (out_root / cond_rel).parent.mkdir(parents=True, exist_ok=True)
    (out_root / target_rel).parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out_root / cond_rel)
    target.save(out_root / target_rel)
    save_depth_raw_and_mask(
        out_root=out_root,
        depth_rel=depth_raw_rel,
        mask_rel=valid_mask_rel,
        depth_m=depth_for_metadata,
        valid_mask=metadata_valid,
    )

    return {
        "conditioning_image": cond_rel.as_posix(),
        "image": target_rel.as_posix(),
        "depth_raw": depth_raw_rel.as_posix(),
        "loss_mask": valid_mask_rel.as_posix(),
        "valid_mask": valid_mask_rel.as_posix(),
        "text": args.prompt,
        "task": "depth_estimation",
        "source_dataset": "synscapes",
        "depth_encoding": args.depth_encoding,
        "id": basename,
        "scene": basename,
        "preprocess_target": "bilinear",
        "metadata": {
            "stem": pair.stem,
            "scene": basename,
            "normalization": args.normalization,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "low_percentile": args.low_percentile,
            "high_percentile": args.high_percentile,
            "near_is_bright": True,
            "depth_encoding": args.depth_encoding,
            "geometry_version": GEOMETRY_DATA_VERSION,
            "canonical_depth_quantity": CANONICAL_DEPTH_QUANTITY,
            "native_depth_quantity": "renderer_projective_z_distance_from_image_plane",
            "source_depth_evidence": (
                "Synscapes depth EXR and per-frame camera intrinsics; existing "
                "canonical normal preparation treats the published renderer "
                "depth as z-buffer/image-plane distance."
            ),
            "target_preparation": (
                "bilinear"
                if args.depth_encoding == DEPTH_ENCODING_GRAYSCALE
                else "metric_field_then_vision_banana_rgb_encode"
            ),
            "resize_policy": "metric_field_then_encode_with_hard_mask",
            "mask_policy": "hard_nearest_neighbor",
            "raw_depth_units": "meters",
        },
    }


def prepare_from_root(pairs: list[SynscapesPair], out_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    def process(pair: SynscapesPair) -> dict[str, Any] | None:
        try:
            return write_pair(
                pair,
                out_root=out_root,
                rgb_loader=load_rgb_from_root,
                depth_loader=lambda path: load_exr_depth(path=path, channel=args.depth_channel),
                args=args,
            )
        except Exception as exc:
            if args.strict:
                raise
            print(f"WARNING: skipped {pair.stem}: {exc}", flush=True)
            return None

    if args.num_workers <= 1:
        rows = [process(pair) for pair in tqdm(pairs, desc="Preparing Synscapes depth")]
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            rows = list(
                tqdm(
                    executor.map(process, pairs),
                    total=len(pairs),
                    desc=f"Preparing Synscapes depth ({args.num_workers} workers)",
                )
            )
    return [row for row in rows if row is not None]


def prepare_from_tar(tar_path: Path, pairs: list[SynscapesPair], out_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    # TarFile.extractfile uses a shared seek cursor and therefore serializes this
    # expensive EXR path. Resolve byte offsets once, then use thread-safe pread so
    # --num-workers accelerates archived inputs as well as extracted folders.
    required = {str(pair.rgb_ref) for pair in pairs} | {str(pair.depth_ref) for pair in pairs}
    with tarfile.open(tar_path, "r:*") as archive:
        offsets = {
            member.name: (member.offset_data, member.size)
            for member in archive
            if member.isfile() and member.name in required
        }
    missing = required - offsets.keys()
    if missing:
        raise FileNotFoundError(f"Synscapes tar is missing {len(missing)} required members")
    fd = os.open(tar_path, os.O_RDONLY)

    def member_bytes(name: str) -> bytes:
        offset, size = offsets[name]
        data = os.pread(fd, size, offset)
        if len(data) != size:
            raise OSError(f"Short tar read for {name}")
        return data

    def process(pair: SynscapesPair) -> dict[str, Any] | None:
        try:
            return write_pair(
                pair,
                out_root=out_root,
                rgb_loader=lambda name: ImageOps.exif_transpose(Image.open(io.BytesIO(member_bytes(name)))).convert("RGB"),
                depth_loader=lambda name: load_exr_depth(data=member_bytes(name), channel=args.depth_channel),
                args=args,
            )
        except Exception as exc:
            if args.strict:
                raise
            print(f"WARNING: skipped {pair.stem}: {exc}", flush=True)
            return None

    try:
        workers = max(1, args.num_workers)
        if workers == 1:
            rows = [process(pair) for pair in tqdm(pairs, desc="Preparing Synscapes depth from tar")]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                rows = list(tqdm(executor.map(process, pairs), total=len(pairs), desc=f"Preparing Synscapes depth from tar ({workers} workers)"))
    finally:
        os.close(fd)
    return [row for row in rows if row is not None]


def write_metadata(out_root: Path, rows: list[dict[str, Any]]) -> None:
    with (out_root / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_prepared_tar(out_root: Path, tar_path: Path) -> None:
    tar_path = tar_path.expanduser().resolve()
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = tar_path.with_suffix(tar_path.suffix + ".partial")
    if tmp_path.exists():
        tmp_path.unlink()
    with tarfile.open(tmp_path, "w") as archive:
        archive.add(out_root, arcname=out_root.name)
    tmp_path.replace(tar_path)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Synscapes RGB-to-depth training pairs.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synscapes-root", type=Path, help="Extracted Synscapes root or parent directory.")
    source.add_argument("--synscapes-tar", type=Path, help="Synscapes tar containing Synscapes/img/{rgb,depth}.")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--prepared-tar", type=Path, default=None, help="Optional tar of the prepared dataset folder.")
    parser.add_argument("--rgb-subdir", default="rgb", choices=["rgb", "rgb-2k"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=1, help="Used for extracted-folder input.")
    parser.add_argument("--max-side", type=int, default=0, help="Resize larger RGB/depth targets to this side; <=0 preserves.")
    parser.add_argument(
        "--depth-encoding",
        "--depth_encoding",
        choices=DEPTH_ENCODINGS,
        default=DEPTH_ENCODING_VISION_BANANA_RGB,
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--normalization", choices=["inverse", "log", "linear"], default="inverse")
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=250.0)
    parser.add_argument("--low-percentile", type=float, default=1.0)
    parser.add_argument("--high-percentile", type=float, default=99.0)
    parser.add_argument("--invalid-value", type=int, default=0)
    parser.add_argument(
        "--invalid-rgb",
        type=lambda value: tuple(int(part) for part in value.split(",")),
        default=(0, 0, 0),
        help="RGB color for invalid pixels in Vision Banana targets, formatted as R,G,B.",
    )
    parser.add_argument("--depth-channel", choices=["first", "last", "mean", "red", "green", "blue"], default="first")
    parser.add_argument("--strict", action="store_true", help="Fail on the first corrupt/unreadable sample.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.low_percentile < 0 or args.high_percentile > 100 or args.low_percentile >= args.high_percentile:
        raise ValueError("--low-percentile must be smaller than --high-percentile within [0, 100].")
    if len(args.invalid_rgb) != 3 or any(part < 0 or part > 255 for part in args.invalid_rgb):
        raise ValueError("--invalid-rgb must contain three comma-separated uint8 values.")
    if args.prompt is None:
        args.prompt = (
            GRAYSCALE_DEPTH_PROMPT
            if args.depth_encoding == DEPTH_ENCODING_GRAYSCALE
            else VISION_BANANA_DEPTH_PROMPT
        )

    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "conditioning_images").mkdir(parents=True, exist_ok=True)
    (out_root / "target_images").mkdir(parents=True, exist_ok=True)
    (out_root / "depth_raw").mkdir(parents=True, exist_ok=True)
    (out_root / "valid_masks").mkdir(parents=True, exist_ok=True)

    if args.synscapes_root is not None:
        pairs = list_root_pairs(args.synscapes_root, args.rgb_subdir)
        pairs = sample_pairs(pairs, args.limit, args.seed)
        rows = prepare_from_root(pairs, out_root, args)
    else:
        pairs = list_tar_pairs(args.synscapes_tar, args.rgb_subdir)
        pairs = sample_pairs(pairs, args.limit, args.seed)
        rows = prepare_from_tar(args.synscapes_tar.expanduser().resolve(), pairs, out_root, args)

    if not rows:
        raise RuntimeError("No Synscapes depth rows were written.")
    write_metadata(out_root, rows)
    if args.prepared_tar is not None:
        write_prepared_tar(out_root, args.prepared_tar)

    print("Synscapes depth preparation complete.")
    print(f"  pairs_found: {len(pairs):,}")
    print(f"  rows_written: {len(rows):,}")
    print(f"  depth_encoding: {args.depth_encoding}")
    print(f"  out_root: {out_root}")
    if args.prepared_tar is not None:
        print(f"  prepared_tar: {args.prepared_tar.expanduser().resolve()}")


if __name__ == "__main__":
    main()
