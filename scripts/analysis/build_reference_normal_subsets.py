#!/usr/bin/env python3
"""Build low-inode exact-reference surface-normal subset TARs for evaluation."""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import tarfile
import tempfile
import zipfile
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

CANONICAL_NORMAL_CONVENTION = "camera_x_right_y_up_z_toward_camera"


def take_limit(rows: list, limit: int) -> list:
    """Return all rows when limit <= 0, otherwise a deterministic prefix."""
    return rows if limit <= 0 else rows[:limit]


def write_info(root: Path, **values) -> None:
    encoding = str(values.get("normal_encoding", ""))
    if encoding.startswith("float_xyz"):
        values.setdefault("stored_normal_convention", CANONICAL_NORMAL_CONVENTION)
        values.setdefault("gt_to_canonical_permutation", [0, 1, 2])
        values.setdefault("gt_to_canonical_signs", [1, 1, 1])
    (root / "dataset_info.json").write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def archive(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    with tarfile.open(partial, "w") as handle:
        handle.add(root, arcname=root.name)
    with tarfile.open(partial) as handle:
        handle.getmembers()
    partial.replace(output)


def decode_rgb16_png(data: bytes) -> np.ndarray:
    """Decode the non-interlaced 16-bit RGB PNGs used by DSINE VKITTI."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Invalid PNG signature")
    cursor = 8
    compressed = bytearray()
    width = height = None
    while cursor < len(data):
        size = struct.unpack_from(">I", data, cursor)[0]
        kind = data[cursor + 4 : cursor + 8]
        payload = data[cursor + 8 : cursor + 8 + size]
        cursor += 12 + size
        if kind == b"IHDR":
            width, height, bit_depth, color_type, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", payload)
            )
            if (bit_depth, color_type, compression, filtering, interlace) != (16, 2, 0, 0, 0):
                raise ValueError(
                    "Expected a non-interlaced 16-bit RGB PNG, got "
                    f"depth={bit_depth} color={color_type} interlace={interlace}"
                )
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break
    if width is None or height is None:
        raise ValueError("PNG is missing IHDR")
    raw = zlib.decompress(bytes(compressed))
    stride = width * 6
    expected = height * (stride + 1)
    if len(raw) != expected:
        raise ValueError(f"Unexpected PNG payload size: {len(raw)} != {expected}")
    rows = np.empty((height, stride), dtype=np.uint8)
    source = memoryview(raw)
    for y in range(height):
        filter_type = source[y * (stride + 1)]
        scan = np.frombuffer(source[y * (stride + 1) + 1 : (y + 1) * (stride + 1)], dtype=np.uint8)
        recon = rows[y]
        previous = rows[y - 1] if y else np.zeros(stride, dtype=np.uint8)
        if filter_type == 0:
            recon[:] = scan
        elif filter_type == 1:
            for x in range(stride):
                recon[x] = (int(scan[x]) + (int(recon[x - 6]) if x >= 6 else 0)) & 255
        elif filter_type == 2:
            recon[:] = (scan.astype(np.uint16) + previous.astype(np.uint16)).astype(np.uint8)
        elif filter_type in {3, 4}:
            for x in range(stride):
                left = int(recon[x - 6]) if x >= 6 else 0
                up = int(previous[x])
                upper_left = int(previous[x - 6]) if x >= 6 else 0
                if filter_type == 3:
                    predictor = (left + up) // 2
                else:
                    p = left + up - upper_left
                    distances = (abs(p - left), abs(p - up), abs(p - upper_left))
                    predictor = (left, up, upper_left)[distances.index(min(distances))]
                recon[x] = (int(scan[x]) + predictor) & 255
        else:
            raise ValueError(f"Unsupported PNG filter {filter_type}")
    return rows.reshape(height, width, 6).view(">u2").reshape(height, width, 3)


def copy_marigold(
    source: Path, pair_list: Path, dataset: str, limit: int, destination: Path
) -> Path:
    if dataset == "nyu":
        root = destination / "nyu_surface_normals"
        manifest = root / "nyu_test.txt"
        source_base = source / "nyuv2" / "test"
    else:
        root = destination / "scannet_depth_normals"
        manifest = root / "scannet_val_standard.txt"
        source_base = source / "scannet"
    rows = [line.split() for line in pair_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = take_limit(rows, limit)
    output_rows = []
    for rgb_rel, normal_rel, *_ in rows:
        rgb_source = source_base / rgb_rel
        normal_source = source_base / normal_rel
        rgb_out = Path("images") / rgb_rel
        normal_out = Path("normals") / normal_rel
        (root / rgb_out).parent.mkdir(parents=True, exist_ok=True)
        (root / normal_out).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rgb_source, root / rgb_out)
        copy_dsine_normal_as_marigold(normal_source, root / normal_out)
        output_rows.append(f"{rgb_out.as_posix()} {normal_out.as_posix()}\n")
    manifest.write_text("".join(output_rows), encoding="utf-8")
    write_info(
        root,
        dataset=f"{dataset}_surface_normals",
        count=len(rows),
        ground_truth_source="Marigold official normals evaluation archive",
        validity_mask="implicit finite nonzero float normal",
        normal_encoding="float_xyz_npy",
        subset="first entries in the official Marigold file list",
    )
    return root


def copy_sintel(source: Path, pair_list: Path, limit: int, destination: Path) -> Path:
    root = destination / "sintel_surface_normals"
    source_base = source / "sintel"
    rows = take_limit([
        line.split()
        for line in pair_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ], limit)
    output_rows = []
    for rgb_rel, normal_rel, *_ in rows:
        rgb_source = source_base / rgb_rel
        normal_source = source_base / normal_rel
        rgb_out = Path("images") / rgb_rel
        normal_out = Path("normals") / normal_rel
        (root / rgb_out).parent.mkdir(parents=True, exist_ok=True)
        (root / normal_out).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rgb_source, root / rgb_out)
        copy_dsine_normal_as_marigold(normal_source, root / normal_out)
        output_rows.append(f"{rgb_out.as_posix()} {normal_out.as_posix()}\n")
    (root / "sintel_test.txt").write_text("".join(output_rows), encoding="utf-8")
    write_info(
        root,
        dataset="sintel_surface_normals",
        count=len(rows),
        ground_truth_source="Marigold official Sintel normals evaluation archive",
        validity_mask="implicit finite nonzero float normal",
        normal_generation="DSINE discontinuity-aware plane fitting",
        normal_encoding="float_xyz_npy",
        subset="first entries in Marigold's frozen filtered Sintel split",
    )
    return root


def copy_dsine_normal_as_marigold(requested_npy: Path, output_npy: Path) -> None:
    """Reproduce Marigold's NPY targets exactly from DSINE PNG/EXR assets."""
    if requested_npy.is_file():
        shutil.copy2(requested_npy, output_npy)
        return

    png = requested_npy.with_suffix(".png")
    exr = requested_npy.with_suffix(".exr")
    if png.is_file():
        encoded = np.asarray(Image.open(png).convert("RGBA"), dtype=np.uint8)[..., :3]
        valid = np.sum(encoded, axis=-1) > 0
        # This is the exact DSINE -> Marigold conversion used by the official
        # evaluation package: decode RGB, reflect camera x, normalize, zero invalid.
        normals = encoded.astype(np.float64) / 255.0 * 2.0 - 1.0
        normals[..., 0] *= -1.0
        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = np.divide(normals, norm, out=np.zeros_like(normals), where=norm > 1e-8)
        normals[~valid] = 0.0
        np.save(output_npy, normals)
        return
    if exr.is_file():
        import Imath
        import OpenEXR

        handle = OpenEXR.InputFile(str(exr))
        window = handle.header()["dataWindow"]
        width = window.max.x - window.min.x + 1
        height = window.max.y - window.min.y + 1
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        normals = np.stack(
            [
                np.frombuffer(handle.channel(channel, pixel_type), dtype=np.float32).reshape(height, width)
                for channel in "RGB"
            ],
            axis=-1,
        )
        normals[..., 0] *= -1.0
        np.save(output_npy, normals)
        return
    raise FileNotFoundError(
        f"No Marigold NPY or corresponding DSINE PNG/EXR for {requested_npy}"
    )


def copy_ibims(source: Path, limit: int, destination: Path) -> Path:
    """Package Marigold/DSINE's flat iBims reference directory."""
    source_root = source / "ibims" if (source / "ibims").is_dir() else source
    root = destination / "ibims"
    image_paths = take_limit(sorted(source_root.glob("*_img.png")), limit)
    if not image_paths:
        raise FileNotFoundError(f"No *_img.png files found below {source_root}")
    for image_path in image_paths:
        normal_path = image_path.with_name(
            image_path.name.replace("_img.png", "_normal.npy")
        )
        if not normal_path.is_file():
            raise FileNotFoundError(normal_path)
        root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, root / image_path.name)
        shutil.copy2(normal_path, root / normal_path.name)
    write_info(
        root,
        dataset="ibims1_surface_normals",
        count=len(image_paths),
        ground_truth_source="Marigold official normals evaluation archive; DSINE iBims targets",
        ground_truth_kind="depth-derived discontinuity-aware plane-fit normals",
        validity_mask="implicit finite nonzero float normal",
        normal_encoding="float_xyz_npy",
        subset=("full official set" if limit <= 0 else "deterministic filename prefix"),
    )
    return root


def build_vkitti(source_zip: Path, split: Path, limit: int, destination: Path) -> Path:
    root = destination / "vkitti2_dsine_surface_normals"
    rows = take_limit(
        [line.strip() for line in split.read_text(encoding="utf-8").splitlines() if line.strip()],
        limit,
    )
    output_rows = []
    with zipfile.ZipFile(source_zip) as source:
        for rgb_rel in rows:
            normal_rel = rgb_rel.replace("_img.jpg", "_normal.png")
            rgb_out = Path("images") / rgb_rel
            normal_out = Path("normals") / normal_rel.replace(".png", ".npy")
            (root / rgb_out).parent.mkdir(parents=True, exist_ok=True)
            (root / normal_out).parent.mkdir(parents=True, exist_ok=True)
            (root / rgb_out).write_bytes(source.read(f"dsine_eval/vkitti/{rgb_rel}"))
            encoded = decode_rgb16_png(source.read(f"dsine_eval/vkitti/{normal_rel}"))
            # Decode, make invalid vectors zero, and reflect x from DSINE's
            # convention into the Marigold/OVB convention.
            valid = np.sum(encoded, axis=2) > 0
            normals = encoded.astype(np.float32) / 65535.0 * 2.0 - 1.0
            normals[..., 0] *= -1.0
            normals[~valid] = 0.0
            norm = np.linalg.norm(normals, axis=2, keepdims=True)
            normals = np.divide(normals, norm, out=np.zeros_like(normals), where=norm > 1e-8)
            np.save(root / normal_out, normals)
            output_rows.append(f"{rgb_out.as_posix()} {normal_out.as_posix()}\n")
    (root / "vkitti2_test.txt").write_text("".join(output_rows), encoding="utf-8")
    write_info(
        root,
        dataset="vkitti2_dsine_surface_normals",
        count=len(rows),
        ground_truth_source="DSINE official evaluation bundle",
        validity_mask="implicit nonzero uint16 normal PNG",
        normal_generation="discontinuity-aware plane fitting",
        normal_encoding="float_xyz_npy after DSINE-to-Marigold x reflection",
        subset="first entries in DSINE's frozen sampled split",
    )
    return root


def selected_tar_members(path: Path, wanted: set[str]) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    with tarfile.open(path, "r:gz") as source:
        for member in source:
            name = member.name.lstrip("./")
            if name not in wanted or not member.isfile():
                continue
            handle = source.extractfile(member)
            if handle is None:
                raise RuntimeError(f"Could not read {member.name} from {path}")
            found[name] = handle.read()
            if len(found) == len(wanted):
                break
    missing = sorted(wanted - found.keys())
    if missing:
        raise FileNotFoundError(f"Missing members in {path}: {missing[:10]}")
    return found


def build_diode(source: Path, pair_list: Path, limit: int, destination: Path) -> Path:
    root = destination / "diode_surface_normals"
    with pair_list.open(newline="", encoding="utf-8") as handle:
        rows = take_limit(list(csv.reader(handle)), limit)
    image_names = [row[0].strip().lstrip("./") for row in rows]
    normal_names = [
        row[3].strip().lstrip("./").replace("val_normal/", "val/", 1)
        for row in rows
    ]
    images = selected_tar_members(source / "val.tar.gz", set(image_names))
    normals = selected_tar_members(source / "val_normals.tar.gz", set(normal_names))
    output_rows = []
    for image_name, normal_name in zip(image_names, normal_names):
        relative = Path(*Path(image_name).parts[1:])
        image_out = Path("images") / relative
        normal_out = Path("normals") / relative.with_name(
            relative.stem + "_normal.npy"
        )
        (root / image_out).parent.mkdir(parents=True, exist_ok=True)
        (root / normal_out).parent.mkdir(parents=True, exist_ok=True)
        (root / image_out).write_bytes(images[image_name])
        (root / normal_out).write_bytes(normals[normal_name])
        output_rows.append(f"{image_out.as_posix()} {normal_out.as_posix()}\n")
    (root / "diode_indoor.txt").write_text("".join(output_rows), encoding="utf-8")
    (root / "diode_outdoor.txt").write_text("", encoding="utf-8")
    write_info(
        root,
        dataset="diode_surface_normals",
        count=len(rows),
        ground_truth_source="official DIODE validation normal NPY archive",
        validity_mask="implicit finite nonzero float normal",
        normal_encoding="float_xyz_npy",
        subset="first entries in official val_indoors.csv",
    )
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("nyu", "ibims1", "scannet", "sintel", "vkitti2", "diode"), required=True
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pair-list", type=Path)
    parser.add_argument(
        "--limit", type=int, default=32,
        help="Deterministic prefix size; use 0 to build the full reference asset.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix=f"{args.dataset}-reference-subset-") as tmp:
        destination = Path(tmp)
        if args.dataset in {"nyu", "scannet"}:
            if args.pair_list is None:
                parser.error("--pair-list is required for Marigold subsets")
            root = copy_marigold(
                args.source, args.pair_list, args.dataset, args.limit, destination
            )
        elif args.dataset == "ibims1":
            root = copy_ibims(args.source, args.limit, destination)
        elif args.dataset == "sintel":
            if args.pair_list is None:
                parser.error("--pair-list is required for the Marigold Sintel subset")
            root = copy_sintel(
                args.source, args.pair_list, args.limit, destination
            )
        elif args.dataset == "vkitti2":
            if args.pair_list is None:
                parser.error("--pair-list must point to DSINE's vkitti split")
            root = build_vkitti(
                args.source, args.pair_list, args.limit, destination
            )
        else:
            if args.pair_list is None:
                parser.error("--pair-list must point to DIODE's val_indoors.csv")
            root = build_diode(
                args.source, args.pair_list, args.limit, destination
            )
        archive(root, args.output)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
