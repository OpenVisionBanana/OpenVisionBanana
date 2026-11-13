#!/usr/bin/env python3
"""Build a COCO panoptic manifest for a complete COCONut-XL installation.

The COCONut-L release contains a conventional COCO JSON, while the XL release
contains one ``segments_info`` JSON per Objects365 image.  This tool turns both
layouts into the single ``panoptic_train2017.json`` expected by the COCONut
training setup.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
from pathlib import Path


OBJECT365_NAME = re.compile(r"(objects365_v2_\d+)")


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def object365_name(value: object) -> str | None:
    match = OBJECT365_NAME.search(str(value or ""))
    return match.group(1) if match else None


def png_size(path: Path) -> tuple[int, int]:
    """Return a PNG's width and height without decoding the mask."""
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def require_list(value: object, label: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"Expected a list for {label}")
    return value


def load_large(manifest_path: Path, mask_dir: Path) -> list[dict]:
    data = read_json(manifest_path)
    images = require_list(data.get("images"), f"{manifest_path}: images")
    annotations = require_list(data.get("annotations"), f"{manifest_path}: annotations")
    image_by_id = {image.get("id"): image for image in images if "id" in image}
    records = []

    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError(f"Invalid COCONut-L annotation in {manifest_path}")
        image = image_by_id.get(annotation.get("image_id"))
        name = (
            object365_name(annotation.get("object365_file_name"))
            or object365_name(annotation.get("file_name"))
            or object365_name(image.get("object365_file_name") if image else None)
            or object365_name(image.get("object365_name") if image else None)
        )
        if name is None:
            raise ValueError(f"Cannot resolve Objects365 name for L image {annotation.get('image_id')}")
        mask_path = mask_dir / f"{name}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"COCONut-L mask is missing: {mask_path}")
        width, height = png_size(mask_path)
        if image is not None:
            width = image.get("width", width)
            height = image.get("height", height)
        records.append(
            {
                "name": name,
                "width": width,
                "height": height,
                "segments_info": require_list(annotation.get("segments_info"), f"L {name}: segments_info"),
                "image": image,
                "annotation": annotation,
            }
        )
    return records


def load_xlarge(root: Path) -> list[dict]:
    release_root = root / "coconuts_xlarge"
    info_dir = release_root / "panseg_info"
    mask_dir = release_root / "panseg"
    if not info_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected panseg and panseg_info under {release_root}")

    records = []
    info_paths = sorted(info_dir.glob("*.json"))
    for index, info_path in enumerate(info_paths, start=1):
        name = info_path.stem
        mask_path = mask_dir / f"{name}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"COCONut-XL mask is missing: {mask_path}")
        width, height = png_size(mask_path)
        records.append(
            {
                "name": name,
                "width": width,
                "height": height,
                "segments_info": require_list(read_json(info_path), f"XL {info_path}: segments_info"),
                "image": None,
                "annotation": None,
            }
        )
        if index % 20_000 == 0:
            print(f"Read COCONut-XL metadata: {index}/{len(info_paths)}", flush=True)
    return records


def validate_segments(segments: list, category_ids: set[int], label: str) -> None:
    for segment in segments:
        if not isinstance(segment, dict) or "id" not in segment or "category_id" not in segment:
            raise ValueError(f"Invalid segment in {label}: {segment!r}")
        if segment["category_id"] not in category_ids:
            raise ValueError(f"Unknown category {segment['category_id']} in {label}")


def append_record(output_images: list, output_annotations: list, record: dict, image_id: int) -> None:
    name = record["name"]
    source_image = record["image"]
    if source_image is None:
        image = {"id": image_id, "file_name": f"{name}.jpg", "width": record["width"], "height": record["height"]}
    else:
        image = dict(source_image)
        image.update({"id": image_id, "file_name": f"{name}.jpg", "width": record["width"], "height": record["height"]})
    image["object365_name"] = name

    source_annotation = record["annotation"]
    annotation = {} if source_annotation is None else dict(source_annotation)
    annotation.update({"image_id": image_id, "file_name": f"{name}.png", "segments_info": record["segments_info"]})
    annotation["object365_name"] = name
    output_images.append(image)
    output_annotations.append(annotation)


def merge(base_path: Path, large_path: Path, large_masks: Path, xlarge_root: Path, output_path: Path) -> None:
    base = read_json(base_path)
    base_images = require_list(base.get("images"), f"{base_path}: images")
    base_annotations = require_list(base.get("annotations"), f"{base_path}: annotations")
    categories = require_list(base.get("categories"), f"{base_path}: categories")
    category_ids = {category["id"] for category in categories if isinstance(category, dict) and "id" in category}
    if not category_ids:
        raise ValueError(f"No valid categories in {base_path}")

    large_records = load_large(large_path, large_masks)
    xlarge_records = load_xlarge(xlarge_root)
    images = list(base_images)
    annotations = list(base_annotations)
    used_ids = {image["id"] for image in images if isinstance(image, dict) and "id" in image}
    used_names = {image.get("file_name") for image in images if isinstance(image, dict)}
    used_objects: set[str] = set()
    next_id = max((image_id for image_id in used_ids if isinstance(image_id, int)), default=-1) + 1

    def add(record: dict) -> bool:
        nonlocal next_id
        name = record["name"]
        if name in used_objects:
            return False
        image_name, mask_name = f"{name}.jpg", f"{name}.png"
        if image_name in used_names:
            raise ValueError(f"Image filename collision: {image_name}")
        validate_segments(record["segments_info"], category_ids, name)
        while next_id in used_ids:
            next_id += 1
        append_record(images, annotations, record, next_id)
        used_ids.add(next_id)
        used_names.add(image_name)
        used_objects.add(name)
        next_id += 1
        return True

    # Existing symlinks use XL masks when a release overlaps with L, so use the
    # matching XL metadata in those 189 duplicate cases.
    xl_added = sum(add(record) for record in xlarge_records)
    large_added = sum(add(record) for record in large_records)
    large_duplicates = len(large_records) - large_added

    merged = dict(base)
    merged["images"] = images
    merged["annotations"] = annotations
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle)
    os.replace(temporary, output_path)
    print(
        f"Wrote {output_path}: {len(base_images)} B + {large_added} L + {xl_added} XL images "
        f"({large_duplicates} overlapping L masks use XL metadata)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--large-manifest", type=Path, required=True)
    parser.add_argument("--large-masks", type=Path, required=True)
    parser.add_argument("--xlarge-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merge(args.base, args.large_manifest, args.large_masks, args.xlarge_root, args.output)


if __name__ == "__main__":
    main()
