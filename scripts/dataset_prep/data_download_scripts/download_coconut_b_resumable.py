#!/usr/bin/env python3
"""Resume-safe materialization of the official COCONut-B Hugging Face dataset."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Any


def output_json_key(target: ast.expr) -> str | None:
    """Return the output_json['key'] assignment key, if any."""
    if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
        return None
    if target.value.id != "output_json":
        return None
    key = target.slice
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return key.value
    return None


def metadata_from_official_downloader(path: Path) -> dict[str, Any]:
    """Reuse the exact COCO category, license, and info literals published upstream."""
    values: dict[str, Any] = {}
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            key = output_json_key(target)
            if key in {"categories", "licenses", "info"}:
                try:
                    values[key] = ast.literal_eval(node.value)
                except ValueError as exc:
                    raise RuntimeError(f"Could not parse output_json[{key!r}] from {path}") from exc
    missing = {"categories", "licenses", "info"} - values.keys()
    if missing:
        raise RuntimeError(f"Official downloader is missing metadata keys: {sorted(missing)}")
    return values


def atomic_dump(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle)
    os.replace(temporary, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize COCONut-B with restart support. Existing mask count is "
            "treated as a contiguous completed prefix from an earlier invocation."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--official-downloader",
        type=Path,
        required=True,
        help="The official download_coconut.py used to recover COCO metadata.",
    )
    parser.add_argument("--dataset", default="xdeng77/coconut_b")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError as exc:
        raise SystemExit("Install the required dependencies: pip install datasets tqdm") from exc

    output_dir = args.output_dir
    mask_dir = output_dir / "coconut_b"
    output_json = output_dir / "coconut_b.json"
    mask_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split="train")
    total = len(dataset)
    existing = sum(1 for _ in mask_dir.glob("*.png"))
    if existing > total:
        raise RuntimeError(
            f"Found {existing} masks in {mask_dir}, but {args.dataset} has only {total} samples."
        )

    if output_json.is_file() and existing == total:
        print(f"COCONut-B is already complete: {output_json}")
        return

    if existing:
        print(f"Resuming COCONut-B at sample {existing}/{total}; existing masks are preserved.")

    for index in tqdm(range(existing, total), initial=existing, total=total, desc="Saving COCONut-B masks"):
        item = dataset[index]
        filename = item["segments_info"]["file_name"]
        mask_path = mask_dir / filename
        if mask_path.exists():
            raise RuntimeError(
                f"Refusing to overwrite an unexpected existing mask at resume index {index}: {mask_path}"
            )
        item["mask"].save(mask_path)

    metadata = metadata_from_official_downloader(args.official_downloader)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for item in tqdm(dataset, total=total, desc="Writing COCONut-B manifest"):
        images.append(item["image_info"])
        annotations.append(item["segments_info"])

    atomic_dump(
        {
            "images": images,
            "annotations": annotations,
            "categories": metadata["categories"],
            "licenses": metadata["licenses"],
            "info": metadata["info"],
        },
        output_json,
    )
    print(f"Wrote {output_json} with {total} images and annotations.")


if __name__ == "__main__":
    main()
