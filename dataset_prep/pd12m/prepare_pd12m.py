#!/usr/bin/env python3
"""Create reproducible local WebDataset shards from Spawning/PD12M metadata."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import io
import json
import os
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Iterable, Iterator

import requests
from PIL import Image, ImageOps
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.dataset_source_policy import validate_dataset_source


DEFAULT_DATASET = "Spawning/PD12M"
DEFAULT_MAX_MEMBER_BYTES = 64 * 1024 * 1024
SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff", "image/bmp"}
REQUIRED_FIELDS = ("id", "url", "caption", "width", "height")


def normalize_caption(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def stable_selection_score(sample_id: str, seed: int) -> int:
    payload = f"{seed}\0{sample_id}".encode("utf-8", errors="surrogatepass")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "big")


def row_is_eligible(
    row: dict[str, Any],
    *,
    min_caption_chars: int = 10,
    min_side: int = 512,
    max_aspect_ratio: float = 2.5,
    supported_mime_types: set[str] | None = None,
) -> bool:
    if any(row.get(field) in (None, "") for field in REQUIRED_FIELDS):
        return False
    caption = normalize_caption(row.get("caption"))
    if len(caption) < min_caption_chars:
        return False
    try:
        width, height = int(row["width"]), int(row["height"])
    except (TypeError, ValueError):
        return False
    if width <= 0 or height <= 0 or min(width, height) < min_side:
        return False
    if max(width / height, height / width) > max_aspect_ratio:
        return False
    allowed = supported_mime_types or SUPPORTED_MIME_TYPES
    return str(row.get("mime_type", "")).lower() in allowed


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield row


def resolve_hub_revision(dataset_name: str, requested_revision: str | None) -> str:
    validate_dataset_source(dataset_name)
    if requested_revision:
        return requested_revision
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - dependency error
        raise RuntimeError("huggingface_hub is required to resolve the PD12M revision") from exc
    info = HfApi().dataset_info(dataset_name)
    if not info.sha:
        raise RuntimeError(f"The Hub did not return a commit SHA for {dataset_name}")
    return info.sha


def iter_hub_metadata(dataset_name: str, revision: str) -> Iterable[dict[str, Any]]:
    validate_dataset_source(dataset_name)
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency error
        raise RuntimeError("datasets is required to stream PD12M metadata") from exc
    return load_dataset(dataset_name, split="train", streaming=True, revision=revision)


def select_rows(
    rows: Iterable[dict[str, Any]],
    *,
    count: int,
    seed: int,
    min_caption_chars: int,
    min_side: int,
    max_aspect_ratio: float,
    scan_limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select the globally lowest stable hashes using O(count) memory."""
    if count <= 0:
        raise ValueError("count must be positive")
    heap: list[tuple[int, str, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    stats = {"scanned": 0, "eligible": 0, "duplicate": 0}
    for row in rows:
        if scan_limit is not None and stats["scanned"] >= scan_limit:
            break
        stats["scanned"] += 1
        if not row_is_eligible(
            row,
            min_caption_chars=min_caption_chars,
            min_side=min_side,
            max_aspect_ratio=max_aspect_ratio,
        ):
            continue
        sample_id = str(row["id"])
        original_hash = str(row.get("hash") or "")
        if sample_id in seen_ids or (original_hash and original_hash in seen_hashes):
            stats["duplicate"] += 1
            continue
        seen_ids.add(sample_id)
        if original_hash:
            seen_hashes.add(original_hash)
        stats["eligible"] += 1
        score = stable_selection_score(sample_id, seed)
        normalized = {key: row.get(key) for key in (
            "id", "url", "caption", "width", "height", "mime_type", "hash", "license", "source"
        )}
        normalized["id"] = sample_id
        normalized["caption"] = normalize_caption(row["caption"])
        # Python's heap is a min-heap; negative scores keep the worst retained row on top.
        item = (-score, sample_id, normalized)
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    selected = [(-neg_score, sample_id, row) for neg_score, sample_id, row in heap]
    selected.sort(key=lambda item: (item[0], item[1]))
    result = []
    for rank, (score, _, row) in enumerate(selected):
        row["selection_rank"] = rank
        row["selection_score"] = f"{score:032x}"
        result.append(row)
    return result, stats


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


@dataclass
class DownloadResult:
    row: dict[str, Any]
    image_bytes: bytes | None
    width: int | None = None
    height: int | None = None
    error: str | None = None


def _download_bytes(
    url: str,
    *,
    connect_timeout: float,
    read_timeout: float,
    max_bytes: int,
) -> bytes:
    validate_dataset_source(url)
    with requests.get(url, stream=True, timeout=(connect_timeout, read_timeout)) as response:
        validate_dataset_source(response.url)
        response.raise_for_status()
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > max_bytes:
            raise ValueError(f"content-length {content_length} exceeds max_bytes={max_bytes}")
        output = io.BytesIO()
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            output.write(chunk)
            if output.tell() > max_bytes:
                raise ValueError(f"download exceeds max_bytes={max_bytes}")
        return output.getvalue()


def download_and_normalize(
    row: dict[str, Any],
    *,
    connect_timeout: float,
    read_timeout: float,
    max_bytes: int,
    max_member_bytes: int,
    retries: int,
    retry_backoff: float,
    jpeg_quality: int,
    verify_source_hash: bool,
) -> DownloadResult:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            source = _download_bytes(
                str(row["url"]),
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                max_bytes=max_bytes,
            )
            if verify_source_hash and row.get("hash"):
                actual_hash = hashlib.md5(source).hexdigest()  # nosec B324 - upstream compatibility check
                if actual_hash.lower() != str(row["hash"]).lower():
                    raise ValueError(f"source MD5 mismatch: expected={row['hash']} actual={actual_hash}")
            with Image.open(io.BytesIO(source)) as image:
                image.load()
                image = ImageOps.exif_transpose(image).convert("RGB")
                actual_size = image.size
                declared_size = (int(row["width"]), int(row["height"]))
                if actual_size not in {declared_size, declared_size[::-1]}:
                    raise ValueError(f"dimension mismatch: metadata={declared_size} decoded={actual_size}")
                encoded = io.BytesIO()
                image.save(encoded, format="JPEG", quality=jpeg_quality, optimize=True)
            normalized = encoded.getvalue()
            if len(normalized) > max_member_bytes:
                raise ValueError(
                    f"normalized JPEG {len(normalized)} exceeds max_member_bytes={max_member_bytes}"
                )
            return DownloadResult(row=row, image_bytes=normalized, width=actual_size[0], height=actual_size[1])
        except Exception as exc:  # network and decoder failures are row-local
            last_error = exc
            if attempt < retries:
                time.sleep(retry_backoff * (2**attempt))
    return DownloadResult(row=row, image_bytes=None, error=f"{type(last_error).__name__}: {last_error}")


def _tar_add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(payload))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_shard(path: Path, results: Iterable[DownloadResult]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".incomplete")
    keys: list[str] = []
    with tarfile.open(tmp, "w") as tar:
        for result in results:
            if result.image_bytes is None:
                continue
            row = result.row
            key = str(row["id"])
            metadata = {
                "id": key,
                "sample_id": key,
                "task": "text_to_image",
                "source_dataset": "pd12m",
                "original_width": int(row["width"]),
                "original_height": int(row["height"]),
                "decoded_width": result.width,
                "decoded_height": result.height,
                "mime_type": row.get("mime_type"),
                "license": row.get("license"),
                "source": row.get("source"),
                "original_hash": row.get("hash"),
                "url": row.get("url"),
            }
            _tar_add_bytes(tar, f"{key}.jpg", result.image_bytes)
            _tar_add_bytes(tar, f"{key}.txt", row["caption"].encode("utf-8"))
            _tar_add_bytes(tar, f"{key}.json", json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            keys.append(key)
    os.replace(tmp, path)
    return {
        "shard": path.name,
        "sample_count": len(keys),
        "byte_size": path.stat().st_size,
        "sha256": file_sha256(path),
        "first_key": keys[0] if keys else None,
        "last_key": keys[-1] if keys else None,
    }


def build_shards(args: argparse.Namespace, selected: list[dict[str, Any]]) -> None:
    shards_dir = args.out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    existing = {}
    shards_manifest = args.out_dir / "shards.jsonl"
    if shards_manifest.exists() and not args.force_shards:
        existing = {row["shard"]: row for row in iter_jsonl(shards_manifest)}
    manifest_rows: list[dict[str, Any]] = []
    failures_path = args.out_dir / "download_failures.jsonl"
    failures: list[dict[str, Any]] = (
        list(iter_jsonl(failures_path)) if failures_path.exists() and not args.force_shards else []
    )
    successful = 0
    for shard_index, start in enumerate(range(0, len(selected), args.samples_per_shard)):
        shard_name = f"pd12m-train-{shard_index:06d}.tar"
        shard_path = shards_dir / shard_name
        prior = existing.get(shard_name)
        if prior and shard_path.is_file() and shard_path.stat().st_size == int(prior["byte_size"]):
            if file_sha256(shard_path) == prior["sha256"]:
                manifest_rows.append(prior)
                successful += int(prior["sample_count"])
                continue
        chunk = selected[start : start + args.samples_per_shard]
        end = start + len(chunk)
        failures = [
            failure
            for failure in failures
            if not (start <= int(failure.get("selection_rank", -1)) < end)
        ]
        worker = lambda row: download_and_normalize(
            row,
            connect_timeout=args.connect_timeout,
            read_timeout=args.read_timeout,
            max_bytes=args.max_bytes,
            max_member_bytes=args.max_member_bytes,
            retries=args.retries,
            retry_backoff=args.retry_backoff,
            jpeg_quality=args.jpeg_quality,
            verify_source_hash=args.verify_source_hash,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            downloaded = tqdm(executor.map(worker, chunk), total=len(chunk), desc=shard_name, leave=False)

            def observed_results():
                for result in downloaded:
                    if result.error:
                        failures.append(
                            {
                                "id": result.row.get("id"),
                                "url": result.row.get("url"),
                                "selection_rank": result.row.get("selection_rank"),
                                "error": result.error,
                            }
                        )
                    yield result

            manifest = write_shard(shard_path, observed_results())
        manifest["selection_start"] = start
        manifest["selection_end"] = start + len(chunk)
        manifest_rows.append(manifest)
        successful += manifest["sample_count"]
        write_jsonl(shards_manifest, manifest_rows)
        write_jsonl(failures_path, failures)
    write_jsonl(shards_manifest, manifest_rows)
    failures.sort(key=lambda row: int(row.get("selection_rank", -1)))
    write_jsonl(failures_path, failures)
    selection_info = json.loads((args.out_dir / "selection_info.json").read_text(encoding="utf-8"))
    write_json(args.out_dir / "dataset_info.json", {
        "format": "webdataset",
        "task": "text_to_image",
        "source_dataset": "pd12m",
        "dataset_name": selection_info["dataset_name"],
        "dataset_revision": selection_info["dataset_revision"],
        "requested_samples": len(selected),
        "successful_samples": successful,
        "failed_samples": len(failures),
        "samples_per_shard": args.samples_per_shard,
        "num_shards": len(manifest_rows),
        "image_encoding": {"format": "jpeg", "quality": args.jpeg_quality},
        "max_member_bytes": args.max_member_bytes,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=("smoke", "pilot", "500k"), default=None)
    parser.add_argument("--stage", choices=("all", "select", "download"), default="all")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--metadata-jsonl", type=Path, default=None, help="Offline/test metadata source.")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scan-limit", type=int, default=None)
    parser.add_argument("--min-caption-chars", type=int, default=10)
    parser.add_argument("--min-side", type=int, default=512)
    parser.add_argument("--max-aspect-ratio", type=float, default=2.5)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--samples-per-shard", type=int, default=1000)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--read-timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-backoff", type=float, default=1.0)
    parser.add_argument("--max-bytes", type=int, default=50 * 1024 * 1024)
    parser.add_argument("--max-member-bytes", type=int, default=DEFAULT_MAX_MEMBER_BYTES)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--verify-source-hash", action="store_true")
    parser.add_argument("--force-selection", action="store_true")
    parser.add_argument("--force-shards", action="store_true")
    args = parser.parse_args(argv)
    preset_counts = {"smoke": 1000, "pilot": 100000, "500k": 500000}
    args.count = args.count if args.count is not None else preset_counts.get(args.preset or "500k")
    if args.scan_limit is None and args.preset == "smoke":
        args.scan_limit = 100000
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if args.max_bytes <= 0 or args.max_member_bytes <= 0:
        parser.error("--max-bytes and --max-member-bytes must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.out_dir / "selected_pd12m.jsonl"
    info_path = args.out_dir / "selection_info.json"
    if args.stage in {"all", "select"}:
        if selected_path.exists() and info_path.exists() and not args.force_selection:
            print(f"Selection already exists; reusing {selected_path}")
        else:
            revision = "local-jsonl" if args.metadata_jsonl else resolve_hub_revision(args.dataset_name, args.revision)
            rows = iter_jsonl(args.metadata_jsonl) if args.metadata_jsonl else iter_hub_metadata(args.dataset_name, revision)
            selected, stats = select_rows(
                tqdm(rows, desc="Scanning PD12M metadata"),
                count=args.count,
                seed=args.seed,
                min_caption_chars=args.min_caption_chars,
                min_side=args.min_side,
                max_aspect_ratio=args.max_aspect_ratio,
                scan_limit=args.scan_limit,
            )
            if len(selected) < args.count:
                raise RuntimeError(f"Only {len(selected):,} eligible unique rows found; requested {args.count:,}")
            write_jsonl(selected_path, selected)
            write_json(info_path, {
                "dataset_name": args.dataset_name,
                "dataset_revision": revision,
                "seed": args.seed,
                "requested_samples": args.count,
                "selected_samples": len(selected),
                "scan_limit": args.scan_limit,
                "filters": {
                    "min_caption_chars": args.min_caption_chars,
                    "min_side": args.min_side,
                    "max_aspect_ratio": args.max_aspect_ratio,
                    "supported_mime_types": sorted(SUPPORTED_MIME_TYPES),
                },
                "selection_algorithm": "lowest_blake2b_128(seed,NUL,id)",
                "stats": stats,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            print(f"Selected {len(selected):,} rows -> {selected_path}")
    if args.stage in {"all", "download"}:
        if not selected_path.exists() or not info_path.exists():
            raise FileNotFoundError("Selection files are missing; run --stage select first")
        build_shards(args, list(iter_jsonl(selected_path)))
        print(f"Prepared PD12M WebDataset at {args.out_dir}")


if __name__ == "__main__":
    main()
