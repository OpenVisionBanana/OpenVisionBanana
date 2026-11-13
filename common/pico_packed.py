"""Resumable Apple Pico-Banana acquisition and bounded-memory TAR preparation.

A completed shard is the resume unit. Interrupted shards are rebuilt, never
accepted as complete. Neither path extracts the corpus into loose files.
"""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import tarfile

from common.release_data import download_with_resume, sha256
from common.dataset_source_policy import validate_dataset_source

METADATA_URL = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/jsonl/sft.jsonl"
IMAGE_BASE = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/"
SHARD_ROWS = 256


def add(archive, name, data):
    member = tarfile.TarInfo(name)
    member.size, member.mode = len(data), 0o644
    archive.addfile(member, io.BytesIO(data))


def atomic_json(path, value):
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    partial.replace(path)


def acquire(directory, settings, scratch, fetch=download_with_resume):
    directory, scratch = Path(directory), Path(scratch)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "shards").mkdir(exist_ok=True)
    metadata = directory / "sft.jsonl"
    if not metadata.exists():
        fetch(METADATA_URL, metadata)
    rows = [json.loads(line) for line in metadata.read_text().splitlines() if line.strip()]
    limit = int(settings["limit"]) if "limit" in settings else None
    if limit is not None and (limit < 1 or limit > len(rows)):
        raise ValueError("Pico limit must be positive and no larger than the author metadata")
    rows = rows[:limit]
    identity = dict(schema_version=1, metadata_url=METADATA_URL,
                    metadata_sha256=sha256(metadata), requested_limit=limit,
                    total_rows=len(rows), shard_rows=SHARD_ROWS)
    plan = directory / "packing_plan.json"
    if plan.exists() and json.loads(plan.read_text()) != identity:
        raise ValueError("Pico acquisition settings/metadata changed; use a separate raw directory")
    atomic_json(plan, identity)
    shards = []
    for start in range(0, len(rows), SHARD_ROWS):
        path = directory / "shards" / f"{start // SHARD_ROWS:06d}.tar"
        receipt = path.with_suffix(".json")
        count = min(SHARD_ROWS, len(rows) - start)
        if path.exists() and receipt.exists():
            info = json.loads(receipt.read_text())
            if info.get("start") != start or info.get("rows") != count or info.get("sha256") != sha256(path):
                raise ValueError(f"Pico shard integrity failure: {path}")
            shards.append(info)
            continue
        partial = path.with_suffix(".tar.partial")
        with tarfile.open(partial, "w") as packed:
            for index in range(start, start + count):
                record = rows[index]
                target = PurePosixPath(record["output_image"])
                if target.is_absolute() or ".." in target.parts:
                    raise ValueError("Unsafe Pico target path")
                urls = {"source": record["open_image_input_url"], "target": IMAGE_BASE + str(target)}
                hashes = {}
                for kind, url in urls.items():
                    validate_dataset_source(url)
                    if not url.startswith("https://"):
                        raise ValueError("Pico image source must use HTTPS")
                    temporary = scratch / f"pico_{kind}"
                    # A download belongs to this URL only; do not resume a
                    # previous row's partial image under the same temporary name.
                    temporary.unlink(missing_ok=True)
                    Path(str(temporary) + ".partial").unlink(missing_ok=True)
                    fetch(url, temporary)
                    data = temporary.read_bytes()
                    if not data:
                        raise ValueError(f"Empty Pico {kind} at row {index}")
                    hashes[kind] = hashlib.sha256(data).hexdigest()
                    add(packed, f"{index:08d}/{kind}", data)
                    temporary.unlink()
                add(packed, f"{index:08d}/record.json", json.dumps(dict(
                    record=record, urls=urls, image_sha256=hashes), sort_keys=True).encode())
        partial.replace(path)
        info = dict(path=str(path.relative_to(directory)), start=start, rows=count, sha256=sha256(path))
        atomic_json(receipt, info)
        shards.append(info)
    result = dict(identity, shards=shards, smoke_only=limit is not None)
    atomic_json(directory / "packed.json", result)
    return result


def prepare(directory, output, limit=None, seed=42):
    from PIL import Image, ImageOps
    from dataset_prep.pico_banana.prepare_pico_banana import safe_stem, add_task_prefix
    directory, output = Path(directory), Path(output)
    manifest = json.loads((directory / "packed.json").read_text())
    expected = 0
    for shard in manifest["shards"]:
        if shard["start"] != expected or sha256(directory / shard["path"]) != shard["sha256"]:
            raise ValueError("Incomplete or corrupt Pico packed source")
        expected += shard["rows"]
    if expected != manifest["total_rows"] or expected < 1:
        raise ValueError("Incomplete Pico packed source")
    if limit is not None and (limit < 1 or limit > expected):
        raise ValueError("Pico preparation limit exceeds the acquired rows")
    indices = list(range(expected))
    random.Random(seed).shuffle(indices)
    selected = {original: position for position, original in enumerate(indices[:limit])}
    rows = []
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    with tarfile.open(partial, "w") as target:
        for shard in manifest["shards"]:
            with tarfile.open(directory / shard["path"], "r:") as source:
                for original in range(shard["start"], shard["start"] + shard["rows"]):
                    if original not in selected:
                        continue
                    prefix = f"{original:08d}"
                    record = json.load(source.extractfile(prefix + "/record.json"))
                    meta = record["record"]
                    instruction = meta.get("text") or meta.get("summarized_text")
                    if not instruction:
                        raise ValueError(f"Missing Pico instruction: {original}")
                    stem = f"pico_{selected[original]:08d}_{safe_stem(str(meta.get('id') or Path(meta['output_image']).stem))}"
                    paths = {}
                    for kind, field in (("source", "conditioning_image"), ("target", "image")):
                        data = source.extractfile(prefix + "/" + kind).read()
                        if hashlib.sha256(data).hexdigest() != record["image_sha256"][kind]:
                            raise ValueError(f"Corrupt Pico {kind}: {original}")
                        with Image.open(io.BytesIO(data)) as image:
                            image = ImageOps.exif_transpose(image).convert("RGB")
                            if max(image.size) > 1024:
                                scale = 1024 / max(image.size)
                                image = image.resize(tuple(max(1, round(s * scale)) for s in image.size), Image.Resampling.LANCZOS)
                            buffer = io.BytesIO()
                            image.save(buffer, format="PNG")
                        paths[field] = f"images/{stem}_{kind}.png"
                        add(target, paths[field], buffer.getvalue())
                    rows.append(dict(paths, id=stem, task="image_editing", source_dataset="pico_banana_400k",
                        text=add_task_prefix(str(instruction), "[TASK: IMAGE_EDITING]", False),
                        metadata=dict(edit_type=meta.get("edit_type"), source_metadata=meta)))
        rows.sort(key=lambda row: row["id"])
        add(target, "metadata.jsonl", ("\n".join(json.dumps(row) for row in rows) + "\n").encode())
    partial.replace(output)
    return len(rows)
