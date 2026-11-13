#!/usr/bin/env python3
"""Prepare standalone editing or generation data with the existing converters."""
import argparse
from pathlib import Path
import subprocess
import sys
from copy import copy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.release_data import ensure_archive, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("pico_banana", "pd12m", "identity"), required=True)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int)
    p.add_argument("--staging-root", type=Path)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    from common.node_scratch import workspace
    with workspace(a.dataset) as scratch:
        prepare(a, scratch)


def prepare(a, scratch):
    if a.limit is not None and a.limit < 1:
        raise ValueError("--limit must be positive")
    from common.creative_publication import staging_root, publish
    relative = archive_relative(a.dataset)
    output = a.root / relative
    if output.exists() and not a.overwrite:
        raise FileExistsError(output)
    with staging_root(a.root) as staged:
        temporary = copy(a)
        temporary.root = staged
        temporary.overwrite = False
        build(temporary, scratch)
        publish(staged, a.root, [relative], a.overwrite)


def archive_relative(dataset):
    return Path("train") / {"pico_banana": "ImgEdit/pico_banana_sft_full.tar",
                           "identity": "ImgEdit/identity_50000.tar",
                           "pd12m": "ImgGeneration/pd12m_250000.tar"}[dataset]


def build(a, scratch):
    relative = archive_relative(a.dataset)
    output = a.root / relative
    work = scratch / a.dataset
    if output.exists() and not a.overwrite:
        raise FileExistsError(output)
    if work.exists() and not a.overwrite:
        raise FileExistsError(work)
    if a.dataset == "identity":
        prepare_identity_archive(a.raw_dir, output, a.limit if a.limit is not None else 50000, a.seed)
    elif a.dataset == "pico_banana":
        if (a.raw_dir / "packed.json").is_file():
            from common.pico_packed import prepare as prepare_packed
            prepare_packed(a.raw_dir, output, a.limit, a.seed)
            finish(output)
            return
        command = [sys.executable, str(ROOT / "dataset_prep/pico_banana/prepare_pico_banana.py"),
                   "--pico-root", str(a.raw_dir), "--metadata", str(a.raw_dir / "openimages/sft_with_local_source_image_path.jsonl"),
                   "--out-root", str(work), "--num-workers", str(a.workers), "--seed", str(a.seed), "--copy-mode", "copy"]
        if a.limit:
            command += ["--limit", str(a.limit)]
    else:
        command = [sys.executable, str(ROOT / "scripts/dataset_prep/materialize_pd12m_canonical.py"),
                   "--shards-dir", str(a.raw_dir / "shards"), "--out", str(work),
                   "--seed", str(a.seed), "--limit", str(a.limit if a.limit is not None else 250000),
                   "--require-count"]
    if a.dataset != "identity":
        subprocess.run(command, check=True)
        ensure_archive(work, output, False)
    finish(output)


def finish(output):
    Path(str(output)+".sha256").write_text(sha256(output)+"  "+output.name+"\n")
    command = [sys.executable, str(ROOT / "scripts/dataset_prep/build_prepared_tar_index.py"), str(output)]
    subprocess.run(command, check=True)


def prepare_identity_archive(source, output, limit, seed):
    """Use only original conditioning images from an indexed Pico TAR."""
    import io
    import json
    import random
    import tarfile
    from common.indexed_tar import load_index, read_reference_bytes
    from common.pico_packed import add
    from dataset_prep.identity.prepare_identity import safe_stem, maybe_resize
    rows, _ = load_index(Path(str(source) + ".ovb-index.sqlite"), source)
    if any(row.get("task") != "image_editing" for row in rows):
        raise ValueError("Identity requires a prepared Pico editing archive")
    images = sorted({row["conditioning_image"] for row in rows}, key=lambda ref: ref.member)
    if len(images) < limit:
        raise ValueError(f"Requested {limit} identity samples, but only {len(images)} conditioning images are available")
    random.Random(seed).shuffle(images)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = []
    with tarfile.open(output, "w") as archive:
        for index, reference in enumerate(images[:limit]):
            data = read_reference_bytes(reference)
            from PIL import Image
            with Image.open(io.BytesIO(data)) as image:
                image = maybe_resize(image, 1024)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                data = buffer.getvalue()
            stem = f"id_{index:08d}_{safe_stem(Path(reference.member).stem)}"
            cond = f"conditioning_images/{stem}_source.png"
            target = f"target_images/{stem}_target.png"
            add(archive, cond, data)
            link = tarfile.TarInfo(target)
            link.type, link.linkname = tarfile.LNKTYPE, cond
            archive.addfile(link)
            metadata.append(dict(conditioning_image=cond, image=target, id=stem, task="identity",
                                 source_dataset="identity_from_pico_banana",
                                 text="[TASK: IDENTITY] Reproduce this image exactly without making any changes."))
        add(archive, "metadata.jsonl", ("\n".join(json.dumps(row) for row in metadata) + "\n").encode())


if __name__ == "__main__":
    main()
