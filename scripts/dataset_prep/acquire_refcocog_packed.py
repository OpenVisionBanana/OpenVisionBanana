#!/usr/bin/env python3
"""Acquire original refer annotations and retain COCO photos in their ZIP."""
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.coco_eval_holdouts import _read_refcocog_umd_val_ids
from common.node_scratch import workspace
from common.refcocog_packed import annotations, check_images, publish_copy, zip_entries
from common.release_data import download_with_resume

ANNOTATIONS_URL = "https://web.archive.org/web/20220413012904id_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip"
# Same public COCO bucket, using an HTTPS hostname with a valid certificate.
IMAGES_URL = "https://s3.amazonaws.com/images.cocodataset.org/zips/train2014.zip"


def acquire_annotations(raw, fetch=download_with_resume):
    raw.mkdir(parents=True, exist_ok=True)
    if not (raw / "refs(umd).p").is_file() or not (raw / "instances.json").is_file():
        archive = raw / "refcocog.zip"
        if not archive.exists():
            fetch(os.environ.get("REFCOCOG_ZIP_URL", ANNOTATIONS_URL), archive)
        with zipfile.ZipFile(archive) as source:
            entries = zip_entries(source)
            with workspace("refcocog-annotations") as work:
                for name in ("refs(umd).p", "instances.json"):
                    target = raw / name
                    if target.exists():
                        continue
                    matches = [m for path, m in entries.items() if Path(path).name == name and not m.is_dir()]
                    if len(matches) != 1:
                        raise ValueError(f"Expected exactly one {name} in original annotation ZIP")
                    from common.node_scratch import require_capacity
                    require_capacity(work, matches[0].file_size, 1)
                    staged = work / name
                    staged.write_bytes(source.read(matches[0]))
                    publish_copy(staged, target)
    directory, _, _ = annotations(raw)
    # Keep the existing full UMD val ID/hash contract, including on reuse.
    _read_refcocog_umd_val_ids(directory)
    return directory


def acquire(raw, shared_zip, fetch=download_with_resume):
    acquire_annotations(raw, fetch)
    if not (raw / "train2014").is_dir() and not (raw / "train2014.zip").is_file():
        shared_zip.parent.mkdir(parents=True, exist_ok=True)
        if not shared_zip.is_file():
            fetch(os.environ.get("TRAIN2014_URL", IMAGES_URL), shared_zip)
        target = raw / "train2014.zip"
        if target.absolute() != shared_zip.absolute():
            target.symlink_to(os.path.relpath(shared_zip.resolve(), target.parent))
    check_images(raw)


if __name__ == "__main__":
    acquire(Path(os.environ["OVB_REFCOCOG_RAW"]).resolve(),
            Path(os.environ["TRAIN2014_ZIP"]).absolute())
