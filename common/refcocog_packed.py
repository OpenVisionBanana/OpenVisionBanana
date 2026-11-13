"""Archive I/O around the existing RefCOCOg converters; no split/mask changes."""
from contextlib import contextmanager
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import shutil
import stat
import tempfile
import zipfile

from common.node_scratch import require_capacity, workspace


def zip_entries(archive):
    """Validate the entire ZIP namespace before copying any selected members."""
    entries = {}
    for item in archive.infolist():
        path = PurePosixPath(item.filename)
        mode = item.external_attr >> 16
        if (not item.filename or path.is_absolute() or ".." in path.parts
                or "\\" in item.filename or path.as_posix() != item.filename.rstrip("/")
                or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR))):
            raise ValueError(f"Unsafe ZIP member: {item.filename}")
        if path.as_posix() in entries:
            raise ValueError(f"Duplicate ZIP member: {item.filename}")
        entries[path.as_posix()] = item
    return entries


def annotations(raw):
    """Read only trusted original refer pickle files, as the legacy reader does."""
    matches = sorted(raw.rglob("refs(umd).p"))
    if len(matches) != 1:
        raise ValueError(f"Expected one original refs(umd).p below {raw}")
    directory = matches[0].parent
    with matches[0].open("rb") as handle:
        try:
            refs = pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0)
            refs = pickle.load(handle, encoding="latin1")
    instances = json.loads((directory / "instances.json").read_text())
    images = {image["id"]: image for image in instances["images"]}
    anns = {ann["id"]: ann for ann in instances["annotations"]}
    names = {}
    for ref in refs:
        ann = anns[ref["ann_id"]]
        image = images[ann["image_id"]]
        name = image["file_name"]
        if not name or PurePosixPath(name).name != name or "\\" in name or name in (".", ".."):
            raise ValueError(f"Unsafe COCO image name: {name}")
        names[ref["ref_id"]] = name
    return directory, refs, names


def image_archive(raw):
    path = raw / "train2014.zip"
    if not path.is_file():
        raise FileNotFoundError(f"Expected packed COCO train2014 photos at {path}")
    return path


def required_images(raw, split=None):
    directory, refs, names = annotations(raw)
    selected = {names[ref["ref_id"]] for ref in refs if split is None or ref["split"] == split}
    if not selected:
        raise ValueError(f"No RefCOCOg images for split {split}")
    return directory, selected


def check_images(raw):
    _, selected = required_images(raw)
    if (raw / "train2014").is_dir():
        missing = [name for name in selected if not (raw / "train2014" / name).is_file()]
    else:
        with zipfile.ZipFile(image_archive(raw)) as archive:
            entries = zip_entries(archive)
            missing = [name for name in selected if "train2014/" + name not in entries
                       or entries["train2014/" + name].file_size <= 0]
    if missing:
        raise ValueError(f"Missing {len(missing)} referenced COCO images, e.g. {missing[:3]}")


@contextmanager
def validation_images(raw):
    """Stage only UMD val photos, never all COCO train2014 images."""
    directory, selected = required_images(raw, "val")
    with workspace("refcocog-eval") as work:
        destination = work / "refcocog/train2014"
        destination.mkdir(parents=True)
        if (raw / "train2014").is_dir():
            sources = [raw / "train2014" / name for name in sorted(selected)]
            require_capacity(work, sum(path.stat().st_size for path in sources), len(sources) + 4)
            for source in sources:
                shutil.copyfile(source, destination / source.name)
        else:
            with zipfile.ZipFile(image_archive(raw)) as archive:
                entries = zip_entries(archive)
                members = [entries["train2014/" + name] for name in sorted(selected)]
                require_capacity(work, sum(member.file_size for member in members), len(members) + 4)
                for member in members:
                    if member.file_size <= 0 or member.is_dir():
                        raise ValueError(f"Empty COCO image: {member.filename}")
                    # ZipExtFile verifies CRC while streaming; no extractall.
                    with archive.open(member) as source, (destination / Path(member.filename).name).open("wb") as target:
                        shutil.copyfileobj(source, target)
        yield directory, work / "refcocog"


def publish_copy(source, target):
    """Atomically create a persistent file; refuse to overwrite existing inputs."""
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=".refcocog-", suffix=".partial", dir=target.parent)
    os.close(handle)
    partial = Path(name)
    try:
        shutil.copyfile(source, partial)
        os.link(partial, target)
    finally:
        partial.unlink(missing_ok=True)
