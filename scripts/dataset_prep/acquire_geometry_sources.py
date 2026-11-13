#!/usr/bin/env python3
"""Acquire author-distributed geometry assets into the unified raw tree."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.release_data import download_with_resume, sha256, safe_extract_tar
from common.official_sources import DEPTH_BASE, DEPTH_FILES, MARIGOLD_RAW, NORMAL_BASE, NORMAL_LISTS


def extract_zip(path, destination):
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()) or member.external_attr >> 16 & 0o170000 == 0o120000:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        archive.extractall(destination)


def sample_taskonomy(urls, directory, count):
    """Copy matched official TAR members without downloading whole buildings."""
    from common.http_archive import HTTPRangeFile
    from dataset_prep.taskonomy.prepare_taskonomy_depth_normals import frame_key_from_member
    keys = None
    records = []
    for modality in ("rgb", "depth_zbuffer", "normal"):
        url = urls[modality]
        output = directory / Path(url).name
        partial = output.with_suffix(output.suffix + ".partial")
        found = set()
        with HTTPRangeFile(url, block_size=4 * 1024**2) as stream, tarfile.open(fileobj=stream, mode="r:") as source, tarfile.open(partial, "w") as target:
            for member in source:
                key = frame_key_from_member(member.name, "_domain_" + modality)
                if not member.isfile() or key is None or (keys is not None and key not in keys):
                    continue
                if key in found:
                    raise ValueError(f"Duplicate Taskonomy sample: {member.name}")
                target.addfile(member, source.extractfile(member))
                found.add(key)
                if len(found) == count:
                    break
            transferred = stream.transferred
        if len(found) != count or (keys is not None and found != keys):
            raise ValueError(f"Incomplete official Taskonomy modality: {modality}")
        if keys is None:
            keys = found
        partial.replace(output)
        records.append(dict(url=url, path=output.name, sha256=sha256(output),
                            sample_keys=sorted(found), transferred_bytes=transferred, smoke_only=True))
    return records


def acquire(name, directory, settings):
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    def fetch(url, filename):
        path = directory / filename
        receipt = path.with_name(path.name + ".source.json")
        if path.is_file() and receipt.is_file():
            old = json.loads(receipt.read_text())
            if old.get("url") != url or old.get("sha256") != sha256(path):
                raise ValueError(f"Changed download identity: {path}")
        else:
            if not path.is_file():
                download_with_resume(url, path)
            elif path.stat().st_size == 0:
                raise ValueError(f"Empty staged source: {path}")
            if path.suffix == ".zip" and not zipfile.is_zipfile(path):
                raise ValueError(f"Invalid source ZIP: {path}")
            old = dict(url=url, path=filename, bytes=path.stat().st_size, sha256=sha256(path))
            receipt.write_text(json.dumps(old, indent=2) + "\n")
        records.append(old)
        return path
    def unpack(path, destination):
        marker = destination / ("." + path.name + ".extracted")
        if not marker.exists():
            destination.mkdir(parents=True, exist_ok=True)
            if zipfile.is_zipfile(path):
                extract_zip(path, destination)
            else:
                safe_extract_tar(path, destination)
            marker.touch()
    if name == "pd12m":
        command = [sys.executable, str(ROOT / "dataset_prep/pd12m/prepare_pd12m.py"),
                   "--out-dir", str(directory), "--preset", "smoke", "--stage", "all",
                   "--count", str(settings.get("count", 3)), "--scan-limit", str(settings.get("scan-limit", 100)),
                   "--workers", str(settings.get("workers", 2)), "--retries", "1", "--read-timeout", "15"]
        subprocess.run(command, check=True)
        records.append(dict(source="Spawning/PD12M", smoke_only=True,
                            selection_sha256=sha256(directory / "selection_info.json")))
    elif name == "pico_banana":
        from common.pico_packed import acquire as acquire_pico
        from common.node_scratch import workspace
        with workspace("pico-images") as scratch:
            records.append(acquire_pico(directory, settings, scratch))
    elif name == "marigold_depth":
        requested = set(settings.get("datasets", ",".join(DEPTH_FILES)).split(","))
        if not requested or requested - DEPTH_FILES.keys():
            raise ValueError("Unknown Marigold depth source selection")
        for dataset, (archive, split) in DEPTH_FILES.items():
            split_file = fetch(f"{MARIGOLD_RAW}/evaluation/data_split/{split}", f"data_split/{split}")
            if dataset not in requested:
                continue
            url = f"{DEPTH_BASE}/{dataset}/{archive}"
            frames = int(settings.get("frames", 0))
            if frames:
                from common.http_archive import HTTPRangeFile
                lines = [l.split() for l in split_file.read_text().splitlines() if l.strip()]
                selected_rows = [r for r in lines if len(r) >= 2 and r[1] != "None"][:frames]
                wanted = {v for row in selected_rows for v in row[:3 if dataset == "diode" else 2]}
                destination = directory / "archives" / dataset / archive
                destination.parent.mkdir(parents=True, exist_ok=True)
                found = set()
                with HTTPRangeFile(url) as stream, tarfile.open(fileobj=stream, mode="r:") as source, tarfile.open(str(destination)+".partial", "w") as target:
                    for member in source:
                        normalized = member.name.removeprefix("./")
                        if normalized not in wanted or not member.isfile():
                            continue
                        target.addfile(member, source.extractfile(member))
                        found.add(normalized)
                        if found == wanted:
                            break
                if found != wanted:
                    raise ValueError(f"Missing official depth smoke members: {wanted-found}")
                Path(str(destination)+".partial").replace(destination)
                records.append(dict(url=url, smoke_members=sorted(found), sha256=sha256(destination)))
            else:
                fetch(url, f"archives/{dataset}/{archive}")
        # These lists also define exclusions for NZS training.
        for split in NORMAL_LISTS.values():
            fetch(f"{MARIGOLD_RAW}/evaluation/data_split/{split}", f"data_split/{split}")
    elif name == "marigold_normals":
        for split in NORMAL_LISTS.values():
            fetch(f"{MARIGOLD_RAW}/evaluation/data_split/{split}", f"data_split/{split}")
        frames = int(settings.get("frames", 0))
        if frames:
            from common.http_archive import HTTPRangeFile
            for filename in ("evaluation_dataset.zip", "sintel.zip"):
                url = f"{NORMAL_BASE}/{filename}"
                with HTTPRangeFile(url) as stream, zipfile.ZipFile(stream) as source:
                    names = set(source.namelist())
                    wanted = set()
                    datasets = ("sintel",) if filename == "sintel.zip" else ("nyu", "scannet", "ibims1")
                    for dataset in datasets:
                        rows = (directory / "data_split" / NORMAL_LISTS[dataset]).read_text().splitlines()[:frames]
                        prefix = {"nyu": "nyuv2/test/", "scannet": "scannet/", "ibims1": "ibims/ibims/", "sintel": "sintel/"}[dataset]
                        for row in rows:
                            for value in row.split()[:2]:
                                wanted.add(prefix + value)
                    missing = wanted - names
                    if missing:
                        raise ValueError(f"Official normal sample members missing: {sorted(missing)[:4]}")
                    partial = directory / (filename + ".partial")
                    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as target:
                        for member in sorted(wanted):
                            data = source.read(member)
                            target.writestr(member, data)
                            import hashlib
                            records.append(dict(url=url, member=member, sha256=hashlib.sha256(data).hexdigest()))
                    partial.replace(directory / filename)
        else:
            for filename in ("evaluation_dataset.zip", "sintel.zip"):
                fetch(f"{NORMAL_BASE}/{filename}", filename)
    elif name == "hypersim":
        tool = fetch("https://raw.githubusercontent.com/apple-aiml-research/ml-hypersim/main/code/python/tools/dataset_download_images.py", "upstream/dataset_download_images.py")
        tree = ast.parse(tool.read_text())
        urls = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "urls_to_download" for t in node.targets))
        requested = settings.get("scenes", "all")
        scenes = set(requested.split(",")) if requested != "all" else None
        selected = [url for url in urls if scenes is None or Path(url).stem in scenes]
        if not selected or (scenes is not None and {Path(u).stem for u in selected} != scenes):
            raise ValueError("Unknown Hypersim scene selection")
        for url in selected:
            frames = int(settings.get("frames", 0))
            if frames:
                from common.http_archive import HTTPRangeFile
                with HTTPRangeFile(url) as stream, zipfile.ZipFile(stream) as archive:
                    colors = sorted(n for n in archive.namelist() if n.endswith(".color.hdf5"))[:frames]
                    if len(colors) != frames:
                        raise ValueError("Requested Hypersim smoke frames are unavailable")
                    wanted = set()
                    for color in colors:
                        base = color.replace("_final_hdf5/", "_geometry_hdf5/").removesuffix(".color.hdf5")
                        wanted.update((color, base+".depth_meters.hdf5", base+".normal_cam.hdf5"))
                    wanted.update(n for n in archive.namelist() if "/_detail/" in n or n.endswith(".csv"))
                    for member in sorted(wanted):
                        target = (directory / member).resolve()
                        if not target.is_relative_to(directory.resolve()):
                            raise ValueError("Unsafe Hypersim member")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member) as src, target.open("wb") as dst:
                            import shutil
                            shutil.copyfileobj(src, dst)
                        records.append(dict(url=url, member=member, sha256=sha256(target)))
            else:
                # Geometry and segmentation readers consume each HDF5 member
                # directly. Keep the scene ZIP without a persistent loose tree.
                fetch(url, "archives/" + Path(url).name)
        fetch("https://raw.githubusercontent.com/apple-aiml-research/ml-hypersim/main/contrib/mikeroberts3000/metadata_camera_parameters.csv", "metadata_camera_parameters.csv")
    elif name == "taskonomy":
        links = fetch("https://datasets.epfl.ch/taskonomy/links.txt", "links.txt")
        groups = {}
        for url in links.read_text().splitlines():
            match = re.search(r"/([^/]+)_(rgb|depth_zbuffer|normal)\.tar$", url.strip())
            if match:
                groups.setdefault(match[1], {})[match[2]] = url.strip()
        buildings = sorted(k for k, v in groups.items() if len(v) == 3)
        count = settings.get("num-buildings", "all")
        if count != "all":
            buildings = sorted(random.Random(int(settings.get("seed", 42))).sample(buildings, int(count)))
        if not buildings:
            raise ValueError("No complete Taskonomy modality triplets")
        for building in buildings:
            frames = int(settings.get("frames", 0))
            if frames:
                if frames < 1 or count == "all":
                    raise ValueError("Taskonomy smoke frames require a positive count and bounded buildings")
                records.extend(sample_taskonomy(groups[building], directory, frames))
                continue
            for url in groups[building].values():
                path = fetch(url, Path(url).name)
                with tarfile.open(path) as archive:
                    for _ in archive:
                        pass
    elif name == "replica":
        parts = [fetch("https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/"
                       f"replica_v1_0.tar.gz.parta{letter}", f"archives/replica_v1_0.tar.gz.parta{letter}")
                 for letter in "abcdefghijklmnopq"]
        combined = directory / "archives/replica_v1_0.tar.gz"
        if not combined.is_file():
            import shutil
            partial = combined.with_suffix(".partial")
            with partial.open("wb") as output:
                for part in parts:
                    with part.open("rb") as source:
                        shutil.copyfileobj(source, output)
            partial.replace(combined)
        unpack(combined, directory)
        unpack(fetch("https://dl.fbaipublicfiles.com/habitat/Replica/additional_habitat_configs.zip", "additional_habitat_configs.zip"), directory)
    elif name in {"diode", "diode_normals"}:
        for split in (("train", "val") if name == "diode" else ("val",)):
            for suffix in ("", "_normals"):
                archive = f"{split}{suffix}.tar.gz"
                path = fetch(f"https://diode-dataset.s3.amazonaws.com/{archive}", archive)
                # Preparation streams original archives. Full extraction would
                # add hundreds of GB and tens of thousands of source files.
        unpack(fetch("https://diode-1254389886.cos.ap-hongkong.myqcloud.com/data_list.zip", "data_list.zip"), directory)
    elif name == "dsine":
        fetch("https://raw.githubusercontent.com/baegwangbin/DSINE/ef0c2afa32b4dd19cb8ca4567c652802cd92591c/data/datasets/vkitti/split/vkitti.txt", "vkitti.txt")
        staged = list(directory.rglob("dsine_eval.zip"))
        if not staged:
            # Fetch only the dataset, not model checkpoints sharing its folder.
            subprocess.run([sys.executable, "-m", "gdown",
                            "https://drive.google.com/uc?id=1u6_-Zvg7ufKwlq0ManQgL-xJL_DwUCR_",
                            "-O", str(directory / "dsine_eval.zip")], check=True)
            staged = list(directory.rglob("dsine_eval.zip"))
        if len(staged) != 1:
            raise ValueError("Place the author's dsine_eval.zip in " + str(directory))
        from dataset_prep.surface_normals.prepare_dsine_nyu_eval import EXPECTED_SOURCE_SHA256
        if sha256(staged[0]) != EXPECTED_SOURCE_SHA256:
            raise ValueError("DSINE source checksum mismatch; expected the pinned author's 2024-04-23 bundle")
        with zipfile.ZipFile(staged[0]) as archive:
            if not any(n.startswith("dsine_eval/vkitti/") for n in archive.namelist()):
                raise ValueError("DSINE archive lacks VKITTI benchmark targets")
        records.append(dict(path=str(staged[0].relative_to(directory)), sha256=sha256(staged[0])))
    else:
        raise ValueError(f"Unknown official source: {name}")
    payload = dict(source=name, settings=settings, files=records)
    temp = directory / "sources.json.partial"
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(directory / "sources.json")


if __name__ == "__main__":
    acquire(os.environ["OVB_OFFICIAL_SOURCE"], Path(os.environ["OVB_RAW_DIRECTORY"]),
            json.loads(os.environ.get("OVB_DOWNLOAD_SETTINGS", "{}")))
