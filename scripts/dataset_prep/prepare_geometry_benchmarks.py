#!/usr/bin/env python3
"""Install exact author-distributed geometry benchmark targets and split lists."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.official_sources import DEPTH_FILES, NORMAL_LISTS, MARIGOLD_COMMIT
from common.release_data import sha256
from common.node_scratch import workspace, require_capacity
from scripts.dataset_prep.acquire_geometry_sources import extract_zip
from scripts.analysis.build_reference_normal_subsets import copy_marigold, copy_ibims, copy_sintel, build_vkitti, build_diode, archive


def extract_selected(path, destination, wanted=None):
    """Extract regular files only, and reject paths outside the source tree."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as source:
        selected = [m for m in source.infolist() if not m.is_dir() and (wanted is None or m.filename in wanted)]
        parents = {str(p) for m in selected for p in Path(m.filename).parents}
        require_capacity(destination, sum(m.file_size for m in selected), len(selected) + len(parents))
        for member in source.infolist():
            if member.is_dir() or (wanted is not None and member.filename not in wanted):
                continue
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()) or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Unsafe source member: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def prepare(dataset, raw, root, limit=None, overwrite=False):
    with workspace(f"benchmark-{dataset}") as scratch:
        return prepare_in_workspace(dataset, raw, root, scratch, limit, overwrite)


def prepare_in_workspace(dataset, raw, root, work, limit=None, overwrite=False):
    output_task = "Depth" if dataset == "marigold_depth" else "Normals"
    output = root / "eval" / output_task
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    targets = (["nyu", "kitti_eigen", "eth3d", "scannet", "diode"] if dataset == "marigold_depth"
               else ["vkitti2"] if dataset == "dsine" else ["diode"] if dataset == "diode_normals" else ["nyu", "scannet", "ibims1", "sintel"])
    settings_file = raw / "sources.json"
    settings = json.loads(settings_file.read_text()).get("settings", {}) if settings_file.is_file() else {}
    selected_depth = settings.get("datasets", ",".join(DEPTH_FILES)).split(",")
    if settings.get("frames") and (not limit or limit > int(settings["frames"])):
        raise ValueError("Smoke source assets require a matching preparation limit")
    if dataset == "marigold_depth":
        targets = [{"nyuv2": "nyu", "kitti": "kitti_eigen"}.get(n, n) for n in selected_depth]
    for name in targets:
        if (output / f"{name}.tar").exists() and not overwrite:
            raise FileExistsError(output / f"{name}.tar")
    if dataset == "marigold_depth":
        for name, (filename, split) in DEPTH_FILES.items():
            if name not in selected_depth:
                continue
            lines = [r.split() for r in (raw / "data_split" / split).read_text().splitlines() if r.strip()]
            valid_rows = [r for r in lines if len(r) >= 2 and r[1] != "None"]
            expected = {"nyuv2": 654, "kitti": 652, "eth3d": 454, "scannet": 800, "diode": 771}[name]
            if len(valid_rows) != expected or len({r[0] for r in valid_rows}) != expected:
                raise ValueError(f"Unexpected or duplicate author depth split: {split}")
            source = raw / "archives" / name / filename
            # Keep the author-provided depth bytes. Scoring applies the masks
            # and crops specified by the matching pinned split adapter.
            target = output / f"{'nyu' if name == 'nyuv2' else 'kitti_eigen' if name == 'kitti' else name}.tar"
            partial = target.with_suffix(".tar.partial")
            with tarfile.open(source) as handle:
                members = {m.name.removeprefix("./") for m in handle if m.isfile()}
                rows = valid_rows[:limit] if limit else valid_rows
                required = {value for row in rows for value in row[:3 if name == "diode" else 2]}
                missing = required - members
                if missing:
                    raise ValueError(f"Missing author depth benchmark members in {source}: {sorted(missing)[:4]}")
            shutil.copyfile(source, partial)
            partial.replace(target)
        splits = root / "manifests/benchmark_splits"
        shutil.copytree(raw / "data_split", splits, dirs_exist_ok=True)
    elif dataset == "diode_normals":
        if len((raw / "data_list/val_indoors.csv").read_text().splitlines()) != 325:
            raise ValueError("Expected the 325-image DIODE indoor normal split")
        converted = build_diode(raw, raw / "data_list/val_indoors.csv", limit or 0, work)
        archive(converted, output / "diode.tar")
    elif dataset == "dsine":
        source = next(raw.rglob("dsine_eval.zip"))
        split = raw / "vkitti.txt"
        if len([l for l in split.read_text().splitlines() if l.strip()]) != 1000:
            raise ValueError("Expected the 1,000-image DSINE VKITTI split")
        converted = build_vkitti(source, split, limit or 0, work)
        archive(converted, output / "vkitti2.tar")
    else:
        # Only official benchmark members are retained by the established
        # converters. Float targets and their implicit masks are preserved.
        extract_selected(raw / "evaluation_dataset.zip", work / "source")
        extract_selected(raw / "sintel.zip", work / "source")
        source = next((p for p in (work / "source", *(work / "source").iterdir())
                       if (p / "nyuv2/test").is_dir()), None)
        if source is None:
            raise FileNotFoundError("Author normal bundle has no nyuv2/test directory")
        for name in targets:
            split = raw / "data_split" / NORMAL_LISTS[name]
            expected = {"nyu": 654, "scannet": 300, "ibims1": 100, "sintel": 627}[name]
            if len([s for s in split.read_text().splitlines() if s.strip()]) != expected:
                raise ValueError(f"Unexpected official normal split: {split}")
            if name in {"nyu", "scannet"}:
                converted = copy_marigold(source, split, name, limit or 0, work)
            elif name == "ibims1":
                ibims_source = source / "ibims" if (source / "ibims/ibims").is_dir() else source
                converted = copy_ibims(ibims_source, limit or 0, work)
            else:
                sintel_source = next((p for p in (work / "source", *(work / "source").iterdir())
                                      if (p / "sintel").is_dir()), source)
                converted = copy_sintel(sintel_source, split, limit or 0, work)
            archive(converted, output / f"{name}.tar")
    upstream_commit = "ef0c2afa32b4dd19cb8ca4567c652802cd92591c" if dataset == "dsine" else MARIGOLD_COMMIT if dataset.startswith("marigold") else None
    provenance = dict(source=dataset, upstream_commit=upstream_commit,
                      smoke_limit=limit, artifacts={p.name: sha256(p) for p in (output / f"{n}.tar" for n in targets)})
    manifests = root / "manifests"
    manifests.mkdir(exist_ok=True)
    (manifests / f"{dataset}.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("marigold_depth", "marigold_normals", "dsine", "diode_normals"), required=True)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    prepare(a.dataset, a.raw_dir, a.root, a.limit, a.overwrite)
