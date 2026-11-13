#!/usr/bin/env python3
"""Build a training pool from the paper's prepared per-source members.

Every member is one indexed tar under prepared/<version>/train/<Task>/ —
semseg/instanceseg one tar per source, refseg one tar per annotation source,
plus the editing/identity/generation members. Members are shared: a pool is
only a manifest (prepared/<version>/manifests/<pool>_pool.json) that selects
members and weights them, so building a second pool reuses every member the
first one already published and converts only the missing sources.

All members use the paper settings (best64 semseg colours, fixed-draw 32
left-to-right instance colours, 336px refseg minimum, seed 42, full
Pico-Banana, 250k PD12M, 50k identity). Weights follow the paper: each task's
mass (non-geometry total 0.7), split inside the task by prepared row count.

zeroshot     the zero-shot sources: Cityscapes, COCO, Synscapes, IDD, WildDash 2,
             nuImages, COCONut-XL, PACO-LVIS, ReasonSeg and SteerViT's COCO rows
             stay out.
nonzeroshot  the non-zero-shot sources: zeroshot plus all benchmark-domain sources.
<any name>   your own pool: --pool my_pool --sources <names>. --list-sources
             prints every member source with its raw catalog datasets.

Each manifest keeps the 0.7 other-task budget. The complete official-source
recipes in configs/data_zs.yaml and configs/data_nzs.yaml add geometry and
write the final training manifest after checking the prepared archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from common.release_data import BY_NAME, raw_directory  # noqa: E402

# Paper non-geometry task masses. Tasks a custom pool omits return their share
# of the 0.7 other-task budget to the remaining tasks.
TASK_MASSES = {
    "semseg": 0.1432, "instanceseg": 0.1432, "refseg": 0.1316833,
    "pico_banana": 0.1132359, "identity": 0.0554449, "pd12m": 0.1132359,
}
OTHER_TASK_BUDGET = 0.7
SAMPLES_PER_EPOCH = 1_000_000


@dataclass(frozen=True)
class SegSource:
    """One segmentation conversion source and the member tars it publishes."""
    semseg: tuple[str, ...]        # tar stems under train/Semseg/
    instanceseg: tuple[str, ...]   # tar stems under train/Instanceseg/
    zeroshot: bool
    raw: tuple[str, ...]           # catalog dataset names it needs downloaded
    disable: tuple[tuple[str, str], ...]  # env that excludes it from a run


ABSENT = "{work}/absent/missing.tar"  # a path that never exists: the source is skipped.

SEG_SOURCES: dict[str, SegSource] = {
    "ade20k": SegSource(("ade20k",), ("ade20k",), True, ("ade20k",),
                        (("ADE20K_TAR", ABSENT), ("ADE_INST_TAR", ABSENT))),
    "voc": SegSource(("voc", "pascal_context"), ("voc",), True, ("voc2012",),
                     (("VOC_TAR", ABSENT), ("VOC_INST_TAR", ABSENT))),
    "acdc": SegSource(("acdc",), ("acdc",), True, ("acdc",), (("ACDC_TAR", ABSENT),)),
    "bdd": SegSource(("bdd",), (), True, ("bdd100k",), (("BDD_TAR", ABSENT),)),
    "mapillary": SegSource(("mapillary",), ("mapillary",), True, ("mapillary_vistas",),
                           (("MAPILLARY_TAR", ABSENT), ("MAPILLARY_SEMSEG_ENABLE", "0"))),
    "goose": SegSource(("goose",), ("goose",), True, ("goose",), (("GOOSE_TAR", ABSENT),)),
    "muses": SegSource(("muses",), ("muses",), True, ("muses",), (("MUSES_TAR", ABSENT),)),
    "vipseg": SegSource(("vipseg",), ("vipseg",), True, ("vipseg",), (("VIPSEG_TAR", ABSENT),)),
    "waymo": SegSource(("waymo",), ("waymo",), True, ("waymo",), (("WAYMO_TAR", ABSENT),)),
    "foodseg103": SegSource(("foodseg103",), (), True, ("foodseg103",), (("FOODSEG103_TAR", ABSENT),)),
    "trans10k": SegSource(("trans10k",), (), True, ("trans10k",), (("TRANS10K_TAR", ABSENT),)),
    "railsem19": SegSource(("railsem19",), (), True, ("railsem19",), (("RAILSEM19_TAR", ABSENT),)),
    "ovis": SegSource((), ("ovis",), True, ("ovis",), (("OVIS_TAR", ABSENT),)),
    "hypersim": SegSource(("hypersim",), ("hypersim",), True, ("hypersim",),
                          (("HYPERSIM_SEG_TAR", ABSENT), ("ALLOW_MISSING_HYPERSIM", "1"))),
    "cityscapes": SegSource(("cityscapes",), (), False, ("cityscapes",), (("INCLUDE_CITYSCAPES", "0"),)),
    "synscapes": SegSource(("synscapes",), ("synscapes",), False, ("synscapes",), (("SYNSCAPES_ENABLE", "0"),)),
    "idd": SegSource(("idd",), ("idd",), False, ("idd",), (("IDD_ENABLE", "0"),)),
    "wilddash2": SegSource(("wilddash2",), ("wilddash2",), False, ("wilddash2",), (("WILDDASH2_ENABLE", "0"),)),
    "nuimages": SegSource(("nuimages",), ("nuimages",), False, ("nuimages",), (("NUIMAGES_ENABLE", "0"),)),
    "coconut_xl": SegSource(("coconutxl_coco", "coconutxl_objects365"),
                            ("coconutxl_coco", "coconutxl_objects365"), False,
                            ("coconut_xl", "coco2017"), (("COCONUT_XL_ENABLE", "0"),)),
    "paco_lvis": SegSource((), ("paco_lvis",), False, ("paco_lvis",), (("PACO_LVIS_ENABLE", "0"),)),
}

# Refseg members: one prepared tar per annotation source (train/Refseg/refseg_<name>.tar).
# SteerViT entries are (local dataset dir under raw/Refseg/steervit or None for the HF
# cache, extra converter arguments, image roots to stage). vg_mapillary is the
# zero-shot refseg member: SteerViT without its COCO rows.
STEERVIT_MEMBERS: dict[str, tuple[str | None, tuple[str, ...], tuple[str, ...]]] = {
    "vg_mapillary": (None, ("--include-datasets", "visual genome,mapillary", "--ignore-coco"),
                     ("visual_genome", "mapillary")),
    "refcoco": (None, ("--include-datasets", "refcoco"), ("mscoco",)),
    "refcocoplus": (None, ("--include-datasets", "refcoco+"), ("mscoco",)),
    "refcocog": (None, ("--include-datasets", "refcocog"), ("mscoco",)),
    "lvis": (None, ("--include-datasets", "lvis"), ("mscoco",)),
    "grefcoco": ("SteerViT-gRefCOCO", (), ("mscoco",)),
    "phrasecut": ("SteerViT-PhraseCutDataset", (), ("visual_genome",)),
}
REFSEG_MEMBERS = tuple(STEERVIT_MEMBERS) + ("reasonseg",)
REFSEG_ZEROSHOT = ("vg_mapillary",)
REFSEG_RAW = {
    "vg_mapillary": ("steervit", "visual_genome", "mapillary_vistas", "refcocog"),
    "grefcoco": ("steervit", "coco2014", "refcocog"),
    "phrasecut": ("steervit", "visual_genome", "refcocog"),
    "reasonseg": ("reasonseg",),
    **dict.fromkeys(("refcoco", "refcocoplus", "refcocog", "lvis"),
                    ("steervit", "coco2014", "refcocog")),
}

SINGLES = {  # member name -> (Task directory, tar stem, raw catalog datasets)
    "pico_banana": ("ImgEdit", "pico_banana_sft_full", ("pico_banana",)),
    "identity": ("ImgEdit", "identity_50000", ("pico_banana",)),
    "pd12m": ("ImgGeneration", "pd12m_250000", ("pd12m",)),
}

POOLS = {
    "zeroshot": ([name for name, s in SEG_SOURCES.items() if s.zeroshot],
                 list(REFSEG_ZEROSHOT), list(SINGLES)),
    "nonzeroshot": (list(SEG_SOURCES), list(REFSEG_MEMBERS), list(SINGLES)),
}
ALL_SOURCES = {**{n: "seg" for n in SEG_SOURCES}, **{n: "refseg" for n in REFSEG_MEMBERS},
               **{n: "single" for n in SINGLES}}


def raw(root: Path, name: str, *parts: str) -> str:
    return str(raw_directory(root, BY_NAME[name]).joinpath(*parts))


def published(archive: Path) -> bool:
    return archive.is_file() and Path(f"{archive}.ovb-index.sqlite").is_file()


def rows_in(index: Path) -> int:
    with sqlite3.connect(f"file:{index}?mode=ro", uri=True) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    rows = int(metadata["num_rows"])
    if rows <= 0:
        raise SystemExit(f"ERROR: indexed tar has no rows: {index}")
    return rows


def unpublish(archive: Path) -> None:
    for path in (archive, Path(f"{archive}.ovb-index.sqlite"), Path(f"{archive}.sha256")):
        path.unlink(missing_ok=True)


def run(command: list[str], env: dict[str, str] | None, dry_run: bool) -> None:
    shown = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted((env or {}).items()))
    print(f"+ {shown} {shlex.join(str(part) for part in command)}".strip(), flush=True)
    if dry_run:
        return
    process_env = os.environ.copy()
    process_env.update(env or {})
    # Tools installed next to this interpreter must be found without activation.
    process_env["PATH"] = os.pathsep.join(filter(None, (str(Path(sys.executable).parent), process_env.get("PATH"))))
    subprocess.run([str(part) for part in command], cwd=REPO, env=process_env, check=True)


def list_sources() -> None:
    print("source             task(s)                pool      raw catalog dataset(s) to download")
    for name, source in SEG_SOURCES.items():
        tasks = ",".join(filter(None, ("Semseg" if source.semseg else "", "Instanceseg" if source.instanceseg else "")))
        pool = "ZS+NZS" if source.zeroshot else "NZS"
        print(f"{name:18} {tasks:22} {pool:9} {','.join(source.raw)}")
    for name in REFSEG_MEMBERS:
        pool = "ZS+NZS" if name in REFSEG_ZEROSHOT else "NZS"
        print(f"{name:18} {'Refseg':22} {pool:9} {','.join(REFSEG_RAW[name])}")
    for name, (task, _, raw_names) in SINGLES.items():
        print(f"{name:18} {task:22} {'ZS+NZS':9} {','.join(raw_names)}")


# ---- Segmentation members ------------------------------------------------------

def seg_tar(root: Path, task: str, stem: str) -> Path:
    return root / "train" / task / f"{stem}.tar"


def seg_members(root: Path, name: str) -> list[tuple[str, Path]]:
    source = SEG_SOURCES[name]
    return ([("Semseg", seg_tar(root, "Semseg", stem)) for stem in source.semseg]
            + [("Instanceseg", seg_tar(root, "Instanceseg", stem)) for stem in source.instanceseg])


def stage_env(raw_root: Path, work: Path, needed_seg: list[str], singles: list[str],
              args: argparse.Namespace) -> dict[str, str]:
    """Environment for one conversion run building exactly the named sources."""
    workers = str(args.workers)
    env = {
        "ROOT": str(REPO),
        "PYTHON": sys.executable,
        "DATA_ROOT": str(raw_root),
        "STAGING_ROOT": str(args.staging_root),
        "PREPARED_ROOT": str(work),
        "CONFIG_OUT": str(work / "open_visionbanana_mix.json"),
        "MAKE_TOML": "0",
        # Inputs, one raw/<Group>/<dataset>/ folder each.
        "SEMSEG_SOURCE_ROOT": str(raw_root / "Semseg"),
        "INSTANCESEG_SOURCE_ROOT": str(raw_root / "Instanceseg"),
        "SHARED_SEG_ROOT": str(raw_root / "Shared_seg"),
        "REFSEG_SOURCE_ROOT": str(raw_root / "Refseg"),
        "REFCOCOG_UMD_ROOT": raw(raw_root, "refcocog"),
        "ADE20K_TAR": raw(raw_root, "ade20k", "ade20k.tar"),
        "ADE_INST_TAR": raw(raw_root, "ade20k", "ade20k_instanceseg.tar"),
        "ADE_INST_INFO_DIR": raw(raw_root, "ade20k"),
        "VOC_TAR": raw(raw_root, "voc2012", "voc.tar"),
        "VOC_INST_TAR": raw(raw_root, "voc2012", "VOC2012.tar"),
        "ACDC_TAR": raw(raw_root, "acdc", "acdc.tar"),
        "BDD_TAR": raw(raw_root, "bdd100k", "bdd_10k.tar"),
        "MAPILLARY_TAR": raw(raw_root, "mapillary_vistas", "mapillary.tar"),
        "GOOSE_TAR": raw(raw_root, "goose", "goose.tar"),
        "MUSES_TAR": raw(raw_root, "muses", "muses.tar"),
        "VIPSEG_TAR": raw(raw_root, "vipseg", "vipseg.tar"),
        "WAYMO_TAR": raw(raw_root, "waymo", "waymo.tar"),
        "OVIS_TAR": raw(raw_root, "ovis", "ovis.tar"),
        "FOODSEG103_TAR": raw(raw_root, "foodseg103", "foodseg103.tar"),
        "TRANS10K_TAR": raw(raw_root, "trans10k", "trans10k.tar"),
        "RAILSEM19_TAR": raw(raw_root, "railsem19", "railsem19.tar"),
        "HYPERSIM_SEG_TAR": str(work.parent / "hypersim_seg.tar"),
        "CITYSCAPES_TAR": raw(raw_root, "cityscapes", "cityscapes.tar"),
        "CITYSCAPES_SPLITS": "train",
        "SYNSCAPES_SOURCE_ROOT": raw(raw_root, "synscapes"),
        "SYNSCAPES_TAR": raw(raw_root, "synscapes", "synscapes.tar"),
        "IDD_TAR": raw(raw_root, "idd", "idd_1_2.tar"),
        "WILDDASH2_TAR": raw(raw_root, "wilddash2", "wd_public_v2p.tar"),
        "NUIMAGES_TAR": raw(raw_root, "nuimages", "nuimages.tar"),
        "NUIMAGES_SAMPLE_INTERVAL_SECONDS": "10",
        "COCONUT_XL_SOURCE_ROOT": raw(raw_root, "coconut_xl", "coco"),
        "PACO_LVIS_TAR": raw(raw_root, "paco_lvis", "paco-lvis-DatasetNinja.tar"),
        "HF_HOME": raw(raw_root, "steervit", "huggingface"),
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PICO_SOURCE_ROOT": raw(raw_root, "pico_banana"),
        "SKIP_PICO_TARGET_DOWNLOAD": "1",
        "PD12M_SHARDS_DIR": raw(raw_root, "pd12m", "shards"),
        # Paper settings.
        "SEMSEG_COLOR_MAPPING": "best64_maxnegdraw",
        "INSTANCESEG_DRAW_MODE": "fixeddraw_32_lr",
        "MAPILLARY_SEMSEG_ENABLE": "1", "INSTANCESEG_ENABLE": "1",
        "SEMSEG_NUM_IMAGES": str(args.semseg_num_images or ""),
        "INSTANCESEG_LIMIT": str(args.semseg_num_images or ""),
        "PICO_LIMIT": str(args.limit or ""),
        "PD12M_LIMIT": str(min(args.limit, 250000) if args.limit else 250000),
        "IDENTITY_LIMIT": str(min(args.limit, 50000) if args.limit else 50000),
        "SEMSEG_WEIGHT": "0.247525", "INSTANCESEG_WEIGHT": "0.247525",
        "PICO_WEIGHT": "0.118812", "PD12M_WEIGHT": "0.118812", "IDENTITY_WEIGHT": "0.079207",
        # Refseg members are built per source outside this pipeline.
        "REFSEG_ENABLE": "0", "REASONSEG_REFSEG_ENABLE": "0",
        # Members stay directories: they are split into per-source tars afterwards.
        "ARCHIVE_SEGMENTATION_MEMBERS": "0",
        "SEMSEG_NUM_WORKERS": workers, "INSTANCESEG_NUM_WORKERS": workers,
        "PICO_NUM_WORKERS": workers,
        "USE_CONTAINER": "0",
        "SEMSEG_OUT": str(work / "semseg"),
        "INSTANCESEG_OUT": str(work / "instanceseg"),
        # Member folder names become the tar inner roots: keep the published names.
        "PICO_OUT": str(work / SINGLES["pico_banana"][1]),
        "IDENTITY_OUT": str(work / SINGLES["identity"][1]),
        "PD12M_OUT": str(work / SINGLES["pd12m"][1]),
    }
    for name, source in SEG_SOURCES.items():
        if name not in needed_seg:
            env.update((key, value.format(work=work)) for key, value in source.disable)
    for name, flag in (("pico_banana", "PICO_ENABLE"), ("identity", "IDENTITY_ENABLE")):
        if name not in singles:
            env[flag] = "0"
    if "pd12m" not in singles:
        env["PD12M_WEIGHT"] = "0"
    if not any(SEG_SOURCES[name].semseg for name in needed_seg):
        # The pipeline requires a semantic member; a pre-seeded placeholder makes it
        # reuse "existing" output so editing/generation members can build alone. The
        # run's own specs validation is skipped (archiving validates each member).
        placeholder = work / "semseg_placeholder"
        env["SEMSEG_OUT"] = str(placeholder)
        env["VALIDATE"] = "0"
        if not args.dry_run:
            placeholder.mkdir(parents=True, exist_ok=True)
            (placeholder / "metadata.jsonl").write_text("{}\n", encoding="utf-8")
    return env


def hypersim_segmentation(raw_root: Path, tar: Path, args: argparse.Namespace) -> None:
    if tar.is_file() and tar.stat().st_size:
        print(f"Using existing {tar}")
        return
    scenes = raw_directory(raw_root, BY_NAME["hypersim"])
    if not any(scenes.glob("**/scene_cam_*_final_hdf5")) and not any((scenes / "archives").glob("*.zip")) and not args.dry_run:
        if args.allow_missing_hypersim:
            print(f"WARNING: no Hypersim scenes under {scenes}; building without Hypersim.")
            return
        raise SystemExit(f"ERROR: the pool needs Hypersim scenes under {scenes} "
                         f"(or pass --set {args.pool}_pool.allow-missing-hypersim=true).")
    # Same converter and frozen settings as the shell helper, now with ZIP input
    # support and loose outputs inside the pool's owned scratch workspace.
    run([sys.executable, "-m", "dataset_prep.hypersim.prepare_hypersim_segmentation",
         "--raw-root", str(scenes), "--out-root", str(tar.parent / "hypersim_seg"),
         "--prepared-tar", str(tar), "--all-frames", "--val-ratio", "0.02",
         "--min-labeled-fraction", "0.10", "--max-side", "0", "--num-workers", str(args.workers),
         "--seed", "42", "--overwrite"], None, args.dry_run)


def split_and_publish(root: Path, work: Path, task: str, args: argparse.Namespace) -> None:
    """Split one combined task member into per-source tars and publish them."""
    member = work / task.lower()
    metadata = member / "metadata.jsonl"
    if args.dry_run or not (metadata.is_file() and metadata.stat().st_size):
        if args.dry_run:
            print(f"+ split {member} into per-source tars -> {root / 'train' / task}")
        return
    run([sys.executable, str(REPO / "scripts/dataset_prep/split_prepared_member_by_source.py"),
         "--member", str(member), "--report", str(member / "per_source_members.json")], None, False)
    destination = root / "train" / task
    destination.mkdir(parents=True, exist_ok=True)
    report = json.loads((member / "per_source_members.json").read_text(encoding="utf-8"))
    for item in report["members"]:
        archive = Path(item["archive"])
        for suffix in ("", ".ovb-index.sqlite", ".sha256"):
            source = Path(f"{archive}{suffix}")
            target = destination / source.name
            if source.is_file() and not target.exists():
                shutil.move(str(source), str(target))
    shutil.rmtree(member, ignore_errors=True)


# ---- Refseg members ------------------------------------------------------------

def extract_archive_subset(archive: Path, destination: Path, keep, dry_run: bool) -> None:
    """Extract the entries `keep` maps to a relative path, atomically, resumably."""
    if dry_run:
        print(f"+ extract {archive} -> {destination}")
        return
    root = destination.resolve()

    def write(rel: str, opener) -> None:
        target = destination / rel
        if root not in target.resolve().parents:
            raise SystemExit(f"ERROR: unsafe path in {archive}: {rel}")
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".partial")
        with opener() as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output, 1 << 20)
        temporary.replace(target)

    with archive.open("rb") as probe:
        is_zip = probe.read(2) == b"PK"
    if is_zip:
        with zipfile.ZipFile(archive) as source:
            for entry in source.infolist():
                rel = None if entry.is_dir() else keep(entry.filename)
                if rel:
                    write(rel, lambda entry=entry: source.open(entry))
    else:
        with tarfile.open(archive) as source:
            for member in source:
                rel = keep(member.name) if member.isfile() else None
                if rel:
                    write(rel, lambda member=member: source.extractfile(member))


def stage_refseg_sources(raw_root: Path, staging: Path, needed: set[str],
                         args: argparse.Namespace) -> dict[str, Path]:
    """Stage the SteerViT converter's image roots (Visual Genome, Mapillary, COCO-2014)."""
    roots: dict[str, Path] = {}
    if "visual_genome" in needed:
        vg = staging / "visual_genome"
        if (vg / "VG_100K").is_dir() and (vg / "VG_100K_2").is_dir():
            print(f"Using staged Visual Genome: {vg}")
        else:
            if not args.dry_run:
                vg.mkdir(parents=True, exist_ok=True)
            run(["tar", "-xf", raw(raw_root, "visual_genome", "visual_genome.tar"), "-C", str(vg)],
                None, args.dry_run)
        roots["visual_genome"] = vg
    if "mapillary" in needed:
        mapillary = staging / "mapillary"
        images = mapillary / "mapillary_vistasv2"
        if (images / "training/images").is_dir() and (images / "validation/images").is_dir():
            print(f"Using staged Mapillary images: {mapillary}")
        else:
            def keep(name: str) -> str | None:
                name = name.removeprefix("mapillary_vistasv2/")
                if name.startswith(("training/images/", "validation/images/")):
                    return f"mapillary_vistasv2/{name}"
                return None
            extract_archive_subset(Path(raw(raw_root, "mapillary_vistas", "mapillary.tar")),
                                   mapillary, keep, args.dry_run)
        roots["mapillary"] = mapillary
    if "mscoco" in needed:
        coco = staging / "coco2014"
        # The RefCOCOg download already extracted train2014; link instead of re-extracting.
        refcocog_train = Path(raw(raw_root, "refcocog", "train2014"))
        if refcocog_train.is_dir() and not (coco / "train2014").exists() and not args.dry_run:
            coco.mkdir(parents=True, exist_ok=True)
            (coco / "train2014").symlink_to(refcocog_train)
        for split in ("train2014", "val2014"):
            if (coco / split).is_dir():
                continue
            archive = Path(raw(raw_root, "coco2014", f"{split}.zip"))
            if split == "train2014" and not archive.is_file():
                # The RefCOCOg source may own the same original ZIP directly.
                archive = Path(raw(raw_root, "refcocog", "train2014.zip"))
            if not archive.is_file() and not args.dry_run:
                raise SystemExit(f"ERROR: COCO-2014 {split} images are absent and {archive} is missing "
                                 "(download the coco2014 source first).")
            extract_archive_subset(archive, coco,
                                   lambda name, split=split: name if name.startswith(f"{split}/") else None,
                                   args.dry_run)
        roots["mscoco"] = coco
    return roots


def archive_member(member: Path, archive: Path, name: str, specs: Path, args: argparse.Namespace) -> None:
    run([sys.executable, str(REPO / "scripts/dataset_prep/archive_prepared_member.py"),
         "--member", str(member), "--specs", str(specs), "--name", name,
         "--archive", str(archive), "--remove-member"], None, args.dry_run)


def member_specs(work: Path, name: str, member: Path, args: argparse.Namespace) -> Path:
    specs = work / f"{name}.specs.json"
    if not args.dry_run:
        specs.parent.mkdir(parents=True, exist_ok=True)
        specs.write_text(json.dumps({
            "samples_per_epoch": SAMPLES_PER_EPOCH, "seed": 42, "shuffle": True,
            "datasets": [{"name": name, "path": str(member), "weight": 1.0}],
        }, indent=2) + "\n", encoding="utf-8")
    return specs


def build_refseg_members(raw_root: Path, root: Path, work: Path, needed: list[str],
                         args: argparse.Namespace) -> None:
    """Build and publish the selected per-source refseg members (train/Refseg/)."""
    out_dir = root / "train/Refseg"
    steervit = raw_directory(raw_root, BY_NAME["steervit"])
    pending = [name for name in needed if name in STEERVIT_MEMBERS]
    roots = stage_refseg_sources(raw_root, args.staging_root / "refseg_sources",
                                 {need for name in pending for need in STEERVIT_MEMBERS[name][2]}, args)
    env = {"HF_HOME": str(steervit / "huggingface"),
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    for name in needed:
        member = work / f"refseg_{name}"
        archive = out_dir / f"refseg_{name}.tar"
        if args.overwrite and not args.dry_run:
            unpublish(archive)
        metadata = member / "metadata.jsonl"
        if args.overwrite or not (metadata.is_file() and metadata.stat().st_size):
            if name == "reasonseg":
                command = [sys.executable, str(REPO / "dataset_prep/refseg/prepare_reasonseg.py"),
                           "--reasonseg-root", raw(raw_root, "reasonseg", "reasonseg_train_test"),
                           "--reasonseg-val-dir", raw(raw_root, "reasonseg", "reasonseg_val"),
                           "--out-root", str(member), "--splits", "train",
                           "--expressions-per-image", "1", "--seed", "42"]
                run(command, None, args.dry_run)
            else:
                dataset, extra, needs = STEERVIT_MEMBERS[name]
                command = [sys.executable, str(REPO / "dataset_prep/refseg/prepare_steervit_refseg.py"),
                           "--out-root", str(member),
                           "--split", "train" if dataset else "train,validation",
                           "--min-width", "336", "--min-height", "336", "--seed", "42",
                           "--num-workers", str(args.workers),
                           "--refcocog-umd-dir", raw(raw_root, "refcocog"), *extra]
                if dataset:
                    command += ["--hf-datasets", str(steervit / dataset)]
                for need in needs:
                    command += [f"--{need.replace('_', '-')}", str(roots[need])]
                if args.limit:
                    command += ["--limit", str(args.limit)]
                run(command, env, args.dry_run)
        archive_member(member, archive, f"refseg_{name}",
                       member_specs(work, f"refseg_{name}", member, args), args)


# ---- Manifest ------------------------------------------------------------------

def build_creative_members(raw_root, root, needed, args):
    from contextlib import nullcontext
    from common.creative_publication import staging_root, publish
    from scripts.dataset_prep.prepare_creative_training import archive_relative
    # All requested creative conversions finish before any old member changes.
    with (nullcontext(root / ".creative-dry-run") if args.dry_run else staging_root(root)) as staged:
        pico = staged / archive_relative("pico_banana")
        existing_pico = root / archive_relative("pico_banana")
        build_pico = "pico_banana" in needed or ("identity" in needed and not published(existing_pico))
        order = (["pico_banana"] if build_pico else []) + [name for name in ("identity", "pd12m") if name in needed]
        for name in order:
            source = (pico if build_pico else existing_pico) if name == "identity" else raw(raw_root, name)
            command = [sys.executable, str(REPO / "scripts/dataset_prep/prepare_creative_training.py"),
                       "--dataset", name, "--raw-dir", str(source), "--root", str(staged),
                       "--workers", str(args.workers), "--seed", "42"]
            count = {"identity": 50000, "pd12m": 250000}.get(name)
            if args.limit or count:
                command += ["--limit", str(min(args.limit, count) if args.limit and count else args.limit or count)]
            run(command, None, args.dry_run)
        if not args.dry_run:
            publish(staged, root, [archive_relative(name) for name in needed], args.overwrite)

def tar_entry(name: str, archive: Path, weight: float, manifest_dir: Path) -> dict:
    index = Path(f"{archive}.ovb-index.sqlite")
    if not archive.is_file() or not index.is_file():
        raise SystemExit(f"ERROR: member {name} was not published as {archive} (+ index)")
    # Relative to the manifest, so the prepared version can be moved as a whole.
    return {"name": name, "type": "indexed_prepared_tar", "weight": weight,
            "path": os.path.relpath(archive, manifest_dir),
            "index_path": os.path.relpath(index, manifest_dir)}


def publish_manifest(root: Path, pool: str, seg: list[str], refseg: list[str],
                     singles: list[str], dry_run: bool) -> Path:
    path = root / "manifests" / f"{pool}_pool.json"
    print(f"Writing {path}")
    if dry_run:
        return path
    by_task: dict[str, list[tuple[str, Path]]] = {
        "semseg": [(stem, seg_tar(root, "Semseg", stem)) for name in seg for stem in SEG_SOURCES[name].semseg],
        "instanceseg": [(stem, seg_tar(root, "Instanceseg", stem)) for name in seg
                        for stem in SEG_SOURCES[name].instanceseg],
        "refseg": [(name, root / "train/Refseg" / f"refseg_{name}.tar") for name in refseg],
        **{name: [(name, root / "train" / SINGLES[name][0] / f"{SINGLES[name][1]}.tar")]
           for name in singles},
    }
    for task, members in by_task.items():
        absent = [name for name, archive in members if not published(archive)]
        if absent:
            # Only a deliberately absent member reaches this point (for example
            # hypersim under --allow-missing-hypersim); row shares rebalance.
            print(f"WARNING: omitting unpublished {task} member(s) from the manifest: {', '.join(absent)}")
            by_task[task] = [(name, archive) for name, archive in members if published(archive)]
    by_task = {task: members for task, members in by_task.items() if members}
    # Tasks a pool omits return their mass to the others: every manifest keeps
    # the paper's 0.7 other-task budget so geometry is added identically.
    scale = OTHER_TASK_BUDGET / sum(TASK_MASSES[task] for task in by_task)
    datasets = []
    for task, members in by_task.items():
        rows = {name: rows_in(Path(f"{archive}.ovb-index.sqlite")) for name, archive in members}
        total = sum(rows.values())
        prefix = f"{task}_" if task in ("semseg", "instanceseg") else ""
        for name, archive in members:
            label = f"refseg_{name}" if task == "refseg" else f"{prefix}{name}" if prefix else name
            datasets.append(tar_entry(label, archive, TASK_MASSES[task] * scale * rows[name] / total, path.parent))
    manifest = {"samples_per_epoch": SAMPLES_PER_EPOCH, "seed": 42, "shuffle": True, "datasets": datasets,
                "note": f"{pool} pool: other-task members (mass {OTHER_TASK_BUDGET}); "
                        "the complete official-source setup recipe adds the geometry members."}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


# ---- Main ----------------------------------------------------------------------

def select_pool(args: argparse.Namespace) -> tuple[list[str], list[str], list[str]]:
    if args.pool in POOLS:
        if args.sources:
            raise SystemExit(f"ERROR: --sources only applies to a custom pool, not {args.pool!r}.")
        return POOLS[args.pool]
    if not args.sources:
        raise SystemExit(f"ERROR: unknown pool {args.pool!r}: pass --sources to define a custom pool "
                         "(see --list-sources), or use zeroshot / nonzeroshot.")
    names = [part.strip() for part in args.sources.split(",") if part.strip()]
    unknown = sorted(set(names) - set(ALL_SOURCES))
    if unknown:
        raise SystemExit(f"ERROR: unknown source(s): {', '.join(unknown)}. See --list-sources.")
    return ([n for n in names if ALL_SOURCES[n] == "seg"],
            [n for n in names if ALL_SOURCES[n] == "refseg"],
            [n for n in names if ALL_SOURCES[n] == "single"])


def main() -> int:
    from common.node_scratch import workspace
    with workspace("training-pool") as scratch:
        return build(scratch)


def build(scratch) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", help="zeroshot, nonzeroshot, or any name for a custom pool (with --sources).")
    parser.add_argument("--sources", help="Comma-separated member sources for a custom pool (--list-sources).")
    parser.add_argument("--list-sources", action="store_true")
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--root", type=Path, help="prepared/<version>")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--staging-root", type=Path, default=None,
                        help="Large local scratch for extracted sources (default: $TMPDIR/open_visionbanana_data).")
    parser.add_argument("--allow-missing-hypersim", action="store_true")
    parser.add_argument("--semseg-num-images", type=int, default=0, help="Cap semseg pairs (smoke tests).")
    parser.add_argument("--limit", type=int, default=0, help="Cap Pico/refseg/PD12M/identity rows (smoke tests).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.list_sources:
        list_sources()
        return 0
    if not (args.pool and args.raw_root and args.root):
        parser.error("--pool, --raw-root and --root are required (or --list-sources).")
    args.pool = args.pool.removesuffix("_pool")
    seg, refseg, singles = select_pool(args)

    raw_root = args.raw_root.expanduser().resolve()
    root = args.root.expanduser().resolve()
    work = scratch / "work/train" / f"{args.pool}_pool"
    if args.staging_root is not None and not str(args.staging_root).strip(" ."):
        raise SystemExit("ERROR: --staging-root is empty (an unset shell variable?); "
                         "pass a scratch directory with room for the extracted archives.")
    staging_identity = hashlib.sha256(f"{raw_root}\n{root}\n{args.pool}".encode()).hexdigest()[:12]
    args.staging_root = (args.staging_root or scratch / "sources" / staging_identity).expanduser()
    manifest = root / "manifests" / f"{args.pool}_pool.json"
    if manifest.is_file() and not args.overwrite:
        print(f"{args.pool} pool already built: {manifest} (--overwrite rebuilds)")
        return 0

    # Members already published (by any pool) are reused; only the rest is built.
    needed_seg = [name for name in seg
                  if args.overwrite or not all(published(archive) for _, archive in seg_members(root, name))]
    if ("hypersim" in needed_seg and args.allow_missing_hypersim
            and not any(raw_directory(raw_root, BY_NAME["hypersim"]).glob("**/scene_cam_*_final_hdf5"))
            and not any((raw_directory(raw_root, BY_NAME["hypersim"]) / "archives").glob("*.zip"))):
        # Without raw scenes this member can never be built, so it must not keep
        # a stage run alive on its own (--allow-missing-hypersim acknowledged it).
        print("Skipping hypersim member: no raw scenes (--allow-missing-hypersim).")
        needed_seg.remove("hypersim")
    needed_refseg = [name for name in refseg
                     if args.overwrite or not published(root / "train/Refseg" / f"refseg_{name}.tar")]
    needed_singles = [name for name in singles
                      if args.overwrite or not published(root / "train" / SINGLES[name][0] / f"{SINGLES[name][1]}.tar")]
    for done, label in ((set(seg) - set(needed_seg), "segmentation sources"),
                        (set(refseg) - set(needed_refseg), "refseg members"),
                        (set(singles) - set(needed_singles), "members")):
        if done:
            print(f"Reusing published {label}: {', '.join(sorted(done))}")

    creative = [name for name in needed_singles if name in {"pico_banana", "identity", "pd12m"}]
    if creative:
        build_creative_members(raw_root, root, creative, args)
        needed_singles = [name for name in needed_singles if name not in creative]

    if args.overwrite and not args.dry_run:
        for name in needed_seg:
            for _, archive in seg_members(root, name):
                unpublish(archive)
        for name in needed_singles:
            unpublish(root / "train" / SINGLES[name][0] / f"{SINGLES[name][1]}.tar")

    if needed_seg or needed_singles:
        env = stage_env(raw_root, work, needed_seg, needed_singles, args)
        if args.overwrite and not args.dry_run:
            env["FORCE"] = "1"
        if not args.dry_run:
            work.mkdir(parents=True, exist_ok=True)
            for name in needed_singles:
                (root / "train" / SINGLES[name][0]).mkdir(parents=True, exist_ok=True)
        if "hypersim" in needed_seg:
            hypersim_segmentation(raw_root, Path(env["HYPERSIM_SEG_TAR"]), args)
        run(["bash", str(REPO / "scripts/dataset_prep/stage_and_prepare_open_visionbanana_pool.sh")],
            env, args.dry_run)
        for task in ("Semseg", "Instanceseg"):
            split_and_publish(root, work, task, args)
        for name in needed_singles:
            task, stem, _ = SINGLES[name]
            archive_member(work / stem, root / "train" / task / f"{stem}.tar", name,
                           member_specs(work, name, work / stem, args), args)
    if needed_refseg:
        build_refseg_members(raw_root, root, work, needed_refseg, args)

    print(f"Pool manifest: {publish_manifest(root, args.pool, seg, refseg, singles, args.dry_run)}")
    # The workspace context removes only this invocation's scratch. Never remove
    # the shared prepared root/work tree, which may belong to another stage.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
