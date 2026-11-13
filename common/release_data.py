"""Dataset catalog and portable helpers for the public release CLIs.

This module describes *official raw sources* only.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Step:
    """One acquisition script run with environment overrides.

    Values may use {dir} (the dataset's raw directory), {raw} (the raw root),
    {repo} (this checkout) and {python} (the running interpreter).
    """
    script: str
    env: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Dataset:
    name: str
    raw_group: str
    tasks: tuple[str, ...]
    access: str
    source: str
    files: tuple[tuple[str, str], ...] = ()
    preparer: str | None = None
    note: str = ""
    # Relative paths or glob patterns used only to check whether an official
    # source has been staged.  Individual converters remain authoritative for
    # validating the complete raw layout.
    expected: tuple[str, ...] = ()
    # Scripted acquisition (download, validate and pack). A gated source's script
    # checks the files the user placed in {dir} and prints instructions otherwise.
    steps: tuple[Step, ...] = ()
    # Training pools only: the raw sources the pool is built from.
    sources: tuple[str, ...] = ()
    # Extra inputs that supplement this source, without replacing it.
    dependencies: tuple[str, ...] = ()


DL = "scripts/dataset_prep/data_download_scripts/"


def _step(script: str, **env: str) -> Step:
    return Step(script if "/" in script else DL + script, tuple(env.items()))


def _archive(script: str, **env: str) -> Step:
    """A packer that writes its archive into {dir} and keeps its work files there."""
    return _step(script, ARCHIVE_DIR="{dir}", **env)


CATALOG: tuple[Dataset, ...] = (
    # ---- Segmentation sources --------------------------------------------------
    # Shared_seg feeds more than one segmentation task; Semseg/Instanceseg/Refseg
    # feed one. Each scripted source ends as the archive(s) named in `expected`.
    Dataset("ade20k", "Shared_seg", ("Semseg", "Instanceseg"), "automatic", "MIT CSAIL ADE20K",
            note="SceneParsing ADEChallengeData2016 (semantic) and ChallengeData2017 (instance) releases.",
            expected=("ade20k.tar", "ade20k_instanceseg.tar", "instanceInfo100_train.txt"),
            steps=(_step("download_datasets_semseg_ade_voc.sh", ARCHIVE_DIR="{dir}", WORK="{dir}/_download", SKIP_VOC="1"),
                   _step("download_datasets_ade20k_instance.sh", ARCHIVE_DIR="{dir}", ADE_DIR="{dir}"))),
    Dataset("voc2012", "Shared_seg", ("Semseg", "Instanceseg"), "automatic", "PASCAL VOC 2012 + PASCAL-Context",
            note="voc.tar holds the semantic labels and the PASCAL-Context 59-class labels; VOC2012.tar the instance release.",
            expected=("voc.tar", "VOC2012.tar"),
            steps=(_step("download_datasets_semseg_ade_voc.sh", ARCHIVE_DIR="{dir}", WORK="{dir}/_download", SKIP_ADE="1"),
                   _step("download_datasets_pascal_context.sh", ARCHIVE_DIR="{dir}", VOC_TAR="{dir}/voc.tar", WORK="{dir}/_pascal_context"),
                   _step("download_datasets_voc2012_instance.sh", ARCHIVE_DIR="{dir}"))),
    Dataset("coco2017", "Shared_seg", ("Semseg", "Instanceseg", "Refseg"), "automatic",
            "COCO", (
                ("http://images.cocodataset.org/zips/train2017.zip", "train2017.zip"),
                ("http://images.cocodataset.org/zips/val2017.zip", "val2017.zip"),
                ("http://images.cocodataset.org/zips/unlabeled2017.zip", "unlabeled2017.zip"),
                ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "annotations_trainval2017.zip"),
            ), note="Official zips; COCONut-XL and COCO-Object-80 reuse them."),
    Dataset("coco2014", "Shared_seg", ("Refseg",), "automatic",
            "COCO", (
                ("http://images.cocodataset.org/zips/train2014.zip", "train2014.zip"),
                ("http://images.cocodataset.org/zips/val2014.zip", "val2014.zip"),
                ("http://images.cocodataset.org/annotations/annotations_trainval2014.zip", "annotations_trainval2014.zip"),
            ), note="Official zips. train2014 holds the RefCOCO-family photos and val2014 the LVIS refseg photos."),
    Dataset("coconut_xl", "Shared_seg", ("Semseg", "Instanceseg"), "automatic", "COCONut-XL",
            note="Very large: COCO 2017 + Objects365 photos with COCONut panoptic masks.",
            expected=("coco/annotations/panoptic_train2017.json",),
            steps=(_step("download_datasets_coconut_xlarge.sh", ARCHIVE_DIR="{raw}/Shared_seg", TARGET_DIR="{dir}",
                         COCO2017_ARCHIVE_DIR="{raw}/Shared_seg/coco2017"),)),
    Dataset("mapillary_vistas", "Shared_seg", ("Semseg", "Instanceseg", "Refseg"), "manual", "Mapillary Vistas v2.0",
            note="Accept the license at https://www.mapillary.com/dataset/vistas and place the official zip in this folder.",
            expected=("mapillary.tar",),
            steps=(_step("download_datasets_mapillary.sh", SHARED_SEG_ROOT="{dir}", MAPILLARY_DIR="{dir}", PYTHON="{python}"),)),
    Dataset("acdc", "Shared_seg", ("Semseg", "Instanceseg"), "manual", "ACDC",
            note="Log in at https://acdc.vision.ee.ethz.ch/download and place rgb_anon_trainvaltest.zip, gt_trainval.zip and gt_panoptic_trainval.zip in this folder.",
            expected=("acdc.tar",),
            steps=(_archive("download_datasets_acdc.sh", ACDC_DIR="{dir}"),)),
    Dataset("goose", "Shared_seg", ("Semseg", "Instanceseg"), "automatic", "GOOSE 2D",
            expected=("goose.tar",), steps=(_archive("download_datasets_goose.sh", WORK="{dir}/_download"),)),
    Dataset("muses", "Shared_seg", ("Semseg", "Instanceseg"), "automatic", "MUSES",
            expected=("muses.tar",), steps=(_archive("download_datasets_muses.sh", WORK="{dir}/_download"),)),
    Dataset("vipseg", "Shared_seg", ("Semseg", "Instanceseg"), "automatic", "VIPSeg",
            note="Fetched from Google Drive with gdown.",
            expected=("vipseg.tar",), steps=(_archive("download_datasets_vipseg.sh", WORK="{dir}/_download"),)),
    Dataset("waymo", "Shared_seg", ("Semseg", "Instanceseg"), "manual", "Waymo Open camera panoptic",
            note="Accept the terms at https://waymo.com/open/terms and authenticate with `gcloud auth login`; the script then downloads.",
            expected=("waymo.tar",),
            steps=(_archive("download_datasets_waymo.sh", WORK="{dir}/_download", WAYMO_BUILD_ROOT="{dir}/_build"),)),
    Dataset("idd", "Shared_seg", ("Semseg", "Instanceseg"), "manual", "IDD Segmentation",
            note="Log in at https://idd.insaan.iiit.ac.in/ and place idd-segmentation.tar.gz and idd-20k-II.tar.gz in this folder.",
            expected=("idd_1_2.tar",), steps=(_archive("download_datasets_idd.sh", IDD_DIR="{dir}"),)),
    Dataset("nuimages", "Shared_seg", ("Semseg", "Instanceseg"), "manual", "nuImages",
            note="Log in at https://www.nuscenes.org/nuimages and place nuimages-v1.0-all-metadata.tgz and nuimages-v1.0-all-samples.tgz in this folder.",
            expected=("nuimages.tar",), steps=(_archive("download_datasets_nuimages.sh", NUIMAGES_DIR="{dir}"),)),
    Dataset("wilddash2", "Shared_seg", ("Semseg", "Instanceseg"), "manual", "WildDash 2",
            note="Log in at https://www.wilddash.cc/download and place wd_public_v2p.zip in this folder.",
            expected=("wd_public_v2p.tar",),
            steps=(_archive("download_datasets_wilddash2.sh", WILDDASH2_DIR="{dir}", PYTHON="{python}"),)),
    Dataset("cityscapes", "Semseg", ("Semseg",), "manual", "Cityscapes",
            note="Log in at https://www.cityscapes-dataset.com/downloads/ and place gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip in this folder (needs pip install cityscapesscripts).",
            expected=("cityscapes.tar",),
            steps=(_step("scripts/dataset_prep/acquire_cityscapes_packed.py", CITYSCAPES_DIR="{dir}"),)),
    Dataset("bdd100k", "Semseg", ("Semseg",), "manual", "BDD100K",
            note="Log in at http://bdd-data.berkeley.edu/ and place the 10K images and semantic-segmentation label zips in this folder.",
            expected=("bdd_10k.tar",), steps=(_archive("download_datasets_bdd.sh", BDD_DIR="{dir}"),)),
    Dataset("foodseg103", "Semseg", ("Semseg",), "automatic", "FoodSeg103",
            expected=("foodseg103.tar",), steps=(_archive("download_datasets_foodseg103.sh", WORK="{dir}/_download"),)),
    Dataset("trans10k", "Semseg", ("Semseg",), "automatic", "Trans10K-v2",
            expected=("trans10k.tar",), steps=(_archive("download_datasets_trans10k.sh", WORK="{dir}/_download"),)),
    Dataset("railsem19", "Semseg", ("Semseg",), "manual", "RailSem19",
            note="Register at https://www.wilddash.cc/railsem19 and place rs19_val.zip in this folder.",
            expected=("railsem19.tar",), steps=(_archive("download_datasets_railsem19.sh", WORK="{dir}/_download"),)),
    Dataset("coco_object_80", "Semseg", ("Semseg",), "automatic", "COCO-Object-80 (COCO 2017 val + COCO-Stuff)",
            note="Evaluation only; reuses the COCO 2017 val zip.",
            expected=("coco.tar",),
            steps=(_archive("download_datasets_coco_object80.sh", COCO2017_ARCHIVE_DIR="{raw}/Shared_seg/coco2017"),)),
    Dataset("saco_gold", "Instanceseg", ("Instanceseg",), "manual", "SA-Co/Gold",
            note="Gated by Meta (license form). Download the annotation JSONs into gt-annotations/ from "
                 "https://huggingface.co/datasets/facebook/SACo-Gold, the MetaCLIP images into metaclip-images/ from "
                 "https://universe.roboflow.com/sa-co-gold, and the SA-1B images into sa1b-images/ via the sa_co_gold.tar "
                 "link at https://ai.meta.com/datasets/segment-anything-downloads/ (details: "
                 "https://github.com/facebookresearch/sam3/blob/main/scripts/eval/gold/README.md).",
            expected=("sa1b-images", "metaclip-images", "gt-annotations")),
    Dataset("paco_lvis", "Instanceseg", ("Instanceseg",), "automatic", "PACO-LVIS (DatasetNinja export)",
            expected=("paco-lvis-DatasetNinja.tar",), steps=(_archive("download_datasets_paco_lvis.sh"),)),
    Dataset("ovis", "Instanceseg", ("Instanceseg",), "automatic", "OVIS",
            note="Fetched from Google Drive with gdown.",
            expected=("ovis.tar",), steps=(_archive("download_datasets_ovis.sh", WORK="{dir}/_download"),)),
    Dataset("refcocog", "Refseg", ("Refseg",), "automatic", "RefCOCOg (UMD)",
            note="Original refer annotations plus COCO train2014 photos kept in the shared ZIP; legacy loose images remain supported.",
            expected=("**/refs(umd).p", "**/instances.json", "train2014|train2014.zip"),
            steps=(Step("scripts/dataset_prep/acquire_refcocog_packed.py",
                        (("OVB_REFCOCOG_RAW", "{dir}"),
                         ("TRAIN2014_ZIP", "{raw}/Shared_seg/coco2014/train2014.zip"))),)),
    Dataset("reasonseg", "Refseg", ("Refseg",), "manual", "ReasonSeg (LISA)",
            note="Download train.zip, val.zip and test.zip from https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy; put train.zip and test.zip in reasonseg_train_test/ and val.zip in reasonseg_val/ below this folder.",
            expected=("reasonseg_val/*.json", "reasonseg_train_test/train.tar"),
            steps=(_archive("download_datasets_reasonseg.sh"),)),
    Dataset("steervit", "Refseg", ("Refseg",), "automatic", "SteerViT (+ gRefCOCO, PhraseCut)",
            note="Hugging Face datasets; the zero-shot pool keeps its Visual Genome and Mapillary rows.",
            expected=("huggingface/open_visionbanana_steervit_cache.json", "SteerViT-gRefCOCO/data", "SteerViT-PhraseCutDataset/data"),
            steps=(_step("scripts/dataset_prep/download_steervit_metadata.sh", HF_HOME="{dir}/huggingface", PYTHON="{python}"),
                   _step("download_datasets_refseg_train_sources.sh", OUT="{dir}", STEERVIT_DIR="{dir}", SKIP_VISUAL_GENOME="1",
                         MAPILLARY_TAR="{raw}/Shared_seg/mapillary_vistas/mapillary.tar"))),
    Dataset("visual_genome", "Refseg", ("Refseg",), "automatic", "Visual Genome images",
            note="Photos for the SteerViT Visual Genome and PhraseCut rows.",
            expected=("visual_genome.tar",),
            steps=(_step("download_datasets_refseg_train_sources.sh", OUT="{dir}", SKIP_STEERVIT_EXTRA="1",
                         MAPILLARY_TAR="{raw}/Shared_seg/mapillary_vistas/mapillary.tar"),)),
    # ---- Image editing / generation -------------------------------------------
    Dataset("pico_banana", "ImgEdit", ("ImgEdit",), "automatic", "Pico-Banana-400K",
            note="Checkout, OpenImages sources and ~258k edited targets (~400 GB, parallel download).",
            expected=("openimages/sft_with_local_source_image_path.jsonl", "openimages/images/positive-edit"),
            steps=(_step("pico_banana_400k.sh", ROOT="{dir}", PYTHON="{python}"),)),
    Dataset("imgedit_bench", "ImgEdit", ("ImgEdit",), "automatic", "ImgEdit-Bench",
            note="Benchmark tar (extracted to Benchmark/) and the ImgEdit judge model.",
            expected=("Benchmark/singleturn", "ImgEdit_Judge"),
            steps=(_step("download_imgeditbench.sh", ROOT="{dir}", OUT_DIR="{dir}", EXTRACT_BENCHMARK="1", PYTHON="{python}"),)),
    Dataset("genai_bench", "ImgGeneration", ("ImgGeneration",), "automatic", "GenAI-Bench",
            (("https://huggingface.co/datasets/BaiqiL/GenAI-Bench-1600/resolve/main/genai_image.json", "genai_image.json"),
             ("https://huggingface.co/datasets/BaiqiL/GenAI-Bench-1600/resolve/main/genai_skills.json", "genai_skills.json")),
            note="Official GenAI-Bench-1600 prompt and skill files."),
    Dataset("pd12m", "ImgGeneration", ("ImgGeneration",), "automatic", "PD12M (500k selection)",
            note="WebDataset shards of the selected PD12M rows.",
            expected=("shards.jsonl",),
            steps=(_step("scripts/dataset_prep/download_pd12m.sh", RAW_ROOT="{raw}", PD12M_RAW_DIR="{dir}"),)),
    Dataset("nyuv2", "Shared_geo", ("Depth", "Normals"), "automatic", "NYU Depth V2",
            (("https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat", "nyu_depth_v2_labeled.mat"),),
            note="The labeled MAT download is automatic. DSINE normal assets/official splits must be added manually when normals are required."),
    Dataset("diode", "Shared_geo", ("Depth", "Normals"), "manual", "DIODE", preparer="diode",
            note="Download the official DIODE validation RGB, depth, normal, mask, and data-list archives from https://diode-dataset.org/; extract them so data_list/val_indoors.csv and data_list/val_outdoor.csv resolve below this directory.",
            expected=("data_list/val_indoors.csv", "data_list/val_outdoor.csv")),
    Dataset("eth3d", "Shared_geo", ("Depth", "Normals"), "manual", "ETH3D", preparer="eth3d",
            note="From https://www.eth3d.net/datasets download the low-resolution multi-view training images and low-resolution multi-view training ground truth. Keep two_view_training.7z and two_view_training_gt.7z here, or extract both trees here.",
            expected=("two_view_training.7z|two_view_training", "two_view_training_gt.7z|two_view_training_gt")),
    Dataset("ibims1", "Shared_geo", ("Depth", "Normals"), "manual", "iBims-1", preparer="ibims1",
            note="Request iBims-1 from https://www.ibims1.org/ and place ibims1_core_mat.zip and imagelist.txt here. Optional augmented evaluation additionally uses ibims1_aux_augmented.zip.",
            expected=("ibims1_core_mat.zip", "imagelist.txt")),
    Dataset("kitti", "Shared_geo", ("Depth", "Normals"), "manual", "KITTI depth completion", preparer="kitti",
            note="Download the official KITTI depth-selection validation archive from https://www.cvlibs.net/datasets/kitti/eval_depth.php and place depth_selection.zip here.",
            expected=("depth_selection.zip",)),
    Dataset("kitti_eigen", "Shared_geo", ("Depth",), "manual", "KITTI Eigen split",
            note="Download official KITTI raw drives, calibration and Eigen split metadata. A release converter remains TODO."),
    Dataset("nuscenes", "Shared_geo", ("Depth", "Normals"), "manual", "nuScenes", preparer="nuscenes",
            note="Accept the terms at https://www.nuscenes.org/nuscenes#download and extract the official trainval metadata, samples and sweeps here.",
            expected=("v1.0-trainval", "samples", "sweeps")),
    Dataset("vkitti2", "Shared_geo", ("Depth", "Normals"), "manual", "Virtual KITTI 2", preparer="vkitti2",
            note="Download the official RGB, depth, and text-GT archives from https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/ and keep the three archives here.",
            expected=("*rgb*.tar*", "*depth*.tar*", "*textgt*.tar*")),
    Dataset("scannet", "Shared_geo", ("Depth", "Normals"), "manual", "ScanNet v2", preparer="scannet",
            note="Follow https://github.com/ScanNet/ScanNet#scannet-data, accept the terms, use the official downloader, and place scannet_frames_25k.zip plus scannetv2_val.txt here.",
            expected=("**/scannet_frames_25k.zip|**/scannet_frames_25k", "**/scannetv2_val.txt")),
    Dataset("hypersim", "Shared_geo", ("Depth", "Normals"), "manual", "Hypersim", preparer="hypersim",
            note="Follow the official download instructions at https://github.com/apple/ml-hypersim and extract scene_* trees here. Each selected camera requires scene_cam_*_final_hdf5 color files and matching scene_cam_*_geometry_hdf5 depth_meters and normal_cam files.",
            expected=("**/scene_cam_*_final_hdf5/frame.*.color.hdf5",)),
    Dataset("matterport3d", "Shared_geo", ("Depth", "Normals"), "manual", "Matterport3D", preparer="matterport3d",
            note="Request access at https://niessner.github.io/Matterport/, accept the terms and use its official download script. Keep one <scan_id>.tar per scan in this directory; each tar must contain the image, depth, normal, pose, and calibration assets consumed by the converter.",
            expected=("*.tar",)),
    Dataset("replica", "Shared_geo", ("Depth", "Normals"), "manual", "Replica", preparer="replica",
            note="Accept the license and download official scenes from https://github.com/facebookresearch/Replica-Dataset. Preparation additionally requires Habitat-Sim; the corrected Ptex training package is recommended instead of legacy vertex-colour rendering."),
    Dataset("synscapes", "Shared_geo", ("Depth", "Normals", "Semseg", "Instanceseg"), "automatic", "Synscapes",
            note="Public release (~337 GB, non-commercial license https://synscapes.on.liu.se/download.html), extracted to Synscapes/. The unified depth/normals converter remains TODO.",
            expected=("Synscapes/img/rgb", "Synscapes/meta"),
            steps=(_step("download_datasets_synscapes.sh", ARCHIVE_DIR="{dir}", TARGET_DIR="{dir}", CHUNKS_DIR="{dir}/chunks"),)),
    Dataset("taskonomy", "Shared_geo", ("Depth", "Normals"), "manual", "Taskonomy", preparer="taskonomy",
            note="Use the official tools at https://github.com/StanfordVL/taskonomy/tree/master/data and place matched <building>_rgb.tar, <building>_depth_zbuffer.tar, and <building>_normal.tar archives here. The same building identifier must exist for all three modalities.",
            expected=("*_rgb.tar", "*_depth_zbuffer.tar", "*_normal.tar")),
    # ---- Training pools (built by `--splits train`) -----------------------------
    Dataset("zeroshot_pool", "", ("Semseg", "Instanceseg", "Refseg", "ImgEdit", "ImgGeneration"), "derived",
            "Zero-shot training pool (v3)",
            note="Segmentation, editing, identity and generation members of the zero-shot pool; add geometry with the published package (--pool zeroshot). "
                 "refcocog only supplies the evaluation-image exclusion list; its rows stay out.",
            sources=("ade20k", "voc2012", "acdc", "bdd100k", "mapillary_vistas", "goose", "muses", "vipseg", "waymo",
                     "foodseg103", "trans10k", "railsem19", "ovis", "hypersim", "steervit", "visual_genome",
                     "pico_banana", "pd12m", "refcocog")),
    Dataset("nonzeroshot_pool", "", ("Semseg", "Instanceseg", "Refseg", "ImgEdit", "ImgGeneration"), "derived",
            "Non-zero-shot training pool (v4)",
            note="Adds the benchmark-domain sources to the zero-shot sources; add geometry with the published package (--pool base).",
            sources=("ade20k", "voc2012", "acdc", "bdd100k", "mapillary_vistas", "goose", "muses", "vipseg", "waymo",
                     "foodseg103", "trans10k", "railsem19", "ovis", "hypersim", "steervit", "visual_genome",
                     "pico_banana", "pd12m", "cityscapes", "coco2014", "coco2017", "coconut_xl", "paco_lvis",
                     "synscapes", "idd", "nuimages", "wilddash2", "refcocog", "reasonseg")),
)

from common.official_sources import extend_catalog
CATALOG = extend_catalog(CATALOG)
BY_NAME = {item.name: item for item in CATALOG}

# How the paper used each source. Training pools: ZS = zero-shot, NZS =
# non-zero-shot. Benchmarks are the evaluations reported in EVALUATION.md.
PAPER_TRAINING_POOLS = {
    **dict.fromkeys(BY_NAME["zeroshot_pool"].sources, "ZS+NZS"),
    **dict.fromkeys(set(BY_NAME["nonzeroshot_pool"].sources) - set(BY_NAME["zeroshot_pool"].sources), "NZS"),
    # RefCOCOg rows train only the NZS pool; the ZS build reads just its holdout list.
    "refcocog": "NZS",
    "matterport3d": "ZS+NZS", "replica": "ZS+NZS", "synscapes": "ZS+NZS", "taskonomy": "ZS+NZS",
    "nyuv2": "NZS", "diode": "NZS", "scannet": "NZS", "vkitti2": "NZS",
    "zeroshot_pool": "ZS", "nonzeroshot_pool": "NZS",
}
PAPER_BENCHMARKS = {
    "cityscapes", "coco_object_80", "saco_gold", "refcocog", "reasonseg", "imgedit_bench",
    "genai_bench", "nyuv2", "kitti_eigen", "eth3d", "scannet", "diode", "ibims1", "vkitti2",
}


def list_header() -> str:
    return ("train: paper training pools (ZS = zero-shot, NZS = non-zero-shot); eval: paper benchmark; "
            "converter: splits scripts/prepare_data.py builds (pool = prepared inside a training pool)")


def list_row(item: Dataset, train_implemented: set[str], eval_implemented: set[str]) -> str:
    """One --list line: paper usage first, then unified-converter support."""
    splits = [split for split, names in (("train", train_implemented), ("eval", eval_implemented))
              if item.name in names]
    if any(item.name in BY_NAME[name].sources for name in train_implemented if name in BY_NAME):
        splits.insert(0, "pool")
    pools = PAPER_TRAINING_POOLS.get(item.name, "-")
    benchmark = "yes" if item.name in PAPER_BENCHMARKS else "-"
    return (f"{item.name:18} {item.access:10} train={pools:7} eval={benchmark:4} "
            f"converter={','.join(splits) or 'TODO'}")
ALIASES = {
    "nyu": "nyuv2",
    "bdd": "bdd100k",
    "mapillary": "mapillary_vistas",
    "saco": "saco_gold",
    "coconut": "coconut_xl",
    "pico": "pico_banana",
    "imgedit": "imgedit_bench",
    "genai": "genai_bench",
    "vg": "visual_genome",
    "zeroshot": "zeroshot_pool",
    "nonzeroshot": "nonzeroshot_pool",
}


def parse_selection(value: str) -> list[Dataset]:
    names = [part.strip().lower().replace("-", "_") for part in value.split(",") if part.strip()]
    if not names or names == ["all"]:
        return list(CATALOG)
    names = [ALIASES.get(name, name) for name in names]
    unknown = sorted(set(names) - BY_NAME.keys())
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(unknown)}. Use --list.")
    return [BY_NAME[name] for name in dict.fromkeys(names)]


def raw_directory(root: Path, item: Dataset) -> Path:
    return root / item.raw_group / item.name


# Written into a scripted source's raw directory once all its steps succeeded.
DOWNLOAD_MARKER = ".ovb-download-complete"


def acquisition_items(selected: Iterable[Dataset]) -> list[Dataset]:
    """Expand pool sources and additive dependencies in dependency order."""
    result: dict[str, Dataset] = {}
    visiting = set()
    def visit(item):
        if item.name in visiting:
            raise ValueError(f"Cyclic acquisition dependency: {item.name}")
        if item.name in result:
            return
        visiting.add(item.name)
        for name in item.dependencies + item.sources:
            if name not in BY_NAME:
                raise ValueError(f"Unknown acquisition dependency: {item.name} -> {name}")
            visit(BY_NAME[name])
        if not item.sources:
            result[item.name] = item
        visiting.remove(item.name)
    for item in selected:
        visit(item)
    return list(result.values())


def available_acquisition_items(selected, missing_names):
    """Exclude missing gated sources and sources that depend on them."""
    return [item for item in selected
            if not any(source.name in missing_names for source in acquisition_items([item]))]


def step_command(item: Dataset, step: Step, root: Path) -> tuple[list[str], dict[str, str]]:
    """Command and environment overrides for one acquisition step of `item`."""
    repo = Path(__file__).resolve().parents[1]
    values = {"dir": str(raw_directory(root, item)), "raw": str(root), "repo": str(repo), "python": sys.executable}
    env = {key: value.format(**values) for key, value in step.env}
    script = repo / step.script
    command = [sys.executable, str(script)] if script.suffix == ".py" else ["bash", str(script)]
    return command, env


def staged_source_status(root: Path, item: Dataset) -> tuple[bool, tuple[str, ...]]:
    """Return a lightweight staging status for one official raw source.

    This deliberately does not replace converter validation. It catches absent
    downloads and known required top-level assets before a preparation job is
    submitted.
    """
    if item.sources:
        missing = tuple(f"{name}: {', '.join(details)}" for name in item.sources
                        for ready, details in [staged_source_status(root, BY_NAME[name])] if not ready)
        return not missing, missing
    directory = raw_directory(root, item)
    expected = item.expected or tuple(filename for _, filename in item.files)
    if not directory.is_dir():
        return False, (f"missing directory: {directory}",)
    official_geometry = any(step.script.endswith("acquire_geometry_sources.py") for step in item.steps)
    if item.steps and not official_geometry and not (directory / DOWNLOAD_MARKER).is_file():
        # Scripted sources count as staged only after every step has succeeded.
        missing = tuple(pattern for pattern in expected if not any(directory.glob(pattern.split("|")[0])))
        return False, missing or (f"acquisition not completed ({DOWNLOAD_MARKER} missing)",)
    if expected:
        def present(pattern: str) -> bool:
            # A vertical bar denotes alternative accepted layouts, for example
            # an official archive or its extracted directory.
            alternatives = (part.strip() for part in pattern.split("|"))
            return any(
                path.is_dir() or (path.is_file() and path.stat().st_size > 0)
                for alternative in alternatives
                for path in directory.glob(alternative)
            )

        missing = tuple(pattern for pattern in expected if not present(pattern))
        return not missing, missing
    try:
        next(directory.iterdir())
    except StopIteration:
        return False, ("directory is empty",)
    return True, ()


def download_with_resume(url: str, destination: Path, *, attempts=3,
                         initial_delay=1., maximum_delay=60.) -> None:
    """Retry temporary network failures while preserving partial file bytes."""
    from common.dataset_source_policy import validate_dataset_source
    from common.http_retry import transient_download_error, retry_delay
    validate_dataset_source(url)
    if type(attempts) is not int or attempts < 1 or not 0 <= initial_delay <= maximum_delay:
        raise ValueError("Download attempts must be positive and retry delays nonnegative and ordered")
    for attempt in range(1, attempts + 1):
        try:
            _download_once(url, destination)
            return
        except Exception as error:
            if attempt == attempts or not transient_download_error(error):
                raise
            delay = retry_delay(error, attempt, initial_delay, maximum_delay)
            if isinstance(error, urllib.error.HTTPError):
                error.close()
            print(f"RETRY {destination.name}: temporary network failure; "
                  f"attempt {attempt + 1}/{attempts} in {delay:g}s", file=sys.stderr)
            time.sleep(delay)


def _download_once(url: str, destination: Path) -> None:
    from common.dataset_source_policy import validate_dataset_source
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    start = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "open-visionbanana-release/1"})
    if start:
        request.add_header("Range", f"bytes={start}-")
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - catalog URLs are explicit.
        validate_dataset_source(getattr(response, "url", url))
        append = start and getattr(response, "status", 200) == 206
        if append and not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
            raise ValueError(f"Invalid resume response for {url}")
        expected = response.headers.get("Content-Length")
        before = start if append else 0
        with partial.open("ab" if append else "wb") as output:
            shutil.copyfileobj(response, output, length=8 << 20)
        if expected is not None and partial.stat().st_size != before + int(expected):
            raise ConnectionError(f"Truncated download: {destination.name}")
    partial.replace(destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_tar(archive: Path, destination: Path) -> Path:
    """Extract an evaluation archive after rejecting traversal and links."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:*") as handle:
        members = handle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Unsafe path in {archive}: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Links are not permitted in prepared release archives: {member.name}")
        handle.extractall(destination, members=members)  # noqa: S202 - checked above.
    return destination


def run_python(script: Path, arguments: Iterable[str], dry_run: bool) -> None:
    command = [sys.executable, str(script), *map(str, arguments)]
    print("+", " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def write_manifest(root: Path, records: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "preparation_manifest.json"
    path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_archive(source_dir: Path, archive: Path, dry_run: bool) -> None:
    """Create a deterministic-enough portable tar when a converter emits a tree."""
    if dry_run:
        print(f"+ tar -cf {archive} -C {source_dir.parent} {source_dir.name}")
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".partial")
    with tarfile.open(temporary, "w") as handle:
        handle.add(source_dir, arcname=source_dir.name, recursive=True)
    temporary.replace(archive)
