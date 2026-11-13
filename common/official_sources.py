"""Official geometry sources shared by acquisition and preparation.

Author-distributed benchmark assets retain their upstream splits and targets.
Research uploads by OVB are deliberately absent from this catalog.
"""
from dataclasses import replace

MARIGOLD_COMMIT = "18466672fb8661152434a0efe1184939164cda19"
MARIGOLD_RAW = f"https://raw.githubusercontent.com/huawei-bayerlab/marigold-v2/{MARIGOLD_COMMIT}"
DEPTH_BASE = "https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset"
NORMAL_BASE = "https://share.phys.ethz.ch/~pf/bingkedata/marigold/marigold_normals"
DEPTH_FILES = {
    "nyuv2": ("nyu_labeled_extracted.tar", "nyu_depth/labeled/filename_list_test.txt"),
    "kitti": ("kitti_eigen_split_test.tar", "kitti_depth/eigen_test_files_with_gt.txt"),
    "eth3d": ("eth3d.tar", "eth3d_depth/eth3d_filename_list.txt"),
    "scannet": ("scannet_val_sampled_800_1.tar", "scannet_depth/scannet_val_sampled_list_800_1.txt"),
    "diode": ("diode_val.tar", "diode_depth/diode_val_all_filename_list.txt"),
}
NORMAL_LISTS = {
    "nyu": "nyu_normals/nyuv2_test.txt",
    "scannet": "scannet_normals/scannet_test.txt",
    "ibims1": "ibims_normals/ibims_test.txt",
    "sintel": "sintel_normals/sintel_filtered.txt",
}
SCANNET_SPLITS = ("scannet_depth/scannet_val_sampled_list_800_1.txt", "scannet_normals/scannet_test.txt")


def scannet_exclusions(root):
    """Validate the pinned benchmark lists and return physical location IDs."""
    import re
    heldout = set()
    for filename, count in zip(SCANNET_SPLITS, (800, 300)):
        path = root / filename
        rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != count or len({row[0] for row in rows}) != count:
            raise ValueError(f"Invalid ScanNet exclusion split: {path}. Expected {count} unique rows.")
        for row in rows:
            match = re.search(r"scene\d{4}_\d{2}", row[0])
            if not match or len(row) < 2:
                raise ValueError(f"Malformed ScanNet exclusion row in {path}: {row}")
            heldout.add(match.group().split("_")[0])
    return heldout

# Inputs to existing manual acquisition scripts. The catalog's `expected`
# fields usually describe their packed outputs, so they cannot serve as the
# only check before a first acquisition.
MANUAL_INPUTS = {
    "acdc": ("rgb_anon_trainvaltest.zip", "gt_trainval.zip", "gt_panoptic_trainval.zip"),
    "cityscapes": ("leftImg8bit_trainvaltest.zip", "gtFine_trainvaltest.zip"),
    "mapillary_vistas": ("*.zip",), "bdd100k": ("*.zip",),
    "idd": ("idd-segmentation.tar.gz", "idd-20k-II.tar.gz"),
    "nuimages": ("nuimages-v1.0-all-metadata.tgz", "nuimages-v1.0-all-samples.tgz"),
    "wilddash2": ("wd_public_v2p.zip",), "railsem19": ("rs19_val.zip",),
    "reasonseg": ("reasonseg_train_test/train.zip|reasonseg_train_test/train.tar", "reasonseg_val/val.zip|reasonseg_val/*.json"),
}


def complete_cityscapes_raw_archive(path, expected_counts=None):
    """Recognize the raw TAR layout already supported by the Cityscapes script.

    This checks structural completeness, not license acceptance or acquisition
    provenance. Original downloads remain preferable for fresh public setup.
    """
    import tarfile
    from pathlib import PurePosixPath
    expected_counts = expected_counts or {"train": 2975, "val": 500}
    if not path.is_file():
        return False
    try:
        with tarfile.open(path, "r:") as archive:
            names = []
            for member in archive:
                parts = PurePosixPath(member.name)
                if parts.is_absolute() or ".." in parts.parts or not (member.isfile() or member.isdir()):
                    return False
                if member.isfile():
                    names.append(member.name.removeprefix("./"))
        if len(names) != len(set(names)):
            return False
        available = set(names)
        for split, count in expected_counts.items():
            images = [n for n in names if n.startswith(f"cityscapes/leftImg8bit/{split}/")
                      and n.endswith("_leftImg8bit.png")]
            if len(images) != count:
                return False
            for image in images:
                target = image.replace("/leftImg8bit/", "/gtFine/").removesuffix("_leftImg8bit.png")
                if any(target+suffix not in available for suffix in
                       ("_gtFine_labelIds.png", "_gtFine_labelTrainIds.png", "_gtFine_polygons.json")):
                    return False
        return True
    except (OSError, tarfile.TarError):
        return False


def complete_synscapes_raw_archive(path, expected_frames=25000):
    """Header-only completeness check for a packed source shared by all tasks."""
    import tarfile
    from pathlib import PurePosixPath
    groups = {name: set() for name in ("rgb", "depth", "class", "instance", "meta")}
    seen = set()
    try:
        with tarfile.open(path, "r:") as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
                    return False
                if not member.isfile():
                    continue
                if member.name in seen or member.size <= 0:
                    return False
                seen.add(member.name)
                for group in groups:
                    prefix = "Synscapes/meta/" if group == "meta" else f"Synscapes/img/{group}/"
                    suffixes = (".json",) if group == "meta" else (".exr",) if group == "depth" else (".png", ".jpg", ".jpeg")
                    if member.name.startswith(prefix) and name.suffix.lower() in suffixes:
                        groups[group].add(name.stem)
        return len(groups["rgb"]) == expected_frames and all(keys == groups["rgb"] for keys in groups.values())
    except (OSError, tarfile.TarError):
        return False


def missing_manual_assets(root, sources):
    from common.release_data import raw_directory, staged_source_status
    missing = []
    for item in sources:
        if item.access != "manual" or staged_source_status(root, item)[0]:
            continue
        directory = raw_directory(root, item)
        if item.name == "cityscapes" and complete_cityscapes_raw_archive(directory / "cityscapes.tar"):
            # The packed acquisition adapter validates this original-layout TAR.
            # The downloader publishes its marker only after that check passes.
            continue
        patterns = MANUAL_INPUTS.get(item.name, item.expected)
        details = tuple(pattern for pattern in patterns
                        if not any(p.is_dir() or (p.is_file() and p.stat().st_size > 0)
                                   for part in pattern.split("|") for p in directory.glob(part)))
        if item.name == "waymo":
            # Its established downloader requires an authenticated SDK session.
            import subprocess
            import shutil
            ready = shutil.which("gcloud") and subprocess.run(
                ["gcloud", "auth", "print-access-token"], stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, check=False).returncode == 0
            if not ready:
                details = ("gcloud authentication after accepting the Waymo terms",)
        elif not patterns:
            details = ("official source assets described above",)
        if details:
            missing.append((item, directory.resolve(), details))
    return missing


def extend_catalog(catalog):
    """Extend the existing catalog without changing the tested 2D entries."""
    from common.release_data import Dataset, Step
    script = "scripts/dataset_prep/acquire_geometry_sources.py"
    def acquire(name):
        return (Step(script, (("OVB_OFFICIAL_SOURCE", name), ("OVB_RAW_DIRECTORY", "{dir}"))),)
    updates = {
        "synscapes": dict(expected=("Synscapes/img/rgb|synscapes.tar", "Synscapes/meta|synscapes.tar"),
            note="Public Synscapes release (~337 GB). Existing original-layout synscapes.tar can remain packed. "
            "The existing downloader is shared with segmentation; geometry preparation uses "
            "the metric depth and normal annotations. Non-commercial terms: https://synscapes.on.liu.se/download.html"),
        "hypersim": dict(access="automatic", steps=acquire("hypersim"),
            expected=("archives/*.zip|**/scene_cam_*_final_hdf5/frame.*.color.hdf5", "metadata_camera_parameters.csv"),
            note="Official Apple scene ZIPs and camera metadata. Readers use original ZIP members without scene extraction. "
                 "Use --set hypersim.scenes=ai_001_001 to acquire one scene."),
        "taskonomy": dict(access="automatic", steps=acquire("taskonomy"),
            note="Official EPFL modality archives, selected by complete building triplets. "
                 "Use --set taskonomy.num-buildings=1 for a small acquisition."),
        "replica": dict(access="automatic", steps=acquire("replica"),
            expected=("**/mesh.ply", "**/textures", "**/*.navmesh"),
            note="Official Replica v1 release and Habitat configurations. Preparation needs Habitat-Sim "
                 "and the Replica SDK Ptex renderer. Vertex-colour RGB is not used by the default recipes."),
        "vkitti2": dict(access="automatic", files=(
            ("https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar", "vkitti_2.0.3_rgb.tar"),
            ("https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_depth.tar", "vkitti_2.0.3_depth.tar"),
            ("https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_textgt.tar.gz", "vkitti_2.0.3_textgt.tar.gz")),
            note="Official Naver RGB, metric depth and camera metadata. NZS uses non-clone variants."),
        "diode": dict(access="automatic", steps=acquire("diode"),
            expected=("train.tar.gz", "train_normals.tar.gz", "val.tar.gz", "val_normals.tar.gz"),
            note="Official DIODE RGB/depth/masks and normal archives. Training uses the training split; "
                 "benchmark evaluation uses separate validation data."),
        "nyuv2": dict(dependencies=("dsine",),
            files=catalog[[x.name for x in catalog].index("nyuv2")].files + (
            ("https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat", "splits.mat"),),
            note="Official labeled MAT and train/test split; training normals use the 795 GeoNet "
                 "labels in the author's DSINE ZIP, acquired by the dsine dependency. No depth-derived replacement."),
        "pico_banana": dict(steps=acquire("pico_banana"),
            expected=("packed.json", "sources.json", "shards/*.tar"),
            note="Apple SFT metadata and edited images, paired with original OpenImages URLs. "
                 "Full and limited acquisitions use the same resumable packed-shard path; no persistent extraction."),
        "scannet": dict(dependencies=("geometry_splits",), expected=("**/scannet_frames_25k.zip|**/scannet_frames_25k", "**/scannetv2_train.txt", "**/scannetv2_val.txt"),
            note="Request access through https://github.com/ScanNet/ScanNet#scannet-data. "
                 "Complete the Terms of Use agreement with your institutional email and send it to "
                 "scannet@googlegroups.com as instructed by the authors. After approval, use their "
                 "download instructions to obtain scannet_frames_25k.zip. Place that original archive "
                 "and scannetv2_train.txt/scannetv2_val.txt in this raw directory; retain the ZIP without extraction. "
                 "The split files are available in the official repository under Tasks/Benchmark/. "
                 "This adapter expects the 25k frame release, not raw .sens scans. "
                 "Rerun the same setup command after staging. "
                 "NZS training excludes physical locations used by either benchmark split."),
        "matterport3d": dict(expected=tuple("*.tar|**/" + modality + ".zip" for modality in
                 ("undistorted_color_images", "undistorted_depth_images", "undistorted_normal_images", "undistorted_camera_parameters")),
            note="Request access at https://niessner.github.io/Matterport/. Fill and sign the Terms of Use "
                 "agreement and send it to matterport3d@googlegroups.com as instructed by the authors. "
                 "After approval, use the supplied official downloader for undistorted_color_images, "
                 "undistorted_depth_images, undistorted_normal_images and undistorted_camera_parameters. "
                 "Preserve one directory per scan containing those four original ZIPs under this raw "
                 "directory, for example <scan_id>/undistorted_color_images.zip. Do not extract the ZIPs. "
                 "Existing per-scan raw TARs are also accepted. Rerun the same setup command after staging."),
    }
    result = [replace(x, **updates[x.name]) if x.name in updates else x for x in catalog]
    for x in result:
        if x.name in {"zeroshot_pool", "nonzeroshot_pool"}:
            result[result.index(x)] = replace(x, note=x.note.split("; add geometry")[0] +
                ". Combine with official geometry using configs/data_zs.yaml or configs/data_nzs.yaml.")
    result += [
        Dataset("geometry_splits", "Shared_geo", (), "automatic", "Pinned author benchmark split lists",
            files=tuple((f"{MARIGOLD_RAW}/evaluation/data_split/{name}", "data_split/" + name) for name in SCANNET_SPLITS),
            note="Small metadata files required to exclude benchmark locations from ScanNet training."),
        Dataset("marigold_depth", "Shared_geo", ("Depth",), "automatic", "Marigold V2 depth benchmarks",
            steps=acquire("marigold_depth"), expected=("sources.json",),
            note="Author-distributed NYUv2/KITTI Eigen/ETH3D/ScanNet/DIODE targets with pinned split lists. "
                 "https://github.com/huawei-bayerlab/marigold-v2#evaluation"),
        Dataset("marigold_normals", "Shared_geo", ("Normals",), "automatic", "Marigold normal benchmarks",
            steps=acquire("marigold_normals"), expected=("evaluation_dataset.zip", "sintel.zip", "sources.json"),
            note="Author-distributed NYUv2, ScanNet, iBims-1 and Sintel float normal targets. "
                 "https://github.com/huawei-bayerlab/marigold-v2#evaluation"),
        Dataset("dsine", "Shared_geo", ("Normals",), "automatic", "DSINE evaluation assets",
            steps=acquire("dsine"), expected=("**/dsine_eval.zip", "vkitti.txt"),
            note="Author's public dsine_eval.zip, including the fixed VKITTI benchmark. "
                 "https://github.com/baegwangbin/DSINE#evaluation. "
                 "If Google Drive limits automated access, download dsine_eval.zip from the linked folder here."),
        Dataset("diode_normals", "Shared_geo", ("Normals",), "automatic", "DIODE validation normals",
            steps=acquire("diode_normals"), expected=("val.tar.gz", "val_normals.tar.gz", "data_list/val_indoors.csv"),
            note="Official DIODE validation RGB, float normal targets and indoor split. https://diode-dataset.org/"),
    ]
    return tuple(result)
