"""Canonical source identities for the geometry benchmarks used by OVB."""
from __future__ import annotations

import re
import hashlib
from pathlib import Path

ALIASES = {"nyuv2":"nyu", "nyu":"nyu", "scannetv2":"scannet", "scannet":"scannet",
           "kitti_eigen":"kitti", "kitti":"kitti", "diode":"diode",
           "vkitti2_nonclone":"vkitti2", "vkitti2":"vkitti2", "ibims":"ibims1"}
BENCHMARK_DOMAINS = {"nyu", "scannet", "kitti", "diode", "eth3d", "ibims1", "vkitti2", "sintel"}
UNSUPPORTED_TRAINING_DOMAINS = {"eth3d", "ibims1", "sintel"}
PINNED_SPLIT_ROOT = Path('data/marigold_v2_depth_eval_20260920/data_split')
SCANNET_SPLITS_V1 = {
    'scannet_depth/scannet_val_sampled_list_800_1.txt': (800, '066c76d6d8f93d6d7bed812e9c3bb0c375f08e7ff76569e13ce9c000dd74ead5'),
    'scannet_normals/scannet_test.txt': (300, '88b4b1df66f0e241734e2fd766687700aac6a99834ffbcd8441b2b956f5c55ed'),
}


def validate_scannet_splits(root, expected=None):
    """Validate each list separately; production v1 pins both bytes and counts.

    Alternative roots may supply their own explicit versioned identities. Small
    fixtures use structural validation, never masquerading as the full benchmark.
    """
    root = Path(root)
    if expected is None and root.resolve() == PINNED_SPLIT_ROOT.resolve():
        expected = SCANNET_SPLITS_V1
    result = {}
    for relative in SCANNET_SPLITS_V1:
        path = root/relative
        content = path.read_bytes()
        lines = [line.strip() for line in content.decode().splitlines()
                 if line.strip() and not line.lstrip().startswith('#')]
        if not lines:
            raise ValueError(f'Empty ScanNet evaluation split: {relative}')
        ids = set()
        for line in lines:
            first = line.split()[0]
            captures = [x for x in first.split('/') if re.fullmatch(r'scene\d{4}_\d{2}', x)]
            frame = re.fullmatch(r'(\d+)(?:_(?:rgb|img))?', Path(first).stem)
            if len(captures) != 1 or frame is None:
                raise ValueError(f'Ambiguous ScanNet evaluation path in {relative}: {first!r}')
            identity = (captures[0], int(frame[1]))
            if identity in ids:
                raise ValueError(f'Duplicate ScanNet evaluation frame in {relative}: {identity}')
            ids.add(identity)
        checksum = hashlib.sha256(content).hexdigest()
        if expected is not None:
            if relative not in expected or (len(lines), checksum) != tuple(expected[relative]):
                raise ValueError(f'ScanNet split version/count/checksum mismatch: {relative}')
        result[relative] = {'images': len(lines), 'sha256': checksum,
                            'identity_pinned': expected is not None}
    return result


def scannet_location_id(scene):
    """Physical space, not capture: scene0001_00 and _01 share scene0001."""
    match = re.fullmatch(r"(scene\d{4})(?:_\d{2})?", str(scene))
    if match is None:
        raise ValueError(f"Ambiguous ScanNet scene/location identity: {scene!r}")
    return match[1]


def domain(row):
    metadata = row.get("metadata") or {}
    source = str(row.get("source_dataset") or metadata.get("dataset") or "").lower()
    return ALIASES.get(source, source)


def source_identity(row):
    metadata = row.get("metadata") or {}
    identifier = str(metadata.get("source_sample_id") or row.get("id") or "")
    scene = str(row.get("scene") or metadata.get("scene") or "")
    source = domain(row)
    if source == "nyu":
        index = metadata.get("source_index_1based")
        if index is None and metadata.get("source_id_0based") is not None:
            index = int(metadata["source_id_0based"]) + 1
        if index is None:
            match = re.fullmatch(r"nyu_(\d+)", identifier)
            if match: index = int(match[1])
        if index is None:
            raise ValueError(f"Ambiguous NYU training identity: {identifier}")
        identifier = f"nyu_{int(index):04d}"
    elif source == "scannet":
        match = re.fullmatch(r"(scene\d{4}_\d{2})_(\d+)", identifier)
        if match is None:
            raise ValueError(f"Ambiguous ScanNet source scene/frame ID: {identifier!r}")
        capture = match[1]
        for declared in (row.get("scene"), metadata.get("scene")):
            if declared and str(declared) != capture:
                raise ValueError(f"Conflicting ScanNet scene IDs: {declared!r} vs {capture!r}")
        scene = capture
        identifier = f"{scene}_{int(match[2]):06d}"
    return source, identifier, scene


def load_registry(root: Path):
    validate_scannet_splits(root)
    files = {
        "scannet_normals":"scannet_normals/scannet_test.txt",
        "scannet_depth":"scannet_depth/scannet_val_sampled_list_800_1.txt",
        "nyu_normals":"nyu_normals/nyuv2_test.txt",
        "nyu_depth":"nyu_depth/labeled/filename_list_test.txt",
        "kitti_depth":"kitti_depth/eigen_test_files_with_gt.txt",
        "diode_depth":"diode_depth/diode_val_all_filename_list.txt",
        "eth3d_depth":"eth3d_depth/eth3d_filename_list.txt",
        "ibims_normals":"ibims_normals/ibims_test.txt",
        "sintel_normals":"sintel_normals/sintel_filtered.txt",
    }
    registry = {key:{"ids":set(), "scenes":set()} for key in BENCHMARK_DOMAINS}
    registry["scannet"]["locations"] = set()
    for lane, relative in files.items():
        lines = (root / relative).read_text().splitlines()
        source = lane.split('_')[0]
        if source == "ibims": source = "ibims1"
        for line in lines:
            if not line.strip() or line.lstrip().startswith('#'): continue
            first = line.split()[0]
            parts = first.split('/')
            if source == "scannet":
                captures = [x for x in parts if re.fullmatch(r"scene\d{4}_\d{2}",x)]
                if len(captures) != 1:
                    raise ValueError(f"Ambiguous ScanNet evaluation path in {relative}: {first!r}")
                scene = captures[0]
                frame = int(Path(first).stem.split('_')[0])
                registry[source]["scenes"].add(scene)
                registry[source]["locations"].add(scannet_location_id(scene))
                registry[source]["ids"].add(f"{scene}_{frame:06d}")
            elif source == "nyu":
                index = int(re.search(r"\d+",Path(first).stem)[0]) + (1 if lane.endswith('normals') else 0)
                registry[source]["ids"].add(f"nyu_{index:04d}")
            elif source == "kitti":
                scene = next(x for x in parts if '_drive_' in x)
                camera = next(x for x in parts if re.fullmatch(r'image_\d+',x))
                registry[source]["scenes"].add(scene)
                registry[source]["ids"].add(f"{scene}_{camera}_{Path(first).stem}")
            elif source == "diode":
                registry[source]["ids"].add(Path(first).stem)
                registry[source]["scenes"].add('/'.join(parts[:-1]))
    if not registry["scannet"]["locations"]:
        raise ValueError("Empty ScanNet evaluation-location registry")
    # ScanNet holds out every recording at each depth/normal evaluation location.
    # DIODE normals are the indoor subset of these IDs; VKITTI2 evaluation uses
    # the clone variant.
    return registry, [root / relative for relative in files.values()]


def exclusion_reason(row, registry):
    source, identifier, scene = source_identity(row)
    if source in UNSUPPORTED_TRAINING_DOMAINS:
        raise ValueError(f'No validated training exclusion parser for {source}; evaluation-only domain')
    if source not in registry: return None
    if identifier in registry[source]["ids"]: return "exact_evaluation_frame"
    if source in {"scannet", "kitti", "diode"} and scene in registry[source]["scenes"]:
        return "evaluation_scene"
    if source == "scannet":
        # Registries without a locations set derive it from the captures.
        locations = registry[source].get("locations")
        if locations is None:
            locations = {scannet_location_id(capture) for capture in registry[source]["scenes"]}
        if scannet_location_id(scene) in locations:
            return "evaluation_location"
    if source == "vkitti2":
        metadata = row.get("metadata") or {}
        if metadata.get("variant") == "clone" or '_clone_' in identifier:
            return "evaluation_clone_variant"
    return None
