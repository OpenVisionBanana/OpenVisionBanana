#!/usr/bin/env python3
"""Prepare official raw datasets into versioned OVB train/evaluation archives."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.release_data import (
    ALIASES,
    BY_NAME,
    CATALOG,
    list_header,
    list_row,
    parse_selection,
    raw_directory,
    run_python,
    sha256,
    staged_source_status,
    acquisition_items,
)

REPO = Path(__file__).resolve().parents[1]
GEOMETRY_EVAL = {"diode", "eth3d", "ibims1", "kitti", "nuscenes", "vkitti2", "scannet"}
GEOMETRY_TRAIN = {"hypersim", "matterport3d", "replica", "taskonomy", "synscapes", "nyuv2", "diode", "scannet", "vkitti2"}
# Segmentation, editing and generation: benchmarks are materialized by
# scripts/dataset_prep/prepare_task_eval.py, training pools are built by
# scripts/dataset_prep/build_training_pool.py.
TASK_EVAL = {"cityscapes", "coco_object_80", "saco_gold", "refcocog", "reasonseg", "imgedit_bench", "genai_bench"}
POOL_TRAIN = {"zeroshot_pool", "nonzeroshot_pool"}
SINGLE_TRAIN = {"pico_banana", "pd12m"}
AUTHOR_BENCHMARKS = {"marigold_depth", "marigold_normals", "dsine", "diode_normals"}
EVAL_IMPLEMENTED = GEOMETRY_EVAL | TASK_EVAL | AUTHOR_BENCHMARKS
TRAIN_IMPLEMENTED = GEOMETRY_TRAIN | POOL_TRAIN | SINGLE_TRAIN

# Converter-specific; kept in lockstep with each underlying parser.
EVAL_OVERRIDES: dict[str, set[str]] = {
    **{name: set() for name in AUTHOR_BENCHMARKS},
    "diode": {"domains", "cx", "cy", "fx", "fy", "min-depth", "max-depth",
              "min-normal-norm", "invalid-normal-rgb", "geometry-audit"},
    "eth3d": {"baseline-unit", "force-extract", "min-depth", "max-depth",
               "gradient-radius", "mask-erosion-radius", "max-neighbor-depth-ratio",
               "min-normal-valid-pixels", "normal-encoding", "invalid-normal-rgb"},
    "ibims1": {"include-aux-augmented", "aux-augmentations", "min-depth", "max-depth",
                "depth-validity-protocol", "geometry-version", "gradient-radius",
                "mask-erosion-radius", "invalid-normal-rgb"},
    "kitti": {"min-depth", "max-depth", "gradient-radius", "mask-erosion-radius",
               "max-neighbor-depth-ratio", "min-normal-valid-pixels",
               "normal-encoding", "invalid-normal-rgb"},
    "nuscenes": {"scene-list", "cameras", "limit", "version", "min-depth", "max-depth",
                  "normal-k-neighbors", "normal-min-neighbors", "normal-encoding",
                  "invalid-normal-rgb"},
    "vkitti2": {"scenes", "variants", "cameras", "limit", "depth-png-scale",
                 "min-depth", "max-depth", "gradient-radius", "mask-erosion-radius",
                 "max-neighbor-depth-ratio", "min-normal-valid-pixels",
                 "normal-encoding", "invalid-normal-rgb"},
    "scannet": {"frames-root", "val-scenes-file", "min-depth", "max-depth",
                 "min-valid-fraction", "gradient-radius", "mask-erosion-radius",
                 "normal-encoding", "invalid-normal-rgb", "strict"},
    **{name: set() for name in TASK_EVAL},
}

TRAIN_OVERRIDES: dict[str, set[str]] = {
    "hypersim": {"samples-per-scene", "target-samples", "all-frames", "val-ratio",
                  "min-depth", "max-depth", "min-valid-fraction", "max-side",
                  "camera-parameters-csv", "legacy-euclidean-depth", "depth-encoding",
                  "depth-invalid-rgb", "invalid-normal-rgb", "normal-encoding",
                  "write-raw-normals", "raw-normal-dtype", "include-val-in-metadata",
                  "allow-empty", "delete-raw-scenes-after-convert", "strict"},
    "taskonomy": {"depth-type", "num-buildings", "building-list", "samples-per-building",
                   "target-samples", "all-frames", "train-ratio", "min-depth", "max-depth",
                   "min-valid-fraction", "max-side", "depth-encoding", "depth-invalid-rgb",
                   "invalid-normal-rgb", "normal-encoding", "write-raw-normals",
                   "raw-normal-dtype", "include-val-in-metadata", "strict", "legacy-geometry"},
    "replica": {"num-scenes", "scene-list", "samples-per-scene", "max-retry-factor",
                "val-ratio", "width", "height", "hfov", "camera-height", "camera-world-up-axis", "pitch-min",
                "pitch-max", "min-depth", "max-depth", "min-depth-std",
                "min-valid-fraction", "min-rgb-mean", "depth-encoding",
                "depth-invalid-rgb", "invalid-normal-rgb", "normal-encoding",
                "normal-source", "depth-convention", "scene-id-mode", "gpu-device-id",
                "write-raw-normals", "raw-normal-dtype", "no-orient-z-positive",
                "include-val-in-metadata", "allow-empty", "strict",
                "legacy-unverified-rendered-normals", "allow-vertex-color-rgb", "ptex-renderer"},
    "matterport3d": {"profile", "samples-per-scene", "max-scenes", "scans-file",
                     "start-index", "min-depth", "max-depth", "min-valid-fraction",
                     "force-scenes", "strict", "strict-calibration", "write-raw-depth",
                     "write-raw-normals", "normal-encoding"},
    **{name: {"staging-root", "allow-missing-hypersim", "semseg-num-images", "limit"}
       for name in ("zeroshot_pool", "nonzeroshot_pool")},
    "synscapes": {"limit", "max-side"},
    **{name: {"limit"} for name in ("nyuv2", "diode", "scannet", "vkitti2")},
    **{name: {"staging-root", "limit"} for name in SINGLE_TRAIN},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--destination", type=Path, default=Path("data/prepared"))
    parser.add_argument("--work-dir", type=Path,
                        help="Existing writable directory for temporary files; defaults to the system temporary directory.")
    parser.add_argument("--version", default="V1_prepared_seg_geo")
    parser.add_argument("--datasets", default="all", help="Comma-separated names or all.")
    parser.add_argument("--splits", default="train,eval", help="train, eval, or both.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--dataset-workers",
        type=int,
        default=1,
        help="Number of independent dataset converters to run concurrently.",
    )
    parser.add_argument("--set", action="append", default=[], metavar="DATASET.KEY=VALUE",
                        help="Dataset-specific override; repeat as needed.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-build-indexes", action="store_true",
                        help="Skip indexed-tar sidecars and the generated geometry training spec.")
    parser.add_argument("--samples-per-epoch", type=int, default=1_000_000,
                        help="Sample count written to the generated geometry-only training spec.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--geometry-policy", choices=("base-clean",),
                        help="Materialize fixed geometry targets, masks and training-only filtering.")
    parser.add_argument("--training-pool", choices=("zs", "nzs"),
                        help="Combine the matching 2D pool and selected geometry into one training manifest.")
    parser.add_argument(
        "--allow-unsupported",
        action="store_true",
        help="Skip requested dataset/split pairs marked TODO instead of failing before preparation.",
    )
    parser.add_argument(
        "--skip-staging-check",
        action="store_true",
        help="Skip the lightweight raw-source presence check; converters still validate their inputs.",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--describe",
        metavar="DATASET",
        help="Show split support and accepted DATASET.KEY override keys, then exit.",
    )
    return parser.parse_args()


def parse_overrides(values: list[str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for value in values:
        try:
            left, setting = value.split("=", 1)
            dataset, key = left.split(".", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid --set {value!r}; expected DATASET.KEY=VALUE") from exc
        dataset = dataset.lower().replace("-", "_")
        dataset = ALIASES.get(dataset, dataset)
        if dataset not in BY_NAME:
            raise ValueError(f"Unknown dataset in --set: {dataset}")
        result.setdefault(dataset, {})[key.replace("_", "-")] = setting
    return result


def add_overrides(command: list[str], overrides: dict[str, str], allowed: set[str]) -> None:
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"Unsupported override(s): {', '.join(sorted(unknown))}")
    for key, value in sorted(overrides.items()):
        if value.lower() in {"true", "yes", "on"}:
            command.append(f"--{key}")
        elif value.lower() not in {"false", "no", "off"}:
            command.extend((f"--{key}", value))


def geometry_eval_command(name: str, raw: Path, root: Path, args, overrides) -> list[str] | None:
    if name in AUTHOR_BENCHMARKS:
        command = [str(REPO / "scripts/dataset_prep/prepare_geometry_benchmarks.py"),
                   "--dataset", name, "--raw-dir", str(raw), "--root", str(root)]
        if args.max_samples:
            command += ["--limit", str(args.max_samples)]
        if args.overwrite:
            command.append("--overwrite")
        return command
    depth_dir, normal_dir = root / "eval/Depth", root / "eval/Normals"
    work = root / "work/eval" / name
    common = {
        "diode": ("dataset_prep/diode/prepare_diode_eval.py", ["--source-root", raw]),
        "eth3d": ("dataset_prep/eth3d/prepare_eth3d_eval.py", ["--eth3d-root", raw]),
        "ibims1": ("dataset_prep/ibims1/prepare_ibims1_eval.py", ["--source-root", raw]),
        "kitti": ("dataset_prep/kitti/prepare_kitti_eval.py", ["--selection-zip", raw / "depth_selection.zip"]),
        "nuscenes": ("dataset_prep/nuscenes/prepare_nuscenes_eval.py", ["--source-root", raw, "--split", "val"]),
    }
    if name in common:
        script, initial = common[name]
        command = [*map(str, initial), "--task", "both",
                   "--depth-out-root", str(work / "depth"), "--normals-out-root", str(work / "normals"),
                   "--depth-tar-output", str(depth_dir / f"{name}.tar"),
                   "--normals-tar-output", str(normal_dir / f"{name}.tar")]
        if args.max_samples and name == "nuscenes":
            command += ["--limit", str(args.max_samples)]
        add_overrides(command, overrides, EVAL_OVERRIDES[name])
        if args.overwrite:
            command.append("--overwrite" if name != "scannet" else "--force")
        return [str(REPO / script), *command]
    if name == "vkitti2":
        def one(pattern: str) -> Path:
            matches = sorted(raw.glob(pattern))
            if not matches and args.dry_run:
                placeholders = {
                    "*rgb*.tar*": "vkitti_2.0.3_rgb.tar.gz",
                    "*depth*.tar*": "vkitti_2.0.3_depth.tar.gz",
                    "*textgt*.tar*": "vkitti_2.0.3_textgt.tar.gz",
                }
                return raw / placeholders[pattern]
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected exactly one {pattern} under {raw}; found {len(matches)}")
            return matches[0]
        command = [str(REPO / "dataset_prep/vkitti2/prepare_vkitti2_eval.py"),
                   "--rgb-tar", str(one("*rgb*.tar*")), "--depth-tar", str(one("*depth*.tar*")),
                   "--textgt-tar", str(one("*textgt*.tar*")), "--task", "both",
                   "--depth-out-root", str(work / "depth"), "--normals-out-root", str(work / "normals"),
                   "--depth-tar-output", str(depth_dir / "vkitti2.tar"),
                   "--normals-tar-output", str(normal_dir / "vkitti2.tar")]
        if args.max_samples:
            command += ["--limit", str(args.max_samples)]
        add_overrides(command, overrides, EVAL_OVERRIDES[name])
        if args.overwrite:
            command.append("--overwrite")
        return command
    if name == "scannet":
        shared = root / "work/eval/scannet.tar"
        command = [str(REPO / "dataset_prep/scannet/prepare_scannet_depth_normals.py"),
                   "--source-root", str(raw), "--out-root", str(work / "dataset"), "--prepared-tar", str(shared),
                   "--extract-zip"]
        add_overrides(command, overrides, EVAL_OVERRIDES[name])
        if args.overwrite:
            command.append("--force")
        return command
    return None


def task_eval_command(name: str, raw: Path, root: Path, args, overrides) -> list[str] | None:
    if name not in TASK_EVAL:
        return None
    task = BY_NAME[name].tasks[0]
    command = [str(REPO / "scripts/dataset_prep/prepare_task_eval.py"), "--dataset", name,
               "--raw-dir", str(raw), "--out-dir", str(root / "eval" / task / name)]
    add_overrides(command, overrides, EVAL_OVERRIDES[name])
    if args.overwrite:
        command.append("--overwrite")
    return command


def pool_train_command(name: str, raw_root: Path, root: Path, args, overrides) -> list[str] | None:
    if name not in POOL_TRAIN | SINGLE_TRAIN:
        return None
    if name in SINGLE_TRAIN:
        command = [str(REPO / "scripts/dataset_prep/prepare_creative_training.py"), "--dataset", name,
                   "--raw-dir", str(raw_directory(raw_root, BY_NAME[name])), "--root", str(root),
                   "--workers", str(args.workers), "--seed", str(args.seed)]
        add_overrides(command, overrides, TRAIN_OVERRIDES[name])
        if args.overwrite:
            command.append("--overwrite")
        return command
    command = [str(REPO / "scripts/dataset_prep/build_training_pool.py"), "--pool", name.removesuffix("_pool"),
               "--raw-root", str(raw_root), "--root", str(root), "--workers", str(args.workers)]
    if name in SINGLE_TRAIN:
        command += ["--sources", name]
    # The global --max-samples stays with the geometry converters: the paper pools
    # have fixed member sizes. Cap a pool explicitly with --set <pool>.limit=N.
    add_overrides(command, overrides, TRAIN_OVERRIDES[name])
    if args.overwrite:
        command.append("--overwrite")
    return command


def geometry_train_command(name: str, raw: Path, root: Path, args, overrides) -> list[str] | None:
    if getattr(args, "geometry_policy", None):
        root = root / "work/native_geometry"
    depth_tar = root / "train/Depth" / f"{name}.tar"
    normal_tar = root / "train/Normals" / f"{name}.tar"
    work = root / "work/train" / name
    base = ["--depth-out-root", str(work / "depth"), "--normals-out-root", str(work / "normals"),
            "--depth-prepared-tar", str(depth_tar), "--normals-prepared-tar", str(normal_tar),
            "--tasks", "depth", "normals", "--seed", str(args.seed)]
    allowed = TRAIN_OVERRIDES[name]
    if name in {"nyuv2", "diode", "scannet", "vkitti2"}:
        command = [str(REPO / "scripts/dataset_prep/prepare_geometry_training.py"),
                   "--dataset", name, "--raw-dir", str(raw), "--output", str(root / "train"),
                   "--split-root", str(args.raw_root.expanduser().resolve() / "Shared_geo/geometry_splits/data_split")]
        if args.max_samples:
            command += ["--limit", str(args.max_samples)]
        add_overrides(command, overrides, allowed)
        if args.overwrite:
            command.append("--overwrite")
        return command
    if name == "synscapes":
        command = [str(REPO / "scripts/dataset_prep/prepare_synscapes_geometry.py"),
                   "--raw-dir", str(raw), "--output", str(root / "train"),
                   "--seed", str(args.seed), "--workers", str(args.workers)]
        if args.max_samples:
            command += ["--limit", str(args.max_samples)]
        add_overrides(command, overrides, allowed)
        if args.overwrite:
            command.append("--overwrite")
        return command
    if name == "hypersim":
        command = [str(REPO / "dataset_prep/hypersim/prepare_hypersim_depth_normals.py"),
                   "--raw-root", str(raw), *base, "--num-workers", str(args.workers)]
        if "camera-parameters-csv" not in overrides:
            command += ["--camera-parameters-csv", str(raw / "metadata_camera_parameters.csv")]
        if args.max_samples:
            command += ["--target-samples", str(args.max_samples)]
    elif name == "taskonomy":
        command = [str(REPO / "dataset_prep/taskonomy/prepare_taskonomy_depth_normals.py"),
                   "--raw-tar-root", str(raw), *base]
        if args.max_samples:
            command += ["--target-samples", str(args.max_samples)]
    elif name == "replica":
        command = [str(REPO / "dataset_prep/replica/prepare_replica_depth_normals.py"),
                   "--replica-root", str(raw), *base]
    elif name == "matterport3d":
        command = [str(REPO / "scripts/dataset_prep/prepare_matterport_official.py"),
                   "--source-tar-root", str(raw), "--scratch-root", str(work / "scratch"),
                   "--shard-root", str(work / "shards"), "--depth-tar", str(depth_tar),
                   "--normals-tar", str(normal_tar), "--seed", str(args.seed),
                   "--scene-workers", str(args.workers)]
    else:
        return None
    if getattr(args, "geometry_policy", None):
        if name != "matterport3d":
            command += ["--depth-encoding", "vision_banana_rgb"]
        command += ["--normal-encoding", "vision_banana_x_left_red"]
        if name == "matterport3d" and "write-raw-depth" not in overrides:
            command.append("--write-raw-depth")
    add_overrides(command, overrides, allowed)
    if args.overwrite and name != "matterport3d":
        command.append("--overwrite")
    elif args.overwrite and name == "matterport3d":
        command.append("--force-scenes")
    return command


def finalize_outputs(root: Path, args: argparse.Namespace, records: list[dict]) -> None:
    """Atomically merge manifests and rebuild indexes/checksums.

    The lock makes disjoint per-dataset invocations safe on multiple nodes.
    Individual converters must still target different dataset archives.
    """
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    with (manifests / ".prepare.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        manifest_path = manifests / "preparation_manifest.json"
        previous: list[dict] = []
        if manifest_path.is_file():
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, list):
                raise ValueError(f"Expected a JSON list in {manifest_path}")
            previous = loaded
        merged = {record["dataset"]: record for record in previous}
        for record in records:
            prior = merged.get(record["dataset"])
            if prior:
                record = {
                    **prior,
                    **record,
                    "splits": sorted(set(prior.get("splits", ())) | set(record.get("splits", ()))),
                }
            merged[record["dataset"]] = record
        if "train" in {part.strip() for part in args.splits.split(",")} and not args.no_build_indexes:
            from common.indexed_tar import load_index
            for archive in sorted((root / "train").glob("**/*.tar")):
                index = archive.with_name(archive.name + ".ovb-index.sqlite")
                if index.is_file():
                    try:
                        load_index(index, archive)
                        continue
                    except (ValueError, KeyError, OSError):
                        if not args.overwrite:
                            raise
                command = [str(archive), "--output", str(index)]
                if index.exists():
                    command.append("--force")
                run_python(REPO / "scripts/dataset_prep/build_prepared_tar_index.py", command, False)
        if "train" in {part.strip() for part in args.splits.split(",")} and not args.no_build_indexes:
            task_archives = {
                "Depth": sorted((root / "train/Depth").glob("*.tar")),
                "Normals": sorted((root / "train/Normals").glob("*.tar")),
            }
            specs = []
            task_totals = {"Depth": 1.0 / 3.0, "Normals": 2.0 / 3.0}
            for task, archives in task_archives.items():
                for archive in archives:
                    index = archive.with_name(archive.name + ".ovb-index.sqlite")
                    specs.append({
                        "name": f"{task.lower()}_{archive.stem}",
                        "type": "indexed_prepared_tar",
                        # Resolve these relative to the manifest so the whole
                        # prepared version can be copied without path edits.
                        "path": str(Path("..") / archive.relative_to(root)),
                        "index_path": str(Path("..") / index.relative_to(root)),
                        "task": "depth_estimation" if task == "Depth" else "surface_normals",
                        "source_dataset": archive.stem,
                        "weight": task_totals[task] / len(archives),
                    })
            if specs:
                training_spec = {
                    "samples_per_epoch": args.samples_per_epoch,
                    "seed": args.seed,
                    "shuffle": True,
                    "sampling_strategy": "without_replacement",
                    "datasets": specs,
                    "note": "Geometry-only 10:20 depth:normals template; add non-geometry pools before final multitask training.",
                }
                spec_path = manifests / "geometry_training_specs.json"
                temporary_spec = spec_path.with_suffix(".json.partial")
                temporary_spec.write_text(json.dumps(training_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                temporary_spec.replace(spec_path)

        checksum_lines = []
        archives = sorted((root / "train").glob("**/*.tar")) + sorted((root / "eval").glob("**/*.tar"))
        # Indexes and prompt lists are also inputs to loading/evaluation.
        artifacts = archives + sorted((root / "train").glob("**/*.ovb-index.sqlite"))
        for suffix in ("*.jsonl", "*.json"):
            artifacts += sorted((root / "eval").glob("**/" + suffix))
        cache_path = manifests / ".sha256_cache.json"
        fresh = {}
        for archive in artifacts:
            key = str(archive.relative_to(root))
            stat = archive.stat()
            stamp = [stat.st_size, stat.st_mtime_ns]
            from common.indexed_tar import verified_sha256
            digest = verified_sha256(archive)
            fresh[key] = [*stamp, digest]
            checksum_lines.append(f"{digest}  {key}")
        temporary_cache = cache_path.with_suffix(".partial")
        temporary_cache.write_text(json.dumps(fresh, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        temporary_cache.replace(cache_path)
        checksum_path = root / "SHA256SUMS"
        temporary_checksums = checksum_path.with_suffix(".partial")
        temporary_checksums.write_text(
            "\n".join(checksum_lines) + ("\n" if checksum_lines else ""),
            encoding="utf-8",
        )
        temporary_checksums.replace(checksum_path)
        temporary_manifest = manifest_path.with_suffix(".json.partial")
        temporary_manifest.write_text(json.dumps(list(merged.values()), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary_manifest.replace(manifest_path)


def execute_preparation(job, root, args):
    """Keep the tested 2D pool path and resume the new adapters by stage."""
    name, split, command = job
    resumable = (split == "train" and name in GEOMETRY_TRAIN | SINGLE_TRAIN
                 or split == "eval" and name in AUTHOR_BENCHMARKS | {"imgedit_bench", "genai_bench", "refcocog"})
    if args.dry_run or not resumable:
        run_python(Path(command[0]), command[1:], args.dry_run)
        if split == "train" and name in GEOMETRY_TRAIN and args.geometry_policy:
            run_python(REPO / "scripts/dataset_prep/materialize_geometry_source.py", [
                "--dataset", name, "--source-root", str(root / "work/native_geometry/train"),
                "--output-root", str(root / "train")], args.dry_run)
        return
    from common.data_preparation_state import (converter_identity, input_identity, run_stage,
                                               stage_lock, files_below)
    from common.official_sources import scannet_exclusions
    if split == "train" and name == "scannet":
        scannet_exclusions(args.raw_root.resolve() / "Shared_geo/geometry_splits/data_split")
    with stage_lock(root, name, split):
        inputs = [(i.name, raw_directory(args.raw_root.resolve(), i))
                  for i in acquisition_items([BY_NAME[name]])]
        for index, value in enumerate(command[1:]):
            if index and command[index] in {"--ptex-renderer", "--camera-parameters-csv"}:
                inputs.append((command[index], Path(value)))
        normalized = [str(v).replace(str(root), "<prepared>").replace(str(args.raw_root.resolve()), "<raw>")
                      .replace(str(REPO), "<repo>") for v in command if v not in {"--overwrite", "--force-scenes"}]
        identity = dict(command=normalized, seed=args.seed, max_samples=args.max_samples,
                        sources=input_identity(inputs), converter=converter_identity(REPO))
        geometry = split == "train" and name in GEOMETRY_TRAIN
        native_root = root / "work/native_geometry" if geometry and args.geometry_policy else root
        def outputs():
            if geometry:
                return [native_root / "train" / task / f"{name}.tar" for task in ("Depth", "Normals")]
            if split == "train":
                relative = "ImgEdit/pico_banana_sft_full.tar" if name == "pico_banana" else "ImgGeneration/pd12m_250000.tar"
                return [root / "train" / relative]
            if name in {"imgedit_bench", "genai_bench", "refcocog"}:
                task = {"imgedit_bench": "ImgEdit", "genai_bench": "ImgGeneration", "refcocog": "Refseg"}[name]
                directory = root / "eval" / task / name
                if name == "refcocog" and directory.with_suffix(".tar").is_file() and not directory.exists():
                    return [directory.with_suffix(".tar")]
                return list(files_below(directory)) if directory.exists() else []
            names = {"marigold_depth": ("Depth", ["nyu", "kitti_eigen", "eth3d", "scannet", "diode"]),
                     "marigold_normals": ("Normals", ["nyu", "scannet", "ibims1", "sintel"]),
                     "dsine": ("Normals", ["vkitti2"]), "diode_normals": ("Normals", ["diode"])}
            task, datasets = names[name]
            if name == "marigold_depth":
                receipt = raw_directory(args.raw_root.resolve(), BY_NAME[name]) / "sources.json"
                settings = json.loads(receipt.read_text()).get("settings", {}) if receipt.exists() else {}
                if settings.get("datasets"):
                    datasets = [{"nyuv2": "nyu", "kitti": "kitti_eigen"}.get(n, n) for n in settings["datasets"].split(",")]
            artifacts = [root / "eval" / task / f"{n}.tar" for n in datasets]
            artifacts.append(root / "manifests" / f"{name}.json")
            if name == "marigold_depth" and (root / "manifests/benchmark_splits").exists():
                artifacts += list(files_below(root / "manifests/benchmark_splits"))
            return artifacts
        def convert(force):
            current = list(command)
            flag = "--force-scenes" if name == "matterport3d" else "--overwrite"
            if force and flag not in current:
                current.append(flag)
            from common.node_scratch import workspace, conversion_command
            if any(flag in current for flag in ("--depth-out-root", "--normals-out-root", "--scratch-root", "--shard-root")):
                with workspace(f"{name}-{split}") as scratch:
                    current = conversion_command(current, scratch)
                    run_python(Path(current[0]), current[1:], False)
            else:
                run_python(Path(current[0]), current[1:], False)
        native = run_stage(root, f"{name}_{split}_convert", identity, outputs, convert, args.overwrite)
        if geometry and args.geometry_policy:
            def filtered_outputs():
                return [root / "train" / t / f"{name}.tar" for t in ("Depth", "Normals")] + [root / "manifests" / f"{name}_filtering.json"]
            def materialize(force):
                values = ["--dataset", name, "--source-root", str(native_root / "train"), "--output-root", str(root / "train")]
                if force:
                    values.append("--overwrite")
                run_python(REPO / "scripts/dataset_prep/materialize_geometry_source.py", values, False)
            filter_identity = dict(policy=args.geometry_policy, native=native["artifacts"], converter=identity["converter"])
            run_stage(root, f"{name}_{split}_filter", filter_identity, filtered_outputs, materialize, args.overwrite)


def main() -> int:
    args = parse_args()
    from common.node_scratch import configure_work_dir
    configure_work_dir(getattr(args, "work_dir", None))
    if args.describe:
        try:
            selected = parse_selection(args.describe)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if len(selected) != 1:
            raise SystemExit("--describe accepts exactly one dataset")
        item = selected[0]
        print(f"{item.name}: {item.source}")
        print(f"Raw directory: raw/{item.raw_group}/{item.name}")
        print("Training adapter: " + ("implemented" if item.name in TRAIN_IMPLEMENTED else "TODO"))
        if item.name in TRAIN_OVERRIDES:
            print("Training override keys: " + ", ".join(sorted(TRAIN_OVERRIDES[item.name])))
        print("Evaluation adapter: " + ("implemented" if item.name in EVAL_IMPLEMENTED else "TODO"))
        if item.name in EVAL_OVERRIDES:
            print("Evaluation override keys: " + ", ".join(sorted(EVAL_OVERRIDES[item.name])))
        print(item.note)
        return 0
    if args.list:
        print(list_header())
        for item in CATALOG:
            print(f"{list_row(item, TRAIN_IMPLEMENTED, EVAL_IMPLEMENTED):64} {','.join(item.tasks)}")
        return 0
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.dataset_workers < 1:
        raise SystemExit("--dataset-workers must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise SystemExit("--max-samples must be positive")
    if args.samples_per_epoch < 1:
        raise SystemExit("--samples-per-epoch must be positive")
    if not args.version or Path(args.version).name != args.version or args.version in {".", ".."}:
        raise SystemExit("--version must be one directory name, not a path")
    splits = {part.strip() for part in args.splits.split(",") if part.strip()}
    if not splits or not splits <= {"train", "eval"}:
        raise SystemExit("--splits must contain train and/or eval")
    try:
        overrides = parse_overrides(args.set)
        selected = parse_selection(args.datasets)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    root = args.destination.expanduser().resolve() / args.version
    unsupported = [
        f"{item.name}:{split}"
        for item in selected
        for split in sorted(splits)
        if ((split == "eval" and item.name not in EVAL_IMPLEMENTED)
            or (split == "train" and item.name not in TRAIN_IMPLEMENTED))
    ]
    if unsupported and not args.allow_unsupported:
        print("Not yet implemented for this release CLI: " + ", ".join(unsupported), file=sys.stderr)
        print(
            "No preparation was started. Select implemented pairs shown by --list, "
            "or pass --allow-unsupported to skip TODO entries explicitly.",
            file=sys.stderr,
        )
        return 2

    supported_items = {
        item.name: item
        for item in selected
        if (("eval" in splits and item.name in EVAL_IMPLEMENTED)
            or ("train" in splits and item.name in TRAIN_IMPLEMENTED))
    }
    if not args.skip_staging_check and not args.dry_run:
        staging_failures = []
        raw_root = args.raw_root.expanduser().resolve()
        for item in supported_items.values():
            sources = tuple(source.name for source in acquisition_items([item]))
            if sources and str(overrides.get(item.name, {}).get("allow-missing-hypersim", "")
                               ).lower() in {"1", "true", "yes"}:
                # The pool builder skips the hypersim member on request, so its
                # raw scenes are not required for this run.
                sources = tuple(name for name in sources if name != "hypersim")
            if sources:
                details = tuple(f"{name}: {', '.join(missing)}" for name in sources
                                for ready, missing in [staged_source_status(raw_root, BY_NAME[name])]
                                if not ready)
                ready = not details
            else:
                ready, details = staged_source_status(raw_root, item)
            if not ready:
                staging_failures.append(f"{item.name}: {', '.join(details)}")
        if staging_failures:
            print("Raw sources are not staged:\n  " + "\n  ".join(staging_failures), file=sys.stderr)
            nested = args.raw_root.expanduser().resolve() / "raw"
            if nested.is_dir():
                print(f"Hint: --raw-root should name the raw/ folder itself; did you mean {nested}?",
                      file=sys.stderr)
            print("Run scripts/download_data.py --status for acquisition paths.", file=sys.stderr)
            return 2

    if not args.dry_run:
        for split in ("train", "eval"):
            for task in ("Semseg", "Instanceseg", "Refseg", "ImgEdit", "ImgGeneration", "Depth", "Normals"):
                (root / split / task).mkdir(parents=True, exist_ok=True)

    records = []
    jobs: list[tuple[str, str, list[str]]] = []
    for item in selected:
        raw = raw_directory(args.raw_root.expanduser().resolve(), item)
        done_splits = []
        for split in sorted(splits):
            if f"{item.name}:{split}" in unsupported:
                print(f"TODO {item.name}:{split}: no frozen release adapter. {item.note}")
                continue
            settings = overrides.get(item.name, {})
            try:
                if split == "eval":
                    command = (task_eval_command(item.name, raw, root, args, settings)
                               or geometry_eval_command(item.name, raw, root, args, settings))
                else:
                    command = (pool_train_command(item.name, args.raw_root.expanduser().resolve(), root, args, settings)
                               or geometry_train_command(item.name, raw, root, args, settings))
            except (ValueError, FileNotFoundError) as exc:
                raise SystemExit(f"{item.name}:{split}: {exc}") from exc
            if command is None:  # Keep the declared support sets and builders in lockstep.
                raise RuntimeError(f"Internal error: {item.name}:{split} is declared implemented but has no command")
            done_splits.append(split)
            jobs.append((item.name, split, command))
        if done_splits:
            records.append({"dataset": item.name, "raw_source": item.source, "raw_root": str(raw),
                            "splits": done_splits, "seed": args.seed, "max_samples": args.max_samples,
                            "geometry_policy": args.geometry_policy,
                            "overrides": overrides.get(item.name, {})})

    def execute(job: tuple[str, str, list[str]]) -> None:
        name, split, command = job
        execute_preparation(job, root, args)
        if name == "scannet" and split == "eval" and not args.dry_run:
            shared = root / "work/eval/scannet.tar"
            for task in ("Depth", "Normals"):
                target = root / "eval" / task / "scannet.tar"
                shutil.copy2(shared, target)

    if args.dry_run or args.dataset_workers == 1:
        for job in jobs:
            execute(job)
    else:
        failures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.dataset_workers) as pool:
            future_map = {pool.submit(execute, job): job for job in jobs}
            for future in concurrent.futures.as_completed(future_map):
                try:
                    future.result()
                except Exception as exc:
                    name, split, _ = future_map[future]
                    failures.append(f"{name}:{split}: {exc}")
        if failures:
            print("Preparation failures:\n  " + "\n  ".join(failures), file=sys.stderr)
            return 1

    if not args.dry_run:
        finalize_outputs(root, args, records)
    if unsupported:
        print("\nSkipped unsupported pairs by explicit request: " + ", ".join(unsupported))
    return 0


if __name__ == "__main__":
    if "--config" in sys.argv[1:] or any(v.startswith("--config=") for v in sys.argv[1:]):
        from scripts.setup_data import main as configured
        raise SystemExit(configured([*sys.argv[1:], "--prepare-only"]))
    raise SystemExit(main())
