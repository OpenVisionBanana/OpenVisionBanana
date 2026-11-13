#!/usr/bin/env python3
"""Download and prepare OVB data from a YAML recipe or the command line.

Raw route: official sources go to <data_root>/raw/<Group>/<dataset>/ and are
prepared into <data_root>/prepared/<version>/{train,eval}/<Task>/. Covers the
segmentation, editing and generation benchmarks, the zero-shot training pool
(zeroshot_pool) and the depth/normals converters. Complete ZS and NZS recipes
use official sources and author-provided benchmark assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from common.release_data import (CATALOG, acquisition_items, available_acquisition_items, list_header, list_row, parse_selection,
                                 raw_directory, staged_source_status)
from scripts import prepare_data

DEFAULTS = dict(data_root="data", version="V1_prepared_seg_geo",
                datasets=["hypersim", "taskonomy"], splits=["train"], seed=42,
                download_workers=4, prepare_workers=32, dataset_workers=3,
                max_samples=None, samples_per_epoch=1_000_000, overwrite=False,
                preparation={})
DEFAULTS.update(selections=None, acquisition={}, geometry_policy=None, training_pool=None, work_dir=None)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO / "configs/data_zs.yaml")
    p.add_argument("--datasets", help="One name, comma-separated names, or all; unsupported pairs fail. "
                                      "A training pool (zeroshot_pool) also downloads its sources.")
    p.add_argument("--splits", help="train, eval, or train,eval")
    p.add_argument("--data-root", type=Path)
    p.add_argument("--work-dir", type=Path,
                   help="Existing writable directory for temporary files; defaults to the system temporary directory.")
    p.add_argument("--route", choices=("raw", "published"))
    p.add_argument("--seed", type=int)
    for name in ("download-workers", "prepare-workers", "dataset-workers", "max-samples", "samples-per-epoch"):
        p.add_argument("--" + name, type=int)
    p.add_argument("--set", action="append", default=[], metavar="DATASET.KEY=VALUE")
    p.add_argument("--overwrite", action="store_true", default=None)
    p.add_argument("--list", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--download-only", action="store_true", help="Acquire the configured official sources")
    mode.add_argument("--prepare-only", action="store_true", help="Prepare already staged sources")
    return p


def names(value, field):
    if isinstance(value, str):
        result = [v.strip() for v in value.split(",") if v.strip()]
    elif isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value):
        result = [v.strip() for v in value]
    else:
        raise ValueError(f"{field} must be a name/list or comma-separated names")
    if not result:
        raise ValueError(f"{field} cannot be empty")
    return list(dict.fromkeys(result))


def setting(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, list) and all(isinstance(v, (str, int, float)) for v in value):
        return ",".join(map(str, value))
    raise ValueError("Preparation settings must be scalar values or lists of scalars")


def resolve(args):
    try:
        import yaml
    except ImportError as exc:
        raise ValueError("Install PyYAML from requirements.txt to read the configuration") from exc
    try:
        with args.config.expanduser().open(encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {args.config}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("Configuration must be a YAML mapping")
    route = args.route or loaded.get("route", "raw")
    if route == "published":
        raise ValueError("The temporary OVB published-data route is retired. Use configs/data_zs.yaml or configs/data_nzs.yaml with official sources.")
    if route != "raw":
        raise ValueError("route must be raw or published")
    loaded = dict(loaded)
    loaded.pop("route", None)
    unknown = set(loaded) - DEFAULTS.keys()
    if unknown:
        raise ValueError("Unknown configuration keys: " + ", ".join(sorted(map(str, unknown))))
    cfg = {**DEFAULTS, **loaded}
    for key in DEFAULTS:
        if key != "preparation" and getattr(args, key, None) is not None:
            value = getattr(args, key)
            cfg[key] = str(value) if isinstance(value, Path) else value
    if cfg["selections"] is not None:
        if not isinstance(cfg["selections"], dict) or not cfg["selections"] or not set(cfg["selections"]) <= {"train", "eval"}:
            raise ValueError("selections must map train/eval to dataset lists")
        cfg["selections"] = {s: [i.name for i in parse_selection(",".join(names(v, "selections."+s)))]
                             for s, v in cfg["selections"].items()}
        if args.datasets:
            cfg["selections"] = None
            cfg["training_pool"] = None
        else:
            if args.splits:
                requested = names(args.splits, "splits")
                if not set(requested) <= set(cfg["selections"]):
                    raise ValueError("Requested split is absent from this recipe")
                cfg["selections"] = {s: cfg["selections"][s] for s in requested}
            cfg["datasets"] = list(dict.fromkeys(n for v in cfg["selections"].values() for n in v))
            cfg["splits"] = list(cfg["selections"])
    cfg["datasets"] = names(cfg["datasets"], "datasets")
    selected = parse_selection(",".join(cfg["datasets"]))
    cfg["datasets"] = [item.name for item in selected]
    cfg["splits"] = names(cfg["splits"], "splits")
    if not set(cfg["splits"]) <= {"train", "eval"}:
        raise ValueError("splits must contain train and/or eval")
    for key in ("download_workers", "prepare_workers", "dataset_workers", "samples_per_epoch", "max_samples", "seed"):
        value = cfg[key]
        if key == "max_samples" and value is None:
            continue
        if type(value) is not int or (key != "seed" and value < 1):
            raise ValueError(f"{key} must be {'an integer' if key == 'seed' else 'a positive integer'}")
    if type(cfg["overwrite"]) is not bool:
        raise ValueError("overwrite must be true or false")
    version = cfg["version"]
    if not isinstance(version, str) or not version or Path(version).name != version or version in {".", ".."}:
        raise ValueError("version must be a single directory name")
    if not isinstance(cfg["data_root"], str) or not cfg["data_root"].strip():
        raise ValueError("data_root must be a nonempty path")
    cfg["data_root"] = str(Path(cfg["data_root"]).expanduser().resolve())
    if cfg["work_dir"] is not None:
        if not isinstance(cfg["work_dir"], str) or not cfg["work_dir"].strip():
            raise ValueError("work_dir must be a nonempty path or null")
        from common.node_scratch import validate_work_dir
        cfg["work_dir"] = str(validate_work_dir(cfg["work_dir"]))
    prep = cfg["preparation"]
    if not isinstance(prep, dict):
        raise ValueError("preparation must be a dataset-to-settings mapping")
    overrides = []
    for dataset, values in prep.items():
        if not isinstance(dataset, str) or not isinstance(values, dict):
            raise ValueError("preparation must map dataset names to setting mappings")
        for key, value in values.items():
            if not isinstance(key, str):
                raise ValueError("Preparation keys must be strings")
            overrides.append(f"{dataset}.{key}={setting(value)}")
    resolved = prepare_data.parse_overrides(overrides + args.set)
    for dataset, values in resolved.items():
        allowed = prepare_data.TRAIN_OVERRIDES.get(dataset, set()) | prepare_data.EVAL_OVERRIDES.get(dataset, set())
        prepare_data.add_overrides([], values, allowed)
        if values.get("delete-raw-scenes-after-convert", "false").lower() not in {"false", "no", "off"}:
            raise ValueError("setup_data never deletes raw sources; delete-raw-scenes-after-convert is prohibited")
    cfg["preparation"] = {key: resolved[key] for key in cfg["datasets"] if key in resolved}
    if "replica" in cfg["datasets"] and "train" in cfg["splits"]:
        replica = cfg["preparation"].setdefault("replica", {})
        if replica.get("allow-vertex-color-rgb", "false").lower() not in {"true", "yes", "on"} or replica.get("ptex-renderer"):
            path = Path(replica.get("ptex-renderer", "tools/replica/renderer/ReplicaSDK/ReplicaRenderer")).expanduser()
            replica["ptex-renderer"] = str((Path(cfg["data_root"]) / path).resolve())
    pairs = [(item, split) for item in selected for split in cfg["splits"]
             if cfg["selections"] is None or item.name in cfg["selections"].get(split, [])]
    unsupported = [f"{item.name}:{split}" for item, split in pairs
                   if item.name not in (prepare_data.TRAIN_IMPLEMENTED if split == "train" else prepare_data.EVAL_IMPLEMENTED)]
    if unsupported:
        raise ValueError("Unsupported dataset/split pairs: " + ", ".join(unsupported) + ". Use --list; nothing started.")
    for item in selected:
        for split in (s for candidate, s in pairs if candidate.name == item.name):
            allowed = (prepare_data.TRAIN_OVERRIDES if split == "train" else prepare_data.EVAL_OVERRIDES).get(item.name, set())
            prepare_data.add_overrides([], cfg["preparation"].get(item.name, {}), allowed)
        replica = cfg["preparation"].get("replica", {})
        if item.name == "replica" and not replica.get("ptex-renderer") and replica.get("allow-vertex-color-rgb", "false").lower() not in {"true", "yes", "on"}:
            raise ValueError("Replica requires --set replica.ptex-renderer=/path/to/ReplicaRenderer. See DATASET.md for the official SDK build.")
    if cfg["geometry_policy"] not in {None, "base-clean"}:
        raise ValueError("geometry_policy must be base-clean")
    if cfg["training_pool"] not in {None, "zs", "nzs"}:
        raise ValueError("training_pool must be zs or nzs")
    if cfg["training_pool"] and "train" in cfg["splits"]:
        from scripts.dataset_prep.finalize_training_recipe import ZS_SOURCES, NZS_SOURCES
        expected = ZS_SOURCES if cfg["training_pool"] == "zs" else NZS_SOURCES
        pool = "zeroshot_pool" if cfg["training_pool"] == "zs" else "nonzeroshot_pool"
        training = set(cfg["selections"].get("train", [])) if cfg["selections"] else set(cfg["datasets"])
        if not expected | {pool} <= training or cfg["geometry_policy"] != "base-clean":
            raise ValueError("A complete training recipe requires its matching 2D pool, all geometry sources and geometry_policy: base-clean")
    acquisition = cfg["acquisition"]
    if not isinstance(acquisition, dict):
        raise ValueError("acquisition must map source names to settings")
    for source, values in acquisition.items():
        if source not in {i.name for i in CATALOG} or not isinstance(values, dict):
            raise ValueError("Invalid acquisition source/settings: " + str(source))
        allowed = {"hypersim": {"scenes", "frames"}, "taskonomy": {"num-buildings", "seed", "frames"}, "marigold_normals": {"frames"}, "marigold_depth": {"frames", "datasets"}, "pico_banana": {"limit"}, "pd12m": {"count", "scan-limit", "workers"}}
        if set(values) - allowed.get(source, set()):
            raise ValueError("Unknown acquisition settings for " + source)
        if source == "hypersim" and "frames" in values:
            if int(values["frames"]) < 1 or values.get("scenes", "all") == "all" or cfg["training_pool"]:
                raise ValueError("Hypersim smoke frames require an explicit scene and a standalone smoke recipe")
        if source == "taskonomy" and "frames" in values:
            if int(values["frames"]) < 1 or values.get("num-buildings", "all") == "all" or cfg["training_pool"]:
                raise ValueError("Taskonomy smoke frames require bounded buildings and a standalone smoke recipe")
        if source in {"marigold_normals", "marigold_depth"} and "frames" in values and (int(values["frames"]) < 1 or cfg["training_pool"]):
            raise ValueError("Normal benchmark smoke frames require a positive count and a standalone smoke recipe")
        if source == "pico_banana" and (int(values.get("limit", 1)) < 1 or cfg["training_pool"]):
            raise ValueError("A Pico download limit is allowed only in a separate smoke recipe")
        if source == "pd12m" and (any(int(v) < 1 for v in values.values()) or cfg["training_pool"]):
            raise ValueError("PD12M smoke acquisition requires positive settings and a separate smoke recipe")
    return cfg, selected




def commands(cfg):
    root = Path(cfg["data_root"])
    download = [sys.executable, str(REPO / "scripts/download_data.py"), "--destination", str(root / "raw"),
                "--datasets", ",".join(cfg["datasets"]), "--workers", str(cfg["download_workers"])]
    prepare = [sys.executable, str(REPO / "scripts/prepare_data.py"), "--raw-root", str(root / "raw"),
               "--destination", str(root / "prepared"), "--version", cfg["version"],
               "--datasets", ",".join(cfg["datasets"]), "--splits", ",".join(cfg["splits"]),
               "--seed", str(cfg["seed"]), "--workers", str(cfg["prepare_workers"]),
               "--dataset-workers", str(cfg["dataset_workers"]), "--samples-per-epoch", str(cfg["samples_per_epoch"])]
    if cfg["max_samples"] is not None:
        prepare += ["--max-samples", str(cfg["max_samples"])]
    if cfg["overwrite"]:
        prepare.append("--overwrite")
    for dataset, settings in cfg["preparation"].items():
        for key, value in sorted(settings.items()):
            prepare += ["--set", f"{dataset}.{key}={value}"]
    sources = {i.name for i in acquisition_items(parse_selection(",".join(cfg["datasets"])))}
    for dataset, settings in cfg.get("acquisition", {}).items():
        if dataset in sources:
            for key, value in settings.items():
                download += ["--set", f"{dataset}.{key}={setting(value)}"]
    if cfg.get("geometry_policy"):
        prepare += ["--geometry-policy", cfg["geometry_policy"]]
    if cfg.get("work_dir"):
        for command in (download, prepare):
            command += ["--work-dir", cfg["work_dir"]]
    return download, prepare


def preparation_commands(cfg):
    if cfg.get("selections") is None:
        return [commands(cfg)[1]]
    result = []
    for split, datasets in cfg["selections"].items():
        group = dict(cfg, datasets=datasets, splits=[split],
                     preparation={k: v for k, v in cfg["preparation"].items() if k in datasets})
        result.append(commands(group)[1])
    return result


def run(command):
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def publish_config(cfg, config_path):
    payload = {"status": "preparation_succeeded", "config_file": str(config_path.expanduser().resolve()),
               "resolved": cfg, "route": cfg.get("route", "official_raw_conversion")}
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(data).hexdigest()
    folder = Path(cfg["data_root"]) / "prepared" / cfg["version"] / "manifests"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"setup_data-{digest}.json"
    # Unique content-addressed records preserve earlier successful invocations.
    for target, content in ((path, data), (Path(str(path) + ".sha256"), f"{digest}  {path.name}\n".encode())):
        fd, temporary = tempfile.mkstemp(prefix=".setup-", dir=folder)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    print(f"Prepared successfully; resolved configuration: {path}")


def main(argv=None):
    args = parser().parse_args(argv)
    if args.list:
        print(list_header())
        for item in CATALOG:
            print(list_row(item, prepare_data.TRAIN_IMPLEMENTED, prepare_data.EVAL_IMPLEMENTED))
        return 0
    try:
        cfg, selected = resolve(args)
        raw = Path(cfg["data_root"]) / "raw"
        missing = []
        sources = acquisition_items(selected)
        for item in sources:
            receipt = raw_directory(raw, item) / "sources.json"
            if receipt.is_file():
                recorded = json.loads(receipt.read_text()).get("settings", {})
                expected = {k: setting(v) for k, v in cfg["acquisition"].get(item.name, {}).items()}
                smoke_keys = {"frames", "limit", "count"} & recorded.keys()
                if smoke_keys and any(str(recorded[k]) != expected.get(k) for k in smoke_keys):
                    raise ValueError(f"{item.name} contains smoke-only source assets. Use a separate data root for the full recipe.")
        from common.official_sources import missing_manual_assets
        for item, directory, details in missing_manual_assets(raw, sources):
            missing.append(item)
            print(f"\nMissing manual assets: {item.name}\nSource: {item.source}\n{item.note}\nRequired: {', '.join(details)}\nPlace assets in: {directory}", file=sys.stderr)
        download, prepare = commands(cfg)
        preparations = preparation_commands(cfg)
        if missing:
            rerun = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
            print("\nAfter staging all missing assets, rerun:\n" + shlex.join(rerun), file=sys.stderr)
            if args.dry_run:
                print("Planned download: " + shlex.join(download))
                for command in preparations:
                    print("Planned preparation: " + shlex.join(command))
            elif not args.prepare_only and available_acquisition_items(sources, {item.name for item in missing}):
                try:
                    run(download)
                except subprocess.CalledProcessError as exc:
                    if exc.returncode != 2:
                        raise
                print("Available public sources were processed. Preparation awaits the missing gated inputs.", file=sys.stderr)
            return 2
        if args.dry_run:
            print("Planned download: " + shlex.join(download))
            if not args.download_only:
                for command in preparations:
                    run(command + ["--dry-run"])
            return 0
        if not args.download_only and cfg.get("training_pool") == "nzs" and "train" in cfg["splits"]:
            from scripts.dataset_prep.finalize_training_recipe import ensure_diode_convention_verified
            ensure_diode_convention_verified()
        replica = cfg["preparation"].get("replica", {})
        if not args.download_only and "replica" in cfg["datasets"] and "train" in cfg["splits"] and replica.get("ptex-renderer"):
            renderer = Path(replica["ptex-renderer"]).expanduser().resolve()
            if not renderer.is_file() or not os.access(renderer, os.X_OK):
                raise ValueError(f"Missing Replica renderer: {renderer}. Build it with: " + shlex.join([
                    "bash", str(REPO / "scripts/dataset_prep/build_replica_ptex.sh"),
                    str(Path(cfg["data_root"]) / "tools/replica")]) + ". Or set replica.ptex-renderer to an existing executable.")
        if not args.prepare_only:
            run(download)
        failures = [item.name for item in sources if not staged_source_status(raw, item)[0]]
        if failures:
            raise ValueError("Sources still missing after download: " + ", ".join(failures))
        if args.download_only:
            print("Official sources are staged. Run the same recipe with --prepare-only next.")
            return 0
        for command in preparations:
            run(command)
        if cfg.get("training_pool") and "train" in cfg["splits"]:
            run([sys.executable, str(REPO / "scripts/dataset_prep/finalize_training_recipe.py"),
                 "--root", str(Path(cfg["data_root"]) / "prepared" / cfg["version"]),
                 "--pool", cfg["training_pool"], "--seed", str(cfg["seed"]),
                 "--samples-per-epoch", str(cfg["samples_per_epoch"])])
        publish_config(cfg, args.config)
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
