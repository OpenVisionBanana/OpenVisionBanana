#!/usr/bin/env python3
"""Check paths, dependencies, models, data, and artifacts before expensive work."""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import shutil
from pathlib import Path


GEOMETRY_SUITES = {
    "full": {
        "depth-eval": (
            "depth",
            (
                "nyu",
                "ibims1",
                "diode",
                "kitti",
                "kitti_eigen",
                "eth3d",
                "nuscenes",
                "vkitti2",
                "scannet",
            ),
        ),
        "normals-eval": (
            "surface_normals",
            (
                "nyu",
                "ibims1",
                "diode",
                "kitti",
                "eth3d",
                "nuscenes",
                "vkitti2",
                "scannet",
                "sintel",
            ),
        ),
    },
    "ablation": {
        "depth-eval": ("depth", ("nuscenes", "scannet", "vkitti2")),
        "normals-eval": (
            "surface_normals",
            ("nuscenes", "scannet", "vkitti2"),
        ),
    },
}


def geometry_archive_filename(
    suite: str,
    dataset: str,
    *,
    ibims1_depth_protocol: str,
    ibims1_normals_protocol: str,
) -> str:
    """Return the installer filename selected by the evaluator protocol."""

    if dataset != "ibims1":
        return f"{dataset}.tar"
    if suite == "depth-eval":
        protocol = ibims1_depth_protocol
    elif suite == "normals-eval":
        protocol = ibims1_normals_protocol
    else:
        raise ValueError(f"Unsupported geometry suite: {suite}")
    return "ibims1_official.tar" if protocol == "ibims_official" else "ibims1.tar"


def validate_specs(path: Path) -> list[str]:
    problems = []
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read dataset specs {path}: {exc}"]
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        return [f"dataset specs contain no datasets: {path}"]
    def resolve_entry_path(value: object) -> Path:
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        return candidate.resolve()

    for item in datasets:
        name = item.get("name", "<unnamed>") if isinstance(item, dict) else "<invalid>"
        if not isinstance(item, dict):
            problems.append(f"invalid dataset entry: {item!r}")
            continue
        kind = item.get("type", "prepared_folder")
        if kind == "prepared_folder":
            root = resolve_entry_path(item.get("path", ""))
            if not (root / "metadata.jsonl").is_file():
                problems.append(f"{name}: prepared metadata missing under {root}")
        elif kind == "indexed_prepared_tar":
            for field in ("path", "index_path"):
                candidate = resolve_entry_path(item.get(field, ""))
                if not candidate.is_file():
                    problems.append(f"{name}: {field} missing: {candidate}")
        elif kind == "webdataset_text_to_image":
            patterns = item.get("shards", [])
            patterns = [patterns] if isinstance(patterns, str) else patterns
            resolved_patterns = [str(resolve_entry_path(pattern)) for pattern in patterns]
            if not patterns or not any(glob.glob(pattern) for pattern in resolved_patterns if "{" not in pattern):
                if any("{" in pattern for pattern in resolved_patterns):
                    continue  # Brace expansion is validated by the trainer.
                problems.append(f"{name}: no WebDataset shards match {patterns}")
        else:
            problems.append(f"{name}: unsupported dataset type {kind!r}")
    return problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("setup", "train", "depth-eval", "normals-eval", "genai-eval", "all"),
        default="setup",
    )
    parser.add_argument("--root", type=Path, help="Storage root; defaults to OVB_STORAGE_ROOT.")
    parser.add_argument("--model-family", choices=("klein",), default=os.environ.get("MODEL_FAMILY", "klein"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dataset-specs", type=Path)
    parser.add_argument("--transformer-path", type=Path)
    parser.add_argument("--lora-path", type=Path)
    parser.add_argument("--prompts-jsonl", type=Path)
    parser.add_argument("--archive-root", type=Path, help="Geometry archive root; defaults to PREPARED_ROOT.")
    parser.add_argument(
        "--geometry-profile",
        choices=("full", "ablation"),
        default=os.environ.get("GEOMETRY_PROFILE", "full"),
        help="Geometry release profile whose archive set must be present.",
    )
    parser.add_argument(
        "--ibims1-depth-protocol",
        choices=("nyu_10m", "ibims_official"),
        default=os.environ.get("IBIMS1_DEPTH_PROTOCOL", "ibims_official"),
    )
    parser.add_argument(
        "--ibims1-normals-protocol",
        choices=("nyu_10m", "ibims_official"),
        default=os.environ.get("IBIMS1_NORMALS_PROTOCOL", "ibims_official"),
    )
    parser.add_argument(
        "--explicit-paths",
        action="store_true",
        help="Validate explicitly supplied task paths without requiring a storage root.",
    )
    parser.add_argument("--strict-data", action="store_true", help="Require every archive in an evaluation suite.")
    parser.add_argument("--create-dirs", action="store_true", help="Create the derived storage directories.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_value = args.root or (Path(os.environ["OVB_STORAGE_ROOT"]) if os.environ.get("OVB_STORAGE_ROOT") else None)
    if root_value is None and not args.explicit_paths:
        print("FAIL storage root: set OVB_STORAGE_ROOT or pass --root")
        return 2
    if root_value is None:
        storage = None
        raw = Path(os.environ.get("RAW_ROOT", Path.cwd())).expanduser().resolve()
        if args.archive_root:
            prepared = args.archive_root.expanduser().resolve()
        else:
            prepared = Path(
                os.environ.get("PREPARED_ROOT", Path.cwd())
            ).expanduser().resolve()
        if args.model:
            models = args.model.expanduser().resolve().parent
        else:
            models = Path(
                os.environ.get("MODEL_ROOT", Path.cwd())
            ).expanduser().resolve()
        outputs = Path(os.environ.get("OUTPUT_ROOT", Path.cwd())).expanduser().resolve()
        derived: tuple[Path, ...] = ()
    else:
        storage = root_value.expanduser().resolve()
    if args.root:
        raw, prepared, models, outputs = (
            storage / "raw",
            storage / "prepared",
            storage / "models",
            storage / "outputs",
        )
    elif storage is not None:
        raw = Path(os.environ.get("RAW_ROOT", storage / "raw")).expanduser().resolve()
        prepared = Path(os.environ.get("PREPARED_ROOT", storage / "prepared")).expanduser().resolve()
        models = Path(os.environ.get("MODEL_ROOT", storage / "models")).expanduser().resolve()
        outputs = Path(os.environ.get("OUTPUT_ROOT", storage / "outputs")).expanduser().resolve()
    if storage is not None:
        derived = (raw, prepared, models, outputs)
    if args.create_dirs:
        for path in derived:
            path.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    warnings: list[str] = []

    print(f"storage root: {storage if storage is not None else 'explicit task paths'}")
    for path in derived:
        state = "ready" if path.is_dir() else "missing"
        detail = ""
        if path.is_dir():
            detail = f" ({shutil.disk_usage(path).free / 1024**3:.1f} GiB free)"
        print(f"  {state:7} {path}{detail}")
        if not path.is_dir():
            failures.append(f"directory does not exist: {path}")
        elif not os.access(path, os.W_OK | os.X_OK):
            failures.append(f"directory is not writable: {path}")

    tmpdir = Path(os.environ.get("TMPDIR", "/tmp")).expanduser()
    if args.create_dirs:
        tmpdir.mkdir(parents=True, exist_ok=True)
    if not tmpdir.is_dir() or not os.access(tmpdir, os.W_OK | os.X_OK):
        failures.append(f"TMPDIR is missing or not writable: {tmpdir}")
    else:
        print(f"scratch: {tmpdir.resolve()} ({shutil.disk_usage(tmpdir).free / 1024**3:.1f} GiB free)")

    profiles = (
        {args.profile}
        if args.profile != "all"
        else {"setup", "train", "depth-eval", "normals-eval", "genai-eval"}
    )
    modules = set()
    if "setup" in profiles:
        for command in ("bash", "tar"):
            if shutil.which(command) is None:
                failures.append(f"required setup command is missing: {command}")
        if not any(shutil.which(command) for command in ("wget", "curl")):
            warnings.append("neither wget nor curl is available; automatic acquisition cannot run")
        for command in ("git", "unzip"):
            if shutil.which(command) is None:
                warnings.append(f"optional setup command is missing: {command}")
        if importlib.util.find_spec("datasets") is None:
            warnings.append(
                "Python module 'datasets' is missing; install project requirements before caching SteerViT"
            )
    if "train" in profiles:
        modules.update(("torch", "accelerate", "diffusers", "transformers", "PIL", "numpy"))
    if profiles & {"depth-eval", "normals-eval"}:
        modules.update(("torch", "diffusers", "PIL", "numpy", "cv2"))
    if "genai-eval" in profiles:
        modules.update(("torch", "diffusers", "PIL"))
        if os.environ.get("RUN_SCORE", "1").lower() not in {"0", "false", "no", "off"}:
            if os.environ.get("SCORE_SIF"):
                warnings.append("t2v_metrics must be available in SCORE_SIF; it is not checked in the generation runtime")
            else:
                modules.add("t2v_metrics")
    for module in sorted(modules):
        if importlib.util.find_spec(module) is None:
            failures.append(f"Python module is missing: {module}")

    if args.transformer_path and args.lora_path:
        failures.append("transformer and LoRA overrides are mutually exclusive")
    for label, path in (("transformer", args.transformer_path), ("LoRA", args.lora_path)):
        if path and not path.expanduser().exists():
            failures.append(f"{label} artifact not found: {path}")
    if args.transformer_path and args.transformer_path.expanduser().exists():
        artifact = args.transformer_path.expanduser()
        candidates = (artifact, artifact / "transformer")
        if not any((candidate / "config.json").is_file() for candidate in candidates):
            failures.append(f"transformer artifact has no config.json: {artifact}")
    if args.lora_path and args.lora_path.expanduser().exists():
        artifact = args.lora_path.expanduser()
        if not artifact.is_dir() or not list(artifact.glob("*lora*.safetensors")):
            failures.append(f"LoRA artifact has no standard *lora*.safetensors file: {artifact}")

    if profiles != {"setup"}:
        configured_model = args.model or (
            Path(os.environ["MODEL"]) if os.environ.get("MODEL") else models / "FLUX.2-klein-base-9B"
        )
        configured_model = configured_model.expanduser().resolve()
        print(f"model: {configured_model}")
        if not configured_model.is_dir() or not (configured_model / "model_index.json").is_file():
            failures.append(f"complete local {args.model_family} model snapshot not found: {configured_model}")

    if "train" in profiles:
        specs = args.dataset_specs or (Path(os.environ["DATASET_SPECS"]) if os.environ.get("DATASET_SPECS") else None)
        if specs is None:
            warnings.append("DATASET_SPECS is not selected yet")
        elif not specs.expanduser().is_file():
            failures.append(f"dataset specs not found: {specs}")
        else:
            failures.extend(validate_specs(specs.expanduser()))

    geometry_suites = GEOMETRY_SUITES[args.geometry_profile]
    for profile in profiles & geometry_suites.keys():
        folder, names = geometry_suites[profile]
        if args.archive_root:
            archive_root = args.archive_root.expanduser().resolve()
        elif args.geometry_profile == "ablation":
            archive_root = prepared / "profiles" / "ablation"
        else:
            archive_root = prepared
        ready = []
        missing = []
        for name in names:
            archive_name = geometry_archive_filename(
                profile,
                name,
                ibims1_depth_protocol=args.ibims1_depth_protocol,
                ibims1_normals_protocol=args.ibims1_normals_protocol,
            )
            path = archive_root / folder / archive_name
            (ready if path.is_file() else missing).append(name)
        print(f"{profile}: ready={','.join(ready) or 'none'}")
        if missing:
            print(f"{profile}: missing={','.join(missing)}")
            message = f"{profile} missing archives: {', '.join(missing)}"
            (failures if args.strict_data else warnings).append(message)
        if not ready:
            failures.append(f"{profile} has no prepared archives")

    if "genai-eval" in profiles:
        prompts = args.prompts_jsonl or (
            Path(os.environ["PROMPTS_JSONL"])
            if os.environ.get("PROMPTS_JSONL")
            else prepared / "genai-bench" / "prompts.jsonl"
        )
        source = Path(os.environ.get("GENAI_SOURCE_DIR", raw / "genai-bench"))
        if not prompts.expanduser().is_file() and not (source / "genai_image.json").is_file():
            failures.append(
                f"GenAI prompts missing: provide {prompts} or source metadata under {source}"
            )
        if os.environ.get("RUN_SCORE", "1").lower() not in {"0", "false", "no", "off"}:
            score_model_value = os.environ.get("SCORE_MODEL_PATH")
            if score_model_value:
                score_model = Path(score_model_value).expanduser().resolve()
                if not score_model.is_dir() or not (score_model / "config.json").is_file():
                    failures.append(f"local GenAI scorer snapshot not found: {score_model}")
            else:
                warnings.append(
                    "SCORE_MODEL_PATH is unset; ensure the evaluator model already exists in the scoring environment's Hugging Face cache"
                )

    for warning in warnings:
        print(f"WARN {warning}")
    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        print(f"Preflight failed with {len(failures)} problem(s).")
        return 1
    print("Preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
