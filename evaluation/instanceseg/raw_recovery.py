"""Verified, lossless raw-PNG replay for interrupted InstanceSeg shards.

Replays use the same task/batch order as inference. This is not a diffusion
checkpoint: a partly saved batch is regenerated in full, retaining cached RGBs.
Legacy RGBs require the original launcher's submission.log as provenance.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path

from PIL import Image
from evaluation.instanceseg.postprocess import postprocess_protocol


def raw_prediction_path(out_dir, sample, repeat_index, palette_repeats, generation_repeats):
    palette_index, generation_index = divmod(repeat_index, generation_repeats)
    suffix = ""
    if repeat_index:
        if palette_repeats > 1 and generation_repeats > 1:
            suffix = f"_p{palette_index}_g{generation_index}"
        elif generation_repeats > 1:
            suffix = f"_g{generation_index}"
        else:
            suffix = f"_r{palette_index}"
    return Path(out_dir) / sample.subset / "pred_rgb_raw" / f"{sample.image_id}{suffix}.png"


def read_saved_rgb(path):
    if not path.exists():
        return None
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError("expected an RGB PNG")
            image.load()  # Detect truncated/corrupt files, not just a valid header.
            return image.copy()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot replay saved prediction {path}: {exc}. File left untouched.") from exc


def save_rgb_atomic(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            image.save(stream, format="PNG")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def lock_raw_shard(out_dir, rank):
    directory = Path(out_dir) / "raw_recovery"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"rank_{rank:05d}.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns raw-recovery shard {rank}") from exc
        yield


def read_submission(path):
    """Parse data only; never source/eval the submission log as shell code."""
    lines = Path(path).read_text().splitlines()
    exports = [line[len("Export: "):] for line in lines if line.startswith("Export: ")]
    if len(exports) != 1 or not exports[0].startswith("ALL,"):
        raise ValueError(f"Expected exactly one original Export: ALL,... record in {path}")
    config = {}
    for item in exports[0].split(",")[1:]:
        key, value = item.split("=", 1)
        if key in config:
            raise ValueError(f"Duplicate submission setting: {key}")
        config[key] = value
    return config, lines


def _path(value):
    if value is None or str(value) == "":
        return None
    path = Path(value)
    return str(path.resolve()) if path.exists() else str(value)


def recovery_consensus_mode(config):
    """Restore a recorded mode, including runs whose launcher omitted the export."""
    if "ISEG_CONSENSUS_MODE" in config:
        return config["ISEG_CONSENSUS_MODE"]
    modes = set()
    for path in (Path(config["OUT_DIR"]) / "raw_recovery").glob("rank_*.json"):
        saved = json.loads(path.read_text())
        if saved.get("schema_version", 1) >= 3:
            modes.add(saved["settings"]["consensus_mode"])
    if len(modes) > 1:
        raise ValueError("Saved raw protocols disagree on consensus_mode")
    return modes.pop() if modes else "strict_majority"


def raw_protocol(args, shard, samples, positive, *, schema_version=3, image_root=None):
    fields = (
        "steps", "guidance_scale", "generation_seed", "generation_seed_repeats",
        "sample_seed", "prompt_set", "prompt_template", "prompt_template_index",
        "randomize_palette", "randomdraw_repeats", "native_resolution", "batch_size",
        "gate", "limit", "max_instances", "tau", "theta_size", "theta_erosion",
        "gamma", "instance_score", "iou_type",
    )
    settings = {name: getattr(args, name) for name in fields}
    if schema_version >= 2:
        settings["repeat_index_offset"] = getattr(args, "repeat_index_offset", 0)
    if schema_version >= 3:
        settings["consensus_mode"] = args.consensus_mode
    refinement = postprocess_protocol(args)
    if refinement:
        settings["postprocessing"] = refinement
    settings.update(model=_path(args.model), transformer_path=_path(args.transformer_path),
                    prompt_seed=args.prompt_seed if args.prompt_seed is not None else args.sample_seed,
                    legend_len=args.legend_len if args.prompt_set == "legend" else None,
                    notation=args.notation if args.prompt_set == "legend" else None,
                    bg_color=list(args.bg_color))
    digest = hashlib.sha256()
    for sample in samples:
        path = Path(sample.image_path)
        try:
            relative = path.relative_to(args.saco_root)
        except ValueError:
            relative = path  # Explicit external images retain their absolute identity.
        if schema_version >= 2:
            path = relative
        elif image_root is not None:
            if relative.is_absolute():
                raise ValueError("Cannot relocate an image outside the dataset root")
            path = Path(image_root) / relative
        record = [sample.subset, sample.image_id, sample.np_text,
                  str(path), sample.gt_h, sample.gt_w, bool(positive(sample))]
        digest.update(json.dumps(record, separators=(",", ":")).encode())
        digest.update(b"\n")
    palette_root = Path(__file__).resolve().parents[2] / "common/instanceseg"
    palette_hashes = {name: hashlib.sha256((palette_root / name).read_bytes()).hexdigest()
                      for name in ("instance_palette_named.json", "instance_palette.json")}
    return {"schema_version": schema_version, "rank": shard.rank, "world_size": shard.world_size,
            "settings": settings, "sample_sha256": digest.hexdigest(),
            "palette_sha256": palette_hashes}


def validate_legacy_submission(args, shard):
    """Narrow legacy bridge for the logged fixed-LR palette-repeat launchers.

Older runs have no per-shard protocol digest. Require their explicit run spec,
and refuse settings not recorded by that wrapper rather than guessing them.
"""
    if postprocess_protocol(args):
        raise ValueError("Legacy raw replay used no refinement; disable snapping, gapfill and GrabCut "
                         "or use --postprocess_only with a new output directory")
    source = Path(args.out_dir) / "logs/submission.log"
    if not source.is_file():
        raise ValueError("Raw replay needs a matching raw_recovery protocol or original logs/submission.log")
    config, _ = read_submission(source)
    saved_mode = recovery_consensus_mode(config)
    if args.randomdraw_repeats * args.generation_seed_repeats > 1 and saved_mode != args.consensus_mode:
        raise ValueError(f"Raw replay submission mismatch: consensus_mode: saved={saved_mode!r}, "
                         f"requested={args.consensus_mode!r}; use --consensus_mode {saved_mode}")
    if args.prompt_set != "fixeddraw_32_lr" or args.prompt_template is not None:
        raise ValueError("Legacy raw replay currently requires the logged fixeddraw_32_lr prompt contract")
    if args.prompt_seed not in (None, args.sample_seed):
        raise ValueError("Legacy submission did not record a separate prompt_seed")
    if (list(args.bg_color), args.theta_size, args.theta_erosion, args.gamma) != ([0, 0, 0], 2e-4, 0.1, 5.0):
        raise ValueError("Legacy submission only supports the original default decoder settings (except tau)")
    expected = {
        "NUM_SHARDS": str(shard.world_size), "STEPS": str(args.steps),
        "GENERATION_SEED": str(args.generation_seed),
        "GENERATION_SEED_REPEATS": str(args.generation_seed_repeats),
        "SAMPLE_SEED": str(args.sample_seed), "PROMPT_SET": args.prompt_set,
        "ISEG_RANDOMDRAW_REPEATS": str(args.randomdraw_repeats),
        "RANDOMDRAW_REPEATS": str(args.randomdraw_repeats),
        "RANDOMIZE_PALETTE": str(int(args.randomize_palette)),
        "NATIVE_RESOLUTION": str(int(args.native_resolution)),
        "ISEG_BATCH_SIZE": str(args.batch_size), "SAVE_RGB": "1", "GATE": args.gate,
        "LIMIT": "" if args.limit is None else str(args.limit),
        "MAX_INSTANCES": "" if args.max_instances is None else str(args.max_instances),
        "SUBSETS": " ".join(args.subsets or []),
        "PROMPT_TEMPLATE_INDEX": "" if args.prompt_template_index is None else str(args.prompt_template_index),
        "IOU_TYPE": args.iou_type,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Raw replay submission mismatch: {key}: saved={config.get(key)!r}, requested={value!r}")
    for key, value in {"TAU": args.tau, "GUIDANCE_SCALE": args.guidance_scale,
                       "INSTANCE_SCORE": args.instance_score}.items():
        if key not in config or float(config[key]) != value:
            raise ValueError(f"Raw replay submission mismatch: {key}")
    for key, value in {"OUT_DIR": args.out_dir, "MODEL": args.model,
                       "LOCAL_MODEL": args.model, "TRANSFORMER_PATH": args.transformer_path,
                       "SACO_ROOT": args.saco_root, "GATE_FILE": args.gate_file}.items():
        if key not in config or _path(config[key]) != _path(value):
            raise ValueError(f"Raw replay submission mismatch: {key}")


def logged_saco_roots(out_dir, shard):
    """Read only each shard log's short header; never trust paths as shell code."""
    roots = set()
    for log in (Path(out_dir) / "logs").glob(f"eval-iseg-*-shard-{shard.rank}.log"):
        with log.open() as stream:
            header = [stream.readline() for _ in range(32)]
        if not any(line.strip() == f"SHARD:       {shard.rank} / {shard.world_size}" for line in header):
            continue
        for line in header:
            if line.strip().startswith("saco root:"):
                roots.add(line.split("saco root:", 1)[1].split("    (tar:", 1)[0].strip())
    return roots


def verify_raw_protocol(args, shard, protocol, samples=None, positive=None):
    path = Path(args.out_dir) / "raw_recovery" / f"rank_{shard.rank:05d}_of_{shard.world_size:05d}.json"
    upgraded = path.with_suffix(".v3.json")
    if upgraded.exists():
        path = upgraded
    elif path.with_suffix(".v2.json").exists():
        path = path.with_suffix(".v2.json")
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("schema_version") in (1, 2) and protocol.get("schema_version") == 3:
            # Both older schemas predate the mode switch and used strict majority.
            if args.randomdraw_repeats * args.generation_seed_repeats > 1 and args.consensus_mode != "strict_majority":
                raise ValueError("Raw replay protocol mismatch: legacy repeats used strict_majority; "
                                 "use --consensus_mode strict_majority")
            if saved["schema_version"] == 2:
                legacy = {**protocol, "schema_version": 2,
                          "settings": {key: value for key, value in protocol["settings"].items()
                                       if key != "consensus_mode"}}
                if legacy != saved:
                    raise ValueError("Raw replay protocol mismatch: legacy settings/samples/gate/palette differ")
                return upgraded
        if saved.get("schema_version") == 1 and protocol.get("schema_version") in (2, 3):
            if samples is None or positive is None or getattr(args, "repeat_index_offset", 0):
                raise ValueError("Raw replay protocol mismatch: cannot verify legacy sample paths")
            # Validate the EXACT old digest, substituting only a logged staging
            # prefix. This does not waive sample, gate, palette or settings checks.
            roots = [None, *sorted(logged_saco_roots(args.out_dir, shard))]
            for root in roots:
                legacy = raw_protocol(args, shard, samples, positive, schema_version=1, image_root=root)
                if legacy == saved:
                    return upgraded  # Preserve the original v1 provenance file.
            raise ValueError("Raw replay protocol mismatch: legacy settings/samples/gate/palette differ")
        if saved != protocol:
            changed = [key for key in protocol if saved.get(key) != protocol[key]]
            raise ValueError(f"Raw replay protocol mismatch ({', '.join(changed)}); saved files left untouched")
    elif args.resume_from_rgb:
        validate_legacy_submission(args, shard)
    return path
