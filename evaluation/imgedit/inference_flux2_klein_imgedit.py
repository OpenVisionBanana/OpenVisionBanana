"""Generate Flux2-Klein outputs in the official ImgEdit-Bench layout.

Official ImgEdit quantitative suites:
- Basic-Bench: metadata in ImgEdit/Benchmark/Basic/basic_edit.json, source
  images under Benchmark/singleturn, predictions named <key>.png.
- UGE-Bench: metadata in ImgEdit/Benchmark/UGE/UGE_edit.json, source images
  under Benchmark/hard, predictions named <key>.png.

The official Multi-Turn bench is generation/manual-judgment only, so this file
can generate multi-turn outputs but does not report an official score for them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image
from tqdm import tqdm


REPO_IMGEDIT_ROOT = Path(__file__).resolve().parent / "ImgEdit"
DEFAULT_BENCHMARK_ROOT = "/mnt/AI_Dataset/_unzipped/imgedit_data/Benchmark"


@dataclass(frozen=True)
class BenchSample:
    key: str
    source_image: Path
    prompt: str
    suite: str
    edit_type: Optional[str] = None
    turn_prompts: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShardInfo:
    rank: int
    world_size: int
    local_rank: int
    device: str
    distributed: bool


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
    return rows


def sort_json_keys(keys) -> list[str]:
    return list(keys)


def sorted_turn_prompts(row: dict) -> tuple[str, ...]:
    turns = []
    for key, value in row.items():
        match = re.fullmatch(r"turn(\d+)", str(key))
        if match and value:
            turns.append((int(match.group(1)), str(value)))
    turns.sort(key=lambda item: item[0])
    return tuple(value for _, value in turns)


def load_edit_info(path: Path) -> dict[str, dict]:
    if path.suffix.lower() == ".jsonl":
        return {str(index): row for index, row in enumerate(read_jsonl(path), start=1)}
    data = read_json(path)
    if isinstance(data, dict):
        return {str(key): value for key, value in data.items()}
    if isinstance(data, list):
        return {str(index): row for index, row in enumerate(data, start=1)}
    raise ValueError(f"Unsupported edit metadata format: {path}")


def default_edit_json(suite: str, benchmark_root: Path) -> Path:
    if suite == "basic":
        official = REPO_IMGEDIT_ROOT / "Benchmark" / "Basic" / "basic_edit.json"
        if official.is_file():
            return official
        return benchmark_root / "singleturn" / "singleturn.json"
    if suite == "uge":
        official = REPO_IMGEDIT_ROOT / "Benchmark" / "UGE" / "UGE_edit.json"
        if official.is_file():
            return official
        return benchmark_root / "hard" / "annotation.jsonl"
    raise ValueError(f"No default edit JSON for suite {suite!r}")


def default_origin_root(benchmark_root: Path, suite: str) -> Path:
    if suite == "basic":
        return benchmark_root / "singleturn"
    if suite == "uge":
        return benchmark_root / "hard"
    if suite == "multiturn":
        return benchmark_root / "multiturn"
    raise ValueError(f"No default origin root for suite {suite!r}")


def build_quantitative_samples(
    suite: str,
    edit_json: Path,
    origin_root: Path,
    fallback_search_root: Optional[Path],
    limit: Optional[int],
) -> list[BenchSample]:
    edit_info = load_edit_info(edit_json)
    samples = []
    for key in sort_json_keys(edit_info.keys()):
        row = edit_info[key]
        source = resolve_source_image(origin_root, row["id"], fallback_search_root)
        if not source.is_file():
            raise FileNotFoundError(f"Missing source image for key {key}: {source}")
        samples.append(
            BenchSample(
                key=str(key),
                source_image=source,
                prompt=str(row["prompt"]),
                suite=suite,
                edit_type=row.get("edit_type"),
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def resolve_source_image(origin_root: Path, image_id: str, fallback_search_root: Optional[Path]) -> Path:
    source = origin_root / image_id
    if source.is_file() or fallback_search_root is None:
        return source
    matches = sorted(fallback_search_root.rglob(Path(image_id).name), key=lambda path: path.as_posix())
    if matches:
        return matches[0]
    return source


def build_multiturn_samples(
    origin_root: Path,
    tasks: set[str],
    limit: Optional[int],
) -> list[BenchSample]:
    samples = []
    for task_dir in sorted(path for path in origin_root.iterdir() if path.is_dir()):
        if tasks and task_dir.name not in tasks:
            continue
        annotation = task_dir / "annotation.json"
        rows = read_jsonl(annotation)
        for row_index, row in enumerate(rows, start=1):
            prompts = sorted_turn_prompts(row)
            if not prompts:
                continue
            source = task_dir / row["id"]
            if not source.is_file():
                raise FileNotFoundError(f"Missing source image: {source}")
            key = f"{task_dir.name}_{row_index:04d}"
            samples.append(
                BenchSample(
                    key=key,
                    source_image=source,
                    prompt=prompts[0],
                    suite="multiturn",
                    turn_prompts=prompts,
                )
            )
            if limit is not None and len(samples) >= limit:
                return samples
    return samples


def setup_sharding(args: argparse.Namespace) -> ShardInfo:
    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", str(env_rank)))

    if args.num_shards is not None or args.shard_index is not None:
        world_size = args.num_shards if args.num_shards is not None else 1
        rank = args.shard_index if args.shard_index is not None else 0
        distributed = False
    else:
        world_size = env_world_size
        rank = env_rank
        distributed = world_size > 1

    if world_size < 1:
        raise ValueError(f"--num_shards/WORLD_SIZE must be positive, got {world_size}.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Shard/rank index {rank} is out of range for {world_size} shards.")

    if torch is not None and torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        local_rank = args.local_rank if args.local_rank is not None else env_local_rank
        local_rank = int(local_rank) % max(num_devices, 1)
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        local_rank = 0
        device = "cpu"

    if (
        torch is not None
        and distributed
        and torch.distributed.is_available()
        and not torch.distributed.is_initialized()
    ):
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)

    return ShardInfo(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        distributed=distributed and torch is not None and torch.distributed.is_initialized(),
    )


def barrier(shard: ShardInfo) -> None:
    if shard.distributed:
        import torch

        torch.distributed.barrier()


def cleanup_distributed(shard: ShardInfo) -> None:
    if shard.distributed:
        import torch

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def merge_shard_manifests(shard_manifest_dir: Path, manifest_path: Path) -> None:
    """Concatenate every per-rank shard manifest into the final manifest.jsonl.

    Ranks are merged in lexical (== ascending rank) order. Safe to call only when
    all shards are known to have finished writing their manifests.
    """
    paths = sorted(shard_manifest_dir.glob("rank_*_of_*.jsonl"))
    if not paths:
        return
    with manifest_path.open("w", encoding="utf-8") as merged:
        for path in paths:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    merged.write(line)


def shard_samples(samples: Sequence[BenchSample], shard: ShardInfo) -> list[BenchSample]:
    return list(samples[shard.rank :: shard.world_size])


def resolve_transformer_load_args(transformer_path: str) -> tuple[str, str | None]:
    path = Path(transformer_path).expanduser()
    if (path / "transformer" / "config.json").is_file():
        return str(path), "transformer"
    if (path / "config.json").is_file():
        return str(path), None

    tried = [
        path / "config.json",
        path / "transformer" / "config.json",
    ]
    raise FileNotFoundError(
        "Could not find fine-tuned Flux2 transformer weights.\n"
        f"Received --transformer_path: {transformer_path}\n"
        f"Resolved path: {path.resolve() if path.exists() else path}\n"
        "Expected one of these files to exist:\n"
        + "\n".join(f"  - {candidate}" for candidate in tried)
        + "\n\nPass either a training output/checkpoint directory that contains "
        "`transformer/config.json`, or pass the transformer directory itself."
    )


def load_pipeline(model_name: str, transformer_path: Optional[str], lora_path: Optional[str] = None):
    import torch
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel

    if transformer_path:
        transformer_root, transformer_subfolder = resolve_transformer_load_args(transformer_path)
        kwargs = {"torch_dtype": torch.bfloat16}
        if transformer_subfolder:
            kwargs["subfolder"] = transformer_subfolder
        transformer = Flux2Transformer2DModel.from_pretrained(transformer_root, **kwargs)
        pipe = Flux2KleinPipeline.from_pretrained(
            model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
    else:
        pipe = Flux2KleinPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    if lora_path:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from common.flux2_runtime import load_lora_into_pipeline
        load_lora_into_pipeline(pipe, lora_path)

    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe, torch


def format_model_prompt(prefix: str, instruction: str) -> str:
    prefix = prefix.strip()
    instruction = instruction.strip()
    return f"{prefix} {instruction}" if prefix else instruction


def generate_one(pipe, torch, image: Image.Image, prompt: str, args, generator=None) -> Image.Image:
    width, height = image.size
    kwargs = {
        "prompt": format_model_prompt(args.prompt_prefix, prompt),
        "image": image,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
    }
    if args.native_resolution:
        kwargs["width"] = width
        kwargs["height"] = height
    if generator is not None:
        kwargs["generator"] = generator
    with torch.inference_mode():
        return pipe(**kwargs).images[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-base-9B")
    parser.add_argument(
        "--transformer_path",
        default=None,
        help="Fine-tuned run dir, checkpoint dir, or direct transformer dir.",
    )
    parser.add_argument("--lora_path", default=None, help="Standard Diffusers LoRA directory (holds pytorch_lora_weights.safetensors) applied on top of --model.")
    parser.add_argument(
        "--benchmark_root",
        default=DEFAULT_BENCHMARK_ROOT,
        help="Extracted ImgEdit Benchmark directory.",
    )
    parser.add_argument(
        "--suite",
        choices=("basic", "uge", "multiturn"),
        default="basic",
        help="Official quantitative suites are basic and uge.",
    )
    parser.add_argument("--edit_json", default=None, help="Override metadata JSON for basic/uge.")
    parser.add_argument("--origin_img_root", default=None, help="Override source image root.")
    parser.add_argument(
        "--fallback_search_root",
        default=None,
        help=(
            "Optional tree searched by basename when a metadata image is absent from origin_img_root. "
            "Defaults to benchmark_root for UGE because some extracted archives place shared images elsewhere."
        ),
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--prompt_prefix", default="[TASK: IMAGE_EDITING]")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--native_resolution", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Manual sharding: total number of evaluator shards. Torchrun WORLD_SIZE is used if omitted.",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=None,
        help="Manual sharding: this shard index. Torchrun RANK is used if omitted.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=None,
        help="CUDA device index for this process. Usually set automatically by torchrun.",
    )
    parser.add_argument(
        "--multiturn_tasks",
        default="",
        help="Comma-separated multiturn task names. Empty means all tasks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard = setup_sharding(args)
    benchmark_root = Path(args.benchmark_root).expanduser()
    origin_root = Path(args.origin_img_root).expanduser() if args.origin_img_root else default_origin_root(benchmark_root, args.suite)
    fallback_search_root = (
        Path(args.fallback_search_root).expanduser()
        if args.fallback_search_root
        else (benchmark_root if args.suite == "uge" else None)
    )

    if args.suite in {"basic", "uge"}:
        edit_json = Path(args.edit_json).expanduser() if args.edit_json else default_edit_json(args.suite, benchmark_root)
        all_samples = build_quantitative_samples(args.suite, edit_json, origin_root, fallback_search_root, args.limit)
    else:
        edit_json = None
        tasks = {task.strip() for task in args.multiturn_tasks.split(",") if task.strip()}
        all_samples = build_multiturn_samples(origin_root, tasks, args.limit)

    samples = shard_samples(all_samples, shard)

    if shard.rank == 0:
        print(f"Suite: {args.suite}")
        print(f"Benchmark root: {benchmark_root}")
        print(f"Origin image root: {origin_root}")
        if edit_json:
            print(f"Edit JSON: {edit_json}")
        print(f"Output directory: {Path(args.out_dir).expanduser()}")
        print(f"Resolved samples: {len(all_samples)}")
        print(f"Evaluator shards: {shard.world_size}")
    print(
        f"Rank {shard.rank}/{shard.world_size}: "
        f"{len(samples)} of {len(all_samples)} samples on {shard.device}"
    )
    for sample in samples[:5]:
        print(f"  rank {shard.rank} sample {sample.key}: {sample.source_image}")
        if sample.turn_prompts:
            print(f"    turns: {len(sample.turn_prompts)}")

    if args.dry_run:
        cleanup_distributed(shard)
        return

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Edited prediction images go in an edited_images/ subfolder so the suite dir
    # is not cluttered with hundreds of <key>.png alongside the manifest/configs.
    # The judge reads from this same subfolder.
    edited_images_dir = out_dir / "edited_images"
    edited_images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    shard_manifest_dir = out_dir / "shard_manifests"
    shard_manifest_dir.mkdir(parents=True, exist_ok=True)
    shard_manifest_path = shard_manifest_dir / f"rank_{shard.rank:05d}_of_{shard.world_size:05d}.jsonl"
    config_path = out_dir / "run_config.json"

    pipe, torch = load_pipeline(args.model, args.transformer_path, args.lora_path)
    generator = torch.Generator(device="cuda").manual_seed(args.seed) if args.seed is not None else None

    if shard.rank == 0:
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "suite": args.suite,
                    "model": args.model,
                    "transformer_path": args.transformer_path,
                    "lora_path": args.lora_path,
                    "benchmark_root": str(benchmark_root),
                    "origin_img_root": str(origin_root),
                    "fallback_search_root": str(fallback_search_root) if fallback_search_root else None,
                    "edit_json": str(edit_json) if edit_json else None,
                    "steps": args.steps,
                    "guidance_scale": args.guidance_scale,
                    "prompt_prefix": args.prompt_prefix,
                    "seed": args.seed,
                    "native_resolution": args.native_resolution,
                    "world_size": shard.world_size,
                },
                handle,
                indent=2,
            )

    with shard_manifest_path.open("w", encoding="utf-8") as manifest:
        progress = tqdm(samples, desc=f"Rank {shard.rank}/{shard.world_size}", position=shard.rank, leave=shard.rank == 0)
        for sample in progress:
            if sample.turn_prompts:
                current = Image.open(sample.source_image).convert("RGB")
                turn_outputs = []
                turn_dir = out_dir / sample.key
                turn_dir.mkdir(parents=True, exist_ok=True)
                for turn_index, turn_prompt in enumerate(sample.turn_prompts, start=1):
                    output_path = turn_dir / f"turn_{turn_index}.png"
                    if output_path.exists() and not args.overwrite:
                        current = Image.open(output_path).convert("RGB")
                    else:
                        current = generate_one(pipe, torch, current, turn_prompt, args, generator=generator)
                        current.save(output_path)
                    turn_outputs.append(str(output_path))
                final_output = edited_images_dir / f"{sample.key}.png"
                current.save(final_output)
            else:
                output_path = edited_images_dir / f"{sample.key}.png"
                if output_path.exists() and not args.overwrite:
                    final_output = output_path
                    turn_outputs = []
                else:
                    image = Image.open(sample.source_image).convert("RGB")
                    edited = generate_one(pipe, torch, image, sample.prompt, args, generator=generator)
                    edited.save(output_path)
                    final_output = output_path
                    turn_outputs = []

            record = {
                "key": sample.key,
                "suite": sample.suite,
                "source_image": str(sample.source_image),
                "output_image": str(final_output),
                "prompt": sample.prompt,
                "edit_type": sample.edit_type,
                "turn_prompts": list(sample.turn_prompts),
                "turn_outputs": turn_outputs,
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()

    barrier(shard)

    # Merge only in the torchrun path, where all shard manifests are present at
    # the barrier. Cross-job shards (--num_shards/--shard_index) finish at
    # different times, so their merge is deferred to the judge step.
    if shard.rank == 0 and shard.distributed:
        merge_shard_manifests(shard_manifest_dir, manifest_path)
        print(f"Saved official-layout predictions to: {out_dir}")
        print(f"Saved manifest: {manifest_path}")

    cleanup_distributed(shard)


if __name__ == "__main__":
    main()
