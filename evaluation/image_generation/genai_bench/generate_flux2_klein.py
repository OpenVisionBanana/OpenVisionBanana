#!/usr/bin/env python3
"""Generate FLUX.2-Klein images for GenAI-Bench prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


@dataclass(frozen=True)
class PromptSample:
    prompt_id: str
    prompt: str
    tags: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-base-9B")
    parser.add_argument("--model_family", choices=("klein", "dev"), default="klein")
    parser.add_argument("--lora_path")
    parser.add_argument("--model_offload", action="store_true")
    parser.add_argument("--transformer_path", help="Fine-tuned run dir, checkpoint dir, or direct transformer dir.")
    parser.add_argument("--model_tag", default="f2k9b_pretrained")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt_prefix", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def read_prompts(path: Path, limit: Optional[int]) -> list[PromptSample]:
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and len(samples) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt_id = str(row.get("id", f"{line_number:05d}"))
            prompt = str(row["prompt"]).strip()
            tags = tuple(str(tag) for tag in row.get("tags", []))
            samples.append(PromptSample(prompt_id=prompt_id, prompt=prompt, tags=tags))
    return samples


def derive_seed(base_seed: int, prompt_id: str) -> int:
    """Per-prompt seed that does not depend on shard count or resume order.

    Derived from (base_seed, prompt_id) so a prompt's noise does not change with
    --num_shards or a resume. sha256 rather than hash(): the latter is salted per
    process.
    """
    digest = hashlib.sha256(f"{base_seed}:{prompt_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def shard_samples(samples: Sequence[PromptSample], num_shards: int, shard_index: int) -> list[PromptSample]:
    if num_shards < 1:
        raise ValueError("--num_shards must be >= 1.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard_index must be in [0, num_shards).")
    return list(samples[shard_index::num_shards])


def resolve_transformer_load_args(transformer_path: str) -> tuple[str, str | None]:
    path = Path(transformer_path).expanduser()
    if (path / "transformer" / "config.json").is_file():
        return str(path), "transformer"
    if (path / "config.json").is_file():
        return str(path), None
    tried = [path / "config.json", path / "transformer" / "config.json"]
    raise FileNotFoundError(
        "Could not find fine-tuned Flux2 transformer weights.\n"
        f"Received --transformer_path: {transformer_path}\n"
        "Expected one of:\n"
        + "\n".join(f"  - {candidate}" for candidate in tried)
    )


def load_pipeline(model_name: str, transformer_path: Optional[str], device: str, model_family="klein", lora_path=None, offload=False):
    from common.flux2_runtime import load_flux2_pipeline
    pipe, torch, _metadata = load_flux2_pipeline(
        model_name, model_family=model_family, transformer_path=transformer_path,
        lora_path=lora_path, device=device, offload=offload,
    )
    return pipe, torch


def setup_device(local_rank: Optional[int]) -> str:
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    visible_count = torch.cuda.device_count()
    rank = int(local_rank if local_rank is not None else os.environ.get("LOCAL_RANK", "0"))
    rank = rank % max(visible_count, 1)
    torch.cuda.set_device(rank)
    return f"cuda:{rank}"


def format_prompt(prefix: str, prompt: str) -> str:
    prefix = prefix.strip()
    prompt = prompt.strip()
    return f"{prefix} {prompt}" if prefix else prompt


def merge_shard_manifests(out_dir: Path) -> None:
    shard_dir = out_dir / "shard_manifests"
    paths = sorted(shard_dir.glob("rank_*_of_*.jsonl"))
    if not paths:
        return
    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as merged:
        for path in paths:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    merged.write(line)


def progress_iter(items, description: str):
    try:
        from tqdm import tqdm

        return tqdm(items, desc=description)
    except ModuleNotFoundError:
        print(description)
        return items


def main() -> None:
    args = parse_args()
    prompts_path = Path(args.prompts_jsonl).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    image_dir = out_dir / "images"
    shard_dir = out_dir / "shard_manifests"
    config_path = out_dir / "run_config.json"

    all_samples = read_prompts(prompts_path, args.limit)
    samples = shard_samples(all_samples, args.num_shards, args.shard_index)

    print(f"Prompts: {len(all_samples)}")
    print(f"Shard {args.shard_index}/{args.num_shards}: {len(samples)} prompts")
    print(f"Output directory: {out_dir}")
    for sample in samples[:5]:
        print(f"  {sample.prompt_id}: {sample.prompt[:100]}")
    if args.dry_run:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "benchmark": "genai_bench",
                "prompts_jsonl": str(prompts_path),
                "model": args.model,
                "model_family": args.model_family,
                "transformer_path": args.transformer_path,
                "lora_path": args.lora_path,
                "model_tag": args.model_tag,
                "checkpoint": args.checkpoint,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "height": args.height,
                "width": args.width,
                "seed": args.seed,
                "prompt_prefix": args.prompt_prefix,
                "num_prompts": len(all_samples),
                "num_shards": args.num_shards,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    device = setup_device(args.local_rank)
    pipe, torch = load_pipeline(
        args.model, args.transformer_path, device,
        args.model_family, args.lora_path, args.model_offload,
    )
    shard_manifest = shard_dir / f"rank_{args.shard_index:05d}_of_{args.num_shards:05d}.jsonl"
    with shard_manifest.open("w", encoding="utf-8") as manifest:
        for sample in progress_iter(samples, f"GenAI-Bench shard {args.shard_index}/{args.num_shards}"):
            image_path = image_dir / f"{sample.prompt_id}.png"
            prompt_seed = derive_seed(args.seed, sample.prompt_id) if args.seed is not None else None
            if not image_path.exists() or args.overwrite:
                generation_kwargs = {
                    "prompt": format_prompt(args.prompt_prefix, sample.prompt),
                    "num_inference_steps": args.steps,
                    "guidance_scale": args.guidance_scale,
                }
                if args.height is not None:
                    generation_kwargs["height"] = args.height
                if args.width is not None:
                    generation_kwargs["width"] = args.width
                if prompt_seed is not None:
                    generation_kwargs["generator"] = torch.Generator(device=device).manual_seed(prompt_seed)
                with torch.inference_mode():
                    image = pipe(**generation_kwargs).images[0]
                image.save(image_path)
            record = {
                "id": sample.prompt_id,
                "prompt": sample.prompt,
                "tags": list(sample.tags),
                "image": image_path.relative_to(out_dir).as_posix(),
                "model_tag": args.model_tag,
                "checkpoint": args.checkpoint,
                "seed": args.seed,
                "prompt_seed": prompt_seed,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
            }
            manifest.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            manifest.flush()

    if args.num_shards == 1:
        merge_shard_manifests(out_dir)
    print(f"Saved shard manifest: {shard_manifest}")
    if (out_dir / "manifest.jsonl").exists():
        print(f"Saved manifest: {out_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
