#!/usr/bin/env python3
"""Validate final inference weights and the resumable terminal checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing or empty training artifact: {path}")


def validate_diffusers_weights(directory: Path) -> None:
    require_nonempty(directory / "config.json")
    indexes = list(directory.glob("*.safetensors.index.json"))
    if indexes:
        payload = json.loads(indexes[0].read_text())
        shards = sorted(set(payload.get("weight_map", {}).values()))
        if not shards:
            raise SystemExit(f"empty safetensors weight map: {indexes[0]}")
        for shard in shards:
            require_nonempty(directory / shard)
        return
    weights = list(directory.glob("*.safetensors")) + list(directory.glob("*.bin"))
    if not weights:
        raise SystemExit(f"no model weights found in {directory}")
    for weight in weights:
        require_nonempty(weight)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--train-method", choices=("full", "lora"), default="full")
    args = parser.parse_args()

    checkpoint = args.run_dir / f"checkpoint-{args.max_steps}"
    if not checkpoint.is_dir():
        raise SystemExit(f"terminal recovery checkpoint is missing: {checkpoint}")
    require_nonempty(checkpoint / "optimizer.bin")
    require_nonempty(checkpoint / "scheduler.bin")
    for rank in range(args.num_processes):
        require_nonempty(checkpoint / f"random_states_{rank}.pkl")
    sampler = checkpoint / "sampler_state.json"
    require_nonempty(sampler)
    sampler_payload = json.loads(sampler.read_text())
    if int(sampler_payload.get("global_step", -1)) != args.max_steps:
        raise SystemExit(f"terminal sampler state has the wrong global step: {sampler_payload}")

    if args.train_method == "lora":
        lora = list(args.run_dir.glob("*lora*.safetensors"))
        if not lora:
            raise SystemExit(f"final LoRA weights are missing from {args.run_dir}")
        for path in lora:
            require_nonempty(path)
    else:
        validate_diffusers_weights(args.run_dir / "transformer")
        validate_diffusers_weights(checkpoint / "transformer")
    print(f"validated terminal training artifacts at step {args.max_steps}")


if __name__ == "__main__":
    main()
