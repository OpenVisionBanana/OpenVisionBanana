"""Shared, backward-compatible FLUX.2 pipeline loading for evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


MODEL_FAMILIES = ("klein", "dev")


def normalize_model_family(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {"flux2_klein": "klein", "flux2_dev": "dev", "flux2": "dev"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model family {value!r}; choose from {MODEL_FAMILIES}.")
    return normalized


def resolve_transformer_root(path: str | os.PathLike[str]) -> Path:
    root = Path(path).expanduser()
    candidates = (root, root / "transformer")
    for candidate in candidates:
        if candidate.is_dir() and (
            (candidate / "config.json").is_file()
            or any(candidate.glob("diffusion_pytorch_model*.safetensors"))
        ):
            return candidate
    raise FileNotFoundError(
        f"Could not resolve a Diffusers transformer from {root}; expected config.json or transformer/."
    )


def resolve_lora_weight(path: str | os.PathLike[str]) -> Path:
    """Return the one standard Diffusers LoRA weight file in an artifact directory."""
    root = Path(path).expanduser()
    candidates = sorted(root.glob("*lora*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No standard Diffusers LoRA safetensors file found in: {root}")
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one standard Diffusers LoRA safetensors file in {root}; "
            f"found: {', '.join(candidate.name for candidate in candidates)}"
        )
    return candidates[0]


def load_lora_into_pipeline(pipe, lora_path: str | os.PathLike[str] | None):
    """Apply a standard Diffusers LoRA artifact directory to a loaded pipeline (no-op if None)."""
    if not lora_path:
        return pipe
    lora_root = Path(lora_path).expanduser()
    if not lora_root.is_dir():
        raise FileNotFoundError(f"LoRA directory not found: {lora_root}")
    # Diffusers intentionally refuses to guess a filename when offline.
    # Pass the artifact's explicit standard filename so offline runs work.
    pipe.load_lora_weights(str(lora_root), weight_name=resolve_lora_weight(lora_root).name)
    return pipe


def artifact_metadata(
    model_family: str, model: str, transformer_path: str | None, lora_path: str | None
) -> dict[str, Any]:
    family = normalize_model_family(model_family)
    if transformer_path and lora_path:
        raise ValueError("--transformer_path and --lora_path are mutually exclusive.")
    method = "lora" if lora_path else "partial_full_rank" if transformer_path else "base"
    return {
        "model_family": family,
        "base_model": model,
        "adaptation_method": method,
        "transformer_path": transformer_path,
        "lora_path": lora_path,
        "pipeline_class": "Flux2Pipeline" if family == "dev" else "Flux2KleinPipeline",
    }


def load_flux2_pipeline(
    model: str,
    *,
    model_family: str = "klein",
    transformer_path: str | None = None,
    lora_path: str | None = None,
    device: str = "cuda",
    offload: bool = False,
    offload_mode: str | None = None,
    torch_dtype=None,
):
    """Load a base, standard LoRA, or partial-transformer FLUX.2 pipeline."""
    import torch
    from diffusers import Flux2KleinPipeline, Flux2Pipeline, Flux2Transformer2DModel

    metadata = artifact_metadata(model_family, model, transformer_path, lora_path)
    family = metadata["model_family"]
    dtype = torch_dtype or torch.bfloat16
    local_only = Path(model).expanduser().exists() or os.environ.get("HF_HUB_OFFLINE") == "1"
    common = {"torch_dtype": dtype, "local_files_only": local_only}
    pipeline_cls = Flux2Pipeline if family == "dev" else Flux2KleinPipeline
    if transformer_path:
        transformer_root = resolve_transformer_root(transformer_path)
        transformer = Flux2Transformer2DModel.from_pretrained(str(transformer_root), **common)
        pipe = pipeline_cls.from_pretrained(model, transformer=transformer, **common)
    else:
        pipe = pipeline_cls.from_pretrained(model, **common)
    load_lora_into_pipeline(pipe, lora_path)
    mode = offload_mode or ("model" if offload else "none")
    if mode == "model":
        pipe.enable_model_cpu_offload(device=device)
    elif mode == "group":
        pipe.enable_group_offload(
            onload_device=torch.device(device),
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=1,
        )
    elif mode == "sequential":
        pipe.enable_sequential_cpu_offload(device=device)
    elif mode == "none":
        pipe.to(device)
    else:
        raise ValueError(
            f"Unknown offload mode {mode!r}; choose none, model, group, or sequential."
        )
    pipe.set_progress_bar_config(disable=True)
    return pipe, torch, metadata


def add_flux2_cli_args(parser, *, include_model: bool = False) -> None:
    """Add only missing shared arguments to an argparse parser."""
    existing = {action.dest for action in parser._actions}
    if include_model and "model" not in existing:
        parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-base-9B")
    if "model_family" not in existing:
        parser.add_argument("--model_family", choices=MODEL_FAMILIES, default="klein")
    if "transformer_path" not in existing:
        parser.add_argument("--transformer_path")
    if "lora_path" not in existing:
        parser.add_argument("--lora_path")
    if "model_offload" not in existing:
        parser.add_argument(
            "--model_offload", action="store_true", help="Enable Diffusers model CPU offload."
        )
