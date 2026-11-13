#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from the FLUX.2 [klein] LoRA fine-tuning script in the diffusers
# DreamBooth examples (https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)
# and extended to full fine-tuning of the transformer on multi-task pools.

# /// script
# dependencies = [
#     "diffusers @ git+https://github.com/huggingface/diffusers.git",
#     "torch>=2.0.0",
#     "accelerate>=0.31.0",
#     "transformers>=4.41.2",
#     "ftfy",
#     "tensorboard",
#     "Jinja2",
#     "sentencepiece",
#     "torchvision",
#     "datasets",
#     "bitsandbytes",
#     "prodigyopt",
# ]
# ///

from __future__ import annotations

"""Train a partially fine-tuned Flux2-Klein transformer for image-to-image tasks.

This script is a self-contained training entrypoint. In broad strokes it:
1. Parses command-line arguments that describe the base model, dataset, and
   optimization settings.
2. Loads the pretrained Flux2-Klein components and freezes everything except
   selected late transformer blocks and output layers.
3. Builds a training dataset from either a local folder, a Hugging Face
   dataset, or a weighted pool of prepared task datasets described by JSON.
4. Optionally precomputes VAE latents and text embeddings to trade disk space
   for lower GPU usage and faster training steps.
5. Runs the flow-matching training loop, periodically validating and saving
   resumable checkpoints.
6. Writes the fine-tuned transformer weights to ``<output_dir>/transformer``
   together with a small model card.

Expected inputs:
- ``--pretrained_model_name_or_path``: a compatible Flux2-Klein base model.
- Exactly one dataset source: ``--dataset_specs``, ``--dataset_name``, or
  ``--instance_data_dir``.
- For image-to-image training, rows should provide both a target image and a
  conditioning image, usually through ``image`` and ``conditioning_image``
  columns or equivalent column names passed on the CLI.

Important outputs:
- ``<output_dir>/transformer``: the final fine-tuned transformer weights.
- ``<output_dir>/checkpoint-*``: optional training-state checkpoints.
- ``<output_dir>/dataset_specs.resolved.json``: the fully resolved dataset pool
  configuration when ``--dataset_specs`` is used.
- ``<output_dir>/precomputed_cache``: optional latent/text caches when caching
  is enabled.

This FSDP variant keeps the regular command-line surface, but defaults to
Accelerate FSDP full sharding when launched with multiple processes. It is meant
for memory-heavy full-transformer or many-last-layer fine-tuning runs.
"""

import argparse
import copy
import faulthandler
import hashlib
import io
import itertools
import json
import logging
import math
import os
import random
import re
import signal
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from huggingface_hub import ModelCard, ModelCardData, create_repo, snapshot_download, upload_folder
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms
from common.geometry_auxiliary_losses import (
    angular_normal_loss,
    palette_manifold_loss,
    recover_clean_latents,
    robust_metric_depth_loss,
    unit_normal_loss,
)
from common.training.masked_reduction import normalized_distributed_loss
from common.geometry_protocol_prompts import (
    BARRON_SCALAR_DEPTH_PROMPT,
    CANONICAL_DEPTH_PROMPT,
    CONCISE_INVERSE_DEPTH_PROMPT,
    CONCISE_LINEAR_DEPTH_PROMPT,
    CONCISE_NORMAL_PROMPT,
    DETAILED_INVERSE_DEPTH_PROMPT,
    DETAILED_NORMAL_PROMPT,
    PAPER_RECONSTRUCTED_BARRON_DEPTH_PROMPT,
    PAPER_RECONSTRUCTED_INVERSE_DEPTH_PROMPT,
    PAPER_RECONSTRUCTED_NORMAL_PROMPT,
    VISION_BANANA_NORMAL_PROMPT,
)
from common.surface_normal_encoding import (
    LEGACY_DIRECT_NORMAL_ENCODING,
    SUPPORTED_NORMAL_ENCODINGS,
    VISION_BANANA_NORMAL_ENCODING,
    decode_surface_normals_rgb,
    encode_surface_normals_rgb,
    validate_normal_encoding,
)
from common.indexed_tar import IndexedTarReference, load_index, open_reference_image
from dataset_prep.depth.encoding import (
    DEPTH_ENCODING_BARRON_SCALAR_GRAYSCALE,
    DEPTH_ENCODING_RELATIVE_INVERSE_GRAYSCALE,
    DEPTH_ENCODING_RELATIVE_LINEAR_GRAYSCALE,
    DEPTH_ENCODING_VISION_BANANA_RGB,
    decode_vision_banana_rgb_to_depth,
    encode_depth_to_relative_grayscale,
    encode_depth_to_vision_banana_rgb,
    project_rgb_to_vision_banana_path,
)
from dataset_prep.geometry_conventions import resize_normal_field, resize_scalar_field
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

import diffusers
from diffusers import (
    AutoencoderKLFlux2,
    BitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.models.modeling_utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    _add_variant,
    safetensors,
    split_torch_state_dict_into_shards,
)
from diffusers.training_utils import (
    _collate_lora_metadata,
    _to_cpu_contiguous,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    find_nearest_bucket,
    free_memory,
    get_fsdp_kwargs_from_accelerator,
    offload_models,
    parse_buckets_string,
    wrap_with_fsdp,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
    load_image,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module

from common.training.bucket_index_cache import load_or_build_bucket_indices
from common.training.packed_prompt_cache import (
    PackedPromptCache,
    PackedPromptCacheWriter,
    finalize_packed_prompt_cache,
    prompt_cache_fingerprint,
    prompt_key,
)
from common.training.pool_sampling import limit_distinct_per_group, sample_indices


if getattr(torch, "distributed", None) is not None:
    import torch.distributed as dist

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.39.0.dev0")

logger = get_logger(__name__)


class DataRowLoadTimeout(TimeoutError):
    """Raised when one prepared row blocks data delivery past its configured limit."""


_ACTIVE_DATALOADER_ROW_LABEL: str | None = None


def _dump_python_stack(signum, _frame):
    """Emit a stack and the in-flight row when the wrapper detects a stall."""
    active_row = _ACTIVE_DATALOADER_ROW_LABEL or "<no row currently loading>"
    print(
        f"DATALOADER DIAGNOSTIC: signal={signum}, active_row={active_row}",
        file=sys.stderr,
        flush=True,
    )
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)


def _install_dataloader_worker_diagnostics(worker_id):
    """Let the Slurm watchdog obtain useful stacks from worker subprocesses."""
    del worker_id
    faulthandler.enable(file=sys.stderr, all_threads=True)
    signal.signal(signal.SIGUSR2, _dump_python_stack)


@contextmanager
def row_load_deadline(seconds: float, row_label: str):
    """Bound one CPU-side image load without changing normal row semantics."""
    if seconds <= 0:
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)

    def raise_timeout(_signum, _frame):
        raise DataRowLoadTimeout(f"timed out after {seconds:.0f}s while loading {row_label}")

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)
EXIF_ORIENTATION_TAG = 274
# These EXIF orientations represent rotated images whose displayed width/height
# are effectively swapped compared with the raw stored pixels.
TRANSPOSED_EXIF_ORIENTATIONS = {5, 6, 7, 8}


def log_info(message: str, log_fn=None):
    """Log through an optional callback and fall back to the module logger."""
    if log_fn is not None:
        log_fn(message)
        return
    try:
        logger.info(message)
    except RuntimeError:
        logging.getLogger(__name__).info(message)


def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = None,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
    fp8_training=False,
    load_existing_card=False,
):
    """Create a minimal Hugging Face model card for the trained transformer."""
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = f"""
# Flux.2 [Klein] partial transformer fine-tune - {repo_id}

<Gallery />

## Model description

These are {repo_id} partial transformer fine-tuning weights for {base_model}.

The weights were trained using an image-to-image Flux2-Klein trainer that freezes the base model except for selected late transformer layers and output layers.

FP8 training? {fp8_training}

## Trigger words

You should use `{instance_prompt}` to trigger the image generation or image-editing behavior.

## Use it with diffusers

```py
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
import torch

transformer = Flux2Transformer2DModel.from_pretrained("{repo_id}", subfolder="transformer", torch_dtype=torch.bfloat16)
pipeline = Flux2KleinPipeline.from_pretrained("{base_model}", transformer=transformer, torch_dtype=torch.bfloat16).to("cuda")
image = pipeline("{validation_prompt if validation_prompt else instance_prompt}").images[0]
```

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/black-forest-labs/FLUX.2/blob/main/LICENSE.md).
"""
    if load_existing_card:
        # A Hub push may intentionally preserve metadata from an existing card.
        card_repo_id = repo_id[:96].rstrip("-._") or "model"
        model_card = load_or_create_model_card(
            repo_id_or_path=card_repo_id,
            from_training=True,
            license="other",
            base_model=base_model,
            prompt=instance_prompt,
            model_description=model_description,
            widget=widget_dict,
        )
    else:
        # load_or_create_model_card may treat an output basename as a Hub
        # repository; build the card directly for local runs.
        model_card = ModelCard.from_template(
            card_data=ModelCardData(
                license="other",
                library_name="diffusers",
                base_model=base_model,
                instance_prompt=instance_prompt,
                widget=widget_dict,
            ),
            model_description=model_description,
        )
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "flux2",
        "flux2-diffusers",
        "partial-finetune",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    torch_dtype,
    is_final_validation=False,
):
    """Run validation inference and send the generated images to active trackers."""
    args.num_validation_images = args.num_validation_images if args.num_validation_images else 1
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(dtype=torch_dtype)
    pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()

    images = []
    for _ in range(args.num_validation_images):
        with autocast_ctx:
            image = pipeline(
                image=pipeline_args.get("image"),
                prompt_embeds=pipeline_args["prompt_embeds"],
                negative_prompt_embeds=pipeline_args["negative_prompt_embeds"],
                generator=generator,
            ).images[0]
            images.append(image)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    free_memory()

    return images


def load_validation_prompt_records(path: str | None, default_seed: int | None) -> list[dict[str, Any]]:
    """Read plain-text or JSONL validation prompts with stable IDs and seeds."""
    if not path:
        return []
    records = []
    seen_ids = set()
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                row = json.loads(line)
                prompt = str(row.get("prompt", "")).strip()
                prompt_id = str(row.get("id", f"prompt-{line_number:04d}"))
                seed = int(row.get("seed", (default_seed or 0) + line_number - 1))
            else:
                prompt = line
                prompt_id = f"prompt-{line_number:04d}"
                seed = (default_seed or 0) + line_number - 1
            if not prompt:
                raise ValueError(f"Empty prompt at {path}:{line_number}")
            if prompt_id in seen_ids:
                raise ValueError(f"Duplicate validation prompt id {prompt_id!r} at {path}:{line_number}")
            seen_ids.add(prompt_id)
            records.append({"id": prompt_id, "prompt": prompt, "seed": seed})
    if not records:
        raise ValueError(f"Validation prompts file contains no prompts: {path}")
    return records


def log_fixed_text_validation(
    pipeline,
    text_encoding_pipeline,
    records,
    args,
    accelerator,
    step_label,
    torch_dtype,
):
    """Generate and persist a reproducible text-only development prompt set."""
    output_dir = Path(args.output_dir) / "validation_text_to_image" / str(step_label)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = pipeline.to(dtype=torch_dtype)
    pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=True)
    manifest = []
    for record in records:
        with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
            prompt_embeds, _ = text_encoding_pipeline.encode_prompt(
                prompt=record["prompt"], max_sequence_length=args.max_sequence_length
            )
            negative_prompt_embeds, _ = text_encoding_pipeline.encode_prompt(
                prompt="", max_sequence_length=args.max_sequence_length
            )
        generator = torch.Generator(device=accelerator.device).manual_seed(record["seed"])
        image = pipeline(
            image=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=generator,
            num_inference_steps=args.validation_num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.validation_height,
            width=args.validation_width,
        ).images[0]
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", record["id"]).strip("._") or "prompt"
        id_digest = hashlib.sha256(record["id"].encode("utf-8")).hexdigest()[:8]
        filename = f"{safe_id}-{id_digest}.png"
        image.save(output_dir / filename)
        manifest.append(
            {
                **record,
                "image": filename,
                "conditioning_image": None,
                "num_inference_steps": args.validation_num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "height": args.validation_height,
                "width": args.validation_width,
                "base_model": args.pretrained_model_name_or_path,
            }
        )
    (output_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest),
        encoding="utf-8",
    )
    return manifest


def module_filter_fn(mod: torch.nn.Module, fqn: str):
    """Skip modules that are unsafe or unsupported for float8 conversion."""
    # do not convert the output module
    if fqn == "proj_out":
        return False
    # do not convert linear modules with weight dimensions not divisible by 16
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    return True


def _set_trainable(module: torch.nn.Module, trainable: bool = True):
    """Toggle ``requires_grad`` for every parameter inside ``module``."""
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)


def parse_layer_index_spec(index_spec: str | None, num_layers: int, arg_name: str) -> list[int]:
    """Parse a comma-separated layer index list, accepting negative indices."""
    if not index_spec:
        return []

    indices = []
    seen = set()
    for raw_token in index_spec.split(","):
        token = raw_token.strip()
        if not token:
            continue
        try:
            index = int(token)
        except ValueError as error:
            raise ValueError(
                f"{arg_name} must be a comma-separated list of layer indices, "
                f"for example '0,6,12,-1'. Got {token!r}."
            ) from error

        if index < 0:
            index += num_layers
        if index < 0 or index >= num_layers:
            raise ValueError(f"{arg_name} index {raw_token!r} is out of range for {num_layers} layers.")
        if index not in seen:
            indices.append(index)
            seen.add(index)

    return indices


def configure_partial_transformer_training(transformer: Flux2Transformer2DModel, args):
    """Freeze the transformer, then unfreeze all params or the selected layers."""
    scope_contract = getattr(args, "trainable_scope_contract", "release_colorexp")
    if scope_contract not in {"separate_embeddings", "release_colorexp"}:
        raise ValueError(f"Unsupported trainable scope contract: {scope_contract!r}")
    transformer.requires_grad_(False)
    trainable_modules = []

    num_double = len(transformer.transformer_blocks)
    num_single = len(transformer.single_transformer_blocks)
    if getattr(args, "train_full_transformer", False):
        transformer.requires_grad_(True)
        trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in transformer.parameters())
        return ["<all transformer parameters>"], trainable_params, total_params

    train_all_transformer_blocks = getattr(args, "train_all_transformer_blocks", False)
    if args.train_last_transformer_blocks < 0 or args.train_last_transformer_blocks > num_double:
        raise ValueError(
            f"--train_last_transformer_blocks must be between 0 and {num_double}, "
            f"got {args.train_last_transformer_blocks}."
        )
    if args.train_last_single_transformer_blocks < 0 or args.train_last_single_transformer_blocks > num_single:
        raise ValueError(
            f"--train_last_single_transformer_blocks must be between 0 and {num_single}, "
            f"got {args.train_last_single_transformer_blocks}."
        )

    if getattr(args, "train_non_block_layers", False):
        non_block_modules = (
            "time_guidance_embed",
            "double_stream_modulation_img",
            "double_stream_modulation_txt",
            "single_stream_modulation",
        )
        if scope_contract == "release_colorexp":
            # Preserve the upstream default; separate_embeddings explicitly
            # selects the independently controlled input-layer scope.
            non_block_modules += ("x_embedder", "context_embedder")
        for module_name in non_block_modules:
            module = getattr(transformer, module_name, None)
            if module is not None:
                _set_trainable(module)
                trainable_modules.append(module_name)

    if train_all_transformer_blocks:
        double_indices = set(range(num_double))
    else:
        double_start = num_double - args.train_last_transformer_blocks
        double_indices = set(range(double_start, num_double))
        double_indices.update(
            parse_layer_index_spec(
                getattr(args, "train_transformer_block_indices", None),
                num_double,
                "--train_transformer_block_indices",
            )
        )
    for index in sorted(double_indices):
        _set_trainable(transformer.transformer_blocks[index])
        trainable_modules.append(f"transformer_blocks.{index}")

    if train_all_transformer_blocks:
        single_indices = set(range(num_single))
    else:
        single_start = num_single - args.train_last_single_transformer_blocks
        single_indices = set(range(single_start, num_single))
        single_indices.update(
            parse_layer_index_spec(
                getattr(args, "train_single_transformer_block_indices", None),
                num_single,
                "--train_single_transformer_block_indices",
            )
        )
    for index in sorted(single_indices):
        _set_trainable(transformer.single_transformer_blocks[index])
        trainable_modules.append(f"single_transformer_blocks.{index}")

    if args.train_output_layers or train_all_transformer_blocks:
        _set_trainable(transformer.norm_out)
        _set_trainable(transformer.proj_out)
        trainable_modules.extend(["norm_out", "proj_out"])

    if getattr(args, "train_input_layers", False):
        # Input/conditioning projections form a separate adaptation scope.
        # Modulation layers are handled by --train_non_block_layers.
        input_modules = ("pos_embed", "x_embedder", "context_embedder")
        if scope_contract == "release_colorexp":
            input_modules += ("time_guidance_embed",)
        for module_name in input_modules:
            module = getattr(transformer, module_name, None)
            if module is not None and any(True for _ in module.parameters()):
                _set_trainable(module)
                trainable_modules.append(module_name)

    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in transformer.parameters())
    if trainable_params == 0:
        raise ValueError("No transformer parameters were selected for training.")

    return list(dict.fromkeys(trainable_modules)), trainable_params, total_params


def configure_lora_training(transformer: Flux2Transformer2DModel, args):
    """Attach standard Diffusers/PEFT LoRA adapters to every Klein attention block."""
    transformer.requires_grad_(False)
    if args.lora_layers:
        target_modules = [value.strip() for value in args.lora_layers.split(",") if value.strip()]
    else:
        target_modules = ["to_k", "to_q", "to_v", "to_out.0", "to_qkv_mlp_proj"]
        target_modules.extend(
            f"single_transformer_blocks.{index}.attn.to_out"
            for index in range(len(transformer.single_transformer_blocks))
        )
    config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                        init_lora_weights="gaussian", target_modules=target_modules)
    transformer.add_adapter(config)
    trainable = sum(parameter.numel() for parameter in transformer.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in transformer.parameters())
    if trainable == 0:
        raise ValueError("LoRA configuration selected no trainable Klein parameters.")
    return target_modules, trainable, total


def build_transformer_fsdp_plugin(args):
    """Create the FSDP policy used by this entrypoint, or return None if disabled."""
    if not args.fsdp_transformer:
        return None

    from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig

    transformer_cls_names = [
        class_name.strip() for class_name in args.fsdp_transformer_cls_to_wrap.split(",") if class_name.strip()
    ]
    if not transformer_cls_names:
        raise ValueError("--fsdp_transformer_cls_to_wrap must contain at least one class name.")

    # HYBRID_SHARD over 4 GPUs/node leaves ~32 GiB of parameter shards per rank,
    # which does not fit the 9B model; FULL_SHARD is required.
    return FullyShardedDataParallelPlugin(
        fsdp_version=1,
        reshard_after_forward="FULL_SHARD",
        backward_prefetch="BACKWARD_PRE",
        auto_wrap_policy="TRANSFORMER_BASED_WRAP",
        transformer_cls_names_to_wrap=transformer_cls_names,
        use_orig_params=True,
        limit_all_gathers=True,
        cpu_offload=False,
        forward_prefetch=False,
        # FULL_STATE_DICT consolidates the whole AdamW state on rank 0 at every
        # save; SHARDED_STATE_DICT writes per-rank shards that Accelerate resumes directly.
        state_dict_type="SHARDED_STATE_DICT",
        state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
    )


def _maybe_upcast_state_dict(state_dict: dict[str, torch.Tensor], upcast: bool) -> dict[str, torch.Tensor]:
    if not upcast:
        return state_dict
    return {name: tensor.float() if tensor.is_floating_point() else tensor for name, tensor in state_dict.items()}


def save_diffusers_model_with_state_dict(
    model: Flux2Transformer2DModel,
    state_dict: dict[str, torch.Tensor],
    save_directory: str,
    *,
    variant: str | None = None,
    safe_serialization: bool = True,
    max_shard_size: int | str = "10GB",
):
    """Save a Diffusers model using an externally gathered state dict.

    Diffusers' ``save_pretrained`` calls ``model.state_dict()`` internally.
    For FSDP that would re-enter state-dict gathering in the wrong place, so we
    mirror the small save routine while using Accelerate's gathered state dict.
    """
    if os.path.isfile(save_directory):
        raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file.")

    weights_name = SAFETENSORS_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
    weights_name = _add_variant(weights_name, variant)
    weights_name_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(
        ".safetensors", "{suffix}.safetensors"
    )

    os.makedirs(save_directory, exist_ok=True)
    model.save_config(save_directory)

    state_dict_split = split_torch_state_dict_into_shards(
        state_dict, max_shard_size=max_shard_size, filename_pattern=weights_name_pattern
    )

    index_names = {_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant), _add_variant(WEIGHTS_INDEX_NAME, variant)}
    for filename in os.listdir(save_directory):
        if filename in state_dict_split.filename_to_tensors:
            continue
        full_filename = os.path.join(save_directory, filename)
        if not os.path.isfile(full_filename):
            continue
        if filename in index_names or (
            filename.startswith("diffusion_pytorch_model")
            and (filename.endswith(".safetensors") or filename.endswith(".bin"))
        ):
            os.remove(full_filename)

    for filename, tensors in state_dict_split.filename_to_tensors.items():
        shard = {tensor_name: state_dict[tensor_name].contiguous() for tensor_name in tensors}
        filepath = os.path.join(save_directory, filename)
        if safe_serialization:
            safetensors.torch.save_file(shard, filepath, metadata={"format": "pt"})
        else:
            torch.save(shard, filepath)

    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }
        index_name = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
        index_path = os.path.join(save_directory, _add_variant(index_name, variant))
        with open(index_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(index, indent=2, sort_keys=True) + "\n")
        logging.getLogger(__name__).info(
            f"Model weights were split into {len(state_dict_split.filename_to_tensors)} shards. "
            f"Index saved to {index_path}."
        )
    else:
        logging.getLogger(__name__).info(f"Model weights saved in {os.path.join(save_directory, weights_name)}")


def parse_args(input_args=None):
    """Parse CLI arguments and enforce the dataset-selection invariants.

    The script supports three mutually exclusive data-entry modes:
    ``--dataset_specs`` for weighted multi-dataset training,
    ``--dataset_name`` for a Hugging Face dataset or local dataset folder, and
    ``--instance_data_dir`` for a simple image folder.
    """
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--train_method", choices=["lora", "partial_full_rank"], default="partial_full_rank")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_layers", default=None)
    # Base model and runtime configuration.
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--bnb_quantization_config_path",
        type=str,
        default=None,
        help="Quantization config in a JSON file that will be used to define the bitsandbytes quant config of the DiT.",
    )
    parser.add_argument(
        "--do_fp8_training",
        action="store_true",
        help="if we are doing FP8 training.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )
    parser.add_argument(
        "--dataset_specs",
        type=str,
        default=None,
        help="Path to JSON config describing multiple prepared datasets and sampling weights.",
    )
    parser.add_argument(
        "--depth_target_encoding",
        choices=(
            DEPTH_ENCODING_VISION_BANANA_RGB,
            DEPTH_ENCODING_BARRON_SCALAR_GRAYSCALE,
            DEPTH_ENCODING_RELATIVE_INVERSE_GRAYSCALE,
            DEPTH_ENCODING_RELATIVE_LINEAR_GRAYSCALE,
        ),
        default=DEPTH_ENCODING_VISION_BANANA_RGB,
        help=(
            "Depth target representation. The scalar ablation deterministically "
            "projects frozen Vision Banana RGB targets back to their bounded "
            "scalar and replicates it as grayscale."
        ),
    )
    parser.add_argument(
        "--depth_target_resize_policy",
        choices=("encoded_rgb", "metric_field_then_encode"),
        default="metric_field_then_encode",
        help="Resize frozen depth RGB directly or decode, resize metric depth, and re-encode.",
    )
    parser.add_argument(
        "--normal_target_resize_policy",
        choices=("encoded_rgb", "vector_field_renormalize_then_encode"),
        default="vector_field_renormalize_then_encode",
        help="Resize frozen normal RGB directly or resize decoded XYZ, renormalize, and re-encode.",
    )
    parser.add_argument(
        "--normal_target_encoding",
        choices=SUPPORTED_NORMAL_ENCODINGS,
        default=VISION_BANANA_NORMAL_ENCODING,
        help=(
            "RGB storage contract for surface-normal targets. Use "
            f"{LEGACY_DIRECT_NORMAL_ENCODING!r} only for frozen pre-v3 data."
        ),
    )
    parser.add_argument(
        "--depth_prompt_contract",
        choices=(
            "canonical_v2",
            "campaign_short_inverse",
            "campaign_detailed_inverse",
            "campaign_paper_inverse",
            "campaign_paper_barron",
            "row",
        ),
        default="canonical_v2",
        help=(
            "Use one task-global canonical metric camera-z prompt for every "
            "depth row, or preserve frozen row prompts for ablations."
        ),
    )
    parser.add_argument(
        "--normal_prompt_contract",
        choices=(
            "vision_banana_v3",
            "campaign_short",
            "campaign_detailed",
            "campaign_paper",
            "row",
        ),
        default="vision_banana_v3",
        help=(
            "Use the task-global v3 prompt for every normal row, or preserve "
            "the frozen row prompt for legacy/prompt-ablation reproduction."
        ),
    )
    parser.add_argument(
        "--allow_unverified_normal_sources",
        action="store_true",
        help=(
            "Legacy reproduction only: permit normal rows whose stored "
            "native-to-canonical transform has not passed an audit."
        ),
    )
    parser.add_argument(
        "--allow_legacy_hypersim_range_depth",
        action="store_true",
        help=(
            "Legacy reproduction only: permit old Hypersim depth rows that "
            "store Euclidean camera range instead of projective camera-z."
        ),
    )
    parser.add_argument(
        "--samples_per_epoch",
        type=int,
        default=None,
        help="Override samples_per_epoch in dataset_specs. Used only with --dataset_specs.",
    )
    parser.add_argument(
        "--dataset_pool_seed",
        type=int,
        default=None,
        help="Optional seed for weighted dataset sampling. Defaults to --seed when omitted.",
    )
    parser.add_argument(
        "--resample_dataset_each_epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to resample weighted dataset rows each epoch/window. Used only with --dataset_specs.",
    )
    parser.add_argument(
        "--sampling_strategy",
        choices=["with_replacement", "without_replacement"],
        default=None,
        help="Weighted-pool sampling strategy. Defaults to dataset_specs, then without_replacement.",
    )
    parser.add_argument(
        "--max_refseg_expressions_per_image",
        type=int,
        default=None,
        help="Deterministically retain at most this many distinct RefSeg expressions per conditioning image.",
    )
    parser.add_argument(
        "--task_homogeneous_batches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, batches are grouped by bucket and task/source instead of only by bucket.",
    )
    parser.add_argument(
        "--prompt_encoding_mode",
        choices=["auto", "on_the_fly", "precompute", "packed_precompute"],
        default="auto",
        help=(
            "How per-row prompts are encoded. 'auto' uses on-the-fly encoding for streaming WebDataset "
            "sources and precompute otherwise. 'packed_precompute' caches all eligible unique prompts in "
            "sharded safetensors addressed by prompt hash."
        ),
    )
    parser.add_argument(
        "--prompt_precompute_max_unique",
        type=int,
        default=2048,
        help=(
            "Maximum number of unique per-row prompts that 'auto' may precompute. "
            "Larger prompt sets are encoded on the fly to avoid creating a very large "
            "disk-backed tensor cache. Set to 0 to make 'auto' always encode custom "
            "prompts on the fly. Explicit precompute is refused above this limit unless "
            "--allow_large_prompt_precompute_cache is also passed."
        ),
    )
    parser.add_argument(
        "--allow_large_prompt_precompute_cache",
        action="store_true",
        help=(
            "Permit explicit precompute mode to exceed --prompt_precompute_max_unique. "
            "This can consume many tens of thousands of inodes and hundreds of GB; "
            "use only after reviewing the cache estimate and filesystem quota."
        ),
    )
    parser.add_argument(
        "--packed_prompt_shard_bytes",
        type=int,
        default=1 << 30,
        help="Approximate maximum size of each packed prompt safetensors shard.",
    )
    parser.add_argument(
        "--validation_prompts_file",
        type=str,
        default=None,
        help="Optional UTF-8 text/JSONL file of fixed text-only validation prompts.",
    )
    parser.add_argument(
        "--validation_num_inference_steps",
        type=int,
        default=28,
        help="Diffusion steps used for fixed text-only validation prompts.",
    )
    parser.add_argument("--validation_height", type=int, default=None)
    parser.add_argument("--validation_width", type=int, default=None)

    # Dataset storage and schema options.
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--cond_image_column",
        type=str,
        default=None,
        help="Column in the dataset containing the condition image. Must be specified when performing I2I fine-tuning",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    # Prompts, validation, and trainable-layer selection.
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=False,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        help="path to an image that is used during validation as the condition image to verify that the model is learning.",
    )
    parser.add_argument(
        "--skip_final_inference",
        default=False,
        action="store_true",
        help="Whether to skip the final inference step with the fine-tuned transformer upon training completion.",
    )
    parser.add_argument(
        "--final_validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during a final validation to verify that the model is learning. Ignored if `--validation_prompt` is provided.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--train_full_transformer",
        action="store_true",
        help=(
            "Fine-tune every parameter inside the transformer, including double/single blocks, input embedders, "
            "conditioning embedders, modulation layers, and output layers. This is the literal full-transformer "
            "mode and uses the most optimizer memory. It overrides partial block selection and "
            "--train_all_transformer_blocks."
        ),
    )
    parser.add_argument(
        "--train_all_transformer_blocks",
        action="store_true",
        help=(
            "Fine-tune all double-stream transformer_blocks, all single_transformer_blocks, and norm_out/proj_out "
            "output layers, but do not automatically train input embedders or other non-block modulation layers. "
            "This matches the older temp/open_visionbanana full-block behavior. Use --train_full_transformer "
            "instead when you really want every transformer parameter trainable."
        ),
    )
    parser.add_argument(
        "--train_non_block_layers",
        action="store_true",
        help=(
            "Also fine-tune selected non-block transformer layers: input/context embedders and timestep/guidance "
            "modulation layers. This is useful when partial block training is too narrow, but it is still not the "
            "same as --train_full_transformer."
        ),
    )
    parser.add_argument(
        "--train_last_transformer_blocks",
        type=int,
        default=2,
        help="Number of final double-stream transformer_blocks to fine-tune.",
    )
    parser.add_argument(
        "--train_transformer_block_indices",
        type=str,
        default=None,
        help=(
            "Comma-separated zero-based double-stream transformer_blocks to fine-tune in addition to "
            "--train_last_transformer_blocks. Negative indices count from the end, e.g. '0,6,12,-1'."
        ),
    )
    parser.add_argument(
        "--train_last_single_transformer_blocks",
        type=int,
        default=4,
        help="Number of final single_transformer_blocks to fine-tune.",
    )
    parser.add_argument(
        "--train_single_transformer_block_indices",
        type=str,
        default=None,
        help=(
            "Comma-separated zero-based single_transformer_blocks to fine-tune in addition to "
            "--train_last_single_transformer_blocks. Negative indices count from the end, e.g. '0,8,16,-1'."
        ),
    )
    parser.add_argument(
        "--train_output_layers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fine-tune final norm_out and proj_out layers. Enabled by default for partial runs. "
            "--train_all_transformer_blocks always trains these output layers even if this is disabled."
        ),
    )
    parser.add_argument(
        "--train_input_layers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also fine-tune input/conditioning embedders. Off by default for existing partial runs; "
            "use with FSDP full-finetune jobs when the root FSDP unit must be uniformly trainable."
        ),
    )

    # Core training schedule.
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux2-klein-partial-finetune",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--aspect_ratio_buckets",
        type=str,
        default=None,
        help=(
            "Aspect ratio buckets to use for training. Define as a string of 'h1,w1;h2,w2;...'. "
            "e.g. '1024,1024;768,1360;1360,768;880,1168;1168,880;1248,832;832,1248'"
            "Images will be resized and cropped to fit the nearest bucket. If provided, --resolution is ignored."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--prompt_embedding_batch_size",
        type=int,
        default=None,
        help="Batch size used only while precomputing text embeddings for custom prompts. Defaults to train_batch_size.",
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--permanent_checkpoint_steps",
        type=str,
        default="",
        help="Comma-separated checkpoint steps that are never removed by rolling retention.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--allow_resume_code_change",
        action="store_true",
        help=(
            "Permit resuming a recoverable checkpoint after an intentional change to the audited "
            "trainer/loader code. Dataset specification and base-model provenance remain immutable."
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )

    # Optimizer and scheduler settings.
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--dataset_index_num_workers",
        type=int,
        default=1,
        help=(
            "Number of threads to use while indexing dataset rows into aspect-ratio buckets. "
            "Uses a size-only image read path and defaults to 1."
        ),
    )
    parser.add_argument(
        "--dataloader_timeout_seconds",
        type=float,
        default=0.0,
        help="Fail a worker queue wait after this many seconds; 0 disables the failsafe.",
    )
    parser.add_argument(
        "--row_load_timeout_seconds",
        type=float,
        default=0.0,
        help=(
            "Maximum CPU-side image/decode/transform time for one prepared row. "
            "A timed-out row is logged and replaced with a zero-loss placeholder; 0 disables this guard."
        ),
    )
    parser.add_argument(
        "--dataloader_prefetch_factor",
        type=int,
        default=2,
        help="Per-worker prefetched batches when subprocess data loading is enabled.",
    )
    parser.add_argument(
        "--dataloader_persistent_workers",
        action="store_true",
        help="Keep random-access dataloader workers alive across synthetic epochs.",
    )
    parser.add_argument(
        "--bucket_index_cache_dir",
        type=str,
        default=None,
        help=(
            "Optional durable directory for a verified row-to-aspect-ratio-bucket index. "
            "Prepared-pool restarts reuse it instead of reopening every target image header."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    # Logging, precision, checkpointing, and distributed helpers.
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
    parser.add_argument(
        "--loss_mask_min_valid_fraction",
        type=float,
        default=0.5,
        help=(
            "Minimum fraction of valid source pixels required for a downsampled latent loss position "
            "to contribute when a row provides loss_mask. 0.5 keeps majority-valid latent positions."
        ),
    )
    parser.add_argument(
        "--loss_mask_mode", type=str, default="hard", choices=["hard", "soft"],
        help="Use the historical hard latent threshold or fractional area coverage for masked loss.",
    )
    parser.add_argument(
        "--loss_mask_erosion_radius", type=int, default=0,
        help="Erode transformed image-space validity masks by this many pixels before latent downsampling.",
    )
    parser.add_argument(
        "--loss_mask_reduction", type=str, default="per_sample", choices=["per_sample", "global_valid"],
        help="Normalize masked loss per sample (historical) or over all valid weighted elements in the local batch.",
    )
    parser.add_argument(
        "--loss_mask_min_sample_valid_fraction", type=float, default=0.0,
        help="Exclude samples below this final latent valid-weight fraction; empty masks are always excluded.",
    )
    parser.add_argument(
        "--allow_legacy_rng_resume", action="store_true",
        help="Explicit migration of checkpoints without isolated RNG state; not exact reproduction.",
    )
    parser.add_argument(
        "--trainable_scope_contract", choices=["release_colorexp", "separate_embeddings"],
        default="release_colorexp",
        help="Preserve release parameter scopes by default; separate_embeddings independently controls input/context embeddings.",
    )
    parser.add_argument(
        "--allow_missing_geometry_masks", action="store_true",
        help="Legacy reproduction only: permit geometry rows without source validity masks, even for unmasked loss.",
    )
    parser.add_argument(
        "--loss_mask_policy",
        type=str,
        default="disabled",
        choices=["auto", "enabled", "disabled"],
        help=(
            "Controls per-pixel loss masks from a row's loss_mask column. 'auto' uses masks when present and "
            "falls back to whole-image loss when missing; this is the current open_visionbanana behavior. "
            "'enabled' requires every row to provide loss_mask and fails early if any row is missing one. "
            "'disabled' computes whole-image loss; geometry preprocessing still uses source validity "
            "to avoid interpolating undefined fill values into valid targets."
        ),
    )
    parser.add_argument("--palette_manifold_loss_weight", type=float, default=0.0)
    parser.add_argument("--unit_normal_loss_weight", type=float, default=0.0)
    parser.add_argument("--angular_normal_aux_weight", type=float, default=0.0)
    parser.add_argument("--metric_depth_aux_weight", type=float, default=0.0)
    parser.add_argument(
        "--metric_depth_aux_max_depth",
        type=float,
        default=80.0,
        help="Maximum decoded depth used by the robust log-metric auxiliary.",
    )
    parser.add_argument(
        "--precompute_cache_dir",
        type=str,
        default=None,
        help=(
            "Directory for disk-backed latent/text caches. "
            "Defaults to <output_dir>/precomputed_cache."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "By default, the in-memory training dtypes are preserved."
        ),
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Whether to offload the VAE and the text encoder to CPU when they are not used.",
    )

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--enable_npu_flash_attention", action="store_true", help="Enabla Flash Attention for NPU")
    parser.add_argument("--fsdp_text_encoder", action="store_true", help="Use FSDP for text encoder")
    parser.add_argument(
        "--fsdp_transformer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use Accelerate FSDP full sharding for the transformer when this script is launched with multiple "
            "processes. Disabled by default so the canonical trainer keeps the original non-FSDP behavior unless "
            "a job explicitly enables it."
        ),
    )
    parser.add_argument(
        "--fsdp_transformer_cls_to_wrap",
        type=str,
        default="Flux2TransformerBlock,Flux2SingleTransformerBlock",
        help=(
            "Comma-separated transformer block class names to wrap with FSDP. The default matches Flux2-Klein."
        ),
    )
    parser.add_argument(
        "--fsdp_transformer_dtype_policy",
        type=str,
        default="fp32_all",
        choices=["auto", "bf16", "fp32_trainable", "fp32_all"],
        help=(
            "Controls transformer parameter dtypes when mixed precision and --fsdp_transformer are used. "
            "'auto' preserves current open_visionbanana behavior: partial FSDP keeps transformer params in the "
            "mixed-precision dtype for low memory, while non-FSDP or literal full-transformer training upcasts "
            "trainable params for optimizer stability. 'bf16' keeps FSDP transformer params in the mixed-precision "
            "dtype, usually bf16, minimizing memory but giving trainable params no fp32 master copy. "
            "'fp32_trainable' upcasts only trainable params; this can improve optimizer precision but may trigger "
            "FSDP uniform-dtype errors in partial runs if a wrapped unit mixes frozen bf16 and trainable fp32 params. "
            "'fp32_all' upcasts the whole transformer to fp32 under FSDP, matching temp/open_visionbanana's robust "
            "uniform-fp32 behavior at roughly 2x parameter memory."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_specs is not None:
        if args.cond_image_column is None:
            args.cond_image_column = "conditioning_image"
        if args.caption_column is None:
            args.caption_column = "text"

    if args.cond_image_column is not None and args.image_column is None:
        raise ValueError("`--cond_image_column` requires `--image_column`.")
    if not 0.0 < args.loss_mask_min_valid_fraction <= 1.0:
        raise ValueError("`--loss_mask_min_valid_fraction` must be in the interval (0, 1].")
    if args.loss_mask_erosion_radius < 0:
        raise ValueError("`--loss_mask_erosion_radius` must be non-negative.")
    if not 0.0 <= args.loss_mask_min_sample_valid_fraction <= 1.0:
        raise ValueError("`--loss_mask_min_sample_valid_fraction` must be in the interval [0, 1].")
    if args.dataloader_timeout_seconds < 0.0:
        raise ValueError("`--dataloader_timeout_seconds` must be non-negative.")
    if args.row_load_timeout_seconds < 0.0:
        raise ValueError("`--row_load_timeout_seconds` must be non-negative.")
    if args.dataloader_prefetch_factor < 1:
        raise ValueError("`--dataloader_prefetch_factor` must be at least 1.")
    auxiliary_weights = {
        "palette": args.palette_manifold_loss_weight,
        "unit_normal": args.unit_normal_loss_weight,
        "angular_normal": args.angular_normal_aux_weight,
        "metric_depth": args.metric_depth_aux_weight,
    }
    if any(weight < 0.0 for weight in auxiliary_weights.values()):
        raise ValueError("Geometry auxiliary-loss weights must be non-negative.")
    if sum(weight > 0.0 for weight in auxiliary_weights.values()) > 1:
        raise ValueError("Enable at most one geometry auxiliary loss per scientific run.")
    if args.cache_latents and any(weight > 0.0 for weight in auxiliary_weights.values()):
        raise ValueError("Decoded geometry auxiliary losses require `--no-cache_latents`.")
    if any(weight > 0.0 for weight in auxiliary_weights.values()) and not args.task_homogeneous_batches:
        raise ValueError("Geometry auxiliary losses require --task_homogeneous_batches; mixed-task batches can silently skip the auxiliary.")
    if args.metric_depth_aux_max_depth <= 0.0:
        raise ValueError("`--metric_depth_aux_max_depth` must be positive.")
    if (
        args.palette_manifold_loss_weight > 0.0
        or args.metric_depth_aux_weight > 0.0
    ) and args.depth_target_encoding != DEPTH_ENCODING_VISION_BANANA_RGB:
        raise ValueError("Depth RGB auxiliaries require `--depth_target_encoding vision_banana_rgb`.")
    if args.max_refseg_expressions_per_image is not None and args.max_refseg_expressions_per_image < 1:
        raise ValueError("`--max_refseg_expressions_per_image` must be at least 1.")
    if args.packed_prompt_shard_bytes <= 0:
        raise ValueError("`--packed_prompt_shard_bytes` must be positive.")
    if args.validation_num_inference_steps <= 0:
        raise ValueError("`--validation_num_inference_steps` must be positive.")
    if (args.validation_height is None) != (args.validation_width is None):
        raise ValueError("Set both --validation_height and --validation_width, or neither.")
    if args.train_full_transformer and args.train_all_transformer_blocks:
        raise ValueError("Choose only one of `--train_full_transformer` or `--train_all_transformer_blocks`.")

    dataset_sources = [
        args.dataset_specs is not None,
        args.dataset_name is not None,
        args.instance_data_dir is not None,
    ]
    if sum(dataset_sources) != 1:
        raise ValueError("Specify exactly one of `--dataset_specs`, `--dataset_name`, or `--instance_data_dir`.")

    if args.samples_per_epoch is not None and args.dataset_specs is None:
        raise ValueError("`--samples_per_epoch` can only be used with `--dataset_specs`.")
    if args.dataset_specs is not None:
        specs_path = Path(args.dataset_specs).expanduser()
        if specs_path.is_file():
            raw_config = json.loads(specs_path.read_text(encoding="utf-8"))
            has_streaming = any(
                isinstance(spec, dict) and spec.get("type") == "webdataset_text_to_image"
                for spec in (raw_config.get("datasets", []) if isinstance(raw_config, dict) else [])
            )
            if has_streaming and args.cache_latents:
                raise ValueError(
                    "--cache_latents is incompatible with streaming WebDataset sources; "
                    "use on-the-fly VAE encoding."
                )
            if has_streaming and args.prompt_encoding_mode == "precompute":
                raise ValueError(
                    "--prompt_encoding_mode precompute is incompatible with streaming WebDataset sources."
                )

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


SEGMENTATION_TASKS = {"semantic_segmentation", "instance_segmentation", "referring_segmentation"}
TEXT_ONLY_TASKS = {"text_to_image", "text2image", "image_generation"}
DEPTH_TASKS = {"depth_estimation", "monocular_depth_estimation"}
SURFACE_NORMAL_TASKS = {"surface_normal_estimation", "surface_normals", "normal_estimation", "surface_normal"}

# This field carries both interpolation labels and geometry-preparation labels;
# accept the known legacy and canonical names only. The geometry conversion
# itself is selected later from the task-level resize contract.
LEGACY_BILINEAR_PREPROCESS_TARGET_POLICIES = {
    "resize_metric_depth_then_encode",
    "resize_normal_vector_then_encode",
    "habitat_depth_sensor_metric_then_vision_banana_rgb_encode",
    "habitat_depth_sensor_metric_then_grayscale_log_encode",
    "rendered_or_depth_derived_camera_space_normals",
    "taskonomy_depth_decode_metric_resize_then_vision_banana_rgb_encode",
    "taskonomy_depth_decode_metric_resize_then_grayscale_log_encode",
    "resize_xyz_then_renormalize_then_encode",
    "native_xyz_then_normalize_then_encode",
}


@dataclass
class DatasetSpec:
    """One folder-backed or streaming entry from ``--dataset_specs``."""

    name: str
    path: Path | None
    weight: float
    type: str = "prepared_folder"
    shards: str | list[str] | None = None
    task: str | None = None
    source_dataset: str | None = None
    streaming: dict[str, Any] | None = None
    index_path: Path | None = None
    sampling: str = "row"
    filters: dict[str, Any] | None = None


@dataclass
class PreparedRow:
    """Lightweight metadata for one training row before images are loaded lazily."""

    dataset_name: str
    dataset_root: Path | None
    row_id: str | None
    conditioning_image: Any
    image: Any
    text: str
    loss_mask: Any | None = None
    task: str | None = None
    source_dataset: str | None = None
    preprocess_target: str | None = None
    referential_expression: str | None = None
    scene: str | None = None
    normal_encoding: str | None = None
    geometry_version: str | None = None
    canonical_depth_quantity: str | None = None
    stored_normal_permutation: tuple[int, int, int] = (0, 1, 2)
    stored_normal_signs: tuple[int, int, int] = (1, 1, 1)
    normal_contract_status: str | None = None


def prepared_row_normal_encoding(raw_row: dict[str, Any]) -> str | None:
    """Resolve explicit v3/v2 normal storage, conservatively treating old rows as v2."""
    if normalize_task_name(raw_row.get("task")) not in SURFACE_NORMAL_TASKS:
        return None
    metadata = raw_row.get("metadata") or {}
    candidate = raw_row.get("normal_encoding") or (
        metadata.get("normal_encoding") if isinstance(metadata, dict) else None
    )
    if candidate in SUPPORTED_NORMAL_ENCODINGS:
        return str(candidate)
    geometry_version = raw_row.get("geometry_version") or (
        metadata.get("geometry_version") if isinstance(metadata, dict) else None
    )
    if geometry_version == "geometry_vision_banana_v3":
        return VISION_BANANA_NORMAL_ENCODING
    # Frozen v2 archives used formula strings (or no representation metadata).
    return LEGACY_DIRECT_NORMAL_ENCODING


def prepared_row_geometry_contract(raw_row: dict[str, Any]) -> dict[str, Any]:
    """Resolve whether a frozen geometry row is canonical and how to repair it."""
    metadata = raw_row.get("metadata") or {}
    metadata = metadata if isinstance(metadata, dict) else {}
    task = normalize_task_name(raw_row.get("task"))
    source = str(
        raw_row.get("source_dataset") or metadata.get("dataset") or ""
    ).strip().lower()
    geometry_version = raw_row.get("geometry_version") or metadata.get(
        "geometry_version"
    )
    canonical_depth_quantity = raw_row.get(
        "canonical_depth_quantity"
    ) or metadata.get("canonical_depth_quantity")
    contract = {
        "geometry_version": (
            str(geometry_version) if geometry_version is not None else None
        ),
        "canonical_depth_quantity": (
            str(canonical_depth_quantity)
            if canonical_depth_quantity is not None
            else None
        ),
        "stored_normal_permutation": (0, 1, 2),
        "stored_normal_signs": (1, 1, 1),
        "normal_contract_status": None,
    }
    if task not in SURFACE_NORMAL_TASKS:
        return contract

    source_versions = {
        "nyuv2": "nyuv2_geonet_dsine_v1",
        "diode": "diode_identity_reproduced_v2",
    }
    if source in source_versions and geometry_version == source_versions[source]:
        if metadata.get("canonical_normal_convention") == "camera_x_right_y_up_z_toward_camera":
            contract["normal_contract_status"] = "versioned_canonical"
            return contract

    if geometry_version in {
        "geometry_canonical_v2",
        "geometry_vision_banana_v3",
        "geometry_camera_facing_v4",
    }:
        contract["normal_contract_status"] = "versioned_canonical"
        return contract

    # Stored-axis transforms of the pre-v2 archives; see dataset_prep.geometry_conventions.
    if source in {"hypersim", "matterport3d"}:
        contract["normal_contract_status"] = "legacy_audited_identity"
    elif source in {"synscapes", "replica"}:
        contract["stored_normal_signs"] = (-1, 1, 1)
        contract["normal_contract_status"] = "legacy_audited_depth_derived"
    else:
        contract["normal_contract_status"] = "unverified"
    return contract


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries with friendly errors."""
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} of {path} is not a JSON object.")
            rows.append(row)
    return rows


def normalize_task_name(task: Any) -> str | None:
    """Normalize task labels so downstream routing can compare stable strings."""
    if task is None:
        return None
    task = str(task).strip().lower().replace("-", "_").replace(" ", "_")
    return task or None


def load_dataset_specs(specs_path: str, samples_per_epoch_override: int | None = None) -> tuple[list[DatasetSpec], int, int, dict[str, Any]]:
    """Load a weighted dataset-pool config file.

    Returns:
        ``(specs, samples_per_epoch, seed, raw_config)`` where ``specs`` is the
        validated list of dataset directories.
    """
    path = Path(specs_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset specs file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Dataset specs file must contain a JSON object: {path}")

    raw_specs = config.get("datasets")
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError(f"Dataset specs file must contain a non-empty `datasets` list: {path}")

    samples_per_epoch = samples_per_epoch_override if samples_per_epoch_override is not None else config.get("samples_per_epoch")
    if samples_per_epoch is None:
        raise ValueError("`samples_per_epoch` must be set in dataset_specs or passed via --samples_per_epoch.")
    samples_per_epoch = int(samples_per_epoch)
    if samples_per_epoch <= 0:
        raise ValueError(f"`samples_per_epoch` must be positive, got {samples_per_epoch}.")

    seed = int(config.get("seed", 0))

    def resolve_local_spec_path(value: Any) -> Path:
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        return candidate.resolve()

    specs = []
    seen_names = set()
    for index, raw_spec in enumerate(raw_specs):
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Dataset spec at index {index} must be a JSON object.")
        name = str(raw_spec.get("name", "")).strip()
        if not name:
            raise ValueError(f"Dataset spec at index {index} is missing a non-empty `name`.")
        if name in seen_names:
            raise ValueError(f"Duplicate dataset spec name: {name}")
        seen_names.add(name)

        weight = float(raw_spec.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"Dataset spec '{name}' has non-positive weight: {weight}.")
        dataset_type = str(raw_spec.get("type", "prepared_folder")).strip().lower()
        if dataset_type == "prepared_folder":
            if "path" not in raw_spec:
                raise ValueError(f"Dataset spec '{name}' is missing required field `path`.")
            dataset_path = resolve_local_spec_path(raw_spec["path"])
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset spec '{name}' path does not exist: {dataset_path}")
            metadata_path = dataset_path / "metadata.jsonl"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Dataset spec '{name}' is missing metadata.jsonl at {metadata_path}")
            specs.append(DatasetSpec(name=name, path=dataset_path, weight=weight))
        elif dataset_type == "indexed_prepared_tar":
            if "path" not in raw_spec or "index_path" not in raw_spec:
                raise ValueError(
                    f"Dataset spec '{name}' must set `path` and `index_path` for indexed_prepared_tar."
                )
            dataset_path = resolve_local_spec_path(raw_spec["path"])
            index_path = resolve_local_spec_path(raw_spec["index_path"])
            if not dataset_path.is_file():
                raise FileNotFoundError(f"Dataset spec '{name}' tar does not exist: {dataset_path}")
            if not index_path.is_file():
                raise FileNotFoundError(f"Dataset spec '{name}' index does not exist: {index_path}")
            specs.append(
                DatasetSpec(
                    name=name,
                    path=dataset_path,
                    index_path=index_path,
                    weight=weight,
                    type=dataset_type,
                    sampling=str(raw_spec.get("sampling", "row")),
                    filters=raw_spec.get("filters"),
                )
            )
        elif dataset_type == "webdataset_text_to_image":
            shards = raw_spec.get("shards")
            if not isinstance(shards, (str, list)) or not shards:
                raise ValueError(f"Dataset spec '{name}' must set non-empty `shards`.")
            try:
                from braceexpand import braceexpand
            except ImportError as exc:
                raise ImportError(
                    "Streaming dataset specs require braceexpand==0.1.7 and webdataset==1.0.2."
                ) from exc
            patterns = [shards] if isinstance(shards, str) else shards
            expanded_shards = [candidate for pattern in patterns for candidate in braceexpand(str(pattern))]
            resolved_shards = [str(resolve_local_spec_path(candidate)) for candidate in expanded_shards]
            missing_shards = [candidate for candidate in resolved_shards if not Path(candidate).is_file()]
            if missing_shards:
                preview = ", ".join(missing_shards[:3])
                extra = "" if len(missing_shards) <= 3 else f" (and {len(missing_shards) - 3} more)"
                raise FileNotFoundError(f"Dataset spec '{name}' is missing local shards: {preview}{extra}")
            task = normalize_task_name(raw_spec.get("task", "text_to_image"))
            if task not in TEXT_ONLY_TASKS:
                raise ValueError(f"Dataset spec '{name}' streaming task must be text-only, got {task!r}.")
            streaming = raw_spec.get("streaming", {})
            if not isinstance(streaming, dict):
                raise ValueError(f"Dataset spec '{name}' field `streaming` must be an object.")
            for flag_name in ("resampled", "shard_shuffle"):
                if flag_name in streaming and not isinstance(streaming[flag_name], bool):
                    raise ValueError(f"Dataset spec '{name}' streaming.{flag_name} must be boolean.")
            if int(streaming.get("sample_shuffle_buffer", 10000)) <= 0:
                raise ValueError(f"Dataset spec '{name}' sample_shuffle_buffer must be positive.")
            error_policy = streaming.get("error_policy", "warn_and_continue")
            if error_policy not in {"warn_and_continue", "raise"}:
                raise ValueError(f"Dataset spec '{name}' has unsupported error_policy={error_policy!r}.")
            specs.append(
                DatasetSpec(
                    name=name,
                    path=None,
                    weight=weight,
                    type=dataset_type,
                    shards=resolved_shards,
                    task=task,
                    source_dataset=str(raw_spec.get("source_dataset", "pd12m")),
                    streaming=streaming,
                )
            )
        else:
            raise ValueError(f"Dataset spec '{name}' has unsupported type={dataset_type!r}.")

    return specs, samples_per_epoch, seed, config


class PreparedDatasetIndex:
    """Validate and expose the rows for one prepared dataset directory."""

    def __init__(
        self,
        spec: DatasetSpec,
        image_column: str,
        cond_image_column: str,
        caption_column: str,
        max_refseg_expressions_per_image: int | None = None,
        seed: int = 0,
    ):
        """Load row metadata and remember which columns hold each required field."""
        self.spec = spec
        self.image_column = image_column
        self.cond_image_column = cond_image_column
        self.caption_column = caption_column
        self.max_refseg_expressions_per_image = max_refseg_expressions_per_image
        self.seed = int(seed)
        if self.spec.path is None:
            raise ValueError(f"PreparedDatasetIndex cannot open streaming dataset '{self.spec.name}'.")
        self.rows = self._load_rows()
        if self.max_refseg_expressions_per_image is not None and self._is_refseg_source():
            self.rows = self._limit_refseg_expressions(self.rows)

    def _is_refseg_source(self) -> bool:
        return "refseg" in self.spec.name.lower() or any(row.task == "refseg" for row in self.rows)

    def _limit_refseg_expressions(self, rows: list[PreparedRow]) -> list[PreparedRow]:
        limit = int(self.max_refseg_expressions_per_image)
        if limit < 1:
            raise ValueError("max_refseg_expressions_per_image must be at least 1")
        for ordinal, row in enumerate(rows):
            if row.conditioning_image in (None, ""):
                raise ValueError(
                    f"RefSeg row {ordinal + 1} in '{self.spec.name}' has no conditioning image."
                )
        selected = set(
            limit_distinct_per_group(
                rows,
                lambda row: str(row.conditioning_image),
                lambda row: row.referential_expression or row.text,
                limit,
                self.seed,
            )
        )
        limited = [row for ordinal, row in enumerate(rows) if ordinal in selected]
        log_info(
            f"RefSeg expression cap retained {len(limited):,}/{len(rows):,} rows "
            f"(max {limit} per conditioning image)."
        )
        return limited

    def _require_field(self, row: dict[str, Any], row_number: int, field: str) -> Any:
        """Return a required metadata field or raise a row-specific error."""
        value = row.get(field)
        if value is None or value == "":
            raise ValueError(f"Row {row_number} in dataset '{self.spec.name}' is missing required field '{field}'.")
        return value

    def _load_rows(self) -> list[PreparedRow]:
        """Convert ``metadata.jsonl`` rows into normalized ``PreparedRow`` objects."""
        if self.spec.type == "indexed_prepared_tar":
            if self.spec.index_path is None:
                raise ValueError(f"Indexed dataset '{self.spec.name}' has no index_path")
            raw_rows, _ = load_index(self.spec.index_path, self.spec.path)
            prepared_rows = []
            for row_number, raw_row in enumerate(raw_rows, start=1):
                if self.spec.filters and any(
                    str(raw_row.get(field) or (raw_row.get("metadata") or {}).get(field)) != str(expected)
                    for field, expected in self.spec.filters.items()
                ):
                    continue
                image = self._require_field(raw_row, row_number, self.image_column)
                text = self._require_field(raw_row, row_number, self.caption_column)
                task = normalize_task_name(raw_row.get("task"))
                conditioning_image = raw_row.get(self.cond_image_column)
                if conditioning_image in (None, "") and task not in TEXT_ONLY_TASKS:
                    raise ValueError(
                        f"Row {row_number} in dataset '{self.spec.name}' is missing conditioning image."
                    )
                prepared_rows.append(
                    # Contract fields are resolved once here so every worker
                    # applies the same deterministic legacy repair.
                    PreparedRow(
                        dataset_name=self.spec.name,
                        dataset_root=None,
                        row_id=str(raw_row.get("id")) if raw_row.get("id") is not None else None,
                        conditioning_image=conditioning_image,
                        image=image,
                        loss_mask=raw_row.get("loss_mask") or raw_row.get("valid_mask"),
                        text=str(text),
                        task=task,
                        source_dataset=raw_row.get("source_dataset"),
                        preprocess_target=raw_row.get("preprocess_target"),
                        referential_expression=str(
                            raw_row.get("referential_expression")
                            or (raw_row.get("metadata") or {}).get("referential_expression")
                            or text
                        ),
                        scene=str(raw_row.get("scene") or (raw_row.get("metadata") or {}).get("scene") or "") or None,
                        normal_encoding=prepared_row_normal_encoding(raw_row),
                        **prepared_row_geometry_contract(raw_row),
                    )
                )
            if not prepared_rows:
                raise ValueError(f"Dataset spec '{self.spec.name}' has no indexed rows.")
            return prepared_rows
        metadata_path = self.spec.path / "metadata.jsonl"
        prepared_rows = []
        with metadata_path.open("r", encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {row_number} of {metadata_path}: {exc}") from exc
                if not isinstance(raw_row, dict):
                    raise ValueError(f"Line {row_number} of {metadata_path} is not a JSON object.")

                image = self._require_field(raw_row, row_number, self.image_column)
                text = self._require_field(raw_row, row_number, self.caption_column)
                task = normalize_task_name(raw_row.get("task"))
                conditioning_image = raw_row.get(self.cond_image_column)
                if conditioning_image in (None, "") and task not in TEXT_ONLY_TASKS:
                    raise ValueError(
                        f"Row {row_number} in dataset '{self.spec.name}' is missing required field "
                        f"'{self.cond_image_column}' for conditioned task {task!r}."
                    )
                if conditioning_image == "":
                    conditioning_image = None
                prepared_rows.append(
                    PreparedRow(
                        dataset_name=self.spec.name,
                        dataset_root=self.spec.path,
                        row_id=str(raw_row.get("id")) if raw_row.get("id") is not None else None,
                        conditioning_image=conditioning_image,
                        image=image,
                        loss_mask=raw_row.get("loss_mask") or raw_row.get("valid_mask"),
                        text=str(text),
                        task=task,
                        source_dataset=raw_row.get("source_dataset"),
                        preprocess_target=raw_row.get("preprocess_target"),
                        referential_expression=str(
                            raw_row.get("referential_expression")
                            or (raw_row.get("metadata") or {}).get("referential_expression")
                            or text
                        ),
                        scene=str(raw_row.get("scene") or (raw_row.get("metadata") or {}).get("scene") or "") or None,
                        normal_encoding=prepared_row_normal_encoding(raw_row),
                        **prepared_row_geometry_contract(raw_row),
                    )
                )
        if not prepared_rows:
            raise ValueError(f"Dataset spec '{self.spec.name}' has no rows in {metadata_path}.")
        return prepared_rows


class WeightedDatasetPool:
    """Sample rows from several prepared datasets according to per-dataset weights."""

    def __init__(
        self,
        specs: list[DatasetSpec],
        samples_per_epoch: int,
        seed: int,
        image_column: str,
        cond_image_column: str,
        caption_column: str,
        shuffle: bool = True,
        sampling_strategy: str = "without_replacement",
        max_refseg_expressions_per_image: int | None = None,
    ):
        """Load every prepared dataset index and keep the sampling parameters."""
        self.specs = specs
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.shuffle = shuffle
        if sampling_strategy not in {"with_replacement", "without_replacement"}:
            raise ValueError(f"Unsupported sampling_strategy={sampling_strategy!r}.")
        self.sampling_strategy = sampling_strategy
        self.max_refseg_expressions_per_image = max_refseg_expressions_per_image
        self.indices = [
            PreparedDatasetIndex(
                spec,
                image_column,
                cond_image_column,
                caption_column,
                max_refseg_expressions_per_image=max_refseg_expressions_per_image,
                seed=seed,
            )
            for spec in self.specs
            if spec.type in {"prepared_folder", "indexed_prepared_tar"}
        ]
        if len(self.indices) != len(self.specs):
            raise ValueError("WeightedDatasetPool accepts prepared_folder and indexed_prepared_tar specs only.")
        self.last_sample_counts: dict[str, int] = {}
        self.rows: list[PreparedRow] = []
        self.source_pool_indices: dict[str, list[int]] = {}
        for index in self.indices:
            start = len(self.rows)
            self.rows.extend(index.rows)
            self.source_pool_indices[index.spec.name] = list(range(start, len(self.rows)))

    def __len__(self):
        """Return the synthetic epoch size after weighted sampling."""
        return self.samples_per_epoch

    def dataset_sizes(self) -> dict[str, int]:
        """Report how many source rows each prepared dataset contains."""
        return {index.spec.name: len(index.rows) for index in self.indices}

    def _target_counts(self) -> dict[str, int]:
        """Convert floating-point dataset weights into integer row counts.

        The floor of each ideal count is taken first, then any leftover rows are
        assigned to the datasets with the largest fractional remainders so the
        totals still sum to ``samples_per_epoch``.
        """
        total_weight = sum(index.spec.weight for index in self.indices)
        raw_counts = [
            (index, self.samples_per_epoch * index.spec.weight / total_weight) for index in self.indices
        ]
        floors = {index.spec.name: int(math.floor(count)) for index, count in raw_counts}
        remainder = self.samples_per_epoch - sum(floors.values())
        fractions = sorted(
            ((count - math.floor(count), index.spec.name) for index, count in raw_counts),
            reverse=True,
        )
        for _, name in fractions[:remainder]:
            floors[name] += 1
        return floors

    def sample_epoch_indices(self, epoch: int) -> list[int]:
        """Draw stable full-pool row indices for one deterministic synthetic epoch."""
        rng = random.Random(self.seed + int(epoch))
        counts = self._target_counts()
        selected = []
        for index in self.indices:
            requested = counts[index.spec.name]
            population = self.source_pool_indices[index.spec.name]
            if index.spec.sampling == "scene_balanced":
                groups: dict[str, list[int]] = defaultdict(list)
                for row_number, pool_index in enumerate(population):
                    row = self.rows[pool_index]
                    groups[row.scene or f"__row_{row_number}"].append(pool_index)
                ordered_groups = list(groups.values())
                sampled = [rng.choice(ordered_groups[i % len(ordered_groups)]) for i in range(requested)]
                rng.shuffle(sampled)
            else:
                sampled = sample_indices(rng, population, requested, self.sampling_strategy)
            selected.extend(sampled)
        if self.shuffle:
            rng.shuffle(selected)
        self.last_sample_counts = dict(Counter(self.rows[pool_index].dataset_name for pool_index in selected))
        return selected

    def sample_epoch(self, epoch: int) -> list[PreparedRow]:
        return [self.rows[index] for index in self.sample_epoch_indices(epoch)]

    def resolved_config(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the effective pool settings."""
        return {
            "samples_per_epoch": self.samples_per_epoch,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "sampling_strategy": self.sampling_strategy,
            "max_refseg_expressions_per_image": self.max_refseg_expressions_per_image,
            "datasets": [
                {
                    "name": index.spec.name,
                    "path": str(index.spec.path),
                    "weight": index.spec.weight,
                    "type": index.spec.type,
                    "index_path": str(index.spec.index_path) if index.spec.index_path else None,
                    "sampling": index.spec.sampling,
                    "filters": index.spec.filters,
                    "num_rows": len(index.rows),
                }
                for index in self.indices
            ],
        }

    def log_summary(self, log_fn=None):
        """Print a human-readable summary of the loaded dataset pool."""
        log_info("Dataset pool loaded:", log_fn)
        for index in self.indices:
            log_info(f"  {index.spec.name}: {len(index.rows):,} rows, weight={index.spec.weight}", log_fn)
        log_info(
            f"  samples_per_epoch={self.samples_per_epoch:,}, seed={self.seed}, shuffle={self.shuffle}, "
            f"sampling_strategy={self.sampling_strategy}, "
            f"max_refseg_expressions_per_image={self.max_refseg_expressions_per_image}",
            log_fn,
        )


def save_resolved_dataset_specs(dataset_pool: WeightedDatasetPool, output_dir: str):
    """Write resolved dataset specs and copy known dataset-prep sidecars."""
    output_path = Path(output_dir) / "dataset_specs.resolved.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset_pool.resolved_config(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    for index in dataset_pool.indices:
        sidecar = Path(index.spec.path) / "diversify_config.json"
        if sidecar.is_file():
            try:
                shutil.copyfile(sidecar, Path(output_dir) / f"{index.spec.name}_diversify_config.json")
            except OSError:
                pass


def save_resolved_mixed_dataset_specs(
    specs: list[DatasetSpec], samples_per_epoch: int, seed: int, shuffle: bool, output_dir: str
):
    """Persist the effective folder/streaming source configuration."""
    payload = {
        "samples_per_epoch": samples_per_epoch,
        "seed": seed,
        "shuffle": shuffle,
        "datasets": [],
    }
    for spec in specs:
        item = {"name": spec.name, "type": spec.type, "weight": spec.weight}
        if spec.path is not None:
            item["path"] = str(spec.path)
            if spec.index_path is not None:
                item["index_path"] = str(spec.index_path)
            item["sampling"] = spec.sampling
            if spec.filters:
                item["filters"] = spec.filters
        else:
            item.update(
                {
                    "shards": spec.shards,
                    "task": spec.task,
                    "source_dataset": spec.source_dataset,
                    "streaming": spec.streaming or {},
                }
            )
        payload["datasets"].append(item)
    output_path = Path(output_dir) / "dataset_specs.resolved.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_training_args(args, output_dir: str):
    """Write resolved argparse settings next to checkpoints."""
    output_path = Path(output_dir) / "training_args.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _git_commit_or_none(repo_root: Path) -> str | None:
    """Return the checkout's HEAD, or None where git is unavailable.

    A missing git binary raises FileNotFoundError from the exec itself, which
    check=False does not cover. The resume guard enforces the code_sha256
    digests, not the commit hash, so provenance must never abort training.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return completed.stdout.strip() or None


def save_runtime_provenance(args, output_dir: str) -> None:
    """Freeze the code identity used by the first checkpointed training attempt.

    Enforcing the trainer hash prevents a resumed chain from mixing code versions.
    A launch that fails before creating any checkpoint may refresh the file.
    """
    output_path = Path(output_dir) / "runtime_provenance.json"
    trainer_path = Path(__file__).resolve()
    repo_root = trainer_path.parent
    tracked_paths = [
        trainer_path,
        repo_root / "common/geometry_auxiliary_losses.py",
        repo_root / "common/indexed_tar.py",
        repo_root / "common/training/bucket_index_cache.py",
        repo_root / "common/training/masked_reduction.py",
        repo_root / "common/training/resume_rng.py",
        repo_root / "common/training/fsdp_resume.py",
        repo_root / "common/surface_normal_encoding.py",
        repo_root / "common/geometry_protocol_prompts.py",
        repo_root / "dataset_prep/geometry_conventions.py",
        repo_root / "dataset_prep/depth/encoding.py",
        repo_root / "scripts/training/run_open_visionbanana_pool_flux2_klein.sh",
    ]
    code_sha256 = {
        str(path.relative_to(repo_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tracked_paths
        if path.is_file()
    }
    dataset_specs_path = Path(args.dataset_specs).resolve() if args.dataset_specs else None
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_or_none(repo_root),
        "code_sha256": code_sha256,
        "dataset_specs_sha256": (
            hashlib.sha256(dataset_specs_path.read_bytes()).hexdigest()
            if dataset_specs_path is not None and dataset_specs_path.is_file()
            else None
        ),
        "base_model": args.pretrained_model_name_or_path,
        "torch_version": torch.__version__,
        "diffusers_version": diffusers.__version__,
        "transformers_version": transformers.__version__,
    }
    if output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        immutable = ("code_sha256", "dataset_specs_sha256", "base_model")
        mismatches = [key for key in immutable if existing.get(key) != payload.get(key)]
        if mismatches:
            # A directory left by a failed save (FSDP writes model state before
            # optimizer/RNG state) is not resumable and must not block a refresh.
            # Cover both the legacy full-state files and the sharded layout.
            def has_recoverable_state(checkpoint_path: Path) -> bool:
                scheduler = checkpoint_path / "scheduler.bin"
                random_states = list(checkpoint_path.glob("random_states_*.pkl"))
                if not scheduler.is_file() or scheduler.stat().st_size == 0 or not random_states:
                    return False
                full_state = checkpoint_path / "optimizer.bin"
                if full_state.is_file() and full_state.stat().st_size > 0:
                    return True
                sharded_optimizer = checkpoint_path / "optimizer_0" / ".metadata"
                sharded_models = (
                    checkpoint_path / "pytorch_model_fsdp_0" / ".metadata",
                    checkpoint_path / "pytorch_model_0" / ".metadata",
                )
                return (
                    any(model.is_file() and model.stat().st_size > 0 for model in sharded_models)
                    and sharded_optimizer.is_file()
                    and sharded_optimizer.stat().st_size > 0
                )

            has_recoverable_checkpoint = any(
                has_recoverable_state(path) for path in output_path.parent.glob("checkpoint-*") if path.is_dir()
            )
            code_only_change = set(mismatches) == {"code_sha256"}
            if has_recoverable_checkpoint and not (args.allow_resume_code_change and code_only_change):
                raise ValueError(
                    "Refusing to resume with changed runtime provenance fields: " + ", ".join(mismatches)
                )
            if has_recoverable_checkpoint:
                history = list(existing.get("code_change_history", []))
                history.append(
                    {
                        "resumed_utc": datetime.now(timezone.utc).isoformat(),
                        "previous_code_sha256": existing.get("code_sha256"),
                        "new_code_sha256": payload["code_sha256"],
                        "reason": "allow_resume_code_change",
                    }
                )
                payload["code_change_history"] = history
                logging.getLogger(__name__).warning(
                    "Resuming with an explicitly allowed code-only provenance change; "
                    "the prior and new code hashes were recorded in runtime_provenance.json."
                )
            else:
                logging.getLogger(__name__).warning(
                    "Refreshing stale runtime provenance after a pre-checkpoint failure; changed fields: %s",
                    ", ".join(mismatches),
                )
        else:
            return
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)


def truncate_training_metrics(output_dir: str, maximum_step: int) -> int:
    """Drop metric rows produced after the checkpoint being resumed.

    A hard failure can occur between rolling checkpoints.  Those later rows no
    longer describe the restored model state and must not coexist with replayed
    steps in the dashboard or downstream analysis.
    """
    path = Path(output_dir) / "training_metrics/metrics.jsonl"
    if not path.is_file():
        return 0
    kept = []
    removed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
                step = int(payload["global_step"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                removed += 1
                continue
            if step <= int(maximum_step):
                kept.append(json.dumps(payload, sort_keys=True) + "\n")
            else:
                removed += 1
    if removed:
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temporary.write_text("".join(kept), encoding="utf-8")
        os.replace(temporary, path)
    return removed


def get_target_interpolation(task: str | None, preprocess_target: Any = None):
    """Choose how target images should be resized before training.

    Segmentation-like targets default to nearest-neighbor interpolation so
    discrete labels are not blurred. Depth, surface-normal, and natural-image
    targets default to bilinear interpolation.
    """
    if preprocess_target is not None:
        policy = str(preprocess_target).strip().lower().replace("-", "_")
        if policy in {"nearest", "nearest_neighbor"}:
            return transforms.InterpolationMode.NEAREST
        if policy in {
            "bilinear",
            "linear",
            "raw_depth_resize_then_encode",
            *LEGACY_BILINEAR_PREPROCESS_TARGET_POLICIES,
        }:
            return transforms.InterpolationMode.BILINEAR
        if policy in {"bicubic", "cubic"}:
            return transforms.InterpolationMode.BICUBIC
        if policy in {"lanczos"}:
            return transforms.InterpolationMode.LANCZOS
        raise ValueError(f"Unsupported preprocess_target policy: {preprocess_target}")
    if normalize_task_name(task) in SEGMENTATION_TASKS:
        return transforms.InterpolationMode.NEAREST
    if normalize_task_name(task) in DEPTH_TASKS | SURFACE_NORMAL_TASKS:
        return transforms.InterpolationMode.BILINEAR
    return transforms.InterpolationMode.BILINEAR


def validate_target_interpolation_policies(dataset_pool: WeightedDatasetPool) -> None:
    """Reject unsupported target resize policies before caching or training."""
    invalid = []
    for index in dataset_pool.indices:
        for row_number, row in enumerate(index.rows, start=1):
            try:
                get_target_interpolation(row.task, row.preprocess_target)
            except ValueError:
                invalid.append(
                    f"dataset={index.spec.name!r}, row={row_number}, row_id={row.row_id!r}, "
                    f"policy={row.preprocess_target!r}"
                )
                if len(invalid) == 5:
                    break
        if len(invalid) == 5:
            break
    if invalid:
        details = "; ".join(invalid)
        raise ValueError(f"Unsupported preprocess_target policy in prepared dataset metadata: {details}")


def load_dataset_image(value, dataset_root: Path | None = None, column_name: str = "image"):
    """Load one image from a PIL object, bytes blob, or filesystem path."""
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, IndexedTarReference):
        return open_reference_image(value)
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"]))
        if value.get("path") is not None:
            value = value["path"]
        else:
            raise TypeError(f"Unsupported image dict for column {column_name}: keys={list(value)}")
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        candidates = [path]
        if not path.is_absolute() and dataset_root is not None:
            candidates.insert(0, dataset_root / path)
        for candidate in candidates:
            if candidate.exists():
                return Image.open(candidate)
        tried = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Could not resolve image path from column {column_name}: {value!r}. Tried: {tried}")
    raise TypeError(f"Unsupported image value for column {column_name}: {type(value).__name__}")


def transform_depth_target(image: Image.Image, encoding: str) -> Image.Image:
    if encoding == DEPTH_ENCODING_VISION_BANANA_RGB:
        return image
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if encoding == DEPTH_ENCODING_BARRON_SCALAR_GRAYSCALE:
        scalar = project_rgb_to_vision_banana_path(rgb).t
        grayscale = np.rint(scalar * 255.0).astype(np.uint8)
    elif encoding in {
        DEPTH_ENCODING_RELATIVE_INVERSE_GRAYSCALE,
        DEPTH_ENCODING_RELATIVE_LINEAR_GRAYSCALE,
    }:
        depth = decode_vision_banana_rgb_to_depth(rgb)
        grayscale = encode_depth_to_relative_grayscale(
            depth,
            depth > 0,
            space=(
                "inverse"
                if encoding == DEPTH_ENCODING_RELATIVE_INVERSE_GRAYSCALE
                else "linear"
            ),
        )
    else:
        raise ValueError(f"Unsupported depth target encoding: {encoding}")
    return Image.fromarray(grayscale, mode="L").convert("RGB")


def resize_depth_metric_then_encode(
    image: Image.Image,
    size: tuple[int, int],
    valid_mask: Image.Image | None = None,
) -> Image.Image:
    """Resize metric depth without blending invalid RGB padding into the field."""
    height, width = size
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    depth = decode_vision_banana_rgb_to_depth(rgb)
    source_valid = np.isfinite(depth) & (depth > 0)
    if valid_mask is not None:
        source_valid &= np.asarray(valid_mask.convert("L"), dtype=np.uint8) > 0
    resized_depth, resized_valid = resize_scalar_field(
        depth,
        source_valid,
        (width, height),
    )
    encoded = encode_depth_to_vision_banana_rgb(
        resized_depth,
        resized_valid,
    )
    return Image.fromarray(encoded, mode="RGB")


def resize_normal_vector_then_encode(
    image: Image.Image,
    size: tuple[int, int],
    encoding: str = VISION_BANANA_NORMAL_ENCODING,
    valid_mask: Image.Image | None = None,
) -> Image.Image:
    """Resize decoded XYZ components, renormalize valid vectors, and re-encode."""
    height, width = size
    vectors, source_valid = decode_surface_normals_rgb(
        np.asarray(image.convert("RGB")),
        encoding=encoding,
        min_norm=0.05,
    )
    if valid_mask is not None:
        source_valid &= np.asarray(valid_mask.convert("L"), dtype=np.uint8) > 0
    resized, valid = resize_normal_field(
        vectors,
        source_valid,
        (width, height),
    )
    encoded = encode_surface_normals_rgb(
        resized,
        valid,
        encoding=encoding,
    )
    return Image.fromarray(encoded, mode="RGB")


def convert_surface_normal_image_encoding(
    image: Image.Image,
    source_encoding: str,
    target_encoding: str,
    *,
    permutation: tuple[int, int, int] = (0, 1, 2),
    signs: tuple[int, int, int] = (1, 1, 1),
) -> Image.Image:
    """Transcode a frozen normal target through the physical canonical frame."""
    if (
        source_encoding == target_encoding
        and permutation == (0, 1, 2)
        and signs == (1, 1, 1)
    ):
        return image.convert("RGB")
    normals, valid = decode_surface_normals_rgb(
        np.asarray(image.convert("RGB")),
        encoding=source_encoding,
    )
    normals = normals[..., list(permutation)] * np.asarray(signs, dtype=np.float32)
    return Image.fromarray(
        encode_surface_normals_rgb(normals, valid, encoding=target_encoding),
        mode="RGB",
    )


def final_normal_target_validity(target: torch.Tensor) -> torch.Tensor:
    """Validity of the final RGB normal target in the VAE's [-1,1] range.

    Stored-axis sign/permutation changes preserve vector length. Source masks
    alone do not cover invalid vectors introduced by conversion/interpolation.
    This uses the same neutral-fill threshold as the RGB normal decoder.
    """
    return (
        torch.isfinite(target).all(dim=0, keepdim=True)
        & (torch.linalg.vector_norm(target, dim=0, keepdim=True) >= 0.05)
    )


def horizontal_flip_canonical_normal_rgb(image: Image.Image) -> Image.Image:
    """Flip an encoded camera-space normal map and negate canonical x.

    A spatial horizontal flip alone changes the image geometry but leaves the
    encoded vector in the old camera basis.  Canonical ``+x=image right``
    therefore requires ``x'=-x`` after the spatial flip.
    """
    rgb = np.asarray(TF.hflip(image.convert("RGB")), dtype=np.uint8).copy()
    rgb[..., 0] = 255 - rgb[..., 0]
    return Image.fromarray(rgb, mode="RGB")


def transform_depth_prompt(
    prompt: str, encoding: str, prompt_contract: str = "canonical_v2"
) -> str:
    # ``campaign_*`` contracts replace row text during dataset construction;
    # do not collapse them back to the default representation prompt here.
    if prompt_contract.startswith("campaign_"):
        return prompt
    if encoding == DEPTH_ENCODING_VISION_BANANA_RGB:
        return prompt
    if encoding == DEPTH_ENCODING_BARRON_SCALAR_GRAYSCALE:
        return BARRON_SCALAR_DEPTH_PROMPT
    if encoding == DEPTH_ENCODING_RELATIVE_INVERSE_GRAYSCALE:
        return CONCISE_INVERSE_DEPTH_PROMPT
    if encoding == DEPTH_ENCODING_RELATIVE_LINEAR_GRAYSCALE:
        return CONCISE_LINEAR_DEPTH_PROMPT
    raise ValueError(f"Unsupported depth target encoding: {encoding}")


def effective_instance_prompt(
    row: PreparedRow,
    fallback_prompt: str,
    depth_target_encoding: str,
    depth_prompt_contract: str,
) -> str:
    """The exact text emitted by the dataset, including depth representation."""
    prompt = row.text if row.text else fallback_prompt
    if normalize_task_name(row.task) in DEPTH_TASKS:
        return transform_depth_prompt(prompt, depth_target_encoding, depth_prompt_contract)
    return prompt


def oriented_image_size(image: Image.Image) -> tuple[int, int]:
    """Return the display size of an image after considering EXIF rotation tags."""
    width, height = image.size
    try:
        orientation = image.getexif().get(EXIF_ORIENTATION_TAG)
    except Exception:
        orientation = None
    if orientation in TRANSPOSED_EXIF_ORIENTATIONS:
        return height, width
    return width, height


def load_dataset_image_size(value, dataset_root: Path | None = None, column_name: str = "image") -> tuple[int, int]:
    """Read image dimensions without keeping the image object around longer than needed."""
    if value is None:
        raise ValueError(f"Image value for column {column_name} is missing.")
    if isinstance(value, Image.Image):
        return oriented_image_size(value)
    if isinstance(value, IndexedTarReference):
        with open_reference_image(value) as image:
            return oriented_image_size(image)
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            with Image.open(io.BytesIO(value["bytes"])) as image:
                return oriented_image_size(image)
        if value.get("path") is not None:
            value = value["path"]
        else:
            raise TypeError(f"Unsupported image dict for column {column_name}: keys={list(value)}")
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        candidates = [path]
        if not path.is_absolute() and dataset_root is not None:
            candidates.insert(0, dataset_root / path)
        for candidate in candidates:
            if candidate.exists():
                with Image.open(candidate) as image:
                    return oriented_image_size(image)
        tried = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Could not resolve image path from column {column_name}: {value!r}. Tried: {tried}")
    raise TypeError(f"Unsupported image value for column {column_name}: {type(value).__name__}")


class DreamBoothDataset(Dataset):
    """
    Training dataset for target images, optional conditioning images, and prompts.

    Rows stay lightweight metadata until ``__getitem__`` loads the images.
    Aspect-ratio buckets are computed up front, and a conditioning image gets
    the same crop/flip as its target.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        size=1024,
        repeats=1,
        center_crop=False,
        buckets=None,
        training_args=None,
        dataset_pool: WeightedDatasetPool | None = None,
    ):
        """Build the row list from the chosen dataset source and pre-index buckets."""
        self.args = training_args
        if self.args is None:
            self.args = globals().get("args")
        if self.args is None:
            raise ValueError("DreamBoothDataset requires parsed training args.")

        self.size = size
        self.center_crop = center_crop
        self.instance_prompt = instance_prompt
        self.buckets = buckets or [(size, size)]
        self.dataset_pool = dataset_pool
        # Inspection tools construct this dataset with a Namespace that lacks
        # the geometry CLI flags; parse_args always supplies them for training.
        geometry_defaults = {
            "depth_target_encoding": DEPTH_ENCODING_VISION_BANANA_RGB,
            "depth_target_resize_policy": "encoded_rgb",
            "normal_target_resize_policy": "encoded_rgb",
            "normal_target_encoding": LEGACY_DIRECT_NORMAL_ENCODING,
            "depth_prompt_contract": "row",
            "normal_prompt_contract": "row",
            "allow_unverified_normal_sources": False,
            "allow_legacy_hypersim_range_depth": False,
        }
        for name, default in geometry_defaults.items():
            if not hasattr(self.args, name):
                setattr(self.args, name, default)
        self.resample_dataset_each_epoch = dataset_pool is not None and self.args.resample_dataset_each_epoch
        self.current_epoch = 0
        self.custom_instance_prompts = False

        if self.dataset_pool is not None:
            if self.resample_dataset_each_epoch:
                self.rows = self.dataset_pool.rows
            else:
                # Fixed trials materialize only the deterministic epoch-0 draw,
                # bounding bucket indexing and packed prompt caching by
                # --samples_per_epoch rather than the complete source pool.
                self.rows = self.dataset_pool.sample_epoch(0)
                self._log_sample_counts(0)
            self.custom_instance_prompts = True
        elif self.args.dataset_name is not None:
            self.rows = self._load_rows_from_dataset_name(repeats)
        else:
            self.rows = self._load_rows_from_instance_dir(instance_data_root, repeats)

        self._apply_geometry_contract()
        self._validate_loss_mask_policy()

        self._rebuild_bucket_index()

    def _apply_geometry_contract(self) -> None:
        """Normalize all surface-normal targets/prompts before sampling or caching."""
        rows = list(self._all_source_rows())
        unsafe_normals = []
        unsafe_hypersim_depth = []
        for row in rows:
            task = normalize_task_name(row.task)
            source = str(row.source_dataset or "").strip().lower()
            if (
                task in DEPTH_TASKS
                and source == "hypersim"
                and row.canonical_depth_quantity
                != "projective_camera_z_meters"
                and not self.args.allow_legacy_hypersim_range_depth
            ):
                unsafe_hypersim_depth.append(row)
            if (
                task in DEPTH_TASKS
                and self.args.depth_prompt_contract == "canonical_v2"
            ):
                row.text = CANONICAL_DEPTH_PROMPT
            depth_campaign_prompts = {
                "campaign_short_inverse": CONCISE_INVERSE_DEPTH_PROMPT,
                "campaign_detailed_inverse": DETAILED_INVERSE_DEPTH_PROMPT,
                "campaign_paper_inverse": PAPER_RECONSTRUCTED_INVERSE_DEPTH_PROMPT,
                "campaign_paper_barron": PAPER_RECONSTRUCTED_BARRON_DEPTH_PROMPT,
            }
            if task in DEPTH_TASKS and self.args.depth_prompt_contract in depth_campaign_prompts:
                row.text = depth_campaign_prompts[self.args.depth_prompt_contract]
            if task not in SURFACE_NORMAL_TASKS:
                continue
            if (
                row.normal_contract_status == "unverified"
                and not self.args.allow_unverified_normal_sources
            ):
                unsafe_normals.append(row)
            if row.normal_encoding is None:
                row.normal_encoding = LEGACY_DIRECT_NORMAL_ENCODING
            validate_normal_encoding(row.normal_encoding)
            if self.args.normal_prompt_contract == "vision_banana_v3":
                row.text = VISION_BANANA_NORMAL_PROMPT
            normal_campaign_prompts = {
                "campaign_short": CONCISE_NORMAL_PROMPT,
                "campaign_detailed": DETAILED_NORMAL_PROMPT,
                "campaign_paper": PAPER_RECONSTRUCTED_NORMAL_PROMPT,
            }
            if self.args.normal_prompt_contract in normal_campaign_prompts:
                row.text = normal_campaign_prompts[self.args.normal_prompt_contract]
        if unsafe_normals:
            sources = sorted(
                {
                    str(row.source_dataset or row.dataset_name)
                    for row in unsafe_normals
                }
            )
            raise ValueError(
                "Refusing unverified surface-normal training source(s): "
                + ", ".join(sources)
                + ". Rebuild with a versioned canonical transform or use "
                "--allow_unverified_normal_sources only for legacy reproduction."
            )
        if unsafe_hypersim_depth:
            raise ValueError(
                "Refusing legacy Hypersim Euclidean-range depth rows. Rebuild "
                "Hypersim with projective camera-z metadata or use "
                "--allow_legacy_hypersim_range_depth only for legacy reproduction."
            )

    def _all_source_rows(self):
        """Return every possible source row, including unsampled weighted-pool rows."""
        if getattr(self, "dataset_pool", None) is not None:
            return iter(self.dataset_pool.rows)
        return iter(self.rows)

    def _validate_loss_mask_policy(self):
        """Fail early when a run explicitly requires masks but some rows lack them."""
        require_every_row = getattr(self.args, "loss_mask_policy", "auto") == "enabled"
        require_geometry = not getattr(self.args, "allow_missing_geometry_masks", False)
        missing_preview = []
        missing_count = 0
        for row in self._all_source_rows():
            geometry = normalize_task_name(row.task) in DEPTH_TASKS | SURFACE_NORMAL_TASKS
            if row.loss_mask is None and (require_every_row or (require_geometry and geometry)):
                missing_count += 1
                if len(missing_preview) < 5:
                    missing_preview.append(self._row_label(row))
        if missing_count:
            preview = ", ".join(missing_preview)
            extra = "" if missing_count <= 5 else f", and {missing_count - 5} more"
            raise ValueError(
                "Geometry requires source validity masks in BOTH masked and unmasked runs; "
                "loss_mask_policy=enabled additionally requires masks for other tasks. "
                f"Missing masks for: {preview}{extra}."
            )

    def _make_row(
        self,
        dataset_name: str,
        dataset_root: Path | None,
        row_id: str | None,
        image: Any,
        conditioning_image: Any,
        text: str | None,
        loss_mask: Any | None = None,
        task: str | None = None,
        source_dataset: str | None = None,
        preprocess_target: str | None = None,
    ) -> PreparedRow:
        """Create the normalized internal representation for one training row."""
        return PreparedRow(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            row_id=row_id,
            conditioning_image=conditioning_image,
            image=image,
            text=text if text else self.instance_prompt,
            loss_mask=loss_mask,
            task=normalize_task_name(task),
            source_dataset=source_dataset,
            preprocess_target=preprocess_target,
        )

    def _load_rows_from_dataset_name(self, repeats: int) -> list[PreparedRow]:
        """Load rows from a Hugging Face dataset or a local dataset folder."""
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "You are trying to load your data using the datasets library. If you wish to train using custom "
                "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                "local folder containing images only, specify --instance_data_dir instead."
            ) from exc

        dataset_path = Path(self.args.dataset_name).expanduser()
        dataset_root = dataset_path.resolve() if dataset_path.exists() else None
        metadata_path = dataset_path / "metadata.jsonl"
        if dataset_path.exists() and metadata_path.exists():
            logger.info(f"Loading local metadata rows from {metadata_path}")
            dataset = load_dataset("json", data_files={"train": str(metadata_path)}, cache_dir=self.args.cache_dir)
        else:
            dataset = load_dataset(
                self.args.dataset_name,
                self.args.dataset_config_name,
                cache_dir=self.args.cache_dir,
            )

        column_names = dataset["train"].column_names
        if self.args.cond_image_column is not None and self.args.cond_image_column not in column_names:
            raise ValueError(
                f"`--cond_image_column` value '{self.args.cond_image_column}' not found in dataset columns. "
                f"Dataset columns are: {', '.join(column_names)}"
            )
        if self.args.image_column is None:
            image_column = column_names[0]
            logger.info(f"image column defaulting to {image_column}")
        else:
            image_column = self.args.image_column
            if image_column not in column_names:
                raise ValueError(
                    f"`--image_column` value '{self.args.image_column}' not found in dataset columns. "
                    f"Dataset columns are: {', '.join(column_names)}"
                )

        if self.args.caption_column is not None and self.args.caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{self.args.caption_column}' not found in dataset columns. "
                f"Dataset columns are: {', '.join(column_names)}"
            )

        self.custom_instance_prompts = self.args.caption_column is not None
        if self.args.caption_column is None:
            logger.info(
                "No caption column provided, defaulting to instance_prompt for all images. If your dataset "
                "contains captions/prompts for the images, make sure to specify the column as --caption_column"
            )

        rows = []
        train_split = dataset["train"]
        for row_index in range(len(train_split)):
            # We normalize each row into the same lightweight shape regardless of
            # whether the data came from a Hub dataset or a local metadata file.
            raw = train_split[row_index]
            text = raw.get(self.args.caption_column) if self.args.caption_column is not None else self.instance_prompt
            cond_image = raw.get(self.args.cond_image_column) if self.args.cond_image_column is not None else None
            row = self._make_row(
                dataset_name=Path(self.args.dataset_name).name if dataset_path.exists() else str(self.args.dataset_name),
                dataset_root=dataset_root,
                row_id=str(raw.get("id")) if isinstance(raw, dict) and raw.get("id") is not None else str(row_index),
                image=raw[image_column],
                conditioning_image=cond_image,
                text=text,
                loss_mask=raw.get("loss_mask") if isinstance(raw, dict) else None,
                task=raw.get("task") if isinstance(raw, dict) else None,
                source_dataset=raw.get("source_dataset") if isinstance(raw, dict) else None,
                preprocess_target=raw.get("preprocess_target") if isinstance(raw, dict) else None,
            )
            rows.extend(itertools.repeat(row, repeats))
        return list(rows)

    def _load_rows_from_instance_dir(self, instance_data_root: str, repeats: int) -> list[PreparedRow]:
        """Treat every file in a local folder as a target image using one shared prompt."""
        instance_data_root = Path(instance_data_root).expanduser().resolve()
        if not instance_data_root.exists():
            raise ValueError(f"Instance images root does not exist: {instance_data_root}")

        image_paths = sorted(path for path in instance_data_root.iterdir() if path.is_file())
        rows = []
        for row_index, image_path in enumerate(image_paths):
            row = self._make_row(
                dataset_name=instance_data_root.name,
                dataset_root=None,
                row_id=str(row_index),
                image=image_path,
                conditioning_image=None,
                text=self.instance_prompt,
            )
            rows.extend(itertools.repeat(row, repeats))
        return list(rows)

    def _log_sample_counts(self, epoch: int):
        """Log how many rows were drawn from each dataset in the current pool sample."""
        if self.dataset_pool is None:
            return
        counts = self.dataset_pool.last_sample_counts
        log_info(f"Sampled dataset-pool epoch {epoch}:")
        for name in sorted(counts):
            log_info(f"  {name}: {counts[name]:,} rows")

    def set_epoch(self, epoch: int):
        """Record the epoch; the batch sampler owns the stable-index epoch plan."""
        self.current_epoch = epoch

    def _row_label(self, row: PreparedRow) -> str:
        """Return a short identifier used in data-loading error messages."""
        return f"dataset='{row.dataset_name}', row_id='{row.row_id}'"

    def _load_row_image(self, row: PreparedRow, value: Any, column_name: str):
        """Load one row image, apply EXIF orientation, and normalize it to RGB."""
        try:
            image = load_dataset_image(value, row.dataset_root, column_name)
            image = exif_transpose(image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image
        except Exception as exc:
            raise type(exc)(f"Could not load {column_name} for {self._row_label(row)}: {exc}") from exc

    def _load_row_loss_mask(self, row: PreparedRow):
        """Load an optional 8-bit loss mask where nonzero pixels are valid."""
        try:
            image = load_dataset_image(row.loss_mask, row.dataset_root, "loss_mask")
            image = exif_transpose(image)
            if image.mode != "L":
                image = image.convert("L")
            return image
        except Exception as exc:
            raise type(exc)(f"Could not load loss_mask for {self._row_label(row)}: {exc}") from exc

    def _load_row_image_size(self, row: PreparedRow, value: Any, column_name: str) -> tuple[int, int]:
        """Read just the display size for one row image with row-aware error text."""
        try:
            return load_dataset_image_size(value, row.dataset_root, column_name)
        except Exception as exc:
            raise type(exc)(f"Could not read size for {column_name} in {self._row_label(row)}: {exc}") from exc

    def _build_bucket_entry(self, row: PreparedRow):
        """Assign one row to the closest aspect-ratio bucket."""
        width, height = self._load_row_image_size(row, row.image, self.args.image_column)
        bucket_idx = find_nearest_bucket(height, width, self.buckets)
        # First tuple slot kept for callers expecting ``(image, bucket_idx)``.
        return (None, bucket_idx)

    @staticmethod
    def _bucket_cache_file_identity(path: Path | None) -> dict[str, Any] | None:
        if path is None:
            return None
        stat = path.stat()
        return {"path": str(path.resolve()), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}

    @staticmethod
    def _legacy_bucket_cache_metadata_compatible(
        cached_metadata: dict[str, Any], current_metadata: dict[str, Any]
    ) -> bool:
        """Accept v1 entries that only differ by the retired per-row digest."""
        legacy_metadata = dict(cached_metadata)
        legacy_metadata.pop("row_identity_sha256", None)
        return legacy_metadata == current_metadata

    def _bucket_index_cache_metadata(self) -> dict[str, Any] | None:
        """Fingerprint immutable source manifests that determine prepared row order."""
        if self.dataset_pool is None:
            return None

        sources = []
        for index in self.dataset_pool.indices:
            spec = index.spec
            source_files = []
            if spec.type == "indexed_prepared_tar":
                source_files = [self._bucket_cache_file_identity(spec.path), self._bucket_cache_file_identity(spec.index_path)]
            elif spec.path is not None:
                source_files = [self._bucket_cache_file_identity(spec.path / "metadata.jsonl")]
            sources.append(
                {
                    "name": spec.name,
                    "type": spec.type,
                    "source_files": source_files,
                    "filters": spec.filters,
                    "sampling": spec.sampling,
                    "rows": len(index.rows),
                }
            )
        return {
            "kind": "dreambooth_prepared_bucket_index",
            "buckets": [[int(height), int(width)] for height, width in self.buckets],
            "image_column": self.args.image_column,
            "max_refseg_expressions_per_image": self.dataset_pool.max_refseg_expressions_per_image,
            "sources": sources,
        }

    def _build_bucket_indices(self) -> np.ndarray:
        """Read target-image headers once and return a compact bucket-index vector."""
        num_workers = max(1, int(getattr(self.args, "dataset_index_num_workers", 1) or 1))
        desc = f"Indexing dataset rows ({num_workers} workers)" if num_workers > 1 else "Indexing dataset rows"
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                entries = list(
                    tqdm(
                        executor.map(self._build_bucket_entry, self.rows),
                        total=len(self.rows),
                        desc=desc,
                        leave=False,
                    )
                )
        else:
            entries = [
                self._build_bucket_entry(row)
                for row in tqdm(self.rows, total=len(self.rows), desc=desc, leave=False)
            ]
        return np.fromiter((int(bucket_idx) for _, bucket_idx in entries), dtype=np.uint16, count=len(entries))

    def _set_bucket_indices(self, bucket_indices: np.ndarray) -> None:
        if len(bucket_indices) != len(self.rows):
            raise ValueError(f"Bucket-index length {len(bucket_indices)} does not match {len(self.rows)} dataset rows.")
        # Sampler and cache-latent paths expect the tuple layout; only the on-disk form is compact.
        self.pixel_values = [(None, int(bucket_idx)) for bucket_idx in bucket_indices]
        self.num_instance_images = len(self.rows)
        self._length = self.num_instance_images

    def _rebuild_bucket_index(self):
        """Compute or reuse one immutable bucket assignment for every eligible row."""
        cache_dir = str(getattr(self.args, "bucket_index_cache_dir", "") or "").strip()
        metadata = self._bucket_index_cache_metadata() if cache_dir else None
        if cache_dir and metadata is not None:
            try:
                Path(cache_dir).expanduser().mkdir(parents=True, exist_ok=True)
            except OSError as error:
                log_info(f"Bucket-index cache unavailable at {cache_dir}: {error}; indexing locally.")
            else:
                bucket_indices, reused = load_or_build_bucket_indices(
                    cache_dir=cache_dir,
                    metadata=metadata,
                    row_count=len(self.rows),
                    bucket_count=len(self.buckets),
                    build_indices=self._build_bucket_indices,
                    log_fn=log_info,
                    metadata_compatible=self._legacy_bucket_cache_metadata_compatible,
                )
                self._set_bucket_indices(bucket_indices)
                action = "Reused" if reused else "Built"
                log_info(f"{action} persistent bucket index for {self.num_instance_images:,} rows.")
                return
        elif cache_dir:
            log_info("Bucket-index cache is disabled for this non-prepared or in-memory dataset source.")

        log_info(f"Building bucket index for {len(self.rows):,} dataset rows...")
        self._set_bucket_indices(self._build_bucket_indices())
        log_info(f"Finished bucket index build for {self.num_instance_images:,} rows.")

    def __len__(self):
        """Return how many row entries the dataset exposes for this epoch."""
        return self._length

    def _timeout_placeholder(
        self,
        row: PreparedRow,
        dataset_index: int,
        bucket_idx: int,
        size: tuple[int, int],
    ) -> dict[str, Any]:
        """Return a shape-compatible batch member whose loss is explicitly ignored."""
        target_height, target_width = size
        prompt = effective_instance_prompt(
            row, self.instance_prompt, self.args.depth_target_encoding,
            self.args.depth_prompt_contract,
        )
        example: dict[str, Any] = {
            "dataset_index": dataset_index,
            "instance_images": torch.zeros((3, target_height, target_width), dtype=torch.float32),
            "bucket_idx": bucket_idx,
            "instance_prompt": prompt,
            "task": row.task,
            "dataset_name": row.dataset_name,
            "source_dataset": row.source_dataset,
            "row_id": row.row_id,
            "sample_valid": False,
        }
        if row.conditioning_image is not None:
            example["cond_images"] = torch.zeros((3, target_height, target_width), dtype=torch.float32)
        if getattr(self.args, "loss_mask_policy", "auto") != "disabled":
            example["loss_mask"] = torch.zeros((1, target_height, target_width), dtype=torch.float32)
        return example

    def __getitem__(self, index):
        """Load one row, bounded so a pathological decode cannot idle all GPUs."""
        global _ACTIVE_DATALOADER_ROW_LABEL
        dataset_index = index % self.num_instance_images
        row = self.rows[dataset_index]
        _, bucket_idx = self.pixel_values[dataset_index]
        target_height, target_width = self.buckets[bucket_idx]
        row_label = self._row_label(row)
        timeout_seconds = float(getattr(self.args, "row_load_timeout_seconds", 0.0) or 0.0)
        _ACTIVE_DATALOADER_ROW_LABEL = row_label
        try:
            with row_load_deadline(timeout_seconds, row_label):
                return self._load_example(
                    row, dataset_index, bucket_idx, (target_height, target_width)
                )
        except DataRowLoadTimeout as error:
            print(
                "DATALOADER ROW TIMEOUT: "
                f"{error}; substituting a zero-loss placeholder for this occurrence.",
                file=sys.stderr,
                flush=True,
            )
            return self._timeout_placeholder(
                row, dataset_index, bucket_idx, (target_height, target_width)
            )
        finally:
            _ACTIVE_DATALOADER_ROW_LABEL = None

    def _load_example(
        self,
        row: PreparedRow,
        dataset_index: int,
        bucket_idx: int,
        size: tuple[int, int],
    ) -> dict[str, Any]:
        """Load and transform a normal prepared row."""
        example: dict[str, Any] = {}

        instance_image = self._load_row_image(row, row.image, self.args.image_column)
        normalized_task = normalize_task_name(row.task)
        use_loss_mask = getattr(self.args, "loss_mask_policy", "auto") != "disabled"
        geometry_resize_uses_mask = (
            normalized_task in DEPTH_TASKS
            and self.args.depth_target_resize_policy == "metric_field_then_encode"
        ) or (
            normalized_task in SURFACE_NORMAL_TASKS
            and self.args.normal_target_resize_policy == "vector_field_renormalize_then_encode"
        )
        # Field resizing needs the source validity mask even in unmasked runs;
        # otherwise invalid normal fill (mid-gray) is normalized into a spurious direction.
        preprocessing_mask = (
            self._load_row_loss_mask(row)
            if row.loss_mask is not None and (use_loss_mask or geometry_resize_uses_mask)
            else None
        )
        loss_mask = preprocessing_mask if use_loss_mask else None
        if normalized_task in SURFACE_NORMAL_TASKS:
            instance_image = convert_surface_normal_image_encoding(
                instance_image,
                row.normal_encoding or LEGACY_DIRECT_NORMAL_ENCODING,
                self.args.normal_target_encoding,
                permutation=row.stored_normal_permutation,
                signs=row.stored_normal_signs,
            )
        if (
            normalized_task in DEPTH_TASKS
            and self.args.depth_target_resize_policy == "metric_field_then_encode"
        ):
            instance_image = resize_depth_metric_then_encode(
                instance_image, size, preprocessing_mask
            )
        elif (
            normalized_task in SURFACE_NORMAL_TASKS
            and self.args.normal_target_resize_policy == "vector_field_renormalize_then_encode"
        ):
            instance_image = resize_normal_vector_then_encode(
                instance_image,
                size,
                self.args.normal_target_encoding,
                preprocessing_mask,
            )
        if normalized_task in DEPTH_TASKS:
            instance_image = transform_depth_target(
                instance_image, self.args.depth_target_encoding
            )
        cond_image = None
        if row.conditioning_image is not None:
            cond_image = self._load_row_image(row, row.conditioning_image, self.args.cond_image_column)

        preprocess_target = row.preprocess_target
        target_interpolation = get_target_interpolation(row.task, preprocess_target)
        # The target and conditioning image must stay pixel-aligned, so both go
        # through the same crop and optional horizontal flip.
        instance_image, cond_image, loss_mask = self.paired_transform(
            instance_image,
            dest_image=cond_image,
            loss_mask=loss_mask,
            size=size,
            center_crop=self.args.center_crop,
            random_flip=self.args.random_flip,
            canonical_normal_target=normalized_task in SURFACE_NORMAL_TASKS,
            target_interpolation=target_interpolation,
            conditioning_interpolation=transforms.InterpolationMode.BILINEAR,
            fill_missing_loss_mask=use_loss_mask,
            augmentation=self._augmentation_rng(row, dataset_index),
        )
        if use_loss_mask and normalized_task in SURFACE_NORMAL_TASKS:
            loss_mask = loss_mask * final_normal_target_validity(instance_image).to(loss_mask.dtype)

        example["dataset_index"] = dataset_index
        example["instance_images"] = instance_image
        example["sample_valid"] = True
        if use_loss_mask:
            example["loss_mask"] = loss_mask
        example["bucket_idx"] = bucket_idx
        if cond_image is not None:
            example["cond_images"] = cond_image

        instance_prompt = effective_instance_prompt(
            row, self.instance_prompt, self.args.depth_target_encoding,
            self.args.depth_prompt_contract,
        )
        example["instance_prompt"] = instance_prompt
        example["task"] = row.task
        example["dataset_name"] = row.dataset_name
        example["source_dataset"] = row.source_dataset
        example["row_id"] = row.row_id

        return example

    def _augmentation_rng(self, row, dataset_index):
        from common.training.resume_rng import augmentation_rng
        pool = getattr(self, "dataset_pool", None)
        seed = pool.seed if pool is not None else getattr(self.args, "seed", 0)
        return augmentation_rng(seed, getattr(self, "current_epoch", 0), row.dataset_name, row.task, row.row_id, dataset_index)

    def paired_transform(
        self,
        image,
        dest_image=None,
        loss_mask=None,
        size=(224, 224),
        center_crop=False,
        random_flip=False,
        canonical_normal_target=False,
        target_interpolation=transforms.InterpolationMode.BILINEAR,
        conditioning_interpolation=transforms.InterpolationMode.BILINEAR,
        fill_missing_loss_mask=True,
        augmentation=None,
    ):
        """Resize, crop, flip, and normalize paired images in a synchronized way."""
        target_resize = transforms.Resize(size, interpolation=target_interpolation)
        image = target_resize(image)
        if dest_image is not None:
            cond_resize = transforms.Resize(size, interpolation=conditioning_interpolation)
            dest_image = cond_resize(dest_image)
        if loss_mask is not None:
            mask_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST)
            loss_mask = mask_resize(loss_mask)

        if center_crop:
            crop = transforms.CenterCrop(size)
            image = crop(image)
            if dest_image is not None:
                dest_image = crop(dest_image)
            if loss_mask is not None:
                loss_mask = crop(loss_mask)
        else:
            # get_params returns (i, j, h, w)
            if augmentation is None:
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=size)
            else:
                width, height = image.size
                h, w = size
                i, j = augmentation.randrange(height-h+1), augmentation.randrange(width-w+1)
            image = TF.crop(image, i, j, h, w)
            if dest_image is not None:
                dest_image = TF.crop(dest_image, i, j, h, w)
            if loss_mask is not None:
                loss_mask = TF.crop(loss_mask, i, j, h, w)

        # 3. Random horizontal flip with the SAME coin flip
        if random_flip:
            do_flip = (augmentation or random).random() < 0.5
            if do_flip:
                image = (
                    horizontal_flip_canonical_normal_rgb(image)
                    if canonical_normal_target
                    else TF.hflip(image)
                )
                if dest_image is not None:
                    dest_image = TF.hflip(dest_image)
                if loss_mask is not None:
                    loss_mask = TF.hflip(loss_mask)

        # Convert PIL images to the normalized tensor range expected by the VAE.
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.5], [0.5])
        image = normalize(to_tensor(image))
        if dest_image is not None:
            dest_image = normalize(to_tensor(dest_image))
        if loss_mask is not None:
            loss_mask = (to_tensor(loss_mask) > 0.5).float()
        elif fill_missing_loss_mask:
            loss_mask = torch.ones((1, size[0], size[1]), dtype=image.dtype)

        return image, dest_image, loss_mask


def collate_fn(examples):
    """Merge dataset examples into one batch dictionary for the training loop."""
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    dataset_indices = torch.tensor([example["dataset_index"] for example in examples], dtype=torch.long)
    sample_valid = torch.tensor([bool(example.get("sample_valid", True)) for example in examples], dtype=torch.bool)

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {
        "pixel_values": pixel_values,
        "prompts": prompts,
        "dataset_indices": dataset_indices,
        "sample_valid": sample_valid,
        "tasks": [example.get("task") for example in examples],
        "dataset_names": [example.get("dataset_name") for example in examples],
        "source_datasets": [example.get("source_dataset") for example in examples],
        "row_ids": [example.get("row_id") for example in examples],
    }
    has_conditioning = ["cond_images" in example for example in examples]
    if any(has_conditioning):
        # The model expects a consistent batch structure. Mixing rows with and
        # without conditioning images would make the latent preparation ambiguous.
        if not all(has_conditioning):
            raise ValueError(
                "A batch cannot mix image-conditioned and text-only rows. Use --train_batch_size=1 "
                "or bucket the two task types separately."
            )
        cond_pixel_values = [example["cond_images"] for example in examples]
        cond_pixel_values = torch.stack(cond_pixel_values)
        cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()
        batch.update({"cond_pixel_values": cond_pixel_values})
    batch["has_conditioning"] = bool(has_conditioning and all(has_conditioning))
    if "loss_mask" in examples[0]:
        loss_masks = [example["loss_mask"] for example in examples]
        loss_masks = torch.stack(loss_masks)
        loss_masks = loss_masks.to(memory_format=torch.contiguous_format).float()
        batch["loss_masks"] = loss_masks
    return batch


def prepare_latent_loss_mask(loss_mask, target_spatial_shape, *, mode="hard", min_valid_fraction=0.5, erosion_radius=0):
    """Convert an image validity mask into a testable hard or soft latent mask."""
    mask = loss_mask.float().clamp(0.0, 1.0)
    if erosion_radius > 0:
        invalid = 1.0 - mask
        kernel = 2 * int(erosion_radius) + 1
        invalid = torch.nn.functional.max_pool2d(invalid, kernel_size=kernel, stride=1, padding=erosion_radius)
        mask = 1.0 - invalid
    if mask.shape[-2:] != tuple(target_spatial_shape):
        mask = torch.nn.functional.interpolate(mask, size=target_spatial_shape, mode="area")
    if mode == "hard":
        mask = (mask >= min_valid_fraction).to(dtype=loss_mask.dtype)
    elif mode != "soft":
        raise ValueError(f"Unsupported loss-mask mode: {mode}")
    return mask.clamp(0.0, 1.0)


def eligible_loss_mask_samples(loss_mask, expanded_loss_mask, min_sample_valid_fraction=0.0):
    """Return per-sample eligibility while always rejecting empty latent masks."""
    batch_size = loss_mask.shape[0]
    valid_counts = expanded_loss_mask.reshape(batch_size, -1).sum(1)
    sample_valid_fraction = loss_mask.reshape(batch_size, -1).mean(1)
    eligible = (valid_counts > 0) & (
        sample_valid_fraction >= min_sample_valid_fraction
    )
    return valid_counts, sample_valid_fraction, eligible


def append_conditioning_tokens(target_tokens, target_ids, conditioning_tokens=None, conditioning_ids=None):
    """Compose the transformer image stream without fabricating text-only conditioning."""
    target_length = target_tokens.shape[1]
    if conditioning_tokens is None:
        if conditioning_ids is not None:
            raise ValueError("conditioning_ids were provided without conditioning_tokens")
        return target_tokens, target_ids, target_length
    if conditioning_ids is None:
        raise ValueError("conditioning_tokens require matching conditioning_ids")
    if conditioning_tokens.shape[0] != target_tokens.shape[0]:
        raise ValueError("target and conditioning batch sizes differ")
    return (
        torch.cat([target_tokens, conditioning_tokens], dim=1),
        torch.cat([target_ids, conditioning_ids], dim=1),
        target_length,
    )


class BucketBatchSampler(BatchSampler):
    """Batch sampler that groups rows by aspect-ratio bucket and optional task."""

    def __init__(
        self,
        dataset: DreamBoothDataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle_batches_each_epoch: bool = True,
        task_homogeneous_batches: bool = False,
        seed: int | None = None,
    ):
        """Build the initial grouping of dataset indices into mini-batches."""
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got drop_last={}".format(drop_last))

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle_batches_each_epoch = shuffle_batches_each_epoch
        self.task_homogeneous_batches = task_homogeneous_batches
        self.seed = seed if seed is not None else random.randrange(2**31)
        self.epoch = 0

        self.refresh()

    def _group_key(self, idx: int, bucket_idx: int):
        """Return the grouping key used to keep compatible rows in the same batch."""
        if not self.task_homogeneous_batches:
            return (bucket_idx,)
        row = self.dataset.rows[idx]
        task_or_dataset = row.task or row.dataset_name
        return (bucket_idx, task_or_dataset, row.conditioning_image is not None, row.dataset_name)

    def refresh(self):
        """Rebuild batches for the current epoch from the latest bucket assignments."""
        rng = random.Random(self.seed + self.epoch)
        grouped_indices = defaultdict(list)
        if self.dataset.dataset_pool is None or not self.dataset.resample_dataset_each_epoch:
            selected_indices = range(len(self.dataset.pixel_values))
        else:
            selected_indices = self.dataset.dataset_pool.sample_epoch_indices(self.epoch)
        for idx in selected_indices:
            _, bucket_idx = self.dataset.pixel_values[idx]
            grouped_indices[self._group_key(idx, bucket_idx)].append(idx)

        self.batches = []
        for key in sorted(grouped_indices):
            indices_in_group = grouped_indices[key]
            # Shuffle inside each compatible group first so batches are not biased
            # toward the original file or dataset ordering.
            rng.shuffle(indices_in_group)
            for i in range(0, len(indices_in_group), self.batch_size):
                batch = indices_in_group[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    if self.dataset.dataset_pool is None:
                        continue
                    # Replacement-sampled pools can safely pad a compatible
                    # bucket/task tail with rows from that same group.
                    batch = batch + rng.choices(indices_in_group, k=self.batch_size - len(batch))
                self.batches.append(batch)

        if self.dataset.dataset_pool is not None:
            # Padding each compatible group can create a few excess batches.
            # Keep the epoch length fixed for scheduler and exact-resume math.
            target_batches = math.ceil(self.dataset.dataset_pool.samples_per_epoch / self.batch_size)
            rng.shuffle(self.batches)
            self.batches = self.batches[:target_batches]
        elif self.shuffle_batches_each_epoch:
            rng.shuffle(self.batches)
        elif self.epoch == 0:
            # Mix buckets once while keeping the order stable for step-indexed caches.
            rng.shuffle(self.batches)
        self.sampler_len = len(self.batches)

    def set_epoch(self, epoch: int):
        """Advance the sampler epoch and let the dataset refresh itself if needed."""
        self.epoch = int(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(self.epoch)
        self.refresh()
        self.dataset._log_sample_counts(self.epoch)

    def __iter__(self):
        """Yield one list of dataset indices per batch."""
        if self.shuffle_batches_each_epoch:
            self.refresh()
        for batch in self.batches:
            yield batch

    def __len__(self):
        """Return the number of batches available in the current epoch."""
        return self.sampler_len


class WebDatasetTextToImageBatches(IterableDataset):
    """Stream validated text-to-image tar samples as bucket-homogeneous batches."""

    def __init__(self, spec, training_args, buckets, batch_size, seed):
        self.spec = spec
        self.args = training_args
        self.buckets = buckets
        self.batch_size = batch_size
        self.seed = int(seed or 0)
        self.epoch = 0
        self.streaming = spec.streaming or {}

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _handler(self, exc):
        if hasattr(self, "_active_counters"):
            self._active_counters["stream_failures"] += 1
            self._active_counters["skipped_samples"] += 1
        if self.streaming.get("error_policy", "warn_and_continue") == "raise":
            return False
        logger.warning(f"Skipping corrupt WebDataset sample from {self.spec.name}: {exc}")
        return True

    def _pipeline(self):
        try:
            import webdataset as wds
        except ImportError as exc:
            raise ImportError(
                "Streaming dataset specs require webdataset==1.0.2. Install the project requirements first."
            ) from exc
        # Only validated local shards and explicit jpg/txt/json members are decoded.
        # WebDataset's global secure mode disables local file URLs, so security is
        # enforced here through the narrow decoder/member contract instead.
        urls = self.spec.shards
        resampled = bool(self.streaming.get("resampled", True))
        if resampled:
            shard_source = wds.ResampledShards(urls, deterministic=True, seed=self.seed + self.epoch)
            stages = [shard_source, wds.split_by_node, wds.split_by_worker]
        else:
            stages = [wds.SimpleShardList(urls), wds.split_by_node, wds.split_by_worker]
            if self.streaming.get("shard_shuffle", True):
                stages.append(wds.shuffle(100, rng=random.Random(self.seed + self.epoch)))
        stages.extend(
            [
                wds.tarfile_to_samples(handler=self._handler),
                wds.shuffle(
                    int(self.streaming.get("sample_shuffle_buffer", 10000)),
                    rng=random.Random(self.seed + 100003 + self.epoch),
                ),
                wds.decode("pil", handler=self._handler),
                wds.to_tuple("__key__", "__url__", "jpg;jpeg;png;webp", "txt", "json", handler=self._handler),
            ]
        )
        return wds.DataPipeline(*stages)

    def _example(self, raw):
        key, _url, image, caption, metadata = raw
        if isinstance(caption, bytes):
            caption = caption.decode("utf-8")
        caption = " ".join(str(caption).strip().split())
        if not caption:
            raise ValueError(f"empty caption for key {key}")
        if isinstance(metadata, (bytes, str)):
            metadata = json.loads(metadata)
        if metadata.get("task") != "text_to_image" or metadata.get("source_dataset") != self.spec.source_dataset:
            raise ValueError(f"inconsistent task/source metadata for key {key}")
        image = exif_transpose(image).convert("RGB")
        bucket_idx = find_nearest_bucket(image.height, image.width, self.buckets)
        target_height, target_width = self.buckets[bucket_idx]
        size = (target_height, target_width)
        image = transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)(image)
        sample_seed = int.from_bytes(
            hashlib.sha256(f"{self.seed}:{self.epoch}:{self.spec.name}:{key}".encode()).digest()[:8],
            "big",
        )
        augmentation_rng = random.Random(sample_seed)
        if self.args.center_crop:
            image = transforms.CenterCrop(size)(image)
        else:
            # Streaming resume reconstructs and skips the deterministic prefix of
            # an epoch. Derive augmentation randomness from sample identity so
            # replaying that prefix does not advance process-global RNG state.
            h, w = size
            i = augmentation_rng.randint(0, max(0, image.height - h))
            j = augmentation_rng.randint(0, max(0, image.width - w))
            image = TF.crop(image, i, j, h, w)
        if self.args.random_flip and augmentation_rng.random() < 0.5:
            image = TF.hflip(image)
        tensor = transforms.Normalize([0.5], [0.5])(transforms.ToTensor()(image))
        example = {
            "dataset_index": -1,
            "instance_images": tensor,
            "bucket_idx": bucket_idx,
            "instance_prompt": caption,
            "task": self.spec.task,
            "dataset_name": self.spec.name,
            "source_dataset": self.spec.source_dataset,
            "row_id": str(key),
        }
        if self.args.loss_mask_policy != "disabled":
            example["loss_mask"] = torch.ones((1, target_height, target_width), dtype=tensor.dtype)
        return example

    def __iter__(self):
        queues = defaultdict(list)
        self._active_counters = Counter()
        last_report = Counter()
        opened_shards = set()
        for raw in self._pipeline():
            self._active_counters["samples_seen"] += 1
            shard_url = str(raw[1])
            if shard_url not in opened_shards:
                opened_shards.add(shard_url)
                self._active_counters["shards_opened"] += 1
            try:
                example = self._example(raw)
            except Exception as exc:
                self._active_counters["metadata_or_decode_failures"] += 1
                self._active_counters["skipped_samples"] += 1
                if self.streaming.get("error_policy", "warn_and_continue") == "raise":
                    raise
                logger.warning(f"Skipping invalid WebDataset row from {self.spec.name}: {exc}")
                continue
            key = (
                example["bucket_idx"],
                example["task"],
                False,
                example["dataset_name"],
            )
            queue = queues[key]
            queue.append(example)
            if len(queue) == self.batch_size:
                batch = collate_fn(queue)
                batch["has_conditioning"] = False
                self._active_counters["samples_returned"] += len(queue)
                batch["stream_counters"] = {
                    key: value - last_report.get(key, 0) for key, value in self._active_counters.items()
                }
                last_report = Counter(self._active_counters)
                yield batch
                queues[key] = []


def allocate_weighted_counts(weights: dict[str, float], total: int) -> dict[str, int]:
    """Round positive source weights to deterministic integer quotas.

    Largest remainders receive the leftover slots, with the source name as a
    stable tie-breaker.  Keeping this calculation outside the scheduler lets
    dataset construction allocate only the rows a source can consume.
    """
    total = int(total)
    if total <= 0:
        raise ValueError("weighted allocation total must be positive")
    if not weights:
        raise ValueError("weighted allocation requires at least one source")
    normalized = {str(name): float(weight) for name, weight in weights.items()}
    if any(not math.isfinite(weight) or weight <= 0 for weight in normalized.values()):
        raise ValueError("weighted allocation source weights must be finite and positive")
    total_weight = sum(normalized.values())
    ideals = {name: total * weight / total_weight for name, weight in normalized.items()}
    counts = {name: math.floor(value) for name, value in ideals.items()}
    remainder = total - sum(counts.values())
    order = sorted(normalized, key=lambda name: (ideals[name] - counts[name], name), reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


class WeightedSourceBatchScheduler:
    """Choose complete source batches using deterministic exact epoch quotas."""

    def __init__(self, loaders, weights, batches_per_epoch, seed):
        self.loaders = dict(loaders)
        self.weights = dict(weights)
        self.batches_per_epoch = int(batches_per_epoch)
        self.seed = int(seed or 0)
        self.epoch = 0
        self.position = 0
        self.source_batch_counts = {name: 0 for name in self.loaders}
        self.stream_counters = Counter()
        if self.batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be positive")
        if set(self.loaders) != set(self.weights) or not self.loaders:
            raise ValueError("loaders and weights must contain the same non-empty source names")

    def _target_counts(self):
        return allocate_weighted_counts(self.weights, self.batches_per_epoch)

    def _schedule(self):
        schedule = []
        for name, count in self._target_counts().items():
            schedule.extend([name] * count)
        random.Random(self.seed + self.epoch).shuffle(schedule)
        return schedule

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.position = 0
        self.source_batch_counts = {name: 0 for name in self.loaders}
        self.stream_counters = Counter()
        for loader in self.loaders.values():
            # Accelerate wraps a custom batch sampler in BatchSamplerShard,
            # which does not expose set_epoch.  Walk through wrapper layers so
            # BucketBatchSampler can resample rows *and* rebuild bucket groups.
            batch_sampler = getattr(loader, "batch_sampler", None)
            seen = set()
            while batch_sampler is not None and id(batch_sampler) not in seen:
                seen.add(id(batch_sampler))
                if hasattr(batch_sampler, "set_epoch"):
                    batch_sampler.set_epoch(epoch)
                    break
                batch_sampler = getattr(batch_sampler, "batch_sampler", None)
            else:
                dataset = getattr(loader, "dataset", None)
                if hasattr(dataset, "set_epoch"):
                    dataset.set_epoch(epoch)

    def __iter__(self):
        iterators = {name: iter(loader) for name, loader in self.loaders.items()}
        schedule = self._schedule()
        for schedule_index in range(self.position, len(schedule)):
            name = schedule[schedule_index]
            try:
                batch = next(iterators[name])
            except StopIteration:
                iterators[name] = iter(self.loaders[name])
                try:
                    batch = next(iterators[name])
                except StopIteration as exc:
                    raise RuntimeError(f"Dataset source {name!r} produced no complete batches") from exc
            batch["scheduled_source"] = name
            self.stream_counters.update(batch.get("stream_counters", {}))
            self.position = schedule_index + 1
            self.source_batch_counts[name] += 1
            yield batch

    def __len__(self):
        return self.batches_per_epoch

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "position": self.position,
            "source_batch_counts": dict(self.source_batch_counts),
            "stream_counters": dict(self.stream_counters),
        }

    def load_state_dict(self, state):
        self.epoch = int(state.get("epoch", 0))
        self.position = int(state.get("position", 0))
        counts = state.get("source_batch_counts", {})
        self.source_batch_counts = {name: int(counts.get(name, 0)) for name in self.loaders}
        self.stream_counters = Counter({key: int(value) for key, value in state.get("stream_counters", {}).items()})


class StreamingDatasetFacade:
    """Expose the small dataset attributes used by the common training loop."""

    custom_instance_prompts = True
    resample_dataset_each_epoch = False

    def __init__(self, length):
        self.length = int(length)

    def __len__(self):
        return self.length


class PromptDataset(Dataset):
    """Small helper dataset that repeats one prompt ``num_samples`` times."""

    def __init__(self, prompt, num_samples):
        """Store the prompt text and the number of repeated samples to expose."""
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        """Return how many prompt entries the dataset should yield."""
        return self.num_samples

    def __getitem__(self, index):
        """Return the repeated prompt together with a stable numeric index."""
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def decoded_geometry_auxiliary_enabled(args: argparse.Namespace) -> bool:
    """Return whether training must retain the VAE graph through backward."""
    return any(
        weight > 0.0
        for weight in (
            args.palette_manifold_loss_weight,
            args.metric_depth_aux_weight,
            args.unit_normal_loss_weight,
            args.angular_normal_aux_weight,
        )
    )


def main(args):
    """Run the full training job from model loading through final save.

    Inputs:
    - ``args``: the parsed command-line arguments from ``parse_args``.

    Outputs:
    - Writes checkpoints, logs, and the final ``transformer`` weights into
      ``args.output_dir``.
    - Optionally uploads the result to the Hugging Face Hub.
    """
    if args.dataloader_persistent_workers and args.random_flip:
        raise ValueError("Persistent workers do not propagate augmentation epochs; disable persistent workers with random flips")
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    if args.do_fp8_training:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    logging_dir = Path(args.output_dir, args.logging_dir)

    # Set up Accelerate before any heavyweight model work so logging, precision,
    # and distributed behavior are consistent across the entire run.
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    fsdp_plugin = build_transformer_fsdp_plugin(args)
    kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(hours=4))]
    if fsdp_plugin is None:
        kwargs_handlers.append(
            DistributedDataParallelKwargs(
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
        )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=kwargs_handlers,
        fsdp_plugin=fsdp_plugin,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            save_runtime_provenance(args, args.output_dir)
            if not (args.resume_from_checkpoint and (Path(args.output_dir)/"training_args.json").exists()):
                save_training_args(args, args.output_dir)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id
    accelerator.wait_for_everyone()

    # In multi-GPU runs, concurrent first-time downloads can race on the shared
    # Hugging Face cache and leave some ranks seeing incomplete snapshots.
    if accelerator.is_main_process and not Path(args.pretrained_model_name_or_path).expanduser().exists():
        snapshot_download(
            repo_id=args.pretrained_model_name_or_path,
            revision=args.revision,
            cache_dir=args.cache_dir,
        )
    accelerator.wait_for_everyone()

    # Load the text components first because prompts may be encoded up front and
    # cached, even if the text encoder is never trained.
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load all pretrained model components. The training loop updates only the
    # selected transformer layers; the VAE and text encoder are inference-only.
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        revision=args.revision,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(accelerator.device)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        accelerator.device
    )

    quantization_config = None
    if args.bnb_quantization_config_path is not None:
        with open(args.bnb_quantization_config_path, "r") as f:
            config_kwargs = json.load(f)
            if "load_in_4bit" in config_kwargs and config_kwargs["load_in_4bit"]:
                config_kwargs["bnb_4bit_compute_dtype"] = weight_dtype
        quantization_config = BitsAndBytesConfig(**config_kwargs)

    transformer = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        quantization_config=quantization_config,
        torch_dtype=weight_dtype,
    )
    if args.bnb_quantization_config_path is not None:
        from peft import prepare_model_for_kbit_training

        transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)

    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder.requires_grad_(False)

    # Freeze non-transformer components. The transformer is selectively unfrozen below.
    vae.requires_grad_(False)

    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            transformer.set_attention_backend("_native_npu")
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu device ")

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # A decoded auxiliary loss backpropagates through the frozen VAE, so its
    # weights must stay on the accelerator until backward completes.
    differentiable_vae_decode = decoded_geometry_auxiliary_enabled(args)
    vae_offload_enabled = args.offload and not differentiable_vae_decode
    if args.offload and differentiable_vae_decode:
        logger.info("Keeping frozen VAE on %s for differentiable auxiliary decode.", accelerator.device)
    to_kwargs = (
        {"dtype": weight_dtype, "device": accelerator.device}
        if not vae_offload_enabled
        else {"dtype": weight_dtype}
    )
    # flux vae is stable in bf16 so load it in weight_dtype to reduce memory
    vae.to(**to_kwargs)
    # we never offload the transformer to CPU, so we can just use the accelerator device
    transformer_to_kwargs = (
        {"device": accelerator.device}
        if args.bnb_quantization_config_path is not None
        else {"device": accelerator.device, "dtype": weight_dtype}
    )

    is_fsdp = getattr(accelerator.state, "fsdp_plugin", None) is not None
    if not is_fsdp:
        transformer.to(**transformer_to_kwargs)

    if args.do_fp8_training:
        convert_to_float8_training(
            transformer, module_filter_fn=module_filter_fn, config=Float8LinearConfig(pad_inner_dim=True)
        )

    text_encoder.to(**to_kwargs)
    # Initialize a text encoding pipeline and keep it to CPU for now.
    text_encoding_pipeline = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
        revision=args.revision,
    )

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.train_method == "lora":
        if is_fsdp:
            raise ValueError("Klein LoRA uses replicated DDP; set --no-fsdp-transformer.")
        trainable_modules, trainable_params, total_transformer_params = configure_lora_training(transformer, args)
    else:
        trainable_modules, trainable_params, total_transformer_params = configure_partial_transformer_training(
            transformer, args
        )
    logger.info("Trainable transformer modules: " + ", ".join(trainable_modules))
    logger.info(
        f"Trainable transformer parameters: {trainable_params:,} / {total_transformer_params:,} "
        f"({100 * trainable_params / total_transformer_params:.4f}%)"
    )

    def unwrap_model(model):
        """Strip Accelerate/compile wrappers so the raw model can be saved or inspected."""
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_transformer_pretrained(model, output_dir: str, *, upcast: bool = False):
        """Save ``model`` as a normal Diffusers transformer folder."""
        transformer_dir = os.path.join(output_dir, "transformer")
        if is_fsdp:
            state_dict = accelerator.get_state_dict(model)
            if accelerator.is_main_process:
                state_dict = _maybe_upcast_state_dict(state_dict, upcast)
                save_diffusers_model_with_state_dict(
                    unwrap_model(model),
                    state_dict,
                    transformer_dir,
                    variant=args.variant,
                )
            del state_dict
            return

        if accelerator.is_main_process:
            model_to_save = unwrap_model(model)
            if args.bnb_quantization_config_path is None and upcast:
                model_to_save.to(torch.float32)
            model_to_save.save_pretrained(transformer_dir)

    def save_lora_pretrained(model, output_dir: str):
        model_to_save = unwrap_model(model)
        state_dict = _to_cpu_contiguous(get_peft_model_state_dict(model_to_save))
        Flux2KleinPipeline.save_lora_weights(
            output_dir,
            transformer_lora_layers=state_dict,
            **_collate_lora_metadata({"transformer": model_to_save}),
        )

    # Save/load the full transformer in checkpoints. Only selected layers are trainable,
    # but a normal Diffusers transformer folder is easiest to reload for inference.
    def save_model_hook(models, weights, output_dir):
        """Teach Accelerate how to save checkpoints in standard Diffusers layout."""
        checkpoint_name = Path(output_dir).name
        try:
            checkpoint_step = int(checkpoint_name.removeprefix("checkpoint-"))
        except ValueError:
            checkpoint_step = None

        # Exporting a Diffusers transformer from sharded FSDP state needs a
        # full-state gather; do it only at permanent milestones.
        export_transformer = not is_fsdp or checkpoint_step in permanent_checkpoint_steps
        for model in models:
            if not isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                raise ValueError(f"unexpected save model: {model.__class__}")
            if args.train_method == "lora":
                if accelerator.is_main_process:
                    save_lora_pretrained(model, output_dir)
            elif export_transformer:
                save_transformer_pretrained(model, output_dir)
            elif accelerator.is_main_process:
                logger.info(
                    "Keeping checkpoint-%s as sharded FSDP recovery state; "
                    "skipping full Diffusers transformer export.",
                    checkpoint_step,
                )

        while weights:
            weights.pop()

    def load_model_hook(models, input_dir):
        """Mirror ``save_model_hook`` so resume/load restores the transformer state."""
        # Accelerate restores FSDP recovery shards before invoking load hooks.
        # Loading a Diffusers transformer here would be redundant and would
        # fail for rolling checkpoints, which deliberately omit that export.
        if is_fsdp:
            return
        load_model = None
        lora_state_dict = None
        if args.train_method == "lora":
            lora_state_dict = Flux2KleinPipeline.lora_state_dict(input_dir)
        else:
            load_model = Flux2Transformer2DModel.from_pretrained(input_dir, subfolder="transformer", torch_dtype=weight_dtype)
        while len(models) > 0:
            model = models.pop()
            if not isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                raise ValueError(f"unexpected save model: {model.__class__}")
            if args.train_method == "lora":
                transformer_state = {key.removeprefix("transformer."): value for key, value in lora_state_dict.items() if key.startswith("transformer.")}
                incompatible = set_peft_model_state_dict(
                    unwrap_model(model), convert_unet_state_dict_to_peft(transformer_state), adapter_name="default"
                )
                if getattr(incompatible, "unexpected_keys", None):
                    raise ValueError(f"Unexpected LoRA checkpoint keys: {incompatible.unexpected_keys}")
            elif is_fsdp:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp import StateDictType

                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                    model.load_state_dict(load_model.state_dict())
            else:
                unwrap_model(model).load_state_dict(load_model.state_dict())
        del load_model, lora_state_dict

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    dtype_policy = args.fsdp_transformer_dtype_policy
    is_literal_full_transformer = trainable_params == total_transformer_params
    should_upcast_whole_transformer = (
        args.mixed_precision in {"fp16", "bf16"}
        and is_fsdp
        and (
            dtype_policy == "fp32_all"
            or (dtype_policy == "auto" and is_literal_full_transformer)
        )
    )
    should_upcast_trainable_params = (
        args.mixed_precision in {"fp16", "bf16"}
        and not should_upcast_whole_transformer
        and (
            not is_fsdp
            or dtype_policy == "fp32_trainable"
        )
    )
    if (
        args.mixed_precision in {"fp16", "bf16"}
        and is_fsdp
        and dtype_policy == "fp32_trainable"
        and not is_literal_full_transformer
    ):
        logger.warning(
            "--fsdp_transformer_dtype_policy fp32_trainable upcasts only trainable transformer params. "
            "For partial FSDP this may trigger FSDP uniform-dtype errors if a wrapped unit contains both "
            "frozen mixed-precision params and trainable fp32 params. Use 'bf16' for lowest memory or "
            "'fp32_all' for temp/open_visionbanana-style uniform fp32."
        )

    if should_upcast_whole_transformer:
        transformer.to(torch.float32)
        logger.info(
            "FSDP transformer dtype policy '%s': upcast the whole transformer to fp32 for uniform FSDP params.",
            dtype_policy,
        )
    elif should_upcast_trainable_params:
        models = [transformer]
        cast_training_params(models, dtype=torch.float32)
        trainable_dtype_counts = Counter(str(p.dtype) for p in transformer.parameters() if p.requires_grad)
        logger.info(
            "Trainable transformer parameter dtypes after upcast: "
            + ", ".join(f"{dtype}={count}" for dtype, count in sorted(trainable_dtype_counts.items()))
        )
    elif args.mixed_precision in {"fp16", "bf16"}:
        trainable_dtype_counts = Counter(str(p.dtype) for p in transformer.parameters() if p.requires_grad)
        logger.info(
            f"FSDP transformer dtype policy '{dtype_policy}': trainable transformer parameter dtypes kept as loaded: "
            + ", ".join(f"{dtype}={count}" for dtype, count in sorted(trainable_dtype_counts.items()))
        )

    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_trainable_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if is_fsdp and args.use_8bit_adam:
        logger.warning(
            "use_8bit_adam is not compatible with FSDP checkpoint saving; falling back to fp32 AdamW."
        )
        args.use_8bit_adam = False

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    if args.aspect_ratio_buckets is not None:
        buckets = parse_buckets_string(args.aspect_ratio_buckets)
    else:
        buckets = [(args.resolution, args.resolution)]
    logger.info(f"Using parsed aspect ratio buckets: {buckets}")

    dataset_pool = None
    dataset_specs = None
    raw_dataset_specs = None
    samples_per_epoch = None
    has_streaming_sources = False
    effective_sampling_strategy = args.sampling_strategy or "without_replacement"
    effective_max_refseg_expressions = args.max_refseg_expressions_per_image
    effective_dataset_pool_seed = args.dataset_pool_seed if args.dataset_pool_seed is not None else args.seed
    from common.training.resume_rng import TrainingRNG, loader_generator, resume_contract
    if args.dataset_specs is not None:
        dataset_specs, samples_per_epoch, config_seed, raw_dataset_specs = load_dataset_specs(
            args.dataset_specs, args.samples_per_epoch
        )
        dataset_pool_seed = effective_dataset_pool_seed if effective_dataset_pool_seed is not None else config_seed
        effective_dataset_pool_seed = dataset_pool_seed
        effective_sampling_strategy = args.sampling_strategy or raw_dataset_specs.get(
            "sampling_strategy", "without_replacement"
        )
        if effective_max_refseg_expressions is None:
            effective_max_refseg_expressions = raw_dataset_specs.get("max_refseg_expressions_per_image")
        if effective_max_refseg_expressions is not None:
            effective_max_refseg_expressions = int(effective_max_refseg_expressions)
        has_streaming_sources = any(spec.type == "webdataset_text_to_image" for spec in dataset_specs)
        if not has_streaming_sources:
            dataset_pool = WeightedDatasetPool(
                specs=dataset_specs,
                samples_per_epoch=samples_per_epoch,
                seed=dataset_pool_seed,
                image_column=args.image_column,
                cond_image_column=args.cond_image_column,
                caption_column=args.caption_column,
                shuffle=bool(raw_dataset_specs.get("shuffle", True)),
                sampling_strategy=effective_sampling_strategy,
                max_refseg_expressions_per_image=effective_max_refseg_expressions,
            )
            dataset_pool.log_summary()
            validate_target_interpolation_policies(dataset_pool)
            if accelerator.is_main_process:
                save_resolved_dataset_specs(dataset_pool, args.output_dir)
        elif accelerator.is_main_process:
            save_resolved_mixed_dataset_specs(
                dataset_specs,
                samples_per_epoch,
                dataset_pool_seed,
                bool(raw_dataset_specs.get("shuffle", True)),
                args.output_dir,
            )

    use_loss_masks = args.loss_mask_policy != "disabled"
    on_the_fly_prompt_encoding = args.prompt_encoding_mode == "on_the_fly" or (
        args.prompt_encoding_mode == "auto" and has_streaming_sources
    )
    packed_prompt_encoding = args.prompt_encoding_mode == "packed_precompute" or (
        args.prompt_encoding_mode == "auto"
        and args.dataset_specs is not None
        and args.resample_dataset_each_epoch
        and not has_streaming_sources
    )
    if has_streaming_sources and args.cache_latents:
        raise ValueError(
            "--cache_latents is incompatible with streaming WebDataset sources. "
            "Use on-the-fly VAE encoding for PD12M."
        )
    if has_streaming_sources and args.prompt_encoding_mode in {"precompute", "packed_precompute"}:
        raise ValueError(
            "Precomputed prompt encoding is incompatible with streaming WebDataset sources; "
            "use auto or on_the_fly."
        )

    if has_streaming_sources:
        args.task_homogeneous_batches = True
        global_batch_size = args.train_batch_size * max(1, accelerator.num_processes)
        requested_batches_per_epoch = max(
            1,
            math.ceil(samples_per_epoch / global_batch_size),
        )
        batches_per_epoch = (
            math.ceil(requested_batches_per_epoch / args.gradient_accumulation_steps)
            * args.gradient_accumulation_steps
        )
        source_loaders = {}
        source_weights = {spec.name: spec.weight for spec in dataset_specs}
        source_batch_quotas = allocate_weighted_counts(source_weights, batches_per_epoch)
        prepared_row_budgets = {}
        for source_index, spec in enumerate(dataset_specs):
            source_seed = int(effective_dataset_pool_seed or 0) + source_index * 1009
            if spec.type in {"prepared_folder", "indexed_prepared_tar"}:
                # Accelerate shards a random-access loader over all ranks.  A
                # source therefore needs its local batch quota times the global
                # batch size, not the complete mixed-pool epoch for every source.
                source_samples = max(global_batch_size, source_batch_quotas[spec.name] * global_batch_size)
                prepared_row_budgets[spec.name] = source_samples
                source_pool = WeightedDatasetPool(
                    specs=[spec],
                    samples_per_epoch=source_samples,
                    seed=source_seed,
                    image_column=args.image_column,
                    cond_image_column=args.cond_image_column,
                    caption_column=args.caption_column,
                    shuffle=True,
                    sampling_strategy=effective_sampling_strategy,
                    max_refseg_expressions_per_image=effective_max_refseg_expressions,
                )
                validate_target_interpolation_policies(source_pool)
                source_dataset = DreamBoothDataset(
                    instance_data_root=None,
                    instance_prompt=args.instance_prompt,
                    size=args.resolution,
                    repeats=1,
                    center_crop=args.center_crop,
                    buckets=buckets,
                    training_args=args,
                    dataset_pool=source_pool,
                )
                source_sampler = BucketBatchSampler(
                    source_dataset,
                    batch_size=args.train_batch_size,
                    drop_last=True,
                    shuffle_batches_each_epoch=True,
                    task_homogeneous_batches=True,
                    seed=source_seed,
                )
                loader = torch.utils.data.DataLoader(
                    source_dataset,
                    batch_sampler=source_sampler,
                    collate_fn=collate_fn,
                    num_workers=args.dataloader_num_workers,
                    generator=loader_generator(source_seed, spec.name),
                )
                source_loaders[spec.name] = accelerator.prepare_data_loader(loader)
            else:
                stream_dataset = WebDatasetTextToImageBatches(
                    spec=spec,
                    training_args=args,
                    buckets=buckets,
                    batch_size=args.train_batch_size,
                    seed=source_seed,
                )
                source_loaders[spec.name] = torch.utils.data.DataLoader(
                    stream_dataset,
                    batch_size=None,
                    num_workers=args.dataloader_num_workers,
                    generator=loader_generator(source_seed, spec.name),
                )
        train_dataloader = WeightedSourceBatchScheduler(
            source_loaders,
            source_weights,
            batches_per_epoch=batches_per_epoch,
            seed=effective_dataset_pool_seed,
        )
        accelerator.register_for_checkpointing(train_dataloader)
        train_dataset = StreamingDatasetFacade(
            batches_per_epoch * args.train_batch_size * max(1, accelerator.num_processes)
        )
        batch_sampler = train_dataloader
        logger.info(
            "Streaming mixed training enabled: requested_batches=%s batches_per_epoch=%s "
            "source_quotas=%s prompt_encoding=on_the_fly",
            requested_batches_per_epoch,
            batches_per_epoch,
            train_dataloader._target_counts(),
        )
        logger.info("Prepared-source row budgets before rank sharding: %s", prepared_row_budgets)
    else:
        # Existing random-access folder/Hugging Face dataset path.
        train_dataset = DreamBoothDataset(
            instance_data_root=args.instance_data_dir,
            instance_prompt=args.instance_prompt,
            size=args.resolution,
            repeats=args.repeats,
            center_crop=args.center_crop,
            buckets=buckets,
            training_args=args,
            dataset_pool=dataset_pool,
        )
        if (
            args.prompt_encoding_mode in {"auto", "precompute"}
            and train_dataset.custom_instance_prompts
        ):
            unique_prompt_limit = max(0, args.prompt_precompute_max_unique)
            unique_prompts_seen = set()
            for row in train_dataset.rows:
                unique_prompts_seen.add(row.text)
                if len(unique_prompts_seen) > unique_prompt_limit:
                    if args.prompt_encoding_mode == "auto":
                        on_the_fly_prompt_encoding = True
                        logger.info(
                            "Auto prompt encoding selected on-the-fly mode: more than %s unique "
                            "prompts would require a large disk-backed tensor cache.",
                            f"{unique_prompt_limit:,}",
                        )
                    elif not args.allow_large_prompt_precompute_cache:
                        raise ValueError(
                            "Explicit prompt precompute exceeds the safety limit of "
                            f"{unique_prompt_limit:,} unique prompts. Use on_the_fly/auto, "
                            "raise --prompt_precompute_max_unique after checking quota, or "
                            "deliberately pass --allow_large_prompt_precompute_cache."
                        )
                    break
        if args.dataset_specs is not None and args.resample_dataset_each_epoch and args.cache_latents:
            raise ValueError(
                "Dynamic dataset-pool resampling is incompatible with row-indexed latent caches. "
                "Keep --no-cache_latents for Flux2 Klein replacement sampling."
            )
        if (
            args.dataset_specs is not None
            and args.resample_dataset_each_epoch
            and args.prompt_encoding_mode == "precompute"
        ):
            raise ValueError(
                "Row-indexed prompt precompute is incompatible with epoch resampling; use "
                "--prompt_encoding_mode packed_precompute or on_the_fly."
            )
        precompute_latents = args.cache_latents or (
            train_dataset.custom_instance_prompts
            and not on_the_fly_prompt_encoding
            and not packed_prompt_encoding
        )
        has_step_indexed_caches = args.cache_latents
        batch_sampler = BucketBatchSampler(
            train_dataset,
            batch_size=args.train_batch_size,
            drop_last=True,
            shuffle_batches_each_epoch=not has_step_indexed_caches,
            task_homogeneous_batches=args.task_homogeneous_batches,
            seed=effective_dataset_pool_seed,
        )
        loader_kwargs: dict[str, Any] = {
            "num_workers": args.dataloader_num_workers,
        }
        if args.dataloader_num_workers > 0:
            if args.dataloader_timeout_seconds > 0:
                loader_kwargs["timeout"] = args.dataloader_timeout_seconds
            loader_kwargs["prefetch_factor"] = args.dataloader_prefetch_factor
            loader_kwargs["worker_init_fn"] = _install_dataloader_worker_diagnostics
            loader_kwargs["persistent_workers"] = bool(args.dataloader_persistent_workers)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=lambda examples: collate_fn(examples),
            generator=loader_generator(effective_dataset_pool_seed, "prepared_pool"),
            **loader_kwargs,
        )
    precompute_latents = False if has_streaming_sources else precompute_latents

    def compute_text_embeddings(prompt, text_encoding_pipeline):
        """Encode one prompt or a list of prompts into the transformer text inputs."""
        with torch.no_grad():
            prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
                prompt=prompt, max_sequence_length=args.max_sequence_length
            )
        return prompt_embeds, text_ids

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    if not train_dataset.custom_instance_prompts:
        with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
            instance_prompt_hidden_states, instance_text_ids = compute_text_embeddings(
                args.instance_prompt, text_encoding_pipeline
            )

    validation_prompt_records = load_validation_prompt_records(args.validation_prompts_file, args.seed)
    if args.validation_prompt is not None:
        validation_kwargs = {}
        if args.validation_image is not None:
            validation_kwargs["image"] = load_image(args.validation_image).convert("RGB")
        with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
            validation_kwargs["prompt_embeds"], _text_ids = compute_text_embeddings(
                args.validation_prompt, text_encoding_pipeline
            )
            validation_kwargs["negative_prompt_embeds"], _text_ids = compute_text_embeddings(
                "", text_encoding_pipeline
            )

    # Init FSDP for text encoder
    if args.fsdp_text_encoder:
        fsdp_kwargs = get_fsdp_kwargs_from_accelerator(accelerator)
        text_encoder_fsdp = wrap_with_fsdp(
            model=text_encoding_pipeline.text_encoder,
            device=accelerator.device,
            offload=args.offload,
            limit_all_gathers=True,
            use_orig_params=True,
            fsdp_kwargs=fsdp_kwargs,
        )

        text_encoding_pipeline.text_encoder = text_encoder_fsdp
        dist.barrier()

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.
    if not train_dataset.custom_instance_prompts:
        prompt_embeds = instance_prompt_hidden_states
        text_ids = instance_text_ids

    # if cache_latents is set to True, we encode images to latents and store them.
    # Similar to pre-encoding in the case of a single instance prompt, if custom prompts are provided
    # we encode them in advance as well.
    packed_prompt_cache = None
    if precompute_latents:
        # Disk-backed caches are used automatically for multi-process runs so each
        # worker can share the same precomputed artifacts.
        use_disk_precompute_cache = accelerator.num_processes > 1 or args.precompute_cache_dir is not None
        precompute_cache_dir = (
            Path(args.precompute_cache_dir)
            if args.precompute_cache_dir is not None
            else Path(args.output_dir) / "precomputed_cache"
        )
        # Prompt embeddings are cached once per unique prompt, not once per row.
        # This manifest lets the script detect whether an on-disk prompt cache
        # still matches the current dataset ordering and prompt text content.
        prompt_text_manifest_path = precompute_cache_dir / "prompt_text_manifest.json"

        def precompute_cache_path(kind, cache_index):
            """Return the on-disk cache file path for one precomputed tensor."""
            return precompute_cache_dir / kind / f"{int(cache_index):08d}.pt"

        def build_prompt_text_cache_signature(rows: list[PreparedRow]) -> str:
            """Hash the prompt cache inputs so stale on-disk caches can be detected."""
            hasher = hashlib.sha256()
            hasher.update(str(args.pretrained_model_name_or_path).encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(str(args.max_sequence_length).encode("utf-8"))
            hasher.update(b"\0")
            for row in rows:
                prompt = effective_instance_prompt(
                    row, train_dataset.instance_prompt, args.depth_target_encoding,
                    args.depth_prompt_contract,
                )
                hasher.update(prompt.encode("utf-8", errors="surrogatepass"))
                hasher.update(b"\0")
            return hasher.hexdigest()

        def prompt_text_cache_is_complete(expected_signature: str, num_unique_prompts: int) -> bool:
            """Check whether the prompt/text cache on disk is safe to reuse as-is."""
            if not prompt_text_manifest_path.exists():
                return False
            try:
                manifest = json.loads(prompt_text_manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False

            if not isinstance(manifest, dict):
                return False
            if manifest.get("version") != 1:
                return False
            if manifest.get("signature") != expected_signature:
                return False
            if int(manifest.get("num_unique_prompts", -1)) != num_unique_prompts:
                return False

            for prompt_cache_index in range(num_unique_prompts):
                if not precompute_cache_path("prompt_embeds", prompt_cache_index).is_file():
                    return False
                if not precompute_cache_path("text_ids", prompt_cache_index).is_file():
                    return False
            return True

        prompt_embeds_cache = [None] * len(train_dataset)
        text_ids_cache = [None] * len(train_dataset)
        latents_cache = [None] * len(train_dataset)
        cond_latents_cache = [None] * len(train_dataset)
        loss_masks_cache = [None] * len(train_dataset) if use_loss_masks else None

        should_write_precompute_cache = True
        if use_disk_precompute_cache and accelerator.is_main_process:
            cache_kinds = ["latents", "cond_latents", "prompt_embeds", "text_ids"]
            if use_loss_masks:
                cache_kinds.append("loss_masks")
            for cache_kind in cache_kinds:
                (precompute_cache_dir / cache_kind).mkdir(parents=True, exist_ok=True)
            logger.info(f"Using disk-backed precompute cache at {precompute_cache_dir}")
        if use_disk_precompute_cache:
            accelerator.wait_for_everyone()
            if accelerator.num_processes > 1:
                logger.info(f"Sharding disk-backed precompute cache work across {accelerator.num_processes} ranks.")

        if should_write_precompute_cache and args.cache_latents:
            for batch_index, batch in enumerate(
                tqdm(
                    train_dataloader,
                    desc="Caching latents",
                    disable=not accelerator.is_local_main_process,
                )
            ):
                # In distributed runs, each rank writes only its shard of cache
                # files to avoid duplicate work.
                if use_disk_precompute_cache and batch_index % accelerator.num_processes != accelerator.process_index:
                    continue
                batch_indices = batch["dataset_indices"].tolist()
                with torch.no_grad():
                    with offload_models(vae, device=accelerator.device, offload=vae_offload_enabled):
                        batch["pixel_values"] = batch["pixel_values"].to(
                            accelerator.device, non_blocking=True, dtype=vae.dtype
                        )
                        latents = vae.encode(batch["pixel_values"]).latent_dist.mode().detach().cpu()
                        for cache_index, latent in zip(batch_indices, latents):
                            cached_latent = latent.unsqueeze(0)
                            if use_disk_precompute_cache:
                                torch.save(cached_latent, precompute_cache_path("latents", cache_index))
                            else:
                                latents_cache[cache_index] = cached_latent
                        del latents
                        if "cond_pixel_values" in batch:
                            batch["cond_pixel_values"] = batch["cond_pixel_values"].to(
                                accelerator.device, non_blocking=True, dtype=vae.dtype
                            )
                            cond_latents = vae.encode(batch["cond_pixel_values"]).latent_dist.mode().detach().cpu()
                            for cache_index, latent in zip(batch_indices, cond_latents):
                                cached_latent = latent.unsqueeze(0)
                                if use_disk_precompute_cache:
                                    torch.save(cached_latent, precompute_cache_path("cond_latents", cache_index))
                                else:
                                    cond_latents_cache[cache_index] = cached_latent
                            del cond_latents
                        if use_loss_masks:
                            loss_masks = batch["loss_masks"].detach().cpu()
                            for cache_index, loss_mask in zip(batch_indices, loss_masks):
                                cached_loss_mask = loss_mask.unsqueeze(0).clone()
                                if use_disk_precompute_cache:
                                    torch.save(cached_loss_mask, precompute_cache_path("loss_masks", cache_index))
                                else:
                                    loss_masks_cache[cache_index] = cached_loss_mask
                            del loss_masks
                del batch
                if torch.cuda.is_available() and batch_index % 100 == 0:
                    torch.cuda.empty_cache()

        if should_write_precompute_cache and train_dataset.custom_instance_prompts and not on_the_fly_prompt_encoding:
            prompt_batch_size = max(
                1,
                args.prompt_embedding_batch_size
                if args.prompt_embedding_batch_size is not None
                else args.train_batch_size,
            )
            prompt_to_indices: dict[str, list[int]] = defaultdict(list)
            for cache_index, row in enumerate(train_dataset.rows):
                prompt_to_indices[effective_instance_prompt(
                    row, train_dataset.instance_prompt, args.depth_target_encoding,
                    args.depth_prompt_contract,
                )].append(cache_index)
            unique_prompts = list(prompt_to_indices.keys())
            prompt_cache_indices = {prompt: idx for idx, prompt in enumerate(unique_prompts)}
            prompt_text_cache_signature = build_prompt_text_cache_signature(train_dataset.rows)
            reuse_prompt_text_cache = (
                use_disk_precompute_cache
                and prompt_text_cache_is_complete(prompt_text_cache_signature, len(unique_prompts))
            )
            # Only the process-local prompt subset is encoded on each rank. Later
            # the per-prompt cache files are referenced back out to every row
            # that uses that prompt.
            local_unique_prompts = (
                unique_prompts[accelerator.process_index :: accelerator.num_processes]
                if use_disk_precompute_cache
                else unique_prompts
            )
            if reuse_prompt_text_cache:
                logger.info(
                    f"Reusing existing disk-backed prompt/text cache at {precompute_cache_dir} "
                    f"for {len(unique_prompts):,} unique prompts."
                )
            else:
                logger.info(
                    f"Precomputing text embeddings for {len(unique_prompts):,} unique prompts "
                    f"covering {len(train_dataset):,} rows with batch size {prompt_batch_size}."
                )
                if use_disk_precompute_cache:
                    logger.info(f"Rank {accelerator.process_index} caching {len(local_unique_prompts):,} unique prompts.")
                for start in tqdm(
                    range(0, len(local_unique_prompts), prompt_batch_size),
                    desc="Caching text embeddings",
                    disable=not accelerator.is_local_main_process,
                ):
                    prompts = local_unique_prompts[start : min(start + prompt_batch_size, len(local_unique_prompts))]
                    with torch.no_grad():
                        if args.fsdp_text_encoder:
                            prompt_embeds, text_ids = compute_text_embeddings(prompts, text_encoding_pipeline)
                        else:
                            with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
                                prompt_embeds, text_ids = compute_text_embeddings(prompts, text_encoding_pipeline)
                        prompt_embeds = prompt_embeds.detach().cpu()
                        text_ids = text_ids.detach().cpu()
                        for prompt, prompt_embed, text_id in zip(prompts, prompt_embeds, text_ids):
                            cached_prompt_embed = prompt_embed.unsqueeze(0).clone()
                            cached_text_id = text_id.unsqueeze(0).clone()
                            if use_disk_precompute_cache:
                                prompt_cache_index = prompt_cache_indices[prompt]
                                torch.save(cached_prompt_embed, precompute_cache_path("prompt_embeds", prompt_cache_index))
                                torch.save(cached_text_id, precompute_cache_path("text_ids", prompt_cache_index))
                            else:
                                for cache_index in prompt_to_indices[prompt]:
                                    prompt_embeds_cache[cache_index] = cached_prompt_embed
                                    text_ids_cache[cache_index] = cached_text_id
                        del prompt_embeds, text_ids
                    if torch.cuda.is_available() and start % (100 * prompt_batch_size) == 0:
                        torch.cuda.empty_cache()

        if use_disk_precompute_cache:
            accelerator.wait_for_everyone()
            if args.cache_latents:
                latents_cache = [precompute_cache_path("latents", i) for i in range(len(train_dataset))]
                cond_latents_cache = [precompute_cache_path("cond_latents", i) for i in range(len(train_dataset))]
                if use_loss_masks:
                    loss_masks_cache = [precompute_cache_path("loss_masks", i) for i in range(len(train_dataset))]
            if train_dataset.custom_instance_prompts and not on_the_fly_prompt_encoding:
                if accelerator.is_main_process:
                    # Persist enough metadata to decide on the next run whether
                    # the unique-prompt cache can be reused without recomputing.
                    manifest = {
                        "version": 1,
                        "signature": prompt_text_cache_signature,
                        "num_rows": len(train_dataset),
                        "num_unique_prompts": len(unique_prompts),
                    }
                    prompt_text_manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
                prompt_embeds_cache = [None] * len(train_dataset)
                text_ids_cache = [None] * len(train_dataset)
                for prompt, cache_indices in prompt_to_indices.items():
                    prompt_cache_index = prompt_cache_indices[prompt]
                    prompt_embed_path = precompute_cache_path("prompt_embeds", prompt_cache_index)
                    text_id_path = precompute_cache_path("text_ids", prompt_cache_index)
                    for cache_index in cache_indices:
                        prompt_embeds_cache[cache_index] = prompt_embed_path
                        text_ids_cache[cache_index] = text_id_path

    if packed_prompt_encoding and train_dataset.custom_instance_prompts:
        unique_prompts = sorted({
            effective_instance_prompt(
                row, train_dataset.instance_prompt, args.depth_target_encoding,
                args.depth_prompt_contract,
            )
            for row in train_dataset.rows
        })
        packed_fingerprint = prompt_cache_fingerprint(
            unique_prompts,
            str(args.pretrained_model_name_or_path),
            args.max_sequence_length,
        )
        packed_root_base = (
            Path(args.precompute_cache_dir)
            if args.precompute_cache_dir is not None
            else Path(args.output_dir) / "precomputed_cache"
        )
        packed_root = packed_root_base / "packed_prompts" / packed_fingerprint[:16]
        expected_prompt_keys = [prompt_key(prompt) for prompt in unique_prompts]
        cache_ready = False
        try:
            packed_prompt_cache = PackedPromptCache(packed_root, packed_fingerprint)
            cache_ready = packed_prompt_cache.contains_all(expected_prompt_keys)
        except (FileNotFoundError, RuntimeError, json.JSONDecodeError):
            packed_prompt_cache = None

        if not cache_ready:
            logger.info(
                "Building packed prompt cache for %s unique prompts at %s",
                f"{len(unique_prompts):,}",
                packed_root,
            )
            local_prompts = unique_prompts[accelerator.process_index :: accelerator.num_processes]
            writer = PackedPromptCacheWriter(
                packed_root,
                packed_fingerprint,
                accelerator.process_index,
                args.packed_prompt_shard_bytes,
            )
            prompt_batch_size = max(1, int(args.prompt_embedding_batch_size or args.train_batch_size))
            for start in tqdm(
                range(0, len(local_prompts), prompt_batch_size),
                desc="Packing prompt embeddings",
                disable=not accelerator.is_local_main_process,
            ):
                prompts = local_prompts[start : start + prompt_batch_size]
                with torch.no_grad():
                    if args.fsdp_text_encoder:
                        encoded_prompts, encoded_text_ids = compute_text_embeddings(prompts, text_encoding_pipeline)
                    else:
                        with offload_models(
                            text_encoding_pipeline, device=accelerator.device, offload=args.offload
                        ):
                            encoded_prompts, encoded_text_ids = compute_text_embeddings(
                                prompts, text_encoding_pipeline
                            )
                    for prompt, prompt_embed, encoded_ids in zip(
                        prompts, encoded_prompts, encoded_text_ids
                    ):
                        writer.add(prompt_key(prompt), prompt_embed, encoded_ids)
                    del encoded_prompts, encoded_text_ids
            writer.close()
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                finalize_packed_prompt_cache(
                    packed_root,
                    packed_fingerprint,
                    accelerator.num_processes,
                    expected_prompt_keys,
                )
            accelerator.wait_for_everyone()
            packed_prompt_cache = PackedPromptCache(packed_root, packed_fingerprint)
        logger.info("Using packed prompt cache at %s", packed_root)

    # move back to cpu before deleting to ensure memory is freed see: https://github.com/huggingface/diffusers/issues/11376#issue-3008144624
    if args.cache_latents:
        vae = vae.to("cpu")
        del vae

    # move back to cpu before deleting to ensure memory is freed see: https://github.com/huggingface/diffusers/issues/11376#issue-3008144624
    if on_the_fly_prompt_encoding:
        text_encoding_pipeline = text_encoding_pipeline.to("cpu" if args.offload else accelerator.device)
    else:
        text_encoding_pipeline = text_encoding_pipeline.to("cpu")
    del text_encoder, tokenizer
    free_memory()

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = (
            len(train_dataloader)
            if has_streaming_sources
            else math.ceil(len(train_dataloader) / accelerator.num_processes)
        )
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * accelerator.num_processes * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if has_streaming_sources:
        transformer, optimizer, lr_scheduler = accelerator.prepare(transformer, optimizer, lr_scheduler)
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "flux2-klein-partial-image2img"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    resume_step = None
    stop_requested = False

    def request_graceful_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        logger.warning(f"Received signal {signum}; checkpointing at the next synchronized optimizer step.")

    signal.signal(signal.SIGUSR1, request_graceful_stop)
    signal.signal(signal.SIGUSR2, _dump_python_stack)
    permanent_checkpoint_steps = {
        int(value.strip())
        for value in str(args.permanent_checkpoint_steps or "").split(",")
        if value.strip()
    }

    def write_sampler_state(checkpoint_dir: str, checkpoint_step: int, checkpoint_epoch: int) -> None:
        """Persist the deterministic data cursor alongside Accelerate's RNG state."""
        if not accelerator.is_main_process:
            return
        batches_consumed = (
            checkpoint_step - checkpoint_epoch * num_update_steps_per_epoch
        ) * args.gradient_accumulation_steps
        payload = {
            "schema_version": 2,
            "global_step": checkpoint_step,
            "epoch": checkpoint_epoch,
            "batches_consumed_in_epoch": batches_consumed,
            "dataset_pool_seed": effective_dataset_pool_seed,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "resume_contract": training_rng.contract,
        }
        target = Path(checkpoint_dir) / "sampler_state.json"
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, target)

    def checkpoint_is_valid(checkpoint_dir: str | os.PathLike) -> bool:
        """Return whether a checkpoint contains every state needed for exact recovery."""
        checkpoint_path = Path(checkpoint_dir)
        required = [checkpoint_path / "scheduler.bin", checkpoint_path / "sampler_state.json"]
        required.extend(checkpoint_path / f"random_states_{rank}.pkl" for rank in range(accelerator.num_processes))
        if not args.allow_legacy_rng_resume:
            required.extend(checkpoint_path / f"training_rng_{rank}.pt" for rank in range(accelerator.num_processes))
        if any(not item.is_file() or item.stat().st_size == 0 for item in required):
            return False
        if is_fsdp:
            # torch.distributed.checkpoint writes the metadata file only after
            # it has successfully committed the model/optimizer shard set.
            def complete_sharded_state(path: Path) -> bool:
                metadata = path / ".metadata"
                return (
                    metadata.is_file()
                    and metadata.stat().st_size > 0
                    and any(item.is_file() and item.stat().st_size > 0 for item in path.glob("*.distcp"))
                )

            model_state_dirs = (
                checkpoint_path / "pytorch_model_fsdp_0",
                checkpoint_path / "pytorch_model_0",
            )
            return any(complete_sharded_state(path) for path in model_state_dirs) and complete_sharded_state(
                checkpoint_path / "optimizer_0"
            )
        required.append(checkpoint_path / "optimizer.bin")
        if any(not item.is_file() or item.stat().st_size == 0 for item in required):
            return False
        if args.train_method == "lora":
            return any(item.stat().st_size > 0 for item in checkpoint_path.glob("*lora*.safetensors"))
        transformer_path = checkpoint_path / "transformer"
        return (transformer_path / "config.json").is_file() and any(
            item.stat().st_size > 0
            for pattern in ("*.safetensors", "*.bin")
            for item in transformer_path.glob(pattern)
        )

    def prune_stale_resume_artifacts(keep_step: int) -> None:
        """Drop resume-only state (optimizer, FSDP model shards, RNG) from every
        checkpoint older than ``keep_step``, keeping only the exported
        ``transformer/`` weights there.

        A stripped checkpoint fails ``checkpoint_is_valid`` and is skipped by
        ``resume_from_checkpoint=latest``.
        """
        if not accelerator.is_main_process:
            return
        output_dir = Path(args.output_dir)
        for checkpoint_path in output_dir.glob("checkpoint-*"):
            if not checkpoint_path.is_dir():
                continue
            try:
                step = int(checkpoint_path.name.split("-")[1])
            except (IndexError, ValueError):
                continue
            if step >= keep_step:
                continue
            stale = [
                checkpoint_path / "optimizer.bin",
                checkpoint_path / "pytorch_model_fsdp.bin",
                *checkpoint_path.glob("random_states_*.pkl"),
                *checkpoint_path.glob("training_rng_*.pt"),
            ]
            if is_fsdp:
                stale.extend(
                    [
                        checkpoint_path / "optimizer_0",
                        checkpoint_path / "pytorch_model_fsdp_0",
                        checkpoint_path / "pytorch_model_0",
                    ]
                )
            for item in stale:
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink(missing_ok=True)
                except OSError as error:
                    logger.warning(f"Could not remove stale resume artifact {item}: {error}")
        logger.info(f"Pruned resume artifacts from checkpoints older than checkpoint-{keep_step}")

    training_rng = TrainingRNG(args.seed, accelerator.process_index, accelerator.device,
                              resume_contract(args, accelerator.num_processes, effective_dataset_pool_seed))
    accelerator.register_save_state_pre_hook(lambda models, weights, directory: training_rng.save(directory))

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint.
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            valid_dirs = [d for d in dirs if checkpoint_is_valid(os.path.join(args.output_dir, d))]
            ignored_dirs = sorted(set(dirs) - set(valid_dirs))
            if ignored_dirs:
                logger.warning(f"Ignoring incomplete checkpoints: {', '.join(ignored_dirs)}")
            path = valid_dirs[-1] if valid_dirs else None
            if dirs and path is None:
                raise ValueError("No complete compatible checkpoint; refusing to silently restart existing training. "
                                 "Legacy checkpoints require explicit --allow_legacy_rng_resume.")

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            restored_rng = training_rng.load(os.path.join(args.output_dir, path),
                                            allow_legacy=args.allow_legacy_rng_resume)
            if not restored_rng:
                logger.warning("Explicit legacy RNG migration: continuation is NOT exact reproduction.")
            from common.training.fsdp_resume import load_training_state
            load_training_state(accelerator, os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            sampler_state_path = Path(args.output_dir) / path / "sampler_state.json"
            if sampler_state_path.is_file():
                sampler_state = json.loads(sampler_state_path.read_text(encoding="utf-8"))
                if int(sampler_state.get("global_step", -1)) != global_step:
                    raise ValueError(f"Sampler state does not match checkpoint step: {sampler_state_path}")
                for key, expected in (("dataset_pool_seed", effective_dataset_pool_seed),
                                      ("gradient_accumulation_steps", args.gradient_accumulation_steps)):
                    if not restored_rng and key=="dataset_pool_seed" and sampler_state.get(key) is None:
                        logger.warning("Legacy checkpoint has no resolved dataset seed; exact data provenance is unverified.")
                        continue
                    if key in sampler_state and sampler_state[key] != expected:
                        raise ValueError(f"Incompatible resume {key}: {sampler_state_path}")
                first_epoch = int(sampler_state["epoch"])
                resume_step = int(sampler_state["batches_consumed_in_epoch"])
            else:
                # Backward-compatible recovery for checkpoints created before
                # sampler_state.json became part of the checkpoint contract.
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = (global_step - first_epoch * num_update_steps_per_epoch) * args.gradient_accumulation_steps
                logger.warning(f"Checkpoint {path} predates sampler-state persistence; reconstructing its cursor.")
            logger.info(f"Resume will skip {resume_step:,} already-trained batches in epoch {first_epoch}.")
            if accelerator.is_main_process:
                removed_metrics = truncate_training_metrics(args.output_dir, global_step)
                if removed_metrics:
                    logger.warning(
                        "Removed %s training-metric rows newer than resumed checkpoint step %s.",
                        removed_metrics,
                        global_step,
                    )
            accelerator.wait_for_everyone()

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        """Map scheduler timesteps to sigma values with the right broadcast shape."""
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def get_cached_batch(cache, batch_indices, cache_name, dtype=None):
        """Load a batch of cached tensors from memory or disk by dataset index."""
        values = []
        for cache_index in batch_indices.tolist():
            value = cache[int(cache_index)]
            if value is None:
                raise RuntimeError(f"Missing cached {cache_name} for dataset index {int(cache_index)}.")
            if isinstance(value, (str, os.PathLike, Path)):
                if not Path(value).exists():
                    raise RuntimeError(f"Missing cached {cache_name} file for dataset index {int(cache_index)}: {value}")
                value = torch.load(value, map_location="cpu", weights_only=True)
            values.append(value)
        tensor = torch.cat(values, dim=0).to(device=accelerator.device, non_blocking=True)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor

    train_loss = 0.0
    train_auxiliary_loss = 0.0
    last_batch_end = time.perf_counter()
    optimizer_window_start = last_batch_end
    accumulated_data_wait = 0.0
    optimizer_window_rows: list[dict[str, Any]] = []
    for epoch in range(first_epoch, args.num_train_epochs):
        if has_streaming_sources:
            batch_sampler.set_epoch(epoch)
        elif train_dataset.resample_dataset_each_epoch or batch_sampler.shuffle_batches_each_epoch:
            batch_sampler.set_epoch(epoch)
        transformer.train()

        active_dataloader = train_dataloader
        if resume_step is not None and epoch == first_epoch and resume_step > 0:
            active_dataloader = (
                itertools.islice(train_dataloader, resume_step, None)
                if has_streaming_sources
                else accelerator.skip_first_batches(train_dataloader, resume_step)
            )

        for step, batch in enumerate(active_dataloader):
            batch_ready = time.perf_counter()
            accumulated_data_wait += batch_ready - last_batch_end
            models_to_accumulate = [transformer]
            prompts = batch["prompts"]
            optimizer_window_rows.extend(
                {
                    "dataset": dataset_name,
                    "source_dataset": source_dataset,
                    "task": task,
                    "row_id": row_id,
                }
                for dataset_name, source_dataset, task, row_id in zip(
                    batch.get("dataset_names", []),
                    batch.get("source_datasets", []),
                    batch.get("tasks", []),
                    batch.get("row_ids", []),
                )
            )
            has_conditioning = bool(batch.get("has_conditioning", "cond_pixel_values" in batch))
            if has_conditioning != ("cond_pixel_values" in batch):
                raise ValueError("Batch conditioning flag does not match cond_pixel_values presence.")

            with accelerator.accumulate(models_to_accumulate):
                batch_indices = batch["dataset_indices"]
                if on_the_fly_prompt_encoding:
                    with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
                        prompt_embeds, text_ids = compute_text_embeddings(prompts, text_encoding_pipeline)
                elif packed_prompt_encoding and train_dataset.custom_instance_prompts:
                    prompt_embeds, text_ids = packed_prompt_cache.load_batch(prompts)
                    prompt_embeds = prompt_embeds.to(accelerator.device, non_blocking=True)
                    text_ids = text_ids.to(accelerator.device, non_blocking=True)
                elif train_dataset.custom_instance_prompts:
                    # Custom prompts were pre-encoded per dataset row, so pull the
                    # exact cached embedding for each example in the batch.
                    prompt_embeds = get_cached_batch(prompt_embeds_cache, batch_indices, "prompt embeddings")
                    text_ids = get_cached_batch(text_ids_cache, batch_indices, "text ids")
                else:
                    # When every row shares one prompt, we encode once and simply
                    # repeat the same embedding across the batch.
                    num_repeat_elements = len(prompts)
                    prompt_embeds = prompt_embeds.repeat(num_repeat_elements, 1, 1)
                    text_ids = text_ids.repeat(num_repeat_elements, 1, 1)

                # Convert images to latent space
                if args.cache_latents:
                    model_input = get_cached_batch(latents_cache, batch_indices, "latents", dtype=weight_dtype)
                    loss_mask = (
                        get_cached_batch(loss_masks_cache, batch_indices, "loss masks", dtype=torch.float32)
                        if use_loss_masks
                        else None
                    )
                    cond_model_input = None
                    if "cond_pixel_values" in batch:
                        cond_model_input = get_cached_batch(
                            cond_latents_cache, batch_indices, "conditioning latents", dtype=weight_dtype
                        )
                else:
                    loss_mask = (
                        batch["loss_masks"].to(device=accelerator.device, dtype=torch.float32)
                        if use_loss_masks
                        else None
                    )
                    with offload_models(vae, device=accelerator.device, offload=vae_offload_enabled):
                        pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=vae.dtype)
                        model_input = vae.encode(pixel_values).latent_dist.mode()
                        cond_model_input = None
                        if "cond_pixel_values" in batch:
                            cond_pixel_values = batch["cond_pixel_values"].to(device=accelerator.device, dtype=vae.dtype)
                            cond_model_input = vae.encode(cond_pixel_values).latent_dist.mode()

                # The transformer operates on token-like latent patches rather than
                # on 2D feature maps, so the VAE latents are patchified next.
                model_input = Flux2KleinPipeline._patchify_latents(model_input)
                model_input = (model_input - latents_bn_mean) / latents_bn_std

                model_input_ids = Flux2KleinPipeline._prepare_latent_ids(model_input).to(device=model_input.device)
                cond_model_input_ids = None
                if cond_model_input is not None:
                    cond_model_input = Flux2KleinPipeline._patchify_latents(cond_model_input)
                    cond_model_input = (cond_model_input - latents_bn_mean) / latents_bn_std
                    # Conditioning image patches are concatenated onto the same
                    # token stream, and their image-position ids are built to match.
                    cond_model_input_list = [
                        cond_model_input[i].unsqueeze(0) for i in range(cond_model_input.shape[0])
                    ]
                    cond_model_input_ids = Flux2KleinPipeline._prepare_image_ids(cond_model_input_list).to(
                        device=cond_model_input.device
                    )
                    cond_model_input_ids = cond_model_input_ids.view(
                        cond_model_input.shape[0], -1, model_input_ids.shape[-1]
                    )

                # Sample noise that we'll add to the latents
                noise = torch.randn(model_input.shape, device=model_input.device, dtype=model_input.dtype,
                                    generator=training_rng.noise)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                    generator=training_rng.timesteps,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                # [B, C, H, W] -> [B, H*W, C]
                packed_noisy_model_input = Flux2KleinPipeline._pack_latents(noisy_model_input)
                orig_input_ids_shape = model_input_ids.shape

                packed_cond_model_input = None
                if cond_model_input is not None:
                    # Append the conditioning tokens so the transformer can attend
                    # over both the noisy target latents and the source image.
                    packed_cond_model_input = Flux2KleinPipeline._pack_latents(cond_model_input)
                packed_noisy_model_input, model_input_ids, target_token_length = append_conditioning_tokens(
                    packed_noisy_model_input,
                    model_input_ids,
                    packed_cond_model_input,
                    cond_model_input_ids,
                )

                # handle guidance
                if unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.full([1], args.guidance_scale, device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,  # (B, image_seq_len, C)
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,  # B, text_seq_len, 4
                    img_ids=model_input_ids,  # B, image_seq_len, 4
                    return_dict=False,
                )[0]
                # Drop predictions that correspond to conditioning tokens; the loss
                # is only defined on the target image latents.
                model_pred = model_pred[:, :target_token_length, :]
                model_input_ids = model_input_ids[:, : orig_input_ids_shape[1], :]

                model_pred = Flux2KleinPipeline._unpack_latents_with_ids(model_pred, model_input_ids)

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - model_input

                per_element_loss = weighting.float() * (model_pred.float() - target.float()) ** 2
                sample_valid = batch.get("sample_valid")
                if sample_valid is None:
                    sample_valid = torch.ones(target.shape[0], device=target.device, dtype=torch.bool)
                else:
                    sample_valid = sample_valid.to(device=target.device, dtype=torch.bool)
                if use_loss_masks:
                    loss_mask = loss_mask.to(device=target.device, dtype=per_element_loss.dtype)
                    loss_mask = prepare_latent_loss_mask(
                        loss_mask, target.shape[-2:], mode=args.loss_mask_mode,
                        min_valid_fraction=args.loss_mask_min_valid_fraction,
                        erosion_radius=args.loss_mask_erosion_radius,
                    )
                    expanded_loss_mask = loss_mask.expand_as(per_element_loss)
                    valid_counts, sample_valid_fraction, eligible = eligible_loss_mask_samples(
                        loss_mask,
                        expanded_loss_mask,
                        args.loss_mask_min_sample_valid_fraction,
                    )
                    eligible = eligible & sample_valid
                    if args.loss_mask_reduction == "global_valid":
                        eligible_weights = expanded_loss_mask * eligible[:, None, None, None]
                        loss = normalized_distributed_loss((per_element_loss * eligible_weights).sum(), eligible_weights.sum())
                    else:
                        per_sample = (per_element_loss * expanded_loss_mask).reshape(target.shape[0], -1).sum(1)
                        per_sample = per_sample / valid_counts.clamp_min(1.0)
                        eligible_float = eligible.to(per_sample.dtype)
                        loss = normalized_distributed_loss((per_sample * eligible_float).sum(), eligible_float.sum())
                else:
                    # Whole-image loss, excluding timeout placeholders from dataloader workers.
                    per_sample = torch.mean(per_element_loss.reshape(target.shape[0], -1), 1)
                    valid_weights = sample_valid.to(per_sample.dtype)
                    loss = normalized_distributed_loss((per_sample * valid_weights).sum(), valid_weights.sum())
                loss = loss.mean()
                auxiliary_loss = None
                normalized_tasks = [normalize_task_name(task) for task in batch.get("tasks", [])]
                depth_batch = bool(normalized_tasks) and all(task in DEPTH_TASKS for task in normalized_tasks)
                normal_batch = bool(normalized_tasks) and all(
                    task in SURFACE_NORMAL_TASKS for task in normalized_tasks
                )
                active_auxiliary_weight = 0.0
                if depth_batch:
                    active_auxiliary_weight = (
                        args.palette_manifold_loss_weight
                        or args.metric_depth_aux_weight
                    )
                elif normal_batch:
                    active_auxiliary_weight = (
                        args.unit_normal_loss_weight
                        or args.angular_normal_aux_weight
                    )
                if active_auxiliary_weight > 0.0 and bool(sample_valid.all()):
                    predicted_clean = recover_clean_latents(
                        noisy_model_input,
                        model_pred,
                        sigmas,
                    )
                    predicted_clean = predicted_clean * latents_bn_std + latents_bn_mean
                    predicted_clean = Flux2KleinPipeline._unpatchify_latents(predicted_clean)
                    with offload_models(vae, device=accelerator.device, offload=vae_offload_enabled):
                        predicted_pixels = vae.decode(
                            predicted_clean.to(dtype=vae.dtype),
                            return_dict=False,
                        )[0]
                    predicted_rgb = (predicted_pixels.float() + 1.0) * 0.5
                    target_rgb = (
                        batch["pixel_values"].to(
                            device=predicted_rgb.device,
                            dtype=torch.float32,
                        )
                        + 1.0
                    ) * 0.5
                    auxiliary_mask = (
                        batch["loss_masks"].to(
                            device=predicted_rgb.device,
                            dtype=torch.float32,
                        )
                        if use_loss_masks and "loss_masks" in batch
                        else None
                    )
                    if args.palette_manifold_loss_weight > 0.0 and depth_batch:
                        auxiliary_loss = palette_manifold_loss(predicted_rgb, auxiliary_mask)
                        loss = loss + args.palette_manifold_loss_weight * auxiliary_loss
                    elif args.metric_depth_aux_weight > 0.0 and depth_batch:
                        auxiliary_loss = robust_metric_depth_loss(
                            predicted_rgb,
                            target_rgb,
                            auxiliary_mask,
                            max_depth=args.metric_depth_aux_max_depth,
                        )
                        loss = loss + args.metric_depth_aux_weight * auxiliary_loss
                    elif args.unit_normal_loss_weight > 0.0 and normal_batch:
                        auxiliary_loss = unit_normal_loss(
                            predicted_rgb,
                            auxiliary_mask,
                            normal_encoding=args.normal_target_encoding,
                        )
                        loss = loss + args.unit_normal_loss_weight * auxiliary_loss
                    elif args.angular_normal_aux_weight > 0.0 and normal_batch:
                        auxiliary_loss = angular_normal_loss(
                            predicted_rgb,
                            target_rgb,
                            auxiliary_mask,
                            normal_encoding=args.normal_target_encoding,
                        )
                        loss = loss + args.angular_normal_aux_weight * auxiliary_loss
                train_loss += loss.detach().float().item() / args.gradient_accumulation_steps
                if auxiliary_loss is not None:
                    train_auxiliary_loss += (
                        auxiliary_loss.detach().float().item()
                        / args.gradient_accumulation_steps
                    )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = [p for p in transformer.parameters() if p.requires_grad]
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Permanent milestones are save points as well as retention
                # exemptions. They need not align with the rolling checkpoint
                # interval (for example, 6,250 with a 500-step interval).
                if (
                    global_step % args.checkpointing_steps == 0
                    or global_step in permanent_checkpoint_steps
                ):
                    # Before saving state, prune only rolling checkpoints. Permanent
                    # milestones are excluded from this count and never removed.
                    if accelerator.is_main_process and args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        rolling_checkpoints = [
                            checkpoint
                            for checkpoint in checkpoints
                            if int(checkpoint.split("-")[1]) not in permanent_checkpoint_steps
                        ]

                        if len(rolling_checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(rolling_checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = rolling_checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    accelerator.wait_for_everyone()
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    if accelerator.is_main_process and os.path.isdir(save_path) and not checkpoint_is_valid(save_path):
                        logger.warning(f"Removing incomplete checkpoint before rewriting: {save_path}")
                        shutil.rmtree(save_path)
                    accelerator.wait_for_everyone()
                    if not checkpoint_is_valid(save_path):
                        accelerator.save_state(save_path)
                        accelerator.wait_for_everyone()
                        write_sampler_state(save_path, global_step, epoch)
                    else:
                        logger.warning(f"Checkpoint already exists at {save_path}; reusing it")
                    accelerator.wait_for_everyone()
                    logger.info(f"Saved state to {save_path}")
                    if accelerator.is_main_process and checkpoint_is_valid(save_path):
                        prune_stale_resume_artifacts(global_step)

                if stop_requested:
                    accelerator.wait_for_everyone()
                    if (
                        global_step % args.checkpointing_steps != 0
                        and global_step not in permanent_checkpoint_steps
                    ):
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        if accelerator.is_main_process and os.path.isdir(save_path) and not checkpoint_is_valid(save_path):
                            logger.warning(f"Removing incomplete signal checkpoint before rewriting: {save_path}")
                            shutil.rmtree(save_path)
                        accelerator.wait_for_everyone()
                        if not checkpoint_is_valid(save_path):
                            accelerator.save_state(save_path)
                            accelerator.wait_for_everyone()
                            write_sampler_state(save_path, global_step, epoch)
                        else:
                            logger.warning(f"Signal checkpoint already exists at {save_path}; reusing it")
                        accelerator.wait_for_everyone()
                        logger.warning(f"Saved signal checkpoint to {save_path}")
                        if accelerator.is_main_process and checkpoint_is_valid(save_path):
                            prune_stale_resume_artifacts(global_step)
                    accelerator.wait_for_everyone()
                    accelerator.end_training()
                    raise SystemExit(75)

                window_seconds = max(time.perf_counter() - optimizer_window_start, 1e-9)
                logs = {"loss": train_loss, "lr": lr_scheduler.get_last_lr()[0],
                        "data_wait_seconds": accumulated_data_wait,
                        "data_wait_fraction": accumulated_data_wait / window_seconds}
                if train_auxiliary_loss:
                    logs["auxiliary_loss"] = train_auxiliary_loss
                if has_streaming_sources:
                    for source_name, count in train_dataloader.source_batch_counts.items():
                        logs[f"source_batches/{source_name}"] = count
                        logs[f"source_samples/{source_name}"] = (
                            count * args.train_batch_size * max(1, accelerator.num_processes)
                        )
                    for counter_name, count in train_dataloader.stream_counters.items():
                        logs[f"stream/{counter_name}"] = count
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                if accelerator.is_main_process:
                    metrics_path = Path(args.output_dir) / "training_metrics" / "metrics.jsonl"
                    metrics_path.parent.mkdir(parents=True, exist_ok=True)
                    record = {
                        "global_step": global_step,
                        "epoch": epoch,
                        **logs,
                        "rows": optimizer_window_rows,
                    }
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                train_loss = 0.0
                train_auxiliary_loss = 0.0
                accumulated_data_wait = 0.0
                optimizer_window_rows = []
                optimizer_window_start = time.perf_counter()

            if global_step >= args.max_train_steps:
                break
            last_batch_end = time.perf_counter()

        if accelerator.is_main_process:
            if (args.validation_prompt is not None or validation_prompt_records) and epoch % args.validation_epochs == 0:
                # create pipeline
                pipeline = Flux2KleinPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    text_encoder=None,
                    tokenizer=None,
                    transformer=unwrap_model(transformer),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                images = []
                if args.validation_prompt is not None:
                    images = log_validation(
                        pipeline=pipeline,
                        args=args,
                        accelerator=accelerator,
                        pipeline_args=validation_kwargs,
                        epoch=epoch,
                        torch_dtype=weight_dtype,
                    )
                if validation_prompt_records:
                    log_fixed_text_validation(
                        pipeline,
                        text_encoding_pipeline,
                        validation_prompt_records,
                        args,
                        accelerator,
                        f"epoch-{epoch:04d}",
                        weight_dtype,
                    )

                del pipeline
                free_memory()

    # Save the fine-tuned transformer. The VAE and text encoder remain loaded from the base model.
    accelerator.wait_for_everyone()
    if args.train_method == "lora":
        if accelerator.is_main_process:
            save_lora_pretrained(transformer, args.output_dir)
    else:
        save_transformer_pretrained(transformer, args.output_dir, upcast=args.upcast_before_saving)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        images = []
        run_validation = (
            (args.validation_prompt and args.num_validation_images > 0)
            or args.final_validation_prompt
            or validation_prompt_records
        )
        should_run_final_inference = not args.skip_final_inference and run_validation
        if should_run_final_inference:
            transformer_for_pipeline = (
                Flux2Transformer2DModel.from_pretrained(
                    args.output_dir,
                    subfolder="transformer",
                    torch_dtype=weight_dtype,
                )
                if is_fsdp
                else unwrap_model(transformer)
            )
            pipeline = Flux2KleinPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=None,
                tokenizer=None,
                transformer=transformer_for_pipeline,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )

            # run inference
            images = []
            if args.validation_prompt and args.num_validation_images > 0:
                images = log_validation(
                    pipeline=pipeline,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=validation_kwargs,
                    epoch=epoch,
                    is_final_validation=True,
                    torch_dtype=weight_dtype,
                )
            if validation_prompt_records:
                log_fixed_text_validation(
                    pipeline,
                    text_encoding_pipeline,
                    validation_prompt_records,
                    args,
                    accelerator,
                    "final",
                    weight_dtype,
                )
            del pipeline
            free_memory()

        validation_prompt = args.validation_prompt if args.validation_prompt else args.final_validation_prompt
        save_model_card(
            (args.hub_model_id or Path(args.output_dir).name) if not args.push_to_hub else repo_id,
            images=images,
            base_model=args.pretrained_model_name_or_path,
            instance_prompt=args.instance_prompt,
            validation_prompt=validation_prompt,
            repo_folder=args.output_dir,
            fp8_training=args.do_fp8_training,
            load_existing_card=args.push_to_hub,
        )

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except (ValueError, FileNotFoundError, json.JSONDecodeError, ImportError) as exc:
        # Deterministic configuration failures exit 2 so a scheduler wrapper can
        # mark the run fatal. Print on every rank: Accelerate silences non-main loggers.
        rank = os.environ.get("RANK", "unknown")
        print(
            f"[RANK {rank}] Fatal configuration or dataset error: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        logger.exception("Fatal configuration or dataset error: %s", exc, main_process_only=False)
        raise SystemExit(2) from exc
