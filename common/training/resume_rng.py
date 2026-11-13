"""Versioned, rank-local diffusion RNG and sample-identity augmentation.

Loader construction and replaying skipped rows must not consume diffusion RNG.
Legacy checkpoints can be migrated explicitly, but cannot reproduce missing states.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import random

POLICY = "isolated_diffusion_sample_augmentation_v1"


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, default=str).encode()).digest()[:8], "little") % (2**63 - 1)


def augmentation_rng(seed, epoch, dataset, task, row_id, occurrence):
    return random.Random(stable_seed(POLICY, seed, epoch, dataset, task, row_id, occurrence))


def resume_contract(args, world_size, dataset_seed):
    # Residency/cache/loader-worker count deliberately do not change this recipe.
    names = ("seed", "train_batch_size", "gradient_accumulation_steps", "resolution",
             "center_crop", "random_flip", "max_train_steps", "learning_rate",
             "lr_scheduler", "lr_warmup_steps", "weighting_scheme", "logit_mean",
             "logit_std", "mode_scale", "loss_mask_policy", "depth_target_encoding",
             "normal_target_encoding", "depth_target_resize_policy",
             "normal_target_resize_policy", "depth_prompt_contract", "normal_prompt_contract",
             "mixed_precision", "train_method", "trainable_scope_contract",
             "train_last_transformer_blocks", "train_last_single_transformer_blocks",
             "resample_dataset_each_epoch", "samples_per_epoch", "sampling_strategy",
             "optimizer", "adam_beta1", "adam_beta2", "adam_epsilon", "adam_weight_decay",
             "max_grad_norm", "scale_lr", "loss_mask_reduction", "loss_mask_mode",
             "loss_mask_min_valid_fraction", "loss_mask_erosion_radius",
             "loss_mask_min_sample_valid_fraction", "angular_normal_aux_weight",
             "unit_normal_loss_weight", "metric_depth_aux_weight", "palette_manifold_loss_weight")
    names += ("train_full_transformer", "train_input_layers", "train_output_layers",
              "train_transformer_block_indices", "train_single_transformer_block_indices",
              "fsdp_transformer_dtype_policy", "max_sequence_length", "guidance_scale")
    result = {name: getattr(args, name, None) for name in names}
    path = getattr(args, "dataset_specs", None)
    result.update(world_size=int(world_size), dataset_pool_seed=dataset_seed,
                  dataset_specs_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None,
                  rng_policy=POLICY)
    return result


class TrainingRNG:
    def __init__(self, seed, rank, device, contract):
        import torch
        self.rank = int(rank)
        self.contract = contract
        self.timesteps = torch.Generator(device="cpu").manual_seed(stable_seed(POLICY, seed, rank, "timesteps"))
        self.noise = torch.Generator(device=device).manual_seed(stable_seed(POLICY, seed, rank, "noise"))

    def filename(self, directory):
        return Path(directory) / f"training_rng_{self.rank}.pt"

    def save(self, directory):
        import torch
        target = self.filename(directory)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        torch.save(dict(policy=POLICY, rank=self.rank, contract=self.contract,
                        timesteps=self.timesteps.get_state(), noise=self.noise.get_state()), temporary)
        os.replace(temporary, target)

    def load(self, directory, *, allow_legacy=False):
        import torch
        path = self.filename(directory)
        if not path.is_file():
            if allow_legacy:
                return False
            raise ValueError("Checkpoint lacks isolated diffusion RNG; explicit --allow_legacy_rng_resume is required (not exact reproduction)")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("policy") != POLICY or state.get("rank") != self.rank:
            raise ValueError("Incompatible checkpoint RNG policy/rank")
        differences = [key for key in set(self.contract) | set(state["contract"])
                       if self.contract.get(key) != state["contract"].get(key)]
        if differences:
            raise ValueError("Incompatible resume training contract: " + ", ".join(sorted(differences)))
        self.timesteps.set_state(state["timesteps"])
        self.noise.set_state(state["noise"])
        return True


def loader_generator(seed, source):
    import torch
    return torch.Generator(device="cpu").manual_seed(stable_seed(POLICY, seed, "loader", source))
