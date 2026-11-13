"""Correct masked-loss normalization when ranks have unequal valid counts."""
import torch


def normalized_distributed_loss(numerator, denominator):
    count=denominator.detach().to(device=numerator.device,dtype=torch.float32).clone()
    ranks=1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        ranks=torch.distributed.get_world_size()
        torch.distributed.all_reduce(count,op=torch.distributed.ReduceOp.SUM)
    # DDP/FSDP average gradients across ranks. Compensate for that averaging
    # after reducing the valid denominator, without all-reducing autograd data.
    return numerator*ranks/count.clamp_min(1.)
