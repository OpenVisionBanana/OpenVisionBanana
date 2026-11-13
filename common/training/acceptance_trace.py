"""Opt-in, bounded numerical traces for short training acceptance jobs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def tensor_summary(tensor):
    # Bounded samples, not whole-model copies. Norms cover the whole tensor.
    import torch
    value = tensor.detach().float()
    flat = value.reshape(-1)
    return {'shape': list(value.shape), 'finite': bool(torch.isfinite(value).all()),
            'norm': float(torch.linalg.vector_norm(value)),
            'sample': flat[:16].cpu().tolist(),
            'sha256': hashlib.sha256(value.contiguous().cpu().numpy().tobytes()).hexdigest()}


def trace_update(path, rank, step, rows, prompts, tensors, gradients, weights, loss, grad_norm, metadata=None):
    import math
    record = {'rank': rank, 'step': step, 'rows': rows,
              'prompts_sha256': hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
              'loss': float(loss), 'gradient_norm': float(grad_norm),
              'tensors': {k: tensor_summary(v) for k, v in tensors.items()},
              'gradients': gradients, 'weights': weights, 'metadata': metadata or {}}
    if not math.isfinite(record['loss']) or not math.isfinite(record['gradient_norm']):
        raise ValueError('Acceptance: nonfinite loss or global gradient norm')
    if any(not v['finite'] for v in [*record['tensors'].values(), *gradients.values(), *weights.values()]):
        raise ValueError('Acceptance: nonfinite forward, gradient, or parameter sample')
    destination = Path(path)/f'rank_{rank}.jsonl'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False, sort_keys=True)+'\n')
