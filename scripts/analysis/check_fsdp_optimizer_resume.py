#!/usr/bin/env python3
"""Small real-FSDP restart check; no model downloads or training data required."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from common.training.fsdp_resume import load_training_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plugin = FullyShardedDataParallelPlugin(
        fsdp_version=1, state_dict_type='SHARDED_STATE_DICT',
        use_orig_params=True, sync_module_states=True)
    accelerator = Accelerator(fsdp_plugin=plugin)
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 16))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    model, optimizer = accelerator.prepare(model, optimizer)
    x = torch.arange(128, device=accelerator.device, dtype=torch.float32).reshape(4, 32)/128
    def update():
        loss = model(x).square().mean()
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        return float(loss.detach())
    update()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = str(args.output/'checkpoint')
    accelerator.save_state(checkpoint)
    expected_loss = update()
    expected = [p.detach().clone() for p in model.parameters()]
    # Reproduce the fresh-process empty optimizer template explicitly.
    optimizer.state.clear()
    load_training_state(accelerator, checkpoint)
    restored_states = len(optimizer.state)
    actual_loss = update()
    difference = max(float((p.detach()-v).abs().max()) if p.numel() else 0.
                     for p, v in zip(model.parameters(), expected))
    difference_tensor = torch.tensor(difference, device=accelerator.device)
    torch.distributed.all_reduce(difference_tensor, op=torch.distributed.ReduceOp.MAX)
    passed = restored_states > 0 and float(difference_tensor) <= 1e-7 and abs(expected_loss-actual_loss) <= 1e-7
    report = {'status': 'PASS' if passed else 'FAIL', 'rank': accelerator.process_index,
              'restored_optimizer_states': restored_states, 'max_weight_difference': float(difference_tensor),
              'expected_loss': expected_loss, 'resumed_loss': actual_loss}
    (args.output/f'rank_{accelerator.process_index}.json').write_text(json.dumps(report, indent=2)+'\n')
    accelerator.wait_for_everyone()
    if not passed:
        raise RuntimeError(report)
    accelerator.print(json.dumps(report))


if __name__ == '__main__':
    main()
