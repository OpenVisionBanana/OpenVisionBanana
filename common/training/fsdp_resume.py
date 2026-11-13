"""Restore FSDP1 optimizer moments from checkpoint metadata, not an empty template."""
from pathlib import Path


def load_training_state(accelerator, checkpoint):
    import accelerate.accelerator as accelerator_module
    from torch.distributed.fsdp import StateDictType

    plugin = getattr(accelerator.state, 'fsdp_plugin', None)
    if (plugin is None or getattr(plugin, 'fsdp_version', 1) != 1
            or plugin.state_dict_type != StateDictType.SHARDED_STATE_DICT):
        accelerator.load_state(checkpoint)
        return

    original = accelerator_module.load_fsdp_optimizer

    def load_optimizer(fsdp_plugin, owner, optimizer, model, input_dir,
                       optimizer_index=0, adapter_only=False):
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        # Accelerate's empty optimizer.state_dict() template does not contain
        # Adam's lazily initialized moments. Allocate them from saved metadata.
        reader = dcp.FileSystemReader(Path(input_dir)/f'optimizer_{optimizer_index}')
        owner.wait_for_everyone()
        with FSDP.state_dict_type(model, fsdp_plugin.state_dict_type,
                                 fsdp_plugin.state_dict_config,
                                 fsdp_plugin.optim_state_dict_config):
            restored = load_sharded_optimizer_state_dict(
                model.state_dict(), optimizer_key='optimizer', storage_reader=reader)
            state = restored['optimizer']
            if not state.get('state'):
                raise ValueError('Checkpoint contains no optimizer moments; refusing an optimizer reset')
            optimizer.load_state_dict(FSDP.optim_state_dict_to_load(
                model=model, optim=optimizer, optim_state_dict=state))

    # Scope the compatibility workaround to this synchronous restore only.
    accelerator_module.load_fsdp_optimizer = load_optimizer
    try:
        accelerator.load_state(checkpoint)
    finally:
        accelerator_module.load_fsdp_optimizer = original
