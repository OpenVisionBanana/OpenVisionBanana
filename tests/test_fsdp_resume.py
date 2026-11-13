import types
import unittest
from unittest.mock import patch

import accelerate.accelerator as module
from torch.distributed.fsdp import StateDictType
from common.training.fsdp_resume import load_training_state


class ResumeTests(unittest.TestCase):
    def test_non_fsdp_uses_native_loader(self):
        calls = []
        accelerator = types.SimpleNamespace(state=types.SimpleNamespace(fsdp_plugin=None),
                                            load_state=calls.append)
        load_training_state(accelerator, 'checkpoint')
        self.assertEqual(calls, ['checkpoint'])

    def test_scoped_patch_restored_even_on_failure(self):
        original = module.load_fsdp_optimizer
        plugin = types.SimpleNamespace(fsdp_version=1, state_dict_type=StateDictType.SHARDED_STATE_DICT)
        def fail(path):
            self.assertIsNot(module.load_fsdp_optimizer, original)
            raise RuntimeError('fixture failure')
        accelerator = types.SimpleNamespace(state=types.SimpleNamespace(fsdp_plugin=plugin), load_state=fail)
        with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
            load_training_state(accelerator, 'checkpoint')
        self.assertIs(module.load_fsdp_optimizer, original)


if __name__ == '__main__':
    unittest.main()
