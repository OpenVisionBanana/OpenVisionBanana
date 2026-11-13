import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class PackedPromptCacheTest(unittest.TestCase):
    def test_distributed_writer_finalize_and_duplicate_batch_lookup(self):
        from common.training.packed_prompt_cache import (
            PackedPromptCache,
            PackedPromptCacheWriter,
            finalize_packed_prompt_cache,
            prompt_cache_fingerprint,
            prompt_key,
        )

        prompts = ["alpha", "beta", "gamma"]
        fingerprint = prompt_cache_fingerprint(prompts, "model", 512)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for writer_id, writer_prompts in enumerate((prompts[::2], prompts[1::2])):
                writer = PackedPromptCacheWriter(root, fingerprint, writer_id, shard_bytes=1)
                for index, prompt in enumerate(writer_prompts):
                    writer.add(
                        prompt_key(prompt),
                        torch.full((2, 3), float(index + writer_id)),
                        torch.full((2, 3), index + writer_id, dtype=torch.int64),
                    )
                writer.close()
            finalize_packed_prompt_cache(root, fingerprint, 2, map(prompt_key, prompts))
            cache = PackedPromptCache(root, fingerprint)
            embeds, ids = cache.load_batch(["beta", "alpha", "beta"])
            self.assertEqual(tuple(embeds.shape), (3, 2, 3))
            self.assertTrue(torch.equal(embeds[0], embeds[2]))
            self.assertTrue(torch.equal(ids[0], ids[2]))


if __name__ == "__main__":
    unittest.main()
