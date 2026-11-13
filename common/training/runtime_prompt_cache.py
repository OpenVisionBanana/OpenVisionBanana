"""Bounded, process-local cache of exact frozen-encoder batch outputs.

Whole prompt batches are keys: padding and batch-dependent numerical behavior
are preserved. No caption is rewritten and no embeddings are stored on disk.
"""
from collections import OrderedDict


class RuntimePromptCache:
    def __init__(self, max_bytes=0, verify_first_hit=False, admit_after=1):
        self.max_bytes = max(0, int(max_bytes))
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0
        self.verify_first_hit = verify_first_hit
        self.verified = False
        self.admit_after = max(1, admit_after)
        self.seen = OrderedDict()

    def encode(self, prompts, encode, device):
        key = (prompts,) if isinstance(prompts, str) else tuple(prompts)
        if self.max_bytes and key in self.entries:
            self.hits += 1
            tensors = self.entries.pop(key)
            self.entries[key] = tensors
            # Return independent tensors; consumers cannot corrupt the cache.
            result = tuple(t.to(device=device, copy=True) for t in tensors)
            if self.verify_first_hit and not self.verified:
                import torch
                fresh = encode()
                for cached, actual in zip(result, fresh, strict=True):
                    torch.testing.assert_close(cached, actual, rtol=1e-4, atol=1e-4)
                self.verified = True
            return result
        self.misses += 1
        outputs = encode()
        if not self.max_bytes:
            return outputs
        count = self.seen.pop(key, 0) + 1
        self.seen[key] = count
        while len(self.seen) > 2048:
            self.seen.popitem(last=False)
        size = sum(t.numel() * t.element_size() for t in outputs)
        if self.max_bytes and size <= self.max_bytes and count >= self.admit_after:
            while self.entries and self.bytes + size > self.max_bytes:
                _, evicted = self.entries.popitem(last=False)
                self.bytes -= sum(t.numel() * t.element_size() for t in evicted)
            self.entries[key] = tuple(t.detach().to('cpu', copy=True) for t in outputs)
            self.bytes += size
        return outputs
