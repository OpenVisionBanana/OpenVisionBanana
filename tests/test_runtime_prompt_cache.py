import torch
from common.training.runtime_prompt_cache import RuntimePromptCache


def test_exact_batch_cache_is_bounded_and_returns_independent_tensors():
    calls=[]
    def encode():
        calls.append(1)
        return torch.arange(12,dtype=torch.float32).view(2,6), torch.arange(6).view(2,3)
    cache=RuntimePromptCache(128,verify_first_hit=True)
    original=cache.encode(['a','a'],encode,'cpu')
    hit=cache.encode(['a','a'],encode,'cpu')
    assert cache.verified and len(calls)==2
    for a,b in zip(original,hit): assert torch.equal(a,b)
    hit[0].zero_()
    recovered=cache.encode(['a','a'],encode,'cpu')
    assert torch.equal(recovered[0],original[0]) and len(calls)==2
    cache.encode(['b','b'],encode,'cpu')
    assert cache.bytes<=128 and len(cache.entries)==1


def test_cache_preserves_batch_shape_order_and_does_not_consume_rng_on_hit():
    cache=RuntimePromptCache(1024)
    calls=[]
    def encode(): calls.append(1);return (torch.ones(2,3),torch.zeros(2,3))
    cache.encode(['a','b'],encode,'cpu')
    before=torch.get_rng_state().clone()
    cache.encode(['a','b'],encode,'cpu')
    assert torch.equal(before,torch.get_rng_state())
    cache.encode(['b','a'],encode,'cpu')
    cache.encode(['a'],encode,'cpu')
    assert len(calls)==3


def test_disabled_or_oversized_cache_never_accumulates_entries():
    def encode(): return (torch.ones(1024),torch.ones(3))
    for budget in (0,8):
        cache=RuntimePromptCache(budget)
        cache.encode(['a'],encode,'cpu');cache.encode(['a'],encode,'cpu')
        assert not cache.entries and cache.bytes==0 and cache.misses==2


def test_repeated_prompt_admission_keeps_one_off_captions_out():
    cache=RuntimePromptCache(1024,admit_after=2)
    def encode(): return (torch.ones(2,3),torch.zeros(2,3))
    cache.encode(['task','task'],encode,'cpu')
    assert not cache.entries
    cache.encode(['task','task'],encode,'cpu')
    assert len(cache.entries)==1
    for i in range(10):cache.encode([f'caption{i}'],encode,'cpu')
    assert list(cache.entries)==[('task','task')]
