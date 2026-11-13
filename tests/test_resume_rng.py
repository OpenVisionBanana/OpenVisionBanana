from types import SimpleNamespace
import pytest
import torch

from common.training.resume_rng import TrainingRNG, augmentation_rng, loader_generator, resume_contract


def rng(rank=0, **changes):
    args = SimpleNamespace(seed=42, dataset_specs=None, gradient_accumulation_steps=1, **changes)
    return TrainingRNG(42, rank, "cpu", resume_contract(args, 2, 42))


def draw(state):
    return torch.rand(8, generator=state.timesteps), torch.randn(8, generator=state.noise)


def test_resume_ignores_loader_startup_and_skip_randomness(tmp_path):
    first = rng()
    draw(first)
    first.save(tmp_path)
    expected = draw(first)
    restarted = rng()
    assert restarted.load(tmp_path)
    # Both legacy loader startup and replayed rows can perturb global CPU RNG.
    iter(torch.utils.data.DataLoader([1, 2], num_workers=0))
    torch.rand(100)
    assert all(torch.equal(a, b) for a, b in zip(expected, draw(restarted)))


def test_loader_does_not_advance_global_rng():
    before = torch.get_rng_state().clone()
    list(torch.utils.data.DataLoader([1, 2], generator=loader_generator(42, "source")))
    assert torch.equal(before, torch.get_rng_state())


def test_rank_states_are_separate_and_every_rank_restores(tmp_path):
    a, b = rng(0), rng(1)
    assert not torch.equal(draw(a)[0], draw(b)[0])
    a.save(tmp_path); b.save(tmp_path)
    expected_a, expected_b = draw(a), draw(b)
    a, b = rng(0), rng(1)
    a.load(tmp_path); b.load(tmp_path)
    assert all(torch.equal(x, y) for x, y in zip(expected_a, draw(a)))
    assert all(torch.equal(x, y) for x, y in zip(expected_b, draw(b)))


@pytest.mark.parametrize("field,value", [("world_size", 4), ("seed", 43),
    ("gradient_accumulation_steps", 2), ("train_batch_size", 2), ("dataset_specs_sha256", "other")])
def test_incompatible_recipe_is_rejected_even_when_legacy_allowed(tmp_path, field, value):
    first = rng(); first.save(tmp_path)
    new = rng(); new.contract = {**new.contract, field: value}
    with pytest.raises(ValueError, match="Incompatible resume"):
        new.load(tmp_path, allow_legacy=True)


def test_legacy_migration_requires_explicit_opt_in(tmp_path):
    with pytest.raises(ValueError, match="allow_legacy_rng_resume"):
        rng().load(tmp_path)
    assert rng().load(tmp_path, allow_legacy=True) is False


def test_augmentation_replays_at_epoch_boundary_without_global_rng():
    before = torch.get_rng_state().clone()
    def draws(epoch, occurrence):
        local = augmentation_rng(42, epoch, "replica", "normals", "frame", occurrence)
        return [local.random() for _ in range(8)]
    first_epoch, next_epoch = draws(0, 0), draws(1, 0)
    assert first_epoch == draws(0, 0) and next_epoch == draws(1, 0)
    assert first_epoch != next_epoch and first_epoch != draws(0, 1)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("workers", [0, 2])
def test_small_loader_replay_across_epoch_boundary(tmp_path, workers):
    state = rng()
    state.save(tmp_path)
    expected = [draw(state) for _ in range(4)]
    restored = rng(); restored.load(tmp_path)
    actual = []
    for epoch in range(2):
        loader = torch.utils.data.DataLoader(list(range(2)), num_workers=workers,
                                             generator=loader_generator(42, "source"))
        for _ in loader:
            actual.append(draw(restored))
    assert all(torch.equal(x, y) for aa, bb in zip(expected, actual) for x, y in zip(aa, bb))
