"""Dependency-free deterministic sampling helpers."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Sequence
from typing import TypeVar


T = TypeVar("T")


def limit_distinct_per_group(
    rows: Sequence[T],
    group_key: Callable[[T], str],
    distinct_key: Callable[[T], str],
    limit: int,
    seed: int,
) -> list[int]:
    """Return original-order indices for a seeded cap of distinct values per group."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    groups: dict[str, dict[str, tuple[bytes, int]]] = {}
    for ordinal, row in enumerate(rows):
        group = group_key(row)
        distinct = distinct_key(row)
        priority = hashlib.sha256(
            f"{seed}\0{group}\0{distinct}".encode("utf-8", errors="surrogatepass")
        ).digest()
        current = groups.setdefault(group, {}).get(distinct)
        candidate = (priority, ordinal)
        if current is None or candidate < current:
            groups[group][distinct] = candidate
    selected = {
        ordinal
        for values in groups.values()
        for _, ordinal in sorted(values.values())[:limit]
    }
    return [ordinal for ordinal in range(len(rows)) if ordinal in selected]


def sample_indices(
    rng: random.Random,
    population: Sequence[int],
    requested: int,
    strategy: str,
) -> list[int]:
    if not population and requested:
        raise ValueError("cannot sample from an empty population")
    if strategy == "with_replacement":
        return rng.choices(population, k=requested)
    if strategy != "without_replacement":
        raise ValueError(f"unsupported sampling strategy: {strategy}")
    if requested <= len(population):
        return rng.sample(population, requested)
    return [rng.choice(population) for _ in range(requested)]
