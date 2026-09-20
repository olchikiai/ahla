"""Seeded, reproducible train/val dataset split for the ``[train]`` tier.

Pure and framework-free. Generic over the element type (no ``Sample`` import) to
avoid a circular import. The val-fraction range is enforced by
``validate_config``, so it is not re-checked here.
"""

from __future__ import annotations

import random
from typing import TypeVar

__all__ = ["partition"]

# Generic element type: partitioner never inspects an element.
T = TypeVar("T")


def partition(
    samples: list[T], val_fraction: float, seed: int
) -> tuple[list[T], list[T]]:
    """Split ``samples`` into ``(training, validation)``.

    Shuffles a copy with ``random.Random(seed)`` (input never mutated) and takes
    the first ``round(len(samples) * val_fraction)`` elements as the
    Validation_Set. The seeded shuffle makes the split reproducible; val_fraction
    0.0 leaves validation empty.
    """
    # Shuffle a copy so the caller's list is never mutated.
    shuffled = list(samples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    # round() (banker's rounding); round(0.0) == 0 -> empty validation.
    validation_size = round(len(shuffled) * val_fraction)

    validation = shuffled[:validation_size]
    training = shuffled[validation_size:]
    return training, validation
