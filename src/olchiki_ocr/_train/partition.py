"""Dataset partitioner for the ``[train]`` tier.

Splits a Dataset into a Training_Set and a Validation_Set. Each element is
assigned to exactly one partition; the two partitions are disjoint and their
union equals the input list. The configured fraction of elements is assigned to
the Validation_Set, a fraction of 0.0 leaves the Validation_Set empty and puts
every element in the Training_Set, and a dedicated seeded RNG makes the
partition identical across runs for identical inputs and seed.

This module is *pure* and framework-free: it depends only on the standard
library. It is deliberately **generic over its element type** rather than
importing the ``Sample`` dataclass (which is defined in ``synthgen`` and is not
required for the split logic). This avoids a forward/circular import while
letting the partitioner operate on any list of Samples - or any homogeneous
list - unchanged. The caller records the resulting Training_Set and
Validation_Set counts in ``RunStats``; this function returns the two lists.

The validation-fraction range (0.0 to 1.0 inclusive) is enforced up front by
``validate_config``, so it is not re-validated here; the function behaves
correctly across that whole range regardless.
"""

from __future__ import annotations

import random
from typing import TypeVar

__all__ = ["partition"]

# Generic element type: the partitioner never inspects an element, so it works
# on any homogeneous list (e.g. list[Sample]) without importing Sample.
T = TypeVar("T")


def partition(
    samples: list[T], val_fraction: float, seed: int
) -> tuple[list[T], list[T]]:
    """Split ``samples`` into ``(training, validation)`` partitions.

    A copy of ``samples`` is shuffled with a dedicated ``random.Random(seed)``
    so the caller's list is never mutated and the shuffle order is fully
    determined by ``seed`` (giving an identical partition across runs for
    identical inputs and seed). The first
    ``round(len(samples) * val_fraction)`` elements of the shuffled copy become
    the Validation_Set and the remainder become the Training_Set.

    Because the result is a partition of a single shuffled copy, every element
    lands in exactly one partition, the two partitions are disjoint, and their
    union equals ``samples``. Python's ``round`` (banker's rounding) is used so
    the Validation_Set size equals ``round(n * val_fraction)`` exactly. When
    ``val_fraction`` is 0.0 the validation size is ``round(0.0) == 0``, so every
    element goes to the Training_Set and the Validation_Set is empty.

    Args:
        samples: The Dataset to split. Not mutated.
        val_fraction: Proportion of elements to place in the Validation_Set,
            expected in the range 0.0 to 1.0 inclusive (enforced earlier by
            ``validate_config``).
        seed: Seed for the dedicated RNG driving the shuffle, making the
            partition reproducible.

    Returns:
        A ``(training, validation)`` tuple of lists whose combined length equals
        ``len(samples)``.
    """
    # Shuffle a copy so the caller's list is never mutated.
    shuffled = list(samples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    # Banker's rounding via round() so the size matches round(n * fraction)
    # exactly. round(0.0) == 0 -> empty validation.
    validation_size = round(len(shuffled) * val_fraction)

    validation = shuffled[:validation_size]
    training = shuffled[validation_size:]
    return training, validation
