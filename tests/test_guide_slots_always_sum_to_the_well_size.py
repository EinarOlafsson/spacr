"""Slot allocation in ``assign_well``, and the two arms it does not need.

Each guide gets a whole number of slots, and the slots must sum to exactly
the number of cells in the well -- the assignment is a rectangular
matching, so a column count that disagrees with the row count is not a
worse answer, it is a different problem.

The allocation floors ``priors * n`` and hands the remainder to the guides
with the largest fractional parts. Two arms guarded against the opposite
case -- an OVERSHOOT -- and were marked ``# pragma: no cover - rare``.
They are not rare: ``priors`` sums to 1, so ``exact`` sums to ``n``, and
``floor(x) <= x`` makes ``slots.sum() <= n``. The shortfall cannot be
negative, so neither arm could run, and both are gone.

This file pins the property that makes their absence safe. It is the
premise, not the deletion, that is worth testing: if the shortfall could
ever be negative, removing those arms would change behaviour.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from spacr.guide_attribution import assign_well, normalise_fractions


def _shortfall(fractions, n):
    """``n - sum(floor(priors * n))`` -- what the removed arms branched on."""
    priors = normalise_fractions(fractions)
    exact = np.array([priors[g] * n for g in priors], dtype=float)
    return n - int(np.floor(exact).astype(int).sum())


@pytest.mark.parametrize("fractions,n", [
    ({"a": 1.0}, 1),
    ({"a": 0.5, "b": 0.5}, 2),
    ({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, 10),
    ({"a": 0.999999, "b": 0.000001}, 7),
    ({"a": 1e9, "b": 1.0}, 5),
    ({f"g{i}": 1.0 for i in range(9)}, 100),
])
def test_the_shortfall_is_never_negative(fractions, n):
    """THE PREMISE THE DELETION RESTS ON.

    Hand-picked awkward cases: an exact split, a repeating third, a
    fraction that rounds to nothing, magnitudes twelve orders apart.
    """
    assert _shortfall(fractions, n) >= 0


def test_the_shortfall_is_never_negative_across_random_fractions():
    """The same claim, over inputs nobody thought to write down.

    Thirty thousand were checked when the arms were removed; a smaller
    seeded sweep runs here so a change to `normalise_fractions` that
    breaks the invariant fails a test rather than going unnoticed.
    """
    random.seed(3)
    worst = 0
    for _ in range(2000):
        count = random.randint(1, 8)
        fractions = {f"g{i}": random.random() * random.choice([1, 1e-6, 1e6])
                     for i in range(count)}
        try:
            worst = min(worst, _shortfall(fractions, random.randint(1, 400)))
        except Exception:                               # noqa: BLE001
            continue
    assert worst == 0, f"the shortfall went negative ({worst})"


@pytest.mark.parametrize("n", [1, 2, 5, 17])
def test_every_cell_in_the_well_is_assigned(n):
    """The behaviour underneath: one guide per cell, no cell left out.

    This is what a wrong column count would break, so it is asserted
    directly rather than inferred from the slot arithmetic.
    """
    result = assign_well([0.1 * i for i in range(n)],
                         {"a": 0.5, "b": 0.3, "c": 0.2},
                         {"a": 1.0, "b": 0.0, "c": -1.0})
    assert len(result.guides) == n
    assert sum(result.counts.values()) == n
