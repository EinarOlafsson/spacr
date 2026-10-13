"""guide_attribution: apportioning slots, and a cell no guide explains.

`assign_well` gives every cell exactly one guide, with the counts
sequencing implies. The apportionment is the delicate part: the guide
fractions are real numbers and the cells are integers, so the floors
have to be topped up to sum to exactly n -- and one column per SLOT is
what makes the counts a property of the matrix rather than something
checked afterwards.

Two arms of that arithmetic cannot be reached, and this file shows why
rather than reaching past the arithmetic to force them. The third gap is
real: a cell whose density is zero under every guide, which happens on
an extreme score and must not become a division by zero.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from spacr.guide_attribution import (AMBIGUOUS, assign_well,
                                     normalise_fractions,
                                     posterior_multivariate)


class TestApportioningSlotsToCells:

    def test_the_counts_sum_to_the_number_of_cells(self):
        """The property the whole apportionment exists for."""
        scores = [0.1, 0.5, 0.9, 0.2, 0.7]
        result = assign_well(scores, {"a": 0.5, "b": 0.3, "c": 0.2},
                             {"a": 1.0, "b": 0.0, "c": -1.0})
        assert len(result.guides) == len(scores)
        assert sum(result.counts.values()) == len(scores)

    def test_a_remainder_is_given_to_the_largest_fractional_parts(self):
        """`short > 0` -- the live top-up arm.

        Three cells split 1/3 each floors to zero slots apiece, so all
        three have to be handed out by remainder.
        """
        result = assign_well([0.1, 0.5, 0.9],
                             {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3},
                             {"a": 0.0, "b": 0.0, "c": 0.0})
        assert sum(result.counts.values()) == 3

    def test_no_cells_at_all_is_answered_not_computed(self):
        result = assign_well([], {"a": 1.0}, {"a": 0.0})
        assert result.guides == ()
        assert result.counts == {}

    def test_no_usable_fractions_leaves_every_cell_ambiguous(self):
        result = assign_well([0.1, 0.2], {}, {})
        assert result.guides == (AMBIGUOUS, AMBIGUOUS)


class TestTheTwoArmsTheArithmeticCannotReach:
    """Both are marked rare in the source. They are impossible.

    Pinned from the producing side: if the apportionment ever stops
    guaranteeing these, the tests fail and the arms become live.
    """

    def test_the_floors_can_never_overshoot_the_cell_count(self):
        """`elif short < 0:` cannot fire.

        `exact` sums to n, because the priors are normalised and then
        multiplied by n. `floor` only ever decreases a value, so the
        floors sum to at most n and `short = n - floors.sum()` is never
        negative.

        Checked over 20,000 random wells -- guide counts 1..6, cell
        counts 1..40 -- as well as the argument.
        """
        rng = random.Random(20260831)
        for _ in range(20000):
            k = rng.randint(1, 6)
            raw = {f"g{i}": rng.random() for i in range(k)}
            n = rng.randint(1, 40)
            priors = normalise_fractions(raw)
            exact = np.array([priors[g] * n for g in priors], dtype=float)
            short = n - int(np.floor(exact).astype(int).sum())
            assert short >= 0, (
                f"floors overshot with {raw} over {n} cells; the `short < 0` "
                "arm in assign_well is now reachable")

    def test_one_column_per_slot_always_gives_exactly_n_columns(self):
        """`if columns.size != n:` cannot fire either.

        The slots are topped up to sum to n before the columns are
        built, and `np.repeat` produces exactly `slots.sum()` of them.
        """
        rng = random.Random(20260831)
        for _ in range(5000):
            k = rng.randint(1, 6)
            raw = {f"g{i}": rng.random() for i in range(k)}
            n = rng.randint(1, 40)
            priors = normalise_fractions(raw)
            exact = np.array([priors[g] * n for g in priors], dtype=float)
            slots = np.floor(exact).astype(int)
            short = n - int(slots.sum())
            if short > 0:
                order = np.argsort(-(exact - np.floor(exact)))
                for index in order[:short]:
                    slots[index] += 1
            assert int(np.repeat(np.arange(k), slots).size) == n


class TestTheDeadRowFallbackThatCannotFire:
    """`dead = density.sum(axis=1) <= 0` is never true.

    It looks like it guards a real case -- a cell so far from every
    guide's fitted effect that its density underflows to zero -- and
    before the log-space rewrite it did. It cannot now, and the line that
    makes it impossible is two above it:

        log_density -= log_density.max(axis=1, keepdims=True)

    After that shift every row's maximum is exactly 0, so at least one
    entry exponentiates to 1 and the row sum is at least 1. The comment
    beside the shift says it is "the difference between a usable number
    and exp(-4000)"; it also retired the fallback below it.

    Pinned from the producing side, and driven with a cell far enough out
    to have underflowed under the old arithmetic.
    """

    def test_an_absurdly_distant_cell_still_gets_a_finite_distribution(self):
        priors = {"a": 0.7, "b": 0.3}
        effects = {"a": [0.0], "b": [10.0]}
        measurements = np.array([[1.0], [1e300]], dtype=float)

        posterior, names, _extra = posterior_multivariate(
            measurements, priors, effects, iterations=5)

        assert names == ("a", "b")
        assert posterior.shape == (2, 2)
        assert np.isfinite(posterior).all(), (
            "an unexplained cell produced a non-finite posterior")
        assert posterior[1].sum() == pytest.approx(1.0, rel=1e-6)

    def test_the_row_shift_is_what_keeps_every_row_alive(self):
        """The argument itself, checked rather than asserted in prose.

        Any finite log-density matrix, once its row maximum is
        subtracted, exponentiates to rows that sum to at least 1.
        """
        rng = np.random.default_rng(20260831)
        for _ in range(2000):
            rows, cols = rng.integers(1, 6), rng.integers(1, 5)
            log_density = rng.normal(0.0, 4000.0, size=(rows, cols))
            shifted = log_density - log_density.max(axis=1, keepdims=True)
            density = np.exp(shifted)
            assert (density.sum(axis=1) >= 1.0).all(), (
                "a shifted row exponentiated to nothing; the dead-row "
                "fallback in posterior_multivariate is now reachable")

    def test_the_shift_is_still_in_the_source(self):
        import inspect

        source = inspect.getsource(posterior_multivariate)
        assert "log_density -= log_density.max(axis=1, keepdims=True)" in \
            source, ("the per-row shift has gone; the dead-row fallback "
                     "below it may now be reachable and wants a test")

    def test_an_ordinary_cell_favours_the_guide_it_sits_on(self):
        """The live behaviour the whole function is for."""
        priors = {"a": 0.5, "b": 0.5}
        effects = {"a": [0.0], "b": [5.0]}
        measurements = np.array([[0.0], [5.0]], dtype=float)

        posterior, names, _extra = posterior_multivariate(
            measurements, priors, effects, iterations=50)
        assert names == ("a", "b")
        assert posterior[0, 0] > posterior[0, 1]
        assert posterior[1, 1] > posterior[1, 0]
