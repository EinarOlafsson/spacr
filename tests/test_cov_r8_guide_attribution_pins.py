"""Three guards in guide_attribution that the arithmetic above them shuts.

All three are one line between an assignment problem and a wrong answer
rather than a crash, which is why each is worth keeping and why each pin
runs the arithmetic instead of quoting it.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from spacr import guide_attribution as G


def _fractions(**guides):
    return dict(guides)


class TestHandingOutTheSlots:
    """Each guide gets ``prior * n`` slots, summing to ``n`` exactly."""

    def test_the_counts_match_the_fractions(self):
        scores = np.linspace(-2.0, 2.0, 10)
        result = G.assign_well(scores, _fractions(a=0.5, b=0.5), {})

        assert sum(result.counts.values()) == 10
        assert set(result.counts) <= {"a", "b"}

    def test_an_uneven_split_still_sums_to_the_cell_count(self):
        scores = np.linspace(-2.0, 2.0, 7)
        result = G.assign_well(scores, _fractions(a=1 / 3, b=1 / 3, c=1 / 3),
                               {})

        assert sum(result.counts.values()) == 7

    def test_flooring_can_never_overshoot_the_cell_count(self):
        """THE PIN, for ``elif short < 0``.

        ``exact`` is ``prior * n`` over normalised priors, so it sums to
        exactly ``n``; ``floor`` only ever moves a value DOWN, so the
        floored sum is at most ``n`` and ``short`` is never negative.
        The give-back loop below it cannot run.

        It is right to keep -- handing out more slots than cells makes
        the assignment matrix wider than it is tall, and
        ``linear_sum_assignment`` then leaves cells unassigned with no
        complaint. Run over a spread of awkward splits rather than
        argued, because "the priors are normalised" is the premise and
        floating point is where premises like that fail.
        """
        rng = np.random.default_rng(11)
        for n in (1, 2, 3, 7, 10, 97, 1000):
            for k in (1, 2, 3, 5, 9):
                raw = rng.random(k) + 1e-9
                priors = G.normalise_fractions(
                    {f"g{i}": float(v) for i, v in enumerate(raw)})
                exact = np.array([priors[g] * n for g in priors], dtype=float)
                floored = int(np.floor(exact).sum())
                assert floored <= n, (
                    f"floor overshot for n={n}, k={k}: {floored} > {n}")

        source = inspect.getsource(G.assign_well)
        assert "priors = normalise_fractions(fractions)" in source, (
            "the priors are no longer normalised before they are scaled, "
            "so exact need not sum to n and the give-back loop is live")

    def test_the_slot_columns_always_number_the_cells(self):
        """THE PIN, for ``if columns.size != n``.

        ``columns`` is ``repeat(arange(k), slots)``, whose size is
        ``slots.sum()`` -- and the top-up above has just made that
        exactly ``n``. So the truncation below cannot run.

        Also worth keeping: truncating silently would drop the LAST
        guide's slots, which is a bias toward whatever the name order
        happens to be.
        """
        source = inspect.getsource(G.assign_well)
        assert "columns = np.repeat(np.arange(len(names)), slots)" in source
        assert "if short > 0:" in source and "slots[index] += 1" in source

        rng = np.random.default_rng(5)
        for n in (1, 4, 9, 33):
            raw = rng.random(4) + 1e-9
            priors = G.normalise_fractions(
                {f"g{i}": float(v) for i, v in enumerate(raw)})
            exact = np.array([priors[g] * n for g in priors], dtype=float)
            slots = np.floor(exact).astype(int)
            short = n - int(slots.sum())
            if short > 0:
                for index in np.argsort(-(exact - np.floor(exact)))[:short]:
                    slots[index] += 1
            columns = np.repeat(np.arange(len(priors)), slots)
            assert columns.size == n, (
                f"the top-up left {columns.size} columns for {n} cells")

    def test_a_well_with_no_cells_or_no_guides_is_answered_not_computed(self):
        assert G.assign_well([], _fractions(a=1.0), {}).guides == ()
        empty = G.assign_well([0.1, 0.2], {}, {})
        assert set(empty.guides) == {G.AMBIGUOUS}
        assert empty.cost == float("inf")


class TestTheMultivariatePosterior:

    def _inputs(self, n_cells=12, n_guides=3, seed=2):
        rng = np.random.default_rng(seed)
        measurements = np.exp(rng.normal(size=(n_cells, 2)))
        guides = tuple(f"g{i}" for i in range(n_guides))
        priors = {g: 1.0 / n_guides for g in guides}
        effects = {g: [0.2 * i, -0.1 * i] for i, g in enumerate(guides)}
        return measurements, priors, effects

    def test_every_cell_gets_a_distribution_that_sums_to_one(self):
        measurements, priors, effects = self._inputs()

        posterior, names, _scale = G.posterior_multivariate(
            measurements, priors, effects)

        assert posterior.shape == (12, 3)
        assert tuple(names) == tuple(priors)
        assert np.allclose(posterior.sum(axis=1), 1.0)

    def test_no_row_can_have_zero_density_under_every_guide(self):
        """THE PIN, for ``if dead.any()``.

        The log densities are shifted so the largest in each row is
        zero, and ``exp(0)`` is 1 -- so every row holds at least one 1.0
        and its sum is at least 1. The fallback to the priors cannot
        run.

        It is right to keep: a row of zeros would divide by zero in the
        normalisation below and put NaN into an assignment table that
        then reads as "no guide" for a cell that has one. Run over
        deliberately hostile inputs -- a cell a thousand sigma away from
        every centre, and one at every centre at once.
        """
        measurements, priors, effects = self._inputs()
        measurements[0] = 1e6                    # far outside every centre
        measurements[1] = 1e-6                   # far below every centre

        posterior, _names, _scale = G.posterior_multivariate(
            measurements, priors, effects)

        assert np.isfinite(posterior).all(), (
            "a NaN reached the posterior, which is what the dead-row "
            "fallback exists to prevent")
        assert np.allclose(posterior.sum(axis=1), 1.0)

        source = inspect.getsource(G.posterior_multivariate)
        assert "log_density -= log_density.max(axis=1, keepdims=True)" \
            in source, (
            "the per-row shift is gone, so exp() can now underflow every "
            "entry of a row to zero and the dead-row fallback is live")

        shifted = np.array([[-1e6, -2e6], [0.0, -3.0]])
        shifted = shifted - shifted.max(axis=1, keepdims=True)
        assert (np.exp(shifted).sum(axis=1) > 0).all(), (
            "a shifted row underflowed to zero everywhere")
