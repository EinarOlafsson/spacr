"""Two edges in regression QC: a fit with no quartiles, and an empty SVD.

Both sit where a statistic would otherwise be quoted off data that
cannot support it -- which in a QC report is worse than no statistic,
because the reader takes the number as the answer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from matplotlib.figure import Figure

from spacr import regression_qc as rq


def _axes():
    """A bare axes on an object-oriented figure, as the report driver makes."""
    return Figure(figsize=(4.6, 3.6)).add_subplot(1, 1, 1)


class _Fit:
    def __init__(self, **attributes):
        self.__dict__.update(attributes)


def _context(fitted, y, X=None, **kwargs):
    response = np.asarray(y, dtype=float)
    n = response.size
    if X is None:
        X = pd.DataFrame({"intercept": np.ones(n),
                          "x": np.linspace(0.0, 1.0, n)})
    return rq.build_context(_Fit(fittedvalues=np.asarray(fitted, dtype=float)),
                            X, response, **kwargs)


# ---------------------------------------------------------------------------
# _panel_scale_location -- a fit whose quartiles collapse to two edges
# ---------------------------------------------------------------------------

class TestTheBrownForsytheQuartiles:

    def test_a_real_spread_of_predictions_gets_both_statistics(self):
        n = 40
        fitted = np.linspace(0.0, 10.0, n)
        rng = np.random.default_rng(3)
        ctx = _context(fitted, fitted + rng.normal(size=n))

        stats = rq._panel_scale_location(ctx, _axes())

        assert np.isfinite(stats["spearman_rho"])
        assert np.isfinite(stats["levene_p"])
        assert np.isfinite(stats["quartile_sd_ratio"])

    def test_a_fit_with_only_two_distinct_quartile_edges_falls_back(self):
        """THE UNCOVERED ARC: fewer than three edges is fewer than two
        buckets.

        Three edges are two buckets, which is the minimum a
        Brown-Forsythe test can compare. A fit that predicts almost one
        value -- a model whose only live predictor is nearly constant --
        puts every quantile but the last at the same place, so the
        bucketing would produce a single group and
        ``np.searchsorted`` would then clip every row into it.

        The panel prints ``nan`` and falls back to Spearman, because a
        p-value quoted off one group reads as evidence the variance is
        fine.
        """
        n = 24
        fitted = np.array([0.0] * (n - 1) + [10.0])
        rng = np.random.default_rng(5)
        ctx = _context(fitted, fitted + rng.normal(size=n))

        edges = np.unique(np.quantile(fitted, [0.0, 0.25, 0.5, 0.75, 1.0]))
        assert edges.size == 2, "the fixture no longer collapses the quartiles"
        assert np.ptp(fitted) > 0, "the earlier guard would have caught this"

        stats = rq._panel_scale_location(ctx, _axes())

        assert np.isnan(stats["levene_p"])
        assert np.isnan(stats["quartile_sd_ratio"])
        assert np.isfinite(stats["spearman_rho"]) or np.isnan(
            stats["spearman_rho"])
        assert stats["n_points"] == n

    def test_a_fit_that_predicts_one_value_never_reaches_the_quartiles(self):
        """The guard above it: no spread at all, so no buckets to make."""
        n = 24
        fitted = np.full(n, 4.0)
        rng = np.random.default_rng(6)
        ctx = _context(fitted, fitted + rng.normal(size=n))

        stats = rq._panel_scale_location(ctx, _axes())

        assert np.isnan(stats["levene_p"])
        assert np.isnan(stats["quartile_sd_ratio"])


# ---------------------------------------------------------------------------
# condition_number -- there is always at least one singular value
# ---------------------------------------------------------------------------

class TestTheConditionNumbersRatio:

    def test_an_orthogonal_design_is_perfectly_conditioned(self):
        scaled, unscaled, singular = rq.condition_number(np.eye(3))

        assert round(scaled, 6) == 1.0
        assert round(unscaled, 6) == 1.0
        assert singular.size == 3

    def test_a_duplicated_predictor_is_singular_on_any_runner(self):
        """Why the ratio uses a numerical-rank tolerance rather than == 0."""
        X = np.column_stack([np.ones(20), np.linspace(0, 1, 20)])
        X = np.column_stack([X, X[:, 1]])          # an exact duplicate

        scaled, _unscaled, singular = rq.condition_number(X)

        assert not np.isfinite(scaled) or scaled > 1e8, (
            "a duplicated predictor was not reported as singular")
        assert singular.size == 3

    def test_an_empty_design_matrix_is_refused_before_any_svd(self):
        """THE PIN.

        ``if sv.size == 0`` inside the ratio can never fire: the only way
        an SVD returns no singular values is a matrix with a zero
        dimension, and the guard at the top of the function raises for
        that first -- with a message naming the shape, which is the one
        a caller can act on.

        The inner guard is still right to keep: it returns ``inf``, and
        ``inf`` is the honest condition number of a matrix with no
        columns. But it is the outer refusal that has to hold, and this
        is what fails if it stops holding.
        """
        for bad in (np.empty((0, 3)), np.empty((5, 0)), np.empty((0, 0))):
            with pytest.raises(ValueError, match="non-empty 2-D array"):
                rq.condition_number(bad)

        with pytest.raises(ValueError, match="non-empty 2-D array"):
            rq.condition_number(np.arange(5.0))     # 1-D

        for shape in ((1, 1), (2, 1), (1, 2), (7, 3)):
            _scaled, _unscaled, singular = rq.condition_number(
                np.ones(shape) + np.arange(shape[1], dtype=float))
            assert singular.size == min(shape), (
                f"an SVD of {shape} returned {singular.size} singular values")
            assert singular.size >= 1

    def test_an_all_zero_column_leaves_the_matrix_singular(self):
        """The scaling deliberately does not rescue a zero column: a
        column with no direction is exactly what makes the matrix
        singular, and the singular value of 0 that follows is the
        honest answer."""
        X = np.column_stack([np.ones(10), np.zeros(10),
                             np.linspace(0, 1, 10)])

        scaled, _unscaled, singular = rq.condition_number(X)

        assert singular.size == 3
        assert float(singular[-1]) == pytest.approx(0.0, abs=1e-12)
        assert not np.isfinite(scaled) or scaled > 1e8
