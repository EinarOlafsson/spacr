"""Two numeric guards: a cell with no evidence, and a design with no
columns.

Both look like they protect a real case and neither can fire, for
reasons that live a few lines above them. Driven where the surrounding
behaviour can be, pinned where the arm cannot.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from spacr import guide_attribution as G


def _priors(*guides):
    share = 1.0 / len(guides)
    return {guide: share for guide in guides}


class TestEveryCellKeepsSomeEvidence:

    def test_a_two_guide_posterior_sums_to_one_per_cell(self):
        """The behaviour the guard sits inside."""
        measurements = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 2.0]])
        priors = _priors("g1", "g2")
        effects = {"g1": [1.0, 0.0], "g2": [0.0, 1.0]}

        r, guides, report = G.posterior_multivariate(
            measurements, priors, effects)

        assert guides == ("g1", "g2")
        assert r.shape == (3, 2)
        assert np.allclose(r.sum(axis=1), 1.0)
        assert report["n_measurements"] == 2.0

    def test_the_shift_leaves_one_exact_zero_in_every_row(self):
        """THE PIN, for ``if dead.any()``.

        The per-cell maximum is subtracted from the log densities before
        they are exponentiated -- the shift cancels in the
        normalisation, and is the difference between a usable number and
        exp(-4000). It also means every row keeps one entry at exactly
        zero, so ``exp`` gives it a 1 and the row sum is at least 1.

        The densities themselves are floored at 1e-300 before the log,
        so no entry is -inf and the maximum is always finite. A row that
        sums to zero therefore cannot exist, and the fallback to the
        prior below cannot run.
        """
        source = inspect.getsource(G.posterior_multivariate)
        floor = source.index("np.log(np.maximum(density, 1e-300))")
        shift = source.index("log_density -= log_density.max(", floor)
        dead = source.index("dead = density.sum(axis=1) <= 0", shift)

        assert floor < shift < dead, (
            "the maximum-shift no longer sits between the density floor "
            "and the dead-row check, so a row of zeros is possible again")

        # And the arithmetic itself, over the shapes the function meets.
        for rows, columns in ((1, 1), (3, 2), (5, 4)):
            log_density = np.random.default_rng(0).normal(
                size=(rows, columns)) * 500.0
            shifted = log_density - log_density.max(axis=1, keepdims=True)
            assert np.all(np.exp(shifted).sum(axis=1) >= 1.0)

    def test_a_cell_whose_every_measurement_is_missing_still_gets_a_row(self):
        """The case the dead-row fallback LOOKS like it is for, driven to
        show it is handled earlier: a column with nothing finite is
        skipped entirely, so the cell keeps a flat log density rather
        than an empty one."""
        measurements = np.array([[np.nan, np.nan], [1.0, 2.0]])
        priors = _priors("g1", "g2")

        r, _guides, _report = G.posterior_multivariate(
            measurements, priors, {"g1": [1.0, 0.0], "g2": [0.0, 1.0]})

        assert np.all(np.isfinite(r))
        assert np.allclose(r.sum(axis=1), 1.0)

    def test_a_guide_nothing_was_fitted_for_is_flat_rather_than_absent(self):
        """Documented as the honest prior, and worth driving: dropping it
        would renormalise the others and overstate them."""
        measurements = np.array([[1.0], [2.0], [3.0]])
        priors = _priors("fitted", "unfitted")

        r, guides, _report = G.posterior_multivariate(
            measurements, priors, {"fitted": [1.0]})

        assert guides == ("fitted", "unfitted")
        assert r.shape[1] == 2
        assert np.all(r[:, 1] > 0), (
            "a guide with no fitted effect was given zero mass, so the "
            "remaining guides absorb its share")

    def test_an_empty_input_returns_the_shape_the_caller_expects(self):
        r, guides, report = G.posterior_multivariate(
            np.zeros((0, 2)), _priors("g1"), {})

        assert r.shape == (0, 1)
        assert guides == ("g1",)
        assert report["scale_factor"] == 1.0


class TestTheConditionRatio:

    def test_a_design_with_no_columns_never_reaches_the_ratio(self):
        """THE PIN, for ``if sv.size == 0: return np.inf``.

        ``_ratio`` is handed the singular values of the design, so an
        empty vector means a design with no columns -- and the caller has
        already refused one by then. Held by ORDER rather than by
        driving, since the arm cannot be entered through the function.
        """
        from spacr import regression_qc as Q

        source = inspect.getsource(Q)
        ratio = source.index("def _ratio(sv):")
        svd = source.rindex("np.linalg.svd(Xm", 0, ratio)

        assert svd < ratio, (
            "_ratio is now defined before the SVD it consumes, so its "
            "empty-input arm may be reachable")
        assert "if sv.size == 0:" in source[ratio:ratio + 200]

    def test_an_all_zero_column_is_scaled_by_one_rather_than_dropped(self):
        """The comment above the guard is the substance: a zero column
        has no direction, scaling it by 1 leaves it zero, and the zero
        singular value that follows is the honest answer -- the design IS
        singular, and hiding that would report a fit as identified when
        it is not."""
        Xm = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        norms = np.linalg.norm(Xm, axis=0)

        assert norms[1] == 0.0
        safe = np.where(norms > 0, norms, 1.0)
        assert safe[1] == 1.0

        singular = np.linalg.svd(Xm / safe, compute_uv=False)
        assert singular[-1] == pytest.approx(0.0, abs=1e-12)

    def test_the_rank_tolerance_matches_numpy_rather_than_zero(self):
        """Why the check is a tolerance and not ``== 0``: LAPACK builds
        disagree about the last singular value of a rank-deficient
        matrix, so an exact comparison makes a duplicated predictor
        singular on one runner and fine on another."""
        from spacr import regression_qc as Q

        source = inspect.getsource(Q)
        assert "np.finfo(sv.dtype).eps * max(Xm.shape) * float(sv[0])" in source

        duplicated = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        sv = np.linalg.svd(duplicated, compute_uv=False)
        tolerance = np.finfo(sv.dtype).eps * max(duplicated.shape) * float(sv[0])

        assert sv[-1] <= tolerance
        assert int(np.linalg.matrix_rank(duplicated)) == 1
