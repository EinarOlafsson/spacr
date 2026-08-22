"""A penalty is only large or small relative to the design it is applied to.

Instruction 236 C7, found by refitting the tsg101 screen under every
regression type. `group_lasso_lambda` shipped as 0.05 -- a NUMBER, and the
number that the panel posted whether or not anybody had looked at it. That
design's own ceiling, the penalty above which nothing at all survives, is
0.1285. So the default was nearly half of it, all 297 gene blocks came back
exactly zero, and the run was refused.

`choose_lambda` cross-validates over a path measured down from that
ceiling, which is what makes it scale-free: a design of fractions and a
design of counts have ceilings three orders of magnitude apart, and any
fixed number suits one and not the other.

TWO THINGS IT REFUSES TO DO, both found by driving it rather than reading
it:

* It will not choose the empty fit. With a weak signal the emptiest model
  often predicts held-out wells best, and "no gene does anything" as the
  output of a mean-squared-error argmin is not an answer anybody can check.

* It does not read selection off the folds. A penalty at which four fifths
  of the wells keep a gene and all of them together keep none was chosen on
  the real screen, and the full fit was then refused by the caller's guard.
  The error is held-out; the selection is measured on the fit the caller
  will actually get.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import group_lasso as gl


def _screen(rows=140, genes=6, guides=3, seed=0, noise=0.3,
            nuisance_scale=0.0):
    """A design with one gene that matters, optionally with big nuisance
    columns beside it -- plate/row/column dummies, in the real thing."""
    rng = np.random.default_rng(seed)
    columns = genes * guides
    design = rng.normal(size=(rows, columns))
    labels = [f"gene{i // guides}" for i in range(columns)]
    response = design[:, :guides].sum(axis=1) * 1.5 \
        + rng.normal(0, noise, rows)
    if nuisance_scale:
        extra = rng.normal(size=(rows, 3)) * nuisance_scale
        design = np.hstack([design, extra])
        labels += ["row_a", "row_b", "row_c"]
        response = response + extra.sum(axis=1)
    wanted = np.array([not name.startswith("row_") for name in labels])
    return design, response, labels, wanted


class TestThePath:
    def test_it_starts_at_the_ceiling(self):
        """Anything above `max_lambda` gives the same all-zero fit, so a
        path that starts higher spends its first points on one answer."""
        design, response, labels, _wanted = _screen()
        path = gl.penalty_path(design, response, labels)
        assert path[0] == pytest.approx(
            gl.max_lambda(design, response, labels))

    def test_it_descends(self):
        design, response, labels, _wanted = _screen()
        path = gl.penalty_path(design, response, labels)
        assert np.all(np.diff(path) < 0)

    def test_it_is_relative_rather_than_absolute(self):
        """The whole reason a fixed default failed: scale the response by a
        thousand and every sensible penalty moves with it."""
        design, response, labels, _wanted = _screen()
        small = gl.penalty_path(design, response, labels)
        large = gl.penalty_path(design, response * 1000.0, labels)
        assert large[0] == pytest.approx(small[0] * 1000.0, rel=1e-6)

    def test_a_design_with_no_columns_to_penalise_gives_one_point(self):
        path = gl.penalty_path(np.zeros((10, 2)), np.zeros(10), ["a", "a"])
        assert path.size == 1


class TestChoosingIt:
    def test_it_finds_a_penalty_that_recovers_the_planted_gene(self):
        design, response, labels, _wanted = _screen()
        chosen = gl.choose_lambda(design, response, labels)
        effects = gl.gene_effects(design, response, labels, lam=chosen)
        assert effects.iloc[0]["gene"] == "gene0"
        assert bool(effects.iloc[0]["selected"])

    def test_it_never_returns_the_empty_fit(self):
        """Pure noise: every penalty on the path is defensible and the
        emptiest predicts best. It must still return something fittable
        rather than a penalty known to select nothing."""
        rng = np.random.default_rng(3)
        design = rng.normal(size=(90, 9))
        labels = [f"gene{i // 3}" for i in range(9)]
        response = rng.normal(size=90)
        chosen = gl.choose_lambda(design, response, labels)
        assert chosen >= 0
        assert chosen <= gl.max_lambda(design, response, labels)

    def test_the_nuisance_columns_do_not_count_as_a_selection(self):
        """THE DEFECT. Row and column dummies are singleton groups with far
        larger correlations than any guide block, so they stand at
        penalties that have already emptied every gene -- and a fit whose
        every gRNA coefficient is zero reads downstream as a screen with no
        hits."""
        design, response, labels, wanted = _screen(nuisance_scale=9.0)
        chosen = gl.choose_lambda(design, response, labels, required=wanted)
        beta, _intercept, _converged = gl.fit(design, response, labels,
                                              lam=chosen)
        assert np.any(beta[wanted]), (
            "chose a penalty at which no gene survives")

    def test_the_full_fit_is_what_decides_whether_anything_was_selected(
            self):
        """Measured on the screen: a penalty kept a gene on four fifths of
        the wells and none on all of them, so the caller's guard refused the
        very fit the cross-validation had recommended."""
        design, response, labels, wanted = _screen(nuisance_scale=9.0,
                                                   noise=1.2, seed=5)
        chosen = gl.choose_lambda(design, response, labels, required=wanted)
        beta, _intercept, _converged = gl.fit(design, response, labels,
                                              lam=chosen)
        assert np.any(beta[wanted])

    def test_a_required_mask_of_the_wrong_length_is_refused(self):
        design, response, labels, _wanted = _screen()
        with pytest.raises(ValueError, match="required has"):
            gl.choose_lambda(design, response, labels,
                             required=np.ones(3, dtype=bool))

    def test_the_same_seed_gives_the_same_penalty(self):
        """Two runs of one screen have to agree, or the methods section
        cannot name the penalty that was used."""
        design, response, labels, _wanted = _screen()
        first = gl.choose_lambda(design, response, labels, seed=7)
        second = gl.choose_lambda(design, response, labels, seed=7)
        assert first == second

    def test_it_beats_the_old_fixed_default_on_a_fraction_scale_design(self):
        """The reproduction, in miniature: a response of fractions, where
        0.05 is most of the way to the ceiling."""
        design, response, labels, _wanted = _screen()
        response = (response - response.min())
        response = response / (response.max() * 4.0)      # into (0, 0.25)
        ceiling = gl.max_lambda(design, response, labels)
        assert ceiling < 0.05, "the fixture is not fraction-scale enough"

        empty, _i, _c = gl.fit(design, response, labels, lam=0.05)
        assert not np.any(empty), "0.05 was supposed to empty this design"

        chosen = gl.choose_lambda(design, response, labels)
        beta, _intercept, _converged = gl.fit(design, response, labels,
                                              lam=chosen)
        assert np.any(beta)


class TestTheSettingSaysAuto:
    def test_the_shipped_default_cross_validates(self):
        from spacr.regression_spec import _MODEL_LEVEL_DEFAULTS

        assert _MODEL_LEVEL_DEFAULTS['group_lasso_lambda'] == 'auto'

    def test_a_settings_file_written_before_today_is_migrated(self):
        """0.05 was what the panel posted for the whole of this setting's
        life, so every saved file carries it and none of them chose it.
        Left in place it would ask for a penalty under every OTHER type,
        which refuses a setting it cannot read."""
        from spacr.settings import (LEGACY_GROUP_LASSO_LAMBDA,
                                    get_perform_regression_default_settings)

        settings = get_perform_regression_default_settings(
            {'regression_type': 'ols',
             'group_lasso_lambda': LEGACY_GROUP_LASSO_LAMBDA})
        assert settings['group_lasso_lambda'] == 'auto'

    def test_a_penalty_somebody_actually_chose_is_kept(self):
        from spacr.settings import get_perform_regression_default_settings

        settings = get_perform_regression_default_settings(
            {'regression_type': 'group_lasso',
             'group_lasso_lambda': 0.004})
        assert settings['group_lasso_lambda'] == 0.004

    def test_auto_passes_validation(self):
        from spacr.settings import get_perform_regression_default_settings

        settings = get_perform_regression_default_settings(
            {'regression_type': 'group_lasso',
             'group_lasso_lambda': 'auto'})
        assert settings['group_lasso_lambda'] == 'auto'

    def test_a_negative_penalty_is_still_refused(self):
        """The loosening is for 'auto' and a blank, not for nonsense."""
        from spacr.settings import get_perform_regression_default_settings

        with pytest.raises(ValueError, match="group_lasso_lambda"):
            get_perform_regression_default_settings(
                {'regression_type': 'group_lasso',
                 'group_lasso_lambda': -1.0})
