"""An untouched box is not a value, in either of the two places it lands.

Found by refitting the tsg101 screen's OWN saved settings under each
`REGRESSION_TYPES` entry (instruction 236 C7). Twelve of the eighteen types
could not be fitted from the settings the screen itself had written:

* `hinge_threshold` was never typed into, so the saved CSV holds an empty
  cell, which reads back as ``''``. That is not equal to the default of
  ``None``, so `_reject_unused_settings` refused every type that does not
  read it -- ols, wls, rlm, glm and the rest -- naming a setting the user
  had never touched.

* `cov_type` was blank in the same way, and the three binomial types DO
  read it, so it was not refused: it was passed to statsmodels, which
  answered "cov_type not recognized". A crash from inside the library,
  about a value nobody chose.

Both are the same mistake seen from two sides, so both are fixed the same
way: a blank is not a request.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


BLANKS = [None, "", "   ", "\t"]


class TestTheGuardLetsABlankThrough:
    @pytest.mark.parametrize("blank", BLANKS)
    def test_an_unread_setting_left_blank_is_not_refused(self, blank):
        from spacr.ml import _reject_unused_settings

        # 'ols' does not read hinge_threshold; a blank one is not a request.
        _reject_unused_settings('ols', {'hinge_threshold': (blank, None)})

    def test_a_value_that_was_actually_chosen_is_still_refused(self):
        """The guard is the reason a wrong number never becomes a result;
        loosening it for blanks must not loosen it for answers."""
        from spacr.ml import _reject_unused_settings

        with pytest.raises(ValueError, match="hinge_threshold"):
            _reject_unused_settings('ols', {'hinge_threshold': (0.5, None)})

    def test_zero_is_an_answer_not_a_blank(self):
        """`0` is falsy and is a perfectly good threshold, so a truth test
        here would have silently accepted it as "not asked for"."""
        from spacr.ml import _reject_unused_settings

        with pytest.raises(ValueError, match="hinge_threshold"):
            _reject_unused_settings('ols', {'hinge_threshold': (0, None)})

    @pytest.mark.parametrize("blank", BLANKS)
    def test_what_counts_as_blank(self, blank):
        from spacr.ml import _left_blank

        assert _left_blank(blank) is True

    @pytest.mark.parametrize("answered", [0, 0.0, False, "HC3", 1.345])
    def test_what_does_not(self, answered):
        from spacr.ml import _left_blank

        assert _left_blank(answered) is False

    def test_the_screens_own_saved_settings_can_be_refitted(self):
        """The reproduction, in miniature: every policed setting blank, as
        a settings CSV writes them, under a type that reads none of them."""
        from spacr.ml import _reject_unused_settings
        from spacr.regression_spec import _MODEL_LEVEL_DEFAULTS

        _reject_unused_settings('ols', {
            name: ('', default)
            for name, default in _MODEL_LEVEL_DEFAULTS.items()})


class TestABlankCovarianceIsNoCovariance:
    def _well_table(self, rows=90, seed=1):
        rng = np.random.default_rng(seed)
        design = pd.DataFrame({
            "guide_a": rng.integers(0, 2, rows).astype(float),
            "guide_b": rng.integers(0, 2, rows).astype(float),
        })
        response = 0.2 + 0.3 * design["guide_a"] + rng.normal(0, 0.05, rows)
        return design, response.clip(0.01, 0.99)

    @pytest.mark.parametrize("blank", BLANKS)
    @pytest.mark.parametrize("kind", ["logit", "probit", "quasi_binomial"])
    def test_a_binomial_fit_survives_a_blank_box(self, kind, blank):
        """These three READ cov_type, so nothing refused the blank -- it
        reached statsmodels and raised there."""
        from spacr.ml import regression_model

        design, response = self._well_table()
        model = regression_model(design, response, regression_type=kind,
                                 cov_type=blank, verbose=False)
        assert model is not None

    def test_a_real_covariance_estimator_still_reaches_the_fit(self):
        """The normalisation must not have turned every cov_type into
        None: a run labelled HC3 whose standard errors were ordinary would
        be worse than the crash it replaced."""
        from spacr.ml import regression_model

        design, response = self._well_table()
        plain = regression_model(design, response, regression_type='ols',
                                 cov_type=None, verbose=False)
        robust = regression_model(design, response, regression_type='ols',
                                  cov_type='HC3', verbose=False)
        assert not np.allclose(np.asarray(plain.bse, dtype=float),
                               np.asarray(robust.bse, dtype=float))

    def test_a_covariance_estimator_that_is_not_one_still_fails(self):
        """A typo is not a blank, and must not be quietly dropped."""
        from spacr.ml import regression_model

        design, response = self._well_table()
        with pytest.raises(Exception):
            regression_model(design, response, regression_type='ols',
                             cov_type='HC99', verbose=False)
