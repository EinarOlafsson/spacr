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
        assert _reject_unused_settings(
            'ols', {'hinge_threshold': (blank, None)}) is None
        # AND THE GUARD IS STILL LIVE on the same call. Without this the
        # test would pass against a function that had been emptied out.
        with pytest.raises(ValueError, match="hinge_threshold"):
            _reject_unused_settings('ols', {'hinge_threshold': (0.5, None)})

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

        blanks = {name: ('', default)
                  for name, default in _MODEL_LEVEL_DEFAULTS.items()}
        assert len(blanks) >= 6, "the policed set shrank; check this list"
        assert _reject_unused_settings('ols', blanks) is None
        # One of them answered is still refused, by name.
        answered = dict(blanks, quantile=(0.9, 0.5))
        with pytest.raises(ValueError, match="quantile"):
            _reject_unused_settings('ols', answered)


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


class TestABlankHingeThresholdIsNoCut:
    """`hinge` READS hinge_threshold, so the blank was not refused: it
    reached `binarise_response`, which asked `float('')` for a number."""

    @pytest.mark.parametrize("blank", BLANKS)
    def test_the_failure_names_the_setting_rather_than_the_conversion(
            self, blank):
        from spacr.ml import binarise_response

        with pytest.raises(ValueError, match="hinge_threshold"):
            binarise_response(np.linspace(0.0, 1.0, 40), blank)

    @pytest.mark.parametrize("blank", BLANKS)
    def test_a_binary_response_needs_no_cut_at_all(self, blank):
        """A blank means "no cut", and a response that is already 0/1 does
        not need one -- so this is the case that must keep working."""
        from spacr.ml import binarise_response

        out = binarise_response([0, 1, 1, 0, 1], blank)
        assert list(out) == [0.0, 1.0, 1.0, 0.0, 1.0]

    def test_a_cut_that_was_chosen_is_still_applied(self):
        from spacr.ml import binarise_response

        out = binarise_response([0.1, 0.2, 0.6, 0.9], 0.5)
        assert list(out) == [0.0, 0.0, 1.0, 1.0]

    def test_zero_is_a_cut_and_not_a_blank(self):
        from spacr.ml import binarise_response

        out = binarise_response([-1.0, -0.5, 0.5, 1.0], 0)
        assert list(out) == [0.0, 0.0, 1.0, 1.0]


class TestThePenaltyThePanelPostedIsNotAPenaltyAnybodyChose:
    """alpha=1 is the UNPENALISED families' default and shrinks a
    fraction-scale design to nothing. There was already a rescue for it,
    and it spared a FLOAT 1.0 on the reading that an integer is the posted
    default and a float is a deliberate answer.

    That was true of the Tk panel and is not true of this one: the Qt field
    is a double spin box and the settings CSV it writes says `alpha,1.0`.
    So the rescue never fired for anyone running the current GUI, and lasso
    and elasticnet both refused the tsg101 screen -- "shrank all 298
    coefficients to exactly zero at alpha=1.0" -- from a settings file in
    which nobody had touched alpha.
    """

    @pytest.mark.parametrize("posted", [1, 1.0])
    @pytest.mark.parametrize("kind", ["lasso", "ridge", "elasticnet"])
    def test_the_posted_default_cross_validates(self, kind, posted):
        from spacr.settings import get_perform_regression_default_settings

        settings = get_perform_regression_default_settings(
            {'regression_type': kind, 'alpha': posted})
        assert settings['alpha'] == 'auto'

    @pytest.mark.parametrize("chosen", [0.01, 0.5, 2.0, 10])
    def test_a_penalty_somebody_chose_is_kept(self, chosen):
        from spacr.settings import get_perform_regression_default_settings

        settings = get_perform_regression_default_settings(
            {'regression_type': 'lasso', 'alpha': chosen})
        assert settings['alpha'] == chosen

    def test_the_unpenalised_families_keep_their_1(self):
        """They ignore alpha, and `_reject_unused_settings` compares it
        against the default of 1 -- so rewriting it to 'auto' here would
        make every OLS run look like a request."""
        from spacr.settings import get_perform_regression_default_settings

        settings = get_perform_regression_default_settings(
            {'regression_type': 'ols', 'alpha': 1.0})
        assert settings['alpha'] == 1.0

    def test_it_says_so_rather_than_doing_it_quietly(self, capsys):
        """A penalty chosen for the user and never named is one they cannot
        put in a methods section."""
        from spacr.settings import get_perform_regression_default_settings

        get_perform_regression_default_settings(
            {'regression_type': 'lasso', 'alpha': 1.0})
        assert "auto" in capsys.readouterr().out
