"""Instruction 182: BOTH fixes are available and the user picks.

A log-transformed response handed to a family that carries a logit link fits
logit(log(y)) -- a quantity nothing measures -- and the live TSG101 run that
found this reported McFadden's R² of -20.2752 as the symptom.

There are two defensible fixes and they answer different questions:

  'untransformed'  choose the family on the response AS MEASURED and let the
                   family's link do the transforming.
  'transformed'    keep the transform and fit an identity link.

Asked for on 2026-08-20 -- "for 182 i want both options to be available" --
so spaCR offers both rather than choosing. The old behaviour survives as
'warn' ONLY so a run someone already published can be reproduced, and it
says so when selected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from spacr.ml import (DEFAULT_GLM_TRANSFORM_CONFLICT, GLM_TRANSFORM_CONFLICTS,
                      fit_quality_note, regression_model,
                      resolve_glm_transform_conflict)

COLUMNS = ("pred", "log_pred", "fraction")


@pytest.fixture(scope="module")
def transform_rule():
    """Module-scoped: a class-scoped fixture defined as an instance method
    warns, and the rule table is the same object for every test here."""
    from spacr.settings import get_setting_dependencies

    return get_setting_dependencies()["transform"]


@pytest.fixture
def screen():
    """A proportion response and its log, the shape the fault was found on."""
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"const": 1.0, "fraction": rng.uniform(0.0, 1.0, n)})
    raw = pd.Series(
        np.clip(0.2 + 0.4 * X["fraction"] + rng.normal(0.0, 0.05, n),
                1e-3, 1.0 - 1e-3), name="pred")
    return X, raw, pd.Series(np.log1p(raw), name="log_pred")


# ------------------------------------------------------- the resolution

class TestBothOptionsExist:

    def test_the_three_values_are_the_documented_ones(self):
        assert GLM_TRANSFORM_CONFLICTS == ("untransformed", "transformed",
                                           "warn")
        assert DEFAULT_GLM_TRANSFORM_CONFLICT in GLM_TRANSFORM_CONFLICTS

    def test_untransformed_fits_the_response_as_measured(self):
        column, transform, identity, note = resolve_glm_transform_conflict(
            "log_pred", transform="log", resolution="untransformed",
            available=COLUMNS)

        assert column == "pred", "it must fit the column the screen measured"
        assert transform == "", "the transform is no longer in effect"
        assert not identity, "the family's own link does the transforming"
        assert "pred" in note and "once instead of twice" in note

    def test_transformed_keeps_the_transform_and_forces_identity(self):
        column, transform, identity, note = resolve_glm_transform_conflict(
            "log_pred", transform="log", resolution="transformed",
            available=COLUMNS)

        assert column == "log_pred", "the user asked for the transform"
        assert transform == "log"
        assert identity, "so the family must not apply a second one"
        assert "identity link" in note

    def test_warn_is_the_old_behaviour_and_admits_it(self):
        column, transform, identity, note = resolve_glm_transform_conflict(
            "log_pred", transform="log", resolution="warn",
            available=COLUMNS)

        assert (column, transform, identity) == ("log_pred", "log", False)
        assert "reproduced" in note, (
            "the one reason to keep it has to be the reason it gives")

    def test_an_unknown_resolution_is_refused(self):
        with pytest.raises(ValueError, match="glm_transform_conflict"):
            resolve_glm_transform_conflict("log_pred", transform="log",
                                           resolution="whatever")


class TestItDoesNotReachInWhenThereIsNoConflict:
    """The setting resolves a conflict; it is not a second transform switch."""

    @pytest.mark.parametrize("transform", ["sqrt", "square", "", None])
    def test_a_transform_that_is_not_a_link_is_left_alone(self, transform):
        column, kept, identity, note = resolve_glm_transform_conflict(
            "sqrt_pred", transform=transform, resolution="untransformed",
            available=COLUMNS)

        assert (column, kept, identity, note) == ("sqrt_pred", transform,
                                                  False, "")

    @pytest.mark.parametrize("kind", ["ols", "ridge", "mixed", "beta"])
    def test_only_glm_chooses_its_own_family_so_only_glm_has_the_conflict(
            self, kind):
        column, kept, identity, note = resolve_glm_transform_conflict(
            "log_pred", transform="log", resolution="untransformed",
            available=COLUMNS, regression_type=kind)

        assert (column, kept, identity, note) == ("log_pred", "log", False, "")

    def test_a_missing_raw_column_says_so_rather_than_inventing_one(self):
        """It must not invert the transform to manufacture the response."""
        column, kept, identity, note = resolve_glm_transform_conflict(
            "log_pred", transform="log", resolution="untransformed",
            available=["log_pred"])

        assert column == "log_pred", "nothing to switch to"
        assert "not in the frame" in note
        assert "still applies" in note, (
            "a silent fallback would leave the double transform unannounced")


# ------------------------------------------------------------- the fits

class TestTheFitsThemselves:

    def test_untransformed_gives_a_binomial_logit_and_a_positive_r2(
            self, screen):
        X, raw, _ = screen
        model = regression_model(X, raw, "glm", response_name="pred",
                                 transform="")

        assert isinstance(model.family, sm.families.Binomial)
        assert isinstance(model.family.link, sm.families.links.Logit)
        assert 1.0 - model.llf / model.llnull > 0.0, (
            "the instruction's own success test: not negative")

    def test_transformed_gives_a_gaussian_identity(self, screen):
        X, _, log = screen
        model = regression_model(X, log, "glm", response_name="log_pred",
                                 transform="log", glm_force_identity=True)

        assert isinstance(model.family, sm.families.Gaussian)
        assert isinstance(model.family.link, sm.families.links.Identity)

    def test_the_old_behaviour_still_stacks_a_link_on_a_transform(self, screen):
        """Which is the whole reason 'warn' is kept, so it has to still do it."""
        X, _, log = screen
        model = regression_model(X, log, "glm", response_name="log_pred",
                                 transform="log")

        assert isinstance(model.family.link, sm.families.links.Logit)

    def test_the_double_transform_fits_worse_than_the_response_as_measured(
            self, screen):
        """The honest form of the claim, measured rather than asserted.

        THE NEGATIVE McFADDEN WAS A DIFFERENT BUG. Instruction 182 recorded
        R² = -20.2752 and read it as the double transform's symptom -- "not a
        separate problem". It was a separate problem: the pseudo-R² divided
        the true log-likelihood by `null_deviance / -2`, which is the null
        log-likelihood only when the saturated one is zero. That holds for
        0/1 binomial data and not for the per-well PROPORTIONS this pipeline
        fits. On this fixture the SAME expression reported -11.5329 for the
        correctly specified fit, so it was condemning good models too.

        With the null log-likelihood taken properly, the double transform is
        not negative -- it is simply worse, which is the true and much
        smaller claim.
        """
        X, raw, log = screen
        doubled = regression_model(X, log, "glm", response_name="log_pred",
                                   transform="log")
        measured = regression_model(X, raw, "glm", response_name="pred")

        doubled_r2 = 1.0 - doubled.llf / doubled.llnull
        measured_r2 = 1.0 - measured.llf / measured.llnull
        assert 0.0 < doubled_r2 < measured_r2


class TestTheRightNumberForTheFamily:
    """A Gaussian fit must not be summarised with McFadden's R²."""

    def test_a_gaussian_fit_reports_ordinary_r2_and_says_which(self, screen):
        X, _, log = screen
        model = regression_model(X, log, "glm", response_name="log_pred",
                                 transform="log", glm_force_identity=True)
        note = fit_quality_note(model)

        assert note.startswith("R²:")
        assert "not McFadden" in note
        value = float(note.split(":")[1].split()[0])
        assert 0.0 <= value <= 1.0, (
            "an ordinary R² is bounded; the McFadden expression this "
            "replaces reported 467.1217 for exactly this fit")

    def test_a_binomial_fit_still_reports_mcfadden(self, screen):
        X, raw, _ = screen
        model = regression_model(X, raw, "glm", response_name="pred")

        assert fit_quality_note(model).startswith("McFadden's R²:")

    def test_a_negative_value_is_still_flagged(self):
        """Driven directly, because a correctly-computed R² rarely goes
        negative and the flag must not quietly stop working while nothing
        happens to exercise it."""
        from spacr.ml import mcfadden_note

        note = mcfadden_note(-20.2752)

        assert "NEGATIVE" in note and "transformed twice" in note

    def test_a_model_with_no_likelihood_does_not_raise(self):
        class Bare:
            family = None

        assert "not available" in fit_quality_note(Bare())


# --------------------------------------------- the transform goes grey (106)

class TestTheTransformSaysWhenItIsDead:
    """Instruction 182's last owed item, under 106's rule.

    Under 'untransformed' the run reads `transform` and then ignores it,
    because the family's own link does that job. A control the run silently
    discards has to say so -- that is the whole of 106 -- and 182 named this
    as the case for it.
    """

    @pytest.fixture
    def rule(self, transform_rule):
        return transform_rule

    def _settings(self, **over):
        base = {"regression_type": "glm",
                "glm_transform_conflict": "untransformed",
                "transform": "log"}
        base.update(over)
        return base

    def test_it_greys_when_the_link_would_do_the_job_twice(self, rule):
        assert not rule["predicate"](self._settings(), None)

    @pytest.mark.parametrize("transform", ["log", "logit"])
    def test_both_link_like_transforms_are_covered(self, rule, transform):
        assert not rule["predicate"](self._settings(transform=transform), None)

    @pytest.mark.parametrize("transform", ["sqrt", "square", None, ""])
    def test_a_transform_that_is_not_a_link_stays_live(self, rule, transform):
        """It is applied normally under every resolution, so greying it would
        disable a control for a reason that is not true of it."""
        assert rule["predicate"](self._settings(transform=transform), None)

    def test_keeping_the_transform_keeps_the_control(self, rule):
        assert rule["predicate"](
            self._settings(glm_transform_conflict="transformed"), None)

    def test_the_old_behaviour_keeps_the_control_too(self, rule):
        """'warn' applies the transform, whatever else is wrong with it."""
        assert rule["predicate"](
            self._settings(glm_transform_conflict="warn"), None)

    @pytest.mark.parametrize("kind", ["ols", "ridge", "mixed", "beta"])
    def test_a_type_that_does_not_choose_its_family_has_no_conflict(
            self, rule, kind):
        assert rule["predicate"](self._settings(regression_type=kind), None)

    def test_the_reason_names_both_settings_and_the_way_out(self, rule):
        reason = rule["reason"](self._settings(), None)

        assert "transform='log'" in reason
        assert "untransformed" in reason
        assert "'transformed'" in reason, "a greyed control must say the way out"
        assert "kept and saved" in reason, "and that nothing was thrown away"

    def test_the_rule_declares_the_settings_it_reads(self, rule):
        """A predicate reading a setting absent from `sources` would be
        evaluated against a default the user cannot see or change."""
        assert "glm_transform_conflict" in rule["sources"]
        assert "regression_type" in rule["sources"]
