"""Instruction 182 A and C — the family sniffer names the scale it examined.

From a live glm run on the reference screen:

    transform                     log
    Data strictly between 0 and 1. Using Binomial family with Logit link
    Dep. Variable: log_pred   Model Family: Binomial   Link: Logit
    McFadden's R²: -20.2752

Both sentences are individually true and together they are a bug. The values
inside (0, 1) were the LOGGED ones; the family was chosen for a scale the
sentence did not name, and the model then fitted logit(log(p)) -- not a
quantity the screen measures. The negative pseudo-R² is that.

Which of the two transforms to drop is a decision for the maintainer (182's
open half): both readings are defensible science and picking one silently
would change every glm result spaCR has produced. These tests hold the parts
that are true under either -- the sentence names the scale, and the stacking
is refused to be quiet about.
"""
from __future__ import annotations

import numpy as np
import pytest

sm = pytest.importorskip("statsmodels.api")

from spacr.ml import (                                    # noqa: E402
    LINK_LIKE_TRANSFORMS, double_transform_warning, pick_glm_family_and_link,
)


def _proportions(n=200, seed=0):
    return np.clip(np.random.default_rng(seed).random(n), 0.01, 0.99)


# -- A: the sentence names the column and the transform --------------------

def test_the_sentence_names_the_column_it_examined(capsys):
    pick_glm_family_and_link(_proportions(), name="pred")
    assert "pred is strictly between 0 and 1" in capsys.readouterr().out


def test_the_sentence_names_the_transform_that_made_that_column(capsys):
    """"Data strictly between 0 and 1" was true of log_pred and said so of nothing."""
    pick_glm_family_and_link(_proportions(), name="log_pred", transform="log")
    said = capsys.readouterr().out
    assert "log_pred (after transform='log')" in said


def test_a_response_with_no_name_still_reads_as_a_sentence(capsys):
    pick_glm_family_and_link(_proportions())
    assert "the response is strictly between 0 and 1" in capsys.readouterr().out


@pytest.mark.parametrize("values,expected", [
    (np.array([0.0, 1.0, 1.0, 0.0] * 50), "is binary"),
    (np.array([0.0, 0.5, 1.0] * 60), "including the boundaries"),
])
def test_every_branch_names_the_scale_not_just_the_one_that_was_reported(
        values, expected, capsys):
    pick_glm_family_and_link(values, name="pred")
    said = capsys.readouterr().out
    assert "pred" in said and expected in said


# -- C: the double transform is not passed over in silence -----------------

def test_a_log_transform_under_a_logit_link_is_named_as_two_transforms():
    family = sm.families.Binomial(link=sm.families.links.Logit())
    warning = double_transform_warning("log_pred", "log", family)

    assert "transformed TWICE" in warning
    assert "logit(log(y))" in warning
    # BOTH WAYS OUT, because which one is right is the maintainer's call.
    assert "drop the transform" in warning
    assert "regression_type='ols'" in warning


def test_the_warning_connects_itself_to_the_symptom_a_user_will_see():
    family = sm.families.Binomial(link=sm.families.links.Logit())
    assert "McFadden" in double_transform_warning("log_pred", "log", family)


def test_an_identity_link_is_not_a_second_transform():
    """A logged response under a Gaussian identity link is a standard model."""
    family = sm.families.Gaussian(link=sm.families.links.Identity())
    assert double_transform_warning("log_pred", "log", family) == ""


def test_no_transform_means_nothing_to_warn_about():
    family = sm.families.Binomial(link=sm.families.links.Logit())
    assert double_transform_warning("pred", "", family) == ""
    assert double_transform_warning("pred", None, family) == ""


@pytest.mark.parametrize("transform", LINK_LIKE_TRANSFORMS)
def test_every_link_like_transform_is_covered_not_just_log(transform):
    family = sm.families.Binomial(link=sm.families.links.Logit())
    assert "transformed TWICE" in double_transform_warning("y", transform, family)


def test_a_transform_that_is_not_a_link_is_left_alone():
    """`sqrt` and `z_score` change the scale; they do not apply a link."""
    family = sm.families.Binomial(link=sm.families.links.Logit())
    for kind in ("sqrt", "z_score", "none"):
        assert double_transform_warning("y", kind, family) == ""


def test_the_warning_is_printed_at_the_point_the_family_is_chosen(capsys):
    """Before the fit -- by the time it reached the summary the fit had run."""
    pick_glm_family_and_link(_proportions(), name="log_pred", transform="log")
    assert "transformed TWICE" in capsys.readouterr().out


def test_the_family_itself_is_unchanged_because_the_choice_is_not_made_here():
    """182's open half. This warns; it does not pick a side."""
    family = pick_glm_family_and_link(_proportions(), name="log_pred",
                                      transform="log")
    assert isinstance(family, sm.families.Binomial)
