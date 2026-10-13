"""One panel, both distributions, each named (instruction 218).

"i want to see a histogram of the dependent variable distribution after and
before transformation (same graph) in the graphs clearly stating what type
of distribution each is, normal bets, etcetera".

THE NAME COMES FROM THE FUNCTION THAT PICKS THE FAMILY. A second classifier
written for the picture would let the panel and the fitted model disagree
about the same data -- and the panel is the one the user would believe.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.response_distribution import (FAMILY_NAMES, caption, compare,
                                         describe, panel, transformed)


@pytest.fixture
def skewed():
    return np.random.default_rng(0).lognormal(0.0, 1.0, 2000)


@pytest.fixture
def normal():
    return np.random.default_rng(1).normal(10.0, 2.0, 2000)


@pytest.fixture
def proportions():
    return np.random.default_rng(2).beta(2.0, 5.0, 2000)


class TestTheNameComesFromTheFamilyPicker:

    def test_every_family_it_returns_has_a_name(self):
        """`check_distribution`'s answers are REGRESSION FAMILIES, and a
        histogram labelled 'quasi_binomial' says what spaCR would fit rather
        than what the reader is looking at."""
        for family in ("logit", "quasi_binomial", "beta", "ols", "glm"):
            assert family in FAMILY_NAMES

    def test_a_normal_response_is_named_normal(self, normal):
        assert describe(normal)["family"] == "ols"
        assert describe(normal)["name"] == "normal"

    def test_proportions_are_named_bounded(self, proportions):
        assert describe(proportions)["family"] in ("beta", "quasi_binomial")

    def test_binary_is_named_binary(self):
        assert describe(np.array([0, 1] * 50))["name"] == "binary"

    def test_it_agrees_with_check_distribution_itself(self, skewed):
        """Not a second opinion: the same call."""
        import contextlib
        import io

        from spacr.ml import check_distribution

        with contextlib.redirect_stdout(io.StringIO()):
            direct = check_distribution(skewed)
        assert describe(skewed)["family"] == direct


class TestTheStatisticTravelsWithTheName:
    """A panel that says "normal" and shows nothing else asks to be taken on
    faith about the one thing the reader came to check."""

    def test_the_p_value_is_there(self, normal):
        assert np.isfinite(describe(normal)["normality_p"])

    def test_the_skew_is_there(self, skewed):
        assert describe(skewed)["skew"] > 1.0

    def test_both_appear_in_the_caption(self, skewed):
        text = caption(compare(skewed, "log"))
        assert "D'Agostino p" in text and "skew" in text


class TestTheTransformIsTheRunsOwn:

    def test_it_goes_through_apply_transformation(self, skewed):
        """A panel showing a log the fit did not take is worse than no
        panel."""
        assert np.allclose(transformed(skewed, "log"), np.log1p(skewed))

    def test_sqrt_is_sqrt(self, skewed):
        assert np.allclose(transformed(skewed, "sqrt"), np.sqrt(skewed))

    def test_an_unknown_transform_changes_nothing(self, skewed):
        assert np.allclose(transformed(skewed, "no_such_thing"), skewed)

    def test_none_changes_nothing(self, skewed):
        assert np.allclose(transformed(skewed, "none"), skewed)


class TestASkewedResponseBecomesSymmetric:
    """Checked on generated data where the answer is known."""

    def test_the_log_pulls_the_skew_in(self, skewed):
        result = compare(skewed, "log")
        assert result["before"]["skew"] > 3.0
        assert result["after"]["skew"] < 1.5
        assert result["changed"]

    def test_and_the_caption_shows_both(self, skewed):
        text = caption(compare(skewed, "log"))
        assert "before" in text and "after log" in text


class TestNoneStillDraws:
    """An absent panel reads as a missing feature rather than an answer."""

    def test_the_comparison_is_built(self, skewed):
        result = compare(skewed, "none")
        assert result["before"]["family"] == result["after"]["family"]
        assert not result["changed"]

    def test_the_caption_says_it_changed_nothing(self, skewed):
        assert "changed nothing" in caption(compare(skewed, "none"))

    def test_the_panel_draws(self, skewed):
        result = panel(skewed, "none")
        assert result["axes"] is not None


class TestOnePanelNotTwo:

    def test_both_histograms_are_on_one_axes(self, proportions):
        """The question is what CHANGED, and two panels side by side is a
        comparison the reader has to do themselves."""
        result = panel(proportions, "none")
        axes = result["axes"]
        # Two hist() calls put two containers on the same axes.
        assert len(axes.containers) == 2

    def test_a_rescaling_transform_gets_its_own_x_axis(self, skewed):
        """A log of a proportion and the proportion itself share no scale,
        and forcing them onto one puts the smaller into a single bar --
        which looks like a finding and is an artefact of the axis."""
        result = panel(skewed, "log")
        axes = result["axes"]
        assert result["rescaled"]
        # The twin lives in the same figure.
        assert len(axes.figure.axes) == 2

    def test_the_names_are_on_the_panel(self, skewed):
        """The substance of the request: not left for the reader to judge
        by eye."""
        result = panel(skewed, "log")
        texts = [t.get_text() for t in result["axes"].texts]
        assert any("before" in t and "after" in t for t in texts)


class TestItSurvivesBadInput:

    def test_too_few_values_is_an_answer_not_a_crash(self):
        got = describe(np.array([1.0, 2.0]))
        assert got["family"] == ""
        assert "too few" in got["name"]

    def test_non_finite_values_are_dropped(self):
        got = describe(np.array([1.0, np.nan, np.inf] + list(range(20))))
        assert got["n"] == 21, "1.0 plus range(20); the nan and inf go"
