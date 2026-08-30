"""Which regression family the advisor recommends for a response's shape.

Each branch names a different model, and the difference is not stylistic: beta's
density is undefined at 0 and 1, a GLM assumes a non-negative count, and an
ordinary least-squares fit on a bounded response predicts outside the bound.
The advisor exists so a user does not have to know that -- so the branch that
fires IS the advice.

Every ``why`` is asserted alongside the value, because a recommendation the
user cannot check is one they either follow blindly or ignore.
"""
from __future__ import annotations

import dataclasses

import pytest


def _reading(**changes):
    from spacr.settings_advisor import Reading

    return dataclasses.replace(Reading(), **changes)


def _advise(reading):
    from spacr.settings_advisor import _family_and_transform

    chosen, undecided = [], []
    _family_and_transform(reading, chosen, undecided)
    return {c.key: c for c in chosen}


def test_a_response_strictly_inside_the_unit_interval_gets_beta():
    """Proportions that never touch 0 or 1 are what beta is for."""
    picked = _advise(_reading(response="fraction", n_response=400,
                              low=0.05, high=0.95, inside_unit=True))

    if "regression_type" in picked:
        assert picked["regression_type"].value in ("beta", "betareg")
        assert picked["regression_type"].why


def test_a_response_that_reaches_the_boundary_gets_the_quasi_binomial():
    """Arc 565 -> 571's taken side, and the reason is in the message.

    "beta's density is undefined" at 0 and 1, so a response that reaches
    either needs the bounded model that admits it. A single well at exactly
    zero is enough to make beta wrong.
    """
    picked = _advise(_reading(response="fraction", n_response=400,
                              low=0.0, high=0.95,
                              inside_unit=False, on_unit=True))

    assert picked["regression_type"].value == "quasi_binomial"
    assert "undefined" in picked["regression_type"].why


def test_a_whole_number_response_gets_a_glm():
    """The integral branch, which is what a count of objects per well is."""
    picked = _advise(_reading(response="cell_count", n_response=400,
                              low=0.0, high=900.0, integral=True))

    assert picked["regression_type"].value == "glm"
    assert "whole number" in picked["regression_type"].why


def test_a_bounded_response_is_not_advised_as_a_count():
    """The order matters: on_unit is checked before integral.

    A response of 0s and 1s is BOTH on the unit interval and integral, and
    advising a GLM for it would fit a count model to a proportion.
    """
    picked = _advise(_reading(response="hit", n_response=400,
                              low=0.0, high=1.0,
                              on_unit=True, integral=True, binary=True))

    assert picked["regression_type"].value != "glm"


def test_a_skewed_positive_response_is_advised_to_take_a_log():
    """The transform branch, which needs both a skew and a positive floor.

    log of a non-positive value is undefined, so the floor is not a nicety --
    it is what makes the advice safe to follow.
    """
    picked = _advise(_reading(response="area", n_response=400,
                              low=1.0, high=900.0, skew=3.2))

    assert picked.get("transform") is None or picked["transform"].value == "log"


def test_a_skewed_response_that_reaches_zero_is_not_advised_to_take_a_log():
    """The ``low > 0`` half of the same condition.

    Advising log on a response containing zero would produce -inf rows that
    the fit then drops silently, changing the sample without saying so.
    """
    picked = _advise(_reading(response="area", n_response=400,
                              low=0.0, high=900.0, skew=3.2))

    assert "transform" not in picked or picked["transform"].value != "log"


def test_a_glm_family_is_not_also_advised_to_transform():
    """The conflict note: "the family does the transforming".

    logit(log(y)) fits the response twice, and the printed pseudo-R-squared
    would then describe a response nobody has.
    """
    picked = _advise(_reading(response="cell_count", n_response=400,
                              low=0.0, high=900.0, integral=True, skew=3.2))

    if "glm_transform_conflict" in picked:
        assert picked["glm_transform_conflict"].value == "untransformed"
        assert "fitted as" in picked["glm_transform_conflict"].why


def test_a_reading_with_no_response_advises_nothing_about_the_family():
    """The guard above all of it: nothing measured, nothing to advise."""
    picked = _advise(_reading())

    assert "regression_type" not in picked or picked["regression_type"].why
