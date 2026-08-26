"""A link-like transform on a GLM is applied once, not twice.

A GLM chooses its own family and that family carries a link. Asking for
``transform='log'`` or ``'logit'`` on top of one is asking for the same
operation twice -- the model then fits ``logit(log(y))``, which is usually
why McFadden's R-squared comes back negative and meaningless.

THERE IS ONE RIGHT ANSWER AND IT IS NOT A PREFERENCE: fit the response as
measured and let the family's link do the transforming. This used to be a
three-valued setting; the other two values were an ordinary linear model of
the transformed response -- which ``regression_type='ols'`` already gives --
and the double transform itself, kept so an old result could be reproduced.
"""

from __future__ import annotations

import pytest

from spacr.ml import LINK_LIKE_TRANSFORMS, resolve_glm_transform_conflict


def _resolve(column, transform, available, regression_type="glm"):
    return resolve_glm_transform_conflict(
        column, transform=transform, available=available,
        regression_type=regression_type)


@pytest.mark.parametrize("kind", LINK_LIKE_TRANSFORMS)
def test_the_measured_response_is_fitted_and_the_transform_dropped(kind):
    column, transform, force_identity, note = _resolve(
        f"{kind}_frac", kind, [f"{kind}_frac", "frac"])
    assert column == "frac", "the untransformed column should be fitted"
    assert transform == "", "the transform is the family's job now"
    assert force_identity is False
    assert "once instead of twice" in note


@pytest.mark.parametrize("kind", LINK_LIKE_TRANSFORMS)
def test_it_says_so_in_the_run_log(kind):
    note = _resolve(f"{kind}_frac", kind, [f"{kind}_frac", "frac"])[3]
    assert "frac" in note and kind in note, (
        "the note must name both the column fitted and the transform ignored")


@pytest.mark.parametrize("kind", LINK_LIKE_TRANSFORMS)
def test_a_missing_raw_column_is_admitted_not_guessed(kind):
    """Without the untransformed column there is nothing better to fit."""
    column, transform, _, note = _resolve(f"{kind}_frac", kind, [f"{kind}_frac"])
    assert column == f"{kind}_frac"
    assert transform == kind
    assert "twice" in note, "the double transform has to be admitted"


@pytest.mark.parametrize("kind", ("sqrt", "square", "", None))
def test_a_transform_that_is_not_a_link_is_left_alone(kind):
    """Only log and logit collide with a family's link."""
    column, transform, force_identity, note = _resolve("y", kind, ["y"])
    assert column == "y"
    assert transform == kind
    assert force_identity is False
    assert note == ""


@pytest.mark.parametrize("other", ("ols", "wls", "ridge", "quantile"))
def test_only_a_glm_has_the_conflict(other):
    """Everything else is told which family to use, so nothing collides."""
    column, transform, _, note = _resolve(
        "logit_frac", "logit", ["logit_frac", "frac"], regression_type=other)
    assert column == "logit_frac"
    assert transform == "logit"
    assert note == ""


def test_the_choice_is_gone_from_the_settings():
    """It was three values, two of which were reachable another way."""
    from spacr.settings import expected_types

    assert "glm_transform_conflict" not in expected_types


def test_the_resolver_takes_no_resolution_argument():
    """The signature is the contract: there is nothing left to choose."""
    import inspect

    names = inspect.signature(resolve_glm_transform_conflict).parameters
    assert "resolution" not in names
