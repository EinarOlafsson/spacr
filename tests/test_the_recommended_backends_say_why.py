"""Instruction 133 A — which backends answer THIS question, and why each.

    "the ones that are closest to answering a question in a screen like this
     should get a 'Recommended for CRISPR screens' in the text box"

A pooled screen is p >> n and sparse — 823 guides estimated from 610 wells on
the reference screen — and every entry earns its place against that shape
rather than against the general reputation of the method.

The caveat is held as tightly as the badge, because a badge without it reads
as a promise: penalisation, priors and grouping do not create information.
"""
from __future__ import annotations

import pytest

from spacr.qt.screens.settings_model import (INFORMATION_LIMIT_NOTE,
                                             RECOMMENDED_FOR_SCREENS,
                                             regression_model_explainer)
from spacr.regression_spec import REGRESSION_TYPES

BADGE = "RECOMMENDED FOR CRISPR SCREENS"


def _box(kind: str) -> str:
    return regression_model_explainer(kind, "both")


def test_exactly_the_recommended_backends_carry_the_badge():
    flagged = sorted(k for k in REGRESSION_TYPES if BADGE in _box(k))
    assert flagged == sorted(RECOMMENDED_FOR_SCREENS)


def test_the_default_is_one_of_them():
    """`mixed` takes its own branch in the renderer -- the one place a
    missing badge would have mattered most."""
    import spacr.settings as settings

    default = settings.get_perform_regression_default_settings({})[
        "regression_type"]
    assert default in RECOMMENDED_FOR_SCREENS
    assert BADGE in _box(default)


@pytest.mark.parametrize("kind", sorted(RECOMMENDED_FOR_SCREENS))
def test_each_badge_carries_its_own_reason(kind):
    """A badge is a claim; the reason is what makes it checkable."""
    box = _box(kind)
    reason = RECOMMENDED_FOR_SCREENS[kind]
    assert reason.split(" -- ")[0][:40] in box


@pytest.mark.parametrize("kind", sorted(RECOMMENDED_FOR_SCREENS))
def test_the_caveat_travels_with_the_badge(kind):
    """Penalisation, priors and grouping do not create information."""
    box = _box(kind)
    assert "information limit" in box
    assert "do not create information" in box
    # And the exception, which is the actionable half.
    assert "permutation test is the exception" in box


def test_a_backend_that_is_not_recommended_gets_no_badge_and_no_caveat():
    plain = [k for k in REGRESSION_TYPES if k not in RECOMMENDED_FOR_SCREENS]
    assert plain, "every backend is recommended, which recommends nothing"
    for kind in plain:
        assert BADGE not in _box(kind)


def test_the_caveat_is_written_once():
    """Six copies of a sentence is six things to keep in step."""
    import inspect

    from spacr.qt.screens import settings_model

    source = inspect.getsource(settings_model)
    # The text appears in the constant and is REFERENCED elsewhere, never
    # retyped.
    assert source.count("do not create information") == 1
    assert INFORMATION_LIMIT_NOTE
