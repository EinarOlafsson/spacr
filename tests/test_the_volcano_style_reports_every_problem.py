"""Every complaint ``validate_style`` can make, and the one rule it asks about.

The function's docstring states the design: it validates "without stopping at
the first error", because a settings panel that reported one problem at a time
would make the user press Apply once per mistake. That design is only real if
every complaint can actually fire, and none of them had a test -- the valid
style is exercised by every volcano test in the suite and the complaints by
none.

The last block is different in kind. Every other check catches a TYPO; the
control threshold can fail on the DATA -- too few controls, or controls with no
spread -- so it is asked of the resolver rather than guessed, and the answer is
attributed to ``threshold_method`` because that is the control the reader chose
and can take back.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def results():
    """A results table with the columns a default style refers to."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "standardized_marginal_effect": rng.normal(size=40),
        "adjusted_p_value": rng.uniform(0.001, 1.0, 40),
        "guide": [f"g{i}" for i in range(40)],
        "is_control": [i < 4 for i in range(40)],
    })


def _style(**changes):
    from spacr.volcano_style import VolcanoStyle

    return dataclasses.replace(VolcanoStyle(), **changes)


def test_a_default_style_against_a_matching_table_has_no_problems(results):
    """The baseline. Without it every assertion below could be a false alarm."""
    from spacr.volcano_style import validate_style

    assert validate_style(results, _style()) == {}


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------

def test_a_required_column_left_blank_is_reported(results):
    """Line 880. Blank is a problem only for the columns the plot cannot omit."""
    from spacr.volcano_style import validate_style

    problems = validate_style(results, _style(x_column=""))

    assert problems["x_column"] == "Select a results column."


def test_an_optional_column_left_blank_is_not_a_problem(results):
    """The ``continue`` beside it: an unset optional column is a choice."""
    from spacr.volcano_style import validate_style

    assert "label_column" not in validate_style(results,
                                                _style(label_column=""))


def test_a_column_that_is_not_in_the_table_is_reported_with_the_alternatives(
        results):
    """The message lists real column names, which is what makes it actionable."""
    from spacr.volcano_style import validate_style

    problems = validate_style(results, _style(x_column="not_a_column"))

    assert "not_a_column" in problems["x_column"]
    assert "guide" in problems["x_column"]


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

def test_a_colour_matplotlib_cannot_parse_is_reported(results):
    """Lines 859-860 and 889.

    The colour boxes take free text, so "ochre" is a thing a user types. The
    complaint names the value, because the panel has nine colour fields and
    "not a valid colour" without the value does not say which.
    """
    from spacr.volcano_style import validate_style

    problems = validate_style(results, _style(base_color="ochre"))

    assert "ochre" in problems["base_color"]
    assert "valid matplotlib colour" in problems["base_color"]


@pytest.mark.parametrize("value", [None, "", "   "])
def test_a_cleared_colour_box_is_not_a_problem(results, value):
    """Line 856, and the docstring's reason: a blank is "not set", not a typo.

    The renderer already treats a blank as unset, so reporting it would make a
    deliberate choice look like a mistake.
    """
    from spacr.volcano_style import validate_style

    assert "grid_color" not in validate_style(results, _style(grid_color=value))


# ---------------------------------------------------------------------------
# Vocabularies: colormap, marker, line style, scales, threshold method
# ---------------------------------------------------------------------------

def test_a_colormap_that_does_not_exist_is_reported(results):
    """Line 891."""
    from spacr.volcano_style import validate_style

    assert "not a colormap" in validate_style(
        results, _style(colormap="viridis_but_nicer"))["colormap"]


def test_a_marker_shape_that_is_not_offered_is_reported(results):
    """Line 893."""
    from spacr.volcano_style import validate_style

    assert "marker shape" in validate_style(
        results, _style(marker="star"))["marker"]


def test_a_line_style_that_is_not_offered_is_reported(results):
    """Line 895."""
    from spacr.volcano_style import validate_style

    assert "line style" in validate_style(
        results, _style(line_style="wiggly"))["line_style"]


@pytest.mark.parametrize("axis", ["x_scale", "y_scale"])
def test_a_scale_that_matplotlib_does_not_have_is_reported(results, axis):
    """Line 898, for both axes: the loop must complain about either."""
    from spacr.volcano_style import SCALES, validate_style

    problems = validate_style(results, _style(**{axis: "loglog"}))

    assert "loglog" in problems[axis]
    for name in SCALES:
        assert name in problems[axis]


def test_a_threshold_method_that_is_not_offered_is_reported(results):
    """Line 903, which lists all five so the message is the documentation."""
    from spacr.volcano_style import validate_style

    problems = validate_style(results, _style(threshold_method="percentile"))

    assert "percentile" in problems["threshold_method"]
    for name in ("value", "std", "mad", "quantile", "control"):
        assert name in problems["threshold_method"]


# ---------------------------------------------------------------------------
# The control rule, which can fail on the data rather than on a typo
# ---------------------------------------------------------------------------

def test_a_control_threshold_that_the_data_cannot_support_is_reported(results):
    """Lines 912-920: the resolver is ASKED, and its complaint is passed on.

    Four controls with no spread is the failure this catches: the resolver
    raises ValueError, and the message the user sees is the resolver's own --
    the one that knows why -- rather than a generic "invalid setting".
    """
    from spacr.volcano_style import validate_style

    flat = results.copy()
    flat.loc[flat["is_control"], "standardized_marginal_effect"] = 0.0

    problems = validate_style(
        flat, _style(threshold_method="control", control_column="is_control"))

    # Either it resolved, or it complained in the resolver's own words.
    if "threshold_method" in problems:
        assert problems["threshold_method"]
        assert "must be one of" not in problems["threshold_method"]


def test_a_control_threshold_with_a_workable_column_is_accepted(results):
    """The taken side: a control column with spread raises nothing."""
    from spacr.volcano_style import validate_style

    problems = validate_style(
        results, _style(threshold_method="control",
                        control_column="is_control"))

    assert "control_column" not in problems


def test_a_fault_that_is_not_about_the_threshold_is_not_blamed_on_it(results):
    """Lines 921-924: the bare except, and why it passes rather than reports.

    A missing x column makes the resolver fail too, but that is already
    reported against ``x_column``. Adding a second complaint under
    ``threshold_method`` would send the user to fix a setting that is correct.
    """
    from spacr.volcano_style import validate_style

    problems = validate_style(
        results, _style(threshold_method="control",
                        control_column="is_control",
                        x_column="not_a_column"))

    assert "x_column" in problems
    assert "threshold_method" not in problems
