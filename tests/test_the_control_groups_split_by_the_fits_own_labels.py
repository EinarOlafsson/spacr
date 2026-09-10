"""Splitting a coefficient table into control groups for a fast plot.

The docstring gives the rule and its reason: the labels match
``spacr.figures.panels.control_separation`` exactly -- ``nc`` is "negative",
``pc`` is "positive" -- because a control called "negative" on screen and "nc"
in the paper figure is the same disagreement in a smaller place.

The uncovered arc is a table with no ``feature`` column, which is what an
aggregated or hand-built frame looks like: the values are still plottable, the
per-point keys simply are not there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_the_labels_match_the_paper_figures_names():
    """The mapping the docstring promises, asserted rather than assumed."""
    from spacr.figures.fast_render import _control_groups

    frame = pd.DataFrame({
        "condition": ["nc", "nc", "pc", "other"],
        "effect": [0.1, 0.2, 1.0, 0.5],
        "feature": ["g1", "g2", "g3", "g4"],
    })

    groups, keys = _control_groups(frame, "effect")

    assert set(groups) == {"negative", "positive", "screen"}
    assert list(groups["negative"]) == [0.1, 0.2]
    assert keys["negative"] == ["g1", "g2"]


def test_a_group_with_no_rows_is_left_out_entirely():
    """An absent label is not an empty group.

    An empty group would draw an empty violin on the plot, which reads as a
    control that was measured and found to be nothing.
    """
    from spacr.figures.fast_render import _control_groups

    frame = pd.DataFrame({"condition": ["nc", "nc"], "effect": [0.1, 0.2],
                          "feature": ["g1", "g2"]})

    groups, _keys = _control_groups(frame, "effect")

    assert set(groups) == {"negative"}


def test_a_table_without_a_feature_column_still_yields_its_groups():
    """Arc 293 -> 289: the values are gathered and the keys are not.

    An aggregated table has effects and no per-guide feature. The plot can
    still draw the distributions -- it just cannot label a clicked point --
    and refusing the whole split for a missing key column would lose the panel.
    """
    from spacr.figures.fast_render import _control_groups

    frame = pd.DataFrame({"condition": ["nc", "pc"], "effect": [0.1, 1.0]})

    groups, keys = _control_groups(frame, "effect")

    assert set(groups) == {"negative", "positive"}
    assert keys == {}


def test_no_condition_column_yields_nothing():
    """The guard above the loop: without labels there are no control groups."""
    from spacr.figures.fast_render import _control_groups

    frame = pd.DataFrame({"effect": [0.1, 0.2]})

    assert _control_groups(frame, "effect") == ({}, {})


def test_no_effect_column_yields_nothing():
    """The other half of the same guard."""
    from spacr.figures.fast_render import _control_groups

    frame = pd.DataFrame({"condition": ["nc"], "effect": [0.1]})

    assert _control_groups(frame, None) == ({}, {})


def test_the_condition_column_is_found_under_any_of_its_names():
    """``condition``, ``control`` and ``class`` are all accepted spellings.

    A fit writes one of the three depending on how the design was built, and
    the panel must not depend on which.
    """
    from spacr.figures.fast_render import _control_groups

    for name in ("condition", "control", "class"):
        frame = pd.DataFrame({name: ["nc"], "effect": [0.1]})
        groups, _keys = _control_groups(frame, "effect")
        assert set(groups) == {"negative"}, name
