"""What an axis cutoff says about itself, and what a cleared one leaves behind.

A cutoff is offered to the user through the axis menu, and the menu quotes it
back -- "Clear cutoffs (≤ 500)" -- so the wording of a ONE-SIDED cutoff is
the part the user actually reads. The two open-topped and open-bottomed
spellings are the common ones: cutting a long tail off the bottom while
letting the top follow the data is the gesture the control exists for.

The store around them is a container, and the container's promise is that a
cutoff is either present or absent -- never stored as an empty pair -- so
"which measurements are cut off?" has exactly one answer whichever way it is
asked.
"""
from __future__ import annotations

import pytest

from spacr.qt.widgets.gate_canvas import (
    AxisCutoff, AxisCutoffs, axis_menu_items,
)


def test_a_one_sided_cutoff_reads_as_an_inequality_not_a_range():
    """An open end is quoted as ``≥``/``≤``, not as a range with a guess in it.

    The menu prints this string. A high-only cutoff rendered as a range would
    have to invent the low end, and the number it invented would read as
    something the user had set.
    """
    assert AxisCutoff(low=10).describe() == "≥ 10"
    assert AxisCutoff(high=500).describe() == "≤ 500"
    assert AxisCutoff(low=10, high=500).describe() == "10 – 500"


def test_an_unset_cutoff_describes_itself_as_none():
    """Nothing pinned reads as ``none`` rather than as a blank string."""
    empty = AxisCutoff()
    assert empty.is_set is False
    assert empty.describe() == "none"


def test_the_axis_menu_quotes_a_high_only_cutoff_on_its_clearing_row():
    """The clearing row names what it would undo, in the one-sided wording.

    The row is the only place the current cutoff appears once the dialog is
    shut, so a user who has forgotten what they set reads it here.
    """
    rows = axis_menu_items("y", "area", cutoff=AxisCutoff(high=500),
                           on_clear=lambda: None)
    clearing = [row for row in rows if row.label
                and row.label.startswith("Clear cutoffs")][0]
    assert clearing.label == "Clear cutoffs (≤ 500)"
    assert clearing.enabled is True
    assert clearing.why == ""


def test_clearing_one_measurement_takes_it_out_of_the_listing():
    """A cleared column stops being reported as cut off, by every route."""
    cutoffs = AxisCutoffs(initial={"area": AxisCutoff(low=10)})
    assert "area" in cutoffs

    assert cutoffs.clear("area") is True
    assert cutoffs.columns() == ()
    assert list(cutoffs) == []
    assert "area" not in cutoffs
    assert cutoffs.clear("area") is False


