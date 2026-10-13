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


def test_the_store_lists_its_measurements_in_the_order_they_were_cut():
    """``columns()`` and iteration agree, and both follow insertion order.

    The order is what a "cut off: area, intensity" summary reads out, so a
    dict that reshuffled would make the same session read differently twice.
    """
    cutoffs = AxisCutoffs()
    cutoffs.set("area", low=10)
    cutoffs.set("intensity", high=500)
    cutoffs.set("perimeter", low=1, high=2)

    assert cutoffs.columns() == ("area", "intensity", "perimeter")
    assert list(cutoffs) == ["area", "intensity", "perimeter"]
    assert len(cutoffs) == 3


def test_clearing_one_measurement_takes_it_out_of_the_listing():
    """A cleared column stops being reported as cut off, by every route."""
    cutoffs = AxisCutoffs(initial={"area": AxisCutoff(low=10)})
    assert "area" in cutoffs

    assert cutoffs.clear("area") is True
    assert cutoffs.columns() == ()
    assert list(cutoffs) == []
    assert "area" not in cutoffs
    assert cutoffs.clear("area") is False


def test_clearing_everything_reports_how_many_cutoffs_went():
    """``clear_all`` answers the count, so a "reset" can say what it undid.

    Answering ``None`` would leave the caller unable to tell a reset that
    changed the picture from one that did nothing at all.
    """
    cutoffs = AxisCutoffs()
    cutoffs.set("area", low=10)
    cutoffs.set("intensity", high=500)

    assert cutoffs.clear_all() == 2
    assert len(cutoffs) == 0
    assert cutoffs.columns() == ()
    assert cutoffs.clear_all() == 0


def test_setting_both_ends_to_nothing_forgets_the_measurement():
    """An empty cutoff is absent, not stored -- the container's one rule."""
    cutoffs = AxisCutoffs()
    cutoffs.set("area", low=10)
    stored = cutoffs.set("area", low=None, high=None)

    assert stored.is_set is False
    assert cutoffs.columns() == ()
    assert cutoffs.get("area").describe() == "none"
