"""Tests for the process-wide linked selection.

The behaviours worth pinning are the ones that would make two open views
disagree about what the user is looking at, and the one that would make a lasso
destructive.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.qt.linked_selection import LinkedSelection, linked_selection
from spacr.selection import CategoryFilter, DataFilter, RangeFilter, Selection


@pytest.fixture
def link() -> LinkedSelection:
    """A fresh instance — never the process-wide one, which other tests share."""
    return LinkedSelection()


def _frame(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({
        "plateID": ["p1"] * n,
        "rowID": [f"r{i % 2 + 1}" for i in range(n)],
        "columnID": [f"c{i % 3 + 1}" for i in range(n)],
        "fieldID": ["f1"] * n,
        "object_label": list(range(1, n + 1)),
        "area": [10.0 * (i + 1) for i in range(n)],
    })


def test_the_accessor_is_a_singleton():
    assert linked_selection() is linked_selection()


def test_it_starts_with_no_filter_and_no_selection(link):
    assert link.filter.is_empty
    assert not link.selection.is_active


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def test_setting_a_filter_emits_only_filter_changed(link, qtbot):
    """The two signals cost different amounts to honour, so they stay apart.

    A filter change makes a view re-query and re-lay-out; a selection change
    usually only repaints. One combined signal would make every lasso reload a
    million-row table.
    """
    seen = {"filter": 0, "selection": 0}
    link.filter_changed.connect(lambda: seen.__setitem__("filter",
                                                         seen["filter"] + 1))
    link.selection_changed.connect(
        lambda: seen.__setitem__("selection", seen["selection"] + 1))

    link.set_filter(DataFilter().add(RangeFilter("area", low=20.0)))
    assert seen == {"filter": 1, "selection": 0}


def test_setting_a_selection_emits_only_selection_changed(link):
    seen = {"filter": 0, "selection": 0}
    link.filter_changed.connect(lambda: seen.__setitem__("filter",
                                                         seen["filter"] + 1))
    link.selection_changed.connect(
        lambda: seen.__setitem__("selection", seen["selection"] + 1))

    link.select_frame(_frame(3), source="umap")
    assert seen == {"filter": 0, "selection": 1}


def test_an_identical_looking_filter_still_emits(link):
    """No equality short-circuit, deliberately.

    A caller that mutated a DataFilter in place and handed the same object
    back would compare equal to itself and emit nothing, leaving views showing
    a population that no longer matches the controls.
    """
    f = DataFilter().add(RangeFilter("area", low=20.0))
    link.set_filter(f)

    fired = []
    link.filter_changed.connect(lambda: fired.append(1))
    f.add(RangeFilter("area", low=50.0))     # mutated in place
    link.set_filter(f)                        # same object back
    assert fired == [1], "a re-set must emit even when the object is unchanged"


def test_clearing_emits_too(link):
    link.set_filter(DataFilter().add(RangeFilter("area", low=20.0)))
    link.select_frame(_frame(2), source="plate")

    fired = {"filter": 0, "selection": 0}
    link.filter_changed.connect(
        lambda: fired.__setitem__("filter", fired["filter"] + 1))
    link.selection_changed.connect(
        lambda: fired.__setitem__("selection", fired["selection"] + 1))

    link.clear_filter()
    link.clear_selection()
    assert fired == {"filter": 1, "selection": 1}
    assert link.filter.is_empty
    assert not link.selection.is_active


# ---------------------------------------------------------------------------
# What a view actually asks for
# ---------------------------------------------------------------------------

def test_visible_applies_the_filter(link):
    df = _frame(6)
    link.set_filter(DataFilter().add(RangeFilter("area", low=40.0)))
    out = link.visible(df)
    assert len(out) == 3
    assert (out["area"] >= 40.0).all()


def test_visible_does_not_apply_the_selection(link):
    """A selection HIGHLIGHTS; it must never hide.

    A view that dropped unselected rows would make the lasso destructive — you
    could not see what you had excluded, or undo it by lassoing more.
    """
    df = _frame(6)
    link.select_frame(df.iloc[[0]], source="umap")
    assert len(link.visible(df)) == 6


def test_a_filter_and_a_selection_compose_without_interfering(link):
    df = _frame(6)
    link.set_filter(DataFilter().add(CategoryFilter("rowID", ("r1",))))
    link.select_frame(df.iloc[[0, 1]], source="umap")

    shown = link.visible(df)
    assert set(shown["rowID"]) == {"r1"}
    # The selection still resolves against the narrowed frame.
    assert link.selection.mask_for(shown).sum() == 1


def test_the_selection_carries_its_source(link):
    """So a view can ignore the echo of its own selection.

    Without it every lasso costs a repaint in the view that drew it, and a
    view that normalises what it publishes can loop.
    """
    link.select_frame(_frame(2), source="plate_view")
    assert link.selection.source == "plate_view"


def test_set_selection_accepts_a_prebuilt_selection(link):
    sel = Selection.from_frame(_frame(3), source="db_browser")
    link.set_selection(sel)
    assert link.selection is sel
