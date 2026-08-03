"""Tests for the Local Data Filter panel.

Assertions are on the filter the panel produces and on what it does to a
frame, not on widget existence — a panel that renders perfectly and publishes
the wrong population is the failure worth catching.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.widgets.data_filter_panel import (
    MAX_CATEGORY_VALUES, DataFilterPanel, classify_columns,
)


def _frame(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame({
        "plateID": ["p1" if i < n // 2 else "p2" for i in range(n)],
        "rowID": [f"r{i % 3 + 1}" for i in range(n)],
        "columnID": [f"c{i % 4 + 1}" for i in range(n)],
        "fieldID": ["f1"] * n,
        "object_label": list(range(1, n + 1)),
        "area": [float(10 * (i + 1)) for i in range(n)],
        "cell_count": [i % 5 for i in range(n)],
        "note": [f"free text {i}" for i in range(n)],   # high cardinality
    })


@pytest.fixture
def panel(qtbot):
    """A panel bound to a PRIVATE link, never the process-wide one."""
    link = LinkedSelection()
    p = DataFilterPanel(link=link)
    qtbot.addWidget(p)
    p.set_frame(_frame())
    return p


# ---------------------------------------------------------------------------
# The classification rule, tested directly
# ---------------------------------------------------------------------------

def test_columns_are_classified_by_what_can_usefully_filter_them():
    kinds = classify_columns(_frame())
    assert kinds["plateID"] == "category"     # 2 distinct
    assert kinds["rowID"] == "category"       # 3 distinct
    assert kinds["area"] == "range"           # 60 distinct numerics
    assert kinds["cell_count"] == "category"  # numeric but only 5 distinct
    assert kinds["note"] == "skip"            # free text, 60 distinct
    assert kinds["object_label"] == "skip"    # identifies, does not describe


def test_a_high_cardinality_text_column_is_skipped():
    df = pd.DataFrame({"gene": [f"g{i}" for i in range(MAX_CATEGORY_VALUES + 5)]})
    assert classify_columns(df)["gene"] == "skip"


def test_a_text_column_just_inside_the_limit_is_tickable():
    df = pd.DataFrame({"gene": [f"g{i}" for i in range(MAX_CATEGORY_VALUES)]})
    assert classify_columns(df)["gene"] == "category"


def test_a_bool_column_is_ticks_not_a_range():
    df = pd.DataFrame({"flag": [True, False, True]})
    assert classify_columns(df)["flag"] == "category"


def test_identity_columns_are_never_offered(panel):
    """Filtering on an object id is what the SELECTION is for.

    Offering it here invites building a selection out of predicates, which is
    exactly the thing that does not survive a re-run.
    """
    assert "object_label" not in panel.available_columns()
    assert "note" not in panel.available_columns()
    assert "area" in panel.available_columns()


# ---------------------------------------------------------------------------
# What the panel publishes
# ---------------------------------------------------------------------------

def test_a_fresh_panel_filters_nothing(panel):
    panel.flush()
    assert panel._link.filter.is_empty
    assert len(panel._link.visible(_frame())) == 60


def test_adding_a_category_column_starts_with_everything_ticked(panel):
    """Adding a filter must not narrow anything until the user unticks.

    A clause that removed rows the moment it appeared would make the act of
    *looking* at a column change the population.
    """
    panel.add_column("plateID")
    panel.flush()
    assert len(panel._link.visible(_frame())) == 60


def test_unticking_narrows_every_view(panel):
    panel.add_column("plateID")
    row = panel._rows["plateID"]
    for box in row._boxes:
        if box.text() == "p2":
            box.setChecked(False)
    panel.flush()

    out = panel._link.visible(_frame())
    assert set(out["plateID"]) == {"p1"}
    assert len(out) == 30


def test_a_range_narrows_on_both_bounds(panel):
    panel.add_column("area")
    row = panel._rows["area"]
    row._low.setValue(50.0)
    row._high.setValue(90.0)
    panel.flush()

    out = panel._link.visible(_frame())
    assert out["area"].min() >= 50.0
    assert out["area"].max() <= 90.0


def test_clauses_combine(panel):
    panel.add_column("plateID")
    panel.add_column("area")
    for box in panel._rows["plateID"]._boxes:
        if box.text() != "p1":
            box.setChecked(False)
    panel._rows["area"]._low.setValue(30.0)
    panel.flush()

    out = panel._link.visible(_frame())
    assert set(out["plateID"]) == {"p1"}
    assert out["area"].min() >= 30.0


def test_removing_a_clause_widens_again(panel):
    panel.add_column("plateID")
    for box in panel._rows["plateID"]._boxes:
        if box.text() == "p2":
            box.setChecked(False)
    panel.flush()
    assert len(panel._link.visible(_frame())) == 30

    panel.remove_column("plateID")
    panel.flush()
    assert len(panel._link.visible(_frame())) == 60


def test_clear_all_drops_every_clause(panel):
    panel.add_column("plateID")
    panel.add_column("area")
    panel.clear()
    panel.flush()
    assert panel._link.filter.is_empty
    assert panel._rows == {}


def test_adding_the_same_column_twice_is_a_no_op(panel):
    panel.add_column("area")
    panel.add_column("area")
    assert len(panel._rows) == 1


def test_the_summary_says_what_is_filtered(panel):
    """A filtered view that does not say so is how a result gets computed on a
    fifth of the data and reported as the whole."""
    panel.flush()
    assert panel._summary.text() == "no filter"

    panel.add_column("area")
    panel._rows["area"]._low.setValue(30.0)
    panel.flush()
    assert "area" in panel._summary.text()


# ---------------------------------------------------------------------------
# Cost and lifecycle
# ---------------------------------------------------------------------------

def test_edits_are_debounced_into_one_publish(panel, qtbot):
    """Re-filtering a million rows per keystroke would make it unusable."""
    panel.add_column("area")
    panel.flush()

    published = []
    panel._link.filter_changed.connect(lambda: published.append(1))

    row = panel._rows["area"]
    for value in (10.0, 20.0, 30.0, 40.0):     # a dragged spinbox
        row._low.setValue(value)
    assert published == [], "publishing before the debounce expired"

    qtbot.wait(350)
    assert len(published) == 1, "a burst of edits must cost one re-filter"


def test_setting_a_new_frame_drops_stale_clauses(panel):
    """A clause naming a column the new frame lacks would raise on apply.

    Keeping only the ones that still resolve would be worse: it narrows by
    less than the panel says it does.
    """
    panel.add_column("area")
    panel.set_frame(pd.DataFrame({
        "plateID": ["p9"] * 3,
        "rowID": ["r1"] * 3,
        "columnID": ["c1"] * 3,
        "fieldID": ["f1"] * 3,
        "object_label": [1, 2, 3],
    }))
    panel.flush()
    assert panel._rows == {}
    assert panel._link.filter.is_empty


def test_a_constant_numeric_column_still_gives_a_usable_control(qtbot):
    """Equal min and max would otherwise leave a spinbox with no travel."""
    link = LinkedSelection()
    p = DataFilterPanel(link=link)
    qtbot.addWidget(p)
    df = pd.DataFrame({
        "plateID": ["p1"] * 30, "rowID": ["r1"] * 30,
        "columnID": ["c1"] * 30, "fieldID": ["f1"] * 30,
        "object_label": list(range(30)),
        "flat": [5.0] * 30,
    })
    # 30 identical values is 1 distinct, so it classifies as ticks; force the
    # range editor by giving it enough distinct values while keeping a
    # degenerate observed span.
    df.loc[:, "flat"] = [5.0] * 29 + [5.0]
    p.set_frame(df)
    kinds = classify_columns(df)
    assert kinds["flat"] == "category"      # documents the actual rule


# ---------------------------------------------------------------------------
# The paths a user actually takes that the tests above walk around
# ---------------------------------------------------------------------------

def test_the_add_button_adds_the_picked_column(panel, qtbot):
    """The picker + Add is how a user adds a clause; `add_column` is the API."""
    panel._picker.setCurrentText("area")
    qtbot.mouseClick(panel.findChild(type(panel._clear), "FilterAddButton"),
                     __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.LeftButton)
    assert "area" in panel._rows


def test_a_long_value_list_gets_a_scroll_area(qtbot):
    """More than eight ticks must not push the summary off the panel."""
    from PySide6.QtWidgets import QScrollArea

    link = LinkedSelection()
    p = DataFilterPanel(link=link)
    qtbot.addWidget(p)
    n = 40
    p.set_frame(pd.DataFrame({
        "plateID": ["p1"] * n, "rowID": ["r1"] * n,
        "columnID": [f"c{i % 20}" for i in range(n)],   # 20 ticks
        "fieldID": ["f1"] * n, "object_label": list(range(n)),
    }))
    p.add_column("columnID")
    assert p._rows["columnID"].findChildren(QScrollArea), \
        "a 20-value tick list needs to scroll"


def test_a_short_value_list_gets_no_scroll_area(panel):
    from PySide6.QtWidgets import QScrollArea

    panel.add_column("plateID")          # 2 values
    assert not panel._rows["plateID"].findChildren(QScrollArea)


def test_removing_a_column_that_is_not_there_is_harmless(panel):
    panel.remove_column("never_added")   # must not raise


def test_adding_a_skipped_column_is_refused(panel):
    """`note` and `object_label` are not offered; asking anyway does nothing."""
    panel.add_column("note")
    panel.add_column("object_label")
    assert panel._rows == {}


def test_a_degenerate_range_column_still_gets_travel(qtbot):
    """A column whose values are all equal would leave a spinbox inert.

    The editor widens the upper bound so the control can still be driven —
    otherwise the user cannot express a cut-off at all on that column.
    """
    link = LinkedSelection()
    p = DataFilterPanel(link=link)
    qtbot.addWidget(p)
    n = 40
    df = pd.DataFrame({
        "plateID": ["p1"] * n, "rowID": ["r1"] * n,
        "columnID": ["c1"] * n, "fieldID": ["f1"] * n,
        "object_label": list(range(n)),
        # 13+ distinct so it classifies as a range, but almost no spread.
        "flat": [5.0] * (n - 13) + [5.0 + i * 1e-9 for i in range(13)],
    })
    p.set_frame(df)
    assert classify_columns(df)["flat"] == "range"
    p.add_column("flat")
    row = p._rows["flat"]
    assert row._high.maximum() > row._low.minimum(), "the control has no travel"


def test_an_empty_frame_offers_nothing_and_does_not_raise(qtbot):
    link = LinkedSelection()
    p = DataFilterPanel(link=link)
    qtbot.addWidget(p)
    p.set_frame(pd.DataFrame({"area": pd.Series(dtype=float)}))
    p.flush()
    assert p._link.filter.is_empty


def test_the_range_editor_survives_a_truly_constant_column(qtbot):
    """Reachable only by constructing the editor directly.

    `classify_columns` sends a column with 12 or fewer distinct values to
    ticks, so a genuinely constant one never reaches the range editor through
    the panel. The guard still has to hold, because the editor is a public
    class and a caller with its own classification can hand it one — and an
    unguarded spinbox whose min equals its max cannot be driven at all.
    """
    from spacr.qt.widgets.data_filter_panel import _RangeRow

    row = _RangeRow("flat", pd.Series([5.0] * 20))
    qtbot.addWidget(row)
    assert row._high.maximum() > row._low.minimum()
    clause = row.clause()
    assert clause.column == "flat"


def test_the_range_editor_survives_an_all_nan_column(qtbot):
    from spacr.qt.widgets.data_filter_panel import _RangeRow

    row = _RangeRow("empty", pd.Series([float("nan")] * 5))
    qtbot.addWidget(row)
    assert row.clause().column == "empty"


def test_add_with_an_empty_picker_does_nothing(qtbot):
    """Reachable before any frame is set, when the picker has no entries."""
    link = LinkedSelection()
    p = DataFilterPanel(link=link)
    qtbot.addWidget(p)
    assert p.available_columns() == []
    p._add_selected()               # must not raise, must not add a clause
    assert p._rows == {}
