"""Recovering summary rows for reflow, and clearing the summary panel body.

`split_rows` recovers the fixed-width label/value rows of a written summary:
wrapped values are joined back onto their label, prose lines become labelless
rows, blank lines never become rows, and a body with no label column is not
rows at all.  Clearing the panel keeps its action row and is repeatable.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.folding_summary import FoldingSummaryView, split_rows  # noqa: E402

pytestmark = pytest.mark.qt


def test_split_rows_recovers_label_value_rows(subtests):
    """Exact rows for plain, wrapped, prose, blank-line and empty bodies."""
    for case, body, expected in (
            ("a plain two-column body", "  plate      plate1\n  wells      96\n",
             [("plate", "plate1"), ("wells", "96")]),
            ("no label column is not rows", "just a sentence about the run", []),
            ("an empty body", "", []),
            ("no body", None, []),
            ("a wrapped value joins its label",
             "  plate      plate1\n"
             "  note       a value long enough that the writer\n"
             "             wrapped it onto a second line\n"
             "  wells      96\n",
             [("plate", "plate1"),
              ("note", "a value long enough that the writer wrapped it onto "
                       "a second line"),
              ("wells", "96")]),
            ("several continuations join in order",
             "  note       one\n             two\n             three\n",
             [("note", "one two three")]),
            ("a line before any label is a labelless row",
             "preamble\n  plate      plate1\n",
             [("", "preamble"), ("plate", "plate1")]),
            ("a line too short to be a row is kept as its own line",
             "  effect        0.42\nshort\n", [("effect", "0.42"), ("", "short")]),
            ("blank lines never become rows",
             "  effect          0.42\n\n   \n  p value         0.01\n",
             [("effect", "0.42"), ("p value", "0.01")]),
    ):
        with subtests.test(msg=case):
            assert split_rows(body) == expected


def test_clearing_the_summary_body_keeps_the_action_row(qtbot, subtests):
    """_clear empties the sections, keeps Copy/Save, and copes with no action row."""
    with subtests.test(msg="the action row survives a clear, twice"):
        view = FoldingSummaryView()
        qtbot.addWidget(view)
        view.setPlainText("  effect        0.42\n")
        actions = view._actions
        assert actions is not None
        view._clear()
        assert view._layout.indexOf(actions) >= 0
        assert view._sections == []
        view._clear()
        assert view._layout.indexOf(actions) >= 0
        assert view._sections == []

    with subtests.test(msg="a panel with no action row still clears"):
        view = FoldingSummaryView()
        qtbot.addWidget(view)
        view._actions = None
        view._clear()
        assert view._sections == []
