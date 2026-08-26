"""The Runs table's context menu, and the row it must refuse to act on.

Every entry on that menu acts on a RECORD -- load this run, open it beside,
remove it, delete its folder from disk. The rows are read back through the
panel's frame rather than off the table, because the table holds display
strings and a folder path is not something to reconstruct from one.

So a row the frame cannot supply a record for is a row the menu has nothing
to act on. Building it anyway would offer "Delete from disk" over a run the
panel could not name, which is the one entry that must never fire at a guess.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from spacr.qt.widgets.sweep_runs import SweepRunsPanel           # noqa: E402


class _StubMenu:
    """A menu that is never entered, so no C++ event loop is started."""

    def __init__(self):
        self.shown_at = None

    def exec(self, position):
        self.shown_at = position
        return None


def _panel(qtbot):
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    return panel


def _watch_the_menu(panel, monkeypatch):
    """Record every set of records a menu was built for."""
    built = []

    def _build(records):
        built.append(list(records))
        return _StubMenu()

    monkeypatch.setattr(panel, "_build_run_menu", _build)
    return built


def _centre_of_first_row(panel):
    item = panel.table.table.item(0, 0)
    assert item is not None, "the table has no row to right-click"
    return panel.table.table.visualItemRect(item).center()


def test_a_right_click_on_a_recorded_run_opens_its_menu(qtbot, monkeypatch):
    """The control: a row the frame can name gets the menu it is entitled to."""
    panel = _panel(qtbot)
    built = _watch_the_menu(panel, monkeypatch)
    panel.record_run("ols_1", folder="/x/ols_1")

    panel._run_menu(_centre_of_first_row(panel))

    assert len(built) == 1
    assert built[0][0]["run"] == "ols_1"


def test_a_row_the_frame_cannot_name_gets_no_menu(qtbot, monkeypatch):
    """A table filled from somewhere other than the panel's own frame.

    The rows are real Qt items and `itemAt` finds one, so selecting it is
    tried -- and the selection still yields no record, because the frame the
    records are read back from does not have it. Nothing may be offered over
    a run that cannot be identified.
    """
    panel = _panel(qtbot)
    built = _watch_the_menu(panel, monkeypatch)
    panel.table.set_frame(pd.DataFrame({"run": ["ghost"], "status": ["ok"]}))
    assert panel.table.table.rowCount() == 1
    assert panel.selected_runs() == []

    panel._run_menu(_centre_of_first_row(panel))

    assert built == [], "a menu was offered over a row with no run behind it"


def test_a_right_click_on_empty_space_below_the_rows_opens_no_menu(
        qtbot, monkeypatch):
    from PySide6.QtCore import QPoint

    panel = _panel(qtbot)
    built = _watch_the_menu(panel, monkeypatch)
    panel.record_run("ols_1", folder="/x/ols_1")
    panel.table.table.clearSelection()

    panel._run_menu(QPoint(4, 100000))

    assert built == []
