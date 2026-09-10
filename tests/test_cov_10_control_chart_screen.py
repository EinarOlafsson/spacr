"""The Control Charts screen when the form, or the file, is not cooperating.

The screen's contract is that the picture always agrees with the form and
that a refusal is a sentence on screen rather than a traceback. The branches
here are the ones where something has to be re-derived or given up on: a
control column changed under the level list, a spec the engine will not
accept, a database with more than one table in it, and the three file
choosers a user can cancel. Each has to leave a screen that still works.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

pytestmark = pytest.mark.qt

from PySide6.QtWidgets import QFileDialog                # noqa: E402

from spacr.qt.screens.control_chart import (            # noqa: E402
    ControlChartCanvas, ControlChartScreen)


def _campaign(plates: int = 12) -> pd.DataFrame:
    rows = []
    for index in range(plates):
        for level, base in (("neg", 100.0), ("pos", 20.0)):
            for offset in (-1.0, 1.0):
                rows.append({"plateID": f"P{index + 1:02d}",
                             "run_order": index,
                             "well_type": level,
                             "signal": base + offset})
    return pd.DataFrame(rows)


@pytest.fixture
def screen(qtbot):
    made = ControlChartScreen(threaded=False)
    qtbot.addWidget(made)
    return made


def test_a_canvas_with_nothing_drawn_reports_no_result(qtbot):
    """The host asks the canvas what it is showing before it offers an
    export. A fresh canvas has to say "nothing" rather than the last
    screen's chart."""
    canvas = ControlChartCanvas()
    qtbot.addWidget(canvas)
    assert canvas.result is None


def test_reloading_the_same_table_keeps_the_columns_the_user_chose(screen):
    """A second read of the same file must not throw the form back to its
    guesses; the user's picks are still valid for the same columns."""
    frame = _campaign()
    screen.set_frame(frame)
    screen._order.setCurrentText("plateID")
    screen.set_frame(frame)
    assert screen._order.currentText() == "plateID"
    assert screen._plate.currentText() == "plateID"


def test_changing_the_control_column_rebuilds_its_levels(screen):
    """The tick list and the two Z' pickers all name levels OF the control
    column. Leaving the old column's levels behind would offer values that
    are not in the new column at all."""
    frame = _campaign()
    frame["batch"] = ["A", "B"] * (len(frame) // 2)
    screen.set_frame(frame)
    screen._positive.setCurrentText("pos")

    screen._control_column.setCurrentText("batch")
    screen._on_control_column("batch")
    levels = [screen._levels.item(i).text()
              for i in range(screen._levels.count())]
    assert levels == ["A", "B"]

    screen._control_column.setCurrentText("well_type")
    screen._on_control_column("well_type")
    assert [screen._levels.item(i).text()
            for i in range(screen._levels.count())] == ["neg", "pos"]

    # A refill that does NOT change the column keeps the Z' picker's choice:
    # "pos" is still a level of well_type, and clearing it would make every
    # unrelated edit to the form throw away the named positive control.
    screen._positive.setCurrentText("pos")
    screen._refill_levels(frame)
    assert screen._positive.currentText() == "pos"


def test_a_form_the_engine_refuses_becomes_a_sentence_not_a_traceback(screen):
    """The spec re-validates independently of what the widgets allow, and it
    is the authority. Its refusal has to reach the report line, with no chart
    left on screen claiming to be the answer."""
    screen.set_frame(_campaign())
    assert screen.result is not None
    screen._baseline.setRange(1, 500)
    screen._baseline.setValue(1)
    screen.recompute()
    assert screen.result is None
    assert "is not a baseline" in screen.report.toPlainText()


def _database(tmp_path):
    path = str(tmp_path / "measurements.db")
    frame = _campaign()
    with sqlite3.connect(path) as db:
        frame.to_sql("cell", db, index=False)
        frame.to_sql("nucleus", db, index=False)
    return path


def test_opening_a_named_table_selects_it_in_the_picker(screen, tmp_path):
    """A caller that already knows which table it wants -- a run folder, a
    restored session -- must land on that table, and the picker has to show
    the one that was read."""
    path = _database(tmp_path)
    screen.load_path(path, table="nucleus")
    assert screen._table_picker.currentText() == "nucleus"
    assert screen._table_picker.isVisible() or not screen.isVisible()
    assert screen.result is not None


def test_picking_another_table_rereads_the_same_file(screen, tmp_path):
    """The picker is the only way to look at a second table in one database;
    if it did not re-read, it would relabel the chart of the first."""
    path = _database(tmp_path)
    screen.load_path(path, table="cell")
    screen._on_table_picked("nucleus")
    assert screen._path == path
    assert "nucleus" in screen._source.text()


def test_a_cancelled_open_leaves_the_screen_as_it_was(screen, monkeypatch):
    """Cancelling the chooser must not clear the chart already drawn."""
    screen.set_frame(_campaign())
    drawn = screen.result
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    screen.choose_table()
    assert screen.result is drawn


def test_choosing_a_file_from_the_dialog_loads_it(screen, monkeypatch,
                                                  tmp_path):
    """The chooser is the ordinary way in; picking a file has to reach the
    same load path a caller uses directly."""
    csv = tmp_path / "campaign.csv"
    _campaign().to_csv(csv, index=False)
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(csv), "")))
    screen.choose_table()
    assert screen.result is not None
    assert "campaign.csv" in screen._source.text()


def test_a_cancelled_export_writes_nothing(screen, monkeypatch, tmp_path):
    """Cancelling must not write a file under the default name in whatever
    directory the app happens to be running from."""
    screen.set_frame(_campaign())
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    screen.choose_export()
    assert not list(tmp_path.glob("*.csv"))


def test_choosing_an_export_destination_writes_the_points(screen, monkeypatch,
                                                          tmp_path):
    """One row per plate is what a reader takes away from this screen; the
    chooser has to hand that path to the writer unchanged."""
    screen.set_frame(_campaign())
    target = tmp_path / "points.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    screen.choose_export()
    written = pd.read_csv(target)
    assert len(written) == 12
    assert "points.csv" in screen._source.text()
