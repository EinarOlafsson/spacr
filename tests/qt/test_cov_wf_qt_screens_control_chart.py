"""Control Charts: the five guards that fire when the form is not ready yet.

``tests/qt/test_control_chart_screen.py`` drives the screen once a campaign is
loaded and the pickers name real columns.  This file goes after the other
half — the moments where a widget signal arrives *before* there is anything to
chart, or points at a column that is no longer in the frame:

* a canvas whose matplotlib build has no owned-timer ``cancel_pending_draw``
  (the fix lives in :func:`spacr.qt.widgets.graph_builder._canvas_class`, and
  the screen only asks for it with ``getattr``);
* the control column emptied, so the level list and both Z' pickers have to be
  emptied with it rather than keep offering levels of a column nobody picked;
* the control column changing before any table is loaded;
* a widget edited while ``set_frame`` is still refilling eleven pickers, which
  must not fire eleven charts;
* the table picker changing before a file has been opened.

Every one of these is a "nothing happens" branch, so each test also drives the
neighbouring input that *does* make it happen — a guard whose body is a bare
return passes any test that only checks the quiet half.

Offscreen, no dialogs, no sleeps; the screen runs its jobs inline
(``threaded=False``) so a signal and its consequence are one call apart.
"""
from __future__ import annotations

import os
import sqlite3

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

pytestmark = pytest.mark.qt

from PySide6.QtWidgets import QWidget                     # noqa: E402

from spacr.qt.screens.control_chart import (              # noqa: E402
    ControlChartCanvas, ControlChartScreen)


def _campaign(plates: int = 30) -> pd.DataFrame:
    """Well-level rows for ``plates`` plates: two controls, two wells each.

    The negative control steps up after plate twenty, so a chart drawn from a
    twenty-plate baseline and one drawn from a twelve-plate baseline are
    different pictures — which is what makes "did a recompute happen?"
    answerable by looking at the result rather than by counting calls.
    """
    rows = []
    for index in range(plates):
        drift = 0.0 if index < 20 else 2.5 * (index - 19)
        for level, base in (("neg", 100.0), ("pos", 20.0)):
            for offset in (-1.0, 1.0):
                rows.append({"plateID": f"P{index + 1:02d}",
                             "run_order": index,
                             "well_type": level,
                             "batch": "A" if index % 2 else "B",
                             "signal": base + offset
                                       + (drift if level == "neg" else 0.0)})
    return pd.DataFrame(rows)


def _database(tmp_path) -> str:
    """A two-table measurement database, as ``measurements.db`` really is."""
    path = str(tmp_path / "measurements.db")
    frame = _campaign(12)
    with sqlite3.connect(path) as db:
        frame.to_sql("cell", db, index=False)
        frame.to_sql("nucleus", db, index=False)
    return path


@pytest.fixture
def screen(qtbot) -> ControlChartScreen:
    """A screen whose table read and chart run inline on the calling thread."""
    made = ControlChartScreen(threaded=False)
    qtbot.addWidget(made)
    return made


def _levels(made: ControlChartScreen):
    return [made._levels.item(i).text() for i in range(made._levels.count())]


def _picker_items(box):
    return [box.itemText(i) for i in range(box.count())]


# ---------------------------------------------------------------------------
# The canvas: a backend without the owned-timer fix
# ---------------------------------------------------------------------------

class _BackendWithoutTheFix(QWidget):
    """A figure canvas that never grew ``cancel_pending_draw``.

    The attribute is present but is not a method — the shape a stale
    matplotlib backend, or a host that swapped the canvas out, actually
    presents. Calling it would raise ``TypeError``.
    """

    cancel_pending_draw = "not a method"


def test_a_canvas_without_the_owned_timer_fix_still_closes(qtbot):
    """Closing the chart must cancel the deferred draw when the backend has
    one — an idle draw that fires after Qt has deleted the canvas is a
    segfault on close, not an exception — and must close anyway when the
    backend has no such method, because a screen that cannot be closed is
    worse than one that leaks a timer."""
    with_fix = ControlChartCanvas()
    qtbot.addWidget(with_fix)
    cancelled = []
    with_fix.canvas.cancel_pending_draw = lambda: cancelled.append("cancelled")
    assert with_fix.close() is True
    assert cancelled == ["cancelled"]

    without_fix = ControlChartCanvas()
    qtbot.addWidget(without_fix)
    without_fix.canvas = _BackendWithoutTheFix(without_fix)
    assert without_fix.close() is True
    assert without_fix.isVisible() is False
    # Not called, not replaced: the guard read it and left it alone.
    assert without_fix.canvas.cancel_pending_draw == "not a method"


# ---------------------------------------------------------------------------
# The control column and its levels
# ---------------------------------------------------------------------------

def test_clearing_the_control_column_empties_the_levels_it_described(screen):
    """The tick list and the two Z' pickers name levels OF the control
    column. Emptying the column means "the table is already only the
    control", and levels left behind from the old column would go straight
    into the spec — charting a filter the form no longer shows."""
    frame = _campaign(12)
    screen.set_frame(frame)
    screen._control_column.setCurrentText("well_type")
    assert _levels(screen) == ["neg", "pos"]
    assert _picker_items(screen._positive) == ["", "neg", "pos"]

    screen._control_column.setCurrentText("")

    assert _levels(screen) == []
    assert _picker_items(screen._positive) == [""]
    assert _picker_items(screen._negative) == [""]
    # The chart is still drawn — every well of every plate, unfiltered.
    assert screen.spec().control_column is None
    assert screen.result is not None
    assert len(screen.result) == 12


def test_a_control_column_the_frame_no_longer_has_leaves_the_list_empty(screen):
    """A picker outlives the frame it was filled from: load a campaign,
    pick ``batch``, then open the run that has no ``batch`` column. The level
    list has to come back empty rather than raise ``KeyError`` deep inside the
    refill and leave the screen half-built."""
    screen.set_frame(_campaign(12))
    screen._control_column.setCurrentText("batch")
    assert _levels(screen) == ["A", "B"]

    screen._refill_levels(_campaign(12).drop(columns=["batch"]))

    assert _levels(screen) == []
    assert _picker_items(screen._negative) == [""]
    # ...and a frame that still has the column refills it again.
    screen._refill_levels(_campaign(12))
    assert _levels(screen) == ["A", "B"]


def test_a_control_column_change_before_any_table_draws_nothing(screen):
    """The pickers are wired at construction, so a host that seeds the
    control column — a restored session, a settings file — reaches this slot
    with no frame at all. It must not refill a level list from ``None``, and
    the screen has to stay on its opening instruction rather than show an
    empty chart as if a table had been read."""
    opening = screen.report.toPlainText()

    screen._control_column.addItem("well_type")

    assert screen._control_column.currentText() == "well_type"
    assert _levels(screen) == []
    assert screen.result is None
    assert screen.report.toPlainText() == opening

    # The same signal, once a frame is there, does refill the list.
    screen.set_frame(_campaign(12))
    screen._control_column.setCurrentText("batch")
    assert _levels(screen) == ["A", "B"]


# ---------------------------------------------------------------------------
# Edits that arrive while the form is being refilled
# ---------------------------------------------------------------------------

def test_a_widget_edited_while_a_table_loads_does_not_recompute(screen):
    """``set_frame`` refills eleven pickers before it charts anything, and
    every one of them is connected to ``recompute``. Without the loading
    latch the user would get eleven charts per file — each one computed from
    a half-filled form, the last one right — instead of the single chart
    ``set_frame`` ends with."""
    screen.set_frame(_campaign(30))
    first = screen.result
    assert first is not None
    assert len(first.baseline_plates) == 20         # the default baseline

    screen._loading = True                          # the state set_frame holds
    screen._baseline.setValue(10)                   # a real valueChanged
    assert screen.result is first                   # nothing was recharted
    assert len(screen.result.baseline_plates) == 20

    screen._loading = False
    screen._baseline.setValue(12)
    assert screen.result is not first
    assert len(screen.result.baseline_plates) == 12


# ---------------------------------------------------------------------------
# The table picker
# ---------------------------------------------------------------------------

def test_the_table_picker_reads_nothing_until_a_file_is_open(screen, tmp_path):
    """Filling the picker sets its current text, which emits the same signal
    a click does. With no file open there is nothing to re-read, and a
    ``load_path(None)`` would be an exception in the log and a dead screen;
    once a database is open the very same signal has to re-read the file."""
    screen._table_picker.addItems(["cell", "nucleus"])

    assert screen._table_picker.currentText() == "cell"
    assert screen._path is None
    assert screen.result is None
    assert screen._source.text() == "no table loaded"

    path = _database(tmp_path)
    screen.load_path(path, table="cell")
    charted = screen.result
    assert charted is not None

    screen._table_picker.setCurrentText("nucleus")

    assert screen._path == path
    assert "nucleus" in screen._source.text()
    assert screen.result is not charted
    assert len(screen.result) == 12
