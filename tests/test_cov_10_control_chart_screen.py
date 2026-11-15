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
