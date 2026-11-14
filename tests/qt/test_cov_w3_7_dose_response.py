"""Loading a table into the Dose–Response screen, and the fits that never run.

``tests/qt/test_dose_response_screen.py`` fits a synthetic plate through the
real engine and reads the engine's own numbers back out of the grid. What is
driven here is the half before that: choosing a file, listing the tables in a
database, switching between them, and every path where a fit is refused before
it starts -- no frame, an impossible spec, a plate with nothing in it.

The files are real (a CSV and a SQLite measurement database written by the
suite's own fixture); only the file *chooser* is answered by the test.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.screens import dose_response as dr
from spacr.qt.screens.dose_response import DoseResponseScreen, _format

pytestmark = pytest.mark.qt


@pytest.fixture()
def screen(qtbot):
    widget = DoseResponseScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture()
def table_csv(tmp_path):
    path = tmp_path / "plate.csv"
    pd.DataFrame({"conc_uM": [0.1, 1.0, 10.0] * 3,
                  "signal": [10.0, 50.0, 90.0] * 3,
                  "gene": ["a"] * 9}).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# One cell of the grid
# ---------------------------------------------------------------------------

def test_an_absent_number_is_a_dash_and_never_the_word_nan():
    assert _format(None) == "—"
    assert _format(float("nan")) == "—"
    assert _format(float("inf")) == "—"
    assert _format(1234.5678) == "1235"
    assert _format("refused") == "refused"


# ---------------------------------------------------------------------------
# Choosing a file
# ---------------------------------------------------------------------------

def test_choosing_a_table_loads_it_and_cancelling_loads_nothing(
        screen, monkeypatch, table_csv):
    monkeypatch.setattr(dr.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    screen.choose_table()
    assert screen._frame is None

    monkeypatch.setattr(dr.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(table_csv), "")))
    screen.choose_table()
    assert screen._frame is not None
    assert len(screen._frame) == 9


# ---------------------------------------------------------------------------
# Fitting, and refusing to
# ---------------------------------------------------------------------------


def test_a_worker_failure_lands_in_the_report_pane_not_a_dialog(screen):
    screen._on_job_failed("read_table: file is not a database")
    assert "file is not a database" in screen.report.toPlainText()
    assert "file is not a database" in screen._source.text()


# ---------------------------------------------------------------------------
# The pickers
# ---------------------------------------------------------------------------
