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

import sqlite3

import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.screens import dose_response as dr
from spacr.qt.screens.dose_response import DoseResponseScreen, _format
from spacr.qt.widgets.dose_response import (CI_WALD, DoseResponseSet,
                                            DoseResponseSpec)

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


@pytest.fixture()
def two_table_db(tmp_path):
    """A database with two readable tables, the shape a plate really has."""
    path = tmp_path / "measurements.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE object (conc_uM REAL, signal REAL)")
        db.executemany("INSERT INTO object VALUES (?, ?)",
                       [(0.1, 10.0), (1.0, 50.0), (10.0, 90.0)])
        db.execute("CREATE TABLE well (conc_uM REAL, other REAL)")
        db.execute("INSERT INTO well VALUES (0.5, 1.0)")
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


def test_a_csv_has_no_table_picker(screen, table_csv):
    screen.load_path(str(table_csv))
    assert not screen._table_picker.isVisible()
    assert screen._table_picker.count() == 0
    assert "plate.csv" in screen._source.text()
    assert "9 rows" in screen._source.text()


def test_a_database_offers_its_tables_and_reads_the_chosen_one(screen,
                                                               two_table_db):
    screen.load_path(str(two_table_db))
    offered = [screen._table_picker.itemText(i)
               for i in range(screen._table_picker.count())]
    assert offered == ["object", "well"]
    assert list(screen._frame.columns) == ["conc_uM", "signal"]
    assert "· object" in screen._source.text()

    screen._on_table_picked("well")
    assert list(screen._frame.columns) == ["conc_uM", "other"]
    assert "· well" in screen._source.text()


def test_the_named_table_is_the_one_read(screen, two_table_db):
    screen.load_path(str(two_table_db), table="well")
    assert screen._table_picker.currentText() == "well"
    assert list(screen._frame.columns) == ["conc_uM", "other"]


def test_picking_no_table_reloads_nothing(screen, two_table_db):
    screen.load_path(str(two_table_db))
    before = screen._source.text()
    screen._on_table_picked("")
    assert screen._source.text() == before


def test_a_file_that_is_not_a_database_says_which_file_and_why(screen,
                                                               tmp_path):
    """A refusal names the file: a bare exception in the source line is not
    an answer to "what did I just drop on this screen?"."""
    broken = tmp_path / "notes.db"
    broken.write_bytes(b"this is not a database")
    screen.load_path(str(broken))
    assert "could not read notes.db" in screen._source.text()
    assert screen._frame is None


# ---------------------------------------------------------------------------
# Fitting, and refusing to
# ---------------------------------------------------------------------------

def test_with_no_table_loaded_there_is_nothing_to_fit(screen):
    screen.fit()
    assert screen.report.toPlainText() == ""
    assert screen.result_set() is None


def test_a_spec_the_engine_will_not_accept_is_reported_not_raised(screen,
                                                                  table_csv):
    """A stored setting from a newer build reaches the picker as data."""
    screen.load_path(str(table_csv))
    screen.ci_picker.addItem("From a newer build", "bootstrap")
    screen.ci_picker.setCurrentIndex(screen.ci_picker.count() - 1)
    screen.fit()
    assert "unknown ci_method 'bootstrap'" in screen.report.toPlainText()
    assert screen.result_set() is None


def test_a_fit_with_no_groups_reports_instead_of_drawing_a_curve(screen,
                                                                 table_csv):
    screen.load_path(str(table_csv))
    empty = DoseResponseSet(fits=(), spec=DoseResponseSpec(
        concentration="conc_uM", response="signal", ci_method=CI_WALD))
    screen._on_fitted(empty)
    assert screen.table.rowCount() == 0
    assert screen.report.toPlainText() == empty.report()


def test_selecting_a_row_before_any_fit_draws_nothing(screen, table_csv):
    screen.load_path(str(table_csv))
    screen.table.setRowCount(1)
    screen.table.selectRow(0)
    screen._on_row_selected()          # no fit yet; nothing to show
    assert screen.result_set() is None


def test_a_worker_failure_lands_in_the_report_pane_not_a_dialog(screen):
    screen._on_job_failed("read_table: file is not a database")
    assert "file is not a database" in screen.report.toPlainText()
    assert "file is not a database" in screen._source.text()


# ---------------------------------------------------------------------------
# The pickers
# ---------------------------------------------------------------------------

def test_a_reload_keeps_the_column_the_user_had_chosen(screen):
    """Re-reading the same plate must not move the dose column underneath."""
    # Five distinct positive doses: fewer than MIN_DOSES cannot be a
    # dilution series, and the picker rightly offers nothing at all.
    frame = pd.DataFrame({"conc_uM": [0.01, 0.1, 1.0, 10.0, 100.0] * 3,
                          "dose_nM": [10.0, 100.0, 1e3, 1e4, 1e5] * 3,
                          "signal": [5.0, 20.0, 50.0, 80.0, 95.0] * 3})
    screen.set_frame(frame, label="first")
    screen.concentration_picker.setCurrentText("dose_nM")

    screen.set_frame(frame, label="again")
    assert screen.concentration_picker.currentText() == "dose_nM"
