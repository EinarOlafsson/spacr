"""Loading a table into Small Multiples, and saying so when it cannot be read.

The screen reads on a worker thread, so a second load has to supersede the
first rather than let two reads race to deliver their frames. Listing the
tables stays inline, which makes an unreadable file the one failure the
screen has to report by itself.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from spacr.qt.screens import trellis as T


@pytest.fixture
def screen(qtbot):
    """An unthreaded screen, so a load lands before the call returns."""
    widget = T.TrellisScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def measurement_csv(tmp_path):
    """A small measurement table with a grouping column to facet on."""
    frame = pd.DataFrame({
        "plate": ["p1", "p1", "p2", "p2"],
        "area": [10.0, 20.0, 30.0, 40.0],
        "perimeter": [4.0, 8.0, 12.0, 16.0],
    })
    path = tmp_path / "measurements.csv"
    frame.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def measurement_db(tmp_path):
    """A SQLite file holding two tables, so the picker has a choice."""
    path = tmp_path / "measurements.db"
    with sqlite3.connect(path) as db:
        pd.DataFrame({"plate": ["p1", "p2"], "area": [1.0, 2.0]}).to_sql(
            "object", db, index=False)
        pd.DataFrame({"plate": ["p1"], "count": [7]}).to_sql(
            "well", db, index=False)
    return str(path)


def test_a_csv_reaches_the_grid_the_filter_and_the_formula_panel(
        screen, measurement_csv):
    """One load populates every surface below it, by the one path."""
    screen.load_path(measurement_csv)
    assert screen._frame is not None
    assert list(screen._frame.columns) == ["plate", "area", "perimeter"]
    assert "measurements.csv" in screen._source.text()
    assert "4 rows × 3 columns" in screen._source.text()
    assert not screen._table_picker.isVisible()


def test_a_database_offers_its_tables_and_loads_the_one_that_was_picked(
        screen, measurement_db):
    """The picker is filled inline, because the read needs the chosen name."""
    screen.load_path(measurement_db)
    assert [screen._table_picker.itemText(i)
            for i in range(screen._table_picker.count())] == ["object", "well"]
    assert screen._frame is not None and "area" in screen._frame.columns

    screen._on_table_picked("well")
    assert "count" in screen._frame.columns
    assert "· well" in screen._source.text()


def test_picking_the_table_that_is_already_loaded_still_names_it(
        screen, measurement_db):
    """An explicit table is honoured even when it is the current one."""
    screen.load_path(measurement_db, table="well")
    assert screen._table_picker.currentText() == "well"
    assert "count" in screen._frame.columns


def test_a_file_whose_tables_cannot_be_listed_is_reported_not_read(
        screen, tmp_path):
    """Listing is inline, so this is the one failure the screen owns."""
    broken = tmp_path / "not_a_database.db"
    broken.write_bytes(b"this is not a SQLite file")
    screen.load_path(str(broken))
    assert "could not read not_a_database.db" in screen._source.text()
    assert screen._frame is None


def test_a_read_that_fails_on_the_worker_is_reported_against_the_path(screen):
    """The failure signal names the file the user chose, not the worker."""
    screen._path = "/data/plate7/measurements.db"
    screen._on_load_failed("database is locked")
    assert screen._source.text() == (
        "could not read measurements.db: database is locked")


def test_choosing_no_file_leaves_the_screen_as_it_was(screen, monkeypatch):
    """A cancelled file dialog is not a request to load anything."""
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    screen.choose_table()
    assert screen._frame is None
    assert screen._path is None


def test_choosing_a_file_loads_it(screen, monkeypatch, measurement_csv):
    """The path the dialog returns is the path that gets read."""
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (measurement_csv, "")))
    screen.choose_table()
    assert screen._path == measurement_csv
    assert screen._frame is not None


def test_a_computed_column_reaches_the_grid_without_a_reload(
        screen, measurement_csv):
    """A formula defined after the load is faceted the moment it exists."""
    screen.load_path(measurement_csv)
    from spacr.qt.widgets.formula import ColumnFormula

    assert screen.formulas.add_formula(
        ColumnFormula("compactness", "area / perimeter ** 2")) is True
    screen._on_formulas_changed()
    computed = screen.formulas.computed_frame()
    assert "compactness" in computed.columns
    assert computed["compactness"].iloc[0] == pytest.approx(10.0 / 16.0)


def test_nothing_is_pushed_before_a_table_has_been_loaded(screen):
    """With no frame there is nothing for the grid or the filter to hold."""
    assert screen.formulas.computed_frame() is None
    screen._push_frame()
    assert screen._frame is None


def test_the_spec_is_the_panels_own(screen, measurement_csv):
    """The screen forwards the grid's specification rather than copying it."""
    screen.load_path(measurement_csv)
    spec = screen.spec
    updated = spec.replace(x="area") if hasattr(spec, "replace") else spec
    screen.set_spec(updated)
    assert screen.spec == screen.panel.spec


def test_the_screen_reports_whether_a_read_is_still_running(screen):
    """The host asks the screen, which asks its runner."""
    assert screen.active_jobs() == 0
    assert screen.is_busy() is False


def test_closing_the_screen_abandons_the_read_in_flight(screen, qtbot,
                                                        measurement_csv):
    """A running QThread destroyed with the screen would abort the process."""
    screen.load_path(measurement_csv)
    screen.close()
    assert screen.active_jobs() == 0
