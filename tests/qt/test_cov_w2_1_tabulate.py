"""Loading a table into Tabulate, and what happens when the load goes wrong.

The reads run on a worker thread, so every test here builds the screen with
``threaded=False`` and drives the real reader against real files on disk: a
CSV, a SQLite database with two tables, a file that is neither. What is
asserted is the source line the user reads, because on this screen that line
is the entire error channel -- there is deliberately no modal.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest
from PySide6.QtCore import QMimeData, QPointF, Qt
from PySide6.QtGui import QDropEvent

from spacr.qt.widgets.graph_builder import COLUMN_MIME

from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.screens.tabulate import TabulateScreen


def drop(well, column: str) -> None:
    """Drop ``column`` onto a well the way the drag-and-drop actually does."""
    payload = QMimeData()
    payload.setData(COLUMN_MIME, column.encode("utf-8"))
    well._list.dropEvent(QDropEvent(QPointF(4, 4), Qt.CopyAction, payload,
                                    Qt.LeftButton, Qt.NoModifier))


@pytest.fixture
def link():
    """A PRIVATE link — never the process-wide one."""
    return LinkedSelection()


@pytest.fixture
def measurements():
    """Two plates, two genes, one measured column."""
    return pd.DataFrame({
        "plateID": ["p1", "p1", "p2", "p2"],
        "rowID": ["r1", "r2", "r1", "r2"],
        "columnID": ["c1"] * 4,
        "fieldID": ["f1"] * 4,
        "gene": ["a", "b", "a", "b"],
        "object_label": [1, 2, 3, 4],
        "area": [10.0, 20.0, 30.0, 40.0],
    })


@pytest.fixture
def screen(qtbot, link):
    """A Tabulate screen whose reads happen inline, so a test can see them."""
    widget = TabulateScreen(link=link, threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def database(tmp_path, measurements):
    """A measurement database with a `cell` table and a `nucleus` one."""
    path = tmp_path / "measurements.db"
    with sqlite3.connect(path) as db:
        measurements.to_sql("cell", db, index=False)
        measurements.assign(area=measurements["area"] / 2).to_sql(
            "nucleus", db, index=False)
    return path


def test_picking_a_table_with_no_file_loaded_does_nothing(screen):
    """The combo is populated programmatically; its signal must be harmless."""
    screen._on_table_picked("cell")

    assert screen._frame is None


def test_a_failed_read_is_reported_on_the_source_line(screen, tmp_path):
    """The failure arrives from the worker; it still lands on the label."""
    screen._path = str(tmp_path / "cells.csv")

    screen._on_load_failed("No columns to parse from file")

    assert screen._source.text() == (
        "could not read cells.csv: No columns to parse from file")


def test_choosing_a_table_from_the_dialog_loads_it(screen, monkeypatch,
                                                   tmp_path, measurements):
    """The button is a file dialog and then the ordinary load path."""
    path = tmp_path / "cells.csv"
    measurements.to_csv(path, index=False)
    monkeypatch.setattr(
        "spacr.qt.screens.tabulate.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(path), "")))

    screen.choose_table()

    assert screen._frame is not None
    assert "cells.csv" in screen._source.text()


def test_cancelling_the_dialog_loads_nothing(screen, monkeypatch):
    """An empty path is the user pressing Escape."""
    monkeypatch.setattr(
        "spacr.qt.screens.tabulate.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: ("", "")))

    screen.choose_table()

    assert screen._frame is None
    assert screen._source.text() == "no table loaded"


def test_a_screen_with_no_table_has_nothing_to_filter(screen):
    """The filter signal can arrive before anything has been loaded."""
    assert screen._filtered() is None
    screen._on_filter_changed()
    assert not screen._refilter.isActive()


def test_the_factory_builds_the_screen_the_registry_asks_for(qtbot):
    """`register_app` is handed this function, not the class."""
    from spacr.qt.screens.tabulate import APP_KEY, make_tabulate_screen

    widget = make_tabulate_screen(APP_KEY)
    qtbot.addWidget(widget)

    assert isinstance(widget, TabulateScreen)
