"""Loading a table into the Graph Builder, and routing a brush to Annotate.

The screen's whole error channel is the source label -- there is deliberately
no modal, because a dialog nobody can dismiss is how a headless run hangs. So
each of these drives a real read against a real file, or a real brushed
selection, and reads that label back.
"""
from __future__ import annotations

import sqlite3
import types

import pandas as pd
import pytest

from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.screens.graph_builder import (
    GraphBuilderScreen, read_table, table_names,
)


@pytest.fixture
def link():
    """A PRIVATE link — never the process-wide one."""
    return LinkedSelection()


@pytest.fixture
def measurements():
    """Four objects across two plates, with one measured column."""
    return pd.DataFrame({
        "plateID": ["p1", "p1", "p2", "p2"],
        "rowID": ["r1", "r2", "r1", "r2"],
        "columnID": ["c1"] * 4,
        "fieldID": ["f1"] * 4,
        "object_label": [1, 2, 3, 4],
        "area": [10.0, 20.0, 30.0, 40.0],
    })


@pytest.fixture
def database(tmp_path, measurements):
    """A database with a `cell` table and an unranked extra one."""
    path = tmp_path / "measurements.db"
    with sqlite3.connect(path) as db:
        measurements.to_sql("cell", db, index=False)
        measurements.to_sql("zebra", db, index=False)
    return path


@pytest.fixture
def screen(qtbot, link):
    """A Graph Builder screen whose reads happen inline."""
    widget = GraphBuilderScreen(link=link, threaded=False)
    qtbot.addWidget(widget)
    return widget


def test_the_preferred_tables_are_offered_first(database):
    """`cell` before `zebra`, whatever order sqlite_master lists them in."""
    assert table_names(str(database)) == ["cell", "zebra"]


def test_a_row_cap_is_applied_in_sql(database):
    """The cap exists for a file too big to read at all, so it is not pandas."""
    capped = read_table(str(database), "cell", limit=2)

    assert len(capped) == 2
    assert len(read_table(str(database), "cell")) == 4


def test_a_tsv_is_read_with_tabs(tmp_path, measurements):
    """A comma separator on a TSV yields one column of joined text."""
    path = tmp_path / "cells.tsv"
    measurements.to_csv(path, sep="\t", index=False)

    frame = read_table(str(path))

    assert list(frame.columns) == list(measurements.columns)
    assert len(frame) == 4


def test_a_failed_read_is_reported_on_the_source_line(screen, tmp_path):
    """The failure arrives from the worker and still lands on the label."""
    screen._path = str(tmp_path / "cells.csv")

    screen._on_load_failed("No columns to parse from file")

    assert screen._source.text() == (
        "could not read cells.csv: No columns to parse from file")


def test_choosing_a_table_from_the_dialog_loads_it(screen, monkeypatch,
                                                   database):
    """The button is a file dialog and then the ordinary load path."""
    monkeypatch.setattr(
        "spacr.qt.screens.graph_builder.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(database), "")))

    screen.choose_table()

    assert screen._frame is not None
    assert "measurements.db" in screen._source.text()
    assert screen._table_picker.currentText() == "cell"


def test_cancelling_the_dialog_loads_nothing(screen, monkeypatch):
    """An empty path is the user pressing Escape."""
    monkeypatch.setattr(
        "spacr.qt.screens.graph_builder.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: ("", "")))

    screen.choose_table()

    assert screen._frame is None


def test_picking_a_table_with_no_file_loaded_does_nothing(screen):
    """The combo is populated programmatically; its signal must be harmless."""
    screen._on_table_picked("cell")

    assert screen._frame is None
