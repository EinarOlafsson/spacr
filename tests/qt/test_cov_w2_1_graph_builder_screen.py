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
from spacr.selection import Selection


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


def test_a_file_whose_tables_cannot_be_listed_says_so_inline(screen,
                                                             tmp_path):
    """No modal, and the frame stays as it was."""
    broken = tmp_path / "not-a-database.db"
    broken.write_bytes(b"this is not sqlite")

    screen.load_path(str(broken))

    assert "could not read not-a-database.db" in screen._source.text()
    assert screen._frame is None


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


def test_picking_another_table_reloads_the_same_file(screen, database):
    """The picker is wired to the loader, not to a cached frame."""
    screen.load_path(str(database))

    screen._on_table_picked("zebra")

    assert "· zebra" in screen._source.text()


def test_picking_a_table_with_no_file_loaded_does_nothing(screen):
    """The combo is populated programmatically; its signal must be harmless."""
    screen._on_table_picked("cell")

    assert screen._frame is None


def test_opening_a_selection_that_does_not_exist_asks_for_a_brush(screen,
                                                                  measurements):
    """The resting state is not an empty result, and must not be sent on."""
    screen.set_frame(measurements)

    screen._open_selection()

    assert screen._source.text() == "Brush a region first — nothing is selected."


def test_a_brush_around_blank_space_is_also_nothing_to_open(screen,
                                                            measurements):
    """An explicit but empty selection has no objects to show either."""
    screen.set_frame(measurements)
    screen.builder.canvas.link.set_selection(Selection(keys=measurements.index[:0],
                                         source="graph_builder"))

    screen._open_selection()

    assert "nothing is selected" in screen._source.text()


def test_with_nothing_able_to_show_crops_the_screen_says_so(screen,
                                                            measurements,
                                                            monkeypatch):
    """Better than a NoObjectOpener traceback from a button press."""
    screen.set_frame(measurements)
    screen.builder.canvas.link.set_selection(Selection(keys=measurements.index[:2],
                                         source="graph_builder"))
    monkeypatch.setattr("spacr.qt.linked_selection.has_object_opener",
                        lambda kind: False)

    screen._open_selection()

    assert screen._source.text() == (
        "Nothing can show crops yet — open the Annotate screen once.")


def test_a_brushed_selection_reaches_the_registered_opener(screen,
                                                           measurements,
                                                           monkeypatch):
    """The screen never imports Annotate; the request is routed."""
    screen.set_frame(measurements)
    screen.builder.canvas.link.set_selection(Selection(keys=measurements.index[:2],
                                         source="graph_builder"))
    monkeypatch.setattr("spacr.qt.linked_selection.has_object_opener",
                        lambda kind: True)
    opened = []
    screen.builder.canvas.link.register_object_opener(
        "annotate", lambda request: opened.append(request))

    screen._open_selection()

    assert len(opened) == 1
    assert [str(key) for key in opened[0].keys] == ["0", "1"]
    assert "brushed in the Graph Builder" in opened[0].reason


def test_an_opener_that_raises_is_reported_not_propagated(screen,
                                                          measurements,
                                                          monkeypatch):
    """A button press may not end the session with a traceback."""
    screen.set_frame(measurements)
    screen.builder.canvas.link.set_selection(Selection(keys=measurements.index[:2],
                                         source="graph_builder"))
    monkeypatch.setattr("spacr.qt.linked_selection.has_object_opener",
                        lambda kind: True)

    def _explode(request):
        raise RuntimeError("the crops are on a disconnected share")

    screen.builder.canvas.link.register_object_opener("annotate", _explode)

    screen._open_selection()

    assert "could not open those objects" in screen._source.text()
    assert "disconnected share" in screen._source.text()
