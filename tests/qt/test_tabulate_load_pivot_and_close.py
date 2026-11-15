"""Tabulate: loading a table, filtering and pivoting it, plotting it, closing.

The screen is built with ``threaded=False`` and drives the real reader against
real files: a CSV, a two-table SQLite database and a file that is neither. The
source line is the screen's only error channel (no modal), so every case reads
back its exact text along with the loaded frame and the table picker. The
pivot, the shared filter, the plot hand-off and closing are walked the same
way.
"""
from __future__ import annotations

import sqlite3
import types

import pandas as pd
import pytest
from PySide6.QtCore import QMimeData, QPointF, Qt
from PySide6.QtGui import QDropEvent

from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.screens.tabulate import TabulateScreen
from spacr.qt.widgets.graph_builder import COLUMN_MIME
from spacr.qt.widgets.pivot_builder import AXIS_ROWS, AXIS_VALUES
from spacr.qt.widgets.pivot_spec import PivotSpec, pivot
from spacr.selection import CategoryFilter, DataFilter

pytestmark = pytest.mark.qt

AREAS = [10.0, 20.0, 30.0, 40.0]


def drop(well, column: str) -> None:
    """Drop ``column`` onto a pivot well the way drag-and-drop does."""
    payload = QMimeData()
    payload.setData(COLUMN_MIME, column.encode("utf-8"))
    well._list.dropEvent(QDropEvent(QPointF(4, 4), Qt.CopyAction, payload,
                                    Qt.LeftButton, Qt.NoModifier))


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
        "area": AREAS,
    })


@pytest.fixture
def screen(qtbot):
    """A screen on a PRIVATE link whose reads happen inline."""
    widget = TabulateScreen(link=LinkedSelection(), threaded=False)
    qtbot.addWidget(widget)
    return widget


def _picker(screen):
    picker = screen._table_picker
    return [picker.itemText(i) for i in range(picker.count())]


def test_load_path_reads_csvs_and_database_tables_and_reports_refusals(
        screen, tmp_path, measurements, subtests):
    """CSV, first table, named table, picked table, and an unreadable file."""
    csv = tmp_path / "cells.csv"
    measurements.to_csv(csv, index=False)
    database = tmp_path / "measurements.db"
    with sqlite3.connect(database) as db:
        measurements.to_sql("cell", db, index=False)
        measurements.assign(area=measurements["area"] / 2).to_sql(
            "nucleus", db, index=False)
    broken = tmp_path / "not-a-database.db"
    broken.write_bytes(b"this is not sqlite")
    halves = [a / 2 for a in AREAS]

    cases = [
        ("a csv loads with no table picker",
         lambda: screen.load_path(str(csv)), [], "",
         "cells.csv · 4 rows × 7 columns", AREAS),
        ("a database loads its first table",
         lambda: screen.load_path(str(database)), ["cell", "nucleus"], "cell",
         "measurements.db · cell · 4 rows × 7 columns", AREAS),
        ("a named table is the one read",
         lambda: screen.load_path(str(database), table="nucleus"),
         ["cell", "nucleus"], "nucleus",
         "measurements.db · nucleus · 4 rows × 7 columns", halves),
        ("picking another table reloads the same file",
         lambda: (screen.load_path(str(database)),
                  screen._on_table_picked("nucleus")),
         ["cell", "nucleus"], "nucleus",
         "measurements.db · nucleus · 4 rows × 7 columns", halves),
    ]
    for case_id, act, tables, current, source, areas in cases:
        with subtests.test(msg=case_id):
            act()
            assert _picker(screen) == tables
            assert screen._table_picker.isVisibleTo(screen) is bool(tables)
            assert screen._table_picker.currentText() == current
            assert screen._source.text() == source
            assert screen._frame["area"].tolist() == areas
            assert screen.is_busy() is False
            assert screen.active_jobs() == 0

    with subtests.test(msg="a file whose tables cannot be listed"):
        screen._frame = None
        screen.load_path(str(broken))
        assert screen._source.text() == (
            "could not read not-a-database.db: file is not a database")
        assert screen._frame is None
        assert screen._path == str(broken)


def test_the_pivot_follows_the_shared_filter_and_reports_its_shape(
        screen, measurements, subtests):
    """A foreign filter keeps every row; a matching one re-aggregates."""
    screen.set_frame(measurements)
    with subtests.test(msg="set_frame names the size"):
        assert screen._source.text() == "4 rows × 7 columns"

    with subtests.test(msg="a computed pivot reports its shape"):
        result = pivot(measurements, PivotSpec(rows=("plateID",),
                                               values=("area",),
                                               aggs=("mean",)))
        screen._on_computed(result)
        assert screen._source.text() == "4 rows → 2 × 1 table"

    with subtests.test(msg="a filter on a column this table lacks"):
        screen._link.set_filter(
            DataFilter([CategoryFilter("nothing_here", ("x",))]))
        assert screen._filtered()["area"].tolist() == AREAS
        screen._link.set_filter(DataFilter([]))

    with subtests.test(msg="a matching filter narrows the aggregation"):
        drop(screen.pivot.wells[AXIS_ROWS], "plateID")
        drop(screen.pivot.wells[AXIS_VALUES], "area")
        screen.pivot.recompute()
        assert screen.pivot.result.n_at("area", 0, 0) == 2
        screen._link.set_filter(DataFilter([CategoryFilter("gene", ("a",))]))
        assert screen._refilter.isActive() is True
        screen._recompute_filtered()
        screen.pivot.recompute()
        assert screen.pivot.result.n_at("area", 0, 0) == 1
        assert screen._filtered()["area"].tolist() == [10.0, 30.0]


def test_plot_summary_refuses_empty_tables_and_hands_over_the_summary(
        screen, measurements, subtests):
    """No pivot or an empty frame is refused; a summary reaches the graph."""
    screen.set_frame(measurements)
    refusal = ("Nothing to plot — build a table with at least one "
               "non-empty cell first.")
    for case_id, frame in [("an explicit empty frame", pd.DataFrame()),
                           ("no pivot built yet", None)]:
        with subtests.test(msg=case_id):
            screen.plot_summary(frame)
            assert screen._source.text() == refusal
            assert screen.graph.canvas.render_data is None

    with subtests.test(msg="the summary reaches the graph builder"):
        drop(screen.pivot.wells[AXIS_ROWS], "plateID")
        drop(screen.pivot.wells[AXIS_VALUES], "area")
        screen.pivot.recompute()
        screen.plot_summary()
        assert screen._source.text() == (
            "plotting the summary · 2 cell(s) — drag a key onto X and a "
            "statistic onto Y")
        frame = screen.graph.canvas._frame
        assert frame["mean"].tolist() == [15.0, 35.0]
        assert frame["plateID"].tolist() == ["p1", "p2"]


def test_closing_never_raises_whatever_the_link_state(screen, measurements,
                                                     subtests):
    """A second close and a link that will not disconnect are both quiet."""
    screen.set_frame(measurements)
    with subtests.test(msg="closing twice"):
        screen._refilter.start()
        screen.close()
        screen.close()
        assert screen._refilter.isActive() is False
        assert screen.is_busy() is False

    with subtests.test(msg="a link that refuses to disconnect"):
        refused = []

        def _refuse(slot):
            refused.append(slot)
            raise TypeError("already disconnected")

        screen._link = types.SimpleNamespace(
            filter_changed=types.SimpleNamespace(disconnect=_refuse))
        screen._refilter.start()
        screen.close()
        assert refused == [screen._on_filter_changed]
        assert screen._refilter.isActive() is False
