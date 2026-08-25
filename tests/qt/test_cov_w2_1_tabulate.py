"""Loading a table into Tabulate, and what happens when the load goes wrong.

The reads run on a worker thread, so every test here builds the screen with
``threaded=False`` and drives the real reader against real files on disk: a
CSV, a SQLite database with two tables, a file that is neither. What is
asserted is the source line the user reads, because on this screen that line
is the entire error channel -- there is deliberately no modal.
"""
from __future__ import annotations

import sqlite3
import types

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


def test_a_csv_loads_and_the_source_line_names_it(screen, tmp_path,
                                                  measurements):
    """The label is the only place the screen says what it is showing."""
    path = tmp_path / "cells.csv"
    measurements.to_csv(path, index=False)

    screen.load_path(str(path))

    assert screen._frame is not None
    assert len(screen._frame) == 4
    assert "cells.csv" in screen._source.text()
    assert "4 rows × 7 columns" in screen._source.text()
    assert not screen._table_picker.isVisible()
    assert not screen.is_busy()
    assert screen.active_jobs() == 0


def test_a_database_offers_its_tables_and_loads_the_first(screen, database):
    """The picker has to be filled before the read is dispatched."""
    screen.load_path(str(database))

    assert [screen._table_picker.itemText(i)
            for i in range(screen._table_picker.count())] == ["cell", "nucleus"]
    assert screen._table_picker.currentText() == "cell"
    assert "measurements.db" in screen._source.text()
    assert "· cell" in screen._source.text()
    assert screen._frame["area"].tolist() == [10.0, 20.0, 30.0, 40.0]


def test_a_named_table_is_the_one_that_is_read(screen, database):
    """`load_path(path, table=...)` is how the picker reloads."""
    screen.load_path(str(database), table="nucleus")

    assert screen._table_picker.currentText() == "nucleus"
    assert screen._frame["area"].tolist() == [5.0, 10.0, 15.0, 20.0]


def test_picking_another_table_reloads_the_same_file(screen, database):
    """The picker is wired to the loader, not to a cached frame."""
    screen.load_path(str(database))

    screen._on_table_picked("nucleus")

    assert screen._frame["area"].tolist() == [5.0, 10.0, 15.0, 20.0]


def test_picking_a_table_with_no_file_loaded_does_nothing(screen):
    """The combo is populated programmatically; its signal must be harmless."""
    screen._on_table_picked("cell")

    assert screen._frame is None


def test_a_file_whose_tables_cannot_be_listed_says_so_inline(screen,
                                                             tmp_path):
    """No modal: a dialog nobody can dismiss is how a headless run hangs."""
    broken = tmp_path / "not-a-database.db"
    broken.write_bytes(b"this is not sqlite")

    screen.load_path(str(broken))

    assert "could not read not-a-database.db" in screen._source.text()
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


def test_a_filter_that_does_not_apply_leaves_the_table_alone(screen,
                                                             measurements):
    """A filter drawn on another population must not empty this one."""
    from spacr.selection import CategoryFilter, DataFilter

    screen.set_frame(measurements)
    screen._link.set_filter(DataFilter([CategoryFilter("nothing_here", ("x",))]))

    assert len(screen._filtered()) == 4


def test_an_empty_table_is_not_plotted(screen, measurements):
    """Handing the Graph Builder nothing would leave an empty chart on screen."""
    screen.set_frame(measurements)

    screen.plot_summary(pd.DataFrame())

    assert "Nothing to plot" in screen._source.text()
    assert screen.graph.canvas.render_data is None


def test_plotting_with_no_pivot_built_says_what_to_do_first(screen,
                                                            measurements):
    """`plot_summary()` with no argument asks the pivot, which has no cells."""
    screen.set_frame(measurements)

    screen.plot_summary()

    assert "build a table with at least one non-empty cell" in \
        screen._source.text()


def test_a_computed_pivot_reports_its_shape(screen, measurements):
    """The source line doubles as the pivot's own result line."""
    from spacr.qt.widgets.pivot_spec import PivotSpec, pivot

    screen.set_frame(measurements)
    result = pivot(measurements, PivotSpec(rows=("plateID",),
                                           values=("area",), aggs=("mean",)))

    screen._on_computed(result)

    assert screen._source.text() == "4 rows → 2 × 1 table"


def test_closing_twice_does_not_raise(screen, measurements):
    """A screen closed after its link was already disconnected."""
    screen.set_frame(measurements)

    screen.close()
    screen.close()

    assert not screen.is_busy()


def test_a_screen_with_no_table_has_nothing_to_filter(screen):
    """The filter signal can arrive before anything has been loaded."""
    assert screen._filtered() is None
    screen._on_filter_changed()
    assert not screen._refilter.isActive()


def test_a_filter_change_re_aggregates_the_narrowed_population(screen,
                                                               measurements):
    """A mean of the unfiltered rows beside a filtered plot is the mismatch."""
    from spacr.qt.widgets.pivot_builder import AXIS_ROWS, AXIS_VALUES
    from spacr.selection import CategoryFilter, DataFilter

    screen.set_frame(measurements)
    drop(screen.pivot.wells[AXIS_ROWS], "plateID")
    drop(screen.pivot.wells[AXIS_VALUES], "area")
    screen.pivot.recompute()
    assert screen.pivot.result.n_at("area", 0, 0) == 2

    screen._link.set_filter(DataFilter([CategoryFilter("gene", ("a",))]))
    assert screen._refilter.isActive()
    screen._recompute_filtered()
    screen.pivot.recompute()

    assert screen.pivot.result.n_at("area", 0, 0) == 1


def test_the_summary_reaches_the_graph_builder(screen, measurements):
    """The chart is of the summary, through the ordinary Graph Builder."""
    from spacr.qt.widgets.pivot_builder import AXIS_ROWS, AXIS_VALUES

    screen.set_frame(measurements)
    drop(screen.pivot.wells[AXIS_ROWS], "plateID")
    drop(screen.pivot.wells[AXIS_VALUES], "area")
    screen.pivot.recompute()

    screen.plot_summary()

    assert "plotting the summary" in screen._source.text()
    assert "mean" in list(screen.graph.canvas._frame.columns)


def test_a_link_that_will_not_disconnect_does_not_break_the_close(screen,
                                                                  measurements):
    """Closing a screen may not raise, whatever state the link is in."""
    screen.set_frame(measurements)

    def _refuse(slot):
        raise TypeError("already disconnected")

    screen._link = types.SimpleNamespace(
        filter_changed=types.SimpleNamespace(disconnect=_refuse))

    screen.close()

    assert not screen._refilter.isActive()


def test_the_factory_builds_the_screen_the_registry_asks_for(qtbot):
    """`register_app` is handed this function, not the class."""
    from spacr.qt.screens.tabulate import APP_KEY, make_tabulate_screen

    widget = make_tabulate_screen(APP_KEY)
    qtbot.addWidget(widget)

    assert isinstance(widget, TabulateScreen)
