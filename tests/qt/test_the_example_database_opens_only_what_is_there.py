"""Opening the example database from the browsers copes with what is there.

Pinned behaviour of the "Load test data" hand-off on
:class:`spacr.qt.screens.db_browser.DbBrowserScreen` and
:class:`spacr.qt.screens.plate_view.PlateViewScreen`:

* an example database that cannot be opened leaves the Database Browser
  with no tables and an error on its status line;
* an example database without the per-cell table is still listed, and
  the browser does not pretend to select a table it lacks;
* the Plate View, handed an example it cannot open, reports it and forgets
  the measurement it was going to draw, so the next database the user opens
  is not coloured by the example's column.
"""
from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens.db_browser import DbBrowserScreen  # noqa: E402
from spacr.qt.screens.plate_view import PlateViewScreen  # noqa: E402
from spacr.qt.widgets.measurements_example import EXAMPLE_TABLE  # noqa: E402

pytestmark = pytest.mark.qt


def _database(tmp_path, tables):
    path = tmp_path / "measurements.db"
    con = sqlite3.connect(path)
    try:
        for name in tables:
            con.execute(f'CREATE TABLE "{name}" (object_label INTEGER)')
            con.execute(f'INSERT INTO "{name}" VALUES (1)')
        con.commit()
    finally:
        con.close()
    return path


def test_an_example_that_cannot_be_opened_lists_nothing(qtbot, tmp_path):
    screen = DbBrowserScreen(threaded=False)
    qtbot.addWidget(screen)

    screen._open_the_example(str(tmp_path), tmp_path / "missing.db")

    assert screen.tables() == []
    assert screen.status_text() != ""
    assert screen.current_table() == ""


def test_an_example_without_the_cell_table_is_still_listed(qtbot, tmp_path):
    assert EXAMPLE_TABLE == "cell"
    database = _database(tmp_path, ["annotations", "nucleus"])
    screen = DbBrowserScreen(threaded=False)
    qtbot.addWidget(screen)

    screen._open_the_example(str(tmp_path), database)

    assert screen.tables() == ["annotations", "nucleus"]
    assert screen.current_table() != EXAMPLE_TABLE


def test_the_plate_view_forgets_the_example_measurement_it_could_not_open(
        qtbot, tmp_path):
    screen = PlateViewScreen(threaded=False)
    qtbot.addWidget(screen)
    missing = tmp_path / "measurements" / "measurements.db"

    screen._open_the_example(str(tmp_path), missing)

    assert screen._path_edit.text() == str(missing)
    assert screen.status_text() != ""
    assert screen._measurement_to_render == ""
