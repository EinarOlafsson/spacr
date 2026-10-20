"""The Database Browser's table stays cheap to paint, search and export.

Lesson 34 painted at 17-22 fps and froze for seconds on a 218-column cell
table. Three causes, each bounded here by call counts rather than time so
the test does not depend on the machine:

* every keystroke in the column search reset the model and, once 60 or fewer
  columns matched, ran ``resizeColumnsToContents`` -- one ``data()`` call per
  role per loaded cell (650 ms for one key, 3.2 s for the word);
* every paint formatted every visible value again;
* the CSV export held the interpreter lock for a whole 5000-row C call, so
  the window could not paint while it ran.
"""
from __future__ import annotations

import csv
import sqlite3

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from spacr.qt.screens import db_browser  # noqa: E402
from spacr.qt.screens.db_browser import DbBrowserScreen, PreviewModel  # noqa: E402

ROWS = 120
FEATURES = 80


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "measurements.db"
    names = [f"cell_feature_{i}" for i in range(FEATURES)]
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE cell (id INTEGER PRIMARY KEY, "
                + ", ".join(f"{n} REAL" for n in names) + ")")
    con.executemany(
        f"INSERT INTO cell VALUES ({', '.join('?' * (FEATURES + 1))})",
        [(r, *[r * 1.5 + c / 7 for c in range(FEATURES)])
         for r in range(ROWS)])
    con.commit()
    con.close()
    return str(path)


@pytest.fixture
def counted(monkeypatch):
    calls = {"data": 0, "format": 0}
    data = PreviewModel.data
    display = db_browser._display_text

    def counting_data(self, index, role=Qt.DisplayRole):
        calls["data"] += 1
        return data(self, index, role)

    def counting_display(value):
        calls["format"] += 1
        return display(value)

    monkeypatch.setattr(PreviewModel, "data", counting_data)
    monkeypatch.setattr(db_browser, "_display_text", counting_display)
    return calls


def _screen(qtbot, database):
    screen = DbBrowserScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.resize(1200, 800)
    screen.show()
    qtbot.waitExposed(screen)
    screen._page_size_box.setValue(ROWS)
    assert screen.set_database(database)
    assert screen.select_table("cell")
    assert screen.loaded_rows() == ROWS
    return screen


def test_typing_a_column_search_filters_once_after_the_pause(qtbot, database):
    screen = _screen(qtbot, database)
    resets = []
    screen._model.modelReset.connect(lambda: resets.append(1))
    screen._col_search.setFocus()
    QTest.keyClicks(screen._col_search, "feature_7")
    assert resets == [], "a keystroke reset the model before the pause"
    qtbot.waitUntil(lambda: bool(resets), timeout=3000)
    assert len(resets) == 1
    assert screen.visible_columns() == [
        "cell_feature_7"] + [f"cell_feature_{i}" for i in range(70, 80)]


def test_reading_the_columns_right_after_typing_sees_the_search(qtbot, database):
    screen = _screen(qtbot, database)
    screen._col_search.setText("feature_12")
    assert screen.visible_columns() == ["cell_feature_12"]


def test_fitting_the_columns_does_not_ask_the_model_for_every_cell(
        qtbot, database, counted):
    screen = _screen(qtbot, database)
    counted["data"] = 0
    screen.set_column_filter("feature_1")
    shown = screen.visible_columns()
    assert 1 < len(shown) <= db_browser.AUTOSIZE_MAX_COLUMNS
    assert counted["data"] < ROWS, (
        f"{counted['data']} data() calls to fit {len(shown)} columns; "
        f"resizeColumnsToContents makes about {ROWS * len(shown) * 7}")
    header = screen._view.horizontalHeader()
    metrics = screen._view.fontMetrics()
    for section, name in enumerate(shown):
        widest = max(metrics.horizontalAdvance(screen._model.data(
            screen._model.index(row, section))) for row in range(ROWS))
        assert header.sectionSize(section) >= widest
        assert header.sectionSize(section) >= metrics.horizontalAdvance(name)


def test_a_repaint_formats_nothing_new(qtbot, database, counted):
    screen = _screen(qtbot, database)
    viewport = screen._view.viewport()
    viewport.repaint()
    counted["format"] = 0
    for _ in range(3):
        viewport.repaint()
    assert counted["format"] == 0


def test_the_export_writes_small_slices_and_every_row(
        qtbot, database, tmp_path, monkeypatch):
    screen = _screen(qtbot, database)
    sizes = []
    real_writer = csv.writer

    class Recording:
        def __init__(self, handle):
            self._writer = real_writer(handle)

        def writerow(self, row):
            return self._writer.writerow(row)

        def writerows(self, rows):
            rows = list(rows)
            sizes.append(len(rows))
            return self._writer.writerows(rows)

    monkeypatch.setattr(db_browser.csv, "writer", Recording)
    out = tmp_path / "out.csv"
    assert screen.export_csv(str(out))
    assert sum(sizes) == ROWS
    assert max(sizes) <= db_browser._CSV_ROWS_PER_WRITE
    with open(out, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == ROWS + 1
    assert rows[0][0] == "id"
