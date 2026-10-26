"""Merging tables in Graph Builder never holds the GUI thread for long.

Lesson 58 ticks cell, nucleus and pathogen in Merge tables, validates the
preview and creates the merged table: 2,341 rows by 986 columns. Recording
on the workstation, the window froze for five to six seconds. Measured on
the tutorial database, the GUI thread spent 3.8 s inserting about 700 rule
rows one at a time with a combo box each, every time a table was ticked, and
the preview's ``head(12).to_string()`` deep-copied the merge definition from
``frame.attrs`` into each of the 986 columns it formatted.

This drives the same flow on a synthetic database of the same width with the
real worker threads, and times every gap between 5 ms heartbeats of the GUI
event loop from the dialog opening to the merged table being shown.
"""
from __future__ import annotations

import sqlite3
import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

#: The longest the GUI thread may go without serving its event loop.
#: Measured on this data: 2,232 ms before the fix; after it about 100 ms on
#: an idle machine and 268 ms with four test workers and another suite
#: sharing the CPU, where the worker threads compete for the interpreter.
BUDGET_MS = 500


@pytest.fixture
def wide_database(tmp_path):
    rng = np.random.default_rng(58)
    path = str(tmp_path / "measurements.db")
    cells = 2400
    ident = dict(plateID=["plate1"] * cells, rowID=["r1"] * cells, columnID=["c1"] * cells,
                 fieldID=[str(i // 60) for i in range(cells)])
    labels = [i % 60 + 1 for i in range(cells)]
    with sqlite3.connect(path) as db:
        pd.DataFrame({**ident, "object_label": labels,
                      **{f"cell_m{i}": rng.normal(size=cells) for i in range(270)}}
                     ).to_sql("cell", db, index=False)
        for table in ("nucleus", "pathogen"):
            pd.DataFrame({**ident, "object_label": labels, "cell_id": labels,
                          **{f"{table}_m{i}": rng.normal(size=cells) for i in range(350)}}
                         ).to_sql(table, db, index=False)
    return path


def _pump(app, done, limit=180.0):
    deadline = time.monotonic() + limit
    while not done():
        assert time.monotonic() < deadline, "the merge did not finish"
        app.processEvents()
        time.sleep(0.002)


def test_merging_three_wide_tables_never_blocks_the_gui(qtbot, wide_database):
    from spacr.qt.screens.graph_builder import GraphBuilderScreen
    from spacr.qt.widgets.merge_tables_dialog import MergeTablesDialog

    app = QApplication.instance()
    screen = GraphBuilderScreen()
    qtbot.addWidget(screen)
    screen.resize(1600, 1000)
    screen.show()
    screen.load_path(wide_database, "cell")
    _pump(app, lambda: not screen.is_busy() and screen._frame is not None)
    settled = time.monotonic() + 0.5
    _pump(app, lambda: time.monotonic() > settled)

    beats = []
    timer = QTimer()
    timer.setInterval(5)
    timer.timeout.connect(lambda: beats.append(time.perf_counter()))
    timer.start()
    beats.append(time.perf_counter())

    dialog = MergeTablesDialog(wide_database, screen, selected=["cell"])
    dialog.show()
    for index in range(dialog.tables.count()):
        item = dialog.tables.item(index)
        if item.text() in ("nucleus", "pathogen"):
            item.setCheckState(Qt.Checked)
            app.processEvents()
    dialog.name.setText("cell_merged")
    _pump(app, lambda: dialog.rules.rowCount() > 600)
    dialog.validate_preview()
    _pump(app, dialog.create.isEnabled)
    assert "First 12 output rows" in dialog.preview_text.toPlainText()
    dialog.accept()
    screen.use_derived_table(dialog.definition)
    _pump(app, lambda: screen._table_picker.currentText() == "cell_merged"
          and not screen.is_busy())
    deadline = time.monotonic() + 1.0
    _pump(app, lambda: time.monotonic() > deadline)
    timer.stop()
    beats.append(time.perf_counter())

    assert len(screen._frame) == 2400 and len(screen._frame.columns) > 900
    longest = max(b - a for a, b in zip(beats, beats[1:])) * 1000
    assert longest < BUDGET_MS, f"the GUI thread was blocked for {longest:.0f} ms"
