"""The lineage screen keeps talking when the thing it asked for fails.

Reading a database, opening crops and naming collisions are three things the
screen delegates to code it does not control. Each one failing has to leave a
sentence on the status line rather than an exception in a signal handler,
because a lineage tree with no message is indistinguishable from a project
that genuinely has no lineage.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import lineage as screen_module

F1 = ("plate1", "r1", "c1", "f1")


def _rows(table, entries):
    out = []
    for field, label, parent in entries:
        plate, row, column, field_id = field
        record = {"plateID": plate, "rowID": row, "columnID": column,
                  "fieldID": field_id, "object_label": label,
                  f"{table}_area": 100.0 + label}
        if parent is not None:
            record["cell_id"] = parent
        out.append(record)
    return pd.DataFrame(out)


@pytest.fixture()
def frames():
    """One cell holding one nucleus, which is enough to build a tree."""
    return {
        "cell": _rows("cell", [(F1, 7, None)]),
        "nucleus": _rows("nucleus", [(F1, 1, 7)]),
    }


@pytest.fixture()
def db(tmp_path, frames):
    path = tmp_path / "measurements.db"
    connection = sqlite3.connect(path)
    try:
        for table, frame in frames.items():
            frame.to_sql(table, connection, index=False)
    finally:
        connection.close()
    return str(path)


@pytest.fixture()
def screen(qtbot):
    view = screen_module.LineageScreen(threaded=False)
    qtbot.addWidget(view)
    return view


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_browsing_to_a_database_loads_it(screen, db, monkeypatch):
    """Browse is the same act as typing the path and pressing the button."""
    monkeypatch.setattr(screen_module.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (db, "")))

    screen._choose_db()

    assert screen._db.text() == db
    assert screen.tree.topLevelItemCount() == 1


def test_cancelling_the_browse_dialog_loads_nothing(screen, monkeypatch):
    """A cancelled dialog must not blank the field or start a read."""
    screen._db.setText("/somewhere/measurements.db")
    monkeypatch.setattr(screen_module.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))

    screen._choose_db()

    assert screen._db.text() == "/somewhere/measurements.db"
    assert screen.tree.topLevelItemCount() == 0


def test_a_read_that_fails_says_so_on_the_status_line(screen):
    """A worker failure reaches the user as words, in the error colour.

    Silence here reads as "this project has no lineage", which is a
    different and much more alarming fact than "the file would not open".
    """
    screen._on_job_failed("no measurements database at /nowhere")

    assert screen.status.text() == "no measurements database at /nowhere"
    assert "color:" in screen.status.styleSheet()
