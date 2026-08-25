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

from spacr import lineage as lin
from spacr.qt.linked_selection import (DEFAULT_OPEN_KIND,
                                       register_object_opener,
                                       unregister_object_opener)
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

def test_building_a_tree_with_no_database_named_asks_for_one(screen):
    """The button is always live, so pressing it early must explain itself."""
    screen.load()

    assert "Choose a measurements database first." in screen.status.text()


def test_a_path_that_is_not_a_file_asks_for_one_too(screen, tmp_path):
    """A typed path that does not exist is the same situation as none."""
    screen._db.setText(str(tmp_path / "not_here.db"))
    screen.load()

    assert "Choose a measurements database first." in screen.status.text()


def test_naming_a_real_database_builds_its_tree(screen, db):
    """The control: loading really does read the tables and fill the tree."""
    screen._db.setText(db)
    screen.load()

    assert screen.tree.topLevelItemCount() == 1
    assert "1 of 1 parent object(s)" in screen.status.text()


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


# ---------------------------------------------------------------------------
# Key collisions
# ---------------------------------------------------------------------------

def test_two_objects_sharing_one_key_are_announced_in_the_summary(
        screen, frames, monkeypatch):
    """The alarm for a key that names more than one object.

    Every other view treats an object key as unique, so a collision means
    opening a family shows fewer crops than the family has objects -- with
    nothing on screen saying why unless this note appears.
    """
    monkeypatch.setattr(
        lin, "forest_key_collisions",
        lambda _forest: {"plate1_r1_c1_f1_7": ["cell", "nucleus"]})

    screen.set_frames(frames)

    note = screen.collision_note()
    assert "1 object key(s) name more than one object" in note
    assert "plate1_r1_c1_f1_7 is a cell and a nucleus" in note
    assert note in screen.summary.text()


def test_no_collision_leaves_the_summary_free_of_alarms(screen, frames):
    """The control: the note is empty in the ordinary case."""
    screen.set_frames(frames)

    assert screen.collision_note() == ""
    assert "name more than one object" not in screen.summary.text()


# ---------------------------------------------------------------------------
# Opening crops
# ---------------------------------------------------------------------------

@pytest.fixture()
def opener():
    """Register something that can open crops, and take it away afterwards."""
    def open_it(request):
        return list(request.keys)

    register_object_opener(DEFAULT_OPEN_KIND, open_it)
    yield open_it
    unregister_object_opener(DEFAULT_OPEN_KIND, open_it)


def test_opening_with_nowhere_to_open_points_at_annotate(screen, frames):
    """Crops are shown by Annotate, so the message names it."""
    screen.set_frames(frames)

    assert screen._open(["plate1_r1_c1_f1_7"], "because") is None
    assert "Annotate screen" in screen.status.text()


def test_an_opener_that_raises_leaves_the_reason_on_the_status_line(
        screen, frames, opener, monkeypatch):
    """A destination that blows up must not take the lineage screen with it.

    The tree is still usable afterwards, and the user is told what went
    wrong instead of watching a double-click do nothing.
    """
    def refuse(_keys, **_kwargs):
        raise RuntimeError("the annotate screen has no crops for those")

    monkeypatch.setattr(screen, "open_objects", refuse)
    screen.set_frames(frames)

    assert screen._open(["plate1_r1_c1_f1_7"], "because") is None
    assert "Could not open those objects" in screen.status.text()
    assert "no crops for those" in screen.status.text()


def test_an_opener_that_works_returns_what_it_opened(screen, frames, opener):
    """The control, so the failure message above means something."""
    screen.set_frames(frames)

    assert screen._open(["plate1_r1_c1_f1_7"], "because") == [
        "plate1_r1_c1_f1_7"]
