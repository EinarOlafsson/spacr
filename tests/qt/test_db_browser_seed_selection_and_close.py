"""Database Browser: the seed hand-off, programmatic row selection and close.

``apply_seed`` is how another screen says "open this database on this table at
this column"; it fails soft, so a table or column this database lacks (or a
database that cannot list its tables) leaves the browser where it was. Row
selection is walked over a table of requests -- in range, out of range, before
any table, with no selection model -- and an empty view selection must never
be published to the shared link. Closing survives a half-gone link.
"""
from __future__ import annotations

import sqlite3

import pytest

from spacr.qt.screens.db_browser import DbBrowserScreen

pytestmark = pytest.mark.qt

_KEYED = ("plateID", "rowID", "columnID", "fieldID", "object_label")


@pytest.fixture
def measurements_db(tmp_path):
    """A keyed ``cell`` table (5 rows) and a keyless ``notes`` one (3 rows)."""
    path = tmp_path / "plate1" / "measurements" / "measurements.db"
    path.parent.mkdir(parents=True)
    con = sqlite3.connect(path)
    try:
        con.execute(
            "CREATE TABLE cell (%s, cell_area REAL)"
            % ", ".join(f'"{c}" TEXT' for c in _KEYED))
        con.executemany(
            "INSERT INTO cell VALUES (?, ?, ?, ?, ?, ?)",
            [("plate1", "r1", "c1", "f1", str(i), 100.0 + i)
             for i in range(5)])
        con.execute("CREATE TABLE notes (name TEXT, n INTEGER)")
        con.executemany("INSERT INTO notes VALUES (?, ?)",
                        [(f"note{i}", i) for i in range(3)])
        con.commit()
    finally:
        con.close()
    return str(path)


@pytest.fixture
def screen(qtbot):
    """A synchronous browser with nothing opened yet."""
    widget = DbBrowserScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


def _scrolls(widget):
    """Record the column of every ``scrollTo`` the view is asked for."""
    reached = []
    widget._view.scrollTo = (
        lambda index, *a, **k: reached.append(index.column()))
    return reached


def test_apply_seed_opens_what_it_names_and_ignores_what_it_cannot(
        screen, measurements_db, monkeypatch, subtests):
    """Database, table and column land; stale names leave the browser put."""
    cell_columns = list(_KEYED) + ["cell_area"]
    reached = _scrolls(screen)

    with subtests.test(msg="a full seed opens the table and scrolls"):
        screen.apply_seed({"db_path": measurements_db, "table": "cell",
                           "column": "cell_area"})
        assert screen.current_table() == "cell"
        assert screen.visible_columns() == cell_columns
        assert reached == [5]
        assert screen._seed_column is None

    with subtests.test(msg="a table this database lacks is ignored"):
        screen.apply_seed({"table": "no_such_table"})
        assert screen.current_table() == "cell"

    with subtests.test(msg="a database that cannot list tables is ignored"):
        def unreadable():
            raise sqlite3.DatabaseError("the file is locked")

        with monkeypatch.context() as patch:
            patch.setattr(screen._db, "tables", unreadable)
            screen.apply_seed({"table": "notes"})
        assert screen.current_table() == "cell"

    with subtests.test(msg="a column this table lacks is a miss"):
        reached.clear()
        screen.apply_seed({"db_path": measurements_db, "table": "notes",
                           "column": "cell_area"})
        assert screen.current_table() == "notes"
        assert screen.visible_columns() == ["name", "n"]
        assert reached == []
        assert screen._seed_column is None


def test_a_threaded_seed_scrolls_to_its_column_once_the_rows_arrive(
        qtbot, measurements_db):
    """The shipping (threaded) browser has no columns when the seed lands."""
    widget = DbBrowserScreen()
    qtbot.addWidget(widget)
    reached = _scrolls(widget)

    widget.apply_seed({"db_path": measurements_db, "table": "cell",
                       "column": "cell_area"})
    assert widget.visible_columns() == []
    assert widget._seed_column == "cell_area"

    qtbot.waitUntil(lambda: bool(widget.visible_columns()), timeout=10000)
    assert reached == [5]
    assert widget._seed_column is None


def test_row_selection_keeps_only_real_rows_and_publishes_only_a_choice(
        screen, measurements_db, monkeypatch, subtests):
    """In-range rows select; anything else clears; empty is never published."""
    with subtests.test(msg="before any table there is nothing to select"):
        assert screen.select_rows([0, 1]) == []
        assert screen.selected_rows() == []

    screen.apply_seed({"db_path": measurements_db, "table": "cell"})
    assert screen.row_count() == 5
    cases = [
        ("in range rows are selected", [3, 2, 3], [2, 3]),
        ("a mix keeps only the rows in range", [4, 9, -1], [4]),
        ("all out of range clears the old highlight", [10, -1], []),
    ]
    for case_id, asked, got in cases:
        with subtests.test(msg=case_id):
            screen.select_rows([1])
            assert screen.select_rows(asked) == got
            assert screen.selected_rows() == got

    attempts = []
    real = screen.publish_selection

    def spy(*args, **kwargs):
        attempts.append(len(args[0]))
        return real(*args, **kwargs)

    with subtests.test(msg="an empty view selection is not published"):
        screen.select_rows([])
        screen.link.clear_selection()
        screen.publish_selection = spy
        screen._on_view_selection_changed()
        assert attempts == []
        assert screen.link.selection.is_active is False

    with subtests.test(msg="a chosen row is published to the link"):
        screen.select_rows([0, 2])
        attempts.clear()
        screen._on_view_selection_changed()
        assert attempts == [2]
        assert screen.link.selection.is_active is True
        screen.link.clear_selection()

    with subtests.test(msg="a view with no selection model selects nothing"):
        monkeypatch.setattr(screen._view, "selectionModel", lambda: None)
        assert screen.selected_rows() == []
        assert screen.select_rows([0]) == []


def test_closing_survives_a_link_whose_other_half_is_already_gone(
        screen, measurements_db, monkeypatch):
    """At interpreter teardown the link's C++ side may already be deleted."""
    screen.apply_seed({"db_path": measurements_db, "table": "cell"})
    screen.show()
    calls = []

    def gone():
        calls.append("unlink")
        raise RuntimeError("Internal C++ object already deleted")

    monkeypatch.setattr(screen, "unlink_selection", gone)
    assert screen.close() is True
    assert calls == ["unlink"]
    assert screen.isVisible() is False
