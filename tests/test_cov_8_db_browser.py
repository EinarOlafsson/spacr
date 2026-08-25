"""The Database Browser's two seams to the rest of the app, and their refusals.

The browser is reached two ways that are not a user typing in it. Another
screen can *seed* it -- "open this database on this table at this column" --
and the process-wide selection link can push a selection into it or pull one
out. Both are conveniences, so both fail soft: a stale seed opens what it
can and ignores the rest, and a table with no object identity in it
(``png_list``, a summary, somebody's own scratch table) simply does not take
part in the shared selection rather than raising into the view.

Those refusals are the behaviour under test here. A seed that opened nothing
and then went on to select a table, or a keyless table that raised out of a
selection handler, would both surface as the browser failing to open.
"""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens.db_browser import DbBrowserScreen      # noqa: E402
from spacr.selection import Selection, as_key_index          # noqa: E402

pytestmark = pytest.mark.qt


_KEYED = ("plateID", "rowID", "columnID", "fieldID", "object_label")


@pytest.fixture
def measurements_db(tmp_path):
    """A run folder holding a keyed ``cell`` table and a keyless ``notes`` one."""
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
        con.execute(
            "CREATE TABLE well_summary (plateID TEXT, rowID TEXT, "
            "columnID TEXT, mean_area REAL)")
        con.executemany(
            "INSERT INTO well_summary VALUES (?, ?, ?, ?)",
            [("plate1", "r1", f"c{i}", 100.0 + i) for i in range(3)])
        con.execute("CREATE TABLE notes (name TEXT, n INTEGER)")
        con.executemany("INSERT INTO notes VALUES (?, ?)",
                        [(f"note{i}", i) for i in range(3)])
        con.commit()
    finally:
        con.close()
    return str(path)


@pytest.fixture
def screen(qtbot, measurements_db):
    """A synchronous browser with nothing opened yet."""
    widget = DbBrowserScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# apply_seed
# ---------------------------------------------------------------------------

def test_a_seed_opens_the_database_table_and_column_it_names(
        screen, measurements_db):
    """The whole point of the seam: arrive where the other screen pointed."""
    screen.apply_seed({"db_path": measurements_db, "table": "cell",
                       "column": "cell_area"})

    assert screen.current_table() == "cell"
    assert "cell_area" in screen.visible_columns()


def test_a_seed_naming_a_database_that_will_not_open_stops_there(
        screen, tmp_path):
    """A stale seed must not go on to ask an unopened database for a table."""
    screen.apply_seed({"db_path": str(tmp_path / "gone" / "measurements.db"),
                       "table": "cell"})

    assert screen.current_table() == ""
    assert screen.status_text()


def test_a_seed_naming_a_table_this_database_lacks_is_ignored(
        screen, measurements_db):
    """A table name from another run leaves the browser on what it opened."""
    screen.apply_seed({"db_path": measurements_db, "table": "cell"})

    screen.apply_seed({"table": "no_such_table"})

    assert screen.current_table() == "cell"


def test_a_seed_survives_a_database_that_cannot_list_its_tables(
        screen, measurements_db, monkeypatch):
    """A database that fails mid-seed is ignored, not raised out of."""
    screen.apply_seed({"db_path": measurements_db, "table": "cell"})

    def unreadable():
        raise sqlite3.DatabaseError("the file is locked")

    monkeypatch.setattr(screen._db, "tables", unreadable)

    screen.apply_seed({"table": "notes"})

    assert screen.current_table() == "cell"


def test_a_seed_naming_a_column_this_table_lacks_is_ignored(
        screen, measurements_db):
    """A column from another table is a miss, not a failure to open."""
    screen.apply_seed({"db_path": measurements_db, "table": "notes",
                       "column": "cell_area"})

    assert screen.current_table() == "notes"
    assert "cell_area" not in screen.visible_columns()


# ---------------------------------------------------------------------------
# sorting
# ---------------------------------------------------------------------------

def test_a_header_click_outside_the_columns_changes_no_sort(
        screen, measurements_db):
    """Header sections outlive a column set; a stale one must do nothing."""
    screen.apply_seed({"db_path": measurements_db, "table": "cell"})
    screen._on_header_clicked(0)
    before = screen._sort

    screen._on_header_clicked(len(screen.visible_columns()) + 3)

    assert screen._sort == before


# ---------------------------------------------------------------------------
# the shared selection and filter
# ---------------------------------------------------------------------------

def _spy_on_publishing(screen):
    """Record every publish attempt while still running the real one."""
    attempts = []
    real = screen.publish_selection

    def spy(*args, **kwargs):
        attempts.append(args)
        return real(*args, **kwargs)

    screen.publish_selection = spy
    return attempts


class _BrokenLink:
    """A link whose shared state can no longer be read."""

    @property
    def filter(self):
        raise RuntimeError("the shared link is gone")


def test_a_shared_filter_that_cannot_be_read_hides_nothing(
        screen, measurements_db, monkeypatch):
    """An unreadable link costs the filter, never the rows on screen."""
    screen.apply_seed({"db_path": measurements_db, "table": "cell"})
    real_link = screen._link
    screen._link = _BrokenLink()
    try:
        screen._apply_linked_filter()

        assert screen.hidden_rows() == []
        assert screen.row_count() == 5
    finally:
        screen._link = real_link


@pytest.mark.parametrize("table", ["notes", "well_summary"])
def test_a_table_with_no_object_identity_answers_no_shared_selection(
        screen, measurements_db, table):
    """Neither a keyless table nor a well summary can name single objects.

    ``notes`` carries none of the key columns; ``well_summary`` carries the
    plate/row/column half and stops there, which is the shape that raises
    rather than returning an empty answer.
    """
    screen.apply_seed({"db_path": measurements_db, "table": table})
    assert screen.row_count() == 3
    selection = Selection(keys=as_key_index(["plate1_r1_c1_f1_1"]))

    assert screen.rows_for_selection(selection) == []


def test_selecting_a_row_of_a_keyless_table_publishes_nothing(
        screen, measurements_db):
    """Selecting in ``notes`` is a local act, not something others follow."""
    screen.apply_seed({"db_path": measurements_db, "table": "notes"})
    screen.link.clear_selection()
    screen.select_rows([0, 1])
    attempts = _spy_on_publishing(screen)

    screen._on_view_selection_changed()

    assert len(attempts) == 1, "the handler did try to publish"
    assert screen.link.selection.is_active is False


def test_an_empty_view_selection_is_not_published_as_a_selection(
        screen, measurements_db):
    """Qt clears the selection on every model reset; that is not a lasso."""
    screen.apply_seed({"db_path": measurements_db, "table": "cell"})
    screen.link.clear_selection()
    attempts = _spy_on_publishing(screen)

    screen._on_view_selection_changed()

    assert attempts == []
    assert screen.link.selection.is_active is False


def test_rows_cannot_be_selected_before_a_table_is_loaded(screen):
    """With no columns there is no row to select, and no error either."""
    assert screen.select_rows([0, 1]) == []
    assert screen.selected_rows() == []


def test_a_view_with_no_selection_model_reports_no_selected_rows(
        screen, measurements_db, monkeypatch):
    """A view being torn down must not make the browser raise."""
    screen.apply_seed({"db_path": measurements_db, "table": "cell"})
    monkeypatch.setattr(screen._view, "selectionModel", lambda: None)

    assert screen.selected_rows() == []


def test_an_incoming_selection_is_ignored_before_a_table_is_loaded(screen):
    """A link push can arrive before the user has opened anything."""
    selection = Selection(keys=as_key_index(["plate1_r1_c1_f1_1"]))

    assert screen.on_linked_selection_changed(selection) is None
    assert screen.selected_rows() == []


def test_closing_survives_a_link_whose_other_half_is_already_gone(
        screen, measurements_db, monkeypatch):
    """At interpreter teardown the link's C++ side may outlive nothing."""
    screen.apply_seed({"db_path": measurements_db, "table": "cell"})

    def gone():
        raise RuntimeError("Internal C++ object already deleted")

    monkeypatch.setattr(screen, "unlink_selection", gone)

    screen.close()

    assert screen.isVisible() is False
