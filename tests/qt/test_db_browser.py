"""Database Browser — the Tools module that replaces the sqlite3 CLI.

Everything here runs offscreen against a *real* temporary
``measurements.db`` built the way ``tests/conftest.py::synth_sqlite_db``
builds one (a wide ``cell`` feature table plus the ``png_list``
annotation table), just with enough rows to exercise paging.

The suite pins the four properties the panel lives or dies by:

* it is **registered** as a Tools app with a title, an intro and a real icon;
* it **pages** — ``LIMIT ? OFFSET ?``, never ``SELECT *`` over the table;
* it is **read-only** — writes are refused by SQLite and the file on
  disk is byte-identical after a full browse/filter/export cycle;
* it is **injection-safe** — values are bound, identifiers are checked
  against the live schema, and errors land inline instead of in a modal
  dialog (a QMessageBox would hang a headless run forever).
"""
from __future__ import annotations

import csv
import hashlib
import os
import sqlite3
import threading

import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from spacr.qt.screens.db_browser import (
    DEFAULT_PAGE_SIZE,
    DbBrowserScreen,
    OPERATORS,
    PreviewModel,
    ReadOnlyDb,
    build_where,
    quote_ident,
    resolve_db_path,
    validate_raw_predicate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_ROWS = 250

CELL_COLUMNS = (
    "plate", "row", "column", "well", "field", "prc", "object_label",
    "cell_area", "cell_channel_0_mean_intensity",
    "cell_channel_1_mean_intensity", "cell_channel_1_percentile_75",
    "nucleus_area", "pathogen_count",
)


def _make_rows():
    """Deterministic synthetic measurements — no RNG, so expectations are exact."""
    rows = []
    for i in range(N_ROWS):
        well = f"{chr(ord('A') + (i % 8))}{(i % 12) + 1:02d}"
        field = (i % 3) + 1
        rows.append((
            "plate1",                 # plate
            (i % 8) + 1,              # row
            (i % 12) + 1,             # column
            well,                     # well
            field,                    # field
            f"plate1_{well}_{field}",  # prc
            i + 1,                    # object_label
            100.0 + i,                # cell_area
            500.0 + i,                # cell_channel_0_mean_intensity
            1500.0 + i,               # cell_channel_1_mean_intensity
            2500.0 + i,               # cell_channel_1_percentile_75
            50.0 + i,                 # nucleus_area
            i % 5,                    # pathogen_count
        ))
    return rows


class _Db:
    """Handle on the temporary database + the data that went into it."""

    def __init__(self, src, path, rows):
        self.src = str(src)          # run folder (the `src` users type)
        self.path = str(path)        # <src>/measurements/measurements.db
        self.rows = rows
        self.columns = list(CELL_COLUMNS)

    def index(self, column):
        return self.columns.index(column)

    def digest(self):
        with open(self.path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    def siblings(self):
        """Files next to the .db — proves no -wal / -journal appears."""
        d = os.path.dirname(self.path)
        return sorted(os.listdir(d))


@pytest.fixture
def measdb(tmp_path):
    """A real ``<src>/measurements/measurements.db`` with 250 cell rows."""
    src = tmp_path / "plate1"
    meas = src / "measurements"
    meas.mkdir(parents=True)
    db_path = meas / "measurements.db"
    rows = _make_rows()

    con = sqlite3.connect(db_path)
    try:
        cols_sql = ", ".join(
            f'"{c}" ' + ("TEXT" if c in ("plate", "well", "prc")
                         else "REAL" if "area" in c or "intensity" in c
                         or "percentile" in c else "INTEGER")
            for c in CELL_COLUMNS)
        con.execute(f"CREATE TABLE cell ({cols_sql})")
        con.executemany(
            f"INSERT INTO cell VALUES ({', '.join('?' * len(CELL_COLUMNS))})",
            rows)
        # The annotation table many spacr helpers assume exists.
        con.execute("CREATE TABLE png_list (prc TEXT, annotation INTEGER)")
        con.executemany(
            "INSERT INTO png_list VALUES (?, ?)",
            [(r[CELL_COLUMNS.index("prc")], 0) for r in rows[:20]])
        con.commit()
    finally:
        con.close()
    return _Db(src, db_path, rows)


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Blow up loudly if any code path under test opens a modal dialog.

    ``MakeMasksScreen._load_current`` once hung the whole headless suite
    on a QMessageBox; this fixture makes that failure mode impossible to
    reintroduce here without a red test.
    """
    from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

    def _boom(*_a, **_k):
        raise AssertionError(
            "a modal dialog was opened — errors must be reported inline")

    for name in ("about", "critical", "information", "question", "warning"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_boom))
    for name in ("exec", "exec_", "open", "show"):
        monkeypatch.setattr(QMessageBox, name, _boom, raising=False)
    monkeypatch.setattr(QDialog, "exec", _boom, raising=False)
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(_boom))
    yield


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    """A synchronous screen — queries run inline so assertions are exact."""
    w = DbBrowserScreen(threaded=False)
    qtbot.addWidget(w)
    return w


# ---------------------------------------------------------------------------
# Registration: APPS / titles / intro / icon / MainWindow wiring
# ---------------------------------------------------------------------------

def test_db_browser_is_registered_as_a_tools_app():
    from spacr.qt.app import APPS
    entry = next((a for a in APPS if a[0] == "db_browser"), None)
    assert entry is not None, "db_browser missing from APPS"
    key, name, desc, section = entry
    assert name == "Database Browser"
    assert desc and desc.strip()
    assert section == "Tools", f"db_browser filed under {section!r}, want Tools"


def test_db_browser_has_a_title_and_an_intro():
    from spacr.qt.screens.app_screen import APP_INTROS, APP_TITLES
    assert APP_TITLES.get("db_browser", "").strip() == "Database Browser"
    intro = APP_INTROS.get("db_browser", "")
    assert len(intro.split()) >= 10, f"intro is too thin: {intro!r}"
    assert intro.endswith(".")


def test_db_browser_icon_resolves_to_a_real_resource_file():
    from spacr.qt import app as qt_app
    assert "db_browser" in qt_app._ICON_OVERRIDES, (
        "db_browser needs an _ICON_OVERRIDES entry — no binary was added")
    here = os.path.dirname(os.path.abspath(qt_app.__file__))
    path = os.path.normpath(os.path.join(
        here, "..", "resources", "icons", qt_app._ICON_OVERRIDES["db_browser"]))
    assert os.path.isfile(path), f"missing icon file: {path}"


def test_icon_provider_returns_a_non_null_icon(qtbot, qt_theme_applied):
    from PySide6.QtGui import QIcon
    from spacr.qt.app import _icon_for_app
    icon = _icon_for_app("db_browser")
    assert isinstance(icon, QIcon)
    assert not icon.isNull(), "db_browser icon is null — the PNG failed to load"


def test_sidebar_lists_the_database_browser(qtbot, qt_theme_applied):
    from PySide6.QtWidgets import QPushButton
    from spacr.qt.app import Sidebar
    bar = Sidebar()
    qtbot.addWidget(bar)
    labels = {b.accessibleName() for b in bar.findChildren(QPushButton)}
    assert "Database Browser" in labels


def test_main_window_builds_the_db_browser_screen(qtbot, qt_theme_applied):
    from spacr.qt.app import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    win._on_nav_selected("db_browser")
    assert isinstance(win._stack.currentWidget(), DbBrowserScreen)


def test_screen_builds_offscreen_without_raising(qtbot, qt_theme_applied):
    w = DbBrowserScreen()
    qtbot.addWidget(w)
    assert w.database_path() == ""
    assert w.current_table() == ""
    assert w.row_count() == 0
    # The read-only promise is stated in the UI, not just in the code.
    assert any("read-only" in lbl.text().lower()
               for lbl in w.findChildren(QLabel)), \
        "the screen must tell the user it is read-only"


# ---------------------------------------------------------------------------
# Picking a database
# ---------------------------------------------------------------------------

def test_resolve_db_path_accepts_the_file_itself(measdb):
    assert resolve_db_path(measdb.path) == os.path.abspath(measdb.path)


def test_resolve_db_path_accepts_a_run_src_folder(measdb):
    """`<src>/measurements/measurements.db`, the way the rest of spacr does it."""
    assert resolve_db_path(measdb.src) == os.path.abspath(measdb.path)


def test_resolve_db_path_accepts_the_measurements_folder(measdb):
    meas_dir = os.path.dirname(measdb.path)
    assert resolve_db_path(meas_dir) == os.path.abspath(measdb.path)


def test_resolve_db_path_rejects_empty_and_missing(tmp_path):
    with pytest.raises(ValueError):
        resolve_db_path("")
    with pytest.raises(FileNotFoundError):
        resolve_db_path(str(tmp_path / "nope" / "measurements.db"))
    empty = tmp_path / "empty_run"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_db_path(str(empty))


def test_set_database_from_src_folder_opens_the_db(screen, measdb):
    assert screen.set_database(measdb.src) is True
    assert screen.database_path() == os.path.abspath(measdb.path)
    assert screen.last_error == ""


def test_database_opened_signal_carries_the_resolved_path(qtbot, screen, measdb):
    with qtbot.waitSignal(screen.database_opened, timeout=2000) as blocker:
        screen.set_database(measdb.path)
    assert blocker.args[0] == os.path.abspath(measdb.path)


# ---------------------------------------------------------------------------
# Tables, preview, row count
# ---------------------------------------------------------------------------

def test_tables_are_listed_and_the_first_is_previewed(screen, measdb):
    screen.set_database(measdb.path)
    assert screen.tables() == ["cell", "png_list"]
    assert screen.current_table() == "cell"
    assert screen.preview_columns() == list(CELL_COLUMNS)
    assert len(screen.preview_rows()) == DEFAULT_PAGE_SIZE


def test_row_count_is_the_real_count_star(screen, measdb):
    screen.set_database(measdb.path)
    assert screen.row_count() == N_ROWS
    # ...and the preview holds only one page of it, not all 250 rows.
    assert len(screen.preview_rows()) == DEFAULT_PAGE_SIZE < N_ROWS


def test_selecting_another_table_reloads_the_preview(screen, measdb):
    screen.set_database(measdb.path)
    assert screen.select_table("png_list") is True
    assert screen.current_table() == "png_list"
    assert screen.preview_columns() == ["prc", "annotation"]
    assert screen.row_count() == 20


def test_selecting_an_unknown_table_reports_inline(screen, measdb):
    screen.set_database(measdb.path)
    assert screen.select_table("no_such_table") is False
    assert "no_such_table" in screen.status_text()
    assert screen.last_error
    # ...and the previous table is untouched.
    assert screen.current_table() == "cell"


# ---------------------------------------------------------------------------
# Paging
# ---------------------------------------------------------------------------

def test_page_two_returns_different_rows(screen, measdb):
    screen.set_database(measdb.path)
    label = measdb.index("object_label")
    page1 = [r[label] for r in screen.preview_rows()]
    screen.next_page()
    assert screen.page_index() == 1
    page2 = [r[label] for r in screen.preview_rows()]
    assert len(page2) == DEFAULT_PAGE_SIZE
    assert page1 != page2
    assert not set(page1) & set(page2), "pages overlap — OFFSET is wrong"
    assert page1 == list(range(1, 101))
    assert page2 == list(range(101, 201))
    screen.prev_page()
    assert screen.page_index() == 0
    assert [r[label] for r in screen.preview_rows()] == page1


def test_last_page_is_partial_and_next_stops_there(screen, measdb):
    screen.set_database(measdb.path)
    screen.next_page()
    screen.next_page()
    assert screen.page_index() == 2
    assert len(screen.preview_rows()) == N_ROWS - 2 * DEFAULT_PAGE_SIZE == 50
    screen.next_page()          # past the end — a no-op, not an error
    assert screen.page_index() == 2
    assert screen.last_error == ""


def test_paging_never_selects_the_whole_table(measdb):
    """The preview SQL is bounded; the page size itself is a bound param."""
    db = ReadOnlyDb(measdb.path)
    cols, rows = db.page("cell", limit=10, offset=30)
    assert len(rows) == 10
    assert "LIMIT ? OFFSET ?" in db.last_sql
    assert "SELECT *" not in db.last_sql
    assert '"cell_channel_1_percentile_75"' in db.last_sql
    label = cols.index("object_label")
    assert [r[label] for r in rows] == list(range(31, 41))


def test_changing_page_size_resets_to_the_first_page(screen, measdb):
    screen.set_database(measdb.path)
    screen.next_page()
    assert screen.page_index() == 1
    screen._page_size_box.setValue(25)
    assert screen.page_index() == 0
    assert screen.page_size() == 25
    assert len(screen.preview_rows()) == 25


def test_next_button_is_disabled_on_the_last_page(screen, measdb):
    screen.set_database(measdb.path)
    assert screen._btn_next.isEnabled()
    assert not screen._btn_prev.isEnabled()
    screen.next_page()
    screen.next_page()
    assert not screen._btn_next.isEnabled()
    assert screen._btn_prev.isEnabled()


# ---------------------------------------------------------------------------
# Column search
# ---------------------------------------------------------------------------

def test_column_search_narrows_and_clearing_restores(screen, measdb):
    screen.set_database(measdb.path)
    assert screen.visible_columns() == list(CELL_COLUMNS)

    screen.set_column_filter("percentile")
    assert screen.visible_columns() == ["cell_channel_1_percentile_75"]

    screen.set_column_filter("channel_1")
    assert screen.visible_columns() == [
        "cell_channel_1_mean_intensity", "cell_channel_1_percentile_75"]

    screen.set_column_filter("")
    assert screen.visible_columns() == list(CELL_COLUMNS)


def test_column_search_is_case_insensitive_and_shown_in_the_count(screen, measdb):
    screen.set_database(measdb.path)
    screen.set_column_filter("AREA")
    assert screen.visible_columns() == ["cell_area", "nucleus_area"]
    assert screen._col_count_label.text() == f"2 of {len(CELL_COLUMNS)} columns"
    screen.set_column_filter("")
    assert screen._col_count_label.text() == f"{len(CELL_COLUMNS)} columns"


def test_column_search_does_not_requery_the_database(screen, measdb):
    """500-column tables need this to be a view operation, not a round trip."""
    screen.set_database(measdb.path)
    before = screen._db.last_sql
    screen.set_column_filter("channel")
    screen.set_column_filter("")
    assert screen._db.last_sql == before
    # The underlying page is intact — only the mapping changed.
    assert len(screen.preview_rows()[0]) == len(CELL_COLUMNS)


def test_column_search_survives_paging(screen, measdb):
    screen.set_database(measdb.path)
    screen.set_column_filter("percentile")
    screen.next_page()
    assert screen.visible_columns() == ["cell_channel_1_percentile_75"]


def test_preview_model_maps_visible_columns_to_view_indices(qtbot):
    m = PreviewModel()
    m.set_page(["a_x", "b_y", "a_z"], [(1, 2, 3), (4, 5, 6)])
    assert m.columnCount() == 3
    m.set_column_filter("a_")
    assert m.columnCount() == 2
    assert m.rowCount() == 2
    assert m.data(m.index(1, 1)) == "6"
    assert m.headerData(1, Qt.Horizontal) == "a_z"
    m.set_column_filter("")
    assert m.columnCount() == 3


# ---------------------------------------------------------------------------
# Filtering + export
# ---------------------------------------------------------------------------

def test_structured_filter_narrows_the_row_count(screen, measdb):
    screen.set_database(measdb.path)
    expected = sum(1 for r in measdb.rows if r[measdb.index("well")] == "A01")
    assert expected == 11
    assert screen.set_filter("well", "=", "A01") is True
    assert screen.row_count() == expected
    assert screen.last_error == ""


def test_clearing_the_filter_restores_the_full_count(screen, measdb):
    screen.set_database(measdb.path)
    screen.set_filter("well", "=", "A01")
    screen.clear_filter()
    assert screen.where_clause() is None
    assert screen.row_count() == N_ROWS


def test_numeric_filter_compares_numerically(screen, measdb):
    screen.set_database(measdb.path)
    assert screen.set_filter("cell_area", ">", "300") is True
    expected = sum(1 for r in measdb.rows if r[measdb.index("cell_area")] > 300)
    assert screen.row_count() == expected > 0


def test_filtered_export_writes_a_csv_matching_the_filter(screen, measdb, tmp_path):
    screen.set_database(measdb.path)
    screen.set_filter("well", "=", "A01")
    out = tmp_path / "a01.csv"
    assert screen.export_csv(str(out)) is True

    with open(out, newline="", encoding="utf-8") as fh:
        got = list(csv.reader(fh))
    assert got[0] == list(CELL_COLUMNS)
    body = got[1:]
    expected = [r for r in measdb.rows if r[measdb.index("well")] == "A01"]
    assert len(body) == len(expected) == 11
    label = measdb.index("object_label")
    assert [int(r[label]) for r in body] == [r[label] for r in expected]
    assert {r[measdb.index("well")] for r in body} == {"A01"}
    assert "Exported 11 rows" in screen.status_text()


def test_export_is_not_limited_to_the_visible_page(screen, measdb, tmp_path):
    """Export walks the whole filtered result, not just the 100 rows on screen."""
    screen.set_database(measdb.path)
    out = tmp_path / "all.csv"
    assert screen.export_csv(str(out)) is True
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == N_ROWS + 1


def test_export_honours_the_column_search(screen, measdb, tmp_path):
    screen.set_database(measdb.path)
    screen.set_column_filter("area")
    out = tmp_path / "areas.csv"
    assert screen.export_csv(str(out)) is True
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["cell_area", "nucleus_area"]
    assert len(rows) == N_ROWS + 1


def test_raw_predicate_filter_works(screen, measdb):
    screen.set_database(measdb.path)
    assert screen.set_raw_filter("cell_area > 300 AND pathogen_count = 0") is True
    expected = sum(1 for r in measdb.rows
                   if r[measdb.index("cell_area")] > 300
                   and r[measdb.index("pathogen_count")] == 0)
    assert screen.row_count() == expected > 0


def test_malformed_filter_reports_inline_and_does_not_raise(screen, measdb):
    """No exception, no dialog (the autouse fixture would fire), just text."""
    screen.set_database(measdb.path)
    before = screen.row_count()
    assert screen.set_raw_filter("cell_area >>> ") is False
    assert screen.last_error
    assert screen.status_text().startswith("Filter error")
    # The screen is still usable afterwards.
    assert screen.row_count() == before
    screen.clear_filter()
    assert screen.row_count() == N_ROWS
    assert screen.last_error == ""


def test_filter_on_an_unknown_column_reports_inline(screen, measdb):
    screen.set_database(measdb.path)
    assert screen.set_raw_filter("no_such_column = 1") is False
    assert "Filter error" in screen.status_text()
    assert screen.set_filter("not_a_column", "=", "1") is False
    assert "Unknown column" in screen.status_text()


def test_empty_structured_value_is_treated_as_no_filter(screen, measdb):
    screen.set_database(measdb.path)
    assert screen.set_filter("well", "=", "") is True
    assert screen.where_clause() is None
    assert screen.row_count() == N_ROWS


# ---------------------------------------------------------------------------
# Read-only
# ---------------------------------------------------------------------------

def test_connection_refuses_writes(measdb):
    db = ReadOnlyDb(measdb.path)
    before = measdb.digest()
    con = db.connect()
    try:
        with pytest.raises(sqlite3.Error):
            con.execute("INSERT INTO cell (plate) VALUES ('hacked')")
            con.commit()
        with pytest.raises(sqlite3.Error):
            con.execute("UPDATE cell SET cell_area = 0")
            con.commit()
        with pytest.raises(sqlite3.Error):
            con.execute("DROP TABLE png_list")
            con.commit()
    finally:
        con.close()
    assert measdb.digest() == before, "the database file changed on disk"
    assert "png_list" in db.tables(refresh=True)


def test_a_full_browse_cycle_leaves_the_file_byte_identical(screen, measdb,
                                                             tmp_path):
    before, siblings = measdb.digest(), measdb.siblings()
    screen.set_database(measdb.src)
    screen.next_page()
    screen.set_column_filter("channel")
    screen.set_filter("cell_area", ">=", "200")
    screen.export_csv(str(tmp_path / "out.csv"))
    screen.select_table("png_list")
    screen.clear_filter()
    assert measdb.digest() == before
    # No -wal / -journal side files either — mode=ro never journals.
    assert measdb.siblings() == siblings


def test_read_only_uri_is_used(measdb):
    db = ReadOnlyDb(measdb.path)
    assert db.uri.startswith("file:")
    assert db.uri.endswith("?mode=ro")


def test_query_only_pragma_is_on(measdb):
    db = ReadOnlyDb(measdb.path)
    con = db.connect()
    try:
        assert con.execute("PRAGMA query_only").fetchone()[0] == 1
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Injection safety
# ---------------------------------------------------------------------------

EVIL = "'; DROP TABLE png_list; --"


def test_sql_metacharacter_value_is_bound_as_data(screen, measdb):
    screen.set_database(measdb.path)
    assert screen.set_filter("well", "=", EVIL) is True
    # It simply matches nothing — and png_list is still there.
    assert screen.row_count() == 0
    assert screen.last_error == ""
    assert "png_list" in screen._db.tables(refresh=True)
    assert screen.tables() == ["cell", "png_list"]


def test_metacharacter_value_survives_export(screen, measdb, tmp_path):
    screen.set_database(measdb.path)
    screen.set_filter("well", "=", EVIL)
    out = tmp_path / "evil.csv"
    assert screen.export_csv(str(out)) is True
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows == [list(CELL_COLUMNS)]     # header only
    assert "png_list" in screen._db.tables(refresh=True)


def test_build_where_binds_the_value_instead_of_formatting_it():
    sql, params = build_where("well", "=", EVIL, CELL_COLUMNS)
    assert sql == '"well" = ?'
    assert params == (EVIL,)
    assert "DROP" not in sql


def test_build_where_rejects_columns_that_are_not_in_the_schema():
    with pytest.raises(ValueError):
        build_where("well; DROP TABLE cell", "=", "A01", CELL_COLUMNS)
    with pytest.raises(ValueError):
        build_where("well", "no-such-op", "A01", CELL_COLUMNS)


def test_like_operators_escape_wildcards(measdb):
    db = ReadOnlyDb(measdb.path)
    cols = db.columns("cell")
    sql, params = build_where("well", "contains", "100%", cols)
    assert "ESCAPE" in sql
    assert params == ("%100\\%%",)
    assert db.count("cell", sql, params) == 0


def test_null_operators_take_no_parameters():
    sql, params = build_where("well", "is null", "ignored", CELL_COLUMNS)
    assert sql == '"well" IS NULL'
    assert params == ()
    assert OPERATORS["is null"][1] == 0


def test_quote_ident_escapes_embedded_quotes():
    assert quote_ident('weird"name') == '"weird""name"'


def test_validate_raw_predicate_refuses_scripts_and_writes():
    assert validate_raw_predicate(" cell_area > 1 ") == "cell_area > 1"
    for bad in ("a = 1; DROP TABLE cell",
                "a = 1 -- comment",
                "a = 1 /* comment */",
                "1=1 OR (SELECT 1); DELETE FROM cell",
                "DROP TABLE cell",
                "PRAGMA writable_schema = 1",
                ""):
        with pytest.raises(ValueError):
            validate_raw_predicate(bad)


def test_check_table_gate_keeps_identifiers_out_of_sql(measdb):
    db = ReadOnlyDb(measdb.path)
    with pytest.raises(ValueError):
        db.check_table('cell" ; DROP TABLE png_list; --')
    with pytest.raises(ValueError):
        db.columns("nope")
    with pytest.raises(ValueError):
        db.check_columns("cell", ["cell_area", "not_a_column"])


# ---------------------------------------------------------------------------
# Bad inputs
# ---------------------------------------------------------------------------

def test_missing_file_reports_inline(screen, tmp_path):
    assert screen.set_database(str(tmp_path / "gone" / "measurements.db")) is False
    assert screen.last_error
    assert screen.database_path() == ""
    assert screen.tables() == []


def test_non_sqlite_file_reports_inline(screen, tmp_path):
    junk = tmp_path / "notadb.db"
    junk.write_text("this is a CSV, not a database\n1,2,3\n" * 100)
    assert screen.set_database(str(junk)) is False
    assert "not a SQLite database" in screen.status_text()
    assert screen.last_error


def test_empty_path_reports_inline(screen):
    assert screen.set_database("") is False
    assert screen.last_error


def test_folder_without_a_database_reports_inline(screen, tmp_path):
    empty = tmp_path / "run_without_measurements"
    empty.mkdir()
    assert screen.set_database(str(empty)) is False
    assert "measurements.db" in screen.status_text()


def test_actions_are_disabled_until_a_database_is_open(screen, measdb):
    assert not screen._btn_apply.isEnabled()
    assert not screen._btn_export.isEnabled()
    screen.set_database(measdb.path)
    assert screen._btn_apply.isEnabled()
    assert screen._btn_export.isEnabled()


def test_export_without_a_database_reports_inline(screen, tmp_path):
    assert screen.export_csv(str(tmp_path / "nope.csv")) is False
    assert screen.last_error
    assert not (tmp_path / "nope.csv").exists()


def test_apply_filter_without_a_database_reports_inline(screen):
    assert screen.apply_filter() is False
    assert screen.last_error


# ---------------------------------------------------------------------------
# Off-thread execution
# ---------------------------------------------------------------------------

def test_query_runs_off_the_gui_thread(qtbot, qt_theme_applied, measdb,
                                        monkeypatch):
    """The COUNT(*) + page query must not execute on the GUI thread."""
    gui_thread = threading.get_ident()
    seen = []
    real_count = ReadOnlyDb.count

    def _spy(self, *a, **k):
        seen.append(threading.get_ident())
        return real_count(self, *a, **k)

    monkeypatch.setattr(ReadOnlyDb, "count", _spy)

    w = DbBrowserScreen(threaded=True)
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.job_finished, timeout=10000) as blocker:
        w.set_database(measdb.path)
    assert blocker.args[0] is True
    assert seen, "COUNT(*) never ran"
    assert all(t != gui_thread for t in seen), \
        "the query ran on the GUI thread — the window would freeze"
    assert w.row_count() == N_ROWS
    assert len(w.preview_rows()) == DEFAULT_PAGE_SIZE
    qtbot.waitUntil(lambda: not w.is_busy(), timeout=10000)


def test_threaded_export_reports_inline_on_failure(qtbot, qt_theme_applied,
                                                    measdb, tmp_path):
    w = DbBrowserScreen(threaded=True)
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.job_finished, timeout=10000):
        w.set_database(measdb.path)
    qtbot.waitUntil(lambda: not w.is_busy(), timeout=10000)
    # A directory is not a writable CSV target -> the worker raises, and the
    # screen must turn that into inline text, not a dialog or a crash.
    target = tmp_path / "a_directory"
    target.mkdir()
    with qtbot.waitSignal(w.job_finished, timeout=10000) as blocker:
        w.export_csv(str(target))
    assert blocker.args[0] is False
    assert w.last_error
    assert "failed" in w.status_text().lower()


def test_threaded_query_retires_its_thread(qtbot, qt_theme_applied, measdb):
    w = DbBrowserScreen(threaded=True)
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.job_finished, timeout=10000):
        w.set_database(measdb.path)
    qtbot.waitUntil(lambda: w.active_jobs() == 0, timeout=10000)
    assert w._thread is None and w._worker is None
    w.close()   # must not abort on a live QThread


def test_overlapping_threaded_jobs_do_not_drop_a_live_thread(
        qtbot, qt_theme_applied, measdb, tmp_path):
    """`worker.finished` frees the UI before `thread.finished` retires the
    thread, so job N+1 starts while job N is still winding down. Job N's
    retirement must not release job N+1's QThread — that used to abort the
    interpreter with "QThread: Destroyed while thread is still running"."""
    w = DbBrowserScreen(threaded=True)
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.job_finished, timeout=10000):
        w.set_database(measdb.path)
    for i in range(4):
        with qtbot.waitSignal(w.job_finished, timeout=10000) as blocker:
            w.export_csv(str(tmp_path / f"chain_{i}.csv"))
        assert blocker.args[0] is True
        assert w._thread is not None or w.active_jobs() == 0
    qtbot.waitUntil(lambda: w.active_jobs() == 0, timeout=10000)
    for i in range(4):
        assert (tmp_path / f"chain_{i}.csv").exists()
