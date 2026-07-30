"""Database Browser — the Tools module that replaces the sqlite3 CLI.

Everything here runs offscreen against a *real* temporary
``measurements.db`` built the way ``tests/conftest.py::synth_sqlite_db``
builds one (a wide ``cell`` feature table plus the ``png_list``
annotation table), just with enough rows to exercise chunked loading.

The suite pins the properties the panel lives or dies by:

* it is **registered** as a Tools app with a title, an intro and a real icon;
* it **chunks** — keyset paging on ``rowid``, never ``SELECT *`` over the
  table and never a deep ``OFFSET``;
* the count is **honest** — an estimate is always labelled as one;
* a load is **cancellable** — switching table or database mid-load must
  not race the abandoned rows into the new view;
* it is **read-only by default** — writes are refused by SQLite and the
  file on disk is byte-identical after a full browse/filter/export cycle;
* editing is **opt-in and guarded** — a Preferences switch, a database
  the user chose, an explicit confirmation, a unique row address, and a
  type check, all of which fail closed to read-only;
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
import time
import warnings

import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from spacr.qt.screens.db_browser import (
    DEFAULT_PAGE_SIZE,
    DbBrowserScreen,
    EditRefused,
    OPERATORS,
    PreviewModel,
    ReadOnlyDb,
    WritableDb,
    build_update,
    build_where,
    coerce_for_column,
    column_affinity,
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

    def read_all(self):
        """Every ``cell`` row, in rowid order, straight from the file."""
        con = sqlite3.connect(self.path)
        try:
            return con.execute(
                f"SELECT {', '.join(CELL_COLUMNS)} FROM cell "
                f"ORDER BY rowid").fetchall()
        finally:
            con.close()

    def typeof(self, column, rowid=1):
        con = sqlite3.connect(self.path)
        try:
            return con.execute(
                f"SELECT typeof({column}), {column} FROM cell "
                f"WHERE rowid = ?", (rowid,)).fetchone()
        finally:
            con.close()

    def exec(self, sql):
        """Run one DDL/DML statement against the file (fixtures only)."""
        con = sqlite3.connect(self.path)
        try:
            con.execute(sql)
            con.commit()
        finally:
            con.close()


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


@pytest.fixture
def other_db(tmp_path):
    """A second, structurally different database — for switch/reset tests."""
    path = tmp_path / "other" / "measurements.db"
    path.parent.mkdir(parents=True)
    con = sqlite3.connect(path)
    try:
        con.execute("CREATE TABLE other_table (name TEXT, n INTEGER)")
        con.executemany("INSERT INTO other_table VALUES (?, ?)",
                        [(f"n{i}", i) for i in range(7)])
        con.commit()
    finally:
        con.close()
    return str(path)


@pytest.fixture(autouse=True)
def _isolated_qsettings(monkeypatch, qt_theme_applied, tmp_path):
    """Route QSettings into a temp .ini so the edit preference is deterministic.

    Without this the browser would read (and the preference tests would
    write) the developer's real spaCR settings — and "editing is off by
    default" would pass or fail depending on whose machine it ran on.
    """
    from PySide6.QtCore import QCoreApplication, QSettings
    QCoreApplication.setOrganizationName("spacr-test")
    QCoreApplication.setApplicationName("qt-db-browser-test")
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope,
                      str(tmp_path / "qsettings"))
    QSettings("spacr", "qt").clear()
    try:
        from spacr.qt.first_run import mark_tour_seen
        mark_tour_seen()
    except Exception:
        pass
    yield


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Blow up loudly if any code path under test opens a modal dialog.

    ``MakeMasksScreen._load_current`` once hung the whole headless suite
    on a QMessageBox; this fixture makes that failure mode impossible to
    reintroduce here without a red test. The single deliberate dialog —
    the edit-mode confirmation — is injectable, so tests replace
    ``screen.confirm_edit_mode`` instead of letting it open.
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


@pytest.fixture
def allow_editing():
    """Turn the Preferences opt-in on (isolated QSettings, see above)."""
    from spacr.qt.preferences import (
        get_db_browser_editable, set_db_browser_editable,
    )
    set_db_browser_editable(True)
    # QSettings writes back lazily; if this ever reads False the tests
    # below would fail for a reason that has nothing to do with them.
    assert get_db_browser_editable() is True
    return True


@pytest.fixture
def editable(screen, measdb, allow_editing):
    """A screen with edit mode armed: preference on, confirmation accepted."""
    screen.confirm_edit_mode = lambda _msg: True
    screen.set_database(measdb.path)
    assert screen.enable_edit_mode() is True, screen.status_text()
    return screen


# ---------------------------------------------------------------------------
# Registration: APPS / titles / intro / icon / MainWindow wiring
# ---------------------------------------------------------------------------

def test_db_browser_is_registered_under_data_as_alpha():
    """It is how measurements get back out of a project — that is Data.

    It spent #16i filed under "Alpha modules", a section that no longer
    exists. Not signed off is a stage now, drawn as the tile's hover
    colour, and it is asserted separately here."""
    from spacr.qt.app import APPS
    entry = next((a for a in APPS if a[0] == "db_browser"), None)
    assert entry is not None, "db_browser missing from APPS"
    key, name, desc, section = entry
    assert name == "Database Browser"
    assert desc and desc.strip()
    from spacr.qt.app import SECTION_DATA, app_stage
    assert section == SECTION_DATA, (
        f"db_browser filed under {section!r}; it is how measurements "
        "get back out of a project")
    assert app_stage(key) == "alpha"


def test_db_browser_has_a_title_and_an_intro():
    from spacr.qt.screens.app_screen import APP_INTROS, APP_TITLES
    assert APP_TITLES.get("db_browser", "").strip() == "Database Browser"
    intro = APP_INTROS.get("db_browser", "")
    assert len(intro.split()) >= 10, f"intro is too thin: {intro!r}"
    assert intro.endswith(".")


def test_db_browser_icon_resolves_to_a_real_resource_file():
    """A file on disk backs the tile, wherever the name comes from.

    This used to *require* an ``_ICON_OVERRIDES`` entry, because no
    binary had been added and the screen borrowed ``map_barcodes.png``
    (ruled bars read as table columns) — which meant Database Browser
    and Map Barcodes drew the same picture. The user has since chosen
    artwork for it, installed as ``db_browser.png``, which ``app_icon``
    finds with no override at all."""
    from spacr.qt import app as qt_app
    here = os.path.dirname(os.path.abspath(qt_app.__file__))
    filename = qt_app._ICON_OVERRIDES.get("db_browser", "db_browser.png")
    path = os.path.normpath(
        os.path.join(here, "..", "resources", "icons", filename))
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
    assert w.acceptDrops() is True
    assert type(w._dnd_handler).__name__ == "DatabaseDropHandler"


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


def test_the_file_pickers_open_what_the_user_chose(screen, measdb, monkeypatch):
    """The two 'Choose…' buttons, with the dialog stubbed out."""
    from PySide6.QtWidgets import QFileDialog
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (measdb.path, "")))
    screen._btn_pick_db.click()
    assert screen.database_path() == os.path.abspath(measdb.path)

    screen.set_database("")             # back to nothing
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: measdb.src))
    screen._btn_pick_src.click()
    assert screen.database_path() == os.path.abspath(measdb.path)


def test_a_cancelled_file_picker_changes_nothing(screen, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    screen._btn_pick_db.click()
    screen._btn_pick_src.click()
    assert screen.database_path() == ""
    assert screen.last_error == ""


def test_typing_a_path_and_pressing_return_opens_it(screen, measdb):
    screen._path_edit.setText(measdb.src)
    screen._path_edit.returnPressed.emit()
    assert screen.database_path() == os.path.abspath(measdb.path)
    assert screen.tables() == ["cell", "png_list"]


def test_the_export_picker_uses_the_table_name_as_a_default(screen, measdb,
                                                             tmp_path,
                                                             monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    screen.set_database(measdb.path)
    seen = []
    out = tmp_path / "picked.csv"

    def _fake(_parent, _title, default, _filter):
        seen.append(default)
        return str(out), ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(_fake))
    screen._btn_export.click()
    assert seen == ["cell.csv"]
    assert out.exists()

    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    screen._btn_export.click()          # cancelled — nothing more is written


def test_an_unopenable_file_is_explained_in_plain_words():
    msg = DbBrowserScreen._humanise(
        sqlite3.OperationalError("unable to open database file"), "/nope.db")
    assert msg.startswith("Could not open /nope.db")


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
    assert screen.row_count_is_estimate() is False
    # ...and only one chunk of it is in memory, not all 250 rows.
    assert screen.loaded_rows() == DEFAULT_PAGE_SIZE < N_ROWS


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
# Chunked loading — the first paint must not wait for the table
# ---------------------------------------------------------------------------

def _sql_spy(monkeypatch):
    """Record every statement ReadOnlyDb issues, in order."""
    seen = []
    real = ReadOnlyDb._execute

    def spy(self, con, sql, params=()):
        seen.append(sql)
        return real(self, con, sql, params)

    monkeypatch.setattr(ReadOnlyDb, "_execute", spy)
    return seen


def test_the_first_chunk_is_painted_before_the_table_is_counted(
        screen, measdb, monkeypatch):
    """One bounded SELECT, then COUNT(*) — never the other way round."""
    seen = _sql_spy(monkeypatch)
    screen.set_database(measdb.path)

    selects = [i for i, s in enumerate(seen)
               if s.startswith('SELECT "_rowid_", "plate"')]
    counts = [i for i, s in enumerate(seen) if "COUNT(*)" in s]
    assert selects and counts, f"expected both queries, got {seen}"
    assert selects[0] < counts[0], (
        "COUNT(*) ran before the first chunk — the user waits on a full "
        "scan before seeing a single row")
    assert len(counts) == 1, "COUNT(*) ran more than once"
    # The whole table was never read: one chunk of 100 out of 250 rows.
    assert screen.loaded_rows() == DEFAULT_PAGE_SIZE
    assert screen.row_count() == N_ROWS


def test_chunks_are_keyset_paged_and_never_use_offset(measdb):
    """OFFSET makes chunk 500 cost 500× chunk 1. Keyset does not."""
    db = ReadOnlyDb(measdb.path)
    cols, rows, keys = db.chunk("cell", limit=10)
    first_sql = db.last_sql
    assert "LIMIT ?" in first_sql
    assert "OFFSET" not in first_sql
    assert "SELECT *" not in first_sql
    assert first_sql.startswith('SELECT "_rowid_", "plate"')
    assert 'ORDER BY "_rowid_"' in first_sql
    assert keys == [(i,) for i in range(1, 11)]

    cols, rows, keys = db.chunk("cell", limit=10, after=keys[-1])
    assert '"_rowid_" > ?' in db.last_sql
    assert "OFFSET" not in db.last_sql
    label = cols.index("object_label")
    assert [r[label] for r in rows] == list(range(11, 21))


def test_fetching_more_appends_the_next_rows(screen, measdb):
    screen.set_database(measdb.path)
    label = measdb.index("object_label")
    first = [r[label] for r in screen.preview_rows()]
    assert first == list(range(1, 101))

    assert screen.fetch_more() is True
    got = [r[label] for r in screen.preview_rows()]
    assert len(got) == 200
    assert got == list(range(1, 201)), "chunks overlapped or skipped rows"


def test_scrolling_eventually_reaches_the_true_row_count(screen, measdb):
    screen.set_database(measdb.path)
    guard = 0
    while screen.fetch_more():
        guard += 1
        assert guard < 20, "fetch_more never terminated"
    assert screen.loaded_rows() == N_ROWS
    assert screen.row_count() == N_ROWS
    assert screen.row_count_is_estimate() is False
    assert screen.is_fully_loaded() is True
    label = measdb.index("object_label")
    assert [r[label] for r in screen.preview_rows()] == list(range(1, N_ROWS + 1))


def test_the_model_asks_for_more_through_can_fetch_more(screen, measdb):
    """The Qt-idiomatic path: the view calls canFetchMore/fetchMore."""
    screen.set_database(measdb.path)
    model = screen._model
    assert model.canFetchMore() is True
    model.fetchMore()
    assert screen.loaded_rows() == 2 * DEFAULT_PAGE_SIZE
    while screen.fetch_more():
        pass
    assert model.canFetchMore() is False
    model.fetchMore()       # a no-op, not an error
    assert screen.loaded_rows() == N_ROWS


def test_can_fetch_more_is_false_for_a_child_index(screen, measdb):
    screen.set_database(measdb.path)
    model = screen._model
    child = model.index(0, 0)
    assert model.canFetchMore(child) is False
    model.fetchMore(child)
    assert screen.loaded_rows() == DEFAULT_PAGE_SIZE


def test_a_second_fetch_is_ignored_while_one_is_in_flight(screen, measdb):
    """A scroll must not queue ten chunk jobs."""
    screen.set_database(measdb.path)
    calls = []
    real = DbBrowserScreen._fetch_chunk

    def reentrant(self, token, first):
        calls.append(first)
        self._chunk_jobs += 1        # pretend a job is still running
        try:
            assert self.fetch_more() is False
        finally:
            self._chunk_jobs -= 1
        return real(self, token, first)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(DbBrowserScreen, "_fetch_chunk", reentrant)
        assert screen.fetch_more() is True
    assert calls == [False]


def test_an_empty_table_loads_cleanly(screen, measdb):
    measdb.exec("CREATE TABLE empty_table (a INTEGER, b TEXT)")
    screen.set_database(measdb.path)
    assert screen.select_table("empty_table") is True
    assert screen.preview_rows() == []
    assert screen.row_count() == 0
    assert screen.row_count_is_estimate() is False
    assert screen.is_fully_loaded() is True
    assert screen.fetch_more() is False
    assert screen.last_error == ""
    assert "0 rows" in screen._rows_label.text()


def test_a_table_with_exactly_one_chunk_loads_cleanly(screen, measdb):
    measdb.exec("CREATE TABLE exact_page AS "
                "SELECT * FROM cell LIMIT 100")
    screen.set_database(measdb.path)
    assert screen.select_table("exact_page") is True
    assert screen.loaded_rows() == DEFAULT_PAGE_SIZE
    # A full chunk is ambiguous, so COUNT(*) settles it — and 100 == 100
    # means there is nothing left to fetch.
    assert screen.row_count() == DEFAULT_PAGE_SIZE
    assert screen.is_fully_loaded() is True
    assert screen.row_count_is_estimate() is False
    assert screen.fetch_more() is False


def test_changing_the_fetch_size_reloads_from_the_start(screen, measdb):
    screen.set_database(measdb.path)
    screen.fetch_more()
    assert screen.loaded_rows() == 200
    screen._page_size_box.setValue(25)
    assert screen.page_size() == 25
    assert screen.loaded_rows() == 25
    assert len(screen.preview_rows()) == 25


def test_load_more_button_switches_off_at_the_end_of_the_table(screen, measdb):
    screen.set_database(measdb.path)
    assert screen._btn_more.isEnabled()
    while screen.fetch_more():
        pass
    assert not screen._btn_more.isEnabled()


def test_fetch_more_without_a_database_is_a_no_op(screen):
    assert screen.fetch_more() is False
    assert screen.last_error == ""


def test_refresh_without_a_database_is_a_no_op(screen):
    screen.refresh()
    assert screen.row_count() == 0


# ---------------------------------------------------------------------------
# Counting — an estimate must never masquerade as a fact
# ---------------------------------------------------------------------------

def test_an_estimated_count_is_labelled_as_an_estimate(screen, measdb):
    screen.auto_count = False           # browse on the estimate alone
    screen.set_database(measdb.path)
    assert screen.row_count_is_estimate() is True
    text = screen._rows_label.text()
    assert "estimate" in text, f"estimate shown as fact: {text!r}"
    assert "≈" in text
    assert "estimate" in screen.status_text()
    # max(rowid) happens to be exact here, but it is still labelled.
    assert screen.row_count() == N_ROWS


def test_the_estimate_is_replaced_by_the_exact_count(screen, measdb):
    screen.auto_count = False
    screen.set_database(measdb.path)
    assert screen.row_count_is_estimate() is True
    screen.refresh_count()
    assert screen.row_count_is_estimate() is False
    assert screen.row_count() == N_ROWS
    assert "estimate" not in screen._rows_label.text()
    assert screen._rows_label.text() == "showing 100 of 250 rows"


def test_max_rowid_overestimates_after_deletes_and_says_so(screen, measdb):
    """Deleted rows leave gaps: max(rowid) is an upper bound, not a count."""
    measdb.exec("DELETE FROM cell WHERE rowid % 2 = 0")
    screen.auto_count = False
    screen.set_database(measdb.path)
    assert screen.row_count() == 249             # max(rowid), not the count
    assert screen.row_count_is_estimate() is True
    assert "estimate" in screen._rows_label.text()
    screen.refresh_count()
    assert screen.row_count() == 125             # the truth
    assert screen.row_count_is_estimate() is False


def test_a_filtered_load_has_no_estimate_and_says_counting(screen, measdb,
                                                            monkeypatch):
    """max(rowid) says nothing about a filtered result, so we don't pretend."""
    screen.auto_count = False
    screen.set_database(measdb.path)
    assert screen.set_filter("cell_area", ">=", "0") is True
    assert screen.row_count_is_estimate() is True
    assert "counting" in screen._rows_label.text()
    assert screen.row_count() == screen.loaded_rows()
    screen.refresh_count()
    assert screen.row_count() == N_ROWS


def test_estimate_count_returns_none_for_a_view(measdb):
    measdb.exec("CREATE VIEW cell_view AS SELECT * FROM cell")
    db = ReadOnlyDb(measdb.path)
    assert db.estimate_count("cell_view") is None
    assert db.estimate_count("cell") == N_ROWS


def test_estimate_count_returns_none_for_an_empty_table(measdb):
    measdb.exec("CREATE TABLE no_rows_here (a INTEGER)")
    db = ReadOnlyDb(measdb.path)
    assert db.estimate_count("no_rows_here") is None


def test_refresh_count_without_a_table_is_a_no_op(screen):
    screen.refresh_count()
    assert screen.row_count() == 0


# ---------------------------------------------------------------------------
# Cancellation — the bug that makes async loading worse than sync loading
# ---------------------------------------------------------------------------

def test_a_stale_chunk_is_dropped_instead_of_painted(screen, measdb):
    screen.set_database(measdb.path)
    before = screen.preview_rows()
    stale = {"token": screen._token - 1, "first": True,
             "columns": ["hacked"], "rows": [("boom",)] * 5,
             "keys": [(1,)] * 5, "limit": 100, "estimate": 9999}
    screen._apply_chunk(stale)
    assert screen.preview_rows() == before
    assert screen.preview_columns() == list(CELL_COLUMNS)
    screen._apply_chunk({})              # a job that returned nothing
    assert screen.preview_rows() == before


def test_a_stale_count_is_dropped(screen, measdb):
    screen.set_database(measdb.path)
    screen._apply_count({"token": screen._token - 1, "count": 999999})
    assert screen.row_count() == N_ROWS


def _slow_chunk(monkeypatch, table, delay=0.3):
    """Make chunks of one table slow, so a switch can land mid-load."""
    real = ReadOnlyDb.chunk

    def slow(self, name, *a, **k):
        if name == table:
            time.sleep(delay)
        return real(self, name, *a, **k)

    monkeypatch.setattr(ReadOnlyDb, "chunk", slow)


def test_switching_table_mid_load_abandons_the_first_load(
        qtbot, qt_theme_applied, measdb, monkeypatch):
    _slow_chunk(monkeypatch, "cell")
    w = DbBrowserScreen(threaded=True)
    qtbot.addWidget(w)
    w.set_database(measdb.path)          # starts the slow 'cell' chunk
    assert w.current_table() == "cell"
    w.select_table("png_list")           # ...and moves on before it lands
    qtbot.waitUntil(lambda: not w.is_busy(), timeout=20000)
    qtbot.waitUntil(lambda: w.active_jobs() == 0, timeout=20000)

    assert w.current_table() == "png_list"
    assert w.preview_columns() == ["prc", "annotation"]
    assert len(w.preview_rows()) == 20, \
        "the abandoned load raced its rows into the new table's view"
    assert w.row_count() == 20
    w.close()


def test_switching_database_mid_load_abandons_the_first_load(
        qtbot, qt_theme_applied, measdb, other_db, monkeypatch):
    _slow_chunk(monkeypatch, "cell")
    w = DbBrowserScreen(threaded=True)
    qtbot.addWidget(w)
    w.set_database(measdb.path)
    w.set_database(other_db)             # different file, mid-load
    qtbot.waitUntil(lambda: not w.is_busy(), timeout=20000)
    qtbot.waitUntil(lambda: w.active_jobs() == 0, timeout=20000)

    assert w.database_path() == os.path.abspath(other_db)
    assert w.current_table() == "other_table"
    assert w.preview_columns() == ["name", "n"]
    assert len(w.preview_rows()) == 7
    assert w.row_count() == 7
    w.close()


def test_a_superseded_load_is_dropped_before_it_costs_a_thread(
        qtbot, qt_theme_applied, measdb, monkeypatch):
    """Jobs queue behind the running one; stale ones never start at all."""
    _slow_chunk(monkeypatch, "cell")
    w = DbBrowserScreen(threaded=True)
    qtbot.addWidget(w)
    w.set_database(measdb.path)          # slow 'cell' chunk, now running
    w.select_table("png_list")           # queued behind it...
    w.select_table("cell")               # ...and superseded before it starts
    assert w.queued_jobs() >= 2
    qtbot.waitUntil(lambda: not w.is_busy(), timeout=20000)
    qtbot.waitUntil(lambda: w.active_jobs() == 0, timeout=20000)

    assert w.queued_jobs() == 0
    assert w.current_table() == "cell"
    assert w.preview_columns() == list(CELL_COLUMNS)
    assert w.loaded_rows() == DEFAULT_PAGE_SIZE
    assert w.row_count() == N_ROWS
    w.close()


def test_closing_drops_the_queue(qtbot, qt_theme_applied, measdb, monkeypatch):
    _slow_chunk(monkeypatch, "cell")
    w = DbBrowserScreen(threaded=True)
    qtbot.addWidget(w)
    w.set_database(measdb.path)
    w.select_table("png_list")
    assert w.queued_jobs() >= 1
    w.close()
    assert w.queued_jobs() == 0
    qtbot.waitUntil(lambda: not w.is_busy(), timeout=20000)


def test_a_cancelled_load_does_not_leave_the_screen_busy(
        qtbot, qt_theme_applied, measdb, monkeypatch):
    """Bookkeeping runs for abandoned jobs even though painting does not."""
    _slow_chunk(monkeypatch, "cell")
    w = DbBrowserScreen(threaded=True)
    qtbot.addWidget(w)
    w.set_database(measdb.path)
    w.select_table("png_list")
    qtbot.waitUntil(lambda: not w.is_busy(), timeout=20000)
    assert w._pending == {}
    assert w._chunk_jobs == 0
    assert w._load_jobs == 0
    qtbot.waitUntil(lambda: w.active_jobs() == 0, timeout=10000)
    w.close()


# ---------------------------------------------------------------------------
# Sorting — correct, or off and said so
# ---------------------------------------------------------------------------

def test_sorting_is_off_until_the_whole_table_is_loaded(screen, measdb):
    screen.set_database(measdb.path)
    assert screen._view.isSortingEnabled() is False
    assert "Sorting is off" in screen._sort_note.text()
    while screen.fetch_more():
        pass
    assert screen._view.isSortingEnabled() is True
    assert "click a column header to sort" in screen._sort_note.text()


def test_sorting_a_fully_loaded_table_sorts_every_row(screen, measdb):
    screen.set_database(measdb.path)
    while screen.fetch_more():
        pass
    col = screen.visible_columns().index("cell_area")
    screen._model.sort(col, Qt.DescendingOrder)
    label = measdb.index("object_label")
    assert [r[label] for r in screen.preview_rows()][:3] == [250, 249, 248]
    screen._model.sort(col, Qt.AscendingOrder)
    assert [r[label] for r in screen.preview_rows()][:3] == [1, 2, 3]


def test_sorting_keeps_each_row_with_its_own_key(editable, measdb):
    """After a sort, an edit must still hit the row it looks like it hits."""
    while editable.fetch_more():
        pass
    col = editable.visible_columns().index("cell_area")
    editable._model.sort(col, Qt.DescendingOrder)
    assert editable._model.row_key(0) == (N_ROWS,)      # the biggest area
    assert editable.edit_cell(0, "pathogen_count", "42") is True
    after = measdb.read_all()
    changed = [i for i, r in enumerate(after)
               if r != tuple(measdb.rows[i])]
    assert changed == [N_ROWS - 1]


def test_sort_ignores_impossible_columns(qtbot):
    m = PreviewModel()
    m.sort(0)                       # no rows at all
    m.set_page(["a"], [(2,), (1,)])
    m.sort(-1)                      # the "no sort indicator" case
    assert m.rows() == [(2,), (1,)]
    m.sort(9)
    assert m.rows() == [(2,), (1,)]


def test_sort_orders_nulls_numbers_then_text(qtbot):
    m = PreviewModel()
    m.set_page(["a"], [("zz",), (None,), (3,), (1.5,)])
    m.sort(0, Qt.AscendingOrder)
    assert [r[0] for r in m.rows()] == [None, 1.5, 3, "zz"]


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
    # The underlying rows are intact — only the mapping changed.
    assert len(screen.preview_rows()[0]) == len(CELL_COLUMNS)


def test_column_search_survives_fetching_more(screen, measdb):
    screen.set_database(measdb.path)
    screen.set_column_filter("percentile")
    screen.fetch_more()
    assert screen.visible_columns() == ["cell_channel_1_percentile_75"]
    assert screen.loaded_rows() == 200


def test_preview_model_maps_visible_columns_to_view_indices(qtbot):
    m = PreviewModel()
    m.set_page(["a_x", "b_y", "a_z"], [(1, 2, 3), (4, 5, 6)])
    assert m.columnCount() == 3
    m.set_column_filter("a_")
    assert m.column_filter() == "a_"
    assert m.columnCount() == 2
    assert m.rowCount() == 2
    assert m.data(m.index(1, 1)) == "6"
    assert m.headerData(1, Qt.Horizontal) == "a_z"
    m.set_column_filter(None)
    assert m.column_filter() == ""
    assert m.columnCount() == 3


def test_preview_model_formats_values_for_display_and_editing(qtbot):
    m = PreviewModel()
    m.set_page(["a", "b", "c", "d"],
               [(None, 1.0 / 3.0, b"\x00\x01\x02", "text")])
    assert m.data(m.index(0, 0)) == ""
    assert m.data(m.index(0, 1)) == "0.333333"
    assert m.data(m.index(0, 2)) == "<3 bytes>"
    assert m.data(m.index(0, 3)) == "text"
    # ...but the editor sees the exact stored float, not the rounded one.
    assert float(m.data(m.index(0, 1), Qt.EditRole)) == 1.0 / 3.0
    assert m.data(m.index(0, 0), Qt.EditRole) == ""
    assert m.data(m.index(0, 0), Qt.DecorationRole) is None
    assert m.data(m.index(9, 9)) is None
    assert m.headerData(0, Qt.Vertical) == "1"
    assert m.headerData(99, Qt.Horizontal) is None
    assert m.headerData(0, Qt.Horizontal, Qt.EditRole) is None


def test_column_count_is_blank_before_anything_is_loaded(screen):
    screen.set_column_filter("area")
    assert screen._col_count_label.text() == ""
    assert screen.visible_columns() == []


def test_the_model_survives_an_index_that_outlived_its_page(qtbot):
    """A delegate can hold an index from before a reset; it must not raise."""
    m = PreviewModel()
    m.set_page(["a", "b"], [(1, 2)], keys=[(1,)])
    stale = m.index(0, 1)
    m.set_editable(True)
    m.set_commit_hook(lambda *_a: True)
    m.set_page(["a"], [(1,)], keys=[(1,)])      # one column now
    assert stale.isValid()
    assert m.data(stale) is None
    assert m.setData(stale, "x") is False


def test_preview_model_append_and_keys(qtbot):
    m = PreviewModel()
    m.set_page(["a"], [(1,)], keys=[(1,)])
    assert m.append_rows([], []) == 0
    assert m.append_rows([(2,), (3,)], [(2,), (3,)]) == 2
    assert m.rowCount() == 3
    assert m.row_key(2) == (3,)
    assert m.row_key(99) is None
    assert m.value(0, "a") == 1
    assert m.value(99, "a") is None
    assert m.value(0, "nope") is None
    assert m.set_value(99, "a", 5) is False
    assert m.set_value(0, "nope", 5) is False
    # Rows appended without keys are not addressable.
    m.append_rows([(4,)])
    assert m.row_key(3) is None
    assert m.rowCount(m.index(0, 0)) == 0
    assert m.columnCount(m.index(0, 0)) == 0


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


def test_a_filter_is_applied_in_sql_not_to_the_loaded_rows(screen, measdb):
    """Partial data cannot make a filter wrong — SQLite does the filtering."""
    screen.set_database(measdb.path)
    assert screen.set_filter("pathogen_count", "=", "0") is True
    pc = measdb.index("pathogen_count")
    assert screen.loaded_rows() == 50 == screen.row_count()
    assert all(r[pc] == 0 for r in screen.preview_rows())


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


def test_export_is_not_limited_to_the_loaded_chunk(screen, measdb, tmp_path):
    """Export walks the whole filtered result, not just the 100 rows on screen."""
    screen.set_database(measdb.path)
    out = tmp_path / "all.csv"
    assert screen.export_csv(str(out)) is True
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == N_ROWS + 1


def test_export_creates_the_destination_folder(screen, measdb, tmp_path):
    screen.set_database(measdb.path)
    out = tmp_path / "reports" / "2026" / "cells.csv"
    assert screen.export_csv(str(out)) is True
    assert out.exists()


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


def test_a_write_shaped_raw_filter_is_refused_before_it_reaches_sqlite(
        screen, measdb):
    screen.set_database(measdb.path)
    assert screen.set_raw_filter("1=1; DROP TABLE cell") is False
    assert "Filter error" in screen.status_text()
    assert screen.set_raw_filter("DELETE FROM cell") is False
    assert "not allowed" in screen.status_text()
    assert "cell" in screen._db.tables(refresh=True)


def test_a_decimal_filter_value_is_bound_as_a_number(measdb):
    sql, params = build_where("cell_area", ">", "1.5e2", CELL_COLUMNS)
    assert params == (150.0,)
    db = ReadOnlyDb(measdb.path)
    assert db.count("cell", sql, params) == sum(
        1 for r in measdb.rows if r[CELL_COLUMNS.index("cell_area")] > 150.0)


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


def test_a_valueless_operator_needs_no_value(screen, measdb):
    screen.set_database(measdb.path)
    assert screen.set_filter("well", "is not null", "") is True
    assert screen.where_clause() == '"well" IS NOT NULL'
    assert screen.row_count() == N_ROWS
    assert not screen._filter_value.isEnabled()


# ---------------------------------------------------------------------------
# Read-only by default
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
    screen.fetch_more()
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


def test_edit_mode_is_off_by_default(screen, measdb):
    screen.set_database(measdb.path)
    assert screen.editing_allowed_by_preference() is False
    assert screen.edit_mode_enabled() is False
    assert screen._model.is_editable() is False
    from PySide6.QtWidgets import QAbstractItemView
    assert screen._view.editTriggers() == QAbstractItemView.NoEditTriggers
    # The UI says *why*, rather than silently swallowing a double-click.
    assert "Preferences" in screen._edit_note.text()
    assert not screen._edit_check.isEnabled()


def test_a_write_attempt_is_refused_and_the_file_is_unchanged(screen, measdb):
    screen.set_database(measdb.path)
    before = measdb.digest()
    assert screen.edit_cell(0, "cell_area", "999") is False
    assert "read-only" in screen.status_text().lower()
    assert screen.last_error
    assert measdb.digest() == before
    assert measdb.read_all() == [tuple(r) for r in measdb.rows]


def test_setdata_is_refused_while_read_only(screen, measdb):
    screen.set_database(measdb.path)
    before = measdb.digest()
    model = screen._model
    assert model.setData(model.index(0, 7), "999") is False
    assert model.flags(model.index(0, 7)) & Qt.ItemIsEditable == Qt.NoItemFlags
    assert measdb.digest() == before


# ---------------------------------------------------------------------------
# Edit mode — the preference, the confirmation, the guards
# ---------------------------------------------------------------------------

def test_the_edit_preference_round_trips_through_qsettings():
    from spacr.qt.preferences import (
        get_db_browser_editable, set_db_browser_editable,
    )
    assert get_db_browser_editable() is False       # off by default
    set_db_browser_editable(True)
    assert get_db_browser_editable() is True
    set_db_browser_editable(False)
    assert get_db_browser_editable() is False


def test_a_corrupt_edit_preference_falls_back_to_read_only():
    from PySide6.QtCore import QSettings
    from spacr.qt.preferences import get_db_browser_editable
    QSettings("spacr", "qt").setValue("prefs/db_browser_editable", "garbage")
    assert get_db_browser_editable() is False
    QSettings("spacr", "qt").setValue("prefs/db_browser_editable", "yes")
    assert get_db_browser_editable() is True


def test_the_preference_reader_copes_with_every_backend_spelling():
    """QSettings hands back bools, ints or strings depending on the backend."""
    from spacr.qt.preferences import _as_bool
    assert _as_bool(None, False) is False
    assert _as_bool(None, True) is True
    assert _as_bool(True, False) is True
    assert _as_bool(1, False) is True
    assert _as_bool(0, True) is False
    assert _as_bool(1.0, False) is True
    for yes in ("true", "TRUE", "1", "yes", "on"):
        assert _as_bool(yes, False) is True
    for no in ("false", "0", "no", "off", ""):
        assert _as_bool(no, True) is False
    assert _as_bool("wat", False) is False       # unreadable -> the default


def test_the_preferences_dialog_carries_the_edit_toggle(qtbot, qt_theme_applied):
    from PySide6.QtWidgets import QCheckBox
    from spacr.qt.preferences import (
        PreferencesDialog, get_db_browser_editable, set_db_browser_editable,
    )
    set_db_browser_editable(True)
    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    boxes = [c for c in dlg.findChildren(QCheckBox)
             if "editing" in c.text().lower()]
    assert boxes, "no 'allow editing' checkbox in the Preferences dialog"
    assert boxes[0].isChecked() is True
    boxes[0].setChecked(False)
    dlg.findChildren(type(dlg))          # keep the dialog referenced
    from PySide6.QtWidgets import QDialogButtonBox
    buttons = dlg.findChild(QDialogButtonBox)
    buttons.accepted.emit()
    assert get_db_browser_editable() is False


def test_edit_mode_cannot_be_armed_while_the_preference_is_off(screen, measdb):
    screen.confirm_edit_mode = lambda _m: pytest.fail(
        "the confirmation must not even be asked while the preference is off")
    screen.set_database(measdb.path)
    assert screen.enable_edit_mode() is False
    assert screen.edit_mode_enabled() is False
    assert "Preferences" in screen.status_text()


def test_ticking_the_checkbox_alone_does_not_enable_edit_mode(
        screen, measdb, allow_editing):
    """A stray click must not arm a read-write connection."""
    asked = []
    screen.confirm_edit_mode = lambda msg: asked.append(msg) or False
    screen.set_database(measdb.path)
    screen._edit_check.setChecked(True)
    assert asked, "the checkbox did not even ask"
    assert screen.edit_mode_enabled() is False
    assert screen._edit_check.isChecked() is False, \
        "the checkbox stayed ticked while edit mode is off — it lies"
    assert screen._model.is_editable() is False
    assert "not enabled" in screen.status_text()


def test_confirming_arms_edit_mode_and_unticking_disarms_it(
        screen, measdb, allow_editing, qtbot):
    screen.confirm_edit_mode = lambda _m: True
    screen.set_database(measdb.path)
    with qtbot.waitSignal(screen.edit_mode_changed, timeout=2000) as blocker:
        screen._edit_check.setChecked(True)
    assert blocker.args[0] is True
    assert screen.edit_mode_enabled() is True
    assert screen._model.is_editable() is True
    assert "EDIT MODE" in screen._edit_note.text()

    with qtbot.waitSignal(screen.edit_mode_changed, timeout=2000) as blocker:
        screen._edit_check.setChecked(False)
    assert blocker.args[0] is False
    assert screen.edit_mode_enabled() is False
    assert screen._model.is_editable() is False
    assert "read-only" in screen.status_text()


def test_disable_edit_mode_is_idempotent(screen, measdb):
    screen.set_database(measdb.path)
    screen.disable_edit_mode()
    screen.disable_edit_mode()
    assert screen.edit_mode_enabled() is False


def test_edit_mode_needs_a_database(screen):
    screen.confirm_edit_mode = lambda _m: True
    assert screen.enable_edit_mode() is False
    assert "Open a database" in screen.status_text()


def test_a_raising_confirmation_leaves_the_screen_read_only(
        screen, measdb, allow_editing):
    def _explode(_msg):
        raise RuntimeError("no window server")
    screen.confirm_edit_mode = _explode
    screen.set_database(measdb.path)
    assert screen.enable_edit_mode() is False
    assert screen.edit_mode_enabled() is False
    assert "no window server" in screen.status_text()


def test_the_default_confirmation_is_a_dialog_that_must_be_accepted(
        screen, measdb, allow_editing, monkeypatch):
    """The one deliberate dialog. It shows the exact statement first."""
    from PySide6.QtWidgets import QMessageBox
    screen.set_database(measdb.path)
    shown = []

    def _cancel(self):
        shown.append(self.informativeText())
        return QMessageBox.Cancel

    monkeypatch.setattr(QMessageBox, "exec", _cancel, raising=False)
    assert screen.enable_edit_mode() is False
    assert shown and 'UPDATE "cell"' in shown[0]
    assert 'WHERE "_rowid_" = ?' in shown[0]
    assert "no undo" in shown[0]

    monkeypatch.setattr(QMessageBox, "exec",
                        lambda self: QMessageBox.Yes, raising=False)
    assert screen.enable_edit_mode() is True, screen.status_text()
    assert screen.edit_mode_enabled() is True


def test_the_confirmation_names_the_key_of_a_without_rowid_table(
        screen, measdb, allow_editing):
    measdb.exec("CREATE TABLE kv (plate TEXT, well TEXT, value REAL, "
                "PRIMARY KEY (plate, well)) WITHOUT ROWID")
    seen = []
    screen.confirm_edit_mode = lambda msg: bool(seen.append(msg)) or True
    screen.set_database(measdb.path)
    screen.select_table("kv")
    assert screen.enable_edit_mode() is True, screen.status_text()
    assert 'WHERE "plate" = ? AND "well" = ?' in seen[0]


def test_arming_edit_mode_on_a_database_with_no_tables_is_harmless(
        screen, tmp_path, allow_editing):
    """No table means no key to name in the confirmation, and nothing to edit."""
    path = tmp_path / "blank.db"
    sqlite3.connect(path).close()
    seen = []
    screen.confirm_edit_mode = lambda msg: bool(seen.append(msg)) or True
    assert screen.set_database(str(path)) is True
    assert screen.enable_edit_mode() is True, screen.status_text()
    assert 'UPDATE "<table>"' in seen[0]
    assert 'WHERE "rowid" = ?' in seen[0]   # illustrative prose, not real SQL
    # ...and with no table the cells are not editable either.
    assert screen._model.is_editable() is False
    assert screen.edit_cell(0, "anything", "1") is False


def test_edit_mode_refuses_a_database_the_user_did_not_choose(
        screen, measdb, allow_editing):
    """Edit mode never opens a file spaCR picked on the user's behalf."""
    screen.confirm_edit_mode = lambda _m: True
    assert screen.set_database(measdb.path, explicit=False) is True
    assert screen.enable_edit_mode() is False
    assert "chosen by you" in screen.status_text()
    assert "opened for you" in screen._edit_note.text()
    # Choosing it explicitly is all it takes.
    screen.set_database(measdb.path)
    assert screen.enable_edit_mode() is True, screen.status_text()


def test_loading_a_different_database_resets_edit_mode(editable, other_db,
                                                        measdb):
    assert editable.edit_mode_enabled() is True
    editable.set_database(other_db)
    assert editable.edit_mode_enabled() is False
    assert editable._edit_check.isChecked() is False
    assert editable._model.is_editable() is False
    # ...and even re-opening the same file starts read-only again.
    editable.set_database(measdb.path)
    assert editable.edit_mode_enabled() is False


def test_edit_mode_is_armed_for_one_file_only(editable, other_db):
    """Belt and braces: the armed path and the open path must agree."""
    editable._db = ReadOnlyDb(other_db)      # simulate a desynchronised open
    assert editable.edit_cell(0, "cell_area", "1") is False
    assert "different database" in editable.status_text()
    assert editable.edit_mode_enabled() is False


# ---------------------------------------------------------------------------
# Edit mode — the write itself
# ---------------------------------------------------------------------------

def test_an_edit_writes_exactly_one_row(editable, measdb):
    assert editable.edit_cell(0, "cell_area", "9999.5") is True
    assert editable.last_error == ""
    after = measdb.read_all()
    assert len(after) == N_ROWS
    area = measdb.index("cell_area")
    for i, (got, want) in enumerate(zip(after, measdb.rows)):
        if i == 0:
            assert got[area] == 9999.5
            assert got[:area] == tuple(want[:area])
            assert got[area + 1:] == tuple(want[area + 1:])
        else:
            assert got == tuple(want), f"row {i} was collateral damage"
    # The view shows the new value without a reload.
    assert editable.preview_rows()[0][area] == 9999.5


def test_an_edit_shows_the_exact_sql_before_it_runs(editable, measdb,
                                                     monkeypatch):
    """The statement is on screen even when the write itself then fails."""
    def _explode(*_a, **_k):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(WritableDb, "update_cell", _explode)

    assert editable.edit_cell(2, "pathogen_count", "3") is False
    assert editable.pending_edit_sql() == (
        'UPDATE "cell" SET "pathogen_count" = ? WHERE "_rowid_" = ?')
    assert 'UPDATE "cell"' in editable.sql_text()
    assert "[3, 3]" in editable.sql_text()     # value, then the rowid
    assert "database is locked" in editable.status_text()
    assert measdb.read_all() == [tuple(r) for r in measdb.rows]


def test_an_uncoercible_value_is_rejected_not_stored_as_text(editable, measdb):
    before = measdb.digest()
    assert editable.edit_cell(0, "pathogen_count", "not a number") is False
    assert "not a whole number" in editable.status_text()
    assert editable.last_error
    assert measdb.digest() == before
    kind, value = measdb.typeof("pathogen_count", rowid=1)
    assert kind == "integer", "SQLite stored text in an INTEGER column"
    assert value == measdb.rows[0][measdb.index("pathogen_count")]
    # A REAL column is just as strict.
    assert editable.edit_cell(0, "cell_area", "1,5") is False
    assert "not a number" in editable.status_text()
    assert measdb.digest() == before


def test_an_empty_edit_writes_null(editable, measdb):
    assert editable.edit_cell(1, "well", "") is True
    kind, value = measdb.typeof("well", rowid=2)
    assert kind == "null" and value is None


def test_editing_a_cell_to_its_current_value_writes_nothing(editable, measdb):
    before = measdb.digest()
    current = measdb.rows[0][measdb.index("well")]
    assert editable.edit_cell(0, "well", current) is True
    assert measdb.digest() == before
    assert "already that value" in editable.status_text()


def test_editing_an_unknown_column_is_refused(editable, measdb):
    assert editable.edit_cell(0, "not_a_column", "1") is False
    assert "not a column" in editable.status_text()


def test_a_table_with_no_unique_row_address_refuses_the_edit(
        screen, measdb, allow_editing):
    measdb.exec("CREATE VIEW cell_view AS SELECT * FROM cell")
    screen.confirm_edit_mode = lambda _m: True
    screen.set_database(measdb.path)
    assert screen.select_table("cell_view") is True
    assert screen.enable_edit_mode() is True, screen.status_text()

    before = measdb.digest()
    assert screen.edit_cell(0, "cell_area", "1") is False
    msg = screen.status_text()
    assert "no rowid and no primary key" in msg
    assert "the row you clicked" in msg
    assert measdb.digest() == before
    # ...and the UI does not pretend the cells are editable either.
    assert screen._model.is_editable() is False
    assert "read-only" in screen._edit_note.text()


def test_a_without_rowid_table_is_edited_through_its_primary_key(
        screen, measdb, allow_editing):
    measdb.exec("CREATE TABLE kv (plate TEXT, well TEXT, value REAL, "
                "PRIMARY KEY (plate, well)) WITHOUT ROWID")
    measdb.exec("INSERT INTO kv VALUES ('plate1', 'A01', 1.0), "
                "('plate1', 'A02', 2.0)")
    screen.confirm_edit_mode = lambda _m: True
    screen.set_database(measdb.path)
    screen.select_table("kv")
    assert screen.enable_edit_mode() is True, screen.status_text()
    assert screen._db.row_key("kv") == ("pk", ["plate", "well"])

    assert screen.edit_cell(1, "value", "42.5") is True
    assert screen.pending_edit_sql() == (
        'UPDATE "kv" SET "value" = ? WHERE "plate" = ? AND "well" = ?')
    con = sqlite3.connect(measdb.path)
    try:
        assert con.execute("SELECT well, value FROM kv ORDER BY well"
                            ).fetchall() == [("A01", 1.0), ("A02", 42.5)]
    finally:
        con.close()


def test_setdata_routes_a_committed_editor_through_edit_cell(editable, measdb):
    model = editable._model
    col = editable.visible_columns().index("well")
    idx = model.index(0, col)
    assert model.flags(idx) & Qt.ItemIsEditable
    assert model.setData(idx, "ZZ99") is True
    assert measdb.read_all()[0][measdb.index("well")] == "ZZ99"
    assert model.data(idx) == "ZZ99"
    # Other roles are not writes.
    assert model.setData(idx, "nope", Qt.DecorationRole) is False
    assert model.setData(model.index(0, 999), "nope") is False


def test_setdata_without_a_commit_hook_returns_false(qtbot):
    m = PreviewModel()
    m.set_page(["a"], [(1,)], keys=[(1,)])
    m.set_editable(True)
    assert m.setData(m.index(0, 0), "2") is False


def test_edit_mode_without_a_table_is_refused(screen, measdb, allow_editing):
    screen.confirm_edit_mode = lambda _m: True
    screen.set_database(measdb.path)
    screen.enable_edit_mode()
    screen._table = ""
    assert screen.edit_cell(0, "cell_area", "1") is False
    assert "pick a table" in screen.status_text()


def test_an_edit_of_a_row_that_is_gone_is_refused(editable, measdb):
    """The probe fires before the UPDATE, so a vanished row writes nothing."""
    measdb.exec("DELETE FROM cell WHERE rowid = 1")
    before = measdb.digest()
    assert editable.edit_cell(0, "cell_area", "5") is False
    assert "matches 0 rows" in editable.status_text()
    assert measdb.digest() == before


def test_a_row_without_a_key_is_refused(editable):
    """A row fetched without a key can never be addressed."""
    editable._model.append_rows([tuple(range(len(CELL_COLUMNS)))])
    last = editable._model.rowCount() - 1
    assert editable._model.row_key(last) is None
    assert editable.edit_cell(last, "cell_area", "1") is False
    assert "no rowid and no primary key" in editable.status_text()


# ---------------------------------------------------------------------------
# WritableDb + the pure edit helpers
# ---------------------------------------------------------------------------

def test_writable_db_refuses_an_address_matching_many_rows(measdb):
    """The guard that stands between a typo and mass corruption."""
    db = WritableDb(measdb.path)
    before = measdb.digest()
    with pytest.raises(EditRefused) as exc:
        db.update_cell("cell", "cell_area", 0.0, ["plate"], ["plate1"])
    assert "matches 250 rows" in str(exc.value)
    assert measdb.digest() == before


def test_writable_db_refuses_views_and_unknown_names(measdb):
    measdb.exec("CREATE VIEW cell_view AS SELECT * FROM cell")
    db = WritableDb(measdb.path)
    with pytest.raises(EditRefused):
        db.update_cell("cell_view", "cell_area", 1.0, ["rowid"], [1])
    with pytest.raises(EditRefused):
        db.update_cell("cell", "not_a_column", 1.0, ["rowid"], [1])
    with pytest.raises(EditRefused):
        db.update_cell("cell", "cell_area", 1.0, ["not_a_key"], [1])
    with pytest.raises(EditRefused):
        db.update_cell("cell", "cell_area", 1.0, [], [])
    with pytest.raises(EditRefused):
        db.update_cell("cell", "cell_area", 1.0, ["rowid"], [1, 2])
    assert measdb.read_all() == [tuple(r) for r in measdb.rows]


def test_writable_db_rolls_back_when_the_update_misses(measdb, monkeypatch):
    """rowcount != 1 must roll back, not commit and hope."""
    db = WritableDb(measdb.path)
    real_connect = db.connect

    class _Liar:
        """A connection whose UPDATE reports it touched nothing."""

        def __init__(self, con):
            self._con = con

        def __getattr__(self, name):
            return getattr(self._con, name)

        def execute(self, sql, *a):
            cur = self._con.execute(sql, *a)
            if sql.startswith("UPDATE"):
                class _Zero:
                    rowcount = 0
                return _Zero()
            return cur

    monkeypatch.setattr(db, "connect", lambda: _Liar(real_connect()))
    with pytest.raises(EditRefused) as exc:
        db.update_cell("cell", "cell_area", 1.0, ["rowid"], [1])
    assert "rolled back" in str(exc.value)
    assert measdb.read_all() == [tuple(r) for r in measdb.rows]


def test_writable_db_lets_sqlite_errors_through_and_rolls_back(measdb):
    measdb.exec("CREATE TABLE strict_t (a INTEGER NOT NULL, b TEXT)")
    measdb.exec("INSERT INTO strict_t VALUES (1, 'keep')")
    db = WritableDb(measdb.path)
    with pytest.raises(sqlite3.IntegrityError):
        # An UPDATE that violates a constraint SQLite itself enforces.
        db.update_cell("strict_t", "a", None, ["rowid"], [1])
    con = sqlite3.connect(measdb.path)
    try:
        assert con.execute("SELECT a, b FROM strict_t").fetchall() == [
            (1, "keep")], "the failed UPDATE was not rolled back"
    finally:
        con.close()
    assert measdb.read_all() == [tuple(r) for r in measdb.rows]


def test_build_update_is_the_only_statement_shape():
    assert build_update("cell", "cell_area", ["rowid"]) == (
        'UPDATE "cell" SET "cell_area" = ? WHERE "rowid" = ?')
    assert build_update("t", "c", ["a", "b"]) == (
        'UPDATE "t" SET "c" = ? WHERE "a" = ? AND "b" = ?')
    with pytest.raises(EditRefused):
        build_update("cell", "cell_area", [])
    # Nothing the user types can escape the quoting.
    assert build_update('a"b', 'c"d', ["rowid"]).startswith(
        'UPDATE "a""b" SET "c""d" = ?')


def test_column_affinity_follows_the_sqlite_rules():
    assert column_affinity("INTEGER") == "INTEGER"
    assert column_affinity("BIGINT") == "INTEGER"
    assert column_affinity("VARCHAR(255)") == "TEXT"
    assert column_affinity("CLOB") == "TEXT"
    assert column_affinity("BLOB") == "BLOB"
    assert column_affinity("") == "BLOB"
    assert column_affinity(None) == "BLOB"
    assert column_affinity("DOUBLE PRECISION") == "REAL"
    assert column_affinity("FLOAT") == "REAL"
    assert column_affinity("REAL") == "REAL"
    assert column_affinity("DECIMAL(10,5)") == "NUMERIC"
    assert column_affinity("BOOLEAN") == "NUMERIC"


def test_coerce_for_column_types_the_value_or_refuses():
    assert coerce_for_column("42", "INTEGER") == 42
    assert coerce_for_column(" -7 ", "INTEGER") == -7
    assert coerce_for_column("1.5", "REAL") == 1.5
    assert coerce_for_column("2", "REAL") == 2.0
    assert coerce_for_column("1e3", "REAL") == 1000.0
    assert coerce_for_column("42", "TEXT") == "42"
    assert coerce_for_column(42, "TEXT") == "42"
    assert coerce_for_column("", "INTEGER") is None
    assert coerce_for_column("   ", "TEXT") is None
    assert coerce_for_column(None, "TEXT") is None
    # NUMERIC / untyped mirror SQLite itself.
    assert coerce_for_column("3", "NUMERIC") == 3
    assert coerce_for_column("3.5", "NUMERIC") == 3.5
    assert coerce_for_column("later", "NUMERIC") == "later"
    assert coerce_for_column("7", "") == 7
    assert coerce_for_column("7.5", "") == 7.5
    assert coerce_for_column("seven", "") == "seven"

    for text, decl in (("abc", "INTEGER"), ("1.5", "INTEGER"),
                       ("abc", "REAL"), ("1,5", "REAL")):
        with pytest.raises(ValueError):
            coerce_for_column(text, decl, "col")


def test_coerce_for_column_refuses_binary_columns():
    with pytest.raises(ValueError) as exc:
        coerce_for_column("hello", "BLOB", "png")
    assert "binary" in str(exc.value)


def test_row_key_reports_how_a_row_can_be_addressed(measdb):
    measdb.exec("CREATE VIEW cell_view AS SELECT * FROM cell")
    measdb.exec("CREATE TABLE kv (a TEXT, b TEXT, v REAL, "
                "PRIMARY KEY (a, b)) WITHOUT ROWID")
    db = ReadOnlyDb(measdb.path)
    # `_rowid_`, not `rowid`: SQLite identifiers are case-insensitive and
    # png_list DECLARES a rowID column, which makes the bare name resolve to
    # that column. row_key now asks for a spelling the table cannot shadow,
    # so an edit addresses one row instead of a whole plate row.
    assert db.row_key("cell") == ("rowid", ["_rowid_"])
    assert db.row_key("cell") == ("rowid", ["_rowid_"])   # cached
    assert db.row_key("kv") == ("pk", ["a", "b"])
    assert db.row_key("cell_view") == ("", [])


def test_column_types_are_read_from_the_schema(measdb):
    db = ReadOnlyDb(measdb.path)
    types = db.column_types("cell")
    assert types["cell_area"] == "REAL"
    assert types["well"] == "TEXT"
    assert types["pathogen_count"] == "INTEGER"
    assert db.column_types("cell") == types               # cached


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


def test_a_metacharacter_value_cannot_escape_an_edit(editable, measdb):
    assert editable.edit_cell(0, "well", EVIL) is True
    assert measdb.read_all()[0][measdb.index("well")] == EVIL
    assert "png_list" in editable._db.tables(refresh=True)
    assert len(measdb.read_all()) == N_ROWS


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


def test_a_database_with_no_tables_opens_and_says_so(screen, tmp_path):
    path = tmp_path / "blank.db"
    sqlite3.connect(path).close()
    assert screen.set_database(str(path)) is True
    assert "no tables" in screen.status_text()
    assert screen.tables() == []
    assert screen.current_table() == ""


def test_a_listing_failure_reports_inline(screen, measdb, monkeypatch):
    def _explode(self, refresh=False):
        raise sqlite3.OperationalError("disk I/O error")
    monkeypatch.setattr(ReadOnlyDb, "tables", _explode)
    assert screen.set_database(measdb.path) is False
    assert "disk I/O error" in screen.status_text()


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


def test_select_table_without_a_database_reports_inline(screen):
    assert screen.select_table("cell") is False
    assert "No database open" in screen.status_text()


def test_a_second_export_is_refused_while_one_is_running(screen, measdb,
                                                          tmp_path):
    screen.set_database(measdb.path)
    screen._export_busy = True
    assert screen.export_csv(str(tmp_path / "x.csv")) is False
    assert "already running" in screen.status_text()
    screen._export_busy = False


def test_a_failing_query_reports_inline(screen, measdb, monkeypatch):
    screen.set_database(measdb.path)

    def _explode(self, *a, **k):
        raise sqlite3.OperationalError("disk I/O error")
    monkeypatch.setattr(ReadOnlyDb, "chunk", _explode)
    screen.refresh()
    assert "Query failed" in screen.status_text()
    assert "disk I/O error" in screen.status_text()


def test_a_failing_completion_handler_reports_inline(screen, measdb,
                                                      monkeypatch):
    screen.set_database(measdb.path)

    def _explode(self, result):
        raise RuntimeError("bad result")
    monkeypatch.setattr(DbBrowserScreen, "_apply_chunk", _explode)
    screen.refresh()
    assert "bad result" in screen.status_text()


# ---------------------------------------------------------------------------
# Off-thread execution
# ---------------------------------------------------------------------------

def test_query_runs_off_the_gui_thread(qtbot, qt_theme_applied, measdb,
                                        monkeypatch):
    """The chunk query must not execute on the GUI thread."""
    gui_thread = threading.get_ident()
    seen = []
    real_chunk = ReadOnlyDb.chunk

    def _spy(self, *a, **k):
        seen.append(threading.get_ident())
        return real_chunk(self, *a, **k)

    monkeypatch.setattr(ReadOnlyDb, "chunk", _spy)

    w = DbBrowserScreen(threaded=True)
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.job_finished, timeout=10000) as blocker:
        w.set_database(measdb.path)
    assert blocker.args[0] is True
    assert seen, "the chunk query never ran"
    assert all(t != gui_thread for t in seen), \
        "the query ran on the GUI thread — the window would freeze"
    qtbot.waitUntil(lambda: not w.is_busy(), timeout=10000)
    qtbot.waitUntil(lambda: w.active_jobs() == 0, timeout=10000)
    assert w.row_count() == N_ROWS
    assert w.loaded_rows() == DEFAULT_PAGE_SIZE
    w.close()


def test_completion_handlers_run_on_the_gui_thread(qtbot, qt_theme_applied,
                                                    measdb, monkeypatch):
    """PipelineWorker.finished fires in the worker thread. Everything that
    touches a model or a widget has to be dragged back before it runs."""
    from PySide6.QtCore import QThread
    seen = []
    real = DbBrowserScreen._apply_chunk

    def _spy(self, result):
        seen.append((QThread.currentThread(), self.thread()))
        return real(self, result)

    monkeypatch.setattr(DbBrowserScreen, "_apply_chunk", _spy)

    w = DbBrowserScreen(threaded=True)
    qtbot.addWidget(w)
    w.set_database(measdb.path)
    qtbot.waitUntil(lambda: w.loaded_rows() > 0, timeout=10000)
    qtbot.waitUntil(lambda: not w.is_busy(), timeout=10000)
    qtbot.waitUntil(lambda: w.active_jobs() == 0, timeout=10000)
    assert seen, "_apply_chunk never ran"
    for current, own in seen:
        assert current is own, \
            "a completion handler ran off the GUI thread"
    w.close()


def test_each_worker_opens_and_closes_its_own_connection(
        qtbot, qt_theme_applied, measdb, monkeypatch):
    """sqlite3 connections are not shareable across threads."""
    gui_thread = threading.get_ident()
    records = []
    real_connect = ReadOnlyDb.connect

    class _Tracked:
        def __init__(self, con, record):
            self._con = con
            self._record = record

        def __getattr__(self, name):
            return getattr(self._con, name)

        def close(self):
            self._record["closed"] = True
            self._con.close()

    def _spy(self):
        record = {"thread": threading.get_ident(), "closed": False}
        records.append(record)
        return _Tracked(real_connect(self), record)

    monkeypatch.setattr(ReadOnlyDb, "connect", _spy)

    w = DbBrowserScreen(threaded=True)
    qtbot.addWidget(w)
    w.set_database(measdb.path)
    qtbot.waitUntil(lambda: not w.is_busy(), timeout=10000)
    qtbot.waitUntil(lambda: w.active_jobs() == 0, timeout=10000)

    off_gui = [r for r in records if r["thread"] != gui_thread]
    assert off_gui, "no connection was opened on a worker thread"
    assert all(r["closed"] for r in records), \
        "a sqlite connection was left open"
    w.close()


def test_threaded_export_reports_inline_on_failure(qtbot, qt_theme_applied,
                                                    measdb, tmp_path):
    w = DbBrowserScreen(threaded=True)
    qtbot.addWidget(w)
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
    qtbot.waitUntil(lambda: w.active_jobs() == 0, timeout=10000)
    w.close()


def test_threaded_query_retires_its_thread(qtbot, qt_theme_applied, measdb):
    w = DbBrowserScreen(threaded=True)
    qtbot.addWidget(w)
    w.set_database(measdb.path)
    qtbot.waitUntil(lambda: not w.is_busy(), timeout=10000)
    qtbot.waitUntil(lambda: w.active_jobs() == 0, timeout=10000)
    assert w._thread is None and w._worker is None
    assert w._pending == {}
    w.close()   # must not abort on a live QThread


def test_thread_startup_has_no_signal_disconnect_warning(
        qtbot, qt_theme_applied, measdb):
    """The shared worker no longer self-deletes, so DB Browser must not try
    to disconnect a nonexistent deleteLater slot for every queued query."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        w = DbBrowserScreen(threaded=True)
        qtbot.addWidget(w)
        w.set_database(measdb.path)
        qtbot.waitUntil(lambda: not w.is_busy(), timeout=10000)
        qtbot.waitUntil(lambda: w.active_jobs() == 0, timeout=10000)
        w.close()


def test_overlapping_threaded_jobs_do_not_drop_a_live_thread(
        qtbot, qt_theme_applied, measdb, tmp_path):
    """`worker.finished` frees the UI before `thread.finished` retires the
    thread, so job N+1 starts while job N is still winding down. Job N's
    retirement must not release job N+1's QThread — that used to abort the
    interpreter with "QThread: Destroyed while thread is still running"."""
    w = DbBrowserScreen(threaded=True)
    qtbot.addWidget(w)
    w.set_database(measdb.path)
    qtbot.waitUntil(lambda: not w.is_busy(), timeout=10000)
    for i in range(4):
        with qtbot.waitSignal(w.job_finished, timeout=10000) as blocker:
            w.export_csv(str(tmp_path / f"chain_{i}.csv"))
        assert blocker.args[0] is True
        assert w._thread is not None or w.active_jobs() == 0
    qtbot.waitUntil(lambda: w.active_jobs() == 0, timeout=10000)
    for i in range(4):
        assert (tmp_path / f"chain_{i}.csv").exists()
    w.close()


def test_a_settled_job_is_only_finished_once(screen, measdb):
    """A duplicate delivery must not pop another job's completion."""
    screen.set_database(measdb.path)
    screen._on_job_settled(9999, True)      # never-started job id
    assert screen.last_error == ""
    assert screen.row_count() == N_ROWS


def test_a_threaded_completion_handler_that_raises_reports_inline(screen):
    """The GUI-thread half of the job plumbing, driven directly."""
    def _explode(_result):
        raise RuntimeError("handler exploded")
    screen._pending[42] = ({"result": None}, _explode, "chunk")
    screen._acquire("chunk")
    screen._on_job_settled(42, True)
    assert "handler exploded" in screen.status_text()
    assert screen._chunk_jobs == 0


def test_the_worker_payload_helper_carries_the_result_back():
    """PipelineWorker's finished(bool) cannot carry a result; the dict does."""
    from spacr.qt.screens.db_browser import _capture_result
    box = {}
    _capture_result(lambda: {"rows": [1, 2]}, box)
    assert box["result"] == {"rows": [1, 2]}


def test_closing_waits_for_a_thread_that_is_still_running(
        qtbot, qt_theme_applied, measdb, monkeypatch):
    """A QThread destroyed while running takes the process down with it."""
    _slow_chunk(monkeypatch, "cell", delay=0.2)
    w = DbBrowserScreen(threaded=True)
    qtbot.addWidget(w)
    w.set_database(measdb.path)
    assert w.active_jobs() >= 1
    w.close()                       # must quit + wait, not abandon
    qtbot.waitUntil(lambda: not w.is_busy(), timeout=20000)
    qtbot.waitUntil(lambda: w.active_jobs() == 0, timeout=20000)


def test_closing_survives_a_thread_wrapper_qt_already_deleted(
        qtbot, qt_theme_applied):
    """PySide6 raises RuntimeError from a dead wrapper; close must not."""
    class _Dead:
        def isRunning(self):
            raise RuntimeError("Internal C++ object already deleted.")

    w = DbBrowserScreen(threaded=True)
    qtbot.addWidget(w)
    w._jobs[999] = (_Dead(), None)
    w.close()
    assert w.active_jobs() == 1      # nothing retired it; nothing crashed
    w._jobs.clear()
