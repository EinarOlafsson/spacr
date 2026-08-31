"""Database Browser — the corners a normal browse never turns.

Everything the main suite (``tests/qt/test_db_browser.py``) drives goes
through a plain rowid table, one worker thread at a time and a filter row
the user filled in from the combo boxes. The paths below are what happens
when one of those assumptions is false:

* a table with **no row id** — a ``WITHOUT ROWID`` table or a view. spaCR's
  own ``png_list``/``cell`` tables have one, but a user's database is their
  database, and the SQL builders must not emit ``ORDER BY <nothing>``;
* a **keyless view sorted by a column** — no key means no tiebreak column,
  and the ORDER BY has to survive its absence;
* a cell edited in a column the **column search is hiding**;
* a table that appeared in the file **after** the sidebar was listed;
* selection and filter calls made **programmatically** with arguments no
  widget could have produced (out-of-range rows, an operator that is not
  in ``OPERATORS``);
* the thread bookkeeping when **more than one** job is outstanding — the
  sweep must retire only the stopped ones, and closing must wait only on
  the running ones. Getting either backwards is a hard process crash
  (a QThread collected while running) or a screen that is busy forever.
"""
from __future__ import annotations

import sqlite3

import pytest

from spacr.qt.screens.db_browser import (
    DbBrowserScreen,
    PreviewModel,
    ReadOnlyDb,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_ROWS = 30


@pytest.fixture
def db_path(tmp_path):
    """A database holding all three row-identity shapes SQLite offers.

    ``cell`` has an implicit row id, ``barcodes`` is ``WITHOUT ROWID`` (its
    identity is the declared primary key) and ``area_view`` is a view (no
    identity at all). The browser has to page, sort and export every one.
    """
    path = tmp_path / "measurements" / "measurements.db"
    path.parent.mkdir(parents=True)
    con = sqlite3.connect(path)
    try:
        con.execute(
            "CREATE TABLE cell (plate TEXT, well TEXT, cell_area REAL, "
            "pathogen_count INTEGER)")
        con.executemany(
            "INSERT INTO cell VALUES (?, ?, ?, ?)",
            [("plate1", f"A{i + 1:02d}", 100.0 + i, i % 5)
             for i in range(N_ROWS)])
        con.execute(
            "CREATE TABLE barcodes (barcode TEXT PRIMARY KEY, tally INTEGER) "
            "WITHOUT ROWID")
        con.executemany(
            "INSERT INTO barcodes VALUES (?, ?)",
            [(f"bc{i:03d}", i * 3) for i in range(8)])
        # Deliberately duplicated pathogen_count values: a sort with no
        # tiebreak is only wrong when values repeat.
        con.execute(
            "CREATE VIEW area_view AS SELECT well, pathogen_count FROM cell")
        con.commit()
    finally:
        con.close()
    return str(path)


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    """A synchronous screen — queries run inline so assertions are exact."""
    w = DbBrowserScreen(threaded=False)
    qtbot.addWidget(w)
    return w


class _FakeThread:
    """Stand-in for a QThread whose running state the test decides.

    Real threads race; these paths are about *which* of several jobs the
    bookkeeping touches, and that has to be pinned exactly. The same
    injection idiom the main suite uses for its already-deleted-wrapper
    test.
    """

    def __init__(self, running: bool):
        self._running = running
        self.calls = []

    def isRunning(self):  # noqa: N802 - QThread's spelling
        return self._running

    def quit(self):
        self.calls.append("quit")

    def wait(self, msecs):
        self.calls.append(("wait", msecs))
        return True


# ---------------------------------------------------------------------------
# SQL builders: tables and views with no usable row id
# ---------------------------------------------------------------------------

def test_export_of_a_rowid_less_table_omits_the_order_by(db_path):
    """A CSV export must not be built around a key the table hasn't got.

    ``select_sql`` appends ``ORDER BY <key>`` so an exported CSV comes out
    in a stable, reproducible order. A ``WITHOUT ROWID`` table has no
    ``rowid`` to name — reaching for ``key_cols[0]`` regardless would raise
    IndexError before a single byte reached the file, and the user would be
    told their export failed with a Python traceback rather than getting
    their data.
    """
    db = ReadOnlyDb(db_path)

    assert db.row_key("barcodes") == ("pk", ["barcode"])
    keyless_sql = db.select_sql("barcodes", ["barcode", "tally"])
    assert "ORDER BY" not in keyless_sql
    assert keyless_sql == 'SELECT "barcode", "tally" FROM "barcodes"'

    # Same builder, same call shape, on a table that *does* have a row id:
    # this is what the missing clause above would otherwise have looked like.
    assert db.row_key("cell") == ("rowid", ["_rowid_"])
    assert db.select_sql("cell", ["well"]) == (
        'SELECT "well" FROM "cell" ORDER BY "_rowid_"')


def test_sorting_a_keyless_view_orders_by_the_column_alone(db_path):
    """A view can be sorted even though no column can break the tie.

    Chunked paging appends the key column to every ORDER BY as a tiebreak,
    because rows sharing a value are otherwise free to come back in a
    different order in two chunks of one scroll — showing a row twice and
    another not at all. A view has no key column to append. The sort must
    still be issued (an unsorted view after clicking the header reads as a
    dead control), just without the tiebreak term.
    """
    db = ReadOnlyDb(db_path)
    assert db.row_key("area_view") == ("", [])

    sql = db.chunk_sql("area_view", ["well", "pathogen_count"], [],
                       order_by=("pathogen_count", True))
    assert sql == (
        'SELECT "well", "pathogen_count" FROM "area_view" '
        'ORDER BY "pathogen_count" DESC LIMIT ? OFFSET ?')

    # And the same call with a key column present, to show the tiebreak
    # term this one is missing is real and not merely never emitted.
    keyed = db.chunk_sql("cell", ["well"], ["_rowid_"],
                         order_by=("pathogen_count", True))
    assert 'ORDER BY "pathogen_count" DESC, "_rowid_" ASC' in keyed

    cols, rows, keys = db.chunk("area_view", limit=5,
                                order_by=("pathogen_count", True))
    assert cols == ["well", "pathogen_count"]
    assert [r[1] for r in rows] == [4, 4, 4, 4, 4]
    assert keys == [None] * 5      # no identity -> no edit is addressable


# ---------------------------------------------------------------------------
# PreviewModel: writing back into a column the search is hiding
# ---------------------------------------------------------------------------

def test_editing_a_column_the_search_hides_still_updates_the_page():
    """An edit must land in the page even when the cell is off-screen.

    The column search is a pure view operation — it never re-queries — so
    the model keeps every fetched column and maps only some into the view.
    An edit committed while a search is narrowing the columns (perfectly
    possible: the search box can be typed into after the editor opened)
    still has to write into the row, or the next scroll would repaint the
    stale value and the user would believe their UPDATE was lost. What it
    must *not* do is emit ``dataChanged`` for a column index the view does
    not have, which addresses a cell outside the model's own extent.
    """
    model = PreviewModel()
    model.set_page(["plate", "cell_area", "note"],
                   [("plate1", 100.0, "before")], keys=[(1,)])
    model.set_column_filter("area")
    assert model.visible_columns() == ["cell_area"]

    repainted = []
    model.dataChanged.connect(
        lambda tl, br, roles=None: repainted.append((tl.row(), tl.column())))

    assert model.set_value(0, "note", "after") is True
    assert model.value(0, "note") == "after"
    assert model.rows() == [("plate1", 100.0, "after")]
    assert repainted == []          # hidden column -> no index to repaint

    # The visible column, same call: this is the repaint the line above is
    # asserting the absence of.
    assert model.set_value(0, "cell_area", 250.0) is True
    assert repainted == [(0, 0)]
    assert model.value(0, "cell_area") == 250.0


# ---------------------------------------------------------------------------
# A table that appeared after the sidebar was listed
# ---------------------------------------------------------------------------

def test_a_table_created_after_the_sidebar_was_listed_still_opens(
        screen, db_path):
    """Selecting a table the sidebar never listed must load it, not hang.

    spaCR writes new tables into ``measurements.db`` while the browser may
    already have it open, and ``apply_seed`` can be handed one of those
    names by another screen. The sidebar is a snapshot taken when the file
    was opened, so the loop that syncs the highlighted row can fail to find
    the name — and when it does, the preview must still switch. Falling out
    of that loop into anything other than ``refresh()`` would leave the user
    looking at the previous table while ``current_table()`` claimed the new
    one.
    """
    assert screen.set_database(db_path) is True
    sidebar = screen.tables()
    assert "cell" in sidebar and "late_arrival" not in sidebar

    con = sqlite3.connect(db_path)
    try:
        con.execute("CREATE TABLE late_arrival (tag TEXT, score REAL)")
        con.executemany("INSERT INTO late_arrival VALUES (?, ?)",
                        [("t1", 0.5), ("t2", 1.5)])
        con.commit()
    finally:
        con.close()
    screen._db.tables(refresh=True)      # schema catches up; sidebar does not

    assert screen.select_table("late_arrival") is True
    assert screen.current_table() == "late_arrival"
    assert screen.preview_columns() == ["tag", "score"]
    assert screen.preview_rows() == [("t1", 0.5), ("t2", 1.5)]
    assert screen.tables() == sidebar     # nothing to highlight, nothing added


# ---------------------------------------------------------------------------
# Programmatic selection and filtering with arguments no widget produces
# ---------------------------------------------------------------------------

def test_selecting_rows_that_are_all_out_of_range_clears_the_selection(
        screen, db_path):
    """A stale linked selection must not leave old rows looking selected.

    ``select_rows`` is what a linked view calls when the shared selection
    changes, and the rows it names were computed against whatever that
    other view has loaded. After switching to a shorter table every one of
    them can be past the end. Returning early *before* clearing would leave
    the previous highlight standing, so the table would claim to have rows
    selected that the shared selection no longer contains.
    """
    assert screen.set_database(db_path) is True
    assert screen.select_table("cell") is True

    assert screen.select_rows([2, 3]) == [2, 3]
    assert screen.selected_rows() == [2, 3]

    total = screen.row_count()
    assert total == N_ROWS
    assert screen.select_rows([total + 5, -1]) == []
    assert screen.selected_rows() == []


def test_an_unknown_filter_operator_never_reaches_the_sql(screen, db_path):
    """An operator spaCR does not know must not be spliced into a WHERE.

    ``set_filter`` is the programmatic seam (seeds, linked views, scripted
    demos), so its ``op`` argument is not constrained by the combo box the
    user sees.

    REWRITTEN 2026-08-31. This asserted that an unknown operator FALLS
    BACK to whichever operator happens to be selected, and returns True.
    The screen refuses it instead, says so, and returns False -- and the
    refusal is the better contract: a caller passing an operator spaCR
    does not know has a bug, and quietly filtering their data with "="
    would give them wrong numbers without a word. Refusing is a real
    answer, and it is the one the code gives.

    What the test was actually written to protect -- that the unknown
    string never reaches generated SQL -- is asserted either way, and
    still is.
    """
    assert screen.set_database(db_path) is True
    assert screen.select_table("cell") is True

    assert screen.set_filter("cell_area", "\u2248", "115") is False
    assert "\u2248" in screen.status_text()
    assert "Unknown operator" in screen.status_text()
    assert "\u2248" not in (screen.where_clause() or "")

    # The same call with an operator that *is* known, so the refusal above
    # is visibly a refusal and not the only thing set_filter can do.
    assert screen.set_filter("cell_area", ">=", "128") is True
    assert screen.where_clause() == '"cell_area" >= ?'
    assert len(screen.preview_rows()) == 2


def test_an_unknown_filter_column_is_refused_the_same_way(screen, db_path):
    """The column check above it, which had no test of its own."""
    assert screen.set_database(db_path) is True
    assert screen.select_table("cell") is True

    assert screen.set_filter("no_such_column", "=", "1") is False
    assert "Unknown column" in screen.status_text()
    assert "no_such_column" not in (screen.where_clause() or "")


# ---------------------------------------------------------------------------
# Thread bookkeeping with more than one job outstanding
# ---------------------------------------------------------------------------

def test_the_sweep_retires_stopped_jobs_and_leaves_the_running_one(
        qtbot, qt_theme_applied):
    """Dropping a running job's references crashes the whole process.

    ``_retire_finished_jobs`` runs on every ``thread.finished``, and with
    two jobs outstanding it sees both. It must release only the one whose
    event loop has actually exited: a QThread garbage-collected while it is
    still running takes the interpreter down with it, and that is exactly
    what a "clear the refs" sweep would do to the survivor. Retiring a job
    that is not the current one must also leave ``_thread``/``_worker``
    pointing at the job that *is*.
    """
    w = DbBrowserScreen(threaded=True)
    qtbot.addWidget(w)
    alive, stopped = _FakeThread(True), _FakeThread(False)
    alive_worker, stopped_worker = object(), object()
    w._jobs[11] = (alive, alive_worker)
    w._jobs[12] = (stopped, stopped_worker)
    w._thread, w._worker = alive, alive_worker
    try:
        w._retire_finished_jobs()
        assert list(w._jobs) == [11]
        assert w.active_jobs() == 1
        assert w._jobs[11] == (alive, alive_worker)
        assert w._thread is alive
        assert w._worker is alive_worker
    finally:
        w._jobs.clear()
        w._thread = w._worker = None


def test_closing_waits_only_on_the_threads_that_are_still_running(
        qtbot, qt_theme_applied):
    """Closing must not block for five seconds per finished thread.

    ``closeEvent`` quits and waits every job it still holds so a running
    QThread is never destroyed under Qt's feet. Threads that have already
    stopped are often still in ``_jobs`` — retirement is queued onto the GUI
    thread and may not have run — and calling ``wait(5000)`` on each of them
    would stall closing the screen for whole seconds with nothing happening.
    """
    w = DbBrowserScreen(threaded=True)
    qtbot.addWidget(w)
    running, finished = _FakeThread(True), _FakeThread(False)
    w._jobs[21] = (running, object())
    w._jobs[22] = (finished, object())
    try:
        w.close()
        assert running.calls == ["quit", ("wait", 5000)]
        assert finished.calls == []
    finally:
        w._jobs.clear()


def test_a_worker_traceback_with_nothing_in_it_still_reports_a_failure(
        screen, db_path):
    """A blank traceback must not swallow the fact that the query failed.

    ``_on_worker_error_text`` shows the last non-empty line of a worker
    traceback inline — never a dialog, which would hang a headless run.
    A worker that dies without producing any text (killed mid-write, or an
    exception whose formatting itself failed) leaves nothing to quote, and
    the loop finds no line at all. The status still has to say the query
    failed: silently leaving the previous "Opened ..." message up would tell
    the user their filter returned nothing when it in fact never ran.
    """
    assert screen.set_database(db_path) is True
    assert screen.select_table("cell") is True
    assert screen.status_text() == "cell: 30 rows · 4 columns · read-only"

    screen._on_worker_error_text("   \n  \n ")
    assert screen.status_text() == "Query failed: "
    assert screen.last_error == "Query failed: "

    # A real traceback through the same slot, so the empty string above is
    # visibly the degenerate case of a slot that normally quotes a line.
    screen._on_worker_error_text(
        "Traceback (most recent call last):\n"
        '  File "db_browser.py", line 1, in chunk\n'
        "sqlite3.OperationalError: no such column: zzz\n")
    assert screen.status_text() == (
        "Query failed: sqlite3.OperationalError: no such column: zzz")
