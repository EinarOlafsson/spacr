"""Opening a measurement table dispatches; it never opens the file inline.

THE FREEZE, 2026-09-04. `GraphBuilderScreen.load_path` threaded the read and
kept the listing inline, on the argument that one `sqlite_master` query costs
0.4 ms:

    GraphBuilderScreen.choose_table          (a plain clicked slot)
      -> load_path
        -> table_names(path)
          -> sqlite3.connect('file:<path>?mode=ro')

0.4 ms is what a disk that answers costs. One of the maintainer's paths was
under ``/nas_mnt``, an ``autofs`` mount whose share was asleep, and a single
stat on it had not returned after TWENTY SECONDS -- and `connect` must open
the file before it can read a byte of `sqlite_master`. Loading a
measurements.db from there, or dropping a project folder that resolves onto
one, froze the whole window. It left no traceback, because a stalled event
loop is not a crash; it was reported as "opening map barcodes crashes spacr",
plus hover flicker and glimpses of other screens.

The tests below pin both halves of the fix: the call returns while the
database is still parked, and every table still reaches the picker once the
worker has answered -- the listing moved, it did not go away.
"""
from __future__ import annotations

import sqlite3
import threading
import time

import pandas as pd
import pytest

pytest.importorskip("PySide6")

#: Longer than any human would call responsive, shorter than the twenty
#: seconds actually measured. A test that waited the real duration would be
#: a test nobody runs.
SLOW_S = 8.0


@pytest.fixture
def sleeping_database(monkeypatch):
    """Every ``sqlite3.connect`` parks, the way a sleeping mount does.

    Patched on the module, not on the screen: `table_names` and `read_table`
    both open the file, and the point is that NEITHER may be reached from the
    GUI thread. The worker is released from the test body rather than at
    teardown so the failure lands on a screen that is still alive -- a job
    delivering into a half-destroyed widget is a different bug, and not this
    test's business.
    """
    released = threading.Event()
    opened = threading.Event()

    def slow_connect(*args, **kwargs):
        opened.set()
        released.wait(SLOW_S)
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(sqlite3, "connect", slow_connect)
    try:
        yield opened, released
    finally:
        released.set()


def _screen(qtbot, **kwargs):
    from spacr.qt.screens.graph_builder import GraphBuilderScreen

    screen = GraphBuilderScreen(**kwargs)
    qtbot.addWidget(screen)
    return screen


def test_load_path_returns_before_the_database_opens(
        qtbot, sleeping_database, tmp_path):
    """The property the freeze violated: it dispatches, it does not read."""
    opened, released = sleeping_database
    screen = _screen(qtbot)

    started = time.monotonic()
    screen.load_path(str(tmp_path / "measurements.db"))
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"load_path took {elapsed:.1f}s -- it is listing the tables on the "
        "GUI thread again, which is the freeze")
    assert opened.wait(5.0), (
        "the database was never opened at all; the test proves nothing")

    # Let the parked worker go while the screen is still there to be told.
    released.set()
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=10000)
    assert "could not read measurements.db" in screen._source.text()


def test_choosing_a_file_from_the_dialog_returns_just_as_fast(
        qtbot, sleeping_database, tmp_path, monkeypatch):
    """The button is a `clicked` slot: that is how the freeze was reached."""
    opened, released = sleeping_database
    monkeypatch.setattr(
        "spacr.qt.screens.graph_builder.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(tmp_path / "measurements.db"), "")))
    screen = _screen(qtbot)

    started = time.monotonic()
    screen.choose_table()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"choose_table took {elapsed:.1f}s waiting for the database")
    assert opened.wait(5.0)
    released.set()
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=10000)


def test_the_picker_still_lists_every_table_once_the_worker_answers(
        qtbot, tmp_path):
    """The listing moved off the GUI thread; it did not go away.

    Rule two of the freeze work: nothing the user saw may disappear, it may
    only arrive a moment later.
    """
    database = tmp_path / "measurements.db"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE zebra (a REAL)")
        db.execute("INSERT INTO zebra VALUES (2.0)")
        db.execute("CREATE TABLE cell (a REAL)")
        db.execute("INSERT INTO cell VALUES (1.0)")

    screen = _screen(qtbot)
    screen.load_path(str(database))
    qtbot.waitUntil(lambda: not screen.is_busy(), timeout=10000)

    listed = [screen._table_picker.itemText(i)
              for i in range(screen._table_picker.count())]
    assert listed == ["cell", "zebra"], "preferred tables first, as before"
    assert screen._table_picker.isVisibleTo(screen) is True
    assert screen._table_picker.currentText() == "cell"
    assert "cell" in screen._source.text()
    assert screen._frame is not None
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=10000)


def test_filling_the_picker_does_not_start_a_second_load(qtbot, tmp_path):
    """`addItems` moves the current index, and that index is wired to a load.

    Populating the picker from the delivery callback puts it inside the
    signal it used to be safely outside of.
    """
    database = tmp_path / "measurements.db"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE cell (a REAL)")
        db.execute("INSERT INTO cell VALUES (1.0)")
        db.execute("CREATE TABLE nucleus (a REAL)")
        db.execute("INSERT INTO nucleus VALUES (2.0)")

    screen = _screen(qtbot, threaded=False)
    loads = []
    original = screen.load_path
    screen.load_path = lambda path, table=None: (
        loads.append((path, table)), original(path, table))[1]

    screen.load_path(str(database))

    assert loads == [(str(database), None)], (
        f"the picker reloaded the table it had just loaded: {loads}")


def test_a_csv_is_never_opened_as_a_database(qtbot, tmp_path, monkeypatch):
    """`sqlite_master` has nothing to say about a text file.

    Kept from the inline version: listing tables on a CSV raises, and the
    screen would report "could not read" for a file it reads perfectly well.
    """
    csv = tmp_path / "plate.csv"
    csv.write_text("a,b\n1,2\n")

    asked = []
    monkeypatch.setattr("spacr.qt.screens.graph_builder.table_names",
                        lambda path: asked.append(path) or ["cell"])

    screen = _screen(qtbot, threaded=False)
    screen.load_path(str(csv))

    assert asked == []
    assert screen._table_picker.isVisibleTo(screen) is False
    assert "could not read" not in screen._source.text()
    assert screen._frame is not None


# ---------------------------------------------------------------------------
# The edges the first pass left: a stale failure, and a lost picker
# ---------------------------------------------------------------------------


def test_a_superseded_failure_never_paints_over_the_load_that_won(
        qtbot, tmp_path, monkeypatch):
    """`JobRunner.job_failed` carries no generation. This is what that costs.

    `JobRunner.cancel` bumps a generation and the runner then refuses to
    deliver a stale job's RESULT -- but a stale job's FAILURE is emitted by
    `_on_worker_error_text` with no such check. So the exact gesture the
    freeze work is about, giving up on a sleeping share and opening something
    local instead, ended with the local table on screen under the label
    "could not read <the local file>: unable to open database file": the
    parked job's error, twenty seconds late, attributed to the wrong file.

    The fix is not a guard on the slot -- the slot cannot tell which job it
    is hearing from. It is that a read which fails comes back as data, on the
    same generation-guarded path as the frame.
    """
    from spacr.qt.screens import graph_builder as gb

    parked = threading.Event()
    released = threading.Event()
    asleep = str(tmp_path / "asleep.db")
    local = str(tmp_path / "local.db")

    def names(path):
        if path == asleep:
            parked.set()
            released.wait(SLOW_S)
            raise sqlite3.OperationalError("unable to open database file")
        return ["cell"]

    monkeypatch.setattr(gb, "table_names", names)
    monkeypatch.setattr(
        gb, "read_table",
        lambda path, table=None, limit=None: pd.DataFrame({"area": [1.0, 2.0]}))

    screen = _screen(qtbot)
    screen.load_path(asleep)
    assert parked.wait(5.0), "the slow load never started; the test is void"
    screen.load_path(local)                       # the user gave up and moved
    qtbot.waitUntil(lambda: screen._frame is not None, timeout=10000)

    won = screen._source.text()
    assert "local.db" in won and "could not read" not in won, won

    released.set()                                # the share finally answers
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=10000)
    qtbot.wait(250)                               # and any queued slot runs

    assert screen._source.text() == won, (
        "the abandoned load reported its failure against the file that "
        f"replaced it: {screen._source.text()!r}")
    assert screen._frame is not None


def test_a_table_that_will_not_read_still_leaves_every_table_offered(
        qtbot, tmp_path, monkeypatch):
    """Rule two again: the picker is how the user reaches the next table.

    Inline, the listing happened first and unconditionally, so a read that
    blew up on one table of a database left the other tables sitting in the
    picker, one click away. Moving the listing into the job put it behind the
    read's exception: the whole job failed, `_on_frame_loaded` never ran, and
    the user was left with an error message and an empty picker on a database
    whose other tables are fine.
    """
    from spacr.qt.screens import graph_builder as gb

    def unreadable(path, table=None, limit=None):
        raise sqlite3.DatabaseError("database disk image is malformed")

    monkeypatch.setattr(gb, "table_names", lambda path: ["cell", "zebra"])
    monkeypatch.setattr(gb, "read_table", unreadable)

    screen = _screen(qtbot, threaded=False)
    screen.load_path(str(tmp_path / "measurements.db"))

    listed = [screen._table_picker.itemText(i)
              for i in range(screen._table_picker.count())]
    assert listed == ["cell", "zebra"], (
        f"the tables that listed fine must still be offered; got {listed}")
    assert screen._table_picker.isVisibleTo(screen) is True
    assert "could not read measurements.db" in screen._source.text()
    assert "database disk image is malformed" in screen._source.text()
    assert screen._frame is None, "and nothing is plotted"


def test_a_file_that_is_not_a_database_offers_no_tables_at_all(
        qtbot, tmp_path):
    """The contrast: when the LISTING is what failed there is nothing to show.

    Kept as its own case because the two halves of the job fail differently
    on purpose -- this one must not leave the previous file's tables in the
    picker, where clicking one would reload the file that just failed.
    """
    broken = tmp_path / "not-a-database.db"
    broken.write_bytes(b"this is not sqlite")

    screen = _screen(qtbot, threaded=False)
    screen.load_path(str(broken))

    assert screen._table_picker.count() == 0
    assert screen._table_picker.isVisibleTo(screen) is False
    assert "could not read not-a-database.db" in screen._source.text()
    assert screen._frame is None


def test_the_failure_reads_the_same_as_it_did_from_the_runner(qtbot,
                                                              tmp_path,
                                                              monkeypatch):
    """Moving the failure off `job_failed` must not reword it.

    `JobRunner._on_worker_error_text` shows the last line of the worker's
    traceback, so that is the sentence the user has been reading. `_one_line`
    reproduces it rather than substituting `str(exc)`, which would have
    dropped the exception's name from every message on the screen.
    """
    from spacr.qt.screens import graph_builder as gb

    def unreadable(path, table=None, limit=None):
        raise sqlite3.OperationalError("no such table: ghost")

    monkeypatch.setattr(gb, "table_names", lambda path: ["cell"])
    monkeypatch.setattr(gb, "read_table", unreadable)

    screen = _screen(qtbot, threaded=False)
    screen.load_path(str(tmp_path / "measurements.db"), table="ghost")

    assert screen._source.text() == (
        "could not read measurements.db: "
        "sqlite3.OperationalError: no such table: ghost")
