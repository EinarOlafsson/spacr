"""The source control describes a merge without freezing on the files.

THE FREEZE, 2026-09-04. `DatabaseSetWidget._refresh_summary` ran on the GUI
thread, on paths the user chose, and did two blocking things in a row:

    _rebuild -> _refresh_summary
      -> os.path.isfile('<plate>/measurements/measurements.db')  per source
      -> spacr.multi_database.describe_merge
        -> sqlite3.connect('file:<same path>?mode=ro', timeout=30)  x5

and it is called while a settings panel is being laid out, on every drop, and
on every workspace restore. One of the maintainer's plate folders was under an
``autofs`` mount whose share was asleep, where a single ``os.path.exists`` had
NOT RETURNED AFTER TWENTY SECONDS -- the stat is what triggers the automount.
The whole application was frozen, with no traceback, because a stalled event
loop is not a crash; it was reported as "opening map barcodes crashes spacr".

WHAT IS ASSERTED HERE. Not that the work is skipped -- every line the summary
printed before, it still prints -- but that the CALL RETURNS, and that the
answer lands afterwards. The blocking primitive is made to sleep
:data:`SLOW_S`; a fixed widget answers in milliseconds, and the widget as it
was would take the full sleep.
"""
from __future__ import annotations

import os
import sqlite3
import time

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.database_set import DatabaseSetWidget

#: Longer than any human would call responsive, far shorter than the twenty
#: seconds actually measured. A test that waited the real duration would be a
#: test nobody runs.
SLOW_S = 8.0

#: How long the answer may take to land once it is off the GUI thread. It is
#: a sqlite read of a tmp_path file, so this is orders of magnitude of slack.
LANDS_MS = 15000


def _sleep_like_a_sleeping_mount(seconds: float = SLOW_S) -> None:
    """Sleep, and let go if the thread doing it has been asked to stop.

    On the GUI thread nothing interrupts this, so a widget that still reads
    inline waits the whole of :data:`SLOW_S` -- which is the measurement these
    tests take. On a worker, `JobRunner.shutdown` requests an interruption at
    teardown and this lets go, so the session does not end with a parked
    thread burning a core for the rest of the suite.
    """
    from PySide6.QtCore import QThread

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            thread = QThread.currentThread()
            if thread is not None and thread.isInterruptionRequested():
                return
        except RuntimeError:
            return
        time.sleep(0.02)


def _plate(root, name, *, extra=None):
    """A plate folder in spaCR's own layout, with a measurements database."""
    folder = os.path.join(str(root), name)
    os.makedirs(os.path.join(folder, "measurements"), exist_ok=True)
    columns = ["plateID TEXT", "rowID TEXT", "columnID TEXT", "area REAL"]
    values = [name, "r1", "c1", 12.0]
    if extra:
        columns.append(f"{extra} REAL")
        values.append(1.0)
    with sqlite3.connect(os.path.join(folder, "measurements",
                                      "measurements.db")) as db:
        db.execute(f"CREATE TABLE cell ({', '.join(columns)})")
        db.execute(f"INSERT INTO cell VALUES ({', '.join('?' * len(values))})",
                   values)
    return folder


def _threaded_widget(qtbot, **kwargs):
    """A widget that reads on a worker -- the shape the application builds.

    Under pytest the widget defaults to `JobRunner(threaded=False)` so a test
    can read the summary on the next line; this file is the one that must have
    the real thing, so it asks for it.

    ``threaded`` is passed only if the constructor takes it, so that against
    the widget as it was these tests fail on the WAIT they are about rather
    than on a TypeError about a keyword that had not been added yet.
    """
    import inspect

    if "threaded" in inspect.signature(
            DatabaseSetWidget.__init__).parameters:
        kwargs["threaded"] = True
    made = DatabaseSetWidget(**kwargs)
    qtbot.addWidget(made)
    return made


def _let_the_worker_go(made) -> None:
    """Retire EVERY runner the widget has before Qt destroys the widget.

    Qt ABORTS THE PROCESS if a running QThread is collected, and these tests
    deliberately leave one asleep. `shutdown` requests an interruption and
    parks a thread that will not stop rather than terminating it, so this is
    bounded and safe.

    Asked of the widget, not of one named attribute: the summary read and the
    restore check are two runners, and the restore test leaves the second one
    asleep for `SLOW_S`. `getattr` rather than a bare call so that these tests
    run against the widget as it WAS, which had neither.
    """
    stop = getattr(made, "shutdown", None)
    if callable(stop):
        stop()
        return
    for name in ("_jobs", "_presence_jobs"):
        jobs = getattr(made, name, None)
        if jobs is not None:
            jobs.shutdown(200)


@pytest.fixture
def widget(qtbot):
    made = _threaded_widget(qtbot, mode="folder")
    yield made
    _let_the_worker_go(made)


@pytest.fixture
def slow_describe(monkeypatch):
    """Make reading the databases take :data:`SLOW_S`, as a sleeping mount does."""
    import spacr.multi_database as multi_database

    def crawl(paths, table="cell", **_kwargs):
        _sleep_like_a_sleeping_mount()
        raise AssertionError("the GUI thread waited for the databases")

    monkeypatch.setattr(multi_database, "describe_merge", crawl)
    return crawl


def test_adding_a_source_returns_before_the_databases_are_read(
        qtbot, widget, slow_describe, tmp_path):
    """The property the freeze violated: adding a plate does not wait."""
    roots = [_plate(tmp_path, "plate1"), _plate(tmp_path, "plate2")]

    started = time.monotonic()
    widget.add_sources(roots)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"add_sources took {elapsed:.1f}s -- it is describing the merge on "
        "the GUI thread again, which is the freeze")
    # And the user is told why the numbers are not there yet, rather than
    # being shown a blank line that looks like "these files hold nothing".
    assert "reading" in widget.summary.text()


def test_a_restored_workspace_does_not_stat_its_sources_inline(
        qtbot, widget, monkeypatch, tmp_path):
    """A workspace saved days ago holds exactly the paths most likely asleep.

    `os.path.exists` is patched only for the pretend mount, so nothing else in
    the process pays for it.
    """
    asleep = "/nas_mnt_asleep/plate1"
    real_exists = os.path.exists

    def sleepy(path):
        if str(path).startswith("/nas_mnt_asleep"):
            _sleep_like_a_sleeping_mount()
            return True
        return real_exists(path)

    monkeypatch.setattr(os.path, "exists", sleepy)

    started = time.monotonic()
    attached = widget.apply_workspace_state({"sources": [asleep]})
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"apply_workspace_state took {elapsed:.1f}s -- it is stat-ing the "
        "restored sources on the GUI thread")
    # Attached optimistically. A restore that dropped a plate because a mount
    # was asleep would be worse than the freeze it replaces, and the summary
    # names a database that really is missing either way.
    assert attached is True
    assert widget.sources() == [asleep]


def test_the_cost_of_the_merge_still_arrives(qtbot, widget, tmp_path):
    """Off the GUI thread is only correct if the answer still lands.

    The same assertions as the synchronous test in
    ``test_the_databases_are_a_working_set.py``, waited for instead of read
    on the next line: the rows, and the measurement about to be dropped.
    """
    roots = [_plate(tmp_path, "plate1"),
             _plate(tmp_path, "plate2", extra="perimeter")]

    widget.add_sources(roots)
    qtbot.waitUntil(lambda: widget.plan() is not None, timeout=LANDS_MS)

    text = widget.summary.text()
    assert "2 database(s)" in text
    assert "2 rows" in text
    assert "perimeter" in text


def test_a_source_with_no_database_is_still_named(qtbot, widget, tmp_path):
    """The stat moved to the worker, so its answer has to come back too.

    In folder mode the database is two levels below what the user picked, so
    silence would be indistinguishable from "that plate was measured".
    """
    never_measured = tmp_path / "never_measured"
    never_measured.mkdir()

    widget.add_sources([str(never_measured)])
    qtbot.waitUntil(
        lambda: "no measurements database" in widget.summary.text(),
        timeout=LANDS_MS)
    assert widget.plan() is None


def test_a_file_that_is_not_a_database_still_says_so(qtbot, tmp_path):
    """A failure is an ANSWER this summary has, and it survives the move."""
    fake = tmp_path / "measurements.db"
    fake.write_bytes(b"this is not a SQLite file at all")
    made = _threaded_widget(qtbot)
    try:
        made.add_sources([str(fake)])
        qtbot.waitUntil(
            lambda: made.summary.text().startswith("could not read"),
            timeout=LANDS_MS)
        assert "'cell'" in made.summary.text()
        assert made.plan() is None
    finally:
        _let_the_worker_go(made)


def test_three_plates_dropped_at_once_are_one_read(qtbot, widget,
                                                   slow_describe, tmp_path):
    """A drop is one `_rebuild` per plate, and they all ask the same question.

    Coalesced rather than queued: three worker threads racing on the same
    sqlite files is the cost this fix was supposed to remove, not add.
    """
    for name in ("plate1", "plate2", "plate3"):
        widget.add_sources([_plate(tmp_path, name)])

    assert widget._reading is True
    assert widget._read_again is True, (
        "a read arriving mid-flight should be remembered, not queued")
    assert widget._jobs.pending_jobs() == 1, (
        f"{widget._jobs.pending_jobs()} reads in flight for one drop")


def test_a_stale_answer_does_not_land_on_the_current_set(qtbot, widget,
                                                         tmp_path):
    """The set can change while a read is in flight; the old answer is dropped.

    Otherwise a slow mount would paint yesterday's row count over the plates
    the user has just chosen -- the summary saying one thing and the chips
    another, which is the failure this widget exists to prevent.
    """
    widget.add_sources([_plate(tmp_path, "plate1")])
    stale = widget._summary_token
    widget.add_sources([_plate(tmp_path, "plate2")])

    widget._summary_arrived(stale, ("error", (1, "an answer about the old set")))
    assert "an answer about the old set" not in widget.summary.text()


# --------------------------------------------------------------------------- #
#  The edges of the move, which the first pass left open
# --------------------------------------------------------------------------- #

def test_a_read_that_dies_does_not_leave_the_placeholder_lying(
        qtbot, widget, monkeypatch, tmp_path):
    """"reading 1 database(s)…" must not be the last word.

    `JobRunner` hands its result only to a job that SUCCEEDED, so a read that
    dies some other way reaches `_summary_arrived` never. The placeholder
    written before the submit would then stay on screen for the life of the
    widget, saying "still working" about work that has stopped -- and
    `_reading` would stay true with it, so every later change to the set would
    coalesce into a read that is never going to run.
    """
    import spacr.qt.widgets.database_set as database_set

    def fall_over(paths, table):
        raise RuntimeError("the read fell over")

    monkeypatch.setattr(database_set, "_read_the_merge", fall_over)

    widget.add_sources([_plate(tmp_path, "plate1")])
    qtbot.waitUntil(lambda: "reading" not in widget.summary.text(),
                    timeout=LANDS_MS)

    assert widget.summary.text().startswith("could not read")
    assert "the read fell over" in widget.summary.text()
    assert widget._reading is False, (
        "a read that died left the widget believing one is still in flight")

    # And the set is still editable: the next change reads again rather than
    # coalescing into the dead one.
    monkeypatch.undo()
    widget.add_sources([_plate(tmp_path, "plate2")])
    qtbot.waitUntil(lambda: widget.plan() is not None, timeout=LANDS_MS)
    assert "2 database(s)" in widget.summary.text()


def test_reading_the_merge_answers_rather_than_raises(monkeypatch, tmp_path):
    """Every path out of the worker is one of the summary's three answers.

    Including the ones that are not sqlite's fault: the import and the stat
    loop are inside the guard too, because a worker that raises delivers
    nothing at all.
    """
    from spacr.qt.widgets.database_set import _read_the_merge

    def refuse(path):
        raise OSError("the mount went away mid-stat")

    monkeypatch.setattr(os.path, "isfile", refuse)

    kind, payload = _read_the_merge(("/data/plate1/measurements.db",), "cell")
    assert kind == "error"
    assert payload[0] == 1
    assert "the mount went away mid-stat" in payload[1]


def test_a_pruned_restore_tells_the_panel_the_set_changed(qtbot, tmp_path):
    """The panel FOLLOWS the set, and now the pruning happens after it looks.

    `settings_model` rebuilds the fields that offer columns and rows from
    these databases on `value_changed`. While the check was inline the prune
    happened before the panel ever saw the set; off the GUI thread it happens
    after, so a silent prune would leave those fields offering the columns of
    a plate that has moved.
    """
    made = DatabaseSetWidget(mode="folder")       # unthreaded: pytest default
    qtbot.addWidget(made)
    here = _plate(tmp_path, "plate1")
    seen = []
    made.value_changed.connect(lambda: seen.append(made.sources()))

    assert made.apply_workspace_state(
        {"sources": [here, str(tmp_path / "moved_away")]}) is True

    assert made.sources() == [here]
    assert seen and seen[-1] == [here], (
        "the set was pruned without telling anything that follows it")


def test_the_restore_check_cannot_settle_the_summary_read(qtbot, widget,
                                                          slow_describe,
                                                          monkeypatch,
                                                          tmp_path):
    """Two runners, because `job_finished` carries no job identity.

    Shared, a restore check that fails would let go of the summary read's
    in-flight flag and paint its own error over a read that is still running
    -- the wrong file's answer on screen, which is the failure the token guard
    exists to prevent, arriving by another door.
    """
    import threading

    import spacr.qt.widgets.database_set as database_set

    fell_over = threading.Event()

    def refuse(sources):
        fell_over.set()
        raise RuntimeError("the restore check fell over")

    monkeypatch.setattr(database_set, "_the_ones_still_there", refuse)

    # The summary read of this plate is `slow_describe`, so it is still in
    # flight for SLOW_S -- the whole window this test looks at.
    widget.apply_workspace_state({"sources": [_plate(tmp_path, "plate1")]})
    qtbot.waitUntil(fell_over.is_set, timeout=LANDS_MS)
    # Long enough for the failure to be delivered on the GUI thread, and far
    # short of SLOW_S, so the read genuinely has not finished.
    qtbot.wait(750)

    assert widget._reading is True, (
        "the restore check settled the summary read it knows nothing about")
    assert "reading" in widget.summary.text(), (
        "the restore check's failure was painted over a read still running")
