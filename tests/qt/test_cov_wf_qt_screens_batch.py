"""Batch Runner — the branches that only a *disordered* night takes.

Everything here is a path the screen takes when its own bookkeeping and the
queue underneath it have momentarily disagreed, which over a seven-hour
unattended run is the normal case rather than the exotic one:

* a progress report that carries no total, no message, or names a job the user
  has since removed — the screen must absorb it, not reset the progress bar,
  wipe the status line or emit a status change for a job that is gone;
* a job whose row has not been drawn yet (or has just been undrawn), which is
  exactly the window ``_refresh_table`` opens between growing the table and
  filling it;
* the worker-thread retirement sweep, which must leave a thread that is *still
  running* alone and must not null out the live ``_thread`` handle when some
  other job's thread retires — getting that wrong is what once left
  ``active_jobs()`` permanently above zero;
* a theme whose palette has no colour for a status, and a queue file naming a
  module this build does not have.

Nothing here starts a process. No modal dialog is ever opened: every failure in
this screen is contractually inline, because a QMessageBox in an overnight run
is a hang until morning.
"""

from __future__ import annotations

import pytest

from PySide6.QtCore import Qt

from spacr import batch as bt
import spacr.qt.screens.batch as batch_mod
from spacr.qt.screens.batch import COLUMNS, BatchScreen

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _plate(tmp_path, name="plate1"):
    src = tmp_path / name
    src.mkdir(parents=True, exist_ok=True)
    (src / f"{name}_A01_T0001F001L01A01Z01C01.tif").write_bytes(b"")
    return str(src)


def _settings_csv(tmp_path, name, **values):
    path = tmp_path / name
    lines = ["Key,Value"] + [f"{k},{v}" for k, v in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


class _FakeRunner:
    """Stands in for ``spacr-run``; nothing real is ever executed."""

    def __call__(self, job, settings_path, log_path):
        return 0


class _FakeThread:
    """A QThread stand-in for the retirement sweep, which only asks one thing.

    ``bridge.thread_has_stopped`` calls ``isRunning()`` and nothing else, so a
    running worker can be represented here without starting a real QThread —
    which, left running at teardown, takes the process down with it.
    """

    def __init__(self, running: bool, name: str = ""):
        self._running = running
        self.name = name

    def isRunning(self) -> bool:
        return self._running


@pytest.fixture()
def screen(qtbot):
    """A synchronous screen, so every assertion is exact."""
    widget = BatchScreen(threaded=False, runner=_FakeRunner())
    qtbot.addWidget(widget)
    return widget


@pytest.fixture()
def settings(tmp_path):
    return _settings_csv(tmp_path, "mask.csv", src=_plate(tmp_path),
                         cell_channel=0)


# ---------------------------------------------------------------------------
# progress reports the screen must absorb
# ---------------------------------------------------------------------------


def test_a_report_with_no_total_leaves_the_progress_bar_scaled(screen, settings):
    """A queue-level event that carries no total must not rescale the bar.

    ``run()`` sizes the progress bar to the number of jobs before the first
    report arrives. Several events (``queue_stopped``, and anything emitted
    before the queue is counted) carry ``total=0``; taking that literally would
    set the bar's range to 0 mid-run, so the "3 / 7 jobs" the user checks at
    2 a.m. would read "3 / 0" and the bar would show as indeterminate.
    """
    screen.add_job("mask", settings)
    screen._progress.setRange(0, 7)

    screen._on_progress(bt.Progress(event="queue_stopped", total=0))
    assert screen._progress.maximum() == 7, "total=0 must not rescale the bar"

    screen._on_progress(bt.Progress(event="queue_started", total=3))
    assert screen._progress.maximum() == 3, "a real total does rescale it"


def test_a_report_with_no_message_keeps_the_status_line_it_found(screen,
                                                                 settings):
    """An event with no message must leave the last real message on screen.

    ``job_started`` events are emitted with an empty message for jobs the
    queue is only bookkeeping. Writing that empty string into the status label
    would blank the one line telling the user what the queue is doing, and
    would also silently clear ``last_error`` — the screen's only error channel,
    since it never opens a dialog.
    """
    screen.add_job("mask", settings)
    screen._set_status("Running 1 job(s), one at a time…")

    screen._on_progress(bt.Progress(event="job_started", job_id="mask-1",
                                    message=""))
    assert screen.status_text() == "Running 1 job(s), one at a time…"

    screen._on_progress(bt.Progress(event="job_finished", job_id="mask-1",
                                    status=bt.STATUS_FAILED,
                                    message="mask-1 failed: exit code 1"))
    assert screen.status_text() == "mask-1 failed: exit code 1"
    assert screen.last_error == "mask-1 failed: exit code 1"


def test_a_report_about_a_removed_job_emits_no_status_change(screen, settings):
    """A late report for a job the user deleted must not be re-announced.

    Reports cross a thread boundary, so one can arrive after the user has
    removed that job from the queue. ``job_status_changed`` is what other
    surfaces listen on; emitting it for an id that no longer exists would have
    them look up a job that is gone.
    """
    screen.add_job("mask", settings)
    seen = []
    screen.job_status_changed.connect(lambda job_id, status: seen.append((job_id, status)))

    screen._on_progress(bt.Progress(event="job_finished", job_id="mask-99",
                                    status=bt.STATUS_SUCCESS, message="stale"))
    assert seen == [], "no job with that id is in the queue"

    screen.queue().find("mask-1").status = bt.STATUS_SUCCESS
    screen._on_progress(bt.Progress(event="job_finished", job_id="mask-1",
                                    status=bt.STATUS_SUCCESS, message="done"))
    assert seen == [("mask-1", bt.STATUS_SUCCESS)]


# ---------------------------------------------------------------------------
# rows that are not drawn yet
# ---------------------------------------------------------------------------


def test_a_row_with_no_cells_yet_has_no_job_id(screen, settings):
    """A grown-but-unfilled row must resolve to no job, not to a crash.

    ``_refresh_table`` grows the table with ``setRowCount`` and only then
    fills the cells, and growing a table can make Qt re-emit the selection.
    Any lookup landing in that window sees a row whose column-0 item is None;
    reading ``.data(Qt.UserRole)`` off it would raise ``AttributeError`` inside
    a Qt slot, where it becomes an unhandled traceback rather than an error the
    user can see.
    """
    screen.add_job("mask", settings)
    screen._table.setRowCount(2)          # exactly what _refresh_table does first

    assert screen._job_id_at(1) == "", "row 1 has no item to carry an id"
    assert screen._job_id_at(0) == "mask-1", "row 0 was filled and still resolves"
    assert screen._row_of_job("mask-1") == 0
    screen._table.setCurrentCell(1, 0)
    assert screen.selected_job() is None, "an undrawn row selects no job"


# ---------------------------------------------------------------------------
# retiring worker threads
# ---------------------------------------------------------------------------


def test_the_sweep_keeps_a_thread_that_is_still_running(screen):
    """A running job must survive the retirement sweep of a finished one.

    ``_retire_finished_jobs`` is connected to *every* job thread's ``finished``
    and sweeps the whole list rather than naming a sender. If it retired
    threads it had not checked, the strong reference to a still-running QThread
    would be dropped and a garbage-collected running QThread takes the whole
    process down — the crash this sweep exists to avoid.
    """
    live, dead = _FakeThread(True, "live"), _FakeThread(False, "dead")
    live_worker, dead_worker = object(), object()
    screen._jobs = [(live, live_worker), (dead, dead_worker)]
    screen._thread, screen._worker = live, live_worker

    screen._retire_finished_jobs()

    assert screen.active_jobs() == 1
    assert screen._jobs == [(live, live_worker)], "only the stopped job retired"
    assert screen._thread is live, "the live handle is not cleared by another's exit"
    assert screen._worker is live_worker


def test_retiring_the_current_thread_releases_its_worker(screen):
    """The live handles must be released when the job holding them ends.

    ``_thread``/``_worker`` are the strong references keeping the running job
    alive. Left dangling after the job finished, the worker (and everything its
    closure captured — the whole queue) is never collected, and the next run
    would start with the previous run's handles still installed.
    """
    done = _FakeThread(False, "done")
    worker = object()
    screen._jobs = [(done, worker)]
    screen._thread, screen._worker = done, worker

    screen._retire_finished_jobs()

    assert screen.active_jobs() == 0
    assert screen._thread is None
    assert screen._worker is None


# ---------------------------------------------------------------------------
# the elapsed-time tick
# ---------------------------------------------------------------------------


def test_the_tick_skips_a_running_job_that_has_no_row(screen, settings):
    """The one-second tick must survive a queue the table has not caught up to.

    The tick fires on a QTimer, independently of every refresh, so it can land
    between the queue gaining a running job and the table being redrawn. Writing
    to row -1 raises inside a timer slot, which in an unattended run means a
    traceback per second for the rest of the night.
    """
    screen.add_job("mask", settings)
    job = screen.queue().find("mask-1")
    job.status = bt.STATUS_RUNNING
    job.started = bt._now_iso()
    screen._table.setRowCount(0)          # the table has not been redrawn yet

    screen._refresh_running_row()
    assert screen._table.rowCount() == 0, "no row was invented for the job"

    screen._refresh_table()
    screen._refresh_running_row()
    ticked = screen.row_values(0)[COLUMNS.index("Time")]
    assert ticked.endswith("s"), f"the drawn row does tick, got {ticked!r}"
    assert ticked != "—"


def test_the_tick_only_reloads_the_log_of_the_selected_job(screen, tmp_path,
                                                           settings):
    """Ticking must not drag another job's log into the pane the user is reading.

    The log pane follows the *selection*, and the tick runs every second while a
    job the user is not looking at writes megabytes. If the tick reloaded the
    running job's log regardless of selection, a user inspecting why job 3 failed
    would have the pane yanked out from under them once a second.
    """
    second = _settings_csv(tmp_path, "mask2.csv",
                           src=_plate(tmp_path, "plate2"), cell_channel=0)
    screen.add_job("mask", settings, label="first")
    screen.add_job("mask", second, label="second")
    running = screen.queue().find("mask-2")
    running.status = bt.STATUS_RUNNING
    running.started = bt._now_iso()
    log = tmp_path / "mask-2.log"
    log.write_text("cellpose: 1/40\n", encoding="utf-8")
    running.log_path = str(log)
    screen._refresh_table()
    screen._table.setCurrentCell(0, 0)     # the user is reading job 1
    assert screen.log_text() == "", "job 1 has no log of its own"

    screen._refresh_running_row()

    assert screen.log_text() == "", "the tick left the selected job's pane alone"
    assert screen.row_values(1)[COLUMNS.index("Time")] != "—"

    screen._table.setCurrentCell(1, 0)     # now the user is watching job 2
    log.write_text("cellpose: 2/40\n", encoding="utf-8")
    screen._refresh_running_row()
    assert screen.log_text() == "cellpose: 2/40\n"


# ---------------------------------------------------------------------------
# a palette with nothing to say, and a module this build does not have
# ---------------------------------------------------------------------------


def test_a_status_with_no_palette_colour_is_still_drawn(screen, settings,
                                                        monkeypatch):
    """A theme missing a status colour must leave the text readable, not blank.

    ``active_palette()`` is resolved per repaint and a theme (or a preferences
    file that failed to read) can come back without an entry for a role. Handing
    that empty string to ``QColor`` yields an *invalid* colour, which paints
    black-on-black in the dark theme; the row is drawn in the table's own
    foreground instead.
    """
    palette = dict(batch_mod.active_palette())
    palette["fg_muted"] = ""              # no colour for 'pending'
    monkeypatch.setattr(batch_mod, "active_palette", lambda: palette)
    screen.add_job("mask", settings)

    screen._refresh_table()
    pending = screen._table.item(0, COLUMNS.index("Status"))
    assert pending.text() == bt.STATUS_PENDING
    assert pending.foreground().style() == Qt.NoBrush, \
        "no colour was set, so the view's own foreground is used"

    screen.queue().find("mask-1").status = bt.STATUS_SUCCESS
    screen._refresh_table()
    ok = screen._table.item(0, COLUMNS.index("Status"))
    assert ok.foreground().style() != Qt.NoBrush
    assert ok.foreground().color().name().lower() == \
        _hex(palette["success"]), "a real palette entry is applied"


def _hex(colour: str) -> str:
    from PySide6.QtGui import QColor

    return QColor(colour).name().lower()


def test_selecting_a_job_whose_module_is_unknown_keeps_the_combo(screen,
                                                                 tmp_path):
    """A queue file from another spaCR must not silently retarget the combo.

    Queue files are hand-editable and portable, so one can name a module this
    build does not offer. ``findData`` returns -1 for it; using that index would
    clear the combo (or, with a plain ``setCurrentIndex``, leave it on an
    unrelated module), so the next Add would silently queue the wrong module.
    The rest of the editor still has to fill in, or the job cannot be inspected.
    """
    import json

    path = tmp_path / "queue.json"
    path.write_text(json.dumps({
        "spacr_queue": 1,
        "jobs": [{"module": "quantum_mask", "id": "future-1",
                  "label": "from a newer spaCR", "settings": "s.csv"}],
    }), encoding="utf-8")
    screen._module_combo.setCurrentIndex(screen._module_combo.findData("mask"))
    assert screen.load_queue_from(str(path)) is True

    assert screen.select_job("future-1") is True

    assert screen._module_combo.currentData() == "mask", \
        "an unknown module leaves the combo where the user put it"
    assert screen._label_edit.text() == "from a newer spaCR"
    assert screen._settings_edit.text() == "s.csv"
