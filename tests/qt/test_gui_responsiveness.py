"""The GUI thread must keep running the event loop while spaCR works.

This is the file that pins the feature. Everything else about background work
-- which screen threads what, which spinner turns -- is an implementation
detail of the one property a user actually notices: **the window keeps
repainting**. So the tests here do not inspect threads. They install an
event-loop watchdog, drive a real screen through a slow load, and assert an
upper bound on the longest gap between two consecutive timer ticks.

The technique is the standard event-loop lag monitor. A ``QTimer`` on the GUI
thread records ``perf_counter()`` at every tick; the gap since the previous
tick is exactly how long the GUI thread spent inside something that never
returned to the event loop. A 2 s gap is a 2 s frozen window. Nothing else
measures what the user feels, and in particular timing the *call* does not:
a call that returns quickly can still have posted work that blocks later.

Budgets are stated, not derived. They are deliberately far above the numbers
measured on the machine this was written on, because CI is slower and a
flaky responsiveness test gets deleted rather than fixed. The measured
before/after, on a warm local SSD with a 200 000-row x 48-column table:

    TabulateScreen.load_path       2034 ms  ->  1.2 ms to dispatch
    PCAScreen.load_path            3045 ms  ->  0.8 ms
    GraphBuilderScreen.load_path   2170 ms  ->  0.8 ms
    HomePage.refresh                774 ms  ->   16 ms

The budget below is 400 ms: a tenth of the worst "before", and 300x the
worst "after".

The second pass (item 6.20c) took the rest of the list, worst first. Same
watchdog, same method, same table:

    AppScreen._on_file_issue       2420 ms  ->   29 ms   (worst in the tree)
    AnnotateScreen._refresh_total  4844 ms  ->  328 ms
    PlateViewScreen.recompute x5   1012 ms  ->  3.7 ms
    PCAPanel.recompute             3784 ms  ->  881 ms
    first module open (mask)        914 ms  ->  485 ms
    AppScreen._refresh_usage       28.6 ms  ->  1.5 ms

Two of those are worth reading rather than skimming.

``_on_file_issue`` is the one that mattered most and is not on the numbers
above at its true size: the 2420 ms was measured against 1.2 s stand-ins for
``gh auth token`` and ``api.github.com``. The real call has an 8 s subprocess
timeout and a 20 s ``urlopen`` timeout, so the worst case it replaced was
**28 seconds of frozen window in response to one click**.

``first module open`` was not threaded, it was *deleted*. The 429 ms it lost
was ``import spacr.gui_utils``, reached for one pure-Python function; that
function now lives in ``spacr.settings_spec``, which imports nothing. An
import is usually removable and work usually is not, so the first question
about any stall here is which of the two it is. Threading a 3 s import off
the GUI thread still costs the memory and still makes the user wait.
"""
from __future__ import annotations

import os
import sqlite3
import time

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import QObject, Qt, QTimer

#: The longest the GUI thread may stop pumping events, in seconds, while a
#: background load runs. Generous on purpose -- see the module docstring.
STALL_BUDGET_S = 0.400

#: PCA gets its own, larger budget, and the difference is a known residual
#: rather than slack. The sklearn decomposition is no longer the reason: it
#: now runs on a worker like the read (see ``PCAPanel.recompute``, and the
#: contract note in its docstring for how the synchronous return was
#: resolved). What is left is the **drawing**, which cannot move: painting a
#: 200 000-point scatter costs 685 ms in ``canvas.set_result`` and 212 ms in
#: ``_apply_view``, measured separately, and a matplotlib canvas is a GUI
#: object. Threading it is not "hard", it is undefined behaviour.
#:
#: So this budget is no longer measuring computation at all; it is measuring
#: one redraw. It was 1.000 while the fit was inline, and it is 1.500 now --
#: which looks like the wrong direction until you notice what changed with
#: it: `PCAScreen.active_jobs()` now includes the panel's, so the watchdog
#: covers the *draw* as well as the read, and the draw was never inside the
#: window before. 881 ms measured on a 200 000-row table with the machine
#: otherwise idle leaves a 1.000 budget no headroom at all, and this file's
#: own rule is that a flaky responsiveness test gets deleted rather than
#: fixed. Stated rather than hidden, so it cannot quietly grow again.
PCA_STALL_BUDGET_S = 1.500

#: Per-screen budgets. Anything absent uses :data:`STALL_BUDGET_S`.
BUDGETS = {"pca": PCA_STALL_BUDGET_S}

#: Counting the Annotate population reaches ``spacr.io``, whose import pulls
#: torch: 2.7 s and 785 MB, once per process, and it happens on the worker.
#: A pure-Python import holds the GIL, so the GUI thread still loses time to
#: it even though it is not the thread doing the importing -- 328 ms cold,
#: 198 ms warm, measured. That is the residual this budget covers, and it is
#: an *import*, so the way to close it is to stop reaching for
#: ``_read_and_join_tables`` rather than to thread harder.
ANNOTATE_STALL_BUDGET_S = 1.000


class LoopWatchdog(QObject):
    """Record the gap between consecutive GUI-thread timer ticks."""

    def __init__(self, parent=None, interval_ms: int = 1):
        super().__init__(parent)
        self._last = time.perf_counter()
        self.worst = 0.0
        self.ticks = 0
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)

    def start(self):
        self._last = time.perf_counter()
        self.worst = 0.0
        self.ticks = 0
        self._timer.start()

    def stop(self):
        self._timer.stop()

    def _tick(self):
        now = time.perf_counter()
        gap = now - self._last
        self._last = now
        self.ticks += 1
        if gap > self.worst:
            self.worst = gap


@pytest.fixture
def big_db(tmp_path):
    """A measurement table big enough that reading it is unmistakably slow.

    30 000 rows x 40 float columns -- around 10 MB of sqlite, which
    ``pd.read_sql_query`` takes a few hundred ms to turn into a frame. Small
    enough to build in a second, big enough that running it on the GUI
    thread would blow the budget several times over.
    """
    path = tmp_path / "measurements.db"
    rng = np.random.default_rng(0)
    n = 30_000
    frame = pd.DataFrame(
        {f"cell_channel_{i}_mean_intensity": rng.uniform(0, 5e4, n)
         for i in range(40)})
    frame["plate"] = "plate1"
    frame["well"] = [f"A{(i % 12) + 1:02d}" for i in range(n)]
    frame["object_label"] = np.arange(n)
    with sqlite3.connect(path) as conn:
        frame.to_sql("cell", conn, index=False)
    return str(path)


def _drive(qtbot, dog, done, budget_s=20.0):
    """Pump the event loop until ``done()``, never blocking it."""
    end = time.perf_counter() + budget_s
    while time.perf_counter() < end and not done():
        qtbot.wait(20)
    qtbot.wait(50)
    dog.stop()


@pytest.mark.parametrize("screen_name", ["tabulate", "pca", "graph_builder"])
def test_loading_a_measurement_table_never_freezes_the_gui_thread(
        qtbot, big_db, screen_name):
    """The load that used to block for seconds now blocks for milliseconds.

    Drives the real screen the way the file dialog does, and asserts on the
    event loop rather than on the presence of a thread -- a screen could
    thread the read and still block on delivery, and the user would not be
    able to tell the difference.
    """
    import importlib
    module = importlib.import_module(f"spacr.qt.screens.{screen_name}")
    cls = {"tabulate": "TabulateScreen", "pca": "PCAScreen",
           "graph_builder": "GraphBuilderScreen"}[screen_name]
    screen = getattr(module, cls)()
    qtbot.addWidget(screen)
    screen.resize(900, 700)
    screen.show()
    qtbot.waitExposed(screen)
    qtbot.wait(100)

    dog = LoopWatchdog(screen)
    dog.start()
    dispatch = time.perf_counter()
    screen.load_path(big_db, "cell")
    dispatch = time.perf_counter() - dispatch
    _drive(qtbot, dog,
           lambda: not screen.is_busy() and screen.active_jobs() == 0)

    # The call itself must return immediately -- it dispatches, it does not
    # read. This is the part that is fully fixed on all three screens.
    assert dispatch < 0.100, (
        f"{cls}.load_path took {dispatch * 1000:.0f} ms to return; it is "
        "still doing the read on the GUI thread")
    budget = BUDGETS.get(screen_name, STALL_BUDGET_S)
    assert dog.ticks > 10, "the watchdog never ran; the measurement is void"
    assert dog.worst < budget, (
        f"{cls}.load_path stalled the GUI thread for "
        f"{dog.worst * 1000:.0f} ms (budget {budget * 1000:.0f} ms)")
    # And it actually loaded, rather than being responsive by doing nothing.
    assert screen._frame is not None
    assert len(screen._frame) == 30_000


def test_the_load_really_is_slow_enough_for_the_budget_to_mean_something(
        big_db):
    """Guard against the fixture shrinking until the test proves nothing.

    If reading the table inline were already under the budget, the test
    above would pass with the threading removed. It is not.
    """
    from spacr.qt.screens.graph_builder import read_table

    start = time.perf_counter()
    frame = read_table(big_db, "cell")
    elapsed = time.perf_counter() - start
    assert len(frame) == 30_000
    assert elapsed > 0.05, (
        f"reading the fixture took only {elapsed * 1000:.0f} ms; it is no "
        "longer a meaningful stand-in for a real measurement table")


def test_a_completed_job_leaves_no_job_behind(qtbot, big_db):
    """``active_jobs()`` returns to zero, and so does the run registry.

    This is the assertion that catches the ``thread.finished`` closure bug:
    with a closure the job is never retired, so this sits at 1 forever.
    """
    from spacr.qt.bridge import registry
    from spacr.qt.screens.tabulate import TabulateScreen

    screen = TabulateScreen()
    qtbot.addWidget(screen)
    before = len(registry().active())

    screen.load_path(big_db, "cell")
    assert screen.active_jobs() >= 1
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=20000)
    assert not screen.is_busy()
    qtbot.waitUntil(lambda: len(registry().active()) == before, timeout=20000)


def test_leaving_a_screen_mid_load_cancels_cleanly(qtbot, big_db):
    """Closing during a load must not crash, leak a thread, or deliver.

    Qt aborts the process if a running QThread is destroyed, and a worker
    that delivers into a closed widget is a use-after-free. Both are the
    reason ``JobRunner.shutdown`` exists.
    """
    from spacr.qt.screens.pca import PCAScreen

    screen = PCAScreen()
    qtbot.addWidget(screen)
    screen.show()
    qtbot.waitExposed(screen)

    delivered = []
    original = screen._on_frame_loaded
    screen._on_frame_loaded = lambda payload: delivered.append(payload)

    screen.load_path(big_db, "cell")
    assert screen.active_jobs() >= 1
    screen.close()                       # mid-load, deliberately

    # shutdown() drains: nothing is left running, and nothing was handed to
    # a widget that is on its way out.
    assert screen.active_jobs() == 0
    qtbot.wait(300)
    assert delivered == []
    assert original is not None


def test_two_loads_in_a_row_deliver_the_second_one(qtbot, big_db, tmp_path):
    """A superseded load must not paint over the one the user asked for."""
    from spacr.qt.screens.tabulate import TabulateScreen

    other = tmp_path / "small.db"
    with sqlite3.connect(other) as conn:
        pd.DataFrame({"a": [1, 2, 3]}).to_sql("cell", conn, index=False)

    screen = TabulateScreen()
    qtbot.addWidget(screen)
    screen.load_path(big_db, "cell")          # the slow one
    screen.load_path(str(other), "cell")      # supersedes it
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=20000)
    assert screen._frame is not None
    assert len(screen._frame) == 3            # the second one won


def test_returning_home_does_not_walk_the_run_journal_on_the_gui_thread(
        qtbot, tmp_path, monkeypatch):
    """Home's journal read is the most-travelled navigation in the app."""
    from spacr.qt.widgets import home as home_mod

    # A journal walk that is unambiguously slow, in the place the panels
    # read from. Patching the panel's own reader keeps this a test of where
    # the call happens, not of how fast the journal is.
    def slow_runs(*_a, **_k):
        time.sleep(0.6)
        return []

    def slow_totals(*_a, **_k):
        time.sleep(0.6)
        return {"total_runs": 7, "mask_runs": 1, "measure_runs": 2,
                "models_recorded": 3}

    monkeypatch.setattr(home_mod.RecentRunsPanel, "read", slow_runs)
    monkeypatch.setattr(home_mod.TotalsPanel, "read", slow_totals)

    page = home_mod.HomePage([("mask", "Mask", "d", "s")], lambda _k: None)
    qtbot.addWidget(page)
    page.resize(900, 700)
    page.show()
    qtbot.waitExposed(page)
    qtbot.wait(100)

    dog = LoopWatchdog(page)
    dog.start()
    page.refresh()
    _drive(qtbot, dog, lambda: page.active_jobs() == 0, budget_s=15.0)

    assert dog.ticks > 10
    assert dog.worst < STALL_BUDGET_S, (
        f"HomePage.refresh stalled the GUI thread for "
        f"{dog.worst * 1000:.0f} ms")
    # And it really did read and paint, rather than staying responsive by
    # skipping the work: "7" is the total the slow reader returned.
    from PySide6.QtWidgets import QLabel
    texts = [w.text() for w in page._totals.findChildren(QLabel)]
    assert "7" in texts, texts


# ---------------------------------------------------------------------------
# Item 6.20c — the rest of the list
#
# Same watchdog, same rule: assert on the event loop, not on the presence of
# a thread. A screen can thread its work and still block on delivery, and the
# user cannot tell the difference between that and never having threaded it.
# ---------------------------------------------------------------------------

@pytest.fixture
def annotate_db(tmp_path):
    """A ``measurements.db`` whose join is genuinely expensive.

    ``fetch_filtered_paths`` -- the branch behind a threshold filter --
    calls ``spacr.io._read_and_join_tables``, which reads every measurement
    table into pandas and merges them on the object key. That join is what
    used to run on the GUI thread on every settings apply, so the fixture
    has to be big enough for it to cost something real: 20 000 objects, two
    measurement tables, and a ``png_list`` carrying the ``cell_id`` the join
    is anchored on.
    """
    path = tmp_path / "measurements.db"
    rng = np.random.default_rng(3)
    n = 20_000
    frame = pd.DataFrame(
        {f"cell_channel_{i}_mean_intensity": rng.uniform(0, 5e4, n)
         for i in range(12)})
    frame["plateID"] = "plate1"
    frame["rowID"] = [f"r{(i % 8) + 1:02d}" for i in range(n)]
    frame["columnID"] = [f"c{(i % 12) + 1:02d}" for i in range(n)]
    frame["fieldID"] = [str((i % 4) + 1) for i in range(n)]
    frame["object_label"] = np.arange(n)
    frame["label"] = frame["object_label"]
    frame["prcfo"] = [
        f"plate1_{r}_{c}_{f}_{o}" for r, c, f, o in zip(
            frame["rowID"], frame["columnID"], frame["fieldID"],
            frame["object_label"])]
    frame["cell_area"] = rng.uniform(100, 2000, n)
    png = pd.DataFrame({
        "prcfo": frame["prcfo"],
        "cell_id": [f"o{i}" for i in range(n)],
        "png_path": [f"/nowhere/plate1_{i}_cell.png" for i in range(n)],
        "plateID": "plate1",
        "rowID": frame["rowID"],
        "columnID": frame["columnID"],
        "fieldID": frame["fieldID"],
        "test": [None] * n,
    })
    with sqlite3.connect(path) as conn:
        frame.to_sql("cell", conn, index=False)
        frame.to_sql("nucleus", conn, index=False)
        png.to_sql("png_list", conn, index=False)
    return str(path)


def _console_text(console) -> str:
    """Concatenate every stdout/error block rendered in a ConsolePanel."""
    from spacr.qt.widgets.console_panel import _StdoutBlock
    return "\n".join(b.text() for b in console.findChildren(_StdoutBlock))


class _BusyWatcher(QObject):
    """Did the window have anything to say it was busy, while it was busy?

    Samples the process-wide run registry -- :func:`spacr.qt.bridge.registry`
    -- on a 5 ms timer for the length of an operation. That registry is what
    drives :class:`~spacr.qt.widgets.activity_spinner.ActivitySpinner`, and
    ``make_thread`` maintains it, so a non-empty reading is exactly the fact
    worth pinning: **the work went through ``make_thread``**. Work that
    bypassed it would leave the user with a responsive window that gives no
    sign of doing anything, which is its own bug and is the one this catches.

    It deliberately does *not* assert on the spinner widget. Two reasons, and
    the first is the substantive one: the spinner defers appearing by two
    seconds so that brief work never flashes one, so for a job of about a
    second the shipped widget correctly shows nothing and an assertion that
    it turned would be asserting the opposite of the intended behaviour.
    Second, it would couple this file to that widget's timing policy, which
    is a preference the user can change.
    """

    def __init__(self, parent=None, interval_ms: int = 5):
        super().__init__(parent)
        self.busy_seen = False
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._sample)
        self._timer.start()

    def _sample(self):
        from spacr.qt.bridge import registry
        try:
            if registry().active():
                self.busy_seen = True
        except RuntimeError:            # pragma: no cover - teardown race
            self._timer.stop()

    def stop(self):
        self._timer.stop()


# -- 1. the worst one: network I/O in a click handler ----------------------

def test_filing_an_issue_does_not_freeze_the_window_for_half_a_minute(
        qtbot, monkeypatch):
    """``gh auth token`` + ``urlopen`` used to run in the click handler.

    Worst case there is an 8 s subprocess timeout followed by a 20 s HTTP
    timeout: 28 seconds of a window that does not repaint, has no cursor and
    cannot be cancelled, because somebody clicked "File as GitHub issue".
    The stand-ins below are 0.6 s each, which is enough to blow the budget
    several times over while keeping the test quick.
    """
    from spacr.qt.screens.app_screen import AppScreen

    def slow_file_issue(tb, active_app="", settings=None):
        time.sleep(1.2)          # the network, in one lump
        return "https://github.com/EinarOlafsson/spacr/issues/1"

    monkeypatch.setattr("spacr.qt.ai.issue_report.file_issue",
                        slow_file_issue)

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    screen._last_error_text = "Traceback (most recent call last):\nboom"
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=20000)

    busy = _BusyWatcher(screen)
    dog = LoopWatchdog(screen)
    dog.start()
    dispatch = time.perf_counter()
    screen._on_file_issue()
    dispatch = time.perf_counter() - dispatch
    _drive(qtbot, dog,
           lambda: not screen.is_busy() and screen.active_jobs() == 0,
           budget_s=25.0)

    assert dispatch < 0.100, (
        f"_on_file_issue took {dispatch * 1000:.0f} ms to return; it is "
        "still reaching GitHub on the GUI thread")
    assert dog.ticks > 10, "the watchdog never ran; the measurement is void"
    assert dog.worst < STALL_BUDGET_S, (
        f"filing an issue stalled the GUI thread for "
        f"{dog.worst * 1000:.0f} ms (budget {STALL_BUDGET_S * 1000:.0f} ms)")
    # And it really filed, rather than being responsive by doing nothing.
    text = _console_text(screen._console)
    assert "opened pre-filled report" in text, text[-400:]
    assert busy.busy_seen, (
        "the run registry never saw this job — the work went off the GUI "
        "thread without going through make_thread, so nothing can tell the "
        "user the window is busy")


def test_a_failed_issue_report_still_reaches_the_console(qtbot, monkeypatch):
    """The failure moved threads; it must not have moved out of sight.

    ``_on_error`` used to wrap the call in ``try/except`` and print
    "[issue] auto-file failed". Once the call is asynchronous that ``except``
    cannot see it, and a bug report that silently fails to send is worse than
    one that fails loudly.
    """
    from spacr.qt.screens.app_screen import AppScreen

    def boom(*_a, **_k):
        raise RuntimeError("github is down")

    monkeypatch.setattr("spacr.qt.ai.issue_report.file_issue", boom)
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    screen._last_error_text = "TB"
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=20000)

    screen._on_file_issue()
    qtbot.waitUntil(lambda: not screen.is_busy(), timeout=20000)
    assert "github is down" in _console_text(screen._console)


def test_the_usage_poll_does_not_shell_out_on_the_gui_thread(qtbot):
    """``GPUtil.getGPUs()`` spawns ``nvidia-smi``: 25 ms, every 2 seconds."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=20000)

    dog = LoopWatchdog(screen)
    dog.start()
    for _ in range(5):
        screen._refresh_usage()
        qtbot.wait(30)
    _drive(qtbot, dog, lambda: screen.active_jobs() == 0, budget_s=10.0)

    assert dog.ticks > 10
    assert dog.worst < STALL_BUDGET_S, (
        f"the usage poll stalled the GUI thread for "
        f"{dog.worst * 1000:.0f} ms")


def test_an_overlapping_usage_poll_is_skipped_rather_than_queued(qtbot):
    """A 2 s timer must not build a backlog on a machine slower than it."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=20000)
    for _ in range(20):
        screen._refresh_usage()
    assert screen.active_jobs() <= 1, (
        "twenty ticks queued twenty jobs; a slow nvidia-smi would leave the "
        "screen permanently behind")
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=20000)


def test_closing_a_module_mid_poll_leaves_no_surviving_thread(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    screen._refresh_usage()
    screen.close()
    assert screen.active_jobs() == 0
    qtbot.wait(200)


# -- 2. the heaviest single call: Annotate's population count --------------

def _annotate_screen(qtbot, db_path):
    from spacr.qt.screens.annotate import AnnotateScreen

    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    s = screen._settings
    s.db_path = db_path
    s.annotation_column = "test"
    s.measurement = ["cell_area"]
    s.threshold = [500.0]
    s.threshold_direction = ["higher"]
    assert screen._filter_active(), "the join branch is the expensive one"
    return screen


def test_counting_the_annotate_population_never_freezes_the_gui_thread(
        qtbot, annotate_db):
    """``_read_and_join_tables`` on every settings apply, inline, was 4.8 s."""
    screen = _annotate_screen(qtbot, annotate_db)
    screen.show()
    qtbot.waitExposed(screen)
    qtbot.wait(50)

    busy = _BusyWatcher(screen)
    dog = LoopWatchdog(screen)
    dog.start()
    dispatch = time.perf_counter()
    screen._refresh_total()
    dispatch = time.perf_counter() - dispatch
    _drive(qtbot, dog,
           lambda: not screen.is_busy() and screen.active_jobs() == 0,
           budget_s=90.0)

    assert dispatch < 0.100, (
        f"_refresh_total took {dispatch * 1000:.0f} ms to return; it is "
        "still joining the measurement tables on the GUI thread")
    assert dog.ticks > 10
    assert dog.worst < ANNOTATE_STALL_BUDGET_S, (
        f"counting the population stalled the GUI thread for "
        f"{dog.worst * 1000:.0f} ms "
        f"(budget {ANNOTATE_STALL_BUDGET_S * 1000:.0f} ms)")
    # It counted, rather than staying responsive by skipping the work. The
    # threshold keeps roughly the objects above 500 of a uniform 100..2000.
    assert screen._total > 0
    assert screen._filtered_rows is not None
    assert len(screen._filtered_rows) == screen._total
    assert busy.busy_seen, "the run registry never saw the count"


def test_the_annotate_count_really_is_slow_enough_to_matter(annotate_db):
    """Guard against the fixture shrinking until the test proves nothing."""
    from spacr.qt.screens.annotate import _compute_total
    from spacr.qt.screens.annotate import AnnotateSettings

    s = AnnotateSettings()
    s.db_path = annotate_db
    s.annotation_column = "test"
    s.measurement = ["cell_area"]
    s.threshold = [500.0]
    s.threshold_direction = ["higher"]
    start = time.perf_counter()
    out = _compute_total(s, True)
    elapsed = time.perf_counter() - start
    assert out["total"] > 0
    assert elapsed > 0.05, (
        f"the join took only {elapsed * 1000:.0f} ms; the fixture is no "
        "longer a meaningful stand-in for a real measurement database")


def test_a_superseded_annotate_count_does_not_land(qtbot, annotate_db):
    """Two settings applies in a row: the second one's count is the answer."""
    screen = _annotate_screen(qtbot, annotate_db)
    screen._refresh_total()                      # the wide one
    screen._settings.threshold = [1900.0]        # far fewer objects
    screen._refresh_total()                      # supersedes it
    qtbot.waitUntil(lambda: not screen.is_busy() and screen.active_jobs() == 0,
                    timeout=90000)
    assert screen._total < 4000, (
        "the superseded count painted over the one that was asked for")


def test_leaving_annotate_mid_count_cancels_and_cannot_hang(qtbot,
                                                            annotate_db):
    """Closing during the join must not deliver, leak a thread, or block.

    The close budget matters as much as the cancellation. ``closeEvent``
    used to call ``QThread.wait()`` with no argument on two workers, which is
    ULONG_MAX milliseconds: a wedged worker hung the close forever, with the
    window still on screen and nothing to click.
    """
    screen = _annotate_screen(qtbot, annotate_db)
    delivered = []
    screen._apply_total = lambda outcome, then=None: delivered.append(outcome)

    screen._refresh_total()
    assert screen.active_jobs() >= 1

    start = time.perf_counter()
    screen.close()
    elapsed = time.perf_counter() - start

    assert screen.active_jobs() == 0
    assert elapsed < 5.0, (
        f"closeEvent took {elapsed:.1f} s; an unbounded wait() is back")
    qtbot.wait(300)
    assert delivered == [], "a count was delivered into a closing screen"


def test_a_wedged_worker_cannot_hold_the_annotate_window_open(qtbot,
                                                              annotate_db):
    """The bounded wait, proved against a worker that refuses to stop.

    A ``QThread`` that ignores ``requestInterruption`` is exactly the case
    the unbounded ``wait()`` could not survive. The close must still finish,
    and the thread must be *parked* rather than terminated -- terminating a
    Python thread is a corrupt heap, which is the whole reason
    ``drain_thread`` exists.
    """
    from spacr.qt.screens.annotate import CLOSE_DRAIN_MS, _PageLoadWorker

    class _Stubborn(_PageLoadWorker):
        def run(self):
            time.sleep(2.0)      # outlives the close, ignores interruption

    screen = _annotate_screen(qtbot, annotate_db)
    stubborn = _Stubborn(0, [], lambda *a, **k: None)
    stubborn.start()
    qtbot.wait(50)
    screen._page_worker = stubborn
    # Wired the way `_queue_page_load` wires a real one, so `closeEvent`
    # walks the same disconnect path rather than a shortened one.
    stubborn.done.connect(screen._on_page_loaded)
    stubborn.finished.connect(screen._on_page_worker_finished)

    start = time.perf_counter()
    screen.close()
    elapsed = time.perf_counter() - start
    assert elapsed < (CLOSE_DRAIN_MS / 1000.0) + 2.0
    # Parked, not terminated: it is still allowed to finish on its own.
    qtbot.waitUntil(lambda: not stubborn.isRunning(), timeout=10000)


# -- 3. the import that was deleted rather than threaded -------------------

def test_opening_a_module_does_not_import_the_tk_interface(qtbot):
    """The Qt settings panel must not reach ``spacr.gui_utils``.

    It wanted one function, ``convert_settings_dict_for_gui``, which is a
    hundred lines of dictionary lookups. Reaching it through ``gui_utils``
    imported that module's *Tk* dependencies -- IPython, matplotlib.pyplot,
    cv2, tkinter, huggingface_hub, requests, PIL, screeninfo -- for 770 ms
    on the GUI thread, and that was the whole remaining cost of opening the
    first module.

    Asserting on ``sys.modules`` rather than on a duration, because the
    duration is what CI is bad at measuring and the import is the actual
    fact. ``spacr.utils`` is checked in the same breath: it costs 3.2 s and
    900 MB, and the Qt layer is deliberately built never to touch it.
    """
    import subprocess
    import sys as _sys

    # A subprocess, because by the time this test runs the suite has imported
    # most of spaCR and `sys.modules` here says nothing about a fresh launch.
    code = (
        "import os; os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')\n"
        "from PySide6.QtWidgets import QApplication\n"
        "app = QApplication([])\n"
        "from spacr.qt.screens.settings_model import SettingsWidgets\n"
        "SettingsWidgets('mask').build_sections()\n"
        "import sys\n"
        "print(','.join(m for m in ('spacr.gui_utils', 'spacr.utils',"
        " 'torch', 'cv2', 'IPython', 'tkinter') if m in sys.modules))\n"
    )
    out = subprocess.run([_sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    leaked = out.stdout.strip()
    assert leaked == "", (
        f"building a settings panel imported {leaked}. Each of those is "
        "seconds on the GUI thread of a module open; the function this "
        "needs lives in spacr.settings_spec, which imports nothing.")


def test_the_settings_spec_module_imports_nothing_expensive():
    """The point of the module, asserted rather than trusted."""
    import subprocess
    import sys as _sys

    code = (
        "import sys, time\n"
        "t = time.perf_counter()\n"
        "import spacr.settings_spec as s\n"
        "elapsed = time.perf_counter() - t\n"
        "assert s.convert_settings_dict_for_gui({'verbose': True}) == "
        "{'verbose': ('check', None, True)}\n"
        "print(elapsed, ','.join(m for m in ('torch', 'cv2', 'matplotlib',"
        " 'tkinter', 'IPython', 'pandas', 'numpy') if m in sys.modules))\n"
    )
    out = subprocess.run([_sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    elapsed, leaked = (out.stdout.strip().split(" ") + [""])[:2]
    assert leaked == "", f"spacr.settings_spec dragged in {leaked}"
    assert float(elapsed) < 0.100, (
        f"importing spacr.settings_spec took {float(elapsed) * 1000:.0f} ms")


def test_gui_utils_still_exports_the_function_it_used_to_own():
    """The Tk interface and every existing caller are unaffected."""
    from spacr.gui_utils import (_TORCHVISION_MODELS_CURATED,
                                 _torchvision_model_names,
                                 convert_settings_dict_for_gui)
    from spacr import settings_spec

    assert convert_settings_dict_for_gui is \
        settings_spec.convert_settings_dict_for_gui
    assert _torchvision_model_names is settings_spec._torchvision_model_names
    assert "resnet50" in _TORCHVISION_MODELS_CURATED
    assert convert_settings_dict_for_gui({"epochs": 10}) == \
        {"epochs": ("entry", None, 10)}


# -- 4. the spin-box drag ---------------------------------------------------

@pytest.fixture
def plate_db(tmp_path):
    """A plate big enough for the groupby behind a spin-box tick to show."""
    path = tmp_path / "measurements.db"
    rng = np.random.default_rng(7)
    n = 120_000
    frame = pd.DataFrame(
        {f"cell_channel_{i}_mean_intensity": rng.uniform(0, 5e4, n)
         for i in range(8)})
    frame["plate"] = "plate1"
    frame["row_name"] = [f"r{(i % 8) + 1:02d}" for i in range(n)]
    frame["column_name"] = [f"c{(i % 12) + 1:02d}" for i in range(n)]
    frame["field"] = [str((i % 4) + 1) for i in range(n)]
    frame["object_label"] = np.arange(n)
    with sqlite3.connect(path) as conn:
        frame.to_sql("cell", conn, index=False)
    return str(path)


def test_dragging_the_min_objects_box_never_freezes_the_plate_view(
        qtbot, plate_db):
    """487 ms of groupby fired on every tick of a spin box.

    Two things are asserted, because either alone leaves the drag broken:
    the aggregation is off the GUI thread, *and* a burst of ticks produces
    one aggregation rather than one per tick.
    """
    from spacr.qt.screens.plate_view import PlateViewScreen

    screen = PlateViewScreen(threaded=True)
    qtbot.addWidget(screen)
    screen.open_database(plate_db)
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=60000)
    screen.render_plate()
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=60000)
    assert screen._frame is not None

    recomputes = []
    real_recompute = screen.recompute

    def counted():
        recomputes.append(1)
        return real_recompute()

    screen.recompute = counted

    dog = LoopWatchdog(screen)
    dog.start()
    dispatch = time.perf_counter()
    for _ in range(8):                        # the drag
        screen._min_count_box.setValue(screen._min_count_box.value() + 1)
    dispatch = time.perf_counter() - dispatch
    _drive(qtbot, dog, lambda: screen.active_jobs() == 0, budget_s=60.0)

    assert dispatch < 0.100, (
        f"eight spin-box ticks took {dispatch * 1000:.0f} ms; the "
        "aggregation is still running on the GUI thread")
    assert dog.ticks > 10
    assert dog.worst < STALL_BUDGET_S, (
        f"the drag stalled the GUI thread for {dog.worst * 1000:.0f} ms")
    assert len(recomputes) <= 2, (
        f"eight ticks started {len(recomputes)} aggregations; they are not "
        "being coalesced, so the plate the user stopped on is drawn last")
    assert screen._layout_df is not None, "the drag drew nothing at all"


def test_an_unthreaded_plate_view_still_returns_its_answer(qtbot, plate_db):
    """The contract ``threaded=False`` preserves, stated as a test.

    ``recompute()`` returns "drawn, or being drawn". Unthreaded it can only
    mean the first, and every existing caller relies on that.
    """
    from spacr.qt.screens.plate_view import PlateViewScreen

    screen = PlateViewScreen(threaded=False)
    qtbot.addWidget(screen)
    assert screen.recompute() is False        # nothing loaded: a refusal
    screen.open_database(plate_db)
    screen.render_plate()
    assert screen.recompute() is True
    assert screen._layout_df is not None      # drawn by the time it returned


# -- 5. the sklearn fit -----------------------------------------------------

def test_recomputing_a_pca_never_freezes_the_gui_thread(qtbot, big_db):
    """The decomposition moved to a worker; only the redraw is left."""
    from spacr.qt.screens.pca import PCAScreen

    screen = PCAScreen()
    qtbot.addWidget(screen)
    screen.show()
    qtbot.waitExposed(screen)
    screen.load_path(big_db, "cell")
    qtbot.waitUntil(lambda: not screen.is_busy() and screen.active_jobs() == 0,
                    timeout=90000)

    busy = _BusyWatcher(screen)
    dog = LoopWatchdog(screen)
    dog.start()
    dispatch = time.perf_counter()
    assert screen.pca.recompute() is None      # threaded: the result is later
    dispatch = time.perf_counter() - dispatch
    _drive(qtbot, dog,
           lambda: not screen.is_busy() and screen.active_jobs() == 0,
           budget_s=90.0)

    assert dispatch < 0.100, (
        f"PCAPanel.recompute took {dispatch * 1000:.0f} ms to return; the "
        "fit is still on the GUI thread")
    assert dog.ticks > 10
    assert dog.worst < PCA_STALL_BUDGET_S, (
        f"the decomposition stalled the GUI thread for "
        f"{dog.worst * 1000:.0f} ms "
        f"(budget {PCA_STALL_BUDGET_S * 1000:.0f} ms)")
    assert screen.pca.result is not None, "responsive by computing nothing"
    assert busy.busy_seen, (
        "the run registry never saw the decomposition")


def test_an_unthreaded_pca_panel_still_returns_its_result(qtbot):
    """The other half of the contract in ``PCAPanel.recompute``'s docstring."""
    from spacr.qt.widgets.pca_view import PCAPanel

    panel = PCAPanel()                        # threaded=False by default
    qtbot.addWidget(panel)
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({f"f{i}": rng.normal(size=400) for i in range(6)})
    panel.set_frame(frame, compute=False)
    result = panel.recompute()
    assert result is not None, (
        "an unthreaded panel must return its PCAResult from the call — that "
        "is the whole reason the flag exists")
    assert result is panel.result


def test_leaving_the_pca_screen_mid_fit_leaves_no_surviving_thread(
        qtbot, big_db):
    from spacr.qt.screens.pca import PCAScreen

    screen = PCAScreen()
    qtbot.addWidget(screen)
    screen.load_path(big_db, "cell")
    qtbot.waitUntil(lambda: not screen.is_busy() and screen.active_jobs() == 0,
                    timeout=90000)
    delivered = []
    screen.pca._on_fit_done = lambda outcome: delivered.append(outcome)

    screen.pca.recompute()
    assert screen.pca.active_jobs() >= 1
    screen.close()                       # mid-fit, deliberately
    assert screen.active_jobs() == 0, (
        "closing the screen left the panel's decomposition running")
    qtbot.wait(300)
    assert delivered == [], "a fit was delivered into a closing panel"


def test_closing_a_module_stops_the_settings_panel_s_own_workers(qtbot):
    """A settings widget with a worker is a child, and children get no close.

    ``RowExclusionEditor`` reads a column's distinct values off a worker
    thread. It lives inside the settings panel, so when navigation destroys
    the screen it never receives a ``closeEvent`` of its own to shut that
    down from — the screen has to do it, and the screen is the only thing
    that knows the panel is going away.

    Asserted by capability rather than by class name, which is also how the
    screen does it: a settings widget that acquires a worker later is covered
    without anyone having to remember this.
    """
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("umap")
    qtbot.addWidget(screen)

    class _Widget:
        def __init__(self):
            self.stopped = 0

        def shutdown(self):
            self.stopped += 1

    spy = _Widget()
    screen._settings_model._widgets["_spy"] = spy
    screen.close()
    assert spy.stopped == 1, (
        "the settings panel's background work outlived the screen")


def test_a_hostile_settings_widget_cannot_block_a_close(qtbot):
    """Teardown must not be stoppable by one widget misbehaving."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)

    class _Hostile:
        def shutdown(self):
            raise RuntimeError("C++ object already deleted")

    screen._settings_model._widgets["_hostile"] = _Hostile()
    screen.close()          # must not raise
    assert screen.active_jobs() == 0
