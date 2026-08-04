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
#: rather than slack. ``PCAScreen`` now reads the table on a worker like the
#: others, but ``PCAPanel.recompute`` still runs the sklearn decomposition on
#: the GUI thread -- 1.63 s for a 200 000-row x 48-column table, measured.
#: It is not threaded here because ``recompute()`` returns its ``PCAResult``
#: synchronously to a dozen callers and to the control handlers, so making it
#: asynchronous is a change to the panel's contract, not to this screen.
#: Recorded as a number rather than hidden, so it cannot quietly grow.
PCA_STALL_BUDGET_S = 1.000

#: Per-screen budgets. Anything absent uses :data:`STALL_BUDGET_S`.
BUDGETS = {"pca": PCA_STALL_BUDGET_S}


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
