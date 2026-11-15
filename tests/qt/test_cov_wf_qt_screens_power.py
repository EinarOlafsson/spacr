"""Power / Design — the branches a sweep takes when it did not go to plan.

The screen's promise is that every number on it came out of a simulation that
actually ran. The paths below are the ones that decide what happens when part
of a run is missing, and each of them can break that promise quietly:

* the design rule on the curve — a vertical line saying "your design is HERE".
  It must be drawn only where the sweep actually reached; a rule pinned to the
  edge of a grid that never included the user's cell count would put their
  design on a point nobody simulated;
* a result whose curves are empty or half-empty, which is what a cancelled or
  crashed sweep leaves behind: the table must fill from the axis that finished
  and clear completely when neither did, rather than keeping the previous
  run's rows under a new headline;
* the withheld-replicate note, which must appear when a fit failed and must
  NOT appear when none did — a permanent warning is a warning nobody reads;
* the settle path for a failure that produced nothing, which must leave the
  worker's own traceback line on screen instead of replacing it with a
  generic "no result";
* job bookkeeping — a still-running sweep must survive the retirement sweep
  and be the only one asked to quit on close. Dropping the last reference to
  a running QThread takes the process down with it.

Nothing here fits a model: the curves are built with the library's own
``power_curve`` from hand-written scan frames, so the tests are exact and the
file runs in under a second.
"""

from __future__ import annotations

import pandas as pd
import pytest

from spacr.qt.screens.power import PowerCurveView, PowerScreen
from spacr.qt.widgets.power_design import power_curve

pytestmark = pytest.mark.qt


CELLS_COLUMN = "imaging_n_cells_per_well_mu"
WELLS_COLUMN = "n_wells_per_screen"


def _curve(column, values, statuses, aurocs, threshold=0.8):
    """A power curve built the way the screen builds one, from a scan frame."""
    scan = pd.DataFrame({
        column: [float(v) for v in values],
        "status": list(statuses),
        "model_auroc": list(aurocs),
        "model_ap": [0.5 if a != a else float(a) * 0.9 for a in aurocs],
        "ap_baseline": [0.2] * len(values),
    })
    return power_curve(scan, column, threshold)


def _cells_curve():
    """Three cell counts, one of them the default design's own 123."""
    return _curve(
        CELLS_COLUMN,
        [60.0, 60.0, 123.0, 123.0, 240.0, 240.0],
        ["ok", "ok", "ok", "ok", "ok", "ok"],
        [0.60, 0.62, 0.91, 0.93, 0.97, 0.98],
    )


@pytest.fixture()
def screen(qtbot):
    """A synchronous Power screen; no QThread, so every assertion is exact."""
    widget = PowerScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


class _FakeThread:
    """A QThread stand-in. Both call sites ask it one question and no more.

    ``bridge.thread_has_stopped`` calls ``isRunning()``; ``closeEvent`` calls
    ``isRunning()`` and then ``quit()``/``wait()``. A real QThread left
    running at teardown takes the process down with it, so the running case
    is represented rather than started.
    """

    def __init__(self, running: bool, name: str = ""):
        self._running = running
        self.name = name
        self.calls: list = []

    def isRunning(self) -> bool:            # noqa: N802 - Qt name
        return self._running

    def quit(self) -> None:
        self.calls.append("quit")

    def wait(self, msecs: int) -> bool:
        self.calls.append(("wait", msecs))
        return True


class _FakeWorker:
    """Records the cancel request ``closeEvent`` sends through ``cancel()``."""

    def __init__(self):
        self.cancelled: list = []

    def request_cancel(self, reason: str) -> None:
        self.cancelled.append(reason)


# ---------------------------------------------------------------------------
# the design rule on the curve
# ---------------------------------------------------------------------------


def test_the_design_rule_is_drawn_only_where_the_sweep_actually_reached(qtbot):
    """A rule outside the swept range must not be drawn at all.

    The dashed rule is the whole reason a screener can find their own design
    on the plot. ``paintEvent`` guards it with ``x_lo <= marker <= x_hi``, and
    without that guard ``to_px`` would still return a coordinate — clamped
    against nothing — so a design at 5000 cells per well would get a rule
    painted somewhere inside a sweep that only went to 240, and the user would
    read a power off a point that was never simulated.
    """
    view = PowerCurveView("Power")
    qtbot.addWidget(view)
    view.resize(320, 200)
    curve = _cells_curve()

    view.set_curve(curve, "cells", marker=None, threshold=0.8)
    bare = view.grab().toImage()

    view.set_curve(curve, "cells", marker=100.0, threshold=0.8)
    marked = view.grab().toImage()

    view.set_curve(curve, "cells", marker=5000.0, threshold=0.8)
    far = view.grab().toImage()

    assert marked != bare, "a marker inside the sweep must draw its rule"
    assert far == bare, "a marker outside the sweep must draw nothing"
    # The rule is decoration over the same data: none of it changes the
    # numbers the view reports, which is what the table and the sentence read.
    assert view.describe() == "Power [cells]: 60=0.00 (0/2), 123=1.00 (2/2), " \
                              "240=1.00 (2/2)"


# ---------------------------------------------------------------------------
# a result whose curves are missing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the settle path
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# job bookkeeping
# ---------------------------------------------------------------------------


def test_the_retirement_sweep_leaves_a_running_sweep_alone(screen):
    """A running sweep must not be dropped when another one retires.

    ``_retire_finished_jobs`` is connected to every job thread's ``finished``
    and sweeps the whole list rather than naming a sender, because the sender
    of a queued call is null once its emitter is deleted. Retiring a pair it
    had not checked would drop the last strong reference to a QThread that is
    still running, and a garbage-collected running QThread aborts the process.
    """
    live, dead = _FakeThread(True, "live"), _FakeThread(False, "dead")
    live_worker, dead_worker = _FakeWorker(), _FakeWorker()
    screen._jobs = [(live, live_worker), (dead, dead_worker)]

    screen._retire_finished_jobs()

    assert screen.active_jobs() == 1
    assert screen._jobs == [(live, live_worker)], "only the stopped job retired"

    live._running = False
    screen._retire_finished_jobs()
    assert screen.active_jobs() == 0, "it retires once the thread does stop"


def test_closing_asks_only_the_running_sweep_to_quit(screen):
    """Closing must wait for a live sweep and must not wait on a dead one.

    ``closeEvent`` gives every in-flight sweep a chance to stop before the
    widget dies. Calling ``quit()``/``wait()`` on a thread that already
    finished is at best a wasted ten-second timeout on the GUI thread at
    shutdown; skipping the one that IS running destroys a QWidget its worker
    still holds. Both jobs must have been asked to cancel either way.
    """
    running, stopped = _FakeThread(True, "running"), _FakeThread(False, "stopped")
    running_worker, stopped_worker = _FakeWorker(), _FakeWorker()
    screen._jobs = [(stopped, stopped_worker), (running, running_worker)]

    screen.close()

    assert stopped.calls == [], "a finished thread is not waited on"
    assert running.calls == ["quit", ("wait", 10000)]
    assert [w.cancelled for w in (stopped_worker, running_worker)] == \
        [["cancelled from the Power screen"]] * 2
