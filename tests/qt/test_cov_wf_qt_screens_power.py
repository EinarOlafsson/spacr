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

from PySide6.QtWidgets import QLabel

from spacr.qt.screens.power import PowerCurveView, PowerScreen
from spacr.qt.widgets.power_design import DesignSpec, power_curve

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


def _wells_curve():
    """Two well counts, the larger one better than the smaller."""
    return _curve(
        WELLS_COLUMN,
        [768.0, 768.0, 1536.0, 1536.0],
        ["ok", "ok", "ok", "ok"],
        [0.70, 0.72, 0.95, 0.96],
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


def test_a_result_with_no_curves_clears_the_table_and_reports_done(screen):
    """An empty result must wipe the previous run's rows, not sit under them.

    ``_apply_result`` is called for every settled sweep, including ones that
    produced no curve at all (a cancel before the first fit, a backend that
    fell over). If the table were left alone the screen would show the last
    good run's per-point numbers under a headline saying there was no run —
    the exact confusion between "simulated" and "left over" this screen is
    built to prevent.
    """
    spec = DesignSpec()
    screen._apply_result({"spec": spec,
                          "cells_curve": _cells_curve(),
                          "wells_curve": _wells_curve()})
    assert len(screen.table_rows()) == 5, "three cell points plus two well points"
    assert "123 cells per well" in screen.answer_text()

    screen._apply_result({"spec": spec,
                          "cells_curve": None,
                          "wells_curve": None})

    assert screen.table_rows() == [], "the stale rows survived an empty result"
    assert screen.answer_text() == "No run yet — set the design and press Run."
    assert screen.status_text() == "Done."
    assert screen.status_is_error() is False


def test_the_withheld_note_appears_only_when_a_fit_was_actually_withheld(screen):
    """A replicate that produced no fit is counted as a non-detection.

    That is a real cost to the number on screen — three replicates of which
    one crashed is a power over three, not over two — so the sentence has to
    say it happened. It equally has to stay silent when it did not: a note
    printed after every clean sweep is a note the screener stops reading, and
    then the one run where two fits died looks like every other run.
    """
    spec = DesignSpec()
    clean = {"spec": spec, "cells_curve": _cells_curve(), "wells_curve": None}
    screen._apply_result(clean)
    assert screen.status_text() == "Done.", "a clean sweep must not be annotated"

    broken = _curve(
        CELLS_COLUMN,
        [123.0, 123.0, 123.0],
        ["ok", "not_converged", "failed"],
        [0.95, float("nan"), float("nan")],
    )
    screen._apply_result({"spec": spec, "cells_curve": broken,
                          "wells_curve": None})

    said = screen.status_text()
    assert "2 replicate(s) produced no usable fit" in said
    assert "counted as non-detections" in said
    assert screen.table_rows()[0][3] == "1/3", "the denominator keeps all three"


def test_the_table_fills_from_whichever_axis_finished(screen):
    """A half-finished sweep must still show the points it did simulate.

    Stop is not abandon: cancelling between the two scans leaves one axis with
    real rows and the other with nothing. ``_fill_table`` walks the two curves
    independently so the finished axis is tabulated against the design's own
    fixed value for the other one — throwing the rows away because the run as
    a whole did not complete would make Stop destructive.
    """
    spec = DesignSpec()

    screen._apply_result({"spec": spec, "cells_curve": _cells_curve(),
                          "wells_curve": None})
    cells_only = screen.table_rows()
    assert [row[0] for row in cells_only] == ["60", "123", "240"]
    assert {row[1] for row in cells_only} == {"1536"}, \
        "the well count is the design's, held fixed across the cells sweep"

    screen._apply_result({"spec": spec, "cells_curve": None,
                          "wells_curve": _wells_curve()})
    wells_only = screen.table_rows()
    assert [row[1] for row in wells_only] == ["768", "1536"]
    assert {row[0] for row in wells_only} == {"123"}, \
        "the cell count is the design's, held fixed across the wells sweep"
    assert wells_only[1][2] == "100%", "1536 wells detected in both replicates"


def test_a_metric_no_fit_produced_is_an_em_dash_not_a_number(screen):
    """A point where every fit died must not print a plausible AUROC.

    ``mean_auroc`` averages only the replicates that converged, so a point
    with none is NaN. Formatted as a float that reads as ``nan`` — or worse,
    coerced to 0.5 — it would sit in the same column as the real numbers and
    be read as one. The em dash is the only honest answer, and the
    not-converged/failed columns beside it say why.
    """
    spec = DesignSpec()
    dead = _curve(
        CELLS_COLUMN,
        [123.0, 123.0],
        ["failed", "not_converged"],
        [float("nan"), float("nan")],
    )
    screen._apply_result({"spec": spec, "cells_curve": dead,
                          "wells_curve": None})

    row = screen.table_rows()[0]
    assert row[2] == "0%" and row[3] == "0/2"
    assert row[4] == "—" and row[5] == "—", "a withheld metric is an em dash"
    assert row[7] == "1" and row[8] == "1", "one not converged, one failed"

    screen._apply_result({"spec": spec, "cells_curve": _cells_curve(),
                          "wells_curve": None})
    assert screen.table_rows()[0][4] == "0.61", "a real mean is still printed"


# ---------------------------------------------------------------------------
# the settle path
# ---------------------------------------------------------------------------


def test_a_failure_that_produced_nothing_keeps_the_workers_own_error(screen):
    """The traceback line the worker sent must survive the job settling.

    ``worker.error`` arrives first and puts the real cause on the status line;
    ``worker.finished`` arrives after it with ``ok=False`` and no result. If
    that second call also wrote a status the user would be told "The sweep
    produced no result" — true, useless, and it would have erased the one
    line saying WHY. The generic line is reserved for the stranger case where
    the worker reported success and still handed back nothing.
    """
    settled: list = []
    screen.job_finished.connect(settled.append)
    screen._pending.clear()
    screen._cancel = None
    screen._on_worker_error_text("Traceback…\nRuntimeError: no CUDA device")

    screen._on_job_settled(False)

    assert screen.status_text() == \
        "The sweep failed: RuntimeError: no CUDA device"
    assert screen.status_is_error() is True
    assert settled == [False], "a resultless settle is not a success"

    screen._on_job_settled(True)
    assert screen.status_text() == "The sweep produced no result."
    assert settled == [False, False]


def test_the_status_line_still_speaks_when_it_has_no_style_to_repolish(screen):
    """The message matters more than its colour.

    The error colour is a dynamic property, so it only takes effect when the
    widget's style re-polishes it. ``style()`` can be gone — a widget being
    torn down, a label not yet in a styled hierarchy — and reaching through it
    unguarded would raise inside the one code path whose entire job is to
    report failures without a dialog. The user would lose the failure message
    and get a traceback in its place.
    """
    assert screen._status.style() is not None, "a live label has a style"
    screen._set_status("Running 9 fits…", error=False)
    assert screen.status_text() == "Running 9 fits…"

    class _StylelessLabel(QLabel):
        """A label whose style has gone, as one being torn down has."""

        def style(self):
            return None

    screen._status = _StylelessLabel(screen)
    screen._set_status("The sweep failed: no backend", error=True)

    assert screen.status_text() == "The sweep failed: no backend"
    assert screen.status_is_error() is True, \
        "the error flag is recorded even when nothing can repaint it"


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
