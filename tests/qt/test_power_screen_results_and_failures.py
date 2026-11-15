"""Power / Design: what a settled sweep puts on screen, and how it says it failed.

The screen's promise is that every number on it came out of a simulation that
actually ran.  So the table fills from whichever axis finished and clears when
neither did; a withheld metric is an em dash; the withheld-replicate note
appears only when a fit was withheld; a failure that produced nothing keeps
the worker's own traceback line; and none of it is ever a dialog.

The curves are built with the library's own ``power_curve`` from hand-written
scan frames, so every cell is exact.  One real (tiny) sweep is run, because
"stopped early" has to come from the cancel machinery itself.
"""
from __future__ import annotations

import threading
import warnings

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLabel                             # noqa: E402

import spacr.qt.screens.power as power                           # noqa: E402
from spacr.qt.screens.power import PowerScreen, run_power_sweep  # noqa: E402
from spacr.qt.widgets.power_design import DesignSpec, power_curve  # noqa: E402

pytestmark = pytest.mark.qt

CELLS_COLUMN = "imaging_n_cells_per_well_mu"
WELLS_COLUMN = "n_wells_per_screen"
NO_RUN = "No run yet — set the design and press Run."
NO_RESULT = "The sweep produced no result."
STOPPED_EARLY = ("Stopped early — the curves show only the points that "
                 "finished, not a design that ran out of power.")
CLIPPED = ("2 simulated screen(s) clipped a gene-in-well probability above 1, "
           "so the realised constructs per well is below the 4.6 you asked "
           "for. Raise the library evenness or lower constructs per well.")
WITHHELD_TWO = ("2 replicate(s) produced no usable fit and are counted as "
                "non-detections.")

#: The table rows the two hand-written curves give at the default design
#: (123 cells per well, 1536 wells): cells, wells, power, detected/replicates,
#: mean AUROC, mean AP, AP baseline, not converged, failed.
CELL_ROWS = [
    ["60", "1536", "0%", "0/2", "0.61", "0.55", "0.200", "0", "0"],
    ["123", "1536", "100%", "2/2", "0.92", "0.83", "0.200", "0", "0"],
    ["240", "1536", "100%", "2/2", "0.97", "0.88", "0.200", "0", "0"],
]
WELL_ROWS = [
    ["123", "768", "0%", "0/2", "0.71", "0.64", "0.200", "0", "0"],
    ["123", "1536", "100%", "2/2", "0.95", "0.86", "0.200", "0", "0"],
]


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
    return _curve(CELLS_COLUMN, [60.0, 60.0, 123.0, 123.0, 240.0, 240.0],
                  ["ok"] * 6, [0.60, 0.62, 0.91, 0.93, 0.97, 0.98])


def _wells_curve():
    return _curve(WELLS_COLUMN, [768.0, 768.0, 1536.0, 1536.0], ["ok"] * 4,
                  [0.70, 0.72, 0.95, 0.96])


def _tiny(**changes) -> DesignSpec:
    """The smallest design the simulator will still fit."""
    base = dict(
        n_genes=16, n_grnas_per_gene=1, cells_per_well=32.0,
        wells_per_plate=96, n_plates=1, constructs_per_well=4.0,
        background_positive_rate=0.10, effect_fold=6.0, hit_rate=0.25,
        reads_per_well=8000.0, gene_abundance_alpha=5.0,
        cells_per_well_var=200.0, class_pos_var=0.005, class_neg_var=0.005,
        sequencing_cells_per_well=300.0, pcr_factor_mu=1.0,
        pcr_factor_var=0.3, read_depth_cv=0.0,
        n_replicates=1, detection_auroc=0.80, seed=11, backend="torch",
    )
    base.update(changes)
    return DesignSpec(**base)


@pytest.fixture()
def screen(qtbot):
    """A synchronous Power screen; no QThread, so every assertion is exact."""
    widget = PowerScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


class _DeadWorker:
    """A worker whose C++ half has already gone."""

    def request_cancel(self, reason):
        raise RuntimeError("wrapped C/C++ object has been deleted")


class _DeadThread:
    def isRunning(self):                       # noqa: N802 - Qt name
        raise RuntimeError("wrapped C/C++ object has been deleted")


def test_power_screen_apply_result_tabulates_what_ran_and_notes_what_did_not(
        subtests, screen):
    """Rows, headline, curve descriptions and status for every result shape."""
    spec = DesignSpec()
    broken = _curve(CELLS_COLUMN, [123.0] * 3, ["ok", "not_converged",
                                                "failed"],
                    [0.95, float("nan"), float("nan")])
    dead = _curve(CELLS_COLUMN, [123.0, 123.0], ["failed", "not_converged"],
                  [float("nan"), float("nan")])
    cases = (
        # case id, result extras, rows, status, error
        ("both axes", {"cells_curve": _cells_curve(),
                       "wells_curve": _wells_curve()},
         CELL_ROWS + WELL_ROWS, "Done.", False),
        ("no curves at all clears the table",
         {"cells_curve": None, "wells_curve": None}, [], "Done.", False),
        ("only the cells axis finished",
         {"cells_curve": _cells_curve(), "wells_curve": None}, CELL_ROWS,
         "Done.", False),
        ("only the wells axis finished",
         {"cells_curve": None, "wells_curve": _wells_curve()}, WELL_ROWS,
         "Done.", False),
        ("a withheld fit keeps its denominator",
         {"cells_curve": broken, "wells_curve": None},
         [["123", "1536", "33%", "1/3", "0.95", "0.85", "0.200", "1", "1"]],
         WITHHELD_TWO, False),
        ("a point where every fit died is an em dash",
         {"cells_curve": dead, "wells_curve": None},
         [["123", "1536", "0%", "0/2", "—", "—", "—", "1", "1"]],
         WITHHELD_TWO, False),
        ("stopped early and clipped",
         {"cells_curve": None, "wells_curve": None, "cancelled": True,
          "n_clipped_screens": 2}, [], f"{STOPPED_EARLY} {CLIPPED}", False),
    )
    for case_id, extras, rows, status, error in cases:
        with subtests.test(msg=case_id):
            result = {"spec": spec, **extras}
            screen._apply_result(result)
            assert screen.result() is result
            assert screen.table_rows() == rows
            assert screen.status_text() == status
            assert screen.status_is_error() is error

    with subtests.test(msg="the headline and the curves describe the run"):
        screen._apply_result({"spec": spec, "cells_curve": _cells_curve(),
                              "wells_curve": None})
        assert screen.answer_text().startswith(
            "At 123 cells per well and 4.6 constructs per well, 1536 wells "
            "detect a 6.7-fold effect")
        assert screen._cells_view.describe() == (
            "Detection probability vs cells per well [cells imaged per "
            "well]: 60=0.00 (0/2), 123=1.00 (2/2), 240=1.00 (2/2)")
        assert screen._wells_view.describe() == (
            "Detection probability vs wells: no data")
        screen._apply_result({"spec": spec, "cells_curve": None,
                              "wells_curve": None})
        assert screen.answer_text() == NO_RUN

    with subtests.test(msg="no result at all"):
        screen._apply_result(None)
        assert screen.result() is None
        assert screen.status_text() == NO_RESULT
        assert screen.status_is_error() is True


def test_power_screen_failures_become_one_inline_line(subtests, screen):
    """A traceback, an empty traceback and a messageless exception, verbatim."""
    cases = (
        ("traceback ends in the exception",
         lambda: screen._on_worker_error_text(
             "Traceback (most recent call last):\n"
             "  File \"x.py\", line 3, in run\n"
             "ValueError: n_wells must be at least 2\n\n"),
         "The sweep failed: ValueError: n_wells must be at least 2"),
        ("an empty traceback",
         lambda: screen._on_worker_error_text("   \n\n"),
         "The sweep failed: unknown error"),
        ("an exception with no message is named by its type",
         lambda: screen._on_job_error(KeyboardInterrupt()),
         "The sweep failed: KeyboardInterrupt"),
        ("an exception with a message",
         lambda: screen._on_job_error(OSError("disk full")),
         "The sweep failed: disk full"),
    )
    for case_id, act, said in cases:
        with subtests.test(msg=case_id):
            screen._set_status("Running…", error=False)
            act()
            assert screen.status_text() == said
            assert screen.status_is_error() is True

    with subtests.test(msg="no style to repolish still writes the line"):
        assert screen._status.style() is not None
        screen._set_status("Running 9 fits…", error=False)
        assert screen.status_text() == "Running 9 fits…"
        assert screen.status_is_error() is False

        class _StylelessLabel(QLabel):
            def style(self):
                return None

        screen._status = _StylelessLabel(screen)
        screen._set_status("The sweep failed: no backend", error=True)
        assert screen.status_text() == "The sweep failed: no backend"
        assert screen.status_is_error() is True


def test_power_screen_settles_a_job_with_the_most_specific_message(
        subtests, screen, monkeypatch):
    """Failed, empty, cancelled, unrenderable and inline-raising settles."""
    settled = []
    screen.job_finished.connect(settled.append)

    with subtests.test(msg="a failure that produced nothing keeps its error"):
        screen._pending.clear()
        screen._cancel = None
        screen._on_worker_error_text("Traceback…\nRuntimeError: no CUDA device")
        screen._on_job_settled(False)
        assert screen.status_text() == (
            "The sweep failed: RuntimeError: no CUDA device")
        assert screen.status_is_error() is True
        assert settled == [False]

    with subtests.test(msg="success with no result is still a failure"):
        screen._on_job_settled(True)
        assert screen.status_text() == NO_RESULT
        assert screen.status_is_error() is True
        assert settled == [False, False]

    with subtests.test(msg="a cancel before the first fit says that"):
        screen._cancel = threading.Event()
        screen._cancel.set()
        screen._on_job_settled(False)
        assert screen.status_text() == "Stopped before the first fit finished."
        assert screen.status_is_error() is False
        assert settled == [False, False, False]
        screen._cancel = None

    with subtests.test(msg="a result that renders settles True"):
        screen._busy = True
        result = {"spec": DesignSpec(), "cells_curve": _cells_curve(),
                  "wells_curve": None}
        screen._pending.append({"result": result})
        screen._on_job_settled(True)
        assert screen.is_busy() is False
        assert screen.table_rows() == CELL_ROWS
        assert screen.status_text() == "Done."
        assert settled[-1] is True

    with subtests.test(msg="a render that raises reports it and fails"):
        def explode(result):
            raise ValueError("the curve has no auroc column")

        monkeypatch.setattr(screen, "_apply_result", explode)
        screen._pending.append({"result": {"spec": _tiny()}})
        screen._on_job_settled(True)
        assert screen.status_text() == (
            "The sweep failed: the curve has no auroc column")
        assert settled[-1] is False
        monkeypatch.undo()

    with subtests.test(msg="an inline sweep that raises"):
        def broken_sweep(payload):
            raise RuntimeError("the backend is not installed")

        monkeypatch.setattr(power, "run_power_sweep", broken_sweep)
        screen.set_spec(_tiny())
        before = len(settled)
        assert screen.run() is False
        assert screen.status_text() == (
            "The sweep failed: the backend is not installed")
        assert screen.status_is_error() is True
        assert screen.is_busy() is False
        assert settled[before:] == [False]


def test_power_screen_cancel_survives_a_worker_that_has_already_gone(screen):
    """The QThread may have retired between the click and the handler."""
    screen._cancel = threading.Event()
    screen._jobs.append((_DeadThread(), _DeadWorker()))
    screen._busy = True
    try:
        screen.cancel()
        assert screen._cancel.is_set()
        assert screen.status_text() == "Stopping after the fit in flight…"
        assert screen.status_is_error() is False
    finally:
        screen._jobs.clear()


@pytest.mark.slow
def test_power_sweep_stopped_between_points_keeps_what_finished(screen):
    """A cancel set before the first point stops after it, with no progress hook."""
    cancel = threading.Event()
    cancel.set()
    payload = {"spec": _tiny(), "cancel": cancel,
               "fit_kwargs": {"n_steps": 40, "n_draws": 16}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        returned = run_power_sweep(payload)
    result = payload["result"]
    assert returned is result
    assert result["cancelled"] is True
    assert result["wells_scan"].empty, "the second sweep must not have run"
    assert len(result["cells_scan"]) == 1, "it stops after the fit in flight"

    screen._apply_result(result)
    assert screen.status_text().startswith(STOPPED_EARLY)
    assert screen.result() is result
    assert len(screen.table_rows()) == 1
