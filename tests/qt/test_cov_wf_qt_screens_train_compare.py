"""Training Runs — the branches that only open when a comparison goes sideways.

:mod:`spacr.qt.screens.train_compare` draws several runs' curves on one axes
and diffs their settings. The paths pinned here are the ones a user meets on a
bad day, and each of them can quietly say something false:

* the metric and fold pickers are live *before* anything has been compared. A
  picker that redrew (or re-compared) against no comparison would either throw
  on the first click after a scan or answer "Tick at least one run" at a user
  who has not asked for an overlay yet;
* a run folder with checkpoints but no curves is listed and can be overlaid.
  The plot then has no lines and therefore **no legend**, and the styling pass
  still has to run over it — a screen that only styles a populated axes leaves
  a white matplotlib rectangle on a dark page;
* ``optimal_threshold`` is logged per epoch and has no better direction. The
  click read-out must report its last value and must *not* invent a "best",
  because "best optimal_threshold 0.5" is a fabrication;
* a scan that fails half-way leaves the old curves on the canvas while the run
  list has already moved on. Clicking one of those stale lines must not
  attribute it to a run that is no longer listed, and must not print a metric
  reading for a metric that curve never logged;
* the retirement sweep runs on *every* job thread's ``finished`` and must keep
  the jobs that are still running — a garbage-collected running QThread takes
  the process down with it.

Everything runs offscreen against real temporary run folders built by the
builders in ``tests/test_train_compare.py``.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr.qt.screens.train_compare import TrainCompareScreen

from tests.test_train_compare import BASE_SETTINGS, make_run

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _contain_the_settings_climb(monkeypatch):
    """Keep ``load_run``'s search for ``settings/`` inside this test's tmp tree.

    Copied from ``tests/test_train_compare.py``: the climb walks six parents
    looking for the folder ``save_settings`` writes to, which from a run folder
    five levels under ``tmp_path`` reaches shared ground under /tmp that other
    tests drop a ``settings/`` folder into. Clamping to five keeps every level
    these runs actually use and stops one short of the shared parent, so the
    diff a test reads is the diff its own files produced.
    """
    import spacr.train_compare as _tc
    monkeypatch.setattr(_tc, "_SETTINGS_SEARCH_DEPTH", 5, raising=True)


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    """A synchronous screen — scans run inline, so every assertion is exact."""
    widget = TrainCompareScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def run_root(tmp_path):
    """Two ordinary runs, one 2-fold run, and one folder with no curves."""
    root = str(tmp_path / "tree")
    os.makedirs(root, exist_ok=True)
    make_run(root, "dsA", epochs=10,
             train=np.linspace(0.50, 0.80, 10),
             val=np.linspace(0.48, 0.74, 10),
             settings={**BASE_SETTINGS, "epochs": 10, "n_jobs": 30})
    make_run(root, "dsB", epochs=25,
             train=np.linspace(0.50, 0.93, 25),
             val=np.linspace(0.48, 0.86, 25),
             settings={**BASE_SETTINGS, "epochs": 25,
                       "learning_rate": 0.001, "n_jobs": 8})
    make_run(root, "dsCV", model_type="resnet50", epochs=6,
             folds={1: (np.linspace(0.5, 0.8, 6), np.linspace(0.5, 0.7, 6)),
                    2: (np.linspace(0.5, 0.9, 6), np.linspace(0.5, 0.8, 6))},
             settings={**BASE_SETTINGS, "model_type": "resnet50",
                       "cross_validation_folds": 2})
    broken = os.path.join(root, "dsX", "model", "maxvit_t", "rgb", "epochs_3")
    os.makedirs(broken)
    with open(os.path.join(broken, "maxvit_t_epoch_3_channels_rgb.pth"),
              "wb") as handle:
        handle.write(b"not really a checkpoint")
    return root


def _id_for(screen, needle):
    """The discovered run id whose folder path contains ``needle``."""
    for run in screen.runs():
        if needle in str(run.path):
            return run.run_id
    raise AssertionError(f"no discovered run under {needle}: "
                         f"{[str(r.path) for r in screen.runs()]}")


class _FakeThread:
    """A QThread stand-in for the retirement sweep, which asks one thing.

    ``bridge.thread_has_stopped`` calls ``isRunning()`` and nothing else, so a
    still-running worker can be represented without starting a real QThread —
    which, left running at teardown, takes the process down with it.
    """

    def __init__(self, running: bool, name: str = ""):
        self._running = running
        self.name = name

    def isRunning(self) -> bool:
        return self._running


# ---------------------------------------------------------------------------
# The pickers are live before anything has been compared
# ---------------------------------------------------------------------------

def test_picking_a_metric_before_any_overlay_draws_nothing_yet(screen,
                                                               run_root):
    """The metric picker is filled by the scan, long before an overlay exists.

    A user scans, reads the run list, and reaches for "Metric" before ticking
    anything — the combo is already populated, so the signal fires with no
    comparison in hand. Redrawing there would style an axes for a comparison
    that does not exist; the choice has to be remembered instead and honoured
    by the first real overlay.
    """
    assert screen.scan(run_root) is True
    assert "loss" in screen.available_metrics()

    assert screen.set_metric("loss") is True
    assert screen.selected_metric() == "loss"
    assert screen.comparison() is None, "no overlay was asked for"
    assert screen.figure().axes == [], "nothing may be drawn yet"
    assert screen.series_labels() == []

    # ...and the same choice, once a run is ticked, is what gets drawn.
    b = _id_for(screen, "dsB")
    assert screen.select_runs([b]) is True
    assert screen.overlay() is True
    assert screen.selected_metric() == "loss"
    assert screen.figure().axes[0].get_ylabel() == "loss"
    assert screen.series_labels() == [f"{b} · train", f"{b} · val"]


def test_picking_a_fold_mode_before_any_overlay_does_not_compare(screen,
                                                                 run_root):
    """The fold picker must not run a comparison nobody asked for.

    ``per fold`` / ``mean`` is a rendering choice, and the user sets it while
    reading the run list. If the picker re-compared on every change it would
    answer a user who has ticked nothing with "Tick at least one run to
    overlay" in red — an error message produced by the app's own signal, not by
    anything the user did wrong.
    """
    assert screen.scan(run_root) is True
    status_after_scan = screen.status_text()
    assert status_after_scan.startswith("Found 4 runs")

    assert screen.set_fold_mode("mean") is True
    assert screen.fold_mode() == "mean"
    assert screen.comparison() is None, "no comparison may be built here"
    assert screen.last_error == ""
    assert screen.status_text() == status_after_scan

    # The mode is honoured by the overlay the user does ask for.
    cv = _id_for(screen, "dsCV")
    assert screen.select_runs([cv]) is True
    assert screen.overlay() is True
    assert screen.comparison().fold_mode == "mean"
    assert screen.series_labels() == [
        f"{cv} · train · mean of 2 folds ±sd",
        f"{cv} · val · mean of 2 folds ±sd"]


# ---------------------------------------------------------------------------
# A run with no curves still gets a styled, legend-less plot
# ---------------------------------------------------------------------------

def test_a_run_with_no_curves_is_overlaid_as_an_empty_styled_axes(screen,
                                                                  run_root):
    """A checkpoint folder with no train.csv is comparable for its settings.

    It draws no lines, so matplotlib makes no legend — and the palette pass
    still has to run over that axes. Skipping the styling when there is nothing
    to legend would leave the empty plot at matplotlib's white default on a
    dark page, which is exactly the state a user meets right after scanning a
    half-finished run.
    """
    assert screen.scan(run_root) is True
    broken = _id_for(screen, "dsX")
    assert screen.select_runs([broken]) is True

    assert screen.overlay() is True
    axes = screen.figure().axes[0]
    assert screen.series_labels() == [], "a run with no curves plots no lines"
    assert axes.get_legend() is None, "nothing drawn, so nothing to legend"
    assert "no curves in" in screen.status_text()
    assert [t.get_text() for t in axes.texts] == [
        "no selected run logged 'accuracy'"]
    # The axes was still styled: the page's own surface, not matplotlib white.
    assert axes.patch.get_facecolor() != (1.0, 1.0, 1.0, 1.0)

    # A run that does have curves gets the legend the empty one could not.
    a = _id_for(screen, "dsA")
    assert screen.select_runs([a]) is True
    assert screen.overlay() is True
    legend = screen.figure().axes[0].get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == [
        f"{a} · train", f"{a} · val"]


# ---------------------------------------------------------------------------
# The click read-out
# ---------------------------------------------------------------------------

def test_a_metric_with_no_direction_reports_last_but_never_best(screen,
                                                                run_root):
    """"Best optimal_threshold" would be a fabrication, so it is not printed.

    ``optimal_threshold`` is written per epoch by ``evaluate_model_performance``
    and neither its largest nor its smallest value is "best". The read-out under
    the plot is what a user quotes when comparing runs, so a line reading "best
    optimal_threshold 0.5000 @ 25" would invent a result the run never claimed.
    """
    assert screen.scan(run_root) is True
    b = _id_for(screen, "dsB")
    assert screen.select_runs([b]) is True
    assert screen.overlay() is True
    label = f"{b} · val"

    accuracy_text = screen.identify_series(label)
    assert "best accuracy" in accuracy_text, "accuracy does have a best"
    assert "optimistic" in accuracy_text

    assert screen.set_metric("optimal_threshold") is True
    text = screen.identify_series(label)
    assert text.startswith(f"{label} · epochs 1–25")
    assert "last optimal_threshold 0.5000 @ 25" in text
    assert "best" not in text, "a directionless metric has no best epoch"
    assert screen.picked_text() == text
    assert "dsB" in text, "the folder the curve came from is named"


def test_a_stale_curve_is_not_attributed_to_a_run_that_is_gone(
        screen, run_root, tmp_path):
    """After a half-applied scan the canvas still shows the previous run.

    ``_apply_runs`` replaces the run list first and clears the canvas last, and
    the clear touches Qt — a canvas whose C++ half has gone raises RuntimeError
    there, which is the failure mode ``bridge.thread_has_stopped`` exists for.
    The screen then holds new runs behind old lines. Clicking one of those lines
    must not print the *new* run's folder next to the *old* run's curve, and
    must not quote a metric reading for a metric that curve never logged: both
    would be a confident, wrong answer to "which run is this line?".
    """
    assert screen.scan(run_root) is True
    b = _id_for(screen, "dsB")
    assert screen.select_runs([b]) is True
    assert screen.overlay() is True
    label = f"{b} · val"
    assert label in screen.series_labels()

    # A second tree whose only run logs a metric the first tree never had.
    other = tmp_path / "other"
    dst = other / "dsD" / "model" / "maxvit_t" / "rgb" / "epochs_4"
    dst.mkdir(parents=True)
    (dst / "train.csv").write_text(
        "epoch,dice\n1,0.10\n2,0.20\n3,0.30\n4,0.40\n", encoding="utf-8")

    def _canvas_is_gone():
        raise RuntimeError("Internal C++ object (PanelCanvas) already deleted")

    screen._clear_plot = _canvas_is_gone          # the canvas Qt took away
    assert screen.scan(str(other)) is False, "the scan did not fully apply"
    del screen._clear_plot                        # a fresh canvas, next scan

    assert "already deleted" in screen.last_error, "the failure is reported"
    assert len(screen.run_ids()) == 1, "the new tree's one run took the list"
    assert b not in screen.run_ids(), "the old run is no longer listed"
    assert screen.selected_metric() == "dice"
    assert label in screen.series_labels(), "the old curve is still drawn"

    text = screen.identify_series(label)
    assert text == f"{label} · epochs 1–25", (
        "a stale curve names itself and nothing it can no longer vouch for")
    assert "dsB" not in text, "the run behind it is gone, so no folder is named"
    assert "dice" not in text, "that curve never logged the current metric"

    # Re-scanning the tree the curve came from restores the full read-out.
    assert screen.scan(run_root) is True
    b_again = _id_for(screen, "dsB")
    assert screen.select_runs([b_again]) is True
    assert screen.overlay() is True
    restored = screen.identify_series(f"{b_again} · val")
    assert "dsB" in restored and "last accuracy" in restored


# ---------------------------------------------------------------------------
# Job bookkeeping
# ---------------------------------------------------------------------------

def test_the_retirement_sweep_keeps_a_job_that_is_still_running(screen):
    """A running scan must survive the retirement sweep of a finished one.

    ``_retire_finished_jobs`` is connected to *every* job thread's ``finished``
    and sweeps the whole list rather than naming a sender. If it retired the
    threads it had not checked, the strong reference to a still-running QThread
    would be dropped, and a garbage-collected running QThread takes the whole
    process down — the crash this list of ``(thread, worker)`` pairs exists to
    prevent.
    """
    live, done = _FakeThread(True, "live"), _FakeThread(False, "done")
    live_worker, done_worker = object(), object()
    screen._jobs = [(live, live_worker), (done, done_worker)]

    screen._retire_finished_jobs()

    assert screen.active_jobs() == 1
    assert screen._jobs == [(live, live_worker)], "only the stopped job retired"

    # And when that one stops too, the sweep takes it.
    live._running = False
    screen._retire_finished_jobs()
    assert screen.active_jobs() == 0
    assert screen._jobs == []
