"""The Power screen's failure paths, its cancel, and its withheld numbers.

A sweep is minutes of simulation started from a form the user can keep
editing. Everything here is about what the screen does when that goes wrong:
a worker that raises, a render that raises, a sweep stopped between grid
points, a metric the model would not produce. None of them may end in a
dialog (a modal hangs a headless run), and none of them may end in a number
that looks like a result.

One real -- if very small -- sweep is run, because "stopped early" has to be
produced by the cancel machinery rather than described by a fixture.
"""

from __future__ import annotations

import math
import threading
import warnings

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens.power import PowerScreen, run_power_sweep  # noqa: E402
from spacr.qt.widgets.power_design import DesignSpec  # noqa: E402


#: The smallest design the simulator will still fit, for tests whose subject
#: is the screen's plumbing rather than the statistics.
def _tiny(**changes) -> DesignSpec:
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


THREAD_FIT = {"n_steps": 40, "n_draws": 16}


@pytest.fixture()
def screen(qtbot):
    widget = PowerScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# Derived labels
# ---------------------------------------------------------------------------

def test_scoring_per_guide_says_the_library_is_the_construct_count(screen):
    """Guides and genes are different library sizes and different power.

    With ``score_per='guide'`` every construct gets its own coefficient, so
    the note has to price the dilution rather than imply the guides insure
    each other.
    """
    screen.set_spec(_tiny(score_per="guide", n_genes=24, n_grnas_per_gene=4))
    note = screen._library_note.text()
    assert "96 constructs get their own coefficient" in note
    assert "24 genes x 4 guides" in note


def test_scoring_per_gene_says_the_guides_were_pooled(screen):
    """The other half of the same choice, so the note above means something."""
    screen.set_spec(_tiny(score_per="gene", n_genes=24, n_grnas_per_gene=4))
    assert "24 genes get a coefficient" in screen._library_note.text()


@pytest.mark.parametrize("seconds,expected", [
    (30.0, "30 s"), (600.0, "10 min"), (7200.0, "2.0 h"), (36000.0, "10.0 h"),
])
def test_a_runtime_estimate_is_given_in_units_a_user_can_act_on(seconds,
                                                                expected):
    """"36000 s" is a number nobody can decide whether to wait for."""
    assert PowerScreen._humanise(seconds) == expected


# ---------------------------------------------------------------------------
# Withheld metrics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, "n/a", float("nan"), float("inf")])
def test_a_metric_the_model_withheld_is_a_dash_and_never_a_number(value):
    """A withheld AUROC printed as 0.50 reads as "the design found nothing".

    Those are different findings: one is a measurement, the other is the
    absence of one, and a table that spells them the same way cannot be read.
    """
    assert PowerScreen._fmt(value) == "—"


def test_a_real_metric_is_formatted_to_the_places_asked_for():
    """The dash above must not be swallowing an ordinary number."""
    assert PowerScreen._fmt(0.8123, 2) == "0.81"


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------

def test_an_inline_sweep_that_raises_is_reported_on_the_status_line(
        screen, monkeypatch):
    """Never a dialog: a modal hangs a headless run, and this screen is run
    headless in tests and on the cluster."""
    import spacr.qt.screens.power as power

    def explode(payload):
        raise RuntimeError("the backend is not installed")

    monkeypatch.setattr(power, "run_power_sweep", explode)
    screen.set_spec(_tiny())
    seen = []
    screen.job_finished.connect(seen.append)
    assert screen.run() is False
    assert "the backend is not installed" in screen.status_text()
    assert screen.status_is_error() is True
    assert screen.is_busy() is False
    assert seen == [False]


def test_a_worker_traceback_becomes_one_line_ending_in_the_exception(screen):
    """The last non-blank line of a traceback is the message; the rest is
    frames the user cannot act on."""
    screen._on_worker_error_text(
        "Traceback (most recent call last):\n"
        "  File \"x.py\", line 3, in run\n"
        "ValueError: n_wells must be at least 2\n\n")
    assert screen.status_text() == \
        "The sweep failed: ValueError: n_wells must be at least 2"
    assert screen.status_is_error() is True


def test_an_empty_worker_traceback_still_says_the_sweep_failed(screen):
    """Silence after a failure reads as a sweep that is still running."""
    screen._on_worker_error_text("   \n\n")
    assert screen.status_text() == "The sweep failed: unknown error"


def test_an_exception_with_no_message_is_named_by_its_type(screen):
    """``str(KeyboardInterrupt())`` is empty, and so was the status line."""
    screen._on_job_error(KeyboardInterrupt())
    assert screen.status_text() == "The sweep failed: KeyboardInterrupt"
    assert screen.status_is_error() is True


def test_a_sweep_that_produced_nothing_says_so(screen):
    """An empty answer panel is indistinguishable from a sweep still running."""
    screen._apply_result(None)
    assert screen.status_text() == "The sweep produced no result."
    assert screen.status_is_error() is True


def test_a_settled_job_whose_render_raises_reports_it_and_fails(screen,
                                                                monkeypatch):
    """The sweep succeeded and the drawing did not; say which."""
    def explode(result):
        raise ValueError("the curve has no auroc column")

    monkeypatch.setattr(screen, "_apply_result", explode)
    screen._pending.append({"result": {"spec": _tiny()}})
    seen = []
    screen.job_finished.connect(seen.append)
    screen._on_job_settled(True)
    assert "the curve has no auroc column" in screen.status_text()
    assert seen == [False]


def test_a_settled_job_with_no_result_at_all_says_so(screen):
    """A worker that reported success and returned nothing is still a failure."""
    seen = []
    screen.job_finished.connect(seen.append)
    screen._on_job_settled(True)
    assert screen.status_text() == "The sweep produced no result."
    assert screen.status_is_error() is True
    assert seen == [False]


def test_a_job_settled_after_a_cancel_before_the_first_fit_says_that(screen):
    """"produced no result" would read as a broken sweep, not a stopped one."""
    screen._cancel = threading.Event()
    screen._cancel.set()
    screen._on_job_settled(False)
    assert screen.status_text() == "Stopped before the first fit finished."
    assert screen.status_is_error() is False


# ---------------------------------------------------------------------------
# Cancelling and shutting down
# ---------------------------------------------------------------------------

class _DeadWorker:
    """A worker whose C++ half has already gone."""

    def request_cancel(self, reason):
        raise RuntimeError("wrapped C/C++ object has been deleted")


class _DeadThread:
    def isRunning(self):
        raise RuntimeError("wrapped C/C++ object has been deleted")


def test_cancelling_survives_a_worker_that_has_already_gone(screen):
    """The QThread may have retired between the click and the handler."""
    screen._cancel = threading.Event()
    screen._jobs.append((_DeadThread(), _DeadWorker()))
    screen._busy = True
    screen.cancel()
    assert screen._cancel.is_set()
    assert "Stopping after the fit in flight" in screen.status_text()
    screen._jobs.clear()


def test_closing_survives_a_thread_that_has_already_gone(screen):
    """Qt deletes the wrapper before the widget in some teardown orders."""
    from PySide6.QtGui import QCloseEvent

    screen._jobs.append((_DeadThread(), _DeadWorker()))
    screen._cancel = threading.Event()
    event = QCloseEvent()
    screen.closeEvent(event)
    assert event.isAccepted(), "the widget refused to close"
    assert screen._cancel.is_set(), "the sweep was not asked to stop"
    screen._jobs.clear()


def test_worker_progress_is_printed_in_the_spelling_home_parses(screen,
                                                                capsys):
    """It crosses the thread boundary as stdout, which is the only safe way.

    A signal emitted from the worker and connected with a direct connection
    would touch widgets off the GUI thread, which has aborted this process
    before. The ``Progress: n/total`` spelling is what ``bridge._PROGRESS_RE``
    reads, so the Home screen's bar fills without knowing about power.
    """
    screen._worker_progress(3, 12, "cells per well")
    assert capsys.readouterr().out.strip() == "Progress: 3/12 (cells per well)"


# ---------------------------------------------------------------------------
# A real sweep, stopped
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_a_sweep_stopped_between_points_keeps_what_finished(screen):
    """Stop must not be destructive.

    The fit itself is atomic, so cancelling takes effect between grid points:
    the points that finished are real and are drawn, and the screen says the
    curve is partial rather than letting it read as a design that ran out of
    power.
    """
    cancel = threading.Event()
    cancel.set()
    payload = {"spec": _tiny(), "cancel": cancel,
               "fit_kwargs": dict(THREAD_FIT)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_power_sweep(payload)
    result = payload["result"]
    assert result["cancelled"] is True
    assert result["wells_scan"].empty, "the second sweep must not have run"

    screen._apply_result(result)
    said = screen.status_text()
    assert "Stopped early" in said
    assert "ran out of power" in said
    assert screen.result() is result
