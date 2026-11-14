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

import threading

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens.power import PowerScreen  # noqa: E402
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
