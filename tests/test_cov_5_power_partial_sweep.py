"""A power sweep stopped after it had already produced points.

Stop is not abandon. The fit itself is atomic, so a cancel lands between grid
points: the points that finished are real simulations and must be kept and
drawn, the axis that never started must come back empty rather than absent,
and the curve has to be labelled as partial — otherwise a design that was
interrupted reads as a design that ran out of power.
"""
from __future__ import annotations

import os
import threading
import warnings

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from spacr.qt.screens.power import PowerScreen, run_power_sweep  # noqa: E402
from spacr.qt.widgets.power_design import DesignSpec             # noqa: E402

pytestmark = pytest.mark.qt

#: The smallest design the simulator will still fit.
TINY = dict(
    n_genes=16, n_grnas_per_gene=1, cells_per_well=32.0,
    wells_per_plate=96, n_plates=1, constructs_per_well=4.0,
    background_positive_rate=0.10, effect_fold=6.0, hit_rate=0.25,
    reads_per_well=8000.0, gene_abundance_alpha=5.0,
    cells_per_well_var=200.0, class_pos_var=0.005, class_neg_var=0.005,
    sequencing_cells_per_well=300.0, pcr_factor_mu=1.0,
    pcr_factor_var=0.3, read_depth_cv=0.0,
    n_replicates=1, detection_auroc=0.80, seed=11, backend="torch",
)

THREAD_FIT = {"n_steps": 40, "n_draws": 16}


def test_the_points_that_finished_before_the_stop_are_kept(qtbot):
    """Cancelling after the first point must not throw that point away."""
    cancel = threading.Event()
    seen = []

    def progress(done, total, label):
        seen.append((done, total, label))
        cancel.set()          # stop as soon as one point has been simulated

    payload = {"spec": DesignSpec(**TINY), "cancel": cancel,
               "progress": progress, "fit_kwargs": dict(THREAD_FIT)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_power_sweep(payload)

    assert seen, "the progress hook was never called"
    assert result["cancelled"] is True
    assert len(result["cells_scan"]) >= 1, "the finished point was discarded"
    assert result["wells_scan"].empty
    # An empty second sweep still has to be a frame with the same columns, or
    # the table and the curve below it cannot be built from it.
    assert list(result["wells_scan"].columns) == list(
        result["cells_scan"].columns)

    screen = PowerScreen(threaded=False)
    qtbot.addWidget(screen)
    screen._apply_result(result)

    said = screen.status_text()
    assert "Stopped early" in said
    assert "ran out of power" in said


def test_a_sweep_nobody_stopped_runs_both_axes(qtbot):
    """The control: without a cancel event both grids are scanned."""
    payload = {"spec": DesignSpec(**TINY), "fit_kwargs": dict(THREAD_FIT)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_power_sweep(payload)

    assert result["cancelled"] is False
    assert not result["cells_scan"].empty
    assert not result["wells_scan"].empty

    screen = PowerScreen(threaded=False)
    qtbot.addWidget(screen)
    screen._apply_result(result)
    assert "Stopped early" not in screen.status_text()
