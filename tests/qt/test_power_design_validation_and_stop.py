"""What a power design refuses before simulating, and what a stopped sweep keeps.

``DesignSpec.validate`` names every parameter that stops the sweep, all at
once and in the order of the form, and judges a rate's variance only once the
rate itself is in range.  A sweep stopped after its first grid point keeps the
points that finished, returns the axis that never started as an empty frame
with the same columns, and the Power screen labels the curve as stopped early
rather than as a design that ran out of power.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture.
"""
from __future__ import annotations

import threading
import warnings

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens.power import PowerScreen, run_power_sweep  # noqa: E402
from spacr.qt.widgets.power_design import DesignSpec  # noqa: E402

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

BACKGROUND_RANGE = "The background positive rate has to be in [0, 1]."


def test_design_spec_validate_names_every_problem_in_one_pass(subtests):
    """Each wrong field yields its sentence; a clean design yields none."""
    cases = [
        ("the default design is clean", {}, []),
        ("a positive rate above one skips its variance check",
         dict(background_positive_rate=1.5),
         [BACKGROUND_RANGE,
          "The hit-cell positive rate is 0.999, so its variance cannot reach "
          "0.0010 (the Bernoulli bound). 0.1 is too spread out — lower the "
          "variance or move the rate away from 0 and 1."]),
        ("a negative rate breaks the background and the hit rate",
         dict(background_positive_rate=-0.2),
         [BACKGROUND_RANGE,
          "The hit-cell positive rate has to be in [0, 1]."]),
        ("every problem at once",
         dict(n_genes=1, n_grnas_per_gene=0, cells_per_well=0.0,
              constructs_per_well=0.0, reads_per_well=-1.0),
         ["The library needs at least 2 genes.",
          "There has to be at least 1 guide per gene.",
          "Cells imaged per well has to be more than 0.",
          "Constructs per well has to be more than 0.",
          "A prevalence of 0.025 over 1 library units expects 0.03 hits. "
          "With no hits in the library AUROC is undefined and the sweep "
          "reports nothing, which is not the same finding as no power.",
          "Reads per well cannot be negative."]),
    ]
    for case_id, over, expected in cases:
        with subtests.test(msg=case_id):
            assert DesignSpec(**over).validate() == expected


def test_a_power_sweep_keeps_what_finished_before_a_stop(qtbot, subtests):
    """A stop after one point keeps it; an unstopped sweep scans both axes."""
    def stop_after_first(seen):
        cancel = threading.Event()

        def progress(done, total, label):
            seen.append((done, total, label))
            cancel.set()
        return {"cancel": cancel, "progress": progress}

    cases = [
        # id, stopped
        ("stopped after the first point", True),
        ("nobody stopped it", False),
    ]
    for case_id, stopped in cases:
        with subtests.test(msg=case_id):
            seen: list = []
            payload = {"spec": DesignSpec(**TINY),
                       "fit_kwargs": dict(THREAD_FIT)}
            if stopped:
                payload.update(stop_after_first(seen))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = run_power_sweep(payload)

            assert result["cancelled"] is stopped
            assert len(result["cells_scan"]) >= 1
            assert result["wells_scan"].empty is stopped
            assert list(result["wells_scan"].columns) == list(
                result["cells_scan"].columns)
            if stopped:
                assert seen, "the progress hook was never called"

            screen = PowerScreen(threaded=False)
            qtbot.addWidget(screen)
            screen._apply_result(result)
            said = screen.status_text()
            assert ("Stopped early" in said) is stopped
            if stopped:
                assert "ran out of power" in said
