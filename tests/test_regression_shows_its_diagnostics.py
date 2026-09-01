"""Instruction 322: the diagnostics are computed, and say why when they cannot be.

`spacr.regression_diagnostics` has computed QQ, residual-vs-fitted,
scale-location, leverage, Cook's distance, design rank, condition number,
collinear pairs, VIFs and genomic inflation since it was written -- and
nothing called it. The checks that would have caught the failure it was
written for were unreachable by a user.

That failure is the first test here: 824 guides in 587 wells, fitted
simultaneously, returning a confident coefficient and P value for every guide
out of a rank-deficient matrix. It must produce FAIL.

The tests assert NUMBERS rather than that a figure appeared, because the
module was deliberately written to return plain numbers and that is the
affordance to use -- a test that only checks a PNG exists passes just as well
when every value in it is wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import regression_diagnostics as rd


def _wells_by_guides(n_wells, n_guides, guides_per_well=4, seed=0,
                    constant_row_sum=False):
    """A well-by-guide abundance matrix, the shape a pooled screen produces.

    ROW SUMS VARY BY DEFAULT, and that is not cosmetic. A matrix whose every
    row sums to the same number is collinear with the intercept
    ``design_report`` prepends -- the dummy-variable trap -- so it is
    rank-deficient however many wells it has. The first version of this
    fixture normalised each well to 1.0 and made the WELL-POWERED control fail
    too, which is the fixture being wrong rather than the report.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(0.0,
                         index=[f"w{i}" for i in range(n_wells)],
                         columns=[f"g{j}" for j in range(n_guides)])
    for well in frame.index:
        picked = rng.choice(n_guides, size=guides_per_well, replace=False)
        weights = rng.uniform(0.5, 1.5, size=guides_per_well)
        if constant_row_sum:
            weights = weights / weights.sum()
        frame.loc[well, [f"g{j}" for j in picked]] = weights
    return frame


def test_824_guides_in_587_wells_is_a_FAIL():
    """The failure the module was written for, as a regression test.

    More parameters than observations: the design cannot identify them, so
    every coefficient it reports is one of infinitely many solutions. The fit
    still returns a confident P value for each, which is why this has to be
    caught by the DESIGN rather than by looking at the output.
    """
    fractions = _wells_by_guides(587, 824)
    report = rd.design_report(fractions)

    assert report["identifiable"] is False, (
        "824 parameters from 587 wells cannot be identified")
    verdict = rd.score_design(report)
    assert str(verdict.level).upper() == "FAIL", (
        f"a rank-deficient design must FAIL, got {verdict.level}: "
        f"{verdict.headline}")


def test_a_well_powered_design_is_not_failed():
    """The other side, so the test above is not just asserting pessimism.

    Without this, a score_design that returned FAIL unconditionally would pass
    the test above and be useless.
    """
    fractions = _wells_by_guides(600, 40)
    report = rd.design_report(fractions)

    assert report["identifiable"] is True
    assert str(rd.score_design(report).level).upper() in ("PASS", "WARN"), (
        "a design with 15 wells per parameter must not FAIL")


def test_a_constant_row_sum_is_rank_deficient_however_many_wells():
    """The trap the fixture fell into, pinned because it is a real one.

    Fractions normalised per well sum to 1.0 in every row, so their columns
    span the intercept and the design loses a degree of freedom no number of
    wells restores. A screen whose counts are normalised before fitting is in
    this state, and the report is right to say so.
    """
    normalised = _wells_by_guides(600, 40, constant_row_sum=True)
    assert rd.design_report(normalised)["identifiable"] is False


def test_the_design_report_needs_no_fit_at_all():
    """Which is why it is unconditional in ml.perform_regression.

    RRA never forms a linear predictor, so it has no residuals -- but it has a
    design, and the design is where the rank-deficiency lives.
    """
    fractions = _wells_by_guides(100, 20)
    report = rd.design_report(fractions)
    assert "identifiable" in report and "condition_number" in report


def test_a_model_without_residuals_is_named_rather_than_silent():
    """322's actual content: "whenever possible" is about the other case.

    A missing QQ plot and an inapplicable one look identical to a reader, and
    only one of them is fine.
    """
    from spacr.ml import RESIDUAL_FREE_MODELS

    assert "rra" in RESIDUAL_FREE_MODELS
    reason = RESIDUAL_FREE_MODELS["rra"]
    assert "rank" in reason.lower(), reason
    assert len(reason) > 40, "the reason has to be a sentence, not a label"


def test_diagnostic_inputs_are_read_off_a_fitted_model():
    """The duck-typed read, driven with an object shaped like statsmodels."""
    from spacr.ml import _diagnostic_inputs

    class FakeModel:
        fittedvalues = np.array([1.0, 2.0, 3.0])
        resid = np.array([0.5, -0.5, 0.0])

        class model:
            exog = np.eye(3)

    observed, fitted, design = _diagnostic_inputs(FakeModel())
    assert observed == pytest.approx([1.5, 1.5, 3.0]), (
        "observed is fitted + residual")
    assert fitted == pytest.approx([1.0, 2.0, 3.0])
    assert design.shape == (3, 3)


def test_a_model_exposing_nothing_yields_no_inputs_rather_than_raising():
    """RRA's shape. It must return None, not explode the run."""
    from spacr.ml import _diagnostic_inputs

    class RankOnly:
        pass

    assert _diagnostic_inputs(RankOnly()) == (None, None, None)


def test_malformed_model_arrays_yield_no_diagnostic_inputs():
    """A backend's non-numeric diagnostics must not take down fitted results."""
    from spacr.ml import _diagnostic_inputs

    class MalformedModel:
        fittedvalues = ["not-a-number"]
        resid = [0.5]

    assert _diagnostic_inputs(MalformedModel()) == (None, None, None)


def test_a_blank_results_folder_skips_regression_diagnostics():
    from spacr.ml import _write_regression_diagnostics

    assert _write_regression_diagnostics(None, pd.DataFrame(), {}, {}) == {}


def test_empty_fits_still_write_the_design_and_residual_reason(
        tmp_path, monkeypatch):
    from spacr.ml import _write_regression_diagnostics

    (tmp_path / "diagnostics").mkdir()
    monkeypatch.setattr(
        rd,
        "write_diagnostic_suite",
        lambda destination, **_kwargs: {
            "design": str(tmp_path / "design.json"),
        },
    )

    written = _write_regression_diagnostics(
        str(tmp_path),
        pd.DataFrame({"g1": [0.2]}),
        {},
        {"regression_type": "rra"},
    )

    note = tmp_path / "diagnostics" / "residual_panels_not_available.txt"
    assert written["design"].endswith("design.json")
    assert written["residuals_unavailable"] == str(note)
    assert "Robust Rank Aggregation" in note.read_text(encoding="utf-8")


def test_a_diagnostic_writer_failure_is_reported_without_losing_results(
        tmp_path, monkeypatch, capsys):
    from spacr.ml import _write_regression_diagnostics

    def fail(*_args, **_kwargs):
        raise RuntimeError("panel failed")

    monkeypatch.setattr(rd, "write_diagnostic_suite", fail)

    assert _write_regression_diagnostics(
        str(tmp_path), pd.DataFrame(), {}, {"regression_type": "ols"},
    ) == {}
    assert "Diagnostics could not be written: RuntimeError: panel failed" in (
        capsys.readouterr().out
    )


def test_an_unwritable_residual_note_does_not_erase_other_diagnostics(
        tmp_path, monkeypatch, capsys):
    from spacr.ml import _write_regression_diagnostics

    monkeypatch.setattr(
        rd,
        "write_diagnostic_suite",
        lambda *_args, **_kwargs: {"design": "design.json"},
    )

    def refuse_note(*_args, **_kwargs):
        raise OSError("read only")

    monkeypatch.setattr("builtins.open", refuse_note)
    written = _write_regression_diagnostics(
        str(tmp_path),
        pd.DataFrame({"g1": [0.2]}),
        {"gene": (object(), pd.DataFrame(), "rra")},
        {},
    )

    assert written == {"design": "design.json"}
    assert "Residual diagnostics were not computed" in capsys.readouterr().out
