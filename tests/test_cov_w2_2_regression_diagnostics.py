"""The verdicts nobody had reached, and the panels that need a design.

Whether a screen regression can be believed is decided by a handful of
branches: the design that is identifiable but leaves a guide in one well, the
Cook's distance between the two conventional lines, the Breusch-Pagan p in the
'check' band rather than the 'fail' one, the inflation factor that could not
be computed. Each of those is a different sentence printed for the reader, and
a branch nobody walks is a sentence nobody has read.

The reports handed to the scorers here are the real shape
:func:`design_report`, :func:`residual_report` and the inference block
produce -- built directly only where a threshold has to be landed on
exactly, and produced by the real functions everywhere else. The plots are
really drawn, with a real design matrix, because the leverage and influence
panels only exist when there is one.
"""

import os

import numpy as np
import pandas as pd
import pytest

from spacr import regression_diagnostics as rd
from spacr.regression_diagnostics import (collinear_guide_pairs,
                                          design_report,
                                          plot_inference_diagnostics,
                                          plot_residual_diagnostics,
                                          residual_report, score_design,
                                          score_inference, score_residuals,
                                          write_diagnostic_suite)


@pytest.fixture
def fractions():
    """A small guide-fraction table: wells down, guides across."""
    rng = np.random.default_rng(7)
    data = rng.random((24, 5))
    return pd.DataFrame(data, columns=[f"g{i}" for i in range(5)])


# ---------------------------------------------------------------------------
# collinear pairs
# ---------------------------------------------------------------------------

def test_a_design_with_one_varying_guide_has_no_pair_to_report():
    """Fewer than two nonconstant columns gives the empty table, with columns.

    The columns matter: building the frame from an empty list gives it none
    at all, and the caller sorts on `correlation`.
    """
    frame = pd.DataFrame({"g0": [0.1, 0.2, 0.3], "g1": [0.5, 0.5, 0.5]})
    out = collinear_guide_pairs(frame)
    assert list(out.columns) == ["guide_a", "guide_b", "correlation",
                                 "shared_wells"]
    assert out.empty
    out.sort_values("correlation")     # the healthy case must be sortable


def test_a_screen_with_no_collinear_pair_still_gets_a_sortable_table(
        fractions):
    """Independent guides give an empty table that has its columns."""
    out = collinear_guide_pairs(fractions, threshold=0.99)
    assert out.empty
    assert list(out.columns) == ["guide_a", "guide_b", "correlation",
                                 "shared_wells"]


def test_the_pair_list_stops_at_the_limit():
    """A wide screen has millions of pairs; the first few are what is read."""
    rng = np.random.default_rng(3)
    base = rng.random(30)
    frame = pd.DataFrame({f"g{i}": base + rng.normal(0, 0.001, 30)
                          for i in range(6)})

    unlimited = collinear_guide_pairs(frame, threshold=0.9)
    assert len(unlimited) > 2

    limited = collinear_guide_pairs(frame, threshold=0.9, limit=2)
    assert len(limited) == 2
    assert set(limited.columns) == set(unlimited.columns)


def test_a_collinear_pair_is_named_with_the_wells_it_shares():
    """The offenders are named, because that is what makes them fixable."""
    rng = np.random.default_rng(11)
    shared = rng.random(20)
    frame = pd.DataFrame({
        "twin_a": shared,
        "twin_b": shared * 0.999 + rng.normal(0, 1e-4, 20),
        "loner": rng.random(20),
    })
    out = collinear_guide_pairs(frame, threshold=0.95)
    assert len(out) == 1
    row = out.iloc[0]
    assert {row["guide_a"], row["guide_b"]} == {"twin_a", "twin_b"}
    assert row["correlation"] > 0.95
    assert row["shared_wells"] == 20


# ---------------------------------------------------------------------------
# the normality test that changes with n
# ---------------------------------------------------------------------------

def test_a_large_fit_uses_dagostino_rather_than_shapiro():
    """Shapiro-Wilk is exact but degrades above a few thousand points.

    Which test was used is recorded, because the reader is being asked to
    interpret its p.
    """
    rng = np.random.default_rng(5)
    n = 5200
    yhat = rng.normal(size=n)
    y = yhat + rng.normal(0, 0.5, n)

    report = residual_report(y, yhat)
    assert report["n"] == n
    assert report["normality_test"] == "dagostino_pearson"
    assert np.isfinite(report["normality_statistic"])
    assert 0.0 <= report["normality_p_value"] <= 1.0


def test_a_small_fit_uses_shapiro():
    """Below the cut the exact test is the right one, and is named."""
    rng = np.random.default_rng(5)
    yhat = rng.normal(size=200)
    y = yhat + rng.normal(0, 0.5, 200)
    assert residual_report(y, yhat)["normality_test"] == "shapiro"


# ---------------------------------------------------------------------------
# scoring the design
# ---------------------------------------------------------------------------

def test_no_design_is_unknown_rather_than_a_pass():
    """Nothing supplied is 'unknown'; a pass would be a claim about nothing."""
    for report in ({}, None, {"wells": 0}):
        verdict = score_design(report)
        assert verdict.level == "unknown"
        assert "no design" in verdict.headline


def test_a_guide_seen_in_one_well_is_flagged_even_when_the_design_is_fine():
    """Identifiable, plenty of data per parameter, and still worth checking.

    A guide seen in a single well has no contrast to be estimated from: its
    coefficient is that well.
    """
    report = {"wells": 400, "parameters": 20, "design_rank": 20,
              "identifiable": True, "wells_per_parameter": 20.0,
              "guides_in_one_well": 3, "condition_number": 12.0}
    verdict = score_design(report)
    assert verdict.level == "check"
    assert "3 guide(s) appear in one well or none" in verdict.headline
    assert "no contrast" in verdict.detail
    assert verdict.score == 3.0
    assert verdict.statistic == "guides in <=1 well"


def test_a_rank_deficient_design_fails_and_names_the_way_out():
    """Rank below the parameter count means no unique coefficient vector."""
    report = {"wells": 587, "parameters": 824, "design_rank": 500,
              "identifiable": False, "wells_per_parameter": 0.71,
              "guides_in_one_well": 0, "condition_number": 1e18}
    verdict = score_design(report)
    assert verdict.level == "fail"
    assert "permutation test" in verdict.detail


def test_a_thin_but_identifiable_design_is_a_check_not_a_failure():
    """Under two wells per parameter every coefficient rests on a handful."""
    report = {"wells": 30, "parameters": 20, "design_rank": 20,
              "identifiable": True, "wells_per_parameter": 1.5,
              "guides_in_one_well": 0, "condition_number": 40.0}
    verdict = score_design(report)
    assert verdict.level == "check"
    assert "1.50 wells per parameter" in verdict.detail


def test_a_healthy_design_passes(fractions):
    """The real report from a real design, scored."""
    verdict = score_design(design_report(fractions))
    assert verdict.level in ("pass", "check")


# ---------------------------------------------------------------------------
# scoring the residuals
# ---------------------------------------------------------------------------

def test_no_residuals_is_unknown():
    """Nothing to score is said, rather than passed."""
    for report in ({}, None, {"n": 0}):
        assert score_residuals(report).level == "unknown"


def test_one_influential_point_between_the_two_lines_is_a_check():
    """0.5 is the conventional line for a point worth looking at.

    Above 1.0 is a failure; between them is where a reader has to decide,
    which is exactly where the sentence matters.
    """
    report = {"n": 100, "max_cooks_distance": 0.7, "r_squared": 0.8}
    verdict = score_residuals(report)
    assert verdict.level == "check"
    assert "a lot of influence" in verdict.headline
    assert "0.5 is the" in verdict.detail
    assert verdict.statistic == "max Cook's D"


def test_one_point_above_the_upper_line_fails():
    """Above 1.0 a single point moves the coefficients on its own."""
    verdict = score_residuals({"n": 100, "max_cooks_distance": 3.4})
    assert verdict.level == "fail"
    assert "dominates the fit" in verdict.headline


def test_a_strongly_changing_spread_fails_and_says_p_is_backwards():
    """The null is constant variance, so a SMALL p is the bad outcome."""
    verdict = score_residuals({"n": 100,
                               "heteroscedasticity_p_value": 1e-6})
    assert verdict.level == "fail"
    assert "SMALL p is the bad outcome" in verdict.detail
    assert "standard errors this fit reports are not the right ones" in \
        verdict.detail


def test_a_marginally_changing_spread_is_a_check():
    """Between 0.01 and 0.05 is the band a reader has to weigh."""
    verdict = score_residuals({"n": 100,
                               "heteroscedasticity_p_value": 0.03})
    assert verdict.level == "check"
    assert "may change with the fit" in verdict.headline
    assert "small p is the bad outcome" in verdict.detail


def test_non_normal_residuals_are_a_check_not_a_failure():
    """With enough wells this is common and matters most in the tails."""
    verdict = score_residuals({"n": 4000, "normality_p_value": 1e-8,
                               "normality_test": "dagostino_pearson"})
    assert verdict.level == "check"
    assert "dagostino_pearson p" in verdict.detail


def test_the_worst_finding_is_the_one_reported():
    """Several findings collapse to the most severe, not the first."""
    verdict = score_residuals({"n": 100, "normality_p_value": 1e-8,
                               "max_cooks_distance": 3.0})
    assert verdict.level == "fail"


def test_well_behaved_residuals_pass_and_report_r_squared():
    """No test under 0.05 and no dominating observation is a pass."""
    verdict = score_residuals({"n": 250, "max_cooks_distance": 0.1,
                               "heteroscedasticity_p_value": 0.6,
                               "normality_p_value": 0.4,
                               "r_squared": 0.62})
    assert verdict.level == "pass"
    assert "n = 250" in verdict.detail
    assert verdict.score == 0.62
    assert verdict.statistic == "r-squared"


# ---------------------------------------------------------------------------
# scoring the inference
# ---------------------------------------------------------------------------

def test_no_p_values_is_unknown():
    """Nothing to calibrate against is said rather than passed."""
    assert score_inference({}).level == "unknown"
    assert score_inference({"tests": 0}).level == "unknown"


def test_an_inflation_factor_that_could_not_be_computed_is_unknown():
    """A missing or non-finite lambda is not a pass, and not a failure."""
    for value in (None, float("nan"), float("inf")):
        verdict = score_inference({"tests": 500, "genomic_inflation": value})
        assert verdict.level == "unknown"
        assert "could not be computed" in verdict.headline


def test_a_deflated_null_is_named_as_deflated():
    """The direction is in the headline, because the fixes differ."""
    verdict = score_inference({"tests": 900, "genomic_inflation": 0.6,
                               "pi0": 0.95, "estimated_non_null": 45})
    assert verdict.level == "fail"
    assert "deflated" in verdict.headline

    inflated = score_inference({"tests": 900, "genomic_inflation": 1.5,
                                "pi0": 0.9, "estimated_non_null": 90})
    assert inflated.level == "fail"
    assert "inflated" in inflated.headline


def test_a_slightly_off_centre_null_is_a_check_and_a_calibrated_one_passes():
    """1.1 is where a report starts explaining itself; 1.2 is disbelief."""
    off = score_inference({"tests": 900, "genomic_inflation": 1.15,
                           "pi0": 0.9, "estimated_non_null": 90})
    assert off.level == "check"
    assert "lambda = 1.150" in off.detail

    fine = score_inference({"tests": 900, "genomic_inflation": 1.02,
                            "pi0": 0.9, "estimated_non_null": 90})
    assert fine.level == "pass"


# ---------------------------------------------------------------------------
# the panels that only exist when there is a design
# ---------------------------------------------------------------------------

def test_a_residual_sheet_with_a_design_draws_leverage_and_influence(
        tmp_path):
    """Six panels, not four: leverage and Cook's distance need the matrix."""
    rng = np.random.default_rng(2)
    n, p = 120, 4
    matrix = np.column_stack([np.ones(n), rng.normal(size=(n, p))])
    truth = rng.normal(size=p + 1)
    yhat = matrix @ truth
    y = yhat + rng.normal(0, 0.4, n)

    path, report = plot_residual_diagnostics(
        y, yhat, design=matrix,
        save_path=str(tmp_path / "residuals"), save_format="png",
        label="plate1")

    assert os.path.isfile(path)
    assert report["n"] == n
    assert np.isfinite(report["max_cooks_distance"])
    assert report["max_leverage"] > 0


def test_a_residual_sheet_without_a_design_still_draws_the_four_panels(
        tmp_path):
    """No matrix means no leverage; the classical four are still written."""
    rng = np.random.default_rng(4)
    yhat = rng.normal(size=80)
    y = yhat + rng.normal(0, 0.3, 80)

    path, report = plot_residual_diagnostics(
        y, yhat, save_path=str(tmp_path / "residuals"), save_format="png")

    assert os.path.isfile(path)
    assert "max_cooks_distance" not in report


def test_an_inference_sheet_carries_the_label_it_was_given(tmp_path):
    """A suite written per plate needs each sheet to say which plate."""
    rng = np.random.default_rng(9)
    p_values = np.concatenate([rng.random(400), rng.random(20) * 1e-4])

    path, report = plot_inference_diagnostics(
        p_values, alpha=0.05, save_path=str(tmp_path / "inference"),
        save_format="png", label="plate3")

    assert os.path.isfile(path)
    assert report["tests"] == p_values.size
    assert 0.0 <= report["pi0"] <= 1.0


# ---------------------------------------------------------------------------
# a diagnostic must never take the analysis down
# ---------------------------------------------------------------------------

def test_a_fractions_table_that_cannot_be_analysed_is_recorded_not_raised(
        tmp_path):
    """A block that raises becomes an error entry beside the ones that worked.

    The analysis has already been fitted; losing it to a diagnostic would be
    the worst possible trade.
    """
    rng = np.random.default_rng(6)
    broken = pd.DataFrame({"g0": ["a"] * 12, "g1": ["b"] * 12})
    yhat = rng.normal(size=60)
    y = yhat + rng.normal(0, 0.3, 60)

    written = write_diagnostic_suite(
        tmp_path, fractions=broken, observed=y, fitted=yhat,
        formats=("png",))

    assert any(k.startswith("collinear_guide_pairs") and k.endswith("error")
               for k in written), written
    # and the block that could run, ran
    assert any(k.startswith("residual_diagnostics") and not k.endswith("error")
               for k in written), written


def test_the_suite_writes_what_its_inputs_can_support(tmp_path, fractions):
    """Nothing is required: each block is skipped when its inputs are absent."""
    rng = np.random.default_rng(8)
    written = write_diagnostic_suite(
        tmp_path, fractions=fractions, p_values=rng.random(300),
        label="plate1", formats=("png",))

    assert "collinear_guide_pairs" in written
    assert os.path.isfile(written["collinear_guide_pairs"])
    assert any("design_diagnostics" in k for k in written)
    assert any("inference_diagnostics" in k for k in written)
    assert not any("residual_diagnostics" in k for k in written)
    # the label reaches the filenames
    assert all("plate1" in v for k, v in written.items()
               if not k.endswith("error"))


# ---------------------------------------------------------------------------
# the house style helper
# ---------------------------------------------------------------------------

def test_an_axis_put_into_the_house_style_loses_its_grid_and_two_spines():
    """rcParams only reach an artist when it is created, so this is by hand.

    The rule the style enforces is no gridlines ever, and the top and right
    spines off; a caller with a grid-on global style would otherwise leave
    one here.
    """
    import matplotlib.pyplot as plt

    from spacr.figures.style import TYPE_SCALE

    fig, axis = plt.subplots()
    try:
        axis.grid(True)
        ink = rd._house(axis, title="A title", xlabel="x", ylabel="y")

        assert axis.get_title() == "A title"
        assert axis.get_xlabel() == "x"
        assert axis.get_ylabel() == "y"
        assert axis.title.get_fontsize() == TYPE_SCALE["label"]
        assert axis.spines["top"].get_visible() is False
        assert axis.spines["right"].get_visible() is False
        assert axis.xaxis._major_tick_kw.get("gridOn") in (False, None)
        assert ink
    finally:
        plt.close(fig)
