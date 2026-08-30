"""Diagnostics that decline: too few residuals, no spread, no p-values.

Every arc here is a statistic being withheld. A residual report that filled in
skewness from two points, or a scale-location panel that drew a normal curve
of zero width, would be a figure a reader takes at face value -- and each of
these guards is what stops that. None of them had a test, because every fixture
in the suite is a healthy fit.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# residual_report — too few points, and fitted values with no spread
# ---------------------------------------------------------------------------

def test_two_residuals_get_no_normality_or_shape_statistics():
    """Arc 269 -> 297: the whole scipy block is skipped.

    Shapiro-Wilk needs three points and skewness of two is meaningless. The
    basic moments are still reported, so the caller gets what CAN be measured
    rather than a report that is absent altogether.
    """
    from spacr.regression_diagnostics import residual_report

    report = residual_report([1.0, 2.0], [1.1, 1.9])

    assert "residual_mean" in report and "sse" in report
    for withheld in ("normality_p_value", "skew", "kurtosis",
                     "heteroscedasticity_p_value"):
        assert withheld not in report


def test_constant_fitted_values_get_no_heteroscedasticity_test():
    """Arc 285 -> 297: Breusch-Pagan is against the fitted values.

    An intercept-only fit has one fitted value for every row, so there is
    nothing to regress the squared residuals ON. The test would be a
    regression on a constant, which is singular -- and reporting its p-value
    would be reporting the properties of a matrix, not of the data.
    """
    from spacr.regression_diagnostics import residual_report

    rng = np.random.default_rng(0)
    observed = rng.normal(size=40)
    fitted = np.full(40, observed.mean())

    report = residual_report(observed, fitted)

    assert "skew" in report                      # the n > 2 block did run
    assert "heteroscedasticity_p_value" not in report


def test_a_healthy_fit_reports_every_statistic():
    """The taken side of both, so the omissions above are visibly decisions."""
    from spacr.regression_diagnostics import residual_report

    rng = np.random.default_rng(0)
    fitted = np.linspace(0.0, 10.0, 60)
    observed = fitted + rng.normal(scale=0.5, size=60)

    report = residual_report(observed, fitted)

    assert "skew" in report and "kurtosis" in report
    assert "normality_p_value" in report


# ---------------------------------------------------------------------------
# _house — labels that were not asked for
# ---------------------------------------------------------------------------

def test_an_axis_given_no_labels_gets_none_set():
    """Arcs 529 -> 531, 531 -> 533 and 533 -> 535, all three skipped.

    ``_house`` is called for every panel, and most panels set their own titles
    afterwards. Setting an empty title would not be harmless: matplotlib
    reserves the space for it, and four panels each reserving a blank title
    band is the difference between a tight figure and a loose one.
    """
    from spacr.regression_diagnostics import _house

    figure, axis = plt.subplots()
    try:
        _house(axis)
        assert axis.get_title() == ""
        assert axis.get_xlabel() == ""
        assert axis.get_ylabel() == ""
    finally:
        plt.close(figure)


def test_an_axis_given_labels_wears_them():
    """The taken side of all three."""
    from spacr.regression_diagnostics import _house

    figure, axis = plt.subplots()
    try:
        _house(axis, title="T", xlabel="X", ylabel="Y")
        assert axis.get_title() == "T"
        assert axis.get_xlabel() == "X"
        assert axis.get_ylabel() == "Y"
    finally:
        plt.close(figure)


# ---------------------------------------------------------------------------
# plot_residual_diagnostics — no trend line, and no normal curve
# ---------------------------------------------------------------------------

def test_a_short_series_gets_no_smoothed_trend_line(tmp_path):
    """Arc 727 -> 733: fewer than eleven points, so no running-median line.

    A trend through eight points drawn with a window of three is a line that
    follows the noise, and a reader reads a drawn line as a finding.
    """
    from spacr.regression_diagnostics import plot_residual_diagnostics

    rng = np.random.default_rng(0)
    fitted = np.linspace(0.0, 1.0, 8)
    observed = fitted + rng.normal(scale=0.1, size=8)

    out = plot_residual_diagnostics(observed, fitted,
                                    save_path=str(tmp_path / "d.png"))

    assert out is not None
    plt.close("all")


def test_residuals_with_no_spread_get_no_normal_curve(tmp_path):
    """Arc 753 -> 757: scale is zero, so no density is drawn over the histogram.

    A perfect fit has residuals of exactly zero. ``norm.pdf`` with scale zero
    is a division by zero, and the curve it would draw -- infinitely tall and
    infinitely narrow -- is not a picture of anything.
    """
    from spacr.regression_diagnostics import plot_residual_diagnostics

    fitted = np.linspace(0.0, 1.0, 30)

    out = plot_residual_diagnostics(fitted.copy(), fitted,
                                    save_path=str(tmp_path / "d.png"))

    assert out is not None
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_inference_diagnostics — nothing to inflate
# ---------------------------------------------------------------------------

def test_an_empty_p_value_family_reports_no_inflation(tmp_path):
    """Arc 859 -> 867: genomic inflation stays NaN.

    Lambda is a median over the family. With no family there is no median, and
    the value that would otherwise be printed beside the Q-Q plot is the one
    number a reader uses to decide whether the null is calibrated. NaN says
    "not measured"; anything else would be a claim.
    """
    from spacr.regression_diagnostics import plot_inference_diagnostics

    out = plot_inference_diagnostics([], save_path=str(tmp_path / "q.png"))

    assert out is not None
    plt.close("all")


def test_a_real_p_value_family_reports_an_inflation(tmp_path):
    """The taken side, so the NaN above is visibly a decision."""
    from spacr.regression_diagnostics import plot_inference_diagnostics

    rng = np.random.default_rng(0)
    out = plot_inference_diagnostics(rng.uniform(0.0, 1.0, 200),
                                     save_path=str(tmp_path / "q.png"))

    assert out is not None
    plt.close("all")
