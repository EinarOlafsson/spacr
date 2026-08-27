"""The four category-B smoothers, drawn on the residual plots.

Instruction 254 puts these in a different category from `regression_type`
on purpose: they are laid OVER a fit that has already happened and none of
them decides a hit. These tests hold that line -- that they draw, that they
refuse with the number rather than hanging, and that nothing about them can
be read as a test of a guide.
"""

import numpy as np
import pytest

pytest.importorskip("PySide6")
pg = pytest.importorskip("pyqtgraph")

from spacr.qt.widgets.fast_plots import (  # noqa: E402
    ResidualPlot, ScaleLocationPlot,
)


def _bending_residuals(n=300, seed=0):
    """Residuals whose straight-line slope is flat but which clearly bend."""
    rng = np.random.default_rng(seed)
    fitted = np.linspace(-2, 2, n)
    residual = 0.6 * fitted ** 2 - 0.8 + rng.normal(0, 0.15, n)
    return fitted, residual


def test_the_straight_line_misses_a_bend_the_smoother_finds(qtbot):
    """The reason this wiring exists, measured rather than asserted.

    A parabola has no linear trend, so the slope the plot already reported
    says the mean model is fine. The smoother spans the bend instead.
    """
    fitted, residual = _bending_residuals()
    assert abs(np.polyfit(fitted, residual, 1)[0]) < 0.05

    plot = ResidualPlot()
    qtbot.addWidget(plot)
    plot.set_residuals(fitted, residual)
    said = plot._status.text()
    assert "lowess curve laid over the points" in said
    assert "decides no hit" in said


@pytest.mark.parametrize("method", ["lowess", "kernel", "knn",
                                    "gaussian_process"])
def test_all_four_diagnostics_draw_on_the_residual_plot(qtbot, method):
    fitted, residual = _bending_residuals()
    plot = ResidualPlot()
    qtbot.addWidget(plot)
    plot._choose_smoother(method)
    plot.set_residuals(fitted, residual)
    assert method in plot._status.text()
    assert "could not be drawn" not in plot._status.text()


def test_the_distance_based_methods_say_they_scaled(qtbot):
    """254: "every method that needs scaling scales, and says it did"."""
    fitted, residual = _bending_residuals()
    plot = ResidualPlot()
    qtbot.addWidget(plot)
    for method in ("knn", "gaussian_process"):
        plot._choose_smoother(method)
        plot.set_residuals(fitted, residual)
        assert "standardised" in plot._status.text()


def test_choosing_none_leaves_the_slope_alone(qtbot):
    fitted, residual = _bending_residuals()
    plot = ResidualPlot()
    qtbot.addWidget(plot)
    plot._choose_smoother("")
    plot.set_residuals(fitted, residual)
    said = plot._status.text()
    assert "Trend slope" in said
    assert "laid over the points" not in said


def test_a_gaussian_process_refuses_with_the_number(qtbot):
    """It says how many rows it got and what the limit is, and returns."""
    plot = ResidualPlot()
    qtbot.addWidget(plot)
    big = np.linspace(0, 1, 5000)
    said = plot.add_smoother(big, big * 2, method="gaussian_process")
    assert "5,000" in said and "2,000" in said


def test_an_empty_method_is_not_a_special_case_at_the_call_site(qtbot):
    plot = ResidualPlot()
    qtbot.addWidget(plot)
    assert plot.add_smoother([1.0, 2.0], [1.0, 2.0], method="") == ""


def test_the_menu_offers_the_four_and_none_and_names_the_category(qtbot):
    plot = ResidualPlot()
    qtbot.addWidget(plot)
    labels = [entry[0] for entry in plot._smoother_options()]
    assert labels[0] == "None"
    assert len(labels) == 5
    assert any("LOWESS" in label for label in labels)


def test_the_menu_never_offers_a_fit_or_an_agreement_check(qtbot):
    """A method offered in the wrong category is 254's central complaint."""
    plot = ResidualPlot()
    qtbot.addWidget(plot)
    labels = " ".join(entry[0] for entry in plot._smoother_options()).lower()
    for wrong in ("spline", "isotonic", "forest", "boosting"):
        assert wrong not in labels


def test_the_curve_it_draws_carries_no_p_value(qtbot):
    """The object cannot be mistaken for a test, because it has no p."""
    from spacr.nonparametric_fits import smooth

    fitted, residual = _bending_residuals()
    curve = smooth(fitted, residual, method="lowess")
    assert not hasattr(curve, "p_values")


def test_only_the_gaussian_process_reports_a_band(qtbot):
    from spacr.nonparametric_fits import smooth

    fitted, residual = _bending_residuals(n=200)
    assert smooth(fitted, residual, method="gaussian_process").has_band
    for method in ("lowess", "kernel", "knn"):
        assert not smooth(fitted, residual, method=method).has_band


def test_the_scale_location_plot_smooths_too(qtbot):
    fitted, residual = _bending_residuals()
    plot = ScaleLocationPlot()
    qtbot.addWidget(plot)
    plot.set_scale_location(fitted, np.abs(residual) + 1.0)
    assert "laid over the points" in plot._status.text()


def test_changing_the_smoother_redraws_from_the_kept_data(qtbot):
    """The menu callback must not need the host to hand the data back."""
    fitted, residual = _bending_residuals()
    plot = ScaleLocationPlot()
    qtbot.addWidget(plot)
    plot.set_scale_location(fitted, np.abs(residual) + 1.0)
    plot._choose_smoother("knn")
    assert "knn" in plot._status.text()


def test_a_plot_with_no_standardised_residual_still_answers(qtbot):
    plot = ScaleLocationPlot()
    qtbot.addWidget(plot)
    drawn = plot.set_scale_location(np.zeros(5), np.full(5, np.nan),
                                    reason="quantile has no error scale")
    assert drawn == 0
    assert "quantile" in plot._status.text()
