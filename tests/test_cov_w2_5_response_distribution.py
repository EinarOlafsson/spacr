"""The response-shape panel answers even when its statistics do not.

Every part of this module is diagnostic: it exists to tell the user what the
transform did to the response. So a failure anywhere inside it — scipy, the
family classifier, or the transformer itself — has to degrade to a plainly
worded answer rather than to a traceback, and the tests below inject each
failure at the call that produces it.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from spacr import response_distribution as rd


@pytest.fixture
def proportions():
    """A skewed, strictly-inside-(0, 1) response of realistic size."""
    rng = np.random.default_rng(0)
    return np.clip(rng.beta(2.0, 8.0, size=200), 1e-4, 1 - 1e-4)


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------

def test_too_few_values_are_named_as_such_not_classified():
    """Under eight observations there is no family and no statistics."""
    out = rd.describe([0.1, 0.2, 0.3])

    assert out["n"] == 3
    assert out["family"] == ""
    assert out["name"] == "too few values to name"
    assert np.isnan(out["low"]) and np.isnan(out["high"])
    assert np.isnan(out["skew"]) and np.isnan(out["normality_p"])


def test_non_finite_values_are_excluded_from_the_count():
    """NaN and inf never reach the statistics."""
    out = rd.describe([0.1, np.nan, np.inf, -np.inf, 0.9])

    assert out["n"] == 2


def test_a_real_response_gets_a_family_a_range_and_a_p_value(proportions):
    """The panel's numbers come from the same test that chose the family."""
    out = rd.describe(proportions)

    assert out["n"] == proportions.size
    assert out["low"] == pytest.approx(float(proportions.min()))
    assert out["high"] == pytest.approx(float(proportions.max()))
    assert np.isfinite(out["skew"]) and out["skew"] > 0
    assert 0.0 <= out["normality_p"] <= 1.0
    assert out["family"]
    assert out["name"] == rd.FAMILY_NAMES.get(out["family"], out["family"])


def test_an_unknown_family_is_shown_under_its_own_name(monkeypatch,
                                                       proportions):
    """A family with no friendly label is displayed verbatim."""
    from spacr import ml

    monkeypatch.setattr(ml, "check_distribution", lambda data: "poisson_ish")

    out = rd.describe(proportions)

    assert out["family"] == "poisson_ish"
    assert out["name"] == "poisson_ish"


def test_a_broken_shape_measurement_still_leaves_a_family(monkeypatch, caplog,
                                                          proportions):
    """scipy failing costs the skew and the p-value, not the classification."""
    import scipy.stats

    monkeypatch.setattr(scipy.stats, "normaltest",
                        lambda data: (_ for _ in ()).throw(
                            RuntimeError("no shape today")))

    with caplog.at_level(logging.DEBUG, logger="spacr.response_distribution"):
        out = rd.describe(proportions)

    assert np.isnan(out["normality_p"])
    assert out["family"], "the family survives a scipy failure"
    assert "could not measure the response's shape" in caplog.text


def test_a_classifier_that_raises_leaves_the_summary_unnamed(monkeypatch,
                                                             caplog,
                                                             proportions):
    """The range still reports; the family stays empty."""
    from spacr import ml

    def explode(data):
        raise ValueError("cannot classify")

    monkeypatch.setattr(ml, "check_distribution", explode)

    with caplog.at_level(logging.DEBUG, logger="spacr.response_distribution"):
        out = rd.describe(proportions)

    assert out["family"] == ""
    assert out["name"] == "too few values to name"
    assert np.isfinite(out["low"])
    assert "could not classify the response" in caplog.text


def test_the_classifiers_own_printing_never_reaches_the_caller(capsys,
                                                               proportions):
    """``check_distribution`` prints its reasoning; the panel swallows it."""
    rd.describe(proportions)

    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# transformed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["", "   ", "none", "NONE", None])
def test_no_transform_returns_the_response_untouched(name, proportions):
    """An absent or ``none`` transform is the identity."""
    out = rd.transformed(proportions, name)

    assert np.array_equal(out, np.asarray(proportions, dtype=float))


def test_a_real_transform_actually_moves_the_values(proportions):
    """``log`` produces different numbers of the same length."""
    out = rd.transformed(proportions, "log")

    assert out.shape == proportions.shape
    assert not np.allclose(out, proportions)


def test_a_transformer_that_cannot_be_built_returns_the_input(monkeypatch,
                                                              caplog,
                                                              proportions):
    """A failure inside ``apply_transformation`` is not fatal."""
    from spacr import ml

    def explode(data, name):
        raise RuntimeError("no transformer")

    monkeypatch.setattr(ml, "apply_transformation", explode)

    with caplog.at_level(logging.DEBUG, logger="spacr.response_distribution"):
        out = rd.transformed(proportions, "log")

    assert np.array_equal(out, proportions)
    assert "could not build the transformer" in caplog.text


def test_an_unsupported_transform_returns_the_input(monkeypatch, proportions):
    """``apply_transformation`` answering ``None`` means leave it alone."""
    from spacr import ml

    monkeypatch.setattr(ml, "apply_transformation", lambda data, name: None)

    assert np.array_equal(rd.transformed(proportions, "log"), proportions)


def test_a_transform_that_fails_on_this_response_returns_the_input(
        monkeypatch, caplog, proportions):
    """``fit_transform`` raising leaves the untransformed values in place."""
    from spacr import ml

    class Refuses:
        def fit_transform(self, data):
            raise ValueError("negative values")

    monkeypatch.setattr(ml, "apply_transformation",
                        lambda data, name: Refuses())

    with caplog.at_level(logging.DEBUG, logger="spacr.response_distribution"):
        out = rd.transformed(proportions, "log")

    assert np.array_equal(out, proportions)
    assert "the transform failed on this response" in caplog.text


# ---------------------------------------------------------------------------
# compare and caption
# ---------------------------------------------------------------------------

def test_a_transform_that_changed_nothing_says_so(proportions):
    """``changed`` is false and the caption admits the two plots are one."""
    result = rd.compare(proportions, "none")

    assert result["transform"] == "none"
    assert result["changed"] is False
    assert result["rescaled"] is False
    text = rd.caption(result)
    assert "changed nothing" in text
    assert "same data, drawn" in text


def test_a_rescaling_transform_is_flagged_for_its_own_axis(proportions):
    """``log`` changes the values and needs a second horizontal axis."""
    result = rd.compare(proportions, "log")

    assert result["changed"] is True
    assert result["rescaled"] is True
    assert result["values_after"].shape == result["values_before"].shape
    text = rd.caption(result)
    assert text.startswith("before:")
    assert "after log" in text


def test_the_caption_carries_the_statistics_that_named_the_family(
        proportions):
    """A named family brings its p-value and skew onto the panel."""
    text = rd.caption(rd.compare(proportions, "log"))

    assert "D'Agostino p =" in text
    assert "skew " in text


def test_an_unnamed_summary_contributes_only_its_name():
    """Too few values means the caption says so and adds no statistics."""
    text = rd.caption(rd.compare([0.1, 0.2, 0.3], "none"))

    assert "too few values to name" in text
    assert "D'Agostino" not in text


def test_a_family_without_measurable_statistics_shows_only_its_label():
    """A finite family with NaN statistics prints no empty parentheses."""
    part = {"family": "ols", "name": "normal",
            "normality_p": float("nan"), "skew": float("nan")}
    result = {"changed": True, "transform": "none",
              "before": part, "after": part}

    text = rd.caption(result)

    assert "normal" in text
    assert "(" not in text


# ---------------------------------------------------------------------------
# fast_panel
# ---------------------------------------------------------------------------

def test_nothing_finite_draws_nothing():
    """An all-NaN response returns None rather than an empty plot."""
    assert rd.fast_panel([np.nan, np.inf, -np.inf], "none") is None


def test_the_fast_panel_draws_both_outlines_and_captions_them(qapp,
                                                              proportions):
    """Two step outlines land on one pair of axes with the caption below."""
    from spacr.qt.widgets.fast_plots import FastPlot

    plot = FastPlot(title="t", x_label="x", y_label="y")
    before = len(plot.plot.listDataItems())

    got = rd.fast_panel(proportions, "log", plot=plot,
                        dependent_variable="fraction infected")

    assert got is plot
    drawn = plot.plot.listDataItems()[before:]
    assert len(drawn) == 2, "one outline for before, one for after"
    for item in drawn:
        xs, ys = item.getData()
        assert xs.size == ys.size and xs.size > 2
        assert np.all(ys >= 0)
    plot.deleteLater()


def test_the_fast_panel_makes_its_own_plot_when_given_none(qapp,
                                                           proportions):
    """Omitting the plot creates one labelled with the response's name."""
    plot = rd.fast_panel(proportions, "none",
                         dependent_variable="fraction infected")

    assert plot is not None
    assert len(plot.plot.listDataItems()) == 2
    plot.deleteLater()


def test_an_empty_transformed_series_is_skipped_not_histogrammed(
        monkeypatch, qapp, proportions):
    """A transformer that returns nothing costs the second outline only."""
    from spacr import ml
    from spacr.qt.widgets.fast_plots import FastPlot

    class Empties:
        def fit_transform(self, data):
            return np.empty((0, 1))

    monkeypatch.setattr(ml, "apply_transformation", lambda data, name: Empties())
    plot = FastPlot(title="t", x_label="x", y_label="y")

    got = rd.fast_panel(proportions, "log", plot=plot)

    assert got is plot
    assert len(plot.plot.listDataItems()) == 1
    plot.deleteLater()


def test_the_pen_is_two_pixels_wide(qapp):
    """The outline is drawn heavy enough to read over the other one."""
    pen = rd._pen("#4C72B0", "before")

    assert pen.widthF() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# panel (matplotlib)
# ---------------------------------------------------------------------------

def test_the_matplotlib_panel_shares_one_axis_when_nothing_was_rescaled(
        proportions):
    """An unchanged response gets one axis, one bin edge set, one legend."""
    from matplotlib.figure import Figure

    ax = Figure().add_subplot(111)

    result = rd.panel(proportions, "none", ax=ax, dependent_variable="score")

    assert result["axes"] is ax
    assert ax.get_xlabel() == "score"
    assert ax.get_ylabel() == "wells"
    assert ax.get_legend() is not None
    assert not ax.figure.axes[1:], "no twin axis was needed"


def test_the_matplotlib_panel_twins_the_axis_for_a_rescaling_transform(
        proportions):
    """``log`` gets its own horizontal axis and a two-entry legend."""
    result = rd.panel(proportions, "log", dependent_variable="score")

    ax = result["axes"]
    twins = [other for other in ax.figure.axes if other is not ax]
    assert twins, "the transformed values need their own scale"
    assert twins[0].get_xlabel() == "after log"
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["before", "after log"]


def test_the_panel_labels_the_response_generically_when_unnamed(proportions):
    """With no dependent variable the axis still says what it holds."""
    result = rd.panel(proportions, "none")

    assert result["axes"].get_xlabel() == "response"
    assert "before and after" in result["axes"].get_title()
