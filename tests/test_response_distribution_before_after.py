"""The response-shape panel answers even when its statistics do not.

:mod:`spacr.response_distribution` tells the user what a transform did to the
regression response. A failure anywhere inside it -- scipy, the family
classifier, the transformer -- degrades to a plainly worded answer rather
than a traceback. One test per unit: :func:`describe`, :func:`transformed`,
:func:`compare` with its :func:`caption`, the pyqtgraph :func:`fast_panel`
and the matplotlib :func:`panel`. The response is a seeded beta sample, so
every summary is pinned exactly.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from spacr import ml
from spacr import response_distribution as rd

_BETA = "beta — bounded, strictly inside (0, 1)"
_FEW = "too few values to name"
_BEFORE = {"n": 200, "family": "beta", "name": _BETA,
           "normality_p": 0.000676275531230837, "skew": 0.6988013258782808,
           "low": 0.02085246378696011, "high": 0.52059607815667}
_AFTER_LOG = {**_BEFORE, "normality_p": 0.011907045147765203,
              "skew": 0.5051500805896232, "low": 0.020638027063349218,
              "high": 0.41910241466687004}
_UNCHANGED_TAIL = (". The two histograms are the same data, drawn twice, "
                   "which is what this setting is currently doing.")


@pytest.fixture
def proportions():
    """A skewed, strictly-inside-(0, 1) response of realistic size."""
    rng = np.random.default_rng(0)
    return np.clip(rng.beta(2.0, 8.0, size=200), 1e-4, 1 - 1e-4)


def _assert_summary(out, expected):
    """Every key equal; NaN matches NaN, floats to 1e-9 relative."""
    assert set(out) == set(expected)
    for key, value in expected.items():
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(out[key]), key
        elif isinstance(value, float):
            assert out[key] == pytest.approx(value, rel=1e-9), key
        else:
            assert out[key] == value, key


def _raise(exc):
    def boom(*_args, **_kwargs):
        raise exc
    return boom


def test_describe_names_the_family_or_says_why_not(subtests, monkeypatch,
                                                   caplog, capsys,
                                                   proportions):
    """Exact summaries for few values, non-finite values and each failure."""
    nan = float("nan")
    unnamed = {"family": "", "name": _FEW, "normality_p": nan, "skew": nan,
               "low": nan, "high": nan}

    with subtests.test(msg="too few values are named as such"):
        _assert_summary(rd.describe([0.1, 0.2, 0.3]), {**unnamed, "n": 3})

    with subtests.test(msg="non-finite values are excluded from the count"):
        _assert_summary(rd.describe([0.1, np.nan, np.inf, -np.inf, 0.9]),
                        {**unnamed, "n": 2})

    with subtests.test(msg="a real response gets a family, range and p"):
        capsys.readouterr()
        _assert_summary(rd.describe(proportions), _BEFORE)
        assert _BEFORE["low"] == float(proportions.min())
        # check_distribution prints its reasoning; the panel swallows it.
        assert capsys.readouterr().out == ""

    with subtests.test(msg="an unknown family is shown under its own name"):
        with monkeypatch.context() as patch:
            patch.setattr(ml, "check_distribution", lambda data: "poisson_ish")
            _assert_summary(rd.describe(proportions),
                            {**_BEFORE, "family": "poisson_ish",
                             "name": "poisson_ish"})

    with subtests.test(msg="a broken shape measurement keeps the family"):
        import scipy.stats

        caplog.clear()
        with monkeypatch.context() as patch, caplog.at_level(
                logging.DEBUG, logger="spacr.response_distribution"):
            patch.setattr(scipy.stats, "normaltest",
                          _raise(RuntimeError("no shape today")))
            out = rd.describe(proportions)
        _assert_summary(out, {**_BEFORE, "normality_p": nan})
        assert [r.getMessage() for r in caplog.records] == [
            "could not measure the response's shape"]

    with subtests.test(msg="a classifier that raises leaves it unnamed"):
        caplog.clear()
        with monkeypatch.context() as patch, caplog.at_level(
                logging.DEBUG, logger="spacr.response_distribution"):
            patch.setattr(ml, "check_distribution",
                          _raise(ValueError("cannot classify")))
            out = rd.describe(proportions)
        _assert_summary(out, {**_BEFORE, "family": "", "name": _FEW})
        assert [r.getMessage() for r in caplog.records] == [
            "could not classify the response"]


class _Refuses:
    def fit_transform(self, data):
        raise ValueError("negative values")


def test_transformed_returns_the_input_unless_a_transform_applies(
        subtests, monkeypatch, caplog, proportions):
    """Identity names, real transforms and each failure mode give exact arrays."""
    data = np.asarray(proportions, dtype=float)

    for name in ("", "   ", "none", "NONE", None):
        with subtests.test(msg=f"no transform: {name!r}"):
            out = rd.transformed(proportions, name)
            assert out.dtype == float
            np.testing.assert_array_equal(out, data)

    with subtests.test(msg="log moves the values to log1p"):
        out = rd.transformed(proportions, " LOG ")
        assert out.shape == proportions.shape
        np.testing.assert_allclose(out, np.log1p(data), rtol=1e-12)

    failures = [
        ("a transformer that cannot be built", _raise(RuntimeError("no")),
         ["could not build the transformer"]),
        ("an unsupported transform", lambda data, name: None, []),
        ("a transform that fails on this response",
         lambda data, name: _Refuses(),
         ["the transform failed on this response"]),
    ]
    for case_id, builder, logged in failures:
        with subtests.test(msg=case_id):
            caplog.clear()
            with monkeypatch.context() as patch, caplog.at_level(
                    logging.DEBUG, logger="spacr.response_distribution"):
                patch.setattr(ml, "apply_transformation", builder)
                out = rd.transformed(proportions, "log")
            np.testing.assert_array_equal(out, data)
            assert [r.getMessage() for r in caplog.records] == logged


def test_compare_flags_change_and_rescaling_and_captions_them(subtests,
                                                              proportions):
    """``changed``/``rescaled`` flags and the exact caption per transform."""
    before = (f"{_BETA} (D'Agostino p = 0.000676, skew +0.70)")

    with subtests.test(msg="a transform that changed nothing says so"):
        result = rd.compare(proportions, "none")
        assert (result["transform"], result["changed"], result["rescaled"]) \
            == ("none", False, False)
        _assert_summary(result["after"], _BEFORE)
        assert rd.caption(result) == (
            f"transform 'none' changed nothing: {before}{_UNCHANGED_TAIL}")

    with subtests.test(msg="an empty name normalises to none"):
        result = rd.compare(proportions, "  ")
        assert (result["transform"], result["changed"]) == ("none", False)

    with subtests.test(msg="a rescaling transform is flagged for its axis"):
        result = rd.compare(np.append(proportions, np.nan), "log")
        assert (result["transform"], result["changed"], result["rescaled"]) \
            == ("log", True, True)
        assert result["values_before"].shape == (200,)
        assert result["values_after"].shape == (200,)
        _assert_summary(result["before"], _BEFORE)
        _assert_summary(result["after"], _AFTER_LOG)
        assert rd.caption(result) == (
            f"before: {before}   →   after log: "
            f"{_BETA} (D'Agostino p = 0.0119, skew +0.51)")

    with subtests.test(msg="too few values contribute only their name"):
        assert rd.caption(rd.compare([0.1, 0.2, 0.3], "none")) == (
            f"transform 'none' changed nothing: {_FEW}{_UNCHANGED_TAIL}")

    with subtests.test(msg="a family with NaN statistics shows only its label"):
        part = {"family": "ols", "name": "normal",
                "normality_p": float("nan"), "skew": float("nan")}
        assert rd.caption({"changed": True, "transform": "none",
                           "before": part, "after": part}) == (
            "before: normal   →   after none: normal")

    with subtests.test(msg="a finite p alone is shown without a skew"):
        part = {"family": "ols", "name": "normal", "normality_p": 0.5,
                "skew": float("nan")}
        assert rd.caption({"changed": True, "transform": "sqrt",
                           "before": part, "after": part}) == (
            "before: normal (D'Agostino p = 0.5)   →   after sqrt: "
            "normal (D'Agostino p = 0.5)")


class _Empties:
    def fit_transform(self, data):
        return np.empty((0, 1))


def test_fast_panel_draws_one_outline_per_series(subtests, qapp, monkeypatch,
                                                 proportions):
    """Nothing finite draws nothing; each non-empty series is one outline."""
    from spacr.qt.widgets.fast_plots import FastPlot

    with subtests.test(msg="nothing finite draws nothing"):
        assert rd.fast_panel([np.nan, np.inf, -np.inf], "none") is None

    bins = 2 * int(np.sqrt(200))  # 28 bins: 56 step points per outline
    with subtests.test(msg="both outlines land on the given plot"):
        plot = FastPlot(title="t", x_label="x", y_label="y")
        before = len(plot.plot.listDataItems())
        got = rd.fast_panel(proportions, "log", plot=plot,
                            dependent_variable="fraction infected")
        assert got is plot
        drawn = plot.plot.listDataItems()[before:]
        assert len(drawn) == 2
        for item, series in zip(drawn, (proportions, np.log1p(proportions))):
            xs, ys = item.getData()
            counts, edges = np.histogram(series, bins=bins)
            np.testing.assert_allclose(xs, np.repeat(edges, 2)[1:-1])
            np.testing.assert_array_equal(ys, np.repeat(counts, 2))
        plot.deleteLater()

    with subtests.test(msg="a plot is made when none is given"):
        plot = rd.fast_panel(proportions, "none",
                             dependent_variable="fraction infected")
        assert isinstance(plot, FastPlot)
        assert len(plot.plot.listDataItems()) == 2
        plot.deleteLater()

    with subtests.test(msg="an empty transformed series is skipped"):
        with monkeypatch.context() as patch:
            patch.setattr(ml, "apply_transformation",
                          lambda data, name: _Empties())
            plot = FastPlot(title="t", x_label="x", y_label="y")
            assert rd.fast_panel(proportions, "log", plot=plot) is plot
        assert len(plot.plot.listDataItems()) == 1
        plot.deleteLater()


def _legend(ax):
    return [text.get_text() for text in ax.get_legend().get_texts()]


def test_panel_shares_or_twins_the_axis_and_always_has_a_legend(subtests,
                                                                proportions):
    """Axis labels, twin axes, bar counts and legends for each transform."""
    from matplotlib.figure import Figure

    with subtests.test(msg="nothing rescaled shares one axis on a given ax"):
        ax = Figure().add_subplot(111)
        result = rd.panel(proportions, "none", ax=ax,
                          dependent_variable="score")
        assert result["axes"] is ax
        assert (ax.get_xlabel(), ax.get_ylabel()) == ("score", "wells")
        assert ax.get_title() == "Response before and after the transform"
        assert _legend(ax) == ["before", "after none"]
        assert ax.figure.axes == [ax]
        assert len(ax.patches) == 80
        assert ax.texts[0].get_text() == rd.caption(result)

    with subtests.test(msg="unnamed response is labelled generically"):
        result = rd.panel(proportions, "none")
        ax = result["axes"]
        assert ax.get_xlabel() == "response"
        assert ax.get_title() == "Response before and after the transform"
        assert ax.figure._spacr_user_style_applied

    with subtests.test(msg="an unchanged logit is drawn on one axis"):
        values = np.random.default_rng(4).uniform(0.01, 0.99, 300)
        result = rd.panel(values, "logit")
        ax = result["axes"]
        assert (result["rescaled"], result["changed"]) == (True, False)
        assert ax.figure.axes == [ax]
        assert ax.get_xlabel() == "response"
        assert _legend(ax) == ["before", "after logit"]

    for transform, values in (("log", proportions),
                              ("sqrt", np.linspace(0.05, 0.95, 40))):
        with subtests.test(msg=f"{transform} twins the axis"):
            result = rd.panel(values, transform, dependent_variable="score")
            ax = result["axes"]
            twins = [other for other in ax.figure.axes if other is not ax]
            assert len(twins) == 1
            assert twins[0].get_xlabel() == f"after {transform}"
            assert len(ax.patches) == 40 and len(twins[0].patches) == 40
            assert _legend(ax) == ["before", f"after {transform}"]
            assert ax.get_xlabel() == "score"
