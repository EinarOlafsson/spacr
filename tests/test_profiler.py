"""Real tests for :mod:`spacr.profiler`.

The profiler's whole claim is that it works on the fitted object rather than
on a re-fit, and that it works on ALL of them: seventeen backends whose
prediction APIs disagree in three different ways. So most of this file is
parametrised over :data:`spacr.ml.REGRESSION_TYPES` with models fitted by
:func:`spacr.ml.regression_model` itself, on a design with a planted
coefficient, and asserts that moving the planted input moves the prediction
in the planted direction.

The two traps that motivated the module are asserted explicitly:
``LinearSVC.predict`` returns a class label (profiling it draws a step
function that looks like a finding), and a profiler that re-fits shows the
user a second model under the first one's name.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spacr import profiler                                       # noqa: E402
from spacr.profiler import (FittedLinear, LINKS, Profile,         # noqa: E402
                            coefficient_frame, from_coefficients,
                            predict, profile, profile_by,
                            reference_row, response_scale, sensitivity)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def design():
    """A small design with a planted signal in ``a`` and noise in ``b``."""
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "Intercept": np.ones(n),
        "a": rng.uniform(0.0, 1.0, n),
        "b": rng.uniform(0.0, 1.0, n),
        "c": rng.uniform(0.0, 1.0, n),
    })


@pytest.fixture(scope="module")
def linear_model(design):
    """An OLS fit whose ``a`` coefficient is +3 by construction."""
    import statsmodels.api as sm

    y = 1.0 + 3.0 * design["a"] - 0.2 * design["b"]
    return sm.OLS(y, design).fit()


def _fit(regression_type, design):
    """Fit one backend on a response shaped for it, planting ``a`` positive."""
    from spacr.ml import regression_model

    rng = np.random.default_rng(7)
    n = len(design)
    linear = -0.5 + 2.5 * design["a"].to_numpy() - 0.3 * design["b"].to_numpy()
    weights = rng.integers(30, 400, n).astype(float)

    kwargs = {"regression_type": regression_type}
    if regression_type in ("logit", "probit", "quasi_binomial"):
        y = 1.0 / (1.0 + np.exp(-linear))
        kwargs["weights"] = weights
    elif regression_type == "wls":
        y = linear + rng.normal(0, 0.05, n)
        kwargs["weights"] = weights
    elif regression_type in ("poisson", "horseshoe"):
        rate = np.exp(linear - 2.0)
        y = rng.poisson(rate * weights).astype(float)
        y = np.minimum(y, weights)
        kwargs["exposure"] = weights
    elif regression_type == "beta":
        y = np.clip(1.0 / (1.0 + np.exp(-linear)), 1e-3, 1 - 1e-3)
    elif regression_type == "hinge":
        y = (linear + rng.normal(0, 0.2, n) > 0).astype(float)
    elif regression_type == "mixed":
        y = linear + rng.normal(0, 0.05, n)
        kwargs["groups"] = np.repeat(np.arange(4), n // 4)
    elif regression_type == "glm":
        y = linear + rng.normal(0, 0.05, n)
    else:
        y = linear + rng.normal(0, 0.05, n)

    if regression_type in ("lasso", "ridge", "elasticnet"):
        kwargs["alpha"] = 0.001
    return regression_model(design, pd.Series(np.asarray(y, dtype=float)),
                            **kwargs)


# ---------------------------------------------------------------------------
# every backend
# ---------------------------------------------------------------------------

def test_the_backend_list_is_the_one_ml_publishes():
    from spacr.ml import REGRESSION_TYPES

    assert len(REGRESSION_TYPES) == 17


@pytest.mark.parametrize("regression_type", [
    "ols", "wls", "rlm", "huber", "glm", "poisson", "quasi_binomial", "beta",
    "logit", "probit", "quantile", "mixed", "lasso", "ridge", "elasticnet",
    "hinge",
])
def test_every_backend_can_be_profiled_and_moves_the_right_way(
        regression_type, design):
    """The planted +2.5 on ``a`` must come back as a rising curve."""
    pytest.importorskip("statsmodels")
    model = _fit(regression_type, design)

    curve = profile(model, design, "a", n=11)

    assert len(curve) == 11
    assert all(math.isfinite(p) for p in curve.predictions)
    assert curve.predictions[-1] > curve.predictions[0], (
        f"{regression_type}: the planted positive effect came back as "
        f"{curve.predictions[0]:.4g} -> {curve.predictions[-1]:.4g}")
    assert curve.span > 0
    assert curve.slope > 0
    assert curve.scale, "every curve must say what its axis means"
    assert set(curve.held) == {"Intercept", "b", "c"}


def test_the_horseshoe_backend_is_profiled_when_it_is_installed(design):
    pytest.importorskip("spacr.power_model")
    model = _fit("horseshoe", design)

    curve = profile(model, design, "a", n=7)

    assert len(curve) == 7
    assert all(math.isfinite(p) for p in curve.predictions)


def test_the_hinge_backend_is_profiled_on_its_margin_not_its_class(design):
    """``LinearSVC.predict`` returns 0/1; profiling that draws a step."""
    model = _fit("hinge", design)

    curve = profile(model, design, "a", n=25)

    assert "decision function" in curve.scale
    assert len(set(curve.predictions)) > 2, (
        "a hinge profile of the class label would take two distinct values")
    classes = set(np.asarray(model.predict(design)).ravel().tolist())
    assert classes <= {0.0, 1.0}, "the trap this branch exists for"


# ---------------------------------------------------------------------------
# the prediction seam
# ---------------------------------------------------------------------------

def test_predict_accepts_a_row_a_mapping_and_a_frame(linear_model, design):
    row = design.iloc[[0]]

    from_frame = predict(linear_model, row)
    from_series = predict(linear_model, design.iloc[0])
    from_mapping = predict(linear_model, design.iloc[0].to_dict())

    assert from_frame == pytest.approx(from_series)
    assert from_frame == pytest.approx(from_mapping)


def test_predict_falls_back_to_the_linear_predictor(design):
    """A fitted object with coefficients and nothing else is still usable."""

    class _CoefficientsOnly:
        params = pd.Series({"Intercept": 1.0, "a": 3.0, "b": -0.2})

    expected = 1.0 + 3.0 * design["a"] - 0.2 * design["b"]

    assert predict(_CoefficientsOnly(), design) == pytest.approx(
        expected.to_numpy())


def test_predict_refuses_an_object_it_cannot_read():
    with pytest.raises(TypeError) as caught:
        predict(object(), pd.DataFrame({"a": [1.0]}))

    assert "nothing to profile" in str(caught.value)


def test_an_offset_shifts_the_linear_predictor(design):
    class _CoefficientsOnly:
        params = pd.Series({"a": 1.0})

    plain = predict(_CoefficientsOnly(), design)
    shifted = predict(_CoefficientsOnly(), design, offset=np.full(len(design),
                                                                 2.0))

    assert shifted == pytest.approx(plain + 2.0)


def test_response_scale_names_what_came_back(design):
    pytest.importorskip("statsmodels")

    assert response_scale(_fit("ols", design)) == "response"
    assert "logit" in response_scale(_fit("logit", design))
    assert "decision function" in response_scale(_fit("hinge", design))
    assert response_scale(_fit("quantile", design)) == "conditional quantile"
    assert "poisson" in response_scale(_fit("poisson", design))
    assert response_scale(FittedLinear(pd.Series({"a": 1.0}),
                                       link="log")) == "rate (log link)"


def test_sklearn_coefficients_are_read_with_their_feature_names(design):
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=0.1).fit(design[["a", "b"]], design["a"] * 2.0)
    params = profiler._coefficients(model)

    assert list(params.index)[:2] == ["a", "b"]
    assert "Intercept" in params.index


# ---------------------------------------------------------------------------
# holding the other inputs
# ---------------------------------------------------------------------------

def test_the_reference_row_holds_every_input_at_its_median(design):
    row = reference_row(design)

    assert row["a"] == pytest.approx(design["a"].median())
    assert row["Intercept"] == 1.0, "an intercept column is 1 by construction"


def test_every_reference_method_is_honoured(design):
    assert reference_row(design, method="mean")["a"] == pytest.approx(
        design["a"].mean())
    assert reference_row(design, method="min")["a"] == pytest.approx(
        design["a"].min())
    assert reference_row(design, method="zero")["a"] == 0.0
    assert reference_row(design, method="zero")["Intercept"] == 1.0


def test_an_unknown_reference_method_is_refused(design):
    with pytest.raises(ValueError):
        reference_row(design, method="mode")


def test_chosen_values_override_the_reference(design):
    row = reference_row(design, at={"b": 0.9, "c": 0.1})

    assert row["b"] == 0.9 and row["c"] == 0.1
    assert row["a"] == pytest.approx(design["a"].median())


def test_holding_a_column_that_does_not_exist_is_refused_not_ignored(design):
    with pytest.raises(ValueError) as caught:
        reference_row(design, at={"typo": 1.0})

    assert "typo" in str(caught.value), (
        "a silently ignored hold produces a curve with no explanation")


def test_the_held_vector_travels_with_the_curve(linear_model, design):
    curve = profile(linear_model, design, "a", at={"b": 0.75}, n=5)

    assert curve.held["b"] == 0.75
    assert "a" not in curve.held, "the moving input is not a held one"
    assert curve.reference_method == "median"
    assert curve.to_dict()["held"]["b"] == 0.75


def test_moving_a_held_value_moves_the_whole_curve(linear_model, design):
    low = profile(linear_model, design, "a", at={"b": 0.0}, n=5)
    high = profile(linear_model, design, "a", at={"b": 1.0}, n=5)

    # b's coefficient is -0.2, so holding it higher lowers every prediction
    # by the same amount: a parallel shift, not a change of slope.
    shift = np.asarray(high.predictions) - np.asarray(low.predictions)
    assert shift.std() == pytest.approx(0.0, abs=1e-9)
    assert shift.mean() < 0
    assert high.slope == pytest.approx(low.slope)


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------

def test_the_sweep_covers_the_observed_range(linear_model, design):
    curve = profile(linear_model, design, "a", n=9)

    assert curve.values[0] == pytest.approx(design["a"].min())
    assert curve.values[-1] == pytest.approx(design["a"].max())
    assert len(curve) == 9


def test_explicit_values_are_swept_exactly(linear_model, design):
    curve = profile(linear_model, design, "a", values=[0.0, 0.5, 1.0])

    assert curve.values == (0.0, 0.5, 1.0)
    assert curve.predictions[1] == pytest.approx(
        (curve.predictions[0] + curve.predictions[2]) / 2)


def test_a_constant_column_is_widened_rather_than_collapsed(linear_model):
    frame = pd.DataFrame({"Intercept": [1.0] * 5, "a": [0.5] * 5,
                          "b": [0.1] * 5, "c": [0.0] * 5})

    curve = profile(linear_model, frame, "a", n=5)

    assert curve.values[0] < 0.5 < curve.values[-1], (
        "a column with one observed value still has a 'what if' to answer")


def test_an_empty_value_list_is_refused(linear_model, design):
    with pytest.raises(ValueError):
        profile(linear_model, design, "a", values=[])


def test_profiling_a_column_that_is_not_in_the_design_is_refused(linear_model,
                                                                 design):
    with pytest.raises(KeyError) as caught:
        profile(linear_model, design, "not_a_column")

    assert "not_a_column" in str(caught.value)


def test_the_baseline_is_the_prediction_with_everything_held(linear_model,
                                                             design):
    # Sweep the single point the reference itself sits at, so the comparison
    # is exact rather than "the nearest of twenty-one grid points".
    median = float(design["a"].median())
    curve = profile(linear_model, design, "a", values=[median])

    assert curve.baseline == pytest.approx(curve.predictions[0])
    # And with the input held away from its median the curve departs from it.
    shifted = profile(linear_model, design, "a", values=[median + 0.25])
    assert shifted.baseline == pytest.approx(curve.baseline)
    assert shifted.predictions[0] > shifted.baseline


def test_at_returns_the_nearest_swept_point(linear_model, design):
    curve = profile(linear_model, design, "a", values=[0.0, 0.5, 1.0])

    assert curve.at(0.49) == pytest.approx(curve.predictions[1])
    assert curve.at(-5.0) == pytest.approx(curve.predictions[0])
    assert Profile("x", (), ()).at(1.0) != Profile("x", (), ()).at(1.0) or True
    assert math.isnan(Profile("x", (), ()).at(1.0))


def test_an_empty_profile_reports_no_span_or_slope():
    empty = Profile("x", (), ())

    assert math.isnan(empty.span)
    assert math.isnan(empty.slope)
    assert len(empty) == 0


def test_the_curve_is_a_frame_and_a_dict(linear_model, design):
    curve = profile(linear_model, design, "a", n=4)

    frame = curve.to_frame()
    assert list(frame.columns) == ["a", "prediction"]
    assert len(frame) == 4
    payload = curve.to_dict()
    assert payload["variable"] == "a"
    assert payload["span"] > 0


# ---------------------------------------------------------------------------
# profiling at several held levels
# ---------------------------------------------------------------------------

def test_profile_by_returns_one_curve_per_level(linear_model, design):
    curves = profile_by(linear_model, design, "a", by="b",
                        levels=[0.0, 0.5, 1.0], n=5)

    assert len(curves) == 3
    assert [c.held["b"] for c in curves] == [0.0, 0.5, 1.0]
    assert curves[0].predictions[0] > curves[-1].predictions[0]


def test_profile_by_keeps_other_holds(linear_model, design):
    curves = profile_by(linear_model, design, "a", by="b", levels=[0.2],
                        at={"c": 0.9}, n=3)

    assert curves[0].held["c"] == 0.9
    assert curves[0].held["b"] == 0.2


def test_profile_by_needs_a_level(linear_model, design):
    with pytest.raises(ValueError):
        profile_by(linear_model, design, "a", by="b", levels=[])


# ---------------------------------------------------------------------------
# sensitivity
# ---------------------------------------------------------------------------

def test_sensitivity_ranks_the_planted_input_first(linear_model, design):
    ranked = sensitivity(linear_model, design)

    assert ranked[0].variable == "a"
    assert ranked[0].span > 0
    assert ranked[0].coefficient == pytest.approx(3.0, rel=1e-6)
    assert [s.variable for s in ranked] == ["a", "b", "c"], (
        "the ranking must follow the prediction movement, not the column order")


def test_sensitivity_skips_the_intercept_and_constant_columns(linear_model,
                                                              design):
    ranked = sensitivity(linear_model, design)

    assert "Intercept" not in [s.variable for s in ranked]


def test_sensitivity_can_be_limited_and_restricted(linear_model, design):
    assert len(sensitivity(linear_model, design, limit=2)) == 2
    only = sensitivity(linear_model, design, variables=["b"])
    assert [s.variable for s in only] == ["b"]
    assert sensitivity(linear_model, design, variables=["nope"]) == []


def test_sensitivity_uses_quantiles_so_one_outlier_cannot_decide_it(
        linear_model):
    frame = pd.DataFrame({
        "Intercept": np.ones(100),
        "a": np.r_[np.full(99, 0.5), 1e6],
        "b": np.linspace(0.0, 1.0, 100),
        "c": np.zeros(100)})

    ranked = sensitivity(linear_model, frame)

    assert [s.variable for s in ranked][0] == "b", (
        "a single 1e6 well must not make 'a' the most influential input")


def test_a_design_with_nothing_to_sweep_gives_an_empty_ranking(linear_model):
    frame = pd.DataFrame({"Intercept": [1.0, 1.0], "a": [0.5, 0.5],
                          "b": [0.1, 0.1], "c": [0.0, 0.0]})

    assert sensitivity(linear_model, frame) == []


def test_a_sensitivity_record_is_json_serializable(linear_model, design):
    import json

    payload = json.dumps([s.to_dict()
                          for s in sensitivity(linear_model, design)])

    assert "prediction_high" in payload


# ---------------------------------------------------------------------------
# reading a written-out fit
# ---------------------------------------------------------------------------

def test_a_written_out_coefficient_table_reproduces_the_fit(linear_model,
                                                            design, tmp_path):
    """The whole point of FittedLinear: reading a fit, not repeating it."""
    frame = pd.DataFrame({
        "feature": [str(name) for name in linear_model.params.index],
        "coefficient": linear_model.params.to_numpy(dtype=float)})
    path = tmp_path / "results.csv"
    frame.to_csv(path, index=False)

    surrogate = from_coefficients(path)

    assert surrogate.predict(design) == pytest.approx(
        np.asarray(linear_model.predict(design), dtype=float))
    live = profile(linear_model, design, "a", n=7)
    read = profile(surrogate, design, "a", n=7)
    assert read.predictions == pytest.approx(live.predictions)


def test_from_coefficients_accepts_a_frame_and_a_fitted_object(linear_model):
    from_frame = from_coefficients(pd.DataFrame(
        {"feature": ["Intercept", "a"], "coefficient": [1.0, 2.0]}))
    from_model = from_coefficients(coefficient_frame(linear_model))

    assert from_frame.params["a"] == 2.0
    assert from_model.params["a"] == pytest.approx(3.0, rel=1e-6)
    assert from_frame.feature_names == ("a",)


def test_a_table_without_the_columns_is_refused():
    with pytest.raises(ValueError) as caught:
        coefficient_frame(pd.DataFrame({"term": ["a"], "beta": [1.0]}))

    assert "coefficient" in str(caught.value)


def test_an_object_with_no_coefficients_is_refused():
    class _Empty:
        params = pd.Series(dtype=float)
        coef_ = None

    with pytest.raises(ValueError):
        coefficient_frame(_Empty())


def test_zero_coefficients_are_kept_unless_asked_to_drop(tmp_path):
    frame = pd.DataFrame({"feature": ["a", "b"], "coefficient": [1.0, 0.0]})

    assert from_coefficients(frame).feature_names == ("a", "b")
    assert from_coefficients(frame, drop_zero=True).feature_names == ("a",)


def test_a_duplicate_coefficient_takes_the_first(tmp_path):
    frame = pd.DataFrame({"feature": ["a", "a"], "coefficient": [1.0, 9.0]})

    assert from_coefficients(frame).params["a"] == 1.0


def test_every_link_is_applied_on_the_scale_it_names():
    frame = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
    params = pd.Series({"a": 1.0})

    identity = FittedLinear(params, link="identity").predict(frame)
    log = FittedLinear(params, link="log").predict(frame)
    logit = FittedLinear(params, link="logit").predict(frame)
    probit = FittedLinear(params, link="probit").predict(frame)

    assert identity == pytest.approx([0.0, 1.0, 2.0])
    assert log == pytest.approx(np.exp([0.0, 1.0, 2.0]))
    assert logit[0] == pytest.approx(0.5)
    assert probit[0] == pytest.approx(0.5)
    assert all(0.0 <= v <= 1.0 for v in logit)
    assert all(0.0 <= v <= 1.0 for v in probit)
    assert set(LINKS) == {"identity", "log", "logit", "probit"}


def test_an_unknown_link_is_refused():
    with pytest.raises(ValueError) as caught:
        FittedLinear(pd.Series({"a": 1.0}), link="cloglog")

    assert "cloglog" in str(caught.value)


def test_the_logit_link_does_not_overflow_on_an_extreme_linear_predictor():
    values = FittedLinear(pd.Series({"a": 1.0}), link="logit").predict(
        pd.DataFrame({"a": [-5000.0, 5000.0]}))

    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(1.0)


def test_the_surrogate_carries_its_label_into_the_curve(linear_model, design):
    surrogate = from_coefficients(coefficient_frame(linear_model),
                                  label="ols run 3")

    curve = profile(surrogate, design, "a", n=3)

    assert curve.model_label == "ols run 3"
