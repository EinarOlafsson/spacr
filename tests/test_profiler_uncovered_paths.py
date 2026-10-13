"""The profiler's coefficient reader and ranker on the objects that are not
a tidy statsmodels result.

Three of the four fitted-object shapes :func:`spacr.profiler.predict` has to
cope with never carry an intercept the way scikit-learn does: the horseshoe
fitter exposes ``coef_`` and nothing else, a penalised fit can arrive with an
``intercept_`` that is an empty array, and a model handed to
:func:`~spacr.profiler.sensitivity` may carry no coefficient table at all.
Each of those has to produce a number the user can read rather than an
invented intercept or a coefficient copied from the wrong column.

The last case here is the one that has a real trigger in the wild: a
penalised backend shrinks every coefficient to exactly zero, and
``from_coefficients(..., drop_zero=True)`` therefore builds a model with no
coefficients whatsoever. Predicting from that must say so instead of
returning a column of zeros that would be plotted as a flat, meaningless
curve.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from spacr.profiler import (FittedLinear, from_coefficients, predict,
                            sensitivity)


class _CoefficientsWithoutAnIntercept:
    """The horseshoe fitter's surface: ``coef_``, names, no intercept."""

    coef_ = np.array([2.0, -1.0])
    feature_names_in_ = ["a", "b"]


class _CoefficientsWithAnEmptyIntercept:
    """A fit whose ``intercept_`` came back as a zero-length array."""

    coef_ = np.array([2.0, -1.0])
    feature_names_in_ = ["a", "b"]
    intercept_ = np.array([])


class _CoefficientsWithAnIntercept:
    """The same fit, with an intercept that is actually there."""

    coef_ = np.array([2.0, -1.0])
    feature_names_in_ = ["a", "b"]
    intercept_ = np.array([5.0])


class _PredictsWithoutCoefficients:
    """A fitted object that can predict but publishes no coefficient table."""

    def predict(self, exog):
        return (3.0 * np.asarray(exog["a"], dtype=float)
                - 1.0 * np.asarray(exog["b"], dtype=float))


class _PredictsAndNamesOnlyOneInput:
    """Predicts from two inputs but only publishes a coefficient for one."""

    params = pd.Series({"a": 3.0})

    def predict(self, exog):
        return (3.0 * np.asarray(exog["a"], dtype=float)
                - 1.0 * np.asarray(exog["b"], dtype=float))


@pytest.fixture
def two_column_design():
    """A design whose 5th and 95th percentiles are known exactly."""
    values = np.linspace(0.0, 1.0, 101)
    return pd.DataFrame({"a": values, "b": values})


def test_a_fit_without_an_intercept_attribute_gets_no_intercept_added():
    """``coef_`` alone predicts ``X @ beta`` with nothing added on top."""
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    got = predict(_CoefficientsWithoutAnIntercept(), frame)

    # 2*a - b, and no intercept term invented for it.
    np.testing.assert_allclose(got, [-1.0, 0.0])


def test_an_empty_intercept_array_contributes_nothing_to_the_prediction():
    """A zero-length ``intercept_`` is not read as a shift of zero-th entry."""
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    empty = predict(_CoefficientsWithAnEmptyIntercept(), frame)
    present = predict(_CoefficientsWithAnIntercept(), frame)

    np.testing.assert_allclose(empty, [-1.0, 0.0])
    # The same coefficients with a real intercept shift by exactly 5.
    np.testing.assert_allclose(present, empty + 5.0)


def test_sensitivity_reports_nan_when_the_model_publishes_no_coefficients(
        two_column_design):
    """A model that can only predict still gets a span; its coefficient is nan."""
    ranked = sensitivity(_PredictsWithoutCoefficients(), two_column_design)

    assert [s.variable for s in ranked] == ["a", "b"]
    by_name = {s.variable: s for s in ranked}
    # 5th and 95th percentiles of linspace(0, 1, 101) are 0.05 and 0.95.
    assert by_name["a"].low == pytest.approx(0.05)
    assert by_name["a"].high == pytest.approx(0.95)
    # a moves the prediction by 3 * 0.9, b by -1 * 0.9, with the other held.
    assert by_name["a"].span == pytest.approx(3.0 * 0.9)
    assert by_name["b"].span == pytest.approx(-0.9)
    assert math.isnan(by_name["a"].coefficient)
    assert math.isnan(by_name["b"].coefficient)


def test_sensitivity_leaves_an_unnamed_input_without_a_coefficient(
        two_column_design):
    """Only the inputs the coefficient table names get a coefficient."""
    ranked = sensitivity(_PredictsAndNamesOnlyOneInput(), two_column_design)

    by_name = {s.variable: s for s in ranked}
    assert by_name["a"].coefficient == pytest.approx(3.0)
    assert math.isnan(by_name["b"].coefficient)
    # The span is measured from the predictions either way, not the table.
    assert by_name["b"].span == pytest.approx(-0.9)


def test_a_model_whose_coefficients_were_all_dropped_refuses_to_predict():
    """``drop_zero`` on an all-zero fit leaves nothing to predict from."""
    table = pd.DataFrame({"feature": ["Intercept", "a", "b"],
                          "coefficient": [0.0, 0.0, 0.0]})

    fitted = from_coefficients(table, drop_zero=True)

    assert len(fitted.params) == 0
    with pytest.raises(TypeError, match="no coefficients to predict from"):
        fitted.predict(pd.DataFrame({"a": [1.0], "b": [2.0]}))


def test_an_empty_coefficient_series_is_still_a_usable_object():
    """The empty model reports no features rather than failing to build."""
    fitted = FittedLinear(params=pd.Series(dtype=float), link="logit")

    assert fitted.feature_names == ()
    with pytest.raises(TypeError, match="no coefficients to predict from"):
        fitted.predict({"a": 1.0})
