"""The profiler predicts from whatever the fitted object actually offers.

``predict`` is pointed at four different kinds of fitted object -- a
statsmodels result that takes an offset, one that refuses the keyword, one
whose design does not line up at all, and a bare coefficient table. Getting
any of them wrong draws a curve, so the failure is a plot that means nothing
rather than an error. The sweep and the ranking have the same problem: a
column that is not numeric and an input that never moves must drop out
visibly instead of contributing a flat, meaningless line.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from spacr.profiler import Profile, predict, profile, sensitivity


class _TakesAnOffset:
    """Quacks like a statsmodels count result: predict(exog, offset=...)."""

    params = pd.Series({"x": 2.0, "Intercept": 1.0})

    def predict(self, exog, offset=None):
        base = 2.0 * np.asarray(exog["x"], dtype=float) + 1.0
        if offset is None:
            return base
        return base + np.asarray(offset, dtype=float)


class _RefusesTheOffsetKeyword:
    """A scikit-learn regressor: predict(X) and nothing else."""

    params = pd.Series({"x": 2.0, "Intercept": 1.0})

    def predict(self, exog):
        return 2.0 * np.asarray(exog["x"], dtype=float) + 1.0


class _PredictThatCannotAlign:
    """A result whose predict raises on a design it does not recognise."""

    params = pd.Series({"x": 3.0, "Intercept": 0.5})

    def predict(self, _exog, **_kwargs):
        raise KeyError("design matrix does not match the formula")


class _NothingToPredictFrom:
    """Neither a predict method nor coefficients."""


_ROWS = pd.DataFrame({"x": [1.0, 2.0]})


def test_a_model_that_takes_an_offset_is_given_one():
    """A count model fitted with log exposure must be offset when profiled."""
    out = predict(_TakesAnOffset(), _ROWS, offset=[10.0, 20.0])
    assert list(out) == [13.0, 25.0]


def test_a_model_that_refuses_the_offset_keyword_is_predicted_without_it():
    """The keyword is an attempt, not a requirement; the curve still draws."""
    out = predict(_RefusesTheOffsetKeyword(), _ROWS, offset=[10.0, 20.0])
    assert list(out) == [3.0, 5.0]


def test_a_predict_that_cannot_align_falls_back_to_the_coefficients():
    """Coefficients align by name, which is what makes the fallback safe."""
    out = predict(_PredictThatCannotAlign(), _ROWS)
    assert list(out) == [3.5, 6.5]


def test_the_intercept_is_added_even_though_the_design_has_no_such_column():
    """A design matrix without an Intercept column still gets the constant."""
    out = predict(_PredictThatCannotAlign(), pd.DataFrame({"x": [0.0]}))
    assert list(out) == [0.5]


def test_a_bare_array_of_design_rows_is_accepted():
    """Callers hand over rows, not always a frame; both must predict."""
    class _ByPosition:
        params = pd.Series({0: 2.0, 1: 3.0})

    out = predict(_ByPosition(), [[1.0, 1.0], [2.0, 0.0]])
    assert list(out) == [5.0, 4.0]


def test_an_object_with_nothing_to_predict_from_is_refused():
    """Guessing here would draw a curve nobody could interpret."""
    with pytest.raises(TypeError) as excinfo:
        predict(_NothingToPredictFrom(), _ROWS)
    assert "_NothingToPredictFrom" in str(excinfo.value)


def test_a_non_numeric_column_is_swept_over_a_stated_unit_range():
    """A column with no numbers has no range, so the sweep states its own."""
    design = pd.DataFrame({"x": ["low", "high", "low"], "y": [1.0, 2.0, 3.0]})
    curve = profile(_RefusesTheOffsetKeyword(), design, "x", n=5)
    assert curve.values[0] == 0.0
    assert curve.values[-1] == 1.0
    assert len(curve) == 5


def test_a_curve_that_never_moved_reports_no_slope():
    """One value swept twice has no run, and 0/0 is unknown, not zero."""
    flat = Profile(variable="x", values=(2.0, 2.0), predictions=(5.0, 5.0))
    assert math.isnan(flat.slope)


def test_a_non_numeric_input_is_left_out_of_the_ranking():
    """An input with no numbers cannot be swept, so it cannot be ranked."""
    design = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0],
                           "label": ["a", "b", "c", "d"]})
    ranked = sensitivity(_RefusesTheOffsetKeyword(), design,
                         variables=["x", "label"])
    assert [s.variable for s in ranked] == ["x"]
