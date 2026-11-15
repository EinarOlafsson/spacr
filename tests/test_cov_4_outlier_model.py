"""Robust outlier detection answers for the degenerate columns real plates have.

Every branch here is a column that breaks the ordinary arithmetic: a feature
with nothing finite in it, a feature where more than half the objects share
one value so the spread is exactly zero, a well whose median is missing for
one feature, and two features that are the same measurement in different
units. Dividing by any of those spreads flags the whole tail, which is the
one failure mode a robust test exists to prevent -- so each has a stated
answer and a stated note, and the note has to reach the report.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.qt.widgets.outlier_model import (METHOD_IQR, METHOD_MAD,
                                            METHOD_MAHALANOBIS, OutlierError,
                                            OutlierResult, OutlierSpec,
                                            median_absolute_deviation,
                                            robust_scale, tukey_fences)


# -- the scale estimators ----------------------------------------------------

def test_a_column_with_nothing_finite_has_no_deviation():
    """nan, not zero: no spread measured is not a spread of zero."""
    assert np.isnan(median_absolute_deviation([np.nan, np.inf, -np.inf]))


def test_a_column_with_nothing_finite_is_reported_as_empty():
    """The caller has to be able to tell "no data" from "no variation"."""
    centre, scale, note = robust_scale([np.nan, np.inf])
    assert np.isnan(centre)
    assert scale == 0.0
    assert note == "empty"


def test_tukey_fences_on_nothing_finite_is_reported_as_empty():
    """A fence built from no values would flag everything or nothing."""
    q1, q3, low, high, note = tukey_fences([np.nan, np.inf])
    assert note == "empty"
    assert all(np.isnan(v) for v in (q1, q3, low, high))


def test_tukey_fences_on_a_constant_column_flag_nothing():
    """One value over every object means nothing in it can be an outlier."""
    q1, q3, low, high, note = tukey_fences([7.0] * 20)
    assert note == "constant"
    assert (q1, q3, low, high) == (7.0, 7.0, 7.0, 7.0)


# -- the spec ----------------------------------------------------------------

def test_a_support_fraction_inside_the_range_is_kept_as_a_float():
    """The MCD's support share is a user setting and must survive as given."""
    spec = OutlierSpec(features=("v",), method=METHOD_MAHALANOBIS,
                       support_fraction=1)
    assert spec.support_fraction == 1.0
    assert isinstance(spec.support_fraction, float)


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
def test_a_support_fraction_outside_the_range_is_refused(bad):
    """Outside (0, 1] the MCD has no objects to fit on, or too many."""
    with pytest.raises(OutlierError) as excinfo:
        OutlierSpec(features=("v",), support_fraction=bad)
    assert "support_fraction" in str(excinfo.value)


# -- reading the wells -------------------------------------------------------

def _result(**kwargs):
    base = dict(method=METHOD_MAD, features=("v",),
                scores=np.zeros(1), flags=np.zeros(1, dtype=bool),
                reasons=("",), threshold=3.5, n_rows_in=1, n_scored=1)
    base.update(kwargs)
    return OutlierResult(**base)


def test_a_well_pass_that_produced_nothing_is_said_in_the_headline():
    """"No well was scored" is a finding, not a blank."""
    result = _result(well_keys=("prc",))
    assert "no well was scored" in result.headline()


# -- the caveats -------------------------------------------------------------

def test_objects_that_could_not_be_scored_are_reported_as_still_present():
    """Neither flagged nor cleared, and never removed from the frame."""
    result = _result(n_rows_in=10, n_scored=6)
    text = " ".join(result.caveats())
    assert "could not" in text and "still in" in text


def test_infinities_are_reported_as_treated_missing():
    """An infinity from a ratio would be the outlier and the scale at once."""
    result = _result(n_non_finite=3)
    assert any("±inf" in line for line in result.caveats())


# -- the report --------------------------------------------------------------

def test_the_iqr_report_names_the_fence_it_used():
    """A flag without its fence cannot be checked against the data."""
    result = _result(method=METHOD_IQR, threshold=1.5,
                     centres={"v": 5.0}, fences={"v": (1.0, 9.0)})
    text = result.report()
    assert "iqr(c=1.5)" in result.method_label()
    assert "fence [1, 9]" in text


# -- per-feature scoring -----------------------------------------------------

# -- the multivariate rule ---------------------------------------------------
