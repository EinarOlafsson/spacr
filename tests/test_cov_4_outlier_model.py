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
import pandas as pd
import pytest

from spacr.qt.widgets.outlier_model import (METHOD_IQR, METHOD_MAD,
                                            METHOD_MAHALANOBIS, OutlierError,
                                            OutlierResult, OutlierSpec,
                                            detect_outliers,
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


def test_a_result_with_no_well_pass_names_no_flagged_wells():
    """With the well pass off there are no wells to have flagged."""
    result = _result()
    assert result.flagged_wells() == ()
    assert result.unscored_wells() == ()


def test_a_one_column_well_key_yields_plain_names_not_one_tuples():
    """``prc`` is already the whole name of the well."""
    wells = pd.DataFrame({"prc": ["p1_r1_c1", "p1_r1_c2"],
                          "well_outlier": [True, False],
                          "well_scored": [True, False]})
    result = _result(wells=wells, well_keys=("prc",))
    assert result.flagged_wells() == ("p1_r1_c1",)
    assert result.unscored_wells() == ("p1_r1_c2",)


def test_a_multi_column_well_key_yields_tuples():
    """Three columns name one well, and all three have to come back."""
    wells = pd.DataFrame({"plateID": ["p1"], "rowID": ["A"],
                          "columnID": ["1"], "well_outlier": [True],
                          "well_scored": [False]})
    result = _result(wells=wells,
                     well_keys=("plateID", "rowID", "columnID"))
    assert result.flagged_wells() == (("p1", "A", "1"),)
    assert result.unscored_wells() == (("p1", "A", "1"),)


def test_a_well_table_that_lost_its_key_columns_names_no_wells():
    """A summary rebuilt without its keys cannot say which well it means."""
    wells = pd.DataFrame({"well_outlier": [True], "well_scored": [False]})
    result = _result(wells=wells, well_keys=("prc",))
    assert result.flagged_wells() == ()
    assert result.unscored_wells() == ()


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

def test_a_tied_column_under_the_fence_rule_scores_every_object_zero():
    """Zero spread must not divide; nothing in a tied column is an outlier."""
    frame = pd.DataFrame({"v": [7.0] * 20})
    result = detect_outliers(
        frame, OutlierSpec(features=("v",), method=METHOD_IQR,
                           per_well=False))
    assert result.n_flagged == 0
    assert np.allclose(result.scores, 0.0)


def test_a_feature_with_no_finite_value_is_noted_by_name():
    """Silently scoring nothing would read as "this column is clean"."""
    frame = pd.DataFrame({"v": [1.0, 2.0, 3.0, 40.0],
                          "empty": [np.nan] * 4})
    result = detect_outliers(
        frame, OutlierSpec(features=("v", "empty"), per_well=False))
    assert any("'empty'" in note and "no finite value" in note
               for note in result.notes), result.notes


def test_objects_scored_on_only_some_features_are_counted_and_noted():
    """Scoring on what an object has is a choice, and has to be visible."""
    frame = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "b": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]})
    result = detect_outliers(
        frame, OutlierSpec(features=("a", "b"), per_well=False))
    assert any("scored on the features they have" in note
               for note in result.notes), result.notes


# -- the multivariate rule ---------------------------------------------------

def test_a_covariance_that_cannot_be_inverted_is_refused_with_the_way_out(
        monkeypatch):
    """A singular covariance has no inverse and so no distance."""
    import sklearn.covariance as cov

    class _Singular:
        def __init__(self, **_kwargs):
            pass

        def fit(self, _matrix):
            raise np.linalg.LinAlgError("singular matrix")

    monkeypatch.setattr(cov, "MinCovDet", _Singular)
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(size=40),
                          "b": rng.normal(size=40)})
    with pytest.raises(OutlierError) as excinfo:
        detect_outliers(frame, OutlierSpec(features=("a", "b"),
                                           method=METHOD_MAHALANOBIS,
                                           per_well=False))
    message = str(excinfo.value)
    assert "collinear" in message
    assert "PCA" in message


def test_an_object_missing_a_feature_is_told_why_it_was_not_scored():
    """The multivariate distance needs every tested feature."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"a": rng.normal(size=40),
                          "b": rng.normal(size=40)})
    frame.loc[0, "b"] = np.nan
    result = detect_outliers(frame, OutlierSpec(features=("a", "b"),
                                                method=METHOD_MAHALANOBIS,
                                                per_well=False))
    assert "needs a finite value" in result.reasons[0]


def test_a_well_with_no_median_for_every_feature_is_told_why(monkeypatch):
    """A well missing one feature entirely cannot enter a joint distance."""
    rng = np.random.default_rng(5)
    rows = []
    for well in range(8):
        for _ in range(25):
            rows.append({
                "plateID": "p1", "rowID": "A", "columnID": str(well),
                "a": float(rng.normal()),
                "b": np.nan if well == 3 else float(rng.normal()),
            })
    frame = pd.DataFrame(rows)
    result = detect_outliers(
        frame,
        OutlierSpec(features=("a", "b"), method=METHOD_MAHALANOBIS,
                    per_well=True))
    reasons = list(result.wells["well_outlier_reason"])
    assert any("no median for every feature" in reason
               for reason in reasons), reasons
