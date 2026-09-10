"""The sentence beside a flagged object, and what the spec keeps.

A flag with no reason is a number a user cannot argue with, so every flagged
row carries one sentence naming the features that put it over the line. The
sentence is built only for the rows that need one, which is why the builder
is a free function driven directly here rather than through a whole scan.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.qt.widgets.outlier_model import (
    METHOD_IQR, METHOD_MAD, OutlierSpec, _reasons_per_feature,
)

FEATURES = ["area", "perimeter", "intensity"]
FENCES = {"area": (10.0, 100.0), "perimeter": (5.0, 20.0),
          "intensity": (0.0, 1.0)}


def _no_flags(n):
    return np.zeros(n, dtype=bool)


def test_a_flagged_object_is_told_which_feature_flagged_it():
    """The z-score is quoted with the feature and the side, because "this
    object is an outlier" is not a fact anyone can check."""
    spec = OutlierSpec(features=tuple(FEATURES), method=METHOD_MAD, k=3.5)
    scores = np.array([[9.4, 0.2, 0.1]])
    matrix = np.array([[900.0, 12.0, 0.5]])
    reasons = _reasons_per_feature(scores, matrix, FEATURES,
                                   np.array([True]), _no_flags(1), spec,
                                   FENCES)
    assert reasons == ["area high (z=9.4)"]


def test_a_value_below_its_fence_is_named_as_low():
    spec = OutlierSpec(features=tuple(FEATURES), method=METHOD_MAD, k=3.5)
    scores = np.array([[8.0, 0.2, 0.1]])
    matrix = np.array([[0.5, 12.0, 0.5]])
    reasons = _reasons_per_feature(scores, matrix, FEATURES,
                                   np.array([True]), _no_flags(1), spec,
                                   FENCES)
    assert reasons == ["area low (z=8.0)"]


def test_at_most_three_features_are_named_worst_first():
    """A row that is odd in eleven features would otherwise produce a
    sentence nobody reads to the end."""
    features = [f"f{i}" for i in range(5)]
    spec = OutlierSpec(features=tuple(features), method=METHOD_MAD, k=3.0)
    scores = np.array([[4.0, 9.0, 3.5, 12.0, 6.0]])
    matrix = np.array([[100.0] * 5])
    fences = {name: (0.0, 1.0) for name in features}
    reasons = _reasons_per_feature(scores, matrix, features, np.array([True]),
                                   _no_flags(1), spec, fences)
    assert reasons[0] == ("f3 high (z=12.0), f1 high (z=9.0), "
                          "f4 high (z=6.0)")


def test_the_fence_test_says_how_far_past_the_quartile_it_went():
    """Tukey's fence has no z-score, so the sentence quotes the multiple of
    the IQR instead -- the number the method's own threshold is in."""
    spec = OutlierSpec(features=tuple(FEATURES), method=METHOD_IQR, c=1.5)
    scores = np.array([[2.75, 0.1, 0.0]])
    matrix = np.array([[900.0, 12.0, 0.5]])
    reasons = _reasons_per_feature(scores, matrix, FEATURES,
                                   np.array([True]), _no_flags(1), spec,
                                   FENCES)
    assert reasons == ["area high (2.75·IQR past the quartile)"]


def test_a_row_with_no_usable_feature_says_it_was_not_scored():
    """Not scored is not the same as clean, and a blank cell would read as
    clean."""
    spec = OutlierSpec(features=tuple(FEATURES), method=METHOD_MAD, k=3.5)
    scores = np.full((2, 3), np.nan)
    matrix = np.full((2, 3), np.nan)
    reasons = _reasons_per_feature(scores, matrix, FEATURES, _no_flags(2),
                                   np.array([False, True]), spec, FENCES)
    assert reasons[0] == ""
    assert reasons[1] == "not scored: no finite value for any tested feature"


def test_a_flag_no_feature_accounts_for_gets_no_invented_sentence():
    """The flags a scan produces always come from these same scores, so this
    only happens when a caller supplies its own. It has to leave the row
    without a reason rather than name the least innocent feature it can
    find -- a sentence that says 'area high' about a value inside its fence
    is worse than no sentence.
    """
    spec = OutlierSpec(features=tuple(FEATURES), method=METHOD_MAD, k=3.5)
    # The unaccounted-for row comes FIRST, so a row it skipped cannot be the
    # last one and the rows after it still get their sentences.
    scores = np.array([[0.3, 0.2, 0.1], [9.4, 0.2, 0.1]])
    matrix = np.array([[40.0, 12.0, 0.5], [900.0, 12.0, 0.5]])
    reasons = _reasons_per_feature(scores, matrix, FEATURES,
                                   np.array([True, True]), _no_flags(2), spec,
                                   FENCES)
    assert reasons[0] == ""
    assert reasons[1] == "area high (z=9.4)"


def test_only_flagged_rows_pay_for_a_sentence():
    """One formatted string per object over 200,000 objects is a second of
    wall clock for text nobody reads."""
    spec = OutlierSpec(features=tuple(FEATURES), method=METHOD_MAD, k=3.5)
    scores = np.tile(np.array([[9.4, 0.2, 0.1]]), (4, 1))
    matrix = np.tile(np.array([[900.0, 12.0, 0.5]]), (4, 1))
    flags = np.array([False, True, False, False])
    reasons = _reasons_per_feature(scores, matrix, FEATURES, flags,
                                   _no_flags(4), spec, FENCES)
    assert reasons == ["", "area high (z=9.4)", "", ""]


# --------------------------------------------------------------------------
# What the spec keeps


def test_a_blank_feature_or_well_key_is_dropped():
    """The panel hands over one entry per row of its list, and a row the user
    cleared is an empty string -- which as a column name matches nothing and
    would be reported as a missing feature."""
    spec = OutlierSpec(features=("area", "", "area", "perimeter"),
                       well_keys=("plate", "", "well", "plate"))
    assert spec.features == ("area", "perimeter")
    assert spec.well_keys == ("plate", "well")


def test_a_spec_read_back_without_a_feature_list_keeps_its_defaults():
    """A settings file written by an older build has fewer keys, and the
    fields it does not carry have to fall back rather than become empty
    tuples of a different type."""
    spec = OutlierSpec.from_dict({"method": METHOD_IQR, "c": 2.0,
                                  "ignored": "not a field"})
    assert spec.method == METHOD_IQR
    assert spec.c == 2.0
    assert spec.features == ()
    assert spec.well_keys == ()
    assert spec.threshold() == 2.0


def test_a_spec_read_back_with_feature_lists_makes_them_tuples():
    """JSON gives lists back; the spec is frozen and hashable and needs
    tuples."""
    spec = OutlierSpec.from_dict({"features": ["area", "perimeter"],
                                  "well_keys": ["plate", "well"]})
    assert spec.features == ("area", "perimeter")
    assert spec.well_keys == ("plate", "well")
