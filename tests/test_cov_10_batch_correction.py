"""Batch correction on the inputs that must not be silently absorbed.

Every branch below decides whether a plate effect is removed, is refused, or
is removed together with the biology. The cases are the ones a tidy frame
never produces: a control specification that arrived as bytes or as a nested
list, metadata whose index does not line up with the features, a covariate
column that never varies, a batch with one constant feature, and a control
column whose values cannot be hashed. Each of them has a defined answer, and
a correction that guessed instead would report a cleaner diagnostic on a
worse table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spacr import batch_correction as bc


def test_a_control_value_that_is_not_comparable_does_not_abort_the_match():
    """A nested list among the control values cannot be compared elementwise
    against the column. The other encodings still have to be tried, so a
    usable value sitting beside the unusable one still matches."""
    series = pd.Series(["neg", "trt", "neg"])
    mask = bc._match_values(series, [["neg", "trt"], "neg"])
    assert list(mask) == [True, False, True]


def test_a_numerically_encoded_control_matches_a_numeric_column():
    """Control columns are often 0/1. Matching only on text would miss the
    row whose value is the integer 0 rather than the string "0"."""
    series = pd.Series([0, 1, 0])
    mask = bc._match_values(series, [0])
    assert list(mask) == [True, False, True]


def test_a_spread_that_cannot_be_measured_is_reported_as_unknown():
    """The centroid diagnostic must never take down a correction that
    succeeded. A frame whose columns cannot be averaged has no measurable
    spread, and the answer is ``None`` -- absent, not zero, which would read
    as "the batches already agree"."""
    text_only = pd.DataFrame({"f1": ["x", "y", "x", "y"]})
    assert bc._centroid_spread(text_only, pd.Series(["A", "A", "B", "B"])) is None


def test_prior_shape_and_scale_are_infinite_when_there_is_no_variance():
    """With every per-feature dispersion identical the method of moments has
    no variance to divide by. An infinite prior shrinks each estimate all the
    way to the pooled value, which is the correct limit -- a division by zero
    would be a NaN in the corrected table."""
    flat = np.array([2.0, 2.0, 2.0, 2.0])
    assert bc._a_prior(flat) == np.inf
    assert bc._b_prior(flat) == np.inf
