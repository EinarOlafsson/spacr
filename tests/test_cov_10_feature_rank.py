"""Feature ranking when a statistic has nothing to work with.

The explorer ranks features by how far apart the two classes sit. The
branches here are the degenerate ones: a class level with no rows behind it,
a binning that puts no observation anywhere, and a class distribution that
carries no entropy for the mutual information to normalise by. Each has one
correct answer -- NaN for "cannot be computed", 0.0 for "explains nothing" --
and the two must not be swapped, because a NaN drops the feature out of the
ranking while a zero keeps it at the bottom of a real list.
"""
from __future__ import annotations

import numpy as np

from spacr.qt.widgets import feature_rank as FR
from spacr.qt.widgets.feature_rank import ClassSummary


def test_a_class_with_no_rows_is_summarised_as_absent_not_as_zero():
    """A level the user selected that has no objects behind it must not be
    drawn as a distribution centred on zero. n is 0 and every quantile is
    NaN, which is what stops the panel drawing a box at the origin."""
    values = np.array([1.0, 2.0, 3.0])
    keys = np.array(["a", "a", "a"], dtype=object)
    summaries = FR._summaries(values, keys, ["a", "b"])

    absent = [s for s in summaries if s.level == "b"][0]
    assert isinstance(absent, ClassSummary)
    assert absent.n == 0
    assert all(np.isnan(v) for v in (absent.median, absent.q25, absent.q75,
                                     absent.low, absent.high))
    present = [s for s in summaries if s.level == "a"][0]
    assert present.n == 3 and present.median == 2.0
