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
import pytest

from spacr.qt.widgets import feature_rank as FR
from spacr.qt.widgets.feature_rank import ClassSummary, mutual_info_of


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


def test_a_binning_that_captures_no_observation_is_not_a_score(monkeypatch):
    """With no counts at all there is no joint distribution to compute, so
    the answer is NaN. Returning 0.0 would rank the feature as "measured and
    uninformative" rather than "not measurable here"."""
    monkeypatch.setattr(
        FR.np, "histogram",
        lambda a, bins=None, **kw: (np.zeros(len(bins) - 1, dtype=int), bins))
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([0.5, 1.5, 2.5, 3.5])
    assert np.isnan(mutual_info_of(a, b, bins=4))


def test_a_class_split_with_no_entropy_explains_nothing(monkeypatch):
    """Mutual information is normalised by the entropy of the class label.
    When every counted observation belongs to one class that entropy is zero;
    the answer is 0.0 rather than a division that yields NaN or infinity."""
    state = {"first": True}

    def _one_sided(a, bins=None, **kw):
        counts = np.zeros(len(bins) - 1, dtype=int)
        if state["first"]:
            counts[0] = 4
            state["first"] = False
        return counts, bins

    monkeypatch.setattr(FR.np, "histogram", _one_sided)
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([0.5, 1.5, 2.5, 3.5])
    assert mutual_info_of(a, b, bins=4) == 0.0


def test_a_real_split_still_scores_between_zero_and_one():
    """The guards above must not have replaced the ordinary answer: two
    separated classes score above zero and never above one."""
    a = np.zeros(50)
    b = np.ones(50)
    score = mutual_info_of(a, b, bins=4)
    assert 0.0 < score <= 1.0
