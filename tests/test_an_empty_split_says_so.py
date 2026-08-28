"""An empty split is refused in words, not by sklearn (issue #110)."""
from __future__ import annotations

import numpy as np
import pytest

from spacr.classifier_evaluation import grouped_split


def test_no_labelled_objects_names_the_cause():
    """The filed traceback was sklearn's, three frames below anything a user
    recognises, naming neither the setting nor what to do about it."""
    with pytest.raises(ValueError) as caught:
        grouped_split([], [], 0.2, group_by="well")

    said = str(caught.value)
    assert "no labelled objects" in said
    # It has to say what to look at, or it is the old error with new words.
    assert "positive_control" in said and "negative_control" in said
    # And it must not be sklearn's.
    assert "n_samples=0" not in said


def test_the_named_holdout_path_is_covered_too():
    """That path divides by len(y), so an empty array is a ZeroDivisionError
    there rather than the ValueError -- both are refused before either."""
    with pytest.raises(ValueError) as caught:
        grouped_split([], [], 0.2, group_by="plate", hold_out_groups=["p1"])
    assert "no labelled objects" in str(caught.value)


def test_mismatched_group_and_label_counts_are_refused():
    """A mismatch misaligns the two and produces a split that looks valid."""
    with pytest.raises(ValueError) as caught:
        grouped_split(["a", "b"], [0, 1, 0], 0.2, group_by="well")
    assert "one to one" in str(caught.value)


def test_a_real_split_is_unaffected():
    """The guard must refuse only the degenerate shape."""
    groups = np.array([f"w{i // 4}" for i in range(40)], dtype=object)
    labels = np.array([0, 1] * 20)
    train_idx, test_idx, report = grouped_split(
        groups, labels, 0.25, seed=0, group_by="well")

    assert len(train_idx) > 0 and len(test_idx) > 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(train_idx) + len(test_idx) == len(labels)
    # Both classes on both sides, which is what the grouped split promises.
    assert set(labels[train_idx]) == {0, 1}
    assert set(labels[test_idx]) == {0, 1}
    assert report.total_groups == 10
