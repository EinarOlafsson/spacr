"""`_multiclass_metrics`: an empty split, and a one-hot guard it retired.

An empty validation split is a valid evaluator result. Its metrics are
undefined, its class schema is known, and no fabricated sample should be
introduced merely to make scikit-learn accept the call -- 1.7 rejects
empty arrays in `confusion_matrix`, which is why the function answers
that case itself.

That early answer is what makes the `if len(y_true):` guard further down
unreachable, and this file pins the two together.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from spacr.deep_spacr import _multiclass_metrics


class TestAnEmptyValidationSplit:

    def test_the_metrics_are_undefined_rather_than_zero(self):
        """NaN and 0.0 mean different things in a results table.

        Zero accuracy is a claim about a model. NaN says the split had
        nothing to measure, which is the truth.
        """
        out = _multiclass_metrics(np.array([], dtype=int),
                                  np.zeros((0, 3), dtype=float))
        for key in ("accuracy", "prauc", "f1_macro", "optimal_threshold"):
            assert np.isnan(out[key]), f"{key} should be undefined, not a value"

    def test_the_class_schema_survives_an_empty_split(self):
        """The number of classes is known even when no sample arrived.

        A downstream table keyed by class would otherwise lose its
        columns for a fold that happened to be empty.
        """
        out = _multiclass_metrics(np.array([], dtype=int),
                                  np.zeros((0, 4), dtype=float))
        assert out["per_class_accuracy"] == [0.0] * 4
        assert out["class_support"] == [0] * 4

    @pytest.mark.parametrize("classes", [1, 2, 5])
    def test_the_schema_follows_the_probability_matrix(self, classes):
        out = _multiclass_metrics(np.array([], dtype=int),
                                  np.zeros((0, classes), dtype=float))
        assert len(out["per_class_accuracy"]) == classes


class TestAnOrdinarySplit:

    def test_a_perfect_classifier_scores_one(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        prob = np.eye(3)[y_true].astype(float)
        out = _multiclass_metrics(y_true, prob)
        assert out["accuracy"] == pytest.approx(1.0)
        assert out["per_class_accuracy"] == pytest.approx([1.0, 1.0, 1.0])
        assert out["class_support"] == [2, 2, 2]

    def test_a_class_with_no_samples_scores_zero_not_nan(self):
        """`np.where(row_sums > 0, ...)` -- a class nobody presented.

        Zero is right here and NaN was right above: the split DID have
        samples, this class simply had none of them, so its accuracy is
        a real measurement of nothing rather than an unknown.
        """
        y_true = np.array([0, 0, 1])
        prob = np.eye(3)[y_true].astype(float)
        out = _multiclass_metrics(y_true, prob)
        assert out["class_support"][2] == 0
        assert out["per_class_accuracy"][2] == 0.0

    def test_a_wrong_classifier_scores_below_one(self):
        y_true = np.array([0, 0, 1, 1])
        prob = np.eye(2)[1 - y_true].astype(float)
        out = _multiclass_metrics(y_true, prob)
        assert out["accuracy"] == pytest.approx(0.0)


class TestTheOneHotGuardThatCannotFire:
    """`if len(y_true):` before the one-hot fill is never false.

    The function has already returned for an empty `y_true` at the top,
    so by the time the one-hot matrix is built there is always at least
    one row. The guard is a second defence against a case the early
    return has taken.

    Pinned rather than forced: reaching it would mean skipping the early
    return, which tests nothing about the function.
    """

    def test_the_empty_case_is_answered_before_the_one_hot_is_built(self):
        source = inspect.getsource(_multiclass_metrics)
        assert "if len(y_true) == 0:" in source, (
            "the empty-split early return has gone; the `if len(y_true):` "
            "guard below it may now be reachable")
        early = source.index("if len(y_true) == 0:")
        one_hot = source.index("y_true_oh = np.zeros(")
        assert early < one_hot, (
            "the one-hot matrix is now built before the empty split is "
            "answered")

    def test_every_non_empty_split_reaches_the_one_hot_fill(self):
        """So the guard is true whenever it is evaluated."""
        for n in (1, 2, 7):
            y_true = np.zeros(n, dtype=int)
            prob = np.ones((n, 2), dtype=float) / 2.0
            out = _multiclass_metrics(y_true, prob)
            assert not np.isnan(out["accuracy"]), (
                f"a split of {n} row(s) was treated as empty")
