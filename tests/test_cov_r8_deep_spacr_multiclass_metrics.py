"""`_multiclass_metrics`: an empty split and its unconditional one-hot fill.

An empty validation split is a valid evaluator result. Its metrics are
undefined, its class schema is known, and no fabricated sample should be
introduced merely to make scikit-learn accept the call -- 1.7 rejects
empty arrays in `confusion_matrix`, which is why the function answers
that case itself.

That early answer makes every split reaching the one-hot fill non-empty, so
the redundant second guard is removed and this file pins the two together.
"""
from __future__ import annotations

import inspect


from spacr.deep_spacr import _multiclass_metrics


class TestTheUnconditionalOneHotFill:
    """The one-hot fill is reached only for non-empty labels.

    The function has already returned for an empty `y_true` at the top,
    so by the time the one-hot matrix is built there is always at least
    one row. The guard is a second defence against a case the early
    return has taken.

    Pinned rather than forced: reaching it would mean skipping the early
    return, which tests nothing about the function. The duplicate guard was
    therefore removed rather than excluded from coverage.
    """

    def test_the_empty_case_is_answered_before_the_one_hot_is_built(self):
        source = inspect.getsource(_multiclass_metrics)
        assert "if len(y_true) == 0:" in source, (
            "the empty-split early return has gone, so the unconditional "
            "one-hot fill may now receive no labels")
        early = source.index("if len(y_true) == 0:")
        one_hot = source.index("y_true_oh = np.zeros(")
        assert early < one_hot, (
            "the one-hot matrix is now built before the empty split is "
            "answered")
        assert "if len(y_true):" not in source

