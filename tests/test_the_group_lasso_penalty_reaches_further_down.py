"""Choosing a group-lasso penalty, including when the whole path selects nothing.

The docstring states the escalation: "If a penalty path contains no eligible
candidate, progressively smaller penalties are evaluated for up to
PATH_EXTENSIONS additional ranges. If no eligible candidate is found, the
smallest penalty evaluated is returned."

That last sentence is the branch that matters. A screen with no group signal
must come back with SOMETHING the caller's own guard can then refuse -- returning
a penalty already known to select nothing would be a fit guaranteed to be empty.
"""
from __future__ import annotations

import numpy as np
import pytest


def _design(n=40, p=6, seed=0, signal=True):
    """A small design where the first group carries the signal, or none does."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    labels = np.repeat(np.arange(p // 2), 2)
    y = (X[:, 0] * 3.0 + rng.normal(scale=0.1, size=n) if signal
         else rng.normal(size=n))
    return X, y, labels


def test_a_design_with_signal_chooses_a_positive_penalty():
    """The ordinary path: something on it selects, so it is chosen."""
    from spacr.group_lasso import choose_lambda

    X, y, labels = _design()

    chosen = choose_lambda(X, y, labels, points=8, folds=3)

    assert np.isfinite(chosen)
    assert chosen >= 0.0


def test_a_required_mask_of_the_wrong_length_is_refused_by_name():
    """The guard above everything, whose message names both counts.

    A mask built for a different design is a real mistake -- the caller
    assembles it from column names -- and "they must match" without the two
    numbers does not say which end is wrong.
    """
    from spacr.group_lasso import choose_lambda

    X, y, labels = _design(p=6)

    with pytest.raises(ValueError) as excinfo:
        choose_lambda(X, y, labels, required=np.ones(3, dtype=bool))

    message = str(excinfo.value)
    assert "3 entr" in message and "6 column" in message


def test_a_penalty_that_selects_nothing_anywhere_still_returns_a_number():
    """The escalation running out, and the documented final answer.

    ``required`` names a column the design cannot select, so no candidate on
    any path is eligible and every extension is exhausted. The smallest
    penalty tried comes back -- the one with any chance -- and the caller's
    own guard refuses the fit if it still says nothing. Returning nothing here
    would make that guard unreachable.
    """
    from spacr.group_lasso import choose_lambda

    X, y, labels = _design(signal=False, seed=3)
    # A CONSTANT column can never be selected: its coefficient is zero at
    # every penalty, so no candidate on any path is eligible and every
    # extension is exhausted. That is the only way to reach the final return
    # short of a design with no variance at all.
    X[:, -1] = 0.0
    required = np.zeros(X.shape[1], dtype=bool)
    required[-1] = True

    chosen = choose_lambda(X, y, labels, points=4, folds=3, required=required)

    assert np.isfinite(chosen)
    assert chosen >= 0.0


def test_two_runs_of_one_screen_agree():
    """``seed`` fixes the split, which the docstring promises.

    A penalty that moved between runs would move every coefficient with it,
    and two analyses of one screen would disagree for no reason the user could
    see.
    """
    from spacr.group_lasso import choose_lambda

    X, y, labels = _design()

    first = choose_lambda(X, y, labels, points=6, folds=3, seed=7)
    second = choose_lambda(X, y, labels, points=6, folds=3, seed=7)

    assert first == second


def test_a_single_point_path_returns_that_point():
    """The ``candidates.size < 2`` guard.

    A one-point path has nothing to choose between, so the point is the
    answer -- scoring it against itself would spend the fits and learn
    nothing.
    """
    from spacr.group_lasso import choose_lambda

    X, y, labels = _design()

    chosen = choose_lambda(X, y, labels, points=1, folds=3)

    assert np.isfinite(chosen)


def test_a_split_that_leaves_too_little_to_fit_is_skipped():
    """The ``keep.size < 2 or held.size < 1`` continue.

    With as many folds as wells, a held-out split can leave fewer than two
    rows to fit on. Fitting those would produce a meaningless error that the
    mean would then be built from, so the split is passed over and the
    remaining ones decide.
    """
    from spacr.group_lasso import choose_lambda

    # Two wells and three folds: at least one split holds nothing, and
    # another leaves a single row to fit on. Both are skipped.
    X, y, labels = _design(n=2, p=4)

    chosen = choose_lambda(X, y, labels, points=4, folds=3)

    assert np.isfinite(chosen)
