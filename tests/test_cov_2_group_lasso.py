"""The penalty chooser's edges: a one-point path, and a path that selects nothing.

:func:`spacr.group_lasso.choose_lambda` exists because a shipped default
penalty emptied every gene block on a real screen. Two edges decide whether it
can still answer at all: a path with a single candidate has no cross-validation
to do, and a design in which no candidate keeps a gene must still hand back the
smallest penalty tried rather than nothing. Both are exercised here, because a
chooser that raised or returned ``None`` in either case would take the whole
regression down.
"""
from __future__ import annotations

import numpy as np

from spacr import group_lasso as gl


def test_a_single_candidate_path_is_returned_without_cross_validation():
    """One point on the path is the answer; there is nothing to compare it to.

    ``points=1`` makes :func:`penalty_path` yield the ceiling alone. Splitting
    wells to score a field of one would only cost time, so the chooser short
    circuits -- and must return that candidate as a float, not the array.
    """
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([1.0, -1.0])
    labels = ["g", "g"]

    only = gl.penalty_path(X, y, labels, points=1)
    assert only.size == 1

    chosen = gl.choose_lambda(X, y, labels, points=1)
    assert isinstance(chosen, float)
    assert chosen == float(only[0])


def test_a_path_that_selects_nothing_reaches_down_and_returns_the_smallest():
    """When every candidate empties the fit, the smallest penalty tried wins.

    A design with two wells cannot be cross-validated -- every fold leaves one
    well to train on -- so no candidate ever gets a finite held-out error and
    none of them can be chosen on merit. The chooser must then reach further
    down the decades and finally hand back the smallest penalty it reached,
    which the caller's own guard can refuse. Returning the ceiling instead
    would hand back a penalty already known to be empty.
    """
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([1.0, -1.0])
    labels = ["g", "g"]

    ceiling = float(gl.penalty_path(X, y, labels, points=3)[0])
    chosen = gl.choose_lambda(X, y, labels, folds=2, points=3)

    # Each extension multiplies the floor by 0.01, so the answer sits
    # PATH_EXTENSIONS decades of that below the deepest first-pass candidate.
    expected = ceiling * gl.PATH_DEPTH * (0.01 ** gl.PATH_EXTENSIONS)
    assert np.isclose(chosen, expected, rtol=1e-9)
    assert 0.0 < chosen < ceiling
