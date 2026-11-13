"""A Walk that is interrupted, resumed, or given a fit function that fails.

A hyperparameter Walk is hours of fits, so every one of these is a real
operating condition rather than an edge case: a machine that was rebooted, a
checkpoint written by an older build, a configuration UMAP refuses, a round in
which nothing succeeded. What is pinned here is that none of them costs the
work already done, and none of them ends the search silently.
"""
from __future__ import annotations

import pytest

from spacr.hyperparam import WalkAxis, walk_search

# ---------------------------------------------------------------------------
# A fit function that will not cooperate
# ---------------------------------------------------------------------------


def test_a_cancelled_walk_is_cancelled_and_not_recorded_as_a_failure(
        tmp_path):
    """``PipelineCancelled`` is the user pressing Stop.

    Catching it as a trial error would turn a deliberate stop into a table
    full of red rows and let the walk carry on to the next configuration.
    """
    from spacr.qt.bridge import PipelineCancelled

    def _stopped(_params):
        raise PipelineCancelled("stopped by the user")

    with pytest.raises(PipelineCancelled):
        walk_search(_stopped, {"n_neighbors": 10},
                    [WalkAxis("n_neighbors", step=5.0, integer=True)],
                    n_trials=2)
