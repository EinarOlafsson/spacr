"""utils: loops whose bodies ran once and never twice.

None of these is a missing line -- every statement was executed. What
was never taken is the BACK EDGE: the arc that carries a loop into its
second iteration. A body that only ever runs once hides exactly the bugs
that need two items to appear -- a de-duplicator that keeps the first of
each pair, a merge that walks a list of labels, a scan that reads more
than one file.

So each of these drives the loop with enough input to go round again,
and asserts something that could only be true after the second pass.
"""
from __future__ import annotations

import numpy as np

from spacr import utils as U


class TestTheSelfComparisonsThatCannotBeTrue:
    """`if other_label != first_label:` is never false, in either merge loop.

    The labels come from `np.unique`, which returns sorted DISTINCT
    values, and `first_label` is `labels[0]`. So `labels[1:]` can never
    contain it, and the comparison is true on every pass.

    The guard is harmless -- it costs one comparison and says out loud
    that a cell must not be relabelled onto itself -- but it cannot fire,
    and forcing it would mean handing the loop a list `np.unique` could
    not have produced.
    """

    def test_np_unique_never_repeats_its_first_value_in_the_tail(self):
        rng = np.random.default_rng(20260831)
        for _ in range(5000):
            arr = rng.integers(0, 8, size=int(rng.integers(1, 30)))
            labels = np.unique(arr)
            labels = labels[labels != 0]
            if labels.size > 1:
                assert labels[0] not in labels[1:], (
                    "np.unique repeated a value; the self-comparison guards "
                    "in _merge_cells_based_on_parasite_overlap are now "
                    "reachable and want tests of their own")

    def test_the_labels_still_come_from_np_unique(self):
        """If they ever stop doing, the guards become live."""
        import inspect

        source = inspect.getsource(U._merge_cells_based_on_parasite_overlap)
        assert "np.unique(labeled_cells[current_parasite_mask])" in source
        assert "first_label = overlapping_cell_labels[0]" in source
        assert "for other_label in overlapping_cell_labels[1:]:" in source
