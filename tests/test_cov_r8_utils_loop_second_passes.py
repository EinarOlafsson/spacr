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
import pytest

from spacr import utils as U


class TestDeDuplicatingSuggestions:
    """`suggest_training_changes` de-duplicates while preserving order.

    Order matters: the suggestions are advice, and advice reordered is
    advice reprioritised.
    """

    @staticmethod
    def _suggestions(dst, rows):
        import pandas as pd

        train = dst / "train.csv"
        pd.DataFrame(rows).to_csv(train, index=False)
        return train

    def test_the_real_function_de_duplicates_its_own_output(self, tmp_path):
        """Driven through the real function on a real progress file."""
        import pandas as pd

        dst = tmp_path / "run"
        dst.mkdir()
        rows = [{"epoch": i, "accuracy": 0.5, "loss": 1.0,
                 "val_accuracy": 0.5, "val_loss": 1.0} for i in range(1, 31)]
        pd.DataFrame(rows).to_csv(dst / "train.csv", index=False)

        out = U.suggest_training_changes(str(dst))
        suggestions = out.get("suggestions", [])
        assert len(suggestions) == len(set(suggestions)), (
            "a suggestion was repeated; the de-duplication did not run over "
            "every item")


class TestMergingCellsThatShareAParasite:
    """`_merge_cells_based_on_parasite_overlap` walks a list of labels.

    The inner loop relabels every overlapping cell onto the first one.
    With a single overlapping label there is nothing to relabel and the
    loop never goes round -- so a merge that only ever joined the first
    pair would look correct.
    """

    @staticmethod
    def _three_cells_under_one_parasite():
        """Three touching cells, one parasite lying across all three."""
        cell = np.zeros((12, 30), dtype=np.uint16)
        cell[2:10, 1:9] = 1
        cell[2:10, 11:19] = 2
        cell[2:10, 21:29] = 3

        parasite = np.zeros((12, 30), dtype=np.uint16)
        parasite[4:8, 1:29] = 1          # spans all three cells

        empty = np.zeros((12, 30), dtype=np.uint16)
        return parasite, cell, empty, empty

    def test_three_cells_under_one_parasite_are_given_one_label(self):
        """The second and third passes through the relabel loop.

        Two cells take the inner loop round once; three take it round
        twice, which is the arc that was never covered.

        Asserted on `cell_mask`, which the merge rewrites IN PLACE --
        that is where `cell_mask[cell_mask == other] = first` lands. The
        returned mask is re-labelled by connected component afterwards,
        so three blocks that share a parasite but do not touch come back
        as three components again. Both facts are worth knowing, and the
        next test pins the second one.
        """
        parasite, cell, nuclei, organelle = \
            self._three_cells_under_one_parasite()
        U._merge_cells_based_on_parasite_overlap(
            parasite, cell, nuclei, organelle, overlap_threshold=1,
            perimeter_threshold=1)

        labels = {int(v) for v in np.unique(cell) if v}
        assert labels == {1}, (
            f"three cells sharing one parasite kept {len(labels)} labels; "
            "the relabel loop did not walk the whole list")

    def test_the_returned_mask_is_relabelled_by_connected_component(self):
        """So a merge of cells that do not touch still returns them apart.

        Recorded because it is surprising: the merge unifies the LABEL,
        and the return then renumbers by connectivity. Two cells that
        share a parasite but sit apart are one object by label and two
        by geometry, and the function answers with the geometry.
        """
        parasite, cell, nuclei, organelle = \
            self._three_cells_under_one_parasite()
        returned = U._merge_cells_based_on_parasite_overlap(
            parasite, cell, nuclei, organelle, overlap_threshold=1,
            perimeter_threshold=1)
        assert len({int(v) for v in np.unique(returned) if v}) == 3

    def test_two_touching_cells_come_back_as_one_object(self):
        """And when they DO touch, the merge is visible in the answer."""
        cell = np.zeros((12, 20), dtype=np.uint16)
        cell[2:10, 1:9] = 1
        cell[2:10, 9:17] = 2            # adjacent, sharing a border
        parasite = np.zeros((12, 20), dtype=np.uint16)
        parasite[4:8, 3:15] = 1
        empty = np.zeros((12, 20), dtype=np.uint16)

        returned = U._merge_cells_based_on_parasite_overlap(
            parasite, cell, empty, empty, overlap_threshold=1,
            perimeter_threshold=1)
        assert len({int(v) for v in np.unique(returned) if v}) == 1

    def test_cells_with_no_shared_parasite_are_left_apart(self):
        """The other side: a merge that joined everything would pass the
        test above and be badly wrong."""
        cell = np.zeros((12, 30), dtype=np.uint16)
        cell[2:10, 1:9] = 1
        cell[2:10, 21:29] = 2

        parasite = np.zeros((12, 30), dtype=np.uint16)
        parasite[4:8, 2:8] = 1           # inside cell 1 only

        empty = np.zeros((12, 30), dtype=np.uint16)
        merged = U._merge_cells_based_on_parasite_overlap(
            parasite, cell, empty, empty, overlap_threshold=1,
            perimeter_threshold=1000)

        labels = {int(v) for v in np.unique(merged) if v}
        assert len(labels) == 2, (
            "two cells sharing no parasite were merged anyway")


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
