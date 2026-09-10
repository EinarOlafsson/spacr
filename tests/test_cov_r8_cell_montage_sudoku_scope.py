"""`_sudoku_calls` narrowing to a guide's wells, and a guard that cannot fire.

The Sudoku assignment gives one guide name per cell. Before it runs, the
screen is narrowed to the wells that hold this guide plus the wells that
anchor its rivals -- because a posterior is a comparison, and there is
nothing to compare a lone guide against.

Every refusal along the way SAYS why, in the run's notes. A montage that
silently produced nothing would leave the user with no idea which of
four conditions was not met.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from spacr import cell_montage as CM


def _work(wells=("A01", "A01", "A02"), scores=(0.9, 0.8, 0.7)):
    """Cells with a score, a measurement, and a well key."""
    return pd.DataFrame({
        "plateID": ["p1"] * len(wells),
        "rowID": [w[0] for w in wells],
        "columnID": [w[1:] for w in wells],
        "cell_area": np.linspace(100.0, 200.0, len(wells)),
        "score": list(scores),
    })


KEYS = ("plateID", "rowID", "columnID")


class TestTheRefusalsThatSayWhy:

    def test_no_scored_cell_is_reported(self):
        work = _work(scores=(np.nan, np.nan, np.nan))
        notes: list = []
        result = CM._sudoku_calls(work, pd.DataFrame(), KEYS, "guide",
                                  "fraction", "score", "guide_x", notes)
        assert result is None
        assert any("no cell carries a classification score" in n
                   for n in notes), notes

    def test_no_well_with_a_guide_fraction_is_reported(self):
        """Without fractions there is nothing to constrain the propagation."""
        notes: list = []
        result = CM._sudoku_calls(_work(), pd.DataFrame(), KEYS, "guide",
                                  "fraction", "score", "guide_x", notes)
        assert result is None
        assert any("no well has a guide fraction" in n for n in notes), notes

    def test_a_guide_in_none_of_these_wells_is_reported_by_name(self):
        """And the note says what it looked for, so a mis-typed guide
        identifier is findable rather than mysterious."""
        counts = pd.DataFrame({
            "plateID": ["p1", "p1"],
            "rowID": ["A", "A"],
            "columnID": ["01", "02"],
            "guide": ["other_guide", "other_guide"],
            "fraction": [1.0, 1.0],
        })
        notes: list = []
        result = CM._sudoku_calls(_work(), counts, KEYS, "guide",
                                  "fraction", "score", "guide_x", notes)
        assert result is None
        assert any("is in none of these wells" in n for n in notes), notes
        assert any("guide_x" in n for n in notes), notes


class TestTheEmptyRowSelectionThatCannotHappen:
    """`if not rows: ... return None` is unreachable.

    `fractions` is built by walking `set(wells)` -- the wells the CELLS
    are in -- and keeping only those with a guide fraction. So every key
    in `fractions` is a well some cell occupies.

    `mine` is drawn from `fractions`, `keep` is `mine` union the
    anchoring wells (also from `fractions`), and `rows` selects the
    cells whose well is in `keep`. Since `mine` is non-empty by the
    guard above it, at least one cell always matches.

    Pinned from the producing side: forcing it would mean handing the
    function a `fractions` map it could not have built.
    """

    def test_the_fractions_are_keyed_by_the_cells_own_wells(self):
        source = inspect.getsource(CM._sudoku_calls)
        assert "for label in sorted(set(wells)):" in source, (
            "the fractions are no longer built from the cells' own wells, "
            "so the empty-row guard below may now be reachable")
        assert "rows = [i for i, w in enumerate(wells) if w in keep]" in source

    def test_a_guide_present_in_a_cells_well_selects_that_cell(self):
        """The live path, which is what makes the guard unreachable.

        The guide is in A01, cells are in A01 and A02, so the selection
        is non-empty -- and it is narrowed, not the whole screen.
        """
        counts = pd.DataFrame({
            "plateID": ["p1", "p1"],
            "rowID": ["A", "A"],
            "columnID": ["01", "02"],
            "guide": ["guide_x", "other_guide"],
            "fraction": [1.0, 1.0],
        })
        notes: list = []
        CM._sudoku_calls(_work(), counts, KEYS, "guide", "fraction",
                         "score", "guide_x", notes)
        assert any("well(s) hold guide_x" in n for n in notes), notes
