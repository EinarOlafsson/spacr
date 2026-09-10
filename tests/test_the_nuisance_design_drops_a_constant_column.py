"""Building the nuisance design, and the columns that contribute nothing.

The design is an intercept plus block and nuisance fixed effects, and it must
be FULL RANK -- the function checks that and raises otherwise. Every arc here
is about not adding a column that would break that: a categorical with one
level contributes no dummy after ``drop_first``, and adding an empty block
would put a zero-width array into ``column_stack``.

A constant nuisance column is not contrived. A screen run on one plate has a
constant plate column, and every well shares it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _outcomes(**columns):
    return pd.DataFrame(columns)


def test_a_categorical_nuisance_with_one_level_adds_no_column():
    """Arc 202 -> 189: ``drop_first`` leaves nothing, so nothing is appended.

    One plate means one level, and the dummy frame is empty. Appending it
    would stack a zero-width array; keeping it out is what leaves the design
    full rank, which is the condition the function raises on.
    """
    from spacr.guide_permutation import _nuisance_design

    outcomes = _outcomes(block=["b1", "b1", "b2", "b2"],
                         plate=["plate1"] * 4)

    design = _nuisance_design(outcomes, "block", ["plate"])

    # Intercept plus one block dummy, and nothing for the constant plate.
    assert design.shape == (4, 2)
    assert np.linalg.matrix_rank(design) == design.shape[1]


def test_a_categorical_nuisance_with_two_levels_adds_one_column():
    """The taken side, so the omission above is visibly a decision.

    The plate must CROSS the blocks. A plate that lines up with the block one
    for one is collinear with it, and the rank check refuses that design --
    correctly, and it is a different behaviour from the one under test here.
    """
    from spacr.guide_permutation import _nuisance_design

    outcomes = _outcomes(block=["b1", "b1", "b2", "b2"],
                         plate=["plate1", "plate2", "plate1", "plate2"])

    design = _nuisance_design(outcomes, "block", ["plate"])

    assert design.shape[1] == 3
    assert np.linalg.matrix_rank(design) == design.shape[1]


def test_a_numeric_nuisance_is_taken_as_one_column():
    """The numeric branch, which needs no dummy coding at all."""
    from spacr.guide_permutation import _nuisance_design

    outcomes = _outcomes(block=["b1", "b1", "b2", "b2"],
                         cell_count=[100.0, 120.0, 90.0, 110.0])

    design = _nuisance_design(outcomes, "block", ["cell_count"])

    assert design.shape == (4, 3)


def test_a_numeric_nuisance_that_is_not_finite_is_refused_by_name():
    """The raise beside it: a NaN covariate silently drops rows downstream."""
    from spacr.guide_permutation import _nuisance_design

    outcomes = _outcomes(block=["b1", "b1", "b2", "b2"],
                         cell_count=[100.0, np.nan, 90.0, 110.0])

    with pytest.raises(ValueError, match="cell_count"):
        _nuisance_design(outcomes, "block", ["cell_count"])


def test_the_block_column_is_not_added_twice():
    """The ``continue``: naming the block among the nuisances is harmless.

    A settings file that lists the block column in both places is an ordinary
    mistake, and adding it twice would make the design rank-deficient and
    raise -- turning a harmless duplication into a stopped run.
    """
    from spacr.guide_permutation import _nuisance_design

    outcomes = _outcomes(block=["b1", "b1", "b2", "b2"])

    design = _nuisance_design(outcomes, "block", ["block"])

    assert design.shape == (4, 2)
    assert np.linalg.matrix_rank(design) == design.shape[1]


def test_an_absent_block_column_is_refused_by_name():
    """The guard above everything."""
    from spacr.guide_permutation import _nuisance_design

    with pytest.raises(ValueError, match="block"):
        _nuisance_design(_outcomes(other=["x"]), "block", [])
