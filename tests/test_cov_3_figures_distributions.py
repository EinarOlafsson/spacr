"""A response with no well column is counted as it was given.

The histogram collapses a per-well response to one value per well so a
fifteen-guide well is not counted fifteen times. That collapse depends on
knowing which column names the well, and a table that does not carry one --
or that names a column the frame does not have -- must fall back to the raw
values and SAY it did not deduplicate, rather than guessing a well column
and silently changing n.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spacr.figures.distributions import one_value_per_well


def _frame():
    return pd.DataFrame({
        "well": ["A01", "A01", "A02", "A02", "A02"],
        "log_pred": [1.0, 1.0, 2.0, 2.0, 2.0],
    })


def test_no_well_column_named_leaves_every_row_counted():
    """`well=None` is the headless caller that has no well information. It
    must report deduplicated=False so the caption does not claim a per-well
    n it never computed."""
    values, deduplicated = one_value_per_well(_frame(), "log_pred", None)

    assert deduplicated is False
    assert values.size == 5
    assert sorted(np.asarray(values)) == [1.0, 1.0, 2.0, 2.0, 2.0]


def test_a_well_column_the_table_lacks_is_the_same_fallback():
    """A stale column name from a previous table must not raise KeyError in
    the middle of drawing a figure."""
    values, deduplicated = one_value_per_well(_frame(), "log_pred", "wellID")

    assert deduplicated is False
    assert values.size == 5


def test_a_genuine_per_well_response_is_collapsed_to_one_row_each():
    """The contrast that makes the fallback meaningful: when the well column
    is present and the response is constant within a well, n drops to the
    number of wells."""
    values, deduplicated = one_value_per_well(_frame(), "log_pred", "well")

    assert deduplicated is True
    assert values.size == 2
