"""The pivot engine when the table cannot be the whole table.

A pivot is read as if it were a summary of everything in the frame. The
branches here are the ones that make that untrue -- an axis whose cartesian
product is too large to draw, a grid capped by its cell count, source rows
whose keys fall outside what is shown -- plus the small accessors a panel
uses to describe the spec back to the user. Every one of them either prints a
notice or hands back a name; a table that trimmed itself in silence is the
one failure this module exists to prevent.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.qt.widgets import pivot_spec as PS
from spacr.qt.widgets.pivot_spec import (
    MEAN,
    N,
    PivotError,
    PivotSpec,
)


def _frame():
    return pd.DataFrame({
        "plateID": ["p1", "p1", "p2", "p2"],
        "gene": ["a", "b", "a", "b"],
        "value": [1.0, 2.0, 3.0, 4.0],
    })


# ---------------------------------------------------------------------------
# Editing and describing a spec
# ---------------------------------------------------------------------------

def test_every_edit_returns_a_new_spec_and_leaves_the_old_one_alone():
    """Each drag onto an axis is one edit. A spec that mutated in place would
    make "undo the last drag" impossible and would change a spec another
    panel is still holding."""
    base = PivotSpec(rows=("plateID",))
    rows = base.with_rows(["gene"])
    cols = base.with_cols(["gene"])
    values = base.with_values(["value"])
    aggs = base.with_aggs([MEAN])

    assert base.rows == ("plateID",)
    assert rows.rows == ("gene",)
    assert cols.cols == ("gene",)
    assert values.values == ("value",)
    assert aggs.aggs == (N, MEAN)


def test_a_spec_names_every_column_it_will_read_once():
    """The reader asks this before it queries; a column listed twice would be
    fetched twice, and one omitted would come back missing."""
    spec = PivotSpec(rows=("plateID",), cols=("gene",),
                     values=("value", "value"))
    assert spec.used_columns() == ("plateID", "gene", "value")


def test_a_spec_describes_both_axes_and_its_cells():
    """The sentence under the table is how a reader checks that the picture
    is the one they asked for."""
    spec = PivotSpec(rows=("plateID",), cols=("gene",), values=("value",),
                     aggs=(MEAN,))
    text = spec.describe()
    assert "rows: plateID" in text
    assert "columns: gene" in text
    assert "mean(value)" in text


def test_a_contingency_spec_describes_its_cells_as_a_count():
    """With no value column the cells are counts, and the description has to
    say so rather than naming a statistic of nothing."""
    assert "cells: n" in PivotSpec(rows=("plateID",)).describe()


# ---------------------------------------------------------------------------
# Results that are empty, capped, or partial
# ---------------------------------------------------------------------------


def test_an_axis_key_that_is_not_a_column_is_refused_by_name():
    """The refusal names the key and says how many columns the table does
    have, so the user can see it is the spec that is stale rather than the
    data that is empty."""
    with pytest.raises(PivotError, match="not a column of this table"):
        PS._levels_of(_frame(), "missing")
