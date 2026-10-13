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

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets import pivot_spec as PS
from spacr.qt.widgets.pivot_spec import (
    MEAN,
    N,
    PivotError,
    PivotSpec,
    pivot,
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

def test_a_table_with_no_rows_at_all_has_no_n_range():
    """``None`` is not ``(0, 0)``. A cell range of zero would read as "every
    cell holds nothing measured", which is a different claim from "there is
    no table here"."""
    empty = pd.DataFrame({"plateID": pd.Series(dtype=str),
                          "value": pd.Series(dtype=float)})
    result = pivot(empty, PivotSpec(rows=("plateID",), values=("value",)))
    assert result.n_range() is None
    assert result.shape[0] == 0


def test_a_row_axis_too_large_for_the_product_falls_back_to_what_occurs(
        monkeypatch):
    """Two keys with six levels each is thirty-six rows of which six exist.
    Drawing the product would fill the table with empty rows, so the observed
    combinations are used and the swap is announced."""
    monkeypatch.setattr(PS, "MAX_ROWS", 10)
    frame = pd.DataFrame({
        "plateID": [f"p{i}" for i in range(6)],
        "gene": [f"g{i}" for i in range(6)],
        "value": np.arange(6, dtype=float),
    })
    result = pivot(frame, PivotSpec(rows=("plateID", "gene"),
                                    values=("value",)))
    assert result.shape[0] == 6
    assert "would be the full grid" in result.notice
    assert result.hidden_rows == 0


def test_even_the_observed_rows_are_capped_and_the_cut_is_named(monkeypatch):
    """When what occurs is still too long the table is truncated. Both the
    cap and the source rows that fell outside it are reported, because the
    displayed numbers are then a subset of the frame."""
    monkeypatch.setattr(PS, "MAX_ROWS", 3)
    frame = pd.DataFrame({
        "plateID": [f"p{i}" for i in range(6)],
        "gene": [f"g{i}" for i in range(6)],
        "value": np.arange(6, dtype=float),
    })
    result = pivot(frame, PivotSpec(rows=("plateID", "gene"),
                                    values=("value",)))
    assert result.shape[0] == 3
    assert "capped at 3" in result.notice
    assert result.hidden_rows == 3
    summary = result.summary()
    assert "3 source row(s) outside the shown levels" in summary
    assert "capped at 3" in summary


def test_a_grid_over_the_cell_ceiling_loses_rows_not_columns(monkeypatch):
    """A column removed loses a whole series; a row removed loses one group,
    and the table is read down the page. The trim is announced either way."""
    monkeypatch.setattr(PS, "MAX_CELLS", 2)
    frame = pd.DataFrame({
        "plateID": ["p1", "p2", "p3"],
        "gene": ["a", "a", "a"],
        "value": [1.0, 2.0, 3.0],
    })
    result = pivot(frame, PivotSpec(rows=("plateID",), cols=("gene",),
                                    values=("value",)))
    assert result.shape == (2, 1)
    assert "grid capped at 2 cells" in result.notice


def test_an_axis_key_that_is_not_a_column_is_refused_by_name():
    """The refusal names the key and says how many columns the table does
    have, so the user can see it is the spec that is stale rather than the
    data that is empty."""
    with pytest.raises(PivotError, match="not a column of this table"):
        PS._levels_of(_frame(), "missing")


def test_a_missing_row_key_reaches_the_user_as_a_pivot_error():
    """Every bad spec is refused as a :class:`PivotError` carrying a sentence
    the panel can show, the commonest mistake -- a spec saved against another
    table -- included. The axis keys are checked before the label frame is
    built, so a row key the frame does not have is named rather than escaping
    as a bare ``KeyError`` from pandas."""
    with pytest.raises(PivotError, match="not a column of this table"):
        pivot(_frame(), PivotSpec(rows=("missing",), values=("value",)))


def test_a_table_with_no_row_keys_still_labels_its_single_row():
    """Exported with columns only, the one row needs a name. A blank leading
    cell in the CSV reads as a missing key rather than "all of it"."""
    result = pivot(_frame(), PivotSpec(cols=("gene",), values=("value",)))
    wide = result.to_frame()
    assert wide["rows"].tolist() == ["all"]
    assert result.shape[0] == 1
