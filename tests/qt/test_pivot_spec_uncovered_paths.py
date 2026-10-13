"""What a pivot says when a part of it is absent.

The caption under a pivot table is the only place a reader is told how many
cells hold data, how big the smallest one is, and which of them are too small
to read. Each of those clauses has to disappear cleanly when there is nothing
to say -- a table of nothing that still claims an n range would be a caption
inventing its own numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.pivot_spec import (
    LOW_N, MEAN, N, PivotSpec, _clean, pivot,
)


# --------------------------------------------------------------------------
# Axis keys


def test_a_blank_or_missing_axis_key_is_dropped():
    """A pivot keyed on '' would group every row into one unnamed level, and
    a None key raises inside pandas -- neither is what a half-filled form
    means."""
    assert _clean(["plate", None, "", "plate", "gene"]) == ("plate", "gene")
    assert _clean([None, ""]) == ()
    assert _clean(None) == ()


def test_the_spec_drops_blanks_from_every_axis_it_is_given():
    """The form hands over one entry per combo box, and an untouched combo
    box is an empty string."""
    spec = PivotSpec(rows=["plate", "", None], cols=["", "gene"],
                     values=["v", ""])
    assert spec.rows == ("plate",)
    assert spec.cols == ("gene",)
    assert spec.values == ("v",)


def test_a_table_with_no_row_key_describes_only_its_columns():
    """One row of totals across the columns is a legitimate pivot, and its
    description must not open with an empty 'rows:' clause."""
    described = PivotSpec(cols=("gene",), values=("v",),
                          aggs=(N, MEAN)).describe()
    assert described.startswith("columns: gene")
    assert "rows:" not in described
    assert "cells: n(v), mean(v)" in described


# --------------------------------------------------------------------------
# The caption under the table


def test_a_pivot_of_an_empty_frame_reports_no_cells_and_no_n():
    """A filter that matched nothing still has to produce a table object, and
    its caption has to say 'nothing' rather than quote a range over no data."""
    frame = pd.DataFrame({"plate": pd.Series([], dtype=object),
                          "v": pd.Series([], dtype=float)})
    result = pivot(frame, PivotSpec(rows=("plate",), values=("v",),
                                    aggs=(N, MEAN)))

    assert result.shape == (0, 1)
    assert int(result.present.sum()) == 0
    assert result.n_range() is None
    assert result.low_n_cells() == 0

    summary = result.summary()
    assert "0 with data" in summary
    assert "n per cell" not in summary
    assert "n ≤" not in summary


def test_a_table_whose_cells_are_all_big_enough_says_nothing_about_low_n():
    """The low-n clause is a warning, so it has to be absent when there is
    nothing to warn about -- otherwise it stops being read."""
    n_per_cell = LOW_N * 4
    frame = pd.DataFrame({
        "plate": ["p1"] * n_per_cell + ["p2"] * n_per_cell,
        "v": np.arange(2.0 * n_per_cell),
    })
    result = pivot(frame, PivotSpec(rows=("plate",), values=("v",),
                                    aggs=(N, MEAN)))

    assert result.n_range() == (n_per_cell, n_per_cell)
    assert result.low_n_cells() == 0

    summary = result.summary()
    assert f"n per cell {n_per_cell:,}–{n_per_cell:,}" in summary
    assert "n ≤" not in summary


def test_the_low_n_clause_comes_back_when_a_cell_is_small():
    """The pair of the test above: the warning is absent because the cells
    are big, not because the clause stopped working."""
    frame = pd.DataFrame({"plate": ["p1", "p1", "p2"],
                          "v": [1.0, 2.0, 3.0]})
    result = pivot(frame, PivotSpec(rows=("plate",), values=("v",),
                                    aggs=(N, MEAN)))
    assert result.low_n_cells() == 2
    assert f"2 cell(s) at n ≤ {LOW_N}" in result.summary()
