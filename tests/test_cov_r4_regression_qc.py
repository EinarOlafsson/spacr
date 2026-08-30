"""Regression QC's positional panels: every group is a group with wells in it.

The plate/row/column panels are the only place in this module where a
statistic is computed by *partitioning* the residuals rather than transforming
them, and the partition is the thing that has to hold. Two guards inside
``_positional_effect_panel`` -- ``if edge.size and interior.size`` before the
edge statistic, and ``if v.size`` before a group's raw points are drawn -- are
only ever false if a group came back empty, and a group can only come back
empty if ``_grouped_residuals`` stops being a partition of the fitted rows.

So what is pinned here is the partition itself: every group carries at least
one residual, and the groups together carry each fitted row exactly once, for
each of the two metadata-alignment routes ``_align_metadata`` allows (by index
from a frame longer than the fit, and by length) and for the label types a
real plate map produces -- integers, floats and a missing well.

Pinned alongside it is the edge statistic those guards protect: it fires only
when the panel was asked for it *and* the layout has an interior, and when it
fires the two outer groups -- and only those -- are recoloured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from matplotlib.figure import Figure

from spacr import regression_qc as rq
from spacr import schema


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def _axes():
    """A bare axes on an object-oriented figure, as the report driver makes."""
    return Figure(figsize=(4.6, 3.6)).add_subplot(1, 1, 1)


class _Fit:
    """A minimal stand-in for a fitted model: whatever attributes are given."""

    def __init__(self, **attributes):
        self.__dict__.update(attributes)


#: Four rows of six wells. The outer rows sit high, one interior row sits
#: lowest of all, and every well is nudged off its row's centre so the
#: Kruskal-Wallis test has within-group spread to work with.
_NUDGE = np.array([-0.05, -0.03, -0.01, 0.01, 0.03, 0.05])
_ROW_OFFSETS = {"r01": 0.8, "r02": 0.0, "r03": -1.5, "r04": 0.8}
_ROW_LABELS = np.repeat(list(_ROW_OFFSETS), 6)
_ROW_RESID = np.concatenate([offset + _NUDGE for offset in _ROW_OFFSETS.values()])


def _plate_context(resid, metadata, index=None):
    """A context whose residuals are exactly ``resid``, carrying a plate map.

    ``fittedvalues`` are zero, so the residual the panels group is the response
    itself and the numbers in the assertions below are the numbers that were
    written here.
    """
    n = len(resid)
    design = pd.DataFrame({"intercept": np.ones(n),
                           "x": np.linspace(0.0, 1.0, n)},
                          index=index if index is not None else range(n))
    return rq.build_context(_Fit(fittedvalues=np.zeros(n)), design,
                            np.asarray(resid, dtype=float), metadata=metadata)


# ---------------------------------------------------------------------------
# the partition
# ---------------------------------------------------------------------------

def test_grouping_residuals_by_a_plate_column_is_a_partition_of_the_fitted_rows():
    """A group with no wells in it would put a hole in every positional panel.

    ``_grouped_residuals`` is reached only through the three positional panels,
    which expose the group *labels* but never the group *sizes*, so the helper
    is called directly here -- the sizes are the whole invariant.

    Both routes ``_align_metadata`` accepts are driven, because they are the
    two ways the frame can stop matching the fit: a plate map handed in whole
    (longer than the fit, aligned on the surviving index) and one already cut
    to the fitted rows (aligned on length). Alongside them are the label types
    a real plate map arrives with -- an integer column, a float column and a
    well whose row was never recorded -- since every one of them is stringified
    before the grouping and a type that stringified inconsistently would be the
    way an empty group got built.
    """
    resid = np.arange(12.0)

    # Route 1: the whole plate map, twice as long as the fit, aligned by index.
    fitted_index = [i for i in range(24) if i % 2 == 0]
    whole_map = pd.DataFrame(
        {schema.ROW_KEY: [1, 1, 2, 2, 3, 3] * 4,
         schema.COLUMN_KEY: np.repeat([1.5, 2.5, 3.5], 8),
         schema.PLATE_KEY: ["p1"] * 12 + [None] * 12},
        index=range(24))
    by_index = _plate_context(resid, whole_map, index=fitted_index)

    # Route 2: the same twelve rows, handed in already cut, index thrown away.
    cut_map = whole_map.loc[fitted_index].reset_index(drop=True)
    by_length = _plate_context(resid, cut_map)

    for ctx in (by_index, by_length):
        for column in (schema.ROW_KEY, schema.COLUMN_KEY, schema.PLATE_KEY):
            groups, values = rq._grouped_residuals(ctx, column)
            sizes = [v.size for v in values]
            # Every group is inhabited, and the groups between them account for
            # each fitted row exactly once -- no row dropped, none counted twice.
            assert min(sizes) >= 1
            assert sum(sizes) == ctx.n == 12
            assert sorted(np.concatenate(values)) == sorted(resid)
            # The labels are the distinct stringified column values, so a group
            # can never name a well that is not there.
            assert set(groups) == set(ctx.metadata[column].astype(str))

    # The two routes selected the same wells, which is what makes the loop
    # above two independent checks of one invariant rather than one repeated.
    assert list(by_index.metadata[schema.ROW_KEY]) == \
        list(by_length.metadata[schema.ROW_KEY])
    # The unrecorded plate is a group of its own, not a group of nothing: half
    # the wells carry 'p1' and half carry the stringified blank.
    plate_groups, plate_values = rq._grouped_residuals(by_length,
                                                       schema.PLATE_KEY)
    assert len(plate_groups) == 2
    assert sorted(v.size for v in plate_values) == [6, 6]


# ---------------------------------------------------------------------------
# the edge statistic the partition feeds
# ---------------------------------------------------------------------------

def test_the_edge_statistic_needs_both_an_edge_and_an_interior_to_exist():
    """An "edge effect" computed without an interior is a number about nothing.

    The row and column panels shade the outer groups because wells on the rim
    of a plate evaporate; the plate panel must not, because plate 1 and plate 4
    are not neighbours. The same twenty-four residuals are therefore grouped
    three ways here -- by row (four groups, edges wanted), by plate (four
    groups, edges not wanted) and by column (three groups, so the interior is a
    single group and there is nothing to compare an edge with) -- and only the
    first is entitled to the statistic.
    """
    metadata = pd.DataFrame({
        schema.ROW_KEY: _ROW_LABELS,
        schema.PLATE_KEY: _ROW_LABELS,
        schema.COLUMN_KEY: np.repeat(["c01", "c02", "c03"], 8)})
    ctx = _plate_context(_ROW_RESID, metadata)

    row_ax, plate_ax, column_ax = _axes(), _axes(), _axes()
    row = rq._panel_row_effects(ctx, row_ax)
    plate = rq._panel_plate_effects(ctx, plate_ax)
    column = rq._panel_column_effects(ctx, column_ax)

    # Rows: the edges sit at +0.8 and the interior median is halfway down the
    # pooled r02/r03 wells, so the gap is real and larger than half a residual
    # standard deviation -- which is what buys the outer rows their colour.
    assert row["n_groups"] == 4
    assert np.isclose(row["edge_minus_interior_median"], 1.55)
    assert set(row["highlighted_groups"]) == {"r01", "r03", "r04"}
    # r03 is the largest |median| and is interior, so it is coloured by the
    # Kruskal-Wallis rule; r01 and r04 are coloured by the edge rule.
    assert row["worst_group"] == "r03"
    assert row["kruskal_p"] < 0.05

    # Plates: identical groups, identical residuals, no edge statistic -- the
    # flag is the only difference, which is what makes the NaN meaningful.
    assert plate["n_groups"] == 4
    assert plate["groups"] == row["groups"]
    assert np.isnan(plate["edge_minus_interior_median"])
    assert plate["highlighted_groups"] == ["r03"]

    # Columns: edges are wanted but three groups leave a one-group interior,
    # which the panel refuses to call an interior.
    assert column["n_groups"] == 3
    assert np.isnan(column["edge_minus_interior_median"])

    # Every group was drawn, on all three panels: one scatter collection of raw
    # points per group, which is the loop the empty-group guard protects.
    assert len(row_ax.collections) == 4
    assert len(plate_ax.collections) == 4
    assert len(column_ax.collections) == 3
    # And the edge verdict reached the figure, not just the stats dict.
    row_text = "\n".join(t.get_text() for t in row_ax.texts)
    plate_text = "\n".join(t.get_text() for t in plate_ax.texts)
    assert "edge - interior median" in row_text
    assert "edge - interior median" not in plate_text
