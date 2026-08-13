"""The merge has to respect what a cell can actually contain.

Instruction 77, answering the maintainer's own statement of the cardinality::

    cytoplasm    EXACTLY ONE per cell
    nucleus      MANY per cell
    pathogen     MANY per cell
    organelle    MANY per cell

Three things follow, and this file pins each:

1. A many-per-cell child is CONSOLIDATED to one row per cell before it is
   joined. Joining it raw multiplies rows -- one per (cell, child) pair --
   and every downstream count, mean and test then treats one cell as N.

2. The join type follows ``consolidate_on_cell``, EXCEPT for pathogen and
   organelle, which follow ``keep_uninfected``. An uninfected cell is a cell
   and is usually the control population; instruction 77 item (c) measured
   what dropping it does (object p = 4e-39 against well p = 0.25 on the same
   data).

3. The aggregation is per measurement TYPE -- areas and counts add, means
   average, minima take the minimum -- and a NaN inside a group is never
   silently skipped without the row saying so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.merge_tables import (
    MergePolicy,
    aggregation_for,
    roll_up,
)
from spacr.object_roles import is_one_row_per_cell


IDENTITY = ["plateID", "rowID", "columnID", "fieldID"]
KEYS = IDENTITY + ["cell_id"]


def _children(rows):
    """A child table: one row per child, all in one field."""
    frame = pd.DataFrame(rows)
    for column, value in zip(IDENTITY, ["p1", "r1", "c1", "f1"]):
        frame[column] = value
    return frame


# ---------------------------------------------------------------------------
# 1. Cardinality
# ---------------------------------------------------------------------------

def test_the_declared_cardinality_matches_the_biology():
    """One cytoplasm; many of everything else."""
    assert is_one_row_per_cell("cytoplasm")
    assert is_one_row_per_cell("cell")
    for many in ("nucleus", "pathogen", "organelle"):
        assert not is_one_row_per_cell(many), (
            f"{many} can occur more than once per cell and must be rolled up")


def test_a_many_per_cell_child_becomes_one_row_per_cell():
    """The row multiplication this exists to prevent."""
    child = _children([
        {"cell_id": 1, "area": 10.0},
        {"cell_id": 1, "area": 20.0},
        {"cell_id": 1, "area": 30.0},
        {"cell_id": 2, "area": 40.0},
    ])
    out = roll_up(child, KEYS, name="pathogen", policy=MergePolicy())

    assert len(out) == 2, "four pathogens in two cells must give two rows"
    assert sorted(out["cell_id"]) == [1, 2]


# ---------------------------------------------------------------------------
# 2. The join type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("table", ["nucleus", "pathogen", "organelle"])
def test_consolidate_off_keeps_cells_with_no_child(table):
    assert MergePolicy(consolidate_on_cell=False).how_for(table) == "left"


def test_consolidate_on_makes_nucleus_inner():
    """A cell with no nucleus is debris, not a cell."""
    assert MergePolicy(consolidate_on_cell=True).how_for("nucleus") == "inner"


@pytest.mark.parametrize("table", ["pathogen", "organelle"])
def test_consolidating_does_not_delete_the_uninfected_controls(table):
    """The exception, and instruction 77 item (c) is why it exists.

    Consolidating on the cell must not silently condition every result on
    infection. The uninfected cells are the comparison group.
    """
    policy = MergePolicy(consolidate_on_cell=True, keep_uninfected=True)
    assert policy.how_for(table) == "left"


@pytest.mark.parametrize("table", ["pathogen", "organelle"])
def test_restricting_to_infected_cells_is_available_and_explicit(table):
    policy = MergePolicy(consolidate_on_cell=True, keep_uninfected=False)
    assert policy.how_for(table) == "inner"


def test_a_one_per_cell_table_is_unaffected_by_consolidation():
    """There is nothing to consolidate, so the switch must not move it."""
    for consolidate in (True, False):
        assert MergePolicy(
            consolidate_on_cell=consolidate).how_for("cytoplasm") == "left"


def test_consolidate_on_cell_is_the_recommended_default():
    assert MergePolicy().consolidate_on_cell is True
    assert MergePolicy().keep_uninfected is True


# ---------------------------------------------------------------------------
# 3. Aggregation per measurement type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("column,expected", [
    ("area", "sum"),
    ("filled_area", "sum"),
    ("volume", "sum"),
    ("count", "sum"),
    ("integrated_intensity", "sum"),
    ("mean_intensity", "mean"),
    ("channel_1_mean_intensity", "mean"),
    ("median_intensity", "median"),
    ("min_intensity", "min"),
    ("max_intensity", "max"),
    ("perimeter", "mean"),
    ("eccentricity", "mean"),
    ("object_label", "first"),
])
def test_each_measurement_combines_the_way_it_should(column, expected):
    """Summing a mean is meaningless; averaging an area under-reports it."""
    assert aggregation_for(column) == expected


def test_areas_add_and_means_average_on_real_rows():
    """The rules, applied, rather than the rules asserted about themselves."""
    child = _children([
        {"cell_id": 1, "area": 10.0, "mean_intensity": 2.0,
         "min_intensity": 5.0, "max_intensity": 9.0},
        {"cell_id": 1, "area": 30.0, "mean_intensity": 4.0,
         "min_intensity": 1.0, "max_intensity": 7.0},
    ])
    out = roll_up(child, KEYS, name="pathogen", policy=MergePolicy())
    row = out.iloc[0]

    assert row["pathogen_area"] == 40.0, "areas add"
    assert row["pathogen_mean_intensity"] == 3.0, "means average"
    assert row["pathogen_min_intensity"] == 1.0, "minima take the minimum"
    assert row["pathogen_max_intensity"] == 9.0, "maxima take the maximum"
    assert row["pathogen_count"] == 2


# ---------------------------------------------------------------------------
# 4. NA values, which must never vanish quietly
# ---------------------------------------------------------------------------

def test_a_skipped_nan_is_visible_in_the_row():
    """pandas skips NaN in every aggregation and says nothing.

    Three pathogens, one unmeasurable area: the sum is of TWO while the
    count says three. Both numbers are defensible; the pair being
    unmarked is not.
    """
    child = _children([
        {"cell_id": 1, "area": 10.0},
        {"cell_id": 1, "area": np.nan},
        {"cell_id": 1, "area": 30.0},
    ])
    out = roll_up(child, KEYS, name="pathogen", policy=MergePolicy())
    row = out.iloc[0]

    assert row["pathogen_area"] == 40.0
    assert row["pathogen_count"] == 3
    assert row["pathogen_measured"] == 2, (
        "the row must record that one child contributed nothing")
    assert row["pathogen_measured"] < row["pathogen_count"]


def test_a_complete_group_reports_measured_equal_to_count():
    child = _children([
        {"cell_id": 1, "area": 10.0},
        {"cell_id": 1, "area": 30.0},
    ])
    out = roll_up(child, KEYS, name="pathogen", policy=MergePolicy())
    row = out.iloc[0]
    assert row["pathogen_measured"] == row["pathogen_count"] == 2


def test_the_shortfall_is_reported_by_name(caplog):
    """A user must be able to find out WHICH measurement was short."""
    import logging

    child = _children([
        {"cell_id": 1, "area": 10.0, "mean_intensity": np.nan},
        {"cell_id": 1, "area": np.nan, "mean_intensity": 2.0},
    ])
    with caplog.at_level(logging.INFO, logger="spacr.merge_tables"):
        roll_up(child, KEYS, name="pathogen", policy=MergePolicy())

    text = caplog.text
    assert "area" in text and "mean_intensity" in text
    assert "silently" in text


def test_a_table_of_identity_columns_only_still_reports_measured():
    """`measured` must exist on every rolled-up table, or a downstream
    test for it becomes a KeyError on exactly the tables that are fine."""
    child = _children([{"cell_id": 1, "object_label": 7}])
    out = roll_up(child, KEYS, name="pathogen", policy=MergePolicy())
    assert "pathogen_measured" in out.columns
    assert out.iloc[0]["pathogen_measured"] == out.iloc[0]["pathogen_count"]
