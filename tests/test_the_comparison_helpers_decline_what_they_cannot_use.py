"""Small helpers behind the gene/measurement comparison, on their empty cases.

Each returns an emptiness rather than raising, and each emptiness means
something the caller acts on: no wells for this group, no control wells at all,
no second column to combine with, no groups to compare. A raise in any of them
would take down a panel the user opened to look at something else.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _objects(n=6):
    """An object table with a well label and two measurements."""
    return pd.DataFrame(
        {"prc": [f"p1_r1_c{i % 3 + 1}" for i in range(n)],
         "area": np.linspace(10.0, 20.0, n),
         "perimeter": np.linspace(1.0, 2.0, n)},
        index=[f"o{i}" for i in range(n)])


# ---------------------------------------------------------------------------
# wells_of — a group whose members are not in the table
# ---------------------------------------------------------------------------

def test_a_group_whose_objects_are_absent_is_left_out():
    """Arc 184 -> 182: the loop passes over it.

    A saved group names objects by id, and a table filtered since it was saved
    no longer holds them. An empty tuple for that group would draw an empty
    series on the comparison, which reads as a group that was measured and
    found to be nothing.
    """
    from spacr.gene_measurement_compare import wells_of

    out = wells_of(_objects(), {"present": ["o0", "o1"],
                                "gone": ["not_in_the_table"]})

    assert "present" in out
    assert "gone" not in out


def test_a_group_that_is_present_names_its_wells_once_each():
    """The taken side, de-duplicated in first-occurrence order."""
    from spacr.gene_measurement_compare import wells_of

    out = wells_of(_objects(), {"all": [f"o{i}" for i in range(6)]})

    assert out["all"] == ("p1_r1_c1", "p1_r1_c2", "p1_r1_c3")


def test_a_table_with_no_well_column_names_no_wells():
    """The guard above the loop: without a well label there is nothing to name."""
    from spacr.gene_measurement_compare import wells_of

    assert wells_of(pd.DataFrame({"area": [1.0]}), {"a": [0]}) == {}


def test_no_groups_at_all_names_no_wells():
    """The empty mapping, which an untouched picker sends."""
    from spacr.gene_measurement_compare import wells_of

    assert wells_of(_objects(), {}) == {}
    assert wells_of(_objects(), None) == {}


# ---------------------------------------------------------------------------
# control_wells — a count table that cannot answer
# ---------------------------------------------------------------------------

def test_a_count_table_without_the_guide_column_names_no_controls():
    """Arc 218 -> 219: the guide column decides which rows are controls.

    Returning () rather than raising lets the comparison draw without a
    control band, which is a weaker figure and still a figure.
    """
    from spacr.gene_measurement_compare import control_wells

    counts = pd.DataFrame({"prc": ["p1_r1_c1"], "count": [10]})

    assert control_wells(counts, ["nc"]) == ()


def test_a_count_table_without_a_well_column_names_no_controls():
    """The other half of the same guard."""
    from spacr.gene_measurement_compare import control_wells

    counts = pd.DataFrame({"grna": ["g1"], "count": [10]})

    assert control_wells(counts, ["nc"]) == ()


# ---------------------------------------------------------------------------
# combine — one column, no operator
# ---------------------------------------------------------------------------

def test_one_measurement_with_no_operator_is_returned_as_itself():
    """Arc 266 -> 267: the common case, and it must not touch ``second``.

    A user comparing one measurement supplies no operator and often no second
    column at all. Reading ``objects[second]`` first would raise a KeyError on
    an empty string.
    """
    from spacr.gene_measurement_compare import combine

    values, name, kind = combine(_objects(), "area", "", "")

    assert name == "area"
    assert kind == 0
    assert len(values) == 6


def test_two_measurements_are_combined_and_named_together():
    """The taken side, whose name is what the axis will read."""
    from spacr.gene_measurement_compare import combine

    values, name, _kind = combine(_objects(), "area", "/", "perimeter")

    assert name == "area / perimeter"
    assert np.isfinite(values).all()


def test_a_division_by_zero_is_missing_and_not_infinite():
    """The docstring's promise, which keeps an artefact out of a comparison.

    "Division by zero or a non-finite denominator produces a missing value; it
    is never converted to zero or infinity." An infinity would dominate every
    axis it appeared on.
    """
    from spacr.gene_measurement_compare import combine

    objects = _objects()
    objects.loc["o0", "perimeter"] = 0.0

    values, _name, _kind = combine(objects, "area", "/", "perimeter")

    assert not np.isfinite(values.iloc[0])
    assert not np.isinf(values.iloc[0])
