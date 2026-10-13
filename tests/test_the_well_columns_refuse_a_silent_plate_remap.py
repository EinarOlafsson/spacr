"""Deriving row and column from a well, and refusing to overwrite them.

The refusal is the point. If a table already carries a row or column and the
well says something different, one of the two is wrong -- and quietly
preferring the well would remap the plate under the user, moving every
measurement to a different position with nothing said. The message names the
first disagreeing row so the user can look at it.

The agreeing case is the one that had never run, and it is the common one: a
table that carries all three, consistently, must be left exactly as it is.
"""
from __future__ import annotations

import pandas as pd
import pytest


def _needed():
    from spacr import schema

    return (schema.ROW_KEY, schema.COLUMN_KEY)


def test_row_and_column_are_derived_when_absent():
    """The base case: neither column present, both filled from the well."""
    from spacr import schema
    from spacr.metadata_resolution import _derive_well_columns

    frame = pd.DataFrame({"well": ["A01", "B12"]})

    out, derived = _derive_well_columns(frame, "well", _needed())

    assert derived is True
    assert schema.ROW_KEY in out.columns and schema.COLUMN_KEY in out.columns
    assert out[schema.ROW_KEY].tolist() == ["r1", "r2"]


def test_an_existing_column_that_agrees_is_left_alone():
    """Arc 180 -> 174: ``inconsistent.any()`` is False, so nothing is written.

    A measurement table that already carries rowID, columnID AND a well is the
    ordinary case, and re-deriving over the top of agreeing values is work
    that can only introduce a difference.
    """
    from spacr import schema
    from spacr.metadata_resolution import _derive_well_columns

    frame = pd.DataFrame({"well": ["A01", "B12"],
                          schema.ROW_KEY: ["r1", "r2"],
                          schema.COLUMN_KEY: ["c1", "c12"]})

    out, derived = _derive_well_columns(frame, "well", _needed())

    assert derived is True
    assert out[schema.ROW_KEY].tolist() == ["r1", "r2"]
    assert out[schema.COLUMN_KEY].tolist() == ["c1", "c12"]


def test_an_existing_column_that_disagrees_is_refused_by_row():
    """The raise: a silent plate remap is worse than a stopped run.

    The message names the first disagreeing index, because a user told only
    "they disagree" has 1,536 rows to search.
    """
    from spacr import schema
    from spacr.metadata_resolution import _derive_well_columns

    frame = pd.DataFrame({"well": ["A01", "B12"],
                          schema.ROW_KEY: ["r1", "r9"]})

    with pytest.raises(ValueError) as excinfo:
        _derive_well_columns(frame, "well", _needed())

    message = str(excinfo.value)
    assert "silent plate remap" in message
    assert "well" in message


def test_a_well_column_that_is_not_there_derives_nothing():
    """The first guard, so the tests above are reached deliberately."""
    from spacr.metadata_resolution import _derive_well_columns

    frame = pd.DataFrame({"area": [1.0]})

    out, derived = _derive_well_columns(frame, "well", _needed())

    assert derived is False
    assert out is frame


def test_a_well_that_cannot_be_parsed_derives_nothing():
    """The second guard: one unparseable well and the whole column is declined.

    All or nothing is deliberate -- deriving row and column for some rows and
    not others would leave a frame that looks complete and joins wrongly.
    """
    from spacr.metadata_resolution import _derive_well_columns

    frame = pd.DataFrame({"well": ["A01", "not a well"]})

    out, derived = _derive_well_columns(frame, "well", _needed())

    assert derived is False
    assert out is frame
