"""Choosing the columns a cell table and a fraction table are joined on.

A well is a row AND a column. Joining on one of them alone silently pools every
plate's row A together, so the refusal here is not tidiness -- it is the
difference between attributing a hit to a well and attributing it to a stripe
across the plate.

The uncovered arc is a caller who asked for a key that is not row/column at
all, which the partial-key rule must not apply to.
"""
from __future__ import annotations

import pandas as pd
import pytest


def _tables(*columns):
    frame = pd.DataFrame({c: ["x"] for c in columns})
    return frame, frame.copy()


def test_a_full_well_key_present_in_both_tables_is_used():
    """The ordinary case."""
    from spacr.hit_attribution import _well_columns

    cells, fractions = _tables("plateID", "rowID", "columnID", "value")

    assert _well_columns(cells, fractions,
                         ["plateID", "rowID", "columnID"]) == [
        "plateID", "rowID", "columnID"]


def test_half_a_well_key_is_refused():
    """The partial-key raise.

    Joining on rowID without columnID pools row A of every column into one
    group, so a hit attributed there is attributed to a stripe across the
    plate rather than to a well.
    """
    from spacr.hit_attribution import HitAttributionError, _well_columns

    cells = pd.DataFrame({"plateID": ["p1"], "rowID": ["r1"], "value": [1]})
    fractions = pd.DataFrame({"plateID": ["p1"], "rowID": ["r1"],
                              "columnID": ["c1"]})

    with pytest.raises(HitAttributionError, match="partial key"):
        _well_columns(cells, fractions, ["plateID", "rowID", "columnID"])


def test_a_key_that_is_not_row_and_column_skips_the_partial_rule():
    """Arc 92 -> 96: the rule applies only when BOTH were requested.

    A caller passing an explicit key -- ``prc``, say, which carries the whole
    well in one column -- is not making a partial-well mistake, and refusing
    it would make the ``well_columns`` parameter unusable for exactly the
    tables it exists for.
    """
    from spacr.hit_attribution import _well_columns

    cells, fractions = _tables("prc", "value")

    assert _well_columns(cells, fractions, ["prc"]) == ["prc"]


def test_tables_that_share_no_key_are_refused_with_the_expected_names():
    """The raise above both, which tells the user what to supply.

    "share no well key" without naming plateID/rowID/columnID leaves them
    guessing which spelling this code wants.
    """
    from spacr.hit_attribution import HitAttributionError, _well_columns

    cells = pd.DataFrame({"something": ["x"]})
    fractions = pd.DataFrame({"other": ["y"]})

    with pytest.raises(HitAttributionError) as excinfo:
        _well_columns(cells, fractions, ["plateID", "rowID", "columnID"])

    message = str(excinfo.value)
    assert "plateID" in message and "rowID" in message
    assert "well_columns" in message


def test_a_column_present_in_only_one_table_is_not_part_of_the_key():
    """The intersection: a key column must exist on BOTH sides to join on.

    A column present in one table only would make the merge drop every row,
    which reads as "no cells matched" rather than "the tables disagree".
    """
    from spacr.hit_attribution import _well_columns

    cells = pd.DataFrame({"plateID": ["p1"], "rowID": ["r1"],
                          "columnID": ["c1"], "extra": ["e"]})
    fractions = pd.DataFrame({"plateID": ["p1"], "rowID": ["r1"],
                              "columnID": ["c1"]})

    assert "extra" not in _well_columns(
        cells, fractions, ["plateID", "rowID", "columnID", "extra"])
