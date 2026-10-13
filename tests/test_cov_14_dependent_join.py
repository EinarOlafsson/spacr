"""The dependent-variable join refuses a route it cannot honour, and says so.

Two ways a join goes nowhere, and both have to stay distinguishable from a
join that worked:

* A frame that HAS a path column whose values carry no plate/well/field/object
  structure is not a path route at all. Handing back a frame of empty strings
  would let the join "succeed" on a key made entirely of blanks, matching
  every row to every row.
* A report with no route is the record of a join that never happened, and the
  sentence shown to the user has to say that rather than formatting a
  zero-row success.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr import dependent_join


def test_a_path_column_without_recoverable_ids_is_not_a_route():
    """A path column of unparseable text yields no ID frame at all.

    ``_from_paths`` returning an all-blank frame instead of ``None`` would
    make every row's join key identical, which matches everything.
    """
    frame = pd.DataFrame({"png_path": ["notes.txt", "readme", ""]})

    assert dependent_join._from_paths(frame) is None


def test_a_path_column_with_ids_is_a_route():
    """The same helper does return the five IDs when the path carries them."""
    frame = pd.DataFrame({"png_path": ["/x/plate1_A01_1_7.png"]})

    recovered = dependent_join._from_paths(frame)

    assert recovered is not None
    assert set(dependent_join.ID_COLUMNS).issubset(recovered.columns)
    assert recovered.iloc[0]["plateID"] == "plate1"
    assert recovered.iloc[0]["objectID"] == "7"


def test_a_missing_path_column_is_not_a_route():
    """No path column at all is also ``None``, not an empty frame."""
    assert dependent_join._from_paths(pd.DataFrame({"value": [1.0]})) is None


def test_an_unjoined_report_says_the_join_did_not_happen():
    """``describe`` on a routeless report must not read like a success.

    A report whose route is empty is the record of a join that was never
    made; formatting it as "0 of 0 rows matched" reads as a join that ran and
    found nothing, which sends the reader looking for the wrong problem.
    """
    said = dependent_join.describe({"route": "", "matched": 0, "rows": 0})

    assert said == "the dependent variable was not joined"


def test_a_report_with_no_route_key_at_all_says_the_same():
    """A report missing the key entirely takes the same branch."""
    assert dependent_join.describe({}) == "the dependent variable was not joined"


def test_a_joined_report_names_its_route_and_counts():
    """The success sentence carries the route and both row counts."""
    said = dependent_join.describe(
        {"route": dependent_join.ROUTES[0][0], "matched": 3, "rows": 4})

    assert dependent_join.ROUTES[0][0] in said
    assert "3 of 4 rows matched" in said
    assert "fallback" not in said


def test_a_fallback_route_is_labelled_as_one():
    """Any route but the direct one is marked a fallback in the sentence."""
    said = dependent_join.describe(
        {"route": dependent_join.ROUTES[1][0], "matched": 2, "rows": 2})

    assert "fallback" in said
