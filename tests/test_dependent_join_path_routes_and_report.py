"""The dependent-variable join's path route and the sentence that reports it.

A path column whose values carry no plate/well/field/object structure is not
a route: handing back a frame of blanks would let the join "succeed" on a key
made entirely of blanks.  And a report with no route is a join that never
happened, which the sentence must say rather than formatting a zero-row
success.
"""
from __future__ import annotations

import pandas as pd

from spacr import dependent_join as dj

PARSED = {"plateID": "plate1", "well": "A01", "rowID": "r1", "columnID": "c1",
          "fieldID": "1", "objectID": "7"}
BLANK = dict.fromkeys(PARSED)


def test_from_paths_recovers_ids_only_from_a_parseable_path_column(subtests):
    """The IDs come back per row from the first path column found; no column or no parseable row is None."""
    cases = [
        ("no_path_column", pd.DataFrame({"value": [1.0]}), None),
        ("a_path_column_of_unparseable_text",
         pd.DataFrame({"png_path": ["notes.txt", "readme", ""]}), None),
        ("a_crop_path", pd.DataFrame({"png_path": ["/x/plate1_A01_1_7.png"]}), [PARSED]),
        ("a_prcfo_key", pd.DataFrame({"prcfo": ["plate1_A01_1_7"]}), [PARSED]),
        ("one_parseable_row_of_two",
         pd.DataFrame({"png_path": ["/x/plate1_A01_1_7.png", "notes.txt"]}, index=[4, 9]),
         [PARSED, BLANK]),
    ]
    for case_id, frame, expected in cases:
        with subtests.test(msg=case_id):
            recovered = dj._from_paths(frame)
            if expected is None:
                assert recovered is None
                continue
            assert list(recovered.index) == list(frame.index)
            assert set(dj.ID_COLUMNS) <= set(recovered.columns)
            got = recovered.astype(object).where(recovered.notna(), None)
            assert got.to_dict("records") == expected


def test_describe_says_whether_and_how_the_join_happened(subtests):
    """No route reads as "not joined"; a route names itself, both counts, and whether it is a fallback."""
    fallback = " (a fallback -- a column from the direct join is missing upstream)"
    cases = [
        ("empty_route", {"route": "", "matched": 0, "rows": 0},
         "the dependent variable was not joined"),
        ("no_route_key", {}, "the dependent variable was not joined"),
        ("the_direct_route", {"route": dj.ROUTES[0][0], "matched": 3, "rows": 4},
         "joined the dependent variable by the ID columns: 3 of 4 rows matched."),
        ("a_fallback_route", {"route": dj.ROUTES[1][0], "matched": 2, "rows": 2},
         "joined the dependent variable by the image path, split into its parts: "
         f"2 of 2 rows matched{fallback}."),
        ("thousands_are_grouped", {"route": dj.ROUTES[2][0], "matched": 1234, "rows": 56789},
         "joined the dependent variable by the well from the path, translated to row "
         f"and column: 1,234 of 56,789 rows matched{fallback}."),
    ]
    for case_id, report, expected in cases:
        with subtests.test(msg=case_id):
            assert dj.describe(report) == expected
