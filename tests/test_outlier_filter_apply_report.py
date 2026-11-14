"""Table-driven test of ``spacr.outlier_filter.apply`` and its per-criterion report.

A filter that could not run must say so: a threshold that is not a number
and a criterion the table has no column for each produce a report row with
the reason and remove nothing; a zero threshold disables a criterion
silently; an empty or missing table passes through with an empty report.
"""
from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from spacr import outlier_filter


def _frame():
    """Five objects: object 3 has an outlying cell area, object 4 a bright nucleus."""
    return pd.DataFrame({
        "cell_area": [10.0, 11.0, 10.5, 300.0, 10.2],
        "nucleus_mean_intensity": [1.0, 1.1, 1.05, 1.0, 50.0],
        "object_id": range(5),
    })


def _row(criterion, caption, column, mads, removed, note=""):
    """One expected report record."""
    return {"criterion": criterion, "caption": caption, "column": column,
            "mads": mads, "removed": removed, "note": note}


CELL_AREA_HIT = _row("cell_area", "cell area", "cell_area", 5.0, 1)


def test_apply_filters_kept_rows_and_reports_every_enabled_criterion(subtests):
    """Exact kept object ids and exact report rows for each settings table."""
    no_nucleus_area = ("this table has no nucleus area column, so nothing "
                       "was filtered on it")
    cases = [
        ("no settings removes nothing", None, [0, 1, 2, 3, 4], []),
        ("None threshold disables the criterion",
         {"cell_area_outlier_mads": None}, [0, 1, 2, 3, 4], []),
        ("zero threshold disables silently",
         {"cell_area_outlier_mads": 0}, [0, 1, 2, 3, 4], []),
        ("negative threshold disables silently",
         {"cell_area_outlier_mads": -2}, [0, 1, 2, 3, 4], []),
        ("a number of MADs removes the outlier",
         {"cell_area_outlier_mads": 5}, [0, 1, 2, 4], [CELL_AREA_HIT]),
        ("a numeric string is read as a number",
         {"cell_area_outlier_mads": "5"}, [0, 1, 2, 4], [CELL_AREA_HIT]),
        ("a word is reported, not ignored",
         {"cell_area_outlier_mads": "five"}, [0, 1, 2, 3, 4],
         [_row("cell_area", "cell area", "", None, 0,
               "'five' is not a number of MADs")]),
        ("a non-scalar is reported too",
         {"cell_area_outlier_mads": [5]}, [0, 1, 2, 3, 4],
         [_row("cell_area", "cell area", "", None, 0,
               "[5] is not a number of MADs")]),
        ("a criterion with no column names its caption",
         {"nucleus_area_outlier_mads": 5}, [0, 1, 2, 3, 4],
         [_row("nucleus_area", "nucleus area", "", 5.0, 0, no_nucleus_area)]),
        ("criteria combine in CRITERIA order",
         {"nucleus_intensity_outlier_mads": 5, "nucleus_area_outlier_mads": 3,
          "cell_area_outlier_mads": 5},
         [0, 1, 2],
         [CELL_AREA_HIT,
          _row("nucleus_area", "nucleus area", "", 3.0, 0, no_nucleus_area),
          _row("nucleus_intensity", "nucleus channel intensity",
               "nucleus_mean_intensity", 5.0, 1)]),
    ]
    for case_id, settings, kept, report in cases:
        with subtests.test(msg=case_id):
            frame = _frame()
            out, got = outlier_filter.apply(frame, settings)
            assert out["object_id"].tolist() == kept
            assert got == report
            pdt.assert_frame_equal(frame, _frame())

    with subtests.test(msg="an empty table is returned as is"):
        empty = pd.DataFrame({"cell_area": []})
        out, got = outlier_filter.apply(empty, {"cell_area_outlier_mads": 5})
        assert out is empty
        assert got == []

    with subtests.test(msg="no table at all becomes an empty frame"):
        out, got = outlier_filter.apply(None, {"cell_area_outlier_mads": 5})
        pdt.assert_frame_equal(out, pd.DataFrame())
        assert got == []
