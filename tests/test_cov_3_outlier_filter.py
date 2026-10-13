"""An outlier filter that could not run says so instead of saying nothing.

A filter the user switched on and that removed nothing looks identical, in a
run log, to a filter that never ran. Two ways it can fail to run -- a
threshold that is not a number, and a criterion the table has no column for
-- both have to produce a report row carrying the reason, and both have to
leave the table untouched rather than raising in the middle of a pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import outlier_filter


def _frame():
    return pd.DataFrame({"cell_area": [10.0, 11.0, 10.5, 300.0, 10.2],
                         "object_id": range(5)})


def test_an_empty_table_is_returned_untouched_with_an_empty_report():
    """A pipeline stage that receives nothing must hand nothing on, not
    raise -- an empty measurement table is a normal well."""
    empty = pd.DataFrame({"cell_area": []})

    out, report = outlier_filter.apply(empty, {"cell_area_outlier_mads": 5})

    assert report == []
    assert len(out) == 0
    assert list(out.columns) == ["cell_area"]


def test_no_table_at_all_becomes_an_empty_frame():
    """`None` reaches here from a stage that produced no measurements; the
    caller downstream indexes the result, so it must be a frame."""
    out, report = outlier_filter.apply(None, {"cell_area_outlier_mads": 5})

    assert isinstance(out, pd.DataFrame)
    assert out.empty
    assert report == []


def test_a_threshold_that_is_not_a_number_is_reported_not_ignored():
    """A typed 'five' in the settings file must not silently disable the
    filter: the report row names the offending value so the user can see
    which setting did nothing."""
    out, report = outlier_filter.apply(_frame(),
                                       {"cell_area_outlier_mads": "five"})

    assert len(out) == 5, "nothing should have been removed"
    assert len(report) == 1
    row = report[0]
    assert row["criterion"] == "cell_area"
    assert row["removed"] == 0
    assert row["mads"] is None
    assert "'five'" in row["note"]
    assert "not a number" in row["note"]


def test_a_zero_threshold_disables_the_criterion_silently():
    """Zero MADs is how a filter is switched off, not a failure -- it earns
    no report row, so the run output does not list disabled filters."""
    out, report = outlier_filter.apply(_frame(),
                                       {"cell_area_outlier_mads": 0})

    assert len(out) == 5
    assert report == []


@pytest.mark.parametrize("mads", [-1.0, 0.0, np.nan, np.inf, -np.inf])
def test_a_disabled_direct_threshold_never_marks_every_finite_value(mads):
    """The low-level helper follows the same disabled-threshold contract."""
    values = [9.0, 10.0, 11.0, 12.0, 100.0]
    assert not outlier_filter.outliers(values, mads=mads).any()
    assert outlier_filter.outliers(values, mads=5.0).tolist() == [
        False, False, False, False, True,
    ]


def test_a_criterion_with_no_column_is_reported_with_the_label():
    """The note names the human label so the user can match it to the
    checkbox they ticked."""
    frame = pd.DataFrame({"cell_area": [1.0, 2.0, 3.0]})

    _out, report = outlier_filter.apply(
        frame, {"nucleus_area_outlier_mads": 5})

    assert len(report) == 1
    assert report[0]["column"] == ""
    assert "nucleus area" in report[0]["note"]


def test_the_run_summary_prints_a_note_instead_of_a_removal_count():
    """`describe` must not print '0 object(s) beyond None MADs' for a row
    that never ran; the reason replaces the count."""
    report = [{"criterion": "cell_area", "caption": "cell area", "column": "",
               "mads": None, "removed": 0,
               "note": "'five' is not a number of MADs"},
              {"criterion": "nucleus_area", "caption": "nucleus area",
               "column": "nucleus_area", "mads": 5.0, "removed": 3,
               "note": ""}]

    text = outlier_filter.describe(report)
    lines = text.splitlines()

    assert lines[0].startswith("Outliers removed")
    assert "  cell area: 'five' is not a number of MADs" in lines
    assert "MADs of the median" not in lines[1]
    assert any("nucleus area (nucleus_area): 3 object(s)" in line
               for line in lines), text
