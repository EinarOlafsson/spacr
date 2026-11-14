"""An outlier filter that could not run says so instead of saying nothing.

A filter the user switched on and that removed nothing looks identical, in a
run log, to a filter that never ran. Two ways it can fail to run -- a
threshold that is not a number, and a criterion the table has no column for
-- both have to produce a report row carrying the reason, and both have to
leave the table untouched rather than raising in the middle of a pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import outlier_filter


@pytest.mark.parametrize("mads", [-1.0, 0.0, np.nan, np.inf, -np.inf])
def test_a_disabled_direct_threshold_never_marks_every_finite_value(mads):
    """The low-level helper follows the same disabled-threshold contract."""
    values = [9.0, 10.0, 11.0, 12.0, 100.0]
    assert not outlier_filter.outliers(values, mads=mads).any()
    assert outlier_filter.outliers(values, mads=5.0).tolist() == [
        False, False, False, False, True,
    ]


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
