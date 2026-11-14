"""Control-chart estimation at the edges of what a campaign can supply.

A control chart's limits are a claim about a process, and every branch here
decides whether that claim is made from enough data to mean anything: a
subgroup constant asked for beyond the range the gamma function survives, a
baseline named by plates that are not in the table, a campaign shorter than
the baseline it asked for, run rules over a series too short to contain them,
and a Z' table with no run-order column of its own. Each has to either refuse
by name or say what it did instead, because limits computed quietly from the
wrong points look exactly like limits computed correctly.
"""
from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets import control_chart as CC
from spacr.qt.widgets.control_chart import (
    ESTIMATOR_MOVING_RANGE,
    ControlChartError,
    ControlChartSpec,
    c4,
    zprime_frame,
)


def test_a_subgroup_too_large_for_the_gamma_function_still_has_a_constant():
    """Γ(171) overflows a float, so the exact formula cannot be evaluated for
    a subgroup of 400 wells. The asymptotic value is used instead -- and it
    has to be, because returning NaN or raising would take down a chart over
    a plate with more control wells than usual."""
    assert c4(400) == pytest.approx(1.0 - 0.75 / 400)
    assert c4(343) == pytest.approx(1.0, abs=3e-3)
    # The exact formula is still used just below the ceiling, and the two
    # agree to the precision that matters.
    assert c4(342) == pytest.approx(1.0, abs=1e-3)


def test_a_subgroup_of_one_has_no_spread_to_unbias():
    """The refusal names the chart that does handle one well per plate,
    because that is the user's next move."""
    with pytest.raises(ControlChartError, match=ESTIMATOR_MOVING_RANGE):
        c4(1)


def test_a_numeric_baseline_cutoff_must_be_a_number():
    """The cut-off is compared against the order keys. A word compared with a
    float is a TypeError at render time; refusing it names the column and the
    value instead."""
    with pytest.raises(ControlChartError, match="has to be a number"):
        CC._cutoff_key("last tuesday", "numeric")
    assert CC._cutoff_key("12", "numeric") == 12.0


def test_a_text_baseline_cutoff_is_compared_the_way_the_plates_are():
    """Plate ids sort naturally, so ``P9`` comes before ``P10``. The cut-off
    has to be put in the same currency or the baseline would be chosen by
    ASCII order."""
    assert CC._cutoff_key("P10", "text") == CC._natural_key("P10")
    assert CC._cutoff_key("P9", "text") < CC._cutoff_key("P10", "text")


def test_a_series_too_short_for_a_trend_reports_no_trend():
    """Six points in a row is six points; a series of four cannot contain
    one, and inventing a partial run would flag a plate for a pattern that is
    not there."""
    assert CC._rule_3(np.array([0.0, 1.0, 2.0, 3.0]), 6) == []


def test_a_series_shorter_than_the_window_reports_no_violation():
    """"Two of three beyond two sigma" needs three points to look at."""
    assert CC._rule_k_of_m(np.array([3.0, 3.5]), 2, 3, 2.0) == []


def _zprime_frame_rows(order_column):
    rows = []
    for index in range(4):
        for level, base in (("pos", 100.0), ("neg", 10.0)):
            for offset in (-1.0, 1.0):
                row = {"plateID": f"P{index + 1:02d}", "well_type": level,
                       "signal": base + offset}
                if order_column:
                    row["run_order"] = index
                rows.append(row)
    return pd.DataFrame(rows)


def test_zprime_needs_a_column_for_its_control_levels_to_be_levels_of():
    """Two level names with no column to look them up in cannot select any
    well, so the assay window would be computed over the whole plate."""
    with pytest.raises(ControlChartError, match="needs control_column set"):
        zprime_frame(_zprime_frame_rows(True),
                     ControlChartSpec(value="signal", plate="plateID",
                                      positive_levels=("pos",),
                                      negative_levels=("neg",)))


def test_zprime_without_a_run_order_column_falls_back_to_the_plate_ids():
    """With no order column the plate id is the only ordering available, and
    it has to sort naturally -- ``P9`` before ``P10`` -- or the chart's run
    rules read the campaign in the wrong sequence."""
    frame = _zprime_frame_rows(False)
    table = zprime_frame(
        frame, ControlChartSpec(value="signal", plate="plateID",
                                control_column="well_type",
                                control_levels=("pos", "neg"),
                                positive_levels=("pos",),
                                negative_levels=("neg",)))
    assert list(table["plate"]) == ["P01", "P02", "P03", "P04"]
    assert list(table["order_index"]) == [0, 1, 2, 3]
    assert (table["zprime"] < 1.0).all()
