"""Plate QC says "undefined" rather than printing a number it did not compute.

The module's rule is that a NaN in a QC report is indistinguishable from
"nobody looked", so every statistic that cannot be computed comes back as
None and is rendered as the reason. These tests drive the paths where that
rule is under pressure: values that are not numbers, a comparison scipy
cannot rank, wells whose labels cannot be read, a forced plate geometry too
small for the data, and a report asked for a ring or an axis it never
measured.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as _sps

from spacr import plate_qc as pq


# ---------------------------------------------------------------------------
# Undefined numbers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, "not a number", object(),
                                   float("nan"), float("inf")])
def test_a_value_that_is_not_a_finite_number_is_undefined_not_zero(value):
    """Returning 0.0 here would put a measured-looking zero on a QC report
    for a statistic that was never computed."""
    assert pq._finite(value) is None


def test_a_comparison_with_an_empty_group_has_no_answer():
    """An outer ring with no wells cannot be compared with the interior, and
    a p-value invented for it would read as evidence of no effect."""
    assert pq._rank_compare(np.array([]), np.array([1.0, 2.0, 3.0])) == (None, None)
    assert pq._rank_compare(np.array([1.0, 2.0]), np.array([])) == (None, None)


def test_a_rank_test_that_refuses_the_data_reports_no_result(monkeypatch):
    """scipy declines some inputs outright. The refusal has to surface as
    'undefined', never as an exception out of a QC pass over a plate."""
    def refuse(*_args, **_kwargs):
        raise ValueError("scipy will not rank this")

    monkeypatch.setattr(_sps, "mannwhitneyu", refuse)

    assert pq._rank_compare(np.array([1.0, 2.0, 3.0]),
                            np.array([4.0, 5.0, 6.0])) == (None, None)


def test_a_number_outside_the_readable_range_is_written_in_exponent_form():
    """A plate of raw integrated intensities runs to millions; printing them
    at three decimal places makes a QC line unreadable."""
    assert pq._fmt_num(1.0e6) == "1e+06"
    assert pq._fmt_num(1.0e-6) == "1e-06"
    assert pq._fmt_num(1.5) == "1.5"


# ---------------------------------------------------------------------------
# Well labels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label", [None, "A0", "not a well", ""])
def test_a_label_that_is_not_a_well_position_is_refused(label):
    """A well at column 0 does not exist. Accepting it would shift the whole
    ring geometry by one column."""
    assert pq._parse_well_label(label) is None


def test_a_real_well_label_still_parses():
    """The contrast that shows the refusals are refusals, not a broken
    parser."""
    assert pq._parse_well_label("A01") == (1, 1)
    assert pq._parse_well_label("AF48") == (32, 48)


def test_an_identifier_with_too_few_parts_is_not_a_plate_row_column():
    """`plateID_rowID_columnID` needs three tokens. Two tokens is a
    different identifier, and guessing which is which would silently lay the
    plate out transposed."""
    assert pq._prc_parts(pd.Series(["p1_r1", "p2_r2"])) is None


# ---------------------------------------------------------------------------
# Locating wells
# ---------------------------------------------------------------------------

def test_row_and_column_columns_take_their_plate_from_the_identifier():
    """A frame carrying rowID/columnID and a `prc` string has a plate name
    already; falling back to 'p1' would merge two plates into one heatmap."""
    frame = pd.DataFrame({"rowID": ["r1", "r2"], "columnID": ["c1", "c2"],
                          "prc": ["p9_r1_c1", "p9_r2_c2"], "v": [1.0, 2.0]})

    located, _notes = pq._identify_wells(frame)

    assert located["plateID"].tolist() == ["p9", "p9"]


def test_a_frame_with_no_plate_column_says_it_invented_the_plate_name():
    """Every well lands on plate 'p1'. That is a choice, and the note is the
    only thing that stops it reading as a plate name from the data."""
    frame = pd.DataFrame({"rowID": ["r1", "r2"], "columnID": ["c1", "c2"],
                          "v": [1.0, 2.0]})

    located, notes = pq._identify_wells(frame)

    assert located["plateID"].tolist() == ["p1", "p1"]
    assert any("every well assigned to plate 'p1'" in n for n in notes)


def test_labels_that_cannot_be_read_leave_an_empty_layout_that_counts_them():
    """An empty grid is the honest answer, and the count of skipped rows is
    what tells the user their labels are the problem."""
    frame = pd.DataFrame({"rowID": ["??", "??"], "columnID": ["??", "??"],
                          "v": [1.0, 2.0]})

    layout = pq.plate_layout(frame, value_col="v")

    assert len(layout) == 0
    assert list(layout.columns) == list(pq._empty_layout().columns)
    assert layout.attrs["n_unparsed_rows"] == 2
    assert any("could not be read as a well position" in n
               for n in layout.attrs["notes"])


# ---------------------------------------------------------------------------
# Forced geometry
# ---------------------------------------------------------------------------

def _sixteen_by_sixteen():
    return pd.DataFrame({"rowID": [f"r{i}" for i in range(1, 17)],
                         "columnID": [f"c{i}" for i in range(1, 17)],
                         "v": [float(i) for i in range(16)]})


def test_a_plate_format_that_is_not_a_plate_is_refused_by_name():
    """The message lists the formats that exist, because the caller passing
    999 has no other way to find out what it should have passed."""
    with pytest.raises(ValueError) as excinfo:
        pq.plate_layout(_sixteen_by_sixteen(), value_col="v",
                        plate_format=999)

    assert "999" in str(excinfo.value)
    assert "384" in str(excinfo.value)


def test_a_forced_geometry_smaller_than_the_data_is_grown_and_reported():
    """Wells outside the forced grid must not be dropped off the plate. The
    grid grows to hold them and the note says the forced format did not
    fit."""
    layout = pq.plate_layout(_sixteen_by_sixteen(), value_col="v",
                             plate_format=96)

    assert layout.attrs["plate_format"] == 96
    assert (layout.attrs["n_rows"], layout.attrs["n_cols"]) == (16, 16)
    assert any("smaller than the observed 16x16 extent" in n
               for n in layout.attrs["notes"])
    assert "P16" in set(layout["well"]), (
        "the well past the forced 8x12 grid was dropped")


# ---------------------------------------------------------------------------
# Colour limits and trends
# ---------------------------------------------------------------------------

def test_reversed_colour_limits_are_put_back_in_order():
    """A user who types the high limit first gets the scale they meant, not
    an inverted colour map that reads as an inverted result."""
    layout = pq.plate_layout(_sixteen_by_sixteen(), value_col="v")

    assert pq.colour_limits(layout, min_max=[10, 2]) == (2.0, 10.0)


def test_a_plate_of_unmeasured_wells_yields_no_trend_rows():
    """The wells exist but none has a value. Emitting per-row means of
    nothing would be a table of NaN presented as a summary."""
    frame = pd.DataFrame({"prc": ["p1_r1_c1", "p1_r1_c2", "p1_r2_c1"],
                          "v": [np.nan, np.nan, np.nan]})

    trends = pq.row_column_trends(frame, value_col="v")

    assert len(trends) == 0
    assert "spearman_rho" in trends.columns
    assert "axis" in trends.columns


# ---------------------------------------------------------------------------
# Reading a report back
# ---------------------------------------------------------------------------

def _report(**kwargs):
    return pq.EdgeEffectReport(plate="p1", value_col="v", grouping="mean",
                               **kwargs)


def test_an_axis_that_was_never_measured_comes_back_as_nothing():
    """A caller asking for the row gradient of a plate that has none must
    get None, not a GradientStats full of zeros."""
    report = _report(gradients=[pq.GradientStats(
        axis="column", spearman_rho=0.4, p_value=0.01, first_label="1",
        last_label="12", delta_first_last=1.0, pct_first_last=10.0,
        detected=True)])

    assert report.gradient("row") is None
    assert report.gradient("column").axis == "column"


def test_a_ring_the_profile_never_reached_comes_back_as_nothing():
    """The profile stops at the core, so a caller can ask for a ring that
    was deliberately not compared."""
    ring = pq.RingStats(ring=0, n_wells=44, median=1.0, mean=1.0, delta=0.2,
                        pct=20.0, p_value=0.001, cliffs_delta=0.6)
    report = _report(rings=[ring])

    assert report.ring(0) is ring
    assert report.ring(3) is None


def test_an_edge_difference_nobody_could_measure_says_so_in_words():
    """'+0.0 %' would read as 'measured, and it is zero'. The words are the
    difference between no effect and no measurement."""
    assert _report().magnitude == "an undetermined amount"
    assert "absolute units" in _report(median_difference=0.25).magnitude
    assert _report(pct_difference=12.0,
                   median_difference=0.25).magnitude == "+12.0 %"


# ---------------------------------------------------------------------------
# The ring profile
# ---------------------------------------------------------------------------

def test_a_plate_with_no_wells_has_no_ring_profile():
    """There is no core to compare rings against, so the profile is empty
    rather than a list of rings compared with nothing."""
    assert pq._ring_profile(np.array([]), np.array([]), 2, 4, _report()) == []


def test_a_ring_with_no_surviving_wells_is_skipped_not_reported_as_empty():
    """min_count can empty a whole ring. A RingStats over zero wells would
    put a row on the profile claiming a median it does not have."""
    values = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
    ring_index = np.array([1, 1, 1, 2, 2, 2])

    rings = pq._ring_profile(values, ring_index, 2, 4, _report())

    assert [r.ring for r in rings] == [1], "the empty ring 0 was reported"
    assert rings[0].n_wells == 3
