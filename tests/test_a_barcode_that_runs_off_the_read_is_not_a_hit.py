"""Item 288: two edges of the barcode search the other tests step around.

* A long barcode sharing its prefix with a short one can start near the end
  of a read and run off it. That is not a hit, and it must not stop the
  search from finding the long one in a read it does fit.
* The two search thresholds that are counts and shares refuse a value that
  could never produce a real verdict, naming the setting.
"""
from __future__ import annotations

import pytest

from spacr import barcode_search as bs


@pytest.fixture
def table():
    """A 9-base and a 12-base barcode with the same 8-base prefix."""
    return bs.BarcodeTable(name="t", sequences={"ACGTACGTC": "short9",
                                                "ACGTACGTAAAA": "long12"})


def test_a_long_barcode_running_off_the_end_is_not_a_hit(table):
    hits = bs.annotate_read("GGACGTACGTAA", [table], {"t": bs.AS_GIVEN})
    assert hits == ()


def test_the_same_barcode_is_found_where_it_fits(table):
    hits = bs.annotate_read("GGACGTACGTAAAA", [table], {"t": bs.AS_GIVEN})
    assert [(h.start, h.end, h.barcode) for h in hits] == [(2, 14, "long12")]


def test_a_verdict_that_needs_no_reads_is_refused():
    with pytest.raises(ValueError) as excinfo:
        bs.SearchThresholds(min_reads_for_verdict=0)
    assert "min_reads_for_verdict must be at least 1" in str(excinfo.value)


@pytest.mark.parametrize("share", [0.0, 1.5])
def test_an_offset_window_share_outside_zero_to_one_is_refused(share):
    with pytest.raises(ValueError) as excinfo:
        bs.SearchThresholds(offset_window_coverage=share)
    assert "offset_window_coverage is a share of hits" in str(excinfo.value)
