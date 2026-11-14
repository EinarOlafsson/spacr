"""Exchangeability diagnostics degrade to NaN instead of dividing by zero.

Both statistics normalise by a sum of squares that a real residual vector can
make zero -- a perfectly fitted block leaves residuals of exactly 0, and a
block with two usable wells has no within-level spread to measure. A quiet
``nan`` says "not measurable here"; an unguarded division would put ``inf`` or
a ZeroDivisionError into a QC report the user reads as a verdict.
"""
from __future__ import annotations

import math

import pytest

from spacr.permutation_qc import autocorrelation, block_residual_report


def test_all_zero_residuals_have_no_measurable_autocorrelation():
    """Durbin-Watson is 0/0 when every residual is zero: report nan."""
    assert math.isnan(autocorrelation([0.0, 0.0, 0.0, 0.0]))


def test_a_flat_nonzero_series_still_reports_a_number():
    """The zero guard must not swallow an ordinary constant-residual block."""
    value = autocorrelation([2.0, 2.0, 2.0])
    assert value == 0.0, "constant non-zero residuals have zero serial diff"


def test_block_report_refuses_an_unpaired_block_vector():
    """Every residual must belong to exactly one permutation block."""
    with pytest.raises(
            ValueError,
            match=r"residuals and blocks must have the same length; got 3 and 2"):
        block_residual_report([1.0, 2.0, 3.0], ["p1", "p1"])


def test_block_report_names_an_unpaired_position_column():
    """All named position columns are validated before a report is built."""
    with pytest.raises(
            ValueError,
            match=r"position column 'rowID'.*got 2 and 3"):
        block_residual_report(
            [1.0, 2.0, 3.0], ["p1", "p1", "p1"],
            {"rowID": ["A", "B"]})
