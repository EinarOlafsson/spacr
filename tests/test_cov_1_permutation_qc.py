"""Exchangeability diagnostics degrade to NaN instead of dividing by zero.

Both statistics normalise by a sum of squares that a real residual vector can
make zero -- a perfectly fitted block leaves residuals of exactly 0, and a
block with two usable wells has no within-level spread to measure. A quiet
``nan`` says "not measurable here"; an unguarded division would put ``inf`` or
a ZeroDivisionError into a QC report the user reads as a verdict.
"""
from __future__ import annotations

import math

from spacr.permutation_qc import autocorrelation, position_effect


def test_all_zero_residuals_have_no_measurable_autocorrelation():
    """Durbin-Watson is 0/0 when every residual is zero: report nan."""
    assert math.isnan(autocorrelation([0.0, 0.0, 0.0, 0.0]))


def test_a_flat_nonzero_series_still_reports_a_number():
    """The zero guard must not swallow an ordinary constant-residual block."""
    value = autocorrelation([2.0, 2.0, 2.0])
    assert value == 0.0, "constant non-zero residuals have zero serial diff"


def test_too_few_wells_make_the_position_effect_unmeasurable():
    """Two finite residuals cannot separate between- from within-position."""
    result = position_effect([1.0, 2.0], ["A", "B"])
    assert math.isnan(result["eta_squared"])
    assert result["levels"] == 0.0


def test_non_finite_residuals_are_dropped_before_the_count():
    """A block of mostly-NaN residuals is too small, not accidentally big.

    ``np.nan`` rows come from wells the model could not fit. Counting them
    would let a block with one real well claim a positional effect.
    """
    result = position_effect([1.0, float("nan"), float("inf"), 2.0],
                             ["A", "B", "A", "B"])
    assert math.isnan(result["eta_squared"])
    assert result["levels"] == 0.0
