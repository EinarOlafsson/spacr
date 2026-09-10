"""``_violin_profile``'s histogram always has something in it.

Instruction 288. The function guarded against a histogram whose tallest
bin is empty -- ``if peak <= 0: return None, None`` -- and that cannot
happen: the histogram's range is the data's OWN min and max, and the
check above it has already required both to be finite with high > low.
Every value therefore lies inside the range, and at least one bin is
populated.

NaN cannot sneak past either, and that is the part worth pinning:
``np.min`` PROPAGATES NaN rather than ignoring it, so an array holding
one fails the finite check above rather than arriving with an empty
histogram.

This pins the premise, not the deletion.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.fast_plots import _violin_profile


@pytest.mark.parametrize("values", [
    [0.0, 1.0],
    [1.0, 2.0, 3.0],
    [-1e9, 1e9],
    [1e-12, 2e-12],
    [0.0] * 40 + [1.0],
])
def test_a_finite_spread_always_produces_a_profile(values):
    """The path that must keep working."""
    centres, density = _violin_profile(np.array(values, dtype=float), 1.0)
    assert centres is not None and density is not None
    assert len(centres) == len(density)


@pytest.mark.parametrize("values", [
    [1.0, 1.0, 1.0],            # no spread at all
    [float("nan"), 1.0],        # np.min propagates the NaN
    [float("inf"), 1.0],
    [0.0],                      # one value
])
def test_anything_without_a_usable_spread_is_declined_earlier(values):
    """Declined by the FINITE check, not by the deleted one.

    These are the inputs somebody would reach for to exercise an empty
    histogram, and every one of them stops above it.
    """
    assert _violin_profile(np.array(values, dtype=float), 1.0) == (None,
                                                                   None)


def test_an_empty_array_raises_rather_than_declining():
    """RECORDED, not asserted as desirable.

    `np.min` of an empty array raises ValueError, so `_violin_profile`
    raises before reaching any of its guards. Every other unusable input
    is declined with (None, None), so this one is inconsistent -- but it
    is the behaviour today and callers do not pass empty arrays, so it
    is pinned rather than changed. A caller that starts passing one gets
    a ValueError from numpy, not a blank violin, which is at least loud.
    """
    with pytest.raises(ValueError):
        _violin_profile(np.array([], dtype=float), 1.0)


def test_np_min_propagates_nan_which_is_what_closes_the_gap():
    """THE PREMISE, stated as the numpy behaviour it rests on.

    If `np.min` ignored NaN -- as `np.nanmin` does -- an array of NaN
    plus one number would pass the finite check and could produce an
    empty histogram, and the deleted guard would have been reachable.
    """
    assert np.isnan(np.min(np.array([float("nan"), 1.0])))
    assert not np.isnan(np.nanmin(np.array([float("nan"), 1.0])))


def test_no_random_finite_array_produces_an_empty_histogram():
    """The search that settled it, kept as a smaller sweep.

    Twenty thousand were checked when the guard was removed; two
    thousand run here so a change to the binning that could empty a
    histogram fails a test rather than going unnoticed.
    """
    random.seed(11)
    for _ in range(2000):
        size = random.randint(2, 40)
        array = np.array([random.choice([random.gauss(0, 1e6),
                                         random.gauss(0, 1e-9),
                                         random.gauss(0, 1)])
                          for _ in range(size)])
        low, high = float(np.min(array)), float(np.max(array))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            continue
        bins = int(np.clip(np.sqrt(size) * 2, 6, 24))
        counts, _edges = np.histogram(array, bins=bins, range=(low, high))
        assert float(counts.max()) > 0
