"""The cycle axis, which has no DAPI after cycle 1.

Instruction 372 PART 10 measured that this acquisition images Hoechst ONCE:
cycle 1 is a five-channel stack and cycles 2 to 11 carry four base channels
and no DAPI. PART 12 measured the replacement -- the reference
implementation's ``SBS_mean`` -- at peak/mean 169 to 281 over 20 of 20
cycles, against 23 to 85 for the tile axis on DAPI.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.ops_cycles import (accepted_cycles, register_cycles,
                              sbs_mean_target)


def _spots(shape=(96, 96), n=40, seed=0):
    """A sparse punctate field, which is what a base channel looks like."""
    rng = np.random.default_rng(seed)
    field = np.zeros(shape, dtype=np.float32)
    ys = rng.integers(4, shape[0] - 4, n)
    xs = rng.integers(4, shape[1] - 4, n)
    field[ys, xs] = 1000.0
    return field


def test_the_maximum_keeps_a_spot_the_mean_would_divide_away():
    """A read is bright in ONE channel per cycle -- that is a base call.

    Averaging four channels divides every spot by four and leaves the
    background where it was; the maximum keeps each spot at its own
    brightness whichever channel carried it.
    """
    bright, dark = _spots(seed=1), np.zeros((96, 96), dtype=np.float32)
    target = sbs_mean_target([bright, dark, dark, dark])
    mean_of_same = np.mean(np.stack([bright, dark, dark, dark]), axis=0)
    assert target.max() > 0
    assert target.max() > mean_of_same.max() / bright.max()


def test_the_target_is_normalised_and_clipped():
    """So one saturated spot cannot dominate the correlation by itself."""
    field = _spots(seed=2)
    field[10, 10] = 1e6
    target = sbs_mean_target([field, field, field, field])
    assert target.max() <= 1.0
    assert target.dtype == np.float32


def test_a_cycle_that_did_not_move_is_found_at_zero():
    field = sbs_mean_target([_spots(seed=3)])
    found = register_cycles(field, {2: field})
    assert found[2].accepted
    assert found[2].shift == (0, 0)


def test_a_shifted_cycle_is_found_where_it_was_put():
    field = sbs_mean_target([_spots(seed=4)])
    moved = np.roll(np.roll(field, 5, axis=0), -3, axis=1)
    found = register_cycles(field, {2: moved})
    assert found[2].accepted
    # THE SHIFT IS THE CORRECTION, NOT THE MOVE. It is what lands `moved`
    # back on `field`, so a roll of (+5, -3) is answered with (-5, +3).
    # Asserted against a known np.roll rather than described, because a
    # convention stated in prose is one nobody can check.
    assert found[2].shift == (-5, 3)


def test_every_cycle_is_registered_against_the_reference_not_its_predecessor():
    """A bad link must not displace the cycles after it.

    Cycle 3 is nonsense. Chained, it would carry cycle 4 with it; here 4 is
    measured against the reference and is unaffected.
    """
    rng = np.random.default_rng(5)
    field = sbs_mean_target([_spots(seed=5)])
    cycles = {
        2: np.roll(field, 2, axis=0),
        3: rng.random(field.shape).astype(np.float32),
        4: np.roll(field, 7, axis=0),
    }
    found = register_cycles(field, cycles, tolerance=16)
    assert found[2].shift == (-2, 0)
    assert found[4].shift == (-7, 0), "cycle 4 followed cycle 3's failure"


def test_a_refused_cycle_is_reported_rather_than_dropped():
    """`n_cycles` has to know a cycle was TRIED and refused.

    A missing key and a refused registration mean different things to a
    barcode assembled from what survived.
    """
    rng = np.random.default_rng(6)
    field = sbs_mean_target([_spots(seed=6)])
    found = register_cycles(
        field, {2: field, 3: rng.random(field.shape).astype(np.float32)},
        tolerance=4)
    assert set(found) == {2, 3}
    assert found[3].accepted is False
    assert accepted_cycles(found) == (2,)


def test_channels_that_disagree_in_shape_are_refused_loudly():
    with pytest.raises(ValueError, match="share a shape"):
        sbs_mean_target([np.zeros((8, 8), np.float32), np.zeros((8, 9), np.float32)])


def test_no_channels_is_refused_rather_than_returning_an_empty_field():
    with pytest.raises(ValueError, match="at least one channel"):
        sbs_mean_target([])
