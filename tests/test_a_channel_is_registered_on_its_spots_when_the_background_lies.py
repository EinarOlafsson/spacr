"""Channel registration on the reads, for the cycle the background misplaces.

The decode-yield study of 2026-11-04 (item 752) found the weak wells of the
Funk et al. 2022 plates -- B A1/A2, C A2/A3/B2/B3, D A1 -- each carrying one
cycle whose channels the background registration had left at (0, 0) while
they had drifted 10 to 25 px: the illumination and the camera put a peak at
zero shift into every pair of channels taken at one stage position. That one
cycle held 70 to 84 % of the wells' base-call errors.

Here a field whose shared background does not move while the reads of one
cycle's channels do: ``background`` keeps the wrong (0, 0), ``spots`` finds
each channel's drift, and the brieflow median unmixing is checked to undo a
known bleed-through.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy import ndimage

from spacr.ops_cycles import align_field
from spacr.ops_sbs import compensate_crosstalk

SIZE = 256
#: The drift of cycle 2's four channels, as the bad wells showed it.
DRIFT = [(0, 0), (-6, 3), (-9, 4), (-11, 5)]


def _planes(seed=0):
    """Two cycles of four channels: a static background and moving reads."""
    rng = np.random.default_rng(seed)
    background = ndimage.gaussian_filter(rng.random((SIZE, SIZE)), 8)
    background = (background - background.min()) / np.ptp(background) * 3000 + 500
    reads = rng.integers(20, SIZE - 20, size=(400, 2))
    bases = rng.integers(0, 4, size=(400, 2))
    planes = {}
    for cycle in (1, 2):
        channels = []
        for k in range(4):
            spots = np.zeros((SIZE, SIZE), np.float32)
            dy, dx = DRIFT[k] if cycle == 2 else (0, 0)
            for (y, x), base in zip(reads, bases[:, cycle - 1]):
                if base == k:
                    spots[y + dy, x + dx] = 1.0
            spots = ndimage.gaussian_filter(spots, 1.0) * 40000
            channels.append(background + spots + rng.normal(0, 5, spots.shape))
        planes[cycle] = channels
    return planes


def _total(field, cycle):
    """Each channel's applied total shift (cycle plus channel)."""
    cy, cx = field.cycle_shifts[cycle]
    return [(cy + dy, cx + dx) for dy, dx in field.channel_shifts[cycle]]


def test_the_background_keeps_the_zero_peak_and_the_spots_find_the_drift():
    planes = _planes()
    background = align_field(planes, reference=1, gpu=False)
    spots = align_field(planes, reference=1, gpu=False, channel_mode="spots")
    wanted = [(-dy, -dx) for dy, dx in DRIFT]
    assert _total(background, 2) != wanted
    assert _total(spots, 2) == wanted
    assert {pair[0] for pair in spots.spot_replaced} == {2}
    assert background.spot_replaced == ()


def test_a_field_the_background_gets_right_is_left_alone():
    """No drift at all: the spot check agrees and replaces nothing."""
    planes = _planes()
    planes[2] = [plane.copy() for plane in planes[1]]
    field = align_field(planes, reference=1, gpu=False, channel_mode="spots")
    assert field.spot_replaced == ()


def test_an_unknown_channel_mode_is_refused():
    with pytest.raises(ValueError, match="channel_mode"):
        align_field(_planes(), reference=1, gpu=False, channel_mode="dapi")


def test_argmax_median_unmixing_undoes_a_known_bleed_through():
    """Pure bases mixed by a known matrix come back with the right argmax."""
    rng = np.random.default_rng(3)
    truth = rng.integers(0, 4, size=4000)
    pure = np.eye(4)[truth] * rng.uniform(800, 1200, size=(4000, 1))
    mix = np.array([[1.0, 0.0, 0.0, 0.0],
                    [0.9, 1.0, 0.0, 0.0],     # G bleeds strongly into T
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.6, 1.0]])    # A bleeds into C
    # A positive floor in every channel, as a filtered spot has: the median
    # over spots is that floor, which is what the normalisation divides by.
    observed = (pure @ mix.T + 200
                + rng.normal(0, 60, size=(4000, 4))).astype(np.float32)
    raw_right = (observed.argmax(1) == truth).mean()
    fixed = compensate_crosstalk(observed[:, None, :], method="argmax_median",
                                 gpu=False)[:, 0, :]
    assert (fixed.argmax(1) == truth).mean() > 0.99 > raw_right
