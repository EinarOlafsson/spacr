"""Where a runaway flood is cut, capped and tapered.

Pins the colour distance of the flood itself and the pixel budget: a budget
that is already met, and a cap taken from a click that is not in the region.
The directional cut, the taper and the whole wand are walked as case tables
in ``tests/qt/test_wand_rescue_cut_cap_taper.py``. Geometry only -- nothing
here imports Qt.
"""
from __future__ import annotations

import numpy as np

from spacr.qt import wand_rescue as W


# --------------------------------------------------------------------------
# the flood itself
# --------------------------------------------------------------------------

def test_a_colour_flood_measures_distance_across_all_the_channels():
    """Straight across the channels, not channel by channel: a pixel that is
    three greener and four redder than the click is five away from it, so a
    tolerance of 4.5 leaves it out where a per-channel rule would take it."""
    image = np.full((5, 5, 3), 10.0, dtype=np.float32)
    image[0, 4] = (13.0, 14.0, 10.0)          # 3-4-5 from the seed's colour

    tight = W.flood_region(image, seed_x=0, seed_y=0, tolerance=4.5)
    loose = W.flood_region(image, seed_x=0, seed_y=0, tolerance=5.5)

    assert tight.shape == (5, 5)              # one plane, not three
    assert not tight[0, 4] and loose[0, 4]
    assert int(tight.sum()) == 24 and int(loose.sum()) == 25


# --------------------------------------------------------------------------
# the pixel budget
# --------------------------------------------------------------------------

def _block(size: int = 4) -> np.ndarray:
    region = np.zeros((10, 10), dtype=bool)
    region[2:2 + size, 2:2 + size] = True
    return region


def test_a_region_already_within_the_budget_is_handed_back_whole():
    """Growing it out again from the seed would reorder nothing and could
    only lose pixels to a rounding of the limit."""
    region = _block()                          # 16 pixels
    kept = W.cap_region_from_seed(region, (3, 3), 100)
    assert np.array_equal(kept, region) and kept is not region

    # The same region over the budget IS trimmed, so the whole answer above
    # is the budget being met rather than the cap doing nothing.
    assert int(W.cap_region_from_seed(region, (3, 3), 5).sum()) == 5


def test_a_cap_taken_from_a_click_off_the_region_keeps_nothing():
    """Nearest is measured through the region, so a click that is not in it
    has nothing to grow from. Keeping an arbitrary prefix of the scan order
    would hand back a piece of a thing the user never clicked."""
    region = _block()
    assert int(W.cap_region_from_seed(region, (0, 0), 5).sum()) == 0
    assert int(W.cap_region_from_seed(region, (3, 3), 5).sum()) == 5


# --------------------------------------------------------------------------
# Three guards in this module cannot be made to fire, and are left standing
# rather than silenced. Written down here so the next reader does not spend
# the afternoon looking for an input that reaches them.
#
# 1. `taper_region_to_intensity`, "not background.any() or not
#    foreground.any()" (the third early return).
#    * `foreground` always holds the click: the guard above it has already
#      returned unless `provisional[y, x]`, and the disc term
#      `((xx - x)**2 + (yy - y)**2 <= 4) & provisional` is true at (y, x)
#      itself, whatever `binary_erosion` did.
#    * `background` is never empty either: `removed.any()` has been checked,
#      every set pixel of a non-empty binary mask is at least 1 away from
#      the nearest unset one, so `depth.max() >= 1` and the fallback cutoff
#      `max(1.0, depth.max() * 0.75)` is at most `depth.max()` -- the
#      deepest pixel of `removed` always satisfies it.
#    Four thousand random (mask, click, margin, erosion) combinations reach
#    it zero times.
#
# 2. `wand_region`, the false side of "if tapered.any()" after the
#    directional cut. `taper_region_to_intensity` cannot return an empty
#    mask for the arguments given here: every one of its exits is either
#    `provisional.copy()` or `components == components[y, x]`, `provisional`
#    is the region the flood was cut down to and still holds the click, and
#    the watershed keeps a marker pixel's own label -- so the answer always
#    holds the click too.
#
# 3. `wand_region`, the exhausted-loop exit of "for band in sorted(...)"
#    after the cap. The band list is built from a set that always contains
#    1, and at `band == 1` every pixel of `removed` is a background marker
#    (`depth >= 1` everywhere in it), so the only unmarked pixels lie inside
#    `bounded` and the result is a subset of it: `0 < tapered.sum() <= limit`
#    holds and the loop breaks. The last band can never be rejected.
