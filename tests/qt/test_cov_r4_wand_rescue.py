"""Where a runaway flood is cut, capped and tapered.

Pins the rescue paths that a clean flood never reaches: a click outside the
frame, a leak that runs up or to the left of the click, a budget that is
already met, a cap taken from a click that is not in the region, a colour
image collapsed to one channel, and a capped flood whose edge is left as the
arc the cap drew. Geometry only -- nothing here imports Qt.
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
# the directional runaway detector
# --------------------------------------------------------------------------

def _leaks_upward():
    """A narrow bar under the click, opening into the full width above it."""
    region = np.zeros((20, 20), dtype=bool)
    region[6:15, 8:12] = True
    region[0:6, :] = True
    return region


def _leaks_leftward():
    """The same shape turned on its side."""
    region = np.zeros((20, 20), dtype=bool)
    region[8:12, 6:15] = True
    region[:, 0:6] = True
    return region


_DETECTOR = dict(ratio=2.0, warmup=2, min_baseline=1, confirm=1)


def test_a_click_outside_the_frame_cuts_nothing():
    """The profiles are measured from the clicked scanline. With no such
    scanline there is nothing to measure, and cutting on a guess would
    delete part of a flood nobody said was wrong."""
    region = _leaks_upward()
    out, cuts = W.trim_directional_runaway(region, (99, 99), **_DETECTOR)
    assert cuts == {}
    assert np.array_equal(out, region) and out is not region

    # The same region, clicked inside it, is cut -- so the empty answer
    # above is the click being off the frame and not the detector being deaf.
    _out, cuts = W.trim_directional_runaway(region, (10, 9), **_DETECTOR)
    assert cuts == {"up": 5}


def test_a_flood_that_escapes_upwards_is_cut_above_the_click():
    """The cut is reported as an image row, not as a distance from the
    click: a caller drawing it has only the image to draw it on."""
    region = _leaks_upward()
    out, cuts = W.trim_directional_runaway(region, (10, 9), **_DETECTOR)

    assert cuts == {"up": 5}
    assert not out[:6].any()                  # the escape, and the cut row
    assert out[6:15, 8:12].all()              # the object the click was on
    assert int(out.sum()) == 36


def test_a_flood_that_escapes_leftwards_is_cut_left_of_the_click():
    """Cut as an image column, and the object the click was on survives."""
    region = _leaks_leftward()
    out, cuts = W.trim_directional_runaway(region, (9, 10), **_DETECTOR)

    assert cuts == {"left": 5}
    assert not out[:, :6].any()
    assert out[8:12, 6:15].all()
    assert int(out.sum()) == 36


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
# the gradient taper
# --------------------------------------------------------------------------

def _tapering_case():
    image = np.zeros((24, 24), dtype=np.float32)
    image[6:18, 6:18] = 200.0
    flooded = np.zeros((24, 24), dtype=bool)
    flooded[6:18, 6:18] = True
    provisional = np.zeros((24, 24), dtype=bool)
    provisional[6:14, 6:18] = True             # a straight cut across it
    return image, flooded, provisional


def test_the_taper_reads_a_colour_image_as_its_channel_mean():
    """One intensity per pixel is what a gradient is taken of. Tapering each
    channel separately would give three different boundaries for one edge."""
    image, flooded, provisional = _tapering_case()
    grey = W.taper_region_to_intensity(image, flooded, provisional, (10, 10),
                                       margin=2)
    colour = np.repeat(image[:, :, None], 3, axis=2)
    across = W.taper_region_to_intensity(colour, flooded, provisional,
                                         (10, 10), margin=2)

    assert not np.array_equal(grey, provisional)   # the taper really moved it
    assert np.array_equal(grey, across)


def test_a_taper_from_a_click_off_the_provisional_edge_changes_nothing():
    """The result always keeps the connected piece the click is in, and a
    click that is not in the provisional region names no such piece."""
    image, flooded, provisional = _tapering_case()
    off = W.taper_region_to_intensity(image, flooded, provisional, (20, 20),
                                      margin=2)
    assert np.array_equal(off, provisional) and off is not provisional

    on = W.taper_region_to_intensity(image, flooded, provisional, (10, 10),
                                     margin=2)
    assert not np.array_equal(on, provisional)


# --------------------------------------------------------------------------
# the whole wand
# --------------------------------------------------------------------------

def _leaky_image():
    """An object joined to a bright field by a one-pixel seam."""
    image = np.zeros((64, 64), dtype=np.float32)
    image[20:31, 20:31] = 200.0                # the object
    image[25, 30:41] = 200.0                   # the seam it escapes along
    image[10:51, 40:61] = 200.0                # the field it escapes into
    return image


def test_a_wand_on_a_colour_image_answers_as_the_same_image_in_grey():
    """The rescues are measured on one intensity, so a colour image has to
    be collapsed once, up front -- not rescued three times."""
    image = _leaky_image()
    colour = np.repeat(image[:, :, None], 3, axis=2)

    grey_region, grey_report = W.wand_region(image, 25, 25, 10.0,
                                             intensity_border=False,
                                             gradient_taper=False)
    region, report = W.wand_region(colour, 25, 25, 10.0,
                                   intensity_border=False,
                                   gradient_taper=False)

    assert report["cuts"] == ["right"]         # the leak was seen either way
    assert np.array_equal(region, grey_region)
    assert report == grey_report


def test_with_the_intensity_border_off_the_straight_cut_is_what_is_kept():
    """Each rescue can be turned off on its own. Without the reflood there
    is no refined tolerance to report, and the half-plane cut the detector
    made is the answer."""
    image = _leaky_image()
    region, report = W.wand_region(image, 25, 25, 10.0,
                                   intensity_border=False,
                                   gradient_taper=False)

    assert report["cuts"] == ["right"]
    assert report["intensity_border"] is False
    assert report["refined_tolerance"] == 10.0
    assert report["flooded_px"] == 991 and report["kept_px"] == 130
    assert not region[:, 40:].any()            # cut at the reported column
    assert region[20:31, 20:31].all()          # the object the click was on


def test_a_capped_flood_with_the_taper_off_keeps_exactly_the_budget():
    """The cap ends in an arc. Leaving it there is what "no gradient taper"
    means, and the budget is then met exactly rather than approximately."""
    image = np.zeros((40, 40), dtype=np.float32)
    image[10:31, 10:31] = 200.0                # 441 pixels, budget of 40

    region, report = W.wand_region(image, 20, 20, 10.0, max_pixels=40,
                                   gradient_taper=False)
    assert report["capped"] is True
    assert report["tapered"] is False
    assert int(region.sum()) == 40 == report["kept_px"]

    # With the taper on, the same cap IS given an intensity edge, so the
    # untapered answer above is the switch and not a taper that never runs.
    _region, tapered = W.wand_region(image, 20, 20, 10.0, max_pixels=40,
                                     gradient_taper=True)
    assert tapered["capped"] is True and tapered["tapered"] is True


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
