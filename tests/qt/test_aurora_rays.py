"""Sharper rays that breathe, each at its own rate.

Asked for as: "the rays of light should be even sharper and they should be
dynamic, meaning each ray should at different speeds be changing length to
look more like real aurora."
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def engine(qapp):
    from spacr.qt.widgets.ambient import make_engine

    return make_engine("aurora", "spacr", "#000000")


# ---------------------------------------------------------------------------
# Sharper
# ---------------------------------------------------------------------------

def test_the_rays_are_narrow():
    """Two numbers make an edge and only together.

    Narrowing alone leaves thin rays on a bright sheet, which reads as a
    lighter curtain rather than as a defined ray -- so the floor between
    them has to come down with the widths.
    """
    from spacr.qt.widgets.ambient import AURORA_TILE_FLOOR, AURORA_TILE_RAYS

    widest = max(half for _centre, half, _strength in AURORA_TILE_RAYS)
    assert widest <= 0.08, (
        f"the widest ray covers {widest:.3f} of the tile; it was sharpened "
        f"to well under 0.08")
    assert AURORA_TILE_FLOOR <= 0.40, (
        f"the sheet between the rays is at {AURORA_TILE_FLOOR}, bright "
        f"enough to swallow the edge the narrowing just made")


def test_the_rays_do_not_overlap():
    """Overlapping bands make one wide ray out of two narrow ones."""
    from spacr.qt.widgets.ambient import AURORA_TILE_RAYS

    spans = sorted((centre - half, centre + half)
                   for centre, half, _strength in AURORA_TILE_RAYS)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert start > end, f"rays overlap: {spans}"


# ---------------------------------------------------------------------------
# Alive
# ---------------------------------------------------------------------------

def test_each_ray_breathes_at_its_own_rate(engine):
    """Three periods that are not multiples of one another.

    Multiples would return the three to the same arrangement on a short
    cycle, and the eye finds a repeat far faster than it finds a rhythm.
    """
    from spacr.qt.widgets.ambient import AURORA_RAY_LIFE

    periods = [period for period, _offset in AURORA_RAY_LIFE]
    assert len(set(periods)) == len(periods), "two rays share a period"
    for i, a in enumerate(periods):
        for b in periods[i + 1:]:
            ratio = max(a, b) / min(a, b)
            assert abs(ratio - round(ratio)) > 0.05, (
                f"{a}s and {b}s are near-multiples; they would re-sync")


def test_the_lengths_actually_change(engine):
    curtain = engine.curtains[0]
    seen = set()
    for t in range(0, 60, 3):
        engine.set_time(float(t))
        seen.add(engine.ray_lengths(curtain))
    assert len(seen) > 5, (
        f"only {len(seen)} distinct arrangements in a minute; the rays are "
        f"not breathing")


def test_the_full_range_is_used(engine):
    """The quantisation must not eat the range.

    Quantising the MAPPED value rather than the unit gave four levels
    spanning 0.80..0.98 out of an intended 0.55..0.98 -- a fifth of the
    depth, and it looked like a subtle shimmer rather than like weather.
    """
    from spacr.qt.widgets.ambient import AURORA_RAY_LENGTH

    low, high = AURORA_RAY_LENGTH
    seen = []
    for t in range(0, 120):
        engine.set_time(float(t))
        seen.extend(engine.ray_lengths(engine.curtains[0]))
    assert min(seen) == pytest.approx(low, abs=0.02)
    assert max(seen) == pytest.approx(high, abs=0.02)


def test_a_ray_never_vanishes_or_fills_the_curtain(engine):
    """Either extreme stops reading as a ray.

    Zero leaves a gap that looks like a rendering fault; the full height
    reads as the curtain itself.
    """
    from spacr.qt.widgets.ambient import AURORA_RAY_LENGTH

    low, high = AURORA_RAY_LENGTH
    assert 0.0 < low < high <= 1.0
    assert low >= 0.3, "the shortest ray is short enough to look broken"


def test_the_three_rays_are_not_in_step(engine):
    """Otherwise it is one ray drawn three times."""
    curtain = engine.curtains[0]
    identical = 0
    for t in range(0, 60):
        engine.set_time(float(t))
        lengths = engine.ray_lengths(curtain)
        if len(set(lengths)) == 1:
            identical += 1
    assert identical < 20, (
        f"the three rays held the same length on {identical} of 60 samples")


def test_the_lengths_are_quantised(engine):
    """Or every tile is rebuilt every frame.

    The tile is a cached texture keyed partly on these values; a length
    that followed the clock exactly would defeat the cache the aurora
    depends on for its frame budget.
    """
    from spacr.qt.widgets.ambient import AURORA_LENGTH_STEPS

    seen = set()
    for t in range(0, 400):
        engine.set_time(t * 0.1)
        seen.update(engine.ray_lengths(engine.curtains[0]))
    assert len(seen) <= AURORA_LENGTH_STEPS, (
        f"{len(seen)} distinct lengths for {AURORA_LENGTH_STEPS} steps")


def test_the_tile_cache_can_hold_the_working_set():
    """A cache one short of the working set is cleared every frame.

    The file already carries that lesson in a comment; the breathing
    multiplies the working set by the number of length steps.
    """
    from spacr.qt.widgets.ambient import (AURORA_HUE_STEPS,
                                          AURORA_LENGTH_STEPS,
                                          AURORA_TILE_CACHE)

    busiest = 9 * AURORA_HUE_STEPS * AURORA_LENGTH_STEPS
    assert AURORA_TILE_CACHE >= busiest, (
        f"cache holds {AURORA_TILE_CACHE}, the densest aurora needs "
        f"{busiest}")
