"""The round-well search says no in each way a circle cannot hold a count.

`spacr.ops_layout.round_well_layout` returns a layout only when some radius
holds exactly the fields asked for. 372 PART 14-M wrote four refusals into it
and `_exact_radius` that no test drove: a window of radii narrower than float
noise, a window bound below every field, a column count no radius fills to
the count, and a tile wider than every radius the search tries. A refusal
nobody has seen fire is a refusal nobody knows still fires.
"""
from __future__ import annotations

import pytest

from spacr.ops_layout import _exact_radius, round_well_layout


def _inside(radius_squared):
    """How many grid positions sit strictly inside a centre-point circle."""
    reach = int(radius_squared ** 0.5) + 1
    return sum(1 for x in range(-reach, reach + 1)
               for y in range(-reach, reach + 1)
               if x * x + y * y < radius_squared)


def test_a_window_narrower_than_float_noise_holds_no_well():
    """(1, 7) and (5, 5) enter a centre-point circle together, at r^2 = 50.

    A footprint of 1e-12 pitches parts the two orbits by about 3e-13 of a
    pitch, so a well holding the (1, 7) orbit and not the (5, 5) one exists
    only in a window no float radius lands in reliably. The docstring's rule
    is that such a window is refused rather than bisected onto: a radius that
    close keeps some mirror-image fields and drops others.
    """
    count = _inside(50) + 8

    assert _exact_radius(count, 1e-12, 8.0) is None
    with pytest.raises(ValueError, match=f"exactly {count} fields"):
        round_well_layout(count, half_tile=1e-12)
    # The same count, with a footprint that parts the orbits by a real
    # margin, has a window and its middle.
    assert _exact_radius(count, 1e-3, 8.0) == pytest.approx(7.0723, abs=1e-3)


def test_a_bound_below_every_field_lists_no_window():
    """The search only calls this above a radius holding a field, so this is
    the guard for a caller who does not: nothing listed, nothing returned."""
    assert _exact_radius(1, 1.0, 0.5) is None


def test_a_column_count_that_cuts_the_circle_is_refused():
    """Two columns cannot hold the five fields the free circle's window does.

    The lattice window exists; the two-column layout at its radius holds a
    different number, and the search stops rather than hand back a layout of
    the wrong size.
    """
    with pytest.raises(ValueError, match="no round well holds exactly 5 fields"):
        round_well_layout(5, columns=2, half_tile=0.0)


def test_a_tile_wider_than_every_radius_tried_is_refused():
    """Half a tile of 5 pitches puts even the centre field past the search."""
    with pytest.raises(ValueError,
                       match="exactly 1 fields at a half tile of 5.0"):
        round_well_layout(1, half_tile=5.0)
