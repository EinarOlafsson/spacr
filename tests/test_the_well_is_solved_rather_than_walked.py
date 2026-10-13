"""The solve from pairwise displacements to absolute tile positions.

Instruction 372's layout and registration live in :mod:`spacr.ops_layout`
and :mod:`spacr.ops_register` and carry their own tests. What is pinned here
is the third verb: every edge solved at once rather than walked from a seed.
"""
from __future__ import annotations

import pytest

from spacr.ops_solve import solve_placements

PITCH = 1267.0
TILE = 1480.0


# --------------------------------------------------------------------------
# From displacements to positions.
# --------------------------------------------------------------------------


def test_the_positions_come_out_of_every_edge_at_once():
    """A square of four tiles, solved from its four displacements."""
    edges = {
        (0, 1): (PITCH, 0.0),
        (0, 2): (0.0, PITCH),
        (1, 3): (0.0, PITCH),
        (2, 3): (PITCH, 0.0),
    }
    placed = solve_placements(edges, [0, 1, 2, 3])
    assert placed[0] == pytest.approx((0.0, 0.0))
    assert placed[1] == pytest.approx((PITCH, 0.0))
    assert placed[2] == pytest.approx((0.0, PITCH))
    assert placed[3] == pytest.approx((PITCH, PITCH))


def test_a_disagreeing_edge_is_spread_rather_than_believed():
    """The reason for a solve instead of a walk.

    The diagonal path to tile 3 says one thing and the other says another.
    A walk from the seed would hand tile 3 whichever answer it happened to
    reach first; the solve splits the difference, so no tile carries the
    whole of another edge's error.
    """
    edges = {
        (0, 1): (PITCH, 0.0),
        (0, 2): (0.0, PITCH),
        (1, 3): (0.0, PITCH),
        (2, 3): (PITCH + 40.0, 0.0),
    }
    placed = solve_placements(edges, [0, 1, 2, 3])
    assert placed[0] == pytest.approx((0.0, 0.0))
    # Neither 1267 nor 1307, and strictly between them.
    assert PITCH < placed[3][0] < PITCH + 40.0


def test_a_well_that_registers_in_two_pieces_still_returns_both():
    """A component with no pin of its own makes the solve singular.

    Failing there would throw away the half of the well that DID register,
    which is the opposite of what a stitch that drops tiles should do.
    """
    edges = {(0, 1): (PITCH, 0.0), (2, 3): (0.0, PITCH)}
    placed = solve_placements(edges, [0, 1, 2, 3])
    assert set(placed) == {0, 1, 2, 3}
    assert placed[0] == pytest.approx((0.0, 0.0))
    assert placed[1] == pytest.approx((PITCH, 0.0))
    assert placed[2] == pytest.approx((0.0, 0.0))
    assert placed[3] == pytest.approx((0.0, PITCH))


def test_a_site_nothing_registered_against_is_still_placed():
    placed = solve_placements({(0, 1): (PITCH, 0.0)}, [0, 1, 9])
    assert set(placed) == {0, 1, 9}
    assert placed[9] == pytest.approx((0.0, 0.0))
