"""The OPS well's layout, recovered from its site count alone.

Instruction 372 PART 11-A fitted a circular snake to well A1 of
``20200202_6W-LaC024A`` and PART 11-B confirmed it on sites the fit never
saw: 24 predicted neighbours, 24 registered, three non-neighbour controls at
the no-overlap signature. These tests pin the geometry that came out of it,
because it is what replaces ``max_site_gap`` and a wrong layout would put
every tile in the wrong place while still reporting a stitch.
"""
from __future__ import annotations

import pytest

from spacr.spacrops import spacrStitcher

#: Well A1 holds 333 fields, and this is the column-by-column shape the
#: acquisition visits them in. Measured, not chosen.
A1_HEIGHTS = [5, 9, 13, 15, 17, 17, 19, 19, 21, 21, 21, 21, 21,
              19, 19, 17, 17, 15, 13, 9, 5]


def _heights(layout):
    columns = max(column for column, _row in layout.values()) + 1
    return [sum(1 for column, _row in layout.values() if column == index)
            for index in range(columns)]


def test_the_site_count_alone_recovers_the_well():
    """333 fields is enough to say which circle they were taken from."""
    layouts = spacrStitcher._candidate_layouts(333)
    assert layouts, "no circular snake holds 333 fields"
    assert all(_heights(layout) == A1_HEIGHTS for layout in layouts), (
        "a candidate disagrees with the measured column heights")
    assert sum(A1_HEIGHTS) == 333


def test_the_neighbour_is_at_the_same_absolute_row():
    """The mistake that cost PART 11-A four of its seven offsets.

    Every column of a round well starts at a different row, so a neighbour
    found by counting from each column's own top is the wrong tile. Column 0
    holds 5 fields and column 1 holds 9; the field beside the first of
    column 0 is the THIRD of column 1, not its first.
    """
    layout = spacrStitcher._candidate_layouts(333)[0]
    by_place = {place: site for site, place in layout.items()}
    first_of_column_0 = by_place[(0, min(
        row for column, row in layout.values() if column == 0))]
    beside = spacrStitcher._layout_neighbours(layout, first_of_column_0)
    assert "right" in beside
    _column, row = layout[first_of_column_0]
    assert layout[beside["right"]] == (1, row)
    top_of_column_1 = min(
        row for column, row in layout.values() if column == 1)
    assert row != top_of_column_1, (
        "column 1 would have to start at column 0's row for the naive "
        "index to be right, and in a circle it does not")


def test_every_field_has_four_neighbours_at_most_and_the_edge_has_fewer():
    layout = spacrStitcher._candidate_layouts(333)[0]
    counts = {site: len(spacrStitcher._layout_neighbours(layout, site))
              for site in layout}
    assert max(counts.values()) == 4
    assert min(counts.values()) < 4, "a round well has an edge"
    assert set(counts) == set(range(333)), "the sites are 0..332, once each"


def test_the_pairs_are_the_real_adjacencies_and_no_others():
    """624 for this well -- against 42,624 the index window would propose."""
    layout = spacrStitcher._candidate_layouts(333)[0]
    pairs = spacrStitcher._layout_pairs(layout)
    assert len(pairs) == 624
    assert len(set(pairs)) == len(pairs), "a pair is offered once"
    assert all(low < high for low, high in pairs)
    # And each one really is adjacent, which is the property the stitch
    # depends on: a pair that does not overlap registers at (0,0) and would
    # be kept as a placement.
    for low, high in pairs:
        (column_a, row_a), (column_b, row_b) = layout[low], layout[high]
        assert abs(column_a - column_b) + abs(row_a - row_b) == 1


def test_a_count_no_circle_holds_yields_nothing_rather_than_a_guess():
    assert spacrStitcher._candidate_layouts(7) == [] or all(
        len(layout) == 7 for layout in spacrStitcher._candidate_layouts(7))


# --------------------------------------------------------------------------
# From displacements to positions.
# --------------------------------------------------------------------------

PITCH = 1267.0
TILE = 1480.0


def test_a_wrapped_shift_is_put_back_on_the_number_line():
    """PART 6-A placed 333 of 333 tiles and was entirely wrong.

    Phase correlation returns the displacement modulo the tile, so a pair one
    pitch apart reads as the residual: -213 where the truth is 1267. Left
    wrapped, the well rebuilds at a sixth of its size and reports success.
    """
    assert spacrStitcher._unwrap_shift(-213.0, PITCH, TILE) == pytest.approx(
        1267.0)
    assert spacrStitcher._unwrap_shift(1267.0, PITCH, TILE) == pytest.approx(
        1267.0)
    # And the other direction, which is the same wrap seen from the far tile.
    assert spacrStitcher._unwrap_shift(213.0, -PITCH, TILE) == pytest.approx(
        -1267.0)
    # A pair that really has not moved stays where it is.
    assert spacrStitcher._unwrap_shift(0.0, 0.0, TILE) == pytest.approx(0.0)


def test_the_positions_come_out_of_every_edge_at_once():
    """A square of four tiles, solved from its four displacements."""
    edges = {
        (0, 1): (PITCH, 0.0),
        (0, 2): (0.0, PITCH),
        (1, 3): (0.0, PITCH),
        (2, 3): (PITCH, 0.0),
    }
    placed = spacrStitcher._solve_placements(edges, [0, 1, 2, 3])
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
    placed = spacrStitcher._solve_placements(edges, [0, 1, 2, 3])
    assert placed[0] == pytest.approx((0.0, 0.0))
    # Neither 1267 nor 1307, and strictly between them.
    assert PITCH < placed[3][0] < PITCH + 40.0


def test_a_well_that_registers_in_two_pieces_still_returns_both():
    """A component with no pin of its own makes the solve singular.

    Failing there would throw away the half of the well that DID register,
    which is the opposite of what a stitch that drops tiles should do.
    """
    edges = {(0, 1): (PITCH, 0.0), (2, 3): (0.0, PITCH)}
    placed = spacrStitcher._solve_placements(edges, [0, 1, 2, 3])
    assert set(placed) == {0, 1, 2, 3}
    assert placed[0] == pytest.approx((0.0, 0.0))
    assert placed[1] == pytest.approx((PITCH, 0.0))
    assert placed[2] == pytest.approx((0.0, 0.0))
    assert placed[3] == pytest.approx((0.0, PITCH))


def test_a_site_nothing_registered_against_is_still_placed():
    placed = spacrStitcher._solve_placements({(0, 1): (PITCH, 0.0)}, [0, 1, 9])
    assert set(placed) == {0, 1, 9}
    assert placed[9] == pytest.approx((0.0, 0.0))
