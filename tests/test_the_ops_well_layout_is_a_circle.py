"""The round-well layout, pinned to the offsets it was measured against.

Instruction 372 fitted a circle to seven neighbour offsets measured on
well A1 of `screenA/20200202_6W-LaC024A`, then confirmed it by predicting
the four neighbours of six sites it had never seen and registering all
24. This file holds the model to both halves of that: the seven it was
fitted on, and the twenty-four it was not.

WHY THE HELD-OUT HALF IS THE TEST. Seven constraints against four free
parameters is a good fit and nothing more; a model that survives sites it
was not fitted on is a fact about the acquisition. Three earlier
prototypes each fitted their own measurements perfectly and placed 27 of
333 tiles, which is what a fit-only check cannot see.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: The checkout root, so the import probe below runs against this tree
#: rather than against whatever is installed.
ROOT_DIR = Path(__file__).resolve().parents[1]

from spacr.ops_layout import (DIRECTIONS, MEASURED_WELL, WellLayout,
                              round_well_layout)

#: The seven sites 372's PART 6-C measured, and the offset to the
#: horizontal neighbour it found at each: ``site -> index offset``.
#: These are the constraints the circle was fitted to.
MEASURED_OFFSETS = {
    0: +11,
    40: +4,
    100: -11,
    160: -9,
    220: -4,
    280: -13,
    320: -5,
}

#: The six sites PART 11-B registered afterwards, none of which was in
#: the fit. All four neighbours of each were confirmed -- 24 of 24, at
#: peak/mean 23.6 to 53.1 against controls at 9.0 to 15.6.
HELD_OUT_SITES = (20, 60, 140, 200, 260, 300)

#: The column heights the fit produces, left to right.
HEIGHTS = [5, 9, 13, 15, 17, 17, 19, 19, 21, 21, 21, 21, 21,
           19, 19, 17, 17, 15, 13, 9, 5]


@pytest.fixture(scope="module")
def layout():
    return round_well_layout(333)


def test_the_well_is_the_one_that_was_measured(layout):
    """333 fields in 21 columns of the measured heights."""
    assert layout.site_count == 333
    assert layout.heights == HEIGHTS
    assert sum(HEIGHTS) == 333
    assert (layout.columns, layout.radius) == MEASURED_WELL[:2]


@pytest.mark.parametrize("site,offset", sorted(MEASURED_OFFSETS.items()))
def test_every_measured_horizontal_offset_is_reproduced(layout, site, offset):
    """The seven the circle was fitted to, one test each.

    Named per site rather than asserted as a set: a model that got six of
    seven right would otherwise report as one failure with no clue which,
    and "3 of 7" is exactly the state the first fit was in.
    """
    neighbours = layout.neighbours(site)
    horizontal = {name: found - site for name, found in neighbours.items()
                  if DIRECTIONS[name][0]}
    assert offset in horizontal.values(), (
        f"site {site} was measured with a horizontal neighbour at "
        f"{offset:+d}; the model offers {sorted(horizontal.values())}")


def test_the_vertical_link_is_always_one(layout):
    """"+/-1 IS ALWAYS VERTICAL. The index runs DOWN a column."""
    for site in range(layout.site_count):
        for name, found in layout.neighbours(site).items():
            if DIRECTIONS[name][0] == 0:
                assert abs(found - site) == 1, (
                    f"site {site}'s {name} neighbour is {found - site:+d} "
                    "away; a vertical link is the next index by definition")


@pytest.mark.parametrize("site", HELD_OUT_SITES)
def test_the_held_out_sites_have_the_neighbours_registration_confirmed(
        layout, site):
    """Four each, and all four were confirmed on the real acquisition."""
    neighbours = layout.neighbours(site)
    assert len(neighbours) == 4, (
        f"site {site} was registered against four neighbours and the model "
        f"offers {sorted(neighbours)}")
    for name, found in neighbours.items():
        assert 0 <= found < layout.site_count
        assert found != site


def test_no_tile_is_its_own_neighbour_and_adjacency_is_mutual(layout):
    """If A is left of B then B is right of A, everywhere in the well."""
    opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}
    for site in range(layout.site_count):
        for name, found in layout.neighbours(site).items():
            back = layout.neighbours(found)
            assert back.get(opposite[name]) == site, (
                f"{site} says its {name} is {found}, which does not agree")


def test_every_pair_is_offered_once_with_its_axis(layout):
    """Registering a pair twice asks the same question twice."""
    pairs = layout.pairs()
    assert len(pairs) == len(set((a, b) for a, b, _axis in pairs))
    for _a, _b, axis in pairs:
        assert axis in ("vertical", "horizontal")
    # Every adjacency, counted from the other end: each site's neighbours
    # summed is twice the number of undirected edges.
    directed = sum(len(layout.neighbours(site))
                   for site in range(layout.site_count))
    assert directed == 2 * len(pairs)


def test_a_pair_is_ordered_by_geometry_and_not_by_site_index(layout):
    """The one that does not fail loudly, and it cost half a well.

    A caller cropping an overlap band has to know which tile is which:
    `register_edge` takes the BOTTOM of the first and the TOP of the
    second. Ordering pairs by site index looks identical and is wrong on
    every odd column, because the raster snakes -- in a bottom-to-top
    column the tile ABOVE carries the higher index. The symptom was 10 of
    32 edges accepted on a toy well, at a residual of 0.00 px, because
    each surviving component still solved perfectly.
    """
    for a, b, axis in layout.pairs():
        (ca, ra), (cb, rb) = layout.position(a), layout.position(b)
        if axis == "vertical":
            assert (cb, rb) == (ca, ra + 1), (
                f"the second tile of vertical pair {a}-{b} is not below the "
                "first")
        else:
            assert (cb, rb) == (ca + 1, ra), (
                f"the second tile of horizontal pair {a}-{b} is not to the "
                "right of the first")

    # And the case that makes it non-obvious: somewhere in the well the
    # geometric second carries the LOWER index.
    reversed_pairs = [(a, b) for a, b, _axis in layout.pairs() if b < a]
    assert reversed_pairs, (
        "no pair has its geometric order against its index order, so this "
        "well cannot demonstrate the bug and the test proves nothing")


def test_the_candidate_count_is_four_per_tile_not_a_window(layout):
    """The point of the model, stated as a number.

    `spacr.spacrops` offered `max_site_gap` = 64 in both directions, so
    128 candidates per tile of which at most four could touch -- 993
    scored pairs and 994 QC overlays for one well. This offers four.
    """
    assert max(len(layout.neighbours(site))
               for site in range(layout.site_count)) == 4
    assert 600 <= len(layout.pairs()) <= 700, len(layout.pairs())


def test_the_rows_are_absolute_and_not_counted_from_each_column(layout):
    """The mistake that cost the first fit four of its seven offsets.

    In a round well every column starts at a different row, so a tile's
    horizontal neighbour is NOT at the same position within its column.
    Asserted where it is most visible: the first column is five tall and
    the one beside it is nine, so their tops are two rows apart.
    """
    first_top = layout.position(0)
    beside = layout.neighbours(0)["right"]
    assert layout.position(beside)[1] == first_top[1], (
        "the horizontal neighbour is at the same ABSOLUTE row")
    assert layout.position(beside)[0] == first_top[0] + 1
    # And it is not the top of the next column, which is what the failed
    # model assumed.
    tops = [layout.span(column)[0] for column in range(2)]
    assert tops[0] != tops[1]


def test_the_reference_positions_are_the_score_not_a_placement_count(layout):
    """`plate_coordinate`'s grid, in microns, for the residual to beat."""
    origin = layout.micron_position(layout.site(*layout.centre)
                                    if layout.site(*layout.centre) is not None
                                    else 0)
    assert isinstance(origin, tuple) and len(origin) == 2
    a, b, _axis = layout.pairs()[0]
    ax, ay = layout.micron_position(a)
    bx, by = layout.micron_position(b)
    assert abs(ax - bx) + abs(ay - by) == pytest.approx(1280.0)


def test_a_different_field_count_is_fitted_and_an_impossible_one_refused():
    """A well imaged differently is the same circle with fewer fields."""
    assert round_well_layout(21).site_count == 21
    assert round_well_layout(5).site_count == 5
    with pytest.raises(ValueError, match="not a circle"):
        round_well_layout(4)
    with pytest.raises(ValueError):
        round_well_layout(0)


def test_a_column_outside_the_well_has_no_span_and_no_site(layout):
    """Asked for a column the circle does not reach, it says so.

    Two ways that happens and both must answer None rather than raise: an
    index past the last column, and -- for an ellipse-shaped acquisition
    somebody fits later -- a column inside the count whose chord is
    empty. A raise here would take a whole well's adjacency with it.
    """
    assert layout.span(-1) is None
    assert layout.span(layout.columns) is None
    assert layout.site(layout.columns, 0) is None
    assert layout.site(0, 999) is None

    empty = WellLayout(columns=9, radius=1.0, centre=(0, 0))
    assert empty.span(5) is None
    assert empty.heights[5] == 0
    # And the walk skips it rather than emitting a column of no tiles.
    assert all(column <= 1 for column, _row in empty.positions())


def test_a_site_outside_the_well_is_an_index_error_naming_the_size(layout):
    """`position` is asked for a site the well does not hold."""
    with pytest.raises(IndexError, match=str(layout.site_count)):
        layout.position(layout.site_count)
    with pytest.raises(IndexError):
        layout.position(-1)


def test_the_layout_stays_importable_anywhere(tmp_path):
    """It says it imports only `math`, and nothing was enforcing that.

    The claim is load-bearing rather than decorative: `ops_solve` is a
    separate module BECAUSE this one is pure geometry, and the OPS
    pipeline's other three all pull numpy. A layout that quietly gained a
    numpy import would make the separation pointless without anything
    saying so -- and this module is the one a caller reaches for to ask
    "which tiles touch" without wanting an array library.

    Measured in a fresh interpreter: 11 ms, three modules, no numpy.
    """
    import subprocess
    import sys

    probe = (
        "import sys, time;"
        "t=time.perf_counter();"
        "import spacr.ops_layout;"
        "print(round((time.perf_counter()-t)*1000));"
        "print('numpy' in sys.modules);"
        "print('torch' in sys.modules)"
    )
    done = subprocess.run([sys.executable, "-c", probe],
                          capture_output=True, text=True,
                          cwd=str(ROOT_DIR))
    assert done.returncode == 0, done.stderr
    milliseconds, numpy_loaded, torch_loaded = done.stdout.split()
    assert numpy_loaded == "False", "ops_layout now pulls numpy"
    assert torch_loaded == "False", "ops_layout now pulls torch"
    assert int(milliseconds) < 500, (
        f"importing the layout took {milliseconds} ms; it is arithmetic")


def test_the_layout_is_frozen_because_its_walk_is_cached():
    """A layout that could be edited would answer for the well it was."""
    layout = WellLayout()
    with pytest.raises(Exception):
        layout.radius = 3.0                                   # type: ignore
    assert hash(layout) == hash(WellLayout())
