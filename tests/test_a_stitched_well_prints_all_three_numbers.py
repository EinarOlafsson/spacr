"""The A2 driver, and the reason it reports a canvas as well as a residual.

Instruction 372's PART 11-C: the first end-to-end run of this pipeline
reported 333 of 333 tiles placed at a median residual of 0.2 px, into a
canvas of 35,374 x 35,367 -- and it was WRONG by one pitch per column in
both axes. The driver negated the correlation's shift, the unwrap chose
the representative one period out, and it did so IDENTICALLY IN EVERY
EDGE of a direction. Every edge still agreed with every other edge, so the
residual was perfect and the count was perfect.

A UNIFORM ERROR IS INVISIBLE TO A RESIDUAL. That is the property this
file exists to hold: `test_a_uniform_error_is_invisible_to_the_residual`
reproduces exactly that failure and asserts that the count and the
residual both still look perfect while the canvas does not, because a
test that only checks the good case would have passed on the wrong run
too.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from spacr.ops_layout import round_well_layout
from spacr.ops_stitch import StitchedWell, stitch_well

TILE = 384
OVERLAP = int(TILE * 0.14)
STEP = TILE - OVERLAP


@pytest.fixture(scope="module")
def toy_well():
    """A 21-field round well of blobs, laid out on a known raster.

    Small enough to stitch in a few seconds and shaped like the real one:
    five columns of 3, 5, 5, 5, 3, snaked, at the acquisition's own 14 %
    overlap. The layout is the same code the 333-field well uses.
    """
    from scipy.ndimage import gaussian_filter

    layout = round_well_layout(21)
    rng = np.random.default_rng(2)
    rows, columns = max(layout.heights), layout.columns
    height = STEP * rows + TILE + 40
    width = STEP * columns + TILE + 40
    dots = np.zeros((height, width))
    count = (height * width) // 300
    dots[rng.integers(0, height, count), rng.integers(0, width, count)] = 1.0
    field = gaussian_filter(dots, 3.0) * (1200 * 2 * math.pi * 9.0) + 40.0
    first_row = min(row for _column, row in layout.positions())

    def read(site):
        column, row = layout.position(site)
        top = (row - first_row) * STEP + 20
        left = column * STEP + 20
        return rng.poisson(field[top:top + TILE,
                                 left:left + TILE]).astype(float)

    return layout, read


@pytest.fixture(scope="module")
def stitched(toy_well):
    layout, read = toy_well
    return stitch_well(read, layout, overlap=OVERLAP, tolerance=4, gpu=False)


def test_every_tile_is_placed_and_the_geometry_closes(stitched):
    """The count, the residual and the canvas, all three."""
    assert stitched.placed == stitched.layout.site_count
    assert stitched.residuals.size, "no accepted edge placed both its tiles"
    assert float(np.median(stitched.residuals)) < 1.0, "the solve did not close"
    assert stitched.canvas_agrees(0.02), (
        f"{stitched.canvas} against the layout's {stitched.expected_canvas}")


def test_the_summary_says_all_three_on_one_line(stitched):
    """Two of the three lie on their own, so they are printed together."""
    line = stitched.summary()
    assert "placed" in line and "edges" in line
    assert "residual" in line
    assert "canvas" in line and "layout says" in line


def test_a_uniform_error_is_invisible_to_the_residual(stitched):
    """The failure PART 11-C found, reproduced and asserted.

    Every edge of a direction shifted by one pitch: the count is
    unchanged, the residual is unchanged, and only the canvas notices.
    """
    pitch = TILE - OVERLAP
    broken_edges = {}
    for (a, b), one in stitched.edges.items():
        dy, dx = one.shift
        axis_is_vertical = abs(dy) > abs(dx)
        moved = ((dy + pitch, dx) if axis_is_vertical else (dy, dx + pitch))

        class _Moved:
            accepted = getattr(one, "accepted", True)
            shift = moved

        broken_edges[(a, b)] = _Moved()

    from spacr.ops_solve import solve_placements

    sites = list(range(stitched.layout.site_count))
    placements = solve_placements(
        {pair: one.shift for pair, one in broken_edges.items()
         if one.accepted}, sites)
    broken = StitchedWell(placements=placements, edges=broken_edges,
                          tile_shape=stitched.tile_shape,
                          layout=stitched.layout, overlap=OVERLAP)

    assert broken.placed == stitched.placed, "the count did not notice"
    assert float(np.median(broken.residuals)) == pytest.approx(
        float(np.median(stitched.residuals)), abs=0.5), (
        "the residual noticed, so this is not a uniform error and the test "
        "is not reproducing the failure it was written for")
    assert not broken.canvas_agrees(0.02), (
        "the canvas did not notice a whole pitch per column, which is the "
        "one check that caught this on the real well")


def test_the_expected_canvas_is_arithmetic_and_not_a_measurement(stitched):
    """21 columns is 20 pitches plus one tile -- the check's whole value."""
    layout = stitched.layout
    height, width = stitched.tile_shape
    columns = sum(1 for one in layout.heights if one)
    rows = max(layout.heights)
    assert stitched.expected_canvas == (
        (rows - 1) * (height - OVERLAP) + height,
        (columns - 1) * (width - OVERLAP) + width)


def test_an_unreadable_tile_costs_its_pairs_and_not_the_well(toy_well):
    """One bad file is not a failed acquisition."""
    layout, read = toy_well

    def sometimes(site):
        if site == 7:
            raise OSError("that tile is not readable")
        return read(site)

    well = stitch_well(sometimes, layout, overlap=OVERLAP, tolerance=4,
                       gpu=False)
    assert well.placed >= layout.site_count - 1
    assert well.proposed == len(layout.pairs())


def test_a_three_dimensional_tile_is_refused_with_the_reason(toy_well):
    """The geometry is solved on ONE channel, and PART 10 says which."""
    layout, read = toy_well

    def stacked(site):
        return np.stack([read(site)] * 5)

    with pytest.raises(ValueError, match="DAPI plane"):
        stitch_well(stacked, layout, overlap=OVERLAP, gpu=False)
    with pytest.raises(ValueError, match="no sites"):
        stitch_well(read, layout, sites=[], gpu=False)


def test_a_refused_edge_is_kept_rather_than_dropped(stitched):
    """A pair that failed is a fact about the acquisition."""
    assert stitched.proposed == len(stitched.layout.pairs())
    assert stitched.accepted <= stitched.proposed
    assert all(hasattr(one, "peak_ratio") for one in stitched.edges.values())
