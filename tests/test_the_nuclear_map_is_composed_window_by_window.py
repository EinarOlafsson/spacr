"""372 Phase B1: the Hoechst composite, built a window at a time.

WHY WINDOWS AND NOT A CANVAS. Well A1 is 26,855 x 26,865 -- 1.4 GB as uint16
for one channel -- and `ops_stitch` deliberately solves the placements without
holding the pixels. B2 then asks segmentation to run "window by window"
anyway, so the memory problem and the segmentation plan have the same answer:
compose the window the segmenter is about to read, and never materialise the
whole thing.

WHY AVERAGE AT ALL, with the number rather than the adjective. PART 15
measured the gain at about 1.15x on well A1 -- the raster's own 14% overlap,
where two tiles saw the same nuclei twice. An earlier draft claimed 3.3x from
averaging eleven cycles; there is only one DAPI acquisition, so that argument
does not apply. These tests pin the smaller, real number.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.ops_compose import (ComposeError, Window, compose_window,
                               overlap_gain, windows_over)

TILE, OVERLAP = 100, 14
STRIDE = TILE - OVERLAP


@pytest.fixture
def raster():
    """A 3x3 raster over one truth image, each tile with its own noise."""
    rng = np.random.default_rng(0)
    size = STRIDE * 2 + TILE
    truth = rng.random((size, size)).astype(np.float32)
    places, tiles = {}, {}
    site = 0
    for row in range(3):
        for col in range(3):
            top, left = row * STRIDE, col * STRIDE
            places[site] = (top, left)
            tiles[site] = (truth[top:top + TILE, left:left + TILE]
                           + rng.normal(0, 0.05, (TILE, TILE)).astype(np.float32))
            site += 1
    return truth, places, tiles, size


def test_averaging_beats_a_single_tile(raster):
    """THE WHOLE POINT. If the composite is not closer to the truth than one
    tile, the averaging is costing reads and buying nothing."""
    truth, places, tiles, size = raster
    image, _ = compose_window(Window(0, 0, size, size), places,
                              tiles.__getitem__, tile_shape=(TILE, TILE))
    top, left = places[4]
    window_truth = truth[top:top + TILE, left:left + TILE]
    single = float(np.abs(tiles[4] - window_truth).mean())
    composed = float(np.abs(image[top:top + TILE, left:left + TILE]
                            - window_truth).mean())
    assert composed < single


def test_the_gain_is_about_what_part_15_measured(raster):
    """1.15x on the real well. A composer reporting much more on the same
    geometry is averaging something twice."""
    _, places, tiles, size = raster
    _, coverage = compose_window(Window(0, 0, size, size), places,
                                 tiles.__getitem__, tile_shape=(TILE, TILE))
    assert 1.0 < overlap_gain(coverage) < 1.3


def test_coverage_tells_an_unreached_pixel_from_a_black_one(raster):
    """A pixel no tile reached is not a dark pixel, and only the count can
    say which it is -- which is why coverage is returned, not discarded."""
    _, places, tiles, _ = raster
    image, coverage = compose_window(Window(1000, 1000, 50, 50), places,
                                     tiles.__getitem__, tile_shape=(TILE, TILE))
    assert coverage.max() == 0
    assert float(image.max()) == 0.0


def test_only_the_tiles_that_touch_the_window_are_read(raster):
    """A well of 333 tiles must cost a handful of reads per window, not 333."""
    _, places, tiles, _ = raster
    seen = []

    def read(site):
        seen.append(site)
        return tiles[site]

    compose_window(Window(0, 0, 40, 40), places, read, tile_shape=(TILE, TILE))
    assert len(seen) < len(places)


def test_a_window_carries_its_offset_into_the_well_frame():
    """B4 numbers objects in the WELL frame, so a window has to know where it
    sits or every later phase renumbers per window."""
    window = Window(top=8000, left=8000, height=3000, width=3000)
    assert window.offset() == (8000, 8000)
    assert (window.bottom, window.right) == (11000, 11000)


def test_windows_cover_the_canvas_and_overlap(raster):
    """B3 sews labels across seams, which needs both windows to have seen the
    whole object -- so the overlap has to exceed the largest object, not
    merely touch."""
    _, _, _, size = raster
    windows = list(windows_over((size, size), size=120, overlap=20))
    assert windows[0].top == 0 and windows[0].left == 0
    assert max(w.bottom for w in windows) == size
    assert max(w.right for w in windows) == size
    tops = sorted({w.top for w in windows})
    assert tops[1] - tops[0] == 100          # stride = size - overlap


def test_an_overlap_as_big_as_the_window_is_refused():
    """The stride would be zero and the iteration would never advance."""
    with pytest.raises(ComposeError, match="never advances"):
        list(windows_over((500, 500), size=100, overlap=100))


def test_a_three_dimensional_tile_is_refused_with_the_reason():
    """Composing a five-channel stack silently would average channels
    together; the geometry is solved on one plane and so is this."""
    places = {0: (0, 0)}
    stack = np.zeros((5, 10, 10), dtype=np.float32)
    with pytest.raises(ComposeError, match="one channel at a time"):
        compose_window(Window(0, 0, 10, 10), places, lambda s: stack,
                       tile_shape=(10, 10))


def test_a_window_with_no_size_is_refused_rather_than_returning_nothing():
    """A zero-height window would compose an empty array and report success.

    B2 asks for the window the segmenter is about to read; handing it a
    (0, N) array is a segmentation of nothing that looks like a segmentation
    of nothing found.
    """
    places = {0: (0, 0)}
    tile = np.zeros((10, 10), dtype=np.float32)
    for window in (Window(0, 0, 0, 10), Window(0, 0, 10, 0)):
        with pytest.raises(ComposeError, match="positive height and width"):
            compose_window(window, places, lambda s: tile, tile_shape=(10, 10))


def test_a_window_smaller_than_one_pixel_is_refused_at_the_tiler():
    """The same refusal from the other end: ``windows_over`` never yields it."""
    with pytest.raises(ComposeError, match="window size must be positive"):
        list(windows_over((500, 500), size=0, overlap=0))
    with pytest.raises(ComposeError, match="window size must be positive"):
        list(windows_over((500, 500), size=-4096, overlap=0))


def test_the_tile_shape_is_read_from_the_first_tile_when_it_is_not_given(
    raster,
):
    """Callers that do not already know the tile size get it measured once.

    The result is the same image as when the shape is handed in, and the
    probe costs ONE extra read of one tile -- not a read of all 333 sites in
    the well, which is the exact cost `compose_window` exists to avoid. That
    bound is what is asserted, because "it also works without the shape" on
    its own would not notice a per-tile probe.
    """
    truth, places, tiles, _size = raster
    reads = []

    def read(site):
        reads.append(site)
        return tiles[site]

    window = Window(top=0, left=0, height=TILE, width=TILE)
    given, _ = compose_window(window, places, tiles.__getitem__,
                              tile_shape=(TILE, TILE))
    inferred, coverage = compose_window(window, places, read, tile_shape=None)

    assert np.allclose(inferred, given)
    assert coverage.max() > 0
    touching = len(set(reads))
    assert len(reads) == touching + 1, "the shape is probed exactly once"
    assert touching < len(places), "a window does not read the whole well"


def test_the_gain_of_a_window_no_tile_reached_is_zero_not_a_nan(raster):
    """`overlap_gain` takes the mean of the covered pixels, and there are none.

    An empty mean is NaN, and a NaN gain printed beside 1.15 reads as a
    measurement rather than as "this window is outside the raster".
    """
    _truth, places, tiles, size = raster
    far_away = Window(top=size * 4, left=size * 4, height=TILE, width=TILE)

    _image, coverage = compose_window(far_away, places, tiles.__getitem__,
                                      tile_shape=(TILE, TILE))

    assert coverage.max() == 0
    assert overlap_gain(coverage) == 0.0
