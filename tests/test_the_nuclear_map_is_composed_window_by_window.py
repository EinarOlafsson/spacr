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
