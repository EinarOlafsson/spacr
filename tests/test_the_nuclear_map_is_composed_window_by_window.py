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
does not apply. These tests pin the smaller, real number, with the plain
average asked for by name (``blend='average'``).

WHY FEATHER BY DEFAULT (item 931). A plain average keeps a brightness step
where one tile ends, and on Funk et al. 2022 its seams stepped more often
than an unstitched tile does; weighting each tile by its distance to its
own edge removed that. The feather tests below pin the weight definition of
the reference script, the seam it removes and that coverage is unchanged.
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
                              tiles.__getitem__, tile_shape=(TILE, TILE),
                              blend="average")
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
                                 tiles.__getitem__, tile_shape=(TILE, TILE),
                                 blend="average")
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


def _script_feather(window, places, tiles, shape):
    """The reference script's none:feather composition (s1it2_seam_arms.py).

    Weight = min(row, column, height - 1 - row, width - 1 - column) + 1 in
    the tile's own 0-based pixels, float64 sums, ``sum(w x tile) / sum(w)``.
    """
    height, width = shape
    total = np.zeros((window.height, window.width))
    weights = np.zeros_like(total)
    yy, xx = np.mgrid[window.top:window.bottom, window.left:window.right]
    for site in sorted(places):
        ty, tx = (int(round(float(v))) for v in places[site])
        if (ty + height <= window.top or ty >= window.bottom
                or tx + width <= window.left or tx >= window.right):
            continue
        wy0, wx0 = max(ty, window.top), max(tx, window.left)
        wy1 = min(ty + height, window.bottom)
        wx1 = min(tx + width, window.right)
        sl = (slice(wy0 - window.top, wy1 - window.top),
              slice(wx0 - window.left, wx1 - window.left))
        patch = tiles[site][wy0 - ty:wy1 - ty, wx0 - tx:wx1 - tx]
        ly, lx = yy[sl] - ty, xx[sl] - tx
        w = np.minimum.reduce([ly, lx, height - 1 - ly,
                               width - 1 - lx]).astype(float) + 1
        total[sl] += patch * w
        weights[sl] += w
    return np.where(weights > 0, total / np.maximum(weights, 1e-9), 0)


def test_feather_is_the_default_and_matches_the_reference_script(raster):
    """Item 931: the default composite IS the script's none:feather, to the
    float32 the composer returns -- on whole windows and on windows that
    start inside a tile, where the weight must still count from the tile's
    own edge, not the window's."""
    _truth, places, tiles, size = raster
    for window in (Window(0, 0, size, size), Window(37, 51, 90, 70),
                   Window(80, 3, 20, 150)):
        default, _ = compose_window(window, places, tiles.__getitem__,
                                    tile_shape=(TILE, TILE))
        feather, _ = compose_window(window, places, tiles.__getitem__,
                                    tile_shape=(TILE, TILE), blend="feather")
        reference = _script_feather(window, places, tiles, (TILE, TILE))
        assert default.dtype == np.float32
        assert default.tobytes() == feather.tobytes()
        assert np.array_equal(default, reference.astype(np.float32))


def test_feather_leaves_no_step_where_a_brighter_tile_ends():
    """THE POINT OF FEATHERING. Two flat tiles of different brightness: the
    average jumps from the two-tile mean to one tile at the overlap's edge,
    the feather moves there by less than one weight step."""
    tile_a = np.full((20, 40), 100.0, dtype=np.float32)
    tile_b = np.full((20, 40), 200.0, dtype=np.float32)
    places = {0: (0, 0), 1: (0, 30)}          # columns 30-39 overlap
    tiles = {0: tile_a, 1: tile_b}
    window = Window(0, 0, 20, 70)
    average, _ = compose_window(window, places, tiles.__getitem__,
                                tile_shape=(20, 40), blend="average")
    feather, _ = compose_window(window, places, tiles.__getitem__,
                                tile_shape=(20, 40), blend="feather")
    row = 10
    # tile a's last column is 39; from column 40 tile b is alone
    assert average[row, 40] - average[row, 39] == pytest.approx(50.0)
    assert feather[row, 40] - feather[row, 39] < 10.0
    assert np.all(np.diff(feather[row, 25:45]) >= 0)


def test_one_tile_alone_comes_back_unchanged_under_feather(raster):
    """Where one tile covers a pixel the weights cancel: the feathered value
    is that tile's own value, exactly, not a value scaled by its weight."""
    _truth, places, tiles, _size = raster
    top, left = places[4]
    window = Window(top + OVERLAP, left + OVERLAP, TILE - 2 * OVERLAP,
                    TILE - 2 * OVERLAP)
    image, coverage = compose_window(window, places, tiles.__getitem__,
                                     tile_shape=(TILE, TILE))
    assert coverage.max() == 1
    assert np.array_equal(
        image, tiles[4][OVERLAP:TILE - OVERLAP, OVERLAP:TILE - OVERLAP])


def test_coverage_is_the_same_count_for_either_blend(raster):
    """Coverage says how many tiles reached a pixel; the blend weighs them
    but must not change the count `overlap_gain` and the segmenter read."""
    _truth, places, tiles, size = raster
    window = Window(0, 0, size, size)
    _a, average = compose_window(window, places, tiles.__getitem__,
                                 tile_shape=(TILE, TILE), blend="average")
    _f, feather = compose_window(window, places, tiles.__getitem__,
                                 tile_shape=(TILE, TILE), blend="feather")
    assert average.dtype == feather.dtype == np.uint16
    assert np.array_equal(average, feather)


def test_an_unknown_blend_is_refused_with_the_choices(raster):
    """A typo must not fall back to either blend silently."""
    _truth, places, tiles, size = raster
    with pytest.raises(ComposeError, match="feather"):
        compose_window(Window(0, 0, size, size), places, tiles.__getitem__,
                       tile_shape=(TILE, TILE), blend="median")


def test_the_engine_reads_the_blend_from_settings():
    """``ops_compose_blend`` reaches the composer: empty is feather, the two
    names pass, case and spaces aside, and anything else is refused."""
    from spacr.ops_engine import _compose_blend

    assert _compose_blend({}) == "feather"
    assert _compose_blend({"ops_compose_blend": None}) == "feather"
    assert _compose_blend({"ops_compose_blend": " Average "}) == "average"
    assert _compose_blend({"ops_compose_blend": "feather"}) == "feather"
    with pytest.raises(ValueError, match="ops_compose_blend"):
        _compose_blend({"ops_compose_blend": "voronoi"})
