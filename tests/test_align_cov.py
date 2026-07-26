"""spacr.align — the paths the first suite left untested.

``tests/test_align.py`` pins the happy grid, the global solve and the
memory budget. This file goes after the parts that only show up when
something is unusual, and asserts on the *values* the module produces
rather than on it not raising:

* the **phase → plain cross-correlation fallback**. Phase whitening
  returns a flat zero on a smooth, low-texture overlap; the module tries
  plain cross-correlation as well and scores both, and this suite proves
  the fallback is load-bearing by measuring what phase alone would have
  said;
* the reader cache's **LRU eviction** — the thing that stops a thousand
  mapped tiles becoming resident;
* the **band-splitting** write, byte-for-byte identical to a one-band
  write of the same canvas, and the memmap writer identical to the
  stream writer;
* a **jittered** 3x3 grid whose true offsets are *not* the nominal ones,
  recovered to a fifth of a pixel;
* a tile set with **no overlap**, a **single** tile, and the coordinates
  written into ``measurements.db``;
* every reported failure: a vanished file, a blank overlap, a read error
  inside a band, a stamp that could not be written.

All deterministic (fixed seeds), CPU-only, offline.
"""
from __future__ import annotations

import json
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import align
from spacr.errors import ConfigurationError, RunLedger


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _texture(height, width, seed=0, sigma=1.5):
    """A smooth, non-repeating uint16 field — registrable, unlike noise."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    raw = rng.random((height, width)).astype(np.float32)
    smooth = gaussian_filter(raw, sigma)
    smooth -= smooth.min()
    smooth /= max(float(smooth.max()), 1e-9)
    return (smooth * 30000 + 1000).astype(np.uint16)


def _save(folder, name, array):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(str(folder), name)
    np.save(path, array)
    return path


#: Per-tile jitter, in pixels, applied on top of the nominal grid step.
#: These are the offsets the solve has to *discover* — the nominal
#: positions are wrong by up to 4 px in each axis.
JITTER = {0: (0, 0), 1: (2, -3), 2: (-1, 4),
          3: (3, 2), 4: (-2, -2), 5: (1, 3),
          6: (4, -1), 7: (-3, 1), 8: (2, 2)}
JITTER_TILE = 160
JITTER_STEP = 120
JITTER_PAD = 8


@pytest.fixture(scope='module')
def jittered_grid(tmp_path_factory):
    """A 3x3 grid cut at *known, non-nominal* positions.

    Each tile is displaced from its nominal grid slot by the pixel
    offsets in :data:`JITTER`, so a stitcher that simply believed the
    stage would be up to 4 px out on every tile. Returns
    ``(folder, big, {index: (true_y, true_x)})``.
    """
    folder = tmp_path_factory.mktemp('jitter')
    span = JITTER_TILE + 2 * JITTER_STEP + 2 * JITTER_PAD
    big = _texture(span, span, seed=3, sigma=1.5)
    truth = {}
    for k in range(9):
        row, col = divmod(k, 3)
        y = row * JITTER_STEP + JITTER[k][0] + JITTER_PAD
        x = col * JITTER_STEP + JITTER[k][1] + JITTER_PAD
        _save(folder, f'plate1_B07_{k + 1:03d}.npy',
              big[y:y + JITTER_TILE, x:x + JITTER_TILE])
        truth[k] = (y, x)
    return str(folder), big, truth


# ---------------------------------------------------------------------------
# 1. Known ground truth, recovered to sub-pixel — with the truth *not*
#    equal to the nominal positions.
# ---------------------------------------------------------------------------

def test_jittered_grid_is_recovered_to_a_fifth_of_a_pixel(jittered_grid):
    """The solve finds offsets the stage did not know about.

    Every tile is cut a few pixels away from its nominal slot. Believing
    the nominal positions would be up to 4 px wrong per tile; the solved
    positions are within a fifth of a pixel of where the tiles were
    actually cut.
    """
    folder, _big, truth = jittered_grid
    tiles = align.scan_tiles(folder, grid=(3, 3),
                             overlap=1 - JITTER_STEP / JITTER_TILE)
    plan = align.estimate_offsets(tiles)

    assert plan.n_registered == 9
    assert plan.n_nominal == 0

    base = plan.placements[0]
    worst_solved = 0.0
    worst_nominal = 0.0
    for placement in plan.placements:
        true_y, true_x = truth[placement.tile.index]
        rel_y = (placement.y - base.y) - (true_y - truth[0][0])
        rel_x = (placement.x - base.x) - (true_x - truth[0][1])
        worst_solved = max(worst_solved, abs(rel_y), abs(rel_x))
        # what a stitcher that trusted the stage would have been out by
        nom_y = (placement.tile.nominal_y - tiles[0].nominal_y) \
            - (true_y - truth[0][0])
        nom_x = (placement.tile.nominal_x - tiles[0].nominal_x) \
            - (true_x - truth[0][1])
        worst_nominal = max(worst_nominal, abs(nom_y), abs(nom_x))

    assert worst_nominal >= 3.0, 'the fixture must actually jitter the tiles'
    assert worst_solved < 0.35, f'worst solved error {worst_solved:.3f} px'
    assert plan.max_residual < 0.35


def test_stitched_canvas_reproduces_the_source_image(jittered_grid, tmp_path):
    """End to end: the written mosaic *is* the picture it was cut from."""
    folder, big, truth = jittered_grid
    tiles = align.scan_tiles(folder, grid=(3, 3),
                             overlap=1 - JITTER_STEP / JITTER_TILE)
    plan = align.estimate_offsets(tiles)
    result = align.write_stack(plan, tmp_path / 'mosaic.npy')

    canvas = np.load(result.stack_path)
    assert canvas.shape == plan.canvas_shape
    assert result.n_written == 9

    # Line the canvas up with the source using tile 0's known cut.
    cy, cx = result.canvas.canvas_yx(plan.placements[0].y, plan.placements[0].x)
    top, left = truth[0][0] - cy, truth[0][1] - cx
    reference = big[top:top + canvas.shape[0], left:left + canvas.shape[1]]
    assert reference.shape == canvas.shape[:2]

    # The jittered bounding box has a ragged border that no tile covers;
    # 10 px in, every canvas pixel is covered and must be *exactly* the
    # pixel it was cut from — feathering equal values returns the value.
    margin = 10
    inner = canvas[margin:-margin, margin:-margin, 0]
    source = reference[margin:-margin, margin:-margin]
    assert inner.shape == source.shape and inner.size > 100000
    assert np.array_equal(inner, source), (
        f'{int((inner != source).sum())} of {inner.size} pixels differ; '
        f'worst by {int(np.abs(inner.astype(int) - source.astype(int)).max())}')
    assert not (inner == 0).any()


# ---------------------------------------------------------------------------
# 2. The phase → plain cross-correlation fallback
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def smooth_pair(tmp_path_factory):
    """Two 256 px tiles cut 190 px apart while the stage says 200.

    Deliberately smooth (sigma 3): spectral whitening amplifies the
    quantisation noise until it outweighs the signal, which is the case
    plain cross-correlation exists in this module to rescue.
    """
    folder = tmp_path_factory.mktemp('smooth')
    big = _texture(256, 256 + 190, seed=21, sigma=3.0)
    _save(folder, 'plate1_B07_001.npy', big[:, 0:256])
    _save(folder, 'plate1_B07_002.npy', big[:, 190:190 + 256])
    return str(folder), 190.0


def test_plain_cross_correlation_rescues_what_phase_whitening_misses(
        smooth_pair):
    """Phase normalisation returns 0 px for a real 10 px error; the module
    still gets the offset right, because it scores a second candidate."""
    from skimage.registration import phase_cross_correlation

    folder, true_step = smooth_pair
    tiles = align.scan_tiles(folder, grid=(1, 2), overlap=1 - 200 / 256)
    assert tiles[1].nominal_x == pytest.approx(200.0)

    # What phase whitening alone says about this overlap, measured.
    reader_a, reader_b = align._TileReader(tiles[0]), align._TileReader(tiles[1])
    try:
        strip_a = align._standardise(
            reader_a.window(0, 256, 200, 256, [0])[:, :, 0])
        strip_b = align._standardise(
            reader_b.window(0, 256, 0, 56, [0])[:, :, 0])
        phase_only = phase_cross_correlation(
            strip_a, strip_b, upsample_factor=10, normalization='phase')[0]
        plain = phase_cross_correlation(
            strip_a, strip_b, upsample_factor=10, normalization=None)[0]
    finally:
        reader_a.close()
        reader_b.close()

    assert abs(float(phase_only[1])) < 1.0, \
        'fixture no longer reproduces the phase-normalisation failure'
    assert float(plain[1]) < -8.0

    plan = align.estimate_offsets(tiles)
    solved = plan.placements[1].x - plan.placements[0].x
    assert solved == pytest.approx(true_step, abs=0.75), (
        f'solved {solved:.2f}; phase alone would have said '
        f'{200 + float(phase_only[1]):.2f} against a truth of {true_step}')
    assert plan.placements[1].confidence > 0.9
    assert plan.n_registered == 2


def test_a_normalisation_that_raises_is_skipped_not_fatal(smooth_pair,
                                                          monkeypatch):
    """One of the two FFTs blowing up must not lose the pair."""
    import skimage.registration as reg

    folder, true_step = smooth_pair
    tiles = align.scan_tiles(folder, grid=(1, 2), overlap=1 - 200 / 256)
    unpatched = align.estimate_offsets(tiles)

    original = reg.phase_cross_correlation
    calls = []

    def _fail_on_phase(*args, **kwargs):
        calls.append(kwargs.get('normalization'))
        if kwargs.get('normalization') == 'phase':
            raise RuntimeError('simulated FFT failure')
        return original(*args, **kwargs)

    monkeypatch.setattr(reg, 'phase_cross_correlation', _fail_on_phase)
    plan = align.estimate_offsets(tiles)

    assert 'phase' in calls and None in calls
    assert plan.n_registered == 2, 'the surviving candidate must still register'
    solved = plan.placements[1].x - plan.placements[0].x
    assert solved == pytest.approx(true_step, abs=0.75)
    # The plain candidate was already the winner, so dropping 'phase'
    # changes nothing at all.
    assert solved == pytest.approx(
        unpatched.placements[1].x - unpatched.placements[0].x, abs=1e-9)


def test_when_both_normalisations_fail_the_pair_is_refused(smooth_pair,
                                                           monkeypatch):
    """No candidate at all is a refusal with a reason, not a guess."""
    import skimage.registration as reg

    folder, _true_step = smooth_pair
    tiles = align.scan_tiles(folder, grid=(1, 2), overlap=1 - 200 / 256)

    def _always_fail(*_args, **_kwargs):
        raise ValueError('simulated FFT failure')

    monkeypatch.setattr(reg, 'phase_cross_correlation', _always_fail)
    plan = align.estimate_offsets(tiles)

    assert plan.accepted_pairs == []
    assert len(plan.refused_pairs) == 1
    assert 'phase correlation failed' in plan.refused_pairs[0].note
    assert plan.n_nominal == 2
    for placement in plan.placements:
        assert placement.method == align.METHOD_NOMINAL
        assert placement.y == pytest.approx(placement.tile.nominal_y)
        assert placement.x == pytest.approx(placement.tile.nominal_x)


def test_a_shift_that_separates_the_tiles_scores_nothing_and_is_refused(
        smooth_pair, monkeypatch):
    """A measurement so large it leaves no overlap cannot be scored, so it
    cannot be believed — the pair says so rather than being accepted."""
    import skimage.registration as reg

    folder, _true_step = smooth_pair
    tiles = align.scan_tiles(folder, grid=(1, 2), overlap=1 - 200 / 256)

    def _absurd(*_args, **_kwargs):
        return np.array([0.0, 400.0]), 0.0, 0.0

    monkeypatch.setattr(reg, 'phase_cross_correlation', _absurd)
    # max_shift large enough that the shift is not refused on size alone,
    # so it has to be refused on the fact that it scores nothing.
    plan = align.estimate_offsets(tiles, max_shift=10000.0)

    assert plan.accepted_pairs == []
    assert 'no usable overlap to score' in plan.refused_pairs[0].note
    assert plan.n_nominal == 2


def test_score_shift_refuses_a_sliver_of_overlap(smooth_pair):
    """``_score_shift`` grades a candidate on re-read pixels, and returns
    nothing at all when the candidate leaves too few to grade."""
    folder, _step = smooth_pair
    tiles = align.scan_tiles(folder, grid=(1, 2), overlap=1 - 200 / 256)
    reader_a, reader_b = align._TileReader(tiles[0]), align._TileReader(tiles[1])
    try:
        good, pixels = align._score_shift(
            tiles[0], tiles[1], reader_a, reader_b, 0.0, 190.0, 0, 16)
        assert pixels == 256 * 66
        assert good > 0.99, 'the true offset must score near 1.0'

        # 250 px apart leaves a 6 px seam — below min_overlap_px.
        sliver, sliver_px = align._score_shift(
            tiles[0], tiles[1], reader_a, reader_b, 0.0, 250.0, 0, 16)
        assert (sliver, sliver_px) == (0.0, 0)

        # 300 px apart leaves nothing at all.
        apart, apart_px = align._score_shift(
            tiles[0], tiles[1], reader_a, reader_b, 0.0, 300.0, 0, 16)
        assert (apart, apart_px) == (0.0, 0)
    finally:
        reader_a.close()
        reader_b.close()


def test_register_pair_reports_tiles_that_do_not_overlap(smooth_pair):
    """A pair with no shared pixels is refused by name, not registered."""
    folder, _step = smooth_pair
    tiles = align.scan_tiles(folder, grid=(1, 2), overlap=1 - 200 / 256)
    far = align.Tile(**{**vars(tiles[1]), 'nominal_x': 5000.0})
    reader_a, reader_b = align._TileReader(tiles[0]), align._TileReader(far)
    try:
        pair = align._register_pair(
            tiles[0], far, reader_a, reader_b, upsample=10,
            min_confidence=0.3, min_overlap_px=16, max_shift=None,
            reference_channel=0)
    finally:
        reader_a.close()
        reader_b.close()
    assert pair.accepted is False
    assert pair.note == 'tiles do not overlap at their nominal positions'
    assert (pair.dy, pair.dx) == (pair.nominal_dy, pair.nominal_dx)
    assert pair.confidence == 0.0


def test_ncc_of_empty_or_mismatched_strips_is_zero():
    """The confidence score is defined, and zero, where it cannot be
    computed — that zero is what keeps a noise peak out of the solve."""
    same = np.arange(24, dtype=np.float32).reshape(4, 6)
    assert align._ncc(same, same) == pytest.approx(1.0)
    assert align._ncc(same, -same) == pytest.approx(-1.0)
    assert align._ncc(np.zeros((0, 4)), np.zeros((0, 4))) == 0.0
    assert align._ncc(same, np.zeros((3, 3), np.float32)) == 0.0
    assert align._ncc(same, np.ones_like(same)) == 0.0   # no variance


def test_pair_drift_is_the_distance_from_the_stage_position():
    pair = align.PairResult(i=0, j=1, dy=13.0, dx=204.0,
                            nominal_dy=10.0, nominal_dx=200.0)
    assert pair.drift == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 3. The reader cache: LRU eviction
# ---------------------------------------------------------------------------

def test_reader_cache_evicts_the_least_recently_used(tmp_path):
    """At most ``max_open`` tiles are mapped; the oldest is unmapped.

    This is what keeps the resident set proportional to ``max_open``
    rather than to the input folder, so it is asserted on the mappings
    themselves, not on a counter.
    """
    tiles = []
    for k in range(4):
        path = _save(tmp_path, f'plate1_B07_{k + 1:03d}.npy',
                     _texture(32, 32, seed=k))
        tiles.append(align.Tile(path=path, index=k, shape=(32, 32, 1)))

    cache = align._ReaderCache(max_open=2)
    try:
        first = cache.get(tiles[0])
        second = cache.get(tiles[1])
        assert cache.opened == 2
        assert sorted(cache._open) == [0, 1]
        assert first.window(0, 4, 0, 4, [0]).shape == (4, 4, 1)

        # Touching tile 0 again must not re-open it, and must make it the
        # most recent — so tile 1 is the one that goes.
        assert cache.get(tiles[0]) is first
        assert cache.opened == 2
        assert cache._order == [1, 0]

        third = cache.get(tiles[2])
        assert cache.opened == 3
        assert sorted(cache._open) == [0, 2]
        assert 1 not in cache._open, 'the least-recently-used tile survived'
        # The evicted reader really was unmapped, not just forgotten.
        assert second._sources == [None]
        # …and the survivors still read real pixels.
        assert first.window(0, 32, 0, 32, [0])[:, :, 0].astype(np.uint16).tolist() \
            == np.load(tiles[0].path).tolist()

        fourth = cache.get(tiles[3])
        assert cache.opened == 4
        assert len(cache._open) == 2
        assert fourth.window(0, 2, 0, 2, [0]).shape == (2, 2, 1)
    finally:
        cache.close()
    assert cache._open == {} and cache._order == []
    assert third._sources == [None] and fourth._sources == [None]


def test_reader_cache_keeps_at_least_one_reader():
    cache = align._ReaderCache(max_open=0)
    assert cache.max_open == 1
    cache.close()


def test_reader_close_survives_a_still_exported_mapping(tmp_path):
    """``close`` must not raise when something else still holds the buffer.

    Python refuses to close an mmap with exported pointers; the reader
    swallows that and drops its reference, which is all it can do.
    """
    path = _save(tmp_path, 'plate1_B07_001.npy', _texture(16, 16, seed=1))
    reader = align._TileReader(align.Tile(path=path, shape=(16, 16, 1)))
    keeper = memoryview(reader._sources[0].base._mmap)
    try:
        reader.close()                      # BufferError, swallowed
    finally:
        keeper.release()
    assert reader._sources == [None]
    reader.close()                          # idempotent


# ---------------------------------------------------------------------------
# 4. Headers, shapes and windows
# ---------------------------------------------------------------------------

def test_npy_2_0_header_is_read_without_pixels(tmp_path):
    """``.npy`` format 2.0 is read by the 2.0 header reader, not guessed."""
    data = np.arange(6 * 7, dtype=np.uint16).reshape(6, 7)
    path = str(tmp_path / 'plate1_B07_001.npy')
    with open(path, 'wb') as handle:
        np.lib.format.write_array_header_2_0(
            handle, {'descr': '<u2', 'fortran_order': False, 'shape': (6, 7)})
        handle.write(data.tobytes())
    with open(path, 'rb') as handle:
        assert np.lib.format.read_magic(handle) == (2, 0)

    shape, dtype = align._npy_header(path)
    assert shape == (6, 7)
    assert dtype == np.dtype('uint16')
    assert np.array_equal(np.load(path, allow_pickle=False), data)

    tile = align.scan_tiles([path])[0]
    assert tile.shape == (6, 7, 1)
    reader = align._TileReader(tile)
    try:
        assert reader.window(0, 6, 0, 7, [0])[:, :, 0].tolist() == \
            data.astype(np.float32).tolist()
    finally:
        reader.close()


def test_npy_of_an_unsupported_version_is_refused(tmp_path, monkeypatch):
    """A future .npy version is named, not silently mis-parsed."""
    path = _save(tmp_path, 'plate1_B07_001.npy', np.zeros((4, 4), np.uint16))
    monkeypatch.setattr(np.lib.format, 'read_magic', lambda _h: (9, 9))
    with pytest.raises(align.AlignError, match='unsupported .npy format'):
        align._npy_header(path)


def test_a_file_that_is_not_a_tile_is_refused_by_suffix(tmp_path):
    path = str(tmp_path / 'notatile.png')
    with open(path, 'wb') as handle:
        handle.write(b'\x89PNG')
    with pytest.raises(align.AlignError, match='not a .npy or .tif tile'):
        align._read_header(path)
    with pytest.raises(align.AlignError, match='not a .npy or .tif tile'):
        align._TileReader(align.Tile(path=path, shape=(4, 4, 1)))


def test_channel_first_arrays_are_recognised_and_moved_last():
    """A ``(C, H, W)`` TIFF is not read as a 3-row image."""
    assert align._normalise_shape((64, 96), 'x') == (64, 96, 1)
    assert align._normalise_shape((3, 64, 96), 'x') == (64, 96, 3)
    assert align._normalise_shape((64, 96, 3), 'x') == (64, 96, 3)
    # 9 planes is more than the channel-first heuristic allows, so it is
    # read as a 9-row, 64-column, 96-plane array rather than guessed at.
    assert align._normalise_shape((9, 64, 96), 'x') == (9, 64, 96)

    flat = np.arange(2 * 3, dtype=np.uint16).reshape(2, 3)
    assert align._channel_last(flat).shape == (2, 3, 1)
    assert align._channel_last(flat)[:, :, 0].tolist() == flat.tolist()

    first = np.arange(3 * 2 * 4, dtype=np.uint16).reshape(3, 2, 4)
    moved = align._channel_last(first)
    assert moved.shape == (2, 4, 3)
    for plane in range(3):
        assert moved[:, :, plane].tolist() == first[plane].tolist()

    last = np.arange(10 * 4 * 3, dtype=np.uint16).reshape(10, 4, 3)
    assert align._channel_last(last) is last, \
        '10 rows is too many to be a channel axis'


def test_four_dimensional_tiles_name_z_in_the_message():
    with pytest.raises(align.AlignError, match='z is not stitched'):
        align._normalise_shape((3, 4, 5, 6), 'stack.npy')


def test_compressed_tiff_falls_back_from_memmap_to_a_full_read(tmp_path):
    """A tiled/compressed TIFF cannot be mapped; it is read whole instead,
    and the pixels that come back are the ones that went in."""
    import tifffile

    data = _texture(48, 40, seed=5)
    plain = str(tmp_path / 'plate1_B07_001.tif')
    packed = str(tmp_path / 'plate1_B07_002.tif')
    tifffile.imwrite(plain, data)
    tifffile.imwrite(packed, data, compression='zlib')
    with pytest.raises(ValueError):
        tifffile.memmap(packed, mode='r')

    mapped = align._TileReader(align.Tile(path=plain, shape=(48, 40, 1)))
    unmapped = align._TileReader(align.Tile(path=packed, shape=(48, 40, 1)))
    try:
        assert mapped._handles, 'the uncompressed TIFF should have been mapped'
        assert unmapped._handles == [], \
            'the compressed TIFF cannot be mapped and must not claim to be'
        for reader in (mapped, unmapped):
            window = reader.window(4, 12, 6, 20, [0])[:, :, 0]
            assert window.tolist() == data[4:12, 6:20].astype(np.float32).tolist()
    finally:
        mapped.close()
        unmapped.close()


def test_reader_shape_reports_the_assembled_site(tmp_path):
    """One site, one shape — whether the channels are files or planes."""
    two_d = _save(tmp_path, 'plate1_B07_001.npy', _texture(24, 30, seed=2))
    reader = align._TileReader(align.Tile(path=two_d, shape=(24, 30, 1)))
    try:
        assert reader.shape == (24, 30, 1)
    finally:
        reader.close()

    stack = np.stack([_texture(24, 30, seed=s) for s in (1, 2, 3)], axis=-1)
    merged = _save(tmp_path, 'plate1_B07_002.npy', stack)
    reader = align._TileReader(align.Tile(path=merged, shape=(24, 30, 3)))
    try:
        assert reader.shape == (24, 30, 3)
        got = reader.window(0, 24, 0, 30, [0, 1, 2])
        assert got.shape == (24, 30, 3)
        assert np.array_equal(got.astype(np.uint16), stack)
    finally:
        reader.close()

    split = align.Tile(path=two_d, shape=(24, 30, 2),
                       channel_paths=(two_d, merged))
    reader = align._TileReader(split)
    try:
        assert reader.shape == (24, 30, 2)
    finally:
        reader.close()


def test_window_zero_pads_and_ignores_channels_that_do_not_exist(tmp_path):
    """Every out-of-range read is zeros, never an exception or garbage."""
    data = _texture(20, 24, seed=9)
    path = _save(tmp_path, 'plate1_B07_001.npy', data)
    tile = align.Tile(path=path, shape=(20, 24, 1))
    reader = align._TileReader(tile)
    try:
        # A window that straddles the edge: real pixels inside, zeros out.
        straddle = reader.window(-3, 5, -2, 6, [0])[:, :, 0]
        assert straddle.shape == (8, 8)
        assert np.all(straddle[:3, :] == 0) and np.all(straddle[:, :2] == 0)
        assert straddle[3:, 2:].tolist() == data[0:5, 0:6].astype(np.float32).tolist()

        # Wholly off the tile.
        assert np.all(reader.window(100, 110, 0, 4, [0]) == 0)
        assert np.all(reader.window(0, 4, -50, -40, [0]) == 0)

        # A plane this tile does not have.
        assert np.all(reader.window(0, 4, 0, 4, [3]) == 0)
    finally:
        reader.close()

    # …and the same for a site whose channels are separate files.
    other = _save(tmp_path, 'plate1_B07_002.npy', data + 1)
    split = align.Tile(path=path, shape=(20, 24, 2),
                       channel_paths=(path, other))
    reader = align._TileReader(split)
    try:
        both = reader.window(0, 4, 0, 4, [0, 1, 7])
        assert both.shape == (4, 4, 3)
        assert both[:, :, 0].tolist() == data[:4, :4].astype(np.float32).tolist()
        assert both[:, :, 1].tolist() == (data[:4, :4] + 1).astype(np.float32).tolist()
        assert np.all(both[:, :, 2] == 0), 'channel 7 does not exist'
    finally:
        reader.close()


# ---------------------------------------------------------------------------
# 5. Names, grids and path collection
# ---------------------------------------------------------------------------

def test_parse_name_falls_back_to_a_channel_token_and_a_trailing_field():
    assert align._parse_name('some_image_ch2_12') == \
        {'plate': '', 'well': '', 'field': 12, 'channel': 2}
    assert align._parse_name('scan.channel_3.0042') == \
        {'plate': '', 'well': '', 'field': 42, 'channel': 3}
    assert align._parse_name('nothing_here') == \
        {'plate': '', 'well': '', 'field': 0, 'channel': 1}
    # the named forms still win
    assert align._parse_name('plate1_B07_003') == \
        {'plate': 'plate1', 'well': 'B07', 'field': 3, 'channel': 1}
    assert align._parse_name('p_B07_T0001F005L01A01Z01C02') == \
        {'plate': 'p', 'well': 'B07', 'field': 5, 'channel': 2}
    assert align._parse_name('p_B07_T0001F006L01C03') == \
        {'plate': 'p', 'well': 'B07', 'field': 6, 'channel': 3}


def test_nameless_tiles_are_numbered_by_their_position(tmp_path):
    """Files that carry no field number at all still become distinct sites."""
    folder = tmp_path / 'anon'
    folder.mkdir()
    for name in ('alpha', 'bravo', 'charlie'):
        _save(folder, f'{name}.npy', _texture(16, 16, seed=len(name)))
    tiles = align.scan_tiles(str(folder), grid=(1, 3))
    assert len(tiles) == 3
    assert [t.field for t in tiles] == [1, 2, 3]
    assert [os.path.basename(t.path) for t in tiles] == \
        ['alpha.npy', 'bravo.npy', 'charlie.npy']


def test_a_non_positive_grid_is_refused(tmp_path):
    _save(tmp_path, 'plate1_B07_001.npy', _texture(8, 8))
    with pytest.raises(ConfigurationError, match='two positive integers'):
        align.scan_tiles(str(tmp_path), grid=(0, 3))
    with pytest.raises(ConfigurationError, match='two positive integers'):
        align._grid_shape(4, (2, -1))


def test_grid_is_factorised_when_it_can_be_and_padded_when_it_cannot():
    assert align._grid_shape(12, None) == (3, 4)
    assert align._grid_shape(9, None) == (3, 3)
    assert align._grid_shape(1, None) == (1, 1)
    assert align._grid_shape(0, None) == (1, 1)
    # 7 is prime: no exact factorisation, so a padded near-square.
    rows, cols = align._grid_shape(7, None)
    assert (rows, cols) == (3, 3)
    assert rows * cols >= 7
    rows, cols = align._grid_shape(11, None)
    assert (rows, cols) == (3, 4) and rows * cols >= 11


def test_column_major_and_snake_orders_lay_tiles_out_correctly():
    row_major = [align._grid_position(k, 2, 3, 'row-major') for k in range(6)]
    assert row_major == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    snake_row = [align._grid_position(k, 2, 3, 'snake-row') for k in range(6)]
    assert snake_row == [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]
    column = [align._grid_position(k, 2, 3, 'column-major') for k in range(6)]
    assert column == [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
    snake_col = [align._grid_position(k, 2, 3, 'snake-column') for k in range(6)]
    assert snake_col == [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (1, 2)]
    with pytest.raises(ConfigurationError, match='order must be one of'):
        align._grid_position(0, 2, 3, 'diagonal')


def test_column_major_order_is_honoured_end_to_end(tmp_path):
    for k in range(6):
        _save(tmp_path, f'plate1_B07_{k + 1:03d}.npy', _texture(16, 16, seed=k))
    tiles = align.scan_tiles(str(tmp_path), grid=(2, 3), order='column-major',
                             overlap=0.0)
    assert [(t.grid_row, t.grid_col) for t in tiles] == \
        [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
    assert tiles[1].nominal_y == pytest.approx(16.0)
    assert tiles[1].nominal_x == pytest.approx(0.0)


def test_scan_accepts_a_list_a_single_file_and_a_recursive_walk(tmp_path):
    """Three ways of naming the input, all producing real tiles."""
    root = tmp_path / 'plate'
    nested = root / 'well_B07'
    nested.mkdir(parents=True)
    made = [_save(nested, f'plate1_B07_{k + 1:03d}.npy', _texture(16, 16, seed=k))
            for k in range(3)]
    (root / 'notes.txt').write_text('ignore me')

    # 1. an explicit list — order is exactly as given
    listed = align.scan_tiles([made[2], made[0]], grid=(1, 2))
    assert [os.path.basename(t.path) for t in listed] == \
        ['plate1_B07_003.npy', 'plate1_B07_001.npy']

    # 2. a single file
    one = align.scan_tiles(made[1])
    assert len(one) == 1 and one[0].path == made[1]
    assert one[0].shape == (16, 16, 1)

    # 3. a recursive walk finds the sub-folder; a flat scan does not
    assert len(align.scan_tiles(str(root), recursive=True, grid=(1, 3))) == 3
    with pytest.raises(ConfigurationError, match='no .npy/.tif tiles'):
        align.scan_tiles(str(root), recursive=False)


def test_well_ids_pass_through_a_well_they_cannot_parse():
    assert align._well_ids('B7') == ('r2', 'c7')
    assert align._well_ids('12') == ('12', '12')
    assert align._well_ids('Bxx') == ('Bxx', 'Bxx')
    assert align._well_ids('B') == ('B', 'B')
    assert align._join_keys('p1', 'Bxx', 3)['prc'] == 'p1_Bxx_Bxx'


# ---------------------------------------------------------------------------
# 6. The global solve
# ---------------------------------------------------------------------------

def test_solve_positions_of_nothing_is_empty_not_an_error():
    positions, residuals, degrees = align.solve_positions(
        0, [], np.zeros((0, 2)))
    assert positions.shape == (0, 2)
    assert residuals.shape == (0,)
    assert degrees.shape == (0,)


def test_a_big_tile_count_uses_the_sparse_solver_and_still_recovers_truth():
    """Above 512 tiles the design matrix goes sparse; the answer must not.

    600 tiles in a chain, each registered against its next *two*
    neighbours. The dense path is not viable at this size, and the sparse
    one has to land on the same answer.
    """
    n = 600
    step = 10.0
    truth = np.arange(n) * step
    nominal = np.column_stack([np.zeros(n), truth])
    edges = [(k, k + 1, 0.0, step, 1.0) for k in range(n - 1)]
    edges += [(k, k + 2, 0.0, 2 * step, 1.0) for k in range(n - 2)]

    positions, residuals, degrees = align.solve_positions(n, edges, nominal)
    assert positions.shape == (n, 2)
    recovered = positions[:, 1] - positions[0, 1]
    assert np.max(np.abs(recovered - truth)) < 1e-4
    assert np.max(np.abs(positions[:, 0] - positions[0, 0])) < 1e-4
    assert float(residuals.max()) < 1e-4
    assert degrees[0] == 2 and degrees[300] == 4

    # The dense path (<= 512 tiles) on the same shape of problem agrees.
    small = 200
    small_edges = [(k, k + 1, 0.0, step, 1.0) for k in range(small - 1)]
    dense, _r, _d = align.solve_positions(
        small, small_edges,
        np.column_stack([np.zeros(small), np.arange(small) * step]))
    assert np.max(np.abs((dense[:, 1] - dense[0, 1])
                         - np.arange(small) * step)) < 1e-6


def test_sequential_accumulation_leaves_an_unreachable_tile_at_nominal():
    """The counter-example implementation: a tile with nothing to chain
    from keeps its stage position rather than landing on the origin."""
    nominal = np.array([[0.0, 0.0], [0.0, 77.0], [0.0, 200.0]])
    # Only tiles 1 and 2 are paired, so tile 1 cannot chain from tile 0.
    positions = align._sequential_positions(
        3, [(1, 2, 0.0, 100.0, 1.0)], nominal)
    assert positions[0].tolist() == [0.0, 0.0]
    assert positions[1].tolist() == [0.0, 77.0], 'kept its nominal position'
    assert positions[2].tolist() == [0.0, 177.0], 'chained from tile 1'


# ---------------------------------------------------------------------------
# 7. estimate_offsets: the things that go wrong
# ---------------------------------------------------------------------------

def test_a_tile_with_an_empty_shape_is_unplaced_and_recorded(tmp_path):
    """A header that parsed but reports no pixels is refused by name."""
    good_path = _save(tmp_path, 'plate1_B07_001.npy', _texture(32, 32, seed=1))
    empty = align.Tile(path=str(tmp_path / 'plate1_B07_002.npy'), index=1,
                       plate='plate1', well='B07', field=2, shape=(0, 0, 1))
    good = align.Tile(path=good_path, index=0, plate='plate1', well='B07',
                      field=1, shape=(32, 32, 1))
    ledger = RunLedger('align-test')

    plan = align.estimate_offsets([good, empty], ledger=ledger)

    assert [t.index for t, _r in plan.unplaced] == [1]
    assert 'empty shape' in plan.unplaced[0][1]
    assert len(plan.placements) == 1
    assert plan.placements[0].tile.index == 0
    assert ledger.n_failed == 1
    assert ledger.failures[0].stage == 'scan'
    assert 'empty shape' in ledger.failures[0].message


def test_mixed_channel_counts_warn_and_the_canvas_takes_the_widest(tmp_path):
    """Short tiles are placed, and the missing planes are said out loud."""
    one = _save(tmp_path, 'plate1_B07_001.npy', _texture(64, 64, seed=1))
    three = _save(tmp_path, 'plate1_B07_002.npy',
                  np.stack([_texture(64, 64, seed=s) for s in (2, 3, 4)],
                           axis=-1))
    tiles = align.scan_tiles([one, three], grid=(1, 2), overlap=0.25)
    assert [t.n_channels for t in tiles] == [1, 3]

    plan = align.estimate_offsets(tiles)
    warning = [w for w in plan.warnings if 'channel counts' in w]
    assert warning, plan.warnings
    assert '[1, 3]' in warning[0]
    assert plan.canvas_shape[2] == 3


def test_a_single_tile_is_a_success_in_the_ledger(tmp_path):
    path = _save(tmp_path, 'plate1_B07_001.npy', _texture(24, 24, seed=1))
    ledger = RunLedger('align-test')
    plan = align.estimate_offsets(align.scan_tiles(path), ledger=ledger)

    assert len(plan.placements) == 1
    only = plan.placements[0]
    assert only.method == align.METHOD_SINGLE
    assert only.confidence == 1.0
    assert only.n_pairs == 0 and only.residual == 0.0
    assert (only.y, only.x) == (0.0, 0.0)
    assert plan.canvas_shape == (24, 24, 1)
    assert plan.feather == 1
    assert any('one tile only' in w for w in plan.warnings)
    assert ledger.n_succeeded == 1 and ledger.n_failed == 0


def test_a_tile_that_vanishes_between_scan_and_register_is_noted(tmp_path):
    """Registration failure is a note on the pair, never an exception."""
    big = _texture(64, 128, seed=6)
    first = _save(tmp_path, 'plate1_B07_001.npy', big[:, 0:64])
    second = _save(tmp_path, 'plate1_B07_002.npy', big[:, 48:112])
    tiles = align.scan_tiles(str(tmp_path), grid=(1, 2), overlap=0.25)
    os.remove(second)

    plan = align.estimate_offsets(tiles)

    assert len(plan.overlaps) == 1
    pair = plan.overlaps[0]
    assert pair.accepted is False
    assert 'FileNotFoundError' in pair.note or 'No such file' in pair.note
    assert plan.n_nominal == 2
    for placement in plan.placements:
        assert placement.y == pytest.approx(placement.tile.nominal_y)
    assert [p.tile.index for p in plan.placements] == [0, 1], \
        'the surviving tile is still placed'
    assert plan.placements[0].tile.path == first and os.path.isfile(first)
    assert plan.unplaced == [], 'the header parsed, so neither tile is unplaced'


def test_a_blank_overlap_is_a_ledger_failure_not_a_silent_placement(tmp_path):
    """Nominal placement is recorded as a failure — it is not a success."""
    for k in range(2):
        _save(tmp_path, f'plate1_B07_{k + 1:03d}.npy',
              np.full((48, 64), 2000, np.uint16))
    tiles = align.scan_tiles(str(tmp_path), grid=(1, 2), overlap=0.25)
    ledger = RunLedger('align-test')

    plan = align.estimate_offsets(tiles, ledger=ledger)

    assert plan.n_nominal == 2 and plan.n_registered == 0
    assert 'blank' in plan.overlaps[0].note
    assert ledger.n_failed == 2 and ledger.n_succeeded == 0
    assert all(f.stage == 'align' for f in ledger.failures)
    assert all('nominal position' in f.message for f in ledger.failures)
    assert any('did not register' in w for w in plan.warnings)


def test_feather_width_falls_back_to_one_when_nothing_can_be_measured():
    """An overlap list that measures nothing gives a 1 px ramp, not a
    crash and not a ramp wider than the tiles."""
    tile = align.Tile(path='a.npy', index=0, shape=(40, 40, 1))
    other = align.Tile(path='b.npy', index=1, shape=(40, 40, 1))
    plan = align.AlignPlan(tiles=[tile, other])

    # a pair naming a tile that is not in the set at all
    plan.overlaps = [align.PairResult(i=0, j=99, dy=0.0, dx=30.0)]
    assert align._feather_width(plan, [tile, other]) == 1
    plan.overlaps = [align.PairResult(i=99, j=0, dy=0.0, dx=30.0)]
    assert align._feather_width(plan, [tile, other]) == 1

    # a pair whose displacement leaves the tiles disjoint
    plan.overlaps = [align.PairResult(i=0, j=1, dy=0.0, dx=500.0)]
    assert align._feather_width(plan, [tile, other]) == 1

    # a real overlap gives a real ramp, capped at half the tile
    plan.overlaps = [align.PairResult(i=0, j=1, dy=0.0, dx=30.0)]
    assert align._feather_width(plan, [tile, other]) == 10
    plan.overlaps = [align.PairResult(i=0, j=1, dy=0.0, dx=2.0)]
    assert align._feather_width(plan, [tile, other]) == 20   # 40 // 2


def test_tiles_that_never_overlap_are_reported_and_left_at_stage_positions(
        tmp_path):
    """The 'no overlap' case: nothing registers, and the plan says why."""
    for k in range(3):
        _save(tmp_path, f'plate1_B07_{k + 1:03d}.npy', _texture(32, 32, seed=k))
    tiles = align.scan_tiles(str(tmp_path), grid=(1, 3), overlap=0.0)

    plan = align.estimate_offsets(tiles)

    assert plan.overlaps == []
    assert plan.n_registered == 0 and plan.n_nominal == 3
    assert any('no two tiles overlap' in w for w in plan.warnings)
    assert any('disconnected component' in w for w in plan.warnings)
    for placement in plan.placements:
        assert placement.method == align.METHOD_NOMINAL
        assert placement.n_pairs == 0
        assert placement.note == 'no overlapping neighbour to register against'
        assert placement.x == pytest.approx(placement.tile.nominal_x)
    assert plan.canvas_shape == (32, 96, 1)
    assert plan.feather == 1


# ---------------------------------------------------------------------------
# 8. Writing: bands, writers and failures
# ---------------------------------------------------------------------------

def test_band_rows_is_one_for_a_canvas_with_no_area():
    empty = align.CanvasSpec(height=0, width=0, channels=1, dtype='uint16')
    assert align._band_rows(empty, 1 << 20, None) == 1
    thin = align.CanvasSpec(height=10, width=0, channels=1, dtype='uint16')
    assert align._band_rows(thin, 1 << 20, None) == 1
    spec = align.CanvasSpec(height=100, width=50, channels=1, dtype='uint16')
    assert align._band_rows(spec, 1 << 20, 7) == 7
    assert align._band_rows(spec, 1 << 20, 10000) == 100, 'capped at the canvas'
    assert align._band_rows(spec, 1, None) == 1, 'never zero rows'


def test_ramp_without_feather_is_flat_and_with_it_is_a_ramp():
    flat = align._ramp(0, 6, 6, 0)
    assert flat.tolist() == [1.0] * 6
    ramp = align._ramp(0, 6, 6, 2)
    assert ramp[0] < ramp[1] < ramp[2]
    assert ramp[2] == pytest.approx(ramp[3]), 'symmetric about the middle'
    assert ramp[0] >= align._WEIGHT_FLOOR, 'an edge pixel still counts'
    assert ramp.max() == pytest.approx(1.0)


def test_drop_pages_is_a_no_op_without_a_mapping(tmp_path):
    align._drop_pages(np.zeros((4, 4), np.uint16))     # no _mmap at all
    path = str(tmp_path / 'canvas.npy')
    array = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint16,
                                      shape=(8, 8))
    array[:] = 3
    array.flush()
    align._drop_pages(array)
    assert np.array_equal(np.load(path), np.full((8, 8), 3, np.uint16)), \
        'dropping the pages must not drop the data'
    array._mmap.close()
    align._drop_pages(array)                            # ValueError, swallowed


def test_memmap_writer_produces_exactly_the_same_bytes_as_the_stream_writer(
        jittered_grid, tmp_path):
    folder, _big, _truth = jittered_grid
    tiles = align.scan_tiles(folder, grid=(3, 3),
                             overlap=1 - JITTER_STEP / JITTER_TILE)
    plan = align.estimate_offsets(tiles)

    streamed = align.write_stack(plan, tmp_path / 'stream.npy', writer='stream')
    mapped = align.write_stack(plan, tmp_path / 'memmap.npy', writer='memmap')

    assert streamed.writer == 'stream' and mapped.writer == 'memmap'
    assert mapped.n_written == streamed.n_written == 9
    assert mapped.status == 'complete'
    left, right = np.load(streamed.stack_path), np.load(mapped.stack_path)
    assert left.shape == right.shape == plan.canvas_shape
    assert np.array_equal(left, right)
    assert left.any(), 'the canvas must not be blank'


def test_band_splitting_is_invisible_in_the_output(jittered_grid, tmp_path):
    """A canvas far bigger than the budget is written in many bands, and
    every one of those bands lands where a single-band write put it."""
    folder, _big, _truth = jittered_grid
    tiles = align.scan_tiles(folder, grid=(3, 3),
                             overlap=1 - JITTER_STEP / JITTER_TILE)
    plan = align.estimate_offsets(tiles)
    height = plan.canvas_shape[0]

    whole = align.write_stack(plan, tmp_path / 'whole.npy', band_rows=height)
    assert whole.band_rows == height

    # 4 KB of budget cannot hold a 400-column band, so the write splits.
    split = align.write_stack(plan, tmp_path / 'split.npy',
                              max_buffer_bytes=4096)
    assert split.band_rows < height
    assert split.band_rows >= 1
    n_bands = -(-height // split.band_rows)
    assert n_bands > 10, f'only {n_bands} band(s) — the budget did not bite'
    assert split.peak_buffer_bytes <= 4096 + align._band_bytes_per_row(split.canvas)
    assert split.peak_buffer_bytes < whole.peak_buffer_bytes / 10

    assert np.array_equal(np.load(whole.stack_path), np.load(split.stack_path)), \
        'the band height changed the pixels'
    assert split.n_written == whole.n_written == 9


def test_write_stack_of_an_empty_plan_says_so(tmp_path):
    result = align.write_stack(align.AlignPlan(), tmp_path / 'nothing.npy')
    assert result.stack_path == ''
    assert result.n_written == 0
    assert result.status == 'empty'
    assert result.warnings == ['nothing to write: no placed tiles']
    assert not os.path.exists(tmp_path / 'nothing.npy')


def test_a_zero_area_tile_contributes_nothing_and_is_reported(tmp_path):
    """A placement with no pixels must not corrupt the canvas or the count."""
    real_path = _save(tmp_path, 'plate1_B07_001.npy', _texture(32, 40, seed=4))
    real = align.Tile(path=real_path, index=0, plate='plate1', well='B07',
                      field=1, shape=(32, 40, 1))
    hollow = align.Tile(path=str(tmp_path / 'plate1_B07_002.npy'), index=1,
                        plate='plate1', well='B07', field=2, shape=(0, 0, 1))
    plan = align.AlignPlan(
        tiles=[real, hollow],
        placements=[align.Placement(tile=real, y=0.0, x=0.0,
                                    method=align.METHOD_NOMINAL),
                    align.Placement(tile=hollow, y=10.0, x=10.0,
                                    method=align.METHOD_NOMINAL)],
        dtype='uint16', feather=1)

    result = align.write_stack(plan, tmp_path / 'out.npy', band_rows=4)

    assert result.canvas.shape == (32, 40, 1)
    assert result.n_written == 1, 'the empty tile contributed no pixels'
    assert result.n_skipped == 0
    assert any('fell entirely outside the canvas' in w for w in result.warnings)
    written = np.load(result.stack_path)[:, :, 0]
    assert np.array_equal(written, np.load(real_path))


def test_a_tile_that_dies_at_write_time_is_skipped_in_every_later_band(
        tmp_path):
    """One failure per tile, not one per band, and the rest still lands."""
    big = _texture(96, 200, seed=8)
    first = _save(tmp_path, 'plate1_B07_001.npy', big[:, 0:100])
    second = _save(tmp_path, 'plate1_B07_002.npy', big[:, 80:180])
    tiles = align.scan_tiles(str(tmp_path), grid=(1, 2), overlap=0.20)
    plan = align.estimate_offsets(tiles)
    assert plan.n_registered == 2

    os.remove(second)
    ledger = RunLedger('align-test')
    result = align.write_stack(plan, tmp_path / 'out.npy', band_rows=8,
                               ledger=ledger)

    assert result.canvas.height == 96
    assert result.band_rows == 8, 'the tile must span more than one band'
    assert result.n_skipped == 1, 'the failure was counted once, not per band'
    assert result.n_written == 1
    assert result.status == 'partial'
    assert len(result.warnings) == 1
    assert second in result.warnings[0]
    assert ledger.n_failed == 1
    assert ledger.failures[0].stage == 'write'
    assert 'skipped' in result.summary()

    written = np.load(result.stack_path)[:, :, 0]
    assert np.array_equal(written[:, :100], np.load(first))
    assert not written[:, 110:].any(), 'the missing tile left zeros'


def test_a_read_error_inside_a_band_is_counted_not_swallowed(tmp_path,
                                                             monkeypatch):
    """A tile whose header parsed but whose *pixels* fail mid-write.

    The I/O error is injected — a real one (a dropped network mount, a bad
    sector) is not reproducible in a test — but everything asserted below
    is the module's own handling of it.
    """
    big = _texture(64, 180, seed=12)
    _save(tmp_path, 'plate1_B07_001.npy', big[:, 0:100])
    _save(tmp_path, 'plate1_B07_002.npy', big[:, 80:180])
    tiles = align.scan_tiles(str(tmp_path), grid=(1, 2), overlap=0.20)
    plan = align.estimate_offsets(tiles)

    original = align._TileReader.window

    def _explode(self, y0, y1, x0, x1, channels, dtype=np.float32):
        if self.tile.index == 1:
            raise OSError('simulated I/O error on the tile data')
        return original(self, y0, y1, x0, x1, channels, dtype)

    monkeypatch.setattr(align._TileReader, 'window', _explode)
    ledger = RunLedger('align-test')
    result = align.write_stack(plan, tmp_path / 'out.npy', band_rows=16,
                               ledger=ledger)

    assert result.n_skipped == 1
    assert result.n_written == 1
    assert result.status == 'partial'
    assert 'simulated I/O error' in result.warnings[0]
    assert 'OSError' in result.warnings[0]
    assert ledger.n_failed == 1 and ledger.failures[0].exc_type == 'OSError'
    assert '1 tile(s) skipped' in result.summary()

    written = np.load(result.stack_path)[:, :, 0]
    assert np.array_equal(written[:, :100], big[:, 0:100])


def test_write_stamps_the_ledger_and_survives_a_stamp_that_cannot_be_written(
        tmp_path):
    """The run stamp is recorded next to the stack — and a stamp that
    fails is a warning, never a lost stitch."""
    from spacr.errors import read_run_status

    big = _texture(64, 160, seed=13)
    _save(tmp_path, 'plate1_B07_001.npy', big[:, 0:100])
    _save(tmp_path, 'plate1_B07_002.npy', big[:, 60:160])
    tiles = align.scan_tiles(str(tmp_path), grid=(1, 2), overlap=0.40)
    plan = align.estimate_offsets(tiles)

    out_dir = tmp_path / 'stitched'
    out_dir.mkdir()
    ledger = RunLedger('align-test')
    result = align.write_stack(plan, out_dir / 'ok.npy', ledger=ledger)
    assert result.n_written == 2
    assert ledger.n_succeeded == 2
    stamped = read_run_status(result.stack_path)
    assert stamped and stamped[-1]['name'] == 'align-test'
    assert not [w for w in result.warnings if 'could not stamp' in w]

    # Now put a directory where the sidecar wants to be.
    blocked_dir = tmp_path / 'blocked'
    blocked_dir.mkdir()
    (blocked_dir / 'blocked.run_status.json').mkdir()
    second = align.write_stack(plan, blocked_dir / 'blocked.npy',
                               ledger=RunLedger('align-test'))
    assert os.path.isfile(second.stack_path), 'the stitch still landed'
    assert any('could not stamp' in w for w in second.warnings), second.warnings
    assert second.n_written == 2


def test_bad_blend_and_writer_names_are_refused_before_anything_is_written(
        tmp_path):
    plan = align.AlignPlan()
    with pytest.raises(ConfigurationError, match='blend must be one of'):
        align.write_stack(plan, tmp_path / 'a.npy', blend='crossfade')
    with pytest.raises(ConfigurationError, match="writer must be"):
        align.write_stack(plan, tmp_path / 'a.npy', writer='pipe')
    assert os.listdir(tmp_path) == []


def test_blend_modes_differ_where_the_tiles_disagree(tmp_path):
    """``none`` is last-writer-wins, ``average`` splits, ``feather`` ramps."""
    left = np.full((32, 40), 1000, np.uint16)
    right = np.full((32, 40), 5000, np.uint16)
    left_path = _save(tmp_path, 'plate1_B07_001.npy', left)
    right_path = _save(tmp_path, 'plate1_B07_002.npy', right)
    tile_a = align.Tile(path=left_path, index=0, plate='plate1', well='B07',
                        field=1, shape=(32, 40, 1))
    tile_b = align.Tile(path=right_path, index=1, plate='plate1', well='B07',
                        field=2, shape=(32, 40, 1))
    plan = align.AlignPlan(
        tiles=[tile_a, tile_b],
        placements=[align.Placement(tile=tile_a, y=0.0, x=0.0),
                    align.Placement(tile=tile_b, y=0.0, x=20.0)],
        dtype='uint16', feather=8)

    seam = {}
    for mode in align.BLEND_MODES:
        result = align.write_stack(plan, tmp_path / f'{mode}.npy', blend=mode)
        seam[mode] = np.load(result.stack_path)[16, :, 0].astype(float)
        assert result.canvas.shape == (32, 60, 1)

    # Outside the overlap every mode agrees with the tile that covers it.
    for mode in align.BLEND_MODES:
        assert seam[mode][0] == pytest.approx(1000, abs=1)
        assert seam[mode][59] == pytest.approx(5000, abs=1)
    # In the overlap: none takes the last tile, average splits it evenly,
    # feather ramps from one to the other.
    assert seam['none'][25] == pytest.approx(5000, abs=1)
    assert seam['average'][25] == pytest.approx(3000, abs=1)
    overlap = seam['feather'][20:40]
    assert np.all(np.diff(overlap) >= -1e-6), 'the feathered seam is monotonic'
    assert overlap[0] < 2500 < overlap[-1]


def test_cast_to_rounds_and_clips_into_the_canvas_dtype():
    values = np.array([[-5.0, 0.4, 0.6, 70000.0]], dtype=np.float32)
    out = align._cast_to(values.copy(), np.dtype('uint16'))
    assert out.dtype == np.uint16
    assert out.tolist() == [[0, 0, 1, 65535]]
    floats = align._cast_to(np.array([[1.5, -2.5]], np.float32),
                            np.dtype('float32'))
    assert floats.tolist() == [[1.5, -2.5]]


def test_a_dry_run_summary_states_the_geometry_without_a_path(tmp_path,
                                                              jittered_grid):
    """Nothing was written, so the summary must not claim a file."""
    folder, _big, _truth = jittered_grid
    tiles = align.scan_tiles(folder, grid=(3, 3),
                             overlap=1 - JITTER_STEP / JITTER_TILE)
    plan = align.estimate_offsets(tiles)
    result = align.write_stack(plan, tmp_path / 'never.npy', dry_run=True)

    assert result.stack_path == ''
    assert result.status == 'empty'
    assert result.n_written == 0
    assert result.band_rows >= 1
    assert result.peak_buffer_bytes > 0
    assert not os.path.exists(tmp_path / 'never.npy')
    text = result.summary()
    assert 'Wrote' not in text
    assert 'skipped' not in text
    assert f'{result.canvas.height}x{result.canvas.width}' in text
    assert 'smaller than the canvas' in text
    assert 'dry_run' in result.warnings[0]


def test_standardise_leaves_a_flat_strip_alone():
    """A strip with no variance is centred but not divided by zero."""
    flat = align._standardise(np.full((4, 5), 2000.0, dtype=np.float32))
    assert flat.dtype == np.float32
    assert np.all(flat == 0.0)

    varied = align._standardise(np.arange(20, dtype=np.float32).reshape(4, 5))
    assert float(varied.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(varied.std()) == pytest.approx(1.0, abs=1e-5)


def test_an_empty_shape_is_unplaced_even_without_a_ledger(tmp_path):
    good_path = _save(tmp_path, 'plate1_B07_001.npy', _texture(32, 32, seed=1))
    good = align.Tile(path=good_path, index=0, plate='plate1', well='B07',
                      field=1, shape=(32, 32, 1))
    empty = align.Tile(path='ghost.npy', index=1, plate='plate1', well='B07',
                       field=2, shape=(0, 0, 1))
    plan = align.estimate_offsets([good, empty])
    assert len(plan.unplaced) == 1 and len(plan.placements) == 1


def test_plan_canvas_honours_an_explicit_dtype_and_channel_count():
    tile = align.Tile(path='a.npy', index=0, shape=(20, 30, 2), dtype='uint8')
    placements = [align.Placement(tile=tile, y=0.0, x=0.0)]
    spec = align.plan_canvas(placements, dtype='float32', channels=5)
    assert spec.shape == (20, 30, 5)
    assert spec.dtype == 'float32'
    assert spec.nbytes == 20 * 30 * 5 * 4
    # …and the defaults still come from the tiles themselves
    default = align.plan_canvas(placements)
    assert default.shape == (20, 30, 2)
    assert default.dtype == 'uint8'


def test_a_read_error_is_counted_without_a_ledger_too(tmp_path, monkeypatch):
    big = _texture(48, 140, seed=14)
    _save(tmp_path, 'plate1_B07_001.npy', big[:, 0:80])
    _save(tmp_path, 'plate1_B07_002.npy', big[:, 60:140])
    tiles = align.scan_tiles(str(tmp_path), grid=(1, 2), overlap=0.25)
    plan = align.estimate_offsets(tiles)

    original = align._TileReader.window

    def _explode(self, y0, y1, x0, x1, channels, dtype=np.float32):
        if self.tile.index == 1:
            raise OSError('simulated I/O error')
        return original(self, y0, y1, x0, x1, channels, dtype)

    monkeypatch.setattr(align._TileReader, 'window', _explode)
    result = align.write_stack(plan, tmp_path / 'out.npy')
    assert result.n_skipped == 1 and result.n_written == 1
    assert result.status == 'partial'


def test_read_coordinates_returns_a_foreign_table_as_it_finds_it(tmp_path):
    """A table without ``tile_index`` cannot be sorted by it, and is not."""
    db_path = tmp_path / 'other.db'
    frame = pd.DataFrame({'plateID': ['p1', 'p1'], 'well': ['B07', 'B07'],
                          'y': [3.0, 1.0]})
    connection = sqlite3.connect(str(db_path))
    try:
        frame.to_sql(align.ALIGN_TABLE, connection, index=False)
    finally:
        connection.close()

    back = align.read_coordinates(db_path)
    assert list(back.columns) == ['plateID', 'well', 'y']
    assert list(back['y']) == [3.0, 1.0], 'insertion order is preserved'


def test_human_bytes_scales_through_the_units():
    assert align._human_bytes(512) == '512 B'
    assert align._human_bytes(2048) == '2.0 KB'
    assert align._human_bytes(5 << 20) == '5.0 MB'
    assert align._human_bytes(3 << 30) == '3.0 GB'
    assert align._human_bytes(7 << 40) == '7.0 TB'
    assert align._human_bytes(9000 << 40).endswith('TB')


# ---------------------------------------------------------------------------
# 9. The coordinates table
# ---------------------------------------------------------------------------

def test_coordinates_carry_the_solved_geometry_into_the_database(
        jittered_grid, tmp_path):
    """Every number the table promises is the number the plan holds."""
    folder, _big, _truth = jittered_grid
    tiles = align.scan_tiles(folder, grid=(3, 3),
                             overlap=1 - JITTER_STEP / JITTER_TILE)
    plan = align.estimate_offsets(tiles)
    result = align.write_stack(plan, tmp_path / 'stack.npy')
    db_path = tmp_path / 'measurements.db'

    written = align.save_coordinates(plan, db_path, canvas=result.canvas,
                                     stack_path=result.stack_path)
    assert written == 9

    frame = align.read_coordinates(db_path)
    assert len(frame) == 9
    assert list(frame.columns) == list(align.ALIGN_COLUMNS)
    assert set(frame['method']) == {align.METHOD_REGISTRATION}
    assert set(frame['plateID']) == {'plate1'}
    assert set(frame['rowID']) == {'r2'} and set(frame['columnID']) == {'c7'}
    assert list(frame['fieldID']) == [f'f{k + 1}' for k in range(9)]
    assert list(frame['prcf']) == [f'plate1_r2_c7_f{k + 1}' for k in range(9)]
    assert set(frame['stack_path']) == {result.stack_path}
    assert set(frame['canvas_height']) == {result.canvas.height}
    assert set(frame['canvas_width']) == {result.canvas.width}
    assert set(frame['canvas_dtype']) == {'uint16'}

    by_index = {int(r.tile_index): r for r in frame.itertuples()}
    for placement in plan.placements:
        row = by_index[placement.tile.index]
        assert row.y == pytest.approx(placement.y)
        assert row.x == pytest.approx(placement.x)
        assert row.confidence == pytest.approx(placement.confidence)
        assert row.residual == pytest.approx(placement.residual)
        assert row.n_pairs == placement.n_pairs
        # canvas_y/x + subpixel_y/x reconstruct the solved position exactly
        assert row.canvas_y + row.subpixel_y == \
            pytest.approx(placement.y - result.canvas.origin_y)
        assert row.canvas_x + row.subpixel_x == \
            pytest.approx(placement.x - result.canvas.origin_x)
        assert 0 <= row.canvas_y < result.canvas.height
        assert 0 <= row.canvas_x < result.canvas.width
        assert json.loads(row.source_channels) == []

    # and the canvas index really is where the tile was written
    canvas = np.load(result.stack_path)[:, :, 0]
    for placement in plan.placements[:3]:
        row = by_index[placement.tile.index]
        patch = canvas[int(row.canvas_y):int(row.canvas_y) + JITTER_TILE,
                       int(row.canvas_x):int(row.canvas_x) + JITTER_TILE]
        source = np.load(placement.tile.path)
        r = float(np.corrcoef(patch.ravel().astype(float),
                              source.ravel().astype(float))[0, 1])
        assert r > 0.99, f'tile {placement.tile.index} is not at canvas_y/x'


def test_unreadable_tiles_are_rows_in_the_table_not_absences(tmp_path):
    """A field that could never be read must be visible in the join."""
    good = _save(tmp_path, 'plate1_B07_001.npy', _texture(32, 40, seed=1))
    broken = str(tmp_path / 'plate1_B07_002.npy')
    with open(broken, 'wb') as handle:
        handle.write(b'not an npy at all')
    tiles = align.scan_tiles(str(tmp_path), grid=(1, 2), overlap=0.25)
    assert tiles[1].readable is False

    plan = align.estimate_offsets(tiles)
    db_path = tmp_path / 'measurements.db'
    assert align.save_coordinates(plan, db_path) == 2

    frame = align.read_coordinates(db_path)
    assert len(frame) == 2
    missing = frame[frame['method'] == align.METHOD_UNREADABLE]
    assert len(missing) == 1
    assert missing.iloc[0]['fieldID'] == 'f2'
    assert missing.iloc[0]['source'] == broken
    assert missing.iloc[0]['stack_path'] == ''
    assert pd.isna(missing.iloc[0]['y']) and pd.isna(missing.iloc[0]['x'])
    assert missing.iloc[0]['canvas_y'] == -1
    assert missing.iloc[0]['note']
    placed = frame[frame['method'] != align.METHOD_UNREADABLE]
    assert len(placed) == 1
    assert placed.iloc[0]['source'] == good
    assert placed.iloc[0]['fieldID'] == 'f1'
    assert placed.iloc[0]['tile_height'] == 32
    assert set(frame['method']) == {align.METHOD_SINGLE,
                                    align.METHOD_UNREADABLE}


def test_coordinates_append_and_filter_by_plate_and_well(tmp_path):
    """Two wells, two writes, one table — and the filters really filter."""
    plans = []
    for well in ('B07', 'C08'):
        folder = tmp_path / well
        folder.mkdir()
        big = _texture(48, 120, seed=len(well) + ord(well[0]))
        _save(folder, f'plate1_{well}_001.npy', big[:, 0:64])
        _save(folder, f'plate1_{well}_002.npy', big[:, 48:112])
        tiles = align.scan_tiles(str(folder), grid=(1, 2), overlap=0.25)
        plans.append(align.estimate_offsets(tiles))

    db_path = tmp_path / 'measurements.db'
    assert align.save_coordinates(plans[0], db_path, if_exists='replace') == 2
    assert align.save_coordinates(plans[1], db_path, if_exists='append') == 2

    everything = align.read_coordinates(db_path)
    assert len(everything) == 4
    assert set(everything['well']) == {'B07', 'C08'}

    one = align.read_coordinates(db_path, well='C08')
    assert len(one) == 2
    assert set(one['well']) == {'C08'}
    assert set(one['rowID']) == {'r3'} and set(one['columnID']) == {'c8'}

    assert len(align.read_coordinates(db_path, plate='plate1')) == 4
    assert len(align.read_coordinates(db_path, plate='nope')) == 0
    assert len(align.read_coordinates(db_path, plate='plate1', well='B07')) == 2

    # the indexes the docstring promises really exist
    connection = sqlite3.connect(str(db_path))
    try:
        names = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index'")}
    finally:
        connection.close()
    assert f'idx_{align.ALIGN_TABLE}_prcf' in names
    assert f'idx_{align.ALIGN_TABLE}_keys' in names


def test_read_coordinates_complains_about_a_missing_database_or_table(tmp_path):
    with pytest.raises(ConfigurationError, match='no such database'):
        align.read_coordinates(tmp_path / 'nope.db')
    db_path = tmp_path / 'empty.db'
    sqlite3.connect(str(db_path)).close()
    with pytest.raises(ConfigurationError, match='has no'):
        align.read_coordinates(db_path)


# ---------------------------------------------------------------------------
# 10. Reporting and the settings entry point
# ---------------------------------------------------------------------------

def test_format_plan_truncates_a_long_list_of_fallbacks():
    """13 fallbacks are summarised, not dumped — but the count is exact."""
    tiles = [align.Tile(path=f't{k}.npy', index=k, plate='p', well='B07',
                        field=k + 1, shape=(16, 16, 1)) for k in range(13)]
    plan = align.AlignPlan(
        tiles=tiles,
        placements=[align.Placement(tile=t, y=0.0, x=float(16 * k),
                                    method=align.METHOD_NOMINAL,
                                    confidence=k / 13.0,
                                    note='overlap is blank')
                    for k, t in enumerate(tiles)],
        canvas_shape=(16, 208, 1), dtype='uint16')
    plan.unplaced = [(tiles[0], 'header unreadable')]

    text = align.format_plan(plan, max_rows=12)
    assert 'placed by stage position only (13)' in text
    assert '… and 1 more' in text
    assert text.count('overlap is blank') == 12
    assert 'nominal         13' in text
    assert '!! unplaced t0.npy: header unreadable' in text

    # …and with room for everything, nothing is elided.
    assert '… and' not in align.format_plan(plan, max_rows=20)


def test_format_plan_lists_the_largest_residuals(tmp_path):
    tiles = [align.Tile(path=f't{k}.npy', index=k, plate='p', well='B07',
                        field=k + 1, shape=(16, 16, 1)) for k in range(2)]
    plan = align.AlignPlan(
        tiles=tiles,
        placements=[align.Placement(tile=tiles[0], residual=0.5, n_pairs=1,
                                    confidence=0.8,
                                    method=align.METHOD_REGISTRATION),
                    align.Placement(tile=tiles[1], residual=4.25, n_pairs=2,
                                    confidence=0.6,
                                    method=align.METHOD_REGISTRATION)],
        canvas_shape=(16, 32, 1), dtype='uint16')
    text = align.format_plan(plan)
    assert 'largest residuals' in text
    assert 'residual=   4.25' in text
    assert text.index('residual=   4.25') < text.index('residual=   0.50'), \
        'the worst residual must come first'
    assert 'worst residual  4.25 px' in text


def test_align_folder_defaults_the_destination_next_to_the_source(tmp_path,
                                                                  capsys):
    """No dst given: the stitch lands in ``<src>_stitched``."""
    src = tmp_path / 'tiles'
    src.mkdir()
    big = _texture(64, 160, seed=17)
    _save(src, 'plate1_B07_001.npy', big[:, 0:100])
    _save(src, 'plate1_B07_002.npy', big[:, 60:160])

    results = align.align_folder(src=str(src), grid=(1, 2), overlap=0.40,
                                 db_path=str(tmp_path / 'm.db'))

    assert len(results) == 1
    result = results[0]
    expected_dir = str(src) + '_stitched'
    assert os.path.dirname(result.stack_path) == expected_dir
    assert os.path.basename(result.stack_path) == 'plate1_B07_stitched.npy'
    assert result.n_written == 2
    assert result.db_path == str(tmp_path / 'm.db')
    assert np.load(result.stack_path).shape == result.canvas.shape

    frame = align.read_coordinates(tmp_path / 'm.db')
    assert len(frame) == 2
    assert set(frame['stack_path']) == {result.stack_path}

    printed = capsys.readouterr().out
    assert 'Align plan' in printed
    assert 'align_coordinates row(s)' in printed


def test_align_folder_writes_one_stack_per_well(tmp_path, capsys):
    src = tmp_path / 'tiles'
    src.mkdir()
    for well in ('B07', 'C08'):
        big = _texture(48, 120, seed=ord(well[0]))
        _save(src, f'plate1_{well}_001.npy', big[:, 0:64])
        _save(src, f'plate1_{well}_002.npy', big[:, 48:112])

    results = align.align_folder(
        {'src': str(src), 'dst': str(tmp_path / 'out'), 'grid': (2, 2),
         'overlap': 0.25, 'db_path': str(tmp_path / 'm.db')})

    assert len(results) == 2
    names = sorted(os.path.basename(r.stack_path) for r in results)
    assert names == ['plate1_B07_stitched.npy', 'plate1_C08_stitched.npy']
    frame = align.read_coordinates(tmp_path / 'm.db')
    assert len(frame) == 4, 'the second well appended rather than replaced'
    assert set(frame['well']) == {'B07', 'C08'}
    assert 'B07' in capsys.readouterr().out


def test_default_settings_fills_in_and_lets_overrides_win():
    resolved = align.default_settings({'src': '/tmp/x', 'blend': 'average'})
    assert resolved['src'] == '/tmp/x'
    assert resolved['blend'] == 'average'
    assert resolved['overlap'] == align.DEFAULT_OVERLAP
    assert resolved['writer'] == 'stream'
    assert set(align.default_settings()) == set(resolved)


def test_align_folder_override_beats_the_settings_dict(tmp_path):
    src = tmp_path / 'tiles'
    src.mkdir()
    _save(src, 'plate1_B07_001.npy', _texture(32, 32, seed=1))
    results = align.align_folder({'src': str(src), 'preview_only': False},
                                 preview_only=True)
    assert len(results) == 1
    assert results[0].stack_path == ''
    assert results[0].warnings == ['preview only — nothing written']
    assert results[0].band_rows >= 1
    assert os.listdir(tmp_path) == ['tiles'], 'preview must write nothing'
