"""spacr.align — stitching an arbitrary number of tiles without the canvas in RAM.

The suite pins the properties this module lives or dies by:

* a grid with **known** offsets is recovered to sub-pixel accuracy;
* the offsets are solved **globally**, so error does not accumulate along a
  chain the way sequential accumulation makes it (the load-bearing test);
* a **blank overlap** does not produce a confident placement — it falls
  back to the nominal stage position and is marked as such;
* the canvas is sized from the offsets, **negative ones included**;
* the canvas is **never materialised in RAM**: the write refuses to make
  any array anywhere near canvas-sized, and the peak buffer it reports is
  a fraction of the output;
* the seam is **feathered**, so two tiles that disagree join by a ramp
  rather than a step;
* the coordinates **round-trip** through ``measurements.db`` and join to a
  ``cell`` table on the same four keys every other spaCR table uses;
* a single tile, tiles that never overlap, mixed dtypes and an unreadable
  tile are each handled and each *reported*;
* every channel of a site shares **one** alignment solution;
* importing the module does not drag in torch, cellpose or Qt.

Everything is deterministic (fixed seeds), CPU-only and offline.
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from spacr import align
from spacr.errors import ConfigurationError


# ---------------------------------------------------------------------------
# Synthetic tile builders
# ---------------------------------------------------------------------------

def _texture(height, width, seed=0, sigma=2.0):
    """A smooth, non-repeating uint16 field — registrable, unlike noise."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    raw = rng.random((height, width)).astype(np.float32)
    smooth = gaussian_filter(raw, sigma)
    smooth -= smooth.min()
    smooth /= max(float(smooth.max()), 1e-9)
    return (smooth * 30000 + 1000).astype(np.uint16)


def _write_grid(folder, big, rows, cols, tile, step, plate='plate1',
                well='B07', dtype=None, suffix='.npy'):
    """Cut ``big`` into a ``rows`` x ``cols`` grid and write it out.

    :returns: ``{tile_index: (true_y, true_x)}``.
    """
    os.makedirs(folder, exist_ok=True)
    truth = {}
    k = 0
    for r in range(rows):
        for c in range(cols):
            y, x = r * step, c * step
            crop = big[y:y + tile, x:x + tile]
            if dtype is not None:
                crop = crop.astype(dtype)
            path = os.path.join(folder, f'{plate}_{well}_{k + 1:03d}{suffix}')
            if suffix == '.npy':
                np.save(path, crop)
            else:
                import tifffile
                tifffile.imwrite(path, crop)
            truth[k] = (y, x)
            k += 1
    return truth


@pytest.fixture(scope='module')
def grid_3x3(tmp_path_factory):
    """A 3x3 grid of 256x256 tiles stepped 200 px — 56 px of overlap."""
    folder = tmp_path_factory.mktemp('grid3')
    big = _texture(900, 900, seed=7)
    truth = _write_grid(str(folder), big, 3, 3, 256, 200)
    return str(folder), big, truth


# ---------------------------------------------------------------------------
# 1. Known offsets are recovered to sub-pixel accuracy
# ---------------------------------------------------------------------------

def test_scan_tiles_reads_headers_only(grid_3x3):
    """Metadata comes out; no pixel is read (the header path is public API)."""
    folder, _big, _truth = grid_3x3
    tiles = align.scan_tiles(folder, grid=(3, 3), overlap=1 - 200 / 256)
    assert len(tiles) == 9
    assert [t.index for t in tiles] == list(range(9))
    assert all(t.shape == (256, 256, 1) for t in tiles)
    assert all(t.dtype == 'uint16' for t in tiles)
    assert all(t.readable for t in tiles)
    assert (tiles[0].plate, tiles[0].well, tiles[0].field) == ('plate1', 'B07', 1)
    # Nominal layout: row-major, step = tile * (1 - overlap) = 200.
    assert [(t.grid_row, t.grid_col) for t in tiles[:4]] == \
        [(0, 0), (0, 1), (0, 2), (1, 0)]
    assert tiles[4].nominal_y == pytest.approx(200.0)
    assert tiles[4].nominal_x == pytest.approx(200.0)


def test_known_offsets_recovered_subpixel(grid_3x3):
    """Every tile lands within a fraction of a pixel of where it was cut."""
    folder, _big, truth = grid_3x3
    tiles = align.scan_tiles(folder, grid=(3, 3), overlap=1 - 200 / 256)
    plan = align.estimate_offsets(tiles)

    assert plan.n_registered == 9
    assert plan.n_nominal == 0
    assert len(plan.accepted_pairs) == 20        # 4-neighbour + diagonals

    origin_y, origin_x = plan.origin
    for placement in plan.placements:
        true_y, true_x = truth[placement.tile.index]
        assert placement.y - origin_y == pytest.approx(true_y, abs=0.25)
        assert placement.x - origin_x == pytest.approx(true_x, abs=0.25)
        assert placement.method == align.METHOD_REGISTRATION
        assert placement.confidence > 0.9
    assert plan.max_residual < 0.25


def test_registration_reads_only_the_overlap_strip(grid_3x3, monkeypatch):
    """The registration pass never asks a tile for more than its overlap."""
    folder, _big, _truth = grid_3x3
    tiles = align.scan_tiles(folder, grid=(3, 3), overlap=1 - 200 / 256)

    seen = []
    original = align._TileReader.window

    def _spy(self, y0, y1, x0, x1, channels, dtype=np.float32):
        seen.append((int(y1) - int(y0), int(x1) - int(x0)))
        return original(self, y0, y1, x0, x1, channels, dtype)

    monkeypatch.setattr(align._TileReader, 'window', _spy)
    align.estimate_offsets(tiles)

    assert seen, 'no windows were read at all'
    # A tile is 256x256; the largest read is the 256x56 / 56x256 overlap.
    biggest = max(h * w for h, w in seen)
    assert biggest <= 256 * 60, f'read a {biggest}-pixel window from a tile'
    assert all(h <= 256 and w <= 256 for h, w in seen)


# ---------------------------------------------------------------------------
# 2. The load-bearing one: global solve vs sequential accumulation
# ---------------------------------------------------------------------------

N_CHAIN = 10
CHAIN_STEP = 20.0
CHAIN_TRUTH = np.arange(N_CHAIN) * CHAIN_STEP
CHAIN_NOMINAL = np.column_stack([np.zeros(N_CHAIN), CHAIN_TRUTH])


def _chain_edges(radius, bias=0.0, outlier=None):
    """Pair measurements for a 1-D chain of ``N_CHAIN`` heavily-overlapped tiles.

    ``radius`` is how many tiles along still overlap — a chain stepped by a
    quarter of a tile has radius 3. ``bias`` is a fixed sub-pixel error
    present in *every* measurement; ``outlier`` is ``((i, j), error)`` for
    one pair that got it badly wrong.
    """
    edges = []
    for distance in range(1, radius + 1):
        for k in range(N_CHAIN - distance):
            error = bias
            if outlier is not None and (k, k + distance) == outlier[0]:
                error = outlier[1]
            edges.append((k, k + distance, 0.0,
                          distance * CHAIN_STEP + error, 1.0))
    return edges


def _errors(positions):
    """Per-tile positional error, gauge-fixed on tile 0."""
    return np.abs(positions[:, 1] - positions[0, 1] - CHAIN_TRUTH)


def test_global_solve_beats_sequential_drift():
    """A 10-tile chain: accumulation drifts, the global solve does not.

    THE load-bearing test. Ten tiles in a row overlapping by 75%, so each
    also registers against the tiles two and three along — the redundancy
    a real acquisition has and a sequential walk throws away.

    Part one is drift proper. Every pairwise measurement carries the same
    fixed +0.3 px error, which is what phase correlation returns when the
    overlap is vignetted or asymmetric. Chaining adds it up: the error
    grows *exactly linearly*, 0.3 px per tile, and the last tile is 2.7 px
    out with nothing in the output admitting it. The global solve sees the
    same +0.3 on a two-tile hop as on a one-tile hop, and those two
    statements disagree; least squares splits the difference, so the error
    grows sub-linearly and the far end is more than twice as good — and
    the leftover disagreement comes back as a residual instead of
    vanishing.

    Part two is why that matters more than the numbers suggest. One pair
    is off by 5 px — a dust speck, a bubble, one bad overlap. Sequential
    accumulation chains straight through it and every single tile after it
    inherits the whole 5 px. The global solve outvotes it with the
    surrounding edges and, crucially, the residual peaks on exactly the
    two tiles of the bad pair, which is what makes the failure findable.
    """
    # -- part one: a systematic bias in every measurement ----------------
    edges = _chain_edges(radius=3, bias=0.3)
    solved, residuals, degrees = align.solve_positions(
        N_CHAIN, edges, CHAIN_NOMINAL)
    chained = align._sequential_positions(N_CHAIN, edges, CHAIN_NOMINAL)

    solved_err, chained_err = _errors(solved), _errors(chained)

    # Sequential drift is linear and unmistakable: +0.3 px per tile.
    assert np.allclose(chained_err, 0.3 * np.arange(N_CHAIN), atol=1e-6)
    assert chained_err[-1] == pytest.approx(2.7, abs=1e-6)
    # The global solve grows sub-linearly and ends up >2x closer.
    assert solved_err[-1] < chained_err[-1] / 2.0
    assert solved_err[-1] < 0.15 * N_CHAIN     # sub-linear in the chain length
    # The disagreement it could not remove is reported, not averaged away.
    assert residuals.max() > 0.05
    assert (degrees > 0).all()

    # -- part two: one pair that got it badly wrong ----------------------
    bad = ((4, 5), 5.0)
    edges = _chain_edges(radius=2, outlier=bad)
    solved, residuals, _degrees = align.solve_positions(
        N_CHAIN, edges, CHAIN_NOMINAL)
    chained = align._sequential_positions(N_CHAIN, edges, CHAIN_NOMINAL)

    solved_err, chained_err = _errors(solved), _errors(chained)

    # Sequential hands the entire 5 px to every tile downstream of the bad
    # pair, and leaves everything upstream perfect — so nothing anywhere
    # looks wrong.
    assert np.allclose(chained_err[:5], 0.0, atol=1e-6)
    assert np.allclose(chained_err[5:], 5.0, atol=1e-6)
    # The global solve absorbs it: no tile is more than 2 px out.
    assert solved_err.max() < 2.0, solved_err
    assert solved_err.max() < chained_err.max() / 2.5
    # And it points at the culprit — the residual peaks on tiles 4 and 5.
    assert int(np.argmax(residuals)) in (4, 5)
    assert residuals[4] == pytest.approx(residuals[5], rel=1e-6)
    assert residuals[4] > 3 * residuals[0]


def test_a_pure_chain_gives_the_same_answer_as_accumulating():
    """Honesty check: with no redundancy there is nothing to win.

    A pair graph that is a tree has exactly one path between any two
    tiles, so least squares and sequential accumulation agree to machine
    precision. Redundancy is what buys the improvement, and this pins that
    the module is not claiming otherwise.
    """
    edges = _chain_edges(radius=1, bias=0.3)
    solved, residuals, _ = align.solve_positions(N_CHAIN, edges, CHAIN_NOMINAL)
    chained = align._sequential_positions(N_CHAIN, edges, CHAIN_NOMINAL)
    assert np.allclose(_errors(solved), _errors(chained), atol=1e-3)
    # No edge disagrees with any other, so there is no residual to report
    # (bar the anchor's negligible pull toward the stage positions).
    assert residuals.max() < 1e-4


def test_a_tile_with_no_pairs_lands_on_its_nominal_position():
    """The anchor equations place an isolated tile, they do not zero it."""
    nominal = np.array([[0.0, 0.0], [0.0, 20.0], [7.0, 900.0]])
    edges = [(0, 1, 0.0, 20.0, 1.0)]
    solved, residuals, degrees = align.solve_positions(3, edges, nominal)
    assert degrees[2] == 0
    assert residuals[2] == 0.0
    assert solved[2] == pytest.approx([7.0, 900.0], abs=1e-6)


def test_global_solve_bridges_a_pair_that_did_not_register(tmp_path):
    """A broken link in a real chain does not derail everything after it.

    Ten real tiles overlapping by 60%, so each registers against both the
    next tile and the one after. The middle *adjacent* pair is destroyed —
    its strip is blanked — which is precisely the case where sequential
    accumulation has nothing to chain through. The redundant edges carry
    the solve across it, and the tiles beyond the break still land where
    they were cut.
    """
    tile, step = 200, 80                        # 120 px overlap = 60%
    big = _texture(300, step * 9 + tile + 10, seed=11)
    folder = tmp_path / 'chain'
    folder.mkdir()
    for k in range(10):
        crop = big[:tile, k * step:k * step + tile].copy()
        if k == 5:
            # Tile 5's left half is blank, so pair (4,5) has no signal in
            # its overlap at all. Pair (3,5) and (5,7) still do.
            crop[:, :step] = 0
        np.save(folder / f'plate1_B07_{k + 1:03d}.npy', crop)

    tiles = align.scan_tiles(str(folder), grid=(1, 10), overlap=1 - step / tile)
    plan = align.estimate_offsets(tiles, neighbour_radius=2)

    refused = {(p.i, p.j) for p in plan.refused_pairs}
    assert (4, 5) in refused, 'the destroyed pair should not have registered'

    origin_x = plan.origin[1]
    for placement in plan.placements:
        expected = placement.tile.index * step
        assert placement.x - origin_x == pytest.approx(expected, abs=1.0), (
            f'tile {placement.tile.index} landed at '
            f'{placement.x - origin_x:.2f}, expected {expected}')
    assert plan.n_registered == 10


# ---------------------------------------------------------------------------
# 3. A blank overlap is refused, not believed
# ---------------------------------------------------------------------------

def test_blank_overlap_falls_back_to_nominal_and_says_so(tmp_path):
    """Two tiles whose overlap is featureless keep their stage positions."""
    folder = tmp_path / 'blank'
    folder.mkdir()
    left = _texture(200, 200, seed=3)
    right = _texture(200, 200, seed=4)
    left[:, 150:] = 500                          # flat overlap on both sides
    right[:, :50] = 500
    np.save(folder / 'plate1_B07_001.npy', left)
    np.save(folder / 'plate1_B07_002.npy', right)

    tiles = align.scan_tiles(str(folder), grid=(1, 2), overlap=0.25)
    plan = align.estimate_offsets(tiles)

    assert len(plan.overlaps) == 1
    pair = plan.overlaps[0]
    assert pair.accepted is False
    assert pair.confidence == 0.0
    assert 'blank' in pair.note.lower()

    assert plan.n_registered == 0
    assert plan.n_nominal == 2
    for placement in plan.placements:
        assert placement.method == align.METHOD_NOMINAL
        assert placement.confidence == 0.0
        assert placement.y == pytest.approx(placement.tile.nominal_y)
        assert placement.x == pytest.approx(placement.tile.nominal_x)
    assert any('nominal' in w for w in plan.warnings)
    assert align.METHOD_NOMINAL in align.format_plan(plan)


def test_noise_overlap_is_refused_on_confidence(tmp_path):
    """Uncorrelated overlaps score below threshold and are refused.

    Phase correlation returns a sharp, confident-looking peak for two
    unrelated noise fields. The cross-correlation check is what stops it
    becoming a placement.
    """
    folder = tmp_path / 'noise'
    folder.mkdir()
    np.save(folder / 'plate1_B07_001.npy', _texture(200, 200, seed=21))
    np.save(folder / 'plate1_B07_002.npy', _texture(200, 200, seed=99))

    tiles = align.scan_tiles(str(folder), grid=(1, 2), overlap=0.25)
    plan = align.estimate_offsets(tiles)

    pair = plan.overlaps[0]
    assert pair.accepted is False
    assert 'min_confidence' in pair.note
    assert plan.n_nominal == 2


def test_max_shift_refuses_an_implausible_correction(tmp_path):
    """A correction larger than max_shift is refused rather than applied."""
    folder = tmp_path / 'shift'
    folder.mkdir()
    big = _texture(200, 300, seed=5)
    np.save(folder / 'plate1_B07_001.npy', big[:, 0:200])
    np.save(folder / 'plate1_B07_002.npy', big[:, 60:260])

    # overlap=0.6 puts the nominal at +80; the truth is +60, so the
    # correction is -20 px within a 120 px wide overlap.
    tiles = align.scan_tiles(str(folder), grid=(1, 2), overlap=0.6)
    assert tiles[1].nominal_x == pytest.approx(80.0)

    generous = align.estimate_offsets(tiles)
    assert generous.n_registered == 2
    assert (generous.placements[1].x - generous.placements[0].x) == \
        pytest.approx(60.0, abs=0.25)

    strict = align.estimate_offsets(tiles, max_shift=5.0)
    assert strict.overlaps[0].accepted is False
    assert 'max_shift' in strict.overlaps[0].note
    assert strict.n_nominal == 2
    assert strict.placements[1].x == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# 4. Canvas geometry, including negative offsets
# ---------------------------------------------------------------------------

def _bare_tile(index, height, width, dtype='uint16', channels=1):
    return align.Tile(path=f'/nowhere/{index}.npy', index=index,
                      shape=(height, width, channels), dtype=dtype)


def test_plan_canvas_from_positive_offsets():
    """Bounding box of the placed tiles, exactly."""
    placements = [
        align.Placement(tile=_bare_tile(0, 100, 100), y=0, x=0),
        align.Placement(tile=_bare_tile(1, 100, 100), y=0, x=80),
        align.Placement(tile=_bare_tile(2, 100, 100), y=80, x=0),
        align.Placement(tile=_bare_tile(3, 100, 100), y=80, x=80),
    ]
    spec = align.plan_canvas(placements)
    assert spec.shape == (180, 180, 1)
    assert (spec.origin_y, spec.origin_x) == (0.0, 0.0)
    assert spec.nbytes == 180 * 180 * 2


def test_plan_canvas_handles_negative_offsets():
    """A tile above/left of the origin widens the canvas, it is not clipped."""
    placements = [
        align.Placement(tile=_bare_tile(0, 100, 100), y=0, x=0),
        align.Placement(tile=_bare_tile(1, 100, 100), y=-30.5, x=-12.25),
        align.Placement(tile=_bare_tile(2, 100, 100), y=90, x=90),
    ]
    spec = align.plan_canvas(placements)
    assert (spec.origin_y, spec.origin_x) == (-31.0, -13.0)
    assert spec.shape == (221, 203, 1)
    # Every tile fits inside the canvas once the origin is applied.
    for placement in placements:
        cy, cx = spec.canvas_yx(placement.y, placement.x)
        assert cy >= 0 and cx >= 0
        assert cy + placement.tile.height <= spec.height
        assert cx + placement.tile.width <= spec.width


def test_plan_canvas_promotes_dtype_and_channels():
    """The canvas is wide enough for the widest tile in both senses."""
    placements = [
        align.Placement(tile=_bare_tile(0, 10, 10, 'uint8', 1), y=0, x=0),
        align.Placement(tile=_bare_tile(1, 10, 10, 'uint16', 3), y=0, x=5),
    ]
    spec = align.plan_canvas(placements)
    assert spec.dtype == 'uint16'
    assert spec.channels == 3


def test_plan_canvas_of_nothing_is_empty_not_an_error():
    spec = align.plan_canvas([])
    assert spec.shape == (0, 0, 0)
    assert spec.nbytes == 0


# ---------------------------------------------------------------------------
# 5. The memory bound
# ---------------------------------------------------------------------------

def test_write_never_allocates_anything_near_the_canvas(tmp_path, monkeypatch):
    """Aligning into a canvas far bigger than the tiles allocates no canvas.

    Sixteen 128x128 tiles spread over a 4096x4096 canvas (32 MB). Every
    numpy allocator is booby-trapped: anything asking for more than an
    eighth of the canvas blows up. If ``write_stack`` ever built the
    canvas — or a float32 copy of it — in RAM, this fails.
    """
    folder = tmp_path / 'sparse'
    folder.mkdir()
    tiles = []
    for k in range(16):
        path = folder / f'plate1_B07_{k + 1:03d}.npy'
        np.save(path, _texture(128, 128, seed=100 + k))
        r, c = divmod(k, 4)
        tiles.append(align.Tile(
            path=str(path), index=k, plate='plate1', well='B07', field=k + 1,
            shape=(128, 128, 1), dtype='uint16', grid_row=r, grid_col=c,
            nominal_y=r * 1300.0, nominal_x=c * 1300.0))

    plan = align.estimate_offsets(tiles)
    spec = align.plan_canvas(plan.placements)
    canvas_bytes = spec.nbytes
    assert canvas_bytes > 30_000_000, spec.shape

    limit = canvas_bytes // 8
    real_zeros, real_empty, real_full = np.zeros, np.empty, np.full

    def _guard(name, real):
        def _wrapped(shape, *args, **kwargs):
            try:
                count = int(np.prod(shape))
            except TypeError:
                count = 1
            if count * 8 > limit:
                raise AssertionError(
                    f'np.{name} asked for {count} elements — that is '
                    f'canvas-sized, and the canvas must live on disk')
            return real(shape, *args, **kwargs)
        return _wrapped

    monkeypatch.setattr(np, 'zeros', _guard('zeros', real_zeros))
    monkeypatch.setattr(np, 'empty', _guard('empty', real_empty))
    monkeypatch.setattr(np, 'full', _guard('full', real_full))

    result = align.write_stack(plan, str(tmp_path / 'out'), band_rows=64)

    monkeypatch.undo()
    assert result.peak_buffer_bytes * 8 < canvas_bytes
    assert os.path.getsize(result.stack_path) > canvas_bytes

    # And the output really is a full-size on-disk array.
    written = np.load(result.stack_path, mmap_mode='r')
    assert isinstance(written, np.memmap)
    assert written.shape == spec.shape


def test_band_rows_scale_with_the_budget_not_the_canvas():
    """The RAM plan is a function of the budget; the canvas may be anything."""
    small = align.CanvasSpec(height=1000, width=4096, channels=1, dtype='uint16')
    huge = align.CanvasSpec(height=200000, width=4096, channels=1, dtype='uint16')
    budget = 8 << 20
    assert (align._band_rows(small, budget, None)
            == align._band_rows(huge, budget, None))
    assert align._band_rows(huge, budget, None) * \
        align._band_bytes_per_row(huge) <= budget


def test_dry_run_reports_the_geometry_and_writes_nothing(tmp_path, grid_3x3):
    folder, _big, _truth = grid_3x3
    tiles = align.scan_tiles(folder, grid=(3, 3), overlap=1 - 200 / 256)
    plan = align.estimate_offsets(tiles)
    out = tmp_path / 'dry'
    result = align.write_stack(plan, str(out), dry_run=True)
    assert result.stack_path == ''
    assert result.canvas.shape == plan.canvas_shape
    assert result.band_rows > 0
    assert not out.exists()


# ---------------------------------------------------------------------------
# 6. Seam blending
# ---------------------------------------------------------------------------

def test_feathered_seam_has_no_step_between_equal_tiles(tmp_path):
    """Two tiles of the same constant reconstruct that constant exactly."""
    folder = tmp_path / 'flat'
    folder.mkdir()
    for k in range(2):
        np.save(folder / f'plate1_B07_{k + 1:03d}.npy',
                np.full((64, 100), 4321, dtype=np.uint16))
    tiles = align.scan_tiles(str(folder), grid=(1, 2), overlap=0.3)
    plan = align.estimate_offsets(tiles)          # blank overlap -> nominal
    result = align.write_stack(plan, str(tmp_path / 'out'), band_rows=16)

    canvas = np.load(result.stack_path)[:, :, 0]
    assert canvas.min() == 4321 and canvas.max() == 4321


def test_feathering_turns_a_step_into_a_ramp(tmp_path):
    """Tiles that disagree join by a gradient, not by a visible edge.

    A hard cut between a 1000-valued tile and a 5000-valued one leaves a
    4000-count edge running the height of the canvas — a straight line
    that any downstream segmentation will happily call an object boundary.
    The feather spreads it over the overlap.
    """
    folder = tmp_path / 'contrast'
    folder.mkdir()
    np.save(folder / 'plate1_B07_001.npy', np.full((64, 100), 1000, np.uint16))
    np.save(folder / 'plate1_B07_002.npy', np.full((64, 100), 5000, np.uint16))
    tiles = align.scan_tiles(str(folder), grid=(1, 2), overlap=0.4)
    plan = align.estimate_offsets(tiles)
    overlap_px = 40

    hard = np.load(align.write_stack(
        plan, str(tmp_path / 'hard'), blend='none',
        band_rows=16).stack_path)[:, :, 0]
    soft = np.load(align.write_stack(
        plan, str(tmp_path / 'soft'), blend='feather', feather=overlap_px,
        band_rows=16).stack_path)[:, :, 0]

    row_hard = hard[32].astype(np.int32)
    row_soft = soft[32].astype(np.int32)

    # The hard cut has a single 4000-count cliff.
    assert np.abs(np.diff(row_hard)).max() >= 3900
    # The feather's worst step is a small fraction of that...
    worst_soft = int(np.abs(np.diff(row_soft)).max())
    assert worst_soft < 4000 / 10, worst_soft
    # ...and the transition is monotone across the seam, i.e. a ramp.
    seam = row_soft[55:105]
    assert np.all(np.diff(seam) >= 0)
    assert row_soft[0] == 1000 and row_soft[-1] == 5000


def test_feather_width_is_taken_from_the_real_overlap(grid_3x3):
    folder, _big, _truth = grid_3x3
    tiles = align.scan_tiles(folder, grid=(3, 3), overlap=1 - 200 / 256)
    plan = align.estimate_offsets(tiles)
    assert 40 <= plan.feather <= 64, plan.feather


# ---------------------------------------------------------------------------
# 7. The database: round-trip and join
# ---------------------------------------------------------------------------

def test_join_keys_match_utils_map_wells():
    """The keys are byte-identical to what the rest of spaCR writes.

    ``align`` cannot import ``spacr.utils`` (it pulls in torch, and this
    module is on the GUI path), so the eight lines are reimplemented. This
    is the test that keeps the copy honest — if it drifts, the join in
    :func:`spacr.align.save_coordinates` silently returns no rows.
    """
    from spacr.utils import _map_wells
    for plate, well, field in [('plate1', 'B07', 3), ('p2', 'A1', 12),
                               ('x', 'H12', 1), ('plate1', '7', 2)]:
        expected_plate, row, column, field_id, prcf = _map_wells(
            f'{plate}_{well}_{field}')
        keys = align._join_keys(plate, well, field)
        assert keys['plateID'] == expected_plate
        assert keys['rowID'] == row
        assert keys['columnID'] == column
        assert keys['fieldID'] == field_id
        assert keys['prcf'] == prcf


def test_coordinates_round_trip_and_join_to_cell(tmp_path, grid_3x3):
    """Write, read back, and join to a synthetic ``cell`` table on the keys."""
    folder, _big, _truth = grid_3x3
    tiles = align.scan_tiles(folder, grid=(3, 3), overlap=1 - 200 / 256)
    plan = align.estimate_offsets(tiles)
    result = align.write_stack(plan, str(tmp_path / 'out'))

    db_path = tmp_path / 'measurements.db'
    # A cell table keyed exactly the way spaCR keys them.
    rows = []
    for tile in tiles:
        keys = align._join_keys(tile.plate, tile.well, tile.field)
        for obj in range(3):
            rows.append({**keys, 'object_label': obj + 1,
                         'cell_area': 100.0 + obj})
    connection = sqlite3.connect(db_path)
    try:
        pd.DataFrame(rows).to_sql('cell', connection, index=False)
    finally:
        connection.close()

    written = align.save_coordinates(plan, db_path, canvas=result.canvas,
                                     stack_path=result.stack_path)
    assert written == 9

    frame = align.read_coordinates(db_path)
    assert list(frame.columns) == list(align.ALIGN_COLUMNS)
    assert len(frame) == 9
    assert set(frame['method']) == {align.METHOD_REGISTRATION}
    assert (frame['canvas_y'] >= 0).all() and (frame['canvas_x'] >= 0).all()
    assert (frame['stack_path'] == result.stack_path).all()

    # Coordinates survive the trip unchanged.
    by_index = {int(r.tile_index): r for r in frame.itertuples()}
    for placement in plan.placements:
        row = by_index[placement.tile.index]
        assert row.y == pytest.approx(placement.y)
        assert row.x == pytest.approx(placement.x)
        assert row.residual == pytest.approx(placement.residual)

    # The four-key join every other spaCR table uses.
    connection = sqlite3.connect(db_path)
    try:
        joined = pd.read_sql_query(
            f'SELECT c.object_label, a.canvas_y, a.canvas_x, a.method '
            f'FROM cell AS c JOIN {align.ALIGN_TABLE} AS a '
            f'  ON  c.plateID  = a.plateID AND c.rowID    = a.rowID '
            f'  AND c.columnID = a.columnID AND c.fieldID = a.fieldID',
            connection)
        one_col = pd.read_sql_query(
            f'SELECT count(*) AS n FROM cell AS c '
            f'JOIN {align.ALIGN_TABLE} AS a ON c.prcf = a.prcf', connection)
    finally:
        connection.close()
    assert len(joined) == 27                    # 9 fields x 3 objects
    assert int(one_col['n'][0]) == 27
    assert set(joined['method']) == {align.METHOD_REGISTRATION}


def test_read_coordinates_filters_and_complains_clearly(tmp_path, grid_3x3):
    folder, _big, _truth = grid_3x3
    plan = align.estimate_offsets(
        align.scan_tiles(folder, grid=(3, 3), overlap=1 - 200 / 256))
    db_path = tmp_path / 'm.db'

    with pytest.raises(ConfigurationError, match='no such database'):
        align.read_coordinates(db_path)

    align.save_coordinates(plan, db_path)
    assert len(align.read_coordinates(db_path, plate='plate1')) == 9
    assert len(align.read_coordinates(db_path, plate='nope')) == 0
    assert len(align.read_coordinates(db_path, well='B07')) == 9

    with pytest.raises(ConfigurationError, match='has no'):
        align.read_coordinates(db_path, table='not_a_table')


def test_save_coordinates_accepts_several_plans(tmp_path, grid_3x3):
    """One table for a whole plate — several wells, one write."""
    folder, _big, _truth = grid_3x3
    tiles = align.scan_tiles(folder, grid=(3, 3), overlap=1 - 200 / 256)
    plan_a = align.estimate_offsets(tiles)
    other = [align.Tile(**{**vars(t), 'well': 'C08'}) for t in tiles]
    plan_b = align.estimate_offsets(other)
    db_path = tmp_path / 'm.db'
    assert align.save_coordinates([plan_a, plan_b], db_path) == 18
    frame = align.read_coordinates(db_path)
    assert set(frame['rowID']) == {'r2', 'r3'}


# ---------------------------------------------------------------------------
# 8. Edge cases, each handled and each reported
# ---------------------------------------------------------------------------

def test_single_tile_is_a_copy_and_is_labelled(tmp_path):
    folder = tmp_path / 'one'
    folder.mkdir()
    data = _texture(64, 80, seed=1)
    np.save(folder / 'plate1_B07_001.npy', data)

    plan = align.estimate_offsets(align.scan_tiles(str(folder)))
    assert len(plan.placements) == 1
    assert plan.placements[0].method == align.METHOD_SINGLE
    assert plan.placements[0].confidence == 1.0
    assert plan.canvas_shape == (64, 80, 1)
    assert any('one tile' in w for w in plan.warnings)

    result = align.write_stack(plan, str(tmp_path / 'out'))
    assert np.array_equal(np.load(result.stack_path)[:, :, 0], data)


def test_non_overlapping_tiles_are_reported_not_registered(tmp_path):
    """Tiles that never touch get their stage positions and a warning."""
    folder = tmp_path / 'apart'
    folder.mkdir()
    for k in range(2):
        np.save(folder / f'plate1_B07_{k + 1:03d}.npy', _texture(50, 50, seed=k))
    tiles = align.scan_tiles(str(folder), grid=(1, 2), overlap=0.0)
    plan = align.estimate_offsets(tiles)

    assert plan.overlaps == []
    assert plan.n_nominal == 2
    assert any('no two tiles overlap' in w for w in plan.warnings)
    assert plan.canvas_shape == (50, 100, 1)

    result = align.write_stack(plan, str(tmp_path / 'out'))
    assert result.n_written == 2
    canvas = np.load(result.stack_path)[:, :, 0]
    assert np.array_equal(canvas[:, :50], np.load(folder / 'plate1_B07_001.npy'))
    assert np.array_equal(canvas[:, 50:], np.load(folder / 'plate1_B07_002.npy'))


def test_mixed_dtypes_are_promoted_and_warned_about(tmp_path):
    folder = tmp_path / 'mixed'
    folder.mkdir()
    big = _texture(64, 160, seed=6)
    np.save(folder / 'plate1_B07_001.npy', big[:, 0:100].astype(np.uint8))
    np.save(folder / 'plate1_B07_002.npy', big[:, 60:160].astype(np.uint16))

    tiles = align.scan_tiles(str(folder), grid=(1, 2), overlap=0.4)
    plan = align.estimate_offsets(tiles)

    assert plan.dtype == 'uint16'
    assert any('mixed dtypes' in w for w in plan.warnings)
    result = align.write_stack(plan, str(tmp_path / 'out'))
    assert np.load(result.stack_path).dtype == np.uint16


def test_mixed_shapes_are_reported_and_the_canvas_fits_them_all(tmp_path):
    folder = tmp_path / 'shapes'
    folder.mkdir()
    np.save(folder / 'plate1_B07_001.npy', _texture(64, 100, seed=8))
    np.save(folder / 'plate1_B07_002.npy', _texture(80, 100, seed=9))
    tiles = align.scan_tiles(str(folder), grid=(1, 2), overlap=0.2)
    plan = align.estimate_offsets(tiles)
    assert any('different shapes' in w for w in plan.warnings)
    assert plan.canvas_shape[0] == 80


def test_unreadable_tile_is_reported_and_written_into_the_table(tmp_path):
    """A corrupt tile is named, excluded from the canvas, and kept in the DB."""
    folder = tmp_path / 'broken'
    folder.mkdir()
    big = _texture(64, 160, seed=12)
    np.save(folder / 'plate1_B07_001.npy', big[:, 0:100])
    np.save(folder / 'plate1_B07_002.npy', big[:, 60:160])
    (folder / 'plate1_B07_003.npy').write_bytes(b'\x93NUMPY not really')

    tiles = align.scan_tiles(str(folder), grid=(1, 3), overlap=0.4)
    assert sum(1 for t in tiles if not t.readable) == 1

    plan = align.estimate_offsets(tiles)
    assert len(plan.placements) == 2
    assert len(plan.unplaced) == 1
    bad_tile, reason = plan.unplaced[0]
    assert bad_tile.path.endswith('003.npy')
    assert reason
    assert 'unplaced' in align.format_plan(plan)

    db_path = tmp_path / 'm.db'
    assert align.save_coordinates(plan, db_path) == 3
    frame = align.read_coordinates(db_path)
    bad = frame[frame['method'] == align.METHOD_UNREADABLE]
    assert len(bad) == 1
    assert bool(np.isnan(bad.iloc[0]['y']))
    assert int(bad.iloc[0]['canvas_y']) == -1


def test_tile_that_dies_at_write_time_is_counted_not_swallowed(tmp_path):
    """A header that parses but pixels that do not is a *write* failure."""
    folder = tmp_path / 'late'
    folder.mkdir()
    big = _texture(64, 160, seed=13)
    good = folder / 'plate1_B07_001.npy'
    bad = folder / 'plate1_B07_002.npy'
    np.save(good, big[:, 0:100])
    np.save(bad, big[:, 60:160])

    tiles = align.scan_tiles(str(folder), grid=(1, 2), overlap=0.4)
    plan = align.estimate_offsets(tiles)
    # Truncate after planning: the header still reads, the data does not.
    bad.write_bytes(b'not an npy at all')

    result = align.write_stack(plan, str(tmp_path / 'out'))
    assert result.n_skipped == 1
    assert result.status == 'partial'
    assert any('plate1_B07_002.npy' in w for w in result.warnings)
    assert result.n_written == 1


def test_no_readable_tiles_is_a_configuration_error(tmp_path):
    folder = tmp_path / 'allbad'
    folder.mkdir()
    (folder / 'plate1_B07_001.npy').write_bytes(b'\x93NUMPY junk')
    tiles = align.scan_tiles(str(folder))
    with pytest.raises(ConfigurationError, match='none of the'):
        align.estimate_offsets(tiles)


def test_missing_source_and_empty_source_complain_clearly(tmp_path):
    with pytest.raises(ConfigurationError, match='does not exist'):
        align.scan_tiles(str(tmp_path / 'nope'))
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(ConfigurationError, match='no .npy/.tif tiles'):
        align.scan_tiles(str(empty))


def test_four_dimensional_tile_is_refused_with_a_z_message(tmp_path):
    """z is not stitched, and the message says what to do instead."""
    path = tmp_path / 'plate1_B07_001.npy'
    np.save(path, np.zeros((3, 4, 8, 8), dtype=np.uint16))
    tiles = align.scan_tiles(str(tmp_path))
    assert not tiles[0].readable
    assert 'z is not stitched' in tiles[0].error


def test_grid_too_small_and_bad_options_are_refused(tmp_path):
    folder = tmp_path / 'g'
    folder.mkdir()
    for k in range(4):
        np.save(folder / f'plate1_B07_{k + 1:03d}.npy', np.zeros((8, 8), np.uint16))
    with pytest.raises(ConfigurationError, match='has room for'):
        align.scan_tiles(str(folder), grid=(1, 2))
    with pytest.raises(ConfigurationError, match='overlap must be'):
        align.scan_tiles(str(folder), overlap=1.5)
    with pytest.raises(ConfigurationError, match='order must be'):
        align.scan_tiles(str(folder), order='spiral')

    tiles = align.scan_tiles(str(folder), grid=(2, 2))
    plan = align.estimate_offsets(tiles)
    with pytest.raises(ConfigurationError, match='blend must be'):
        align.write_stack(plan, str(tmp_path / 'o'), blend='magic')
    with pytest.raises(ConfigurationError, match="writer must be"):
        align.write_stack(plan, str(tmp_path / 'o'), writer='magic')


def test_existing_output_is_not_clobbered_without_overwrite(tmp_path, grid_3x3):
    folder, _big, _truth = grid_3x3
    plan = align.estimate_offsets(
        align.scan_tiles(folder, grid=(3, 3), overlap=1 - 200 / 256))
    out = tmp_path / 'stitched.npy'
    align.write_stack(plan, str(out))
    with pytest.raises(ConfigurationError, match='already exists'):
        align.write_stack(plan, str(out))
    align.write_stack(plan, str(out), overwrite=True)      # allowed


def test_snake_order_lays_alternate_rows_backwards(tmp_path):
    folder = tmp_path / 'snake'
    folder.mkdir()
    for k in range(6):
        np.save(folder / f'plate1_B07_{k + 1:03d}.npy', np.zeros((10, 10), np.uint16))
    tiles = align.scan_tiles(str(folder), grid=(2, 3), overlap=0.0,
                             order='snake-row')
    assert [(t.grid_row, t.grid_col) for t in tiles] == \
        [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]


def test_explicit_positions_override_the_grid(tmp_path):
    folder = tmp_path / 'pos'
    folder.mkdir()
    for k in range(2):
        np.save(folder / f'plate1_B07_{k + 1:03d}.npy', np.zeros((10, 10), np.uint16))
    tiles = align.scan_tiles(str(folder), grid=(1, 2),
                             positions={1: (5.0, 6.0), 2: (7.0, 8.0)})
    assert (tiles[0].nominal_y, tiles[0].nominal_x) == (5.0, 6.0)
    assert (tiles[1].nominal_y, tiles[1].nominal_x) == (7.0, 8.0)


# ---------------------------------------------------------------------------
# 9. Channels share one solution
# ---------------------------------------------------------------------------

def test_channels_share_one_solution_from_the_reference_channel(tmp_path):
    """Two channels, one placement — and channel 1 cannot move it.

    Channel 2 is deliberately *misregistrable*: its overlap is blank. If
    the channels were solved independently, channel 2 would fall back to
    nominal and the composite would shear. Because registration runs on
    the reference channel only, both planes land on channel 1's answer.
    """
    folder = tmp_path / 'chan'
    folder.mkdir()
    big = _texture(64, 200, seed=15)
    true_shift = 40                                # tiles overlap by 60
    for k, x0 in enumerate((0, true_shift)):
        c1 = big[:, x0:x0 + 100]
        c2 = np.full((64, 100), 777, dtype=np.uint16)
        np.save(folder / f'plate1_B07_T0001F{k + 1:03d}L01A01Z01C01.npy', c1)
        np.save(folder / f'plate1_B07_T0001F{k + 1:03d}L01A01Z01C02.npy', c2)

    tiles = align.scan_tiles(str(folder), grid=(1, 2), overlap=0.5)
    assert len(tiles) == 2, 'the two channels must collapse into one site'
    assert tiles[0].n_channels == 2
    assert len(tiles[0].channel_paths) == 2

    plan_c0 = align.estimate_offsets(tiles, reference_channel=0)
    assert plan_c0.n_registered == 2
    assert (plan_c0.placements[1].x - plan_c0.placements[0].x) == \
        pytest.approx(true_shift, abs=0.25)

    # Registering on the blank channel instead gives nominal — proof that
    # the reference channel is what decides, and why it must be one channel.
    plan_c1 = align.estimate_offsets(tiles, reference_channel=1)
    assert plan_c1.n_nominal == 2

    result = align.write_stack(plan_c0, str(tmp_path / 'out'))
    canvas = np.load(result.stack_path)
    assert canvas.shape[0] == 64 and canvas.shape[2] == 2
    # Both planes span the same width, set by channel 1's registration.
    assert canvas.shape[1] == pytest.approx(100 + true_shift, abs=2)
    # One solution: channel 2's constant covers exactly the footprint
    # channel 1's registration implies, with no offset of its own. (The
    # outermost column can be canvas padding from the fractional origin,
    # so the footprints are compared, not the raw extremes.)
    covered_c0 = canvas[:, :, 0] > 0
    covered_c1 = canvas[:, :, 1] > 0
    assert np.array_equal(covered_c0, covered_c1)
    assert set(np.unique(canvas[:, :, 1][covered_c1])) == {777}


def test_reference_channel_out_of_range_falls_back_to_channel_zero(tmp_path):
    folder = tmp_path / 'refchan'
    folder.mkdir()
    for k in range(2):
        np.save(folder / f'plate1_B07_{k + 1:03d}.npy', _texture(32, 32, seed=k))
    tiles = align.scan_tiles(str(folder), grid=(1, 2), overlap=0.0,
                             reference_channel=99)
    assert all(t.channel == 0 for t in tiles)


def test_three_dimensional_npy_goes_through_crops_merged_field(tmp_path):
    """A merged (H, W, C) stack is read with the on-demand crop machinery."""
    folder = tmp_path / 'merged'
    folder.mkdir()
    big = _texture(64, 200, seed=17)
    for k, x0 in enumerate((0, 60)):
        stack = np.stack([big[:, x0:x0 + 100],
                          big[:, x0:x0 + 100] // 2], axis=-1)
        np.save(folder / f'plate1_B07_{k + 1:03d}.npy', stack)

    tiles = align.scan_tiles(str(folder), grid=(1, 2), overlap=0.4)
    assert tiles[0].shape == (64, 100, 2)
    assert tiles[0].channel_paths == ()

    reader = align._TileReader(tiles[0])
    try:
        assert reader._fields[0] is not None, \
            'a 3-D .npy must be opened through crops.open_merged_field'
        window = reader.window(0, 8, 0, 8, [0, 1])
        assert window.shape == (8, 8, 2)
        assert np.allclose(window[:, :, 0], big[:8, :8])
    finally:
        reader.close()

    plan = align.estimate_offsets(tiles)
    assert plan.n_registered == 2
    assert plan.canvas_shape == (64, 160, 2)


# ---------------------------------------------------------------------------
# 10. TIFF input, folder entry point, reporting
# ---------------------------------------------------------------------------

def test_tiff_tiles_register_the_same_way(tmp_path):
    tifffile = pytest.importorskip('tifffile')
    folder = tmp_path / 'tif'
    folder.mkdir()
    big = _texture(96, 260, seed=19)
    for k, x0 in enumerate((0, 80)):
        tifffile.imwrite(folder / f'plate1_B07_{k + 1:03d}.tif',
                         big[:, x0:x0 + 160])
    tiles = align.scan_tiles(str(folder), grid=(1, 2), overlap=0.5)
    assert tiles[0].shape == (96, 160, 1)
    plan = align.estimate_offsets(tiles)
    assert plan.n_registered == 2
    assert (plan.placements[1].x - plan.placements[0].x) == \
        pytest.approx(80, abs=0.25)


def test_align_folder_end_to_end(tmp_path, grid_3x3, capsys):
    """The settings-dict entry point: plan printed, stack written, DB filled."""
    folder, _big, _truth = grid_3x3
    results = align.align_folder({
        'src': folder,
        'dst': str(tmp_path / 'out'),
        'db_path': str(tmp_path / 'measurements.db'),
        'grid': (3, 3),
        'overlap': 1 - 200 / 256,
        'overwrite': True,
    })
    printed = capsys.readouterr().out
    assert 'Align plan' in printed
    assert 'registered      9' in printed
    assert len(results) == 1
    assert os.path.isfile(results[0].stack_path)
    assert results[0].db_path
    assert len(align.read_coordinates(tmp_path / 'measurements.db')) == 9


def test_align_folder_preview_writes_nothing(tmp_path, grid_3x3):
    folder, _big, _truth = grid_3x3
    results = align.align_folder({'src': folder, 'grid': (3, 3),
                                  'overlap': 1 - 200 / 256,
                                  'preview_only': True})
    assert results[0].stack_path == ''
    assert results[0].canvas.height > 0
    assert not os.path.exists(str(tmp_path / 'out'))


def test_align_folder_needs_a_src():
    with pytest.raises(ConfigurationError, match="needs a 'src'"):
        align.align_folder({})


def test_group_tiles_splits_by_well_and_renumbers():
    tiles = [
        align.Tile(path='a', index=0, plate='p', well='B07', field=1),
        align.Tile(path='b', index=1, plate='p', well='C08', field=1),
        align.Tile(path='c', index=2, plate='p', well='B07', field=2),
    ]
    groups = align.group_tiles(tiles)
    assert set(groups) == {('p', 'B07'), ('p', 'C08')}
    assert [t.index for t in groups[('p', 'B07')]] == [0, 1]
    assert [t.field for t in groups[('p', 'B07')]] == [1, 2]


def test_default_settings_round_trips_overrides():
    settings = align.default_settings({'src': 'x', 'blend': 'average'})
    assert settings['src'] == 'x'
    assert settings['blend'] == 'average'
    assert settings['writer'] == 'stream'
    assert settings['max_buffer_bytes'] == align.DEFAULT_MAX_BUFFER_BYTES


def test_format_plan_leads_with_what_went_wrong(tmp_path):
    folder = tmp_path / 'fmt'
    folder.mkdir()
    np.save(folder / 'plate1_B07_001.npy', np.full((40, 60), 100, np.uint16))
    np.save(folder / 'plate1_B07_002.npy', np.full((40, 60), 100, np.uint16))
    plan = align.estimate_offsets(
        align.scan_tiles(str(folder), grid=(1, 2), overlap=0.3))
    text = align.format_plan(plan)
    assert 'placed by stage position only' in text
    assert 'canvas' in text and 'nominal' in text
    assert 'feather' in text


def test_result_summary_states_the_memory_ratio(tmp_path, grid_3x3):
    folder, _big, _truth = grid_3x3
    plan = align.estimate_offsets(
        align.scan_tiles(folder, grid=(3, 3), overlap=1 - 200 / 256))
    result = align.write_stack(plan, str(tmp_path / 'out'), band_rows=32)
    text = result.summary()
    assert 'Peak RAM buffer' in text
    assert 'smaller than the canvas' in text
    assert result.peak_buffer_bytes < result.canvas.nbytes


def test_run_ledger_records_per_tile_outcomes(tmp_path):
    from spacr.errors import RunLedger
    folder = tmp_path / 'ledger'
    folder.mkdir()
    big = _texture(64, 160, seed=23)
    np.save(folder / 'plate1_B07_001.npy', big[:, 0:100])
    np.save(folder / 'plate1_B07_002.npy', big[:, 60:160])
    (folder / 'plate1_B07_003.npy').write_bytes(b'\x93NUMPY junk')

    ledger = RunLedger('align-test')
    tiles = align.scan_tiles(str(folder), grid=(1, 3), overlap=0.4)
    align.estimate_offsets(tiles, ledger=ledger)
    assert ledger.n_failed >= 1
    assert ledger.n_succeeded == 2


def test_subpixel_write_resamples_without_shifting_the_canvas(tmp_path):
    """The sub-pixel option changes pixels, not geometry."""
    folder = tmp_path / 'sub'
    folder.mkdir()
    big = _texture(64, 200, seed=25)
    np.save(folder / 'plate1_B07_001.npy', big[:, 0:120])
    np.save(folder / 'plate1_B07_002.npy', big[:, 70:190])
    tiles = align.scan_tiles(str(folder), grid=(1, 2), overlap=0.4)
    plan = align.estimate_offsets(tiles)
    plain = align.write_stack(plan, str(tmp_path / 'a.npy'))
    fancy = align.write_stack(plan, str(tmp_path / 'b.npy'), subpixel=True)
    assert plain.canvas.shape == fancy.canvas.shape
    assert np.load(plain.stack_path).shape == np.load(fancy.stack_path).shape


# ---------------------------------------------------------------------------
# 11. Import hygiene
# ---------------------------------------------------------------------------

#: Import the module without running ``spacr/__init__.py``.
#:
#: ``spacr`` is an eager package — its ``__init__`` imports ``core``,
#: ``utils`` and friends, which import torch — so ``import spacr.align``
#: can never be clean and would test nothing. A stub package object with
#: the right ``__path__`` lets ``align``'s own relative imports resolve
#: while leaving the package ``__init__`` unexecuted, which is exactly the
#: dependency set this module is responsible for.
_ISOLATED_IMPORT = """
import sys, types
before = set(sys.modules)
pkg = types.ModuleType('spacr')
pkg.__path__ = [{spacr_dir!r}]
sys.modules['spacr'] = pkg
import spacr.align
added = {{m.split('.')[0] for m in sys.modules}} - {{m.split('.')[0] for m in before}}
heavy = {{'torch', 'cellpose', 'PySide6', 'PyQt5', 'PyQt6', 'tensorflow',
          'tkinter', 'matplotlib'}}
print(','.join(sorted(added & heavy)))
"""


def test_import_does_not_pull_in_torch_cellpose_or_qt():
    """A GUI thumbnail path must not pay for a deep-learning stack.

    Run in a subprocess: by the time this file executes, the rest of the
    suite has certainly imported torch already, so checking
    ``sys.modules`` in-process would prove nothing.

    Two details make the check honest. The modules already loaded when the
    subprocess starts are subtracted, so a ``sitecustomize`` on the path
    (the coverage runner ships one that pre-imports torch) cannot mask or
    fake the result. And ``spacr/__init__.py`` is *not* executed — it
    eagerly imports ``core``/``utils``, which import torch, so
    ``import spacr.align`` through the real package could never be clean
    and would measure the package rather than this module. A stub package
    with the right ``__path__`` lets ``align``'s relative imports resolve
    while leaving the package ``__init__`` alone, which is exactly the
    dependency set this module is responsible for.
    """
    spacr_dir = os.path.dirname(os.path.abspath(align.__file__))
    proc = subprocess.run(
        [sys.executable, '-c', _ISOLATED_IMPORT.format(spacr_dir=spacr_dir)],
        capture_output=True, text=True,
        cwd=os.path.dirname(spacr_dir))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == '', \
        f'spacr.align dragged in: {proc.stdout.strip()}'
