"""A field cut into overlapping tiles comes back as the field.

THE MEASURE IS RECONSTRUCTION, not "it produced an image". A mosaic assembled
at the wrong overlap is a field with a duplicated band through it, every
object in that band measured twice, and nothing on screen saying so -- which
is the failure mode the whole import module was written against. So each test
here takes a KNOWN image, cuts it up the way a microscope would, and asserts
the pixels come back.

The four things that can go wrong, one test each:

* the overlap is wrong -- the field is stretched or a band is doubled;
* the ORDER is wrong -- a serpentine acquisition read row-major puts the
  third tile where the fourth belongs, and looks almost right;
* the GRID is wrong -- 6 tiles are 2x3 or 3x2 and a tile index cannot say;
* there is no evidence at all -- blank tiles correlate on nothing, and the
  answer has to be "butt-joined, and I am telling you so".
"""
from __future__ import annotations

from pathlib import Path
import builtins
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

import spacr.image_stitch as image_stitch
from spacr.image_stitch import (MIN_CONFIDENCE, Mosaic, Placement,
                                arrangement_of, grid_shape, plan_mosaic,
                                read_stage_positions, stitch_tiles)

TILE = 64
OVERLAP = 16
STEP = TILE - OVERLAP


@pytest.fixture
def field():
    """A field with structure everywhere, so every seam has something to
    correlate. Uniform noise, not a gradient: a gradient correlates almost
    as well at the wrong offset as at the right one."""
    rng = np.random.default_rng(20260902)
    return rng.integers(0, 4000, size=(STEP + TILE, STEP + TILE),
                        dtype=np.uint16)


def _cut(field, root: Path, order):
    """Write ``field`` as overlapping tiles, visited in ``order``.

    :param order: ``(row, col)`` pairs in acquisition order — which is what
        a tile INDEX actually encodes, and all it encodes.
    """
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, (row, col) in enumerate(order, start=1):
        y, x = row * STEP, col * STEP
        path = root / f"tile{index:02d}.tif"
        tifffile.imwrite(str(path), field[y:y + TILE, x:x + TILE])
        paths.append(path)
    return paths


def _smooth_field(shape, seed):
    """Noise with a point-spread function over it — an IMAGE, not static.

    White noise is the hardest case for finding a displacement and the
    easiest for noticing one is wrong, which is why the other fixtures use
    it. It is also nothing a microscope produces: a PSF is at least two
    pixels wide, so real neighbouring pixels are correlated, and a stage
    that misses its row by three pixels still leaves an overlap that
    correlates. Testing jitter against white noise would be testing against
    an image no objective can form.
    """
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 4000, size=shape).astype(np.float64)
    blurred = raw.copy()
    for axis in (0, 1):
        for shift in (-2, -1, 1, 2):
            blurred += np.roll(raw, shift, axis=axis)
    return (blurred / 9).astype(np.uint16)


ROW_MAJOR = [(0, 0), (0, 1), (1, 0), (1, 1)]
SERPENTINE = [(0, 0), (0, 1), (1, 1), (1, 0)]


# ---------------------------------------------------------------------------
# The grid arithmetic, which is a guess and says so
# ---------------------------------------------------------------------------

def test_the_grid_is_as_square_as_the_count_allows():
    assert grid_shape(4) == (2, 2)
    assert grid_shape(6) == (2, 3)
    assert grid_shape(9) == (3, 3)
    assert grid_shape(3) == (1, 3)
    assert grid_shape(1) == (1, 1)
    assert grid_shape(0) == (0, 0)


def test_serpentine_reverses_every_other_row():
    """The difference between the two arrangements IS the bug they exist to
    catch: index 2 belongs at (1, 1) snaked and at (1, 0) row-major."""
    assert [arrangement_of(i, 2, 2, "row_major") for i in range(4)] == ROW_MAJOR
    assert [arrangement_of(i, 2, 2, "serpentine_rows")
            for i in range(4)] == SERPENTINE
    assert arrangement_of(1, 2, 2, "column_major") == (1, 0)
    assert arrangement_of(2, 2, 2, "serpentine_columns") == (1, 1)
    with pytest.raises(ValueError):
        arrangement_of(0, 2, 2, "spiral")


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def test_a_field_cut_into_four_comes_back_as_the_field(field, tmp_path):
    paths = _cut(field, tmp_path / "rows", ROW_MAJOR)

    mosaic = plan_mosaic(paths)
    assert mosaic.how == "correlated"
    assert (mosaic.rows, mosaic.cols) == (2, 2)
    assert mosaic.overlap == (OVERLAP, OVERLAP)
    assert mosaic.confidence > 0.9

    stitched, _mosaic = stitch_tiles(paths)
    assert stitched.shape == field.shape
    assert np.array_equal(stitched, field)


def test_a_serpentine_acquisition_is_not_read_row_major(field, tmp_path):
    """Read row-major, this field comes back with its bottom two tiles
    swapped — an image that looks like a field and is not one."""
    paths = _cut(field, tmp_path / "snake", SERPENTINE)

    mosaic = plan_mosaic(paths)
    assert mosaic.arrangement == "serpentine_rows"
    stitched, _mosaic = stitch_tiles(paths, mosaic=mosaic)
    assert np.array_equal(stitched, field)


def test_a_six_tile_field_finds_its_own_shape(tmp_path):
    """2x3 or 3x2: the count cannot say, so the pixels do."""
    rng = np.random.default_rng(7)
    field = rng.integers(0, 4000, size=(2 * STEP + TILE, STEP + TILE),
                         dtype=np.uint16)
    order = [(r, c) for r in range(3) for c in range(2)]
    paths = _cut(field, tmp_path / "six", order)

    mosaic = plan_mosaic(paths)
    assert (mosaic.rows, mosaic.cols) == (3, 2)
    stitched, _mosaic = stitch_tiles(paths, mosaic=mosaic)
    assert np.array_equal(stitched, field)


def test_the_overlap_is_measured_rather_than_assumed(tmp_path):
    """A different overlap gives a different mosaic, from the same count of
    tiles — which is the whole reason it is not a setting."""
    rng = np.random.default_rng(11)
    for overlap in (4, 24):
        step = TILE - overlap
        field = rng.integers(0, 4000, size=(step + TILE, step + TILE),
                             dtype=np.uint16)
        root = tmp_path / f"ov{overlap}"
        root.mkdir()
        paths = []
        for index, (row, col) in enumerate(ROW_MAJOR, start=1):
            y, x = row * step, col * step
            path = root / f"tile{index:02d}.tif"
            tifffile.imwrite(str(path), field[y:y + TILE, x:x + TILE])
            paths.append(path)

        stitched, mosaic = stitch_tiles(paths)
        assert mosaic.overlap == (overlap, overlap), mosaic.describe()
        assert np.array_equal(stitched, field)


def test_the_overlap_is_averaged_rather_than_overwritten(tmp_path):
    """Two tiles covering one pixel are two measurements of it. Taking the
    last writes a step at every seam that a segmenter reads as an edge."""
    root = tmp_path / "flat"
    root.mkdir()
    left = np.full((TILE, TILE), 100, dtype=np.uint16)
    right = np.full((TILE, TILE), 200, dtype=np.uint16)
    tifffile.imwrite(str(root / "tile01.tif"), left)
    tifffile.imwrite(str(root / "tile02.tif"), right)

    mosaic = Mosaic(placements=plan_mosaic([root / "tile01.tif"]).placements
                    + (type(plan_mosaic([root / "tile01.tif"]).placements[0])
                       (2, 0, 1, 0, STEP),),
                    height=TILE, width=STEP + TILE, rows=1, cols=2,
                    arrangement="row_major", how="correlated", confidence=1.0,
                    overlap=(0, OVERLAP))
    stitched, _mosaic = stitch_tiles(
        [root / "tile01.tif", root / "tile02.tif"], mosaic=mosaic)
    assert stitched[0, 0] == 100
    assert stitched[0, -1] == 200
    assert stitched[0, STEP] == 150


# ---------------------------------------------------------------------------
# When there is no evidence
# ---------------------------------------------------------------------------

def test_blank_tiles_are_butt_joined_and_say_so(tmp_path):
    """The corpus's tiled tree is blank, and so is many a real edge field.
    Nothing correlates, so nothing is claimed: the tiles go side by side and
    the mosaic reports that its seams are unverified."""
    root = tmp_path / "blank"
    root.mkdir()
    paths = []
    for index in range(1, 5):
        path = root / f"tile{index:02d}.tif"
        tifffile.imwrite(str(path), np.zeros((TILE, TILE), np.uint16))
        paths.append(path)

    mosaic = plan_mosaic(paths)
    assert mosaic.how == "assumed"
    assert mosaic.confidence < MIN_CONFIDENCE
    assert mosaic.is_believed is False
    assert (mosaic.height, mosaic.width) == (2 * TILE, 2 * TILE)
    assert "unverified" in mosaic.describe()

    stitched, _mosaic = stitch_tiles(paths)
    assert stitched.shape == (2 * TILE, 2 * TILE)


def test_one_tile_is_not_a_mosaic(tmp_path):
    path = tmp_path / "only.tif"
    tifffile.imwrite(str(path), np.zeros((TILE, TILE), np.uint16))
    mosaic = plan_mosaic([path])
    assert mosaic.how == "single"
    assert mosaic.is_believed
    assert "nothing to stitch" in mosaic.describe()


def test_nothing_at_all_is_not_a_mosaic():
    assert plan_mosaic([]) is None
    assert stitch_tiles([]) == (None, None)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_a_multipage_tile_is_refused_rather_than_flattened(tmp_path):
    """Its pages are Z, T or channel, and taking the first would stitch one
    plane of a stack into a field the caller believes is the whole thing."""
    root = tmp_path / "stack"
    root.mkdir()
    for index in range(1, 5):
        tifffile.imwrite(str(root / f"tile{index:02d}.tif"),
                         np.zeros((3, TILE, TILE), np.uint16),
                         metadata={"axes": "ZYX"}, photometric="minisblack")
    assert plan_mosaic(sorted(root.iterdir())) is None


def test_tiles_of_two_sizes_are_refused(tmp_path):
    root = tmp_path / "ragged"
    root.mkdir()
    tifffile.imwrite(str(root / "tile01.tif"), np.zeros((TILE, TILE), np.uint16))
    tifffile.imwrite(str(root / "tile02.tif"),
                     np.zeros((TILE, TILE // 2), np.uint16))
    assert plan_mosaic(sorted(root.iterdir())) is None


def test_a_file_that_cannot_be_read_is_refused(tmp_path):
    root = tmp_path / "broken"
    root.mkdir()
    tifffile.imwrite(str(root / "tile01.tif"), np.zeros((TILE, TILE), np.uint16))
    (root / "tile02.tif").write_bytes(b"not a tiff")
    assert plan_mosaic(sorted(root.iterdir())) is None


# ---------------------------------------------------------------------------
# The best evidence: the file's own stage coordinates
# ---------------------------------------------------------------------------

def _ome_tile(path, data, x_um, y_um, um_per_px):
    tifffile.imwrite(str(path), data, ome=True, photometric="minisblack",
                     metadata={"PhysicalSizeX": um_per_px,
                               "PhysicalSizeY": um_per_px,
                               "Plane": {"PositionX": x_um,
                                         "PositionY": y_um}})


def test_stage_coordinates_place_tiles_that_correlate_on_nothing(tmp_path):
    """The one case blank tiles can still be placed exactly: the microscope
    wrote down where it was."""
    root = tmp_path / "ome"
    root.mkdir()
    um_per_px = 0.5
    paths = []
    for index, (row, col) in enumerate(ROW_MAJOR, start=1):
        path = root / f"tile{index:02d}.ome.tif"
        _ome_tile(path, np.zeros((TILE, TILE), np.uint16),
                  x_um=col * STEP * um_per_px, y_um=row * STEP * um_per_px,
                  um_per_px=um_per_px)
        paths.append(path)

    positions = read_stage_positions(paths)
    if positions is None:
        pytest.skip("this tifffile does not round-trip Plane positions")
    mosaic = plan_mosaic(paths)
    assert mosaic.how == "stage"
    assert (mosaic.rows, mosaic.cols) == (2, 2)
    assert mosaic.overlap == (OVERLAP, OVERLAP)
    assert (mosaic.height, mosaic.width) == (STEP + TILE, STEP + TILE)


def test_a_plain_tiff_has_no_stage_coordinates(tmp_path):
    path = tmp_path / "plain.tif"
    tifffile.imwrite(str(path), np.zeros((TILE, TILE), np.uint16))
    assert read_stage_positions([path]) is None
    assert read_stage_positions([tmp_path / "missing.tif"]) is None


def test_mosaic_descriptions_name_the_evidence_that_placed_the_tiles():
    placement = (Placement(1, 0, 0, 0, 0),)
    stage = Mosaic(placement, 8, 8, 1, 1, "row_major", "stage")
    correlated = Mosaic(placement, 8, 8, 1, 1, "row_major", "correlated",
                        confidence=0.875, overlap=(2, 3))

    assert "stage coordinates" in stage.describe()
    assert "overlapping 3x2 px" in correlated.describe()
    assert "confidence 0.88" in correlated.describe()


def test_stage_positions_fall_back_when_tifffile_is_unavailable(monkeypatch):
    real_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "tifffile":
            raise ImportError("optional reader is absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    assert read_stage_positions(["unused.tif"]) is None


def test_incomplete_ome_position_metadata_is_not_a_pixel_position():
    incomplete = SimpleNamespace(
        ome_metadata=('PositionX="10" PositionY="20" '
                      'PhysicalSizeX="0.5"'))
    zero_scale = SimpleNamespace(
        ome_metadata=('PositionX="10" PositionY="20" '
                      'PhysicalSizeX="0" PhysicalSizeY="0.5"'))
    complete = SimpleNamespace(
        ome_metadata=('PositionX="10" PositionY="20" '
                      'PhysicalSizeX="0.5" PhysicalSizeY="0.5"'))

    assert image_stitch._position_from(incomplete) is None
    assert image_stitch._position_from(zero_scale) is None
    assert image_stitch._position_from(complete) == (40.0, 20.0)


def test_too_thin_and_too_small_overlaps_have_zero_confidence():
    small = np.arange(16, dtype=float).reshape(4, 4)
    large = np.arange(10_000, dtype=float).reshape(100, 100)

    assert image_stitch._score_overlap(small, small, 3, 0) == 0.0
    assert image_stitch._score_overlap(large, large, 98, 98) == 0.0
    assert image_stitch._score_overlap(large, large, 0, 0) == pytest.approx(1.0)


def test_overlap_constants_always_make_a_nonempty_search_band():
    """Pin the premise that makes two old defensive guards unreachable."""
    minimum = image_stitch.MIN_OVERLAP_FRACTION
    maximum = image_stitch.MAX_OVERLAP_FRACTION
    assert 0 <= minimum < maximum < 1
    for span in range(1, 10_001):
        low = int(span * (1 - maximum))
        high = int(span * (1 - minimum))
        assert 0 <= low <= high < span


def test_pair_search_skips_offsets_with_fewer_than_four_lines(monkeypatch):
    tile = np.arange(64, dtype=float).reshape(4, 16)
    real_sliding_ncc = image_stitch._sliding_ncc
    measured = []

    def measure_only_useful_bands(first, second):
        assert first.shape[0] >= 4
        assert second.shape[0] >= 4
        measured.append(first.shape)
        return real_sliding_ncc(first, second)

    monkeypatch.setattr(image_stitch, "_sliding_ncc", measure_only_useful_bands)

    shift, score = image_stitch._pair_shift(tile, tile, "x")

    assert measured
    assert len(shift) == 2
    assert np.isfinite(score)


def test_one_unreadable_tile_is_not_a_single_tile_mosaic(tmp_path):
    unreadable = tmp_path / "not-a-tiff.tif"
    unreadable.write_bytes(b"broken")

    assert plan_mosaic([unreadable]) is None


def test_one_stage_stop_has_no_measurable_spacing():
    assert image_stitch._spacing([]) == 0
    assert image_stitch._spacing([17]) == 0
    assert image_stitch._spacing([7, 17]) == 10


def test_every_declared_arrangement_maps_a_grid_once_and_in_bounds():
    expected = {(row, col) for row in range(3) for col in range(4)}
    for arrangement in image_stitch.ARRANGEMENTS:
        actual = {
            arrangement_of(index, 3, 4, arrangement)
            for index in range(12)
        }
        assert actual == expected


def test_private_mosaic_planner_reports_an_unknown_arrangement():
    tile = np.zeros((8, 8), dtype=np.uint16)
    with pytest.raises(ValueError, match="unknown arrangement"):
        image_stitch._mosaic_by_correlation(
            [tile], [1], 1, 1, "spiral", 8, 8)


def test_a_reverse_believed_edge_can_complete_the_grid(monkeypatch):
    """A tile reachable around one side may recover its missing neighbour."""
    replies = iter([
        ((0, 0), 0.0),  # origin -> top-right is not believed
        ((4, 0), 1.0),  # origin -> bottom-left
        ((4, 0), 1.0),  # top-right -> bottom-right
        ((0, 4), 1.0),  # bottom-left -> bottom-right
    ])
    monkeypatch.setattr(image_stitch, "_pair_shift",
                        lambda _first, _second, _axis: next(replies))
    images = [np.zeros((8, 8), dtype=np.uint16) for _ in range(4)]

    mosaic = image_stitch._mosaic_by_correlation(
        images, [1, 2, 3, 4], 2, 2, "row_major", 8, 8)

    positions = {p.tile: (p.y, p.x) for p in mosaic.placements}
    assert positions == {1: (0, 0), 2: (0, 4), 3: (4, 0), 4: (4, 4)}


def test_a_disappearing_plan_violates_the_readable_tiles_invariant(
        tmp_path, monkeypatch):
    path = tmp_path / "tile.tif"
    tifffile.imwrite(path, np.zeros((8, 8), dtype=np.uint16))
    monkeypatch.setattr(image_stitch, "plan_mosaic",
                        lambda _paths, _tiles=None: None)

    with pytest.raises(AssertionError):
        stitch_tiles([path])


def test_float_tiles_remain_float_without_integer_rounding(tmp_path):
    path = tmp_path / "float.tif"
    tile = np.full((8, 8), 0.375, dtype=np.float32)
    tifffile.imwrite(path, tile)

    stitched, mosaic = stitch_tiles([path])

    assert mosaic.how == "single"
    assert stitched.dtype == np.float32
    assert np.array_equal(stitched, tile)


# ---------------------------------------------------------------------------
# What a stage actually does
# ---------------------------------------------------------------------------

def test_a_stage_that_does_not_step_exactly_is_followed(tmp_path):
    """Real stages miss the row by a few pixels. A uniform grid pitch puts
    that error into the seam, where it doubles a structure or drops a strip;
    each tile is placed from its own measured displacement instead."""
    canvas = _smooth_field((TILE + STEP + 8, TILE + STEP + 8), seed=3)
    root = tmp_path / "jitter"
    root.mkdir()
    #: (y, x) of each tile: the right-hand column sits 3 px low, the bottom
    #: row 2 px right — the sort of miss a stage makes every field.
    corners = [(0, 0), (3, STEP), (STEP, 2), (STEP + 3, STEP + 2)]
    paths = []
    for index, (y, x) in enumerate(corners, start=1):
        path = root / f"tile{index:02d}.tif"
        tifffile.imwrite(str(path), canvas[y:y + TILE, x:x + TILE])
        paths.append(path)

    mosaic = plan_mosaic(paths)
    assert mosaic.how == "correlated"
    placed = {p.tile: (p.y, p.x) for p in mosaic.placements}
    assert placed == {1: (0, 0), 2: (3, STEP), 3: (STEP, 2),
                      4: (STEP + 3, STEP + 2)}

    stitched, _mosaic = stitch_tiles(paths, mosaic=mosaic)
    # A JITTERED MOSAIC IS NOT A RECTANGLE. Its bounding box has corners no
    # tile covered, and those stay zero -- so the check is that every pixel
    # a tile DID cover is the pixel that was there, which is the claim the
    # stitch actually makes.
    covered = np.zeros(stitched.shape, bool)
    for placement in mosaic.placements:
        covered[placement.y:placement.y + TILE,
                placement.x:placement.x + TILE] = True
    expected = canvas[:mosaic.height, :mosaic.width]
    assert np.array_equal(stitched[covered], expected[covered])
    assert covered.mean() > 0.9


def test_a_full_size_tile_is_placed_from_a_sample_of_its_rows(tmp_path):
    """A 512 px tile has four times the rows of the fixtures above and must
    not cost four times as much to place: the search reads a sample, and the
    winner is then scored on the whole overlap."""
    import time

    tile, overlap = 512, 48
    step = tile - overlap
    rng = np.random.default_rng(5)
    field = rng.integers(0, 4000, size=(step + tile, step + tile),
                         dtype=np.uint16)
    root = tmp_path / "big"
    root.mkdir()
    paths = []
    for index, (row, col) in enumerate(ROW_MAJOR, start=1):
        path = root / f"tile{index:02d}.tif"
        tifffile.imwrite(str(path),
                         field[row * step:row * step + tile,
                               col * step:col * step + tile])
        paths.append(path)

    started = time.perf_counter()
    stitched, mosaic = stitch_tiles(paths)
    elapsed = time.perf_counter() - started

    assert mosaic.overlap == (overlap, overlap)
    assert np.array_equal(stitched, field)
    # Generous: the point is that it is not quadratic in the tile edge, and
    # a field of four 512 px tiles takes well under a second when it is not.
    assert elapsed < 20, f"placing four 512 px tiles took {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# End to end, through the import
# ---------------------------------------------------------------------------

def test_a_tiled_acquisition_imports_as_whole_fields(tmp_path):
    """The whole point, measured end to end: tiles in, fields out, and the
    pixels are the pixels. The corpus's tiled tree proves the plumbing; this
    proves the picture, because blank tiles cannot."""
    from spacr.image_import import apply_import, plan_import

    field = _smooth_field((STEP + TILE, STEP + TILE), seed=42)
    root = tmp_path / "acquisition"
    root.mkdir()
    for channel in (1, 2):
        for index, (row, col) in enumerate(ROW_MAJOR, start=1):
            name = f"plate1_A01_F001_tile{index:02d}_C{channel}.tif"
            tifffile.imwrite(
                str(root / name),
                (field[row * STEP:row * STEP + TILE,
                       col * STEP:col * STEP + TILE] + channel).astype(
                           np.uint16))

    plan = plan_import(root)
    assert plan.counts().get("tile") == 4
    assert not plan.problems()

    result = apply_import(plan, tmp_path / "project")
    assert result.written == 2, "one image per channel, not one per tile"
    assert result.stitched == 2
    assert not result.skipped
    assert not result.unverified, "a real field should correlate"

    for channel in (1, 2):
        name = f"plate1_A01_T0001F001L01A01Z01C{channel:02d}.tif"
        written = tifffile.imread(str(result.destination / name))
        assert written.shape == field.shape
        assert np.array_equal(written, (field + channel).astype(np.uint16))
