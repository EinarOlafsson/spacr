"""spacr.align: what the filename decides, what the solve weighs, what is recorded.

* **The name is the key.**  :func:`~spacr.align.scan_tiles` decides from a
  filename alone which files are one stage position.  Getting it wrong drops
  tiles or composites two wells into one canvas as if they were channels.
* **Scan-time choices reach the registration**: the reference channel picked
  at scan time, and a reader cache at its documented minimum.
* **The least-squares solve** treats ``confidence`` as an edge weight, spreads
  an inconsistency over every tile's residual, and its default anchor never
  drags a registered chain back to the stage positions.
* **The coordinates table** keys every tile the way the rest of spaCR does,
  with or without a written canvas and with or without the per-well split.

Everything is deterministic (fixed seeds), CPU-only, offline, and no tile is
larger than 128x128.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from spacr import align
from spacr.errors import ConfigurationError


def _texture(height, width, seed=0, sigma=2.0):
    """A smooth, non-repeating uint16 field: registrable, unlike noise."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    smooth = gaussian_filter(rng.random((height, width)).astype(np.float32), sigma)
    smooth -= smooth.min()
    smooth /= max(float(smooth.max()), 1e-9)
    return (smooth * 30000 + 1000).astype(np.uint16)


TILE = 128
STEP = 80
OVERLAP = 1 - STEP / TILE


def _write_2x2(folder):
    """Cut one texture into a 2x2 grid of ``TILE`` tiles stepped ``STEP``."""
    os.makedirs(folder, exist_ok=True)
    big = _texture(TILE + STEP + 8, TILE + STEP + 8, seed=7)
    for k, (row, col) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        np.save(os.path.join(folder, f"plate1_B07_{k + 1:03d}.npy"),
                big[row * STEP:row * STEP + TILE, col * STEP:col * STEP + TILE])


def _yokogawa_1536(folder):
    for field in range(1, 5):
        np.save(folder / f"plate1_AA01_T0001F{field:03d}L01A01Z01C01.npy",
                _texture(32, 32, seed=field))
    return {"grid": (2, 2), "overlap": 0.25}


def _underscore_plate_two_wells(folder):
    for well, value in (("B07", 111), ("C08", 999)):
        for field in (1, 2):
            np.save(folder / f"exp1_plate1_{well}_{field:03d}.npy",
                    np.full((32, 32), value, np.uint16))
    return {"grid": (2, 2), "overlap": 0.25}


NAMING = [
    ("a_1536_well_keeps_one_tile_per_field", _yokogawa_1536,
     [("AA01", 1, ()), ("AA01", 2, ()), ("AA01", 3, ()), ("AA01", 4, ())], 1),
    ("two_wells_of_an_underscore_plate_stay_two_sites", _underscore_plate_two_wells,
     [("B07", 1, ()), ("B07", 2, ()), ("C08", 1, ()), ("C08", 2, ())], 2),
]


def test_scan_tiles_keys_one_tile_per_stage_position(tmp_path, subtests):
    """Every field of every well is its own tile, never a channel of another well's site."""
    for case_id, write, expected, n_groups in NAMING:
        with subtests.test(msg=case_id):
            folder = tmp_path / case_id
            folder.mkdir()
            kwargs = write(folder)
            tiles = align.scan_tiles(str(folder), **kwargs)
            assert sorted((t.well, t.field, t.channel_paths) for t in tiles) == expected
            for tile in tiles:
                sources = tile.channel_paths or (tile.path,)
                assert all(f"_{tile.well}_" in os.path.basename(p) for p in sources)
            assert len(align.group_tiles(tiles)) == n_groups


def test_registration_honours_what_the_scan_chose(tmp_path, subtests):
    """The scan-time reference channel drives registration, and a one-tile cache never unregisters a plate."""
    with subtests.test(msg="reference_channel_from_the_scan"):
        folder = tmp_path / "refchan"
        folder.mkdir()
        big = _texture(64, 200, seed=15)
        for k, x0 in enumerate((0, 40)):
            np.save(folder / f"plate1_B07_T0001F{k + 1:03d}L01A01Z01C01.npy",
                    big[:, x0:x0 + 100])
            np.save(folder / f"plate1_B07_T0001F{k + 1:03d}L01A01Z01C02.npy",
                    np.full((64, 100), 777, np.uint16))
        tiles = align.scan_tiles(str(folder), grid=(1, 2), overlap=0.5,
                                 reference_channel=2)
        assert [t.channel for t in tiles] == [1, 1]
        plan = align.estimate_offsets(tiles)
        # Channel 2 is a flat constant, so honouring the choice means nothing registers.
        assert (plan.n_nominal, plan.n_registered) == (2, 0)
        assert "blank" in plan.placements[0].note
        assert [p.method for p in plan.placements] == [align.METHOD_NOMINAL] * 2

    with subtests.test(msg="a_one_tile_reader_cache"):
        folder = tmp_path / "tiny_cache"
        _write_2x2(str(folder))
        tiles = align.scan_tiles(str(folder), grid=(2, 2), overlap=OVERLAP)
        roomy = align.estimate_offsets(tiles)
        try:
            plan = align.estimate_offsets(tiles, max_open_tiles=1)
        except ConfigurationError:
            plan = None  # refusing the value outright is a legitimate answer
        if plan is not None:
            assert (plan.n_registered, plan.n_nominal) == (4, 0)
            assert [p.note for p in plan.overlaps if "AttributeError" in p.note] == []
            assert [(p.y, p.x) for p in plan.placements] == \
                [(p.y, p.x) for p in roomy.placements]


NOMINAL = np.array([[0.0, 0.0], [0.0, 100.0], [0.0, 200.0]])


def _span(direct_w, chain_w):
    edges = [(0, 1, 0.0, 100.0, chain_w), (1, 2, 0.0, 100.0, chain_w),
             (0, 2, 0.0, 225.0, direct_w)]
    positions, _residuals, _degrees = align.solve_positions(3, edges, NOMINAL)
    return float(positions[2, 1] - positions[0, 1])


def test_solve_positions_weighs_edges_spreads_residuals_and_keeps_the_gauge_light(subtests):
    """Confidence is a weight, an inconsistent triangle leaves 25/3 px on every tile, the anchor only fixes the gauge."""
    with subtests.test(msg="confidence_is_the_edge_weight"):
        # 0-1 and 1-2 say "100 apart", 0-2 says "225": the trusted side wins.
        assert _span(0.5, 0.5) == pytest.approx(216.667, abs=0.01)
        assert _span(0.9, 0.1) == pytest.approx(224.85, abs=0.05)
        assert _span(0.1, 0.9) == pytest.approx(200.60, abs=0.05)

    for case_id, far, residual in [("inconsistent_triangle", 225.0, 25 / 3),
                                   ("consistent_triangle", 200.0, 0.0)]:
        with subtests.test(msg=case_id):
            edges = [(0, 1, 0.0, 100.0, 0.5), (1, 2, 0.0, 100.0, 0.5),
                     (0, 2, 0.0, far, 0.5)]
            _positions, residuals, degrees = align.solve_positions(3, edges, NOMINAL)
            assert list(degrees) == [2, 2, 2]
            assert residuals == pytest.approx([residual] * 3, abs=0.01)

    for case_id, weight, step in [("default_anchor_keeps_the_measured_step",
                                   align.DEFAULT_ANCHOR_WEIGHT, 90.0),
                                  ("a_heavy_anchor_would_drag_it_to_nominal", 10.0, 99.9)]:
        with subtests.test(msg=case_id):
            edges = [(0, 1, 0.0, 90.0, 0.8), (1, 2, 0.0, 90.0, 0.8)]
            positions, _r, _d = align.solve_positions(3, edges, NOMINAL,
                                                      anchor_weight=weight)
            steps = np.diff(positions[:, 1])
            assert steps == pytest.approx([step, step], abs=0.05)
            assert positions[:, 0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def _single_1536_tile(tmp_path):
    tile = align.Tile(path=str(tmp_path / "x.npy"), index=0, plate="plate1",
                      well="AA01", field=3, shape=(16, 16, 1))
    plan = align.AlignPlan(
        tiles=[tile],
        placements=[align.Placement(tile=tile, y=0.0, x=0.0, confidence=1.0,
                                    method=align.METHOD_SINGLE)],
        dtype="uint16")
    db_path = tmp_path / "m.db"
    assert align.save_coordinates(plan, db_path) == 1
    return db_path


def test_read_coordinates_keys_every_tile_the_way_spacr_does(tmp_path, subtests):
    """Coordinates rows carry spaCR's row/column/field keys, with or without a canvas or a per-well split."""
    with subtests.test(msg="a_1536_well_is_r27_c1"):
        db_path = _single_1536_tile(tmp_path / "k")
        frame = align.read_coordinates(db_path)
        assert frame[["rowID", "columnID", "fieldID", "prcf"]].values.tolist() == [
            ["r27", "c1", "f3", "plate1_r27_c1_f3"]]
        assert len(align.read_coordinates(db_path, well="AA01")) == 1
        assert len(align.read_coordinates(db_path, well="AA02")) == 0

    with subtests.test(msg="save_stack_false_records_without_writing_pixels"):
        root = tmp_path / "plan_only"
        src = root / "tiles"
        src.mkdir(parents=True)
        big = _texture(64, 160, seed=17)
        np.save(src / "plate1_B07_001.npy", big[:, 0:100])
        np.save(src / "plate1_B07_002.npy", big[:, 60:160])
        db_path = root / "m.db"
        results = align.align_folder(src=str(src), dst=str(root / "out"),
                                     grid=(1, 2), overlap=0.40,
                                     save_stack=False, db_path=str(db_path))
        assert len(results) == 1
        result = results[0]
        assert (result.stack_path, result.n_written) == ("", 0)
        assert result.canvas.shape == (64, 160, 1)
        assert result.canvas.nbytes == 64 * 160 * 2
        assert result.band_rows >= 1
        assert [p for p in root.rglob("*.npy") if p.parent != src] == []
        assert not (root / "out").exists()
        frame = align.read_coordinates(db_path)
        assert frame[["stack_path", "method", "canvas_width"]].values.tolist() == [
            ["", align.METHOD_REGISTRATION, 160]] * 2

    with subtests.test(msg="group_by_well_false_puts_every_well_on_one_canvas"):
        root = tmp_path / "one_canvas"
        src = root / "tiles"
        src.mkdir(parents=True)
        big = _texture(64, 160, seed=23)
        for well, x0 in (("B07", 0), ("C08", 60)):
            np.save(src / f"plate1_{well}_001.npy", big[:32, x0:x0 + 100])
            np.save(src / f"plate1_{well}_002.npy", big[32:, x0:x0 + 100])
        db_path = root / "m.db"
        results = align.align_folder(src=str(src), dst=str(root / "out"),
                                     grid=(2, 2), overlap=0.25,
                                     group_by_well=False, db_path=str(db_path))
        assert len(results) == 1
        result = results[0]
        assert result.n_written == 4
        assert os.path.basename(result.stack_path) == "tiles_stitched.npy"
        assert np.load(result.stack_path).shape == result.canvas.shape
        frame = align.read_coordinates(db_path)
        assert sorted(zip(frame["well"], frame["rowID"], frame["columnID"])) == [
            ("B07", "r2", "c7"), ("B07", "r2", "c7"),
            ("C08", "r3", "c8"), ("C08", "r3", "c8")]
        assert set(frame["stack_path"]) == {result.stack_path}
