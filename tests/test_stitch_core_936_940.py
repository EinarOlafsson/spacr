"""Items 936 and 940 and the S1 stitch speed-ups.

936: Align & Stitch and the OPS stitch share one registration, solve and
composition core (``spacr.stitch_core``); the legacy Align & Stitch engine is
unchanged, and the core engine with OPS settings gives the OPS edges and
placements. The speed-ups (float32 strip registration, parallel edges,
read-ahead of the tiles) leave the integer shifts, the acceptances and the
placements as they were. 940: a placement file written after a full stitch
lets a run on a subset of the well place its tiles without solving, and its
windows equal the full run's.
"""
from __future__ import annotations

import ast
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from spacr import align, ops_compose, ops_cycles, ops_engine, ops_register, ops_solve, ops_stitch
from spacr import stitch_core as core
from spacr.ops_layout import round_well_layout

ROOT = Path(__file__).resolve().parents[1]
TILE = 384
OVERLAP = int(TILE * 0.14)
STEP = TILE - OVERLAP
SKEW = 3


def _toy_tiles(sites=21, seed=7):
    """A round well of blob tiles cut from one canvas, with a small stage skew.

    :returns: ``(layout, {site: tile}, {site: (y, x)})`` -- the true
        positions, so a solve can be checked against them.
    """
    from scipy.ndimage import gaussian_filter

    layout = round_well_layout(sites)
    rng = np.random.default_rng(seed)
    rows, columns = max(layout.heights), layout.columns
    first_row = min(row for _column, row in layout.positions())
    height = STEP * (rows + 2) + TILE + 80
    width = STEP * columns + TILE + 80
    dots = np.zeros((height, width))
    count = (height * width) // 100
    dots[rng.integers(0, height, count), rng.integers(0, width, count)] = 1.0
    field = gaussian_filter(dots, 1.5) * (3000 * 2 * math.pi * 2.25) + 40.0
    tiles, truth = {}, {}
    for site in range(layout.site_count):
        column, row = layout.position(site)
        top = (row - first_row) * STEP + 20 + column * SKEW
        left = column * STEP + 20
        tiles[site] = rng.poisson(field[top:top + TILE, left:left + TILE]
                                  ).astype(np.float32)
        truth[site] = (float(top), float(left))
    return layout, tiles, truth


@pytest.fixture(scope="module")
def toy():
    """One toy well for the module."""
    return _toy_tiles()


def _edges(well):
    """``(a, b) -> (dy, dx, accepted)`` of a stitched well."""
    return {pair: (one.dy, one.dx, one.accepted) for pair, one in well.edges.items()}


def test_the_core_hands_out_the_ops_objects_themselves():
    """936 test 8: one implementation, re-exported, never copied."""
    pairs = {
        "phase_correlate": ops_register, "unwrap": ops_register,
        "available_backends": ops_register, "register_edge": ops_register,
        "register_pairs": ops_register, "Registration": ops_register,
        "_read_peak": ops_register, "solve_placements": ops_solve,
        "compose_window": ops_compose, "windows_over": ops_compose,
        "Window": ops_compose, "ComposeError": ops_compose,
        "overlap_gain": ops_compose, "stitch_well": ops_stitch,
        "StitchedWell": ops_stitch,
    }
    for name, module in pairs.items():
        assert getattr(core, name) is getattr(module, name), name
    assert core.FFTEngine is ops_cycles._FFTEngine
    assert align.feather_weights is core.feather_weights
    assert align._ramp is core._ramp


def test_no_second_registration_solve_or_composer_is_defined():
    """936 test 8: a grep ratchet over ``spacr/``."""
    allowed = {
        "phase_correlate": {"spacr/ops_register.py"},
        "compose_window": {"spacr/ops_compose.py"},
        "solve_positions": {"spacr/stitch_core.py", "spacr/align.py"},
        "feather_weights": {"spacr/stitch_core.py"},
        "register_strip_pair": {"spacr/stitch_core.py"},
    }
    found = {name: set() for name in allowed}
    for path in (ROOT / "spacr").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name in found:
                found[node.name].add(path.relative_to(ROOT).as_posix())
    assert found == allowed
    source = (ROOT / "spacr/align.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    solve = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "solve_positions")
    assert "np.linalg.lstsq" not in ast.get_source_segment(source, solve), (
        "align.solve_positions is an adapter over the core solve")


def test_float32_strips_give_the_float64_shifts_and_placements(toy):
    """The speed card's identity gate on a toy well: integers and placements equal."""
    layout, tiles, _truth = toy
    wells = {precision: ops_stitch.stitch_well(
        tiles, layout, overlap=OVERLAP, tolerance=4, gpu=False,
        precision=precision) for precision in ("float64", "float32")}
    assert _edges(wells["float32"]) == _edges(wells["float64"])
    assert wells["float64"].accepted == wells["float64"].proposed
    assert wells["float32"].placements == wells["float64"].placements
    ratios = [(wells["float32"].edges[p].peak_ratio, wells["float64"].edges[p].peak_ratio)
              for p in wells["float64"].edges]
    assert max(abs(a - b) / b for a, b in ratios) < 1e-3
    assert {one.backend for one in wells["float32"].edges.values()} == {"scipy"}
    assert {one.backend for one in wells["float64"].edges.values()} == {"numpy"}


def test_float64_is_the_arithmetic_before_936(toy):
    """'float64' keeps the old surface bit for bit (the default of phase_correlate)."""
    layout, tiles, _truth = toy
    a, b, axis = layout.pairs()[0]
    band = ops_register._edge_band(TILE, ops_register.STRIP_FRACTION)
    first, second = tiles[a][-band:, :].astype(np.float64), tiles[b][:band, :].astype(np.float64)
    a = np.fft.rfft2(first)
    b = np.fft.rfft2(second)
    spectrum = a * np.conj(b)
    magnitude = np.abs(spectrum)
    with np.errstate(invalid="ignore", divide="ignore"):
        spectrum = np.where(magnitude > 0, spectrum / magnitude, 0)
    spectrum[0, 0] = 0
    expected = np.fft.irfft2(spectrum, s=first.shape)
    assert np.array_equal(ops_register._surface_numpy(first, second), expected)
    found = ops_register.phase_correlate(first, second, gpu=False,
                                         expected=(band - OVERLAP, 0))
    flat = int(np.argmax(expected))
    assert found.peak_ratio == float(expected.flat[flat]) / float(np.abs(expected).mean())


def test_an_unknown_precision_is_refused():
    with pytest.raises(ValueError, match="precision must be"):
        ops_register.phase_correlate(np.ones((8, 8)), np.ones((8, 8)),
                                     gpu=False, precision="float16")


def test_parallel_edges_give_the_serial_edges_and_placements(toy):
    """Speed card candidate 4: two worker processes, the same well."""
    layout, tiles, _truth = toy
    serial = ops_stitch.stitch_well(tiles, layout, overlap=OVERLAP, tolerance=4,
                                    gpu=False, precision="float32")
    parallel = ops_stitch.stitch_well(tiles, layout, overlap=OVERLAP,
                                      tolerance=4, gpu=False,
                                      precision="float32", edge_workers=2)
    assert list(parallel.edges) == list(serial.edges)
    assert _edges(parallel) == _edges(serial)
    assert parallel.placements == serial.placements


def test_the_read_ahead_returns_what_one_reader_returns(toy):
    layout, tiles, _truth = toy
    calls = []

    def read(site):
        calls.append(site)
        return tiles[site]

    order = ops_engine._first_use_order(layout, sorted(tiles))
    reader = ops_engine._TileReadAhead(read, order, threads=3, keep=4)
    try:
        asked = [0, 0] + [s for pair in layout.pairs() for s in pair[:2]] + [5, 0]
        for site in asked:
            assert reader.get(site) is tiles[site]
    finally:
        reader.close()
    assert set(calls) == set(tiles)


def _write_toy_well(folder, tiles):
    """Write the toy tiles as single-channel TIFFs and index them like the engine."""
    import tifffile

    files = {}
    for site, tile in tiles.items():
        path = os.path.join(folder, f"10X_c1_A1_DAPI_Site-{site}.tif")
        tifffile.imwrite(path, np.clip(tile, 0, 65535).astype(np.uint16))
        files[site] = {"DAPI": path}
    return {1: files}


def _geometry(db):
    from spacr.ops_store import read_table

    return read_table(db, "ops_geometry").sort_values("site").reset_index(drop=True)


@pytest.fixture(scope="module")
def engine_well(toy, tmp_path_factory):
    """The toy well stitched by the engine with one reader and with four."""
    _layout, tiles, _truth = toy
    folder = tmp_path_factory.mktemp("toy_well")
    cycle_files = _write_toy_well(str(folder), tiles)
    settings = {"ops_raster_overlap": OVERLAP}
    out = {}
    for threads in (1, 4):
        db = str(folder / f"run_{threads}" / "measurements.db")
        os.makedirs(os.path.dirname(db))
        report = ops_engine._stitch(db, "P1", "A1", cycle_files, 1,
                                    dict(settings, ops_stitch_read_threads=threads),
                                    gpu=False)
        out[threads] = (db, report)
    return cycle_files, out


def test_reader_threads_leave_ops_geometry_bit_identical(engine_well):
    """Speed card candidate 6: ops_geometry equal byte for byte."""
    _files, runs = engine_well
    one, four = _geometry(runs[1][0]), _geometry(runs[4][0])
    assert one.equals(four)
    assert runs[4][1]["read_threads"] == 4 and runs[1][1]["read_threads"] == 1
    assert runs[4][1]["edges_accepted"] == runs[4][1]["edges_proposed"]
    assert runs[4][1]["placement"] == "solved"
    assert runs[4][1]["registration_precision"] == "float32"


def test_a_full_stitch_writes_its_placement_file(engine_well):
    """940: export on by default; the file holds the ops_geometry rows exactly."""
    _files, runs = engine_well
    db, report = runs[4]
    path = report["placement_seed_file"]
    assert path == os.path.join(os.path.dirname(db), "A1", "P1_A1.ops_placement.json")
    seed = core.load_placement_seed(path)
    geometry = _geometry(db)
    assert core.seed_to_placements(seed) == {
        int(r.site): (float(r.y), float(r.x)) for r in geometry.itertuples()}
    assert seed["origin"] == [float(geometry["origin_y"][0]), float(geometry["origin_x"][0])]
    assert seed["dtype"] == "uint16" and seed["n_tiles_in_well"] == 21
    assert seed["acquisition"] == "sbs" and seed["cycle"] == 1
    assert seed["canvas"] == list(ops_engine._compose_canvas(
        core.seed_to_placements(seed), (TILE, TILE)))


def test_a_seeded_subset_places_its_tiles_where_the_full_run_did(engine_well, tmp_path):
    """940 test 2: the subset's ops_geometry rows are the full run's, same frame."""
    cycle_files, runs = engine_well
    full_db, report = runs[4]
    keep = [3, 4, 5, 6, 10, 11, 12, 20]
    subset = {1: {site: cycle_files[1][site] for site in keep}}
    db = str(tmp_path / "measurements.db")
    seeded = ops_engine._stitch(db, "P1", "A1", subset, 1, {
        "ops_raster_overlap": OVERLAP, "ops_placement_seed": report["placement_seed_file"],
    }, gpu=False)
    assert seeded["placement"] == "seeded"
    assert seeded["edges_proposed"] is None and seeded["edges_accepted"] is None
    assert seeded["placement_seed"]["tiles_used"] == len(keep)
    assert seeded["placement_seed"]["tiles_in_file"] == 21
    full = _geometry(full_db).set_index("site")
    sub = _geometry(db).set_index("site")
    assert list(sub.index) == keep
    for column in ("y", "x", "origin_y", "origin_x", "tile_height", "tile_width"):
        assert (sub[column] == full.loc[keep, column]).all(), column


def test_seeded_windows_equal_the_full_run_crop(engine_well, toy):
    """940 test 2: every window covered only by listed tiles has the full run's pixels."""
    _layout, tiles, _truth = toy
    _files, runs = engine_well
    seed = core.load_placement_seed(runs[4][1]["placement_seed_file"])
    every = core.seed_to_placements(seed)
    keep = {site: every[site] for site in (3, 4, 5, 6, 10, 11, 12, 20)}
    canvas = tuple(seed["canvas"])
    windows = list(core.windows_over(canvas, size=128, overlap=16))
    partial = {tuple(w) for w in ops_engine._partial_windows(
        windows, keep, {s: p for s, p in every.items() if s not in keep}, (TILE, TILE))}
    clean = 0
    for window in windows:
        if (window.top, window.left) in partial:
            continue
        mine, count = core.compose_window(window, keep, tiles.__getitem__,
                                          tile_shape=(TILE, TILE))
        if not count.any():
            continue
        theirs, _ = core.compose_window(window, every, tiles.__getitem__,
                                        tile_shape=(TILE, TILE))
        assert np.array_equal(mine, theirs), (window.top, window.left)
        clean += 1
    assert clean > 3 and partial


def _seed_file(tmp_path, **changes):
    """A small placement file with ``changes`` applied to its arguments."""
    arguments = dict(plate="P1", well="A1", placements={0: (0.0, 0.0), 1: (217.0, 3.0),
                                                        2: (434.0, 6.5)},
                     tile_shape=(256, 256), origin=(10.0, 20.0), canvas=(691, 263),
                     acquisition="sbs", cycle=1, dtype="uint16", n_tiles_in_well=3,
                     raster_overlap=35,
                     tile_files={0: ("a.tif", 10), 1: ("b.tif", 11), 2: ("c.tif", 12)},
                     created="2026-11-13T00:00:00+00:00")
    arguments.update(changes)
    return core.write_placement_seed(str(tmp_path / "P1_A1.ops_placement.json"), **arguments)


def test_the_file_round_trips_exactly_and_writes_the_same_bytes_twice(tmp_path):
    """940 test 4."""
    path = _seed_file(tmp_path, placements={0: (0.1 + 0.2, 1 / 3), 1: (217.0000000001, -0.0)})
    first = Path(path).read_bytes()
    seed = core.load_placement_seed(path)
    assert core.seed_to_placements(seed) == {0: (0.1 + 0.2, 1 / 3), 1: (217.0000000001, -0.0)}
    _seed_file(tmp_path, placements={0: (0.1 + 0.2, 1 / 3), 1: (217.0000000001, -0.0)})
    assert Path(path).read_bytes() == first
    assert seed["sha256"] and seed["schema_version"] == core.PLACEMENT_SCHEMA_VERSION


def test_a_version_1_file_is_read_by_this_reader():
    """940 test 4: the version-1 fixture is kept in the tests."""
    seed = core.load_placement_seed(ROOT / "tests/data/ops_placement_v1.json")
    assert seed["schema_version"] == 1
    assert core.seed_to_placements(seed) == {0: (0.0, 0.0), 1: (217.0, 3.0), 2: (434.0, 6.5)}


@pytest.mark.parametrize("change, check, message", [
    ({}, dict(plate="P2"), "is for plate P1 well A1, not plate P2 well A1"),
    ({}, dict(well="B1"), "is for plate P1 well A1, not plate P1 well B1"),
    ({}, dict(sites=[0, 1, 2, 5]), r"has no position for sites \[5\] \(1 of the 4"),
    ({}, dict(sites=[0, 1], mode="strict"), r"lists 3 tiles and this run has 2.*not in this run: \[2\] \(1\)"),
    ({}, dict(n_tiles_in_well=2, mode="strict", sites=[0, 1, 2]),
     "written for a layout of 3 tiles and this run's layout has 2"),
    ({}, dict(n_tiles_in_well=4), "written for a layout of 3 tiles and this run's layout has 4"),
    ({}, dict(tile_shape=(128, 256)), r"is for \[256, 256\] px tiles and this run's tiles are \[128, 256\]"),
    ({}, dict(dtype="uint8"), "is for uint16 tiles and this run's tiles are uint8"),
    ({}, dict(acquisition="phenotype"), "describes the sbs acquisition, not phenotype"),
    ({}, dict(cycle=2), "was solved on cycle 1, and this run's reference cycle is 2"),
    ({"acquisition": "phenotype"}, dict(acquisition="phenotype", align_model="tile_affine"),
     "holds no tile groups, and ops_align_model 'tile_affine' cannot fit them"),
    ({}, dict(tile_files={0: ("a.tif", 10), 1: ("b.tif", 99), 2: ("c.tif", 12)}, mode="strict"),
     r"the source files of sites \[1\] \(1\) changed"),
])
def test_a_seed_that_does_not_fit_the_run_is_refused(tmp_path, change, check, message):
    """940 test 5, each message checked."""
    seed = core.load_placement_seed(_seed_file(tmp_path, **change))
    arguments = dict(plate="P1", well="A1", sites=[0, 1, 2])
    arguments.update(check)
    with pytest.raises(ValueError, match=message):
        core.validate_placement_seed(seed, **arguments)


def test_changed_files_only_warn_under_subset_and_padding_is_ignored(tmp_path):
    seed = core.load_placement_seed(_seed_file(tmp_path))
    warnings = core.validate_placement_seed(
        seed, plate="P1", well="a01", sites=[1],
        tile_files={1: ("b.tif", 99)})
    assert len(warnings) == 1 and "changed since placement file" in warnings[0]


def test_a_seed_from_another_spacr_version_is_accepted_with_a_warning(tmp_path):
    path = _seed_file(tmp_path)
    content = json.loads(Path(path).read_text())
    content["provenance"]["spacr_version"] = "1.5.1.7"
    Path(path).write_text(json.dumps(content))
    warnings = core.validate_placement_seed(core.load_placement_seed(path),
                                            plate="P1", well="A1", sites=[0])
    assert any("written by spaCR 1.5.1.7" in w for w in warnings)


@pytest.mark.parametrize("content, message", [
    ('{"schema": "spacr.ops_placement", "schema_version": 2}',
     "placement file version 2 is newer than this spaCR"),
    ('{"schema": "spacr.ops_placement", "schema_ver', "is not valid JSON"),
    ('{"schema": "other", "schema_version": 1}', "is not a placement file"),
    ('{"schema": "spacr.ops_placement", "schema_version": 1}', "is missing plate, well"),
])
def test_a_newer_corrupt_or_foreign_file_is_refused(tmp_path, content, message):
    path = tmp_path / "x.ops_placement.json"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        core.load_placement_seed(path)


def test_the_pixel_check_catches_a_tile_five_pixels_off(engine_well, tmp_path):
    """940 test 6: a wrong offset raises and names the pair; a right one changes nothing."""
    cycle_files, runs = engine_well
    path = runs[4][1]["placement_seed_file"]
    content = json.loads(Path(path).read_text())
    moved = tmp_path / "moved.ops_placement.json"
    for tile in content["tiles"]:
        if tile["site"] == 1:
            tile["y"] += 5.0
    moved.write_text(json.dumps(content))
    settings = {"ops_raster_overlap": OVERLAP, "ops_placement_seed_check_pixels": True}
    with pytest.raises(ValueError, match=r"sites \d+ and \d+ of well A1 are .* apart in the file"):
        ops_engine._stitch(str(tmp_path / "m.db"), "P1", "A1", cycle_files, 1,
                           dict(settings, ops_placement_seed=str(moved)), gpu=False)
    report = ops_engine._stitch(str(tmp_path / "ok.db"), "P1", "A1", cycle_files, 1,
                                dict(settings, ops_placement_seed=path), gpu=False)
    assert report["placement_seed"]["check"]["passed"]
    assert report["placement_seed"]["check"]["pairs"]
    assert _geometry(str(tmp_path / "ok.db")).equals(_geometry(runs[4][0]))


def test_the_database_is_a_seed_too(engine_well, tmp_path):
    """940: reading straight from measurements.db stays supported."""
    cycle_files, runs = engine_well
    report = ops_engine._stitch(str(tmp_path / "m.db"), "P1", "A1", cycle_files, 1,
                                {"ops_raster_overlap": OVERLAP,
                                 "ops_placement_seed": runs[1][0]}, gpu=False)
    assert report["placement"] == "seeded"
    assert _geometry(str(tmp_path / "m.db")).equals(_geometry(runs[1][0]))


def test_old_stamped_settings_keep_the_export_off():
    """The DEFAULT_CHANGES row: a file stamped before 940's version keeps export off."""
    from spacr import settings as spacr_settings

    version = max(spacr_settings.DEFAULT_CHANGES)
    assert spacr_settings.DEFAULT_CHANGES[version] == {"ops_placement_seed_export": False}
    assert spacr_settings.SETTINGS_SCHEMA_VERSION == version
    old = {}
    applied = spacr_settings._fold_older_defaults(old, version - 1)
    assert old["ops_placement_seed_export"] is False
    assert applied["ops_placement_seed_export"] == (False, version)
    current = {}
    spacr_settings._fold_older_defaults(current, version)
    assert "ops_placement_seed_export" not in current
    from spacr.ops_settings import OPS_DEFAULTS

    assert OPS_DEFAULTS["ops_placement_seed_export"] is True
    assert OPS_DEFAULTS["ops_stitch_read_threads"] == 4
    assert OPS_DEFAULTS["ops_stitch_edge_workers"] == 1


def test_an_unseeded_well_writes_no_seed_when_export_is_off(toy, tmp_path):
    _layout, tiles, _truth = toy
    cycle_files = _write_toy_well(str(tmp_path), {s: tiles[s] for s in tiles})
    report = ops_engine._stitch(str(tmp_path / "m.db"), "P1", "A1", cycle_files, 1,
                                {"ops_raster_overlap": OVERLAP,
                                 "ops_placement_seed_export": False}, gpu=False)
    assert "placement_seed_file" not in report
    assert not (tmp_path / "A1").exists()


def _legacy_solve(n_tiles, edges, nominal, anchor_weight=1e-3):
    """align.solve_positions before item 936, verbatim, as the reference."""
    nominal = np.asarray(nominal, dtype=np.float64).reshape(n_tiles, 2)
    edge_list = [(int(i), int(j), float(dy), float(dx), float(w)) for i, j, dy, dx, w in edges]
    n_rows = len(edge_list) + n_tiles
    rows, cols, vals = [], [], []
    rhs = np.zeros((n_rows, 2), dtype=np.float64)
    for r, (i, j, dy, dx, w) in enumerate(edge_list):
        weight = max(float(w), 1e-6)
        rows.extend((r, r))
        cols.extend((j, i))
        vals.extend((weight, -weight))
        rhs[r, 0] = weight * dy
        rhs[r, 1] = weight * dx
    for k in range(n_tiles):
        r = len(edge_list) + k
        rows.append(r)
        cols.append(k)
        vals.append(float(anchor_weight))
        rhs[r] = float(anchor_weight) * nominal[k]
    design = np.zeros((n_rows, n_tiles), dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        design[r, c] += v
    solution, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    return solution


def test_the_legacy_align_solve_gives_the_same_bits():
    """936 test 1: align.solve_positions through the core is the old arithmetic."""
    rng = np.random.default_rng(3)
    n = 30
    nominal = rng.normal(size=(n, 2)) * 100
    edges = [(i, j, float(rng.normal() * 5), float(rng.normal() * 5), float(rng.random()))
             for i in range(n) for j in range(i + 1, min(n, i + 4))]
    edges.append(edges[0])
    solution, _residuals, _degrees = align.solve_positions(n, edges, nominal)
    assert np.array_equal(solution, _legacy_solve(n, edges, nominal))


def test_the_core_solve_is_the_same_bits_twice():
    """936 test 7, through the anchored gauge on one BLAS thread."""
    rng = np.random.default_rng(4)
    edges = {(i, i + 1): (float(rng.normal()), float(rng.normal())) for i in range(40)}
    anchors = {k: (float(k), 0.0) for k in range(41)}
    first = core.solve_positions(edges, list(range(41)), gauge="anchor", anchors=anchors)
    second = core.solve_positions(edges, list(range(41)), gauge="anchor", anchors=anchors)
    assert first == second
    with pytest.raises(ValueError, match="needs an anchor for every site"):
        core.solve_positions(edges, list(range(41)), gauge="anchor", anchors={0: (0, 0)})
    with pytest.raises(ValueError, match="unweighted and has no anchors"):
        core.solve_positions(edges, list(range(41)), anchors=anchors)


def _align_tiles(folder, layout, tiles):
    """The toy tiles as Align & Stitch tiles at their raster positions."""
    first_row = min(row for _column, row in layout.positions())
    found = []
    for site in range(layout.site_count):
        column, row = layout.position(site)
        path = os.path.join(folder, f"tile_{site:03d}.npy")
        np.save(path, tiles[site])
        found.append(align.Tile(path=path, index=site, plate="P1", well="A1",
                                field=site + 1, shape=(TILE, TILE, 1),
                                dtype="float32", grid_row=row - first_row,
                                grid_col=column,
                                nominal_y=float((row - first_row) * STEP),
                                nominal_x=float(column * STEP)))
    return found


OPS_EQUIVALENT = dict(align_engine="core", acceptance="peak_ratio", strip_fraction=0.30,
                      expected_overlap_px=OVERLAP, tolerance_px=4, solve_gauge="lowest_site",
                      registration_precision="float32")


def test_the_core_engine_with_ops_settings_gives_the_ops_stitch(toy, tmp_path):
    """936 test 3: the same edges, acceptances and placements as stitch_well."""
    layout, tiles, _truth = toy
    found = _align_tiles(str(tmp_path), layout, tiles)
    options = align._core_options(align.default_settings(OPS_EQUIVALENT))
    plan = align._estimate_offsets(found, upsample=1, core=options)
    ops = ops_stitch.stitch_well(tiles, layout, overlap=OVERLAP, tolerance=4, gpu=False,
                                 precision="float32")
    by_pair = {(p.i, p.j): p for p in plan.overlaps}
    assert set(by_pair) == set(ops.edges)
    for pair, one in ops.edges.items():
        mine = by_pair[pair]
        assert mine.accepted == one.accepted
        if one.accepted:
            assert (mine.dy, mine.dx) == (float(one.dy), float(one.dx))
    origin = ops.placements[0]
    for placement in plan.placements:
        y, x = ops.placements[placement.tile.index]
        assert abs((placement.y - plan.placements[0].y) - (y - origin[0])) <= 1e-9
        assert abs((placement.x - plan.placements[0].x) - (x - origin[1])) <= 1e-9


def test_core_windows_are_the_ops_composer_windows(toy, tmp_path):
    """936 test 3: the windows writer composes what compose_window composes."""
    layout, tiles, _truth = toy
    found = _align_tiles(str(tmp_path), layout, tiles)
    options = align._core_options(align.default_settings(OPS_EQUIVALENT))
    plan = align._estimate_offsets(found, upsample=1, core=options)
    result = align.write_windows(plan, str(tmp_path / "out"), size=200, overlap=20,
                                 blend="feather_distance", label="P1_A1")
    index = json.loads((Path(result.stack_path) / "windows.json").read_text())
    spec = align.plan_canvas(plan.placements, dtype=plan.dtype)
    where = {p.tile.index: (p.y - spec.origin_y, p.x - spec.origin_x) for p in plan.placements}
    assert index["windows"]
    for entry in index["windows"]:
        window = core.Window(entry["top"], entry["left"], entry["height"], entry["width"])
        expected, _ = core.compose_window(window, where, tiles.__getitem__,
                                          tile_shape=(TILE, TILE), blend="feather")
        written = np.load(Path(result.stack_path) / entry["file"])
        assert np.array_equal(written[:, :, 0], expected)


def test_the_core_engine_through_align_folder_and_its_seed(toy, tmp_path):
    """align_folder runs the core engine, writes its seed, and a seeded run reuses it."""
    layout, tiles, _truth = toy
    found = _align_tiles(str(tmp_path), layout, tiles)
    settings = align.default_settings(dict(OPS_EQUIVALENT, solve_gauge="anchor",
                                           acceptance="both", upsample=10,
                                           align_placement_seed_export=True))
    options = align._core_options(settings)
    plan = align._estimate_offsets(found, upsample=10, min_confidence=0.3, core=options)
    assert plan.n_registered == len(found)
    path = align._export_align_seed(plan, str(tmp_path / "out"), "P1_A1")
    seeded = align._estimate_offsets(found, core=dict(options, align_placement_seed=path))
    assert [(p.y, p.x) for p in seeded.placements] == [(p.y, p.x) for p in plan.placements]
    assert {p.method for p in seeded.placements} == {align.METHOD_SEEDED}


def test_subpixel_refinement_matches_the_skimage_phase_refinement():
    """The core's 1/10 px refinement is skimage's 'phase' refinement, without skimage."""
    from scipy.ndimage import gaussian_filter, shift as subshift
    from skimage.registration import phase_cross_correlation

    rng = np.random.default_rng(5)
    dots = np.zeros((400, 300))
    dots[rng.integers(0, 400, 600), rng.integers(0, 300, 600)] = 1.0
    big = gaussian_filter(dots, 2.5) * 30000.0 + 50
    first = rng.poisson(big[:200]).astype(float)
    for true in ((150.3, 1.4), (149.7, 2.2), (150.5, -0.6)):
        moved = subshift(big, (-true[0], -true[1]), order=3)[:200]
        second = rng.poisson(np.clip(moved, 0, None)).astype(float)
        found = core.register_strip_pair(first, second, "vertical", strip_fraction=0.30,
                                         expected_overlap=50, acceptance="both",
                                         upsample=10)
        assert found.accepted and found.ncc > 0.9
        reference = phase_cross_correlation(first[-60:], second[:60], upsample_factor=10,
                                            normalization="phase")[0]
        assert abs(found.dy - 140 - reference[0]) <= 0.1 + 1e-9
        assert abs(found.dx - reference[1]) <= 0.1 + 1e-9
        assert abs(found.dy - true[0]) <= 1.0 and abs(found.dx - true[1]) <= 1.0


def test_non_adjacent_pairs_are_refused(toy):
    """936 test 4, on the toy well: tiles that do not touch are not accepted."""
    layout, tiles, _truth = toy
    adjacent = {(a, b) for a, b, _axis in layout.pairs()}
    controls = [(a, b) for a in range(0, 21, 3) for b in range(a + 2, 21, 5)
                if (a, b) not in adjacent and (b, a) not in adjacent][:12]
    found = {acceptance: [core.register_strip_pair(
        tiles[a], tiles[b], "vertical", expected_overlap=OVERLAP, tolerance=4,
        acceptance=acceptance).accepted for a, b in controls]
        for acceptance in ("peak_ratio", "both")}
    assert len(controls) == 12
    assert not any(found["both"])
    assert sum(found["peak_ratio"]) <= 1, (
        "the layout window admits a chance peak now and then; more is a defect")


def test_every_new_align_key_has_a_default_and_the_ops_mapping_is_complete():
    from spacr.settings import ALPHA_FEATURES

    keys = set(ALPHA_FEATURES[936]["module_settings"]["align"])
    defaults = align.default_settings()
    assert keys <= set(defaults)
    mirrored = set(core.OPS_TO_ALIGN_KEYS.values())
    assert mirrored <= set(defaults)
    assert keys - mirrored == {"align_engine", "align_placement_seed_export"}
    assert defaults["align_engine"] == "legacy"


def test_the_core_options_refuse_unknown_values():
    with pytest.raises(Exception, match="align_engine must be one of"):
        align._core_options({"align_engine": "fast"})
    with pytest.raises(Exception, match="acceptance must be one of"):
        align._core_options({"align_engine": "core", "acceptance": "maybe"})
    assert align._core_options({"align_engine": "legacy"}) is None
