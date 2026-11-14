"""Per-tile affine alignment of phenotype fields to the sequencing nuclei (item 930).

OPS stage 1, arm (vii): after the per-field similarity re-solve, one affine per
field and one per sequencing tile group of the field (at least
``ops_align_tile_min_nuclei`` nuclei), fitted on mutual nearest pairs with
trimmed passes, each nucleus moved by its tile group's affine. The tests use a
synthetic well whose truth differs by tile, so a single transform per field
cannot recover it. 'similarity' leaves the merge outputs as before item 930.

Since 2026-11-13 'tile_affine' is the default (maintainer decision; it left
alpha). Settings files stamped before schema version 933, or not stamped,
rerun with 'similarity' through ``DEFAULT_CHANGES[933]``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from spacr import ops_engine
from spacr.ops_merge import (affine_polish, affine_within_bounds,
                             field_to_well, field_transform,
                             refine_field_transform, tile_affine_to_well,
                             tile_affine_transform)
from spacr.ops_store import read_table, write_table

PLATE = "P1"
WELL = "A1"
FIELD = 1200.0
SCALE = 0.5
TILE = 300.0
#: The four sequencing tiles: origin (y, x), per-tile shift (y, x) in well
#: px and rotation in degrees about the tile centre.
TILES = (((0.0, 0.0), (0.6, -0.4), 0.04),
         ((0.0, TILE), (-0.5, 0.7), -0.03),
         ((TILE, 0.0), (0.3, 0.8), 0.02),
         ((TILE, TILE), (-0.7, -0.5), -0.05))


def _centres():
    return np.array([[y + TILE / 2, x + TILE / 2] for (y, x), _, _ in TILES])


def _truth(points, seed):
    """Field points in the well frame: the seed, then their tile's own shift."""
    well = field_to_well(points, seed)
    _, own = cKDTree(_centres()).query(well)
    out = well.copy()
    for k, ((_y, _x), shift, degrees) in enumerate(TILES):
        members = own == k
        a = np.radians(degrees)
        rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        centre = _centres()[k]
        out[members] = ((well[members] - centre) @ rotation.T + centre
                        + np.asarray(shift))
    return out


def _synthetic(seed_rng=7):
    """Field points on a jittered grid away from the tile seams, and their truth."""
    rng = np.random.default_rng(seed_rng)
    seed = field_transform(SCALE, 0.3, (TILE, TILE), (FIELD, FIELD))
    grid = np.array([[y, x] for y in np.arange(12.0, FIELD - 12, 30.0)
                     for x in np.arange(12.0, FIELD - 12, 30.0)])
    points = grid + rng.uniform(-6, 6, grid.shape)
    well = field_to_well(points, seed)
    away = np.min(np.abs(well - TILE), axis=1) > 3.0
    points = points[away]
    truth = _truth(points, seed)
    extra = rng.uniform(0, 2 * TILE, (60, 2))
    targets = np.vstack([truth, extra])
    return points, targets, truth, seed


def _fit(points, targets, seed, **options):
    start, _inliers, _residual, refined = refine_field_transform(
        points, targets, seed, radius=3.0)
    assert refined
    return tile_affine_transform(points, targets, start, seed, _centres(),
                                 refined=refined, **options)


def test_per_tile_truth_is_recovered_within_a_tenth_of_a_pixel():
    points, targets, truth, seed = _synthetic()
    field, groups = _fit(points, targets, seed)
    assert sorted(groups) == [0, 1, 2, 3]
    moved = tile_affine_to_well(points, field, groups, _centres())
    error = np.hypot(*(moved - truth).T)
    assert error.max() < 0.1
    single = np.hypot(*(field_to_well(points, field) - truth).T)
    assert single.mean() > 0.3


def test_small_groups_keep_the_field_affine():
    points, targets, truth, seed = _synthetic()
    field, groups = _fit(points, targets, seed, min_nuclei=10 ** 6)
    assert groups == {}
    moved = tile_affine_to_well(points, field, groups, _centres())
    assert np.array_equal(moved, points @ field[1].T + field[2])
    _, own = cKDTree(_centres()).query(moved)
    counts = np.bincount(own, minlength=4)
    middle = int(np.sort(counts)[1])
    _field, some = _fit(points, targets, seed, min_nuclei=middle + 1)
    assert set(some) == {k for k in range(4) if counts[k] > middle}


def test_an_unrefined_field_keeps_its_start_and_gets_no_groups():
    points, targets, _truth_points, seed = _synthetic()
    field, groups = tile_affine_transform(points, targets, seed, seed,
                                          _centres(), refined=False)
    assert groups == {}
    assert np.array_equal(field[1], SCALE * np.asarray(seed[1]))
    assert np.array_equal(field[2], seed[2])


def test_the_fit_gives_the_same_bits_on_one_and_four_threads():
    threadpoolctl = pytest.importorskip("threadpoolctl")
    points, targets, _truth_points, seed = _synthetic()
    runs = []
    for threads in (1, 4, 1):
        with threadpoolctl.threadpool_limits(limits=threads, user_api="blas"):
            field, groups = _fit(points, targets, seed)
        runs.append(np.concatenate(
            [field[1].ravel(), field[2]]
            + [np.r_[g[1].ravel(), g[2]] for _k, g in sorted(groups.items())]
        ).tobytes())
    assert runs[0] == runs[1] == runs[2]


def test_the_polish_and_bounds_follow_the_reference_rules():
    points, targets, truth, seed = _synthetic()
    matrix = SCALE * np.asarray(seed[1])
    assert affine_within_bounds(matrix, seed)
    assert not affine_within_bounds(matrix * 1.03, seed)
    assert not affine_within_bounds(matrix @ np.diag([1.03, 0.99]), seed)
    assert not affine_within_bounds(matrix @ np.diag([1.0, -1.0]), seed)
    fit = affine_polish(points, targets, matrix, seed[2])
    assert fit is not None and fit[2] >= 0.9 * len(points)
    assert affine_polish(points[:15], targets, matrix, seed[2]) is None


def _well(tmp_path, *, geometry=True):
    """One field over four sequencing tiles whose positions differ by tile."""
    points, targets, truth, seed = _synthetic()
    off = field_transform(SCALE, 0.3, (TILE + 1.2, TILE - 0.8), (FIELD, FIELD))
    db = str(tmp_path / "measurements.db")
    write_table(db, "ops_phenotype", pd.DataFrame([{
        "plate": PLATE, "well": WELL, "site": 0, "centre_y": TILE + 1.2,
        "centre_x": TILE - 0.8, "is_anchor": 0, "scale": SCALE,
        "rotation_deg": 0.3, "offset_y": float(off[2][0]),
        "offset_x": float(off[2][1]), "field_height": FIELD,
        "field_width": FIELD, "transform_source": "raster"}]))
    write_table(db, "ops_objects", pd.DataFrame({
        "plate": PLATE, "well": WELL, "object_id": np.arange(len(targets)),
        "centroid_y": targets[:, 0], "centroid_x": targets[:, 1]}))
    if geometry:
        write_table(db, "ops_geometry", pd.DataFrame([{
            "plate": PLATE, "well": WELL, "cycle": 1, "site": 10 + k,
            "y": y, "x": x, "origin_y": y, "origin_x": x,
            "tile_height": TILE, "tile_width": TILE}
            for k, ((y, x), _s, _d) in enumerate(TILES)]))
    pd.DataFrame({"plate": PLATE, "well": WELL, "site": 0,
                  "label": np.arange(1, len(points) + 1),
                  "centroid_row": points[:, 0],
                  "centroid_col": points[:, 1]}).to_parquet(
        tmp_path / f"ops_phenotype_cells_{WELL}.parquet")
    return db


def _tables(db):
    return (read_table(db, "ops_merge").reset_index(drop=True),
            read_table(db, "ops_phenotype").reset_index(drop=True))


SIMILARITY = {"ops_align_model": "similarity"}


def test_similarity_leaves_the_merge_as_before_930(tmp_path):
    db = _well(tmp_path)
    ops_engine._ph_merge(db, PLATE, WELL, dict(SIMILARITY), str(tmp_path),
                         False)
    default = _tables(db)
    report = ops_engine._ph_merge(db, PLATE, WELL, {
        "ops_align_model": "similarity", "ops_align_tile_min_nuclei": 5,
        "ops_align_match_radius": 1.0, "ops_align_trim_passes": 1},
        str(tmp_path), False)
    assert "align" not in report
    explicit = _tables(db)
    pd.testing.assert_frame_equal(default[0], explicit[0])
    pd.testing.assert_frame_equal(default[1], explicit[1])
    assert not [c for c in explicit[1].columns if c.startswith("align_")]


def test_the_default_model_is_tile_affine(tmp_path):
    """A settings dict without ops_align_model, and the module defaults,
    run the tile-affine model and give the explicit tile-affine tables."""
    from spacr.ops_settings import OPS_DEFAULTS

    assert OPS_DEFAULTS["ops_align_model"] == "tile_affine"
    assert ops_engine._PH_ALIGN_MODELS[0] == "tile_affine"
    db = _well(tmp_path)
    report = ops_engine._ph_merge(db, PLATE, WELL, {}, str(tmp_path), False)
    assert report["align"]["model"] == "tile_affine"
    default = _tables(db)
    assert default[1].iloc[0]["align_model"] == "tile_affine"
    ops_engine._ph_merge(db, PLATE, WELL, {"ops_align_model": "tile_affine"},
                         str(tmp_path), False)
    explicit = _tables(db)
    pd.testing.assert_frame_equal(default[0], explicit[0])
    pd.testing.assert_frame_equal(default[1], explicit[1])
    defaults = {key: OPS_DEFAULTS[key] for key in OPS_DEFAULTS
                if key.startswith("ops_align_")}
    ops_engine._ph_merge(db, PLATE, WELL, defaults, str(tmp_path), False)
    pd.testing.assert_frame_equal(default[0], _tables(db)[0])


def test_tile_affine_links_closer_and_records_its_geometry(tmp_path):
    db = _well(tmp_path)
    ops_engine._ph_merge(db, PLATE, WELL, dict(SIMILARITY), str(tmp_path),
                         False)
    similarity = _tables(db)
    report = ops_engine._ph_merge(db, PLATE, WELL,
                                  {"ops_align_model": "tile_affine"},
                                  str(tmp_path), False)
    tiled = _tables(db)
    assert report["align"]["model"] == "tile_affine"
    assert report["align"]["tile_groups"] == 4
    assert tiled[0]["distance"].median() < 0.05
    assert similarity[0]["distance"].median() > 0.2
    row = tiled[1].iloc[0]
    assert row["align_model"] == "tile_affine" and row["align_device"] == "cpu"
    assert int(row["align_tile_groups"]) == 4
    groups = json.loads(row["align_tile_transforms"])
    assert [g[0] for g in groups] == [10, 11, 12, 13]
    assert all(len(g) == 8 and g[1] >= 200 for g in groups)
    # the similarity re-solve it started from is still stored
    assert row["refined_scale"] == similarity[1].iloc[0]["refined_scale"]
    # a later similarity run leaves no tile-affine geometry behind
    ops_engine._ph_merge(db, PLATE, WELL, dict(SIMILARITY), str(tmp_path),
                         False)
    again = _tables(db)
    pd.testing.assert_frame_equal(similarity[0], again[0])
    pd.testing.assert_frame_equal(similarity[1], again[1])


def test_tile_affine_reruns_give_the_same_tables(tmp_path):
    db = _well(tmp_path)
    settings = {"ops_align_model": "tile_affine", "ops_ownership": "a1b"}
    ops_engine._ph_merge(db, PLATE, WELL, settings, str(tmp_path), False)
    first = _tables(db)
    ops_engine._ph_merge(db, PLATE, WELL, settings, str(tmp_path), False)
    second = _tables(db)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])


def test_tile_affine_needs_the_tile_positions_and_a_known_model(tmp_path):
    db = _well(tmp_path, geometry=False)
    with pytest.raises(ValueError, match="ops_geometry"):
        ops_engine._ph_merge(db, PLATE, WELL,
                             {"ops_align_model": "tile_affine"},
                             str(tmp_path), False)
    with pytest.raises(ValueError, match="ops_align_model"):
        ops_engine._ph_merge(db, PLATE, WELL, {"ops_align_model": "spline"},
                             str(tmp_path), False)
    with pytest.raises(ValueError, match="ops_align_trim_passes"):
        ops_engine._ph_merge(db, PLATE, WELL,
                             {"ops_align_model": "tile_affine",
                              "ops_align_trim_passes": 0},
                             str(tmp_path), False)


def _settings_file(path, rows):
    import csv

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Key", "Value"])
        writer.writerows(rows)
    return str(path)


def test_a_settings_file_from_before_the_switch_reruns_with_similarity(
        tmp_path, caplog):
    """The reproducibility rule: files without a stamp, or stamped 931 or
    932, were written while 'similarity' was the default and rerun with it;
    a file stamped 933 or later, and a run resolved today, get
    'tile_affine'; a key the file sets always wins. The row is keyed 933,
    not 930, because files were already stamped 932 when 930's default
    changed."""
    import logging

    from spacr.cli import resolve_module, resolve_settings
    from spacr.settings import (DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION,
                                SETTINGS_VERSION_KEY)

    assert DEFAULT_CHANGES[933] == {"ops_align_model": "similarity"}
    assert SETTINGS_SCHEMA_VERSION >= 933
    assert 930 not in DEFAULT_CHANGES
    ops = resolve_module("ops")
    with caplog.at_level(logging.WARNING, logger="spacr.settings"):
        old = resolve_settings(ops, _settings_file(
            tmp_path / "old.csv", [("plate", "P1")]))
    assert old["ops_align_model"] == "similarity"
    assert "ops_align_model='similarity'" in caplog.text
    for stamp in ("931", "932"):
        older = resolve_settings(ops, _settings_file(
            tmp_path / f"{stamp}.csv",
            [(SETTINGS_VERSION_KEY, stamp), ("plate", "P1")]))
        assert older["ops_align_model"] == "similarity", stamp
        # the tile_affine parameters kept their defaults; no row for them
        assert older["ops_align_tile_min_nuclei"] == 200
    new = resolve_settings(ops, _settings_file(
        tmp_path / "933.csv", [(SETTINGS_VERSION_KEY, "933"), ("plate", "P1")]))
    assert new["ops_align_model"] == "tile_affine"
    assert resolve_settings(ops, None)["ops_align_model"] == "tile_affine"
    explicit = resolve_settings(ops, _settings_file(
        tmp_path / "explicit.csv",
        [(SETTINGS_VERSION_KEY, "932"), ("plate", "P1"),
         ("ops_align_model", "tile_affine")]))
    assert explicit["ops_align_model"] == "tile_affine"
    chosen = resolve_settings(ops, _settings_file(
        tmp_path / "chosen.csv",
        [(SETTINGS_VERSION_KEY, str(SETTINGS_SCHEMA_VERSION)),
         ("plate", "P1"), ("ops_align_model", "similarity")]))
    assert chosen["ops_align_model"] == "similarity"


def test_the_align_settings_left_alpha():
    from spacr.ops_settings import OPS_TOOLTIPS
    from spacr.settings import ALPHA_FEATURES

    assert 930 not in ALPHA_FEATURES
    keys = ("ops_align_model", "ops_align_tile_min_nuclei",
            "ops_align_match_radius", "ops_align_trim_passes")
    for entry in ALPHA_FEATURES.values():
        assert not set(keys) & set(entry.get("settings", ())), entry
    for key in keys:
        assert not OPS_TOOLTIPS[key].rstrip().endswith("Alpha."), key
    assert "Default 'tile_affine'." in OPS_TOOLTIPS["ops_align_model"]
