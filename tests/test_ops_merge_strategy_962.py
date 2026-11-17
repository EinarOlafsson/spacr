"""OPS merge routes: tile pattern, well mosaic and the automatic fallback (item 962).

Brieflow (Nat Commun 2026) merges phenotype and sequencing cells either tile
by tile, from a hash of each tile's cell pattern, or on two stitched well
mosaics placed from the microscope's stage coordinates, for sparse wells
whose tiles hold too few cells for a pattern. ``ops_merge_strategy`` offers
both next to spaCR's own merge ('current', the default, unchanged), plus
'auto': the tile route, switched to the stitch route for a well whose share
of weak tile pairs is too high. The tests use a synthetic well of nine
phenotype fields over four sequencing tiles, dense and thinned, whose true
pairs are known.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from spacr import ops_engine
from spacr.ops_merge import (_choose_route, _compose, _triangle_votes,
                             _weak_pair, _well_similarity, field_to_well,
                             field_transform)
from spacr.ops_store import read_table, write_table

PLATE = "P1"
WELL = "A1"
FIELD = 640.0
SCALE = 0.5
TILE = 480.0
DEGREES = 0.4
#: Field centres, well px: a 3 x 3 grid with overlap, inside the 2 x 2 tiles.
CENTRES = [(170.0 + 300.0 * a, 170.0 + 300.0 * b)
           for a in range(3) for b in range(3)]
#: How far the stored placement of each field is off its truth, well px.
PLACEMENT_ERROR = (2.5, -1.5)


def _nuclei(rng, spacing=18.0, size=2 * TILE):
    """Sequencing nuclei on a jittered grid over the four tiles."""
    grid = np.array([[y, x] for y in np.arange(6.0, size - 6, spacing)
                     for x in np.arange(6.0, size - 6, spacing)])
    return grid + rng.uniform(-5.0, 5.0, grid.shape)


def _truth(site):
    return field_transform(SCALE, DEGREES, CENTRES[site], (FIELD, FIELD))


def _well(tmp_path, *, keep=1.0, seed=3, offsets=None):
    """The synthetic well; ``keep`` thins the cells, as the proof does.

    :returns: ``(db, truth)``, ``truth`` mapping ``(site, label)`` to the
        object id of the same cell.
    """
    rng = np.random.default_rng(seed)
    targets = _nuclei(rng)
    kept = rng.random(len(targets)) < keep
    rows, truth = [], {}
    for site in range(len(CENTRES)):
        scale, rotation, translation = _truth(site)
        local = ((targets - translation) @ rotation) / scale
        inside = np.flatnonzero(kept & np.all((local >= 2.0)
                                              & (local <= FIELD - 2.0), axis=1))
        jitter = rng.normal(0.0, 0.4, (len(inside), 2))
        for label, (k, point) in enumerate(zip(inside, local[inside] + jitter),
                                           start=1):
            rows.append((PLATE, WELL, site, label, point[0], point[1]))
            truth[(site, label)] = int(k)
    write_table(str(tmp_path / "measurements.db"), "ops_phenotype",
                pd.DataFrame([_placement(site, offsets) for site in
                              range(len(CENTRES))]))
    db = str(tmp_path / "measurements.db")
    write_table(db, "ops_objects", pd.DataFrame({
        "plate": PLATE, "well": WELL,
        "object_id": np.flatnonzero(kept).astype(np.int64),
        "centroid_y": targets[kept, 0], "centroid_x": targets[kept, 1]}))
    write_table(db, "ops_geometry", pd.DataFrame([{
        "plate": PLATE, "well": WELL, "cycle": 1, "site": 10 + 2 * a + b,
        "y": TILE * a, "x": TILE * b, "origin_y": TILE * a,
        "origin_x": TILE * b, "tile_height": TILE, "tile_width": TILE}
        for a in range(2) for b in range(2)]))
    pd.DataFrame(rows, columns=["plate", "well", "site", "label",
                                "centroid_row", "centroid_col"]).to_parquet(
        tmp_path / f"ops_phenotype_cells_{WELL}.parquet")
    return db, truth


def _placement(site, offsets=None):
    shift = np.asarray(PLACEMENT_ERROR) if offsets is None else offsets[site]
    centre = np.asarray(CENTRES[site]) + shift
    placed = field_transform(SCALE, DEGREES, centre, (FIELD, FIELD))
    return {"plate": PLATE, "well": WELL, "site": site,
            "centre_y": centre[0], "centre_x": centre[1], "is_anchor": 0,
            "scale": SCALE, "rotation_deg": DEGREES,
            "offset_y": float(placed[2][0]), "offset_x": float(placed[2][1]),
            "field_height": FIELD, "field_width": FIELD,
            "transform_source": "raster"}


def _merge(db, tmp_path, **settings):
    base = {"ops_align_model": "similarity"}
    base.update(settings)
    report = ops_engine._ph_merge(db, PLATE, WELL, base, str(tmp_path), False)
    return report, read_table(db, "ops_merge").reset_index(drop=True)


def _recall(frame, truth):
    """Correct links over the cells in the truth, counted once per nucleus.

    A nucleus under two overlapping fields is one cell; the merge keeps one
    of its two copies.
    """
    found = {(int(r.site), int(r.label)): int(r.object_id)
             for r in frame.itertuples()}
    hits = {value for key, value in truth.items() if found.get(key) == value}
    return len(hits) / max(1, len(set(truth.values())))


def _positions(tmp_path):
    """A stage-position table: tiles and fields where they truly are."""
    rows = [{"well": WELL, "modality": "sbs", "site": 10 + 2 * a + b,
             "y": TILE * a, "x": TILE * b}
            for a in range(2) for b in range(2)]
    for site, centre in enumerate(CENTRES):
        corner = (np.asarray(centre) - SCALE * FIELD / 2.0) / SCALE
        rows.append({"well": WELL, "modality": "phenotype", "site": site,
                     "y": corner[0], "x": corner[1]})
    path = tmp_path / "positions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_current_is_the_default_and_leaves_the_merge_as_before(tmp_path):
    from spacr.ops_settings import OPS_DEFAULTS

    assert OPS_DEFAULTS["ops_merge_strategy"] == "current"
    assert ops_engine._MERGE_STRATEGIES[0] == "current"
    db, _truth_pairs = _well(tmp_path)
    report, default = _merge(db, tmp_path)
    phenotype = read_table(db, "ops_phenotype").reset_index(drop=True)
    assert "strategy" not in report
    explicit_report, explicit = _merge(
        db, tmp_path, ops_merge_strategy="current",
        ops_merge_stitch_source="registered", ops_merge_min_votes=3,
        ops_merge_min_match_fraction=0.9, ops_merge_fallback_share=0.0)
    assert "strategy" not in explicit_report
    pd.testing.assert_frame_equal(default, explicit)
    pd.testing.assert_frame_equal(
        phenotype, read_table(db, "ops_phenotype").reset_index(drop=True))
    assert not os.path.exists(tmp_path / "merge_strategy_report.csv")


def test_the_tile_route_recovers_the_dense_pairs(tmp_path):
    db, truth = _well(tmp_path)
    _report, current = _merge(db, tmp_path)
    report, tiled = _merge(db, tmp_path, ops_merge_strategy="tile")
    row = report["strategy"]
    assert row["strategy_used"] == "tile" and row["weak_tile_pairs"] == 0
    assert row["tile_pairs"] == len(CENTRES)
    assert row["votes_p10"] >= 10 and row["match_fraction_p10"] >= 0.5
    assert _recall(tiled, truth) >= 0.95
    assert abs(_recall(tiled, truth) - _recall(current, truth)) <= 0.005


def test_auto_keeps_the_tile_route_on_a_dense_well(tmp_path):
    db, truth = _well(tmp_path)
    _report, current = _merge(db, tmp_path)
    report, auto = _merge(db, tmp_path, ops_merge_strategy="auto")
    assert report["strategy"]["strategy_used"] == "tile"
    assert abs(_recall(auto, truth) - _recall(current, truth)) <= 0.005
    table = pd.read_csv(tmp_path / "merge_strategy_report.csv")
    assert table.loc[0, "strategy_requested"] == "auto"
    assert table.loc[0, "strategy_used"] == "tile"
    assert {"reason", "weak_tile_pairs", "votes_p50", "match_fraction_p50",
            "residual_px_median", "matched_sbs_pct",
            "matched_phenotype_pct", "seconds"} <= set(table.columns)


def test_auto_falls_back_to_the_stitch_route_on_a_sparse_well(tmp_path):
    db, truth = _well(tmp_path, keep=0.02)
    _report, tiled = _merge(db, tmp_path, ops_merge_strategy="tile",
                            ops_merge_min_votes=10)
    report, auto = _merge(db, tmp_path, ops_merge_strategy="auto",
                          ops_merge_stitch_source="registered",
                          ops_merge_min_votes=10)
    row = report["strategy"]
    assert row["strategy_used"] == "stitch", row["reason"]
    assert "weak" in row["reason"]
    assert row["weak_tile_pairs"] > 0.25 * row["tile_pairs"]
    assert _recall(auto, truth) > _recall(tiled, truth)
    assert _recall(auto, truth) >= 0.9


def test_the_stitch_route_places_tiles_from_stage_positions(tmp_path):
    db, truth = _well(tmp_path)
    report, stitched = _merge(
        db, tmp_path, ops_merge_strategy="stitch",
        ops_merge_stitch_source="metadata",
        ops_merge_stage_positions=_positions(tmp_path))
    row = report["strategy"]
    assert row["strategy_used"] == "stitch"
    assert row["stitch_source"] == "metadata"
    assert row["well_regions"] >= 1 and abs(row["well_scale"] - 1.0) < 0.01
    assert _recall(stitched, truth) >= 0.95


def test_metadata_without_stage_positions_is_refused_clearly(tmp_path):
    db, _truth_pairs = _well(tmp_path)
    with pytest.raises(ValueError, match="stage position"):
        _merge(db, tmp_path, ops_merge_strategy="stitch",
               ops_merge_stitch_source="metadata")
    table = tmp_path / "bad.csv"
    pd.DataFrame({"well": [WELL], "site": [0]}).to_csv(table, index=False)
    with pytest.raises(ValueError, match="needs the columns"):
        _merge(db, tmp_path, ops_merge_strategy="stitch",
               ops_merge_stage_positions=str(table))


def test_a_shifted_placement_finds_almost_nothing(tmp_path):
    """Negative control: each field's placement moved a whole field away."""
    rng = np.random.default_rng(11)
    offsets = {site: rng.choice([-1.0, 1.0], 2) * 2.0 * SCALE * FIELD
               for site in range(len(CENTRES))}
    db, truth = _well(tmp_path, offsets=offsets)
    report, tiled = _merge(db, tmp_path, ops_merge_strategy="tile")
    assert _recall(tiled, truth) <= 0.05
    assert report["strategy"]["weak_tile_pairs"] >= len(CENTRES) - 2


def test_reruns_give_the_same_tables_and_report(tmp_path):
    db, _truth_pairs = _well(tmp_path, keep=0.02)
    settings = {"ops_merge_strategy": "auto",
                "ops_merge_stitch_source": "registered"}
    _report, first = _merge(db, tmp_path, **settings)
    one = pd.read_csv(tmp_path / "merge_strategy_report.csv")
    _report, second = _merge(db, tmp_path, **settings)
    two = pd.read_csv(tmp_path / "merge_strategy_report.csv")
    pd.testing.assert_frame_equal(first, second)
    columns = [c for c in one.columns if "seconds" not in c]
    pd.testing.assert_frame_equal(one[columns], two[columns])
    assert len(two) == 1


def test_unknown_values_are_refused(tmp_path):
    db, _truth_pairs = _well(tmp_path)
    with pytest.raises(ValueError, match="ops_merge_strategy"):
        _merge(db, tmp_path, ops_merge_strategy="mosaic")
    with pytest.raises(ValueError, match="ops_merge_stitch_source"):
        _merge(db, tmp_path, ops_merge_strategy="stitch",
               ops_merge_stitch_source="guess")
    with pytest.raises(ValueError, match="ops_merge_fallback_share"):
        _merge(db, tmp_path, ops_merge_strategy="auto",
               ops_merge_fallback_share=1.5)


def test_votes_tell_a_pattern_from_an_unrelated_one():
    rng = np.random.default_rng(5)
    points = rng.uniform(0, 2000, (600, 2))
    truth = field_transform(0.25, 0.7, (900.0, 1200.0), (2000, 2000))
    target = field_to_well(points, truth) + rng.normal(0, 0.3, (600, 2))
    fit, votes = _triangle_votes(points, target, scale=0.25)
    assert votes >= 100
    assert abs(fit[0] - 0.25) < 1e-3
    assert np.abs(field_to_well(points, fit) - field_to_well(points, truth)
                  ).max() < 1.0
    _fit, unrelated = _triangle_votes(rng.uniform(0, 2000, (600, 2)), target,
                                      scale=0.25)
    assert unrelated < 10
    assert _triangle_votes(points[:2], target) == (None, 0)


def test_the_well_similarity_and_composition():
    rng = np.random.default_rng(9)
    target = rng.uniform(0, 3000, (6000, 2))
    shift = (1.0, np.eye(2), np.array([7.5, -4.0]))
    source = field_to_well(target, (1.0, np.eye(2), -shift[2]))
    fit, votes, regions = _well_similarity(source, target, scale=1.0)
    assert regions >= 4 and votes > 0
    assert np.allclose(fit[2], shift[2], atol=0.1)
    inner = field_transform(0.5, 0.0, (100.0, 100.0), (200, 200))
    both = _compose(fit, inner)
    point = np.array([[10.0, 20.0]])
    assert np.allclose(field_to_well(point, both),
                       field_to_well(field_to_well(point, inner), fit))
    identity, none, used = _well_similarity(source[:2], target)
    assert none == 0 and used == 0 and identity[0] == 1.0


def test_the_fallback_rule():
    assert _weak_pair(9, 0.9, 10, 0.5)
    assert _weak_pair(50, 0.4, 10, 0.5)
    assert _weak_pair(50, float("nan"), 10, 0.5)
    assert not _weak_pair(10, 0.5, 10, 0.5)
    assert _choose_route(3, 10, 0.25)[0] == "stitch"
    assert _choose_route(2, 10, 0.25)[0] == "tile"
    assert _choose_route(0, 0, 0.25)[0] == "stitch"


def test_the_route_settings_are_alpha():
    from spacr.settings import ALPHA_FEATURES
    from spacr.settings_spec import convert_settings_dict_for_gui

    keys = set(ALPHA_FEATURES[962]["settings"])
    assert keys == {"ops_merge_strategy", "ops_merge_stitch_source",
                    "ops_merge_stage_positions", "ops_merge_min_votes",
                    "ops_merge_min_match_fraction",
                    "ops_merge_fallback_share"}
    gui = convert_settings_dict_for_gui({"ops_merge_strategy": "current"})
    assert "auto" in str(gui["ops_merge_strategy"])
