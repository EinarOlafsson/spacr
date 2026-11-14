"""Core ownership (item 852), flat-field correction (853) and Funk et al.'s
features (857) on the OPS phenotype path.

Each is checked against planted truth: two overlapping fields whose closer
match is the margin copy, a flat that is constant over every cell (where
the analytic correction is exact), and objects whose mass displacement and
correlations have closed-form values. The plate A equivalence with the
pilot's scripts is recorded in the paper folder, not here.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
pytest.importorskip("pyarrow")

from spacr import ops_engine
from spacr.ops_flatfield import (build_flat_maps, choose_flat_fields,
                                 correct_features, correct_field,
                                 correction_inputs, correction_kind,
                                 field_blocks, flagged_features, flat_planes,
                                 load_flat_maps, mask_pass, write_flat_maps)
from spacr.ops_merge import (deduplicate_matches, deepest_field_matches,
                             field_footprint, field_transform,
                             footprint_depth, overlapping_fields)
from spacr.ops_sample import funk_object_features, measure_phenotype_field
from spacr.ops_store import read_table, write_table

PLATE = "P1"
SIZE = 200


# --- 852: ownership ---------------------------------------------------------

def test_footprint_depth_is_the_distance_to_the_nearest_edge():
    low, high = field_footprint(field_transform(1.0, 0.0, (100, 100),
                                                (SIZE, SIZE)), (SIZE, SIZE))
    assert np.allclose(low, [0, 0]) and np.allclose(high, [200, 200])
    depth = footprint_depth(np.array([[100, 100], [10, 150], [-5, 50]]),
                            low, high)
    assert depth.tolist() == [100, 10, -5]


def test_overlapping_fields_ignore_fields_that_only_touch():
    prints = {0: (np.array([0, 0]), np.array([200, 200])),
              1: (np.array([0, 150]), np.array([200, 350])),
              2: (np.array([0, 350]), np.array([200, 550]))}
    assert overlapping_fields(prints) == {0: [1], 1: [0], 2: []}


def test_closer_match_keeps_the_margin_copy_and_a1b_the_deep_one():
    rows = pd.DataFrame({"site": [0, 1, 0], "label": [1, 1, 2],
                         "object_id": [7, 7, 8], "distance": [1.0, 0.5, 0.2],
                         "sbs_depth_px": [40.0, 10.0, 90.0]})
    closer, dropped = deduplicate_matches(rows)
    assert dropped == 1 and closer.loc[closer.object_id == 7, "site"].item() == 1
    deep, dropped = deepest_field_matches(rows)
    assert dropped == 1 and deep.loc[deep.object_id == 7, "site"].item() == 0
    tie = rows.assign(sbs_depth_px=[10.0, 10.0, 90.0])
    assert deepest_field_matches(tie)[0].loc[
        lambda f: f.object_id == 7, "site"].item() == 1


def _two_fields(db):
    """Fields 0 and 1 side by side at scale 1, overlapping in x 150-200."""
    rows = []
    for site, cx in ((0, 100.0), (1, 250.0)):
        transform = field_transform(1.0, 0.0, (100.0, cx), (SIZE, SIZE))
        rows.append({"plate": PLATE, "well": "A1", "site": site,
                     "centre_y": 100.0, "centre_x": cx, "is_anchor": 1,
                     "scale": 1.0, "rotation_deg": 0.0,
                     "offset_y": float(transform[2][0]),
                     "offset_x": float(transform[2][1]),
                     "field_height": SIZE, "field_width": SIZE,
                     "transform_source": "anchor"})
    write_table(db, "ops_phenotype", pd.DataFrame(rows))


def _overlap_cells(tmp_path, db):
    """Cells in well x: 50 (field 0 only), 160 (both, deeper in field 0),
    190 (both, deeper in field 1), 300 (field 1 only), plus an unmatched
    cell at 170 seen by both fields. Field 1's copies match closer."""
    objects = pd.DataFrame({"plate": PLATE, "well": "A1",
                            "object_id": [1, 2, 3, 4],
                            "centroid_y": 100.0,
                            "centroid_x": [50.0, 160.0, 190.0, 300.0]})
    write_table(db, "ops_objects", objects)
    cells = []
    for site, origin, jitter in ((0, 0.0, 1.0), (1, 150.0, 0.5)):
        label = 0
        for x in (50.0, 160.0, 190.0, 300.0, 170.0):
            local = x - origin
            if not 0 <= local <= SIZE:
                continue
            label += 1
            offset = 0.0 if x == 170.0 else jitter
            cells.append({"plate": PLATE, "well": "A1", "site": site,
                          "label": label, "centroid_row": 100.0 + offset,
                          "centroid_col": local,
                          "well_x": x})
    frame = pd.DataFrame(cells)
    frame.drop(columns="well_x").to_parquet(
        tmp_path / "ops_phenotype_cells_A1.parquet")
    return frame


@pytest.mark.parametrize("rule", ["closer_match", "a1b", "a1"])
def test_merge_ownership_rules(tmp_path, rule):
    db = str(tmp_path / "measurements.db")
    _two_fields(db)
    cells = _overlap_cells(tmp_path, db)
    # no ops_geometry in this synthetic well: merge with 'similarity'
    report = ops_engine._ph_merge(db, PLATE, "A1", {"ops_ownership": rule,
                                                    "ops_align_model": "similarity"},
                                  str(tmp_path), False)
    merged = read_table(db, "ops_merge").merge(
        cells[["site", "label", "well_x"]], on=["site", "label"])
    site_of = dict(zip(merged.object_id, merged.site))
    assert sorted(site_of) == [1, 2, 3, 4]
    assert site_of[1] == 0 and site_of[4] == 1 and site_of[3] == 1
    if rule == "closer_match":
        assert site_of[2] == 1  # the closer, margin copy
        assert not os.path.isfile(ops_engine._ph_ownership_path(
            str(tmp_path), "A1"))
        assert report["ownership"] == {"rule": "closer_match"}
        return
    assert site_of[2] == 0  # the deep copy
    owner = pd.read_parquet(ops_engine._ph_ownership_path(str(tmp_path), "A1"))
    owner = owner.merge(cells[["site", "label", "well_x"]].rename(
        columns={"well_x": "planted_x"}), on=["site", "label"])
    flags = {(int(r.site), r.planted_x): r for r in owner.itertuples()}
    # the unmatched cell at 170 is owned by field 0 only (A1 fallback)
    assert not flags[(0, 170.0)].any_match and flags[(0, 170.0)].owned
    assert flags[(1, 170.0)].in_overlap_margin and not flags[(1, 170.0)].owned
    assert flags[(1, 160.0)].in_overlap_margin and not flags[(1, 160.0)].owned
    assert flags[(0, 190.0)].in_overlap_margin and not flags[(0, 190.0)].owned
    assert int(owner["owned"].sum()) == 5
    assert report["ownership"]["owned"] == 5
    assert report["ownership"]["switched_from_closer_match"] == 1
    assert (owner["ownership_rule"] == rule).all()
    assert "sbs_depth_px" in read_table(db, "ops_merge")


def test_an_unknown_ownership_rule_is_refused(tmp_path):
    with pytest.raises(ValueError, match="ops_ownership must be one of"):
        ops_engine._ph_ownership_rule({"ops_ownership": "nearest"})


# --- 853: flat-field ---------------------------------------------------------

def _pilot_build(blocks, channels):
    """The plate A pilot's flat.py / build_flat.py, verbatim but for I/O."""
    from scipy import ndimage as ndi

    H = blocks.shape[1]
    yy, xx = np.mgrid[:H, :H] / (H - 1) * 2 - 1

    def poly(m, deg=4):
        cols = [(xx ** i * yy ** j).ravel() for i in range(deg + 1)
                for j in range(deg + 1 - i)]
        X = np.stack(cols, 1)
        c, *_ = np.linalg.lstsq(X, m.ravel(), rcond=None)
        return (X @ c).reshape(m.shape)
    out, offsets = {}, {}
    for c, ch in enumerate(channels):
        med = np.median(blocks[..., c], 0)
        off = float(np.percentile(blocks[..., c], 0.1))
        sm = poly(ndi.gaussian_filter(med, 6), 4) if ch == "DAPI" else \
            ndi.gaussian_filter(med, 6)
        ctr = sm[H // 2 - 9:H // 2 + 9, H // 2 - 9:H // 2 + 9].mean()
        out[ch] = np.clip((sm - off) / (ctr - off), 0.25, None).astype(np.float32)
        offsets[ch] = off
    return out, offsets


def _vignetted_fields(n=12, size=320, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    r2 = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) / (size / 2) ** 2
    light = 1.0 - 0.4 * r2
    fields = []
    for _ in range(n):
        signal = rng.gamma(2.0, 200.0, (size, size, 2))
        fields.append((300 + light[..., None] * signal).astype(np.float32))
    return fields


def test_flat_maps_match_the_pilot_method_and_are_never_overwritten(tmp_path):
    channels = ("DAPI", "GFP")
    blocks = np.stack([field_blocks(f) for f in _vignetted_fields()])
    assert blocks.shape == (12, 20, 20, 2)
    maps, method = build_flat_maps(blocks, channels)
    reference, offsets = _pilot_build(blocks, channels)
    for ch in channels:
        assert np.array_equal(maps[ch], reference[ch])
        assert method[ch]["offset"] == offsets[ch]
        assert maps[ch].min() >= 0.25
    fields = [("A1", 3), ("A2", 5)]
    info = write_flat_maps(str(tmp_path / "flat"), PLATE, maps, method, fields)
    assert set(info["md5"]) == set(channels)
    assert info["fields_by_well"] == {"A1": [3], "A2": [5]}
    loaded, loaded_offsets, loaded_info = load_flat_maps(
        str(tmp_path / "flat"), ["dapi", "GFP"])
    assert np.array_equal(loaded["dapi"], maps["DAPI"])
    assert loaded_offsets["GFP"] == method["GFP"]["offset"]
    assert loaded_info["md5"] == info["md5"]
    with pytest.raises(FileExistsError, match="never overwritten"):
        write_flat_maps(str(tmp_path / "flat"), PLATE, maps, method, fields)


def test_flat_fields_are_a_seeded_draw_in_pool_order():
    pool = [("A1", s) for s in range(50)] + [("A2", s) for s in range(50)]
    first = choose_flat_fields(pool, 10)
    assert first == choose_flat_fields(pool, 10)
    assert first == sorted(first, key=pool.index)
    rng = np.random.default_rng(0)
    assert first == [pool[i] for i in sorted(rng.choice(100, 10, replace=False))]
    assert len(choose_flat_fields(pool[:4], 10)) == 4


def test_exact_correction_divides_out_the_flat_and_keeps_the_offset():
    flat = {"DAPI": np.full((4, 4), 0.5, np.float32)}
    planes = flat_planes(flat, (64, 64))
    assert planes["DAPI"].shape == (64, 64)
    stack = np.full((64, 64, 1), 300.0, np.float32)
    stack[..., 0] += 0.5 * 100.0
    out = correct_field(stack, planes, {"DAPI": 300.0}, ["DAPI"])
    assert np.allclose(out, 400.0)
    assert np.allclose(stack, 350.0)  # the input is not changed


def _blocky_field(seed=0):
    """Cells inside 40 px blocks whose flat is constant per block."""
    rng = np.random.default_rng(seed)
    size = 240
    nuclei = np.zeros((size, size), np.int32)
    cells = np.zeros((size, size), np.int32)
    flat = np.zeros((size, size), np.float32)
    label = 0
    for by in range(0, size, 40):
        for bx in range(0, size, 40):
            flat[by:by + 40, bx:bx + 40] = rng.uniform(0.4, 1.1)
            if by == 0 and bx == 0:
                continue  # one empty block
            label += 1
            cells[by + 4:by + 36, bx + 5:bx + 34] = label
            nuclei[by + 12:by + 26, bx + 12:bx + 25] = label
    cells[0:20, 200:204] = 99  # a cell cut by the top edge
    nuclei[0:6, 200:204] = 99
    true = rng.gamma(3.0, 150.0, (size, size, 2)).astype(np.float32)
    offset = {"DAPI": 300.0, "GFP": 320.0}
    observed = np.stack([offset[c] + flat * true[..., i]
                         for i, c in enumerate(("DAPI", "GFP"))], -1)
    return nuclei, cells, observed.astype(np.float32), flat, offset


def test_mask_pass_gives_edge_flags_counts_and_gains():
    nuclei, cells, _image, flat, _offset = _blocky_field()
    table = mask_pass(nuclei, cells, {"DAPI": 1.0 / flat.astype(np.float64)})
    row = table.set_index("label")
    assert bool(row.loc[99, "touches_field_edge"])
    assert bool(row.loc[99, "nucleus_touches_field_edge"])
    assert not row.drop(index=99)["touches_field_edge"].any()
    assert row.loc[1, "cell_px"] == 32 * 29
    assert row.loc[1, "nucleus_px"] == 14 * 13
    assert row.loc[1, "cytoplasm_px"] == 32 * 29 - 14 * 13
    assert np.isclose(row.loc[1, "cell_DAPI_g"], 1.0 / flat[4, 45])
    assert np.isclose(row.loc[1, "nucleus_cy"], 12 + 6.5)
    assert set(mask_pass(nuclei, cells).columns).isdisjoint({"cell_DAPI_g"})
    assert len(mask_pass(np.zeros((5, 5)), np.zeros((5, 5)))) == 0


def test_analytic_correction_equals_the_exact_one_where_the_flat_is_constant():
    nuclei, cells, image, flat, offset = _blocky_field()
    channels = ["DAPI", "GFP"]
    raw = measure_phenotype_field(nuclei, cells, image, channels,
                                  texture=False)
    planes = {c: flat for c in channels}
    exact = measure_phenotype_field(
        nuclei, cells, correct_field(image, planes, offset, channels),
        channels, texture=False)
    gains = mask_pass(nuclei, cells, {c: 1.0 / flat.astype(np.float64)
                                      for c in channels})
    raw = raw.merge(gains, on="label")
    features = [c for c in ops_engine._ph_feature_columns(raw.columns)
                if correction_kind(c, channels) and c in exact]
    kinds = {correction_kind(c, channels) for c in features}
    assert kinds == {"loc", "spread", "int", "cv", "gini"}
    analytic = correct_features(raw, raw, features, offset)
    exact = exact.set_index("label").loc[raw["label"]].reset_index(drop=True)
    for column in features:
        if column.endswith("_mode_intensity"):
            continue
        assert np.allclose(analytic[column], exact[column], rtol=2e-5,
                           atol=1e-3), column
    assert flagged_features(features, channels) == [
        c for c in features if c.endswith("_mode_intensity")]
    untouched = "nucleus_area"
    assert correction_kind(untouched, channels) is None


def test_correction_inputs_add_the_mean_and_std_cv_and_gini_need():
    assert correction_inputs(["cell_dapi_cv_intensity"], ["DAPI"]) == [
        "cell_dapi_cv_intensity", "cell_dapi_mean_intensity",
        "cell_dapi_std_intensity"]
    assert correction_kind("cell_dapi_mass_displacement", ["DAPI"]) is None
    assert correction_kind("cell_correlation_dapi_gfp", ["DAPI", "GFP"]) is None


def test_options_marker_refuses_to_mix_parts(tmp_path):
    folder = str(tmp_path)
    ops_engine._ph_options_marker(folder, {"flatfield": None})
    ops_engine._ph_options_marker(folder, {"flatfield": None})
    with pytest.raises(ValueError, match="holds parts made with"):
        ops_engine._ph_options_marker(folder, {"flatfield": {"DAPI": "x"}})
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "site-1.parquet").write_bytes(b"")
    ops_engine._ph_options_marker(str(legacy), {"flatfield": None,
                                                "funk_features": False})
    other = tmp_path / "other"
    other.mkdir()
    (other / "site-1.parquet").write_bytes(b"")
    with pytest.raises(ValueError):
        ops_engine._ph_options_marker(str(other), {"funk_features": True})


def test_an_unknown_flatfield_mode_is_refused():
    with pytest.raises(ValueError, match="ops_flatfield must be one of"):
        ops_engine._ph_flatfield_mode({"ops_flatfield": "basic"})


# --- 852 + 853 in the aggregate phase ------------------------------------------

def test_aggregate_leaves_out_cut_cells_and_corrects_intensities(tmp_path):
    from tests.test_ops_aggregate_blockwise import _old_aggregate, _plate

    (tmp_path / "new").mkdir()
    db, wells, _ = _plate(tmp_path / "new")
    reference = tmp_path / "ref"
    reference.mkdir()
    channels = ["DAPI", "GFP"]
    maps = {c: np.ones((4, 4), np.float32) for c in channels}
    method = {"DAPI": {"offset": 300.0}, "GFP": {"offset": 20.0}}
    folder = str(tmp_path / "flat")
    write_flat_maps(folder, PLATE, maps, method, [("A1", 0)])
    offsets = {"DAPI": 300.0, "GFP": 20.0}
    rng = np.random.default_rng(3)
    for well in wells:
        cells = pd.read_parquet(tmp_path / "new" /
                                f"ops_phenotype_cells_{well}.parquet")
        passed = cells[["site", "label"]].copy()
        passed["touches_field_edge"] = rng.random(len(cells)) < 0.1
        for comp in ("cell", "nucleus", "cytoplasm"):
            passed[f"{comp}_px"] = rng.integers(50, 400, len(cells)).astype(float)
            for ch in channels:
                passed[f"{comp}_{ch}_g"] = rng.uniform(0.9, 2.5, len(cells))
        passed.insert(0, "well", well)
        passed.insert(0, "plate", PLATE)
        passed.to_parquet(ops_engine._ph_maskpass_path(str(tmp_path / "new"),
                                                       well))
        keep = ~passed["touches_field_edge"].to_numpy()
        features = ops_engine._ph_feature_columns(cells.columns)
        fixed = correct_features(cells, passed, features, offsets)
        out = cells.copy()
        out[features] = fixed[features].to_numpy()
        out[keep].to_parquet(reference / f"ops_phenotype_cells_{well}.parquet")
    settings = {"ops_design": str(tmp_path / "new" / "design.csv"),
                "ops_ph_channels": "DAPI,GFP", "ops_bootstrap": 0,
                "ops_feature_qc": False}
    want_guides, want_genes = _old_aggregate(db, PLATE, wells, settings,
                                             str(reference))
    with pytest.raises(ValueError, match="ph_masks"):
        ops_engine._ph_aggregate(db, PLATE, wells, {
            **settings, "ops_ownership": "a1b"}, str(tmp_path / "ref"))
    report = ops_engine._ph_aggregate(db, PLATE, wells, {
        **settings, "ops_ownership": "a1b", "ops_flatfield": "analytic",
        "ops_flatfield_maps": folder}, str(tmp_path / "new"))
    guides = pd.read_parquet(report["outputs"]["guides_parquet"])
    genes = pd.read_parquet(report["outputs"]["genes_parquet"])
    pd.testing.assert_frame_equal(guides.reset_index(drop=True),
                                  want_guides.reset_index(drop=True),
                                  check_dtype=False, rtol=1e-6)
    pd.testing.assert_frame_equal(genes.reset_index(drop=True),
                                  want_genes.reset_index(drop=True),
                                  check_dtype=False, rtol=1e-6)
    assert report["ownership"]["cells_cut_by_field_edge"] > 0
    assert report["flatfield"]["mode"] == "analytic"
    assert report["flatfield"]["offsets"] == offsets


# --- 857: Funk et al.'s features ------------------------------------------------

def test_mass_displacement_and_correlation_have_their_closed_forms():
    labels = np.zeros((6, 8), np.int64)
    labels[1, 1] = labels[1, 3] = 1          # two pixels, one row apart: 2 px
    labels[3:5, 4:8] = 2                     # a 2 x 4 block
    image = np.zeros((6, 8, 3), np.float64)
    image[1, 1, 0], image[1, 3, 0] = 1.0, 3.0
    image[1, 1, 1], image[1, 3, 1] = 5.0, 5.0
    ramp = np.arange(8, dtype=float)[None, :].repeat(6, 0)
    image[3:5, 4:8, 0] = ramp[3:5, 4:8]
    image[3:5, 4:8, 1] = 2 * ramp[3:5, 4:8] + 1
    image[3:5, 4:8, 2] = -ramp[3:5, 4:8]
    out = funk_object_features(labels, image, ["DAPI", "GFP", "A594"],
                               "cell").set_index("label")
    # label 1: centroid column 2, weighted (1*1 + 3*3) / 4 = 2.5
    assert np.isclose(out.loc[1, "cell_dapi_mass_displacement"], 0.5)
    assert np.isclose(out.loc[1, "cell_gfp_mass_displacement"], 0.0)
    assert np.isnan(out.loc[1, "cell_correlation_gfp_a594"])  # no spread
    # label 2: weighted column sum(x^2) / sum(x) over x = 4..7
    x = np.arange(4, 8, dtype=float)
    assert np.isclose(out.loc[2, "cell_dapi_mass_displacement"],
                      (x ** 2).sum() / x.sum() - x.mean())
    assert np.isclose(out.loc[2, "cell_correlation_dapi_gfp"], 1.0)
    assert np.isclose(out.loc[2, "cell_correlation_dapi_a594"], -1.0)
    assert list(out.columns) == [
        "cell_dapi_mass_displacement", "cell_gfp_mass_displacement",
        "cell_a594_mass_displacement", "cell_correlation_dapi_gfp",
        "cell_correlation_dapi_a594", "cell_correlation_gfp_a594"]


def test_measure_adds_funk_features_only_when_asked():
    from skimage.measure import regionprops

    nuclei, cells, image, _flat, _offset = _blocky_field()
    plain = measure_phenotype_field(nuclei, cells, image, ["DAPI", "GFP"],
                                    texture=False)
    funk = measure_phenotype_field(nuclei, cells, image, ["DAPI", "GFP"],
                                   texture=False, funk_features=True)
    added = sorted(set(funk.columns) - set(plain.columns))
    assert added == sorted([
        "nucleus_dapi_mass_displacement", "nucleus_gfp_mass_displacement",
        "nucleus_correlation_dapi_gfp", "nucleus_form_factor",
        "cell_dapi_mass_displacement", "cell_gfp_mass_displacement",
        "cell_correlation_dapi_gfp", "cell_form_factor"])
    pd.testing.assert_frame_equal(funk[plain.columns], plain)
    props = {r.label: r for r in regionprops(nuclei)}
    for row in funk.itertuples():
        r = props[row.label]
        assert np.isclose(row.nucleus_form_factor,
                          4 * np.pi * r.area / r.perimeter ** 2)
    assert set(added) <= set(ops_engine._ph_feature_columns(funk.columns))
