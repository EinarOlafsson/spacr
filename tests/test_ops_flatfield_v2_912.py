"""Item 912: a DNA flat-field map from per-nucleus intensities.

A synthetic plate has a known steep vignette, nuclei of one known brightness
and a background that falls less steeply than the nuclei (stray light). The
nuclear-median map must recover the corners; the pixel median must not; a
map read at the wrong place must fail the corner ring; a changed map must
refuse a resumed run.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from spacr import ops_flatfield as F
from spacr.ops_engine import _ph_flat_planes, run_ops

SIZE = 20
BLOCK = 16
FIELD = SIZE * BLOCK
OFFSET = 100.0
TRUE = 1000.0
CENTRE = FIELD / 2


def vignette(y, x):
    """The true illumination: 1 at the centre, about 0.2 in the corners."""
    r2 = ((np.asarray(y, float) - CENTRE) ** 2 + (np.asarray(x, float) - CENTRE) ** 2) / (FIELD / 2) ** 2
    return np.exp(-0.8 * r2)


def nucleus_frame(fields=60, per_field=120, seed=0):
    """Per-nucleus table: centroids, labels and the measured nuclear mean."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(fields):
        well = "A1" if f % 2 else "B1"
        cy = rng.uniform(2, FIELD - 3, per_field)
        cx = rng.uniform(2, FIELD - 3, per_field)
        rows.append(pd.DataFrame({"well": well, "site": f, "label": np.arange(1, per_field + 1),
                                  "nucleus_cy": cy, "nucleus_cx": cx,
                                  "nucleus_dapi_mean_intensity": OFFSET + vignette(cy, cx) * TRUE}))
    return pd.concat(rows, ignore_index=True)


def field_image(seed):
    """One field: a stray-light background and square nuclei on a grid."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:FIELD, :FIELD]
    image = OFFSET + np.sqrt(vignette(yy, xx)) * 60.0
    labels = np.zeros((FIELD, FIELD), np.int32)
    label = 0
    for y in range(3, FIELD - 4, 9):
        for x in range(3, FIELD - 4, 9):
            if rng.random() < 0.5:
                continue
            label += 1
            labels[y:y + 4, x:x + 4] = label
            image[y:y + 4, x:x + 4] = OFFSET + vignette(yy[y:y + 4, x:x + 4], xx[y:y + 4, x:x + 4]) * TRUE
    return image, labels


def corner_error(flat, lookup="zoom_endpoints", centre_offset=8.0):
    """Largest relative error of the map over the true vignette, outer ring between the block centres."""
    plane = F.flat_to_field(flat, (FIELD, FIELD), lookup=lookup, block=BLOCK, centre_offset=centre_offset)
    yy, xx = np.mgrid[:FIELD, :FIELD]
    centres = np.arange(SIZE) * BLOCK + 8
    scale = vignette(centres[:, None], centres[None, :])[SIZE // 2 - 2:SIZE // 2 + 2, SIZE // 2 - 2:SIZE // 2 + 2].mean()
    truth = vignette(yy, xx) / scale
    r = np.hypot(yy - CENTRE, xx - CENTRE)
    inside = (yy >= 8) & (yy <= FIELD - 9) & (xx >= 8) & (xx <= FIELD - 9)
    ring = (r > 0.75 * FIELD / 2) & (truth > 0.25) & inside
    return float(np.max(np.abs(plane[ring] / truth[ring] - 1)))


def test_nuclear_median_map_recovers_the_corners_and_the_pixel_median_does_not():
    nuclei = nucleus_frame(fields=300)
    flat, info = F.build_nuclear_flat(nuclei, column="nucleus_dapi_mean_intensity", offset=OFFSET,
                                      field_px=FIELD, centre_cells=4)
    assert flat.dtype == np.float32 and flat.shape == (SIZE, SIZE)
    assert info["estimator"] == "nuclear_median" and info["n_training_nuclei"] == len(nuclei)
    assert corner_error(flat, "block_centre") < 0.03
    image, _labels = field_image(1)
    pixel = F.field_blocks(np.stack([field_image(s)[0] for s in range(5)])[..., None].mean(0))[..., 0]
    pixel_flat, _centre = F.normalise_flat(pixel, OFFSET, floor=0.10, centre_cells=4)
    assert corner_error(pixel_flat, "block_centre") > 0.2


def test_zoom_lookup_fails_the_corner_and_block_centre_lookup_passes():
    nuclei = nucleus_frame(fields=300, seed=3)
    flat, _ = F.build_nuclear_flat(nuclei, column="nucleus_dapi_mean_intensity", offset=OFFSET,
                                   field_px=FIELD, centre_cells=4, map_grid="block_centre")
    assert corner_error(flat, "block_centre") < 0.03
    assert corner_error(flat, "zoom_endpoints") > 0.02
    lookup_map, _ = F.build_nuclear_flat(nuclei, column="nucleus_dapi_mean_intensity", offset=OFFSET,
                                         field_px=FIELD, centre_cells=4, map_grid="lookup")
    assert corner_error(lookup_map, "zoom_endpoints") < corner_error(flat, "zoom_endpoints")
    t = F.lookup_positions(SIZE, FIELD, block=BLOCK, centre_offset=8.0)
    assert t[0] == pytest.approx(-0.5) and t[-1] == pytest.approx((FIELD - 1 - 8) / BLOCK)


@pytest.mark.parametrize("fit,sigma", [("gaussian", 16), ("local_linear", 32), ("local_quadratic_log", 48)])
def test_every_fit_returns_a_normalised_floored_map(fit, sigma):
    median, count = F.nuclear_median_grid(*[nucleus_frame(fields=10)[c] for c in
                                            ("nucleus_cy", "nucleus_cx", "nucleus_dapi_mean_intensity")],
                                          block=BLOCK, size=SIZE)
    median[0, 0], count[0, 0] = np.nan, 0
    flat, info = F.fit_nuclear_flat(median, count, OFFSET, fit=fit, sigma_px=sigma, centre_cells=4, floor=0.3)
    assert flat.min() >= 0.3 - 1e-7
    assert flat[8:12, 8:12].mean() == pytest.approx(1.0, abs=0.02)
    assert info["ridge"] == (None if fit == "gaussian" else F._DEFAULT_RIDGE[fit])


def test_gaussian_fit_fills_cells_out_of_reach_from_the_nearest():
    median = np.full((SIZE, SIZE), np.nan)
    count = np.zeros((SIZE, SIZE))
    median[8:12, 8:12], count[8:12, 8:12] = 600.0, 5
    flat, info = F.fit_nuclear_flat(median, count, OFFSET, fit="gaussian", sigma_px=8, centre_cells=4)
    assert info["filled_cells"] > 0 and np.isfinite(flat).all()


def test_bad_fit_grid_and_lookup_are_refused():
    median, count = np.ones((4, 4)), np.ones((4, 4))
    with pytest.raises(ValueError, match="fit must be one of"):
        F.fit_nuclear_flat(median, count, 0.0, fit="spline")
    with pytest.raises(ValueError, match="map grid"):
        F.fit_nuclear_flat(median, count, 0.0, map_grid="corners")
    with pytest.raises(ValueError, match="no lookup grid"):
        F.fit_nuclear_flat(median, count, 0.0, fit="gaussian", map_grid="lookup")
    with pytest.raises(ValueError, match="lookup must be one of"):
        F.flat_to_field(median, (8, 8), lookup="nearest")


def test_build_refuses_without_offset_columns_or_nuclei():
    nuclei = nucleus_frame(fields=2)
    with pytest.raises(ValueError, match="camera offset"):
        F.build_nuclear_flat(nuclei, column="nucleus_dapi_mean_intensity", offset=None)
    with pytest.raises(ValueError, match="lacks"):
        F.build_nuclear_flat(nuclei.drop(columns="site"), column="nucleus_dapi_mean_intensity", offset=1.0)
    with pytest.raises(ValueError, match="no nucleus is left"):
        F.build_nuclear_flat(nuclei, column="nucleus_dapi_mean_intensity", offset=OFFSET,
                             holdout=[("A1", 1), ("B1", 0)])


def test_build_takes_the_field_size_from_the_centroids_and_leaves_held_out_fields_out():
    nuclei = nucleus_frame(fields=6)
    flat, info = F.build_nuclear_flat(nuclei, column="nucleus_dapi_mean_intensity", offset=OFFSET,
                                      holdout=[("A1", 1)], centre_cells=4)
    assert info["field_px"] == FIELD and info["size"] == SIZE
    assert info["n_heldout_nuclei"] == 120 and info["held_out_fields"] == ["A1:1"]
    assert set(info["versions"]) == {"numpy", "scipy", "pandas"}


def test_holdout_fields_draws_the_same_number_per_well():
    fields = [(w, s) for w in ("A1", "A2", "B1") for s in range(10)]
    held = F.holdout_fields(fields, 6, seed=0)
    assert len(held) == 6 and {w for w, _ in held} == {"A1", "A2", "B1"}
    assert held == F.holdout_fields(fields, 6, seed=0) and F.holdout_fields([], 4) == []


def test_flat_planes_reads_old_maps_by_zoom_and_new_maps_by_their_record():
    flat = np.linspace(0.5, 1.0, 16, dtype=np.float32).reshape(4, 4)
    zoomed = F.flat_planes({"DAPI": flat}, (64, 64))["DAPI"]
    from scipy import ndimage

    assert np.array_equal(zoomed, ndimage.zoom(flat, 16.0, order=1)[:64, :64])
    lookups = F.flat_lookups({"method": {"DAPI": {"lookup": "block_centre", "block": 16, "centre_offset": 8.0},
                                         "GFP": {"offset": 1.0}}})
    assert lookups == {"DAPI": {"lookup": "block_centre", "block": 16, "centre_offset": 8.0}}
    centred = F.flat_planes({"DAPI": flat}, (64, 64), lookups=lookups)["DAPI"]
    assert centred.dtype == np.float32
    assert centred[8, 8] == pytest.approx(flat[0, 0]) and centred[24, 40] == pytest.approx(flat[1, 2])
    assert centred[0, 0] == pytest.approx(flat[0, 0])
    assert F.flat_lookups(None) == {}


def _base_maps(folder):
    """A pre-912 maps folder with DAPI and GFP."""
    maps = {"DAPI": np.full((SIZE, SIZE), 0.9, np.float32), "GFP": np.full((SIZE, SIZE), 0.8, np.float32)}
    method = {"DAPI": {"offset": OFFSET}, "GFP": {"offset": 50.0}}
    return F.write_flat_maps(str(folder), "P1", maps, method, [("A1", 1)])


def _run(tmp_path, nuclei=None, **extra):
    """A synthetic run folder with nuclei, held-out images and masks."""
    root = tmp_path / "run"
    images = tmp_path / "images"
    images.mkdir()
    nuclei = nucleus_frame(fields=12) if nuclei is None else nuclei
    held = [("A1", 1), ("B1", 2)]
    for well, site in held:
        image, labels = field_image(site)
        np.save(images / f"dapi_{well}_{site}.npy", image.astype(np.float32))
        masks = root / well / "phenotype_masks"
        masks.mkdir(parents=True, exist_ok=True)
        np.savez(masks / f"site-{site}.npz", nuclei=labels, cells=labels)
        lab = np.arange(1, labels.max() + 1)
        cy = np.array([np.argwhere(labels == k)[:, 0].mean() for k in lab])
        cx = np.array([np.argwhere(labels == k)[:, 1].mean() for k in lab])
        means = np.array([image[labels == k].mean() for k in lab])
        nuclei = nuclei[~((nuclei.well == well) & (nuclei.site == site))]
        nuclei = pd.concat([nuclei, pd.DataFrame({"well": well, "site": site, "label": lab, "nucleus_cy": cy,
                                                  "nucleus_cx": cx, "nucleus_dapi_mean_intensity": means,
                                                  "nucleus_DAPI_g": 1 / 0.9})], ignore_index=True)
    table = tmp_path / "nuclei.parquet"
    nuclei.to_parquet(table)
    base = tmp_path / "base"
    _base_maps(base)
    settings = {"plate": "P1", "dst_root": str(root), "ops_flatfield_nuclear_channel": "DAPI",
                "ops_flatfield_nuclei": str(table), "ops_flatfield_maps": str(base),
                "ops_flatfield_holdout_fields": ["A1:1", "B1:2"], "ops_flatfield_holdout_images": str(images),
                "ops_flatfield_centre_cells": 4, "ops_flatfield_radial_edges": [0, 60, 120, 160, 230],
                "ops_flatfield_bootstrap": 20, "ops_flatfield_min_bin_cells": 5, **extra}
    return settings


def test_settings_run_builds_validates_and_records_the_map(tmp_path):
    settings = _run(tmp_path, ops_flatfield_compare_maps={"flat_one": str(tmp_path / "one.npy")})
    np.save(tmp_path / "one.npy", np.ones((SIZE, SIZE), np.float32))
    result = run_ops(settings, phases=["flat_nuclear", "flat_validate"])
    record = result["plates"]["P1"]
    folder = record["folder"]
    assert sorted(n for n in os.listdir(folder) if n.endswith(".npy")) == ["flat_P1_DAPI.npy", "flat_P1_GFP.npy"]
    info = json.load(open(os.path.join(folder, "flat_info.json")))
    assert info["md5_record"] and info["method"]["DAPI"]["lookup"] == "block_centre"
    assert info["method"]["GFP"] == {"offset": 50.0}
    summary = record["validation"]
    assert summary["exact_nuclear"]["max_abs_dev_bins_ge100_cells"] < 0.03
    assert summary["exact_earlier"]["max_abs_dev_bins_ge100_cells"] > 0.2
    assert summary["as measured"]["max_abs_dev_bins_ge100_cells"] > 0.2
    assert summary["sanity"]["max_abs_diff_gain_vs_run"] < 1e-6
    assert "same held-out nuclei" in summary["matching"]
    radial = pd.read_csv(os.path.join(folder, "validation", "radial.csv"))
    assert {"exact_flat_one", "analytic_nuclear", "as measured"} <= set(radial.version)
    with pytest.raises(FileExistsError):
        run_ops(settings, phases=["flat_nuclear"])


def test_a_changed_map_refuses_a_resumed_run(tmp_path):
    settings = _run(tmp_path)
    run_ops(settings, phases=["flat_nuclear"])
    folder = F._nuclear_folder(settings, "P1")
    maps, _offsets, info = F.load_flat_maps(folder, ["DAPI"])
    first = F.record_flat_md5(settings["dst_root"], info)
    assert first["md5"]["DAPI"] == info["md5"]["DAPI"]
    assert F.record_flat_md5(settings["dst_root"], info) == first
    assert F.record_flat_md5(settings["dst_root"], {"md5": {"DAPI": "x"}}) is None
    np.save(os.path.join(folder, "flat_P1_DAPI.npy"), maps["DAPI"] * 0.5)
    changed = dict(info, md5={"DAPI": "0" * 32, "GFP": info["md5"]["GFP"]})
    with pytest.raises(ValueError, match="changed since this run first used it"):
        F.record_flat_md5(settings["dst_root"], changed)
    extra = dict(info, md5={"AF750": "1" * 32})
    assert F.record_flat_md5(settings["dst_root"], extra)["md5"]["AF750"] == "1" * 32


def test_engine_reads_a_new_map_at_block_centres(tmp_path):
    settings = _run(tmp_path)
    run_ops(settings, phases=["flat_nuclear"])
    folder = F._nuclear_folder(settings, "P1")
    planes, offsets, inverse = _ph_flat_planes(folder, ("DAPI", "GFP"), (FIELD, FIELD))
    flat = np.load(os.path.join(folder, "flat_P1_DAPI.npy"))
    assert planes["DAPI"][8, 8] == pytest.approx(flat[0, 0])
    assert offsets["DAPI"] == OFFSET and inverse["GFP"][0, 0] == pytest.approx(1 / 0.8)


def test_run_tables_are_joined_when_no_table_is_given(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    pd.DataFrame({"plate": "P", "well": "A1", "site": [1, 1], "label": [1, 2], "nucleus_cy": [5.0, 9.0],
                  "nucleus_cx": [5.0, 9.0], "nucleus_DAPI_g": [1.0, 1.1]}).to_parquet(
        root / "ops_maskpass_A1.parquet")
    pd.DataFrame({"site": [1, 1], "label": [1, 2], "nucleus_dapi_mean_intensity": [5.0, 6.0]}).to_parquet(
        root / "ops_phenotype_cells_A1.parquet")
    pd.DataFrame({"site": [1], "label": [1]}).to_parquet(root / "ops_maskpass_B1.parquet")
    table = F.nucleus_table({"dst_root": str(root)}, "P", "DAPI")
    assert list(table.well) == ["A1", "A1"] and "nucleus_DAPI_g" in table
    with pytest.raises(ValueError, match="no per-nucleus table"):
        F.nucleus_table({"dst_root": str(tmp_path / "empty")}, "P", "DAPI")
    pd.DataFrame({"well": ["A1"]}).to_parquet(tmp_path / "bad.parquet")
    with pytest.raises(ValueError, match="lacks"):
        F.nucleus_table({"ops_flatfield_nuclei": {"P": str(tmp_path / "bad.parquet")}}, "P", "DAPI")
    folder = tmp_path / "tables"
    folder.mkdir()
    pd.DataFrame({"well": ["A1"], "site": [1], "label": [1], "nucleus_cy": [1.0], "nucleus_cx": [1.0],
                  "nucleus_dapi_mean_intensity": [3.0]}).to_parquet(folder / "a.parquet")
    assert len(F.nucleus_table({"ops_flatfield_nuclei": str(folder)}, "P", "DAPI")) == 1


def test_flat_phases_refuse_bad_settings(tmp_path):
    settings = _run(tmp_path)
    with pytest.raises(ValueError, match="ops_flatfield_nuclear_channel"):
        run_ops(dict(settings, ops_flatfield_nuclear_channel=None), phases=["flat_nuclear"])
    with pytest.raises(ValueError, match="camera offset"):
        run_ops(dict(settings, ops_flatfield_maps=None), phases=["flat_nuclear"])
    with pytest.raises(ValueError, match="ops_flatfield_lookup"):
        run_ops(dict(settings, ops_flatfield_lookup="edge"), phases=["flat_nuclear"])
    with pytest.raises(ValueError, match="run on their own"):
        run_ops(settings, phases=["flat_nuclear", "ph_masks"])
    built = dict(settings, ops_flatfield_dark={"P1": OFFSET}, ops_flatfield_holdout_fields=None,
                 ops_flatfield_holdout_n=2)
    run_ops(built, phases=["flat_nuclear"])
    with pytest.raises(ValueError, match="holdout_images"):
        run_ops(dict(built, ops_flatfield_holdout_images=None), phases=["flat_validate"])
    with pytest.raises(FileNotFoundError, match="no image of held-out field"):
        F._field_image(str(tmp_path), "A9", 9)


def test_listed_holdout_fields_may_be_one_string(tmp_path):
    settings = _run(tmp_path, ops_flatfield_holdout_fields="A1:1; B1:2", ops_flatfield_dark=OFFSET,
                    ops_flatfield_maps=None)
    record = run_ops(settings, phases=["flat_nuclear", "flat_validate"])["plates"]["P1"]
    assert record["held_out"] == 2 and "exact_earlier" not in record["validation"]


def test_validate_refuses_fields_outside_the_table():
    image, labels = field_image(0)
    with pytest.raises(ValueError, match="no held-out nucleus"):
        F.validate_nuclear_flat(nucleus_frame(fields=1), {("Z9", 1): (image, labels)},
                                {"one": np.ones((SIZE, SIZE))}, column="nucleus_dapi_mean_intensity",
                                offset=OFFSET)
