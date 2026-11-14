"""Item 933: the OPS nuclear-median flat-field in spaCR's general illumination correction.

The general path must build the map with the OPS functions themselves (one
implementation), give OPS map bytes and OPS corrected pixels for the same
nuclei and options, leave every existing estimator unchanged, and refuse
what it cannot do honestly: no offset, a nucleus table that was already
corrected, and the mask stage that has to find the nuclei first.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sqlite3
import types

import numpy as np
import pandas as pd
import pytest

from spacr import illumination as ill
from spacr import ops_flatfield as F

SIZE = 20
BLOCK = 16
FIELD = SIZE * BLOCK
OFFSET = 100.0
TRUE = 1000.0
CENTRE = FIELD / 2


def vignette(y, x, shape=(FIELD, FIELD)):
    """1 at the centre, about 0.2 in the corners."""
    cy, cx = shape[0] / 2, shape[1] / 2
    r2 = ((np.asarray(y, float) - cy) ** 2 + (np.asarray(x, float) - cx) ** 2) / (max(shape) / 2) ** 2
    return np.exp(-0.8 * r2)


def nucleus_frame(fields=80, per_field=120, seed=0, shuffle=False):
    """One row per nucleus: plate, well, site, centroid, raw nuclear mean."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(fields):
        cy = rng.uniform(2, FIELD - 3, per_field)
        cx = rng.uniform(2, FIELD - 3, per_field)
        value = OFFSET + vignette(cy, cx) * TRUE
        if shuffle:
            order = rng.permutation(per_field)
            cy, cx = cy[order], cx[order]
        rows.append(pd.DataFrame({
            "plate": "plateX", "well": "A1" if f % 2 else "B1", "site": f,
            "label": np.arange(1, per_field + 1), "nucleus_cy": cy, "nucleus_cx": cx,
            "nucleus_channel_0_mean_intensity": value}))
    return pd.concat(rows, ignore_index=True)


def merged_plate(folder, fields=4, shape=(FIELD, FIELD), seed=1):
    """Merged fields: DNA plane 0, a second stain plane 1, nucleus masks plane 2."""
    rng = np.random.default_rng(seed)
    os.makedirs(folder, exist_ok=True)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    vig = vignette(yy, xx, shape)
    for f in range(fields):
        stack = np.zeros(shape + (3,), np.float32)
        stack[..., 0] = OFFSET + np.sqrt(vig) * 60.0
        stack[..., 1] = OFFSET + vig * 400.0 + rng.normal(0, 3, shape)
        label = 0
        for y in range(3, shape[0] - 6, 9):
            for x in range(3, shape[1] - 6, 9):
                if rng.random() < 0.4:
                    continue
                label += 1
                stack[y:y + 4, x:x + 4, 2] = label
                stack[y:y + 4, x:x + 4, 0] = OFFSET + vig[y:y + 4, x:x + 4] * TRUE
        well = "A01" if f % 2 else "B01"
        np.save(os.path.join(folder, f"plateX_{well}_{f}_1.npy"), stack)
    return folder


def md5(array):
    """md5 of an array's bytes."""
    return hashlib.md5(np.ascontiguousarray(array).tobytes()).hexdigest()


def options(**extra):
    """The OPS defaults of build_nuclear_flat, plus overrides."""
    base = {"block": BLOCK, "fit": "local_quadratic_log", "sigma_px": 32.0, "ridge": None,
            "floor": 0.10, "centre_cells": 18, "map_grid": "block_centre",
            "centre_offset": 8.0, "lookup": "block_centre"}
    base.update(extra)
    return base


def test_the_general_estimator_calls_the_ops_functions_not_a_copy():
    import inspect

    source = inspect.getsource(ill._nuclear_planes)
    assert "from .ops_flatfield import build_nuclear_flat, flat_planes" in source
    for name in ("nuclear_median_grid", "fit_nuclear_flat", "normalise_flat", "_local_fit"):
        assert f"def {name}(" not in inspect.getsource(ill)


@pytest.mark.parametrize("grid", ["block_centre", "lookup"])
def test_the_general_map_is_the_ops_map_byte_for_byte(tmp_path, grid):
    nuclei = nucleus_frame()
    src = merged_plate(str(tmp_path / "merged"), fields=2)
    held = F.holdout_fields(sorted({(w, int(s)) for w, s in zip(nuclei.well, nuclei.site)}), 6, 0)
    model = ill.estimate_nuclear_illumination(
        src, [0], nuclear_channel=0, dark_per_plate={"plateX": OFFSET}, nuclei=nuclei,
        options=options(map_grid=grid, holdout=held), verbose=False)
    item = model.field_for("plateX")
    ops_flat, ops_info = F.build_nuclear_flat(
        nuclei.rename(columns={"nucleus_channel_0_mean_intensity": "nucleus_dapi_mean_intensity"}),
        column="nucleus_dapi_mean_intensity", offset=OFFSET, holdout=held, field_px=FIELD,
        **options(map_grid=grid))
    assert md5(item.block_maps[0]) == md5(ops_flat)
    assert item.map_records[0]["n_heldout_nuclei"] == ops_info["n_heldout_nuclei"] > 0
    ops_plane = F.flat_planes({"DAPI": ops_flat}, (FIELD, FIELD),
                              F.flat_lookups({"method": {"DAPI": {"lookup": "block_centre"}}}))["DAPI"]
    assert np.array_equal(item.flatfield[0], ops_plane)


def test_the_settings_path_gives_the_map_the_ops_settings_give(tmp_path):
    from spacr.ops_settings import ops_defaults

    nuclei = nucleus_frame()
    table = tmp_path / "nuclei.parquet"
    nuclei.to_parquet(table)
    src = merged_plate(str(tmp_path / "merged"), fields=2)
    ops = ops_defaults({"ops_flatfield_dark": OFFSET, "ops_flatfield_map_grid": "lookup",
                        "ops_flatfield_sigma_px": 48.0})
    general = {value: ops[key] for key, value in ill.OPS_TO_ILLUMINATION_KEYS.items()}
    general.update(illumination_nuclear_channel=0, illumination_nuclei=str(table),
                   illumination_correction=True, illumination_estimator="nuclear_median",
                   illumination_qc=False, src=src, channels=[0], verbose=False)
    prepared = ill.prepare_illumination_model(ill.illumination_settings(general))
    item = prepared.model.field_for("plateX")
    renamed = nuclei.rename(columns={"nucleus_channel_0_mean_intensity": "nucleus_dapi_mean_intensity"})
    reference, _info = F.build_nuclear_flat(renamed, column="nucleus_dapi_mean_intensity",
                                            offset=OFFSET, field_px=FIELD,
                                            **{k: v for k, v in F._nuclear_options(ops).items()
                                               if k != "field_px"})
    assert md5(item.block_maps[0]) == md5(reference)
    loaded = ill.load_illumination_model(prepared.model_path).field_for("plateX")
    assert isinstance(loaded, ill.NuclearIlluminationField)
    assert md5(loaded.block_maps[0]) == md5(reference)
    assert np.array_equal(loaded.flatfield, item.flatfield)
    assert loaded.restore == (True,) and float(loaded.dark[0]) == np.float32(OFFSET)


def test_corrected_pixels_equal_ops_correct_field(tmp_path):
    src = merged_plate(str(tmp_path / "merged"), fields=4)
    model = ill.estimate_nuclear_illumination(
        src, [0, 1], nuclear_channel=0, dark=0.0, dark_per_plate={"plateX": OFFSET},
        nucleus_mask_dim=2, options=options(), verbose=False)
    item = model.field_for("plateX")
    path = sorted(glob.glob(os.path.join(src, "*.npy")))[0]
    stack = np.load(path)[..., :2].astype(np.float32)
    context = types.SimpleNamespace(file_name=os.path.basename(path), channels=[0, 1])
    corrected = ill.IlluminationCorrector(model, verbose=False)(stack.copy(), context)
    planes = F.flat_planes({"DAPI": item.block_maps[0]}, (FIELD, FIELD),
                           {"DAPI": {"lookup": "block_centre", "block": BLOCK, "centre_offset": 8.0}})
    expected = F.correct_field(stack[..., :1], planes, {"DAPI": OFFSET}, ["DAPI"])
    assert corrected.dtype == np.float32
    assert np.array_equal(corrected[..., 0], expected[..., 0])
    gain = (1.0 / item.flatfield[1]).astype(np.float32)
    assert np.array_equal(corrected[..., 1], (stack[..., 1] - np.float32(0.0)) * gain)
    window = item.correct_window(stack[40:90, 200:280], [0, 1], y0=40, x0=200)
    assert np.array_equal(window, corrected[40:90, 200:280])


def test_measured_nuclear_means_match_the_exact_ops_remeasure(tmp_path):
    src = merged_plate(str(tmp_path / "merged"), fields=4)
    model = ill.estimate_nuclear_illumination(
        src, [0], nuclear_channel=0, dark_per_plate={"plateX": OFFSET},
        nucleus_mask_dim=2, options=options(), verbose=False)
    item = model.field_for("plateX")
    path = sorted(glob.glob(os.path.join(src, "*.npy")))[1]
    stack = np.load(path)
    context = types.SimpleNamespace(file_name=os.path.basename(path), channels=[0])
    corrected = ill.IlluminationCorrector(model, verbose=False)(stack[..., :1].copy(), context)[..., 0]
    labels = stack[..., 2].astype(np.int64)
    exact = F.field_nucleus_means(stack[..., 0], labels, {"n": item.block_maps[0]}, OFFSET,
                                  lookups={"n": {"lookup": "block_centre", "block": BLOCK,
                                                 "centre_offset": 8.0}})
    count = np.bincount(labels.ravel())
    ours = (np.bincount(labels.ravel(), corrected.astype(np.float64).ravel()) / np.maximum(count, 1))
    assert np.allclose(ours[exact.label.to_numpy()], exact.exact_n.to_numpy(), rtol=1e-6, atol=0)


def test_the_map_from_masks_recovers_the_corners_better_than_the_polynomial(tmp_path):
    src = merged_plate(str(tmp_path / "merged"), fields=16)
    settings = ill.illumination_settings({
        "src": src, "channels": [0], "nucleus_mask_dim": 2, "verbose": False,
        "illumination_estimator": "nuclear_median", "illumination_nuclear_channel": 0,
        "illumination_dark_per_plate": {"plateX": OFFSET}, "illumination_holdout_n": 4,
        "illumination_max_fields": 16})
    radial = ill.validate_illumination(settings, save_dir=str(tmp_path / "validation"), bootstrap=50)
    summary = radial.attrs["summary"]["plateX"]
    nuclear = summary["exact_nuclear_median"]["max_abs_dev_bins_ge100_cells"]
    polynomial = summary["exact_polynomial"]["max_abs_dev_bins_ge100_cells"]
    raw = summary["as measured"]["max_abs_dev_bins_ge100_cells"]
    assert nuclear < polynomial < raw
    assert nuclear < 0.05
    assert os.path.isfile(tmp_path / "validation" / "radial.csv")
    assert json.loads((tmp_path / "validation" / "radial_summary.json").read_text())["plateX"]["offset"] == OFFSET


def test_shuffled_positions_give_a_flat_map():
    nuclei = nucleus_frame(fields=300, shuffle=True)
    flat, _info = F.build_nuclear_flat(nuclei, column="nucleus_channel_0_mean_intensity",
                                       offset=OFFSET, field_px=FIELD, **options())
    real, _info = F.build_nuclear_flat(nucleus_frame(fields=300), offset=OFFSET, field_px=FIELD,
                                       column="nucleus_channel_0_mean_intensity", **options())
    assert float(1 - real.min()) > 0.6
    assert float(np.abs(flat - 1).mean()) <= 0.02
    assert float(np.max(np.abs(flat - 1))) <= 0.05


def test_rectangular_fields_are_mapped_per_axis():
    rng = np.random.default_rng(0)
    shape = (FIELD, FIELD + 4 * BLOCK)
    cy = rng.uniform(0, shape[0], 40000)
    cx = rng.uniform(0, shape[1], 40000)
    median, count = F.nuclear_median_grid(cy, cx, OFFSET + vignette(cy, cx, shape) * TRUE,
                                          block=BLOCK, size=(shape[0] // BLOCK, shape[1] // BLOCK))
    assert median.shape == (SIZE, SIZE + 4) and count.sum() == 40000
    for grid in ("block_centre", "lookup"):
        flat, _info = F.fit_nuclear_flat(median, count, OFFSET, map_grid=grid, field_px=shape,
                                         centre_cells=4)
        assert flat.shape == (SIZE, SIZE + 4)
        assert abs(float(flat[SIZE // 2, (SIZE + 4) // 2]) - 1) < 0.02
        assert float(flat[0, 0]) < 0.4


def test_default_estimators_are_untouched(tmp_path):
    src = merged_plate(str(tmp_path / "merged"), fields=6)
    model = ill.estimate_illumination(src, [0, 1], estimator="polynomial", dark=5.0, verbose=False)
    item = model.field_for("plateX")
    assert type(item) is ill.IlluminationField
    path = model.save(str(tmp_path / "model.npz"))
    with np.load(path) as handle:
        manifest = json.loads(str(handle["manifest"]))
        assert manifest["format"] == 1
        assert sorted(handle.files) == ["field0", "manifest"]
    assert set(manifest["index"]["field0"]) == {
        "plate", "channels", "dark", "n_fields", "estimator", "degree", "bin_size", "floored", "key"}
    stack = np.load(sorted(glob.glob(os.path.join(src, "*.npy")))[0])[..., :2].astype(np.float32)
    context = types.SimpleNamespace(file_name="plateX_A01_1_1.npy", channels=[0, 1])
    corrected = ill.IlluminationCorrector(model, verbose=False)(stack.copy(), context)
    gains = np.stack([1.0 / item.flatfield[0], 1.0 / item.flatfield[1]], -1).astype(np.float32)
    assert np.array_equal(corrected, (stack - np.float32(5.0)) * gains)
    assert not np.any(item._restore_stack([0, 1]))


def test_the_defaults_keep_the_polynomial_and_add_only_alpha_keys():
    from spacr.settings import ALPHA_FEATURES

    defaults = ill.illumination_settings()
    assert defaults["illumination_estimator"] == "polynomial"
    assert defaults["illumination_correction"] is False
    gated = set(ALPHA_FEATURES[933]["settings"])
    assert gated == set(ill.NUCLEAR_SETTINGS_DEFAULTS)
    assert ALPHA_FEATURES[933]["choices"] == {"illumination_estimator": ("nuclear_median",)}
    for key in gated:
        assert key in ill._TOOLTIPS and key in ill._TYPES


def test_the_ops_key_map_is_complete():
    from spacr.ops_settings import ops_defaults

    ops = ops_defaults({})
    general = ill.illumination_settings()
    for key, value in ill.OPS_TO_ILLUMINATION_KEYS.items():
        assert key in ops and value in general
    built = set(F._nuclear_options(ops)) - {"field_px"}
    mapped = {ill.OPS_TO_ILLUMINATION_KEYS[k] for k in ill.OPS_TO_ILLUMINATION_KEYS}
    assert set(ill._nuclear_settings_options(general)) == built
    for name in built:
        assert any(v.endswith(name if name != "block" else "block_px") for v in mapped), name


@pytest.mark.parametrize("dark,per_plate", [(0.0, None), (0.0, {"other": 5.0}), (float("nan"), None)])
def test_no_offset_is_refused(tmp_path, dark, per_plate):
    src = merged_plate(str(tmp_path / "merged"), fields=2)
    with pytest.raises(ill.IlluminationError, match="never taken as 0"):
        ill.estimate_nuclear_illumination(src, [0], nuclear_channel=0, dark=dark,
                                          dark_per_plate=per_plate, nucleus_mask_dim=2,
                                          verbose=False)


def test_a_table_measured_with_correction_is_refused(tmp_path):
    src = merged_plate(str(tmp_path / "merged"), fields=2)
    database = tmp_path / "measurements.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE settings (setting_key TEXT, setting_value TEXT)")
        connection.execute("INSERT INTO settings VALUES ('illumination_correction', 'True')")
        connection.execute('CREATE TABLE nucleus (file_name TEXT, object_label INT, '
                           '"nucleus_channel_0_centroid_weighted-0" REAL, '
                           '"nucleus_channel_0_centroid_weighted-1" REAL, '
                           'nucleus_channel_0_mean_intensity REAL)')
    with pytest.raises(ill.IlluminationError, match="nucleus_channel_0_mean_intensity"):
        ill.estimate_nuclear_illumination(src, [0], nuclear_channel=0, dark=OFFSET,
                                          nuclei=str(database), verbose=False)


def test_an_uncorrected_database_is_read(tmp_path):
    src = merged_plate(str(tmp_path / "merged"), fields=4)
    table, _shape = ill._nuclei_from_merged(sorted(glob.glob(os.path.join(src, "*.npy"))), 0, 2)
    database = tmp_path / "measurements.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE settings (setting_key TEXT, setting_value TEXT)")
        connection.execute("INSERT INTO settings VALUES ('illumination_correction', 'False')")
        pd.DataFrame({"file_name": table.file, "object_label": table.label,
                      "nucleus_channel_0_centroid_weighted-0": table.nucleus_cy,
                      "nucleus_channel_0_centroid_weighted-1": table.nucleus_cx,
                      "nucleus_channel_0_mean_intensity": table.nucleus_channel_0_mean_intensity,
                      }).to_sql("nucleus", connection, index=False)
    from_db = ill.estimate_nuclear_illumination(src, [0], nuclear_channel=0, dark=OFFSET,
                                                nuclei=str(database), verbose=False)
    from_masks = ill.estimate_nuclear_illumination(src, [0], nuclear_channel=0, dark=OFFSET,
                                                   nucleus_mask_dim=2, verbose=False)
    assert md5(from_db.field_for("plateX").block_maps[0]) == md5(from_masks.field_for("plateX").block_maps[0])


def test_mask_generation_refuses_nuclear_median(tmp_path):
    src = merged_plate(str(tmp_path / "merged"), fields=2)
    settings = ill.illumination_settings({
        "src": src, "channels": [0], "illumination_correction": True,
        "illumination_estimator": "nuclear_median", "illumination_nuclear_channel": 0,
        "illumination_dark": OFFSET})
    with pytest.raises(ill.IlluminationError, match="illumination_for_later_masks"):
        ill.prepare_segmentation_illumination(settings, pipeline_style="v1")


def test_a_missing_nuclear_channel_is_refused(tmp_path):
    src = merged_plate(str(tmp_path / "merged"), fields=2)
    settings = ill.illumination_settings({
        "src": src, "channels": [0], "illumination_correction": True, "verbose": False,
        "illumination_estimator": "nuclear_median", "illumination_dark": OFFSET,
        "illumination_qc": False})
    with pytest.raises(ill.IlluminationError, match="illumination_nuclear_channel"):
        ill.prepare_illumination_model(settings)
    with pytest.raises(ill.IlluminationError, match="estimate_nuclear_illumination"):
        ill.estimate_illumination(src, [0], estimator="nuclear_median", verbose=False)


def test_a_plate_with_too_few_nuclei_per_map_cell_is_refused(tmp_path):
    src = tmp_path / "merged"
    src.mkdir()
    stack = np.zeros((FIELD, FIELD, 3), np.float32)
    for label, (y, x) in enumerate([(20, 20), (100, 200), (250, 60), (300, 300)], start=1):
        stack[y:y + 4, x:x + 4, 2] = label
        stack[y:y + 4, x:x + 4, 0] = OFFSET + TRUE
    np.save(src / "plateX_A01_0_1.npy", stack)
    src = str(src)
    with pytest.raises(ill.IlluminationError, match="nuclei per"):
        ill.estimate_nuclear_illumination(src, [0], nuclear_channel=0, dark=OFFSET,
                                          nucleus_mask_dim=2, verbose=False)
