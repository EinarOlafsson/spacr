"""Per-channel OPS flat-field fits, with object-median maps built after segmentation (item 1006).

A synthetic plate: a steep vignette on the nuclei's excitation (corners at 40 % of the
centre), nuclei of known DNA content, and between them a stray-light background with its own
shallow profile (corners at 90 %). The nuclear-median map recovers the vignette at the
corners within 1 %; the pixel-median polynomial map, made mostly of background pixels, does not.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

from spacr import ops_engine
from spacr import ops_flatfield as ff

SIZE = 640
OFFSET = 100.0
PLATE = "P1"


def _vignette(depth=0.6):
    """The true flat: 1 at the centre, ``1 - depth`` at the corners."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(float)
    c = (SIZE - 1) / 2.0
    return 1.0 - depth * ((yy - c) ** 2 + (xx - c) ** 2) / (2 * c ** 2)


def _field(seed):
    """One field: ``(raw DNA plane, nucleus labels)``; G1 nuclei of content 1000."""
    rng = np.random.default_rng(seed)
    labels = np.zeros((SIZE, SIZE), np.int32)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    k = 0
    for cy in range(2, SIZE, 22):
        for cx in range(2, SIZE, 22):
            y, x = cy + rng.integers(-2, 3), cx + rng.integers(-2, 3)
            k += 1
            labels[(yy - y) ** 2 + (xx - x) ** 2 <= 25] = k
    content = np.where(labels > 0, 1000.0 * np.exp(rng.normal(0, 0.03, k + 1))[labels], 0.0)
    signal = np.where(labels > 0, content * _vignette(), 300.0 * _vignette(0.1))
    image = OFFSET + signal + rng.normal(0, 2, (SIZE, SIZE))
    return image.astype(np.float32), labels


def _table(fields=24):
    """The nucleus table of ``fields`` synthetic fields."""
    import pandas as pd

    rows = []
    for site in range(fields):
        image, labels = _field(site)
        rows.append(ff.field_object_means(image, labels, labels, well="A1", site=site, channel="DAPI"))
    return pd.concat(rows, ignore_index=True)


def _truth_cells(flat_shape, block=16):
    """The true flat at each map cell's centre, normalised like the map (centre 18 x 18 = 1)."""
    v = _vignette()
    cells = v[block // 2::block, block // 2::block][:flat_shape[0], :flat_shape[1]]
    h = flat_shape[0] // 2
    return cells / cells[h - 9:h + 9, h - 9:h + 9].mean()


def _corners(a, n=18):
    """The mean of the four ``n`` x ``n`` corners: the map record's ``corner_mean`` blocks."""
    return float(np.mean([a[:n, :n].mean(), a[:n, -n:].mean(), a[-n:, :n].mean(), a[-n:, -n:].mean()]))


def test_the_nuclear_median_recovers_a_steep_vignette_and_the_polynomial_does_not():
    table = _table()
    flat, info = ff.build_nuclear_flat(table, column="nucleus_dapi_mean_intensity", offset=OFFSET,
                                       field_px=SIZE)
    truth = _truth_cells(flat.shape)
    assert abs(_corners(flat) / _corners(truth) - 1) < 0.01
    blocks = np.stack([ff.field_blocks(_field(s)[0][..., None]) for s in range(24)])
    maps, _method = ff.build_flat_maps(blocks, ["DAPI"], polynomial_channels=("DAPI",))
    assert abs(_corners(maps["DAPI"]) / _corners(truth) - 1) > 0.02
    assert info["n_training_nuclei"] == len(table)


def test_field_object_means_gives_the_columns_of_the_912_table():
    image, labels = _field(0)
    table = ff.field_object_means(image, labels, labels, well="B2", site=7, channel="DAPI")
    assert {"well", "site", "label", "nucleus_cy", "nucleus_cx", "nucleus_touches_field_edge",
            "nucleus_dapi_mean_intensity"} <= set(table.columns)
    assert set(table["well"]) == {"B2"} and set(table["site"]) == {7}
    one = table.iloc[0]
    mask = labels == one["label"]
    assert np.isclose(one["nucleus_dapi_mean_intensity"], image[mask].mean(), rtol=1e-6)


def _pending_folder(tmp_path, fits=None):
    """A maps folder written by ph_flatfield with the object-fit channels pending."""
    fits = fits or {"DAPI": "nuclear_median"}
    blocks = np.stack([ff.field_blocks(np.stack([_field(s)[0], _field(s)[0] * 0.5], -1))
                       for s in range(4)])
    maps, method = ff.build_flat_maps(blocks, ["DAPI", "GFP"], polynomial_channels=("DAPI",))
    pending = [ff.mark_pending(maps, method, c, f) for c, f in fits.items()]
    folder = str(tmp_path / "maps")
    ff.write_flat_maps(folder, PLATE, maps, method, [("A1", s) for s in range(4)],
                       extra={"pending": pending, "channel_fit": fits})
    return folder, method


def test_the_other_maps_are_byte_identical_and_pending_ones_refuse(tmp_path):
    folder, method = _pending_folder(tmp_path)
    blocks = np.stack([ff.field_blocks(np.stack([_field(s)[0], _field(s)[0] * 0.5], -1))
                       for s in range(4)])
    maps, _m = ff.build_flat_maps(blocks, ["DAPI", "GFP"], polynomial_channels=("DAPI",))
    reference = str(tmp_path / "reference")
    ff.write_flat_maps(reference, PLATE, maps, _m, [("A1", s) for s in range(4)])
    md5 = [hashlib.md5(open(os.path.join(f, f"flat_{PLATE}_GFP.npy"), "rb").read()).hexdigest()
           for f in (folder, reference)]
    assert md5[0] == md5[1]
    assert not os.path.exists(os.path.join(folder, f"flat_{PLATE}_DAPI.npy"))
    assert method["DAPI"]["pending"] == "nuclear_median" and method["DAPI"]["offset"] > 0
    with pytest.raises(ValueError, match="ph_flatfield_objects"):
        ff.load_flat_maps(folder, ["DAPI", "GFP"])
    assert set(ff.load_flat_maps(folder, ["GFP"])[0]) == {"GFP"}
    with pytest.raises(ValueError, match="object fit"):
        ff.mark_pending(dict(maps), dict(_m), "GFP", "polynomial")


def _segmented_run(tmp_path, monkeypatch, sites=12):
    """A run folder with ph_segment masks; DAPI and GFP planes read through a stand-in."""
    images = {}
    for site in range(sites):
        image, labels = _field(site)
        images[site] = np.stack([image, image * 0.5], -1)
        folder = tmp_path / "run" / "A1" / "phenotype_masks"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / f"site-{site}.npz", nuclei=labels, cells=labels)
    order = {"DAPI": 0, "GFP": 1}
    monkeypatch.setattr(ops_engine, "_ph_read_field",
                        lambda files, channels, unreadable, secondary=None:
                        images[int(files["site"])][..., [order[c] for c in channels]])
    index = {"A1": {1: {site: {"site": site} for site in range(sites)}}}
    return index


def test_the_phase_builds_every_pending_map_once_and_keeps_its_md5(tmp_path, monkeypatch):
    folder, _method = _pending_folder(tmp_path, {"DAPI": "nuclear_median", "GFP": "cell_median"})
    index = _segmented_run(tmp_path, monkeypatch)
    settings = {"ops_flatfield_maps": folder, "ops_flatfield_dark": OFFSET,
                "ops_flatfield_field_px": SIZE, "ops_flatfield_holdout_n": 2}
    report = ops_engine._ph_flatfield_objects(PLATE, ["A1"], index, settings, str(tmp_path / "run"))
    assert report["fields"] == 12 and set(report["channels"]) == {"DAPI", "GFP"}
    dapi = report["channels"]["DAPI"]
    assert dapi["fit"] == "nuclear_median" and dapi["held_out_fields"] == 2
    assert dapi["offset_source"] == "ops_flatfield_dark"
    maps, offsets, info = ff.load_flat_maps(folder, ["DAPI", "GFP"])
    assert offsets["DAPI"] == OFFSET and info["md5_record"] and not info["pending"]
    assert info["md5"]["DAPI"] == dapi["md5"]
    assert info["method"]["GFP"]["estimator"] == "cell_median"
    assert abs(_corners(maps["DAPI"]) / _corners(_truth_cells(maps["DAPI"].shape)) - 1) < 0.01
    for channel in ("DAPI", "GFP"):
        assert os.path.isfile(os.path.join(folder, f"objects_{PLATE}_{channel}.parquet"))
    again = ops_engine._ph_flatfield_objects(PLATE, ["A1"], index, settings,
                                             str(tmp_path / "run"))
    assert again == {"reused": folder, "md5": info["md5"]}
    with pytest.raises(ValueError, match="not pending"):
        ff.complete_pending_flat(folder, PLATE, "DAPI", maps["DAPI"], {"offset": OFFSET})


def test_the_offset_is_set_per_channel_or_measured_and_never_zero(tmp_path, monkeypatch):
    folder, method = _pending_folder(tmp_path)
    index = _segmented_run(tmp_path, monkeypatch, sites=4)
    report = ops_engine._ph_flatfield_objects(
        PLATE, ["A1"], index, {"ops_flatfield_maps": folder, "ops_flatfield_field_px": SIZE,
                               "ops_flatfield_holdout_n": 1}, str(tmp_path / "run"))
    dapi = report["channels"]["DAPI"]
    assert dapi["offset_source"] == "ph_flatfield" and dapi["offset"] == method["DAPI"]["offset"]
    measured = {"DAPI": {"offset": 7.0}}
    assert ops_engine._ph_channel_offset({"ops_flatfield_dark": {"dapi": 3}}, PLATE, "DAPI",
                                         measured) == (3.0, "ops_flatfield_dark")
    assert ops_engine._ph_channel_offset({"ops_flatfield_dark": {PLATE: {"DAPI": 4}}}, PLATE,
                                         "DAPI", measured) == (4.0, "ops_flatfield_dark")
    assert ops_engine._ph_channel_offset({"ops_flatfield_dark": 5}, PLATE, "DAPI",
                                         measured) == (5.0, "ops_flatfield_dark")
    assert ops_engine._ph_channel_offset({}, PLATE, "DAPI", measured) == (7.0, "ph_flatfield")
    info_path = os.path.join(str(tmp_path), "nooffset", "flat_info.json")
    os.makedirs(os.path.dirname(info_path))
    json.dump({"plate": PLATE, "method": {"DAPI": {"pending": "nuclear_median"}},
               "pending": ["DAPI"]}, open(info_path, "w"))
    with pytest.raises(ValueError, match="offset"):
        ops_engine._ph_flatfield_objects(PLATE, ["A1"], index,
                                         {"ops_flatfield_maps": os.path.dirname(info_path)},
                                         str(tmp_path / "run"))


def test_the_field_choice_is_all_a_seeded_count_or_a_list(tmp_path):
    available = [("A1", s) for s in range(10)]
    assert ops_engine._ph_object_fields({}, available) == available
    four = ops_engine._ph_object_fields({"ops_flatfield_object_fields": 4}, available)
    assert len(four) == 4 and four == ops_engine._ph_object_fields(
        {"ops_flatfield_object_fields": "4"}, available)
    listed = tmp_path / "fields.json"
    listed.write_text(json.dumps([["A1", 2], ["A1", 5]]))
    assert ops_engine._ph_object_fields({"ops_flatfield_object_fields": str(listed)},
                                        available) == [("A1", 2), ("A1", 5)]
    listed.write_text(json.dumps([["A1", 99]]))
    with pytest.raises(ValueError, match="without phenotype masks"):
        ops_engine._ph_object_fields({"ops_flatfield_object_fields": str(listed)}, available)


def test_the_channel_fits_default_to_today_and_are_checked():
    channels = ["DAPI", "GFP", "A594"]
    assert ops_engine._ph_channel_fits({}, channels) == {
        "DAPI": "polynomial", "GFP": "pixel_median", "A594": "pixel_median"}
    fits = ops_engine._ph_channel_fits(
        {"ops_flatfield_channel_fit": {"dapi": "Nuclear_Median", "A594": "cell_median"}}, channels)
    assert fits == {"DAPI": "nuclear_median", "GFP": "pixel_median", "A594": "cell_median"}
    with pytest.raises(ValueError, match="does not have"):
        ops_engine._ph_channel_fits({"ops_flatfield_channel_fit": {"CY5": "polynomial"}}, channels)
    with pytest.raises(ValueError, match="must give each channel"):
        ops_engine._ph_channel_fits({"ops_flatfield_channel_fit": {"GFP": "spline"}}, channels)


def test_the_settings_are_alpha_and_flatfield_is_on_by_default():
    from spacr.ops_settings import OPS_DEFAULTS, OPS_TOOLTIPS
    from spacr.settings import (ALPHA_FEATURES, DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION,
                                _fold_older_defaults)

    assert set(ALPHA_FEATURES[1006]["settings"]) == {"ops_flatfield_channel_fit",
                                                     "ops_flatfield_object_fields"}
    assert OPS_DEFAULTS["ops_flatfield_channel_fit"] is None
    for key in ALPHA_FEATURES[1006]["settings"]:
        assert len(OPS_TOOLTIPS[key]) <= 600 and "Alpha" in OPS_TOOLTIPS[key]
    assert OPS_DEFAULTS["ops_flatfield"] == "exact" and "Default 'exact'" in OPS_TOOLTIPS["ops_flatfield"]
    assert SETTINGS_SCHEMA_VERSION >= 952 and DEFAULT_CHANGES[952] == {"ops_flatfield": "none"}
    old = {}
    assert _fold_older_defaults(old, 948)["ops_flatfield"] == ("none", 952)
    assert old["ops_flatfield"] == "none"
    assert _fold_older_defaults({}, 952) == {}


@pytest.mark.skipif(not os.environ.get("SPACR_912_NUCLEI"),
                    reason="the 912 reference nucleus table is at Home (SPACR_912_NUCLEI)")
def test_home_check_reproduces_the_912_reference_map():
    """Home's one-call check: the 912 plate A nucleus table and its 24 held-out fields give md5 f0dbfb9d."""
    from spacr.tabular import read_table

    table = read_table(os.environ["SPACR_912_NUCLEI"], canonicalise=False, report=None)
    held = [(str(x).split(":")[0], int(str(x).split(":")[1]))
            for x in os.environ["SPACR_912_HELDOUT"].split(",")]
    flat, _info = ff.build_nuclear_flat(table, column="nucleus_dapi_mean_intensity",
                                        offset=float(os.environ["SPACR_912_OFFSET"]), holdout=held,
                                        map_grid="lookup", lookup="zoom_endpoints")
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "flat_912_check.npy")
    np.save(path, np.asarray(flat, np.float32))
    assert hashlib.md5(open(path, "rb").read()).hexdigest() == "f0dbfb9da3e6ab3c1e38a63485491146"


def test_the_api_fallback_is_the_ops_default():
    """run_ops without ops_flatfield runs as the command line, the application and a settings file do."""
    from spacr.ops_settings import OPS_DEFAULTS

    assert ops_engine._ph_flatfield_mode({}) == OPS_DEFAULTS["ops_flatfield"] == "exact"
    assert ops_engine._ph_flatfield_mode({"ops_flatfield": None}) == OPS_DEFAULTS["ops_flatfield"]
    assert ops_engine._ph_flatfield_mode({"ops_flatfield": "None"}) == "none"
