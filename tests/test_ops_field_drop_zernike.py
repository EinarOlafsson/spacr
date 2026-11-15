"""OPS failed-field drop, non-finite row drop and the Zernike setting (items 860-862).

All three are alpha options, off by default; an earlier run gives the same
tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import ops_engine
from spacr.ops_merge import field_transform
from spacr.ops_sample import measure_phenotype_field
from spacr.ops_store import write_table

PLATE = "P1"
SIZE = 200


def _two_fields_one_misplaced(tmp_path):
    """Field 0 sits where its cells are; field 1's cells match nothing."""
    db = str(tmp_path / "measurements.db")
    rows = []
    for site, cx in ((0, 100.0), (1, 320.0)):
        transform = field_transform(1.0, 0.0, (100.0, cx), (SIZE, SIZE))
        rows.append({"plate": PLATE, "well": "A1", "site": site,
                     "centre_y": 100.0, "centre_x": cx, "is_anchor": 1,
                     "scale": 1.0, "rotation_deg": 0.0,
                     "offset_y": float(transform[2][0]),
                     "offset_x": float(transform[2][1]),
                     "field_height": SIZE, "field_width": SIZE,
                     "transform_source": "anchor"})
    write_table(db, "ops_phenotype", pd.DataFrame(rows))
    rng = np.random.default_rng(1)
    # dense sequencing nuclei across both footprints
    ty = rng.uniform(0, SIZE, 400)
    tx = rng.uniform(0, 420, 400)
    write_table(db, "ops_objects", pd.DataFrame({
        "plate": PLATE, "well": "A1", "object_id": np.arange(400),
        "centroid_y": ty, "centroid_x": tx}))
    cells = []
    for site, origin in ((0, 0.0), (1, 220.0)):
        inside = np.flatnonzero((tx >= origin) & (tx <= origin + SIZE))
        for label, i in enumerate(inside, start=1):
            if site == 0:
                y, x = ty[i] + rng.normal(0, 0.4), tx[i] - origin
            else:  # unrelated positions: links only by chance
                y, x = rng.uniform(0, SIZE), rng.uniform(0, SIZE)
            cells.append({"plate": PLATE, "well": "A1", "site": site,
                          "label": label, "centroid_row": y,
                          "centroid_col": x})
    pd.DataFrame(cells).to_parquet(tmp_path / "ops_phenotype_cells_A1.parquet")
    return db


def test_merge_writes_the_field_check_only_when_asked(tmp_path):
    db = _two_fields_one_misplaced(tmp_path)
    # no ops_geometry in this synthetic well: merge with 'similarity'
    off = ops_engine._ph_merge(db, PLATE, "A1", {"ops_align_model": "similarity"},
                               str(tmp_path), False)
    assert off["field_check"] is None
    assert not (tmp_path / "ops_fieldcheck_A1.parquet").exists()
    report = ops_engine._ph_merge(db, PLATE, "A1",
                                  {"ops_drop_failed_fields": True,
                                   "ops_align_model": "similarity"},
                                  str(tmp_path), False)
    assert report["field_check"]["dropped"] == [1]
    check = pd.read_parquet(tmp_path / "ops_fieldcheck_A1.parquet")
    by_site = check.set_index("site")
    assert not by_site.loc[0, "dropped"] and by_site.loc[0, "reason"] == ""
    assert by_site.loc[1, "dropped"] and by_site.loc[1, "reason"] != ""
    assert by_site.loc[0, "link_over_chance"] > 3
    assert by_site.loc[1, "link_over_chance"] < 3
    assert 0 < by_site.loc[1, "chance_rate"] < 1


def test_the_pilot_rule_text():
    # dropped = (refined == 0 and ratio < 3) or ratio < 2
    ratio = np.array([1.0, 2.5, 2.5, 4.0, 1.9])
    refined = np.array([1, 0, 1, 0, 1])
    got = ((refined == 0) & (ratio < 3)) | (ratio < 2)
    assert got.tolist() == [True, True, False, False, True]


def test_aggregate_drops_failed_fields_and_nonfinite_rows(tmp_path):
    from tests.test_ops_aggregate_blockwise import _plate, _settings

    db, wells, _ = _plate(tmp_path)
    settings = _settings(tmp_path, ops_bootstrap=0)
    base = ops_engine._ph_aggregate(db, PLATE, wells, settings, str(tmp_path))
    assert base["dropped_fields"] is None
    assert base["dropped_nonfinite_rows"] is None
    # label 8 of each well carries one NaN among its 8 features
    only = ops_engine._ph_aggregate(
        db, PLATE, wells, {**settings, "ops_drop_nonfinite_rows": True},
        str(tmp_path))
    assert only["dropped_nonfinite_rows"] == 3
    assert only["cells_in_design"] == base["cells_in_design"] - 3
    with pytest.raises(ValueError, match="field check of well"):
        ops_engine._ph_aggregate(
            db, PLATE, wells, {**settings, "ops_drop_failed_fields": True},
            str(tmp_path))
    for well in wells:
        pd.DataFrame({
            "plate": PLATE, "well": well, "site": [0, 1, 2, 3, 4],
            "cells": 80, "matched": [80, 80, 5, 80, 80],
            "link_rate": [1.0, 1.0, 0.06, 1.0, 1.0],
            "sbs_nuclei_under_field": 50, "chance_rate": 0.05,
            "refined": [1, 1, 0, 1, 1], "refined_inliers": 30,
            "link_over_chance": [20.0, 20.0, 1.2, 20.0, 20.0],
            "dropped": [False, False, True, False, False],
            "reason": ["", "", "transform refinement failed and links "
                       "under 3x chance", "", ""]}).to_parquet(
            ops_engine._ph_fieldcheck_path(str(tmp_path), well))
    both = ops_engine._ph_aggregate(
        db, PLATE, wells, {**settings, "ops_drop_failed_fields": True,
                           "ops_drop_nonfinite_rows": True}, str(tmp_path))
    assert both["dropped_fields"]["fields_dropped"] == [
        f"{w}/2" for w in wells]
    assert both["dropped_fields"]["cells_dropped"] > 0
    listed = pd.read_csv(tmp_path / f"dropped_fields_{PLATE}.csv")
    assert list(listed["site"]) == [2, 2, 2]
    assert listed["reason"].str.contains("3x chance").all()
    assert both["cells_in_design"] < only["cells_in_design"]


def test_zernike_is_off_by_default_and_adds_the_moments_for_cell_and_nucleus():
    pytest.importorskip("mahotas")
    cells = np.zeros((60, 60), np.int32)
    nuclei = np.zeros((60, 60), np.int32)
    cells[10:40, 10:45] = 1
    nuclei[18:30, 20:34] = 1
    cells[45:58, 45:58] = 2
    nuclei[48:55, 48:55] = 2
    image = np.random.default_rng(0).uniform(100, 900, (60, 60, 1)).astype(
        np.float32)
    plain = measure_phenotype_field(nuclei, cells, image, ["DAPI"],
                                    texture=False)
    zern = measure_phenotype_field(nuclei, cells, image, ["DAPI"],
                                   texture=False, zernike=True)
    assert not [c for c in plain.columns if "zernike" in c]
    added = sorted(set(zern.columns) - set(plain.columns))
    assert added == sorted([f"{p}_zernike_{i}" for p in ("cell", "nucleus")
                            for i in range(25)])
    pd.testing.assert_frame_equal(zern[plain.columns], plain)
    from skimage.measure import regionprops_table

    from spacr.measure import _calculate_zernike

    frame = pd.DataFrame(regionprops_table(nuclei, properties=["label"]))
    direct = _calculate_zernike(nuclei, frame.copy(), degree=8)
    for i in range(25):
        assert np.array_equal(zern[f"nucleus_zernike_{i}"].to_numpy(),
                              direct[f"zernike_{i}"].to_numpy())


def test_new_settings_default_off_and_the_marker_tolerates_old_runs(tmp_path):
    from spacr.ops_settings import OPS_DEFAULTS

    for key in ("ops_ph_zernike", "ops_drop_failed_fields",
                "ops_drop_nonfinite_rows"):
        assert OPS_DEFAULTS[key] is False
    folder = tmp_path / "parts"
    folder.mkdir()
    (folder / "part_options.json").write_text(
        '{"flatfield": null, "funk_features": false}')
    ops_engine._ph_options_marker(str(folder), {
        "flatfield": None, "funk_features": False, "zernike": False})
    with pytest.raises(ValueError):
        ops_engine._ph_options_marker(str(folder), {
            "flatfield": None, "funk_features": False, "zernike": True})
