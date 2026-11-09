"""Item 870: profiling_normalization='ljosa_percentile' (Ljosa et al. 2013 scaling).

Each plate's features are mapped so that the 1st percentile of its control
cells is 0 and the 99th is 1, as run_and_score.py's ``--norm ljosa`` does; a
feature with no range on a plate is dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import sp_stats as s
from tests.test_image_based_profiling import _measurement_db, _screen


def test_well_percentiles_scale_the_controls_to_zero_and_one():
    wells, features = _screen(seed=4)
    negative = wells["treatment"].eq("neg")
    out = s._normalize_profiles(wells, features, method="ljosa_percentile",
                                reference=negative)
    for plate, part in wells.groupby("plateID"):
        ref = part.loc[negative[part.index], features].to_numpy(float)
        low, high = np.percentile(ref, [1, 99], axis=0)
        expected = (part[features].to_numpy(float) - low) / (high - low)
        np.testing.assert_allclose(out.loc[part.index, features].to_numpy(), expected)
    assert out.attrs["flat_features"] == []


def test_cell_percentiles_are_used_and_flat_features_listed():
    wells, features = _screen(seed=5)
    wells["cell_flat"] = 1.0
    features = features + ["cell_flat"]
    cells = pd.concat([wells.assign(cell_f0=wells["cell_f0"] + k) for k in range(5)],
                      ignore_index=True)
    control = cells["treatment"].eq("neg")
    low, high = s._control_percentiles(cells, features, control)
    expected_low = cells[control].groupby("plateID")[features].quantile(0.01)
    pd.testing.assert_frame_equal(low, expected_low)
    out = s._normalize_profiles(wells, features, method="ljosa_percentile",
                                percentiles=(low, high))
    first = wells["plateID"].eq("p1")
    np.testing.assert_allclose(
        out.loc[first, "cell_f1"],
        (wells.loc[first, "cell_f1"] - low.loc["p1", "cell_f1"])
        / (high.loc["p1", "cell_f1"] - low.loc["p1", "cell_f1"]))
    assert out.attrs["flat_features"] == ["cell_flat"]
    np.testing.assert_allclose(out["cell_flat"], 0.0)
    with pytest.raises(s._ProfilingError, match="no control percentiles for plate p2"):
        s._normalize_profiles(wells, features, method="ljosa_percentile",
                              percentiles=(low.loc[["p1"]], high.loc[["p1"]]))
    with pytest.raises(s._ProfilingError, match="lack 1 feature"):
        s._normalize_profiles(wells, features, method="ljosa_percentile",
                              percentiles=(low.drop(columns="cell_f3"), high))
    with pytest.raises(s._ProfilingError, match="no control cell"):
        s._control_percentiles(cells, features, np.zeros(len(cells), bool))


def test_profile_wells_drops_features_with_no_range():
    wells, features = _screen(seed=6)
    wells["cell_flat"] = 2.0
    result = s._profile_wells(wells, features + ["cell_flat"], group_columns=["treatment"],
                              negative_controls=["neg"], normalization="ljosa_percentile",
                              feature_selection=[], null_size=100, report=None)
    assert result.excluded["zero_percentile_range"] == ["cell_flat"]
    assert "cell_flat" not in result.kept and len(result.kept) == len(features)
    assert result.options["normalization"] == "ljosa_percentile"
    ctrl = result.selected["treatment"].eq("neg")
    values = result.selected.loc[ctrl, result.kept].to_numpy()
    assert -0.2 < values.min() and values.max() < 1.2


def test_a_measure_run_scales_against_its_control_cells(tmp_path):
    rng = np.random.default_rng(13)
    layout = ["neg"] * 4 + ["hit"] * 3 + ["dud"] * 3
    db = _measurement_db(str(tmp_path / "m" / "measurements.db"), "p1", rng, layout)
    plate_map = tmp_path / "plate_map.csv"
    pd.DataFrame({"rowID": ["r1"] * len(layout),
                  "columnID": [f"c{i + 1}" for i in range(len(layout))],
                  "treatment": layout}).to_csv(plate_map, index=False)
    settings = dict(profiling_metadata=str(plate_map), profiling_treatment_column="treatment",
                    profiling_negative_control="neg", profiling_feature_selection=[],
                    profiling_normalization="ljosa_percentile")
    result, written = s._profile_measurements(settings, db, report=None)
    wells = result.wells
    negative = wells["treatment"].eq("neg")
    low, high = s._reference_object_percentiles([db], wells, negative)
    assert list(low.index) == ["p1"] and "cell_area" in low.columns
    expected = (wells["cell_area"] - low.loc["p1", "cell_area"]) / (
        high.loc["p1", "cell_area"] - low.loc["p1", "cell_area"])
    np.testing.assert_allclose(result.normalized["cell_area"], expected)
    assert any(name.startswith("well_profiles") for name in written)
    with pytest.raises(s._ProfilingError, match="no object of a negative-control"):
        s._reference_object_percentiles([db], wells, pd.Series(False, index=wells.index))


def test_an_empty_object_table_adds_no_percentiles(tmp_path):
    """A pathogen table with no rows is skipped, not read as a plate."""
    import sqlite3

    rng = np.random.default_rng(14)
    layout = ["neg"] * 4 + ["hit"] * 3
    db = _measurement_db(str(tmp_path / "measurements.db"), "p1", rng, layout)
    wells = pd.DataFrame({"plateID": ["p1"] * len(layout),
                          "rowID": ["r1"] * len(layout),
                          "columnID": [f"c{i + 1}" for i in range(len(layout))]})
    negative = pd.Series([name == "neg" for name in layout], index=wells.index)
    before_low, before_high = s._reference_object_percentiles([db], wells, negative)
    with sqlite3.connect(db) as connection:
        connection.execute(
            "CREATE TABLE pathogen (plateID TEXT, rowID TEXT, columnID TEXT, "
            "fieldID TEXT, prcf TEXT, object_label INTEGER, pathogen_area REAL)")
    low, high = s._reference_object_percentiles([db], wells, negative)
    assert not any(name.startswith("pathogen_") for name in low.columns)
    pd.testing.assert_frame_equal(low, before_low)
    pd.testing.assert_frame_equal(high, before_high)


def test_ljosa_percentile_is_offered_in_the_measure_panel():
    from spacr.settings import tooltips
    from spacr.settings_spec import convert_settings_dict_for_gui

    kind, choices, default = convert_settings_dict_for_gui(
        {"profiling_normalization": "mad_robustize"})["profiling_normalization"]
    assert kind == "combo" and "ljosa_percentile" in choices and "tvn" in choices
    assert "ljosa_percentile" in s._PROFILE_NORMALIZATIONS
    text = tooltips["profiling_normalization"]
    assert "ljosa_percentile" in text and len(text) <= 600
