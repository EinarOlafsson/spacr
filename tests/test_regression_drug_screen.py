"""Regression in drug-screen mode: drugs and doses against vehicle wells, no guides.

A synthetic two-plate screen with a known plate offset, one active and one
inactive compound, written as a spaCR measurements database plus a plate map.
"""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.ml import perform_regression
from spacr.settings import get_perform_regression_default_settings

ROWS = "ABCD"


def _screen(tmp_path, *, cells=3, seed=0):
    """Write measurements.db (cell table) and plate_map.csv; return their paths.

    Each plate holds 4 vehicle wells and, for drugs ACT and NUL, two wells at
    each of 1, 10 and 100 uM. ACT raises the response by 1 per log10 unit
    above 0.1 uM; NUL does nothing. Plate 2 sits 5 units higher throughout.
    """
    rng = np.random.default_rng(seed)
    layout = [("DMSO", 0.0)] * 4 + [(drug, dose) for drug in ("ACT", "NUL")
                                    for dose in (1.0, 10.0, 100.0) for _ in range(2)]
    objects, plate_map = [], []
    for plate, offset in (("plate1", 0.0), ("plate2", 5.0)):
        for index, (drug, dose) in enumerate(layout):
            row, column = index // 4 + 1, index % 4 + 1
            effect = (np.log10(dose) + 1.0) if drug == "ACT" else 0.0
            well = f"{'ABCDEFGH'[row - 1]}{column:02d}"
            plate_map.append(dict(plateID=plate, wellID=well, drug=drug, concentration=dose))
            for cell in range(cells):
                objects.append(dict(plateID=plate, rowID=f"r{row}", columnID=f"c{column}",
                                    fieldID="f1", object_label=cell + 1,
                                    intensity=10.0 + offset + effect + rng.normal(0, 0.15)))
    db = tmp_path / "measurements" / "measurements.db"
    db.parent.mkdir()
    with sqlite3.connect(db) as connection:
        pd.DataFrame(objects).to_sql("cell", connection, index=False)
    plate_csv = tmp_path / "plate_map.csv"
    pd.DataFrame(plate_map).to_csv(plate_csv, index=False)
    return db, plate_csv


def _settings(db, plate_csv, **extra):
    """Drug-screen regression settings for the synthetic screen."""
    settings = dict(screen_type="drug_screen", drug_data=str(db), drug_table="cell",
                    treatment_map=str(plate_csv), dependent_variable="intensity",
                    vehicle_control="DMSO", concentration_unit="uM")
    settings.update(extra)
    return settings


def test_drug_effects_are_plate_adjusted_differences_from_vehicle(tmp_path):
    db, plate_csv = _screen(tmp_path)
    outcome = perform_regression(_settings(db, plate_csv))
    effects = outcome["effects"].set_index("treatment")
    assert list(effects.index) == [f"{d} {c} uM" for d in ("ACT", "NUL") for c in (1, 10, 100)]
    for dose, truth in ((1, 1.0), (10, 2.0), (100, 3.0)):
        row = effects.loc[f"ACT {dose} uM"]
        assert row["ci_low"] < truth < row["ci_high"]
        assert row["q_value"] < 0.001 and row["significant"]
        assert row["n_wells"] == 4 and row["n_plates"] == 2
    nul = effects.loc[[f"NUL {c} uM" for c in (1, 10, 100)]]
    assert (nul["ci_low"] < 0).all() and (nul["ci_high"] > 0).all()
    assert not nul["significant"].any()
    assert set(effects["status"]) == {"tested"}
    assert set(effects["concentration_unit"]) == {"uM"}

    assert "dose_trends" not in outcome
    assert set(effects["plate_effect"]) == {True}

    wells = outcome["well_data"]
    assert len(wells) == 32 and int(wells["vehicle"].sum()) == 8
    assert set(wells["n_objects"]) == {3}
    folder = tmp_path / "measurements" / "results" / "drug_screen" / "intensity"
    assert outcome["res_folder"] == str(folder)
    for name in ("well_data.csv", "drug_effects.csv"):
        assert (folder / name).is_file()
    assert not (folder / "dose_trends.csv").exists()
    assert outcome["paths"]["figure"]
    written = pd.read_csv(folder / "drug_effects.csv")
    assert {"effect", "standard_error", "ci_low", "ci_high", "p_value", "q_value",
            "percent_of_vehicle", "vehicle_mean"} <= set(written.columns)


def test_treatment_columns_may_sit_in_a_per_well_table_without_crops_or_guides(tmp_path):
    db, plate_csv = _screen(tmp_path, cells=1)
    with sqlite3.connect(db) as connection:
        cells = pd.read_sql("SELECT * FROM cell", connection)
    plate_map = pd.read_csv(plate_csv)
    cells["wellID"] = [f"{'ABCD'[int(r[1:]) - 1]}{int(c[1:]):02d}"
                       for r, c in zip(cells["rowID"], cells["columnID"])]
    table = cells.merge(plate_map, on=["plateID", "wellID"])
    table = table.drop(columns=["rowID", "columnID", "fieldID", "object_label"])
    table = table.rename(columns={"drug": "compound", "concentration": "dose_uM"})
    csv = tmp_path / "per_well.csv"
    table.to_csv(csv, index=False)
    assert not {"grna", "gene", "count"} & set(table.columns)
    outcome = perform_regression(dict(
        screen_type="drug_screen", drug_data=str(csv), dependent_variable="intensity",
        drug_column="compound", concentration_column="dose_uM",
        vehicle_control=["DMSO"], concentration_unit="uM", agg_type="median",
        inference="nonparametric"))
    effects = outcome["effects"].set_index("treatment")
    assert effects.loc["ACT 100 uM", "significant"]
    assert outcome["res_folder"].startswith(str(tmp_path / "results" / "drug_screen"))


def test_a_treatment_seen_in_one_well_is_reported_but_not_tested(tmp_path):
    db, plate_csv = _screen(tmp_path)
    plate_map = pd.read_csv(plate_csv)
    plate_map.loc[(plate_map["plateID"] == "plate1") & (plate_map["wellID"] == "B01"),
                  ["drug", "concentration"]] = ["RARE", 3.0]
    plate_map.to_csv(plate_csv, index=False)
    effects = perform_regression(_settings(db, plate_csv))["effects"].set_index("treatment")
    assert effects.loc["RARE 3 uM", "status"] == "single well, not tested"
    assert np.isnan(effects.loc["RARE 3 uM", "p_value"])
    assert effects.loc["ACT 1 uM", "n_wells"] == 3


@pytest.mark.parametrize("change, message", [
    (dict(vehicle_control=None), "vehicle_control is empty"),
    (dict(concentration_unit=""), "concentration_unit is empty"),
    (dict(vehicle_control="water"), "add its drug value to vehicle_control"),
    (dict(drug_column="compound"), "has no compound column"),
    (dict(dependent_variable="area"), "is not a column"),
    (dict(dependent_variable=["intensity", "area"]), "one response at a time"),
    (dict(drug_data="missing.db"), "is not a file"),
])
def test_missing_drug_screen_inputs_are_named(tmp_path, change, message):
    db, plate_csv = _screen(tmp_path)
    with pytest.raises(ValueError, match=message):
        perform_regression(_settings(db, plate_csv, **change))


def test_plate_maps_that_cannot_support_the_comparison_are_refused(tmp_path):
    db, plate_csv = _screen(tmp_path)
    plate_map = pd.read_csv(plate_csv)
    no_vehicle = plate_map.copy()
    no_vehicle.loc[(no_vehicle["plateID"] == "plate2") & (no_vehicle["drug"] == "DMSO"),
                   ["drug", "concentration"]] = ["NUL", 1.0]
    no_vehicle.to_csv(plate_csv, index=False)
    with pytest.raises(ValueError, match="plate2 hold treated wells but no vehicle wells"):
        perform_regression(_settings(db, plate_csv))
    zero = plate_map.copy()
    zero.loc[zero["drug"] == "ACT", "concentration"] = 0
    zero.to_csv(plate_csv, index=False)
    with pytest.raises(ValueError, match="no positive concentration"):
        perform_regression(_settings(db, plate_csv))
    twice = pd.concat([plate_map, plate_map.iloc[[5]].assign(drug="NUL")])
    twice.to_csv(plate_csv, index=False)
    with pytest.raises(ValueError, match="more than one treatment"):
        perform_regression(_settings(db, plate_csv))
    lone = plate_map.copy()
    lone.loc[lone["drug"] == "DMSO", ["drug", "concentration"]] = ["NUL", 1.0]
    lone.loc[0, ["drug", "concentration"]] = ["DMSO", 0.0]
    lone.to_csv(plate_csv, index=False)
    with pytest.raises(ValueError, match="Found 1 vehicle well"):
        perform_regression(_settings(db, plate_csv))
    plate_map.to_csv(plate_csv, index=False)
    with pytest.raises(ValueError, match="compares wells"):
        perform_regression(_settings(db, plate_csv, analysis_unit="cell"))


def test_drug_screen_is_a_screen_type_that_inference_leaves_alone():
    for inference, mode in (("nonparametric", "guide_permutation"),
                            ("parametric", "regression")):
        settings = get_perform_regression_default_settings(
            dict(screen_type="drug_screen", inference=inference))
        assert settings["screen_type"] == "drug_screen"
        assert settings["analysis_mode"] == mode
    defaults = get_perform_regression_default_settings({})
    assert defaults["screen_type"] == "pooled"
    assert defaults["analysis_mode"] == "guide_permutation"
    assert defaults["vehicle_control"] is None and defaults["concentration_unit"] is None
    assert defaults["drug_column"] == "drug" and defaults["drug_table"] == "cell"


def test_effects_use_ordinary_least_squares_errors_with_an_optional_plate_term(tmp_path):
    """One plate, well names only: no plate term, and the numbers match plain OLS."""
    import statsmodels.api as sm

    db, plate_csv = _screen(tmp_path, cells=1)
    with sqlite3.connect(db) as connection:
        cells = pd.read_sql("SELECT * FROM cell", connection)
    plate_map = pd.read_csv(plate_csv)
    one = cells[cells["plateID"] == "plate1"].copy()
    one["wellID"] = [f"{'ABCD'[int(r[1:]) - 1]}{int(c[1:]):02d}"
                     for r, c in zip(one["rowID"], one["columnID"])]
    table = one.merge(plate_map[plate_map["plateID"] == "plate1"], on=["plateID", "wellID"])
    table = table[["wellID", "intensity", "drug", "concentration"]]
    csv = tmp_path / "one_plate.csv"
    table.to_csv(csv, index=False)
    outcome = perform_regression(dict(
        screen_type="drug_screen", drug_data=str(csv), dependent_variable="intensity",
        vehicle_control="DMSO", concentration_unit="uM"))
    effects = outcome["effects"].set_index("treatment")
    assert set(effects["plate_effect"]) == {False}
    levels = [f"{d} {c} uM" for d in ("ACT", "NUL") for c in (1, 10, 100)]
    label = [("vehicle" if d == "DMSO" else f"{d} {c:g} uM")
             for d, c in zip(table["drug"], table["concentration"])]
    design = pd.DataFrame({level: [float(x == level) for x in label] for level in levels})
    fit = sm.OLS(table["intensity"].to_numpy(), sm.add_constant(design)).fit()
    for level in levels:
        assert effects.loc[level, "effect"] == pytest.approx(fit.params[level])
        assert effects.loc[level, "standard_error"] == pytest.approx(fit.bse[level])
        assert effects.loc[level, "p_value"] == pytest.approx(fit.pvalues[level])
