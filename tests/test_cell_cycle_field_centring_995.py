"""Item 995: field centring and an external mitotic call for Measure's DNA fit.

Fields of nuclei with planted log2 offsets and known G1/S/G2 shares check that
centring every field on its G1 peak restores the 2:1 peak ratio and the fitted
weights, that the uncentred fit is broader, that small fields are not
shifted, that an external mitotic call is left out of the peaks and the fit
and called M, and that a per-well fit is made once per well. The defaults
leave the Measure output exactly as before, and the OPS ``measure_fc`` path
calls the same function on the cell tables.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import measure
from spacr.measure import (
    _CELL_CYCLE_FIT_TABLE,
    _cell_cycle_fit_groups,
    _cell_cycle_fit_rows,
    _cell_table_phases_measure_fc,
    _classify_cell_cycle,
    _dna_field_offsets,
    _external_mitotic_call,
    _field_centred_dna_phases,
    _phases_by_measurements,
)

SHARES = (0.60, 0.25, 0.15)
FIELD = 900


def _fields(n, *, sd_off=0.10, cv=0.041, seed=0, shares=SHARES, field=FIELD):
    """DNA content of ``n`` nuclei in fields with planted log2 offsets."""
    rng = np.random.default_rng(seed)
    k = rng.choice(3, n, p=shares)
    c = np.where(k == 0, 1.0, np.where(k == 2, 2.0, rng.uniform(1, 2, n)))
    c = np.clip(c * (1 + rng.normal(0, cv, n)), 0.05, None)
    site = np.arange(n) // field
    off = rng.normal(0, sd_off, site.max() + 1)[site] if sd_off else 0.0
    content = 2.0 ** (np.log2(c) + 22.8 + off)
    return content, site, np.array(["G1", "S", "G2"])[k]


def _one_group(n):
    return [("w", np.arange(n))]


def _weights(fit):
    w = np.array(fit.weights[:3])
    return w / w.sum()


def test_centring_recovers_the_peak_ratio_and_the_weights():
    n = 100_000
    content, site, truth = _fields(n)
    frame, fits = _field_centred_dna_phases(
        content, groups=_one_group(n), fields=site)
    fit = fits["w"]["fit"]
    assert fit.g2 / fit.g1 == pytest.approx(2.0, abs=0.05)
    for share, (name, weight) in zip(
            SHARES, zip(("G1", "S", "G2"), _weights(fit))):
        assert abs(weight - share) <= 0.01, (name, weight, share)
    assert fits["w"]["fields_shifted"] == len(np.unique(site))
    assert np.std(frame["field_offset_log2"]) > 0.05
    assert set(frame["phase"]) <= {"G1", "S", "G2", "subG1", ">4N"}


def test_without_centring_the_same_fields_give_broader_peaks():
    n = 100_000
    content, site, _ = _fields(n)
    _, centred = _field_centred_dna_phases(
        content, groups=_one_group(n), fields=site)
    raw, pooled = _field_centred_dna_phases(
        content, groups=_one_group(n), fields=site, centre=False)
    cv = {name: f["w"]["fit"].sd1 / f["w"]["fit"].g1
          for name, f in (("centred", centred), ("pooled", pooled))}
    assert cv["pooled"] > cv["centred"] * 1.3, cv
    assert (raw["field_offset_log2"] == 0).all()
    assert np.array_equal(raw["dna_content_centred"].to_numpy(), content)


def test_a_field_below_the_minimum_is_not_shifted():
    rng = np.random.default_rng(1)
    log_dna = np.concatenate([rng.normal(22.8, 0.05, 400),
                              rng.normal(23.3, 0.05, 400),
                              rng.normal(24.0, 0.05, 49)])
    fields = np.repeat([0, 1, 2], [400, 400, 49])
    offsets, shifted = _dna_field_offsets(
        log_dna, fields, np.ones(log_dna.size, bool), min_cells=50)
    assert shifted == 2
    assert (offsets[fields == 2] == 0).all()
    assert offsets[0] == pytest.approx(-offsets[400])
    assert abs(offsets[400] - offsets[0]) == pytest.approx(0.5, abs=0.05)
    offsets, shifted = _dna_field_offsets(
        log_dna, fields, np.ones(log_dna.size, bool), min_cells=49)
    assert shifted == 3 and (offsets[fields == 2] != 0).all()


def test_mitotic_nuclei_are_left_out_of_the_peaks_and_the_fit_and_called_m():
    n = 9_000
    content, site, _ = _fields(n, seed=2)
    mitotic = np.zeros(n, bool)
    mitotic[::10] = True
    bright = content.copy()
    bright[mitotic] *= 3.5
    with_call, fits = _field_centred_dna_phases(
        bright, groups=_one_group(n), fields=site, mitotic=mitotic)
    alone, alone_fits = _field_centred_dna_phases(
        content[~mitotic], groups=_one_group(int((~mitotic).sum())),
        fields=site[~mitotic])
    assert (with_call["phase"][mitotic] == "M").all()
    assert with_call["phase"][~mitotic].tolist() == alone["phase"].tolist()
    assert np.array_equal(with_call["field_offset_log2"][~mitotic],
                          alone["field_offset_log2"])
    assert fits["w"]["fit"].n == alone_fits["w"]["fit"].n
    assert fits["w"]["fit"].g1 == alone_fits["w"]["fit"].g1


def test_fields_with_one_offset_give_the_calls_of_no_centring():
    n = 20_000
    content, site, _ = _fields(n, sd_off=0.0, seed=3)
    content = content * 1.37
    centred, _ = _field_centred_dna_phases(
        content, groups=_one_group(n), fields=np.zeros(n, int))
    plain, _ = _field_centred_dna_phases(
        content, groups=_one_group(n), centre=False)
    assert (centred["field_offset_log2"] == 0).all()
    assert centred["phase"].tolist() == plain["phase"].tolist()


def _nuclei(n_per_field=300, wells=("c1", "c2"), fields=3, seed=4,
            mitotic_share=0.05):
    rng = np.random.default_rng(seed)
    parts = []
    label = 0
    for w, column in enumerate(wells):
        for f in range(fields):
            content, _, _ = _fields(n_per_field, sd_off=0.0, seed=seed + 10 * w + f)
            content = content / 2 ** 15 * 2 ** rng.normal(0, 0.1)
            area = np.full(n_per_field, 300.0)
            ring = np.full(n_per_field, 10.0)
            parts.append(pd.DataFrame({
                "object_label": np.arange(label + 1, label + n_per_field + 1),
                "plateID": "plate1", "rowID": "r1", "columnID": column,
                "fieldID": f"f{f}", "nucleus_area": area,
                "nucleus_channel_0_integrated_intensity": content + ring * area,
                "nucleus_channel_0_mean_intensity": content / area + ring,
                "nucleus_channel_0_outside_percentile_50": ring,
                "mitotic_856": rng.random(n_per_field) < mitotic_share}))
            label += n_per_field
    table = pd.concat(parts, ignore_index=True)
    table["prcf"] = (table["plateID"] + "_" + table["rowID"] + "_"
                     + table["columnID"] + "_" + table["fieldID"])
    return table


def test_fit_group_well_fits_once_per_well():
    table = _nuclei()
    groups = _cell_cycle_fit_groups(table, "well")
    assert [name for name, _ in groups] == [
        ("plate1", "r1", "c1"), ("plate1", "r1", "c2")]
    assert [name for name, _ in _cell_cycle_fit_groups(table, "plate")] == [
        ("plate1",)]
    measured, fits = _phases_by_measurements(
        table, column=0, field_centre=True, fit_group="well")
    assert set(fits) == {("plate1", "r1", "c1"), ("plate1", "r1", "c2")}
    rows = measured.attrs["cell_cycle_fits"]
    assert len(rows) == 2 and (rows["fields_shifted"] == 3).all()
    assert set(rows["fit_group"]) == {"well"}
    with pytest.raises(ValueError, match="cell_cycle_fit_group"):
        _cell_cycle_fit_groups(table, "field")
    with pytest.raises(ValueError, match="cell_cycle_field_min_cells"):
        _phases_by_measurements(table, column=0, field_centre=True,
                                field_min_cells=0)


def test_the_external_call_replaces_the_condensation_rule():
    table = _nuclei()
    mitotic = _external_mitotic_call(table, "mitotic_856")
    measured, _ = _phases_by_measurements(
        table, column=0, field_centre=True, fit_group="well",
        mitotic=mitotic, mitotic_ratio=0.0)
    phases = measured["phase_measurements"].to_numpy()
    assert (phases[mitotic] == "M").all()
    assert not (phases[~mitotic] == "M").any()
    assert _external_mitotic_call(table, "") is None
    assert _external_mitotic_call(
        pd.DataFrame({"m": ["True", "false", None, 1]}), "m").tolist() == [
            True, False, False, True]
    with pytest.raises(ValueError, match="cell_cycle_mitotic_column"):
        _external_mitotic_call(table, "no_such_column")


def _db(tmp_path, table):
    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as conn:
        table.to_sql("nucleus", conn, index=False)
    return str(db)


def _read(db, name):
    with sqlite3.connect(db) as conn:
        return pd.read_sql_query(f"SELECT * FROM {name}", conn)


def test_the_defaults_leave_the_measure_output_unchanged(tmp_path):
    table = _nuclei()
    db = _db(tmp_path, table)
    base = {"channels": [0], "cell_cycle_method": "measurements",
            "plot": False}
    plain, plain_report = _classify_cell_cycle(db, dict(base))
    first = _read(db, "cell_cycle")
    defaults, default_report = _classify_cell_cycle(db, {
        **base, "cell_cycle_field_centre": False,
        "cell_cycle_field_min_cells": 50, "cell_cycle_fit_group": "plate",
        "cell_cycle_mitotic_column": ""})
    pd.testing.assert_frame_equal(plain, defaults)
    pd.testing.assert_frame_equal(first, _read(db, "cell_cycle"))
    assert plain_report == default_report
    assert "field_offset_log2" not in plain
    with sqlite3.connect(db) as conn:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert _CELL_CYCLE_FIT_TABLE not in names


def test_a_settings_rerun_of_the_centred_path_is_identical(tmp_path):
    table = _nuclei()
    db = _db(tmp_path, table)
    settings = {"channels": [0], "cell_cycle_method": "measurements",
                "plot": False, "cell_cycle_field_centre": True,
                "cell_cycle_fit_group": "well",
                "cell_cycle_mitotic_column": "mitotic_856"}
    first, report = _classify_cell_cycle(db, dict(settings))
    tables = {name: _read(db, name) for name in (
        "cell_cycle", "cell_cycle_well", _CELL_CYCLE_FIT_TABLE)}
    again, again_report = _classify_cell_cycle(db, dict(settings))
    pd.testing.assert_frame_equal(first, again)
    for name, before in tables.items():
        pd.testing.assert_frame_equal(before, _read(db, name))
    assert report == again_report
    assert {"field_offset_log2", "dna_content_centred"} <= set(first)
    assert report["fit_group"] == "well" and len(report["fits"]) == 2
    assert (first.loc[table["mitotic_856"].to_numpy(),
                      "phase_measurements"] == "M").all()


def test_the_ops_measure_fc_path_calls_the_same_function_per_well():
    table = _nuclei()
    content = (table["nucleus_channel_0_integrated_intensity"]
               - table["nucleus_channel_0_outside_percentile_50"]
               * table["nucleus_area"])
    cells = pd.DataFrame({
        "well": table["columnID"].to_numpy(),
        "site": table["fieldID"].to_numpy(),
        "mitotic": table["mitotic_856"].to_numpy(),
        "_dna_integrated": content.to_numpy()},
        index=np.arange(len(table)) + 100)
    frame, fits = _cell_table_phases_measure_fc(cells)
    measured, _ = _phases_by_measurements(
        table, column=0, field_centre=True, fit_group="well",
        mitotic=table["mitotic_856"].to_numpy())
    assert frame.index.equals(cells.index)
    assert frame["phase"].tolist() == measured["phase_measurements"].tolist()
    rows = _cell_cycle_fit_rows(fits, "well")
    assert rows["group"].tolist() == ["c1", "c2"]
    with pytest.raises(ValueError, match="site"):
        _cell_table_phases_measure_fc(cells.drop(columns="site"))


def test_the_ops_profile_method_is_validated_and_folds_the_phases():
    from spacr import ops_engine

    assert ops_engine._MEASURE_FC_PROFILE_PHASES == {
        "G1": "G1", "subG1": "G1", "S": "SG2", "G2": "SG2", ">4N": "SG2",
        "M": "M"}
    cells = pd.DataFrame({"well": ["A1"], "sgRNA": ["g"], "gene": ["x"]})
    with pytest.raises(ValueError, match="ops_profile_cell_cycle_method"):
        ops_engine._ph_profile_options(
            {"ops_profile_cell_cycle_method": "other"}, cells,
            np.zeros((1, 1)), pd.DataFrame(), pd.DataFrame({"gene": []}),
            "", "p")


def test_the_new_settings_are_alpha_and_off_by_default():
    from spacr import ops_settings, settings

    measure_defaults = settings.get_measure_crop_settings({})
    assert measure_defaults["cell_cycle_field_centre"] is False
    assert measure_defaults["cell_cycle_field_min_cells"] == 50
    assert measure_defaults["cell_cycle_fit_group"] == "plate"
    assert measure_defaults["cell_cycle_mitotic_column"] == ""
    assert ops_settings.OPS_DEFAULTS["ops_profile_cell_cycle_method"] == ""
    assert set(settings.ALPHA_FEATURES[995]["settings"]) == {
        "cell_cycle_field_centre", "cell_cycle_field_min_cells",
        "cell_cycle_fit_group", "cell_cycle_mitotic_column",
        "ops_profile_cell_cycle_method"}
    assert measure._CELL_CYCLE_FIELD_MIN_CELLS == 50
