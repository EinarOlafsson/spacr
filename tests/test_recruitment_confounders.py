"""Tests for spacr.recruitment_confounders and its analyze_recruitment wiring."""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.recruitment_confounders import (
    adjust_for_covariates, apply_field_qc, confounder_report,
    field_density, flag_crowded_fields)


def _make_db(path, plate="P_1"):
    """Fields: f1 has 2 cells/2 pathogens, f2 has 3 cells/1 pathogen."""
    with sqlite3.connect(path) as con:
        con.execute("create table cell (plateID, rowID, columnID, fieldID, cell_id)")
        con.execute("create table pathogen (plateID, rowID, columnID, fieldID, cell_id)")
        for fld, n_cell, cell_ids in (("1", 2, [1, 2]), ("2", 3, [1])):
            for i in range(n_cell):
                con.execute("insert into cell values (?,?,?,?,?)", (plate, "r1", "c1", fld, i + 1))
            for cid in cell_ids:
                con.execute("insert into pathogen values (?,?,?,?,?)", (plate, "r1", "c1", fld, cid))
        con.execute("insert into pathogen values (?,?,?,?,?)", (plate, "r1", "c1", "2", 0))


def test_field_density_counts(tmp_path):
    db = tmp_path / "m.db"
    _make_db(db, plate="P%5F1")
    f = field_density(str(db)).set_index("fieldID")
    assert f.loc["1", "n_cells_field"] == 2 and f.loc["1", "n_pathogens_field"] == 2
    assert f.loc["2", "n_cells_field"] == 3 and f.loc["2", "n_pathogens_field"] == 2
    assert f.loc["2", "n_pathogens_in_cells_field"] == 1
    assert f.loc["1", "pathogens_per_cell_field"] == pytest.approx(1.0)
    assert (f["plateID"] == "P_1").all()


def _fields():
    n = 12
    df = pd.DataFrame(dict(plateID="P_1", rowID="r1", columnID="c1",
                           fieldID=[str(i) for i in range(n)],
                           n_cells_field=40, n_pathogens_field=np.linspace(38, 44, n)))
    df["pathogens_per_cell_field"] = df["n_pathogens_field"] / df["n_cells_field"]
    df.loc[0, ["n_pathogens_field", "pathogens_per_cell_field"]] = [400, 10.0]  # over-infected
    df.loc[1, "n_cells_field"] = 3  # sparse
    return df


def test_flag_crowded_fields_flags_over_infected_and_sparse():
    out = flag_crowded_fields(_fields())
    bad = out.set_index("fieldID")
    assert not bad.loc["0", "field_qc_pass"] and "n_pathogens_field" in bad.loc["0", "field_qc_reason"]
    assert not bad.loc["1", "field_qc_pass"] and "n_cells_field<10" in bad.loc["1", "field_qc_reason"]
    assert bad.drop(index=["0", "1"])["field_qc_pass"].all()


def test_apply_field_qc_joins_escaped_plate_id():
    fields = flag_crowded_fields(_fields())
    objs = pd.DataFrame(dict(plateID="P%5F1", rowID="r1", columnID="c1",
                             fieldID=["0", "1", "2", "99"], recruitment=1.0))
    kept = apply_field_qc(objs, fields)
    # 0 and 1 fail; 2 passes; 99 is unknown and kept
    assert sorted(kept["fieldID"]) == ["2", "99"]
    assert "n_cells_field" in kept.columns
    assert apply_field_qc(objs, fields, drop_failed=False).shape[0] == 4


def _synthetic(seed=0, effect=0.30, n=600):
    rng = np.random.default_rng(seed)
    cond = np.repeat(["ctrl", "mut"], n // 2)
    # covariate imbalanced between conditions
    cov = rng.normal(0, 1, n) + np.where(cond == "mut", 1.5, 0.0)
    log_y = 0.4 * cov + np.where(cond == "mut", effect, 0.0) + rng.normal(0, 0.1, n)
    return pd.DataFrame(dict(condition=cond, depth=cov + 5.0, recruitment=np.exp(log_y)))


def test_adjust_for_covariates_recovers_condition_effect():
    df = _synthetic()
    raw = np.log(df.groupby("condition")["recruitment"].mean())
    naive = raw["mut"] - raw["ctrl"]
    out, model = adjust_for_covariates(df, "recruitment", "condition", ["depth"], log_covariates=False)
    adj = np.log(out.groupby("condition")["recruitment_adjusted"].mean())
    est = adj["mut"] - adj["ctrl"]
    assert abs(naive - 0.30) > 0.3          # confounded
    assert est == pytest.approx(0.30, abs=0.05)
    assert model.loc[0, "beta_per_sd"] > 0
    rep = confounder_report(df, "recruitment", "condition", ["depth"], log_covariates=False)
    assert bool(rep.loc[0, "likely_confounder"])


def test_analyze_recruitment_field_qc_and_adjustment(tmp_path):
    import os
    from tests.test_cov_submodules_recruitment_plaques import (  # reuse fixture helpers
        _merged_recruitment_df, _base_settings, _Recorder)
    import spacr.io as SIO
    import spacr.plot as SPLT

    mp = pytest.MonkeyPatch()
    try:
        src = tmp_path / "plate"
        (src / "measurements").mkdir(parents=True)
        df = _merged_recruitment_df(n_cells=6)
        df["recruitment"] = 1.0  # overwritten by the pipeline
        # vary cytoplasm intensity so recruitment varies per object
        for chan in range(4):
            df[f"cytoplasm_channel_{chan}_mean_intensity"] += np.arange(len(df)) % 5
        mp.setattr(SIO, "_read_and_merge_data", lambda locs, tables, **k: (df.copy(), []))
        for name in ("plot_image_mask_overlay", "_plot_controls", "_plot_recruitment"):
            mp.setattr(SPLT, name, _Recorder())
        # db: field "1" of both wells; one well is over-crowded with parasites
        with sqlite3.connect(src / "measurements" / "measurements.db") as con:
            con.execute("create table cell (plateID, rowID, columnID, fieldID, cell_id)")
            con.execute("create table pathogen (plateID, rowID, columnID, fieldID, cell_id)")
            for rid, cid, n_cell, n_path in (("r1", "c1", 20, 20), ("r4", "c4", 20, 400)):
                for i in range(n_cell):
                    con.execute("insert into cell values (?,?,?,?,?)", ("plate1", rid, cid, "1", i + 1))
                for i in range(n_path):
                    con.execute("insert into pathogen values (?,?,?,?,?)", ("plate1", rid, cid, "1", i % n_cell + 1))
            # extra sane fields so the robust limit is defined
            for f in range(2, 8):
                for rid, cid in (("r1", "c1"), ("r4", "c4")):
                    for i in range(20 + f):
                        con.execute("insert into cell values (?,?,?,?,?)", ("plate1", rid, cid, str(f), i + 1))
                    for i in range(20 + f + (f % 3)):
                        con.execute("insert into pathogen values (?,?,?,?,?)", ("plate1", rid, cid, str(f), i % (20 + f) + 1))
        from spacr.submodules import analyze_recruitment
        cells, wells = analyze_recruitment(_base_settings(
            src, field_qc=True, confounder_covariates=["n_cells_field", "pathogens_per_cell_field"],
            adjust_covariates=["pathogens_per_cell_field"], field_qc_min_cells=5))
    finally:
        mp.undo()
    res = src / "results"
    fq = pd.read_csv(res / "field_qc.csv")
    assert not fq["field_qc_pass"].all()
    assert (res / "confounder_report.csv").exists() and (res / "covariate_model.csv").exists()
    assert "recruitment_adjusted" in cells.columns and "recruitment_adjusted" in wells.columns
    # the over-infected field (r4/c4 field 1) was removed
    assert len(cells) < 12
