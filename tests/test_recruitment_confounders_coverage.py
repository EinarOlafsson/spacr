"""Option paths of spacr.recruitment_confounders: skipped limits, absent
tables and columns, and the unlogged / partially logged designs."""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import recruitment_confounders as rc


def _cell_only_db(path):
    with sqlite3.connect(path) as con:
        con.execute("create table cell (plateID, rowID, columnID, fieldID)")
        for fld, n in (("1", 2), ("2", 3)):
            for _ in range(n):
                con.execute("insert into cell values (?,?,?,?)", ("P%255F1", "r1", "c1", fld))
        con.execute("insert into cell values (?,?,?,?)", ("P%255F1", "r1", "c2", "1"))


def test_well_density_without_a_pathogen_table_counts_only_cells(tmp_path):
    db = tmp_path / "m.db"
    _cell_only_db(db)
    wells = rc.well_density(str(db), pathogen_table=None).set_index("columnID")
    assert wells.loc["c1", "n_cells_well"] == 5
    assert wells.loc["c1", "n_fields_well"] == 2
    assert wells.loc["c2", "n_cells_well"] == 1
    assert "pathogens_per_cell_well" not in wells.columns
    # the doubly escaped plate id is decoded
    assert set(wells["plateID"]) == {"P_1"}
    # a named pathogen table that does not exist behaves the same
    fields = rc.field_density(str(db), pathogen_table="pathogen")
    assert "n_pathogens_field" not in fields.columns
    assert fields["n_cells_field"].sum() == 6


def test_apply_well_density_attaches_counts_and_leaves_unknown_wells_missing():
    wells = pd.DataFrame({"plateID": ["P%5F1"], "rowID": ["r1"], "columnID": ["c1"],
                          "n_cells_well": [5], "n_fields_well": [2]})
    df = pd.DataFrame({"plateID": ["P_1", "P_1"], "rowID": ["r1", "r1"],
                       "columnID": ["c1", "c9"], "n_cells_well": [0, 0], "x": [1, 2]})
    out = rc.apply_well_density(df, wells)
    assert out["n_cells_well"].tolist()[0] == 5
    assert np.isnan(out["n_cells_well"].tolist()[1])
    assert out["x"].tolist() == [1, 2]
    assert "n_cells_well" in df.columns and df["n_cells_well"].tolist() == [0, 0]


def test_robust_upper_is_infinite_without_values_or_spread():
    assert rc._robust_upper(pd.Series([np.nan, np.nan]), 3.0) == np.inf
    assert rc._robust_upper(pd.Series([2.0, 2.0, 2.0]), 3.0) == np.inf
    # median 2, MAD 1 * 1.4826
    assert rc._robust_upper(pd.Series([1.0, 2.0, 3.0]), 2.0) == pytest.approx(2 + 2 * 1.4826)


def _fields():
    return pd.DataFrame({"plateID": "P", "rowID": "r1", "columnID": "c1",
                         "fieldID": ["1", "2", "3"],
                         "n_cells_field": [50, 5, 40],
                         "pathogens_per_cell_field": [0.5, 1.0, 3.0]})


def test_flag_crowded_fields_with_only_an_absolute_ceiling():
    out = rc.flag_crowded_fields(_fields(), mad_threshold=None, min_cells=None,
                                 max_pathogens_per_cell=2.0, by=None)
    assert out["field_qc_pass"].tolist() == [True, True, False]
    assert out["field_qc_reason"].iat[2] == "pathogens_per_cell_field>2.0"
    assert not [c for c in out.columns if c.startswith("limit_")]


def test_flag_crowded_fields_skips_columns_that_are_absent():
    out = rc.flag_crowded_fields(_fields(), columns=("n_pathogens_field",), min_cells=10)
    assert "limit_n_pathogens_field" not in out.columns
    assert out["field_qc_reason"].tolist() == ["", "n_cells_field<10", ""]


def test_apply_field_qc_without_qc_columns_keeps_every_row():
    fields = _fields()
    df = pd.DataFrame({"plateID": "P", "rowID": "r1", "columnID": "c1",
                       "fieldID": ["1", "3", "9"], "v": [1, 2, 3]})
    out = rc.apply_field_qc(df, fields, keep_columns=("n_cells_field",))
    assert len(out) == 3 and "field_qc_pass" not in out.columns
    assert out["n_cells_field"].tolist()[:2] == [50, 40]
    assert np.isnan(out["n_cells_field"].tolist()[2])


def _objects(seed=0, n=60):
    rng = np.random.default_rng(seed)
    cond = np.repeat(["a", "b"], n // 2)
    cov = rng.uniform(1, 10, n) + (cond == "b") * 4.0
    y = 2.0 + 0.5 * cov + (cond == "b") * 1.0 + rng.normal(0, 0.05, n)
    return pd.DataFrame({"y": y, "cond": cond, "cov": cov, "other": rng.normal(0, 1, n)})


def test_adjust_without_logs_removes_the_linear_covariate_effect():
    df = _objects()
    out, model = rc.adjust_for_covariates(df, "y", "cond", ["cov", "missing"],
                                          log_outcome=False, log_covariates=False)
    assert model["covariate"].tolist() == ["cov"]
    sd = model["sd"].iat[0]
    assert model["beta_per_sd"].iat[0] == pytest.approx(0.5 * sd, rel=0.02)
    adj = out.groupby("cond")["y_adjusted"].mean()
    assert adj["b"] - adj["a"] == pytest.approx(1.0, abs=0.05)


def test_adjust_with_a_list_of_logged_covariates():
    df = _objects()
    _, model = rc.adjust_for_covariates(df, "y", "cond", ["cov", "other"],
                                        log_outcome=False, log_covariates=["cov"],
                                        out_column="yy")
    scales = model.set_index("covariate")
    # 'cov' is log1p-transformed before standardising, 'other' is not
    assert scales.loc["cov", "mean"] == pytest.approx(np.log1p(df["cov"]).mean())
    assert scales.loc["other", "mean"] == pytest.approx(df["other"].mean())


def test_adjust_without_usable_covariates_copies_the_outcome():
    df = pd.DataFrame({"y": ["1.5", "x"], "cond": ["a", "b"]})
    out, model = rc.adjust_for_covariates(df, "y", "cond", ["absent"])
    assert out["y_adjusted"].iloc[0] == 1.5 and np.isnan(out["y_adjusted"].iloc[1])
    assert model.empty
    assert list(model.columns) == ["covariate", "beta_per_sd", "se", "t", "mean", "sd", "n"]


def test_confounder_report_skips_absent_and_constant_covariates():
    df = _objects().assign(flat=1.0)
    report = rc.confounder_report(df, "y", "cond", ["absent", "flat", "cov"])
    assert report["covariate"].tolist() == ["cov"]
    assert bool(report["likely_confounder"].iat[0]) is True
    assert report["imbalance_eta2"].iat[0] > 0.1
    assert rc.confounder_report(df, "y", "cond", ["absent"]).empty
