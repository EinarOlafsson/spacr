"""Coverage-fill for spacr.sim pure-logic simulation helpers (no GPU)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import sim as SIM


# ---------------------------------------------------------------------------
# gene list / plate map
# ---------------------------------------------------------------------------

def test_generate_gene_list():
    out = SIM.generate_gene_list(5, 20)
    assert len(out) == 5 and len(set(out)) == 5
    assert all(0 <= g < 20 for g in out)


def test_generate_plate_map():
    df = SIM.generate_plate_map(2)
    assert len(df) == 2 * 16 * 24
    assert {"plate_id", "row_id", "column_id", "plate_row_column"} <= set(df.columns)


# ---------------------------------------------------------------------------
# gini variants
# ---------------------------------------------------------------------------

def test_gini_variants_equal_vs_unequal():
    equal = np.ones(10)
    unequal = np.array([0.0] * 9 + [100.0])
    for fn in (SIM.gini_coefficient, SIM.gini_gene_well, SIM.gini):
        g_eq = fn(equal)
        g_un = fn(unequal)
        assert abs(g_eq) < 0.05          # near-zero for equality
        assert g_un > g_eq               # more inequality → larger gini


# ---------------------------------------------------------------------------
# distribution generators
# ---------------------------------------------------------------------------

def test_dist_gen():
    df = pd.DataFrame({"a": range(50)})
    data, length = SIM.dist_gen(10, 3, df)
    assert length == 50 and len(data) == 50


def test_generate_gene_weights():
    df = pd.DataFrame({"a": range(100)})
    w = SIM.generate_gene_weights(0.5, 0.05, df)
    assert len(w) == 100 and (0 <= w).all() and (w <= 1).all()


def test_normalize_array():
    out = SIM.normalize_array(np.array([2.0, 4.0, 6.0]))
    assert out.min() == 0.0 and out.max() == 1.0


def test_generate_power_law_distribution():
    d = SIM.generate_power_law_distribution(10, 1.5)
    assert len(d) == 10 and abs(d.sum() - 1.0) < 1e-9


def test_power_law_dist_gen():
    df = pd.DataFrame({"a": range(30)})
    dist = SIM.power_law_dist_gen(df, avg=100, well_ineq_coeff=1.2)
    assert len(dist) == 30


# ---------------------------------------------------------------------------
# ROC / PR helpers
# ---------------------------------------------------------------------------

def _cell_scores(n=100):
    rng = np.random.default_rng(0)
    is_active = rng.integers(0, 2, n)
    # score correlated with is_active so curves are well-defined
    score = np.clip(is_active * 0.5 + rng.normal(0.25, 0.2, n), 0, 1)
    return pd.DataFrame({"is_active": is_active, "score": score})


def test_compute_roc_auc():
    d = SIM.compute_roc_auc(_cell_scores())
    assert "roc_auc" in d and 0.0 <= d["roc_auc"] <= 1.0


def test_compute_precision_recall_and_optimum():
    pr = SIM.compute_precision_recall(_cell_scores())
    assert "pr_auc" in pr
    opt = SIM.get_optimum_threshold(pr)
    assert 0.0 <= opt <= 1.0


def test_update_scores_and_get_cm():
    cs = _cell_scores()
    cs2, cm = SIM.update_scores_and_get_cm(cs, 0.5)
    assert cm.shape == (2, 2)


def test_cell_level_roc_auc():
    roc_df, pr_df, cs, cm = SIM.cell_level_roc_auc(_cell_scores())
    assert isinstance(roc_df, pd.DataFrame) and cm.shape == (2, 2)


def test_generate_well_score():
    rng = np.random.default_rng(1)
    n = 60
    cs = pd.DataFrame({
        "plate_row_column": rng.choice(["1_1_1", "1_1_2", "1_2_1"], n),
        "is_active": rng.integers(0, 2, n),
        "gene_id": rng.integers(0, 5, n),
    })
    ws = SIM.generate_well_score(cs)
    assert "score" in ws.columns and "gene_list" in ws.columns


# ---------------------------------------------------------------------------
# param / misc helpers
# ---------------------------------------------------------------------------

def test_validate_and_adjust_beta_params():
    params = [{
        "positive_mean": 0.5, "negative_mean": 0.5,
        "positive_variance": 10.0,   # infeasible → capped
        "negative_variance": 10.0,
    }]
    out = SIM.validate_and_adjust_beta_params(params)
    assert out[0]["positive_variance"] < 0.5
    assert out[0]["negative_variance"] < 0.5


def test_generate_integers_and_floats():
    assert SIM.generate_integers(0, 10, 2) == [0, 2, 4, 6, 8, 10]
    floats = SIM.generate_floats(0.0, 1.0, 0.25)
    assert floats == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_remove_columns_with_single_value():
    df = pd.DataFrame({"const": [1, 1, 1], "vary": [1, 2, 3]})
    out = SIM.remove_columns_with_single_value(df)
    assert list(out.columns) == ["vary"]


def test_remove_constant_columns():
    df = pd.DataFrame({"const": [5, 5], "vary": [1, 2]})
    out = SIM.remove_constant_columns(df)
    assert list(out.columns) == ["vary"]


def test_read_simulations_table(tmp_path):
    import sqlite3
    db = tmp_path / "s.db"
    with sqlite3.connect(str(db)) as conn:
        pd.DataFrame({"a": [1, 2]}).to_sql("simulations", conn, index=False)
    out = SIM.read_simulations_table(str(db))
    assert len(out) == 2
    # missing table → None
    empty = tmp_path / "e.db"
    sqlite3.connect(str(empty)).close()
    assert SIM.read_simulations_table(str(empty)) is None


# ---------------------------------------------------------------------------
# simulation engine chain: run_experiment -> classifier -> sequence_plates
# ---------------------------------------------------------------------------

def test_classifier_and_errors():
    df = pd.DataFrame({"is_active": [1, 0, 1, 0, 1, 0]})
    out = SIM.classifier(0.8, 0.02, 0.2, 0.02, 0.9, df)
    assert "score" in out.columns and out["score"].between(0, 1).all()
    # invalid mean
    with pytest.raises(ValueError):
        SIM.classifier(1.5, 0.02, 0.2, 0.02, 0.9, df.copy())
    # invalid variance (>= mean*(1-mean))
    with pytest.raises(ValueError):
        SIM.classifier(0.5, 1.0, 0.2, 0.02, 0.9, df.copy())


def test_run_experiment_chain():
    plate_map = SIM.generate_plate_map(1)   # one 384-well plate
    active = list(range(1, 6))
    cell_df, gpw_df, wpg_df, df_ls = SIM.run_experiment(
        plate_map, number_of_genes=20, active_gene_list=active,
        avg_genes_per_well=5, sd_genes_per_well=2,
        avg_cells_per_well=10, sd_cells_per_well=3,
        well_ineq_coeff=1.2, gene_ineq_coeff=1.2)
    assert {"gene_id", "is_active", "plate_row_column"} <= set(cell_df.columns)
    assert len(df_ls) == 6

    # classifier assigns scores
    scored = SIM.classifier(0.8, 0.02, 0.2, 0.02, 0.85, cell_df)
    assert "score" in scored.columns

    # well scores + sequencing
    well_score = SIM.generate_well_score(scored)
    frac_map, metadata = SIM.sequence_plates(
        well_score, number_of_genes=20,
        avg_reads_per_gene=100, sd_reads_per_gene=20,
        sequencing_error=0.05)
    assert "sum_reads" in metadata.columns
    assert len(frac_map) == len(well_score)
