"""Behavioural tests for :mod:`spacr.sim` — the pooled-screen simulator.

Everything in ``sim.py`` is random-number driven, so these tests seed both the
``random`` and ``numpy.random`` global generators and then assert on things that
are invariant given the seed: exact shapes, exact row/column membership,
conservation laws (read totals, fraction sums), and distributional properties
that hold with an enormous margin (mean of 10k Poisson draws, ROC-AUC of a
perfectly separable classifier).

Several tests below are regression tests for bugs found while writing them;
they are marked with a ``BUG:`` comment naming the defect they pin down.
"""
from __future__ import annotations

import copy
import random
import sqlite3

import numpy as np
import pandas as pd
import pytest

import matplotlib
@pytest.fixture(autouse=True)
def _never_write_figures_into_the_repo(tmp_path, monkeypatch):
    """Run every test in this file from a temp directory.

    sim's plotting entry points default to the RELATIVE folder ``figures``,
    which resolves against the process's current directory. The tests that do
    not chdir themselves were therefore writing ``figures/feature_importance``
    and ``figures/permutation_importance`` into the repo working tree. Tests
    that chdir on their own are unaffected -- they simply chdir again.
    """
    monkeypatch.chdir(tmp_path)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spacr import sim as S


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _seed(n: int = 0) -> None:
    """Seed both global RNGs that sim.py draws from."""
    random.seed(n)
    np.random.seed(n)


def _settings(**overrides):
    """A small but complete settings dict for run_simulation/run_and_save."""
    base = dict(
        name="unit",
        variable="classifier_accuracy",
        src="unset",
        plot=False,
        start_time="260101",
        number_of_genes=30,
        number_of_active_genes=10,
        number_of_control_genes=5,
        nr_plates=1,
        avg_genes_per_well=5,
        sd_genes_per_well=2,
        avg_cells_per_well=10,
        sd_cells_per_well=3,
        well_ineq_coeff=1.2,
        gene_ineq_coeff=1.2,
        positive_mean=0.8,
        positive_variance=0.02,
        negative_mean=0.2,
        negative_variance=0.02,
        classifier_accuracy=0.9,
        avg_reads_per_gene=100,
        sd_reads_per_gene=20,
        sequencing_error=0.01,
    )
    base.update(overrides)
    return base


@pytest.fixture(scope="module")
def sim_output():
    """One full simulation, computed once (~0.3 s) and deep-copied per use."""
    _seed(7)
    output, dists = S.run_simulation(_settings())
    return output, dists


@pytest.fixture
def sim_out(sim_output):
    output, dists = sim_output
    return copy.deepcopy(output), copy.deepcopy(dists)


def _small_plate_map(n_wells=6):
    """A hand-built plate map — run_experiment only needs these four columns."""
    rows = [f"1_1_{c}" for c in range(1, n_wells + 1)]
    df = pd.DataFrame({"plate_row_column": rows})
    df["plate_id"], df["row_id"], df["column_id"] = zip(*[r.split("_") for r in rows])
    return df


# ===========================================================================
# distribution primitives
# ===========================================================================

def test_dist_gen_matches_requested_mean():
    """Gamma-Poisson mixture: sample mean tracks the requested mean."""
    _seed(1)
    df = pd.DataFrame({"x": range(20000)})
    data, length = S.dist_gen(mean=40.0, sd=10.0, df=df)
    assert length == 20000
    assert data.shape == (20000,)
    assert data.dtype.kind in "iu"          # Poisson draws are integers
    assert (data >= 0).all()
    assert data.mean() == pytest.approx(40.0, rel=0.03)
    # Var(Poisson(Gamma)) = mean + sd^2 = 40 + 100 = 140
    assert data.var() == pytest.approx(140.0, rel=0.10)


def test_generate_gene_weights_matches_requested_moments():
    _seed(2)
    df = pd.DataFrame({"x": range(50000)})
    w = S.generate_gene_weights(positive_mean=0.7, positive_variance=0.02, df=df)
    assert w.shape == (50000,)
    assert w.min() > 0.0 and w.max() < 1.0
    assert w.mean() == pytest.approx(0.7, abs=0.01)
    assert w.var() == pytest.approx(0.02, abs=0.002)


def test_power_law_dist_gen_scales_by_avg():
    """Every draw is one of the power-law probabilities multiplied by avg."""
    _seed(3)
    df = pd.DataFrame({"x": range(40)})
    dist = S.power_law_dist_gen(df, avg=250.0, well_ineq_coeff=1.3)
    probs = S.generate_power_law_distribution(40, 1.3)
    assert dist.shape == (40,)
    allowed = set(np.round(probs * 250.0, 12))
    assert set(np.round(dist, 12)) <= allowed
    assert dist.max() <= probs.max() * 250.0 + 1e-12


def test_gini_variants_agree_on_the_same_sample():
    """gini_gene_well is the memory-cheap form of gini_coefficient."""
    rng = np.random.default_rng(11)
    x = rng.uniform(1.0, 100.0, size=200)
    assert S.gini_gene_well(x) == pytest.approx(S.gini_coefficient(x), rel=1e-9)
    # The ranked formulation agrees to within the 1/n small-sample offset.
    assert S.gini(x) == pytest.approx(S.gini_coefficient(x), abs=1.0 / len(x))


def test_gini_two_point_distribution_exact():
    """n-1 zeros and one positive value → gini = (n-1)/n exactly."""
    x = np.array([0.0] * 9 + [10.0])
    assert S.gini_coefficient(x) == pytest.approx(0.9)
    assert S.gini_gene_well(x) == pytest.approx(0.9)


def test_generate_floats_integer_step_is_not_rounded_to_tens():
    """BUG (fixed): str(1) has no '.', so find('.') returned -1 and every value
    was rounded to the nearest ten — generate_floats(0, 5, 1) gave all zeros."""
    assert S.generate_floats(0, 5, 1) == [0, 1, 2, 3, 4, 5]
    assert S.generate_floats(0.0, 1.0, 0.25) == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_generate_floats_reaches_the_inclusive_upper_bound():
    """BUG (fixed): ``current += step`` drifted (0.1 x 3 = 0.30000000000000004)
    so the documented inclusive endpoint was dropped from 0.1-step sweeps."""
    assert S.generate_floats(0.0, 0.3, 0.1) == [0.0, 0.1, 0.2, 0.3]
    assert S.generate_floats(0.0, 0.7, 0.1)[-1] == 0.7
    assert S.generate_floats(1.0, 2.0, 0.2) == [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    assert S.generate_floats(1.0, 0.0, 0.25) == []      # empty range


# ===========================================================================
# run_experiment
# ===========================================================================

def test_run_experiment_accepts_a_filtered_plate_map():
    """BUG (fixed): run_experiment addressed wells with ``.loc[i]`` for
    i in range(len(plate_map)), so any caller-filtered plate map (which is
    exactly what run_simulation hands it) raised KeyError: 0."""
    _seed(4)
    pm = S.generate_plate_map(1)
    filtered = pm[~pm["column_id"].isin(["1", "2", "3", "23", "24"])]
    assert filtered.index[0] == 3          # non-contiguous index — the trigger

    cell_df, _, _, _ = S.run_experiment(
        filtered, number_of_genes=20, active_gene_list=[1, 2, 3],
        avg_genes_per_well=5, sd_genes_per_well=2,
        avg_cells_per_well=10, sd_cells_per_well=3,
        well_ineq_coeff=1.2, gene_ineq_coeff=1.2)

    assert len(cell_df) > 0
    # Only wells that survived the filter may appear.
    assert set(cell_df["plate_row_column"]) <= set(filtered["plate_row_column"])
    assert set(cell_df["column_id"]).isdisjoint({"1", "2", "3", "23", "24"})


def test_run_experiment_stops_when_genes_outnumber_wells():
    """The gene→well loop breaks once gene-1 runs past the per-well draw array."""
    _seed(5)
    pm = _small_plate_map(6)
    cell_df, gpw_df, wpg_df, dists = S.run_experiment(
        pm, number_of_genes=50, active_gene_list=[1, 2],
        avg_genes_per_well=4, sd_genes_per_well=1,
        avg_cells_per_well=8, sd_cells_per_well=2,
        well_ineq_coeff=1.1, gene_ineq_coeff=1.1)
    # Only the first len(plate_map) genes can ever be placed.
    assert cell_df["gene_id"].max() <= len(pm)
    assert set(cell_df["gene_id"]) <= set(range(1, len(pm) + 1))
    # gene_weights still covers all 50 genes even though only 6 were placed.
    assert dists[4].shape == (50,)
    assert dists[5].shape == (len(pm),)


def test_run_experiment_caps_wells_per_gene_at_plate_size():
    """A gene asked for more wells than exist is capped to len(plate_map)-1.

    Without the cap ``np.random.choice(..., replace=False)`` raises
    "Cannot take a larger sample than population".
    """
    _seed(6)
    pm = _small_plate_map(6)
    cell_df, _, _, _ = S.run_experiment(
        pm, number_of_genes=4, active_gene_list=[1],
        avg_genes_per_well=200, sd_genes_per_well=20,   # gpw >> 6 wells
        avg_cells_per_well=5, sd_cells_per_well=1,
        well_ineq_coeff=1.1, gene_ineq_coeff=1.1)
    assert set(cell_df["plate_row_column"]) <= set(pm["plate_row_column"])
    # Capped at len-1, so at least one well must be missing for every gene.
    for gene, grp in cell_df.groupby("gene_id"):
        assert grp["plate_row_column"].nunique() <= len(pm) - 1


def test_run_experiment_labels_active_genes_and_reports_ginis():
    _seed(7)
    pm = S.generate_plate_map(1)
    active = [1, 2, 3, 4, 5]
    cell_df, gpw_df, wpg_df, dists = S.run_experiment(
        pm, number_of_genes=20, active_gene_list=active,
        avg_genes_per_well=5, sd_genes_per_well=2,
        avg_cells_per_well=10, sd_cells_per_well=3,
        well_ineq_coeff=1.2, gene_ineq_coeff=1.2)

    # is_active is exactly the membership indicator — no noise at this stage.
    expected = cell_df["gene_id"].isin(active).astype(int)
    assert (cell_df["is_active"] == expected).all()
    assert set(cell_df["is_active"]) <= {0, 1}

    # rank columns are 1..n and the counts are sorted ascending.
    assert gpw_df["rank"].tolist() == list(range(1, len(gpw_df) + 1))
    assert wpg_df["rank"].tolist() == list(range(1, len(wpg_df) + 1))
    assert gpw_df["genes_per_well"].is_monotonic_increasing
    assert wpg_df["wells_per_gene"].is_monotonic_increasing
    assert len(gpw_df) == cell_df["plate_row_column"].nunique()
    assert len(wpg_df) == cell_df["gene_id"].nunique()

    gene_counts, well_counts, gini_well, gini_gene, gene_w, well_w = dists
    assert len(gene_counts) == len(gpw_df)
    assert len(well_counts) == len(wpg_df)
    assert gini_well.shape == (cell_df["plate_row_column"].nunique(),)
    assert gini_gene.shape == (cell_df["gene_id"].nunique(),)
    assert ((gini_well >= 0) & (gini_well <= 1)).all()
    assert gene_w.sum() == pytest.approx(1.0)
    assert well_w.sum() == pytest.approx(1.0)


# ===========================================================================
# classifier / ROC / PR
# ===========================================================================

def test_classifier_flips_labels_at_the_stated_rate():
    """With accuracy p, exactly ~p of rows are drawn from their own Beta."""
    _seed(8)
    df = pd.DataFrame({"is_active": [1] * 5000 + [0] * 5000})
    out = S.classifier(0.9, 0.005, 0.1, 0.005, classifier_accuracy=0.75, df=df)
    actives = out.loc[out["is_active"] == 1, "score"]
    # Correctly-drawn actives sit near 0.9, flipped ones near 0.1.
    frac_high = (actives > 0.5).mean()
    assert frac_high == pytest.approx(0.75, abs=0.03)
    inactives = out.loc[out["is_active"] == 0, "score"]
    assert (inactives < 0.5).mean() == pytest.approx(0.75, abs=0.03)
    assert out["score"].between(0, 1).all()


def test_classifier_rejects_infeasible_variance_for_each_class():
    df = pd.DataFrame({"is_active": [1, 0]})
    with pytest.raises(ValueError, match="Variance must be positive"):
        S.classifier(0.5, 0.30, 0.2, 0.02, 0.9, df.copy())   # var >= 0.25
    with pytest.raises(ValueError, match="Variance must be positive"):
        S.classifier(0.5, 0.02, 0.2, 0.20, 0.9, df.copy())   # neg var >= 0.16
    with pytest.raises(ValueError, match="Mean must be between"):
        S.classifier(0.5, 0.02, 0.0, 0.02, 0.9, df.copy())   # negative_mean == 0


def test_compute_precision_recall_threshold_column_is_aligned():
    """BUG (fixed): thresholds were padded at the FRONT (np.insert(th, 0, 0)),
    which shifted every precision/recall row onto the threshold below it."""
    y = [1] * 5 + [0] * 5
    s = [0.9] * 5 + [0.1] * 5
    pr = S.compute_precision_recall(pd.DataFrame({"is_active": y, "score": s}))
    assert list(pr["threshold"]) == [0.1, 0.9, 1.0]
    assert list(pr["precision"]) == [0.5, 1.0, 1.0]
    assert list(pr["recall"]) == [1.0, 1.0, 0.0]
    # The F1-optimal row is (precision 1, recall 1) → its threshold is 0.9.
    assert S.get_optimum_threshold(pr) == pytest.approx(0.9)


def test_cell_level_roc_auc_on_perfectly_separable_scores():
    """BUG (fixed): the mis-aligned threshold made the optimum 0.1 instead of
    0.9, so every negative was predicted positive — cm was [[0,5],[0,5]]."""
    y = [1] * 5 + [0] * 5
    s = [0.9] * 5 + [0.1] * 5
    roc_df, pr_df, scored, cm = S.cell_level_roc_auc(
        pd.DataFrame({"is_active": y, "score": s}))
    assert np.array_equal(cm, np.array([[5, 0], [0, 5]]))
    assert roc_df["roc_auc"].iloc[0] == pytest.approx(1.0)
    assert pr_df["pr_auc"].iloc[0] == pytest.approx(1.0)
    assert pr_df["optimum"].iloc[0] == pytest.approx(0.9)
    # update_scores_and_get_cm names the new column after the threshold.
    assert 0.9 in scored.columns
    assert scored[0.9].tolist() == [1] * 5 + [0] * 5


def test_update_scores_and_get_cm_uses_a_greater_or_equal_test():
    cs = pd.DataFrame({"is_active": [1, 1, 0, 0], "score": [0.5, 0.4, 0.5, 0.1]})
    out, cm = S.update_scores_and_get_cm(cs, 0.5)
    assert out[0.5].tolist() == [1, 0, 1, 0]
    assert np.array_equal(cm, np.array([[1, 1], [1, 1]]))


def test_compute_roc_auc_returns_the_full_curve():
    y = [1] * 4 + [0] * 4
    s = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]
    roc = S.compute_roc_auc(pd.DataFrame({"is_active": y, "score": s}))
    assert roc["roc_auc"] == pytest.approx(1.0)
    assert roc["fpr"][0] == 0.0 and roc["fpr"][-1] == 1.0
    assert roc["tpr"][-1] == 1.0
    assert len(roc["threshold"]) == len(roc["tpr"]) == len(roc["fpr"])


# ===========================================================================
# generate_well_score / sequence_plates
# ===========================================================================

def _two_well_scores():
    return pd.DataFrame(
        {"average_active_score": [0.5, 0.0],
         "gene_list": [[1, 2, 3], []],
         "score": [np.log10(1.5), 0.0]},
        index=pd.Index(["1_1_4", "1_1_5"], name="plate_row_column"))


def test_generate_well_score_aggregates_exactly():
    cs = pd.DataFrame({
        "plate_row_column": ["w1"] * 4 + ["w2"] * 2,
        "is_active": [1, 1, 0, 0, 1, 1],
        "gene_id": [3, 3, 7, 1, 5, 5],
    })
    ws = S.generate_well_score(cs)
    assert list(ws.index) == ["w1", "w2"]
    assert ws.loc["w1", "average_active_score"] == pytest.approx(0.5)
    assert ws.loc["w1", "gene_list"] == [1, 3, 7]      # np.unique → sorted
    assert ws.loc["w2", "gene_list"] == [5]
    assert ws.loc["w1", "score"] == pytest.approx(np.log10(1.5))
    assert ws.loc["w2", "score"] == pytest.approx(np.log10(2.0))


def test_sequence_plates_without_error_keeps_reads_in_their_own_well():
    _seed(9)
    ws = _two_well_scores()
    frac, meta = S.sequence_plates(ws, number_of_genes=5,
                                   avg_reads_per_gene=1000, sd_reads_per_gene=50,
                                   sequencing_error=0.0)
    assert list(frac.columns) == [f"gene_{i}" for i in range(6)]
    assert list(frac.index) == ["1_1_4", "1_1_5"]
    # Well 1 carries genes 1,2,3 only; well 2 has an empty gene_list.
    assert (frac.loc["1_1_4", ["gene_1", "gene_2", "gene_3"]] > 0).all()
    assert frac.loc["1_1_4", ["gene_0", "gene_4", "gene_5"]].tolist() == [0, 0, 0]
    assert frac.loc["1_1_5"].tolist() == [0.0] * 6      # 0/0 → fillna(0)

    assert meta.loc["1_1_4", "genes_in_well"] == 3
    assert meta.loc["1_1_5", "genes_in_well"] == 0
    assert meta.loc["1_1_4", "sum_fractions"] == pytest.approx(1.0)
    assert meta.loc["1_1_5", "sum_fractions"] == 0.0
    assert meta.loc["1_1_4", "sum_reads"] > 0
    assert meta.loc["1_1_5", "sum_reads"] == 0


def test_sequence_plates_error_reads_are_counted_in_the_well_they_land_in():
    """BUG (fixed): sum_reads was appended inside the loop, so reads that a
    later well mis-assigned to an earlier/other well were never counted —
    a well could report genes_in_well > 0 with sum_reads == 0."""
    _seed(10)
    ws = _two_well_scores()          # only the first well has genes
    frac, meta = S.sequence_plates(ws, number_of_genes=5,
                                   avg_reads_per_gene=1000, sd_reads_per_gene=50,
                                   sequencing_error=1.0)   # every read misplaced
    # Every well holding a non-zero fraction must report the reads behind it.
    for well in frac.index:
        if meta.loc[well, "genes_in_well"] > 0:
            assert meta.loc[well, "sum_reads"] > 0
            assert meta.loc[well, "sum_fractions"] == pytest.approx(1.0)
    assert meta["sum_reads"].sum() > 0


def test_sequence_plates_single_well_error_lands_back_in_the_same_well():
    """With one well, the 'wrong well' is the same well → deterministic."""
    _seed(11)
    ws = pd.DataFrame({"gene_list": [[1, 2]]},
                      index=pd.Index(["only"], name="plate_row_column"))
    frac, meta = S.sequence_plates(ws, number_of_genes=3,
                                   avg_reads_per_gene=500, sd_reads_per_gene=25,
                                   sequencing_error=1.0)
    assert meta.loc["only", "genes_in_well"] == 2
    assert meta.loc["only", "sum_fractions"] == pytest.approx(1.0)
    assert frac.loc["only", "gene_0"] == 0.0
    assert frac.loc["only", "gene_3"] == 0.0
    assert frac.loc["only", ["gene_1", "gene_2"]].sum() == pytest.approx(1.0)


# ===========================================================================
# regression_roc_auc
# ===========================================================================

def _regression_results():
    """Hand-built regression output with a known right answer.

    controls are genes 7 & 8 with coef ±0.01 → var (ddof=1) = 2e-4, so the hit
    cutoff is |0| + 3 * 2e-4 = 6e-4.
    """
    return pd.DataFrame({
        "gene":   [f"gene_{i}" for i in range(10)],
        "coef":   [0.0001, 0.5, -0.6, 0.0001, 0.4,
                   0.0002, -0.0003, 0.01, -0.01, 0.0001],
        "std err": [0.01] * 10,
        "P>|t|":  [0.50, 0.0, 0.002, 0.90, 0.001,
                   0.40, 0.60, 0.30, 0.20, 0.02],
    })


def test_regression_roc_auc_exact_confusion_matrix_and_stats():
    res, roc_df, pr_df, cm, stats = S.regression_roc_auc(
        _regression_results(), active_gene_list=[1, 2, 3],
        control_gene_list=[7, 8], alpha=0.05, optimal=False)

    assert res["color"].tolist() == [
        "inactive", "active", "active", "active", "inactive",
        "inactive", "inactive", "control", "control", "inactive"]
    assert res["active"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    assert res["size"].tolist() == res["active"].tolist()

    # p is clipped at 1e-4 before the log, so p=0 → logp exactly 4.
    assert res.loc[1, "logp"] == pytest.approx(4.0)
    assert res.loc[0, "logp"] == pytest.approx(-np.log10(0.5))

    # hits = |coef| >= cutoff AND p <= alpha
    assert res["score"].tolist() == [0, 1, 1, 0, 1, 0, 0, 0, 0, 0]

    row = stats.iloc[0]
    assert row["cutoff"] == pytest.approx(3 * 2e-4)
    assert (row["TP"], row["FP"], row["TN"], row["FN"]) == (2, 1, 6, 1)
    assert np.array_equal(cm, np.array([[6, 1], [1, 2]]))
    assert row["accuracy"] == pytest.approx(8 / 10)
    assert row["roc_auc"] == pytest.approx(0.5 * (1 / 7) * (2 / 3)
                                           + (6 / 7) * (2 / 3 + 1) / 2)
    assert row["active_mean"] == pytest.approx((0.5 - 0.6 + 0.0001) / 3)
    assert row["inactive_mean"] == pytest.approx(
        np.mean([0.0001, 0.4, 0.0002, -0.0003, 0.0001]))
    assert row["active_var"] == pytest.approx(
        np.var([0.5, -0.6, 0.0001], ddof=1))
    assert row["inactive_std"] == pytest.approx(
        np.std([0.0001, 0.4, 0.0002, -0.0003, 0.0001], ddof=1))
    assert roc_df["roc_auc"].nunique() == 1
    assert 0.0 < pr_df["pr_auc"].iloc[0] <= 1.0
    # renamed column, and the 0.5-threshold prediction column exists
    assert "p" in res.columns and "P>|t|" not in res.columns
    assert res[0.5].tolist() == res["score"].tolist()


def test_regression_roc_auc_optimal_threshold_comes_from_the_pr_table():
    """BUG (fixed): optimal_threshold was ``f1_score.idxmax()`` — a row *index*,
    not a threshold — and was then compared against the score column."""
    res, roc_df, pr_df, cm, stats = S.regression_roc_auc(
        _regression_results(), active_gene_list=[1, 2, 3],
        control_gene_list=[7, 8], alpha=0.05, optimal=False)
    best_row = pr_df["f1_score"].idxmax()
    assert best_row == 1                                    # not the first row
    assert stats.iloc[0]["optimal_threshold"] == pytest.approx(
        pr_df.loc[best_row, "threshold"])
    assert stats.iloc[0]["optimal_threshold"] == pytest.approx(1.0)
    # threshold padding is at the tail, matching precision/recall row-for-row
    assert list(pr_df["threshold"]) == [0.0, 1.0, 1.0]


def test_regression_roc_auc_optimal_true_uses_that_threshold():
    res, _, pr_df, cm, stats = S.regression_roc_auc(
        _regression_results(), active_gene_list=[1, 2, 3],
        control_gene_list=[7, 8], alpha=0.05, optimal=True)
    thr = stats.iloc[0]["optimal_threshold"]
    assert thr in res.columns
    assert res[thr].tolist() == [1 if v >= thr else 0 for v in res["score"]]
    assert np.array_equal(cm, np.array([[6, 1], [1, 2]]))


def test_regression_roc_auc_alpha_gates_hits():
    """Tightening alpha below every p-value removes all hits."""
    res, _, _, cm, stats = S.regression_roc_auc(
        _regression_results(), active_gene_list=[1, 2, 3],
        control_gene_list=[7, 8], alpha=1e-6, optimal=False)
    assert res["score"].tolist() == [0] * 10
    assert (stats.iloc[0]["TP"], stats.iloc[0]["FP"]) == (0, 0)
    assert stats.iloc[0]["FN"] == 3


# ===========================================================================
# plotting primitives
# ===========================================================================

def _step_hist_edges(ax, which=0):
    """Bin edges of a seaborn ``element='step'`` histogram (a filled polygon)."""
    verts = ax.collections[which].get_paths()[0].vertices
    return np.unique(np.round(verts[:, 0], 6))


def test_plot_histogram_default_binwidth_and_log_axis():
    df = pd.DataFrame({"v": [0.1, 0.2, 0.2, 0.3, 0.9]})
    fig, ax = plt.subplots()
    S.plot_histogram(df, "v", ax, "teal", "the title", binwidth=None, log=True)
    assert ax.get_title() == "the title"
    assert ax.get_xlabel() == "v"
    assert ax.get_ylabel() == "Density"          # stat='density'
    assert ax.get_yscale() == "log"
    edges = _step_hist_edges(ax)
    assert len(edges) > 1
    assert edges[0] == pytest.approx(0.1)        # spans the data, seaborn bins
    assert edges[-1] == pytest.approx(0.9)


def test_plot_histogram_explicit_binwidth_controls_the_bins():
    df = pd.DataFrame({"v": np.linspace(0.0, 1.0, 101)})
    fig, ax = plt.subplots()
    S.plot_histogram(df, "v", ax, "slategray", "bins", binwidth=0.1, log=False)
    assert ax.get_yscale() == "linear"
    edges = _step_hist_edges(ax)
    assert edges.tolist() == pytest.approx(np.arange(0.0, 1.05, 0.1).tolist())
    # stat='density' → uniform data over a unit interval gives height ~1
    heights = ax.collections[0].get_paths()[0].vertices[:, 1]
    assert 0.8 < heights.max() < 1.3


def test_plot_roc_pr_draws_curve_and_random_reference():
    data = pd.DataFrame({"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.8, 1.0]})
    fig, ax = plt.subplots()
    S.plot_roc_pr(data, ax, "ROC", "fpr", "tpr")
    assert len(ax.lines) == 2
    assert list(ax.lines[0].get_xdata()) == [0.0, 0.5, 1.0]
    assert list(ax.lines[0].get_ydata()) == [0.0, 0.8, 1.0]
    assert list(ax.lines[1].get_xdata()) == [0, 1]     # diagonal reference
    assert ax.lines[1].get_linestyle() == "--"
    assert ax.get_title() == "ROC"
    assert ax.get_xlabel() == "fpr" and ax.get_ylabel() == "tpr"
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["random classifier"]


def test_plot_confusion_matrix_annotates_counts_and_percentages():
    cm = np.array([[6, 1], [1, 2]])
    fig, ax = plt.subplots()
    S.plot_confusion_matrix(cm, ax, "CM")
    texts = [t.get_text() for t in ax.texts]
    assert len(texts) == 4
    assert texts[0] == "True Neg\n6\n60.00%"
    assert texts[1] == "False Pos\n1\n10.00%"
    assert texts[2] == "False Neg\n1\n10.00%"
    assert texts[3] == "True Pos\n2\n20.00%"
    assert ax.get_title() == "CM"
    assert [t.get_text() for t in ax.xaxis.get_ticklabels()] == ["False", "True"]
    assert [t.get_text() for t in ax.yaxis.get_ticklabels()] == ["False", "True"]


# ===========================================================================
# run_simulation end-to-end
# ===========================================================================

def test_run_simulation_drops_the_control_columns():
    """BUG (fixed): the filter tested for 'c1'..'c24' but generate_plate_map
    writes bare numbers, so the outer control columns were never excluded."""
    _seed(12)
    output, dists = S.run_simulation(_settings())
    cell_scores = output[0]
    assert set(cell_scores["column_id"]).isdisjoint({"1", "2", "3", "23", "24"})
    assert set(cell_scores["column_id"]) <= {str(c) for c in range(4, 23)}
    # 19 usable columns x 16 rows on one plate
    assert cell_scores["plate_row_column"].nunique() <= 19 * 16


def test_run_simulation_returns_consistent_tables(sim_out):
    output, dists = sim_out
    (cell_scores, cell_roc, cell_pr, cell_cm, well_score, frac_map, metadata,
     results_df, reg_roc, reg_pr, reg_cm, sim_stats, gpw_df, wpg_df) = output

    assert len(output) == 14
    assert len(dists) == 6

    # cells -> wells -> sequencing all describe the same set of wells
    wells = set(cell_scores["plate_row_column"])
    assert set(well_score.index) == wells
    assert list(frac_map.index) == list(well_score.index)
    assert list(metadata.index) == list(well_score.index)

    # one regression row per gene (gene_0 .. gene_N), constant row dropped
    assert results_df["gene"].tolist() == [f"gene_{i}" for i in range(31)]
    assert len(results_df) == 31

    # confusion matrices account for every observation
    assert cell_cm.sum() == len(cell_scores)
    assert reg_cm.sum() == len(results_df)

    stats = sim_stats.iloc[0]
    assert stats["TP"] + stats["FP"] + stats["TN"] + stats["FN"] == len(results_df)
    assert stats["accuracy"] == pytest.approx(
        (stats["TP"] + stats["TN"]) / len(results_df))
    assert 0.0 <= stats["prauc"] <= 1.0
    assert 0.0 <= stats["roc_auc"] <= 1.0

    # gene fractions: every well either sums to 1 or is entirely empty
    sums = frac_map.sum(axis=1)
    assert set(np.round(sums, 9)) <= {0.0, 1.0}
    assert np.allclose(metadata["sum_fractions"].to_numpy(), sums.to_numpy())
    assert (metadata["genes_in_well"] >= 0).all()
    # a well with reads has genes, and vice versa
    assert ((metadata["sum_reads"] > 0) == (metadata["genes_in_well"] > 0)).all()

    # well score is log10(mean(is_active) + 1)
    assert well_score["score"].tolist() == pytest.approx(
        np.log10(well_score["average_active_score"] + 1).tolist())

    assert cell_roc["roc_auc"].nunique() == 1
    assert set(cell_pr.columns) == {"threshold", "precision", "recall",
                                    "f1_score", "pr_auc", "optimum"}


def test_run_simulation_high_accuracy_classifier_beats_a_bad_one():
    """The whole point of the simulator: better classifiers → better cell AUC."""
    _seed(13)
    good = S.run_simulation(_settings(classifier_accuracy=0.99))[0]
    _seed(13)
    bad = S.run_simulation(_settings(classifier_accuracy=0.55))[0]
    good_auc = good[1]["roc_auc"].iloc[0]
    bad_auc = bad[1]["roc_auc"].iloc[0]
    assert good_auc > 0.9, good_auc
    assert bad_auc < good_auc


# ===========================================================================
# figures over a full simulation
# ===========================================================================

def test_visualize_all_builds_the_thirteen_panel_figure(sim_out):
    output, _ = sim_out
    fig = S.visualize_all(output)
    # 13 panels, plus one seaborn colorbar per confusion-matrix heatmap
    assert len(fig.axes) == 15
    panels = fig.axes[:13]
    titles = [a.get_title() for a in panels]
    assert titles[0].startswith("gene/well (gini = ")
    assert titles[1].startswith("well/gene (Gini = ")
    assert titles[2] == "Cell scores"
    assert titles[3] == "Well scores"
    assert titles[4] == "ROC (Cell)"
    assert titles[5] == "Precision recall (Cell)"
    assert titles[6] == "Confusion Matrix Cell"
    assert titles[7] == "Well score"
    assert titles[8].startswith("Regression, threshold ")
    assert titles[9] == "Effect score error"
    assert titles[10] == "ROC (gene)"
    assert titles[11] == "Precision recall (gene)"
    assert titles[12] == "Confusion Matrix Reg"
    for a in panels:
        assert a.spines["top"].get_visible() is False
        assert a.spines["right"].get_visible() is False
    # the volcano panel carries one scatter per category
    labels = [c.get_label() for c in panels[8].collections]
    assert labels == ["inactive", "control", "active"]
    # both confusion-matrix panels are annotated with all four quadrants
    assert len(panels[6].texts) == 4
    assert len(panels[12].texts) == 4
    assert panels[6].texts[0].get_text().startswith("True Neg\n")


def test_vis_dists_writes_one_pdf_per_run(tmp_path, sim_out):
    _, dists = sim_out
    S.vis_dists(dists, str(tmp_path), "classifier_accuracy", 3)
    out = tmp_path / "dists" / "3_figure.pdf"
    assert out.is_file()
    assert out.stat().st_size > 0
    assert out.read_bytes()[:4] == b"%PDF"


def test_save_plot_writes_under_the_variable_folder(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    S.save_plot(fig, str(tmp_path), "well_ineq_coeff", 12)
    out = tmp_path / "well_ineq_coeff" / "12_figure.pdf"
    assert out.is_file()
    assert out.read_bytes()[:4] == b"%PDF"


# ===========================================================================
# persistence
# ===========================================================================

def test_create_database_creates_a_usable_sqlite_file(tmp_path):
    db = tmp_path / "made.db"
    S.create_database(str(db))
    assert db.is_file()
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (a INTEGER)")
    conn.close()


def test_create_database_reports_failure_without_raising(tmp_path, capsys):
    S.create_database(str(tmp_path / "no_such_dir" / "x.db"))
    assert "unable to open database file" in capsys.readouterr().out


def test_append_database_appends_rows(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    S.append_database(str(tmp_path), df, "simulations")
    S.append_database(str(tmp_path), df, "simulations")
    back = S.read_simulations_table(str(tmp_path / "simulations.db"))
    assert len(back) == 4
    assert back["a"].tolist() == [1, 2, 1, 2]
    assert back["b"].tolist() == ["x", "y", "x", "y"]


def test_append_database_unwritable_target_is_reported(tmp_path, capsys):
    """BUG (fixed): sqlite3.connect() raising left ``conn`` unbound, so the
    finally-block turned the handled OperationalError into UnboundLocalError."""
    missing = tmp_path / "does" / "not" / "exist"
    S.append_database(str(missing), pd.DataFrame({"a": [1]}), "simulations")
    assert "SQLite error:" in capsys.readouterr().out
    assert not missing.exists()


def test_read_simulations_table_missing_table_returns_none(tmp_path, capsys):
    db = tmp_path / "empty.db"
    sqlite3.connect(str(db)).close()
    assert S.read_simulations_table(str(db)) is None
    assert "An error occurred:" in capsys.readouterr().out


def test_save_data_summary_row_carries_settings_and_stats(tmp_path, sim_out):
    output, _ = sim_out
    settings = _settings(src=str(tmp_path))
    S.save_data(str(tmp_path), output, settings, save_all=False, i=4,
                variable="classifier_accuracy")

    back = S.read_simulations_table(str(tmp_path / "simulations.db"))
    assert len(back) == 1
    row = back.iloc[0]
    assert row["classifier_accuracy"] == settings["classifier_accuracy"]
    assert row["number_of_genes"] == settings["number_of_genes"]
    assert row["variable_classifier_accuracy_sim_nr"] == 4
    # sim_stats columns came through the concat
    assert row["prauc"] == pytest.approx(output[11].iloc[0]["prauc"])
    assert row["TP"] == output[11].iloc[0]["TP"]
    # gini columns recomputed from the per-well / per-gene tables
    assert row["genes_per_well_gini"] == pytest.approx(
        S.gini(output[12]["genes_per_well"].tolist()))
    assert row["wells_per_gene_gini"] == pytest.approx(
        S.gini(output[13]["wells_per_gene"].tolist()))
    assert isinstance(row["date"], str) and row["date"][:2] == "20"


def test_save_data_save_all_writes_every_table(tmp_path, sim_out):
    output, _ = sim_out
    S.save_data(str(tmp_path), output, _settings(src=str(tmp_path)),
                save_all=True, i=0, variable="v")

    conn = sqlite3.connect(str(tmp_path / "simulations.db"))
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    expected = {'settings', 'cell_scores', 'cell_roc', 'cell_precision_recall',
                'cell_confusion_matrix', 'well_score', 'gene_fraction_map',
                'metadata', 'regression_results', 'regression_roc',
                'regression_precision_recall', 'regression_confusion_matrix',
                'sim_stats', 'genes_per_well', 'wells_per_gene'}
    assert expected <= names

    n_cells = conn.execute("SELECT COUNT(*) FROM cell_scores").fetchone()[0]
    assert n_cells == len(output[0])
    n_stats = conn.execute("SELECT COUNT(*) FROM sim_stats").fetchone()[0]
    assert n_stats == 1
    # well_score.gene_list is stringified so sqlite can store it
    gl = conn.execute("SELECT gene_list FROM well_score LIMIT 1").fetchone()[0]
    assert isinstance(gl, str) and gl.startswith("[")
    conn.close()


def test_save_data_reports_a_bad_output_without_raising(tmp_path, capsys):
    """Truncated output → IndexError inside, printed, never propagated."""
    S.save_data(str(tmp_path), [pd.DataFrame({"a": [1]})], _settings(),
                save_all=False, i=0, variable="v")
    out = capsys.readouterr().out
    assert "An error occurred while saving data:" in out
    assert "Traceback" in out


# ===========================================================================
# run_and_save
# ===========================================================================

def test_run_and_save_writes_the_database_and_returns_timing(tmp_path):
    _seed(14)
    time_ls = []
    settings = _settings(src=str(tmp_path), plot=False, name="runA",
                         start_time="260102")
    i, sim_time, extra = S.run_and_save(0, settings, time_ls, total_sims=1)

    assert i == 0
    assert extra is None
    assert sim_time > 0 and time_ls == [sim_time]
    assert settings["sim_time"] == sim_time

    db = tmp_path / "260102" / "runA" / "simulations.db"
    assert db.is_file()
    back = S.read_simulations_table(str(db))
    assert len(back) == 1
    assert back.iloc[0]["variable_classifier_accuracy_sim_nr"] == 0
    assert back.iloc[0]["sim_time"] == pytest.approx(sim_time)


def test_run_and_save_with_plot_writes_both_figures(tmp_path):
    _seed(15)
    time_ls = []
    settings = _settings(src=str(tmp_path), plot=True, name="runB",
                         start_time="260103", variable="nr_plates")
    S.run_and_save(2, settings, time_ls, total_sims=1)

    root = tmp_path / "260103" / "runB"
    assert (root / "dists" / "2_figure.pdf").is_file()
    assert (root / "nr_plates" / "2_figure.pdf").is_file()
    assert (root / "simulations.db").is_file()
    assert len(time_ls) == 1


def test_run_and_save_honours_an_opt_in_random_seed(tmp_path, monkeypatch):
    """BUG (fixed): run_and_save overwrote settings['random_seed'] with False
    before testing it, so the documented setting could never take effect."""
    seeded = []
    monkeypatch.setattr(S.random, "seed", lambda v: seeded.append(v))
    settings = _settings(src=str(tmp_path), plot=False, name="runC",
                         start_time="260104", random_seed=True)
    S.run_and_save(0, settings, [], total_sims=1)
    assert seeded == [42]

    seeded.clear()
    settings2 = _settings(src=str(tmp_path), plot=False, name="runD",
                          start_time="260104", random_seed=False)
    S.run_and_save(0, settings2, [], total_sims=1)
    assert seeded == []


# ===========================================================================
# parameter sweeps
# ===========================================================================

def _sweep_settings(**overrides):
    base = dict(
        replicates=2,
        avg_genes_per_well=[4, 8],
        avg_cells_per_well=[20],
        classifier_accuracy=[0.8, 0.95],
        avg_reads_per_gene=[100],
        sequencing_error=[0.01],
        well_ineq_coeff=[1.2],
        gene_ineq_coeff=[1.2],
        nr_plates=[1],
        number_of_genes=[30],
        number_of_active_genes=[10],
        number_of_control_genes=5,
        max_workers=1,
        src="unset",
        plot=False,
        name="sweep",
        variable="classifier_accuracy",
    )
    base.update(overrides)
    return base


def test_generate_paramiters_expands_the_cartesian_product(capsys):
    _seed(16)
    sims = S.generate_paramiters(_sweep_settings())
    # 2 avg_genes x 1 cells x 2 accuracies x 1 positive_mean x ... x 2 replicates
    assert len(sims) == 2 * 2 * 2
    assert "Running 8 simulations." in capsys.readouterr().out

    # every swept key is now a scalar, and the derived keys follow the rules
    for s in sims:
        assert s["avg_genes_per_well"] in (4, 8)
        assert s["sd_genes_per_well"] == s["avg_genes_per_well"] / 2
        assert s["sd_cells_per_well"] == s["avg_cells_per_well"] / 2
        assert s["sd_reads_per_gene"] == s["avg_reads_per_gene"] / 2
        assert s["negative_mean"] == pytest.approx(1 - s["positive_mean"])
        assert s["positive_variance"] == pytest.approx((1 - s["positive_mean"]) / 2)
        assert s["negative_variance"] == s["positive_variance"]
        assert s["classifier_accuracy"] in (0.8, 0.95)

    combos = sorted((s["avg_genes_per_well"], s["classifier_accuracy"]) for s in sims)
    assert combos == [(4, 0.8), (4, 0.8), (4, 0.95), (4, 0.95),
                      (8, 0.8), (8, 0.8), (8, 0.95), (8, 0.95)]
    # deepcopy per run: mutating one must not touch its neighbours
    sims[0]["nr_plates"] = 99
    assert sims[1]["nr_plates"] == 1


def test_generate_paramiters_defaults_positive_mean_when_not_swept():
    _seed(17)
    sims = S.generate_paramiters(_sweep_settings())
    assert {s["positive_mean"] for s in sims} == {0.8}


def test_generate_paramiters_sweeps_an_explicit_positive_mean_list():
    """BUG (fixed): settings['positive_mean'] was overwritten with [0.8]
    unconditionally, so a caller-supplied sweep of that key was discarded."""
    _seed(18)
    sims = S.generate_paramiters(_sweep_settings(positive_mean=[0.7, 0.9]))
    assert {s["positive_mean"] for s in sims} == {0.7, 0.9}
    assert len(sims) == 2 * 2 * 2 * 2
    for s in sims:
        assert s["negative_mean"] == pytest.approx(1 - s["positive_mean"])


def test_generate_paramiters_caps_infeasible_beta_variance(capsys):
    """positive_mean 0.1 → variance (1-0.1)/2 = 0.45 > 0.1*0.9 → clamped."""
    _seed(19)
    sims = S.generate_paramiters(_sweep_settings(positive_mean=[0.1]))
    out = capsys.readouterr().out
    assert "changed positive variance" in out
    for s in sims:
        max_pos = s["positive_mean"] * (1 - s["positive_mean"])
        assert s["positive_variance"] == pytest.approx(max_pos * 0.99)
        # and the clamped params are actually usable by the classifier
        S.classifier(s["positive_mean"], s["positive_variance"],
                     s["negative_mean"], s["negative_variance"], 0.9,
                     pd.DataFrame({"is_active": [1, 0]}))


def test_validate_and_adjust_beta_params_leaves_feasible_values_alone(capsys):
    params = [{"positive_mean": 0.8, "negative_mean": 0.2,
               "positive_variance": 0.02, "negative_variance": 0.03}]
    out = S.validate_and_adjust_beta_params(params)
    assert out[0]["positive_variance"] == 0.02
    assert out[0]["negative_variance"] == 0.03
    assert capsys.readouterr().out == ""


# ===========================================================================
# run_multiple_simulations — the process pool is the only thing faked
# ===========================================================================

class _FakeAsyncResult:
    def __init__(self, func, arglist, raise_on_get=None):
        self.polls = 0
        self._raise = raise_on_get
        self._value = None
        if raise_on_get is None:
            self._value = [func(*a) for a in arglist]

    def ready(self):
        self.polls += 1
        return self.polls > 1          # force one pass through the progress loop

    def get(self):
        if self._raise is not None:
            raise self._raise
        return self._value


class _FakePool:
    instances = []

    def __init__(self, workers, raise_on_get=None):
        self.workers = workers
        self.arglist = None
        self.raise_on_get = raise_on_get
        _FakePool.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starmap_async(self, func, arglist):
        self.arglist = list(arglist)
        return _FakeAsyncResult(func, self.arglist, self.raise_on_get)


class _FakeManager:
    def __init__(self, backing=None):
        self._backing = backing

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def list(self):
        return [] if self._backing is None else self._backing


@pytest.fixture
def fake_pool(monkeypatch):
    _FakePool.instances = []
    monkeypatch.setattr(S, "Pool", _FakePool)
    monkeypatch.setattr(S, "Manager", _FakeManager)
    return _FakePool


def test_run_multiple_simulations_runs_every_generated_sim(tmp_path, fake_pool, capsys):
    _seed(20)
    settings = _sweep_settings(src=str(tmp_path), replicates=1,
                               avg_genes_per_well=[4], classifier_accuracy=[0.9])
    S.run_multiple_simulations(settings)

    pool = fake_pool.instances[-1]
    assert pool.workers == 1
    assert len(pool.arglist) == 1
    index, sim_settings, time_ls, total = pool.arglist[0]
    assert index == 0
    assert total == 1
    assert sim_settings["classifier_accuracy"] == 0.9
    # start_time was stamped on the parent settings and inherited by each run
    assert settings["start_time"] == sim_settings["start_time"]
    assert len(settings["start_time"]) == 6

    db = tmp_path / settings["start_time"] / "sweep" / "simulations.db"
    assert db.is_file()
    assert len(S.read_simulations_table(str(db))) == 1
    assert "Progress: " in capsys.readouterr().out


def test_run_multiple_simulations_defaults_workers_from_cpu_count(
        tmp_path, fake_pool, monkeypatch):
    """BUG (fixed): cpu_count()-4 is <= 0 on small machines → Pool(0)."""
    monkeypatch.setattr(S, "cpu_count", lambda: 2)
    _seed(21)
    settings = _sweep_settings(src=str(tmp_path), replicates=1, max_workers=0,
                               avg_genes_per_well=[4], classifier_accuracy=[0.9])
    S.run_multiple_simulations(settings)
    assert fake_pool.instances[-1].workers == 1

    monkeypatch.setattr(S, "cpu_count", lambda: 12)
    _seed(21)
    S.run_multiple_simulations(settings)
    assert fake_pool.instances[-1].workers == 8


def test_run_multiple_simulations_reports_a_worker_exception(
        tmp_path, monkeypatch, capsys):
    boom = RuntimeError("worker exploded")
    monkeypatch.setattr(S, "Manager", _FakeManager)
    monkeypatch.setattr(
        S, "Pool", lambda workers: _FakePool(workers, raise_on_get=boom))
    _seed(22)
    settings = _sweep_settings(src=str(tmp_path), replicates=1,
                               avg_genes_per_well=[4], classifier_accuracy=[0.9])
    S.run_multiple_simulations(settings)          # must not propagate
    out = capsys.readouterr().out
    assert "worker exploded" in out
    assert "Traceback" in out


class _FlakyList(list):
    """A manager-list stand-in whose first len() call fails."""

    def __init__(self):
        super().__init__()
        self.len_calls = 0

    def __len__(self):
        self.len_calls += 1
        if self.len_calls == 1:
            raise RuntimeError("manager link lost")
        return super().__len__()


def test_run_multiple_simulations_survives_a_progress_read_failure(
        tmp_path, monkeypatch, capsys):
    flaky = _FlakyList()
    monkeypatch.setattr(S, "Manager", lambda: _FakeManager(flaky))
    monkeypatch.setattr(S, "Pool", _FakePool)
    _FakePool.instances = []
    _seed(23)
    settings = _sweep_settings(src=str(tmp_path), replicates=1,
                               avg_genes_per_well=[4], classifier_accuracy=[0.9])
    S.run_multiple_simulations(settings)
    out = capsys.readouterr().out
    assert "manager link lost" in out
    assert flaky.len_calls >= 1
    # the simulation itself still completed and was written to disk
    db = tmp_path / settings["start_time"] / "sweep" / "simulations.db"
    assert db.is_file()


# ===========================================================================
# sweep-result analysis plots
# ===========================================================================

GROUPING_VARS = ['number_of_active_genes', 'number_of_control_genes',
                 'avg_reads_per_gene', 'classifier_accuracy', 'nr_plates',
                 'number_of_genes', 'avg_genes_per_well', 'avg_cells_per_well',
                 'sequencing_error', 'well_ineq_coeff', 'gene_ineq_coeff']


def _sweep_results(n_reps=3, accuracies=(0.8, 0.95)):
    """A tidy sweep table: nr_plates varies within each accuracy condition."""
    rows = []
    for acc in accuracies:
        for plates in (1, 2, 4):
            for rep in range(n_reps):
                rows.append({
                    'number_of_active_genes': 10,
                    'number_of_control_genes': 5,
                    'avg_reads_per_gene': 100,
                    'classifier_accuracy': acc,
                    'nr_plates': plates,
                    'number_of_genes': 30,
                    'avg_genes_per_well': 5,
                    'avg_cells_per_well': 20,
                    'sequencing_error': 0.01,
                    'well_ineq_coeff': 1.2,
                    'gene_ineq_coeff': 1.2,
                    'prauc': 0.1 * plates + acc + 0.01 * rep,
                })
    return pd.DataFrame(rows)


def test_plot_simulations_default_clean_groups_the_varying_conditions():
    """BUG (fixed): the clean branch referenced ``relevant_data`` before it was
    assigned, so plot_simulations raised UnboundLocalError on every call."""
    df = _sweep_results()
    fig = S.plot_simulations(df, "nr_plates")
    # classifier_accuracy is the only other varying grouping var → 2 panels
    visible = [a for a in fig.axes if a.get_visible()]
    assert len(visible) == 2
    for ax in visible:
        assert ax.get_xlabel() == "nr_plates"
        assert ax.get_ylabel() == "Precision-Recall AUC (PRAUC)"
        assert list(ax.lines[0].get_xdata()) == [1, 2, 4]
        assert ax.get_legend() is None            # legend=False by default

    means = sorted(tuple(np.round(a.lines[0].get_ydata(), 6)) for a in visible)
    expected = sorted(
        tuple(np.round(df[df.classifier_accuracy == acc]
                       .groupby("nr_plates")["prauc"].mean().values, 6))
        for acc in (0.8, 0.95))
    assert means == expected


def test_plot_simulations_without_clean_uses_every_grouping_var():
    df = _sweep_results()
    fig = S.plot_simulations(df, "nr_plates", clean=False, legend=True, grid=True)
    visible = [a for a in fig.axes if a.get_visible()]
    assert len(visible) == 2                       # constants add no new splits
    assert visible[0].get_legend() is not None


def test_plot_simulations_single_panel_when_nothing_else_varies():
    df = _sweep_results()
    df = df[df["classifier_accuracy"] == 0.8]
    fig = S.plot_simulations(df, "nr_plates")
    visible = [a for a in fig.axes if a.get_visible()]
    assert len(visible) == 1
    assert list(visible[0].lines[0].get_xdata()) == [1, 2, 4]
    assert list(np.round(visible[0].lines[0].get_ydata(), 6)) == list(
        np.round(df.groupby("nr_plates")["prauc"].mean().values, 6))


def test_plot_simulations_x_rotation_and_verbose_annotation():
    df = _sweep_results()
    fig = S.plot_simulations(df, "nr_plates", x_rotation=90, verbose=True)
    ax = [a for a in fig.axes if a.get_visible()][0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["1", "2", "4"]
    assert {t.get_rotation() for t in ax.get_xticklabels()} == {90.0}
    assert any("classifier_accuracy:" in t.get_text() for t in ax.texts)


def test_plot_simulations_hides_the_unused_grid_cells():
    """3 conditions land in a 2x2 grid → the 4th panel must be switched off."""
    df = _sweep_results(accuracies=(0.7, 0.85, 0.95))
    fig = S.plot_simulations(df, "nr_plates")
    assert len(fig.axes) == 4
    assert [a.get_visible() for a in fig.axes] == [True, True, True, False]


def test_plot_simulations_missing_columns_raises_value_error():
    df = _sweep_results().drop(columns=["sequencing_error", "prauc"])
    with pytest.raises(ValueError, match="DataFrame must contain"):
        S.plot_simulations(df, "nr_plates")


def _importance_df(n=60, seed=0):
    """prauc is driven almost entirely by classifier_accuracy."""
    rng = np.random.default_rng(seed)
    cols = {v: rng.uniform(1.0, 10.0, n) for v in GROUPING_VARS}
    df = pd.DataFrame(cols)
    df["prauc"] = 5.0 * df["classifier_accuracy"] + rng.normal(0, 0.001, n)
    return df


def test_plot_feature_importance_labels_the_bars_it_draws():
    """BUG (fixed): tick labels were built from ``indices[::-1]`` while the bars
    used ``indices``, so the tallest bar was labelled with the least important
    feature."""
    fig = S.plot_feature_importance(_importance_df(), target="prauc", clean=True)
    ax = fig.axes[0]
    widths = [p.get_width() for p in ax.patches]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert widths[0] == max(widths)
    assert widths[0] > 0.9                       # essentially all the signal
    assert labels[0] == "classifier_accuracy"
    assert sorted(labels) == sorted(GROUPING_VARS)
    assert sum(widths) == pytest.approx(1.0)


def test_plot_feature_importance_excludes_a_list_of_features():
    fig = S.plot_feature_importance(
        _importance_df(), target="prauc",
        exclude=["nr_plates", "sequencing_error"], clean=True)
    labels = {t.get_text() for t in fig.axes[0].get_yticklabels()}
    assert "nr_plates" not in labels and "sequencing_error" not in labels
    assert len(labels) == len(GROUPING_VARS) - 2
    assert "classifier_accuracy" in labels


def test_plot_feature_importance_excludes_a_single_feature():
    fig = S.plot_feature_importance(
        _importance_df(), target="prauc", exclude="classifier_accuracy")
    labels = {t.get_text() for t in fig.axes[0].get_yticklabels()}
    assert "classifier_accuracy" not in labels
    assert len(labels) == len(GROUPING_VARS) - 1


def test_calculate_permutation_importance_labels_come_from_the_features():
    """BUG (fixed): labels were read from ``df.columns`` while the sort index
    ran over ``features`` — they only lined up when the DataFrame happened to
    start with exactly those columns in that order."""
    df = _importance_df()
    # put the target first so df.columns and features no longer coincide
    df = df[["prauc"] + GROUPING_VARS]
    fig = S.calculate_permutation_importance(df, target="prauc", n_repeats=3)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    widths = [p.get_width() for p in ax.patches]
    assert sorted(labels) == sorted(GROUPING_VARS)
    assert "prauc" not in labels
    # bars are sorted ascending, so the dominant feature is last
    assert widths[-1] == max(widths)
    assert labels[-1] == "classifier_accuracy"


def test_calculate_permutation_importance_excludes_a_list(capsys):
    """BUG (fixed): a list exclude ran both branches — the second one called
    features.remove(<the list>) and raised ValueError."""
    fig = S.calculate_permutation_importance(
        _importance_df(), target="prauc", exclude=["nr_plates"], n_repeats=2)
    labels = {t.get_text() for t in fig.axes[0].get_yticklabels()}
    assert "nr_plates" not in labels
    assert len(labels) == len(GROUPING_VARS) - 1


def test_calculate_permutation_importance_unknown_exclude_is_ignored():
    """BUG (fixed): features.remove() raised ValueError for a name that clean
    had already dropped (or that was simply misspelled)."""
    fig = S.calculate_permutation_importance(
        _importance_df(), target="prauc", exclude="not_a_feature", n_repeats=2)
    labels = {t.get_text() for t in fig.axes[0].get_yticklabels()}
    assert len(labels) == len(GROUPING_VARS)


def test_plot_correlation_matrix_covers_the_varying_columns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fig = S.plot_correlation_matrix(_correlation_df(), annot=True, clean=True)
    ax = fig.axes[0]
    ticks = [t.get_text() for t in ax.get_xticklabels()]
    assert "classifier_accuracy" in ticks and "prauc" in ticks
    assert len(ticks) == 17
    assert (tmp_path / "figures" / "correlation_matrix" / "1_figure.pdf").is_file()


def _correlation_df():
    df = _importance_df()
    df["optimal_threshold"] = np.linspace(0.2, 0.8, len(df))
    df["accuracy"] = np.linspace(0.5, 1.0, len(df))
    df["roc_auc"] = np.linspace(0.6, 0.99, len(df))
    df["genes_per_well_gini"] = np.linspace(0.1, 0.9, len(df))
    df["wells_per_gene_gini"] = np.linspace(0.2, 0.8, len(df))
    return df


def test_plot_correlation_matrix_without_clean_keeps_constant_columns(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = _correlation_df()
    df["nr_plates"] = 3.0                      # constant → kept when clean=False
    fig = S.plot_correlation_matrix(df, annot=False, clean=False)
    ticks = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert "nr_plates" in ticks
    assert len(ticks) == 17


def test_plot_feature_importance_without_clean_keeps_constant_features(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = _importance_df()
    df["nr_plates"] = 3.0
    fig = S.plot_feature_importance(df, target="prauc", clean=False)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    widths = [p.get_width() for p in ax.patches]
    assert "nr_plates" in labels
    assert len(labels) == len(GROUPING_VARS)
    # a constant feature can never be split on → zero importance
    assert widths[labels.index("nr_plates")] == pytest.approx(0.0)
    assert labels[0] == "classifier_accuracy"


def test_calculate_permutation_importance_without_clean(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = _importance_df()
    df["nr_plates"] = 3.0
    fig = S.calculate_permutation_importance(
        df, target="prauc", n_repeats=2, clean=False)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert sorted(labels) == sorted(GROUPING_VARS)
    assert labels[-1] == "classifier_accuracy"


def test_plot_partial_dependences_without_clean(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fig = S.plot_partial_dependences(_importance_df(), target="prauc", clean=False)
    titles = [a.get_title() for a in fig.axes if a.get_title()]
    assert titles == GROUPING_VARS


def test_generate_shap_summary_plot_without_clean(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fig = S.generate_shap_summary_plot(_importance_df(n=30), target="prauc",
                                       clean=False)
    assert fig is not None
    assert (tmp_path / "figures" / "shap" / "1_figure.pdf").is_file()


def test_plot_partial_dependences_with_a_single_row_of_axes(tmp_path, monkeypatch):
    """BUG (fixed): with <= 4 features the code wrapped the whole axes array in
    a list, handing all four axes to the first feature — sklearn rejected it
    with "Expected ax to have 1 axes, got 4"."""
    monkeypatch.chdir(tmp_path)
    df = _importance_df()
    # features keep GROUPING_VARS order after `clean` filters them
    keep = ['avg_reads_per_gene', 'classifier_accuracy', 'nr_plates']
    # make every other grouping var constant so `clean` drops it
    for col in GROUPING_VARS:
        if col not in keep:
            df[col] = 1.0
    fig = S.plot_partial_dependences(df, target="prauc", clean=True)
    visible = [a for a in fig.axes if a.get_visible()]
    assert [a.get_title() for a in visible if a.get_title()] == keep
    assert len([a for a in fig.axes if not a.get_visible()]) == 1  # 4th cell hidden
    # each feature got its own partial-dependence sub-axes
    pdp = [a for a in visible if a.get_ylabel() == "Partial dependence"]
    assert sorted(a.get_xlabel() for a in pdp) == sorted(keep)
    assert (tmp_path / "figures" / "partial_dependences" / "1_figure.pdf").is_file()


def test_plot_partial_dependences_multi_row_grid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fig = S.plot_partial_dependences(_importance_df(), target="prauc", clean=True)
    titles = [a.get_title() for a in fig.axes if a.get_visible() and a.get_title()]
    assert set(titles) == set(GROUPING_VARS)
    hidden = [a for a in fig.axes if not a.get_visible()]
    assert len(hidden) == 1          # 11 features in a 3x4 grid


def test_generate_shap_summary_plot_writes_a_figure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fig = S.generate_shap_summary_plot(_importance_df(n=40), target="prauc")
    assert fig is not None
    out = tmp_path / "figures" / "shap" / "1_figure.pdf"
    assert out.is_file()
    assert out.read_bytes()[:4] == b"%PDF"


def test_save_shap_plot_writes_and_announces(tmp_path, capsys):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    S.save_shap_plot(fig, str(tmp_path), "shap", 7)
    out = tmp_path / "shap" / "7_figure.pdf"
    assert out.is_file()
    assert f"Saved figure as {out}" in capsys.readouterr().out


# ===========================================================================
# small pure helpers
# ===========================================================================

def test_generate_gene_list_draws_without_replacement():
    _seed(24)
    genes = S.generate_gene_list(30, 100)
    assert len(genes) == 30 == len(set(genes))
    assert all(0 <= g < 100 for g in genes)
    # asking for the whole pool returns a permutation of it
    assert sorted(S.generate_gene_list(100, 100)) == list(range(100))


def test_generate_plate_map_is_row_major_over_384_wells():
    pm = S.generate_plate_map(2)
    assert len(pm) == 2 * 384
    assert pm.iloc[0]["plate_row_column"] == "1_1_1"
    assert pm.iloc[23]["plate_row_column"] == "1_1_24"
    assert pm.iloc[24]["plate_row_column"] == "1_2_1"
    assert pm.iloc[383]["plate_row_column"] == "1_16_24"
    assert pm.iloc[384]["plate_row_column"] == "2_1_1"
    assert pm["plate_row_column"].is_unique


def test_normalize_array_endpoints_and_linearity():
    a = np.array([-4.0, 0.0, 4.0, 12.0])
    out = S.normalize_array(a)
    assert out.tolist() == [0.0, 0.25, 0.5, 1.0]


def test_generate_integers_inclusive_upper_bound():
    assert S.generate_integers(2, 8, 3) == [2, 5, 8]
    assert S.generate_integers(0, 0, 1) == [0]


def test_remove_columns_with_single_value_and_remove_constant_columns():
    df = pd.DataFrame({"const": [1, 1, 1], "vary": [1, 2, 3], "text": list("aaa")})
    out = S.remove_columns_with_single_value(df)
    assert list(out.columns) == ["vary"]
    assert out["vary"].tolist() == [1, 2, 3]
    out2 = S.remove_constant_columns(df)
    assert list(out2.columns) == ["vary"]
