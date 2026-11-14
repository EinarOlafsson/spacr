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
# classifier / ROC / PR
# ===========================================================================


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


def test_compute_precision_recall_defines_zero_over_zero_f1_as_zero():
    frame = pd.DataFrame({"is_active": [0, 1], "score": [1.0, 0.0]})
    with np.errstate(all="raise"):
        result = S.compute_precision_recall(frame)
    assert np.isfinite(result["f1_score"]).all()
    assert result["f1_score"][1] == 0.0


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
# generate_well_score
# ===========================================================================


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
# parameter sweeps
# ===========================================================================


def test_validate_and_adjust_beta_params_leaves_feasible_values_alone(capsys):
    params = [{"positive_mean": 0.8, "negative_mean": 0.2,
               "positive_variance": 0.02, "negative_variance": 0.03}]
    out = S.validate_and_adjust_beta_params(params)
    assert out[0]["positive_variance"] == 0.02
    assert out[0]["negative_variance"] == 0.03
    assert capsys.readouterr().out == ""


# ===========================================================================
# sweep-result analysis plots
# ===========================================================================

GROUPING_VARS = ['number_of_active_genes', 'number_of_control_genes',
                 'avg_reads_per_gene', 'classifier_accuracy', 'nr_plates',
                 'number_of_genes', 'avg_genes_per_well', 'avg_cells_per_well',
                 'sequencing_error', 'well_ineq_coeff', 'gene_ineq_coeff']


def _importance_df(n=60, seed=0):
    """prauc is driven almost entirely by classifier_accuracy."""
    rng = np.random.default_rng(seed)
    cols = {v: rng.uniform(1.0, 10.0, n) for v in GROUPING_VARS}
    df = pd.DataFrame(cols)
    df["prauc"] = 5.0 * df["classifier_accuracy"] + rng.normal(0, 0.001, n)
    return df


def test_generate_shap_summary_plot_without_clean(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fig = S.generate_shap_summary_plot(_importance_df(n=30), target="prauc",
                                       clean=False)
    assert fig is not None
    assert (tmp_path / "figures" / "shap" / "1_figure.pdf").is_file()


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
