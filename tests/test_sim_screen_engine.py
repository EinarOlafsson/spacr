"""Behaviour of the pooled-screen simulator engine in :mod:`spacr.sim`.

Covers one unit per test: the cell-level experiment (``run_experiment``), the
noisy classifier, plate sequencing, regression scoring against ground truth,
sweep expansion (``generate_parameters``), the single-run worker
(``run_and_save``) and the pooled sweep driver (``run_multiple_simulations``).

Everything in ``sim.py`` draws from the global ``random`` and ``numpy.random``
generators, so each case seeds both and asserts on values that are invariant
given the seed: exact shapes and memberships, conservation laws, and exact
confusion matrices and statistics for hand-built inputs.
"""
from __future__ import annotations

import random
import re

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spacr.database_concurrency as database_concurrency
from spacr import sim as S


@pytest.fixture(autouse=True)
def _in_tmp_and_close_figures(tmp_path, monkeypatch):
    """Run from a temp folder (sim writes relative ``figures``) and close figures."""
    monkeypatch.chdir(tmp_path)
    yield
    plt.close("all")


def _seed(n: int = 0) -> None:
    """Seed both global RNGs that sim.py draws from."""
    random.seed(n)
    np.random.seed(n)


def _settings(**overrides):
    """A small but complete settings dict for run_simulation / run_and_save."""
    base = dict(
        name="unit", variable="classifier_accuracy", src="unset", plot=False,
        start_time="260101", number_of_genes=30, number_of_active_genes=10,
        number_of_control_genes=5, nr_plates=1, avg_genes_per_well=5,
        sd_genes_per_well=2, avg_cells_per_well=10, sd_cells_per_well=3,
        well_ineq_coeff=1.2, gene_ineq_coeff=1.2, positive_mean=0.8,
        positive_variance=0.02, negative_mean=0.2, negative_variance=0.02,
        classifier_accuracy=0.9, avg_reads_per_gene=100, sd_reads_per_gene=20,
        sequencing_error=0.01)
    base.update(overrides)
    return base


def _sweep_settings(**overrides):
    """A sweep-settings dict as run_multiple_simulations receives it."""
    base = dict(
        replicates=2, avg_genes_per_well=[4, 8], avg_cells_per_well=[20],
        classifier_accuracy=[0.8, 0.95], avg_reads_per_gene=[100],
        sequencing_error=[0.01], well_ineq_coeff=[1.2], gene_ineq_coeff=[1.2],
        nr_plates=[1], number_of_genes=[30], number_of_active_genes=[10],
        number_of_control_genes=5, max_workers=1, src="unset", plot=False,
        name="sweep", variable="classifier_accuracy")
    base.update(overrides)
    return base


def _small_plate_map(n_wells=6):
    """A hand-built plate map with the four columns run_experiment reads."""
    rows = [f"1_1_{c}" for c in range(1, n_wells + 1)]
    df = pd.DataFrame({"plate_row_column": rows})
    df["plate_id"], df["row_id"], df["column_id"] = zip(*[r.split("_") for r in rows])
    return df


def _filtered_plate_map():
    """One 384-well plate with columns 1-3 and 23-24 removed (index starts at 3)."""
    pm = S.generate_plate_map(1)
    return pm[~pm["column_id"].isin(["1", "2", "3", "23", "24"])]


CELL_COLUMNS = {"plate_row_column", "plate_id", "row_id", "column_id",
                "genes_in_well", "gene_id", "is_active"}


def test_run_experiment_places_cells_by_gene_and_reports_distributions(subtests):
    """run_experiment: cells only in plate-map wells, exact active labels, ranked counts, capped draws."""
    cases = [
        # id, seed, plate map, genes, actives, avg genes/well, sd, cells, sd, coeff
        ("full_plate", 7, S.generate_plate_map(1), 20, [1, 2, 3, 4, 5], 5, 2, 10, 3, 1.2),
        # BUG (fixed): wells were addressed with .loc[i], so a filtered map raised KeyError: 0.
        ("filtered_plate_map", 4, _filtered_plate_map(), 20, [1, 2, 3], 5, 2, 10, 3, 1.2),
        # the gene loop stops once genes outnumber the per-well draw array
        ("genes_outnumber_wells", 5, _small_plate_map(6), 50, [1, 2], 4, 1, 8, 2, 1.1),
        # more wells per gene than exist is capped to len(plate_map) - 1
        ("wells_per_gene_capped", 6, _small_plate_map(6), 4, [1], 200, 20, 5, 1, 1.1),
    ]
    for (case_id, seed, pm, n_genes, active, gpw, gpw_sd, cpw, cpw_sd, coeff) in cases:
        with subtests.test(msg=case_id):
            _seed(seed)
            cell_df, gpw_df, wpg_df, dists = S.run_experiment(
                pm, number_of_genes=n_genes, active_gene_list=active,
                avg_genes_per_well=gpw, sd_genes_per_well=gpw_sd,
                avg_cells_per_well=cpw, sd_cells_per_well=cpw_sd,
                well_ineq_coeff=coeff, gene_ineq_coeff=coeff)
            assert set(cell_df.columns) == CELL_COLUMNS
            assert len(cell_df) > 0
            assert set(cell_df["plate_row_column"]) <= set(pm["plate_row_column"])
            assert (cell_df["is_active"] == cell_df["gene_id"].isin(active).astype(int)).all()
            assert set(cell_df["is_active"]) <= {0, 1}

            n_wells = cell_df["plate_row_column"].nunique()
            n_placed = cell_df["gene_id"].nunique()
            assert gpw_df["rank"].tolist() == list(range(1, n_wells + 1))
            assert wpg_df["rank"].tolist() == list(range(1, n_placed + 1))
            assert gpw_df["genes_per_well"].is_monotonic_increasing
            assert wpg_df["wells_per_gene"].is_monotonic_increasing
            assert gpw_df["genes_per_well"].tolist() == sorted(
                cell_df.groupby("plate_row_column")["gene_id"].nunique())
            assert wpg_df["wells_per_gene"].tolist() == sorted(
                cell_df.groupby("gene_id")["plate_row_column"].nunique())

            gene_counts, well_counts, gini_well, gini_gene, gene_w, well_w = dists
            assert gene_counts == gpw_df["genes_per_well"].tolist()
            assert well_counts == wpg_df["wells_per_gene"].tolist()
            assert gini_well.shape == (n_wells,)
            assert gini_gene.shape == (n_placed,)
            assert ((gini_well >= 0) & (gini_well <= 1)).all()
            assert gene_w.shape == (n_genes,)
            assert well_w.shape == (len(pm),)
            assert gene_w.sum() == pytest.approx(1.0)
            assert well_w.sum() == pytest.approx(1.0)

            if case_id == "filtered_plate_map":
                assert pm.index[0] == 3
                assert set(cell_df["column_id"]).isdisjoint({"1", "2", "3", "23", "24"})
            if case_id == "genes_outnumber_wells":
                assert set(cell_df["gene_id"]) <= set(range(1, len(pm) + 1))
            if case_id == "wells_per_gene_capped":
                per_gene = cell_df.groupby("gene_id")["plate_row_column"].nunique()
                assert (per_gene <= len(pm) - 1).all()


def test_classifier_draws_scores_from_the_label_or_flipped_beta(subtests):
    """classifier: accuracy sets the share drawn from the own-class Beta; infeasible Betas are refused."""
    labels = pd.DataFrame({"is_active": [1] * 5000 + [0] * 5000})
    # id, accuracy, expected share of actives > 0.5 (== share of inactives < 0.5), tolerance
    rates = [("always_right", 1.0, 1.0, 0.0), ("always_flipped", 0.0, 0.0, 0.0),
             ("three_quarters", 0.75, 0.75, 0.03)]
    for case_id, accuracy, share, tol in rates:
        with subtests.test(msg=case_id):
            _seed(8)
            out = S.classifier(0.9, 0.001, 0.1, 0.001, classifier_accuracy=accuracy,
                               df=labels.copy())
            assert list(out.columns) == ["is_active", "score"]
            assert out["score"].between(0, 1).all()
            actives = out.loc[out["is_active"] == 1, "score"]
            inactives = out.loc[out["is_active"] == 0, "score"]
            assert (actives > 0.5).mean() == pytest.approx(share, abs=tol)
            assert (inactives < 0.5).mean() == pytest.approx(share, abs=tol)

    with subtests.test(msg="input_frame_gains_the_score_column"):
        _seed(1)
        df = pd.DataFrame({"is_active": [1, 0, 1, 0, 1, 0]})
        out = S.classifier(0.8, 0.02, 0.2, 0.02, 0.9, df)
        assert out is df and "score" in df.columns

    variance = "Variance must be positive and less than "
    mean = "Mean must be between 0 and 1 exclusively."
    refusals = [
        # id, pos mean, pos var, neg mean, neg var, message
        ("positive_variance_above_max", 0.5, 0.30, 0.2, 0.02, variance + "0.25"),
        ("positive_variance_at_max", 0.5, 0.25, 0.2, 0.02, variance + "0.25"),
        ("positive_variance_huge", 0.5, 1.0, 0.2, 0.02, variance + "0.25"),
        ("positive_variance_zero", 0.5, 0.0, 0.2, 0.02, variance + "0.25"),
        ("negative_variance_above_max", 0.5, 0.02, 0.2, 0.20, variance + str(0.2 * 0.8)),
        ("negative_mean_zero", 0.5, 0.02, 0.0, 0.02, mean),
        ("positive_mean_one", 1.0, 0.02, 0.2, 0.02, mean),
        ("positive_mean_above_one", 1.5, 0.02, 0.2, 0.02, mean),
    ]
    for case_id, pm_, pv, nm, nv, message in refusals:
        with subtests.test(msg=case_id):
            with pytest.raises(ValueError, match=re.escape(message)):
                S.classifier(pm_, pv, nm, nv, 0.9, pd.DataFrame({"is_active": [1, 0]}))


def _two_well_scores():
    """Two wells: the first carries genes 1-3, the second none."""
    return pd.DataFrame(
        {"average_active_score": [0.5, 0.0], "gene_list": [[1, 2, 3], []],
         "score": [np.log10(1.5), 0.0]},
        index=pd.Index(["1_1_4", "1_1_5"], name="plate_row_column"))


def _chain_well_scores():
    """Well scores from a full run_experiment -> classifier -> generate_well_score chain."""
    cell_df, _, _, _ = S.run_experiment(
        S.generate_plate_map(1), number_of_genes=20, active_gene_list=[1, 2, 3, 4, 5],
        avg_genes_per_well=5, sd_genes_per_well=2, avg_cells_per_well=10,
        sd_cells_per_well=3, well_ineq_coeff=1.2, gene_ineq_coeff=1.2)
    scored = S.classifier(0.8, 0.02, 0.2, 0.02, 0.85, cell_df)
    return S.generate_well_score(scored)


def test_sequence_plates_conserves_reads_into_per_well_fractions(subtests):
    """sequence_plates: fractions sum to 1 in every well with reads, metadata counts what landed there."""
    single = pd.DataFrame({"gene_list": [[1, 2]]},
                          index=pd.Index(["only"], name="plate_row_column"))
    cases = [
        # id, seed, well scores (callable), genes, avg reads, sd, error
        ("no_error_keeps_reads_home", 9, _two_well_scores, 5, 1000, 50, 0.0),
        # BUG (fixed): sum_reads ignored reads a later well mis-assigned to another well.
        ("every_read_misplaced", 10, _two_well_scores, 5, 1000, 50, 1.0),
        # with one well, the 'wrong well' is the same well
        ("single_well_error_returns_home", 11, lambda: single, 3, 500, 25, 1.0),
        ("experiment_chain", 3, _chain_well_scores, 20, 100, 20, 0.05),
    ]
    for case_id, seed, make, n_genes, avg, sd, error in cases:
        with subtests.test(msg=case_id):
            _seed(seed)
            ws = make()
            frac, meta = S.sequence_plates(ws, number_of_genes=n_genes,
                                           avg_reads_per_gene=avg, sd_reads_per_gene=sd,
                                           sequencing_error=error)
            assert list(frac.columns) == [f"gene_{i}" for i in range(n_genes + 1)]
            assert list(frac.index) == list(ws.index) == list(meta.index)
            assert list(meta.columns) == ["genes_in_well", "sum_fractions", "sum_reads"]
            assert meta["genes_in_well"].tolist() == (frac > 0).sum(axis=1).tolist()
            with_reads = meta["sum_reads"] > 0
            assert np.allclose(meta.loc[with_reads, "sum_fractions"], 1.0)
            assert (meta.loc[~with_reads, "sum_fractions"] == 0.0).all()
            assert (meta.loc[meta["genes_in_well"] > 0, "sum_reads"] > 0).all()
            assert meta["sum_reads"].sum() > 0
            if error == 0.0:
                # every read stays in its own well, on exactly the listed genes
                for well, genes in ws["gene_list"].items():
                    on = [f"gene_{g}" for g in genes]
                    assert (frac.loc[well, on] > 0).all()
                    assert frac.loc[well].drop(on).tolist() == [0.0] * (n_genes + 1 - len(on))
                assert meta["genes_in_well"].tolist() == [3, 0]
                assert meta["sum_reads"].tolist()[1] == 0
            if case_id == "single_well_error_returns_home":
                assert meta.loc["only", "genes_in_well"] == 2
                assert frac.loc["only", ["gene_0", "gene_3"]].tolist() == [0.0, 0.0]
                assert frac.loc["only", ["gene_1", "gene_2"]].sum() == pytest.approx(1.0)


def _regression_results():
    """Regression output with a known answer: controls 7, 8 at +-0.01 give cutoff 6e-4."""
    return pd.DataFrame({
        "gene": [f"gene_{i}" for i in range(10)],
        "coef": [0.0001, 0.5, -0.6, 0.0001, 0.4, 0.0002, -0.0003, 0.01, -0.01, 0.0001],
        "std err": [0.01] * 10,
        "P>|t|": [0.50, 0.0, 0.002, 0.90, 0.001, 0.40, 0.60, 0.30, 0.20, 0.02],
    })


INACTIVE_COEFS = [0.0001, 0.4, 0.0002, -0.0003, 0.0001]
ACTIVE_COEFS = [0.5, -0.6, 0.0001]


def test_regression_roc_auc_scores_hits_against_the_truth(subtests):
    """regression_roc_auc: exact hit calls, confusion matrix, thresholds and group statistics."""
    gated_auc = 0.5 * (1 / 7) * (2 / 3) + (6 / 7) * (2 / 3 + 1) / 2
    cases = [
        # id, alpha, optimal, hit scores, cm, (TP, FP, TN, FN), roc_auc, pr_auc,
        # PR-table thresholds, F1-optimal threshold
        ("default_half_threshold", 0.05, False, [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
         [[6, 1], [1, 2]], (2, 1, 6, 1), gated_auc, 43 / 60, [0.0, 1.0, 1.0], 1.0),
        ("optimal_threshold_column", 0.05, True, [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
         [[6, 1], [1, 2]], (2, 1, 6, 1), gated_auc, 43 / 60, [0.0, 1.0, 1.0], 1.0),
        ("alpha_below_every_p", 1e-6, False, [0] * 10,
         [[7, 0], [3, 0]], (0, 0, 7, 3), 0.5, 0.65, [0.0, 1.0], 0.0),
    ]
    for (case_id, alpha, optimal, hits, cm_expected, counts, roc_auc, pr_auc,
         thresholds, thr) in cases:
        with subtests.test(msg=case_id):
            res, roc_df, pr_df, cm, stats = S.regression_roc_auc(
                _regression_results(), active_gene_list=[1, 2, 3],
                control_gene_list=[7, 8], alpha=alpha, optimal=optimal)
            assert res["color"].tolist() == [
                "inactive", "active", "active", "active", "inactive",
                "inactive", "inactive", "control", "control", "inactive"]
            assert res["active"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
            assert res["size"].tolist() == res["active"].tolist()
            # p is clipped at 1e-4 before the log, so p = 0 gives logp exactly 4.
            assert res.loc[1, "logp"] == pytest.approx(4.0)
            assert res.loc[0, "logp"] == pytest.approx(-np.log10(0.5))
            prediction = thr if optimal else 0.5
            assert list(res.columns) == ["gene", "coef", "std err", "p", "active", "color",
                                         "size", "logp", "score", prediction]
            assert res["score"].tolist() == hits
            assert cm.tolist() == cm_expected

            row = stats.iloc[0]
            assert (row["TP"], row["FP"], row["TN"], row["FN"]) == counts
            assert row["accuracy"] == pytest.approx((counts[0] + counts[2]) / 10)
            assert row["cutoff"] == pytest.approx(3 * 2e-4)
            assert row["roc_auc"] == pytest.approx(roc_auc)
            assert roc_df["roc_auc"].tolist() == [pytest.approx(roc_auc)] * len(roc_df)
            assert row["active_mean"] == pytest.approx(np.mean(ACTIVE_COEFS))
            assert row["active_var"] == pytest.approx(np.var(ACTIVE_COEFS, ddof=1))
            assert row["active_std"] == pytest.approx(np.std(ACTIVE_COEFS, ddof=1))
            assert row["inactive_mean"] == pytest.approx(np.mean(INACTIVE_COEFS))
            assert row["inactive_var"] == pytest.approx(np.var(INACTIVE_COEFS, ddof=1))
            assert row["inactive_std"] == pytest.approx(np.std(INACTIVE_COEFS, ddof=1))
            # BUG (fixed): optimal_threshold was the F1 row index, not its threshold.
            best = pr_df["f1_score"].idxmax()
            assert row["optimal_threshold"] == pytest.approx(pr_df.loc[best, "threshold"])
            assert row["optimal_threshold"] == thr
            assert pr_df["threshold"].tolist() == thresholds
            assert row["prauc"] == pytest.approx(pr_auc)
            assert pr_df["pr_auc"].tolist() == [pytest.approx(pr_auc)] * len(thresholds)
            assert res[prediction].tolist() == [int(v >= prediction) for v in res["score"]]


def test_generate_parameters_expands_the_sweep_into_feasible_runs(subtests, capsys):
    """generate_parameters: one deep-copied run per combination and replicate, derived keys, capped variances."""
    assert S.generate_paramiters is S.generate_parameters
    cases = [
        # id, overrides, expected run count, expected positive means
        ("default_positive_mean", {}, 8, {0.8}),
        # a scalar positive_mean is not a sweep and falls back to [0.8]
        ("scalar_positive_mean_is_reset", {"positive_mean": 0.6}, 8, {0.8}),
        # BUG (fixed): a caller-supplied positive_mean sweep was overwritten with [0.8].
        ("explicit_positive_mean_sweep", {"positive_mean": [0.7, 0.9]}, 16, {0.7, 0.9}),
        # mean 0.1 -> variance 0.45 exceeds 0.1 * 0.9 for both classes -> clamped to 99 %
        ("infeasible_beta_variance_capped", {"positive_mean": [0.1]}, 8, {0.1}),
    ]
    for case_id, overrides, count, means in cases:
        with subtests.test(msg=case_id):
            capsys.readouterr()
            _seed(16)
            sims = S.generate_parameters(_sweep_settings(**overrides))
            out = capsys.readouterr().out
            assert len(sims) == count
            assert out.endswith(f"Running {count} simulations.\n")
            assert {s["positive_mean"] for s in sims} == means
            combos = sorted((s["avg_genes_per_well"], s["classifier_accuracy"]) for s in sims)
            per = count // 4
            assert combos == sorted([(4, 0.8), (4, 0.95), (8, 0.8), (8, 0.95)] * per)
            for s in sims:
                assert s["sd_genes_per_well"] == s["avg_genes_per_well"] / 2
                assert s["sd_cells_per_well"] == s["avg_cells_per_well"] / 2
                assert s["sd_reads_per_gene"] == s["avg_reads_per_gene"] / 2
                assert s["negative_mean"] == pytest.approx(1 - s["positive_mean"])
                assert (s["avg_cells_per_well"], s["avg_reads_per_gene"], s["nr_plates"],
                        s["number_of_genes"], s["number_of_active_genes"]) == (20, 100, 1, 30, 10)
                pos_max = s["positive_mean"] * (1 - s["positive_mean"])
                neg_max = s["negative_mean"] * (1 - s["negative_mean"])
                if case_id == "infeasible_beta_variance_capped":
                    assert s["positive_variance"] == pytest.approx(pos_max * 0.99)
                    assert s["negative_variance"] == pytest.approx(neg_max * 0.99)
                    # the clamped parameters are usable by the classifier
                    S.classifier(s["positive_mean"], s["positive_variance"],
                                 s["negative_mean"], s["negative_variance"], 0.9,
                                 pd.DataFrame({"is_active": [1, 0]}))
                else:
                    assert s["positive_variance"] == pytest.approx((1 - s["positive_mean"]) / 2)
                    assert s["negative_variance"] == s["positive_variance"]
            changed = out.count("changed positive variance from 0.45 to ")
            assert changed == (count if case_id == "infeasible_beta_variance_capped" else 0)
            assert out.count("changed negative variance from 0.45 to ") == changed
            # deep copy per run: mutating one leaves its neighbours alone
            sims[0]["nr_plates"] = 99
            assert sims[1]["nr_plates"] == 1


def test_run_and_save_runs_one_simulation_and_writes_its_outputs(tmp_path, monkeypatch, subtests):
    """run_and_save: database row and timing, optional figures, opt-in fixed seed."""
    with subtests.test(msg="database_and_timing"):
        _seed(14)
        time_ls = []
        settings = _settings(src=str(tmp_path), name="runA", start_time="260102")
        i, sim_time, extra = S.run_and_save(0, settings, time_ls, total_sims=1)
        assert (i, extra) == (0, None)
        assert sim_time > 0 and time_ls == [sim_time]
        assert settings["sim_time"] == sim_time
        root = tmp_path / "260102" / "runA"
        assert sorted(p.name for p in root.iterdir()) == ["simulations.db"]
        back = S.read_simulations_table(str(root / "simulations.db"))
        assert len(back) == 1
        assert back.iloc[0]["variable_classifier_accuracy_sim_nr"] == 0
        assert back.iloc[0]["sim_time"] == pytest.approx(sim_time)

    with subtests.test(msg="plot_writes_both_figures"):
        _seed(15)
        time_ls = []
        settings = _settings(src=str(tmp_path), plot=True, name="runB",
                             start_time="260103", variable="nr_plates")
        assert S.run_and_save(2, settings, time_ls, total_sims=1)[0] == 2
        root = tmp_path / "260103" / "runB"
        assert (root / "dists" / "2_figure.pdf").is_file()
        assert (root / "nr_plates" / "2_figure.pdf").is_file()
        assert (root / "simulations.db").is_file()
        assert len(time_ls) == 1

    # BUG (fixed): settings['random_seed'] was overwritten with False before it was read.
    for case_id, flag, expected in [("seed_opt_in", True, [42]), ("seed_off", False, [])]:
        with subtests.test(msg=case_id), monkeypatch.context() as patch:
            seeded = []
            patch.setattr(S.random, "seed", lambda v: seeded.append(v))
            S.run_and_save(0, _settings(src=str(tmp_path), name=f"run_{case_id}",
                                        start_time="260104", random_seed=flag),
                           [], total_sims=1)
            assert seeded == expected


# ---------------------------------------------------------------------------
# run_multiple_simulations: the process pool and manager are the only fakes
# ---------------------------------------------------------------------------

class _FakeAsyncResult:
    """Runs the work eagerly; reports not-ready once so the progress loop runs."""

    def __init__(self, func, arglist, raise_on_get=None):
        self.polls = 0
        self._raise = raise_on_get
        self._value = None if raise_on_get is not None else [func(*a) for a in arglist]

    def ready(self):
        self.polls += 1
        return self.polls > 1

    def get(self):
        if self._raise is not None:
            raise self._raise
        return self._value


class _FakePool:
    """An in-process Pool that calls the initializer like the real one."""

    instances = []

    def __init__(self, workers, raise_on_get=None, initializer=None, initargs=()):
        self.workers = workers
        self.arglist = None
        self.raise_on_get = raise_on_get
        self.closed = False
        _FakePool.instances.append(self)
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starmap_async(self, func, arglist):
        self.arglist = list(arglist)
        return _FakeAsyncResult(func, self.arglist, self.raise_on_get)

    def close(self):
        self.closed = True

    def join(self):
        assert self.closed


class _FakeManager:
    """A Manager whose list() is a plain (or caller-given) list."""

    def __init__(self, backing=None):
        self._backing = backing

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def list(self):
        return [] if self._backing is None else self._backing


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


def _one_run(tmp_path, **overrides):
    """A sweep of exactly one simulation (no queue budget: the preference is read)."""
    return _sweep_settings(src=str(tmp_path), replicates=1, avg_genes_per_well=[4],
                           classifier_accuracy=[0.9], **overrides)


def test_run_multiple_simulations_runs_the_sweep_through_one_writer(tmp_path, monkeypatch,
                                                                    subtests, capsys):
    """run_multiple_simulations: every run reaches the pool and the database; worker count and failures."""
    monkeypatch.setattr(S, "_SIMULATION_WRITE_ENDPOINT", None)

    with subtests.test(msg="every_generated_sim_is_run_and_saved"), \
            monkeypatch.context() as patch:
        patch.setattr(S, "Pool", _FakePool)
        patch.setattr(S, "Manager", _FakeManager)
        _FakePool.instances = []
        _seed(20)
        settings = _one_run(tmp_path / "a")
        assert S.run_multiple_simulations(settings) is None
        pool, = _FakePool.instances
        assert pool.workers == 1
        assert len(pool.arglist) == 1
        index, sim_settings, time_ls, total = pool.arglist[0]
        assert (index, total) == (0, 1)
        assert sim_settings["classifier_accuracy"] == 0.9
        assert len(time_ls) == 1
        # start_time is stamped on the parent settings and inherited by each run
        assert settings["start_time"] == sim_settings["start_time"]
        assert re.fullmatch(r"\d{6}", settings["start_time"])
        db = tmp_path / "a" / settings["start_time"] / "sweep" / "simulations.db"
        assert len(S.read_simulations_table(str(db))) == 1
        assert "Progress: 1/1 Time/simulation " in capsys.readouterr().out

    # BUG (fixed): cpu_count() - 4 was <= 0 on small machines, giving Pool(0).
    for case_id, cpus, workers in [("two_cpus_floor_at_one", 2, 1), ("twelve_cpus", 12, 8)]:
        with subtests.test(msg=case_id), monkeypatch.context() as patch:
            patch.setattr(S, "Pool", _FakePool)
            patch.setattr(S, "Manager", _FakeManager)
            patch.setattr(S, "cpu_count", lambda: cpus)
            _FakePool.instances = []
            _seed(21)
            S.run_multiple_simulations(_one_run(tmp_path / case_id, max_workers=0))
            assert [p.workers for p in _FakePool.instances] == [workers]

    with subtests.test(msg="progress_read_failure_is_reported_not_fatal"), \
            monkeypatch.context() as patch:
        flaky = _FlakyList()
        patch.setattr(S, "Manager", lambda: _FakeManager(flaky))
        patch.setattr(S, "Pool", _FakePool)
        _seed(23)
        capsys.readouterr()
        settings = _one_run(tmp_path / "flaky")
        S.run_multiple_simulations(settings)
        out = capsys.readouterr().out
        assert "manager link lost" in out and "Traceback" in out
        assert flaky.len_calls == 1
        db = tmp_path / "flaky" / settings["start_time"] / "sweep" / "simulations.db"
        assert db.is_file()

    with subtests.test(msg="final_worker_exception_propagates_unchanged"), \
            monkeypatch.context() as patch:
        boom = RuntimeError("worker exploded")
        patch.setattr(S, "Manager", _FakeManager)
        patch.setattr(S, "Pool", lambda workers, **options: _FakePool(
            workers, raise_on_get=boom, **options))
        _seed(22)
        with pytest.raises(RuntimeError, match="worker exploded") as raised:
            S.run_multiple_simulations(_one_run(tmp_path / "boom"))
        assert raised.value is boom


class _Writer:
    """A _DatabaseWriteQueue double recording its lifecycle; finish() always fails."""

    instances = []

    def __init__(self, path, commit, report, ram_gib):
        self.path = path
        self.ram_gib = ram_gib
        self.endpoint = object()
        self._thread = None
        self.events = []
        _Writer.instances.append(self)

    def start(self):
        self._thread = object()
        self.events.append("start")

    def cancel(self):
        self.events.append("cancel")

    def finish(self):
        self.events.append("finish")
        raise OSError("writer could not flush")


class _LostPool(_FakePool):
    """A pool whose task submission fails after the writer has started."""

    def starmap_async(self, func, arglist):
        list(arglist)
        raise KeyError("pool lost")


def _refuse_pool(*args, **kwargs):
    raise MemoryError("no pool")


def test_run_multiple_simulations_refuses_bad_budgets_and_cleans_up_the_writer(
        tmp_path, monkeypatch, subtests):
    """run_multiple_simulations: queue budget outside 0-64 GiB refused; a started writer is cancelled."""
    monkeypatch.setattr(S, "_SIMULATION_WRITE_ENDPOINT", None)
    monkeypatch.setattr(S, "Manager", _FakeManager)
    monkeypatch.setattr(database_concurrency, "_DatabaseWriteQueue", _Writer)
    cases = [
        # id, budget, Pool, exception, message, writer events (None = no writer built)
        ("budget_below_zero", -0.5, None, ValueError, "between 0 and 64 GiB", None),
        ("budget_above_64", 64.5, None, ValueError, "between 0 and 64 GiB", None),
        ("pool_never_started_skips_cleanup", 1, _refuse_pool, MemoryError, "no pool", []),
        # the writer's own finish() error must not mask the original cause
        ("started_writer_cancelled", 1, _LostPool, KeyError, "pool lost",
         ["start", "cancel", "finish"]),
    ]
    for case_id, budget, pool, error, message, events in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as patch:
            _Writer.instances = []
            if pool is not None:
                patch.setattr(S, "Pool", pool)
            settings = _one_run(tmp_path, database_write_queue_gib=budget)
            with pytest.raises(error, match=message):
                S.run_multiple_simulations(settings)
            if events is None:
                assert _Writer.instances == []
                continue
            writer, = _Writer.instances
            assert writer.events == events
            assert writer.ram_gib == 1.0
            assert writer.path == str(tmp_path / settings["start_time"]
                                      / ".simulation_write_queue")
            if not events:
                assert writer._thread is None
