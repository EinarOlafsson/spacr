"""Edge branches of the BMC paper items (871-911) that the item tests do not reach.

Each test pins what the branch does, not only that it runs: the guide-efficacy
fit with degenerate weights or rows, the cell-cycle phases on chosen G1 cells,
phase profiles of a well without control cells, the penetrance test's empty
folds and guide-heterogeneity guards, the escaper classes, the per-well
embedding statistics read in chunks, and the classifier-evaluation guards.
"""
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import gene_measurement_compare as G
from spacr import ml


def test_efficacy_fit_with_non_positive_weights_and_zero_rows():
    rows = np.array([[1.0, 2.0], [2.0, 4.0]])
    m, e, done = G._rank_one_efficacy(rows, np.array([-1.0, -1.0]), iters=5, emax=1.0)
    assert done >= 1 and e.max() == pytest.approx(1.0)
    assert np.all(np.isfinite(m))
    zero_m, zero_e, zero_done = G._rank_one_efficacy(np.zeros((3, 2)), np.ones(3), iters=5, emax=1.0)
    assert zero_done == 1
    assert np.array_equal(zero_e, np.ones(3)) and np.array_equal(zero_m, np.zeros(2))


def test_phases_use_the_given_g1_cells_and_wells_without_controls_stay_uncentred():
    rng = np.random.default_rng(0)
    n = 400
    cells = pd.DataFrame({"dna": np.r_[rng.normal(1000, 40, n // 2), rng.normal(1900, 60, n // 2)],
                          "mitotic": False, "well": "A1"})
    control = pd.Series(False, index=cells.index)
    chosen = pd.Series(np.r_[np.ones(n // 2, bool), np.zeros(n // 2, bool)], index=cells.index)
    phases = G.cell_cycle_phases(cells, dna="dna", control=control, mode_cells=chosen)
    assert (phases["phase"].iloc[: n // 2] == "G1").mean() > 0.9
    assert (phases["phase"].iloc[n // 2:] == "SG2").mean() > 0.9
    frame = pd.DataFrame({"well": ["A1", "A1", "B1", "B1"], "phase": ["G1"] * 4,
                          "sgRNA": ["nt1", "g1", "nt2", "g2"],
                          "gene": ["nontargeting", "G", "nontargeting", "G"],
                          "PC_0": [1.0, 3.0, 5.0, 9.0]})
    guides, _genes = G.phase_profiles(frame, ["PC_0"], control=pd.Series([True, False, False, False]),
                                      guide="sgRNA")
    by_guide = guides.set_index("guide")["G1_PC_0"]
    assert by_guide["g1"] == pytest.approx(2.0)
    assert by_guide["g2"] == pytest.approx(9.0)


def _penetrance(x, folds):
    n = len(x)
    return G._Penetrance(x, np.array(["g"] * n), folds, np.arange(n) % 4, folds=3, lam=0.5, reps=5,
                         simulations=2, seed=1, dbic=10.0, min_separation=1.0)


def test_crossfit_skips_folds_without_gene_cells_or_training_cells():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(60, 3))
    fold = np.r_[np.zeros(30, int), np.ones(30, int)]
    test = _penetrance(x, fold)
    u, z = test.crossfit(np.arange(30), np.arange(30, 60))
    assert np.isnan(u).all() and np.isnan(z).all()


@pytest.mark.parametrize("mu1, expected_resp", [(1000.0, 0), (2.0, 1)])
def test_guide_heterogeneity_needs_responders_and_non_responders(monkeypatch, mu1, expected_resp):
    x = np.zeros((40, 2))
    test = G._Penetrance(x, np.array(["a", "b"] * 20), np.zeros(40, int), np.arange(40) % 4, folds=1,
                         lam=0.5, reps=5, simulations=0, seed=1, dbic=10.0, min_separation=1.0)
    monkeypatch.setattr(test, "crossfit", lambda rows, ref: (np.linspace(0.01, 0.99, len(rows)),
                                                              np.full(len(rows), 5.0)))
    monkeypatch.setattr(G, "escaper_statistics",
                        lambda z, dbic, min_separation: ({"escaper_raw": False, "mix_mu1": mu1},
                                                         np.zeros(len(z))))
    row, guides, _cells = test.analyse("G", np.arange(40), np.arange(0), "active gene")
    assert np.isnan(row["guide_het_p"]) and row["guide_driven"] is False
    assert (guides["resp_frac"] == expected_resp).all()


def test_escaper_classes_name_each_explanation():
    base = {"set": "escaper", "guide_driven": False, "phase_chi_p": 0.5, "cycling_share_diff": 0.0,
            "Pwrong_mw_p": 0.5, "error_explainable_share": 0.0, "nonresp_share": 1.0, "ashman_D": 1.0}
    assert G._escaper_class({**base, "phase_chi_p": 1e-4, "cycling_share_diff": -0.3}) == "cell-cycle-coupled"
    assert G._escaper_class({**base, "Pwrong_mw_p": 1e-3, "error_explainable_share": 0.6}) == "genotype-error-like"
    assert G._escaper_class({**base, "ashman_D": 2.5}) == "within-guide escapers (candidate)"
    assert G._escaper_class(base) == "weakly bimodal (heavy-tailed shift)"


def test_well_statistics_read_in_chunks_that_hold_no_control_rows():
    embedding = np.arange(24, dtype=float).reshape(8, 3)
    rows = np.arange(8)
    wells = np.array(["A1"] * 8)
    control = np.array([True, False, False, False, False, False, False, True])
    mean, sd, median, mad = G._well_statistics(embedding, rows, wells, control, chunk=2)
    assert np.allclose(mean["A1"], embedding.mean(0))
    assert np.allclose(median["A1"], np.median(embedding[control], 0))


def test_classifier_metrics_without_a_valid_auroc_and_unseen_classes():
    out = ml.classifier_metrics([1, 2, 1, 2], [1, 2, 2, 2], [1, 2], "model",
                                y_score=[[np.nan, np.nan]] * 4)
    assert np.isnan(out["auroc"]) and out["accuracy"] == pytest.approx(0.75)
    df = pd.DataFrame({"f": np.r_[np.zeros(10), np.ones(10)], "label": [1] * 10 + [2] * 10})
    folds = [("all", df.index, df.index)]
    predictions, scores = ml._ml_eval_predict(df, ["f"], folds, [1], "logistic_regression", 5)
    assert list(scores.columns) == [1] and scores[1].between(0, 1).all()
    assert set(predictions) == {1, 2}


def test_annotation_sample_without_a_feature_table_reuses_existing_columns(tmp_path, monkeypatch):
    src = tmp_path / "plate"
    (src / "measurements").mkdir(parents=True)
    db = src / "measurements" / "measurements.db"
    png = pd.DataFrame({"png_path": [f"/c/o{i}.png" for i in range(20)],
                        "file_name": [f"o{i}.png" for i in range(20)],
                        "annotate": [None] * 20, "annotate_sample": [None] * 20})
    with sqlite3.connect(db) as con:
        png.to_sql("png_list", con, index=False)
    sample = ml.export_annotation_sample({"src": str(src), "annotate_sample_n": 5, "annotate_sample_seed": 3})
    assert len(sample) == 5
    with sqlite3.connect(db) as con:
        marked = pd.read_sql("SELECT annotate_sample FROM png_list", con)
    assert sorted(marked["annotate_sample"].dropna().astype(int)) == [1, 2, 3, 4, 5]


def test_a_training_run_with_ml_eval_report_evaluates_what_it_fitted(tmp_path, rng, monkeypatch):
    from tests.test_cov_ml_ml_scores import N_OBJ, _make_src, _ml_settings

    annotation = [float(i % 2) for i in range(N_OBJ)]
    src = _make_src(tmp_path, "plate_eval", rng, annotation=annotation)
    calls = []
    monkeypatch.setattr(ml, "_evaluate_after_training",
                        lambda settings, output, label: calls.append((label, len(output[0]))))
    output, _ = ml.generate_ml_scores(_ml_settings(src, annotation_column="test", positive_control=None,
                                                   negative_control=None, ml_eval_report=True))
    assert calls == [("test", len(output[0]))] or (len(calls) == 1 and calls[0][1] == len(output[0]))
    assert os.path.isdir(src)


def _votes(objects, codes, guides, **kw):
    from spacr import ops_sbs

    n, cycles = len(codes), len(codes[0])
    data = np.zeros((n, cycles, 4), np.float32)
    for i, code in enumerate(codes):
        for c, letter in enumerate(code):
            data[i, c, "GTAC".index(letter)] = 1.0
    margin = np.ones((n, cycles), np.float32)
    return ops_sbs.vote_mapped_reads(np.asarray(objects), codes, np.ones(n, np.float32), data, margin,
                                     ops_sbs.LibraryMatcher(guides), **kw)


def test_mapped_votes_count_corrections_ties_and_weak_majorities():
    guides = ["GTAC", "CATG"]
    evidence = {}
    corrected = _votes([1, 1], ["GTAA", "GTAA"], guides, max_distance=1, evidence=evidence)
    assert corrected[1]["guide"] == "GTAC" and corrected[1]["n_corrected"] == 2
    assert dict(evidence["strength"]) == {}
    refused = _votes([1, 1], ["GTAA", "GTAA"], guides, max_distance=1, runner_up_margin=0.1)
    assert refused == {}
    assert _votes([2, 2], ["GTAC", "CATG"], guides) == {}
    weak = _votes([3, 3, 3], ["GTAC", "GTAC", "CATG"], guides, min_fraction=0.8)
    assert weak == {}


def test_unmixing_falls_back_to_identity_on_singular_axes_and_boundaries_need_points(monkeypatch):
    from spacr import ops_sbs

    values = np.arange(1, 49, dtype=float).reshape(6, 2, 4)

    def singular(matrix):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(np.linalg, "inv", singular)
    unmixed = ops_sbs._argmax_median_unmix64(values)
    flat = values.reshape(-1, 4)
    assert np.allclose(unmixed.reshape(-1, 4), flat / np.median(flat, axis=0))
    monkeypatch.undo()
    gap, index = ops_sbs.nearest_boundaries(np.zeros((0, 2)), np.array([[1.0, 1.0]]), np.array([4.0]))
    assert gap.shape == (0, 2) and index.shape == (0, 2)
    gap, index = ops_sbs.nearest_boundaries(np.array([[0.0, 0.0]]), np.zeros((0, 2)), np.zeros(0))
    assert np.isinf(gap).all() and (index == -1).all()


def test_library_calibration_skips_a_cycle_whose_axes_are_singular(monkeypatch):
    from spacr import ops_sbs

    rng = np.random.default_rng(0)
    guides, codes, values = _library_reads(rng)
    _, expected, _ = ops_sbs.called_bases(values, compensate=True, method="argmax_median", gpu=False)
    real_inv = np.linalg.inv
    calls = {"n": 0}

    def singular_after_first(matrix):
        calls["n"] += 1
        if calls["n"] > 1:
            raise np.linalg.LinAlgError("singular")
        return real_inv(matrix)

    monkeypatch.setattr(np.linalg, "inv", singular_after_first)
    _, called, margin = ops_sbs.library_calibrated_bases(values, guides)
    assert calls["n"] > 1
    assert len(called) == len(codes) and margin.shape == (len(codes), values.shape[1])


def _library_reads(rng):
    from tests.test_ops_decode_rule_a_s2_873 import _library_reads as reads

    return reads(rng)
