"""Item 894: MoA profiling evaluation with the CellProfiler Analyst classifiers and controls."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from spacr import cpa_classifiers as cpa
from spacr import moa_benchmark as mb
from spacr import sp_stats


MOAS = {"a1": "alpha", "a2": "alpha", "b1": "beta", "b2": "beta", "c1": "gamma", "c2": "gamma"}


def _wells(seed: int = 0, n_features: int = 6) -> pd.DataFrame:
    """Two plates (two weeks) with DMSO; six compounds, three per week, at two concentrations."""
    rng = np.random.default_rng(seed)
    centres = {"alpha": 0, "beta": 1, "gamma": 2}
    rows = []
    for plate in ("Week1_100", "Week2_200"):
        for replicate in range(3):
            rows.append({"plate": plate, "well": f"A{replicate:02d}", "compound": "DMSO",
                         "concentration": 0.0,
                         **{f"f{i}": rng.normal(0, 1) for i in range(n_features)}})
        for index, (compound, moa) in enumerate(MOAS.items()):
            if compound[-1] != plate[4]:
                continue
            for concentration in (0.1, 1.0):
                values = rng.normal(0, 0.1, n_features)
                values[centres[moa]] += 4.0 * concentration + 2.0
                rows.append({"plate": plate, "well": f"B{index:02d}{int(concentration * 10)}",
                             "compound": compound, "concentration": concentration,
                             **{f"f{i}": values[i] for i in range(n_features)}})
    return pd.DataFrame(rows)


def _labels() -> pd.DataFrame:
    return pd.DataFrame([{"compound": c, "concentration": k, "moa": m}
                         for c, m in MOAS.items() for k in (0.1, 1.0)]
                        + [{"compound": "DMSO", "concentration": 0.0, "moa": "DMSO"}])


def _percentiles(wells: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for plate, part in wells[wells.compound == "DMSO"].groupby("plate"):
        q = part.filter(like="f").quantile([0.01, 0.99])
        q.insert(0, "q", q.index.values)
        q.insert(0, "plate", plate)
        parts.append(q.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True)


def _settings(**over):
    settings = {"moa_eval_labels": _labels(), "moa_eval_methods": "1nn_cosine,lda",
                "moa_eval_seeds": "0,1", "moa_eval_shuffles": 2, "moa_eval_knn_shuffles": 5,
                "moa_eval_normalise": "zscore", "moa_eval_feature_qc": "none"}
    settings.update(over)
    return settings


def test_the_cpa_classifiers_fit_and_predict_as_cpa_builds_them():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 1, (20, 4)), rng.normal(3, 1, (20, 4)), rng.normal(-3, 1, (20, 4))])
    y = np.repeat([0, 1, 2], 20)
    for name in cpa.CPA_NAMES:
        model = cpa.make_cpa_model(f"cpa_{name}", seed=3).fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (60, 3) and np.allclose(proba.sum(axis=1), 1.0), name
        assert (model.predict(X) == y).mean() > 0.6, name
    forest = cpa.make_cpa_model("RandomForestClassifier", seed=7).fit(X, y)
    assert forest.estimator.get_params()["random_state"] == 7 and forest.scaler is None
    assert cpa.make_cpa_model("SVC").fit(X, y).scaler is not None
    with pytest.raises(ValueError, match="CPA offers"):
        cpa.make_cpa_model("cpa_Perceptron")


def test_fast_gentle_boosting_finds_the_separating_stump_and_stops():
    values = np.array([[0.1, 5.0], [0.2, 4.0], [0.9, 5.5], [1.0, 4.5]], np.float32)
    labels = -np.ones((4, 2), np.float32)
    labels[[0, 1], 0] = 1
    labels[[2, 3], 1] = 1
    model = cpa.fast_gentle_boosting(labels, values, rules=5)
    assert len(model) == 1
    feature, threshold, a, b, margin = model[0]
    assert feature == 0 and threshold == np.float32(0.2)
    assert a.argmax() == 1 and b.argmax() == 0 and margin == pytest.approx(1.0)
    assert cpa.fast_gentle_boosting(labels, np.zeros((4, 0), np.float32)) == []
    clf = cpa.CPAClassifier("FastGentleBoosting").fit(values, np.array(["x", "x", "y", "y"]))
    assert clf.predict(values).tolist() == ["x", "x", "y", "y"]
    noisy = np.random.default_rng(0).normal(size=(30, 3)).astype(np.float32)
    noisy_labels = -np.ones((30, 3), np.float32)
    noisy_labels[np.arange(30), np.arange(30) % 3] = 1
    assert len(cpa.fast_gentle_boosting(noisy_labels, noisy, rules=4)) == 4


def test_leave_one_compound_out_folds_on_a_toy_set():
    wells = _wells().merge(_labels(), on=["compound", "concentration"])
    wells = wells[wells.moa != "DMSO"].assign(week=lambda d: d.plate.str[:5])
    for model in ("lda", "shrinkage_lda", "knn", "cpa_LDA", "cpa_KNeighborsClassifier"):
        result = mb.loco_supervised(wells, model=model, batch_column="week",
                                    features=[f"f{i}" for i in range(6)], controls=())
        table = result["table"].set_index("evaluation")
        assert table.loc["NSC", "n_treatments"] == 12, model
        assert table.loc["NSC", "accuracy"] > (0.5 if "LDA" in model.upper() else 0.0), model
        assert table.loc["NSCB", "n_treatments"] == 12 and table.loc["NSCB", "n_wells"] > 0, model
    with pytest.raises(ValueError, match="model must be one of"):
        mb.loco_supervised(wells, model="cpa_Perceptron")
    assert mb._supervised_name("CPA_lda") == "cpa_LDA" and mb._supervised_name("LDA") == "lda"


def test_the_shuffled_control_follows_the_reference_seed_convention():
    wells = _wells().merge(_labels(), on=["compound", "concentration"])
    wells = wells[wells.moa != "DMSO"]
    options = dict(n_permutations=3, features=[f"f{i}" for i in range(6)], controls=())
    same = mb.shuffled_label_control("lda", wells, seed=0, rng_seed=0, **options)
    plain = mb.shuffled_label_control("lda", wells, seed=0, **options)
    shifted = mb.shuffled_label_control("lda", wells, seed=0, rng_seed=1, **options)
    assert same["accuracy"].tolist() == plain["accuracy"].tolist()
    assert shifted["n_permutations"].tolist() == [3]
    assert 0.0 <= shifted["accuracy"].iloc[0] <= 1.0


def test_moa_evaluation_scores_every_method_seed_and_control():
    wells = _wells()
    result = sp_stats.moa_evaluation(wells, None, _settings())
    summary = result["summary"]
    assert set(summary["method"]) == {"1nn_cosine", "lda"}
    assert sorted(summary["seed"].unique()) == [0, 1]
    assert set(summary["protocol"]) == {"NSC", "NSCB"}
    nsc = summary[(summary.method == "lda") & (summary.protocol == "NSC")]
    assert min(nsc["accuracy"]) > 0.5 and nsc["n_treatments"].tolist() == [12, 12]
    controls = result["controls"]
    assert set(controls["control"]) == {"shuffled", "majority"}
    majority = controls[controls.control == "majority"]
    assert majority["accuracy"].iloc[0] == pytest.approx(4 / 12)
    shuffled = controls[(controls.control == "shuffled") & (controls.method == "1nn_cosine")]
    assert shuffled["n_permutations"].iloc[0] == 5
    per_moa = result["per_moa"]
    assert set(per_moa["moa"]) == {"alpha", "beta", "gamma"}
    assert per_moa.groupby(["method", "protocol", "seed"])["n_treatments"].sum().eq(12).all()
    confusion = result["confusion"]["lda"]
    diagonal = confusion[confusion["moa"] == confusion["predicted_moa"]]
    assert diagonal[(diagonal.protocol == "NSC") & (diagonal.seed == 0)]["count"].sum() > 6
    assert result["features"] == [f"f{i}" for i in range(6)]


def test_normalisations_feature_qc_and_metadata_join():
    wells = _wells(n_features=4)
    wells.loc[wells.index[3:6], "f3"] = np.nan
    wells["f2"] = wells["f2"] * 1e6
    wells["Location_x"] = 1.0
    metadata = wells[["plate", "well", "compound", "concentration"]].rename(
        columns={"plate": "P", "well": "W", "compound": "C", "concentration": "K"})
    bare = wells.drop(columns=["compound", "concentration"])
    common = dict(moa_eval_methods="1nn_cosine", moa_eval_seeds="0", moa_eval_controls="",
                  moa_eval_metadata_columns="P,W,C,K", moa_eval_protocols="NSC")
    for method in ("zscore", "robust", "none"):
        result = sp_stats.moa_evaluation(bare, metadata,
                                         _settings(moa_eval_normalise=method, **common))
        assert "Location_x" not in result["features"]
        assert set(result["summary"]["protocol"]) == {"NSC"}
    qc = sp_stats.moa_evaluation(bare, metadata, _settings(
        moa_eval_normalise="zscore", moa_eval_feature_qc="standard", **common))
    assert "f3" in qc["qc_dropped"] and "f3" not in qc["features"]
    ljosa = sp_stats.moa_evaluation(wells, None, _settings(
        moa_eval_normalise="ljosa", moa_eval_percentiles=_percentiles(wells),
        moa_eval_methods="1nn_cosine", moa_eval_seeds="0", moa_eval_controls="majority"))
    assert set(ljosa["controls"]["control"]) == {"majority"}
    assert len(ljosa["features"]) == 4


def test_moa_evaluation_refuses_bad_settings():
    wells = _wells()
    for over, match in ((dict(moa_eval_normalise="tvn"), "moa_eval_normalise"),
                        (dict(moa_eval_protocols="NSCX"), "protocols"),
                        (dict(moa_eval_methods="cpa_Perceptron"), "unknown MoA"),
                        (dict(moa_eval_normalise="ljosa"), "percentiles"),
                        (dict(moa_eval_labels=""), "moa_eval_labels")):
        with pytest.raises(ValueError, match=match):
            sp_stats.moa_evaluation(wells, None, _settings(**over))
    with pytest.raises(ValueError, match="metadata"):
        sp_stats.moa_evaluation(wells.drop(columns=["compound"]), None, _settings())
    with pytest.raises(ValueError, match="metadata columns"):
        sp_stats.moa_evaluation(wells.drop(columns=["compound"]), wells,
                                _settings(moa_eval_metadata_columns="plate,well"))
    no_control = wells[wells.compound != "DMSO"]
    with pytest.raises(ValueError, match="control well"):
        sp_stats.moa_evaluation(no_control, None, _settings())


def test_run_profiling_evaluation_writes_the_result_files(tmp_path):
    wells = _wells()
    path = tmp_path / "wells_mean.csv"
    wells.to_csv(path, index=False)
    labels = tmp_path / "moa.csv"
    _labels().to_csv(labels, index=False)
    result = mb.run_profiling_evaluation({
        "moa_eval_profiles": str(path), "moa_eval_labels": str(labels),
        "moa_eval_methods": "1nn_cosine,knn", "moa_eval_seeds": "0", "moa_eval_shuffles": 1,
        "moa_eval_knn_shuffles": 2, "moa_eval_normalise": "zscore"})
    folder = tmp_path / "results" / "profiling_evaluation"
    names = sorted(p.name for p in folder.iterdir())
    assert names == ["confusion_1nn_cosine.csv", "confusion_knn.csv", "controls.csv",
                     "evaluation.json", "per_moa.csv", "predictions.csv", "summary.csv"]
    summary = pd.read_csv(folder / "summary.csv")
    assert set(summary["method"]) == {"1nn_cosine", "knn"} and len(summary) == 4
    record = json.loads((folder / "evaluation.json").read_text())
    assert record["n_features"] == 6 and record["settings"]["moa_eval_seeds"] == "0"
    assert len(result["paths"]) == 7
    with pytest.raises(ValueError, match="moa_eval_profiles"):
        mb.run_profiling_evaluation({})


def test_the_evaluation_is_registered_and_alpha():
    from spacr import cli
    from spacr.settings import ALPHA_FEATURES

    module = cli.resolve_module("profiling_evaluation")
    assert module is not None and module.entry == "spacr.moa_benchmark:run_profiling_evaluation"
    defaults = cli.module_defaults(module)
    assert defaults["moa_eval_seeds"] == "0,1,2,3,4"
    assert defaults["moa_eval_exclude"] == sp_stats.MOA_EVALUATION_EXCLUDE
    assert set(ALPHA_FEATURES[894]["module_settings"]["profiling_evaluation"]) == \
        set(mb._EVALUATION_DEFAULTS)


def test_a_profile_table_that_already_names_the_moa_and_the_batch_needs_neither_table():
    wells = _wells()
    settings = _settings(moa_eval_methods="1nn_cosine", moa_eval_seeds="0", moa_eval_controls="")
    joined = sp_stats.moa_evaluation(wells, None, settings)
    labelled = wells.assign(moa=wells["compound"].map({**MOAS, "DMSO": "DMSO"}),
                            week=wells["plate"].str.split("_").str[0])
    own = sp_stats.moa_evaluation(labelled, None, dict(settings, moa_eval_labels=None))
    assert set(own["summary"]["protocol"]) == {"NSC", "NSCB"}
    pd.testing.assert_frame_equal(own["summary"].reset_index(drop=True),
                                  joined["summary"].reset_index(drop=True))
