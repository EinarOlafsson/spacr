"""Cross-validated searches of ``spacr.hyperparam`` and what they fit.

One table-driven test per unit, each row run as a subtest: the leak-safe fold
builder (``build_folds``), the fold-averaging runner (``cv_search``), the
classical-ML estimator ladder (``build_sklearn_model``) and its fold fit
function (``sklearn_cv_fit_fn``), the deep per-fold fit function
(``classify_cv_fit_fn``), the measurement-database loader
(``load_search_data``) and the GUI entry point (``run_search_for_app``).
Folds are real scikit-learn / spaCR splits on small synthetic plates; deep
training and optional boosting backends are replaced by recording doubles.
"""
from __future__ import annotations

import sqlite3
import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest

import spacr.io as io_module
from spacr import hyperparam as hp
from spacr.hyperparam import SearchSpace


def _labels(n=60):
    """Alternating binary labels, enough of each class for five folds."""
    return np.array([i % 2 for i in range(n)])


def _wells(n=60, per_well=6):
    """One well id per block of rows, the way a plate is laid out."""
    return np.array([f"p1_A_{i // per_well}" for i in range(n)])


def _fold_fit(params, train_idx, val_idx):
    """Deterministic stand-in estimator: deeper scores higher."""
    return 0.5 + 0.01 * float(params["depth"]) + 0.001 * len(val_idx)


def _fold_csv(values, column="accuracy"):
    """A per-fold results frame of the shape a CV training run writes."""
    return pd.DataFrame({"fold": range(1, len(values) + 1), column: values})


def _separable(n=48, seed=0):
    """A two-class problem an estimator can actually learn."""
    rng = np.random.default_rng(seed)
    labels = np.array([i % 2 for i in range(n)])
    return rng.normal(size=(n, 4)) + labels[:, None] * 3.0, labels


def _used(folds):
    """Every index a fold list hands out, train and validation together."""
    out = []
    for tr, va in folds:
        out.extend(np.asarray(tr).tolist())
        out.extend(np.asarray(va).tolist())
    return out


# ===========================================================================
# build_folds
# ===========================================================================

def test_build_folds_keeps_groups_whole_and_the_test_split_out(subtests):
    """Fewer than two folds, an out-of-range test index, an all-excluded
    pool, missing group ids and anonymous crop names are refused; grouped
    folds never split a well, partition the non-test pool, and say so when
    the split is by cell or filenames went unparsed."""
    for cid, args, kwargs, message in [
        ("one-fold", (_labels(), 1), dict(groups=_wells()),
         "n_folds must be at least 2 to cross-validate, got 1. With one fold "
         "there is no held-out data to score on."),
        ("index-outside", (_labels(), 3), dict(groups=_wells(),
                                               exclude=[0, 99]),
         "excluded (test) indices must lie inside [0, 60), got min 0 max 99."),
        ("negative-index", (_labels(), 3), dict(groups=_wells(),
                                                exclude=[-1]),
         "excluded (test) indices must lie inside [0, 60), got min -1 max -1."),
        ("everything-excluded", (_labels(), 3), dict(groups=_wells(),
                                                     exclude=range(60)),
         "Every sample was excluded as test data, so there is nothing left "
         "to cross-validate on."),
        ("no-group-ids", (_labels(), 3), dict(group_by="well"),
         "No group ids were available for group_by='well'. Pass `groups=` or "
         "`filenames=`; a random fallback would make related crops cross "
         "folds and report an optimistic search score."),
    ]:
        with subtests.test(case=cid):
            with pytest.raises(ValueError) as info:
                hp.build_folds(*args, **kwargs)
            assert str(info.value) == message

    with subtests.test(case="anonymous-crop-name"):
        with pytest.raises(ValueError, match="does not encode a well"):
            hp.build_folds(_labels(), 2,
                           filenames=[f"crop_{i}.png" for i in range(60)],
                           group_by="well")

    for cid, exclude in [("no-exclusion", None), ("empty-exclusion", []),
                         ("first-two-wells", list(range(12)))]:
        with subtests.test(case=f"grouped-{cid}"):
            labels, wells = _labels(), _wells()
            folds, notes = hp.build_folds(labels, 3, groups=wells,
                                          exclude=exclude)
            pool = sorted(set(range(60)) - set(exclude or []))
            assert notes == [] and len(folds) == 3
            assert sorted(np.concatenate([va for _tr, va in folds])
                          .tolist()) == pool
            for tr, va in folds:
                assert sorted(tr.tolist() + va.tolist()) == pool
                assert not set(wells[tr]) & set(wells[va])

    with subtests.test(case="small-explicit-groups-with-exclusion"):
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        groups = ["A1", "A1", "A2", "A2", "B1", "B1", "B2", "B2"]
        folds, _ = hp.build_folds(labels, n_folds=2, groups=groups,
                                  exclude=[0, 4])
        assert sorted(set(_used(folds))) == [1, 2, 3, 5, 6, 7]

    with subtests.test(case="by-cell-warns-it-inflates"):
        folds, notes = hp.build_folds(_labels(), 3, group_by="cell")
        assert [len(va) for _tr, va in folds] == [20, 20, 20]
        assert notes == [
            "group_by='cell': folds are a plain stratified split. Object "
            "crops from the same well will straddle folds, which inflates "
            "scores."]

    with subtests.test(case="wells-parsed-from-filenames"):
        names = [f"plate1_A0{1 + i // 20}_f{i % 3}_o{i}.png"
                 for i in range(60)]
        folds, notes = hp.build_folds(_labels(), 2, filenames=names,
                                      group_by="well")
        assert notes == []
        assert sorted(len(va) for _tr, va in folds) == [20, 40]
        wells = np.array([n.split("_")[1] for n in names])
        for tr, va in folds:
            assert not set(wells[tr]) & set(wells[va])

    with subtests.test(case="unparsed-filenames-are-reported"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(io_module, "_cv_group_ids",
                       lambda paths, level, verbose=True: (
                           [f"w{i // 20}" for i in range(len(paths))], 3))
            _folds, notes = hp.build_folds(
                _labels(), 2, filenames=[f"c{i}.png" for i in range(60)],
                group_by="well")
        assert notes == ["3 filenames did not carry a 'well' level and "
                         "became their own group."]


# ===========================================================================
# cv_search
# ===========================================================================

def test_cv_search_averages_folds_and_refuses_test_leakage(subtests):
    """Each configuration's score is its fold mean with the spread beside
    it; the notes say how the folds were grouped and how many test samples
    were excluded; folds touching the test split are refused; random mode
    samples distinct configurations up to the space size; a fold with no or
    a non-finite score fails only its own trial."""
    with subtests.test(case="grid-is-the-mean-over-folds"):
        result = hp.cv_search(_fold_fit, SearchSpace({"depth": [1, 2, 3]}),
                              labels=_labels(), groups=_wells(), n_folds=3)
        folds, _ = hp.build_folds(_labels(), 3, groups=_wells())
        sizes = [len(va) for _tr, va in folds]
        for trial in result.trials:
            expected = [0.5 + 0.01 * trial.params["depth"] + 0.001 * s
                        for s in sizes]
            assert trial.extra_metrics["fold_scores"] == pytest.approx(
                expected)
            assert trial.score == pytest.approx(np.mean(expected))
            assert trial.extra_metrics["fold_std"] == pytest.approx(
                np.std(expected))
            assert trial.extra_metrics["n_folds"] == 3
        assert result.best.params == {"depth": 3}
        assert result.notes[:3] == [
            "Scored on 3 grouped (group_by='well') cross-validation folds, "
            "seed 0. 0 test samples were excluded from every fold, so no "
            "configuration was selected using test data.",
            "Grid search over depth=[1, 2, 3]  (3 configurations)",
            "Each trial's score is the mean over 3 folds; the fold-to-fold "
            "standard deviation is reported per trial and is used as the "
            "noise yardstick when judging whether the winner is real."]

    with subtests.test(case="ungrouped-says-so"):
        result = hp.cv_search(_fold_fit, SearchSpace({"depth": [1]}),
                              labels=_labels(), group_by="cell", n_folds=2)
        assert result.notes[0].startswith(
            "Scored on 2 ungrouped cross-validation folds, seed 0.")
        assert result.notes[2].startswith("group_by='cell': folds are a ")

    with subtests.test(case="folds-touching-test-are-refused"):
        folds = [(np.arange(0, 30), np.arange(30, 60))]
        with pytest.raises(ValueError) as info:
            hp.cv_search(_fold_fit, SearchSpace({"depth": [1]}),
                         labels=_labels(), folds=folds, test_idx=[5, 6, 7])
        assert str(info.value).startswith(
            "fold 0 contains 3 held-out test indices (e.g. [5, 6, 7]). "
            "Scoring a hyperparameter search on the test split leaks it")

    with subtests.test(case="honest-prebuilt-folds-are-accepted"):
        folds = [(np.arange(10, 35), np.arange(35, 60)),
                 (np.arange(35, 60), np.arange(10, 35))]
        result = hp.cv_search(_fold_fit, SearchSpace({"depth": [1, 2]}),
                              labels=_labels(), folds=folds,
                              test_idx=range(10))
        assert result.ok
        assert result.notes[0] == (
            "Scored on 2 ungrouped cross-validation folds, seed 0. 10 test "
            "samples were excluded from every fold, so no configuration was "
            "selected using test data.")
        assert result.best.extra_metrics["fold_scores"] == pytest.approx(
            [0.545, 0.545])

    with subtests.test(case="test-split-never-reaches-the-fit"):
        handed = set()

        def _record(params, tr, va):
            handed.update(np.asarray(tr).tolist())
            handed.update(np.asarray(va).tolist())
            return 0.5

        hp.cv_search(_record, SearchSpace({"depth": [1]}), labels=_labels(),
                     groups=_wells(), n_folds=3, test_idx=range(12))
        assert handed == set(range(12, 60))

    with subtests.test(case="random-samples-distinct-configurations"):
        space = SearchSpace({"depth": [1, 2, 3, 4], "width": [10, 20]})
        result = hp.cv_search(lambda p, tr, va: float(p["depth"] + p["width"]),
                              space, labels=_labels(), groups=_wells(),
                              n_folds=2, n_trials=5, seed=3)
        assert [t.label() for t in result.trials] == [
            "depth=2, width=10", "depth=3, width=20", "depth=1, width=10",
            "depth=4, width=20", "depth=4, width=10"]
        assert result.notes[1] == ("Random search: 5 of 8 configurations "
                                   "sampled with seed 3")

    with subtests.test(case="random-capped-at-space-size"):
        result = hp.cv_search(_fold_fit, SearchSpace({"depth": [1, 2, 3, 4]}),
                              labels=_labels(), groups=_wells(), n_folds=2,
                              n_trials=99, seed=1)
        assert sorted(t.params["depth"] for t in result.trials) == [1, 2, 3, 4]

    with subtests.test(case="unscored-fold-fails-the-trial"):
        result = hp.cv_search(lambda p, tr, va: None,
                              SearchSpace({"depth": [1]}), labels=_labels(),
                              groups=_wells(), n_folds=2)
        assert not result.ok
        assert result.trials[0].error == (
            "ValueError: fit function returned no score for a fold; every "
            "fold must be scored for the mean to mean anything")

    with subtests.test(case="nan-fold-loses-only-its-trial"):
        result = hp.cv_search(
            lambda p, tr, va: float("nan") if p["depth"] == 2 else 0.7,
            SearchSpace({"depth": [1, 2]}), labels=_labels(),
            groups=_wells(), n_folds=2)
        assert result.best.params == {"depth": 1}
        assert result.best.score == pytest.approx(0.7)
        failed = result.trials[1]
        assert failed.score is None
        assert failed.error.startswith(
            "ValueError: fit function returned a non-finite score (nan)")

    with subtests.test(case="per-fold-extras-are-collected"):
        result = hp.cv_search(
            lambda p, tr, va: (0.7, {"n_train": len(tr), "tag": "x"}),
            SearchSpace({"depth": [1]}), labels=_labels(), groups=_wells(),
            n_folds=3)
        folds, _ = hp.build_folds(_labels(), 3, groups=_wells())
        assert result.best.extra_metrics["fold_n_train"] == [
            len(tr) for tr, _va in folds]
        assert result.best.extra_metrics["fold_tag"] == ["x", "x", "x"]


# ===========================================================================
# build_sklearn_model
# ===========================================================================

def _recording_backend(mp, module_name, class_name):
    class RecordingEstimator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    backend = types.ModuleType(module_name)
    setattr(backend, class_name, RecordingEstimator)
    mp.setitem(sys.modules, module_name, backend)
    return RecordingEstimator


def test_build_sklearn_model_builds_the_estimator_the_real_run_fits(subtests):
    """Each offered model type builds its estimator with the trial's
    parameters mapped exactly (C = 1/reg_lambda, never divided by zero; the
    boosting backends get their own names); a missing optional backend
    names its pip package and an unknown type lists the real ones."""
    params = {"n_estimators": 7, "learning_rate": 0.2, "reg_alpha": 0.4,
              "reg_lambda": 2.0, "max_depth": 3}
    for cid, model_type, cls, expected in [
        ("random_forest", "random_forest", "RandomForestClassifier",
         dict(n_estimators=7, max_depth=3, random_state=1, n_jobs=2)),
        ("extra_trees", "extra_trees", "ExtraTreesClassifier",
         dict(n_estimators=7, max_depth=3, random_state=1, n_jobs=2)),
        ("gradient_boosting", "gradient_boosting",
         "HistGradientBoostingClassifier",
         dict(max_iter=7, learning_rate=0.2, random_state=1)),
        ("mlp", "mlp", "MLPClassifier",
         dict(max_iter=200, learning_rate_init=0.2, alpha=0.4,
              random_state=1)),
    ]:
        with subtests.test(case=cid):
            model = hp.build_sklearn_model(model_type, params, seed=1,
                                           n_jobs=2)
            assert type(model).__name__ == cls
            got = model.get_params()
            assert {k: got[k] for k in expected} == expected

    for cid, reg_lambda, c_value in [("lambda-4", 4.0, 0.25),
                                     ("lambda-0", 0.0, 1e9),
                                     ("default", None, 1.0)]:
        with subtests.test(case=f"logistic-{cid}"):
            p = {} if reg_lambda is None else {"reg_lambda": reg_lambda}
            model = hp.build_sklearn_model("logistic_regression", p, seed=5)
            assert [type(step).__name__ for step in model] == [
                "StandardScaler", "LogisticRegression"]
            assert model[-1].C == pytest.approx(c_value)
            assert (model[-1].max_iter, model[-1].random_state) == (1000, 5)

    with subtests.test(case="shrinkage-lda-is-a-pipeline"):
        model = hp.build_sklearn_model("shrinkage_lda", {})
        assert type(model).__name__ == "Pipeline"

    with subtests.test(case="svm-is-the-calibrated-svc"):
        model = hp.build_sklearn_model("svm", {}, seed=3)
        assert type(model).__name__ == "CalibratedClassifierCV"
        assert type(model.estimator).__name__ == "SVC"
        assert (model.method, model.cv, model.n_jobs, model.ensemble,
                model.estimator.random_state) == ("sigmoid", 3, 1, False, 3)
        features, labels = _separable()
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            model.fit(features, labels)
        assert model.predict_proba(features[:2]).shape == (2, 2)

    with subtests.test(case="xgboost"):
        pytest.importorskip("xgboost")
        model = hp.build_sklearn_model(
            "xgboost", {"n_estimators": 7, "learning_rate": 0.3,
                        "reg_alpha": 0.5, "reg_lambda": 2.0}, seed=11)
        assert type(model).__name__ == "XGBClassifier"
        got = model.get_params()
        assert (got["n_estimators"], got["random_state"],
                got["eval_metric"]) == (7, 11, "logloss")
        assert (got["learning_rate"], got["reg_alpha"],
                got["reg_lambda"]) == pytest.approx((0.3, 0.5, 2.0))

    for cid, module_name, class_name, expected in [
        ("lightgbm", "lightgbm", "LGBMClassifier",
         {"n_estimators": 17, "learning_rate": 0.03, "reg_alpha": 0.4,
          "reg_lambda": 2.5, "random_state": 9, "n_jobs": 3, "verbose": -1}),
        ("catboost", "catboost", "CatBoostClassifier",
         {"iterations": 17, "learning_rate": 0.03, "l2_leaf_reg": 2.5,
          "random_state": 9, "verbose": False}),
    ]:
        with subtests.test(case=f"installed-{cid}"):
            with pytest.MonkeyPatch.context() as mp:
                recording = _recording_backend(mp, module_name, class_name)
                model = hp.build_sklearn_model(
                    cid, {"n_estimators": 17, "learning_rate": 0.03,
                          "reg_alpha": 0.4, "reg_lambda": 2.5},
                    seed=9, n_jobs=3)
            assert isinstance(model, recording)
            assert model.kwargs == expected

    for package in ("xgboost", "lightgbm", "catboost"):
        with subtests.test(case=f"missing-{package}"):
            with pytest.MonkeyPatch.context() as mp:
                mp.setitem(sys.modules, package, None)
                with pytest.raises(ImportError) as info:
                    hp.build_sklearn_model(package, {})
            assert str(info.value) == (
                f"model_type_ml='{package}' needs the '{package}' package: "
                f"pip install {package}")

    with subtests.test(case="unknown-type-lists-the-real-ones"):
        with pytest.raises(ValueError) as info:
            hp.build_sklearn_model("randomforest", {})
        assert str(info.value) == (
            "Unsupported model_type_ml 'randomforest'. Choose one of "
            "random_forest, extra_trees, logistic_regression, "
            "gradient_boosting, xgboost, lightgbm, catboost, svm, mlp.")


# ===========================================================================
# sklearn_cv_fit_fn
# ===========================================================================

def test_sklearn_cv_fit_fn_fits_on_train_and_scores_on_validation(subtests):
    """The fold fit function sees only the two index arrays it is handed,
    scores accuracy / ROC-AUC / F1 (macro for three classes) on the
    validation rows, ranks by the margin when there are no probabilities,
    and reports the fold sizes."""
    features, labels = _separable()
    for criterion in ("accuracy", "roc_auc", "f1"):
        with subtests.test(case=f"binary-{criterion}"):
            fit = hp.sklearn_cv_fit_fn(features, labels,
                                       model_type="logistic_regression",
                                       criterion=criterion, n_jobs=1)
            assert fit({"n_estimators": 10}, np.arange(0, 32),
                       np.arange(32, 48)) == (1.0, {"n_train": 32,
                                                    "n_val": 16})

    with subtests.test(case="three-classes-macro-f1"):
        rng = np.random.default_rng(1)
        labels3 = np.array([i % 3 for i in range(60)])
        features3 = rng.normal(size=(60, 3)) + labels3[:, None] * 4.0
        fit = hp.sklearn_cv_fit_fn(features3, labels3,
                                   model_type="logistic_regression",
                                   criterion="f1", n_jobs=1)
        assert fit({}, np.arange(0, 45), np.arange(45, 60)) == (
            1.0, {"n_train": 45, "n_val": 15})

    with subtests.test(case="margin-when-no-probabilities"):
        from sklearn.linear_model import RidgeClassifier

        built = []
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(hp, "build_sklearn_model",
                       lambda *a, **k: built.append((a, k)) or
                       RidgeClassifier())
            fit = hp.sklearn_cv_fit_fn(features, labels, criterion="roc_auc",
                                       seed=4, n_jobs=2)
            assert fit({"x": 1}, np.arange(0, 32), np.arange(32, 48)) == (
                1.0, {"n_train": 32, "n_val": 16})
        assert built == [(("xgboost", {"x": 1}), {"seed": 4, "n_jobs": 2})]

    with subtests.test(case="accuracy-of-a-wrong-model-is-exact"):
        flipped = 1 - labels
        fit = hp.sklearn_cv_fit_fn(features, np.concatenate(
            [labels[:32], flipped[32:]]), model_type="logistic_regression",
            n_jobs=1)
        assert fit({}, np.arange(0, 32), np.arange(32, 48))[0] == 0.0


# ===========================================================================
# classify_cv_fit_fn
# ===========================================================================

def test_classify_cv_fit_fn_scores_a_configuration_by_its_fold_mean(subtests):
    """The deep fit function layers the trial onto the settings, forces the
    fold count, keeps the grouping, and returns the fold mean with spread,
    min and max; a run with no results, a missing criterion column or no
    usable fold value fails the trial by message."""
    with subtests.test(case="fewer-than-two-folds"):
        with pytest.raises(ValueError) as info:
            hp.classify_cv_fit_fn({}, n_folds=1)
        assert str(info.value) == (
            "Classify (CV) search needs at least 2 folds to score a "
            "configuration on held-out data, got 1.")

    with subtests.test(case="mean-spread-and-forwarded-config"):
        seen = {}

        def _train(cfg):
            seen.update(cfg)
            return "/runs/fold_results.csv"

        read = []
        fit = hp.classify_cv_fit_fn(
            {"src": "/data", "cv_group_by": "plate"}, n_folds=3,
            train_fn=_train,
            read_fold_csv=lambda p: read.append(p) or _fold_csv(
                [0.6, 0.8, 0.7]))
        score, extra = fit({"learning_rate": 0.001})
        assert score == pytest.approx(0.7)
        assert extra == pytest.approx({"fold_std": 0.1, "n_folds": 3,
                                       "fold_min": 0.6, "fold_max": 0.8})
        assert seen == {"src": "/data", "cv_group_by": "plate",
                        "learning_rate": 0.001, "cross_validation_folds": 3}
        assert read == ["/runs/fold_results.csv"]

    with subtests.test(case="grouping-defaults-to-well"):
        seen = {}
        fit = hp.classify_cv_fit_fn(
            {}, n_folds=2, train_fn=lambda cfg: seen.update(cfg) or "/f.csv",
            read_fold_csv=lambda p: _fold_csv([0.5, 0.5]))
        assert fit({}) == (0.5, {"fold_std": 0.0, "n_folds": 2,
                                 "fold_min": 0.5, "fold_max": 0.5})
        assert seen == {"cross_validation_folds": 2, "cv_group_by": "well"}

    for cid, values, expected in [
        ("single-fold-zero-spread", [0.9], (0.9, 1)),
        ("nan-fold-is-dropped", [0.9, float("nan")], (0.9, 1)),
    ]:
        with subtests.test(case=cid):
            fit = hp.classify_cv_fit_fn(
                {}, n_folds=2, train_fn=lambda cfg: "/f.csv",
                read_fold_csv=lambda p, v=values: _fold_csv(v))
            score, extra = fit({})
            assert (score, extra["n_folds"]) == expected
            assert extra["fold_std"] == 0.0

    for cid, kwargs, message in [
        ("no-per-fold-results", dict(train_fn=lambda cfg: ""),
         "the training run produced no per-fold results, so this "
         "configuration cannot be scored (every fold may have died)"),
        ("criterion-not-reported", dict(
            criterion="prauc", train_fn=lambda cfg: "/f.csv",
            read_fold_csv=lambda p: _fold_csv([0.9, 0.8])),
         "the per-fold results have no 'prauc' column (available: "
         "['fold', 'accuracy'])"),
        ("all-nan-folds", dict(
            train_fn=lambda cfg: "/f.csv",
            read_fold_csv=lambda p: _fold_csv([float("nan")] * 2)),
         "no fold reported a usable 'accuracy' value"),
    ]:
        with subtests.test(case=cid):
            fit = hp.classify_cv_fit_fn({}, n_folds=2, **kwargs)
            with pytest.raises(ValueError) as info:
                fit({})
            assert str(info.value) == message


# ===========================================================================
# load_search_data
# ===========================================================================

@pytest.fixture
def measured_plate(tmp_path):
    """Build a plate folder with the measurements database Measure writes."""

    def _build(n=48, columns=("c01", "c02"), annotation=None, name="plate01"):
        src = tmp_path / name
        (src / "measurements").mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        cols = [columns[i % len(columns)] for i in range(n)]
        rows = [f"r{1 + (i // 6) % 8:02d}" for i in range(n)]
        cell = pd.DataFrame({
            "object_label": list(range(1, n + 1)),
            "prcf": [f"plate1_{r}_{c}_1" for r, c in zip(rows, cols)],
            "plateID": ["plate1"] * n, "rowID": rows,
            "columnID": cols, "fieldID": ["1"] * n,
            "cell_area": rng.uniform(100, 400, n),
            "cell_perimeter": rng.uniform(30, 90, n),
            "cell_channel_0_mean_intensity": rng.uniform(100, 900, n),
            "cell_channel_1_mean_intensity": rng.uniform(100, 900, n),
        })
        if annotation is not None:
            cell["annotation"] = annotation
        png = pd.DataFrame({
            "cell_id": [f"o{i}" for i in range(1, n + 1)],
            "png_path": [f"/crops/plate1_{r}_{c}_1_{i}.png"
                         for i, (r, c) in enumerate(zip(rows, cols), start=1)],
            "plateID": ["plate1"] * n, "rowID": rows,
            "columnID": cols, "fieldID": ["1"] * n,
        })
        con = sqlite3.connect(str(src / "measurements" / "measurements.db"))
        try:
            if n:
                cell.to_sql("cell", con, index=False, if_exists="replace")
                png.to_sql("png_list", con, index=False, if_exists="replace")
            else:
                cell.astype({"cell_area": float}).to_sql(
                    "cell", con, index=False)
                png.to_sql("png_list", con, index=False)
        finally:
            con.close()
        return src

    return _build


def _settings(src, **extra):
    values = {"src": str(src), "tables": ["cell"],
              "positive_control_id": "c02", "negative_control_id": "c01"}
    values.update(extra)
    return values


def test_load_search_data_reads_the_matrix_and_builds_honest_labels(
        subtests, measured_plate):
    """No source, an empty database, a table with no label source and a
    one-class result are refused by message; UMAP gets features only, a
    supervised search gets 0/1 labels from the control columns (or the
    annotation column, which wins) with well groups, and a row limit
    subsamples with a note."""
    for src in (None, "", "path", "/path/to/src", "/path"):
        with subtests.test(case=f"no-source-{src!r}"):
            with pytest.raises(ValueError) as info:
                hp.load_search_data("umap", {"src": src})
            assert str(info.value) == (
                "No source folder is set. Point 'src' at an experiment "
                "directory containing measurements/measurements.db before "
                "searching.")

    with subtests.test(case="empty-database-says-run-measure"):
        src = measured_plate(n=0, name="empty")
        with pytest.raises(ValueError, match=r"\. Run Measure first so there "
                                             r"is a measurements table"):
            hp.load_search_data("umap", _settings(src))

    src = measured_plate()
    with subtests.test(case="umap-features-only"):
        data = hp.load_search_data("umap", _settings(src))
        assert data.features.shape[0] == 48
        assert data.features.dtype == np.float64
        assert data.labels is None and data.groups is None
        assert len(data.frame) == 48

    with subtests.test(case="controls-become-labels"):
        data = hp.load_search_data("ml_analyze", _settings(src))
        assert data.labels.tolist() == [i % 2 for i in range(48)]
        assert data.groups.tolist()[:2] == ["plate1_r01_c01",
                                            "plate1_r01_c02"]
        assert data.notes == [
            "Labels derived from controls in 'columnID': 'c01'=0, 'c02'=1 "
            "(48 control rows kept). A model that separates two control "
            "columns has also learned to recognise those columns' plate "
            "position."]

    with subtests.test(case="annotation-wins-over-controls"):
        annotated = measured_plate(
            annotation=[(i // 2) % 2 for i in range(48)], name="annotated")
        data = hp.load_search_data(
            "ml_analyze", _settings(annotated, annotation_column="annotation"))
        assert data.labels.tolist() == [(i // 2) % 2 for i in range(48)]
        assert data.notes == ["Labels taken from the 'annotation' annotation "
                              "column (48 of 48 rows are annotated)."]

    with subtests.test(case="row-limit-subsamples"):
        data = hp.load_search_data("umap", _settings(src, row_limit=20))
        assert data.features.shape[0] == 20
        assert data.notes == [
            "Sub-sampled to 20 of the available rows (row_limit); a search on "
            "a subsample can rank configurations differently from the full "
            "run."]

    with subtests.test(case="no-label-source"):
        with pytest.raises(ValueError) as info:
            hp.load_search_data("ml_analyze",
                                _settings(src, location_column="treatment"))
        assert str(info.value) == (
            "Cannot build labels: neither an annotation column nor the "
            "location column 'treatment' is present in the measurement "
            "table.")

    with subtests.test(case="one-class-survives"):
        single = measured_plate(columns=("c01",), name="single")
        with pytest.raises(ValueError, match="^Only one class survived label "
                                             "construction"):
            hp.load_search_data("ml_analyze", _settings(single))


# ===========================================================================
# run_search_for_app
# ===========================================================================

def test_run_search_for_app_routes_each_app_to_its_search(subtests,
                                                          measured_plate,
                                                          monkeypatch):
    """Unknown apps, modes and criteria are refused by message; classical
    ML cross-validates on the loaded matrix (grid or random), Classify and
    the merged screen's torch family train one CV run per configuration,
    UMAP carries the data notes above its own, and Activation loads its own
    images and forwards its attribution settings."""
    for cid, app, kwargs, message in [
        ("app-without-search", "annotate", {},
         "No hyperparameter search is defined for app 'annotate'. Searchable "
         "apps: "),
        ("third-mode", "ml_analyze", {"mode": "bayesian"},
         "mode must be 'grid' or 'random', got 'bayesian'."),
        ("criterion-not-offered", "ml_analyze", {"criterion": "silhouette"},
         "Criterion 'silhouette' is not available for 'ml_analyze'; choose "
         "one of "),
    ]:
        with subtests.test(case=cid):
            with pytest.raises(ValueError) as info:
                hp.run_search_for_app(app, {}, SearchSpace({"a": [1]}),
                                      **kwargs)
            assert str(info.value).startswith(message)

    src = measured_plate()
    settings = _settings(src, model_type_ml="logistic_regression", n_jobs=1)
    with subtests.test(case="classical-ml-grid"):
        data = hp.load_search_data("ml_analyze", _settings(src))
        result = hp.run_search_for_app(
            "ml_analyze", settings, SearchSpace({"n_estimators": [5, 10]}),
            data=data, n_folds=2, seed=0)
        assert result.ok and result.metric == "accuracy"
        assert [t.params for t in result.trials] == [{"n_estimators": 5},
                                                     {"n_estimators": 10}]
        assert result.notes[0] == data.notes[0]
        assert result.notes[1].startswith(
            "Scored on 2 grouped (group_by='well') cross-validation folds")

    with subtests.test(case="classical-ml-random-loads-its-own-matrix"):
        result = hp.run_search_for_app(
            "ml_analyze", settings,
            SearchSpace({"n_estimators": [5, 10, 20, 40]}), mode="random",
            n_trials=2, n_folds=2)
        assert len(result.trials) == 2
        assert result.notes[0].startswith("Labels derived from controls")

    for cid, app, extra, mode, expected_n in [
        ("classify-grid", "classify", {"src": "/data"}, "grid", 2),
        ("merged-torch-random", "classify_merged",
         {"classifier_family": "torch"}, "random", 1),
    ]:
        with subtests.test(case=cid):
            import spacr.deep_spacr as deep

            trained = []
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(deep, "train_test_model",
                           lambda cfg, out=trained: out.append(dict(cfg)) or "/f.csv")
                mp.setattr("pandas.read_csv",
                           lambda path, *a, **k: _fold_csv([0.7, 0.9]))
                result = hp.run_search_for_app(
                    app, extra, SearchSpace({"learning_rate": [1e-4, 1e-3]}),
                    mode=mode, n_trials=1, n_folds=2)
            assert len(result.trials) == expected_n
            assert result.best.score == pytest.approx(0.8)
            assert all(cfg["cross_validation_folds"] == 2 for cfg in trained)
            assert len(trained) == expected_n
            assert (f"Cost: {expected_n} configurations × 2 folds models "
                    f"trained.") in result.notes

    with subtests.test(case="umap-carries-the-data-notes"):
        seen = {}

        def _fake_umap_search(features, searched, **kwargs):
            seen.update(kwargs)
            return hp.SearchResult(notes=["ranking note"],
                                   metric="trustworthiness")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(hp, "umap_search", _fake_umap_search)
            data = hp.SearchData(features=np.zeros((4, 2)),
                                 notes=["matrix note"])
            result = hp.run_search_for_app(
                "umap", {"src": "/data"}, SearchSpace({"n_neighbors": [5]}),
                criterion="trustworthiness", data=data,
                checkpoint_path="/tmp/x.json", umap_components=3)
        assert result.notes == ["matrix note", "ranking note"]
        assert (seen["metric"], seen["n_components"], seen["checkpoint_path"],
                seen["backend"]) == ("trustworthiness", 3, "/tmp/x.json",
                                     "cpu")

    with subtests.test(case="activation-loads-its-own-data"):
        seen = {}

        def _search(data, space, **kwargs):
            seen["data"], seen["kwargs"] = data, kwargs
            return hp.SearchResult(metric=kwargs["criterion"])

        monkeypatch.setattr(hp, "load_activation_data",
                            lambda settings: "activation-data")
        monkeypatch.setattr(hp, "activation_search", _search)
        result = hp.run_search_for_app(
            "activation", {"attribution_steps": 4,
                           "attribution_baseline": "zero",
                           "sanity_check": False},
            SearchSpace({"cam_type": ["gradcam"]}))
        assert result.metric == "deletion_auc"
        assert seen["data"] == "activation-data"
        assert {k: seen["kwargs"][k] for k in (
            "criterion", "mode", "n_steps", "baseline",
            "run_sanity_check")} == {
            "criterion": "deletion_auc", "mode": "grid", "n_steps": 4,
            "baseline": "zero", "run_sanity_check": False}
