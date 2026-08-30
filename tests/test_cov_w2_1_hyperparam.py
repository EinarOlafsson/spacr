"""Cross-validated search, the estimators it builds, and the report it prints.

The pieces here are the ones a hyperparameter sweep is actually made of: the
fold builder that keeps a well on one side of a split, the leak check that
refuses folds touching the test indices, the estimator ladder the classical-ML
sweep constructs, and the text report a user reads afterwards. Everything is
driven with real numeric data and real scikit-learn estimators -- a fold that
straddles a well is the defect this module exists to prevent, and only a real
split can show it did not happen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import hyperparam as hp
from spacr.hyperparam import SearchSpace


# ---------------------------------------------------------------------------
# Folds


def _labels(n=60):
    """Alternating binary labels, enough of each class for five folds."""
    return np.array([i % 2 for i in range(n)])


def _wells(n=60, per_well=6):
    """One well id per block of rows, the way a plate is laid out."""
    return np.array([f"p1_A_{i // per_well}" for i in range(n)])


def test_one_fold_cannot_cross_validate_anything():
    """With one fold there is no held-out data to score on."""
    with pytest.raises(ValueError, match="at least 2 to cross-validate"):
        hp.build_folds(_labels(), 1, groups=_wells())


def test_a_test_index_outside_the_data_is_refused():
    """An index that does not exist would silently exclude nothing."""
    with pytest.raises(ValueError, match=r"must lie inside \[0, 60\)"):
        hp.build_folds(_labels(), 3, groups=_wells(), exclude=[0, 99])


def test_excluding_everything_leaves_nothing_to_search_on():
    """A test split that swallowed the data is a configuration error."""
    with pytest.raises(ValueError, match="Every sample was excluded"):
        hp.build_folds(_labels(), 3, groups=_wells(), exclude=range(60))


def test_no_group_ids_is_refused_rather_than_guessed():
    """A random fallback would report an optimistic score and say nothing."""
    with pytest.raises(ValueError, match="No group ids were available"):
        hp.build_folds(_labels(), 3, group_by="well")


def test_a_well_never_straddles_a_fold():
    """Crops from one well share focus and seeding; splitting them inflates."""
    labels, wells = _labels(), _wells()

    folds, warnings = hp.build_folds(labels, 3, groups=wells)

    assert warnings == []
    assert len(folds) == 3
    for train_idx, val_idx in folds:
        assert not (set(wells[train_idx]) & set(wells[val_idx]))
        assert not (set(train_idx.tolist()) & set(val_idx.tolist()))


def test_the_test_split_is_in_no_fold_at_all():
    """The whole point of excluding it is that no configuration sees it."""
    labels, wells = _labels(), _wells()
    held_out = list(range(0, 12))

    folds, _warnings = hp.build_folds(labels, 3, groups=wells,
                                      exclude=held_out)

    for train_idx, val_idx in folds:
        assert not set(held_out) & set(train_idx.tolist())
        assert not set(held_out) & set(val_idx.tolist())


def test_splitting_by_cell_says_it_inflates_the_score():
    """`group_by='cell'` is allowed, and it is never allowed to be silent."""
    folds, warnings = hp.build_folds(_labels(), 3, group_by="cell")

    assert len(folds) == 3
    assert any("straddle folds" in message for message in warnings)


def test_group_ids_can_be_parsed_from_crop_filenames():
    """The crops on disk carry their well in the name; use it."""
    names = [f"plate1_A0{1 + i // 20}_f{i % 3}_o{i}.png" for i in range(60)]

    folds, warnings = hp.build_folds(_labels(), 2, filenames=names,
                                     group_by="well")

    assert len(folds) == 2
    assert all(isinstance(message, str) for message in warnings)


def test_an_anonymous_crop_name_is_refused_not_turned_into_a_group():
    """A singleton pseudo-group would make a random split look leakage-safe."""
    names = [f"crop_{i}.png" for i in range(60)]

    with pytest.raises(ValueError, match="does not encode a well"):
        hp.build_folds(_labels(), 2, filenames=names, group_by="well")


def test_unparsed_filenames_are_reported_when_the_parser_reports_them(
        monkeypatch):
    """The count comes from the parser; whatever it reports must be surfaced."""
    import spacr.io as io_module

    monkeypatch.setattr(io_module, "_cv_group_ids",
                        lambda paths, level, verbose=True: (
                            [f"w{i // 20}" for i in range(len(paths))], 3))

    _folds, warnings = hp.build_folds(_labels(), 2,
                                      filenames=[f"c{i}.png" for i in range(60)],
                                      group_by="well")

    assert any("3 filenames did not carry a 'well' level" in message
               for message in warnings)


# ---------------------------------------------------------------------------
# cv_search


def _fold_fit(params, train_idx, val_idx):
    """A deterministic stand-in estimator: bigger depth scores higher."""
    return 0.5 + 0.01 * float(params["depth"]) + 0.001 * len(val_idx)


def test_a_grid_is_scored_on_every_fold_and_averaged():
    """The trial score is the mean, and the spread comes with it."""
    space = SearchSpace({"depth": [1, 2, 3]})

    result = hp.cv_search(_fold_fit, space, labels=_labels(),
                          groups=_wells(), n_folds=3)

    assert len(result.trials) == 3
    assert result.best.params["depth"] == 3
    assert result.best.extra_metrics["n_folds"] == 3
    assert len(result.best.extra_metrics["fold_scores"]) == 3
    assert "fold_std" in result.best.extra_metrics
    assert any("no configuration was selected using test data" in note
               for note in result.notes)


def test_the_notes_say_the_folds_were_grouped():
    """A reader must be able to tell a grouped search from an ungrouped one."""
    space = SearchSpace({"depth": [1]})

    grouped = hp.cv_search(_fold_fit, space, labels=_labels(),
                           groups=_wells(), n_folds=2)
    ungrouped = hp.cv_search(_fold_fit, space, labels=_labels(),
                             group_by="cell", n_folds=2)

    assert "grouped (group_by='well')" in grouped.notes[0]
    assert "ungrouped" in ungrouped.notes[0]


def test_folds_that_touch_the_test_split_are_refused(caplog):
    """Scoring a search on test data makes the final estimate meaningless."""
    space = SearchSpace({"depth": [1]})
    folds = [(np.arange(0, 30), np.arange(30, 60))]

    with pytest.raises(ValueError, match="held-out test indices"):
        hp.cv_search(_fold_fit, space, labels=_labels(), folds=folds,
                     test_idx=[5, 6, 7])


def test_pre_built_folds_that_avoid_the_test_split_are_accepted():
    """A caller who built their own honest folds is not made to rebuild them."""
    space = SearchSpace({"depth": [1, 2]})
    folds = [(np.arange(10, 35), np.arange(35, 60)),
             (np.arange(35, 60), np.arange(10, 35))]

    result = hp.cv_search(_fold_fit, space, labels=_labels(), folds=folds,
                          test_idx=range(10))

    assert result.ok
    assert "10 test samples were excluded" in result.notes[0]


def test_a_random_cv_search_samples_distinct_configurations():
    """Sampling with replacement would score the same setting twice."""
    space = SearchSpace({"depth": [1, 2, 3, 4], "width": [10, 20]})

    result = hp.cv_search(lambda p, tr, va: float(p["depth"] + p["width"]),
                          space, labels=_labels(), groups=_wells(),
                          n_folds=2, n_trials=5, seed=3)

    labelled = [trial.label() for trial in result.trials]
    assert len(labelled) == 5
    assert len(set(labelled)) == 5
    assert any("Random search: 5 of 8" in note for note in result.notes)


def test_a_random_cv_search_cannot_ask_for_more_than_exists():
    """Six configurations out of four is four, not four plus two repeats."""
    space = SearchSpace({"depth": [1, 2, 3, 4]})

    result = hp.cv_search(_fold_fit, space, labels=_labels(),
                          groups=_wells(), n_folds=2, n_trials=99, seed=1)

    assert len(result.trials) == 4


def test_a_fold_that_returns_no_score_fails_the_trial():
    """A mean over folds is meaningless if one fold did not report."""
    space = SearchSpace({"depth": [1]})

    result = hp.cv_search(lambda p, tr, va: None, space, labels=_labels(),
                          groups=_wells(), n_folds=2)

    assert not result.ok
    assert "every fold must be scored" in result.trials[0].error


def test_extra_fold_metrics_are_collected_per_fold():
    """Whatever the fit function reports per fold is kept per fold."""
    space = SearchSpace({"depth": [1]})

    def _fit(params, train_idx, val_idx):
        return 0.7, {"n_train": len(train_idx)}

    result = hp.cv_search(_fit, space, labels=_labels(), groups=_wells(),
                          n_folds=3)

    assert len(result.best.extra_metrics["fold_n_train"]) == 3


# ---------------------------------------------------------------------------
# The report


def _searched(metric="score", **kwargs):
    """A real completed sweep whose scores differ from one another."""
    space = SearchSpace({"depth": [1, 2, 3, 4]})
    return hp.grid_search(lambda p: 0.5 + 0.05 * p["depth"], space,
                          metric=metric, **kwargs)


def test_the_report_names_the_criterion_and_ranks_the_trials():
    """The table is the point, but the criterion has to be readable first."""
    text = hp.format_search(_searched())

    assert text.splitlines()[0].startswith(
        "Hyperparameter search — criterion: score")
    assert "rank" in text and "params" in text
    assert "Best: depth=4" in text
    assert "Spread over 4 successful trials" in text


def test_a_stopped_sweep_is_never_presented_as_a_finished_one():
    """A partial sweep says so in the header, not in a footnote."""
    calls = []

    def _stop():
        calls.append(1)
        return len(calls) > 2

    text = hp.format_search(_searched(should_stop=_stop))

    assert "[PARTIAL — stopped early, not a completed sweep]" in text


def test_a_umap_criterion_carries_its_caveat():
    """UMAP has no ground truth; the report may not imply otherwise."""
    text = hp.format_search(_searched(metric="trustworthiness"))

    assert hp.UMAP_CRITERIA["trustworthiness"] in text


def test_an_activation_criterion_carries_its_caveat():
    """Attribution has no ground truth either, and it says so."""
    text = hp.format_search(_searched(metric="insertion_auc"))

    assert hp.ACTIVATION_CRITERIA["insertion_auc"] in text


def test_a_sweep_with_no_trials_says_exactly_that():
    """An empty result must not print a table with no rows under it."""
    assert "No trials were run." in hp.format_search(hp.SearchResult())


def test_a_sweep_where_everything_failed_lists_the_failures():
    """The errors are the only useful output such a sweep has."""
    def _always_fails(params):
        raise RuntimeError(f"no model for depth {params['depth']}")

    result = hp.grid_search(_always_fails, SearchSpace({"depth": [1, 2]}))
    text = hp.format_search(result)

    assert "All 2 trials failed:" in text
    assert "no model for depth 1" in text


def test_failures_beside_successes_are_listed_after_the_table():
    """A sweep that half-worked must not look like one that worked."""
    def _sometimes(params):
        if params["depth"] == 2:
            raise RuntimeError("depth 2 diverged")
        return 0.5 + 0.05 * params["depth"]

    text = hp.format_search(hp.grid_search(_sometimes,
                                           SearchSpace({"depth": [1, 2, 3]})))

    assert "Failed trials (1):" in text
    assert "depth 2 diverged" in text


def test_a_long_ranking_is_truncated_and_says_how_much_it_hid():
    """Twenty rows is a report; two hundred is a wall."""
    space = SearchSpace({"depth": list(range(10))})
    result = hp.grid_search(lambda p: 0.01 * p["depth"], space)

    text = hp.format_search(result, max_rows=3)

    assert "… 7 more" in text


def test_a_winner_inside_the_noise_is_called_arbitrary():
    """When the hyperparameter did not matter, the winner is not a finding."""
    space = SearchSpace({"depth": [1, 2, 3]})

    def _noisy(params, train_idx, val_idx):
        return 0.80 + 0.0001 * params["depth"] + 0.02 * (int(val_idx[0]) % 7)

    text = hp.format_search(hp.cv_search(_noisy, space, labels=_labels(),
                                         groups=_wells(), n_folds=3))

    assert "Noise yardstick:" in text
    assert "WITHIN NOISE" in text
    assert "Treat the winner as arbitrary." in text


def test_the_pareto_front_is_printed_when_objectives_are_declared():
    """Multi-objective UMAP has no single winner; the front is the answer."""
    space = SearchSpace({"depth": [1, 2, 3]})

    def _two_objectives(params):
        depth = params["depth"]
        return 0.5, {"stability": 0.1 * depth, "cluster_structure": 0.9 - 0.1 * depth}

    result = hp.grid_search(_two_objectives, space)
    result.objectives = {"stability": True, "cluster_structure": True}

    text = hp.format_search(result)

    assert "Pareto front (3 non-dominated configuration(s)):" in text
    assert "stability=0.1000" in text


# ---------------------------------------------------------------------------
# The estimator ladder


@pytest.mark.parametrize("model_type, expected, optional_package", [
    ("random_forest", "RandomForestClassifier", None),
    ("extra_trees", "ExtraTreesClassifier", None),
    ("logistic_regression", "LogisticRegression", None),
    ("gradient_boosting", "HistGradientBoostingClassifier", None),
    ("xgboost", "XGBClassifier", None),
    ("lightgbm", "LGBMClassifier", "lightgbm"),
    ("catboost", "CatBoostClassifier", "catboost"),
    ("svm", "CalibratedClassifierCV", None),
    ("mlp", "MLPClassifier", None),
])
def test_every_offered_model_type_builds_its_estimator(
        model_type, expected, optional_package):
    """Each offered backend constructs or names its optional installation."""
    try:
        model = hp.build_sklearn_model(
            model_type,
            {"n_estimators": 7, "learning_rate": 0.2,
             "reg_lambda": 2.0, "max_depth": 3},
            seed=1,
        )
    except ImportError as exc:
        if optional_package is None:
            raise
        assert f"pip install {optional_package}" in str(exc)
        return

    assert type(model).__name__ == expected


def test_an_unsupported_model_type_lists_the_ones_that_exist():
    """A typo in a settings file must name the alternatives."""
    with pytest.raises(ValueError, match="random_forest, extra_trees"):
        hp.build_sklearn_model("randomforest", {})


def test_the_regularisation_strength_is_the_inverse_of_reg_lambda():
    """`C` and `reg_lambda` pull in opposite directions; the mapping is fixed."""
    model = hp.build_sklearn_model("logistic_regression", {"reg_lambda": 4.0})

    assert model.C == pytest.approx(0.25)


def test_a_zero_reg_lambda_does_not_divide_by_zero():
    """An unregularised request is a huge C, not a ZeroDivisionError."""
    model = hp.build_sklearn_model(
        "logistic_regression", {"reg_lambda": 0.0})
    assert model.C > 1e8


# ---------------------------------------------------------------------------
# The classical-ML fit function


def _separable(n=48, seed=0):
    """A two-class problem an estimator can actually learn."""
    rng = np.random.default_rng(seed)
    labels = np.array([i % 2 for i in range(n)])
    features = rng.normal(size=(n, 4)) + labels[:, None] * 3.0
    return features, labels


@pytest.mark.parametrize("criterion", ["accuracy", "roc_auc", "f1"])
def test_a_fold_is_fitted_on_train_and_scored_on_validation(criterion):
    """The fit function is handed two index arrays and never any others."""
    features, labels = _separable()
    fit = hp.sklearn_cv_fit_fn(features, labels, model_type="logistic_regression",
                               criterion=criterion, n_jobs=1)

    score, extra = fit({"n_estimators": 10}, np.arange(0, 32), np.arange(32, 48))

    assert 0.0 <= score <= 1.0
    assert extra == {"n_train": 32, "n_val": 16}


def test_three_classes_are_scored_with_a_macro_f1():
    """A binary F1 on three classes would raise, not degrade."""
    rng = np.random.default_rng(1)
    labels = np.array([i % 3 for i in range(60)])
    features = rng.normal(size=(60, 3)) + labels[:, None] * 4.0
    fit = hp.sklearn_cv_fit_fn(features, labels,
                               model_type="logistic_regression",
                               criterion="f1", n_jobs=1)

    score, _extra = fit({}, np.arange(0, 45), np.arange(45, 60))

    assert score > 0.5


def test_an_estimator_without_probabilities_is_scored_on_its_margin(
        monkeypatch):
    """`roc_auc` needs a ranking, and a decision function is one."""
    from sklearn.linear_model import RidgeClassifier

    features, labels = _separable()
    assert not hasattr(RidgeClassifier(), "predict_proba")
    monkeypatch.setattr(hp, "build_sklearn_model",
                        lambda *a, **k: RidgeClassifier())
    fit = hp.sklearn_cv_fit_fn(features, labels, criterion="roc_auc")

    score, _extra = fit({}, np.arange(0, 32), np.arange(32, 48))

    assert 0.0 <= score <= 1.0


def test_the_searched_svm_is_the_svm_the_real_run_fits():
    """The search must configure the estimator ``ml_analysis`` will fit.

    ``spacr.ml`` builds an SVM as ``CalibratedClassifierCV(SVC(...))`` because
    scikit-learn 1.9 deprecated ``SVC(probability=True)`` and removes it in
    1.11. A search that still builds the deprecated form ranks a different
    estimator from the one the run uses, and stops working at 1.11.
    """
    import warnings

    features, labels = _separable()
    model = hp.build_sklearn_model("svm", {})

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model.fit(features, labels)

    assert hasattr(model, "predict_proba")


@pytest.mark.parametrize("model_type, package", [
    ("xgboost", "xgboost"),
    ("lightgbm", "lightgbm"),
    ("catboost", "catboost"),
])
def test_a_missing_optional_backend_names_the_package_to_install(
        monkeypatch, model_type, package):
    """The three gradient-boosting backends are optional; say which is absent."""
    import sys

    monkeypatch.setitem(sys.modules, package, None)

    with pytest.raises(ImportError, match=f"pip install {package}"):
        hp.build_sklearn_model(model_type, {})


# ---------------------------------------------------------------------------
# The deep cross-validated fit function


def _fold_csv(values, column="accuracy"):
    """A per-fold results frame of the shape a CV training run writes."""
    return pd.DataFrame({"fold": range(1, len(values) + 1), column: values})


def test_a_deep_search_needs_at_least_two_folds():
    """One fold scores a configuration on data it was chosen with."""
    with pytest.raises(ValueError, match="at least 2 folds"):
        hp.classify_cv_fit_fn({}, n_folds=1)


def test_a_configuration_is_scored_by_the_mean_over_its_folds():
    """The trial score is the mean, and the spread comes with it."""
    seen = {}

    def _train(cfg):
        seen.update(cfg)
        return "/runs/fold_results.csv"

    fit = hp.classify_cv_fit_fn({"src": "/data", "cv_group_by": "well"},
                                n_folds=3,
                                train_fn=_train,
                                read_fold_csv=lambda p: _fold_csv(
                                    [0.6, 0.8, 0.7]))

    score, extra = fit({"learning_rate": 0.001})

    assert score == pytest.approx(0.7)
    assert extra["n_folds"] == 3
    assert extra["fold_min"] == pytest.approx(0.6)
    assert extra["fold_max"] == pytest.approx(0.8)
    assert extra["fold_std"] > 0
    assert seen["cross_validation_folds"] == 3
    assert seen["cv_group_by"] == "well"
    assert seen["learning_rate"] == 0.001


def test_a_single_fold_result_reports_a_zero_spread_not_a_nan():
    """A NaN standard deviation would poison the noise yardstick."""
    fit = hp.classify_cv_fit_fn({}, n_folds=2, train_fn=lambda cfg: "/f.csv",
                                read_fold_csv=lambda p: _fold_csv([0.9]))

    _score, extra = fit({})

    assert extra["fold_std"] == 0.0


def test_a_training_run_that_produced_nothing_cannot_be_scored():
    """Every fold may have died; that is a failed trial, not a zero."""
    fit = hp.classify_cv_fit_fn({}, n_folds=2, train_fn=lambda cfg: "")

    with pytest.raises(ValueError, match="no per-fold results"):
        fit({})


def test_a_criterion_the_folds_never_reported_lists_what_they_did():
    """Naming the available columns is the difference from a KeyError."""
    fit = hp.classify_cv_fit_fn({}, criterion="prauc", n_folds=2,
                                train_fn=lambda cfg: "/f.csv",
                                read_fold_csv=lambda p: _fold_csv([0.9, 0.8]))

    with pytest.raises(ValueError, match=r"no 'prauc' column"):
        fit({})


def test_folds_that_reported_no_usable_number_fail_the_trial():
    """A column of NaNs is not a score, however many rows it has."""
    fit = hp.classify_cv_fit_fn({}, n_folds=2, train_fn=lambda cfg: "/f.csv",
                                read_fold_csv=lambda p: _fold_csv(
                                    [float("nan"), float("nan")]))

    with pytest.raises(ValueError, match="no fold reported a usable"):
        fit({})


# ---------------------------------------------------------------------------
# Wells, and the matrix a search runs on


def test_a_table_with_no_plate_columns_cannot_be_grouped_by_well():
    """The folds are then ungrouped, and the scores are optimistic."""
    groups, warning = hp._well_groups(pd.DataFrame({"cell_area": [1.0, 2.0]}))

    assert groups is None
    assert "the folds are ungrouped" in warning


def test_well_ids_are_plate_row_and_column_joined():
    """One id per well, so a well lands on one side of every split."""
    frame = pd.DataFrame({"plateID": ["p1", "p1"], "rowID": ["A", "A"],
                          "columnID": ["01", "02"]})

    groups, warning = hp._well_groups(frame)

    assert warning is None
    assert list(groups) == ["p1_A_01", "p1_A_02"]


@pytest.fixture
def measured_plate(tmp_path):
    """A real plate folder with a measurements database Measure could have written."""
    import sqlite3

    def _build(n=48, columns=("c01", "c02"), extra_cell=None,
               annotation=None):
        src = tmp_path / "plate01"
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
        if extra_cell is not None:
            cell = cell.drop(columns=list(extra_cell))
        png = pd.DataFrame({
            "cell_id": [f"o{i}" for i in range(1, n + 1)],
            "png_path": [f"/crops/plate1_{r}_{c}_1_{i}.png"
                         for i, (r, c) in enumerate(zip(rows, cols), start=1)],
            "plateID": ["plate1"] * n, "rowID": rows,
            "columnID": cols, "fieldID": ["1"] * n,
        })
        con = sqlite3.connect(str(src / "measurements" / "measurements.db"))
        try:
            cell.to_sql("cell", con, index=False, if_exists="replace")
            png.to_sql("png_list", con, index=False, if_exists="replace")
        finally:
            con.close()
        return src

    return _build


def _settings(src, **extra):
    values = {"src": str(src), "tables": ["cell"],
              "positive_control": "c02", "negative_control": "c01"}
    values.update(extra)
    return values


def test_a_search_with_no_source_folder_says_what_to_point_it_at():
    """'path' is the placeholder the settings panel ships with."""
    for src in (None, "", "path", "/path/to/src"):
        with pytest.raises(ValueError, match="No source folder is set"):
            hp.load_search_data("umap", {"src": src})


def test_an_unsupervised_search_gets_features_and_no_labels(measured_plate):
    """UMAP needs the matrix and nothing else."""
    data = hp.load_search_data("umap", _settings(measured_plate()))

    assert data.features.shape[0] == 48
    assert data.features.shape[1] >= 2
    assert data.labels is None
    assert data.groups is None


def test_a_supervised_search_labels_the_two_control_columns(measured_plate):
    """And says so, because separating controls also separates plate position."""
    data = hp.load_search_data("ml_analyze", _settings(measured_plate()))

    assert sorted(set(data.labels.tolist())) == [0, 1]
    assert len(data.groups) == len(data.labels)
    assert any("Labels derived from controls" in note for note in data.notes)


def test_an_annotation_column_wins_over_the_controls(measured_plate):
    """A user who annotated is not overruled by the control columns."""
    src = measured_plate(annotation=[i % 2 for i in range(48)])

    data = hp.load_search_data(
        "ml_analyze", _settings(src, annotation_column="annotation"))

    assert any("Labels taken from the 'annotation' annotation column"
               in note for note in data.notes)
    assert sorted(set(data.labels.tolist())) == [0, 1]


def test_a_row_limit_subsamples_and_says_the_ranking_may_move(measured_plate):
    """A search on a subsample can rank configurations differently."""
    data = hp.load_search_data("umap",
                               _settings(measured_plate(), row_limit=20))

    assert data.features.shape[0] == 20
    assert any("Sub-sampled to 20" in note for note in data.notes)


def test_a_table_with_no_labels_at_all_is_refused(measured_plate):
    """Neither an annotation column nor a location column is fatal."""
    src = measured_plate()

    with pytest.raises(ValueError, match="Cannot build labels"):
        hp.load_search_data("ml_analyze",
                            _settings(src, location_column="treatment"))


def test_one_surviving_class_is_nothing_to_classify(measured_plate):
    """Every row a negative control leaves no contrast to learn."""
    src = measured_plate(columns=("c01",))

    with pytest.raises(ValueError, match="Only one class survived"):
        hp.load_search_data("ml_analyze", _settings(src))


def test_a_database_with_no_rows_says_to_run_measure_first(tmp_path):
    """An empty measurements table is a missing step, not an empty search."""
    import sqlite3

    src = tmp_path / "empty"
    (src / "measurements").mkdir(parents=True)
    empty = {name: pd.Series(dtype="object") for name in
             ("prcf", "plateID", "rowID", "columnID", "fieldID")}
    con = sqlite3.connect(str(src / "measurements" / "measurements.db"))
    try:
        pd.DataFrame({"object_label": pd.Series(dtype="int64"), **empty,
                      "cell_area": pd.Series(dtype="float64")}).to_sql(
            "cell", con, index=False)
        pd.DataFrame({"cell_id": pd.Series(dtype="object"),
                      "png_path": pd.Series(dtype="object"),
                      "plateID": empty["plateID"], "rowID": empty["rowID"],
                      "columnID": empty["columnID"],
                      "fieldID": empty["fieldID"]}).to_sql(
            "png_list", con, index=False)
    finally:
        con.close()

    with pytest.raises(ValueError, match="Run Measure"):
        hp.load_search_data("umap", _settings(src))


# ---------------------------------------------------------------------------
# The entry point the GUI's Search button calls


def test_an_app_with_no_search_lists_the_ones_that_have_one():
    """A key that never had a sweep must name the keys that do."""
    with pytest.raises(ValueError, match="Searchable apps:"):
        hp.run_search_for_app("annotate", {}, SearchSpace({"a": [1]}))


def test_only_grid_and_random_are_modes():
    """A third mode would silently become one of the two."""
    with pytest.raises(ValueError, match="mode must be 'grid' or 'random'"):
        hp.run_search_for_app("ml_analyze", {}, SearchSpace({"a": [1]}),
                              mode="bayesian")


def test_a_criterion_the_app_does_not_offer_is_refused():
    """Ranking by a metric the app never computes would rank by nothing."""
    with pytest.raises(ValueError, match="is not available for 'ml_analyze'"):
        hp.run_search_for_app("ml_analyze", {}, SearchSpace({"a": [1]}),
                              criterion="silhouette")


def test_the_classical_ml_search_cross_validates_on_the_loaded_matrix(
        measured_plate):
    """The whole path: database to matrix to grouped folds to a ranking."""
    src = measured_plate()
    data = hp.load_search_data("ml_analyze", _settings(src))
    space = SearchSpace({"n_estimators": [5, 10]})

    result = hp.run_search_for_app(
        "ml_analyze", _settings(src, model_type_ml="logistic_regression",
                                n_jobs=1),
        space, data=data, n_folds=2, seed=0)

    assert result.ok
    assert len(result.trials) == 2
    assert result.metric == "accuracy"
    assert any("cross-validation folds" in note for note in result.notes)


def test_a_random_classical_ml_search_samples_the_space(measured_plate):
    """`mode='random'` evaluates n_trials configurations, not the whole grid."""
    src = measured_plate()
    data = hp.load_search_data("ml_analyze", _settings(src))
    space = SearchSpace({"n_estimators": [5, 10, 20, 40]})

    result = hp.run_search_for_app(
        "ml_analyze", _settings(src, model_type_ml="logistic_regression",
                                n_jobs=1),
        space, data=data, mode="random", n_trials=2, n_folds=2)

    assert len(result.trials) == 2


def test_a_deep_search_trains_one_cross_validated_run_per_configuration(
        monkeypatch):
    """Each trial is g x n_folds models, and the notes say so."""
    import spacr.deep_spacr as deep

    monkeypatch.setattr(deep, "train_test_model",
                        lambda cfg: "/runs/fold_results.csv")
    monkeypatch.setattr("pandas.read_csv",
                        lambda path, *a, **k: _fold_csv([0.7, 0.9]))
    space = SearchSpace({"learning_rate": [1e-4, 1e-3]})

    result = hp.run_search_for_app("classify", {"src": "/data"}, space,
                                   n_folds=2)

    assert result.ok
    assert result.best.score == pytest.approx(0.8)
    assert any("2 configurations × 2 folds models trained" in note
               for note in result.notes)


def test_the_merged_screen_takes_the_deep_path_for_a_torch_family(monkeypatch):
    """`classifier_family` decides which half of the merged screen searched."""
    import spacr.deep_spacr as deep

    monkeypatch.setattr(deep, "train_test_model", lambda cfg: "/f.csv")
    monkeypatch.setattr("pandas.read_csv",
                        lambda path, *a, **k: _fold_csv([0.5, 0.5]))

    result = hp.run_search_for_app(
        "classify_merged", {"classifier_family": "torch"},
        SearchSpace({"learning_rate": [1e-3, 1e-4]}), mode="random",
        n_trials=1, n_folds=2)

    assert len(result.trials) == 1
    assert result.ok


def test_a_umap_search_carries_the_data_notes_into_its_result(monkeypatch):
    """A caveat about the matrix belongs above the ranking, not beside it."""
    space = SearchSpace({"n_neighbors": [5]})

    def _fake_umap_search(features, searched, **kwargs):
        assert kwargs["metric"] == "trustworthiness"
        return hp.SearchResult(notes=["ranking note"], metric="trustworthiness")

    monkeypatch.setattr(hp, "umap_search", _fake_umap_search)
    data = hp.SearchData(features=np.zeros((4, 2)), notes=["matrix note"])

    result = hp.run_search_for_app("umap", {"src": "/data"}, space,
                                   criterion="trustworthiness", data=data)

    assert result.notes == ["matrix note", "ranking note"]


def test_the_search_loads_the_matrix_itself_when_none_is_handed_to_it(
        measured_plate):
    """The GUI passes settings, not data; the database read happens here."""
    src = measured_plate()

    result = hp.run_search_for_app(
        "ml_analyze", _settings(src, model_type_ml="logistic_regression",
                                n_jobs=1),
        SearchSpace({"n_estimators": [5]}), n_folds=2)

    assert result.ok
    assert any("Labels derived from controls" in note for note in result.notes)


def test_an_activation_search_loads_its_own_images_and_model(monkeypatch):
    """The attribution sweep has its own loader; the app key routes to it."""
    seen = {}

    def _load(settings):
        seen["settings"] = dict(settings)
        return "activation-data"

    def _search(data, space, **kwargs):
        seen["data"] = data
        seen["kwargs"] = kwargs
        return hp.SearchResult(metric=kwargs["criterion"])

    monkeypatch.setattr(hp, "load_activation_data", _load)
    monkeypatch.setattr(hp, "activation_search", _search)

    result = hp.run_search_for_app(
        "activation", {"attribution_steps": 4, "attribution_baseline": "zero",
                       "sanity_check": False},
        SearchSpace({"cam_type": ["gradcam"]}))

    assert result.metric == "deletion_auc"
    assert seen["data"] == "activation-data"
    assert seen["kwargs"]["n_steps"] == 4
    assert seen["kwargs"]["baseline"] == "zero"
    assert seen["kwargs"]["run_sanity_check"] is False


# ---------------------------------------------------------------------------
# The space itself


@pytest.mark.parametrize("params, message", [
    (["n_neighbors", 5], "must be a mapping"),
    ({}, "Search space is empty"),
    ({"": [1]}, "non-empty strings"),
    ({"n_neighbors": 5}, "must be a list or tuple"),
    ({"n_neighbors": "5"}, "must be a list or tuple"),
    ({"n_neighbors": []}, "empty value list"),
])
def test_a_space_that_cannot_be_searched_says_what_is_wrong(params, message):
    """Every refusal shows the caller the shape it should have written."""
    with pytest.raises(ValueError, match=message):
        SearchSpace(params)


# ---------------------------------------------------------------------------
# Random search


def test_a_random_search_is_reproducible_from_its_seed():
    """A sweep rerun on Tuesday must evaluate Monday's configurations."""
    space = SearchSpace({"a": [1, 2, 3, 4], "b": [10, 20, 30]})

    first = hp.random_search(lambda p: p["a"] * p["b"], space, 4, seed=7)
    second = hp.random_search(lambda p: p["a"] * p["b"], space, 4, seed=7)

    assert [t.params for t in first.trials] == [t.params for t in second.trials]
    assert len(first.trials) == 4
    assert "Random search: 4 of 12 configurations" in first.notes[0]


def test_a_random_search_never_evaluates_one_configuration_twice():
    """Two identical rows in the table are one measurement, twice charged."""
    space = SearchSpace({"a": [1, 2, 3]})

    result = hp.random_search(lambda p: float(p["a"]), space, 3, seed=1)

    assert sorted(t.params["a"] for t in result.trials) == [1, 2, 3]


def test_asking_for_more_configurations_than_exist_is_an_exhaustive_grid():
    """And it says so, rather than reporting a random sample of everything."""
    space = SearchSpace({"a": [1, 2]})

    result = hp.random_search(lambda p: float(p["a"]), space, 10, seed=0)

    assert len(result.trials) == 2
    assert any("this is an exhaustive grid, not a random sample" in note
               for note in result.notes)


def test_duplicates_are_allowed_when_the_caller_asks_for_them():
    """Repeated draws are a legitimate way to measure run-to-run variation."""
    space = SearchSpace({"a": [1, 2]})

    result = hp.random_search(lambda p: float(p["a"]), space, 6, seed=0,
                              allow_duplicates=True)

    assert len(result.trials) == 6


@pytest.mark.parametrize("bad", [0, -3])
def test_a_search_with_no_trials_has_nothing_to_report(bad):
    """Zero trials is a configuration error, not an empty result."""
    with pytest.raises(ValueError, match="at least 1"):
        hp.random_search(lambda p: 1.0, SearchSpace({"a": [1]}), bad)


def test_a_trial_count_that_is_not_a_number_is_refused():
    """A string from a settings file must not become one trial by accident."""
    with pytest.raises(ValueError, match="must be a positive integer"):
        hp.random_search(lambda p: 1.0, SearchSpace({"a": [1]}), "many")


def test_the_rejection_sampler_falls_back_to_the_grid(monkeypatch):
    """A sampler that keeps drawing the same point still covers the space."""
    space = SearchSpace({"a": [1, 2, 3]})
    monkeypatch.setattr(SearchSpace, "sample", lambda self, rng: {"a": 1})

    result = hp.random_search(lambda p: float(p["a"]), space, 3, seed=0)

    assert sorted(t.params["a"] for t in result.trials) == [1, 2, 3]


# ---------------------------------------------------------------------------
# What a fit function is allowed to return


def test_a_fit_function_may_return_a_bare_number():
    """The simplest possible fit function is the documented one."""
    assert hp._normalise_outcome(0.5) == (0.5, {})


def test_a_fit_function_may_return_a_dict_with_a_score():
    """Everything else in the dict becomes an extra metric."""
    score, extra = hp._normalise_outcome({"score": 0.25, "auc": 0.9})

    assert score == 0.25
    assert extra == {"auc": 0.9}


def test_a_dict_without_a_score_says_what_to_return_instead():
    """A silent None would rank the trial as failed for no stated reason."""
    with pytest.raises(TypeError, match="without a 'score' key"):
        hp._normalise_outcome({"auc": 0.9})


def test_a_score_that_is_not_a_number_is_refused():
    """A string score would sort lexicographically and rank nonsense."""
    with pytest.raises(TypeError, match="which is not a number"):
        hp._normalise_outcome("excellent")


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_score_fails_the_trial_rather_than_winning_it(bad):
    """`inf` would win every comparison; `nan` would lose every one."""
    with pytest.raises(ValueError, match="non-finite score"):
        hp._normalise_outcome(bad)


def test_a_none_score_is_no_score_and_not_an_error():
    """The runner turns it into a recorded failure with its own message."""
    assert hp._normalise_outcome((None, {"why": "diverged"})) == (
        None, {"why": "diverged"})


# ---------------------------------------------------------------------------
# The shared runner


def test_progress_is_reported_for_failed_trials_too():
    """A progress bar that stalls on a failure looks like a hang."""
    seen = []

    def _sometimes(params):
        if params["a"] == 2:
            raise RuntimeError("diverged")
        return float(params["a"])

    result = hp.grid_search(_sometimes, SearchSpace({"a": [1, 2, 3]}),
                            on_trial=lambda t, done, total: seen.append(
                                (done, total, t.ok)))

    assert seen == [(1, 3, True), (2, 3, False), (3, 3, True)]
    assert result.trials[1].error == "RuntimeError: diverged"
    assert any("were recorded rather than dropped" in note
               for note in result.notes)


def test_a_cancelled_run_is_not_caught_and_recorded_as_a_failed_trial():
    """Cancellation is the user stopping the run, not a bad configuration."""
    from spacr.cancellation import PipelineCancelled

    def _cancelled(params):
        raise PipelineCancelled("stopped")

    with pytest.raises(PipelineCancelled):
        hp.grid_search(_cancelled, SearchSpace({"a": [1, 2]}))


def test_a_single_configuration_is_a_measurement_not_a_search():
    """Reporting a 'winner' out of one is the thing this note prevents."""
    result = hp.grid_search(lambda p: 0.5, SearchSpace({"a": [1]}))

    assert any("this is a single measurement, not a search" in note
               for note in result.notes)
    assert any("evaluates one setting" in note for note in result.notes)


def test_a_sweep_where_nothing_scored_says_there_is_no_winner():
    """An empty ranking must not be presented as a result."""
    def _fails(params):
        raise RuntimeError("no")

    result = hp.grid_search(_fails, SearchSpace({"a": [1, 2]}))

    assert result.best is None
    assert any("no winner to report" in note for note in result.notes)


def test_a_completed_trial_can_be_replayed_instead_of_refitted():
    """A resumed search must not pay for the trials it already ran."""
    space = SearchSpace({"a": [1, 2]})
    combos = space.grid()
    done = hp.Trial(params=dict(combos[0]), score=0.9, index=99)
    fitted = []

    result = hp._run_trials(lambda p: fitted.append(p) or 0.1, combos, space,
                            "score",
                            prior_trials={hp._trial_key(combos[0]): done})

    assert fitted == [combos[1]]
    assert result.best is done
    assert done.index == 0


# ---------------------------------------------------------------------------
# Result slices


def test_a_result_with_no_successful_trials_has_no_statistics():
    """None, not zero: zero is a score and this has none."""
    stats = hp.SearchResult().score_stats()

    assert stats == {"n": 0, "best": None, "worst": None, "mean": None,
                     "std": None, "spread": None}


def test_a_fold_spread_that_is_not_a_number_is_not_a_noise_yardstick():
    """A string in the metrics dict falls back to the across-trial spread."""
    trials = [hp.Trial(params={"a": 1}, score=0.9, index=0,
                       extra_metrics={"fold_std": "wide"}),
              hp.Trial(params={"a": 2}, score=0.1, index=1)]
    result = hp.SearchResult(trials=trials, best=trials[0])

    noise, source = result.noise_level()

    assert source == "standard deviation across trials"
    assert noise == pytest.approx(0.4)


def test_one_trial_alone_cannot_estimate_noise():
    """With nothing to compare against, the winner is simply the only one."""
    trial = hp.Trial(params={"a": 1}, score=0.9, index=0)
    result = hp.SearchResult(trials=[trial], best=trial)

    assert result.noise_level() == (
        None, "not enough successful trials to estimate noise")
    assert result.trials_within_noise() == [trial]
    assert result.within_noise() is False


def test_a_dominated_configuration_is_off_the_pareto_front():
    """Worse on every objective is not a trade-off, it is just worse."""
    trials = [
        hp.Trial(params={"a": 1}, score=0.9, index=0,
                 extra_metrics={"x": 0.9, "y": 0.9}),
        hp.Trial(params={"a": 2}, score=0.5, index=1,
                 extra_metrics={"x": 0.2, "y": 0.2}),
        hp.Trial(params={"a": 3}, score=0.4, index=2,
                 extra_metrics={"x": 0.1, "y": 1.0}),
    ]
    result = hp.SearchResult(trials=trials, best=trials[0],
                             objectives={"x": True, "y": True})

    front = result.pareto_front()

    assert [t.params["a"] for t in front] == [1, 3]


def test_a_trial_missing_an_objective_is_not_on_the_front():
    """An unscored objective cannot be compared, so it cannot dominate."""
    trials = [
        hp.Trial(params={"a": 1}, score=0.9, index=0, extra_metrics={"x": 0.9}),
        hp.Trial(params={"a": 2}, score=0.8, index=1,
                 extra_metrics={"x": float("nan")}),
        hp.Trial(params={"a": 3}, score=0.7, index=2, extra_metrics={}),
    ]
    result = hp.SearchResult(trials=trials, best=trials[0],
                             objectives={"x": True})

    assert [t.params["a"] for t in result.pareto_front()] == [1]


def test_a_lower_is_better_objective_prefers_the_smaller_value():
    """Deletion AUC is better when it is small; the front has to know."""
    trials = [
        hp.Trial(params={"a": 1}, score=0.1, index=0, extra_metrics={"d": 0.1}),
        hp.Trial(params={"a": 2}, score=0.8, index=1, extra_metrics={"d": 0.8}),
    ]
    result = hp.SearchResult(trials=trials, best=trials[0],
                             higher_is_better=False,
                             objectives={"d": False})

    assert [t.params["a"] for t in result.pareto_front()] == [1]


def test_every_trial_becomes_one_table_row_successes_first():
    """The GUI table is built from this; a failure still needs its row."""
    def _sometimes(params):
        if params["a"] == 2:
            raise RuntimeError("diverged")
        return float(params["a"])

    rows = hp.grid_search(_sometimes, SearchSpace({"a": [1, 2, 3]}),
                          metric="accuracy").as_rows()

    assert [row["rank"] for row in rows] == [1, 2, None]
    assert [row["params"]["a"] for row in rows] == [3, 1, 2]
    assert rows[2]["score"] is None
    assert rows[2]["error"] == "RuntimeError: diverged"
    assert all(row["metric"] == "accuracy" for row in rows)
    assert not any(row["pareto"] for row in rows)


# ---------------------------------------------------------------------------
# Where a UMAP search checkpoints itself


def test_an_explicit_checkpoint_path_is_used_as_given(tmp_path):
    """Only an explicit path may create a project tree that is not there."""
    wanted = tmp_path / "elsewhere" / "cp.json"

    assert hp.umap_checkpoint_path({"checkpoint_path": str(wanted)}) == \
        str(wanted)


def test_a_search_with_no_source_has_no_checkpoint():
    """Nothing to infer a project from is None, not a path under the cwd."""
    assert hp.umap_checkpoint_path({}) is None
    assert hp.umap_checkpoint_path({"src": []}) is None


def test_a_source_that_does_not_exist_has_no_checkpoint(tmp_path):
    """A placeholder 'src' must not create a checkpoint tree."""
    assert hp.umap_checkpoint_path({"src": str(tmp_path / "absent")}) is None


def test_a_database_source_checkpoints_under_its_project(tmp_path):
    """The path is the project's, however deep into it the user pointed."""
    project = tmp_path / "plate01"
    (project / "measurements").mkdir(parents=True)
    db = project / "measurements" / "measurements.db"
    db.write_bytes(b"")
    expected = str(project / "results" / ".spacr_checkpoints" /
                   "umap_search.json")

    assert hp.umap_checkpoint_path({"src": [None, str(db)]}) == expected
    assert hp.umap_checkpoint_path(
        {"src": str(project / "measurements")}) == expected


# ---------------------------------------------------------------------------
# Persisting a UMAP search


def test_two_equal_arrays_have_one_fingerprint():
    """The signature decides whether a checkpoint may be resumed at all."""
    a = np.arange(12, dtype=float).reshape(3, 4)

    assert hp._array_fingerprint(a) == hp._array_fingerprint(a.copy())
    assert hp._array_fingerprint(a) != hp._array_fingerprint(a + 1)
    assert hp._array_fingerprint(a) != hp._array_fingerprint(
        a.astype(np.float32))


def test_an_object_array_is_fingerprinted_through_its_values():
    """A matrix of strings has no buffer to digest, and still needs an id."""
    words = np.array([["a", "b"], ["c", "d"]], dtype=object)

    assert hp._array_fingerprint(words) == hp._array_fingerprint(words.copy())
    assert hp._array_fingerprint(words) != hp._array_fingerprint(
        np.array([["a", "b"], ["c", "e"]], dtype=object))


def test_an_embedding_is_written_whole_or_not_at_all(tmp_path):
    """A half-written .npy would be loaded on resume as a corrupt embedding."""
    target = tmp_path / "deep" / "embedding.npy"
    array = np.arange(6, dtype=float).reshape(3, 2)

    hp._save_array_atomic(target, array)

    assert np.array_equal(np.load(target), array)
    assert [p.name for p in target.parent.iterdir()] == ["embedding.npy"]


def test_a_failed_write_leaves_no_temporary_behind(tmp_path, monkeypatch):
    """A directory of .tmp files is what "atomic" is supposed to prevent."""
    target = tmp_path / "embedding.npy"

    def _explode(stream, arr, allow_pickle=False):
        raise OSError("the disk filled up")

    monkeypatch.setattr(np, "save", _explode)

    with pytest.raises(OSError):
        hp._save_array_atomic(target, np.zeros(3))

    assert list(tmp_path.iterdir()) == []


def test_a_temporary_that_cannot_be_removed_does_not_hide_the_real_error(
        tmp_path, monkeypatch):
    """The write failure is the news; a failed cleanup must not replace it."""
    def _explode(stream, arr, allow_pickle=False):
        raise OSError("the disk filled up")

    def _cannot_unlink(path):
        raise OSError("permission denied")

    monkeypatch.setattr(np, "save", _explode)
    monkeypatch.setattr("os.unlink", _cannot_unlink)

    with pytest.raises(OSError, match="the disk filled up"):
        hp._save_array_atomic(tmp_path / "embedding.npy", np.zeros(3))


# ---------------------------------------------------------------------------
# Resuming a UMAP search from its checkpoint


def _checkpoint(tmp_path, *, resume=False, keep_embeddings=True,
                signature="matrix-a"):
    return hp._UmapCheckpoint(str(tmp_path / "umap_search.json"),
                              {"features": signature}, resume,
                              keep_embeddings)


def test_a_recorded_trial_comes_back_with_its_embedding(tmp_path):
    """The embedding is the expensive part; a resume that lost it is no resume."""
    store = _checkpoint(tmp_path)
    embedding = np.arange(8, dtype=float).reshape(4, 2)
    trial = hp.Trial(params={"n_neighbors": 5}, score=0.8, index=0,
                     duration=1.5,
                     extra_metrics={"embedding": embedding,
                                    "trustworthiness": 0.8})
    store.record(trial, round_index=2, state={"round": 2})
    store.finish({"round": 2})

    reopened = _checkpoint(tmp_path, resume=True)
    loaded = reopened.load()

    assert reopened.resumed
    assert reopened.state == {"round": 2}
    (restored, round_index), = loaded.values()
    assert round_index == 2
    assert restored.params == {"n_neighbors": 5}
    assert restored.score == 0.8
    assert np.array_equal(restored.extra_metrics["embedding"], embedding)
    assert restored.extra_metrics["trustworthiness"] == 0.8


def test_a_trial_whose_embedding_file_vanished_is_recomputed(tmp_path):
    """A missing artifact means the trial is not usable, not that it is empty."""
    store = _checkpoint(tmp_path)
    store.record(hp.Trial(params={"n_neighbors": 5}, score=0.8, index=0,
                          extra_metrics={"embedding": np.zeros((3, 2))}))
    store.finish()
    for artifact in (tmp_path / "umap_search.json.d").iterdir():
        artifact.unlink()

    assert _checkpoint(tmp_path, resume=True).load() == {}


def test_a_scored_trial_with_no_embedding_is_recomputed_when_one_is_wanted(
        tmp_path):
    """A search that keeps embeddings cannot use a trial that has none."""
    store = _checkpoint(tmp_path)
    store.record(hp.Trial(params={"n_neighbors": 5}, score=0.8, index=0))
    store.finish()

    assert _checkpoint(tmp_path, resume=True).load() == {}


def test_a_failed_trial_is_remembered_without_an_embedding(tmp_path):
    """It failed; there is nothing to recompute and nothing to store."""
    store = _checkpoint(tmp_path)
    store.record(hp.Trial(params={"n_neighbors": 5}, index=0,
                          error="RuntimeError: no"))
    store.update({"round": 1})
    store.finish()

    (restored, _round), = _checkpoint(tmp_path, resume=True).load().values()

    assert restored.error == "RuntimeError: no"
    assert restored.score is None


def test_a_checkpoint_for_a_different_matrix_is_not_resumed(tmp_path):
    """Resuming onto other data would rank configurations on two datasets."""
    from spacr.checkpoint import CheckpointMismatch

    store = _checkpoint(tmp_path, signature="matrix-a")
    store.record(hp.Trial(params={"n_neighbors": 5}, score=0.8, index=0))
    store.finish()

    with pytest.raises(CheckpointMismatch):
        _checkpoint(tmp_path, resume=True, signature="matrix-b")


# ---------------------------------------------------------------------------
# What the settings panel may offer


def test_the_metric_list_comes_from_the_installed_umap():
    """A metric the installed umap rejects would fail deep inside the run."""
    metrics = hp.umap_metrics()

    assert "euclidean" in metrics
    assert "cosine" in metrics
    assert metrics == tuple(sorted(metrics))


def test_the_panel_still_offers_metrics_without_umap_installed(monkeypatch):
    """A user configuring a run on a laptop and running it elsewhere."""
    import builtins

    real_import = builtins.__import__

    def _no_umap(name, *args, **kwargs):
        if name.startswith("umap"):
            raise ImportError("No module named 'umap'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_umap)

    assert hp.umap_metrics() == hp.UMAP_METRICS


def test_umap_reports_itself_available_when_it_imports():
    """The panel greys the Run button on this answer."""
    available, message = hp.umap_available()

    assert available is True
    assert message == ""


def test_an_unimportable_umap_is_reported_with_the_install_hint(monkeypatch):
    """The message is what the user is shown instead of a traceback."""
    import spacr.utils as utils

    class _Absent:
        def __getattr__(self, name):
            raise ImportError("No module named 'umap'")

    monkeypatch.setattr(utils, "umap", _Absent())

    assert hp.umap_available() == (False, hp.UMAP_MISSING_MESSAGE)


def test_an_incompatible_umap_reports_the_compatibility_problem(monkeypatch):
    """A version clash is a different fix from a missing package."""
    import spacr.utils as utils
    from spacr.utils import OptionalDependencyCompatibilityError

    class _Clashing:
        def __getattr__(self, name):
            raise OptionalDependencyCompatibilityError(
                "umap-learn needs numba < 0.62")

    monkeypatch.setattr(utils, "umap", _Clashing())

    available, message = hp.umap_available()

    assert available is False
    assert "numba" in message


# ---------------------------------------------------------------------------
# The attribution sweep's guards


def test_an_attribution_criterion_that_does_not_exist_lists_the_ones_that_do():
    """Each measures a different property, so the name has to be exact."""
    data = hp.ActivationSearchData(model=object(), images=[object()])

    with pytest.raises(ValueError, match="Unknown Activation criterion"):
        hp.activation_fit_fn(data, criterion="iou")


def test_an_attribution_sweep_with_no_images_says_where_to_point_it():
    """Nothing to attribute is a configuration problem with a fix."""
    data = hp.ActivationSearchData(model=object(), images=[])

    with pytest.raises(ValueError, match="no images to score on"):
        hp.activation_fit_fn(data)


def test_an_attribution_sweep_has_only_two_modes():
    """A third mode would silently become one of the two."""
    data = hp.ActivationSearchData(model=object(), images=[object()])

    with pytest.raises(ValueError, match="mode must be 'grid' or 'random'"):
        hp.activation_search(data, SearchSpace({"cam_type": ["saliency"]}),
                             mode="bayesian")


# ---------------------------------------------------------------------------
# Scoring an embedding, without fitting one


def _blobs(n_per=12, seed=0):
    """Three well-separated clusters in five dimensions, and their labels."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0, 0, 0, 0], [10.0, 0, 0, 0, 0],
                        [0.0, 10, 0, 0, 0]])
    features = np.vstack([c + rng.normal(scale=0.3, size=(n_per, 5))
                          for c in centres])
    labels = np.repeat([0, 1, 2], n_per)
    return features, labels


def test_a_faithful_embedding_scores_high_on_both_neighbourhood_criteria():
    """Trustworthiness and continuity are the same measure, spaces swapped."""
    features, labels = _blobs()
    embedding = features[:, :2]

    scores = hp._umap_scores(features, embedding, labels, k=15)

    assert scores["trustworthiness"] > 0.9
    assert scores["continuity"] > 0.9
    assert scores["silhouette"] > 0.5
    assert scores["neighbourhood_k"] == float(min(15, (len(features) - 1) // 2))


def test_labels_that_do_not_describe_the_rows_score_no_silhouette():
    """A silhouette over the wrong labels would be a number that means nothing."""
    features, _labels = _blobs()

    assert "silhouette" not in hp._umap_scores(features, features[:, :2],
                                               np.zeros(3), k=5)
    assert "silhouette" not in hp._umap_scores(features, features[:, :2],
                                               np.zeros(len(features)), k=5)


def test_stability_ignores_rotation_and_scaling():
    """A rotated embedding is the same embedding; only neighbours count."""
    features, _labels = _blobs()
    embedding = features[:, :2]
    angle = np.pi / 3
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])

    assert hp.embedding_stability([embedding, embedding @ rotation * 5.0],
                                  neighbourhood_k=5) == pytest.approx(1.0)


def test_two_unrelated_embeddings_share_almost_no_neighbours():
    """Which is what a low stability score is supposed to mean."""
    rng = np.random.default_rng(0)
    first = rng.normal(size=(40, 2))
    second = rng.normal(size=(40, 2))

    assert hp.embedding_stability([first, second], neighbourhood_k=5) < 0.4


def test_stability_needs_at_least_two_repeats():
    """It is a repeat-to-repeat measure; one fit has nothing to compare to."""
    with pytest.raises(ValueError, match="at least two repeats"):
        hp.embedding_stability([np.zeros((5, 2))])


@pytest.mark.parametrize("embeddings, message", [
    ([np.zeros(5), np.zeros(5)], "at least 3 rows"),
    ([np.zeros((2, 2)), np.zeros((2, 2))], "at least 3 rows"),
    ([np.zeros((5, 2)), np.zeros((4, 2))], "same sample shape"),
    ([np.zeros((5, 2)), np.full((5, 2), np.nan)], "NaN or infinite"),
])
def test_embeddings_that_cannot_be_compared_are_refused(embeddings, message):
    """Every refusal names the property the caller violated."""
    with pytest.raises(ValueError, match=message):
        hp.embedding_stability(embeddings)


def test_supplied_labels_are_used_for_the_cluster_structure():
    """When the user has classes, discovering different ones would be wrong."""
    features, labels = _blobs()

    normalised, raw, method, n_clusters = hp._cluster_structure(
        features[:, :2], labels, seed=0)

    assert method == "supplied_labels"
    assert n_clusters == 3
    assert raw > 0.5
    assert 0.0 <= normalised <= 1.0


def test_without_labels_the_partition_is_discovered():
    """K-means over 2..8 clusters, best silhouette wins."""
    features, _labels = _blobs()

    normalised, raw, method, n_clusters = hp._cluster_structure(
        features[:, :2], None, seed=0)

    assert method == "discovered_kmeans"
    assert n_clusters == 3
    assert raw > 0.5
    assert normalised == pytest.approx(min(1.0, raw))


def test_cluster_discovery_needs_something_to_cluster():
    """Three points do not have a cluster structure."""
    with pytest.raises(ValueError, match="at least 4 rows"):
        hp._cluster_structure(np.zeros((3, 2)), None, seed=0)


def test_an_embedding_with_no_resolvable_clusters_scores_zero():
    """Identical points partition into nothing, which is a real answer."""
    normalised, raw, method, n_clusters = hp._cluster_structure(
        np.zeros((6, 2)), None, seed=0)

    assert (normalised, raw) == (0.0, 0.0)
    assert method == "no_resolved_clusters"
    assert n_clusters == 1


def test_naming_one_objective_weight_does_not_zero_the_others():
    """`{'stability': 1.0}` ends up near 0.59, not 1.0."""
    weights = hp._objective_weights({"stability": 1.0})

    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["stability"] == pytest.approx(1.0 / 1.7)
    assert weights["cluster_structure"] > 0


def test_the_default_weights_are_already_normalised():
    """The defaults are what a sweep uses when nothing is chosen."""
    assert hp._objective_weights(None) == pytest.approx(
        {name: value / sum(hp.DEFAULT_UMAP_OBJECTIVE_WEIGHTS.values())
         for name, value in hp.DEFAULT_UMAP_OBJECTIVE_WEIGHTS.items()})


@pytest.mark.parametrize("weights, message", [
    ({"speed": 1.0}, "Unknown UMAP objective weight"),
    ({"stability": "lots"}, "must be numeric"),
    ({"stability": -1.0}, "finite and"),
    ({"stability": float("nan")}, "finite and"),
    ({"stability": 0.0, "cluster_structure": 0.0,
      "neighborhood_preservation": 0.0}, "must be positive"),
])
def test_a_weighting_that_cannot_rank_anything_is_refused(weights, message):
    """A composite over zero weights would be the same number every time."""
    with pytest.raises(ValueError, match=message):
        hp._objective_weights(weights)


def test_the_three_objectives_are_reported_beside_their_composite():
    """The individual values are the result; the composite only ranks."""
    features, labels = _blobs()
    rng = np.random.default_rng(1)
    repeats = [features[:, :2] + rng.normal(scale=0.01, size=(len(features), 2))
               for _ in range(3)]

    scores = hp.umap_objective_scores(features, repeats, labels=labels,
                                      neighbourhood_k=5, seed=0)

    assert set(hp.UMAP_OBJECTIVES) <= set(scores)
    assert scores["cluster_structure_method"] == "supplied_labels"
    assert scores["cluster_counts"] == [3, 3, 3]
    assert scores["stability_repeats"] == 3
    assert scores["silhouette"] > 0.5
    assert 0.0 < scores["multi_objective"] <= 1.0
    assert sum(scores["objective_weights"].values()) == pytest.approx(1.0)


def test_a_collapsed_objective_drags_the_composite_down():
    """A geometric mean is why one excellent property cannot hide a failure."""
    features, _labels = _blobs()
    rng = np.random.default_rng(2)
    scattered = [rng.normal(size=(len(features), 2)) for _ in range(2)]

    good = hp.umap_objective_scores(
        features, [features[:, :2], features[:, :2]], neighbourhood_k=5)
    bad = hp.umap_objective_scores(features, scattered, neighbourhood_k=5)

    assert bad["multi_objective"] < good["multi_objective"]


def test_multi_objective_scoring_needs_repeats():
    """Stability cannot be measured from a single fit."""
    features, _labels = _blobs()

    with pytest.raises(ValueError, match="at least two stability repeats"):
        hp.umap_objective_scores(features, [features[:, :2]])


def test_umap_fits_the_embedding_the_search_scores(tmp_path):
    """The default embed function is the one a real sweep uses."""
    features, _labels = _blobs(n_per=8)

    embedding = hp._default_umap_embed(features, {"n_neighbors": 4,
                                                  "min_dist": 0.1}, seed=0)

    assert embedding.shape == (len(features), 2)
    assert np.isfinite(embedding).all()


# ---------------------------------------------------------------------------
# One attribution sweep, end to end on a hand-wired model


@pytest.fixture
def corner_model():
    """A model whose logit depends only on the top-left corner of the image."""
    import torch
    import torch.nn as nn

    class Corner(nn.Module):
        """Nothing is trained; the weights are set by hand."""

        def __init__(self):
            """Wire one identity convolution and a corner-only head."""
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 3, padding=1)
            self.head = nn.Linear(1, 2)
            with torch.no_grad():
                self.conv.weight.zero_()
                self.conv.weight[0, 0, 1, 1] = 1.0
                self.conv.bias.zero_()
                self.head.weight.fill_(8.0)
                self.head.bias.zero_()

        def forward(self, x):
            """Score the top-left 3x3 corner only."""
            feature = self.conv(x)
            mask = torch.zeros_like(feature)
            mask[:, :, :3, :3] = 1.0
            return self.head((feature * mask).mean(dim=(2, 3)))

    image = torch.zeros(1, 16, 16)
    image[0, :3, :3] = 2.0
    return hp.ActivationSearchData(model=Corner().eval(), images=[image],
                                   filenames=["synthetic"],
                                   model_type="corner")


def test_an_injected_attribution_is_scored_like_any_other(corner_model):
    """The override exists so a sweep can be driven without torchcam."""
    from spacr.attribution import attribute

    seen = []

    def _attribute(model, image, params):
        seen.append(dict(params))
        return attribute(model, image, "saliency", model_type="corner")

    fit = hp.activation_fit_fn(corner_model, n_steps=3,
                               run_sanity_check=False,
                               attribute_fn=_attribute)

    score, extra = fit({"cam_type": "gradcam"})

    assert seen == [{"cam_type": "gradcam"}]
    assert extra["n_images"] == 1
    assert extra["criterion"] == "deletion_auc"
    assert score == pytest.approx(extra["deletion_auc"])
    assert "insertion_auc" in extra
    assert "attribution" in extra


def test_smoothgrad_is_used_when_the_trial_asks_for_samples(corner_model):
    """`smoothgrad_samples` above one averages the map over noisy copies."""
    fit = hp.activation_fit_fn(corner_model, n_steps=3,
                               run_sanity_check=False)

    score, extra = fit({"cam_type": "saliency", "smoothgrad_samples": 2,
                        "smoothgrad_sigma": 0.1})

    assert extra["n_images"] == 1
    assert np.isfinite(score)


def test_a_random_attribution_sweep_samples_its_methods(corner_model):
    """`mode='random'` picks configurations instead of running the whole grid."""
    from spacr.attribution import attribute

    result = hp.activation_search(
        corner_model, SearchSpace({"cam_type": ["saliency", "gradcam"]}),
        mode="random", n_trials=1, n_steps=3, run_sanity_check=False,
        attribute_fn=lambda model, image, params: attribute(
            model, image, "saliency", model_type="corner"))

    assert len(result.trials) == 1
    assert result.higher_is_better is False
    assert any("no ground truth" in note for note in result.notes)
    assert any("sanity check was skipped" in note for note in result.notes)
