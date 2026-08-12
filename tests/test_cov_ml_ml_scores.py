"""CPU-only coverage for the classical-ML scoring block of ``spacr.ml``.

Covers ``generate_ml_scores`` (the ``measurements.db`` -> model ->
``results/`` wrapper) and every branch of ``ml_analysis`` that the rest of
the suite leaves cold: the nested ``_match_control_values`` fallbacks, the
``prune_features`` SelectKBest path, the verbose prints, the
cross-validation loop and each ``model_type`` selector (including the
optional lightgbm / catboost dependencies, exercised both present -- via a
stub module -- and missing).

Everything runs on a synthetic ``measurements.db`` built here (cell /
cytoplasm / nucleus / pathogen / png_list, Yokogawa-style metadata) so no
network, GPU or Cellpose is involved.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import types

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402

# spacr.ml imports spacr.utils lazily inside the functions under test; pull it
# in at collection time so the (heavy, one-off) umap/numba import is not billed
# to whichever test happens to run first.
import spacr.utils  # noqa: E402,F401


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# synthetic measurements.db
# ---------------------------------------------------------------------------

ROWS = ["r1", "r2"]
COLS = ["c1", "c2", "c3"]          # c1 = negative control, c2 = positive
FIELDS = ["f1", "f2"]
OBJ_PER_FIELD = 4
N_OBJ = len(ROWS) * len(COLS) * len(FIELDS) * OBJ_PER_FIELD   # 48


def _grid(plate):
    """(plateID, rowID, columnID, fieldID) for every synthetic object."""
    return [
        (plate, r, c, f)
        for r in ROWS
        for c in COLS
        for f in FIELDS
        for _ in range(OBJ_PER_FIELD)
    ]


def _entity_frame(rng, entity, plate, with_cell_id, with_centroids=True,
                  drop_columns=()):
    """One measurement table with the metadata columns spacr joins on."""
    grid = _grid(plate)
    n = len(grid)
    cols = {
        "object_label": np.arange(1, n + 1),
        "plateID": [g[0] for g in grid],
        "rowID": [g[1] for g in grid],
        "columnID": [g[2] for g in grid],
        "fieldID": [g[3] for g in grid],
        "prcf": [f"{g[0]}_{g[1]}_{g[2]}_{g[3]}" for g in grid],
        "prc": [f"{g[0]}_{g[1]}_{g[2]}" for g in grid],
    }
    if with_cell_id:
        # Parent-cell link: an INTEGER label (spacr prefixes 'o' itself).
        cols["cell_id"] = np.arange(1, n + 1)
    # Class signal: the positive-control column is brighter in channel 1.
    signal = np.array([1.0 if g[2] == "c2" else (0.0 if g[2] == "c1" else 0.5)
                       for g in grid])
    for ch in (0, 1):
        boost = 900.0 * signal if ch == 1 else 0.0
        cols[f"{entity}_channel_{ch}_mean_intensity"] = rng.uniform(500, 1500, n) + boost
        cols[f"{entity}_channel_{ch}_percentile_75"] = rng.uniform(500, 1500, n) + boost / 2
        if with_centroids:
            cols[f"{entity}_channel_{ch}_centroid_weighted-0"] = rng.uniform(0, 100, n)
            cols[f"{entity}_channel_{ch}_centroid_weighted-1"] = rng.uniform(0, 100, n)
    cols[f"{entity}_channel_1_std_intensity"] = rng.uniform(10, 200, n)
    cols[f"{entity}_area"] = rng.uniform(200, 4000, n)
    cols[f"{entity}_feret_diameter_max"] = rng.uniform(10, 40, n)
    df = pd.DataFrame(cols)
    return df.drop(columns=[c for c in drop_columns if c in df.columns])


def _make_src(tmp_path, name, rng, plate="plate1", annotation=None,
              with_centroids=True, drop_columns=()):
    """Build ``<tmp_path>/<name>/measurements/measurements.db``; return the src dir."""
    src = tmp_path / name
    meas = src / "measurements"
    meas.mkdir(parents=True)
    con = sqlite3.connect(meas / "measurements.db")
    try:
        for entity in ("cell", "cytoplasm"):
            _entity_frame(rng, entity, plate, False, with_centroids,
                          drop_columns).to_sql(entity, con, index=False)
        for entity in ("nucleus", "pathogen"):
            _entity_frame(rng, entity, plate, True, with_centroids,
                          drop_columns).to_sql(entity, con, index=False)
        grid = _grid(plate)
        png = pd.DataFrame({
            # png_list stores the already-prefixed 'o<N>' string form.
            "cell_id": [f"o{i + 1}" for i in range(len(grid))],
            "plateID": [g[0] for g in grid],
            "rowID": [g[1] for g in grid],
            "columnID": [g[2] for g in grid],
            "fieldID": [g[3] for g in grid],
            "prcf": [f"{g[0]}_{g[1]}_{g[2]}_{g[3]}" for g in grid],
            "prcfo": [f"{g[0]}_{g[1]}_{g[2]}_{g[3]}_o{i + 1}"
                      for i, g in enumerate(grid)],
            "png_path": [f"/nonexistent/{i}.png" for i in range(len(grid))],
        })
        if annotation is not None:
            png["test"] = annotation
        png.to_sql("png_list", con, index=False)
    finally:
        con.close()
    return str(src)


def _ml_settings(src, **over):
    """Small, fast settings for generate_ml_scores."""
    s = dict(
        src=src,
        channel_of_interest=1,
        model_type_ml="random_forest",
        heatmap_feature="predictions",
        n_estimators=5,
        n_repeats=1,
        test_size=0.25,
        minimum_cell_count=1,
        cross_validation=False,
        remove_highly_correlated_features=False,
        nuclei_limit=True,
        pathogen_limit=3,
        save_to_db=False,
        verbose=False,
        n_jobs=1,
    )
    s.update(over)
    return s


# ---------------------------------------------------------------------------
# generate_ml_scores
# ---------------------------------------------------------------------------

def test_generate_ml_scores_writes_every_artifact_and_updates_png_list(tmp_path, rng):
    """The full DB -> model -> results/ path: 10-tuple + heatmap returned,
    all CSV/PDF artifacts on disk, and the scores back on png_list."""
    from spacr.ml import generate_ml_scores

    src = _make_src(tmp_path, "plate1", rng)
    settings = _ml_settings(src, save_to_db=True)
    output, plate_heatmap = generate_ml_scores(settings)

    # --- returned objects
    assert isinstance(plate_heatmap, matplotlib.figure.Figure)
    assert len(output) == 10
    scored_df, permutation_df, feature_importance_df = output[0], output[1], output[2]
    assert len(scored_df) == N_OBJ
    assert {"predictions", "prcfo", "prc", "data_usage"}.issubset(scored_df.columns)
    assert set(scored_df["predictions"].unique()).issubset({0, 1})
    # only channel-1 features survive filter_dataframe_features
    features = output[9]
    assert features and all("channel_1" in f for f in features)

    # --- artifacts on disk
    res = os.path.join(src, "results", "random_forest", "channel_1")
    for name in ("results.csv", "permutation.csv", "feature_importance.csv",
                 "ml_features.csv", "random_forest_model.csv",
                 "permutation.pdf", "feature_importance.pdf", "shap.pdf",
                 "plate_heatmap.pdf"):
        assert os.path.getsize(os.path.join(res, name)) > 0, name

    written = pd.read_csv(os.path.join(res, "results.csv"))
    assert len(written) == N_OBJ
    assert "predictions" in written.columns
    assert pd.read_csv(os.path.join(res, "ml_features.csv"))["feature"].tolist() == features
    assert len(pd.read_csv(os.path.join(res, "permutation.csv"))) == len(permutation_df)
    assert len(pd.read_csv(os.path.join(res, "feature_importance.csv"))) == len(feature_importance_df)
    # the settings snapshot for a str src (not the *_list variant)
    assert os.path.isfile(os.path.join(src, "settings", "generate_ml_scores.csv"))

    # --- the scores are back on png_list, on the right row
    #
    # Written by spacr.predictions.merge_ml_predictions, which replaced
    # utils.add_column_to_database here. Three things changed with it: the
    # merge happens whether or not save_to_db is set (the model scored the
    # whole database, so the whole database gets the scores), the class is
    # stored as the model produced it -- the old writer replaced every 0 with
    # a 2, the Annotate app's label encoding, so png_list disagreed with the
    # results.csv from the same run -- and a re-run updates in place instead
    # of appending a 'predictions_1' sibling. ml_pred is new: it carries the
    # probability, which the ML stage never stored at all, namespaced so a
    # later CV run writing 'pred' cannot overwrite it.
    assert settings["csv_path"] == os.path.join(res, "results.csv")
    assert settings["table_name"] == "png_list"
    con = sqlite3.connect(os.path.join(src, "measurements", "measurements.db"))
    try:
        png_back = pd.read_sql_query(
            "SELECT prcfo, predictions, ml_pred FROM png_list", con)
        names = [r[1] for r in con.execute("PRAGMA table_info(png_list)")]
    finally:
        con.close()
    assert len(png_back) == N_OBJ
    assert png_back["predictions"].notna().all()
    assert png_back["ml_pred"].notna().all()
    assert set(png_back["predictions"].unique()).issubset({0, 1})
    assert names.count("predictions") == 1
    assert not [n for n in names if n.startswith("predictions_")]
    # every object got *its own* score, matched on prcfo
    by_prcfo = dict(zip(png_back["prcfo"], png_back["predictions"]))
    assert by_prcfo == dict(zip(scored_df["prcfo"], scored_df["predictions"]))


def test_generate_ml_scores_annotation_column_balances_single_class(tmp_path, rng):
    """One annotated class + NaNs: the missing half is sampled from the
    unannotated rows, and the controls are derived from the annotation.

    The DB deliberately lacks the weighted-centroid columns, so the
    pathogen<->nucleus shortest-distance step fails and is swallowed.
    """
    from spacr.ml import generate_ml_scores

    # 12 of the 48 objects annotated with 1.0, the rest NaN.
    annotation = [1.0 if i % 4 == 0 else None for i in range(N_OBJ)]
    src = _make_src(tmp_path, "plate_annot1", rng, annotation=annotation,
                    with_centroids=False)
    settings = _ml_settings(src, annotation_column="test")
    output, _ = generate_ml_scores(settings)

    # annotation column drives the control assignment
    assert settings["location_column"] == "test"
    assert settings["positive_control"] == "1.0"
    assert settings["negative_control"] == "2.0"
    counts = output[0]["test"].value_counts().to_dict()
    assert counts["1.0"] == 12          # originally annotated
    assert counts["2.0"] == 12          # sampled from the unannotated rows
    assert counts["nan"] == N_OBJ - 24  # untouched
    # both classes were actually used for training
    assert set(output[0]["data_usage"].unique()) == {"train", "test", "not_used"}
    # the shortest-distance feature could not be computed (no centroids)
    assert "pathogen_nucleus_shortest_distance" not in output[0].columns


def test_generate_ml_scores_annotation_column_autoselects_controls(tmp_path, rng):
    """Two annotated classes with no controls configured: positive/negative
    control are auto-derived from the unique annotation values."""
    from spacr.ml import generate_ml_scores

    annotation = [float(i % 2) for i in range(N_OBJ)]
    src = _make_src(tmp_path, "plate_annot2", rng, annotation=annotation)
    settings = _ml_settings(src, annotation_column="test",
                            positive_control=None, negative_control=None)
    output, _ = generate_ml_scores(settings)

    assert settings["positive_control"] == "0.0"
    assert settings["negative_control"] == "1.0"
    # every row is a control here, so nothing is left unused
    assert "not_used" not in set(output[0]["data_usage"].unique())
    assert len(output[0]) == N_OBJ


def test_generate_ml_scores_annotation_column_missing_raises(tmp_path, rng):
    """A png_list without the requested annotation column is a hard error."""
    from spacr.ml import generate_ml_scores

    src = _make_src(tmp_path, "plate_annot3", rng, annotation=None)
    with pytest.raises(ValueError, match="prcfo"):
        generate_ml_scores(_ml_settings(src, annotation_column="test"))


def test_generate_ml_scores_unknown_heatmap_feature_raises(tmp_path, rng):
    """heatmap_feature must name a numeric column of the scored frame."""
    from spacr.ml import generate_ml_scores

    src = _make_src(tmp_path, "plate_heat", rng)
    with pytest.raises(ValueError, match="not_a_feature"):
        generate_ml_scores(_ml_settings(src, heatmap_feature="not_a_feature"))


def test_generate_ml_scores_survives_missing_pathogen_channel_column(tmp_path, rng):
    """recruitment must be skipped (not crash) when the pathogen intensity
    column is absent while the cytoplasm one is present."""
    from spacr.ml import generate_ml_scores

    src = _make_src(tmp_path, "plate_norec", rng,
                    drop_columns=("pathogen_channel_1_mean_intensity",))
    output, _ = generate_ml_scores(_ml_settings(src))
    assert "predictions" in output[0].columns


# ---------------------------------------------------------------------------
# ml_analysis — feature frame + shared kwargs
# ---------------------------------------------------------------------------

FEATURES = [
    "cell_channel_3_mean_intensity",
    "cell_channel_3_percentile_75",
    "nucleus_channel_3_mean_intensity",
    "cytoplasm_channel_3_mean_intensity",
    "pathogen_channel_3_mean_intensity",
    "cell_channel_3_std_intensity",
]

COMMON = dict(channel_of_interest=3, location_column="columnID", n_repeats=1,
              n_jobs=1, remove_highly_correlated_features=False, test_size=0.25)


def _feature_df(per_class=40, loc_values=("c1", "c2", "c3")):
    """Per-object features with a prcfo-style (5 part) index."""
    rng = np.random.default_rng(8)
    rows, index = [], []
    for loc, centre in zip(loc_values, (0.3, 0.9, 0.5)):
        for _ in range(per_class):
            row = {"columnID": loc}
            for f in FEATURES:
                row[f] = float(rng.normal(centre, 0.12) + rng.normal(0, 0.05))
            rows.append(row)
            index.append(f"plate1_r1_{loc}_f1_o{len(index)}")
    return pd.DataFrame(rows, index=index)


def test_ml_analysis_prunes_features_and_drops_cells_per_well(capsys, rng):
    """verbose + prune_features + a model that exposes feature_importances_:
    SelectKBest caps the feature space, cells_per_well is dropped and both
    figures are produced."""
    import spacr.ml as ML

    df = _feature_df()
    df["cells_per_well"] = 10
    output, figs = ML.ml_analysis(
        df, positive_control="c2", negative_control="c1",
        model_type="extra_trees", n_estimators=8,
        prune_features=True, top_features=4, verbose=True, **COMMON)

    scored_df, permutation_df, feature_importance_df = output[0], output[1], output[2]
    features = output[9]
    # SelectKBest kept exactly k features, all from the original space
    assert len(features) == 4
    assert set(features).issubset(set(FEATURES))
    assert list(output[4].columns) == features        # X_train
    # the metadata column never reaches the model
    assert "cells_per_well" not in scored_df.columns
    assert len(permutation_df) == 4
    assert len(feature_importance_df) == 4
    assert type(output[3]).__name__ == "ExtraTreesClassifier"
    assert isinstance(figs[0], matplotlib.figure.Figure)
    assert isinstance(figs[1], matplotlib.figure.Figure)
    out = capsys.readouterr().out
    assert "Removed 2 features using SelectKBest" in out
    assert "Optimal threshold:" in out
    assert "Classification Report:" in out


def test_ml_analysis_cross_validation_averages_folds(capsys, rng):
    """cross_validation=True aggregates 5 folds and re-scores every row."""
    import spacr.ml as ML

    output, figs = ML.ml_analysis(
        _feature_df(), positive_control="c2", negative_control="c1",
        model_type="random_forest", n_estimators=8,
        cross_validation=True, verbose=True, **COMMON)

    scored_df, metrics_df = output[0], output[8]
    assert "accuracy" in metrics_df.index
    # the refit model scored every row of the frame, not just the controls
    assert scored_df["predictions"].notna().all()
    assert {"prediction_probability_class_0",
            "prediction_probability_class_1"}.issubset(scored_df.columns)
    probs = scored_df[["prediction_probability_class_0",
                       "prediction_probability_class_1"]].sum(axis=1)
    assert np.allclose(probs.to_numpy(), 1.0)
    out = capsys.readouterr().out
    assert "Fold 5 - Optimal threshold:" in out
    assert "Fold 5 Classification Report:" in out


def test_ml_analysis_applies_batch_correction_before_training(capsys, rng):
    """The shared correction runs on features while plate metadata stays out."""
    import spacr.ml as ML

    df = _feature_df(per_class=24)
    within_condition = np.tile(
        np.repeat(["plate1", "plate2"], 12),
        3,
    )
    df["plateID"] = within_condition
    shifted = df["plateID"].eq("plate2")
    df.loc[shifted, FEATURES] = df.loc[shifted, FEATURES] + 10.0

    output, _figs = ML.ml_analysis(
        df,
        positive_control="c2",
        negative_control="c1",
        model_type="random_forest",
        n_estimators=8,
        verbose=False,
        batch_correction="center",
        batch_column="plateID",
        **COMMON,
    )

    assert output[0]["predictions"].notna().all()
    assert "plateID" not in output[9]
    assert "Batch correction center:" in capsys.readouterr().out


def test_ml_analysis_matches_list_controls_numerically(rng):
    """Controls given as a list: each entry is matched exactly, numerically
    and as a stripped string, and un-matchable entries are skipped."""
    import spacr.ml as ML

    df = _feature_df(loc_values=("1", "2", "3"))
    output, _ = ML.ml_analysis(
        df,
        # the nested list can be compared neither elementwise nor numerically
        negative_control=["1", ["x", "y"]],
        positive_control=["2"],
        model_type="random_forest", n_estimators=8, verbose=False, **COMMON)

    usage = output[0]["data_usage"].value_counts().to_dict()
    assert usage["not_used"] == 40           # the '3' wells
    assert usage["train"] + usage["test"] == 80
    assert usage["test"] == 20               # test_size=0.25 of the controls
    # the '3' wells were never labelled
    assert output[0].loc[output[0]["columnID"] == "3", "data_usage"].eq("not_used").all()


def test_ml_analysis_duplicate_location_column_matches_nothing(rng):
    """A duplicated location column makes df[location_column] a DataFrame, so
    all three matching strategies fail and no control rows are found.

    This used to surface as sklearn's "n_samples=0" from inside
    train_test_split, three frames below anything a user recognises -- the
    traceback that was auto-filed ten times as issues #79-#90. It is now
    named where it happens, and the DUPLICATE COLUMN gets its own sentence
    because the fix is to the table rather than to the control values: no
    amount of correcting positive_control helps.
    """
    import spacr.ml as ML

    df = _feature_df(per_class=20)
    df["dup"] = df["columnID"]
    df.columns = ["columnID" if c == "dup" else c for c in df.columns]

    with pytest.raises(ValueError, match="columns named 'columnID'"):
        ML.ml_analysis(df, positive_control="c2", negative_control="c1",
                       model_type="random_forest", n_estimators=5,
                       verbose=False, **COMMON)


@pytest.mark.parametrize("model_type,cls_name", [("svm", "CalibratedClassifierCV"),
                                                 ("mlp", "MLPClassifier")])
def test_ml_analysis_models_without_feature_importances(model_type, cls_name, rng):
    """svm / mlp have no feature_importances_ -> empty importance table and
    no importance figure, but a fully scored frame."""
    import spacr.ml as ML

    output, figs = ML.ml_analysis(
        _feature_df(per_class=30), positive_control="c2", negative_control="c1",
        model_type=model_type, n_estimators=20, verbose=False, **COMMON)

    assert type(output[3]).__name__ == cls_name
    assert output[2].empty
    assert figs[1] is None
    assert isinstance(figs[0], matplotlib.figure.Figure)
    assert output[0]["predictions"].isin([0, 1]).all()


# --- optional gradient-boosting backends -----------------------------------
# lightgbm / catboost are optional extras. Stub modules let the *present*
# branch run (and pin the constructor arguments spacr passes), while
# sys.modules[...] = None forces the ImportError branch deterministically.

class _StubLGBMClassifier(RandomForestClassifier):
    """Stand-in for lightgbm.LGBMClassifier with the same constructor kwargs."""

    def __init__(self, n_estimators=10, learning_rate=0.1, reg_alpha=0.0,
                 reg_lambda=0.0, random_state=None, n_jobs=None):
        super().__init__(n_estimators=n_estimators, random_state=random_state,
                         n_jobs=n_jobs)
        self.learning_rate = learning_rate
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda


class _StubCatBoostClassifier(RandomForestClassifier):
    """Stand-in for catboost.CatBoostClassifier with the same constructor kwargs."""

    def __init__(self, iterations=10, learning_rate=0.1, l2_leaf_reg=1.0,
                 random_state=None, thread_count=None, verbose=False):
        super().__init__(n_estimators=iterations, random_state=random_state,
                         n_jobs=thread_count)
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.thread_count = thread_count
        self.verbose = verbose


def _install_stub(monkeypatch, mod_name, attr_name, cls):
    module = types.ModuleType(mod_name)
    setattr(module, attr_name, cls)
    monkeypatch.setitem(sys.modules, mod_name, module)


def test_ml_analysis_lightgbm_forwards_hyperparameters(monkeypatch, rng):
    """model_type='lightgbm' builds LGBMClassifier with spacr's hyperparameters."""
    import spacr.ml as ML

    _install_stub(monkeypatch, "lightgbm", "LGBMClassifier", _StubLGBMClassifier)
    output, _ = ML.ml_analysis(
        _feature_df(per_class=30), positive_control="c2", negative_control="c1",
        model_type="lightgbm", n_estimators=6, learning_rate=0.05,
        reg_alpha=0.3, reg_lambda=2.0, verbose=False, **COMMON)

    model = output[3]
    assert isinstance(model, _StubLGBMClassifier)
    assert model.get_params() == {
        "n_estimators": 6, "learning_rate": 0.05, "reg_alpha": 0.3,
        "reg_lambda": 2.0, "random_state": 42, "n_jobs": 1,
    }
    assert output[0]["predictions"].isin([0, 1]).all()


def test_ml_analysis_catboost_forwards_hyperparameters(monkeypatch, rng):
    """model_type='catboost' maps spacr's kwargs onto CatBoost's names."""
    import spacr.ml as ML

    _install_stub(monkeypatch, "catboost", "CatBoostClassifier",
                  _StubCatBoostClassifier)
    output, _ = ML.ml_analysis(
        _feature_df(per_class=30), positive_control="c2", negative_control="c1",
        model_type="catboost", n_estimators=7, learning_rate=0.05,
        reg_lambda=3.0, verbose=False, **COMMON)

    model = output[3]
    assert isinstance(model, _StubCatBoostClassifier)
    assert model.get_params() == {
        "iterations": 7, "learning_rate": 0.05, "l2_leaf_reg": 3.0,
        "random_state": 42, "thread_count": 1, "verbose": False,
    }
    assert output[0]["predictions"].isin([0, 1]).all()


@pytest.mark.parametrize("model_type", ["lightgbm", "catboost"])
def test_ml_analysis_missing_optional_backend_raises_actionable_importerror(
        model_type, monkeypatch, rng):
    """When the optional package cannot be imported the user gets an
    ImportError naming the pip install, not a bare ModuleNotFoundError."""
    import spacr.ml as ML

    # None in sys.modules makes `from <mod> import ...` raise ImportError
    # regardless of whether the package happens to be installed.
    monkeypatch.setitem(sys.modules, model_type, None)
    with pytest.raises(ImportError, match=f"pip install {model_type}"):
        ML.ml_analysis(_feature_df(per_class=20), positive_control="c2",
                       negative_control="c1", model_type=model_type,
                       n_estimators=5, verbose=False, **COMMON)
