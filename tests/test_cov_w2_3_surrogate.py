"""The surrogate's refusals, its fallbacks, and its SHAP normalisation.

A surrogate explains a CV classifier only when it can reproduce it, and
:mod:`spacr.surrogate` is mostly the machinery around that caveat: the
join that has to be one-to-one, the backends that must not be silently
substituted, and the several shapes SHAP has returned across releases.

Every fit here is deliberately tiny -- a dozen trees, one permutation
repeat -- because what is being checked is which branch was taken, not
how well a forest separates 400 synthetic objects.
"""
from __future__ import annotations

import builtins
import importlib.metadata
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import surrogate

#: A fit small enough to run in a test and still exercise every branch.
QUICK = {"n_estimators": 8, "n_repeats": 1, "shap_max_samples": 20,
         "verbose": False, "model_options": {"n_jobs": 1}}


def _driven_frame(n=200, seed=0, driver_strength=2.0):
    """Objects whose CV class is decided by ``cell_area`` alone."""
    rng = np.random.default_rng(seed)
    area = rng.normal(500, 120, n)
    frame = pd.DataFrame({
        "cell_area": area,
        "cell_channel_1_mean_intensity": rng.normal(0, 1, n),
        "noise_b": rng.normal(0, 1, n),
        "plateID": ["p1"] * n,
        "rowID": [f"r{(i // 25) + 1}" for i in range(n)],
        "columnID": [f"c{(i // 10) + 1}" for i in range(n)],
        "fieldID": [f"f{(i % 4) + 1}" for i in range(n)],
    })
    logit = driver_strength * (area - 500) / 120
    frame["cv_prediction"] = (logit + rng.normal(0, 0.2, n) > 0).astype(int)
    return frame


def _spacr_database(where, objects=140, seed=0):
    """A database shaped the way spaCR writes one: ``prcfo`` in png_list only."""
    rng = np.random.default_rng(seed)
    path = str(where / "measurements.db")
    rows, pngs = [], []
    for index in range(objects):
        row, column = f"r{index % 7 + 1}", f"c{index % 10 + 1}"
        name = f"plate1_{row}{column}_f1_o{index}.png"
        area = float(rng.normal(500, 120))
        rows.append((index, "pplate1", row, column, "f1", area,
                     float(rng.normal()), float(rng.normal())))
        pngs.append((f"/somewhere/data/{name}", name, "pplate1", row, column,
                     "f1", f"pplate1_{row}_{column}_f1_o{index}", f"o{index}",
                     area))
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE cell (object_label INTEGER, plateID TEXT, rowID TEXT, "
        "columnID TEXT, fieldID TEXT, cell_area REAL, cell_solidity REAL, "
        "cell_perimeter REAL)")
    connection.executemany("INSERT INTO cell VALUES (?,?,?,?,?,?,?,?)", rows)
    connection.execute(
        "CREATE TABLE png_list (png_path TEXT, file_name TEXT, plateID TEXT, "
        "rowID TEXT, columnID TEXT, fieldID TEXT, prcfo TEXT, cell_id TEXT, "
        "area REAL)")
    connection.executemany(
        "INSERT INTO png_list VALUES (?,?,?,?,?,?,?,?,?)", pngs)
    connection.commit()
    connection.close()
    return path, [png[1] for png in pngs], [row[5] for row in rows]


# ---------------------------------------------------------------------------
# SurrogateResult.top
# ---------------------------------------------------------------------------

def test_an_unfaithful_surrogate_withholds_its_ranking_entirely():
    """The ranking is of noise, so ``top`` hands back no rows at all."""
    result = surrogate.SurrogateResult(
        fidelity=0.50, baseline=0.50,
        importance=pd.DataFrame({"feature": ["a", "b"],
                                 "permutation": [0.3, 0.1]}),
        n_objects=100)

    assert not result.is_faithful
    top = result.top(5)
    assert list(top.columns) == ["feature", "permutation"]
    assert len(top) == 0


def test_top_falls_back_to_the_stored_order_when_nothing_was_measured():
    """With no gain, permutation or shap column there is nothing to sort on."""
    result = surrogate.SurrogateResult(
        fidelity=0.9, baseline=0.5,
        importance=pd.DataFrame({"feature": ["a", "b", "c"]}),
        n_objects=100)

    assert result.top(2)["feature"].tolist() == ["a", "b"]
    assert result.fidelity_improvement == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

def test_a_missing_distribution_is_reported_with_what_to_install(monkeypatch):
    """An absent backend stays visible and says how to enable it."""
    real_version = importlib.metadata.version

    def version(distribution):
        if distribution == "xgboost":
            raise importlib.metadata.PackageNotFoundError(distribution)
        return real_version(distribution)

    monkeypatch.setattr(importlib.metadata, "version", version)

    backends = surrogate.available_backends()

    assert backends["xgboost"] == {
        "available": False, "version": "",
        "reason": "install xgboost to enable this backend"}
    assert backends["random_forest"]["available"] is True
    assert backends["random_forest"]["version"]


def test_the_version_of_something_that_is_not_installed_is_blank():
    """A missing distribution has no version, and that is not an error."""
    assert surrogate._package_version("no-such-distribution-w2-3") == ""
    assert surrogate._package_version("numpy")


# ---------------------------------------------------------------------------
# the join
# ---------------------------------------------------------------------------

def test_a_missing_database_is_named_before_anything_is_read(tmp_path):
    """The path is in the message, because the wrong path is the usual fault."""
    missing = tmp_path / "nowhere" / "measurements.db"
    with pytest.raises(surrogate.SurrogateError,
                       match="no measurements database at"):
        surrogate._read_png_list(str(missing))


def test_a_prcfo_index_is_promoted_to_a_column_and_used_as_the_key(tmp_path,
                                                                   monkeypatch):
    """Some readers hand back ``prcfo`` as the index; it is still the key."""
    path, names, _areas = _spacr_database(tmp_path, objects=30)
    real_join = surrogate.__dict__  # kept for clarity; the patch is below

    from spacr import io

    def indexed(db_path, *args, **kwargs):
        frame = pd.DataFrame({
            "prcfo": [f"pplate1_r{i % 7 + 1}_c{i % 10 + 1}_f1_o{i}"
                      for i in range(30)],
            "cell_area": np.linspace(100, 900, 30),
        }).set_index("prcfo")
        return frame

    monkeypatch.setattr(io, "_read_and_join_tables", indexed)

    predictions = pd.DataFrame({"path": names,
                                "pred": [i % 2 for i in range(30)]})
    frame = surrogate.build_surrogate_frame(path, predictions)

    assert "prcfo" in frame.columns
    assert len(frame) == 30
    assert frame["cv_prediction"].nunique() == 2


def test_a_feature_table_sharing_no_key_names_what_each_side_has(tmp_path,
                                                                 monkeypatch):
    """Neither prcfo nor png_path in both is a refusal, not a silent empty fit."""
    path, names, _areas = _spacr_database(tmp_path, objects=30)
    from spacr import io

    monkeypatch.setattr(
        io, "_read_and_join_tables",
        lambda *args, **kwargs: pd.DataFrame({"cell_area": [1.0, 2.0]}))

    predictions = pd.DataFrame({"path": names, "pred": [0] * 30})
    with pytest.raises(surrogate.SurrogateError, match="share no object key"):
        surrogate.build_surrogate_frame(path, predictions)


def test_crops_that_match_no_measured_object_are_refused(tmp_path, monkeypatch):
    """A join that matches crops but no features explains nothing."""
    path, names, _areas = _spacr_database(tmp_path, objects=30)
    from spacr import io

    monkeypatch.setattr(
        io, "_read_and_join_tables",
        lambda *args, **kwargs: pd.DataFrame({
            "prcfo": ["a_different_object"], "cell_area": [1.0]}))

    predictions = pd.DataFrame({"path": names, "pred": [0] * 30})
    with pytest.raises(surrogate.SurrogateError,
                       match="matched no\n?\\s*measured objects"):
        surrogate.build_surrogate_frame(path, predictions)


# ---------------------------------------------------------------------------
# estimator selection
# ---------------------------------------------------------------------------

def test_xgboost_is_refused_rather_than_quietly_replaced(monkeypatch):
    """An unavailable optional backend must never become a random forest."""
    monkeypatch.setattr(
        surrogate, "available_model_families",
        lambda: {"xgboost": {"available": False, "label": "XGBoost",
                             "reason": "Install the optional xgboost package."}})

    with pytest.raises(surrogate.SurrogateError, match="not\n?\\s*installed"):
        surrogate._make_estimator("xgboost", n_estimators=4, random_seed=0,
                                  model_options={}, y_train=pd.Series([0, 1]))


def test_an_xgboost_that_will_not_import_says_so(monkeypatch):
    """Installed-but-broken is a different failure from not installed."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError("libgomp.so.1: cannot open shared object file")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(
        surrogate, "available_model_families",
        lambda: {"xgboost": {"available": True, "label": "XGBoost",
                             "reason": ""}})
    monkeypatch.setattr(builtins, "__import__", blocked)

    with pytest.raises(surrogate.SurrogateError,
                       match="could not be imported"):
        surrogate._make_estimator("xgboost", n_estimators=4, random_seed=0,
                                  model_options={}, y_train=pd.Series([0, 1]))


# ---------------------------------------------------------------------------
# correlation audit and feature families
# ---------------------------------------------------------------------------

def test_a_wide_frame_audits_its_highest_variance_features_and_says_so():
    """The audit is capped, and the cap is written into the warnings."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "wide": rng.normal(0, 100, 50),
        "narrow": rng.normal(0, 0.001, 50),
        "middling": rng.normal(0, 1, 50),
    })
    frame["wide_copy"] = frame["wide"] * 2 + rng.normal(0, 0.01, 50)
    warnings = []

    pairs = surrogate._correlation_pairs(frame, 0.9, warnings, max_features=2)

    assert warnings == [
        "correlation audit used the 2 highest-variance of 4 features"]
    assert set(pairs.columns) == {"feature_a", "feature_b", "spearman"}
    assert pairs["spearman"].abs().min() >= 0.9


def test_one_feature_cannot_be_correlated_with_anything():
    """A single column produces the empty table, not an exception."""
    warnings = []
    pairs = surrogate._correlation_pairs(
        pd.DataFrame({"only": [1.0, 2.0, 3.0]}), 0.9, warnings)

    assert pairs.empty
    assert list(pairs.columns) == ["feature_a", "feature_b", "spearman"]
    assert warnings == []


def test_features_with_no_object_prefix_fall_back_to_what_they_measure():
    """Texture, intensity and location are families in their own right."""
    assert surrogate._feature_family("channel_1_texture_contrast") == "texture"
    assert surrogate._feature_family("channel_2_mean_intensity") == "intensity"
    assert surrogate._feature_family("distance_to_edge") == "spatial"
    assert surrogate._feature_family("location_x") == "spatial"
    assert surrogate._feature_family("cell_area") == "cell"
    assert surrogate._feature_family("shape_factor") == "other"


# ---------------------------------------------------------------------------
# fit_surrogate
# ---------------------------------------------------------------------------

def test_an_unusable_split_unit_is_a_surrogate_error_not_a_value_error():
    """Every refusal this module makes arrives as one exception type."""
    with pytest.raises(surrogate.SurrogateError):
        surrogate.fit_surrogate(_driven_frame(60), split_by="galaxy", **QUICK)


def test_a_helper_returning_only_importance_is_still_accepted(monkeypatch):
    """The SHAP helper's older return shapes stay usable."""
    frame = _driven_frame(120)

    monkeypatch.setattr(
        surrogate, "_shap_importance",
        lambda model, x_test, max_samples, warnings, **kwargs:
            np.arange(x_test.shape[1], dtype=float))
    plain = surrogate.fit_surrogate(frame, **QUICK)
    assert "shap" in plain.importance.columns
    assert plain.shap_values.empty

    monkeypatch.setattr(
        surrogate, "_shap_importance",
        lambda model, x_test, max_samples, warnings, **kwargs: (
            np.arange(x_test.shape[1], dtype=float),
            pd.DataFrame({"cell_area": [0.1]})))
    paired = surrogate.fit_surrogate(frame, **QUICK)
    assert "shap" in paired.importance.columns
    assert list(paired.shap_values.columns) == ["cell_area"]


def test_probabilities_that_cannot_be_computed_cost_the_columns_not_the_fit(
        monkeypatch):
    """A predict_proba that raises is recorded and the result still arrives.

    Histogram gradient boosting is the family used here because its
    ``predict`` goes through the raw decision function rather than
    through ``predict_proba``: replacing the probabilities on a random
    forest also replaces its predictions, which is a different failure
    from the one being tested.
    """
    real_make = surrogate._make_estimator

    def broken(*args, **kwargs):
        model, fit_y, decode, info = real_make(*args, **kwargs)

        class _NoProbabilities(type(model)):
            def predict_proba(self, *a, **k):
                raise RuntimeError("this model will not say how sure it is")

        model.__class__ = _NoProbabilities
        return model, fit_y, decode, info

    monkeypatch.setattr(surrogate, "_make_estimator", broken)

    options = dict(QUICK, model_family="hist_gradient_boosting",
                   model_options={})
    result = surrogate.fit_surrogate(_driven_frame(120), **options)

    assert any("held-out probabilities unavailable" in w
               for w in result.warnings), result.warnings
    assert not any(c.startswith("probability_") for c in result.held_out.columns)
    assert "surrogate_prediction" in result.held_out.columns


def test_a_verbose_fit_prints_the_summary_it_returns(capsys):
    """``verbose=True`` is the whole report, not a progress line."""
    options = dict(QUICK, verbose=True)
    result = surrogate.fit_surrogate(_driven_frame(120), **options)

    printed = capsys.readouterr().out
    assert "Surrogate model of the CV classifier" in printed
    assert f"surrogate fidelity : {result.fidelity:.3f}" in printed


# ---------------------------------------------------------------------------
# _shap_importance
# ---------------------------------------------------------------------------

class _Explainer:
    """Stands in for ``shap.TreeExplainer`` and returns a chosen array."""

    def __init__(self, values, *, reject_check_additivity=False, raises=None):
        self._values = values
        self._reject = reject_check_additivity
        self._raises = raises

    def __call__(self, model):
        if self._raises is not None:
            raise self._raises
        return self

    def shap_values(self, sample, **kwargs):
        if self._reject and "check_additivity" in kwargs:
            raise TypeError("shap_values() got an unexpected keyword argument")
        values = self._values
        return values(sample) if callable(values) else values


@pytest.fixture
def sample_frame():
    return pd.DataFrame({"a": np.linspace(0, 1, 12),
                         "b": np.linspace(1, 0, 12),
                         "c": np.zeros(12)})


def _install(monkeypatch, explainer):
    import shap
    monkeypatch.setattr(shap, "TreeExplainer", explainer)


class _Model:
    """The least a model has to be for the SHAP helper."""

    def predict_proba(self, sample):
        return np.tile([0.4, 0.6], (len(sample), 1))


def test_a_missing_shap_costs_the_column_and_says_how_to_get_it(monkeypatch,
                                                                sample_frame):
    """The optional dependency is optional, and its absence is explained."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("no module named shap")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    warnings = []

    assert surrogate._shap_importance(_Model(), sample_frame, 50, warnings) is None
    assert "shap is not installed" in warnings[0]
    assert "pip install shap" in warnings[0]


def test_an_explainer_that_rejects_check_additivity_is_called_again(
        monkeypatch, sample_frame):
    """Older SHAP has no ``check_additivity``; the retry is not a failure."""
    _install(monkeypatch, _Explainer(np.ones((12, 3)),
                                     reject_check_additivity=True))
    warnings = []

    importance = surrogate._shap_importance(_Model(), sample_frame, 50, warnings)

    assert importance is not None
    assert list(importance) == [1.0, 1.0, 1.0]
    assert warnings == []


def test_an_explainer_that_raises_costs_the_column_not_the_analysis(
        monkeypatch, sample_frame):
    """SHAP is the expensive third of three, and it fails alone."""
    _install(monkeypatch,
             _Explainer(None, raises=RuntimeError("no tree model here")))
    warnings = []

    assert surrogate._shap_importance(_Model(), sample_frame, 50, warnings) is None
    assert "SHAP failed (RuntimeError: no tree model here)" in warnings[0]
    assert "gain and permutation columns are unaffected" in warnings[0]


def test_a_per_class_list_of_matrices_is_stacked_not_averaged_wrongly(
        monkeypatch, sample_frame):
    """Old SHAP returns one matrix per class; features stay on axis 1."""
    per_class = [np.full((12, 3), 2.0), np.full((12, 3), 4.0)]
    _install(monkeypatch, _Explainer(per_class))
    warnings = []

    importance = surrogate._shap_importance(_Model(), sample_frame, 50, warnings)

    assert importance.shape == (3,)
    assert list(importance) == [3.0, 3.0, 3.0]
    assert warnings == []


def test_a_list_of_the_wrong_rank_is_refused(monkeypatch, sample_frame):
    """A list of vectors is not one matrix per class."""
    _install(monkeypatch, _Explainer([np.ones(12), np.ones(12)]))
    warnings = []

    assert surrogate._shap_importance(_Model(), sample_frame, 50, warnings) is None
    assert warnings == ["unexpected list-shaped SHAP output; column omitted"]


def test_a_class_first_cube_is_moved_so_features_stay_on_axis_one(
        monkeypatch, sample_frame):
    """``(classes, rows, features)`` is normalised, not mis-averaged."""
    first = np.tile([1.0, 10.0, 100.0], (12, 1))
    second = np.tile([3.0, 30.0, 300.0], (12, 1))
    cube = np.stack([first, second])
    assert cube.shape == (2, 12, 3)
    _install(monkeypatch, _Explainer(cube))
    warnings = []

    result = surrogate._shap_importance(_Model(), sample_frame, 50, warnings,
                                        return_details=True)

    importance, signed, used = result
    assert list(importance) == [2.0, 20.0, 200.0]
    assert list(signed.columns) == ["a", "b", "c"]
    assert list(used.columns) == ["a", "b", "c"]
    assert used.shape == sample_frame.shape
    # predict_proba picks class 1 for every row, so the signed values are the
    # second slice rather than the first.
    assert signed.iloc[0].tolist() == [3.0, 30.0, 300.0]
    assert warnings == []


def test_a_cube_that_matches_no_axis_is_refused(monkeypatch, sample_frame):
    """An unrecognised shape is named rather than averaged into nonsense."""
    _install(monkeypatch, _Explainer(np.ones((5, 6, 7))))
    warnings = []

    assert surrogate._shap_importance(
        _Model(), sample_frame, 50, warnings) is None
    assert "unexpected SHAP output shape (5, 6, 7)" in warnings[0]


def test_a_model_without_probabilities_takes_the_first_class_slice(
        monkeypatch, sample_frame):
    """No predict_proba is not a failure; there is nothing to choose with."""
    cube = np.stack([np.full((12, 3), 1.0), np.full((12, 3), 3.0)], axis=2)
    assert cube.shape == (12, 3, 2)
    _install(monkeypatch, _Explainer(cube))
    warnings = []

    class _Bare:
        pass

    _importance, signed, _used = surrogate._shap_importance(
        _Bare(), sample_frame, 50, warnings, return_details=True)

    assert (signed.to_numpy() == 1.0).all()
    assert warnings == []


def test_probabilities_that_raise_fall_back_to_the_first_class_slice(
        monkeypatch, sample_frame):
    """The signed values still arrive when the model will not rank classes."""
    cube = np.stack([np.full((12, 3), 1.0), np.full((12, 3), 3.0)], axis=2)
    _install(monkeypatch, _Explainer(cube))

    class _Broken:
        def predict_proba(self, sample):
            raise RuntimeError("no probabilities from this one")

    warnings = []
    _importance, signed, _used = surrogate._shap_importance(
        _Broken(), sample_frame, 50, warnings, return_details=True)

    assert (signed.to_numpy() == 1.0).all()


def test_a_one_dimensional_shap_output_is_refused(monkeypatch, sample_frame):
    """A vector is neither per-object nor per-class, so it is dropped."""
    _install(monkeypatch, _Explainer(np.ones(3)))
    warnings = []

    assert surrogate._shap_importance(
        _Model(), sample_frame, 50, warnings) is None
    assert "unexpected SHAP output shape (3,)" in warnings[0]


def test_a_matrix_that_is_not_the_samples_shape_is_refused(monkeypatch,
                                                          sample_frame):
    """Two columns for three features is not a per-object explanation."""
    _install(monkeypatch, _Explainer(np.ones((12, 2))))
    warnings = []

    assert surrogate._shap_importance(
        _Model(), sample_frame, 50, warnings) is None
    assert "unexpected SHAP output shape (12, 2)" in warnings[0]


def test_a_capped_sample_is_named_and_the_ranking_still_covers_every_feature(
        monkeypatch, sample_frame):
    """SHAP is O(rows), so the cap is applied and written down."""
    _install(monkeypatch, _Explainer(lambda sample: np.ones(sample.shape)))
    warnings = []

    importance = surrogate._shap_importance(_Model(), sample_frame, 4, warnings)

    assert list(importance) == [1.0, 1.0, 1.0]
    assert "SHAP computed on 4 of 12 held-out objects" in warnings[0]


# ---------------------------------------------------------------------------
# the Explain CV module
# ---------------------------------------------------------------------------

def test_explain_cv_refuses_a_database_or_prediction_file_that_is_not_there(
        tmp_path):
    """Neither input is inferred, so both absences are named."""
    with pytest.raises(surrogate.SurrogateError,
                       match="no measurements database at"):
        surrogate.run_explain_cv({"db_path": str(tmp_path / "gone.db"),
                                  "predictions_file": str(tmp_path / "p.csv")})

    db = tmp_path / "measurements.db"
    db.write_bytes(b"")
    with pytest.raises(surrogate.SurrogateError,
                       match="no existing prediction CSV at"):
        surrogate.run_explain_cv({"db_path": str(db),
                                  "predictions_file": str(tmp_path / "p.csv")})


def test_explain_cv_writes_its_bundle_beside_the_predictions_by_default(
        tmp_path):
    """With no ``dst`` the artefacts land next to the file that was explained."""
    path, names, areas = _spacr_database(tmp_path, objects=140)
    predictions = pd.DataFrame({
        "path": names,
        "pred": [int(area > 500) for area in areas]})
    predictions_file = tmp_path / "cv_predictions.csv"
    predictions.to_csv(predictions_file, index=False)

    out = surrogate.run_explain_cv({
        "db_path": str(path),
        "predictions_file": str(predictions_file),
        "surrogate_n_estimators": 8,
        "surrogate_n_repeats": 1,
        "surrogate_shap_max_samples": 20,
        "verbose": False,
    })

    result = out["result"]
    assert isinstance(result, surrogate.SurrogateResult)
    assert result.n_objects == 140
    expected_root = str(tmp_path / "explain_cv_model")
    assert os.path.isdir(expected_root)
    assert out["paths"]["importance"] == os.path.join(
        expected_root, "feature_importance.csv")
    assert pd.read_csv(out["paths"]["importance"]).shape[0] >= 1


def test_registering_the_module_twice_does_not_replace_what_is_there():
    """Import already registered it; a second call is a no-op that says so."""
    from spacr.settings import has_registered_defaults

    assert has_registered_defaults(surrogate.APP_KEY)
    assert surrogate.register_explain_cv_settings() is False


# ---------------------------------------------------------------------------
# what the report says when the ranking is not to be trusted
# ---------------------------------------------------------------------------

def test_top_falls_back_to_a_column_that_was_measured():
    """Asked to sort by something absent, it uses the best one present."""
    result = surrogate.SurrogateResult(
        fidelity=0.9, baseline=0.5,
        importance=pd.DataFrame({"feature": ["a", "b", "c"],
                                 "shap": [0.1, 0.9, 0.5]}),
        n_objects=100)
    assert result.top(3, by="permutation")["feature"].tolist() == ["b", "c", "a"]


def test_an_unfaithful_summary_says_the_ranking_is_withheld():
    """A ranking of noise presented as a ranking is the failure to avoid."""
    result = surrogate.SurrogateResult(
        fidelity=0.50, baseline=0.52,
        importance=pd.DataFrame({"feature": ["a"], "permutation": [0.3]}),
        n_objects=100)
    text = result.summary(5)
    assert "does NOT reproduce the CV model" in text
    assert "Top 5 features: withheld" in text


def test_a_faithful_summary_prints_the_ranking():
    """When it reproduces the model, the features are the point of the report."""
    result = surrogate.SurrogateResult(
        fidelity=0.92, baseline=0.50,
        importance=pd.DataFrame({"feature": ["cell_area"],
                                 "permutation": [0.3]}),
        n_objects=100)
    text = result.summary(3)
    assert "Top 3 features:" in text
    assert "cell_area" in text


# ---------------------------------------------------------------------------
# building the frame
# ---------------------------------------------------------------------------

def test_a_predictions_frame_missing_its_columns_says_what_it_holds(tmp_path):
    """Naming the columns that ARE there is what identifies the wrong file."""
    path, _names, _areas = _spacr_database(tmp_path, objects=30)
    predictions = pd.DataFrame({"file": ["a.png"], "label": [1]})
    with pytest.raises(surrogate.SurrogateError, match="no 'path' column"):
        surrogate.build_surrogate_frame(path, predictions)
    with pytest.raises(surrogate.SurrogateError, match=r"\['file', 'label'\]"):
        surrogate.build_surrogate_frame(path, predictions)


def test_a_png_list_without_crop_paths_cannot_be_bridged(tmp_path):
    """``png_path`` is the only identity both sides of the join have."""
    path = str(tmp_path / "measurements.db")
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE png_list (file_name TEXT)")
    connection.execute("INSERT INTO png_list VALUES ('a.png')")
    connection.commit()
    connection.close()
    predictions = pd.DataFrame({"path": ["a.png"], "pred": [1]})
    with pytest.raises(surrogate.SurrogateError,
                       match="png_list has no png_path column"):
        surrogate.build_surrogate_frame(path, predictions)


def test_predictions_from_another_dataset_are_refused(tmp_path):
    """No crop matched, so the predictions describe a different experiment."""
    path, _names, _areas = _spacr_database(tmp_path, objects=30)
    predictions = pd.DataFrame({"path": ["from_another_screen.png"],
                                "pred": [1]})
    with pytest.raises(surrogate.SurrogateError,
                       match="probably made on a different dataset"):
        surrogate.build_surrogate_frame(path, predictions)


# ---------------------------------------------------------------------------
# which model families can run here
# ---------------------------------------------------------------------------

def test_every_model_family_is_listed_whether_or_not_it_can_run():
    """XGBoost stays visible when absent so the GUI can explain the greying."""
    families = surrogate.available_model_families()
    assert set(families) == set(surrogate.MODEL_FAMILIES)
    for key, entry in families.items():
        assert entry["label"]
        assert isinstance(entry["available"], bool)
        assert entry["reason"] == "" or "xgboost" in entry["reason"]


def test_a_model_family_nobody_offers_is_refused_by_name():
    """The refusal lists the fixed choices rather than picking one."""
    with pytest.raises(surrogate.SurrogateError, match="unknown surrogate model"):
        surrogate._make_estimator("magic_forest", n_estimators=4,
                                  random_seed=0, model_options={},
                                  y_train=pd.Series([0, 1]))


# ---------------------------------------------------------------------------
# what fit_surrogate refuses
# ---------------------------------------------------------------------------

def test_a_frame_with_no_cv_prediction_column_is_refused():
    """There is nothing to reproduce without the model's own answers."""
    with pytest.raises(surrogate.SurrogateError,
                       match="no cv_prediction column"):
        surrogate.fit_surrogate(pd.DataFrame({"cell_area": [1.0, 2.0]}),
                                **QUICK)


def test_a_frame_of_only_leaky_columns_leaves_nothing_to_fit():
    """A model's own scores predict its predictions and explain nothing."""
    frame = pd.DataFrame({
        "cv_prediction": [0, 1] * 15,
        "pred_probability": np.linspace(0, 1, 30),
    })
    with pytest.raises(surrogate.SurrogateError,
                       match="no usable numeric features"):
        surrogate.fit_surrogate(frame, **QUICK)


def test_excluding_the_only_driver_is_reported_and_still_fits():
    """The exclusion is honoured, and the leaky drop is said out loud."""
    frame = _driven_frame(120)
    frame["prediction_score"] = frame["cv_prediction"].astype(float)
    result = surrogate.fit_surrogate(frame, exclude=["noise_b"], **QUICK)
    assert "noise_b" not in result.importance["feature"].tolist()
    assert any("would leak the answer" in warning
               for warning in result.warnings)
    assert "prediction_score" not in result.importance["feature"].tolist()


def test_too_few_complete_objects_are_refused_with_the_count():
    """A surrogate fitted on a dozen objects is not worth reading."""
    frame = _driven_frame(15)
    with pytest.raises(surrogate.SurrogateError,
                       match="only 15 objects have complete features"):
        surrogate.fit_surrogate(frame, **QUICK)


def test_a_cv_model_that_chose_one_class_leaves_nothing_to_separate():
    """There is no boundary to explain when every object got the same label."""
    frame = _driven_frame(60)
    frame["cv_prediction"] = 0
    with pytest.raises(surrogate.SurrogateError,
                       match="every object in one class"):
        surrogate.fit_surrogate(frame, **QUICK)


# ---------------------------------------------------------------------------
# the optional XGBoost backend, when it really is installed
# ---------------------------------------------------------------------------

def test_an_xgboost_surrogate_reports_decoded_classes():
    """The estimator is fitted on encoded labels and reports the real ones."""
    pytest.importorskip("xgboost")
    frame = _driven_frame(120)
    frame["cv_prediction"] = frame["cv_prediction"].map({0: "neg", 1: "pos"})
    result = surrogate.fit_surrogate(frame, model_family="xgboost",
                                     n_estimators=6, n_repeats=1,
                                     shap_max_samples=20, verbose=False)
    assert result.model_family == "xgboost"
    assert result.backend.startswith("xgboost.")
    assert set(result.held_out["surrogate_prediction"]) <= {"neg", "pos"}
    assert {"probability_neg", "probability_pos"} <= set(
        result.held_out.columns)
    assert not result.importance.empty
