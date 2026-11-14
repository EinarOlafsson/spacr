"""Table-driven tests of the surrogate's edges: SHAP layouts, probability columns, blank figures.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture.  Three places in :mod:`spacr.surrogate` decide whether
a user is handed a number or nothing at all, and all fail quietly when wrong:

* ``_shap_importance`` normalises every SHAP output layout onto the feature
  axis, or refuses the layout with a stated reason;
* ``fit_surrogate`` writes one ``probability_<class>`` column per class only
  when the estimator returns a per-class matrix, and names a failure;
* ``write_surrogate_result`` draws the ranking and SHAP dependence panels only
  when there is something honest to draw, without losing the tables.

``run_explain_cv`` is driven end to end on a small spaCR-shaped database to
pin that ``dst`` is the folder written to.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

from spacr import surrogate

#: A fit small enough for a test and still large enough to split by well.
QUICK = {"n_estimators": 6, "n_repeats": 1, "shap_max_samples": 10,
         "verbose": False, "model_options": {"n_jobs": 1}}

_REAL_MAKE_ESTIMATOR = surrogate._make_estimator


# ---------------------------------------------------------------------------
# _shap_importance
# ---------------------------------------------------------------------------

class _FakeShap:
    """Stand-in for the optional ``shap`` dependency, returning ``values(sample)``."""

    def __init__(self, values):
        outer = self
        self._values = values

        class TreeExplainer:
            def __init__(self, model):
                self.model = model

            def shap_values(self, sample, check_additivity=True):
                return outer._values(sample)

        self.TreeExplainer = TreeExplainer


def test_shap_importance_puts_every_accepted_layout_on_the_feature_axis(subtests, monkeypatch):
    """2-D, both 3-D layouts and per-class lists give exact mean |SHAP|; other shapes are refused."""
    x_test = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0],
                           "b": [4.0, 3.0, 2.0, 1.0],
                           "c": [0.5, 0.5, 1.5, 2.5]})
    accepted = [
        ("2-D objects x features",
         lambda s: np.arange(s.size, dtype=float).reshape(s.shape), [4.5, 5.5, 6.5]),
        ("3-D classes last",
         lambda s: np.arange(s.size * 2, dtype=float).reshape(s.shape[0], s.shape[1], 2),
         [9.5, 11.5, 13.5]),
        ("3-D classes first",
         lambda s: np.arange(s.size * 2, dtype=float).reshape(2, s.shape[0], s.shape[1]),
         [10.5, 11.5, 12.5]),
        ("list per class",
         lambda s: [np.arange(s.size, dtype=float).reshape(s.shape), np.ones(s.shape)],
         [2.75, 3.25, 3.75]),
        ("signs do not cancel",
         lambda s: -np.arange(s.size, dtype=float).reshape(s.shape), [4.5, 5.5, 6.5]),
    ]
    for case_id, maker, expected in accepted:
        with subtests.test(msg=case_id):
            monkeypatch.setitem(sys.modules, "shap", _FakeShap(maker))
            warnings: list = []
            importance = surrogate._shap_importance(object(), x_test, 100, warnings)
            np.testing.assert_allclose(importance, expected)
            assert warnings == []

    refused = [
        ("2-D with an extra feature", lambda s: np.zeros((s.shape[0], s.shape[1] + 1)),
         "unexpected SHAP output shape (4, 4); column omitted"),
        ("3-D on neither axis", lambda s: np.zeros((2, 2, 2)),
         "unexpected SHAP output shape (2, 2, 2) for sample (4, 3); column omitted"),
        ("list of 3-D arrays", lambda s: [np.zeros((2, 2, 2))],
         "unexpected list-shaped SHAP output; column omitted"),
        ("empty list", lambda s: [], "unexpected list-shaped SHAP output; column omitted"),
        ("one-dimensional", lambda s: np.zeros(3),
         "unexpected SHAP output shape (3,); column omitted"),
    ]
    for case_id, maker, message in refused:
        with subtests.test(msg=f"refused: {case_id}"):
            monkeypatch.setitem(sys.modules, "shap", _FakeShap(maker))
            warnings = []
            assert surrogate._shap_importance(object(), x_test, 100, warnings) is None
            assert warnings == [message]


# ---------------------------------------------------------------------------
# fit_surrogate: held-out probabilities
# ---------------------------------------------------------------------------

def _driven_frame(n=90, seed=0):
    """Objects whose CV class is decided by ``cell_area`` alone."""
    rng = np.random.default_rng(seed)
    area = rng.normal(500, 120, n)
    frame = pd.DataFrame({
        "cell_area": area,
        "cell_channel_1_mean_intensity": rng.normal(0, 1, n),
        "noise_b": rng.normal(0, 1, n),
        "plateID": ["p1"] * n,
        "rowID": [f"r{(i // 12) + 1}" for i in range(n)],
        "columnID": [f"c{(i // 5) + 1}" for i in range(n)],
        "fieldID": [f"f{(i % 4) + 1}" for i in range(n)],
    })
    logit = 2.0 * (area - 500) / 120
    frame["cv_prediction"] = (logit + rng.normal(0, 0.2, n) > 0).astype(int)
    return frame


class _NoProbabilities:
    """A fitted estimator that predicts a class but will not score one (LinearSVC-like)."""

    def __init__(self, inner):
        self.inner = inner

    def fit(self, x, y):
        self.inner.fit(x, y)
        self.classes_ = self.inner.classes_
        self.feature_importances_ = self.inner.feature_importances_
        return self

    def predict(self, x):
        return self.inner.predict(x)

    def score(self, x, y):
        return self.inner.score(x, y)


class _FlatProbabilities(_NoProbabilities):
    """Scores only the positive class, as a vector rather than a matrix."""

    def predict_proba(self, x):
        return np.asarray(self.inner.predict_proba(x))[:, 1]


class _BrokenProbabilities(_NoProbabilities):
    """Has the method and raises from it."""

    def predict_proba(self, x):
        raise RuntimeError("this estimator cannot score classes")


def _wrapping(wrapper):
    """A ``_make_estimator`` that hands ``fit_surrogate`` ``wrapper`` of the real estimator."""

    def make(*args, **kwargs):
        model, fit_y, decode, backend = _REAL_MAKE_ESTIMATOR(*args, **kwargs)
        return wrapper(model), fit_y, decode, backend

    return make


def test_fit_surrogate_writes_probability_columns_only_from_a_per_class_matrix(subtests,
                                                                              monkeypatch):
    """No predict_proba or a bare vector costs the columns silently; a raising one is named."""
    frame = _driven_frame()
    scored = surrogate.fit_surrogate(frame, **QUICK)
    with subtests.test(msg="a real classifier scores both classes"):
        columns = [c for c in scored.held_out.columns if c.startswith("probability_")]
        assert columns == ["probability_0", "probability_1"]
        np.testing.assert_allclose(scored.held_out[columns].sum(axis=1), 1.0)

    cases = [
        ("no predict_proba", _NoProbabilities, []),
        ("a probability vector", _FlatProbabilities, []),
        ("a predict_proba that raises", _BrokenProbabilities,
         ["held-out probabilities unavailable: this estimator cannot score classes"]),
    ]
    for case_id, wrapper, named in cases:
        with subtests.test(msg=case_id):
            monkeypatch.setattr(surrogate, "_make_estimator", _wrapping(wrapper))
            result = surrogate.fit_surrogate(frame, **QUICK)
            held = result.held_out
            assert [c for c in held.columns if c.startswith("probability_")] == []
            assert [w for w in result.warnings if "probabilities unavailable" in w] == named
            # Everything else in the held-out table still arrives, unchanged.
            assert held["cv_prediction"].tolist() == scored.held_out["cv_prediction"].tolist()
            assert held["surrogate_prediction"].tolist() == \
                scored.held_out["surrogate_prediction"].tolist()
            assert result.fidelity == pytest.approx(scored.fidelity)
            assert "permutation" in result.importance.columns


# ---------------------------------------------------------------------------
# write_surrogate_result
# ---------------------------------------------------------------------------

def _hand_result(fidelity=0.91, baseline=0.50):
    """A finished result, small enough to write and draw in a test."""
    index = pd.Index([f"o{i}" for i in range(6)], name="object_index")
    importance = pd.DataFrame({
        "feature": ["cell_area", "noise_a"],
        "gain": [0.70, 0.30],
        "permutation": [0.52, 0.02],
        "shap": [0.61, 0.05],
        "feature_family": ["cell", "other"],
    })
    signed = pd.DataFrame({"cell_area": np.linspace(-1.0, 1.0, 6),
                           "noise_a": np.linspace(0.2, -0.2, 6)}, index=index)
    values = pd.DataFrame({"cell_area": np.linspace(400.0, 600.0, 6),
                           "noise_a": np.linspace(-1.0, 1.0, 6)}, index=index)
    return surrogate.SurrogateResult(
        fidelity=fidelity, baseline=baseline, importance=importance,
        n_objects=6, class_counts={0: 3, 1: 3},
        shap_values=signed, shap_feature_values=values,
        feature_columns=["cell_area", "noise_a"])


def _unmeasured():
    """A faithful result whose ranking carries no importance measure at all."""
    result = _hand_result()
    result.importance = pd.DataFrame({"feature": ["cell_area", "noise_a"],
                                      "feature_family": ["cell", "other"]})
    result.shap_values = pd.DataFrame()
    result.shap_feature_values = pd.DataFrame()
    return result


def _renamed():
    """A faithful result whose SHAP tables name features the ranking does not."""
    result = _hand_result()
    renamed = {"cell_area": "cell_area_um2", "noise_a": "noise_a_v2"}
    result.shap_values = result.shap_values.rename(columns=renamed)
    result.shap_feature_values = result.shap_feature_values.rename(columns=renamed)
    return result


def test_write_surrogate_result_keeps_its_tables_and_draws_only_honest_figures(subtests,
                                                                              tmp_path):
    """Faithful bundles get ranking + dependence; unfaithful, unmeasured or misnamed ones get less."""
    all_figures = ["importance_pdf", "importance_png",
                   "shap_dependence_pdf", "shap_dependence_png"]
    tables = ["feature_importance.csv", "held_out_shap_feature_values.csv",
              "held_out_signed_shap.csv", "manifest.json", "summary.txt"]
    cases = [
        ("faithful", _hand_result(), all_figures,
         tables + ["feature_importance.pdf", "feature_importance.png",
                   "shap_dependence.pdf", "shap_dependence.png"],
         {"importance_presented": True, "fidelity_improvement": 0.41}),
        ("fails the fidelity gate", _hand_result(fidelity=0.51), [], tables,
         {"importance_presented": False, "fidelity_improvement": 0.01}),
        ("nothing measured", _unmeasured(), [],
         ["feature_importance.csv", "manifest.json", "summary.txt"],
         {"importance_presented": True, "shap_object_indices": []}),
        ("SHAP tables name other features", _renamed(),
         ["importance_pdf", "importance_png"],
         tables + ["feature_importance.pdf", "feature_importance.png"],
         {"importance_presented": True}),
    ]
    for case_id, result, figures, files, manifest_fields in cases:
        with subtests.test(msg=case_id):
            out = tmp_path / case_id.replace(" ", "_")
            paths = surrogate.write_surrogate_result(result, str(out))
            assert sorted(k for k in paths
                          if k.startswith(("importance_", "shap_dependence"))) == figures
            for key in figures:
                assert os.path.isfile(paths[key])
            assert sorted(os.listdir(out)) == sorted(files)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            for key, value in manifest_fields.items():
                assert manifest[key] == pytest.approx(value), key
            assert pd.read_csv(paths["importance"])["feature"].tolist() == [
                "cell_area", "noise_a"]

    with subtests.test(msg="the unfaithful summary withholds the ranking"):
        summary = (tmp_path / "fails_the_fidelity_gate" / "summary.txt").read_text(
            encoding="utf-8")
        assert "The surrogate does NOT reproduce the CV model." in summary
        assert "Top 15 features: withheld" in summary

    with subtests.test(msg="renamed SHAP tables are still written as they are"):
        local = tmp_path / "SHAP_tables_name_other_features" / "held_out_signed_shap.csv"
        assert list(pd.read_csv(local).columns) == [
            "object_index", "cell_area_um2", "noise_a_v2"]


# ---------------------------------------------------------------------------
# run_explain_cv
# ---------------------------------------------------------------------------

def _spacr_database(where, objects=120, seed=1):
    """A database shaped the way spaCR writes one: ``prcfo`` in png_list."""
    rng = np.random.default_rng(seed)
    path = str(where / "measurements.db")
    rows, pngs = [], []
    for index in range(objects):
        row, column = f"r{index % 6 + 1}", f"c{index % 10 + 1}"
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
    connection.executemany("INSERT INTO png_list VALUES (?,?,?,?,?,?,?,?,?)", pngs)
    connection.commit()
    connection.close()
    return path, [png[1] for png in pngs], [row[5] for row in rows]


def test_run_explain_cv_writes_into_the_folder_it_was_given(subtests, tmp_path):
    """A chosen ``dst`` is obeyed and leaves the default empty; a blank one lands beside the CSV."""
    database, names, areas = _spacr_database(tmp_path)
    predictions_file = tmp_path / "cv_predictions.csv"
    pd.DataFrame({"path": names, "pred": [int(a > 500) for a in areas]}).to_csv(
        predictions_file, index=False)
    settings = {
        "db_path": str(database),
        "predictions_file": str(predictions_file),
        "surrogate_n_estimators": 6,
        "surrogate_n_repeats": 1,
        "surrogate_shap_max_samples": 10,
        "verbose": False,
    }
    chosen = tmp_path / "runs" / "seed_zero"
    default = tmp_path / "explain_cv_model"
    for case_id, extra, folder in (("chosen destination", {"dst": str(chosen)}, chosen),
                                   ("blank destination", {}, default)):
        with subtests.test(msg=case_id):
            existed = default.exists()
            out = surrogate.run_explain_cv(dict(settings, **extra))
            assert out["paths"]["manifest"] == str(folder / "manifest.json")
            assert out["paths"]["summary"] == str(folder / "summary.txt")
            assert (folder / "summary.txt").is_file()
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            assert manifest["n_objects"] == out["result"].n_objects == 120
            if folder is chosen:
                assert not existed and not default.exists()
