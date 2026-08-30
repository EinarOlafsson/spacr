"""The surrogate's last unguarded edges: probability columns and blank figures.

Two places in :mod:`spacr.surrogate` decide whether a user is handed a
number or nothing at all, and both fail quietly when they are wrong.

The first is the held-out table. Every row carries the CV class and the
surrogate's reproduction of it, and -- when the estimator can score classes
-- one ``probability_<class>`` column per class, which is what someone sorts
by to find the objects the surrogate was least sure about. An estimator with
no ``predict_proba``, or one whose probabilities arrive as a bare vector
rather than a per-class matrix, must cost those columns and nothing else:
the fidelity, the ranking and the rest of the table still have to arrive.

The second is :func:`spacr.surrogate.write_surrogate_result`. It draws the
ranking and the SHAP dependence panels only when there is something honest
to draw -- a surrogate that cleared the fidelity gate, a measured importance
column, SHAP tables that name the features the ranking names. Drawing anyway
would either publish a ranking of noise or ask matplotlib for zero subplots
and take the whole bundle down with it, losing the CSVs and the manifest
that were already written.
"""
from __future__ import annotations

import json
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import surrogate

#: A fit small enough for a test and still large enough to split by well.
QUICK = {"n_estimators": 6, "n_repeats": 1, "shap_max_samples": 10,
         "verbose": False, "model_options": {"n_jobs": 1}}

_REAL_MAKE_ESTIMATOR = surrogate._make_estimator


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
    """A fitted estimator that predicts a class but will not score one.

    Not hypothetical: margin classifiers (``LinearSVC``) and any wrapper
    around a hand-rolled model reach :func:`fit_surrogate` this way.
    """

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


def _wrapped_estimator(monkeypatch, wrapper):
    """Fit the real estimator, then hand ``fit_surrogate`` ``wrapper`` of it."""

    def make(*args, **kwargs):
        model, fit_y, decode, backend = _REAL_MAKE_ESTIMATOR(*args, **kwargs)
        return wrapper(model), fit_y, decode, backend

    monkeypatch.setattr(surrogate, "_make_estimator", make)


def _probability_columns(result):
    return [c for c in result.held_out.columns if c.startswith("probability_")]


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


def _figure_keys(paths):
    return sorted(key for key in paths
                  if key.startswith(("importance_", "shap_dependence")))


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
    connection.executemany(
        "INSERT INTO png_list VALUES (?,?,?,?,?,?,?,?,?)", pngs)
    connection.commit()
    connection.close()
    return path, [png[1] for png in pngs], [row[5] for row in rows]


# ---------------------------------------------------------------------------
# held-out probabilities
# ---------------------------------------------------------------------------

def test_held_out_probabilities_arrive_only_when_the_model_scores_classes(
        monkeypatch):
    """A model that cannot score classes costs the probability columns only.

    The per-class probabilities are how a user finds the objects the
    surrogate was least certain about, so they are written when the
    estimator can produce them. When it cannot, the run must still deliver
    the held-out table -- the CV class, the surrogate's reproduction of it
    and the identifiers that lead back to the crop -- rather than failing.
    """
    frame = _driven_frame()

    scored = surrogate.fit_surrogate(frame, **QUICK)
    assert _probability_columns(scored) == ["probability_0", "probability_1"]
    both = scored.held_out[["probability_0", "probability_1"]].to_numpy()
    assert np.allclose(both.sum(axis=1), 1.0)

    _wrapped_estimator(monkeypatch, _NoProbabilities)
    bare = surrogate.fit_surrogate(frame, **QUICK)

    assert _probability_columns(bare) == []
    assert bare.held_out["cv_prediction"].tolist() == \
        scored.held_out["cv_prediction"].tolist()
    assert bare.held_out["surrogate_prediction"].tolist() == \
        scored.held_out["surrogate_prediction"].tolist()
    assert bare.fidelity == pytest.approx(scored.fidelity)
    assert "permutation" in bare.importance.columns


def test_a_probability_vector_is_dropped_while_a_failing_one_is_named(
        monkeypatch):
    """Only a per-class matrix becomes columns, and a failure is written down.

    ``predict_proba`` has returned a bare P(class 1) vector from more than
    one wrapper. Writing it as ``probability_0`` would label the positive
    class's score with the negative class's name -- a mislabelled column
    nobody would question. It is dropped silently because nothing went
    wrong; an estimator that RAISES is a different matter, and that one is
    reported in the result's warnings where the reader will see it.
    """
    frame = _driven_frame()

    _wrapped_estimator(monkeypatch, _FlatProbabilities)
    flat = surrogate.fit_surrogate(frame, **QUICK)

    _wrapped_estimator(monkeypatch, _BrokenProbabilities)
    broken = surrogate.fit_surrogate(frame, **QUICK)

    monkeypatch.setattr(surrogate, "_make_estimator", _REAL_MAKE_ESTIMATOR)
    scored = surrogate.fit_surrogate(frame, **QUICK)

    assert _probability_columns(scored) == ["probability_0", "probability_1"]
    assert _probability_columns(flat) == []
    assert _probability_columns(broken) == []
    assert [w for w in flat.warnings if "probabilities unavailable" in w] == []
    assert [w for w in broken.warnings if "probabilities unavailable" in w] == [
        "held-out probabilities unavailable: this estimator cannot score classes"]
    assert flat.held_out["surrogate_prediction"].tolist() == \
        scored.held_out["surrogate_prediction"].tolist()


# ---------------------------------------------------------------------------
# what the bundle refuses to draw
# ---------------------------------------------------------------------------

def test_an_unfaithful_bundle_keeps_its_tables_and_draws_no_ranking(tmp_path):
    """A surrogate that failed the fidelity gate must not ship a bar chart.

    A figure is the part of a bundle that ends up in a slide deck, detached
    from the caveat in ``summary.txt``. A ranking drawn from a surrogate
    that scores at the majority-class baseline is a ranking of how the
    features predict each other, so it is never drawn -- while the tables
    and the manifest that record WHY are still written.
    """
    good = surrogate.write_surrogate_result(
        _hand_result(), str(tmp_path / "faithful"))
    assert _figure_keys(good) == ["importance_pdf", "importance_png",
                                  "shap_dependence_pdf", "shap_dependence_png"]
    assert (tmp_path / "faithful" / "feature_importance.png").is_file()

    weak = _hand_result(fidelity=0.51, baseline=0.50)
    paths = surrogate.write_surrogate_result(weak, str(tmp_path / "weak"))

    assert not weak.is_faithful
    assert _figure_keys(paths) == []
    assert sorted(os.listdir(tmp_path / "weak")) == [
        "feature_importance.csv", "held_out_shap_feature_values.csv",
        "held_out_signed_shap.csv", "manifest.json", "summary.txt"]
    manifest = json.loads((tmp_path / "weak" / "manifest.json").read_text())
    assert manifest["importance_presented"] is False
    assert manifest["fidelity_improvement"] == pytest.approx(0.01)
    summary = (tmp_path / "weak" / "summary.txt").read_text()
    assert "The surrogate does NOT reproduce the CV model." in summary
    assert "Top 15 features: withheld" in summary


def test_a_ranking_with_nothing_measured_is_written_but_not_drawn(tmp_path):
    """Zero measured importances must not become a zero-panel figure.

    ``plt.subplots(1, 0)`` raises, and it would raise here after the CSVs
    and the manifest had already been written -- the caller would get an
    exception instead of the bundle that is sitting complete on disk. A
    ranking with no gain, permutation or SHAP column is written as a table
    and left undrawn.
    """
    measured = surrogate.write_surrogate_result(
        _hand_result(), str(tmp_path / "measured"))
    assert "importance_png" in measured

    unmeasured = _hand_result()
    unmeasured.importance = pd.DataFrame({
        "feature": ["cell_area", "noise_a"],
        "feature_family": ["cell", "other"]})
    unmeasured.shap_values = pd.DataFrame()
    unmeasured.shap_feature_values = pd.DataFrame()

    paths = surrogate.write_surrogate_result(
        unmeasured, str(tmp_path / "unmeasured"))

    assert unmeasured.is_faithful
    assert _figure_keys(paths) == []
    assert pd.read_csv(paths["importance"])["feature"].tolist() == [
        "cell_area", "noise_a"]
    manifest = json.loads((tmp_path / "unmeasured" / "manifest.json").read_text())
    assert manifest["importance_presented"] is True
    assert manifest["shap_object_indices"] == []


def test_dependence_panels_are_skipped_when_the_shap_tables_name_others(
        tmp_path):
    """A SHAP table that names other features cannot be plotted against them.

    The dependence panels pair one feature's signed SHAP values with that
    same feature's measured values. If the ranking's top features are not
    columns of both SHAP tables -- a result assembled from tables that were
    renamed or filtered apart -- indexing them would raise a KeyError and
    lose the bundle. Nothing plottable means no dependence figure, and the
    ranking figure is still drawn.
    """
    aligned = surrogate.write_surrogate_result(
        _hand_result(), str(tmp_path / "aligned"))
    assert aligned["shap_dependence_pdf"] == str(
        tmp_path / "aligned" / "shap_dependence.pdf")
    assert os.path.isfile(aligned["shap_dependence_pdf"])

    renamed = {"cell_area": "cell_area_um2", "noise_a": "noise_a_v2"}
    mismatched = _hand_result()
    mismatched.shap_values = mismatched.shap_values.rename(columns=renamed)
    mismatched.shap_feature_values = \
        mismatched.shap_feature_values.rename(columns=renamed)

    paths = surrogate.write_surrogate_result(
        mismatched, str(tmp_path / "mismatched"))

    assert _figure_keys(paths) == ["importance_pdf", "importance_png"]
    assert not (tmp_path / "mismatched" / "shap_dependence.pdf").exists()
    assert list(pd.read_csv(paths["local_shap"]).columns) == [
        "object_index", "cell_area_um2", "noise_a_v2"]


# ---------------------------------------------------------------------------
# the Explain CV module's destination
# ---------------------------------------------------------------------------

def test_explain_cv_writes_into_the_folder_it_was_given(tmp_path):
    """``dst`` is obeyed, so two analyses of one screen do not overwrite.

    With ``dst`` blank the bundle lands beside the prediction CSV, which is
    the right default and the wrong answer when a second run -- another
    seed, another backend -- would then overwrite the first. A destination
    the user chose has to be the destination used, and nothing may be left
    in the default place.
    """
    database, names, areas = _spacr_database(tmp_path)
    predictions = pd.DataFrame({"path": names,
                               "pred": [int(area > 500) for area in areas]})
    predictions_file = tmp_path / "cv_predictions.csv"
    predictions.to_csv(predictions_file, index=False)
    settings = {
        "db_path": str(database),
        "predictions_file": str(predictions_file),
        "surrogate_n_estimators": 6,
        "surrogate_n_repeats": 1,
        "surrogate_shap_max_samples": 10,
        "verbose": False,
    }
    chosen = tmp_path / "runs" / "seed_zero"

    out = surrogate.run_explain_cv(dict(settings, dst=str(chosen)))

    assert out["paths"]["manifest"] == str(chosen / "manifest.json")
    assert out["paths"]["summary"] == str(chosen / "summary.txt")
    manifest = json.loads((chosen / "manifest.json").read_text())
    assert manifest["n_objects"] == out["result"].n_objects == 120
    assert not (tmp_path / "explain_cv_model").exists()

    default = surrogate.run_explain_cv(settings)

    assert default["paths"]["manifest"] == str(
        tmp_path / "explain_cv_model" / "manifest.json")
    assert (tmp_path / "explain_cv_model" / "summary.txt").is_file()
