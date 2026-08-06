"""The surrogate model: what a CV classifier responded to, in feature units.

Activation maps say WHERE a model looked. A surrogate says WHAT it responded
to, by fitting a model to reproduce the CV model's predictions from measured
features and asking that model what it used.

The assertion that matters most is the one about FIDELITY. A surrogate that
cannot reproduce the CV model has explained nothing, and its feature ranking
is a ranking of how the features predict each other. That number is the one
people skip, so it is the one most heavily tested here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import surrogate


def _frame(n=400, seed=0, driver_strength=1.0):
    """Objects whose CV prediction is driven by ONE known feature.

    `cell_area` decides the class; `noise_*` do not. A surrogate worth
    anything ranks cell_area first, and that is checkable rather than
    impressionistic.
    """
    rng = np.random.default_rng(seed)
    area = rng.normal(500, 120, n)
    frame = pd.DataFrame({
        "cell_area": area,
        "noise_a": rng.normal(0, 1, n),
        "noise_b": rng.normal(0, 1, n),
        "noise_c": rng.normal(0, 1, n),
    })
    logit = driver_strength * (area - 500) / 120
    frame["cv_prediction"] = (logit + rng.normal(0, 0.25, n) > 0).astype(int)
    return frame


# ---------------------------------------------------------------------------
# Fidelity -- the caveat that decides whether the rest is worth reading
# ---------------------------------------------------------------------------

def test_a_faithful_surrogate_beats_the_baseline_and_finds_the_driver():
    result = surrogate.fit_surrogate(_frame(), verbose=False)
    assert result.is_faithful
    assert result.fidelity > result.baseline
    top = result.top(1, by="permutation")["feature"].tolist()
    assert top == ["cell_area"], result.importance


def test_an_unfaithful_surrogate_says_so_in_the_first_lines():
    """When the CV prediction is unrelated to every feature, the surrogate
    cannot reproduce it -- and the report must lead with that rather than
    with a ranking of noise."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({
        "noise_a": rng.normal(0, 1, 300),
        "noise_b": rng.normal(0, 1, 300),
        "noise_c": rng.normal(0, 1, 300),
    })
    frame["cv_prediction"] = rng.integers(0, 2, 300)

    result = surrogate.fit_surrogate(frame, verbose=False)
    assert not result.is_faithful, result.fidelity
    summary = result.summary()
    assert "does NOT reproduce" in summary
    assert "Do not" in summary
    # The warning comes BEFORE the feature table, or nobody reads it.
    assert summary.index("does NOT reproduce") < summary.index("Top ")


def test_the_baseline_is_the_majority_class_rate():
    """Fidelity means nothing without it: 0.9 is excellent on balanced
    classes and worthless when 90% of objects are one class."""
    frame = _frame(n=400)
    # Set BOTH sides explicitly: assigning only the first 360 left the rest
    # at whatever they were, so the majority was larger than intended and
    # the assertion was testing my arithmetic rather than the code.
    frame["cv_prediction"] = 0
    frame.loc[frame.index[-40:], "cv_prediction"] = 1
    result = surrogate.fit_surrogate(frame, verbose=False)
    assert result.baseline == pytest.approx(360 / 400, abs=0.02)
    assert result.class_counts == {0: 360, 1: 40}


# ---------------------------------------------------------------------------
# Leakage -- the way to make a surrogate look brilliant and mean nothing
# ---------------------------------------------------------------------------

def test_a_column_holding_the_answer_is_dropped_and_reported():
    frame = _frame()
    # The CV model's own score, copied into the feature table. It predicts
    # the prediction perfectly and explains none of it.
    frame["cell_prediction_score"] = frame["cv_prediction"].astype(float)
    result = surrogate.fit_surrogate(frame, verbose=False)

    assert "cell_prediction_score" not in set(result.importance["feature"])
    assert any("leak" in w for w in result.warnings), result.warnings


def test_identifier_columns_never_become_features():
    """A plate id is a perfect predictor of anything that varies by plate."""
    frame = _frame()
    frame["plateID"] = 1
    frame["object_label"] = np.arange(len(frame))
    result = surrogate.fit_surrogate(frame, verbose=False)
    features = set(result.importance["feature"])
    assert "plateID" not in features
    assert "object_label" not in features


def test_extra_exclusions_are_honoured():
    result = surrogate.fit_surrogate(
        _frame(), exclude=["noise_a"], verbose=False)
    assert "noise_a" not in set(result.importance["feature"])


# ---------------------------------------------------------------------------
# The three importances
# ---------------------------------------------------------------------------

def test_all_three_importances_are_reported_when_shap_is_available():
    shap = pytest.importorskip("shap")
    result = surrogate.fit_surrogate(_frame(n=200), verbose=False)
    for column in ("gain", "permutation", "shap"):
        assert column in result.importance.columns, result.importance.columns
    # They are separate measures, not copies of one another.
    assert not np.allclose(result.importance["gain"],
                           result.importance["permutation"])


def test_a_missing_shap_costs_the_column_not_the_analysis(monkeypatch):
    """Decoration must never be load-bearing: the most expensive of the
    three is optional, and its absence must not take the other two down."""
    monkeypatch.setattr(surrogate, "_shap_importance",
                        lambda *a, **k: a[-1].append("shap is not installed")
                        or None)
    result = surrogate.fit_surrogate(_frame(n=200), verbose=False)
    assert "gain" in result.importance.columns
    assert "permutation" in result.importance.columns
    assert "shap" not in result.importance.columns


def test_permutation_is_measured_on_held_out_data():
    """Measured on the training rows it rewards memorisation, and every
    feature looks important."""
    import inspect
    source = inspect.getsource(surrogate.fit_surrogate)
    assert "permutation_importance(" in source
    assert "x_test, y_test" in source, (
        "permutation importance must be computed on the held-out split")


def test_top_falls_back_rather_than_raising():
    """A run without SHAP still has to report something."""
    result = surrogate.fit_surrogate(_frame(n=200), verbose=False)
    assert not result.top(3, by="a_measure_that_does_not_exist").empty


# ---------------------------------------------------------------------------
# Refusals -- each would otherwise produce a confident, meaningless ranking
# ---------------------------------------------------------------------------

def test_one_class_is_refused():
    frame = _frame(n=100)
    frame["cv_prediction"] = 1
    with pytest.raises(surrogate.SurrogateError, match="one class"):
        surrogate.fit_surrogate(frame, verbose=False)


def test_too_few_objects_is_refused():
    with pytest.raises(surrogate.SurrogateError, match="too few"):
        surrogate.fit_surrogate(_frame(n=15), verbose=False)


def test_a_frame_with_no_predictions_is_refused():
    with pytest.raises(surrogate.SurrogateError, match="no cv_prediction"):
        surrogate.fit_surrogate(pd.DataFrame({"a": [1, 2, 3]}), verbose=False)


def test_no_usable_features_is_refused():
    frame = pd.DataFrame({
        "plateID": [1] * 60,
        "cell_prediction": np.arange(60, dtype=float),
        "cv_prediction": ([0] * 30) + ([1] * 30),
    })
    with pytest.raises(surrogate.SurrogateError, match="no usable numeric"):
        surrogate.fit_surrogate(frame, verbose=False)


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------

def test_predictions_that_match_no_crop_are_refused(tmp_path):
    """Silently fitting on an empty join would report a fidelity of nan and
    a ranking of nothing."""
    import sqlite3

    db = tmp_path / "measurements.db"
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE "png_list" (png_path TEXT, prcfo TEXT)')
    con.execute('INSERT INTO "png_list" VALUES ("/a/one.png", "p1")')
    con.commit(); con.close()

    preds = pd.DataFrame({"path": ["/b/other.png"], "pred": [1]})
    with pytest.raises(surrogate.SurrogateError, match="no crop in png_list"):
        surrogate.build_surrogate_frame(str(db), preds)


def test_a_predictions_frame_missing_its_columns_says_which(tmp_path):
    import sqlite3

    db = tmp_path / "measurements.db"
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE "png_list" (png_path TEXT, prcfo TEXT)')
    con.commit(); con.close()

    with pytest.raises(surrogate.SurrogateError, match="no 'pred' column"):
        surrogate.build_surrogate_frame(
            str(db), pd.DataFrame({"path": ["/a/one.png"]}))
