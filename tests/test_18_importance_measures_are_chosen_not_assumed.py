"""Item 288: item 18's selectable importance measures, one choice at a time.

Item 18 made the surrogate's three feature-importance measures -- native
gain, held-out permutation and SHAP -- selectable, and added KernelSHAP for
models TreeSHAP cannot read. ``test_surrogate.py`` pins the ranking with all
three on. This file pins what a CHOICE does:

* the measures can be named as a list, a comma- or space-separated string,
  or left to the default, and an unknown one is refused by name;
* a measure left out is not computed, and the table is sorted by what was;
* SHAP without the shap package costs the SHAP column and says why, and a
  run left with no measure at all still returns the features, unsorted,
  and writes no empty plot;
* KernelSHAP on a small table uses the rows themselves as the reference;
* an unknown SHAP explainer is refused.

The data plant one informative feature, so the right ranking is known.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from sklearn.linear_model import LogisticRegression  # noqa: E402

from spacr import surrogate  # noqa: E402


def _xy(n=120, seed=0):
    """``signal`` decides the class; ``noise_a`` and ``noise_b`` do not."""
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"signal": rng.normal(0, 1, n),
                      "noise_a": rng.normal(0, 1, n),
                      "noise_b": rng.normal(0, 1, n)})
    y = (x["signal"] + rng.normal(0, 0.1, n) > 0).astype(int)
    return x, y


def _frame(n=240, seed=0):
    """A surrogate frame whose CV prediction follows ``cell_area``."""
    rng = np.random.default_rng(seed)
    area = rng.normal(500, 120, n)
    frame = pd.DataFrame({
        "cell_area": area,
        "noise_a": rng.normal(0, 1, n),
        "plateID": ["p1"] * n,
        "rowID": [f"r{(i // 60) + 1}" for i in range(n)],
        "columnID": [f"c{(i // 12) + 1}" for i in range(n)],
        "fieldID": [f"f{(i % 4) + 1}" for i in range(n)],
    })
    frame["cv_prediction"] = ((area - 500) / 120
                              + rng.normal(0, 0.25, n) > 0).astype(int)
    return frame


@pytest.fixture
def forest():
    from sklearn.ensemble import RandomForestClassifier

    x, y = _xy()
    return RandomForestClassifier(n_estimators=25, random_state=0).fit(x, y)


@pytest.fixture
def no_shap(monkeypatch):
    monkeypatch.setitem(sys.modules, "shap", None)


def test_the_default_is_every_measure():
    assert surrogate._checked_importance_methods(None) == \
        list(surrogate.IMPORTANCE_METHODS)


def test_a_string_names_the_measures_in_the_canonical_order():
    assert surrogate._checked_importance_methods("shap, gain") == \
        ["gain", "shap"]
    assert surrogate._checked_importance_methods("permutation  gain") == \
        ["gain", "permutation"]


def test_an_unknown_measure_is_refused_by_name(forest):
    x, y = _xy()
    with pytest.raises(surrogate.SurrogateError) as excinfo:
        surrogate.rank_feature_importance(forest, x, y, methods=["lime"])
    assert "'lime'" in str(excinfo.value)


def test_default_methods_compute_all_three(forest):
    x, y = _xy()
    table, paths = surrogate.rank_feature_importance(forest, x, y,
                                                     methods=None,
                                                     n_repeats=2)
    assert {"gain", "permutation", "shap"} <= set(table.columns)
    assert table["feature"].iloc[0] == "signal"
    assert paths == {}


def test_gain_alone_is_not_permuted_or_explained(forest):
    x, y = _xy()
    table, _paths = surrogate.rank_feature_importance(forest, x, y,
                                                      methods="gain")
    assert list(table.columns) == ["rank", "feature", "gain"]
    assert table["feature"].iloc[0] == "signal"
    assert list(table["rank"]) == [1, 2, 3]


def test_without_shap_the_column_goes_and_the_reason_stays(
        forest, no_shap, tmp_path):
    """No measure left: the features are returned unsorted and no empty plot
    is drawn."""
    x, y = _xy()
    table, paths = surrogate.rank_feature_importance(
        forest, x, y, methods=["shap"], destination=str(tmp_path))
    assert list(table.columns) == ["rank", "feature"]
    assert list(table["feature"]) == ["signal", "noise_a", "noise_b"]
    assert any("shap is not installed" in w for w in table.attrs["warnings"])
    assert set(paths) == {"importance"}
    assert sorted(p.name for p in tmp_path.iterdir()) == \
        ["feature_importance.csv"]


def test_kernel_shap_on_a_small_table_uses_the_rows_as_reference():
    pytest.importorskip("shap")
    x, y = _xy(n=18)
    model = LogisticRegression().fit(x, y)
    table, _paths = surrogate.rank_feature_importance(
        model, x, y, methods=["shap"], shap_explainer="kernel")
    assert list(table.columns) == ["rank", "feature", "shap"]
    assert table["feature"].iloc[0] == "signal"
    assert (table["shap"] >= 0).all()


def test_an_unknown_shap_explainer_is_refused(forest):
    pytest.importorskip("shap")
    x, y = _xy()
    with pytest.raises(surrogate.SurrogateError) as excinfo:
        surrogate.rank_feature_importance(forest, x, y, methods=["shap"],
                                          shap_explainer="deep")
    assert "unknown SHAP explainer 'deep'" in str(excinfo.value)


def test_a_surrogate_asked_only_for_gain_ranks_by_gain():
    result = surrogate.fit_surrogate(_frame(), n_estimators=25,
                                     importance_methods=["gain"],
                                     verbose=False)
    assert "permutation" not in result.importance.columns
    assert "shap" not in result.importance.columns
    assert result.importance["feature"].iloc[0] == "cell_area"
    assert list(result.importance["gain"]) == \
        sorted(result.importance["gain"], reverse=True)


def test_a_surrogate_left_with_no_measure_still_lists_its_features(no_shap):
    result = surrogate.fit_surrogate(_frame(), n_estimators=25,
                                     importance_methods=["shap"],
                                     verbose=False)
    assert set(result.importance["feature"]) == {"cell_area", "noise_a"}
    assert not {"gain", "permutation", "shap"} & \
        set(result.importance.columns)
    assert result.family_importance.empty
    assert any("shap is not installed" in w for w in result.warnings)
