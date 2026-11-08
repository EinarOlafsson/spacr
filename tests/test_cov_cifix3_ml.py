"""Held behaviour for the XGBoost CPU fallback probes and the multiclass guards in spacr.ml.

The probes fall back to "not CUDA" when xgboost or torch cannot be
imported; the multiclass helpers refuse badly shaped class lists, duplicate
location columns, overlapping classes and classes with no rows; and a
three-class run writes its confusion matrix, names classes it never trained
on and prints the matrix when verbose.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import spacr.ml as ml
from tests.test_cov_ml_ml_scores import _make_src, _ml_settings
from tests.test_multiclass_ml_scores import COMMON, three_class_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _equals(column, entry):
    values = entry if isinstance(entry, list) else [entry]
    return column.isin(values)


def test_regularised_model_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Unsupported regularised model: 'svm'"):
        ml._regularised_linear_model("svm")


def test_xgboost_cuda_build_is_false_when_xgboost_cannot_be_imported(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgboost", None)
    assert ml._xgboost_cuda_build() is False


def test_cuda_device_probe_counts_a_failed_torch_import_as_no_device(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert ml._cuda_device_usable() is False


def test_class_arms_rejects_a_value_that_is_not_a_list():
    with pytest.raises(ValueError, match="classes must be a list of class values"):
        ml._class_arms(5)


def test_multiclass_frames_refuse_a_duplicated_location_column():
    df = pd.DataFrame([["a", "a"], ["b", "b"]], columns=["loc", "loc"])
    with pytest.raises(ValueError, match="2 columns named 'loc'"):
        ml._multiclass_frames(df, "loc", ["a", "b"], _equals)


def test_multiclass_frames_refuse_rows_claimed_by_two_classes():
    df = pd.DataFrame({"loc": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="3 row\\(s\\) of 'loc' match more than one"):
        ml._multiclass_frames(df, "loc", ["a", "b"],
                              lambda column, entry: column.notna())


def test_multiclass_frames_name_an_empty_class_and_truncate_the_values_shown(capsys):
    df = pd.DataFrame({"loc": [f"w{i:02d}" for i in range(20)]})
    with pytest.raises(ValueError) as caught:
        ml._multiclass_frames(df, "loc", ["w00", "missing"],
                              _equals, verbose=True)
    message = str(caught.value)
    assert "['missing']" in message
    assert "... (20 distinct values)" in message
    assert "Class 0 (w00): 1 samples" in capsys.readouterr().out


def test_unused_classes_are_named_and_the_confusion_matrix_is_printed(capsys):
    frame = three_class_frame()
    extra = frame[frame["columnID"] == "c3"].copy()
    extra["columnID"] = "c4"
    extra.index = [f"plate1_r1_c4_f1_o{1000 + i}" for i in range(len(extra))]
    frame = pd.concat([frame, extra])
    output, _ = ml.ml_analysis(
        frame, model_type="logistic_regression", classes=["c1", "c2", "c3"],
        **{**COMMON, "verbose": True})
    printed = capsys.readouterr().out
    assert "1 class(es) of 'columnID' are not in the training set" in printed
    assert "['c4']" in printed
    assert "This fit has 3 classes" in printed
    assert "Confusion matrix (rows true class, columns predicted):" in printed
    assert "true_c1" in printed
    assert output[3].spacr_class_labels_ == ["c1", "c2", "c3"]


def test_generate_ml_scores_writes_the_multiclass_confusion_matrix(tmp_path, rng):
    src = _make_src(tmp_path, "plate1", rng)
    settings = _ml_settings(src, ml_classes=["c1", "c2", "c3"])
    output, _ = ml.generate_ml_scores(settings)
    results = os.path.join(src, "results", "random_forest", "channel_1")
    path = os.path.join(results, "random_forest_model_confusion_matrix.csv")
    written = pd.read_csv(path, index_col=0)
    assert list(written.index) == ["true_c1", "true_c2", "true_c3"]
    assert list(written.columns) == ["predicted_c1", "predicted_c2", "predicted_c3"]
    assert int(written.to_numpy().sum()) == int(output[3].spacr_confusion_matrix_.to_numpy().sum())
