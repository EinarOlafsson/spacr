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


def case_regularised_model_rejects_unknown_kind():
    _seen = []
    with pytest.raises(ValueError, match="Unsupported regularised model: 'svm'") as _exc1:
        ml._regularised_linear_model("svm")
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_xgboost_cuda_build_is_false_when_xgboost_cannot_be_imported(monkeypatch):
    _seen = []
    monkeypatch.setitem(sys.modules, "xgboost", None)
    _seen.append(bool(
        ml._xgboost_cuda_build() is False))
    return _seen


def case_cuda_device_probe_counts_a_failed_torch_import_as_no_device(monkeypatch):
    _seen = []
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    _seen.append(bool(
        ml._cuda_device_usable() is False))
    return _seen


def case_class_arms_rejects_a_value_that_is_not_a_list():
    _seen = []
    with pytest.raises(ValueError, match="classes must be a list of class values") as _exc1:
        ml._class_arms(5)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_multiclass_frames_refuse_a_duplicated_location_column():
    _seen = []
    df = pd.DataFrame([["a", "a"], ["b", "b"]], columns=["loc", "loc"])
    with pytest.raises(ValueError, match="2 columns named 'loc'") as _exc1:
        ml._multiclass_frames(df, "loc", ["a", "b"], _equals)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_multiclass_frames_refuse_rows_claimed_by_two_classes():
    _seen = []
    df = pd.DataFrame({"loc": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="3 row\\(s\\) of 'loc' match more than one") as _exc1:
        ml._multiclass_frames(df, "loc", ["a", "b"],
                              lambda column, entry: column.notna())
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_multiclass_frames_name_an_empty_class_and_truncate_the_values_shown(capsys):
    _seen = []
    df = pd.DataFrame({"loc": [f"w{i:02d}" for i in range(20)]})
    with pytest.raises(ValueError) as caught:
        ml._multiclass_frames(df, "loc", ["w00", "missing"],
                              _equals, verbose=True)
    _seen.append(('ValueError' if isinstance(caught.value, ValueError) else type(caught.value).__name__))
    message = str(caught.value)
    _seen.append(("['missing']" if "['missing']" in (message) else ("missing", "['missing']")))
    _seen.append(('... (20 distinct values)' if '... (20 distinct values)' in (message) else ("missing", '... (20 distinct values)')))
    _seen.append(('Class 0 (w00): 1 samples' if 'Class 0 (w00): 1 samples' in (capsys.readouterr().out) else ("missing", 'Class 0 (w00): 1 samples')))
    return _seen


def case_unused_classes_are_named_and_the_confusion_matrix_is_printed(capsys):
    _seen = []
    frame = three_class_frame()
    extra = frame[frame["columnID"] == "c3"].copy()
    extra["columnID"] = "c4"
    extra.index = [f"plate1_r1_c4_f1_o{1000 + i}" for i in range(len(extra))]
    frame = pd.concat([frame, extra])
    output, _ = ml.ml_analysis(
        frame, model_type="logistic_regression", classes=["c1", "c2", "c3"],
        **{**COMMON, "verbose": True})
    printed = capsys.readouterr().out
    _seen.append(("1 class(es) of 'columnID' are not in the training set" if "1 class(es) of 'columnID' are not in the training set" in (printed) else ("missing", "1 class(es) of 'columnID' are not in the training set")))
    _seen.append(("['c4']" if "['c4']" in (printed) else ("missing", "['c4']")))
    _seen.append(('This fit has 3 classes' if 'This fit has 3 classes' in (printed) else ("missing", 'This fit has 3 classes')))
    _seen.append(('Confusion matrix (rows true class, columns predicted):' if 'Confusion matrix (rows true class, columns predicted):' in (printed) else ("missing", 'Confusion matrix (rows true class, columns predicted):')))
    _seen.append(('true_c1' if 'true_c1' in (printed) else ("missing", 'true_c1')))
    _seen.append(keep(output[3].spacr_class_labels_))
    return _seen


def case_generate_ml_scores_writes_the_multiclass_confusion_matrix(tmp_path, rng):
    _seen = []
    src = _make_src(tmp_path, "plate1", rng)
    settings = _ml_settings(src, ml_classes=["c1", "c2", "c3"])
    output, _ = ml.generate_ml_scores(settings)
    results = os.path.join(src, "results", "random_forest", "channel_1")
    path = os.path.join(results, "random_forest_model_confusion_matrix.csv")
    written = pd.read_csv(path, index_col=0)
    _seen.append(keep(list(written.index)))
    _seen.append(keep(list(written.columns)))
    _seen.append(bool(
        int(written.to_numpy().sum()) == int(output[3].spacr_confusion_matrix_.to_numpy().sum())))
    return _seen


from tests.observed_case import keep, observe  # noqa: E402

@pytest.mark.parametrize(("case", "expected"), [
    pytest.param(case_multiclass_frames_refuse_a_duplicated_location_column, ['ValueError'], id="multiclass_frames_refuse_a_duplicated_location_column"),
    pytest.param(case_multiclass_frames_refuse_rows_claimed_by_two_classes, ['ValueError'], id="multiclass_frames_refuse_rows_claimed_by_two_classes"),
    pytest.param(case_multiclass_frames_name_an_empty_class_and_truncate_the_values_shown, ['ValueError', "['missing']", '... (20 distinct values)', 'Class 0 (w00): 1 samples'], id="multiclass_frames_name_an_empty_class_and_truncate_the_values_shown"),
    pytest.param(case_unused_classes_are_named_and_the_confusion_matrix_is_printed, ["1 class(es) of 'columnID' are not in the training set", "['c4']", 'This fit has 3 classes', 'Confusion matrix (rows true class, columns predicted):', 'true_c1', ['c1', 'c2', 'c3']], id="unused_classes_are_named_and_the_confusion_matrix_is_printed"),
    pytest.param(case_generate_ml_scores_writes_the_multiclass_confusion_matrix, [['true_c1', 'true_c2', 'true_c3'], ['predicted_c1', 'predicted_c2', 'predicted_c3'], True], id="generate_ml_scores_writes_the_multiclass_confusion_matrix"),
    pytest.param(case_regularised_model_rejects_unknown_kind, ['ValueError'], id="regularised_model_rejects_unknown_kind"),
    pytest.param(case_xgboost_cuda_build_is_false_when_xgboost_cannot_be_imported, [True], id="xgboost_cuda_build_is_false_when_xgboost_cannot_be_imported"),
    pytest.param(case_cuda_device_probe_counts_a_failed_torch_import_as_no_device, [True], id="cuda_device_probe_counts_a_failed_torch_import_as_no_device"),
    pytest.param(case_class_arms_rejects_a_value_that_is_not_a_list, ['ValueError'], id="class_arms_rejects_a_value_that_is_not_a_list"),
])
def test_cifix3_ml_contracts(request, case, expected):
    """One contract per row: the case records what it checked, the row states it."""
    observed = observe(case, request)
    assert observed == expected
