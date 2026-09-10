"""A k-fold summary that says nothing about the classes is not a summary.

Instruction 236 B5 and D13, driven on a three-class dataset built from
plate1 of the tsg101 screen and trained on CPU.

WHAT WAS FOUND. `CV_METRIC_KEYS` was a fixed tuple: accuracy, loss, prauc,
neg_accuracy, pos_accuracy. Three of those five are binary-shaped, and the
epoch metrics set them to NaN on a multiclass run -- their own comments say
"not meaningful in multiclass". Anything NaN is dropped from the spread, so
a cross-validation over three classes summarised ACCURACY, LOSS AND PRAUC
and nothing about the classes at all.

Two things were missing and both already existed elsewhere in the module:

* `f1_macro` -- the metric that matters on an imbalanced screen, because
  accuracy is dominated by the majority class. Computed every epoch, never
  aggregated.
* the per-class accuracies, which live in the metrics dict as a LIST under
  'per_class_accuracy' that no spread statistic can aggregate.
  `attach_per_class_columns` is what turns them into one scalar column per
  class, and it had only ever been called on the way to the epoch CSVs.

The three-fold spread over plate1's c1/c2/c3 now shows a standard deviation
of 0.47 on one class's accuracy, which is the single most useful number in
the file and was not in it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.deep_spacr import (CV_METRIC_KEYS, PER_CLASS_ACC_PREFIX,
                              attach_per_class_columns, cv_metric_keys,
                              summarize_cv_metrics)


def _binary_metrics():
    return {"accuracy": 0.9, "loss": 0.3, "prauc": 0.88,
            "neg_accuracy": 0.87, "pos_accuracy": 0.93,
            "optimal_threshold": 0.51, "f1_macro": 0.9,
            "num_classes": 2, "epoch": 1,
            "per_class_accuracy": [0.87, 0.93], "class_support": [40, 40]}


def _multiclass_metrics():
    """What `compute_metrics` returns for three classes: the binary keys
    present and NaN, because they are not meaningful."""
    return {"accuracy": 0.4, "loss": 1.1, "prauc": 0.44,
            "neg_accuracy": np.nan, "pos_accuracy": np.nan,
            "optimal_threshold": np.nan, "f1_macro": 0.33,
            "num_classes": 3, "epoch": 1,
            "per_class_accuracy": [0.27, 0.54, 0.33],
            "class_support": [30, 30, 30]}


class TestWhichMetricsCross:
    def test_macro_f1_is_carried(self):
        """It was computed every epoch and aggregated never. On an
        imbalanced screen it is the metric accuracy hides."""
        assert "f1_macro" in CV_METRIC_KEYS

    def test_the_binary_names_are_kept_for_two_classes(self):
        """They cost nothing where they do not apply -- NaN is dropped from
        the spread -- and removing them would take a real number away from
        every binary run."""
        assert {"neg_accuracy", "pos_accuracy"} <= set(CV_METRIC_KEYS)

    def test_three_classes_bring_their_per_class_columns(self):
        metrics = attach_per_class_columns(_multiclass_metrics(),
                                           ["c1", "c2", "c3"])
        keys = cv_metric_keys(metrics)
        assert f"{PER_CLASS_ACC_PREFIX}c1" in keys
        assert f"{PER_CLASS_ACC_PREFIX}c2" in keys
        assert f"{PER_CLASS_ACC_PREFIX}c3" in keys

    def test_two_classes_still_get_theirs(self):
        metrics = attach_per_class_columns(_binary_metrics(), ["neg", "pos"])
        keys = cv_metric_keys(metrics)
        assert f"{PER_CLASS_ACC_PREFIX}neg" in keys
        assert f"{PER_CLASS_ACC_PREFIX}pos" in keys

    def test_it_asks_the_metrics_rather_than_a_list(self):
        """How many classes there are is the user's choice, so the columns
        cannot be enumerated in advance."""
        metrics = attach_per_class_columns(
            {"accuracy": 0.5, "per_class_accuracy": [0.1] * 7,
             "class_support": [10] * 7, "num_classes": 7},
            [f"class{i}" for i in range(7)])
        per_class = [k for k in cv_metric_keys(metrics)
                     if k.startswith(PER_CLASS_ACC_PREFIX)]
        assert len(per_class) == 7

    def test_the_prefix_is_the_modules_own(self):
        """A second constant spelling 'acc_class_' is exactly the
        redundancy this instruction is about."""
        assert PER_CLASS_ACC_PREFIX == "acc_class_"


class TestTheSpread:
    def _folds(self, metrics, names, n=3, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for fold in range(1, n + 1):
            one = attach_per_class_columns(dict(metrics), names)
            one = {k: (v + rng.normal(0, 0.05)
                       if isinstance(v, float) and np.isfinite(v) else v)
                   for k, v in one.items()}
            one["fold"] = fold
            rows.append(one)
        return pd.DataFrame(rows)

    def test_a_three_class_run_reports_its_classes(self):
        """THE DEFECT. Three rows -- accuracy, loss, prauc -- and nothing
        naming a class."""
        summary = summarize_cv_metrics(
            self._folds(_multiclass_metrics(), ["c1", "c2", "c3"]))
        reported = set(summary["metric"])
        assert {"acc_class_c1", "acc_class_c2", "acc_class_c3"} <= reported
        assert "f1_macro" in reported

    def test_the_binary_columns_are_dropped_when_they_are_nan(self):
        """They are NaN by construction in multiclass, and a row of NaNs in
        a spread table reads as a metric that failed rather than one that
        does not apply."""
        summary = summarize_cv_metrics(
            self._folds(_multiclass_metrics(), ["c1", "c2", "c3"]))
        assert "neg_accuracy" not in set(summary["metric"])

    def test_a_two_class_run_keeps_everything_it_had(self):
        summary = summarize_cv_metrics(
            self._folds(_binary_metrics(), ["neg", "pos"]))
        reported = set(summary["metric"])
        assert {"accuracy", "loss", "prauc", "neg_accuracy",
                "pos_accuracy"} <= reported

    def test_the_spread_is_what_k_fold_is_for(self):
        """A single split can be lucky; only the fold-to-fold standard
        deviation says by how much. On plate1's three classes it is 0.47 on
        one of them."""
        summary = summarize_cv_metrics(
            self._folds(_multiclass_metrics(), ["c1", "c2", "c3"]))
        for column in ("n_folds", "mean", "std", "min", "max", "range",
                       "cv_percent"):
            assert column in summary.columns

    def test_one_fold_reports_no_standard_deviation_rather_than_zero(self):
        """A std of 0 from a single sample would read as perfect
        stability."""
        summary = summarize_cv_metrics(
            self._folds(_binary_metrics(), ["neg", "pos"], n=1))
        assert summary["std"].isna().all()
