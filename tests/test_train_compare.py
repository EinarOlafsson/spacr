"""Training-run comparison — :mod:`spacr.train_compare`.

Every run folder here is built the way :func:`spacr.io._save_progress` builds
one: ``train.csv`` / ``validation.csv`` written with a pandas index column and
a header on the first epoch only, appended to one row at a time, in a
``<src>/model/<model_type>/<channels>/epochs_<N>`` tree with the settings
snapshot in ``<src>/settings/train_test_<model_type>_<N>.csv``. Cross-validated
runs get the ``fold_<i>/`` subfolders
:func:`spacr.deep_spacr._cross_validate_model` writes.

The properties pinned here are the ones that decide whether the dashboard
informs or misleads:

* runs of different length are drawn at their own lengths — never truncated to
  the shortest or padded to the longest;
* a k-fold run keeps its folds, and a fold mean is labelled as a mean;
* two identical runs say "no differences" in words, not with a blank table;
* environment drift is bucketed away from the knobs the user turned;
* both best-epoch and last-epoch are reported, because best-epoch on a
  validation curve is optimistically biased;
* a folder with no curves / no settings / a zero-epoch log is reported and does
  not stop the other runs being compared;
* every series label carries run, split and fold;
* the module never imports torch.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr.train_compare import (
    Comparison,
    TrainingRun,
    available_metrics,
    compare_runs,
    diff_settings,
    find_runs,
    format_comparison,
    is_env_key,
    load_run,
    metric_direction,
    plot_curves,
)


# ---------------------------------------------------------------------------
# Synthetic run folders, built exactly the way training writes them
# ---------------------------------------------------------------------------

def _rows(accuracies, losses=None):
    """One dict per epoch, shaped like an ``evaluate_model_performance`` dict."""
    out = []
    for i, acc in enumerate(accuracies, start=1):
        loss = (losses[i - 1] if losses is not None else 1.0 / i)
        out.append({
            "accuracy": float(acc),
            "neg_accuracy": float(acc) - 0.02,
            "pos_accuracy": float(acc) + 0.02,
            "prauc": float(acc) * 0.9,
            "optimal_threshold": 0.5,
            "loss": float(loss),
            "epoch": i,
            # `evaluate_model_performance` writes this straight copy of
            # `accuracy`; the reader must not offer it as a second metric.
            "Accuracy": float(acc),
        })
    return out


def write_curve(folder, split, accuracies, losses=None):
    """Write one progress CSV the way ``spacr.io._save_progress`` does.

    That matters: the accumulators are cleared every epoch, so each append is a
    one-row frame whose index is 0 — the index column of a real ``train.csv``
    is ``0`` on every line and only the ``epoch`` column is usable.
    """
    os.makedirs(folder, exist_ok=True)
    df = pd.DataFrame(_rows(accuracies, losses))
    path = os.path.join(folder,
                        "train.csv" if split == "train" else "validation.csv")
    for i in range(len(df)):
        one = df.iloc[[i]].reset_index(drop=True)
        if i == 0:
            with open(path, "w") as f:
                one.to_csv(f, index=True, header=True)
        else:
            with open(path, "a") as f:
                one.to_csv(f, index=True, header=False)
    return path


def write_settings(src, model_type, epochs, settings):
    """Write a ``Key,Value`` settings CSV where ``save_settings`` writes it."""
    folder = os.path.join(src, "settings")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"train_test_{model_type}_{epochs}.csv")
    pd.DataFrame(list(settings.items()),
                 columns=["Key", "Value"]).to_csv(path, index=False)
    return path


@pytest.fixture(autouse=True)
def _contain_the_settings_climb(monkeypatch):
    """Keep load_run's search for ``settings/`` inside the test's own tmp tree.

    load_run climbs ``_SETTINGS_SEARCH_DEPTH`` (6) parents looking for the
    folder ``save_settings`` writes to. ``make_run`` puts ``dst`` five levels
    below the root it is handed, so at the shipped depth the climb reaches the
    root's PARENT — shared ground under /tmp that other tests write into, and
    several of them create a ``settings/`` folder there. The "no settings
    found" tests then pass in isolation and fail in a full-suite run depending
    on which test ran first.

    Clamping to 5 keeps every level these tests actually exercise (``dst`` up
    to the run root) and stops one short of the shared parent. The product
    default is untouched; this only bounds it here.
    """
    import spacr.train_compare as _tc
    monkeypatch.setattr(_tc, "_SETTINGS_SEARCH_DEPTH", 5, raising=True)


def make_run(root, name, model_type="maxvit_t", epochs=10, channels="rgb",
             train=None, val=None, settings=None, folds=None):
    """Build one run folder under ``root/<name>`` and return its ``dst`` path.

    :param train: train accuracies, one per epoch (``None`` writes no file).
    :param val: validation accuracies (``None`` writes no file).
    :param folds: ``{fold_number: (train_accs, val_accs)}`` for a k-fold run;
        when given, ``train``/``val`` are ignored.
    """
    src = os.path.join(root, name)
    dst = os.path.join(src, "model", model_type, channels, f"epochs_{epochs}")
    os.makedirs(dst, exist_ok=True)
    if folds:
        for number, (tr, va) in sorted(folds.items()):
            fold_dir = os.path.join(dst, f"fold_{number}")
            if tr is not None:
                write_curve(fold_dir, "train", tr)
            if va is not None:
                write_curve(fold_dir, "val", va)
    else:
        if train is not None:
            write_curve(dst, "train", train)
        if val is not None:
            write_curve(dst, "val", val)
    if settings is not None:
        write_settings(src, model_type, epochs, settings)
    return dst


BASE_SETTINGS = {
    "src": "/data/screen",
    "model_type": "maxvit_t",
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 0.0001,
    "loss_type": "focal_loss",
    "augment": True,
    "classes": "['nc', 'pc']",
}


@pytest.fixture
def two_runs(tmp_path):
    """Two ordinary runs of different length: 10 epochs and 25."""
    root = str(tmp_path)
    a = make_run(root, "dsA", epochs=10,
                 train=np.linspace(0.50, 0.80, 10),
                 val=np.linspace(0.48, 0.74, 10),
                 settings={**BASE_SETTINGS, "epochs": 10})
    b = make_run(root, "dsB", epochs=25,
                 train=np.linspace(0.50, 0.93, 25),
                 val=np.linspace(0.48, 0.86, 25),
                 settings={**BASE_SETTINGS, "epochs": 25,
                           "learning_rate": 0.001})
    return root, load_run(a, run_id="A"), load_run(b, run_id="B")


# ---------------------------------------------------------------------------
# Reading what training actually writes
# ---------------------------------------------------------------------------

def test_load_run_reads_the_progress_csvs_training_writes(two_runs):
    _root, a, _b = two_runs
    assert a.has_curves
    assert a.n_epochs == 10
    # Both splits present, ten epochs each, epochs numbered 1..10 from the
    # `epoch` column (the CSV's index column is 0 on every row).
    assert sorted(set(a.curves["split"])) == ["train", "val"]
    for split in ("train", "val"):
        block = a.curves[a.curves["split"] == split]
        assert list(block["epoch"]) == [float(e) for e in range(1, 11)]


def test_duplicate_accuracy_column_is_not_offered_as_a_second_metric(two_runs):
    _root, a, _b = two_runs
    metrics = a.metrics()
    assert "accuracy" in metrics
    assert "Accuracy" not in metrics
    assert metrics.count("accuracy") == 1


def test_settings_are_recovered_from_the_dataset_settings_folder(two_runs):
    _root, a, _b = two_runs
    assert a.settings["model_type"] == "maxvit_t"
    assert a.settings_path.endswith("train_test_maxvit_t_10.csv")
    assert a.notes == []


def test_find_runs_returns_each_run_once_and_never_a_fold_folder(tmp_path):
    root = str(tmp_path)
    make_run(root, "dsA", epochs=10, train=[0.5] * 10, val=[0.5] * 10,
             settings=BASE_SETTINGS)
    make_run(root, "dsC", model_type="resnet50", epochs=6,
             folds={1: ([0.5] * 6, [0.5] * 6),
                    2: ([0.5] * 6, [0.5] * 6)},
             settings=BASE_SETTINGS)
    runs = find_runs(root)
    assert len(runs) == 2
    assert not any("fold_" in r.run_id for r in runs)
    cv = next(r for r in runs if r.is_cv)
    assert cv.folds == ["fold_1", "fold_2"]


# ---------------------------------------------------------------------------
# Different epoch counts: no truncation, no padding
# ---------------------------------------------------------------------------

def test_runs_of_different_length_overlay_at_their_own_lengths(two_runs):
    _root, a, b = two_runs
    comparison = compare_runs([a, b])
    lengths = {s.label: s.n_epochs for s in comparison.series}
    assert lengths == {"A · train": 10, "A · val": 10,
                       "B · train": 25, "B · val": 25}
    assert comparison.lengths_differ()

    fig = plot_curves(comparison, "accuracy")
    try:
        ax = fig.axes[0]
        drawn = {line.get_label(): len(line.get_xdata()) for line in ax.lines}
        # Not truncated to 10 and not padded to 25 — each line is its own run.
        assert drawn == {"A · train": 10, "A · val": 10,
                         "B · train": 25, "B · val": 25}
        # And no NaN padding snuck in at the tail of the short run.
        short = next(l for l in ax.lines if l.get_label() == "A · val")
        assert np.isfinite(np.asarray(short.get_ydata(), dtype=float)).all()
        assert float(np.max(short.get_xdata())) == 10.0
    finally:
        fig.clf()


def test_report_says_the_runs_are_different_lengths(two_runs):
    _root, a, b = two_runs
    text = format_comparison(compare_runs([a, b]), "accuracy")
    assert "different epoch counts" in text
    assert "truncated or padded" in text
    assert "1–10" in text and "1–25" in text


# ---------------------------------------------------------------------------
# k-fold
# ---------------------------------------------------------------------------

@pytest.fixture
def cv_run(tmp_path):
    """A 3-fold run whose third fold early-stopped at epoch 5 of 8."""
    dst = make_run(str(tmp_path), "dsCV", model_type="resnet50", epochs=8,
                   folds={
                       1: (np.linspace(0.50, 0.80, 8), np.linspace(0.48, 0.72, 8)),
                       2: (np.linspace(0.50, 0.84, 8), np.linspace(0.48, 0.78, 8)),
                       3: (np.linspace(0.50, 0.70, 5), np.linspace(0.48, 0.60, 5)),
                   },
                   settings={**BASE_SETTINGS, "model_type": "resnet50",
                             "cross_validation_folds": 3})
    return load_run(dst, run_id="CV")


def test_kfold_run_yields_one_series_per_fold(cv_run):
    comparison = compare_runs([cv_run], folds="per_fold")
    labels = comparison.labels()
    assert labels == [
        "CV · train · fold 1", "CV · train · fold 2", "CV · train · fold 3",
        "CV · val · fold 1", "CV · val · fold 2", "CV · val · fold 3",
    ]
    # The fold that stopped early keeps its own length.
    assert comparison.series_for("CV · val · fold 3").n_epochs == 5
    assert comparison.series_for("CV · val · fold 1").n_epochs == 8


def test_fold_mean_is_labelled_a_mean_and_carries_its_spread(cv_run):
    comparison = compare_runs([cv_run], folds="mean")
    labels = comparison.labels()
    assert labels == ["CV · train · mean of 3 folds ±sd",
                      "CV · val · mean of 3 folds ±sd"]
    mean = comparison.series_for("CV · val · mean of 3 folds ±sd")
    assert mean.kind == "mean"
    assert "mean" in mean.label
    sd = mean.sd("accuracy")
    assert sd is not None and np.isfinite(sd).any()
    # Epoch 1: mean of the three folds' first values.
    assert mean.values("accuracy")[0] == pytest.approx(0.48)


def test_fold_mean_says_when_its_tail_averages_fewer_folds(cv_run):
    comparison = compare_runs([cv_run], folds="mean")
    mean = comparison.series_for("CV · val · mean of 3 folds ±sd")
    assert mean.support() == (2, 3)
    assert mean.support_drops_at() == 6
    text = format_comparison(comparison, "accuracy")
    assert "only 2 of 3 folds reach that far" in text
    assert "average across folds, not a single run" in text


def test_folds_both_draws_folds_and_the_mean(cv_run):
    comparison = compare_runs([cv_run], folds="both")
    labels = comparison.labels()
    assert "CV · val · fold 1" in labels
    assert "CV · val · mean of 3 folds ±sd" in labels
    assert comparison.fold_mode == "both"
    fig = plot_curves(comparison, "accuracy")
    try:
        assert len(fig.axes[0].lines) == len(comparison.series_with("accuracy"))
        assert "shaded band is ±1 sd across folds" in _axes_notes(fig.axes[0])
    finally:
        fig.clf()


def test_unknown_fold_mode_is_refused(cv_run):
    with pytest.raises(ValueError, match="folds must be one of"):
        compare_runs([cv_run], folds="average")


def _axes_notes(ax) -> str:
    return "\n".join(t.get_text() for t in ax.texts)


# ---------------------------------------------------------------------------
# Series labels
# ---------------------------------------------------------------------------

def test_every_series_label_carries_run_split_and_fold(cv_run, two_runs):
    _root, a, _b = two_runs
    comparison = compare_runs([cv_run, a], folds="both")
    for series in comparison.series:
        assert series.run_id in series.label
        assert series.split in series.label
        if series.kind == "fold":
            assert series.fold.replace("_", " ") in series.label
        if series.kind == "mean":
            assert "mean of 3 folds" in series.label
    # Nothing is labelled with the run alone — that would invite reading a
    # train curve as a held-out result.
    assert all(label not in ("CV", "A") for label in comparison.labels())


def test_plot_annotates_an_axes_holding_both_splits(two_runs):
    _root, a, b = two_runs
    fig = plot_curves(compare_runs([a, b]), "accuracy")
    try:
        notes = _axes_notes(fig.axes[0])
        assert "mixes train" in notes and "validation" in notes
        assert "overfitting" in notes
    finally:
        fig.clf()


def test_plot_exposes_the_series_behind_each_line(two_runs):
    _root, a, b = two_runs
    comparison = compare_runs([a, b])
    fig = plot_curves(comparison, "accuracy")
    try:
        mapping = fig.spacr_series_by_label
        assert set(mapping) == set(comparison.labels())
        assert mapping["B · val"].run_id == "B"
        assert mapping["B · val"].split == "val"
    finally:
        fig.clf()


def test_plot_of_a_metric_nobody_logged_says_so_instead_of_raising(two_runs):
    _root, a, b = two_runs
    fig = plot_curves(compare_runs([a, b]), "f1_macro")
    try:
        assert "no selected run logged 'f1_macro'" in _axes_notes(fig.axes[0])
        assert fig.spacr_series_by_label == {}
    finally:
        fig.clf()


# ---------------------------------------------------------------------------
# Settings diff — identical, changed, environment drift, schema drift
# ---------------------------------------------------------------------------

def test_identical_settings_report_no_differences_in_words(tmp_path):
    root = str(tmp_path)
    same = dict(BASE_SETTINGS)
    a = load_run(make_run(root, "dsA", epochs=10, train=[0.5] * 10,
                          settings=same), run_id="A")
    b = load_run(make_run(root, "dsB", epochs=10, train=[0.6] * 10,
                          settings=same), run_id="B")
    diff = compare_runs([a, b]).settings_diff
    assert diff["identical"] is True
    assert diff["changed"] == []
    assert diff["env"] == []
    assert diff["drift"] == []
    assert diff["same"] == len(same)

    text = format_comparison(compare_runs([a, b]), "accuracy")
    assert "Settings: no differences" in text
    assert "identical settings" in text


def test_changed_settings_are_the_signal(two_runs):
    _root, a, b = two_runs
    diff = compare_runs([a, b]).settings_diff
    keys = [c["key"] for c in diff["changed"]]
    assert keys == ["epochs", "learning_rate"]
    lr = next(c for c in diff["changed"] if c["key"] == "learning_rate")
    assert lr["values"] == {"A": "0.0001", "B": "0.001"}
    assert diff["identical"] is False


def test_environment_only_differences_do_not_count_as_changed(tmp_path):
    root = str(tmp_path)
    base = dict(BASE_SETTINGS)
    a = load_run(make_run(root, "dsA", epochs=10, train=[0.5] * 10,
                          settings={**base, "n_jobs": 30, "dst": "/box1/out",
                                    "start_time": "2026-01-01T09:00"}),
                 run_id="A")
    b = load_run(make_run(root, "dsB", epochs=10, train=[0.5] * 10,
                          settings={**base, "n_jobs": 8, "dst": "/box2/out",
                                    "start_time": "2026-02-02T17:30"}),
                 run_id="B")
    diff = compare_runs([a, b]).settings_diff
    assert diff["changed"] == [], "environment drift leaked into 'changed'"
    assert sorted(e["key"] for e in diff["env"]) == ["dst", "n_jobs",
                                                     "start_time"]
    assert diff["identical"] is False
    text = format_comparison(compare_runs([a, b]), "accuracy")
    assert "Settings changed (0 of" in text
    assert "Environment drift (3)" in text


def test_env_classification_is_token_wise_not_substring():
    # `start_time` is drift; `update_freq` contains "date" as a substring and
    # must not be mistaken for one. `src` is a path but a different dataset is
    # a real difference, so it stays in `changed`.
    assert is_env_key("start_time")
    assert is_env_key("torch_version")
    assert is_env_key("n_jobs")
    assert not is_env_key("update_freq")
    assert not is_env_key("src")
    assert not is_env_key("learning_rate")
    assert is_env_key("my_knob", env_keys=["my_knob"])


def test_schema_drift_is_bucketed_separately_from_changed(tmp_path):
    root = str(tmp_path)
    a = load_run(make_run(root, "dsA", epochs=10, train=[0.5] * 10,
                          settings={**BASE_SETTINGS, "old_option": 1}),
                 run_id="A")
    b = load_run(make_run(root, "dsB", epochs=10, train=[0.5] * 10,
                          settings={**BASE_SETTINGS, "new_option": 2}),
                 run_id="B")
    diff = compare_runs([a, b]).settings_diff
    assert [d["key"] for d in diff["drift"]] == ["new_option", "old_option"]
    assert diff["changed"] == []
    drift = {d["key"]: d for d in diff["drift"]}
    assert drift["old_option"]["present"] == ["A"]
    assert drift["old_option"]["missing"] == ["B"]
    text = format_comparison(compare_runs([a, b]), "accuracy")
    assert "Schema drift: 2 key(s) missing" in text


def test_values_are_compared_structurally_not_by_repr(tmp_path):
    # `classes` round-trips through the settings CSV as the string
    # "['nc', 'pc']"; a live dict holds the list. Those are the same setting.
    root = str(tmp_path)
    a = load_run(make_run(root, "dsA", epochs=10, train=[0.5] * 10,
                          settings=BASE_SETTINGS), run_id="A")
    b = TrainingRun(run_id="B", path=a.path,
                    settings={**BASE_SETTINGS, "classes": ["nc", "pc"]})
    diff = diff_settings([a, b])
    assert [c["key"] for c in diff["changed"]] == []
    assert diff["identical"] is True


def test_a_diff_needs_two_runs_with_settings(tmp_path):
    root = str(tmp_path)
    a = load_run(make_run(root, "dsA", epochs=10, train=[0.5] * 10,
                          settings=BASE_SETTINGS), run_id="A")
    b = load_run(make_run(root, "dsB", epochs=10, train=[0.5] * 10),
                 run_id="B")
    diff = compare_runs([a, b]).settings_diff
    assert diff["no_settings"] == ["B"]
    assert diff["identical"] is False
    assert diff["changed"] == []
    text = format_comparison(compare_runs([a, b]), "accuracy")
    assert "fewer than two runs have a settings snapshot" in text


# ---------------------------------------------------------------------------
# Best epoch vs last epoch
# ---------------------------------------------------------------------------

def test_best_and_last_differ_on_a_non_monotonic_curve(tmp_path):
    # Validation accuracy peaks at epoch 3 and then degrades — the classic
    # shape that makes "best" and "last" different numbers.
    dst = make_run(str(tmp_path), "dsA", epochs=5,
                   train=[0.5, 0.6, 0.7, 0.8, 0.9],
                   val=[0.50, 0.70, 0.90, 0.72, 0.60],
                   settings=BASE_SETTINGS)
    run = load_run(dst, run_id="A")
    series = compare_runs([run]).series_for("A · val")

    best = series.best("accuracy")
    last = series.last("accuracy")
    assert best == {"epoch": 3, "value": pytest.approx(0.90), "direction": "max"}
    assert last == {"epoch": 5, "value": pytest.approx(0.60)}
    assert best["value"] != last["value"]

    entry = run.final_metrics["A · val"]
    assert entry["best"]["accuracy"]["epoch"] == 3
    assert entry["last"]["accuracy"]["epoch"] == 5

    text = format_comparison(compare_runs([run]), "accuracy")
    assert "0.9000" in text and "0.6000" in text
    assert "optimistically biased" in text
    assert "'last' is unbiased" in text


def test_loss_is_minimised_and_directionless_metrics_have_no_best(tmp_path):
    dst = make_run(str(tmp_path), "dsA", epochs=4,
                   train=[0.5, 0.6, 0.7, 0.8],
                   val=[0.5, 0.6, 0.7, 0.8],
                   settings=BASE_SETTINGS)
    series = compare_runs([load_run(dst, run_id="A")]).series_for("A · val")
    assert metric_direction("loss") == "min"
    assert series.best("loss")["epoch"] == 4      # loss = 1/epoch, lowest last
    # `optimal_threshold` is logged per epoch but has no direction; calling the
    # largest one "best" would be a fabrication.
    assert metric_direction("optimal_threshold") is None
    assert series.best("optimal_threshold") is None
    assert series.last("optimal_threshold") is not None


def test_plot_can_mark_the_best_epoch(tmp_path):
    dst = make_run(str(tmp_path), "dsA", epochs=5,
                   val=[0.50, 0.70, 0.90, 0.72, 0.60],
                   settings=BASE_SETTINGS)
    comparison = compare_runs([load_run(dst, run_id="A")])
    fig = plot_curves(comparison, "accuracy", mark_best=True)
    try:
        markers = [l for l in fig.axes[0].lines if l.get_marker() == "o"]
        assert markers and list(markers[0].get_xdata()) == [3]
    finally:
        fig.clf()


# ---------------------------------------------------------------------------
# Broken run folders
# ---------------------------------------------------------------------------

def test_a_folder_with_checkpoints_but_no_curves_is_reported_not_dropped(tmp_path):
    root = str(tmp_path)
    good = make_run(root, "dsA", epochs=10, train=[0.5] * 10, val=[0.5] * 10,
                    settings=BASE_SETTINGS)
    broken = os.path.join(root, "dsX", "model", "maxvit_t", "rgb", "epochs_3")
    os.makedirs(broken)
    with open(os.path.join(broken, "maxvit_t_epoch_3_channels_rgb.pth"), "wb") as f:
        f.write(b"not really a checkpoint")

    runs = find_runs(root)
    ids = {r.run_id for r in runs}
    assert len(runs) == 2, ids
    bad = next(r for r in runs if not r.has_curves)
    assert any("no per-epoch curves" in n for n in bad.notes)
    assert any("no train.csv" in n for n in bad.notes)
    assert any("no settings found" in n for n in bad.notes)

    # The good run is still comparable, and the problem travels with the
    # comparison instead of vanishing.
    comparison = compare_runs(runs)
    assert [s.run_id for s in comparison.series].count(
        load_run(good).run_id) == 2
    assert any("no per-epoch curves" in p["note"] for p in comparison.problems)


def test_a_zero_epoch_log_is_reported_and_does_not_break_the_comparison(tmp_path):
    root = str(tmp_path)
    make_run(root, "dsA", epochs=10, train=[0.5] * 10, val=[0.5] * 10,
             settings=BASE_SETTINGS)
    empty = os.path.join(root, "dsZ", "model", "maxvit_t", "rgb", "epochs_4")
    os.makedirs(empty)
    # A header and nothing else: training started, wrote the header, died.
    with open(os.path.join(empty, "train.csv"), "w") as f:
        f.write(",accuracy,loss,epoch\n")

    runs = find_runs(root)
    assert len(runs) == 2
    zero = next(r for r in runs if not r.has_curves)
    assert any("zero-epoch log" in n for n in zero.notes)
    comparison = compare_runs(runs)
    # The healthy run still produced its two series.
    assert len(comparison.series) == 2
    assert any("zero-epoch log" in p["note"] for p in comparison.problems)
    assert "zero-epoch log" in format_comparison(comparison, "accuracy")


def test_a_run_with_no_settings_still_plots(tmp_path):
    dst = make_run(str(tmp_path), "dsA", epochs=6, train=[0.5] * 6,
                   val=[0.6] * 6)
    run = load_run(dst, run_id="A")
    assert run.settings == {}
    assert any("no settings found" in n for n in run.notes)
    assert run.has_curves
    comparison = compare_runs([run])
    assert len(comparison.series) == 2


def test_an_unreadable_curve_file_is_reported_not_raised(tmp_path):
    dst = make_run(str(tmp_path), "dsA", epochs=6, val=[0.5] * 6,
                   settings=BASE_SETTINGS)
    with open(os.path.join(dst, "train.csv"), "w") as f:
        f.write("")          # zero bytes — pandas raises EmptyDataError
    run = load_run(dst, run_id="A")
    assert any("is empty (0 bytes)" in n for n in run.notes)
    assert run.has_curves    # the validation curve survived
    assert compare_runs([run]).labels() == ["A · val"]


def test_load_run_still_raises_for_a_folder_that_is_not_there(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run(os.path.join(str(tmp_path), "nope"))
    with pytest.raises(FileNotFoundError):
        find_runs(os.path.join(str(tmp_path), "nope"))


def test_an_empty_tree_finds_nothing_without_complaining(tmp_path):
    assert find_runs(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# Odds and ends
# ---------------------------------------------------------------------------

def test_available_metrics_puts_the_common_ones_first(two_runs):
    _root, a, b = two_runs
    metrics = available_metrics([a, b])
    assert metrics[:3] == ["accuracy", "loss", "prauc"]
    assert "optimal_threshold" in metrics


def test_summary_line_names_the_run_its_shape_and_key_settings(two_runs):
    _root, a, _b = two_runs
    line = a.summary_line()
    assert "A" in line and "10 epochs" in line and "train+val" in line
    assert "model_type=maxvit_t" in line


def test_reused_run_folder_with_restarted_epochs_is_flagged(tmp_path):
    dst = make_run(str(tmp_path), "dsA", epochs=4, train=[0.5, 0.6, 0.7, 0.8],
                   settings=BASE_SETTINGS)
    # _save_progress appends without a header, so a second run into the same
    # dst leaves epochs 1,2,3,4,1,2 in one file.
    path = os.path.join(dst, "train.csv")
    extra = pd.DataFrame(_rows([0.4, 0.45]))
    with open(path, "a") as f:
        extra.to_csv(f, index=True, header=False)
    run = load_run(dst, run_id="A")
    assert any("not increasing" in n for n in run.notes)
    assert run.has_curves


def test_comparison_of_nothing_is_empty_but_valid():
    comparison = compare_runs([])
    assert isinstance(comparison, Comparison)
    assert comparison.series == []
    assert comparison.metrics == []
    assert comparison.settings_diff["identical"] is False
    assert "not comparable" in format_comparison(comparison)


# ---------------------------------------------------------------------------
# No torch
# ---------------------------------------------------------------------------

def test_importing_train_compare_does_not_pull_in_torch():
    """The dashboard reads CSVs; it must not drag the training stack in.

    Checked in a subprocess because another test in the same session may
    already have imported torch.
    """
    code = textwrap.dedent("""
        import sys
        before = set(sys.modules)
        import spacr.train_compare as tc
        tc.metric_direction('accuracy')
        tc.diff_settings([])
        pulled = sorted(m for m in set(sys.modules) - before
                        if m == 'torch' or m.startswith('torch.'))
        assert not pulled, pulled
        print('clean')
    """)
    env = dict(os.environ)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # PYTHONPATH is replaced rather than prepended to: a developer's coverage
    # shim on it can pre-import torch through sitecustomize, which would make
    # this test pass for the wrong reason. The before/after snapshot above is
    # the second guard against exactly that.
    env["PYTHONPATH"] = repo
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, env=env, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout


# ---------------------------------------------------------------------------
# Settings recovery: every route, including the broken ones
# ---------------------------------------------------------------------------

def test_a_run_journal_folder_is_read_from_settings_json_and_manifest(tmp_path):
    """A run replayed through ``spacr repro`` leaves a journal folder.

    Those carry ``settings.json`` and a ``manifest.json`` with an environment
    snapshot, so the env diff has something real to compare.
    """
    import json
    runs = []
    for name, torch_version, lr in (("run_a", "2.6.0", 0.0001),
                                    ("run_b", "2.7.1", 0.0001)):
        folder = tmp_path / name
        folder.mkdir()
        write_curve(str(folder), "val", [0.5, 0.6, 0.7])
        (folder / "settings.json").write_text(json.dumps(
            {"model_type": "maxvit_t", "learning_rate": lr}))
        (folder / "manifest.json").write_text(json.dumps(
            {"app_key": "classify", "status": "success",
             "env": {"spacr": "1.3.6", "torch": torch_version,
                     "python": "3.10.19"}}))
        runs.append(load_run(str(folder), run_id=name))

    assert runs[0].settings["model_type"] == "maxvit_t"
    assert runs[0].settings_path.endswith("settings.json")
    assert runs[0].manifest["app_key"] == "classify"

    diff = compare_runs(runs).settings_diff
    assert diff["changed"] == []
    assert [e["key"] for e in diff["env_manifest"]] == ["torch"]
    assert diff["env_manifest"][0]["values"] == {"run_a": "2.6.0",
                                                 "run_b": "2.7.1"}
    text = format_comparison(compare_runs(runs), "accuracy")
    assert "Environment drift (1)" in text
    assert "2.6.0" in text and "2.7.1" in text


def test_manifest_env_is_dropped_entirely_when_one_side_has_none(tmp_path):
    # The same rule as run_journal._diff_env: one unreadable manifest must not
    # invent a dozen "changed to None" rows from the other side.
    import json
    a = tmp_path / "a"
    a.mkdir()
    (a / "settings.json").write_text(json.dumps({"k": 1}))
    (a / "manifest.json").write_text(json.dumps({"env": {"torch": "2.6.0"}}))
    b = tmp_path / "b"
    b.mkdir()
    (b / "settings.json").write_text(json.dumps({"k": 1}))
    (b / "manifest.json").write_text("{ not json")
    runs = [load_run(str(a), run_id="A"), load_run(str(b), run_id="B")]
    assert runs[1].manifest == {}
    assert compare_runs(runs).settings_diff["env_manifest"] == []


def test_a_settings_json_that_is_not_an_object_is_reported(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "settings.json").write_text("[1, 2, 3]")
    run = load_run(str(folder), run_id="A")
    assert any("settings.json is a list" in n for n in run.notes)
    assert run.settings == {}


def test_an_unreadable_settings_json_falls_through_with_a_note(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "settings.json").write_text("{ not json at all")
    run = load_run(str(folder), run_id="A")
    assert any("settings.json unreadable (JSONDecodeError)" in n
               for n in run.notes)


def test_a_settings_csv_inside_the_run_folder_is_used(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "settings.csv").write_text("Key,Value\nmodel_type,resnet50\n")
    run = load_run(str(folder), run_id="A")
    assert run.settings == {"model_type": "resnet50"}
    assert run.settings_path.endswith("settings.csv")


def test_an_unreadable_settings_csv_is_reported_not_raised(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "settings.csv").write_bytes(b"Key,Value\nmodel\x00type,x\n")
    run = load_run(str(folder), run_id="A")
    assert any("settings.csv unreadable" in n for n in run.notes)
    assert run.settings == {}


def test_an_ambiguous_settings_folder_names_the_file_it_picked(tmp_path):
    import time
    root = str(tmp_path)
    # A run folder that is not named epochs_<N>, so the exact-name match
    # cannot fire and the picker has to choose.
    src = tmp_path / "ds"
    dst = src / "model_out"
    dst.mkdir(parents=True)
    write_curve(str(dst), "train", [0.5, 0.6])
    settings_dir = src / "settings"
    settings_dir.mkdir()
    for name, lr in (("train_maxvit_t_10", 1e-4), ("train_resnet50_25", 1e-3)):
        pd.DataFrame([("learning_rate", lr)],
                     columns=["Key", "Value"]).to_csv(
                         settings_dir / f"{name}.csv", index=False)
        time.sleep(0.01)
    os.utime(settings_dir / "train_resnet50_25.csv", (2_000_000, 2_000_000))
    os.utime(settings_dir / "train_maxvit_t_10.csv", (1_000_000, 1_000_000))

    run = load_run(str(dst), run_id="A")
    assert run.settings_path.endswith("train_resnet50_25.csv")
    assert any("1 other settings file(s)" in n for n in run.notes)
    # The id falls back to the folder name when the layout is not the one
    # train_test_model builds.
    assert load_run(str(dst)).run_id == "model_out"
    assert root  # the tree really was under tmp_path


def test_an_unreadable_manifest_is_ignored(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "manifest.json").write_text("[]")
    assert load_run(str(folder), run_id="A").manifest == {}


# ---------------------------------------------------------------------------
# Malformed progress CSVs
# ---------------------------------------------------------------------------

def test_a_log_without_an_epoch_column_falls_back_to_row_order(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "train.csv").write_text("accuracy,loss\n0.5,1.0\n0.7,0.5\n")
    run = load_run(str(folder), run_id="A")
    assert any("no 'epoch' column — using row order" in n for n in run.notes)
    series = compare_runs([run]).series_for("A · train")
    assert list(series.epochs) == [1.0, 2.0]


def test_rows_with_an_unreadable_epoch_are_dropped_and_counted(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "train.csv").write_text(
        "epoch,accuracy\n1,0.5\nlater,0.6\n3,0.7\n")
    run = load_run(str(folder), run_id="A")
    assert any("1 row(s) with an unreadable epoch were dropped" in n
               for n in run.notes)
    series = compare_runs([run]).series_for("A · train")
    assert list(series.epochs) == [1.0, 3.0]


def test_a_csv_pandas_cannot_parse_is_reported_not_raised(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    # An unterminated quoted field — what a write interrupted mid-row leaves.
    (folder / "train.csv").write_text('epoch,accuracy\n1,"0.5\n2,0.6\n')
    run = load_run(str(folder), run_id="A")
    assert any("could not be parsed (ParserError" in n for n in run.notes)
    assert not run.has_curves


def test_notes_from_several_folds_are_grouped_by_message(tmp_path):
    dst = make_run(str(tmp_path), "dsCV", model_type="resnet50", epochs=4,
                   folds={1: ([0.5] * 4, None), 2: ([0.6] * 4, None)},
                   settings=BASE_SETTINGS)
    run = load_run(dst, run_id="CV")
    grouped = [n for n in run.notes if "no validation.csv" in n]
    assert grouped == ["no validation.csv (fold_1, fold_2)"]
    # …and the folds still produce their train series.
    assert compare_runs([run]).labels() == ["CV · train · fold 1",
                                            "CV · train · fold 2"]


def test_folds_are_ordered_numerically_not_lexically(tmp_path):
    dst = make_run(str(tmp_path), "dsCV", model_type="resnet50", epochs=3,
                   folds={n: ([0.5] * 3, None) for n in (1, 2, 10)},
                   settings=BASE_SETTINGS)
    run = load_run(dst, run_id="CV")
    labels = compare_runs([run]).labels()
    assert labels == ["CV · train · fold 1", "CV · train · fold 2",
                      "CV · train · fold 10"]


def test_a_metric_that_is_nan_throughout_prints_as_nan(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "train.csv").write_text(
        "epoch,accuracy,prauc\n1,0.5,0.4\n2,0.6,\n")
    run = load_run(str(folder), run_id="A")
    text = format_comparison(compare_runs([run]), "prauc")
    # Epoch 2's prauc is missing, so "last" is epoch 1 — a NaN tail must not
    # be reported as the final value.
    assert "0.4000" in text
    series = compare_runs([run]).series_for("A · train")
    assert series.last("prauc")["epoch"] == 1
    assert np.isnan(series.values("prauc")[1])


# ---------------------------------------------------------------------------
# Scanning odd trees
# ---------------------------------------------------------------------------

def test_hidden_folders_are_not_scanned(tmp_path):
    root = str(tmp_path)
    make_run(root, "dsA", epochs=4, train=[0.5] * 4, settings=BASE_SETTINGS)
    make_run(root, ".trash", epochs=4, train=[0.5] * 4)
    runs = find_runs(root)
    assert len(runs) == 1
    assert ".trash" not in str(runs[0].path)


def test_a_folder_that_cannot_be_read_does_not_abort_the_scan(tmp_path,
                                                              monkeypatch):
    from pathlib import Path as _Path
    root = str(tmp_path)
    make_run(root, "dsA", epochs=4, train=[0.5] * 4, settings=BASE_SETTINGS)
    locked = tmp_path / "locked"
    locked.mkdir()
    real_iterdir = _Path.iterdir

    def blocked_iterdir(self):
        if self == locked:
            raise PermissionError("Permission denied")
        return real_iterdir(self)

    monkeypatch.setattr(_Path, "iterdir", blocked_iterdir)
    runs = find_runs(root)
    assert len(runs) == 1
    # And asked about the locked folder directly, load_run reports rather
    # than raises: no folds readable, no curves, no settings.
    blind = load_run(str(locked), run_id="L")
    assert blind.folds == []
    assert not blind.has_curves


def test_a_folder_whose_mtime_is_unreadable_sorts_last(tmp_path):
    # find_runs sorts newest-first; a folder that vanished between the walk
    # and the sort must not take the whole scan down with it.
    from pathlib import Path
    from spacr.train_compare import _folder_mtime
    assert _folder_mtime(tmp_path) > 0
    assert _folder_mtime(Path(tmp_path) / "gone") == 0.0


def test_a_settings_folder_that_cannot_be_listed_is_skipped(tmp_path,
                                                            monkeypatch):
    from pathlib import Path as _Path
    dst = make_run(str(tmp_path), "dsA", epochs=4, train=[0.5] * 4,
                   settings=BASE_SETTINGS)
    settings_dir = tmp_path / "dsA" / "settings"
    real_glob = _Path.glob

    def blocked_glob(self, pattern, *args, **kwargs):
        if self == settings_dir:
            raise PermissionError("Permission denied")
        return real_glob(self, pattern, *args, **kwargs)

    monkeypatch.setattr(_Path, "glob", blocked_glob)
    run = load_run(dst, run_id="A")
    assert run.settings == {}
    assert any("no settings found" in n for n in run.notes)


# ---------------------------------------------------------------------------
# Rendering details
# ---------------------------------------------------------------------------

def test_three_runs_render_one_column_each(tmp_path):
    root = str(tmp_path)
    runs = []
    for i, lr in enumerate((1e-4, 1e-3, 1e-2), start=1):
        dst = make_run(root, f"ds{i}", epochs=4, train=[0.5] * 4,
                       settings={**BASE_SETTINGS, "learning_rate": lr})
        runs.append(load_run(dst, run_id=f"R{i}"))
    text = format_comparison(compare_runs(runs), "accuracy")
    assert "setting" in text and "R1" in text and "R2" in text and "R3" in text
    assert "0.0001" in text and "0.001" in text and "0.01" in text


def test_plot_can_be_restricted_to_some_series(two_runs):
    _root, a, b = two_runs
    comparison = compare_runs([a, b])
    fig = plot_curves(comparison, "accuracy", labels=["A · val", "B · val"])
    try:
        assert sorted(fig.spacr_series_by_label) == ["A · val", "B · val"]
        assert len(fig.axes[0].lines) == 2
        # One split only now, so the mixed-splits warning must not appear.
        assert "mixes train" not in _axes_notes(fig.axes[0])
    finally:
        fig.clf()


def test_epoch_ranges_are_exposed_per_series(two_runs):
    _root, a, b = two_runs
    ranges = compare_runs([a, b]).epoch_ranges()
    assert ranges["A · val"] == (1, 10)
    assert ranges["B · val"] == (1, 25)


def test_a_single_epoch_run_reports_one_epoch_not_a_range(tmp_path):
    dst = make_run(str(tmp_path), "dsA", epochs=1, train=[0.5],
                   settings=BASE_SETTINGS)
    comparison = compare_runs([load_run(dst, run_id="A")])
    assert comparison.series[0].epoch_range() == (1, 1)
    assert not comparison.lengths_differ()
    text = format_comparison(comparison, "accuracy")
    assert "different epoch counts" not in text


# ---------------------------------------------------------------------------
# Accessors on the edges
# ---------------------------------------------------------------------------

def test_a_run_with_no_curves_reports_zero_epochs(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    run = load_run(str(folder), run_id="A")
    assert run.n_epochs == 0
    assert run.metrics() == []
    assert run.final_metrics == {}
    assert "no curves" in run.summary_line()
    assert run.is_cv is False


def test_series_accessors_are_safe_for_absent_metrics(two_runs):
    from spacr.train_compare import Series
    _root, a, _b = two_runs
    comparison = compare_runs([a])
    series = comparison.series_for("A · val")
    assert list(series.values("nope")) == []
    assert series.has("nope") is False
    assert series.best("nope") is None
    assert series.last("nope") is None
    # sd/support are meaningful only for a fold mean.
    assert series.sd("accuracy") is None
    assert series.support() is None
    assert series.support_drops_at() is None
    assert comparison.series_for("no such label") is None

    # A frame whose epochs are all unusable has no range to report.
    empty = Series(run_id="A", split="val", fold="", kind="single",
                   label="A · val", frame=pd.DataFrame({"epoch": [np.nan]}))
    assert empty.epoch_range() == (0, 0)


def test_a_fold_mean_whose_folds_all_finish_says_nothing_about_support(tmp_path):
    dst = make_run(str(tmp_path), "dsCV", model_type="resnet50", epochs=4,
                   folds={1: ([0.5] * 4, [0.5] * 4), 2: ([0.7] * 4, [0.7] * 4)},
                   settings=BASE_SETTINGS)
    comparison = compare_runs([load_run(dst, run_id="CV")], folds="both")
    mean = comparison.series_for("CV · val · mean of 2 folds ±sd")
    assert mean.support() == (2, 2)
    assert mean.support_drops_at() is None
    text = format_comparison(comparison, "accuracy")
    assert "folds reach that far" not in text
    assert "average across folds, not a single run" in text


def test_scan_depth_can_be_capped(tmp_path):
    root = str(tmp_path)
    make_run(root, "dsA", epochs=4, train=[0.5] * 4, settings=BASE_SETTINGS)
    # The run sits five levels down (ds/model/model_type/channels/epochs_N).
    assert find_runs(root, max_depth=2) == []
    assert len(find_runs(root, max_depth=5)) == 1


def test_pointing_straight_at_a_fold_loads_that_fold(tmp_path):
    dst = make_run(str(tmp_path), "dsCV", model_type="resnet50", epochs=4,
                   folds={1: ([0.5] * 4, [0.5] * 4), 2: ([0.7] * 4, [0.7] * 4)},
                   settings=BASE_SETTINGS)
    # Scanning the run folder finds one run holding both folds …
    assert len(find_runs(dst)) == 1
    # … but a caller who names a fold folder gets that fold, not nothing.
    one = find_runs(os.path.join(dst, "fold_2"))
    assert len(one) == 1
    assert one[0].run_id == "fold_2"
    assert one[0].folds == []


def test_a_settings_folder_with_no_csv_is_skipped(tmp_path):
    src = tmp_path / "ds"
    dst = src / "model" / "maxvit_t" / "rgb" / "epochs_4"
    dst.mkdir(parents=True)
    write_curve(str(dst), "train", [0.5] * 4)
    (src / "settings").mkdir()
    (src / "settings" / "notes.txt").write_text("not a settings file")
    run = load_run(str(dst), run_id="A")
    assert run.settings == {}
    assert any("no settings found" in n for n in run.notes)


def test_an_unreadable_settings_csv_in_the_settings_folder_is_reported(tmp_path):
    src = tmp_path / "ds"
    dst = src / "model" / "maxvit_t" / "rgb" / "epochs_4"
    dst.mkdir(parents=True)
    write_curve(str(dst), "train", [0.5] * 4)
    settings_dir = src / "settings"
    settings_dir.mkdir()
    (settings_dir / "train_test_maxvit_t_4.csv").write_bytes(
        b"Key,Value\nmodel\x00type,x\n")
    run = load_run(str(dst), run_id="A")
    assert any("train_test_maxvit_t_4.csv unreadable" in n for n in run.notes)
    assert any("no settings found" in n for n in run.notes)
    assert run.settings == {}
