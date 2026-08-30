"""The edges of the training-run comparison that a broken log actually reaches.

:mod:`spacr.train_compare` reads whatever ``spacr.io._save_progress`` left on
disk, and what it left is not always a well-formed curve: a log killed before
its first epoch has a header and no rows, a metric can be present on the train
side and missing on the validation side, a k-fold run can have exactly one
fold folder written, and two of the columns training writes
(``optimal_threshold``, ``train_time``) have no "best" at all.

Each test here drives one of those degenerate inputs together with the healthy
input it is the degenerate twin of, so the assertion that something is absent
is made in the same test that shows what its presence looks like. The
behaviours pinned:

* a zero-row log is not annotated as "using row order" — there are no rows to
  order, and the zero-epoch note is the true one;
* a metric only one split logged is reported for that split and simply not
  reported for the other, rather than as a fabricated NaN "final value";
* a mean over a single fold draws no ±sd band, because one fold has no spread;
* a directionless metric gets no best-epoch marker and no best-epoch caveat,
  because calling the largest ``optimal_threshold`` "best" would be a
  fabrication;
* curves whose epochs cannot be read produce no fold-mean series instead of a
  crash inside the groupby.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from spacr.train_compare import (
    TrainingRun,
    compare_runs,
    format_comparison,
    load_run,
    plot_curves,
)


# ---------------------------------------------------------------------------
# Run folders, written the way training writes them
# ---------------------------------------------------------------------------

def _write_csv(folder, name, frame):
    """Write one progress CSV with the index column a real train.csv carries."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    frame.to_csv(path, index=True)
    return path


def _curve(epochs, accuracy, **extra):
    data = {"epoch": list(epochs), "accuracy": list(accuracy)}
    data.update({k: list(v) for k, v in extra.items()})
    return pd.DataFrame(data)


@pytest.fixture(autouse=True)
def _close_figures():
    """No figure from one test may be counted by the next."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Reading a log that stopped before it logged anything
# ---------------------------------------------------------------------------

def test_a_zero_row_log_is_not_described_as_ordered_by_row(tmp_path):
    """A header-only log gets the zero-epoch note, never the row-order one.

    ``_save_progress`` writes the header on the first epoch, so a run killed
    during epoch 1 leaves a file with column names and no rows. If that file
    also has no ``epoch`` column, the reader still synthesises one from the row
    index — and the note that goes with that ("using row order") would tell the
    user the x axis of a curve that does not exist. What they need to be told
    is that the run logged nothing. The same folder's validation.csv *does*
    have rows and no epoch column, which is the case the row-order note is for,
    so both notes are decided here against the same run.
    """
    dst = tmp_path / "epochs_10"
    dst.mkdir()
    # Header only: killed before epoch 1 was appended.
    (dst / "train.csv").write_text("accuracy,loss\n")
    # Rows, but an older writer that never wrote an epoch column.
    (dst / "validation.csv").write_text("accuracy,loss\n0.51,0.90\n0.58,0.71\n")

    run = load_run(dst, run_id="R")
    notes = "\n".join(run.notes)

    assert "train.csv has a header but no epoch rows (zero-epoch log)" in notes
    assert "train.csv has no 'epoch' column" not in notes
    # The twin: rows with no epoch column DO earn the row-order note, and are
    # numbered 1..n from the row index.
    assert "validation.csv has no 'epoch' column — using row order" in notes
    val = run.curves[run.curves["split"] == "val"]
    assert list(val["epoch"]) == [1, 2]
    assert list(val["accuracy"]) == [0.51, 0.58]
    # Only the validation rows survived; the empty train log contributed none.
    assert sorted(set(run.curves["split"])) == ["val"]


# ---------------------------------------------------------------------------
# A metric that only one split logged
# ---------------------------------------------------------------------------

def test_a_metric_missing_from_one_split_is_omitted_not_reported_as_nan(tmp_path):
    """``final_metrics`` reports a metric only for the splits that logged it.

    The curves of a run are concatenated into one long frame, so a column that
    exists only on the train side appears on the validation rows as NaN. If the
    summary took those NaNs at face value it would print a final validation
    ``prauc`` for a run that never computed one — a made-up held-out number,
    which is the worst kind of wrong in this dashboard. Both sides are asserted
    from one run: train keeps its prauc, val simply has no prauc entry.
    """
    dst = tmp_path / "epochs_3"
    _write_csv(dst, "train.csv",
               _curve([1, 2, 3], [0.50, 0.62, 0.71], prauc=[0.4, 0.5, 0.6]))
    _write_csv(dst, "validation.csv", _curve([1, 2, 3], [0.48, 0.57, 0.66]))

    run = load_run(dst, run_id="R")
    # The union column exists on the frame; it is empty on the val rows.
    assert "prauc" in run.curves.columns
    val_rows = run.curves[run.curves["split"] == "val"]
    assert bool(val_rows["prauc"].isna().all())

    train = run.final_metrics["R · train"]
    val = run.final_metrics["R · val"]
    assert sorted(train["last"]) == ["accuracy", "prauc"]
    assert train["last"]["prauc"] == {"epoch": 3, "value": pytest.approx(0.6)}
    assert train["best"]["prauc"] == {"epoch": 3, "value": pytest.approx(0.6),
                                      "direction": "max"}
    assert sorted(val["last"]) == ["accuracy"]
    assert sorted(val["best"]) == ["accuracy"]
    assert val["last"]["accuracy"]["value"] == pytest.approx(0.66)


# ---------------------------------------------------------------------------
# A fold mean with nothing to average
# ---------------------------------------------------------------------------

def test_a_mean_over_one_fold_draws_no_spread_band(tmp_path):
    """One fold has no fold-to-fold spread, so no ±sd band is drawn.

    A k-fold run whose other folds crashed (or has not finished) leaves a
    single ``fold_1/``. ``folds='mean'`` still produces a mean series for it —
    that is the mode the user asked for — but the sd across one fold is
    undefined, and a band drawn at ±nan would either vanish silently or paint a
    zero-width ribbon that reads as "no variance across folds", which is a
    claim nobody measured. The two-fold run in the same test shows the band
    that a real mean does get.
    """
    one = tmp_path / "one" / "epochs_4"
    _write_csv(one / "fold_1", "train.csv", _curve([1, 2, 3], [0.5, 0.6, 0.7]))
    two = tmp_path / "two" / "epochs_4"
    _write_csv(two / "fold_1", "train.csv", _curve([1, 2, 3], [0.5, 0.6, 0.7]))
    _write_csv(two / "fold_2", "train.csv", _curve([1, 2, 3], [0.4, 0.5, 0.8]))

    lonely = compare_runs([load_run(one, run_id="ONE")], folds="mean")
    paired = compare_runs([load_run(two, run_id="TWO")], folds="mean")

    (solo,) = lonely.series
    assert solo.label == "ONE · train · mean of 1 folds ±sd"
    assert bool(np.isnan(solo.sd("accuracy")).all())
    (both,) = paired.series
    assert both.sd("accuracy")[0] == pytest.approx(
        float(np.std([0.5, 0.4], ddof=1)))

    solo_fig = plot_curves(lonely, "accuracy", band=True)
    paired_fig = plot_curves(paired, "accuracy", band=True)
    # fill_between is the only collection either axes can acquire.
    assert len(solo_fig.axes[0].collections) == 0
    assert len(paired_fig.axes[0].collections) == 1
    # The line itself is still drawn for the single fold — only the band is
    # withheld — and it carries the mean of the one fold.
    (line,) = solo_fig.axes[0].lines
    assert list(line.get_ydata()) == [0.5, 0.6, 0.7]


# ---------------------------------------------------------------------------
# A metric with no meaningful best
# ---------------------------------------------------------------------------

def test_a_directionless_metric_gets_no_best_epoch_marker(tmp_path):
    """``mark_best`` marks nothing on ``optimal_threshold``.

    ``evaluate_model_performance`` logs ``optimal_threshold`` per epoch, but
    neither the largest nor the smallest threshold is "best" — the value is a
    property of the epoch's decision boundary, not a score. Marking one anyway
    would invite the user to pick the epoch under the dot. The same axes call
    on ``accuracy``, which does have a direction, shows the marker that a real
    metric earns.
    """
    dst = tmp_path / "epochs_3"
    _write_csv(dst, "train.csv",
               _curve([1, 2, 3], [0.50, 0.62, 0.71],
                      optimal_threshold=[0.50, 0.55, 0.61]))
    comparison = compare_runs([load_run(dst, run_id="T")])
    (series,) = comparison.series
    assert series.best("optimal_threshold") is None
    assert series.best("accuracy") == {"epoch": 3, "value": pytest.approx(0.71),
                                       "direction": "max"}

    plain = plot_curves(comparison, "optimal_threshold", mark_best=True)
    marked = plot_curves(comparison, "accuracy", mark_best=True)
    # One line is the curve; the second, on the accuracy axes, is the marker.
    assert len(plain.axes[0].lines) == 1
    assert list(plain.axes[0].lines[0].get_ydata()) == [0.50, 0.55, 0.61]
    assert len(marked.axes[0].lines) == 2
    assert list(marked.axes[0].lines[1].get_xydata()[0]) == [3, 0.71]


def test_a_directionless_metric_is_reported_without_the_best_epoch_caveat(tmp_path):
    """The optimistic-bias caveat is printed only where a "best" was printed.

    The caveat explains that a best epoch chosen from the very curve it scores
    is an optimistically biased estimate. On a metric with no direction the
    table's ``best`` column is a dash, so printing the caveat would explain a
    number that is not on the page and imply the dash means something worse
    than "not applicable". The accuracy report from the same run carries it.
    """
    dst = tmp_path / "epochs_3"
    _write_csv(dst, "train.csv",
               _curve([1, 2, 3], [0.50, 0.62, 0.71],
                      optimal_threshold=[0.50, 0.55, 0.61]))
    comparison = compare_runs([load_run(dst, run_id="T")])

    threshold_report = format_comparison(comparison, metric="optimal_threshold")
    accuracy_report = format_comparison(comparison, metric="accuracy")

    assert "Curves — optimal_threshold (1 series" in threshold_report
    row = [ln for ln in threshold_report.splitlines()
           if ln.strip().startswith("T · train")]
    assert len(row) == 1
    # last value present, best column dashed.
    assert "0.6100" in row[0]
    assert row[0].split()[-2:] == ["—", "—"]
    assert "optimistically biased" not in threshold_report
    assert "optimistically biased" in accuracy_report
    assert "0.7100" in accuracy_report


# ---------------------------------------------------------------------------
# Curves whose epochs cannot be read
# ---------------------------------------------------------------------------

def test_folds_with_unreadable_epochs_yield_no_mean_instead_of_crashing(tmp_path):
    """A fold block with no usable epoch produces no mean series.

    The fold mean is grouped by epoch, so a caller that assembles
    :class:`TrainingRun.curves` itself — the GUI hands one in, and the field is
    part of the public dataclass — can present a block whose epoch column is
    entirely unparseable. Grouping it yields no groups at all; the mean has to
    come back as "nothing to average" rather than as an empty frame that the
    plotter would then try to draw. The healthy run in the same comparison
    still produces its mean, so a single unreadable run cannot silence the
    others.
    """
    broken = TrainingRun(
        run_id="BROKEN",
        path=tmp_path / "broken",
        curves=pd.DataFrame({
            "run_id": ["BROKEN"] * 4,
            "split": ["train"] * 4,
            "fold": ["fold_1", "fold_1", "fold_2", "fold_2"],
            "epoch": [np.nan] * 4,
            "accuracy": [0.50, 0.61, 0.55, 0.66],
        }),
        folds=["fold_1", "fold_2"],
    )
    good_dst = tmp_path / "good" / "epochs_4"
    _write_csv(good_dst / "fold_1", "train.csv", _curve([1, 2], [0.5, 0.6]))
    _write_csv(good_dst / "fold_2", "train.csv", _curve([1, 2], [0.4, 0.7]))
    good = load_run(good_dst, run_id="GOOD")

    comparison = compare_runs([broken, good], folds="mean")

    labels = comparison.labels()
    assert labels == ["GOOD · train · mean of 2 folds ±sd"]
    assert not [s for s in comparison.series if s.run_id == "BROKEN"]
    (mean,) = comparison.series
    assert list(mean.frame["epoch"]) == [1, 2]
    assert list(mean.frame["accuracy"]) == [pytest.approx(0.45),
                                            pytest.approx(0.65)]
    # And the broken run is still in the comparison, with its curves, so the
    # report can name it rather than dropping it silently.
    assert [r.run_id for r in comparison.runs] == ["BROKEN", "GOOD"]
    assert len(broken.curves) == 4
