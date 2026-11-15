"""Training logs a broken run leaves behind, read by ``spacr.train_compare.load_run``.

A header-only log gets the zero-epoch note (never the row-order one), a metric
only one split logged is reported for that split only, and fold blocks whose
epochs cannot be read give no fold mean instead of a crash -- each beside the
healthy twin that shows what presence looks like.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from spacr.train_compare import TrainingRun, compare_runs, load_run  # noqa: E402


def _write_csv(folder, name, frame):
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / name, index=True)


def _curve(epochs, accuracy, **extra):
    data = {"epoch": list(epochs), "accuracy": list(accuracy)}
    data.update({k: list(v) for k, v in extra.items()})
    return pd.DataFrame(data)


def test_a_degenerate_training_log_loads_with_honest_notes_and_metrics(
        subtests, tmp_path):
    """Zero-row logs, one-sided metrics and unreadable epochs load exactly."""
    with subtests.test(msg="header-only train log beside an epoch-less val log"):
        dst = tmp_path / "zero" / "epochs_10"
        dst.mkdir(parents=True)
        (dst / "train.csv").write_text("accuracy,loss\n", encoding="utf-8")
        (dst / "validation.csv").write_text(
            "accuracy,loss\n0.51,0.90\n0.58,0.71\n", encoding="utf-8")
        run = load_run(dst, run_id="R")
        notes = "\n".join(run.notes)
        assert "train.csv has a header but no epoch rows (zero-epoch log)" in notes
        assert "train.csv has no 'epoch' column" not in notes
        assert "validation.csv has no 'epoch' column — using row order" in notes
        assert sorted(set(run.curves["split"])) == ["val"]
        val = run.curves[run.curves["split"] == "val"]
        assert list(val["epoch"]) == [1, 2]
        assert list(val["accuracy"]) == [0.51, 0.58]

    with subtests.test(msg="a metric only the train split logged"):
        dst = tmp_path / "onesided" / "epochs_3"
        _write_csv(dst, "train.csv",
                   _curve([1, 2, 3], [0.50, 0.62, 0.71], prauc=[0.4, 0.5, 0.6]))
        _write_csv(dst, "validation.csv", _curve([1, 2, 3], [0.48, 0.57, 0.66]))
        run = load_run(dst, run_id="R")
        assert "prauc" in run.curves.columns
        assert bool(run.curves.loc[run.curves["split"] == "val",
                                   "prauc"].isna().all())
        train = run.final_metrics["R · train"]
        val = run.final_metrics["R · val"]
        assert sorted(train["last"]) == ["accuracy", "prauc"]
        assert train["last"]["prauc"] == {"epoch": 3, "value": pytest.approx(0.6)}
        assert train["best"]["prauc"] == {"epoch": 3, "value": pytest.approx(0.6),
                                          "direction": "max"}
        assert sorted(val["last"]) == ["accuracy"]
        assert sorted(val["best"]) == ["accuracy"]
        assert val["last"]["accuracy"]["value"] == pytest.approx(0.66)

    with subtests.test(msg="fold blocks with unreadable epochs give no mean"):
        broken = TrainingRun(
            run_id="BROKEN", path=tmp_path / "broken",
            curves=pd.DataFrame({
                "run_id": ["BROKEN"] * 4, "split": ["train"] * 4,
                "fold": ["fold_1", "fold_1", "fold_2", "fold_2"],
                "epoch": [np.nan] * 4,
                "accuracy": [0.50, 0.61, 0.55, 0.66]}),
            folds=["fold_1", "fold_2"])
        good_dst = tmp_path / "good" / "epochs_4"
        _write_csv(good_dst / "fold_1", "train.csv", _curve([1, 2], [0.5, 0.6]))
        _write_csv(good_dst / "fold_2", "train.csv", _curve([1, 2], [0.4, 0.7]))
        comparison = compare_runs([broken, load_run(good_dst, run_id="GOOD")],
                                  folds="mean")
        assert comparison.labels() == ["GOOD · train · mean of 2 folds ±sd"]
        (mean,) = comparison.series
        assert list(mean.frame["epoch"]) == [1, 2]
        assert list(mean.frame["accuracy"]) == [pytest.approx(0.45),
                                                pytest.approx(0.65)]
        assert [r.run_id for r in comparison.runs] == ["BROKEN", "GOOD"]
        assert len(broken.curves) == 4
