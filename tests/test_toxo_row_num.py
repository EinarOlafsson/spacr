"""Regression pin: ``spacr.toxo.generate_score_heatmap`` must not leak ``row_num``.

``plot_multi_channel_heatmap`` adds an integer ``row_num`` column purely to sort
plate rows numerically. It used to add it to the CALLER's frame — only the
subsequent ``drop`` was applied to a local slice — and the guard that was meant
to clean it up tested for ``'row_number'``, a name nothing ever creates. So the
sort helper survived into the returned frame, into ``*_data.csv`` and, because
``calculate_mae`` treats every numeric column as a channel, into the MAE table
as a bogus ``row_num`` channel whose "error" is the row index.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


ROWS = [f"r{i}" for i in range(1, 9)]
COL = "c3"
CTRL = ["sgA", "sgB"]


def _write_scores(path, seed):
    rng = np.random.default_rng(seed)
    pd.DataFrame({
        "columnID": [COL] * len(ROWS),
        "rowID": ROWS,
        "pred": rng.uniform(0, 1, len(ROWS)),
    }).to_csv(path, index=False)


def _write_counts(path):
    rng = np.random.default_rng(5)
    rows = [{"columnID": COL, "rowID": r, "grna_name": g,
             "count": int(rng.integers(10, 500))}
            for r in ROWS for g in CTRL]
    pd.DataFrame(rows).to_csv(path, index=False)


def _settings(tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    for i, model in enumerate(("modelA", "modelB")):
        sub = folder / model
        sub.mkdir()
        _write_scores(str(sub / "scores.csv"), seed=11 + i)

    counts = tmp_path / "counts.csv"
    _write_counts(str(counts))
    cv = tmp_path / "cv.csv"
    _write_scores(str(cv), seed=9)
    dst = tmp_path / "out"
    dst.mkdir()

    return {
        "folders": [str(folder)], "csv_name": "scores.csv",
        "data_column": "pred", "csv": str(counts), "cv_csv": str(cv),
        "data_column_cv": "pred", "plateID": 1, "columnID": COL,
        "control_sgrnas": CTRL, "fraction_grna": "sgA", "dst": str(dst),
    }


def test_generate_score_heatmap_does_not_leak_row_num(tmp_path):
    """The sort helper must not survive in the returned frame, the saved CSV
    or the MAE channel list."""
    from spacr.toxo import generate_score_heatmap

    settings = _settings(tmp_path)
    out = generate_score_heatmap(settings)

    assert isinstance(out, pd.DataFrame)
    assert "row_num" not in out.columns

    saved = pd.read_csv(os.path.join(settings["dst"],
                                     "scores_comparison_plate_1_data.csv"))
    assert "row_num" not in saved.columns

    mae = pd.read_csv(os.path.join(settings["dst"],
                                   "mae_scores_comparison_plate_1.csv"))
    channels = set(mae["Channel"])
    assert "row_num" not in channels
    # only the real score channels are compared against 'fraction'
    assert channels == {"modelA_pred", "modelB_pred", "pred"}


def test_plot_multi_channel_heatmap_does_not_mutate_its_argument(tmp_path):
    """The heatmap helper takes a copy, so its caller's frame is untouched.

    Driven through the public entry point, which is the only way the nested
    helper is reachable: a spy on the frame identity would not survive the
    merge chain, so this asserts the observable consequence instead — every
    column of the returned frame was present before the plot ran.
    """
    from spacr.toxo import generate_score_heatmap

    settings = _settings(tmp_path)
    out = generate_score_heatmap(settings)

    # 'fraction' + 'prc' + the merge keys + one column per model + the cv score.
    assert set(out.columns) == {
        "fraction", "prc", "plateID", "rowID", "columnID",
        "modelA_pred", "modelB_pred", "pred",
    }
    # every remaining numeric column is a score in [0, 1], never a row index
    numeric = out.select_dtypes(include=[float, int])
    assert set(numeric.columns) == {"fraction", "modelA_pred", "modelB_pred", "pred"}
    assert ((numeric >= 0) & (numeric <= 1)).all().all()
