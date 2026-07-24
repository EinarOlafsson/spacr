"""Coverage-fill batch 2 for spacr.toxo: generate_score_heatmap."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spacr import toxo as T


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


ROWS = [f"r{i}" for i in range(1, 9)]
COL = "c3"
CTRL = ["sgA", "sgB"]


def _scores_csv(path, seed):
    rng = np.random.default_rng(seed)
    pd.DataFrame({
        "column_name": [COL] * len(ROWS),
        "rowID": ROWS,
        "pred": rng.uniform(0, 1, len(ROWS)),
    }).to_csv(path, index=False)


def _mixed_csv(path):
    rng = np.random.default_rng(5)
    rows = []
    for r in ROWS:
        for g in CTRL:
            rows.append({"column_name": COL, "rowID": r, "grna_name": g,
                         "count": int(rng.integers(10, 500))})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_generate_score_heatmap(tmp_path):
    # folders/<sub>/scores.csv  (two model folders)
    folder = tmp_path / "models"; folder.mkdir()
    for m in ("modelA", "modelB"):
        d = folder / m; d.mkdir()
        _scores_csv(str(d / "scores.csv"), seed=hash(m) % 100)

    mixed = tmp_path / "mixed.csv"; _mixed_csv(str(mixed))
    cv = tmp_path / "cv.csv"; _scores_csv(str(cv), seed=9)
    dst = tmp_path / "out"; dst.mkdir()

    settings = {
        "folders": [str(folder)], "csv_name": "scores.csv",
        "data_column": "pred", "csv": str(mixed), "cv_csv": str(cv),
        "data_column_cv": "pred", "plateID": 1, "columnID": COL,
        "control_sgrnas": CTRL, "fraction_grna": "sgA", "dst": str(dst),
    }
    out = T.generate_score_heatmap(settings)
    assert isinstance(out, pd.DataFrame)
    # writes the comparison artifacts
    assert list(dst.glob("*.pdf")) and list(dst.glob("*.csv"))
