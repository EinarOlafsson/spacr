"""Anomaly scoring against the negative control, on a screen with known hits.

Four wells per plate carry a phenotype in half their cells that no single
feature names; every detector must rank them above the negative-control
wells (well AUROC), score controls without having seen them, and hand the
top outliers over for review.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from spacr.sp_stats import (HitScoringError, _auroc, _draw_anomaly_review,
                            _score_anomalies, _write_anomaly_report)

_HITS = ("B03", "C05", "D07", "E02")


def _screen(seed: int = 1, cells: int = 30) -> pd.DataFrame:
    """Two 96-well plates, columns 1 and 12 negative control, four hits."""
    rng = np.random.default_rng(seed)
    frames = []
    for plate in ("P1", "P2"):
        offset = rng.normal(0.0, 0.5, 8)
        for row in "ABCDEFGH":
            for col in range(1, 13):
                well = f"{row}{col:02d}"
                values = rng.normal(0.0, 1.0, (cells, 8)) + offset
                if well in _HITS:
                    half = cells // 2
                    values[:half, 4] += 4.0
                    values[:half, 6] -= 3.0
                frame = pd.DataFrame(values,
                                     columns=[f"feature_{i}" for i in range(8)])
                frame["plateID"] = plate
                frame["rowID"] = row
                frame["columnID"] = col
                frame["condition"] = "neg" if col in (1, 12) else "compound"
                frame["png_path"] = ""
                frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize("method", ["mahalanobis", "knn", "iforest", "gmm"])
def test_known_hits_rank_above_controls(method):
    result = _score_anomalies(_screen(), control_column="condition",
                              negative_levels="neg", known_hits=_HITS,
                              method=method)
    assert result.auroc is not None and result.auroc >= 0.95
    assert result.cell_auroc > 0.65
    top = set(result.ranked_wells()["well"].head(8))
    assert top == set(_HITS)
    assert result.wells["known_hit"].sum() == 8
    assert "AUROC" in result.report()


def test_controls_are_scored_by_a_detector_that_never_saw_them():
    result = _score_anomalies(_screen(seed=3), control_column="condition",
                              negative_levels="neg", quantile=0.95)
    controls = result.cells[result.cells["role"] == "negative"]
    rate = float(controls["outlier"].mean())
    assert 0.03 <= rate <= 0.07
    negative_wells = result.wells[result.wells["role"] == "negative"]
    assert negative_wells["rank"].isna().all()
    assert result.auroc is None


def test_plate_offsets_are_not_phenotypes():
    frame = _screen(seed=5)
    shifted = frame["plateID"] == "P2"
    features = [c for c in frame.columns if c.startswith("feature_")]
    frame.loc[shifted, features] += 6.0
    result = _score_anomalies(frame, control_column="condition",
                              negative_levels="neg", known_hits=_HITS)
    assert result.auroc >= 0.95


def test_top_outliers_are_handed_over_for_review(tmp_path):
    from matplotlib.figure import Figure

    result = _score_anomalies(_screen(), control_column="condition",
                              negative_levels="neg", known_hits=_HITS)
    top = result.top_outliers(10)
    assert len(top) == 10
    assert (top["role"] != "negative").all()
    assert top["well"].isin(_HITS).mean() >= 0.8
    assert top["score"].is_monotonic_decreasing
    figure = Figure()
    assert _draw_anomaly_review(figure, result, limit=10) == 10
    written = _write_anomaly_report(result, tmp_path)
    for name in ("anomaly_wells", "anomaly_cells", "anomaly_top_outliers",
                 "anomaly_review"):
        assert os.path.exists(written[name]), name


def test_crops_are_drawn_when_the_table_points_at_them(tmp_path):
    from matplotlib.figure import Figure
    from PIL import Image

    frame = _screen()
    crop = tmp_path / "crop.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(crop)
    frame["png_path"] = str(crop)
    result = _score_anomalies(frame, control_column="condition",
                              negative_levels="neg")
    figure = Figure()
    assert _draw_anomaly_review(figure, result, limit=6) == 6
    assert len(figure.axes) == 6


def test_refusals_are_plain():
    frame = _screen()
    with pytest.raises(HitScoringError, match="negative control"):
        _score_anomalies(frame, control_column="condition")
    with pytest.raises(HitScoringError, match="method"):
        _score_anomalies(frame, control_column="condition",
                         negative_levels="neg", method="flow")
    with pytest.raises(HitScoringError, match="reference"):
        _score_anomalies(frame.head(15), control_column="condition",
                         negative_levels="neg")


def test_auroc_matches_its_definition():
    assert _auroc([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert _auroc([1.0], [1.0]) == 0.5
    assert _auroc([], [1.0]) is None
