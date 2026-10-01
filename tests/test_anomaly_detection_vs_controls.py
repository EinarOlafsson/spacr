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

from spacr.sp_stats import (HitScoringError, _anomaly_halves, _auroc,
                            _draw_anomaly_review, _score_anomalies,
                            _write_anomaly_report)

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


def test_a_phenotype_off_the_controls_subspace_is_caught():
    """Controls vary along two directions; the hits move along a third."""
    rng = np.random.default_rng(7)
    loadings = rng.normal(0.0, 1.0, (2, 12))
    basis, _ = np.linalg.qr(np.vstack([loadings, rng.normal(0.0, 1.0,
                                                              (1, 12))]).T)
    away = basis[:, 2]
    frames = []
    for row in "ABCDEFGH":
        for col in range(1, 13):
            well = f"{row}{col:02d}"
            values = (rng.normal(0.0, 1.0, (40, 2)) @ loadings
                      + rng.normal(0.0, 0.02, (40, 12)))
            if well in _HITS:
                values += 0.3 * away
            frame = pd.DataFrame(values,
                                 columns=[f"feature_{i}" for i in range(12)])
            frame["plateID"] = "P1"
            frame["rowID"] = row
            frame["columnID"] = col
            frame["condition"] = "neg" if col in (1, 12) else "compound"
            frames.append(frame)
    result = _score_anomalies(pd.concat(frames, ignore_index=True),
                              control_column="condition",
                              negative_levels="neg", known_hits=_HITS,
                              components=2)
    assert result.auroc == pytest.approx(1.0)
    assert set(result.ranked_wells()["well"].head(4)) == set(_HITS)
    hits = result.wells[result.wells["known_hit"]]
    assert (hits["mean_percentile"] > 0.9).all()


def test_cross_fitting_keeps_each_control_well_in_one_half():
    frame = _screen(seed=2)
    result = _score_anomalies(frame, control_column="condition",
                              negative_levels="neg")
    controls = result.wells[result.wells["role"] == "negative"]
    assert controls["mean_percentile"].between(0.2, 0.8).all()
    index = np.arange(10)
    plates = np.array(["P1"] * 10)
    rows = np.array(["A"] * 6 + ["B"] * 4)
    cols = np.ones(10, dtype=int)
    first, second = _anomaly_halves(index, plates, rows, cols,
                                    np.random.default_rng(0))
    assert {tuple(first), tuple(second)} == {tuple(range(6)),
                                             tuple(range(6, 10))}
    alone = _anomaly_halves(index, plates, np.array(["A"] * 10), cols,
                            np.random.default_rng(0))
    assert sorted(np.concatenate(alone)) == list(range(10))
    assert len(alone[0]) == 5


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


# ---------------------------------------------------------------------------
# Edges the coverage ratchet found untested (dispatch 36794763761)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,match", [
    (dict(quantile=0.2), "quantile must be at least 0.5"),
    (dict(treatment_column="drug"), "treatment column 'drug'"),
    (dict(negative_levels="neg", control_column=None), "no control column"),
    (dict(features=["feature_0", "nope"]), "no feature column 'nope'"),
    (dict(features=[]), "no numeric feature"),
    (dict(negative_wells="A99"), None),
])
def test_scoring_refuses_what_it_cannot_score(kwargs, match):
    frame = _screen(cells=4)
    arguments = dict(control_column="condition", negative_levels="neg")
    arguments.update(kwargs)
    with pytest.raises(HitScoringError, match=match):
        _score_anomalies(frame, **arguments)


def test_an_empty_table_or_one_with_no_readable_well_is_refused():
    with pytest.raises(HitScoringError, match="the table is empty"):
        _score_anomalies(_screen().head(0), control_column="condition",
                         negative_levels="neg")
    frame = _screen(cells=2)
    frame["rowID"] = "?"
    with pytest.raises(HitScoringError, match="readable well"):
        _score_anomalies(frame, control_column="condition",
                         negative_levels="neg")


def test_a_table_the_well_reader_rejects_is_refused_plainly():
    frame = _screen(cells=2).drop(columns=["rowID", "columnID"])
    with pytest.raises(HitScoringError):
        _score_anomalies(frame, control_column="condition",
                         negative_levels="neg")


def test_constant_control_features_are_refused():
    frame = _screen(cells=4)
    for i in range(8):
        frame[f"feature_{i}"] = 1.0
    with pytest.raises(HitScoringError, match="constant across the negative"):
        _score_anomalies(frame, control_column="condition",
                         negative_levels="neg")


def test_control_wells_named_by_spec_and_a_capped_reference_with_treatments():
    frame = _screen(cells=6)
    frame["drug"] = np.where(frame["columnID"] == 5, "hit_compound", "other")
    result = _score_anomalies(frame, negative_wells="c1, c12",
                              treatment_column="drug", max_reference=40,
                              method="knn", neighbours=3)
    assert result.n_reference <= 40 * 2
    assert "treatment" in result.wells.columns
    assert "treatment" in result.cells.columns
    with pytest.raises(HitScoringError):
        _score_anomalies(frame, negative_wells="not a well spec")


def test_identical_control_wells_still_get_a_finite_well_z():
    frame = _screen(cells=4)
    controls = frame["condition"] == "neg"
    for i in range(8):
        frame.loc[controls, f"feature_{i}"] = frame.loc[
            controls, f"feature_{i}"].mean() + np.tile(
                [0.0, 0.1, 0.2, 0.3], int(controls.sum()) // 4)
    result = _score_anomalies(frame, control_column="condition",
                              negative_levels="neg")
    assert np.isfinite(result.wells["anomaly_z"]).all()


def test_embeddings_are_preferred_and_a_single_control_well_splits_its_objects():
    import numpy.random as npr

    from spacr.sp_stats import _anomaly_features

    frame = pd.DataFrame({"emb_0": [1.0], "emb_1": [2.0], "area": [3.0]})
    assert _anomaly_features(frame, []) == ["emb_0", "emb_1"]
    index = np.arange(10)
    first, second = _anomaly_halves(index, ["P"] * 10, [1] * 10, [1] * 10,
                                    npr.default_rng(0))
    assert len(first) == len(second) == 5
    assert not set(first) & set(second)


def test_a_review_crop_that_is_missing_or_unreadable_is_none(tmp_path):
    from spacr.sp_stats import _review_crop

    assert _review_crop(None) is None
    assert _review_crop(str(tmp_path / "gone.png")) is None
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    assert _review_crop(str(bad)) is None


def test_a_report_without_wells_or_known_hits_says_less():
    result = _score_anomalies(_screen(cells=4), control_column="condition",
                              negative_levels="neg")
    assert result.auroc is None
    text = result.report()
    assert "AUROC" not in text and "Most unlike the controls" in text


def test_a_review_with_nothing_drawn_writes_only_the_tables(tmp_path,
                                                           monkeypatch):
    import spacr.sp_stats as sp_stats

    result = _score_anomalies(_screen(cells=4), control_column="condition",
                              negative_levels="neg")
    monkeypatch.setattr(sp_stats, "_draw_anomaly_review",
                        lambda figure, result, target=None: 0)
    written = _write_anomaly_report(result, tmp_path / "anomaly")
    assert "anomaly_review" not in written
    assert set(written) >= {"anomaly_wells", "anomaly_cells"}


def test_a_report_with_no_wells_ranked_names_no_top_well(monkeypatch):
    result = _score_anomalies(_screen(cells=4), control_column="condition",
                              negative_levels="neg")
    monkeypatch.setattr(type(result), "ranked_wells",
                        lambda self: self.wells.head(0))
    assert "Most unlike the controls" not in result.report()


def test_two_control_wells_too_small_to_halve_are_split_by_object():
    import numpy.random as npr

    index = np.arange(3)
    first, second = _anomaly_halves(index, ["P"] * 3, [1, 1, 2], [1, 1, 1],
                                    npr.default_rng(0))
    assert len(first) + len(second) == 3 and min(len(first), len(second)) >= 1


def test_a_plate_with_too_few_controls_is_centred_on_all_controls():
    frame = _screen(cells=6)
    lonely = (frame["plateID"] == "P2") & (frame["condition"] == "neg")
    keep = ~lonely | (frame.index.isin(frame[lonely].index[:2]))
    result = _score_anomalies(frame[keep].reset_index(drop=True),
                              control_column="condition",
                              negative_levels="neg")
    assert len(result.wells)


def test_a_named_plate_column_overrides_the_readers_plate():
    frame = _screen(cells=4).rename(columns={"plateID": "barcode"})
    result = _score_anomalies(frame, control_column="condition",
                              negative_levels="neg", plate_column="barcode")
    assert set(result.wells["plateID"]) == {"P1", "P2"}
