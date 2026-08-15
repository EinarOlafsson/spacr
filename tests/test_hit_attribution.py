"""Regression-hit to candidate-cell attribution contracts."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.hit_attribution import (
    HitAttributionError, build_hit_cell_frame, fit_hit_attribution,
    promote_hit_calls, undo_hit_promotion, write_hit_attribution,
)


def _inputs(seed=3, target_wells=8, control_wells=8, cells_per_well=12):
    rng = np.random.default_rng(seed)
    cells, fractions = [], []
    for target, count, offset in ((False, control_wells, 0),
                                  (True, target_wells, 100)):
        for well_index in range(count):
            plate = f"p{1 + well_index % 4}"
            row = f"r{1 + (offset + well_index) // 12}"
            column = f"c{1 + (offset + well_index) % 12}"
            fraction = 0.0 if not target else 0.08 + 0.02 * well_index
            fractions.extend([
                {"plateID": plate, "rowID": row, "columnID": column,
                 "grna": "123_1", "fraction": fraction},
                {"plateID": plate, "rowID": row, "columnID": column,
                 "grna": "other_1", "fraction": 1.0 - fraction},
            ])
            for cell_index in range(cells_per_well):
                cells.append({
                    "prcfo": f"{plate}_{row}_{column}_f1_o{cell_index}",
                    "plateID": plate, "rowID": row, "columnID": column,
                    "fieldID": "f1", "object_label": cell_index,
                    "XGBoost_score": rng.normal(1.5 if target else 0.0, 0.7),
                    "cell_area": rng.normal(2.0 if target else 0.0, 0.6),
                    "cell_texture": rng.normal(1.0 if target else 0.0, 0.7),
                })
    return pd.DataFrame(cells), pd.DataFrame(fractions)


def _fit(**kwargs):
    cells, fractions = _inputs(**kwargs)
    frame = build_hit_cell_frame(
        cells, fractions, target_guides=["123_1"],
        score_column="XGBoost_score", direction="positive")
    return fit_hit_attribution(
        frame, target_gene="123", split_by="well", random_seed=4,
        n_bootstrap=100, n_permutations=150)


def test_crossfit_holds_wells_out_and_excludes_original_score_by_default():
    result = _fit()
    assert result.split_level == "well"
    assert result.cells["hit_like_probability"].between(0, 1).all()
    assert result.cells["attribution_fold"].nunique() >= 3
    assert "XGBoost_score" not in result.feature_columns
    assert {"cell_area", "cell_texture"}.issubset(result.feature_columns)
    assert result.validation["independent_unit"] == "well"
    assert result.validation["n_target_wells"] == 8
    assert result.validation["prevalence_difference"] > 0


def test_duplicate_well_guide_fraction_is_refused():
    cells, fractions = _inputs()
    fractions = pd.concat([fractions, fractions.iloc[[0]]], ignore_index=True)
    with pytest.raises(HitAttributionError, match="more than one row"):
        build_hit_cell_frame(
            cells, fractions, target_guides=["123_1"],
            score_column="XGBoost_score")


def test_manual_score_inclusion_is_explicit_and_warned():
    cells, fractions = _inputs()
    frame = build_hit_cell_frame(
        cells, fractions, target_guides=["123_1"],
        score_column="XGBoost_score")
    result = fit_hit_attribution(
        frame, target_gene="123", split_by="well",
        include_original_score=True, n_bootstrap=20, n_permutations=20)
    assert "XGBoost_score" in result.feature_columns
    assert any("original CV score" in warning for warning in result.warnings)


def test_write_promote_and_undo_never_overwrite_manual_annotation(tmp_path):
    result = _fit(cells_per_well=4)
    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as connection:
        connection.execute(
            "CREATE TABLE png_list (prcfo TEXT PRIMARY KEY, manual_review INTEGER)")
        connection.executemany(
            "INSERT INTO png_list VALUES (?, ?)",
            [(value, 7) for value in result.cells["prcfo"]])
    run_id = write_hit_attribution(str(db), result, run_id="run-1")
    promotion = promote_hit_calls(
        str(db), result, run_id=run_id,
        annotation_column="candidate_review")
    with sqlite3.connect(db) as connection:
        rows = pd.read_sql_query(
            "SELECT manual_review, candidate_review FROM png_list", connection)
    assert set(rows["manual_review"]) == {7}
    assert rows["candidate_review"].notna().any()
    assert undo_hit_promotion(str(db), promotion) > 0
    with sqlite3.connect(db) as connection:
        rows = pd.read_sql_query(
            "SELECT manual_review, candidate_review FROM png_list", connection)
    assert set(rows["manual_review"]) == {7}
    assert rows["candidate_review"].isna().all()
