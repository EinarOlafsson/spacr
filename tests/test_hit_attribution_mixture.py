"""Hierarchical guide-fraction attribution on a planted multi-plate screen."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from spacr.hit_attribution import (
    build_hit_cell_frame, fit_hit_attribution, promote_hit_calls,
    undo_hit_promotion, write_hit_attribution,
)


def _screen(seed=11, cells_per_well=24):
    rng = np.random.default_rng(seed)
    cells, fractions = [], []
    designs = [0.0, 0.2, 0.45, 0.75, 0.0, 0.3, 0.6, 0.0]
    for plate in range(3):
        for well, fraction in enumerate(designs):
            row, column = f"r{well // 4 + 1}", f"c{well % 4 + 1}"
            guide = "EAF1_1" if well % 2 == 0 else "EAF1_2"
            fractions += [
                {"plateID": f"p{plate}", "rowID": row, "columnID": column,
                 "grna": guide, "fraction": fraction},
                {"plateID": f"p{plate}", "rowID": row, "columnID": column,
                 "grna": "NTC", "fraction": 1 - fraction},
            ]
            for cell in range(cells_per_well):
                truth = int(rng.random() < fraction)
                area = rng.normal(2.5 * truth + 0.08 * plate, 0.7)
                texture = rng.normal(1.4 * truth, 0.85)
                score = 1 / (1 + np.exp(-(area + rng.normal(0, 0.4))))
                cells.append({
                    "prcfo": f"p{plate}_{row}_{column}_f1_o{cell}",
                    "plateID": f"p{plate}", "rowID": row,
                    "columnID": column, "fieldID": "f1",
                    "object_label": cell, "cell_area": area,
                    "cell_texture": texture, "noise": rng.normal(),
                    "phenotype_score": score, "planted_identity": truth,
                })
    cell_frame = pd.DataFrame(cells)
    fraction_frame = pd.DataFrame(fractions)
    joined = build_hit_cell_frame(
        cell_frame, fraction_frame, target_guides=["EAF1_1", "EAF1_2"],
        score_column="phenotype_score")
    return joined


def _fit():
    return fit_hit_attribution(
        _screen(), target_gene="EAF1",
        feature_columns=["cell_area", "cell_texture", "noise"],
        split_by="plate", threshold=0.7, n_bootstrap=80,
        n_permutations=80, random_seed=5)


def test_plate_crossfit_recovers_planted_cells_without_original_cv_score():
    result = _fit()
    assert result.split_level == "plate"
    assert "phenotype_score" not in result.feature_columns
    assert result.cells["attribution_fold"].nunique() == 3
    assert roc_auc_score(
        result.cells["planted_identity"],
        result.cells["hit_like_probability"]) > 0.82
    assert result.validation["prevalence_difference"] > 0.15
    assert result.validation["bootstrap_ci_low"] > 0
    assert set(result.guide_evidence["guide"]) == {"EAF1_1", "EAF1_2"}


def test_probability_is_learned_not_forced_to_equal_well_read_fraction():
    result = _fit()
    difference = np.abs(
        result.wells["mean_hit_like_probability"] -
        result.wells["target_guide_fraction"])
    assert (difference > 0.03).any()
    assert "not observed guide identity" in result.summary()


def test_versioned_storage_and_fresh_annotation_promotion_are_reversible(tmp_path):
    result = _fit()
    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE png_list (prcfo TEXT PRIMARY KEY)")
        connection.executemany(
            "INSERT INTO png_list VALUES (?)",
            [(value,) for value in result.cells["prcfo"]])
    run_id = write_hit_attribution(str(db), result, run_id="eaf1-run")
    with sqlite3.connect(db) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM object_hit_attribution").fetchone()[0] == len(result.cells)
    promotion = promote_hit_calls(
        str(db), result, run_id=run_id, annotation_column="eaf1_hit_like")
    expected = int(result.cells["hit_like_call"].sum())
    with sqlite3.connect(db) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM png_list WHERE eaf1_hit_like=1").fetchone()[0] == expected
    assert undo_hit_promotion(str(db), promotion) == expected
