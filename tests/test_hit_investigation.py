"""Miniature file-to-database regression-hit investigation workflow."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from spacr.hit_investigation import evaluate_blinded_reviews, investigate_hit


def _write_screen(tmp_path):
    rng = np.random.default_rng(17)
    database = tmp_path / "measurements.db"
    cells, crops, predictions, fractions = [], [], [], []
    object_number = 0
    for plate_index in range(3):
        plate = f"plate{plate_index + 1}"
        for well_index, fraction in enumerate((0.0, 0.25, 0.0, 0.65)):
            row, column, field = "r01", f"c{well_index + 1:02d}", "f1"
            prcf = f"{plate}_{row}_{column}_{field}"
            fractions.extend([
                {"plateID": plate, "rowID": row, "columnID": column,
                 "grna": "EAF1_1", "fraction": fraction},
                {"plateID": plate, "rowID": row, "columnID": column,
                 "grna": "NTC", "fraction": 1.0 - fraction},
            ])
            for label in range(1, 13):
                object_number += 1
                identity = int(rng.random() < fraction)
                area = rng.normal(2.2 * identity + 0.08 * plate_index, 0.55)
                texture = rng.normal(1.4 * identity, 0.65)
                prcfo = f"{prcf}_o{label}"
                path = str(tmp_path / "crops" / f"{prcfo}.png")
                cells.append({
                    "prcfo": prcfo, "prcf": prcf, "plateID": plate,
                    "rowID": row, "columnID": column, "fieldID": field,
                    "object_label": label, "cell_area": area,
                    "cell_texture": texture,
                })
                crops.append({
                    "cell_id": f"o{label}", "png_path": path,
                    "plateID": plate, "rowID": row,
                    "columnID": column, "fieldID": field,
                })
                predictions.append({
                    "prcfo": prcfo,
                    "phenotype_score": 1 / (1 + np.exp(-area)),
                })
    with sqlite3.connect(database) as connection:
        pd.DataFrame(cells).to_sql("cell", connection, index=False)
        pd.DataFrame(crops).to_sql("png_list", connection, index=False)
    prediction_file = tmp_path / "predictions.csv"
    fraction_file = tmp_path / "guide_fractions.csv"
    pd.DataFrame(predictions).to_csv(prediction_file, index=False)
    pd.DataFrame(fractions).to_csv(fraction_file, index=False)
    results = tmp_path / "regression_run"
    results.mkdir()
    pd.DataFrame({"gene": ["EAF1"], "effect": [0.7], "q_value": [0.01]}).to_csv(
        results / "results_gene.csv", index=False)
    return database, prediction_file, fraction_file, results, len(cells)


def test_exact_regression_run_reaches_versioned_cells_embedding_and_gallery(tmp_path):
    database, predictions, fractions, results, n_cells = _write_screen(tmp_path)
    payload = investigate_hit({
        "db_path": str(database),
        "predictions_file": str(predictions),
        "guide_fractions_file": str(fractions),
        "results_folder": str(results),
        "target_gene": "EAF1",
        "target_guides": ["EAF1_1"],
        "score_column": "phenotype_score",
        "hit_effect": 0.7,
        "hit_fdr": 0.01,
        "hit_feature_columns": ["cell_area", "cell_texture"],
        "hit_bootstrap": 30,
        "hit_permutations": 40,
        "hit_pipeline_permutations": 3,
        "hit_gallery_per_stratum": 3,
        "verbose": False,
    })

    result = payload["result"]
    assert len(result.cells) == n_cells
    assert result.split_level == "plate"
    assert result.source_regression_run.startswith(str(results.resolve()))
    assert "#sha256=" in result.source_regression_run
    assert set(result.feature_columns) == {"cell_area", "cell_texture"}
    assert result.validation["refitted_permutations"] == 3
    assert 0 <= result.validation["guide_fraction_refit_p_value"] <= 1
    assert 0 <= result.validation["well_label_refit_p_value"] <= 1
    assert payload["embedding"].shape[0] == n_cells
    assert set(result.threshold_sensitivity["threshold"]).issuperset(
        {0.5, 0.7, 0.9})
    assert set(payload["gallery_key"]["review_stratum"]).issuperset(
        {"high", "borderline", "low", "control_false_looking"})
    assert "hit_like_probability" not in payload["gallery"]
    assert {"review_id", "reviewer_id", "reviewer_label"}.issubset(
        payload["gallery"])
    assert all((results / "hit_investigation" / "EAF1" / name).is_file()
               for name in (
                   "candidate_cells.csv", "well_level_evidence.csv",
                   "independent_guide_evidence.csv",
                   "threshold_sensitivity.csv",
                   "control_fitted_embedding.csv",
                   "blinded_review_gallery_manifest.csv",
                   "blinded_review_gallery_key.csv", "manifest.json"))
    with sqlite3.connect(database) as connection:
        stored = connection.execute(
            "SELECT COUNT(*) FROM object_hit_attribution").fetchone()[0]
        assert stored == n_cells
        columns = {row[1] for row in connection.execute(
            'PRAGMA table_info("png_list")')}
    assert not any(column.endswith("hit_like") for column in columns)


def test_blinded_review_report_includes_calibration_and_agreement():
    key = pd.DataFrame({
        "review_id": ["a", "b", "c", "d"],
        "hit_like_probability": [0.9, 0.8, 0.2, 0.1],
    })
    reviews = pd.DataFrame([
        {"review_id": item, "reviewer_id": reviewer,
         "reviewer_label": label}
        for reviewer, labels in (("r1", [1, 1, 0, 0]),
                                 ("r2", [1, 0, 0, 0]))
        for item, label in zip(key["review_id"], labels)
    ])
    report = evaluate_blinded_reviews(reviews, key)
    assert report["n_reviewers"] == 2
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert 0 <= report["brier_score"] <= 1
    assert -1 <= report["mean_pairwise_cohen_kappa"] <= 1
