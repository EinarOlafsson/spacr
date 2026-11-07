"""Predictions are summarised per well and per treatment.

Profiling benchmarks such as BBBC021 score a classifier by treatment: the
mean class probability over a treatment's wells, then the most probable
class. These tests check that arithmetic on hand-made probabilities, then the
two places Classify writes it: the tar inference CSV and the out-of-fold
predictions of cross-validation.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image


def _plate_map(path, rows):
    """Write a plate map with plate, well, compound and concentration."""
    pd.DataFrame(rows, columns=["plate", "well", "compound",
                                "concentration"]).to_csv(path, index=False)
    return str(path)


def test_well_and_treatment_means_and_counts():
    """A treatment averages its wells, each well its objects."""
    from spacr.classifier_evaluation import _prediction_score_tables

    probabilities = np.array([
        [0.9, 0.1], [0.7, 0.3], [0.2, 0.8],
        [0.4, 0.6],
        [0.1, 0.9], [0.3, 0.7],
    ])
    predicted = probabilities.argmax(axis=1)
    wells = ["p1_r1_c1"] * 3 + ["p1_r1_c2"] + ["p1_r1_c3"] * 2
    treatments = [("cmpA",)] * 4 + [("cmpB",)] * 2
    labels = [0, 0, 0, 0, 1, 1]
    per_well, per_treatment = _prediction_score_tables(
        probabilities, predicted, wells, classes=["nc", "pc"],
        treatments=treatments, treatment_columns=["compound"], labels=labels)

    well = per_well.set_index("prc")
    assert well.loc["p1_r1_c1", "n_objects"] == 3
    assert well.loc["p1_r1_c1", "mean_prob_nc"] == pytest.approx(0.6)
    assert well.loc["p1_r1_c1", "n_pred_nc"] == 2
    assert well.loc["p1_r1_c1", "n_pred_pc"] == 1
    assert well.loc["p1_r1_c2", "predicted_label"] == 1
    assert not well.loc["p1_r1_c2", "correct"]

    treatment = per_treatment.set_index("compound")
    assert treatment.loc["cmpA", "n_wells"] == 2
    assert treatment.loc["cmpA", "n_objects"] == 4
    assert treatment.loc["cmpA", "mean_prob_nc"] == pytest.approx((0.6 + 0.4) / 2)
    assert treatment.loc["cmpA", "n_pred_pc"] == 2
    assert treatment.loc["cmpB", "mean_prob_pc"] == pytest.approx(0.8)
    assert treatment.loc["cmpB", "predicted_label"] == 1
    assert list(treatment["correct"]) == [True, True]


def test_a_well_counts_once_however_many_objects_it_has():
    """Object means and well means differ; the treatment uses well means."""
    from spacr.classifier_evaluation import _prediction_score_tables

    probabilities = np.array([[0.9, 0.1]] * 9 + [[0.1, 0.9]])
    wells = ["p1_r1_c1"] * 9 + ["p1_r1_c2"]
    _well, per_treatment = _prediction_score_tables(
        probabilities, probabilities.argmax(1), wells,
        treatments=[("t",)] * 10, treatment_columns=["compound"])
    assert per_treatment.loc[0, "mean_prob_class_0"] == pytest.approx(0.5)
    assert per_treatment.loc[0, "n_pred_class_0"] == 9


def test_treatment_of_two_columns_is_one_row_per_pair(tmp_path):
    """compound and concentration together name the treatment."""
    from spacr.classifier_evaluation import _write_prediction_scores

    plate_map = _plate_map(tmp_path / "map.csv", [
        ("p1", "A01", "cmpA", 0.1), ("p1", "A02", "cmpA", 1.0),
        ("p1", "A03", "cmpA", 1.0)])
    probabilities = np.array([[0.2, 0.8], [0.6, 0.4], [0.8, 0.2]])
    out = _write_prediction_scores(
        str(tmp_path / "result"), probabilities, probabilities.argmax(1),
        ["p1_r1_c1", "p1_r1_c2", "p1_r1_c3"],
        treatment_setting="compound, concentration", metadata_path=plate_map)
    table = pd.read_csv(out["per_treatment"])
    assert list(table["concentration"]) == [0.1, 1.0]
    assert list(table["n_wells"]) == [1, 2]
    assert table.loc[1, "mean_prob_class_0"] == pytest.approx(0.7)
    assert os.path.isfile(str(tmp_path / "result_per_well.csv"))
    assert out["treatment_accuracy"] is None


def test_wells_missing_from_the_plate_map_are_left_out(tmp_path, capsys):
    """Scoring a long inference does not fail on one unmapped well."""
    from spacr.classifier_evaluation import _write_prediction_scores

    plate_map = _plate_map(tmp_path / "map.csv", [("p1", "A01", "cmpA", 1)])
    probabilities = np.array([[0.2, 0.8], [0.6, 0.4]])
    out = _write_prediction_scores(
        str(tmp_path / "r"), probabilities, [1, 0], ["p1_r1_c1", "p1_r1_c2"],
        treatment_setting=["compound"], metadata_path=plate_map)
    table = pd.read_csv(out["per_treatment"])
    assert list(table["n_objects"]) == [1]
    assert "1 of 2 objects have no compound" in capsys.readouterr().out
    assert len(pd.read_csv(out["per_well"])) == 2


def test_tar_predictions_get_per_well_and_per_treatment_tables(tmp_path):
    """apply_model_to_tar's frame is scored by its prc wells."""
    import spacr.deep_spacr as DS
    from spacr.utils import process_vision_results

    plate_map = _plate_map(tmp_path / "map.csv", [
        ("plate1", "A01", "cmpA", 1), ("plate1", "A02", "cmpB", 1)])
    frame = pd.DataFrame({
        "path": ["plate1_A01_1_1.png", "plate1_A01_1_2.png",
                 "plate1_A02_1_1.png"],
        "pred": [0.9, 0.7, 0.2]})
    frame = process_vision_results(frame, 0.5)
    result = tmp_path / "out_result.csv"
    scores = DS._score_tar_predictions(frame, str(result), {
        "prediction_treatment_column": "compound",
        "well_metadata_path": plate_map,
        "class_folder_names": ["nc", "pc"]})
    per_well = pd.read_csv(scores["per_well"])
    assert list(per_well["n_objects"]) == [2, 1]
    assert per_well.loc[0, "mean_prob_pc"] == pytest.approx(0.8)
    per_treatment = pd.read_csv(scores["per_treatment"]).set_index("compound")
    assert per_treatment.loc["cmpA", "predicted_label"] == 1
    assert per_treatment.loc["cmpB", "n_pred_nc"] == 1
    assert scores["per_well"].endswith("out_result_per_well.csv")


def test_multiclass_tar_predictions_use_every_class_column(tmp_path):
    """With prob_class_<i> columns, all classes are averaged."""
    import spacr.deep_spacr as DS
    from spacr.utils import process_vision_results

    frame = pd.DataFrame({
        "path": ["plate1_A01_1_1.png", "plate1_A01_1_2.png"],
        "pred": [0.6, 0.5], "predicted_label": [2, 0],
        "prob_class_0": [0.1, 0.5], "prob_class_1": [0.3, 0.2],
        "prob_class_2": [0.6, 0.3]})
    frame = process_vision_results(frame, 0.5)
    scores = DS._score_tar_predictions(frame, str(tmp_path / "m.csv"), {})
    per_well = pd.read_csv(scores["per_well"])
    assert per_well.loc[0, "mean_prob_class_2"] == pytest.approx(0.45)
    assert per_well.loc[0, "n_pred_class_0"] == 1
    assert scores["per_treatment"] is None


def test_cross_validation_scores_treatments_out_of_fold(tmp_path, monkeypatch):
    """Classify CV writes per-treatment accuracy from held-out predictions."""
    import torch
    import torch.nn as nn

    import spacr.deep_spacr as DS

    rng = np.random.default_rng(0)
    rows = []
    for cls, wells in (("nc", ("A01", "A02")), ("pc", ("A03", "A04"))):
        folder = tmp_path / "train" / cls
        folder.mkdir(parents=True)
        for well in wells:
            for obj in range(3):
                base = 0 if cls == "nc" else 230
                pixels = (base + rng.integers(0, 25, (8, 8, 3))).astype(np.uint8)
                Image.fromarray(pixels).save(folder / f"plate1_{well}_1_{obj + 1}.png")
            rows.append(("plate1", well, f"cmp_{well}", 1))
    plate_map = _plate_map(tmp_path / "map.csv", rows)

    class _Bright(nn.Module):
        """Predicts pc for bright crops, nc for dark ones."""

        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(20.0))

        def forward(self, x):
            mean = x.mean(dim=(1, 2, 3)) - 0.5
            return torch.stack([-mean, mean], dim=1) * self.scale

    monkeypatch.setattr(DS, "train_model", lambda **k: (_Bright(), None))
    out = DS.train_test_model({
        "src": str(tmp_path), "classes": ["nc", "pc"], "train": True,
        "test": False, "epochs": 1, "image_size": 8, "batch_size": 4,
        "n_jobs": 0, "augment": False, "pin_memory": False,
        "normalize": False, "cross_validation_enabled": True,
        "cv_group_by": "well", "cv_group_column": "compound",
        "well_metadata_path": plate_map,
        "prediction_treatment_column": "compound",
        "evaluation_calibration": "none",
    })
    stem = out[:-len("_per_fold.csv")]
    per_treatment = pd.read_csv(f"{stem}_per_treatment.csv")
    assert len(per_treatment) == 4
    assert per_treatment["correct"].all()
    assert list(per_treatment["true_label"]) == [0, 0, 1, 1]
    per_well = pd.read_csv(f"{stem}_per_well.csv")
    assert list(per_well["n_objects"]) == [3, 3, 3, 3]
