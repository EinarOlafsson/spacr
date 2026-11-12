"""Item 899, the CNN route: a classifier trained in Classify gets the held-out
evaluation and controls of the feature route (``train_eval_split``).

Leave-one-well-out (or plate, or random) folds come from the cross-validation
folds; the model rows, the majority-class control and the shuffled-label
control (the whole cross-validation again, training labels permuted within
each fold) are written to ``<model folder>/ml_eval/metrics.csv`` with the
columns of :func:`spacr.ml.evaluate_classifier`. The network is replaced by a
fixed model here, so the tests pin the folds, the label permutation and the
table, not a training run.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

WELLS = {"A01": "nc", "A02": "nc", "A03": "pc", "A04": "pc"}


def _dataset(root, per_well=3, seed=0):
    """Dark nc and bright pc crops, ``per_well`` per well, named plate_well_field_object."""
    rng = np.random.default_rng(seed)
    for well, cls in WELLS.items():
        folder = root / "train" / cls
        folder.mkdir(parents=True, exist_ok=True)
        for obj in range(per_well):
            base = 0 if cls == "nc" else 230
            pixels = (base + rng.integers(0, 25, (8, 8, 3))).astype(np.uint8)
            Image.fromarray(pixels).save(folder / f"plate1_{well}_1_{obj + 1}.png")


def _metrics_files(root):
    """Every ml_eval/metrics.csv a run wrote under ``root``."""
    import glob

    return sorted(glob.glob(os.path.join(str(root), "model", "**", "ml_eval", "metrics.csv"),
                            recursive=True))


def _well(path):
    """The well of a crop name."""
    return os.path.basename(path).split("_")[1]


def _settings(root, **extra):
    """Small CPU settings for a held-out Classify run."""
    return {"src": str(root), "classes": ["nc", "pc"], "train": True, "test": False,
            "epochs": 1, "image_size": 8, "batch_size": 4, "n_jobs": 0, "augment": False,
            "model_type": "resnet18",
            "pin_memory": False, "normalize": False, "evaluation_calibration": "none",
            "cv_group_by": "well", "train_eval_split": "leave_one_well_out", **extra}


@pytest.fixture
def fake_training(monkeypatch):
    """Replace the network by a brightness rule and record every training call."""
    import torch
    import torch.nn as nn

    import spacr.deep_spacr as DS
    from spacr.io import dataset_filenames, dataset_labels

    calls = []

    class _Bright(nn.Module):
        """Predicts pc for bright crops and nc for dark ones."""

        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(20.0))

        def forward(self, x):
            mean = x.mean(dim=(1, 2, 3)) - 0.5
            return torch.stack([-mean, mean], dim=1) * self.scale

    def fake_train_model(**kwargs):
        train = kwargs["train_loaders"].dataset
        calls.append({"train": train, "labels": dataset_labels(train),
                      "train_wells": {_well(p) for p in dataset_filenames(train)},
                      "held_wells": {_well(p) for p in
                                     dataset_filenames(kwargs["val_loaders"].dataset)}})
        return _Bright(), None

    monkeypatch.setattr(DS, "train_model", fake_train_model)
    return calls


def test_leave_one_well_out_with_both_controls(tmp_path, fake_training):
    import spacr.deep_spacr as DS
    from spacr.io import dataset_labels

    _dataset(tmp_path)
    settings = _settings(tmp_path, train_eval_controls="majority, shuffled",
                         train_eval_shuffle_repeats=2, train_eval_shuffle_seed=3)
    DS.train_test_model(settings)
    assert len(fake_training) == 4 * 3
    for call in fake_training:
        assert len(call["held_wells"]) == 1 and not call["held_wells"] & call["train_wells"]
    for run in (1, 2):
        rng = np.random.default_rng(3 + run - 1)
        for call in fake_training[4 * run:4 * run + 4]:
            assert isinstance(call["train"], DS._PermutedLabels)
            true = np.asarray(dataset_labels(call["train"].dataset))
            assert call["labels"] == rng.permutation(true).tolist()
    for call in fake_training[:4]:
        assert not isinstance(call["train"], DS._PermutedLabels)
    paths = _metrics_files(tmp_path)
    assert len(paths) == 1 and "shuffled_seed" not in paths[0]
    metrics = pd.read_csv(paths[0])
    assert list(metrics.columns[:3]) == ["role", "model", "fold"]
    assert len(metrics) == 3 * 5
    pooled = metrics[metrics["fold"] == "pooled"].set_index("model")
    assert pooled.loc["resnet18, leave_one_well_out", "accuracy"] == 1.0
    assert pooled.loc["resnet18, leave_one_well_out", "n"] == 12
    assert pooled.loc["control: majority class (nc)", "accuracy"] == 0.5
    assert pooled.loc["control: majority class (nc)", "auroc"] == 0.5
    assert "control: shuffled training labels (mean of 2)" in pooled.index
    folds = metrics.loc[metrics["role"] == "model", "fold"].tolist()
    assert folds == ["pooled", "plate1_A01", "plate1_A02", "plate1_A03", "plate1_A04"]
    assert {"nc precision", "pc recall", "balanced_accuracy", "f1_macro"} <= set(metrics.columns)


def test_the_table_has_the_feature_route_columns(tmp_path, fake_training):
    import spacr.deep_spacr as DS
    from spacr.ml import evaluate_classifier

    _dataset(tmp_path)
    settings = _settings(tmp_path, train_eval_controls="majority")
    DS.train_test_model(settings)
    cnn = pd.read_csv(_metrics_files(tmp_path)[0])
    assert len(fake_training) == 4 and len(cnn) == 10
    frame = pd.DataFrame({"label": ["nc", "nc", "pc", "pc"] * 2, "x": [0, 1, 5, 6] * 2,
                          "well": ["A", "B", "A", "B"] * 2})
    feature = evaluate_classifier(frame, ["x"], {"ml_eval_controls": "majority",
                                                 "ml_model_type": "logistic_regression"})
    assert list(cnn.columns) == list(feature.columns)


def test_split_settings_become_cross_validation_settings():
    from spacr.deep_spacr import _apply_train_eval_split

    plate = {"train_eval_split": "leave_one_plate_out", "train_eval_controls": "majority"}
    _apply_train_eval_split(plate)
    assert plate["cross_validation_enabled"] and plate["cv_group_column"] == "plate"
    random = {"train_eval_split": "random", "cross_validation_folds": 3,
              "cv_group_column": "compound"}
    _apply_train_eval_split(random)
    assert random["cv_group_column"] == "" and random["cross_validation_folds"] == 3
    well = {"train_eval_split": "leave_one_well_out", "cross_validation_folds": 0}
    _apply_train_eval_split(well)
    assert well["cv_group_column"] == "well" and well["cross_validation_folds"] == 5


@pytest.mark.parametrize("bad, message", [
    ({"train_eval_split": "leave_one_field_out"}, "train_eval_split must be"),
    ({"train_eval_split": "random", "train_eval_controls": "majority, oracle"},
     "train_eval_controls must name"),
    ({"train_eval_split": "random", "nested_cv_inner_folds": 2}, "nested_cv_inner_folds"),
])
def test_bad_split_settings_are_refused(bad, message):
    from spacr.deep_spacr import _apply_train_eval_split

    with pytest.raises(ValueError, match=message):
        _apply_train_eval_split(dict(bad))


def test_none_keeps_the_usual_split(tmp_path, fake_training):
    import spacr.deep_spacr as DS

    _dataset(tmp_path)
    settings = _settings(tmp_path, train_eval_split="none", cross_validation_enabled=True,
                         cross_validation_folds=2)
    DS.train_test_model(settings)
    assert len(fake_training) == 2
    assert not _metrics_files(tmp_path)


def test_train_eval_settings_are_registered_alpha():
    from spacr.settings import ALPHA_FEATURES, get_train_test_model_settings

    names = {"train_eval_split", "train_eval_controls", "train_eval_shuffle_seed",
             "train_eval_shuffle_repeats"}
    assert names <= set(ALPHA_FEATURES[899]["settings"])
    defaults = get_train_test_model_settings({"src": "x", "classes": ["a", "b"]})
    assert defaults["train_eval_split"] == "none"
    assert defaults["train_eval_controls"] == "majority, shuffled"
    assert defaults["train_eval_shuffle_seed"] == 0 and defaults["train_eval_shuffle_repeats"] == 1
