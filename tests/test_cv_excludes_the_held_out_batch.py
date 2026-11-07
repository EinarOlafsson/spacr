"""Cross-validation can train each fold without its held-out batches.

Under the not-same-compound-or-batch (NSCB) rule a held-out compound is
scored by a model that saw neither that compound nor any plate or imaging
week it was imaged in, because a batch effect shared with training wells
otherwise flatters the score. These tests build 8 x 8 PNG crops and a plate
map with an imaging week, and check what each fold trains on.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

#: well -> (class folder, compound, imaging week), three crops per well.
WELLS = {
    ("plate1", "A01"): ("nc", "cmpA", "w1"),
    ("plate1", "A02"): ("nc", "cmpB", "w1"),
    ("plate1", "A03"): ("pc", "cmpC", "w1"),
    ("plate2", "B01"): ("nc", "cmpA", "w2"),
    ("plate2", "B02"): ("pc", "cmpD", "w2"),
    ("plate3", "C01"): ("pc", "cmpE", "w3"),
    ("plate3", "C02"): ("nc", "cmpF", "w3"),
}


def _dataset(root, wells=WELLS, per_well=3):
    """Write the crop tree and the plate map; return the plate-map path."""
    rng = np.random.default_rng(0)
    rows = []
    for (plate, well), (cls, compound, week) in wells.items():
        folder = root / "train" / cls
        folder.mkdir(parents=True, exist_ok=True)
        for obj in range(per_well):
            pixels = rng.integers(0, 255, (8, 8, 3)).astype(np.uint8)
            Image.fromarray(pixels).save(folder / f"{plate}_{well}_1_{obj + 1}.png")
        rows.append({"plate": plate, "well": well, "compound": compound,
                     "week": week})
    path = root / "plate_map.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _meta(path, index):
    """(compound, week, plate) of a crop from WELLS."""
    plate, well = os.path.basename(path).split("_")[:2]
    _cls, compound, week = WELLS[(plate, well)]
    return (compound, week, plate)[index]


def _folds(root, metadata, **kwargs):
    """CV loaders with small CPU settings, grouped by compound."""
    from spacr.io import generate_cv_loaders

    kwargs.setdefault("group_column", "compound")
    return generate_cv_loaders(
        str(root), n_splits=2, image_size=8, batch_size=4,
        classes=["nc", "pc"], n_jobs=0, group_by="well",
        metadata_path=metadata, **kwargs)


def test_training_never_shares_a_week_with_the_held_out_compound(tmp_path):
    """cmpA is imaged in weeks 1 and 2, so its fold trains on week 3 only."""
    from spacr.io import dataset_filenames

    metadata = _dataset(tmp_path)
    fold_loaders, info = _folds(tmp_path, metadata, batch_column="week")
    by_compound = dict(zip(info["held_out"], fold_loaders))
    train_a, val_a = by_compound["cmpA"]
    assert {_meta(p, 1) for p in dataset_filenames(train_a.dataset)} == {"w3"}
    assert len(dataset_filenames(val_a.dataset)) == 6
    for compound, (train_loader, val_loader) in by_compound.items():
        held_weeks = {_meta(p, 1) for p in dataset_filenames(val_loader.dataset)}
        trained = dataset_filenames(train_loader.dataset)
        assert not held_weeks & {_meta(p, 1) for p in trained}, compound
        assert compound not in {_meta(p, 0) for p in trained}
    table = info["fold_table"].set_index("held_out")
    assert table.loc["cmpA", "held_out_batches"] == "w1|w2"
    assert table.loc["cmpA", "n_train_excluded"] == 9
    assert table.loc["cmpA", "n_train"] == 6
    assert table.loc["cmpE", "held_out_batches"] == "w3"


def test_without_the_rule_the_same_week_is_trained_on(tmp_path):
    """The control: plain leave-one-compound-out keeps week-mates in training."""
    from spacr.io import dataset_filenames

    metadata = _dataset(tmp_path)
    fold_loaders, info = _folds(tmp_path, metadata)
    train_a, _val = dict(zip(info["held_out"], fold_loaders))["cmpA"]
    weeks = {_meta(p, 1) for p in dataset_filenames(train_a.dataset)}
    assert weeks == {"w1", "w2", "w3"}
    assert "n_train_excluded" not in info["fold_table"]


def test_a_class_lost_to_the_rule_is_named(tmp_path, capsys):
    """With plates as batches, a fold can lose every crop of a class.

    Holding out cmpA (plate1) also drops cmpB, the only pc compound, and
    holding out cmpB drops cmpA, leaving plate2's nc crops; holding out cmpC
    keeps both classes. The fold table and the warnings say so.
    """
    wells = {
        ("plate1", "A01"): ("nc", "cmpA", "w1"),
        ("plate1", "A02"): ("pc", "cmpB", "w1"),
        ("plate2", "B01"): ("nc", "cmpC", "w2"),
    }
    metadata = _dataset(tmp_path, wells=wells)
    _loaders, info = _folds(tmp_path, metadata, batch_column="plate")
    table = info["fold_table"].set_index("held_out")
    assert table.loc["cmpA", "train_classes_missing"] == "pc"
    assert table.loc["cmpB", "train_classes_missing"] == "pc"
    assert table.loc["cmpC", "train_classes_missing"] == ""
    assert sum("cannot predict" in w for w in info["warnings"]) == 2
    assert "cannot predict" in capsys.readouterr().out


def test_a_fold_with_nothing_left_to_train_on_is_refused(tmp_path):
    """One batch for everything means the rule leaves no training data."""
    metadata = _dataset(tmp_path)
    frame = pd.read_csv(metadata).assign(week="w1")
    frame.to_csv(metadata, index=False)
    with pytest.raises(ValueError, match="leaves nothing to train on"):
        _folds(tmp_path, metadata, batch_column="week")


def test_plate_batches_come_from_the_crop_names(tmp_path):
    """plate needs no plate-map column: it is read from the file names."""
    from spacr.io import dataset_filenames

    _dataset(tmp_path)
    fold_loaders, _info = _folds(tmp_path, None, group_column=None,
                                 batch_column="plate")
    for train_loader, val_loader in fold_loaders:
        held = {_meta(p, 2) for p in dataset_filenames(val_loader.dataset)}
        trained = {_meta(p, 2) for p in dataset_filenames(train_loader.dataset)}
        assert not held & trained


def test_classify_cv_applies_the_rule_end_to_end(tmp_path, monkeypatch):
    """train_test_model passes the reduced training sets to every fold."""
    import torch.nn as nn

    import spacr.deep_spacr as DS
    from spacr.io import dataset_filenames

    metadata = _dataset(tmp_path)
    overlaps = []

    class _Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 8 * 8, 2)

        def forward(self, x):
            return self.fc(x.flatten(1))

    def fake_train_model(**kwargs):
        trained = {_meta(p, 1) for p in
                   dataset_filenames(kwargs["train_loaders"].dataset)}
        held = {_meta(p, 1) for p in
                dataset_filenames(kwargs["val_loaders"].dataset)}
        overlaps.append(trained & held)
        return _Head(), None

    monkeypatch.setattr(DS, "train_model", fake_train_model)
    out = DS.train_test_model({
        "src": str(tmp_path), "classes": ["nc", "pc"], "train": True,
        "test": False, "epochs": 1, "image_size": 8, "batch_size": 4,
        "n_jobs": 0, "augment": False, "pin_memory": False,
        "normalize": False, "cross_validation_enabled": True,
        "cv_group_by": "well", "cv_group_column": "compound",
        "cv_exclude_batch_column": "week", "well_metadata_path": metadata,
        "evaluation_calibration": "none",
    })
    assert overlaps == [set()] * 6
    per_fold = pd.read_csv(out)
    assert list(per_fold["n_train"]) == [6, 12, 12, 15, 15, 15]
