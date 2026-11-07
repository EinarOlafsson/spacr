"""Cross-validation folds by any metadata column, such as a compound.

``cv_group_by`` holds out cells, fields, wells or plates. A benchmark like
BBBC021 asks a different question: does the model recognise the mechanism of
a compound it has never seen? That needs one fold per compound, and the
compound is not in the crop name; it is in the plate map. These tests build
8 x 8 PNG crops named ``plate_well_field_object`` and a plate-map CSV, then
check the folds that come out.
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
    ("plate2", "B03"): ("pc", "cmpC", "w2"),
}


def _dataset(root, wells=WELLS, per_well=3, seed=0):
    """Write the crop tree and the plate map; return the plate-map path."""
    rng = np.random.default_rng(seed)
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


def _compound_of(path):
    """Look a crop's compound up in WELLS from its file name."""
    plate, well = os.path.basename(path).split("_")[:2]
    return WELLS[(plate, well)][1]


def _loaders(root, metadata, **kwargs):
    """Build CV loaders over ``root`` with small, CPU-only settings."""
    from spacr.io import generate_cv_loaders

    kwargs.setdefault("group_by", "well")
    return generate_cv_loaders(
        str(root), n_splits=kwargs.pop("n_splits", 2), image_size=8,
        batch_size=4, classes=["nc", "pc"], n_jobs=0, metadata_path=metadata,
        **kwargs)


def test_one_fold_per_compound_holds_out_exactly_that_compound(tmp_path):
    """Four compounds give four folds, each validating one compound only."""
    from spacr.io import dataset_filenames

    metadata = _dataset(tmp_path)
    fold_loaders, info = _loaders(tmp_path, metadata,
                                  group_column="compound")
    assert info["held_out"] == ["cmpA", "cmpB", "cmpC", "cmpD"]
    assert list(info["fold_table"]["held_out"]) == info["held_out"]
    assert len(fold_loaders) == 4
    seen = []
    for compound, (train_loader, val_loader) in zip(info["held_out"],
                                                    fold_loaders):
        validation = dataset_filenames(val_loader.dataset)
        training = dataset_filenames(train_loader.dataset)
        assert {_compound_of(p) for p in validation} == {compound}
        assert compound not in {_compound_of(p) for p in training}
        assert len(validation) + len(training) == 18
        seen.extend(validation)
    assert sorted(seen) == sorted(dataset_filenames(info["dataset"]))


def test_n_splits_is_not_used_with_a_group_column(tmp_path):
    """The fold count follows the values, whatever cross_validation_folds says."""
    metadata = _dataset(tmp_path)
    fold_loaders, _info = _loaders(tmp_path, metadata, n_splits=0,
                                   group_column="compound")
    assert len(fold_loaders) == 4


def test_a_level_in_the_crop_name_needs_no_plate_map(tmp_path):
    """Grouping by plate reads the plate from the names; no table needed."""
    _dataset(tmp_path)
    fold_loaders, info = _loaders(tmp_path, None, group_column="plate",
                                  group_by="well")
    assert info["held_out"] == ["plate1", "plate2"]
    assert len(fold_loaders) == 2


def test_a_metadata_column_without_a_plate_map_is_refused(tmp_path):
    """A compound column cannot be invented from the crop names."""
    _dataset(tmp_path)
    with pytest.raises(ValueError, match="well_metadata_path"):
        _loaders(tmp_path, None, group_column="compound")


def test_a_well_missing_from_the_plate_map_is_refused(tmp_path):
    """Every crop needs a value; a missing well is named in the error."""
    metadata = _dataset(tmp_path)
    frame = pd.read_csv(metadata)
    frame = frame[frame["well"] != "B02"]
    frame.to_csv(metadata, index=False)
    with pytest.raises(ValueError, match=r"3 of 18 .*plate2_r2_c2"):
        _loaders(tmp_path, metadata, group_column="compound")


def test_an_unknown_column_lists_the_columns_there_are(tmp_path):
    """A misspelt column name gets the table's real columns back."""
    metadata = _dataset(tmp_path)
    with pytest.raises(ValueError, match="no column 'moa'.*compound"):
        _loaders(tmp_path, metadata, group_column="moa")


def test_two_rows_of_one_well_that_disagree_are_refused(tmp_path):
    """A plate map giving a well two compounds is an error, not a guess."""
    metadata = _dataset(tmp_path)
    frame = pd.read_csv(metadata)
    extra = frame.iloc[[0]].assign(compound="cmpZ")
    pd.concat([frame, extra]).to_csv(metadata, index=False)
    with pytest.raises(ValueError, match="two different values"):
        _loaders(tmp_path, metadata, group_column="compound")


def test_row_and_column_identities_are_read_as_well(tmp_path):
    """A plate map with rowID/columnID instead of a well name also works."""
    from spacr.classifier_evaluation import _crop_metadata_values

    _dataset(tmp_path)
    path = tmp_path / "rc.csv"
    pd.DataFrame({"plateID": ["plate1", "plate1"], "rowID": ["r1", "r1"],
                  "columnID": ["c1", "c2"],
                  "compound": ["cmpA", "cmpB"]}).to_csv(path, index=False)
    values = _crop_metadata_values(
        ["/x/plate1_A01_1_1.png", "/x/plate1_A02_1_9.png"], "compound",
        str(path))
    assert values.tolist() == ["cmpA", "cmpB"]


def test_classify_cv_runs_leave_one_compound_out(tmp_path, monkeypatch):
    """train_test_model trains one model per compound and scores every crop once."""
    import torch.nn as nn

    import spacr.deep_spacr as DS
    from spacr.io import dataset_filenames

    metadata = _dataset(tmp_path)
    held_by_call = []

    class _Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 8 * 8, 2)

        def forward(self, x):
            return self.fc(x.flatten(1))

    def fake_train_model(**kwargs):
        validation = dataset_filenames(kwargs["val_loaders"].dataset)
        held_by_call.append({_compound_of(p) for p in validation})
        return _Head(), None

    monkeypatch.setattr(DS, "train_model", fake_train_model)
    out = DS.train_test_model({
        "src": str(tmp_path), "classes": ["nc", "pc"], "train": True,
        "test": False, "epochs": 1, "image_size": 8, "batch_size": 4,
        "n_jobs": 0, "augment": False, "pin_memory": False,
        "normalize": False, "cross_validation_enabled": True,
        "cv_group_by": "well", "cv_group_column": "compound",
        "well_metadata_path": metadata, "evaluation_calibration": "none",
    })
    assert held_by_call == [{"cmpA"}, {"cmpB"}, {"cmpC"}, {"cmpD"}]
    assert out.endswith("_cv4_per_fold.csv")
    per_fold = pd.read_csv(out)
    assert list(per_fold["fold"]) == [1, 2, 3, 4]
    predictions = pd.read_csv(os.path.join(os.path.dirname(out), "evaluation",
                                           "oof_predictions.csv"))
    assert len(predictions) == 18
    composition = [p for p in os.listdir(os.path.dirname(out))
                   if p.endswith("_fold_composition.csv")]
    table = pd.read_csv(os.path.join(os.path.dirname(out), composition[0]))
    assert list(table["held_out"]) == ["cmpA", "cmpB", "cmpC", "cmpD"]


def test_nested_cv_with_a_group_column_is_refused(tmp_path, monkeypatch):
    """Inner folds come from cross_validation_folds, so the two cannot mix."""
    import spacr.deep_spacr as DS

    metadata = _dataset(tmp_path)
    monkeypatch.setattr(DS, "train_model", lambda **k: (None, None))
    with pytest.raises(ValueError, match="nested_cv_inner_folds"):
        DS.train_test_model({
            "src": str(tmp_path), "classes": ["nc", "pc"], "train": True,
            "test": False, "epochs": 1, "image_size": 8, "batch_size": 4,
            "n_jobs": 0, "augment": False, "pin_memory": False,
            "normalize": False, "cross_validation_folds": 2,
            "nested_cv_inner_folds": 2, "cv_group_by": "well",
            "cv_group_column": "compound", "well_metadata_path": metadata,
        })
