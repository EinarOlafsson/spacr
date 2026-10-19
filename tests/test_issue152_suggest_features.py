"""Issue #152: a Suggest round fits on the tables and feature classes chosen.

The annotated objects are the ``png_list`` rows labelled in the annotation
column; their features come from the measurement tables the user selects.
Each object is matched to its OWN row in its own table, to its parent
cell's row in a parent table, and to the mean of the rows inside that cell
in any other child table.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import active_learning as al

WELLS = (("A01", "r1", "c1"), ("A02", "r1", "c2"),
         ("B01", "r2", "c1"), ("B02", "r2", "c2"))
PER_WELL = 6


def _project(root, crop_mode="cell"):
    """A cell and a nucleus table with two channels, and a crop table.

    ``cell_tag`` and ``nucleus_tag`` are unique per object, so a row taken
    from the wrong object is a wrong number, not a wrong shape. Nucleus
    ``n`` lies in cell ``n``; its label is ``n + 100``.
    """
    db = os.path.join(str(root), "measurements.db")
    cells, nuclei, crops = [], [], []
    for well_index, (well, row_id, column_id) in enumerate(WELLS):
        prcf = f"plate1_{row_id}_{column_id}_f1"
        for obj in range(1, PER_WELL + 1):
            cls = 1 if obj % 2 else 2
            tag = float(well_index * 100 + obj)
            cells.append({
                "plateID": "plate1", "rowID": row_id, "columnID": column_id,
                "fieldID": "f1", "prcf": prcf, "object_label": obj,
                "cell_area": 500.0 + 400.0 * cls,
                "cell_channel_1_mean_intensity": 10.0 * cls,
                "cell_channel_2_mean_intensity": 3.0,
                "cell_tag": tag})
            nuclei.append({
                "plateID": "plate1", "rowID": row_id, "columnID": column_id,
                "fieldID": "f1", "prcf": prcf, "object_label": obj + 100,
                "cell_id": float(obj),
                "nucleus_area": 50.0 + 40.0 * cls,
                "nucleus_channel_1_mean_intensity": 5.0 * cls,
                "nucleus_tag": tag + 0.5})
            if crop_mode == "cell":
                name = f"plate1_{well}_1_{obj}.png"
                crop = {"prcfo": f"{prcf}_o{obj}", "cell_id": f"o{obj}"}
            else:
                name = f"plate1_{well}_1_{obj}_{obj + 100}.png"
                crop = {"prcfo": f"{prcf}_o{obj + 100}",
                        "nucleus_id": f"o{obj + 100}"}
            crops.append({"png_path": f"/nas/src/data/{name}",
                          "file_name": name, "plateID": "plate1",
                          "rowID": row_id, "columnID": column_id,
                          "fieldID": "f1", "annotate": cls, "_tag": tag,
                          **crop})
    con = sqlite3.connect(db)
    try:
        pd.DataFrame(cells).to_sql("cell", con, index=False)
        pd.DataFrame(nuclei).to_sql("nucleus", con, index=False)
        pd.DataFrame(crops).drop(columns=["_tag"]).to_sql(
            "png_list", con, index=False)
        con.commit()
    finally:
        con.close()
    return db, crops


def test_a_nucleus_crop_gets_its_own_nucleus_row(tmp_path):
    db, crops = _project(tmp_path, crop_mode="nucleus")

    features = al.round_features(db, tables=["nucleus"])

    assert len(features) == len(crops)
    assert all(c.startswith("nucleus_") for c in features.columns)
    for crop in crops:
        assert features.loc[crop["png_path"], "nucleus_tag"] == \
            crop["_tag"] + 0.5


def test_a_parent_table_joins_through_the_parent_cell(tmp_path):
    db, crops = _project(tmp_path, crop_mode="nucleus")

    features = al.round_features(db, tables=["nucleus", "cell"])

    for crop in crops:
        row = features.loc[crop["png_path"]]
        assert row["cell_tag"] == crop["_tag"]
        assert row["nucleus_tag"] == crop["_tag"] + 0.5


def test_a_cell_crop_takes_the_mean_of_the_nuclei_inside_it(tmp_path):
    db, crops = _project(tmp_path, crop_mode="cell")

    only_cell = al.round_features(db, tables=["cell"])
    with_nucleus = al.round_features(db, tables=["cell", "nucleus"])

    assert not any(c.startswith("nucleus_") for c in only_cell.columns)
    assert "nucleus_tag" in with_nucleus.columns
    for crop in crops:
        assert with_nucleus.loc[crop["png_path"], "nucleus_tag"] == \
            crop["_tag"] + 0.5


def test_the_default_table_is_the_annotated_object_s_own(tmp_path):
    db, _ = _project(tmp_path, crop_mode="nucleus")

    choices = al._suggest_feature_choices(db)

    assert choices["tables"] == ["cell", "nucleus"]
    assert choices["default_tables"] == ["nucleus"]
    assert choices["channels"] == ["channel_1", "channel_2", al.NO_CHANNEL]
    assert {"intensity", "morphology"} <= set(choices["families"])
    features = al.round_features(db, tables=None)
    assert all(c.startswith("nucleus_") for c in features.columns)


def test_a_channel_selection_keeps_only_that_channel(tmp_path):
    db, _ = _project(tmp_path)

    features = al.round_features(db, tables=["cell"],
                                 channels=["channel_1"])

    assert list(features.columns) == ["cell_channel_1_mean_intensity"]


def test_a_family_selection_keeps_only_that_family(tmp_path):
    db, _ = _project(tmp_path)

    shape = al.round_features(db, tables=["cell"], families=["morphology"])
    shape_without_channel = al.round_features(
        db, tables=["cell"], channels=[al.NO_CHANNEL])

    assert list(shape.columns) == ["cell_area"]
    assert set(shape_without_channel.columns) == {"cell_area", "cell_tag"}


def test_the_selection_reaches_the_training_columns(tmp_path):
    """The round's model is fitted on the selected columns and no others."""
    db, _ = _project(tmp_path)

    result = al.retrain_round(
        db, "annotate", features=al.round_features(
            db, tables=["cell"], channels=["channel_1"]),
        group_by="none", min_labels=4, save_model=True, write_card=False,
        model_dir=str(tmp_path / "models"), round_index=0)

    import joblib
    saved = joblib.load(result.model_path)
    assert saved["features"] == ["cell_channel_1_mean_intensity"]


def test_a_selection_that_leaves_no_column_says_so(tmp_path):
    db, _ = _project(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        al.round_features(db, tables=["cell"], channels=["channel_9"])
    assert "leave no feature column in cell" in str(excinfo.value)


def test_crops_matching_no_object_name_a_key_from_each_side(tmp_path):
    db, _ = _project(tmp_path, crop_mode="nucleus")
    con = sqlite3.connect(db)
    try:
        con.execute("UPDATE png_list SET prcfo = 'other_' || prcfo")
        con.commit()
    finally:
        con.close()

    with pytest.raises(ValueError) as excinfo:
        al.round_features(db, tables=["nucleus"])
    message = str(excinfo.value)
    assert "None of the 24 crops in png_list matches" in message
    assert "'other_plate1_r1_c1_f1_o101'" in message
    assert "'plate1_r1_c1_f1_o101'" in message


def test_a_crop_without_its_own_row_is_left_out(tmp_path):
    """Missing features would become zeros in the fit; the crop is dropped."""
    db, crops = _project(tmp_path, crop_mode="nucleus")
    con = sqlite3.connect(db)
    try:
        con.execute("DELETE FROM nucleus WHERE object_label = 101")
        con.commit()
    finally:
        con.close()

    features = al.round_features(db, tables=["nucleus"])

    assert len(features) == len(crops) - len(WELLS)
    assert not np.isnan(features.to_numpy(dtype=float)).any()
