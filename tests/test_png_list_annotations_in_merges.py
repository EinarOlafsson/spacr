"""Annotations made in the Annotate screen survive merging onto cells.

The shape these tests use is the one a real measurements.db has after cell
crops are annotated: ``png_list`` is one row per crop, its ``file_name`` is
the crop's own name (``plate_A_1_1_17.png``) rather than the image name the
object tables carry (``plate_A_1_1``), and its object id is text (``o17``).
Merging cell + pathogen + png_list used to match no crop at all and to
carry only the crop path, never the annotation; a custom merge onto a
pathogen table that had both ``cell_id`` and ``pathogen_cell_id`` produced
two columns of the same name, and the Graph Builder failed to load it with
"The truth value of a Series is ambiguous".
"""
import sqlite3

import pandas as pd
import pytest

from spacr import derived_tables as d
from spacr.merge_tables import MergeError, _png_annotation_columns, merge_tables

IMAGE = {"plateID": "plate1", "rowID": "r1", "columnID": "c1", "fieldID": "1"}


def _cells():
    return pd.DataFrame({**{k: [v] * 3 for k, v in IMAGE.items()},
                         "file_name": ["plate1_A_1_1"] * 3,
                         "object_label": [1, 2, 3],
                         "cell_area": [100.0, 200.0, 300.0]})


def _pathogens():
    return pd.DataFrame({**{k: [v] * 4 for k, v in IMAGE.items()},
                         "file_name": ["plate1_A_1_1"] * 4,
                         "object_label": [1, 2, 3, 4],
                         "cell_id": [1.0, 1.0, 2.0, 3.0],
                         "pathogen_cell_id": [1, 1, 2, 3],
                         "pathogen_area": [5.0, 6.0, 7.0, 8.0]})


def _crops():
    """Crops of cells 1 and 2 only; cell 3 was never cropped."""
    return pd.DataFrame({
        "png_path": ["/crops/plate1_A_1_1_1.png", "/crops/plate1_A_1_1_2.png"],
        "file_name": ["plate1_A_1_1_1.png", "plate1_A_1_1_2.png"],
        **{k: [v] * 2 for k, v in IMAGE.items()},
        "prcfo": ["plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o2"],
        "cell_id": ["o1", "o2"],
        "annotate": [1, 2],
        "annotate_verdict": [None, 1],
        "condition": ["ctrl", "drug"],
        "al_prob_0": [0.9, 0.2],
        "al_prob_1": [0.1, 0.8],
    })


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "measurements.db")
    with sqlite3.connect(path) as con:
        _cells().to_sql("cell", con, index=False)
        _pathogens().to_sql("pathogen", con, index=False)
        _crops().to_sql("png_list", con, index=False)
    return path


def test_the_default_merge_carries_annotations_onto_each_cell(db):
    merged = merge_tables(db, ["cell", "pathogen", "png_list"]).set_index("object_label")

    assert len(merged) == 3
    assert merged.loc[1, "png_list_annotate"] == 1
    assert merged.loc[2, "png_list_annotate"] == 2
    assert pd.isna(merged.loc[3, "png_list_annotate"])
    assert merged.loc[2, "png_list_condition"] == "drug"
    assert merged.loc[2, "png_list_annotate_verdict"] == 1
    assert merged.loc[1, "png_list_path"] == "/crops/plate1_A_1_1_1.png"
    assert merged.loc[1, "pathogen_count"] == 2


def test_model_probabilities_and_crop_metadata_are_not_carried(db):
    merged = merge_tables(db, ["cell", "png_list"])

    assert not [c for c in merged.columns if c.startswith("png_list_al_prob")]
    for metadata in ("png_list_prcfo", "png_list_cell_id", "png_list_file_name",
                     "png_list_png_path"):
        assert metadata not in merged.columns


def test_the_annotation_column_rule():
    columns = list(_crops().columns) + ["prediction_probability_class_0", "throw"]
    assert _png_annotation_columns(columns, exclude={"png_path"}) == [
        "annotate", "annotate_verdict", "condition", "throw"]


def test_an_object_cropped_twice_keeps_the_labelled_crop(tmp_path):
    again = _crops().iloc[[0]].assign(png_path="/crops/again/plate1_A_1_1_1.png")
    crops = pd.concat([_crops().assign(annotate=[None, 2]), again], ignore_index=True)
    path = str(tmp_path / "m.db")
    with sqlite3.connect(path) as con:
        _cells().to_sql("cell", con, index=False)
        crops.to_sql("png_list", con, index=False)
    merged = merge_tables(path, ["cell", "png_list"]).set_index("object_label")

    assert len(merged) == 3
    assert merged.loc[1, "png_list_annotate"] == 1


def test_a_saved_default_merge_reads_with_annotations(db):
    frame, report = d.execute(db, d.default_definition(db, ["cell", "pathogen", "png_list"]))

    assert report["output_rows"] == 3
    assert frame["png_list_annotate"].notna().sum() == 2


def test_a_custom_png_list_join_matches_crop_names_and_text_ids(db):
    definition = d.default_definition(db, ["cell", "png_list"])
    definition.update(mode="custom", acknowledged=True)
    definition["joins"][0]["relationship"] = "one-to-one"

    frame, report = d.execute(db, definition)

    join = report["joins"][0]
    # Cell 3 was never cropped; every crop found its cell.
    assert (join["unmatched_base"], join["unmatched_child"]) == (1, 0)
    assert len(frame) == 2
    assert sorted(frame["png_list_annotate"].tolist()) == [1, 2]


def test_a_custom_merge_onto_a_base_with_both_id_spellings_has_unique_columns(db):
    definition = d.default_definition(db, ["cell", "pathogen"], base="pathogen")
    definition.update(mode="custom", acknowledged=True)
    definition["joins"][0].update(
        left_keys=list(IMAGE) + ["file_name", "object_label"],
        right_keys=list(IMAGE) + ["file_name", "object_label"],
        relationship="one-to-one", how="left")

    frame, _report = d.execute(db, definition)

    assert not frame.columns.duplicated().any()
    assert {"cell_id", "pathogen_cell_id"} <= set(frame.columns)


def test_repeated_output_columns_are_refused(db, monkeypatch):
    real = d._read

    def doubled(path, table):
        frame = real(path, table)
        if table == "cell":
            frame = pd.concat([frame, frame[["cell_area"]]], axis=1)
        return frame

    monkeypatch.setattr(d, "_read", doubled)
    definition = d.default_definition(db, ["cell", "png_list"])
    definition.update(mode="custom", acknowledged=True)
    definition["joins"][0]["relationship"] = "one-to-one"

    with pytest.raises(MergeError, match="more than one column called cell_area"):
        d.execute(db, definition)


def test_column_classification_skips_a_repeated_name():
    from spacr.qt.widgets.data_filter_panel import _classify_columns_uncached

    frame = pd.DataFrame([[1, 2, "a"], [3, 4, "b"]], columns=["x", "x", "label"])
    kinds = _classify_columns_uncached(frame)

    assert kinds["x"] == "skip"
    assert kinds["label"] == "category"
