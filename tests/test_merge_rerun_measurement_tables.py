"""Issue #183: tables archived by a Measure re-run merge like the current ones.

Measuring a project again renames the previous population to ``cell_1``,
``cytoplasm_1``, ``pathogen_1`` and so on. Default merging matched table
names exactly, so ``cell_1`` was refused as "not one row per cell".
"""
import sqlite3

import pandas as pd
import pytest

from spacr.derived_tables import default_definition, execute
from spacr.merge_tables import MergeError, merge_tables
from spacr.object_roles import (
    _table_role,
    _table_role_and_run,
    anchor_column,
    is_one_row_per_cell,
    join_how,
)


def _frames():
    image = {"plateID": "p1", "rowID": "r1", "columnID": "c1"}
    cell = pd.DataFrame([{**image, "fieldID": "f1", "object_label": 1, "area": 100},
                         {**image, "fieldID": "f2", "object_label": 1, "area": 200},
                         {**image, "fieldID": "f2", "object_label": 2, "area": 300}])
    cytoplasm = cell.assign(area=[60, 150, 210], mean_intensity=[1.0, 2.0, 3.0])
    pathogen = pd.DataFrame([{**image, "fieldID": field, "object_label": i + 7,
                              "cell_id": parent, "area": area}
                             for i, (field, parent, area) in enumerate(
                                 [("f1", 1, 2), ("f1", 1, 5), ("f2", 2, 11)])])
    return {"cell": cell, "cytoplasm": cytoplasm, "pathogen": pathogen}


def _database(tmp_path, *, current=True, rerun=True):
    path = str(tmp_path / "measurements.db")
    frames = _frames()
    with sqlite3.connect(path) as db:
        for table, frame in frames.items():
            if current:
                frame.to_sql(table, db, index=False)
            if rerun:
                frame.to_sql(f"{table}_1", db, index=False)
    return path


def _unsuffixed(frame):
    return frame.rename(columns={c: c.replace("_1_", "_", 1) for c in frame.columns})


@pytest.mark.parametrize("table", ["cell_1", "cytoplasm_1", "Cell_2", "cytoplasm_12"])
def test_rerun_cell_tables_are_one_row_per_cell(table):
    assert is_one_row_per_cell(table)
    assert anchor_column(table) == "object_label"


@pytest.mark.parametrize("table", ["pathogen_1", "nucleus_1", "organelle_3"])
def test_rerun_child_tables_roll_up(table):
    assert not is_one_row_per_cell(table)
    assert anchor_column(table) == "cell_id"


def test_rerun_suffix_is_only_stripped_from_known_roles():
    assert _table_role_and_run("pathogen_1") == ("pathogen", 1)
    assert _table_role_and_run("cell") == ("cell", 0)
    assert _table_role("observations_2") == "observations_2"
    assert join_how("pathogen_1", keep_uninfected=False) == "inner"
    with pytest.raises(ValueError):
        anchor_column("observations_2")


def test_default_merge_of_a_rerun_set_matches_the_current_set(tmp_path):
    path = _database(tmp_path)
    current, current_report = execute(path, default_definition(
        path, ["cell", "cytoplasm", "pathogen"]))
    definition = default_definition(path, ["pathogen_1", "cytoplasm_1", "cell_1"])
    assert definition["base"] == "cell_1"
    rerun, rerun_report = execute(path, definition)
    pd.testing.assert_frame_equal(_unsuffixed(rerun), current, check_like=True)
    assert "cytoplasm_1_mean_intensity" in rerun
    assert rerun.pathogen_1_area.iloc[[0, 2]].tolist() == [7, 11]
    assert rerun_report["image_provenance"] == current_report["image_provenance"]
    assert [j["relationship"] for j in definition["joins"]] == ["one-to-many", "one-to-one"]


def test_cytoplasm_rerun_table_is_a_default_output_level(tmp_path):
    path = _database(tmp_path, current=False)
    frame, report = execute(path, default_definition(
        path, ["cytoplasm_1", "pathogen_1"], base="cytoplasm_1"))
    assert len(frame) == 3
    assert report["base"] == "cytoplasm_1"


def test_a_child_table_is_still_refused_as_default_output(tmp_path):
    path = _database(tmp_path, current=False)
    with pytest.raises(MergeError, match="one row per cell"):
        execute(path, default_definition(path, ["pathogen_1", "cell_1"], base="pathogen_1"))


def test_tables_from_different_runs_are_refused_with_a_clear_message(tmp_path):
    path = _database(tmp_path)
    with pytest.raises(MergeError, match="different Measure runs") as caught:
        execute(path, default_definition(path, ["cell", "pathogen_1"], base="cell"))
    assert "cell_1, cytoplasm_1 and pathogen_1" in str(caught.value)
    with pytest.raises(MergeError, match="different Measure runs"):
        merge_tables(path, ["cytoplasm_1"])
