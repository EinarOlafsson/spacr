"""Original filename enrichment through the real merge popup and saved recipe."""
import copy
import hashlib
import json
import sqlite3

import pandas as pd
import pytest
from PySide6.QtWidgets import QFileDialog

from spacr.derived_tables import default_definition, execute, load_definitions, save_definition
from spacr.qt.widgets.merge_tables_dialog import MergeTablesDialog


@pytest.fixture
def sources(tmp_path):
    database = tmp_path / "measurements.db"
    table = pd.DataFrame({"filename": ["renamed_a.tif", "missing.tif", "renamed_a.tif"],
                          "measurement": [9.0, 2.0, 6.0]})
    identity = dict(plateID=["p1"] * 3, rowID=["r1"] * 3,
                    columnID=["c1"] * 3, fieldID=["f1"] * 3)
    with sqlite3.connect(database) as db:
        table.to_sql("readings", db, index=False)
        pd.DataFrame({**identity, "object_label": [1, 2, 3],
                      "area": [100., 200., 300.]}).to_sql("cell", db, index=False)
        pd.DataFrame({**identity, "object_label": [7, 8, 9], "cell_id": [1, 1, 2],
                      "area": [2., 3., 7.]}).to_sql("pathogen", db, index=False)
    mapping = tmp_path / "rename_log.csv"
    pd.DataFrame({"Renamed TIFF": ["renamed_a.tif", "p1_A01_T0001F001L01A01Z01C01.tif"],
                  "Original File": ["/raw/drug_A.tif", "/raw/drug_B.tif"]}).to_csv(mapping, index=False)
    return database, mapping, table


def _choose_mapping(dialog, monkeypatch, mapping):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(mapping), ""))
    dialog.original_filenames.click()
    assert not dialog.create.isEnabled()
    assert str(mapping) in dialog.filename_note.text()


def test_single_table_popup_retains_rows_and_binds_replay_to_reviewed_mapping(
    qtbot, monkeypatch, sources,
):
    database, mapping, source = sources
    source_bytes = database.read_bytes()
    dialog = MergeTablesDialog(str(database), selected=["readings"], threaded=False)
    qtbot.addWidget(dialog)
    assert not dialog.create.isEnabled()
    _choose_mapping(dialog, monkeypatch, mapping)
    assert dialog.configuration()["mode"] == "metadata"
    assert not dialog.customize.isEnabled()
    dialog.validate_preview()
    assert dialog.create.isEnabled(), dialog.preview_text.toPlainText()
    frame = dialog.result_frame
    pd.testing.assert_frame_equal(frame[source.columns], source)
    assert frame.original_filename.fillna("").tolist() == ["drug_A.tif", "", "drug_A.tif"]
    assert "2 matched rows; 1 unmatched rows" in dialog.preview_text.toPlainText()
    assert not frame.attrs["image_provenance"]
    recipe = copy.deepcopy(dialog.definition)
    assert recipe["original_filenames"]["sha256"] == hashlib.sha256(mapping.read_bytes()).hexdigest()
    save_definition(database, recipe)
    saved = load_definitions(database)[recipe["name"]]
    replay, report = execute(database, saved)
    pd.testing.assert_frame_equal(replay, frame)
    assert report["output_rows"] == 3
    assert database.read_bytes() == source_bytes

    reopened = MergeTablesDialog(str(database), initial_definition=saved, threaded=False)
    qtbot.addWidget(reopened)
    assert reopened.configuration()["original_filenames"] == saved["original_filenames"]
    mapping.write_text(mapping.read_text().replace("drug_A", "other_condition"))
    reopened.validate_preview()
    assert not reopened.create.isEnabled()
    assert "changed" in reopened.preview_text.toPlainText().lower()
    with pytest.raises(ValueError, match="changed"):
        execute(database, saved)
    _choose_mapping(reopened, monkeypatch, mapping)
    reopened.validate_preview()
    assert reopened.create.isEnabled(), reopened.preview_text.toPlainText()
    assert reopened.result_frame.original_filename.iloc[0] == "other_condition.tif"
    reopened.name.setText("Another reviewed result")
    assert not reopened.create.isEnabled()
    assert reopened.result_frame is None


@pytest.mark.parametrize("custom", [False, True])
def test_reopened_multi_table_merge_preserves_relationships_and_aggregation(
    qtbot, monkeypatch, sources, custom,
):
    database, mapping, _ = sources
    recipe = default_definition(database, ["cell", "pathogen"], name="Reviewed merge", base="cell")
    if custom:
        recipe["mode"] = "custom"
        recipe["acknowledged"] = True
        recipe["joins"][0]["overrides"] = {"area": "mean"}
    original = copy.deepcopy(recipe)
    expected, _ = execute(database, recipe)
    dialog = MergeTablesDialog(str(database), initial_definition=recipe, threaded=False)
    qtbot.addWidget(dialog)
    _choose_mapping(dialog, monkeypatch, mapping)
    configured = dialog.configuration()
    assert configured["mode"] == recipe["mode"]
    assert configured["base_keys"] == recipe["base_keys"]
    assert configured["joins"] == recipe["joins"]
    assert configured["acknowledged"] == recipe["acknowledged"]
    dialog.validate_preview()
    assert dialog.create.isEnabled(), dialog.preview_text.toPlainText()
    pd.testing.assert_frame_equal(dialog.result_frame[expected.columns], expected)
    assert dialog.result_frame.original_filename.tolist() == ["drug_B.tif"] * len(expected)
    assert recipe == original
    dialog.clear_filenames.click()
    assert "original_filenames" not in dialog.configuration()
    assert dialog.configuration()["joins"] == recipe["joins"]
    assert not dialog.create.isEnabled()


@pytest.mark.parametrize("custom", [False, True])
def test_existing_single_source_recipe_keeps_its_reviewed_mode(
    qtbot, monkeypatch, sources, custom,
):
    database, mapping, _ = sources
    recipe = default_definition(database, ["cell"], name="Reviewed cells", base="cell")
    if custom:
        recipe.update(mode="custom", acknowledged=True)
    expected, _ = execute(database, recipe)
    dialog = MergeTablesDialog(str(database), initial_definition=recipe, threaded=False)
    qtbot.addWidget(dialog)
    _choose_mapping(dialog, monkeypatch, mapping)
    assert dialog.configuration()["mode"] == recipe["mode"]
    assert dialog.configuration()["base_keys"] == recipe["base_keys"]
    dialog.validate_preview()
    assert dialog.create.isEnabled(), dialog.preview_text.toPlainText()
    pd.testing.assert_frame_equal(dialog.result_frame[expected.columns], expected)


def test_programmatic_graph_merge_saves_the_bound_map_digest(qtbot, sources, tmp_path):
    from spacr.condition_annotations import new_definition
    from spacr.derived_tables import sidecar_path
    from spacr.qt.screens.graph_builder import GraphBuilderScreen

    database, mapping, _ = sources
    screen = GraphBuilderScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.load_path(str(database), "readings")
    recipe = default_definition(database, ["readings"], base="readings", name="Names restored")
    recipe.update(mode="metadata", base_keys=[], original_filenames={
        "map_path": str(mapping), "output_column": "original_filename"})
    screen.use_derived_table(recipe)
    assert screen._table_picker.currentText() == "Names restored", screen._source.text()
    bound = load_definitions(database)["Names restored"]
    assert bound["original_filenames"]["sha256"] == hashlib.sha256(mapping.read_bytes()).hexdigest()
    assert "sha256" not in recipe["original_filenames"]
    assert "original_filename" in screen._frame
    assert not screen._to_annotate.isEnabled()
    chart = tmp_path / "restored_names.json"
    screen.save_chart(str(chart))
    screen.load_path(str(database), "readings")
    screen.load_chart(str(chart))
    assert screen._merge_definition["original_filenames"]["sha256"] == bound["original_filenames"]["sha256"]
    assert screen._frame.original_filename.iloc[0] == "drug_A.tif"

    conditions = new_definition(screen._annotation_base_frame, screen._condition_source)
    conditions["conditions"] = [dict(name="Drug", metadata_column="original_filename",
                                      include="drug_A", exclude="", manual_rows=[])]
    screen.apply_condition_definition(conditions)
    screen.save_chart(str(chart))
    invalid_chart = json.loads(chart.read_text())
    invalid_chart["condition_annotation"]["conditions"][0]["include"] = "["
    chart.write_text(json.dumps(invalid_chart))
    # A same-name recipe may have been reviewed since the chart was saved.
    # Rejected condition rules must not overwrite it as a load side effect.
    prior = copy.deepcopy(bound)
    prior["original_filenames"]["output_column"] = "prior_original_name"
    save_definition(database, prior)
    sidecar_before = sidecar_path(database).read_bytes()
    frame_before = screen._frame.copy(deep=True)
    screen.load_chart(str(chart))
    assert "invalid include" in screen._source.text().lower()
    assert sidecar_path(database).read_bytes() == sidecar_before
    pd.testing.assert_frame_equal(screen._frame, frame_before)
