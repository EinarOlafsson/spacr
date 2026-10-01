"""Both real screens consume, persist and validate the same derived tables."""
import sqlite3

import pandas as pd
import pytest
from PySide6.QtWidgets import QDialog

from spacr.qt.widgets.merge_tables_dialog import CustomMergeDialog, MergeTablesDialog


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "measurements.db")
    ident = dict(plateID=["p1"] * 3, rowID=["r1"] * 3, columnID=["c1"] * 3, fieldID=["f1"] * 3)
    with sqlite3.connect(path) as connection:
        pd.DataFrame({**ident, "object_label": [1, 2, 3], "area": [100., 200., 300.]}).to_sql("cell", connection, index=False)
        pd.DataFrame({**ident, "object_label": [7, 8, 9], "cell_id": [1, 1, 2],
                      "area": [2., 3., 7.], "mean_intensity": [5., 7., 11.]}).to_sql("pathogen", connection, index=False)
    return path


def test_dialog_preview_invalidation_custom_cancel_acknowledgment_and_reset(qtbot, db):
    dialog = MergeTablesDialog(db, selected=["cell", "pathogen"], threaded=False)
    qtbot.addWidget(dialog)
    dialog.validate_preview()
    assert dialog.create.isEnabled(), dialog.preview_text.toPlainText()
    assert "3 output rows" in dialog.preview_text.toPlainText()
    dialog.name.setText("Changed")
    assert not dialog.create.isEnabled()
    original = dialog.configuration()
    editor = CustomMergeDialog(db, original, dialog)
    qtbot.addWidget(editor)
    assert not editor.buttons.button(editor.buttons.StandardButton.Ok).isEnabled()
    editor.base_keys.setText("wrong")
    editor.reject()
    assert dialog.configuration() == original
    editor = CustomMergeDialog(db, original, dialog)
    qtbot.addWidget(editor)
    editor.acknowledge.setChecked(True)
    editor.accept()
    assert editor.result() == QDialog.Accepted
    assert editor.definition["acknowledged"]
    dialog._custom = editor.definition
    dialog._show_state()
    assert "Custom rules active" in dialog.state.text()
    dialog._reset()
    assert "spaCR defaults active" in dialog.state.text()
    assert not dialog.configuration()["acknowledged"]


def test_external_mapping_controls_produce_a_valid_custom_result(qtbot, tmp_path, monkeypatch):
    path = str(tmp_path / "external.db")
    with sqlite3.connect(path) as connection:
        pd.DataFrame({"image": ["a", "b"], "sample": [1, 1], "reading": [10., 20.]}).to_sql("samples", connection, index=False)
        pd.DataFrame({"scene": ["a", "a", "b"], "parent": [1, 1, 1], "value": [1., 3., 7.]}).to_sql("events", connection, index=False)
    dialog = MergeTablesDialog(path, selected=["samples", "events"], threaded=False)
    qtbot.addWidget(dialog)
    dialog.base.setCurrentText("samples")

    def edit_and_accept(editor):
        editor.base_keys.setText("image, sample")
        left, right, relation, how, identifiers = editor._rows[0]
        left.setText("image, sample")
        right.setText("scene, parent")
        relation.setCurrentText("one-to-many")
        how.setCurrentText("left")
        editor.acknowledge.setChecked(True)
        editor.accept()
        return editor.result()

    monkeypatch.setattr(CustomMergeDialog, "exec", edit_and_accept)
    dialog.customize.click()
    assert "Custom rules active" in dialog.state.text()
    dialog._rule_changed("events", "value", "sum")
    dialog.validate_preview()
    assert dialog.create.isEnabled(), dialog.preview_text.toPlainText()
    assert dialog.result_frame.events_value.tolist() == [4., 7.]
    assert dialog.definition["joins"][0]["right_keys"] == ["scene", "parent"]
    assert "navigation is unavailable" in dialog.preview_text.toPlainText()
