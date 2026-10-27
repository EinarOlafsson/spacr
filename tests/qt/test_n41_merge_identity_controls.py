"""The real merge controls distinguish omitted FOV keys from repeated rows."""
import sqlite3

from spacr.derived_tables import default_definition
from tests.test_n41_observation_key_diagnostics import _database
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


def test_real_non_timelapse_merge_preview_custom_correction_and_gate_handoff(
        qtbot, private_ai_profile, tmp_path):
    from spacr.qt.widgets.merge_tables_dialog import CustomMergeDialog, MergeTablesDialog
    from spacr.qt.screens.gate_editor import GateEditorScreen
    path, _ = _database(tmp_path)
    dialog = MergeTablesDialog(str(path), selected=('cell', 'nucleus'), threaded=False)
    qtbot.addWidget(dialog)
    dialog.validate_preview()
    assert dialog.create.isEnabled()
    assert dialog.result_frame.nucleus_area.tolist() == [5., 11.]
    editor = CustomMergeDialog(str(path), dialog.configuration())
    qtbot.addWidget(editor)
    editor.acknowledge.setChecked(True)
    editor.base_keys.setText('object_label')
    editor.accept()
    dialog._custom = editor.definition
    dialog.validate_preview()
    assert not dialog.create.isEnabled()
    assert 'Add these existing identity columns' in dialog.preview_text.toPlainText()
    assert 'fieldID' in dialog.preview_text.toPlainText()
    assert 'time' not in dialog.preview_text.toPlainText().lower()
    dialog._reset()
    dialog.validate_preview()
    assert dialog.create.isEnabled()
    screen = GateEditorScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.load_path(str(path), 'cell')
    screen.use_derived_table(dialog.definition)
    assert len(screen._frame) == 2
    assert screen._frame.nucleus_area.tolist() == [5., 11.]
    assert not any('time' in column.lower() for column in screen._frame)
    screen.close()
    editor.close()
    dialog.close()


def test_real_merge_preview_refuses_true_repeated_full_identities(
        qtbot, private_ai_profile, tmp_path, monkeypatch):
    from spacr.qt.widgets import merge_tables_dialog
    from spacr.qt.widgets.merge_tables_dialog import MergeTablesDialog
    prompted = []
    monkeypatch.setattr(merge_tables_dialog, "_confirm_duplicate_observations",
                        lambda parent, details: prompted.append(details) or False)
    path, cell = _database(tmp_path)
    with sqlite3.connect(path) as db:
        cell.to_sql('cell', db, index=False, if_exists='append')
    dialog = MergeTablesDialog(str(path), selected=('cell', 'nucleus'), threaded=False)
    qtbot.addWidget(dialog)
    dialog.validate_preview()
    assert not dialog.create.isEnabled()
    message = dialog.preview_text.toPlainText()
    assert prompted == [message]
    assert 'full available image/object identity' in message
    assert 'measurement-run history' in message and 'time' not in message.lower()
    assert dialog.result_frame is None
    dialog.close()
