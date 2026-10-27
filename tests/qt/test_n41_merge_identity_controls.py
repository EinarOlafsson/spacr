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


def test_reported_filename_duplicates_use_choice_without_worker_error(
        qtbot, private_ai_profile, tmp_path, monkeypatch):
    from spacr.qt.widgets import merge_tables_dialog as mtd

    path, cell = _database(tmp_path)
    cell = cell.assign(file_name=['field1.tif', 'field2.tif'])
    with sqlite3.connect(path) as db:
        cell.to_sql('cell', db, index=False, if_exists='replace')
        cell.to_sql('cell', db, index=False, if_exists='append')
    before = path.read_bytes()
    decisions = iter((False, True))
    prompts = []
    def choose(parent, details):
        prompts.append(details)
        return next(decisions)
    monkeypatch.setattr(mtd, '_confirm_duplicate_observations', choose)
    dialog = mtd.MergeTablesDialog(str(path), selected=('cell',), threaded=True)
    qtbot.addWidget(dialog)
    failures = []
    dialog._jobs.job_failed.connect(failures.append)
    dialog.validate_preview()
    qtbot.waitUntil(lambda: len(prompts) == 1)
    assert dialog.result_frame is None and not dialog.create.isEnabled()
    assert 'file_name' in prompts[0] and 'time' not in prompts[0].lower()
    dialog.validate_preview()
    qtbot.waitUntil(dialog.create.isEnabled)
    assert len(prompts) == 2 and not failures
    assert len(dialog.result_frame) == 2
    assert dialog.result_frame.cell_area.tolist() == (cell.area * 2).tolist()
    assert path.read_bytes() == before
    dialog.close()


def test_object_counts_validation_stays_editable_without_worker_failure(
        qtbot, private_ai_profile, tmp_path):
    from spacr.qt.widgets.merge_tables_dialog import MergeTablesDialog
    import pandas as pd

    path = tmp_path / 'measurements.db'
    with sqlite3.connect(path) as db:
        pd.DataFrame({'plateID': ['p'], 'rowID': ['A'], 'columnID': [1],
                      'fieldID': [1], 'cell_count': [3]}).to_sql('object_counts', db, index=False)
    before = path.read_bytes()
    dialog = MergeTablesDialog(str(path), selected=('object_counts',), threaded=True)
    qtbot.addWidget(dialog)
    failures = []
    dialog._jobs.job_failed.connect(failures.append)
    dialog.validate_preview()
    qtbot.waitUntil(lambda: 'Customize merging' in dialog.preview_text.toPlainText())
    assert 'another observation level' in dialog.preview_text.toPlainText()
    assert 'missing keys object_label' not in dialog.preview_text.toPlainText()
    assert not failures and dialog.result_frame is None and not dialog.create.isEnabled()
    definition = dialog.configuration()
    definition.update(mode='custom', acknowledged=True,
                      base_keys=['plateID', 'rowID', 'columnID', 'fieldID'])
    dialog._custom = definition
    dialog.validate_preview()
    qtbot.waitUntil(dialog.create.isEnabled)
    assert dialog.result_frame.object_counts_cell_count.tolist() == [3]
    assert not failures and path.read_bytes() == before
    dialog.close()


def test_unexpected_preview_errors_still_propagate(monkeypatch):
    import pytest
    from spacr.qt.widgets import merge_tables_dialog as mtd

    def broken(*args):
        raise RuntimeError('unexpected implementation failure')
    monkeypatch.setattr(mtd, 'execute', broken)
    with pytest.raises(RuntimeError, match='unexpected implementation failure'):
        mtd._validated_preview('unused', {})
