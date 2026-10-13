"""Portable schema buttons bind to the current table without applying implicitly."""
import copy
import json
import threading

import pandas as pd
from PySide6.QtCore import Qt

from spacr import condition_annotations as backend
from spacr.qt.widgets import condition_annotation_dialog as ui


def dialog_for(qtbot, *, frame=None, threaded=False):
    if frame is None:
        frame = pd.DataFrame({'filename': ['WT_rep1.tif', 'KO_rep2.tif', 'unknown.tif']})
    dialog = ui.ConditionAnnotationDialog(frame, backend.source_context(), threaded=threaded)
    qtbot.addWidget(dialog)
    dialog.show()
    return dialog


def click(qtbot, button):
    qtbot.mouseClick(button, Qt.LeftButton)


def filename_recipe(qtbot, dialog):
    dialog.column_kind.setCurrentIndex(dialog.column_kind.findData('extract'))
    dialog.extract_pattern.setText(r'(?P<genotype>[^_]+)_(?P<replicate>rep\d+)')
    click(qtbot, dialog.named_groups_button)
    dialog.column_selector.setCurrentIndex(1)
    click(qtbot, dialog.add_column)
    dialog.output_column.setText('condition')
    dialog.column_kind.setCurrentIndex(dialog.column_kind.findData('combine'))
    for i, name in enumerate(('genotype', 'replicate')):
        if i:
            dialog.template_text.setText('_')
            click(qtbot, dialog.template_add_text)
        dialog.template_palette.setCurrentItem(dialog.template_palette.findItems(name, Qt.MatchExactly)[0])
        click(qtbot, dialog.template_add_column)
    click(qtbot, dialog.preview_button)


def files(monkeypatch, path):
    monkeypatch.setattr(ui.QFileDialog, 'getSaveFileName', lambda *_args, **_kwargs: (str(path), 'JSON'))
    monkeypatch.setattr(ui.QFileDialog, 'getOpenFileName', lambda *_args, **_kwargs: (str(path), 'JSON'))


def test_save_then_load_other_table_retains_all_recipe_columns_without_apply(qtbot, tmp_path, monkeypatch):
    path = tmp_path / 'portable.json'
    files(monkeypatch, path)
    original = dialog_for(qtbot)
    filename_recipe(qtbot, original)
    assert original.save_schema_button.isEnabled(), original.status.text()
    click(qtbot, original.save_schema_button)
    payload = json.loads(path.read_text())
    assert payload['format'] == 'spacr.annotation-schema'
    assert 'source' not in payload and 'source_fingerprint' not in payload
    assert [c['column'] for c in payload['columns']] == ['genotype', 'replicate', 'condition']
    target = dialog_for(qtbot, frame=pd.DataFrame({'filename': ['KO_rep7.tif', 'WT_rep9.tif'], 'extra': [1, 2]}))
    source = target.frame.copy(deep=True)
    click(qtbot, target.load_schema_button)
    assert target.result() == 0
    assert target.apply_button.isEnabled(), target.status.text()
    assert target.result_frame.condition.tolist() == ['KO_rep7', 'WT_rep9']
    assert list(target.result_frame.columns[-3:]) == ['genotype', 'replicate', 'condition']
    pd.testing.assert_frame_equal(target.frame, source)
    click(qtbot, target.apply_button)
    assert target.result() == 1


def test_bad_schema_keeps_valid_draft_preview_and_apply(qtbot, tmp_path, monkeypatch):
    path = tmp_path / 'bad.json'
    files(monkeypatch, path)
    dialog = dialog_for(qtbot)
    filename_recipe(qtbot, dialog)
    previous = copy.deepcopy(dialog.configuration())
    result = dialog.result_frame
    path.write_text('{not json')
    click(qtbot, dialog.load_schema_button)
    assert dialog.configuration() == previous
    assert dialog.result_frame is result
    assert dialog.apply_button.isEnabled()
    assert dialog.save_schema_button.isEnabled()
    assert 'not loaded' in dialog.status.text()
    click(qtbot, dialog.save_schema_button)
    payload = json.loads(path.read_text())
    payload['columns'][0]['metadata_column'] = 'missing_column'
    path.write_text(json.dumps(payload))
    click(qtbot, dialog.load_schema_button)
    assert dialog.configuration() == previous
    assert dialog.result_frame is result
    assert dialog.apply_button.isEnabled()
    payload['columns'] = [{'column': 'invalid', 'kind': 'rules', 'conditions': [{
        'name': 'bad', 'metadata_column': ['filename'], 'include': 'WT'}]}]
    path.write_text(json.dumps(payload))
    click(qtbot, dialog.load_schema_button)
    assert dialog.configuration() == previous
    assert dialog.result_frame is result
    assert dialog.apply_button.isEnabled()


def test_manual_row_tokens_omitted_with_notice_but_regex_reused(qtbot, tmp_path, monkeypatch):
    path = tmp_path / 'manual.json'
    files(monkeypatch, path)
    dialog = dialog_for(qtbot)
    box = dialog.boxes[0]
    box.match_mode.setCurrentIndex(0)
    box.name.setText('selected')
    box.include.setText('^WT')
    box.manual_rows = [dialog.source_model.tokens[1]]
    box._show_manual_rows()
    click(qtbot, dialog.preview_button)
    assert dialog.result_frame.condition.notna().sum() == 2
    click(qtbot, dialog.save_schema_button)
    assert '1 manual row assignments were omitted' in dialog.status.text()
    assert dialog.source_model.tokens[1] not in path.read_text()
    target = dialog_for(qtbot, frame=pd.DataFrame({'filename': ['KO_rep2.tif', 'WT_rep1.tif']}))
    click(qtbot, target.load_schema_button)
    assert target.result_frame.condition.fillna('').tolist() == ['', 'selected']
    assert target.boxes[0].manual_rows == []


def test_stale_threaded_schema_load_cannot_replace_newer_draft(qtbot, tmp_path, monkeypatch):
    path = tmp_path / 'schema.json'
    files(monkeypatch, path)
    producer = dialog_for(qtbot)
    filename_recipe(qtbot, producer)
    click(qtbot, producer.save_schema_button)
    entered, release = threading.Event(), threading.Event()
    real_load = backend._load_schema

    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return real_load(*args)

    monkeypatch.setattr(backend, '_load_schema', blocked)
    dialog = dialog_for(qtbot, threaded=True)
    click(qtbot, dialog.load_schema_button)
    qtbot.waitUntil(entered.is_set, timeout=5000)
    try:
        dialog.output_column.setText('newer')
        dialog.boxes[0].match_mode.setCurrentIndex(0)
        dialog.boxes[0].include.setText('^WT')
        click(qtbot, dialog.preview_button)
        qtbot.waitUntil(lambda: dialog.apply_button.isEnabled(), timeout=5000)
        assert dialog.result_frame.newer.notna().sum() == 1
    finally:
        release.set()
    qtbot.waitUntil(lambda: dialog._jobs.active_jobs() == 0, timeout=5000)
    assert dialog.output_column.text() == 'newer'
    assert list(dialog.result_frame.columns) == ['filename', 'newer']
    dialog.reject()


def test_schema_criteria_retains_exact_and_regex_exclusions(qtbot, tmp_path, monkeypatch):
    path = tmp_path / 'compound.json'
    files(monkeypatch, path)
    for extra in ({'match_mode': 'values', 'exclude_values': ['KO_rep2.tif']},
                  {'exclude': '^KO'}):
        rule = {'name': 'kept', 'metadata_column': 'filename', 'include': '',
                'exclude': '', 'manual_rows': [], 'match': 'all',
                'criteria': [{'metadata_column': 'filename', 'operator': 'regex', 'value': '.*'}], **extra}
        path.write_text(json.dumps({'format': 'spacr.annotation-schema', 'version': 1,
                                   'recipe_version': 3, 'manual_rows': 'excluded',
                                   'columns': [{'column': 'assigned', 'kind': 'rules', 'conditions': [rule]}]}))
        dialog = dialog_for(qtbot)
        click(qtbot, dialog.load_schema_button)
        assert dialog.apply_button.isEnabled(), dialog.status.text()
        assert dialog.boxes[0].match_mode.currentData() == 'criteria'
        assert dialog.result_frame.assigned.fillna('').tolist() == ['kept', '', 'kept']
        click(qtbot, dialog.save_schema_button)
        saved_rule = json.loads(path.read_text())['columns'][0]['conditions'][0]
        for key, value in extra.items():
            assert saved_rule[key] == value
        dialog.reject()


def test_threaded_save_keeps_chosen_snapshot_while_editor_changes(qtbot, tmp_path, monkeypatch):
    path = tmp_path / 'snapshot.json'
    files(monkeypatch, path)
    dialog = dialog_for(qtbot, threaded=True)
    filename_recipe(qtbot, dialog)
    qtbot.waitUntil(lambda: dialog.save_schema_button.isEnabled(), timeout=5000)
    real_save = backend._save_schema
    entered, release = threading.Event(), threading.Event()

    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return real_save(*args)

    monkeypatch.setattr(backend, '_save_schema', blocked)
    click(qtbot, dialog.save_schema_button)
    qtbot.waitUntil(entered.is_set, timeout=5000)
    try:
        assert not path.exists()
        dialog.output_column.setText('renamed')
        click(qtbot, dialog.preview_button)
        qtbot.waitUntil(lambda: dialog.apply_button.isEnabled(), timeout=5000)
        assert not dialog.save_schema_button.isEnabled()
        assert 'renamed' in dialog.result_frame
    finally:
        release.set()
    qtbot.waitUntil(lambda: dialog._schema_saves.active_jobs() == 0, timeout=5000)
    qtbot.waitUntil(lambda: dialog.save_schema_button.isEnabled(), timeout=5000)
    assert json.loads(path.read_text())['columns'][-1]['column'] == 'condition'
    assert dialog.configuration()['columns'][-1]['column'] == 'renamed'
    dialog.reject()


def test_explicit_empty_rules_survive_schema_save_load_and_column_switch(qtbot, tmp_path, monkeypatch):
    path = tmp_path / 'empty.json'
    files(monkeypatch, path)
    dialog = dialog_for(qtbot)
    dialog.remove_box(dialog.boxes[0])
    click(qtbot, dialog.preview_button)
    assert dialog.apply_button.isEnabled()
    assert dialog.result_frame.condition.isna().all()
    click(qtbot, dialog.save_schema_button)
    target = dialog_for(qtbot)
    click(qtbot, target.load_schema_button)
    assert target.boxes == []
    assert target.configuration()['conditions'] == []
    assert target.apply_button.isEnabled(), target.status.text()
    assert target.result_frame.condition.isna().all()
    click(qtbot, target.add_column)
    assert len(target.boxes) == 1  # New columns still offer a fresh rule editor.
    target.column_selector.setCurrentIndex(0)
    assert target.boxes == []
    reopened = ui.ConditionAnnotationDialog(target.frame, backend.source_context(),
                                            definition=target.configuration(), threaded=False)
    qtbot.addWidget(reopened)
    assert reopened.boxes == []
