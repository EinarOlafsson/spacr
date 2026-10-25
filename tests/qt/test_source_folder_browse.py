"""Source-folder browsing keeps each control's existing value and commit path."""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QToolButton, QWidget

from spacr.qt.screens.settings_model import (
    SettingsWidgets, _ListEditor, _ScalarEdit, _TrainingFolderEdit,
    _add_folder_browse_action,
)


def _folder_action(edit):
    return next(action for action in edit.actions()
                if action.objectName() == 'SourceFolderBrowse')


@pytest.mark.parametrize('key', ['src', 'microscope_simulated_folder', 'n2v_model'])
def test_folder_selection_uses_the_existing_setting_commit(qtbot, monkeypatch, key):
    parent = QWidget()
    qtbot.addWidget(parent)
    model = SettingsWidgets('mask', parent=parent)
    edit = model._widget_for('entry', None, '/old source', key)
    model._widgets[key] = edit
    committed = []
    edit.editingFinished.connect(lambda: committed.append(edit.text()))
    calls = []

    def choose(owner, title, initial):
        calls.append((owner, initial))
        return '/chosen source/with spaces'

    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', choose)
    _folder_action(edit).trigger()
    assert calls == [(edit, '/old source')]
    assert committed == ['/chosen source/with spaces']
    assert model.collect()[key] == '/chosen source/with spaces'


def test_cancel_keeps_typed_source_and_emits_no_commit(qtbot, monkeypatch):
    edit = _ScalarEdit()
    qtbot.addWidget(edit)
    edit.setText('/typed source')
    action = _add_folder_browse_action(edit)
    commits = []
    edit.editingFinished.connect(lambda: commits.append(edit.text()))
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args: '')
    action.trigger()
    assert edit.text() == '/typed source'
    assert not commits
    assert _add_folder_browse_action(edit) is action
    assert len(edit.actions()) == 1


def test_folder_action_is_keyboard_accessible(qtbot, monkeypatch):
    edit = _ScalarEdit()
    qtbot.addWidget(edit)
    action = _add_folder_browse_action(edit)
    button = next(button for button in edit.findChildren(QToolButton)
                  if button.defaultAction() is action)
    assert button.focusPolicy() == Qt.StrongFocus
    assert button.accessibleName() == action.toolTip()
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args: '/keyboard')
    edit.show()
    button.setFocus()
    qtbot.keyClick(button, Qt.Key_Space)
    assert edit.text() == '/keyboard'


def test_existing_multiple_source_text_keeps_order(qtbot, monkeypatch):
    edit = _ScalarEdit()
    qtbot.addWidget(edit)
    edit.setText("['/plate one', '/plate two']")
    action = _add_folder_browse_action(edit)
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args: '/plate three')
    action.trigger()
    assert edit.text() == "['/plate one', '/plate two', '/plate three']"
    action.trigger()
    assert edit.text() == "['/plate one', '/plate two', '/plate three']"


def test_grouped_folder_browser_appends_only_to_its_own_group(qtbot, monkeypatch):
    edit = _ListEditor(key='src', default=[['/first'], ['/second']],
                       nested_capable=True, element_type=str)
    qtbot.addWidget(edit)
    commits = []
    edit._committed.connect(lambda: commits.append(edit.get_value()))
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args: '/new, with comma')
    _folder_action(edit._strips[1]._entry).trigger()
    assert edit.get_value() == [['/first'], ['/second', '/new, with comma']]
    assert commits[-1] == edit.get_value()
    edit.set_value([['/restored']])
    assert _folder_action(edit._strips[0]._entry)


def test_training_folder_keeps_one_existing_browser(qtbot, monkeypatch):
    edit = _TrainingFolderEdit('/training')
    qtbot.addWidget(edit)
    action = _folder_action(edit)
    assert _add_folder_browse_action(edit) is action
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args: '/new training')
    action.trigger()
    assert edit.get_value() == '/new training'
    assert len(edit.actions()) == 1


@pytest.mark.parametrize('app', ['classify_merged', 'measure', 'map_barcodes'])
def test_source_row_keeps_editable_paths_and_compact_actions(qtbot, monkeypatch, app):
    from PySide6.QtWidgets import QLineEdit
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets import qc_field_browser

    opened = []
    monkeypatch.setattr(qc_field_browser, '_open_overview_module',
                        lambda owner, settings: opened.append((owner, settings['src'])))
    screen = AppScreen(app)
    qtbot.addWidget(screen)
    model = screen._settings_model
    edit = model._widgets['src']
    assert isinstance(edit, QLineEdit) and not edit.isReadOnly()
    assert model.set_value_for_key('src', '/typed source')
    assert edit.text() == '/typed source'
    for scalar in ('123', 'None', ' folder with spaces '):
        edit.setText(scalar)
        assert model.collect()['src'] == scalar
    edit.setText('/pasted source')
    edit.editingFinished.emit()
    assert model.collect()['src'] == '/pasted source'
    actions = edit.actions()
    assert len([a for a in actions if a.objectName() == 'SourceFolderBrowse']) == 1
    assert len([a for a in actions if a.objectName() == 'CloudSourceBrowse']) == 1
    overview = [a for a in actions if a.objectName() == 'GenerateMasksFieldOverview']
    assert len(overview) == (0 if app == 'map_barcodes' else 1)
    if overview:
        overview[0].trigger()
        assert opened == [(screen, '/pasted source')]
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args: '')
    _folder_action(edit).trigger()
    assert model.collect()['src'] == '/pasted source'
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args: '/chosen source')
    _folder_action(edit).trigger()
    assert model.collect()['src'] == '/chosen source'
    assert model.set_value_for_key('src', ['/plate one', '/plate two'])
    assert model.collect()['src'] == ['/plate one', '/plate two']
    assert screen._settings_src_path() == '/plate one'
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args: '/plate three')
    _folder_action(edit).trigger()
    assert model.collect()['src'] == ['/plate one', '/plate two', '/plate three']
    if overview:
        overview[0].trigger()
        assert opened[-1] == (screen, ['/plate one', '/plate two', '/plate three'])
    grouped = [['/control'], ['/treated']]
    screen.apply_settings_dict({'src': grouped})
    assert model.collect()['src'] == grouped
    assert screen._settings_src_path() == '/control'
    assert model._current_source_folder() == '/control'
    _folder_action(edit).trigger()
    assert model.collect()['src'] == [['/control'], ['/treated', '/plate three']]


@pytest.mark.parametrize('app', ['classify_merged', 'measure', 'map_barcodes'])
def test_source_cloud_action_keeps_existing_dialog_commit(qtbot, monkeypatch, app):
    from PySide6.QtWidgets import QDialog
    from spacr.qt.screens import settings_model
    from spacr.qt.screens.app_screen import AppScreen

    class CloudChoice:
        def __init__(self, *args):
            self.args = args
        def exec(self):
            return QDialog.Accepted
        def location(self):
            return 's3://chosen-bucket/images'
        def options(self):
            return {'cloud_anonymous': True, 'cloud_endpoint': 'https://example.test'}

    monkeypatch.setattr(settings_model, '_CloudBrowserDialog', CloudChoice)
    screen = AppScreen(app)
    qtbot.addWidget(screen)
    model = screen._settings_model
    edit = model._widgets['src']
    cloud = next(a for a in edit.actions() if a.objectName() == 'CloudSourceBrowse')
    button = next(b for b in edit.findChildren(QToolButton)
                  if b.defaultAction() is cloud)
    assert button.focusPolicy() == Qt.StrongFocus
    assert button.accessibleName() == cloud.toolTip()
    cloud.trigger()
    assert model.collect()['src'] == 's3://chosen-bucket/images'
    for key, value in CloudChoice().options().items():
        if key in model._widgets:
            assert model.collect()[key] == value
