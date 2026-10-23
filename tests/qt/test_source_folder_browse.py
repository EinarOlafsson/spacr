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
