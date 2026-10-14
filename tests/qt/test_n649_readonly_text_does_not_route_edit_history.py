"""Edit history cannot bypass a text editor's read-only state."""
from types import SimpleNamespace

import pytest
from PySide6.QtGui import QTextCursor, QUndoStack
from PySide6.QtWidgets import QApplication, QLineEdit, QPlainTextEdit, QTextEdit

from spacr.qt.shortcuts import _record_edit
from tests.qt.test_edit_menu_undo_redo import _screen, _window


@pytest.mark.parametrize('editor_type', [QLineEdit, QTextEdit, QPlainTextEdit])
@pytest.mark.parametrize('readonly', [False, True])
def test_edit_actions_follow_current_text_permissions(qtbot, editor_type, readonly):
    window = _window(qtbot)
    editor = editor_type()
    screen = _screen(window, editor)
    model = SimpleNamespace(undo_stack=QUndoStack(screen))
    screen._settings_model = model
    setting = {'value': 2}
    _record_edit(model.undo_stack, 'Change setting',
                 lambda value: setting.__setitem__('value', value), 1, 2)
    if isinstance(editor, QLineEdit):
        editor.setText('abc')
        editor.insert('d')
        text = editor.text
    else:
        editor.setPlainText('abc')
        editor.moveCursor(QTextCursor.End)
        editor.insertPlainText('d')
        text = editor.toPlainText
    assert text() == 'abcd'
    window.show()
    window.activateWindow()
    editor.setFocus()
    qtbot.waitUntil(lambda: QApplication.focusWidget() is editor)
    window._edit_menu_about_to_show()
    undo, redo = window._act_edit_undo, window._act_edit_redo
    assert undo.isEnabled()
    editor.setReadOnly(readonly)
    undo.trigger()
    if readonly:
        assert text() == 'abcd'
        window._refresh_edit_actions()
        assert not undo.isEnabled() and not redo.isEnabled()
    else:
        assert text() == 'abc'
        assert redo.isEnabled()
        redo.trigger()
        assert text() == 'abcd'
    assert setting['value'] == 2 and model.undo_stack.canUndo()
