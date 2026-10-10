"""Edit menu boundaries: rich text editors, vanished focus and no editor."""

import pytest
from PySide6.QtWidgets import QPlainTextEdit, QTextEdit, QVBoxLayout, QWidget

from spacr.qt.app import MainWindow


def _window(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def _screen(window, child):
    screen = QWidget()
    QVBoxLayout(screen).addWidget(child)
    window._stack.addWidget(screen)
    window._stack.setCurrentWidget(screen)
    return screen


@pytest.mark.parametrize("editor_type", [QTextEdit, QPlainTextEdit])
def test_a_focused_document_editor_owns_the_edit_menu(qtbot, editor_type):
    window = _window(qtbot)
    editor = editor_type()
    _screen(window, editor)
    editor.setPlainText("one")
    editor.selectAll()
    editor.insertPlainText("two")
    window._edit_focus = editor
    window._refresh_edit_actions()
    assert window._act_edit_undo.isEnabled()
    assert not window._act_edit_redo.isEnabled()
    window._invoke_edit_action(redo=False)
    assert editor.toPlainText() == "one"
    assert window._act_edit_redo.isEnabled()
    window._invoke_edit_action(redo=True)
    assert editor.toPlainText() == "two"


def test_a_deleted_focus_widget_is_forgotten_and_no_editor_does_nothing(qtbot):
    window = _window(qtbot)
    editor = QTextEdit()
    _screen(window, editor)

    class Gone:
        pass

    gone = Gone()
    window._edit_focus = gone
    original = window._stack.currentWidget().isAncestorOf

    def deleted(widget):
        if widget is gone:
            raise RuntimeError("Internal C++ object already deleted.")
        return original(widget)

    window._stack.currentWidget().isAncestorOf = deleted
    assert window._edit_target() is None
    assert window._edit_focus is None
    window._invoke_edit_action(redo=False)
    assert not window._act_edit_undo.isEnabled()
    assert not window._act_edit_redo.isEnabled()


def test_without_a_current_screen_there_is_no_edit_target(qtbot, monkeypatch):
    window = _window(qtbot)
    monkeypatch.setattr(window._stack, "currentWidget", lambda: None)
    assert window._edit_target() is None


def test_a_history_that_vanishes_mid_refresh_disables_both_entries(qtbot, monkeypatch):
    window = _window(qtbot)

    def vanished():
        raise RuntimeError("Internal C++ object already deleted.")

    monkeypatch.setattr(window, "_edit_target",
                        lambda: (None, None, vanished, vanished))
    window._act_edit_undo.setEnabled(True)
    window._act_edit_redo.setEnabled(True)
    window._refresh_edit_actions()
    assert not window._act_edit_undo.isEnabled()
    assert not window._act_edit_redo.isEnabled()


def test_a_focused_button_or_a_buttonless_history_is_not_an_editor(qtbot):
    from PySide6.QtWidgets import QPushButton

    window = _window(qtbot)
    button = QPushButton("Run")
    screen = _screen(window, button)
    window._edit_focus = button
    assert window._edit_target() is None
    calls = []
    screen._history = []
    screen._box_history = []
    screen._on_undo = lambda: calls.append("undo")
    screen._on_redo = lambda: calls.append("redo")
    assert window._edit_target() is None
    window._invoke_edit_action(redo=True)
    assert calls == []


def test_an_unavailable_step_is_not_run(qtbot):
    window = _window(qtbot)
    editor = QTextEdit()
    _screen(window, editor)
    editor.setPlainText("kept")
    editor.document().clearUndoRedoStacks()
    window._edit_focus = editor
    window._invoke_edit_action(redo=False)
    window._invoke_edit_action(redo=True)
    assert editor.toPlainText() == "kept"
    assert not window._act_edit_undo.isEnabled()
