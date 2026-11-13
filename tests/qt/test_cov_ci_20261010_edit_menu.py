"""Edit menu boundaries: rich text editors, vanished focus and no editor."""

from PySide6.QtWidgets import QVBoxLayout, QWidget

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


