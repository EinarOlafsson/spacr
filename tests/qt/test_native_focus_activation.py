"""Only native Cocoa window-manager refusal can skip keyboard-focus checks."""

from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton, QVBoxLayout, QWidget

from tests.qt import conftest


class _Window:
    def __init__(self, active=False, visible=True):
        self.active = active
        self.visible = visible
        self.calls = []

    def raise_(self):
        self.calls.append("raise")

    def activateWindow(self):
        self.calls.append("activate")

    def isActiveWindow(self):
        return self.active

    def isVisible(self):
        return self.visible


def test_an_active_top_level_keeps_focus_assertions_on_the_normal_path(monkeypatch):
    window = _Window(active=True)
    widget = SimpleNamespace(window=lambda: window)
    waits, events = [], []
    with monkeypatch.context() as patch:
        patch.setattr(QApplication, "processEvents", staticmethod(lambda: events.append(True)))
        conftest._activate_focus_window(widget, SimpleNamespace(wait=waits.append))
    assert window.calls == ["raise", "activate"]
    assert waits == [] and events == [True]


def test_delayed_activation_is_allowed_before_the_five_second_limit(monkeypatch):
    window = _Window()
    waits = []

    def wait(milliseconds):
        waits.append(milliseconds)
        window.active = len(waits) == 2

    with monkeypatch.context() as patch:
        patch.setattr(QApplication, "processEvents", staticmethod(lambda: None))
        conftest._activate_focus_window(SimpleNamespace(window=lambda: window),
                                       SimpleNamespace(wait=wait))
    assert window.calls == ["raise", "activate"]
    assert waits == [100, 100]


def test_a_window_that_was_not_shown_fails_before_any_platform_skip():
    window = _Window(visible=False)
    waits = []
    with pytest.raises(AssertionError, match="must be shown"):
        conftest._activate_focus_window(SimpleNamespace(window=lambda: window),
                                       SimpleNamespace(wait=waits.append))
    assert window.calls == [] and waits == []


def test_only_native_cocoa_refusal_skips_after_bounded_activation(monkeypatch):
    window = _Window()
    waits = []
    with monkeypatch.context() as patch:
        patch.setattr(conftest.sys, "platform", "darwin")
        patch.setattr(QApplication, "platformName", staticmethod(lambda: "cocoa"))
        with pytest.raises(pytest.skip.Exception, match="macOS cooperative activation"):
            conftest._activate_focus_window(SimpleNamespace(window=lambda: window),
                                           SimpleNamespace(wait=waits.append))
    assert window.calls == ["raise", "activate"]
    assert waits == [100] * 50


@pytest.mark.parametrize("system, backend", [
    ("darwin", "offscreen"),
    ("linux", "xcb"),
    ("win32", "windows"),
])
def test_other_activation_failures_are_not_hidden_by_a_skip(monkeypatch, system, backend):
    window = _Window()
    waits = []
    with monkeypatch.context() as patch:
        patch.setattr(conftest.sys, "platform", system)
        patch.setattr(QApplication, "platformName", staticmethod(lambda: backend))
        with pytest.raises(pytest.fail.Exception, match="focus needs the active window"):
            conftest._activate_focus_window(SimpleNamespace(window=lambda: window),
                                           SimpleNamespace(wait=waits.append))
    assert window.calls == ["raise", "activate"]
    assert waits == [100] * 50


def test_all_control_navigation_is_test_local_and_restores_the_real_tab_route(qtbot, qapp):
    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    before, button, after = QLineEdit(), QPushButton("Native button"), QLineEdit()
    button.setFocusPolicy(Qt.TabFocus)
    for widget in (before, button, after):
        layout.addWidget(widget)
    host.show()
    conftest._activate_focus_window(host, qtbot)
    hints = qapp.styleHints()
    original = hints.tabFocusBehavior()

    def next_from_text():
        before.setFocus(Qt.TabFocusReason)
        qtbot.waitUntil(before.hasFocus)
        qtbot.keyClick(before, Qt.Key_Tab)

    try:
        hints.setTabFocusBehavior(Qt.TabFocusTextControls)
        next_from_text()
        assert after.hasFocus() and not button.hasFocus()
        with conftest._all_controls_tab_focus(qapp):
            assert hints.tabFocusBehavior() == Qt.TabFocusAllControls
            next_from_text()
            assert button.hasFocus() and not after.hasFocus()
        assert hints.tabFocusBehavior() == Qt.TabFocusTextControls
        next_from_text()
        assert after.hasFocus() and not button.hasFocus()
    finally:
        hints.setTabFocusBehavior(original)
