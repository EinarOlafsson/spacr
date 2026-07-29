"""Annotate's collapsible Console + AI controls."""
from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtCore import Qt


def _console_text(console) -> str:
    from spacr.qt.widgets.console_panel import _StdoutBlock

    return "\n".join(
        block.text() for block in console.findChildren(_StdoutBlock))


def test_annotate_has_collapsible_console_and_bottom_ai_controls(
        qtbot, monkeypatch):
    from spacr.qt import ai as ai_module
    monkeypatch.setattr(ai_module, "configured_providers", lambda: [])
    from spacr.qt.screens.annotate import AnnotateScreen
    from spacr.qt.widgets import AiToggleLabel, ConsolePanel

    screen = AnnotateScreen()
    qtbot.addWidget(screen)

    assert isinstance(screen._console, ConsolePanel)
    assert isinstance(screen._ai_switch, AiToggleLabel)
    assert screen._console_wrap.isHidden()
    assert screen._console_switch.text() == "Console ▾"

    screen._console_switch.setChecked(True)
    assert not screen._console_wrap.isHidden()
    assert screen._console_switch.text() == "Console ▴"
    screen._console_switch.setChecked(False)
    assert screen._console_wrap.isHidden()


def test_ai_button_expands_console_and_explains_missing_provider(
        qtbot, monkeypatch):
    from spacr.qt import ai as ai_module
    monkeypatch.setattr(ai_module, "configured_providers", lambda: [])
    from spacr.qt.screens.annotate import AnnotateScreen

    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    screen._ai_switch.setChecked(True)

    assert screen._console_switch.isChecked()
    assert not screen._console_wrap.isHidden()
    assert screen._ai_switch.isChecked() is False
    assert "No vendor CLI installed" in _console_text(screen._console)


def test_ai_button_selects_first_provider_and_menu_can_switch(
        qtbot, monkeypatch):
    from spacr.qt import ai as ai_module
    providers = [
        SimpleNamespace(name="codex", label="Codex"),
        SimpleNamespace(name="claude", label="Claude"),
    ]
    monkeypatch.setattr(ai_module, "configured_providers", lambda: providers)
    from spacr.qt.screens.annotate import AnnotateScreen

    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    screen._ai_switch.setChecked(True)
    assert screen._console._current_provider_name == "codex"

    claude_action = next(
        action for action in screen._ai_menu.actions()
        if action.text() == "Claude")
    claude_action.trigger()
    assert screen._console._current_provider_name == "claude"


def test_annotate_close_drains_ai_console(qtbot, monkeypatch):
    from spacr.qt import ai as ai_module
    monkeypatch.setattr(ai_module, "configured_providers", lambda: [])
    from spacr.qt.screens.annotate import AnnotateScreen

    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    calls = []
    monkeypatch.setattr(screen._console, "shutdown", lambda: calls.append(True))
    screen.close()
    assert calls == [True]
