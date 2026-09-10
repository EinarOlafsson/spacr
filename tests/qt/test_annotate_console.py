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


def test_ai_button_takes_the_provider_the_preference_names(qtbot, monkeypatch):
    """The switch honours Preferences, and falls back to the first installed.

    WAS A MENU TEST. Annotate used to carry a "\u25be" beside the AI switch
    that listed the providers, and this drove it. That menu is gone: which
    assistant to use is a preference with one answer for the whole
    application, not a control repeated on every module's actions row, and it
    did not even persist -- `get_preferred_provider` existed and was read by
    nothing while the chevron wrote to the console.

    So the question the test asks is the same one, put to the new place: does
    turning the switch on select the right provider?
    """
    from spacr.qt import ai as ai_module
    from spacr.qt import preferences as prefs
    providers = [
        SimpleNamespace(name="codex", label="Codex"),
        SimpleNamespace(name="claude", label="Claude"),
    ]
    monkeypatch.setattr(ai_module, "configured_providers", lambda: providers)
    from spacr.qt.screens.annotate import AnnotateScreen

    was = prefs.get_preferred_provider()
    try:
        # Nothing chosen: the first installed one, as the menu's default did.
        prefs.set_preferred_provider("")
        screen = AnnotateScreen()
        qtbot.addWidget(screen)
        screen._ai_switch.setChecked(True)
        assert screen._console._current_provider_name == "codex"

        # Chosen in Preferences: that one, which is what the menu used to do
        # by hand and now survives the module being closed and reopened.
        prefs.set_preferred_provider("claude")
        second = AnnotateScreen()
        qtbot.addWidget(second)
        second._ai_switch.setChecked(True)
        assert second._console._current_provider_name == "claude"
    finally:
        prefs.set_preferred_provider(was)

    assert not hasattr(screen, "_ai_menu"), "the provider chevron came back"


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
