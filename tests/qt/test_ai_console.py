"""Tests for the AI Console — CLI-subprocess providers, prompts,
panel wiring, and Explain-error flow.

The vendor CLIs (`claude` / `codex` / `gemini`) may or may not be
installed on the test machine. Tests use `shutil.which` monkeypatches
so behaviour is deterministic.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Providers registry
# ---------------------------------------------------------------------------

def test_list_providers_returns_three():
    from spacr.qt.ai import providers
    names = {p.name for p in providers.list_providers()}
    assert names == {"claude", "codex", "gemini"}


def test_each_provider_has_cli_and_install_hint():
    from spacr.qt.ai import providers
    for p in providers.list_providers():
        assert p.cli_name
        assert p.install_hint
        assert p.login_command


def test_is_installed_uses_which(monkeypatch):
    """Force `shutil.which` to lie so we can verify is_installed()
    tracks the CLI's presence on PATH."""
    from spacr.qt.ai import providers as pmod

    fake = {"claude": "/opt/bin/claude"}   # only claude "installed"
    monkeypatch.setattr(
        pmod.shutil, "which",
        lambda name: fake.get(name),
    )
    ps = {p.name: p for p in pmod.list_providers()}
    assert ps["claude"].is_installed()
    assert not ps["codex"].is_installed()
    assert not ps["gemini"].is_installed()


def test_configured_providers_matches_installed(monkeypatch):
    from spacr.qt.ai import providers as pmod
    monkeypatch.setattr(pmod.shutil, "which",
                          lambda name: "/opt/bin/" + name if name == "gemini" else None)
    configured = [p.name for p in pmod.configured_providers()]
    assert configured == ["gemini"]


def test_get_provider_by_name():
    from spacr.qt.ai import providers
    assert providers.get_provider("claude").name == "claude"
    assert providers.get_provider("codex").name == "codex"
    assert providers.get_provider("gemini").name == "gemini"
    assert providers.get_provider("unknown") is None


def test_source_of_key_reports_state(monkeypatch):
    """Legacy shim now reports CLI install state instead of API keys."""
    from spacr.qt.ai import providers as pmod
    monkeypatch.setattr(pmod.shutil, "which",
                          lambda name: "/opt/bin/claude" if name == "claude" else None)
    ps = {p.name: p for p in pmod.list_providers()}
    assert "CLI found" in ps["claude"].source_of_key()
    assert "not installed" in ps["codex"].source_of_key()


# ---------------------------------------------------------------------------
# _format_conversation
# ---------------------------------------------------------------------------

def test_format_conversation_prefixes_roles():
    from spacr.qt.ai.providers import _format_conversation
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "who are you"},
    ]
    out = _format_conversation(msgs, system="be nice")
    assert "System:" in out and "be nice" in out
    assert "User:\nhi" in out
    assert "Assistant:\nhello" in out
    assert out.rstrip().endswith("who are you")


# ---------------------------------------------------------------------------
# Legacy keys.py stub still importable
# ---------------------------------------------------------------------------

def test_keys_module_still_importable_but_inert():
    from spacr.qt.ai import keys as ai_keys
    assert ai_keys.get_key("claude") is None
    assert ai_keys.set_key("claude", "x") is False
    assert "n/a" in ai_keys.source_of("claude")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def test_default_prompt_mentions_spacr():
    from spacr.qt.ai import prompts
    body = prompts.default_system_prompt()
    assert "spaCR" in body or "spacr" in body


def test_error_explainer_prompt_asks_for_short_manual():
    from spacr.qt.ai import prompts
    body = prompts.error_explainer_prompt()
    assert "6" in body


def test_wrap_error_includes_active_app():
    from spacr.qt.ai.prompts import wrap_error_for_prompt
    body = wrap_error_for_prompt("boom", active_app="mask")
    assert "mask" in body
    assert "boom" in body


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

def test_ai_chat_panel_starts_on_empty_state(qtbot, qt_theme_applied, monkeypatch):
    """With no CLIs installed the panel should show the empty state."""
    from spacr.qt.ai import providers as pmod
    monkeypatch.setattr(pmod.shutil, "which", lambda name: None)

    from spacr.qt.widgets import AIChatPanel
    panel = AIChatPanel()
    qtbot.addWidget(panel)
    assert panel._stack.currentWidget() is panel._empty_state
    assert not panel._btn_send.isEnabled()


def test_ai_chat_panel_shows_chat_when_cli_installed(qtbot, qt_theme_applied,
                                                       monkeypatch):
    from spacr.qt.ai import providers as pmod
    monkeypatch.setattr(pmod.shutil, "which",
                          lambda n: f"/opt/bin/{n}" if n == "claude" else None)

    from spacr.qt.widgets import AIChatPanel
    panel = AIChatPanel()
    qtbot.addWidget(panel)
    # Provider combo populated with the installed CLI
    labels = [panel._provider_combo.itemText(i)
              for i in range(panel._provider_combo.count())]
    assert any("Claude" in l for l in labels)
    assert panel._stack.currentWidget() is panel._chat_scroll


# ---------------------------------------------------------------------------
# AppScreen Explain-error wiring
# ---------------------------------------------------------------------------

def test_app_screen_has_disabled_explain_button(qtbot, qt_theme_applied):
    from spacr.qt.screens.app_screen import AppScreen
    s = AppScreen("mask")
    qtbot.addWidget(s)
    assert hasattr(s, "_btn_explain")
    assert not s._btn_explain.isEnabled()


def test_app_screen_error_signal_carries_traceback(qtbot, qt_theme_applied):
    from spacr.qt.screens.app_screen import AppScreen
    s = AppScreen("mask")
    qtbot.addWidget(s)
    s._on_pipeline_error("boom\n  at line 3")
    assert s._btn_explain.isEnabled()
    with qtbot.waitSignal(s.error_explain_requested, timeout=1000) as blocker:
        s._btn_explain.click()
    args = blocker.args
    assert "boom" in args[0]
    assert args[1] == "mask"
