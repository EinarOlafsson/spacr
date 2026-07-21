"""Tests for the AI Console — key store, provider registry, screen wiring.

The vendor SDKs (anthropic / openai / google-generativeai) are NOT
required for these tests. The registry lists all three regardless of
whether their SDKs are installed; `is_configured()` just returns False
when they aren't.
"""
from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# keys.py
# ---------------------------------------------------------------------------

def test_get_key_reads_env_var_first(monkeypatch):
    from spacr.qt.ai import keys as ai_keys
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-value")
    assert ai_keys.get_key("anthropic") == "sk-env-value"
    assert ai_keys.source_of("anthropic").startswith("env ")


def test_get_key_returns_none_when_absent(monkeypatch):
    from spacr.qt.ai import keys as ai_keys
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    # Also monkey-patch keyring to look empty so this test doesn't
    # depend on the developer's real keychain state.
    import spacr.qt.ai.keys as keys_module

    class _FakeKeyring:
        @staticmethod
        def get_password(service, account): return None
        @staticmethod
        def set_password(service, account, key): return None
        @staticmethod
        def delete_password(service, account): return None

    monkeypatch.setattr(keys_module, "keyring", _FakeKeyring, raising=False)
    # get_key imports keyring lazily; patch sys.modules too so the
    # inline `import keyring` inside get_key sees the fake.
    import sys
    monkeypatch.setitem(sys.modules, "keyring", _FakeKeyring)
    assert ai_keys.get_key("anthropic") is None
    assert ai_keys.get_key("openai") is None
    assert ai_keys.get_key("google") is None


# ---------------------------------------------------------------------------
# providers.py
# ---------------------------------------------------------------------------

def test_list_providers_returns_three():
    from spacr.qt.ai import providers
    names = {p.name for p in providers.list_providers()}
    assert names == {"anthropic", "openai", "google"}


def test_configured_providers_empty_when_no_keys(monkeypatch):
    from spacr.qt.ai import providers
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    import sys
    class _K:
        @staticmethod
        def get_password(*a, **kw): return None
        @staticmethod
        def set_password(*a, **kw): return None
        @staticmethod
        def delete_password(*a, **kw): return None
    monkeypatch.setitem(sys.modules, "keyring", _K)
    assert providers.configured_providers() == []


def test_get_provider_by_name():
    from spacr.qt.ai import providers
    p = providers.get_provider("anthropic")
    assert p is not None
    assert p.name == "anthropic"
    assert providers.get_provider("not-a-provider") is None


def test_provider_install_hints_populated():
    from spacr.qt.ai import providers
    for p in providers.list_providers():
        assert p.install_hint.startswith("pip install")


# ---------------------------------------------------------------------------
# prompts.py
# ---------------------------------------------------------------------------

def test_default_prompt_mentions_spacr():
    from spacr.qt.ai import prompts
    body = prompts.default_system_prompt()
    assert "SpaCR" in body or "spacr" in body


def test_error_explainer_prompt_asks_for_short_manual():
    from spacr.qt.ai import prompts
    body = prompts.error_explainer_prompt()
    assert "6" in body   # <=6 steps constraint appears


def test_wrap_error_includes_active_app():
    from spacr.qt.ai.prompts import wrap_error_for_prompt
    body = wrap_error_for_prompt("boom", active_app="mask")
    assert "mask" in body
    assert "boom" in body


# ---------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------

def test_ai_chat_panel_starts_on_empty_state(qtbot, qt_theme_applied, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    import sys
    class _K:
        @staticmethod
        def get_password(*a, **kw): return None
        @staticmethod
        def set_password(*a, **kw): return None
        @staticmethod
        def delete_password(*a, **kw): return None
    monkeypatch.setitem(sys.modules, "keyring", _K)

    from spacr.qt.widgets import AIChatPanel
    panel = AIChatPanel()
    qtbot.addWidget(panel)
    # Without any configured provider we render the empty-state pane
    assert panel._stack.currentWidget() is panel._empty_state
    assert not panel._btn_send.isEnabled()


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
