"""Tests for spacr.qt.ai.settings — persisted speed + prompt override."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_qsettings(monkeypatch, qt_theme_applied, tmp_path):
    """Point QSettings at an isolated .ini file so tests don't leak
    into the developer's real preferences."""
    from PySide6.QtCore import QCoreApplication, QSettings
    QCoreApplication.setOrganizationName("spacr-test")
    QCoreApplication.setApplicationName("qt-ai-settings-test")
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope,
                        str(tmp_path))
    # Reset any prior state per test
    from spacr.qt.ai import settings as s
    QSettings("spacr", "qt").clear()
    # Re-mark first-launch tour seen after the QSettings clear so the
    # autouse conftest fixture keeps its promise.
    try:
        from spacr.qt.first_run import mark_tour_seen
        mark_tour_seen()
    except Exception:
        pass
    yield


def test_speed_default_is_balanced(qt_theme_applied):
    from spacr.qt.ai import settings as s
    assert s.get_response_speed() == "balanced"


def test_speed_roundtrip(qt_theme_applied):
    from spacr.qt.ai import settings as s
    s.set_response_speed("fast")
    assert s.get_response_speed() == "fast"
    s.set_response_speed("deep")
    assert s.get_response_speed() == "deep"


def test_speed_rejects_unknown_value(qt_theme_applied):
    from spacr.qt.ai import settings as s
    with pytest.raises(ValueError):
        s.set_response_speed("turbo")


def test_speed_recovers_from_corrupted_value(qt_theme_applied):
    from spacr.qt.ai import settings as s
    from PySide6.QtCore import QSettings
    QSettings("spacr", "qt").setValue("ai/response_speed", "garbage")
    assert s.get_response_speed() == "balanced"


@pytest.mark.parametrize("provider,expected_first", [
    ("claude", "--model"),
    ("codex",  "--model"),
    ("gemini", "--model"),
])
def test_provider_args_returns_speed_flags(qt_theme_applied,
                                              provider, expected_first):
    from spacr.qt.ai import settings as s
    s.set_response_speed("balanced")
    args = s.provider_args(provider)
    assert args, f"{provider!r} should get some argv for balanced"
    assert args[0] == expected_first


def test_provider_args_empty_for_unknown_provider(qt_theme_applied):
    from spacr.qt.ai import settings as s
    assert s.provider_args("nonexistent-provider") == []


def test_system_prompt_default_matches_baseline(qt_theme_applied):
    from spacr.qt.ai import settings as s
    from spacr.qt.ai.prompts import default_system_prompt
    assert s.get_system_prompt() == default_system_prompt()
    assert s.is_system_prompt_overridden() is False


def test_system_prompt_override(qt_theme_applied):
    from spacr.qt.ai import settings as s
    s.set_system_prompt("Answer only in haiku form.")
    assert s.get_system_prompt() == "Answer only in haiku form."
    assert s.is_system_prompt_overridden() is True


def test_system_prompt_empty_string_clears_override(qt_theme_applied):
    from spacr.qt.ai import settings as s
    s.set_system_prompt("something custom")
    assert s.is_system_prompt_overridden() is True
    s.set_system_prompt("   ")   # whitespace-only counts as empty
    assert s.is_system_prompt_overridden() is False


def test_system_prompt_reset(qt_theme_applied):
    from spacr.qt.ai import settings as s
    from spacr.qt.ai.prompts import default_system_prompt
    s.set_system_prompt("my override")
    s.reset_system_prompt()
    assert s.get_system_prompt() == default_system_prompt()
    assert s.is_system_prompt_overridden() is False


def test_providers_dialog_has_settings_tab(qtbot, qt_theme_applied):
    """Smoke-check the tabbed dialog boots and has the Settings tab."""
    from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog
    dlg = _ProvidersDialog()
    qtbot.addWidget(dlg)
    tabs = dlg.findChildren(type(dlg.layout().itemAt(0).widget()))
    # Find the QTabWidget child directly
    from PySide6.QtWidgets import QTabWidget
    tabwidget = dlg.findChild(QTabWidget)
    assert tabwidget is not None
    labels = [tabwidget.tabText(i) for i in range(tabwidget.count())]
    assert "Providers" in labels
    assert "Settings" in labels
