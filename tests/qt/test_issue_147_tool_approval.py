"""GitHub #147: a tool call that needs approval gets an approval prompt.

``claude -p`` cannot show its own permission prompt, so a command such as
``gh auth status`` was refused silently and the assistant kept asking the
user to approve a prompt that never appeared. The provider now reads the
JSON result's ``permission_denials``; the console asks in a dialog and, when
the user allows it, resumes the same session with exactly that call allowed.
Every other CLI is told there is no prompt, so it hands the command over.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32",
                                reason="the fake claude is a POSIX script")

DENIAL = {"tool_name": "Bash", "tool_use_id": "toolu_1",
          "tool_input": {"command": "gh auth status", "description": "Check"}}


def _result(text, denials=(), session="sess-1", is_error=False):
    return json.dumps({"type": "result", "subtype": "success",
                       "is_error": is_error, "result": text,
                       "session_id": session,
                       "permission_denials": list(denials)}) + "\n"


@pytest.fixture()
def fake_claude(tmp_path, monkeypatch):
    """A ``claude`` on PATH that denies ``gh auth status`` until it is allowed."""
    log = tmp_path / "calls.jsonl"
    script = tmp_path / "claude"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "argv = sys.argv[1:]\n"
        f"open({str(log)!r}, 'a').write(json.dumps(argv) + '\\n')\n"
        "allowed = argv[argv.index('--allowedTools') + 1:] if '--allowedTools' in argv else []\n"
        "if 'Bash(gh auth status)' in allowed:\n"
        f"    sys.stdout.write({_result('Logged in to github.com')!r})\n"
        "else:\n"
        f"    sys.stdout.write({_result('I need to run gh auth status.', [DENIAL])!r})\n",
        encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")

    def calls():
        return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

    return calls


@pytest.fixture()
def isolated_ai_settings(monkeypatch, tmp_path):
    from PySide6.QtCore import QSettings

    from spacr.qt.ai import settings as ai_settings

    store = QSettings(str(tmp_path / "ai.ini"), QSettings.IniFormat)
    monkeypatch.setattr(ai_settings, "_settings", lambda: store)
    return ai_settings


def test_the_provider_keeps_the_denied_calls_and_the_session(
        fake_claude, isolated_ai_settings):
    from spacr.qt.ai.providers import ClaudeCliProvider

    provider = ClaudeCliProvider()
    reply = "".join(provider.stream_chat([{"role": "user", "content": "gh?"}]))
    assert reply == "I need to run gh auth status.\n"
    assert provider.last_permission_denials == [DENIAL]
    assert provider.last_session_id == "sess-1"
    assert fake_claude()[0][-2:] == ["--output-format", "json"]


def test_allow_once_resumes_with_exactly_that_call_for_one_reply(
        fake_claude, isolated_ai_settings):
    from spacr.qt.ai.providers import ClaudeCliProvider

    provider = ClaudeCliProvider()
    assert provider.allow_once([DENIAL], "sess-1") == ["Bash(gh auth status)"]
    reply = "".join(provider.stream_chat([
        {"role": "user", "content": "gh?"},
        {"role": "user", "content": "Approved; run it."}]))
    assert reply == "Logged in to github.com\n"
    assert provider.last_permission_denials == []
    first = fake_claude()[0]
    assert first[:2] == ["-p", "Approved; run it."]
    assert first[first.index("--resume") + 1] == "sess-1"
    assert first[first.index("--allowedTools") + 1] == "Bash(gh auth status)"

    "".join(provider.stream_chat([{"role": "user", "content": "again"}]))
    second = fake_claude()[1]
    assert "--resume" not in second and "--allowedTools" not in second


def test_a_failed_json_result_quotes_its_text_not_the_json(monkeypatch):
    from spacr.qt.ai import providers as pmod

    def failing(argv, **kwargs):
        yield _result("Not logged in · Please run /login", is_error=True)
        raise pmod.ProviderFailed("claude", 1, '{"type": "result"}')

    monkeypatch.setattr(pmod, "_stream_process", failing)
    provider = pmod.ClaudeCliProvider()
    with pytest.raises(pmod.ProviderFailed) as caught:
        list(provider.stream_chat([{"role": "user", "content": "hi"}], model="haiku"))
    assert "Not logged in" in str(caught.value)
    assert '"type"' not in str(caught.value)


@pytest.mark.parametrize("denial, rule, line", [
    (DENIAL, "Bash(gh auth status)", "gh auth status"),
    ({"tool_name": "WebFetch", "tool_input": {"url": "https://api.github.com/x"}},
     "WebFetch(domain:api.github.com)", "WebFetch: https://api.github.com/x"),
    ({"tool_name": "Write", "tool_input": {"file_path": "/tmp/a.txt"}},
     "Write", "Write: /tmp/a.txt"),
    ({"tool_input": {}}, "", "?"),
])
def test_each_denied_call_becomes_one_rule_and_one_line(denial, rule, line):
    from spacr.qt.ai.providers import describe_tool_call, tool_rule

    assert tool_rule(denial) == rule
    assert describe_tool_call(denial) == line


def test_every_cli_is_told_how_approval_works():
    from spacr.qt.ai.prompts import tool_approval_note

    assert "approval dialog" in tool_approval_note("claude")
    for name in ("codex", "gemini"):
        note = tool_approval_note(name)
        assert "no approval prompt" in note and "own terminal" in note


def _panel(qtbot, monkeypatch, provider):
    from spacr.qt import ai as ai_module
    from spacr.qt.ai import providers as pmod
    from spacr.qt.widgets.console_panel import ConsolePanel

    monkeypatch.setattr(ai_module, "get_provider", lambda name: provider)
    monkeypatch.setattr(pmod, "get_provider", lambda name: provider)
    panel = ConsolePanel()
    qtbot.addWidget(panel)
    panel.set_ai_provider("claude")
    panel.set_ai_active(True)
    return panel


def _wait_idle(panel, qtbot):
    qtbot.waitUntil(lambda: panel._ai_thread is None, timeout=15000)


def _answer_dialog(monkeypatch, button_text, shown):
    from PySide6.QtWidgets import QMessageBox

    def exec_(box):
        shown.append((box.objectName(), box.windowTitle(), box.text()))
        for button in box.buttons():
            if button.text() == button_text:
                button.click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", exec_)


def test_the_console_asks_and_an_allowed_call_runs(
        qtbot, qt_theme_applied, monkeypatch, fake_claude, isolated_ai_settings):
    from spacr.qt.ai.providers import ClaudeCliProvider

    shown = []
    _answer_dialog(monkeypatch, "Allow once", shown)
    panel = _panel(qtbot, monkeypatch, ClaudeCliProvider())
    try:
        panel._input.setPlainText("Is gh signed in?")
        panel._on_submit()
        _wait_idle(panel, qtbot)
        qtbot.waitUntil(lambda: len(fake_claude()) == 2, timeout=15000)
        _wait_idle(panel, qtbot)
    finally:
        panel.shutdown()

    assert len(shown) == 1
    name, title, text = shown[0]
    assert name == "aiToolApprovalBox"
    assert title == "Allow the assistant to run this?"
    assert "gh auth status" in text
    first, second = fake_claude()
    system = first[first.index("--append-system-prompt") + 1]
    assert "approval dialog" in system
    assert second[second.index("--resume") + 1] == "sess-1"
    assert "Bash(gh auth status)" in second
    assert panel._ai_messages[-1] == {"role": "assistant",
                                      "content": "Logged in to github.com\n"}


def test_a_refused_call_is_not_run_and_the_user_is_told(
        qtbot, qt_theme_applied, monkeypatch, fake_claude, isolated_ai_settings):
    from spacr.qt.ai.providers import ClaudeCliProvider

    shown, notices = [], []
    _answer_dialog(monkeypatch, "Don't allow", shown)
    provider = ClaudeCliProvider()
    panel = _panel(qtbot, monkeypatch, provider)
    monkeypatch.setattr(panel, "append_notice",
                        lambda source, **values: notices.append(source.format(**values)))
    try:
        panel._input.setPlainText("Is gh signed in?")
        panel._on_submit()
        _wait_idle(panel, qtbot)
    finally:
        panel.shutdown()

    assert len(shown) == 1
    assert len(fake_claude()) == 1
    assert provider._approved_tools == [] and provider._approved_resume == ""
    assert any(n.startswith("[AI] Not run: gh auth status.") for n in notices)
