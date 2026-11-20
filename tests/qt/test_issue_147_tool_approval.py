"""GitHub #147: a blocked tool call must not leave the user waiting for a prompt.

``claude -p`` cannot show its own permission prompt, so a command such as
``gh auth status`` was refused silently and the assistant kept asking the
user to approve a prompt that never appeared. Since 2026-11-18 every spaCR
AI call runs without tools, so the console's approval dialog is gone: every
CLI is told it has no tools and hands the user the command to run instead.
The provider still reads the JSON result's ``permission_denials``."""
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


def test_every_cli_is_told_it_has_no_tools():
    from spacr.qt.ai.prompts import tool_approval_note

    notes = {name: tool_approval_note(name) for name in ("claude", "codex", "gemini")}
    assert len(set(notes.values())) == 1
    note = notes["claude"]
    assert "no tools" in note and "own terminal" in note and "approval dialog" not in note
