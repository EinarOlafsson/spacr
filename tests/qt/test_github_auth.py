"""Tests for GitHub auth + direct issue creation."""
from __future__ import annotations

import io
import json

import pytest


@pytest.fixture(autouse=True)
def _clear_token(qt_theme_applied):
    from spacr.qt.ai import github_auth
    github_auth.set_stored_token("")
    yield
    github_auth.set_stored_token("")


def test_stored_token_roundtrip():
    from spacr.qt.ai import github_auth
    github_auth.set_stored_token("ghp_abc123")
    assert github_auth.get_stored_token() == "ghp_abc123"
    assert github_auth.is_authenticated()
    assert github_auth.auth_source() == "token"
    github_auth.set_stored_token("")
    assert github_auth.get_stored_token() == ""


def test_resolve_prefers_stored_then_env(monkeypatch):
    from spacr.qt.ai import github_auth
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    # No stored, no env, and force gh CLI to fail -> not authenticated.
    monkeypatch.setattr(github_auth, "_gh_cli_token", lambda: "")
    assert not github_auth.is_authenticated()
    # env token picked up
    monkeypatch.setenv("GITHUB_TOKEN", "env_tok")
    assert github_auth.resolve_token() == ("env_tok", "env")
    # stored beats env
    github_auth.set_stored_token("stored_tok")
    assert github_auth.resolve_token() == ("stored_tok", "token")


def test_create_issue_posts_and_returns_url(monkeypatch):
    from spacr.qt.ai import github_auth
    github_auth.set_stored_token("ghp_x")
    captured = {}

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def _fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["auth"] = req.headers.get("Authorization")
        captured["body"] = json.loads(req.data.decode())
        return _Resp(json.dumps(
            {"html_url": "https://github.com/o/r/issues/7"}).encode())

    monkeypatch.setattr(github_auth.urllib.request, "urlopen", _fake_urlopen)
    ok, url = github_auth.create_issue("o/r", "boom", "trace", labels=["auto-filed"])
    assert ok and url == "https://github.com/o/r/issues/7"
    assert captured["url"] == "https://api.github.com/repos/o/r/issues"
    assert captured["auth"] == "Bearer ghp_x"
    assert captured["body"]["title"] == "boom"
    assert captured["body"]["labels"] == ["auto-filed"]


def test_create_issue_without_token_fails(monkeypatch):
    from spacr.qt.ai import github_auth
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(github_auth, "_gh_cli_token", lambda: "")
    ok, err = github_auth.create_issue("o/r", "t", "b")
    assert not ok and "Not signed in" in err


def test_file_issue_uses_api_when_authenticated(monkeypatch):
    from spacr.qt.ai import issue_report, github_auth
    github_auth.set_stored_token("ghp_x")
    monkeypatch.setattr(github_auth, "create_issue",
                        lambda *a, **k: (True, "https://github.com/o/r/issues/9"))
    opened = {"browser": False}
    monkeypatch.setattr(issue_report, "open_issue_in_browser",
                        lambda url: opened.__setitem__("browser", True))
    url = issue_report.file_issue("Traceback...", active_app="mask")
    assert url == "https://github.com/o/r/issues/9"
    assert opened["browser"] is False   # no browser when signed in
