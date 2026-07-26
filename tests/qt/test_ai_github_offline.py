"""Offline tests for ``spacr.qt.ai.github_auth`` (+ keys / settings stubs).

Every HTTP call is intercepted at the transport boundary
(``urllib.request.urlopen``) and the request that WOULD have been sent is
asserted: URL, method, all five headers and the JSON body shape. The
response-parsing and error paths — 401, 422, 429, network timeout,
malformed JSON, empty body — are exercised with synthetic responses.

Nothing here opens a socket, and the QSettings store is redirected to a
temp .ini file so the developer's real preferences are never touched.
"""
from __future__ import annotations

import email.message
import io
import json
import socket
import subprocess
import urllib.error

import pytest

from spacr.qt.ai import github_auth as _gh_mod
from spacr.qt.ai import settings as _ai_settings_mod

# Captured at import time, i.e. before any fixture redirects them, so the
# real QSettings handles stay testable.
_REAL_GH_SETTINGS = _gh_mod._settings
_REAL_AI_SETTINGS = _ai_settings_mod._settings


@pytest.fixture(autouse=True)
def gh_store(monkeypatch, tmp_path, qt_theme_applied):
    """Redirect github_auth's QSettings to an isolated .ini file."""
    from PySide6.QtCore import QSettings
    from spacr.qt.ai import github_auth

    store = QSettings(str(tmp_path / "gh.ini"), QSettings.IniFormat)
    monkeypatch.setattr(github_auth, "_settings", lambda: store)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)

    # Never let a developer machine's real `gh` CLI answer during a test.
    # Patched at the subprocess boundary so the real _gh_cli_token() body
    # stays under test; individual tests re-patch `run` with their own.
    def _no_gh(*a, **k):
        raise FileNotFoundError("gh")

    monkeypatch.setattr(github_auth.subprocess, "run", _no_gh)
    return store


class _Resp(io.BytesIO):
    """Context-manager response, like the object urlopen returns."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code, reason, body: bytes):
    hdrs = email.message.Message()
    hdrs["Content-Type"] = "application/json"
    return urllib.error.HTTPError(
        "https://api.github.com/repos/o/r/issues", code, reason, hdrs,
        io.BytesIO(body),
    )


def _stub_urlopen(monkeypatch, handler):
    """Replace urlopen with `handler(req, timeout)`; record what it saw."""
    from spacr.qt.ai import github_auth
    seen = {}

    def _fake(req, timeout=None):
        seen["url"] = req.full_url
        seen["method"] = req.get_method()
        seen["headers"] = dict(req.headers)
        seen["timeout"] = timeout
        seen["raw_body"] = req.data
        return handler(req, timeout)

    monkeypatch.setattr(github_auth.urllib.request, "urlopen", _fake)
    return seen


# ---------------------------------------------------------------------------
# Token storage
# ---------------------------------------------------------------------------

def test_both_stores_use_the_shared_spacr_qt_qsettings(qt_theme_applied):
    """The PAT and the AI prefs must land in the same QSettings store the
    rest of the Qt GUI uses, or they silently vanish between launches."""
    for handle in (_REAL_GH_SETTINGS(), _REAL_AI_SETTINGS()):
        assert handle.organizationName() == "spacr"
        assert handle.applicationName() == "qt"


def test_stored_token_is_trimmed_on_save():
    from spacr.qt.ai import github_auth as ga
    ga.set_stored_token("  ghp_padded  ")
    assert ga.get_stored_token() == "ghp_padded"


def test_whitespace_only_token_clears_the_store():
    from spacr.qt.ai import github_auth as ga
    ga.set_stored_token("ghp_real")
    assert ga.is_authenticated()
    ga.set_stored_token("   ")
    assert ga.get_stored_token() == ""
    assert ga.auth_source() is None


def test_none_token_clears_the_store():
    from spacr.qt.ai import github_auth as ga
    ga.set_stored_token("ghp_real")
    ga.set_stored_token(None)
    assert ga.get_stored_token() == ""


# ---------------------------------------------------------------------------
# Token resolution order: stored > env > gh CLI
# ---------------------------------------------------------------------------

def test_resolve_token_falls_through_to_gh_cli(monkeypatch):
    from spacr.qt.ai import github_auth as ga
    monkeypatch.setattr(ga, "_gh_cli_token", lambda: "gho_from_cli")
    assert ga.resolve_token() == ("gho_from_cli", "gh")
    assert ga.auth_source() == "gh"
    assert ga.is_authenticated() is True


def test_resolve_token_prefers_github_token_over_gh_token(monkeypatch):
    from spacr.qt.ai import github_auth as ga
    monkeypatch.setenv("GH_TOKEN", "second")
    monkeypatch.setenv("GITHUB_TOKEN", "first")
    assert ga.resolve_token() == ("first", "env")


def test_resolve_token_uses_gh_token_when_github_token_blank(monkeypatch):
    from spacr.qt.ai import github_auth as ga
    monkeypatch.setenv("GITHUB_TOKEN", "   ")     # blank -> skipped
    monkeypatch.setenv("GH_TOKEN", "  fallback  ")
    assert ga.resolve_token() == ("fallback", "env")


def test_resolve_token_returns_none_source_when_nothing_available():
    from spacr.qt.ai import github_auth as ga
    assert ga.resolve_token() == ("", None)
    assert ga.is_authenticated() is False


# ---------------------------------------------------------------------------
# gh CLI probe — subprocess boundary
# ---------------------------------------------------------------------------

def _run_result(returncode, stdout=""):
    return subprocess.CompletedProcess(["gh", "auth", "token"],
                                       returncode, stdout, "")


def test_gh_cli_token_reads_stdout(monkeypatch):
    from spacr.qt.ai import github_auth as ga
    seen = {}

    def _run(argv, **kw):
        seen["argv"] = argv
        seen["kw"] = kw
        return _run_result(0, "gho_abc123\n")

    monkeypatch.setattr(ga.subprocess, "run", _run)
    assert ga._gh_cli_token() == "gho_abc123"
    assert seen["argv"] == ["gh", "auth", "token"]
    assert seen["kw"]["capture_output"] is True     # never leaks to the console
    assert seen["kw"]["text"] is True
    assert seen["kw"]["timeout"] == 8               # bounded, can't hang the UI


def test_gh_cli_token_empty_when_not_logged_in(monkeypatch):
    from spacr.qt.ai import github_auth as ga
    monkeypatch.setattr(ga.subprocess, "run",
                        lambda *a, **k: _run_result(1, ""))
    assert ga._gh_cli_token() == ""


@pytest.mark.parametrize("exc", [
    FileNotFoundError("gh"),
    subprocess.TimeoutExpired(cmd="gh", timeout=8),
    subprocess.SubprocessError("boom"),
    OSError("exec format error"),
])
def test_gh_cli_token_swallows_every_launch_failure(monkeypatch, exc):
    from spacr.qt.ai import github_auth as ga

    def _boom(*a, **k):
        raise exc

    monkeypatch.setattr(ga.subprocess, "run", _boom)
    assert ga._gh_cli_token() == ""


# ---------------------------------------------------------------------------
# create_issue — the request we WOULD have sent
# ---------------------------------------------------------------------------

def test_create_issue_request_shape(monkeypatch):
    from spacr.qt.ai import github_auth as ga
    ga.set_stored_token("ghp_secret_value")

    seen = _stub_urlopen(monkeypatch, lambda req, t: _Resp(json.dumps(
        {"html_url": "https://github.com/o/r/issues/42",
         "number": 42}).encode()))

    ok, url = ga.create_issue("EinarOlafsson/spacr", "crash on masks",
                              "body text", labels=["auto-filed", "bug"])

    assert (ok, url) == (True, "https://github.com/o/r/issues/42")
    assert seen["url"] == \
        "https://api.github.com/repos/EinarOlafsson/spacr/issues"
    assert seen["method"] == "POST"
    assert seen["timeout"] == 20
    # urllib title-cases header keys
    h = {k.lower(): v for k, v in seen["headers"].items()}
    assert h["authorization"] == "Bearer ghp_secret_value"
    assert h["accept"] == "application/vnd.github+json"
    assert h["x-github-api-version"] == "2022-11-28"
    assert h["user-agent"] == "spacr"
    assert h["content-type"] == "application/json"
    body = json.loads(seen["raw_body"].decode("utf-8"))
    assert body == {"title": "crash on masks", "body": "body text",
                    "labels": ["auto-filed", "bug"]}
    assert seen["raw_body"] == json.dumps(body).encode("utf-8")


@pytest.mark.parametrize("labels", [None, [], ()])
def test_create_issue_omits_labels_key_when_empty(monkeypatch, labels):
    from spacr.qt.ai import github_auth as ga
    ga.set_stored_token("ghp_x")
    seen = _stub_urlopen(monkeypatch,
                         lambda req, t: _Resp(b'{"html_url": "u"}'))
    ok, url = ga.create_issue("o/r", "t", "b", labels=labels)
    assert (ok, url) == (True, "u")
    assert "labels" not in json.loads(seen["raw_body"].decode())


def test_create_issue_labels_are_copied_not_aliased(monkeypatch):
    """A caller mutating its list afterwards must not change what was sent."""
    from spacr.qt.ai import github_auth as ga
    ga.set_stored_token("ghp_x")
    sent = {}

    def _handler(req, t):
        sent["body"] = json.loads(req.data.decode())
        return _Resp(b'{"html_url": "u"}')

    _stub_urlopen(monkeypatch, _handler)
    labels = ["auto-filed"]
    ga.create_issue("o/r", "t", "b", labels=labels)
    labels.append("mutated-after")
    assert sent["body"]["labels"] == ["auto-filed"]


def test_create_issue_unicode_body_is_utf8_encoded(monkeypatch):
    from spacr.qt.ai import github_auth as ga
    ga.set_stored_token("ghp_x")
    seen = _stub_urlopen(monkeypatch,
                         lambda req, t: _Resp(b'{"html_url": "u"}'))
    ga.create_issue("o/r", "Ångström ✂", "naïve — résumé")
    decoded = json.loads(seen["raw_body"].decode("utf-8"))
    assert decoded["title"] == "Ångström ✂"
    assert decoded["body"] == "naïve — résumé"


def test_create_issue_uses_env_token_when_nothing_stored(monkeypatch):
    from spacr.qt.ai import github_auth as ga
    monkeypatch.setenv("GITHUB_TOKEN", "env_only_token")
    seen = _stub_urlopen(monkeypatch,
                         lambda req, t: _Resp(b'{"html_url": "u"}'))
    ok, _ = ga.create_issue("o/r", "t", "b")
    assert ok
    assert seen["headers"]["Authorization"] == "Bearer env_only_token"


def test_create_issue_without_any_token_never_touches_the_network(monkeypatch):
    from spacr.qt.ai import github_auth as ga

    def _explode(*a, **k):
        raise AssertionError("urlopen must not be called without a token")

    monkeypatch.setattr(ga.urllib.request, "urlopen", _explode)
    ok, err = ga.create_issue("o/r", "t", "b", labels=["x"])
    assert ok is False
    assert err == "Not signed in to GitHub (no token available)."


# ---------------------------------------------------------------------------
# create_issue — response parsing + error paths
# ---------------------------------------------------------------------------

def test_create_issue_success_without_html_url_returns_empty_string(
        monkeypatch):
    from spacr.qt.ai import github_auth as ga
    ga.set_stored_token("ghp_x")
    _stub_urlopen(monkeypatch, lambda req, t: _Resp(b'{"number": 3}'))
    assert ga.create_issue("o/r", "t", "b") == (True, "")


def test_create_issue_401_reports_github_message(monkeypatch):
    from spacr.qt.ai import github_auth as ga
    ga.set_stored_token("ghp_dead_token")

    def _handler(req, t):
        raise _http_error(401, "Unauthorized", b'{"message": "Bad credentials"}')

    _stub_urlopen(monkeypatch, _handler)
    ok, err = ga.create_issue("o/r", "t", "b")
    assert ok is False
    assert err == "GitHub API error 401: Bad credentials"
    assert "ghp_dead_token" not in err        # the token must not leak


def test_create_issue_422_validation_failure(monkeypatch):
    from spacr.qt.ai import github_auth as ga
    ga.set_stored_token("ghp_x")

    def _handler(req, t):
        raise _http_error(422, "Unprocessable Entity",
                          b'{"message": "Validation Failed", "errors": []}')

    _stub_urlopen(monkeypatch, _handler)
    assert ga.create_issue("o/r", "", "b") == \
        (False, "GitHub API error 422: Validation Failed")


def test_create_issue_429_with_non_json_body_falls_back_to_reason(monkeypatch):
    from spacr.qt.ai import github_auth as ga
    ga.set_stored_token("ghp_x")

    def _handler(req, t):
        raise _http_error(429, "Too Many Requests",
                          b"<html>rate limited</html>")

    _stub_urlopen(monkeypatch, _handler)
    ok, err = ga.create_issue("o/r", "t", "b")
    assert (ok, err) == (False, "GitHub API error 429: Too Many Requests")


def test_create_issue_error_with_empty_message_falls_back_to_reason(
        monkeypatch):
    from spacr.qt.ai import github_auth as ga
    ga.set_stored_token("ghp_x")

    def _handler(req, t):
        raise _http_error(403, "Forbidden", b'{"message": ""}')

    _stub_urlopen(monkeypatch, _handler)
    assert ga.create_issue("o/r", "t", "b") == \
        (False, "GitHub API error 403: Forbidden")


def test_create_issue_error_body_unreadable(monkeypatch):
    """HTTPError whose fp is already consumed must not mask the status."""
    from spacr.qt.ai import github_auth as ga
    ga.set_stored_token("ghp_x")

    class _Unreadable(urllib.error.HTTPError):
        def read(self, *a):
            raise ValueError("stream closed")

    def _handler(req, t):
        hdrs = email.message.Message()
        raise _Unreadable("https://api.github.com/x", 500,
                          "Internal Server Error", hdrs, io.BytesIO(b""))

    _stub_urlopen(monkeypatch, _handler)
    assert ga.create_issue("o/r", "t", "b") == \
        (False, "GitHub API error 500: Internal Server Error")


def test_create_issue_network_timeout(monkeypatch):
    from spacr.qt.ai import github_auth as ga
    ga.set_stored_token("ghp_x")

    def _handler(req, t):
        raise socket.timeout("timed out")

    _stub_urlopen(monkeypatch, _handler)
    ok, err = ga.create_issue("o/r", "t", "b")
    assert ok is False
    assert err.startswith("Failed to reach GitHub: ")
    assert "timed out" in err


def test_create_issue_dns_failure(monkeypatch):
    from spacr.qt.ai import github_auth as ga
    ga.set_stored_token("ghp_x")

    def _handler(req, t):
        raise urllib.error.URLError("[Errno -2] Name or service not known")

    _stub_urlopen(monkeypatch, _handler)
    ok, err = ga.create_issue("o/r", "t", "b")
    assert ok is False
    assert "Name or service not known" in err


def test_create_issue_malformed_success_json_is_reported_not_raised(
        monkeypatch):
    from spacr.qt.ai import github_auth as ga
    ga.set_stored_token("ghp_x")
    _stub_urlopen(monkeypatch, lambda req, t: _Resp(b"<!doctype html>"))
    ok, err = ga.create_issue("o/r", "t", "b")
    assert ok is False
    assert err.startswith("Failed to reach GitHub: ")


def test_create_issue_empty_success_body_is_reported_not_raised(monkeypatch):
    from spacr.qt.ai import github_auth as ga
    ga.set_stored_token("ghp_x")
    _stub_urlopen(monkeypatch, lambda req, t: _Resp(b""))
    ok, err = ga.create_issue("o/r", "t", "b")
    assert ok is False
    assert "Failed to reach GitHub" in err


def test_no_error_path_ever_echoes_the_token(monkeypatch):
    """Whatever goes wrong, the returned message is surfaced in the UI —
    it must never contain the credential."""
    from spacr.qt.ai import github_auth as ga
    tok = "ghp_LEAKCANARY0123456789"
    ga.set_stored_token(tok)

    failures = [
        lambda req, t: (_ for _ in ()).throw(
            _http_error(401, "Unauthorized", b'{"message": "Bad credentials"}')),
        lambda req, t: (_ for _ in ()).throw(
            urllib.error.URLError(f"tried {req.full_url}")),
        lambda req, t: (_ for _ in ()).throw(RuntimeError(str(req.headers))),
        lambda req, t: _Resp(b"not json"),
    ]
    for handler in failures:
        _stub_urlopen(monkeypatch, handler)
        ok, msg = ga.create_issue("o/r", "t", "b")
        assert ok is False
        assert tok not in msg, f"token leaked into {msg!r}"
        assert "<REDACTED>" in msg or "Bearer" not in msg


def test_pat_with_embedded_newline_does_not_leak_via_header_validation(
        monkeypatch):
    """REGRESSION: a PAT pasted from a wrapped terminal keeps an embedded
    newline. ``http.client.putheader`` then raises
    ``ValueError: Invalid header value b'Bearer ghp_…'`` — which used to be
    echoed verbatim into the UI status line (and the log) by create_issue's
    generic error handler."""
    import http.client

    from spacr.qt.ai import github_auth as ga

    tok = "ghp_WRAPPEDPASTE0123456789\nX-Injected: 1"
    ga.set_stored_token(tok)
    # set_stored_token only strips the OUTSIDE, so the newline survives:
    assert "\n" in ga.get_stored_token()

    def _handler(req, t):
        # Reproduce exactly what http.client does with this header.
        conn = http.client.HTTPConnection("api.github.com")
        conn._HTTPConnection__state = http.client._CS_REQ_STARTED
        conn._buffer = []
        conn.putheader("Authorization", req.headers["Authorization"])
        raise AssertionError("putheader should have rejected the value")

    _stub_urlopen(monkeypatch, _handler)
    ok, msg = ga.create_issue("o/r", "t", "b")
    assert ok is False
    assert "Invalid header value" in msg      # the real failure is preserved
    assert "ghp_WRAPPEDPASTE0123456789" not in msg
    assert "<REDACTED>" in msg


def test_scrub_replaces_every_occurrence_and_is_a_noop_without_a_token():
    from spacr.qt.ai import github_auth as ga
    msg = "sent ghp_abc then retried with ghp_abc"
    assert ga._scrub(msg, "ghp_abc") == \
        "sent <REDACTED> then retried with <REDACTED>"
    assert ga._scrub(msg, "") == msg          # no token -> untouched
    assert ga._scrub("clean message", "ghp_abc") == "clean message"


def test_scrub_handles_escaped_and_unanticipated_renderings():
    from spacr.qt.ai import github_auth as ga
    tok = "ghp_line1\nline2"
    # repr-style rendering, as http.client produces
    escaped = r"Invalid header value b'Bearer ghp_line1\nline2'"
    assert "ghp_line1" not in ga._scrub(escaped, tok)
    # a token we never saw still gets caught by the Bearer backstop
    other = "proxy rejected 'Authorization: Bearer ghp_someoneelses_token'"
    out = ga._scrub(other, "")
    assert "ghp_someoneelses_token" not in out
    assert "Bearer <REDACTED>" in out


# ---------------------------------------------------------------------------
# keys.py — the legacy inert stub
# ---------------------------------------------------------------------------

def test_legacy_keys_module_is_fully_inert():
    from spacr.qt.ai import keys
    assert keys.SERVICE_NAME == "spacr-qt-ai"
    for name in ("claude", "codex", "gemini", "anything"):
        assert keys.get_key(name) is None
        assert keys.set_key(name, "sk-secret") is False
        assert keys.delete_key(name) is None
        assert keys.source_of(name) == "n/a (uses vendor CLI login)"


# ---------------------------------------------------------------------------
# settings.py — the two boolean toggles
# ---------------------------------------------------------------------------

@pytest.fixture()
def ai_store(monkeypatch, tmp_path):
    from PySide6.QtCore import QSettings
    from spacr.qt.ai import settings as s
    store = QSettings(str(tmp_path / "ai.ini"), QSettings.IniFormat)
    monkeypatch.setattr(s, "_settings", lambda: store)
    return store


def test_auto_file_issues_defaults_off_and_roundtrips(ai_store):
    from spacr.qt.ai import settings as s
    assert s.get_auto_file_issues() is False
    s.set_auto_file_issues(True)
    assert s.get_auto_file_issues() is True
    s.set_auto_file_issues(False)
    assert s.get_auto_file_issues() is False


def test_route_errors_defaults_on_and_roundtrips(ai_store):
    from spacr.qt.ai import settings as s
    assert s.get_route_errors_through_ai() is True
    s.set_route_errors_through_ai(False)
    assert s.get_route_errors_through_ai() is False
    s.set_route_errors_through_ai(True)
    assert s.get_route_errors_through_ai() is True


@pytest.mark.parametrize("raw,expected", [
    ("true", True), ("True", True), ("1", True), ("yes", True), ("YES", True),
    ("false", False), ("0", False), ("no", False), ("", False),
    ("garbage", False),
])
def test_boolean_settings_parse_ini_strings(ai_store, raw, expected):
    """The INI backend hands booleans back as strings — both toggles must
    survive the round trip through a *.ini* file, not just the registry."""
    from spacr.qt.ai import settings as s
    ai_store.setValue("ai/auto_file_issues", raw)
    ai_store.setValue("ai/route_errors_through_ai", raw)
    assert s.get_auto_file_issues() is expected
    assert s.get_route_errors_through_ai() is expected


def test_set_toggles_coerce_truthy_values(ai_store):
    from spacr.qt.ai import settings as s
    s.set_auto_file_issues("non-empty string")
    assert s.get_auto_file_issues() is True
    s.set_route_errors_through_ai(0)
    assert s.get_route_errors_through_ai() is False


def test_provider_args_tracks_the_speed_setting(ai_store):
    from spacr.qt.ai import settings as s
    expected = {
        "fast":     {"claude": ["--model", "haiku"],
                     "codex":  ["--model", "gpt-5-mini"],
                     "gemini": ["--model", "gemini-2.5-flash"]},
        "balanced": {"claude": ["--model", "sonnet"],
                     "codex":  ["--model", "gpt-5"],
                     "gemini": ["--model", "gemini-2.5-pro"]},
        "deep":     {"claude": ["--model", "opus"],
                     "codex":  ["--model", "gpt-5-pro"],
                     "gemini": ["--model", "gemini-2.5-pro"]},
    }
    for speed, per_provider in expected.items():
        s.set_response_speed(speed)
        for provider, args in per_provider.items():
            assert s.provider_args(provider) == args, (speed, provider)


def test_set_response_speed_rejects_an_unknown_label(ai_store):
    from spacr.qt.ai import settings as s
    s.set_response_speed("deep")
    with pytest.raises(ValueError) as exc:
        s.set_response_speed("turbo")
    assert "turbo" in str(exc.value)
    assert "fast" in str(exc.value)          # names the valid choices
    assert s.get_response_speed() == "deep"  # the old value survives


def test_provider_args_returns_a_fresh_mutable_list(ai_store):
    """Callers do `argv += provider_args(...)`; handing out the SPEED_MAP
    tuple itself would let a caller corrupt the table."""
    from spacr.qt.ai import settings as s
    s.set_response_speed("fast")
    args = s.provider_args("claude")
    args.append("--tampered")
    assert s.provider_args("claude") == ["--model", "haiku"]
