"""Offline tests for ``spacr.qt.ai.issue_report``.

The issue body this module builds is posted to a PUBLIC repository, so
the tests below are mostly about what must NOT be in it: absolute paths,
database filenames and — the part that used to leak — API keys and
access tokens.

No browser is ever opened and no HTTP request is ever made: the
``webbrowser`` and ``github_auth`` boundaries are both stubbed.
"""
from __future__ import annotations

import urllib.parse
import webbrowser
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def no_browser(monkeypatch):
    """Hard-fail if any test actually tries to open a browser."""
    opened = []

    def _open(url, new=0, autoraise=True):
        opened.append(url)
        return True

    monkeypatch.setattr(webbrowser, "open", _open)
    return opened


# ---------------------------------------------------------------------------
# redact_secrets — credentials must never reach a public issue
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("secret", [
    "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
    "gho_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
    "ghs_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
    "github_pat_11ABCDEFG0abcdefghijkl_XYZ0123456789",
    "sk-ant-api03-AAAABBBBCCCCDDDD",
    "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWX",
    "AIzaSyD-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123",
    "xoxb-1234567890-abcdefghij",
])
def test_redact_secrets_removes_known_credential_shapes(secret):
    from spacr.qt.ai import issue_report as ir
    text = f"boom while using {secret} in the request"
    out = ir.redact_secrets(text)
    assert secret not in out
    assert ir.REDACTED in out
    assert out.startswith("boom while using ")
    assert out.endswith(" in the request")


def test_redact_secrets_keeps_the_bearer_scheme_but_drops_the_value():
    from spacr.qt.ai import issue_report as ir
    out = ir.redact_secrets(
        "headers={'Authorization': 'Bearer aVeryLongOpaqueValue123'}")
    assert "aVeryLongOpaqueValue123" not in out
    assert "Bearer <REDACTED>" in out


@pytest.mark.parametrize("line,secret", [
    ("api_key = 'hunter2hunter2'", "hunter2hunter2"),
    ("API-KEY: hunter2hunter2", "hunter2hunter2"),
    ('GITHUB_TOKEN="tok_abcdefghij"', "tok_abcdefghij"),
    ("db_password=SuperSecret99", "SuperSecret99"),
    ("client_secret : abcdefghijkl", "abcdefghijkl"),
])
def test_redact_secrets_strips_assignment_style_credentials(line, secret):
    from spacr.qt.ai import issue_report as ir
    out = ir.redact_secrets(line)
    assert secret not in out
    assert ir.REDACTED in out


def test_redact_secrets_leaves_ordinary_text_alone():
    from spacr.qt.ai import issue_report as ir
    text = ("ValueError: channels must be a list of ints, got None\n"
            "  File \"spacr/core.py\", line 88, in preprocess_generate_masks")
    assert ir.redact_secrets(text) == text
    assert ir.redact_secrets("") == ""
    assert ir.redact_secrets(None) is None


def test_sanitize_path_redacts_secrets_as_well_as_paths(monkeypatch, tmp_path):
    from spacr.qt.ai import issue_report as ir
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    text = (f"loaded {tmp_path}/exp/plate1.db with "
            "token=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345")
    out = ir.sanitize_path(text)
    assert str(tmp_path) not in out
    assert "<DB>" in out
    assert "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345" not in out


def test_sanitize_settings_drops_values_of_secret_named_keys():
    from spacr.qt.ai import issue_report as ir
    out = ir.sanitize_settings({
        "src": "/data/plate1",
        "api_key": "anything-at-all",
        "GITHUB_TOKEN": "short",              # too short for any pattern
        "user_password": "p",                 # 1 char — still dropped
        "n_jobs": 8,
    })
    assert out["api_key"] == ir.REDACTED
    assert out["GITHUB_TOKEN"] == ir.REDACTED
    assert out["user_password"] == ir.REDACTED
    assert out["src"] == "/data/plate1"       # a normal path survives
    assert out["n_jobs"] == 8


def test_sanitize_settings_handles_non_string_keys_and_none_input():
    from spacr.qt.ai import issue_report as ir
    assert ir.sanitize_settings(None) == {}
    assert ir.sanitize_settings({}) == {}
    out = ir.sanitize_settings({7: "seven", None: [1, "a"], "flag": True})
    assert out == {7: "seven", None: [1, "a"], "flag": True}


def test_build_report_body_carries_no_credentials(monkeypatch, tmp_path):
    """The end-to-end guarantee: nothing credential-shaped survives into
    the body that gets posted to a public repo."""
    from spacr.qt.ai import issue_report as ir
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

    tb = (
        'Traceback (most recent call last):\n'
        '  File "spacr/qt/ai/github_auth.py", line 137, in create_issue\n'
        "    headers={'Authorization': 'Bearer "
        "ghp_AAAABBBBCCCCDDDDEEEEFFFF1234'}\n"
        'urllib.error.HTTPError: HTTP Error 401: Unauthorized'
    )
    log = tmp_path / "spacr.log"
    log.write_text("INFO starting\nDEBUG ANTHROPIC_API_KEY="
                   "sk-ant-api03-LOGLEAK000111222\n")
    monkeypatch.setattr(ir, "log_tail",
                        lambda *a, **k: ir.sanitize_path(log.read_text()))

    report = ir.build_report(
        tb, active_app="mask",
        settings={"src": f"{tmp_path}/plates",
                  "openai_api_key": "sk-proj-SETTINGSLEAK00000000"},
    )
    body = report["body"]
    for secret in ("ghp_AAAABBBBCCCCDDDDEEEEFFFF1234",
                   "sk-ant-api03-LOGLEAK000111222",
                   "sk-proj-SETTINGSLEAK00000000"):
        assert secret not in body, f"{secret} leaked into the issue body"
    # The useful signal is still there
    assert "HTTPError" in body
    assert "Bearer <REDACTED>" in body
    assert "### Traceback" in body


def test_build_report_title_never_carries_a_credential():
    from spacr.qt.ai import issue_report as ir
    tb = "RuntimeError: bad token ghp_TITLELEAK01234567890123"
    title = ir.build_report(tb, include_log_tail=False)["title"]
    assert "ghp_TITLELEAK01234567890123" not in title
    assert "RuntimeError" in title


# ---------------------------------------------------------------------------
# Title / fingerprint
# ---------------------------------------------------------------------------

_TB = ('Traceback (most recent call last):\n'
       '  File "spacr/core.py", line 42, in run\n'
       '    boom()\n'
       'ValueError: channels must be a list, got None')


def test_traceback_hash_is_deterministic_and_six_hex_chars():
    from spacr.qt.ai import issue_report as ir
    h = ir._traceback_hash(_TB)
    assert h == ir._traceback_hash(_TB)
    assert len(h) == 6
    assert all(c in "0123456789abcdef" for c in h)


def test_traceback_hash_survives_line_number_churn():
    """REGRESSION: an unrelated edit above the failure shifts every line
    number. The same bug must keep the same fingerprint, or every run
    files a fresh duplicate issue."""
    from spacr.qt.ai import issue_report as ir
    shifted = _TB.replace("line 42", "line 57")
    assert ir._traceback_hash(shifted) == ir._traceback_hash(_TB)


def test_traceback_hash_separates_different_exception_types():
    """REGRESSION: two genuinely different bugs raised from the same frame
    used to collapse onto one fingerprint and be filed as one issue."""
    from spacr.qt.ai import issue_report as ir
    other = _TB.replace("ValueError: channels must be a list, got None",
                        "TypeError: unsupported operand")
    assert ir._traceback_hash(other) != ir._traceback_hash(_TB)


def test_traceback_hash_ignores_the_volatile_exception_message():
    """The message routinely embeds a filename or plate id — including it
    would fork the fingerprint on every run of the same bug."""
    from spacr.qt.ai import issue_report as ir
    same_bug = _TB.replace("channels must be a list, got None",
                           "channels must be a list, got 'plate_042'")
    assert ir._traceback_hash(same_bug) == ir._traceback_hash(_TB)


def test_traceback_hash_separates_different_call_sites():
    from spacr.qt.ai import issue_report as ir
    elsewhere = _TB.replace("spacr/core.py", "spacr/measure.py")
    assert ir._traceback_hash(elsewhere) != ir._traceback_hash(_TB)
    other_func = _TB.replace("in run", "in preprocess")
    assert ir._traceback_hash(other_func) != ir._traceback_hash(_TB)


def test_traceback_hash_ignores_blank_lines():
    from spacr.qt.ai import issue_report as ir
    padded = _TB.replace('    boom()\n', '    boom()\n\n   \n')
    assert ir._traceback_hash(padded) == ir._traceback_hash(_TB)


def test_traceback_hash_ignores_the_source_echo_lines():
    from spacr.qt.ai import issue_report as ir
    reformatted = _TB.replace("    boom()", "    boom(a, b)  # reformatted")
    assert ir._traceback_hash(reformatted) == ir._traceback_hash(_TB)


def test_traceback_hash_falls_back_to_whole_text_without_file_lines():
    from spacr.qt.ai import issue_report as ir
    h = ir._traceback_hash("just a message")
    assert len(h) == 6
    assert h != ir._traceback_hash("a different message")
    assert h == ir._traceback_hash("just a message")


def test_build_report_title_defaults_when_traceback_is_blank():
    from spacr.qt.ai import issue_report as ir
    r = ir.build_report("", include_log_tail=False)
    assert "Runtime error" in r["title"]
    assert r["title"].startswith("[auto ")
    assert "`unknown`" in r["body"]      # active_app placeholder


def test_build_report_title_is_capped_at_120_chars():
    from spacr.qt.ai import issue_report as ir
    tb = "SomeVeryLongCustomError: " + "y" * 400
    r = ir.build_report(tb, active_app="measure", include_log_tail=False)
    assert len(r["title"]) <= 120
    assert "[measure]" in r["title"]


def test_build_report_title_skips_indented_traceback_frames():
    from spacr.qt.ai import issue_report as ir
    tb = ('Traceback (most recent call last):\n'
          '  File "spacr/io.py", line 9, in read\n'
          '    open(p)\n'
          'FileNotFoundError: no such file\n'
          '  (indented trailing noise)\n')
    r = ir.build_report(tb, include_log_tail=False)
    assert "FileNotFoundError: no such file" in r["title"]


def test_build_report_without_log_tail_omits_the_section(monkeypatch):
    from spacr.qt.ai import issue_report as ir
    monkeypatch.setattr(ir, "log_tail",
                        lambda *a, **k: "should never be called")
    body = ir.build_report("E: x", include_log_tail=False)["body"]
    assert "Recent log lines" not in body


def test_build_report_omits_log_section_when_tail_is_empty(monkeypatch):
    from spacr.qt.ai import issue_report as ir
    monkeypatch.setattr(ir, "log_tail", lambda *a, **k: "")
    body = ir.build_report("E: x", include_log_tail=True)["body"]
    assert "Recent log lines" not in body


def test_build_report_includes_log_section_when_tail_is_present(monkeypatch):
    from spacr.qt.ai import issue_report as ir
    monkeypatch.setattr(ir, "log_tail", lambda *a, **k: "line A\nline B\n")
    body = ir.build_report("E: x", include_log_tail=True)["body"]
    assert "<details><summary>Recent log lines</summary>" in body
    assert "line A\nline B" in body
    assert body.count("</details>") == 1


# ---------------------------------------------------------------------------
# log_tail
# ---------------------------------------------------------------------------

def test_log_tail_returns_only_the_last_n_lines(tmp_path, monkeypatch):
    from spacr.qt.ai import issue_report as ir
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    p = tmp_path / "spacr.log"
    p.write_text("".join(f"line {i}\n" for i in range(100)))
    out = ir.log_tail(n_lines=5, log_path=p)
    assert out == "line 95\nline 96\nline 97\nline 98\nline 99\n"


def test_log_tail_sanitizes_home_paths_and_secrets(tmp_path, monkeypatch):
    from spacr.qt.ai import issue_report as ir
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    p = tmp_path / "spacr.log"
    p.write_text(f"opened {home}/data/run.db\n"
                 "auth token=ghp_LOGTAILLEAK0123456789\n")
    out = ir.log_tail(log_path=p)
    assert str(home) not in out
    assert "<DB>" in out
    assert "ghp_LOGTAILLEAK0123456789" not in out


def test_log_tail_returns_empty_for_a_missing_file(tmp_path):
    from spacr.qt.ai import issue_report as ir
    assert ir.log_tail(log_path=tmp_path / "nope.log") == ""


def test_log_tail_resolves_the_default_path_from_logging_util(tmp_path,
                                                               monkeypatch):
    from spacr.qt import logging_util
    from spacr.qt.ai import issue_report as ir

    p = tmp_path / "default.log"
    p.write_text("from the default location\n")
    monkeypatch.setattr(logging_util, "log_path", lambda: p)
    assert ir.log_tail() == "from the default location\n"


def test_log_tail_empty_when_the_default_path_cannot_be_resolved(monkeypatch):
    from spacr.qt import logging_util
    from spacr.qt.ai import issue_report as ir

    def _boom():
        raise RuntimeError("no log configured")

    monkeypatch.setattr(logging_util, "log_path", _boom)
    assert ir.log_tail() == ""


def test_log_tail_tolerates_undecodable_bytes(tmp_path, monkeypatch):
    from spacr.qt.ai import issue_report as ir
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    p = tmp_path / "spacr.log"
    p.write_bytes(b"good line\n\xff\xfe not utf8\n")
    out = ir.log_tail(log_path=p)
    assert "good line" in out
    assert "not utf8" in out


# ---------------------------------------------------------------------------
# Environment block
# ---------------------------------------------------------------------------

def test_env_lines_report_real_versions():
    import sys
    from spacr.qt.ai import issue_report as ir
    lines = ir._env_lines()
    assert len(lines) == 6
    joined = "\n".join(lines)
    assert f"**Python**: {sys.version.split()[0]}" in joined
    for pkg in ("spaCR", "Platform", "PySide6", "torch", "cellpose"):
        assert f"**{pkg}**" in joined


def test_env_lines_say_unknown_when_spacr_version_is_unimportable(monkeypatch):
    import builtins
    from spacr.qt.ai import issue_report as ir

    real_import = builtins.__import__

    def _fake(name, *a, **k):
        if name == "spacr.version":
            raise ImportError("no version module")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _fake)
    assert "- **spaCR**: unknown" in ir._env_lines()


def test_optional_version_reports_not_installed_for_absent_package():
    from spacr.qt.ai import issue_report as ir
    assert ir._optional_version("definitely-not-a-real-package-xyz") == \
        "not installed"


def test_optional_version_reports_a_real_package():
    from spacr.qt.ai import issue_report as ir
    v = ir._optional_version("PySide6")
    assert v != "not installed"
    assert v[0].isdigit()


# ---------------------------------------------------------------------------
# issue_url
# ---------------------------------------------------------------------------

def test_issue_url_round_trips_special_characters():
    from spacr.qt.ai import issue_report as ir
    title = "crash: a&b?c=d #1 100% sure"
    body = "```py\nx = a & b\n```\n<details>"
    url = ir.issue_url(title, body)
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert q["title"] == [title]
    assert q["body"] == [body]


def test_issue_url_honours_custom_repo_and_label():
    from spacr.qt.ai import issue_report as ir
    url = ir.issue_url("t", "b", label="triage", repo="acme/widget")
    assert url.startswith("https://github.com/acme/widget/issues/new?")
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert q["labels"] == ["triage"]


def test_issue_url_keeps_the_head_of_the_body_when_truncating():
    from spacr.qt.ai import issue_report as ir
    body = "### Traceback\nZeroDivisionError here\n" + "z" * 20000
    url = ir.issue_url("t", body)
    assert len(url) <= ir.MAX_URL_LEN + 200
    decoded = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["body"][0]
    assert decoded.startswith("### Traceback\nZeroDivisionError here")
    assert "report truncated to fit GitHub URL limit" in decoded
    assert "~/.spacr/logs/spacr.log" in decoded


def test_issue_url_does_not_truncate_a_small_body():
    from spacr.qt.ai import issue_report as ir
    url = ir.issue_url("t", "small body")
    decoded = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["body"][0]
    assert decoded == "small body"
    assert "truncated" not in decoded


def test_issue_url_truncation_accounts_for_a_long_title():
    """A long title eats into the body budget — the URL must still fit."""
    from spacr.qt.ai import issue_report as ir
    url = ir.issue_url("T" * 119, "b" * 20000)
    assert len(url) <= ir.MAX_URL_LEN + 200


# ---------------------------------------------------------------------------
# Browser opener
# ---------------------------------------------------------------------------

def test_open_issue_in_browser_forwards_url_and_new_tab_flag(monkeypatch):
    from spacr.qt.ai import issue_report as ir
    seen = {}

    def _open(url, new=0, autoraise=True):
        seen["url"] = url
        seen["new"] = new
        return True

    monkeypatch.setattr(webbrowser, "open", _open)
    assert ir.open_issue_in_browser("https://example.test/x") is True
    assert seen == {"url": "https://example.test/x", "new": 2}


def test_open_issue_in_browser_returns_false_when_webbrowser_refuses(
        monkeypatch):
    from spacr.qt.ai import issue_report as ir
    monkeypatch.setattr(webbrowser, "open", lambda *a, **k: False)
    assert ir.open_issue_in_browser("https://example.test/x") is False


def test_open_issue_in_browser_swallows_backend_errors(monkeypatch):
    from spacr.qt.ai import issue_report as ir

    def _boom(*a, **k):
        raise webbrowser.Error("no runnable browser")

    monkeypatch.setattr(webbrowser, "open", _boom)
    assert ir.open_issue_in_browser("https://example.test/x") is False


# ---------------------------------------------------------------------------
# file_issue — API path vs browser fallback
# ---------------------------------------------------------------------------

def _patch_auth(monkeypatch, *, authed, create=None):
    from spacr.qt.ai import github_auth
    monkeypatch.setattr(github_auth, "is_authenticated", lambda: authed)
    if create is not None:
        monkeypatch.setattr(github_auth, "create_issue", create)
    else:
        def _never(*a, **k):
            raise AssertionError("create_issue must not run when signed out")
        monkeypatch.setattr(github_auth, "create_issue", _never)


def test_file_issue_posts_via_api_when_signed_in(monkeypatch, no_browser):
    from spacr.qt.ai import issue_report as ir
    seen = {}

    def _create(repo, title, body, labels=None):
        seen.update(repo=repo, title=title, body=body, labels=labels)
        return True, "https://github.com/EinarOlafsson/spacr/issues/123"

    _patch_auth(monkeypatch, authed=True, create=_create)
    url = ir.file_issue("ValueError: boom", active_app="measure",
                        settings={"src": "/data/p1"})
    assert url == "https://github.com/EinarOlafsson/spacr/issues/123"
    assert no_browser == []                       # no browser round-trip
    assert seen["repo"] == ir.REPO
    assert seen["labels"] == [ir.ISSUE_LABEL]
    assert "[measure]" in seen["title"]
    assert "ValueError: boom" in seen["body"]


def test_file_issue_falls_back_to_browser_when_api_call_fails(monkeypatch,
                                                               no_browser):
    from spacr.qt.ai import issue_report as ir
    _patch_auth(monkeypatch, authed=True,
                create=lambda *a, **k: (False, "GitHub API error 401: nope"))
    url = ir.file_issue("ValueError: boom")
    assert url.startswith("https://github.com/EinarOlafsson/spacr/issues/new?")
    assert no_browser == [url]


def test_file_issue_falls_back_when_api_returns_ok_but_no_url(monkeypatch,
                                                               no_browser):
    from spacr.qt.ai import issue_report as ir
    _patch_auth(monkeypatch, authed=True, create=lambda *a, **k: (True, ""))
    url = ir.file_issue("ValueError: boom")
    assert url.startswith("https://github.com/EinarOlafsson/spacr/issues/new?")
    assert no_browser == [url]


def test_file_issue_falls_back_when_github_auth_explodes(monkeypatch,
                                                          no_browser):
    """A broken/absent github_auth must degrade to the browser flow, not
    take the whole error-reporting path down with it."""
    from spacr.qt.ai import github_auth
    from spacr.qt.ai import issue_report as ir

    def _boom():
        raise RuntimeError("QSettings unavailable")

    monkeypatch.setattr(github_auth, "is_authenticated", _boom)
    url = ir.file_issue("ValueError: boom", active_app="mask")
    assert url.startswith("https://github.com/EinarOlafsson/spacr/issues/new?")
    assert no_browser == [url]


def test_file_issue_uses_browser_when_signed_out(monkeypatch, no_browser):
    from spacr.qt.ai import issue_report as ir
    _patch_auth(monkeypatch, authed=False)
    url = ir.file_issue("KeyError: 'plate'", active_app="annotate")
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert q["labels"] == [ir.ISSUE_LABEL]
    assert "[annotate]" in q["title"][0]
    assert "KeyError" in q["body"][0]
    assert no_browser == [url]


def test_file_issue_never_puts_a_credential_in_the_browser_url(monkeypatch,
                                                               no_browser):
    from spacr.qt.ai import issue_report as ir
    _patch_auth(monkeypatch, authed=False)
    url = ir.file_issue(
        "RuntimeError: refused with Bearer ghp_URLLEAK01234567890123",
        settings={"api_key": "sk-ant-api03-URLSETTINGSLEAK"},
    )
    decoded = urllib.parse.unquote_plus(url)
    assert "ghp_URLLEAK01234567890123" not in decoded
    assert "sk-ant-api03-URLSETTINGSLEAK" not in decoded
