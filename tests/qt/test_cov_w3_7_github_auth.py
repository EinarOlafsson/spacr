"""Reading a token, and the two API calls the offline suite does not reach.

No test in this file touches the network. Every request is answered by a
stand-in ``urlopen``, which is also the only honest way to assert what spaCR
SENDS -- the URL, the method, the headers and the JSON body are read back off
the ``Request`` object the module built.

The token itself comes from the environment, never from the developer's ``gh``
credential store: ``resolve_token`` falls through to ``gh auth token`` and a
test that let it get that far would be reading a real credential.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings

from spacr.qt.ai import github_auth


@pytest.fixture(autouse=True)
def _a_token_from_the_environment(monkeypatch):
    """One token, from the environment, so no test consults the ``gh`` CLI."""
    monkeypatch.setattr(github_auth, "_EPHEMERAL_TOKEN", "")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_covw37token")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(github_auth, "_gh_cli_token",
                        lambda: pytest.fail(
                            "a test reached the developer's gh credential "
                            "store"))


class _Response:
    """The context manager ``urlopen`` returns, holding one JSON body."""

    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def transport(monkeypatch):
    """Answer every request from a queue, and keep the requests sent."""
    sent = []
    answers = []

    def urlopen(request, timeout=None):
        sent.append(request)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return _Response(answer)

    monkeypatch.setattr(github_auth, "_HTTP_OPEN", urlopen)
    return type("Transport", (), {"sent": sent, "answers": answers})()


# ---------------------------------------------------------------------------
# The token
# ---------------------------------------------------------------------------

def test_a_token_an_older_build_persisted_is_erased_on_the_way_past():
    """spaCR does not store credentials; a build that did is undone here."""
    settings = QSettings(github_auth._ORG, github_auth._APP)
    settings.setValue(github_auth._KEY_TOKEN, "ghp_left_on_disk")
    settings.sync()
    assert settings.contains(github_auth._KEY_TOKEN)

    assert github_auth.get_stored_token() == ""
    assert not QSettings(github_auth._ORG,
                         github_auth._APP).contains(github_auth._KEY_TOKEN)


def test_the_environment_is_where_the_token_comes_from():
    assert github_auth.resolve_token() == ("ghp_covw37token", "env")
    assert github_auth.auth_source() == "env"
    assert github_auth.is_authenticated() is True


# ---------------------------------------------------------------------------
# The process-lifetime write guard
# ---------------------------------------------------------------------------

def test_the_write_guard_refuses_inside_a_test_run(monkeypatch):
    """The old inherited escape hatch is deliberately inert."""
    reason = github_auth._refuse_writes_under_test()
    assert reason and "refusing GitHub network access" in reason

    monkeypatch.setenv("SPACR_ALLOW_GITHUB_WRITES", "1")
    assert github_auth._refuse_writes_under_test() == reason


def test_outside_a_test_run_there_is_nothing_to_refuse(monkeypatch):
    monkeypatch.delenv("SPACR_PYTEST_SESSION", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert github_auth._refuse_writes_under_test() is None


# ---------------------------------------------------------------------------
# Searching for an existing report
# ---------------------------------------------------------------------------

def test_the_search_asks_for_one_open_issue_with_the_fingerprint(transport):
    transport.answers.append({"items": [{"number": 12,
                                         "html_url": "https://x/12"}]})
    searched, issue = github_auth.find_issue_by_fingerprint("o/n", "abc123")
    assert searched is True
    assert issue["number"] == 12

    request = transport.sent[0]
    assert request.full_url.startswith(
        "https://api.github.com/search/issues?q=")
    assert "abc123" in request.full_url
    assert "is%3Aopen" in request.full_url
    assert "per_page=1" in request.full_url
    assert request.get_header("Authorization") == "Bearer ghp_covw37token"


def test_no_match_is_a_completed_search_not_a_failed_one(transport):
    """The difference decides whether the reporter files or comments."""
    transport.answers.append({"items": []})
    assert github_auth.find_issue_by_fingerprint("o/n", "abc") == (True, None)


def test_a_search_that_could_not_run_says_so(transport):
    transport.answers.append(urllib.error.URLError("no route to host"))
    assert github_auth.find_issue_by_fingerprint("o/n", "abc") == (False, None)


def test_without_a_token_there_is_no_search(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(github_auth, "_gh_cli_token", lambda: "")
    monkeypatch.setattr(github_auth, "_HTTP_OPEN",
                        lambda *a, **k: pytest.fail("asked GitHub anonymously"))
    assert github_auth.find_issue_by_fingerprint("o/n", "abc") == (False, None)


# ---------------------------------------------------------------------------
# Commenting on one that already exists
# ---------------------------------------------------------------------------

def test_a_second_occurrence_is_posted_as_a_comment(transport):
    transport.answers.append({"html_url": "https://x/12#issuecomment-1"})
    ok, url = github_auth.comment_on_issue("owner/name", 12, "Seen again.")
    assert ok is True
    assert url == "https://x/12#issuecomment-1"

    request = transport.sent[0]
    assert request.full_url == (
        "https://api.github.com/repos/owner/name/issues/12/comments")
    assert request.get_method() == "POST"
    assert json.loads(request.data.decode("utf-8")) == {"body": "Seen again."}
    assert request.get_header("Content-type") == "application/json"


def test_a_comment_that_reflects_the_token_back_is_scrubbed(transport):
    """``putheader`` echoes the header at us when a PAT holds a newline."""
    transport.answers.append(
        ValueError("Invalid header value b'Bearer ghp_covw37token'"))
    ok, message = github_auth.comment_on_issue("owner/name", 12, "body")
    assert ok is False
    assert "ghp_covw37token" not in message
    assert github_auth.REDACTED in message


def test_no_token_means_no_comment(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(github_auth, "_gh_cli_token", lambda: "")
    monkeypatch.setattr(github_auth, "_HTTP_OPEN",
                        lambda *a, **k: pytest.fail("posted anonymously"))
    ok, message = github_auth.comment_on_issue("owner/name", 12, "body")
    assert ok is False
    assert "Not signed in" in message
