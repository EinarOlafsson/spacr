"""The two GitHub calls the issue reporter makes before it files: search and comment.

No test here touches the network. Every request is answered by a stand-in
``urlopen`` installed on ``github_auth._HTTP_OPEN``, which is also how the
tests read back exactly what spaCR SENDS: the URL, the method, the headers,
the JSON body and the timeout. The token comes from the environment; the
``gh`` credential store fails the test if it is consulted.
"""
from __future__ import annotations

import json
import urllib.error

import pytest

pytest.importorskip("PySide6")

from spacr.qt.ai import github_auth  # noqa: E402

pytestmark = pytest.mark.qt

TOKEN = "ghp_covw37token"
REFUSAL = "refusing GitHub network access from inside a test run"
NOT_SIGNED_IN = "Not signed in to GitHub (no token available)."
COMMON_HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-github-api-version": "2022-11-28",
    "User-agent": "spacr",
}


@pytest.fixture(autouse=True)
def _a_token_from_the_environment(monkeypatch):
    """One token, from the environment, so no test consults the ``gh`` CLI."""
    monkeypatch.setattr(github_auth, "_EPHEMERAL_TOKEN", "")
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(github_auth, "_gh_cli_token", lambda: pytest.fail(
        "a test reached the developer's gh credential store"))


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


class _Transport:
    """Answers requests from a queue and keeps every request and timeout."""

    def __init__(self):
        self.sent = []
        self.timeouts = []
        self.answers = []

    def __call__(self, request, timeout=None):
        self.sent.append(request)
        self.timeouts.append(timeout)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return _Response(answer)


def _install(monkeypatch, *answers):
    transport = _Transport()
    transport.answers.extend(answers)
    monkeypatch.setattr(github_auth, "_HTTP_OPEN", transport)
    return transport


def _signed_out(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(github_auth, "_gh_cli_token", lambda: "")
    monkeypatch.setattr(github_auth, "_HTTP_OPEN", lambda *a, **k: pytest.fail(
        "asked GitHub anonymously"))


def _real_transport_without_credentials(monkeypatch):
    """The real transport is in place; credential discovery fails the test."""
    assert github_auth._HTTP_OPEN is github_auth._REAL_HTTP_OPEN
    monkeypatch.setattr(github_auth, "resolve_token", lambda: pytest.fail(
        "a refused test call consulted the developer credential"))


SEARCH_URL = ("https://api.github.com/search/issues?q=repo%3Ao/n%20is%3Aissue"
              "%20is%3Aopen%20in%3Abody%20%22abc123%22&per_page=1")


def test_find_issue_by_fingerprint_reports_whether_the_search_ran(
        subtests, monkeypatch):
    """A completed search (match or not) is True; no token, refusal or error is False."""
    issue = {"number": 12, "html_url": "https://x/12"}
    answered = [
        ("a match is the first item", {"items": [issue, {"number": 13}]},
         (True, issue)),
        ("an empty item list is a completed search", {"items": []},
         (True, None)),
        ("a body with no items key is a completed search", {},
         (True, None)),
        ("a null item list is a completed search", {"items": None},
         (True, None)),
        ("a search that could not run", urllib.error.URLError(
            "no route to host"), (False, None)),
        ("any other failure is a search that did not run",
         ValueError("bad"), (False, None)),
    ]
    for case_id, answer, expected in answered:
        with subtests.test(msg=case_id), monkeypatch.context() as patch:
            transport = _install(patch, answer)
            assert github_auth.find_issue_by_fingerprint(
                "o/n", "abc123") == expected
            [request] = transport.sent
            assert request.full_url == SEARCH_URL
            assert request.get_method() == "GET"
            assert request.data is None
            assert dict(request.header_items()) == COMMON_HEADERS
            assert transport.timeouts == [20]

    with subtests.test(msg="without a token there is no search"), \
            monkeypatch.context() as patch:
        _signed_out(patch)
        assert github_auth.find_issue_by_fingerprint("o/n", "abc") == (
            False, None)

    with subtests.test(msg="the real transport is refused before the token"), \
            monkeypatch.context() as patch:
        _real_transport_without_credentials(patch)
        assert github_auth.find_issue_by_fingerprint("owner/name", "abc123") \
            == (False, None)


def test_comment_on_issue_posts_one_json_comment_or_says_why_not(
        subtests, monkeypatch):
    """POST to the issue's comments; errors come back scrubbed of the token."""
    answered = [
        ("a posted comment returns its url",
         {"html_url": "https://x/12#issuecomment-1"},
         (True, "https://x/12#issuecomment-1")),
        ("a reply with no url is still a posted comment", {}, (True, "")),
        ("a failure that echoes the token is scrubbed",
         ValueError(f"Invalid header value b'Bearer {TOKEN}'"),
         (False, f"Invalid header value b'Bearer {github_auth.REDACTED}'")),
        ("a failure without the token is reported as it stands",
         urllib.error.URLError("no route to host"),
         (False, "<urlopen error no route to host>")),
    ]
    for case_id, answer, expected in answered:
        with subtests.test(msg=case_id), monkeypatch.context() as patch:
            transport = _install(patch, answer)
            assert github_auth.comment_on_issue(
                "owner/name", 12, "Seen again.") == expected
            [request] = transport.sent
            assert request.full_url == (
                "https://api.github.com/repos/owner/name/issues/12/comments")
            assert request.get_method() == "POST"
            assert json.loads(request.data.decode("utf-8")) == {
                "body": "Seen again."}
            assert dict(request.header_items()) == {
                **COMMON_HEADERS, "Content-type": "application/json"}
            assert transport.timeouts == [20]

    with subtests.test(msg="no token means no comment"), \
            monkeypatch.context() as patch:
        _signed_out(patch)
        assert github_auth.comment_on_issue("owner/name", 12, "body") == (
            False, NOT_SIGNED_IN)

    with subtests.test(msg="the real transport is refused at both writes"), \
            monkeypatch.context() as patch:
        _real_transport_without_credentials(patch)
        assert github_auth.comment_on_issue(
            "owner/name", 114, "Seen again.") == (False, REFUSAL)
        assert github_auth.create_issue("owner/name", "title", "body") == (
            False, REFUSAL)
