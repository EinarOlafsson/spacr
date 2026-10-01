"""An error report is built and filed even when the machine will not say who it is.

Pinned behaviour of :mod:`spacr.qt.ai.issue_report`:

* each of the four identity sources (login name, home folder, host name,
  platform node) may raise; the others still feed the redaction, and when
  all four fail the report simply has nothing to redact instead of failing;
* a traceback frame line that names its file without quotes still has its
  line number dropped from the fingerprint, so the same crash keeps one
  fingerprint after an edit moves it;
* automatic filing that breaks while resolving the sign-in reports
  ``FAILED`` with the error's type and message, and never raises.
"""
from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("PySide6")

from spacr.qt.ai import github_auth, issue_report  # noqa: E402


def _boom(*_a, **_k):
    raise OSError("no passwd entry")


def test_a_failing_login_lookup_still_redacts_the_home_folder_name(
        monkeypatch):
    monkeypatch.setattr("getpass.getuser", _boom)
    monkeypatch.setenv("HOME", "/home/marguerite")
    monkeypatch.setattr("socket.gethostname", _boom)
    monkeypatch.setattr("platform.node", _boom)

    words = issue_report._identity_words()

    assert words == [("marguerite", issue_report.USER_PLACEHOLDER)]
    assert issue_report.redact_identity(
        "owner marguerite") == f"owner {issue_report.USER_PLACEHOLDER}"


def test_every_identity_source_failing_leaves_the_report_untouched(
        monkeypatch):
    with monkeypatch.context() as m:
        m.setattr("getpass.getuser", _boom)
        m.setattr(pathlib.Path, "home", classmethod(lambda cls: _boom()))
        m.setattr("socket.gethostname", _boom)
        m.setattr("platform.node", _boom)
        words = issue_report._identity_words()
        text = issue_report.redact_identity("owner marguerite on labhost")

    assert words == []
    assert text == "owner marguerite on labhost"


def test_a_host_lookup_failure_falls_back_to_the_platform_node(monkeypatch):
    monkeypatch.setattr("getpass.getuser", lambda: "marguerite")
    monkeypatch.setenv("HOME", "/home/marguerite")
    monkeypatch.setattr("socket.gethostname", _boom)
    monkeypatch.setattr("platform.node", lambda: "confocal-7.lab.example")

    words = dict(issue_report._identity_words())

    assert words["confocal-7"] == issue_report.HOST_PLACEHOLDER
    assert words["confocal-7.lab.example"] == issue_report.HOST_PLACEHOLDER
    assert words["marguerite"] == issue_report.USER_PLACEHOLDER


def test_an_unquoted_frame_line_keeps_one_fingerprint_across_line_moves():
    before = ("Traceback (most recent call last):\n"
              "  File plugin.py, line 12, in run\n"
              "ValueError: bad plate\n")
    after = before.replace("line 12", "line 97")
    other = before.replace("in run", "in stop")

    assert issue_report.fingerprint_of(before) == (
        issue_report.fingerprint_of(after))
    assert issue_report.fingerprint_of(before) != (
        issue_report.fingerprint_of(other))


def test_a_sign_in_that_breaks_reports_failed_instead_of_raising(
        monkeypatch):
    monkeypatch.setattr(github_auth, "_HTTP_OPEN",
                        lambda *a, **k: pytest.fail("nothing is sent"))

    def _locked():
        raise RuntimeError("keyring locked")

    monkeypatch.setattr(github_auth, "resolve_token", _locked)

    outcome = issue_report.file_without_review(
        {"title": "t", "body": "b", "fingerprint": "abc123"})

    assert outcome == {"status": issue_report.FAILED,
                       "detail": "RuntimeError: keyring locked"}
