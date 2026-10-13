"""Annotate's console has Copy console, and offers File as issue on an error.

Asked for on 2026-09-01. This screen builds its own ConsolePanel rather than
using the generic module screen's, so it had inherited neither control -- the
one pane most likely to be holding a traceback was the one you could not copy
from or file from.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def screen(qtbot):
    from spacr.qt.screens.annotate import AnnotateScreen
    s = AnnotateScreen()
    qtbot.addWidget(s)
    return s


def test_the_console_has_a_copy_button(screen):
    assert screen._btn_copy_console is not None
    assert screen._btn_copy_console.text()


def test_copying_reports_that_it_happened(screen):
    """A clipboard write is silent; a button that looks inert reads as broken."""
    screen._console.append_stdout("hello from the console\n")
    before = screen._btn_copy_console.text()

    screen._on_copy_console()

    assert screen._btn_copy_console.text() != before, (
        "the caption must say the copy happened")


def test_copy_survives_a_console_that_cannot_copy(screen, monkeypatch):
    """A failure to copy must not take the annotation session with it."""
    monkeypatch.setattr(screen._console, "copy_all",
                        lambda: (_ for _ in ()).throw(RuntimeError("nope")))
    notices = []
    monkeypatch.setattr(screen._console, "append_notice",
                        lambda text, **fields: notices.append((text, fields)))

    screen._on_copy_console()          # must not raise

    # SURVIVING IS NOT ENOUGH. A clipboard write is silent, so a copy that
    # failed and said nothing is indistinguishable from one that worked --
    # which is the whole reason this handler reports at all. Assert it
    # reported, and that it carried the cause rather than a bare apology.
    assert len(notices) == 1, f"the failure was swallowed: {notices}"
    text, fields = notices[0]
    assert "Could not copy" in text
    assert "nope" in str(fields.get("detail")), (
        f"the notice does not name what went wrong: {fields}")


def test_the_issue_button_is_hidden_until_something_goes_wrong(screen):
    """A permanently visible report button invites reports with no traceback."""
    assert not screen._btn_file_issue.isVisible()


def test_an_error_offers_the_report_when_the_user_opted_in(screen, monkeypatch):
    from spacr.qt.ai import settings as ai_settings

    monkeypatch.setattr(ai_settings, "get_auto_file_issues", lambda: True)
    screen._console.append_error("Traceback (most recent call last): ...\n")

    assert screen._btn_file_issue.isEnabled(), (
        "an error should reveal the report action for a user who opted in")


def test_an_error_stays_quiet_when_the_user_did_not_opt_in(screen, monkeypatch):
    """Opting in reveals the action; it is never assumed."""
    from spacr.qt.ai import settings as ai_settings

    monkeypatch.setattr(ai_settings, "get_auto_file_issues", lambda: False)
    screen._console.append_error("Traceback (most recent call last): ...\n")

    assert not screen._btn_file_issue.isEnabled()


def test_the_hook_does_not_swallow_the_console_output(screen):
    """The error still has to reach the pane it was written to."""
    screen._console.append_error("a distinctive failure string\n")
    text = screen._console.copy_all()
    assert "a distinctive failure string" in text


def test_pressing_file_as_issue_sends_the_console(screen, qtbot, monkeypatch):
    """PRESS IT. The handler called
    ``file_issue(self, {"screen": "annotate"}, body)`` -- the screen where
    the traceback goes, a dict where the app id goes, the console text where
    the settings go -- so the reporter's first act was
    ``sanitize_path(<AnnotateScreen>)`` and the user got "Could not file the
    issue: 'AnnotateScreen' object has no attribute 'replace'". It was
    hidden behind an opt-in that shipped off; `auto_file_issues` now
    defaults on, so this button is part of the default experience.
    """
    from PySide6.QtWidgets import QDialog

    from spacr.qt.ai import settings as ai_settings
    from spacr.qt.ai.issue_preview import IssuePreviewDialog

    monkeypatch.setattr(ai_settings, "get_auto_file_issues", lambda: True)
    sent = []
    monkeypatch.setattr(IssuePreviewDialog, "exec",
                        lambda dialog: QDialog.Accepted)
    monkeypatch.setattr("spacr.qt.ai.issue_report.submit_report",
                        lambda report: sent.append(report) or "https://x/1")
    monkeypatch.setattr(screen._report_jobs, "_threaded", False)
    screen._console.append_error(
        "Traceback (most recent call last):\n"
        "ValueError: a distinctive annotate failure\n")

    screen._btn_file_issue.click()

    assert len(sent) == 1, f"nothing was submitted: {screen._console.as_text()}"
    assert "a distinctive annotate failure" in sent[0]["body"]
    assert sent[0]["fingerprint"]
    assert "[annotate]" in sent[0]["title"]
    text = screen._console.as_text()
    assert "object has no attribute" not in text
    assert "https://x/1" in text


def test_file_as_issue_refuses_when_reporting_is_set_to_never(
        screen, qtbot, monkeypatch):
    """'never' is a refusal to send, and the button must honour it."""
    from spacr.qt import preferences
    from spacr.qt.ai import settings as ai_settings
    from spacr.qt.ai.issue_preview import IssuePreviewDialog

    monkeypatch.setattr(ai_settings, "get_auto_file_issues", lambda: True)
    monkeypatch.setattr(IssuePreviewDialog, "exec",
                        lambda dialog: pytest.fail("a preview was opened"))
    preferences.set_issue_prompt_mode("never")
    screen._console.append_error("ValueError: boom\n")

    screen._btn_file_issue.click()

    assert "set to 'never'" in screen._console.as_text()
