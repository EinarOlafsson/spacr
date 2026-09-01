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
    screen._on_copy_console()          # must not raise


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
