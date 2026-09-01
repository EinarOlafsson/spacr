"""The console pairs spaCR AI's answer with the error it answers.

The console holds ONE conversation for a whole session. Handing the bug
reporter "the last AI reply" would therefore file, against one crash, a
confident analysis of a different one -- or of whatever the user last asked
about. The pairing is what makes the attached analysis trustworthy.
"""
from __future__ import annotations

import pytest

from spacr.qt.widgets.console_panel import ConsolePanel

FIRST = "TypeError: '>' not supported between instances of 'str' and 'int'"
SECOND = "ValueError: could not broadcast input array"


@pytest.fixture
def console(qapp, monkeypatch):
    panel = ConsolePanel()
    monkeypatch.setattr(panel, "_current_provider", lambda: object())
    monkeypatch.setattr(panel, "_start_stream", lambda **kw: None)
    monkeypatch.setattr(panel, "_console_context_for_question",
                        lambda text: ("", "no context"))
    return panel


def test_an_answer_is_returned_for_the_error_it_explains(console):
    console.open_error_flow(FIRST, active_app="mask")
    console._on_stream_finished(True, "The diameter arrives as a string.")

    assert console.ai_explanation_of(FIRST) == (
        "The diameter arrives as a string.")


def test_it_is_not_returned_for_a_different_error(console):
    """The failure this pairing prevents."""
    console.open_error_flow(FIRST, active_app="mask")
    console._on_stream_finished(True, "The diameter arrives as a string.")

    assert console.ai_explanation_of(SECOND) == ""


def test_nothing_asked_means_nothing_to_attach(console):
    assert console.ai_explanation_of(FIRST) == ""


def test_an_unfinished_stream_attaches_nothing(console):
    """A report filed while the AI is still typing must carry no analysis
    rather than a half-written one."""
    console.open_error_flow(FIRST, active_app="mask")
    assert console.ai_explanation_of(FIRST) == ""


def test_a_failed_stream_attaches_nothing(console):
    console.open_error_flow(FIRST, active_app="mask")
    console._on_stream_finished(False, "provider refused the request")
    assert console.ai_explanation_of(FIRST) == ""


def test_a_question_of_the_users_own_ends_the_pairing(console):
    """Whatever comes back next answers the QUESTION, not the crash."""
    console.open_error_flow(FIRST, active_app="mask")
    console._on_stream_finished(True, "The diameter arrives as a string.")

    console._send_to_ai("how do I change the outline colour?")
    console._on_stream_finished(True, "Use the outline colour setting.")

    assert console.ai_explanation_of(FIRST) == "", (
        "an answer about outline colours would have been filed as the "
        "analysis of a TypeError")


def test_a_second_error_replaces_the_first(console):
    console.open_error_flow(FIRST, active_app="mask")
    console._on_stream_finished(True, "about the first")
    console.open_error_flow(SECOND, active_app="mask")
    console._on_stream_finished(True, "about the second")

    assert console.ai_explanation_of(SECOND) == "about the second"
    assert console.ai_explanation_of(FIRST) == ""


def test_whitespace_does_not_break_the_match(console):
    """The reporter is handed the traceback text again by the screen, not the
    same object, and log formatting can add a trailing newline."""
    console.open_error_flow(FIRST, active_app="mask")
    console._on_stream_finished(True, "an answer")
    assert console.ai_explanation_of(FIRST + "\n") == "an answer"


def test_the_reporter_actually_asks_for_it():
    """A source check: every behaviour above passes just as happily when the
    bug reporter never calls the accessor."""
    from pathlib import Path

    import spacr.qt.screens.app_screen as app_screen

    source = Path(app_screen.__file__).read_text(encoding="utf-8")
    assert "ai_explanation_of(self._last_error_text)" in source
    assert "ai_response=ai_analysis" in source
