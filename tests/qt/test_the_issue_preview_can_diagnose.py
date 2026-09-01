"""The report dialog can ask spaCR AI about the error it is reporting.

The preview is the moment the user is looking hardest at an error, and the last
moment before it goes somewhere public. A diagnosis is worth having in both
directions: it may save the report entirely, and it makes the report better if
it does not.

WHAT DECIDES WHAT HAPPENS, in order: an analysis already in hand is shown; no
provider means the AI cannot be asked at all; a stream already running means a
second request would queue behind it. The AI *toggle* is deliberately not
consulted -- it governs whether spaCR volunteers an explanation, and pressing a
button named Diagnose is asking for one outright.
"""
from __future__ import annotations

import pytest

from spacr.qt.ai.issue_preview import IssuePreviewDialog

TB = "TypeError: '>' not supported between instances of 'str' and 'int'"
REPORT = {"title": "t", "body": "### Traceback\n```\n" + TB + "\n```",
          "fingerprint": "abc123"}


class _Console:
    """The bits of ConsolePanel the dialog actually touches."""

    def __init__(self, provider=True, busy=False, answer=""):
        self._provider = object() if provider else None
        self._ai_thread = object() if busy else None
        self._answer = answer
        self.asked = []

    def _current_provider(self):
        return self._provider

    def ai_explanation_of(self, traceback_text):
        return self._answer if traceback_text.strip() == TB else ""

    def open_error_flow(self, traceback_text, show_raw=True, **kw):
        self.asked.append(traceback_text)
        self._ai_thread = object()


@pytest.fixture
def told(monkeypatch):
    """Capture what the dialog says instead of opening a message box."""
    seen = []
    monkeypatch.setattr(IssuePreviewDialog, "_tell",
                        lambda self, title, message: seen.append(
                            (title, message)))
    return seen


def _dialog(qapp, console):
    return IssuePreviewDialog(REPORT, None, console=console,
                              traceback_text=TB)


def test_the_button_is_there(qapp):
    dialog = _dialog(qapp, _Console())
    assert dialog.diagnose_btn.text() == "Diagnose"


def test_no_provider_prompts_to_link_an_account(qapp, told):
    dialog = _dialog(qapp, _Console(provider=False))
    dialog._on_diagnose()
    assert told and "No AI account linked" in told[0][0]
    assert "Providers" in told[0][1]


def test_no_provider_does_not_ask(qapp, told):
    console = _Console(provider=False)
    _dialog(qapp, console)._on_diagnose()
    assert console.asked == [], "it asked an AI it has no account for"


def test_a_linked_account_asks_even_with_the_ai_toggle_off(qapp, told):
    """The toggle governs whether spaCR VOLUNTEERS an explanation. Pressing
    Diagnose is asking for one outright."""
    console = _Console(provider=True)
    _dialog(qapp, console)._on_diagnose()
    assert console.asked == [TB]
    assert told == [], f"it should just work, but said {told}"


def test_a_stream_in_flight_asks_the_user_to_wait(qapp, told):
    console = _Console(provider=True, busy=True)
    _dialog(qapp, console)._on_diagnose()
    assert told and "still working" in told[0][0]
    assert console.asked == [], "a second request would queue behind the first"


def test_a_finished_analysis_is_added_to_the_report(qapp, told):
    console = _Console(provider=True, answer="The diameter arrives as a str.")
    dialog = _dialog(qapp, console)
    dialog._on_diagnose()
    body = dialog.body_edit.toPlainText()
    assert "The diameter arrives as a str." in body
    assert "spaCR AI's analysis" in body
    assert console.asked == [], "it already had the answer; asking again wastes"


def test_the_user_is_scrolled_to_it(qapp, told):
    console = _Console(provider=True, answer="Here is why.")
    dialog = _dialog(qapp, console)
    dialog._on_diagnose()
    cursor = dialog.body_edit.textCursor()
    assert "analysis" in cursor.block().text(), (
        "the caret was not moved to the diagnosis")


def test_it_is_marked_unreviewed_here_too(qapp, told):
    dialog = _dialog(qapp, _Console(provider=True, answer="Because."))
    dialog._on_diagnose()
    assert "unreviewed" in dialog.body_edit.toPlainText()


def test_the_analysis_survives_the_strip_toggle(qapp, told):
    """`_refresh_body` rebuilds the box from the SOURCE, so an analysis added
    only to the visible text would vanish the moment the toggle moved."""
    dialog = _dialog(qapp, _Console(provider=True, answer="Because."))
    dialog._on_diagnose()
    dialog.strip_paths.setChecked(False)
    assert "Because." in dialog.body_edit.toPlainText()
    dialog.strip_paths.setChecked(True)
    assert "Because." in dialog.body_edit.toPlainText()


def test_pressing_it_twice_does_not_duplicate_the_section(qapp, told):
    dialog = _dialog(qapp, _Console(provider=True, answer="Because."))
    dialog._on_diagnose()
    dialog._on_diagnose()
    assert dialog.body_edit.toPlainText().count("spaCR AI's analysis") == 1


def test_the_approved_report_carries_it(qapp, told):
    """It must reach the payload, not just the screen."""
    dialog = _dialog(qapp, _Console(provider=True, answer="Because."))
    dialog._on_diagnose()
    assert "Because." in dialog.approved_report()["body"]


def test_an_answer_about_another_error_is_not_used(qapp, told):
    """The console holds one conversation; only an answer to THIS error
    counts."""
    console = _Console(provider=True, answer="about something else")
    dialog = IssuePreviewDialog(REPORT, None, console=console,
                                traceback_text="ValueError: different")
    dialog._on_diagnose()
    assert "about something else" not in dialog.body_edit.toPlainText()
    assert console.asked == ["ValueError: different"]


def test_no_console_says_so_rather_than_crashing(qapp, told):
    dialog = IssuePreviewDialog(REPORT, None, console=None, traceback_text=TB)
    dialog._on_diagnose()
    assert told and "not available" in told[0][1]


def test_the_screen_hands_over_the_console_and_traceback():
    """A source check: the button is inert if the screen does not pass them."""
    from pathlib import Path

    import spacr.qt.screens.app_screen as app_screen

    source = Path(app_screen.__file__).read_text(encoding="utf-8")
    assert "console=self._console" in source
    assert "traceback_text=self._last_error_text" in source
