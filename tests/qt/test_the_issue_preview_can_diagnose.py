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


#: Dialogs built by a test, so they can be destroyed when it ends.
#:
#: An IssuePreviewDialog owns the Diagnose poll timer and is parented to
#: nothing here, so left alone it was freed whenever the collector next ran --
#: which landed inside a LATER test's `qapp.exec()`. That is how a test in this
#: file made `tests/qt/test_track_previews.py` fail while passing itself.
_BUILT = []


def _dialog(qapp, console):
    dialog = IssuePreviewDialog(REPORT, None, console=console,
                               traceback_text=TB)
    _BUILT.append(dialog)
    return dialog


@pytest.fixture(autouse=True)
def _destroy_dialogs(qapp):
    """Take every dialog down at the end of the test that built it."""
    yield
    while _BUILT:
        dialog = _BUILT.pop()
        timer = getattr(dialog, "_diagnose_timer", None)
        if timer is not None:
            timer.stop()
        dialog.close()
        dialog.setParent(None)
        dialog.deleteLater()
    qapp.processEvents()


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
    _BUILT.append(dialog)
    dialog._on_diagnose()
    assert "about something else" not in dialog.body_edit.toPlainText()
    assert console.asked == ["ValueError: different"]


def test_no_console_says_so_rather_than_crashing(qapp, told):
    dialog = IssuePreviewDialog(REPORT, None, console=None, traceback_text=TB)
    _BUILT.append(dialog)
    dialog._on_diagnose()
    assert told and "not available" in told[0][1]


def test_the_screen_hands_over_the_console_and_traceback():
    """A source check: the button is inert if the screen does not pass them."""
    from pathlib import Path

    import spacr.qt.screens.app_screen as app_screen

    source = Path(app_screen.__file__).read_text(encoding="utf-8")
    assert "console=self._console" in source
    assert "traceback_text=self._last_error_text" in source


def test_missing_error_text_never_starts_a_request(qapp, told):
    console = _Console()
    dialog = _dialog(qapp, console)
    dialog._traceback_text = ""
    dialog._on_diagnose()
    assert console.asked == []
    assert told == [("Diagnose", "There is no error text to diagnose.")]
    assert dialog.diagnose_btn.isEnabled()
    assert not dialog._diagnose_timer.isActive()


def test_failed_request_restores_the_button_without_leaving_a_timer(qapp, told):
    console = _Console()
    attempted = []

    def fail(traceback_text, show_raw):
        attempted.append((traceback_text, show_raw))
        raise RuntimeError("provider unavailable")

    console.open_error_flow = fail
    dialog = _dialog(qapp, console)
    dialog._on_diagnose()
    assert attempted == [(TB, False)]
    assert told == [("Diagnose", "spaCR AI could not be started.")]
    assert dialog.diagnose_btn.isEnabled()
    assert dialog.diagnose_btn.text() == "Diagnose"
    assert not dialog._diagnose_timer.isActive()


def test_a_failed_cached_lookup_still_allows_a_new_request(qapp, told):
    console = _Console()
    looked_up = []

    def fail(traceback_text):
        looked_up.append(traceback_text)
        raise RuntimeError("conversation was cleared")

    console.ai_explanation_of = fail
    dialog = _dialog(qapp, console)
    dialog._on_diagnose()
    assert looked_up == [TB]
    assert console.asked == [TB]
    assert told == []
    assert dialog._diagnose_timer.isActive()
    assert not dialog.diagnose_btn.isEnabled()


def test_a_polled_answer_ends_polling_and_reaches_the_approved_payload(qapp, told):
    console = _Console()
    dialog = _dialog(qapp, console)
    dialog._on_diagnose()
    assert dialog._diagnose_timer.isActive()
    assert not dialog.diagnose_btn.isEnabled()
    console._answer = "The input diameter is text."
    dialog._check_for_diagnosis()
    assert "The input diameter is text." in dialog.approved_report()["body"]
    assert not dialog._diagnose_timer.isActive()
    assert dialog.diagnose_btn.isEnabled()
    assert dialog.diagnose_btn.text() == "Diagnose"
    assert told == []


def test_an_empty_finished_stream_gets_two_polls_to_deliver_its_answer(qapp, told):
    console = _Console()
    dialog = _dialog(qapp, console)
    dialog._on_diagnose()
    console._ai_thread = None
    for _ in range(2):
        dialog._check_for_diagnosis()
        assert dialog._diagnose_timer.isActive()
        assert told == []
    dialog._check_for_diagnosis()
    assert told == [("Diagnose", "spaCR AI did not return an analysis. "
                     "The console shows what it said.")]
    assert not dialog._diagnose_timer.isActive()
    assert dialog.diagnose_btn.isEnabled()
    assert dialog.approved_report()["body"] == REPORT["body"]


def test_a_stream_that_never_finishes_times_out_at_the_deadline(qapp, told):
    from spacr.qt.ai.issue_preview import DIAGNOSE_POLL_MS, DIAGNOSE_TIMEOUT_MS

    dialog = _dialog(qapp, _Console())
    dialog._on_diagnose()
    dialog._diagnose_elapsed = DIAGNOSE_TIMEOUT_MS - 2 * DIAGNOSE_POLL_MS
    dialog._check_for_diagnosis()
    assert told == []
    assert dialog._diagnose_timer.isActive()
    dialog._check_for_diagnosis()
    assert told == [("Diagnose", "spaCR AI did not answer in time.")]
    assert not dialog._diagnose_timer.isActive()
    assert dialog.diagnose_btn.isEnabled()


def test_an_overlong_answer_is_bounded_and_visibly_truncated(qapp, told):
    from spacr.qt.ai.issue_preview import AI_ANALYSIS_MAX_CHARS

    dialog = _dialog(qapp, _Console(answer="x" * (AI_ANALYSIS_MAX_CHARS + 20)))
    dialog._on_diagnose()
    body = dialog.approved_report()["body"]
    assert "x" * AI_ANALYSIS_MAX_CHARS + "\n\n… (truncated)" in body
    assert "x" * (AI_ANALYSIS_MAX_CHARS + 1) not in body


def test_an_empty_answer_does_not_add_a_section_or_move_the_cursor(qapp):
    dialog = _dialog(qapp, _Console())
    original_position = dialog.body_edit.textCursor().position()
    dialog._show_diagnosis(" \n ")
    dialog._scroll_to_diagnosis()
    assert dialog.approved_report()["body"] == REPORT["body"]
    assert dialog.body_edit.textCursor().position() == original_position
