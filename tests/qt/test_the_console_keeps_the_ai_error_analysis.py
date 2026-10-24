"""The console pairs spaCR AI's answer with the error it answers.

The console holds ONE conversation for a whole session. Handing the bug
reporter "the last AI reply" would therefore file, against one crash, a
confident analysis of a different one -- or of whatever the user last asked
about. The pairing is what makes the attached analysis trustworthy.
"""
from __future__ import annotations

import pytest
from types import SimpleNamespace

from spacr.qt.widgets.console_panel import ConsolePanel

FIRST = "TypeError: '>' not supported between instances of 'str' and 'int'"
SECOND = "ValueError: could not broadcast input array"


@pytest.fixture
def manual_error_batches(monkeypatch):
    """Advance the real manual batcher's quiet clock and retire its own timers."""
    from spacr import crashreport

    now, timers = [0.], []

    def schedule(delay, callback):
        timer = crashreport._ErrorBatcher._schedule(delay, callback)
        timers.append(timer)
        return timer

    batcher = crashreport._ErrorBatcher(clock=lambda: now[0], schedule=schedule)
    monkeypatch.setattr(crashreport, "_ERROR_BATCHER", batcher)
    monkeypatch.setattr(crashreport, "_ERROR_POLICY_CACHE", {})
    monkeypatch.setattr(crashreport, "network_allowed", lambda: True)

    def finish_quiet_window():
        now[0] += 60.
        batcher._tick()

    yield finish_quiet_window
    for timer in timers:
        timer.cancel()
        timer.join(timeout=1.)
        assert not timer.is_alive()


@pytest.fixture
def console(qapp, monkeypatch, manual_error_batches):
    """A console panel that is DESTROYED when the test ends.

    Not tidiness. A ConsolePanel owns a running QTimer (the working dots) and
    keeps retired QThreads, and it has no parent here -- so nothing owned it
    and Python freed it whenever the collector next ran. That landed in the
    middle of a LATER test's `qapp.exec()`, destroying a widget with a live
    timer from inside event dispatch: `tests/qt/test_track_previews.py`
    segfaulted, having done nothing wrong.

    The suite now installs the application's `spacr.qt.gc_policy`, keeping
    collection on the GUI thread. Explicit teardown still matters: the
    policy does not choose when this panel's timers and workers should end.
    """
    panel = ConsolePanel()
    provider = SimpleNamespace(name="claude")
    monkeypatch.setattr(panel, "_current_provider", lambda: provider)
    monkeypatch.setattr(panel, "_start_stream", lambda **kw: None)
    monkeypatch.setattr(panel, "_console_context_for_question",
                        lambda text: ("", "no context"))
    yield panel
    # Stop the timer FIRST: it is the thing Qt cannot have running when the
    # object goes, and `deleteLater` would not have run by then.
    dots = getattr(panel, "_working_dots", None)
    if dots is not None:
        dots.stop()
        panel._working_dots = None
    panel.close()
    panel.setParent(None)
    panel.deleteLater()
    qapp.processEvents()


def test_an_answer_is_returned_for_the_error_it_explains(console, manual_error_batches):
    console.open_error_flow(FIRST, active_app="mask")
    manual_error_batches()
    console._on_stream_finished(True, "The diameter arrives as a string.")

    assert console.ai_explanation_of(FIRST) == (
        "The diameter arrives as a string.")


def test_it_is_not_returned_for_a_different_error(console, manual_error_batches):
    """The failure this pairing prevents."""
    console.open_error_flow(FIRST, active_app="mask")
    manual_error_batches()
    console._on_stream_finished(True, "The diameter arrives as a string.")

    assert console.ai_explanation_of(SECOND) == ""


def test_nothing_asked_means_nothing_to_attach(console):
    assert console.ai_explanation_of(FIRST) == ""


def test_an_unfinished_stream_attaches_nothing(console, manual_error_batches):
    """A report filed while the AI is still typing must carry no analysis
    rather than a half-written one."""
    console.open_error_flow(FIRST, active_app="mask")
    manual_error_batches()
    assert console.ai_explanation_of(FIRST) == ""


def test_a_failed_stream_attaches_nothing(console, manual_error_batches):
    console.open_error_flow(FIRST, active_app="mask")
    manual_error_batches()
    console._on_stream_finished(False, "provider refused the request")
    assert console.ai_explanation_of(FIRST) == ""


def test_a_question_of_the_users_own_ends_the_pairing(console, manual_error_batches):
    """Whatever comes back next answers the QUESTION, not the crash."""
    console.open_error_flow(FIRST, active_app="mask")
    manual_error_batches()
    console._on_stream_finished(True, "The diameter arrives as a string.")

    console._send_to_ai("how do I change the outline colour?")
    console._on_stream_finished(True, "Use the outline colour setting.")

    assert console.ai_explanation_of(FIRST) == "", (
        "an answer about outline colours would have been filed as the "
        "analysis of a TypeError")


def test_a_second_error_replaces_the_first(console, manual_error_batches):
    console.open_error_flow(FIRST, active_app="mask")
    manual_error_batches()
    console._on_stream_finished(True, "about the first")
    console.open_error_flow(SECOND, active_app="mask")
    manual_error_batches()
    console._on_stream_finished(True, "about the second")

    assert console.ai_explanation_of(SECOND) == "about the second"
    assert console.ai_explanation_of(FIRST) == ""


def test_whitespace_does_not_break_the_match(console, manual_error_batches):
    """The reporter is handed the traceback text again by the screen, not the
    same object, and log formatting can add a trailing newline."""
    console.open_error_flow(FIRST, active_app="mask")
    manual_error_batches()
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
