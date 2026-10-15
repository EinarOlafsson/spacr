"""Only complete sign-in URLs may reach the mocked browser callback."""
import pytest

from spacr.qt.ai.pty_sign_in import SignInDialog, strip_terminal_codes


URL = "https://example.invalid/oauth?state=abc123"


class LocalSession:
    def __init__(self):
        self.output = ""
        self.code = None
        self.stops = 0
    def read(self):
        result, self.output = self.output, ""
        return strip_terminal_codes(result)
    def poll(self):
        return self.code
    def stop(self):
        self.stops += 1
    def send(self, text):
        raise AssertionError("No authentication input is used")


@pytest.fixture
def prompt(qtbot):
    session, opened = LocalSession(), []
    dialog = SignInDialog("Local stand-in", [], session_factory=lambda argv: session,
                          open_url=opened.append)
    qtbot.addWidget(dialog)
    dialog._timer.stop()
    return dialog, session, opened


@pytest.mark.parametrize("first,second", [
    ("Open https://", "example.invalid/oauth?state=abc123 "),
    ("Open https://example.", "invalid/oauth?state=abc123 "),
    ("Open https://example.invalid/oa", "uth?state=abc123 "),
])
def test_only_completed_url_is_opened_after_short_reads(prompt, first, second):
    dialog, session, opened = prompt
    session.output = first
    dialog._tick()
    assert opened == []
    session.output = second
    dialog._tick()
    assert opened == [URL]
    assert dialog._urls == [URL]
    assert dialog._link.text() == URL


def test_complete_whitespace_terminated_url_still_opens_immediately(prompt):
    dialog, session, opened = prompt
    session.output = "Open " + URL + " \n"
    dialog._tick()
    assert opened == [URL]


def test_eof_flushes_a_final_url_without_newline_once(prompt):
    dialog, session, opened = prompt
    session.output = URL
    dialog._tick()
    assert opened == []
    session.code = 0
    dialog._tick()
    assert opened == [URL]
    assert dialog.succeeded is True
    assert session.stops == 1


def test_multiple_urls_keep_order_and_do_not_open_duplicates(prompt):
    dialog, session, opened = prompt
    other = "https://example.invalid/second"
    session.output = URL + " " + other[:15]
    dialog._tick()
    assert opened == [URL]
    assert dialog._urls == [URL]
    session.output = other[15:] + " "+URL+" "
    dialog._tick()
    assert dialog._urls == [URL, other]
    assert opened == [URL]
    assert dialog._link.text() == other
    dialog._open_latest()
    assert opened == [URL, other]


def test_existing_ansi_cleanup_survives_url_short_reads(prompt):
    dialog, session, opened = prompt
    session.output = "\x1b[32mhttps://example.invalid/"
    dialog._tick()
    assert opened == []
    session.output = "oauth?state=abc123\x1b[0m\n"
    dialog._tick()
    assert opened == [URL]


def test_cancellation_does_not_flush_an_incomplete_url_to_browser(prompt):
    dialog, session, opened = prompt
    session.output = "https://example.invalid/unfinished"
    dialog._tick()
    dialog.reject()
    assert session.stops == 1
    assert opened == []
