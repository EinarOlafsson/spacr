"""A failed run says what happened to its report and to its AI explanation.

Issue 117 (jak18015, macOS, 1.5.0.8): a Mask test run failed, with "Report
errors as GitHub issues" and "Route errors through AI" both on, and the user
saw neither an issue nor an explanation. The console they pasted shows why,
and none of it was test mode, the retry policy or the platform:

* The AI WAS asked. The ``claude`` CLI's sign-in had expired, so it printed
  ``Failed to authenticate: OAuth session expired and could not be
  refreshed`` and exited 1. spaCR ignored the exit status and treated that
  line as the answer: no "[AI error]", no hint to sign in again, and the line
  was later filed into issues #118 and #121 as "spaCR AI's analysis of this
  error".
* The line landed under "spaCR output", because the run's closing lines were
  written between the question and the reply, and the "spaCR AI" heading
  above stayed empty.
* No issue was filed because none ever is without a click: since instruction
  45 a report goes to the public tracker only after Send is pressed in its
  preview. That is deliberate. What was missing was the console saying so.

The last class drives the whole path the way the user hit it: a Mask screen,
a pipeline that fails through ``runctx``'s ``_give_up`` re-raise, and a
provider child process that really exits 1.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("PySide6")

from spacr.qt.ai import providers  # noqa: E402
from spacr.qt.ai.providers import ProviderFailed  # noqa: E402

AUTH_LINE = ("Failed to authenticate: OAuth session expired and could not "
             "be refreshed")

PICKLE_MESSAGE = (
    "This file contains pickled (object) data. If you trust the file you can "
    "load it unsafely using the `allow_pickle=` keyword argument or "
    "`pickle.load()`.")


@pytest.fixture(autouse=True)
def _no_desktop_notifications(monkeypatch):
    """A failed run announces itself through ``notify-send``/``osascript``;
    a test must not put that on the desktop of whoever runs it."""
    monkeypatch.setattr("spacr.qt.notify.announce_pipeline_finished",
                        lambda *args, **kwargs: None)


def _child_that_prints(line: str, status: int) -> list:
    """A real child process that prints ``line`` and exits with ``status``."""
    code = f"import sys; print({line!r}); sys.exit({status})"
    return [sys.executable, "-c", code]


class TestAProviderThatExitsNonZeroHasFailed:
    """The provider layer: the exit status decides, not the text."""

    def test_the_line_is_streamed_and_then_the_failure_is_raised(self):
        stream = providers._stream_process(_child_that_prints(AUTH_LINE, 1))

        assert next(stream).strip() == AUTH_LINE
        with pytest.raises(ProviderFailed) as failure:
            next(stream)

        assert failure.value.exit_status == 1
        assert failure.value.output_tail == AUTH_LINE
        assert "exit status 1" in str(failure.value)
        assert AUTH_LINE in str(failure.value)

    def test_the_message_names_the_command_that_signs_in_again(self):
        claude = providers.ClaudeCliProvider()

        with pytest.raises(ProviderFailed) as failure:
            list(providers._stream_process(
                _child_that_prints(AUTH_LINE, 1), provider=claude))

        assert f"`{claude.login_command}`" in str(failure.value)
        assert claude._current_proc is None, "the child was not released"

    def test_a_child_that_says_nothing_is_still_a_failure(self):
        code = "import sys; sys.exit(3)"
        with pytest.raises(ProviderFailed) as failure:
            list(providers._stream_process([sys.executable, "-c", code]))
        assert failure.value.exit_status == 3
        assert "printed nothing" in str(failure.value)

    def test_exit_status_zero_is_an_answer(self):
        lines = list(providers._stream_process(
            _child_that_prints("The diameter arrives as a string.", 0)))
        assert [line.strip() for line in lines] == [
            "The diameter arrives as a string."]

    def test_the_quotation_is_the_last_lines_and_is_bounded(self):
        many = "\n".join(f"line {n}" for n in range(20)) + "\n" + "x" * 900
        code = f"import sys; print({many!r}); sys.exit(1)"
        with pytest.raises(ProviderFailed) as failure:
            list(providers._stream_process([sys.executable, "-c", code]))
        tail = failure.value.output_tail
        assert tail.startswith("line 18 line 19 x")
        assert len(tail) <= providers._FAILURE_TAIL_CHARS

    def test_a_child_spacr_stopped_is_not_reported_as_failing(self):
        """Cancel ends the child with a signal on POSIX and with status 1 on
        Windows. Either way it was spaCR's doing, so nothing is raised."""
        code = ("import sys, time; print('first', flush=True); "
                "time.sleep(30)")
        claude = providers.ClaudeCliProvider()
        stream = providers._stream_process([sys.executable, "-c", code],
                                           provider=claude)
        assert next(stream).strip() == "first"

        claude.cancel_stream()

        assert list(stream) == []

    def test_status_one_after_a_cancel_is_not_a_failure_either(
            self, monkeypatch):
        """The Windows shape, where ``terminate`` exits the child with 1."""

        class _Stdout(list):
            def close(self):
                pass

        class _WindowsChild:
            stdin = None

            def __init__(self):
                self.stdout = _Stdout(["partial answer\n"])

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 1

            def poll(self):
                return 1

        child = _WindowsChild()
        monkeypatch.setattr(providers.subprocess, "Popen",
                            lambda argv, **kwargs: child)
        claude = providers.ClaudeCliProvider()
        stream = providers._stream_process(["claude"], provider=claude)
        assert next(stream) == "partial answer\n"
        claude.cancel_stream()
        assert list(stream) == []

    def test_a_status_after_a_child_that_would_not_exit_is_not_quoted(
            self, monkeypatch):
        """The reader's own escalation ends the child; the status that
        follows is the escalation's, not the CLI's."""
        waits = iter([subprocess.TimeoutExpired("claude", 1), 1])

        class _Lines(list):
            def close(self):
                pass

        class _Stubborn:
            stdin = None

            def __init__(self):
                self.stdout = _Lines(["done\n"])

            def wait(self, timeout=None):
                outcome = next(waits)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

            def terminate(self):
                pass

            def poll(self):
                return 1

        monkeypatch.setattr(providers.subprocess, "Popen",
                            lambda argv, **kwargs: _Stubborn())
        assert list(providers._stream_process(["claude"])) == ["done\n"]


class TestTheWorkerReportsIt:
    """The worker turns the raise into ``finished(False, ...)``."""

    class _Failing:
        name = "claude"
        label = "Claude"
        login_command = "claude setup-token"

        def __init__(self):
            self.cancelled = 0

        def cancel_stream(self):
            self.cancelled += 1

        def stream_chat(self, messages, system="", model=None):
            yield AUTH_LINE + "\n"
            raise ProviderFailed("claude", 1, AUTH_LINE,
                                 login_command=self.login_command)

    def _run(self, worker):
        got = []
        worker.finished.connect(lambda ok, text: got.append((ok, text)))
        worker.run()
        return got

    def test_a_failed_provider_finishes_not_ok(self, qtbot):
        from spacr.qt.ai.worker import StreamWorker

        got = self._run(StreamWorker(self._Failing(), []))

        assert len(got) == 1
        ok, text = got[0]
        assert ok is False
        assert text.startswith("ProviderFailed: claude stopped with exit "
                               "status 1: " + AUTH_LINE)
        assert "`claude setup-token`" in text

    def test_a_cancel_is_reported_as_a_cancel(self, qtbot):
        """Ending a child can make its reader raise before any chunk
        arrives -- a closed pipe, a status the provider reports. The user
        pressed Cancel, and that is what they are told."""
        from spacr.qt.ai.worker import StreamWorker

        class _RaisesOnceCancelled(self._Failing):
            def stream_chat(self, messages, system="", model=None):
                raise ProviderFailed("claude", 1, "", login_command="")
                yield

        worker = StreamWorker(_RaisesOnceCancelled(), [])
        worker.cancel()
        got = self._run(worker)
        assert got == [(False, "Cancelled.")]


def _entries(console) -> list:
    """The console's entries as ``(kind, text)``, top to bottom."""
    from spacr.qt.widgets.console_panel import _TopicBar

    out = []
    for index in range(console._entries.count()):
        item = console._entries.itemAt(index)
        widget = item.widget() if item is not None else None
        if widget is None:
            continue
        if isinstance(widget, _TopicBar):
            out.append(("heading", widget.text()))
        elif hasattr(widget, "toPlainText"):
            out.append(("text", widget.toPlainText()))
    return out


def _heading_over(console, needle: str) -> str:
    """The text of the nearest heading above the entry containing ``needle``."""
    heading = ""
    for kind, text in _entries(console):
        if kind == "heading":
            heading = text
        elif needle in text:
            return heading
    raise AssertionError(f"{needle!r} is not in the console")


@pytest.fixture
def console(qapp, monkeypatch):
    """A console whose provider is set and whose stream is started by hand."""
    from spacr.qt.widgets.console_panel import ConsolePanel

    panel = ConsolePanel()
    monkeypatch.setattr(panel, "_current_provider", lambda: object())
    monkeypatch.setattr(panel, "_start_stream", lambda **kw: None)
    yield panel
    dots = getattr(panel, "_working_dots", None)
    if dots is not None:
        dots.stop()
        panel._working_dots = None
    panel.close()
    panel.setParent(None)
    panel.deleteLater()
    qapp.processEvents()


class TestTheReplyKeepsItsHeading:

    def test_a_reply_after_the_run_s_last_lines_is_under_spacr_ai(
            self, console):
        console.open_error_flow("Traceback\nValueError: x", active_app="mask",
                                show_raw=False)
        console.append_stdout("run closed [failed]\n")
        console.append_notice("✗ Failed — see traceback above\n")

        console._on_chunk("The file is a macOS sidecar.\n")

        assert _heading_over(console, "macOS sidecar") == "spaCR AI"
        assert _heading_over(console, "run closed") != "spaCR AI"

    def test_an_uninterrupted_reply_opens_no_second_heading(self, console):
        console.open_error_flow("Traceback\nValueError: x", active_app="mask",
                                show_raw=False)
        console._on_chunk("one\n")
        console._on_chunk("two\n")

        headings = [text for kind, text in _entries(console)
                    if kind == "heading"]
        assert headings.count("spaCR AI") == 1
        assert _heading_over(console, "two") == "spaCR AI"


# ---------------------------------------------------------------------------
# The whole path, the way issue 117 hit it
# ---------------------------------------------------------------------------


def _fails_like_issue_117(settings):
    """A Mask entry that fails the way #117's did: a ValueError inside a
    plate, re-raised by the on_error=stop policy's ``_give_up``."""
    from spacr.errors import RunLedger
    from spacr.runctx import run_context

    ledger = RunLedger("preprocess_generate_masks")
    with run_context("mask", settings, ledger=ledger, log=False) as run:
        for attempt in run.policy.attempts_for(settings["src"],
                                               stage="plate"):
            with attempt:
                raise ValueError(PICKLE_MESSAGE)


_fails_like_issue_117.__qualname__ = "preprocess_generate_masks"


@pytest.fixture
def mask_screen(qtbot, monkeypatch, tmp_path):
    """A Mask screen whose Run fails like #117 and whose AI is signed out.

    Both preferences are on. The provider is the real Claude provider, whose
    child process is a real one that prints the expired-sign-in line and
    exits 1 -- only its command line is replaced, so no ``claude`` runs.
    """
    from spacr import run_journal
    from spacr.qt.ai import issue_report
    from spacr.qt.ai import settings as ai_settings
    from spacr.qt.ai.issue_preview import IssuePreviewDialog
    from spacr.qt import preferences
    from spacr.qt.screens import app_screen

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: runs)
    monkeypatch.setattr(ai_settings, "get_auto_file_issues", lambda: True)
    monkeypatch.setattr(ai_settings, "get_route_errors_through_ai",
                        lambda: True)
    monkeypatch.setattr(preferences, "get_issue_prompt_mode",
                        lambda: preferences.ISSUE_PROMPT_ASK)
    monkeypatch.setattr(app_screen, "resolve_pipeline_entry",
                        lambda key: _fails_like_issue_117)

    def _signed_out(self, messages, system="", model=None):
        yield from providers._stream_process(
            _child_that_prints(AUTH_LINE, 1), provider=self)

    monkeypatch.setattr(providers.ClaudeCliProvider, "stream_chat",
                        _signed_out)

    sent, previews = [], []
    monkeypatch.setattr(issue_report, "submit_report",
                        lambda report: sent.append(report) or "url")
    monkeypatch.setattr(IssuePreviewDialog, "exec",
                        lambda dialog: previews.append(dialog) or 0)

    screen = app_screen.AppScreen("mask")
    qtbot.addWidget(screen)
    screen._console.set_ai_active(True)
    screen._console.set_ai_provider("claude")
    screen.sent, screen.previews = sent, previews
    yield screen
    screen._console.shutdown()


def _run_until_everything_has_answered(qtbot, screen):
    screen._on_run(override={"src": "/Volumes/drive/Projection",
                             "test_mode": True, "on_error": "stop",
                             "hash_inputs": False})

    def _settled():
        text = screen._console.as_text()
        return ("✗ Failed" in text and "[AI error]" in text
                and not screen._console.is_ai_streaming())

    qtbot.waitUntil(_settled, timeout=30000)
    return screen._console.as_text()


class TestIssue117EndToEnd:

    def test_the_ai_failure_is_reported_with_the_way_to_sign_in(
            self, qtbot, mask_screen):
        text = _run_until_everything_has_answered(qtbot, mask_screen)

        assert "An error occurred — asking spaCR AI to explain it" in text
        assert "[AI error] ProviderFailed:" in text
        assert "exit status 1" in text
        assert "`claude setup-token`" in text
        assert _heading_over(mask_screen._console, AUTH_LINE) == "spaCR AI"

    def test_the_failure_is_not_kept_as_the_ai_s_analysis(
            self, qtbot, mask_screen):
        _run_until_everything_has_answered(qtbot, mask_screen)
        error = mask_screen._last_error_text

        assert PICKLE_MESSAGE in error
        assert mask_screen._console.ai_explanation_of(error) == ""

    def test_the_console_says_nothing_was_sent_and_how_to_send_it(
            self, qtbot, mask_screen):
        text = _run_until_everything_has_answered(qtbot, mask_screen)

        failed = text.index("✗ Failed")
        said = text.index("[issue] Nothing was sent to GitHub.")
        assert said > failed
        assert text[failed:said].count("\n") == 1, (
            "the line belongs directly under the failure")
        assert "File as issue" in text[said:]
        assert mask_screen._btn_file_issue.isEnabled()
        assert mask_screen.sent == [] and mask_screen.previews == []

    def test_the_report_it_offers_carries_no_sign_in_error(
            self, qtbot, mask_screen, monkeypatch):
        """#118 and #121 each carried the expired-sign-in line as "spaCR AI's
        analysis of this error". Press the button the way they did."""
        from spacr.qt.ai import issue_preview

        _run_until_everything_has_answered(qtbot, mask_screen)
        built = []
        original = issue_preview.IssuePreviewDialog.__init__

        def _capture(dialog, report, *args, **kwargs):
            built.append(report)
            original(dialog, report, *args, **kwargs)

        monkeypatch.setattr(issue_preview.IssuePreviewDialog, "__init__",
                            _capture)

        mask_screen._btn_file_issue.click()

        assert len(built) == 1
        assert "ValueError" in built[0]["body"]
        assert "Failed to authenticate" not in built[0]["body"]
        assert "spaCR AI's analysis" not in built[0]["body"]
        assert mask_screen.sent == []


class TestTheLineIsSaidOnlyWhenItIsTrue:

    @pytest.fixture
    def screen(self, qtbot, monkeypatch):
        from spacr.qt.screens.app_screen import AppScreen

        monkeypatch.setattr("spacr.qt.ai.settings.get_route_errors_through_ai",
                            lambda: False)
        screen = AppScreen("mask")
        qtbot.addWidget(screen)
        return screen

    @staticmethod
    def _said(screen) -> bool:
        return "Nothing was sent to GitHub" in screen._console.as_text()

    def _fail(self, screen, monkeypatch, *, reporting, mode="ask"):
        monkeypatch.setattr("spacr.qt.ai.settings.get_auto_file_issues",
                            lambda: reporting)
        monkeypatch.setattr("spacr.qt.preferences.get_issue_prompt_mode",
                            lambda: mode)
        screen._on_pipeline_error("Traceback\nValueError: boom")
        screen._on_finished(False)

    def test_not_when_reporting_is_off(self, screen, monkeypatch):
        self._fail(screen, monkeypatch, reporting=False)
        assert not self._said(screen)

    def test_not_when_reporting_is_set_to_never(self, screen, monkeypatch):
        self._fail(screen, monkeypatch, reporting=True, mode="never")
        assert not self._said(screen)

    def test_once_per_failure(self, screen, monkeypatch):
        self._fail(screen, monkeypatch, reporting=True, mode="always")
        screen._on_finished(False)
        assert screen._console.as_text().count(
            "Nothing was sent to GitHub") == 1

    def test_not_when_the_run_was_stopped(self, screen, monkeypatch):
        monkeypatch.setattr("spacr.qt.ai.settings.get_auto_file_issues",
                            lambda: True)
        screen._on_pipeline_error("Traceback\nValueError: boom")

        class _Stopped:
            was_cancelled = True

        screen._worker = _Stopped()
        screen._on_finished(False)
        screen._worker = None
        screen._on_finished(False)
        assert not self._said(screen), (
            "a stopped run carried the line over to the next failure")


def test_the_setting_says_nothing_is_sent_automatically(qtbot):
    """Where the user switched the feature on is where it has to say so."""
    from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog

    dialog = _ProvidersDialog()
    qtbot.addWidget(dialog)
    from PySide6.QtWidgets import QLabel

    captions = " ".join(label.text() for label in dialog.findChildren(QLabel))
    assert "Nothing is sent automatically" in captions
    assert "File as issue" in captions
    assert "one-click" not in dialog._auto_issue_chk.text()
