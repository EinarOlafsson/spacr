"""Item 700: the client paths the first tests left unmeasured.

The hosted provider's fixed answers about itself, the sentences for a
refusal that is not a quota, a report summary asked of no provider or of a
CLI that will not stop, an explanation the relay could not give, the usage
bar before any allowance has arrived, the console's background refresh when
the relay is slow or fails, and the first-run notice when its files cannot
be written.

Nothing here reaches the network: ``tests/conftest.py`` sets
``SPACR_NO_NETWORK`` and every test that lifts it installs a fake relay.
"""
from __future__ import annotations

import threading

import pytest

pytest.importorskip("PySide6")

from spacr import crashreport as cr  # noqa: E402
from spacr.qt.ai import providers  # noqa: E402
from tests.test_700_anonymous_error_reports import FakeRelay  # noqa: E402

TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/home/annasvensson/envs/spacr/lib/python3.11/site-packages/'
    'spacr/measure.py", line 88, in measure_crop\n'
    "    raise KeyError(name)\n"
    "KeyError: 'cell_area'\n")


@pytest.fixture
def relay(monkeypatch):
    """Allow the network for this test and install a fake relay."""
    monkeypatch.delenv(cr.NO_NETWORK_ENV, raising=False)
    fake = FakeRelay()
    monkeypatch.setattr(cr, "_open_url", fake)
    return fake


class _StuckCli:
    """A vendor CLI that never answers and cannot be cancelled."""

    name = "claude"

    def __init__(self):
        self.release = threading.Event()

    def stream_chat(self, messages, system="", model=None):
        """Wait until the test lets go, then answer nothing."""
        self.release.wait(5)
        return iter(())

    def cancel_stream(self):
        """Refuse to stop, as a broken CLI wrapper might."""
        raise RuntimeError("cannot cancel")


class TestTheHostedProviderDescribesItself:
    """No install, no sign-in, and answers named by where they come from."""

    def test_it_is_always_installed_and_signed_in(self):
        hosted = providers.hosted_provider()
        assert hosted.is_installed() is True
        assert hosted.is_logged_in() is True
        assert hosted.check_signed_in() is None

    def test_it_names_the_relay_it_asks(self):
        assert providers.hosted_provider().source_of_key() == \
            f"spaCR relay at {cr.relay_url()}"


class TestWhatTheHostedRequestCarries:
    """Only the newest traceback and the question in front of it are sent."""

    def test_no_conversation_sends_an_empty_question(self):
        summary, question = providers._hosted_request([])
        assert question == ""
        assert isinstance(summary, dict)

    def test_a_traceback_pasted_after_the_question_is_cut_off(self):
        content = "Why did my mask run stop?\n" + TRACEBACK
        _summary, question = providers._hosted_request(
            [{"role": "user", "content": content}])
        assert "Why did my mask run stop?" in question
        assert "Traceback" not in question
        assert "annasvensson" not in question


class TestRefusalSentences:
    """A refusal that is not a used-up allowance gets its own sentence."""

    def test_an_unreachable_relay_says_to_check_the_network(self):
        text = providers.hosted_refusal_message("network")
        assert text == ("spaCR hosted AI could not be reached. Check the "
                        "network connection and try again later.")

    def test_any_other_refusal_says_to_try_later(self):
        text = providers.hosted_refusal_message("error")
        assert text == ("spaCR hosted AI could not answer this time. Try "
                        "again later.")

    def test_no_allowance_has_no_text(self):
        assert providers.hosted_allowance_text(None) == ""
        assert providers.hosted_allowance_text({}) == ""


class TestReportSummaryEdges:
    """No provider, or the hosted one, writes no local summary."""

    def test_no_provider_and_the_hosted_one_write_nothing(self):
        summary = cr.public_summary(traceback_text=TRACEBACK)
        assert providers.write_report_summary(summary, None) == ""
        assert providers.write_report_summary(
            summary, providers.hosted_provider()) == ""

    def test_a_cli_that_will_not_stop_gives_no_summary(self):
        summary = cr.public_summary(traceback_text=TRACEBACK)
        stuck = _StuckCli()
        try:
            assert providers.write_report_summary(summary, stuck,
                                                  timeout=0.05) == ""
        finally:
            stuck.release.set()

    def test_no_provider_leaves_the_report_as_it_was(self):
        summary = cr.public_summary(traceback_text=TRACEBACK)
        assert providers.with_ai_summary(summary, None) == summary


class TestExplainFailureEdges:
    """No provider, a relay that breaks, and an answer that brings news."""

    def test_no_provider_explains_nothing(self):
        summary = cr.public_summary(traceback_text=TRACEBACK)
        assert providers.explain_failure(summary, None) == {
            "text": "", "tier": "", "refused": "none"}

    def test_an_unexpected_relay_error_is_an_error_refusal(self, monkeypatch):
        def _broken(*_args, **_kwargs):
            raise ValueError("bad JSON from the relay")

        monkeypatch.setattr(cr, "diagnose_remotely", _broken)
        summary = cr.public_summary(traceback_text=TRACEBACK)
        assert providers.explain_failure(
            summary, providers.hosted_provider()) == {
                "text": "", "tier": "", "refused": "error"}

    def test_an_answer_with_an_allowance_updates_the_usage(self, monkeypatch):
        allowance = {"ai": {"best_left": 2, "light_left": 9}}
        monkeypatch.setattr(cr, "diagnose_remotely",
                            lambda *_a, **_k: {"advice": "A missing column.",
                                               "tier": "light",
                                               "allowance": allowance})
        hosted = providers.hosted_provider()
        monkeypatch.setattr(hosted, "last_allowance", None)
        summary = cr.public_summary(traceback_text=TRACEBACK)
        out = providers.explain_failure(summary, hosted)
        assert out == {"text": "A missing column.", "tier": "light",
                       "refused": ""}
        assert hosted.last_allowance == allowance


class TestTheUsageBarBeforeAnAllowance:
    """spaCR AI with nothing heard yet shows only the own-CLI hint."""

    def test_no_allowance_yet_hides_the_bar(self):
        percent, text = providers.usage_bar(providers.HOSTED_PROVIDER_NAME,
                                            None)
        assert percent is None
        assert text == providers.hosted_cli_hint()


class TestApplyingAChoiceWithoutConsoles:
    """A console lookup that fails never undoes a saved choice."""

    def test_the_choice_is_saved_even_if_the_consoles_cannot_be_found(
            self, qapp, monkeypatch, relay):
        from PySide6.QtWidgets import QApplication

        from spacr.qt import preferences as prefs

        def _no_widgets():
            raise RuntimeError("the application is shutting down")

        monkeypatch.setattr(QApplication, "allWidgets",
                            staticmethod(_no_widgets))
        assert providers.apply_provider_choice(
            providers.HOSTED_PROVIDER_NAME) is True
        assert prefs.get_preferred_provider() == \
            providers.HOSTED_PROVIDER_NAME


class TestTheConsoleRefreshesOffTheGuiThread:
    """The usage line waits for a slow relay and survives a failing one."""

    def _panel(self, qtbot):
        from spacr.qt.widgets.console_panel import ConsolePanel

        panel = ConsolePanel()
        qtbot.addWidget(panel)
        return panel

    def _text(self, panel):
        from PySide6.QtWidgets import QLabel

        return panel.findChild(QLabel, "AiUsageText").text()

    def test_a_failing_fetch_shows_the_hint(self, qtbot, qt_theme_applied,
                                            monkeypatch):
        def _broken():
            raise OSError("relay down")

        monkeypatch.setattr(cr, "fetch_allowance", _broken)
        panel = self._panel(qtbot)
        panel.set_ai_provider(providers.HOSTED_PROVIDER_NAME)
        qtbot.waitUntil(
            lambda: self._text(panel) == providers.hosted_cli_hint(),
            timeout=5000)

    def test_a_slow_fetch_is_waited_for(self, qtbot, qt_theme_applied,
                                        monkeypatch):
        release = threading.Event()
        allowance = {"ai": {"best_left": 4, "light_left": 10}}

        def _slow():
            release.wait(5)
            return allowance

        monkeypatch.setattr(cr, "fetch_allowance", _slow)
        panel = self._panel(qtbot)
        panel.set_ai_provider(providers.HOSTED_PROVIDER_NAME)
        qtbot.wait(350)
        assert "4 of 5" not in self._text(panel)
        release.set()
        qtbot.waitUntil(lambda: "4 of 5" in self._text(panel), timeout=5000)

    def test_a_switch_meanwhile_keeps_the_new_line(self, qtbot,
                                                   qt_theme_applied,
                                                   monkeypatch):
        release = threading.Event()

        def _slow():
            release.wait(5)
            return {"ai": {"best_left": 4, "light_left": 10}}

        monkeypatch.setattr(cr, "fetch_allowance", _slow)
        panel = self._panel(qtbot)
        panel.set_ai_provider(providers.HOSTED_PROVIDER_NAME)
        panel._current_provider_name = "claude"
        panel._show_ai_usage(None)
        claude_line = self._text(panel)
        release.set()
        qtbot.wait(400)
        assert self._text(panel) == claude_line
        assert "4 of 5" not in claude_line

    def test_an_answer_without_an_allowance_asks_again(self, qtbot,
                                                       qt_theme_applied,
                                                       monkeypatch):
        calls = []
        monkeypatch.setattr(cr, "fetch_allowance",
                            lambda: calls.append(1) or None)
        panel = self._panel(qtbot)
        panel.set_ai_provider(providers.HOSTED_PROVIDER_NAME)
        qtbot.waitUntil(lambda: len(calls) >= 1, timeout=5000)
        qtbot.wait(150)
        before = len(calls)
        monkeypatch.setattr(providers.hosted_provider(), "last_allowance",
                            None)
        panel._on_stream_finished(True, "answer")
        qtbot.waitUntil(lambda: len(calls) > before, timeout=5000)

    def test_a_failed_answer_asks_again(self, qtbot, qt_theme_applied,
                                        monkeypatch):
        calls = []
        monkeypatch.setattr(cr, "fetch_allowance",
                            lambda: calls.append(1) or None)
        panel = self._panel(qtbot)
        panel.set_ai_provider(providers.HOSTED_PROVIDER_NAME)
        qtbot.waitUntil(lambda: len(calls) >= 1, timeout=5000)
        qtbot.wait(150)
        before = len(calls)
        panel._on_stream_finished(False, "relay down")
        qtbot.waitUntil(lambda: len(calls) > before, timeout=5000)


class TestTheNoticeWhenFilesFail:
    """An unwritable profile never stops the window from opening."""

    def test_an_example_that_cannot_be_built_says_why(self, qtbot,
                                                      qt_theme_applied,
                                                      monkeypatch):
        from spacr.qt.install_consent import AnonymousReportExampleDialog

        def _broken():
            raise ValueError("no example")

        monkeypatch.setattr(cr, "example_public_summary", _broken)
        dialog = AnonymousReportExampleDialog()
        qtbot.addWidget(dialog)
        assert dialog.example.toPlainText() == "ValueError: no example"

    def test_a_switch_that_cannot_be_saved_is_left_alone(self, qtbot,
                                                         qt_theme_applied,
                                                         monkeypatch):
        from spacr.qt.install_consent import AnonymousReportNotice

        def _read_only(_on):
            raise OSError("read-only profile")

        notice = AnonymousReportNotice()
        qtbot.addWidget(notice)
        monkeypatch.setattr(cr, "set_anonymous_reports_enabled", _read_only)
        notice._save_choice()
        assert notice.send_reports.isChecked() == \
            cr.anonymous_reports_enabled()

    def test_waiting_reports_are_retried_in_the_background(
            self, qtbot, qt_theme_applied, monkeypatch, relay):
        from spacr.qt.install_consent import maybe_show_anonymous_report_notice

        flushed = threading.Event()
        monkeypatch.setattr(cr, "queued_summaries", lambda: [{"x": 1}])
        monkeypatch.setattr(cr, "flush_report_queue",
                            lambda: flushed.set())
        monkeypatch.setattr(cr, "anonymous_report_notice_due", lambda: False)
        assert maybe_show_anonymous_report_notice(None) is None
        assert flushed.wait(5)

    def test_an_unreadable_profile_shows_no_notice(self, qtbot,
                                                   qt_theme_applied,
                                                   monkeypatch, relay):
        from spacr.qt.install_consent import maybe_show_anonymous_report_notice

        def _unreadable():
            raise OSError("no profile")

        monkeypatch.setattr(cr, "anonymous_report_notice_due", _unreadable)
        assert maybe_show_anonymous_report_notice(None) is None
