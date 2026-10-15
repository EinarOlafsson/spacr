"""Item 700 in the GUI: the switch, the notice, the failed-run wiring, hosted AI.

The switch "Send anonymised error reports" sits in Preferences, is on by
default and has a "What is sent…" example. A notice says so once. A failed
run sends its anonymised summary on the background runner and says in one
console line what became of it, while the GitHub report and its preview stay
as they were. With no vendor CLI signed in, the AI switch uses spaCR's hosted
provider, whose answers are labelled as generated advice and whose quota
refusal is a plain sentence.

Nothing here reaches the network: ``tests/conftest.py`` sets
``SPACR_NO_NETWORK`` and every test that lifts it installs a fake relay.
"""
from __future__ import annotations

import json

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
    "KeyError: 'cell_area in /home/annasvensson/screens/plate7/A01.tif'\n")


@pytest.fixture
def relay(monkeypatch):
    """Allow the network for this test and install a fake relay."""
    monkeypatch.delenv(cr.NO_NETWORK_ENV, raising=False)
    fake = FakeRelay()
    monkeypatch.setattr(cr, "_open_url", fake)
    return fake


@pytest.fixture
def no_vendor_cli(monkeypatch):
    """No Claude, Codex or Gemini CLI is installed and signed in."""
    for cls in (providers.ClaudeCliProvider, providers.CodexCliProvider,
                providers.GeminiCliProvider):
        monkeypatch.setattr(cls, "is_configured", lambda self: False)


class TestPreferences:
    """The switch, on by default, with its example one click away."""

    def _dialog(self, qtbot):
        from spacr.qt import preferences as prefs

        dialog = prefs.PreferencesDialog()
        qtbot.addWidget(dialog)
        return dialog

    def test_one_dropdown_defaults_to_summarize_and_send_all(
            self, qtbot, qt_theme_applied):
        from PySide6.QtWidgets import (QComboBox, QDialogButtonBox,
                                       QPushButton)

        from spacr.qt.widgets.toggle import Toggle

        dialog = self._dialog(qtbot)
        assert dialog.findChild(Toggle, "AnonymousReports") is None
        combo = dialog.findChild(QComboBox, "AiErrorReport")
        assert [combo.itemText(i) for i in range(combo.count())] == [
            "Summarize and send all errors",
            "Summarize and send all errors, with code context",
            "Summarize all errors, send only faults",
            "Summarize all errors, send nothing",
            "Send all errors, no AI summary",
            "Send only faults, no AI summary",
            "Off"]
        assert combo.currentData() == "all"
        assert dialog.findChild(QPushButton, "AnonymousReportsWhatIsSent")
        page = dialog.findChild(object, "PreferencesTabAI")
        assert page.isAncestorOf(combo)
        combo.setCurrentIndex(combo.findData("off"))
        boxes = [box for box in dialog.findChildren(QDialogButtonBox)
                 if box.button(QDialogButtonBox.Ok) is not None
                 or box.button(QDialogButtonBox.Save) is not None]
        assert boxes, "no OK/Save button box in Preferences"
        boxes[0].accepted.emit()
        assert cr.error_report_mode() == "off"
        assert cr.anonymous_reports_enabled() is False

    def test_what_is_sent_shows_an_allowlisted_example(self, qtbot,
                                                       qt_theme_applied):
        from PySide6.QtWidgets import QPlainTextEdit

        from spacr.qt.install_consent import AnonymousReportExampleDialog

        dialog = AnonymousReportExampleDialog()
        qtbot.addWidget(dialog)
        text = dialog.findChild(QPlainTextEdit,
                                "AnonymousReportExampleText").toPlainText()
        example = json.loads(text)
        assert set(example) <= set(cr.PUBLIC_SUMMARY_KEYS)
        assert "patient_042" not in text and "ghp_" not in text


class TestFirstRunNotice:
    """Disclosed once, never modal, and silent while the network is forbidden."""

    def test_shown_once(self, qtbot, qt_theme_applied, monkeypatch):
        from spacr.qt.install_consent import maybe_show_anonymous_report_notice

        monkeypatch.delenv(cr.NO_NETWORK_ENV, raising=False)
        notice = maybe_show_anonymous_report_notice(None)
        assert notice is not None
        qtbot.addWidget(notice)
        assert notice.isVisible() and not notice.isModal()
        assert maybe_show_anonymous_report_notice(None) is None

    def test_turning_it_off_in_the_notice_is_one_click(self, qtbot,
                                                      qt_theme_applied,
                                                      monkeypatch):
        from spacr.qt.install_consent import maybe_show_anonymous_report_notice

        monkeypatch.delenv(cr.NO_NETWORK_ENV, raising=False)
        notice = maybe_show_anonymous_report_notice(None)
        qtbot.addWidget(notice)
        assert notice.send_reports.isChecked()
        notice.send_reports.setChecked(False)
        notice.accept()
        assert cr.anonymous_reports_enabled() is False

    def test_not_shown_without_network(self, qtbot, qt_theme_applied):
        from spacr.qt.install_consent import maybe_show_anonymous_report_notice

        assert maybe_show_anonymous_report_notice(None) is None
        assert cr.anonymous_report_notice_due() is True


class _Jobs:
    """A background runner that records jobs instead of running them."""

    def __init__(self):
        self.jobs = []

    def submit(self, fn, on_done):
        self.jobs.append((fn, on_done))
        return True


class TestFailedRunWiring:
    """A failed run sends its summary off the GUI thread and says so."""

    @pytest.fixture
    def screen(self, qtbot, qt_theme_applied, monkeypatch):
        from spacr.qt.screens.app_screen import AppScreen

        screen = AppScreen("mask")
        qtbot.addWidget(screen)
        notices = []
        monkeypatch.setattr(screen._console, "append_notice",
                            lambda source, **values: notices.append(
                                source.format(**values)))
        screen._notices = notices
        monkeypatch.setattr(screen._console, "fill_ai_entry",
                            lambda block, text: notices.append(
                                "spaCR AI: " + text))
        recorder = _Jobs()
        monkeypatch.setattr(screen._jobs, "submit", recorder.submit)
        screen._recorded = recorder
        monkeypatch.setattr(screen, "_report_summary_provider",
                            providers.hosted_provider)
        cr.set_error_report_mode("send_all")
        return screen

    def _fail(self, screen):
        screen._last_error_text = TRACEBACK
        screen._anonymous_report_due = True
        screen._settle_the_report(failed=True)

    def test_the_send_is_queued_on_the_runner_not_run_inline(self, screen,
                                                             relay):
        self._fail(screen)
        assert relay.calls == [], "nothing ran on the GUI thread"
        assert len(screen._recorded.jobs) == 1

    def test_the_outcome_is_one_line_with_the_issue(self, screen, relay):
        self._fail(screen)
        fn, on_done = screen._recorded.jobs[0]
        on_done(fn())
        assert any("Reported anonymously as issue #12" in line
                   and "/issues/12" in line for line in screen._notices)
        body = relay.calls[0]["body"]
        assert body["module"] == "mask"
        assert body["frames"] == [{"file": "measure.py",
                                   "function": "measure_crop", "line": 88}]
        assert "annasvensson" not in json.dumps(body)
        assert "A01" not in json.dumps(body)

    def test_offline_says_it_will_retry(self, screen, relay):
        relay.answers = [OSError("offline")]
        self._fail(screen)
        fn, on_done = screen._recorded.jobs[0]
        on_done(fn())
        assert any("will retry later" in line for line in screen._notices)

    def test_off_queues_nothing(self, screen, relay):
        cr.set_anonymous_reports_enabled(False)
        self._fail(screen)
        assert all(fn.__name__ != "_send" for fn, _ in screen._recorded.jobs)
        assert relay.calls == []

    def test_a_settings_error_is_a_user_error(self, screen, relay):
        screen._on_pipeline_error(
            "Traceback (most recent call last):\n"
            "spacr.errors.ConfigurationError: src does not exist\n")
        assert screen._anonymous_report_due is True
        assert screen._failure_kind == "user_error"

    def test_the_github_button_is_still_offered(self, screen, relay):
        from spacr.qt.ai import settings as ai_settings

        ai_settings.set_auto_file_issues(True)
        screen._on_pipeline_error(TRACEBACK)
        assert not screen._btn_file_issue.isHidden()


class TestHostedProvider:
    """The account-free provider: default without a CLI, labelled, refusals clear."""

    def test_default_without_a_vendor_cli(self, relay, no_vendor_cli):
        hosted = providers.hosted_provider()
        assert providers.configured_providers() == [hosted]
        assert providers.default_provider() is hosted
        assert providers.get_provider(providers.HOSTED_PROVIDER_NAME) is hosted
        assert hosted not in providers.list_providers()

    def test_spacr_ai_is_the_default_and_a_chosen_cli_is_used(
            self, relay, monkeypatch, no_vendor_cli):
        monkeypatch.setattr(providers.CodexCliProvider, "is_configured",
                            lambda self: True)
        assert providers.default_provider().name == \
            providers.HOSTED_PROVIDER_NAME
        assert [p.name for p in providers.configured_providers()] == [
            providers.HOSTED_PROVIDER_NAME, "codex"]
        assert providers.default_provider("codex").name == "codex"
        assert providers.default_provider("gemini").name == \
            providers.HOSTED_PROVIDER_NAME
        assert providers.hosted_provider().label == "spaCR AI (free)"

    def test_unavailable_when_the_network_is_forbidden(self, no_vendor_cli):
        assert providers.configured_providers() == []
        assert providers.default_provider() is None

    def test_the_ai_switch_falls_back_to_hosted(self, qtbot, qt_theme_applied,
                                                relay, no_vendor_cli):
        from spacr.qt.screens.app_screen import AppScreen

        screen = AppScreen("mask")
        qtbot.addWidget(screen)
        screen._console.set_ai_provider(None)
        screen._on_ai_switch(True)
        assert screen._console._current_provider_name == \
            providers.HOSTED_PROVIDER_NAME

    def test_the_manuscript_never_goes_to_the_hosted_route(self, relay,
                                                           no_vendor_cli):
        from spacr.qt.ai.manuscript import availability

        assert availability().ok is False

    def test_advice_is_labelled_and_only_the_summary_is_sent(self, relay):
        from spacr.qt.ai.prompts import wrap_error_for_prompt

        relay.answers = [(200, {"advice": "The column is cell_area_um.",
                                "model": "qwen3-30b"})]
        messages = [{"role": "user", "content":
                     wrap_error_for_prompt(TRACEBACK, "measure")}]
        reply = "".join(providers.hosted_provider().stream_chat(messages))
        assert reply.startswith("The column is cell_area_um.")
        assert "Generated advice from spaCR hosted AI (qwen3-30b)" in reply
        body = relay.calls[0]["body"]
        assert relay.calls[0]["url"].endswith("/diagnose")
        assert set(body) <= set(cr.PUBLIC_SUMMARY_KEYS) | {"question"}
        assert body["module"] == "measure"
        assert body["exception_type"] == "KeyError"
        assert "annasvensson" not in json.dumps(body)
        assert "Summarise the cause" in body["question"]

    def test_429_says_the_free_daily_limit_is_reached(self, relay):
        relay.answers = [(429, {"refused": "quota"})]
        with pytest.raises(providers.HostedProviderRefused) as refused:
            list(providers.hosted_provider().stream_chat(
                [{"role": "user", "content": "why?"}]))
        assert "You've used today's free spaCR AI questions" in str(
            refused.value)
        assert refused.value.reason == "quota"

    def test_the_console_shows_the_quota_sentence_as_it_stands(self, relay,
                                                              qapp):
        from spacr.qt.ai.worker import StreamWorker

        relay.answers = [(429, {"refused": "quota"})]
        worker = StreamWorker(providers.hosted_provider(),
                              [{"role": "user", "content": "why?"}])
        finished = []
        worker.finished.connect(lambda ok, text: finished.append((ok, text)))
        worker.run()
        assert finished == [(False, providers.hosted_refusal_message("quota"))]
        assert not finished[0][1].startswith("HostedProviderRefused")


class TestDailyAllowanceDisplay:
    """The allowance line, the light-model tag and the quota sentences."""

    ALLOWANCE = {"ai": {"best_left": 4, "light_left": 10}, "reports_left": 5,
                 "pool": "best", "resets_at": "2026-10-15T00:00:00Z"}

    def _when(self):
        return providers.local_reset_time("2026-10-15T00:00:00Z")

    def test_the_allowance_line(self):
        text = providers.hosted_allowance_text(self.ALLOWANCE)
        assert text == (f"4 of 5 best-model questions left today · then 10 "
                        f"on the lighter model · resets at {self._when()}")

    def test_best_used_then_light(self):
        allowance = dict(self.ALLOWANCE, ai={"best_left": 0, "light_left": 7})
        assert providers.hosted_allowance_text(allowance).startswith(
            "Best-model questions used today · 7 of 10 lighter-model")

    def test_a_low_pool_serves_the_light_model(self):
        allowance = dict(self.ALLOWANCE, pool="light")
        assert "everyone gets the lighter model" in \
            providers.hosted_allowance_text(allowance)

    def test_nothing_left(self):
        allowance = dict(self.ALLOWANCE, ai={"best_left": 0, "light_left": 0})
        assert providers.hosted_allowance_text(allowance).startswith(
            "No free spaCR AI questions left today")
        assert "capacity is used up" in providers.hosted_allowance_text(
            dict(self.ALLOWANCE, pool="exhausted"))

    def test_the_reset_is_shown_in_local_time(self):
        from datetime import datetime, timezone

        moment = datetime(2026, 10, 15, tzinfo=timezone.utc)
        assert self._when().startswith(moment.astimezone().strftime("%H:%M"))

    def test_quota_for_this_install(self):
        text = providers.hosted_refusal_message(
            "quota", "install", "2026-10-15T00:00:00Z")
        assert text.startswith("You've used today's free spaCR AI questions. "
                               f"They reset at {self._when()}.")
        assert providers.hosted_cli_hint() in text

    def test_quota_for_everyone(self):
        text = providers.hosted_refusal_message(
            "quota", "global", "2026-10-15T00:00:00Z")
        assert text.startswith("spaCR's free AI capacity is used up for "
                               f"today. Try again after {self._when()}.")
        assert "unlimited use" in text

    def test_global_429_from_the_relay(self, relay):
        relay.answers = [(429, {"refused": "quota", "scope": "global"})]
        with pytest.raises(providers.HostedProviderRefused) as refused:
            list(providers.hosted_provider().stream_chat(
                [{"role": "user", "content": "why?"}]))
        assert "free AI capacity is used up" in str(refused.value)

    def test_a_light_answer_is_tagged(self, relay):
        relay.answers = [(200, {"advice": "Try this.", "model": "light-8b",
                                "tier": "light", "allowance": {
                                    "best_left": 0, "light_left": 9}})]
        hosted = providers.hosted_provider()
        reply = "".join(hosted.stream_chat([{"role": "user",
                                             "content": "why?"}]))
        assert "[lighter model]" in reply
        assert hosted.last_allowance["ai"] == {"best_left": 0,
                                               "light_left": 9}

    def test_a_best_answer_is_not_tagged(self, relay):
        relay.answers = [(200, {"advice": "Try this.", "model": "big",
                                "tier": "best"})]
        reply = "".join(providers.hosted_provider().stream_chat(
            [{"role": "user", "content": "why?"}]))
        assert "[lighter model]" not in reply

    def test_the_console_fetches_the_allowance_on_open(self, qtbot,
                                                       qt_theme_applied,
                                                       relay):
        from PySide6.QtWidgets import QLabel

        from spacr.qt.widgets.console_panel import ConsolePanel

        relay.answers = [(200, self.ALLOWANCE)]
        panel = ConsolePanel()
        qtbot.addWidget(panel)
        label = panel.findChild(QLabel, "AiUsageText")
        row = panel.findChild(object, "AiUsageRow")
        assert row.isHidden()
        panel.set_ai_provider(providers.HOSTED_PROVIDER_NAME)
        qtbot.waitUntil(lambda: "4 of 5" in label.text(), timeout=5000)
        assert "4 of 5 best-model questions left today" in label.text()
        assert providers.hosted_cli_hint() in row.toolTip()
        assert relay.calls[0]["method"] == "GET"
        panel.set_ai_provider("claude")
        assert label.text() == ("Your own Claude account · limits set by "
                                "Anthropic")

    def test_the_line_refreshes_after_an_answer(self, qtbot, qt_theme_applied,
                                                relay, monkeypatch):
        from PySide6.QtWidgets import QLabel

        from spacr.qt.widgets.console_panel import ConsolePanel

        monkeypatch.setattr(cr, "fetch_allowance", lambda: None)
        panel = ConsolePanel()
        qtbot.addWidget(panel)
        panel.set_ai_provider(providers.HOSTED_PROVIDER_NAME)
        monkeypatch.setattr(providers.hosted_provider(), "last_allowance",
                            dict(self.ALLOWANCE,
                                 ai={"best_left": 3, "light_left": 10}))
        panel._on_stream_finished(True, "answer")
        label = panel.findChild(QLabel, "AiUsageText")
        assert "3 of 5 best-model questions left today" in label.text()

    def test_a_quota_refused_report_says_it_waits(self, qtbot,
                                                  qt_theme_applied,
                                                  monkeypatch):
        from spacr.qt.screens.app_screen import AppScreen

        screen = AppScreen("mask")
        qtbot.addWidget(screen)
        notices = []
        monkeypatch.setattr(screen._console, "append_notice",
                            lambda source, **values: notices.append(source))
        screen._on_anonymous_report_sent({"status": cr.REPORT_QUEUED,
                                          "quota": True})
        assert notices == [
            "[report] Will report after the daily limit resets.\n"]

    def test_the_first_run_notice_states_the_allowances(self, qtbot,
                                                        qt_theme_applied):
        from PySide6.QtWidgets import QLabel

        from spacr.qt.install_consent import AnonymousReportNotice

        notice = AnonymousReportNotice()
        qtbot.addWidget(notice)
        text = " ".join(label.text() for label in notice.findChildren(QLabel))
        assert "up to 5 reports a day" in text
        assert "about 5 questions a day on the best model" in text
        assert "10 more on a lighter model" in text



class TestNoAiStepAtStartup:
    """A new user is never asked to pick an AI; the AI starts on."""

    def test_no_provider_question(self):
        from spacr.qt import setup_screen

        assert "ai_provider" not in {q[0] for q in setup_screen.questions()}

    def test_the_setup_dialog_draws_no_provider_step(self, qtbot,
                                                     qt_theme_applied):
        from spacr.qt.widgets.setup_dialog import SetupDialog

        dialog = SetupDialog()
        qtbot.addWidget(dialog)
        assert "ai_provider" not in dialog._editors

    def test_the_ai_is_on_by_default(self):
        from spacr.qt import preferences as prefs

        assert prefs.get_ai_on_by_default() is True
        assert providers.effective_provider_name(
            prefs.get_preferred_provider()) == providers.HOSTED_PROVIDER_NAME


class TestPreferencesProviderSwitch:
    """Four obvious choices, each saying what it needs, applied at once."""

    def _dialog(self, qtbot):
        from spacr.qt import preferences as prefs

        dialog = prefs.PreferencesDialog()
        qtbot.addWidget(dialog)
        return dialog

    def _radio(self, dialog, name):
        from PySide6.QtWidgets import QRadioButton

        return dialog.findChild(QRadioButton, f"AiProvider_{name}")

    def test_four_choices_with_spacr_ai_first_and_on(self, qtbot,
                                                     qt_theme_applied):
        from PySide6.QtWidgets import QLabel

        dialog = self._dialog(qtbot)
        names = [name for name, _l, _n in providers.PROVIDER_CHOICES]
        assert names == [providers.HOSTED_PROVIDER_NAME, "claude", "codex",
                         "gemini"]
        labels = [self._radio(dialog, n).text() for n in names]
        assert labels == ["spaCR AI (free, daily allowance)", "Claude",
                          "GPT (Codex)", "Gemini"]
        assert self._radio(dialog, providers.HOSTED_PROVIDER_NAME).isChecked()
        texts = [label.text() for label in dialog.findChildren(QLabel)]
        assert "Uses your Claude Code login (the claude CLI)." in texts
        note = dialog.findChild(QLabel, "AiProviderAllowance")
        assert not note.isHidden() and "5 questions on the best model" in \
            note.text()

    def test_switching_to_a_ready_cli_applies_at_once(self, qtbot,
                                                      qt_theme_applied,
                                                      monkeypatch, relay):
        from PySide6.QtWidgets import QLabel

        from spacr.qt import preferences as prefs
        from spacr.qt.widgets.console_panel import ConsolePanel

        monkeypatch.setattr(providers.ClaudeCliProvider, "is_configured",
                            lambda self: True)
        monkeypatch.setattr(cr, "fetch_allowance", lambda: None)
        console = ConsolePanel()
        qtbot.addWidget(console)
        console.set_ai_provider(providers.HOSTED_PROVIDER_NAME)
        dialog = self._dialog(qtbot)
        self._radio(dialog, "claude").setChecked(True)
        assert prefs.get_preferred_provider() == "claude"
        assert console._current_provider_name == "claude"
        assert dialog.findChild(QLabel, "AiProviderAllowance").isHidden()
        assert dialog.findChild(QLabel, "AiProviderStatus").isHidden()

    def test_a_cli_not_set_up_is_saved_but_spacr_ai_keeps_answering(
            self, qtbot, qt_theme_applied, monkeypatch, relay, no_vendor_cli):
        from PySide6.QtWidgets import QLabel

        from spacr.qt import preferences as prefs
        from spacr.qt.widgets.console_panel import ConsolePanel

        monkeypatch.setattr(cr, "fetch_allowance", lambda: None)
        console = ConsolePanel()
        qtbot.addWidget(console)
        dialog = self._dialog(qtbot)
        self._radio(dialog, "gemini").setChecked(True)
        assert prefs.get_preferred_provider() == "gemini"
        assert console._current_provider_name == \
            providers.HOSTED_PROVIDER_NAME
        assert not dialog.findChild(QLabel, "AiProviderStatus").isHidden()


class _FakeCli:
    """A vendor CLI provider that answers, fails or hangs as told."""

    name = "claude"

    def __init__(self, answer="", error=None, hang=False):
        self.answer, self.error, self.hang = answer, error, hang
        self.prompts = []
        self.cancelled = False

    def stream_chat(self, messages, system="", model=None):
        import time as _time

        self.prompts.append(messages[-1]["content"])
        if self.error:
            raise self.error
        if self.hang:
            while not self.cancelled:
                _time.sleep(0.01)
            return
        yield self.answer

    def cancel_stream(self):
        self.cancelled = True


class TestReportAiSummary:
    """The report's AI summary: the user's CLI, the relay, or none."""

    def test_spacr_ai_asks_the_relay(self, relay):
        from tests.test_700_anonymous_error_reports import _summary

        summary = providers.with_ai_summary(_summary(),
                                            providers.hosted_provider())
        assert summary["summarize"] is True and "ai_summary" not in summary
        cr.submit_public_summary(summary)
        assert relay.calls[0]["body"]["summarize"] is True

    def test_a_private_cli_writes_it_from_the_sanitized_summary(self, relay):
        from tests.test_700_anonymous_error_reports import _summary

        cli = _FakeCli("The KeyError comes from measure_crop in measure.py; "
                       "see /home/annasvensson/x.tif or mail a@b.org.")
        summary = providers.with_ai_summary(_summary("boom"), cli)
        prompt = cli.prompts[0]
        assert "install_id" not in prompt and "Summarise the likely cause" \
            in prompt and "at most 1200 characters" in prompt
        text = summary["ai_summary"]
        assert "measure_crop" in text
        assert "annasvensson" not in text and "a@b.org" not in text
        cr.submit_public_summary(summary)
        body = relay.calls[0]["body"]
        assert body["ai_summary"] == text and "summarize" not in body

    def test_a_failing_cli_files_without_a_summary(self, relay):
        from tests.test_700_anonymous_error_reports import _summary

        summary = providers.with_ai_summary(
            _summary(), _FakeCli(error=RuntimeError("signed out")))
        assert "ai_summary" not in summary and "summarize" not in summary

    def test_a_slow_cli_is_stopped_and_files_without_a_summary(self):
        from tests.test_700_anonymous_error_reports import _summary

        cli = _FakeCli(hang=True)
        assert providers.write_report_summary(_summary(), cli,
                                              timeout=0.2) == ""
        assert cli.cancelled

    def test_the_summary_is_scrubbed_and_capped(self):
        from tests.test_700_anonymous_error_reports import _summary

        dirty = dict(_summary(), ai_summary="see C:\\Users\\anna\\p.nd2 "
                     "token ghp_0123456789abcdefABCDEF0123 " + "x " * 3000,
                     summarize=True)
        clean = cr._validated_summary(dirty)
        assert len(clean["ai_summary"]) <= cr.MAX_AI_SUMMARY_CHARS
        assert "anna" not in clean["ai_summary"]
        assert "ghp_" not in clean["ai_summary"]
        assert "summarize" not in clean

    @pytest.fixture
    def screen(self, qtbot, qt_theme_applied, monkeypatch):
        from spacr.qt.screens.app_screen import AppScreen

        screen = AppScreen("mask")
        qtbot.addWidget(screen)
        notices = []
        monkeypatch.setattr(screen._console, "append_notice",
                            lambda source, **values: notices.append(
                                source.format(**values)))
        screen._notices = notices
        monkeypatch.setattr(screen._console, "fill_ai_entry",
                            lambda block, text: notices.append(
                                "spaCR AI: " + text))
        recorder = _Jobs()
        monkeypatch.setattr(screen._jobs, "submit", recorder.submit)
        screen._recorded = recorder
        monkeypatch.setattr(screen, "_report_summary_provider",
                            providers.hosted_provider)
        return screen

    def _run(self, screen):
        screen._last_error_text = TRACEBACK
        screen._anonymous_report_due = True
        screen._settle_the_report(failed=True)
        fn, on_done = screen._recorded.jobs[-1]
        on_done(fn())
        return screen._notices[-1]

    def test_spacr_ai_explains_once_and_the_text_is_the_summary(
            self, screen, relay):
        relay.routes["/diagnose"] = [(200, {
            "advice": "The column cell_area is missing; set it in Measure.",
            "model": "big", "tier": "best"})]
        relay.routes["/report"] = [(200, {"issue_number": 7, "issue_url": "u",
                                          "deduplicated": False,
                                          "ai_summary_source": "client"})]
        assert "Reported as issue #7 (with AI summary)" in self._run(screen)
        assert any(line.startswith("spaCR AI: The column cell_area")
                   for line in screen._notices)
        assert len(relay.calls_to("/diagnose")) == 1, "one AI call"
        body = relay.calls_to("/report")[0]["body"]
        assert body["ai_summary"].startswith("The column cell_area")
        assert "summarize" not in body

    def test_without_a_summary_when_the_allowance_is_used(self, screen,
                                                          relay):
        relay.routes["/diagnose"] = [(429, {"refused": "quota",
                                            "scope": "install"})]
        relay.routes["/report"] = [(200, {"issue_number": 8, "issue_url": "u",
                                          "deduplicated": False,
                                          "ai_summary_source": "none"})]
        assert ("Reported as issue #8 (without AI summary: daily AI "
                "allowance used)") in self._run(screen)
        assert any("today's free AI is used up" in line
                   for line in screen._notices)
        assert "ai_summary" not in relay.calls_to("/report")[0]["body"]

    def test_with_a_private_cli_summary(self, screen, relay, monkeypatch):
        cli = _FakeCli("Cause: a missing column in measure_crop.")
        monkeypatch.setattr(screen, "_report_summary_provider", lambda: cli)
        relay.answers = [(200, {"issue_number": 9, "issue_url": "u",
                                "deduplicated": False,
                                "ai_summary_source": "client"})]
        assert "Reported as issue #9 (with AI summary)" in self._run(screen)
        assert relay.calls_to("/diagnose") == []
        assert relay.calls_to("/report")[0]["body"]["ai_summary"].startswith(
            "Cause:")
        assert any(line.startswith("spaCR AI: Cause:")
                   for line in screen._notices)

    def test_a_failing_cli_still_files(self, screen, relay, monkeypatch):
        cli = _FakeCli(error=RuntimeError("signed out"))
        monkeypatch.setattr(screen, "_report_summary_provider", lambda: cli)
        relay.answers = [(200, {"issue_number": 10, "issue_url": "u",
                                "deduplicated": False,
                                "ai_summary_source": "none"})]
        assert "Reported as issue #10 (without AI summary)" in \
            self._run(screen)
        assert "ai_summary" not in relay.calls_to("/report")[0]["body"]

    def test_the_notice_mentions_the_ai_summary(self, qtbot,
                                                 qt_theme_applied):
        from PySide6.QtWidgets import QLabel

        from spacr.qt.install_consent import AnonymousReportNotice

        notice = AnonymousReportNotice()
        qtbot.addWidget(notice)
        text = " ".join(label.text() for label in notice.findChildren(QLabel))
        assert "AI-written summary" in text
        assert "your own AI when one is connected" in text


FAULT_TB = (
    "Traceback (most recent call last):\n"
    '  File "/home/annasvensson/my_script.py", line 5, in <module>\n'
    "    run_my_plate('/home/annasvensson/secret_plate')\n"
    '  File "/home/annasvensson/envs/spacr/lib/python3.11/site-packages/'
    'spacr/measure.py", line 88, in measure_crop\n'
    "    raise KeyError(name)\n"
    "KeyError: 'cell_area'\n")

USER_TB = ("spacr.errors.ConfigurationError: At least one registered "
           "*_channel (for example cell_channel or organelle_channel) must "
           "be set; no masks can be generated.")

#: (explains, sends a fault, sends a user error) for every choice.
EXPECTED = {
    "all": (True, True, True),
    "all_code": (True, True, True),
    "faults": (True, True, False),
    "summarize": (True, False, False),
    "send_all": (False, True, True),
    "send_faults": (False, True, False),
    "off": (False, False, False),
}


class TestEveryAiErrorReportChoice:
    """What is explained and what is sent, for each choice and each kind."""

    @pytest.fixture
    def screen(self, qtbot, qt_theme_applied, monkeypatch):
        from spacr.qt.screens.app_screen import AppScreen

        screen = AppScreen("mask")
        qtbot.addWidget(screen)
        notices = []
        monkeypatch.setattr(screen._console, "append_notice",
                            lambda source, **values: notices.append(
                                source.format(**values)))
        screen._notices = notices
        monkeypatch.setattr(screen._console, "fill_ai_entry",
                            lambda block, text: notices.append(
                                "spaCR AI: " + text))
        recorder = _Jobs()
        monkeypatch.setattr(screen._jobs, "submit", recorder.submit)
        screen._recorded = recorder
        monkeypatch.setattr(screen, "_report_summary_provider",
                            providers.hosted_provider)
        return screen

    def _fail(self, screen, relay, tb):
        relay.routes["/diagnose"] = [(200, {"advice": "Set cell_channel.",
                                            "model": "big", "tier": "best"})]
        relay.routes["/report"] = [(200, {"issue_number": 3, "issue_url": "u",
                                          "deduplicated": False})]
        screen._on_pipeline_error(tb)
        screen._settle_the_report(failed=True)
        for fn, on_done in list(screen._recorded.jobs):
            on_done(fn())

    @pytest.mark.parametrize("mode", list(EXPECTED))
    @pytest.mark.parametrize("kind", ["fault", "user_error"])
    def test_the_choice(self, screen, relay, mode, kind):
        cr.set_error_report_mode(mode)
        self._fail(screen, relay, FAULT_TB if kind == "fault" else USER_TB)
        explains, sends_fault, sends_user = EXPECTED[mode]
        sends = sends_fault if kind == "fault" else sends_user
        assert screen._failure_kind == kind
        assert bool(relay.calls_to("/diagnose")) is explains
        assert any(n.startswith("spaCR AI: Set cell_channel.")
                   for n in screen._notices) is explains
        reports = relay.calls_to("/report")
        assert bool(reports) is sends
        if sends:
            body = reports[0]["body"]
            assert body["kind"] == kind
            assert ("ai_summary" in body) is explains
            codes = [f.get("code") for f in body["frames"]]
            if mode == "all_code" and kind == "fault":
                assert codes == [None, "raise KeyError(name)"], (
                    "code only for the spaCR frame, never the user's")
            else:
                assert codes == [None] * len(codes)
            assert "secret_plate" not in json.dumps(body)

    def test_code_context_never_includes_user_code(self):
        summary = cr.public_summary(traceback_text=FAULT_TB,
                                    code_context=True)
        user, spacr_frame = summary["frames"]
        assert user["file"] == "<user-code>" and "code" not in user
        assert spacr_frame["code"] == "raise KeyError(name)"
        assert all(len(f.get("code", "")) <= cr.MAX_CODE_CHARS
                   for f in summary["frames"])


class TestARunRefusedBeforeItStarts:
    """The maintainer's case: a missing channel, refused before the run."""

    def test_the_refusal_is_explained_and_reported(self, qtbot,
                                                   qt_theme_applied,
                                                   monkeypatch, relay):
        from spacr.errors import raise_if_strict
        from spacr.qt.screens.app_screen import AppScreen

        screen = AppScreen("mask")
        qtbot.addWidget(screen)
        notices = []
        monkeypatch.setattr(screen._console, "append_notice",
                            lambda source, **values: notices.append(
                                source.format(**values)))
        recorder = _Jobs()
        monkeypatch.setattr(screen._jobs, "submit", recorder.submit)
        monkeypatch.setattr(screen, "_report_summary_provider",
                            providers.hosted_provider)
        relay.routes["/diagnose"] = [(200, {"advice": "Set cell_channel.",
                                            "model": "big", "tier": "best"})]
        relay.routes["/report"] = [(200, {"issue_number": 4, "issue_url": "u",
                                          "deduplicated": False})]

        screen._watch_for_refusals()
        raise_if_strict("At least one registered *_channel (for example "
                        "cell_channel or organelle_channel) must be set; no "
                        "masks can be generated.", settings={})
        screen._on_finished(True)

        assert any(n.startswith("✗ The run did not start: At least one "
                                "registered *_channel") for n in notices)
        assert not any(n.startswith("✓ Finished") for n in notices)
        assert len(recorder.jobs) == 1
        fn, on_done = recorder.jobs[0]
        on_done(fn())
        from PySide6.QtWidgets import QPlainTextEdit
        assert any(w.property("consoleContextKind") == "spacr_ai"
                   and w.toPlainText() == "Set cell_channel."
                   for w in screen._console.findChildren(QPlainTextEdit))
        body = relay.calls_to("/report")[0]["body"]
        assert body["kind"] == "user_error"
        assert body["exception_type"] == "ConfigurationError"
        assert "_channel" in body["message"]

    def test_a_clean_run_is_still_finished(self, qtbot, qt_theme_applied,
                                           monkeypatch):
        from spacr.qt.screens.app_screen import AppScreen

        screen = AppScreen("mask")
        qtbot.addWidget(screen)
        notices = []
        monkeypatch.setattr(screen._console, "append_notice",
                            lambda source, **values: notices.append(source))
        screen._watch_for_refusals()
        screen._on_finished(True)
        assert any(n.startswith("✓ Finished") for n in notices)


class TestTheUsageBar:
    """The slim bar under the chat box, for each provider."""

    DYNAMIC = {"ai": {"best_left": 3, "light_left": 10, "neurons_left": 600,
                      "share_neurons": 1000, "share_pct": 2.5,
                      "active_users": 40, "tier_now": "best"},
               "pool": "best", "pool_pct_left": 72.0,
               "resets_at": "2026-10-15T00:00:00Z"}

    def _when(self):
        return providers.local_reset_time("2026-10-15T00:00:00Z")

    def test_spacr_ai_shows_its_share(self):
        pct, text = providers.usage_bar(providers.HOSTED_PROVIDER_NAME,
                                        self.DYNAMIC)
        assert pct == 60
        assert text == (f"60% of your share left · 40 users sharing today · "
                        f"resets {self._when()}")

    def test_the_light_model_is_named(self):
        allowance = json.loads(json.dumps(self.DYNAMIC))
        allowance["ai"]["tier_now"] = "light"
        assert providers.usage_bar(providers.HOSTED_PROVIDER_NAME,
                                   allowance)[1].endswith(" · lighter model")

    def test_an_empty_share(self):
        allowance = json.loads(json.dumps(self.DYNAMIC))
        allowance["ai"]["neurons_left"] = 0
        assert providers.usage_bar(providers.HOSTED_PROVIDER_NAME,
                                   allowance) == (
            0, "Daily share used. Errors are still reported without an AI "
               "summary.")

    def test_an_older_relay_falls_back_to_the_estimate(self):
        old = {"ai": {"best_left": 4, "light_left": 10}, "pool": "best",
               "resets_at": "2026-10-15T00:00:00Z"}
        pct, text = providers.usage_bar(providers.HOSTED_PROVIDER_NAME, old)
        assert pct is None and text.startswith("4 of 5 best-model")

    @pytest.mark.parametrize("name, text", [
        ("claude", "Your own Claude account · limits set by Anthropic"),
        ("codex", "Your own ChatGPT account (Codex) · limits set by OpenAI"),
        ("gemini", "Your own Google account (Gemini) · limits set by Google"),
    ])
    def test_a_cli_has_no_bar_and_no_numbers(self, name, text):
        assert providers.usage_bar(name, self.DYNAMIC) == (None, text)

    def test_the_bar_in_the_console(self, qtbot, qt_theme_applied, relay):
        from PySide6.QtWidgets import QLabel, QProgressBar

        from spacr.qt.widgets.console_panel import ConsolePanel

        relay.answers = [(200, self.DYNAMIC)]
        panel = ConsolePanel()
        qtbot.addWidget(panel)
        bar = panel.findChild(QProgressBar, "AiUsageBar")
        text = panel.findChild(QLabel, "AiUsageText")
        panel.set_ai_provider(providers.HOSTED_PROVIDER_NAME)
        qtbot.waitUntil(lambda: "60%" in text.text(), timeout=5000)
        assert bar.value() == 60 and not bar.isHidden()
        panel.set_ai_provider("gemini")
        assert bar.isHidden()
        assert "limits set by Google" in text.text()
        panel.set_ai_provider(None)
        assert panel.findChild(object, "AiUsageRow").isHidden()
