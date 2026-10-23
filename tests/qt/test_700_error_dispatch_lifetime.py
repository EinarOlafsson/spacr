"""Shared error results never reopen a closed or destroyed console owner."""
import threading

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QSettings
from PySide6.QtWidgets import QWidget
from shiboken6 import isValid

from spacr import crashreport as cr
from tests.test_700_structured_error_dispatch import configured, failure

_ORIGINAL_DELIVERY = cr._deliver_error_result


@pytest.mark.parametrize("owner_state", ["visible", "closed", "destroyed", "two-visible", "all-hidden"])
def test_report_finishes_without_resurrecting_its_owner(configured, qtbot, tmp_path, monkeypatch, owner_state):
    from spacr.qt import preferences
    from spacr.qt.ai import settings
    from spacr.qt.terms import record_agreement
    from spacr.qt.widgets.console_panel import ConsolePanel

    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", lambda: store)
    monkeypatch.setattr(settings, "_settings", lambda: store)
    record_agreement()
    monkeypatch.setattr(cr, "_deliver_error_result", _ORIGINAL_DELIVERY)
    terminal = []
    monkeypatch.setattr(cr, "_write_error_result", lambda outcome: terminal.append(outcome)
                        if not (outcome.get("pending") or outcome.get("quiet")) else None)
    relay, outcomes, ready = configured
    entered, release = threading.Event(), threading.Event()
    original_transport = cr._open_url
    def transport(request, timeout):
        if request.full_url.endswith("/diagnose"):
            entered.set()
            assert release.wait(5)
        return original_transport(request, timeout)
    monkeypatch.setattr(cr, "_open_url", transport)
    owner = QWidget()
    panel = ConsolePanel(parent=owner, follow_log=False)
    owner.resize(600, 500)
    panel.resize(600, 500)
    qtbot.addWidget(owner)
    owner.show()
    panel.show()
    shown = []
    original_show = panel._show_error_report_result
    def show(outcome, *args, **kwargs):
        if not outcome.get("pending"):
            shown.append(outcome)
        return original_show(outcome, *args, **kwargs)
    monkeypatch.setattr(panel, "_show_error_report_result", show)
    second_owner = second_panel = None
    second_shown = []
    if owner_state in ("two-visible", "all-hidden"):
        second_owner = QWidget()
        second_panel = ConsolePanel(parent=second_owner, follow_log=False)
        qtbot.addWidget(second_owner)
        second_owner.show()
        second_panel.show()
        second_original = second_panel._show_error_report_result
        def second_show(outcome, *args, **kwargs):
            if not outcome.get("pending"):
                second_shown.append(outcome)
            return second_original(outcome, *args, **kwargs)
        monkeypatch.setattr(second_panel, "_show_error_report_result", second_show)

    exc, tb = failure()
    worker = cr._dispatch_error(tb, exception=exc)
    assert worker is not None and entered.wait(5)
    if owner_state in ("closed", "destroyed", "all-hidden"):
        owner.close()
        if second_owner is not None:
            second_owner.close()
    if owner_state == "destroyed":
        owner.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert not isValid(owner) and not isValid(panel)
    release.set()
    worker.join(5)
    assert not worker.is_alive()
    QCoreApplication.processEvents()
    assert len(relay.calls_to("/diagnose")) == len(relay.calls_to("/report")) == 1
    summary = relay.calls_to("/report")[0]["body"]
    assert cr.saved_explanation(cr.explanation_signature(summary))["text"] == "Check the array dimensions."
    if owner_state in ("visible", "two-visible"):
        assert len(shown) + len(second_shown) == 1
        assert terminal == []
        if owner_state == "two-visible":
            assert shown == [] and len(second_shown) == 1
        text = (second_panel if owner_state == "two-visible" else panel).as_text()
        assert "Check the array dimensions." in text
        assert "https://github.com/EinarOlafsson/spacr/issues/12" in text
    else:
        assert shown == second_shown == []
        assert len(terminal) == 1
        if isValid(owner):
            assert not owner.isVisible()


def test_formatted_qt_log_display_does_not_report_a_plain_error_twice(configured, qtbot, tmp_path, monkeypatch):
    import logging
    from spacr.qt import preferences
    from spacr.qt.ai import settings
    from spacr.qt.terms import record_agreement
    from spacr.qt.logging_util import QtLogHandler
    from spacr.qt.widgets.console_panel import ConsolePanel
    from spacr.logging_util import _AutomaticErrorHandler

    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", lambda: store)
    monkeypatch.setattr(settings, "_settings", lambda: store)
    record_agreement()
    relay, outcomes, ready = configured
    panel = ConsolePanel(follow_log=False)
    qtbot.addWidget(panel)
    panel.show()
    dispatch = cr._dispatch_error
    workers = []
    def capture(*args, **kwargs):
        worker = dispatch(*args, **kwargs)
        workers.append(worker)
        return worker
    monkeypatch.setattr(cr, "_dispatch_error", capture)
    logger = logging.Logger("spacr.actual_background")
    logger.addHandler(_AutomaticErrorHandler(logging.ERROR))
    qt_handler = QtLogHandler(logging.ERROR)
    qt_handler.record_ready.connect(panel._on_log_record)
    logger.addHandler(qt_handler)
    logger.error("plain background error")
    assert len(workers) == 1
    workers[0].join(5)
    assert not workers[0].is_alive()
    assert ready.is_set()
    assert "plain background error" in panel.as_text()
    assert len(relay.calls_to("/diagnose")) == len(relay.calls_to("/report")) == 1


def test_none_blocks_stale_manual_chat_and_explain_provider(configured, qtbot, tmp_path, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.ai import settings, providers
    from spacr.qt.terms import record_agreement
    from spacr.qt.widgets.console_panel import ConsolePanel

    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", lambda: store)
    monkeypatch.setattr(settings, "_settings", lambda: store)
    record_agreement()
    panel = ConsolePanel(follow_log=False)
    qtbot.addWidget(panel)
    panel._current_provider_name = providers.HOSTED_PROVIDER_NAME
    preferences.set_preferred_provider("none")
    panel.set_ai_active(True)
    panel._input.setPlainText("Explain this")
    panel._on_submit()
    panel.open_error_flow("IndexError: channel outside image")
    assert panel._current_provider() is None
    assert panel._ai_thread is None
    assert configured[0].calls == []
    assert "No provider configured" in panel.as_text()


def test_response_speed_updates_the_headless_provider_arguments(configured, tmp_path, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.ai import settings
    from spacr.qt.terms import record_agreement

    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", lambda: store)
    monkeypatch.setattr(settings, "_settings", lambda: store)
    record_agreement()
    preferences.set_preferred_provider("claude")
    settings.set_response_speed("fast")
    policy, explain, send = cr._automatic_error_policy(cr.KIND_FAULT)
    assert policy["provider"] == "claude"
    assert policy["provider_args"] == ["--model", "haiku"]
    assert explain and send


def test_real_background_failure_transports_one_identity_to_the_screen(configured, qtbot, tmp_path, monkeypatch):
    import logging
    import sys
    import numpy as np
    from PySide6.QtCore import Slot
    from spacr.qt import preferences, bridge
    from spacr.qt.ai import settings
    from spacr.qt.job_runner import JobRunner
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.terms import record_agreement
    from spacr.logging_util import _AutomaticErrorHandler

    original_run = bridge.PipelineWorker.run
    class TracedWorker(bridge.PipelineWorker):
        """Inherit the existing public trace hook on the actual native worker."""
        @Slot()
        def run(self):
            """Delegate unchanged worker behavior and restore its prior hook."""
            previous = sys.gettrace()
            inherited = threading.gettrace() if previous is None else None
            if inherited is not None:
                sys.settrace(inherited)
            try:
                return original_run(self)
            finally:
                if inherited is not None:
                    sys.settrace(previous)
    monkeypatch.setattr(bridge, "PipelineWorker", TracedWorker)
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", lambda: store)
    monkeypatch.setattr(settings, "_settings", lambda: store)
    record_agreement()
    relay, outcomes, ready = configured
    captures = []
    original_dispatch = cr._dispatch_error
    def dispatch(*args, **kwargs):
        worker = original_dispatch(*args, **kwargs)
        captures.append((kwargs.get("module"), kwargs.get("event_id"),
                         getattr(worker, "_spacr_error_occurrence", ""),
                         kwargs.get("origin", "direct")))
        return worker
    monkeypatch.setattr(cr, "_dispatch_error", dispatch)
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    screen.show()
    original_handlers = list(bridge.LOG.handlers)
    monkeypatch.setattr(bridge.LOG, "handlers", original_handlers + [_AutomaticErrorHandler(logging.ERROR)])
    runner = JobRunner(screen, app_key="field_overview", user_visible=False)
    runner.job_failed.connect(screen._on_pipeline_error)
    failures = []
    runner.job_failed.connect(failures.append)
    def invalid_internal_index():
        return np.zeros((2, 2, 3), dtype=np.uint8)[..., 3]
    runner.submit(invalid_internal_index, lambda result: None)
    qtbot.waitUntil(lambda: bool(failures) and runner.active_jobs() == 0, timeout=15000)
    screen._settle_the_report(failed=True)
    assert ready.wait(5)
    assert len(relay.calls_to("/diagnose")) == len(relay.calls_to("/report")) == 1, captures
    assert len({item[1] for item in captures}) == 1
    body = relay.calls_to("/report")[0]["body"]
    assert body["exception_type"] == "IndexError" and body["kind"] == cr.KIND_FAULT
    assert any(frame["function"] == "invalid_internal_index" for frame in body["frames"])
    assert "axis 2" in failures[0]


@pytest.mark.parametrize("choice", ["none", "revoke-pending"])
def test_none_keeps_the_original_error_without_report_notifications(configured, qtbot, tmp_path, monkeypatch, choice):
    import io
    from spacr.qt import preferences
    from spacr.qt.ai import settings, providers
    from spacr.qt.terms import record_agreement
    from spacr.qt.widgets.console_panel import ConsolePanel, _TopicBar

    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", lambda: store)
    monkeypatch.setattr(settings, "_settings", lambda: store)
    preferences.set_issue_prompt_mode(preferences.ISSUE_PROMPT_ALWAYS)
    preferences.set_preferred_provider("none" if choice == "none" else providers.HOSTED_PROVIDER_NAME)
    settings.set_auto_file_issues(True)
    settings.set_route_errors_through_ai(True)
    record_agreement()
    cr.set_error_report_mode("all")
    monkeypatch.setattr(cr, "_deliver_error_result", _ORIGINAL_DELIVERY)
    terminal = io.StringIO()
    monkeypatch.setattr(cr.sys, "__stderr__", terminal)
    relay, outcomes, ready = configured
    entered, release = threading.Event(), threading.Event()
    original_transport = cr._open_url
    def transport(request, timeout):
        if request.full_url.endswith("/diagnose"):
            entered.set()
            assert release.wait(5)
        return original_transport(request, timeout)
    monkeypatch.setattr(cr, "_open_url", transport)
    workers = []
    original_dispatch = cr._dispatch_error
    def dispatch(*args, **kwargs):
        worker = original_dispatch(*args, **kwargs)
        if worker is not None:
            workers.append(worker)
        return worker
    monkeypatch.setattr(cr, "_dispatch_error", dispatch)
    panel = ConsolePanel(follow_log=False)
    qtbot.addWidget(panel)
    panel.show()
    try:
        panel.append_error("IndexError: original channel failure")
        if choice == "revoke-pending":
            qtbot.waitUntil(lambda: entered.is_set() and bool(panel._pending_report_entries), timeout=5000)
            preferences.set_preferred_provider("none")
            release.set()
        qtbot.waitUntil(lambda: bool(workers) and all(not worker.is_alive() for worker in workers), timeout=10000)
        QCoreApplication.processEvents()
        assert "IndexError: original channel failure" in panel.as_text()
        assert "issue #" not in panel.as_text()
        assert not panel._pending_report_entries
        entries = [panel._entries.itemAt(index).widget() for index in range(panel._entries.count())]
        assert not any(isinstance(widget, _TopicBar) and widget.text() == "spaCR AI" for widget in entries)
        assert terminal.getvalue() == ""
        assert len(relay.calls_to("/diagnose")) == (0 if choice == "none" else 1)
        reports = relay.calls_to("/report")
        assert len(reports) == 1 and reports[0]["body"]["summarize"] is False
        assert "ai_summary" not in reports[0]["body"]
    finally:
        release.set()
        for worker in workers:
            worker.join(5)


@pytest.mark.parametrize("failure_at", ["provider", "publish"])
def test_an_unexpected_reporting_failure_retires_the_pending_entry(configured, qtbot, tmp_path, monkeypatch, failure_at):
    from spacr.qt import preferences
    from spacr.qt.ai import settings, providers
    from spacr.qt.terms import record_agreement
    from spacr.qt.widgets import console_panel

    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", lambda: store)
    monkeypatch.setattr(settings, "_settings", lambda: store)
    record_agreement()
    monkeypatch.setattr(cr, "_deliver_error_result", _ORIGINAL_DELIVERY)
    def fail(*args, **kwargs):
        raise OSError("the reporting collaborator is unavailable")
    if failure_at == "provider":
        monkeypatch.setattr(providers, "explain_failure", fail)
    else:
        monkeypatch.setattr(cr, "_send_queued", fail)
    panel = console_panel.ConsolePanel(follow_log=False)
    qtbot.addWidget(panel)
    panel.show()
    workers = []
    original_dispatch = cr._dispatch_error
    def dispatch(*args, **kwargs):
        worker = original_dispatch(*args, **kwargs)
        if worker is not None:
            workers.append(worker)
        return worker
    monkeypatch.setattr(cr, "_dispatch_error", dispatch)
    panel.append_error("IndexError: original channel failure")
    qtbot.waitUntil(lambda: bool(workers) and all(not worker.is_alive() for worker in workers), timeout=10000)
    QCoreApplication.processEvents()
    assert "IndexError: original channel failure" in panel.as_text()
    assert not panel._pending_report_entries
    assert workers[0]._spacr_error_occurrence not in console_panel._ERROR_REPORT_RELAY.pending
    assert not any(dots._timer.isActive() for dots in panel.findChildren(console_panel._WorkingDots))
    relay = configured[0]
    assert len(relay.calls_to("/report")) == (1 if failure_at == "provider" else 0)
    if failure_at == "publish":
        assert len(cr.queued_summaries()) == 1


@pytest.mark.parametrize("provider_type,fast_model,deep_model,model_flag", [
    ("ClaudeCliProvider", "haiku", "opus", "--model"),
    ("CodexCliProvider", "gpt-5-mini", "gpt-5-pro", "--model"),
    ("GeminiCliProvider", "gemini-2.5-flash", "gemini-2.5-pro", "-m"),
])
def test_cli_chat_uses_captured_arguments_then_real_preferences(
        configured, tmp_path, monkeypatch, provider_type, fast_model, deep_model, model_flag):
    from spacr.qt.ai import providers, settings

    store = QSettings(str(tmp_path / "args.ini"), QSettings.IniFormat)
    monkeypatch.setattr(settings, "_settings", lambda: store)
    calls = []
    def process(argv, **kwargs):
        calls.append((list(argv), kwargs["provider"]))
        yield "answer"
    monkeypatch.setattr(providers, "_stream_process", process)
    provider = getattr(providers, provider_type)()
    settings.set_response_speed("fast")
    provider._reporting_args = settings.provider_args(provider.name)
    settings.set_response_speed("deep")
    messages = [{"role": "user", "content": "Explain the failure."}]
    assert list(provider.stream_chat(messages)) == ["answer"]
    assert calls[-1][0][calls[-1][0].index(model_flag) + 1] == fast_model
    assert deep_model not in calls[-1][0]
    del provider._reporting_args
    assert list(provider.stream_chat(messages)) == ["answer"]
    assert calls[-1][0][calls[-1][0].index(model_flag) + 1] == deep_model
    assert len(calls) == 2
    assert all(owner is provider for _, owner in calls)
    assert all(argv[0] == provider.name and "Explain the failure." in argv[2]
               for argv, _ in calls)
