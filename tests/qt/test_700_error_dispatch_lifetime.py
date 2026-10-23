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
    monkeypatch.setattr(cr, "_write_error_result", terminal.append)
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
