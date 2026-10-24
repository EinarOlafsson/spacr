"""Consented central error capture without live network or native crashes."""
import io
import json
import logging
import threading
import traceback

import pytest

from spacr import crashreport as cr
from spacr.logging_util import _AutomaticErrorHandler, _ErrorStream
from tests.test_700_anonymous_error_reports import FakeRelay

_ORIGINAL_DELIVERY = cr._deliver_error_result


@pytest.fixture
def configured(tmp_path, monkeypatch, request):
    """Exercise routing with independent errors separated by a quiet minute.

    The storm-specific cohort drives close arrivals with its own mutable clock.
    """
    moments = iter(range(0, 10000000, 61))
    control = cr._ErrorBatcher(clock=lambda: next(moments),
                             schedule=lambda delay, callback: threading.Timer(delay, callback))
    monkeypatch.setattr(cr, "_ERROR_BATCHER", control)
    request.addfinalizer(lambda: control.timer.cancel() if control.timer is not None else None)
    monkeypatch.setattr(cr, "_reports_dir", lambda: tmp_path / "reports")
    monkeypatch.setattr(cr, "_ERROR_DISPATCH_SEEN", {})
    monkeypatch.setattr(cr, "_ERROR_OCCURRENCE_POLICIES", {})
    monkeypatch.setattr(cr, "_ERROR_RESULT_LISTENERS", [])
    monkeypatch.delenv(cr.NO_NETWORK_ENV, raising=False)
    fake = FakeRelay()
    fake.routes["/diagnose"] = [(200, {"advice": "Check the array dimensions.", "model": "local-test"})]
    monkeypatch.setattr(cr, "_open_url", fake)
    cr._update_state(affirmed_policy={
        "version": cr._REPORTING_TERMS_VERSION,
        "accepted_at": "2026-10-23T12:00:00+00:00",
        "explain": True, "auto_file": True, "prompt": "always", "provider": "",
    })
    cr.set_error_report_mode("all")
    outcomes = []
    ready = threading.Event()
    def delivered(outcome):
        if outcome.get("pending"):
            return
        outcomes.append(outcome)
        ready.set()
    monkeypatch.setattr(cr, "_deliver_error_result", delivered)
    return fake, outcomes, ready


def failure():
    try:
        raise IndexError("bad channel in /home/private_person/plate7/image.tif token=sk-abcdefghijklmnopqrstuv")
    except IndexError as exc:
        return exc, traceback.format_exc()


def joined(thread):
    assert thread is not None
    thread.join(5)
    assert not thread.is_alive()


def test_logging_console_and_terminal_share_one_sanitized_report(configured):
    relay, outcomes, ready = configured
    exc, tb = failure()
    record = logging.LogRecord("spacr.worker", logging.ERROR, __file__, 1,
                               "channel work failed", (), (type(exc), exc, exc.__traceback__))
    _AutomaticErrorHandler(logging.ERROR).emit(record)
    assert ready.wait(5)
    assert cr._dispatch_error(tb, traceback_text=tb, event_id=record._spacr_error_occurrence) is None
    assert len(relay.calls_to("/diagnose")) == len(relay.calls_to("/report")) == 1
    body = relay.calls_to("/report")[0]["body"]
    assert body["exception_type"] == "IndexError"
    assert body["frames"] and body["ai_summary"] == "Check the array dimensions."
    assert body["summarize"] is False
    assert cr.saved_explanation(cr.explanation_signature(body))["text"] == body["ai_summary"]
    for secret in ("private_person", "plate7", "image.tif", "sk-abcdefghijklmnopqrstuv", "/home/"):
        assert secret not in json.dumps(relay.calls)
    assert outcomes[0]["issue_url"].endswith("/12")
    assert cr.queued_summaries() == []


@pytest.mark.parametrize("change", ["no-consent", "notice-only", "bad-date", "off", "network"])
def test_unaffirmed_or_disabled_capture_never_contacts_transport(configured, change, monkeypatch):
    relay, outcomes, ready = configured
    state = cr._read_state()
    if change in ("no-consent", "notice-only"):
        state.pop("affirmed_policy")
        state["notice_shown"] = change == "notice-only"
    if change == "bad-date":
        state["affirmed_policy"]["accepted_at"] = "not-a-timestamp"
    cr._write_state(state)
    if change == "off":
        cr.set_error_report_mode("off")
    if change == "network":
        monkeypatch.setenv(cr.NO_NETWORK_ENV, "1")
    assert cr._dispatch_error("ERROR failed") is None
    assert relay.calls == outcomes == []
    assert cr.queued_summaries() == []


@pytest.mark.parametrize("choice", ["ask", "never", "no-auto-file", "summarize", "none"])
def test_diagnosis_and_issue_permissions_are_separate(configured, choice):
    relay, outcomes, ready = configured
    policy = cr._read_state()["affirmed_policy"]
    if choice in ("ask", "never"):
        policy["prompt"] = choice
    elif choice == "no-auto-file":
        policy["auto_file"] = False
    elif choice == "none":
        policy["provider"] = "none"
    cr._update_state(affirmed_policy=policy)
    if choice == "summarize":
        cr.set_error_report_mode("summarize")
    exc, tb = failure()
    joined(cr._dispatch_error(tb, exception=exc))
    assert len(relay.calls_to("/diagnose")) == (0 if choice == "none" else 1)
    assert len(relay.calls_to("/report")) == (1 if choice == "none" else 0)
    if choice == "none":
        assert outcomes[0].get("explanation") is None
        body = relay.calls_to("/report")[0]["body"]
        assert body["summarize"] is False and "ai_summary" not in body
        assert outcomes[0]["explained"] is False


def test_structured_severity_and_split_terminal_lines_not_incidental_words(configured):
    relay, outcomes, ready = configured
    sink = io.StringIO()
    stream = _ErrorStream(sink)
    stream.write("There was no error in this successful step.\n")
    record = logging.LogRecord("spacr.worker", logging.WARNING, __file__, 1, "ERROR is quoted", (), None)
    _AutomaticErrorHandler(logging.ERROR).emit(record)
    assert relay.calls == []
    stream.write("ERR")
    stream.write("OR: actual terminal failure\n")
    assert ready.wait(5)
    assert len(relay.calls_to("/report")) == 1
    assert sink.getvalue().endswith("ERROR: actual terminal failure\n")


def test_reporter_logging_cannot_recursively_submit(configured, monkeypatch):
    relay, outcomes, ready = configured
    original = cr._open_url
    nested = []
    def transport(request, timeout):
        nested.append(cr._dispatch_error("ERROR reporter transport failure"))
        return original(request, timeout)
    monkeypatch.setattr(cr, "_open_url", transport)
    joined(cr._dispatch_error("ERROR original failure"))
    assert nested and all(value is None for value in nested)
    assert len(relay.calls_to("/report")) == 1


def test_native_record_is_deferred_until_consent_then_consumed_once(configured, tmp_path):
    relay, outcomes, ready = configured
    policy = cr._read_state()["affirmed_policy"]
    cr._update_state(affirmed_policy={})
    path = tmp_path / "crash.log"
    path.write_text('Fatal Python error: Segmentation fault\n\nCurrent thread:\n  File "/home/private_person/source.py", line 4 in run\n', encoding="utf-8")
    assert cr._consume_native_crash(path) is None
    assert relay.calls == []
    cr._update_state(affirmed_policy=policy)
    joined(cr._consume_native_crash(path))
    assert cr._consume_native_crash(path) is None
    assert len(relay.calls_to("/report")) == 1
    body = relay.calls_to("/report")[0]["body"]
    assert body["exception_type"] == "NativeCrash"
    assert body["message"] == "Native process termination: Segmentation fault"
    assert "private_person" not in json.dumps(body)


def test_explicit_native_optout_acknowledges_without_replay(configured, tmp_path):
    relay, outcomes, ready = configured
    path = tmp_path / "crash.log"
    path.write_text("Fatal Python error: Aborted\n", encoding="utf-8")
    cr.set_error_report_mode("off")
    assert cr._consume_native_crash(path) is None
    cr.set_error_report_mode("all")
    assert cr._consume_native_crash(path) is None
    assert relay.calls == []


def test_known_validation_and_cancel_are_not_software_faults(configured):
    from concurrent.futures import CancelledError
    from spacr.errors import ConfigurationError
    relay, outcomes, ready = configured
    cr.set_error_report_mode("faults")
    joined(cr._dispatch_error("Invalid channel", exception=ConfigurationError("Invalid channel")))
    assert len(relay.calls_to("/diagnose")) == 1
    assert relay.calls_to("/report") == []
    assert outcomes[0]["kind"] == cr.KIND_USER_ERROR
    record = logging.LogRecord("spacr.worker", logging.INFO, __file__, 1,
                               "cancelled", (), (CancelledError, CancelledError(), None))
    _AutomaticErrorHandler(logging.ERROR).emit(record)
    assert len(relay.calls_to("/diagnose")) == 1


def test_native_handler_installation_only_prepares_faulthandler(configured, tmp_path, monkeypatch):
    import faulthandler
    relay, outcomes, ready = configured
    enabled = []
    monkeypatch.setattr(cr, "_NATIVE_CAPTURE_HANDLE", None)
    monkeypatch.setattr(cr, "_NATIVE_EXCEPTION_HOOK", object())
    monkeypatch.setattr(faulthandler, "enable", lambda **kwargs: enabled.append(kwargs))
    path = tmp_path / "fatal.log"
    assert cr._install_native_error_capture(path) == str(path)
    handle = cr._NATIVE_CAPTURE_HANDLE
    assert enabled == [{"file": handle, "all_threads": True}]
    assert handle.fileno() >= 0
    assert "spaCR session" in path.read_text()
    assert relay.calls == outcomes == []
    handle.close()


def test_a_new_native_fatal_record_is_not_lost_to_previous_record_dedup(configured, tmp_path):
    relay, outcomes, ready = configured
    path = tmp_path / "crash.log"
    fatal = "Fatal Python error: Segmentation fault\n"
    path.write_text(fatal, encoding="utf-8")
    joined(cr._consume_native_crash(path))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(fatal)
    joined(cr._consume_native_crash(path))
    assert cr._consume_native_crash(path) is None
    assert len(relay.calls_to("/report")) == 2


def test_oversized_unconsented_native_record_keeps_a_bounded_pending_tail(configured, tmp_path, monkeypatch):
    import faulthandler
    relay, outcomes, ready = configured
    policy = cr._read_state()["affirmed_policy"]
    cr._update_state(affirmed_policy={})
    monkeypatch.setattr(cr, "_NATIVE_CAPTURE_HANDLE", None)
    monkeypatch.setattr(cr, "_NATIVE_EXCEPTION_HOOK", object())
    monkeypatch.setattr(faulthandler, "enable", lambda **kwargs: None)
    path = tmp_path / "fatal.log"
    path.write_text(" " * cr.MAX_LOG_BYTES + "\nFatal Python error: Aborted\n", encoding="utf-8")
    cr._install_native_error_capture(path)
    handle = cr._NATIVE_CAPTURE_HANDLE
    header = path.read_text().split("\n=== spaCR session ")[-1]
    assert path.stat().st_size == cr.MAX_LOG_BYTES + len(("\n=== spaCR session " + header).encode())
    assert "Fatal Python error: Aborted" in path.read_text()
    assert relay.calls == []
    cr._update_state(affirmed_policy=policy)
    joined(cr._consume_native_crash(path))
    assert len(relay.calls_to("/report")) == 1
    handle.close()


def test_an_all_mode_typed_error_is_reported_as_a_user_error(configured):
    from spacr.errors import ConfigurationError
    relay, outcomes, ready = configured
    exc = ConfigurationError("Choose an available channel")
    record = logging.LogRecord("spacr.reader", logging.ERROR, __file__, 1,
                               str(exc), (), (type(exc), exc, None))
    _AutomaticErrorHandler(logging.ERROR).emit(record)
    assert ready.wait(5)
    assert len(relay.calls_to("/report")) == 1
    assert relay.calls_to("/report")[0]["body"]["kind"] == cr.KIND_USER_ERROR


def test_revoking_automatic_filing_while_waiting_for_send_lock_prevents_post(configured, monkeypatch):
    relay, outcomes, ready = configured
    entered = threading.Event()
    original = cr._send_queued
    def send(*args, **kwargs):
        entered.set()
        return original(*args, **kwargs)
    monkeypatch.setattr(cr, "_send_queued", send)
    cr._SEND_LOCK.acquire()
    try:
        worker = cr._dispatch_error("ERROR original failure")
        assert entered.wait(5)
        policy = cr._read_state()["affirmed_policy"]
        policy["auto_file"] = False
        cr._update_state(affirmed_policy=policy)
    finally:
        cr._SEND_LOCK.release()
    joined(worker)
    assert relay.calls_to("/report") == []
    assert cr.queued_summaries() == []
    assert outcomes[0]["status"] == cr.REPORT_DISABLED


def test_emitter_does_not_wait_for_sanitization_or_route_to_a_new_provider(configured, monkeypatch):
    relay, outcomes, ready = configured
    entered, release = threading.Event(), threading.Event()
    main_thread = threading.get_ident()
    original = cr.public_summary
    def summary(*args, **kwargs):
        assert threading.get_ident() != main_thread
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(cr, "public_summary", summary)
    worker = cr._dispatch_error("ERROR captured with original provider")
    assert worker is not None and entered.wait(5)
    assert worker.is_alive()
    policy = cr._read_state()["affirmed_policy"]
    policy["provider"] = "codex"
    cr._update_state(affirmed_policy=policy)
    release.set()
    joined(worker)
    assert relay.calls_to("/diagnose") == []
    assert len(relay.calls_to("/report")) == 1
    assert "ai_summary" not in relay.calls_to("/report")[0]["body"]


def test_failing_capture_cannot_hide_original_stream_or_bridge_output(configured, monkeypatch):
    from spacr.qt.bridge import _StreamRedirector
    def broken(*args, **kwargs):
        raise RuntimeError("reporter hook unavailable")
    monkeypatch.setattr(cr, "_capture_error_line", broken)
    original = io.StringIO()
    stream = _ErrorStream(original)
    assert stream.write("ERROR original line\n") == len("ERROR original line\n")
    stream.write("ERROR partial")
    stream.flush()
    assert original.getvalue() == "ERROR original line\nERROR partial"
    received = []
    redirected = _StreamRedirector(received.append)
    redirected.write("ERROR still visible\n")
    assert received == ["ERROR still visible\n"]


def test_headless_dispatch_diagnoses_and_reports_without_importing_qt(tmp_path):
    import os
    import subprocess
    import sys
    script = '''
import importlib.abc, json, sys
class NoQt(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.startswith("PySide6"):
            raise AssertionError("headless reporting imported Qt")
sys.meta_path.insert(0, NoQt())
from spacr import crashreport as cr
cr._update_state(mode="all", affirmed_policy={"version": cr._REPORTING_TERMS_VERSION,
    "accepted_at": "2026-10-23T12:00:00+00:00", "explain": True,
    "auto_file": True, "prompt": "always", "provider": ""})
calls=[]
def post(route, payload, timeout):
    calls.append(route)
    return (200, {"advice": "Check array dimensions.", "model":"fake"}) if route=="/diagnose" else (200, {"issue_number":1,"issue_url":"https://github.com/EinarOlafsson/spacr/issues/1"})
cr._post_json=post
cr._deliver_error_result=lambda outcome: None
thread=cr._dispatch_error("ERROR standalone failure")
assert thread is not None
thread.join(5)
assert not thread.is_alive()
assert calls==["/diagnose","/report"],calls
assert not any(name.startswith("PySide6") for name in sys.modules)
print("headless PASS")
'''
    env = dict(os.environ, SPACR_HOME=str(tmp_path), SPACR_NO_NETWORK="")
    result = subprocess.run([sys.executable, "-c", script], env=env,
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "headless PASS" in result.stdout


def test_explicit_none_never_discovers_or_resolves_an_ai_backend(configured, monkeypatch):
    from spacr.qt.ai import providers
    relay, outcomes, ready = configured
    policy = cr._read_state()["affirmed_policy"]
    policy["provider"] = "none"
    cr._update_state(affirmed_policy=policy)
    calls = []
    monkeypatch.setattr(type(providers._HOSTED), "is_configured", lambda self: calls.append("hosted") or True)
    for provider in providers._PROVIDERS:
        monkeypatch.setattr(type(provider), "is_configured", lambda self: calls.append("vendor") or True)
    assert providers.default_provider("none") is None
    assert providers.default_provider() is None
    assert providers.configured_providers() == []
    assert providers.effective_provider_name("none") == "none"
    assert calls == []
    joined(cr._dispatch_error("ERROR report without provider"))
    assert relay.calls_to("/diagnose") == []
    assert len(relay.calls_to("/report")) == 1


def test_a_printed_terminal_traceback_preserves_its_type_frames_and_privacy(configured):
    relay, outcomes, ready = configured
    exc, tb = failure()
    sink = io.StringIO()
    stream = _ErrorStream(sink)
    for line in tb.splitlines(keepends=True):
        stream.write(line)
    assert ready.wait(5)
    assert sink.getvalue() == tb
    assert len(relay.calls_to("/report")) == 1
    body = relay.calls_to("/report")[0]["body"]
    assert body["exception_type"] == "IndexError" and body["frames"]
    assert "private_person" not in json.dumps(body)


def test_closed_original_terminal_does_not_raise_or_capture_reporter_output(configured, monkeypatch):
    import sys
    stream = io.StringIO()
    stream.close()
    monkeypatch.setattr(sys, "__stderr__", stream)
    cr._write_error_result({"explanation": {"text": "Saved advice"},
                            "issue_url": "https://github.com/EinarOlafsson/spacr/issues/12"})
    assert not getattr(cr._ERROR_DISPATCH_LOCAL, "active", False)
    assert configured[0].calls == []


def test_queued_automatic_retry_respects_revoked_source_code_permission(configured):
    relay, outcomes, ready = configured
    cr.set_error_report_mode("all_code")
    relay.routes["/report"] = [OSError("offline")]
    tb = ('Traceback (most recent call last):\n'
          '  File "/opt/env/lib/python3.11/site-packages/spacr/measure.py", line 88, in measure_crop\n'
          '    raise IndexError("bad channel")\n'
          'IndexError: bad channel\n')
    joined(cr._dispatch_error(tb, traceback_text=tb))
    assert cr.queued_summaries()[0]["frames"][0]["code"] == 'raise IndexError("bad channel")'
    cr.set_error_report_mode("all")
    relay.routes["/report"] = [(200, {"issue_number": 12,
                                    "issue_url": "https://github.com/EinarOlafsson/spacr/issues/12"})]
    assert cr.flush_report_queue()["sent"] == 1
    assert all("code" not in frame for frame in relay.calls_to("/report")[-1]["body"]["frames"])


def test_later_real_error_occurrences_report_again_and_reuse_saved_advice(configured, monkeypatch):
    relay, outcomes, ready = configured
    workers = []
    dispatch = cr._dispatch_error
    def track(*args, **kwargs):
        worker = dispatch(*args, **kwargs)
        if worker is not None:
            workers.append(worker)
        return worker
    monkeypatch.setattr(cr, "_dispatch_error", track)
    exc, tb = failure()
    record = logging.LogRecord("spacr.worker", logging.ERROR, __file__, 1,
                               "channel work failed", (), (type(exc), exc, exc.__traceback__))
    handler = _AutomaticErrorHandler(logging.ERROR)
    handler.emit(record)
    joined(workers[-1])
    previous_workers = len(workers)
    handler.emit(record)
    assert len(workers) == previous_workers
    assert cr._dispatch_error(tb, traceback_text=tb, event_id=record._spacr_error_occurrence) is None
    assert cr._dispatch_error(tb, traceback_text=tb, event_id=record._spacr_error_occurrence) is None
    assert len(relay.calls_to("/report")) == 1
    next_exc, next_tb = failure()
    next_record = logging.LogRecord("spacr.worker", logging.ERROR, __file__, 1,
                                    "channel work failed", (), (type(next_exc), next_exc, next_exc.__traceback__))
    handler.emit(next_record)
    joined(workers[-1])
    assert len(relay.calls_to("/report")) == 2
    assert len(relay.calls_to("/diagnose")) == 1
    assert relay.calls_to("/report")[0]["body"]["ai_summary"] == relay.calls_to("/report")[1]["body"]["ai_summary"]


def test_late_display_cannot_opt_an_originally_unconsented_error_in(configured):
    relay, outcomes, ready = configured
    accepted = cr._read_state()["affirmed_policy"]
    cr._update_state(affirmed_policy={})
    exc, tb = failure()
    record = logging.LogRecord("spacr.worker", logging.ERROR, __file__, 1,
                               "channel work failed", (), (type(exc), exc, exc.__traceback__))
    _AutomaticErrorHandler(logging.ERROR).emit(record)
    cr._update_state(affirmed_policy=accepted)
    assert cr._dispatch_error(tb, traceback_text=tb, event_id=record._spacr_error_occurrence) is None
    assert relay.calls == []
    joined(cr._dispatch_error(tb, traceback_text=tb))
    assert len(relay.calls_to("/report")) == 1


def test_an_independent_terminal_occurrence_is_not_a_previous_log_display(configured, monkeypatch):
    relay, outcomes, ready = configured
    workers = []
    dispatch = cr._dispatch_error
    def track(*args, **kwargs):
        worker = dispatch(*args, **kwargs)
        if worker is not None:
            workers.append(worker)
        return worker
    monkeypatch.setattr(cr, "_dispatch_error", track)
    exc, tb = failure()
    record = logging.LogRecord("spacr.worker", logging.ERROR, __file__, 1,
                               "channel work failed", (), (type(exc), exc, exc.__traceback__))
    _AutomaticErrorHandler(logging.ERROR).emit(record)
    joined(workers[-1])
    stream = _ErrorStream(io.StringIO())
    stream.write(tb)
    joined(workers[-1])
    assert len(relay.calls_to("/report")) == 2
    assert [call["body"]["module"] for call in relay.calls_to("/diagnose")] == [
        "spacrworker", "background"]


def test_reraising_the_same_exception_freezes_a_new_occurrence_policy(configured):
    relay, outcomes, ready = configured
    accepted = cr._read_state()["affirmed_policy"]
    cr._update_state(affirmed_policy={})
    exc, tb = failure()
    first = logging.LogRecord("spacr.worker", logging.ERROR, __file__, 1,
                              "channel work failed", (), (type(exc), exc, exc.__traceback__))
    handler = _AutomaticErrorHandler(logging.ERROR)
    handler.emit(first)
    old_token = first._spacr_error_occurrence
    cr._update_state(affirmed_policy=accepted)
    try:
        raise exc
    except IndexError as raised:
        second = logging.LogRecord("spacr.worker", logging.ERROR, __file__, 1,
                                   "channel work failed", (), (type(raised), raised, raised.__traceback__))
        handler.emit(second)
    assert second._spacr_error_occurrence != old_token
    assert ready.wait(5)
    assert len(relay.calls_to("/report")) == 1
    assert cr._dispatch_error(tb, traceback_text=tb, event_id=old_token) is None
    assert cr._dispatch_error(tb, traceback_text=tb, event_id=second._spacr_error_occurrence) is None


def test_none_reports_without_presenting_a_terminal_result(configured, monkeypatch):
    relay, outcomes, ready = configured
    policy = cr._read_state()["affirmed_policy"]
    policy["provider"] = "none"
    cr._update_state(affirmed_policy=policy)
    destination, original = io.StringIO(), io.StringIO()
    monkeypatch.setattr(cr.sys, "__stderr__", destination)
    monkeypatch.setattr(cr, "_deliver_error_result", _ORIGINAL_DELIVERY)
    workers = []
    dispatch = cr._dispatch_error
    def capture(*args, **kwargs):
        worker = dispatch(*args, **kwargs)
        if worker is not None:
            workers.append(worker)
        return worker
    monkeypatch.setattr(cr, "_dispatch_error", capture)
    _ErrorStream(original).write("ERROR original channel failure\n")
    assert len(workers) == 1
    joined(workers[0])
    assert original.getvalue() == "ERROR original channel failure\n"
    assert destination.getvalue() == ""
    assert relay.calls_to("/diagnose") == []
    reports = relay.calls_to("/report")
    assert len(reports) == 1 and reports[0]["body"]["summarize"] is False
    assert "ai_summary" not in reports[0]["body"]


@pytest.mark.parametrize("expiry", ["capacity", "ttl"])
def test_retired_transport_cannot_resubmit_but_a_new_error_can(configured, monkeypatch, expiry):
    import time
    from types import SimpleNamespace

    relay, outcomes, ready = configured
    policy = cr._read_state()["affirmed_policy"]
    policy["provider"] = "none"
    cr._update_state(affirmed_policy=policy)
    threads = []
    original_dispatch = cr._dispatch_error
    def dispatch(*args, **kwargs):
        result = original_dispatch(*args, **kwargs)
        threads.append(result)
        return result
    monkeypatch.setattr(cr, "_dispatch_error", dispatch)
    handler = _AutomaticErrorHandler(logging.ERROR)
    def emit(number):
        record = logging.LogRecord("spacr.retirement", logging.ERROR, __file__, 1,
                                   f"Real error occurrence {number}", (), None)
        handler.emit(record)
        joined(threads[-1])
        return record
    first = emit(0)
    if expiry == "capacity":
        for number in range(1, 257):
            emit(number)
        expected = 257
    else:
        original_monotonic = time.monotonic
        monkeypatch.setattr(cr, "time", SimpleNamespace(
            time=time.time,
            monotonic=lambda: original_monotonic() + cr.QUEUE_RETENTION_S + 1))
        expected = 1
    assert original_dispatch(first.getMessage(), event_id=first._spacr_error_occurrence) is None
    assert len(relay.calls_to("/report")) == expected
    emit("fresh")
    assert len(relay.calls_to("/report")) == expected + 1
    assert relay.calls_to("/diagnose") == []
    assert all(call["body"]["summarize"] is False for call in relay.calls_to("/report"))
    assert all(outcome["quiet"] for outcome in outcomes)


@pytest.fixture
def uncaught(configured, monkeypatch):
    import sys

    relay, outcomes, ready = configured
    policy = cr._read_state()["affirmed_policy"]
    policy["provider"] = "none"
    cr._update_state(affirmed_policy=policy)
    monkeypatch.setattr(cr, "report_exception", lambda *args, **kwargs: "")
    sink = io.StringIO()
    dispatches = []
    original_dispatch = cr._dispatch_error
    def dispatch(*args, **kwargs):
        result = original_dispatch(*args, **kwargs)
        dispatches.append((dict(kwargs), result))
        return result
    monkeypatch.setattr(cr, "_dispatch_error", dispatch)
    return relay, sink, dispatches


def test_uncaught_traceback_forwards_its_occurrence_but_an_independent_error_is_new(
        uncaught, monkeypatch, tmp_path):
    import sys

    relay, sink, dispatches = uncaught
    stream = _ErrorStream(sink)
    monkeypatch.setattr(sys, "stderr", stream)
    assert sys.stderr is stream
    def previous(kind, value, tb):
        sys.stderr.write("ERROR independent hook failure\n")
        sys.__excepthook__(kind, value, tb)
    monkeypatch.setattr(sys, "excepthook", previous)
    hook = cr.install_excepthook(tmp_path, app="measure")
    exc, _ = failure()
    hook(type(exc), exc, exc.__traceback__)
    for _, worker in dispatches:
        if worker is not None:joined(worker)
    assert len(relay.calls_to("/report")) == 2
    assert relay.calls_to("/diagnose") == []
    reports = [call["body"] for call in relay.calls_to("/report")]
    assert sum(body["exception_type"] == "IndexError" for body in reports) == 1
    assert sum(body["exception_type"] == "LoggedError" for body in reports) == 1
    original = next(worker for kwargs, worker in dispatches if kwargs.get("exception") is exc)
    printed = [(kwargs, worker) for kwargs, worker in dispatches if kwargs.get("traceback_text")]
    assert len(printed) == 1
    assert printed[0][0]["event_id"] == original._spacr_error_occurrence
    assert printed[0][1] is None
    assert "ERROR independent hook failure" in sink.getvalue()
    assert "IndexError: bad channel in /home/private_person/plate7/image.tif" in sink.getvalue()
    assert not hasattr(cr._ERROR_DISPATCH_LOCAL, "traceback_occurrence")


def test_excepthook_restores_existing_context_when_the_previous_hook_raises(
        uncaught, monkeypatch, tmp_path):
    import sys

    relay, sink, dispatches = uncaught
    stream = _ErrorStream(sink)
    monkeypatch.setattr(sys, "stderr", stream)
    assert sys.stderr is stream
    monkeypatch.setattr(cr._ERROR_DISPATCH_LOCAL, "traceback_occurrence", None, raising=False)
    def previous(kind, value, tb):
        sys.__excepthook__(kind, value, tb)
        raise RuntimeError("previous hook failed")
    monkeypatch.setattr(sys, "excepthook", previous)
    hook = cr.install_excepthook(tmp_path)
    exc, _ = failure()
    with pytest.raises(RuntimeError, match="previous hook failed"):
        hook(type(exc), exc, exc.__traceback__)
    for _, worker in dispatches:
        if worker is not None:joined(worker)
    assert cr._ERROR_DISPATCH_LOCAL.traceback_occurrence is None
    assert len(relay.calls_to("/report")) == 1
    assert "IndexError:" in sink.getvalue()


def test_keyboard_interrupt_delegates_without_reporting_and_restores_outer_context(
        uncaught, monkeypatch, tmp_path):
    import sys

    relay, sink, dispatches = uncaught
    stream = _ErrorStream(sink)
    monkeypatch.setattr(sys, "stderr", stream)
    assert sys.stderr is stream
    outer = cr._new_error_occurrence()
    monkeypatch.setattr(cr._ERROR_DISPATCH_LOCAL, "traceback_occurrence", outer, raising=False)
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    hook = cr.install_excepthook(tmp_path)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt as exc:
        hook(type(exc), exc, exc.__traceback__)
    for _, worker in dispatches:
        if worker is not None:joined(worker)
    assert cr._ERROR_DISPATCH_LOCAL.traceback_occurrence == outer
    assert "KeyboardInterrupt" in sink.getvalue()
    assert relay.calls == []


def test_hook_transport_cannot_enable_a_previously_unconsented_occurrence(
        uncaught, monkeypatch, tmp_path):
    import sys

    relay, sink, dispatches = uncaught
    stream = _ErrorStream(sink)
    monkeypatch.setattr(sys, "stderr", stream)
    assert sys.stderr is stream
    agreed = cr._read_state()["affirmed_policy"]
    cr._update_state(affirmed_policy={})
    def previous(kind, value, tb):
        cr._update_state(affirmed_policy=agreed)
        sys.__excepthook__(kind, value, tb)
    monkeypatch.setattr(sys, "excepthook", previous)
    hook = cr.install_excepthook(tmp_path)
    exc, _ = failure()
    hook(type(exc), exc, exc.__traceback__)
    for _, worker in dispatches:
        if worker is not None:joined(worker)
    assert relay.calls == []
    assert not hasattr(cr._ERROR_DISPATCH_LOCAL, "traceback_occurrence")
    sys.stderr.write("ERROR newly consented independent error\n")
    joined(dispatches[-1][1])
    assert len(relay.calls_to("/report")) == 1
    assert relay.calls_to("/diagnose") == []


def test_native_session_capture_replays_its_own_public_versions_and_fixed_sources(configured, tmp_path, monkeypatch):
    import faulthandler
    import hashlib
    relay, outcomes, ready = configured
    relay.routes['/health'] = [(200, {'capabilities': ['native_session_v1']})]
    monkeypatch.setattr(cr, '_NATIVE_CAPTURE_HANDLE', None)
    monkeypatch.setattr(cr, '_NATIVE_EXCEPTION_HOOK', object())
    monkeypatch.setattr(faulthandler, 'enable', lambda **kwargs: None)
    monkeypatch.setattr(cr, '_spacr_version', lambda: '1.2.3')
    monkeypatch.setattr(cr.platform, 'python_version', lambda: '3.12.11')
    path = tmp_path / 'fatal.log'
    assert cr._install_native_error_capture(path) == str(path)
    handle = cr._NATIVE_CAPTURE_HANDLE
    header = json.loads(path.read_text().split('=== spaCR session ')[1].split(' ===')[0])
    assert header['spacr_version'] == '1.2.3'
    assert header['python'] == '3.12.11'
    assert cr._parse_reset(header['started_at']) is not None
    assert set(header['source_sha256']) == {'package', 'app', 'gc_policy', 'crashreport'}
    assert header['source_sha256']['crashreport'] == hashlib.sha256(cr.Path(cr.__file__).read_bytes()).hexdigest()
    with path.open('a') as fatal:
        fatal.write('Fatal Python error: Segmentation fault\n\nCurrent thread:\n  File "/home/private_person/closing.py", line 41 in closeEvent\n')
    handle.close()
    monkeypatch.setattr(cr, '_spacr_version', lambda: '9.9.9')
    monkeypatch.setattr(cr.platform, 'python_version', lambda: '3.14.0')
    joined(cr._consume_native_crash(path))
    body = relay.calls_to('/report')[0]['body']
    assert body['native_session'] == header
    assert body['spacr_version'] == '9.9.9' and body['python'] == '3.14.0'
    assert body['frames'][-1]['function'] == 'closeEvent'
    assert cr._consume_native_crash(path) is None
    assert 'private_person' not in json.dumps(body) and str(tmp_path) not in json.dumps(body)


@pytest.mark.parametrize('prefix', ['', '=== spaCR started (pid 123) ===\n', '=== spaCR session {"known":true\n', '=== spaCR session null ===\n'])
def test_native_legacy_or_truncated_session_is_explicitly_unknown(configured, tmp_path, prefix):
    relay, outcomes, ready = configured
    relay.routes['/health'] = [(200, {'capabilities': ['native_session_v1']})]
    path = tmp_path / 'fatal.log'
    path.write_text(prefix + 'Fatal Python error: Aborted\n')
    joined(cr._consume_native_crash(path))
    assert relay.calls_to('/report')[0]['body']['native_session'] == {'known': False}


def test_native_replay_does_not_borrow_a_following_session_and_retains_previous_header(configured, tmp_path):
    relay, outcomes, ready = configured
    relay.routes['/health'] = [(200, {'capabilities': ['native_session_v1']})]
    first = {'known': True, 'id': '14cbd541-b9a1-41f4-8f50-4b53ba84755a',
             'started_at': '2026-10-23T18:00:00Z', 'spacr_version': '1.2.3',
             'python': '3.12.11', 'qt': '6.11.2', 'source_sha256': {'app': 'a' * 64}}
    second = dict(first, id='6ae2c2af-3e47-4bf8-9a19-f346ba655d34', python='3.14.0')
    path = tmp_path / 'fatal.log'
    path.write_text('=== spaCR session ' + json.dumps(first) + ' ===\n')
    assert cr._consume_native_crash(path) is None
    with path.open('a') as handle:
        handle.write('Fatal Python error: Aborted\n\n=== spaCR session ' + json.dumps(second) + ' ===\n')
    joined(cr._consume_native_crash(path))
    assert relay.calls_to('/report')[0]['body']['native_session'] == first
    with path.open('a') as handle:
        handle.write('Fatal Python error: Aborted\n')
    joined(cr._consume_native_crash(path))
    assert relay.calls_to('/report')[1]['body']['native_session'] == {'known': False}
    assert cr._consume_native_crash(path) is None


@pytest.mark.parametrize('capabilities', [[], ['native_session_v1']])
def test_native_protocol_capability_is_required_before_sending_optional_field(configured, tmp_path, capabilities):
    relay, outcomes, ready = configured
    relay.routes['/health'] = [(200, {'capabilities': capabilities})]
    path = tmp_path / 'fatal.log'
    path.write_text('Fatal Python error: Aborted\n')
    joined(cr._consume_native_crash(path))
    body = relay.calls_to('/report')[0]['body']
    assert ('native_session' in body) == bool(capabilities)
    if not capabilities:
        assert 'not the crashed build' in body['user_note']
    assert all(call['method'] == 'GET' for call in relay.calls_to('/health'))


def test_native_session_rejects_arbitrary_paths_keys_and_digest_values(configured, tmp_path):
    source = {'known': True, 'id': '14cbd541-b9a1-41f4-8f50-4b53ba84755a',
              'started_at': '2026-10-23T18:00:00Z', 'spacr_version': '1.2.3',
              'python': '3.12.11', 'qt': '6.11.2', 'source_sha256': {'app': 'a' * 64}}
    assert cr._validated_summary({'native_session': source})['native_session'] == source
    for malformed in (dict(source, hostname='private-host'),
                      dict(source, source_sha256={str(tmp_path / 'app.py'): 'a' * 64}),
                      dict(source, source_sha256={'app': 'sk-abcdefghijklmnopqrstuv'}),
                      dict(source, python=str(tmp_path / 'python')),
                      dict(source, started_at='2026-02-31T00:00:00Z')):
        assert cr._validated_summary({'native_session': malformed})['native_session'] == {'known': False}


@pytest.mark.parametrize('change', ['off', 'consent', 'network', 'none'])
def test_native_health_lookup_cannot_override_a_new_optout(configured, tmp_path, monkeypatch, change):
    relay, outcomes, ready = configured
    entered = threading.Event()
    release = threading.Event()
    original = cr._open_url
    def gated(request, timeout):
        if request.full_url.endswith('/health'):
            entered.set()
            assert release.wait(5)
        return original(request, timeout)
    relay.routes['/health'] = [(200, {'capabilities': ['native_session_v1']})]
    monkeypatch.setattr(cr, '_open_url', gated)
    path = tmp_path / 'fatal.log'
    path.write_text('Fatal Python error: Aborted\n')
    thread = cr._consume_native_crash(path)
    try:
        assert entered.wait(5)
        if change == 'off':
            cr.set_error_report_mode('off')
        elif change == 'consent':
            cr._update_state(affirmed_policy={})
        elif change == 'network':
            monkeypatch.setenv(cr.NO_NETWORK_ENV, '1')
        else:
            policy = cr._read_state()['affirmed_policy']
            cr._update_state(affirmed_policy=dict(policy, provider='none'))
    finally:
        release.set()
    joined(thread)
    assert relay.calls_to('/diagnose') == []
    if change == 'none':
        assert len(relay.calls_to('/report')) == 1
        assert relay.calls_to('/report')[0]['body']['summarize'] is False
    else:
        assert relay.calls_to('/report') == []


@pytest.mark.parametrize('user_note', ['', 'I pressed Close after opening the overview.', 'x' * 650, 'x' * 1000])
def test_old_relay_note_preserves_public_crash_fingerprints_and_user_note(configured, user_note):
    import hashlib
    relay, outcomes, ready = configured
    session = {'known': True, 'id': '14cbd541-b9a1-41f4-8f50-4b53ba84755a',
               'started_at': '2026-10-23T18:00:00Z', 'spacr_version': '1.5.1.6',
               'python': '3.12.11', 'qt': '6.11.2', 'source_sha256': {}}
    package = cr.Path(cr.__file__).resolve().parent
    for key, relative in (('package', '__init__.py'), ('app', 'qt/app.py'),
                          ('gc_policy', 'qt/gc_policy.py'), ('crashreport', 'crashreport.py')):
        session['source_sha256'][key] = hashlib.sha256((package / relative).read_bytes()).hexdigest()
    summary = cr.public_summary(message='Native process termination: Aborted', module='native', user_note=user_note)
    summary.update(native_session=session, summarize=False)
    status, _ = cr._post_json('/report', summary, 1)
    assert status == 200 and len(relay.calls_to('/report')) == 1
    body = relay.calls_to('/report')[0]['body']
    assert 'native_session' not in body and body['summarize'] is False
    assert relay.calls_to('/diagnose') == []
    note = body['user_note']
    assert note.startswith(user_note) and len(note) <= cr.MAX_NOTE_CHARS
    assert cr.scrub_public_text(note, cr.MAX_NOTE_CHARS) == note
    if len(user_note) < 650:
        assert 'not the crashed build' in note and 'Crashed session: 2026-10-23T18:00:00Z' in note
        assert 'spaCR version v1.5.1.6' in note and 'Python version v3.12.11' in note
        assert 'Qt version v6.11.2' in note
        assert session['id'].replace('-', ' ') in note
        assert 'Public installed-source SHA256 fingerprints (8 groups of 8 hex digits)' in note
        for key, digest in session['source_sha256'].items():
            grouped = ' '.join(digest[index:index + 8] for index in range(0, 64, 8))
            assert key + ' ' + grouped in note
    elif len(user_note) == 650:
        assert 'Crash-session details kept locally (note limit)' in note
        assert 'fingerprints' not in note
    else:
        assert note == user_note


@pytest.mark.parametrize('history', ['overlapping-writers', 'truncated-history'])
def test_shared_native_history_never_assigns_another_writers_session(configured, tmp_path, history):
    relay, outcomes, ready = configured
    first = {'known': True, 'id': '14cbd541-b9a1-41f4-8f50-4b53ba84755a',
             'started_at': '2026-10-23T18:00:00Z', 'spacr_version': '1.2.3',
             'python': '3.12.11', 'qt': '6.11.2', 'source_sha256': {'app': 'a' * 64}}
    second = dict(first, id='6ae2c2af-3e47-4bf8-9a19-f346ba655d34', python='3.14.0')
    path = tmp_path / 'shared-fatal.log'
    with path.open('a') as writer_a, path.open('a') as writer_b:
        if history == 'overlapping-writers':
            writer_a.write('=== spaCR session ' + json.dumps(first) + ' ===\n')
        else:
            writer_a.write('x' * cr.MAX_LOG_BYTES + '\n')
        writer_a.flush()
        writer_b.write('=== spaCR session ' + json.dumps(second) + ' ===\n')
        writer_b.flush()
        writer_a.write('Fatal Python error: Aborted\n\nCurrent thread:\n  File "app.py", line 41 in from_session_A\n')
    joined(cr._consume_native_crash(path))
    body = relay.calls_to('/report')[0]['body']
    assert body['frames'][-1]['function'] == 'from_session_A'
    assert 'Crashed session unknown' in body['user_note']
    assert second['id'].replace('-', ' ') not in body['user_note']
    assert 'Python version v3.14.0' not in body['user_note']
    assert cr._consume_native_crash(path) is None


def test_actual_native_rotation_retains_unknown_history_within_the_existing_cap(configured, tmp_path, monkeypatch):
    import faulthandler
    relay, outcomes, ready = configured
    policy = cr._read_state()['affirmed_policy']
    cr._update_state(affirmed_policy={})
    monkeypatch.setattr(cr, '_NATIVE_CAPTURE_HANDLE', None)
    monkeypatch.setattr(cr, '_NATIVE_EXCEPTION_HOOK', object())
    monkeypatch.setattr(faulthandler, 'enable', lambda **kwargs: None)
    first = {'known': True, 'id': '14cbd541-b9a1-41f4-8f50-4b53ba84755a',
             'started_at': '2026-10-23T18:00:00Z', 'spacr_version': '1.2.3',
             'python': '3.12.11', 'qt': '6.11.2', 'source_sha256': {'app': 'a' * 64}}
    second = dict(first, id='6ae2c2af-3e47-4bf8-9a19-f346ba655d34', python='3.14.0')
    path = tmp_path / 'fatal.log'
    path.write_text('=== spaCR session ' + json.dumps(first) + ' ===\n' + 'x' * cr.MAX_LOG_BYTES
                    + '\n=== spaCR session ' + json.dumps(second) + ' ===\n'
                    + 'Fatal Python error: Aborted\n\nCurrent thread:\n  File "app.py", line 41 in from_session_A\n')
    old_size = path.stat().st_size
    cr._update_state(native_fatal_ack={'path': str(path), 'offset': old_size - 8})
    original_open = cr.Path.open
    rotated = []
    def observe_append(target, mode='r', *args, **kwargs):
        if target == path and mode == 'a':
            rotated.append(target.read_bytes())
        return original_open(target, mode, *args, **kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(cr.Path, 'open', observe_append)
        assert cr._install_native_error_capture(path) == str(path)
    cr._NATIVE_CAPTURE_HANDLE.close()
    assert len(rotated) == 1 and len(rotated[0]) == cr.MAX_LOG_BYTES
    assert cr._read_state()['native_fatal_ack']['offset'] == cr.MAX_LOG_BYTES - 8
    snapshot = tmp_path / 'rotation-window.log'
    snapshot.write_bytes(rotated[0])
    cr._update_state(affirmed_policy=policy)
    joined(cr._consume_native_crash(snapshot))
    body = relay.calls_to('/report')[0]['body']
    assert 'Crashed session unknown' in body['user_note']
    assert 'Python version v3.14.0' not in body['user_note']
    assert rotated[0].startswith(b'\n=== spaCR session history truncated ===\n')


def test_empty_structured_error_still_has_an_honest_public_identity(configured):
    relay, outcomes, ready = configured
    record = logging.LogRecord('spacr.worker', logging.ERROR, __file__, 1, '', (), None)
    cr._capture_error_record(record)
    assert ready.wait(5)
    body = relay.calls_to('/report')[0]['body']
    assert body['exception_type'] == 'LoggedError'
    assert body['module'] == 'spacrworker'
    assert body['message'] == ''
    assert len(relay.calls_to('/report')) == 1


def test_readonly_exception_and_policy_capacity_never_reuse_an_expired_token(configured):
    class ReadOnlyError(RuntimeError):
        def __setattr__(self, name, value):
            raise AttributeError('read-only exception')
    error = ReadOnlyError('failed')
    first = cr._new_error_occurrence(error)
    second = cr._new_error_occurrence(error)
    assert first != second
    for index in range(512):
        cr._new_error_occurrence(event_key='capacity-' + str(index))
    assert cr._dispatch_error_now('old delivery', event_id=first) is None
    assert cr._dispatch_error_now('old delivery', event_id=second) is None
    relay, outcomes, ready = configured
    assert relay.calls == []
    assert not ready.is_set()


def test_worker_start_failure_releases_its_slot_for_the_next_real_error(configured, monkeypatch):
    relay, outcomes, ready = configured
    original = threading.Thread.start
    def fail_start(thread):
        if thread.name == 'spacr-error-report':
            raise RuntimeError('thread cannot start')
        return original(thread)
    with monkeypatch.context() as scoped:
        scoped.setattr(threading.Thread, 'start', fail_start)
        assert cr._dispatch_error_now('RuntimeError: first failure') is None
    joined(cr._dispatch_error_now('RuntimeError: second failure'))
    assert len(relay.calls_to('/report')) == 1
    assert outcomes[-1]['status'] == cr.REPORT_SENT


def test_result_delivery_ignores_dead_and_broken_listeners_and_keeps_original_stderr(configured, monkeypatch):
    import weakref
    relay, outcomes, ready = configured
    def broken(outcome):
        raise RuntimeError('GUI has closed')
    def retired(outcome):
        raise AssertionError('dead listener ran')
    dead = weakref.ref(retired)
    del retired
    monkeypatch.setattr(cr, '_ERROR_RESULT_LISTENERS', [weakref.ref(broken), dead])
    stream = io.StringIO()
    import types
    local_sys = types.SimpleNamespace(**vars(cr.sys))
    local_sys.__stderr__ = stream
    monkeypatch.setattr(cr, 'sys', local_sys)
    _ORIGINAL_DELIVERY({'explanation': {'text': 'Safe advice'}, 'issue_url': 'https://example.org/12'})
    assert stream.getvalue() == '\nSafe advice\nhttps://example.org/12\n'
    assert relay.calls == []


def test_disabled_native_diagnosis_never_even_queries_capabilities(configured):
    relay, outcomes, ready = configured
    cr.set_error_report_mode('off')
    with pytest.raises(cr.RelayRefused, match='automatic reporting disabled'):
        cr._post_json('/diagnose', {'kind': cr.KIND_FAULT, 'native_session': {'known': False}}, 2)
    assert relay.calls == []


@pytest.mark.parametrize('provider', ['none', 'claude'])
def test_native_provider_revocation_during_health_prevents_diagnosis(configured, monkeypatch, provider):
    relay, outcomes, ready = configured
    original = relay
    def revoke(request, timeout):
        if request.full_url.endswith('/health'):
            state = cr._read_state()
            state['affirmed_policy']['provider'] = provider
            cr._write_state(state)
        return original(request, timeout)
    monkeypatch.setattr(cr, '_open_url', revoke)
    with pytest.raises(cr.RelayRefused, match='automatic reporting disabled|AI provider changed'):
        cr._post_json('/diagnose', {'kind': cr.KIND_FAULT, 'native_session': {'known': False}}, 2)
    assert len(relay.calls_to('/health')) == 1
    assert relay.calls_to('/diagnose') == []


def test_failed_capability_lookup_preserves_legacy_native_report(configured):
    relay, outcomes, ready = configured
    relay.routes['/health'] = [OSError('health unavailable')]
    status, answer = cr._post_json('/report', {
        'kind': cr.KIND_FAULT, 'native_session': {'known': False}, 'summarize': False}, 2)
    assert status == 200
    [posted] = relay.calls_to('/report')
    assert 'native_session' not in posted['body']
    assert 'Crashed session unknown' in posted['body']['user_note']
    assert posted['body']['summarize'] is False


def test_offline_hosted_request_is_refused_before_transport(configured, monkeypatch):
    relay, outcomes, ready = configured
    monkeypatch.setenv(cr.NO_NETWORK_ENV, '1')
    with pytest.raises(cr.RelayRefused, match='network disabled'):
        cr._post_json('/diagnose', {'message': 'failed'}, 2)
    assert relay.calls == []


def test_active_queue_is_kept_then_revoked_user_error_is_dropped(configured):
    relay, outcomes, ready = configured
    summary = cr.public_summary(traceback_text='spacr.errors.ConfigurationError: choose a channel')
    path = cr._enqueue(summary, automatic=True)
    assert path is not None
    cr._ERROR_ACTIVE_QUEUES.add(path)
    try:
        assert cr.flush_report_queue()['kept'] == 1
        assert path.exists()
    finally:
        cr._ERROR_ACTIVE_QUEUES.discard(path)
    cr.set_error_report_mode('faults')
    result = cr.flush_report_queue()
    assert result['dropped'] == 1 and not path.exists()
    assert relay.calls_to('/report') == []


@pytest.mark.parametrize('header', ['{broken}', '[]'])
def test_invalid_native_session_header_keeps_real_fatal_frames_but_never_invents_a_build(
        configured, tmp_path, header):
    relay, outcomes, ready = configured
    relay.routes['/health'] = [(200, {'capabilities': ['native_session_v1']})]
    path = tmp_path / 'fatal.log'
    path.write_text('=== spaCR session ' + header + ' ===\n'
                    'Fatal Python error: Aborted\n\nCurrent thread:\n'
                    f'  File "{cr.Path(cr.__file__).parent / "qt" / "app.py"}", line 41 in closeEvent\n')
    joined(cr._consume_native_crash(path))
    body = relay.calls_to('/report')[0]['body']
    assert body['native_session'] == {'known': False}
    assert body['frames'] == [{'file': 'app.py', 'line': 41, 'function': 'closeEvent'}]
    assert cr._consume_native_crash(path) is None


def test_native_log_truncation_resets_the_old_ack_but_expired_logs_are_not_replayed(
        configured, tmp_path):
    import os
    import time
    relay, outcomes, ready = configured
    path = tmp_path / 'fatal.log'
    path.write_text('Fatal Python error: Aborted\n')
    cr._update_state(native_fatal_ack={'path': str(path), 'offset': path.stat().st_size + 100})
    expired = time.time() - cr.QUEUE_RETENTION_S - 60
    os.utime(path, (expired, expired))
    assert cr._consume_native_crash(path) is None
    assert relay.calls == []
    os.utime(path, None)
    joined(cr._consume_native_crash(path))
    assert len(relay.calls_to('/report')) == 1
    assert cr._read_state()['native_fatal_ack']['offset'] == path.stat().st_size


def test_local_native_capture_survives_unreadable_fixed_source_and_broken_loaded_qt_version(
        configured, tmp_path, monkeypatch):
    import faulthandler
    import types
    from pathlib import Path
    relay, outcomes, ready = configured
    path = tmp_path / 'fatal.log'
    monkeypatch.setattr(cr, '_NATIVE_CAPTURE_HANDLE', None)
    monkeypatch.setattr(cr, '_NATIVE_EXCEPTION_HOOK', None)
    previous_hook = cr.sys.excepthook
    local_sys = types.SimpleNamespace(**vars(cr.sys))
    local_sys.modules = dict(cr.sys.modules)
    monkeypatch.setattr(cr, 'sys', local_sys)
    enabled = []
    monkeypatch.setattr(faulthandler, 'enable', lambda **kwargs: enabled.append(kwargs['file']))
    def broken_version():
        raise RuntimeError('Qt binding unavailable')
    local_sys.modules['PySide6.QtCore'] = types.SimpleNamespace(qVersion=broken_version)
    original = Path.read_bytes
    def unreadable(target):
        if target == Path(cr.__file__).resolve():
            raise OSError('source unreadable')
        return original(target)
    monkeypatch.setattr(Path, 'read_bytes', unreadable)
    assert cr._install_native_error_capture(path) == str(path)
    try:
        header = json.loads(path.read_text().split('=== spaCR session ')[1].split(' ===')[0])
        assert header['qt'] == 'unknown'
        assert set(header['source_sha256']) == {'package', 'app', 'gc_policy'}
        assert enabled == [cr._NATIVE_CAPTURE_HANDLE]
        assert cr._NATIVE_EXCEPTION_HOOK is cr.sys.excepthook
        assert cr.sys.excepthook is not previous_hook
        assert relay.calls == []
    finally:
        cr._NATIVE_CAPTURE_HANDLE.close()


def test_real_burst_reports_omission_counts_and_bounded_long_error_details(configured, monkeypatch):
    relay, outcomes, ready = configured
    now = [0.]
    control = cr._ErrorBatcher(clock=lambda: now[0],
        schedule=lambda delay, callback: threading.Timer(delay, callback))
    monkeypatch.setattr(cr, '_ERROR_BATCHER', control)
    workers = []
    actual = cr._dispatch_error_now
    def observe(*args, **kwargs):
        worker = actual(*args, **kwargs)
        if worker is not None:
            workers.append(worker)
        return worker
    monkeypatch.setattr(cr, '_dispatch_error_now', observe)
    try:
        joined(cr._dispatch_error('RuntimeError: initial field'))
        now[0] = 1.
        for index in range(12):
            assert cr._dispatch_error('RuntimeError: field ' + str(index) + ' ' + 'x' * 240 + '\n\n') is None
        now[0] = 61.
        control._tick()
        for worker in workers:
            joined(worker)
        reports = relay.calls_to('/report')
        assert len(reports) == 2
        assert '4 additional occurrences omitted' in reports[-1]['body']['message']
        question = relay.calls_to('/diagnose')[-1]['body']['question']
        assert ' ... ' in question
        assert 'x' * 181 not in question
        assert reports[-1]['body']['kind'] == cr.KIND_FAULT
    finally:
        if control.timer is not None:
            control.timer.cancel()
        for worker in workers:
            worker.join(5)
            assert not worker.is_alive()


def test_same_explicit_producer_token_is_frozen_and_duplicate_direct_delivery_is_once(configured):
    relay, outcomes, ready = configured
    token = cr._new_error_occurrence(event_key='real-explicit-producer')
    cr.set_error_report_mode('off')
    assert cr._new_error_occurrence(event_key='real-explicit-producer') == token
    cr.set_error_report_mode('all')
    joined(cr._dispatch_error_now('RuntimeError: first', event_id=token))
    assert cr._dispatch_error_now('RuntimeError: transported', event_id=token) is None
    assert len(relay.calls_to('/report')) == 1


def test_nested_reporter_output_cannot_reenter_capture(configured, monkeypatch):
    relay, outcomes, ready = configured
    def nested(outcome):
        cr._capture_error_line('ERROR nested reporter message\n')
        assert cr._dispatch_error_now('RuntimeError: nested reporter failure') is None
        outcomes.append(outcome)
    monkeypatch.setattr(cr, '_deliver_error_result', nested)
    joined(cr._dispatch_error('RuntimeError: original'))
    assert len(relay.calls_to('/report')) == 1
    assert outcomes[-1]['status'] == cr.REPORT_SENT


def test_validation_log_without_exception_keeps_typed_callsite_and_public_context(configured, monkeypatch):
    relay, outcomes, ready = configured
    record = logging.LogRecord('spacr.qt.bridge', logging.ERROR, __file__, 1, 'Choose a channel', (), None)
    record.spacr_call_site = [f'  File "{cr.Path(cr.__file__).parent / "qt" / "bridge.py"}", line 42, in run\n']
    previous = getattr(cr._ERROR_DISPATCH_LOCAL, 'validation_context', None)
    with cr._validation_error_context('mask', ['Choose a channel']):
        cr._capture_error_record(record)
    assert getattr(cr._ERROR_DISPATCH_LOCAL, 'validation_context', None) is previous
    assert ready.wait(5)
    [report] = relay.calls_to('/report')
    assert report['body']['kind'] == cr.KIND_USER_ERROR
    assert report['body']['exception_type'] == 'ConfigurationError'
    assert report['body']['module'] == 'mask'
    assert report['body']['screen'] == 'Mask'
    assert report['body']['frames'][-1] == {'file': 'bridge.py', 'function': 'run', 'line': 42}


def test_formatted_logging_stream_does_not_recapture_its_structured_error(configured):
    relay, outcomes, ready = configured
    sink = io.StringIO()
    logger = logging.Logger('spacr.worker')
    structured = _AutomaticErrorHandler(logging.ERROR)
    terminal = logging.StreamHandler(_ErrorStream(sink))
    terminal.setFormatter(logging.Formatter('ERROR %(name)s: %(message)s'))
    logger.addHandler(structured)
    logger.addHandler(terminal)
    try:
        logger.error('original worker error')
        assert ready.wait(5)
        assert sink.getvalue() == 'ERROR spacr.worker: original worker error\n'
        assert len(relay.calls_to('/diagnose')) == len(relay.calls_to('/report')) == 1
    finally:
        logger.removeHandler(structured)
        logger.removeHandler(terminal)
        terminal.close()
        structured.close()


def test_no_network_refusal_returns_queued_outcome_without_any_post(configured, monkeypatch):
    relay, outcomes, ready = configured
    monkeypatch.setenv(cr.NO_NETWORK_ENV, '1')
    result = cr._send_one({'exception_type': 'RuntimeError', 'message': 'failed',
                           'frames': [], 'summarize': True}, 2)
    assert result == {'status': cr.REPORT_QUEUED, 'detail': 'network'}
    assert relay.calls == []


def test_revocation_after_queue_storage_failure_still_sends_nothing(configured):
    relay, outcomes, ready = configured
    occupied = cr._queue_dir()
    occupied.write_text('existing non-directory data')
    summary = {'exception_type': 'RuntimeError', 'message': 'failed', 'kind': cr.KIND_FAULT}
    path = cr._enqueue(summary)
    assert path is None
    cr.set_error_report_mode('off')
    result = cr._send_queued(path, summary, 2,
        allowed=lambda: cr._automatic_error_policy(cr.KIND_FAULT)[2])
    assert result == {'status': cr.REPORT_DISABLED}
    assert occupied.read_text() == 'existing non-directory data'
    assert relay.calls == []


def test_empty_saved_explanations_never_create_a_cache_entry(configured):
    relay, outcomes, ready = configured
    cr.save_explanation('', 'An answer without an error identity')
    cr.save_explanation('valid-identity', '')
    assert cr.saved_explanation('valid-identity') is None
    assert not cr._saved_path().exists()
    assert relay.calls == []


def test_both_final_and_cleanup_delivery_failure_cannot_poison_the_next_report(configured, monkeypatch):
    relay, outcomes, ready = configured
    seen = []
    def failed_delivery(outcome):
        seen.append(outcome)
        if not outcome.get('pending'):
            raise RuntimeError('result receiver disappeared')
    with monkeypatch.context() as scoped:
        scoped.setattr(cr, '_deliver_error_result', failed_delivery)
        joined(cr._dispatch_error('RuntimeError: original field'))
    assert seen[0]['pending']
    assert seen[-1]['quiet'] and seen[-1]['event_id'] == seen[0]['event_id']
    assert cr.queued_summaries() == []
    joined(cr._dispatch_error('RuntimeError: healthy later field'))
    assert len(relay.calls_to('/report')) == 2
    assert outcomes[-1]['status'] == cr.REPORT_SENT
