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
def configured(tmp_path, monkeypatch):
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
    assert "spaCR started" in path.read_text()
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
    assert path.stat().st_size < cr.MAX_LOG_BYTES + 100
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
