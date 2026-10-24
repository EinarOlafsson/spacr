"""Edge paths of crash reporting that the broader cohorts leave unvisited."""
import faulthandler
import json
import sys
import threading
import weakref

import pytest

from spacr import crashreport as cr
from tests.test_700_structured_error_dispatch import configured, failure, joined


def _batcher(now, notices):
    return cr._ErrorBatcher(clock=lambda: now[0],
                            schedule=lambda delay, callback: None,
                            notify=notices.append)


def _payload(text):
    return {"text": text, "module": "m", "provider": "p"}


def test_quiet_minute_flush_without_a_live_timer_resets_cleanly():
    notices, delivered, now = [], [], [0]
    control = _batcher(now, notices)
    control.submit(_payload("first"), lambda b, o: delivered.append(("first", b, o)))
    now[0] = 10
    control.submit(_payload("second"), lambda b, o: delivered.append(("second", b, o)))
    assert control.timer is None and len(control.pending) == 1
    control._deliver = lambda batch, omitted: delivered.append(("flush", len(batch), omitted))
    now[0] = 100
    control._tick()
    assert delivered[-1] == ("flush", 1, 0)
    assert control.started is None and control.last is None
    assert control.timer is None and control.pending == {}


def test_five_minute_storm_without_a_live_timer_stops_and_prompts_once():
    notices, delivered, now = [], [], [0]
    control = _batcher(now, notices)
    for index, moment in enumerate([0, 50, 100, 150, 200, 250, 290]):
        now[0] = moment
        control.submit(_payload(f"e{index}"), lambda b, o: None)
    control._deliver = lambda batch, omitted: delivered.append((len(batch), omitted))
    now[0] = 310
    control._tick()
    assert control.stopped and control.automatic_stopped and control.prompted
    assert control.timer is None and control.pending == {}
    assert notices.count({"storm": True, "event_id": "error-storm"}) == 1
    assert delivered == [(0, 0)]


def test_result_delivery_skips_dead_and_declining_listeners(monkeypatch):
    seen = []

    def declining(outcome):
        seen.append(("declining", outcome["k"]))
        return False

    def accepting(outcome):
        seen.append(("accepting", outcome["k"]))
        return True

    class Gone:
        def __call__(self, outcome):
            return True

    gone = Gone()
    dead = weakref.ref(gone)
    del gone
    assert dead() is None
    monkeypatch.setattr(cr, "_ERROR_RESULT_LISTENERS",
                        [weakref.ref(accepting), dead, weakref.ref(declining)])
    monkeypatch.setattr(cr, "_write_error_result", lambda outcome: seen.append("terminal"))
    cr._deliver_error_result({"k": 1})
    assert seen == [("declining", 1), ("accepting", 1)]


def test_terminal_answer_is_skipped_when_no_original_stream_exists(monkeypatch):
    monkeypatch.setattr(sys, "__stderr__", None)
    monkeypatch.setattr(sys, "stderr", None)
    cr._write_error_result({"explanation": {"text": "advice"}, "issue_url": "https://x/1"})
    assert not getattr(cr._ERROR_DISPATCH_LOCAL, "active", False)


def test_error_lines_emitted_from_the_logging_module_are_not_recaptured(monkeypatch):
    calls = []
    monkeypatch.setattr(cr, "_dispatch_error", lambda *a, **k: calls.append((a, k)))
    namespace = {"__name__": "logging", "cr": cr}
    exec("def emit(text):\n    cr._capture_error_line(text)\n", namespace)
    assert not getattr(cr._ERROR_DISPATCH_LOCAL, "terminal_buffer_flush", False)
    namespace["emit"]("ERROR from the logging module\n")
    assert calls == []
    cr._capture_error_line("ERROR from plain code\n")
    assert len(calls) == 1 and calls[0][0][0] == "ERROR from plain code\n"


def test_background_submission_without_a_completion_callback_still_sends(monkeypatch):
    sent = []
    monkeypatch.setattr(cr, "_sending_allowed", lambda: True)
    monkeypatch.setattr(cr, "_validated_summary", lambda summary: dict(summary))
    monkeypatch.setattr(cr, "_enqueue", lambda clean: "queued-path")
    monkeypatch.setattr(cr, "_send_queued",
                        lambda path, clean, timeout: sent.append((path, clean)) or {"status": cr.REPORT_SENT})
    thread = cr.submit_public_summary_in_background({"kind": "fault"})
    thread.join(5)
    assert not thread.is_alive()
    assert sent == [("queued-path", {"kind": "fault"})]


def test_native_post_refuses_when_the_network_closes_before_the_health_lookup(configured, monkeypatch):
    relay, outcomes, ready = configured
    policy = dict(cr._read_state()["affirmed_policy"], _mode="all")
    monkeypatch.setattr(cr, "_automatic_error_policy", lambda kind: (dict(policy), True, True))
    monkeypatch.setattr(cr, "network_allowed", lambda: False)
    with pytest.raises(cr.RelayRefused, match="automatic reporting disabled"):
        cr._post_json("/report", {"kind": cr.KIND_FAULT, "native_session": {"known": False}}, 2)
    assert relay.calls == []


def test_native_post_keeps_source_lines_in_the_all_code_mode(configured):
    relay, outcomes, ready = configured
    relay.routes["/health"] = [(200, {"capabilities": ["native_session_v1"]})]
    cr.set_error_report_mode("all_code")
    frames = [{"file": "spacr/app.py", "function": "f", "line": 3, "code": "value = 1"}]
    status, _ = cr._post_json("/report", {
        "kind": cr.KIND_FAULT, "native_session": {"known": False},
        "summarize": False, "frames": frames}, 2)
    assert status == 200
    [posted] = relay.calls_to("/report")
    assert posted["body"]["frames"][0]["code"] == "value = 1"


def test_unknown_provider_name_explains_without_a_provider_object(configured, monkeypatch):
    relay, outcomes, ready = configured
    from spacr.qt.ai import providers
    policy = cr._read_state()["affirmed_policy"]
    cr._update_state(affirmed_policy=dict(policy, provider="no-such-provider"))
    received = []

    def explain(summary, provider):
        received.append(provider)
        return {"text": "explained without provider"}

    monkeypatch.setattr(providers, "explain_failure", explain)
    exc, tb = failure()
    joined(cr._dispatch_error(tb, exception=exc))
    assert received == [None]
    assert outcomes and outcomes[-1]["explanation"] == {"text": "explained without provider"}


def _isolate_capture(monkeypatch):
    monkeypatch.setattr(cr, "_NATIVE_CAPTURE_HANDLE", None)
    monkeypatch.setattr(cr, "_NATIVE_EXCEPTION_HOOK", object())
    monkeypatch.setattr(faulthandler, "enable", lambda **kwargs: None)


def _close_capture():
    handle = cr._NATIVE_CAPTURE_HANDLE
    if handle is not None:
        handle.close()


def test_oversized_crash_log_is_trimmed_even_if_the_state_module_vanishes(tmp_path, monkeypatch):
    _isolate_capture(monkeypatch)
    monkeypatch.setattr(cr, "MAX_LOG_BYTES", 2000)
    path = tmp_path / "fatal.log"
    path.write_bytes(b"x" * 6000)
    real = cr._update_state

    def fail_on_ack(**fields):
        if "native_fatal_ack" in fields:
            raise ImportError("state module unavailable")
        return real(**fields)

    monkeypatch.setattr(cr, "_update_state", fail_on_ack)
    try:
        assert cr._install_native_error_capture(path) == str(path)
    finally:
        _close_capture()
    data = path.read_bytes()
    assert data.startswith(b"\n=== spaCR session history truncated ===\n")
    assert b"=== spaCR session {" in data
    assert len(data) < 6000


def test_session_fingerprint_skips_an_oversized_source_file(tmp_path, monkeypatch):
    _isolate_capture(monkeypatch)
    package = tmp_path / "pkg"
    (package / "qt").mkdir(parents=True)
    with open(package / "__init__.py", "wb") as big:
        big.truncate(3 * 1024 * 1024)
    (package / "qt" / "app.py").write_text("app = 1\n")
    monkeypatch.setattr(cr, "__file__", str(package / "crashreport.py"))
    path = tmp_path / "fatal.log"
    try:
        assert cr._install_native_error_capture(path) == str(path)
    finally:
        _close_capture()
    header = json.loads(path.read_text().split("=== spaCR session ")[1].split(" ===")[0])
    assert "package" not in header["source_sha256"]
    assert set(header["source_sha256"]) == {"app"}


def test_native_capture_reports_an_unusable_log_location_as_empty(tmp_path, monkeypatch):
    _isolate_capture(monkeypatch)
    blocker = tmp_path / "plain-file"
    blocker.write_text("not a directory")
    assert cr._install_native_error_capture(blocker / "nested" / "fatal.log") == ""
    assert cr._NATIVE_CAPTURE_HANDLE is None
