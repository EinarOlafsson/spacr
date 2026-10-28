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


def test_untrusted_merge_structure_is_filtered_not_quoted():
    error = ValueError('private raw error')
    error._spacr_merge_structure = ['private', {'reason': 'private_reason'}]
    assert cr._merge_structure_message(error) == ''
    error._spacr_merge_structure.append({
        'reason': 'duplicate_keys', 'table': 'private_table',
        'present': ['plateID', 'secret_column'], 'rows': True,
        'duplicate_rows': 2})
    message = cr._merge_structure_message(error)
    payload = json.loads(message.removeprefix('Merge validation: '))
    assert len(payload['tables']) == 1
    assert payload['tables'][0]['table'] == 'custom'
    assert payload['tables'][0]['present'] == ['plateID']
    assert payload['tables'][0]['rows'] is None
    assert payload['tables'][0]['duplicate_rows'] == 2
    assert 'secret' not in message and 'private' not in message


def test_native_context_rejects_compound_values_and_foreign_sessions(monkeypatch):
    monkeypatch.setattr(cr, '_identity_scrub', lambda text: text)
    assert cr._native_context_values({'screen': ['private'], 'qt': {'path': 'secret'}}) == {}
    owner = '12345678-1234-1234-1234-123456789abc'
    lines = [
        '=== spaCR context {bad} ===',
        '=== spaCR context [] ===',
        '=== spaCR context {"_thread":"bad"} ===',
        '=== spaCR context {"_thread":"0x99","screen":"Foreign"} ===',
        '=== spaCR context ' + json.dumps({'_thread': '0x10', '_session': owner, 'screen': 'Overview'}) + ' ===',
        '=== spaCR context ' + json.dumps({'_thread': '0x10', '_session': owner, 'screen': []}) + ' ===',
    ]
    contexts, session = cr._native_contexts('\n'.join(lines), 'Current thread 0x10 (most recent call first):')
    assert contexts == [{'screen': 'Overview'}]
    assert session == owner
    assert cr._native_contexts('\n'.join(lines), 'Current thread 0x11 (most recent call first):') == ([], '')
    headers = '\n'.join(['=== spaCR session {bad} ===',
                         '=== spaCR session [] ===',
                         '=== spaCR session {"id":"foreign"} ===',
                         '=== spaCR session ' + json.dumps({'id': owner, 'known': True}) + ' ==='])
    assert cr._native_session_by_id(headers, owner) == {'id': owner, 'known': True}
    assert cr._native_session_by_id(headers, 'absent') == {'known': False}


def test_native_platform_survives_unavailable_qt_version(monkeypatch):
    from types import SimpleNamespace

    def unavailable():
        raise RuntimeError('Qt version unavailable')

    monkeypatch.setattr(cr.sys, 'platform', 'darwin')
    monkeypatch.setattr(cr.platform, 'mac_ver', lambda: ('15.0', (), ''))
    monkeypatch.setattr(cr.platform, 'machine', lambda: 'arm64')
    monkeypatch.setattr(cr, '_distribution_version', lambda name: '6.11.3')
    monkeypatch.setitem(sys.modules, 'PySide6.QtCore', SimpleNamespace(qVersion=unavailable))
    assert cr._native_platform_context() == {'os': cr._os_family(), 'os_release': '15.0', 'machine': 'arm64', 'pyside6': '6.11.3'}


def test_consented_native_replay_does_not_invent_screen_or_repeat_event(
        configured, tmp_path, monkeypatch):
    path = tmp_path / 'owned-native.log'
    path.write_text('Fatal Python error: Segmentation fault\n\n'
                    'Current thread 0x10 (most recent call first):\n'
                    '  File "/private/project/worker.py", line 12 in run\n')
    delivered = []
    marker = object()

    def capture(message, **kwargs):
        delivered.append((message, kwargs))
        return marker

    monkeypatch.setattr(cr, '_dispatch_error', capture)
    assert cr._consume_native_crash(path) is marker
    assert len(delivered) == 1
    message, payload = delivered[0]
    assert message == 'Native process termination: Segmentation fault'
    assert not payload['summary'].get('screen')
    assert payload['summary']['exception_type'] == 'NativeCrash'
    assert 'worker' in payload['summary']['user_note']
    assert '/private/' not in json.dumps(payload['summary'])
    assert cr._consume_native_crash(path) is None
    assert len(delivered) == 1


@pytest.mark.parametrize('write_error', [OSError, ValueError, AttributeError])
def test_closed_native_context_sink_cannot_interrupt_the_calling_session(
        monkeypatch, write_error):
    attempted = []

    class ClosedCapture:
        def write(self, line):
            attempted.append(line)
            raise write_error('native log is no longer writable')

    monkeypatch.setattr(cr, '_NATIVE_CAPTURE_HANDLE', ClosedCapture())
    assert cr._note_native_context(qt='6.11.3', private_path='/private/project') is False
    assert len(attempted) == 1
    payload = json.loads(attempted[0].removeprefix('=== spaCR context ').removesuffix(' ===\n'))
    assert payload['qt'] == '6.11.3'
    assert 'private_path' not in payload
    assert '/private/' not in attempted[0]


def test_partial_native_context_without_a_crashing_frame_does_not_invent_a_note(
        configured, tmp_path, monkeypatch):
    path = tmp_path / 'partial-native.log'
    owner = '12345678-1234-1234-1234-123456789abc'
    context = {'_thread': '0x10', '_session': owner, 'qt_platform': 'wayland'}
    path.write_text('=== spaCR context ' + json.dumps(context) + ' ===\n'
                    'Fatal Python error: Segmentation fault\n\n'
                    'Thread 0x10 (most recent call first):\n'
                    '  File "/private/project/worker.py", line 12 in run\n', encoding='utf-8')
    delivered = []
    marker = object()

    def capture(message, **kwargs):
        delivered.append((message, kwargs))
        return marker

    monkeypatch.setattr(cr, '_dispatch_error', capture)
    assert cr._consume_native_crash(path) is marker
    assert len(delivered) == 1
    message, payload = delivered[0]
    assert message == 'Native process termination: Segmentation fault'
    assert payload['summary']['exception_type'] == 'NativeCrash'
    assert not payload['summary'].get('user_note')
    assert not payload['summary'].get('screen')
    assert len(payload['summary']['frames']) == 1
    assert '/private/' not in json.dumps(payload['summary'])
    assert cr._consume_native_crash(path) is None
    assert len(delivered) == 1
