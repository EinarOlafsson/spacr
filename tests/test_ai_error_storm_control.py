"""Bounded AI error batching and persistent hosted admission without live calls."""
from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from spacr import crashreport as cr


class Timer:
    def cancel(self):
        pass


class Clock:
    now = 0.

    def __call__(self):
        return self.now


def batcher():
    clock, sent, notices, timers = Clock(), [], [], []
    def schedule(delay, callback):
        timers.append((delay, callback))
        return Timer()
    control = cr._ErrorBatcher(clock=clock, schedule=schedule, notify=notices.append)
    def deliver(batch, omitted):
        sent.append((batch, omitted))
    return control, clock, sent, notices, timers, deliver



@pytest.fixture
def isolated_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "_reports_dir", lambda: tmp_path)
    monkeypatch.setattr(cr, "network_allowed", lambda: True)
    monkeypatch.setattr(cr, "_ERROR_POLICY_CACHE", {})
    return tmp_path


def test_quiet_window_combines_repeated_and_distinct_errors_exactly_at_sixty_seconds():
    control, clock, sent, notices, timers, deliver = batcher()
    control.submit(dict(text="first"), deliver)
    assert len(sent) == 1
    for moment, text in ((10, "duplicate"), (20, "duplicate"), (30, "different")):
        clock.now = moment
        control.submit(dict(text=text), deliver)
    assert len(sent) == 1 and len(notices) == 1 and len(timers) == 1
    clock.now = 89.999
    control._tick()
    assert len(sent) == 1
    clock.now = 90
    control._tick()
    assert [(item["text"], count) for item, count in sent[-1][0]] == [
        ("duplicate", 2), ("different", 1)]
    control.submit(dict(text="new burst"), deliver)
    assert len(sent) == 3


def test_exact_quiet_boundary_flushes_before_next_error_arrival():
    control, clock, sent, _, _, deliver = batcher()
    control.submit(dict(text="first"), deliver)
    clock.now = 1
    control.submit(dict(text="pending"), deliver)
    clock.now = 61
    control.submit(dict(text="new first"), deliver)
    assert [batch[0][0]["text"] for batch, _ in sent] == ["first", "pending", "new first"]


def test_five_minute_storm_prompts_once_and_blocks_automatic_but_allows_manual():
    control, clock, sent, notices, _, deliver = batcher()
    for moment in range(0, 300, 50):
        clock.now = moment
        control.submit(dict(text=str(moment)), deliver)
    clock.now = 300
    control._tick()
    assert len(sent) == 1
    assert sum(bool(note.get("storm")) for note in notices) == 1
    clock.now = 301
    control.submit(dict(text="automatic"), deliver)
    assert len(sent) == 1
    control.submit(dict(text="explicit", manual=True), deliver)
    assert len(sent) == 2
    for moment in range(350, 650, 50):
        clock.now = moment
        control.submit(dict(text="explicit", manual=True), deliver)
    control._tick()
    assert sum(bool(note.get("storm")) for note in notices) == 1


def test_thousand_errors_keep_bounded_distinct_records_and_all_counts():
    control, clock, sent, notices, timers, deliver = batcher()
    control.submit(dict(text="first"), deliver)
    clock.now = 1
    for index in range(1000):
        control.submit(dict(text="duplicate" if index < 900 else str(index)), deliver,
                       event_id=str(index))
    assert len(control.pending) == 8
    assert len(control.seen) == 512
    assert len(notices) == 1 and len(timers) == 1
    clock.now = 61
    control._tick()
    batch, omitted = sent[-1]
    assert sum(count for _, count in batch) + omitted == 1000
    assert batch[0][1] == 900


def test_rolling_limit_serializes_concurrent_callers_and_expires_at_boundary(isolated_guard):
    def call(_):
        try:
            cr._reserve_hosted_request(now=0)
            return "accepted"
        except cr.RelayRefused as error:
            return error.reason
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(call, range(20)))
    assert results.count("accepted") == 3
    assert results.count("rate") == 17
    with pytest.raises(cr.RelayRefused, match="three requests"):
        cr._reserve_hosted_request(now=59.999)
    cr._reserve_hosted_request(now=60)


def test_three_saturated_burst_minutes_persist_error_only_cooldown_across_restart(isolated_guard):
    moments = (58, 59, 60, 119, 120, 121, 180, 181, 182)
    for now in moments:
        cr._reserve_hosted_request(error=True, now=now)
    assert cr._read_state()["ai_request_guard"]["error_cooldown_until"] == 3782
    with pytest.raises(cr.RelayRefused) as caught:
        cr._reserve_hosted_request(error=True, now=242)
    assert caught.value.reason == "cooldown"
    assert caught.value.resets_at.timestamp() == 3782
    cr._reserve_hosted_request(now=242)
    cr._reserve_hosted_request(error=True, now=3782)


def test_broken_saturation_streak_does_not_start_cooldown(isolated_guard):
    for now in (0, 1, 2, 120, 121, 122, 180, 181, 182):
        cr._reserve_hosted_request(error=True, now=now)
    assert not cr._read_state()["ai_request_guard"].get("error_cooldown_until")


class Response:
    status = 200

    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def read(self, _size):
        return json.dumps(self.data).encode()


def test_actual_transport_guard_covers_chat_and_summaries_but_preserves_plain_reports(isolated_guard, monkeypatch):
    requests = []
    def opened(request, timeout):
        requests.append((request.full_url, json.loads(request.data or b"{}")))
        return Response(dict(advice="explanation", issue_number=42))
    monkeypatch.setattr(cr, "_open_url", opened)
    monkeypatch.setattr(cr.time, "time", lambda: 100.)
    for _ in range(3):
        cr._post_json("/diagnose", {}, 1)
    with pytest.raises(cr.RelayRefused) as caught:
        cr._post_json("/diagnose", {}, 1)
    assert caught.value.reason == "rate" and len(requests) == 3
    cr._post_json("/report", dict(summarize=False), 1)
    cr._send_one(dict(summarize=True), 1)
    assert len(requests) == 5
    assert requests[-1][1]["summarize"] is False


def test_cached_relay_errors_do_not_saturate_but_transport_ping_limit_remains(isolated_guard, monkeypatch):
    monkeypatch.setattr(cr, "_open_url", lambda *_args: Response(dict(advice="cached", cached=True)))
    for now in (58, 59, 60, 119, 120, 121, 180, 181, 182):
        monkeypatch.setattr(cr.time, "time", lambda moment=now: moment)
        with cr._hosted_error_context():
            cr._post_json("/diagnose", {}, 1)
    assert not cr._read_state()["ai_request_guard"].get("error_cooldown_until")
    with pytest.raises(cr.RelayRefused) as caught:
        cr._post_json("/diagnose", {}, 1)
    assert caught.value.reason == "rate"


def test_pending_automatic_policy_survives_occurrence_ledger_rotation(isolated_guard, monkeypatch):
    control, clock, _, _, _, _ = batcher()
    monkeypatch.setattr(cr, "_ERROR_BATCHER", control)
    monkeypatch.setattr(cr, "_ERROR_OCCURRENCE_POLICIES", {})
    monkeypatch.setattr(cr, "_automatic_error_policy", lambda kind: (
        dict(provider="hosted", _mode="all"), True, False))
    sent = []
    def immediate(text, **kwargs):
        event, captured = cr._error_occurrence(kwargs["event_id"])
        sent.append((text, captured))
    monkeypatch.setattr(cr, "_dispatch_error_now", immediate)
    cr._dispatch_error("first")
    clock.now = 1
    for _ in range(1000):
        cr._dispatch_error("ValueError: duplicate")
    assert len(cr._ERROR_OCCURRENCE_POLICIES) <= 512
    clock.now = 61
    control._tick()
    assert len(sent) == 2
    assert sent[-1][1][1] is True
    assert "1000 occurrences" in sent[-1][0]


def test_earlier_inflight_cached_response_cancels_provisional_ninth_call_cooldown(isolated_guard):
    reservations = [cr._reserve_hosted_request(error=True, now=now)
                    for now in (58, 59, 60, 119, 120, 121, 180, 181, 182)]
    assert cr._read_state()["ai_request_guard"]["error_cooldown_until"] == 3782
    cr._refund_cached_error(reservations[-2])
    assert not cr._read_state()["ai_request_guard"]["error_cooldown_until"]
    cr._reserve_hosted_request(error=True, now=242)


@pytest.mark.parametrize("order", (("ConfigurationError: missing input", "ValueError: internal fault"),
                                    ("ValueError: internal fault", "ConfigurationError: missing input")))
def test_mixed_failure_batch_preserves_fault_reporting_regardless_of_order(isolated_guard, monkeypatch, order):
    control, clock, _, _, _, _ = batcher()
    monkeypatch.setattr(cr, "_ERROR_BATCHER", control)
    monkeypatch.setattr(cr, "_ERROR_OCCURRENCE_POLICIES", {})
    monkeypatch.setattr(cr, "_automatic_error_policy", lambda kind: (
        dict(provider="hosted", _mode="faults"), True, cr.mode_sends(kind, "faults")))
    sent = []
    monkeypatch.setattr(cr, "_dispatch_error_now", lambda text, **kwargs: sent.append(kwargs))
    cr._dispatch_error("first")
    for index, text in enumerate(order, 1):
        clock.now = index
        cr._dispatch_error(text)
    clock.now = 62
    control._tick()
    assert len(sent) == 2
    assert sent[-1]["summary"]["kind"] == cr.KIND_FAULT
    assert cr._error_occurrence(sent[-1]["event_id"])[1][2]


def test_delayed_batch_filters_each_revoked_entry_before_selecting_owner(isolated_guard, monkeypatch):
    control, clock, _, _, _, _ = batcher()
    monkeypatch.setattr(cr, "_ERROR_BATCHER", control)
    monkeypatch.setattr(cr, "_ERROR_OCCURRENCE_POLICIES", {})
    allowed = [True]
    monkeypatch.setattr(cr, "_automatic_error_policy", lambda kind: (
        dict(provider="hosted", _mode="all"), allowed[0], allowed[0]))
    sent = []
    monkeypatch.setattr(cr, "_dispatch_error_now", lambda *args, **kwargs: sent.append(kwargs))
    cr._dispatch_error("first")
    clock.now = 1
    cr._dispatch_error("ValueError: pending")
    allowed[0] = False
    clock.now = 61
    control._tick()
    assert len(sent) == 1


def test_all_eight_error_identities_and_counts_reach_actual_hosted_payload(isolated_guard, monkeypatch):
    import hashlib
    control, clock, _, _, _, _ = batcher()
    monkeypatch.setattr(cr, "_ERROR_BATCHER", control)
    monkeypatch.setattr(cr, "_ERROR_OCCURRENCE_POLICIES", {})
    monkeypatch.setattr(cr, "_automatic_error_policy", lambda kind: (
        dict(provider="hosted", _mode="all_code"), True, False))
    summaries = []
    monkeypatch.setattr(cr, "_dispatch_error_now", lambda *args, **kwargs: summaries.append(kwargs.get("summary")))
    cr._dispatch_error("first")
    texts = [f"ValueError: distinct error number {index}" for index in range(8)]
    clock.now = 1
    for text in texts:
        cr._dispatch_error(text)
    clock.now = 61
    control._tick()
    summary = summaries[-1]
    bodies = []
    def opened(request, timeout):
        bodies.append(json.loads(request.data))
        return Response(dict(advice="answer"))
    monkeypatch.setattr(cr, "_open_url", opened)
    with cr._hosted_error_context():
        cr.diagnose_remotely(summary)
    message = bodies[0]["message"]
    assert len(message) <= cr.MAX_MESSAGE_CHARS
    for text in texts:
        assert hashlib.sha256(text.encode()).hexdigest()[:6] in message
    assert message.count("1x") == 8
    assert "error_text" not in bodies[0]
    for text in texts:
        assert text in bodies[0]["question"]
    assert len(bodies[0]["question"]) <= cr.MAX_QUESTION_CHARS
    assert all(text in cr.ai_summary_prompt(summary) for text in texts)


def test_full_automatic_batch_retains_later_explicit_manual_owner():
    control, clock, sent, _, _, deliver = batcher()
    control.submit(dict(text="first"), deliver)
    clock.now = 1
    for index in range(8):
        control.submit(dict(text=str(index)), deliver)
    manual = []
    control.submit(dict(text="explicit", manual=True), lambda batch, omitted: manual.append((batch, omitted)))
    assert len(control.pending) == 8
    clock.now = 61
    control._tick()
    assert len(manual) == 1 and manual[0][1] == 1
    assert any(entry["text"] == "explicit" for entry, _ in manual[0][0])


def test_direct_error_diagnosis_cannot_bypass_hour_pause_but_contextual_chat_can(isolated_guard, monkeypatch):
    monkeypatch.setattr(cr, "_open_url", lambda *_args: Response(dict(advice="answer")))
    for now in (58, 59, 60, 119, 120, 121, 180, 181, 182):
        monkeypatch.setattr(cr.time, "time", lambda moment=now: moment)
        cr.diagnose_remotely(dict(exception_type="ValueError", message="fault"))
    monkeypatch.setattr(cr.time, "time", lambda: 242)
    with pytest.raises(cr.RelayRefused) as caught:
        cr.diagnose_remotely(dict(exception_type="ValueError", message="fault"))
    assert caught.value.reason == "cooldown"
    assert cr.diagnose_remotely(dict(exception_type="ValueError", message="fault"),
                               "How do I choose channels?")["advice"] == "answer"


def test_manual_ai_owner_preserves_automatic_report_without_second_ai_request(isolated_guard, monkeypatch):
    import threading
    from tests.test_700_anonymous_error_reports import FakeRelay
    control, clock, _, _, _, _ = batcher()
    monkeypatch.setattr(cr, "_ERROR_BATCHER", control)
    monkeypatch.setattr(cr, "_ERROR_OCCURRENCE_POLICIES", {})
    monkeypatch.setattr(cr, "_ERROR_DISPATCH_SEEN", {})
    monkeypatch.setattr(cr, "_automatic_error_policy", lambda kind: (
        dict(provider="hosted", _mode="all"), True, True))
    monkeypatch.setattr(cr, "_sending_allowed", lambda: True)
    monkeypatch.setattr(cr, "_deliver_error_result", lambda result: None)
    relay = FakeRelay()
    monkeypatch.setattr(cr, "_open_url", relay)
    control.submit(dict(text="initial", provider="hosted"), lambda *args: None)
    clock.now = 1
    cr._dispatch_error("ValueError: automatic fault", module="mask")
    ai_calls = []
    control.submit(dict(text="KeyError: manual error", module="mask", provider="hosted", manual=True),
                   lambda batch, omitted: ai_calls.append(batch))
    finished = threading.Event()
    original = cr._send_one
    def send(*args):
        result = original(*args)
        finished.set()
        return result
    monkeypatch.setattr(cr, "_send_one", send)
    clock.now = 61
    control._tick()
    assert finished.wait(3)
    assert len(ai_calls) == 1
    assert len(relay.calls_to("/report")) == 1
    assert relay.calls_to("/diagnose") == []
    assert relay.calls_to("/report")[0]["body"]["summarize"] is False


def test_provider_change_preserves_old_report_and_uses_only_new_provider_for_ai(isolated_guard, monkeypatch):
    control, clock, _, _, _, _ = batcher()
    monkeypatch.setattr(cr, "_ERROR_BATCHER", control)
    monkeypatch.setattr(cr, "_ERROR_OCCURRENCE_POLICIES", {})
    selected = ["old"]
    monkeypatch.setattr(cr, "_automatic_error_policy", lambda kind: (
        dict(provider=selected[0], _mode="all"), True, True))
    calls = []
    def dispatched(text, **kwargs):
        policy, explain, send = cr._error_occurrence(kwargs["event_id"])[1]
        calls.append((text, policy["provider"], explain, send))
    monkeypatch.setattr(cr, "_dispatch_error_now", dispatched)
    cr._dispatch_error("first")
    clock.now = 1
    cr._dispatch_error("ValueError: old provider fault")
    selected[0] = "new"
    clock.now = 2
    cr._dispatch_error("KeyError: new provider fault")
    clock.now = 62
    control._tick()
    assert calls[-2][1:] == ("old", False, True)
    assert calls[-1][1:] == ("new", True, True)
    assert "old provider fault" not in calls[-1][0]


def test_enabling_ai_does_not_retroactively_explain_send_only_error(isolated_guard, monkeypatch):
    control, clock, _, _, _, _ = batcher()
    monkeypatch.setattr(cr, "_ERROR_BATCHER", control)
    monkeypatch.setattr(cr, "_ERROR_OCCURRENCE_POLICIES", {})
    enabled = [False]
    monkeypatch.setattr(cr, "_automatic_error_policy", lambda kind: (
        dict(provider="hosted", _mode="all" if enabled[0] else "send_all"), enabled[0], True))
    calls = []
    def dispatched(text, **kwargs):
        calls.append((text, cr._error_occurrence(kwargs["event_id"])[1]))
    monkeypatch.setattr(cr, "_dispatch_error_now", dispatched)
    cr._dispatch_error("first")
    clock.now = 1
    cr._dispatch_error("ValueError: send-only error")
    enabled[0] = True
    clock.now = 2
    cr._dispatch_error("KeyError: explainable error")
    clock.now = 62
    control._tick()
    assert calls[-2][1][1:] == (False, True)
    assert calls[-1][1][1:] == (True, True)
    assert "send-only error" not in calls[-1][0]


def test_backwards_wall_clock_reanchors_admission_and_cached_chat_is_not_an_error(isolated_guard):
    first = cr._reserve_hosted_request(error=True, now=100)
    moved = cr._reserve_hosted_request(error=True, now=90)
    assert moved['anchor'] == 90 and moved['minute'] == 0
    chat = cr._reserve_hosted_request(error=False, now=91)
    before = cr._read_state()
    cr._refund_cached_error(chat)
    assert cr._read_state() == before
    cr._refund_cached_error(first)
    assert cr._read_state()['ai_request_guard']['error_minutes'] == before['ai_request_guard']['error_minutes']


def test_earlier_cached_response_keeps_a_later_three_minute_saturated_window(isolated_guard):
    first = cr._reserve_hosted_request(error=True, now=0)
    for minute in (1, 2, 3):
        for second in (0, 1, 2):
            cr._reserve_hosted_request(error=True, now=minute * 61 + second)
    before = cr._read_state()['ai_request_guard']
    cr._refund_cached_error(first)
    guard = cr._read_state()['ai_request_guard']
    assert guard['cooldown_window'] == before['cooldown_window']
    assert guard['error_minutes'] == before['error_minutes']
    assert guard['error_cooldown_until'] > 185
    with pytest.raises(cr.RelayRefused, match='cooldown'):
        cr._reserve_hosted_request(error=True, now=186)


def test_stale_timer_cannot_flush_a_new_batch_and_early_timer_rearms():
    control, clock, sent, _, timers, deliver = batcher()
    control.submit(dict(text='first'), deliver)
    stale = timers[-1][1]
    clock.now = 1
    control.submit(dict(text='second'), deliver)
    clock.now = 60
    stale()
    assert len(sent) == 1 and len(timers) == 2
    obsolete = timers[-1][1]
    clock.now = 61
    control._tick()
    obsolete()
    assert [[entry['text'] for entry, count in batch] for batch, _ in sent] == [['first'], ['second']]


@pytest.mark.parametrize('failing', ['report-only', 'mixed-report', 'ai-owner'])
def test_one_broken_batch_recipient_cannot_poison_later_delivery(failing):
    seen = []
    def broken(*args):
        raise RuntimeError('recipient closed')
    if failing == 'report-only':
        payload = dict(text='old', provider='none', report_only=broken,
                       ai_eligible=lambda: False, report_eligible=lambda: True)
        entries = [[payload, 1, broken]]
    elif failing == 'mixed-report':
        entries = [[dict(text='automatic', provider='none', report_only=broken,
                         report_eligible=lambda: True), 1, broken],
                   [dict(text='manual', provider='claude', manual=True), 1,
                    lambda batch, omitted: seen.append(batch)]]
    else:
        entries = [[dict(text='old', provider='claude'), 1, broken]]
    cr._ErrorBatcher._deliver(entries, 0)
    cr._ErrorBatcher._deliver([[dict(text='healthy', provider='claude'), 1,
                               lambda batch, omitted: seen.append(batch)]], 0)
    assert seen[-1] == [(dict(text='healthy', provider='claude'), 1)]
    if failing == 'mixed-report':
        assert seen[0] == [(dict(text='manual', provider='claude', manual=True), 1)]


def test_revoked_report_only_batch_drops_without_invoking_its_old_owner():
    called = []
    payload = dict(text='revoked', provider='none', report_only=lambda *args: called.append(args),
                   ai_eligible=lambda: False, report_eligible=lambda: False)
    cr._ErrorBatcher._deliver([[payload, 1, lambda *args: called.append(args)]], 0)
    assert called == []


def test_rate_refusal_preserves_report_without_requesting_a_fourth_ai_call(isolated_guard, monkeypatch):
    from tests.test_700_anonymous_error_reports import FakeRelay
    relay = FakeRelay()
    monkeypatch.setattr(cr, '_open_url', relay)
    for _ in range(3):
        cr._reserve_hosted_request(error=False)
    result = cr._send_one({'exception_type': 'RuntimeError', 'message': 'failed',
                           'frames': [], 'summarize': True}, 2)
    assert result['status'] == cr.REPORT_SENT
    assert result['ai_refusal']['reason'] == 'rate'
    [report] = relay.calls_to('/report')
    assert report['body']['summarize'] is False
    assert relay.calls_to('/diagnose') == []
