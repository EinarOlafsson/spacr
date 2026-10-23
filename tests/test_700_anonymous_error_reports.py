"""Item 700: anonymised error reports through spaCR's relay, no GitHub account.

What leaves the computer is :func:`spacr.crashreport.public_summary` and
nothing else: an allowlisted record, scrubbed of personal data, sized for the
relay's 16 KB limit. These tests plant personal data and look for it in the
output, check the exact schema, the queue's limits and retry rules, that the
switch defaults to on, and that with it off not one socket is opened.

No test reaches the network. ``tests/conftest.py`` sets ``SPACR_NO_NETWORK``
and replaces the transport; each test here that needs a relay installs a fake
``_open_url`` that answers like the relay's contract, and the socket module
is blocked wherever "nothing is sent" is the claim.
"""
from __future__ import annotations

import io
import json
import re
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

from spacr import crashreport as cr

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class _Response:
    """A urllib response: a context manager with ``status`` and ``read``."""

    def __init__(self, status, payload):
        self.status = status
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def getcode(self):
        return self.status

    def read(self, limit=-1):
        return self._body


class FakeRelay:
    """Answers ``/report`` and ``/diagnose`` the way the relay's contract says.

    ``answers`` is a list of ``(status, payload)`` or exceptions, used in
    order; the last one repeats.
    """

    def __init__(self, *answers):
        self.answers = list(answers) or [(200, {
            "issue_number": 12, "issue_url":
            "https://github.com/EinarOlafsson/spacr/issues/12",
            "deduplicated": False})]
        self.calls = []
        self.routes = {}

    def calls_to(self, route):
        """The calls made to one route, such as ``"/report"``."""
        return [c for c in self.calls if c["url"].split("?")[0].endswith(route)]

    def __call__(self, request, timeout):
        self.calls.append({"url": request.full_url,
                           "method": request.get_method(),
                           "timeout": timeout,
                           "body": json.loads(request.data.decode())
                           if request.data else None})
        route = "/" + request.full_url.split("?")[0].rsplit("/", 1)[-1]
        if route in self.routes:
            answers = self.routes[route]
            answer = answers[min(len(self.calls_to(route)) - 1,
                                 len(answers) - 1)]
        else:
            answer = self.answers[min(len(self.calls) - 1,
                                      len(self.answers) - 1)]
        if isinstance(answer, BaseException):
            raise answer
        status, payload = answer
        if status >= 400:
            raise urllib.error.HTTPError(
                request.full_url, status, "refused", {},
                io.BytesIO(json.dumps(payload).encode()))
        return _Response(status, payload)


@pytest.fixture
def relay(monkeypatch):
    """Allow the network for this test and install a fake relay."""
    monkeypatch.delenv(cr.NO_NETWORK_ENV, raising=False)
    monkeypatch.delenv(cr.REPORT_RELAY_ENV, raising=False)
    fake = FakeRelay()
    monkeypatch.setattr(cr, "_open_url", fake)
    return fake


@pytest.fixture
def blocked_sockets(monkeypatch):
    """Make every socket connection fail, and count the attempts.

    The transport is put back to the real ``urllib`` call, so the count
    covers everything the code under test could do, not only the fake.
    """
    attempts = []

    def refuse(*args, **kwargs):
        attempts.append(args)
        raise OSError("socket blocked by the test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(
        cr, "_open_url",
        lambda request, timeout: urllib.request.urlopen(request,
                                                        timeout=timeout))
    return attempts


def _raise_from_nested(depth, message):
    """Raise ``ValueError(message)`` ``depth`` calls deep."""
    if depth <= 0:
        raise ValueError(message)
    _raise_from_nested(depth - 1, message)


def _summary(message="plain failure", depth=2, **kwargs):
    """A real summary of a real exception."""
    try:
        _raise_from_nested(depth, message)
    except ValueError as exc:
        return cr.public_summary(exc, **kwargs)


class TestScrubbing:
    """Personal data planted in a message does not survive into a summary."""

    @pytest.fixture(autouse=True)
    def _identity(self, monkeypatch):
        """This computer is annasvensson on labhost42, home /home/annasvensson."""
        from spacr.qt.ai import issue_report

        monkeypatch.setattr(issue_report, "_identity_words", lambda: [
            ("annasvensson", "<USER>"), ("labhost42", "<HOST>")])
        monkeypatch.setattr(cr, "_home_variants", lambda: [
            "/home/annasvensson", "C:\\Users\\annasvensson"])

    PLANTED = {
        "posix path": "No such file: '/mnt/lab/Patient 042/plate1/A01.tif'",
        "unquoted posix": "opening /srv/screens/secret_project/x failed",
        "home": "cache at /home/annasvensson/.cache/models broken",
        "windows": r"cannot read C:\Users\annasvensson\OneDrive - KI\A01.nd2",
        "unc": r"\\LAB-NAS\screens\plate 1 missing",
        "user": "permission denied for annasvensson",
        "host": "lost connection to labhost42",
        "email": "contact anna.svensson@ki.se for access",
        "ipv4": "connect to 192.168.17.20 refused",
        "ipv6": "connect to fe80::1ff:fe23:4567:890a refused",
        "url": "GET https://anna:hunter2@data.example.com/plates?key=1",
        "github token": "token ghp_0123456789abcdefABCDEF0123 rejected",
        "hex": "session 3f9c1a2b3f9c1a2b3f9c1a2b3f9c1a2b expired",
        "base64": "key AbCdEf0123456789XyZaBcDeFgHiJ= invalid",
        "file": "could not parse measurements_patient7.csv",
        "hostname": "lab-nas.med.example.edu is unreachable",
    }
    LEAKS = ("Patient 042", "A01.tif", "secret_project", "annasvensson",
             "labhost42", "anna.svensson", "ki.se", "192.168.17.20",
             "fe80::", "hunter2", "data.example.com", "ghp_",
             "3f9c1a2b3f9c1a2b", "AbCdEf0123456789", "patient7",
             "LAB-NAS", "OneDrive", "lab-nas", ".cache/models")

    @pytest.mark.parametrize("kind", sorted(PLANTED))
    def test_each_planted_value_is_removed(self, kind):
        scrubbed = cr.scrub_public_text(self.PLANTED[kind], 300)
        for leak in self.LEAKS:
            assert leak not in scrubbed, (kind, scrubbed)

    def test_all_of_it_at_once_through_a_real_exception(self):
        summary = _summary(" ; ".join(self.PLANTED.values()))
        text = json.dumps(summary)
        for leak in self.LEAKS:
            assert leak not in text
        assert len(summary["message"]) <= cr.MAX_MESSAGE_CHARS

    def test_useful_words_are_kept(self):
        message = "'numpy.ndarray' object has no attribute 'shape2' and/or 10/20"
        assert cr.scrub_public_text(message) == message

    def test_frames_are_basenames_and_user_code_is_hidden(self):
        summary = _summary(depth=3)
        files = [frame["file"] for frame in summary["frames"]]
        assert all("/" not in f and "\\" not in f for f in files)
        assert files and set(files) == {"<user-code>"}, (
            "this test file is the user's own code, not a library")

    def test_a_text_traceback_keeps_library_basenames_only(self):
        text = (
            "Traceback (most recent call last):\n"
            '  File "/home/annasvensson/miniforge3/envs/spacr/lib/python3.11/'
            'site-packages/spacr/io.py", line 120, in load_plate\n'
            "    data = read(path)\n"
            '  File "C:\\Users\\annasvensson\\analysis\\my_screen.py", '
            "line 7, in read\n"
            "    raise FileNotFoundError(path)\n"
            "FileNotFoundError: [Errno 2] No such file or directory: "
            "'/home/annasvensson/plate7/B02.tif'\n")
        summary = cr.public_summary(traceback_text=text, module="Mask")
        assert summary["exception_type"] == "FileNotFoundError"
        assert summary["frames"] == [
            {"file": "io.py", "function": "load_plate", "line": 120},
            {"file": "<user-code>", "function": "read", "line": 7}]
        assert summary["module"] == "mask"
        assert "B02" not in summary["message"]
        assert "annasvensson" not in json.dumps(summary)


class TestSchema:
    """Exactly the allowlisted keys, within the relay's caps."""

    BASE = set(cr.PUBLIC_SUMMARY_KEYS) - {"user_note", "dry_run",
                                          "ai_summary", "summarize",
                                          "screen", "error_text", "chain",
                                          "settings", "inputs", "environment",
                                          "run", "log_tail", "cause", "native_session"}

    def test_exactly_the_allowlisted_keys(self):
        summary = _summary(module="measure")
        assert set(summary) == self.BASE
        assert summary["schema_version"] == cr.PUBLIC_SUMMARY_SCHEMA
        assert summary["exception_type"] == "ValueError"
        assert summary["os"] in {"linux", "windows", "macos", "other"}
        assert summary["gpu"] in {"nvidia", "amd", "apple", "none", "unknown"}
        assert summary["module"] == "measure"
        for frame in summary["frames"]:
            assert set(frame) == {"file", "function", "line"}
            assert isinstance(frame["line"], int)

    def test_note_and_dry_run_are_the_only_optional_keys(self):
        summary = _summary(user_note="I pressed Run twice", dry_run=True)
        assert set(summary) == self.BASE | {"user_note", "dry_run"}
        assert summary["dry_run"] is True

    def test_caps_hold_for_huge_inputs(self):
        summary = _summary("x " * 10000, depth=80, user_note="y " * 5000)
        assert len(summary["message"]) <= cr.MAX_MESSAGE_CHARS
        assert len(summary["user_note"]) <= cr.MAX_NOTE_CHARS
        assert len(summary["frames"]) <= cr.MAX_FRAMES
        assert summary["frames"][-1]["function"] == "_raise_from_nested", (
            "the innermost frames are the ones kept")
        assert len(json.dumps(summary).encode()) <= cr.MAX_SUMMARY_BYTES

    def test_unknown_keys_never_survive_validation(self):
        dirty = dict(_summary(), log="secret", home={"HOME": "/home/x"})
        assert set(cr._validated_summary(dirty)) == self.BASE
        kept = cr._validated_summary(dict(_summary(), settings={"src": "/data"}))
        assert "/data" not in json.dumps(kept)

    def test_install_id_is_random_and_stable(self):
        first = cr.install_id()
        assert UUID_RE.match(first)
        assert cr.install_id() == first
        assert _summary()["install_id"] == first

    def test_the_example_shows_the_planted_values_removed(self):
        text = json.dumps(cr.example_public_summary())
        assert "patient_042" not in text and "example.org" not in text
        assert "ghp_" not in text and "<PATH>" in text


class TestDefaultsAndConsent:
    """On by default, disclosed once, and off really means off."""

    def test_reports_are_on_by_default(self):
        assert cr.DEFAULT_ANONYMOUS_REPORTS is True
        assert cr.anonymous_reports_enabled() is True

    def test_the_notice_is_due_once(self):
        assert cr.anonymous_report_notice_due() is True
        cr.mark_anonymous_report_notice_shown()
        assert cr.anonymous_report_notice_due() is False

    def test_the_switch_persists(self):
        cr.set_anonymous_reports_enabled(False)
        assert cr.anonymous_reports_enabled() is False
        cr.set_anonymous_reports_enabled(True)
        assert cr.anonymous_reports_enabled() is True

    def test_off_sends_nothing_with_sockets_blocked(self, monkeypatch,
                                                    blocked_sockets):
        monkeypatch.delenv(cr.NO_NETWORK_ENV, raising=False)
        cr.set_anonymous_reports_enabled(False)
        summary = _summary()
        assert cr.submit_public_summary(summary) == {
            "status": cr.REPORT_DISABLED}
        assert cr.submit_public_summary_in_background(summary) is None
        assert cr.flush_report_queue() == {"sent": 0, "dropped": 0,
                                           "kept": 0}
        cr._report_anonymously(ValueError("x"), "mask")
        assert cr.queued_summaries() == []
        assert blocked_sockets == []

    def test_turning_off_discards_the_queue(self, relay):
        relay.answers = [OSError("offline")]
        cr.submit_public_summary(_summary())
        assert len(cr.queued_summaries()) == 1
        cr.set_anonymous_reports_enabled(False)
        assert cr.queued_summaries() == []

    @pytest.mark.parametrize("value", ["1", "true", "yes"])
    def test_no_network_flag_sends_nothing(self, monkeypatch, blocked_sockets,
                                           value):
        monkeypatch.setenv(cr.NO_NETWORK_ENV, value)
        assert cr.network_allowed() is False
        assert cr.submit_public_summary(_summary())["status"] == \
            cr.REPORT_DISABLED
        with pytest.raises(cr.RelayRefused) as refused:
            cr.diagnose_remotely(_summary(), "why?")
        assert refused.value.reason == "network"
        assert blocked_sockets == []

    def test_safe_mode_sends_nothing(self, monkeypatch, blocked_sockets):
        import spacr.logging_util as lu

        monkeypatch.delenv(cr.NO_NETWORK_ENV, raising=False)
        monkeypatch.setattr(lu, "_in_safe_mode", lambda: True)
        assert cr.submit_public_summary(_summary())["status"] == \
            cr.REPORT_DISABLED
        assert blocked_sockets == []


class TestSending:
    """The post itself: route, body, outcomes."""

    def test_a_report_is_posted_as_json_with_a_short_timeout(self, relay):
        outcome = cr.submit_public_summary(_summary())
        assert outcome == {
            "status": cr.REPORT_SENT, "issue_number": 12,
            "issue_url": "https://github.com/EinarOlafsson/spacr/issues/12",
            "deduplicated": False, "ai_summary_source": "none"}
        call = relay.calls[0]
        assert call["url"] == cr.REPORT_RELAY_URL + "/report"
        assert call["method"] == "POST"
        assert call["timeout"] <= 10
        assert set(call["body"]) <= set(cr.PUBLIC_SUMMARY_KEYS)
        assert cr.queued_summaries() == []

    def test_the_relay_can_be_overridden(self, relay, monkeypatch):
        monkeypatch.setenv(cr.REPORT_RELAY_ENV, "https://relay.test/")
        cr.submit_public_summary(_summary())
        assert relay.calls[0]["url"] == "https://relay.test/report"

    def test_an_insecure_override_is_ignored(self, monkeypatch):
        monkeypatch.setenv(cr.REPORT_RELAY_ENV, "http://evil.example.com")
        assert cr.relay_url() == cr.REPORT_RELAY_URL
        monkeypatch.setenv(cr.REPORT_RELAY_ENV, "http://127.0.0.1:8787")
        assert cr.relay_url() == "http://127.0.0.1:8787"

    def test_a_deduplicated_report_says_so(self, relay):
        relay.answers = [(200, {"issue_number": 5, "issue_url": "u",
                                "deduplicated": True})]
        assert cr.submit_public_summary(_summary())["deduplicated"] is True

    def test_the_excepthook_sends_the_summary_not_the_bundle(
            self, relay, tmp_path, monkeypatch):
        import sys

        monkeypatch.setattr(cr, "report_exception", lambda *a, **k: "")
        monkeypatch.setattr(sys, "excepthook", lambda *a: None)
        cr._update_state(affirmed_policy={
            "version": cr._REPORTING_TERMS_VERSION,
            "accepted_at": "2026-10-23T12:00:00+00:00",
            "explain": False, "auto_file": True, "prompt": "always", "provider": "none",
        })
        hook = cr.install_excepthook(tmp_path, app="measure")
        try:
            _raise_from_nested(1, "boom")
        except ValueError as exc:
            hook(type(exc), exc, exc.__traceback__)
        deadline = time.monotonic() + 5
        while not relay.calls and time.monotonic() < deadline:
            time.sleep(0.01)
        body = relay.calls[0]["body"]
        assert body["module"] == "measure" and body["message"] == "boom"
        assert set(body) <= set(cr.PUBLIC_SUMMARY_KEYS)


class TestQueue:
    """At most 20, at most 7 days, retried on 429 or no network, dropped on 4xx."""

    def test_429_keeps_the_report_for_later(self, relay):
        relay.answers = [(429, {"refused": "quota"})]
        assert cr.submit_public_summary(_summary())["status"] == \
            cr.REPORT_QUEUED
        assert len(cr.queued_summaries()) == 1

    @pytest.mark.parametrize("error", [
        urllib.error.URLError("no route"), TimeoutError("slow"),
        ConnectionResetError("reset")])
    def test_a_network_error_keeps_the_report(self, relay, error):
        relay.answers = [error]
        assert cr.submit_public_summary(_summary())["status"] == \
            cr.REPORT_QUEUED
        assert len(cr.queued_summaries()) == 1

    def test_a_server_error_keeps_the_report(self, relay):
        relay.answers = [(503, {"error": "busy"})]
        assert cr.submit_public_summary(_summary())["status"] == \
            cr.REPORT_QUEUED
        assert len(cr.queued_summaries()) == 1

    def test_a_4xx_drops_the_report(self, relay):
        relay.answers = [(400, {"error": "unknown field"})]
        outcome = cr.submit_public_summary(_summary())
        assert outcome == {"status": cr.REPORT_DROPPED,
                           "detail": "unknown field"}
        assert cr.queued_summaries() == []

    def test_at_most_twenty_are_kept_oldest_dropped(self, relay):
        relay.answers = [OSError("offline")]
        for n in range(25):
            cr.submit_public_summary(_summary(f"error {n}"))
        kept = [s["message"] for s in cr.queued_summaries()]
        assert len(kept) == cr.QUEUE_LIMIT == 20
        assert kept[0] == "error 5" and kept[-1] == "error 24"

    def test_reports_older_than_seven_days_are_deleted(self, relay):
        relay.answers = [OSError("offline")]
        cr.submit_public_summary(_summary("old"))
        cr.submit_public_summary(_summary("new"))
        old = sorted(cr._queue_dir().glob("*.json"))[0]
        entry = json.loads(old.read_text(encoding="utf-8"))
        entry["queued_at"] = time.time() - cr.QUEUE_RETENTION_S - 60
        old.write_text(json.dumps(entry), encoding="utf-8")
        assert [s["message"] for s in cr.queued_summaries()] == ["new"]
        assert cr.QUEUE_RETENTION_S == 7 * 24 * 3600

    def test_a_flush_sends_everything_waiting(self, relay):
        relay.answers = [OSError("offline")]
        for n in range(3):
            cr.submit_public_summary(_summary(f"error {n}"))
        relay.answers = [(200, {"issue_number": 1, "issue_url": "u",
                                "deduplicated": False})]
        before = len(relay.calls)
        assert cr.flush_report_queue() == {"sent": 3, "dropped": 0,
                                           "kept": 0}
        assert [c["body"]["message"] for c in relay.calls[before:]] == [
            "error 0", "error 1", "error 2"]
        assert cr.queued_summaries() == []

    def test_a_flush_stops_at_the_first_429(self, relay):
        relay.answers = [OSError("offline")]
        for n in range(3):
            cr.submit_public_summary(_summary(f"error {n}"))
        relay.answers = [(429, {"refused": "quota"})]
        before = len(relay.calls)
        assert cr.flush_report_queue()["kept"] == 3
        assert len(relay.calls) - before == 1, "one try, not three"

    def test_a_flush_drops_rejected_and_sends_the_rest(self, relay):
        relay.answers = [OSError("offline")]
        for n in range(2):
            cr.submit_public_summary(_summary(f"error {n}"))
        relay.answers = [(400, {"error": "bad"}),
                         (200, {"issue_number": 2, "issue_url": "u",
                                "deduplicated": False})]
        relay.calls.clear()
        assert cr.flush_report_queue() == {"sent": 1, "dropped": 1,
                                           "kept": 0}

    def test_a_successful_send_also_flushes_the_backlog(self, relay):
        relay.answers = [OSError("offline")]
        cr.submit_public_summary(_summary("waiting"))
        relay.answers = [(200, {"issue_number": 3, "issue_url": "u",
                                "deduplicated": False})]
        cr.submit_public_summary(_summary("now"))
        assert cr.queued_summaries() == []


class TestBackground:
    """Sending never holds up the caller."""

    def test_the_background_send_returns_at_once(self, relay, monkeypatch):
        release = threading.Event()
        answered = threading.Event()
        outcomes = []
        slow = relay

        def slow_relay(request, timeout):
            release.wait(10)
            return slow(request, timeout)

        monkeypatch.setattr(cr, "_open_url", slow_relay)
        if True:
            started = time.monotonic()
            thread = cr.submit_public_summary_in_background(
                _summary(), on_done=lambda o: (outcomes.append(o),
                                               answered.set()))
            elapsed = time.monotonic() - started
            assert elapsed < 0.5, f"the caller waited {elapsed:.2f} s"
            assert thread is not None and thread.daemon
            assert len(cr.queued_summaries()) == 1, (
                "queued before returning, so a crash still sends it later")
            release.set()
            assert answered.wait(5)
        assert outcomes[0]["status"] == cr.REPORT_SENT
        thread.join(5)
        assert cr.queued_summaries() == []


class TestDiagnosis:
    """Hosted diagnosis: the summary plus a question, advice back."""

    def test_advice_comes_back(self, relay):
        relay.answers = [(200, {"advice": "Check the channel order.",
                                "model": "qwen3-30b"})]
        answer = cr.diagnose_remotely(_summary(), "why does it fail?")
        assert answer == {"advice": "Check the channel order.",
                          "model": "qwen3-30b", "tier": "best",
                          "cached": False, "cause": ""}
        call = relay.calls[0]
        assert call["url"].endswith("/diagnose")
        assert call["body"]["question"] == "why does it fail?"
        assert set(call["body"]) <= set(cr.PUBLIC_SUMMARY_KEYS) | {"question"}
        assert "user_note" not in call["body"]

    def test_the_question_is_scrubbed_and_capped(self, relay):
        relay.answers = [(200, {"advice": "ok", "model": "m"})]
        cr.diagnose_remotely(_summary(), "see /data/secret/plate.tif " * 400)
        question = relay.calls[0]["body"]["question"]
        assert len(question) <= cr.MAX_QUESTION_CHARS
        assert "secret" not in question and "plate.tif" not in question

    def test_429_is_a_quota_refusal(self, relay):
        relay.answers = [(429, {"refused": "quota"})]
        with pytest.raises(cr.RelayRefused) as refused:
            cr.diagnose_remotely(_summary())
        assert refused.value.reason == "quota"

    def test_no_connection_is_a_network_refusal(self, relay):
        relay.answers = [urllib.error.URLError("offline")]
        with pytest.raises(cr.RelayRefused) as refused:
            cr.diagnose_remotely(_summary())
        assert refused.value.reason == "network"


class TestDailyAllowances:
    """5 reports a day per install; 5 best plus 10 light AI questions."""

    ALLOWANCE = {"ai": {"best_left": 4, "light_left": 10}, "reports_left": 5,
                 "pool": "best", "resets_at": "2026-10-15T00:00:00Z"}

    def test_the_allowance_is_a_get_that_names_only_the_install(self, relay):
        relay.answers = [(200, self.ALLOWANCE)]
        allowance = cr.fetch_allowance()
        call = relay.calls[0]
        assert call["method"] == "GET" and call["body"] is None
        assert call["url"] == (cr.REPORT_RELAY_URL + "/allowance?install_id="
                               + cr.install_id())
        assert allowance == {"ai": {"best_left": 4, "light_left": 10},
                             "reports_left": 5, "pool": "best",
                             "resets_at": "2026-10-15T00:00:00+00:00"}

    def test_no_allowance_without_network_or_answer(self, relay, monkeypatch):
        relay.answers = [OSError("offline")]
        assert cr.fetch_allowance() is None
        monkeypatch.setenv(cr.NO_NETWORK_ENV, "1")
        calls = len(relay.calls)
        assert cr.fetch_allowance() is None
        assert len(relay.calls) == calls

    def test_a_report_carries_its_allowance_back(self, relay):
        relay.answers = [(200, {"issue_number": 9, "issue_url": "u",
                                "deduplicated": False,
                                "allowance": {"reports_left": 3,
                                              "resets_at":
                                              "2026-10-15T00:00:00Z"}})]
        outcome = cr.submit_public_summary(_summary())
        assert outcome["allowance"] == {
            "reports_left": 3, "resets_at": "2026-10-15T00:00:00+00:00"}

    def test_a_quota_refusal_waits_for_the_reset(self, relay):
        relay.answers = [(429, {"refused": "quota", "scope": "install"})]
        outcome = cr.submit_public_summary(_summary("first"))
        assert outcome["quota"] is True and outcome["scope"] == "install"
        assert outcome["resets_at"] == cr.next_reset_utc().isoformat()
        calls = len(relay.calls)
        second = cr.submit_public_summary(_summary("second"))
        assert second["status"] == cr.REPORT_QUEUED and second["quota"]
        assert cr.flush_report_queue()["kept"] == 2
        assert len(relay.calls) == calls, "nothing posted before the reset"
        assert [s["message"] for s in cr.queued_summaries()] == [
            "first", "second"]

    def test_after_the_reset_the_queue_is_sent(self, relay):
        relay.answers = [(429, {"refused": "quota", "scope": "install"})]
        cr.submit_public_summary(_summary("waiting"))
        cr._update_state(reports_paused_until=time.time() - 1)
        relay.answers = [(200, {"issue_number": 1, "issue_url": "u",
                                "deduplicated": False})]
        assert cr.flush_report_queue()["sent"] == 1
        assert cr.queued_summaries() == []

    def test_the_next_reset_is_utc_midnight(self):
        from datetime import datetime, timezone

        now = datetime(2026, 10, 14, 23, 59, tzinfo=timezone.utc)
        assert cr.next_reset_utc(now) == datetime(2026, 10, 15,
                                                  tzinfo=timezone.utc)

    def test_diagnosis_reports_its_tier_and_allowance(self, relay):
        relay.answers = [(200, {"advice": "a", "model": "light-8b",
                                "tier": "light",
                                "allowance": {"best_left": 0,
                                              "light_left": 9,
                                              "resets_at":
                                              "2026-10-15T00:00:00Z"}})]
        answer = cr.diagnose_remotely(_summary())
        assert answer["tier"] == "light"
        assert answer["allowance"]["ai"] == {"best_left": 0, "light_left": 9}

    @pytest.mark.parametrize("scope", ["install", "global", "ip"])
    def test_a_diagnosis_quota_refusal_names_its_scope(self, relay, scope):
        relay.answers = [(429, {"refused": "quota", "scope": scope})]
        with pytest.raises(cr.RelayRefused) as refused:
            cr.diagnose_remotely(_summary())
        assert refused.value.reason == "quota"
        assert refused.value.scope == scope
        assert refused.value.resets_at == cr.next_reset_utc()


class TestDynamicAllowanceAndModes:
    """The relay's dynamic share, and the AI Error report choices."""

    def test_the_dynamic_fields_are_read(self, relay):
        relay.answers = [(200, {
            "ai": {"best_left": 3, "light_left": 10, "neurons_left": 600,
                   "share_neurons": 1000, "share_pct": 2.5,
                   "active_users": 40, "tier_now": "light"},
            "reports_left": 5, "pool": "light", "pool_pct_left": 12.5,
            "resets_at": "2026-10-15T00:00:00Z"})]
        allowance = cr.fetch_allowance()
        assert allowance["ai"] == {"best_left": 3, "light_left": 10,
                                   "neurons_left": 600, "share_neurons": 1000,
                                   "share_pct": 2.5, "active_users": 40,
                                   "tier_now": "light"}
        assert allowance["pool_pct_left"] == 12.5

    def test_the_default_choice(self):
        assert cr.error_report_mode() == "all"
        assert cr.mode_explains() and cr.mode_sends("fault")
        assert cr.mode_sends("user_error") and not cr.mode_code_context()

    def test_the_old_off_switch_is_kept(self):
        cr._update_state(enabled=False)
        assert cr.error_report_mode() == "off"

    def test_an_unknown_choice_is_refused(self):
        with pytest.raises(ValueError):
            cr.set_error_report_mode("everything")

    def test_a_choice_that_sends_nothing_clears_the_queue(self, relay):
        relay.answers = [OSError("offline")]
        cr.submit_public_summary(_summary())
        cr.set_error_report_mode("summarize")
        assert cr.queued_summaries() == []
        assert cr.anonymous_reports_enabled() is False

    def test_kinds(self):
        from spacr.errors import ConfigurationError

        assert cr.classify_failure(ConfigurationError("x")) == "user_error"
        assert cr.classify_failure(ValueError("x")) == "fault"
        assert cr.classify_failure(
            traceback_text="spacr.errors.ConfigurationError: x") == \
            "user_error"
        assert _summary()["kind"] == "fault"

def test_a_drive_relative_windows_path_is_scrubbed():
    """`\\lab-share\\plate1` -- a rooted path without a drive letter, which is
    how Windows prints the parent of `/lab-share/plate1` -- once went out
    in an issue report untouched."""
    from spacr import crashreport

    text = crashreport.scrub_public_text(
        "src does not exist (its parent folder \\secret-lab-700\\plate1 does "
        "not exist either)")
    assert "secret-lab-700" not in text
    assert "<PATH>" in text
