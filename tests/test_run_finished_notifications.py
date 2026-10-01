"""A long run that finishes or fails says so: desktop, email, Slack, ntfy.

Every channel is exercised against a fake on 127.0.0.1 -- a small SMTP
server and an HTTP server started by the test -- so nothing leaves the
machine. What is pinned:

* a finished run and a failed run each send exactly one notification per
  channel, carrying the run name, duration, outcome, the QC summary and
  where the output is;
* a cancelled run, a finished run when only failures are wanted, and a run
  shorter than the minimum send nothing;
* a broken or slow channel never fails, blocks or crashes the run, and the
  other channels still get through;
* secrets live in the OS keyring when there is one, otherwise in a mode-600
  file, and never reach a log line.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import socketserver
import stat
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from spacr import run_journal
from spacr.errors import RunLedger


class _FakeSmtp(socketserver.ThreadingTCPServer):
    """Just enough SMTP to receive a message: EHLO, AUTH PLAIN, MAIL, DATA."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self):
        self.messages = []
        self.logins = []
        super().__init__(("127.0.0.1", 0), _SmtpHandler)


class _SmtpHandler(socketserver.StreamRequestHandler):
    def handle(self):
        def say(text):
            self.wfile.write((text + "\r\n").encode())

        say("220 fake ESMTP")
        while True:
            raw = self.rfile.readline()
            if not raw:
                return
            line = raw.decode(errors="replace").rstrip("\r\n")
            verb = line.split(" ", 1)[0].upper()
            if verb == "EHLO":
                say("250-fake")
                say("250 AUTH PLAIN")
            elif verb == "HELO":
                say("250 fake")
            elif verb == "AUTH":
                token = line.split(" ")[-1]
                self.server.logins.append(
                    base64.b64decode(token).decode().split("\0"))
                say("235 ok")
            elif verb == "DATA":
                say("354 go ahead")
                body = []
                while True:
                    data = self.rfile.readline().decode(errors="replace")
                    if data in (".\r\n", ".\n", ""):
                        break
                    body.append(data)
                self.server.messages.append("".join(body))
                say("250 queued")
            elif verb == "QUIT":
                say("221 bye")
                return
            else:
                say("250 ok")


class _FakeHttp(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        self.posts = []
        super().__init__(("127.0.0.1", 0), _HttpHandler)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


class _HttpHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        self.server.posts.append(
            {"path": self.path, "headers": dict(self.headers), "body": body})
        code = 500 if self.path.startswith("/broken") else 200
        self.send_response(code)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        return


@pytest.fixture
def smtp():
    server = _FakeSmtp()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def http():
    server = _FakeHttp()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def sent(monkeypatch):
    """Every sending thread started, so a test can wait for it."""
    threads = []
    real = run_journal._dispatch_notification

    def record(message, config):
        thread = real(message, config)
        threads.append(thread)
        return thread

    monkeypatch.setattr(run_journal, "_dispatch_notification", record)
    return threads


@pytest.fixture
def desktop(monkeypatch):
    """The desktop channel, caught before it reaches the real desktop."""
    shown = []
    monkeypatch.setattr(run_journal, "_DESKTOP_NOTIFIER",
                        [lambda title, body, failed: shown.append(
                            (title, body, failed))])
    return shown


def _config(smtp, http, **extra):
    config = {
        "enabled": True, "when": "always", "min_minutes": 0,
        "desktop": True, "email": True, "slack": True, "ntfy": True,
        "teams": False, "webhook": False,
        "smtp_host": "127.0.0.1", "smtp_port": smtp.server_address[1],
        "smtp_security": "none", "smtp_user": "lab",
        "email_from": "spacr@lab.example", "email_to": "me@lab.example",
        "ntfy_server": http.url,
    }
    config.update(extra)
    return config


@pytest.fixture
def secrets(http):
    run_journal._store_notify_secret("smtp_password", "hunter2-mail")
    run_journal._store_notify_secret("slack_webhook",
                                     f"{http.url}/services/T0/B0/xyzsecret")
    run_journal._store_notify_secret("ntfy_topic", "spacr-lab-topic-9f3")
    run_journal._store_notify_secret("ntfy_token", "tk_secret_token")
    run_journal._store_notify_secret("teams_webhook",
                                     f"{http.url}/teams?sig=teams-secret")
    run_journal._store_notify_secret("webhook_url",
                                     f"{http.url}/webhook?key=webhook-secret")
    run_journal._store_notify_secret("webhook_token", "webhook-bearer-secret")
    yield
    for name in run_journal._NOTIFY_SECRET_NAMES:
        run_journal._store_notify_secret(name, "")


def _configure(monkeypatch, config):
    monkeypatch.setattr(run_journal, "_notification_config", lambda: config)


def _mail(smtp, index=0):
    """Subject and decoded body of a message the fake server received."""
    import email

    parsed = email.message_from_string(smtp.messages[index])
    return (parsed["Subject"],
            parsed.get_payload(decode=True).decode("utf-8"))


def _join(threads):
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()


def test_a_finished_run_sends_one_notification_with_its_summary(
        monkeypatch, smtp, http, secrets, sent, desktop):
    _configure(monkeypatch, _config(smtp, http, teams=True, webhook=True))
    with run_journal.open_run("mask", {"src": "/data/plate1"}) as run:
        ledger = RunLedger("preprocess_generate_masks")
        for field in range(4):
            with ledger.item(f"field{field}", stage="mask"):
                if field == 3:
                    raise ValueError("unreadable field")
        ledger.finalize()
    _join(sent)

    assert len(sent) == 1
    assert sent[0].results == {"desktop": "sent", "email": "sent",
                               "slack": "sent", "ntfy": "sent",
                               "teams": "sent", "webhook": "sent"}
    assert len(smtp.messages) == 1
    subject, mail = _mail(smtp)
    assert subject == "spaCR run finished: mask"
    for expected in ("Outcome: finished", "Duration: ",
                     "preprocess_generate_masks: 3 of 4 items processed, "
                     "1 failed", "Output: /data/plate1",
                     f"Run record: {run.dir}"):
        assert expected in mail
    assert smtp.logins == [["", "lab", "hunter2-mail"]]

    slack = [p for p in http.posts if p["path"].startswith("/services/")]
    ntfy = [p for p in http.posts if p["path"] == "/spacr-lab-topic-9f3"]
    assert len(slack) == 1 and len(ntfy) == 1
    text = json.loads(slack[0]["body"])["text"]
    assert text.startswith("*spaCR run finished: mask*")
    assert "3 of 4 items processed, 1 failed" in text
    assert ntfy[0]["headers"]["Title"] == "spaCR run finished: mask"
    assert ntfy[0]["headers"]["Priority"] == "default"
    assert ntfy[0]["headers"]["Authorization"] == "Bearer tk_secret_token"
    assert b"Outcome: finished" in ntfy[0]["body"]
    assert len(desktop) == 1
    title, body, failed = desktop[0]
    assert title == "spaCR run finished: mask" and failed is False
    assert "Outcome: finished" in body and "Output: /data/plate1" in body
    teams = [p for p in http.posts if p["path"].startswith("/teams?")]
    webhook = [p for p in http.posts if p["path"].startswith("/webhook?")]
    assert len(teams) == 1 and len(webhook) == 1
    assert teams[0]["headers"]["Content-Type"] == "application/json"
    payload = json.loads(teams[0]["body"])
    assert payload["type"] == "message"
    assert len(payload["attachments"]) == 1
    attachment = payload["attachments"][0]
    assert attachment["contentType"] == "application/vnd.microsoft.card.adaptive"
    assert attachment["contentUrl"] is None
    card = attachment["content"]
    assert card["$schema"] == "http://adaptivecards.io/schemas/adaptive-card.json"
    assert card["type"] == "AdaptiveCard" and card["version"] == "1.2"
    assert all(block["type"] == "TextBlock" and block["wrap"]
               for block in card["body"])
    assert card["body"][0]["weight"] == "Bolder"
    assert [block["text"] for block in card["body"]] == [title, *body.splitlines()]
    assert json.loads(webhook[0]["body"]) == {
        "title": title, "body": body, "failed": False}
    assert webhook[0]["headers"]["Content-Type"] == "application/json"
    assert webhook[0]["headers"]["Authorization"] == "Bearer webhook-bearer-secret"


def test_a_failed_run_sends_one_notification_and_still_raises(
        monkeypatch, smtp, http, secrets, sent, desktop):
    _configure(monkeypatch, _config(smtp, http, teams=True, webhook=True))
    with pytest.raises(RuntimeError, match="boom"):
        with run_journal.open_run("measure", {"src": "/data/plate2"}):
            raise RuntimeError("boom")
    _join(sent)

    assert len(sent) == 1
    assert set(sent[0].results.values()) == {"sent"}
    assert len(smtp.messages) == 1
    subject, mail = _mail(smtp)
    assert subject == "spaCR run failed: measure"
    assert "Outcome: failed" in mail
    assert "Error: RuntimeError: boom" in mail
    ntfy = [p for p in http.posts if p["path"] == "/spacr-lab-topic-9f3"]
    assert len(ntfy) == 1
    assert ntfy[0]["headers"]["Priority"] == "high"
    assert ntfy[0]["headers"]["Tags"] == "x"
    assert desktop[0][0] == "spaCR run failed: measure"
    assert desktop[0][2] is True
    teams = [p for p in http.posts if p["path"].startswith("/teams?")]
    webhook = [p for p in http.posts if p["path"].startswith("/webhook?")]
    assert len(teams) == 1 and len(webhook) == 1
    blocks = json.loads(teams[0]["body"])["attachments"][0]["content"]["body"]
    assert blocks[0]["text"] == "spaCR run failed: measure"
    assert "Error: RuntimeError: boom" in [block["text"] for block in blocks]
    payload = json.loads(webhook[0]["body"])
    assert payload["title"] == "spaCR run failed: measure"
    assert payload["failed"] is True
    assert "Error: RuntimeError: boom" in payload["body"]


def test_runs_that_should_stay_quiet_send_nothing(monkeypatch, smtp, http,
                                                  sent, desktop):
    from spacr.cancellation import PipelineCancelled

    _configure(monkeypatch, _config(smtp, http))
    with pytest.raises(PipelineCancelled):
        with run_journal.open_run("mask", {}):
            raise PipelineCancelled("stopped by the user")
    _configure(monkeypatch, _config(smtp, http, when="failed"))
    with run_journal.open_run("mask", {}):
        pass
    _configure(monkeypatch, _config(smtp, http, min_minutes=60))
    with pytest.raises(RuntimeError):
        with run_journal.open_run("mask", {}):
            raise RuntimeError("quick failure")
    _configure(monkeypatch, None)
    with run_journal.open_run("mask", {}):
        pass
    assert sent == [] and desktop == [] and smtp.messages == []
    assert http.posts == []

    _configure(monkeypatch, _config(smtp, http, when="failed", email=False,
                                    slack=False, ntfy=False))
    with pytest.raises(RuntimeError):
        with run_journal.open_run("mask", {}):
            raise RuntimeError("the one that counts")
    _join(sent)
    assert len(desktop) == 1


def test_a_broken_channel_fails_alone_and_logs_no_secret(
        monkeypatch, smtp, http, sent, desktop, caplog):
    run_journal._store_notify_secret("smtp_password", "hunter2-mail")
    run_journal._store_notify_secret(
        "slack_webhook", f"{http.url}/broken/xyzsecret")
    run_journal._store_notify_secret("ntfy_topic", "spacr-lab-topic-9f3")
    closed_port = smtp.server_address[1]
    smtp.shutdown()
    smtp.server_close()
    _configure(monkeypatch, _config(smtp, http, smtp_port=closed_port))
    caplog.set_level(logging.DEBUG)
    with run_journal.open_run("mask", {}) as run:
        pass
    _join(sent)
    for name in run_journal._NOTIFY_SECRET_NAMES:
        run_journal._store_notify_secret(name, "")

    assert run.status == "success"
    results = sent[0].results
    assert results["desktop"] == "sent" and results["ntfy"] == "sent"
    assert results["email"] != "sent" and results["slack"] != "sent"
    assert "500" in results["slack"]
    logged = caplog.text + json.dumps(results)
    for secret in ("hunter2-mail", "xyzsecret", "spacr-lab-topic-9f3"):
        assert secret not in logged
    assert "run notification by slack failed" in caplog.text


def test_a_slow_channel_never_holds_up_the_run(monkeypatch, smtp, http,
                                               sent):
    release = threading.Event()
    entered = threading.Event()

    def stuck(title, body, failed):
        entered.set()
        release.wait(30)

    monkeypatch.setattr(run_journal, "_DESKTOP_NOTIFIER", [stuck])
    _configure(monkeypatch, _config(smtp, http, email=False, slack=False,
                                    ntfy=False))
    with run_journal.open_run("mask", {}) as run:
        pass
    assert run.status == "success"
    assert entered.wait(10)
    assert sent[0].is_alive()
    release.set()
    _join(sent)
    assert sent[0].results == {"desktop": "sent"}


def test_a_generic_webhook_can_receive_without_a_bearer_token(http):
    """A URL alone works and JSON preserves Unicode and multiline summaries.

    :param http: the local HTTP notification receiver.
    """
    message = {"title": "spaCR: plåt 1", "body": "QC: 3 of 4\nOutput: /data/α",
               "failed": False}
    run_journal._notify_by_webhook(message, f"{http.url}/generic", "")
    assert len(http.posts) == 1
    assert json.loads(http.posts[0]["body"]) == message
    assert "Authorization" not in http.posts[0]["headers"]


@pytest.mark.parametrize("broken", ["teams", "webhook"])
def test_a_broken_new_webhook_does_not_stop_other_channels(http, broken,
                                                          caplog):
    """A local failure is reported while the other new transport still sends.

    :param http: the local HTTP notification receiver.
    :param broken: the adapter whose URL returns a server error.
    :param caplog: captured notification warnings.
    """
    urls = {channel: f"{http.url}/{'broken' if channel == broken else 'ok'}/"
                     f"{channel}-secret" for channel in ("teams", "webhook")}
    results = run_journal._send_notification(
        {"title": "spaCR run finished: mask", "body": "Outcome: finished",
         "failed": False},
        {"teams": True, "webhook": True, "secrets": {
            "teams_webhook": urls["teams"], "webhook_url": urls["webhook"],
            "webhook_token": "bearer-secret"}})
    other = "webhook" if broken == "teams" else "teams"
    assert "500" in results[broken]
    assert results[other] == "sent"
    assert len(http.posts) == 2
    assert f"run notification by {broken} failed" in caplog.text
    for value in (*urls.values(), "teams-secret", "webhook-secret", "bearer-secret"):
        assert value not in caplog.text + json.dumps(results)


@pytest.mark.parametrize("channel,secret_name", [
    ("teams", "teams_webhook"), ("webhook", "webhook_url")])
@pytest.mark.parametrize("address", ["", "file:///tmp/local-secret"])
def test_new_webhooks_report_missing_or_non_http_addresses(monkeypatch, channel,
                                                          secret_name, address):
    """An unconfigured or unsupported address fails without a network request.

    :param monkeypatch: isolates the test from any previously stored secret.
    :param channel: the notification adapter.
    :param secret_name: the adapter's webhook address secret.
    :param address: an empty address or unsupported URL scheme.
    """
    monkeypatch.setattr(run_journal, "_load_notify_secret", lambda name: "")
    results = run_journal._send_notification(
        {"title": "Done", "body": "QC", "failed": False},
        {channel: True, "secrets": {secret_name: address}})
    assert results[channel].startswith("ValueError:")
    assert "local-secret" not in results[channel]


def test_new_webhook_error_details_redact_addresses_and_tokens(monkeypatch,
                                                              caplog):
    """Failures mentioning credentials are scrubbed in results and warnings.

    :param monkeypatch: replaces HTTP with a failing local fake.
    :param caplog: captured notification warnings.
    """
    def fail(url, data, headers):
        """Reject a fake request while echoing its credentials.

        :param url: the fake receiving address.
        :param data: unused JSON payload bytes.
        :param headers: request headers containing the optional token.
        """
        raise RuntimeError(f"{url} rejected {headers.get('Authorization', '')}")

    monkeypatch.setattr(run_journal, "_notify_http_post", fail)
    credentials = {"teams_webhook": "http://127.0.0.1/teams-address-secret",
                   "webhook_url": "http://127.0.0.1/webhook-address-secret",
                   "webhook_token": "webhook-token-secret"}
    results = run_journal._send_notification(
        {"title": "Done", "body": "QC", "failed": False},
        {"teams": True, "webhook": True, "secrets": credentials})
    assert set(results) == {"teams", "webhook"}
    assert all("RuntimeError" in result for result in results.values())
    logged = caplog.text + json.dumps(results)
    for value in credentials.values():
        assert value not in logged


def test_nothing_the_notifier_does_can_fail_a_run(monkeypatch):
    def broken():
        raise RuntimeError("preferences unreadable")

    monkeypatch.setattr(run_journal, "_notification_config", broken)
    with run_journal.open_run("mask", {}) as run:
        pass
    assert run.status == "success"
    assert run_journal._notify_run_finished(run) is None


def test_a_secret_without_a_keyring_goes_to_a_mode_600_file(tmp_path,
                                                            monkeypatch):
    path = tmp_path / "dot-spacr" / "notification_secrets.json"
    monkeypatch.setattr(run_journal, "_notify_secrets_path", lambda: path)
    monkeypatch.setattr(run_journal, "_notify_keyring", lambda: None)

    assert run_journal._store_notify_secret("ntfy_token", "tk-1") == "file"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert json.loads(path.read_text()) == {"ntfy_token": "tk-1"}
    assert run_journal._load_notify_secret("ntfy_token") == "tk-1"
    assert run_journal._load_notify_secret("smtp_password") == ""
    assert run_journal._store_notify_secret("ntfy_token", "") == "forgotten"
    assert not path.exists()
    with pytest.raises(ValueError):
        run_journal._store_notify_secret("api_key", "x")


def test_a_secret_goes_to_the_os_keyring_when_there_is_one(tmp_path,
                                                           monkeypatch):
    class FakeKeyring:
        def __init__(self):
            self.store = {}

        def set_password(self, service, name, value):
            self.store[(service, name)] = value

        def get_password(self, service, name):
            return self.store.get((service, name))

        def delete_password(self, service, name):
            del self.store[(service, name)]

    ring = FakeKeyring()
    path = tmp_path / "notification_secrets.json"
    monkeypatch.setattr(run_journal, "_notify_secrets_path", lambda: path)
    monkeypatch.setattr(run_journal, "_notify_keyring", lambda: ring)

    assert run_journal._store_notify_secret("slack_webhook",
                                            "https://hooks/x") == "keyring"
    assert ring.store == {("spacr-notifications", "slack_webhook"):
                          "https://hooks/x"}
    assert not path.exists()
    assert run_journal._load_notify_secret("slack_webhook") == \
        "https://hooks/x"
    run_journal._store_notify_secret("slack_webhook", "")
    assert ring.store == {}


def test_the_message_reads_as_a_short_summary():
    assert run_journal._notify_duration(42) == "42 s"
    assert run_journal._notify_duration(125) == "2 min 05 s"
    assert run_journal._notify_duration(3720) == "1 h 02 min"
    assert run_journal._notify_scrub(
        "https://hooks.slack.com/T0/secret failed", ["T0/secret"]) == \
        "https://hooks.slack.com/*** failed"
    with pytest.raises(ValueError):
        run_journal._notify_http_post("file:///etc/passwd", b"", {})
