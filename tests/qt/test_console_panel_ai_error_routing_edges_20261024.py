"""Edge paths of the console's error routing: relay delivery, retirement and refusals."""
import weakref
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QPlainTextEdit, QWidget, QVBoxLayout
from shiboken6 import delete as shiboken_delete

from spacr import crashreport as cr
from spacr.qt.ai import providers
from spacr.qt.widgets import console_panel as console

pytestmark = pytest.mark.qt


def _text(panel):
    return "\n".join(b.toPlainText() for b in panel.findChildren(console._StdoutBlock))


@pytest.fixture
def owner_panel(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(cr, "_reports_dir", lambda: tmp_path)
    monkeypatch.setattr(cr, "_ERROR_POLICY_CACHE", {})
    owner = QWidget()
    layout = QVBoxLayout(owner)
    panel = console.ConsolePanel(parent=owner, follow_log=False)
    layout.addWidget(panel)
    qtbot.addWidget(owner)
    owner.resize(600, 400)
    owner.show()
    yield owner, panel
    box = getattr(panel, "_error_storm_box", None)
    if box is not None:
        box.reject()
    panel.shutdown()


@pytest.fixture
def panel(owner_panel):
    return owner_panel[1]


def test_queueing_into_a_deleted_relay_reports_failure(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    relay = console._ErrorReportRelay(host)
    assert relay._queue({"event_id": "a"}) is True
    shiboken_delete(relay)
    assert relay._queue({"event_id": "b"}) is False


def test_storm_with_no_visible_console_falls_back_to_the_terminal_writer(
        qtbot, monkeypatch, tmp_path):
    hidden_owner = QWidget()
    qtbot.addWidget(hidden_owner)
    monkeypatch.setattr(cr, "_reports_dir", lambda: tmp_path)
    hidden = console.ConsolePanel(parent=hidden_owner, follow_log=False, chat=False)
    written = []
    monkeypatch.setattr(cr, "_write_error_result", written.append)
    monkeypatch.setattr(hidden, "_show_error_storm_prompt",
                        lambda: pytest.fail("a console in a hidden window got the prompt"))
    relay = console._ErrorReportRelay(hidden_owner)
    relay.consoles = [lambda: None, weakref.ref(hidden)]
    outcome = {"storm": True, "event_id": "s1"}
    relay._receive(outcome)
    assert written == [outcome]
    hidden.shutdown()


class _FlakyRef:
    """Answers with its widget for the liveness sweep, then reports it gone."""

    def __init__(self, widget):
        self.widget = widget
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.widget if self.calls == 1 else None


def test_relay_retires_the_oldest_pending_owner_when_the_queue_overflows(
        qtbot, monkeypatch, owner_panel):
    owner, shown = owner_panel
    other_owner = QWidget()
    qtbot.addWidget(other_owner)
    other = console.ConsolePanel(parent=other_owner, follow_log=False, chat=False)
    discarded = []
    monkeypatch.setattr(other, "_discard_error_report_entry", discarded.append)
    delivered = []
    monkeypatch.setattr(shown, "_show_error_report_result", delivered.append)
    relay = console._ErrorReportRelay(owner)
    relay.consoles = [weakref.ref(shown)]
    flaky = _FlakyRef(other)
    relay.pending["flaky"] = flaky
    for index in range(cr.QUEUE_LIMIT - 1):
        relay.pending["k%d" % index] = weakref.ref(other)
    assert len(relay.pending) == cr.QUEUE_LIMIT

    relay._receive({"event_id": "new1", "pending": True})
    assert "flaky" not in relay.pending
    assert flaky.calls == 2
    assert discarded == []
    assert len(relay.pending) == cr.QUEUE_LIMIT
    assert relay.pending["new1"]() is shown

    relay._receive({"event_id": "new2", "pending": True})
    assert discarded == ["k0"]
    assert "k0" not in relay.pending
    assert len(relay.pending) == cr.QUEUE_LIMIT
    assert [o["event_id"] for o in delivered] == ["new1", "new2"]
    other.shutdown()


@pytest.mark.usefixtures("console_info_enabled")
def test_stdout_is_still_shown_when_error_capture_raises(panel, monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("capture broke")

    monkeypatch.setattr(cr, "_capture_error_line", boom)
    panel.append_stdout("plain pipeline line\n")
    assert "plain pipeline line" in _text(panel)


@pytest.mark.usefixtures("console_info_enabled")
def test_stdout_already_captured_at_its_source_is_not_captured_twice(panel, monkeypatch):
    seen = []
    monkeypatch.setattr(cr, "_capture_error_line", lambda *a, **k: seen.append(a))
    monkeypatch.setattr(panel, "sender", lambda: SimpleNamespace(_spacr_output_captured=True))
    panel.append_stdout("already captured line\n")
    assert seen == []
    assert "already captured line" in _text(panel)
    monkeypatch.setattr(panel, "sender", lambda: SimpleNamespace(_spacr_output_captured=False))
    panel.append_stdout("fresh line\n")
    assert [a[0] for a in seen] == ["fresh line\n"]


def test_error_text_is_still_shown_when_error_dispatch_raises(panel, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("dispatch broke")

    monkeypatch.setattr(cr, "_dispatch_error", boom)
    panel.append_error("Traceback (most recent call last):\nValueError: shown anyway\n")
    assert "ValueError: shown anyway" in _text(panel)


def test_discarding_an_empty_entry_removes_it_and_a_written_one_stays(panel):
    empty = QPlainTextEdit(panel)
    written = QPlainTextEdit(panel)
    written.setPlainText("an explanation")
    panel._current_topic_label = object()
    panel._pending_report_entries["empty"] = (empty, False)
    panel._pending_report_entries["written"] = (written, False)
    panel._discard_error_report_entry("empty")
    assert "empty" not in panel._pending_report_entries
    assert panel._current_topic_label is None
    assert not empty.isVisible()
    panel._current_topic_label = "kept"
    panel._discard_error_report_entry("written")
    assert "written" not in panel._pending_report_entries
    assert panel._current_topic_label == "kept"
    assert written.toPlainText() == "an explanation"
    panel._discard_error_report_entry("never-registered")


def test_discarding_an_entry_whose_widget_was_deleted_does_not_raise(panel):
    class Gone:
        def toPlainText(self):
            raise RuntimeError("Internal C++ object already deleted.")

    panel._pending_report_entries["gone"] = (Gone(), False)
    panel._discard_error_report_entry("gone")
    assert "gone" not in panel._pending_report_entries


def test_unknown_storm_action_does_nothing(panel, monkeypatch):
    from spacr.qt import shutdown
    monkeypatch.setattr(shutdown, "force_quit_now", lambda: pytest.fail("quit"))
    monkeypatch.setattr(shutdown, "restart_spacr", lambda *a: pytest.fail("restart"))
    notices = []
    monkeypatch.setattr(panel, "append_notice", notices.append)
    panel._force_error_storm("cancel")
    assert notices == []


def test_successful_restart_through_the_window_gives_no_failure_notice(
        qtbot, monkeypatch, tmp_path):
    class Window(QWidget):
        restarts = 0

        def force_restart(self):
            Window.restarts += 1
            return True

    monkeypatch.setattr(cr, "_reports_dir", lambda: tmp_path)
    window = Window()
    qtbot.addWidget(window)
    panel = console.ConsolePanel(parent=window, follow_log=False, chat=False)
    notices = []
    monkeypatch.setattr(panel, "append_notice", notices.append)
    panel._force_error_storm("restart")
    assert Window.restarts == 1
    assert notices == []
    panel.shutdown()


def test_ai_refusal_outcome_shows_the_refusal_sentence(panel):
    refusal = {"reason": "quota", "scope": "install", "resets_at": None}
    panel._show_error_report_result({"ai_refusal": refusal, "quiet": True, "event_id": "r"})
    expected = providers.hosted_refusal_message("quota", "install", None)
    assert expected.strip().splitlines()[0] in _text(panel)


@pytest.mark.parametrize("policy,eligible", (({"provider": "hosted"}, False),
                                             ({"provider": "none"}, True)))
def test_pending_outcome_without_an_eligible_provider_opens_no_entry(
        panel, monkeypatch, policy, eligible):
    monkeypatch.setattr(cr, "_automatic_error_policy", lambda kind: (policy, eligible, None))
    panel._show_error_report_result({"pending": True, "event_id": "p1", "kind": "x"})
    assert panel._pending_report_entries == {}
    assert "asking spaCR AI" not in _text(panel)


def test_pending_outcome_for_a_known_event_does_not_open_a_second_entry(panel, monkeypatch):
    monkeypatch.setattr(cr, "_automatic_error_policy",
                        lambda kind: ({"provider": "hosted"}, True, None))
    panel._show_error_report_result({"pending": True, "event_id": "p2"})
    first = panel._pending_report_entries["p2"]
    blocks = len(panel.findChildren(console._StdoutBlock))
    panel._show_error_report_result({"pending": True, "event_id": "p2"})
    assert panel._pending_report_entries["p2"] is first
    assert len(panel.findChildren(console._StdoutBlock)) == blocks


def test_pending_entries_beyond_the_queue_limit_retire_the_oldest(panel, monkeypatch):
    monkeypatch.setattr(cr, "_automatic_error_policy",
                        lambda kind: ({"provider": "hosted"}, True, None))
    for index in range(cr.QUEUE_LIMIT):
        block = QPlainTextEdit(panel)
        block.setPlainText("text %d" % index)
        panel._pending_report_entries["old%d" % index] = (block, True)
    panel._show_error_report_result({"pending": True, "event_id": "latest"})
    keys = list(panel._pending_report_entries)
    assert len(keys) == cr.QUEUE_LIMIT
    assert "old0" not in keys
    assert keys[-1] == "latest"


def test_unexplained_final_outcome_retires_its_waiting_entry(panel, monkeypatch):
    monkeypatch.setattr(cr, "_automatic_error_policy",
                        lambda kind: ({"provider": "hosted"}, True, None))
    panel._show_error_report_result({"pending": True, "event_id": "u1"})
    assert "u1" in panel._pending_report_entries
    panel._show_error_report_result({"event_id": "u1", "explained": False})
    assert "u1" not in panel._pending_report_entries


def test_hosted_provider_stream_gets_no_tool_approval_note(panel, monkeypatch):
    captured = {}

    class Signal:
        def connect(self, slot):
            pass

    class Thread:
        def start(self):
            captured["started"] = captured.get("started", 0) + 1

        def isRunning(self):
            return False

        def __getattr__(self, name):
            return lambda *args, **kwargs: True

    thread = Thread()
    worker = SimpleNamespace(stage_changed=Signal(), chunk_ready=Signal(), finished=Signal())

    def fake_make(provider, messages, system, parent):
        captured["system"] = system
        return thread, worker

    monkeypatch.setattr(console, "make_stream_thread", fake_make)
    monkeypatch.setattr(panel, "_current_provider", lambda: object())
    panel._current_provider_name = providers.HOSTED_PROVIDER_NAME
    panel._start_stream("base prompt")
    assert captured == {"system": "base prompt", "started": 1}
    panel._current_provider_name = "some-other-provider"
    panel._start_stream("base prompt")
    assert captured["system"].startswith("base prompt\n\n")
    assert len(captured["system"]) > len("base prompt\n\n")
