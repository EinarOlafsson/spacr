"""Late and failed GUI completions leave ownership and user state intact."""
from __future__ import annotations

from types import SimpleNamespace
import threading
import weakref

import shiboken6
from PySide6.QtCore import QObject, QEvent, QRect, QTimer, Signal
from PySide6.QtWidgets import QMainWindow, QStackedWidget, QWidget


def test_a_consumed_timer_callback_cannot_run_twice(qapp):
    from spacr.qt import _fire_after

    timer = QTimer()
    calls = []
    box = [lambda: calls.append("once")]
    _fire_after(timer, box)
    _fire_after(timer, box)
    assert calls == ["once"] and box == []


def test_an_unidentified_worker_completion_cannot_settle_a_job(qapp):
    from spacr.qt.job_runner import JobRunner

    class UnidentifiedWorker(QObject):
        finished = Signal(bool)

    runner = JobRunner()
    worker = UnidentifiedWorker()
    results = []
    runner.job_finished.connect(results.append)
    worker.finished.connect(runner._worker_finished)
    worker.finished.emit(True)
    assert results == []
    assert runner.pending_jobs() == 0 and not runner.is_busy()


def test_a_running_job_is_not_retired_by_an_early_sweep(qapp, qtbot):
    from spacr.qt.job_runner import JobRunner

    entered = threading.Event()
    release = threading.Event()
    runner = JobRunner()

    def work():
        entered.set()
        release.wait(5)
        return "done"

    results = []
    assert runner.submit(work, results.append)
    try:
        assert entered.wait(5)
        held = dict(runner._jobs)
        runner._retire_finished_jobs()
        assert runner._jobs == held
        assert runner.pending_jobs() == 1 and runner.is_busy()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not runner._jobs, timeout=10000)
    assert results == ["done"]
    assert runner.pending_jobs() == 0 and not runner.is_busy()


def test_deleted_walkthroughs_are_discarded_without_touching_a_dead_stack(qapp):
    from spacr.qt import walkthrough

    window = QMainWindow()
    overlay = QWidget(window)
    overlay._walkthrough_app_key = "test"
    window._module_walkthroughs = [overlay]
    shiboken6.delete(overlay)
    walkthrough._dismiss_module_walkthroughs(window)
    assert window._module_walkthroughs == []












def test_invalid_event_targets_are_rejected_without_registering_watches(qapp):
    from spacr.qt import gil_priority as gil

    hub = SimpleNamespace(_objects={})
    invalid = object()
    assert not gil._hub_discard_object(hub, invalid, invalid)
    assert gil._hub_object_watchers(hub, invalid) == ()
    assert not gil._hub_ask_object_watchers(hub, invalid, QEvent(QEvent.User), QEvent.User)
    assert not gil._watch_object_events(None, invalid, ())
    assert not gil._stop_watching_object_events(None, invalid)
    assert hub._objects == {}


def test_geometry_replay_skips_deleted_widgets_and_unrequested_event_types(qapp):
    from spacr.qt import gil_priority as gil

    widget = QWidget()
    dead = QWidget()
    shiboken6.delete(dead)
    vanished = QWidget()
    vanished_ref = weakref.ref(vanished)
    del vanished
    hub = SimpleNamespace(_geometry_watched={1: weakref.ref(widget), 2: weakref.ref(dead),
                                            3: vanished_ref}, _object_kinds=frozenset())
    assert [entry[0] for entry in gil._hub_geometry_before(hub)] == [widget]
    previous = widget.geometry()
    widget.setGeometry(QRect(previous.x() + 1, previous.y() + 1,
                             previous.width() + 10, previous.height() + 10))
    gil._hub_geometry_after(hub, [(dead, False, previous), (widget, widget.isHidden(), previous)])
    assert widget.geometry() != previous


def test_a_failing_object_watcher_reports_its_error_and_leaves_the_event_unclaimed(
        qapp, monkeypatch):
    from spacr.qt import gil_priority as gil

    errors = []
    monkeypatch.setattr(gil.sys, "excepthook", lambda *info: errors.append(info))

    class FailingWatcher(QObject):
        def eventFilter(self, watched, event):
            raise RuntimeError("watcher stopped")

    widget, watcher = QWidget(), FailingWatcher()
    hub = gil._application_event_hub_class()(qapp)
    kind = QEvent.Type.User
    gil._hub_add_object(hub, widget, watcher, {kind})
    assert not gil._hub_ask_object_watchers(hub, widget, QEvent(kind), kind)
    assert len(errors) == 1 and str(errors[0][1]) == "watcher stopped"
    gil._hub_discard_object(hub, widget, watcher)
    hub.deleteLater()


def test_an_update_watcher_can_consume_a_paint_request_before_the_paint(qapp):
    from spacr.qt import gil_priority as gil

    calls = []

    class ConsumeUpdate(QObject):
        def eventFilter(self, watched, event):
            calls.append(event.type())
            return True

    widget, watcher = QWidget(), ConsumeUpdate()
    hub = gil._application_event_hub_class()(qapp)
    kind = QEvent.Type.UpdateRequest
    gil._hub_add_object(hub, widget, watcher, {kind})
    assert hub.eventFilter(widget, QEvent(kind))
    assert calls == [kind]
    gil._hub_discard_object(hub, widget, watcher)
    hub.deleteLater()


def test_removing_a_watch_tolerates_both_a_dead_hub_and_dead_widget(qapp, monkeypatch):
    from spacr.qt import gil_priority as gil

    widget, watcher = QWidget(), QObject()
    shiboken6.delete(widget)
    monkeypatch.setattr(gil, "_application_event_hub", lambda *_args, **_kwargs: object())

    def dead_hub(*_args):
        raise RuntimeError("hub was destroyed")

    monkeypatch.setattr(gil, "_hub_discard_object", dead_hub)
    assert not gil._stop_watching_object_events(widget, watcher)


def test_walkthrough_navigation_survives_a_failed_overlay_cleanup(qapp, monkeypatch):
    from spacr.qt import walkthrough

    window = QMainWindow()
    stack = QStackedWidget(window)
    screen = QWidget()
    screen.app_key = "completion-guard-test"
    stack.addWidget(screen)
    window._stack = stack
    handler = walkthrough._WalkthroughHandler(window)

    def already_closed(*_args, **_kwargs):
        raise RuntimeError("overlay was destroyed")

    monkeypatch.setattr(walkthrough, "_dismiss_module_walkthroughs", already_closed)
    monkeypatch.setattr(walkthrough, "was_seen", lambda _key: False)
    handler.on_current_changed(0)
    assert stack.currentWidget() is screen


def test_a_disconnected_destroyed_emitter_is_safe_to_disconnect_again(qapp):
    from spacr.qt.screens.data_manager import _disconnect_quietly

    class Emitter(QObject):
        changed = Signal()

    emitter = Emitter()
    slot = lambda: None
    emitter.changed.connect(slot)
    signal = emitter.changed
    shiboken6.delete(emitter)
    assert _disconnect_quietly(signal, slot) is None


def test_api_links_reject_an_accidental_non_spacr_target(monkeypatch):
    from spacr.qt.screens import settings_model
    from spacr.qt.screens import setting_api_targets

    monkeypatch.setitem(setting_api_targets.SETTING_API_TARGETS, "guard_key",
                        ("unrelated.module", "function", True))
    assert settings_model._mapped_api_target("guard_key") == ("", "")
















