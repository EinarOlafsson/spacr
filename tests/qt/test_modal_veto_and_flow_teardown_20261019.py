"""No rebuild or restart prompt under a modal; a wrapping layout survives teardown.

Follow-ups to the macOS crash reports (AppKit ``runModalCleanup`` and
use-after-free during widget teardown): an update's restart prompt and a
screen rebuild wait until no modal dialog is open; the Python ``FlowLayout``
answers nothing once its C++ half is gone, which is when a parent's teardown
still asks it for items; and a finished download's callback is bounded by
the caller's widget.
"""
from __future__ import annotations

import gc
import random

import pytest
import shiboken6
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from spacr.qt import app as app_mod
from spacr.qt.widgets.flow import FlowLayout


@pytest.fixture
def modal(monkeypatch):
    state = {"widget": None}
    monkeypatch.setattr(QApplication, "activeModalWidget",
                        staticmethod(lambda: state["widget"]))
    return state


class _Window(QWidget):
    _after_the_modal_closes = app_mod.MainWindow._after_the_modal_closes
    _run_when_no_modal = app_mod.MainWindow._run_when_no_modal
    rebuild_app_screen = app_mod.MainWindow.rebuild_app_screen
    _poll_frozen_update = app_mod.MainWindow._poll_frozen_update


def test_a_deferred_call_waits_for_the_modal_to_close(qtbot, modal):
    window = _Window()
    qtbot.addWidget(window)
    modal["widget"] = QWidget()
    calls = []
    window.note = lambda *args: calls.append(args)
    window._after_the_modal_closes("note", 1, 2)
    window._after_the_modal_closes("note", 3)
    qtbot.wait(500)
    assert calls == []
    modal["widget"] = None
    qtbot.waitUntil(lambda: calls == [(1, 2), (3,)], timeout=3000)
    qtbot.wait(450)
    assert calls == [(1, 2), (3,)]
    assert not window._modal_watch_timer.isActive()


def test_a_deferred_call_is_dropped_with_its_window(qtbot, modal):
    window = _Window()
    modal["widget"] = QWidget()
    calls = []
    window.note = lambda *args: calls.append(args)
    window._after_the_modal_closes("note", 1)
    shiboken6.delete(window)
    modal["widget"] = None
    qtbot.wait(600)
    assert calls == []


def test_a_screen_rebuild_under_a_modal_is_deferred(qtbot, modal):
    window = _Window()
    qtbot.addWidget(window)
    deferred = []
    window._after_the_modal_closes = lambda *call: deferred.append(call)
    modal["widget"] = QWidget()
    window.rebuild_app_screen("measure", {"a": 1})
    assert deferred == [("rebuild_app_screen", "measure", {"a": 1})]
    assert not hasattr(window, "_screens")


def test_the_restart_prompt_waits_for_the_modal(qtbot, modal):
    window = _Window()
    qtbot.addWidget(window)
    asked = []

    class _Ready:
        def status(self):
            return {"state": "ready"}

    window._frozen_update_handshake = _Ready()
    window._frozen_update_plan = {"version": "9.9"}
    window._closing = False
    window._frozen_update_timer = QTimer(window)
    window._frozen_update_timer.start(10_000)
    window._hide_update_progress = lambda: None
    window._confirm_update_restart = lambda message: asked.append(message)
    window._cancel_frozen_update = lambda: asked.append("cancelled")
    modal["widget"] = QWidget()
    window._poll_frozen_update()
    assert asked == []
    assert window._frozen_update_timer.isActive()
    modal["widget"] = None
    window._poll_frozen_update()
    assert asked and asked[0] != "cancelled"
    assert not window._frozen_update_timer.isActive()


def test_a_destroyed_flow_layout_answers_nothing(qapp):
    host = QWidget()
    layout = FlowLayout(host)
    for index in range(5):
        layout.addWidget(QLabel(f"chip {index}", host))
    assert layout.count() == 5
    shiboken6.delete(host)
    assert not shiboken6.isValid(layout)
    assert layout.count() == 0
    assert layout.itemAt(0) is None
    assert layout.takeAt(0) is None
    assert layout.minimumSize().isEmpty()


def test_chips_come_and_go_while_their_hosts_are_torn_down(qapp):
    rng = random.Random(7)
    for round_ in range(200):
        host = QWidget()
        host.resize(300, 200)
        layout = FlowLayout(host)
        chips = [QLabel(f"chip {round_}-{i}", host) for i in range(30)]
        for chip in chips:
            layout.addWidget(chip)
        host.show()
        for chip in rng.sample(chips, 10):
            layout.removeWidget(chip)
            chip.setParent(None)
            chip.deleteLater()
        for i in range(5):
            layout.addWidget(QLabel(f"late {i}", host))
        layout.activate()
        assert layout.count() == 25
        if round_ % 3 == 0:
            host.deleteLater()
        else:
            shiboken6.delete(host)
        if round_ % 5 == 0:
            gc.collect()
        qapp.processEvents()
    for _ in range(5):
        qapp.processEvents()


def test_a_download_callback_is_bounded_by_its_caller(qapp):
    from spacr.qt.hf_download import _callback_owner

    caller = QWidget()
    assert _callback_owner(caller) is caller
    shiboken6.delete(caller)
    assert _callback_owner(caller) is QApplication.instance()
    assert _callback_owner(None) is QApplication.instance()
