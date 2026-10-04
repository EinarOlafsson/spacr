"""Item 641, round 4: module screens are built ahead of time while Home idles.

* a slice runs no step while user input is recent, a button is held, a
  modal is up or a screen is being opened -- input always goes first;
* every step runs on the GUI thread from the prewarm's own timer, never
  from inside an input event;
* it can be turned off (``SPACR_PREWARM=0``, the stored preference, the
  Laptop and Extra Performance levels) and stops when memory is short;
* a prewarmed screen is opened without being built again, and a screen
  opened half built finishes its steps inside that open.
"""
from __future__ import annotations

import threading

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.delenv("SPACR_PREWARM", raising=False)
    monkeypatch.delenv("SPACR_BENCHMARK_JSON", raising=False)
    from spacr.qt.app import MainWindow

    win = MainWindow()
    win.resize(1200, 800)
    win.show()
    for _ in range(3):
        qapp.processEvents()
    yield win
    prewarm = getattr(win, "_screen_prewarm", None)
    if prewarm is not None:
        prewarm.stop()
    win.close()
    win.deleteLater()
    qapp.processEvents()


def _prewarm(window, order, monkeypatch):
    from spacr.qt import app as app_module
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "_screen_prewarm_allowed",
                        lambda: (True, ""))
    monkeypatch.setattr(app_module, "_prewarm_memory_is_short", lambda: False)
    prewarm = app_module._ScreenPrewarm(window, order)
    window._screen_prewarm = prewarm
    return prewarm


def _idle(prewarm, monkeypatch, seconds=60.0):
    from spacr.qt import app as app_module

    now = app_module._time_monotonic()
    prewarm._last_input = now - seconds


def test_no_step_runs_while_input_is_recent(window, qapp, monkeypatch):
    prewarm = _prewarm(window, ["queue"], monkeypatch)
    ran = []
    monkeypatch.setattr(window, "_run_one_prewarm_step",
                        lambda p: ran.append(p))
    prewarm._slice()
    assert ran == []
    assert prewarm.busy_reason() == "input"
    assert prewarm._timer.isActive()


def test_input_events_push_the_next_step_back(window, qapp, monkeypatch):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    prewarm = _prewarm(window, ["queue"], monkeypatch)
    _idle(prewarm, monkeypatch)
    assert prewarm.busy_reason() == ""
    ran = []

    def step(p):
        ran.append(threading.current_thread() is threading.main_thread())

    monkeypatch.setattr(window, "_run_one_prewarm_step", step)
    qapp.sendEvent(window, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A,
                                     Qt.KeyboardModifier.NoModifier, "a"))
    assert prewarm.busy_reason() == "input"
    prewarm._slice()
    assert ran == []
    _idle(prewarm, monkeypatch)
    prewarm._slice()
    assert ran == [True]


def test_no_step_runs_inside_an_input_event(window, qapp, monkeypatch):
    from PySide6.QtCore import QEvent, QObject, Qt
    from PySide6.QtGui import QKeyEvent

    prewarm = _prewarm(window, ["queue"], monkeypatch)
    inside = {"now": False}
    seen = []

    class Watch(QObject):
        def eventFilter(self, watched, event):
            if event.type() == QEvent.Type.KeyPress:
                inside["now"] = True
                _idle(prewarm, monkeypatch)
                qapp.processEvents()
                inside["now"] = False
            return False

    monkeypatch.setattr(window, "_run_one_prewarm_step",
                        lambda p: seen.append(inside["now"]))
    watch = Watch()
    window.installEventFilter(watch)
    try:
        qapp.sendEvent(window, QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_B,
            Qt.KeyboardModifier.NoModifier, "b"))
    finally:
        window.removeEventFilter(watch)
    assert True not in seen


def test_no_step_runs_while_a_screen_opens(window, qapp, monkeypatch):
    prewarm = _prewarm(window, ["queue"], monkeypatch)
    _idle(prewarm, monkeypatch)
    window._opening_a_screen = True
    try:
        assert prewarm.busy_reason() == "opening"
    finally:
        window._opening_a_screen = False
    assert prewarm.busy_reason() == ""


def test_a_prewarmed_screen_opens_without_a_second_build(
        window, qapp, monkeypatch):
    prewarm = _prewarm(window, ["queue"], monkeypatch)
    for _ in range(12):
        if prewarm.finished:
            break
        _idle(prewarm, monkeypatch)
        prewarm._slice()
    assert prewarm.built == ["queue"]
    assert prewarm.finished
    screen = window._screens["queue"]
    from PySide6.QtCore import Qt

    assert not screen.testAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    builds = []
    original = window._build_screen
    monkeypatch.setattr(window, "_build_screen",
                        lambda key: builds.append(key) or original(key))
    window._on_nav_selected("queue")
    qapp.processEvents()
    assert builds == []
    assert window._stack.currentWidget() is screen


def test_a_half_built_screen_finishes_inside_its_open(
        window, qapp, monkeypatch):
    prewarm = _prewarm(window, ["queue"], monkeypatch)
    _idle(prewarm, monkeypatch)
    prewarm._slice()
    assert "queue" not in window._screens
    assert prewarm._key == "queue"
    window._on_nav_selected("queue")
    qapp.processEvents()
    assert window._stack.currentWidget() is window._screens["queue"]
    assert prewarm.built == ["queue"]
    assert prewarm._steps is None


def test_prewarm_stops_when_memory_is_short(window, qapp, monkeypatch):
    from spacr.qt import app as app_module

    prewarm = _prewarm(window, ["queue"], monkeypatch)
    monkeypatch.setattr(app_module, "_prewarm_memory_is_short", lambda: True)
    _idle(prewarm, monkeypatch)
    prewarm._slice()
    assert prewarm.finished
    assert "queue" not in window._screens


@pytest.mark.parametrize("how", ["env", "stored", "laptop",
                                 "extra_performance"])
def test_prewarm_can_be_turned_off(window, monkeypatch, how):
    from spacr.qt import preferences

    monkeypatch.delenv("SPACR_PREWARM", raising=False)
    monkeypatch.setenv("SPACR_LAPTOP_MODE", "0")
    if how == "env":
        monkeypatch.setenv("SPACR_PREWARM", "0")
    elif how == "stored":
        store = preferences._settings()
        old = store.value(preferences._KEY_SCREEN_PREWARM)
        store.setValue(preferences._KEY_SCREEN_PREWARM, False)
        store.sync()
    else:
        monkeypatch.setattr(preferences, "get_performance_level",
                            lambda: how)
    try:
        allowed, why = preferences._screen_prewarm_allowed()
        assert not allowed and why
        assert window._start_the_screen_prewarm() is None
    finally:
        if how == "stored":
            store = preferences._settings()
            if old is None:
                store.remove(preferences._KEY_SCREEN_PREWARM)
            else:
                store.setValue(preferences._KEY_SCREEN_PREWARM, old)
            store.sync()


def test_the_order_is_configurable(monkeypatch):
    from spacr.qt import preferences

    monkeypatch.setenv("SPACR_PREWARM_ORDER", "regression, mask,mask")
    assert preferences._screen_prewarm_order() == ("regression", "mask")
    monkeypatch.setenv("SPACR_PREWARM_ORDER", "")
    assert preferences._screen_prewarm_order() == \
        preferences._SCREEN_PREWARM_ORDER
    assert preferences._SCREEN_PREWARM_ORDER[:3] == (
        "mask", "measure", "make_masks")
