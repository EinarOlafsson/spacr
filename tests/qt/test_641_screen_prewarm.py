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

    win = MainWindow(initial_app="__home__")
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


def test_real_chained_construction_keeps_its_job_for_an_open_host(
        window, qapp, qtbot, monkeypatch):
    """The shutdown guard leaves ordinary owned construction unchanged."""
    from shiboken6 import isValid
    from spacr.qt import chaining as qt_chaining
    from spacr.qt.bridge import registry
    from spacr.qt.job_runner import shutdown_all

    runs = registry()
    registered = []
    original_register = runs.register

    def register(handle):
        registered.append(handle.app_key)
        return original_register(handle)

    monkeypatch.setattr(runs, "register", register)
    screen = None
    try:
        assert not window._closing
        screen = qt_chaining._chained_app_screen("classify_merged", window)
        bar = qt_chaining.chaining_bar(screen)
        assert bar is not None
        assert "classify_merged chaining" in registered
        qtbot.waitUntil(lambda: not bar._resolver.is_busy(), timeout=5000)
    finally:
        shutdown_all()
        qtbot.waitUntil(lambda: not runs.active(), timeout=5000)
        if screen is not None and isValid(screen):
            assert screen.close()
            screen.deleteLater()
        qapp.processEvents()


def test_close_during_real_constructor_starts_no_late_chaining_job(
        window, qapp, qtbot, monkeypatch):
    """An accepted re-entrant close must stop unpublished construction work."""
    _close_during_real_constructor(window, qapp, qtbot, monkeypatch, "classify_merged")


def test_close_during_real_measure_constructor_registers_no_late_work(
        window, qapp, qtbot, monkeypatch):
    """Capture the real Measure factory's registrations after accepted close."""
    _close_during_real_constructor(window, qapp, qtbot, monkeypatch, "measure")


def _close_during_real_constructor(window, qapp, qtbot, monkeypatch, key):
    """Observe the same real constructor/prewarm shutdown route for one key."""
    import traceback
    from shiboken6 import getCppPointer, isValid
    from PySide6.QtCore import QTimer
    from spacr.qt import app as app_module
    from spacr.qt import chaining as qt_chaining
    from spacr.qt import prerun
    from spacr.qt.bridge import registry
    from spacr.qt.job_runner import shutdown_all

    previous_factories = dict(app_module.APP_FACTORIES)
    previous_inner = dict(prerun._INNER)
    qt_chaining.register()
    if key == "measure":
        prerun.register()
    original_factory = app_module.APP_FACTORIES[key]
    runs = registry()
    registrations = []
    close_record = {}
    built = []
    original_register = runs.register

    def register(handle):
        registrations.append({
            "name": handle.app_key,
            "handle_id": id(handle), "handle_cpp": getCppPointer(handle),
            "worker_type": type(handle.worker).__name__,
            "thread_id": id(handle.thread), "kind": handle.kind,
            "owner_id": id(handle.parent()),
            "after_accepted_close": close_record.get("accepted", False),
            "stack": "".join(traceback.format_stack()),
        })
        return original_register(handle)

    def close_window():
        close_record["before"] = [(id(h), h.app_key) for h in runs.active()]
        close_record["stack"] = "".join(traceback.format_stack())
        close_record["accepted"] = window.close()
        close_record["after"] = [(id(h), h.app_key) for h in runs.active()]

    close_timer = QTimer(window)
    close_timer.setSingleShot(True)
    close_timer.timeout.connect(close_window)

    def factory(*args, **kwargs):
        close_timer.start(0)
        screen = original_factory(*args, **kwargs)
        built.append(screen)
        return screen

    monkeypatch.setattr(runs, "register", register)
    monkeypatch.setitem(app_module.APP_FACTORIES, key, factory)
    prewarm = _prewarm(window, [key], monkeypatch)
    _idle(prewarm, monkeypatch)
    try:
        prewarm._slice()
        evidence = {"close": close_record, "registrations": registrations,
                    "prewarm_finished": prewarm.finished,
                    "published": key in window._screens,
                    "constructed": len(built)}
        print("Re-entrant real prewarm evidence:", evidence)
        assert close_record.get("accepted") is True, evidence
        assert "_prewarm_steps" in close_record["stack"], evidence
        assert "settings_model.py" in close_record["stack"], evidence
        assert "QCoreApplication.processEvents" in close_record["stack"], evidence
        assert prewarm.finished, evidence
        assert key not in window._screens, evidence
        assert not [r for r in registrations if r["after_accepted_close"]], evidence
    finally:
        app_module.APP_FACTORIES.clear()
        app_module.APP_FACTORIES.update(previous_factories)
        prerun._INNER.clear()
        prerun._INNER.update(previous_inner)
        prewarm.stop()
        close_timer.stop()
        shutdown_all()
        qtbot.waitUntil(lambda: not runs.active(), timeout=5000)
        for screen in built:
            if isValid(screen):
                screen.close()
                screen.deleteLater()
        qapp.processEvents()


def test_real_measure_factory_keeps_ordinary_open_host_construction(
        window, qapp, qtbot, monkeypatch):
    """The real registered Measure factory still installs its ordinary UI."""
    from shiboken6 import isValid
    from spacr.qt import app as app_module, chaining, prerun
    from spacr.qt.bridge import registry
    from spacr.qt.job_runner import shutdown_all

    previous_factories, previous_inner = dict(app_module.APP_FACTORIES), dict(prerun._INNER)
    runs, registered = registry(), []
    original_register = runs.register

    def register(handle):
        registered.append(handle.app_key)
        return original_register(handle)

    monkeypatch.setattr(runs, "register", register)
    screen = None
    try:
        chaining.register()
        prerun.register()
        assert not window._closing
        screen = window._build_screen("measure")
        assert screen.app_key == "measure"
        bar = chaining.chaining_bar(screen)
        assert bar is not None and "measure chaining" in registered
        assert prerun.qc_banner(screen) is not None
        qtbot.waitUntil(lambda: not bar._resolver.is_busy(), timeout=5000)
    finally:
        app_module.APP_FACTORIES.clear()
        app_module.APP_FACTORIES.update(previous_factories)
        prerun._INNER.clear()
        prerun._INNER.update(previous_inner)
        shutdown_all()
        qtbot.waitUntil(lambda: not registry().active(), timeout=5000)
        if screen is not None and isValid(screen):
            assert screen.close()
            screen.deleteLater()
        qapp.processEvents()


def test_queued_measure_qc_starts_no_job_after_accepted_host_close(
        window, qapp, qtbot, tmp_path, monkeypatch):
    """Normal source debounce cannot register work after its host closed."""
    import traceback
    from shiboken6 import getCppPointer
    from PySide6.QtWidgets import QLineEdit
    from spacr.qt import app as app_module, chaining, prerun
    from spacr.qt.bridge import registry
    from spacr.qt.job_runner import shutdown_all

    previous_factories, previous_inner = dict(app_module.APP_FACTORIES), dict(prerun._INNER)
    runs, registrations, close_record = registry(), [], {}
    original_register = runs.register

    def register(handle):
        registrations.append({
            "name": handle.app_key, "handle_id": id(handle),
            "handle_cpp": getCppPointer(handle), "kind": handle.kind,
            "thread_id": id(handle.thread),
            "after_accepted_close": close_record.get("accepted", False),
            "stack": "".join(traceback.format_stack()),
        })
        return original_register(handle)

    monkeypatch.setattr(runs, "register", register)
    try:
        chaining.register()
        prerun.register()
        window._on_nav_selected("measure")
        screen = window._screens["measure"]
        assert window._stack.currentWidget() is screen
        banner = prerun.qc_banner(screen)
        assert banner is not None
        qtbot.waitUntil(lambda: not runs.active(), timeout=5000)
        field = screen._settings_model._widgets["src"]
        assert isinstance(field, QLineEdit)
        field.setText(str(tmp_path))
        assert prerun._has_src(prerun._src_of(screen))
        assert banner._timer.isActive() and banner._timer.remainingTime() > 0
        close_record["pending_ms"] = banner._timer.remainingTime()
        close_record["registration_index"] = len(registrations)
        close_record["before"] = [(id(h), h.app_key) for h in runs.active()]
        close_record["accepted"] = window.close()
        close_record["after"] = [(id(h), h.app_key) for h in runs.active()]
        assert close_record["accepted"] is True
        assert not [r for r in registrations[close_record["registration_index"]:]
                    if r["name"] == "measure"], registrations
        qtbot.wait(prerun.REFRESH_DELAY_MS + 100)
        evidence = {"close": close_record, "registrations": registrations,
                    "active": [(id(h), h.app_key) for h in runs.active()]}
        print("Queued Measure close evidence:", evidence)
        assert not [r for r in registrations if r["after_accepted_close"]], evidence
    finally:
        app_module.APP_FACTORIES.clear()
        app_module.APP_FACTORIES.update(previous_factories)
        prerun._INNER.clear()
        prerun._INNER.update(previous_inner)
        shutdown_all()
        qtbot.waitUntil(lambda: not runs.active(), timeout=5000)
        qapp.processEvents()


def test_no_step_runs_while_input_is_recent(window, qapp, monkeypatch):
    prewarm = _prewarm(window, ["queue"], monkeypatch)
    ran = []
    monkeypatch.setattr(window, "_run_one_prewarm_step",
                        lambda p: ran.append(p))
    prewarm._slice()
    assert ran == []
    assert prewarm.busy_reason() == "input"
    assert prewarm._timer.isActive()


def test_measure_source_debounce_survives_navigation_hide_and_show(
        window, qapp, qtbot, tmp_path, monkeypatch):
    """An open host keeps normal QC and chaining source-refresh behavior."""
    from spacr.qt import app as app_module, chaining, prerun
    from spacr.qt.bridge import registry
    from spacr.qt.job_runner import shutdown_all

    previous_factories, previous_inner = dict(app_module.APP_FACTORIES), dict(prerun._INNER)
    runs, registered = registry(), []
    original_register = runs.register

    def register(handle):
        registered.append(handle.app_key)
        return original_register(handle)

    monkeypatch.setattr(runs, "register", register)
    try:
        chaining.register()
        prerun.register()
        window._on_nav_selected("measure")
        screen = window._screens["measure"]
        banner, bar = prerun.qc_banner(screen), chaining.chaining_bar(screen)
        assert banner is not None and bar is not None
        qtbot.waitUntil(lambda: not runs.active(), timeout=5000)
        field = screen._settings_model._widgets["src"]
        for shown in (False, True):
            window._on_nav_selected("measure" if shown else "__home__")
            assert screen.isVisible() is shown
            assert not window._closing
            source = tmp_path / ("shown" if shown else "hidden")
            source.mkdir()
            before = {key: registered.count(key) for key in ("measure", "measure chaining")}
            field.setText(str(source))
            field.textEdited.emit(field.text())
            assert banner._timer.isActive() and bar._timer.isActive()
            qtbot.waitUntil(lambda: all(registered.count(key) > count
                for key, count in before.items()), timeout=5000)
            qtbot.waitUntil(lambda: not runs.active(), timeout=5000)
            assert not banner._host_closed() and not bar._host_closed()
    finally:
        app_module.APP_FACTORIES.clear()
        app_module.APP_FACTORIES.update(previous_factories)
        prerun._INNER.clear()
        prerun._INNER.update(previous_inner)
        shutdown_all()
        qtbot.waitUntil(lambda: not runs.active(), timeout=5000)
        qapp.processEvents()


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


def test_prewarm_pauses_off_home_and_resumes_the_same_work(
        window, qapp, monkeypatch):
    """An idle module does not consume new or partially queued Home work."""
    window._on_nav_selected("queue")
    module = window._screens["queue"]
    assert window._stack.currentWidget() is module
    prewarm = _prewarm(window, ["mask", "measure"], monkeypatch)
    calls = []
    monkeypatch.setattr(window, "_run_one_prewarm_step",
                        lambda p: calls.append((p._key, p._steps)))
    _idle(prewarm, monkeypatch)
    prewarm._slice()
    assert calls == []
    assert prewarm._order == ["mask", "measure"]
    assert prewarm._steps is None
    assert prewarm._timer.isActive()
    assert not prewarm.finished

    window._on_nav_selected("__home__")
    assert window._stack.currentWidget() is window._startup
    _idle(prewarm, monkeypatch)
    prewarm._slice()
    assert len(calls) == 1
    key, steps = calls[0]
    assert key == "mask" and steps is not None
    assert prewarm._order == ["measure"]

    window._on_nav_selected("queue")
    _idle(prewarm, monkeypatch)
    prewarm._slice()
    assert calls == [(key, steps)]
    assert prewarm._key == key and prewarm._steps is steps
    assert prewarm._order == ["measure"]
    assert prewarm._timer.isActive()
    assert not prewarm.finished

    window._on_nav_selected("__home__")
    _idle(prewarm, monkeypatch)
    prewarm._slice()
    assert calls == [(key, steps), (key, steps)]


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
