"""The GUI-thread collector never sweeps mid-event, and screens leave no cycles.

Lesson 81 crashed (SIGSEGV, GUI thread "Garbage-collecting" in
``gc_policy.collect_once``) while the recorder pressed a module screen's
splitter handle after closing an Actions fold: Qt objects reachable only
from reference cycles were destroyed inside the collector. Two defences are
pinned here. The timer's sweep waits while a pointer button is held, a
counted pause is open, or a nested event loop runs (bounded, so memory
cannot grow without limit). And the self-referencing closures that kept a
screen's widgets or panels in cycles are gone.
"""
from __future__ import annotations

import gc

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QPoint, Qt, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from spacr.qt import gc_policy  # noqa: E402


@pytest.fixture
def collections(monkeypatch):
    """Record each sweep the gate lets through, with the installed timer held.

    The policy's own timer would otherwise tick during pytest-qt's event
    processing and add sweeps the test did not ask for.
    """
    calls = []
    timer = gc_policy._timer
    running = timer is not None and timer.isActive()
    if running:
        timer.stop()
    monkeypatch.setattr(gc_policy, "collect_once", lambda: calls.append(1) or 0)
    gc_policy._mid_event_since[0] = None
    yield calls
    gc_policy._mid_event_since[0] = None
    if running:
        timer.start()


def test_a_tick_sweeps_when_nothing_is_mid_way(qapp, collections):
    assert gc_policy._collect_on_tick() == 0
    assert collections == [1]


def test_a_tick_waits_while_a_pointer_button_is_held(qapp, collections,
                                                     monkeypatch):
    monkeypatch.setattr(QApplication, "mouseButtons",
                        staticmethod(lambda: Qt.LeftButton))
    gc_policy._mid_event_since[0] = 0.0
    assert gc_policy._collect_on_tick() == -1
    assert collections == []


def test_a_tick_waits_while_a_counted_pause_is_open(qapp, collections):
    with gc_policy._automatic_collection_paused():
        assert gc_policy._collect_on_tick() == -1
    assert collections == []
    assert gc_policy._collect_on_tick() == 0


def test_a_tick_inside_a_nested_loop_waits_then_runs_after_the_limit(
        qapp, collections, monkeypatch):
    results = []
    outer, inner = QEventLoop(), QEventLoop()

    def inside():
        results.append(gc_policy._collect_on_tick())
        monkeypatch.setattr(gc_policy, "_MID_EVENT_LIMIT_SECONDS", 0.0)
        results.append(gc_policy._collect_on_tick())
        inner.quit()
        outer.quit()

    def nest():
        QTimer.singleShot(0, inside)
        inner.exec()

    QTimer.singleShot(0, nest)
    outer.exec()
    assert results == [-1, 0]
    assert collections == [1]


def test_the_installed_timer_sweeps_through_the_gate():
    source = __import__("inspect").getsource(gc_policy.install)
    assert "timer.timeout.connect(_collect_on_tick)" in source


def _cyclic_closures(qualname_part):
    gc.collect()
    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        gc.collect()
        found = [o for o in gc.garbage if callable(o)
                 and qualname_part in getattr(o, "__qualname__", "")]
    finally:
        gc.set_debug(0)
        gc.garbage.clear()
    return found


def test_building_a_screens_settings_leaves_no_cycle(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    for section in getattr(screen, "_settings_sections", ()):
        expand = getattr(section, "set_expanded", None)
        if callable(expand):
            expand(True)
    QApplication.processEvents()
    assert _cyclic_closures("_with_the_controls.<locals>.swap") == []


def test_the_usage_poll_leaves_no_cycle_once_answered(qtbot, monkeypatch):
    from spacr import crashreport
    from spacr.qt.widgets.console_panel import ConsolePanel

    monkeypatch.setattr(crashreport, "fetch_allowance", lambda: None)
    panel = ConsolePanel()
    qtbot.addWidget(panel)
    monkeypatch.setattr(panel, "_hosted_is_current", lambda: True)
    panel.refresh_ai_usage()
    qtbot.wait(400)
    assert _cyclic_closures("refresh_ai_usage.<locals>._collect") == []


def test_folding_actions_and_dragging_the_handle_survive_forced_collections(
        qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    for key in ("regression", "power", "dose_response"):
        screen = AppScreen(key)
        qtbot.addWidget(screen)
        screen.resize(1400, 900)
        screen.show()
        qtbot.waitExposed(screen)
        folder = getattr(screen, "_actions_folder", None)
        if folder is not None and not folder.shut:
            QTest.mouseClick(folder.heading, Qt.LeftButton)
            gc.collect()
        splitter = getattr(screen, "_body_splitter", None)
        if splitter is None or splitter.count() != 2:
            continue
        handle = splitter.handle(1)
        start = handle.rect().center()
        for step in range(3):
            QTest.mousePress(handle, Qt.LeftButton, pos=start)
            gc.collect()
            gc_policy._collect_on_tick()
            QTest.mouseMove(handle, start + QPoint(40 * (step + 1), 0))
            gc.collect()
            QTest.mouseRelease(handle, Qt.LeftButton, pos=handle.rect().center())
            gc.collect()
            QApplication.processEvents()
        assert splitter.sizes()[0] > 0
