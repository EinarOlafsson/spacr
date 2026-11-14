"""Readiness probes, readiness subscribers, the hardware snapshot and ``begin``.

``watch_interactive`` installs a probe that reports a screen once a usable
control has painted after the event loop began; the probe must retire rather
than raise when its widgets die underneath it, and a subscriber that raises
must not stop the next.  The snapshot facts (GPU memory, total memory,
displays, performance level) are reported only when they could be read --
"could not ask" (``None``) is a different fact from zero.  ``begin`` dates the
clock once, from a plausible spawn timestamp or from the module import.
"""
from __future__ import annotations

import importlib.machinery
import os
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("PySide6")

import shiboken6  # noqa: E402
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget  # noqa: E402

from spacr.qt import timing  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def timing_on(monkeypatch):
    """One empty, enabled instrumentation session; probes retired afterwards."""
    monkeypatch.setattr(timing, "ENABLED", True)
    monkeypatch.setattr(timing, "_START", time.perf_counter())
    monkeypatch.setattr(timing, "_EVENT_LOOP_STARTED_AT", None)
    for name in ("_SPANS", "_IMPORTS", "_STALLS", "_MARKS", "_READINESS",
                 "_READY_CALLBACKS", "_ACTIVE_PROBES"):
        monkeypatch.setattr(timing, name, [])
    yield
    for probe in list(timing._ACTIVE_PROBES):
        try:
            probe._retire()
        except (AttributeError, RuntimeError):
            pass


class _PaintedButton(QPushButton):
    """Counts repaints through ``paintEvent`` (the probe calls ``QWidget.update``)."""

    update_calls = 0

    def paintEvent(self, event):  # noqa: D102, N802
        self.update_calls += 1
        super().paintEvent(event)


class _RefusesEventFilter(QWidget):
    """A root already being torn down: ``installEventFilter`` raises."""

    def installEventFilter(self, watcher):  # noqa: D102, N802
        raise RuntimeError("Internal C++ object already deleted.")


def _page(qtbot, *names, root_cls=QWidget, button_cls=QPushButton):
    root = root_cls()
    layout = QVBoxLayout(root)
    buttons = []
    for name in names or ("usable",):
        button = button_cls(name)
        layout.addWidget(button)
        buttons.append(button)
    qtbot.addWidget(root)
    return root, buttons


def _loop_started(monkeypatch):
    monkeypatch.setattr(timing, "_EVENT_LOOP_STARTED_AT", time.perf_counter())


def test_watch_interactive_installs_a_probe_only_when_it_can(
        qtbot, timing_on, monkeypatch, subtests):
    """No widget or timing off: no probe; a control root counts itself."""
    root, (child,) = _page(qtbot)
    button = QPushButton("the whole screen")
    qtbot.addWidget(button)

    with subtests.test(msg="a container watches its child controls"):
        probe = timing.watch_interactive(root, "interactive module", "on")
        assert probe.controls == (child,)
        assert (probe.report_name, probe.report_detail, probe.done) == (
            "interactive module", "on", False)
        assert timing._ACTIVE_PROBES == [probe]

    with subtests.test(msg="a screen that is itself a control counts itself"):
        itself = timing.watch_interactive(button, "interactive module", "b")
        assert itself.controls == (button,)
        assert timing._ACTIVE_PROBES == [probe, itself]

    with subtests.test(msg="the same name and detail replace the old probe"):
        again = timing.watch_interactive(root, "interactive module", "on")
        assert probe.done is True
        assert timing._ACTIVE_PROBES == [itself, again]

    with subtests.test(msg="no widget, or timing off, installs nothing"):
        assert timing.watch_interactive(None, "interactive module", "x") \
            is None
        monkeypatch.setattr(timing, "ENABLED", False)
        other, _ = _page(qtbot)
        assert timing.watch_interactive(other, "interactive module", "off") \
            is None
        assert timing._ACTIVE_PROBES == [itself, again]


def test_a_probe_retires_rather_than_reports_when_its_widgets_die(
        qtbot, timing_on, monkeypatch, subtests):
    """Deleted controls, a deleted root and a refused filter each end exactly."""
    with subtests.test(msg="only visible controls are repainted at loop start"):
        root, (shown, hidden) = _page(qtbot, "shown", "hidden",
                                      button_cls=_PaintedButton)
        timing.watch_interactive(root, "interactive module", "visibility")
        root.show()
        hidden.hide()
        qtbot.wait(20)
        before = (shown.update_calls, hidden.update_calls)
        timing.event_loop_started()
        qtbot.wait(20)
        assert shown.update_calls > before[0]
        assert hidden.update_calls == before[1]
        qtbot.waitUntil(lambda: len(timing._READINESS) == 1, timeout=2000)
        entry = timing._READINESS[0]
        assert (entry["name"], entry["detail"], entry["usable_controls"],
                entry["painted_usable_controls"], entry["controls"],
                entry["budget_s"], entry["within_budget"]) == (
            "interactive module", "visibility", 1, 1, ["_PaintedButton"],
            None, None)
        assert [m["name"] for m in timing._MARKS] == [
            "event loop began", "interactive module"]
        monkeypatch.setattr(timing, "_EVENT_LOOP_STARTED_AT", None)
        timing._READINESS.clear()

    with subtests.test(msg="a control deleted before the loop starts"):
        monkeypatch.setattr(timing, "_EVENT_LOOP_STARTED_AT", None)
        root, (button,) = _page(qtbot)
        probe = timing.watch_interactive(root, "interactive module", "deleted")
        root.show()
        qtbot.wait(20)
        assert probe in timing._ACTIVE_PROBES
        shiboken6.delete(button)
        timing.event_loop_started()
        assert probe.done is True
        assert probe not in timing._ACTIVE_PROBES
        assert timing._READINESS == []

    with subtests.test(msg="a control deleted before the settle is not usable"):
        monkeypatch.setattr(timing, "_EVENT_LOOP_STARTED_AT", None)
        root, (survivor, doomed) = _page(qtbot, "survivor", "doomed")
        survivor.setObjectName("survivor")
        probe = timing.watch_interactive(root, "interactive module", "usable")
        root.show()
        qtbot.wait(30)
        assert probe.painted_controls == {id(survivor), id(doomed)}
        shiboken6.delete(doomed)
        _loop_started(monkeypatch)
        probe._settle()
        entry = timing._READINESS[-1]
        assert (entry["usable_controls"], entry["painted_usable_controls"],
                entry["controls"], entry["root_painted"]) == (
            1, 1, ["survivor"], True)
        assert probe.done is True
        timing._READINESS.clear()

    with subtests.test(msg="a root destroyed before it settles"):
        monkeypatch.setattr(timing, "_EVENT_LOOP_STARTED_AT", None)
        root, _buttons = _page(qtbot)
        healthy = timing.watch_interactive(root, "still here", "probe")
        doomed = timing.watch_interactive(root, "gone", "probe")
        root.show()
        qtbot.wait(30)
        _loop_started(monkeypatch)
        healthy._settle()
        ghost = QWidget()
        shiboken6.delete(ghost)
        doomed.root = ghost
        doomed._settle()
        assert [entry["name"] for entry in timing._READINESS] == [
            "still here"]
        assert doomed.done is True
        assert doomed not in timing._ACTIVE_PROBES
        timing._READINESS.clear()
        monkeypatch.setattr(timing, "_EVENT_LOOP_STARTED_AT", None)

    with subtests.test(msg="a root that refuses an event filter"):
        monkeypatch.setattr(timing, "_EVENT_LOOP_STARTED_AT", None)
        root, _buttons = _page(qtbot, root_cls=_RefusesEventFilter)
        probe = timing.watch_interactive(root, "interactive module",
                                         "no filter")
        root.show()
        qtbot.wait(30)
        timing.event_loop_started()
        qtbot.waitUntil(lambda: len(timing._READINESS) == 1, timeout=2000)
        entry = timing._READINESS[0]
        assert (entry["painted_usable_controls"], entry["root_painted"],
                entry["detail"]) == (1, False, "no filter")
        assert probe.done is True


def test_readiness_subscribers_are_isolated_from_each_other(
        qtbot, timing_on, monkeypatch, subtests):
    """A raising subscriber does not stop the next; (un)subscribing is idempotent."""
    seen: list = []

    def explodes(_entry):
        raise ValueError("a subscriber that cannot cope")

    timing.subscribe_readiness(explodes)
    timing.subscribe_readiness(seen.append)
    timing.subscribe_readiness(seen.append)
    assert timing._READY_CALLBACKS == [explodes, seen.append]

    root, _buttons = _page(qtbot)
    probe = timing.watch_interactive(root, "interactive module", "subscribers",
                                     started_at=timing._START, budget_s=60.0)
    root.show()
    qtbot.wait(30)
    _loop_started(monkeypatch)
    probe._settle()

    with subtests.test(msg="the second subscriber hears it, as a copy"):
        assert [(e["name"], e["detail"], e["budget_s"], e["within_budget"])
                for e in seen] == [
            ("interactive module", "subscribers", 60.0, True)]
        assert seen[0] == timing._READINESS[0]
        assert seen[0] is not timing._READINESS[0]

    with subtests.test(msg="unsubscribe removes once and ignores strangers"):
        timing.unsubscribe_readiness(explodes)
        timing.unsubscribe_readiness(explodes)
        timing.unsubscribe_readiness(print)
        assert timing._READY_CALLBACKS == [seen.append]


def test_gpu_memory_tells_zero_from_could_not_ask(monkeypatch, subtests):
    """No torch or a failing driver is None; an unused allocator is a true zero."""
    def _no_driver():
        raise RuntimeError("CUDA driver could not be reached")

    mib = 1024 * 1024
    # case, torch module (None = absent), expected
    cases = [
        ("torch not imported", None,
         {"allocated_mb": None, "peak_allocated_mb": None}),
        ("torch without cuda", SimpleNamespace(),
         {"allocated_mb": None, "peak_allocated_mb": None}),
        ("allocator never initialised",
         SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)),
         {"allocated_mb": 0.0, "peak_allocated_mb": 0.0}),
        ("allocator in use", SimpleNamespace(cuda=SimpleNamespace(
            is_initialized=lambda: True,
            memory_allocated=lambda: 4 * mib,
            max_memory_allocated=lambda: 8 * mib + mib // 2)),
         {"allocated_mb": 4.0, "peak_allocated_mb": 8.5}),
        ("driver raises", SimpleNamespace(cuda=SimpleNamespace(
            is_initialized=_no_driver)),
         {"allocated_mb": None, "peak_allocated_mb": None}),
    ]
    for case, torch, expected in cases:
        with subtests.test(msg=case):
            if torch is None:
                monkeypatch.delitem(sys.modules, "torch", raising=False)
            else:
                monkeypatch.setitem(sys.modules, "torch", torch)
            assert timing._gpu_memory_mb() == expected


class _WithoutSysconf:
    """``os`` with no usable ``sysconf`` (Windows)."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, item):
        return getattr(self._real, item)

    def sysconf(self, name):
        raise OSError(f"unknown configuration name {name}")


def _screen(name="probe-0"):
    return SimpleNamespace(
        name=lambda: name,
        geometry=lambda: SimpleNamespace(width=lambda: 1920,
                                         height=lambda: 1080),
        devicePixelRatio=lambda: 2.0, refreshRate=lambda: 60.0)


def test_hardware_profile_reports_only_facts_it_could_read(monkeypatch,
                                                            subtests):
    """Memory, displays and performance level fall back to None, never a guess."""
    expected_mb = (os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
                   / (1024.0 * 1024.0))
    with subtests.test(msg="memory: sysconf, then psutil, then unknown"):
        profile = timing._hardware_profile()
        assert profile["total_memory_mb"] == expected_mb
        assert profile["logical_cpu_count"] == os.cpu_count()
        monkeypatch.setattr(timing, "os", _WithoutSysconf(timing.os))
        psutil = ModuleType("psutil")
        psutil.virtual_memory = lambda: SimpleNamespace(total=8 * 1024 ** 3)
        monkeypatch.setitem(sys.modules, "psutil", psutil)
        assert timing._hardware_profile()["total_memory_mb"] == 8192.0
        monkeypatch.setitem(sys.modules, "psutil", None)
        assert timing._hardware_profile()["total_memory_mb"] is None
        monkeypatch.undo()

    widgets = ModuleType("PySide6.QtWidgets")
    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", widgets)

    def _unplugged():
        raise RuntimeError("the screen list changed under enumeration")

    display = {"name": "probe-0", "logical_width": 1920,
               "logical_height": 1080, "device_pixel_ratio": 2.0,
               "refresh_hz": 60.0}
    # case, QApplication stand-in (None = attribute absent), platform, displays
    cases = [
        ("Qt widgets without QApplication", None, None, []),
        ("no application yet", SimpleNamespace(instance=lambda: None), None,
         []),
        ("one display", SimpleNamespace(instance=lambda: SimpleNamespace(
            screens=lambda: [_screen()], platformName=lambda: "probe")),
         "probe", [display]),
        ("two displays", SimpleNamespace(instance=lambda: SimpleNamespace(
            screens=lambda: [_screen(), _screen("probe-1")],
            platformName=lambda: "probe")),
         "probe", [display, dict(display, name="probe-1")]),
        ("enumeration fails", SimpleNamespace(instance=lambda: SimpleNamespace(
            screens=_unplugged, platformName=lambda: "probe")),
         "probe", []),
    ]
    for case, application, platform, displays in cases:
        with subtests.test(msg=f"displays: {case}"):
            if application is None:
                if hasattr(widgets, "QApplication"):
                    del widgets.QApplication
            else:
                widgets.QApplication = application
            profile = timing._hardware_profile()
            assert (profile["qt_platform"], profile["displays"]) == (
                platform, displays)

    def _not_ready():
        raise RuntimeError("preferences are not loaded yet")

    def _module(**attrs):
        module = ModuleType("spacr.qt.preferences")
        module.__dict__.update(attrs)
        return module

    # case, preferences module (None = not imported), level
    levels = [
        ("not imported", None, None),
        ("body not finished: no getter", _module(), None),
        ("getter is not callable", _module(get_performance_level="high"),
         None),
        ("getter raises", _module(get_performance_level=_not_ready), None),
        ("ready", _module(get_performance_level=lambda: "balanced"),
         "balanced"),
        ("a non-string level is named", _module(
            get_performance_level=lambda: 3), "3"),
    ]
    for case, module, level in levels:
        with subtests.test(msg=f"performance level: {case}"):
            if module is None:
                monkeypatch.delitem(sys.modules, "spacr.qt.preferences",
                                    raising=False)
            else:
                monkeypatch.setitem(sys.modules, "spacr.qt.preferences",
                                    module)
            assert timing._hardware_profile()["performance_level"] == level


def test_begin_dates_the_clock_once(timing_on, monkeypatch, subtests):
    """Only a plausible spawn timestamp moves the origin; later calls change nothing."""
    plausible = str(time.time() - 2.0)
    # case, enabled, import timing, env value, moves, marks, latched
    cases = [
        ("timing off", False, False, None, False, [], False),
        ("no spawn timestamp", True, False, None, False,
         [("timing started", "timing module import")], True),
        ("unparseable timestamp", True, False, "not-a-timestamp", False,
         [("timing started", "timing module import")], True),
        ("a timestamp from another era", True, False, "1.0", False,
         [("timing started", "timing module import")], True),
        ("a timestamp from the future", True, False,
         str(time.time() + 3600.0), False,
         [("timing started", "timing module import")], True),
        ("a plausible spawn", True, False, plausible, True,
         [("timing started", "benchmark process spawn")], True),
        ("import timing armed", True, True, None, False,
         [("timing started", "timing module import")], True),
    ]
    source_before = importlib.machinery.SourceFileLoader.exec_module
    extension_before = importlib.machinery.ExtensionFileLoader.exec_module
    try:
        for case, enabled, imports, env, moves, marks, latched in cases:
            with subtests.test(msg=case):
                monkeypatch.setattr(timing, "ENABLED", enabled)
                monkeypatch.setattr(timing, "IMPORT_TIMING_ENABLED", imports)
                monkeypatch.setattr(timing, "_IMPORT_TIMER_INSTALLED", False)
                monkeypatch.setattr(timing.begin, "_done", False,
                                    raising=False)
                monkeypatch.setattr(timing, "_MARKS", [])
                if env is None:
                    monkeypatch.delenv("SPACR_TIMING_PROCESS_START",
                                       raising=False)
                else:
                    monkeypatch.setenv("SPACR_TIMING_PROCESS_START", env)
                origin = timing._START
                timing.begin()
                assert (timing._START < origin) is moves
                if moves:
                    assert 1.5 < origin - timing._START < 60.0
                else:
                    assert timing._START == origin
                assert [(m["name"], m["detail"]) for m in timing._MARKS] == \
                    marks
                assert timing.begin._done is latched
                assert timing._IMPORT_TIMER_INSTALLED is imports
                assert (importlib.machinery.SourceFileLoader.exec_module
                        is not source_before) is imports
                # A second call never moves the clock again.
                moved = timing._START
                monkeypatch.setenv("SPACR_TIMING_PROCESS_START", plausible)
                timing.begin()
                if latched:
                    assert timing._START == moved
                    assert len(timing._MARKS) == len(marks)
                importlib.machinery.SourceFileLoader.exec_module = \
                    source_before
                importlib.machinery.ExtensionFileLoader.exec_module = \
                    extension_before
    finally:
        importlib.machinery.SourceFileLoader.exec_module = source_before
        importlib.machinery.ExtensionFileLoader.exec_module = extension_before
