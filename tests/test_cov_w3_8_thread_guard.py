"""The Qt thread guard: what it wraps, what it reports, and what it lets be.

``install()`` patches ``QObject.__init__``, ``QTimer.start`` and
``QObject.startTimer`` process-wide. Every test here installs into a fresh
copy of the module, captures the three wrappers, and puts the originals
straight back, so the rest of the session runs unguarded. The captured
wrappers are then called directly with real Qt objects.
"""
from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _fresh_module(name="spacr_thread_guard_under_test"):
    """Execute spacr/qt/thread_guard.py again under a private module name."""
    import spacr.qt.thread_guard as installed

    spec = importlib.util.spec_from_file_location(
        name, Path(installed.__file__))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def guard(qapp):
    """A freshly-loaded guard whose wrappers are captured, then unpatched.

    Yields ``(module, wrappers)`` where ``wrappers`` maps ``"timer_start"``,
    ``"object_start"`` and ``"object_init"`` to the functions ``install()``
    put on the Qt classes. The Qt classes are restored before any test body
    runs, so a failure cannot leave the process guarded.
    """
    from PySide6.QtCore import QObject, QTimer

    module = _fresh_module()
    saved = (QObject.__init__, QTimer.start, QObject.startTimer)
    took = module.install()
    wrappers = {
        "object_init": QObject.__init__,
        "timer_start": QTimer.start,
        "object_start": QObject.startTimer,
    }
    QObject.__init__, QTimer.start, QObject.startTimer = saved
    assert took is True
    try:
        yield module, wrappers
    finally:
        QObject.__init__, QTimer.start, QObject.startTimer = saved


def test_install_reports_that_it_took_and_refuses_a_second_time(guard):
    """Installing twice must not wrap the wrapper and double every line."""
    module, _wrappers = guard

    assert module.install() is False


def test_install_declines_when_pyside_cannot_be_imported(monkeypatch):
    """No PySide6 is not a crash — the guard simply does not take."""
    module = _fresh_module("spacr_thread_guard_no_pyside")
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", None)

    assert module.install() is False
    assert module.offences() == []


def test_offences_and_born_off_thread_hand_out_copies(guard):
    """A caller mutating the returned list must not lose the record."""
    module, _wrappers = guard
    module._OFFENCES.append("a stack")
    module._BORN_OFF_THREAD.append("a birth")

    offences = module.offences()
    births = module.born_off_thread()
    offences.clear()
    births.clear()

    assert module.offences() == ["a stack"]
    assert module.born_off_thread() == ["a birth"]


def test_a_timer_started_on_its_own_thread_is_not_reported(guard, qapp):
    """The common path must stay silent, or the one real line is buried."""
    from PySide6.QtCore import QTimer

    module, wrappers = guard
    timer = QTimer()
    timer.setSingleShot(True)

    wrappers["timer_start"](timer, 0)

    assert module.offences() == []
    timer.stop()


def test_a_timer_whose_object_lives_elsewhere_is_reported(guard, qapp, caplog):
    """Affinity is compared with ``==``; a different thread is an offence."""
    import logging

    from PySide6.QtCore import QThread, QTimer

    module, wrappers = guard
    timer = QTimer()
    timer.setSingleShot(True)
    other = QThread()
    timer.thread = lambda: other

    with caplog.at_level(logging.WARNING, logger="spacr.qt.thread_guard"):
        wrappers["timer_start"](timer, 0)

    assert len(module.offences()) == 1
    assert "QTimer.start is illegal here" in caplog.text
    assert "the object lives on" in module_last_message(caplog)
    timer.stop()


def module_last_message(caplog):
    """The formatted text of the most recent captured record."""
    return caplog.records[-1].getMessage()


def test_an_object_that_will_not_say_where_it_lives_is_left_alone(guard, qapp):
    """A ``thread()`` that raises is unanswerable, so nothing is claimed."""
    from PySide6.QtCore import QTimer

    module, wrappers = guard
    timer = QTimer()
    timer.setSingleShot(True)

    def _explode():
        raise RuntimeError("the C++ object is gone")

    timer.thread = _explode
    wrappers["timer_start"](timer, 0)

    assert module.offences() == []
    timer.stop()


def test_an_object_with_no_thread_at_all_is_left_alone(guard, qapp):
    """A null affinity is not evidence of a wrong thread."""
    from PySide6.QtCore import QTimer

    module, wrappers = guard
    timer = QTimer()
    timer.setSingleShot(True)
    timer.thread = lambda: None

    wrappers["timer_start"](timer, 0)

    assert module.offences() == []
    timer.stop()


def test_start_timer_is_guarded_the_same_way(guard, qapp, caplog):
    """``QObject.startTimer`` is the other entry point Qt refuses."""
    import logging

    from PySide6.QtCore import QObject, QThread

    module, wrappers = guard
    obj = QObject()

    timer_id = wrappers["object_start"](obj, 10000)
    assert module.offences() == []
    obj.killTimer(timer_id)

    other = QThread()
    obj.thread = lambda: other
    with caplog.at_level(logging.WARNING, logger="spacr.qt.thread_guard"):
        timer_id = wrappers["object_start"](obj, 10000)

    assert len(module.offences()) == 1
    assert "QObject.startTimer is illegal here" in caplog.text
    del obj.thread
    obj.killTimer(timer_id)


def test_an_object_born_on_a_worker_thread_is_recorded(guard, qapp, caplog):
    """The birth stack is what names the code the timer warning cannot."""
    import logging

    from PySide6.QtCore import QObject

    module, wrappers = guard

    class _Probe(QObject):
        pass

    made = []

    def _build():
        obj = _Probe.__new__(_Probe)
        wrappers["object_init"](obj)
        made.append(obj)

    with caplog.at_level(logging.WARNING, logger="spacr.qt.thread_guard"):
        worker = threading.Thread(target=_build, name="a-worker")
        worker.start()
        worker.join()

    assert len(made) == 1
    stacks = module.born_off_thread()
    assert len(stacks) == 1
    assert "_build" in stacks[0]
    assert "_Probe was CONSTRUCTED on" in caplog.records[-1].getMessage()
    assert "a-worker" in caplog.records[-1].getMessage()


def test_a_flood_of_off_thread_births_stops_at_the_cap(guard, qapp):
    """A legitimate producer must not be able to fill the log."""
    from PySide6.QtCore import QObject

    module, wrappers = guard
    kept = []

    def _build_many():
        for _ in range(module._BORN_LIMIT + 5):
            obj = QObject.__new__(QObject)
            wrappers["object_init"](obj)
            kept.append(obj)

    worker = threading.Thread(target=_build_many, name="a-flood")
    worker.start()
    worker.join()

    assert len(kept) == module._BORN_LIMIT + 5
    assert len(module.born_off_thread()) == module._BORN_LIMIT


def test_a_birth_on_the_gui_thread_is_not_recorded(guard, qapp):
    """Only off-thread construction is interesting."""
    from PySide6.QtCore import QObject

    module, wrappers = guard
    obj = QObject.__new__(QObject)
    wrappers["object_init"](obj)

    assert module.born_off_thread() == []


def test_the_birth_guard_never_breaks_construction(guard, qapp, monkeypatch):
    """A failure inside the diagnostic must not stop the object being built."""
    from PySide6.QtCore import QObject

    module, wrappers = guard

    class _BrokenThreading:
        @staticmethod
        def current_thread():
            raise RuntimeError("threading is unavailable")

        @staticmethod
        def main_thread():
            raise RuntimeError("threading is unavailable")

    monkeypatch.setattr(module, "threading", _BrokenThreading)
    obj = QObject.__new__(QObject)

    wrappers["object_init"](obj)

    assert obj.objectName() == ""
    assert module.born_off_thread() == []
