"""Which objects the exit hook keeps alive, measured in-process.

:func:`spacr.qt._keep_weakly_held_qt_wrappers_to_the_end` runs at
interpreter exit, so ``test_quit_with_weakly_held_wrappers_20261024`` can
only observe that the exit no longer crashes. These tests call it directly
over a controlled object list and record the single reference it hands to
``Py_IncRef``: only Qt wrappers that something holds weakly are kept, the
application object never is, and nothing is pinned when there is nothing to
keep or when Qt cannot be imported.
"""
from __future__ import annotations

import ctypes
import gc
import sys
import weakref

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


class _Plain:
    """A weakly referenceable object that is not a Qt wrapper."""


@pytest.fixture
def pinned(monkeypatch):
    """Record what the hook passes to ``Py_IncRef`` instead of pinning it."""
    calls = []
    monkeypatch.setattr(ctypes, "PYFUNCTYPE",
                        lambda *types: lambda spec: calls.append)
    return calls


def _objects(monkeypatch, objects):
    """Make the hook see exactly ``objects`` as the live heap."""
    monkeypatch.setattr(gc, "get_objects", lambda *args: list(objects))


def test_only_weakly_held_wrappers_are_kept_and_never_the_application(
        qapp, monkeypatch, pinned):
    import spacr.qt
    from PySide6.QtCore import QObject

    first, second, unwatched = QObject(), QObject(), QObject()
    plain = _Plain()
    references = [weakref.ref(first), weakref.ref(second),
                  weakref.ref(qapp), weakref.ref(plain)]
    _objects(monkeypatch, [first, plain, qapp, unwatched, second])

    spacr.qt._keep_weakly_held_qt_wrappers_to_the_end()

    assert len(pinned) == 1
    kept = pinned[0]
    assert len(kept) == 2
    assert kept[0] is first and kept[1] is second
    assert all(reference() is not None for reference in references)


def test_nothing_is_pinned_when_no_wrapper_is_weakly_held(
        qapp, monkeypatch, pinned):
    import spacr.qt
    from PySide6.QtCore import QObject

    unwatched = QObject()
    _objects(monkeypatch, [unwatched, _Plain()])

    spacr.qt._keep_weakly_held_qt_wrappers_to_the_end()

    assert pinned == []


def test_nothing_is_pinned_when_qt_cannot_be_imported(monkeypatch, pinned):
    import spacr.qt

    walked = []
    monkeypatch.setattr(gc, "get_objects", lambda *args: walked.append(1) or [])
    monkeypatch.setitem(sys.modules, "shiboken6", None)

    spacr.qt._keep_weakly_held_qt_wrappers_to_the_end()

    assert pinned == []
    assert walked == []


def test_the_hook_pins_even_after_rubicon_retypes_the_shared_incref(tmp_path):
    """On macOS wgpu imports rubicon-objc, which sets
    ``ctypes.pythonapi.Py_IncRef.argtypes`` to ``POINTER(PyObject)``; the
    shared call then raised ``ctypes.ArgumentError`` and pinned nothing."""
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    code = (
        "import ctypes, sys, weakref\n"
        "ctypes.pythonapi.Py_IncRef.argtypes = [ctypes.POINTER(ctypes.py_object)]\n"
        "from PySide6.QtCore import QCoreApplication, QObject\n"
        "app = QCoreApplication([])\n"
        "import spacr.qt\n"
        "held = QObject(); ref = weakref.ref(held)\n"
        "before = sys.getrefcount(held)\n"
        "spacr.qt._keep_weakly_held_qt_wrappers_to_the_end()\n"
        "print('pinned', sys.getrefcount(held) - before)\n")
    done = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True,
                          text=True, timeout=120, env={**__import__("os").environ,
                                                       "QT_QPA_PLATFORM": "offscreen",
                                                       "PYTHONPATH": str(root)})
    assert "ArgumentError" not in done.stderr, done.stderr
    assert "pinned 1" in done.stdout, (done.stdout, done.stderr)
