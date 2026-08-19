"""An exception out of QThread.run aborts spaCR. None may escape.

Caught in the full suite on 2026-08-19: "Fatal Python error: Aborted", with
one worker thread mid-`PIL.Image.resize` and the message "Error calling
Python override of QThread::run()" interleaved into the dump.

THE MECHANISM, because it is not obvious and it bit twice. A QThread's
`run` executes while its own widget can be torn down underneath it -- Qt
destroys the C++ half with the parent, and the Python shell survives. Every
later call into that shell (`emit`, `isInterruptionRequested`) then raises
`RuntimeError: Internal C++ object already deleted`. Raised anywhere else
that is an ordinary error; raised inside `run` it escapes a VIRTUAL
OVERRIDE, and PySide6 answers that by aborting the process.

So a screen closed at the wrong moment does not lose a thumbnail -- it takes
the whole application down, which is the class of crash the maintainer has
been reporting.
"""
import ast
import pathlib

import pytest

#: Every QThread subclass in the Qt layer. Listed by module so a new one has
#: to be added here deliberately -- see the last test.
WORKER_MODULES = (
    "spacr/qt/app.py",
    "spacr/qt/widgets/timelapse_preview.py",
    "spacr/qt/widgets/motility_preview.py",
    "spacr/qt/widgets/live_preview.py",
    "spacr/qt/widgets/umap_explorer.py",
    "spacr/qt/screens/hyperparam.py",
    "spacr/qt/screens/make_masks.py",
    "spacr/qt/screens/queue.py",
    "spacr/qt/screens/annotate.py",
)

#: Calls that reach the worker's own C++ half and so can raise once it is
#: gone. `emit` is the one that actually bit; the others are the same shape.
DANGEROUS = ("emit", "isInterruptionRequested")


def _run_methods(path):
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        bases = [getattr(b, "id", getattr(b, "attr", "")) for b in cls.bases]
        if "QThread" not in bases:
            continue
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == "run":
                yield cls.name, fn


def _inside_a_try(node, run):
    for parent in ast.walk(run):
        if isinstance(parent, ast.Try):
            if any(sub is node for sub in ast.walk(parent)):
                return True
    return False


def _unguarded(run):
    out = []
    for call in [n for n in ast.walk(run) if isinstance(n, ast.Call)]:
        if getattr(call.func, "attr", "") in DANGEROUS:
            if not _inside_a_try(call, run):
                out.append(call.lineno)
    return out


@pytest.mark.parametrize("path", WORKER_MODULES)
def test_no_qt_call_in_run_is_unguarded(path):
    """`emit_safely(...)` counts as guarded -- it IS the try."""
    offenders = {}
    for name, run in _run_methods(path):
        lines = _unguarded(run)
        if lines:
            offenders[name] = lines
    assert not offenders, (
        f"{path}: a call into the worker's own C++ half sits outside any "
        f"try in run(). Once the screen is gone it raises RuntimeError, and "
        f"an exception out of a QThread::run override ABORTS spaCR. Use "
        f"bridge.emit_safely. Offenders: {offenders}")


def test_every_qthread_in_the_qt_layer_is_covered():
    """A new worker must be added to WORKER_MODULES, not silently skipped."""
    root = pathlib.Path("spacr/qt")
    found = set()
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "(QThread)" in text:
            found.add(str(path))
    assert found == set(WORKER_MODULES), (
        f"QThread subclasses this test does not check: "
        f"{sorted(found - set(WORKER_MODULES))}; "
        f"listed but gone: {sorted(set(WORKER_MODULES) - found)}")


# ------------------------------------------------------- the helper itself


def test_emit_safely_delivers_a_live_signal(qtbot):
    from PySide6.QtCore import QObject, Signal

    from spacr.qt.bridge import emit_safely

    class Source(QObject):
        fired = Signal(int)

    source = Source()
    seen = []
    source.fired.connect(seen.append)

    assert emit_safely(source.fired, 7) is True
    assert seen == [7]


def test_emit_safely_survives_a_destroyed_receiver(qtbot):
    """The real case: the C++ half is gone and `emit` raises."""
    from spacr.qt.bridge import emit_safely

    class Dead:
        def emit(self, *args):
            raise RuntimeError("Internal C++ object already deleted")

    assert emit_safely(Dead(), 1, 2) is False


def test_emit_safely_does_not_swallow_a_real_error(qtbot):
    """A bug in a connected slot must still be visible.

    Only RuntimeError is the teardown signature; anything else is the
    application being wrong and has to surface.
    """
    from spacr.qt.bridge import emit_safely

    class Broken:
        def emit(self, *args):
            raise ValueError("a slot is wrong")

    with pytest.raises(ValueError):
        emit_safely(Broken(), 1)
