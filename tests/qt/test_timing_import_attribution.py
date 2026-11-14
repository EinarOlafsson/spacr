"""Which spaCR line the import timer blames, and what it records when it cannot tell.

``_the_spacr_frame`` decides whether a code path belongs to spaCR: under the
installed package, or carrying a ``/spacr/`` segment outside the interpreter's
own libraries (an environment merely *named* spacr is not the package).
``_ImportTimer.find_spec`` and the ``exec_module`` wrappers that
``_install_import_timer`` puts on the loaders walk the stack for such a frame;
the walk must stop at the top of a worker's short stack, survive a runtime
with no ``sys._getframe``, and never lose the module's name or cost.

Synthetic callers are compiled with a chosen ``co_filename`` (``_function_from``)
so a path can be put on the stack without writing into the tree under test.
"""
from __future__ import annotations

import importlib.machinery
import os
import sys
import threading
import time
from types import ModuleType

import pytest

pytest.importorskip("PySide6")

from spacr.qt import timing  # noqa: E402


@pytest.fixture
def timing_on(monkeypatch):
    """One empty, enabled instrumentation session with no process residue."""
    monkeypatch.setattr(timing, "ENABLED", True)
    monkeypatch.setattr(timing, "_START", time.perf_counter())
    monkeypatch.setattr(timing, "_EVENT_LOOP_STARTED_AT", None)
    for name in ("_SPANS", "_IMPORTS", "_STALLS", "_MARKS", "_READINESS",
                 "_READY_CALLBACKS", "_ACTIVE_PROBES"):
        monkeypatch.setattr(timing, name, [])
    yield


@pytest.fixture
def loader_restored():
    """Put both loaders' ``exec_module`` back however the test ends."""
    source = importlib.machinery.SourceFileLoader.exec_module
    extension = importlib.machinery.ExtensionFileLoader.exec_module
    try:
        yield
    finally:
        importlib.machinery.SourceFileLoader.exec_module = source
        importlib.machinery.ExtensionFileLoader.exec_module = extension


def _function_from(filename, source, name):
    """A function whose frames claim to live in ``filename``."""
    namespace: dict = {}
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102
    return namespace[name]


class _NoFrames:
    """``sys`` whose ``_getframe`` raises, as on a runtime without one."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, item):
        return getattr(self._real, item)

    def _getframe(self, depth=0):
        raise ValueError("call stack is not deep enough")


def _slow_module(tmp_path, name):
    """A real source file and loader whose execution takes about 20 ms."""
    path = tmp_path / f"{name}.py"
    path.write_text("import time\ntime.sleep(0.02)\nVALUE = 1\n",
                    encoding="utf-8")
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    module = ModuleType(name)
    module.__file__ = str(path)
    return loader, module


def _in_thread(target, *args, name="probe-thread"):
    thread = threading.Thread(target=target, args=args, name=name)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive()


def test_the_spacr_frame_names_only_spacr_files(monkeypatch, subtests):
    """Installed-package and source-tree paths are named; libraries and the timer are not."""
    import sysconfig

    import spacr

    real_root = os.path.dirname(os.path.abspath(spacr.__file__))

    def reset():
        monkeypatch.setattr(timing, "_SPACR_ROOT", None)
        monkeypatch.setattr(timing, "_LIBRARY_DIRS", ())

    with subtests.test(msg="forced roots"):
        monkeypatch.setattr(timing, "_SPACR_ROOT", "/real/pkg/spacr/")
        monkeypatch.setattr(timing, "_LIBRARY_DIRS",
                            ("/opt/envs/spacr/lib/python3.12/",))
        cases = [
            ("/real/pkg/spacr/qt/app.py", "qt/app.py"),
            ("/real/pkg/spacr/core.py", "core.py"),
            ("/real/pkg/spacr/qt/timing.py", ""),
            ("", ""),
            ("/opt/envs/spacr/lib/python3.12/site-packages/_pytest/python.py",
             ""),
            ("/nowhere/spacr/qt/asked_here.py", "qt/asked_here.py"),
            ("/a/spacr/b/spacr/c.py", "c.py"),
            ("/nowhere/elsewhere/asked_here.py", ""),
            ("/nowhere/spacrish/app.py", ""),
        ]
        for path, expected in cases:
            assert timing._the_spacr_frame(path) == expected, path

    with subtests.test(msg="resolved from this interpreter"):
        reset()
        assert timing._the_spacr_frame(
            os.path.join(real_root, "qt", "app.py")) == "qt/app.py"
        assert timing._SPACR_ROOT == real_root + os.sep
        assert timing._LIBRARY_DIRS
        assert all(d.endswith(os.sep) and not timing._SPACR_ROOT.startswith(d)
                   for d in timing._LIBRARY_DIRS)
        assert timing._the_spacr_frame(
            os.path.join(real_root, "qt", "timing.py")) == ""
        assert timing._the_spacr_frame("/nowhere/elsewhere/x.py") == ""

    with subtests.test(msg="a library directory holding the package is kept out"):
        reset()
        parent = os.path.dirname(real_root)
        monkeypatch.setattr(sysconfig, "get_paths", lambda: {
            "stdlib": "/opt/py/lib", "purelib": parent})
        assert timing._the_spacr_frame("/nowhere/spacr/qt/x.py") == "qt/x.py"
        assert timing._LIBRARY_DIRS == ("/opt/py/lib" + os.sep,)
        assert timing._the_spacr_frame("/opt/py/lib/spacr/x.py") == ""

    with subtests.test(msg="unreadable library paths leave only the segment test"):
        reset()

        def broken():
            raise OSError("no paths")

        monkeypatch.setattr(sysconfig, "get_paths", broken)
        assert timing._the_spacr_frame("/opt/envs/spacr/lib/x.py") == "lib/x.py"
        assert timing._LIBRARY_DIRS == ()

    with subtests.test(msg="a package that cannot be imported does not raise"):
        import builtins

        reset()
        monkeypatch.undo()
        reset()
        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == "spacr":
                raise ImportError("no spacr here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)
        assert timing._the_spacr_frame("/nowhere/spacr/qt/app.py") == \
            "qt/app.py"
        assert timing._SPACR_ROOT == ""
        assert timing._the_spacr_frame("/nowhere/elsewhere/app.py") == ""


def test_the_import_finder_names_the_spacr_frame_that_asked(
        timing_on, monkeypatch, subtests):
    """``find_spec`` records (module, start, caller) and never resolves an import."""
    body = "def ask(finder, name):\n    return finder.find_spec(name)\n"
    inside = _function_from("/nowhere/spacr/qt/asked_here.py", body, "ask")
    outside = _function_from("/nowhere/elsewhere/asked_here.py", body, "ask")
    inner = _function_from(
        "/nowhere/elsewhere/inner_probe.py",
        "def inner(finder, name, box):\n"
        "    box.append(finder.find_spec(name))\n", "inner")
    outer = _function_from(
        "/nowhere/spacr/qt/outer_probe.py",
        "def outer(finder, name, box, inner):\n"
        "    inner(finder, name, box)\n", "outer")
    box: list = []

    def deep(finder):
        _in_thread(outer, finder, "a_module_for_the_deep_walk", box, inner)

    def shallow(finder):
        _in_thread(inner, finder, "a_module_for_the_short_walk", box)

    # case, how the finder is asked, module name, expected caller
    cases = [
        ("a spaCR caller", lambda f: inside(f, "mod_never_imported_a"),
         "mod_never_imported_a", "qt/asked_here.py:2"),
        ("a caller outside spaCR",
         lambda f: outside(f, "mod_never_imported_b"),
         "mod_never_imported_b", ""),
        ("a worker thread, spaCR two frames up", deep,
         "a_module_for_the_deep_walk", "qt/outer_probe.py:2"),
        ("a worker stack that ends first", shallow,
         "a_module_for_the_short_walk", ""),
    ]
    for case, ask, module, caller in cases:
        with subtests.test(msg=case):
            finder = timing._ImportTimer()
            before = time.perf_counter()
            assert ask(finder) is None
            name, started, by = finder._pending
            assert (name, by) == (module, caller)
            assert before <= started <= time.perf_counter()
    assert box == [None, None]

    with subtests.test(msg="a runtime with no frames still records the import"):
        finder = timing._ImportTimer()
        monkeypatch.setattr(timing, "sys", _NoFrames(sys))
        inside(finder, "mod_never_imported_c")
        monkeypatch.setattr(timing, "sys", sys)
        assert (finder._pending[0], finder._pending[2]) == (
            "mod_never_imported_c", "")
        assert isinstance(finder._pending[1], float)

    with subtests.test(msg="a module already imported is not timed"):
        finder = timing._ImportTimer()
        assert finder.find_spec("os") is None
        assert not hasattr(finder, "_pending")
        assert finder.find_module("os") is None
        assert finder.note() is None
        assert finder._depth == 0


def test_installed_loaders_record_slow_imports_with_their_caller(
        timing_on, loader_restored, monkeypatch, tmp_path, subtests):
    """Each slow import is recorded once with its thread, cost and spaCR caller."""
    with subtests.test(msg="no extension loader: source imports still timed"):
        source_before = importlib.machinery.SourceFileLoader.exec_module
        extension_before = importlib.machinery.ExtensionFileLoader.exec_module
        with monkeypatch.context() as absent:
            absent.setattr(timing, "_IMPORT_TIMER_INSTALLED", False)
            absent.delattr(importlib.machinery, "ExtensionFileLoader")
            timing._install_import_timer()
            assert timing._IMPORT_TIMER_INSTALLED is True
            assert (importlib.machinery.SourceFileLoader.exec_module
                    is not source_before)
        assert (importlib.machinery.ExtensionFileLoader.exec_module
                is extension_before)

    with subtests.test(msg="installing twice wraps once"):
        importlib.machinery.SourceFileLoader.exec_module = source_before
        monkeypatch.setattr(timing, "_IMPORT_TIMER_INSTALLED", False)
        timing._install_import_timer()
        wrapped = importlib.machinery.SourceFileLoader.exec_module
        assert wrapped is not source_before
        assert (importlib.machinery.ExtensionFileLoader.exec_module
                is not extension_before)
        timing._install_import_timer()
        assert importlib.machinery.SourceFileLoader.exec_module is wrapped

    monkeypatch.setattr(timing, "IMPORT_FLOOR_MS", 1.0)
    attributable = _function_from(
        "/nowhere/spacr/qt/importer_probe.py",
        "def run(loader, module):\n"
        "    _call(loader, module)\n"
        "def _call(loader, module):\n"
        "    loader.exec_module(module)\n", "run")
    anonymous = _function_from(
        "/nowhere/elsewhere/thread_probe.py",
        "def run(loader, module):\n    loader.exec_module(module)\n", "run")
    main = threading.current_thread().name

    def on_main(run):
        return lambda loader, module: run(loader, module)

    def on_worker(run):
        return lambda loader, module: _in_thread(run, loader, module,
                                                 name="preloader-probe")

    def without_frames(run):
        def call(loader, module):
            monkeypatch.setattr(timing, "sys", _NoFrames(sys))
            try:
                run(loader, module)
            finally:
                monkeypatch.setattr(timing, "sys", sys)
        return call

    # case, how the import runs, expected caller, expected thread
    cases = [
        ("spaCR caller on the GUI thread", on_main(attributable),
         "qt/importer_probe.py:2", main),
        ("spaCR caller on a worker", on_worker(attributable),
         "qt/importer_probe.py:2", "preloader-probe"),
        ("anonymous worker stack", on_worker(anonymous), "",
         "preloader-probe"),
        ("bare loader on a worker", on_worker(
            lambda loader, module: loader.exec_module(module)), "",
         "preloader-probe"),
        ("no frames from the runtime", without_frames(attributable), "",
         main),
    ]
    for index, (case, run, caller, thread) in enumerate(cases):
        with subtests.test(msg=case):
            loader, module = _slow_module(tmp_path, f"slow_probe_{index}")
            before = len(timing._IMPORTS)
            run(loader, module)
            assert module.VALUE == 1
            rows = timing._IMPORTS[before:]
            assert [(r["name"], r["by"], r["thread"]) for r in rows] == [
                (module.__name__, caller, thread)]
            assert rows[0]["took"] >= 0.019
            assert rows[0]["at"] >= 0.0

    with subtests.test(msg="an import under the floor is not recorded"):
        monkeypatch.setattr(timing, "IMPORT_FLOOR_MS", 60_000.0)
        loader, module = _slow_module(tmp_path, "under_the_floor_probe")
        before = len(timing._IMPORTS)
        attributable(loader, module)
        assert module.VALUE == 1
        assert timing._IMPORTS[before:] == []
