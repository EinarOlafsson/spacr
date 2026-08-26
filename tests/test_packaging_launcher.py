"""The frozen-app launcher must start the interface that still exists.

``packaging/spacr.spec`` freezes ``packaging/spacr_launcher.py`` as the
entry point of the Windows and macOS bundles, so whatever that file
imports is what a user gets when they double-click the installed
application. It is the one code path in the project that no console
script and no test import covers, which is how it kept naming the
removed Tk front end long after the Qt application replaced it: the
import failed, ``main`` returned 2, and the bundle exited without a
window.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

LAUNCHER = (Path(__file__).resolve().parents[1]
            / "packaging" / "spacr_launcher.py")


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "spacr_packaging_launcher", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_launcher_file_is_where_the_spec_points():
    """The spec names this path; a rename would freeze nothing."""
    assert LAUNCHER.is_file()
    spec_text = (LAUNCHER.parent / "spacr.spec").read_text(encoding="utf-8")
    assert "spacr_launcher.py" in spec_text


def test_the_launcher_starts_the_qt_application(monkeypatch):
    """``main`` calls the shipped GUI entry point and returns its code.

    The stub stands in for the real launch so the test never opens a
    window; what is asserted is that the launcher reached the function
    the package actually exports rather than a module that was deleted.
    """
    qt = pytest.importorskip("spacr.qt")
    called = []

    def _fake_run(*args, **kwargs):
        called.append((args, kwargs))
        return 0

    monkeypatch.setattr(qt, "run", _fake_run, raising=True)
    module = _load_launcher()
    assert module.main() == 0, (
        "the launcher did not reach spacr.qt.run; a frozen bundle built "
        "from it opens no window")
    assert called, "spacr.qt.run was never called"


def test_the_launcher_reports_a_broken_install_instead_of_crashing(
        monkeypatch):
    """A missing GUI extra is a message and a non-zero code, not a traceback."""
    module = _load_launcher()
    monkeypatch.setitem(sys.modules, "spacr.qt", None)
    assert module.main() == 2


def test_the_launcher_does_not_name_the_removed_tk_front_end():
    """`spacr.gui` was deleted with the Tk interface; nothing may import it."""
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "spacr.gui" not in source
    assert "gui_app" not in source
