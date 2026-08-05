"""`pip install spacr` (no ``[qt]`` extra) then `spacr` must explain itself.

PySide6 is declared only in the ``qt`` extra (``setup.py``), but the ``spacr``
console script points straight at :func:`spacr.qt.run`, which imported
``spacr.qt.app`` unguarded. On a core-only install that surfaced as a raw
``ModuleNotFoundError: No module named 'PySide6'`` with a traceback through
library internals, and no hint that an extra exists.

The subprocess test at the bottom reproduces that install for real — a fresh
interpreter in which ``PySide6`` genuinely cannot be imported — and drives the
same entry point the console script does.
"""
from __future__ import annotations

import importlib.abc
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import spacr.qt as qt


SPACR_PKG = Path(qt.__file__).resolve().parents[1]
REPO_ROOT = SPACR_PKG.parent


class _RaiseOnImport(importlib.abc.MetaPathFinder):
    """Make one module name fail to import with a chosen exception."""

    def __init__(self, target: str, exc: BaseException):
        self.target = target
        self.exc = exc

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.target:
            raise self.exc
        return None


@pytest.fixture
def break_app_import(monkeypatch):
    """Return a callable that makes ``from .app import launch`` raise ``exc``.

    ``spacr.qt.app`` is dropped from ``sys.modules`` so the import machinery
    actually consults ``sys.meta_path``; monkeypatch restores it afterwards,
    and the real module is never re-executed.
    """

    def _break(exc: BaseException):
        monkeypatch.delitem(sys.modules, "spacr.qt.app", raising=False)
        monkeypatch.setattr(
            sys, "meta_path", [_RaiseOnImport("spacr.qt.app", exc)] + sys.meta_path
        )

    return _break


# --- the guard itself ----------------------------------------------------

def test_missing_pyside6_prints_the_install_command_instead_of_a_traceback(
    break_app_import, capsys
):
    break_app_import(
        ModuleNotFoundError("No module named 'PySide6'", name="PySide6")
    )

    assert qt.run([]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert 'python -m pip install "spacr[qt]"' in captured.err
    assert "PySide6" in captured.err
    assert "Traceback" not in captured.err


def test_the_message_also_offers_the_headless_route(break_app_import, capsys):
    break_app_import(
        ModuleNotFoundError("No module named 'PySide6'", name="PySide6")
    )
    qt.run(["mask"])
    assert "spacr-run --list" in capsys.readouterr().err


def test_a_missing_qt_submodule_is_still_reported_as_the_qt_extra(
    break_app_import, capsys
):
    """`from PySide6.QtCore import ...` sets ``name`` to the *sub*module."""
    break_app_import(
        ModuleNotFoundError("No module named 'PySide6.QtCore'", name="PySide6.QtCore")
    )
    assert qt.run([]) == 1
    assert 'pip install "spacr[qt]"' in capsys.readouterr().err


@pytest.mark.parametrize("module", ["PySide6", "shiboken6", "qtawesome"])
def test_every_qt_extra_distribution_takes_the_friendly_path(
    break_app_import, capsys, module
):
    break_app_import(ModuleNotFoundError(f"No module named {module!r}", name=module))
    assert qt.run([]) == 1
    assert module in capsys.readouterr().err


def test_an_unrelated_import_error_keeps_its_traceback(break_app_import, capsys):
    """A genuine bug inside the GUI package must not be mislabelled."""
    boom = ModuleNotFoundError("No module named 'nonexistent_dep'",
                               name="nonexistent_dep")
    break_app_import(boom)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        qt.run([])

    assert excinfo.value is boom
    assert capsys.readouterr().err == ""


def test_an_import_error_without_a_name_falls_back_to_the_message(
    break_app_import, capsys
):
    """Import hooks and hand-raised ImportErrors leave ``.name`` unset."""
    break_app_import(ImportError("cannot import name 'QtCore' from 'PySide6'"))
    assert qt.run([]) == 1
    assert 'pip install "spacr[qt]"' in capsys.readouterr().err


def test_a_nameless_unrelated_import_error_still_propagates(break_app_import):
    break_app_import(ImportError("circular import in spacr.qt.widgets"))
    with pytest.raises(ImportError, match="circular import"):
        qt.run([])


# --- the classifier, directly -------------------------------------------

@pytest.mark.parametrize(
    "exc, expected",
    [
        (ModuleNotFoundError("x", name="PySide6"), "PySide6"),
        (ModuleNotFoundError("x", name="PySide6.QtWidgets"), "PySide6"),
        (ModuleNotFoundError("x", name="shiboken6"), "shiboken6"),
        (ModuleNotFoundError("x", name="qtawesome"), "qtawesome"),
        (ImportError("no qtawesome here"), "qtawesome"),
        (ImportError("something else entirely"), None),
        (ImportError("boom"), None),
    ],
)
def test_missing_qt_extra_classifies_the_failure(exc, expected):
    assert qt._missing_qt_extra(exc) == expected


def test_missing_qt_extra_ignores_a_none_name():
    exc = ImportError("nothing recognisable")
    assert exc.name is None
    assert qt._missing_qt_extra(exc) is None


# --- the happy path still launches --------------------------------------

def test_a_working_install_still_reaches_launch(monkeypatch):
    """The guard wraps only the import — ``launch`` is called normally."""
    from types import ModuleType

    calls = []
    fake_app = ModuleType("spacr.qt.app")
    fake_app.launch = lambda argv: (calls.append(argv), 7)[1]
    monkeypatch.setitem(sys.modules, "spacr.qt.app", fake_app)

    assert qt.run(["mask"]) == 7
    assert calls == [["mask"]]


def test_an_import_error_raised_during_launch_is_not_swallowed(monkeypatch):
    """A lazy import failing mid-run is a real error, not a missing extra.

    If the ``try`` block had wrapped ``launch(argv)`` too, this would print
    "install spacr[qt]" for a bug that has nothing to do with the extra.
    """
    from types import ModuleType

    def _boom(argv):
        raise ModuleNotFoundError("No module named 'PySide6.QtSvg'",
                                  name="PySide6.QtSvg")

    fake_app = ModuleType("spacr.qt.app")
    fake_app.launch = _boom
    monkeypatch.setitem(sys.modules, "spacr.qt.app", fake_app)

    with pytest.raises(ModuleNotFoundError, match="QtSvg"):
        qt.run([])


# --- end to end, in a process where PySide6 really is absent ------------

_REPRO = textwrap.dedent(
    """
    import importlib.abc, sys

    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            root = fullname.split(".", 1)[0]
            if root in ("PySide6", "shiboken6", "qtawesome"):
                raise ModuleNotFoundError(
                    "No module named %r" % fullname, name=fullname)
            return None

    sys.meta_path.insert(0, Block())

    import spacr.qt
    sys.exit(spacr.qt.run([]))
    """
)


def test_core_only_install_exits_cleanly_with_an_actionable_message(tmp_path):
    script = tmp_path / "core_only_spacr.py"
    script.write_text(_REPRO, encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(tmp_path),
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO_ROOT),
             "HOME": str(tmp_path), "QT_QPA_PLATFORM": "offscreen",
             "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert proc.returncode == 1, proc.stderr
    assert "Traceback" not in proc.stderr
    assert 'python -m pip install "spacr[qt]"' in proc.stderr
    assert "PySide6" in proc.stderr


def test_version_still_answers_without_qt_installed(tmp_path):
    """`spacr --version` must not need the GUI extra at all."""
    script = tmp_path / "core_only_version.py"
    script.write_text(
        _REPRO.replace("spacr.qt.run([])", 'spacr.qt.run(["--version"])'),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(tmp_path),
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO_ROOT),
             "HOME": str(tmp_path), "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=300,
    )

    import spacr.version

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == spacr.version.get_version()
