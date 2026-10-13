"""Quieting launch noise must never cost a warning that mattered.

``spacr.qt`` installs a Qt message handler and a warnings filter at launch.
Both are deliberately narrow, and both have to survive the environments they
run in: PySide6 may be absent (a headless ``spacr-run`` importing the
package for its constants), and the logging subsystem may be unusable at the
moment a Qt warning arrives. In every one of those cases the requirement is
the same -- the line still reaches stderr, and installing the quieter never
raises.

The warnings filter carries a second requirement: it is re-asserted on every
launch, and ``warnings.filterwarnings`` prepends unconditionally, so a
second call must find its own rule and do nothing rather than grow
``warnings.filters`` without bound.
"""

from __future__ import annotations

import builtins
import logging
import warnings

import pytest

pytest.importorskip("PySide6")

import spacr.qt as qtpkg                      # noqa: E402


_NOISE = ("OpenType support missing for script 7, "
          "type 'otf' is not supported")
_INOTIFY = ("inotify_add_watch(\"/home/u/data\") failed: "
            "\"No space left on device\"")
_THREAD = "QBasicTimer::start: Timers cannot be started from another thread"


@pytest.fixture
def installed_handler(monkeypatch):
    """Capture the handler ``_install_quiet_qt_logging`` would install."""
    import PySide6.QtCore as qtcore

    captured = []
    monkeypatch.setattr(qtcore, "qInstallMessageHandler", captured.append)
    qtpkg._install_quiet_qt_logging()
    assert len(captured) == 1, "the quieter installed no handler"
    return captured[0]


def test_without_pyside6_the_quieter_installs_nothing_and_does_not_raise(
        monkeypatch):
    """Importing ``spacr.qt`` on a machine with no Qt must stay harmless."""
    real_import = builtins.__import__
    blocked = []

    def no_qtcore(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "PySide6.QtCore" and "qInstallMessageHandler" in (
                fromlist or ()):
            blocked.append(name)
            raise ImportError("PySide6 is not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", no_qtcore)

    assert qtpkg._install_quiet_qt_logging() is None
    assert blocked == ["PySide6.QtCore"]


def test_known_launch_noise_is_dropped_and_everything_else_is_printed(
        installed_handler, capsys):
    """The filter is a filter: an unrecognised Qt warning still shows."""
    from PySide6.QtCore import QtMsgType

    installed_handler(QtMsgType.QtWarningMsg, None, _NOISE)
    assert capsys.readouterr().err == ""

    installed_handler(QtMsgType.QtWarningMsg, None, "QPixmap: invalid pixmap")
    assert "QPixmap: invalid pixmap" in capsys.readouterr().err


def test_the_inotify_line_is_explained_once_and_not_repeated(
        installed_handler, capsys, monkeypatch):
    """The watch-limit line is somebody else's problem, said plainly."""
    from PySide6.QtCore import QtMsgType

    monkeypatch.setattr(qtpkg, "_SAID_IT_ONCE", False)
    installed_handler(QtMsgType.QtWarningMsg, None, _INOTIFY)
    first = capsys.readouterr().err
    assert "inotify FILE WATCHES" in first
    assert "No space left on device" not in first

    installed_handler(QtMsgType.QtWarningMsg, None, _INOTIFY)
    assert capsys.readouterr().err == ""


def test_a_broken_logger_does_not_swallow_the_qt_warning(
        installed_handler, capsys, monkeypatch):
    """Logging a warning must never become a second, louder failure."""
    from PySide6.QtCore import QtMsgType

    def unusable(*_args, **_kwargs):
        raise RuntimeError("logging is shut down")

    logger = logging.getLogger("spacr.qt")
    monkeypatch.setattr(logger, "warning", unusable)
    monkeypatch.setattr(logger, "log", unusable)

    installed_handler(QtMsgType.QtWarningMsg, None, _THREAD)

    assert _THREAD in capsys.readouterr().err


def test_re_asserting_the_library_filter_does_not_grow_the_filter_list(
        monkeypatch):
    """Called once per launch, so a second call has to be a no-op."""
    monkeypatch.setattr(warnings, "filters", [])

    qtpkg._quiet_library_warnings()
    after_first = list(warnings.filters)
    assert len(after_first) == len(qtpkg._LIBRARY_NOISE)

    qtpkg._quiet_library_warnings()

    assert warnings.filters == after_first
