"""What ``spacr.qt`` decides about the platform and its log lines at launch.

Three launch-time helpers, each of which must never cost the user something
they chose or needed:

* ``_prefer_a_context_the_shaders_can_run_on`` asks for ``xcb`` on a Wayland
  session that has an X server (vispy's GLSL 120 shaders fail on the GL ES
  context native Wayland hands out), and otherwise leaves ``QT_QPA_PLATFORM``
  exactly as it was -- above all an explicit choice such as the ``offscreen``
  the test suite runs under;
* ``_install_quiet_qt_logging`` drops known launch noise, explains the
  inotify line once, and prints everything else -- even with PySide6 absent
  or the logging subsystem broken;
* ``verbose_logger._ensure_file_handler`` keeps exactly one rotating file
  handler, on the ``spacr`` sink and on no child logger.
"""
from __future__ import annotations

import builtins
import logging
import os

import pytest

import spacr.qt as qtpkg

_PLATFORM_VARIABLES = ("QT_QPA_PLATFORM", "WAYLAND_DISPLAY", "DISPLAY",
                       "XDG_SESSION_TYPE")


def test_the_platform_is_xcb_only_for_an_unchosen_wayland_session_with_x(
        subtests, monkeypatch):
    """Every session shape maps to exactly one QT_QPA_PLATFORM outcome."""
    cases = [
        ("wayland display with an X server", {"WAYLAND_DISPLAY": "wayland-0",
                                               "DISPLAY": ":0"}, "xcb"),
        ("session type alone says wayland", {"XDG_SESSION_TYPE": "wayland",
                                             "DISPLAY": ":0"}, "xcb"),
        ("session type in another case", {"XDG_SESSION_TYPE": "Wayland",
                                          "DISPLAY": ":0"}, "xcb"),
        ("an explicit offscreen is never overruled",
         {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0",
          "QT_QPA_PLATFORM": "offscreen"}, "offscreen"),
        ("a deliberate wayland is kept",
         {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0",
          "QT_QPA_PLATFORM": "wayland"}, "wayland"),
        ("an empty platform string is not a choice",
         {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0",
          "QT_QPA_PLATFORM": ""}, "xcb"),
        ("wayland with no X server to fall back to",
         {"WAYLAND_DISPLAY": "wayland-0"}, None),
        ("an x11 session type with no wayland display",
         {"XDG_SESSION_TYPE": "x11", "DISPLAY": ":0"}, None),
        ("a plain X11 session", {"DISPLAY": ":0"}, None),
        ("neither: a container or a cron job", {}, None),
    ]
    for case_id, env, expected in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as patch:
            for name in _PLATFORM_VARIABLES:
                patch.delenv(name, raising=False)
            for name, value in env.items():
                patch.setenv(name, value)
            assert qtpkg._prefer_a_context_the_shaders_can_run_on() is None
            assert os.environ.get("QT_QPA_PLATFORM") == expected
            others = {name: os.environ.get(name)
                      for name in _PLATFORM_VARIABLES[1:]}
            assert others == {name: env.get(name)
                              for name in _PLATFORM_VARIABLES[1:]}


_NOISE = ("OpenType support missing for script 7, "
          "type 'otf' is not supported")
_INOTIFY = ("inotify_add_watch(\"/home/u/data\") failed: "
            "\"No space left on device\"")
_THREAD = "QBasicTimer::start: Timers cannot be started from another thread"


def test_the_qt_message_handler_filters_explains_and_passes_through(
        subtests, monkeypatch, capsys, caplog):
    """Noise is dropped, inotify is explained once, the rest reaches stderr."""
    import PySide6.QtCore as qtcore
    from PySide6.QtCore import QtMsgType

    with subtests.test(msg="without PySide6 nothing is installed"), \
            monkeypatch.context() as patch:
        real_import = builtins.__import__
        blocked = []

        def no_qtcore(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "PySide6.QtCore" and "qInstallMessageHandler" in (
                    fromlist or ()):
                blocked.append(name)
                raise ImportError("PySide6 is not installed")
            return real_import(name, globals, locals, fromlist, level)

        installed = []
        patch.setattr(qtcore, "qInstallMessageHandler", installed.append)
        patch.setattr(builtins, "__import__", no_qtcore)
        assert qtpkg._install_quiet_qt_logging() is None
        assert blocked == ["PySide6.QtCore"]
        assert installed == []

    captured = []
    monkeypatch.setattr(qtcore, "qInstallMessageHandler", captured.append)
    assert qtpkg._install_quiet_qt_logging() is None
    assert len(captured) == 1
    handler = captured[0]
    capsys.readouterr()

    noise = [_NOISE, "This plugin does not support propagateSizeHints()",
             "This plugin does not support raise()",
             "Using Qt multimedia with FFmpeg version 6.1"]
    for message in noise:
        with subtests.test(msg=f"noise is dropped: {message[:30]}"):
            caplog.clear()
            handler(QtMsgType.QtWarningMsg, None, message)
            assert capsys.readouterr().err == ""
            assert caplog.records == []

    printed = [
        (QtMsgType.QtWarningMsg, "Qt warning", logging.WARNING),
        (QtMsgType.QtCriticalMsg, "Qt critical", logging.ERROR),
        (QtMsgType.QtInfoMsg, "Qt info", logging.INFO),
        (QtMsgType.QtDebugMsg, "Qt debug", logging.DEBUG),
    ]
    for mode, label, level in printed:
        with subtests.test(msg=f"{label} is printed and logged"):
            caplog.clear()
            with caplog.at_level(logging.DEBUG, logger="spacr.qt"):
                handler(mode, None, "QPixmap: invalid pixmap")
            assert capsys.readouterr().err == (
                f"{label}: QPixmap: invalid pixmap\n")
            assert [(r.name, r.levelno, r.getMessage())
                    for r in caplog.records] == [
                ("spacr.qt", level, f"{label}: QPixmap: invalid pixmap")]

    with subtests.test(msg="a cross-thread warning also logs the stack"):
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="spacr.qt"):
            handler(QtMsgType.QtWarningMsg, None, _THREAD)
        assert capsys.readouterr().err == f"Qt warning: {_THREAD}\n"
        messages = [r.getMessage() for r in caplog.records]
        assert len(messages) == 2
        assert messages[0].startswith(
            "The Python stack at that warning (thread 'MainThread'):\n")
        assert messages[1] == f"Qt warning: {_THREAD}"

    with subtests.test(msg="a broken logger does not swallow the warning"), \
            monkeypatch.context() as patch:
        def unusable(*_args, **_kwargs):
            raise RuntimeError("logging is shut down")

        logger = logging.getLogger("spacr.qt")
        patch.setattr(logger, "warning", unusable)
        patch.setattr(logger, "log", unusable)
        handler(QtMsgType.QtWarningMsg, None, _THREAD)
        assert capsys.readouterr().err == f"Qt warning: {_THREAD}\n"

    with subtests.test(msg="the inotify line is explained once"), \
            monkeypatch.context() as patch:
        patch.setattr(qtpkg.sys, "platform", "linux")
        patch.setattr(qtpkg, "_SAID_IT_ONCE", False)
        handler(QtMsgType.QtWarningMsg, None, _INOTIFY)
        first = capsys.readouterr().err
        assert "inotify FILE WATCHES" in first
        assert "sudo sysctl -w" in first
        assert "No space left on device" not in first
        assert qtpkg._SAID_IT_ONCE is True
        handler(QtMsgType.QtWarningMsg, None, _INOTIFY)
        assert capsys.readouterr().err == ""

    with subtests.test(msg="the inotify line is silent off Linux"), \
            monkeypatch.context() as patch:
        patch.setattr(qtpkg.sys, "platform", "darwin")
        patch.setattr(qtpkg, "_SAID_IT_ONCE", False)
        handler(QtMsgType.QtWarningMsg, None, _INOTIFY)
        assert capsys.readouterr().err == ""
        assert qtpkg._SAID_IT_ONCE is False


def test_the_log_file_handler_lives_on_the_spacr_sink_only(
        subtests, monkeypatch, tmp_path):
    """One handler, adopted from a child, re-attached, or None when unbuildable."""
    pytest.importorskip("PySide6")
    from logging.handlers import RotatingFileHandler

    from spacr.qt import verbose_logger as vl

    sink = logging.getLogger(vl._SINK_LOGGER)
    child = logging.getLogger("spacr.qt")
    monkeypatch.setattr(vl, "current_log_file", lambda: tmp_path / "x.log")
    monkeypatch.setattr(child, "level", child.level)
    monkeypatch.setattr(sink, "level", sink.level)

    with subtests.test(msg="a handler already on a child moves to the sink"), \
            monkeypatch.context() as patch:
        handler = logging.FileHandler(tmp_path / "x.log")
        made = []
        patch.setattr(vl, "_file_handler", None)
        patch.setattr(vl, "_quicken",
                      lambda built: (made.append(built), built.close(),
                                     handler)[2])
        child.addHandler(handler)
        try:
            assert vl._ensure_file_handler() is handler
            assert vl._file_handler is handler
            assert handler not in child.handlers
            assert handler in sink.handlers
            assert handler.level == logging.INFO
            assert handler.formatter._fmt == (
                "%(asctime)s %(name)s %(levelname)s  %(message)s")
            assert [(type(m), m.baseFilename, m.maxBytes, m.backupCount,
                     m.encoding) for m in made] == [
                (RotatingFileHandler, str(tmp_path / "x.log"),
                 5 * 1024 * 1024, 5, "utf-8")]
        finally:
            sink.removeHandler(handler)
            child.removeHandler(handler)
            handler.close()

    with subtests.test(msg="a cached handler missing from the sink is re-added"), \
            monkeypatch.context() as patch:
        cached = logging.NullHandler()
        patch.setattr(vl, "_file_handler", cached)
        patch.setattr(vl, "_quicken", lambda built: pytest.fail(
            "a second handler was built"))
        try:
            assert vl._ensure_file_handler() is cached
            assert sink.handlers.count(cached) == 1
            assert vl._ensure_file_handler() is cached
            assert sink.handlers.count(cached) == 1
        finally:
            sink.removeHandler(cached)

    with subtests.test(msg="a handler that cannot be built is None"), \
            monkeypatch.context() as patch:
        patch.setattr(vl, "_file_handler", None)

        def refuse(_built):
            _built.close()
            raise OSError("log folder is read-only")

        patch.setattr(vl, "_quicken", refuse)
        before = list(sink.handlers)
        assert vl._ensure_file_handler() is None
        assert vl._file_handler is None
        assert sink.handlers == before
