"""Desktop notification backends: what command each platform actually runs.

``notify`` shells out, so the subprocess call is intercepted and its argument
vector asserted rather than fired -- a test run must not put a toast on the
developer's desktop. Everything else (backend selection, escaping, the
tray fallback, the finished-pipeline wrapper) runs for real.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spacr.qt import notify as notify_module


@pytest.fixture
def recorded_runs(monkeypatch):
    """Capture ``subprocess.run`` calls instead of launching a notifier."""
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return None

    monkeypatch.setattr(notify_module.subprocess, "run", fake_run)
    return calls


def _on(monkeypatch, system, *, which="/usr/bin/notify-send"):
    monkeypatch.setattr(notify_module.platform, "system", lambda: system)
    monkeypatch.setattr(notify_module.shutil, "which", lambda name: which)


def test_linux_calls_notify_send_with_the_app_name(monkeypatch, recorded_runs):
    """libnotify gets the sender, the headline and the body, in that order."""
    _on(monkeypatch, "Linux")
    assert notify_module.notify("Done", "12 plates", app_name="spaCR") is True
    argv, kwargs = recorded_runs[0]
    assert argv == ["notify-send", "-a", "spaCR", "Done", "12 plates"]
    assert kwargs["check"] is False
    assert kwargs["timeout"] == 3


def test_linux_without_notify_send_declines(monkeypatch, recorded_runs):
    """No libnotify binary means no backend accepted the request."""
    _on(monkeypatch, "Linux", which=None)
    assert notify_module.notify("Done") is False
    assert recorded_runs == []


def test_macos_builds_an_applescript_notification(monkeypatch, recorded_runs):
    """Darwin runs ``osascript`` with title, subtitle and body filled in."""
    _on(monkeypatch, "Darwin", which=None)
    assert notify_module.notify("Masks finished", "in 42s",
                                app_name="spaCR") is True
    argv, kwargs = recorded_runs[0]
    assert argv[0:2] == ["osascript", "-e"]
    script = argv[2]
    assert 'display notification "in 42s"' in script
    assert 'with title "spaCR"' in script
    assert 'subtitle "Masks finished"' in script
    assert kwargs["timeout"] == 3


def test_a_quote_in_the_body_cannot_break_out_of_the_applescript(
        monkeypatch, recorded_runs):
    """Double quotes are escaped, so a run name cannot end the string early."""
    _on(monkeypatch, "Darwin", which=None)
    assert notify_module.notify('he said "go"', 'plate "A" done') is True
    script = recorded_runs[0][0][2]
    assert '\\"go\\"' in script
    assert '\\"A\\"' in script
    # Every literal quote in the payload is escaped: only the three pairs
    # AppleScript itself needs are left unescaped.
    assert script.count('"') - script.count('\\"') == 6


def test_esc_tolerates_none():
    """``_esc(None)`` is an empty string, not a crash on ``.replace``."""
    assert notify_module._esc(None) == ""
    assert notify_module._esc("") == ""
    assert notify_module._esc('a"b') == 'a\\"b'


def test_windows_uses_the_toast_notifier(monkeypatch, recorded_runs):
    """A Windows box with win10toast installed shows a threaded toast."""
    import sys
    import types

    shown = {}

    class ToastNotifier:
        def show_toast(self, title, body, duration=None, threaded=None):
            shown.update(title=title, body=body, duration=duration,
                         threaded=threaded)

    module = types.ModuleType("win10toast")
    module.ToastNotifier = ToastNotifier
    monkeypatch.setitem(sys.modules, "win10toast", module)
    _on(monkeypatch, "Windows", which=None)

    assert notify_module.notify("Run finished", "ok") is True
    assert shown == {"title": "Run finished", "body": "ok",
                     "duration": 6, "threaded": True}
    assert recorded_runs == []


def test_windows_without_win10toast_declines(monkeypatch, recorded_runs):
    """Without the optional dependency Windows has no backend at all."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "win10toast":
            raise ImportError("no module named win10toast")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    _on(monkeypatch, "Windows", which=None)
    assert notify_module.notify("Run finished") is False


def test_an_unknown_platform_has_no_backend(monkeypatch, recorded_runs):
    """A system with no shipped backend returns False without running anything."""
    _on(monkeypatch, "Haiku", which=None)
    assert notify_module.notify("Done") is False
    assert recorded_runs == []


def test_a_failing_backend_is_swallowed(monkeypatch, caplog):
    """A notifier that raises never propagates into the pipeline that called it."""
    monkeypatch.setattr(notify_module.platform, "system", lambda: "Linux")

    def explode(name):
        raise OSError("PATH lookup failed")

    monkeypatch.setattr(notify_module.shutil, "which", explode)
    with caplog.at_level("DEBUG", logger="spacr.qt.notify"):
        assert notify_module.notify("Done") is False
    assert any("notify failed" in record.message for record in caplog.records)


# --------------------------------------------------------------------------
# tray fallback
# --------------------------------------------------------------------------


def test_the_tray_fallback_is_a_no_op_without_a_qapplication(monkeypatch):
    """Headless callers get False rather than a Qt error."""
    from PySide6.QtWidgets import QApplication

    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: None))
    assert notify_module.notify_tray("Done", "ok") is False


def test_the_tray_fallback_declines_when_no_tray_exists(qapp, monkeypatch):
    """A desktop with no system tray is not an error either."""
    from PySide6.QtWidgets import QSystemTrayIcon

    monkeypatch.setattr(QSystemTrayIcon, "isSystemTrayAvailable",
                        staticmethod(lambda: False))
    assert notify_module.notify_tray("Done", "ok") is False


def test_the_tray_fallback_shows_a_message_when_a_tray_exists(qapp,
                                                              monkeypatch):
    """With a tray available the badge is actually shown, with the given text."""
    from PySide6.QtWidgets import QSystemTrayIcon

    monkeypatch.setattr(QSystemTrayIcon, "isSystemTrayAvailable",
                        staticmethod(lambda: True))
    messages = []
    monkeypatch.setattr(
        QSystemTrayIcon, "showMessage",
        lambda self, title, body, icon, msecs: messages.append(
            (title, body, icon, msecs)))

    assert notify_module.notify_tray("Masks finished", "in 42s") is True
    assert messages == [("Masks finished", "in 42s",
                         QSystemTrayIcon.Information, 6000)]


def test_a_broken_tray_is_logged_not_raised(qapp, monkeypatch, caplog):
    """An exception from Qt's tray API is swallowed like every other backend."""
    from PySide6.QtWidgets import QSystemTrayIcon

    def explode():
        raise RuntimeError("tray subsystem gone")

    monkeypatch.setattr(QSystemTrayIcon, "isSystemTrayAvailable",
                        staticmethod(explode))
    with caplog.at_level("DEBUG", logger="spacr.qt.notify"):
        assert notify_module.notify_tray("Done") is False
    assert any("tray notify failed" in record.message
               for record in caplog.records)


# --------------------------------------------------------------------------
# announce_pipeline_finished
# --------------------------------------------------------------------------


def test_a_finished_pipeline_announces_its_name_status_and_duration(
        monkeypatch):
    """The headline names the app and outcome; the body carries the elapsed time."""
    seen = {}
    monkeypatch.setattr(notify_module, "notify",
                        lambda title, body: seen.update(
                            title=title, body=body) or True)
    notify_module.announce_pipeline_finished("mask", "success", 42.25)
    assert seen["title"] == "✓ spaCR — mask success"
    assert seen["body"] == "Finished in 42.2s."


def test_a_failed_pipeline_gets_the_warning_glyph(monkeypatch):
    """Anything but success is announced with the warning mark."""
    titles = []
    monkeypatch.setattr(notify_module, "notify",
                        lambda title, body: titles.append(title) or True)
    notify_module.announce_pipeline_finished("measure", "failed", 1.0)
    notify_module.announce_pipeline_finished("umap", "cancelled", 1.0)
    assert titles == ["⚠ spaCR — measure failed", "⚠ spaCR — umap cancelled"]


def test_the_tray_is_used_only_when_the_os_notifier_declines(monkeypatch):
    """The in-app badge is the fallback, not a second notification."""
    tray_calls = []
    monkeypatch.setattr(notify_module, "notify", lambda title, body: False)
    monkeypatch.setattr(notify_module, "notify_tray",
                        lambda title, body: tray_calls.append((title, body)))
    notify_module.announce_pipeline_finished("mask", "success", 3.0)
    assert tray_calls == [("✓ spaCR — mask success", "Finished in 3.0s.")]

    tray_calls.clear()
    monkeypatch.setattr(notify_module, "notify", lambda title, body: True)
    notify_module.announce_pipeline_finished("mask", "success", 3.0)
    assert tray_calls == []
