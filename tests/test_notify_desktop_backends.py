"""Table-driven tests of the desktop notifications in ``spacr.qt.notify``.

One general test per unit: which command each platform's backend runs (the
subprocess call is intercepted, so a test run never puts a toast on the
developer's desktop), when the in-app tray fallback shows a message, and what
the finished-pipeline wrapper announces and through which channel.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spacr.qt import notify as notify_module  # noqa: E402


@pytest.fixture
def recorded_runs(monkeypatch):
    """Capture ``subprocess.run`` calls, run inline, instead of launching a notifier."""
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))

    monkeypatch.setattr(notify_module.subprocess, "run", fake_run)
    monkeypatch.setattr(notify_module, "_in_background",
                        lambda fn, *a, **k: fn(*a, **k))
    return calls


def _toast_module(shown):
    """A stand-in ``win10toast`` whose notifier records what it was asked to show."""
    class ToastNotifier:
        def show_toast(self, title, body, duration=None, threaded=None):
            shown.append({"title": title, "body": body,
                          "duration": duration, "threaded": threaded})

    module = types.ModuleType("win10toast")
    module.ToastNotifier = ToastNotifier
    return module


def test_notify_runs_exactly_the_backend_of_each_platform(subtests, monkeypatch,
                                                          recorded_runs, caplog):
    """Each platform runs its own notifier with exact arguments, or declines with False."""
    def which_raises(name):
        raise OSError("PATH lookup failed")

    darwin_plain = ('display notification "in 42s" with title "spaCR" '
                    'subtitle "Masks finished"')
    darwin_quoted = ('display notification "plate \\"A\\" done" with title '
                     '"spaCR" subtitle "he said \\"go\\""')
    cases = [
        # (id, system, which, win10toast, args, kwargs, result, runs, toasts, log)
        ("linux with libnotify sends sender, headline, body", "Linux",
         lambda name: "/usr/bin/notify-send" if name == "notify-send" else None,
         None, ("Done", "12 plates"), {"app_name": "spaCR"}, True,
         [(["notify-send", "-a", "spaCR", "Done", "12 plates"],
           {"check": False, "timeout": 3})], [], None),
        ("linux passes a custom sender and an empty body", "Linux",
         lambda name: "/usr/bin/notify-send", None, ("Done",),
         {"app_name": "plate run"}, True,
         [(["notify-send", "-a", "plate run", "Done", ""],
           {"check": False, "timeout": 3})], [], None),
        ("linux without notify-send declines", "Linux", lambda name: None,
         None, ("Done",), {}, False, [], [], None),
        ("macos builds the applescript", "Darwin", lambda name: None, None,
         ("Masks finished", "in 42s"), {"app_name": "spaCR"}, True,
         [(["osascript", "-e", darwin_plain], {"check": False, "timeout": 3})],
         [], None),
        ("macos escapes every double quote in title and body", "Darwin",
         lambda name: None, None, ('he said "go"', 'plate "A" done'), {}, True,
         [(["osascript", "-e", darwin_quoted],
           {"check": False, "timeout": 3})], [], None),
        ("windows shows a threaded six-second toast", "Windows",
         lambda name: None, "installed", ("Run finished", "ok"), {}, True,
         [], [{"title": "Run finished", "body": "ok", "duration": 6,
               "threaded": True}], None),
        ("windows without win10toast declines", "Windows", lambda name: None,
         "missing", ("Run finished",), {}, False, [], [], None),
        ("an unknown platform has no backend", "Haiku", lambda name: None,
         None, ("Done",), {}, False, [], [], None),
        ("a backend lookup that raises is swallowed and logged", "Linux",
         which_raises, None, ("Done",), {}, False, [], [],
         "notify failed: PATH lookup failed"),
    ]
    for (case_id, system, which, toast, args, kwargs, result, runs, toasts,
         log) in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as m:
            recorded_runs.clear()
            caplog.clear()
            shown = []
            m.setattr(notify_module.platform, "system", lambda s=system: s)
            m.setattr(notify_module.shutil, "which", which)
            if toast == "installed":
                m.setitem(sys.modules, "win10toast", _toast_module(shown))
            elif toast == "missing":
                m.setitem(sys.modules, "win10toast", None)
            with caplog.at_level("DEBUG", logger="spacr.qt.notify"):
                assert notify_module.notify(*args, **kwargs) is result
            assert recorded_runs == runs
            assert shown == toasts
            messages = [r.getMessage() for r in caplog.records
                        if r.name == "spacr.qt.notify"]
            assert messages == ([log] if log else [])


def test_the_tray_fallback_shows_a_message_only_when_a_tray_exists(
        subtests, qapp, monkeypatch, caplog):
    """No application, no tray or a broken tray all decline; a tray gets the exact message."""
    from PySide6.QtWidgets import QApplication, QSystemTrayIcon

    def explode():
        raise RuntimeError("tray subsystem gone")

    cases = [
        # (id, instance, available, result, messages, log)
        ("headless callers get False", lambda: None, lambda: True, False, [],
         None),
        ("a desktop without a tray declines", None, lambda: False, False, [],
         None),
        ("a tray shows the title and body for six seconds", None,
         lambda: True, True,
         [("Masks finished", "in 42s", QSystemTrayIcon.Information, 6000)],
         None),
        ("a broken tray is logged, not raised", None, explode, False, [],
         "tray notify failed: tray subsystem gone"),
    ]
    for case_id, instance, available, result, expected, log in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as m:
            caplog.clear()
            messages = []
            if instance is not None:
                m.setattr(QApplication, "instance", staticmethod(instance))
            m.setattr(QSystemTrayIcon, "isSystemTrayAvailable",
                      staticmethod(available))
            m.setattr(QSystemTrayIcon, "showMessage",
                      lambda self, title, body, icon, msecs: messages.append(
                          (title, body, icon, msecs)))
            with caplog.at_level("DEBUG", logger="spacr.qt.notify"):
                assert notify_module.notify_tray(
                    "Masks finished", "in 42s") is result
            assert messages == expected
            logged = [r.getMessage() for r in caplog.records
                      if r.name == "spacr.qt.notify"]
            assert logged == ([log] if log else [])


def test_a_finished_pipeline_is_announced_once_with_name_status_and_time(
        subtests, monkeypatch, caplog):
    """The headline and body are exact, the tray is only the fallback, and sound is asked only when loaded."""
    def sound_module(heard, raises=False):
        module = types.ModuleType("spacr.qt.sound")

        def announce_run_end(status):
            if raises:
                raise RuntimeError("no audio device")
            heard.append(status)

        module.announce_run_end = announce_run_end
        return module

    cases = [
        # (id, app, status, elapsed, os_ok, sound, title, body, tray, heard, log)
        ("success gets the tick and one decimal", "mask", "success", 42.25,
         True, None, "✓ spaCR — mask success", "Finished in 42.2s.", [], [],
         None),
        ("failure gets the warning glyph", "measure", "failed", 1.0, True,
         None, "⚠ spaCR — measure failed", "Finished in 1.0s.", [], [], None),
        ("cancelled gets the warning glyph", "umap", "cancelled", 0.04, True,
         None, "⚠ spaCR — umap cancelled", "Finished in 0.0s.", [], [], None),
        ("the tray is used when the os notifier declines", "mask", "success",
         3.0, False, None, "✓ spaCR — mask success", "Finished in 3.0s.",
         [("✓ spaCR — mask success", "Finished in 3.0s.")], [], None),
        ("a loaded sound module hears the status", "mask", "failed", 2.0,
         True, "ok", "⚠ spaCR — mask failed", "Finished in 2.0s.", [],
         ["failed"], None),
        ("a sound that fails is logged and the notice still goes", "mask",
         "success", 2.0, True, "raises", "✓ spaCR — mask success",
         "Finished in 2.0s.", [], [], "run sound failed: no audio device"),
    ]
    for (case_id, app, status, elapsed, os_ok, sound, title, body, tray,
         heard_expected, log) in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as m:
            caplog.clear()
            sent, tray_calls, heard = [], [], []
            m.setattr(notify_module, "notify",
                      lambda t, b: sent.append((t, b)) or os_ok)
            m.setattr(notify_module, "notify_tray",
                      lambda t, b: tray_calls.append((t, b)))
            m.setitem(sys.modules, "spacr.qt.sound",
                      None if sound is None
                      else sound_module(heard, raises=sound == "raises"))
            with caplog.at_level("DEBUG", logger="spacr.qt.notify"):
                notify_module.announce_pipeline_finished(app, status, elapsed)
            assert sent == [(title, body)]
            assert tray_calls == tray
            assert heard == heard_expected
            logged = [r.getMessage() for r in caplog.records
                      if r.name == "spacr.qt.notify"]
            assert logged == ([log] if log else [])
