"""Quitting spaCR, and quitting one run, when asking nicely is not enough.

The state this exists for is a real one and it happened on the reporter's
machine while this was being written: a spaCR they could not kill, which in
the end took a SIGTERM from a terminal the desktop entry never opens.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QPushButton


@pytest.fixture()
def private_store(monkeypatch):
    """Never let a preferences test touch the real ~/.config/spacr."""
    from spacr.qt import preferences

    path = os.path.join(tempfile.mkdtemp(prefix="spacr-quit-"), "user.ini")
    monkeypatch.setattr(
        preferences, "_settings",
        lambda: QSettings(path, QSettings.IniFormat))
    return path


# ---------------------------------------------------------------------------
# 1. The controls exist where they were asked for
# ---------------------------------------------------------------------------

def test_preferences_has_a_quit_button_beside_the_resource_buttons(
        private_store, qtbot):
    """Same tab as Clear RAM / VRAM / CPU.

    It belongs with them and not with Save/Cancel: it is the last of the
    "this machine is not behaving" tools, the one to reach for when
    freeing memory was not enough.
    """
    from spacr.qt.preferences import PreferencesDialog

    dialog = PreferencesDialog()
    qtbot.addWidget(dialog)

    names = {b.objectName() for b in dialog.findChildren(QPushButton)}
    assert "QuitSpacrButton" in names, "the Quit spaCR button is missing"
    for sibling in ("ClearRamButton", "ClearVramButton", "ClearCpuButton"):
        assert sibling in names, f"{sibling} moved; check they share a tab"


def test_the_quit_button_keeps_its_own_name_and_is_red(private_store, qtbot):
    """`style_as_danger` must not take the widget's identity.

    It used to call `setObjectName("DangerButton")`, which renamed the
    button it was colouring -- so every lookup for `QuitSpacrButton` found
    nothing, including the one in the test above.
    """
    from spacr.qt.preferences import PreferencesDialog

    dialog = PreferencesDialog()
    qtbot.addWidget(dialog)
    button = dialog.findChild(QPushButton, "QuitSpacrButton")

    assert button is not None
    assert button.objectName() == "QuitSpacrButton"
    assert "QuitSpacrButton" in button.styleSheet(), (
        "the danger rule is not keyed on this button's own name")
    assert button.property("spacrDanger") is True


# ---------------------------------------------------------------------------
# 2. The five-minute re-prompt
# ---------------------------------------------------------------------------

def test_the_watcher_asks_again_every_five_minutes(qtbot):
    from spacr.qt.shutdown import RECHECK_MS

    assert RECHECK_MS == 5 * 60 * 1000


def test_the_watcher_stops_asking_once_the_work_is_done(qtbot):
    """No prompt for a job that finished while the user was reading."""
    from spacr.qt.shutdown import GracefulQuitWatcher

    asked = []
    running = [True]
    watcher = GracefulQuitWatcher(
        None, lambda: running[0], what="spaCR",
        on_force=lambda: asked.append("forced"), interval_ms=10)
    watcher.start()
    running[0] = False
    qtbot.wait(60)
    assert not asked


def test_a_watcher_started_when_nothing_runs_never_arms(qtbot):
    """Choosing "finish current work" with nothing to finish must be quiet."""
    from spacr.qt.shutdown import GracefulQuitWatcher

    watcher = GracefulQuitWatcher(
        None, lambda: False, what="spaCR", interval_ms=10)
    watcher.start()
    assert not watcher._timer.isActive()


def test_the_prompt_does_not_stack_while_it_is_open(qtbot, monkeypatch):
    """The prompt runs a nested event loop and the timer keeps firing.

    Without the guard the user gets one more dialog every five minutes
    they spend reading the first one, each behind the last.
    """
    from spacr.qt import shutdown

    opened = []

    class _Button:
        def setObjectName(self, *_a):
            pass

    class _Box:
        # The class attributes the real QMessageBox is asked for before it
        # is ever constructed (`QMessageBox.Warning`, the button roles).
        Warning = 0
        DestructiveRole = 0
        RejectRole = 1

        def __init__(self, *a, **k):
            opened.append(1)

        def __getattr__(self, name):
            return lambda *a, **k: None

        def exec(self):
            # Re-enter exactly as a nested event loop would.
            watcher._recheck()

        def clickedButton(self):
            return None

        def addButton(self, *a, **k):
            return _Button()

    monkeypatch.setattr(shutdown, "QMessageBox", _Box)
    watcher = shutdown.GracefulQuitWatcher(
        None, lambda: True, what="spaCR", interval_ms=10)
    watcher._recheck()
    assert len(opened) == 1, f"the prompt stacked {len(opened)} deep"


# ---------------------------------------------------------------------------
# 3. Force means force
# ---------------------------------------------------------------------------

def test_force_quit_uses_os_exit(monkeypatch):
    """Not `sys.exit`, not `QApplication.quit`.

    Both of those unwind -- atexit handlers, Python finalisation, Qt's own
    teardown -- and every one of those can block on the very thread that is
    already wedged. A force quit that can hang is not a force quit.
    """
    from spacr.qt import shutdown

    called = {}
    monkeypatch.setattr(shutdown.os, "_exit",
                        lambda code: called.setdefault("code", code))
    shutdown.force_quit_now(3)
    assert called["code"] == 3


def test_describe_active_names_the_jobs(qtbot):
    """"Something is still running" is not a thing anybody can decide with."""
    from spacr.qt.shutdown import describe_active

    class _Handle:
        app_key = "mask"

        def elapsed(self):
            return 125.0

    text = describe_active([_Handle()])
    assert "mask" in text and "2 min" in text


def test_describe_active_survives_a_handle_mid_retirement(qtbot):
    """Handles retire on their own thread's schedule, not on the prompt's."""
    from spacr.qt.shutdown import describe_active

    class _Broken:
        app_key = "measure"

        def elapsed(self):
            raise RuntimeError("already retired")

    assert "measure" in describe_active([_Broken()])
