"""The disk readout reads the disk on a worker, never on the GUI thread.

THE FREEZE, 2026-09-04. Preferences > Performance > "Check disk space" was
wired straight to `run_resource_action("disk")`, which called
`resource_cleanup.disk_report()` inline in the button's `clicked` slot. That
went:

    run_resource_action("disk")
      -> resource_cleanup.disk_report()
        -> project_paths()  -> os.path.isdir(<every remembered source folder>)
        -> os.stat(path)    for each of them
        -> shutil.disk_usage(path)

and the remembered source folders are paths the USER chose. One of the
maintainer's was under ``/nas_mnt``, an ``autofs`` mount whose share was
asleep, and a single stat on it had NOT RETURNED AFTER TWENTY SECONDS --
the stat is what triggers the automount. The whole interface was frozen
between the confirmation box and the result box, and it left no traceback,
because a stalled event loop is not a crash.

`path_probe` is not the fix for this one: a disk readout needs real device
ids and real byte counts, not a cached yes/no. Work that must genuinely
touch the disk belongs on a worker thread, which is what these tests pin.
"""
from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("PySide6")

from spacr.qt import resource_cleanup as rc

#: Longer than any human would call responsive, far shorter than the twenty
#: seconds actually measured. A test that waited the real duration would be
#: a test nobody runs.
SLOW_S = 6.0


@pytest.fixture
def sleeping_mount(monkeypatch):
    """Reading the disk takes :data:`SLOW_S`, as a sleeping automount does.

    Patched at `resource_cleanup.disk_report` rather than at `os.stat`, and
    that is not laziness: `os.stat` is what Python itself uses to import a
    module and open a file, so slowing it process-wide slows the test runner
    instead of the code under test. The whole of the wait is inside
    `disk_report` -- `project_paths`'s `os.path.isdir`, then `os.stat` and
    `shutil.disk_usage` on every folder it returns -- so making that call
    slow reproduces the freeze exactly, and the worker really does sit in it.
    """
    def slow_report(*_args, **_kwargs):
        time.sleep(SLOW_S)
        return rc.DiskReport((), "")

    monkeypatch.setattr(rc, "disk_report", slow_report)
    return slow_report


def test_pressing_check_disk_returns_at_once(qtbot, monkeypatch,
                                             sleeping_mount):
    """The property that matters: the click handler does not wait.

    Without the fix this takes SLOW_S seconds per remembered folder, on the
    GUI thread, with the confirmation box already dismissed and the result
    box not yet built -- an application that is drawing nothing at all.
    """
    from spacr.qt import preferences as prefs

    monkeypatch.setattr(prefs, "confirm_resource_action", lambda *a, **k: True)
    monkeypatch.setattr(prefs, "_show_resource_result", lambda *a, **k: None)

    started = time.monotonic()
    prefs.run_resource_action("disk")
    elapsed = time.monotonic() - started

    assert elapsed < 0.5, (
        f"the disk button held the GUI thread for {elapsed:.1f} s -- "
        "disk_report is being called inline again")


def test_the_disk_is_read_on_a_worker_thread(qtbot, monkeypatch):
    """Where the stat happens, stated as a fact rather than inferred from a clock.

    A timing assertion alone would still pass if somebody made `disk_report`
    fast; this one fails the moment it goes back onto the GUI thread.
    """
    from spacr.qt import preferences as prefs

    where = {}
    monkeypatch.setattr(prefs, "confirm_resource_action", lambda *a, **k: True)
    monkeypatch.setattr(prefs, "_show_resource_result", lambda *a, **k: None)
    monkeypatch.setattr(
        rc, "disk_report",
        lambda *a, **k: (where.setdefault("thread",
                                          threading.current_thread()),
                         rc.DiskReport((), ""))[1])

    prefs.run_resource_action("disk")
    qtbot.waitUntil(lambda: "thread" in where, timeout=5000)

    assert where["thread"] is not threading.main_thread(), (
        "disk_report ran on the GUI thread")


def test_the_report_still_reaches_the_user(qtbot, monkeypatch):
    """Nothing is dropped by moving it: the same report, a moment later."""
    from spacr.qt import preferences as prefs

    report = rc.DiskReport((rc.DiskEntry("/data", 100, 40, 60),), "")
    shown = []
    monkeypatch.setattr(prefs, "confirm_resource_action", lambda *a, **k: True)
    monkeypatch.setattr(rc, "disk_report", lambda *a, **k: report)
    monkeypatch.setattr(
        prefs, "_show_resource_result",
        lambda action, result, parent=None: shown.append((action, result)))

    assert prefs.run_resource_action("disk") is None, (
        "the report arrives in the callback, not the return value")
    qtbot.waitUntil(lambda: bool(shown), timeout=5000)
    assert shown == [("disk", report)]


def test_declining_still_reads_nothing(qtbot, monkeypatch):
    """The confirmation is still asked BEFORE any work, not after."""
    from spacr.qt import preferences as prefs

    ran = []
    monkeypatch.setattr(prefs, "confirm_resource_action", lambda *a, **k: False)
    monkeypatch.setattr(rc, "disk_report",
                        lambda *a, **k: ran.append("read"))

    assert prefs.run_resource_action("disk") is None
    assert ran == []


def test_the_button_click_itself_does_not_freeze_preferences(qtbot, monkeypatch,
                                                             sleeping_mount,
                                                             qt_theme_applied):
    """The real widget, pressed the way a user presses it.

    The regression this pins is not "the dialog is fast" -- it is that the
    dialog does not touch the filesystem while the user is waiting on it.
    """
    from PySide6.QtWidgets import QPushButton
    from spacr.qt import preferences as prefs
    from spacr.qt.preferences import PreferencesDialog

    monkeypatch.setattr(prefs, "confirm_resource_action", lambda *a, **k: True)
    monkeypatch.setattr(prefs, "_show_resource_result", lambda *a, **k: None)

    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    button = dlg.findChild(QPushButton, "CheckDiskButton")
    assert button is not None, "the Performance tab lost its disk button"

    started = time.monotonic()
    button.click()
    elapsed = time.monotonic() - started

    assert elapsed < 0.5, (
        f"clicking Check disk space blocked the dialog for {elapsed:.1f} s")


def test_the_button_says_why_it_is_unavailable_while_it_reads(qtbot,
                                                              monkeypatch,
                                                              sleeping_mount,
                                                              qt_theme_applied):
    """Disabled and SAYING WHY, never inert -- and given back afterwards.

    Before the fix the button was unpressable during the report because the
    whole application was; keeping it unpressable while the worker reads is
    the same affordance without the freeze.
    """
    from PySide6.QtWidgets import QPushButton
    from spacr.qt import preferences as prefs
    from spacr.qt.preferences import PreferencesDialog

    monkeypatch.setattr(prefs, "confirm_resource_action", lambda *a, **k: True)
    monkeypatch.setattr(prefs, "_show_resource_result", lambda *a, **k: None)

    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    button = dlg.findChild(QPushButton, "CheckDiskButton")
    resting_tip = button.toolTip()

    button.click()
    assert not button.isEnabled(), "a second report can be queued behind this one"
    assert button.toolTip() and button.toolTip() != resting_tip, (
        "a disabled button that does not say why is inert")

    qtbot.waitUntil(button.isEnabled, timeout=int(SLOW_S * 3 * 1000))
    assert button.toolTip() == resting_tip, (
        "the button never got its own explanation back")
