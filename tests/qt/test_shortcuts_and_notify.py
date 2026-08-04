"""Tests for keyboard shortcuts, OS notify, screen-reader labels."""
from __future__ import annotations

import pytest


def test_shortcuts_spec_covers_every_binding():
    """Every ShortcutSpec must have keys + a label + a category."""
    from spacr.qt.shortcuts import SHORTCUTS
    assert len(SHORTCUTS) >= 10
    for s in SHORTCUTS:
        assert s.keys and s.label and s.category


def test_shortcuts_install_adds_qshortcuts(qtbot, qt_theme_applied):
    from spacr.qt.app import MainWindow
    from spacr.qt import shortcuts
    from PySide6.QtGui import QShortcut
    win = MainWindow()
    qtbot.addWidget(win)
    # install() ran in MainWindow.__init__; count QShortcut children.
    scs = win.findChildren(QShortcut)
    # At least Ctrl+H + Ctrl+1..9 + Ctrl+K + Ctrl+, + Ctrl+/ + F1 + ?
    assert len(scs) >= len(shortcuts.SHORTCUTS)


def test_notify_signature_is_safe_when_no_backends(monkeypatch):
    """The notify function must never raise — silent no-op on error."""
    from spacr.qt import notify as n
    # Even if platform detection returns nothing usable, it returns False
    monkeypatch.setattr(n.platform, "system", lambda: "Plan9")
    assert n.notify("hello", "world") is False


def test_notify_esc_escapes_double_quotes():
    from spacr.qt.notify import _esc
    assert _esc('he "said"') == 'he \\"said\\"'
    assert _esc("") == ""
    assert _esc(None) == ""


def test_announce_pipeline_finished_falls_back_to_the_tray(monkeypatch):
    """When the OS backend declines, the tray gets the same message.

    Nothing about "it did not raise" is worth pinning: the reason this
    wrapper exists is the fallback, and the reason the fallback is useful
    is that it carries the module, the status and the elapsed time.
    """
    from spacr.qt import notify as n

    os_calls, tray_calls = [], []
    monkeypatch.setattr(n, "notify",
                        lambda *a, **k: (os_calls.append(a), False)[1])
    monkeypatch.setattr(n, "notify_tray",
                        lambda *a, **k: (tray_calls.append(a), False)[1])

    assert n.announce_pipeline_finished("mask", "success", 42.0) is None

    # The OS backend was tried first, with a message naming the module,
    # the status and the wall-clock time.
    assert len(os_calls) == 1
    title, body = os_calls[0][:2]
    assert "mask" in title and "success" in title and "✓" in title
    assert body == "Finished in 42.0s."
    # It said no, so the tray was asked with exactly the same message.
    assert tray_calls == [(title, body)]

    # Contrast: when the OS backend accepts, the tray is NOT double-fired.
    os_calls.clear()
    tray_calls.clear()
    monkeypatch.setattr(n, "notify",
                        lambda *a, **k: (os_calls.append(a), True)[1])
    n.announce_pipeline_finished("measure", "failed", 7.26)

    assert len(os_calls) == 1
    assert tray_calls == [], "the tray fired even though the OS backend took it"
    title, body = os_calls[0][:2]
    assert "measure" in title and "failed" in title and "⚠" in title
    assert body == "Finished in 7.3s."      # one decimal, rounded


def test_htile_has_accessibility_labels(qt_theme_applied):
    from spacr.qt.widgets.tile import HTile
    t = HTile(text="Mask", description="Segment cells.",
              icon=None, icon_size=32)
    assert t.accessibleName() == "Mask"
    assert t.accessibleDescription() == "Segment cells."


def test_sidebar_buttons_have_accessible_names(qtbot, qt_theme_applied):
    from spacr.qt.app import MainWindow
    from PySide6.QtWidgets import QPushButton
    win = MainWindow()
    qtbot.addWidget(win)
    # At least the app buttons on the sidebar should carry accessible
    # names + descriptions.
    labeled = [
        b for b in win.findChildren(QPushButton)
        if b.accessibleName() and b.accessibleDescription()
    ]
    assert len(labeled) >= 5   # 5+ apps in APPS


def test_show_cheat_sheet_opens_and_closes(qtbot, qt_theme_applied):
    """The cheat sheet really appears, and lists every registered binding.

    ``show_cheat_sheet`` blocks in ``exec()``, so the dialog can only be
    inspected from inside its own event loop — which is also the only way
    to tell "it opened and I closed it" apart from "it never opened".
    """
    from spacr.qt.shortcuts import SHORTCUTS, show_cheat_sheet
    from spacr.qt.app import MainWindow
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialog, QLabel

    win = MainWindow()
    qtbot.addWidget(win)

    seen = {}
    ticks = {"n": 0}
    timer = QTimer()
    timer.setInterval(50)

    def inspect_and_close():
        ticks["n"] += 1
        dlg = QApplication.activeModalWidget()
        if isinstance(dlg, QDialog):
            seen["title"] = dlg.windowTitle()
            seen["min_width"] = dlg.minimumWidth()
            seen["rows"] = [lbl.text() for lbl in dlg.findChildren(QLabel)]
            timer.stop()
            dlg.accept()
        elif ticks["n"] > 40:
            # Give up rather than hang forever; the assertions below then
            # report *why* (no modal dialog was ever put on screen).
            timer.stop()
            for w in QApplication.topLevelWidgets():
                if isinstance(w, QDialog) and w.isVisible():
                    w.reject()

    timer.timeout.connect(inspect_and_close)
    timer.start()
    show_cheat_sheet(win)
    timer.stop()

    assert seen, "show_cheat_sheet never put a modal QDialog on screen"
    assert seen["title"] == "spaCR — Keyboard shortcuts"
    assert seen["min_width"] >= 420

    rows = seen["rows"]
    # One header per category plus one row per binding, all non-empty.
    categories = {s.category for s in SHORTCUTS}
    assert len(rows) == len(SHORTCUTS) + len(categories)
    assert all(r.strip() for r in rows)

    text = "\n".join(rows)
    for spec in SHORTCUTS:
        assert spec.keys in text, f"{spec.keys} missing from the cheat sheet"
        assert spec.label in text, f"{spec.label!r} missing from the cheat sheet"
    for cat in categories:
        assert f"<b>{cat}</b>" in text

    # Contrast: the search above is discriminating, not "in a big blob of
    # HTML everything matches" — a binding that is not registered is absent.
    assert "Ctrl+Shift+Q" not in text
    assert "Reticulate splines" not in text
