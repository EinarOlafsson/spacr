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


def test_announce_pipeline_finished_does_not_raise(monkeypatch):
    """Wrapper never raises even if all notification backends fail."""
    from spacr.qt import notify as n
    monkeypatch.setattr(n, "notify", lambda *_a, **_k: False)
    monkeypatch.setattr(n, "notify_tray", lambda *_a, **_k: False)
    # Should complete cleanly, no exception
    n.announce_pipeline_finished("mask", "success", 42.0)


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
    from spacr.qt.shortcuts import show_cheat_sheet
    from spacr.qt.app import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    # Sanity — verify the dialog can be created + closed without error.
    # We can't actually exec() modally in a test, but we can at least
    # verify the function exists + reaches the QDialog construction.
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialog
    def close_active():
        d = QApplication.activeModalWidget()
        if isinstance(d, QDialog):
            d.accept()
    QTimer.singleShot(200, close_active)
    show_cheat_sheet(win)
