"""
Keyboard-first shortcuts for the spaCR Qt GUI.

Registers global :class:`QShortcut` bindings on the main window so
the whole app is usable without a mouse:

    Ctrl+H        Go home
    Ctrl+1..9     Switch to the Nth app in the sidebar
    Ctrl+K        Open the command palette
    F1  / ?       Show the shortcuts cheat sheet
    Ctrl+,        Open Preferences
    Ctrl+/        Open the AI Console
    Esc           Close any open dialog / popup

:func:`install` is called once from ``MainWindow.__init__``. Every
binding is documented in :data:`SHORTCUTS` so the cheat-sheet
dialog stays in sync with what's actually wired up.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QDialog, QLabel, QMainWindow, QVBoxLayout

LOG = logging.getLogger("spacr.qt.shortcuts")


@dataclass(frozen=True)
class ShortcutSpec:
    """One shortcut declaration."""
    keys:     str
    label:    str
    category: str = "General"


SHORTCUTS: List[ShortcutSpec] = [
    ShortcutSpec("Ctrl+H",       "Go to home",            "Navigation"),
    ShortcutSpec("Ctrl+1",       "Switch to 1st app",      "Navigation"),
    ShortcutSpec("Ctrl+2",       "Switch to 2nd app",      "Navigation"),
    ShortcutSpec("Ctrl+3",       "Switch to 3rd app",      "Navigation"),
    ShortcutSpec("Ctrl+4",       "Switch to 4th app",      "Navigation"),
    ShortcutSpec("Ctrl+5",       "Switch to 5th app",      "Navigation"),
    ShortcutSpec("Ctrl+6",       "Switch to 6th app",      "Navigation"),
    ShortcutSpec("Ctrl+7",       "Switch to 7th app",      "Navigation"),
    ShortcutSpec("Ctrl+8",       "Switch to 8th app",      "Navigation"),
    ShortcutSpec("Ctrl+9",       "Switch to 9th app",      "Navigation"),
    ShortcutSpec("Ctrl+K",       "Open command palette",   "Navigation"),
    ShortcutSpec("Ctrl+,",       "Open preferences",       "Navigation"),
    ShortcutSpec("Ctrl+/",       "Toggle AI Console",      "Actions"),
    ShortcutSpec("F1",           "Show this cheat sheet",  "Help"),
    ShortcutSpec("?",            "Show this cheat sheet",  "Help"),
]


def install(window: QMainWindow) -> None:
    """Wire every shortcut in :data:`SHORTCUTS` onto ``window``.

    Idempotent — safe to call from within reload paths.
    """
    _bind(window, "Ctrl+H", lambda: _nav(window, "__home__"))
    _bind(window, "Ctrl+K", lambda: _open_palette(window))
    _bind(window, "Ctrl+,", lambda: _open_preferences(window))
    _bind(window, "Ctrl+/", lambda: _toggle_ai(window))
    _bind(window, "F1",     lambda: show_cheat_sheet(window))
    _bind(window, "?",      lambda: show_cheat_sheet(window))
    # Ctrl+1 .. Ctrl+9 → nth app in the sidebar
    for i in range(1, 10):
        _bind(window, f"Ctrl+{i}",
                lambda idx=i: _nav_by_index(window, idx - 1))


def _bind(window: QMainWindow, keys: str, cb: Callable[[], None]) -> None:
    sc = QShortcut(QKeySequence(keys), window)
    sc.setContext(Qt.ApplicationShortcut)
    sc.activated.connect(cb)


def _nav(window: QMainWindow, key: str) -> None:
    if hasattr(window, "_on_nav_selected"):
        window._on_nav_selected(key)


def _nav_by_index(window: QMainWindow, idx: int) -> None:
    try:
        from .app import APPS
        if 0 <= idx < len(APPS):
            _nav(window, APPS[idx][0])
    except Exception:
        pass


def _open_palette(window: QMainWindow) -> None:
    try:
        from .command_palette import CommandPalette
        CommandPalette(window).exec()
    except Exception as e:
        LOG.debug("command palette not available: %s", e)


def _open_preferences(window: QMainWindow) -> None:
    try:
        from .preferences_dialog import PreferencesDialog
        PreferencesDialog(window).exec()
    except Exception as e:
        LOG.debug("preferences dialog not available: %s", e)


def _toggle_ai(window: QMainWindow) -> None:
    """Toggle the AI switch on the currently active AppScreen."""
    try:
        from .screens.app_screen import AppScreen
        current = None
        for s in window.findChildren(AppScreen):
            if s.isVisible():
                current = s; break
        if current is not None and hasattr(current, "_ai_switch"):
            current._ai_switch.setChecked(
                not current._ai_switch.isChecked()
            )
    except Exception:
        pass


def show_cheat_sheet(parent) -> None:
    """Show a modal listing every registered shortcut, grouped by category."""
    dlg = QDialog(parent)
    dlg.setWindowTitle("spaCR — Keyboard shortcuts")
    dlg.setMinimumWidth(420)
    layout = QVBoxLayout(dlg)

    # Group by category
    by_cat: dict[str, list[ShortcutSpec]] = {}
    for s in SHORTCUTS:
        by_cat.setdefault(s.category, []).append(s)

    for cat, specs in by_cat.items():
        hdr = QLabel(f"<b>{cat}</b>")
        hdr.setStyleSheet(
            "font-family: 'Open Sans', sans-serif;"
            "font-weight: 600; font-size: 12px;"
            "letter-spacing: 1.5px; margin-top: 8px;"
        )
        layout.addWidget(hdr)
        for s in specs:
            row = QLabel(
                f"<code style='padding:2px 6px; "
                f"background:#1e1e1e; border-radius:3px;'>{s.keys}</code>"
                f"  &nbsp; {s.label}"
            )
            row.setTextFormat(Qt.RichText)
            layout.addWidget(row)

    dlg.exec()
