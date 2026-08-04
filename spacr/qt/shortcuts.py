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

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QDialog, QGridLayout, QLabel, QMainWindow, QVBoxLayout, QWidget,
)

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
    ShortcutSpec("Ctrl+F",       "Search this module's settings", "Actions"),
    ShortcutSpec("Ctrl+Shift+R", "Settings recipes",       "Actions"),
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
    _bind(window, "Ctrl+F", lambda: _focus_settings_search(window))
    _bind(window, "Ctrl+Shift+R", lambda: _open_recipes(window))
    _bind(window, "F1",     lambda: show_cheat_sheet(window))
    _bind(window, "?",      lambda: _help_key(window))
    # Ctrl+1 .. Ctrl+9 → nth app in the sidebar
    for i in range(1, 10):
        _bind(window, f"Ctrl+{i}",
                lambda idx=i: _nav_by_index(window, idx - 1))
    _install_window_hooks(window)


def _install_window_hooks(window: QMainWindow) -> None:
    """Let modules that own a menu entry or a global filter wire themselves.

    This runs once from ``MainWindow.__init__``, after ``_build_menu_bar``,
    which makes it the first moment a module can reach a live menu bar — the
    same route :mod:`spacr.qt.first_run` and :mod:`spacr.qt.command_palette`
    take to find one. Each hook is guarded on its own: an optional help
    entry must never cost anyone a window.
    """
    try:
        from .widgets.feature_dictionary import install_window_hooks
        install_window_hooks(window)
    except Exception:
        LOG.debug("Could not install the feature dictionary hooks",
                  exc_info=True)
    try:
        from .settings_search import install_window_hooks as _search_hooks
        _search_hooks(window)
    except Exception:
        LOG.debug("Could not install the settings search hooks",
                  exc_info=True)
    try:
        from .recipes import install_window_hooks as _recipe_hooks
        _recipe_hooks(window)
    except Exception:
        LOG.debug("Could not install the recipe hooks", exc_info=True)
    try:
        from .preview_registry import install_window_hooks as _preview_hooks
        _preview_hooks(window)
    except Exception:
        LOG.debug("Could not install the preview hooks", exc_info=True)
    try:
        from .walkthrough import install_window_hooks as _walkthrough_hooks
        _walkthrough_hooks(window)
    except Exception:
        LOG.debug("Could not install the walkthrough hooks", exc_info=True)


def _bind(window: QMainWindow, keys: str, cb: Callable[[], None]) -> None:
    sc = QShortcut(QKeySequence(keys), window)
    sc.setContext(Qt.ApplicationShortcut)
    sc.activated.connect(cb)


def _help_key(window: QMainWindow) -> None:
    """Handle bare ``?``: let the active screen claim it, else show the sheet.

    A ``Qt.ApplicationShortcut`` fires before the focused widget's
    ``keyPressEvent``, so without this the global cheat sheet would preempt any
    screen that wants ``?`` for itself — the Annotate screen uses it to toggle
    its inline key legend, and opening a modal sheet over a rapid-labelling
    session is exactly the wrong response.

    A screen opts in by exposing ``handle_key`` and returning True from it.
    ``F1`` remains an unconditional route to the cheat sheet everywhere.
    """
    try:
        screen = window._stack.currentWidget()
    except Exception:
        screen = None
    handler = getattr(screen, "handle_key", None)
    if callable(handler):
        try:
            if handler("?"):
                return
        except Exception:
            pass
    show_cheat_sheet(window)


def _nav(window: QMainWindow, key: str) -> None:
    if hasattr(window, "_on_nav_selected"):
        window._on_nav_selected(key)


def _nav_by_index(window: QMainWindow, idx: int) -> None:
    try:
        from .app import APPS, app_is_visible
        if 0 <= idx < len(APPS) and app_is_visible(APPS[idx][0]):
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
        from .preferences import PreferencesDialog
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


#: objectNames, so the theme can reach the overlay and tests can find it.
OVERLAY_NAME = "ShortcutOverlay"
OVERLAY_CARD_NAME = "ShortcutOverlayCard"


class ShortcutOverlay(QWidget):
    """The ``?`` overlay — every shortcut, over the window, dismissed by any key.

    A modal dialog was the wrong shape for this. The question a user asks by
    pressing ``?`` is "what can I press *here*", and the answer is worth
    about two seconds; a dialog with a title bar and a close button makes
    them commit to a mode, find the button, and leave it. An overlay dims
    what is behind, answers, and disappears on the next keystroke or click —
    including on ``?`` itself, so the key that opened it also closes it.

    Laid out in columns by category rather than one long list, because
    fifteen bindings in one column is a scroll and in three is a glance.
    """

    def __init__(self, window: QWidget):
        super().__init__(window)
        self.setObjectName(OVERLAY_NAME)
        self._window = window
        self.setGeometry(window.rect())
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setFocusPolicy(Qt.StrongFocus)

        self._card = QWidget(self)
        self._card.setObjectName(OVERLAY_CARD_NAME)
        grid = QGridLayout(self._card)
        grid.setContentsMargins(28, 24, 28, 24)
        grid.setHorizontalSpacing(36)
        grid.setVerticalSpacing(6)

        title = QLabel("Keyboard shortcuts", self._card)
        title.setObjectName("ShortcutOverlayTitle")
        grid.addWidget(title, 0, 0, 1, 2)

        by_cat: dict[str, list[ShortcutSpec]] = {}
        for spec in SHORTCUTS:
            by_cat.setdefault(spec.category, []).append(spec)

        column = 0
        for category, specs in by_cat.items():
            row = 1
            header = QLabel(category.upper(), self._card)
            header.setObjectName("ShortcutOverlayCategory")
            grid.addWidget(header, row, column, 1, 2)
            row += 1
            for spec in specs:
                keys = QLabel(spec.keys, self._card)
                keys.setObjectName("ShortcutOverlayKeys")
                keys.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                grid.addWidget(keys, row, column)
                label = QLabel(spec.label, self._card)
                label.setObjectName("ShortcutOverlayLabel")
                grid.addWidget(label, row, column + 1)
                row += 1
            column += 2

        hint = QLabel("Press any key to close.", self._card)
        hint.setObjectName("ShortcutOverlayHint")
        grid.addWidget(hint, grid.rowCount(), 0, 1, max(column, 2))

        self._reposition()
        window.installEventFilter(self)

    # -- painting -----------------------------------------------------
    def paintEvent(self, event) -> None:
        """Dim whatever is behind the card."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 170))
        painter.end()

    def resizeEvent(self, event) -> None:
        """Keep the card centred when the window resizes."""
        self._reposition()

    def _reposition(self) -> None:
        hint = self._card.sizeHint()
        self._card.setGeometry(
            max(0, (self.width() - hint.width()) // 2),
            max(0, (self.height() - hint.height()) // 2),
            hint.width(), hint.height(),
        )

    # -- dismissal ----------------------------------------------------
    def eventFilter(self, obj, event):
        """Track the window's size so the overlay stays full-bleed."""
        if obj is self._window and event.type() == QEvent.Resize:
            self.setGeometry(self._window.rect())
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event) -> None:
        """Any key closes it — that is the whole interaction."""
        self.dismiss()

    def mousePressEvent(self, event) -> None:
        """A click anywhere closes it too."""
        self.dismiss()

    def dismiss(self) -> None:
        """Close the overlay and let go of the window."""
        try:
            self._window.removeEventFilter(self)
        except RuntimeError:
            pass
        self.close()
        self.deleteLater()


def _focus_settings_search(window: QMainWindow) -> None:
    """Put the caret in the current module's settings search box.

    Ctrl+F on a settings form should mean "find a setting", which is the
    only thing on that screen anyone searches. Screens without a strip are
    left alone rather than swallowing the key.
    """
    try:
        screen = window._stack.currentWidget()
    except Exception:
        return
    bar = getattr(screen, "_settings_search", None)
    if bar is None:
        return
    try:
        bar._input.setFocus()
        bar._input.selectAll()
    except Exception:
        LOG.debug("could not focus the settings search box", exc_info=True)


def _open_recipes(window: QMainWindow) -> None:
    """Open the recipe dialog for the module on screen."""
    try:
        from .recipes import _RecipeMenuHandler
        _RecipeMenuHandler(window).on_triggered()
    except Exception as e:
        LOG.debug("recipes not available: %s", e)


def show_cheat_sheet(parent) -> None:
    """Show every registered shortcut, grouped by category.

    An overlay when ``parent`` is a real window, so ``?`` answers and gets
    out of the way. A modal dialog remains the fallback for a parentless or
    zero-sized caller, where an overlay would have nothing to cover.
    """
    if isinstance(parent, QWidget) and parent.width() > 0 \
            and parent.height() > 0:
        existing = getattr(parent, "_spacr_shortcut_overlay", None)
        if existing is not None:
            try:
                existing.dismiss()
            except RuntimeError:
                pass
        overlay = ShortcutOverlay(parent)
        parent._spacr_shortcut_overlay = overlay
        overlay.show()
        overlay.raise_()
        overlay.setFocus()
        return overlay

    dlg = QDialog(parent)
    dlg.setWindowTitle("spaCR — Keyboard shortcuts")
    dlg.setMinimumWidth(420)
    layout = QVBoxLayout(dlg)

    # Group by category
    by_cat: dict[str, list[ShortcutSpec]] = {}
    for s in SHORTCUTS:
        by_cat.setdefault(s.category, []).append(s)

    for cat, specs in by_cat.items():
        from .theme import font_px
        hdr = QLabel(f"<b>{cat}</b>")
        hdr.setStyleSheet(
            "font-family: 'Open Sans', sans-serif;"
            f"font-weight: 600; font-size: {font_px(12)}px;"
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
    return None


def _overlay_qss(palette: dict, opacity) -> str:
    """QSS for the ``?`` overlay, registered through the theme seam."""
    from .theme import font_px, pane_surface
    surface = pane_surface("surface", palette["theme"], opacity)
    return f"""
QWidget#{OVERLAY_CARD_NAME} {{
    background: {surface};
    border: 1px solid {palette["accent"]};
    border-radius: 12px;
}}
QLabel#ShortcutOverlayTitle {{
    font-size: {font_px(18)}px;
    color: {palette["fg"]};
    padding-bottom: 8px;
}}
QLabel#ShortcutOverlayCategory {{
    font-size: {font_px(10)}px;
    font-weight: 600;
    letter-spacing: 2px;
    color: {palette["accent"]};
    padding-top: 10px;
}}
QLabel#ShortcutOverlayKeys {{
    font-family: monospace;
    color: {palette["fg"]};
}}
QLabel#ShortcutOverlayLabel {{
    color: {palette["fg_dim"]};
}}
QLabel#ShortcutOverlayHint {{
    color: {palette["fg_dim"]};
    font-size: {font_px(11)}px;
    padding-top: 12px;
}}
"""


try:  # pragma: no cover - present in every real launch
    from .theme import register_widget_qss as _register_widget_qss
    _register_widget_qss(OVERLAY_NAME, _overlay_qss, replace=True)
except Exception:  # pragma: no cover
    LOG.debug("could not register the shortcut-overlay QSS", exc_info=True)
