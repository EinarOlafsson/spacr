"""
Keyboard-first shortcuts for the spaCR Qt GUI.

Registers global :class:`QShortcut` bindings on the main window so
the whole app is usable without a mouse:

    Ctrl+0        Go home (Cmd+H is Hide on macOS)
    Ctrl+1..9     Switch to the Nth app in the sidebar
    Ctrl+K        Open the command palette
    Ctrl+Shift+H  Search spaCR from the field beside the Help menu
    F1  / ?       Show the shortcuts cheat sheet
    Ctrl+P        Open Preferences
    Ctrl+Alt+0    Put GUI scale, font scale and preview scales back to 100 %
    Ctrl+/        Open the AI Console
    Ctrl+End      Jump to the newest console line
    F11           Toggle full screen
    Esc           Close any open dialog / popup

:func:`install` is called once from ``MainWindow.__init__``. Every
binding is documented in :data:`SHORTCUTS` so the cheat-sheet
dialog stays in sync with what's actually wired up.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

from PySide6.QtCore import QEvent, QRectF, Qt
from PySide6.QtGui import QAction, QColor, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QLabel,
    QMainWindow,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

LOG = logging.getLogger("spacr.qt.shortcuts")


#: Where a shortcut works. A key that only works on one screen and is listed
#: without saying so sends a user to press it somewhere it does nothing.
EVERYWHERE = "anywhere in spaCR"


@dataclass(frozen=True)
class ShortcutSpec:
    """One shortcut declaration.

    :param keys: the binding, in Qt's portable spelling. It is PRINTED
        through `QKeySequence.toString(NativeText)`, so `Ctrl` reads as the
        Command symbol on macOS -- writing "Ctrl+H" into a label would
        hard-code one platform into the help.
    :param label: what the key does.
    :param category: the group it is shown under.
    :param scope: where it works. The default is the whole window; a
        per-screen binding names its screen.
    """
    keys:     str
    label:    str
    category: str = "General"
    scope:    str = EVERYWHERE


SHORTCUTS: List[ShortcutSpec] = [
    ShortcutSpec("Ctrl+0",       "Go to home",            "Navigation"),
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
    ShortcutSpec("Ctrl+P",       "Open preferences",       "Navigation"),
    ShortcutSpec("Ctrl+Shift+A", "Show the full app list", "Navigation"),
    ShortcutSpec("F11",          "Toggle full screen",     "Navigation"),
    ShortcutSpec("Ctrl+/",       "Toggle AI Console",      "Actions"),
    ShortcutSpec("Ctrl+End",     "Jump to the newest console line",
                 "Console"),
    ShortcutSpec("Ctrl+F",       "Search this module's settings", "Actions"),
    ShortcutSpec("Ctrl+Shift+R", "Settings templates",     "Actions"),
    ShortcutSpec("Ctrl+T",       "Pause or resume the background",
                 "Background"),
    ShortcutSpec("Ctrl+R",       "Restart the background", "Background"),
    ShortcutSpec("Ctrl+Shift+F", "Show the background full screen",
                 "Background"),
    ShortcutSpec("Ctrl+B",       "Blank the background",   "Background"),
    ShortcutSpec("Z + scroll",   "Resize the interface text",
                 "Background"),
    ShortcutSpec("Ctrl+Alt+0",   "Reset GUI scale and font scale to 100%",
                 "Navigation"),
    ShortcutSpec("F11",          "Full screen",            "Actions"),
    ShortcutSpec("Ctrl+Shift+H", "Search spaCR from the Help bar", "Help"),
    ShortcutSpec("F1",           "Show this cheat sheet",  "Help"),
    ShortcutSpec("?",            "Show this cheat sheet",  "Help")
]


#: THE PER-SCREEN BINDINGS, kept apart from :data:`SHORTCUTS` on purpose.
#:
#: `SHORTCUTS` is what `install()` BINDS on the main window, and a test
#: asserts every entry in it is wired there. These keys are bound by the
#: screens that own them and do not exist until such a screen is built --
#: so putting them in the same table made "declared" and "wired" stop
#: meaning the same thing, and four tests said so at once.
#:
#: THE MAP IS BOTH (:func:`mapped`). The distinction is real -- one set is
#: always live and the other is not -- and it is the same distinction the
#: `scope` field states to the reader.
SCREEN_SHORTCUTS: List[ShortcutSpec] = [
    ShortcutSpec("Left",         "Previous image",         "Annotate",
                 "the Annotate and Make Masks screens and the QC field "
                 "browser"),
    ShortcutSpec("Right",        "Next image",             "Annotate",
                 "the Annotate and Make Masks screens and the QC field "
                 "browser"),
    ShortcutSpec("PageUp",       "Previous image",         "Annotate",
                 "the Annotate screen"),
    ShortcutSpec("PageDown",     "Next image",             "Annotate",
                 "the Annotate screen"),
    ShortcutSpec("Alt+Left",     "Previous image",         "Annotate",
                 "the Annotate screen"),
    ShortcutSpec("Alt+Right",    "Next image",             "Annotate",
                 "the Annotate screen"),
    ShortcutSpec("Y",            "Confirm the suggested label", "Annotate",
                 "the Annotate screen"),
    ShortcutSpec("N",            "Reject the suggested label", "Annotate",
                 "the Annotate screen"),

    ShortcutSpec("B",            "Brush",                  "Make Masks",
                 "the Make Masks screen"),
    ShortcutSpec("E",            "Erase",                  "Make Masks",
                 "the Make Masks screen"),
    ShortcutSpec("W",            "Magic wand — add",       "Make Masks",
                 "the Make Masks screen"),
    ShortcutSpec("D",            "Draw an object",         "Make Masks",
                 "the Make Masks screen"),
    ShortcutSpec("V",            "Divide an object",       "Make Masks",
                 "the Make Masks screen"),
    ShortcutSpec("R",            "Recrop an object",       "Make Masks",
                 "the Make Masks screen"),
    ShortcutSpec("Z",            "Zoom",                   "Make Masks",
                 "the Make Masks screen"),
    ShortcutSpec("M",            "Live magnifier",         "Make Masks",
                 "the Make Masks screen"),
    ShortcutSpec("Esc",          "Reset the zoom",         "Make Masks",
                 "the Make Masks screen"),
    ShortcutSpec("Ctrl+S",       "Save the mask",          "Make Masks",
                 "the Make Masks screen"),
    ShortcutSpec("Ctrl+Z",       "Undo",                   "Make Masks",
                 "the Make Masks screen"),
    ShortcutSpec("Ctrl+Y",       "Redo",                   "Make Masks",
                 "the Make Masks screen"),
    ShortcutSpec("Ctrl+Shift+Z", "Redo",                   "Make Masks",
                 "the Make Masks screen"),

    ShortcutSpec("Q",            "Quarantine or restore this field",
                 "Field browser",
                 "the QC field browser"),
]


#: Window-wide keys that something OTHER than `install()` binds. They are
#: attached to window actions, so they belong on the map and not in
#: ``install()``'s count.
BOUND_ELSEWHERE = frozenset({
    "Ctrl+Shift+A", "Ctrl+B", "Ctrl+T", "Ctrl+R", "Ctrl+Shift+F", "F11",
    "Ctrl+0", "Ctrl+P",
})


def _is_a_gesture(keys: str) -> bool:
    """Whether ``keys`` describes a hand movement rather than a key sequence.

    RECOGNISED BY SHAPE, NOT BY A LIST OF EXCEPTIONS. A real accelerator is
    "Ctrl+Shift+A" with no spaces; a gesture is prose -- "Z + scroll" -- and
    the spaces around the plus are what say so. A second gesture therefore
    needs no edit here, and a mistyped sequence still fails rather than
    being quietly excused as a gesture.

    The same rule is asserted from the other side in
    `tests/qt/test_shortcut_overlay.py`, which refuses any gesture it has
    not been told about.
    """
    return " + " in str(keys)


def installed() -> List[ShortcutSpec]:
    """The window-wide keys that :func:`install` is responsible for binding.

    Gestures are not among them. A gesture is a modifier held while the
    mouse wheel turns, and it is caught by an event filter rather than by a
    key sequence, so no shortcut object can express it and none is created
    for it. Listing one here would promise a binding that cannot be made.

    Every gesture is still listed on the shortcut map. :func:`mapped`
    returns what the hands can do, and this returns what the shortcut
    objects own.
    """
    return [s for s in SHORTCUTS
            if s.keys not in BOUND_ELSEWHERE and not _is_a_gesture(s.keys)]


def mapped() -> List[ShortcutSpec]:
    """Every shortcut the map describes: window-wide, then per-screen."""
    return list(SHORTCUTS) + list(SCREEN_SHORTCUTS)


def native(keys: str) -> str:
    """``keys`` in the spelling the user's own keyboard has.

    `Ctrl` is the Command symbol on macOS and Qt already knows; writing
    "Ctrl+H" into a label hard-codes one platform into the help.

    :param keys: a key sequence in Qt's portable spelling, such as
        ``'Ctrl+H'``. Returned unchanged when Qt cannot convert it.
    """
    try:
        return QKeySequence(str(keys)).toString(QKeySequence.NativeText) \
            or str(keys)
    except Exception:                                    # noqa: BLE001
        return str(keys)


def discover(window) -> List[ShortcutSpec]:
    """Every shortcut LIVE on ``window``, whether declared or not.

    The declared table is what the map is drawn from, because a per-screen
    binding does not exist until that screen is built and the map has to
    describe it anyway. This is the other half: a shortcut added at runtime
    -- a plugin, a menu action -- appears without anyone editing a list.

    Anything already in :data:`SHORTCUTS` is left to its declaration, which
    is where the label and the scope live.

    :param window: the widget whose child :class:`QShortcut` and
        :class:`QAction` objects are searched; a widget that cannot be searched
        yields an empty list.
    """
    from PySide6.QtGui import QAction

    known = {native(spec.keys) for spec in mapped()}
    out: List[ShortcutSpec] = []
    seen = set()
    try:
        holders = list(window.findChildren(QShortcut)) \
            + list(window.findChildren(QAction))
    except Exception:                                    # noqa: BLE001
        return out
    for holder in holders:
        try:
            sequence = holder.key() if isinstance(holder, QShortcut) \
                else holder.shortcut()
            printed = sequence.toString(QKeySequence.NativeText)
        except Exception:                                # noqa: BLE001
            continue
        if not printed or printed in known or printed in seen:
            continue
        seen.add(printed)
        label = ""
        if isinstance(holder, QAction):
            label = holder.text().replace("&", "").strip()
        out.append(ShortcutSpec(printed, label or "(not described)",
                                "Other"))
    return out


def install(window: QMainWindow) -> None:
    """Wire every shortcut in :data:`SHORTCUTS` onto ``window``.

    Idempotent — safe to call from within reload paths.

    :param window: the main window the application shortcuts are bound to.
    """
    _bind(window, "Ctrl+K", lambda: _open_palette(window))
    _bind(window, "Ctrl+/", lambda: _toggle_ai(window))
    end = _bind(window, "Ctrl+End", lambda: _jump_to_the_newest_line(window))
    if end is not None:
        end.activatedAmbiguously.connect(
            lambda: _jump_to_the_newest_line(window))
    _watch_the_stack_for_consoles(window)
    _bind(window, "Ctrl+F", lambda: _focus_settings_search(window))
    _bind(window, "Ctrl+Shift+H", lambda: _focus_help_search(window))
    _bind(window, "Ctrl+Shift+R", lambda: _open_recipes(window))
    _bind(window, "F1",     lambda: show_cheat_sheet(window))
    _bind(window, "Ctrl+Alt+0", lambda: _reset_every_scale(window))
    _bind(window, "?",      lambda: _help_key(window))
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
        from .help_search import install_window_hooks as _help_search_hooks
        _help_search_hooks(window)
    except Exception:
        LOG.debug("Could not install the help search field", exc_info=True)
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
    try:
        from .screens.map_barcodes import install_window_hooks as _fold_hooks
        _fold_hooks(window)
    except Exception:
        LOG.debug("Could not install the fold-strip hooks", exc_info=True)

    try:
        pin = getattr(window, "pin_all_menu_roles", None)
        if callable(pin):
            pin()
    except Exception:
        LOG.debug("Could not pin the menu roles", exc_info=True)


def _bind(window: QMainWindow, keys: str,
          cb: Callable[[], None]) -> Optional[QShortcut]:
    """Wire ``keys`` on ``window`` and hand the binding back to the caller.

    ONCE PER KEY, which is what makes :func:`install` idempotent in the only
    sense that matters here: a second holder of one key makes it AMBIGUOUS,
    and an ambiguous shortcut fires neither handler -- so a reload path
    calling `install` again would silence every key it re-bound.

    Only the window's OWN shortcuts are consulted. `findChildren` reaches
    the whole tree, and a console panel holding `Ctrl+End` deeper down would
    otherwise look like this key was already wired.

    Returned rather than dropped because a key with a second holder
    somewhere else needs its ambiguous activation connected too.
    """
    sequence = QKeySequence(keys)
    for existing in window.findChildren(
            QShortcut, options=Qt.FindDirectChildrenOnly):
        if existing.key() == sequence:
            return existing
    for action in window.findChildren(QAction):
        if not action.shortcut().isEmpty() and action.shortcut() == sequence:
            return None
    sc = QShortcut(sequence, window)
    sc.setContext(Qt.WindowShortcut)
    sc.activated.connect(cb)
    return sc


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
    """Navigate the window to a module.

    :param window: the main window; one without the navigation slot is
        tolerated, so a shortcut cannot crash a bare dialog.
    :param key: the module to open.
    """
    if hasattr(window, "_on_nav_selected"):
        window._on_nav_selected(key)


def _nav_by_index(window: QMainWindow, idx: int) -> None:
    """Navigate to the nth VISIBLE module, for the number-key shortcuts.

    An index past the end, or one naming a module the user has hidden, does
    nothing -- a shortcut that jumps somewhere unexpected is worse than one
    that does not fire.

    :param window: the main window.
    :param idx: the module's position in the registry.
    """
    try:
        from .app import APPS, app_is_visible
        if 0 <= idx < len(APPS) and app_is_visible(APPS[idx][0]):
            _nav(window, APPS[idx][0])
    except Exception:
        pass


def _open_palette(window: QMainWindow) -> None:
    """Open the command palette.

    :param window: the main window. A build without the palette logs and
        does nothing rather than raising out of a key press.
    """
    try:
        from .command_palette import CommandPalette
        CommandPalette(window).exec()
    except Exception as e:
        LOG.debug("command palette not available: %s", e)


def _open_preferences(window: QMainWindow) -> None:
    """Open the Preferences dialog.

    :param window: the main window.
    """
    try:
        from .preferences import PreferencesDialog
        PreferencesDialog(window).exec()
    except Exception as e:
        LOG.debug("preferences dialog not available: %s", e)


def _reset_every_scale(window: QMainWindow) -> None:
    """Put GUI scale, font scale and every preview scale back to 100 %, now.

    The backup way out of a scale too small to read (item 471): it needs no
    reading and asks nothing. Ctrl+Alt rather than Ctrl+Shift, which some
    Windows keyboard setups take for switching layout, and 0 rather than a
    letter, because Cmd+Option+0 is not one of the combinations macOS
    reserves.

    :param window: the main window.
    """
    try:
        from .gui_scale import reset_every_scale
        reset_every_scale(window)
    except Exception:                                    # noqa: BLE001
        LOG.debug("could not reset the scales", exc_info=True)


def _toggle_ai(window: QMainWindow) -> None:
    """Toggle the AI switch on the currently active AppScreen."""
    try:
        from .screens.app_screen import AppScreen
        current = None
        for s in window.findChildren(AppScreen):
            if s.isVisible():
                current = s
                break
        if current is not None and hasattr(current, "_ai_switch"):
            current._ai_switch.setChecked(
                not current._ai_switch.isChecked()
            )
    except Exception:                                    # noqa: BLE001
        LOG.debug("could not toggle the AI switch", exc_info=True)


def _consoles(window) -> list:
    """Every console panel living under ``window``, newest screens included."""
    try:
        from .widgets.console_panel import ConsolePanel
        return list(window.findChildren(ConsolePanel))
    except Exception:                                    # noqa: BLE001
        LOG.debug("could not look for console panels", exc_info=True)
        return []


def _hand_ctrl_end_to_the_window(window, panels=None) -> None:
    """Stand the consoles' own ``Ctrl+End`` down in favour of the window's.

    Two live bindings for one key are not two chances to be heard: Qt calls
    that ambiguous and fires NEITHER handler. A console panel binds the key
    on itself so the gesture still works when the panel is used on its own,
    and inside a window that binds it too that copy is redundant -- the
    window's reaches the same panel and reaches it from every screen.

    :param panels: the consoles to sweep, when the caller has already found
        them; otherwise they are looked up.
    """
    for panel in (_consoles(window) if panels is None else panels):
        own = getattr(panel, "_end_shortcut", None)
        if own is None:
            continue
        try:
            if own.isEnabled():
                own.setEnabled(False)
        except RuntimeError:                             # noqa: PERF203
            continue


def _watch_the_stack_for_consoles(window) -> None:
    """Sweep each screen as it is shown, since screens are built on demand.

    The console of a module that has never been opened does not exist yet,
    so the stand-down cannot be done once at start-up and be finished.
    """
    _hand_ctrl_end_to_the_window(window)
    try:
        window._stack.currentChanged.connect(
            lambda _index: _hand_ctrl_end_to_the_window(window))
    except Exception:                                    # noqa: BLE001
        LOG.debug("no screen stack to watch for consoles", exc_info=True)


def _jump_to_the_newest_line(window) -> None:
    """Send the console on screen to its newest line.

    A long run writes thousands of lines and the one that matters is the
    last; getting to it must not be a scroll through everything above it.
    Screens without a console are left alone rather than swallowing the key.
    """
    panels = _consoles(window)
    _hand_ctrl_end_to_the_window(window, panels)
    for panel in panels:
        try:
            if not panel.isVisible():
                continue
            panel.jump_to_the_end()
            return
        except Exception:                                # noqa: BLE001
            LOG.debug("could not jump the console to its end", exc_info=True)


#: objectNames, so the theme can reach the overlay and tests can find it.
OVERLAY_NAME = "ShortcutOverlay"
OVERLAY_CARD_NAME = "ShortcutOverlayCard"
OVERLAY_SCROLL_NAME = "ShortcutOverlayScroll"


class _ShortcutCard(QWidget):
    """Paint the rounded shortcut surface independently of global stylesheet timing."""

    def paintEvent(self, event):
        """Keep the background 80-percent opaque and shortcut text fully opaque."""
        from .theme import active_palette

        palette = active_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        fill = QColor(palette["surface"])
        fill.setAlphaF(.8)
        painter.setBrush(fill)
        painter.setPen(QPen(QColor(palette["border"]), 1))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(.5, .5, -.5, -.5), 16, 16)


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

    :param window: the window to cover and to read the bindings from. The
        overlay is drawn OVER it rather than as a dialog of its own, which is
        the whole argument above -- so this is not a parent in the ordinary
        sense but the thing being annotated.
    """

    def __init__(self, window: QWidget):
        """Build the shortcut cheat sheet as a card over the window.

        :param window: the window it covers; the card is centred in it and
            scrolls when the map does not fit.
        """
        from .i18n import tr

        super().__init__(window)
        self.setObjectName(OVERLAY_NAME)
        self._window = window
        self.setGeometry(window.rect())
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setFocusPolicy(Qt.StrongFocus)

        self._card = _ShortcutCard(self)
        self._card.setObjectName(OVERLAY_CARD_NAME)
        self._card.setStyleSheet(
            "QWidget#ShortcutOverlayCard, QScrollArea#ShortcutOverlayScroll, "
            "QScrollArea#ShortcutOverlayScroll QWidget { background: transparent; border: none; }")
        card_layout = QVBoxLayout(self._card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)

        self._scroll = QScrollArea(self._card)
        self._scroll.setObjectName(OVERLAY_SCROLL_NAME)
        self._scroll.setWidgetResizable(False)
        self._scroll.setFocusPolicy(Qt.NoFocus)
        self._scroll.viewport().setAutoFillBackground(False)
        self._scroll.viewport().installEventFilter(self)
        card_layout.addWidget(self._scroll)

        self._card_content = QWidget()
        self._card_content.setAutoFillBackground(False)
        grid = QGridLayout(self._card_content)
        grid.setContentsMargins(28, 24, 28, 24)
        grid.setHorizontalSpacing(36)
        grid.setVerticalSpacing(6)

        title = QLabel(tr("Keyboard shortcuts"), self._card_content)
        title.setObjectName("ShortcutOverlayTitle")
        grid.addWidget(title, 0, 0, 1, 2)

        by_cat: dict[str, list[ShortcutSpec]] = {}
        for spec in mapped() + discover(self.parent()):
            by_cat.setdefault(spec.category, []).append(spec)

        room = max(int(self.width() * 0.9), 640)
        per_pair = 420
        pairs = max(1, min(len(by_cat), room // per_pair))
        band = 1
        column = 0
        for index, (category, specs) in enumerate(by_cat.items()):
            if index and index % pairs == 0:
                band = grid.rowCount() + 1
                column = 0
            row = band
            header = QLabel(tr(category).upper(), self._card_content)
            header.setObjectName("ShortcutOverlayCategory")
            grid.addWidget(header, row, column, 1, 2)
            row += 1
            for spec in specs:
                keys = QLabel(native(spec.keys), self._card_content)
                keys.setObjectName("ShortcutOverlayKeys")
                keys.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                grid.addWidget(keys, row, column)
                said = tr(spec.label)
                if spec.scope and spec.scope != EVERYWHERE:
                    said = f"{said}  —  {tr(spec.scope)}"
                label = QLabel(said, self._card_content)
                label.setObjectName("ShortcutOverlayLabel")
                grid.addWidget(label, row, column + 1)
                row += 1
            column += 2

        hint = QLabel(tr("Press any key to close."), self._card_content)
        hint.setObjectName("ShortcutOverlayHint")
        grid.addWidget(hint, grid.rowCount(), 0, 1, max(column, 2))

        self._scroll.setWidget(self._card_content)
        grid.activate()
        self._card_content.adjustSize()
        self._reposition()
        window.installEventFilter(self)

    def paintEvent(self, event) -> None:
        """Leave the main window visible around the translucent shortcut card.

        :param event: the paint event; ignored, so nothing is painted behind
            the card.
        """
        pass

    def resizeEvent(self, event) -> None:
        """Keep the card centred when the window resizes.

        :param event: the resize event; not read, the card is re-centred for
            the overlay's current size.
        """
        self._reposition()

    def _reposition(self) -> None:
        """Centre the card and size it to its content, within the window.

        The scrollbar's width is added only when the content is actually taller
        than the space -- reserving it unconditionally would leave a gap beside
        a map that fits.
        """
        hint = self._card_content.sizeHint()
        max_width = max(1, self.width() - 24)
        max_height = max(1, self.height() - 24)
        needs_vertical_scroll = hint.height() > max_height
        scrollbar_width = (
            self._scroll.verticalScrollBar().sizeHint().width()
            if needs_vertical_scroll else 0
        )
        width = min(max_width, hint.width() + scrollbar_width)
        height = min(max_height, hint.height())
        self._card_content.resize(hint)
        self._card.setGeometry(
            max(0, (self.width() - width) // 2),
            max(0, (self.height() - height) // 2),
            width, height,
        )

    def eventFilter(self, obj, event):
        """Track the window's size so the overlay stays full-bleed.

        :param obj: the watched object: the covered window or the card's scroll
            viewport.
        :param event: the event delivered to it. A resize of the window resizes
            the overlay to match; a mouse press on the scroll viewport
            dismisses the overlay and is consumed.
        """
        # getattr: the overlay and its window are a reference cycle, so the
        # collector can clear this wrapper before the window's destructor
        # reaches the filter.
        window = getattr(self, "_window", None)
        scroll = getattr(self, "_scroll", None)
        if window is None or scroll is None:
            return super().eventFilter(obj, event)
        if obj is window and event.type() == QEvent.Resize:
            self.setGeometry(window.rect())
        if obj is scroll.viewport() \
                and event.type() == QEvent.MouseButtonPress:
            self.dismiss()
            return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event) -> None:
        """Any key closes it — that is the whole interaction.

        :param event: the key event; which key was pressed is not read.
        """
        self.dismiss()

    def mousePressEvent(self, event) -> None:
        """A click anywhere closes it too.

        :param event: the mouse press event; its button and position are not
            read.
        """
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
    left alone rather than swallowing the key. A settings column collapsed to
    the left is opened first (item 471): the caret cannot go into a box
    nobody can see.
    """
    try:
        screen = window._stack.currentWidget()
    except Exception:
        return
    bar = getattr(screen, "_settings_search", None)
    if bar is None:
        return
    reveal = getattr(screen, "reveal_settings", None)
    if callable(reveal):
        try:
            reveal()
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not open the settings column", exc_info=True)
    try:
        bar._input.setFocus()
        bar._input.selectAll()
    except Exception:
        LOG.debug("could not focus the settings search box", exc_info=True)


def _focus_help_search(window: QMainWindow) -> None:
    """Put the caret in the search box beside the Help menu.

    Ctrl+F already means "find a setting on THIS module", so the field that
    searches the whole program needs its own key rather than a second
    meaning for that one: a key that does two things depending on what is
    on screen is a key nobody trusts.
    """
    try:
        from .help_search import focus_field

        focus_field(window)
    except Exception:
        LOG.debug("could not focus the help search box", exc_info=True)


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

    :param parent: the window to cover. A :class:`QWidget` with a non-zero size
        gets the overlay, replacing any overlay it already has; anything else
        gets a modal dialog parented to it.
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

    from .widgets.workflow_diagram import DiagramDialog
    dlg = DiagramDialog(parent)
    dlg.setWindowTitle("spaCR — Keyboard shortcuts")
    from .preferences import scaled_px
    
    dlg.setMinimumWidth(scaled_px(420))
    layout = QVBoxLayout(dlg)

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
    from .theme import block_surface, font_px
    surface = block_surface("surface", palette["theme"], opacity)
    return f"""
QWidget#{OVERLAY_CARD_NAME} {{
    background: {surface};
    border: 1px solid {palette["accent"]};
    border-radius: 12px;
}}
QScrollArea#{OVERLAY_SCROLL_NAME} {{
    background: transparent;
    border: none;
}}
QScrollArea#{OVERLAY_SCROLL_NAME} > QWidget > QWidget {{
    background: transparent;
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
    color: {palette["fg"]};
}}
QLabel#ShortcutOverlayHint {{
    color: {palette["fg_dim"]};
    font-size: {font_px(11)}px;
    padding-top: 12px;
}}
"""


try:
    from .theme import register_widget_qss as _register_widget_qss
    _register_widget_qss(OVERLAY_NAME, _overlay_qss, replace=True)
except Exception:
    LOG.debug("could not register the shortcut-overlay QSS", exc_info=True)
