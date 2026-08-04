"""Per-module walkthroughs — the coach-marks, once per module, on demand.

:mod:`spacr.qt.first_run` shows one tour, once, about the home screen. That
answered "where am I?" and nothing else: a user who has seen it still opens
Mask for the first time and meets 190 settings under thirteen headings with
no idea which two of them matter, and there is no way to ask again.

This module makes the tour per module and repeatable:

* the first time a module is opened, its own short walkthrough runs;
* every walkthrough is available afterwards from **Help → Walkthroughs**,
  and from the command palette, for any module — not only the one on
  screen;
* "seen" is tracked per module, so a new module added to the shell next
  release introduces itself to an existing user rather than staying silent
  because the one global flag was set in 2026.

A walkthrough is built from the module's own settings layout rather than
hand-written per module. That is what stops it rotting: the steps name the
module's first curated group and its essential settings, both of which come
from :mod:`spacr.qt.screens.settings_model`, so a layout change updates the
walkthrough with it. Modules that want more can register extra steps::

    from spacr.qt.walkthrough import register_steps
    register_steps("mask", [WalkStep("Test mode first", "...")])

Reuses :class:`spacr.qt.first_run._TourOverlay` for the rendering, because a
second dimmed-overlay-with-a-card would be a second thing to keep looking
like the first one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow, QMenu, QWidget

from .first_run import TourStep, _TourOverlay, find_menu

LOG = logging.getLogger("spacr.qt.walkthrough")

_ORG = "spacr"
_APP = "qt"
_KEY_SEEN = "onboarding/module_walkthrough_seen"

#: The Help submenu label. Kept verbatim — ``spacr/qt/i18n.py`` keys its
#: catalog on the English string.
MENU_TITLE = "Walkthroughs"


def _settings():
    from PySide6.QtCore import QSettings
    return QSettings(_ORG, _APP)


# ---------------------------------------------------------------------------
# Seen state, per module
# ---------------------------------------------------------------------------

def was_seen(app_key: str) -> bool:
    """True once ``app_key``'s walkthrough has been finished or dismissed."""
    raw = _settings().value(f"{_KEY_SEEN}/{app_key}", False)
    if isinstance(raw, bool):
        return raw
    return str(raw).lower() in ("true", "1", "yes")


def mark_seen(app_key: str) -> None:
    """Remember that ``app_key``'s walkthrough has been shown."""
    _settings().setValue(f"{_KEY_SEEN}/{app_key}", True)


def reset(app_key: Optional[str] = None) -> None:
    """Forget one module's walkthrough, or every module's.

    :param app_key: the module to reset, or ``None`` for all of them.
    """
    store = _settings()
    if app_key is None:
        store.remove(_KEY_SEEN)
    else:
        store.remove(f"{_KEY_SEEN}/{app_key}")


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

@dataclass
class WalkStep:
    """One narrated coach-mark in a module walkthrough.

    :ivar title: short headline.
    :ivar body: one or two sentences under it.
    :ivar highlight: callable taking the live screen and returning the
        widget to ring, or ``None`` to centre the card.
    """
    title: str
    body: str
    highlight: Optional[Callable[[QWidget], Optional[QWidget]]] = None


#: Module-specific extra steps, appended after the derived ones.
_EXTRA_STEPS: Dict[str, List[WalkStep]] = {}


def register_steps(app_key: str, steps: List[WalkStep],
                   *, replace: bool = False) -> None:
    """Add module-specific steps to ``app_key``'s walkthrough.

    The seam a module uses to say something the layout cannot — "run test
    mode on three fields before committing a plate" is advice, not
    structure, and no amount of reading the settings map produces it.

    :param app_key: the module's app key.
    :param steps: steps appended after the derived ones.
    :param replace: overwrite an existing registration instead of raising.
    :raises ValueError: on a second registration without ``replace``.
    """
    key = str(app_key)
    if key in _EXTRA_STEPS and not replace:
        raise ValueError(
            f"walkthrough steps for {key!r} are already registered; pass "
            "replace=True if that is really what you mean")
    _EXTRA_STEPS[key] = list(steps)


def unregister_steps(app_key: str) -> bool:
    """Drop a module's registered steps. ``True`` if there were any."""
    return _EXTRA_STEPS.pop(str(app_key), None) is not None


def _module_name(app_key: str) -> str:
    try:
        from .app import APPS
        for row in APPS:
            if row[0] == app_key:
                return str(row[1])
    except Exception:
        pass
    return app_key


def _sentence(names: List[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def build_steps(app_key: str) -> List[WalkStep]:
    """The walkthrough for one module, derived from its settings layout.

    Four beats, in the order somebody actually works: what the module is,
    where its inputs go, which settings matter out of how many, and how to
    run it. Every fact in them is read from the registry or the layout, so
    none of them can go stale the way a hand-written paragraph does.

    :param app_key: the module's app key.
    """
    name = _module_name(app_key)
    steps: List[WalkStep] = []

    intro = ""
    try:
        from .app import APPS
        for row in APPS:
            if row[0] == app_key:
                intro = str(row[2])
                break
    except Exception:
        intro = ""
    steps.append(WalkStep(
        title=name,
        body=(intro or f"This is the {name} module.")
             + " Press Esc at any time to close this walkthrough; you can "
               "reopen it from Help → Walkthroughs.",
        highlight=None,
    ))

    groups: List[str] = []
    essentials: List[str] = []
    total = 0
    try:
        from .screens.settings_model import (
            categories_for_app, essential_keys, get_categories,
            resolve_default_settings,
        )
        cats = categories_for_app(app_key, get_categories())
        defaults = resolve_default_settings(app_key)
        groups = [name for name, keys in cats.items()
                  if any(k in defaults for k in keys)]
        essentials = [k for k in essential_keys(app_key, cats)
                      if k in defaults]
        total = len(defaults)
    except Exception:
        LOG.debug("could not read the layout for %r", app_key, exc_info=True)

    if groups:
        steps.append(WalkStep(
            title="Start at the top",
            body=(f"The settings are grouped into {len(groups)}: "
                  f"{_sentence(groups[:4])}"
                  + (", and more." if len(groups) > 4 else ".")
                  + f" “{groups[0]}” is where you point the module at your "
                    "data; everything below it assumes that is right."),
            highlight=_first_section,
        ))

    if essentials and total:
        steps.append(WalkStep(
            title=f"{len(essentials)} settings, not {total}",
            body=(f"The strip above the form opens on Essentials — the "
                  f"{len(essentials)} settings this module cannot run "
                  f"without. Switch it to All settings for the other "
                  f"{total - len(essentials)}, or type in the search box to "
                  "find one by name or by what its description says it "
                  "does."),
            highlight=_search_bar,
        ))

    steps.append(WalkStep(
        title="Run it",
        body=("Run starts the module and streams its log into the console "
              "below. Once a set of settings works, save it as a recipe so "
              "the next plate is one click instead of a form."),
        highlight=_run_button,
    ))

    steps.extend(_EXTRA_STEPS.get(str(app_key), []))
    return steps


def _first_section(screen: QWidget) -> Optional[QWidget]:
    sections = getattr(screen, "_settings_sections", None) or []
    for section in sections:
        if section.isVisible():
            return section
    return sections[0] if sections else None


def _search_bar(screen: QWidget) -> Optional[QWidget]:
    return getattr(screen, "_settings_search", None)


def _run_button(screen: QWidget) -> Optional[QWidget]:
    for attr in ("_btn_run", "_run_btn", "_btn_start"):
        widget = getattr(screen, attr, None)
        if widget is not None:
            return widget
    from PySide6.QtWidgets import QPushButton
    for button in screen.findChildren(QPushButton):
        if button.text().strip().lower() == "run":
            return button
    return None


# ---------------------------------------------------------------------------
# Showing one
# ---------------------------------------------------------------------------

def show_walkthrough(window: QMainWindow, app_key: str,
                     *, force: bool = True) -> Optional[_TourOverlay]:
    """Run ``app_key``'s walkthrough over ``window``.

    Navigates to the module first — a walkthrough that highlights a settings
    group on a screen the user is not looking at highlights nothing.

    :param window: the live main window.
    :param app_key: the module to walk through.
    :param force: show even when it has been seen before. The default, since
        every route to this function except the automatic one is somebody
        asking for it.
    :returns: the overlay, or ``None`` when it was skipped.
    """
    if not force and was_seen(app_key):
        return None
    screen = None
    try:
        nav = getattr(window, "_on_nav_selected", None)
        if callable(nav):
            nav(app_key)
        screen = window._screens.get(app_key)
    except Exception:
        LOG.debug("could not open %r for its walkthrough", app_key,
                  exc_info=True)
    steps = build_steps(app_key)
    if not steps:
        return None
    target = screen if screen is not None else window
    tour = [
        TourStep(
            title=step.title,
            body=step.body,
            highlight=_bind_highlight(step.highlight, target),
        )
        for step in steps
    ]
    overlay = _TourOverlay(window, tour, on_finish=_Seen(app_key).mark)
    overlay.show()
    overlay.raise_()
    overlay.setFocus()
    return overlay


class _Seen:
    """Bound-method callback marking one module's walkthrough seen.

    A tiny object rather than a lambda so the overlay's finish callback is a
    bound method — a closure over ``app_key`` would keep whatever else was
    in that frame alive for as long as the overlay lived.
    """

    def __init__(self, app_key: str):
        self._app_key = app_key

    def mark(self) -> None:
        """Record that this module's walkthrough has been shown."""
        mark_seen(self._app_key)


class _Highlight:
    """Bound-method adapter from a step's screen-taking highlight to the
    window-taking one :class:`spacr.qt.first_run.TourStep` expects."""

    def __init__(self, fn, target):
        self._fn = fn
        self._target = target

    def __call__(self, _window):
        try:
            return self._fn(self._target)
        except Exception:
            return None


def _bind_highlight(fn, target):
    if fn is None:
        return None
    return _Highlight(fn, target)


def maybe_show(window: QMainWindow, app_key: str) -> Optional[_TourOverlay]:
    """Show ``app_key``'s walkthrough if this user has not seen it.

    Called when a module is opened. Per module, so a module added next
    release introduces itself instead of being silenced by a flag set the
    first time the app ever ran.
    """
    if was_seen(app_key):
        return None
    return show_walkthrough(window, app_key, force=True)


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

class _WalkthroughHandler(QObject):
    """Bound-method targets for the menu entries and the screen stack."""

    def __init__(self, window: QMainWindow):
        super().__init__(window)
        self._window = window

    def on_current_changed(self, _index: int) -> None:
        """Offer the walkthrough the first time a module is opened."""
        try:
            screen = self._window._stack.currentWidget()
            app_key = str(getattr(screen, "app_key", "") or "")
        except Exception:
            return
        if not app_key or was_seen(app_key):
            return
        # Only for modules that render the shared settings form; a bespoke
        # screen has no groups to describe and the derived steps would be
        # three sentences of nothing.
        if getattr(screen, "_settings_model", None) is None:
            return
        maybe_show(self._window, app_key)

    def on_reset(self, _checked: bool = False) -> None:
        """Clear every module's seen flag so the walkthroughs run again."""
        reset()


class _MenuTrigger(QObject):
    """One module's Help-menu entry."""

    def __init__(self, window: QMainWindow, app_key: str):
        super().__init__(window)
        self._window = window
        self._app_key = app_key

    def on_triggered(self, _checked: bool = False) -> None:
        """Run this entry's walkthrough."""
        show_walkthrough(self._window, self._app_key, force=True)


def install_help_menu(window: QMainWindow) -> Optional[QMenu]:
    """Add a **Walkthroughs** submenu listing every module.

    Every module, not only the one on screen: the question "how does
    Measure work?" is usually asked from somewhere that is not Measure.

    :returns: the submenu, or ``None`` when there is no Help menu or one is
        already installed.
    """
    help_menu = find_menu(window, "Help")
    if help_menu is None:
        return None
    for act in help_menu.actions():
        if act.text().replace("&", "") == MENU_TITLE:
            return None
    handler = getattr(window, "_walkthrough_handler", None)
    if handler is None:
        handler = _WalkthroughHandler(window)
        window._walkthrough_handler = handler

    submenu = QMenu(MENU_TITLE, window)
    submenu.setToolTipsVisible(True)
    try:
        from .app import APPS, app_is_visible
        rows = [row for row in APPS if app_is_visible(row[0])]
    except Exception:
        rows = []
    for key, name, desc, _section in rows:
        action = QAction(str(name), submenu)
        action.setStatusTip(str(desc))
        trigger = _MenuTrigger(window, key)
        action.triggered.connect(trigger.on_triggered)
        action._spacr_walkthrough_trigger = trigger
        submenu.addAction(action)
    if rows:
        submenu.addSeparator()
    reset_action = QAction("Show all walkthroughs again", submenu)
    reset_action.setStatusTip(
        "Clear the record of which walkthroughs you have seen, so each "
        "module introduces itself once more.")
    reset_action.triggered.connect(handler.on_reset)
    submenu.addAction(reset_action)

    before = None
    for act in help_menu.actions():
        if act.isSeparator():
            before = act
            break
    if before is not None:
        help_menu.insertMenu(before, submenu)
    else:
        help_menu.addMenu(submenu)
    window._walkthrough_menu = submenu
    return submenu


def install_window_hooks(window: QMainWindow) -> Optional[_WalkthroughHandler]:
    """Wire the walkthroughs into a live main window.

    Called once from :func:`spacr.qt.shortcuts.install`.
    """
    install_help_menu(window)
    handler = getattr(window, "_walkthrough_handler", None)
    if handler is None:
        handler = _WalkthroughHandler(window)
        window._walkthrough_handler = handler
    stack = getattr(window, "_stack", None)
    if stack is None or getattr(window, "_walkthrough_wired", False):
        return handler
    try:
        stack.currentChanged.connect(handler.on_current_changed)
    except Exception:
        LOG.debug("could not follow the screen stack", exc_info=True)
        return handler
    window._walkthrough_wired = True
    return handler
