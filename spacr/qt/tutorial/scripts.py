"""Per-app tutorial Step sequences.

Each `_build_<app>_steps(window)` returns a list of engine.Step. The
engine handles narration synthesis + capture + mux; these functions
only choose the narration text, the UI actions, and the cursor
targets.

Every script exercises the same core motion: land on the app, load
a synthetic demo dataset (via the Demos menu we shipped), highlight
the interesting parts of the settings form, then click Run.

Two rules keep a script from quietly pointing at nothing:

1. **Never name a widget that does not exist yet.** A Step's list is
   built before any of its actions have run, so a settings panel or a
   Run button on a screen the script has not opened yet evaluates to
   ``None`` at build time and stays ``None`` for the whole render. Pass
   a zero-argument callable instead — ``target=(lambda: _find_button(
   screen_ref[0], "Run"), None)`` — and the engine resolves it at
   capture time. See ``Director._deref``.
2. **Never hard-code a pixel offset into a container.** Point at the
   widget itself (``_sidebar_button``, ``_find_button``,
   ``_menu_target``) and let the engine take its centre. Literal
   offsets silently drift the moment a row is added to APPS or the
   font scale changes.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

from .engine import Step

LOG = logging.getLogger("spacr.qt.tutorial")

AVAILABLE_TUTORIALS = [
    "home", "mask", "measure", "crop", "classify", "timelapse",
]


def build_steps(app_key: str, window) -> List[Step]:
    """Return the tutorial Step list for ``app_key`` bound to ``window``.

    :param app_key: one of :data:`AVAILABLE_TUTORIALS`.
    :param window: live MainWindow instance the steps drive.
    :raises ValueError: when ``app_key`` is not a known tutorial.
    """
    if app_key == "home":       return _build_home_steps(window)
    if app_key == "mask":       return _build_mask_steps(window)
    if app_key == "measure":    return _build_measure_steps(window)
    if app_key == "crop":       return _build_crop_steps(window)
    if app_key == "classify":   return _build_classify_steps(window)
    if app_key == "timelapse":  return _build_timelapse_steps(window)
    raise ValueError(f"unknown tutorial: {app_key}. "
                       f"Choose from {AVAILABLE_TUTORIALS}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _go_home(window):
    def _do():
        window._on_nav_selected("__home__")
    return _do


def _nav_to(window, app_key: str):
    def _do():
        window._on_nav_selected(app_key)
    return _do


def _load_demo(window, demo_key: str, tmp_root: str):
    """Bypass the file dialog — call the internals directly with a
    scratch destination. Same code path the Demos menu uses."""
    from pathlib import Path

    def _do():
        dst = str(Path(tmp_root) / demo_key)
        Path(dst).mkdir(parents=True, exist_ok=True)
        layout = window._run_demo_generator(demo_key, dst)
        target_app, _ = window.DEMO_TARGETS[demo_key]
        window._on_nav_selected(target_app)
        widget = window._screens.get(target_app)
        if widget is not None:
            window._apply_demo_to_screen(widget, layout)
    return _do


def _sidebar_button(window, key: str):
    """Return the sidebar row for app ``key``.

    Matches on the ``navKey`` Qt property the sidebar stamps on every
    row, because app keys are load-bearing while the display names next
    to them are free to change. Falls back to a label match so a plain
    display name ("Mask") still works, and finally to the sidebar itself
    so a stale key degrades to a vague gesture rather than an exception.

    :param window: live MainWindow.
    :param key: an app key ("mask") or a sidebar label ("Mask").
    """
    from PySide6.QtWidgets import QPushButton
    rows = window._sidebar.findChildren(QPushButton)
    for btn in rows:
        if btn.property("navKey") == key:
            return btn
    for btn in rows:
        if btn.text().strip().lower() == key.lower():
            return btn
    LOG.warning("tutorial: no sidebar row for %r — cursor will land on "
                  "the sidebar as a whole", key)
    return window._sidebar


def _menu_bar(window):
    return window.menuBar()


def _find_menu(window, title: str):
    """The menu-bar menu titled ``title``, ignoring ``&``, or ``None``.

    Found through ``bar.findChildren(QMenu)`` rather than by walking
    ``menuBar().actions()`` and calling ``QAction.menu()``. On PySide6 6.11
    the QMenu wrapper that reading returns is only valid while the QAction
    wrapper it came off is alive, so it goes stale the moment the function
    returns — and keeping the owners alive as attributes segfaults during
    the next event dispatch instead. ``findChildren`` hands back children
    the bar owns in C++, valid for as long as the window is.

    Delegates to :func:`spacr.qt.first_run.find_menu`, which is the same
    lookup; a second copy is a second thing to get wrong.
    """
    try:
        from ..first_run import find_menu
    except Exception:
        return None
    return find_menu(window, title)


def _menu_target(window, title: str):
    """Return ``(menubar, centre-of-the-<title>-menu)`` for a Step target.

    The point is computed from the menu bar's own action geometry rather
    than hard-coded, so a longer menu title upstream of it — or a
    different font scale — cannot leave the cursor pointing at blank
    chrome.

    The geometry is looked up through the menu's own ``menuAction()``, which
    the QMenu owns, rather than through an action plucked out of the bar's
    action list — see :func:`_find_menu` for why the latter does not
    survive.

    :returns: ``(menubar, (x, y))``, or ``(menubar, None)`` when no menu
        with that title exists (the cursor then aims at the bar centre).
    """
    mb = window.menuBar()
    menu = _find_menu(window, title)
    if menu is not None:
        rect = mb.actionGeometry(menu.menuAction())
        return (mb, (rect.center().x(), rect.center().y()))
    LOG.warning("tutorial: menu bar has no %r menu", title)
    return (mb, None)


def _find_button(screen, label: str):
    """Find a QPushButton on ``screen`` whose text matches ``label``.

    An exact (case-insensitive) match wins over a prefix match. Prefix
    alone is not enough: the Mask and Timelapse screens carry both "Run"
    and "Run preview", and child order put "Run preview" first — so a
    step narrating the real run pointed at the preview button instead.
    """
    from PySide6.QtWidgets import QPushButton
    if screen is None:
        return None
    wanted = label.strip().lower()
    prefix_hit = None
    for b in screen.findChildren(QPushButton):
        text = b.text().strip().lower()
        if text == wanted:
            return b
        if prefix_hit is None and text.startswith(wanted):
            prefix_hit = b
    return prefix_hit


# ---------------------------------------------------------------------------
# Home tour
# ---------------------------------------------------------------------------

def _build_home_steps(window) -> List[Step]:
    return [
        Step(
            "Welcome to spaCR — a modern desktop application "
            "for spatial single-cell analysis of microscopy data.",
            action=_go_home(window),
            target=(_sidebar_button(window, "__home__"), None),
            highlight=_sidebar_button(window, "__home__"),
            show_pointer=True,
            hold_ms=400,
        ),
        Step(
            "The left sidebar gives you quick access to every "
            "pipeline in spaCR — grouped into Core, Analysis, "
            "Cellpose, and Sequencing.",
            target=(window._sidebar, None),
            highlight=window._sidebar,
            hold_ms=300,
        ),
        Step(
            "The home page shows every app as a large clickable "
            "tile. Hovering makes each tile pop, and clicking "
            "opens the module.",
            target=(window._stack, None),
            highlight=window._stack,
            hold_ms=500,
        ),
        Step(
            "Every pipeline in spaCR ships with a one-click "
            "synthetic demo dataset. From the Demos menu you can "
            "generate a working example for any module.",
            action=lambda: _open_demos_menu(window),
            target=_menu_target(window, "Demos"),
            highlight=_menu_bar(window),
            hold_ms=800,
        ),
        Step(
            "Let's jump into the mask module to see it in action.",
            action=_nav_to(window, "mask"),
            target=(_sidebar_button(window, "mask"), None),
            highlight=_sidebar_button(window, "mask"),
            show_pointer=True,
            hold_ms=400,
        ),
    ]


def _open_demos_menu(window):
    """Locate the Demos menu for the narration beat about it.

    Popping the menu up for real would grab input for the rest of the
    render, so this only resolves it — and returns what it found, so a
    rename of the menu is detectable instead of silently turning the
    step into a no-op.

    Resolved through :func:`_find_menu`, which reaches the QMenu as a C++
    child of the menu bar. The obvious reading — walk ``menuBar().actions()``
    and call ``QAction.menu()`` — hands back a wrapper that dies with the
    action wrapper it came off, so it was already invalid by the time this
    returned.

    :returns: the QMenu titled "Demos", or ``None``.
    """
    menu = _find_menu(window, "Demos")
    if menu is None:
        LOG.warning("tutorial: no Demos menu on the menu bar")
    return menu


# ---------------------------------------------------------------------------
# Mask module tutorial
# ---------------------------------------------------------------------------

def _build_mask_steps(window) -> List[Step]:
    tmp_root = _tutorial_scratch("mask")
    screen_ref: List[Any] = [None]

    def _capture_screen():
        screen_ref[0] = window._screens.get("mask")

    return [
        Step(
            "This is the mask module — spaCR's front door for "
            "segmenting cells, nuclei, and pathogens using "
            "Cellpose.",
            action=_nav_to(window, "mask"),
            target=(_sidebar_button(window, "mask"), None),
            highlight=_sidebar_button(window, "mask"),
            show_pointer=True,
            hold_ms=400,
        ),
        Step(
            "Rather than pointing you at your own data, we'll load "
            "a synthetic demo from the Demos menu — this generates "
            "a small dataset in the correct format and fills in "
            "every setting.",
            action=lambda: (_load_demo(window, "mask", tmp_root)(),
                             _capture_screen()),
            target=_menu_target(window, "Demos"),
            highlight=_menu_bar(window),
            show_pointer=True,
            hold_ms=800,
        ),
        Step(
            "The settings panel on the left is now populated. "
            "Notice the source folder, the channel layout, "
            "and each object's Cellpose model — cyto for cells, "
            "nuclei for nuclei.",
            target=(lambda: _settings_panel(screen_ref[0]), None),
            highlight=lambda: _settings_panel(screen_ref[0]),
            hold_ms=400,
        ),
        Step(
            "The console on the right will stream every log "
            "record — from spaCR itself, from Cellpose, and from "
            "any warnings raised during the run.",
            target=(lambda: _console_panel(screen_ref[0]), None),
            highlight=lambda: _console_panel(screen_ref[0]),
            hold_ms=400,
        ),
        Step(
            "When you hit Run, spaCR converts your images to a "
            "Yokogawa-style stack, normalises each channel, and "
            "then hands each field to Cellpose to segment.",
            target=(lambda: _find_button(screen_ref[0], "Run"), None),
            highlight=lambda: _find_button(screen_ref[0], "Run"),
            hold_ms=600,
        ),
        Step(
            "Once the run finishes, the masks land in a masks "
            "subfolder next to your images, ready to feed into "
            "the measure module.",
            hold_ms=400,
        ),
    ]


# ---------------------------------------------------------------------------
# Measure module tutorial
# ---------------------------------------------------------------------------

def _build_measure_steps(window) -> List[Step]:
    tmp_root = _tutorial_scratch("measure")
    screen_ref: List[Any] = [None]

    def _capture():
        screen_ref[0] = window._screens.get("measure")

    return [
        Step(
            "The measure module extracts single-object features "
            "from your segmented images — intensity, morphology, "
            "co-localization, texture, and radial distribution.",
            action=_nav_to(window, "measure"),
            target=(_sidebar_button(window, "measure"), None),
            highlight=_sidebar_button(window, "measure"),
            show_pointer=True,
            hold_ms=400,
        ),
        Step(
            "Load the measure demo — this ships pre-built masks "
            "and a measurements database seeded with the correct "
            "schema.",
            action=lambda: (_load_demo(window, "measure", tmp_root)(),
                             _capture()),
            target=_menu_target(window, "Demos"),
            highlight=_menu_bar(window),
            show_pointer=True,
            hold_ms=800,
        ),
        Step(
            "The demo populates the source folder, the channel "
            "layout, and every measurement toggle. The cell, "
            "nucleus, and pathogen channels can be tuned "
            "independently.",
            target=(lambda: _settings_panel(screen_ref[0]), None),
            highlight=lambda: _settings_panel(screen_ref[0]),
            hold_ms=500,
        ),
        Step(
            "Optionally, measure will also crop each object into "
            "a PNG for classify — enable Save PNG and pick a size.",
            target=(lambda: _settings_panel(screen_ref[0]), None),
            highlight=lambda: _settings_panel(screen_ref[0]),
            hold_ms=400,
        ),
        Step(
            "Hitting Run walks every mask, computes features, and "
            "appends rows to measurements.db — one row per object, "
            "per timepoint if you're doing timelapse.",
            target=(lambda: _find_button(screen_ref[0], "Run"), None),
            highlight=lambda: _find_button(screen_ref[0], "Run"),
            hold_ms=500,
        ),
    ]


# ---------------------------------------------------------------------------
# Crop module tutorial
# ---------------------------------------------------------------------------

def _build_crop_steps(window) -> List[Step]:
    tmp_root = _tutorial_scratch("crop")
    screen_ref: List[Any] = [None]

    def _capture():
        screen_ref[0] = window._screens.get("measure")

    return [
        Step(
            "The crop demo lands you in the measure module — "
            "in spaCR, cropping is one of the outputs of measure, "
            "not a standalone step.",
            action=_nav_to(window, "measure"),
            target=(_sidebar_button(window, "measure"), None),
            highlight=_sidebar_button(window, "measure"),
            show_pointer=True,
            hold_ms=400,
        ),
        Step(
            "Load the crop demo — this pre-fills a set of "
            "settings that turn measure into a pure crop-and-save "
            "job.",
            action=lambda: (_load_demo(window, "crop", tmp_root)(),
                             _capture()),
            target=_menu_target(window, "Demos"),
            highlight=_menu_bar(window),
            show_pointer=True,
            hold_ms=800,
        ),
        Step(
            "Save PNG is on, PNG size is 64, and PNG dims picks "
            "which channels get baked into the crop. You'll get "
            "one folder of thumbnails per object type.",
            target=(lambda: _settings_panel(screen_ref[0]), None),
            highlight=lambda: _settings_panel(screen_ref[0]),
            hold_ms=500,
        ),
        Step(
            "The crops are what feed into classify — once you "
            "annotate them, you have a labelled training set for "
            "your own CNN.",
            hold_ms=400,
        ),
    ]


# ---------------------------------------------------------------------------
# Classify module tutorial — hosted in AnnotateScreen
# ---------------------------------------------------------------------------

def _build_classify_steps(window) -> List[Step]:
    tmp_root = _tutorial_scratch("classify")
    screen_ref: List[Any] = [None]

    def _capture():
        screen_ref[0] = window._screens.get("annotate")

    return [
        Step(
            "Classify starts in the annotate module — this is "
            "where you label the crops that measure produced, so "
            "that classify has a training set.",
            action=_nav_to(window, "annotate"),
            target=(_sidebar_button(window, "annotate"), None),
            highlight=_sidebar_button(window, "annotate"),
            show_pointer=True,
            hold_ms=400,
        ),
        Step(
            "Loading the classify demo generates a small folder "
            "of pre-labelled synthetic crops so we can see the "
            "labelling grid without needing real data.",
            action=lambda: (_load_demo(window, "classify", tmp_root)(),
                             _capture()),
            target=_menu_target(window, "Demos"),
            highlight=_menu_bar(window),
            show_pointer=True,
            hold_ms=1000,
        ),
        Step(
            "Each tile is a single-cell crop. Left-click cycles "
            "through class labels — none, one, two, and back to "
            "none — so you can label a whole plate very quickly.",
            target=(lambda: screen_ref[0], None),
            highlight=lambda: screen_ref[0],
            hold_ms=500,
        ),
        Step(
            "When you're done, the Train CV and Train XG buttons "
            "hand your annotations off to classify — either as a "
            "CNN or as an XGBoost model.",
            target=(lambda: _find_button(screen_ref[0], "Train CV"), None),
            highlight=lambda: _find_button(screen_ref[0], "Train CV"),
            hold_ms=500,
        ),
    ]


# ---------------------------------------------------------------------------
# Timelapse module tutorial — the standalone Timelapse module
# ---------------------------------------------------------------------------

def _build_timelapse_steps(window) -> List[Step]:
    tmp_root = _tutorial_scratch("timelapse")
    screen_ref: List[Any] = [None]

    def _capture():
        screen_ref[0] = window._screens.get("timelapse")

    return [
        Step(
            "spaCR handles timelapse natively — every module "
            "understands the T dimension in the Yokogawa filename "
            "convention.",
            action=_nav_to(window, "timelapse"),
            target=(_sidebar_button(window, "timelapse"), None),
            highlight=_sidebar_button(window, "timelapse"),
            show_pointer=True,
            hold_ms=400,
        ),
        Step(
            "Loading the timelapse demo generates eight frames "
            "per field. Every downstream module then handles "
            "tracking, motion, and per-frame analysis "
            "automatically.",
            action=lambda: (_load_demo(window, "timelapse", tmp_root)(),
                             _capture()),
            target=_menu_target(window, "Demos"),
            highlight=_menu_bar(window),
            show_pointer=True,
            hold_ms=800,
        ),
        Step(
            "The Timelapse tab in the settings panel holds the "
            "tracking knobs — which objects to link, the linking "
            "mode, and how far an object may travel between "
            "frames.",
            target=(lambda: _settings_panel(screen_ref[0]), None),
            highlight=lambda: _settings_panel(screen_ref[0]),
            hold_ms=500,
        ),
        Step(
            "Run will then generate a per-frame mask stack, and "
            "measure will produce a longitudinal database with "
            "one row per object per timepoint.",
            target=(lambda: _find_button(screen_ref[0], "Run"), None),
            highlight=lambda: _find_button(screen_ref[0], "Run"),
            hold_ms=400,
        ),
    ]


# ---------------------------------------------------------------------------
# Widget lookup helpers
# ---------------------------------------------------------------------------

def _settings_panel(screen):
    """Return the settings scroll area on ``screen``, or ``None``.

    Taking the first ``QScrollArea`` findChildren hands back is wrong:
    every AppScreen also holds the console's own scroll area
    ("ConsoleScroll"), and child order puts *that* one first — so the
    step narrating "the settings panel on the left" aimed the cursor at
    the console on the right instead. Console descendants are excluded,
    and of what remains the leftmost wins, which is the settings column
    by construction.
    """
    if screen is None:
        return None
    from PySide6.QtWidgets import QScrollArea
    console = _console_panel(screen)
    candidates = [
        w for w in screen.findChildren(QScrollArea)
        if not (console is not None and console.isAncestorOf(w))
    ]
    if not candidates:
        return None
    return min(candidates,
                 key=lambda w: w.mapTo(screen, w.rect().topLeft()).x())


def _console_panel(screen):
    if screen is None:
        return None
    return getattr(screen, "_console", None)


def _tutorial_scratch(name: str) -> str:
    """Per-tutorial scratch dir. Kept out of tmp so demos survive
    inspection after render finishes."""
    from pathlib import Path
    p = Path.home() / ".spacr" / "tutorial-scratch" / name
    p.mkdir(parents=True, exist_ok=True)
    return str(p)
