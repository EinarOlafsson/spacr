"""Per-app tutorial Step sequences.

Each `_build_<app>_steps(window)` returns a list of engine.Step. The
engine handles narration synthesis + capture + mux; these functions
only choose the narration text, the UI actions, and the cursor
targets.

Every script exercises the same core motion: land on the app, load
a synthetic demo dataset (via the Demos menu we shipped), highlight
the interesting parts of the settings form, then click Run.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional

from .engine import Step

AVAILABLE_TUTORIALS = [
    "home", "mask", "measure", "crop", "classify", "timelapse",
]


def build_steps(app_key: str, window) -> List[Step]:
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
    for btn in window._sidebar.findChildren(type(window._sidebar)):
        pass  # not enough info without deeper introspection
    from PySide6.QtWidgets import QPushButton
    for btn in window._sidebar.findChildren(QPushButton):
        if btn.text().strip().lower() == key.lower():
            return btn
    return window._sidebar


def _menu_bar(window):
    return window.menuBar()


def _find_button(screen, label: str):
    """Best-effort: find a QPushButton on `screen` whose text starts
    with `label` (case-insensitive)."""
    from PySide6.QtWidgets import QPushButton
    if screen is None:
        return None
    for b in screen.findChildren(QPushButton):
        if b.text().strip().lower().startswith(label.lower()):
            return b
    return None


# ---------------------------------------------------------------------------
# Home tour
# ---------------------------------------------------------------------------

def _build_home_steps(window) -> List[Step]:
    return [
        Step(
            "Welcome to spaCR — a modern desktop application "
            "for spatial single-cell analysis of microscopy data.",
            action=_go_home(window),
            target=(window._sidebar, (100, 40)),
            hold_ms=400,
        ),
        Step(
            "The left sidebar gives you quick access to every "
            "pipeline in spaCR — grouped into Core, Analysis, "
            "Cellpose, and Sequencing.",
            target=(window._sidebar, (100, 200)),
            hold_ms=300,
        ),
        Step(
            "The home page shows every app as a large clickable "
            "tile. Hovering makes each tile pop, and clicking "
            "opens the module.",
            target=(window._stack, (960, 400)),
            hold_ms=500,
        ),
        Step(
            "Every pipeline in spaCR ships with a one-click "
            "synthetic demo dataset. From the Demos menu you can "
            "generate a working example for any module.",
            action=lambda: _open_demos_menu(window),
            target=(window.menuBar(), (170, 15)),
            hold_ms=800,
        ),
        Step(
            "Let's jump into the mask module to see it in action.",
            action=_nav_to(window, "mask"),
            target=(window._sidebar, (100, 250)),
            hold_ms=400,
        ),
    ]


def _open_demos_menu(window):
    for act in window.menuBar().actions():
        if act.text().replace("&", "") == "Demos":
            # Just show its status tip - actually opening the menu
            # would block. Trigger the first action so we at least
            # show its effect. Actually we want to just idle here.
            return
    return


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
            target=(window._sidebar, (100, 250)),
            hold_ms=400,
        ),
        Step(
            "Rather than pointing you at your own data, we'll load "
            "a synthetic demo from the Demos menu — this generates "
            "a small dataset in the correct format and fills in "
            "every setting.",
            action=lambda: (_load_demo(window, "mask", tmp_root)(),
                             _capture_screen()),
            hold_ms=800,
        ),
        Step(
            "The settings panel on the left is now populated. "
            "Notice the source folder, the channel layout, "
            "and each object's Cellpose model — cyto for cells, "
            "nuclei for nuclei.",
            target=(_settings_panel(screen_ref[0]), None),
            hold_ms=400,
        ),
        Step(
            "The console on the right will stream every log "
            "record — from spaCR itself, from Cellpose, and from "
            "any warnings raised during the run.",
            target=(_console_panel(screen_ref[0]), None),
            hold_ms=400,
        ),
        Step(
            "When you hit Run, spaCR converts your images to a "
            "Yokogawa-style stack, normalises each channel, and "
            "then hands each field to Cellpose to segment.",
            target=(_find_button(screen_ref[0], "Run"), None),
            highlight=_find_button(screen_ref[0], "Run"),
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
            target=(window._sidebar, (100, 280)),
            hold_ms=400,
        ),
        Step(
            "Load the measure demo — this ships pre-built masks "
            "and a measurements database seeded with the correct "
            "schema.",
            action=lambda: (_load_demo(window, "measure", tmp_root)(),
                             _capture()),
            hold_ms=800,
        ),
        Step(
            "The demo populates the source folder, the channel "
            "layout, and every measurement toggle. The cell, "
            "nucleus, and pathogen channels can be tuned "
            "independently.",
            target=(_settings_panel(screen_ref[0]), None),
            hold_ms=500,
        ),
        Step(
            "Optionally, measure will also crop each object into "
            "a PNG for classify — enable Save PNG and pick a size.",
            hold_ms=400,
        ),
        Step(
            "Hitting Run walks every mask, computes features, and "
            "appends rows to measurements.db — one row per object, "
            "per timepoint if you're doing timelapse.",
            target=(_find_button(screen_ref[0], "Run"), None),
            highlight=_find_button(screen_ref[0], "Run"),
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
            hold_ms=400,
        ),
        Step(
            "Load the crop demo — this pre-fills a set of "
            "settings that turn measure into a pure crop-and-save "
            "job.",
            action=lambda: (_load_demo(window, "crop", tmp_root)(),
                             _capture()),
            hold_ms=800,
        ),
        Step(
            "Save PNG is on, PNG size is 64, and PNG dims picks "
            "which channels get baked into the crop. You'll get "
            "one folder of thumbnails per object type.",
            target=(_settings_panel(screen_ref[0]), None),
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
            target=(window._sidebar, (100, 300)),
            hold_ms=400,
        ),
        Step(
            "Loading the classify demo generates a small folder "
            "of pre-labelled synthetic crops so we can see the "
            "labelling grid without needing real data.",
            action=lambda: (_load_demo(window, "classify", tmp_root)(),
                             _capture()),
            hold_ms=1000,
        ),
        Step(
            "Each tile is a single-cell crop. Left-click cycles "
            "through class labels — none, one, two, and back to "
            "none — so you can label a whole plate very quickly.",
            hold_ms=500,
        ),
        Step(
            "When you're done, the Train CV and Train XG buttons "
            "hand your annotations off to classify — either as a "
            "CNN or as an XGBoost model.",
            target=(_find_button(screen_ref[0], "Train"), None),
            highlight=_find_button(screen_ref[0], "Train"),
            hold_ms=500,
        ),
    ]


# ---------------------------------------------------------------------------
# Timelapse module tutorial — hosted in the mask module with timelapse on
# ---------------------------------------------------------------------------

def _build_timelapse_steps(window) -> List[Step]:
    tmp_root = _tutorial_scratch("timelapse")
    screen_ref: List[Any] = [None]

    def _capture():
        screen_ref[0] = window._screens.get("mask")

    return [
        Step(
            "spaCR handles timelapse natively — every module "
            "understands the T dimension in the Yokogawa filename "
            "convention.",
            action=_nav_to(window, "mask"),
            hold_ms=400,
        ),
        Step(
            "Loading the timelapse demo generates eight frames "
            "per field and turns on the timelapse setting. Every "
            "downstream module then handles tracking, motion, and "
            "per-frame analysis automatically.",
            action=lambda: (_load_demo(window, "timelapse", tmp_root)(),
                             _capture()),
            hold_ms=800,
        ),
        Step(
            "Notice the timelapse toggle in the settings panel — "
            "flipping this on tells every downstream module to "
            "treat each field as a temporal stack rather than "
            "independent images.",
            target=(_settings_panel(screen_ref[0]), None),
            hold_ms=500,
        ),
        Step(
            "Run will then generate a per-frame mask stack, and "
            "measure will produce a longitudinal database with "
            "one row per object per timepoint.",
            target=(_find_button(screen_ref[0], "Run"), None),
            hold_ms=400,
        ),
    ]


# ---------------------------------------------------------------------------
# Widget lookup helpers
# ---------------------------------------------------------------------------

def _settings_panel(screen):
    """Return the settings scroll area if we can find it."""
    if screen is None:
        return None
    from PySide6.QtWidgets import QScrollArea
    for w in screen.findChildren(QScrollArea):
        return w
    return None


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
