"""Per-app tutorial Step sequences.

Each `_build_<app>_steps(window)` returns a list of engine.Step. The
engine handles narration synthesis + capture + mux; these functions
only choose the narration text, the UI actions, and the cursor
targets.

Every script follows the same core sequence: open the application, load
a synthetic dataset through the Demos menu, highlight the relevant settings,
and start the run.

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
from typing import Any, List

from .engine import Step

LOG = logging.getLogger("spacr.qt.tutorial")

AVAILABLE_TUTORIALS = [
    "home", "mask", "measure", "crop", "classify", "timelapse",
    "map_barcodes", "regression",
]


def build_steps(app_key: str, window) -> List[Step]:
    """Return the tutorial Step list for ``app_key`` bound to ``window``.

    :param app_key: one of :data:`AVAILABLE_TUTORIALS`.
    :param window: live MainWindow instance the steps drive.
    :raises ValueError: when ``app_key`` is not a known tutorial.
    """
    if app_key == "home":
        return _build_home_steps(window)
    if app_key == "mask":
        return _build_mask_steps(window)
    if app_key == "measure":
        return _build_measure_steps(window)
    if app_key == "crop":
        return _build_crop_steps(window)
    if app_key == "classify":
        return _build_classify_steps(window)
    if app_key == "timelapse":
        return _build_timelapse_steps(window)
    if app_key == "map_barcodes":
        return _build_map_barcodes_steps(window)
    if app_key == "regression":
        return _build_regression_steps(window)
    raise ValueError(f"unknown tutorial: {app_key}. "
                       f"Choose from {AVAILABLE_TUTORIALS}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _go_home(window):
    def _do():
        """Navigate Home."""
        window._on_nav_selected("__home__")
    return _do


def _nav_to(window, app_key: str):
    def _do():
        """Navigate to the bound module."""
        window._on_nav_selected(app_key)
    return _do


def _load_demo(window, demo_key: str, tmp_root: str):
    """Bypass the file dialog — call the internals directly with a
    scratch destination. Same code path the Demos menu uses."""
    from pathlib import Path

    def _do():
        """Make the demo folder and point the screen at it."""
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
    # Demos is a Help submenu and MainWindow deliberately retains the QMenu
    # wrapper that owns its actions. Qt may reparent that submenu when it is
    # inserted under Help, so it is not reliably returned by the menu bar's
    # child walk on every PySide6 version. Use the retained owner first.
    if title == "Demos":
        menu = getattr(window, "_demo_menu", None)
        try:
            # ``_demo_menu`` is the semantic owner, independent of the
            # currently selected language. Comparing its rendered title to
            # the English lookup key made the tutorial lose the menu after a
            # retranslation within the same application session.
            if menu is not None:
                menu.title()  # prove that the retained Qt wrapper is live
                return menu
        except RuntimeError:
            pass
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
        if rect.isValid() and rect.width():
            return (mb, (rect.center().x(), rect.center().y()))
        # A SUBMENU HAS NO PLACE ON THE BAR. Demos moved under Help on
        # 2026-08-23, so its own geometry is empty and the cursor would aim
        # at (0, 0). Point at the top-level menu you actually click to
        # reach it, which is where a user's hand goes.
        parent = _top_level_menu_containing(window, menu)
        if parent is not None:
            rect = mb.actionGeometry(parent.menuAction())
            return (mb, (rect.center().x(), rect.center().y()))
    if title == "Demos":
        # The target users click is Help, not the submenu itself. During a
        # long tutorial-test session Qt can briefly detach the submenu action
        # while pages are being rebuilt even though ``_demo_menu`` remains
        # live. Target the stable top-level Help action directly in that
        # state. The final-action fallback is language-independent; spaCR and
        # Help are the only top-level menus in the current compact bar.
        actions = list(mb.actions())
        help_action = next(
            (action for action in actions
             if action.text().replace("&", "") == "Help"),
            actions[-1] if actions else None,
        )
        if help_action is not None:
            rect = mb.actionGeometry(help_action)
            if rect.isValid() and rect.width():
                return (mb, (rect.center().x(), rect.center().y()))
    LOG.warning("tutorial: menu bar has no %r menu", title)
    return (mb, None)


def _top_level_menu_containing(window, menu):
    """The menu-bar menu that ``menu`` is a submenu of, or None."""
    mb = window.menuBar()
    try:
        # A real rectangle on the bar proves this is already a top-level
        # menu.  Check that first: during long PySide test sessions a
        # released submenu wrapper can be recycled and acquire the same
        # rendered title as an unrelated menu, so title-only relation scans
        # must never be allowed to reclassify a bar menu as its own child.
        rect = mb.actionGeometry(menu.menuAction())
        if rect.isValid() and rect.width():
            return None
    except RuntimeError:
        return None
    try:
        target_title = menu.title().replace("&", "")
    except RuntimeError:
        return None
    for action in mb.actions():
        # Retrieve the bar-owned QMenu wrapper through the same stable lookup
        # used elsewhere. Returning ``action.menu()`` directly can leave the
        # caller with a deleted temporary PySide wrapper after ``action`` is
        # released.
        top = _find_menu(window, action.text().replace("&", ""))
        if top is None:
            continue
        for entry in top.actions():
            try:
                nested = entry.menu()
                # Compare semantic titles only. Temporary PySide wrappers can
                # be recycled after Qt releases them, so Python object
                # identity is not a safe submenu relation across event-loop
                # turns.
                if (nested is not None and
                        nested.title().replace("&", "") == target_title):
                    return top
            except RuntimeError:
                continue
    return None


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
            "The sidebar groups modules by scientific role: Core, Data, "
            "Segmentation models, Results and quality control, Explore, "
            "Assays, and Design.",
            target=(window._sidebar, None),
            highlight=window._sidebar,
            hold_ms=300,
        ),
        Step(
            "Home provides one tile for each primary module. Related "
            "workflows that were consolidated into a module are available "
            "from icon buttons in that module's header.",
            target=(window._stack, None),
            highlight=window._stack,
            hold_ms=500,
        ),
        Step(
            "The Help menu contains Demos for selected core workflows. "
            "Each entry generates a small synthetic dataset, writes its "
            "settings, and opens the corresponding module.",
            action=lambda: _open_demos_menu(window),
            target=_menu_target(window, "Demos"),
            highlight=_menu_bar(window),
            hold_ms=800,
        ),
        Step(
            "Open Mask to examine the first image-analysis stage.",
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
        """Remember the screen this step is about, once it exists.

        The screen is built by the step BEFORE this one, so it cannot be looked
        up when the script is written -- only when the step runs.
        """
        screen_ref[0] = window._screens.get("mask")

    return [
        Step(
            "Mask generates per-object masks for cells, nuclei, pathogens, "
            "and configured organelles from microscopy images using Cellpose "
            "and, for organelles, classical or custom-model methods.",
            action=_nav_to(window, "mask"),
            target=(_sidebar_button(window, "mask"), None),
            highlight=_sidebar_button(window, "mask"),
            show_pointer=True,
            hold_ms=400,
        ),
        Step(
            "Load Mask demo from Help, Demos. The command generates a "
            "small synthetic acquisition, writes a compatible settings "
            "file, and applies those settings to this screen.",
            action=lambda: (_load_demo(window, "mask", tmp_root)(),
                             _capture_screen()),
            target=_menu_target(window, "Demos"),
            highlight=_menu_bar(window),
            show_pointer=True,
            hold_ms=800,
        ),
        Step(
            "The settings panel on the left is now populated. "
            "Verify the source, metadata parser, channel assignments, object "
            "diameters, and each object's segmentation method. Confirm the "
            "model or checkpoint wherever that method requires one.",
            target=(lambda: _settings_panel(screen_ref[0]), None),
            highlight=lambda: _settings_panel(screen_ref[0]),
            hold_ms=400,
        ),
        Step(
            "The console on the right records validation, segmentation "
            "progress, warnings, and output paths. Resolve validation "
            "errors before interpreting a completed run.",
            target=(lambda: _console_panel(screen_ref[0]), None),
            highlight=lambda: _console_panel(screen_ref[0]),
            hold_ms=400,
        ),
        Step(
            "Selecting Run parses the acquisition metadata, assembles each "
            "field and channel stack, applies the configured preprocessing, "
            "and generates the requested object masks.",
            target=(lambda: _find_button(screen_ref[0], "Run"), None),
            highlight=lambda: _find_button(screen_ref[0], "Run"),
            hold_ms=600,
        ),
        Step(
            "The run writes label images under masks and merged image-mask "
            "arrays under merged. Review segmentation quality before using "
            "those arrays in Measure.",
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
        """Remember the screen this step is about, once it exists.

        The screen is built by the step BEFORE this one, so it cannot be looked
        up when the script is written -- only when the step runs.
        """
        screen_ref[0] = window._screens.get("measure")

    return [
        Step(
            "Measure computes per-object intensity, morphology, texture, "
            "colocalization, radial-distribution, and spatial features from "
            "merged image-mask arrays.",
            action=_nav_to(window, "measure"),
            target=(_sidebar_button(window, "measure"), None),
            highlight=_sidebar_button(window, "measure"),
            show_pointer=True,
            hold_ms=400,
        ),
        Step(
            "Load Measure demo from Help, Demos. It provides representative "
            "merged arrays and applies a valid measurement configuration.",
            action=lambda: (_load_demo(window, "measure", tmp_root)(),
                             _capture()),
            target=_menu_target(window, "Demos"),
            highlight=_menu_bar(window),
            show_pointer=True,
            hold_ms=800,
        ),
        Step(
            "The demo populates the source folder, the channel "
            "layout, object tables, and measurement controls. Confirm each "
            "cell, nucleus, pathogen, cytoplasm, and organelle role before "
            "measuring intensities or relationships.",
            target=(lambda: _settings_panel(screen_ref[0]), None),
            highlight=lambda: _settings_panel(screen_ref[0]),
            hold_ms=500,
        ),
        Step(
            "Measure can optionally export object-centred PNG crops for "
            "annotation or image classification. Enable crop export, select "
            "the object types and channels, and record the crop dimensions.",
            target=(lambda: _settings_panel(screen_ref[0]), None),
            highlight=lambda: _settings_panel(screen_ref[0]),
            hold_ms=400,
        ),
        Step(
            "Selecting Run computes the requested features and writes one "
            "database row per measured object, with time-indexed rows when "
            "the source contains tracked frames.",
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
        """Remember the screen this step is about, once it exists.

        The screen is built by the step BEFORE this one, so it cannot be looked
        up when the script is written -- only when the step runs.
        """
        screen_ref[0] = window._screens.get("measure")

    return [
        Step(
            "Crop export is part of Measure rather than a standalone module. "
            "The exported images retain object identifiers that connect "
            "them to measurements and annotations.",
            action=_nav_to(window, "measure"),
            target=(_sidebar_button(window, "measure"), None),
            highlight=_sidebar_button(window, "measure"),
            show_pointer=True,
            hold_ms=400,
        ),
        Step(
            "Load Crop demo from Help, Demos. It configures Measure to "
            "export object crops from the supplied merged arrays.",
            action=lambda: (_load_demo(window, "crop", tmp_root)(),
                             _capture()),
            target=_menu_target(window, "Demos"),
            highlight=_menu_bar(window),
            show_pointer=True,
            hold_ms=800,
        ),
        Step(
            "The crop controls specify the object types, dimensions, channel "
            "composition, and optional contextual dilation. Measure writes "
            "one identified crop collection per selected object type.",
            target=(lambda: _settings_panel(screen_ref[0]), None),
            highlight=lambda: _settings_panel(screen_ref[0]),
            hold_ms=500,
        ),
        Step(
            "After review and annotation, these crops can train the Computer "
            "Vision workflow in Classify. Preserve plate or well groups so "
            "related objects cannot leak across validation splits.",
            hold_ms=400,
        ),
    ]


# ---------------------------------------------------------------------------
# Classify module tutorial — hosted in AnnotateScreen
# ---------------------------------------------------------------------------

def _build_classify_steps(window) -> List[Step]:
    tmp_root = _tutorial_scratch("classify")
    screen_ref: List[Any] = [None]

    classify_ref: List[Any] = [None]

    def _capture():
        """Remember the screen this step is about, once it exists.

        The screen is built by the step BEFORE this one, so it cannot be looked
        up when the script is written -- only when the step runs.
        """
        screen_ref[0] = window._screens.get("annotate")

    def _capture_classify():
        """Remember Classify's screen, once this script has opened it.

        CAPTURED RATHER THAN LOOKED UP, and the difference is not academic:
        reading `window._screens` inside the target would resolve as soon as
        ANY earlier lesson had opened Classify, and the engine's contract is
        that a deferred target is dead until its own step has run.
        """
        classify_ref[0] = window._screens.get("classify_merged")

    return [
        Step(
            "Image classification requires reviewed labels. Begin in "
            "Annotate to assign classes to the object crops produced by "
            "Measure, then hand the labelled project to Classify.",
            action=_nav_to(window, "annotate"),
            target=(_sidebar_button(window, "annotate"), None),
            highlight=_sidebar_button(window, "annotate"),
            show_pointer=True,
            hold_ms=400,
        ),
        Step(
            "Load Classify demo from Help, Demos. It generates a small "
            "synthetic crop collection with example labels and opens the "
            "annotation grid.",
            action=lambda: (_load_demo(window, "classify", tmp_root)(),
                             _capture()),
            target=_menu_target(window, "Demos"),
            highlight=_menu_bar(window),
            show_pointer=True,
            hold_ms=1000,
        ),
        Step(
            "Each tile represents one identified object crop. Assign the "
            "reviewed class with the annotation controls and inspect image "
            "context before resolving ambiguous examples.",
            target=(lambda: screen_ref[0], None),
            highlight=lambda: screen_ref[0],
            hold_ms=500,
        ),
        # ONE BUTTON NOW, NOT TWO. "Train CV" and "Train XG" were merged
        # into a single `Train...` with a menu, and this step was not moved
        # with them -- so its target resolved to None and the tutorial
        # pointed at nothing. The narration named both old buttons too.
        Step(
            "Train opens a menu with the two destinations: on the images "
            "trains a Torch CNN or Transformer on the crops themselves, on "
            "the measured features trains XGBoost on what Measure recorded. "
            "Both open in the consolidated Classify module.",
            target=(lambda: _find_button(screen_ref[0], "Train"), None),
            highlight=lambda: _find_button(screen_ref[0], "Train"),
            hold_ms=500,
        ),
        # AND THEN IT OPENS CLASSIFY. Until 2026-09-04 the tutorial called
        # "classify" stopped here, at Annotate's Train button, having said
        # "both open in the consolidated Classify module" and never opened
        # it. That is the stale module boundary instruction 358 was filed
        # about: a polished lesson teaching a structure the application no
        # longer has.
        Step(
            "Classify is where the training runs. The settings column carries "
            "the model, the split and the training schedule; the actions row "
            "runs it and the console below reports each epoch.",
            action=lambda: (_nav_to(window, "classify_merged")(),
                             _capture_classify()),
            target=(_sidebar_button(window, "classify_merged"), None),
            highlight=_sidebar_button(window, "classify_merged"),
            show_pointer=True,
            hold_ms=700,
        ),
        # FIVE MODULES WERE FOLDED IN HERE, and instruction 358 asks that each
        # one be named and located rather than left for the reader to find.
        # The specialist lessons that still explain them accurately are kept
        # and pointed at rather than re-narrated.
        Step(
            "Five modules that used to be their own tiles now live on this "
            "masthead. Classifier Evaluation judges the trained model, "
            "Explain CV asks which measured features it keyed on, and "
            "Activation Maps shows where in the image it looked.",
            target=(lambda: _fold_button(classify_ref[0],
                                         "classifier_evaluation"), None),
            highlight=lambda: _fold_button(classify_ref[0],
                                           "classifier_evaluation"),
            show_pointer=True,
            hold_ms=900,
        ),
        Step(
            "Training Runs compares two runs against each other, and Feature "
            "Explorer ranks measured features before anything is trained. "
            "Both are appended to the reading sequence rather than inserted "
            "into it, because neither is a step in it.",
            target=(lambda: _fold_button(classify_ref[0],
                                         "train_compare"), None),
            highlight=lambda: _fold_button(classify_ref[0],
                                           "train_compare"),
            show_pointer=True,
            hold_ms=900,
        ),
    ]



# ---------------------------------------------------------------------------
# Map Barcodes tutorial — the one Core module that reads sequencing, not images
# ---------------------------------------------------------------------------
# THE ODD ONE OUT, and the tutorial has to say so early. Every other Core
# module takes microscopy; this one takes FASTQ and produces the table that
# tells Regression which well got which perturbation. A reader who arrives
# expecting images needs that said before anything else.

def _build_map_barcodes_steps(window) -> List[Step]:
    tmp_root = _tutorial_scratch("map_barcodes")
    screen_ref: List[Any] = [None]

    def _capture():
        """Remember the screen, once the step before this one has built it."""
        screen_ref[0] = window._screens.get("map_barcodes")

    return [
        Step(
            "Map Barcodes is the one Core module that reads sequencing rather "
            "than microscopy. It takes the FASTQ from a pooled screen and "
            "works out which perturbation landed in which well, which is what "
            "lets a later regression attribute a phenotype to a gene.",
            action=_nav_to(window, "map_barcodes"),
            target=(_sidebar_button(window, "map_barcodes"), None),
            highlight=_sidebar_button(window, "map_barcodes"),
            show_pointer=True,
            hold_ms=700,
        ),
        Step(
            "Load Map Barcodes demo from Help, Demos. It writes a small FASTQ "
            "and the barcode references that go with it, then points the "
            "module at them, so the run below is real work on real reads "
            "rather than a walkthrough of an empty form.",
            action=lambda: (_load_demo(window, "map_barcodes", tmp_root)(),
                             _capture()),
            target=_menu_target(window, "Demos"),
            highlight=_menu_bar(window),
            show_pointer=True,
            hold_ms=1000,
        ),
        Step(
            "Sequencing Input names the reads. Barcode References names the "
            "three things a read has to be resolved against: the row, the "
            "column, and the guide library. Read Parsing is where the layout "
            "of the read itself is described, and it is the setting most "
            "worth checking before a long run.",
            target=(lambda: _settings_panel(screen_ref[0]), None),
            highlight=lambda: _settings_panel(screen_ref[0]),
            hold_ms=900,
        ),
        Step(
            "Run counts every read that resolves to a row, a column and a "
            "guide, and writes one row per well. The console reports reads "
            "that matched nothing, and that number is the one to look at "
            "first: a parse that is subtly wrong usually fails cleanly rather "
            "than producing plausible nonsense.",
            target=(lambda: _find_button(screen_ref[0], "Run"), None),
            highlight=lambda: _find_button(screen_ref[0], "Run"),
            show_pointer=True,
            hold_ms=800,
        ),
    ]


# ---------------------------------------------------------------------------
# Regression tutorial — the module that consumes what everything else produces
# ---------------------------------------------------------------------------
# NO DEMO OF ITS OWN, and that is honest rather than a gap: regression needs a
# measured screen AND a barcode mapping, so its live data is the OUTPUT of the
# two tutorials before it. The script says which ones and in what order,
# rather than pretending a synthetic single-module dataset would teach the
# thing that matters here.

def _build_regression_steps(window) -> List[Step]:
    screen_ref: List[Any] = [None]

    def _capture():
        """Remember the screen, once the step before this one has built it."""
        screen_ref[0] = window._screens.get("regression")

    return [
        Step(
            "Regression is where a screen becomes a result. It takes the "
            "per-object measurements from Measure and the well-level "
            "perturbations from Map Barcodes, and asks which perturbations "
            "moved the phenotype.",
            action=lambda: (_nav_to(window, "regression")(), _capture()),
            target=(_sidebar_button(window, "regression"), None),
            highlight=_sidebar_button(window, "regression"),
            show_pointer=True,
            hold_ms=700,
        ),
        Step(
            "This module has no demo dataset of its own, and that is the "
            "point rather than an omission: it needs a measured screen and a "
            "barcode mapping. Run the Measure and Map Barcodes lessons first "
            "and their outputs are what you drop here.",
            target=(lambda: _settings_panel(screen_ref[0]), None),
            highlight=lambda: _settings_panel(screen_ref[0]),
            hold_ms=900,
        ),
        Step(
            "Input Tables takes those two outputs. Controls and Filters names "
            "which wells are the reference the rest are judged against, and "
            "getting that wrong invalidates everything downstream of it.",
            target=(lambda: _settings_panel(screen_ref[0]), None),
            highlight=lambda: _settings_panel(screen_ref[0]),
            hold_ms=800,
        ),
        Step(
            "Plate and Batch Correction is not optional on a multi-plate "
            "screen. A plate effect is indistinguishable from a real one "
            "unless it is modelled, and a hit list built without it will "
            "rank whichever plate ran best.",
            target=(lambda: _settings_panel(screen_ref[0]), None),
            highlight=lambda: _settings_panel(screen_ref[0]),
            hold_ms=800,
        ),
        Step(
            "Model and Inference chooses the estimator; the Permutation Test "
            "beside it is what turns a coefficient into a claim, by asking "
            "how often a shuffled label produces an effect this large. Run "
            "writes the fitted table, and the Prediction Profiler and "
            "Investigate Hit modules read it from there.",
            target=(lambda: _find_button(screen_ref[0], "Run"), None),
            highlight=lambda: _find_button(screen_ref[0], "Run"),
            show_pointer=True,
            hold_ms=900,
        ),
    ]


# ---------------------------------------------------------------------------
# Timelapse tutorial — the tracking switch on Mask Generation
# ---------------------------------------------------------------------------
# Timelapse has no destination of its own: it is the mask pipeline with
# tracking turned on, so what is its own is a couple of settings CATEGORIES
# and a switch on the Mask masthead that reveals them. The script therefore
# lands on Mask, loads the timelapse demo -- whose settings file carries
# `timelapse=True`, which moves the switch as it is applied -- and narrates
# the switch and the categories it just revealed.

def _build_timelapse_steps(window) -> List[Step]:
    tmp_root = _tutorial_scratch("timelapse")
    screen_ref: List[Any] = [None]

    def _capture():
        """Remember the screen this step is about, once it exists.

        The screen is built by the step BEFORE this one, so it cannot be looked
        up when the script is written -- only when the step runs.
        """
        screen_ref[0] = window._screens.get("mask")

    return [
        Step(
            "Timelapse is a tracking mode within Mask. It generates a mask "
            "for each time point and links selected objects across frames "
            "using acquisition metadata that includes a time axis.",
            action=_nav_to(window, "mask"),
            target=(_sidebar_button(window, "mask"), None),
            highlight=_sidebar_button(window, "mask"),
            show_pointer=True,
            hold_ms=400,
        ),
        Step(
            "Load Timelapse demo from Help, Demos. It generates eight frames "
            "per field and applies settings with tracking enabled, which "
            "activates the Timelapse switch and reveals its categories.",
            action=lambda: (_load_demo(window, "timelapse", tmp_root)(),
                             _capture()),
            target=_menu_target(window, "Demos"),
            highlight=_menu_bar(window),
            show_pointer=True,
            hold_ms=800,
        ),
        Step(
            "The Timelapse icon in the Mask header is a persistent switch. "
            "It remains highlighted while tracking is enabled and hides the "
            "tracking categories when disabled.",
            target=(lambda: _fold_button(screen_ref[0], "timelapse"), None),
            highlight=lambda: _fold_button(screen_ref[0], "timelapse"),
            show_pointer=True,
            hold_ms=600,
        ),
        Step(
            "The revealed categories specify which objects are linked, the "
            "tracking backend, frame range, displacement and gap constraints, "
            "lifetime filters, diagnostics, and movie outputs.",
            target=(lambda: _settings_panel(screen_ref[0]), None),
            highlight=lambda: _settings_panel(screen_ref[0]),
            hold_ms=500,
        ),
        Step(
            "Run generates per-frame labels and stable track identities. "
            "Use the Motility workflow from Measure when calibrated trajectory "
            "and velocity measurements are required from the finished tracks.",
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


def _fold_button(screen, key: str):
    """The masthead switch a folded module hangs on ``screen``, or ``None``.

    Asked of the strip rather than found by caption: a fold button IS the
    module's icon and carries no text, so a text search cannot see it.
    Resolved at capture time like every other deferred target, because the
    strip is built with the screen and the screen is built by an earlier
    step.
    """
    if screen is None:
        return None
    strip = getattr(screen, "_fold_strip", None)
    if strip is None or not hasattr(strip, "button_for"):
        LOG.warning("tutorial: %r carries no fold strip — the switch step "
                      "will highlight nothing", key)
        return None
    button = strip.button_for(key)
    if button is None:
        LOG.warning("tutorial: no fold switch for %r on this screen", key)
    return button


def _tutorial_scratch(name: str) -> str:
    """Per-tutorial scratch dir. Kept out of tmp so demos survive
    inspection after render finishes."""
    from pathlib import Path
    p = Path.home() / ".spacr" / "tutorial-scratch" / name
    p.mkdir(parents=True, exist_ok=True)
    return str(p)
