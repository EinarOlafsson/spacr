"""
QApplication bootstrap + MainWindow.

`launch(argv)` is the public entry point called by `spacr-qt` and
`python -m spacr.qt`.
"""
from __future__ import annotations

import inspect
import logging
import os
import sys
import threading
import traceback
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, QSize, QThread, Signal
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QStackedWidget,
    QStatusBar,
    QStyle,
    QVBoxLayout,
    QWidget,
)


from . import iconset
# Nothing in this module uses a colour, a spacing or the palette API any
# more — the Home page, the sidebar QSS and `apply_preferences_to_app`
# each own their own. The import stayed behind after they moved out, and
# because it named `PALETTE` it fired the deprecation warning on every
# `import spacr.qt.app` for a value nobody read.
from .widgets.eliding import ElidingPushButton

LOG = logging.getLogger(__name__)

#: Published documentation root. `docs/source/conf.py` copies everything in
#: `docs/source/_extra/` verbatim into the site root (`html_extra_path`), so
#: `docs/source/_extra/tutorials/index.html` publishes at `<root>/tutorials/`
#: — plural. The Help menu pointed at the singular `/tutorial/` for its whole
#: life, which is a 404: no page has ever been served from that path, and
#: `spacr/qt/tutorial/` (the local MP4 renderer) is an unrelated module.
DOCS_BASE_URL = "https://einarolafsson.github.io/spacr"
DOCS_URL = f"{DOCS_BASE_URL}/index.html"
TUTORIALS_URL = f"{DOCS_BASE_URL}/tutorials/"


class _UpdateWorker(QThread):
    """Run one updater operation without blocking the GUI event loop.

    :param operation: stable operation name used in error messages.
    :param fn: zero-argument callable whose return value is emitted.
    :param parent: Qt owner; normally the :class:`MainWindow`.
    """

    succeeded = Signal(object)
    failed = Signal(str, str)

    def __init__(self, operation, fn, parent=None):
        super().__init__(parent)
        self.operation = str(operation)
        self._fn = fn

    def run(self) -> None:
        """Execute the operation and surface every exception."""
        try:
            result = self._fn()
        except Exception:
            details = traceback.format_exc()
            LOG.exception("Updater %s failed", self.operation)
            self.failed.emit(self.operation, details)
            return
        self.succeeded.emit(result)


class _PipelinePreloader:
    """Warms up the heavy pipeline modules so the first click on
    Mask/Measure/Classify/etc. doesn't stall the UI while
    torch/cellpose/pandas/etc. resolve.

    IMPORTANT: preloading runs on the MAIN (GUI) thread, one module
    per event-loop tick, NOT on a background daemon thread. Importing
    C-extension modules that initialise CUDA/GL (torch, cellpose) from
    a non-main thread concurrent with Qt's own GPU init is a classic
    cause of intermittent "Segmentation fault (core dumped)" at
    startup. Each import here blocks the event loop only briefly, and
    a QTimer tick between imports lets Qt process repaints/clicks so
    the UI stays responsive without the off-thread race.
    """

    _MODULES = (
        "spacr.core",
        "spacr.measure",
        "spacr.deep_spacr",
        "spacr.ml",
        "spacr.sequencing",
        "spacr.submodules",
        "spacr.spacr_cellpose",
    )

    def __init__(self):
        self._i = 0
        self._started = False

    def start(self) -> None:
        """Begin the main-thread import chain (no-op if already begun)."""
        if self._started:
            return
        self._started = True
        self._i = 0
        self._step()

    def _step(self) -> None:
        """Import the next module, then schedule the following one on the
        next event-loop tick."""
        from PySide6.QtCore import QTimer
        if self._i >= len(self._MODULES):
            return
        mod = self._MODULES[self._i]
        self._i += 1
        try:
            import importlib
            importlib.import_module(mod)
        except Exception:
            # Preloading is optional, but an import failure still belongs in
            # the diagnostic log so a later first-use failure has context.
            LOG.debug("Could not preload %s", mod, exc_info=True)
        # 50 ms between imports so Qt drains its event queue (repaints,
        # input) before the next potentially-blocking import.
        QTimer.singleShot(50, self._step)


# ---------------------------------------------------------------------------
# The app registry
# ---------------------------------------------------------------------------
#
# Two axes, and only ONE of them is a place.
#
# An app is filed under WHAT IT DOES — Core, Data, Segmentation models,
# Results & QC, Toxoplasma. That is stable: Format Converter is a Data
# app whether or not anyone has finished it, and it is the axis the
# sidebar, the command palette and the Home tabs all group by.
#
# An app is *also* staged by HOW FINISHED IT IS — alpha, beta or stable
# — and 22 of the 30 are not yet signed off. #16i made that second axis
# two extra CATEGORIES, which meant three of the five subject tabs
# drained to nothing and "where is the format converter" acquired two
# answers. #16j undoes exactly that: maturity is now drawn as the
# tile's HOVER COLOUR, with a legend beside the tiles, so it is visible
# on the same tile that is in the right category. One table
# (:data:`APP_STAGE`), one place per app, no second grouping.
#
#   Core            the end-to-end pipeline: images in, single-object
#                   measurements out, hits called.
#   Data            get images and tables in, run many plates, get the
#                   numbers back out.
#   Segmentation    build, train, pick and check the Cellpose models
#     models        the Mask step runs.
#   Results & QC    read what came out and decide whether to believe it.
#   Explore         ask the numbers a question you did not plan for.
#   Toxoplasma      parasite-specific readouts.
#   Design          plan the next experiment before it runs.
#
# A section holds AT MOST `MAX_APPS_PER_SECTION` apps. Past that nobody
# reads the row — which is exactly how "Tools" grew to sixteen entries and
# became unusable. If a section is full, the honest fix is a new section
# with a name that means something, not a longer row.
#
# Explore and Design are the two that fix accordingly: the modules queued
# behind this file (Graph Builder, pivot/formula, gate editor, feature
# explorer, layer viewer; power/design, experiment designer) would have
# taken Results & QC from eight to fifteen. They are DECLARED and EMPTY —
# no tab, no heading, nothing drawn — until their first app registers,
# because a tab that opens on an empty pane is worse than no tab.
#
# The names are as short as they can be and still mean something: they
# are TAB LABELS on Home, where long names would not fit on one line, and
# a tab that has to elide is a tab nobody can read.
SECTION_CORE = "Core"
SECTION_DATA = "Data"
SECTION_MODELS = "Segmentation models"
SECTION_RESULTS = "Results & QC"
SECTION_TOXO = "Toxoplasma"
#: Interactive analysis: build a plot, pivot a table, draw a gate, page
#: through image layers. Results & QC is what a *finished run* produced —
#: this is the user asking the numbers a question they did not plan for,
#: and it is the home of the Graph Builder family. It exists because
#: those apps are neither a pipeline step nor a QC verdict, and filing
#: seven of them under Results & QC would take that row past the cap.
SECTION_EXPLORE = "Explore"
#: Everything that happens BEFORE the microscope: power, sample size,
#: plate layout, controls and replicates. The only section whose apps
#: consume no images, which is exactly why it is not Data.
SECTION_DESIGN = "Design"

#: Every section an app may be filed under, in workflow order: the
#: end-to-end pipeline first, then getting data in and running it at
#: scale, then the segmentation models that pipeline depends on, then
#: reading the results, then asking them questions, then the
#: Toxoplasma-specific assays, and finally planning the next experiment.
#:
#: This is the DECLARATION. :data:`SECTIONS` is the subset that has apps
#: — see :func:`_refresh_sections`.
SECTION_ORDER = (SECTION_CORE, SECTION_DATA, SECTION_MODELS,
                 SECTION_RESULTS, SECTION_EXPLORE, SECTION_TOXO,
                 SECTION_DESIGN)

#: One line per section, drawn under its heading on that category's tab.
#: A category with two apps in it looks broken until it says why.
#:
#: Every section in :data:`SECTION_ORDER` has a line here, including the
#: ones no app has claimed yet: the note is written when the section is
#: named, not when the tab appears, so the first module to register into
#: an empty section gets a described tab rather than a bare heading.
#: :data:`SECTION_NOTES` is the live subset, kept in step with
#: :data:`SECTIONS` by :func:`_refresh_sections`.
_SECTION_NOTE_LIBRARY = {
    SECTION_CORE: "Images in, single-object measurements out, hits called.",
    SECTION_DATA: ("Images and tables into a spaCR project, many plates run "
                   "unattended, the numbers back out."),
    SECTION_MODELS: ("Build, train, pick and check the Cellpose models the "
                     "Mask step runs."),
    SECTION_RESULTS: ("Read what came out, decide whether to believe it, "
                      "hand it to someone else."),
    SECTION_EXPLORE: ("Ask the measurements a question you did not plan "
                      "for: build a plot, pivot a table, draw a gate."),
    SECTION_TOXO: "Parasite-specific readouts.",
    SECTION_DESIGN: ("Plan the experiment before it runs: power, sample "
                     "size, plate layout, controls and replicates."),
}

_PLUGIN_SECTION_MAP = {
    "core": SECTION_CORE,
    "data": SECTION_DATA,
    "models": SECTION_MODELS,
    "results": SECTION_RESULTS,
    "toxo": SECTION_TOXO,
    # `spacr.plugins._SECTIONS` does not accept these two yet, so no
    # plugin can reach them today. They are mapped here so that widening
    # the plugin allow-list is a one-line change there rather than a
    # KeyError that takes every plugin app down with it — the loop below
    # catches per-registry, not per-plugin.
    "explore": SECTION_EXPLORE,
    "design": SECTION_DESIGN,
}

#: Hard cap on apps per section. Enforced by tests, not at runtime — a
#: violation is a design mistake to fix in this table, not something to
#: discover at startup.
#:
#: Raised 9 -> 13 -> 20. Nine was the width of the Core pipeline and
#: nothing more; a cap exactly the size of the biggest section fires on the
#: next app added rather than when a section stops being readable. Twenty
#: was set on request once the registry passed fifty apps: at that size the
#: sections that fill up are the ones doing real work, and splitting Explore
#: into two half-named tabs would have been worse than a longer row.
MAX_APPS_PER_SECTION = 20

#: **The** app list — ``(key, name, description, section)`` per app, in
#: section order. Every consumer in the process reads this one object:
#: the sidebar, the menu bar, Home, the command palette, the shortcut
#: map and the docs count.
#:
#: It is BUILT, not written down. The rows below go through
#: :func:`register_app`, and so does every plugin app and every module
#: that registers itself at import time — one door, so the ordering and
#: the validation cannot differ between a built-in and a newcomer.
#:
#: Mutating this list directly still works and is still wrong: it skips
#: the section validation, the duplicate-key check and the
#: :data:`SECTIONS` refresh.
APPS: List[Tuple[str, str, str, str]] = []

#: app key → screen factory, for apps that ship their own screen.
#:
#: :meth:`MainWindow._build_screen` consults this before its built-in
#: chain, so a module can own its screen without a line in that chain.
#: See :func:`register_app` for the calling convention.
APP_FACTORIES: dict = {}

#: The live subset of :data:`_SECTION_NOTE_LIBRARY` — one note per
#: section in :data:`SECTIONS`, same keys, always. Mutated in place (not
#: rebound) so ``from .app import SECTION_NOTES`` cannot go stale.
SECTION_NOTES: dict = {}

#: app key → everything a registered app needs OUTSIDE :data:`APPS`.
#:
#: A row in ``APPS`` draws a tile. It does not give the tile a header, a
#: blurb, an API link, a translated name, a headless answer or a Run
#: button that runs anything: those live in six tables in five other
#: files, and four finished features sat unreachable for weeks because
#: their authors could not edit them. :func:`register_app` now takes
#: those strings ONCE, keeps them here, and fans them out — see
#: :func:`_publish_meta` for the push and :func:`registered_metadata`
#: for the pull.
#:
#: Fields, all optional: ``name``, ``title``, ``intro``, ``cli_note``,
#: ``api_module``, ``entry``, ``translations``.
APP_META: dict = {}

#: ``(module, dict attribute, APP_META field)`` — the side tables a
#: registration fans out into.
#:
#: Addressed by NAME and looked up in :data:`sys.modules`, never
#: imported: ``spacr.cli`` must answer ``--list`` without loading
#: PySide6, and importing ``app_screen`` from here would be a cycle.
#: A table this module has not seen yet is not skipped — the module
#: pulls what it missed when it is imported, through
#: :func:`registered_metadata`. Push plus pull is what makes the seam
#: order-independent, which is the whole problem: ``app.py`` imports
#: ``spacr.qt.widgets`` before ``register_app`` exists, so a screen
#: module cannot register during this module's own import, and any
#: registration after it is by definition after somebody's snapshot.
_META_TARGETS: Tuple[Tuple[str, str, str], ...] = (
    ("spacr.qt.screens.app_screen", "APP_TITLES", "title"),
    ("spacr.qt.screens.app_screen", "APP_INTROS", "intro"),
    ("spacr.cli", "INTERACTIVE_ONLY", "cli_note"),
    ("spacr.qt.screens.settings_model", "_APP_API_MODULE", "api_module"),
)


def registered_metadata(field: str) -> dict:
    """``{app key: value}`` for one :data:`APP_META` field, empty ones dropped.

    The PULL half of the seam. A side table calls this at the end of its
    own import to absorb every app that registered before it existed::

        _app = sys.modules.get("spacr.qt.app")
        if _app is not None:
            for _key, _value in _app.registered_metadata("title").items():
                APP_TITLES.setdefault(_key, _value)

    ``setdefault``, not assignment: a table's own hand-written entry is
    the more specific one and wins.
    """
    return {key: meta[field] for key, meta in APP_META.items()
            if meta.get(field)}


def _publish_meta(key: str) -> None:
    """Push one app's metadata into every side table already imported.

    The PUSH half of the seam; see :data:`_META_TARGETS`. Failures are
    swallowed per target: a side table that cannot take an entry must
    not take the registration — and the sidebar tile — down with it.
    """
    meta = APP_META.get(key) or {}
    for module_name, attribute, field in _META_TARGETS:
        value = meta.get(field)
        if not value:
            continue
        module = sys.modules.get(module_name)
        table = getattr(module, attribute, None) if module is not None else None
        if isinstance(table, dict):
            table.setdefault(key, value)
    values = meta.get("translations")
    name = meta.get("name")
    if values and name:
        i18n = sys.modules.get("spacr.qt.i18n")
        add = getattr(i18n, "add_translation", None) if i18n is not None else None
        if callable(add):
            try:
                add(name, values)
            except Exception:
                LOG.warning("Could not translate app name %r", name,
                            exc_info=True)


def registered_entry(key: str):
    """Import and return the pipeline callable app ``key`` registered.

    ``None`` when the app registered no ``entry=``, which is what an
    interactive-only app (its own screen, no Run button) does.

    :func:`spacr.qt.bridge.resolve_pipeline_entry` consults this after
    its own built-in chain, so ``register_app(..., entry="mod:func")`` is
    all a new pipeline app needs for its Run button to run something.
    Imported here, on demand, rather than at registration: registering an
    app must not drag numpy, torch or pandas into a process that only
    wanted to draw a sidebar.
    """
    target = (APP_META.get(key) or {}).get("entry")
    if not target:
        return None
    module_name, _, func_name = str(target).partition(":")
    if not module_name or not func_name:
        raise ValueError(
            f"app {key!r} declared entry={target!r}; it must be spelled "
            f"'module:function'")
    import importlib
    return getattr(importlib.import_module(module_name), func_name)


class _LiveSections(list):
    """The live section list — the type of :data:`SECTIONS`.

    A ``list`` rather than the ``tuple`` this used to be, for one
    reason: ``from spacr.qt.app import SECTIONS`` binds the OBJECT, so a
    tuple that :func:`_refresh_sections` *rebinds* leaves every importer
    holding a snapshot from whenever it imported. That is not
    theoretical — the Graph Builder module hit it: it registers the
    first Explore app after ``app.py`` has finished importing, and the
    modules that had already read ``SECTIONS`` never saw Explore appear.
    Mutating one object in place is what makes a late registration
    visible everywhere, and it is how :data:`SECTION_NOTES` and
    :data:`APPS` are already published.

    It compares equal to a tuple as well as to a list because a tuple is
    what this name published for its whole life and
    ``SECTIONS == ("Core", ...)`` is asserted in the suite: changing the
    container must not change what those assertions mean.
    """

    def __eq__(self, other):
        if isinstance(other, tuple):
            return list.__eq__(self, list(other))
        return list.__eq__(self, other)

    def __ne__(self, other):
        equal = self.__eq__(other)
        return equal if equal is NotImplemented else not equal

    #: Unhashable, like every other list. Spelled out because defining
    #: ``__eq__`` on a class that inherits ``__hash__`` would otherwise
    #: leave the two disagreeing.
    __hash__ = None


#: The sections that actually hold apps, in :data:`SECTION_ORDER` order.
#: Rebuilt in place by :func:`_refresh_sections` on every registration.
#:
#: Derived rather than declared because a declared-but-empty section is a
#: tab that opens on an empty pane — the thing
#: ``test_no_category_tab_is_empty`` exists to forbid. A new section is
#: therefore named in :data:`SECTION_ORDER` today and appears in the UI
#: the day its first app registers, which is what lets a module add
#: itself to one without editing this file.
SECTIONS = _LiveSections()


def _refresh_sections() -> None:
    """Recompute :data:`SECTIONS` and :data:`SECTION_NOTES` from ``APPS``.

    Both are updated IN PLACE. Rebinding either would strand every
    module that imported the name rather than the module.
    """
    live = [s for s in SECTION_ORDER if any(row[3] == s for row in APPS)]
    SECTIONS[:] = live
    SECTION_NOTES.clear()
    SECTION_NOTES.update({s: _SECTION_NOTE_LIBRARY[s] for s in live})


def _insert_position(section: str) -> int:
    """Index in :data:`APPS` where a row for ``section`` belongs.

    After the last row of its own section, and after every section that
    comes earlier in :data:`SECTION_ORDER`. Keeping ``APPS`` grouped is
    not cosmetic: the sidebar starts a new heading every time the
    section changes as it walks the list, so a row filed out of order
    draws its section's heading a second time.
    """
    rank = SECTION_ORDER.index(section)
    position = 0
    for index, row in enumerate(APPS):
        try:
            row_rank = SECTION_ORDER.index(row[3])
        except ValueError:
            row_rank = len(SECTION_ORDER)
        if row_rank <= rank:
            position = index + 1
    return position


def register_app(key: str, name: str, desc: str, section: str, *,
                 factory=None, stage: Optional[str] = None,
                 title: Optional[str] = None, intro: Optional[str] = None,
                 cli_note: Optional[str] = None,
                 api_module: Optional[str] = None,
                 entry: Optional[str] = None,
                 defaults_module: Optional[str] = None,
                 translations: Optional[Tuple[str, ...]] = None,
                 ) -> Tuple[str, str, str, str]:
    """Add one app to the registry. The seam a new module registers through.

    Call it at import time from the module that owns the app::

        from spacr.qt.app import register_app, SECTION_EXPLORE, STAGE_ALPHA

        register_app("graph_builder", "Graph Builder",
                     "Drag columns onto x / y / colour / facet",
                     SECTION_EXPLORE, factory=make_screen, stage=STAGE_ALPHA)

    The first four arguments put a tile on Home and a row in the sidebar.
    The keyword arguments after ``stage`` are what make that tile a
    working app: they are the app's strings, given ONCE here, and fanned
    out into the tables that used to need a hand-edit each — see
    :data:`APP_META`.

    :param key: stable app id. Load-bearing — ``bridge``, ``cli``,
        ``validate``, the drag-and-drop handlers, ``settings_model`` and
        saved user state all key off it, so it is chosen once and never
        renamed. Must be unique across built-ins and plugins.
    :param name: display name; the sidebar row, tile and menu entry.
    :param desc: one-line summary; the tooltip and status tip.
    :param section: one of :data:`SECTION_ORDER`. A section with no apps
        has no tab, so registering the first app into a new section is
        what makes that section appear.
    :param factory: optional zero-argument callable returning the app's
        ``QWidget`` screen. It may declare ``app_key`` and/or ``host``
        keyword parameters — ``host`` is the :class:`MainWindow`, for a
        screen that has signals to connect — and is given whichever of
        the two it accepts. Omit it and the app gets the generic
        settings-driven ``AppScreen``, like every pipeline module.
    :param stage: optional :data:`STAGES` member. Omitted means stable,
        which is also what deleting the entry later means.
    :param title: header shown at the top of the app's own screen.
        Defaults to ``name``; give it only when the screen wants the
        longer form ("Illumination Correction" over a tile that reads
        "Illumination"). Reaches ``app_screen.APP_TITLES``.
    :param intro: the paragraph beside that header — what the module
        does, in a sentence or two. Defaults to ``desc``. Reaches
        ``app_screen.APP_INTROS``.
    :param cli_note: for an app with NO headless path: one sentence
        saying so and what to do instead. Reaches
        ``cli.INTERACTIVE_ONLY``, which is what ``spacr-run <key>``
        prints instead of "unknown module". Mutually exclusive with
        ``entry`` in spirit — an app is one or the other.
    :param api_module: dotted-or-slashed module path under the generated
        API docs ("qt/layer_viewer"), for the ⓘ link beside the settings.
        Reaches ``settings_model._APP_API_MODULE``.
    :param entry: ``"module:function"`` of the callable the Run button
        runs. Resolved lazily by :func:`registered_entry` and consulted
        by :func:`spacr.qt.bridge.resolve_pipeline_entry`; without it the
        Run button answers "Not runnable".
    :param defaults_module: the module whose import calls
        :func:`spacr.settings.register_defaults` for this key.
        ``settings_model.resolve_default_settings`` imports it before
        asking whether the key has registered defaults — otherwise a
        module that registers its settings at import has no settings
        panel until something else happens to import it, and the app
        opens on an empty form. Imported on demand, so registering an
        app costs no numpy/pandas/torch at startup.
    :param translations: the display ``name`` in the nine non-English
        UI languages, in :data:`spacr.qt.i18n.LANGUAGES` order. Reaches
        ``i18n._ROWS`` and its catalogs.
    :returns: the row that was appended, so a caller can keep it.
    :raises ValueError: on a duplicate key, an unknown section, an
        unknown stage or an empty name/description.
    :raises TypeError: if ``factory`` is not callable.
    """
    key = str(key)
    if not key:
        raise ValueError("an app needs a key")
    if any(row[0] == key for row in APPS):
        raise ValueError(f"app key {key!r} is already registered")
    if not str(name).strip() or not str(desc).strip():
        raise ValueError(f"app {key!r} needs a name and a description")
    if section not in SECTION_ORDER:
        raise ValueError(
            f"app {key!r} has unknown section {section!r}; declare it in "
            f"SECTION_ORDER (have: {', '.join(SECTION_ORDER)})")
    if stage is not None and stage not in STAGES:
        raise ValueError(f"app {key!r} has unknown stage {stage!r}")
    if factory is not None and not callable(factory):
        raise TypeError(f"app {key!r} factory {factory!r} is not callable")

    row = (key, str(name), str(desc), section)
    APPS.insert(_insert_position(section), row)
    if factory is not None:
        APP_FACTORIES[key] = factory
    if stage is not None:
        APP_STAGE[key] = stage
    APP_META[key] = {
        "name": row[1],
        "title": str(title).strip() if title else row[1],
        "intro": str(intro).strip() if intro else row[2],
        "cli_note": str(cli_note).strip() if cli_note else "",
        "api_module": str(api_module).strip() if api_module else "",
        "entry": str(entry).strip() if entry else "",
        "defaults_module": str(defaults_module).strip() if defaults_module else "",
        "translations": tuple(translations) if translations else (),
    }
    _publish_meta(key)
    _refresh_sections()
    # The cap is a design rule, not a runtime one: a violation is fixed
    # by splitting the section, and refusing to start the app would not
    # help anyone do that. The suite fails on it; this makes a late
    # registration (a plugin, a lazily-imported module) visible too.
    # The warning this used to log is gone by request. It fired on every
    # registration past the cap, once per app, so a full section produced a
    # stream of identical lines at launch -- and it told the reader nothing
    # the suite does not already assert. The cap itself still stands and
    # tests/qt/test_cov_qt_app.py still enforces it.
    return row


def unregister_app(key: str) -> bool:
    """Remove app ``key`` from the registry. ``True`` if there was one.

    The counterpart to :func:`register_app`, for a plugin that unloads
    and for tests that must not leak a registration into the next one —
    a stray row in :data:`APPS` is a stray tile, a stray sidebar entry
    and a stray Ctrl+N binding for every test that follows.
    """
    key = str(key)
    before = len(APPS)
    APPS[:] = [row for row in APPS if row[0] != key]
    APP_FACTORIES.pop(key, None)
    APP_STAGE.pop(key, None)
    meta = APP_META.pop(key, None)
    if meta is not None:
        # The side tables get the row taken back out too, or a plugin
        # that unloads leaves a title, an intro, an API link and a
        # "GUI-only" excuse behind for an app that no longer exists —
        # and `test_the_gui_only_list_holds_no_apps_that_no_longer_
        # exist` is exactly that failure. Only entries this app put
        # there are removed: a hand-written one was not ours to drop.
        for module_name, attribute, field in _META_TARGETS:
            module = sys.modules.get(module_name)
            table = getattr(module, attribute, None) if module else None
            if isinstance(table, dict) and table.get(key) == meta.get(field):
                table.pop(key, None)
    _refresh_sections()
    return len(APPS) != before


def registered_factory(key: str):
    """The factory registered for ``key``, or ``None``."""
    return APP_FACTORIES.get(key)


def _call_screen_factory(factory, key: str, host=None):
    """Invoke a registered screen ``factory`` with the arguments it declares.

    The contract is deliberately "take what you need": ``lambda:
    MyScreen()`` is a complete factory, and a screen that has signals to
    wire declares ``host`` and gets the :class:`MainWindow`. Resolved by
    inspecting the signature rather than by calling and retrying on
    ``TypeError`` — a retry cannot tell a wrong call from a ``TypeError``
    raised *inside* a factory that was called correctly, and would then
    build the screen twice.
    """
    kwargs = {}
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        # Builtins and C callables have no introspectable signature.
        params = {}
    takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD
                    for p in params.values())
    for wanted, value in (("app_key", key), ("host", host)):
        if takes_any or wanted in params:
            kwargs[wanted] = value
    return factory(**kwargs)


_BUILTIN_APPS = [
    # (key, human name, description, section)
    #
    # `section` is what the app IS ABOUT. How finished it is lives in
    # APP_STAGE below and is drawn as a colour, not as a place.
    #
    # NOTE: keys are load-bearing. bridge.resolve_pipeline_entry,
    # cli.INTERACTIVE_ONLY, validate.APP_FUNCTIONS, dnd_handlers,
    # settings_model.resolve_default_settings and saved user state all
    # key off them. Renaming a key silently breaks those; renaming the
    # display name or moving an app between sections is free.
    #
    # -- Core pipeline: images in, single-object measurements out, hits
    #    called. Ctrl+1..9 map to these nine, in this order.
    ("mask",           "Mask",           "Generate cellpose masks for cells, nuclei and pathogens",   SECTION_CORE),
    ("timelapse",      "Timelapse",      "Segment and track objects across the frames of a time series", SECTION_CORE),
    ("motility",       "Motility Assay", "Automated motility assay: track velocity + infection QC",     SECTION_CORE),
    ("measure",        "Measure",        "Measure single-object intensity + morphology features",       SECTION_CORE),
    ("annotate",       "Annotate",       "Annotate single-object images on a grid; save to database",  SECTION_CORE),
    ("classify",       "Classify (CV)",  "Train Torch CNNs/Transformers to classify single objects",   SECTION_CORE),
    ("ml_analyze",     "Classify (ML)",  "Classical ML (XGBoost / random forest / …) on screen features", SECTION_CORE),
    ("map_barcodes",   "Map Barcodes",   "Map sequencing barcodes to screen data",                      SECTION_CORE),
    ("regression",     "Regression",     "Regression analysis of screen scores",                        SECTION_CORE),
    # -- Data & batch runs: get images and tables into a spaCR project,
    #    run many plates unattended, get the numbers back out.
    ("align",          "Align & Stitch", "Register tiles into one stitched canvas, written incrementally so a 20000x20000 mosaic never has to fit in RAM", SECTION_DATA),
    ("convert",        "Format Converter", "ND2/CZI/LIF/OME-TIFF into Yokogawa TIFFs: preview the mapping, then a map file back to the originals", SECTION_DATA),
    ("foreign",        "Import Project", "Someone else's images, masks and measurement table into a spaCR project, with their columns mapped onto spaCR's", SECTION_DATA),
    ("external_masks", "External Masks", "Turn images and externally generated label masks into a measured spaCR project ready for annotation", SECTION_DATA),
    ("queue",          "Plate Queue",    "Chain multiple plates through the same pipeline",             SECTION_DATA),
    ("batch",          "Batch Runner",   "Queue any modules, plates and settings and run them overnight", SECTION_DATA),
    ("distributed_jobs", "Distributed Jobs", "Submit and monitor spaCR runs on SSH workstations, Slurm or cloud/HPC commands", SECTION_DATA),
    ("db_browser",     "Database Browser", "Browse and export measurements.db without the sqlite3 CLI", SECTION_DATA),
    # -- Segmentation models: build, train, pick and check the Cellpose
    #    models the Mask step runs.
    # Not a training screen despite where it sits: MakeMasksScreen is the
    # brush, the flood fill and the object operations, i.e. correcting a
    # mask by hand. It carried Train Cellpose's description verbatim, which
    # is the app directly below it.
    ("make_masks",     "Make Masks",     "Correct a mask by hand: brush, flood fill, relabel, fill, remove small",  SECTION_MODELS),
    ("train_cellpose", "Train Cellpose", "Train custom Cellpose models",                                SECTION_MODELS),
    ("cellpose_masks", "Cellpose Masks", "Cellpose mask generation",                                    SECTION_MODELS),
    ("model_compare",  "Model Compare",  "Two Cellpose models on the same fields: masks side by side, object-count and ARI deltas", SECTION_MODELS),
    ("model_zoo",      "Model Zoo",      "Browse, verify, download and bench Cellpose + classifier models on three of your fields", SECTION_MODELS),
    # -- Results & QC: look at what came out, decide whether to believe it,
    #    and hand it to someone else.
    ("plate_view",     "Plate Viewer",   "Any measurement as a plate heatmap + edge-effect detection",  SECTION_RESULTS),
    ("agreement",      "Annotator Agreement", "Cohen's/Fleiss' κ between annotation columns + a disagreement review", SECTION_RESULTS),
    ("umap",           "Image UMAP",     "Generate UMAP embeddings with image glyphs",                  SECTION_RESULTS),
    ("activation",     "Activation",     "Generate activation maps",                                    SECTION_RESULTS),
    ("train_compare",  "Training Runs",  "Overlay several training runs' curves with their settings diffed side by side", SECTION_RESULTS),
    ("classifier_evaluation", "Classifier Evaluation", "Held-out predictions, nested CV, calibration, leakage and per-plate metrics", SECTION_RESULTS),
    ("run_history",    "Run History",    "Search every job's settings, files, warnings, failures and performance", SECTION_RESULTS),
    ("report",         "Report",         "One-click shareable HTML/PDF: QC verdict, figures, stats, settings, versions", SECTION_RESULTS),
    # -- Toxoplasma assays: parasite-specific readouts.
    ("analyze_plaques", "Plaque Assay",  "Analyze plaque assay data",                                   SECTION_TOXO),
    ("recruitment",    "Recruitment",    "Analyze recruitment data",                                    SECTION_TOXO),
    ("invasion",       "Invasion Assay", "Two-colour outside/inside stain: attached vs invaded parasites, invasion efficiency per well", SECTION_TOXO),
    ("replication",    "Replication Assay", "Endodyogeny: parasites per vacuole, scored into replication rate per condition", SECTION_TOXO),
]


# ---------------------------------------------------------------------------
# Maturity — the second axis, drawn as colour rather than as a place
# ---------------------------------------------------------------------------

STAGE_STABLE = "stable"
STAGE_BETA = "beta"
STAGE_ALPHA = "alpha"

#: Every stage, least finished first — the order the legend lists them.
STAGES = (STAGE_ALPHA, STAGE_BETA, STAGE_STABLE)

#: **The** classification. app key → :data:`STAGE_ALPHA` or
#: :data:`STAGE_BETA`; anything absent is :data:`STAGE_STABLE`.
#:
#: This is the single table the hover colour and the legend both read
#: (through :func:`app_stage` and :func:`home_stages`), and it is the
#: only place maturity is written down — #16i's mistake was that
#: maturity was *also* a section, so an app could be alpha in one table
#: and Data in another and the two could disagree.
#:
#: Signing an app off is deleting its line here. Nothing else moves: the
#: app is already filed under what it does.
APP_STAGE = {
    # -- alpha: built and reachable, not yet trusted end to end (15)
    "align":           STAGE_ALPHA,
    "model_zoo":       STAGE_ALPHA,
    "convert":         STAGE_ALPHA,
    "foreign":         STAGE_ALPHA,
    "external_masks":  STAGE_ALPHA,
    "model_compare":   STAGE_ALPHA,
    "queue":           STAGE_ALPHA,
    "batch":           STAGE_ALPHA,
    "distributed_jobs": STAGE_ALPHA,
    "invasion":        STAGE_ALPHA,
    "db_browser":      STAGE_ALPHA,
    "plate_view":      STAGE_ALPHA,
    "agreement":       STAGE_ALPHA,
    "train_compare":   STAGE_ALPHA,
    "classifier_evaluation": STAGE_ALPHA,
    "run_history":     STAGE_ALPHA,
    "report":          STAGE_ALPHA,
    # -- beta: further along, in regular use, still not signed off (9)
    "make_masks":      STAGE_BETA,
    "train_cellpose":  STAGE_BETA,
    "cellpose_masks":  STAGE_BETA,
    "timelapse":       STAGE_BETA,
    "motility":        STAGE_BETA,
    "analyze_plaques": STAGE_BETA,
    "replication":     STAGE_BETA,
    "umap":            STAGE_BETA,
    "activation":      STAGE_BETA,
}

# The built-ins go through the same door as everything else. Registering
# 34 rows one at a time on every import is what keeps `register_app`
# honest: an ordering or validation mistake in it shows up here, at
# import, rather than the first time somebody adds the 35th app.
for _row in _BUILTIN_APPS:
    register_app(*_row)
del _row


# ---------------------------------------------------------------------------
# Apps that live in their own module
# ---------------------------------------------------------------------------
# The two pipeline modules below are registered here rather than by
# themselves, because they are not Qt modules: `spacr.illumination` and
# `spacr.sequencing_qc` are imported into worker processes and into
# `spacr-run`, and neither may grow an import of PySide6. Their strings
# are theirs; the row is ours. Everything after `section=` is fanned out
# by `register_app` — see APP_META.

register_app(
    "illumination", "Illumination",
    "Estimate the flat-field from the plate itself and divide it out "
    "before any intensity feature is measured",
    SECTION_DATA,
    stage=STAGE_ALPHA,
    title="Illumination Correction",
    intro=(
        "No microscope lights a field evenly, so the same cell measures "
        "brighter at the centre than at a corner — routinely 10–40% on a "
        "widefield screen, and it does not average out of a per-well "
        "aggregate. This estimates the illumination field from the plate's "
        "own merged fields (a per-pixel median across fields, then a smooth "
        "low-order surface), QCs it, and installs it as a preprocessing hook "
        "that every measure worker applies before a single feature is "
        "computed."),
    api_module="illumination",
    entry="spacr.illumination:prepare_illumination_correction",
    defaults_module="spacr.illumination",
    translations=("Belysning", "Beleuchtung", "Iluminación", "照明",
                  "Iluminação", "प्रकाश", "조명", "Lýsing", "Éclairage"),
)

register_app(
    "barcode_qc", "Barcode QC",
    "Did the mapping run work, and where does the abundance threshold go",
    SECTION_RESULTS,
    stage=STAGE_ALPHA,
    title="Barcode QC",
    intro=(
        "Reads per well, starved wells, unmapped reads, barcode collisions, "
        "row/column position effects and library coverage for a finished "
        "mapping run — and then the number everyone used to read off a "
        "histogram once and copy forward: state how many gRNAs per well the "
        "design intends and the abundance threshold that delivers it is "
        "derived, swept either side of, and written out in words."),
    api_module="sequencing_qc",
    entry="spacr.sequencing_qc:barcode_qc",
    defaults_module="spacr.sequencing_qc",
    translations=("Streckkods-QC", "Barcode-QC", "CC de códigos de barras",
                  "条形码质控", "CQ de código de barras", "बारकोड QC",
                  "바코드 QC", "Strikamerkja-QC", "CQ des codes-barres"),
)


#: Modules that own their registry row and call ``register_app``
#: themselves — ``module name`` → ``registration function name``.
#:
#: Imported HERE, at the bottom of this module, because that is the only
#: place a screen module can register from and be seen by everybody.
#: ``app.py`` imports ``spacr.qt.widgets`` at its line 41, before
#: ``register_app`` exists, so nothing reachable from the top of this
#: file can register; and a registration that happens later — when
#: ``spacr.qt.screens`` is first imported, or when ``run()`` walks
#: ``spacr.qt.SELF_REGISTERING_MODULES`` at launch — is a registration
#: that some importer's snapshot of ``APPS`` predates, and that
#: ``import spacr.qt.app`` alone does not produce at all. The inventory
#: tests compare against ``APPS`` after importing this module, so a row
#: that appears only sometimes is a ledger that fails only sometimes.
#:
#: Those two other seams still work and are still the right place for a
#: screen that has no row of its own; every registration function named
#: here is idempotent, so being called from both costs nothing.
_SELF_REGISTERING_APPS = (
    ("spacr.qt.layer_viewer", "register_layer_viewer_app"),
    ("spacr.qt.screens.graph_builder", "register"),
    # The three that arrived just after the seam landed and sat finished,
    # tested and unreachable for the same reason the first four did.
    #
    # Power is the first app of the Design section, so this row is also
    # what makes that tab appear — the section has been declared, noted
    # and empty since the sections were named.
    ("spacr.qt.screens.power", "register"),
    # AnnData Export has no screen of its own: it registers settings, so
    # the generic ``AppScreen`` draws its form, and an ``entry=`` so the
    # Run button on that form runs the export. It is listed here rather
    # than left to its own import-time call because that call is a no-op
    # unless ``spacr.qt.app`` is ALREADY in ``sys.modules`` — deliberately,
    # so a headless export never drags PySide6 in — which made whether the
    # row existed depend on import order.
    ("spacr.anndata_export", "register_anndata_app"),
    # Run Compare registers at its own import and is named in
    # ``spacr.qt.SELF_REGISTERING_MODULES`` too, which only runs at
    # ``run()``. That made the row appear at launch and not under
    # ``import spacr.qt.app``, i.e. exactly the sometimes-there row the
    # note above is about. Both calls are idempotent.
    ("spacr.qt.screens.run_compare", "register"),
    # PCA and Tabulate: both finished, both tested, both defining register()
    # that nothing called. They were held back when Explore was at the
    # MAX_APPS_PER_SECTION ceiling of 13; it is at 8 now, so the reason has
    # expired. Found by the README pass, which declined to advertise a screen
    # with no tile -- which is the right instinct and also how a feature stays
    # invisible for a fortnight.
    ("spacr.qt.screens.pca", "register"),
    ("spacr.qt.screens.tabulate", "register"),
)

import importlib as _importlib

for _module_name, _func_name in _SELF_REGISTERING_APPS:
    try:
        getattr(_importlib.import_module(_module_name), _func_name)()
    except Exception:
        # One screen's import-time bug costs that screen and nothing
        # else. The same posture this file already takes towards
        # plugins, for the same reason: the window still opens.
        LOG.exception("Could not register the app owned by %s", _module_name)
del _module_name, _func_name


# Plugin apps use the same registry rows and maturity annotations as built-ins.
# Contributions can add a key but never replace one.
try:
    from spacr.plugins import plugin_apps as _plugin_apps, record_diagnostic
    for _plugin_app in _plugin_apps():
        # Recomputed per plugin, not snapshotted before the loop: the old
        # snapshot held built-in keys only, so two plugins claiming the
        # same key both landed in APPS and the duplicate only showed up
        # as two identical sidebar rows.
        if _plugin_app.key in {row[0] for row in APPS}:
            record_diagnostic(
                _plugin_app.key,
                f"Plugin app key {_plugin_app.key!r} collides with a built-in "
                "Qt app; the built-in app was kept.",
            )
            continue
        try:
            register_app(
                _plugin_app.key,
                _plugin_app.name,
                _plugin_app.description,
                _PLUGIN_SECTION_MAP[_plugin_app.section],
                stage=_plugin_app.stage,
            )
        except (ValueError, TypeError) as _exc:
            # Everything `spacr.plugins` already validates — section,
            # stage, non-empty name — plus anything it starts allowing
            # that this registry does not. One bad contribution is
            # dropped; the rest of the plugins still load.
            record_diagnostic(
                _plugin_app.key,
                f"Plugin app {_plugin_app.key!r} was not registered: {_exc}",
            )
except Exception:
    LOG.exception("Could not add plugin apps to the Qt registry")


def app_stage(key: str) -> str:
    """How finished ``key`` is — one of :data:`STAGES`.

    Unknown keys read as stable rather than raising: a stage is an
    annotation on an app, and an app with no annotation is one nobody
    has flagged.
    """
    return APP_STAGE.get(key, STAGE_STABLE)


def app_is_visible(key: str) -> bool:
    """Whether ``key`` should appear in module navigation.

    Preferences are imported lazily so this registry remains safe for
    packaging and headless callers. If preferences cannot be read, preserve
    the historical all-modules-visible behaviour.
    """
    try:
        from .preferences import maturity_is_visible
        return maturity_is_visible(app_stage(key))
    except Exception:
        return True


def visible_apps() -> List[Tuple[str, str, str, str]]:
    """Registry rows allowed by the Alpha/Beta visibility preferences."""
    return [row for row in APPS if app_is_visible(row[0])]


def home_stages() -> dict:
    """app key → stage, for every app in :data:`APPS`.

    What :func:`make_home_page` hands the Home page, so the tiles and
    the legend colour from the same table this module owns and the
    widget still knows nothing about what a stage means.
    """
    return {row[0]: app_stage(row[0]) for row in APPS}


def section_members(
    section: str,
    apps: Optional[List[Tuple[str, str, str, str]]] = None,
) -> List[Tuple[str, str, str, str]]:
    """The ``APPS`` rows a category's tab shows, in registry order."""
    source = APPS if apps is None else apps
    return [row for row in source if row[3] == section]


def home_categories(
    apps: Optional[List[Tuple[str, str, str, str]]] = None,
) -> List[Tuple[str, List[str]]]:
    """``(section, [app key])`` for every tab after Home, in tab order.

    Computed rather than written down, so a section cannot acquire a tab
    it has no apps for or lose one it does.
    """
    result = []
    for section in SECTIONS:
        rows = section_members(section, apps)
        if rows:
            result.append((section, [row[0] for row in rows]))
    return result


def home_bands(
    apps: Optional[List[Tuple[str, str, str, str]]] = None,
) -> List[Tuple[str, List[Tuple[str, str, str, str]]]]:
    """``(band, rows)`` for the Home tab, in :data:`SECTIONS` order.

    The same grouping the tabs use — every app once, under what it is
    about. Home is the "all of it" view and each later tab is a filter
    of it, which is only true while both read this one table.

    Bands with no apps are dropped rather than drawn empty. None are,
    today; the guard is what stops a heading appearing over nothing the
    day a section's last app is retired.
    """
    grouped = [(s, section_members(s, apps)) for s in SECTIONS]
    return [(s, rows) for s, rows in grouped if rows]


def make_home_page(parent=None):
    """Build the Home page exactly as the running app builds it.

    The grouping, the stages, the notes and the icon provider are four
    arguments that have to agree, and a test that assembles its own
    HomePage is testing a page that does not ship — which is the exact
    class of bug #16i was: Home grouped by one table and the tabs by
    another. One constructor call, used by :class:`MainWindow` and by
    the suite.
    """
    from .widgets.home import HomePage
    apps = visible_apps()
    return HomePage(
        apps, _icon_for_app, parent,
        section_notes=SECTION_NOTES,
        categories=home_categories(apps),
        bands=[(s, [r[0] for r in rows]) for s, rows in home_bands(apps)],
        stages=home_stages())


#: demo key → the label its entry carries in the Demos menu.
#:
#: Module level rather than inline in ``_build_menus`` because the menu is
#: not the only thing that names a demo: an app screen with no ``src`` set
#: offers the user the demo that would fill it, and it has to name the
#: same one. That hint used to read "use Demos → Mask demo…" on EVERY
#: screen, so Measure, Timelapse, Classify and Sequencing each pointed at
#: a dataset that would not open them.
#:
#: Labels are kept verbatim: :mod:`spacr.qt.i18n` keys its catalog on the
#: English string, so renaming one drops its translation in nine
#: languages.
DEMO_LABELS = {
    "mask":         "Mask demo…",
    "measure":      "Measure demo…",
    "crop":         "Crop demo…",
    "classify":     "Classify demo…",
    "timelapse":    "Timelapse demo…",
    "map_barcodes": "Sequencing demo…",
}


def demo_label_for_app(app_key: str) -> Optional[str]:
    """The Demos-menu label of the demo that opens in app ``app_key``.

    ``None`` when no demo lands there, which is most of the registry —
    the caller says something generic rather than naming a demo that
    would take the user somewhere else.

    Resolved through :attr:`MainWindow.DEMO_TARGETS` (demo key → target
    app) rather than a second table, so a demo that is re-pointed at a
    different app moves its hint with it. The first match wins: two demos
    land on ``measure`` (Measure and Crop) and the one named after the
    app is the one it lists first.
    """
    for demo_key, (target, _generator) in MainWindow.DEMO_TARGETS.items():
        if target == app_key and demo_key in DEMO_LABELS:
            return DEMO_LABELS[demo_key]
    return None


# Explicit key -> icon-filename overrides for cases where the app_key
# doesn't match any resource filename. Add entries here rather than
# renaming resource files.
_ICON_OVERRIDES = {
    "analyze_plaques": "plaque.png",
    "train_cellpose":  "cellpose_masks.png",  # share the Cellpose Masks icon
    "agreement":       "annotate.png",     # shares the Annotate glyph: it
                                           # scores annotation columns
    "plate_view":      "map_barcodes.png", # ruled bars read as a well grid
    "model_compare":   "mask.png",         # mask.png is one field split down
                                           # the middle -- raw objects on one
                                           # side, contours on the other. That
                                           # IS Model Compare: the same field,
                                           # segmented two ways, side by side.
    "model_zoo":       "download.png",     # the zoo is where models come from
    #
    # FOUR entries were REMOVED here — `timelapse`→run.png,
    # `motility`→recruitment.png, `db_browser`→map_barcodes.png and
    # `train_compare`→classify.png — because the user chose artwork for
    # each and it is now installed as `<key>.png`, which `app_icon`
    # finds without being told. An override is for an app that BORROWS
    # another app's picture; it is not the place to record "this app has
    # an icon". (`align` and `foreign` gained artwork in the same round;
    # neither was ever in this table — `align` was in _FORCE_GLYPH below
    # and `foreign` had nothing at all.)
    #
    # `motility` is the one worth remembering. It borrowed
    # recruitment.png, so re-skinning Recruitment silently re-skinned
    # Motility Assay as well — the old recruitment drawing had to be
    # kept and installed as motility.png to stop that. A borrowed icon
    # is a coupling between two apps that nothing declares.
    #
    # Of the six left, FOUR are genuine sharing and are a debt:
    # `train_cellpose` shows Cellpose Masks' picture, `agreement` shows
    # Annotate's, `plate_view` shows Map Barcodes', `model_compare`
    # shows Mask's. The other two are only renames — no app is keyed
    # `plaque` or `download`, so `analyze_plaques` and `model_zoo` are
    # the sole users of that artwork and share it with nobody.
    #
    # queue.png / batch.png / invasion.png / replication.png are drawn for
    # these apps and named after them, so they need no override. They used
    # to: `queue` and `batch` BOTH aliased sequencing.png (a DNA helix,
    # "the closest visual match for now"), which made two different apps
    # render identically as a picture of neither. The new pair carries the
    # distinction that matters -- queue is the same settings over many
    # plates, batch is arbitrary module+plate combinations in sequence.
}

# Keys that render their qtawesome glyph instead of a bundled PNG.
#
# EMPTY, deliberately, and kept rather than deleted: it is the documented
# fallback for an app whose meaning no bundled artwork carries, and
# emptying the set is not the same as removing the escape hatch.
#
# ``align`` was the last entry. No bundled PNG read as "tiles registered
# into ONE canvas", so it drew ``fa5s.border-all`` — a square divided
# into four by its own seams. The user has since chosen
# ``cellpose_all_01`` for it, which is that judgement overruled by the
# person whose app it is; the PNG is installed as ``align.png`` and the
# glyph is out of the way.
#
# WORTH KNOWING: ``cellpose_all_01.png`` was already installed as
# ``cellpose_all.png``, so ``align.png`` is now byte-identical to it.
# No two Qt tiles collide — ``cellpose_all`` is a Tk-only module and is
# not in :data:`APPS` — but the Tk GUI's "Cellpose All" and Qt's
# Align & Stitch draw the same picture, and re-inking one re-inks both.
# The user chose it explicitly; this is the note, not an objection.
#
# ``invasion`` left for the same reason earlier ("no bundled PNG reads
# as inside vs outside") once it had artwork that did.
_FORCE_GLYPH: set = set()


def _icon_for_app(key: str) -> Optional[QIcon]:
    """Return a QIcon for an app key.

    The bundled PNG is re-inked for the active theme by
    :func:`spacr.qt.iconset.app_icon`. Loading it raw is what left
    Format Converter — a solid-black PNG — with no visible icon at all
    on the black home page, and every white icon invisible on the light
    theme. Falls back to a themed qtawesome glyph when there is no PNG.
    """
    try:
        from spacr.plugins import get_app
        plugin_app = get_app(key)
        if plugin_app is not None and plugin_app.icon:
            if os.path.isfile(plugin_app.icon):
                return QIcon(plugin_app.icon)
            return iconset.icon(plugin_app.icon)
    except Exception:
        LOG.debug("Could not resolve plugin icon for %s", key, exc_info=True)
    # Keys that should use their themed qtawesome glyph rather than a bundled
    # PNG (e.g. train_cellpose got a fresh 'brain' glyph).
    if key in _FORCE_GLYPH:
        return iconset.icon(key)
    return iconset.app_icon(key, override=_ICON_OVERRIDES.get(key))


class Sidebar(QWidget):
    """Left navigation column. Emits `nav_selected(str key)` when a tile
    is clicked. `Home` reverts to the startup page."""

    from PySide6.QtCore import Signal
    nav_selected = Signal(str)

    #: Width bounds in px at 100 % font scale. The column starts at
    #: ``WIDTH_MIN`` and widens — up to ``WIDTH_MAX`` — if the longest app
    #: name needs it, so a new long name can't quietly get cut in half.
    WIDTH_MIN = 220
    WIDTH_MAX = 320

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Sidebar")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        title = QLabel("spaCR")
        title.setObjectName("SidebarTitle")
        outer.addWidget(title)

        # The nav rows scroll. Measured at 1440x900 -- the realistic laptop
        # size -- this column stacks a row per app, a heading per section
        # and the title in a plain QVBoxLayout and asks for ~1300 px against
        # 850 available, so the last few apps were simply UNREACHABLE. The
        # title stays pinned; only the rows move.
        #
        # No stylesheet is set on the scroll area on purpose: an unscoped
        # `background: transparent` on a QScrollArea cascades to every
        # descendant and strips the fill off the buttons inside it.
        self._scroll = QScrollArea(self)
        self._scroll.setObjectName("SidebarScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.viewport().setAutoFillBackground(False)

        inner = QWidget()
        inner.setObjectName("SidebarInner")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._scroll.setWidget(inner)
        outer.addWidget(self._scroll, 1)

        self._items: list[ElidingPushButton] = []
        self._section_headers: dict[str, QLabel] = {}

        home = self._make_item("Home", "Back to the start page", "__home__")
        home.setIcon(iconset.icon("home"))
        home.clicked.connect(lambda: self.nav_selected.emit("__home__"))
        layout.addWidget(home)

        # Group apps by section, in APPS order
        current_section = None
        for key, name, desc, section in APPS:
            if section != current_section:
                header = QLabel(section)
                header.setObjectName("SidebarSection")
                layout.addWidget(header)
                self._section_headers[section] = header
                current_section = section
            btn = self._make_item(name, desc, key)
            icon = _icon_for_app(key)
            if icon is not None:
                btn.setIcon(icon)
                btn.setIconSize(QSize(16, 16))
            btn.clicked.connect(lambda checked=False, k=key: self.nav_selected.emit(k))
            layout.addWidget(btn)

        layout.addStretch(1)
        self.refresh_visibility()

    def refresh_visibility(self) -> None:
        """Apply Alpha/Beta filters without rebuilding or reparenting the dock."""
        section_by_key = {key: section for key, _name, _desc, section in APPS}
        visible_sections = set()
        for btn in self._items:
            key = str(btn.property("navKey") or "")
            if key == "__home__":
                btn.setVisible(True)
                continue
            visible = app_is_visible(key)
            btn.setVisible(visible)
            if visible and key in section_by_key:
                visible_sections.add(section_by_key[key])
        for section, header in self._section_headers.items():
            header.setVisible(section in visible_sections)
        self.setFixedWidth(self.fitting_width())

    def refresh_icons(self) -> None:
        """Re-ink every nav icon for the current theme.

        A QIcon bakes its pixmap when it is built, so re-applying the
        stylesheet does not recolour icons that already exist: switch to
        the light theme with a sidebar on screen and every white glyph
        stays white, on white. Each row carries its app key in the
        ``navKey`` property, so rebuilding them is just a re-lookup.
        """
        for btn in self._items:
            key = btn.property("navKey")
            if not key:
                continue
            icon = (iconset.icon("home") if key == "__home__"
                    else _icon_for_app(key))
            if icon is not None:
                btn.setIcon(icon)
                btn.setIconSize(QSize(16, 16))

    def _make_item(self, name: str, desc: str,
                   nav_key: str) -> "ElidingPushButton":
        """Build one navigation row.

        The label elides instead of clipping, and the tooltip leads with
        the app NAME (not just its description) so a shortened label is
        still identifiable on hover.

        :param nav_key: the app key this row navigates to, exposed as the
            ``navKey`` Qt property so callers (and tests) can tell app
            rows apart from the Home row without parsing labels.
        """
        # "&&" so Qt draws the ampersand instead of eating it as a
        # mnemonic: "Align & Stitch" was rendering as "Align _Stitch" in
        # the nav column (QPushButton reads a lone & as an accelerator;
        # the Home tiles are unaffected because they draw their name in
        # a QLabel, which does not).
        btn = ElidingPushButton(f"  {name.replace('&', '&&')}")
        btn.setObjectName("SidebarItem")
        btn.setProperty("navKey", nav_key)
        if nav_key != "__home__":
            btn.setProperty("moduleAppKey", nav_key)
            btn.setProperty("moduleNameSource", name)
            btn.setProperty("moduleSummarySource", desc)
            btn.setProperty("moduleTooltipStyle", "sidebar")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip(f"{name} — {desc}" if desc else name)
        # Accessibility — screen readers announce the app name +
        # its one-line description as the button's role.
        btn.setAccessibleName(name)
        btn.setAccessibleDescription(desc)
        self._items.append(btn)
        return btn

    def fitting_width(self) -> int:
        """Width that shows the longest nav label in full, within bounds.

        Font scale moves both bounds (a 150 % font needs a 150 % column),
        and the widest button's own size hint moves the result inside
        them. Names longer than ``WIDTH_MAX`` elide with a tooltip rather
        than pushing the column across the window.

        Public because the locked dock has to re-apply it: the sidebar is
        re-parented out of the drawer, which had resized it to the
        drawer's width.
        """
        from .preferences import scaled_px
        widest = max((b.sizeHint().width() for b in self._items
                      if not b.isHidden()), default=0)
        return max(scaled_px(self.WIDTH_MIN),
                   min(widest + scaled_px(12), scaled_px(self.WIDTH_MAX)))

    def clipped_items(self) -> list:
        """Nav buttons whose label had to be shortened to fit.

        Empty in a healthy layout; a test asserts that, and anything in
        here still carries its full name in its tooltip.
        """
        return [b for b in self._items if b.is_elided()]


class MainWindow(QMainWindow):
    """Top-level window: sidebar + stacked screens + status bar.

    :param initial_app: optional app key to navigate to on show; when
        omitted the window opens on the Home startup page.
    """

    def __init__(self, initial_app: Optional[str] = None):
        super().__init__()
        self._closing = False
        self.setWindowTitle("spaCR")
        self.setMinimumSize(1200, 720)

        self._build_menu_bar()

        # Central layout: a row that holds an (initially empty) dock slot
        # and the screen stack. By default the app list is a REVEAL over
        # the stack's left edge rather than anything in that slot.
        #
        # The column was 220-320 px of the 1440 a laptop has, on every
        # screen, holding a list most sessions never touch — and it is
        # the reason Home could not fit its five categories plus a state
        # column without scrolling. As a drawer it costs 6 px of trigger
        # strip and is one hover, one click, or Ctrl+B away.
        #
        # The slot is what "lock the dock" fills: see
        # :meth:`apply_dock_mode`. It exists whatever the mode, because a
        # QMainWindow's central widget cannot be swapped without
        # re-parenting the stack, and re-parenting a stack that already
        # holds live screens is how a locked dock would cost you the
        # screen you were looking at.
        #
        # `self._sidebar` is the SAME `Sidebar` object it always was, only
        # reparented: the tutorial highlights it, the command palette and
        # the tests all reach it by that name.
        self._stack = QStackedWidget()
        central = QWidget()
        central.setObjectName("CentralRow")
        row = QHBoxLayout(central)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self._dock_slot = QWidget()
        self._dock_slot.setObjectName("DockSlot")
        slot_col = QVBoxLayout(self._dock_slot)
        slot_col.setContentsMargins(0, 0, 0, 0)
        slot_col.setSpacing(0)
        self._dock_slot.hide()
        row.addWidget(self._dock_slot)
        row.addWidget(self._stack, 1)
        self.setCentralWidget(central)

        self._sidebar = Sidebar()
        self._sidebar.nav_selected.connect(self._on_nav_selected)
        self._sidebar.nav_selected.connect(self._on_drawer_navigated)

        from .widgets.drawer import EdgeDrawer
        self._app_drawer = EdgeDrawer(self._stack, self._sidebar,
                                      width=self._sidebar.width())
        self.apply_dock_mode()

        # Register screens lazily — created on first navigation.
        self._screens: dict[str, QWidget] = {}
        #: App keys in the order the user last LOOKED at them, oldest
        #: first. Not the same thing as ``_screens`` order, which is
        #: creation order and never changes when an app is revisited —
        #: see :meth:`_snapshot_current_screen_settings`.
        self._visit_order: list[str] = []
        self._install_startup_page()

        # Rich status bar: transient message (left) + active app + version
        status = QStatusBar()
        self._status_app_label = QLabel("Home")
        self._status_app_label.setObjectName("Muted")
        self._status_version_label = QLabel(f"spaCR {self._resolve_version()}")
        self._status_version_label.setObjectName("Caption")
        status.addPermanentWidget(self._status_app_label)
        status.addPermanentWidget(self._status_version_label)
        status.showMessage("Ready")
        self.setStatusBar(status)

        # The AI Console now lives inside each pipeline app's Console
        # panel (see spacr.qt.widgets.console_panel). No side-dock.

        # Preload heavy pipeline imports in a background thread AFTER
        # the first screen has been built. Kicking it off pre-nav caused
        # a real circular-import race in spacr.core/IPython on some
        # systems ("partially initialized module 'IPython'"), so we wait
        # a moment before starting.
        from PySide6.QtCore import QTimer
        self._preloader = _PipelinePreloader()
        QTimer.singleShot(1500, self._preloader.start)

        # Keyboard shortcuts — Ctrl+H, Ctrl+1..9, Ctrl+K, F1/?, etc.
        try:
            from . import shortcuts
            shortcuts.install(self)
        except Exception:
            pass

        if initial_app:
            self._on_nav_selected(initial_app)

        # Apply the persisted language after every startup widget exists.
        # New lazy screens are translated separately when first constructed.
        self.refresh_language()

        # First-launch tour — coach-marks over the home layout the
        # first time this user boots spacr. State stored in QSettings,
        # so subsequent launches are silent. Delayed a beat so the
        # window has time to render before the overlay attaches.
        try:
            from PySide6.QtCore import QTimer
            from .first_run import maybe_show_tour
            QTimer.singleShot(800, lambda: maybe_show_tour(self))
        except Exception:
            pass

    def _resolve_version(self) -> str:
        """Return the installed spacr version string, or ``"dev"`` on failure."""
        try:
            import spacr
            return getattr(spacr, "__version__", "") or "dev"
        except Exception:
            return "dev"

    # -- menu -------------------------------------------------------------
    def _build_menu_bar(self):
        mb = self.menuBar()

        app_menu = mb.addMenu("&spaCR")
        self._app_actions: dict[str, QAction] = {}
        for key, name, desc, section in APPS:
            act = QAction(name, self)
            act.setStatusTip(desc)
            # Translate the name and reviewed scientific summary as separate
            # semantic fields; word-by-word translation of the combined text
            # can produce misleading mixed-language help.
            act.setProperty("moduleAppKey", key)
            act.setProperty("moduleNameSource", name)
            act.setProperty("moduleSummarySource", desc)
            act.triggered.connect(lambda checked=False, k=key: self._on_nav_selected(k))
            app_menu.addAction(act)
            self._app_actions[key] = act
        self._refresh_app_action_visibility()
        app_menu.addSeparator()
        act_home = QAction("Home", self)
        act_home.setShortcut(QKeySequence("Ctrl+H"))
        act_home.triggered.connect(lambda: self._on_nav_selected("__home__"))
        app_menu.addAction(act_home)
        # The keyboard + menu route into the edge reveal. A panel you can
        # only summon by hovering a 6 px strip is a panel a keyboard user
        # does not have, so this is not optional decoration.
        act_all = QAction("All apps", self)
        act_all.setShortcut(QKeySequence("Ctrl+B"))
        act_all.setStatusTip(
            "Show the full app list. Also revealed by moving the pointer "
            "to the left edge of the window.")
        act_all.triggered.connect(self.toggle_app_drawer)
        app_menu.addAction(act_all)
        self.addAction(act_all)
        #: Kept so :meth:`apply_dock_mode` can grey it out — a Ctrl+B that
        #: silently does nothing because the dock is hidden is worse than
        #: a menu entry that says so.
        self._act_all_apps = act_all
        app_menu.addSeparator()
        act_prefs = QAction("Preferences…", self)
        act_prefs.setShortcut(QKeySequence("Ctrl+,"))
        act_prefs.triggered.connect(self._open_preferences)
        app_menu.addAction(act_prefs)
        app_menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.setShortcut(QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        app_menu.addAction(act_quit)

        demo_menu = mb.addMenu("&Demos")
        self._demo_actions: dict[str, QAction] = {}
        for app_key, label in DEMO_LABELS.items():
            act = QAction(label, self)
            act.setStatusTip(
                f"Generate a synthetic {app_key} dataset and open it "
                "in the matching app.")
            act.triggered.connect(
                lambda checked=False, k=app_key: self._on_load_demo(k))
            demo_menu.addAction(act)
            target = self.DEMO_TARGETS.get(app_key, (app_key, ""))[0]
            self._demo_actions[app_key] = act
            act.setVisible(app_is_visible(target))
        demo_menu.addSeparator()
        act_e2e = QAction("End-to-end (Mask → Measure → Annotate) real dataset…", self)
        act_e2e.setStatusTip(
            "Download the toxo_mito HF demo dataset + settings pack, "
            "then chain Mask → Measure → Annotate on it.")
        act_e2e.triggered.connect(self._on_e2e_demo)
        demo_menu.addAction(act_e2e)

        help_menu = mb.addMenu("&Help")
        # Label kept verbatim: `spacr/qt/i18n.py` keys its catalog on the
        # English string, so renaming this action drops its translation in
        # all nine languages.
        act_tutorial = QAction("Tutorial (web)", self)
        act_tutorial.setStatusTip(
            "Open the interactive spaCR lesson library in a browser.")
        act_tutorial.setIcon(
            self.style().standardIcon(
                QStyle.StandardPixmap.SP_MessageBoxInformation
            )
        )
        act_tutorial.triggered.connect(
            lambda: self._open_url(TUTORIALS_URL))
        help_menu.addAction(act_tutorial)
        act_docs = QAction("Documentation (web)", self)
        act_docs.setIcon(
            self.style().standardIcon(
                QStyle.StandardPixmap.SP_MessageBoxInformation
            )
        )
        act_docs.triggered.connect(
            lambda: self._open_url(DOCS_URL))
        help_menu.addAction(act_docs)
        act_about = QAction("About spaCR", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)
        help_menu.addSeparator()
        act_update = QAction("Check for updates…", self)
        act_update.triggered.connect(self._check_for_updates)
        help_menu.addAction(act_update)
        act_log = QAction("Open log folder…", self)
        act_log.setStatusTip(
            "Open the ~/.spacr/logs folder. Attach the newest log file "
            "to any bug report — it contains a full trace of every "
            "spaCR run including function entries, button presses, "
            "and pipeline output.")
        act_log.triggered.connect(self._open_log_folder)
        help_menu.addAction(act_log)

    def _open_url(self, url: str):
        """Open ``url`` in the system browser; surface failures in the status bar."""
        import webbrowser
        try:
            webbrowser.open(url)
        except Exception as e:
            self.statusBar().showMessage(f"Failed to open {url}: {e}", 5000)

    # -- demos -----------------------------------------------------------
    # Map each demo key to (target-app key, generator function name).
    # Kept as a class constant so tests can introspect it without launching
    # the file dialog.
    DEMO_TARGETS = {
        "mask":      ("mask",       "generate_mask_demo"),
        "measure":   ("measure",    "generate_measure_demo"),
        "crop":      ("measure",    "generate_crop_demo"),
        "classify":  ("annotate",   "generate_classify_demo"),
        # The timelapse demo writes a settings CSV with timelapse=True; the
        # Mask module no longer has a widget for that key, so it has to land
        # in the Timelapse module or the flag would be silently dropped.
        "timelapse": ("timelapse",  "generate_timelapse_demo"),
        "map_barcodes": ("map_barcodes", "generate_map_barcodes_demo"),
    }

    def _on_load_demo(self, demo_key: str) -> None:
        """Generate a synthetic demo dataset, save its settings, then
        navigate to the matching app and pre-populate it."""
        from PySide6.QtWidgets import QFileDialog
        from pathlib import Path

        target_app, gen_name = self.DEMO_TARGETS[demo_key]

        default = str(Path.home() / "spacr-demos" / demo_key)
        dst = QFileDialog.getExistingDirectory(
            self, f"Choose destination for {demo_key} demo",
            default,
            QFileDialog.ShowDirsOnly | QFileDialog.DontConfirmOverwrite,
        )
        if not dst:
            return
        try:
            layout = self._run_demo_generator(demo_key, dst)
        except Exception as e:
            QMessageBox.warning(self, "Demo generation failed", str(e))
            return

        self._on_nav_selected(target_app)
        widget = self._screens.get(target_app)
        if widget is None:
            return
        try:
            self._apply_demo_to_screen(widget, layout)
            self.statusBar().showMessage(
                f"Loaded {demo_key} demo from {layout.src}", 5000)
        except Exception as e:
            QMessageBox.warning(self, "Demo load failed", str(e))

    def _on_e2e_demo(self) -> None:
        """Confirm, prompt for a folder, download the HF demo dataset,
        then chain Mask -> Measure -> Annotate on it.

        Flow (matches the spec agreed with the user):
          1. Yes/No modal: "do you want to test mask -> Measure ->
             Annotate on a real dataset?"
          2. Folder picker for the local download destination.
          3. QProgressDialog while the toxo_mito + spacr_settings repos
             download in a background thread.
          4. On success, kick off Mask -> Measure -> Annotate. Users
             see the run inside each app's normal console.
        """
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        from pathlib import Path

        answer = QMessageBox.question(
            self, "End-to-end demo",
            "Do you want to test Mask → Measure → Annotate on a real "
            "dataset?\n\n"
            "This will download the toxo_mito demo dataset "
            "(~a few hundred MB) plus the matching settings pack from "
            "Hugging Face, then run the pipeline chain against it.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        default = str(Path.home() / "spacr-demos" / "toxo_mito_e2e")
        dst = QFileDialog.getExistingDirectory(
            self, "Choose folder for the demo dataset",
            default,
            QFileDialog.ShowDirsOnly | QFileDialog.DontConfirmOverwrite,
        )
        if not dst:
            return

        from .hf_download import download_toxo_mito_demo

        def _on_download_done(result, error):
            if result is None:
                QMessageBox.warning(self, "Download",
                    f"The download did not complete:\n{error or 'unknown error'}")
                return
            self.statusBar().showMessage(
                f"Downloaded demo dataset to {result.dataset_path}", 6000)
            self._run_e2e_chain(result.dataset_path,
                                    result.settings_path)

        download_toxo_mito_demo(self, dst, _on_download_done)

    def _run_e2e_chain(self, dataset_path, settings_path) -> None:
        """Run Mask → Measure → Annotate against the freshly-downloaded
        dataset, prompting before each stage.

        The user gets a Continue/Stop popup before each stage kicks off
        so they can inspect the previous run's output before letting
        the next one loose. Non-interactive stages (mask, measure) run
        their pipeline immediately after Continue; the annotate stage
        just opens the annotation UI at the dataset root so the user
        can start labelling.
        """
        from PySide6.QtWidgets import QMessageBox
        from pathlib import Path

        dataset_path  = Path(dataset_path)
        settings_path = Path(settings_path)

        # Helper — load the app's default settings, then override with
        # whatever CSV pack we downloaded for that app.
        def _settings_for(app_key: str) -> dict:
            from .screens.settings_model import resolve_default_settings
            settings = dict(resolve_default_settings(app_key))
            csv = settings_path / f"{app_key}_settings.csv"
            if csv.is_file():
                import csv as _csv
                with csv.open() as fh:
                    for row in _csv.reader(fh):
                        if not row or row[0].startswith("#") or len(row) < 2:
                            continue
                        k, v = row[0].strip(), row[1]
                        if v.lower() in ("true", "false"):
                            v = v.lower() == "true"
                        else:
                            try:
                                v = int(v)
                            except ValueError:
                                try:
                                    v = float(v)
                                except ValueError:
                                    pass
                        settings[k] = v
            settings["src"] = str(dataset_path)
            return settings

        stages = (
            ("mask",     "Mask generation",
             "Ready to start mask generation with the downloaded settings?"),
            ("measure",  "Measurement",
             "Mask stage finished. Ready to start measurement?"),
            ("annotate", "Annotation",
             "Measurement stage finished. Ready to open the annotation UI?"),
        )
        for stage, title, prompt in stages:
            answer = QMessageBox.question(
                self, title, prompt,
                QMessageBox.Yes | QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                self.statusBar().showMessage(
                    f"E2E chain stopped at '{stage}' stage.", 6000)
                return
            settings = _settings_for(stage)
            self._on_nav_selected(stage)
            widget = self._screens.get(stage)
            if widget is None:
                QMessageBox.warning(self, "E2E",
                    f"Couldn't open the '{stage}' screen.")
                return
            try:
                if hasattr(widget, "apply_settings_dict"):
                    widget.apply_settings_dict(settings)
                # Kick off the pipeline automatically for mask + measure;
                # annotate is interactive and opens directly at the
                # loaded dataset so the user can start labelling.
                if stage != "annotate" and hasattr(widget, "_on_run"):
                    widget._on_run()
            except Exception as e:
                QMessageBox.warning(self, f"E2E: {stage} failed", str(e))
                return
        self.statusBar().showMessage(
            "E2E chain launched — check each app's console for progress.",
            8000)

    def _run_demo_generator(self, demo_key: str, dst: str):
        """Isolated for tests — invoke the named generator function
        with `dst` and return whatever it returned."""
        from spacr.qt import synthetic as syn
        _, gen_name = self.DEMO_TARGETS[demo_key]
        gen = getattr(syn, gen_name)
        return gen(dst)

    def _apply_demo_to_screen(self, widget, layout) -> None:
        """Push the demo layout into a target screen, in whatever way
        that screen supports (settings CSV, source folder, or DB path)."""
        from spacr.utils import load_settings

        # AppScreen: load the CSV into its settings model
        if hasattr(widget, "apply_settings_dict") and layout.settings_csv:
            loaded = load_settings(
                str(layout.settings_csv),
                setting_key="Key", setting_value="Value",
            )
            if isinstance(loaded, dict):
                widget.apply_settings_dict(loaded)
                return
        # AnnotateScreen: takes a src folder directly
        if hasattr(widget, "_open_source"):
            widget._open_source(str(layout.src))
            return
        # MakeMasksScreen: opens a folder directly
        if hasattr(widget, "_open_folder"):
            widget._open_folder(str(layout.src))
            return

    def _show_about(self):
        """Show the About panel: mark, name, version, licence, lab.

        Laid out like a macOS About window — mark, name, version, then the
        small print — because that is a shape people already know how to read.

        The licence is named exactly and links to the canonical text rather
        than being paraphrased. PolyForm Noncommercial is not one a reader can
        guess the terms of, and "© Olafsson Lab" alone says nothing about what
        they may actually do with the software.
        """
        from PySide6.QtCore import QSize
        from PySide6.QtWidgets import QDialog, QVBoxLayout

        from .theme import active_palette

        try:
            import spacr
            version = spacr.__version__
        except Exception:
            version = "unknown"

        try:
            import platform

            from PySide6.QtCore import qVersion
            build = (f"Python {platform.python_version()} · Qt {qVersion()}")
        except Exception:
            build = ""

        palette = active_palette()
        dialog = QDialog(self)
        dialog.setWindowTitle("About spaCR")
        dialog.setObjectName("AboutDialog")
        col = QVBoxLayout(dialog)
        col.setContentsMargins(36, 28, 36, 24)
        col.setSpacing(0)

        mark = QLabel()
        mark.setAlignment(Qt.AlignHCenter)
        mark.setStyleSheet("background: transparent;")
        # The PNG straight off disk, not `iconset.icon()`. That helper
        # recolours an icon to the theme's ink so monochrome glyphs stay
        # legible — correct for a toolbar symbol, wrong for a logo, which has
        # its own colours and should look the same on every theme.
        try:
            import os

            from PySide6.QtGui import QPixmap

            pixmap = QPixmap(os.path.join(
                iconset.RESOURCE_DIR, "logo_spacr.png"))
            if not pixmap.isNull():
                mark.setPixmap(pixmap.scaled(
                    QSize(96, 96), Qt.KeepAspectRatio,
                    Qt.SmoothTransformation))
        except Exception:
            pass
        col.addWidget(mark)
        col.addSpacing(14)

        def _line(html, size, *, muted=False, weight=400, gap=0):
            label = QLabel(html)
            label.setAlignment(Qt.AlignHCenter)
            label.setTextFormat(Qt.RichText)
            label.setOpenExternalLinks(True)
            label.setWordWrap(True)
            colour = palette["fg_muted"] if muted else palette["fg"]
            label.setStyleSheet(
                f"background: transparent; color: {colour};"
                f"font-size: {size}px; font-weight: {weight};")
            col.addWidget(label)
            if gap:
                col.addSpacing(gap)

        _line("spaCR", 26, weight=600)
        _line("Spatial phenotype analysis of CRISPR&#8209;Cas9 screens",
              13, muted=True, gap=10)
        _line(f"Version {version}", 12, muted=True)
        _line(build, 11, muted=True, gap=16) if build else col.addSpacing(16)
        _line(
            'Licensed under the '
            '<a href="https://polyformproject.org/licenses/noncommercial/1.0.0">'
            'PolyForm Noncommercial License 1.0.0</a>.<br>'
            'Free for research and other noncommercial use.',
            11, muted=True, gap=10)
        _line("© Olafsson Lab", 11, muted=True)

        dialog.setFixedWidth(420)
        dialog.exec()

    def _open_log_folder(self):
        """Open the ~/.spacr/logs folder in the OS file browser."""
        from .verbose_logger import log_dir
        import webbrowser
        try:
            webbrowser.open(f"file://{log_dir()}")
        except Exception as e:
            self.statusBar().showMessage(
                f"Failed to open log folder: {e}", 5000)

    def _open_preferences(self):
        """Open the Preferences dialog (theme, font size, colour-blind)."""
        try:
            from .preferences import PreferencesDialog
        except Exception as e:
            self.statusBar().showMessage(
                f"Preferences unavailable: {e}", 5000)
            return
        PreferencesDialog(self).exec()
        self.refresh_theme()

    def refresh_theme(self) -> None:
        """Rebuild everything preferences cannot update through QSS alone.

        Three things do not follow a ``setStyleSheet`` call: the Home
        tiles set sizes/margins from the font scale in Python, every
        QIcon baked its pixmap at the theme in force when it was built,
        and the dock's mode and the page's opacity are layout decisions
        rather than colours. Called after the Preferences dialog closes,
        and found by duck-typing from
        :func:`spacr.qt.preferences.apply_preferences_to_app` so a
        preference change from anywhere reaches the widgets.
        """
        try:
            self._sidebar.refresh_icons()
            self._sidebar.refresh_visibility()
        except Exception:
            pass
        try:
            self._refresh_app_action_visibility()
        except Exception:
            pass
        for screen in getattr(self, "_screens", {}).values():
            refresh = getattr(screen, "refresh_maturity_visibility", None)
            if callable(refresh):
                try:
                    refresh()
                except Exception:
                    pass
        try:
            self.apply_dock_mode()
        except Exception:
            pass
        try:
            self._rebuild_startup_page()
        except Exception:
            pass
        self.refresh_language()
        try:
            self._sidebar.refresh_visibility()
        except Exception:
            pass

    def refresh_language(self) -> None:
        """Apply the persisted language to existing static UI text."""
        try:
            from .i18n import retranslate_widget_tree
            retranslate_widget_tree(self)
        except Exception:
            LOG.exception("Could not apply the selected UI language")

    def _refresh_app_action_visibility(self) -> None:
        """Keep the spaCR menu in sync with module maturity preferences."""
        for key, action in getattr(self, "_app_actions", {}).items():
            action.setVisible(app_is_visible(key))
        for demo_key, action in getattr(self, "_demo_actions", {}).items():
            target = self.DEMO_TARGETS.get(demo_key, (demo_key, ""))[0]
            action.setVisible(app_is_visible(target))

    def _rebuild_startup_page(self):
        """Recreate the Home page (e.g. after a font-scale change)."""
        old = getattr(self, "_startup", None)
        was_current = (old is not None
                       and self._stack.currentWidget() is old)
        self._install_startup_page()
        if old is not None:
            self._stack.removeWidget(old)
            # close() before deleteLater() so the outgoing page drops its
            # subscription to the run registry now, rather than staying a
            # live receiver until the deferred delete is flushed.
            try:
                old.close()
            except Exception:
                pass
            old.deleteLater()
        if was_current:
            self._stack.setCurrentWidget(self._startup)

    def _check_for_updates(self):
        """Query PyPI/GitHub in a background thread, prompt to upgrade.

        Both the network call and an accepted ``pip`` upgrade run on
        :class:`_UpdateWorker`; only dialogs and status updates run here.
        """
        try:
            from spacr.updater import check_for_updates
        except Exception as e:
            LOG.exception("Could not import the spaCR updater")
            QMessageBox.warning(self, "Updates",
                                f"Update check unavailable: {e}")
            return

        self.statusBar().showMessage("Checking for updates…", 4000)
        self._start_update_worker(
            "check", check_for_updates, self._on_update_check_done)

    def _start_update_worker(self, operation, fn, on_done) -> None:
        """Start one updater callable and retain it until shutdown."""
        worker = _UpdateWorker(operation, fn, self)
        worker.succeeded.connect(on_done)
        worker.failed.connect(self._on_update_worker_failed)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        self._update_worker = worker

    def _on_update_check_done(self, info) -> None:
        """Handle an :class:`spacr.updater.UpdateInfo` on the GUI thread."""
        if self._closing:
            LOG.debug("Discarding an update result during shutdown")
            return
        if info.error and not info.latest_release:
            QMessageBox.warning(
                self, "Updates",
                f"Couldn't reach update server:\n{info.error}")
            return
        if not info.upgrade_available:
            QMessageBox.information(
                self, "Updates",
                f"You're on {info.installed_version}. No updates.")
            return

        msg = (f"A new version is available.\n\n"
               f"Installed: {info.installed_version}\n"
               f"Latest:    {info.latest_release}\n\n"
               f"Run pip install --upgrade spacr now?")
        if QMessageBox.question(
                self, "Update available", msg) != QMessageBox.Yes:
            return
        try:
            from spacr.updater import run_pip_upgrade
        except Exception as exc:
            LOG.exception("Could not import the spaCR upgrade helper")
            QMessageBox.warning(
                self, "Updates", f"Upgrade unavailable: {exc}")
            return
        self.statusBar().showMessage("Upgrading spaCR…", 4000)
        self._start_update_worker(
            "upgrade", run_pip_upgrade, self._on_upgrade_done)

    def _on_upgrade_done(self, result) -> None:
        """Report a completed package upgrade on the GUI thread.

        ``run_pip_upgrade`` returns ``(exit_code, output)``. A bare int is
        still accepted so an older helper, or a test that patches this with a
        plain return code, does not break.
        """
        if self._closing:
            LOG.debug("Discarding an upgrade result during shutdown")
            return
        if isinstance(result, tuple):
            return_code, output = result
        else:
            return_code, output = result, ""
        if return_code == 0:
            QMessageBox.information(
                self, "Updates",
                "Upgrade finished. Restart spaCR to use it.")
            return
        # These installs launch from a desktop entry with Terminal=false, so
        # "check the terminal for details" named something the user could not
        # open, and the reason was written to a stream nobody was reading.
        # Put the tail of it in the dialog instead.
        lines = [line for line in (output or "").splitlines() if line.strip()]
        detail = "\n".join(lines[-6:]) if lines else "No output was captured."
        QMessageBox.warning(
            self, "Updates",
            f"pip returned exit code {return_code}.\n\n{detail}")

    def _on_update_worker_failed(self, operation: str, details: str) -> None:
        """Report an updater exception instead of losing it in a QThread."""
        if self._closing:
            LOG.debug("Updater %s failed during shutdown:\n%s",
                      operation, details)
            return
        last = next(
            (line for line in reversed(details.splitlines()) if line.strip()),
            "unknown error")
        label = "Update check" if operation == "check" else "Upgrade"
        QMessageBox.warning(self, "Updates", f"{label} failed:\n{last}")

    # -- shutdown ----------------------------------------------------------
    def closeEvent(self, event):
        """Cooperatively drain analysis and UI workers before destruction."""
        from .bridge import registry
        remaining = registry().cancel_all(
            timeout_ms=5000, reason="application shutdown")
        if remaining:
            names = ", ".join(handle.app_key for handle in remaining[:5])
            LOG.warning(
                "Shutdown deferred; workers did not reach a safe boundary: %s",
                names,
            )
            QMessageBox.warning(
                self,
                "Analysis still stopping",
                "spaCR is finishing the current field/trial/job before it can "
                f"close safely ({names}). No worker was force-terminated. "
                "Please close the window again after Stop completes.",
            )
            event.ignore()
            self._closing = False
            return
        self._closing = True
        from .widgets.console_panel import ConsolePanel
        for panel in self.findChildren(ConsolePanel):
            try:
                panel.shutdown()
            except Exception:
                pass
        # Help → "Check for updates…" runs its network call on a QThread
        # parented to this window. Quitting while it's in flight destroys
        # a live QThread, which is the same abort the console drain above
        # exists to prevent. The updater's own socket timeouts are a few
        # seconds, so the wait is bounded twice over.
        worker = getattr(self, "_update_worker", None)
        if worker is not None:
            try:
                worker.wait(5000)
            except RuntimeError:
                pass          # already deleted — nothing left to wait for
        super().closeEvent(event)

    # -- the app-list drawer ----------------------------------------------
    def dock_mode(self) -> str:
        """The user's dock preference — ``auto`` / ``locked`` / ``hidden``.

        Read through here rather than inlined so a headless build, or one
        with an unwritable settings file, still gets the default rather
        than an exception during ``__init__``.
        """
        try:
            from .preferences import get_dock_mode
            return get_dock_mode()
        except Exception:
            return "locked"

    def apply_dock_mode(self, mode: Optional[str] = None) -> None:
        """Put the app list where the preference says it goes.

        Three modes, one ``Sidebar`` object:

        ``auto``    the sidebar lives inside the :class:`EdgeDrawer` and
                    reveals on dwell against the left edge.
        ``locked``  the sidebar is re-parented into the window's dock
                    slot, where it is an ordinary column: it never
                    animates, never overlays the page, and the hot strip
                    is switched off so it cannot also slide in on top of
                    itself.
        ``hidden``  the drawer is closed, its hot strip hidden, and the
                    slot stays empty. The "All apps" action is disabled
                    with a tooltip that says where to turn it back on —
                    a control that silently does nothing is worse than
                    one that is greyed out.

        Idempotent, and safe to call before the menu exists.
        """
        mode = mode or self.dock_mode()
        drawer = getattr(self, "_app_drawer", None)
        sidebar = getattr(self, "_sidebar", None)
        slot = getattr(self, "_dock_slot", None)
        if drawer is None or sidebar is None or slot is None:
            return
        self._dock_mode = mode

        if mode == "locked":
            drawer.close()
            drawer.set_enabled(False)
            if sidebar.parent() is not slot:
                slot.layout().addWidget(sidebar)
            sidebar.setFixedWidth(sidebar.fitting_width())
            sidebar.show()
            slot.show()
        else:
            if sidebar.parent() is not drawer:
                drawer.adopt(sidebar)
            slot.hide()
            drawer.set_enabled(mode == "auto")
            if mode == "hidden":
                drawer.close()

        action = getattr(self, "_act_all_apps", None)
        if action is not None:
            action.setEnabled(mode != "hidden")
            action.setToolTip(
                "The app dock is hidden. Turn it back on in Preferences → "
                "App dock." if mode == "hidden" else
                "Show the full app list.")

    def toggle_app_drawer(self) -> None:
        """Open (and focus) or close the slide-in app list.

        The keyboard and menu path into the reveal, so the panel is not
        hover-only. Note what the drawer is *for* now that Home's first
        tab lists every app: it is not Home's app list, it is the app
        list on **every other screen** — the replacement for the
        permanent 220 px column. From inside Mask, this is the only
        pointer-driven way to reach Measure without going Home first.

        A no-op when the dock is locked (it is already on screen and not
        going anywhere) or hidden (the user asked for it not to be
        there; a shortcut that overrules a preference is a bug).
        """
        if getattr(self, "_dock_mode", "auto") != "auto":
            return
        drawer = getattr(self, "_app_drawer", None)
        if drawer is not None:
            drawer.toggle()

    def _on_drawer_navigated(self, _key: str) -> None:
        """A row in the drawer was clicked — it has done its job, close it.

        Nothing to close when the dock is locked: it is a column, and a
        column that vanished every time you used it would be worse than
        the reveal it replaced.
        """
        if getattr(self, "_dock_mode", "auto") != "auto":
            return
        drawer = getattr(self, "_app_drawer", None)
        if drawer is not None:
            drawer.close()

    # -- navigation -------------------------------------------------------
    def _install_startup_page(self):
        """Instantiate the Home page and add it to the stack."""
        self._startup = make_home_page()
        self._startup.tile_clicked.connect(self._on_nav_selected)
        self._startup.update_check_requested.connect(self._check_for_updates)
        # The hero's "All apps" button is the labelled twin of the edge
        # reveal — a discoverable way in for anyone who never finds the
        # hot strip, and the thing a screenshot can point at.
        try:
            self._startup._btn_all_apps.clicked.connect(self.toggle_app_drawer)
        except Exception:
            pass
        self._stack.addWidget(self._startup)

    def _on_nav_selected(self, key: str):
        """Navigate to app ``key``, lazily instantiating its screen on first use."""
        if key == "__home__":
            # Re-read the things that go stale while Home is off screen:
            # the plate queue, the run journal and the disk/GPU figures.
            # Cheap (a JSON read and three stat calls) and only on a
            # deliberate return to Home, not on a timer.
            try:
                self._startup.refresh()
            except Exception:
                pass
            self._stack.setCurrentWidget(self._startup)
            self._status_app_label.setText("Home")
            self.statusBar().showMessage("Home", 2000)
            return
        if key not in self._screens:
            self._screens[key] = self._build_screen(key)
            # Every screen gets the same page treatment here, because this is
            # the one place they all pass through. It cannot live in
            # `AppScreen`: most screens are not AppScreens — Annotate, Align &
            # Stitch, Format Converter, Import Project, Plate Queue, Batch
            # Runner, Distributed Jobs, Database Browser, Make Masks, Model
            # Compare, Model Zoo, Plate Viewer, Annotator Agreement, Training
            # Runs, Classifier Evaluation, Run History and Report are plain
            # QWidget trees, so they never got the backdrop or the surface
            # clearing and sat as black slabs while the pipeline screens did
            # not.
            try:
                self._theme_screen(self._screens[key], key)
            except Exception:
                # Decoration must never stop a screen from opening.
                LOG.exception("Could not theme the %s screen", key)
            self._stack.addWidget(self._screens[key])
            try:
                from .i18n import retranslate_widget_tree
                retranslate_widget_tree(self._screens[key])
            except Exception:
                LOG.exception("Could not translate the %s screen", key)
        self._stack.setCurrentWidget(self._screens[key])
        # Move this app to the end of the visit list. Revisiting an app
        # has to count as the most recent visit — otherwise "Add current
        # plate" on the Queue screen picks up whichever app was OPENED
        # last rather than the one that was on screen a moment ago.
        if key in self._visit_order:
            self._visit_order.remove(key)
        self._visit_order.append(key)
        # Find nice display name
        from .i18n import tr
        name = tr(next((n for k, n, _d, _s in APPS if k == key), key))
        self._status_app_label.setText(name)
        self.statusBar().showMessage(tr("Opened {name}", name=name), 2000)

    def _on_zoo_compare_requested(self, request: dict) -> None:
        """Open Model Compare preloaded with the two models the zoo selected.

        Goes through ``ModelCompareScreen.configure`` rather than reaching into
        the panels, so restructuring either panel cannot silently break the
        hand-off.

        :param request: ``{'model_a', 'model_b', 'folder', 'n_fields'}``.
        """
        self._on_nav_selected("model_compare")
        screen = self._screens.get("model_compare")
        if screen is None or not hasattr(screen, "configure"):
            return
        screen.configure(
            model_a=request.get("model_a", ""),
            model_b=request.get("model_b", ""),
            folder=request.get("folder", ""),
            n_fields=int(request.get("n_fields", 0) or 0),
        )

    def _theme_screen(self, screen: QWidget, key: str) -> None:
        """Clear a screen's containers and give it the ambient backdrop.

        Skipped for anything that already handles its own: ``AppScreen`` does
        both in its constructor, and the sequencing screen has the DNA rain.
        """
        from .screens.app_screen import AppScreen, uses_ambient_background
        from .theme import clear_container_surfaces

        if isinstance(screen, AppScreen):
            return

        clear_container_surfaces(screen)

        if not uses_ambient_background(key):
            return
        try:
            from .preferences import (get_ambient_enabled, get_ambient_palette,
                                      get_ambient_theme, resolve_effective_theme,
                                      theme_background_path)
            if not get_ambient_enabled():
                return
            from .widgets.ambient import install_ambient
            install_ambient(
                screen, None,
                theme=get_ambient_theme(), palette=get_ambient_palette(),
                backdrop=theme_background_path(resolve_effective_theme()))
        except Exception:
            LOG.exception("Could not install the backdrop for %s", key)

    def _build_screen(self, key: str) -> QWidget:
        """Return a freshly-built screen widget for the given app ``key``.

        Construction only. The page treatment — clearing the containers and
        installing the ambient backdrop — is :meth:`_theme_screen`, applied by
        the caller that puts the screen on screen. Keeping them apart matters:
        `tests/qt/test_all_module_smoke.py` calls this unbound against a
        stand-in host and inspects its bytecode for the ``self._on_*`` slots it
        wires, so a wrapper here hides those names and breaks that contract.
        """
        try:
            from spacr.plugins import get_app, load_object
            plugin_app = get_app(key)
        except Exception:
            LOG.exception("Could not inspect plugin screen contribution %s", key)
            plugin_app = None
        if plugin_app is not None and plugin_app.screen_factory:
            try:
                factory = load_object(plugin_app.screen_factory)
                screen = factory(app_key=key)
                if not isinstance(screen, QWidget):
                    raise TypeError(
                        f"{plugin_app.screen_factory} returned "
                        f"{type(screen).__name__}, expected QWidget"
                    )
                return screen
            except Exception:
                LOG.exception("Could not build plugin screen for %s", key)
                raise
        registered = registered_factory(key)
        if registered is not None:
            screen = _call_screen_factory(registered, key, self)
            if not isinstance(screen, QWidget):
                raise TypeError(
                    f"the factory registered for {key!r} returned "
                    f"{type(screen).__name__}, expected QWidget")
            return screen
        if key == "annotate":
            from .screens.annotate import AnnotateScreen
            screen = AnnotateScreen()
            screen.train_requested.connect(self._on_train_requested)
            return screen
        if key == "make_masks":
            from .screens.make_masks import MakeMasksScreen
            return MakeMasksScreen()
        if key == "queue":
            from .screens.queue import QueueScreen
            screen = QueueScreen()
            screen.wire_add_current(self._snapshot_current_screen_settings)
            return screen
        if key == "db_browser":
            from .screens.db_browser import DbBrowserScreen
            return DbBrowserScreen()
        if key == "agreement":
            from .screens.agreement import AgreementScreen
            return AgreementScreen()
        if key == "plate_view":
            from .screens.plate_view import PlateViewScreen
            return PlateViewScreen()
        if key == "model_compare":
            from .screens.model_compare import ModelCompareScreen
            return ModelCompareScreen()
        if key == "align":
            from .screens.align import AlignScreen
            return AlignScreen()
        if key == "convert":
            from .screens.convert import ConvertScreen
            return ConvertScreen()
        if key == "foreign":
            from .screens.foreign import ForeignScreen
            return ForeignScreen()
        if key == "batch":
            from .screens.batch import BatchScreen
            return BatchScreen()
        if key == "distributed_jobs":
            from .screens.distributed_jobs import DistributedJobsScreen
            return DistributedJobsScreen()
        if key == "model_zoo":
            from .screens.model_zoo import ModelZooScreen
            screen = ModelZooScreen()
            screen.compare_requested.connect(self._on_zoo_compare_requested)
            return screen
        if key == "report":
            from .screens.report import ReportScreen
            return ReportScreen()
        if key == "train_compare":
            from .screens.train_compare import TrainCompareScreen
            return TrainCompareScreen()
        if key == "classifier_evaluation":
            from .screens.classifier_evaluation import ClassifierEvaluationScreen
            return ClassifierEvaluationScreen()
        if key == "run_history":
            from .screens.run_history import RunHistoryScreen
            screen = RunHistoryScreen()
            screen.settings_requested.connect(self._on_train_requested)
            return screen
        from .screens.app_screen import AppScreen
        screen = AppScreen(app_key=key)
        screen.error_explain_requested.connect(self._on_explain_error)
        screen.remote_submit_requested.connect(
            self._on_remote_submit_requested
        )
        return screen

    def _snapshot_current_screen_settings(self):
        """Return ``(app_key, settings_dict)`` for the AppScreen the user
        was looking at when they hit "Add current plate" on the Queue
        screen. Raises when the active screen isn't a normal app."""
        widget = self._stack.currentWidget()
        # Prefer the most-recently-viewed AppScreen — the Queue screen
        # itself isn't one.
        from .screens.app_screen import AppScreen
        if isinstance(widget, AppScreen):
            return widget.app_key, dict(widget._settings_model.collect())
        # Fall back to the last non-queue AppScreen the user visited.
        # Walk the VISIT order, not `_screens` (creation) order: a user
        # who opens Mask, then Measure, then goes back to Mask and hits
        # "Add current plate" means Mask — creation order would hand
        # them Measure's settings under Mask's nose.
        for key in reversed(self._visit_order):
            scr = self._screens.get(key)
            if isinstance(scr, AppScreen):
                return scr.app_key, dict(scr._settings_model.collect())
        raise RuntimeError(
            "No active plate settings — open Mask/Measure/Classify first.")

    def _on_explain_error(self, traceback_text: str, active_app: str) -> None:
        """Legacy hook — the AI now lives inside each AppScreen's
        Console panel, which handles Explain-error directly. This
        method is kept only for backward-compat with subclasses."""
        pass

    def _on_remote_submit_requested(
        self, app_key: str, settings: dict
    ) -> None:
        """Open Distributed Jobs with a snapshot of the current module."""
        self._on_nav_selected("distributed_jobs")
        screen = self._screens.get("distributed_jobs")
        if screen is not None and hasattr(screen, "configure_submission"):
            screen.configure_submission(app_key, settings)

    def _on_train_requested(self, target_key: str, seed: dict) -> None:
        """Navigate to `target_key` (creating the screen if needed) and
        push `seed` values into its settings model. Called by the
        annotate screen's Train CV / Train XG buttons."""
        self._on_nav_selected(target_key)
        widget = self._screens.get(target_key)
        if widget is None:
            return
        model = getattr(widget, "_settings_model", None)
        if model is None:
            return
        widgets = getattr(model, "_widgets", {})
        for key, value in seed.items():
            w = widgets.get(key)
            if w is None:
                continue
            try:
                self._apply_seed_value(w, value)
            except Exception:
                LOG.warning(
                    "Could not seed %s.%s with %r",
                    target_key, key, value, exc_info=True)

    @staticmethod
    def _apply_seed_value(w: QWidget, value) -> None:
        from PySide6.QtWidgets import (
            QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox,
        )
        setter = getattr(w, "set_value", None)
        if callable(setter):
            setter(value)
        elif isinstance(w, QCheckBox):
            w.setChecked(bool(value))
        elif isinstance(w, QSpinBox):
            w.setValue(int(float(value)))
        elif isinstance(w, QDoubleSpinBox):
            w.setValue(float(value))
        elif isinstance(w, QComboBox):
            for i in range(w.count()):
                if w.itemData(i) == value or w.itemText(i) == str(value):
                    w.setCurrentIndex(i)
                    break
        elif isinstance(w, QLineEdit):
            w.setText("" if value is None else str(value))


def _load_bundled_fonts() -> None:
    """Register the bundled Open Sans TTFs with :class:`QFontDatabase`.

    Idempotent — the fonts are only loaded once even if called
    multiple times (Qt tracks the file path).
    """
    from PySide6.QtGui import QFontDatabase
    here = os.path.dirname(os.path.abspath(__file__))
    fonts_dir = os.path.join(here, "resources", "fonts")
    if not os.path.isdir(fonts_dir):
        return
    for name in os.listdir(fonts_dir):
        if name.lower().endswith((".ttf", ".otf")):
            QFontDatabase.addApplicationFont(os.path.join(fonts_dir, name))


def launch(argv: Optional[list[str]] = None) -> int:
    """Bootstrap QApplication and show the main window."""
    if argv is None:
        argv = sys.argv[1:]

    # Support `spacr-qt <app>` to open directly into an app.
    initial_app = argv[0] if argv else None

    # Enable high-DPI early.
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    app = QApplication(sys.argv[:1])
    app.setApplicationName("spaCR")
    app.setOrganizationName("Olafsson Lab")
    app.setWindowIcon(QIcon(os.path.join(
        iconset.RESOURCE_DIR, "app_icon.png")))

    # Lift Qt's default 256 MB QImageReader allocation limit. Large multi-panel
    # figures rendered at high DPI decode to well over 256 MB, and hitting the
    # limit makes QPixmap loads fail (blank figures) and the UI hang. 0 = no
    # limit; the figure queue still caps display resolution for sanity.
    try:
        from PySide6.QtGui import QImageReader
        QImageReader.setAllocationLimit(0)
    except Exception:
        LOG.debug("This Qt build does not expose QImageReader allocation "
                  "limits", exc_info=True)

    # Bundle Open Sans (Regular + Light + SemiBold) so the app renders
    # the same on every OS regardless of what fonts the user has
    # installed. Registered before applying the stylesheet so any
    # `font-family: "Open Sans"` rule resolves.
    _load_bundled_fonts()

    # Apply user preferences (theme + font scale) — falls back to the
    # dark defaults on the first launch when nothing is stored yet.
    from .preferences import apply_preferences_to_app
    apply_preferences_to_app(app)

    # Real Python logging → rotating file + Qt signal so ConsolePanel
    # can render records inline. Set it up before the launch breadcrumb and
    # MainWindow construction so neither is lost.
    from .logging_util import setup_logging
    setup_logging()

    # Every launch drops a timeline marker into the diagnostic log.
    import logging as _lg
    import sys as _sys
    from .verbose_logger import current_log_file
    _lg.getLogger("spacr").info(
        "spaCR launched (python=%s.%s.%s, log=%s)",
        _sys.version_info.major, _sys.version_info.minor,
        _sys.version_info.micro, current_log_file())

    win = MainWindow(initial_app=initial_app)
    # Opens at its own size rather than maximised. Maximising assumes a
    # desktop: over X11 forwarding, VNC or a virtual framebuffer the
    # "available geometry" is whatever the remote session claims, which is
    # frequently one enormous virtual desktop or a 640x480 stub, and the
    # window arrives unusable either way. The user can still maximise it,
    # and the 1200x720 minimum this window declares is a sane opening size
    # on a real display.
    win.show()

    # Pre-warm the heavy imports that a module screen needs (spacr.gui_utils
    # pulls torch + cv2 ≈ 3-4 s; spacr.settings ≈ 1 s) in a BACKGROUND thread
    # while the user looks at the home screen. By the time they open a module
    # these are cached, so the module snaps open instead of freezing on the
    # first import. Importing modules (no Qt objects) off-thread is safe.
    def _prewarm():
        try:
            import importlib
            for mod in ("spacr.settings", "spacr.gui_utils"):
                importlib.import_module(mod)
        except Exception:
            LOG.debug("Could not prewarm GUI settings imports", exc_info=True)
    threading.Thread(target=_prewarm, name="spacr-prewarm",
                     daemon=True).start()

    # aboutToQuit fires no matter how the app exits (window closed,
    # Ctrl+C, SIGTERM, …). Belt-and-suspenders with MainWindow's
    # closeEvent: ensure every ConsolePanel drains its AI thread
    # before Qt starts destroying widgets.
    def _drain_ai():
        from .widgets.console_panel import ConsolePanel
        for panel in win.findChildren(ConsolePanel):
            try:
                panel.shutdown()
            except Exception:
                LOG.debug("Could not shut down a console panel",
                          exc_info=True)
        # Also kill any subprocess still tracked by a provider
        try:
            from . import ai as _ai
            for p in _ai.list_providers():
                p.cancel_stream()
        except Exception:
            LOG.debug("Could not cancel every AI provider", exc_info=True)
    app.aboutToQuit.connect(_drain_ai)

    return app.exec()
