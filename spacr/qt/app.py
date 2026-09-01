"""
QApplication bootstrap + MainWindow.

`launch(argv)` is the public entry point called by `spacr-qt` and
`python -m spacr.qt`.
"""
from __future__ import annotations

import importlib as _importlib
import inspect
import logging
import os
import sys
import threading

from . import timing as _timing
import traceback
from typing import List, Optional, Tuple

from PySide6.QtCore import (QEvent, QObject, QSize, Qt, QThread,
                            Signal)
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QScrollArea,
    QStackedWidget,
    QStatusBar,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


from .hidpi import scaled_for
from .i18n import tr
from . import iconset
# The declared registry rows and the stand-in that defers a screen's import
# until it is built. Cheap on purpose: `app_catalog` imports nothing beyond
# `importlib` and `inspect`, and reads `register_app` back out of this module
# from inside a function, so naming it here is not a cycle.
from .app_catalog import (LazyScreenFactory, declared_for as _declared_for,
                          register_declared as _register_declared)
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


class _FractalFollowsItsScreen(QObject):
    """Keeps the spaceout backdrop the same size as the screen behind it."""

    def __init__(self, widget, screen) -> None:
        super().__init__(screen)
        self._widget = widget

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.Resize:
            try:
                self._widget.setGeometry(watched.rect())
            except Exception:                                # noqa: BLE001
                pass
        return False


def install_the_spaceout_fractal(screen) -> bool:
    """Put the spaceout fractal behind ``screen``, if this is spaceout.

    :returns: True when it was installed, so the caller knows to skip the
        ordinary ambient backdrop. False in every normal launch -- which is
        what keeps the mode hidden -- and false rather than raising when the
        fractal cannot be built, so a machine that cannot draw it still gets
        the animation it always had.

    The settings come from Preferences, which shows them only under
    spaceout, so one place decides both what is offered and what is drawn.
    """
    try:
        from .theme import spaceout_enabled

        if not spaceout_enabled():
            return False
        from .preferences import get_fractal_settings
        from .widgets.fractal_travel import (
            RuntimeControls, Settings, create_fractal_widget)

        values = get_fractal_settings()
        widget = create_fractal_widget(
            Settings(backend=values["backend"], quality=values["quality"],
                     scale=values["scale"]),
            RuntimeControls(speed=values["speed"], dream=values["dream"],
                            variable_speed=values["variable_speed"]),
        )
    except Exception:                                        # noqa: BLE001
        LOG.exception("Could not build the spaceout fractal")
        return False

    try:
        widget.setParent(screen)
        widget.setGeometry(screen.rect())
        widget.lower()
        widget.show()
        # It follows the screen's geometry the way the ambient backdrop
        # does; without this it keeps its first size and a resized window
        # shows bare ground beside it.
        screen.installEventFilter(_FractalFollowsItsScreen(widget, screen))
    except Exception:                                        # noqa: BLE001
        LOG.exception("Could not place the spaceout fractal")
        try:
            widget.shutdown()
        except Exception:                                    # noqa: BLE001
            pass
        return False
    return True


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
        from .bridge import emit_safely

        try:
            result = self._fn()
        except Exception:
            details = traceback.format_exc()
            LOG.exception("Updater %s failed", self.operation)
            # `emit_safely`: an exception out of a QThread::run override
            # aborts the process, and the window may be gone by now.
            emit_safely(self.failed, self.operation, details)
            return
        emit_safely(self.succeeded, result)


#: Held while a heavy module is imported AND while an OpenGL context is
#: created. The preloader used to run on the GUI thread for exactly one
#: reason -- "avoids concurrent Qt, CUDA, and OpenGL initialization" -- and
#: that hazard is real: torch brings CUDA up, and the spaceout fractal brings
#: a GL context up. Serialising the two is what makes the worker thread safe,
#: so the reason to stay on the GUI thread is answered rather than ignored.
HEAVY_IMPORT_LOCK = threading.Lock()


class _DragsTheWindowByTheMenuBar(QObject):
    """Let a frameless window be moved by its menu bar.

    The window is frameless, and the comment at its construction says "the
    menu bar is what you drag it by" -- but nothing implemented that, so the
    window could not be moved at all.

    `startSystemMove()` hands the drag to the compositor, which is the only
    thing that works under Wayland: tracking mouse deltas and calling
    `move()` is ignored there, because a Wayland client does not get to
    choose where its surface sits.

    A press over a menu ACTION is left alone -- `actionAt` answers for that
    -- so opening a menu still opens it, and only the empty strip drags.
    """

    def __init__(self, window):
        super().__init__(window)
        self._window = window

    def eventFilter(self, watched, event):    # noqa: N802 - Qt naming
        if event.type() != QEvent.Type.MouseButtonPress:
            return False
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        try:
            if watched.actionAt(event.position().toPoint()) is not None:
                return False                  # a menu, not the bare strip
            handle = self._window.windowHandle()
            if handle is None:
                return False
            handle.startSystemMove()
            return True
        except Exception:                     # noqa: BLE001
            LOG.debug("could not start a window move", exc_info=True)
            return False


class _PipelinePreloader:
    """Import pipeline modules on a worker thread.

    MEASURED, because the GUI-thread version was costing more than it looked
    like. Importing these seven on the GUI thread, yielding between each,
    left the interface answering THREE timer ticks in 4.24 seconds with a
    single 1408 ms freeze -- 94% of it in the two modules that pull torch,
    and no amount of yielding breaks up one 2.3-second import. On a worker
    thread the same 4.2 seconds gives 224 ticks and a longest freeze of
    185 ms, which is the GIL during a C-extension import and is not
    avoidable by any threading.

    The original docstring's reason for staying on the GUI thread was
    concurrent Qt/CUDA/OpenGL initialisation. That is answered by
    :data:`HEAVY_IMPORT_LOCK` rather than by giving up the thread: the
    fractal's GL canvas takes the same lock, so an import and a context
    creation can never overlap.

    Progress is delivered on the GUI thread by a poll, not by calling back
    from the worker: `on_step` and `on_done` take a loading screen down, and
    touching widgets from a worker is the crash this file has already had
    once today.
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

    def __init__(self, on_step=None, on_done=None):
        """
        :param on_step: called with (completed, total) after each import, so a
            loading screen can show honest progress -- the denominator is
            known before the first module is touched.
        :param on_done: called once, after the last import. This is what takes
            the loading screen down; without it the screen would have to poll.
        """
        self._i = 0
        self._started = False
        self._on_step = on_step
        self._on_done = on_done
        self._done = threading.Event()
        self._reported = 0
        self._poll = None
        self._thread = None

    def total(self) -> int:
        """How many modules will be imported."""
        return len(self._MODULES)

    def start(self) -> None:
        """Begin importing on a worker thread (no-op if already begun)."""
        from PySide6.QtCore import QTimer

        if self._started:
            return
        self._started = True
        self._i = 0
        self._thread = threading.Thread(
            target=self._work, name="spacr-preload", daemon=True)
        self._thread.start()
        # THE CALLBACKS RUN HERE, on the GUI thread. A worker that called
        # `on_step` directly would be touching a loading screen from off the
        # GUI thread, which is the crash this file has already had once.
        self._poll = QTimer()
        self._poll.setInterval(40)
        self._poll.timeout.connect(self._drain)
        self._poll.start()

    def _work(self) -> None:
        """Import every module, one at a time, holding the heavy lock."""
        import importlib

        for module in self._MODULES:
            try:
                with HEAVY_IMPORT_LOCK:
                    importlib.import_module(module)
            except Exception:                                # noqa: BLE001
                LOG.debug("preloading %s failed", module, exc_info=True)
            self._i += 1
        self._done.set()

    def _drain(self) -> None:
        """Report whatever the worker has finished since the last tick."""
        while self._reported < self._i:
            self._reported += 1
            if self._on_step is not None:
                try:
                    self._on_step(self._reported, len(self._MODULES))
                except Exception:                            # noqa: BLE001
                    LOG.debug("preload progress callback failed",
                              exc_info=True)
        if not self._done.is_set():
            return
        if self._poll is not None:
            self._poll.stop()
            self._poll = None
        if self._on_done is not None:
            try:
                self._on_done()
            except Exception:                                # noqa: BLE001
                LOG.debug("preload completion callback failed", exc_info=True)
            self._on_done = None

    def wait(self, timeout: float = 30.0) -> bool:
        """Block until the imports finish. For tests and for shutdown."""
        return self._done.wait(timeout)



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
SECTION_ASSAYS = "Assays"
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
#: The instruments: the things you point AT a project rather than steps
#: the pipeline runs on its own. Editing masks by hand, stitching tiles,
#: reading an embedding, drawing a gate, building a plot, checking
#: quality. Introduced 2026-08-31 when Home was cut to four categories.
SECTION_TOOLS = "Tools"

#: Every section an app may be filed under, in workflow order: the
#: end-to-end pipeline first, then getting data in and running it at
#: scale, then the segmentation models that pipeline depends on, then
#: reading the results, then asking them questions, then the
#: Toxoplasma-specific assays, and finally planning the next experiment.
#:
#: This is the DECLARATION. :data:`SECTIONS` is the subset that has apps
#: — see :func:`_refresh_sections`.
#: FOUR CATEGORIES, in this order. Cut down from seven on 2026-08-31 --
#: the user wrote out the tiles they wanted and these are they. The other
#: three constants above are kept because screens, tests and saved state
#: still name them, and because a section with no apps simply does not
#: draw: :data:`SECTIONS` is the subset that has any, so removing the
#: names would break imports without changing a single pixel.
SECTION_ORDER = (SECTION_CORE, SECTION_DATA, SECTION_TOOLS, SECTION_ASSAYS)

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
    SECTION_CORE: ("Core sequence from microscopy images through "
                   "segmentation, measurements, annotations, classification, "
                   "barcode mapping and regression."),
    SECTION_DATA: ("Import images and tables into spaCR projects and execute "
                   "reproducible multi-plate workflows."),
    SECTION_MODELS: ("Develop, evaluate and manage segmentation and "
                     "classification models."),
    SECTION_RESULTS: ("Review outputs and quality-control results, then "
                      "prepare results for reporting or export."),
    SECTION_EXPLORE: ("Explore measurements using visualization, tabulation, "
                      "gating and feature-analysis tools."),
    SECTION_TOOLS: ("Point these at a project: edit masks by hand, stitch "
                    "tiles, read an embedding, draw a gate, build a plot, "
                    "check quality."),
    SECTION_ASSAYS: "Quantitative readouts for biological assays.",
    SECTION_DESIGN: ("Plan statistical power, sample size, plate layouts, "
                     "controls and replicates."),
}

_PLUGIN_SECTION_MAP = {
    "core": SECTION_CORE,
    "data": SECTION_DATA,
    "models": SECTION_MODELS,
    "results": SECTION_RESULTS,
    "toxo": SECTION_ASSAYS,
    # `spacr.plugins._SECTIONS` does not accept these two yet, so no
    # plugin can reach them today. They are mapped here so that widening
    # the plugin allow-list is a one-line change there rather than a
    # KeyError that takes every plugin app down with it — the loop below
    # catches per-registry, not per-plugin.
    "explore": SECTION_EXPLORE,
    "design": SECTION_DESIGN,
    "tools": SECTION_TOOLS,
}

#: Hard cap on apps per section. Enforced by tests, not at runtime — a
#: violation is a design mistake to fix in this table, not something to
#: discover at startup.
#:
#: Twenty leaves room for established sections to grow while still catching a
#: registry that has become too dense to scan as one group.
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
    """The factory registered for ``key``, or ``None``.

    A declared app is registered with a
    :class:`~spacr.qt.app_catalog.LazyScreenFactory` standing in for the real
    callable, and this is where the stand-in ends: asking for the factory
    imports the screen and returns the module's own function.

    That is deliberate, and it is what makes laziness invisible to everything
    downstream. :func:`_call_screen_factory` decides whether to pass
    ``app_key`` and ``host`` by reading the factory's signature, and a
    stand-in's signature is its own, not the real one's; a test that asserts
    the registered factory ``is`` the module's function would likewise be
    comparing against the wrapper. Resolving here means neither ever sees one.

    The resolved callable replaces the stand-in in :data:`APP_FACTORIES`, so
    the import happens once even if the screen is opened and closed all
    afternoon. A stand-in whose module fails to import is left in place and
    ``None`` is returned: the app falls back to the generic settings screen
    rather than taking the window down.
    """
    factory = APP_FACTORIES.get(key)
    if isinstance(factory, LazyScreenFactory):
        try:
            factory = factory.resolve()
        except Exception:
            LOG.exception("Could not import the screen registered for %s", key)
            return None
        APP_FACTORIES[key] = factory
    return factory


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
    #    called. Ctrl+1..6 map to these six before Ctrl+7..9 continue into
    #    the next apps in sidebar order.
    # CORE IS THE PIPELINE, IN THE ORDER YOU RUN IT: mask, measure,
    # annotate, classify, map barcodes, regression. Nothing else belongs in
    # it -- Timelapse and the Motility assay are assays and are filed as
    # such, and a section that lists everything is a section that sorts
    # nothing.
    ("mask",           "Mask",           "Generate segmentation masks for cells, nuclei, pathogens and organelles from microscopy images using Cellpose and supported alternatives", SECTION_CORE),
    ("measure",        "Measure",        "Quantify per-object intensity and morphology features",       SECTION_CORE),
    ("annotate",       "Annotate",       "Assign annotations to single-object images and store them in the project database",  SECTION_CORE),
    # ONE CLASSIFY SCREEN. "Classify (CV)" and "Classify (ML)" were the two
    # originals kept beside the merged one so a saved settings CSV would
    # keep working -- but three entries for one job is three places to look
    # and two of them are the same run with half the choices. Removed on
    # 2026-08-23 at the maintainer's instruction.
    #
    # THE ENTRY POINTS ARE UNTOUCHED: `deep_spacr` and `generate_ml_scores`
    # are what `classify.classify` dispatches to, so a notebook importing
    # either still works and a settings CSV for either still runs -- through
    # the one screen, which reads `classifier_family`.
    ("classify_merged", "Classify",      "Train classifiers on image crops with PyTorch or on measured features with gradient boosting", SECTION_CORE),
    ("map_barcodes",   "Map Barcodes",   "Map sequencing barcodes to screen data",                      SECTION_CORE),
    ("regression",     "Regression",     "Regression analysis of screen scores",                        SECTION_CORE),
    ("train_compare",  "Training Runs",  "Compare training curves and settings across multiple runs", SECTION_CORE),
    ("convert",        "Format Converter", "Convert ND2, CZI, LIF and OME-TIFF images to Yokogawa TIFF layout and record source mappings", SECTION_DATA),
    ("foreign",        "Import",         "Bring images, masks and measurement tables into a spaCR project -- converting microscope formats, mapping source columns, or adopting masks made elsewhere", SECTION_DATA),
    ("external_masks", "External Masks", "Import images and externally generated label masks as a measured spaCR project ready for annotation", SECTION_DATA),
    ("queue",          "Plate Queue",    "Execute the same processing pipeline across multiple plates", SECTION_DATA),
    ("batch",          "Batch Runner",   "Queue modules, plates and settings for unattended sequential execution", SECTION_DATA),
    ("distributed_jobs", "Distributed Jobs", "Submit and monitor spaCR runs on SSH workstations, Slurm or cloud/HPC commands", SECTION_DATA),
    ("db_browser",     "Database Browser", "Browse, filter and export tables from measurements.db", SECTION_DATA),
    # CLASSIFIER EVALUATION, EXPLAIN CV MODEL AND ACTIVATION ARE BUTTONS ON
    # CLASSIFY.
    # A classifier is trained on one screen and argued about on two
    # others, so both fold onto the Classify masthead
    # (`spacr.qt.screens.classify`) and open their own screen as a page
    # beside the training settings. Neither has a row here any more; what
    # each tile said is `map_barcodes.FOLD_FALLBACK`, and every table a
    # row used to feed -- the drop handler, the API link, the header, the
    # translated name -- names them directly instead.
    ("run_history",    "Run History",    "Search run settings, outputs, warnings, failures and performance metrics", SECTION_DATA),
    ("report",         "Report",         "Generate shareable HTML or PDF reports containing QC results, figures, statistics, settings and software versions", SECTION_DATA),
    # -- Data & batch runs: get images and tables into a spaCR project,
    #    run many plates unattended, get the numbers back out.
    ("align",          "Align & Stitch", "Register and stitch image tiles into an incrementally written mosaic with bounded memory use", SECTION_TOOLS),
    # -- Segmentation models: build, train, pick and check the Cellpose
    #    models the Mask step runs.
    # Not a training screen despite where it sits: MakeMasksScreen is the
    # brush, the flood fill and the object operations, i.e. correcting a
    # mask by hand. It carried Train Cellpose's description verbatim, which
    # is the app directly below it.
    # DATA, NOT MODELS. Make Masks does not train, choose or run a
    # segmentation model: it is hand curation of masks that already exist,
    # which is the same kind of work as the other tools filed under Data.
    ("make_masks",     "Make Masks",     "Edit segmentation masks with brush, flood-fill, relabel, fill and small-object removal tools",  SECTION_TOOLS),
    # THE SEGMENTATION WORKBENCH HAS NO SATELLITE TILES. Training a model,
    # comparing two of them, browsing the zoo and curating a mask by hand are
    # all one loop -- segment, look, correct, train, segment again -- and they
    # were four rows the user had to leave the loop to reach. They are buttons
    # on the Make Masks masthead now (`make_masks.FOLD_ORDER`), each opening
    # the module's own screen as a page beside the editor.
    #
    # THE KEYS ARE STILL REAL, which is the whole difficulty of dropping the
    # rows: `spacr-run train_cellpose` runs, a settings file written for it
    # still loads, a file dropped on the page still lands, and `spacr-run
    # model_compare / model_zoo / curate` still say what to do instead --
    # `cli.INTERACTIVE_ONLY` holds those three sentences in its own literal
    # now that no row carries a `cli_note=`. What went is the tile.
    # -- Results & QC: look at what came out, decide whether to believe it,
    #    and hand it to someone else.
    ("plate_view",     "Plate Viewer",   "Visualize measurements as plate heatmaps and detect edge effects",  SECTION_TOOLS),
    # ANNOTATOR AGREEMENT HAS NO ROW. Scoring how well two annotation
    # passes agree is the sentence after annotating them, so it is a
    # button on the Annotate masthead that opens its own screen, whole
    # (`spacr.qt.screens.annotate`). `cli.INTERACTIVE_ONLY` still names
    # it, so `spacr-run agreement` still says where to find it.
    ("umap",           "Image UMAP",     "Visualize UMAP embeddings with image glyphs",                  SECTION_TOOLS),
    # -- Toxoplasma assays: parasite-specific readouts.
    #
    # TIMELAPSE AND MOTILITY HAVE NO ROW. Timelapse is the mask pipeline
    # with tracking on, so it is a switch on the Mask Generation masthead
    # that reveals its own settings categories (`spacr.qt.screens.mask`);
    # the Motility Assay reads finished masks and writes a measurements
    # table, so it is a button on the Measure masthead that opens its own
    # screen (`spacr.qt.screens.measure`). Both still run from
    # `spacr-run`, from a settings CSV and from a chained hand-off --
    # `spacr.cli.MODULES`, `validate.APP_FUNCTIONS` and
    # `bridge.resolve_pipeline_entry` all still know them. What went is
    # the tile, not the module.
    ("analyze_plaques", "Plaque Assay",  "Quantify plaque assay measurements",                          SECTION_ASSAYS),
    ("recruitment",    "Recruitment",    "Quantify molecular recruitment measurements",                 SECTION_ASSAYS),
    ("invasion",       "Invasion Assay", "Quantify attached and invaded parasites using two-colour differential staining and calculate invasion efficiency per well", SECTION_ASSAYS),
    ("replication",    "Replication Assay", "Quantify parasites per vacuole and calculate replication rates by condition", SECTION_ASSAYS),
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
    # -- alpha: built and reachable, not yet trusted end to end (16)
    # The merged Classify module is new. "stable" is the ABSENCE of a line
    # here, so leaving it out would have claimed a maturity it has not
    # earned -- it dispatches to two pipelines that ARE trusted, but the
    # merged screen itself has not been run on real data yet.
    "classify_merged": STAGE_ALPHA,
    "align":           STAGE_ALPHA,
    "convert":         STAGE_ALPHA,
    "foreign":         STAGE_ALPHA,
    "external_masks":  STAGE_ALPHA,
    "queue":           STAGE_ALPHA,
    "batch":           STAGE_ALPHA,
    "distributed_jobs": STAGE_ALPHA,
    "invasion":        STAGE_ALPHA,
    "db_browser":      STAGE_ALPHA,
    "plate_view":      STAGE_ALPHA,
    "train_compare":   STAGE_ALPHA,
    "run_history":     STAGE_ALPHA,
    "report":          STAGE_ALPHA,
    # -- beta: further along, in regular use, still not signed off
    "make_masks":      STAGE_BETA,
    "analyze_plaques": STAGE_BETA,
    "replication":     STAGE_BETA,
    "umap":            STAGE_BETA,
}

# The built-ins go through the same door as everything else. Registering
# 34 rows one at a time on every import is what keeps `register_app`
# honest: an ordering or validation mistake in it shows up here, at
# import, rather than the first time somebody adds the 35th app.
for _row in _BUILTIN_APPS:
    register_app(*_row)
del _row


# ---------------------------------------------------------------------------
# Apps that lived in their own module
# ---------------------------------------------------------------------------
# Nothing registers here any more, and the section is kept for the note.
# Two pipeline modules were registered from this file rather than from
# themselves, because neither is a Qt module: `spacr.illumination` and
# `spacr.sequencing_qc` are imported into worker processes and into
# `spacr-run`, and neither may grow an import of PySide6 to call
# `register_app` at its own import. Both have since folded into the screen
# that runs them, and a folded module has no row.
#
# ILLUMINATION IS A BUTTON ON MEASURE. Flat-field correction is a property
# of the measure run it changes rather than a run of its own:
# `measure_crop` calls `spacr.illumination.prepare_illumination_correction`
# itself, and the nine `illumination_*` keys are a settings category on
# Measure's own panel. The one thing that panel cannot express is
# estimating and QCing the field WITHOUT measuring the plate -- an hour of
# QC figures before a day of measuring -- so the module keeps its own
# settings form and Run button and opens as a page beside the measure
# settings (`spacr.qt.screens.measure`). `spacr-run illumination` never
# went through this row and is untouched.
#
# BARCODE QC IS A BUTTON ON MAP BARCODES. A mapping run is judged by
# reads per well, starved wells, unmapped reads, collisions, position
# effects and the abundance threshold they imply, so the question "did
# this run work" belongs on the screen that produced the run. It folds
# onto the Map Barcodes masthead and opens as a page beside the mapping
# settings; `spacr-run barcode_qc` and the automatic call from the end of
# the sequencing pipeline never went through this row and are untouched.
#
# EVERYTHING `register_app` FANS OUT DIES WITH THE ROW, so each answer has
# a home that outlives a tile: the entry point in
# `spacr.qt.bridge.resolve_pipeline_entry`, the defaults module in
# `settings_model._FOLDED_DEFAULTS_MODULES`, the API link in
# `settings_model._APP_API_MODULE`, the screen title and intro in
# `app_screen.APP_TITLES` and `APP_INTROS`, and the name, sentence and
# maturity colour the fold button carries in
# `spacr.qt.screens.map_barcodes.FOLD_FALLBACK`.


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
    # THE SIX THAT USED TO ARRIVE BY ACCIDENT. Each of these registers at its
    # own import, and each was reached only because some other screen in this
    # table happened to import it -- Data Manager because Run Compare reads
    # projects, Lineage because the Layer Viewer registers its companions. So
    # the row existed exactly when the import chain that produced it did, and
    # the moment a screen stopped being imported at launch its tile vanished
    # with it. Named here, they are registered because somebody asked for
    # them; the order is the order they used to arrive in, so the tiles keep
    # the positions users know.
    ("spacr.qt.screens.data_manager", "register"),
    ("spacr.qt.screens.pipeline_graph", "register"),
    ("spacr.qt.screens.profiler", "register"),
    ("spacr.qt.screens.qc_dashboard", "register"),
    ("spacr.qt.screens.lineage", "register"),
    ("spacr.qt.screens.experiment_design", "register"),
    ("spacr.qt.layer_viewer", "register_layer_viewer_app"),
    ("spacr.qt.screens.graph_builder", "register"),
    # The three that arrived just after the seam landed and sat finished,
    # tested and unreachable for the same reason the first four did.
    #
    # Power is the first app of the Design section, so this row is also
    # what makes that tab appear — the section has been declared, noted
    # and empty since the sections were named.
    ("spacr.qt.screens.power", "register"),
    # Run Compare registers at its own import and is named in
    # ``spacr.qt.SELF_REGISTERING_MODULES`` too, which only runs at
    # ``run()``. That made the row appear at launch and not under
    # ``import spacr.qt.app``, i.e. exactly the sometimes-there row the
    # note above is about. Both calls are idempotent.
    ("spacr.qt.screens.run_compare", "register"),
    # Tabulate: finished, tested, and defining a register() that nothing
    # called. It was held back when Explore was at the MAX_APPS_PER_SECTION
    # ceiling of 13; it is at 8 now, so the reason has expired. Found by the
    # README pass, which declined to advertise a screen with no tile -- which
    # is the right instinct and also how a feature stays invisible for a
    # fortnight.
    #
    # PCA stood beside it here until it was folded onto Image UMAP: it is a
    # button on that masthead now, opened already pointed at the same
    # measurements database, so there is no row for this table to put in the
    # registry.
    ("spacr.qt.screens.tabulate", "register"),
    ("spacr.qt.screens.investigate_hit", "register"),
)

for _module_name, _func_name in _SELF_REGISTERING_APPS:
    try:
        # THE ROW WITHOUT THE SCREEN. Every module named above declares its
        # row in `app_catalog`, so the registry can be filled in from strings
        # and the screen's own code — pandas, scipy, sklearn behind it — is
        # left unimported until somebody opens the app. `register_declared`
        # returns None for a module that declares nothing, and that module is
        # imported the old way.
        if _declared_for(_module_name) is not None:
            _register_declared(_module_name)
        else:
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
    from spacr.plugins import plugin_apps as _plugin_apps
    from spacr.plugins import record_diagnostic
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


#: Apps that are REGISTERED but get no tile and no sidebar row.
#:
#: A tile says "start here". These are not things a user sets out to do:
#: they are things reached for WHILE doing something else, and a tile for
#: each put them at the same level as Mask and Regression. Instruction 318
#: has the maintainer's list and which of the two doors each one gets --
#: a button inside the module it belongs to, or an entry in Help.
#:
#: THE KEY IS NOT REMOVED, and that distinction is the whole of why this
#: is a set rather than a deletion. `bridge.resolve_pipeline_entry`,
#: `cli.INTERACTIVE_ONLY`, `validate.APP_FUNCTIONS`, `dnd_handlers`,
#: `settings_model.resolve_default_settings` and every saved user state
#: key off these strings. `spacr run_history` from a shell still runs, a
#: saved session that had one of these open still restores it, and a
#: pipeline that names one still resolves it. What changed is what Home
#: OFFERS, not what exists.
TILELESS_APPS = frozenset({
    # Reached from Help -- a user looking something up.
    "feature_dict",
    "run_history",
    "pipeline_graph",
    "project_browser",
    # Reached from a button in the module they belong to.
    "investigate_hit",     # Regression, new tab
    "profiler",            # Regression, new tab
    "train_compare",       # Classify
    "feature_explorer",    # Classify
    "plate_view",          # Graph Builder
    "trellis",             # Graph Builder -- "small multiples"
    "lineage",             # Database Browser
    "tabulate",            # Database Browser
    "layer_viewer",        # QC
    "control_chart",       # QC
    "outliers",            # QC
    # Folded into Import: one module for getting data in, three ways.
    "convert",             # Import -- Format Converter
    "external_masks",      # Import -- External Masks
    # Help menu entries rather than tiles. None of the six is a place to
    # START: each one inspects or administers work that already exists,
    # which is what a menu is for and what a tile is not.
    "report",
    "data_manager",
    "db_browser",
    "queue",
    "batch",
    "distributed_jobs",
})


#: The four tileless apps that get a Help entry, with the label and the
#: status tip each one carries there.
#:
#: SEPARATE FROM `TILELESS_APPS` because they answer different questions:
#: that set says "no tile", this table says "and here is the door
#: instead". An app in the set and not in this table has its door
#: somewhere else -- a button in the module it belongs to.
#: NOT `feature_dict`. It already has a Help entry of its own --
#: `widgets/feature_dictionary.py` installs "Feature Dictionary…" -- which
#: is exactly what the maintainer meant by "it is in the help menue which
#: is enough". Only its TILE was asked for. A second entry here would put
#: the same screen in the same menu twice.
_HELP_MODULES: Tuple[Tuple[str, str, str], ...] = (
    ("run_history", "Run history",
     "Search every recorded job -- its settings, hashed inputs and "
     "outputs, warnings, failures, versions and seeds."),
    ("pipeline_graph", "Pipeline graph",
     "How the modules feed one another, and what each one needs before "
     "it can run."),
    ("project_browser", "Project browser",
     "Every spaCR project this machine knows about, and what is in it."),
    ("db_browser", "Database browser",
     "Browse, filter and export tables from measurements.db."),
    ("report", "Report",
     "Generate a shareable HTML or PDF report of QC results, figures, "
     "statistics, settings and software versions."),
    ("data_manager", "Data manager",
     "Inspect project disk usage and remove derived data while keeping "
     "the source images."),
    ("queue", "Plate queue",
     "Run the same processing pipeline across several plates."),
    ("batch", "Batch runner",
     "Queue modules, plates and settings for unattended sequential "
     "execution."),
    ("distributed_jobs", "Distributed jobs",
     "Submit and monitor spaCR runs on SSH workstations, Slurm, or "
     "cloud and HPC commands."),
)


def app_is_visible(key: str) -> bool:
    """Whether ``key`` should appear in module navigation.

    Preferences are imported lazily so this registry remains safe for
    packaging and headless callers. If preferences cannot be read, preserve
    the historical all-modules-visible behaviour.

    NOT A TILE CHECK. This answers "is the user allowed to reach this
    module", which is the maturity preference and nothing else. A folded
    module is still reachable -- from its host's button, from Help, and
    from the command palette -- so it is still VISIBLE; it just has no
    tile. Use :func:`tiled_apps` for the tile question.

    This function did briefly answer both, and the command palette is
    what caught it: filtering :data:`TILELESS_APPS` out here took nine
    modules out of Ctrl+K, so the folds removed a door instead of moving
    one. Two questions, two functions.
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


#: The tile order, per section, exactly as asked for on 2026-08-31.
#:
#: WRITTEN DOWN BECAUSE REGISTRATION ORDER CANNOT SAY IT. A tile's
#: position used to be the order its module happened to register in, and
#: that order is split across three mechanisms -- the literal rows in
#: this file, `_SELF_REGISTERING_APPS`, and `spacr.qt`'s own module list
#: -- so "Make Masks before Align & Stitch" was not something anyone
#: could state, only something that happened to be true or not. The user
#: wrote out the tiles they wanted in the order they wanted them; this is
#: that list.
#:
#: A key absent from a section's tuple keeps registry order and sorts
#: after every listed one, so registering a new app is still a one-line
#: change that shows up immediately -- at the end of its band, where a
#: new thing belongs until somebody decides where it goes.
SECTION_TILE_ORDER: Dict[str, Tuple[str, ...]] = {
    SECTION_CORE: ("mask", "measure", "annotate", "classify_merged",
                   "map_barcodes", "regression"),
    SECTION_DATA: ("foreign", "run_compare", "experiment_design", "power",
                   "dose_response", "qc_dashboard"),
    SECTION_TOOLS: ("make_masks", "align", "umap", "gate_editor",
                    "graph_builder"),
    SECTION_ASSAYS: ("analyze_plaques", "recruitment", "invasion",
                     "replication"),
}


def tile_sort_key(row: Tuple[str, str, str, str]) -> Tuple[int, int]:
    """Sort key placing ``row`` where :data:`SECTION_TILE_ORDER` says.

    :param row: an ``APPS`` row.
    :returns: ``(section index, position in that section)``. An unlisted
        key sorts after every listed one, keeping registry order among
        the unlisted by virtue of Python's stable sort.
    """
    order = SECTION_TILE_ORDER.get(row[3], ())
    try:
        within = order.index(row[0])
    except ValueError:
        within = len(order)
    try:
        section = SECTION_ORDER.index(row[3])
    except ValueError:
        section = len(SECTION_ORDER)
    return (section, within)


def tiled_apps(
    apps: Optional[List[Tuple[str, str, str, str]]] = None,
) -> List[Tuple[str, str, str, str]]:
    """The ``APPS`` rows that get a TILE, in registry order.

    ``APPS`` is what EXISTS; this is what Home DRAWS. The two stopped
    being the same thing when modules began folding into hosts -- a
    folded module keeps its registry row (it still has a screen, an
    icon, a section and a key to navigate to) and loses only its tile,
    because it is now reached from a button on its host or from the Help
    menu.

    Anything asking "what does the user see on Home", "how many tiles
    are in this band", or "does this tile's label fit" wants this.
    Anything asking "can this key be navigated to", "does this app have
    artwork", or "does every app have a screen" wants ``APPS`` -- a
    folded module must still answer yes to all three.
    """
    source = APPS if apps is None else apps
    return sorted((row for row in source if row[0] not in TILELESS_APPS),
                  key=tile_sort_key)


def section_members(
    section: str,
    apps: Optional[List[Tuple[str, str, str, str]]] = None,
) -> List[Tuple[str, str, str, str]]:
    """The ``APPS`` rows a category's tab shows, in registry order.

    TILELESS APPS ARE NOT MEMBERS. This is what the tab DRAWS and what
    its label COUNTS, so a module with no tile must be out of both --
    otherwise the label reads "Results & QC (7)" over three tiles, which
    is a count of something the user cannot see.

    An explicit ``apps`` list is filtered too. A caller passing its own
    rows is asking "which of these belong to this section", and a
    tileless one does not belong to any tab whichever list it arrives in.
    """
    return [row for row in tiled_apps(apps) if row[3] == section]


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
    # BOTH filters, and they are different questions. `visible_apps` drops
    # what the maturity preference hides; `tiled_apps` drops what has been
    # folded into a host and reached by a button instead. Home is the one
    # surface that wants both -- the command palette and the spaCR menu
    # want only the first, which is why the tile filter does not live in
    # `app_is_visible`.
    apps = tiled_apps(visible_apps())
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

        # COLLAPSED, EXCEPT CORE. The dock listed every section open, which
        # is fifty-six rows and a column taller than the screen -- so the
        # sections were there and did no sorting. Core starts open because
        # it is the pipeline and the reason the dock is on screen; the rest
        # open when their header is clicked.
        self._section_items: dict[str, list] = {}
        self._open_sections: set = {SECTION_CORE}

        current_section = None
        for key, name, desc, section in APPS:
            if section != current_section:
                header = QLabel(section)
                header.setObjectName("SidebarSection")
                header.setCursor(Qt.CursorShape.PointingHandCursor)
                header.setProperty("sectionName", section)
                # A LABEL, NOT A BUTTON: it is already styled as the dock's
                # heading, and a button here would have to be un-styled back
                # into one. The click comes through the event filter.
                header.installEventFilter(self)
                layout.addWidget(header)
                self._section_headers[section] = header
                self._section_items.setdefault(section, [])
                current_section = section
            btn = self._make_item(name, desc, key)
            self._section_items.setdefault(section, []).append(btn)
            icon = _icon_for_app(key)
            if icon is not None:
                btn.setIcon(icon)
                btn.setIconSize(QSize(16, 16))
            btn.clicked.connect(lambda checked=False, k=key: self.nav_selected.emit(k))
            layout.addWidget(btn)

        layout.addStretch(1)
        self._apply_section_state()
        self.refresh_visibility()

    def eventFilter(self, watched, event):      # noqa: N802 - Qt naming
        """Toggle a section when its header is clicked, and light it on hover."""
        section = watched.property("sectionName") if watched else None
        if not section:
            return super().eventFilter(watched, event)
        kind = event.type()
        if (kind == QEvent.Type.MouseButtonRelease
                and event.button() == Qt.MouseButton.LeftButton):
            self.toggle_section(str(section))
            return True
        if kind in (QEvent.Type.Enter, QEvent.Type.Leave):
            watched.setProperty("hovered", kind == QEvent.Type.Enter)
            self._restyle(watched)
        return super().eventFilter(watched, event)

    def section_is_open(self, section: str) -> bool:
        """Whether ``section``'s modules are showing."""
        return str(section) in self._open_sections

    def toggle_section(self, section: str) -> bool:
        """Open or close one section. Returns whether it is now open."""
        section = str(section)
        if section in self._open_sections:
            self._open_sections.discard(section)
        else:
            self._open_sections.add(section)
        self._apply_section_state()
        self.refresh_visibility()
        return section in self._open_sections

    def _apply_section_state(self) -> None:
        """Show a section's modules only while it is open, and colour it."""
        for section, buttons in self._section_items.items():
            wanted = self.section_is_open(section)
            for button in buttons:
                button.setProperty("sectionClosed", not wanted)
        for section, header in self._section_headers.items():
            header.setProperty("open", self.section_is_open(section))
            self._restyle(header)

    @staticmethod
    def _restyle(widget) -> None:
        """Re-read the widget's own QSS after a property changed.

        Qt does not repaint on a property change by itself, so a header that
        just opened keeps the closed colour until something else forces it.
        """
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()

    def refresh_visibility(self) -> None:
        """Apply Alpha/Beta filters without rebuilding or reparenting the dock."""
        section_by_key = {key: section for key, _name, _desc, section in APPS}
        visible_sections = set()
        for btn in self._items:
            key = str(btn.property("navKey") or "")
            if key == "__home__":
                btn.setVisible(True)
                continue
            # A CLOSED SECTION HIDES ITS MODULES, and the maturity filter
            # still applies inside an open one -- the two are separate
            # reasons for a row not to be there.
            visible = app_is_visible(key) and not bool(
                btn.property("sectionClosed"))
            btn.setVisible(visible)
            if app_is_visible(key) and key in section_by_key:
                # THE HEADER STAYS whether the section is open or not: it is
                # what you click to open it. It hides only when every module
                # under it is filtered out by maturity.
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


#: Distance from the icon's edge to the mark inside it, as a fraction.
#:
#: ONE NUMBER FOR THE x AND THE SQUARE. They sit side by side and are read
#: as a pair, so they occupy the same box; the x was drawn to a smaller one
#: and looked like a different kind of control.
CHROME_PAD = 0.18

#: What each window-chrome mark turns when the pointer is on it or the
#: button is held down. The COLOUR IS THE GLYPH's, not a background: red
#: is the one that ends the session, and blue is the accent the rest of
#: spaCR uses for a live control.
CHROME_HOVER = {
    "CloseWindow": "#DC3C3C",
    "FullScreenToggle": "#3C82DC",
    "MinimiseWindow": "#3C82DC",
}


class _ChromeButton(QToolButton):
    """A frameless-window button whose MARK changes colour, not its plate.

    QSS can colour a background on ``:hover`` but not the contents of a
    QIcon, and these three marks are painted rather than shipped -- so
    the hover state is a second painting of the same glyph in the hover
    colour, swapped in on enter and on press and swapped back on leave.
    """

    def __init__(self, parent, painter, colour: str):
        super().__init__(parent)
        self._paint_icon = painter
        self._hover_colour = colour
        self.setAutoRaise(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._show(False)

    def _show(self, lit: bool) -> None:
        self.setIcon(self._paint_icon(colour=self._hover_colour)
                     if lit else self._paint_icon())

    def enterEvent(self, event):        # noqa: N802 - Qt naming
        self._show(True)
        super().enterEvent(event)

    def leaveEvent(self, event):        # noqa: N802 - Qt naming
        self._show(self.isDown())
        super().leaveEvent(event)

    def mousePressEvent(self, event):   # noqa: N802 - Qt naming
        self._show(True)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt naming
        super().mouseReleaseEvent(event)
        self._show(self.underMouse())


class MainWindow(QMainWindow):
    """Top-level window: sidebar + stacked screens + status bar.

    :param initial_app: optional app key to navigate to on show; when
        omitted the window opens on the Home startup page.
    """

    def __init__(self, initial_app: Optional[str] = None):
        super().__init__()
        self._closing = False
        # The compositor may map the native window before a child has drawn.
        # Make that first backing store an opaque splash-coloured surface, so
        # it can never expose stale desktop pixels while LoadingScreen queues
        # its first paint.
        from .widgets.loading_screen import splash_role
        startup_palette = self.palette()
        startup_palette.setColor(
            QPalette.ColorRole.Window,
            QColor(splash_role("splash_bg", "#000000")),
        )
        self.setPalette(startup_palette)
        self.setAutoFillBackground(True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setWindowTitle("spaCR")
        self.setMinimumSize(1200, 720)
        # NO TITLE BAR. Asked for on 2026-08-23: "remove the minus and x bar
        # from the spacr window and just have an icon in the top left for
        # true fullscreen". The window is frameless and the menu bar is what
        # you drag it by; Quit keeps its usual shortcut, so nothing about
        # closing depends on a button that is no longer there.
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)

        self._build_menu_bar()
        self._install_fullscreen_button()

        # Central layout: a row that holds an (initially empty) dock slot
        # and the screen stack. By default the app list is a REVEAL over
        # the stack's left edge rather than anything in that slot.
        #
        # The column was 220-320 px of the 1440 a laptop has, on every
        # screen, holding a list most sessions never touch — and it is
        # the reason Home could not fit its five categories plus a state
        # column without scrolling. As a drawer it costs 6 px of trigger
        # strip and is one hover, one click, or Ctrl+Shift+A away.
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
        status.showMessage(tr("Ready"))
        self.setStatusBar(status)

        # The AI Console now lives inside each pipeline app's Console
        # panel (see spacr.qt.widgets.console_panel). No side-dock.

        # Preload the heavy pipeline imports IMMEDIATELY, behind a loading
        # screen that covers the window until they land.
        #
        # They used to start on a 1500 ms timer, which put a 2.1 s freeze on
        # a window that already looked interactive -- measured on a real
        # launch, `spacr.core` alone is 1968 ms and the chain is 3140 ms. The
        # delay also predates the loading screen: it existed because kicking
        # the chain off pre-nav once caused a circular-import race in
        # spacr.core/IPython ("partially initialized module 'IPython'"), and
        # sleeping through it was the cheap fix. Starting after the first
        # screen is built still satisfies that, and this call site is after
        # it.
        #
        # The imports stay on the MAIN thread. A worker races Qt's own GPU
        # init and segfaults -- see the note on _PipelinePreloader. Blocking
        # the loop is acceptable precisely because nothing interactive is on
        # screen while it happens.
        from PySide6.QtCore import QTimer
        self._loading_screen = self._install_loading_screen()
        self._preloader = _PipelinePreloader(
            on_step=self._on_preload_step, on_done=self._on_preload_done)

        # LOADED WHEN CALLED (instruction 282). Preloading is off by default
        # now, and the maintainer's own timing report is why. Measured on a
        # real launch, the preload thread ground for TWENTY SECONDS:
        #
        #   spacr.core         15.6 s     spacr.deep_spacr    9.8 s
        #   torchvision         8.5 s     torch               6.8 s
        #   torch._dynamo       5.9 s     IPython             2.5 s
        #   sympy               2.9 s     torch.distributed.fsdp  1.8 s
        #
        # The torch COMPILER, sympy, DISTRIBUTED TRAINING and IPython, to
        # draw a window. Importing them ahead of first use was supposed to
        # move the cost earlier; what it actually did was spend it while the
        # user was trying to work, which is worse than spending it when they
        # ask for the thing that needs it.
        #
        # Nothing is lost that was not already paid: the first run of a
        # pipeline imports what it needs, once, exactly as it would have.
        from .preferences import get_preload_policy

        if get_preload_policy() != "eager":
            self._preloader = None
            if self._loading_screen is not None:
                try:
                    self._loading_screen.close()
                except Exception:                            # noqa: BLE001
                    pass
                self._loading_screen = None
        elif self._loading_screen is None:
            # Headless, or the screen could not be built: keep the old
            # deferred start so a test process is not made to pay 3.1 s of
            # imports it may not need.
            QTimer.singleShot(1500, self._preloader.start)
        else:
            self._loading_screen.set_total(self._preloader.total())
            QTimer.singleShot(0, self._preloader.start)

        # Keyboard shortcuts — Ctrl+H, Ctrl+1..9, Ctrl+K, F1/?, etc.
        try:
            from . import shortcuts
            shortcuts.install(self)
        except Exception:
            pass

        if initial_app:
            # Through `open_module`, not straight to the key: `spacr-qt
            # timelapse` is in shell histories and scripts, and that key
            # is a switch on Mask Generation now rather than a screen.
            self.open_module(initial_app)
        else:
            # INSTRUCTION 142: come back to where the Force restart left off.
            # After an explicit `initial_app`, because a user who named a
            # module on the command line is asking for that module, and a
            # saved state that overrode it would be spaCR ignoring what it
            # was just told.
            self.resume_after_restart()

        # 178 D, for the window's own tab bars as well as the screens'. A
        # screen built later takes the arrows off itself; this covers what is
        # already here.
        try:
            from .theme import take_the_scroll_arrows_off
            take_the_scroll_arrows_off(self)
        except Exception:                                # noqa: BLE001
            LOG.debug("could not take the tab scroll arrows off",
                      exc_info=True)

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
            from .install_consent import maybe_show_installer_consent
            # Installer privacy choices precede the product tour. The native
            # installers collect them when they have an interactive surface;
            # an unattended package gets the same all-off page here instead.
            # Parent the delayed callback to the window. A static singleShot
            # outlives a window closed during its first 800 ms, then invokes
            # the tour with a deleted C++ object on the next event-loop spin.
            self._tour_timer = QTimer(self)
            self._tour_timer.setSingleShot(True)
            self._tour_timer.timeout.connect(
                lambda: maybe_show_tour(self))
            self._consent_timer = QTimer(self)
            self._consent_timer.setSingleShot(True)

            def _finish_installer_onboarding():
                maybe_show_installer_consent(self)
                # Start after the modal flow closes, so the tour never opens
                # behind the consent/provider dialogs' nested event loops.
                self._tour_timer.start(500)

            self._consent_timer.timeout.connect(_finish_installer_onboarding)
            self._consent_timer.start(250)
        except Exception:
            pass

    def _install_loading_screen(self):
        """Cover the window with a loading screen when a display is active.

        Return ``None`` for the offscreen platform or when the loading screen
        cannot be created.
        """
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None and app.platformName() == "offscreen":
                return None
            from .widgets.loading_screen import LoadingScreen
            screen = LoadingScreen(parent=self)
            screen.setGeometry(self.rect())
            screen.raise_()
            screen.show()
            return screen
        except Exception:
            # A launch must never fail for want of a splash.
            LOG.debug("could not install the loading screen", exc_info=True)
            return None

    def _on_preload_step(self, done: int, total: int) -> None:
        """Advance the loading screen as each pipeline module lands."""
        screen = getattr(self, "_loading_screen", None)
        if screen is None:
            return
        try:
            screen.set_total(total)
            screen.advance(done)
            # The imports block the loop, so without an explicit repaint the
            # screen would jump from empty to full at the end and show no
            # progress at all.
            screen.repaint()
        except RuntimeError:
            # Deleted underneath us during teardown.
            self._loading_screen = None

    def _on_preload_done(self) -> None:
        """Take the loading screen down and hand the window over."""
        screen = getattr(self, "_loading_screen", None)
        self._loading_screen = None
        if screen is None:
            return
        try:
            screen.hide()
            screen.deleteLater()
        except RuntimeError:
            pass

    def resizeEvent(self, event):
        """Keep the loading screen covering the whole window, and erase the
        menu bar's old band.

        EVERY RESIZE, not only a state change. Dragging an edge moves the
        corner buttons without changing the window state, so `changeEvent`
        never fires -- and the marks are redrawn at their new x over the
        old ones. That is the "sometimes" in the report: it is not
        intermittent, it is every resize that is not a fullscreen toggle.
        """
        super().resizeEvent(event)
        screen = getattr(self, "_loading_screen", None)
        if screen is not None:
            try:
                screen.setGeometry(self.rect())
            except RuntimeError:
                self._loading_screen = None
        self._relay_the_menu_bar()

    def _resolve_version(self) -> str:
        """Return the installed spacr version string, or ``"dev"`` on failure."""
        try:
            import spacr
            return getattr(spacr, "__version__", "") or "dev"
        except Exception:
            return "dev"

    # -- window chrome ----------------------------------------------------
    def _install_fullscreen_button(self):
        """One icon, top left, that toggles TRUE fullscreen.

        The frameless window has no minimise or close button, which is the
        point: the two things the old bar offered were a button that hid
        the application and a button that quit it, and neither is what a
        user reaches for mid-analysis. Fullscreen is.

        Placed as the menu bar's left corner widget, so it sits before the
        first menu rather than beside it, and it is what the top-left
        corner of the window holds.
        """
        # TOP RIGHT, minimise then full screen -- the order a title bar
        # puts them in. Closing is not here: Quit is in the spaCR menu with
        # its usual shortcut, and a stray click on an x mid-analysis costs
        # more than reaching for the menu does.
        from PySide6.QtWidgets import QHBoxLayout, QWidget

        corner = QWidget(self)
        corner.setObjectName("WindowChrome")
        # A plain QWidget paints its own Window palette role.  MainWindow's
        # first-frame palette is deliberately black, so leaving this corner
        # implicit produced one black rectangle behind the three otherwise
        # transparent marks.  Paint no surface here: the menu bar is the
        # title bar and must remain visible through the whole corner widget.
        corner.setAutoFillBackground(False)
        corner.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row = QHBoxLayout(corner)
        row.setContentsMargins(0, 0, 6, 0)
        row.setSpacing(2)

        minimise = _ChromeButton(corner, self._minimise_icon,
                                 CHROME_HOVER["MinimiseWindow"])
        minimise.setObjectName("MinimiseWindow")
        minimise.setToolTip("Minimise")
        minimise.clicked.connect(self.showMinimized)
        row.addWidget(minimise)
        self._minimise_button = minimise

        button = _ChromeButton(corner, self._fullscreen_icon,
                               CHROME_HOVER["FullScreenToggle"])
        button.setObjectName("FullScreenToggle")
        button.setToolTip("Full screen (F11). The window has no title bar; "
                          "drag the menu bar to move it.")
        button.clicked.connect(self.toggle_fullscreen)
        row.addWidget(button)
        self._fullscreen_button = button

        close = _ChromeButton(corner, self._close_icon,
                              CHROME_HOVER["CloseWindow"])
        close.setObjectName("CloseWindow")
        close.setToolTip("Quit spaCR")
        # THE SAME THING QUIT DOES. Not `close()` on the window -- Quit is
        # what every other exit path goes through, and two ways of leaving
        # that differ is how a session ends without saving something.
        close.clicked.connect(self.close)
        row.addWidget(close)
        self._close_button = close

        # THE MARK CHANGES COLOUR, NOT THE PLATE BEHIND IT. The colour is
        # painted into the glyph by `_ChromeButton` (see CHROME_HOVER):
        # red on the x, blue on the square and on the minus. A filled
        # rounded plate behind a 10 px mark reads as a button growing a
        # background rather than as the mark itself lighting up, which is
        # what was asked for.
        # THE BAR'S OWN COLOUR, NOT `transparent`. Same defect as
        # `QMenuBar::item` in theme.py, and reported in the same breath:
        # "there are black boxes behind the minimize, fullscreen and close
        # icons... the black boxes appear only after hovering".
        #
        # `transparent` means paint nothing, and what is behind this
        # corner is the WINDOW, whose palette Window role is the splash
        # colour -- pure black. On Linux the menu bar's fill covers that;
        # on macOS the hover repaint clears to the window first and the
        # black arrives as a plate behind the mark. Painting the bar's
        # colour is identical wherever transparent already worked.
        #
        # TRANSPARENT, NOT "the same colour as the bar".
        #
        # This used to read menu_bar_background() and paint that, so the
        # corner could not drift from the bar it sits on. It drifted
        # anyway, reported 2026-09-01: "the x square and minus in the top
        # right dont always have the same background as the container".
        #
        # A colour copied once at construction is a snapshot. The bar
        # repaints for a theme change, for a palette change, and on macOS
        # for a translucency the copied value never had -- and every one
        # of those leaves three plates in the old colour. Matching by
        # copying is the bug; matching by showing through cannot drift,
        # because there is nothing to keep in step.
        #
        # Safe here in a way it is NOT for the bar itself: transparent
        # means "paint nothing", and these sit INSIDE the menu bar, which
        # paints its own surface. The bar is a top-level surface and would
        # show the desktop through instead.
        #
        # The hover state is unaffected: it is a repaint of the GLYPH in
        # the hover colour, never a plate behind it.
        corner.setStyleSheet("""
            QWidget#WindowChrome {
                background: transparent;
                border: none;
            }
            QWidget#WindowChrome QToolButton,
            QWidget#WindowChrome QToolButton:hover,
            QWidget#WindowChrome QToolButton:pressed,
            QWidget#WindowChrome QToolButton:checked,
            QWidget#WindowChrome QToolButton:disabled {
                background: transparent;
                border: none;
            }
        """)

        self.menuBar().setCornerWidget(corner, Qt.Corner.TopRightCorner)
        self._window_buttons = corner

        # THE MENU BAR IS THE TITLE BAR NOW. Without this the window cannot
        # be moved at all, which is a worse trade than the bar it replaced.
        self._drag_from = None
        self.menuBar().installEventFilter(self)

        # AND THE FRAME WAS WHERE IT WAS RESIZED. Dropping the frame took
        # the grips with it, so the window could be moved and not resized;
        # the edges do it now, handed to the window manager so the drag
        # behaves like every other window on the desktop.
        try:
            from .widgets.glass import let_the_user_resize

            let_the_user_resize(self)
        except Exception:                                    # noqa: BLE001
            LOG.debug("the window could not be made resizable", exc_info=True)

        # THE ACTION THE WINDOW SUBMENU ALREADY HOLDS. A second QAction with
        # the same F11 shortcut is an ambiguous overload, which Qt resolves
        # by firing neither -- so the same object is registered on the
        # window as well, which widens its context instead of competing
        # with it.
        action = getattr(self, "_act_fullscreen", None)
        if action is None:
            action = QAction("Full screen", self)
            action.setShortcut(QKeySequence("F11"))
            action.triggered.connect(self.toggle_fullscreen)
            self._act_fullscreen = action
        self.addAction(action)

    @staticmethod
    def _close_icon(size: int = 18, colour=None):
        """An x, drawn rather than shipped, and the SIZE OF THE SQUARE.

        It sits beside the full-screen mark, so the two are read as a
        pair; drawn to its own smaller box the x looked like a different
        control that happened to be next to one.
        """
        from PySide6.QtGui import QIcon, QPainter, QPen, QPixmap

        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(colour) if colour else QColor(Qt.GlobalColor.gray))
        pen.setWidthF(1.6)
        painter.setPen(pen)
        # THE SAME PAD THE FULL-SCREEN MARK USES, so the x spans exactly
        # the box the square spans.
        pad = size * CHROME_PAD
        painter.drawLine(pad, pad, size - pad, size - pad)
        painter.drawLine(size - pad, pad, pad, size - pad)
        painter.end()
        return QIcon(pixmap)

    @staticmethod
    def _minimise_icon(size: int = 18, colour=None):
        """A single rule, drawn low, the way a minimise mark is."""
        from PySide6.QtGui import QIcon, QPainter, QPen, QPixmap

        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(colour) if colour else QColor(Qt.GlobalColor.gray))
        pen.setWidthF(1.6)
        painter.setPen(pen)
        pad = size * 0.22
        painter.drawLine(pad, size * 0.66, size - pad, size * 0.66)
        painter.end()
        return QIcon(pixmap)

    @staticmethod
    def _fullscreen_icon(size: int = 18, colour=None):
        """The four-corner expand mark, drawn rather than shipped."""
        from PySide6.QtGui import QIcon, QPainter, QPen, QPixmap

        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(colour) if colour else QColor(Qt.GlobalColor.gray))
        pen.setWidthF(1.6)
        painter.setPen(pen)
        arm, pad = size * 0.30, size * CHROME_PAD
        far = size - pad
        for x, y, dx, dy in ((pad, pad, 1, 1), (far, pad, -1, 1),
                             (pad, far, 1, -1), (far, far, -1, -1)):
            painter.drawLine(x, y, x + arm * dx, y)
            painter.drawLine(x, y, x, y + arm * dy)
        painter.end()
        return QIcon(pixmap)

    def eventFilter(self, watched, event):      # noqa: N802 - Qt naming
        """Handle frameless-window menu-bar drag and double-click gestures.

        Teardown can invalidate the Qt menu bar while events are still queued.
        This filter returns ``False`` instead of allowing that condition to
        escape through PySide's callback boundary.

        :param watched: Qt object receiving the event.
        :param event: Qt event delivered to the main window's filter.
        :returns: ``True`` when a menu-bar mouse gesture is consumed;
            otherwise ``False``.
        """
        try:
            bar = self.menuBar() if not self.isFullScreen() else None
        except Exception:                                    # noqa: BLE001
            LOG.debug("the menu bar could not be asked for during an event",
                      exc_info=True)
            return False
        if bar is not None and watched is bar:
            kind = event.type()
            if (kind == QEvent.Type.MouseButtonDblClick
                    and event.button() == Qt.MouseButton.LeftButton):
                self.showNormal() if self.isMaximized() else self.showMaximized()
                return True
            if (kind == QEvent.Type.MouseButtonPress
                    and event.button() == Qt.MouseButton.LeftButton
                    and bar.actionAt(event.position().toPoint()) is None):
                # ONLY ON EMPTY BAR. A press on a menu opens the menu.
                self._drag_from = (event.globalPosition().toPoint()
                                   - self.frameGeometry().topLeft())
            elif kind == QEvent.Type.MouseMove and self._drag_from is not None:
                self.move(event.globalPosition().toPoint() - self._drag_from)
            elif kind == QEvent.Type.MouseButtonRelease:
                self._drag_from = None
        return super().eventFilter(watched, event)

    def toggle_fullscreen(self, *_args) -> bool:
        """Enter or leave true fullscreen. Returns whether it is now full."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
        return self.isFullScreen()

    def changeEvent(self, event):               # noqa: N802 - Qt naming
        """Re-lay the menu bar when the window state changes.

        A menu opens where the BAR SAYS its action is. Going fullscreen
        resizes the bar and its corner widget in one step, and a menu
        opened before the layout has caught up is placed against the
        previous action rectangle -- which is how pressing spaCR drops a
        menu under Help.
        """
        super().changeEvent(event)
        if event.type() != QEvent.Type.WindowStateChange:
            return
        self._relay_the_menu_bar()
        # AND AGAIN ONCE THE NEW SIZE HAS ARRIVED. `changeEvent` is
        # delivered when the STATE changes, which is before the compositor
        # has resized the window -- so a re-lay done only here measures the
        # old geometry and the menu still opens against the previous action
        # rectangle. The zero-timer runs after the resize has been
        # delivered and the layout has settled.
        try:
            from PySide6.QtCore import QTimer

            QTimer.singleShot(0, self._relay_the_menu_bar)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not schedule the menu-bar re-lay", exc_info=True)

    def _relay_the_menu_bar(self) -> None:
        """Re-measure the menu bar and erase what it used to cover.

        TWO JOBS, AND THE SECOND IS THE ONE THAT WAS MISSING.

        Re-measuring puts the actions and the corner buttons where the new
        window size says they belong, so a menu opens under the name that
        was pressed.

        Repainting the bar's band ON THE WINDOW is what removes the ghost.
        The bar and the corner widget both have `setAutoFillBackground`
        FALSE -- deliberately, because the menu bar is the title bar and
        the backdrop has to show through it -- so neither erases the pixels
        it vacates. When the bar re-lays, the marks and the menu names are
        redrawn at their new positions over the old ones, which is exactly
        the reported "slightly misaligned duplicate minus, square, cross"
        and the second copy of "Help".

        `bar.update()` alone cannot fix it: the bar repaints ITSELF, and
        the stale pixels are on the window underneath. The band is taken a
        little taller than the bar so a corner widget that grew downwards
        is covered too.

        Never raises: a window whose bar has been torn down still has to
        finish changing state.
        """
        try:
            bar = self.menuBar()
        except Exception:                                    # noqa: BLE001
            LOG.debug("the menu bar could not be asked for", exc_info=True)
            return
        if bar is None:
            return
        try:
            corner = bar.cornerWidget(Qt.Corner.TopRightCorner)
            if corner is not None:
                corner.adjustSize()
            bar.updateGeometry()
            if bar.layout() is not None:
                bar.layout().activate()
            bar.update()
            band = max(bar.height(), bar.sizeHint().height())
            if corner is not None:
                band = max(band, corner.height())
            self.update(0, 0, self.width(), band + 2)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not re-lay the menu bar", exc_info=True)

    # -- menu -------------------------------------------------------------
    def _build_menu_bar(self):
        mb = self.menuBar()
        # NOT THE NATIVE macOS MENU BAR. Qt defaults this to True on darwin,
        # which moves the whole bar up into the system strip -- and that one
        # default is the cause of THREE separate macOS bugs at once:
        #
        #   1. A native menu bar DRAWS NO CORNER WIDGET. The minimise, full
        #      screen and close marks live in this bar's top-right corner
        #      (see `_install_fullscreen_button`), so on macOS they simply
        #      were not there. The window is frameless on every platform, so
        #      that left a Mac with no window buttons at all.
        #   2. It splits the spaCR menu in two. macOS hoists Preferences,
        #      About and Quit into the application menu -- titled "Python"
        #      for an unbundled launch -- leaving a second, half-empty
        #      "spaCR" menu beside it.
        #   3. Nothing is left in the window to DRAG. The bar is this
        #      window's title bar; in the system strip it cannot move it.
        #
        # The comment that used to sit below this said the relocation "cannot
        # be overridden". It can -- this is how -- and turning it off gives
        # macOS the same one-menu, three-button, draggable bar as Linux.
        if sys.platform == "darwin":
            mb.setNativeMenuBar(False)
        # DRAGGABLE. The window is frameless, so without this it cannot be
        # moved: there is no title bar for the compositor to offer.
        self._menu_drag = _DragsTheWindowByTheMenuBar(self)
        mb.installEventFilter(self._menu_drag)

        app_menu = mb.addMenu("&spaCR")

        # Preferences and Quit FIRST, as asked. This ordering is now what
        # EVERY platform sees: the native menu bar is off on macOS (above),
        # so Qt no longer hoists Preferences and Quit into a separate
        # application menu and the order written here is the order shown.
        act_home = QAction("Home", self)
        act_home.setShortcut(QKeySequence("Ctrl+H"))
        act_home.triggered.connect(lambda: self._on_nav_selected("__home__"))
        app_menu.addAction(act_home)
        act_prefs = QAction("Preferences…", self)
        act_prefs.setShortcut(QKeySequence("Ctrl+,"))
        act_prefs.triggered.connect(self._open_preferences)
        app_menu.addAction(act_prefs)
        act_quit = QAction("Quit", self)
        act_quit.setShortcut(QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        app_menu.addAction(act_quit)
        # MINIMISE AND MAXIMISE GO IN ABOVE QUIT -- but not from here. The
        # Window submenu builds those two actions further down this method,
        # and the SAME objects are inserted here once it has
        # (`_lift_the_window_actions_into_the_spacr_menu`). Two QActions for
        # one behaviour is what this file already warns about for F11: Qt
        # resolves a duplicated shortcut by firing neither, and even without
        # a shortcut a second object is a second enabled state to keep in
        # step. Remembered rather than searched for, because Quit's label is
        # translated and matching on it would break in every other language.
        self._act_quit = act_quit
        app_menu.addSeparator()

        # ONE SUBMENU PER CATEGORY. Fifty-six modules in one flat list is a
        # column taller than most screens, and reading it means reading all
        # of it -- "the modules should be in module category dropdowns to
        # make it more digestable". The categories are the ones Home and the
        # dock already use, in the same order, so the three surfaces agree.
        self._app_actions: dict[str, QAction] = {}
        self._section_menus: dict[str, QMenu] = {}
        for section in SECTION_ORDER:
            members = [row for row in APPS if row[3] == section]
            if not members:
                continue
            submenu = QMenu(section, self)
            app_menu.addMenu(submenu)
            self._section_menus[section] = submenu
            for key, name, desc, _section in members:
                act = QAction(name, self)
                act.setStatusTip(desc)
                # Translate the name and reviewed scientific summary as
                # separate semantic fields; word-by-word translation of the
                # combined text can produce misleading mixed-language help.
                act.setProperty("moduleAppKey", key)
                act.setProperty("moduleNameSource", name)
                act.setProperty("moduleSummarySource", desc)
                act.triggered.connect(
                    lambda checked=False, k=key: self._on_nav_selected(k))
                submenu.addAction(act)
                self._app_actions[key] = act
        self._refresh_app_action_visibility()

        # "All apps" is NOT in the menu: a menu entry whose purpose is not
        # obvious from its name costs attention every time it is read, and
        # this one names a drawer most users never knew existed.
        #
        # The action itself stays, registered on the window rather than on
        # the menu. Ctrl+Shift+A is the keyboard route into the edge reveal --
        # a panel you can otherwise summon only by hovering a 6 px strip is
        # a panel a keyboard user does not have -- and deleting the action
        # would take the shortcut with it.
        act_all = QAction("All apps", self)
        # MOVED OFF Ctrl+B, which was asked for as the blank-background key
        # -- twice, and it was quietly given to Ctrl+Shift+B because this
        # already held it. A shortcut somebody asks for by name and gets
        # something else from is worse than an unfamiliar one.
        act_all.setShortcut(QKeySequence("Ctrl+Shift+A"))
        act_all.setStatusTip(
            "Show the full app list. Also revealed by moving the pointer "
            "to the left edge of the window.")
        act_all.triggered.connect(self.toggle_app_drawer)
        self.addAction(act_all)

        # THE BACKDROP OFF AND ON. Registered on the window like Ctrl+Shift+A
        # above, so it works wherever focus is. It STOPS the animation
        # rather than hiding it: a hidden backdrop that kept rendering would
        # be the worst of both, spending the cores and showing nothing.
        act_backdrop = QAction("Animated background", self)
        act_backdrop.setObjectName("ToggleBackdrop")
        act_backdrop.setCheckable(True)
        act_backdrop.setChecked(True)
        act_backdrop.setShortcut(QKeySequence("Ctrl+T"))
        act_backdrop.setStatusTip(
            "Stop the moving background — useful while looking at images. "
            "It gives its threads back rather than only hiding.")
        act_backdrop.toggled.connect(self._set_backdrop_animating)
        self.addAction(act_backdrop)
        self._act_backdrop = act_backdrop

        # RESTART THE BACKDROP. Listed here rather than left as an
        # undocumented key: a shortcut nobody can find is one nobody uses,
        # and this menu is where the other two live.
        act_restart = QAction("Restart the background", self)
        act_restart.setObjectName("RestartBackdrop")
        act_restart.setShortcut(QKeySequence("Ctrl+R"))
        act_restart.setStatusTip(
            "Send the animation back to the beginning. The spaceout deep "
            "zoom restarts its descent; Up and Down change how fast it "
            "descends, the wheel does the same, and dragging with the "
            "mouse steers it.")
        act_restart.triggered.connect(self._restart_the_backdrop)
        self.addAction(act_restart)
        self._act_restart_backdrop = act_restart

        # THE BACKDROP ON ITS OWN, full screen. Ctrl+Shift+F because Ctrl+F
        # is search everywhere and F11 is the window's own full screen --
        # this is a third thing: the animation with nothing else on top.
        act_saver = QAction("Full-screen background", self)
        act_saver.setObjectName("ShowScreensaver")
        act_saver.setShortcut(QKeySequence("Ctrl+Shift+F"))
        act_saver.setStatusTip(
            "Show only the animated background, full screen, like a "
            "screensaver. Any key or click brings the window back.")
        act_saver.triggered.connect(self._show_the_screensaver)
        self.addAction(act_saver)
        self._act_screensaver = act_saver

        # PAUSE AND GO FLAT. Ctrl+T stops the animation and leaves the last
        # frame up, which is still a picture behind the work; this one also
        # paints the ground flat, which is what "I am looking at images and
        # want nothing behind them" actually asks for.
        #
        # Ctrl+B was explicitly requested for this action. The drawer moved
        # to Ctrl+Shift+A, which keeps both window actions keyboard-reachable.
        act_flat = QAction("Blank the background", self)
        act_flat.setObjectName("BlankBackdrop")
        act_flat.setCheckable(True)
        act_flat.setShortcut(QKeySequence("Ctrl+B"))
        act_flat.setStatusTip(
            "Pause the background and paint it flat — dark grey in a dark "
            "theme, white in a light one.")
        act_flat.toggled.connect(self._set_backdrop_blank)
        self.addAction(act_flat)
        self._act_flat = act_flat
        #: Kept so :meth:`apply_dock_mode` can grey it out — a Ctrl+Shift+A that
        #: silently does nothing because the dock is hidden is worse than
        #: a menu entry that says so.
        self._act_all_apps = act_all

        # DEMOS LIVES UNDER HELP. A demo is something you reach for when
        # you are learning what a module does, which is what the Help menu
        # is for, and it was taking a top-level slot on a bar that has to
        # stay short. Built here, before Help, and added to it below --
        # the submenu is the same QMenu either way.
        # PARENTED TO THE MENU BAR, not to the window. `first_run.find_menu`
        # -- which the walkthrough and the tutorial scripts both use --
        # reaches a menu through `menuBar().findChildren(QMenu)`, because
        # walking the bar's actions returns QMenu wrappers that go stale on
        # PySide6 6.11. A menu parented elsewhere is invisible to that
        # lookup, so Demos would have become unfindable the moment it
        # stopped being a top-level menu.
        demo_menu = QMenu("&Demos", mb)
        self._demo_menu = demo_menu
        self._demo_actions: dict[str, QAction] = {}
        for app_key, label in DEMO_LABELS.items():
            act = QAction(label, self)
            act.setStatusTip(tr(self.DEMO_STATUS_TIP, app=app_key))
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
        help_menu.addMenu(demo_menu)
        help_menu.addSeparator()
        # THE HOTKEY MAP, FIRST (197). Asked for 2026-08-21: "add hotkey map
        # to help tab".
        #
        # `show_cheat_sheet` has drawn this map for a long time and was
        # reachable from exactly two places -- the `?` key, which you have to
        # know about, and the command palette, which you have to know about.
        # The Help menu is where a user who does NOT already know a shortcut
        # goes to look for one, which is the entire population this screen is
        # for.
        #
        # ABOVE THE WEB LINKS because it is the only entry here that answers
        # without a browser.
        act_keys = QAction("Keyboard shortcuts", self)
        act_keys.setStatusTip(
            "Every key spaCR binds, what it does, and where it works.")
        act_keys.triggered.connect(self._show_shortcuts)
        help_menu.addAction(act_keys)

        # THE SETUP SCREEN, REACHABLE AGAIN. It ran once on the first launch
        # and then never -- and it is the only place several of these
        # settings are explained rather than merely offered, so a user who
        # dismissed it lost the explanation along with the questions.
        #
        # IN HELP because that is where somebody goes to be told what a
        # choice means. Preferences is where they go when they already know
        # and want to change it; both exist, and they answer different
        # questions.
        act_setup = QAction("Set spaCR up again…", self)
        act_setup.setStatusTip(
            "The first-run questions, with the explanation of each -- "
            "language, theme, how it runs, the assistant, and what may "
            "leave this machine.")
        act_setup.triggered.connect(self._show_setup)
        help_menu.addAction(act_setup)
        help_menu.addSeparator()

        # THE FOUR LOOK-IT-UP MODULES (318). Each opens exactly the screen
        # it opened from its tile; only the door changed. They are here
        # rather than on Home because none of them is a job a user sets out
        # to do -- they are things you consult, which is what this menu is
        # for.
        #
        # `_on_nav_selected` is the same entry point a tile click uses, so
        # there is one path into a module and not two that can drift.
        for key, label, tip in _HELP_MODULES:
            action = QAction(label, self)
            action.setStatusTip(tip)
            action.triggered.connect(
                lambda _checked=False, k=key: self._on_nav_selected(k))
            help_menu.addAction(action)
        help_menu.addSeparator()
        # NO ICON AND NO "(web)". The icon was
        # `SP_MessageBoxInformation`, the platform's blue circled i, which
        # is the glyph a dialog uses to mean "here is a notice" -- next to
        # a menu label it read as a badge rather than as an illustration of
        # anything. That both entries carried the SAME one made it noise
        # twice over. Where the page opens is said in the status tip, which
        # is where a detail belongs; a label is for what the thing is.
        #
        # The catalog in `spacr/qt/i18n.py` keys on the English string, so
        # both keys moved with these labels -- renaming here alone would
        # drop the translation in nine languages.
        act_tutorial = QAction("Tutorial", self)
        act_tutorial.setStatusTip(
            "Open the interactive spaCR lesson library in a browser.")
        act_tutorial.triggered.connect(
            lambda: self._open_url(TUTORIALS_URL))
        help_menu.addAction(act_tutorial)
        act_docs = QAction("Documentation", self)
        act_docs.setStatusTip(
            "Open the spaCR documentation in a browser.")
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

        help_menu.addSeparator()
        help_menu.addMenu(self._build_window_menu(mb))
        self._lift_the_window_actions_into_the_spacr_menu(app_menu)

        # Every menu action gets an EXPLICIT macOS role, and everything that
        # is not genuinely Preferences/Quit/About gets NoRole. Left to Qt,
        # the role is guessed from the action's TEXT, and an action whose
        # text merely contains "settings" or "options" is silently moved out
        # of its menu into the application menu -- which is how
        # `recipes.MENU_ACTION_TEXT` ("Settings recipes…") ended up as the
        # Preferences item of the macOS "python" menu while the real
        # Preferences and Quit vanished from this one. See spacr.qt.menus.
        #
        # Collected from the menu bar rather than listed by hand, so an
        # action added later is covered without anyone remembering to.
        self._act_preferences = act_prefs
        self._act_quit = act_quit
        self._act_about = act_about
        self.pin_all_menu_roles()

    def _lift_the_window_actions_into_the_spacr_menu(self, app_menu) -> None:
        """Put Minimise and Maximise in the spaCR menu, just above Quit.

        THE SAME TWO QActions the Window submenu holds, added to a second
        menu rather than rebuilt. Qt is happy to show one action in two
        places and keeps their label, enabled state and tick in step for
        free; two objects for one behaviour drift, and if either ever
        gains a shortcut Qt resolves the duplicate by firing NEITHER --
        the trap this file already documents for F11.

        Why they belong here at all: the Window submenu hangs off Help,
        which is not where anyone looks for "make this window smaller".
        On macOS it was worse than obscure -- the native menu bar drew no
        corner widget, so the minimise and full screen marks did not
        exist, and the submenu was the only route to either.

        Called after `_build_window_menu`, which is what creates the
        actions. Inserting rather than appending because the spaCR menu
        continues past Quit into the category submenus, so appending would
        land them at the bottom of the whole list.
        """
        minimise = getattr(self, "_act_minimise", None)
        maximise = getattr(self, "_act_maximise", None)
        quit_action = getattr(self, "_act_quit", None)
        if minimise is None or maximise is None or quit_action is None:
            return
        app_menu.insertAction(quit_action, minimise)
        app_menu.insertAction(quit_action, maximise)
        app_menu.insertSeparator(quit_action)

    def _build_window_menu(self, bar):
        """Build Help ▸ Window: the route that does not move, and the frame
        controls for a frame that is not always there.

        TWO REPORTS, ONE MENU.

        On macOS an action carrying ``PreferencesRole`` or ``QuitRole`` is
        MOVED by the platform out of whatever menu it was added to and into
        the application menu. That is correct, and spaCR keeps it -- but it
        means spaCR's own dropdown cannot hold Preferences or Quit on that
        platform, and a user who looked there and found nothing needs a
        second route. These copies carry ``NoRole``, which is what stops
        macOS relocating them as well and repeating the problem one level
        down; :meth:`pin_all_menu_roles` is what puts it there, and the copy
        being a separate ``QAction`` from the real one is what lets it.

        The window controls answer the other half. This window is frameless
        and draws its own minimise, full-screen and close marks into the
        menu bar's corner, and a corner widget is not guaranteed to be
        visible: on macOS it can end up laid out where nothing shows it, and
        a window whose only close control is that mark then has no close
        control at all. A menu entry does not depend on a corner widget
        landing where the platform expects one.

        PARENTED TO THE MENU BAR, like Demos, because ``first_run.find_menu``
        reaches menus through ``menuBar().findChildren(QMenu)`` and a menu
        parented elsewhere is invisible to it.

        :param bar: the menu bar to parent the submenu to.
        :returns: the ``Window`` submenu, not yet added to anything.
        """
        menu = QMenu("Window", bar)
        self._window_menu = menu

        act_min = QAction("Minimise", self)
        act_min.setStatusTip(
            "Send spaCR to the dock or taskbar. Also the first mark in the "
            "menu bar's top-right corner.")
        act_min.triggered.connect(self.showMinimized)
        menu.addAction(act_min)
        self._act_minimise = act_min

        act_max = QAction("Maximise", self)
        act_max.setStatusTip(
            "Fill the screen, or restore the previous size if the window is "
            "already maximised.")
        act_max.triggered.connect(self._toggle_maximised)
        menu.addAction(act_max)
        self._act_maximise = act_max

        # THE SAME ACTION THE WINDOW ITSELF CARRIES, not a second one with
        # the same shortcut. Two distinct QActions bound to F11 on one
        # window is an ambiguous overload and Qt then fires NEITHER, so a
        # menu copy would have cost the key it advertises.
        act_full = QAction("Full screen", self)
        act_full.setShortcut(QKeySequence("F11"))
        act_full.setStatusTip(
            "True fullscreen. This window has no title bar; drag the menu "
            "bar to move it.")
        act_full.triggered.connect(self.toggle_fullscreen)
        self._act_fullscreen = act_full
        menu.addAction(act_full)

        act_close = QAction("Close window", self)
        act_close.setStatusTip("Close the main window. spaCR quits with it.")
        act_close.triggered.connect(self.close)
        menu.addAction(act_close)

        menu.addSeparator()

        # THE TEXT IS COPIED VERBATIM from the two actions in the spaCR
        # menu. `spacr.qt.i18n` keys its catalog on the English string, so a
        # copy worded differently would be a copy that stays English in the
        # other nine languages.
        act_prefs_here = QAction("Preferences…", self)
        act_prefs_here.setStatusTip(
            "The same Preferences the spaCR menu offers. macOS moves that "
            "one into the application menu; this one stays here.")
        act_prefs_here.triggered.connect(self._open_preferences)
        menu.addAction(act_prefs_here)
        self._act_preferences_here = act_prefs_here

        act_quit_here = QAction("Quit", self)
        act_quit_here.setStatusTip(
            "The same Quit the spaCR menu offers. macOS moves that one into "
            "the application menu; this one stays here.")
        act_quit_here.triggered.connect(self.close)
        menu.addAction(act_quit_here)
        self._act_quit_here = act_quit_here

        return menu

    def _toggle_maximised(self, *_args) -> bool:
        """Maximise the window, or restore it. Returns whether it is now full.

        One entry rather than two, because a Maximise that is greyed out and
        a Restore that is greyed out are two dead menu items where the frame
        button they replace is a single control.
        """
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
        return self.isMaximized()

    def pin_all_menu_roles(self) -> None:
        """Give every menu-bar action an explicit macOS role. Idempotent.

        Called at the end of `_build_menu_bar` AND again from
        `spacr.qt.shortcuts.install`, because `recipes`, `walkthrough` and
        `feature_dictionary` all add to Help afterwards. Each of those also
        pins its own -- defence in depth -- but a central re-sweep is what
        makes a module added later safe without its author knowing this
        problem exists.
        """
        from .menus import pin_menu_roles
        pin_menu_roles(self._menu_bar_actions(),
                       preferences=getattr(self, "_act_preferences", None),
                       quit_action=getattr(self, "_act_quit", None),
                       about=getattr(self, "_act_about", None))

    def _menu_bar_actions(self):
        """Every action on the menu bar, including submenu contents.

        Used to pin macOS menu roles. Walks the menus rather than reading a
        list, because the list is what goes stale -- `recipes.py` and
        `feature_dictionary.py` both add actions to Help from outside this
        method, and neither would appear in anything hand-maintained.

        `bar.findChildren(QMenu)`, NOT `menuBar().actions()` + `QAction.menu()`
        -- the same rule `command_palette.py` and `recipes.py` already follow.
        On PySide6 6.11 the QMenu wrapper the latter returns is only valid
        while the QAction wrapper it came off is alive, and walking the bar
        that way returned an EMPTY list here once construction had finished:
        3 top-level actions immediately after `_build_menu_bar`, 0 after
        `__init__`. Measured, not guessed.
        """
        from PySide6.QtWidgets import QMenu

        out, seen = [], set()
        bar = self.menuBar()
        if bar is None:
            return out
        for menu in bar.findChildren(QMenu):
            # The menu's OWN action too -- the one that opens it from the bar
            # (or from a parent menu). It is an action like any other and Qt
            # will happily give "&Options" a role if left to guess.
            for action in list(menu.actions()) + [menu.menuAction()]:
                if action is None or action.isSeparator():
                    continue
                # Qt's role heuristic matches on TEXT, so an action with no
                # text has nothing to match and cannot be relocated. They
                # turn up here because `menuAction()` CREATES one for a menu
                # that has never been attached to a bar -- i.e. this walk can
                # manufacture them. Skipping keeps the sweep and the audit
                # agreeing about what exists.
                if not action.text():
                    continue
                if id(action) in seen:
                    continue
                seen.add(id(action))
                out.append(action)
        return out

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
        # The timelapse demo writes a settings CSV with timelapse=True, and
        # that key has no widget on the Mask form -- the masthead switch is
        # its control. `AppScreen.apply_settings_dict` moves the switch from
        # the dict it applies, so the demo lands on Mask with tracking
        # already on and its categories already showing.
        "timelapse": ("mask",       "generate_timelapse_demo"),
        "map_barcodes": ("map_barcodes", "generate_map_barcodes_demo"),
    }

    #: Status tip shown for every Demos entry. A template with a placeholder
    #: rather than a sentence built by an f-string: interpolating the app key
    #: BEFORE the lookup asks the catalog for a sentence it can never hold,
    #: which is why all six tips stayed English in every language.
    DEMO_STATUS_TIP = ("Generate a synthetic {app} dataset and open it in "
                       "the matching app.")

    def _on_load_demo(self, demo_key: str) -> None:
        """Generate a synthetic demo dataset, save its settings, then
        navigate to the matching app and pre-populate it."""
        from pathlib import Path

        from PySide6.QtWidgets import QFileDialog

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
        from pathlib import Path

        from PySide6.QtWidgets import QFileDialog, QMessageBox

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
        """Open Mask Generation on the downloaded dataset, ready to run.

        IT DOES NOT RUN ANYTHING. Asked for on 2026-08-31: "the user
        should be able to hit import and then live preview or run".

        This used to be a Mask -> Measure -> Annotate chain that started
        each pipeline itself, behind a Continue/Stop prompt per stage.
        Two things were wrong with that. A demo dataset exists to be
        LOOKED at -- the first thing anyone wants is Live Preview on one
        field, to see what the settings do, and the chain went straight
        past that to a full run. And a Continue prompt before work the
        user did not ask to start is a dialog whose safe answer is No,
        which is a strange thing to greet somebody with.

        So: the settings land in the form, the screen opens, and the
        user presses Live Preview or Run. Measure and Annotate are one
        click away on the same screen when the masks exist.
        """
        from pathlib import Path

        from PySide6.QtWidgets import QMessageBox

        dataset_path  = Path(dataset_path)
        settings_path = Path(settings_path)

        # MIGRATED, NOT MERGED. `settings_from_pack` reads the pack
        # against this build's settings and reports what it could not
        # place -- see spacr/qt/settings_pack.py for why reading a CSV
        # straight over the defaults was the bug rather than the shortcut.
        from .settings_pack import settings_from_pack

        def _settings_for(app_key: str) -> dict:
            settings, report = settings_from_pack(
                app_key, settings_path, src=dataset_path)
            if report.dropped or report.renamed or report.malformed:
                LOG.info("settings pack for %s: %s", app_key, report.summary())
            return settings

        # ONE STAGE. Measure and Annotate are reached from Mask
        # Generation once there are masks to measure; opening them now,
        # against a dataset with no masks in it, would open two screens
        # that can only report that there is nothing to do.
        settings = _settings_for("mask")
        self._on_nav_selected("mask")
        widget = self._screens.get("mask")
        if widget is None:
            QMessageBox.warning(
                self, "Demo dataset",
                "The dataset downloaded, but Mask Generation would not "
                "open. Point it at the folder yourself:\n"
                f"{dataset_path}")
            return
        try:
            if hasattr(widget, "apply_settings_dict"):
                widget.apply_settings_dict(settings)
        except Exception as error:                          # noqa: BLE001
            LOG.exception("Could not apply the demo settings")
            QMessageBox.warning(
                self, "Demo settings",
                "The dataset downloaded and Mask Generation is open, but "
                "its settings could not be filled in automatically:\n"
                f"{type(error).__name__}: {error}")
            return
        self.statusBar().showMessage(
            "Demo dataset loaded with its settings. Press Live Preview to "
            "see one field, or Run to process the plate.", 12000)

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

    def _show_setup(self) -> None:
        """Open the setup slides without applying the first-run gate."""
        try:
            from .widgets.setup_slides import SetupSlides

            SetupSlides(self).exec()
        except Exception:
            LOG.debug("could not open the setup screen", exc_info=True)

    def _show_shortcuts(self) -> None:
        """Open the hotkey map. The same screen `?` and the palette open --
        one map, three doors, not three maps."""
        from .shortcuts import show_cheat_sheet

        show_cheat_sheet(self)

    def _show_about(self):
        """Show the About panel: mark, name, version, licence, lab.

        Laid out like a macOS About window — mark, name, version, then the
        small print — because that is a shape people already know how to read.

        The licence is named exactly and links to the canonical text rather
        than being paraphrased, and the sentence beside it says the thing a
        reader actually wants to know -- that they may use it for anything.
        "© Olafsson Lab" alone says nothing about what they may do with the
        software.
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
                mark.setPixmap(scaled_for(pixmap, mark, QSize(96, 96)))
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
        _line(tr("Spatial phenotype analysis of CRISPR&#8209;Cas9 screens"),
              13, muted=True, gap=10)
        _line(f"Version {version}", 12, muted=True)
        _line(build, 11, muted=True, gap=16) if build else col.addSpacing(16)
        # The license NAME is a legal identifier and stays English; the
        # sentence around it does not. Kept as one placeholder so a language
        # that puts the name elsewhere in the clause can move it.
        license_link = (
            '<a href="https://opensource.org/licenses/BSD-3-Clause">'
            'BSD 3-Clause License</a>')
        _line(
            tr("Licensed under the {name}.", name=license_link) + "<br>"
            + tr("Open source. Free for any use, including commercial."),
            11, muted=True, gap=10)
        _line("© Olafsson Lab", 11, muted=True)

        dialog.setFixedWidth(420)
        dialog.exec()

    def _open_log_folder(self):
        """Open the ~/.spacr/logs folder in the OS file browser."""
        import webbrowser

        from .verbose_logger import log_dir
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
        self._refresh_demo_status_tips()

    def _refresh_demo_status_tips(self) -> None:
        """Re-render the Demos status tips in the current language.

        The retranslation pass caches whatever text a status tip already
        holds and looks THAT up, which cannot work for a tip built from a
        template: the cached sentence has the app key baked into it. These
        six are rebuilt from :attr:`DEMO_STATUS_TIP` instead, so a language
        chosen after the window opened reaches them like any other caption.
        """
        for app_key, action in getattr(self, "_demo_actions", {}).items():
            try:
                action.setStatusTip(tr(self.DEMO_STATUS_TIP, app=app_key))
            except RuntimeError:
                # A deleted action during shutdown must not stop the rest.
                pass

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

        self.statusBar().showMessage(tr("Checking for updates…"), 4000)
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
        self.statusBar().showMessage(tr("Upgrading spaCR…"), 4000)
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
        # Closing a parent widget does not deliver a close event to its child
        # widgets. AppScreen.closeEvent owns cleanup that cannot be left to
        # Qt's child-destruction cascade, notably its parentless pyqtgraph
        # menus and background job runners. Ask each owned screen to close
        # while it is still intact, and honour a screen that defers shutdown
        # because one of its workers has not reached a safe boundary.
        from .screens.app_screen import AppScreen
        seen_screens = set()
        for screen in list(getattr(self, "_screens", {}).values()):
            if not isinstance(screen, AppScreen) or id(screen) in seen_screens:
                continue
            seen_screens.add(id(screen))
            try:
                accepted = screen.close()
            except RuntimeError:
                continue          # already deleted -- nothing left to drain
            except Exception:                                # noqa: BLE001
                LOG.exception("Could not close an owned application screen")
                event.ignore()
                self._closing = False
                return
            if accepted is False:
                LOG.warning("Shutdown deferred by an application screen")
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
        # WITH quitOnLastWindowClosed OFF, THIS IS WHAT ENDS THE PROGRAM.
        # Nothing else may: a figure window closing must not take the session
        # with it, and until this closes the application stays up even with
        # no window on screen.
        if event.isAccepted():
            app = QApplication.instance()
            if app is not None:
                app.quit()

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

    def _set_backdrop_blank(self, blank: bool) -> int:
        """Pause every backdrop and paint the ground flat. Ctrl+B.

        :returns: how many backdrops were hidden, so a test can assert a
            number rather than a screenshot.

        Ctrl+T stops the animation and leaves the last frame on screen,
        which is still a picture. This hides it as well, so what is behind
        the work is the theme's own ground -- dark grey in a dark theme and
        white in a light one, taken from the palette rather than hard-coded,
        because a hard-coded grey is wrong in one of the two themes by
        construction.

        The animation is stopped BEFORE hiding: a hidden backdrop still
        rendering would spend the threads and show nothing for them, which
        is the worst of both.
        """
        hidden = 0
        try:
            for child in self.findChildren(QWidget):
                setter = getattr(child, "set_animating", None)
                if not callable(setter):
                    continue
                try:
                    if blank:
                        setter(False)
                        child.hide()
                    else:
                        child.show()
                        setter(True)
                    hidden += 1
                except Exception:                            # noqa: BLE001
                    LOG.debug("a backdrop would not blank", exc_info=True)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not reach the backdrops", exc_info=True)
        # The ground the backdrop was covering is the theme's own window
        # colour, so nothing has to be painted -- uncovering it is enough.
        # Keeping the two toggles agreeing is what stops Ctrl+T from
        # appearing to do nothing while the background is blanked.
        act = getattr(self, "_act_backdrop", None)
        if act is not None:
            try:
                act.setChecked(not blank)
            except Exception:                                # noqa: BLE001
                pass
        return hidden

    def _set_backdrop_animating(self, on: bool) -> int:
        """Start or stop every backdrop in this window. Ctrl+T.

        :returns: how many backdrops answered, so a test can assert a number
            rather than a screenshot.

        Reaches BOTH kinds. The ambient engines and the spaceout fractal are
        different classes with different lifetimes, but both answer to
        `set_animating` -- which is the reason the fractal was given that
        name rather than only `pause`/`resume`.

        Never raises: a decoration that will not stop must not break a menu.
        """
        answered = 0
        try:
            for child in self.findChildren(QWidget):
                setter = getattr(child, "set_animating", None)
                if not callable(setter):
                    continue
                try:
                    setter(bool(on))
                    answered += 1
                except Exception:                            # noqa: BLE001
                    LOG.debug("a backdrop would not toggle", exc_info=True)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not reach the backdrops", exc_info=True)
        return answered

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

    def resume_after_restart(self) -> str:
        """Reopen the module a Force restart saved. Returns its key, or "".

        TAKEN, NOT READ: `restart_state.take` deletes the state as it hands it
        over, so a crash on the way back up cannot leave spaCR reopening the
        same wedged module on every launch afterwards -- which would turn one
        bad afternoon into a permanently broken installation.

        THE SETTINGS ARE APPLIED, THE RUN IS NOT STARTED. 142 C: the runs do
        not come back, only the configuration, and starting one unasked would
        be the opposite of what somebody who just force-restarted wants.
        """
        from ..restart_state import take

        state = take()
        if not isinstance(state, dict):
            return ""
        key = str(state.get("module") or "")
        if not key:
            return ""
        try:
            # The record was written by whatever screen was open, under
            # whatever key it had then; a module folded since is reopened
            # on the host that took it over.
            key = self.open_module(key)
            screen = self._screens.get(key) if hasattr(self, "_screens") else None
            settings = state.get("settings")
            if screen is not None and isinstance(settings, dict) and settings:
                applied = screen.apply_settings_dict(settings)
                LOG.info("restored %s settings after a restart into %s",
                         applied, key)
        except Exception:                                    # noqa: BLE001
            LOG.exception("could not reopen %s after the restart", key)
            return ""
        return key

    def open_module(self, app_key: str) -> str:
        """Navigate to the screen that carries ``app_key``, folded or not.

        A module folded into a host keeps its key everywhere a key is
        saved: a run journal, the force-restart record, ``spacr-qt
        <app>`` in somebody's shell history. That key no longer names a
        screen, and navigating to it anyway BUILDS one -- an orphan page
        with no sidebar row, no tile and no way back to it, which is the
        second front door the fold exists to remove.

        So the key is resolved to the host that took it over, and the
        fold it names is switched on: asking for Timelapse is asking for
        what the tracking switch reveals, not for mask generation with
        the switch off.

        :param app_key: the key that was asked for.
        :returns: the app key actually opened.
        """
        from .chaining import screen_for_module

        wanted = str(app_key)
        target = screen_for_module(wanted)
        self._on_nav_selected(target)
        if target != wanted:
            self._switch_a_fold_on(target, wanted)
        return target

    def _switch_a_fold_on(self, host_key: str, folded_key: str) -> None:
        """Press ``host_key``'s switch for ``folded_key``, if it has one.

        Driven through the button so the strip shows the state the form
        has. A host that carries no switch for the key costs a lookup:
        the fold may be a page rather than a category, and a page opens
        when the user presses it rather than on arrival.
        """
        screen = self._screens.get(host_key)
        if screen is None:
            return
        try:
            from .screens.mask import fold_set

            folds = fold_set(screen)
            strip = getattr(folds, "strip", None) if folds else None
            button = strip.button_for(folded_key) if strip else None
            if button is not None:
                button.setChecked(True)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not switch %s on in %s", folded_key, host_key,
                      exc_info=True)

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

    def _show_the_screensaver(self) -> bool:
        """Open the backdrop full screen, with nothing else on it.

        :returns: whether it opened.

        A WINDOW OF ITS OWN rather than this one made full screen: hiding
        spaCR's widgets means remembering what was visible, what had focus,
        which docks were open and where the splitters were -- and getting
        any of that wrong leaves the layout rearranged by something meant to
        be a screensaver. A separate window has nothing to restore.
        """
        try:
            from .screensaver import show_screensaver

            saver = show_screensaver(self)
        except Exception:                                    # noqa: BLE001
            LOG.exception("could not open the full-screen background")
            return False
        if saver is None:
            return False
        # HELD, or Python frees the only reference and the window closes the
        # instant it opens.
        self._screensaver = saver
        saver.destroyed.connect(
            lambda *_a: setattr(self, "_screensaver", None))
        return True

    def _restart_the_backdrop(self) -> bool:
        """Send the animation back to its beginning.

        :returns: whether a backdrop was there to restart.

        The menu entry and Ctrl+R both come here, so the shortcut and the
        item cannot drift apart.
        """
        try:
            from .widgets.fractal_travel import (_LIVE_CONTROLS,
                                                 restart_the_dive)

            if not _LIVE_CONTROLS:
                return False
            restart_the_dive()
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not restart the backdrop", exc_info=True)
            return False
        self.statusBar().showMessage(tr("Backdrop restarted"), 1500)
        return True

    def _steer_the_backdrop(self, steps: int) -> bool:
        """Speed the spaceout descent up or slow it down.

        :param steps: notches; positive is faster.
        :returns: whether a backdrop took it.

        Up, Down and the wheel, as in the renderer this pattern came from.
        Handled HERE rather than by the backdrop, which must not accept
        events: it sits behind every control, and a widget that took the
        mouse would eat the click meant for the button on top of it.

        Reaching the window at all means nothing else wanted the key, so an
        arrow pressed in a table or a list still moves the selection.
        """
        try:
            from .widgets.fractal_travel import nudge_zoom_rate

            rate = nudge_zoom_rate(steps)
        except Exception:                                    # noqa: BLE001
            return False
        if not rate:
            return False
        self.statusBar().showMessage(tr("Zoom rate {rate:.2f}×").format(
            rate=rate), 1200)
        return True

    def keyPressEvent(self, event) -> None:
        """Up and Down change the spaceout zoom rate; Ctrl+R starts over."""
        from PySide6.QtCore import Qt

        key = event.key()
        if (key == Qt.Key.Key_R
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier
                and self._restart_the_backdrop()):
            event.accept()
            return
        if key == Qt.Key.Key_Up and self._steer_the_backdrop(1):
            event.accept()
            return
        if key == Qt.Key.Key_Down and self._steer_the_backdrop(-1):
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event) -> None:
        """The wheel does the same, a notch at a time."""
        notches = 0
        try:
            notches = int(event.angleDelta().y() / 120)
        except Exception:                                    # noqa: BLE001
            notches = 0
        if notches and self._steer_the_backdrop(notches):
            event.accept()
            return
        super().wheelEvent(event)

    def rebuild_app_screen(self, key: str, values=None) -> None:
        """Build ``key``'s screen again, carrying ``values`` across.

        :param key: the app whose screen is to be rebuilt.
        :param values: settings to apply to the new screen.

        WHY A WHOLE SCREEN. Which settings a form holds depends on a few of
        its own values -- the organelle count, and whether an object's
        channel names a plane -- so a committed change to one of those means
        a different form, not a changed one. Rebuilding is the same path
        every module open already takes.

        The old screen is dropped from the cache and destroyed, so the new
        one is built from scratch rather than reusing widgets that belong to
        a shape that no longer applies.
        """
        from .screens.app_screen import AppScreen

        old = self._screens.get(key)
        old_preset_owned = {}
        if old is not None:
            try:
                from copy import deepcopy

                old_preset_owned = deepcopy(
                    old._settings_model._organelle_preset_owned)
            except (AttributeError, TypeError):
                old_preset_owned = {}
        # BUILT BEFORE THE OLD ONE IS TAKEN AWAY. Removing it from the stack
        # first drops the window to whatever is left showing -- Home -- so
        # typing a channel value sent the user back to the start screen and
        # then returned them, which is not a visibility toggle by any
        # reading. The stack only ever changes once the replacement exists.
        previous_build_values = AppScreen.values_the_next_screen_is_built_for
        AppScreen.values_the_next_screen_is_built_for = dict(values or {})
        try:
            fresh = self._build_screen(key)
        except Exception:                                    # noqa: BLE001
            LOG.exception("could not rebuild the %s screen", key)
            return
        finally:
            # ALWAYS CLEARED. Every other module open must build from the
            # module's own defaults, and a value left here would shape the
            # next screen somebody opened for reasons they could not see.
            AppScreen.values_the_next_screen_is_built_for = previous_build_values

        try:
            self._theme_screen(fresh, key)
        except Exception:                                    # noqa: BLE001
            LOG.exception("Could not theme the rebuilt %s screen", key)
        try:
            from .i18n import retranslate_widget_tree

            retranslate_widget_tree(fresh)
        except Exception:                                    # noqa: BLE001
            LOG.exception("Could not translate the rebuilt %s screen", key)
        try:
            from .screens.settings_model import retarget_field_tooltips

            retarget_field_tooltips(fresh)
        except Exception:                                    # noqa: BLE001
            LOG.exception("Could not retarget help on the %s screen", key)

        # THE SHAPE IT WAS BUILT FOR, recorded before it is shown: the
        # signals that fire as it settles would otherwise see a shape they
        # have no record of and rebuild it again.
        try:
            fresh._form_shape_on_screen = fresh._form_shape()
        except Exception:                                    # noqa: BLE001
            pass
        try:
            fresh._settings_model._organelle_preset_owned = old_preset_owned
        except AttributeError:
            pass

        self._screens[key] = fresh
        self._stack.addWidget(fresh)
        self._stack.setCurrentWidget(fresh)
        if old is not None:
            try:
                # ``deleteLater`` alone bypasses ``AppScreen.closeEvent``.
                # Closing first retires workers, workspace providers, figure
                # resources and parentless pyqtgraph menus. The replacement
                # was built first so the stack never flashes Home while the
                # comparatively expensive form is constructed.
                if old.close() is False:
                    # A running worker may deliberately defer close. Keep
                    # that live screen instead of destroying work in flight,
                    # and retire the unused replacement cleanly.
                    self._stack.removeWidget(fresh)
                    fresh.setParent(None)
                    fresh.close()
                    fresh.deleteLater()
                    self._screens[key] = old
                    self._stack.setCurrentWidget(old)
                    old.register_workspace()
                    return
                # The old screen and the replacement own the same stable
                # workspace keys. Old's close withdrew them, so publish the
                # replacement again after teardown.
                fresh.register_workspace()
                self._stack.removeWidget(old)
                old.setParent(None)
                old.deleteLater()
            except Exception:                                # noqa: BLE001
                LOG.exception("could not retire the %s screen", key)
                try:
                    fresh.register_workspace()
                except Exception:                            # noqa: BLE001
                    LOG.debug("could not restore %s workspace", key,
                              exc_info=True)

    def _on_nav_selected(self, key: str):
        """Navigate to app ``key``, lazily instantiating its screen on first use."""
        interaction_started = _timing.interval_started("navigation", key)
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
            # Translated here, not left raw: the startup pass renders this
            # label once, and re-applying the English source over it both
            # shows the wrong word and opts the label out of every later
            # retranslation.
            self._status_app_label.setText(tr("Home"))
            self.statusBar().showMessage(tr("Home"), 2000)
            _timing.watch_interactive(
                self._startup, "interactive Home", "__home__",
                started_at=interaction_started,
                budget_s=_timing.HOME_BUDGET_S,
            )
            return
        if key not in self._screens:
            # SOMETHING ON SCREEN BEFORE THE WORK STARTS. The build cannot
            # move off the GUI thread -- Qt forbids making widgets anywhere
            # else, and painting is the GUI thread's too, which is why the
            # backdrop looked frozen while a module opened even though its
            # renderer never stopped. What CAN change is that the user is
            # looking at a module that says it is preparing, rather than at
            # the old screen doing nothing.
            #
            # The build already yields to the event loop on a 25 ms
            # deadline, so this card animates while the widgets are made.
            card = self._show_preparing(key)
            try:
                self._screens[key] = self._build_screen(key)
            finally:
                self._hide_preparing(card)
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
        # THE HELP GOES ON THE NAMES, for every screen and not only the
        # ones built from settings rows. Here because this is the one place
        # they all pass through, and after the translate above: that pass
        # re-applies each setting's tooltip to whatever carries its key, so
        # moving the help before it runs is undone a moment later.
        try:
            from .screens.settings_model import retarget_field_tooltips

            retarget_field_tooltips(self._screens[key])
        except Exception:
            # Help in the wrong place is a blemish, never a reason for a
            # module not to open.
            LOG.exception("Could not retarget help on the %s screen", key)
        self._stack.setCurrentWidget(self._screens[key])
        # Constructor return is not readiness.  The event filter records only
        # after this page and one of its enabled controls have both painted on
        # an event-loop turn, which is the state a user can actually operate.
        _timing.watch_interactive(
            self._screens[key], "interactive module", key,
            started_at=interaction_started,
            budget_s=_timing.MODULE_BUDGET_S,
        )
        # Move this app to the end of the visit list. Revisiting an app
        # has to count as the most recent visit — otherwise "Add current
        # plate" on the Queue screen picks up whichever app was OPENED
        # last rather than the one that was on screen a moment ago.
        if key in self._visit_order:
            self._visit_order.remove(key)
        self._visit_order.append(key)
        # Find nice display name
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

    def _on_investigate_hit_requested(self, request: dict) -> None:
        """Open Investigate Hit on the exact result selected in Hit List."""
        effect = float(request.get("effect", 0.0) or 0.0)
        self._on_train_requested("investigate_hit", {
            "results_folder": request.get("folder", ""),
            "target_gene": request.get("gene", ""),
            "target_guides": list(request.get("guides", ())),
            "hit_effect": effect,
            "hit_fdr": request.get("fdr", 1.0),
            "hit_guide_agreement": request.get("guide_agreement", float("nan")),
            "hit_n_guides": request.get("n_guides", 0),
            "hit_well_support": request.get("well_support", 0),
            "hit_direction": "positive" if effect >= 0 else "negative",
            "hit_phenotype": request.get("phenotype", ""),
        })

    def _theme_screen(self, screen: QWidget, key: str) -> None:
        """Apply late QSS, clear containers and add the ambient backdrop.

        Skipped for anything that already handles its own: ``AppScreen`` does
        the latter two in its constructor, and the sequencing screen has the
        DNA rain.  Late QSS is applied before that branch because every screen
        passes here before it is inserted into the visible stack.
        """
        from .screens.app_screen import AppScreen, uses_ambient_background
        from .theme import clear_container_surfaces, ensure_widget_qss_applied

        # A local stylesheet reaches this root and its descendants without
        # making QApplication re-polish Home and every cached module.  The
        # root is not in the stack yet, so these rules win the first paint.
        ensure_widget_qss_applied(root=screen)

        if isinstance(screen, AppScreen):
            return

        clear_container_surfaces(screen)

        if not uses_ambient_background(key):
            return
        self._install_screen_backdrop(screen, key)

    def _install_screen_backdrop(self, screen: QWidget, key: str) -> None:
        """Put the ambient backdrop behind a screen that has no `AppScreen`.

        Separate from :meth:`_theme_screen` because it is the one part of
        the theming that can honestly answer "not yet": the spaceout
        backdrop needs a GL context, and building one while the pipeline
        preloader is importing torch is the concurrent initialisation
        `HEAVY_IMPORT_LOCK` exists to prevent. Re-entered on a timer in
        that case, where re-running the whole of `_theme_screen` would
        re-apply the stylesheet to the tree for nothing.

        Never raises: a module opens with or without its decoration.
        """
        try:
            from .preferences import (
                get_ambient_enabled,
                get_ambient_palette,
                get_ambient_theme,
                resolve_effective_theme,
                theme_background_path,
            )
            if not get_ambient_enabled():
                return
            # NOT WHILE A HEAVY IMPORT IS RUNNING, and this runs on the GUI
            # thread as a module is being opened -- which is precisely when
            # the preloader is holding the lock. `AppScreen` has taken this
            # care since instruction 315; the screens that build their own
            # had not, so under `spaceout` they froze where `spacr` did not.
            from .widgets.ambient import (install_ambient,
                                          _the_heavy_import_lock_is_free)
            if not _the_heavy_import_lock_is_free():
                self._retry_screen_backdrop(screen, key)
                return
            # The spaceout fractal is installed by `install_ambient` itself
            # (instruction 260), so this caller needs no branch: hooking the
            # three call sites separately is what left the Home screen still
            # showing the old artwork.
            install_ambient(
                screen, None,
                theme=get_ambient_theme(), palette=get_ambient_palette(),
                backdrop=theme_background_path(resolve_effective_theme()))
        except Exception as error:
            # The peek is a check, not a reservation: the preloader re-takes
            # the lock between two imports, so the refusal can still arrive
            # here. It means "not yet" and not "this machine cannot", and
            # logging it as an exception would put a traceback in the console
            # for an ordinary click made during startup.
            try:
                from .widgets.ambient import _the_backdrop_wants_a_retry
            except Exception:                                # noqa: BLE001
                _the_backdrop_wants_a_retry = None
            if (_the_backdrop_wants_a_retry is not None
                    and _the_backdrop_wants_a_retry(error)):
                self._retry_screen_backdrop(screen, key)
                return
            LOG.exception("Could not install the backdrop for %s", key)

    def _retry_screen_backdrop(self, screen: QWidget, key: str) -> None:
        """Come back for :meth:`_install_screen_backdrop` shortly.

        The screen is checked for liveness at the far end rather than
        here: a module can be closed inside the interval, and calling a
        method on a freed QWidget is the "Internal C++ object already
        deleted" storm this file has had before.
        """
        from PySide6.QtCore import QTimer

        def again() -> None:
            try:
                from shiboken6 import isValid
            except Exception:                                # noqa: BLE001
                return
            if not isValid(screen):
                return
            self._install_screen_backdrop(screen, key)

        QTimer.singleShot(120, again)

    def _show_preparing(self, key: str):
        """Put a "preparing" card up before a module is built.

        :returns: the card, to hand back to :meth:`_hide_preparing`, or None
            when there was nowhere to put one. Never raises: a module must
            open whether or not its loading card could be shown.

        It is deliberately NOT a progress bar. The build has no honest
        percentage -- the settings arrive in whatever order the category map
        lists them -- and a bar that jumps or sits still is worse than a
        moving thing that promises nothing.
        """
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtWidgets import QLabel

            from .i18n import tr

            label = registered_metadata(key).get("label") or key
            card = QLabel(tr("Preparing {name}…").format(name=label), self)
            card.setObjectName("PreparingCard")
            card.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            card.setStyleSheet(
                "background: rgba(10,14,24,200); color: rgb(235,240,245); "
                "padding: 14px 22px; border-radius: 10px; font-size: 15px;")
            card.adjustSize()
            card.move(max(0, (self.width() - card.width()) // 2),
                      max(0, (self.height() - card.height()) // 2))
            card.raise_()
            card.show()
            # ONE PAINT BEFORE THE WORK. Without this the card is created
            # and the build starts in the same tick, so it is never drawn
            # and the user sees the freeze it exists to replace.
            from PySide6.QtCore import QCoreApplication, QEventLoop

            QCoreApplication.processEvents(
                QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            return card
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not show the preparing card", exc_info=True)
            return None

    def _hide_preparing(self, card) -> None:
        """Take the card down. Safe with None and after any failure."""
        if card is None:
            return
        try:
            card.hide()
            card.deleteLater()
        except Exception:                                    # noqa: BLE001
            pass

    def _build_screen(self, key: str) -> QWidget:
        with _timing.span("build screen", key):
            return self._build_screen_timed(key)

    def _build_screen_timed(self, key: str) -> QWidget:
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
        if key == "train_cellpose":
            # WRITTEN OUT RATHER THAN CALLING THE MODULE'S OWN FACTORY:
            # tests/qt/test_all_module_smoke.py reads this method's
            # bytecode for the `self._on_*` slots it wires, and a factory
            # call hides them from it.
            from .screens.train_cellpose import CellposeWorkbenchScreen
            screen = CellposeWorkbenchScreen()
            screen.error_explain_requested.connect(self._on_explain_error)
            screen.remote_submit_requested.connect(
                self._on_remote_submit_requested)
            return screen
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
        annotate screen's Train CV / Train XG buttons, and by Run
        History's "load this run's settings".

        THE KEY A RECORD NAMES IS NOT ALWAYS A SCREEN. A module that was
        merged or folded into a host keeps its key in every run journal
        ever written, and navigating to it builds an orphan -- a page
        with no sidebar row, no tile and no way back to it. The key is
        resolved to the screen that carries it first, so a Timelapse run
        saved before the fold reopens on Mask Generation with tracking
        switched on, which is where its settings now live.
        """
        target_key = self.open_module(target_key)
        widget = self._screens.get(target_key)
        if widget is None:
            return
        # A screen that is not settings-driven says what to do with a seed
        # itself. The Database Browser has no settings model -- it takes a
        # database path and a table -- and without this the navigation
        # happened and the seed was silently dropped.
        seeder = getattr(widget, "apply_seed", None)
        if callable(seeder):
            try:
                seeder(dict(seed))
            except Exception:
                LOG.warning("Could not seed %s with %r", target_key, seed,
                            exc_info=True)
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
        # A folded module's pipeline GATE has no widget for the loop above
        # to land in -- the masthead switch is its control -- so a seed
        # that asks for tracking has to move the switch, exactly as an
        # imported settings CSV does.
        sync = getattr(widget, "_sync_folded_switches", None)
        if callable(sync):
            sync(dict(seed))

    @staticmethod
    def _apply_seed_value(w: QWidget, value) -> None:
        from PySide6.QtWidgets import (
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QLineEdit,
            QSpinBox,
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


def _use_open_sans(app, weight: str = "") -> str:
    """Make Open Sans the application font, not merely a registered one.

    THE FONTS WERE LOADED AND NEVER APPLIED. `_load_bundled_fonts` registers
    the family with QFontDatabase, which makes a `font-family: "Open Sans"`
    stylesheet rule resolvable -- but nothing set the APPLICATION font, so
    every widget without such a rule used the platform default. That is what
    Qt was naming in "OpenType support missing for Ubuntu Sans": the app was
    drawing in the system font, not in the one it ships.

    :param weight: ``'light'`` or ``'regular'``; empty reads the preference.
    :returns: the family actually applied, or "" when Open Sans is not
        available -- in which case the platform font is left alone, because a
        missing font is not a reason to refuse to draw.

    Only these two weights are offered. Bold and SemiBold remain registered,
    so a stylesheet that asks for emphasis still gets it; what changes is
    what everything else defaults to.
    """
    from PySide6.QtGui import QFont, QFontDatabase

    if "Open Sans" not in set(QFontDatabase.families()):
        LOG.debug("Open Sans is not registered; leaving the platform font")
        return ""

    if not weight:
        try:
            from .preferences import get_interface_font_weight

            weight = get_interface_font_weight()
        except Exception:                                    # noqa: BLE001
            weight = "regular"

    font = QFont("Open Sans")
    # QFont.Light is 300 and Normal is 400. Asked for by weight rather than
    # by family name: "Open Sans Light" is a family on some platforms and
    # not on others, and the weight works on both.
    font.setWeight(QFont.Weight.Light if str(weight).lower() == "light"
                   else QFont.Weight.Normal)
    # The size the platform chose is kept: the font-scale preference is
    # applied on top of it later, and overriding it here would silently
    # undo that.
    existing = app.font()
    if existing.pointSizeF() > 0:
        font.setPointSizeF(existing.pointSizeF())
    app.setFont(font)
    return font.family()


def _load_bundled_fonts() -> None:
    """Register the bundled Open Sans TTFs with :class:`QFontDatabase`.

    Idempotent — the fonts are only loaded once even if called
    multiple times (Qt tracks the file path).
    """
    from PySide6.QtGui import QFontDatabase
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fonts_dir = os.path.join(
        package_dir, "resources", "font", "open_sans", "static"
    )
    if not os.path.isdir(fonts_dir):
        return
    for name in os.listdir(fonts_dir):
        if name.lower().endswith((".ttf", ".otf")):
            QFontDatabase.addApplicationFont(os.path.join(fonts_dir, name))



#: Where a fatal signal's stacks are written, beside the ordinary log.
CRASH_DUMP_NAME = "spacr-crash.log"


def _install_crash_dump():
    """Write every thread's Python stack if the process dies on a signal.

    The file is opened for APPEND and kept open for the life of the process:
    faulthandler writes from a signal handler, where opening a file is not
    allowed. It is never closed for the same reason.

    :returns: the path being written to, or ``""``.
    """
    import faulthandler

    try:
        from ..logging_util import log_dir
    except Exception:                                    # noqa: BLE001
        log_dir = None
    try:
        folder = log_dir() if callable(log_dir) else None
    except Exception:                                    # noqa: BLE001
        folder = None
    if not folder:
        folder = os.path.join(os.path.expanduser("~"), ".spacr", "logs")
    try:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, CRASH_DUMP_NAME)
        handle = open(path, "a", buffering=1)
        handle.write(f"\n=== spaCR started (pid {os.getpid()}) ===\n")
        faulthandler.enable(file=handle, all_threads=True)
        # Kept on the module so the handle cannot be garbage collected --
        # faulthandler writes to the file descriptor, and a closed one is a
        # crash inside the crash handler.
        globals()["_CRASH_DUMP_FILE"] = handle
        LOG.info("fatal-signal stacks will be written to %s", path)
        return path
    except Exception:                                    # noqa: BLE001
        LOG.debug("could not install the crash dump", exc_info=True)
        return ""

#: The application-wide event filters that exist for DIALOGS, and nothing
#: else -- the card and the rim, the detach-from-the-main-window, and the
#: translation of a dialog built and executed in one expression.
#:
#: Each is a Python callable Qt runs for EVERY event delivered to EVERY
#: object in the application, and the main window's construction delivers
#: tens of thousands of them. Installed before the window was built, they
#: were a large and measurable share of the wait between typing `spacr` and
#: seeing a window, spent asking of every QLabel, every layout item and
#: every menu action whether it is a QDialog. None of them can have anything
#: to do until a dialog exists, and no dialog exists until the event loop is
#: running.
#:
#: The first-run setup screen is the one dialog that opens before the loop,
#: and it needs none of the three: it builds its own card, so
#: `glass.wants_glass` leaves it alone; it goes frameless and top-level in
#: its own constructor; and it translates itself with `tr` and its own
#: `retranslate` pass. Each of those three is held to it by
#: `tests/qt/test_the_window_comes_before_the_dialog_filters.py`.
_DIALOG_FILTERS = (
    ("spacr.qt.dialogs", "detach_all_dialogs"),
    ("spacr.qt.widgets.glass", "install_glass_everywhere"),
    ("spacr.qt.i18n", "install_dialog_translation"),
)


def install_the_dialog_filters(app) -> tuple[str, ...]:
    """Install the registered dialog event filters on a Qt application.

    Installers are idempotent and isolated: a failure in one filter does not
    prevent the remaining filters from being installed.

    :returns: Names of the filters installed successfully.
    """
    import importlib

    installed: list[str] = []
    for module_name, function_name in _DIALOG_FILTERS:
        try:
            getattr(importlib.import_module(module_name), function_name)(app)
        except Exception:                                # noqa: BLE001
            LOG.debug("could not install %s", function_name, exc_info=True)
        else:
            installed.append(function_name)
    return tuple(installed)


def launch(argv: Optional[list[str]] = None) -> int:
    """Bootstrap QApplication and show the main window."""
    _timing.begin()
    if argv is None:
        argv = sys.argv[1:]

    # BEFORE THE BACKDROP IS BUILT. Two unclean exits in a row is treated as
    # a pattern, and the next start is made without the one thing spaCR asks
    # a driver to do -- because the setting that would turn it off is behind
    # the window that never appears.
    from .crash_recovery import (note_that_a_launch_began,
                                 should_start_without_the_backdrop,
                                 take_the_backdrop_out_of_this_launch)

    _unclean = note_that_a_launch_began()
    if should_start_without_the_backdrop(_unclean):
        take_the_backdrop_out_of_this_launch()
        LOG.warning(
            "spaCR did not shut down cleanly %d times running, so this "
            "start has no animated background. It comes back by itself "
            "after one clean run; `safespacr` starts with everything off.",
            _unclean)

    # `--no-setup` and friends are taken out first, because the next line
    # reads argv[0] as a module name and would look one of them up.
    from .setup_screen import take_the_setup_flags

    argv, told_to_skip_setup = take_the_setup_flags(argv)

    # Support `spacr-qt <app>` to open directly into an app.
    initial_app = argv[0] if argv else None

    # A FATAL SIGNAL MUST LEAVE A STACK BEHIND.
    #
    # Reported 2026-08-19 three times: a regression run closes [success] and
    # the process is gone milliseconds later. The log ends mid-session with
    # no shutdown lines, so it is not a clean exit; dmesg and coredumpctl
    # have nothing; and Python prints nothing because the process dies below
    # Python, in Qt or in a C extension. Three hypotheses were tested and
    # eliminated against real sessions -- an off-thread plt.show(), pyplot
    # building Qt canvases on the worker, and quitOnLastWindowClosed -- each
    # costing a launch-and-reproduce cycle for the maintainer.
    #
    # faulthandler writes the Python stack of EVERY thread on SIGSEGV,
    # SIGABRT, SIGBUS and SIGFPE. It costs nothing until one arrives, and the
    # next occurrence then names the frame instead of the minute.
    _install_crash_dump()

    # Enable high-DPI early.
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    # PYPLOT MUST NEVER MAKE A Qt CANVAS. Set here, before any figure can
    # exist, because switching the backend later CLOSES every open figure.
    #
    # A regression runs on a JobRunner worker and draws with pyplot. Under the
    # `qtagg` backend every `plt.figure()` on that worker builds a
    # FigureCanvasQTAgg -- a QObject whose thread affinity is the WORKER. The
    # main thread then renders it, and Qt answers with "QBasicTimer::start:
    # Timers cannot be started from another thread" followed, milliseconds
    # after `run closed [success]`, by the process going away with no Python
    # traceback. That is the reported "it just spontaneously quit", and the
    # `Internal C++ object (FigureCanvasQTAgg) already deleted` errors in the
    # log are the same object seen from the other side.
    #
    # `bridge` already asked for Agg with `force=False`, which does NOTHING
    # once a backend is active -- and by the time a run starts, `qtagg` is.
    #
    # NOTHING IS LOST. The two places that genuinely want a Qt canvas
    # (`figure_queue`, `umap_explorer`) import FigureCanvasQTAgg and build it
    # themselves on the GUI thread, which works under any global backend.
    #
    # SAID IN THE ENVIRONMENT WHEN MATPLOTLIB IS NOT LOADED YET, which on a
    # normal launch it is not: nothing imported up to this line has needed
    # it. `matplotlib.use` can only speak to a matplotlib that exists, so
    # calling it here used to import the whole package -- tens of
    # milliseconds of a launch that has not yet drawn anything -- purely to
    # set a string that MPLBACKEND sets for free, and that matplotlib reads
    # for itself whenever it does load. Assigned rather than `setdefault`:
    # the reason this exists is that a Qt canvas built off the GUI thread
    # kills the process, so the choice is not the caller's to override.
    #
    # And when matplotlib IS already imported the environment is too late --
    # the backend was read at its import -- so that case still forces it.
    try:
        if "matplotlib" in sys.modules:
            import matplotlib

            matplotlib.use("Agg", force=True)
        else:
            os.environ["MPLBACKEND"] = "Agg"
    except Exception:                                    # noqa: BLE001
        pass

    # EVERY DIALOG IS QT'S OWN, NOT THE DESKTOP'S. Instruction 151: a native
    # dialog on this desktop is brokered through xdg-desktop-portal, and a
    # brokered dialog is the tens-of-seconds stall reported as "changing the
    # line width takes like 1 minute" -- the restyle itself measured at
    # 0.000 s. The colour pickers were fixed one call site at a time; there
    # are 117 QFileDialog calls across the widget package and five of them
    # passed the option, so per-site fixing was never going to converge.
    #
    # SET BEFORE THE QApplication EXISTS, which is what Qt requires of this
    # attribute -- after construction it is ignored, silently, which would
    # look exactly like it had worked.
    #
    # The trade is real and worth stating: Qt's dialogs do not carry the
    # desktop's bookmarks or its recent-files list. A file chooser that opens
    # is better than a beautiful one that takes a minute, and a user who
    # wants the native one can still set QT_QPA_PLATFORMTHEME.
    QApplication.setAttribute(Qt.AA_DontUseNativeDialogs, True)

    # THE APPLICATION IS NAMED BEFORE IT EXISTS.
    #
    # On macOS the application menu -- the one beside the Apple logo, and the
    # one Qt moves Preferences, Quit and About INTO -- is built while the
    # Cocoa plugin comes up inside the QApplication constructor, from the
    # application name and from the running bundle's CFBundleName. Naming the
    # application on the line after that constructor is a name the menu never
    # read: it stays "python", "spacr-qt" or "PySideApp" depending on how the
    # launch happened, and the maintainer's report -- "preferences and quit
    # are for some reason not in the spacr dropdown" -- is that menu being
    # somewhere they had no reason to look.
    #
    # `applicationDisplayName` is set too. It was never set at all, and it is
    # the name Qt shows to people rather than the one it keys settings on.
    #
    # Returns what actually took effect rather than what was asked for, so
    # the launch log records the answer instead of the intention.
    from .menus import name_the_application

    _app_name, _app_display_name = name_the_application()

    app = QApplication(sys.argv[:1])
    # NAME THE CALLER OF AN OFF-THREAD TIMER START, because Qt will not. Its
    # own warning has no file, no function and no thread in it, and the event
    # arrives during a real run and is followed by the process dying -- so
    # there is no opportunity to switch instrumentation on afterwards.
    try:
        from .thread_guard import install as _install_thread_guard
        _install_thread_guard()
    except Exception:
        pass
    # Logged as MEASURED rather than as intended: a name that silently failed
    # to take looks exactly like one that worked until somebody opens the
    # menu on a Mac, and this line is what a bug report can be read against.
    LOG.info("application named %r (display %r); Qt reports %r / %r",
             _app_name, _app_display_name, app.applicationName(),
             app.applicationDisplayName())
    # LAPTOP MODE, decided once and SAID. It is the fallback the laptop
    # instruction calls the fallback -- reached after the optimisations, and
    # it turns down what is decorative rather than removing what a module
    # does. Nothing it touches is read by a pipeline, so a run computes the
    # same answer either way; that is what makes deciding automatically
    # acceptable. Overridable through SPACR_LAPTOP_MODE either way.
    try:
        from .laptop_mode import apply as _apply_laptop_mode, describe
        LOG.info("%s", describe())
        _laptop = _apply_laptop_mode()
        if _laptop["changed"]:
            LOG.info("laptop mode changed: %s", ", ".join(_laptop["changed"]))
    except Exception:                                    # pragma: no cover
        LOG.debug("could not decide laptop mode", exc_info=True)
    # Linux shells resolve dock/switcher identity through the desktop-file
    # id (Wayland does not use setWindowIcon for that surface).
    app.setDesktopFileName("io.github.olafssonlab.spacr")
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
    with _timing.span("fonts"):
        _load_bundled_fonts()
        _use_open_sans(app)

    # Apply user preferences (theme + font scale) — falls back to the
    # dark defaults on the first launch when nothing is stored yet.
    from .preferences import apply_preferences_to_app
    apply_preferences_to_app(app)
    # QT'S OWN WORDS. Copy, Paste, Select All, a file dialog's whole chrome
    # and every message box's buttons come from Qt's catalogs, not from
    # spaCR's, so they stay English until this is loaded. This one is not a
    # dialog filter and stays here: it is read while the main window's own
    # menus and buttons are built.
    from .i18n import install_qt_translations
    install_qt_translations(app)

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

    # THE APPLICATION OUTLIVES ITS WINDOWS UNTIL THE MAIN ONE CLOSES.
    #
    # Qt's default is quitOnLastWindowClosed=True: the moment the number of
    # visible top-level windows reaches zero, the event loop stops and the
    # process exits CLEANLY -- no traceback, no core dump, the log simply
    # ends. That is exactly the evidence for the reported "i ran it again and
    # it just spontaneously quit": two runs closed [success], the log stops
    # mid-session, and neither dmesg nor coredumpctl recorded anything,
    # because nothing crashed.
    #
    # A run makes and destroys top-level windows -- a figure canvas being
    # rebuilt, a transient dialog, a progress window -- and any instant with
    # none of them up while the main window is not counted takes the whole
    # application with it.
    #
    # So the main window decides, and it is the only thing that does.
    app.setQuitOnLastWindowClosed(False)

    # THE FIRST RUN ASKS ITS QUESTIONS BEFORE THERE IS AN APPLICATION TO
    # ASK THEM OVER (instruction 221, reordered).
    #
    # It used to run after `win.show()` so that it had something to blur.
    # That put a half-built main window -- wrong language, wrong theme,
    # wrong font -- on screen for as long as the setup took, and then
    # restyled it under the user while they were reading. The answers given
    # here decide how the main window is BUILT, so they have to be given
    # first. The screen carries its own backdrop and does not need a window
    # behind it.
    #
    # A launch can decline: `--no-setup`, `SPACR_NO_SETUP=1`, or an
    # offscreen platform plugin. See `setup_screen.skipped_on_purpose`.
    if not told_to_skip_setup:
        try:
            # THE SLIDES, not the grouped form (instruction 234).
            # `setup_dialog` is 221's version and stays: it is the fallback
            # nothing currently uses, and deleting it would take its tests
            # with it while the new presentation is still settling.
            from .widgets.setup_slides import open_setup_if_needed

            asked = open_setup_if_needed(None)
            # An answer may have changed the language, the theme or the font
            # scale, and the main window has not been built yet -- so it is
            # built from the new values rather than restyled into them.
            #
            # ONLY WHEN THERE WERE ANSWERS. `open_setup_if_needed` returns
            # None when the screen was not shown -- this profile has already
            # answered, or nobody is there to -- which is every launch after
            # the first, and re-applying preferences nothing changed is the
            # whole theme resolved and set on the application twice.
            if asked is not None:
                apply_preferences_to_app(app)
        except Exception:
            # A setup screen is not worth a launch. Every question it asks
            # has a working default, so a user who never sees it is exactly
            # where a user who dismissed it would be.
            LOG.debug("could not open the setup screen", exc_info=True)

    _timing.mark("MainWindow")
    win = MainWindow(initial_app=initial_app)
    benchmark_controller = None
    if os.environ.get("SPACR_BENCHMARK_JSON", "").strip():
        from .startup_benchmark import maybe_start as _maybe_start_benchmark

        # This is an explicitly requested, unattended acceptance run.  An
        # instrumentation setup error must fail the worker promptly instead
        # of opening a GUI that can sit until the driver's outer timeout.
        benchmark_controller = _maybe_start_benchmark(app, win)

    # Home is not ready because its constructor returned or because show()
    # was called.  Install before show so no paint can escape the observer;
    # the probe also requires a callback delivered after app.exec() begins
    # and an enabled, visible control whose own paint event has completed.
    if win._stack.currentWidget() is win._startup:
        _timing.watch_interactive(
            win._startup, "interactive Home", "__home__",
            started_at=_timing.process_started_at(),
            budget_s=_timing.HOME_BUDGET_S,
        )
    # Retain the optional controller with the window.  QObject parenting is
    # sufficient for C++ lifetime, but the explicit Python reference avoids
    # wrapper collection differences across the supported PySide releases.
    win._startup_benchmark_controller = benchmark_controller
    # Opens at its own size rather than maximised. Maximising assumes a
    # desktop: over X11 forwarding, VNC or a virtual framebuffer the
    # "available geometry" is whatever the remote session claims, which is
    # frequently one enormous virtual desktop or a 640x480 stub, and the
    # window arrives unusable either way. The user can still maximise it,
    # and the 1200x720 minimum this window declares is a sane opening size
    # on a real display.
    win.show()

    # AND ONLY NOW THE DIALOG FILTERS. See :data:`_DIALOG_FILTERS`: they are
    # application-wide event filters that concern dialogs alone, so every
    # event the main window's construction delivers used to run three Python
    # callables that could not act on it. Installed here they are in place
    # before the event loop -- which is before any dialog can be opened --
    # and the window they cannot help build arrives 0.4 s sooner.
    install_the_dialog_filters(app)

    # Pre-warm the heavy imports that a module screen needs (spacr.gui_utils
    # pulls torch + cv2 ≈ 3-4 s; spacr.settings ≈ 1 s) in a BACKGROUND thread
    # while the user looks at the home screen. By the time they open a module
    # these are cached, so the module snaps open instead of freezing on the
    # first import. Importing modules (no Qt objects) off-thread is safe.
    def _prewarm():
        try:
            import importlib
            for mod in ("spacr.settings",):
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
        # EVERY JOB RUNNER, NOT ONLY THE CONSOLES. Qt aborts the process if a
        # running QThread is destroyed, and each runner's own `closeEvent`
        # covers a widget being CLOSED -- not the application quitting with a
        # job in flight, where the widget is destroyed without ever closing.
        # Measured 2026-08-19: spaCR died immediately after every successful
        # regression, "run closed [success]" the last line in the log and
        # nothing after it, because the Runs tab's announce had just started a
        # results read on a worker.
        try:
            from .job_runner import shutdown_all
            shutdown_all()
        except Exception:
            LOG.debug("Could not drain the job runners", exc_info=True)
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

    _timing.mark("entering the event loop")
    _timing.watch_the_gui_thread(app)
    # Unlike the mark above, this callback can run only after exec() has begun
    # dispatching.  Readiness probes refuse to report before it arrives.
    from PySide6.QtCore import QTimer as _TimingQTimer

    _TimingQTimer.singleShot(0, _timing.event_loop_started)
    try:
        code = app.exec()
        # A RETURN FROM `exec` IS A CLEAN SHUTDOWN, whatever the exit code:
        # the event loop ran and ended, which a crash never does. Clearing
        # the count here rather than on code == 0 means a run the user quit
        # from an error dialog still counts as "it started fine".
        try:
            from .crash_recovery import note_a_clean_shutdown

            note_a_clean_shutdown()
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not record a clean shutdown", exc_info=True)
        return code
    finally:
        written = _timing.write_report()
        if written:
            print(f"spaCR timing report written to {written}")
