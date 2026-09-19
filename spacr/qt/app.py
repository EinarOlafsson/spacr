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
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import (QEvent, QObject, Qt,
                            QThread, Signal)
from PySide6.QtGui import (QAction, QColor, QIcon, QKeySequence,
                           QPalette)
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QStackedWidget,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


from .hidpi import scaled_for
from .i18n import tr
from . import iconset
from .app_catalog import (LazyScreenFactory, declared_for as _declared_for,
                          register_declared as _register_declared)
from .widgets.dock import Dock

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


def _carry_preview_state(old, fresh) -> None:
    """Move the live preview's loaded image from a retiring screen to its
    replacement.

    Best effort and silent on failure: a rebuild that raised here would cost
    the user their whole screen to save them a re-load, which is the wrong
    trade. Each attribute is copied independently for the same reason.
    """
    if old is None or fresh is None:
        return
    for name in ("_live_preview", "_preview_panel", "_live_panel"):
        source = getattr(old, name, None)
        target = getattr(fresh, name, None)
        if source is None or target is None:
            continue
        for attribute in ("_image", "_image_path", "_path_full", "_settings"):
            try:
                value = getattr(source, attribute, None)
                if value is not None:
                    setattr(target, attribute, value)
            except Exception:                                # noqa: BLE001
                continue
        try:
            if getattr(target, "_image", None) is not None:
                target._show_elided_path()
                target._refresh_source_selectors()
                target._refresh_canvases()
        except Exception:                                    # noqa: BLE001
            pass


class _FractalFollowsItsScreen(QObject):
    """Keeps the spaceout backdrop the same size as the screen behind it.

    :param widget: the backdrop to resize.
    :param screen: the screen whose resizes drive it, and THE QOBJECT PARENT
        -- so this dies with the screen it follows rather than with the
        backdrop it moves.
    """

    def __init__(self, widget, screen) -> None:
        """Take the screen as parent and remember the backdrop to resize."""
        super().__init__(screen)
        self._widget = widget

    def eventFilter(self, watched, event) -> bool:
        """Resize the backdrop to match the screen it sits behind.

        Returns ``False`` always: the resize is observed, never consumed, so the
        screen still lays itself out.

        :param watched: the screen being followed.
        :param event: the event.
        :returns: ``False``.
        """
        if event.type() == QEvent.Type.Resize:
            try:
                self._widget.setGeometry(watched.rect())
            except Exception:                                # noqa: BLE001
                pass
        return False


def _open_at_the_measured_width(window) -> bool:
    """Widen a fresh window to what the modules were measured to need.

    :returns: True when the window was resized.

    A window that opens too narrow cuts off the right-hand side of a
    module's settings. The width comes from `spacr.qt._layout_policy`,
    which reads a generated artifact -- every number in it measured by
    building each module offscreen and asking whether its settings column
    is holding more than it can show.

    IT ONLY EVER GROWS, and that is deliberate rather than cautious. The
    comment above `win.show()` records why this window does not maximise:
    over X11 forwarding, VNC or a virtual framebuffer the "available
    geometry" is whatever the remote session claims, and it is frequently
    a stub. A policy allowed to shrink would take that claim seriously and
    open a window smaller than the one the user has been getting for a
    year. Growing on a bad number is bounded by the clamp; shrinking on
    one is not bounded by anything.

    WITHOUT THE ARTIFACT NOTHING HAPPENS. `layout_policy` falls back to
    1200 px -- the width the text-fit sweep builds every screen at -- and
    this window already opens at 1200, so a wheel that does not carry the
    file behaves exactly as it did.

    Never raises. An opening size is not worth failing a launch over.
    """
    try:
        from ._layout_policy import recommended_window_size, why
        from .preferences import (_get_layout_decision,
                                  _set_layout_decision, get_font_scale)

        handle = window.screen() or QApplication.primaryScreen()
        if handle is None:
            return False
        available = handle.availableGeometry()
        scale = float(get_font_scale() or 1.0)
        metrics = {"available": [available.width(), available.height()],
                   "font_scale": round(scale, 4)}

        recorded = _get_layout_decision()
        if recorded and all(recorded.get(k) == v for k, v in metrics.items()):
            wanted = int(recorded.get("width") or 0)
        else:
            wanted, _height = recommended_window_size(
                (available.width(), available.height()), scale)
            _set_layout_decision({**metrics, "width": int(wanted),
                                 "reason": why(scale)})
        if wanted <= window.width():
            return False
        window.resize(min(int(wanted), available.width()), window.height())
        LOG.info("opened at %d px: %s", window.width(), why(scale))
        return True
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not apply the measured layout policy",
                  exc_info=True)
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
        """Hold the named operation and the callable that performs it."""
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
        """Let a press on the menu bar's empty space drag the window.

        :param window: the frameless window to move. Also the QObject
            parent, so the filter dies with what it moves.
        """
        super().__init__(window)
        self._window = window

    def eventFilter(self, watched, event):    # noqa: N802 - Qt naming
        """Start a system window move when the bare menu strip is pressed.

        The window is frameless, so the menu bar is what you drag it by. A press
        over an actual MENU is let through -- otherwise opening File would move
        the window instead. The move is handed to the compositor rather than
        implemented here, so it snaps and tiles like any other window.

        :param watched: the menu bar.
        :param event: the event.
        :returns: ``True`` when the drag was started and the press consumed,
            ``False`` otherwise.
        """
        if event.type() != QEvent.Type.MouseButtonPress:
            return False
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        try:
            if watched.actionAt(event.position().toPoint()) is not None:
                return False
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
#: A DOCK HEADING, NOT A HOME CATEGORY, and the distinction is the whole
#: reason this is not in :data:`SECTION_ORDER`.
#:
#: The Help-menu modules were filed under Data, so the dock drew them
#: under Data next to the modules that get data in, which is not what
#: any of them does. The maintainer asked for them under a Help heading,
#: lowest in the dock.
#:
#: PUTTING IT IN `SECTION_ORDER` WAS TRIED FIRST AND REVERTED. A section
#: is Home's categorisation: `home_categories`, `home_bands` and
#: `section_members` all mean "a tab and the tiles on it", and thirteen
#: tests defend the invariant that a declared section has tiles --
#: `test_no_section_is_empty` says so by name. Every Help module is
#: TILELESS by construction, so Help can never satisfy that and should
#: not pretend to. The dock groups by this instead, and Home never sees
#: it.
SECTION_HELP = "Help"

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
    SECTION_HELP: ("Look something up or administer work that already "
                   "exists: run history, the pipeline graph, the "
                   "database browser, reports and the job runners."),
    SECTION_DESIGN: ("Plan statistical power, sample size, plate layouts, "
                     "controls and replicates."),
}

#: A plugin's declared section, mapped onto a section Home actually
#: draws.
#:
#: EVERY VALUE HERE MUST BE IN `SECTION_ORDER`, and that is not a style
#: rule. `register_app` refuses a section it does not know, the plugin
#: loop below catches the ValueError per plugin, and the app is then
#: dropped in silence.
#:
#: That is not hypothetical: the 2026-08-31 restructure left
#: Core/Data/Tools/Assays, and this map still pointed `results`,
#: `models`, `explore` and `design` at the retired names. Since
#: `AppContribution.section` DEFAULTS to "results", every plugin app
#: that did not name a section was being thrown away -- and the only
#: sign of it was one red test.
#:
#: The retired names are kept as keys so an existing plugin's manifest
#: still loads; they point at where those modules actually went when the
#: built-ins moved.
_PLUGIN_SECTION_MAP = {
    "core": SECTION_CORE,
    "data": SECTION_DATA,
    "tools": SECTION_TOOLS,
    "toxo": SECTION_ASSAYS,
    "results": SECTION_DATA,
    "models": SECTION_DATA,
    "design": SECTION_DATA,
    "explore": SECTION_TOOLS,
}

#: Hard cap on apps per section. Enforced by tests, not at runtime — a
#: violation is a design mistake to fix in this table, not something to
#: discover at startup.
#:
#: Raised to 40 on 2026-09-05 at the maintainer's instruction. Data had
#: reached exactly twenty -- the previous ceiling -- the moment the three
#: self-registering modules joined the table, so the next registration in
#: that section would have tripped the cap rather than caught a real
#: mistake.
#:
#: The number is a smell detector, not a layout constraint: nothing breaks
#: at any size, and the sections are collapsible, so what it guards against
#: is a section nobody can scan. Forty keeps that guard while leaving the
#: room the registry has turned out to need.
MAX_APPS_PER_SECTION = 40

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
        """Equal to a tuple as well as to a list.

        `SECTIONS` was a tuple for its whole life and the suite asserts
        ``SECTIONS == ("Core", ...)``. Changing the container must not change
        what those assertions MEAN, so a tuple is compared by its contents rather
        than answering False on the type.
        """
        if isinstance(other, tuple):
            return list.__eq__(self, list(other))
        return list.__eq__(self, other)

    def __ne__(self, other):
        """The negation of :meth:`__eq__`, preserving ``NotImplemented``.

        Written out because Python does not derive it from ``__eq__`` for a class
        that defines one, and inheriting list's would compare against the tuple
        by type and disagree with ``==``.
        """
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



#: Apps that are REGISTERED but get no tile and no sidebar row.
#:
#: A tile says "start here". These are not things a user sets out to do:
#: they are things reached for WHILE doing something else, and a tile for
#: each put them at the same level as Mask and Regression. The section list records which of the two doors each one gets --
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
    "feature_dict",
    "run_history",
    "pipeline_graph",
    "project_browser",
    "investigate_hit",
    "profiler",
    "train_compare",
    "feature_explorer",
    "plate_view",
    "trellis",
    "lineage",
    "tabulate",
    "layer_viewer",
    "control_chart",
    "outliers",
    "convert",
    "external_masks",
    "report",
    "data_manager",
    "db_browser",
    "queue",
    "batch",
    "distributed_jobs",
})


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
        params = {}
    takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD
                    for p in params.values())
    for wanted, value in (("app_key", key), ("host", host)):
        if takes_any or wanted in params:
            kwargs[wanted] = value
    return factory(**kwargs)


_BUILTIN_APPS = [
    ("mask",           "Mask",           "Generate segmentation masks for cells, nuclei, pathogens and organelles from microscopy images using Cellpose and supported alternatives", SECTION_CORE),
    ("measure",        "Measure",        "Quantify per-object intensity and morphology features",       SECTION_CORE),
    ("annotate",       "Annotate",       "Assign annotations to single-object images and store them in the project database",  SECTION_CORE),
    ("classify_merged", "Classify",      "Train classifiers on image crops with PyTorch or on measured features with gradient boosting", SECTION_CORE),
    ("map_barcodes",   "Map Barcodes",   "Map sequencing barcodes to screen data",                      SECTION_CORE),
    ("regression",     "Regression",     "Regression analysis of screen scores",                        SECTION_CORE),
    ("convert",        "Format Converter", "Convert ND2, CZI, LIF and OME-TIFF images to Yokogawa TIFF layout and record source mappings", SECTION_DATA),
    ("foreign",        "Import",         "Bring images, masks and measurement tables into a spaCR project -- converting microscope formats, mapping source columns, or adopting masks made elsewhere", SECTION_DATA),
    ("external_masks", "External Masks", "Import images and externally generated label masks as a measured spaCR project ready for annotation", SECTION_DATA),
    ("queue",          "Plate Queue",    "Execute the same processing pipeline across multiple plates", SECTION_DATA),
    ("batch",          "Batch Runner",   "Queue modules, plates and settings for unattended sequential execution", SECTION_DATA),
    ("distributed_jobs", "Distributed Jobs", "Submit and monitor spaCR runs on SSH workstations, Slurm or cloud/HPC commands", SECTION_DATA),
    ("db_browser",     "Database Browser", "Browse, filter and export tables from measurements.db", SECTION_DATA),
    ("run_history",    "Run History",    "Search run settings, outputs, warnings, failures and performance metrics", SECTION_DATA),
    ("report",         "Report",         "Generate shareable HTML or PDF reports containing QC results, figures, statistics, settings and software versions", SECTION_DATA),
    ("train_compare",  "Training Runs",  "Compare training curves and settings across multiple runs", SECTION_TOOLS),
    ("align",          "Align & Stitch", "Register and stitch image tiles into an incrementally written mosaic with bounded memory use", SECTION_TOOLS),
    ("make_masks",     "Make Masks",     "Edit segmentation masks with brush, flood-fill, relabel, fill and small-object removal tools",  SECTION_TOOLS),
    ("plate_view",     "Plate Viewer",   "Visualize measurements as plate heatmaps and detect edge effects",  SECTION_TOOLS),
    ("umap",           "Image UMAP",     "Visualize UMAP embeddings with image glyphs",                  SECTION_TOOLS),
    ("analyze_plaques", "Plaque Assay",  "Quantify plaque assay measurements",                          SECTION_ASSAYS),
    ("recruitment",    "Recruitment",    "Quantify molecular recruitment measurements",                 SECTION_ASSAYS),
    ("invasion",       "Invasion Assay", "Quantify attached and invaded parasites using two-colour differential staining and calculate invasion efficiency per well", SECTION_ASSAYS),
    ("replication",    "Replication Assay", "Quantify parasites per vacuole and calculate replication rates by condition", SECTION_ASSAYS),
]



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
    "make_masks":      STAGE_BETA,
    "analyze_plaques": STAGE_BETA,
    "replication":     STAGE_BETA,
    "umap":            STAGE_BETA,
}

for _row in _BUILTIN_APPS:
    register_app(*_row)
del _row




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
    ("spacr.qt.screens.data_manager", "register"),
    ("spacr.qt.screens.pipeline_graph", "register"),
    ("spacr.qt.screens.profiler", "register"),
    ("spacr.qt.screens.qc_dashboard", "register"),
    ("spacr.qt.screens.lineage", "register"),
    ("spacr.qt.screens.experiment_design", "register"),
    ("spacr.qt.layer_viewer", "register_layer_viewer_app"),
    ("spacr.qt.screens.graph_builder", "register"),
    ("spacr.qt.screens.power", "register"),
    ("spacr.qt.screens.run_compare", "register"),
    ("spacr.qt.screens.tabulate", "register"),
    ("spacr.qt.screens.investigate_hit", "register"),
    ("spacr.qt.screens.dose_response", "register"),
    ("spacr.qt.screens.gate_editor", "register"),
    ("spacr.qt.screens.project_browser", "register"),
)

for _module_name, _func_name in _SELF_REGISTERING_APPS:
    try:
        if _declared_for(_module_name) is not None:
            _register_declared(_module_name)
        else:
            getattr(_importlib.import_module(_module_name), _func_name)()
    except Exception:
        LOG.exception("Could not register the app owned by %s", _module_name)
del _module_name, _func_name


try:
    from spacr.plugins import plugin_apps as _plugin_apps
    from spacr.plugins import record_diagnostic
    for _plugin_app in _plugin_apps():
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
    ("run_history", "Run History",
     "Search every recorded job -- its settings, hashed inputs and "
     "outputs, warnings, failures, versions and seeds."),
    ("pipeline_graph", "Pipeline Graph",
     "How the modules feed one another, and what each one needs before "
     "it can run."),
    ("project_browser", "Project Browser",
     "Every spaCR project this machine knows about, and what is in it."),
    ("db_browser", "Database Browser",
     "Browse, filter and export tables from measurements.db."),
    ("report", "Report",
     "Generate a shareable HTML or PDF report of QC results, figures, "
     "statistics, settings and software versions."),
    ("data_manager", "Data Manager",
     "Inspect project disk usage and remove derived data while keeping "
     "the source images."),
    ("queue", "Plate Queue",
     "Run the same processing pipeline across several plates."),
    ("batch", "Batch Runner",
     "Queue modules, plates and settings for unattended sequential "
     "execution."),
    ("distributed_jobs", "Distributed Jobs",
     "Submit and monitor spaCR runs on SSH workstations, Slurm, or "
     "cloud and HPC commands."),
)


def dock_rows() -> List[Tuple[str, str, str, str]]:
    """The dock's TOP-LEVEL rows: Home's tiles, then Help, in that order.

    ONE RULE: a row is a dock host if, and only if, Home draws a tile for
    it. Everything else in the registry reaches the dock as an indented
    child under its fold host, or under one of the Help modules, and
    nothing appears twice.

    THIS USED TO RETURN ALL OF :data:`APPS`, and the result was a dock that
    did not match the Home screen: modules belonging to Help sat outside it,
    and modules that should have been nested under a host were listed flat.
    The dock is meant to mimic the screen module
    tiles ... everything else is nested and in help dropdown."

    Measured before the change: 36 top-level dock rows against 19 Home
    tiles, and NINE keys drawn twice -- ``convert``, ``external_masks``,
    ``investigate_hit``, ``layer_viewer``, ``lineage``, ``plate_view``,
    ``profiler``, ``tabulate`` and ``train_compare`` each had a top-level
    row AND an indented row under the host they fold into. A module that
    appears in two places in the same column is a module the reader cannot
    learn the location of.

    :func:`tiled_apps` is the authority rather than a second hand-written
    list, so the dock cannot drift from Home again: promoting a module to a
    tile gives it a dock row, and folding one takes its dock row away, both
    without touching this function. That is the same lesson `_HELP_MODULES`
    already encodes below -- 330's note records one bug from a duplicated
    list and the retired-section map records another.

    WHY HELP IS NOT A REAL SECTION: see :data:`SECTION_HELP`. A section is
    Home's categorisation and every Help module is tileless, so Help would
    be a tab with nothing on it. The dock has no such constraint.

    The dock walks these rows in order and starts a new heading whenever
    the section changes, so grouping IS ordering here -- a row out of place
    draws its heading a second time.
    """
    helpers = {row[0] for row in _HELP_MODULES}
    tiles = {row[0] for row in tiled_apps()}
    rank = {section: index for index, section in enumerate(SECTION_ORDER)}
    rows = [
        (key, name, desc, SECTION_HELP if key in helpers else section)
        for key, name, desc, section in APPS
        if key in tiles or key in helpers
    ]
    rows.sort(key=lambda row: rank.get(row[3], len(rank)))
    return rows


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


#: The tile order, per section, exactly as specified.
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
#: Screen modules that host folded modules, beyond the ones
#: `fold_strip.FOLD_HOST_MODULES` already names. These three grew fold
#: strips later and were never added there.
#:
#: Import is no longer among them: Import Images is folded onto it and has
#: no registry row, so `fold_strip` has to walk that host itself to find
#: the name for its button. Naming it in both places would be the same
#: fact written twice, and the copy is the one that goes stale.
_EXTRA_FOLD_HOSTS: Tuple[str, ...] = (
    "spacr.qt.screens.graph_builder",
    "spacr.qt.screens.qc_dashboard",
    "spacr.qt.screens.db_browser",
)


def folded_children() -> Dict[str, Tuple[str, ...]]:
    """``host key -> the module keys folded onto its masthead``.

    ONE MAPPING, read from the hosts themselves, so the dock, the menu
    bar and the fold strips cannot disagree about what belongs where.
    Each host declares its own `FOLDED_APPS`; that tuple is the truth
    and this only collects them.

    The dock and the spaCR menu both draw the nested structure from this,
    showing a folded module one level below the host it belongs to.

    Never raises: a host whose module cannot be imported contributes
    nothing, because a navigation aid must not be able to stop the
    window being built.
    """
    try:
        from .widgets.fold_strip import FOLD_HOST_MODULES
    except Exception:                                    # noqa: BLE001
        FOLD_HOST_MODULES = ()

    found: Dict[str, Tuple[str, ...]] = {}
    for module_name in tuple(FOLD_HOST_MODULES) + _EXTRA_FOLD_HOSTS:
        declared = _declared_folds(module_name)
        if declared is None:
            LOG.debug("fold host %s unavailable", module_name)
            continue
        host, folded = declared
        if host and folded:
            found[str(host)] = tuple(str(k) for k in folded)
    return found


def _declared_constant(module_name: str, name: str):
    """One module-level string constant, read from source without importing.

    :param module_name: dotted name of the module to read.
    :param name: the constant to look for.
    :returns: the string, or ``None`` when it is absent or not a plain string.
    """
    import ast
    import importlib.util
    import pathlib

    try:
        spec = importlib.util.find_spec(module_name)
        tree = ast.parse(
            pathlib.Path(spec.origin).read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        return None
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        else:
            continue
        if not isinstance(value, ast.Constant) or not isinstance(
                value.value, str):
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name:
                return value.value
    return None


def _declared_folds(module_name: str):
    """``(host key, folded keys)`` read from a host's SOURCE, not by importing.

    WHY NOT `import_module`. This runs while the menu bar is being built, so
    importing every fold host pulls their whole dependency tree into the
    process before Home has painted -- `make_masks` alone reached pandas and
    scipy through `curation`, `mask_engine` and the settings model. The
    packaged smoke test asserts that Home crosses no operation-only import
    boundary, and it was failing on exactly that.

    The declarations themselves are simple: a string for the host key and a
    tuple of strings for the folds, with the occasional reference to another
    module-level string constant in the same file (`MASK_FOLDER_KEY`). Those
    are read out of the syntax tree, which costs a file read and no imports.

    :param module_name: dotted name of the host module.
    :returns: ``(host, folded)``, or ``None`` when the module cannot be read.
    """
    import ast
    import importlib.util
    import pathlib

    try:
        spec = importlib.util.find_spec(module_name)
        source = pathlib.Path(spec.origin).read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:                                    # noqa: BLE001
        return None

    #: Module-level names bound to a plain string, so a fold list may name one.
    constants: Dict[str, str] = {}
    declared: Dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        else:
            continue
        if value is None:
            continue
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                constants[target.id] = value.value
            if target.id in ("FOLDED_APPS", "FOLD_ORDER", "APP_KEY",
                             "HOST_KEY"):
                declared[target.id] = value

    #: `from . import activation` -> {"activation": "spacr.qt.screens.activation"},
    #: so `activation.APP_KEY` in a fold list can be resolved the same way.
    siblings: Dict[str, str] = {}
    package = module_name.rpartition(".")[0]
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level:
            base = package
            for _ in range(node.level - 1):
                base = base.rpartition(".")[0]
            if node.module:
                base = f"{base}.{node.module}"
            for alias in node.names:
                siblings[alias.asname or alias.name] = f"{base}.{alias.name}"

    def _as_string(node, _depth=0):
        """The string an AST node stands for, or ``None`` if it is not one.

        Resolves a literal, a module constant, and another module's constant
        read the same way rather than by importing it. ``_depth`` bounds
        that indirection so a constant defined in terms of itself stops
        instead of recursing.
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return constants.get(node.id)
        if isinstance(node, ast.Attribute) and _depth == 0:
            owner = node.value
            if isinstance(owner, ast.Name) and owner.id in siblings:
                other = _declared_constant(siblings[owner.id], node.attr)
                if other is not None:
                    return other
        return None

    host = _as_string(declared.get("APP_KEY")) or _as_string(
        declared.get("HOST_KEY"))
    folded: Tuple[str, ...] = ()
    for name in ("FOLDED_APPS", "FOLD_ORDER"):
        node = declared.get(name)
        if isinstance(node, (ast.Tuple, ast.List)):
            keys = [_as_string(element) for element in node.elts]
            if all(keys):
                folded = tuple(keys)
                break
    return host, folded


SECTION_TILE_ORDER: Dict[str, Tuple[str, ...]] = {
    SECTION_CORE: ("mask", "measure", "annotate", "classify_merged",
                   "map_barcodes", "regression"),
    SECTION_DATA: ("foreign", "embeddings", "run_compare",
                   "experiment_design", "power", "dose_response",
                   "qc_dashboard"),
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


_ICON_OVERRIDES = {
    "train_cellpose":  "cellpose_masks.png",
}

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
    if key in _FORCE_GLYPH:
        return iconset.icon(key)
    return iconset.app_icon(key, override=_ICON_OVERRIDES.get(key))


class Sidebar(Dock):
    """The application's navigation dock.

    The behaviour lives in :class:`spacr.qt.widgets.dock.Dock`, rewritten on
    2026-09-03 after four rounds of fixes to the old one had not settled it.
    This is only the binding to the registry, and it keeps the old NAME
    because the theme's ``#Sidebar`` rules, the tutorial highlighter, the
    home-variant generators and the tests all reach the dock by it.

    What used to be here was 1,095 lines of ``_DockRow`` and ``Sidebar``:
    a translucent slab painted in ``paintEvent`` (the "black box" four
    commits chased), a per-row icon-size model that relaid the column out
    under the pointer, a name drawn only while hovered, and a folded second
    level with its own expand state. See the module docstring of
    :mod:`spacr.qt.widgets.dock` for what replaced each one.

    :param parent: parent widget; ownership only.
    """

    #: Declared so existing callers may still connect to it. IT NEVER FIRES:
    #: the folded second level was removed, so there are no child rows to
    #: select. Kept rather than deleted because a signal that vanished would
    #: fail at the connect site, away from the reason.
    fold_child_selected = Signal(str)

    #: Home, above the headings and outside every category — it is how you
    #: get back, so it must not live inside a section you can shut.
    HOME_ROW = ("__home__", "Home", "Back to the start page", "")

    @staticmethod
    def _dock_icon(key: str):
        """The row's mark. ``app_icon``, NOT ``icon``, for Home.

        Home is the one row whose mark would otherwise differ from what the
        Home screen draws, because ``icon("home")`` answers a Font Awesome
        house. The bundled mask is cut from the application's own tile, and
        ``app_icon`` re-inks it for the theme like every other row.
        """
        if key == "__home__":
            return iconset.app_icon("home")
        return _icon_for_app(key)

    def __init__(self, parent=None):
        """Build the dock with Home above the registered module rows.

        :param parent: parent widget, or ``None``.
        """
        super().__init__([self.HOME_ROW] + list(dock_rows()),
                         icon_for=self._dock_icon,
                         is_visible=app_is_visible, parent=parent)
        self.setObjectName("Sidebar")


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
        """Build one window-chrome button around a painted mark.

        :param parent: parent widget; ownership only.
        :param painter: called with a colour and returning the icon for it.
            A CALLABLE, not an icon, because the hover state is a second
            painting of the same glyph -- QSS can colour a background but
            not the contents of a QIcon.
        :param colour: what the mark turns on hover and on press.
        """
        super().__init__(parent)
        self._paint_icon = painter
        self._hover_colour = colour
        self.setAutoRaise(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._show(False)

    def _show(self, lit: bool) -> None:
        """Repaint the mark, lit or resting.

        The icon is REPAINTED rather than recoloured: QSS can colour a background
        on hover but not the contents of a QIcon, which is why this button paints
        its glyph instead of shipping one.
        """
        self.setIcon(self._paint_icon(colour=self._hover_colour)
                     if lit else self._paint_icon())

    def enterEvent(self, event):        # noqa: N802 - Qt naming
        """Show the mark when the pointer arrives.

        :param event: the enter event.
        """
        self._show(True)
        super().enterEvent(event)

    def leaveEvent(self, event):        # noqa: N802 - Qt naming
        """Hide the mark when the pointer leaves, unless the button is held down.

        :param event: the leave event.
        """
        self._show(self.isDown())
        super().leaveEvent(event)

    def mousePressEvent(self, event):   # noqa: N802 - Qt naming
        """Show the mark while the button is held.

        :param event: the mouse event.
        """
        self._show(True)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt naming
        """Leave the mark shown only if the pointer is still over the button.

        Asked after the base class has handled the release, so ``underMouse``
        reports where the pointer ended rather than where the press began.

        :param event: the mouse event.
        """
        super().mouseReleaseEvent(event)
        self._show(self.underMouse())


def _current_font_scale() -> float:
    """The interface scale in force, or 1.0 if it cannot be read.

    Defensive because the callers are cache bookkeeping on paths that run
    during teardown: a scale that cannot be read must not stop a screen being
    cached, and 1.0 makes the screen look stale exactly once.

    :returns: the persisted interface scale.
    """
    try:
        from .preferences import get_font_scale

        return float(get_font_scale())
    except Exception:                                        # noqa: BLE001
        return 1.0



#: What a venv built by ``uv venv`` says when anything reaches for pip.
#: Lower-cased because the wording differs between interpreters -- "No
#: module named pip" and "No module named `pip`" both appear -- and the
#: three words that matter are the same in every one.
_NO_PIP_IN_THE_OUTPUT = "no module named pip"


def _the_missing_pip_escape(output: str) -> Optional[str]:
    """The command that WOULD work, when the upgrade failed for want of pip.

    :param output: everything the upgrade wrote, as the worker captured it.
    :returns: a command line to show the user, or None when this failure
        was not the missing-pip one.

    THE ONE FAILURE THE APPLICATION CAN ANSWER. The desktop installers
    build their environment with ``uv venv``, which does not seed pip, so
    an install whose updater still runs ``python -m pip`` fails before it
    starts -- and it is a BOOTSTRAP TRAP: the fix cannot arrive by the
    route it fixes. A dialog that names the exact command is the only
    escape that reaches a user who has already hit it.

    Every other exit code gets the output and nothing else. Guessing at a
    remedy for a failure this cannot recognise would send a user to run a
    command that is not their problem.

    Derived the way :func:`spacr.updater.find_uv` derives it, and it falls
    back to the path the installers WRITE rather than returning nothing:
    on the affected machine `find_uv` is what the installed build is too
    old to have, so a name the user can check is worth more than silence.
    """
    if _NO_PIP_IN_THE_OUTPUT not in (output or "").lower():
        return None
    uv = None
    try:
        from spacr.updater import find_uv

        uv = find_uv()
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not look for the bootstrapped uv", exc_info=True)
    if not uv:
        from pathlib import Path

        uv = str(Path(sys.prefix).parent / "bootstrap"
                 / ("uv.exe" if os.name == "nt" else "uv"))
    parts = [uv, "pip", "install", "--upgrade", "--python", sys.executable,
             "spacr"]
    if os.name == "nt":
        import subprocess

        return subprocess.list2cmdline(parts)
    import shlex

    return shlex.join(parts)


#: The in-session paint diagnostic (item 408) is armed only when the process
#: was LAUNCHED with this set to ``1``. Nothing is bound and nothing is shown
#: otherwise.
_PAINT_DIAG_ENV = "SPACR_PAINT_DIAG"
_PAINT_DIAG_KEYS = "Ctrl+Alt+Shift+D"
#: The black the eye sees, not only exact (0, 0, 0) -- 408's second
#: measurement trap: max(r, g, b) at or below this counts as black.
_NEAR_BLACK_MAX = 16
#: The documented top level of the JSON, in the order it is written.
_PAINT_DIAG_KEYS_IN_ORDER = (
    "suspects", "captured_utc", "files", "qt_platform", "spacr_version",
    "spacr_path", "git_head", "git_dirty", "preferences", "environment",
    "window_backdrop", "screen", "stylesheets", "screenshot",
    "top_level_windows", "widgets", "errors",
)


def _install_the_paint_diagnostic(window):
    """Bind the paint-diagnostic key when the launch asked for it.

    A viewport can paint opaque black in a real session while no offscreen
    probe reproduces it, so the widget state behind such a box has to be
    RECORDED where it happens rather than reconstructed.

    :returns: the application-wide ``QShortcut``, or ``None`` when
        ``SPACR_PAINT_DIAG`` was not ``1`` or the key could not be bound.
    """
    if os.environ.get(_PAINT_DIAG_ENV, "").strip() != "1":
        return None
    try:
        from PySide6.QtGui import QShortcut

        shortcut = QShortcut(QKeySequence(_PAINT_DIAG_KEYS), window)
        shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        shortcut.activated.connect(lambda: _dump_paint_diagnostics(window))
        LOG.info("paint diagnostic armed: %s writes to %s",
                 _PAINT_DIAG_KEYS, _paint_diagnostics_folder())
        return shortcut
    except Exception:                                        # noqa: BLE001
        LOG.exception("could not bind the paint diagnostic")
        return None


def _paint_diagnostics_folder():
    """Where a dump goes when the caller does not say."""
    from pathlib import Path

    return Path.home() / ".cache" / "spacr" / "paint_diagnostics"


def _plain(value):
    """``value`` as something ``json`` writes without a fallback."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _sheet_fingerprint(text) -> dict:
    """Length and sha1 of a style sheet, so two sheets compare without
    writing either one into the dump."""
    import hashlib

    from .theme import TRANSPARENT_PROPERTY

    text = text or ""
    return {"length": len(text),
            "sha1": hashlib.sha1(text.encode("utf-8", "replace")).hexdigest(),
            "carries_transparent_rule": TRANSPARENT_PROPERTY in text}


def _colour_record(colour) -> dict:
    """A ``QColor`` as its name and its alpha -- alpha is half of 408."""
    return {"name": colour.name(), "alpha": colour.alpha()}


def _rect_in_window(widget, window) -> list:
    """``[x, y, width, height]`` of ``widget`` in ``window`` coordinates."""
    from PySide6.QtCore import QPoint

    if widget is window:
        origin = QPoint(0, 0)
    elif window.isAncestorOf(widget):
        origin = widget.mapTo(window, QPoint(0, 0))
    else:
        origin = (widget.mapToGlobal(QPoint(0, 0))
                  - window.mapToGlobal(QPoint(0, 0)))
    return [origin.x(), origin.y(), widget.width(), widget.height()]


def _widget_path(widget, stop) -> str:
    """``Class#name > ... > Class#name`` from ``stop`` down to ``widget``."""
    parts = []
    node = widget
    while node is not None:
        name = node.objectName()
        parts.append(type(node).__name__ + (f"#{name}" if name else ""))
        if node is stop:
            break
        node = node.parentWidget()
    return " > ".join(reversed(parts))


def _nearest_sheet_ancestor(widget):
    """The closest ancestor carrying a non-empty style sheet, or ``None``."""
    parent = widget.parentWidget()
    while parent is not None:
        sheet = parent.styleSheet()
        if sheet:
            record = {"class": type(parent).__name__,
                      "objectName": parent.objectName()}
            record.update(_sheet_fingerprint(sheet))
            return record
        parent = parent.parentWidget()
    return None


def _the_transparent_rule_is_on_the_ancestry(widget) -> bool:
    """Whether any sheet from ``widget`` up, or the application's, has the rule.

    Presence only: a more specific rule nearer the widget can still win.
    """
    from .theme import TRANSPARENT_PROPERTY

    node = widget
    while node is not None:
        if TRANSPARENT_PROPERTY in node.styleSheet():
            return True
        node = node.parentWidget()
    app = QApplication.instance()
    return bool(app is not None and TRANSPARENT_PROPERTY in app.styleSheet())


def _paint_state_of(widget, window) -> dict:
    """Everything that decides whether ``widget`` paints its own background."""
    from .theme import SURFACE_PROPERTY, TRANSPARENT_PROPERTY

    attribute = Qt.WidgetAttribute
    palette = widget.palette()
    active = QPalette.ColorGroup.Active
    return {
        "class": type(widget).__name__,
        "objectName": widget.objectName(),
        "geometry": _rect_in_window(widget, window),
        TRANSPARENT_PROPERTY: _plain(widget.property(TRANSPARENT_PROPERTY)),
        SURFACE_PROPERTY: _plain(widget.property(SURFACE_PROPERTY)),
        "autoFillBackground": bool(widget.autoFillBackground()),
        "WA_TranslucentBackground": bool(
            widget.testAttribute(attribute.WA_TranslucentBackground)),
        "WA_OpaquePaintEvent": bool(
            widget.testAttribute(attribute.WA_OpaquePaintEvent)),
        "WA_NoSystemBackground": bool(
            widget.testAttribute(attribute.WA_NoSystemBackground)),
        "styleSheet": _sheet_fingerprint(widget.styleSheet()),
        "palette": {
            "Base": _colour_record(
                palette.color(active, QPalette.ColorRole.Base)),
            "Window": _colour_record(
                palette.color(active, QPalette.ColorRole.Window)),
        },
        "nearest_sheet_ancestor": _nearest_sheet_ancestor(widget),
        "transparent_rule_on_ancestry":
            _the_transparent_rule_is_on_the_ancestry(widget),
    }


def _pixels_of(image):
    """An ``(h, w, 4)`` RGBA ``numpy`` copy of ``image``, or ``None``."""
    import numpy as np
    from PySide6.QtGui import QImage

    image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    height, width = image.height(), image.width()
    if not width or not height:
        return None
    flat = np.frombuffer(image.constBits(), dtype=np.uint8)
    rows = flat[:height * image.bytesPerLine()].reshape(
        height, image.bytesPerLine())
    return rows[:, :width * 4].reshape(height, width, 4).copy()


def _near_black_fraction(pixels, rect, scale) -> Optional[float]:
    """Share of ``rect`` (window coordinates) that is near-black on screen."""
    if pixels is None:
        return None
    import math

    height, width = pixels.shape[:2]
    x, y, w, h = rect
    x0 = max(0, int(math.floor(x * scale[0])))
    y0 = max(0, int(math.floor(y * scale[1])))
    x1 = min(width, int(math.ceil((x + w) * scale[0])))
    y1 = min(height, int(math.ceil((y + h) * scale[1])))
    if x1 <= x0 or y1 <= y0:
        return None
    region = pixels[y0:y1, x0:x1, :3]
    return round(float((region.max(axis=2) <= _NEAR_BLACK_MAX).mean()), 4)


def _paint_suspects(records, rule_in_play) -> list:
    """The records worth reading first, each with the reasons it is here.

    SCROLL AREAS: a visible one whose viewport is untagged or fills its own
    background -- the shape of both 408 widgets. They sort first.

    SHEETS, NARROWED BY MEASUREMENT on a HEALTHY themed Mask. Comparing each
    widget's own or nearest-ancestor sheet digest with the screen's flagged
    50 and 176 of 184 visible widgets, and "own sheet carries the rule and
    differs from the screen's" still flagged 30-33: the settings splitter,
    the form editors and the toggles carry theme-derived sheets of their own,
    and Qt merges every ancestor's sheet, so a sheet that differs is normal.
    The one sheet state flagged is the one that stops the rule outright and
    flagged nothing healthy: a tagged widget with no sheet on its ancestry,
    nor the application's, carrying the rule -- checked only when the rule
    is in play at all. Every digest is still in ``widgets``.
    """
    from .theme import TRANSPARENT_PROPERTY

    suspects = []
    for record in records:
        reasons = []
        viewport = record.get("viewport")
        if viewport is not None:
            if not viewport.get(TRANSPARENT_PROPERTY):
                reasons.append("viewport untagged")
            if viewport.get("autoFillBackground"):
                reasons.append("viewport autoFillBackground")
        if (rule_in_play and record.get(TRANSPARENT_PROPERTY)
                and not record.get("transparent_rule_on_ancestry")):
            reasons.append("tagged, but no sheet on its ancestry carries "
                           "the transparent rule")
        if reasons:
            suspects.append({
                "class": record["class"],
                "objectName": record["objectName"],
                "path": record.get("path"),
                "geometry": record["geometry"],
                "reasons": reasons,
            })
    suspects.sort(key=lambda s: not s["reasons"][0].startswith("viewport"))
    return suspects


def _git_state(where: str):
    """``(HEAD, dirty)`` of the checkout at ``where``, ``(None, None)`` if none."""
    import subprocess

    head = subprocess.run(["git", "-C", where, "rev-parse", "HEAD"],
                          capture_output=True, text=True, timeout=5)
    if head.returncode != 0 or not head.stdout.strip():
        return None, None
    status = subprocess.run(
        ["git", "-C", where, "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, timeout=15)
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return head.stdout.strip(), dirty


def _dump_paint_diagnostics(window, _out_dir=None) -> dict:
    """Write the paint state of the screen on show, and what is ON screen.

    Two files named by the UTC time: ``<stamp>.png``, the window as the
    display holds it (``QScreen.grabWindow`` on the window id -- a
    ``QWidget.grab`` re-renders the tree and can never show a paint fault),
    and ``<stamp>.json``, whose top-level keys are
    :data:`_PAINT_DIAG_KEYS_IN_ORDER`. Both paths go to the console and the
    log.

    :param window: the ``MainWindow``.
    :param _out_dir: where to write; ``~/.cache/spacr/paint_diagnostics``
        when ``None``.
    :returns: the report as written. Never raises: a part that fails is
        logged, named in ``errors``, and the rest is still written.
    """
    report = dict.fromkeys(_PAINT_DIAG_KEYS_IN_ORDER)
    report["suspects"] = []
    report["errors"] = []
    report["files"] = {"json": None, "png": None}
    try:
        _collect_paint_diagnostics(window, _out_dir, report)
    except Exception as exc:                                 # noqa: BLE001
        LOG.exception("the paint diagnostic stopped early")
        report["errors"].append(f"stopped early: {exc!r}")
    return report


def _collect_paint_diagnostics(window, out_dir, report) -> None:
    """Fill ``report`` and write it; see :func:`_dump_paint_diagnostics`."""
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    errors = report["errors"]

    def failed(what):
        """Log the exception being handled and name it in ``errors``."""
        LOG.warning("paint diagnostic: could not %s", what, exc_info=True)
        errors.append(f"{what}: {sys.exc_info()[1]!r}")

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S") + f".{now.microsecond // 1000:03d}Z"
    folder = Path(out_dir) if out_dir is not None \
        else _paint_diagnostics_folder()
    json_path = folder / f"{stamp}.json"
    png_path = folder / f"{stamp}.png"
    report["captured_utc"] = now.isoformat()
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except Exception:                                        # noqa: BLE001
        failed(f"create {folder}")

    # FIRST, before anything below can change a pixel.
    pixels, scale = None, (1.0, 1.0)
    try:
        app = QApplication.instance()
        display = window.screen() or app.primaryScreen()
        pixmap = display.grabWindow(window.winId())
        image = pixmap.toImage()
        report["screenshot"] = {
            "method": "QScreen.grabWindow(window.winId())",
            "display": display.name(),
            "size": [image.width(), image.height()],
            "devicePixelRatio": pixmap.devicePixelRatio(),
            "window_size": [window.width(), window.height()],
        }
        if image.isNull():
            raise RuntimeError("the grab came back empty")
        if pixmap.save(str(png_path), "PNG"):
            report["files"]["png"] = str(png_path)
        else:
            errors.append(f"save {png_path}: QPixmap.save returned False")
        pixels = _pixels_of(image)
        scale = (image.width() / max(1, window.width()),
                 image.height() / max(1, window.height()))
    except Exception:                                        # noqa: BLE001
        failed("grab the window from the display")

    try:
        report["qt_platform"] = QApplication.platformName()
    except Exception:                                        # noqa: BLE001
        failed("read the Qt platform")
    try:
        import spacr

        report["spacr_version"] = getattr(spacr, "__version__", None)
        where = Path(spacr.__file__).resolve().parent
        report["spacr_path"] = str(where)
        report["git_head"], report["git_dirty"] = _git_state(str(where.parent))
    except Exception:                                        # noqa: BLE001
        failed("read the spaCR version and checkout")

    from . import preferences

    wanted = {}
    for name in ("ambient_enabled", "ambient_animation", "ambient_theme",
                 "ambient_palette", "pane_opacity", "theme",
                 "tooltips_box_enabled", "tooltips_bottom_enabled",
                 "object_grid_enabled"):
        try:
            wanted[name] = _plain(getattr(preferences, f"get_{name}")())
        except Exception:                                    # noqa: BLE001
            failed(f"read the {name} preference")
    report["preferences"] = wanted
    report["environment"] = {name: os.environ.get(name)
                             for name in ("SPACR_NO_BACKDROP", "SPACR_NO_GL")}

    try:
        backdrop = window.window_backdrop()
        state = {"present": backdrop is not None}
        if backdrop is not None:
            running = getattr(backdrop, "is_running", None)
            state.update({
                "class": type(backdrop).__name__,
                "visible": bool(backdrop.isVisible()),
                "running": bool(running()) if callable(running) else None,
                "geometry": _rect_in_window(backdrop, window),
            })
        report["window_backdrop"] = state
    except Exception:                                        # noqa: BLE001
        failed("read the window backdrop")

    screen = None
    try:
        stack = getattr(window, "_stack", None)
        screen = stack.currentWidget() if stack is not None else None
        state = {"present": screen is not None}
        if screen is not None:
            own = getattr(screen, "_ambient", None)
            state.update({
                "class": type(screen).__name__,
                "objectName": screen.objectName(),
                "app_key": _plain(getattr(screen, "app_key", None)),
                "geometry": _rect_in_window(screen, window),
                "_ambient": None if own is None else {
                    "class": type(own).__name__,
                    "visible": bool(own.isVisible()),
                    "running": bool(own.is_running())
                    if callable(getattr(own, "is_running", None)) else None,
                },
                "_uses_window_backdrop": _plain(
                    getattr(screen, "_uses_window_backdrop", None)),
                "page_fill": None,
            })
        report["screen"] = state
        if screen is not None and callable(getattr(screen, "page_fill", None)):
            fill = screen.page_fill()
            state["page_fill"] = (None if fill is None
                                  else _colour_record(QColor(fill)))
    except Exception:                                        # noqa: BLE001
        failed("read the current screen")

    try:
        app = QApplication.instance()
        report["stylesheets"] = {
            "window": _sheet_fingerprint(window.styleSheet()),
            "screen": (_sheet_fingerprint(screen.styleSheet())
                       if screen is not None else None),
            "application": _sheet_fingerprint(app.styleSheet()),
        }
    except Exception:                                        # noqa: BLE001
        failed("fingerprint the style sheets")

    try:
        report["top_level_windows"] = [
            {"class": type(top).__name__, "objectName": top.objectName(),
             "windowType": getattr(top.windowType(), "name",
                                   str(top.windowType())),
             "geometry": [top.x(), top.y(), top.width(), top.height()]}
            for top in QApplication.topLevelWidgets() if top.isVisible()]
    except Exception:                                        # noqa: BLE001
        failed("list the top-level windows")

    records = []
    if screen is not None:
        from PySide6.QtWidgets import QAbstractScrollArea

        for widget in screen.findChildren(QWidget):
            try:
                if not widget.isVisible():
                    continue
                record = _paint_state_of(widget, window)
                record["path"] = _widget_path(widget, screen)
                if isinstance(widget, QAbstractScrollArea):
                    viewport = widget.viewport()
                    record["viewport"] = (None if viewport is None
                                          else _paint_state_of(viewport,
                                                               window))
                records.append(record)
            except Exception:                                # noqa: BLE001
                failed(f"read the paint state of a {type(widget).__name__}")
    report["widgets"] = records

    try:
        sheets = report.get("stylesheets") or {}
        rule_in_play = any((sheet or {}).get("carries_transparent_rule")
                           for sheet in sheets.values())
        suspects = _paint_suspects(records, rule_in_play)
        for suspect in suspects:
            suspect["near_black_fraction"] = _near_black_fraction(
                pixels, suspect["geometry"], scale)
        report["suspects"] = suspects
    except Exception:                                        # noqa: BLE001
        failed("pick the suspects")

    try:
        json_path.write_text(json.dumps(report, indent=2, default=str),
                             encoding="utf-8")
        report["files"]["json"] = str(json_path)
        # Rewritten so the file names itself; the first write is the one
        # that proves the folder takes a file at all.
        json_path.write_text(json.dumps(report, indent=2, default=str),
                             encoding="utf-8")
    except Exception:                                        # noqa: BLE001
        failed(f"write {json_path}")

    _say_where_the_paint_diagnostic_went(window, screen, report["files"])


def _say_where_the_paint_diagnostic_went(window, screen, files) -> None:
    """Put both paths on the console on show and in the log."""
    line = (f"Paint diagnostic: {files.get('json')}\n"
            f"On-screen capture: {files.get('png')}\n")
    LOG.info("paint diagnostic written: json=%s png=%s",
             files.get("json"), files.get("png"))
    try:
        from .widgets.console_panel import ConsolePanel

        console = getattr(screen, "_console", None)
        if console is None or not console.isVisible():
            console = next((panel for panel in window.findChildren(ConsolePanel)
                            if panel.isVisible()), console)
        if console is not None:
            console.append_stdout(line)
    except Exception:                                        # noqa: BLE001
        LOG.warning("paint diagnostic: could not write to the console",
                    exc_info=True)


class MainWindow(QMainWindow):
    """Top-level window: sidebar + stacked screens + status bar.

    :param initial_app: optional app key to navigate to on show; when
        omitted the window opens on the Home startup page.
    """

    def __init__(self, initial_app: Optional[str] = None):
        """Build the main window: the dock slot, the screen stack and the status bar.

        The window is frameless and the menu bar is what you drag it by. Its
        first backing store is filled with the splash colour so the compositor
        cannot expose stale desktop pixels before anything has drawn --
        deliberately without ``WA_OpaquePaintEvent``, which is a promise to
        paint every pixel that this window cannot keep once the application
        stylesheet clears ``autoFillBackground`` again. Claiming it left
        whatever was already on screen underneath, and transparent children drew
        on top: overlapping text in the corner, and flicker on every text
        surface over the animated backdrop.

        The dock slot exists in every dock mode, because a ``QMainWindow``'s
        central widget cannot be swapped without re-parenting the stack, and
        re-parenting a stack holding live screens is how locking the dock would
        cost you the screen you were looking at.

        :param initial_app: the module to open on startup; ``None`` opens Home.
        """
        super().__init__()
        self._closing = False
        from .widgets.loading_screen import splash_role
        startup_palette = self.palette()
        startup_palette.setColor(
            QPalette.ColorRole.Window,
            QColor(splash_role("splash_bg", "#000000")),
        )
        self.setPalette(startup_palette)
        self.setAutoFillBackground(True)
        self.setWindowTitle("spaCR")
        self.setMinimumSize(1200, 720)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)

        self._build_menu_bar()
        self._install_fullscreen_button()

        self._stack = QStackedWidget()
        central = QWidget()
        central.setObjectName("CentralRow")
        row = QHBoxLayout(central)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self._dock_slot = QWidget()
        self._dock_slot.setObjectName("DockSlot")
        self._dock_slot.setStyleSheet(
            "QWidget#DockSlot { background: transparent; border: none; }")
        #: The dock column's own backdrop, or ``None``. See
        #: :meth:`_backdrop_the_dock_column`.
        self._dock_backdrop = None
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
        self._sidebar.fold_child_selected.connect(self.open_module)
        self._sidebar.module_hovered.connect(self._show_module_hint)

        from .widgets.drawer import EdgeDrawer
        self._app_drawer = EdgeDrawer(self._stack, self._sidebar,
                                      width=self._sidebar.width())
        self.apply_dock_mode()
        self._backdrop_the_dock_column()

        self._screens: dict[str, QWidget] = {}
        #: The interface scale each cached screen was BUILT at, by key.
        #:
        #: A screen is built once and then cached forever, and everything
        #: `scaled_px` pins inside it -- icon sizes, row heights, tile
        #: geometry -- is read at construction and never read again. So a
        #: screen opened before the scale changed keeps the old sizes for the
        #: rest of the session while a screen opened after it gets the new
        #: ones, and the two sit side by side in the stack at different sizes.
        #: Reported as icons that do not track the slider perfectly; the
        #: Preferences slider had the same half-updated result long before the
        #: hold-Z gesture made it obvious. See :meth:`_rebuild_screens_for_scale`.
        self._screen_scales: dict[str, float] = {}
        #: App keys in the order the user last LOOKED at them, oldest
        #: first. Not the same thing as ``_screens`` order, which is
        #: creation order and never changes when an app is revisited —
        #: see :meth:`_snapshot_current_screen_settings`.
        self._visit_order: list[str] = []
        self._install_startup_page()

        status = QStatusBar()
        self._status_app_label = QLabel("Home")
        self._status_app_label.setObjectName("Muted")
        self._status_version_label = QLabel(f"spaCR {self._resolve_version()}")
        self._status_version_label.setObjectName("Caption")
        status.addPermanentWidget(self._status_app_label)
        status.addPermanentWidget(self._status_version_label)
        status.showMessage(tr("Ready"))
        status.setSizeGripEnabled(False)
        status.setFixedHeight(status.sizeHint().height())
        self.setStatusBar(status)

        try:
            from .module_hints import install_module_hints

            self._module_hints = install_module_hints(self)
        except Exception:                                    # noqa: BLE001
            LOG.debug("module hints unavailable", exc_info=True)
            self._module_hints = None


        from PySide6.QtCore import QTimer
        self._loading_screen = self._install_loading_screen()
        self._preloader = _PipelinePreloader(
            on_step=self._on_preload_step, on_done=self._on_preload_done)

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
            QTimer.singleShot(1500, self._preloader.start)
        else:
            self._loading_screen.set_total(self._preloader.total())
            QTimer.singleShot(0, self._preloader.start)

        try:
            from . import shortcuts
            shortcuts.install(self)
        except Exception:
            pass
        #: Item 408's in-session paint diagnostic; ``None`` unless the
        #: process was launched with ``SPACR_PAINT_DIAG=1``.
        self._paint_diagnostic_shortcut = _install_the_paint_diagnostic(self)

        if initial_app:
            self.open_module(initial_app)
        else:
            self.resume_after_restart()

        try:
            from .theme import take_the_scroll_arrows_off
            take_the_scroll_arrows_off(self)
        except Exception:                                # noqa: BLE001
            LOG.debug("could not take the tab scroll arrows off",
                      exc_info=True)

        self.refresh_language()

        try:
            from PySide6.QtCore import QTimer

            from .first_run import maybe_show_tour
            from .install_consent import maybe_show_installer_consent
            from .preferences import in_safe_mode
            self._tour_timer = QTimer(self)
            self._tour_timer.setSingleShot(True)
            self._tour_timer.timeout.connect(
                lambda: maybe_show_tour(self))
            self._consent_timer = QTimer(self)
            self._consent_timer.setSingleShot(True)

            def _finish_installer_onboarding():
                """Show the installer consent, then start the tour after it closes."""
                maybe_show_installer_consent(self)
                self._tour_timer.start(500)

            self._consent_timer.timeout.connect(_finish_installer_onboarding)
            if not in_safe_mode():
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
            screen.repaint()
        except RuntimeError:
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
        from PySide6.QtWidgets import QHBoxLayout, QWidget

        corner = QWidget(self)
        corner.setObjectName("WindowChrome")
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
        close.clicked.connect(self.close)
        row.addWidget(close)
        self._close_button = close

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

        self._drag_from = None
        self.menuBar().installEventFilter(self)

        try:
            from .widgets.glass import let_the_user_resize

            let_the_user_resize(self)
        except Exception:                                    # noqa: BLE001
            LOG.debug("the window could not be made resizable", exc_info=True)

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

    def _build_menu_bar(self):
        """Build the application menu bar.

        The native macOS bar is deliberately turned off: it draws no corner
        widget, which is where this frameless window's window marks live.
        """
        mb = self.menuBar()
        if sys.platform == "darwin":
            mb.setNativeMenuBar(False)
        self._menu_drag = _DragsTheWindowByTheMenuBar(self)
        mb.installEventFilter(self._menu_drag)

        app_menu = mb.addMenu("&spaCR")

        act_home = QAction("Home", self)
        act_home.setShortcut(QKeySequence("Ctrl+H"))
        act_home.triggered.connect(lambda: self._on_nav_selected("__home__"))
        app_menu.addAction(act_home)
        act_prefs = QAction("Preferences…", self)
        act_prefs.setShortcut(QKeySequence("Ctrl+P"))
        act_prefs.triggered.connect(self._open_preferences)
        app_menu.addAction(act_prefs)
        act_quit = QAction("Quit", self)
        act_quit.setShortcut(QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        app_menu.addAction(act_quit)
        self._act_quit = act_quit
        app_menu.addSeparator()

        self._app_actions: dict[str, QAction] = {}
        self._section_menus: dict[str, QMenu] = {}
        from .widgets.fold_strip import folded_modules

        folded = folded_children()
        catalogue = folded_modules()
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
                act.setProperty("moduleAppKey", key)
                act.setProperty("moduleNameSource", name)
                act.setProperty("moduleSummarySource", desc)
                act.triggered.connect(
                    lambda checked=False, k=key: self._on_nav_selected(k))
                kids = folded.get(key, ())
                if not kids:
                    submenu.addAction(act)
                else:
                    host_menu = QMenu(name, self)
                    host_menu.setProperty("moduleAppKey", key)
                    host_menu.setProperty("moduleNameSource", name)
                    submenu.addMenu(host_menu)
                    host_menu.addAction(act)
                    host_menu.addSeparator()
                    for child in kids:
                        entry = catalogue.get(child)
                        child_name = entry[0] if entry else child
                        child_desc = entry[1] if entry else ""
                        sub_act = QAction(str(child_name), self)
                        if child_desc:
                            sub_act.setStatusTip(str(child_desc))
                        sub_act.setProperty("moduleAppKey", child)
                        sub_act.setProperty("moduleNameSource", str(child_name))
                        sub_act.setProperty("moduleSummarySource", str(child_desc))
                        sub_act.triggered.connect(
                            lambda checked=False, k=child: self.open_module(k))
                        host_menu.addAction(sub_act)
                        self._app_actions[child] = sub_act
                self._app_actions[key] = act
        self._refresh_app_action_visibility()

        act_all = QAction("All apps", self)
        act_all.setShortcut(QKeySequence("Ctrl+Shift+A"))
        act_all.setStatusTip(
            "Show the full app list. Also revealed by moving the pointer "
            "to the left edge of the window.")
        act_all.triggered.connect(self.toggle_app_drawer)
        self.addAction(act_all)

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

        act_saver = QAction("Full-screen background", self)
        act_saver.setObjectName("ShowScreensaver")
        act_saver.setShortcut(QKeySequence("Ctrl+Shift+F"))
        act_saver.setStatusTip(
            "Show only the animated background, full screen, like a "
            "screensaver. Any key or click brings the window back.")
        act_saver.triggered.connect(self._show_the_screensaver)
        self.addAction(act_saver)
        self._act_screensaver = act_saver

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
        act_keys = QAction("Keyboard shortcuts", self)
        act_keys.setStatusTip(
            "Every key spaCR binds, what it does, and where it works.")
        act_keys.triggered.connect(self._show_shortcuts)
        help_menu.addAction(act_keys)

        act_setup = QAction("Set spaCR up again…", self)
        act_setup.setStatusTip(
            "The first-run questions, with the explanation of each -- "
            "language, theme, how it runs, the assistant, and what may "
            "leave this machine.")
        act_setup.triggered.connect(self._show_setup)
        help_menu.addAction(act_setup)
        help_menu.addSeparator()

        for key, label, tip in _HELP_MODULES:
            action = QAction(label, self)
            action.setStatusTip(tip)
            action.triggered.connect(
                lambda _checked=False, k=key: self._on_nav_selected(k))
            help_menu.addAction(action)
        help_menu.addSeparator()
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
            for action in list(menu.actions()) + [menu.menuAction()]:
                if action is None or action.isSeparator():
                    continue
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

    DEMO_TARGETS = {
        "mask":      ("mask",       "generate_mask_demo"),
        "measure":   ("measure",    "generate_measure_demo"),
        "crop":      ("measure",    "generate_crop_demo"),
        "classify":  ("annotate",   "generate_classify_demo"),
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
            """Warn if the demo download failed, else run the chain."""
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

        IT DOES NOT RUN ANYTHING. The user imports, and then chooses live
        preview or run themselves.

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

        from .settings_pack import settings_from_pack

        settings, report = settings_from_pack(
            "mask", settings_path, src=dataset_path)
        if report.source:
            LOG.info("settings pack for %s: %s", "mask", report.summary())
        else:
            LOG.warning("No settings pack found for mask in %s; using defaults.",
                        settings_path)
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
        if report.source:
            self.statusBar().showMessage(
                tr("Demo dataset loaded with its settings. Press Live Preview to "
                   "see one field, or Run to process the plate."), 12000)
        else:
            self.statusBar().showMessage(
                tr("Demo dataset loaded without a settings pack; using defaults. "
                   "Press Live Preview to see one field, or Run to process the "
                   "plate."), 12000)

    def _run_demo_generator(self, demo_key: str, dst: str):
        """Isolated for tests — invoke the named generator function
        with `dst` and return whatever it returned."""
        from spacr.qt import synthetic as syn
        _, gen_name = self.DEMO_TARGETS[demo_key]
        gen = getattr(syn, gen_name)
        return gen(dst)

    def _apply_demo_to_screen(self, widget, layout) -> None:
        """Push the demo layout into a target screen, in whatever way
        that screen supports (settings CSV, source folder, or DB path).

        A BARCODE REFERENCE THE DEMO LEFT EMPTY IS FILLED FROM THE PACKAGE, so
        loading test data never leaves Map Barcodes without its reference
        tables. The three CSVs ship inside the wheel, so this normally costs
        nothing; when package data has been stripped they are fetched from the
        release tag on GitHub and checked against a pinned hash.

        THE SETTINGS SAY WHETHER THIS APPLIES, rather than the app key. A
        map_barcodes settings file DECLARES `grna_csv`, `row_csv` and
        `column_csv`; a mask one does not, so it cannot accidentally gain
        three path settings that mean nothing to it. That also means the
        tutorial runner, which calls this method too, gets the same
        behaviour without being told about it.
        """
        from spacr.settings import (BUNDLED_BARCODE_SETTING,
                                    _fill_missing_barcode_references)
        from spacr.utils import load_settings

        if hasattr(widget, "apply_settings_dict") and layout.settings_csv:
            loaded = load_settings(
                str(layout.settings_csv),
                setting_key="Key", setting_value="Value",
            )
            if isinstance(loaded, dict):
                if any(key in loaded
                       for key in BUNDLED_BARCODE_SETTING.values()):
                    _fill_missing_barcode_references(loaded)
                widget.apply_settings_dict(loaded)
                return
        if hasattr(widget, "_open_source"):
            widget._open_source(str(layout.src))
            return
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
            """One centred line of the About box, at the given size."""
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
        self.show_preferences_on()

    def show_preferences_on(self, tab: str = "", label: str = "") -> bool:
        """Open Preferences, on a named tab, with a named row marked.

        The route the Help search field takes for a preference result: the
        dialog carries nine tabs and a result that opened it on whichever
        one it happened to start with has answered half the question.

        Found by OBJECT NAME rather than by tab index or caption. An index
        moves whenever a tab is added, and a caption is translated -- a
        Korean interface would have matched nothing. Every page sets its own
        ``PreferencesTab*`` name, and ``tools/build_help_search_index.py``
        reads those same names out of this dialog's source, so the two ends
        of the hand-off are the same string by construction.

        :param tab: the page's object name, e.g. ``"PreferencesTabTheme"``;
            ``""`` opens the dialog as it opens from the menu.
        :param label: the row caption to mark; ``""`` marks nothing.
        :returns: True when the named tab was found and shown.
        """
        try:
            from .preferences import PreferencesDialog
        except Exception as e:
            self.statusBar().showMessage(
                f"Preferences unavailable: {e}", 5000)
            return False
        dialog = PreferencesDialog(self)
        found = False
        if tab:
            try:
                from .preferences_navigation import show_tab

                found = show_tab(dialog, tab, label)
            except Exception:
                LOG.exception("could not open Preferences on %r", tab)
        dialog.exec()
        self.refresh_theme()
        return found

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
        try:
            self._backdrop_the_dock_column()
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
                self._drop_a_redundant_screen_backdrop(screen)
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
            from spacr.updater import run_pip_upgrade  # noqa: F401
            from spacr.install_cleanup import find_old_installs
        except Exception as exc:
            LOG.exception("Could not import the spaCR upgrade helper")
            QMessageBox.warning(
                self, "Updates", f"Upgrade unavailable: {exc}")
            return
        # find old spaCR files --> delete old spaCR files --> install new
        # spaCR. Step 1 runs off the GUI thread; what it found is shown before
        # anything is deleted, in _on_old_installs_found.
        self._update_version = info.latest_release
        self.statusBar().showMessage(tr("Upgrading spaCR…"), 4000)
        self._start_update_worker(
            "find", find_old_installs, self._on_old_installs_found)

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
        lines = [line for line in (output or "").splitlines() if line.strip()]
        detail = "\n".join(lines[-6:]) if lines else "No output was captured."
        escape = _the_missing_pip_escape(output)
        if escape:
            detail += (
                "\n\nThis environment has no pip -- it was built with "
                "uv, which does not install one. Run this once, in a "
                "terminal, to upgrade:\n\n" + escape)
        QMessageBox.warning(
            self, "Updates",
            f"pip returned exit code {return_code}.\n\n{detail}")

    def _on_old_installs_found(self, records) -> None:
        """Show what step 1 found, then run steps 2 and 3 in that order.

        Installer-made copies are removed; environments the user made are
        offered with tick boxes. When the running spaCR is itself an
        installer-made copy it cannot delete itself, so a helper process
        finishes the update after this window closes.

        :param records: installations from
            :func:`spacr.install_cleanup.find_old_installs`.
        """
        if self._closing:
            LOG.debug("Discarding the old-install list during shutdown")
            return
        from spacr import install_cleanup, updater

        records = list(records or ())
        ticked = ()
        if any(r.kind == "installer" or (r.kind == "environment" and not r.running)
               for r in records):
            ticked = self._confirm_old_installs(records)
            if ticked is None:
                return
        version = getattr(self, "_update_version", None) or ""
        if any(r.kind == "installer" and r.running for r in records):
            plan = install_cleanup.start_update_helper(
                records, version, ticked=ticked)
            if not plan.get("command"):
                QMessageBox.warning(
                    self, "Updates",
                    tr("Upgrade unavailable: {error}",
                       error=self._removal_reason_text(
                           str(plan.get("error")))))
                return
            QMessageBox.information(
                self, "Updates",
                tr("spaCR will close, remove the older copies, install "
                   "{version} and start again.", version=version))
            self.close()
            return
        self._start_update_worker(
            "upgrade",
            lambda: install_cleanup.run_update_sequence(
                updater.run_pip_upgrade, records=records, ticked=ticked),
            self._on_update_sequence_done)

    def _confirm_old_installs(self, records):
        """List the copies an update removes, with a tick box per environment.

        :param records: installations from
            :func:`spacr.install_cleanup.find_old_installs`.
        :returns: the roots of the ticked environments, or ``None`` when the
            user cancelled.
        """
        from PySide6.QtWidgets import (
            QCheckBox, QDialog, QDialogButtonBox, QLabel, QVBoxLayout,
        )

        dialog = QDialog(self)
        dialog.setObjectName("OldInstallsDialog")
        dialog.setWindowTitle(tr("Remove older spaCR copies"))
        layout = QVBoxLayout(dialog)
        copies = [r for r in records if r.kind == "installer"]
        yours = [r for r in records if r.kind == "environment" and not r.running]
        if copies:
            layout.addWidget(QLabel(tr(
                "These older copies of spaCR will be removed before the new "
                "version is installed:")))
            for record in copies:
                layout.addWidget(QLabel(
                    f"{record.root}  ({record.version or '?'})"))
        boxes = []
        if yours:
            layout.addWidget(QLabel(tr(
                "Environments you made. Tick one to uninstall spaCR from it; "
                "the environment itself is kept.")))
            for record in yours:
                box = QCheckBox(f"{record.root}  ({record.version or '?'})")
                layout.addWidget(box)
                boxes.append((record, box))
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(tr("Remove and update"))
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return None
        return tuple(record.root for record, box in boxes if box.isChecked())

    def _on_update_sequence_done(self, outcome) -> None:
        """Report steps 2 and 3 of an in-app update on the GUI thread.

        :param outcome: ``(reports, install_result)`` from
            :func:`spacr.install_cleanup.run_update_sequence`;
            ``install_result`` is ``None`` when nothing was installed.
        """
        if self._closing:
            LOG.debug("Discarding an update result during shutdown")
            return
        reports, result = outcome
        if result is None:
            failed = [f"{item}: {self._removal_reason_text(why)}"
                      for report in reports for item, why in report.failed]
            QMessageBox.warning(
                self, "Updates",
                tr("The update stopped before installing, because an older "
                   "copy of spaCR could not be removed:")
                + "\n\n" + "\n".join(failed[:8]))
            return
        self._on_upgrade_done(result)

    @staticmethod
    def _removal_reason_text(why: str) -> str:
        """Say why an old copy was not removed, in the interface language.

        :mod:`spacr.install_cleanup` answers in English, because the
        installers run it where no spaCR, and so no catalog, is installed.
        The reasons it can give an in-app update are translated here, each
        written out in full so the catalog builder finds it. Anything else --
        an operating-system error, the last line pip printed -- is not
        spaCR's wording and is shown as it came.

        :param why: a reason from a
            :class:`spacr.install_cleanup.RemovalReport`, or the error
            :func:`spacr.install_cleanup.start_update_helper` returned.
        :returns: the reason to show.
        """
        prefix = "needs administrator rights; run: "
        if why.startswith(prefix):
            return tr("needs administrator rights; run: {command}",
                      command=why[len(prefix):])
        reasons = {
            "needs administrator rights; delete it as an administrator":
                tr("needs administrator rights; delete it as an "
                   "administrator"),
            "in use or not permitted; close anything using it and try again":
                tr("in use or not permitted; close anything using it and "
                   "try again"),
            "not permitted; delete this registry entry by hand":
                tr("not permitted; delete this registry entry by hand"),
            "no Python found in it; run pip uninstall spacr in that "
            "environment":
                tr("no Python found in it; run pip uninstall spacr in that "
                   "environment"),
            "refused: a shared folder, not a spaCR installation; remove "
            "spaCR from it by hand":
                tr("refused: a shared folder, not a spaCR installation; "
                   "remove spaCR from it by hand"),
            "refused: not recognisably a spaCR installation; delete it by "
            "hand if it is one":
                tr("refused: not recognisably a spaCR installation; delete "
                   "it by hand if it is one"),
            "no Python outside the installation could be found to finish "
            "the update; run the new installer instead":
                tr("no Python outside the installation could be found to "
                   "finish the update; run the new installer instead"),
        }
        return reasons.get(why, why)

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
        from .screens.app_screen import AppScreen
        seen_screens = set()
        for screen in list(getattr(self, "_screens", {}).values()):
            if not isinstance(screen, AppScreen) or id(screen) in seen_screens:
                continue
            seen_screens.add(id(screen))
            try:
                accepted = screen.close()
            except RuntimeError:
                continue
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
        worker = getattr(self, "_update_worker", None)
        if worker is not None:
            try:
                worker.wait(5000)
            except RuntimeError:
                pass
        super().closeEvent(event)
        if event.isAccepted():
            app = QApplication.instance()
            if app is not None:
                app.quit()

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

    def _show_module_hint(self, key: str) -> None:
        """Explain the hovered dock row in whichever bottom strip is showing.

        ROUTED RATHER THAN OWNED, because there is no single bar to write to.
        Home carries a `ModuleHintBar` at the foot of the startup page and a
        module screen carries its own per-setting strip at the foot of the
        form; both are the bottom of the window from the reader's side, and
        a third window-level bar under whichever is on screen would stack two
        strips on Home.

        THE SENTENCE IS RESOLVED HERE, once, and handed on.
        `module_summary` falls back to the registry's English description,
        so called without one it answers an empty string -- which is how a
        dock hover over a module screen silently wrote nothing. The window is
        the one object with `APPS` already to hand; a screen would have to
        import the module that built it.

        Silent when the page in front has no strip: a hover is not a place to
        raise anything, and a screen that cannot show help is not broken.
        """
        key = str(key or "")
        if not key:
            return
        handler = getattr(self._stack.currentWidget(), "show_module_hint",
                          None)
        if not callable(handler):
            return
        summary = ""
        try:
            from .i18n_module_summaries import module_summary
            source = next((desc for k, _n, desc, _s in APPS if k == key), "")
            if source:
                summary = module_summary(key, source)
        except Exception:                                        # noqa: BLE001
            summary = ""
        try:
            handler(key, summary)
        except Exception:                                        # noqa: BLE001
            import logging
            logging.getLogger(__name__).debug(
                "could not write the hint for %s", key, exc_info=True)

    def apply_dock_mode(self, mode: Optional[str] = None) -> None:
        """Put the app list where the preference says it goes.

        Two modes, one ``Sidebar`` object:

        ``locked``  the sidebar sits in the window's dock slot, where it
                    is an ordinary column in the layout: it never
                    animates, and because it is a LAYOUT MEMBER rather
                    than an overlay the page beside it is narrower by the
                    dock's width instead of running underneath it.
        ``hidden``  the slot stays empty. The "All apps" action is
                    disabled with a tooltip that says where to turn it
                    back on — a control that silently does nothing is
                    worse than one that is greyed out.

        THERE IS NO REVEAL-ON-HOVER MODE. It slid the dock in OVER the
        page, so the home screen's module tiles sat underneath it and did
        not move aside, and it needed a second container behind the dock's
        own panel to be legible over whatever it covered. Both were the
        complaint. The drawer object still exists and is kept closed and
        disabled, because it also carries the keyboard path that the
        "All apps" action uses.

        Idempotent, and safe to call before the menu exists.
        """
        mode = mode or self.dock_mode()
        drawer = getattr(self, "_app_drawer", None)
        sidebar = getattr(self, "_sidebar", None)
        slot = getattr(self, "_dock_slot", None)
        if drawer is None or sidebar is None or slot is None:
            return
        self._dock_mode = mode

        drawer.close()
        drawer.set_enabled(False)
        if mode == "locked":
            if sidebar.parent() is not slot:
                slot.layout().addWidget(sidebar)
            sidebar.setFixedWidth(sidebar.fitting_width())
            sidebar.show()
            slot.show()
        else:
            slot.hide()

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
        """Put keyboard focus on the app dock. The menu and Ctrl+Shift+A path.

        THE DOCK NO LONGER SLIDES, so there is nothing to open: it is a
        permanent column when the preference is ``locked``. What the action
        still has to do is get a keyboard user INTO it, because a column you
        can only reach by tabbing through the page is one that is hard to
        reach at all.

        A no-op when the dock is hidden -- the user asked for it not to be
        there, and a shortcut that overrules a preference is a bug. The
        action is disabled in that mode anyway, so this is the second half
        of a belt and braces rather than the only guard.
        """
        if getattr(self, "_dock_mode", "locked") != "locked":
            return
        sidebar = getattr(self, "_sidebar", None)
        if sidebar is None:
            return
        try:
            rows = sidebar.rows()
        except Exception:                                    # noqa: BLE001
            LOG.debug("the dock would not list its rows", exc_info=True)
            return
        if rows:
            rows[0].setFocus(Qt.FocusReason.ShortcutFocusReason)

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

        NOTHING TO CLOSE AT ALL NOW. The dock is a column in the layout,
        and a column that vanished every time you used it would be worse
        than the reveal it replaced. Kept as the connection point so
        `nav_selected` still has somewhere to land, and so a future mode
        that does need closing has one place to do it.
        """
        return

    def _install_startup_page(self):
        """Instantiate the Home page and add it to the stack."""
        self._startup = make_home_page()
        self._startup.tile_clicked.connect(self._on_nav_selected)
        self._startup.update_check_requested.connect(self._check_for_updates)
        try:
            self._startup._btn_all_apps.clicked.connect(self.toggle_app_drawer)
        except Exception:
            pass
        self._a_page_joined_the_stack(self._startup)
        self._stack.addWidget(self._startup)
        self._drop_a_redundant_screen_backdrop(self._startup)

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

    def _screen_scale_is_stale(self, key: str) -> bool:
        """Whether ``key``'s cached screen was built at a different scale.

        :param key: the app whose cached screen is in question.
        :returns: True when the screen would be the wrong size on screen.
        """
        built_at = self._screen_scales.get(key)
        if built_at is None:
            return False
        return abs(built_at - _current_font_scale()) > 1e-6

    def _rebuild_for_scale(self, key: str) -> None:
        """Rebuild ``key``'s screen at the current scale, carrying its values.

        Delegates to :meth:`rebuild_app_screen`, which is the tested path: it
        builds the replacement BEFORE retiring the old screen, carries the
        values and the loaded preview across, and declines to destroy a screen
        whose worker is still running.

        The values come from ``_settings_model.collect()``, which is what
        :meth:`spacr.qt.screens.app_screen.AppScreen._rebuild_the_form` uses
        for the same purpose -- NOT from a ``values()`` method. An AppScreen
        has no such method; the ``values`` found by duck-typing on an earlier
        draft of this belonged to an unrelated captions dict, so every rebuild
        silently took the do-nothing branch and the feature never worked.

        A screen with no settings model is not an :class:`AppScreen` --
        Annotate, Make Masks, the Database Browser. Those hold state a rebuild
        would silently discard (a loaded image, a scan in progress), so they
        are left alone and simply recorded at the new scale; the alternative
        is throwing away the user's work to fix an icon size. They come up
        correctly the next time the app starts.

        :param key: the app whose screen is to be rebuilt.
        """
        screen = self._screens.get(key)
        if screen is None:
            return
        model = getattr(screen, "_settings_model", None)
        collect = getattr(model, "collect", None)
        values = None
        if callable(collect):
            try:
                values = dict(collect() or {})
            except Exception:                                # noqa: BLE001
                LOG.exception("could not read the %s form before a rescale",
                              key)
                values = None
        if values is None:
            self._screen_scales[key] = _current_font_scale()
            return
        self.rebuild_app_screen(key, values)

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
        previous_build_values = AppScreen.values_the_next_screen_is_built_for
        AppScreen.values_the_next_screen_is_built_for = dict(values or {})
        try:
            fresh = self._build_screen(key)
        except Exception:                                    # noqa: BLE001
            LOG.exception("could not rebuild the %s screen", key)
            return
        finally:
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

        try:
            fresh._form_shape_on_screen = fresh._form_shape()
        except Exception:                                    # noqa: BLE001
            pass
        try:
            fresh._settings_model._organelle_preset_owned = old_preset_owned
        except AttributeError:
            pass
        _carry_preview_state(old, fresh)

        self._screens[key] = fresh
        self._screen_scales[key] = _current_font_scale()
        self._a_page_joined_the_stack(fresh)
        self._stack.addWidget(fresh)
        self._drop_a_redundant_screen_backdrop(fresh)
        self._stack.setCurrentWidget(fresh)
        if old is not None:
            try:
                if old.close() is False:
                    self._stack.removeWidget(fresh)
                    fresh.setParent(None)
                    fresh.close()
                    fresh.deleteLater()
                    self._screens[key] = old
                    self._stack.setCurrentWidget(old)
                    old.register_workspace()
                    return
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
            try:
                self._startup.refresh()
            except Exception:
                pass
            self._stack.setCurrentWidget(self._startup)
            self._status_app_label.setText(tr("Home"))
            self.statusBar().showMessage(tr("Home"), 2000)
            _timing.watch_interactive(
                self._startup, "interactive Home", "__home__",
                started_at=interaction_started,
                budget_s=_timing.HOME_BUDGET_S,
            )
            return
        if key in self._screens and self._screen_scale_is_stale(key):
            self._rebuild_for_scale(key)
        if key not in self._screens:
            card = self._show_preparing(key)
            try:
                self._screens[key] = self._build_screen(key)
                self._screen_scales[key] = _current_font_scale()
            finally:
                self._hide_preparing(card)
            try:
                self._theme_screen(self._screens[key], key)
            except Exception:
                LOG.exception("Could not theme the %s screen", key)
            self._a_page_joined_the_stack(self._screens[key])
            self._stack.addWidget(self._screens[key])
            self._drop_a_redundant_screen_backdrop(self._screens[key])
            try:
                from .i18n import retranslate_widget_tree
                retranslate_widget_tree(self._screens[key])
            except Exception:
                LOG.exception("Could not translate the %s screen", key)
        try:
            from .screens.settings_model import retarget_field_tooltips

            retarget_field_tooltips(self._screens[key])
        except Exception:
            LOG.exception("Could not retarget help on the %s screen", key)
        self._stack.setCurrentWidget(self._screens[key])
        _timing.watch_interactive(
            self._screens[key], "interactive module", key,
            started_at=interaction_started,
            budget_s=_timing.MODULE_BUDGET_S,
        )
        if key in self._visit_order:
            self._visit_order.remove(key)
        self._visit_order.append(key)
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
            from .widgets.ambient import (install_ambient,
                                          _the_heavy_import_lock_is_free)
            if not _the_heavy_import_lock_is_free():
                self._retry_screen_backdrop(screen, key)
                return
            install_ambient(
                screen, None,
                theme=get_ambient_theme(), palette=get_ambient_palette(),
                backdrop=theme_background_path(resolve_effective_theme()))
        except Exception as error:
            try:
                from .widgets.ambient import _the_backdrop_wants_a_retry
            except Exception:                                # noqa: BLE001
                _the_backdrop_wants_a_retry = None
            if (_the_backdrop_wants_a_retry is not None
                    and _the_backdrop_wants_a_retry(error)):
                self._retry_screen_backdrop(screen, key)
                return
            LOG.exception("Could not install the backdrop for %s", key)

    def _drop_a_redundant_screen_backdrop(self, screen) -> None:
        """Retire a screen's own backdrop when this window already has one.

        ONE BACKDROP FOR THE WINDOW. That rule is stated twice in this file
        with its symptom attached, and both screen classes carry a guard for
        it -- but the guard asks ``self.window()`` from the screen's own
        ``__init__``, where the screen has no parent and so IS its own window.
        It therefore asked the SCREEN whether it had a ``window_backdrop``,
        got None because a screen has no such method, and never once fired.
        HomePage had no guard at all.

        The check cannot be made where it was: a widget with no parent cannot
        know which window will take it, and asking the application for any
        window with a backdrop is a different question with a different answer
        (it finds another window's). So it is asked HERE instead, by the window
        itself, at the moment it takes the screen -- which is the first moment
        the answer exists.

        MEASURED on a 3840x2160 screen at font_scale 2 before
        this: two visible backdrops on Home (3840x2114 with 3400x2114 laid
        over it) and two on a module screen, each shading and blitting a
        full-size field at 12.5 fps, the lower one covered over 87% of its
        area. 950 paints/s and 393 Mpx/s with nobody touching the machine --
        47 full 4K screens repainted per second for a picture nobody could
        see. The cost is counted in pixels, so it is four times worse at 4K
        than at 1080p, which is why this was reported from that machine.
        Idle GUI-thread CPU fell from ~16% of a core to ~10%.

        :param screen: the screen this window has just taken.

        DISABLED 2026-09-07, AND THE MEASUREMENT ABOVE IS WHY IT LOOKED
        RIGHT. Everything in it is true and it still made the product worse:
        the two backdrops WERE both being shaded and blitted, and the lower
        one WAS 87% covered -- but the 13% that was not covered is the
        theme. Retiring the screen's own backdrop left the window's, which
        sits behind the stack, and HomePage's plain `QWidget` containers
        paint `bg` over it. On the dark theme `bg` is `#000000`.

        MEASURED THE SAME WAY BOTH TIMES, from X rather than from
        `QWidget.grab()` (which cannot see the GL-backed widget at all and
        reports black for a working backdrop): at this commit's parent
        `f9fa45319` the home screen is 29.6% chromatic pixels, and at the
        commit itself 2.5%, with 31.7% pure black. The report
        was "i get a black background when the blobs theme is active", and
        1.5.0.1 -- the last desktop install -- measures 38.9%.

        IT WAS OFF FROM 2026-09-07 UNTIL 2026-09-11, waiting for the
        window's backdrop to be actually visible through the screens --
        "the containers above it have to stop painting an opaque `bg`".
        They have. `theme._window_block` now gives an opaque theme the
        transparent window block whenever an animated backdrop is running,
        so a plain `QWidget` no longer paints over it.

        MEASURED BEFORE TURNING IT BACK ON, from X, on a real
        display, home screen at 1600x1000:

            theme   this off         this on, old QSS   this on, new QSS
            dark    58.8% chromatic  3.0% chromatic     60.1% chromatic
                    0.0% black       20.4% PURE BLACK   0.0% black
            cell    93.5% / 0.3%     93.5% / 0.2%       95.7% / 0.1%
            glass   61.7% / 0.0%     67.3% / 0.0%       72.8% / 0.0%

        The middle column is the regression this function caused, on
        demand. The right-hand column is it fixed, with dark better than it
        was with this function disabled entirely.

        `tools/measure_the_home_screen_is_not_black.py` is that
        measurement, and `tools/can_this_display_be_measured.py` must pass
        first -- an unraised or unsettled grab returns a convincing
        near-black that looks exactly like the bug.
        """
        if self.window_backdrop() is None:
            screen._uses_window_backdrop = False
            return
        own = getattr(screen, "_ambient", None)
        if own is None:
            if getattr(screen, "_surrendered_its_backdrop", False):
                screen._uses_window_backdrop = True
                self._stop_painting_over_the_window_backdrop(screen)
            return
        screen._uses_window_backdrop = True
        screen._surrendered_its_backdrop = True
        try:
            from .widgets.ambient import AmbientWidget

            doomed = [own] + [c for c in list(screen.children())
                              if isinstance(c, AmbientWidget) and c is not own]
        except Exception:                                    # noqa: BLE001
            doomed = [own]
        for child in doomed:
            for step in ("set_animating", "setParent", "deleteLater"):
                try:
                    if step == "set_animating":
                        child.set_animating(False)
                    elif step == "setParent":
                        child.setParent(None)
                    else:
                        child.deleteLater()
                except Exception:                            # noqa: BLE001
                    continue
        screen._ambient = None
        try:
            clear = getattr(screen, "_clear_page_surfaces", None)
            if callable(clear):
                clear()
        except Exception:                                    # noqa: BLE001
            pass

    def _stop_painting_over_the_window_backdrop(self, screen) -> None:
        """Clear a screen's surfaces and repaint it, now that it defers again.

        The mirror image of the two steps :meth:`_retire_the_dock_backdrop`
        takes when the window's animation goes away. Setting
        ``_uses_window_backdrop`` is only half of handing the picture back:
        the flag stops ``page_fill`` returning a colour, and this makes the
        containers above it transparent again and asks for the frame that
        shows the difference. Without the repaint the slab stays on screen
        until something else damages the region, which on a settled window
        can be a long time.

        :param screen: the screen that has just started deferring to the
            window's backdrop again.

        Never raises: a screen that cannot be repainted is a cosmetic
        problem, and this runs inside the Preferences save.
        """
        try:
            clear = getattr(screen, "_clear_page_surfaces", None)
            if callable(clear):
                clear()
        except Exception:                                    # noqa: BLE001
            pass
        try:
            screen.update()
        except Exception:                                    # noqa: BLE001
            pass

    def _a_page_joined_the_stack(self, page) -> None:
        """Mark a page so the sheet reaches it, however late it arrives.

        THE GAP `stylesheet_roots` OPENS IF NOBODY CLOSES IT. Marking
        happens while a sheet is being applied, so a module opened for the
        FIRST time after that point was never marked -- and the window is
        deliberately bare, so the page inherits nothing. Measured before
        this existed: one module open, the sheet applied, a second module
        opened, and a probe under the new page resolved to ``#000000`` on
        the dark theme. Black text on a dark screen, a whole module wide.

        Before per-screen sheeting this could not happen: the window
        carried the sheet and every descendant added later inherited it.

        Marking rather than sheeting, because the event filter sheets a
        marked widget on its ``Polish`` or ``Show`` -- which is before its
        first paint, and is also the only moment at which a page that is
        added but not raised should cost anything.

        CALLED BEFORE ``addWidget``, AND THE ORDER IS THE WHOLE POINT.
        ``QStackedWidget.addWidget`` makes the FIRST widget current and
        shows it there and then, so a mark applied afterwards arrives after
        that page's only ``Show`` and the filter never sees it marked. Every
        later page is unaffected -- the stack is no longer empty, so
        ``addWidget`` does not raise them -- which is exactly what makes
        this the kind of defect that ships: it is wrong for one page out of
        forty-five, and that page is Home.

        :param page: the widget that has just been added to the stack.
        """
        try:
            from .theme import mark_as_a_sheet_target
            mark_as_a_sheet_target(page)
        except Exception:                                    # noqa: BLE001
            LOG.exception("Could not mark a new page for the theme sheet")

    def stylesheet_roots(self):
        """The widgets that carry the application sheet, instead of me.

        WHY NOT THE WINDOW. Setting a stylesheet on a window repolishes
        every descendant, and a session that has opened four modules owns
        8,002 widgets of which 7,595 are inside module screens -- ONE of
        which is on screen. Measured on this box, four modules open,
        alternating dark and light:

            the whole window            945 ms first, ~1,479 ms steady
            chrome + the visible page   300 ms first,   ~310 ms steady

        WHAT IS IN THE LIST, and the shape is not the obvious one. The
        chrome is 407 widgets and only TWO of them are reachable from the
        central widget -- the dock, the sidebar and the status strip hang
        off the window itself. An implementation that walked the central
        widget would leave 405 unstyled and look almost right, which is
        why `tests/qt/test_every_widget_wears_the_current_theme.py` does
        exactly that on purpose and asserts it is caught.

        So: every direct child of the window except the central widget,
        every direct child of the central widget except the stack, and the
        stack's CURRENT page. The pages that are not showing are marked
        instead, and sheeted on their own `showEvent` before they are
        painted.
        """
        from PySide6.QtCore import Qt

        from .theme import mark_as_a_sheet_target

        central = self.centralWidget()
        stack = getattr(self, "_stack", None)
        if central is None or stack is None:
            return [self]

        roots = [kid for kid in self.findChildren(
            QWidget, options=Qt.FindDirectChildrenOnly) if kid is not central]
        roots.extend(kid for kid in central.findChildren(
            QWidget, options=Qt.FindDirectChildrenOnly) if kid is not stack)

        current = stack.currentWidget()
        for page in stack.findChildren(QWidget,
                                       options=Qt.FindDirectChildrenOnly):
            mark_as_a_sheet_target(page)
            if page is current or page.isVisible():
                roots.append(page)
        return roots or [self]

    def window_backdrop(self):
        """The one backdrop behind the dock AND the page, or ``None``.

        Read by the screens so they do not build a second one on top of it --
        see :meth:`_install_screen_backdrop`.
        """
        return getattr(self, "_dock_backdrop", None)

    def _backdrop_the_dock_column(self) -> None:
        """Put ONE live backdrop behind everything in the central area.

        THE BACKDROP USED TO BE PER SCREEN, inside the stack, and the dock
        slot is a SIBLING of the stack -- so the animation never reached
        behind the dock. The strip was the window's flat background while the
        page beside it was animated, and a flat rectangle beside a moving one
        reads as a box whatever colour it is. That is why colouring it never
        worked. Proved rather than guessed: hiding the dock made those same
        pixels show the animation, because the stack expanded over them.

        Giving the dock ITS OWN fixed that and introduced the next fault: two
        animations, one per container, running out of step across a seam.
        "i want the theme on one container in the background of everything."

        So there is ONE, on the central widget, behind the dock slot and the
        stack both. The screens ask :meth:`window_backdrop` and decline to
        build their own when it exists -- they still clear their page
        surfaces, which is what lets this one through.

        Never raises: the window opens with or without its decoration.
        """
        try:
            from PySide6.QtCore import QTimer

            from .preferences import (get_ambient_enabled, get_ambient_palette,
                                      get_ambient_theme,
                                      resolve_effective_theme,
                                      theme_background_path)
            host = self.centralWidget()
            if host is None:
                return
            if not get_ambient_enabled():
                self._retire_the_dock_backdrop()
                return
            if getattr(self, "_dock_backdrop", None) is not None:
                return
            from .widgets.ambient import (_the_heavy_import_lock_is_free,
                                          install_ambient)
            if not _the_heavy_import_lock_is_free():
                QTimer.singleShot(400, self._backdrop_the_dock_column)
                return
            self._dock_backdrop = install_ambient(
                host, None,
                theme=get_ambient_theme(), palette=get_ambient_palette(),
                backdrop=theme_background_path(resolve_effective_theme()))
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not put a backdrop behind the dock",
                      exc_info=True)

    def _retire_the_dock_backdrop(self) -> None:
        """Stop and drop the window's backdrop, and tell the screens.

        Two things have to happen and the second is the one that bites.
        Stopping the animation is obvious; clearing every screen's
        ``_uses_window_backdrop`` is not, and without it the screens keep
        declining to paint their own page for a backdrop that is gone --
        the flat ``surface`` slab, which is the black page reported three
        times.

        Never raises: an unparented ``AmbientWidget`` awaiting
        ``deleteLater`` still ticks, so ``set_animating(False)`` is tried
        first and independently of the rest.
        """
        doomed = getattr(self, "_dock_backdrop", None)
        self._dock_backdrop = None
        if doomed is not None:
            for step in ("set_animating", "setParent", "deleteLater"):
                try:
                    if step == "set_animating":
                        doomed.set_animating(False)
                    elif step == "setParent":
                        doomed.setParent(None)
                    else:
                        doomed.deleteLater()
                except Exception:                            # noqa: BLE001
                    continue
        for screen in list(getattr(self, "_screens", {}).values()) + [
                getattr(self, "_startup", None)]:
            if screen is None:
                continue
            try:
                screen._uses_window_backdrop = False
                clear = getattr(screen, "_clear_page_surfaces", None)
                if callable(clear):
                    clear()
                screen.update()
            except Exception:                                # noqa: BLE001
                continue

    def _retry_screen_backdrop(self, screen: QWidget, key: str) -> None:
        """Come back for :meth:`_install_screen_backdrop` shortly.

        The screen is checked for liveness at the far end rather than
        here: a module can be closed inside the interval, and calling a
        method on a freed QWidget is the "Internal C++ object already
        deleted" storm this file has had before.
        """
        from PySide6.QtCore import QTimer

        def again() -> None:
            """Try the backdrop once more, if the screen is still alive."""
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
            from .theme import font_px

            card.setStyleSheet(
                "background: rgba(10,14,24,200); color: rgb(235,240,245); "
                "padding: 14px 22px; border-radius: 10px; "
                f"font-size: {font_px(15)}px;")
            card.adjustSize()
            card.move(max(0, (self.width() - card.width()) // 2),
                      max(0, (self.height() - card.height()) // 2))
            card.raise_()
            card.show()
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
        """Build one module's screen, timed.

        :param key: the module to build.
        :returns: the screen widget.
        """
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
        from .screens.app_screen import AppScreen
        if isinstance(widget, AppScreen):
            return widget.app_key, dict(widget._settings_model.collect())
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
        sync = getattr(widget, "_sync_folded_switches", None)
        if callable(sync):
            sync(dict(seed))

    @staticmethod
    def _apply_seed_value(w: QWidget, value) -> None:
        """Write a seed value into whichever kind of control holds it.

        A widget with its own ``set_value`` is asked first, so a compound
        control decides for itself what a value means. A combo is matched on its
        stored data or its caption, in that order.

        :param w: the control to write into.
        :param value: the value to seed it with.
        """
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
    font.setWeight(QFont.Weight.Light if str(weight).lower() == "light"
                   else QFont.Weight.Normal)
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

    from .setup_screen import take_the_setup_flags

    argv, told_to_skip_setup = take_the_setup_flags(argv)

    initial_app = argv[0] if argv else None

    _install_crash_dump()

    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    try:
        if "matplotlib" in sys.modules:
            import matplotlib

            matplotlib.use("Agg", force=True)
        else:
            os.environ["MPLBACKEND"] = "Agg"
    except Exception:                                    # noqa: BLE001
        pass

    QApplication.setAttribute(Qt.AA_DontUseNativeDialogs, True)

    from .menus import name_the_application

    _app_name, _app_display_name = name_the_application()

    app = QApplication(sys.argv[:1])
    if os.environ.get("SPACR_WATCH_GUI_STALLS"):
        try:
            from .stall_watch import watch_this_application
            watch_this_application(app)
        except Exception:                                    # noqa: BLE001
            LOG.debug("Could not start the GUI stall watch", exc_info=True)
    try:
        from .thread_guard import install as _install_thread_guard
        _install_thread_guard()
    except Exception:
        pass
    try:
        from .gc_policy import install as _install_gc_policy
        _install_gc_policy(app)
    except Exception:
        pass
    LOG.info("application named %r (display %r); Qt reports %r / %r",
             _app_name, _app_display_name, app.applicationName(),
             app.applicationDisplayName())
    try:
        from .laptop_mode import apply as _apply_laptop_mode, describe
        from .preferences import get_performance_level as _level_now
        # 286: THE LEVEL DECIDES, NOT THE MACHINE. `apply()` with no argument
        # measures cores and memory, which overrode the level chosen in the
        # selector -- a two-core Workstation lost its backdrop at every
        # start. The measurement is still logged, as a reading, because "it
        # looks different on my laptop" needs evidence.
        _level = _level_now()
        LOG.info("performance level %s; hardware reading, for diagnosis "
                 "only: %s", _level, describe())
        _laptop = _apply_laptop_mode(_level == "laptop")
        if _laptop["changed"]:
            LOG.info("laptop mode changed: %s", ", ".join(_laptop["changed"]))
    except Exception:                                    # pragma: no cover
        LOG.debug("could not decide laptop mode", exc_info=True)
    app.setDesktopFileName("io.github.olafssonlab.spacr")
    app.setWindowIcon(QIcon(os.path.join(
        iconset.RESOURCE_DIR, "app_icon.png")))

    try:
        from PySide6.QtGui import QImageReader
        QImageReader.setAllocationLimit(0)
    except Exception:
        LOG.debug("This Qt build does not expose QImageReader allocation "
                  "limits", exc_info=True)

    with _timing.span("fonts"):
        _load_bundled_fonts()
        _use_open_sans(app)

    from .logging_util import setup_logging
    setup_logging()

    from .preferences import apply_preferences_to_app
    apply_preferences_to_app(app)
    from .i18n import install_qt_translations
    install_qt_translations(app)

    import logging as _lg
    import sys as _sys

    from .verbose_logger import current_log_file
    _lg.getLogger("spacr").info(
        "spaCR launched (python=%s.%s.%s, log=%s)",
        _sys.version_info.major, _sys.version_info.minor,
        _sys.version_info.micro, current_log_file())

    app.setQuitOnLastWindowClosed(False)

    if not told_to_skip_setup:
        try:
            from .widgets.setup_slides import open_setup_if_needed

            asked = open_setup_if_needed(None)
            if asked is not None:
                apply_preferences_to_app(app)
        except Exception:
            LOG.debug("could not open the setup screen", exc_info=True)

    _timing.mark("MainWindow")
    win = MainWindow(initial_app=initial_app)
    benchmark_controller = None
    if os.environ.get("SPACR_BENCHMARK_JSON", "").strip():
        from .startup_benchmark import maybe_start as _maybe_start_benchmark

        benchmark_controller = _maybe_start_benchmark(app, win)

    if win._stack.currentWidget() is win._startup:
        _timing.watch_interactive(
            win._startup, "interactive Home", "__home__",
            started_at=_timing.process_started_at(),
            budget_s=_timing.HOME_BUDGET_S,
        )
    win._startup_benchmark_controller = benchmark_controller
    _open_at_the_measured_width(win)
    win.show()

    install_the_dialog_filters(app)

    try:
        from .live_zoom import install_live_zoom
        install_live_zoom(app)
    except Exception:                                        # noqa: BLE001
        LOG.exception("the live zoom gesture could not be installed")

    def _prewarm():
        """Import the slow settings module off the GUI thread.

        Runs on a daemon thread while the user looks at the home screen, so
        opening a module finds the import cached instead of paying for it.
        Importing modules creates no Qt objects and is safe off-thread; a
        failure is logged and ignored, because a cold import is slow rather
        than broken.
        """
        try:
            import importlib
            for mod in ("spacr.settings",
                        "spacr.qt.screens.settings_model",
                        "spacr.qt.imagery"):
                importlib.import_module(mod)
        except Exception:
            LOG.debug("Could not prewarm GUI settings imports", exc_info=True)
    from .preferences import in_safe_mode

    if not in_safe_mode():
        threading.Thread(target=_prewarm, name="spacr-prewarm",
                         daemon=True).start()

    def _drain_ai():
        """Stop every job runner before Qt starts destroying widgets.

        Connected to ``aboutToQuit``, which fires however the application
        exits. See the comment below for why this covers every runner and
        not only the consoles.
        """
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
        try:
            from . import ai as _ai
            for p in _ai.list_providers():
                p.cancel_stream()
        except Exception:
            LOG.debug("Could not cancel every AI provider", exc_info=True)
    app.aboutToQuit.connect(_drain_ai)

    _timing.mark("entering the event loop")
    _timing.watch_the_gui_thread(app)
    from PySide6.QtCore import QTimer as _TimingQTimer

    _TimingQTimer.singleShot(0, _timing.event_loop_started)
    try:
        code = app.exec()
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
