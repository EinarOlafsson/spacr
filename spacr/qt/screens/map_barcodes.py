"""Barcode QC integration and shared support for folded modules.

The Barcode QC page assesses reads per well, low-depth wells, unmapped reads,
barcode collisions, positional effects, library coverage and the resulting
abundance threshold. It opens beside the Map Barcodes settings with its full
settings form, run controls, console and figures.

This module also provides the common infrastructure used by host screens:
:func:`install_fold_strip`, :class:`FoldOpener`,
:func:`build_settings_screen` and :func:`install_window_hooks`. These helpers
mount a complete module page, connect host signals and attach its masthead
button without duplicating analytical implementations.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from functools import partial
from typing import Callable, Dict, Optional, Sequence, Tuple

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView,
                               QFrame, QLabel, QPushButton, QScrollArea,
                               QSizePolicy, QSplitter, QTabBar, QTableWidget,
                               QTableWidgetItem, QTabWidget, QToolButton,
                               QVBoxLayout, QWidget)

from ..i18n import tr
from ..theme import install_close_marks
from ..widgets.fold_strip import FoldStrip

LOG = logging.getLogger(__name__)

#: Registry key of the screen this module hangs its strip on.
HOST_KEY = "map_barcodes"

#: Registry keys of the modules folded into it, in the order the strip
#: draws them.
FOLDED_APPS: Tuple[str, ...] = ("barcode_qc",)

#: Opening size of a folded module's window. Wide enough for a settings
#: form beside a console, which is what every settings-driven module is.
FOLD_WINDOW_SIZE = (1180, 760)

#: What each folded module's TILE said: ``key → (name, description,
#: stage)``.
#:
#: :class:`~spacr.qt.widgets.fold_strip.FoldStrip` reads all three out of
#: the app registry, which is right while the module still has a row and
#: answers nothing once the row is dropped -- the tooltip empties and the
#: stage falls back to stable, so an alpha module's button would light
#: blue where its tile lit green-cyan. This is what the tile said, kept so
#: the button can go on saying it.
#:
#: THE ONE TABLE :func:`fold_description` READS, so it holds every key any
#: host folds and not only the ones folded into Map Barcodes. Image UMAP
#: and Regression each kept their own copy beside their own ``FOLDED_APPS``,
#: which read well and answered nothing: both hosts restate their buttons
#: through :func:`restate_fold_button`, which looks here, so a fallback
#: written anywhere else was a table with no reader and three buttons that
#: would have gone mute the day their rows were dropped.
#:
#: THE STAGE IS THE ONE THE MODULE CARRIES IN A RUNNING WINDOW, which is
#: not always the literal in ``app.APP_STAGE``: :func:`spacr.qt.maturity.
#: apply` runs at launch and promotes assessed modules, so a tile that
#: reads alpha under a bare ``import spacr.qt.app`` lights magenta in the
#: window the user actually has open. Copying the literal here gave three
#: of these buttons green-cyan for a beta module.
#:
#: The registry still wins whenever it has the row, and the pair is
#: asserted to agree for every key that has one, so the two cannot drift
#: apart while both exist.
FOLD_FALLBACK: Dict[str, Tuple[str, str, str]] = {
    # THE THREE REGRESSION FOLDS THAT HAD NO FALLBACK, added 2026-09-08.
    #
    # `regression.FOLDED_APPS` named six keys and this table answered for
    # three of them. The other three still have standalone registry rows, so
    # `restate_fold_button` finds their name and description there and the
    # buttons read correctly TODAY -- which is exactly why the gap was
    # invisible. The day those rows are dropped, as the fold intends, three
    # buttons on Regression go mute.
    #
    # `test_the_fold_fallback_is_in_the_table_that_is_actually_read` is the
    # assertion that caught it, and its own docstring says why this table
    # and no other: `install_folds` restates through
    # `map_barcodes.restate_fold_button`, which looks here and nowhere else.
    # THE TEXT IS THE REGISTRY'S, CHARACTER FOR CHARACTER, and the first
    # version of these three was not: a trailing full stop and a `beta`
    # where the row says `alpha` were both caught by
    # `test_the_fold_fallback_agrees_with_whatever_still_knows`. A fallback
    # that paraphrases is a second source of truth, and it drifts silently
    # the moment the row it stands in for is edited -- which is precisely
    # the day this table starts being read.
    "investigate_hit": (
        "Investigate Hit",
        "Link a regression hit to cross-fitted candidate cells and "
        "well-level quantitative evidence",
        "alpha"),
    "profiler": (
        "Prediction Profiler",
        "Evaluate how a fitted model's prediction changes across one input "
        "variable",
        "alpha"),
    "regression_diagnostics": (
        "Diagnostics",
        "Show the diagnostic panels the last regression run wrote beside "
        "its results",
        "alpha"),
    "barcode_qc": (
        "Barcode QC",
        "Assess mapping depth, coverage, collisions and positional effects, "
        "and estimate the abundance threshold for the intended gRNAs per well.",
        "beta"),
    "classifier_evaluation": (
        "Classifier Evaluation",
        "Held-out predictions, nested CV, calibration, leakage and "
        "per-plate metrics",
        "beta"),
    "explain_cv": (
        "Explain CV Model",
        "Reproduce CV decisions from measured features, then inspect gain, "
        "held-out permutation importance and SHAP",
        "alpha"),
    "activation": (
        "Activation",
        "Generate class activation maps for image-classifier predictions",
        "beta"),
    "agreement": (
        "Annotator Agreement",
        "Compute Cohen's or Fleiss' κ across annotation columns and review "
        "discordant crops.",
        "stable"),
    "anndata_export": (
        "AnnData Export",
        "Write the measurements as .h5ad for scanpy and scvi-tools",
        "beta"),
    "illumination": (
        "Illumination",
        "Estimate and assess a flat-field correction model before "
        "measurement",
        "beta"),
    "timelapse": (
        "Timelapse",
        "Segment and track objects across the frames of a time series",
        "beta"),
    "motility": (
        "Motility Assay",
        "Quantify track velocity and straightness and stratify results by "
        "infection state.",
        "beta"),
    # Image UMAP's two other projections of the same measurement table.
    "image_scatter": (
        "Image Scatter",
        "Hover a point to see the cell; click it to open the crop",
        "alpha"),
    "pca": (
        "PCA",
        "Principal components of the measurement table, with a loadings "
        "biplot",
        "alpha"),
    # Regression's three: the figure, the list and the write-up.
    "volcano_explorer": (
        "Volcano Explorer",
        "Open a regression result, click any point for its full record, "
        "restyle the plot and export it as vector PDF or PNG",
        "alpha"),
    "hit_list": (
        "Hit List",
        "Ranked, annotated, filterable hits with effect size, FDR and gRNA "
        "agreement",
        "alpha"),
    "methods_export": (
        "Methods & Results",
        "Draft the methods and results sections from the run, with every "
        "number traced",
        "alpha"),
}


def fold_description(key: str) -> Tuple[str, str, str]:
    """Return the display name, description, and maturity stage for ``key``.

    Registry metadata is preferred while the module remains registered.
    :data:`FOLD_FALLBACK` supplies the same presentation metadata after a
    folded module's standalone registry entry is removed.
    """
    name = description = stage = ""
    try:
        from .. import app as app_module
        for row in getattr(app_module, "APPS", ()):
            if row and row[0] == key:
                name, description = row[1] or "", row[2] or ""
                stage = app_module.app_stage(key)
                break
    except Exception:
        LOG.debug("Could not read the app registry", exc_info=True)
    if not name:
        # THE DECLARED CATALOGUE, before any hand-written table. Several
        # folded modules never had a registry row at all -- they are
        # declared in `app_catalog` and built from it -- and that
        # declaration already carries the name, the sentence and the
        # maturity this button needs. Copying those three strings into a
        # per-host `FOLD_FALLBACK` is the same knowledge written twice,
        # and the copy is the one that goes stale.
        #
        # Only consulted when the registry had nothing: a module that is
        # BOTH registered and declared must present as the registry says,
        # because that is what its tile and its menu entry say.
        try:
            from ..app_catalog import DECLARED_APPS
            for declared in DECLARED_APPS:
                if declared.key == key:
                    name = declared.name or ""
                    description = declared.desc or ""
                    stage = declared.stage or ""
                    break
        except Exception:                               # noqa: BLE001
            LOG.debug("Could not read the declared catalogue", exc_info=True)
    fallback = FOLD_FALLBACK.get(key)
    if fallback is None:
        # NOT EVERY FOLD LANDS HERE. This table holds what the modules
        # folded into THIS screen said; a module folded into Measure or
        # Classify keeps its record on that host instead. The shared
        # resolver walks them all, so a button asks one question rather
        # than each host having to know about every other host's folds.
        try:
            from ..widgets.fold_strip import folded_fallback
            fallback = folded_fallback(key)
        except Exception:                               # noqa: BLE001
            LOG.debug("Could not read the shared fold records",
                      exc_info=True)
            fallback = ("", "", "")
    return (name or fallback[0], description or fallback[1],
            stage or fallback[2])


def restate_fold_button(button, key: str) -> None:
    """Apply the folded module's registered name, description, and stage.

    The operation has no visible effect while the registry still contains
    the module because the strip already uses the same metadata. After the
    registry row is removed, the fallback metadata preserves the module's
    accessible label, tooltip, and maturity-stage styling.
    """
    if button is None:
        return
    name, description, stage = fold_description(key)
    if not name and not description:
        return
    button.setToolTip(f"{name}\n{description}".strip())
    if name:
        button.setAccessibleName(name)
    if not stage:
        return
    # Asked of the button rather than done here: a switch also carries a
    # widget-local ":checked" fill computed from the stage it was built
    # with, and setting the property alone left it lighting stable-blue
    # when it was on while hovering in its own colour.
    set_stage = getattr(button, "set_stage", None)
    if callable(set_stage):
        set_stage(stage)
    elif button.property("stage") != stage:
        button.setProperty("stage", stage)
        button.style().unpolish(button)
        button.style().polish(button)


def folded_module_title(key: str) -> str:
    """Return the window title for folded module ``key``.

    The application title table is preferred so renamed modules remain
    consistent throughout the interface. Fallback metadata is used after a
    module's standalone registry entry is removed.
    """
    try:
        from .app_screen import APP_TITLES
        title = APP_TITLES.get(key)
        if title:
            return str(title)
    except Exception:
        LOG.debug("Could not read the module title table", exc_info=True)
    name = fold_description(key)[0]
    if name:
        return name
    return key.replace("_", " ").title()


def connect_host(screen: QWidget, host_window: Optional[QWidget]) -> None:
    """Connect ``screen``'s host signals to ``host_window``'s slots.

    Connections are derived from :data:`spacr.qt.chaining.HOST_CONNECTIONS`,
    the same mapping used by ``MainWindow._build_screen``. Folded and
    standalone screens therefore expose the same host-level actions.
    """
    if host_window is None:
        return
    try:
        from ..chaining import HOST_CONNECTIONS
    except Exception:
        LOG.debug("Could not read the host connection table", exc_info=True)
        return
    for signal_name, slot_name in HOST_CONNECTIONS.items():
        signal = getattr(screen, signal_name, None)
        slot = getattr(host_window, slot_name, None)
        if signal is None or not callable(slot):
            continue
        try:
            signal.connect(slot)
        except Exception:
            LOG.debug("Could not connect %s", signal_name, exc_info=True)


def build_settings_screen(key: str,
                          host_window: Optional[QWidget] = None) -> QWidget:
    """Build a fully connected settings screen for module ``key``.

    The returned screen has the same host connections and declared pipeline
    ports as the standalone screen, including error explanation and cluster
    execution actions. Modules without declared ports omit the chaining
    controls.

    :param key: the folded module's registry key.
    :param host_window: the main window, when there is one to connect to.
    :returns: the screen.
    """
    from .app_screen import AppScreen

    screen = AppScreen(app_key=key)
    connect_host(screen, host_window)
    try:
        from ..chaining import install_chaining
        install_chaining(screen)
    except Exception:
        LOG.debug("No chaining strip for the folded %s", key, exc_info=True)
    return screen


def build_registered_screen(key: str,
                            host_window: Optional[QWidget] = None) -> QWidget:
    """The screen NAVIGATION builds for ``key``, for a fold button to open.

    Folded modules that still hold a registry row are reached two ways --
    the button on their host's masthead and the command palette -- and
    the two must land on the same screen. Asking the window to build it
    is what guarantees that: `_build_screen` is the one place that knows
    which keys have a dedicated screen class, which are catalogue-driven
    :class:`AppScreen` screens, and which come from a plugin.

    The alternative was a table here mapping ten keys to ten classes,
    which is the same knowledge written a second time and free to drift
    from the first.

    Falls back to :func:`build_settings_screen` when there is no window
    to ask -- the headless and unit-test path, where a settings screen is
    what the catalogue-driven modules would have produced anyway.

    :param key: the folded module's registry key.
    :param host_window: the main window, when there is one.
    :returns: the screen.
    """
    build = getattr(host_window, "_build_screen", None)
    if callable(build):
        try:
            return build(key)
        except Exception:
            LOG.debug("The window could not build %s; falling back to its "
                      "settings screen", key, exc_info=True)
    return build_settings_screen(key, host_window)


def show_as_window(screen: QWidget, owner: Optional[QWidget],
                   title: str) -> QWidget:
    """Show ``screen`` as its own window, owned by ``owner``'s window.

    This fallback is used when the host cannot display the screen as a page;
    see :func:`show_as_page`. The main window owns the resulting window so
    that Qt retains it for the application's lifetime and closes it with the
    application.
    """
    parent = owner.window() if owner is not None else None
    screen.setParent(parent, Qt.Window)
    screen.setWindowTitle(title)
    screen.resize(*FOLD_WINDOW_SIZE)
    # A fallback window bypasses MainWindow._theme_screen. Its builder may
    # have imported another late registrar, so give the window its own scope
    # before the first show just as the ordinary screen-host path does.
    from ..theme import ensure_widget_qss_applied
    ensure_widget_qss_applied(root=screen)
    screen.show()
    screen.raise_()
    screen.activateWindow()
    return screen


# ---------------------------------------------------------------------------
# A fold that is a page on its host, rather than a window over it
# ---------------------------------------------------------------------------
#
# "some new module could take space above the console or become a tab.
# anything to integrate the new module naturally ... if you cannot find
# any other way, then do your new window idea."
#
# A folded module that has a screen of its own -- a bundle browser, a SHAP
# panel, a kappa table, a settings form and its Run button -- is a VIEW ON
# THE HOST'S DATA rather than a set of settings the host already has. So
# it becomes a page beside the host's own: the module itself, whole, but
# inside the window the user is already in rather than floating over it.
#
# NOTHING IS REIMPLEMENTED AND NOTHING IS LOST. It is the same widget the
# window held, with the same signals wired to the same host; only where it
# is mounted changes. Closing its tab keeps the built screen, so the state
# it had -- a loaded bundle, a typed path, a finished run -- is still
# there when the button is pressed again, which is more than the window
# managed.

#: The objectName the host's page strip carries, so one QSS rule can style
#: every one of them and tests can find it without knowing the host.
#:
#: These tabs ARE the page: the module's own screen is what sits under
#: them, so they take the treatment the other full-page tab strips take
#: (Classifier Evaluation, Run History, the Gate Editor) rather than the
#: shipped ``QTabWidget::pane`` rules, whose raw-hex fill would sit over
#: the theme as a flat opaque slab that no opacity setting can reach.
PAGES_NAME = "FoldPages"


def _pages_qss(palette: dict, opacity) -> str:
    """QSS for the page strip, registered through the theme seam."""
    from ..theme import page_tabs_qss
    return page_tabs_qss(PAGES_NAME, palette, opacity)


def _ensure_pages_qss(screen: QWidget) -> None:
    """Register the page strip's block and make sure it is live.

    Registered at the first page rather than at this module's import, and
    for the reason :func:`spacr.qt.theme.ensure_widget_qss_applied` was
    written: a block registered after the application stylesheet was
    composed is simply not in it, and the widget falls through to the
    blanket window fill -- a solid black rectangle on the dark theme. A
    fold page is opened long after launch by definition, so its block is
    installed on the host screen here before the first strip exists.

    ``replace=True``: this module owns the name, so being called again
    re-registers rather than raising and leaving the strip unstyled.
    """
    try:
        from ..theme import ensure_widget_qss_applied, register_widget_qss

        register_widget_qss(PAGES_NAME, _pages_qss, replace=True)
        ensure_widget_qss_applied(PAGES_NAME, root=screen)
    except Exception:
        LOG.debug("Could not register the fold page QSS", exc_info=True)


# Registered at import as well, so a session that builds its stylesheet
# before any fold page exists already carries the rule. The import-time
# registration is what ``theme.WIDGET_QSS_MODULES`` loads; the call above
# is what covers a page made after the sheet was composed. Both, because
# either alone leaves one of the two orders unstyled.
try:
    from ..theme import register_widget_qss as _register_widget_qss

    _register_widget_qss(PAGES_NAME, _pages_qss, replace=True)
except Exception:
    LOG.debug("Could not register the fold page QSS at import", exc_info=True)


def _page_body(screen: QWidget) -> Optional[QWidget]:
    """The widget that IS the host's page -- everything below its masthead.

    Found as the one child of the screen's top-level layout that was given
    the stretch, which is the body on every screen here: the settings /
    runtime splitter on a generic module screen, the editor stack on Make
    Masks, the grid splitter on Annotate. Derived rather than listed by
    attribute name so a screen that renames its body does not silently
    lose its pages.
    """
    layout = screen.layout() if screen is not None else None
    if layout is None or not hasattr(layout, "stretch"):
        return None
    for index in range(layout.count()):
        item = layout.itemAt(index)
        widget = item.widget() if item is not None else None
        if widget is not None and layout.stretch(index) > 0:
            return widget
    return None


def host_pages(screen: QWidget, title: str = "") -> Optional[QTabWidget]:
    """Return the host's page strip, creating it when first requested.

    The host body becomes a non-closable first page and retains its layout
    stretch. Pages opened for folded modules can be closed without destroying
    their underlying screens.

    :param screen: the host module's screen.
    :param title: caption for the host page. If omitted, use
        ``screen._fold_page_title`` and then the registered application name.
    :returns: the page strip, or ``None`` when the host has no page body.
    """
    existing = getattr(screen, "_fold_pages", None)
    if isinstance(existing, QTabWidget):
        # Building the next folded module may have registered more QSS since
        # this strip was created. Refresh the host's one owned suffix before
        # the new child is mounted; otherwise only the first fold is styled.
        _ensure_pages_qss(screen)
        return existing
    body = _page_body(screen)
    layout = screen.layout() if screen is not None else None
    if body is None or layout is None:
        return None
    index = layout.indexOf(body)
    if index < 0:
        return None
    stretch = layout.stretch(index)
    name = (title or str(getattr(screen, "_fold_page_title", "") or "")
            or folded_module_title(getattr(screen, "app_key", "") or ""))
    _ensure_pages_qss(screen)
    pages = QTabWidget(screen)
    pages.setObjectName(PAGES_NAME)
    pages.setDocumentMode(True)
    pages.setTabsClosable(True)
    layout.removeWidget(body)
    pages.addTab(body, name)
    # The host's own page has no close button: there is nothing behind it.
    #
    # HIDDEN, NOT CLEARED. `QTabBar.setTabButton(index, side, None)`
    # destroys the button that was there, and the tab bar goes on holding
    # a pointer to it -- which lands as a segmentation fault in whatever
    # the process happens to be doing when that memory is next touched,
    # three tests away from the line that caused it. Hiding it leaves
    # ownership where Qt put it.
    bar = pages.tabBar()
    for side in (QTabBar.RightSide, QTabBar.LeftSide):
        button = bar.tabButton(0, side)
        if button is not None:
            button.hide()
    # THE APPLICATION'S CLOSE MARK, NOT THIS STRIP'S. The host page's
    # button stays hidden -- `install_close_marks` carries that across --
    # so folding still costs the host nothing. See
    # `theme.install_close_marks`.
    install_close_marks(pages, tooltip=tr("Close"))
    pages.tabCloseRequested.connect(
        partial(_close_fold_page, pages))
    layout.insertWidget(index, pages, stretch)
    screen._fold_pages = pages
    return pages


def _close_fold_page(pages: QTabWidget, index: int) -> None:
    """Take a folded page off the strip, keeping the screen it held.

    The widget is only removed from the strip, never destroyed: pressing
    the button again puts the SAME screen back, with whatever it had
    loaded still loaded.
    """
    if index <= 0:
        return
    page = pages.widget(index)
    pages.removeTab(index)
    if page is not None:
        page.setParent(None)


def hide_as_page(screen: QWidget, host: Optional[QWidget]) -> bool:
    """Take ``screen`` off ``host``'s page strip, keeping the screen.

    The counterpart to :func:`show_as_page`, for a control that closes what
    it opened. It takes the same route the strip's own close mark takes --
    :func:`_close_fold_page` -- so a page closed by a switch and a page
    closed by its cross leave the module in the same state, loaded and off
    the strip rather than destroyed.

    :param screen: the folded module's widget.
    :param host: the host module's screen.
    :returns: True when a page was taken off the strip, False when the host
        has no strip or the widget is not on it.
    """
    pages = getattr(host, "_fold_pages", None) if host is not None else None
    if not isinstance(pages, QTabWidget):
        return False
    index = pages.indexOf(screen)
    # Never index 0: that is the host's own body, which has no close mark
    # for the same reason -- there is nothing behind it.
    if index <= 0:
        return False
    _close_fold_page(pages, index)
    return True


def show_as_page(screen: QWidget, host: Optional[QWidget],
                 title: str) -> Optional[QWidget]:
    """Add ``screen`` to ``host``'s page strip and select it.

    :param screen: widget for the folded module.
    :param host: the host module's screen.
    :param title: page caption, normally the folded module's display name.
    :returns: ``screen``, or ``None`` when the host cannot contain pages.
    """
    pages = host_pages(host) if host is not None else None
    if pages is None:
        return None
    index = pages.indexOf(screen)
    if index < 0:
        index = pages.addTab(screen, title)
        # Qt builds its own small close button for a new tab. Ask for the
        # application's mark here rather than waiting for the strip's
        # watcher, so the page never appears carrying the wrong one.
        install_close_marks(pages, tooltip=tr("Close"))
    # THE MODULE'S OWN MARK ON ITS TAB. A folded module gave up its tile,
    # and the icon is the thing a user already associates with it -- so a
    # page carrying only a title asks them to re-learn a name for
    # something they could recognise at a glance. The key is taken from
    # the screen itself, so a page opened by any host is marked the same.
    key = str(getattr(screen, "app_key", "") or "")
    if key:
        try:
            from .. import iconset

            # See the note in `widgets/fold_strip.py`: resolving by
            # filename alone ignores `_ICON_OVERRIDES` and hands a
            # borrowing module the wrong picture.
            from ..app import _icon_for_app
            icon = _icon_for_app(key)
            if icon is not None and not icon.isNull():
                pages.setTabIcon(index, icon)
        except Exception:                                # noqa: BLE001
            LOG.debug("no mark for the %s page", key, exc_info=True)
    pages.setCurrentIndex(index)
    return screen


class FoldOpener:
    """Open a folded module and reuse its screen between activations.

    The opener is an object so the fold strip controls its lifetime through
    the button connection. The module appears as a page on its host when the
    host supports pages, or as a separate window otherwise. Its screen is
    constructed once and retained, preventing duplicate database handles or
    job runners and preserving loaded state when the user changes pages.

    :param screen: the host screen the button sits on.
    :param key: the folded module's registry key.
    :param build: called with the main window (or None) and returning the
        folded module's screen.
    """

    def __init__(self, screen: QWidget, key: str,
                 build: Callable[[Optional[QWidget]], QWidget]) -> None:
        """Record how to build one folded module's page, without building it.

        :param screen: the host screen the page is opened on.
        :param key: the folded module's registry key.
        :param build: called with a parent to build the page, on first open.
        """
        self.screen = screen
        self.key = key
        self._build = build
        #: The module's screen, once built. Named ``window`` for the
        #: callers that predate pages; it is a page on the host wherever
        #: the host can carry one.
        self.window: Optional[QWidget] = None

    def open(self, _checked: bool = False) -> Optional[QWidget]:
        """Show the folded module; raise it if it is already up."""
        built = self.window
        if built is not None:
            try:
                built.isVisible()
            except RuntimeError:
                # Qt deleted the C++ side under us. Build a fresh one
                # rather than try to resurrect a dangling wrapper.
                built = self.window = None
        if built is None:
            host_window = (self.screen.window()
                           if self.screen is not None else None)
            try:
                built = self._build(host_window)
            except Exception:
                LOG.exception("Could not open the folded module %r", self.key)
                return None
        title = folded_module_title(self.key)
        shown = show_as_page(built, self.screen, title)
        if shown is None:
            shown = show_as_window(built, self.screen, title)
        elif not shown.isVisible():
            shown.show()
        self.window = shown
        return shown


def install_fold_strip(screen: QWidget, host_key: str,
                       folded: Sequence[str],
                       builders: Dict[str, Callable[[Optional[QWidget]],
                                                    QWidget]]
                       ) -> Optional[FoldStrip]:
    """Install buttons for ``folded`` modules on ``screen``'s masthead.

    Repeated calls return the existing strip. Construction failures are
    contained so the host screen remains usable without the optional strip.

    :param screen: the host module's screen.
    :param host_key: registry key required on ``screen``.
    :param folded: the folded modules' keys, in strip order.
    :param builders: mapping from module key to a screen factory.
    :returns: the installed strip, or ``None`` when the screen is not the
        requested host, has no masthead, contains no eligible modules, or
        strip construction fails.
    """
    if getattr(screen, "app_key", None) != host_key:
        return None
    existing = getattr(screen, "_fold_strip", None)
    if isinstance(existing, FoldStrip):
        return existing
    header = getattr(screen, "_header", None)
    if header is None or not hasattr(header, "add_trailing"):
        return None
    openers = []
    entries = []
    for key in folded:
        build = builders.get(key)
        if build is None:
            continue
        opener = FoldOpener(screen, key, build)
        openers.append(opener)
        entries.append((key, opener.open))
    if not entries:
        return None
    try:
        strip = FoldStrip(entries, header)
        for key, _callback in entries:
            restate_fold_button(strip.button_for(key), key)
        header.add_trailing(strip)
    except Exception:
        LOG.debug("Could not build the fold strip for %s", host_key,
                  exc_info=True)
        return None
    # The openers outlive this call only because the screen holds them.
    screen._fold_openers = openers
    screen._fold_strip = strip
    return strip


def _build_barcode_qc(host_window: Optional[QWidget]) -> QWidget:
    """Barcode QC's own screen: the settings-driven module, unchanged."""
    return build_settings_screen("barcode_qc", host_window)


#: One builder per folded module. :func:`install_folds` walks
#: :data:`FOLDED_APPS` and looks each key up here, so the strip's order
#: and the strip's contents cannot disagree.
BUILDERS: Dict[str, Callable[[Optional[QWidget]], QWidget]] = {
    "barcode_qc": _build_barcode_qc,
}


def install_folds(screen: QWidget) -> Optional[FoldStrip]:
    """Put Map Barcodes' fold strip and its live barcode search on ``screen``.

    The search is installed here rather than through a seam of its own because
    this is the call every route to the Map Barcodes screen already passes
    through. A panel installed anywhere else would be present when the screen
    is reached one way and missing when it is reached another.

    :param screen: the screen to install into.
    :returns: the fold strip, or None when this screen hosts no folds.
    """
    install_barcode_search(screen)
    return install_fold_strip(screen, HOST_KEY, FOLDED_APPS, BUILDERS)


# ---------------------------------------------------------------------------
# Reaching the screens the window builds
# ---------------------------------------------------------------------------
#
# A host screen is the generic ``AppScreen``, which knows nothing about
# who folded into it and should not have to. The strips are hung on it
# from outside, as each screen reaches the window's stack -- the route
# :mod:`spacr.qt.preview_registry` and :mod:`spacr.qt.recipes` take to
# put their own controls on a screen they do not own.

#: Host app key → the module in this package that owns its fold strip.
#: A host absent from here has no folds; one pass over the stack serves
#: all of them, so a new fold is a line here and an ``install_folds`` in
#: the host's own module. Screens that build their own masthead (Annotate
#: is one) build their strip with it and are not listed.
FOLD_HOST_MODULES: Dict[str, str] = {
    "map_barcodes": "map_barcodes",
    "classify_merged": "classify",
    "measure": "measure",
    # Mask Generation's two folds are settings categories rather than
    # windows, but they reach their host by the same walk: the screen is
    # the generic `AppScreen`, built by the window, and the strip is hung
    # on it from outside.
    "mask": "mask",
    "regression": "regression",
    "umap": "image_umap",
    # Instruction 318's folds. Each of these hosts gained two buttons for
    # modules that used to hold a Home tile of their own -- a tile says
    # "start here", and none of the six is a job anyone sets out to do:
    # they are second views of something the host is already showing.
    "graph_builder": "graph_builder",
    "db_browser": "db_browser",
    "qc_dashboard": "qc_dashboard",
    # Import: Format Converter and External Masks, folded onto the screen
    # that was Import Project.
    "foreign": "foreign",
}


def install_folds_on(screen: QWidget) -> Optional[FoldStrip]:
    """Install the fold strip declared for ``screen``'s application key.

    The owning module is selected through :data:`FOLD_HOST_MODULES`. Screens
    without a fold declaration are left unchanged.
    """
    key = getattr(screen, "app_key", None)
    module_name = FOLD_HOST_MODULES.get(key) if key else None
    if not module_name:
        return None
    try:
        from importlib import import_module
        module = import_module(f"{__package__}.{module_name}")
        return module.install_folds(screen)
    except Exception:
        LOG.debug("Could not install the folds for %s", key, exc_info=True)
        return None


class _StackWatcher(QObject):
    """Gives each host screen its fold strip as the stack reaches it."""

    def __init__(self, window) -> None:
        """Watch a window's stack and install into each screen as it is shown.

        :param window: the main window. Its stack is read at install time,
            not here, so this works for screens created after the watcher --
            and it is the QObject PARENT, so a currentChanged arriving during
            teardown cannot reach a watcher holding a deleted stack.
        """
        super().__init__(window)
        self._window = window

    def on_current_changed(self, _index: int = 0) -> None:
        """Install into whatever screen the stack just switched to."""
        self.install_current()

    def install_current(self) -> Optional[FoldStrip]:
        """Install into the stack's current widget, if it hosts folds."""
        try:
            screen = self._window._stack.currentWidget()
        except Exception:
            return None
        if screen is None:
            return None
        return install_folds_on(screen)


def install_window_hooks(window) -> Optional[_StackWatcher]:
    """Install fold strips as screens become current in ``window``.

    Repeated calls return the existing stack watcher rather than connecting
    an additional callback.

    :param window: the main window.
    :returns: the watcher, or None when the window has no screen stack.
    """
    stack = getattr(window, "_stack", None)
    if stack is None:
        return None
    existing = getattr(window, "_fold_watcher", None)
    if isinstance(existing, _StackWatcher):
        return existing
    watcher = _StackWatcher(window)
    try:
        stack.currentChanged.connect(watcher.on_current_changed)
    except Exception:
        LOG.debug("Could not follow the screen stack", exc_info=True)
        return None
    window._fold_watcher = watcher
    # The first screen is already current when this runs, and no
    # currentChanged is coming for it.
    QTimer.singleShot(0, watcher.install_current)
    return watcher


# ---------------------------------------------------------------------------
# A fold that is not a window: the module as settings categories on its host
# ---------------------------------------------------------------------------
#
# A WINDOW IS THE LAST RESORT. Some folded modules are not a second screen
# at all -- they are the host's own pipeline with a gate turned on and a
# few extra knobs. Timelapse and Motility on Mask Generation are the case
# the maintainer named: "these buttons just need to toggle the visability
# of their settings categories as they share the rest with [the host]".
#
# So the button reveals the module's own settings CATEGORIES on the host's
# form and turns the pipeline flag they belong to on. Nothing opens,
# nothing is replaced, and the settings the two modules share are edited
# once, in the place the user is already looking.


def _widget_keys(model) -> Dict[int, str]:
    """``id(widget) -> setting key`` for one settings model.

    Keyed on identity rather than on the widget itself because a Qt widget
    is not reliably hashable across wrapper objects, and because this is
    only ever asked about widgets the same model just built.
    """
    return {id(widget): key
            for key, widget in getattr(model, "_widgets", {}).items()}


class CategoryFold:
    """One folded module, mounted on its host as extra settings categories.

    The module's settings form is built through the same path as its own
    screen. Categories containing settings absent from the host are then
    mounted on the host and remain hidden until the fold is enabled.

    The fold switch exclusively controls the mounted categories' visibility.
    They are omitted from ``_settings_sections`` because the maturity filter
    and settings search also change the visibility of sections in that list.
    Consequently, settings search does not include an inactive folded
    category.

    Settings already provided by the host are not duplicated. Because
    :meth:`collect` indexes controls by setting name, duplicate controls would
    create ambiguous values and could replace values entered on the host.
    Existing keys are therefore removed from the folded form, and categories
    with no remaining controls are not mounted.

    :param screen: the host module's ``AppScreen``.
    :param key: the folded module's registry key.
    :param gates: setting names the host's pipeline reads to decide
        whether to do what this module does. They are forced True while
        the fold is on and False while it is off, so the run matches what
        the form is showing.
    """

    def __init__(self, screen: QWidget, key: str,
                 gates: Sequence[str] = ()) -> None:
        """Record one settings category that folds in and out of the host.

        :param screen: the host screen whose form the category joins.
        :param key: the fold's key.
        :param gates: the settings whose values decide whether it applies.
        """
        self.screen = screen
        self.key = key
        self.gates: Tuple[str, ...] = tuple(gates)
        self.sections: list = []
        self.model = None
        self.settings_keys: Tuple[str, ...] = ()
        self._active = False

    # -- mounting ------------------------------------------------------
    def mount(self) -> bool:
        """Build the module's categories and put them on the host, hidden.

        :returns: True when at least one category was mounted. False means
            the host has no settings form, or the folded module has
            nothing this host does not already show -- both of which leave
            the host exactly as it was.
        """
        host_model = getattr(self.screen, "_settings_model", None)
        content = getattr(self.screen, "_settings_content", None)
        layout = content.layout() if content is not None else None
        if host_model is None or layout is None:
            return False
        from .settings_model import (SettingsWidgets,
                                     keys_hidden_by_their_object)

        # ONLY WHAT THIS FOLD ADDS. The loop below keeps a row exactly when
        # the host does not already hold its key, so building the rest was
        # 96% waste: the timelapse fold on the mask screen built 364
        # settings to keep 14, at 1,148 ms on every module open. The host's
        # own keys are skipped up front instead.
        already = set(getattr(host_model, "_widgets", {}))
        model = SettingsWidgets(self.key, parent=content, skip_keys=already)
        built = model.build_sections()
        held = set(getattr(host_model, "_widgets", {}))
        # AND NOT WHAT THIS RUN HAS NO OBJECT FOR. The host builds every
        # object's rows and hides the ones whose channel is unset, so a
        # fold that mounted them would put the host's own hidden category
        # on the form a second time -- the timelapse fold mounted a second
        # PATHOGEN SEGMENTATION card on mask for the one pathogen setting
        # mask's registry spells differently. Judged against the HOST's
        # channels, because it is the host's run these rows would join.
        # THE HOST'S KEYS GO IN WITH THE FOLD'S. The rule gates a role only
        # when that role's switch is on the same panel, so that it never
        # hides a row whose switch lives on a screen the user cannot reach.
        # Here the switch IS reachable -- it is on the host, one card up --
        # and the panel these rows would join is the union of the two.
        try:
            run_hides = set(keys_hidden_by_their_object(
                set(model._widgets) | held,
                host_model._object_visibility_settings()))
            run_hides &= set(model._widgets)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not tell which of %s the run has an object for",
                      self.key, exc_info=True)
            run_hides = set()
        by_widget = _widget_keys(model)
        mounted_keys: list = []
        for source in built:
            title = source.title if hasattr(source, "title") else source[0]
            rows = source.rows if hasattr(source, "rows") else source[1]
            own = [(label, widget) for label, widget in rows
                   if by_widget.get(id(widget)) not in held
                   and by_widget.get(id(widget)) not in run_hides
                   and by_widget.get(id(widget)) is not None]
            if not own:
                continue
            section = self._build_section(str(title), own, by_widget)
            # Before the trailing stretch the panel ends with, or the
            # categories would be pushed off the bottom of the column.
            layout.insertWidget(max(0, layout.count() - 1), section)
            section.setVisible(False)
            self.sections.append(section)
            mounted_keys.extend(by_widget[id(widget)] for _label, widget in own)
        if not self.sections:
            return False
        self.model = model
        self.settings_keys = tuple(mounted_keys)
        # THE HOST NOW COLLECTS THEM. `collect()` walks `_widgets`, so a
        # control that is on the host's form and not in this map is a
        # control the run never sees.
        host_model._widgets.update(
            {key: model._widgets[key] for key in mounted_keys})
        # And the module's settings that have no control -- the ones its
        # own screen does not render either -- ride along as defaults, so
        # the pipeline is handed the same dict its own module would have
        # handed it. The gates are excluded: they are this fold's switch,
        # and their value is decided by the button rather than inherited.
        for name, value in getattr(model, "_defaults", {}).items():
            if name in self.gates:
                continue
            host_model._defaults.setdefault(name, value)
        # AND SO DO THE ROWS THE RUN HAS NO OBJECT FOR. They get no control
        # -- the host already shows that object's category -- but the
        # module's pipeline still reads them by name, so their value rides
        # along exactly as a setting with no control does.
        if run_hides:
            try:
                values = model.collect()
            except Exception:                                # noqa: BLE001
                LOG.debug("could not read %s's unmounted values", self.key,
                          exc_info=True)
                values = {}
            for name in run_hides:
                if name in self.gates or name in held:
                    continue
                if name in values:
                    host_model._defaults.setdefault(name, values[name])
        self.set_active(False)
        return True

    def _build_section(self, title: str, rows, by_widget: Dict[int, str]):
        """One category card, wired to the HOST's help strips.

        The label carries the setting's key and its documentation HTML and
        is filtered by the host screen, so hovering a folded setting fills
        the same hint strip every other setting on the form fills. A row
        wired to the module's own screen would answer into a screen
        nobody is looking at.
        """
        from ..widgets.section import Section
        from .app_screen import settings_section_maturity
        from .settings_model import category_tooltip

        section = Section(title)
        section.setProperty("settingsCategorySource", title)
        section.set_maturity(settings_section_maturity(self.key, title))
        section.set_hint(category_tooltip(self.key, title))
        for label, widget in rows:
            name = by_widget[id(widget)]
            caption = QLabel(str(label))
            caption.setCursor(Qt.WhatsThisCursor)
            caption.setProperty("settingKey", name)
            caption.setProperty("settingsAppKey", self.key)
            html = widget.toolTip()
            caption.setProperty("apiTooltipHtml", html)
            caption.setProperty("apiTooltipDisplayRole", "tooltip")
            # The help lives on the label, as it does on every other row:
            # a tooltip on the field itself pops while the user is typing
            # into it.
            widget.setToolTip("")
            caption.installEventFilter(self.screen)
            section.add_row(caption, widget, info_widget=None,
                            wrap_label=True)
        return section

    # -- switching -----------------------------------------------------
    @property
    def active(self) -> bool:
        """Whether this module is currently part of the host's run."""
        return self._active

    def set_active(self, on: bool) -> None:
        """Show or hide this module's categories on the host's form."""
        self._active = bool(on)
        for section in self.sections:
            section.setVisible(self._active)

    def collect(self) -> Dict[str, object]:
        """Return the settings contributed by this fold.

        The host model collects both host and folded settings. This method
        selects only the keys mounted by the current fold.
        """
        host_model = getattr(self.screen, "_settings_model", None)
        if host_model is None:
            return {}
        values = host_model.collect()
        return {name: values[name] for name in self.settings_keys
                if name in values}


class CategoryFoldSet:
    """Manage category folds and pipeline gates for one host screen.

    A host declares its folded modules and their associated gates. This class
    mounts their settings, builds the masthead controls, and synchronizes gate
    values with the active folds.

    :param screen: the host module's ``AppScreen``.
    :param folds: ``key -> gate names``, in the order the strip draws them.
    :param implies: dependencies as ``key -> keys``. Activating a dependent
        fold also activates its prerequisites. For example, the motility
        assay activates the timelapse branch required for tracking.
    """

    def __init__(self, screen: QWidget,
                 folds: Dict[str, Sequence[str]],
                 implies: Optional[Dict[str, Sequence[str]]] = None) -> None:
        """Build the set of category folds this screen offers.

        :param screen: the host screen.
        :param folds: each fold's key mapped to the settings that gate it; the
            mapping's order is the order the strip shows them in.
        :param implies: folds that turning one on also turns on.
        """
        self.screen = screen
        self.order: Tuple[str, ...] = tuple(folds)
        self.implies = {key: tuple(values)
                        for key, values in (implies or {}).items()}
        self.folds: Dict[str, CategoryFold] = {
            key: CategoryFold(screen, key, gates)
            for key, gates in folds.items()}
        self.strip: Optional[FoldStrip] = None

    # -- building ------------------------------------------------------
    def mount(self) -> Tuple[str, ...]:
        """Mount each fold's categories on the host in a hidden state.

        :returns: keys of folds that contributed at least one setting.
        """
        mounted = []
        for key in self.order:
            if self.folds[key].mount():
                mounted.append(key)
            else:
                LOG.debug("%s folds nothing new into %s", key,
                          getattr(self.screen, "app_key", "?"))
                self.folds.pop(key, None)
        self.order = tuple(mounted)
        self.apply_gates()
        return self.order

    def build_strip(self, parent: Optional[QWidget] = None
                    ) -> Optional[FoldStrip]:
        """The masthead strip: one checkable button per mounted fold."""
        if not self.order:
            return None
        entries = [(key, partial(self.set_active, key), True)
                   for key in self.order]
        strip = FoldStrip(entries, parent)
        for key in self.order:
            restate_fold_button(strip.button_for(key), key)
        self.strip = strip
        return strip

    # -- switching -----------------------------------------------------
    def set_active(self, key: str, on: bool) -> None:
        """Set a fold's state and update its dependency relationships.

        Enabling a dependent fold enables its prerequisites. Disabling a
        prerequisite disables active dependents. Available strip buttons are
        updated through their normal signal path to keep display and form
        state synchronized.
        """
        fold = self.folds.get(key)
        if fold is None:
            return
        fold.set_active(on)
        if on:
            for other in self.implies.get(key, ()):
                self._set_button(other, True)
        else:
            for other, needed in self.implies.items():
                if key in needed and self.is_active(other):
                    self._set_button(other, False)
        self.apply_gates()

    def is_active(self, key: str) -> bool:
        """Whether ``key``'s categories are showing and its gate is on."""
        fold = self.folds.get(key)
        return bool(fold is not None and fold.active)

    def _set_button(self, key: str, on: bool) -> None:
        """Move one fold's button, or the fold itself when there is no strip."""
        button = self.strip.button_for(key) if self.strip is not None else None
        if button is not None:
            # The toggle comes back through `set_active`, so the fold and
            # everything it implies are switched by the same path a user
            # pressing the button takes.
            button.setChecked(bool(on))
            return
        fold = self.folds.get(key)
        if fold is not None:
            fold.set_active(on)

    def apply_gates(self) -> Dict[str, bool]:
        """Derive and store gate values from all active folds.

        Values are recomputed collectively because multiple folds may share a
        gate. Gates are stored in the settings model defaults rather than in
        duplicate form controls; ``collect()`` includes defaults for keys
        without widgets.
        """
        model = getattr(self.screen, "_settings_model", None)
        values = {gate: False
                  for fold in self.folds.values() for gate in fold.gates}
        for fold in self.folds.values():
            if fold.active:
                for gate in fold.gates:
                    values[gate] = True
        if model is not None:
            for gate, value in values.items():
                model._defaults[gate] = value
        return values

    def sync_from_settings(self, settings: Dict[str, object]) -> Tuple[str, ...]:
        """Synchronize fold states with values in a loaded settings mapping.

        Fold gates have no dedicated widgets, so bulk settings application
        cannot update them through the form. This method reads the gate values
        directly and activates the corresponding folds.

        :param settings: the dict that was applied.
        :returns: the keys switched on by it.
        """
        turned_on = []
        for key in self.order:
            gates = self.folds[key].gates
            wanted = bool(gates) and all(
                _reads_as_true(settings.get(gate)) for gate in gates)
            if wanted != self.is_active(key):
                self._set_button(key, wanted)
            if wanted:
                turned_on.append(key)
        return tuple(turned_on)


def _reads_as_true(value) -> bool:
    """Whether a settings value means yes, however it was written down.

    A settings CSV round-trips through text, so the gate arrives as the
    string ``"True"`` as often as it arrives as the bool.
    """
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


# ---------------------------------------------------------------------------
# The live barcode search: settings read off the reads instead of guessed
# ---------------------------------------------------------------------------
#
# "implementing a search function that searches for barcodes and sets
# settings automatically, and the ability to find more than 3, i.e. an
# arbitrary number of barcodes ... The automatic settings mode should be like
# live mode ... the user should also see the matching barcodes in a text
# window with 1 read per row and the different barcodes matches visualized by
# coloring different barcodes different colors."
#
# THE RUN THIS EXISTS TO PREVENT. The Map Barcodes tutorial could not be
# written because a real paired run produced 8,611 consensus rows and zero
# mapped counts, and finished normally while doing it. The reads of a pair are
# read from opposite ends of one fragment, so a barcode that is plain in one
# mate is reverse complemented in the other, and spaCR ships its reference
# tables in one orientation only. Measured on that run, per barcode table and
# per mate:
#
#     barcode type   R1 plain   R1 RC    R2 plain   R2 RC    chance
#     gRNA              0.1%    79.7%      79.0%     0.2%     ~0%
#     column            0.9%    27.9%       2.7%     1.0%     5.2%
#     row               6.8%     6.9%       6.8%     1.3%     6.9%
#
# Reading column barcodes off R1 against the shipped table therefore matched
# 0.9% of reads against a 5.2% coincidence rate -- below noise -- and the run
# mapped nothing without ever saying so.
#
# WHY EVERY RATE ON THIS PANEL IS PRINTED BESIDE ITS CHANCE RATE. Look at the
# row barcodes in that table. They match 6.8% of reads and they are not there
# at all: thirty-two barcodes of eight bases scanned across a hundred and
# fifty base read match somewhere by pure coincidence in
# 32 * (150 - 8 + 1) / 4**8 of reads, which is 6.98%. A panel that showed
# "row barcodes found, 6.8%" would send someone hunting for a bug that does
# not exist, or would auto-configure a mapping that produces garbage. So the
# chance rate has a column of its own directly beside the observed one, the
# enrichment between them has a third, and no verdict says a table is present
# unless the engine's own thresholds clear coincidence by a wide margin. That
# pairing is the feature. Dropping the chance column to save width would turn
# this panel back into the thing it was built to replace.
#
# HOW IT RUNS WITHOUT FREEZING ANYTHING. The engine hands back a complete
# report after every chunk of reads, so this submits one chunk at a time
# through `JobRunner` and submits the next when the previous one lands on the
# GUI thread -- the same route the live preview takes, which is also what puts
# the work in the process-wide run registry that turns the activity spinner.
# Nothing here reads a file on the GUI thread: the folder scan, the reference
# tables, the reads and the annotation all happen inside a submitted job and
# come back as plain data. Measured against the run above, a chunk of two
# thousand reads from each of two mates against four reference tables takes
# 0.25 s, so a twenty thousand read sample settles in about two and a half
# seconds and refines visibly while it does.
#
# AND APPLYING IS A SEPARATE PRESS. The search proposes; the user disposes.
# Silently rewriting settings somebody typed is not acceptable even when the
# rewrite is right, so the proposal is rendered as old value beside new value
# and nothing reaches the form until the Apply button is pressed.

#: Object name of the card the search sits in, so one QSS rule can reach it
#: and a test can find it without knowing the screen's layout.
SEARCH_CARD_NAME = "BarcodeSearchCard"

#: Object name of the masthead toggle that reveals the card.
SEARCH_TOGGLE_NAME = "BarcodeSearchToggle"

#: Object name of the findings table.
SEARCH_TABLE_NAME = "BarcodeSearchFindings"

#: The findings table's columns, in the order they are drawn.
#:
#: THE OBSERVED RATE AND THE CHANCE RATE ARE ADJACENT AND STAY ADJACENT. A
#: reader who sees one without the other believes a coincidence, which is the
#: failure the whole panel is built against, so the two are neighbours here
#: rather than at opposite ends of a wide table where a horizontal scroll can
#: separate them.
SEARCH_COLUMNS: Tuple[str, ...] = (
    "Barcode", "Mate", "Orientation", "Observed", "By chance",
    "Enrichment", "Offset", "Verdict",
)

#: How many reads the text window shows. Enough that a barcode landing in the
#: same columns on row after row is obvious, small enough that sampling them
#: costs a fraction of one search chunk.
READ_SAMPLE_READS = 200

#: What the settings hold when nobody has chosen a sequencing folder yet.
_UNSET_SOURCE = {"", "path", "none", "None"}

#: The three reference tables spaCR shipped before a run could name its own,
#: as setting key and the barcode role the table fills.
_SHIPPED_REFERENCE_KEYS: Tuple[Tuple[str, str], ...] = (
    ("column_csv", "column"),
    ("grna_csv", "grna"),
    ("row_csv", "row"),
)


@dataclass(frozen=True)
class BarcodeSearchPlan:
    """What one search is going to read, worked out from the settings form.

    Planning is separated from searching because the two fail in different
    ways and the user needs to be told which one happened. A plan that cannot
    be made is a sentence about the form, such as a folder holding no
    sequencing files, and it is worth showing before any read is opened; a
    search that finds nothing is a measurement, and means something else
    entirely.

    :ivar fastq_files: the sequencing files to read, as a mapping from the
        label each carries in the report to its path. The labels are the mate
        names, so a report can say which mate carried which barcode.
    :ivar reference_tables: one entry per reference table, holding the name it
        is reported under, the path it is read from, and the barcode role it
        fills.
    :ivar anchor: the fixed vector sequence the mapping run locates its window
        by, searched alongside the tables so that offsets can be expressed
        relative to it. Empty when the settings name none.
    :ivar sample: the name of the sample whose reads are being searched.
    :ivar other_samples: how many further samples sit beside it in the same
        folder. Barcode layout is a property of the library rather than of one
        sample, so one sample settles it for all of them, but the count is
        shown so nobody thinks the others were missed.
    :ivar problem: a sentence saying why this plan cannot be searched, or an
        empty string when it can.
    """

    fastq_files: Dict[str, str]
    reference_tables: Tuple[Tuple[str, str, str], ...]
    anchor: str
    sample: str
    other_samples: int
    problem: str


def _fastq_files_under(source):
    """Return the sequencing files of each sample under one path.

    :param source: a folder of sequencing files, or one sequencing file.
    :returns: a mapping from sample name to a mapping from mate label to path,
        empty when the path holds nothing that can be read.
    """
    path = str(source)
    if os.path.isdir(path):
        from ...io import parse_gz_files

        try:
            found = parse_gz_files(path)
        except Exception:
            LOG.debug("could not list the sequencing files under %s", path,
                      exc_info=True)
            return {}
        return {str(sample): {str(mate): str(file) for mate, file in
                              (mates or {}).items() if file}
                for sample, mates in (found or {}).items()}
    if os.path.isfile(path):
        stem = os.path.basename(path)
        for marker, mate in (("_R1", "R1"), ("_R2", "R2"),
                             ("_1.", "R1"), ("_2.", "R2")):
            if marker in stem:
                return {stem.split(marker)[0]: {mate: path}}
        return {stem: {"R1": path}}
    return {}


def _planned_reference_tables(settings):
    """Return the reference tables a Map Barcodes run would read.

    A run that names a barcode set is decoding whatever that set holds, which
    may be one barcode or ten, so the set is asked first and the three
    reference settings spaCR shipped are only consulted when there is no set.
    Each entry carries the role it fills, because the file name is all the
    search engine can otherwise guess a role from and a barcode called
    anything new would be left roleless.

    :param settings: the Map Barcodes settings as the form holds them.
    :returns: a tuple of entries, each holding the name the table is reported
        under, its path, and the barcode role it fills.
    """
    from ...settings import barcode_set_from_settings

    entries = []
    barcode_set = None
    try:
        barcode_set = barcode_set_from_settings(settings)
    except Exception:
        # A set that cannot be read is a settings mistake the run itself
        # reports in full. The search still has the three shipped references
        # to work with, and saying so twice helps nobody.
        LOG.debug("could not read the barcode set from the settings",
                  exc_info=True)
    if barcode_set is not None:
        for entry in barcode_set:
            path = str(getattr(entry, "csv", "") or "")
            if path and os.path.isfile(path):
                entries.append((str(entry.name), path, str(entry.name)))
        if entries:
            return tuple(entries)
    for key, role in _SHIPPED_REFERENCE_KEYS:
        path = str(settings.get(key) or "")
        if path and os.path.isfile(path):
            entries.append((os.path.splitext(os.path.basename(path))[0],
                            path, role))
    return tuple(entries)


def plan_barcode_search(settings):
    """Work out which reads and which reference tables a search should read.

    The sequencing folder of a real screen holds several samples, and one of
    them is enough. Which mate carries which barcode, which orientation the
    reference tables are stored in relative to the reads, and where in the
    read the barcodes sit are all properties of the library and the sequencing
    run rather than of a sample, so measuring them on the first sample settles
    them for every sample beside it. The others are counted rather than read,
    and the count is reported so that nobody has to wonder whether they were
    forgotten.

    Nothing here opens a read. It lists a folder and checks that the reference
    files exist, so it is cheap enough to run whenever the search button is
    pressed and it can say what is missing before a single byte is decoded.

    :param settings: the Map Barcodes settings as the form holds them.
    :returns: the plan. Its problem is a sentence when there is nothing to
        search and empty when the plan can be carried out.
    """
    settings = dict(settings or {})
    sources = settings.get("src")
    if isinstance(sources, (list, tuple, set)):
        sources = [str(item) for item in sources if str(item).strip()]
    else:
        sources = [str(sources or "").strip()]
    sources = [item for item in sources if item and item not in _UNSET_SOURCE]
    empty = BarcodeSearchPlan(
        fastq_files={}, reference_tables=(), anchor="", sample="",
        other_samples=0, problem="")
    if not sources:
        return _with_problem(empty, tr(
            "Choose the folder holding the sequencing files first. The "
            "search reads a bounded sample of them, so it needs a folder to "
            "read from."))
    samples: Dict[str, Dict[str, str]] = {}
    for source in sources:
        for sample, mates in _fastq_files_under(source).items():
            if mates:
                samples.setdefault(sample, {}).update(mates)
    if not samples:
        return _with_problem(empty, tr(
            "No sequencing files were found under {source}. The search reads "
            "gzip compressed FASTQ files named for the mate they hold.",
            source=sources[0]))
    chosen = sorted(samples)[0]
    files = {mate: samples[chosen][mate]
             for mate in sorted(samples[chosen])}
    tables = _planned_reference_tables(settings)
    plan = BarcodeSearchPlan(
        fastq_files=files,
        reference_tables=tables,
        anchor=str(settings.get("target_sequence") or "").strip().upper(),
        sample=chosen,
        other_samples=max(len(samples) - 1, 0),
        problem="")
    if not tables:
        return _with_problem(plan, tr(
            "None of the barcode reference tables on this form could be "
            "read, so there is nothing to look for in the reads."))
    return plan


def _with_problem(plan, problem):
    """Return the same plan carrying a sentence about why it cannot be used.

    :param plan: the plan as far as it could be worked out.
    :param problem: the sentence to carry.
    :returns: a new plan holding the problem.
    """
    return replace(plan, problem=str(problem))


# ---------------------------------------------------------------------------
# The three pieces of work that happen off the GUI thread
# ---------------------------------------------------------------------------
#
# Each of these takes data and returns data. None of them touches a widget,
# which is what lets `JobRunner` hand them to a worker thread, and each
# returns its failure in the result rather than raising, because an exception
# on a worker thread has nobody to catch it and a panel that goes quiet is
# worse than one that says what went wrong.


def _prepare_barcode_search(settings, max_reads, chunk_reads):
    """Plan a search, read its reference tables and take the first reads.

    Planning, loading and the first chunk are one job rather than three
    because they are one wait from the user's point of view: nothing can be
    shown until reads have been counted against tables, and three round trips
    to a worker thread would only make the first number arrive later.

    :param settings: the Map Barcodes settings as the form holds them.
    :param max_reads: how many reads to sample from each file in total.
    :param chunk_reads: how many reads each step takes from each file.
    :returns: a mapping holding the plan, the loaded tables, the iterator the
        rest of the search advances, the first report, and any error.
    """
    out = {"plan": None, "tables": (), "iterator": None, "report": None,
           "error": ""}
    try:
        from ...barcode_search import iter_barcode_search, load_barcode_table

        plan = plan_barcode_search(settings)
        out["plan"] = plan
        if plan.problem:
            return out
        tables = tuple(
            load_barcode_table(path, name=name, role=role)
            for name, path, role in plan.reference_tables)
        out["tables"] = tables
        iterator = iter_barcode_search(
            dict(plan.fastq_files), tables, max_reads=max_reads,
            chunk_reads=chunk_reads, anchor=plan.anchor or None)
        out["iterator"] = iterator
        out["report"] = next(iterator, None)
    except Exception as exc:                                 # noqa: BLE001
        LOG.info("the barcode search could not be started", exc_info=True)
        out["error"] = str(exc) or exc.__class__.__name__
    return out


def _advance_barcode_search(iterator):
    """Take one more chunk of reads and return the report it produces.

    :param iterator: the iterator a prepared search left behind.
    :returns: a mapping holding the next report, which is None when the reads
        ran out, and any error.
    """
    out = {"report": None, "error": ""}
    try:
        out["report"] = next(iterator, None)
    except Exception as exc:                                 # noqa: BLE001
        LOG.info("the barcode search could not be advanced", exc_info=True)
        out["error"] = str(exc) or exc.__class__.__name__
    return out


def _sample_annotated_reads(path, tables, anchor, limit):
    """Read a bounded sample of reads and locate every barcode inside them.

    Both orientations of every table are searched here, rather than only the
    orientation that won. A reader looking at the reads is checking the
    verdicts rather than trusting them, and a barcode that turns up flipped is
    exactly what they need to see in order to believe an orientation.

    The anchor is drawn alongside the barcodes. It is the landmark the
    extraction window is measured from, so seeing where it lands, and seeing
    the barcodes sitting a fixed distance after it, is what makes a proposed
    offset something a reader can check rather than take on trust.

    :param path: the sequencing file to read from.
    :param tables: the reference tables to look for.
    :param anchor: the fixed vector sequence to draw as well, or empty.
    :param limit: how many reads to take.
    :returns: a mapping holding one entry per read, each the read text and the
        stretches of it that matched, plus any error.
    """
    out = {"rows": (), "error": ""}
    try:
        from ...barcode_search import (ANCHOR_ROLE, BarcodeTable,
                                       iter_annotated_reads)

        tables = tuple(tables)
        if anchor:
            tables += (BarcodeTable(
                name=ANCHOR_ROLE, role=ANCHOR_ROLE,
                sequences={str(anchor).upper(): ANCHOR_ROLE}),)
        rows = []
        for read, hits in iter_annotated_reads(path, tables, limit=limit):
            rows.append((read, tuple(
                (hit.start, hit.end, hit.role or hit.table) for hit in hits)))
        out["rows"] = tuple(rows)
    except Exception as exc:                                 # noqa: BLE001
        LOG.info("could not sample reads for the barcode search",
                 exc_info=True)
        out["error"] = str(exc) or exc.__class__.__name__
    return out


# ---------------------------------------------------------------------------
# Rendering measurements as something a person can act on
# ---------------------------------------------------------------------------


def _same_setting_value(left, right):
    """Return whether two settings values mean the same thing.

    A form holds a number in a spin box and a proposal carries it as an
    integer, so a comparison that only asks whether the objects are equal
    reports a change where there is none and offers to write a value that is
    already there.

    :param left: one value.
    :param right: the other value.
    :returns: True when the two say the same thing.
    """
    if left is right:
        return True
    try:
        if left == right:
            return True
    except Exception:                                        # noqa: BLE001
        pass
    return str(left) == str(right)


def describe_proposed_changes(proposal, current_settings):
    """Return the settings a proposal would change, old value beside new.

    Showing what will change before it changes is the difference between a
    convenience and a trap. Somebody who typed a window length has to be able
    to see that pressing Apply replaces it, and with what, while there is
    still time to decide not to.

    :param proposal: the proposal a finished search produced.
    :param current_settings: the settings as the form holds them now.
    :returns: a tuple of entries, each holding the setting's key, the value
        the form holds and the value the search proposes, for those settings
        whose value would actually change.
    """
    current = dict(current_settings or {})
    changes = []
    for key, value in dict(getattr(proposal, "settings", {}) or {}).items():
        present = current.get(key)
        if key in current and _same_setting_value(present, value):
            continue
        changes.append((str(key), present, value))
    return tuple(sorted(changes, key=lambda entry: entry[0]))


def _orientation_words(orientation):
    """Return a plain phrase for one of the engine's orientation labels.

    :param orientation: the orientation label a finding carries.
    :returns: the phrase to show in the findings table.
    """
    from ...barcode_search import REVERSE_COMPLEMENT

    if orientation == REVERSE_COMPLEMENT:
        return tr("reverse complemented")
    return tr("as stored")


def _rate_words(rate):
    """Return a percentage, without letting a small one read as a zero.

    A guide table of a thousand twenty base sequences has a coincidence rate
    of about one in a hundred billion. Printed to two decimal places that is
    zero, and a reader who sees a chance rate of zero has been told something
    false about what the measurement can settle.

    :param rate: the share of reads, between zero and one.
    :returns: the rendered percentage.
    """
    percent = float(rate) * 100.0
    if percent <= 0.0:
        return "0.00%"
    if percent < 0.01:
        return "<0.01%"
    return f"{percent:.2f}%"


def _enrichment_words(enrichment):
    """Return how far above coincidence an observation sat.

    :param enrichment: the observed rate divided by the expected one.
    :returns: the rendered ratio.
    """
    import math

    value = float(enrichment)
    if math.isinf(value) or value >= 1000.0:
        return tr("over 1000x")
    return f"{value:.1f}x"


class BarcodeSearchPanel(QWidget):
    """The live search: reads in, measured settings out, nothing written yet.

    Press the search button and the panel samples the reads of one sample from
    the folder the form names, measures every reference table against them in
    both orientations and in both mates, and refines what it shows after each
    chunk of reads. It ends by proposing the settings those measurements imply
    and waiting, because applying them is the user's decision.

    What is on screen, and why each part is there. The findings table carries
    the observed rate and the rate expected by coincidence in adjacent
    columns, with the ratio between them beside both, because a rate alone
    cannot be told from an accident and this panel exists to stop somebody
    acting on one. The text window shows reads with their matches coloured by
    barcode type, because a barcode landing in the same columns on row after
    row is the evidence that settles an argument a percentage cannot. The
    proposal shows the value the form holds beside the value the search
    suggests, so what Apply is about to do is legible before it does it.

    What it costs. Every file is read inside a submitted job, one chunk at a
    time, so the interface stays live throughout and a search can be abandoned
    at any point. Cancelling proposes nothing: an interrupted measurement is
    an honest partial report and a dishonest recommendation.

    :param screen: the Map Barcodes screen whose settings form is read for the
        search and written by Apply. May be None, which leaves the panel
        usable as a display with nothing to read from and nothing to write to.
    :param parent: parent widget; ownership only.
    :param threaded: run the search on worker threads. False runs each step
        inline, in the same order and through the same handlers, so a test can
        drive a whole search synchronously.
    :param max_reads: how many reads to sample from each file. The engine's
        own default when omitted.
    :param chunk_reads: how many reads each step of the search takes from each
        file. The engine's own default when omitted.
    :param read_sample: how many reads the text window shows.
    :ivar search_button: starts a search.
    :ivar cancel_button: abandons the search in flight.
    :ivar apply_button: writes the proposed settings into the form; disabled
        until a finished search has proposed something that differs from what
        the form already holds.
    :ivar status: one line saying what the search is doing or has found.
    :ivar findings: the table of measurements, one row per reference table per
        mate per orientation.
    :ivar proposal_label: what Apply would change, and what the search learned
        that no setting can hold.
    :ivar reads: the text window showing sampled reads with matches coloured.
    """

    #: A report arrived and the display has been refreshed from it. Carries
    #: the report, which is complete in shape whether or not the search is.
    search_updated = Signal(object)
    #: The search ended. Carries the final report, or None when it was
    #: cancelled or could not run.
    search_finished = Signal(object)
    #: Apply wrote settings into the form. Carries what was written.
    settings_applied = Signal(object)

    def __init__(self, screen=None, parent: Optional[QWidget] = None, *,
                 threaded: bool = True, max_reads: Optional[int] = None,
                 chunk_reads: Optional[int] = None,
                 read_sample: int = READ_SAMPLE_READS):
        """Build the panel and arm it against one settings screen."""
        super().__init__(parent)
        from ...barcode_search import DEFAULT_CHUNK_READS, DEFAULT_SAMPLE_READS
        from ..job_runner import JobRunner

        self._screen = screen
        self._max_reads = int(max_reads or DEFAULT_SAMPLE_READS)
        self._chunk_reads = int(chunk_reads or DEFAULT_CHUNK_READS)
        self._read_sample = int(read_sample)
        # Every file read goes through here rather than through a thread this
        # file owns, for the reason the live preview gives: `JobRunner`
        # submits through `bridge.make_thread`, and that is what puts the work
        # in the run registry the activity spinner watches.
        self._jobs = JobRunner(self, threaded=threaded,
                               app_key="barcode search")
        self._plan = None
        self._tables: Tuple[object, ...] = ()
        self._iterator = None
        self._report = None
        self._proposal = None
        self._changes: Tuple[Tuple[str, object, object], ...] = ()
        self._running = False
        #: How far the text window has been refreshed: once while the search
        #: is running so that reads appear early, and once more at the end
        #: when the counts behind the colouring have settled.
        self._reads_shown = 0
        self._build_ui()
        self._update_buttons()

    # -- building ---------------------------------------------------------

    def _build_ui(self) -> None:
        """Lay out the controls, the findings table, the proposal and reads."""
        from ..theme import SPACING
        from ..widgets.read_view import ReadView

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING["sm"])

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(SPACING["sm"])
        self.search_button = self._button(
            "Find barcodes",
            "Sample the reads of one sample and measure every reference "
            "table against them, in both orientations and in both mates.")
        self.search_button.clicked.connect(self.on_search_clicked)
        self.cancel_button = self._button(
            "Cancel", "Stop the search. What it has measured so far stays on "
                      "screen, and nothing is proposed from a search that was "
                      "interrupted.")
        self.cancel_button.clicked.connect(self.on_cancel_clicked)
        self.apply_button = self._button(
            "Apply these settings",
            "Write the proposed settings into the form. Nothing is written "
            "until this is pressed.")
        self.apply_button.clicked.connect(self.on_apply_clicked)
        controls.addWidget(self.search_button)
        controls.addWidget(self.cancel_button)
        controls.addWidget(self.apply_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.status = QLabel(self)
        self.status.setObjectName("CardSubtitle")
        self.status.setWordWrap(True)
        self._set_status(
            "Nothing searched yet. The search reads a bounded sample, so it "
            "costs seconds rather than a run.")
        layout.addWidget(self.status)

        self.findings = QTableWidget(0, len(SEARCH_COLUMNS), self)
        self.findings.setObjectName(SEARCH_TABLE_NAME)
        self.findings.setHorizontalHeaderLabels(
            [tr(name) for name in SEARCH_COLUMNS])
        self.findings.verticalHeader().setVisible(False)
        self.findings.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.findings.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.findings.setSelectionMode(QAbstractItemView.SingleSelection)
        header = self.findings.horizontalHeader()
        header.setStretchLastSection(True)
        for column in range(len(SEARCH_COLUMNS) - 1):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self._explain_chance_column()
        self.findings.setSizePolicy(QSizePolicy.Expanding,
                                    QSizePolicy.Expanding)
        self.findings.setMinimumHeight(150)

        self.reads = ReadView(self)
        self.reads.setMinimumHeight(120)

        # THE TWO HALVES SHARE ONE HEIGHT, and which of them deserves it
        # depends on what the reader is doing. Checking a verdict wants rows
        # of reads; comparing tables wants rows of measurements. A splitter
        # lets that be answered by the person looking rather than by a
        # number chosen here, and neither half can be collapsed to nothing.
        self.proposal_label = QLabel(self)
        self.proposal_label.setObjectName("CardSubtitle")
        self.proposal_label.setWordWrap(True)
        self.proposal_label.setTextFormat(Qt.PlainText)
        self.proposal_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.proposal_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # WHAT A MEASUREMENT SAID IS NOT CHROME. The sentences here are the
        # search engine's own account of what it found, assembled per run and
        # naming files, rates and settings keys, so the translator leaves them
        # alone rather than looking each assembled paragraph up as a caption.
        self.proposal_label.setProperty("i18nSkipText", True)
        # SCROLLED RATHER THAN GROWN. The notes are one sentence per barcode
        # plus one per decision, so a run that decodes six barcodes writes a
        # paragraph. Left to size itself, the label took that height out of
        # the reads below it and pushed them off the card entirely.
        notes = QScrollArea(self)
        notes.setObjectName("BarcodeSearchNotes")
        notes.setWidget(self.proposal_label)
        notes.setWidgetResizable(True)
        notes.setFrameShape(QFrame.NoFrame)
        notes.setStyleSheet(
            "QScrollArea#BarcodeSearchNotes, "
            "QScrollArea#BarcodeSearchNotes > QWidget > QWidget "
            "{ background: transparent; }")
        notes.setMinimumHeight(70)

        # THE THREE PANES SHARE ONE HEIGHT, and which of them deserves it
        # depends on what the reader is doing. Checking a verdict wants rows
        # of reads; comparing tables wants rows of measurements; deciding
        # whether to apply wants the notes. A splitter lets that be answered
        # by the person looking rather than by a number chosen here, and no
        # pane can be collapsed to nothing.
        split = QSplitter(Qt.Vertical, self)
        split.setChildrenCollapsible(False)
        split.addWidget(self.findings)
        split.addWidget(self.reads)
        split.addWidget(notes)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 3)
        split.setStretchFactor(2, 1)
        split.setSizes([230, 190, 110])
        layout.addWidget(split, 1)

    def _button(self, caption: str, hint: str) -> QPushButton:
        """Build one control, keeping its English for a later language change.

        The panel is built after the window has already made its one
        translation pass over the screen, so each control translates its own
        caption. The English stays on the widget as the source a later pass
        reads, or the second change of language would have a translation to
        translate rather than the original.

        :param caption: the English caption.
        :param hint: the English tooltip.
        :returns: the button.
        """
        button = QPushButton(self)
        button.setProperty("_spacr_i18n_text", caption)
        button.setText(tr(caption))
        button.setProperty("_spacr_i18n_tooltip", hint)
        button.setToolTip(tr(hint))
        button.setCursor(Qt.PointingHandCursor)
        return button

    def _explain_chance_column(self) -> None:
        """Put the point of the chance column into the column itself.

        The number under this heading is the whole argument of the panel, and
        a heading of two words cannot carry it. Somebody hovering the one
        column they do not recognise is exactly the person who needs the
        sentence.
        """
        hint = ("The share of reads that would match this table by "
                "coincidence alone, given how many barcodes it holds, how "
                "long they are and how long the reads are. An observed rate "
                "near this one is not a finding.")
        item = self.findings.horizontalHeaderItem(
            SEARCH_COLUMNS.index("By chance"))
        if item is not None:
            item.setToolTip(tr(hint))

    # -- the controls -----------------------------------------------------

    def on_search_clicked(self, _checked: bool = False) -> None:
        """Start a search, or restart one that is already running.

        :param _checked: Qt's toggle state, unused.
        """
        if self._running:
            self.cancel_search()
        self.start_search()

    def on_cancel_clicked(self, _checked: bool = False) -> None:
        """Abandon the search in flight.

        :param _checked: Qt's toggle state, unused.
        """
        self.cancel_search()

    def on_apply_clicked(self, _checked: bool = False) -> None:
        """Write the proposed settings into the form.

        :param _checked: Qt's toggle state, unused.
        """
        self.apply_proposal()

    # -- running the search ----------------------------------------------

    def start_search(self) -> bool:
        """Begin a fresh search over the files the settings form names.

        Returns rather than raises when there is nothing to search, because
        the reason is a sentence about the form and the panel shows it.

        :returns: True when a search was started.
        """
        if self._running:
            return False
        settings = self.current_settings()
        self._reset()
        self._running = True
        self._update_buttons()
        self._set_status(
            "Finding the sequencing files and reading the first of them.")
        self._jobs.submit(
            partial(_prepare_barcode_search, settings, self._max_reads,
                    self._chunk_reads),
            self._on_prepared)
        return True

    def cancel_search(self) -> bool:
        """Abandon the search in flight and propose nothing from it.

        The chunk already on a worker thread is not interrupted; it finishes
        and its result is dropped on arrival, which is what cancelling means
        for work that holds no lock and cannot be asked to stop in the middle.

        :returns: True when there was a search to cancel.
        """
        if not self._running:
            return False
        self._running = False
        self._iterator = None
        self._jobs.cancel()
        reads = getattr(self._report, "reads", 0) or 0
        self._set_status(
            "Search cancelled after {reads} reads. Nothing is proposed from "
            "a search that was interrupted.", reads=f"{reads:,}")
        self._update_buttons()
        self.search_finished.emit(None)
        return True

    def is_searching(self) -> bool:
        """Return whether a search is running.

        :returns: True while the panel is working through the reads.
        """
        return bool(self._running)

    def report(self):
        """Return the most recent report, complete or not.

        :returns: the report, or None when no search has produced one.
        """
        return self._report

    def proposal(self):
        """Return the proposal a finished search produced.

        :returns: the proposal, or None when no search has finished.
        """
        return self._proposal

    def proposed_changes(self):
        """Return the settings Apply would write, old value beside new.

        :returns: a tuple of entries, each holding a setting's key, the value
            the form holds and the value the search proposes.
        """
        return self._changes

    def current_settings(self):
        """Return the settings as the form holds them at this moment.

        :returns: the settings dictionary, empty when there is no form.
        """
        model = getattr(self._screen, "_settings_model", None)
        if model is None:
            return {}
        try:
            return dict(model.collect() or {})
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not read the Map Barcodes settings",
                      exc_info=True)
            return {}

    def apply_proposal(self) -> Tuple[str, ...]:
        """Write the proposed settings into the form, on purpose.

        Nothing calls this on its own. A search that finishes enables the
        button and stops there, because a panel that quietly replaced values
        somebody typed would be a worse failure than the one it exists to
        prevent.

        :returns: the keys that were written, in the order they were written.
        """
        model = getattr(self._screen, "_settings_model", None)
        if model is None or not self._changes:
            return ()
        changes = self._changes
        written = {}
        for key, _present, value in changes:
            try:
                if model.set_value_for_key(key, value):
                    written[key] = value
            except Exception:                                # noqa: BLE001
                LOG.debug("could not write %s into the form", key,
                          exc_info=True)
        self._changes = ()
        self._update_buttons()
        self._render_proposal(applied=tuple(written))
        self._set_status(
            "Wrote {count} of the proposed settings into the form. Look at "
            "them, then run the mapping.", count=len(written))
        self.settings_applied.emit(dict(written))
        return tuple(written)

    # -- the steps, as their results arrive on the GUI thread -------------

    def _on_prepared(self, result) -> None:
        """Adopt a prepared search, or say why there is none.

        :param result: what the preparing job returned.
        """
        if not self._running:
            return
        result = result or {}
        if result.get("error"):
            self._stop_with(tr("The search stopped: {reason}",
                               reason=result["error"]))
            return
        plan = result.get("plan")
        self._plan = plan
        if plan is None or plan.problem:
            self._stop_with(getattr(plan, "problem", "") or tr(
                "The search could not work out what to read."))
            return
        self._tables = tuple(result.get("tables") or ())
        self._iterator = result.get("iterator")
        self.reads.set_reads((), kinds=self._barcode_kinds())
        report = result.get("report")
        if report is None:
            self._stop_with(tr(
                "The sequencing files hold no reads the search could read."))
            return
        self._absorb(report)
        self._next_step()

    def _on_reads(self, result) -> None:
        """Show the sampled reads, then carry on with the search.

        :param result: what the read sampling job returned.
        """
        if not self._running:
            return
        rows = (result or {}).get("rows") or ()
        if rows:
            self.reads.set_reads(rows, kinds=self._barcode_kinds())
        self._next_step()

    def _on_chunk(self, result) -> None:
        """Adopt one more chunk of reads and decide what happens next.

        :param result: what the advancing job returned.
        """
        if not self._running:
            return
        result = result or {}
        if result.get("error"):
            self._stop_with(tr("The search stopped: {reason}",
                               reason=result["error"]))
            return
        report = result.get("report")
        if report is None:
            self._finish()
            return
        self._absorb(report)
        self._next_step()

    def _next_step(self) -> None:
        """Submit whatever the search needs next, one job at a time.

        Only ever one job is in flight. The reference tables build their
        search indexes on first use and keep them, so two jobs reading the
        same tables at once would race to build the same index, and a search
        that overlapped its own chunks would report reads out of order.
        """
        if not self._running:
            return
        report = self._report
        complete = bool(getattr(report, "complete", False))
        wanted = 2 if complete else 1
        if self._reads_shown < wanted and self._tables and self._plan:
            self._reads_shown = wanted
            path = next(iter(self._plan.fastq_files.values()), "")
            if path:
                self._jobs.submit(
                    partial(_sample_annotated_reads, path, self._tables,
                            self._plan.anchor, self._read_sample),
                    self._on_reads)
                return
        if complete or self._iterator is None:
            self._finish()
            return
        self._jobs.submit(partial(_advance_barcode_search, self._iterator),
                          self._on_chunk)

    def _finish(self) -> None:
        """End the search and propose the settings its measurements imply."""
        self._running = False
        self._iterator = None
        report = self._report
        if report is not None:
            try:
                from ...barcode_search import propose_map_barcodes_settings

                # Read once. Collecting the form walks every widget on it, and
                # a proposal compared against a second reading would be a
                # proposal compared against a different dictionary.
                settings = self.current_settings()
                self._proposal = propose_map_barcodes_settings(
                    report, base_settings=settings)
                self._changes = describe_proposed_changes(
                    self._proposal, settings)
            except Exception:                                # noqa: BLE001
                LOG.debug("could not propose settings from the search",
                          exc_info=True)
                self._proposal = None
                self._changes = ()
        self._render_proposal()
        self._update_buttons()
        self._set_finished_status()
        self.search_finished.emit(report)

    def _stop_with(self, problem) -> None:
        """End the search because it cannot go on, and say why.

        :param problem: the sentence to show.
        """
        self._running = False
        self._iterator = None
        self._update_buttons()
        # Through the same helper as every other line, so that a language
        # changed afterwards re-renders this one rather than restoring
        # whatever sentence the status carried before the search failed.
        self._set_status("{problem}", problem=str(problem))
        self.search_finished.emit(None)

    # -- showing what was measured ---------------------------------------

    def _absorb(self, report) -> None:
        """Take one report, refresh the display from it and say so.

        :param report: the report the engine just handed back.
        """
        self._report = report
        self._fill_findings(report)
        self._set_running_status(report)
        self.search_updated.emit(report)

    def _fill_findings(self, report) -> None:
        """Draw one row per reference table per mate per orientation.

        The row order is fixed by the barcode, the table, the mate and the
        orientation rather than by how convincing the finding is. A live
        display that sorted by strength would reorder itself under the
        reader's eyes every couple of seconds, and the row somebody was
        reading would be somewhere else by the time they finished the
        sentence.

        :param report: the report to draw.
        """
        from PySide6.QtGui import QColor

        from ..theme import active_palette
        from ...barcode_search import ABSENT, INDETERMINATE, PRESENT

        findings = sorted(
            getattr(report, "findings", ()) or (),
            key=lambda item: ((item.role or ""), item.table, item.file_label,
                              item.orientation))
        palette = active_palette()
        verdict_ink = {
            PRESENT: palette.get("success"),
            ABSENT: palette.get("fg_muted"),
            INDETERMINATE: palette.get("warning"),
        }
        self.findings.setRowCount(len(findings))
        for row, finding in enumerate(findings):
            cells = (
                finding.role or finding.table,
                finding.file_label,
                _orientation_words(finding.orientation),
                _rate_words(finding.observed_rate),
                _rate_words(finding.expected_rate),
                _enrichment_words(finding.enrichment),
                "-" if finding.modal_offset is None
                else str(finding.modal_offset),
                tr(str(finding.verdict)),
            )
            for column, text in enumerate(cells):
                # REUSED WHERE THERE IS ONE, because this is redrawn after
                # every chunk of reads. Replacing the items would drop the
                # row the reader had selected and the place they had
                # scrolled to, three times a second, which is a display
                # nobody can read while it is working.
                item = self.findings.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    self.findings.setItem(row, column, item)
                item.setText(str(text))
                # The sentence the engine wrote about this finding says which
                # check it passed or failed, which is the one thing a reader
                # who disagrees with a verdict needs and the one thing no
                # column is wide enough to hold.
                item.setToolTip(str(finding.reason))
                if column == len(cells) - 1:
                    ink = verdict_ink.get(finding.verdict)
                    if ink:
                        item.setForeground(QColor(ink))

    def _barcode_kinds(self) -> Tuple[str, ...]:
        """Return the barcode types the text window colours, in legend order.

        Types are named here rather than left to be discovered in the sampled
        reads so that a barcode that was searched for and found nowhere still
        appears in the legend. That absence is information: it says the search
        ran and came back empty rather than leaving somebody to wonder whether
        it ran at all.

        :returns: the type names, with the anchor last when one is searched
            for.
        """
        from ...barcode_search import ANCHOR_ROLE

        if self._plan is None:
            return ()
        kinds = []
        for _name, _path, role in self._plan.reference_tables:
            if role and role not in kinds:
                kinds.append(role)
        if self._plan.anchor:
            kinds.append(ANCHOR_ROLE)
        return tuple(kinds)

    def _set_running_status(self, report) -> None:
        """Say how far the search has got and over which files.

        :param report: the report the count is taken from.
        """
        plan = self._plan
        reads = getattr(report, "reads", 0) or 0
        mates = ", ".join(plan.fastq_files) if plan else ""
        if plan is not None and plan.other_samples == 1:
            self._set_status(
                "Sample {sample}: {reads} reads sampled from {mates}. One "
                "more sample sits beside it in the same folder and shares "
                "the library layout, so this one settles both.",
                sample=plan.sample, reads=f"{reads:,}", mates=mates)
            return
        if plan is not None and plan.other_samples:
            self._set_status(
                "Sample {sample}: {reads} reads sampled from {mates}. The "
                "{others} other samples in this folder share the library "
                "layout, so this one settles all of them.",
                sample=plan.sample, reads=f"{reads:,}", mates=mates,
                others=plan.other_samples)
            return
        self._set_status(
            "Sample {sample}: {reads} reads sampled from {mates}.",
            sample=getattr(plan, "sample", ""), reads=f"{reads:,}",
            mates=mates)

    def _set_finished_status(self) -> None:
        """Say what the finished search established, and what it did not."""
        report = self._report
        if report is None:
            return
        from ...barcode_search import ANCHOR_ROLE, PRESENT

        found = []
        for role in getattr(report, "roles", lambda: ())():
            if role == ANCHOR_ROLE:
                continue
            best = report.best_for_role(role)
            if best is not None and best.verdict == PRESENT:
                found.append(role)
        missing = [role for role in report.roles()
                   if role != ANCHOR_ROLE and role not in found]
        reads = report.reads
        if found and not missing:
            self._set_status(
                "Every barcode was established from {reads} reads: {found}. "
                "The proposed settings are below.",
                reads=f"{reads:,}", found=", ".join(found))
        elif found:
            self._set_status(
                "Established {found} from {reads} reads. No reference table "
                "stood clear of coincidence for {missing}, so those settings "
                "are left as they are.",
                found=", ".join(found), reads=f"{reads:,}",
                missing=", ".join(missing))
        else:
            self._set_status(
                "No barcode table stood clear of its own coincidence rate in "
                "{reads} reads. Compare the observed and chance columns "
                "before changing anything: a rate near the chance rate is an "
                "accident rather than a finding.", reads=f"{reads:,}")

    def _render_proposal(self, applied: Tuple[str, ...] = ()) -> None:
        """Write out what Apply would change, and what it cannot change.

        The notes matter as much as the settings. The failure this panel was
        built for is a reference table stored in the opposite orientation to
        the reads, and no setting says which way round a table is stored, so
        the only place that answer can land is a sentence somebody reads.

        :param applied: the keys that have just been written, if any.
        """
        proposal = self._proposal
        if proposal is None:
            self.proposal_label.setText("")
            return
        lines = []
        unresolved = tuple(getattr(proposal, "unresolved_roles", ()) or ())
        if unresolved and not applied:
            # THE WINDOW IS DERIVED FROM WHAT WAS ESTABLISHED, and nothing
            # else. A run whose column and row references never cleared
            # coincidence gets a window around the guide alone, which is the
            # honest answer to what was measured and is still not a window
            # that will decode the reads the regex describes. Saying so above
            # the numbers costs one line and saves a run that maps nothing.
            lines.append(tr(
                "No reference table stood clear of coincidence for {roles}. "
                "A mapping run will not decode those barcodes, and any "
                "window proposed below covers only the barcodes that were "
                "established. Check that the reference table named for each "
                "is the right file, and read the notes below.",
                roles=", ".join(unresolved)))
            lines.append("")
        if applied:
            lines.append(tr("Written into the form: {keys}.",
                            keys=", ".join(applied)))
        elif self._changes:
            lines.append(tr("Apply would change:"))
            for key, present, value in self._changes:
                lines.append(f"    {key}: {present!r} -> {value!r}")
        else:
            lines.append(tr(
                "The form already holds every setting this search would "
                "propose."))
        notes = tuple(getattr(proposal, "notes", ()) or ())
        if notes:
            lines.append("")
            lines.extend(f"    {note}" for note in notes)
        self.proposal_label.setText("\n".join(lines))

    def _set_status(self, text: str, **values) -> None:
        """Put one translated sentence on the status line.

        Set through the helper that keeps the English template and the values
        beside the rendered text, so that a language changed while a search is
        on screen re-renders the sentence rather than leaving the placeholders
        showing.

        :param text: the English sentence, which may carry named placeholders.
        :param values: what to substitute into them.
        """
        from ..i18n import set_translatable_text

        set_translatable_text(self.status, text, **values)

    def _update_buttons(self) -> None:
        """Enable exactly the controls that would do something right now."""
        running = bool(self._running)
        self.cancel_button.setEnabled(running)
        self.apply_button.setEnabled(
            bool(self._changes) and not running
            and getattr(self._screen, "_settings_model", None) is not None)

    def _reset(self) -> None:
        """Forget the previous search so a new one starts from nothing."""
        self._plan = None
        self._tables = ()
        self._iterator = None
        self._report = None
        self._proposal = None
        self._changes = ()
        self._reads_shown = 0
        self.findings.setRowCount(0)
        self.proposal_label.setText("")
        self.reads.clear()

    # -- living in a themed, closable window ------------------------------

    def changeEvent(self, event) -> None:                    # noqa: N802
        """Redraw the measurements when the theme underneath them changes.

        The verdict colours are taken from the palette that was on screen when
        the row was drawn, so a row drawn before a theme change keeps the old
        theme's ink until it is drawn again.

        :param event: the Qt change event being delivered.
        """
        super().changeEvent(event)
        from PySide6.QtCore import QEvent

        if not hasattr(self, "findings"):
            return
        if event.type() in (QEvent.Type.StyleChange,
                            QEvent.Type.PaletteChange,
                            QEvent.Type.ApplicationPaletteChange):
            if self._report is not None:
                self._fill_findings(self._report)

    def shutdown(self) -> None:
        """Abandon any search in flight and leave no worker thread behind.

        Safe to call directly when a screen is torn down without a close
        event, which is how a folded module usually goes away.
        """
        self._running = False
        self._iterator = None
        runner = getattr(self, "_jobs", None)
        if runner is not None:
            runner.shutdown()

    def closeEvent(self, event) -> None:                     # noqa: N802
        """Stop the search rather than let it outlive the panel.

        :param event: the Qt close event being delivered.
        """
        self.shutdown()
        super().closeEvent(event)


def build_barcode_search_card(screen, **kwargs):
    """Build the live search and the card it sits in, unmounted.

    Returned unmounted, as the live preview's card is, so that the caller
    decides where it goes and whether it starts visible.

    :param screen: the Map Barcodes screen the search reads and writes.
    :param kwargs: passed through to :class:`BarcodeSearchPanel`.
    :returns: the panel and the card holding it.
    """
    from ..widgets.card import Card

    card = Card(title=tr("Find barcodes"),
                subtitle=tr(
                    "Measure every reference table against a sample of the "
                    "reads, in both orientations and in both mates, then "
                    "propose the settings those measurements imply."))
    card.setObjectName(SEARCH_CARD_NAME)
    panel = BarcodeSearchPanel(screen, card, **kwargs)
    card.body_layout.addWidget(panel)
    card.setMinimumHeight(560)
    return panel, card


def _insert_above_actions(screen: QWidget, widget: QWidget) -> bool:
    """Put ``widget`` in the runtime panel just above the Run row.

    Both anchors are attributes ``AppScreen`` keeps for exactly this kind of
    reach from outside, which is the same pair :mod:`spacr.qt.prerun` and
    :mod:`spacr.qt.preview_registry` take to mount their own panels. Above the
    actions row is the last thing the eye crosses on the way to Run, which is
    where something that changes what Run will do belongs.

    :param screen: the screen to mount into.
    :param widget: the widget to mount.
    :returns: True when it was mounted.
    """
    wrap = getattr(screen, "_runtime_wrap", None)
    actions = getattr(screen, "_actions_row", None)
    if wrap is None or actions is None:
        return False
    layout = wrap.layout()
    if layout is None:
        return False
    index = layout.indexOf(actions)
    layout.insertWidget(index if index >= 0 else layout.count(), widget)
    return True


def install_barcode_search(screen: QWidget, **kwargs):
    """Give the Map Barcodes screen its live search, behind a toggle.

    Installed rather than built into the screen for the reason every other
    panel on this screen is installed from outside: the screen is the generic
    settings screen, which knows nothing about sequencing and should not have
    to. Repeated calls return the panel that is already there, because the
    stack watcher reaches a screen again every time it becomes current.

    :param screen: the screen to install into. Anything that is not the Map
        Barcodes screen is left alone.
    :param kwargs: passed through to :class:`BarcodeSearchPanel`, which is how
        a test asks for a smaller read sample or an unthreaded search.
    :returns: the panel, or None when the screen is not the Map Barcodes
        screen or has no runtime panel to mount into.
    """
    if getattr(screen, "app_key", None) != HOST_KEY:
        return None
    existing = getattr(screen, "_barcode_search", None)
    if isinstance(existing, BarcodeSearchPanel):
        return existing
    try:
        panel, card = build_barcode_search_card(screen, **kwargs)
    except Exception:
        LOG.debug("could not build the barcode search panel", exc_info=True)
        return None
    if not _insert_above_actions(screen, card):
        card.setParent(None)
        card.deleteLater()
        return None
    card.setVisible(False)

    toggle = QToolButton()
    toggle.setObjectName(SEARCH_TOGGLE_NAME)
    caption = "Find barcodes"
    toggle.setProperty("_spacr_i18n_text", caption)
    toggle.setText(tr(caption))
    toggle.setCheckable(True)
    toggle.setCursor(Qt.PointingHandCursor)
    hint = ("Search the reads for the barcode tables this run uses and "
            "propose the settings that match what they say. Every rate it "
            "reports is shown beside the rate expected by coincidence.")
    toggle.setProperty("_spacr_i18n_tooltip", hint)
    toggle.setToolTip(tr(hint))
    toggle.toggled.connect(card.setVisible)

    bar = getattr(screen, "_settings_search", None)
    if bar is not None and hasattr(bar, "add_trailing_widget"):
        bar.add_trailing_widget(toggle)
    else:
        # No strip on this screen, so the toggle goes above the card rather
        # than nowhere, or the search would be installed and unreachable.
        toggle.setParent(screen)
        _insert_above_actions(screen, toggle)
    screen._barcode_search = panel
    screen._barcode_search_card = card
    screen._barcode_search_toggle = toggle
    return panel
