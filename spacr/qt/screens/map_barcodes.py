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
from functools import partial
from typing import Callable, Dict, Optional, Sequence, Tuple

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import QLabel, QTabBar, QTabWidget, QWidget

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
    `AppScreen`s, and which come from a plugin.

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

            icon = iconset.app_icon(key)
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
    """Put Map Barcodes' fold strip on ``screen``'s masthead."""
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
        from .settings_model import SettingsWidgets

        # ONLY WHAT THIS FOLD ADDS. The loop below keeps a row exactly when
        # the host does not already hold its key, so building the rest was
        # 96% waste: the timelapse fold on the mask screen built 364
        # settings to keep 14, at 1,148 ms on every module open. The host's
        # own keys are skipped up front instead.
        already = set(getattr(host_model, "_widgets", {}))
        model = SettingsWidgets(self.key, parent=content, skip_keys=already)
        built = model.build_sections()
        held = set(getattr(host_model, "_widgets", {}))
        by_widget = _widget_keys(model)
        mounted_keys: list = []
        for source in built:
            title = source.title if hasattr(source, "title") else source[0]
            rows = source.rows if hasattr(source, "rows") else source[1]
            own = [(label, widget) for label, widget in rows
                   if by_widget.get(id(widget)) not in held
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
