"""Map Barcodes, and the QC that says whether the mapping worked.

Barcode QC is not a separate errand. A mapping run is judged by reads per
well, starved wells, unmapped reads, barcode collisions, row/column
position effects and library coverage -- and by the abundance threshold
those numbers imply -- so the question "did this run work" belongs on the
screen that produced the run rather than one tile away from it.

It arrives as a button on the Map Barcodes masthead: the Barcode QC icon
with no text, the module's own one-line description as its tooltip, lit
on hover in the maturity colour its tile used. See
:class:`spacr.qt.widgets.fold_strip.FoldStrip`, which reads that colour
from the same table the tiles read.

NOTHING IS LOST IN THE MOVE. The button opens the Barcode QC module
itself -- the same settings form, the same Run button, the same console
and figures -- as a PAGE beside the mapping settings, so every number the
folded screen could produce it still produces, and the mapping settings
are one tab away rather than behind a window. A window is what a fold
becomes only when its host has no body to make pages out of.

THE SHARED HALF OF A FOLD LIVES HERE. Opening a folded module in a
window, wiring the host signals a sidebar row used to wire, and hanging
the strip off the masthead are the same job on every host, so
:func:`install_fold_strip`, :class:`FoldOpener` and
:func:`build_settings_screen` are written once and imported by the other
hosts. So is :func:`install_window_hooks`, which walks the window's
screen stack and gives each host in :data:`FOLD_HOST_MODULES` its strip:
they
reach into an already-built ``AppScreen`` the way
:func:`spacr.qt.chaining.install_chaining` and
:mod:`spacr.qt.preview_registry` do, and for the same reason -- the
shared screen needs no line about who folded into it.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Callable, Dict, Optional, Sequence, Tuple

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import QLabel, QTabBar, QTabWidget, QWidget

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
        "Did the mapping run work, and where does the abundance threshold "
        "go",
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
    "agreement": (
        "Annotator Agreement",
        "Cohen's/Fleiss' κ between annotation columns + a disagreement "
        "review",
        "stable"),
    "anndata_export": (
        "AnnData Export",
        "Write the measurements as .h5ad for scanpy and scvi-tools",
        "beta"),
    "timelapse": (
        "Timelapse",
        "Segment and track objects across the frames of a time series",
        "beta"),
    "motility": (
        "Motility Assay",
        "Automated motility assay: track velocity + infection QC",
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
    """``(name, description, stage)`` for a folded module.

    The app registry answers while it still holds the module's row; once
    the row has been dropped -- which is what folding a module ends in --
    the answer comes from :data:`FOLD_FALLBACK`, so the button goes on
    carrying the name, the sentence and the maturity colour its tile had.
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
    """Give ``button`` the name, sentence and stage its tile carried.

    A no-op while the registry still holds the row -- the strip has
    already read the same three things from the same place. It is what
    keeps the button honest afterwards, when the row is gone and the
    registry would report no description and a stable-blue hover for a
    module that is neither.
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
    """The window title for folded module ``key``.

    Read from the registry rather than typed here, so a module renamed in
    one place is renamed on its folded window too; a module whose row has
    already been dropped is named by :data:`FOLD_FALLBACK` instead of by
    its key.
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

    The pairs come from :data:`spacr.qt.chaining.HOST_CONNECTIONS`, which
    is the table ``MainWindow._build_screen``'s own wiring is checked
    against -- so a signal added there reaches a folded screen too,
    without anyone having to remember that folds exist.
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
    """Build the generic module screen for ``key``, wired as the sidebar
    wired it.

    A settings-driven module opened from a fold has to be the screen the
    sidebar row used to open, host connections included: "Explain error"
    and "Run on a cluster" are capabilities of that screen, and a fold
    that dropped them would be a fold that lost something. The chaining
    strip is offered for the same reason; a module with no declared ports
    simply does not get one.

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


def show_as_window(screen: QWidget, owner: Optional[QWidget],
                   title: str) -> QWidget:
    """Show ``screen`` as its own window, owned by ``owner``'s window.

    THE LAST RESORT, and only reached when the host cannot carry a page:
    see :func:`show_as_page`. Parented to the main window rather than left
    free-floating so that Qt keeps it alive and it closes with the
    application; a folded module held only by a local name is one the
    garbage collector closes the moment the button handler returns.
    """
    parent = owner.window() if owner is not None else None
    screen.setParent(parent, Qt.Window)
    screen.setWindowTitle(title)
    screen.resize(*FOLD_WINDOW_SIZE)
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


def _ensure_pages_qss() -> None:
    """Register the page strip's block and make sure it is live.

    Registered at the first page rather than at this module's import, and
    for the reason :func:`spacr.qt.theme.ensure_widget_qss_applied` was
    written: a block registered after the application stylesheet was
    composed is simply not in it, and the widget falls through to the
    blanket window fill -- a solid black rectangle on the dark theme. A
    fold page is opened long after launch by definition, so the sheet is
    rebuilt once, here, the first time one exists.

    ``replace=True``: this module owns the name, so being called again
    re-registers rather than raising and leaving the strip unstyled.
    """
    try:
        from ..theme import ensure_widget_qss_applied, register_widget_qss

        register_widget_qss(PAGES_NAME, _pages_qss, replace=True)
        ensure_widget_qss_applied(PAGES_NAME)
    except Exception:
        LOG.debug("Could not register the fold page QSS", exc_info=True)


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
    """The host's page strip, made on first use.

    The host's own body becomes the first page and keeps the stretch it
    had, so a host with no folds open looks exactly as it did with one tab
    across the top. Folded pages are closable; the host's own is not,
    because closing it would leave the module with nothing on screen.

    :param screen: the host module's screen.
    :param title: what to call the host's own page. Falls back to
        ``screen._fold_page_title`` -- which is how a screen that is not
        the generic settings form names itself -- and then to the name the
        registry has for its app key.
    :returns: the strip, or None when the host has no body to wrap.
    """
    existing = getattr(screen, "_fold_pages", None)
    if isinstance(existing, QTabWidget):
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
    _ensure_pages_qss()
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
    """Put ``screen`` on ``host``'s page strip and raise it.

    :param screen: the folded module's own widget.
    :param host: the host module's screen.
    :param title: the page's caption -- the folded module's name.
    :returns: the screen, or None when the host cannot carry pages, which
        is the caller's cue to fall back to a window.
    """
    pages = host_pages(host) if host is not None else None
    if pages is None:
        return None
    index = pages.indexOf(screen)
    if index < 0:
        index = pages.addTab(screen, title)
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
    """Opens one folded module, and reuses the screen it built.

    A plain object rather than a closure, so the button's connection
    holds something the strip owns and that dies with it -- the reason
    :mod:`spacr.qt.recipes` gives for its own button handler.

    THE MODULE ARRIVES AS A PAGE ON ITS HOST, beside the host's own, and
    only becomes a window when the host has no body to make pages out of.
    Either way it is the module's OWN screen that arrives, built once and
    kept, so a second press never makes a second copy of something that
    owns a database handle and a job runner -- and closing its page keeps
    everything it had loaded.

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
    """Put a fold strip for ``folded`` on ``screen``'s masthead.

    Idempotent, and defensive by design: a screen that opens without its
    fold buttons is a smaller screen, while an exception raised here
    would be no screen at all.

    :param screen: the host module's screen.
    :param host_key: the registry key that screen must carry; a screen
        for anything else is left alone, so the seam that calls this can
        be wrong without consequence.
    :param folded: the folded modules' keys, in strip order.
    :param builders: key → callable building that module's screen.
    :returns: the strip, or None when this screen cannot carry one -- it
        is not the host, it has no masthead, one is already installed, or
        building it failed.
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
}


def install_folds_on(screen: QWidget) -> Optional[FoldStrip]:
    """Install whichever host module owns ``screen``'s folds.

    Looks the screen's app key up in :data:`FOLD_HOST_MODULES` and hands the
    screen to that module. A screen with no folds -- which is most of
    them -- costs one dictionary miss.
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
    """Follow ``window``'s screen stack, giving each host its strip.

    Idempotent: a second call returns the watcher the first one made,
    rather than installing a second one that would do the same work
    twice on every tab change.

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

    The module's settings form is built exactly as its own screen builds
    it, and the categories holding settings the host does not already
    offer are moved onto the host's form -- hidden until the fold button
    switches them on.

    THE SWITCH IS THE ONLY THING THAT SHOWS OR HIDES THEM. The mounted
    cards are deliberately kept out of the host's ``_settings_sections``,
    which is the list the maturity filter and the settings search both
    walk setting each section's visibility -- two owners of one widget's
    visibility is a card that reappears the next time the Preferences
    dialog is closed, describing a run the form is not asking for. The
    cost is that the settings search does not reach a folded category
    while its switch is off, which is the smaller of the two.

    WHAT IS ALREADY ON THE HOST IS NOT DUPLICATED. A setting has one
    control or it has two sources of truth, and the second one silently
    loses: ``collect()`` is keyed on the setting name, so a second widget
    for ``src`` would replace the host's own and the folder the user typed
    would stop being read. Every row whose key the host already binds is
    therefore dropped, and a category left with no rows of its own is not
    mounted at all -- which is what makes "its own categories" a derived
    fact rather than a list somebody has to maintain.

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

        model = SettingsWidgets(self.key, parent=content)
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
        """What this fold contributes to the host's settings, on its own.

        The host's ``collect()`` already returns these keys mixed in with
        its own; this is the same values isolated, which is what a test --
        or a caller wanting to hand the folded module's own entry point a
        dict -- asks for.
        """
        host_model = getattr(self.screen, "_settings_model", None)
        if host_model is None:
            return {}
        values = host_model.collect()
        return {name: values[name] for name in self.settings_keys
                if name in values}


class CategoryFoldSet:
    """Every category fold on one host, and the switches that drive them.

    Holds the folds, keeps the pipeline gates consistent with which of
    them are switched on, and builds the masthead strip whose buttons do
    the switching. A host module declares what folds into it and what each
    fold gates; the mechanics are the same everywhere and live here.

    :param screen: the host module's ``AppScreen``.
    :param folds: ``key -> gate names``, in the order the strip draws them.
    :param implies: ``key -> keys``. A fold that cannot mean anything on
        its own switches the folds it depends on with it -- the motility
        assay runs inside the timelapse branch, so asking for it is asking
        for tracking too, and a form that showed the assay's knobs without
        the tracking ones would be describing a run that cannot happen.
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
        """Mount every fold's categories on the host, hidden.

        :returns: the keys that mounted. A module with nothing of its own
            to add is dropped from the set rather than given a button that
            would reveal an empty form.
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
        """Switch one fold on or off, and everything that follows from it.

        Switching a dependent fold on switches what it depends on on;
        switching a dependency off switches off whatever depended on it.
        Driven through the buttons where there are buttons, so the strip
        never shows a state the form does not have.
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
        """Write every gate's value from the folds that are switched on.

        Recomputed from all of them together rather than toggled one at a
        time: two folds can share a gate -- the assay needs the timelapse
        branch as much as tracking does -- and switching one off must not
        cancel the other.

        The values go into the settings model's defaults rather than into
        a control, because a gate has no control on this form: what turns
        it on is the button, and a checkbox saying the same thing in a
        second place is the two-sources-of-truth this fold exists to
        avoid. ``collect()`` reads the defaults for every key with no
        widget, so the run is handed what the buttons say.
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
        """Switch the folds to match a settings dict that was just loaded.

        A settings CSV written by the folded module names its gate, and
        the gate has no widget for the bulk apply to land in -- so without
        this, loading a Timelapse settings file into Mask Generation would
        fill in every tracking control and leave tracking switched off.

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
