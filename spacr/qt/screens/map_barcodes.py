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
and figures -- in a window of its own, so every number the folded screen
could produce it still produces, and the mapping settings stay on screen
behind it instead of being replaced by them.

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
from typing import Callable, Dict, Optional, Sequence, Tuple

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import QWidget

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
#: The registry still wins whenever it has the row, and the pair is
#: asserted to agree for every key that has one, so the two cannot drift
#: apart while both exist.
FOLD_FALLBACK: Dict[str, Tuple[str, str, str]] = {
    "barcode_qc": (
        "Barcode QC",
        "Did the mapping run work, and where does the abundance threshold "
        "go",
        "alpha"),
    "classifier_evaluation": (
        "Classifier Evaluation",
        "Held-out predictions, nested CV, calibration, leakage and "
        "per-plate metrics",
        "alpha"),
    "explain_cv": (
        "Explain CV Model",
        "Reproduce CV decisions from measured features, then inspect gain, "
        "held-out permutation importance and SHAP",
        "alpha"),
    "agreement": (
        "Annotator Agreement",
        "Cohen's/Fleiss' κ between annotation columns + a disagreement "
        "review",
        "alpha"),
    "anndata_export": (
        "AnnData Export",
        "Write the measurements as .h5ad for scanpy and scvi-tools",
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
    fallback = FOLD_FALLBACK.get(key, ("", "", ""))
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
    if stage and button.property("stage") != stage:
        button.setProperty("stage", stage)
        # A property the stylesheet selects on is only read at polish, so
        # a button already on screen keeps the old colour until it is
        # polished again.
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

    Parented to the main window rather than left free-floating so that Qt
    keeps it alive and it closes with the application; a folded module
    held only by a local name is one the garbage collector closes the
    moment the button handler returns.
    """
    parent = owner.window() if owner is not None else None
    screen.setParent(parent, Qt.Window)
    screen.setWindowTitle(title)
    screen.resize(*FOLD_WINDOW_SIZE)
    screen.show()
    screen.raise_()
    screen.activateWindow()
    return screen


class FoldOpener:
    """Opens one folded module, and reuses the window it opened.

    A plain object rather than a closure, so the button's connection
    holds something the strip owns and that dies with it -- the reason
    :mod:`spacr.qt.recipes` gives for its own button handler.

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
        self.window: Optional[QWidget] = None

    def open(self, _checked: bool = False) -> Optional[QWidget]:
        """Show the folded module; raise the window if one is already up.

        A second press must not open a second copy of a screen that owns
        a database handle and a job runner.
        """
        existing = self.window
        if existing is not None:
            try:
                existing.show()
                existing.raise_()
                existing.activateWindow()
                return existing
            except RuntimeError:
                # Qt deleted the C++ side under us. Build a fresh one
                # rather than try to resurrect a dangling wrapper.
                self.window = None
        host_window = self.screen.window() if self.screen is not None else None
        try:
            built = self._build(host_window)
        except Exception:
            LOG.exception("Could not open the folded module %r", self.key)
            return None
        self.window = show_as_window(built, self.screen,
                                     folded_module_title(self.key))
        return self.window


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
