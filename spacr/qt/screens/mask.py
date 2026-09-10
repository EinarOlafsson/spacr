"""The modules folded onto Mask Generation: Timelapse, and OPS.

TWO KINDS OF FOLD MEET HERE, and the difference is what each one does to
this screen. Timelapse is SETTINGS: switching it on mounts its categories
on Mask's own form and sets the pipeline gate the run reads, so its switch
belongs on the masthead strip beside the heading. OPS is a PAGE: it has 63
settings of its own and its own run entry point, none of which is
`preprocess_generate_masks`, so switching it on puts it on this screen's
page strip instead and its switch sits in the actions row beside Live --
"add the ops button to Mask add it beside the Live and 3D buttons in the
same format", 2026-09-09. See :data:`CATEGORY_FOLDS`, :data:`PAGE_FOLDS`
and :func:`install_ops_switch`.

The Timelapse switch enables ``timelapse=True`` for the standard mask
pipeline and displays the additional time-series and tracking settings on the
Mask form. Settings files created for the Timelapse module remain compatible;
:func:`sync_folds` restores the switch state from the pipeline gate.

The integrated tracking preview evaluates object linkage across frames,
whereas the standard Mask preview evaluates segmentation of an individual
field. Motility analysis remains under Measure because it quantifies existing
masks and writes measurements rather than generating masks.

Category mounting and fold-state synchronization are shared with the other
host screens through :mod:`spacr.qt.screens.map_barcodes`.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Sequence, Tuple

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QWidget

from ..widgets.fold_strip import (
    FoldStrip, mark_folded_categories, mark_folded_sections,
)
from .map_barcodes import (
    CategoryFoldSet, FoldOpener, build_settings_screen, hide_as_page,
)

LOG = logging.getLogger(__name__)

#: Registry key of the screen this module hangs its strip on.
HOST_KEY = "mask"

#: Registry keys of the modules folded into it.
#:
#: TRACKING ONLY. Timelapse belongs here because it is segmentation with a
#: time axis: it assigns one identity to a mask across frames, and it
#: overlaps this host's settings almost entirely.
#:
#: The Motility Assay does not. It reads finished masks and WRITES A
#: MEASUREMENTS TABLE -- per-cell rows and per-track velocities into
#: measurements/measurements.db -- which is Measure's job description with
#: a time axis rather than this one's. It folds onto Measure.
#: OPS IS FOLDED HERE TOO, AND IT IS NOT A SETTINGS CATEGORY. Optical
#: pooled screening was folded onto Align & Stitch when it was built,
#: because it is stitching; it is reached from Mask Generation instead,
#: because segmentation is what the plate it stitches is acquired for and
#: this is the screen a user of it is already on. It opens a page of its
#: own rather than mounting categories on this form -- its 63 settings and
#: its own run entry point are not `preprocess_generate_masks` -- so it is
#: in :data:`PAGE_FOLDS` below and not in :data:`FOLD_GATES`.
FOLDED_APPS: Tuple[str, ...] = ("timelapse", "ops")

#: The folds that ARE settings categories on this screen's own form.
#:
#: This is what the masthead strip is built from and what
#: :class:`~spacr.qt.screens.map_barcodes.CategoryFoldSet` mounts. Every
#: key here needs a row in :data:`FOLD_GATES`, because switching one on is
#: exactly setting its gate.
CATEGORY_FOLDS: Tuple[str, ...] = ("timelapse",)

#: The folds that open a page of their own instead.
#:
#: A page fold has no gate and no mounted categories, so it is not on the
#: strip: it is a switch in the actions row, beside Live and the dimension
#: switches, and switching it on puts the module on this screen's page
#: strip. See :func:`install_ops_switch`.
PAGE_FOLDS: Tuple[str, ...] = ("ops",)

#: Name, sentence and maturity for a fold with no registry row to carry
#: them. Moved here with the fold itself, from `align.FOLD_FALLBACK`.
#:
#: ALPHA, HONESTLY. The stitching is measured correct against a plate with
#: known geometry and the decode chain is validated against planted
#: barcodes, but neither has met a real acquisition, one class in
#: `spacrops` has never been run at all, and the phenotype alignment's
#: geometry is unverified. A user opening this should know it is new.
FOLD_FALLBACK: Dict[str, Tuple[str, str, str]] = {
    "ops": (
        "OPS",
        "Stitch a low-magnification genotype acquisition into per-well "
        "mosaics and place the high-magnification phenotype images onto "
        "them, for optical pooled screening.",
        "alpha"),
}

#: ``key -> the settings the pipeline reads to decide it should do this``.
#:
#: These are the seam. :func:`spacr.core.preprocess_generate_masks` groups
#: a plate into time stacks when ``timelapse`` is true, and
#: :mod:`spacr.object` calls :func:`spacr.timelapse.automated_motility_assay`
#: when ``timelapse and motility_analysis`` are both true -- so switching a
#: fold on is exactly setting its gate, and no new pipeline path is
#: involved.
#:
FOLD_GATES: Dict[str, Tuple[str, ...]] = {
    "timelapse": ("timelapse",),
}

#: ``key -> the folds it cannot run without``. Nothing here needs another
#: fold on today; the table stays because the mechanism reads it and a
#: second gated fold would otherwise arrive with no place to say so.
FOLD_IMPLIES: Dict[str, Tuple[str, ...]] = {}

#: ``key -> the categories on THIS screen's OWN form that are its settings``.
#:
#: The fold mounts the categories Mask Generation does not already offer,
#: and those are marked from the fold itself. This table is for the other
#: half: the time-axis settings Mask Generation has always drawn itself,
#: which are Timelapse's subject however early they were written. Marking
#: them says the same thing the mounted cards say -- these belong to the
#: module the switch on the masthead turns on -- rather than leaving one
#: half of a module's settings attributed and the other half anonymous.
FOLD_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "timelapse": ("Time Axes & Tracking (Beta)",),
}


def fold_set(screen: QWidget) -> Optional[CategoryFoldSet]:
    """The set of category folds installed on ``screen``, or None."""
    folds = getattr(screen, "_category_folds", None)
    return folds if isinstance(folds, CategoryFoldSet) else None


class _OfferedPreview(QObject):
    """A folded module's preview panel, offered while its switch is on.

    A ``QObject`` parented to the card, with a bound-method slot, rather
    than a closure hung off the button: a closure capturing the preview
    would keep it alive through the switch that is supposed to own it.

    Off is stronger than merely hidden. Switching the fold off unchecks
    the preview's toggle as well as hiding it, so the card cannot stay on
    screen showing tracks for a run that is no longer tracking.

    :param host: the screen offering the preview. Its ``card`` is the
        QObject parent -- not the host itself, which is what the note above
        means about not keeping it alive through the switch that owns it.
    """

    def __init__(self, host) -> None:
        """Parent to the host's CARD, not the host. See the class note."""
        super().__init__(host.card)
        self._host = host

    def set_offered(self, on: bool) -> None:
        """Show or hide the toggle, and close the card when hiding it."""
        on = bool(on)
        try:
            if not on:
                self._host.toggle.setChecked(False)
                self._host.card.setVisible(False)
            self._host.toggle.setVisible(on)
        except RuntimeError:
            # Qt deleted the card under us; nothing left to offer.
            LOG.debug("the folded preview is gone", exc_info=True)


def fold_previews(screen: QWidget) -> Dict[str, object]:
    """The folded previews attached to ``screen``, keyed by folded app."""
    return dict(getattr(screen, "_fold_previews", {}) or {})


def _offer_fold_previews(screen: QWidget,
                         folds: CategoryFoldSet,
                         strip: FoldStrip) -> Dict[str, _OfferedPreview]:
    """Attach each mounted fold's own preview panel, hidden behind its switch.

    A module whose preview is its whole reason for a runtime panel loses
    it the day its row is dropped, because nothing builds its screen any
    more. Attaching it to the host is what keeps the capability, and
    hiding it behind the switch is what stops Mask Generation growing a
    track preview for a run with no time axis in it.

    Guarded per fold: a preview that cannot be built costs that one
    panel, never the strip.
    """
    from ..preview_registry import attach_folded

    offered: Dict[str, _OfferedPreview] = {}
    for key in folds.order:
        button = strip.button_for(key)
        if button is None:
            continue
        try:
            host = attach_folded(screen, key)
        except Exception:
            LOG.debug("Could not attach %s's preview to %s", key, HOST_KEY,
                      exc_info=True)
            continue
        if host is None:
            continue
        watcher = _OfferedPreview(host)
        button.toggled.connect(watcher.set_offered)
        watcher.set_offered(button.isChecked())
        offered[key] = watcher
    return offered


def mark_fold_sources(screen: QWidget) -> Dict[str, Tuple[str, ...]]:
    """Mark settings categories with their folded module icons.

    Both mounted categories and host categories listed in
    :data:`FOLD_CATEGORIES` are marked. Repeated calls are idempotent, and a
    missing icon does not interrupt screen construction.

    :param screen: Host module screen.
    :returns: Mapping from folded application keys to marked category titles.
    """
    marked: Dict[str, Tuple[str, ...]] = {}
    folds = fold_set(screen)
    for key, fold in (folds.folds.items() if folds is not None else ()):
        try:
            titles = mark_folded_sections(key, getattr(fold, "sections", ()))
        except Exception:
            LOG.debug("Could not mark %s's mounted categories", key,
                      exc_info=True)
            continue
        if titles:
            marked[key] = titles
    try:
        own = mark_folded_categories(
            getattr(screen, "_settings_sections", ()) or (), FOLD_CATEGORIES)
    except Exception:
        LOG.debug("Could not mark %s's own folded categories", HOST_KEY,
                  exc_info=True)
        own = {}
    for key, titles in own.items():
        marked[key] = marked.get(key, ()) + titles
    return marked


# ---------------------------------------------------------------------------
# The example plate
# ---------------------------------------------------------------------------

#: The folder the example plate is written into, under the same cache the
#: example screen already uses, so one directory holds everything spaCR
#: fetched or made for a user who wanted something to press Run on.
EXAMPLE_PLATE_DIRNAME = "mask_example_plate"

#: The button, and what it says it does.
EXAMPLE_BUTTON_TEXT = "Load the example images…"
EXAMPLE_BUTTON_TOOLTIP = (
    "Generate a reproducible example microscopy plate in the spaCR cache, "
    "then fill this form with its folder, channels, and acquisition settings. "
    "This operation does not require a network connection.")


def example_plate_folder() -> str:
    """Return the cache directory used for the example microscopy plate."""
    from ...example_data import cache_folder

    return os.path.join(cache_folder(), EXAMPLE_PLATE_DIRNAME)


def load_the_example_images(screen: QWidget,
                            folder: str = "") -> Dict[str, object]:
    """Generate an example plate and apply its settings to Mask Generation.

    The reproducible synthetic fields follow the filename and channel layout
    expected by the default pipeline. A summary of written images, applied
    settings, and unavailable controls is displayed in the screen console.

    :param screen: Mask Generation screen.
    :param folder: Output directory. Defaults to
        :func:`example_plate_folder`.
    :returns: Result containing ``folder``, ``images``, ``written``,
        ``applied``, ``filled``, and ``unplaced``. Returns an empty mapping if
        the plate could not be written.
    """
    from ..synthetic import demo_settings, generate_mask_demo

    dst = str(folder or example_plate_folder())
    try:
        before = set(os.listdir(dst)) if os.path.isdir(dst) else set()
    except OSError:
        before = set()
    try:
        layout = generate_mask_demo(dst)
    except Exception as error:                                  # noqa: BLE001
        LOG.debug("Could not write the example plate", exc_info=True)
        _say(screen, f"The example plate could not be written to {dst}: "
                     f"{error}\n")
        return {}

    images = [str(path) for path in layout.image_files]
    written = [path for path in images
               if os.path.basename(path) not in before]
    settings = demo_settings("mask", str(layout.src))
    applied = int(screen.apply_settings_dict(settings) or 0)

    widgets = getattr(getattr(screen, "_settings_model", None), "_widgets", {})
    filled = sorted(key for key in settings if key in widgets)
    unplaced = sorted(key for key in settings if key not in widgets)
    notes = layout.notes or {}
    channels = ", ".join(str(c) for c in notes.get("channels", ()))
    said = (f"Example plate ready: {len(images)} image(s) in {dst} "
            f"({len(written)} written now, {len(images) - len(written)} "
            f"already there) — {notes.get('n_fields', 0)} field(s), "
            f"channels {channels}.\n"
            f"src is now {dst}, and {applied} setting(s) on this form were "
            f"filled from the plate: {', '.join(filled)}.\n")
    if unplaced:
        said += (f"{len(unplaced)} of the plate's settings have no control on "
                 f"this form and were not filled: {', '.join(unplaced)}.\n")
    _say(screen, said)
    return {"folder": dst, "images": images, "written": written,
            "applied": applied, "filled": filled, "unplaced": unplaced}


def _say(screen: QWidget, text: str) -> None:
    """Put ``text`` in the screen's console, if it has one."""
    console = getattr(screen, "_console", None)
    if console is None or not hasattr(console, "append_stdout"):
        return
    try:
        console.append_stdout(text)
    except Exception:                                           # noqa: BLE001
        LOG.debug("Could not write to the console", exc_info=True)


def install_example_data_button(screen: QWidget):
    """Add an example-plate button above the source-directory setting.

    Installation is idempotent. The target section is identified from its
    ``src`` control so category renaming does not break placement.

    :param screen: Host module screen.
    :returns: Installed button, or ``None`` if the screen has no suitable
        source-directory section.
    """
    from PySide6.QtWidgets import QPushButton

    if getattr(screen, "app_key", None) != HOST_KEY:
        return None
    existing = getattr(screen, "_example_images_button", None)
    if existing is not None:
        return existing
    widget = getattr(getattr(screen, "_settings_model", None),
                     "_widgets", {}).get("src")
    if widget is None:
        return None
    section = None
    for candidate in getattr(screen, "_settings_sections", ()) or ():
        if hasattr(candidate, "add_prose") and candidate.isAncestorOf(widget):
            section = candidate
            break
    if section is None:
        return None
    # The English source, the way every other caption in the tool is
    # written: the language pass walks the widget tree and renders it from
    # the catalog, and a caption translated here would be rendered twice.
    button = QPushButton(EXAMPLE_BUTTON_TEXT)
    button.setToolTip(EXAMPLE_BUTTON_TOOLTIP)
    button.clicked.connect(lambda: load_the_example_images(screen))
    section.add_prose(button, at_top=True)
    screen._example_images_button = button
    return button


# ---------------------------------------------------------------------------
# The OPS switch
# ---------------------------------------------------------------------------

#: The registry key of the page fold, so the switch and the declaration
#: cannot drift apart.
OPS_KEY: str = PAGE_FOLDS[0]

#: What the switch says. AN ACRONYM, DELIBERATELY UNEXPANDED: "optical
#: pooled screening" does not fit the actions row beside Live, and OPS is
#: what the field calls it.
OPS_TOGGLE_TEXT = "OPS"

#: What it says on hover. The full name is here, where there is room for
#: it, together with the maturity -- the button is the only place a user
#: meets this module, and it is alpha.
OPS_TOGGLE_TOOLTIP = (
    "Show or hide the optical pooled screening page: stitch a "
    "low-magnification genotype acquisition into per-well mosaics and "
    "place the high-magnification phenotype images onto them. Alpha -- "
    "no real acquisition has been through it yet."
)


def _build_ops(host_window: Optional[QWidget]) -> QWidget:
    """OPS's own screen: the settings-driven module, unchanged."""
    return build_settings_screen(OPS_KEY, host_window)


class _OpsPage(QObject):
    """Mask Generation's OPS switch and the page it opens.

    A ``QObject`` parented to the host screen with bound-method slots,
    rather than closures hung off the switch: the page strip's own close
    mark can take the page down without telling the switch, so something
    has to be connected to ``tabCloseRequested`` to put the switch back.
    A closure doing that would keep the host alive through the switch.

    The screen behind the page is built once and kept, the way every
    other fold's is -- pressing the switch again puts the SAME OPS screen
    back, with whatever it had loaded still loaded.

    :param screen: the Mask Generation screen the switch sits on.
    :param switch: the actions-row toggle.
    """

    def __init__(self, screen: QWidget, switch) -> None:
        """Record the switch and how to build the page, building neither."""
        super().__init__(screen)
        self.screen = screen
        self.switch = switch
        self.opener = FoldOpener(screen, OPS_KEY, _build_ops)
        self.page: Optional[QWidget] = None
        self._watching = False

    # -- the switch ----------------------------------------------------
    def set_shown(self, on: bool) -> None:
        """Open or close the OPS page.

        Never raises: this is a slot on a user's click, and a module that
        cannot be built must cost its own page and nothing else. A failed
        open puts the switch back off rather than leaving it lit over
        nothing.

        :param on: the switch's new state.
        """
        if on:
            try:
                self.page = self.opener.open()
            except Exception:                            # noqa: BLE001
                LOG.debug("Could not open the %s page", OPS_KEY,
                          exc_info=True)
                self.page = None
            if self.page is None:
                self._restate(False)
                return
            self._watch_the_strip()
            return
        page = self.page or getattr(self.opener, "window", None)
        self.page = None
        if page is None:
            return
        try:
            if not hide_as_page(page, self.screen):
                # Shown as a window instead, because this host had no page
                # strip to put it on. Hidden, not closed: same screen back.
                page.hide()
        except Exception:                                # noqa: BLE001
            LOG.debug("Could not close the %s page", OPS_KEY, exc_info=True)

    # -- the page strip ------------------------------------------------
    def _watch_the_strip(self) -> None:
        """Follow the page's own close mark, once the strip exists.

        The strip is created by the first page opened on this host, so
        this cannot be connected in ``__init__``.
        """
        if self._watching:
            return
        pages = getattr(self.screen, "_fold_pages", None)
        if pages is None:
            return
        try:
            pages.tabCloseRequested.connect(self._page_closed)
        except Exception:                                # noqa: BLE001
            LOG.debug("Could not follow the page strip for %s", OPS_KEY,
                      exc_info=True)
            return
        self._watching = True

    def _page_closed(self, index: int) -> None:
        """Put the switch back when the page is closed by its own cross.

        :param index: the page the strip is closing.
        """
        pages = getattr(self.screen, "_fold_pages", None)
        if pages is None or self.page is None:
            return
        try:
            ours = pages.indexOf(self.page)
        except RuntimeError:            # Qt deleted the page under us
            ours = -1
        if ours not in (index, -1):
            return
        self.page = None
        self._restate(False)

    def _restate(self, on: bool) -> None:
        """Move the switch without asking it to open or close anything.

        :param on: the state the switch should show.
        """
        switch = self.switch
        if switch is None or bool(switch.isChecked()) == bool(on):
            return
        blocked = switch.blockSignals(True)
        try:
            switch.setChecked(on)
        finally:
            switch.blockSignals(blocked)


def ops_page(screen: QWidget) -> Optional["_OpsPage"]:
    """The OPS switch installed on ``screen``, or None."""
    page = getattr(screen, "_ops_page", None)
    return page if isinstance(page, _OpsPage) else None


def install_ops_switch(screen: QWidget, switch) -> Optional["_OpsPage"]:
    """Wire an actions-row switch to the OPS page.

    Idempotent: a second call on the same screen returns the first
    installation and connects nothing further.

    :param screen: the Mask Generation screen.
    :param switch: the toggle to connect.
    :returns: the installation, or None when this is not the host screen.
    """
    if getattr(screen, "app_key", None) != HOST_KEY or switch is None:
        return None
    existing = ops_page(screen)
    if existing is not None:
        return existing
    page = _OpsPage(screen, switch)
    switch.toggled.connect(page.set_shown)
    # The installation outlives this call only because the screen holds it.
    screen._ops_page = page
    return page


def install_folds(screen: QWidget) -> Optional[FoldStrip]:
    """Put Mask Generation's fold switches on ``screen``'s masthead.

    And the example-plate button on its form: this is the one walk that
    reaches this screen from outside, so everything hung on Mask Generation
    is hung here. See :func:`install_example_data_button`.

    Idempotent, and defensive by design: a screen that opens without its
    switches is a smaller screen, while an exception raised here would be
    no screen at all.

    :param screen: the host module's screen.
    :returns: the strip, or None when this screen cannot carry one -- it
        is not the host, it has no masthead, one is already installed, or
        no folded module had a category of its own to add.
    """
    if getattr(screen, "app_key", None) != HOST_KEY:
        return None
    # BEFORE the folds, and outside their guard: the example plate is what a
    # user with no data of their own presses first, and a fold that cannot be
    # mounted must not take it away.
    try:
        install_example_data_button(screen)
    except Exception:
        LOG.debug("Could not install the example-plate button on %s", HOST_KEY,
                  exc_info=True)
    existing = getattr(screen, "_fold_strip", None)
    if isinstance(existing, FoldStrip):
        return existing
    header = getattr(screen, "_header", None)
    if header is None or not hasattr(header, "add_trailing"):
        return None
    try:
        folds = CategoryFoldSet(
            screen,
            {key: FOLD_GATES[key] for key in CATEGORY_FOLDS},
            implies=FOLD_IMPLIES,
        )
        if not folds.mount():
            return None
        strip = folds.build_strip(header)
        if strip is None:
            return None
        header.add_trailing(strip)
    except Exception:
        LOG.debug("Could not build the fold strip for %s", HOST_KEY,
                  exc_info=True)
        return None
    # The set outlives this call only because the screen holds it.
    screen._category_folds = folds
    screen._fold_strip = strip
    # The panels the folded modules brought with them, offered by their
    # own switches. After the strip is on the masthead: a preview that
    # could not be built must not cost the switches.
    screen._fold_previews = _offer_fold_previews(screen, folds, strip)
    # And the folded modules' icons, on the headings of the settings they
    # became. After the strip, for the same reason the previews are: a
    # mark that cannot be drawn must not cost the switches.
    mark_fold_sources(screen)
    return strip


def sync_folds(screen: QWidget,
               settings: Dict[str, object]) -> Sequence[str]:
    """Move the switches to match a settings dict that was just applied.

    The tracking and assay CONTROLS take their values through the ordinary
    bulk apply, because they are ordinary controls on this form. Their
    gates are not controls -- the switch is -- so a Timelapse settings file
    would otherwise fill in every tracking knob and leave tracking off.

    Safe to call on any screen: one that has no category folds returns an
    empty tuple.

    :param screen: the screen the settings were applied to.
    :param settings: the dict that was applied.
    :returns: the folded keys the settings switched on.
    """
    folds = fold_set(screen)
    if folds is None:
        return ()
    try:
        return folds.sync_from_settings(settings)
    except Exception:
        LOG.debug("Could not sync the folds from the applied settings",
                  exc_info=True)
        return ()
