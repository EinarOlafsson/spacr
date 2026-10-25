"""Tracking settings and auxiliary pages on Mask Generation.

The existing Time control reveals the time-axis settings, mounted tracking
categories and tracking preview. Visibility does not change the ``timelapse``
pipeline setting; its checkbox remains on the form and settings files retain
their processing choices. The legacy Timelapse route opens Time and explicitly
enables tracking.

OPS and Spatial Mask Generation retain their separate page switches and run
entry points. Motility analysis remains under Measure.
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
    CategoryFoldSet, PageFold, _gate_alpha, build_settings_screen,
    hide_as_page, install_page_fold, page_fold, register_page_fold,
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

CATEGORY_FOLDS: Tuple[str, ...] = ("timelapse",)

#: The folds that open a page of their own instead.
#:
#: A page fold has no gate and no mounted categories, so it is not on the
#: strip: it is a switch in the actions row, beside Live and the dimension
#: switches, and switching it on puts the module on this screen's page
#: strip. See :func:`install_ops_switch`.
PAGE_FOLDS: Tuple[str, ...] = ("ops", "spatial_masks")

#: Name, sentence and maturity for a fold with no registry row to carry
#: them. Moved here with the fold itself, from `align.FOLD_FALLBACK`.
#:
#: ALPHA, HONESTLY. The engine has run end to end on two wells of one real
#: plate, A1 and A2, and matched the hand-driven run on A1; nothing wider
#: than that, and it does not place the phenotype images at all. A user
#: opening this should know it is new.
FOLD_FALLBACK: Dict[str, Tuple[str, str, str]] = {
    "ops": (
        "OPS",
        "For optical pooled screening: stitch each well of the sequencing "
        "acquisition, segment its nuclei and decode their barcodes.",
        "alpha"),
}

FOLD_GATES: Dict[str, Tuple[str, ...]] = {
    "timelapse": (),
}

#: ``key -> the folds it cannot run without``. Nothing here needs another
#: fold on today; the table stays because the mechanism reads it and a
#: second gated fold would otherwise arrive with no place to say so.
FOLD_IMPLIES: Dict[str, Tuple[str, ...]] = {}

FOLD_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "timelapse": ("Time Axes & Tracking (Beta)",),
}


def fold_set(screen: QWidget) -> Optional[CategoryFoldSet]:
    """The set of category folds installed on ``screen``, or None.

    :param screen: the module screen to inspect.
    """
    folds = getattr(screen, "_category_folds", None)
    return folds if isinstance(folds, CategoryFoldSet) else None


class _OfferedPreview(QObject):
    """A folded module's preview panel, offered while its switch is on.

    A ``QObject`` parented to the card, with a bound-method slot, rather
    than a closure hung off the button: a closure capturing the preview
    would keep it alive through the switch that is supposed to own it.

    Off is stronger than merely hidden. Switching the fold off unchecks
    the preview's toggle as well as hiding it, so hiding Time closes its
    optional tracking preview without changing processing settings.

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
            LOG.debug("the folded preview is gone", exc_info=True)


def fold_previews(screen: QWidget) -> Dict[str, object]:
    """The folded previews attached to ``screen``, keyed by folded app.

    :param screen: the module screen to inspect; a screen with no folded
        previews gives an empty dict.
    """
    return dict(getattr(screen, "_fold_previews", {}) or {})


def _offer_fold_previews(screen: QWidget,
                         folds: CategoryFoldSet,
                         strip=None) -> Dict[str, _OfferedPreview]:
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
        button = (strip.button_for(key) if strip is not None
                  else screen.dimension_switch("t"))
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
    button = QPushButton(EXAMPLE_BUTTON_TEXT)
    button.setToolTip(EXAMPLE_BUTTON_TOOLTIP)
    button.clicked.connect(lambda: load_the_example_images(screen))
    section.add_prose(button, at_top=True)
    screen._example_images_button = button
    return button



def _install_organize_button(screen: QWidget):
    """Add "Organize images…" above the source-directory setting.

    It opens the organizer in its images-only mode; once the images are
    moved into one Yokogawa-named folder, ``src`` and ``metadata_type``
    are set to read it. Idempotent.

    :param screen: Host module screen.
    :returns: Installed button, or ``None`` when the screen has no
        source-directory section.
    """
    from PySide6.QtWidgets import QPushButton

    from ..i18n import tr

    if getattr(screen, "app_key", None) != HOST_KEY:
        return None
    existing = getattr(screen, "_organize_images_button", None)
    if existing is not None:
        return existing
    widget = getattr(getattr(screen, "_settings_model", None),
                     "_widgets", {}).get("src")
    if widget is None:
        return None
    section = next((c for c in getattr(screen, "_settings_sections", ()) or ()
                    if hasattr(c, "add_prose") and c.isAncestorOf(widget)),
                   None)
    if section is None:
        return None
    button = QPushButton(tr("Organize images…"))
    button.setObjectName("MaskOrganizeImagesButton")
    button.setToolTip(tr(
        "Arrange intensity images from any folder structure or naming, "
        "single- or multi-channel, into the folder Mask Generation reads, "
        "and point src at it. Same popup as Make Masks' Organize for "
        "Measure."))
    button.clicked.connect(lambda _checked=False: _open_organize(screen))
    section.add_prose(button, at_top=True)
    screen._organize_images_button = button
    return button


def _open_organize(screen: QWidget):
    """Open the images-only organizer for ``screen``.

    :param screen: the Mask Generation screen.
    :returns: the popup.
    """
    from ..widgets.organize_for_measure import _open_and_organize

    model = getattr(screen, "_settings_model", None)
    try:
        source = str((model.collect() or {}).get("src") or "")
    except Exception:
        source = ""
    return _open_and_organize(
        screen, "images", source if os.path.isdir(source) else "",
        done=lambda *args: _on_organized(screen, *args))


def _on_organized(screen: QWidget, error, result, plan, target_layout) -> bool:
    """Point the form at the organised folder, or say why it failed.

    :param screen: the Mask Generation screen.
    :param error: the exception the move raised, or None.
    :param result: the :class:`spacr.channel_sorting.ApplyResult`.
    :param plan: the applied plan.
    :param target_layout: the layout written.
    :returns: whether the form now reads the folder.
    """
    from .. import prefs
    from ..i18n import tr
    from ..widgets.organize_for_measure import _mask_generation_settings

    if error is not None or result is None:
        _say(screen, tr("Organizing failed: {error}. Every move made before "
                        "the failure is listed in the manifest.",
                        error=str(error)) + "\n")
        return False
    if target_layout != "mask":
        return False
    screen.apply_settings_dict(_mask_generation_settings(plan))
    prefs.push_recent_source(HOST_KEY, result.dest)
    _say(screen, tr("Moved {moved} file(s) into {dest}; src now reads it. "
                    "Every move is in {manifest}.", moved=result.moved,
                    dest=result.dest, manifest=result.manifest) + "\n")
    return True


#: The registry key of the page fold, so the switch and the declaration
#: cannot drift apart.
OPS_KEY: str = PAGE_FOLDS[0]

#: What the switch says. AN ACRONYM, DELIBERATELY UNEXPANDED: "optical
#: pooled screening" does not fit the actions row beside Live, and OPS is
#: what the field calls it.
#: Module pages opened by an alpha-feature switch, by app key: the
#: ``ALPHA_FEATURES`` widget name of the switch that opens each one.
ALPHA_FEATURE_PAGES = {"ops": "MaskOpsSwitch"}

OPS_TOGGLE_TEXT = "OPS"

#: What it says on hover. The full name is here, where there is room for
#: it, together with the maturity -- the button is the only place a user
#: meets this module, and it is alpha.
OPS_TOGGLE_TOOLTIP = (
    "Show or hide the optical pooled screening page: stitch each well of "
    "the sequencing acquisition, segment its nuclei and decode their "
    "barcodes. Alpha: two wells of one real plate have been run through "
    "it so far."
)


def _build_ops(host_window: Optional[QWidget]) -> QWidget:
    """OPS's own screen: the settings-driven module, unchanged."""
    return build_settings_screen(OPS_KEY, host_window)


class _OpsPage(PageFold):
    """Mask Generation's OPS switch and the page it opens.

    :class:`~spacr.qt.screens.map_barcodes.PageFold` for the key ``"ops"``,
    whose behaviour this class was before it was made general: the screen
    behind the page is built once and kept, the page strip's own close mark
    puts the switch back, and a module that cannot be built leaves the
    switch off.

    :param screen: the Mask Generation screen the switch sits on.
    :param switch: the actions-row toggle.
    """

    def __init__(self, screen: QWidget, switch) -> None:
        """Record the switch and how to build the page, building neither."""
        super().__init__(screen, switch, OPS_KEY, _build_ops)


def ops_page(screen: QWidget) -> Optional["_OpsPage"]:
    """The OPS switch installed on ``screen``, or None.

    :param screen: the module screen to inspect.
    """
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
    register_page_fold(screen, page)
    screen._ops_page = page
    return page


SPATIAL_MASKS_KEY: str = PAGE_FOLDS[1]
"""The registry key of the page Mask Generation's Spatial switch opens."""

SPATIAL_MASKS_TOGGLE_TOOLTIP = (
    "Show or hide the Spatial Mask Generation page: segment a whole Xenium "
    "morphology or Visium HD H&E slide in overlapping tiles into fields "
    "Measure reads, with the fields' places on the slide recorded for "
    "Spatial Map Barcodes. Alpha."
)
"""What Mask Generation's Spatial switch says on hover."""


def spatial_masks_page(screen: QWidget) -> Optional[PageFold]:
    """The Spatial Mask Generation switch installed on ``screen``, or None.

    :param screen: the module screen to inspect.
    """
    return page_fold(screen, SPATIAL_MASKS_KEY)


def install_spatial_masks_switch(screen: QWidget,
                                 switch) -> Optional[PageFold]:
    """Wire Mask Generation's actions-row Spatial switch to its page.

    The switch is alpha: it is named for the alpha gate here and hidden at
    once unless Preferences -> Show alpha features is on.

    :param screen: the Mask Generation screen.
    :param switch: the toggle to connect.
    :returns: the installation, or None when this is not the host screen.
    """
    if getattr(screen, "app_key", None) != HOST_KEY or switch is None:
        return None
    switch.setObjectName("MaskSpatialSwitch")
    fold = install_page_fold(screen, switch, HOST_KEY, SPATIAL_MASKS_KEY)
    _gate_alpha(switch)
    return fold


def install_folds(screen: QWidget) -> Optional[FoldStrip]:
    """Mount tracking controls under the existing Time visibility switch.

    Also install the example-plate and organize buttons. Installation is
    idempotent; no additional masthead strip is created.

    :param screen: Mask Generation screen.
    :returns: ``None``; retained for callers of the former strip installer.
    """
    if getattr(screen, "app_key", None) != HOST_KEY:
        return None
    try:
        install_example_data_button(screen)
    except Exception:
        LOG.debug("Could not install the example-plate button on %s", HOST_KEY,
                  exc_info=True)
    try:
        _install_organize_button(screen)
    except Exception:
        LOG.debug("Could not install the organize button on %s", HOST_KEY,
                  exc_info=True)
    if getattr(screen, "_category_folds", None) is not None:
        return None
    try:
        switch = screen.dimension_switch("t")
        if switch is None:
            return None
        folds = CategoryFoldSet(
            screen, {key: FOLD_GATES[key] for key in CATEGORY_FOLDS},
            implies=FOLD_IMPLIES)
        if not folds.mount():
            return None
        screen._category_folds = folds
        switch.toggled.connect(lambda on: folds.set_active("timelapse", on))
        folds.set_active("timelapse", switch.isChecked())
        screen._fold_previews = _offer_fold_previews(screen, folds)
        mark_fold_sources(screen)
    except Exception:
        LOG.debug("Could not mount Time controls for %s", HOST_KEY,
                  exc_info=True)
    return None


def sync_folds(screen: QWidget,
               settings: Dict[str, object]) -> Sequence[str]:
    """Reveal imported time settings without changing processing flags.

    :param screen: Screen receiving the settings.
    :param settings: Applied settings; absent time flags preserve visibility.
    :returns: Tracking fold key when imported flags reveal Time.
    """
    from .map_barcodes import _reads_as_true

    folds = fold_set(screen)
    if folds is None:
        return ()
    try:
        named = [key for key in ("t_stack", "timelapse") if key in settings]
        if not named:
            return ()
        wanted = any(_reads_as_true(settings[key]) for key in named)
        screen.set_dimension("t", wanted)
        folds.set_active("timelapse", wanted)
        return ("timelapse",) if wanted else ()
    except Exception:
        LOG.debug("Could not sync Time from the applied settings", exc_info=True)
        return ()
