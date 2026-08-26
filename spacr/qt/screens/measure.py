"""Measure, and the three modules that are the rest of the visit.

ILLUMINATION CORRECTION IS PART OF A MEASURE RUN, not a run of its own.
``measure_crop`` calls
:func:`spacr.illumination.prepare_illumination_correction` itself before it
computes a single feature, and the nine ``illumination_*`` keys are a
settings category on Measure's own panel -- so the switch is thrown where
the numbers it changes are produced. What that panel cannot express is
estimating and QCing the field WITHOUT measuring the plate: the QC figure
that says how much of the position bias the correction removed is minutes
of work, and a measure run over the same plate is hours. The module
therefore keeps its own settings form and Run button, folded onto this
masthead, and the settings category and the button are two handles on one
door -- both end in the same ``prepare_illumination_correction``.

AnnData Export has no screen of its own and never wanted one: every knob
it has is a typed, tooltipped settings key, so the generic module form IS
its export dialog. What it does want is to be reachable from the screen
that wrote the tables it exports -- an export is the sentence after
"measure this plate", not a separate destination -- so it folds onto
Measure's masthead as a button.

The button is the AnnData Export icon with no text, its one-line
description as the tooltip, lit on hover in the maturity colour its tile
used -- see :class:`spacr.qt.widgets.fold_strip.FoldStrip`.

NOTHING IS LOST IN THE MOVE. The button opens the export module itself,
settings form and Run button and console, as a PAGE beside the measure
settings rather than in a window over them -- a window is the last resort
for a fold. The headless path is untouched, because the button runs the
same entry point ``spacr-run anndata_export`` runs.

AND THE ICONS ARE NOT SPENT ON THE MASTHEAD ALONE. A folded module's
picture is the thing a user already recognises it by, so it goes wherever
that module turns up: on the tab a page fold arrives as, and on the
heading of the settings category the fold left behind -- Illumination is
a button here AND one of Measure's own categories, and only one of the
two used to say so. See :data:`FOLD_CATEGORIES`.

The shared half of a fold -- opening the module, wiring the host signals
and hanging the strip off the masthead -- lives in
:mod:`spacr.qt.screens.map_barcodes` and is imported rather than
repeated.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Tuple

from PySide6.QtWidgets import QWidget

from ..widgets.fold_strip import FoldStrip, mark_folded_categories
from .map_barcodes import build_settings_screen, install_fold_strip

LOG = logging.getLogger(__name__)

#: Registry key of the screen this module hangs its strip on.
HOST_KEY = "measure"

#: Registry keys of the modules folded into it, in the order the strip
#: draws them -- which is the order a plate goes through them: the field
#: is estimated and divided out before any intensity feature is measured,
#: the AnnData file is written from the tables afterwards.
#:
#: ILLUMINATION KEEPS ITS OWN FORM as well as its settings category here,
#: because the two ask different questions. The category is "correct these
#: fields while measuring them"; the button is "estimate the field and show
#: me the QC before I commit a day to the measure run". Both end in
#: ``prepare_illumination_correction``, so there is one implementation and
#: one set of keys behind the two doors.
#:
#: THE MOTILITY ASSAY IS A MEASUREMENT. It reads finished masks from
#: ``merged/*.npy``, builds per-cell rows, writes them to
#: ``measurements/measurements.db`` and adds per-track velocities -- which
#: is this host's job description with a time axis. It does not make
#: masks, so Mask Generation was the wrong home for it however much its
#: settings looked like tracking's.
#:
#: It opens its own screen rather than becoming a settings category here,
#: and that is not the fold rule being ignored. Its module runs on a
#: folder that has ALREADY been segmented, and Measure's own run has no
#: gate that would fire it -- so there is no seam to reveal, and inventing
#: a pipeline path would be a bigger change than the fold.
FOLDED_APPS: Tuple[str, ...] = ("illumination", "anndata_export", "motility")


#: What the tiles these three folds replaced said, kept so the buttons on
#: this masthead survive the loss of their registry rows.
#:
#: A folded module has no row, and the registry answers a key it does not
#: hold the same way it answers a typo: no name, no sentence, and "stable"
#: for the maturity. Illumination and AnnData Export were both assessed as
#: beta, so without this the two buttons would promise finished code and
#: read as "Illumination" and "Anndata Export" -- the key title-cased,
#: which is not how either module spells itself.
FOLD_FALLBACK = {
    "illumination": (
        "Illumination",
        "Estimate the flat-field from the plate itself and divide it out "
        "before any intensity feature is measured",
        "beta"),
    "anndata_export": (
        "AnnData Export",
        "Write the measured objects out as an AnnData object for "
        "single-cell analysis downstream",
        "beta"),
    "motility": (
        "Motility Assay",
        "Automated motility assay: track velocity + infection QC",
        "beta"),
}


#: ``key -> the categories on THIS screen's OWN form that are its settings``.
#:
#: THE ICON GOES WHERE THE SETTINGS ARE. Illumination's other half is not a
#: page and not a mounted card: the nine ``illumination_*`` keys have been
#: one of Measure's own categories for as long as ``measure_crop`` has
#: corrected fields. So the module has a button on this masthead carrying
#: its icon and a group of settings further down carrying nothing, and a
#: user who pressed the button has no way to see that the heading below is
#: the same module. The mark on the heading is what says so.
#:
#: AnnData Export and the Motility Assay are not here, and that is not an
#: omission: neither has a settings category on this form. They arrive as
#: PAGES, and a page is already marked with its module's icon on its tab --
#: see :func:`spacr.qt.screens.map_barcodes.show_as_page`.
FOLD_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "illumination": ("Illumination Correction",),
}


def _build_illumination(host_window: Optional[QWidget]) -> QWidget:
    """Illumination Correction's own screen: estimate, QC and save a field.

    The settings-driven module, unchanged, so the Run button here runs the
    same ``prepare_illumination_correction`` a measure run calls -- and
    running it alone is the capability Measure's own settings category has
    no way to ask for: the model and its QC figures are written, and a later
    measure run reuses them through ``illumination_model``.
    """
    return build_settings_screen("illumination", host_window)


def _build_anndata_export(host_window: Optional[QWidget]) -> QWidget:
    """AnnData Export's own screen: the settings-driven module, unchanged."""
    return build_settings_screen("anndata_export", host_window)


def _build_motility(host_window: Optional[QWidget]) -> QWidget:
    """The Motility Assay's own screen, run on masks that already exist."""
    return build_settings_screen("motility", host_window)


#: One builder per folded module — see
#: :func:`spacr.qt.screens.map_barcodes.install_fold_strip`.
BUILDERS: Dict[str, Callable[[Optional[QWidget]], QWidget]] = {
    "illumination": _build_illumination,
    "anndata_export": _build_anndata_export,
    "motility": _build_motility,
}


def mark_fold_sources(screen: QWidget) -> Dict[str, Tuple[str, ...]]:
    """Put each folded module's icon beside the headings that are its own.

    Never fatal: a heading with no mark is a heading, while an exception
    raised while a screen is being built is no screen.

    :param screen: the host module's screen.
    :returns: ``key -> the category titles marked``, for the folds that
        marked at least one.
    """
    if getattr(screen, "app_key", None) != HOST_KEY:
        return {}
    try:
        return mark_folded_categories(
            getattr(screen, "_settings_sections", ()) or (), FOLD_CATEGORIES)
    except Exception:
        LOG.debug("Could not mark %s's folded categories", HOST_KEY,
                  exc_info=True)
        return {}


def install_folds(screen: QWidget) -> Optional[FoldStrip]:
    """Put Measure's fold strip on ``screen``'s masthead."""
    strip = install_fold_strip(screen, HOST_KEY, FOLDED_APPS, BUILDERS)
    # The icons the folded modules brought with them go on their settings
    # as well as on their buttons. After the strip: a mark that cannot be
    # drawn must not cost the buttons.
    mark_fold_sources(screen)
    return strip
