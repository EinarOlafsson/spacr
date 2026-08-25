"""Mask Generation, and what it does over a time series.

Timelapse is not a separate destination. It is the mask pipeline with a
flag turned on: ``preprocess_generate_masks_timelapse`` forces
``timelapse=True`` and then calls
:func:`spacr.core.preprocess_generate_masks` unchanged. Every setting it
has that Mask Generation does not is a handful of categories; everything
else -- the folder, the channels, the segmentation models, the filters,
the outputs -- it already shares with this host.

So it opens nothing. It is a switch on the Mask masthead: the module's own
icon, its one-line description as the tooltip, lit in the maturity colour
its tile used, and CHECKABLE, because being part of the run is a state
rather than an action. Pressing it reveals that module's settings
categories on the form already on screen and turns its gate on; pressing
it again hides them and turns the gate off.

WHY THIS HOST AND NOT MAKE MASKS. Make Masks is hand-curation of masks
that already exist -- one field, a brush, and a ledger. Tracking a series
and measuring how fast things move are things mask GENERATION does over a
series, and the overlap is with this module's settings, not with that
one's tools.

NOTHING IS LOST IN THE MOVE.

* Timelapse's own entry point differs from Mask's only by forcing
  ``timelapse`` on and canonicalising the tracking defaults. The switch
  forces the same flag, and the tracking defaults arrive with the
  categories -- including the ones the module renders no control for --
  so the dict handed to the pipeline is the dict its own module would
  have handed it.
* The Motility Assay runs inside the timelapse branch, so its switch
  turns Timelapse's on with it: asking for the assay is asking for
  tracking, and a form offering the assay's knobs beside no tracking
  would be describing a run that cannot happen.
* A settings file written by either module still loads: the tracking and
  assay controls are on this form and take their values like any other,
  and :func:`sync_folds` reads the gate -- which has no control -- back
  onto the buttons.

THE MOTILITY ASSAY IS NOT HERE. It reads masks rather than making them,
and what it produces is a measurements table, so it folds onto Measure.

The mechanics -- mounting a module's categories on a host, keeping the
gates consistent with the switches, and the strip itself -- are shared
with the other hosts and live in :mod:`spacr.qt.screens.map_barcodes`.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence, Tuple

from PySide6.QtWidgets import QWidget

from ..widgets.fold_strip import FoldStrip
from .map_barcodes import CategoryFoldSet

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
FOLDED_APPS: Tuple[str, ...] = ("timelapse",)

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


def fold_set(screen: QWidget) -> Optional[CategoryFoldSet]:
    """The set of category folds installed on ``screen``, or None."""
    folds = getattr(screen, "_category_folds", None)
    return folds if isinstance(folds, CategoryFoldSet) else None


def install_folds(screen: QWidget) -> Optional[FoldStrip]:
    """Put Mask Generation's two switches on ``screen``'s masthead.

    Idempotent, and defensive by design: a screen that opens without its
    switches is a smaller screen, while an exception raised here would be
    no screen at all.

    :param screen: the host module's screen.
    :returns: the strip, or None when this screen cannot carry one -- it
        is not the host, it has no masthead, one is already installed, or
        neither folded module had a category of its own to add.
    """
    if getattr(screen, "app_key", None) != HOST_KEY:
        return None
    existing = getattr(screen, "_fold_strip", None)
    if isinstance(existing, FoldStrip):
        return existing
    header = getattr(screen, "_header", None)
    if header is None or not hasattr(header, "add_trailing"):
        return None
    try:
        folds = CategoryFoldSet(
            screen,
            {key: FOLD_GATES[key] for key in FOLDED_APPS},
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
