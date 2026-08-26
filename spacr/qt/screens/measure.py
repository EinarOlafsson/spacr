"""Illumination, AnnData and motility workflows integrated with Measure.

Measure applies illumination correction through
:func:`spacr.illumination.prepare_illumination_correction`. The integrated
Illumination page can estimate and assess a correction field independently
before a full measurement run. AnnData Export provides a typed settings form
for writing measurement tables as ``.h5ad`` files. Motility Assay quantifies
objects in existing time-series masks and writes per-object and per-track
measurements.

Each workflow opens as a complete page beside the Measure settings and retains
its headless entry point. Shared page and signal integration is implemented by
:mod:`spacr.qt.screens.map_barcodes`.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Tuple

from PySide6.QtWidgets import QWidget

from ..widgets.fold_strip import FoldStrip
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


def install_folds(screen: QWidget) -> Optional[FoldStrip]:
    """Put Measure's fold strip on ``screen``'s masthead."""
    return install_fold_strip(screen, HOST_KEY, FOLDED_APPS, BUILDERS)
