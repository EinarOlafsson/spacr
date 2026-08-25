"""Measure, and the export that turns its tables into an AnnData file.

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

The shared half of a fold -- opening the module, wiring the host signals
and hanging the strip off the masthead -- lives in
:mod:`spacr.qt.screens.map_barcodes` and is imported rather than
repeated.
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
#: draws them.
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
FOLDED_APPS: Tuple[str, ...] = ("anndata_export", "motility")


def _build_anndata_export(host_window: Optional[QWidget]) -> QWidget:
    """AnnData Export's own screen: the settings-driven module, unchanged."""
    return build_settings_screen("anndata_export", host_window)


def _build_motility(host_window: Optional[QWidget]) -> QWidget:
    """The Motility Assay's own screen, run on masks that already exist."""
    return build_settings_screen("motility", host_window)


#: One builder per folded module — see
#: :func:`spacr.qt.screens.map_barcodes.install_fold_strip`.
BUILDERS: Dict[str, Callable[[Optional[QWidget]], QWidget]] = {
    "anndata_export": _build_anndata_export,
    "motility": _build_motility,
}


def install_folds(screen: QWidget) -> Optional[FoldStrip]:
    """Put Measure's fold strip on ``screen``'s masthead."""
    return install_fold_strip(screen, HOST_KEY, FOLDED_APPS, BUILDERS)
