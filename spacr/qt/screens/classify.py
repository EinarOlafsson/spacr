"""Classify, and the two screens that judge what it produced.

A classifier is trained on one screen and then argued about on two
others: Classifier Evaluation asks whether to believe it -- held-out
predictions, nested CV, calibration, leakage checks and per-plate metrics
off a saved evaluation bundle -- and Explain CV Model asks what it is
keying on, reproducing the decisions from measured features and reporting
gain, held-out permutation importance and SHAP. Neither is a separate
destination; both are the second half of the visit that trained the
model, so both fold onto Classify's masthead as buttons.

Each button is the folded module's own icon with no text, its one-line
description as the tooltip, lit on hover in the maturity colour its tile
used -- see :class:`spacr.qt.widgets.fold_strip.FoldStrip`.

NOTHING IS LOST IN THE MOVE. The buttons open the two modules
themselves, in windows of their own, so the bundle browser, the leakage
report, the backend choice, the SHAP panel and every navigation those
screens offer arrive with them, and the training settings stay on screen
behind them.

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
from .map_barcodes import install_fold_strip

LOG = logging.getLogger(__name__)

#: Registry key of the screen this module hangs its strip on. Classify is
#: the merged module -- Torch on crops and gradient boosting on measured
#: features behind one form -- so its key is not ``classify``.
HOST_KEY = "classify_merged"

#: Registry keys of the modules folded into it, in the order the strip
#: draws them: judge the model first, then ask what it is keying on.
FOLDED_APPS: Tuple[str, ...] = ("classifier_evaluation", "explain_cv")


def _navigable(host_window: Optional[QWidget]) -> Optional[QWidget]:
    """``host_window`` if it can be navigated, else None.

    The two folded screens send the user on -- Explain CV offers to open
    Activation Maps, and both seed a training screen -- through the main
    window's ``_on_train_requested``. Handing them anything else would
    turn one of their buttons into an ``AttributeError`` at the moment it
    was pressed.
    """
    if host_window is None:
        return None
    return host_window if callable(
        getattr(host_window, "_on_train_requested", None)) else None


def _build_classifier_evaluation(host_window: Optional[QWidget]) -> QWidget:
    """Classifier Evaluation's own screen, bundle browser included."""
    from .classifier_evaluation import ClassifierEvaluationScreen
    return ClassifierEvaluationScreen()


def _build_explain_cv(host_window: Optional[QWidget]) -> QWidget:
    """Explain CV Model's own screen, with navigation where it works."""
    from .model_explanation import make_model_explanation_screen
    return make_model_explanation_screen(app_key="explain_cv",
                                         host=_navigable(host_window))


#: One builder per folded module — see
#: :func:`spacr.qt.screens.map_barcodes.install_fold_strip`.
BUILDERS: Dict[str, Callable[[Optional[QWidget]], QWidget]] = {
    "classifier_evaluation": _build_classifier_evaluation,
    "explain_cv": _build_explain_cv,
}


def install_folds(screen: QWidget) -> Optional[FoldStrip]:
    """Put Classify's fold strip on ``screen``'s masthead."""
    return install_fold_strip(screen, HOST_KEY, FOLDED_APPS, BUILDERS)
