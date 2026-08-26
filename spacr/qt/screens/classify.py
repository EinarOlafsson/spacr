"""Model-evaluation and interpretation views integrated with Classify.

The Classify masthead provides direct access to three related modules:

* Classifier Evaluation reports held-out predictions, nested
  cross-validation, calibration, leakage checks and per-plate metrics from
  a saved evaluation bundle.
* Explain CV Model reports feature gain, held-out permutation importance and
  SHAP values for classifiers trained on measured features.
* Activation maps image regions associated with predictions from an image
  classifier.

Each module opens as a complete page beside the Classify settings. The shared
page and signal integration is implemented by
:mod:`spacr.qt.screens.map_barcodes`.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Tuple

from PySide6.QtWidgets import QWidget

from ..widgets.fold_strip import FoldStrip
from . import activation
from .map_barcodes import install_fold_strip

LOG = logging.getLogger(__name__)

#: Registry key of the screen this module hangs its strip on. Classify is
#: the merged module -- Torch on crops and gradient boosting on measured
#: features behind one form -- so its key is not ``classify``.
HOST_KEY = "classify_merged"

#: Registry keys of the modules folded into it, in the order the strip
#: draws them: judge the model first, then ask which measured features it
#: is keying on, then where in the image it looked.
FOLDED_APPS: Tuple[str, ...] = ("classifier_evaluation", "explain_cv",
                                activation.APP_KEY)


def _navigable(host_window: Optional[QWidget]) -> Optional[QWidget]:
    """``host_window`` if it can be navigated, else None.

    The folded screens send the user on -- Explain CV offers to open
    Activation Maps, and both it and the evaluation screen seed a
    training screen -- through the main window's ``_on_train_requested``.
    Handing them anything else would turn one of their buttons into an
    ``AttributeError`` at the moment it was pressed.
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
    """Explain CV Model's own screen, with its navigation answered.

    The host it is given is the activation module's ``ExplainNavigator``
    rather than the window: "Open Activation Maps" now has a page on this
    very screen to land on, and asking the window for it reached a key
    nothing knows. Everything else the screen asks its host for is
    forwarded to the window unchanged, and only when that window can
    answer.
    """
    from .model_explanation import make_model_explanation_screen

    navigator = activation.ExplainNavigator(_navigable(host_window))
    screen = make_model_explanation_screen(app_key="explain_cv",
                                           host=navigator)
    navigator.attach(screen)
    return screen


def _build_activation(host_window: Optional[QWidget]) -> QWidget:
    """Activation's own screen: the attribution form and its Run button."""
    return activation.build(host_window)


#: One builder per folded module — see
#: :func:`spacr.qt.screens.map_barcodes.install_fold_strip`.
BUILDERS: Dict[str, Callable[[Optional[QWidget]], QWidget]] = {
    "classifier_evaluation": _build_classifier_evaluation,
    "explain_cv": _build_explain_cv,
    activation.APP_KEY: _build_activation,
}


def install_folds(screen: QWidget) -> Optional[FoldStrip]:
    """Put Classify's fold strip on ``screen``'s masthead."""
    return install_fold_strip(screen, HOST_KEY, FOLDED_APPS, BUILDERS)
