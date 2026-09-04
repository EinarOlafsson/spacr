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
import time
import weakref
from typing import Callable, Dict, Optional, Tuple

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..i18n import tr
from ..theme import ensure_widget_qss_applied, register_widget_qss
from ..widgets.collapsible_section import CollapsibleSection
from ..widgets.fold_strip import FoldStrip
from . import activation
from .map_barcodes import build_registered_screen, install_fold_strip

LOG = logging.getLogger(__name__)

#: Registry key of the screen this module hangs its strip on. Classify is
#: the merged module -- Torch on crops and gradient boosting on measured
#: features behind one form -- so its key is not ``classify``.
HOST_KEY = "classify_merged"

#: The settings-column surface that advertises FlowView without importing its
#: renderer.  The real panel is created only after the user opens the fold.
FLOWVIEW_SECTION_NAME = "ClassifyFlowViewSection"
FLOWVIEW_BODY_NAME = "ClassifyFlowViewBody"

#: A visualisation failure must not cost Classify its settings or Run button.
#: The exception itself is retained in the debug log; this is the useful next
#: action for the person looking at the panel.
FLOWVIEW_OPEN_ERROR = (
    "FlowView could not open. Collapse this section and open it again to retry."
)
FLOWVIEW_TOOLTIP = "Fold FlowView away, or open it again"


def _flowview_section_qss(palette: dict, _opacity=None) -> str:
    """Theme-native box around the lazily constructed FlowView renderer."""

    return f"""
QWidget#{FLOWVIEW_SECTION_NAME} {{
    background-color: {palette["surface"]};
    border: 1px solid {palette["border_soft"]};
    border-radius: 8px;
}}
QWidget#{FLOWVIEW_BODY_NAME} {{
    background: transparent;
    border: none;
}}
QWidget#{FLOWVIEW_SECTION_NAME} QWidget#FlowViewPanel {{
    background: transparent;
    border: none;
}}
"""


register_widget_qss(FLOWVIEW_SECTION_NAME, _flowview_section_qss)

#: Registry keys of the modules folded into it, in the order the strip
#: draws them.  The first three are one reading of one model, in the
#: order that reading is done: judge the model first, then ask which
#: measured features it is keying on, then where in the image it looked.
#:
#: Training Runs and Feature Explorer were appended to that sequence
#: rather than inserted into it, because neither is a step in it.
#: Training Runs diffs two runs against each other, which is a question
#: about a pair rather than about this model, and Feature Explorer ranks
#: features before anything has been trained at all -- so putting either
#: between the three would break the sentence above.
FOLDED_APPS: Tuple[str, ...] = ("classifier_evaluation", "explain_cv",
                                activation.APP_KEY, "train_compare",
                                "feature_explorer")


#: What the tiles these folds replaced said, kept so the buttons on this
#: masthead survive the loss of their registry rows.
#:
#: The registry answers a key it no longer holds exactly as it answers a
#: typo -- no name, no sentence, and "stable" for the maturity -- so
#: without this an Activation button would carry no tooltip at all and
#: light up in the colour of finished code for a module assessed as beta.
FOLD_FALLBACK = {
    activation.APP_KEY: (
        "Activation",
        "Generate activation maps",
        "beta"),
}


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
def _build_train_compare(host_window: Optional[QWidget] = None) -> QWidget:
    """Training Runs, as the window builds it.

    CLASSIFIERS, NOT CELLPOSE. The module reads the per-epoch `train.csv`
    an image-classifier run writes and diffs the two runs' settings; it
    has no notion of a segmentation model. That is what makes Classify
    its host rather than Make Masks -- "why is run B better than run A"
    is a question you only have after training something here.
    """
    return build_registered_screen("train_compare", host_window)


def _build_feature_explorer(host_window: Optional[QWidget] = None) -> QWidget:
    """Feature Explorer, as the window builds it.

    Ranks measured features by how well they separate classes, so it is
    the step BEFORE choosing what to train on -- and useless without the
    class column Classify is already pointed at.
    """
    return build_registered_screen("feature_explorer", host_window)


BUILDERS: Dict[str, Callable[[Optional[QWidget]], QWidget]] = {
    "classifier_evaluation": _build_classifier_evaluation,
    "explain_cv": _build_explain_cv,
    activation.APP_KEY: _build_activation,
    # `train_compare` still holds a registry row; `feature_explorer` never
    # had one and is declared in `app_catalog`. Neither needs a
    # `FOLD_FALLBACK` entry: `fold_description` reads the registry first
    # and the declared catalogue second, so both buttons get their real
    # name, sentence and maturity colour without a third copy here.
    "train_compare": _build_train_compare,
    "feature_explorer": _build_feature_explorer,
}


class LazyFlowViewSection(CollapsibleSection):
    """A collapsed Classify footer that pays for FlowView only when opened.

    The empty ``content`` passed to :class:`CollapsibleSection` is deliberate:
    importing the graphics renderer constructs a sizeable Qt/scientific
    dependency tree.  Classify gets only this small header on first paint;
    the panel, its scene and the approved preview graph arrive on the first
    expansion.

    :param screen: the Classify screen this footer belongs to. Read on the
        first EXPANSION rather than here, which is the whole point of the
        class: nothing about FlowView is imported until someone opens it.
    :param parent: parent widget.
    """

    OPEN_MINIMUM = 420

    def __init__(self, screen: QWidget, parent: QWidget | None = None) -> None:
        self._screen_ref = weakref.ref(screen)
        self._panel = None
        self._error_label: QLabel | None = None
        self._shut_down = False

        body = QWidget(parent)
        body.setObjectName(FLOWVIEW_BODY_NAME)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(6, 0, 6, 6)
        body_layout.setSpacing(6)
        self._body_layout = body_layout

        super().__init__("FlowView", body, expanded=False, parent=parent)
        self.setObjectName(FLOWVIEW_SECTION_NAME)
        # The section is installed after the screen's first translation pass,
        # so render its chrome immediately while retaining the English source
        # properties the next live-language pass needs.
        self._header.setProperty("_spacr_i18n_text", "FlowView")
        self._header.setText(tr("FlowView"))
        self._header.setProperty("_spacr_i18n_tooltip", FLOWVIEW_TOOLTIP)
        self._header.setToolTip(tr(FLOWVIEW_TOOLTIP))
        self.set_open_minimum(self.OPEN_MINIMUM)
        self.toggled.connect(self._flowview_toggled)

    def panel(self):
        """Return the constructed renderer, or ``None`` before first open."""

        return self._panel

    def _settings(self) -> dict:
        """Take one detached snapshot of the Classify form for the preview."""

        screen = self._screen_ref()
        model = getattr(screen, "_settings_model", None) if screen else None
        collect = getattr(model, "collect", None)
        return dict(collect() or {}) if callable(collect) else {}

    def _collector_for_open_panel(self):
        """Enable tracing and return the live collector the panel must follow.

        A fresh process owns an empty generic collector.  Replacing only that
        empty graph with Classify's approved eight-node preview gives the
        opened panel something informative immediately.  A populated global
        collector belongs to a run already under observation and wins.
        """

        from spacr.flowview.classify_blueprint import classify_graph
        from spacr.flowview.collector import Collector
        from spacr.flowview.trace import enable, get_collector

        collector = get_collector()
        try:
            has_live_graph = bool(collector.snapshot().nodes)
        except Exception:  # a broken visualisation never reaches Classify
            has_live_graph = False
        if not has_live_graph:
            graph = classify_graph(
                self._settings(),
                run_id=f"classify-preview-{time.time_ns()}",
            )
            collector = Collector(graph)
        return enable(collector)

    def _clear_error(self) -> None:
        """Remove the error label, if one is showing.

        Clears the attribute BEFORE touching the widget, so a failure while
        deleting it cannot leave the section pointing at a half-destroyed label.
        """
        label = self._error_label
        self._error_label = None
        if label is None:
            return
        self._body_layout.removeWidget(label)
        label.deleteLater()

    def _show_open_error(self, error: Exception) -> None:
        """Surface one recoverable error without letting it escape the fold."""

        self._clear_error()
        try:
            from spacr.flowview.panel import QT_MISSING_MESSAGE
        except Exception:
            message = FLOWVIEW_OPEN_ERROR
        else:
            message = QT_MISSING_MESSAGE if isinstance(error, ImportError) else (
                FLOWVIEW_OPEN_ERROR
            )
        label = QLabel(self.content())
        label.setObjectName("ClassifyFlowViewError")
        label.setProperty("_spacr_i18n_text", message)
        label.setText(tr(message))
        label.setWordWrap(True)
        self._body_layout.addWidget(label)
        self._error_label = label

    def _ensure_panel(self):
        """Build the real FlowView panel once, on the first expansion."""

        if self._panel is not None or self._shut_down:
            return self._panel
        self._clear_error()
        try:
            from spacr.flowview.panel import FlowViewPanel

            collector = self._collector_for_open_panel()
            panel = FlowViewPanel(
                collector,
                self.content(),
                auto_start=False,
                embedded=True,
            )
            self._body_layout.addWidget(panel, 1)
            self._panel = panel
            screen = self._screen_ref()
            if screen is not None:
                ensure_widget_qss_applied(FLOWVIEW_SECTION_NAME, root=screen)
        except Exception as error:  # noqa: BLE001 - optional UI isolation
            LOG.debug("could not open Classify FlowView", exc_info=True)
            self._show_open_error(error)
        return self._panel

    def _flowview_toggled(self, expanded: bool) -> None:
        """Start the panel when the section opens and stop it when it closes.

        The panel is built on first open, which is what makes the section lazy --
        FlowView is not paid for by a user who never expands it.
        """
        if expanded:
            panel = self._ensure_panel()
            if panel is not None and self.isVisible():
                panel.start()
            return
        panel = self._panel
        if panel is not None:
            panel.stop()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt virtual name
        """Resume rendering only for an actually open, visible section."""

        super().showEvent(event)
        if self.is_expanded():
            panel = self._ensure_panel()
            if panel is not None:
                panel.start()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt virtual name
        """A cached Classify page must spend no cycles while hidden."""

        panel = self._panel
        if panel is not None:
            panel.stop()
        super().hideEvent(event)

    def shutdown(self) -> None:
        """Stop and release renderer-owned Qt objects during screen teardown."""

        if self._shut_down:
            return
        self._shut_down = True
        panel = self._panel
        self._panel = None
        if panel is not None:
            panel.stop()
            panel.close()
            panel.deleteLater()
        self._clear_error()


def install_flowview(screen: QWidget) -> Optional[LazyFlowViewSection]:
    """Mount Classify's lazy FlowView box directly below its settings."""

    if getattr(screen, "app_key", None) != HOST_KEY:
        return None
    existing = getattr(screen, "_flowview_section", None)
    if isinstance(existing, LazyFlowViewSection):
        return existing
    try:
        content = getattr(screen, "_settings_content", None)
        layout = content.layout() if content is not None else None
        if layout is None:
            return None
        section = LazyFlowViewSection(screen, content)
        last = layout.itemAt(layout.count() - 1) if layout.count() else None
        insert_at = layout.count() - 1 if (
            last is not None and last.spacerItem() is not None
        ) else layout.count()
        layout.insertWidget(max(0, insert_at), section)
        screen._flowview_section = section
        ensure_widget_qss_applied(FLOWVIEW_SECTION_NAME, root=screen)
        return section
    except Exception:  # noqa: BLE001 - optional UI must not cost Classify
        LOG.debug("could not install Classify FlowView", exc_info=True)
        return None


def install_folds(screen: QWidget) -> Optional[FoldStrip]:
    """Put Classify's FlowView footer and fold strip on ``screen``."""

    try:
        install_flowview(screen)
    except Exception:  # a broken optional panel must not cost the fold strip
        LOG.debug("could not install Classify FlowView", exc_info=True)
    return install_fold_strip(screen, HOST_KEY, FOLDED_APPS, BUILDERS)
