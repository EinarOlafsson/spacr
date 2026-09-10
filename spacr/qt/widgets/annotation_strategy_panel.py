"""Annotation-strategy controls and reports for the Cells montage.

The panel selects from the object rows, guide wells, and score column shown in
the montage. Strategy execution runs in a worker thread, while selection,
hold-out construction, model fitting, and reporting are implemented by
:mod:`spacr.regression_annotation`.
"""
from __future__ import annotations

import logging
import os
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QFrame, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QPlainTextEdit, QPushButton,
                               QScrollArea, QSizePolicy, QSpinBox,
                               QVBoxLayout, QWidget)

LOG = logging.getLogger(__name__)

#: What the panel says before anything has been run.
NOTHING_RUN = ("Select a strategy and choose “Run the strategy”. No cells "
               "are selected and no results are written before the run.")

#: Which settings each strategy reads. Everything else is greyed with the
#: reason on the control, rather than hidden -- a setting that vanishes
#: between two strategies reads as a setting that was lost.
STRATEGY_SETTINGS: Dict[str, Tuple[str, ...]] = {
    "top_score_random": (),
    "uncertainty": ("measure",),
    "diversity": ("n_clusters",),
    "control_anchors": ("positive_control_wells", "negative_control_wells"),
    "pu_learning": (),
    "self_training": ("confidence", "rounds"),
    "two_view_disagreement": (),
    "score_strata": ("n_bins",),
    "neighbour_propagation": ("neighbours", "distance_quantile"),
    "random_holdout": (),
}

#: The leakage modes, as a chooser says them.
LEAKAGE_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("report", "report both — compare fits with and without the score's "
               "input features"),
    ("drop", "exclude them — fit only features not used to calculate the "
             "score"),
    ("keep", "include them — the fit may reproduce the original score"),
)

#: The estimators, as a chooser says them.
MODEL_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("auto", "boosted trees — XGBoost where it is installed, scikit-learn's "
             "histogram gradient boosting where it is not"),
    ("xgboost", "XGBoost"),
    ("hist_gradient_boosting", "scikit-learn histogram gradient boosting"),
)


def wells_of_plans(plans: Sequence[Any]) -> Tuple[str, ...]:
    """Return unique guide wells referenced by montage plans.

    :param plans: Montage plans in display order.
    :returns: Well identifiers in first-occurrence order.
    """
    seen: Dict[str, None] = {}
    for plan in plans or ():
        for well in getattr(plan, "wells", ()) or ():
            name = str(getattr(well, "well", "") or "").strip()
            if name:
                seen.setdefault(name, None)
    return tuple(seen)


class AnnotationStrategyPanel(QWidget):
    """Configure and run cell-annotation strategies from a montage.

    :param objects_provider: Callable returning candidate object rows, or
        ``None`` before montage data are available.
    :param wells_provider: Callable returning selected guide wells.
    :param score_provider: Callable returning the per-object score column.
    :param folder_provider: Callable returning the default output directory.
    :param parent: Parent widget.
    :param threaded: Run computation in a worker thread. Inline execution is
        intended for deterministic testing only.
    :ivar finished: Emitted with the strategy key after a successful run.
    """

    #: Emitted with the strategy key once a run has produced a result.
    finished = Signal(str)

    def __init__(self,
                 objects_provider: Optional[Callable[[], Any]] = None,
                 wells_provider: Optional[Callable[[], Sequence[str]]] = None,
                 score_provider: Optional[Callable[[], str]] = None,
                 folder_provider: Optional[Callable[[], str]] = None,
                 parent: Optional[QWidget] = None, *,
                 threaded: bool = True) -> None:
        """Build the panel that proposes which objects to annotate next.

        Every input arrives through a provider rather than being passed in: the
        panel is built before the run it describes exists, and asking at use
        time is what lets it follow a screen whose table has since changed.

        :param objects_provider: called for the object table.
        :param wells_provider: called for the wells in play.
        :param score_provider: called for the score column to rank on.
        :param folder_provider: called for the run folder to write into.
        :param parent: parent widget, or ``None``.
        :param threaded: propose on a worker thread. Set ``False`` in tests so
            a proposal finishes before it returns.
        """
        super().__init__(parent)
        from ..job_runner import JobRunner

        self._objects_provider = objects_provider
        self._wells_provider = wells_provider
        self._score_provider = score_provider
        self._folder_provider = folder_provider
        # EVERY PIECE OF STATE A CONTROL READS EXISTS BEFORE A SIGNAL IS
        # CONNECTED, which is this package's rule for a widget whose
        # handlers fire during construction.
        self._result = None
        self._running = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # THE CONTROLS SCROLL, THE REPORT DOES NOT. This panel is a tab in
        # the left half of the figures splitter, which is often 500 px tall;
        # without a scroll area the two forms are squashed until their rows
        # overlap and every value on screen is unreadable.
        controls_host = QWidget()
        controls_layout = QVBoxLayout(controls_host)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        self._menu = QComboBox()
        for entry in self._entries():
            title = entry.title if entry.implemented \
                else f"{entry.title} (not yet implemented)"
            self._menu.addItem(title, entry.key)
        self._menu.setToolTip(
            "Select how cells are prioritised for annotation. The description "
            "below states the intended use, assumptions, and computational "
            "cost of each strategy.")
        row = QHBoxLayout()
        row.addWidget(QLabel("Strategy"))
        row.addWidget(self._menu, 1)
        controls_layout.addLayout(row)

        self._about = QLabel("")
        self._about.setWordWrap(True)
        self._about.setTextInteractionFlags(Qt.TextSelectableByMouse)
        controls_layout.addWidget(self._about)

        settings = QGroupBox("How it is run")
        form = QFormLayout(settings)
        form.setLabelAlignment(Qt.AlignRight)

        self._budget = QSpinBox()
        self._budget.setRange(2, 1_000_000)
        self._budget.setValue(100)
        self._budget.setToolTip(
            "Set the target number of cells per class. This controls the "
            "positive set, matched comparison set, and annotation queue.")
        form.addRow("Cells per class", self._budget)

        self._wells = QLineEdit()
        self._wells.setPlaceholderText("filled from the coefficient on screen")
        self._wells.setToolTip(
            "Enter comma-separated guide wells from which positive candidates "
            "are selected. The montage supplies its current wells by default. "
            "Leave empty to allow all displayed wells.")
        form.addRow("Guide wells", self._wells)

        self._split = QComboBox()
        for level, text in (
                ("well", "well — cells of one well never straddle the split"),
                ("field", "field — one field's cells stay together"),
                ("plate", "plate — train on some plates, score on others"),
                ("cell", "cell — no grouping; cells from one well may occur "
                         "in both training and evaluation sets, producing an "
                         "optimistic estimate")):
            self._split.addItem(text, level)
        self._split.setToolTip(
            "Choose the experimental unit kept intact across training and "
            "evaluation sets. Well-, field-, or plate-level grouping reduces "
            "information leakage between related cells.")
        form.addRow("Independence level", self._split)

        self._holdout = QDoubleSpinBox()
        self._holdout.setRange(0.05, 0.90)
        self._holdout.setSingleStep(0.05)
        self._holdout.setDecimals(2)
        self._holdout.setValue(0.25)
        self._holdout.setToolTip(
            "Set the fraction of wells reserved before candidate selection. "
            "Reserved wells are excluded from strategy selection and used "
            "for evaluation.")
        form.addRow("Random hold-out", self._holdout)

        self._leakage = QComboBox()
        for value, text in LEAKAGE_CHOICES:
            self._leakage.addItem(text, value)
        self._leakage.setToolTip(
            "Control whether features used to calculate the source score are "
            "included in the model. Comparing both fits indicates how much "
            "performance remains without those potentially circular inputs.")
        form.addRow("The score's own inputs", self._leakage)

        self._model = QComboBox()
        for value, text in MODEL_CHOICES:
            self._model.addItem(text, value)
        form.addRow("Model", self._model)

        self._label_column = QLineEdit()
        self._label_column.setPlaceholderText(
            "leave empty to cut the score instead")
        self._label_column.setToolTip(
            "Select a column containing reference annotations. If omitted, "
            "reference labels are derived by thresholding the score and the "
            "report identifies this limitation.")
        form.addRow("Annotation column", self._label_column)

        self._seed = QSpinBox()
        self._seed.setRange(0, 1_000_000)
        self._seed.setToolTip(
            "Set the random seed used for hold-out selection, sampling, and "
            "model fitting. Reusing the seed reproduces the same selection.")
        form.addRow("Seed", self._seed)
        controls_layout.addWidget(settings)

        self._strategy_box = QGroupBox("Strategy settings")
        strategy_form = QFormLayout(self._strategy_box)
        strategy_form.setLabelAlignment(Qt.AlignRight)
        self._rows: Dict[str, Tuple[QWidget, QWidget]] = {}
        #: Each strategy setting's own help, kept because the greying
        #: replaces the tooltip with its reason and has to put the help back.
        self._row_help: Dict[str, str] = {}

        self._measure = QComboBox()
        for name in ("margin", "least_confidence", "entropy"):
            self._measure.addItem(name, name)
        self._measure.setToolTip(
            "Choose how predictive uncertainty is ranked. Margin and least "
            "confidence are equivalent for binary predictions; entropy uses "
            "the full class-probability distribution.")
        self._add_row(strategy_form, "measure", "Uncertainty measure",
                      self._measure)

        self._clusters = QSpinBox()
        self._clusters.setRange(0, 100_000)
        self._clusters.setToolTip(
            "Set the number of feature-space clusters across which the "
            "annotation budget is distributed. Zero uses one cluster per "
            "queued cell.")
        self._add_row(strategy_form, "n_clusters", "Clusters", self._clusters)

        self._bins = QSpinBox()
        self._bins.setRange(2, 100)
        self._bins.setValue(10)
        self._bins.setToolTip(
            "Divide the score distribution into this many equal-count strata "
            "and allocate the annotation budget evenly across them.")
        self._add_row(strategy_form, "n_bins", "Score strata", self._bins)

        self._confidence = QDoubleSpinBox()
        self._confidence.setRange(0.50, 0.999)
        self._confidence.setSingleStep(0.01)
        self._confidence.setDecimals(3)
        self._confidence.setValue(0.900)
        self._confidence.setToolTip(
            "Set the minimum predicted probability required to accept a "
            "pseudo-label during self-training. Lower values admit labels "
            "faster but increase the risk of error propagation.")
        self._add_row(strategy_form, "confidence", "Confidence",
                      self._confidence)

        self._rounds = QSpinBox()
        self._rounds.setRange(1, 50)
        self._rounds.setValue(5)
        self._rounds.setToolTip(
            "Set the maximum number of self-training rounds. Training stops "
            "earlier when evaluation performance no longer improves.")
        self._add_row(strategy_form, "rounds", "Rounds", self._rounds)

        self._neighbours = QSpinBox()
        self._neighbours.setRange(1, 100)
        self._neighbours.setValue(5)
        self._neighbours.setToolTip(
            "Set the number of nearest neighbours evaluated for each seed. "
            "Only neighbours within the distance threshold inherit a label.")
        self._add_row(strategy_form, "neighbours", "Neighbours per seed",
                      self._neighbours)

        self._distance = QDoubleSpinBox()
        self._distance.setRange(0.0, 1.0)
        self._distance.setSingleStep(0.05)
        self._distance.setDecimals(2)
        self._distance.setValue(0.10)
        self._distance.setToolTip(
            "Set the propagation radius as a quantile of observed nearest-"
            "neighbour distances. The resolved distance is included in the "
            "report.")
        self._add_row(strategy_form, "distance_quantile", "Distance cut",
                      self._distance)

        self._positive_wells = QLineEdit()
        self._positive_wells.setPlaceholderText("e.g. r1_c1, r1_c2")
        self._positive_wells.setToolTip(
            "Enter comma-separated wells containing experimentally defined "
            "positive controls.")
        self._add_row(strategy_form, "positive_control_wells",
                      "Positive control wells", self._positive_wells)

        self._negative_wells = QLineEdit()
        self._negative_wells.setPlaceholderText("e.g. r4_c11, r4_c12")
        self._negative_wells.setToolTip(
            "Wells whose cells are known negatives.")
        self._add_row(strategy_form, "negative_control_wells",
                      "Negative control wells", self._negative_wells)
        controls_layout.addWidget(self._strategy_box)
        controls_layout.addStretch(1)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setWidget(controls_host)
        layout.addWidget(self._scroll, 3)

        buttons = QHBoxLayout()
        self._run_button = QPushButton("Run the strategy")
        self._run_button.clicked.connect(lambda: self.run())
        buttons.addWidget(self._run_button)
        self._save_button = QPushButton("Save the selection…")
        self._save_button.clicked.connect(lambda: self.save())
        buttons.addWidget(self._save_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self._status = QLabel(NOTHING_RUN)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._report = QPlainTextEdit()
        self._report.setReadOnly(True)
        self._report.setPlaceholderText(
            "The report will describe selected cells, model fitting, "
            "evaluation data, and interpretation limits.")
        self._report.setMinimumHeight(120)
        layout.addWidget(self._report, 2)

        # THE PANEL SITS IN THE LEFT HALF OF THE FIGURES SPLITTER, which
        # starts at 780 px and floors at 520. The prose in these choosers is
        # what a combo measures itself by, so left alone the widget's minimum
        # width forces the whole regression screen wider. The words live in
        # the entries and the tooltips; the boxes elide.
        for box in (self._menu, self._split, self._leakage, self._model,
                    self._measure):
            box.setSizeAdjustPolicy(
                QComboBox.AdjustToMinimumContentsLengthWithIcon)
            box.setMinimumContentsLength(12)
            box.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        for edit in (self._wells, self._label_column, self._positive_wells,
                     self._negative_wells):
            edit.setMinimumWidth(90)
            edit.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        # A word-wrapped label asks for the width its longest paragraph
        # wants. These two are paragraphs, so they say how narrow they can
        # be and wrap instead.
        for prose in (self._about, self._status):
            prose.setMinimumWidth(160)
            prose.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        self._report.setMinimumWidth(160)

        self._jobs = JobRunner(self, threaded=bool(threaded),
                               app_key="annotation strategy")
        self._jobs.job_failed.connect(self._on_job_failed)

        self._menu.currentIndexChanged.connect(self._on_strategy_changed)
        # A FIELD THAT IS THE REASON HAS TO CLEAR THE REASON. Naming the
        # control wells is what makes the anchor strategy runnable, and a
        # button that stayed grey until something else happened would read
        # as a button that does not work.
        #
        # ON THE EMPTINESS, NOT ON EVERY KEYSTROKE: re-checking asks the
        # montage for its object rows, and a montage over a plate is not a
        # thing to concatenate once per character typed.
        self._filled: Dict[str, bool] = {}
        for name, edit in (("positive", self._positive_wells),
                           ("negative", self._negative_wells),
                           ("label", self._label_column)):
            self._filled[name] = bool(edit.text().strip())
            edit.textChanged.connect(partial(self._on_field_filled, name,
                                             edit))

        # HOVER HELP BELONGS ON A SETTING'S NAME, not on the field the user
        # is about to type into. Run BEFORE the first greying, because the
        # greying puts its reason on the field and marks it so this pass
        # leaves that reason where it can be read.
        from ..screens.settings_model import retarget_field_tooltips

        retarget_field_tooltips(self)
        self._on_strategy_changed()
        self.refresh()

    # ------------------------------------------------------------- the menu

    @staticmethod
    def _entries():
        """The strategy table. Imported here so building this widget does
        not pull the fitting machinery into a montage that never opens it."""
        from ... import regression_annotation

        return regression_annotation.STRATEGIES

    def _on_field_filled(self, name: str, edit: QLineEdit, *_args) -> None:
        """Re-check the run button when a field stops or starts being empty.

        The pre-flight reads whether the control wells and the annotation
        column are named at all, so its answer can only change when one of
        them goes from empty to filled or back.
        """
        filled = bool(edit.text().strip())
        if self._filled.get(name) is filled:
            return
        self._filled[name] = filled
        self._refresh_controls()

    def _add_row(self, form: QFormLayout, key: str, title: str,
                 widget: QWidget) -> None:
        """Add one strategy setting, remembering its label and its help."""
        label = QLabel(title)
        form.addRow(label, widget)
        self._rows[key] = (label, widget)
        self._row_help[key] = widget.toolTip()

    def strategy_key(self) -> str:
        """Return the key of the currently selected strategy."""
        return str(self._menu.currentData() or "")

    def set_strategy(self, key: str) -> bool:
        """Select a strategy by key and report whether it was available."""
        index = self._menu.findData(str(key))
        if index < 0:
            return False
        self._menu.setCurrentIndex(index)
        return True

    def about_text(self) -> str:
        """Return the current strategy description."""
        return self._about.text()

    def _on_strategy_changed(self, *_args) -> None:
        """Restate the entry, and grey the settings it does not read."""
        from ... import regression_annotation as strategies

        try:
            entry = strategies.strategy(self.strategy_key())
        except strategies.AnnotationStrategyError:
            self._about.setText("")
            return
        text = entry.describe()
        if entry.key == strategies.TOP_SCORE_RANDOM.key:
            text += (
                "\n\nUse “The score's own inputs” to evaluate circularity. "
                "Select “report both” and compare the fit marked “WITHOUT "
                "them”, which excludes those inputs; retained performance "
                "is reported.")
        self._about.setText(text)
        from ..screens.settings_model import DISABLED_REASON_TOOLTIP

        wanted = set(STRATEGY_SETTINGS.get(entry.key, ()))
        for key, (label, widget) in self._rows.items():
            used = key in wanted
            for part in (label, widget):
                part.setEnabled(used)
            if used:
                # The help goes back where the hover-help pass put it: on the
                # name, not on the field.
                label.setToolTip(self._row_help.get(key, ""))
                widget.setToolTip("")
                widget.setProperty(DISABLED_REASON_TOOLTIP, False)
                continue
            reason = (f"{entry.title} does not read this setting. It is "
                      "disabled for this strategy; its current value is "
                      "preserved for strategies that use it.")
            # ON THE FIELD AS WELL AS THE NAME, and marked so the hover-help
            # pass does not move a disabled control's reason off it.
            widget.setProperty(DISABLED_REASON_TOOLTIP, True)
            widget.setToolTip(reason)
            label.setToolTip(reason)
        self._refresh_controls()

    # ------------------------------------------------------------- the run

    def refresh(self) -> None:
        """Refresh guide wells and control state from the current montage."""
        if not self._wells.text().strip():
            wells = self._wells_provider() if self._wells_provider else ()
            if wells:
                self._wells.setText(", ".join(str(w) for w in wells))
        self._refresh_controls()

    def reason(self) -> str:
        """Return the reason the selected strategy cannot run, or ``''``.

        The check uses the loaded rows and current controls, so unsupported
        data are identified when the strategy is selected rather than after
        a model fit has started.
        """
        if self._running:
            return "A strategy is already running."
        frame = self._objects()
        if frame is None or not len(frame):
            return ("There are no cells to choose from. Select “Show the "
                    "cells” to load objects into the montage before running "
                    "a strategy.")
        from ... import regression_annotation as strategies

        try:
            entry = strategies.strategy(self.strategy_key())
        except strategies.AnnotationStrategyError as refusal:
            return str(refusal)
        if not entry.implemented:
            return (f"{entry.title} is not implemented in this release and "
                    "would select nothing.")
        try:
            return strategies.missing_requirement(
                entry, frame, self._score_column(),
                label_column=self._label_column.text().strip(),
                positive_control_wells=self._named_wells(
                    self._positive_wells),
                negative_control_wells=self._named_wells(
                    self._negative_wells))
        except Exception:
            # The execution path repeats this validation and reports a
            # specific error. A preflight exception must not disable an
            # otherwise valid strategy because of an unusual column dtype.
            LOG.debug("could not pre-flight the strategy", exc_info=True)
            return ""

    def _refresh_controls(self) -> None:
        """A control that cannot act is greyed AND says why."""
        reason = self.reason()
        self._run_button.setEnabled(not reason)
        self._run_button.setToolTip(reason or (
            "Select training cells, fit the model, predict the remaining "
            "displayed cells, and evaluate against the random hold-out."))
        savable = self._result is not None
        self._save_button.setEnabled(savable)
        self._save_button.setToolTip(
            "Write selected cells, hold-out cells, predictions, and the run "
            "report as separate files." if savable else
            "Run a strategy before saving results.")
        # A TAB THAT CANNOT BE FILLED SAYS WHY. The panel is present from the
        # moment the Cells tab is built, so before a montage has loaded the
        # reason is the only thing on it worth reading -- and it must not
        # overwrite the report of a run that has already happened.
        if reason and self._result is None and not self._running:
            self._status.setText(reason)

    def _score_column(self) -> str:
        """The per-object score column the montage is windowing on."""
        if self._score_provider is None:
            return "pred"
        try:
            return str(self._score_provider() or "pred")
        except Exception:
            LOG.debug("could not read the score column", exc_info=True)
            return "pred"

    def _objects(self):
        """The object rows to choose from, or None."""
        if self._objects_provider is None:
            return None
        try:
            return self._objects_provider()
        except Exception:
            LOG.debug("could not read the object rows", exc_info=True)
            return None

    def _named_wells(self, widget: QLineEdit) -> Tuple[str, ...]:
        """A comma-separated well list as a tuple, blanks dropped."""
        return tuple(part.strip() for part in widget.text().split(",")
                     if part.strip())

    def request(self):
        """Build an annotation request, or return ``None`` without cells."""
        frame = self._objects()
        if frame is None or not len(frame):
            return None
        from ... import regression_annotation as strategies

        return strategies.AnnotationRequest(
            frame=frame,
            score_column=self._score_column(),
            group_by=str(self._split.currentData() or "well"),
            wells=self._named_wells(self._wells),
            label_column=self._label_column.text().strip(),
            positive_control_wells=self._named_wells(self._positive_wells),
            negative_control_wells=self._named_wells(self._negative_wells),
            n_positive=int(self._budget.value()),
            holdout_fraction=float(self._holdout.value()),
            seed=int(self._seed.value()),
            leakage=str(self._leakage.currentData() or "report"),
            model=str(self._model.currentData() or "auto"),
            measure=str(self._measure.currentData() or "margin"),
            n_clusters=int(self._clusters.value()),
            n_bins=int(self._bins.value()),
            confidence=float(self._confidence.value()),
            rounds=int(self._rounds.value()),
            neighbours=int(self._neighbours.value()),
            distance_quantile=float(self._distance.value()))

    def run(self) -> bool:
        """Run the selected strategy in the configured job runner.

        :returns: ``True`` if execution was submitted; otherwise ``False``.
        """
        reason = self.reason()
        if reason:
            self._status.setText(reason)
            return False
        request = self.request()
        if request is None:
            self._status.setText(self.reason() or NOTHING_RUN)
            return False
        key = self.strategy_key()
        self._running = True
        self._result = None
        self._refresh_controls()
        self._status.setText(
            f"Running {key}… hold-out wells are reserved before candidate "
            "selection and remain excluded from training.")

        def work():
            """Run the annotation strategy. Off the GUI thread."""
            from ... import regression_annotation as strategies

            try:
                return strategies.run(key, request)
            except strategies.AnnotationStrategyError as refusal:
                return refusal

        return self._jobs.submit(work, self._on_done)

    def _on_done(self, outcome: Any) -> None:
        """A run landed. Always on the GUI thread."""
        self._running = False
        if isinstance(outcome, Exception):
            self._result = None
            self._report.setPlainText(str(outcome))
            self._status.setText(f"{self.strategy_key()} did not run: "
                                 f"{outcome}")
            self._refresh_controls()
            return
        self._result = outcome
        self._report.setPlainText(outcome.summary() if outcome is not None
                                  else NOTHING_RUN)
        roles = outcome.role_counts() if outcome is not None else {}
        chosen = sum(n for role, n in roles.items() if role != "holdout")
        self._status.setText(
            f"{outcome.title}: {chosen:,} cell(s) chosen, "
            f"{roles.get('holdout', 0):,} held aside. Reported performance is "
            "measured on the hold-out set.")
        self._refresh_controls()
        self.finished.emit(self.strategy_key())

    def _on_job_failed(self, message: str) -> None:
        """The runner itself raised. Say so rather than staying blank."""
        self._running = False
        self._status.setText(f"The strategy failed: {message}")
        self._refresh_controls()

    # ------------------------------------------------------------ the files

    def result(self):
        """Return the latest annotation result, or ``None``."""
        return self._result

    def report_text(self) -> str:
        """Return the text displayed in the report pane."""
        return self._report.toPlainText()

    def save(self, folder: Optional[str] = None) -> Dict[str, str]:
        """Write selected cells, hold-out data, predictions, and the report.

        :param folder: Output directory. If omitted, a directory chooser is
            displayed.
        :returns: Mapping from output type to path, or an empty mapping if no
            files were written.
        """
        if self._result is None:
            self._status.setText("There is nothing to save yet.")
            return {}
        target = str(folder or "")
        if not target:
            start = ""
            if self._folder_provider is not None:
                try:
                    start = str(self._folder_provider() or "")
                except Exception:
                    LOG.debug("could not read the run folder", exc_info=True)
            target = QFileDialog.getExistingDirectory(
                self, "Write the annotation selection into…", start)
        if not target:
            return {}
        folder_name = os.path.join(target, f"annotation_{self.strategy_key()}")
        try:
            written = self._result.write(folder_name)
        except OSError as error:
            self._status.setText(f"Could not write into {target}: {error}")
            return {}
        self._status.setText(
            f"Wrote {len(written)} file(s) to {folder_name}: selected cells, "
            "hold-out cells, predictions for remaining cells, and the run "
            "report.")
        return written

    def shutdown(self) -> None:
        """Stop the strategy worker before destroying the widget."""
        try:
            self._jobs.shutdown()
        except Exception:
            LOG.debug("the strategy runner would not shut down",
                      exc_info=True)

    def closeEvent(self, event):             # noqa: N802 - Qt's spelling
        """Stop active work when the panel closes."""
        self.shutdown()
        super().closeEvent(event)
