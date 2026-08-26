"""The Cells tab's annotation strategies: the menu, the run, and the report.

The panel is a tab beside the montage's Summary, not a window, because the
cells it chooses from are the ones already on screen: the object rows the
montage loaded, the guide wells the coefficient picked, and the score
column the montage windowed on.

WHAT THE PANEL IS RESPONSIBLE FOR, and :mod:`spacr.regression_annotation`
is not: naming every strategy with what it is for and what it costs before
it is chosen; greying the settings a strategy does not read, with the
reason on the control; and putting the run on a worker thread, because
fitting a boosted tree over a screen is seconds to minutes and the GUI
thread may not spend them.

WHAT THE MODULE IS RESPONSIBLE FOR, and the panel never second-guesses:
which cells are chosen, how the hold-out is drawn, and what the fit is
allowed to claim. The panel shows :meth:`AnnotationResult.summary` as the
module wrote it, so a number on screen is the number the module measured.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QFrame, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QPlainTextEdit, QPushButton,
                               QScrollArea, QSizePolicy, QSpinBox,
                               QVBoxLayout, QWidget)

LOG = logging.getLogger(__name__)

#: What the panel says before anything has been run.
NOTHING_RUN = ("Pick a strategy and press “Run the strategy”. Nothing is "
               "chosen, fitted or written until you do.")

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
    ("report", "report both — fit with the score's own inputs and without "
               "them, and say how much of the fit survives"),
    ("drop", "drop them — fit only on columns the score is not a function of"),
    ("keep", "keep them — the naive fit, which can succeed by relearning the "
             "score"),
)

#: The estimators, as a chooser says them.
MODEL_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("auto", "boosted trees — XGBoost where it is installed, scikit-learn's "
             "histogram gradient boosting where it is not"),
    ("xgboost", "XGBoost"),
    ("hist_gradient_boosting", "scikit-learn histogram gradient boosting"),
)


def wells_of_plans(plans: Sequence[Any]) -> Tuple[str, ...]:
    """The wells a montage's plans named, in the order they were picked.

    These are the chosen guide wells: the wells the count data reported the
    coefficient present in, which is exactly the population the named
    strategy takes its positives from.

    :param plans: the montage plans on screen.
    :returns: the well names, first seen first.
    """
    seen: Dict[str, None] = {}
    for plan in plans or ():
        for well in getattr(plan, "wells", ()) or ():
            name = str(getattr(well, "well", "") or "").strip()
            if name:
                seen.setdefault(name, None)
    return tuple(seen)


class AnnotationStrategyPanel(QWidget):
    """Choose a strategy, run it off the GUI thread, and read what it did.

    :param objects_provider: returns the object rows to choose from, or
        None when the montage has not loaded any.
    :param wells_provider: returns the chosen guide wells.
    :param score_provider: returns the per-object score column.
    :param folder_provider: returns where a saved result should go.
    :param parent: the tab widget the panel sits in.
    :param threaded: False runs the strategy inline, which is what a test
        wants and what a user must never have.
    :ivar finished: emitted with the strategy key when a run lands.
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
            "Which cells get annotated is the experiment, so the choice is a "
            "named one rather than a default. Each entry says what it is for "
            "and what it costs below.")
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
            "How many cells the positive set holds, how many the matched "
            "contrast draw holds, and how long a queue is. It is an "
            "annotation budget: the number of cells somebody will look at.")
        form.addRow("Cells per class", self._budget)

        self._wells = QLineEdit()
        self._wells.setPlaceholderText("filled from the coefficient on screen")
        self._wells.setToolTip(
            "The guide wells the positives are taken from, comma separated. "
            "Filled in from the wells the montage's own plans named, which "
            "are the wells the count data reported this coefficient in. "
            "Empty means the whole screen is eligible.")
        form.addRow("Guide wells", self._wells)

        self._split = QComboBox()
        for level, text in (
                ("well", "well — cells of one well never straddle the split"),
                ("field", "field — one field's cells stay together"),
                ("plate", "plate — train on some plates, score on others"),
                ("cell", "cell — NO grouping; sibling cells of one well land "
                         "on both sides and the score comes out optimistic")):
            self._split.addItem(text, level)
        self._split.setToolTip(
            "What a split may not cross. Cells of one well are not "
            "independent, so a split that puts some of a well's cells in "
            "train and others in test reports a score the model will not "
            "reach on a new plate.")
        form.addRow("Independence level", self._split)

        self._holdout = QDoubleSpinBox()
        self._holdout.setRange(0.05, 0.90)
        self._holdout.setSingleStep(0.05)
        self._holdout.setDecimals(2)
        self._holdout.setValue(0.25)
        self._holdout.setToolTip(
            "The share of WELLS drawn at random and held aside BEFORE any "
            "strategy chooses anything. No strategy may select from them, "
            "and every number any strategy reports is measured there — "
            "because a strategy scored on the cells it chose is optimistic "
            "by construction.")
        form.addRow("Random hold-out", self._holdout)

        self._leakage = QComboBox()
        for value, text in LEAKAGE_CHOICES:
            self._leakage.addItem(text, value)
        self._leakage.setToolTip(
            "The score already encodes the phenotype, so a model trained on "
            "high-score against random can succeed by relearning the score "
            "and learning no morphology at all. Reporting both fits is what "
            "answers that: read the one without the score's own inputs.")
        form.addRow("The score's own inputs", self._leakage)

        self._model = QComboBox()
        for value, text in MODEL_CHOICES:
            self._model.addItem(text, value)
        form.addRow("Model", self._model)

        self._label_column = QLineEdit()
        self._label_column.setPlaceholderText(
            "leave empty to cut the score instead")
        self._label_column.setToolTip(
            "A column of annotations somebody wrote. With one, the hold-out "
            "is scored against those annotations and the score's trap does "
            "not reach it; without one, the reference label is a cut on the "
            "score itself and every report says so.")
        form.addRow("Annotation column", self._label_column)

        self._seed = QSpinBox()
        self._seed.setRange(0, 1_000_000)
        self._seed.setToolTip(
            "One seed for the hold-out, the random draws and the model. The "
            "same seed chooses the same cells.")
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
            "How 'least sure' is measured. On two classes margin and "
            "least-confidence give the same order; entropy uses the whole "
            "distribution and differs only from three classes up.")
        self._add_row(strategy_form, "measure", "Uncertainty measure",
                      self._measure)

        self._clusters = QSpinBox()
        self._clusters.setRange(0, 100_000)
        self._clusters.setToolTip(
            "How many clusters to spread the budget over. 0 means one per "
            "cell in the budget, which is what makes each queued cell a "
            "representative rather than a sample.")
        self._add_row(strategy_form, "n_clusters", "Clusters", self._clusters)

        self._bins = QSpinBox()
        self._bins.setRange(2, 100)
        self._bins.setValue(10)
        self._bins.setToolTip(
            "How many equal-count strata the score range is cut into. The "
            "budget is divided evenly between them.")
        self._add_row(strategy_form, "n_bins", "Score strata", self._bins)

        self._confidence = QDoubleSpinBox()
        self._confidence.setRange(0.50, 0.999)
        self._confidence.setSingleStep(0.01)
        self._confidence.setDecimals(3)
        self._confidence.setValue(0.900)
        self._confidence.setToolTip(
            "How sure a self-training round has to be before it accepts its "
            "own prediction as a label. Lower is faster and drifts sooner.")
        self._add_row(strategy_form, "confidence", "Confidence",
                      self._confidence)

        self._rounds = QSpinBox()
        self._rounds.setRange(1, 50)
        self._rounds.setValue(5)
        self._rounds.setToolTip(
            "The most self-training rounds to run. It stops earlier when the "
            "audit set stops improving, which is the point of having one.")
        self._add_row(strategy_form, "rounds", "Rounds", self._rounds)

        self._neighbours = QSpinBox()
        self._neighbours.setRange(1, 100)
        self._neighbours.setValue(5)
        self._neighbours.setToolTip(
            "How many neighbours of each seed are considered. The distance "
            "cut decides how many of them actually take the label.")
        self._add_row(strategy_form, "neighbours", "Neighbours per seed",
                      self._neighbours)

        self._distance = QDoubleSpinBox()
        self._distance.setRange(0.0, 1.0)
        self._distance.setSingleStep(0.05)
        self._distance.setDecimals(2)
        self._distance.setValue(0.10)
        self._distance.setToolTip(
            "The propagation radius, as a quantile of the neighbour "
            "distances this screen produced. The radius it resolves to is "
            "printed in the report — too generous a cut manufactures "
            "agreement.")
        self._add_row(strategy_form, "distance_quantile", "Distance cut",
                      self._distance)

        self._positive_wells = QLineEdit()
        self._positive_wells.setPlaceholderText("e.g. r1_c1, r1_c2")
        self._positive_wells.setToolTip(
            "Wells whose cells are known positives. Their labels carry the "
            "experiment's own definition of the phenotype rather than an "
            "annotator's eye.")
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
            "The strategy's own account of what it chose, what it fitted and "
            "what it is allowed to claim.")
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

    def _add_row(self, form: QFormLayout, key: str, title: str,
                 widget: QWidget) -> None:
        """Add one strategy setting, remembering its label and its help."""
        label = QLabel(title)
        form.addRow(label, widget)
        self._rows[key] = (label, widget)
        self._row_help[key] = widget.toolTip()

    def strategy_key(self) -> str:
        """The strategy the menu is showing."""
        return str(self._menu.currentData() or "")

    def set_strategy(self, key: str) -> bool:
        """Show ``key`` on the menu. False when it is not on it."""
        index = self._menu.findData(str(key))
        if index < 0:
            return False
        self._menu.setCurrentIndex(index)
        return True

    def about_text(self) -> str:
        """What the chosen strategy is for, and what it costs."""
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
                "\n\nTHE CONTROL FOR THAT IS THE “The score's own inputs” "
                "setting above: leave it on “report both” and read the fit "
                "WITHOUT them. The share of the fit that survives their "
                "removal is printed with the result.")
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
                      "greyed rather than hidden so it is still where you "
                      "left it when you come back to a strategy that does.")
            # ON THE FIELD AS WELL AS THE NAME, and marked so the hover-help
            # pass does not move a disabled control's reason off it.
            widget.setProperty(DISABLED_REASON_TOOLTIP, True)
            widget.setToolTip(reason)
            label.setToolTip(reason)
        self._refresh_controls()

    # ------------------------------------------------------------- the run

    def refresh(self) -> None:
        """Take the wells and the score from whatever is on screen now."""
        if not self._wells.text().strip():
            wells = self._wells_provider() if self._wells_provider else ()
            if wells:
                self._wells.setText(", ".join(str(w) for w in wells))
        self._refresh_controls()

    def reason(self) -> str:
        """Why the run button cannot act, or ``''``."""
        if self._running:
            return "A strategy is already running."
        frame = self._objects()
        if frame is None or not len(frame):
            return ("There are no cells to choose from yet — press “Show the "
                    "cells” first. A strategy annotates the objects the "
                    "montage loaded, and none are loaded.")
        from ... import regression_annotation as strategies

        try:
            entry = strategies.strategy(self.strategy_key())
        except strategies.AnnotationStrategyError as refusal:
            return str(refusal)
        if not entry.implemented:
            return (f"{entry.title} is on the menu and is not implemented "
                    "yet, so it would select nothing.")
        return ""

    def _refresh_controls(self) -> None:
        """A control that cannot act is greyed AND says why."""
        reason = self.reason()
        self._run_button.setEnabled(not reason)
        self._run_button.setToolTip(reason or (
            "Choose the cells, fit on them, apply the fit to the rest of the "
            "screen and report against the random hold-out."))
        savable = self._result is not None
        self._save_button.setEnabled(savable)
        self._save_button.setToolTip(
            "Write the chosen cells, the hold-out, the predictions for every "
            "other cell and the report, as four files." if savable else
            "There is nothing to save yet — run a strategy first.")
        # A TAB THAT CANNOT BE FILLED SAYS WHY. The panel is present from the
        # moment the Cells tab is built, so before a montage has loaded the
        # reason is the only thing on it worth reading -- and it must not
        # overwrite the report of a run that has already happened.
        if reason and self._result is None and not self._running:
            self._status.setText(reason)

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
        """The :class:`AnnotationRequest` the controls describe, or None."""
        frame = self._objects()
        if frame is None or not len(frame):
            return None
        from ... import regression_annotation as strategies

        score = "pred"
        if self._score_provider is not None:
            try:
                score = str(self._score_provider() or "pred")
            except Exception:
                LOG.debug("could not read the score column", exc_info=True)
        return strategies.AnnotationRequest(
            frame=frame,
            score_column=score,
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
        """Run the chosen strategy off the GUI thread. False when it cannot.

        The strategy itself runs in the worker and returns either a result
        or the refusal it raised; nothing that touches a widget happens
        there. :meth:`_on_done` is a bound method of this GUI-thread
        object, which is how the answer gets back on the right thread.
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
            f"Running {key}… the hold-out wells are drawn first, and nothing "
            "is chosen from them.")

        def work():
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
            f"{roles.get('holdout', 0):,} held aside. Every number below is "
            "measured on the hold-out.")
        self._refresh_controls()
        self.finished.emit(self.strategy_key())

    def _on_job_failed(self, message: str) -> None:
        """The runner itself raised. Say so rather than staying blank."""
        self._running = False
        self._status.setText(f"The strategy failed: {message}")
        self._refresh_controls()

    # ------------------------------------------------------------ the files

    def result(self):
        """The last :class:`AnnotationResult`, or None."""
        return self._result

    def report_text(self) -> str:
        """What the report pane is showing."""
        return self._report.toPlainText()

    def save(self, folder: Optional[str] = None) -> Dict[str, str]:
        """Write the selection, hold-out, predictions and report.

        :param folder: where to write. Asked for when omitted.
        :returns: ``{what: path}``, empty when nothing was written.
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
            f"Wrote {len(written)} file(s) into {folder_name}: the cells "
            "chosen, the hold-out they are measured against, the prediction "
            "for every other cell, and the report.")
        return written

    def shutdown(self) -> None:
        """Stop the worker before the widget goes away."""
        try:
            self._jobs.shutdown()
        except Exception:
            LOG.debug("the strategy runner would not shut down",
                      exc_info=True)

    def closeEvent(self, event):             # noqa: N802 - Qt's spelling
        """Close the runner with the panel."""
        self.shutdown()
        super().closeEvent(event)
