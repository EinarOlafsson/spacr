"""Interactive measurement comparisons for selected cell groups.

The panel delegates grouping and data preparation to
:mod:`spacr.gene_measurement_compare` and statistical-test selection to
:mod:`spacr.sp_stats`. It can be embedded in the Cells tab or opened in a
standalone window without changing the comparison semantics.
"""
import logging
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QPlainTextEdit, QPushButton, QVBoxLayout,
                               QWidget)

from .toggle import Toggle

from ...figures.spread import SPREAD_CHOICES, SPREAD_NONE, SPREAD_SEM
from ...gene_measurement_compare import (CONTRASTS, LEVELS, OPERATORS, PLOTS,
                                         build, control_wells,
                                         join_measurements,
                                         measurements_are_joined, plot, save,
                                         wells_of,
                                         with_statistics)

LOG = logging.getLogger("spacr.qt.measurement_compare")


class _WellChoice(QDialog):
    """Checklist for including or excluding annotated wells.

    A flat checklist supports annotations spanning multiple plates while
    preserving the canonical well labels used by the plate-map picker.
    """

    def __init__(self, offered, chosen=None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Which wells to include")
        self._boxes = []
        outer = QVBoxLayout(self)
        outer.addWidget(QLabel(
            "A well left out is dropped from BOTH sides: it is not moved "
            "into the comparison group."))
        for well in offered:
            # `Toggle`, not a bare check box: it subclasses one, so nothing
            # about the behaviour changes, and it is what every other boolean
            # in spaCR looks like. A test greps the Qt package for the bare
            # constructor precisely to stop the two drifting -- which is why
            # this comment does not write it out.
            box = Toggle(str(well), self)
            # EVERYTHING ON BY DEFAULT. `None` means "all of them", and a
            # panel that opened with nothing ticked would read as "nothing
            # is being compared", which is not what was happening.
            box.setChecked(chosen is None or str(well) in chosen)
            outer.addWidget(box)
            self._boxes.append(box)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok
                                   | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def chosen(self) -> set:
        return {b.text() for b in self._boxes if b.isChecked()}


#: Map Measurement Compare plot names to :mod:`spacr.graph_types` names.
#:
#: ``jitter_box`` uses the same data-shape compatibility rules as
#: ``bar_jitter`` while retaining the panel's established display choice.
_SPEC_KINDS = {
    "jitter_box": "bar_jitter",
    "box": "box",
    "jitter": "jitter",
    "violin": "violin",
    "bar": "bar",
}


class MeasurementComparePanel(QWidget):
    """Compare measurements between selected cell groups and a reference.

    Parameters
    ----------
    objects : pandas.DataFrame
        Object rows available to the montage and reference contrasts.
    groups : dict of str to sequence
        Selected group names mapped to object-index values.
    parent : QWidget, optional
        Parent widget.
    settings : dict, optional
        Run settings saved with exported comparison results.
    databases : sequence of path-like, optional
        Measurement databases available for widening the object table.
    counts : pandas.DataFrame, optional
        Per-well counts used to resolve control wells.
    """

    #: What the join control says when no join is running, and while one is.
    #: One button carries both, because a join that has started is a job to
    #: stop rather than a job to start again.
    JOIN_LABEL = "Join the measurement tables"
    CANCEL_LABEL = "Cancel the join"

    def __init__(self, objects, groups: Dict[str, Any],
                 parent: Optional[QWidget] = None,
                 settings: Optional[Dict[str, Any]] = None,
                 databases: Optional[Any] = None,
                 counts: Optional[Any] = None):
        super().__init__(parent)
        self._objects = objects
        self._groups = dict(groups or {})
        self._settings = dict(settings or {})
        self._databases = tuple(databases or ())
        self._counts = counts
        # THE JOIN RUNS OFF THE GUI THREAD. It reads every object table out
        # of every attached database and joins them onto the crop rows:
        # measured at 3.2 s for one plate's 553 objects, so a four-plate
        # screen of 60,000 is minutes with the window frozen solid --
        # reported as "pressing join the measurements table in the cell tab
        # in the regression module makes spacr unresponsive".
        from ..job_runner import JobRunner

        self._jobs = JobRunner(self, app_key="join measurements")
        self._joining = False
        self._comparison = None
        self._canvas = None
        # THE WELLS THE USER LEFT IN. `None` means "all of them", which is
        # not the same as the full list: a well that appears after a re-run
        # should be included, and a stored full list would silently exclude
        # it.
        self._chosen_wells: Optional[set] = None

        #: The volcano's guide selection, and the well set derived from it.
        #: `None` for the wells means "not narrowed yet", which is different
        #: from "narrowed to nothing" -- and the difference is what lets a
        #: new guide selection re-derive rather than stay empty.
        self._selected_guides: list = []
        self._selected_wells = None

        layout = QVBoxLayout(self)
        row = QHBoxLayout()

        #: The headings, by the field each one governs, so a test can find
        #: them without matching on the text they display.
        self.headings = {}

        row.addWidget(self._heading("measurement"))
        self.measurement = QComboBox()
        # OFFERED FROM THE DATA, never typed: the same rule every other
        # chooser in spaCR follows, and the reason `object_array` stopped
        # being a text box.
        for name in self._numeric_columns():
            self.measurement.addItem(str(name), str(name))
        self.measurement.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.measurement, 1)

        # THE SECOND MEASUREMENT AND THE OPERATOR (179 B). "one mes minus,
        # plus, multiplied by or devided by another mes" -- and the combined
        # column is named for the expression, so the table, the legend and
        # the settings file all say the same thing.
        self.operator = QComboBox()
        for value, label in OPERATORS:
            self.operator.addItem(label, value)
        self.operator.currentIndexChanged.connect(self._on_operator)
        row.addWidget(self.operator)

        self.second = QComboBox()
        self.second.setEnabled(False)
        self.second.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.second, 1)
        self._offer_second()

        row.addWidget(self._heading("level"))
        self.level = QComboBox()
        for value, why in LEVELS:
            self.level.addItem(value, value)
            self.level.setItemData(self.level.count() - 1, why, Qt.ToolTipRole)
        self.level.setCurrentIndex(1)          # well: the unit the screen randomises
        self.level.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.level)

        row.addWidget(self._heading("plot"))
        self.kind = QComboBox()
        for value, label in PLOTS:
            self.kind.addItem(label, value)
        self.kind.currentIndexChanged.connect(self.refresh)
        self.kind.currentIndexChanged.connect(self._offer_the_spread)
        row.addWidget(self.kind)

        # WHAT THE WHISKER MEANS, WHEN THERE IS ONE. "for the cell table
        # graphs if bar is chosen the user should be able to choose SD, Var,
        # or SEM error bars."
        #
        # THEY ARE NOT INTERCHANGEABLE: SD describes the cells, SEM the
        # confidence in their mean, and at n=3000 they differ by a factor of
        # fifty-five -- a reader who assumes the wrong one reads a real
        # effect as noise or noise as a real effect. So the caption under the
        # plot names the one that was drawn, not only this box.
        #
        # ABSENT, NOT INERT, for a graph type that has no error bar (106): a
        # box already draws its quartiles and a jitter draws every point, and
        # a control that cannot change either is a promise that was kept
        # once.
        self.spread = QComboBox()
        for value, label in SPREAD_CHOICES:
            self.spread.addItem(label, value)
        self.spread.setCurrentIndex(max(0, self.spread.findData(SPREAD_SEM)))
        self.spread.setToolTip(
            "What the bar's whisker MEANS. SD describes the spread of the "
            "observations themselves; SEM describes how well their mean is "
            "pinned down and is SD over the square root of n -- at n=3000 "
            "the two differ fifty-five-fold; the variance is SD squared, so "
            "its whisker is in squared units. The caption under the plot "
            "names whichever was drawn.")
        self.spread.currentIndexChanged.connect(self._draw_and_report)
        row.addWidget(self.spread)

        # B2, asked for 2026-08-20: "it should be possible to show only one
        # class". A FILTER ON THE DRAW, not on the build -- the statistics
        # below still describe the whole comparison, because a test computed
        # on one of two groups is not a comparison at all and a panel that
        # quietly re-ran it on the visible half would be reporting a
        # different question than the one on screen.
        row.addWidget(self._heading("show"))
        # THE THREE POPULATIONS (instruction 205). Beside the class filter
        # rather than merged into it: "which objects are on the plot" and
        # "which of the classes on it to draw" are different questions, and
        # one box answering both would have a list whose entries mean two
        # different things.
        from ...well_scope import SCOPES

        self.scope = QComboBox()
        for value, caption in SCOPES:
            self.scope.addItem(caption, value)
        self.scope.setToolTip(
            "WHICH OBJECTS are plotted. 'gRNAs' draws only the objects "
            "annotated with the guides chosen on the volcano. 'gRNAs + "
            "other datapoints in selected wells' adds their WELL-MATES, "
            "drawn distinguishably — that is the comparison that matters, "
            "because a guide's cells against every cell in the screen "
            "compares two different experiments, while a guide's cells "
            "against their own well-mates compares two populations that "
            "shared a plate, a day, a stain and an imaging session.")
        self.scope.currentIndexChanged.connect(self._on_scope)
        row.addWidget(self.scope)

        self.only = QComboBox()
        self.only.setToolTip(
            "Draw one class on its own. The statistics below are always for "
            "the whole comparison — a test needs both sides.")
        self.only.currentIndexChanged.connect(self._draw_and_report)
        row.addWidget(self.only)

        self._offer_the_spread()

        self.save_button = QPushButton("Save…")
        self.save_button.setToolTip(
            "Write the figure, the plotted data, the statistics, the "
            "settings and the cell images into ONE folder.")
        self.save_button.clicked.connect(self.save_everything)
        row.addWidget(self.save_button)
        layout.addLayout(row)

        # ------------------------------------------------------------ 187 B
        # THE CONTRAST IS A SEPARATE ROW because it is a separate decision.
        # The row above chooses WHAT is measured; this one chooses WHAT IT IS
        # HELD AGAINST, and the same cells under three contrasts give three
        # different p-values.
        second_row = QHBoxLayout()
        second_row.addWidget(self._heading("compare"))
        self.contrast = QComboBox()
        for value, label, why in CONTRASTS:
            self.contrast.addItem(label, value)
            self.contrast.setItemData(self.contrast.count() - 1, why,
                                      Qt.ToolTipRole)
        self.contrast.currentIndexChanged.connect(self._on_contrast)
        second_row.addWidget(self.contrast, 1)

        # THE CONTROLS, resolved through `spacr.control_names` (184) -- so a
        # gene, a guide, a prefixed name and a bare one all work, and this
        # panel does not grow a fifth opinion about what a control is.
        self.controls = QLineEdit()
        self.controls.setPlaceholderText("control gene or guide, comma "
                                          "separated")
        self.controls.setToolTip(
            "The control wells to compare against. A gene name takes every "
            "one of its guides; a guide name takes just that guide.")
        self.controls.editingFinished.connect(self.refresh)
        self.controls.setEnabled(False)
        second_row.addWidget(self.controls, 1)

        # AND WHICH OF THE GENE'S WELLS COUNT. "i whould be able to choose
        # which wells to include from the gene annotation" -- a well that
        # failed for an unrelated reason should not have to poison the
        # contrast.
        #
        # A CHECKLIST RATHER THAN 185's PLATE MAP: an annotation's wells span
        # plates, and a plate map can only show one plate at a time. The
        # NAMES are the same either way, which is the part that had to agree.
        self.wells_button = QPushButton("wells…")
        self.wells_button.setToolTip(
            "Choose which of the annotation's wells to include. A well left "
            "out is dropped from BOTH sides of the comparison.")
        self.wells_button.clicked.connect(self.choose_wells)
        second_row.addWidget(self.wells_button)
        layout.addLayout(second_row)

        # ------------------------------------------------------------ 187 A
        # THE JOIN IS OFFERED, NOT SILENTLY SKIPPED. `png_list` holds the
        # crop path and the classification score; every morphological
        # measurement is in the object tables beside it. Offering a short
        # list of measurements with no reason for its shortness is what this
        # is against.
        self._join_row = QHBoxLayout()
        self.join_note = QLabel("")
        self.join_note.setObjectName("Muted")
        self.join_note.setWordWrap(True)
        self._join_row.addWidget(self.join_note, 1)
        self.join_button = QPushButton(self.JOIN_LABEL)
        self.join_button.setToolTip(
            "Read the cell, nucleus, pathogen and cytoplasm tables out of "
            "the attached databases and attach them to these cells, so every "
            "measurement in the screen can be compared.")
        self.join_button.clicked.connect(self._on_join_button)
        self._join_row.addWidget(self.join_button)
        layout.addLayout(self._join_row)

        # TWO MORE BOXES (instruction 213 A). `png_list` and the dependent
        # variable are already the substance of the analysis and neither
        # could be reached from this control -- which offered the four object
        # tables and no reason for stopping there.
        from PySide6.QtWidgets import QCheckBox

        extras = QHBoxLayout()
        self.join_png_list = QCheckBox("png_list")
        self.join_png_list.setChecked(True)
        self.join_png_list.setToolTip(
            "Join the crop table: its path and its classification score. "
            "Every morphological measurement is in the object tables beside "
            "it, and the score is not in any of them.")
        self.join_png_list.toggled.connect(self._on_join_choice)
        extras.addWidget(self.join_png_list)
        self.join_dependent = QCheckBox("dependent variable")
        self.join_dependent.toggled.connect(self._on_join_choice)
        self.join_dependent.setToolTip(
            "Join the table carrying the value the screen is regressing on. "
            "It joins by plateID, rowID, columnID, fieldID and objectID when "
            "they are all there; when one is missing it falls back to "
            "splitting the crop path, then to translating the well into a "
            "row and a column. Which route was used is reported — a join "
            "that silently degraded is one nobody can check.")
        extras.addWidget(self.join_dependent)
        extras.addStretch(1)
        layout.addLayout(extras)

        self._figure_holder = QVBoxLayout()
        layout.addLayout(self._figure_holder, 1)

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(140)
        layout.addWidget(self.report)

        self.resize(900, 720)
        self.refresh()

    # ------------------------------------------------------------- the data

    def _numeric_columns(self) -> list:
        """Every measurement on these objects, identifiers left out."""
        try:
            import pandas as pd

            from ...gene_measurement_sweep import is_measurement

            return [c for c in self._objects.columns
                    if pd.api.types.is_numeric_dtype(self._objects[c])
                    and is_measurement(c)]
        except Exception:                                    # noqa: BLE001
            return []

    def _offer_second(self):
        """Fill the second chooser from the same columns as the first."""
        self.second.blockSignals(True)
        self.second.clear()
        for name in self._numeric_columns():
            self.second.addItem(str(name), str(name))
        self.second.blockSignals(False)

    def _on_operator(self, *_args):
        """A second measurement is only meaningful with an operator."""
        self.second.setEnabled(bool(self.operator.currentData()))
        self.refresh()

    def set_data(self, objects, groups: Dict[str, Any],
                 settings: Optional[Dict[str, Any]] = None):
        """Point the panel at a new montage. The Graph tab calls this rather
        than being rebuilt, so a user's chosen measurement and level survive
        a re-run."""
        self._objects = objects
        self._groups = dict(groups or {})
        if settings is not None:
            self._settings = dict(settings)
        remembered = self.measurement.currentData()
        self.measurement.blockSignals(True)
        self.measurement.clear()
        for name in self._numeric_columns():
            self.measurement.addItem(str(name), str(name))
        index = self.measurement.findData(remembered)
        if index >= 0:
            self.measurement.setCurrentIndex(index)
        self.measurement.blockSignals(False)
        self._offer_second()
        return self.refresh()

    def comparison(self):
        return self._comparison

    # ------------------------------------------------------- 187 B: contrast

    def _on_contrast(self, *_args):
        """The controls field only means anything for one of the three."""
        self.controls.setEnabled(
            str(self.contrast.currentData() or "") == "against_controls")
        self.refresh()

    def _typed_controls(self) -> list:
        """The control names as typed, split the way every other field is."""
        text = str(self.controls.text() or "")
        return [part.strip() for part in text.split(",") if part.strip()]

    def _control_wells(self) -> tuple:
        """Which wells the typed controls occupy, out of the count data."""
        typed = self._typed_controls()
        if not typed or self._counts is None:
            return ()
        try:
            return control_wells(self._counts, typed)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not resolve the control wells", exc_info=True)
            return ()

    def wells_on_offer(self) -> tuple:
        """Return annotated wells in first-occurrence order.

        Returns
        -------
        tuple of str
            Unique wells represented by the current selected groups.
        """
        found: list = []
        for wells in wells_of(self._objects, self._groups).values():
            found.extend(str(w) for w in wells)
        return tuple(dict.fromkeys(found))

    def chosen_wells(self) -> Optional[list]:
        """Return the included wells that remain available.

        Returns
        -------
        list of str or None
            Selected wells intersected with the current inventory, or ``None``
            when all current and future wells are included.
        """
        if self._chosen_wells is None:
            return None
        # INTERSECTED WITH WHAT IS THERE NOW, so a choice made before a
        # re-run cannot name a well the new montage does not have.
        return [w for w in self.wells_on_offer() if w in self._chosen_wells]

    def choose_wells(self, *_args) -> bool:
        """Open the well checklist and apply a changed selection.

        Returns
        -------
        bool
            ``True`` when the accepted selection differs from the previous
            one; ``False`` when unavailable, cancelled, or unchanged.
        """
        offered = self.wells_on_offer()
        if not offered:
            self.report.setPlainText(
                "These object rows do not say which well they came from, so "
                "there are no wells to choose between.")
            return False
        before = self.chosen_wells()
        dialog = _WellChoice(offered, self._chosen_wells, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return False
        self._chosen_wells = dialog.chosen()
        if self.chosen_wells() == before:
            return False
        self.refresh()
        return True

    # ---------------------------------------------------------- 187 A: join

    def _say_about_the_join(self) -> None:
        """Say whether the measurement list is the short one, and why."""
        joined = measurements_are_joined(self._objects)
        offered = self.measurement.count()
        if joined:
            self.join_note.setText(
                f"{offered} measurement(s) offered, from the joined "
                f"measurement tables.")
            self.join_button.setVisible(False)
            return
        self.join_button.setVisible(bool(self._databases))
        if self._databases:
            self.join_note.setText(
                f"Only {offered} measurement(s): these are the columns on "
                f"the crop table. Every morphological measurement -- cell, "
                f"nucleus, pathogen, cytoplasm -- is in the object tables "
                f"and needs the join.")
        else:
            self.join_note.setText(
                f"Only {offered} measurement(s): these are the columns on "
                f"the crop table, and no measurements database is attached "
                f"to join the object tables from.")

    def join_the_tables(self, *_args) -> str:
        """Join attached measurement tables into the panel's object rows.

        Returns
        -------
        str
            Empty after a clean join, or a user-facing explanation when a
            database or object row could not be joined.
        """
        if not self._databases:
            return "no measurements database is attached"
        if self._joining:
            return ""
        # READ ON THE GUI THREAD, because a worker may not touch a widget.
        png_list = bool(self.join_png_list.isChecked())
        objects, databases = self._objects, self._databases
        self._joining = True
        # THE BUTTON BECOMES THE CANCEL. A join of a four-plate screen is
        # minutes of reading, and a run that can only be waited out is the
        # freeze this moved off the GUI thread to avoid, one step removed.
        self.join_button.setEnabled(True)
        self.join_button.setText(self.CANCEL_LABEL)

        def work():
            """Off the GUI thread. Returns, never raises: a failed join is
            a message in the panel, not a traceback in the console."""
            try:
                wide, trouble = join_measurements(objects, databases,
                                                  png_list=png_list)
                return {"wide": wide, "trouble": trouble}
            except Exception as exc:                         # noqa: BLE001
                LOG.debug("could not join the measurement tables",
                          exc_info=True)
                return {"error": str(exc)}

        started = self._jobs.submit(work, self._finish_join)
        if not started:                      # pragma: no cover - JobRunner
            self._joining = False            # always returns True today
            self._reset_the_join_button()
        return ""

    def _on_join_button(self, *_args) -> str:
        """Start the join, or cancel the one already running."""
        if self._joining:
            self.cancel_the_join()
            return ""
        return self.join_the_tables()

    def cancel_the_join(self) -> bool:
        """Abandon a running join and put the button back.

        The worker is asked to stop and its result is dropped on arrival;
        it is not joined, because waiting for it on the GUI thread is the
        freeze the worker exists to avoid.

        :returns: True when a join was in flight to cancel.
        """
        if not self._joining:
            return False
        self._jobs.cancel()
        self._joining = False
        self._reset_the_join_button()
        self.join_note.setText("Join cancelled.")
        return True

    def _reset_the_join_button(self) -> None:
        """Back to the label and the enabled state of an idle join."""
        self.join_button.setEnabled(True)
        self.join_button.setText(self.JOIN_LABEL)

    def closeEvent(self, event):                # noqa: N802 - Qt naming
        """Stop a running join before the widget carrying it goes away.

        Qt aborts the process when a running QThread is destroyed, so the
        panel cannot simply be closed out from under a join of a real
        screen.
        """
        try:
            self._jobs.shutdown()
        except Exception:                                    # noqa: BLE001
            LOG.debug("the join runner would not shut down", exc_info=True)
        self._joining = False
        super().closeEvent(event)

    def _finish_join(self, outcome: Dict[str, Any]) -> str:
        """Put the joined frame into the panel. Always on the GUI thread."""
        self._joining = False
        self._reset_the_join_button()
        if not isinstance(outcome, dict) or "error" in (outcome or {}):
            why = (outcome or {}).get("error", "the join returned nothing")
            self.join_note.setText(f"Could not join: {why}")
            return str(why)
        wide, trouble = outcome["wide"], outcome["trouble"]
        self._joined_once = True
        wide, dependent_note = self._join_the_dependent_variable(wide)
        if dependent_note:
            trouble = f"{trouble} {dependent_note}".strip()
        # THE GROUPS SURVIVE because `join_measurements` keeps the index, and
        # `set_data` re-reads them from the same values.
        self.set_data(wide, self._groups)
        if trouble:
            self.join_note.setText(
                f"{self.join_note.text()} {trouble}".strip())
        return trouble

    def _on_scope(self, *_args) -> None:
        """The population changed: re-derive the wells and redraw."""
        self._selected_wells = None      # derived again from the guides
        self.refresh()

    def set_selected_guides(self, guides) -> None:
        """Set selected guides and derive their wells for the current scope."""
        self._selected_guides = [str(g) for g in (guides or ())]
        self._selected_wells = None
        self.refresh()

    def selected_wells(self) -> list:
        """Return explicit wells or derive them from selected guides."""
        if self._selected_wells is not None:
            return list(self._selected_wells)
        from ...well_scope import wells_of

        return wells_of(self._objects, getattr(self, "_selected_guides", []))

    def set_selected_wells(self, wells) -> None:
        """Set an explicit well subset; ``None`` restores guide-based derivation."""
        self._selected_wells = (None if wells is None
                                else [str(w) for w in wells])
        self.refresh()

    def scoped_objects(self):
        """Return objects in the current display scope and its selection report."""
        from ...well_scope import select

        return select(self._objects,
                      scope=str(self.scope.currentData() or "guides"),
                      guides=getattr(self, "_selected_guides", []),
                      wells=self._selected_wells)

    def _heading(self, field: str) -> QLabel:
        """Create a control heading with persistent field-specific help.

        Help is keyed by the underlying field so it remains valid when a
        displayed heading changes. The shared tooltip filter keeps the content
        accessible while the pointer moves from the heading to the tooltip.
        """
        from ...gene_measurement_compare import HEADING_HELP

        label = QLabel(str(field))
        label.setToolTip(HEADING_HELP.get(str(field), ""))
        label.setToolTipDuration(-1)
        label.setCursor(Qt.WhatsThisCursor)
        try:
            from ..screens.settings_model import _ApiTooltipFilter

            # HELD ON `self`, not made and dropped. Qt keeps a bare pointer
            # to an event filter, so one that only this call referenced is
            # collected as soon as the call returns -- after which the
            # heading stops responding and nothing says why.
            if getattr(self, "_heading_filter", None) is None:
                self._heading_filter = _ApiTooltipFilter(self)
            label.setProperty("apiTooltipHtml", label.toolTip())
            label.setProperty("apiTooltipDisplayRole", "tooltip")
            label.removeEventFilter(self._heading_filter)
            label.installEventFilter(self._heading_filter)
        except Exception:                                    # noqa: BLE001
            # The plain Qt tooltip still shows; only the stay-open behaviour
            # is lost, which is a worse tooltip rather than none.
            LOG.debug("could not install the hover tooltip", exc_info=True)
        self.headings[str(field)] = label
        return label

    def _on_join_choice(self, *_args) -> None:
        """A join box moved: re-join, if a join has already been made.

        WITHOUT THIS THE BOX TAKES EFFECT ON THE NEXT PRESS of a button the
        user has already pressed, which reads as a box that does nothing --
        the same complaint by a slower route.
        """
        if getattr(self, "_joined_once", False):
            self.join_the_tables()

    def _join_the_dependent_variable(self, wide):
        """Attach the dependent-variable table when the option is enabled.

        The returned status identifies any fallback route used, which makes
        missing direct identifiers visible to the user.
        """
        if not getattr(self, "join_dependent", None) or \
                not self.join_dependent.isChecked():
            return wide, ""
        frame = getattr(self, "_dependent_frame", None)
        if frame is None or not len(frame):
            return wide, ("the dependent variable was asked for and no table "
                          "carrying it is attached")
        try:
            from ...dependent_join import describe, join

            out, report = join(wide, frame)
        except Exception as exc:                             # noqa: BLE001
            LOG.debug("could not join the dependent variable", exc_info=True)
            # REPORTED, NOT SWALLOWED. A route that matches nothing is a
            # failure rather than an empty answer.
            return wide, f"the dependent variable did not join: {exc}"
        return out, describe(report)

    def set_dependent_frame(self, frame) -> None:
        """Set the dependent-variable table available to the join action.

        Parameters
        ----------
        frame : pandas.DataFrame
            Table containing dependent variables and object identifiers or
            parseable image paths.
        """
        self._dependent_frame = frame

    # ------------------------------------------------------------ the build

    def refresh(self, *_args):
        """Rebuild, retest and redraw. Returns the comparison, or ``None``."""
        measurement = str(self.measurement.currentData() or "")
        if not measurement:
            self.report.setPlainText(
                "These objects carry no measurement column to compare.")
            self._say_about_the_join()
            return None
        level = str(self.level.currentData() or "well")
        operator = str(self.operator.currentData() or "")
        second = str(self.second.currentData() or "") if operator else ""
        contrast = str(self.contrast.currentData() or "")
        self._comparison = with_statistics(
            build(self._objects, measurement, groups=self._groups,
                  level=level, operator=operator, second=second,
                  contrast=contrast, wells=self.chosen_wells(),
                  controls=self._control_wells()))
        self._offer_classes()
        self._say_about_the_join()
        self._draw()
        self._report()
        return self._comparison

    def _draw_and_report(self, *_args):
        """Redraw for a changed VIEW, without rebuilding the comparison."""
        if self._comparison is None:
            return
        self._draw()
        self._report()

    def _classes(self) -> list:
        """The group labels in the comparison, in a stable order."""
        if self._comparison is None or not len(self._comparison.frame):
            return []
        return [str(g) for g in
                self._comparison.frame["group"].astype(str).unique()]

    def _has_an_error_bar(self) -> bool:
        """Does the chosen graph type draw a whisker at all?

        Only the bar summarises to a single height, so only the bar has
        anything for a spread to describe. A box draws its own quartiles and
        a jitter draws every observation.
        """
        return _SPEC_KINDS.get(str(self.kind.currentData() or ""), "") in (
            "bar", "bar_jitter")

    def _offer_the_spread(self, *_args) -> None:
        """Show the whisker control only where there is a whisker."""
        self.spread.setVisible(self._has_an_error_bar())

    def _offer_classes(self) -> None:
        """Refill the "show" box, keeping the choice when it still exists."""
        names = self._classes()
        before = str(self.only.currentData() or "")
        self.only.blockSignals(True)
        self.only.clear()
        self.only.addItem("every class", "")
        for name in names:
            self.only.addItem(name, name)
        index = self.only.findData(before)
        self.only.setCurrentIndex(index if index >= 0 else 0)
        self.only.blockSignals(False)
        self.only.setEnabled(len(names) > 1)

    def nothing_to_compare_against(self) -> str:
        """Return why the selected cells have no comparison group.

        An empty string means at least two classes are available. When only
        picked cells were loaded, the message identifies ``show all in well``
        as the setting that adds the unpicked comparison cells.
        """
        if self._comparison is None or not len(self._comparison.frame):
            return ""
        if len(self._classes()) > 1:
            return ""
        picker = str((self._settings or {}).get("cell_picking") or "rank")
        if bool((self._settings or {}).get("show_all_in_well")):
            return ("Every cell shown is in one class, so there is nothing to "
                    "compare it against.")
        return (
            f"Only one class: with '{picker}' picking and 'show all in well' "
            f"OFF, the montage holds ONLY the cells that were picked, so "
            f"there is no unpicked group to compare them with. Switch on "
            f"'show all in well' in the picture settings — every cell in the "
            f"well is then drawn and the picked ones are highlighted, which "
            f"gives this graph both sides.")

    def _draw(self):
        from .graph_builder import _canvas_class

        while self._figure_holder.count():
            item = self._figure_holder.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        showing = self._comparison
        only = str(self.only.currentData() or "")
        if only:
            from dataclasses import replace

            frame = showing.frame
            showing = replace(showing,
                              frame=frame[frame["group"].astype(str) == only])
        # PYQTGRAPH, NOT MATPLOTLIB. Asked for four times: "I WANT ALL
        # FIGURES TO BE IN pyqtgraph NOT matplotlib ... i also want to be
        # able to right click on the graph and change the graph typem to
        # whatever is possibel with the underlying data ... and i want to be
        # able to save the figure intp afolder as stats, data, ong figure
        # and pdf figure".
        #
        # THE THREE ARE ONE ASK. A plot that HOLDS ITS DATA can be redrawn
        # as any kind the data supports and can write that data and its
        # statistics beside the picture; a rendered Figure in a layout can
        # do neither, which is why the retyping and the folder save had to
        # be bolted on beside it and kept drifting out of reach.
        self._canvas = self._live_plot(showing)
        if self._canvas is None:
            return
        self._figure_holder.addWidget(self._canvas)

    def _live_plot(self, showing):
        """Return a live grouped comparison or a Matplotlib fallback.

        The fallback preserves access to the comparison if the interactive
        pyqtgraph renderer is unavailable.
        """
        from ...gene_measurement_compare import REST

        frame = getattr(showing, "frame", None)
        if frame is None or not len(frame):
            return None
        try:
            from .grouped_plot import GroupedPlot, PlotSpec

            kind = _SPEC_KINDS.get(
                str(self.kind.currentData() or ""), "bar_jitter")
            spec = PlotSpec(
                frame=frame, value="value", group="group", kind=kind,
                spread=(str(self.spread.currentData() or SPREAD_SEM)
                        if self._has_an_error_bar() else SPREAD_NONE),
                title=str(self.measurement.currentText() or ""),
                y_label=str(self.measurement.currentText() or "value"),
                unit=str(self.level.currentData() or "observation"),
                # "the rest" is the population the selected genes are being
                # compared AGAINST, so it is grey: the ink goes on the
                # claim, not on what the claim is measured against.
                background=REST)
            plot_widget = GroupedPlot(spec, parent=self)
            plot_widget.setMinimumHeight(260)
            return plot_widget
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not draw the comparison in pyqtgraph",
                      exc_info=True)
        from .graph_builder import _canvas_class

        figure = plot(showing,
                      kind=str(self.kind.currentData() or "jitter_box"))
        return _canvas_class()(figure) if figure is not None else None

    def _report(self):
        """n, the assumption checks, the test and why -- in that order.

        THE ORDER IS THE ARGUMENT. A test name above the checks that chose it
        reads as a decision already made; below them it reads as the
        consequence it is.
        """
        comparison = self._comparison
        if comparison is None:
            return
        lines = [f"{comparison.measurement} · {comparison.level} level"]
        if comparison.note:
            lines.append(comparison.note)
        # FIRST AMONG THE REASONS, because it is the one the reader can act
        # on and the one that explains an empty-looking graph.
        one_sided = self.nothing_to_compare_against()
        if one_sided:
            lines.append(one_sided)
        only = str(self.only.currentData() or "")
        if only:
            lines.append(f"Showing '{only}' only; the statistics below are "
                         f"for the whole comparison.")
        counts = comparison.counts()
        if counts:
            lines.append("n: " + ", ".join(f"{k} = {v}"
                                           for k, v in counts.items()))
        for row in comparison.statistics or []:
            for key in ("Normality", "Equal variance"):
                if row.get(key):
                    lines.append(f"{key.lower()}: {row[key]}")
            name = row.get("Test Name", "")
            p_value = row.get("p-value")
            effect = row.get("Effect Size")
            said = f"test: {name}"
            if p_value is not None:
                said += f" · p = {p_value}"
            if effect is not None:
                said += f" · effect size = {effect}"
            lines.append(said)
            if row.get("Why This Test"):
                lines.append(f"why: {row['Why This Test']}")
        if not (comparison.statistics or []):
            lines.append("no test: a comparison needs two groups with "
                         "something in them.")
        self.report.setPlainText("\n".join(lines))

    # ------------------------------------------------------------- saving

    def save_everything(self, folder: str = "") -> dict:
        """Write everything into one folder. Returns what was written."""
        if self._comparison is None:
            return {}
        chosen = str(folder or "")
        if not chosen:
            chosen = QFileDialog.getExistingDirectory(
                self, "Save the comparison into a folder")
        if not chosen:
            return {}
        try:
            written = save(self._comparison, chosen,
                           kind=str(self.kind.currentData() or "jitter_box"),
                           settings=self._settings)
        except Exception as exc:                             # noqa: BLE001
            LOG.debug("could not save the comparison", exc_info=True)
            self.report.setPlainText(
                f"{self.report.toPlainText()}\n\nCould not save: {exc}")
            return {}
        self.report.setPlainText(
            f"{self.report.toPlainText()}\n\nSaved "
            f"{len(written)} item(s) to {chosen}")
        return written


class MeasurementCompareDialog(QDialog):
    """The panel above, in a window. Kept so the button still opens one."""

    def __init__(self, objects, groups: Dict[str, Any],
                 parent: Optional[QWidget] = None,
                 settings: Optional[Dict[str, Any]] = None,
                 databases: Optional[Any] = None,
                 counts: Optional[Any] = None):
        super().__init__(parent)
        self.setWindowTitle("Compare a measurement")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.panel = MeasurementComparePanel(objects, groups, parent=self,
                                             settings=settings,
                                             databases=databases,
                                             counts=counts)
        layout.addWidget(self.panel)
        self.resize(900, 720)

    # The window forwards what the button and the tests ask of it, rather
    # than reimplementing any of it.
    def refresh(self, *args):
        return self.panel.refresh(*args)

    def comparison(self):
        return self.panel.comparison()

    def save_everything(self, folder: str = "") -> dict:
        return self.panel.save_everything(folder)

    @property
    def measurement(self):
        return self.panel.measurement

    @property
    def level(self):
        return self.panel.level

    @property
    def kind(self):
        return self.panel.kind

    @property
    def report(self):
        return self.panel.report

    @property
    def contrast(self):
        return self.panel.contrast

    @property
    def controls(self):
        return self.panel.controls

    def choose_wells(self, *args):
        return self.panel.choose_wells(*args)

    def join_the_tables(self, *args):
        return self.panel.join_the_tables(*args)

    def cancel_the_join(self, *args):
        return self.panel.cancel_the_join(*args)
