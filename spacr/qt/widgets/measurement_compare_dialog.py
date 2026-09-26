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

from ..i18n import tr
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

    def __init__(self, offered, chosen=None, parent: Optional[QWidget] = None,
                 *, title: str = "", note: str = ""):
        """Ask which wells take part in the comparison.

        :param offered: every well that could be included.
        :param chosen: those ticked to start with, or ``None`` for all of
            them.
        :param parent: parent widget.
        :param title: window title; empty keeps the well wording. The same
            checklist picks the Compare panel's gRNAs and its selected wells.
        :param note: the sentence above the list; empty keeps the well one.

        AN EXCLUDED WELL IS REMOVED FROM BOTH GROUPS. It is not moved into
        the comparison group, which is the reading this dialog exists to
        prevent -- the label says so on screen and it is repeated here
        because the parameter names alone suggest a two-way split.
        """
        super().__init__(parent)
        self.setWindowTitle(title or "Which wells to include")
        self._boxes = []
        outer = QVBoxLayout(self)
        outer.addWidget(QLabel(
            note or "An excluded well is removed from both groups; it is not "
            "moved into the comparison group."))
        for well in offered:
            box = Toggle(str(well), self)
            box.setChecked(chosen is None or str(well) in chosen)
            outer.addWidget(box)
            self._boxes.append(box)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok
                                   | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def chosen(self) -> set:
        """The wells currently ticked.

        :returns: their names, as a set -- order carries no meaning here and a
            set makes the membership tests downstream honest.
        """
        return {b.text() for b in self._boxes if b.isChecked()}


#: Map Measurement Compare plot names to :mod:`spacr.graph_types` names.
#:
#: ``jitter_box`` USED TO MAP TO ``bar_jitter``, with a note saying the
#: fitness table had no entry for a box with points over it. It has one --
#: ``box_jitter``, which is also the table's default for groups against a
#: measurement -- so the old mapping had stopped being a compatibility
#: translation and become a silent substitution: the panel's own list said
#: "jitter over box", the live plot drew a BAR with jitter over it, and the
#: Matplotlib fallback beside it drew the box the label promised.
_SPEC_KINDS = {
    "jitter_box": "box_jitter",
    "box": "box",
    "jitter": "jitter",
    "violin": "violin",
    "bar": "bar",
}

#: The inverse of :data:`_SPEC_KINDS`: which entry of
#: :data:`~spacr.gene_measurement_compare.PLOTS` draws a graph type.
#:
#: NOT EVERY GRAPH TYPE HAS ONE. This panel offers five plots and the table
#: names eight, so a user whose DEFAULT GRAPH TYPE is "Bar with jitter" has
#: asked for something this panel cannot draw. It keeps its own first
#: choice then, rather than silently drawing the nearest thing and calling
#: it what was asked for.
_PLOTS_OF_KIND = {kind: plot for plot, kind in _SPEC_KINDS.items()}


class MeasurementComparePanel(QWidget):
    """Compare measurements between selected cell groups and a reference."""

    #: What the join control says when no join is running, and while one is.
    #: One button carries both, because a join that has started is a job to
    #: stop rather than a job to start again.
    JOIN_LABEL = "Join the measurement tables"
    CANCEL_LABEL = "Cancel the join"

    def __init__(self, objects, groups: Dict[str, Any],
                 parent: Optional[QWidget] = None,
                 settings: Optional[Dict[str, Any]] = None,
                 databases: Optional[Any] = None,
                 counts: Optional[Any] = None,
                 results: Optional[Any] = None):
        """Build the comparison panel: the pickers, the plot and the statistics.

        IT OPENS ON THE TOP HITS. Decision 2026-09-25 (item 205): "the
        regression Compare panel opens with the TOP HITS pre-selected (top
        significant guides/genes and the wells that carry them; user can
        change it)". The regression's results table gives the hits
        (:func:`spacr.well_scope._top_hits`); without one, the guides of the
        montage's own groups are the selection. The population box then
        starts on "gRNAs + other datapoints in selected wells", the derived
        wells are shown under the controls, and "gRNAs…" / "selected wells…"
        change either. Nothing to select starts on "All datapoints", so a
        panel with no guides still draws what it drew before.

        The join runs off the GUI thread. It reads every object table out of
        every attached database and joins them onto the crop rows -- measured at
        3.2 s for one plate's 553 objects, so a four-plate screen of 60,000 is
        minutes with the window frozen solid, which is what "pressing join the
        measurements table makes spaCR unresponsive" was.

        The well selection is held as ``None`` for "all of them" rather than as
        the full list: a well that appears after a re-run should be included,
        and a stored full list would silently exclude it.

        :param objects: the object rows for the montage and the contrasts.
        :param groups: selected group names mapped to object-index values.
        :param parent: parent widget, or ``None``.
        :param settings: run settings, saved alongside exported results.
        :param databases: measurement databases available for widening the
            object table.
        :param counts: per-well counts, used to resolve control wells.
        :param results: the regression's coefficient table, as a frame or a
            CSV path, for the opening selection.
        """
        super().__init__(parent)
        self._objects = objects
        self._groups = dict(groups or {})
        self._settings = dict(settings or {})
        self._databases = tuple(databases or ())
        self._counts = counts
        from ..job_runner import JobRunner

        self._jobs = JobRunner(self, app_key="join measurements")
        self._joining = False
        self._comparison = None
        self._canvas = None
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
        for name in self._numeric_columns():
            self.measurement.addItem(str(name), str(name))
        self.measurement.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.measurement, 1)

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
        self.level.setCurrentIndex(1)
        self.level.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.level)

        row.addWidget(self._heading("plot"))
        self.kind = QComboBox()
        for value, label in PLOTS:
            self.kind.addItem(label, value)
        self.kind.setCurrentIndex(self._plot_to_start_on())
        self.kind.currentIndexChanged.connect(self.refresh)
        self.kind.currentIndexChanged.connect(self._offer_the_spread)
        row.addWidget(self.kind)

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

        row.addWidget(self._heading("show"))
        from ...well_scope import SCOPES

        self.scope = QComboBox()
        for value, caption in SCOPES:
            self.scope.addItem(caption, value)
        self.scope.setToolTip(
            "Objects included in the plot. 'gRNAs' displays only objects "
            "annotated with the guides selected on the volcano. 'gRNAs + "
            "other datapoints in selected wells' also displays the remaining "
            "objects from those wells with a distinct encoding. Comparing "
            "objects within the same wells controls for their shared plate, "
            "acquisition, staining, and imaging conditions.")
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
            "settings, and the cell images into one folder.")
        self.save_button.clicked.connect(self.save_everything)
        row.addWidget(self.save_button)
        layout.addLayout(row)

        second_row = QHBoxLayout()
        second_row.addWidget(self._heading("compare"))
        self.contrast = QComboBox()
        for value, label, why in CONTRASTS:
            self.contrast.addItem(label, value)
            self.contrast.setItemData(self.contrast.count() - 1, why,
                                      Qt.ToolTipRole)
        self.contrast.currentIndexChanged.connect(self._on_contrast)
        second_row.addWidget(self.contrast, 1)

        self.controls = QLineEdit()
        self.controls.setPlaceholderText("control gene or guide, comma "
                                          "separated")
        self.controls.setToolTip(
            "Control wells used for the comparison. A gene name selects all "
            "associated guides; a guide name selects only that guide.")
        self.controls.editingFinished.connect(self.refresh)
        self.controls.setEnabled(False)
        second_row.addWidget(self.controls, 1)

        self.wells_button = QPushButton("wells…")
        self.wells_button.setToolTip(
            "Choose which of the annotation's wells to include. A well left "
            "out is removed from both sides of the comparison.")
        self.wells_button.clicked.connect(self.choose_wells)
        second_row.addWidget(self.wells_button)
        layout.addLayout(second_row)

        selection_row = QHBoxLayout()
        self.selection_note = QLabel("")
        self.selection_note.setObjectName("Muted")
        self.selection_note.setWordWrap(True)
        selection_row.addWidget(self.selection_note, 1)
        self.guides_button = QPushButton(tr("gRNAs…"))
        self.guides_button.setToolTip(tr(
            "Choose the gRNAs the plot is about. It opens on the regression's "
            "top significant hits; the selected wells are derived from the "
            "gRNAs again whenever they change."))
        self.guides_button.clicked.connect(self._choose_guides)
        selection_row.addWidget(self.guides_button)
        self.scope_wells_button = QPushButton(tr("selected wells…"))
        self.scope_wells_button.setToolTip(tr(
            "Narrow the wells that carry the chosen gRNAs. Only these wells' "
            "objects are drawn under 'gRNAs + other datapoints in selected "
            "wells'."))
        self.scope_wells_button.clicked.connect(self._choose_selected_wells)
        selection_row.addWidget(self.scope_wells_button)
        layout.addLayout(selection_row)

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
        self._scope_report: Dict[str, Any] = {}
        self._results = results
        self._open_on_the_top_hits(results)
        self.refresh()


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
        a re-run.

        :param objects: the object rows for the montage and the contrasts; its
            numeric measurement columns fill the measurement menu, and the
            previous choice is kept when still offered.
        :param groups: selected group names mapped to object-index values;
            copied.
        """
        self._objects = objects
        self._groups = dict(groups or {})
        if settings is not None:
            self._settings = dict(settings)
        if not self._selected_guides:
            self._open_on_the_top_hits(getattr(self, "_results", None))
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
        """The last comparison computed, if any.

        :returns: the comparison, or None before one has been run.
        """
        return self._comparison


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
        png_list = bool(self.join_png_list.isChecked())
        objects, databases = self._objects, self._databases
        self._joining = True
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
        if not started:
            self._joining = False
            self._reset_the_join_button()
        return ""

    def _on_join_button(self, *_args) -> str:
        """Start the join, or cancel the one already running."""
        if self._joining:
            self.cancel_the_join()
            return ""
        return self.join_the_tables()

    def cancel_the_join(self) -> bool:
        """Request cancellation of the active join without blocking the GUI.

        :returns: ``True`` if a join was active.
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
        """Request cancellation of active work before closing the panel.

        :param event: the close event, passed on to the base class after the
            join worker is asked to shut down.
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
        self.set_data(wide, self._groups)
        if trouble:
            self.join_note.setText(
                f"{self.join_note.text()} {trouble}".strip())
        return trouble

    def _open_on_the_top_hits(self, results=None) -> list:
        """Pre-select the top hits and start on their wells. No redraw.

        :param results: the regression's coefficient table (frame or CSV
            path), or ``None`` to use the montage groups' guides.
        :returns: the guides selected, in table order.
        """
        from ...well_scope import _guides_at, _top_hits

        guides = _top_hits(self._objects, results)
        if not guides:
            members = [m for group in self._groups.values()
                       if group is not None for m in group]
            guides = _guides_at(self._objects, members)
        self._selected_guides = [str(g) for g in guides]
        self._selected_wells = None
        self.scope.blockSignals(True)
        self.scope.setCurrentIndex(max(0, self.scope.findData(
            "wells" if guides else "all")))
        self.scope.blockSignals(False)
        return list(self._selected_guides)

    def _guides_on_offer(self) -> list:
        """Every guide in the object table, in table order."""
        from ...well_scope import GUIDE_COLUMNS

        columns = getattr(self._objects, "columns", ())
        column = next((c for c in GUIDE_COLUMNS if c in columns), None)
        if column is None:
            return []
        return list(dict.fromkeys(self._objects[column].astype(str)))

    def _choose_guides(self, *_args) -> bool:
        """Open the gRNA checklist and apply a changed selection.

        :returns: ``True`` when the selection changed.
        """
        offered = self._guides_on_offer()
        if not offered:
            self.selection_note.setText(tr(
                "These object rows name no gRNA, so there is nothing to "
                "choose between."))
            return False
        before = list(self._selected_guides)
        dialog = _WellChoice(
            offered, set(before), parent=self, title=tr("Which gRNAs to plot"),
            note=tr("The plot opens on the regression's top hits. The "
                    "selected wells follow the gRNAs ticked here."))
        if dialog.exec() != QDialog.Accepted:
            return False
        chosen = [g for g in offered if g in dialog.chosen()]
        if chosen == before:
            return False
        self.set_selected_guides(chosen)
        return True

    def _choose_selected_wells(self, *_args) -> bool:
        """Open the checklist of the wells carrying the chosen gRNAs.

        :returns: ``True`` when the well set changed.
        """
        from ...well_scope import wells_of as scope_wells_of

        offered = scope_wells_of(self._objects, self._selected_guides)
        if not offered:
            self.selection_note.setText(tr(
                "No well carries the chosen gRNAs, so there are no wells to "
                "narrow."))
            return False
        before = self.selected_wells()
        dialog = _WellChoice(
            offered, set(before), parent=self,
            title=tr("Which selected wells to draw"),
            note=tr("These wells carry the chosen gRNAs. A well left out is "
                    "not drawn at all."))
        if dialog.exec() != QDialog.Accepted:
            return False
        chosen = [w for w in offered if w in dialog.chosen()]
        if chosen == before:
            return False
        self.set_selected_wells(chosen)
        return True

    def _show_the_selection(self) -> None:
        """Say which gRNAs and wells the plot is drawing, under the controls."""
        from ...well_scope import describe

        guides = list(self._selected_guides)
        wells = self.selected_wells()
        shown = ", ".join(guides[:6]) + (" …" if len(guides) > 6 else "")
        where = ", ".join(wells[:8]) + (" …" if len(wells) > 8 else "")
        lines = []
        if guides:
            lines.append(tr("gRNAs ({n}): {names}", n=len(guides),
                            names=shown))
            lines.append(tr("selected wells ({n}): {names}", n=len(wells),
                            names=where or tr("none")))
        else:
            lines.append(tr("No gRNA is selected."))
        if self._scope_report:
            lines.append(describe(self._scope_report))
        self.selection_note.setText("\n".join(lines))
        self.scope_wells_button.setEnabled(bool(guides))

    def _on_scope(self, *_args) -> None:
        """The population changed: re-derive the wells and redraw."""
        self._selected_wells = None
        self.refresh()

    def set_selected_guides(self, guides) -> None:
        """Set selected guides and derive their wells for the current scope.

        :param guides: the selected guides, stored as strings (``None`` for
            none); any explicit well subset is cleared so wells are derived
            from them.
        """
        self._selected_guides = [str(g) for g in (() if guides is None else guides)]
        self._selected_wells = None
        self.refresh()

    def selected_wells(self) -> list:
        """Return explicit wells or derive them from selected guides."""
        if self._selected_wells is not None:
            return list(self._selected_wells)
        from ...well_scope import wells_of

        return wells_of(self._objects, getattr(self, "_selected_guides", []))

    def set_selected_wells(self, wells) -> None:
        """Set an explicit well subset; ``None`` restores guide-based derivation.

        :param wells: an explicit well subset, stored as strings, or ``None``
            to derive the wells from the selected guides.
        """
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

            if getattr(self, "_heading_filter", None) is None:
                self._heading_filter = _ApiTooltipFilter(self)
            label.setProperty("apiTooltipHtml", label.toolTip())
            label.setProperty("apiTooltipDisplayRole", "tooltip")
            label.removeEventFilter(self._heading_filter)
            label.installEventFilter(self._heading_filter)
        except Exception:                                    # noqa: BLE001
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
        if contrast in ("against_controls", "against_other_wells"):
            frame = self._objects
            self._scope_report = {"note": tr(
                "The population box is set aside for this contrast: it "
                "compares against wells outside the selection.")}
        else:
            frame, self._scope_report = self.scoped_objects()
        self._show_the_selection()
        self._comparison = with_statistics(
            build(frame, measurement, groups=self._groups,
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

    def _plot_to_start_on(self) -> int:
        """Which of :data:`~spacr.gene_measurement_compare.PLOTS` to open on.

        :returns: an index into the plot box; ``0`` -- the panel's own first
            choice -- when the user has chosen no default graph type, or has
            chosen one this panel does not draw.

        THE SETTING DECIDES THE STARTING POINT HERE TOO. A comparison is
        groups against a measurement, so the default graph type saved for
        that shape is the one this box should open on; changing the box
        still changes this comparison, exactly as before.
        """
        try:
            from ...graph_types import chosen_for

            chosen = chosen_for("categorical_continuous")
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not read the default graph type", exc_info=True)
            return 0
        index = self.kind.findData(_PLOTS_OF_KIND.get(chosen, ""))
        return index if index >= 0 else 0

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
        """Redraw the plot for the current pickers.

        The class filter narrows what is DRAWN and never what the statistics
        below describe: a test computed on one of two groups is not a comparison
        at all, and quietly re-running it on the visible half would report a
        different question than the one on screen.
        """
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
                str(self.kind.currentData() or ""), "box_jitter")
            spec = PlotSpec(
                frame=frame, value="value", group="group", kind=kind,
                spread=(str(self.spread.currentData() or SPREAD_SEM)
                        if self._has_an_error_bar() else SPREAD_NONE),
                title=str(self.measurement.currentText() or ""),
                y_label=str(self.measurement.currentText() or "value"),
                unit=str(self.level.currentData() or "observation"),
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
    """The panel above, in a window. Kept so the button still opens one.

    Every parameter is handed straight to
    :class:`MeasurementComparePanel`, which documents what each one means.

    :param objects: object rows for the montage and reference contrasts.
    :param groups: selected group names mapped to object-index values.
    :param parent: parent widget.
    :param settings: run settings saved alongside exported results.
    :param databases: measurement databases available for widening the
        object table.
    :param counts: per-well counts, used to resolve control wells.
    """

    def __init__(self, objects, groups: Dict[str, Any],
                 parent: Optional[QWidget] = None,
                 settings: Optional[Dict[str, Any]] = None,
                 databases: Optional[Any] = None,
                 counts: Optional[Any] = None):
        """Build the window around one :class:`MeasurementComparePanel`.

        Every argument is handed straight through; the panel documents what each
        one means.

        :param objects: object rows for the montage and reference contrasts.
        :param groups: selected group names mapped to object-index values.
        :param parent: parent widget, or ``None``.
        :param settings: run settings saved alongside exported results.
        :param databases: measurement databases available for widening the
            object table.
        :param counts: per-well counts, used to resolve control wells.
        """
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

    def refresh(self, *args):
        """Forwarded to the panel this dialog wraps.

        :param args: passed straight through.
        :returns: whatever :class:`MeasurementComparePanel` returns.
        """
        return self.panel.refresh(*args)

    def comparison(self):
        """Forwarded to the panel this dialog wraps.

        :returns: whatever :class:`MeasurementComparePanel` answers.
        """
        return self.panel.comparison()

    def save_everything(self, folder: str = "") -> dict:
        """Forwarded to the panel this dialog wraps.

        :param folder: where to write everything.
        :returns: whatever the panel returns.
        """
        return self.panel.save_everything(folder)

    @property
    def measurement(self):
        """Forwarded to the panel this dialog wraps.

        :returns: whatever the panel answers.
        """
        return self.panel.measurement

    @property
    def level(self):
        """Forwarded to the panel this dialog wraps.

        :returns: whatever :class:`MeasurementComparePanel` answers.
        """
        return self.panel.level

    @property
    def kind(self):
        """Forwarded to the panel this dialog wraps.

        :returns: whatever :class:`MeasurementComparePanel` answers.
        """
        return self.panel.kind

    @property
    def report(self):
        """Forwarded to the panel this dialog wraps.

        :returns: whatever :class:`MeasurementComparePanel` answers.
        """
        return self.panel.report

    @property
    def contrast(self):
        """Forwarded to the panel this dialog wraps.

        :returns: whatever :class:`MeasurementComparePanel` answers.
        """
        return self.panel.contrast

    @property
    def controls(self):
        """Forwarded to the panel this dialog wraps.

        :returns: whatever :class:`MeasurementComparePanel` answers.
        """
        return self.panel.controls

    def choose_wells(self, *args):
        """Forwarded to the panel this dialog wraps.

        :param args: passed straight through.
        :returns: whatever :class:`MeasurementComparePanel` returns.
        """
        return self.panel.choose_wells(*args)

    def join_the_tables(self, *args):
        """Forwarded to the panel this dialog wraps.

        :param args: passed straight through.
        :returns: whatever :class:`MeasurementComparePanel` returns.
        """
        return self.panel.join_the_tables(*args)

    def cancel_the_join(self, *args):
        """Forwarded to the panel this dialog wraps.

        :param args: passed straight through.
        :returns: whatever :class:`MeasurementComparePanel` returns.
        """
        return self.panel.cancel_the_join(*args)
