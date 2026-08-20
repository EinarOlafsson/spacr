"""One button: every gene against every measurement.

The engine is :mod:`spacr.gene_measurement_sweep`; this is
the thin panel over it, and it is thin on purpose -- the three corrections
that make the answer trustworthy (identifiers excluded, Benjamini-Hochberg
across the grid, circularity reported per row) live in the engine so a
settings CSV, a macro and this button cannot disagree about them.

IT RUNS OFF THE GUI THREAD, and the reason is today's crash rather than
politeness: a regression built Qt widgets on its own worker and the process
segfaulted somewhere else entirely. So the worker here touches NO widget and
returns plain data, and the figure is rendered by matplotlib's Agg canvas
which has no Qt object to own.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                               QDoubleSpinBox, QFileDialog, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

LOG = logging.getLogger("spacr.qt.sweep_panel")


def _names(text) -> list:
    """A comma-separated box as a list of names, or ``[]``.

    Empty means "exclude nothing", and the caller turns that into an ABSENT
    keyword rather than an empty list: `sweep` distinguishes "no filter" from
    "a filter that matches nothing", and the second would be a silent way to
    keep everything while looking like a choice.
    """
    return [part.strip() for part in str(text or "").split(",")
            if part.strip()]

__all__ = ["SweepPanel", "sweep_inputs"]


def sweep_inputs(cells, counts, *, score_column: str = "pred", scores=None):
    """``(wells, fractions, plates, scores)`` from a merged frame and counts.

    THE PLATE NAMES ARE CANONICALISED ON BOTH SIDES. A score CSV of the real
    screen says `pplate1` where its measurements database says `plate1`, so an
    un-canonicalised join matches no well at all -- and the resulting all-NaN
    circularity column reads as "nothing here is circular", which is the most
    confident possible way to say nothing.
    """
    from ... import schema
    from ...multi_database import normalise_plate_ids

    frame = normalise_plate_ids(cells.copy())
    if "prc" not in frame.columns:
        frame["prc"] = [schema.compose_prc(p, r, c) for p, r, c in
                        zip(frame["plateID"], frame["rowID"],
                            frame["columnID"])]
    numeric = [c for c in frame.columns
               if pd.api.types.is_numeric_dtype(frame[c])]
    wells = frame.groupby("prc", observed=True)[numeric].mean()
    plates = wells.index.to_series().str.split("_").str[0]

    fractions = counts.pivot_table(index="prc", columns="grna",
                                   values="fraction", aggfunc="sum",
                                   fill_value=0.0)
    found = None
    if score_column in frame.columns:
        found = frame.groupby("prc", observed=True)[score_column].mean()
    elif scores is not None and len(scores):
        # The scores live in the run's own score CSVs, not in the measurement
        # tables -- and THEY say `pplate1` where the databases say `plate1`.
        offered = normalise_plate_ids(pd.DataFrame(scores).copy())
        if "prc" not in offered.columns and "plateID" in offered.columns:
            offered["prc"] = [schema.compose_prc(p, r, c) for p, r, c in
                              zip(offered["plateID"], offered["rowID"],
                                  offered["columnID"])]
        column = score_column if score_column in offered.columns else None
        if column and "prc" in offered.columns:
            found = offered.groupby("prc", observed=True)[column].mean()
    if found is not None:
        found = found.reindex(wells.index)
    return wells, fractions, plates, found


class SweepPanel(QWidget):
    """The button, the table, and the picture."""

    finished = Signal(object)

    def __init__(self, cells_provider: Optional[Callable] = None,
                 counts_provider: Optional[Callable] = None,
                 parent: Optional[QWidget] = None, *, threaded: bool = True,
                 scores_provider: Optional[Callable] = None):
        super().__init__(parent)
        from ..job_runner import JobRunner

        self._cells_provider = cells_provider
        self._counts_provider = counts_provider
        #: The per-object scores, so circularity can be computed at all. The
        #: merged measurements frame has no `pred` column -- it is the
        #: measurement tables -- so without this the column is NaN and the
        #: panel says so rather than showing zeros.
        self._scores_provider = scores_provider
        self._result = None
        #: Picture windows this panel opened, kept so Python does not collect
        #: them the moment `show_picture` returns.
        self._pictures: list = []
        self._jobs = JobRunner(self, threaded=bool(threaded),
                               app_key="sweep every gene")
        self._jobs.job_failed.connect(self._failed)

        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.run_button = QPushButton("Sweep every gene against every measurement")
        self.run_button.setToolTip(
            "Every guide against every measurement, blocked on plate and "
            "corrected across the whole grid. Identifier columns are left "
            "out, and every row says how far the classification score "
            "already tracks that measurement — a measurement the score "
            "predicts cannot corroborate anything derived from it.")
        self.run_button.clicked.connect(self.start)
        row.addWidget(self.run_button)

        # THE HELP GOES ON A NAME, so there has to be one. This combo had
        # no label at all, which is why its tooltip sat on the field:
        # `retarget_field_tooltips` pairs a field with a sibling label and
        # correctly leaves a field that has none alone (113).
        self._level_label = QLabel("rank by")
        row.addWidget(self._level_label)
        self.level = QComboBox()
        for value, label in (("gene", "genes"), ("guide", "guides"),
                             ("both", "genes and guides")):
            self.level.addItem(label, value)
        self._level_label.setToolTip(
            "A gene's fraction in a well is the SUM of its guides', which is "
            "the same rule the regression applies — so 'does this gene move "
            "this measurement' is not a different arithmetic from the fit "
            "that found the gene.")
        row.addWidget(self.level)

        # WHICH PICTURE. The heatmap answers "what moved"; it cannot answer
        # "is this gene just over-represented", "what KIND of thing does it
        # move" or "do its own guides agree", and those are the questions
        # that decide whether a hit is worth following up.
        self._picture_label = QLabel("picture")
        row.addWidget(self._picture_label)
        self.picture = QComboBox()
        for value, label in self.PICTURES:
            self.picture.addItem(label, value)
        self._picture_label.setToolTip(
            "Which view of the sweep to draw and to save beside the table. "
            "Each answers a question the others cannot — the heatmap says "
            "what survived, calibration says whether the screen is worth "
            "reading at all, and the rest are in the list.")
        row.addWidget(self.picture)

        # WHAT TO LEAVE OUT, on its own row. Asked for 2026-08-19: "there
        # should be the option to remove columns befor the sweep and remove
        # specific genes or guides and to remove over represented guides".
        # Three separate controls because they are three different
        # judgements -- a column you do not trust, a gene you already know
        # about, and a guide whose breadth is doing the work its biology is
        # being credited with.
        leave_out = QHBoxLayout()
        leave_out.addWidget(QLabel("leave out — measurements"))
        self.drop_columns = QLineEdit()
        self.drop_columns.setPlaceholderText("column names, comma separated")
        self.drop_columns.setToolTip(
            "Measurement columns to keep out of the sweep, by name and comma "
            "separated. Naming the two or three that are wrong is easier "
            "than listing the seven hundred that are not.")
        leave_out.addWidget(self.drop_columns, 1)

        leave_out.addWidget(QLabel("genes or guides"))
        self.drop_genes = QLineEdit()
        self.drop_genes.setPlaceholderText("e.g. 220950, 233460")
        self.drop_genes.setToolTip(
            "Genes or guides to keep out, by name and comma separated. "
            "Matched at BOTH levels, so a gene id works whether the sweep is "
            "running at gene or guide level.")
        leave_out.addWidget(self.drop_genes, 1)

        self.cap_wells = QCheckBox("drop guides in more than")
        self.cap_wells.setToolTip(
            "Leave out a guide present in more than this share of the wells. "
            "THE OVER-REPRESENTATION FILTER: a guide in every well has the "
            "statistical weight of the whole screen behind it, so it clears "
            "the correction on measurements a rarer guide could never reach "
            "— see the 'hits vs representation' picture for what that looks "
            "like on your own screen.")
        leave_out.addWidget(self.cap_wells)
        self.cap_wells_value = QDoubleSpinBox()
        self.cap_wells_value.setDecimals(2)
        self.cap_wells_value.setRange(0.05, 1.0)
        self.cap_wells_value.setSingleStep(0.05)
        self.cap_wells_value.setValue(0.5)
        self.cap_wells_value.setSuffix(" of wells")
        leave_out.addWidget(self.cap_wells_value)
        leave_out.addStretch(1)
        layout.addLayout(leave_out)

        row.addWidget(QLabel("q <"))
        self.alpha = QDoubleSpinBox()
        self.alpha.setDecimals(3)
        self.alpha.setRange(0.001, 0.5)
        self.alpha.setSingleStep(0.01)
        self.alpha.setValue(0.05)
        self.alpha.valueChanged.connect(self._refill)
        row.addWidget(self.alpha)

        self.hide_circular = QCheckBox("hide what the score already tracks")
        self.hide_circular.setToolTip(
            "Leaves out measurements the classification score predicts. They "
            "are not wrong, they just cannot corroborate a result derived "
            "from that score.")
        self.hide_circular.setChecked(True)
        self.hide_circular.stateChanged.connect(self._refill)
        row.addWidget(self.hide_circular)

        row.addStretch(1)
        # SHOW THE PICTURE. The chooser and the ten views existed with no way
        # to look at any of them -- `figure()` was reachable only through
        # Save, so the answer to "i ran a measurement sweep how do i see the
        # graphs?" was "you write them to disk and open them yourself". A
        # picture nobody can look at is the same defect as a setter nobody
        # calls.
        self.show_button = QPushButton("Show picture")
        self.show_button.setToolTip(
            "Draw the chosen view of this sweep in its own window.")
        self.show_button.clicked.connect(self.show_picture)
        self.show_button.setEnabled(False)
        row.addWidget(self.show_button)

        self.save_button = QPushButton("Save table…")
        self.save_button.clicked.connect(self.save)
        self.save_button.setEnabled(False)
        row.addWidget(self.save_button)
        layout.addLayout(row)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["level", "gene / guide", "measurement", "effect", "q",
             "circularity", "wells", "effective"])
        self.table.setSortingEnabled(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.table, 1)

    # ------------------------------------------------------------- running

    def start(self, *_args) -> bool:
        """Run the sweep. Returns whether one was started."""
        if self._cells_provider is None or self._counts_provider is None:
            self.status.setText(
                "Nothing to sweep: this panel has no measurements and no "
                "counts wired to it.")
            return False
        try:
            cells = self._cells_provider()
            counts = self._counts_provider()
        except Exception as error:                       # noqa: BLE001
            self.status.setText(f"Could not read the inputs: {error}")
            return False
        if cells is None or not len(cells) or counts is None or not len(counts):
            self.status.setText(
                "Nothing to sweep: merge the measurement databases first, and "
                "attach the count CSVs the run was fitted on.")
            return False

        self.run_button.setEnabled(False)
        self.status.setText("Sweeping…")
        # THE WORKER TOUCHES NO WIDGET. It returns the result object and the
        # GUI thread does the rest -- which is the rule a regression broke
        # today by building Qt widgets on its own worker.
        scores = None
        if self._scores_provider is not None:
            try:
                scores = self._scores_provider()
            except Exception:                            # noqa: BLE001
                LOG.debug("could not read the scores", exc_info=True)
        level = str(self.level.currentData() or "gene")
        exclusions = self.exclusions()
        return bool(self._jobs.submit(
            lambda: self._work(cells, counts, scores, level, exclusions),
            self._done))

    @staticmethod
    def _work(cells, counts, scores=None, level="gene", exclusions=None):
        from ...gene_measurement_sweep import sweep

        wells, fractions, plates, found = sweep_inputs(cells, counts,
                                                       scores=scores)
        return sweep(wells, fractions, blocks=plates, scores=found,
                     level=level, **dict(exclusions or {}))

    def _done(self, result) -> None:
        self.run_button.setEnabled(True)
        self._result = result
        if result is None:
            self.status.setText("The sweep returned nothing.")
            return
        self.save_button.setEnabled(True)
        self.show_button.setEnabled(True)
        self.status.setText(result.describe())
        self._refill()
        self.finished.emit(result)

    def _failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.status.setText(f"The sweep did not finish: {message}")

    # -------------------------------------------------------------- the view

    def rows(self) -> pd.DataFrame:
        """What the table is showing, as a frame."""
        if self._result is None:
            return pd.DataFrame()
        # A CIRCULARITY BAR THE RESULT CANNOT HONOUR IS REFUSED, not silently
        # applied to a column of NaN -- which returns nothing and looks like
        # an answer.
        bar = 0.15 if (self.hide_circular.isChecked()
                       and self._result.circularity_known) else 1.0
        return self._result.survivors(alpha=float(self.alpha.value()),
                                      max_circularity=bar)

    def _refill(self, *_args) -> None:
        keep = self.rows()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        shown = keep.head(2000)
        self.table.setRowCount(len(shown))
        for row, (_index, entry) in enumerate(shown.iterrows()):
            values = [str(entry.get("level", "guide")), str(entry["guide"]),
                      str(entry["measurement"]),
                      f"{entry['effect']:+.3f}", f"{entry['q']:.2e}",
                      ("—" if pd.isna(entry["circularity"])
                       else f"{entry['circularity']:.2f}"),
                      str(int(entry["n_wells"])),
                      f"{entry['effective_wells']:.0f}"]
            for column, text in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(text))
        self.table.setSortingEnabled(True)
        if len(keep) > len(shown):
            self.status.setText(
                f"{self.status.text()}  Showing the first {len(shown):,} of "
                f"{len(keep):,} — save the table for all of them.")

        # HOVER HELP BELONGS ON THE SETTING'S NAME (instruction 113): a
        # tooltip on the control fires while the user is using it, which is
        # the one moment they did not ask for it. One call rather than a
        # convention to remember -- see `retarget_field_tooltips`.
        from ..screens.settings_model import retarget_field_tooltips

        retarget_field_tooltips(self)

    def exclusions(self) -> dict:
        """What the user asked to leave out, as `sweep` keyword arguments.

        READ OFF THE BOXES rather than stored, so the answer is always the
        one on screen. Empty values mean "exclude nothing", which is why a
        blank box has to come back as an absent key and not as an empty list:
        `sweep` distinguishes "no filter" from "a filter that matches
        nothing".
        """
        out: dict = {}
        columns = _names(self.drop_columns.text())
        if columns:
            out["drop_measurements"] = columns
        genes = _names(self.drop_genes.text())
        if genes:
            out["drop_guides"] = genes
        if self.cap_wells.isChecked():
            out["max_wells_fraction"] = float(self.cap_wells_value.value())
        return out

    def selected_gene(self):
        """The gene the profile picture is about, or ``None``.

        The row the user selected in the table, and failing that the
        strongest survivor -- which is a DEFAULT, not a choice they made, so
        `plot_gene_profile` puts the name in the title. Returning None when
        there is nothing at all keeps the caller from drawing a picture of
        a gene it invented.
        """
        rows = {index.row() for index in self.table.selectedIndexes()}
        if rows:
            item = self.table.item(sorted(rows)[0], 1)   # the gene column
            if item is not None and item.text().strip():
                return item.text().strip()
        if self._result is None or not len(self._result.table):
            return None
        best = self._result.table.sort_values("q")
        return str(best.iloc[0]["guide"]) if len(best) else None

    #: The pictures this panel can draw, in the order the chooser offers
    #: them. Each answers a question the others cannot -- see instruction
    #: 175 and `spacr.gene_measurement_sweep`.
    #: THE HEATMAP STAYS FIRST, and so stays the default. Calibration is the
    #: first thing to READ -- it says whether the other nine are worth
    #: anything -- but a panel that opened on a QQ plot when the user asked
    #: for "a comprehensive table and an intuitive visualisation" would be
    #: answering a question they did not ask.
    PICTURES = (
        ("heatmap", "what survived, clustered"),
        ("calibration", "is the screen calibrated at all?"),
        ("volcano", "every pair: effect against evidence"),
        ("representation", "hits vs how much screen the gene is"),
        ("families", "what KIND of measurement each gene moves"),
        ("profile", "one gene's fingerprint"),
        ("similarity", "which genes behave alike"),
        ("measurements", "which measurements discriminate"),
        ("circularity", "corroboration or restatement"),
        ("concordance", "do a gene's own guides agree?"),
    )

    def figure(self, path: Optional[str] = None, kind: Optional[str] = None):
        """One picture of the sweep, or ``None`` when there is nothing to draw.

        :param kind: one of :data:`PICTURES`; the chooser's current pick by
            default.
        """
        if self._result is None:
            return None
        from ...gene_measurement_sweep import (
            plot_calibration, plot_circularity,
            plot_effect_against_representation, plot_gene_profile,
            plot_gene_similarity, plot_grid_volcano, plot_guide_concordance,
            plot_measurement_families, plot_measurement_hits, plot_sweep)

        wanted = str(kind or self.picture.currentData() or "heatmap")
        alpha = float(self.alpha.value())
        bar = 0.15 if (self.hide_circular.isChecked()
                       and self._result.circularity_known) else 1.0
        # The picture follows the level the panel is showing, so a "both"
        # sweep does not draw a gene and its own guides as if they were
        # independent agreement.
        chosen = str(self.level.currentData() or "gene")
        level = None if chosen == "guide" else "gene"

        if wanted == "calibration":
            return plot_calibration(self._result, path=path, level=level)
        if wanted == "volcano":
            return plot_grid_volcano(self._result, path=path, alpha=alpha,
                                     level=level)
        if wanted == "similarity":
            return plot_gene_similarity(self._result, path=path, alpha=alpha,
                                        level=level)
        if wanted == "measurements":
            return plot_measurement_hits(self._result, path=path, alpha=alpha,
                                         level=level)
        if wanted == "circularity":
            return plot_circularity(self._result, path=path, alpha=alpha,
                                    level=level)
        if wanted == "profile":
            # THE ONE VIEW THAT NEEDS A SUBJECT. The row the user selected in
            # the table is the gene they are asking about; with nothing
            # selected the strongest survivor is the honest default, and it
            # is named in the title so nobody mistakes it for a choice they
            # made.
            gene = self.selected_gene()
            if gene is None:
                return None
            return plot_gene_profile(self._result, gene, path=path,
                                     alpha=alpha)
        if wanted == "representation":
            return plot_effect_against_representation(
                self._result, path=path, alpha=alpha, level=level)
        if wanted == "families":
            return plot_measurement_families(
                self._result, path=path, alpha=alpha, level=level)
        if wanted == "concordance":
            # NOT given `level`: this picture IS the guide comparison, and
            # passing the panel's gene default would leave it nothing to
            # compare. It says so by drawing nothing when the sweep was run
            # at gene level.
            return plot_guide_concordance(self._result, path=path,
                                          alpha=alpha)
        return plot_sweep(self._result, path=path, alpha=alpha,
                          max_circularity=bar, level=level)

    def show_picture(self, *_args):
        """Draw the chosen view in its own window. Returns the dialog, or None.

        A WINDOW RATHER THAN A PANE, because the Measurements tab is already
        four sections in a side panel -- "there are to many elements in the
        measurements tab" -- and a heatmap of forty measurements needs more
        width than that column has.
        """
        from PySide6.QtWidgets import QDialog, QVBoxLayout

        if self._result is None:
            self.status.setText("Run the sweep first — there is nothing to "
                                "draw yet.")
            return None
        kind = str(self.picture.currentData() or "heatmap")
        label = dict(self.PICTURES).get(kind, kind)
        try:
            figure = self.figure(kind=kind)
        except Exception as exc:                         # noqa: BLE001
            LOG.debug("could not draw the sweep picture", exc_info=True)
            self.status.setText(f"That picture could not be drawn: {exc}")
            return None
        if figure is None:
            # NOTHING TO DRAW IS AN ANSWER, and it is said here rather than
            # by opening an empty window -- which reads as a broken button.
            self.status.setText(
                f"Nothing to draw for “{label}” at q < "
                f"{float(self.alpha.value()):g}. "
                "Loosen the q filter, or pick another picture.")
            return None

        from .graph_builder import _canvas_class

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Sweep — {label}")
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(4, 4, 4, 4)
        canvas = _canvas_class()(figure)
        layout.addWidget(canvas)
        dialog.resize(1000, 720)
        dialog.show()
        # KEPT, or Python collects the dialog the moment this returns and the
        # window vanishes as it appears.
        self._pictures.append(dialog)
        return dialog

    def save(self, *_args) -> str:
        """Write the whole table -- not the page on screen."""
        if self._result is None:
            return ""
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save the sweep", "gene_measurement_sweep.csv",
            "CSV (*.csv)")
        if not path:
            return ""
        self._result.table.to_csv(path, index=False)
        figure_path = os.path.splitext(path)[0] + ".png"
        try:
            self.figure(path=figure_path)
        except Exception:                                # noqa: BLE001
            LOG.debug("could not draw the sweep figure", exc_info=True)
        self.status.setText(f"Saved {len(self._result.table):,} rows to {path}")
        return path

    def closeEvent(self, event):                         # noqa: N802 - Qt name
        try:
            self._jobs.shutdown()
        finally:
            super().closeEvent(event)
