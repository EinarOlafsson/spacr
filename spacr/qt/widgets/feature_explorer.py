"""The feature explorer — the ranked list, and the distributions behind it.

The chrome over :mod:`spacr.qt.widgets.feature_rank`. Everything about *how
separated* a feature is lives there, together with which statistic was used and
what it cannot see; this module is a table of the ranking and a small multiple
of the distributions.

Two things the panel refuses to let the ranking hide
-----------------------------------------------------

**The statistic's blind spot is on screen, not in a manual.** The picker's
tooltip carries :data:`~spacr.qt.widgets.feature_rank.STATISTIC_FAILURE_MODES`
verbatim, and a feature flagged
:attr:`~spacr.qt.widgets.feature_rank.FeatureScore.is_shape_not_shift` — the
classes differ in spread, which a rank statistic scores at 0.5 — is marked in
the list even though it ranks low.

**n is in the table.** Every row shows the smallest class behind the score,
because a separation over four objects and one over four thousand are the same
number, and the difference between them is the whole question.

Drawing it
----------

The panel draws the top features as a strip of small histograms, one row per
feature, the classes overlaid on **shared bin edges** — the same rule
:func:`spacr.qt.widgets.graph_spec.scales_for` applies across facets. Two class
histograms drawn on their own edges are two pictures of nothing.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QHeaderView, QLabel, QSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..theme import SPACING, active_palette, mark_surface
from .feature_rank import (
    AUC, DEFAULT_TOP, STATISTIC_FAILURE_MODES, STATISTIC_LABELS, STATISTICS,
    ExplorerError, ExplorerResult, ExplorerSpec, candidate_labels,
    distributions, rank_features,
)
from .toggle import Toggle
from .graph_builder import (_canvas_class, _page_surface_axes,
                            categorical_colours)

LOG = logging.getLogger("spacr.qt.feature_explorer")

__all__ = ["FeatureExplorerPanel", "MAX_DRAWN"]

#: Features drawn as distributions at once. Beyond this the strips are a few
#: pixels tall and say nothing; the table still lists the rest.
MAX_DRAWN = 8

#: Re-ranking is coalesced this long.
DEBOUNCE_MS = 150


class FeatureExplorerPanel(QWidget):
    """Rank features by how well they separate the classes, and draw the top."""

    #: Emitted after every ranking with the :class:`ExplorerResult`.
    ranked = Signal(object)
    #: A feature was selected in the table — carries its name.
    feature_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FeatureExplorerPanel")
        self._frame: Optional[pd.DataFrame] = None
        self._result: Optional[ExplorerResult] = None
        self._spec = ExplorerSpec()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["sm"], SPACING["sm"],
                                 SPACING["sm"], SPACING["sm"])
        outer.setSpacing(SPACING["xs"])

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(SPACING["xs"])

        controls.addWidget(QLabel("Split by", self))
        self._label = QComboBox(self)
        self._label.setObjectName("ExplorerLabelPicker")
        self._label.setToolTip(
            "The column saying which class or condition each object is in.")
        self._label.currentTextChanged.connect(self._schedule)
        controls.addWidget(self._label, 1)

        controls.addWidget(QLabel("Rank by", self))
        self._statistic = QComboBox(self)
        self._statistic.setObjectName("ExplorerStatisticPicker")
        for statistic in STATISTICS:
            self._statistic.addItem(
                STATISTIC_LABELS[statistic].split(" — ")[0], statistic)
        # The blind spots, on screen rather than in a manual.
        self._statistic.setToolTip("\n\n".join(
            f"{STATISTIC_LABELS[s]}\ncannot see: {STATISTIC_FAILURE_MODES[s]}"
            for s in STATISTICS))
        self._statistic.currentIndexChanged.connect(self._schedule)
        controls.addWidget(self._statistic, 1)

        controls.addWidget(QLabel("Top", self))
        self._top = QSpinBox(self)
        self._top.setRange(1, 500)
        self._top.setValue(DEFAULT_TOP)
        self._top.valueChanged.connect(self._schedule)
        controls.addWidget(self._top)

        self._null = Toggle("Shuffle test", self)
        self._null.setToolTip(
            "Permute the class labels and re-rank, 50 times, keeping the best "
            "score each time. The 95th percentile is the separation the best "
            "of your features reaches by chance — a feature below it is not "
            "news. Costs a pass per shuffle.")
        self._null.toggled.connect(self._schedule)
        controls.addWidget(self._null)
        outer.addLayout(controls)

        self._blind = QLabel("", self)
        self._blind.setObjectName("ExplorerBlindSpot")
        self._blind.setWordWrap(True)
        outer.addWidget(self._blind)

        body = QSplitter(Qt.Horizontal, self)
        body.setChildrenCollapsible(False)
        self.table = QTableWidget(self)
        self.table.setObjectName("ExplorerTable")
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Feature", "Separation", "AUC", "KS", "higher in", "min n"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.currentCellChanged.connect(self._on_row_changed)
        # `FeatureExplorerPanel` is transparent scaffolding by design
        # (see the GraphBuilder block), so the ranking table is the page
        # here; the distribution canvas beside it paints its own panel.
        mark_surface(self.table)
        body.addWidget(self.table)

        self._figure_holder = QWidget(self)
        holder = QVBoxLayout(self._figure_holder)
        holder.setContentsMargins(0, 0, 0, 0)
        from matplotlib.figure import Figure
        # No `facecolor`: the canvas paints the page panel in its own
        # `paintEvent` under a transparent figure patch, so a solid one here
        # would put the opaque rectangle straight back.
        self._figure = Figure(figsize=(5.0, 6.0))
        self._canvas = _canvas_class()(self._figure)
        holder.addWidget(self._canvas, 1)
        body.addWidget(self._figure_holder)
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 1)
        outer.addWidget(body, 1)

        self._summary = QLabel("no table loaded", self)
        self._summary.setObjectName("ExplorerSummary")
        self._summary.setWordWrap(True)
        outer.addWidget(self._summary)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self.rank_now)
        # Hover help belongs on a setting's NAME, not on the field the user
        # is about to type into (instruction 113). One post-pass rather than
        # a convention every hand-built row has to remember.
        from ..screens.settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)

    # -- data -------------------------------------------------------------
    def set_frame(self, frame: Optional[pd.DataFrame]) -> None:
        """Point the panel at a table and offer its class columns."""
        self._frame = frame
        previous = self._label.currentText()
        self._label.blockSignals(True)
        self._label.clear()
        if frame is not None:
            self._label.addItems(list(candidate_labels(frame)))
            if previous:
                index = self._label.findText(previous)
                if index >= 0:
                    self._label.setCurrentIndex(index)
        self._label.blockSignals(False)
        self.rank_now()

    @property
    def spec(self) -> ExplorerSpec:
        return self._spec

    @property
    def result(self) -> Optional[ExplorerResult]:
        return self._result

    def set_spec(self, spec: ExplorerSpec) -> None:
        """Push a whole spec in — restoring a saved analysis."""
        self._spec = spec
        index = self._label.findText(spec.label)
        self._label.blockSignals(True)
        if index >= 0:
            self._label.setCurrentIndex(index)
        self._label.blockSignals(False)
        position = self._statistic.findData(spec.statistic)
        if position >= 0:
            self._statistic.blockSignals(True)
            self._statistic.setCurrentIndex(position)
            self._statistic.blockSignals(False)
        self._top.blockSignals(True)
        self._top.setValue(spec.top)
        self._top.blockSignals(False)
        self._null.blockSignals(True)
        self._null.setChecked(bool(spec.n_permutations))
        self._null.blockSignals(False)
        self.rank_now()

    def current_spec(self) -> ExplorerSpec:
        return ExplorerSpec(
            label=self._label.currentText(),
            features=self._spec.features,
            statistic=self._statistic.currentData() or AUC,
            top=int(self._top.value()),
            bins=self._spec.bins,
            n_permutations=50 if self._null.isChecked() else 0,
            seed=self._spec.seed)

    def summary(self) -> str:
        return self._summary.text()

    # -- ranking ----------------------------------------------------------
    def _schedule(self, *_args) -> None:
        self._debounce.start()

    def rank_now(self) -> Optional[ExplorerResult]:
        """Re-rank and redraw. Returns the result, or ``None`` on a refusal."""
        self._debounce.stop()
        statistic = self._statistic.currentData() or AUC
        self._blind.setText(
            f"{STATISTIC_LABELS[statistic]} — cannot see: "
            f"{STATISTIC_FAILURE_MODES[statistic]}")
        if self._frame is None:
            self._summary.setText("no table loaded")
            return None
        self._spec = self.current_spec()
        try:
            result = rank_features(self._frame, self._spec)
        except ExplorerError as exc:
            self._result = None
            self.table.setRowCount(0)
            self._figure.clear()
            self._figure.patch.set_alpha(0.0)
            self._canvas.draw_idle()
            self._summary.setText(str(exc))
            return None
        self._result = result
        self._fill_table(result)
        self._draw(result)
        self._summary.setText(result.summary())
        self.ranked.emit(result)
        return result

    def _fill_table(self, result: ExplorerResult) -> None:
        palette = active_palette()
        self.table.setRowCount(len(result.scores))
        for row, score in enumerate(result.scores):
            cells = [
                score.feature,
                f"{score.score:.3f}",
                f"{score.auc:.3f}" if np.isfinite(score.auc) else "",
                f"{score.ks:.3f}" if np.isfinite(score.ks) else "",
                score.higher_in,
                f"{score.smallest_class:,}",
            ]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, score.feature)
                item.setToolTip(score.describe())
                if score.is_shape_not_shift:
                    item.setForeground(_brush(palette["warning"]))
                elif (result.null_threshold is not None
                      and score.score <= result.null_threshold):
                    item.setForeground(_brush(palette["fg_muted"]))
                self.table.setItem(row, column, item)
        if len(result.scores):
            self.table.selectRow(0)

    def _draw(self, result: ExplorerResult) -> None:
        """A strip per feature, the classes overlaid on shared bin edges."""
        self._figure.clear()
        self._figure.patch.set_alpha(0.0)
        palette = active_palette()
        drawn = result.scores[:MAX_DRAWN]
        if not drawn or self._frame is None:
            self._canvas.draw_idle()
            return
        axes = self._figure.subplots(len(drawn), 1, squeeze=False,
                                     sharex=False)
        colours = categorical_colours()
        for row, score in enumerate(drawn):
            ax = axes[row][0]
            _page_surface_axes(ax, palette)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(palette["border"])
            ax.tick_params(colors=palette["fg_muted"], labelsize=7, length=2)
            edges, counts = distributions(self._frame, score.feature,
                                          result.label, bins=self._spec.bins)
            if not len(edges):
                continue
            centres = (edges[:-1] + edges[1:]) / 2.0
            width = float(np.diff(edges).mean()) * 0.9
            for i, (level, values) in enumerate(counts.items()):
                ax.bar(centres, values, width=width, alpha=0.55,
                       color=colours[i % len(colours)],
                       label=level if row == 0 else None, linewidth=0.0)
            flag = "  ·  shape, not shift" if score.is_shape_not_shift else ""
            ax.set_title(
                f"{score.feature}  ·  {result.spec.statistic} "
                f"{score.score:.3f}  ·  higher in {score.higher_in}"
                f"  ·  min n {score.smallest_class:,}{flag}",
                color=(palette["warning"] if score.is_shape_not_shift
                       else palette["fg_dim"]),
                fontsize=8, pad=2, loc="left")
            ax.set_yticks([])
        if len(counts) > 1:
            self._figure.legend(loc="upper right", frameon=False, fontsize=7)
        self._figure.tight_layout(pad=0.6)
        self._canvas.draw_idle()

    def _on_row_changed(self, row: int, *_args) -> None:
        item = self.table.item(row, 0)
        if item is not None:
            self.feature_selected.emit(item.data(Qt.UserRole))

    def selected_feature(self) -> str:
        item = self.table.item(self.table.currentRow(), 0)
        return item.data(Qt.UserRole) if item is not None else ""

    def closeEvent(self, event):  # noqa: N802 - Qt name
        self._debounce.stop()
        if hasattr(self._canvas, "cancel_pending_draw"):
            self._canvas.cancel_pending_draw()
        super().closeEvent(event)


def _brush(colour: str):
    from PySide6.QtGui import QBrush, QColor
    return QBrush(QColor(colour))
