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
:func:`spacr.qt.widgets.graph_spec.scales_for` applies across facets. Class
histograms drawn on separate bin edges cannot be compared bin by bin.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QHeaderView, QLabel, QScrollArea, QSpinBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..theme import SPACING, active_palette, mark_surface
from .feature_rank import (
    AUC, DEFAULT_TOP, STATISTIC_FAILURE_MODES, STATISTIC_LABELS, STATISTICS,
    ExplorerError, ExplorerResult, ExplorerSpec, candidate_labels,
    distributions, rank_features,
)
from .collapsible_splitter import CollapsibleSplitter
from .toggle import Toggle
from .graph_builder import (_canvas_class, _page_surface_axes,
                            categorical_colours)
from .sortable_table import (
    install_sorting, table_item, _retire_table_cells, _spare_table_item)

LOG = logging.getLogger("spacr.qt.feature_explorer")

__all__ = ["FeatureExplorerPanel", "MAX_DRAWN"]

#: Features drawn as distributions at once. Beyond this the strips are a few
#: pixels tall and say nothing; the table still lists the rest.
MAX_DRAWN = 8

#: Re-ranking is coalesced this long.
DEBOUNCE_MS = 150



def _present_feature_histogram(ax, frame, score, target):
    """Declare the existing class-count histogram without changing its bins or ranking.

    :param ax: axes containing the original per-class bars.
    :param frame: unchanged source measurement table and explicit unit metadata.
    :param score: original ranking result for the displayed feature.
    :param target: existing figure output target.
    :returns: the complete external caption.
    """
    from ...figures.style import apply_axes_style, resolve_style
    from ..i18n import tr
    from .feature_rank import COHEN_D, KS, MUTUAL_INFO

    names = {AUC: tr("absolute rank separation |2 × AUC − 1|"),
             COHEN_D: tr("absolute standardized mean difference"),
             KS: tr("Kolmogorov–Smirnov distance"),
             MUTUAL_INFO: tr("mutual information normalized by class entropy")}
    shape = tr(" Shape differs without a location shift.") if score.is_shape_not_shift else ""
    caption = tr("{feature}: {statistic} = {score:.3f}; values higher in {higher}; minimum class size {minimum:,}.{shape}").format(
        feature=score.feature, statistic=names.get(score.statistic, score.statistic),
        score=score.score, higher=score.higher_in, minimum=score.smallest_class, shape=shape)
    reference = tr("Observed classes use the same feature-bin edges")
    specification = resolve_style(target, "histogram",
        comparison={"reference":reference, "graph_type":"histogram",
                    "experimental_unit":frame.attrs.get("experimental_unit")},
        legend={"caption":caption, "reference":reference,
                "quantities":{"x":score.feature,"y":tr("Observation count")},
                "units":{"x":(frame.attrs.get("units") or {}).get(score.feature),"y":"observations"},
                "abbreviations":{"AUC":tr("Area under the receiver operating characteristic curve")}})
    apply_axes_style(ax, specification, layout=False)
    return caption


def _render_feature_histograms(figure, frame, result, bins, *, colours, alpha, font_points, presentation=True, target="screen"):
    """Draw original class histograms from the existing distribution engine.

    :param figure: destination figure, cleared before drawing.
    :param frame: unchanged source measurement table.
    :param result: original ranking and its ordered feature records.
    :param bins: existing requested histogram bin count.
    :param colours: explicit shared or captured class colours.
    :param alpha: explicit shared or captured data opacity.
    :param font_points: shared or captured legend text size.
    :param presentation: declare current application policy when true;
        captured recreation restores its recorded appearance when false.
    :param target: application presentation target.
    :returns: axes in the original feature order.
    """
    from ...figures.style import panel_letter
    from ...qt.i18n import tr

    figure.clear()
    drawn = result.scores[:MAX_DRAWN]
    if not drawn:
        return []
    axes = list(figure.subplots(len(drawn), 1, squeeze=False, sharex=False).ravel())
    first = None
    captions = []
    for row, score in enumerate(drawn):
        ax = axes[row]
        edges, counts = distributions(frame, score.feature, result.label, bins=bins)
        if not len(edges):
            continue
        centres = (edges[:-1] + edges[1:]) / 2.0
        width = float(np.diff(edges).mean()) * 0.9
        for i, (level, values) in enumerate(counts.items()):
            ax.bar(centres, values, width=width, alpha=alpha,
                   color=colours[i % len(colours)], label=level if first is None else None,
                   linewidth=0.0)
        ax.set_xlabel(score.feature)
        ax.set_ylabel(tr("Observation count"))
        if len(drawn) > 1:
            panel_letter(ax, row)
        if presentation:
            captions.append(_present_feature_histogram(ax, frame, score, target))
        if first is None:
            first = ax
    if first is not None:
        first.legend(loc="upper left", bbox_to_anchor=(1.02,1.), frameon=False, fontsize=font_points)
    if presentation:
        figure._spacr_caption = "\n\n".join(captions)
    return axes


class FeatureExplorerPanel(QWidget):
    """Rank features by how well they separate the classes, and draw the top.

    :param parent: parent widget.
    """

    #: Emitted after every ranking with the :class:`ExplorerResult`.
    ranked = Signal(object)
    #: A feature was selected in the table — carries its name.
    feature_selected = Signal(str)

    def __init__(self, parent=None):
        """Build the feature ranking table beside its distribution plot.

        Each statistic's blind spot is on the picker's tooltip rather than in a
        manual: which differences it cannot see is the thing a reader needs
        before trusting a ranking made with it.

        :param parent: parent widget, or ``None``.
        """
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
            "of your features reaches by chance; a feature below it separates "
            "the classes no better than chance. Each shuffle repeats the "
            "ranking.")
        self._null.toggled.connect(self._schedule)
        controls.addWidget(self._null)
        outer.addLayout(controls)

        self._blind = QLabel("", self)
        self._blind.setObjectName("ExplorerBlindSpot")
        self._blind.setWordWrap(True)
        outer.addWidget(self._blind)

        body = CollapsibleSplitter(Qt.Horizontal, self,
                                   persist_key="feature_explorer::panel")
        self.table = QTableWidget()
        install_sorting(self.table)
        self.table.setObjectName("ExplorerTable")
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Feature", "Separation", "AUC", "KS", "higher in", "min n"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.currentCellChanged.connect(self._on_row_changed)
        mark_surface(self.table)
        body.add_section(self.table, "Feature ranking",
                         persist_key="feature_explorer/Feature ranking")

        self._figure_holder = QWidget()
        holder = QVBoxLayout(self._figure_holder)
        holder.setContentsMargins(0, 0, 0, 0)
        from matplotlib.figure import Figure
        self._figure = Figure(figsize=(5.0, 6.0))
        self._canvas = _canvas_class()(self._figure)
        self._plot_scroll = QScrollArea(self._figure_holder)
        self._plot_scroll.setWidgetResizable(True)
        self._plot_scroll.setFrameShape(QFrame.NoFrame)
        self._plot_scroll.setWidget(self._canvas)
        holder.addWidget(self._plot_scroll, 1)
        body.add_section(self._figure_holder, "Distributions",
                         persist_key="feature_explorer/Distributions")
        outer.addWidget(body, 1)
        self._body_splitter = body

        self._summary = QLabel("no table loaded", self)
        self._summary.setObjectName("ExplorerSummary")
        self._summary.setWordWrap(True)
        outer.addWidget(self._summary)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self.rank_now)
        from ..screens.settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)

    def set_frame(self, frame: Optional[pd.DataFrame]) -> None:
        """Point the panel at a table and offer its class columns.

        :param frame: the object table, or ``None`` to empty the class picker.
            Its :func:`~spacr.qt.widgets.feature_rank.candidate_labels` are
            offered, keeping the previous choice when it is still there, and
            the ranking reruns.
        """
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
        """What the panel is currently set to rank.

        :returns: the explorer spec.
        """
        return self._spec

    @property
    def result(self) -> Optional[ExplorerResult]:
        """The last ranking computed, if any.

        :returns: the result, or None before a run.
        """
        return self._result

    def set_spec(self, spec: ExplorerSpec) -> None:
        """Push a whole spec in — restoring a saved analysis.

        :param spec: the :class:`~spacr.qt.widgets.feature_rank.ExplorerSpec`
            to adopt; its label, statistic, top count and whether a
            permutation null is on are shown in the controls, and the ranking
            reruns.
        """
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
        """The spec as the controls currently read.

        Distinct from :attr:`spec`, which is what the last run used: the two
        differ while the user is changing the controls, and a panel that
        conflated them would report a ranking against settings it did not
        use.

        :returns: the spec the controls describe.
        """
        return ExplorerSpec(
            label=self._label.currentText(),
            features=self._spec.features,
            statistic=self._statistic.currentData() or AUC,
            top=int(self._top.value()),
            bins=self._spec.bins,
            n_permutations=50 if self._null.isChecked() else 0,
            seed=self._spec.seed)

    def summary(self) -> str:
        """What was ranked, over how much, and what was skipped.

        :returns: a one-line summary, empty before a run.
        """
        return self._summary.text()

    def _schedule(self, *_args) -> None:
        """Queue a re-rank after a control changed.

        Debounced, so nudging the top-N spinner ranks once rather than per step.

        :param _args: whatever the emitting control passes; ignored.
        """
        self._debounce.start()

    def rank_now(self) -> Optional[ExplorerResult]:
        """Re-rank and redraw. Returns the result, or ``None`` on a refusal."""
        self._debounce.stop()
        statistic = self._statistic.currentData() or AUC
        self._blind.setText(
            f"{STATISTIC_LABELS[statistic]} — cannot see: "
            f"{STATISTIC_FAILURE_MODES[statistic]}")
        if self._frame is None:
            self._clear_result()
            self._summary.setText("no table loaded")
            return None
        self._spec = self.current_spec()
        try:
            result = rank_features(self._frame, self._spec)
        except ExplorerError as exc:
            self._clear_result()
            self._summary.setText(str(exc))
            return None
        self._result = result
        self._fill_table(result)
        self._draw(result)
        self._summary.setText(result.summary())
        self.ranked.emit(result)
        return result

    def _clear_result(self) -> None:
        """Forget every visual and selected part of the previous ranking."""
        self._result = None
        self.table.clearSelection()
        _retire_table_cells(self.table)
        self._figure.clear()
        self._figure.patch.set_alpha(0.0)
        self._canvas.draw_idle()

    def _fill_table(self, result: ExplorerResult) -> None:
        """Fill the ranking table and select its first row.

        Two kinds of row are marked rather than dropped: one whose difference is
        a change of shape rather than of location, and one scoring at or below
        the shuffle threshold -- a feature no better than chance is worth
        seeing, greyed, rather than silently omitted.

        Signals are blocked while filling, and the selection change is announced
        afterwards only if it actually moved.

        :param result: the computed ranking.
        """
        previous_feature = self.selected_feature()
        palette = active_palette()
        signals_were_blocked = self.table.blockSignals(True)
        try:
            _retire_table_cells(self.table)
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
                    item = _spare_table_item(self.table, text)
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
        finally:
            self.table.blockSignals(signals_were_blocked)

        selected_feature = self.selected_feature()
        if (not signals_were_blocked and selected_feature
                and selected_feature != previous_feature):
            self.feature_selected.emit(selected_feature)

    def _draw(self, result: ExplorerResult) -> None:
        """Present the original per-class counts and ranking through the shared figure owner.

        The ranking engine, feature order, shared-bin calculation, count
        arrays, bar centres and widths are unchanged. Captions retain each
        original score, higher class, minimum count and shape warning.
        A retained physical page scrolls when its complete panels exceed the
        viewport; clearing the plot releases that automatic minimum size.
        Explicit text choices are measured before packing the panel envelopes.
        """
        from ...figures.style import (
            DATA_ALPHA, SERIES, TYPE_SCALE, _apply_user_style,
            _retain_live_axes_geometry, figure_style, pack_axes_geometry,
        )
        from ...figures.bundle import _register_figure_data

        self._figure.clear()
        self._figure._spacr_live_geometry = None
        self._canvas.setMinimumSize(0, 0)
        drawn = result.scores[:MAX_DRAWN]
        if not drawn or self._frame is None:
            self._canvas.draw_idle()
            return
        viewport = tuple(self._figure.get_size_inches())
        with figure_style("screen", kind="histogram"):
            axes = _render_feature_histograms(self._figure, self._frame, result, self._spec.bins,
                colours=SERIES, alpha=DATA_ALPHA, font_points=TYPE_SCALE["legend"])
            self._figure.patch.set_alpha(1.0)
            if axes:
                _apply_user_style(self._figure, kind="histogram", force=True)
                pack_axes_geometry(axes, legend_space=True)
                required = tuple(self._figure.get_size_inches())
                _retain_live_axes_geometry(self._figure, viewport)
                logical_dpi = self._figure.dpi / self._canvas.device_pixel_ratio
                self._canvas.setMinimumSize(int(np.ceil(required[0] * logical_dpi)),
                                            int(np.ceil(required[1] * logical_dpi)))
        captured = [dict(feature=score.feature, statistic=score.statistic, score=float(score.score),
                        higher_in=score.higher_in, smallest_class=int(score.smallest_class),
                        is_shape_not_shift=bool(score.is_shape_not_shift)) for score in drawn]
        recipe = dict(schema=1, label=result.label, classes=list(result.classes), features=captured, bins=int(self._spec.bins),
                      colours=list(SERIES), alpha=DATA_ALPHA, font_points=TYPE_SCALE["legend"])
        _register_figure_data(self._figure, self._frame, x=result.label,
            y=drawn[0].feature, kind="hist", feature_explorer_recipe=recipe)
        self._canvas.draw_idle()

    def _on_row_changed(self, row: int, *_args) -> None:
        """Announce the feature on the newly selected row.

        :param row: the new row.
        :param _args: the remaining cell-change arguments; unused, since the row
            is what identifies the feature.
        """
        item = self.table.item(row, 0)
        if item is not None:
            self.feature_selected.emit(item.data(Qt.UserRole))

    def selected_feature(self) -> str:
        """Which feature the user has selected, if any.

        :returns: the feature name, or ``""``.
        """
        item = self.table.item(self.table.currentRow(), 0)
        return item.data(Qt.UserRole) if item is not None else ""

    def closeEvent(self, event):  # noqa: N802 - Qt name
        """Stop background work before going away.

        :param event: the Qt close event.
        """
        self._debounce.stop()
        if hasattr(self._canvas, "cancel_pending_draw"):
            self._canvas.cancel_pending_draw()
        super().closeEvent(event)


def _brush(colour: str):
    """Build a ``QBrush`` from a colour string.

    Imported inside the call so this module stays importable headlessly.

    :param colour: the colour.
    :returns: the brush.
    """
    from PySide6.QtGui import QBrush, QColor
    return QBrush(QColor(colour))
