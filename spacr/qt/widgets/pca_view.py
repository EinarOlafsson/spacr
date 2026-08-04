"""The PCA surface: a scree plot, a scores plot, and a loadings biplot.

The statistics are in :mod:`spacr.qt.widgets.pca_model` and there is none in
here. This module is the three pictures and the controls, and its one real
design decision is that **the scores plot is a Graph Builder**.

Why the scores plot is a GraphCanvas
------------------------------------
A PCA scores plot is a scatter of two continuous columns that a user wants to
colour by ``gene``, facet by ``plateID``, and brush to pull a cluster into
Annotate. That is the Graph Builder, exactly, with the two columns computed
rather than measured. So :meth:`PCAResult.scores_frame
<spacr.qt.widgets.pca_model.PCAResult.scores_frame>` appends ``PC1…PCk`` to the
source frame and :class:`PCAScoresCanvas` is a
:class:`~spacr.qt.widgets.graph_builder.GraphCanvas` subclass that draws arrows
on top.

Everything that took work in the Graph Builder therefore already works here and
cannot drift out of step with it: the shared-scale facet grid, the fixed
categorical colour order, the large-data policy and its notice, the
selection-dims-never-hides rule, and — the point of the exercise — the brush.
Because the scores frame carries the object key columns, a rectangle dragged
around a cluster in PC space publishes a real
:class:`~spacr.selection.Selection`, and the UMAP, the plate map and the crop
grid highlight the same cells. Reimplementing a scatter here would have meant
reimplementing all of that and getting the brush subtly wrong.

The biplot, and what its arrows do and do not mean
--------------------------------------------------
An arrow is the feature's **Pearson correlation with the two plotted
components** — :attr:`PCAResult.correlations
<spacr.qt.widgets.pca_model.PCAResult.correlations>`, a quantity with a meaning
of its own, rather than a unit-norm loading whose scale is an artefact of the
normalisation. So:

* **direction** is where more of that feature lies, and is exact;
* **relative length** is how much of the feature this plane shows: a feature
  perfectly captured by the plane reaches the dashed unit circle, one pointing
  out of the plane is short. Short means "not visible here", never
  "unimportant";
* **absolute length in data units is meaningless.** Every arrow is multiplied
  by one shared constant chosen to fill the panel, because scores and
  correlations have no common unit. The circle is scaled by the same constant,
  which is what keeps the comparison honest — it is the ruler.

Only the :data:`DEFAULT_ARROWS` best-represented features are drawn. Four
hundred arrows is an ink blot, and the ones left out are the ones pointing
somewhere the reader is not looking.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QPushButton, QSizePolicy,
    QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from ..job_runner import JobRunner
from ..theme import (RADIUS, SPACING, active_palette, font_px,
                     register_widget_qss)
from .graph_spec import SCATTER, GraphSpec
# `_canvas_class` is the owned-timer FigureCanvas fix — a matplotlib canvas
# whose deferred draw cannot fire after Qt has deleted it, which is a segfault
# on close. Imported rather than copied: two copies of a crash fix is one copy
# too many, and the scree plot needs the same protection the scores plot has.
from .graph_builder import (GraphCanvas, _canvas_class, _page_surface_axes,
                            categorical_colours)
from .pca_model import (
    DEFAULT_COMPONENTS, NAN_AUTO, NAN_COMPLETE, NAN_DROP_FEATURES, NAN_MEAN,
    NAN_POLICIES, SCALE_MODES, SCALE_NONE, SCALE_ZSCORE, PCAError, PCAResult,
    PCASpec, candidate_features, component_index, component_name, pca,
)

LOG = logging.getLogger("spacr.qt.pca")

__all__ = [
    "DEFAULT_ARROWS", "ARROW_FILL", "SCREE_COMPONENTS",
    "FeaturePicker", "ScreePlot", "PCAScoresCanvas", "PCAPanel",
    "arrow_scale",
]

#: Feature arrows drawn on the biplot by default. Past about a dozen the
#: labels overlap and the plot stops answering the question it exists for.
DEFAULT_ARROWS = 8

#: How much of the panel's shorter half-axis the longest arrow spans. Under 1
#: so an arrow tip never lands on the frame, where it cannot be read.
ARROW_FILL = 0.82

#: Components the scree plot draws. A scree plot is read for its elbow, which
#: is always in the first few.
SCREE_COMPONENTS = 12

#: NaN policy captions — the panel is where a user meets the decision, so the
#: consequence is in the words rather than in a docstring they will not open.
_NAN_LABELS = {
    NAN_AUTO: "Auto — drop structurally-missing features, then odd rows",
    NAN_COMPLETE: "Complete cases — keep features, drop rows with any NaN",
    NAN_DROP_FEATURES: "Drop features — keep every object",
    NAN_MEAN: "Mean-impute (fabricates values; say so in the figure)",
}

_SCALE_LABELS = {
    SCALE_ZSCORE: "Standardise (unit variance per feature)",
    SCALE_NONE: "Centre only (raw units — px² will win)",
}


def arrow_scale(result: PCAResult, kx: int, ky: int,
                x_limits: Tuple[float, float],
                y_limits: Tuple[float, float],
                *, count: int = DEFAULT_ARROWS,
                fill: float = ARROW_FILL) -> float:
    """The one constant every arrow and the unit circle are multiplied by.

    Scores are in standardised-feature units and correlations are in ``[-1,
    1]``; there is no conversion between them, so a biplot has to pick a
    display scale. It is picked once, from the drawn axes, so that the longest
    arrow spans ``fill`` of the shorter half-range — and applied to the circle
    too, so the reader has a ruler on screen rather than an assurance.

    Returns 0.0 when there is nothing to scale (no finite limits, or every
    correlation zero), which callers read as "draw no arrows".
    """
    if not result.n_components:
        return 0.0
    picked = result.plane_features(kx, ky, count)
    if not picked:
        return 0.0
    lengths = np.hypot(result.correlations[list(picked), kx],
                       result.correlations[list(picked), ky])
    longest = float(np.nanmax(lengths)) if lengths.size else 0.0
    if not np.isfinite(longest) or longest <= 0:
        return 0.0
    half_x = (x_limits[1] - x_limits[0]) / 2.0
    half_y = (y_limits[1] - y_limits[0]) / 2.0
    half = min(abs(half_x), abs(half_y))
    if not np.isfinite(half) or half <= 0:
        return 0.0
    return float(half * fill / longest)


# ---------------------------------------------------------------------------
# Picking features
# ---------------------------------------------------------------------------

class FeaturePicker(QWidget):
    """Which columns go into the decomposition.

    A tick list rather than drop zones: PCA takes tens to hundreds of features
    at once, and dragging four hundred columns one at a time is not a gesture
    anybody makes twice. The default ticks every continuous column, which is
    what :func:`~spacr.qt.widgets.pca_model.candidate_features` offers and what
    a user exploring a new table wants first.
    """

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PCAFeaturePicker")
        self._all: Tuple[str, ...] = ()
        # Per instance, not per class: a set on the class would be shared by
        # every picker in the process, so opening a second PCA screen would
        # silently retick the first one's features.
        self._checked: set = set()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACING["xs"])

        self._search = QLineEdit(self)
        self._search.setObjectName("PCAFeatureSearch")
        self._search.setPlaceholderText("Find a feature…")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._refilter)
        outer.addWidget(self._search)

        self._list = QListWidget(self)
        self._list.setObjectName("PCAFeatureList")
        self._list.setSelectionMode(QAbstractItemView.NoSelection)
        self._list.itemChanged.connect(self._on_item_changed)
        outer.addWidget(self._list, 1)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(SPACING["xs"])
        for text, tip, slot in (
                ("All", "Tick every feature shown", self.select_all),
                ("None", "Untick every feature shown", self.select_none),
                ("Invert", "Swap ticked for unticked", self.invert)):
            button = QPushButton(text, self)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        outer.addLayout(buttons)

        self._count = QLabel("no table loaded", self)
        self._count.setObjectName("PCAFeatureCount")
        outer.addWidget(self._count)

    # -- data -----------------------------------------------------------
    def set_frame(self, frame: Optional[pd.DataFrame]) -> None:
        """Offer ``frame``'s continuous columns, all ticked."""
        self._all = () if frame is None else candidate_features(frame)
        self._checked = set(self._all)
        self._refilter()

    def set_selected(self, features) -> None:
        self._checked = {f for f in features if f in self._all}
        self._refilter()

    def selected(self) -> Tuple[str, ...]:
        """The ticked features, in the offered order (not the click order)."""
        return tuple(f for f in self._all if f in self._checked)

    def available(self) -> Tuple[str, ...]:
        return self._all

    # -- buttons ---------------------------------------------------------
    def select_all(self) -> None:
        self._checked |= set(self._visible())
        self._refilter()

    def select_none(self) -> None:
        self._checked -= set(self._visible())
        self._refilter()

    def invert(self) -> None:
        for name in self._visible():
            self._checked ^= {name}
        self._refilter()

    # -- internals -------------------------------------------------------
    def _visible(self) -> List[str]:
        needle = self._search.text().strip().lower()
        return [n for n in self._all if not needle or needle in n.lower()]

    def _refilter(self) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        for name in self._visible():
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in self._checked
                               else Qt.Unchecked)
            item.setData(Qt.UserRole, name)
            self._list.addItem(item)
        self._list.blockSignals(False)
        total = len(self._all)
        self._count.setText(
            "no table loaded" if not total
            else f"{len(self._checked)} of {total} features ticked")
        self.changed.emit()

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        name = item.data(Qt.UserRole)
        if item.checkState() == Qt.Checked:
            self._checked.add(name)
        else:
            self._checked.discard(name)
        total = len(self._all)
        self._count.setText(f"{len(self._checked)} of {total} features ticked")
        self.changed.emit()


# ---------------------------------------------------------------------------
# Scree
# ---------------------------------------------------------------------------

class ScreePlot(QWidget):
    """Explained variance per component, with the cumulative line over it.

    Both, always. The bars are what people look at for the elbow; the
    cumulative line is what stops "PC1 and PC2 look big" from being mistaken
    for "PC1 and PC2 are most of it" when they are 9% and 7%.

    Clicking a bar emits :attr:`component_picked`, so the scree plot is the
    control that chooses what the scores plot draws rather than a decoration
    beside it.
    """

    component_picked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PCAScreePlot")
        self._result: Optional[PCAResult] = None
        self._highlight: Tuple[int, int] = (0, 1)

        from matplotlib.figure import Figure
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        # No `facecolor` and no inline `background:` — the canvas paints the
        # page panel in its own `paintEvent` under a transparent figure patch.
        self._figure = Figure(figsize=(3.4, 2.4))
        # `panel=False`: the scree plot sits inside `PCAShelf`, which is
        # already a page surface, and a second panel under the figure would
        # read 0.49 at a requested 30 % -- a shade the slider cannot reach.
        self._canvas = _canvas_class()(self._figure, panel=False)
        self._canvas.setMinimumHeight(150)
        outer.addWidget(self._canvas, 1)
        self._canvas.mpl_connect("button_press_event", self._on_click)

    def set_result(self, result: Optional[PCAResult], *,
                   highlight: Tuple[int, int] = (0, 1)) -> None:
        self._result = result
        self._highlight = highlight
        self.render_now()

    def render_now(self) -> None:
        palette = active_palette()
        self._figure.clear()
        # `clear()` restores the rc facecolor and its alpha with it.
        self._figure.patch.set_alpha(0.0)
        ax = self._figure.add_subplot(111)
        _page_surface_axes(ax, palette)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(palette["border"])
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=palette["fg_muted"], labelsize=7, length=3)

        result = self._result
        if result is None or not result.n_components:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, "no components yet", ha="center", va="center",
                    color=palette["fg_muted"], fontsize=9,
                    transform=ax.transAxes)
            self._canvas.draw_idle()
            return

        k = min(result.n_components, SCREE_COMPONENTS)
        ratio = np.asarray(result.explained_variance_ratio[:k], dtype=float)
        series = categorical_colours()
        chosen = set(self._highlight)
        colours = [series[0] if i in chosen else palette["border"]
                   for i in range(k)]
        ax.bar(range(k), ratio * 100.0, width=0.7, color=colours,
               linewidth=0.0)
        ax.set_ylabel("% of variance", color=palette["fg_dim"], fontsize=8)
        ax.set_xticks(range(k))
        ax.set_xticklabels([component_name(i) for i in range(k)],
                           rotation=45, ha="right", fontsize=7)
        ax.grid(True, axis="y", color=palette["border_soft"], linewidth=0.6,
                alpha=0.5)
        ax.set_axisbelow(True)

        twin = ax.twinx()
        twin.plot(range(k), np.cumsum(ratio[:k]) * 100.0, color=series[1],
                  marker="o", markersize=3, linewidth=1.4)
        twin.set_ylim(0, 105)
        twin.set_ylabel("cumulative %", color=series[1], fontsize=8)
        twin.tick_params(colors=palette["fg_muted"], labelsize=7, length=3)
        for side in ("top", "left"):
            twin.spines[side].set_visible(False)
        twin.spines["right"].set_color(palette["border"])

        self._figure.tight_layout(pad=0.5)
        self._canvas.draw_idle()

    def _on_click(self, event) -> None:
        if self._result is None or event.xdata is None:
            return
        index = int(round(float(event.xdata)))
        if 0 <= index < self._result.n_components:
            self.component_picked.emit(index)


# ---------------------------------------------------------------------------
# Scores, with arrows
# ---------------------------------------------------------------------------

class PCAScoresCanvas(GraphCanvas):
    """A :class:`~spacr.qt.widgets.graph_builder.GraphCanvas` that also draws
    the loadings.

    Everything the base class does is untouched — the facet grid, the shared
    scales, the large-data policy, the brush and the linked selection. The
    subclass adds one thing after each render: the feature arrows for whatever
    pair of components happens to be on x and y, read from the spec rather than
    configured separately, so the arrows cannot end up describing a different
    plane from the points.
    """

    def __init__(self, parent=None, *, link=None, source: str = "pca"):
        # Before super().__init__: the base constructor wires a debounce timer
        # to self.render_now, which is this class's override and reads these.
        self._result: Optional[PCAResult] = None
        self._biplot = True
        self._arrow_count = DEFAULT_ARROWS
        self._arrow_scale = 0.0
        super().__init__(parent, link=link, source=source)

    # -- inputs ----------------------------------------------------------
    def set_result(self, result: Optional[PCAResult],
                   frame: Optional[pd.DataFrame]) -> None:
        """Point the canvas at a decomposition and its scores frame.

        Both together, always: a result and a frame that do not match would
        put arrows from one PCA over the points of another, which is a picture
        that looks entirely reasonable and is wrong.
        """
        self._result = result
        self.set_frame(frame)

    @property
    def result(self) -> Optional[PCAResult]:
        return self._result

    def set_biplot(self, on: bool, *, count: Optional[int] = None,
                   render: bool = True) -> None:
        """Turn the arrows on or off, and optionally set how many.

        ``render=False`` is for a caller about to change the spec anyway: the
        spec change redraws, and doing it twice for one user action is a
        visible flicker on a large scatter.
        """
        self._biplot = bool(on)
        if count is not None:
            self._arrow_count = max(0, int(count))
        if render:
            self.render_now()

    @property
    def arrow_scale(self) -> float:
        """The display constant the last render used; 0.0 when no arrows were
        drawn. Public so a test — or a caption — can state the ruler."""
        return self._arrow_scale

    def plane(self) -> Optional[Tuple[int, int]]:
        """``(kx, ky)`` when both axes carry a component, else ``None``.

        Read from the spec, so dragging ``PC3`` onto Y moves the arrows with
        the points and dragging ``area`` onto Y removes them — a biplot of a
        component against a raw measurement is not a biplot.
        """
        kx = component_index(self._spec.x or "")
        ky = component_index(self._spec.y or "")
        if kx is None or ky is None or kx == ky:
            return None
        result = self._result
        if result is None:
            return None
        if kx >= result.n_components or ky >= result.n_components:
            return None
        return kx, ky

    # -- rendering -------------------------------------------------------
    def render_now(self) -> None:
        super().render_now()
        self._arrow_scale = 0.0
        try:
            self._draw_arrows()
        except Exception:  # pragma: no cover - a decoration must never
            # take the chart down with it.
            LOG.debug("could not draw the loading arrows", exc_info=True)

    def _draw_arrows(self) -> None:
        if not self._biplot or not self._arrow_count:
            return
        plane = self.plane()
        if plane is None or self._spec.resolved_kind(self._kinds) != SCATTER:
            return
        result = self._result
        kx, ky = plane
        palette = active_palette()
        axes = self.panel_axes()
        if not axes:
            return

        # One scale for every panel, from the first one: faceted panels share
        # their axes, and an arrow that changed length between panels would
        # make two panels of the same PCA look like two different PCAs.
        first = axes[min(axes)]
        scale = arrow_scale(result, kx, ky, first.get_xlim(), first.get_ylim(),
                            count=self._arrow_count)
        if scale <= 0:
            return
        self._arrow_scale = scale
        picked = result.plane_features(kx, ky, self._arrow_count)
        ink = palette["fg"]

        from matplotlib.patches import Circle
        for ax in axes.values():
            ax.add_patch(Circle(
                (0.0, 0.0), scale, fill=False, linestyle=(0, (3, 3)),
                edgecolor=palette["border"], linewidth=0.8, zorder=4))
            ax.axhline(0.0, color=palette["border_soft"], linewidth=0.7,
                       zorder=3)
            ax.axvline(0.0, color=palette["border_soft"], linewidth=0.7,
                       zorder=3)
            for i in picked:
                dx = float(result.correlations[i, kx]) * scale
                dy = float(result.correlations[i, ky]) * scale
                if not (np.isfinite(dx) and np.isfinite(dy)):
                    continue  # pragma: no cover - correlations are clipped
                ax.annotate(
                    "", xy=(dx, dy), xytext=(0.0, 0.0), zorder=7,
                    arrowprops={"arrowstyle": "-|>", "color": ink,
                                "linewidth": 1.2, "alpha": 0.85,
                                "shrinkA": 0, "shrinkB": 0})
                ax.annotate(
                    result.features[i], xy=(dx, dy), zorder=8,
                    xytext=(3 if dx >= 0 else -3, 3 if dy >= 0 else -3),
                    textcoords="offset points", fontsize=7, color=ink,
                    ha="left" if dx >= 0 else "right",
                    va="bottom" if dy >= 0 else "top")
        self._canvas.draw_idle()


# ---------------------------------------------------------------------------
# The whole surface
# ---------------------------------------------------------------------------

class PCAPanel(QWidget):
    """Feature picker, options, scree, scores + biplot, and the report.

    :param link: a private
        :class:`~spacr.qt.linked_selection.LinkedSelection` for tests; ``None``
        joins the process-wide one, which is what makes brushing a cluster
        here highlight it everywhere else.
    :param threaded: run the decomposition on a worker thread. **Defaults to
        False**, and the default is the interesting part -- see
        :meth:`recompute`. ``PCAScreen`` passes its own ``threaded`` through,
        so the application gets the threaded panel and a panel built directly
        keeps returning its result from the call.
    """

    #: Emitted after every successful decomposition.
    computed = Signal(object)
    #: Emitted with the message when one is refused. The panel shows it too;
    #: the signal is for a host that wants it in a status bar.
    failed = Signal(str)

    def __init__(self, parent=None, *, link=None, source: str = "pca",
                 threaded: bool = False):
        super().__init__(parent)
        self.setObjectName("PCAPanel")
        self._frame: Optional[pd.DataFrame] = None
        self._result: Optional[PCAResult] = None
        self._scores: Optional[pd.DataFrame] = None
        self._building = False
        self._threaded = bool(threaded)
        self._jobs = JobRunner(self, threaded=self._threaded,
                               app_key="pca fit")

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACING["sm"])
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)

        shelf = QWidget(self)
        shelf.setObjectName("PCAShelf")
        shelf_layout = QVBoxLayout(shelf)
        shelf_layout.setContentsMargins(SPACING["sm"], SPACING["sm"],
                                        SPACING["sm"], SPACING["sm"])
        shelf_layout.setSpacing(SPACING["sm"])

        self.features = FeaturePicker(shelf)
        shelf_layout.addWidget(self.features, 1)

        options = QGridLayout()
        options.setContentsMargins(0, 0, 0, 0)
        options.setSpacing(SPACING["xs"])

        self._scaling = QComboBox(shelf)
        for mode in SCALE_MODES:
            self._scaling.addItem(_SCALE_LABELS[mode], mode)
        self._scaling.setToolTip(
            "spaCR features have different units — px², counts, ratios — so "
            "an unscaled PCA finds whichever column has the biggest numbers. "
            "Standardising makes the answer independent of the units.")
        options.addWidget(QLabel("Scale", shelf), 0, 0)
        options.addWidget(self._scaling, 0, 1)

        self._nan = QComboBox(shelf)
        for policy in NAN_POLICIES:
            self._nan.addItem(_NAN_LABELS[policy], policy)
        self._nan.setToolTip(
            "A NaN here is usually meaningful: a pathogen feature is NaN for "
            "a cell with no pathogen. Dropping the feature keeps every "
            "object; dropping the rows keeps every feature but changes which "
            "population you are looking at.")
        options.addWidget(QLabel("NaN", shelf), 1, 0)
        options.addWidget(self._nan, 1, 1)

        self._components = QSpinBox(shelf)
        self._components.setRange(2, 50)
        self._components.setValue(DEFAULT_COMPONENTS)
        self._components.setToolTip("How many components to compute. Capped "
                                    "at the rank of the data.")
        options.addWidget(QLabel("Components", shelf), 2, 0)
        options.addWidget(self._components, 2, 1)

        self._pc_x = QComboBox(shelf)
        self._pc_y = QComboBox(shelf)
        options.addWidget(QLabel("X / Y", shelf), 3, 0)
        pair = QHBoxLayout()
        pair.setContentsMargins(0, 0, 0, 0)
        pair.setSpacing(SPACING["xs"])
        pair.addWidget(self._pc_x)
        pair.addWidget(self._pc_y)
        options.addLayout(pair, 3, 1)

        self._colour = QComboBox(shelf)
        self._colour.setToolTip("Colour the scores by any column of the "
                                "table — the point of the plot is usually "
                                "whether a label separates.")
        options.addWidget(QLabel("Colour", shelf), 4, 0)
        options.addWidget(self._colour, 4, 1)

        self._biplot = QCheckBox("Loadings biplot", shelf)
        self._biplot.setChecked(True)
        self._biplot.setToolTip(
            "Arrows are each feature's correlation with the two components. "
            "Direction and relative length are meaningful; the absolute "
            "length is scaled to fit and the dashed circle is that ruler.")
        options.addWidget(self._biplot, 5, 0)
        self._arrows = QSpinBox(shelf)
        self._arrows.setRange(0, 40)
        self._arrows.setValue(DEFAULT_ARROWS)
        self._arrows.setToolTip("How many arrows — the features best "
                                "represented in this plane.")
        options.addWidget(self._arrows, 5, 1)
        shelf_layout.addLayout(options)

        self._run = QPushButton("Run PCA", shelf)
        self._run.setObjectName("PrimaryButton")
        self._run.clicked.connect(self.recompute)
        shelf_layout.addWidget(self._run)

        self.scree = ScreePlot(shelf)
        shelf_layout.addWidget(self.scree)
        splitter.addWidget(shelf)

        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(SPACING["xs"])
        self.canvas = PCAScoresCanvas(right, link=link, source=source)
        right_layout.addWidget(self.canvas, 1)
        self.report = QLabel("", right)
        self.report.setObjectName("PCAReport")
        self.report.setWordWrap(True)
        self.report.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.report.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        right_layout.addWidget(self.report)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 900])

        for widget in (self._scaling, self._nan):
            widget.currentIndexChanged.connect(self._on_option_changed)
        self._components.valueChanged.connect(self._on_option_changed)
        for widget in (self._pc_x, self._pc_y, self._colour):
            widget.currentIndexChanged.connect(self._on_view_changed)
        self._biplot.toggled.connect(self._on_view_changed)
        self._arrows.valueChanged.connect(self._on_view_changed)
        self.scree.component_picked.connect(self._on_scree_clicked)

    # -- data -------------------------------------------------------------
    def set_frame(self, frame: Optional[pd.DataFrame], *,
                  compute: bool = True) -> None:
        """Point the panel at a table and (by default) decompose it."""
        self._frame = frame
        self.features.set_frame(frame)
        self._building = True
        try:
            self._colour.clear()
            self._colour.addItem("none", "")
            if frame is not None:
                for name in sorted(frame.columns):
                    self._colour.addItem(name, name)
        finally:
            self._building = False
        if compute:
            self.recompute()

    @property
    def result(self) -> Optional[PCAResult]:
        return self._result

    @property
    def scores_frame(self) -> Optional[pd.DataFrame]:
        """The frame the scores plot is drawn from — every original column
        plus ``PC1…PCk``. What "export the scores" writes."""
        return self._scores

    def spec(self) -> PCASpec:
        """The spec the controls currently describe."""
        return PCASpec(
            features=self.features.selected(),
            n_components=self._components.value(),
            scaling=self._scaling.currentData() or SCALE_ZSCORE,
            nan_policy=self._nan.currentData() or NAN_AUTO)

    # -- computing ---------------------------------------------------------
    def recompute(self) -> Optional[PCAResult]:
        """Decompose and redraw. Refusals become a message, never a traceback.

        THE CONTRACT, and how it was resolved. This method returned its
        ``PCAResult`` synchronously, and the sklearn fit behind it is **1.63 s
        on a 200 000-row x 48-column table** -- measured, and recorded as
        ``PCA_STALL_BUDGET_S`` in ``tests/qt/test_gui_responsiveness.py``
        rather than hidden, because that is a 1.63 s frozen window on every
        option change and every filter change.

        The fit now runs on a worker, and this returns ``None`` while it does.
        Rather than pretend that is the same thing, the panel takes a
        ``threaded`` flag:

        * ``threaded=False`` (the default, and what a directly-constructed
          panel gets) runs the fit inline through :class:`JobRunner`'s
          unthreaded path and returns the ``PCAResult`` exactly as before.
        * ``threaded=True`` -- what ``PCAScreen`` passes, so the application
          gets it -- dispatches and returns ``None``. The result arrives at
          :meth:`_on_computed`, which is where the drawing was already done,
          and ``computed`` is emitted from there as it always was.

        A host that wants the result asynchronously has always had
        ``computed``; nothing that listens to it changed. A caller that reads
        the return value gets the old behaviour by not asking for threading,
        which is honest about the fact that a value cannot be returned before
        it has been computed.

        :returns: the decomposition, or ``None`` when it was refused or is
            still running.
        """
        if self._frame is None or self._frame.empty:
            self._show_failure("Load a table first.")
            return None
        frame = self._frame
        spec = self.spec()
        # A superseded fit must not paint over the one the user asked for:
        # dragging the components spin box starts one per value.
        self._jobs.cancel()

        def _fit():
            try:
                result = pca(frame, spec)
            except PCAError as exc:
                return {"error": str(exc)}
            except Exception as exc:  # pragma: no cover - defensive
                LOG.info("PCA failed", exc_info=True)
                return {"error": f"PCA failed: {exc}"}
            # `scores_frame` is another pass over the table; it belongs on
            # this side of the boundary with the fit, not on the GUI thread.
            return {"result": result, "scores": result.scores_frame(frame)}

        self._jobs.submit(_fit, self._on_fit_done)
        # Unthreaded, `submit` has already run the fit and `_on_fit_done`, so
        # `_result` is this call's result. Threaded, it is whatever was on
        # screen before, and returning that would be a lie about which spec it
        # came from.
        return None if self._threaded else self._result

    def _on_fit_done(self, outcome: dict) -> None:
        """Draw one decomposition. GUI thread only."""
        error = outcome.get("error")
        if error is not None:
            self._show_failure(error)
            return
        result = outcome["result"]
        self._result = result
        self._scores = outcome["scores"]
        self._sync_component_pickers(result)
        self.scree.set_result(result, highlight=self._plane())
        self.canvas.set_result(result, self._scores)
        self._apply_view()
        self.report.setText(result.report())
        self.computed.emit(result)

    def active_jobs(self) -> int:
        """How many decompositions are still winding down."""
        return self._jobs.active_jobs()

    def is_busy(self) -> bool:
        """True while a decomposition has not delivered its result."""
        return self._jobs.is_busy()

    def closeEvent(self, event):  # noqa: N802 - Qt name
        """Abandon an in-flight fit rather than let it outlive the panel."""
        self._jobs.shutdown()
        super().closeEvent(event)

    def _show_failure(self, message: str) -> None:
        self._result = None
        self._scores = None
        self.scree.set_result(None)
        self.canvas.set_result(None, None)
        self.report.setText(message)
        self.failed.emit(message)

    # -- the view ----------------------------------------------------------
    def _plane(self) -> Tuple[int, int]:
        """``(kx, ky)`` from the two pickers, defaulting to PC1 against PC2."""
        kx = component_index(self._pc_x.currentData() or "")
        ky = component_index(self._pc_y.currentData() or "")
        return (0 if kx is None else kx), (1 if ky is None else ky)

    def _sync_component_pickers(self, result: PCAResult) -> None:
        """Refill both pickers, keeping the pair the user was looking at."""
        self._building = True
        try:
            for box, default in ((self._pc_x, 0), (self._pc_y, 1)):
                wanted = box.currentData()
                box.clear()
                for i in range(result.n_components):
                    box.addItem(
                        f"{component_name(i)} "
                        f"({result.explained_variance_ratio[i]:.1%})",
                        component_name(i))
                index = box.findData(wanted)
                box.setCurrentIndex(
                    index if index >= 0
                    else min(default, max(0, box.count() - 1)))
        finally:
            self._building = False

    def _apply_view(self) -> None:
        if self._result is None:
            return
        kx, ky = self._plane()
        last = self._result.n_components - 1
        kx, ky = min(kx, last), min(ky, last)
        colour = self._colour.currentData() or None
        spec = GraphSpec(x=component_name(kx), y=component_name(ky),
                         colour=colour, kind=SCATTER)
        # Arrows first with render off, then the spec — one redraw per action.
        self.canvas.set_biplot(self._biplot.isChecked(),
                               count=self._arrows.value(), render=False)
        self.canvas.set_spec(spec)
        self.scree.set_result(self._result, highlight=(kx, ky))

    def _on_option_changed(self, *_args) -> None:
        if self._building:
            return
        self.recompute()

    def _on_view_changed(self, *_args) -> None:
        if self._building:
            return
        self._apply_view()

    def _on_scree_clicked(self, index: int) -> None:
        """Clicking a bar puts that component on X, sliding the old X onto Y.

        Sliding rather than replacing X alone: the pair the user is comparing
        is what they clicked plus what they were already looking at, and
        landing on "PC3 against PC3" is not a plot.
        """
        if self._result is None:
            return
        kx, _ky = self._plane()
        if index == kx:
            return
        self._building = True
        try:
            new_x = self._pc_x.findData(component_name(index))
            new_y = self._pc_y.findData(component_name(kx))
            if new_x >= 0:
                self._pc_x.setCurrentIndex(new_x)
            if new_y >= 0:
                self._pc_y.setCurrentIndex(new_y)
        finally:
            self._building = False
        self._apply_view()

    def closeEvent(self, event):  # noqa: N802 - Qt name
        self.canvas.close()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Styling, through the seam
# ---------------------------------------------------------------------------

def _pca_qss(palette, opacity) -> str:
    from ..theme import pane_surface
    surface_alt = pane_surface("surface_alt", palette["theme"], opacity)
    return f"""
QWidget#PCAShelf {{
    background: {surface_alt};
    border-radius: {RADIUS["md"]}px;
}}
QListWidget#PCAFeatureList {{
    background: transparent;
    border: 1px solid {palette["border_soft"]};
    border-radius: {RADIUS["sm"]}px;
}}
QLabel#PCAFeatureCount {{
    color: {palette["fg_muted"]};
    font-size: {font_px(11)}px;
}}
QLabel#PCAReport {{
    color: {palette["fg_dim"]};
    font-size: {font_px(11)}px;
}}
"""


# Registered at import of this module, which happens when the screen module
# is imported — and the row that does that lives in ``app.py``'s
# ``_SELF_REGISTERING_APPS``, whose loop runs while ``app.py`` itself is being
# imported. That is before ``launch()`` calls ``stylesheet()``, which is the
# deadline: a block registered after the stylesheet is built is missing from
# the one the application was actually given. `spacr.qt.widgets.__init__`
# imports `graph_builder` eagerly for exactly this reason; this module needs no
# such entry only because its screen is imported earlier still.
register_widget_qss("PCA", _pca_qss, replace=True)
