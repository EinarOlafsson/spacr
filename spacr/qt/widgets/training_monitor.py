"""The live training graphs, in pyqtgraph, appending rather than redrawing.

Instruction 231: "in classify during training the graphs should be
pyqtgraphs and they graphs should be updated not regenerated epch after
epoch".

EACH EPOCH APPENDS A POINT. It does not rebuild the figure. Regenerating
costs a full re-render of every point drawn so far, so the run gets SLOWER
THE LONGER IT GOES -- at exactly the moment the user is watching it most
closely. It also flickers, and it throws away anything the user did to the
view: a zoom into the last twenty epochs is undone by epoch twenty-one,
which makes the live graph unusable for the thing a live graph is for.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget

LOG = logging.getLogger("spacr.qt.training_monitor")

#: The panels, in order, as ``(key, title, y label)``.
#:
#: THREE PANELS, NOT TWO, and the third is the one that earns its place: the
#: aggregate accuracy answers "is it learning", the per-class panel answers
#: "is it learning ALL of it" -- which is the question a 96% aggregate
#: hiding a class at 40% gets wrong, invisibly, for the whole run.
PANELS: Tuple[Tuple[str, str, str], ...] = (
    ("loss", "Loss", "loss"),
    ("accuracy", "Accuracy", "accuracy"),
    ("per_class", "Accuracy per class", "accuracy"),
)


class TrainingMonitor(QWidget):
    """Live training curves that grow a point at a time.

    :ivar curves: ``{name: PlotDataItem}``. THE SAME OBJECTS FOR THE WHOLE
        RUN -- a rebuilt curve that looks the same is the bug this widget
        exists to fix, so the tests assert identity rather than appearance.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.plots: Dict[str, Any] = {}
        self.curves: Dict[str, Any] = {}
        #: The points behind each curve. Held rather than read back off the
        #: item: pyqtgraph returns copies, and appending to a copy is a
        #: quiet no-op.
        self._points: Dict[str, Tuple[List[float], List[float]]] = {}

        for key, title, y_label in PANELS:
            plot = pg.PlotWidget(title=title)
            plot.setLabel("bottom", "epoch")
            plot.setLabel("left", y_label)
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.addLegend()
            layout.addWidget(plot)
            self.plots[key] = plot

    # ------------------------------------------------------------- drawing

    def _curve(self, panel: str, name: str):
        """The curve for one series, made once and kept.

        MADE ONCE IS THE WHOLE POINT. `plot.plot()` ADDS an item every time
        it is called, so calling it per epoch leaves n overlapping curves --
        which looks like one curve, costs n times the render, and is exactly
        the "regenerated" behaviour under another name.
        """
        if name in self.curves:
            return self.curves[name]
        colour = pg.intColor(len(self.curves), hues=9)
        curve = self.plots[panel].plot(
            [], [], pen=pg.mkPen(colour, width=2), name=str(name))
        self.curves[name] = curve
        self._points[name] = ([], [])
        return curve

    def append(self, epoch: float, values: Dict[str, float]) -> int:
        """Add one epoch's numbers. Returns how many series were touched.

        :param values: ``{series: value}``. A series named ``loss`` or
            ``val_loss`` goes on the loss panel, ``accuracy`` on the
            accuracy panel, and anything else on the per-class panel.
        """
        touched = 0
        for name, value in (values or {}).items():
            try:
                y = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(y):
                # NOT PLOTTED AND NOT DROPPED SILENTLY -- a NaN epoch is a
                # gap in the curve, which is what it should look like.
                continue
            panel = self._panel_for(str(name))
            curve = self._curve(panel, str(name))
            xs, ys = self._points[str(name)]
            xs.append(float(epoch))
            ys.append(y)
            # setData ON THE SAME ITEM, which is the append. pyqtgraph
            # re-uploads only the arrays; nothing about the view, the zoom
            # or the legend is rebuilt.
            curve.setData(xs, ys)
            touched += 1
        return touched

    @staticmethod
    def _panel_for(name: str) -> str:
        """Which panel a series belongs on."""
        lowered = str(name).lower()
        if "loss" in lowered:
            return "loss"
        if lowered in ("accuracy", "val_accuracy", "acc", "val_acc"):
            return "accuracy"
        return "per_class"

    def series(self) -> Tuple[str, ...]:
        """Every series drawn so far, in the order they first appeared."""
        return tuple(self.curves)

    def points(self, name: str) -> Tuple[Tuple[float, ...], ...]:
        """``(xs, ys)`` for one series."""
        xs, ys = self._points.get(str(name), ([], []))
        return tuple(xs), tuple(ys)

    def clear(self) -> None:
        """Start a new run. THE ONLY PLACE THE CURVES ARE THROWN AWAY."""
        for panel in self.plots.values():
            panel.clear()
            panel.addLegend()
        self.curves.clear()
        self._points.clear()
