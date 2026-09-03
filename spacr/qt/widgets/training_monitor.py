"""Incrementally display loss and accuracy during model training.

The widget retains one plot item per metric and updates its data as epochs
complete. This preserves the current view and avoids creating overlapping
plot items during long training runs.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget

LOG = logging.getLogger("spacr.qt.training_monitor")

#: Plot panels in display order as ``(key, title, y-axis label)``. Separate
#: aggregate and per-class accuracy panels expose class-specific performance
#: that may be obscured by an aggregate metric.
PANELS: Tuple[Tuple[str, str, str], ...] = (
    ("loss", "Loss", "loss"),
    ("accuracy", "Accuracy", "accuracy"),
    ("per_class", "Accuracy per class", "accuracy"),
)


class TrainingMonitor(QWidget):
    """Display training metrics as incrementally updated curves.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.

    Attributes
    ----------
    plots : dict of str to pyqtgraph.PlotWidget
        Plot widgets keyed by ``"loss"``, ``"accuracy"``, and
        ``"per_class"``.
    curves : dict of str to pyqtgraph.PlotDataItem
        Persistent plot items keyed by metric name. Each item is created when
        its metric first appears and reused for subsequent epochs.

    :param parent: parent widget.
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
        """Return the persistent plot item for a metric series."""
        if name in self.curves:
            return self.curves[name]
        colour = pg.intColor(len(self.curves), hues=9)
        curve = self.plots[panel].plot(
            [], [], pen=pg.mkPen(colour, width=2), name=str(name))
        self.curves[name] = curve
        self._points[name] = ([], [])
        return curve

    def append(self, epoch: float, values: Dict[str, float]) -> int:
        """Append finite metric values for one epoch.

        Parameters
        ----------
        epoch : float
            Epoch coordinate assigned to each accepted value.
        values : dict of str to float
            Metric values keyed by series name. Names containing ``loss``
            are placed on the loss panel; aggregate accuracy names are placed
            on the accuracy panel; other names are treated as per-class
            metrics. Non-numeric and non-finite values are ignored.

        Returns
        -------
        int
            Number of series updated.
        """
        touched = 0
        for name, value in (values or {}).items():
            try:
                y = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(y):
                # Omitting the point represents a non-finite epoch as a gap.
                continue
            panel = self._panel_for(str(name))
            curve = self._curve(panel, str(name))
            xs, ys = self._points[str(name)]
            xs.append(float(epoch))
            ys.append(y)
            # Updating the existing item preserves the view and legend.
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
        """Return metric names in the order they first appeared."""
        return tuple(self.curves)

    def points(self, name: str) -> Tuple[Tuple[float, ...], ...]:
        """Return epoch and value coordinates for a metric series.

        Parameters
        ----------
        name : str
            Metric series name.

        Returns
        -------
        tuple of tuple of float
            ``(epochs, values)``. Both tuples are empty when the series has
            not been observed.
        """
        xs, ys = self._points.get(str(name), ([], []))
        return tuple(xs), tuple(ys)

    def clear(self) -> None:
        """Remove all curves and stored points for a new training run."""
        for panel in self.plots.values():
            panel.clear()
            panel.addLegend()
        self.curves.clear()
        self._points.clear()
