"""Draw an ROI on the layer canvas, and hand it to Measure.

The two halves of item ``B14``. :mod:`spacr.roi` is the half that runs in a
measurement worker — the geometry, the keep/drop rule and the environment route
that gets both into a ``spawn`` pool. This is the half that needs a mouse: a
pen that turns clicks into a :class:`spacr.layers.Shape`, and a panel that
turns the shapes layer into a :class:`spacr.roi.RoiSet` and switches the
measurement filter on.

The pen draws into the model, not onto the widget
------------------------------------------------

Every vertex is a world coordinate taken from
:meth:`spacr.layers.Canvas.world_at`, and the polygon being drawn is a real
:class:`~spacr.layers.Shape` in a real :class:`~spacr.layers.ShapesLayer`. So
the half-finished ROI is rendered by the same compositor as everything else,
it survives a zoom, and the finished ROI is already in the form
:meth:`spacr.roi.RoiSet.from_shapes_layer` reads. Nothing about the drawing is
in pixels, which is what lets an ROI drawn on a downsampled preview name the
same region on a full-resolution mask.

The panel says whether it will actually work
--------------------------------------------

Switching the filter on is one call, and the thing worth putting on screen is
not that it succeeded but whether it will reach the processes that do the
measuring. A filter registered in the GUI process alone is a silent no-op in
every ``spawn`` worker: the run finishes, the numbers are for the whole field,
and nothing says so. :func:`spacr.roi.worker_delivery_status` answers that in
advance and the panel shows its answer, warning-coloured when it is "no".
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFileDialog,
                               QHBoxLayout, QLabel, QLineEdit, QVBoxLayout,
                               QWidget)

from ..layers import LayerError, LayerStack, Shape, ShapesLayer, Spacing
from ..roi import (ANY_FIELD, MODES, RoiError, RoiSet, disable_roi_filter,
                   enable_roi_filter, worker_delivery_status)
from .layer_viewer import CanvasTool, LayerCanvas
from .theme import register_widget_qss
from .widgets.preview_controls import FlatButton, FlatComboBox

LOG = logging.getLogger(__name__)

__all__ = [
    "RoiPen",
    "RoiPanel",
    "ROI_LAYER_NAME",
]

#: The name the ROI shapes layer is given in the stack, so the panel can find
#: the one it drew into rather than the first shapes layer it sees.
ROI_LAYER_NAME = "ROI"


def _roi_tool_qss(palette: Dict[str, Any], opacity) -> str:
    """This panel's QSS block, appended to every generated stylesheet."""
    return f"""
QWidget#RoiPanel {{
    background: transparent;
}}
QLabel#RoiStatus {{
    color: {palette["fg_muted"]};
}}
QLabel#RoiStatusWarning {{
    color: {palette["warning"]};
}}
"""


register_widget_qss("RoiPanel", _roi_tool_qss, replace=True)


# ---------------------------------------------------------------------------
# The pen
# ---------------------------------------------------------------------------

class RoiPen(CanvasTool, QObject):
    """Turns clicks on a :class:`~spacr.qt.layer_viewer.LayerCanvas` into a shape.

    Left click adds a vertex, double click (or Return) closes the polygon,
    Backspace takes the last vertex back and Escape abandons the whole thing.
    A rectangle and an ellipse take two clicks — opposite corners — and close
    themselves, because a third click on a two-corner shape has no meaning.

    While it is being drawn the ROI is a ``path`` shape in the layer, so the
    user sees the outline they have so far; when it closes, the path is
    replaced by a real closed shape. Both are model objects: there is no
    parallel "rubber band" drawing that could disagree with what is stored.

    :param layer: the :class:`~spacr.layers.ShapesLayer` to draw into.
    :param kind: ``'polygon'``, ``'rectangle'`` or ``'ellipse'``.
    """

    #: A shape was completed. Carries its index in the layer.
    roi_finished = Signal(int)
    #: The vertex list changed (added, undone, abandoned).
    roi_changed = Signal()

    cursor = Qt.CrossCursor

    def __init__(self, layer: ShapesLayer, *, kind: str = "polygon",
                 parent: Optional[QObject] = None):
        QObject.__init__(self, parent)
        if not isinstance(layer, ShapesLayer):
            raise LayerError(
                f"an ROI pen draws into a ShapesLayer, got {layer!r}")
        self._layer = layer
        self._kind = str(kind).strip().lower()
        if self._kind not in ("polygon", "rectangle", "ellipse"):
            raise LayerError(
                f"an ROI is a closed shape: 'polygon', 'rectangle' or "
                f"'ellipse', not {kind!r}")
        self._pending: List[List[float]] = []
        self._preview = False

    # -- state ------------------------------------------------------------
    @property
    def layer(self) -> ShapesLayer:
        """The layer this pen draws into."""
        return self._layer

    @property
    def kind(self) -> str:
        """The kind of shape being drawn."""
        return self._kind

    @property
    def pending(self) -> np.ndarray:
        """The vertices placed so far, ``(M, ndim)`` in DATA coordinates."""
        if not self._pending:
            return np.zeros((0, self._layer.ndim), dtype=np.float64)
        return np.asarray(self._pending, dtype=np.float64)

    # -- the tool protocol -------------------------------------------------
    def press(self, view: LayerCanvas, world: Dict[str, float],
              event: Any) -> bool:
        """Place a vertex (left button) or take the last one back (right)."""
        button = event.button() if hasattr(event, "button") else Qt.LeftButton
        if button == Qt.RightButton:
            self.undo()
            return True
        if button != Qt.LeftButton:
            return False
        self.add_world(world)
        return True

    def double_click(self, view: LayerCanvas, world: Dict[str, float],
                     event: Any) -> bool:
        """Close the polygon. The second click of the pair is not a vertex."""
        self.close_shape()
        return True

    def key(self, view: LayerCanvas, event: Any) -> bool:
        """Escape abandons, Return closes, Backspace undoes."""
        key = event.key()
        if key == Qt.Key_Escape:
            self.cancel()
            return True
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self.close_shape()
            return True
        if key in (Qt.Key_Backspace, Qt.Key_Delete):
            self.undo()
            return True
        return False

    def detach(self) -> None:
        """Taken off the canvas: drop anything half-drawn."""
        self.cancel()

    # -- editing ------------------------------------------------------------
    def add_world(self, world: Dict[str, float]) -> int:
        """Add one vertex given as ``{axis: world}``; returns the vertex count.

        A rectangle or an ellipse closes itself on the second vertex.
        """
        self._pending.append(list(self._layer.spacing.data_from_map(world)))
        if self._kind in ("rectangle", "ellipse") and len(self._pending) >= 2:
            self.close_shape()
            return 0
        self._refresh_preview()
        self.roi_changed.emit()
        return len(self._pending)

    def undo(self) -> int:
        """Remove the most recent vertex; returns how many are left."""
        if self._pending:
            self._pending.pop()
            self._refresh_preview()
            self.roi_changed.emit()
        return len(self._pending)

    def cancel(self) -> None:
        """Abandon the shape being drawn."""
        if not self._pending and not self._preview:
            return
        self._pending = []
        self._drop_preview()
        self.roi_changed.emit()

    def close_shape(self) -> int:
        """Finish the shape. Returns its index, or ``-1`` if there was not one.

        Too few vertices is not an error: a stray double click on an empty
        canvas should do nothing, not raise out of a mouse handler.
        """
        needed = 3 if self._kind == "polygon" else 2
        if len(self._pending) < needed:
            return -1
        vertices = np.asarray(self._pending, dtype=np.float64)
        self._pending = []
        self._drop_preview()
        index = self._layer.add(Shape(self._kind, vertices, name=""))
        self.roi_changed.emit()
        self.roi_finished.emit(index)
        return index

    # -- the half-drawn outline --------------------------------------------
    def _refresh_preview(self) -> None:
        self._drop_preview()
        if len(self._pending) < 2:
            return
        self._layer.add(Shape("path", np.asarray(self._pending,
                                                 dtype=np.float64),
                              name="drawing", edge_color="cyan"))
        self._preview = True

    def _drop_preview(self) -> None:
        if not self._preview:
            return
        self._preview = False
        if len(self._layer):
            self._layer.remove(len(self._layer) - 1)


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

class RoiPanel(QWidget):
    """Draw an ROI, then measure only inside it.

    :param canvas: the :class:`~spacr.qt.layer_viewer.LayerCanvas` to draw on.
        The panel adds its own shapes layer to that canvas's stack the first
        time drawing starts.
    :param roi_path: where the ROI is written. A worker reads it from there, so
        it has to be a real file; defaults to ``./roi/measure_roi.json``.
    """

    #: The ROI was switched on or off for Measure. Carries the new state.
    filter_changed = Signal(bool)

    def __init__(self, canvas: LayerCanvas, parent=None, *,
                 roi_path: Optional[str] = None):
        super().__init__(parent)
        self.setObjectName("RoiPanel")
        self._canvas = canvas
        self._pen: Optional[RoiPen] = None
        self._roi_path = (os.path.abspath(str(roi_path)) if roi_path
                          else os.path.join(os.getcwd(), "roi",
                                            "measure_roi.json"))
        self._enabled = False
        self._build()
        self._refresh_status()

    # -- construction -------------------------------------------------------
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.draw_button = FlatButton(
            "Draw ROI", self,
            tooltip="Click to place vertices, double-click to close, "
                    "Esc to abandon")
        self.draw_button.setCheckable(True)
        self.draw_button.toggled.connect(self._on_draw_toggled)
        row.addWidget(self.draw_button, 1)
        self.kind_combo = FlatComboBox(self)
        self.kind_combo.addItems(["polygon", "rectangle", "ellipse"])
        self.kind_combo.currentTextChanged.connect(self._on_kind_changed)
        row.addWidget(self.kind_combo, 1)
        self.clear_button = FlatButton("Clear", self,
                                       tooltip="Remove every drawn ROI")
        self.clear_button.clicked.connect(self.clear_rois)
        row.addWidget(self.clear_button)
        outer.addLayout(row)

        rule = QHBoxLayout()
        rule.setSpacing(4)
        self.mode_combo = FlatComboBox(self)
        self.mode_combo.addItems(list(MODES))
        self.mode_combo.setToolTip(
            "centroid: keep an object whose centre is inside.\n"
            "overlap: keep an object with at least this fraction of its "
            "pixels inside.")
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        rule.addWidget(self.mode_combo, 1)
        self.overlap_spin = QDoubleSpinBox(self)
        self.overlap_spin.setRange(0.01, 1.0)
        self.overlap_spin.setSingleStep(0.05)
        self.overlap_spin.setValue(0.5)
        self.overlap_spin.setEnabled(False)
        rule.addWidget(self.overlap_spin)
        self.invert_check = QCheckBox("Outside", self)
        self.invert_check.setToolTip(
            "Measure the objects OUTSIDE the ROI instead — 'exclude this "
            "debris' rather than 'measure this colony'.")
        rule.addWidget(self.invert_check)
        outer.addLayout(rule)

        scope = QHBoxLayout()
        scope.setSpacing(4)
        scope.addWidget(QLabel("Fields", self))
        self.field_edit = QLineEdit(ANY_FIELD, self)
        self.field_edit.setToolTip(
            "The field stems this ROI applies to, comma separated "
            "(plate1_A01_F001). '*' applies it to every field.")
        scope.addWidget(self.field_edit, 1)
        self.path_button = FlatButton(
            "File…", self, tooltip="Where the ROI is saved. A worker process "
                                   "can only reach it through the file system.")
        self.path_button.clicked.connect(self._on_choose_path)
        scope.addWidget(self.path_button)
        outer.addLayout(scope)

        buttons = QHBoxLayout()
        buttons.setSpacing(4)
        self.enable_button = FlatButton(
            "Measure inside ROI", self,
            tooltip="Register the ROI as a Measure region filter, here and in "
                    "every worker process")
        self.enable_button.clicked.connect(self.enable)
        buttons.addWidget(self.enable_button, 1)
        self.disable_button = FlatButton(
            "Measure whole fields", self,
            tooltip="Remove the region filter")
        self.disable_button.clicked.connect(self.disable)
        buttons.addWidget(self.disable_button, 1)
        outer.addLayout(buttons)

        self.status = QLabel("", self)
        self.status.setObjectName("RoiStatus")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

    # -- the shapes layer ---------------------------------------------------
    @property
    def stack(self) -> LayerStack:
        """The stack the canvas is showing."""
        return self._canvas.stack

    @property
    def roi_path(self) -> str:
        """Where the ROI is written for the workers to read."""
        return self._roi_path

    def set_roi_path(self, path: str) -> str:
        """Choose where the ROI file is written; returns the absolute path."""
        self._roi_path = os.path.abspath(str(path))
        self._refresh_status()
        return self._roi_path

    def roi_layer(self, create: bool = True) -> Optional[ShapesLayer]:
        """The ROI shapes layer, adding it to the stack the first time.

        Given the spacing of whatever 2-D layer is already in the stack, so an
        ROI drawn over a µm-calibrated image is stored in µm and one drawn over
        a pixel-spaced field is stored in pixels — the units the measurement
        will be compared against.
        """
        for layer in self.stack:
            if isinstance(layer, ShapesLayer) and layer.name == ROI_LAYER_NAME:
                return layer
        if not create:
            return None
        spacing = None
        for layer in self.stack:
            if layer.ndim == 2:
                spacing = layer.spacing
                break
        if spacing is None:
            spacing = Spacing.isotropic(2, 1.0, units=self.stack.units)
        return self.stack.add_shapes(name=ROI_LAYER_NAME, ndim=2,
                                     spacing=spacing)

    def clear_rois(self) -> int:
        """Remove every drawn ROI; returns how many went."""
        layer = self.roi_layer(create=False)
        if layer is None:
            return 0
        if self._pen is not None:
            self._pen.cancel()
        removed = len(layer)
        while len(layer):
            layer.remove(len(layer) - 1)
        self._refresh_status()
        return removed

    # -- drawing ------------------------------------------------------------
    @property
    def pen(self) -> Optional[RoiPen]:
        """The pen while drawing is switched on, else ``None``."""
        return self._pen

    def start_drawing(self) -> RoiPen:
        """Attach a pen to the canvas and return it."""
        layer = self.roi_layer()
        self._pen = RoiPen(layer, kind=self.kind_combo.currentText(),
                           parent=self)
        self._pen.roi_finished.connect(self._on_roi_finished)
        self._canvas.set_tool(self._pen)
        return self._pen

    def stop_drawing(self) -> None:
        """Take the pen off the canvas, abandoning anything half-drawn."""
        if self._canvas.tool is self._pen and self._pen is not None:
            self._canvas.set_tool(None)
        self._pen = None
        self._refresh_status()

    def _on_draw_toggled(self, checked: bool) -> None:
        if checked:
            self.start_drawing()
        else:
            self.stop_drawing()

    def _on_kind_changed(self, text: str) -> None:
        if self._pen is not None:
            self.stop_drawing()
            self.draw_button.setChecked(True)
            self.start_drawing()

    def _on_mode_changed(self, text: str) -> None:
        self.overlap_spin.setEnabled(text == "overlap")

    def _on_roi_finished(self, _index: int) -> None:
        self._refresh_status()

    def _on_choose_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the ROI", self._roi_path, "ROI (*.json)")
        if path:
            self.set_roi_path(path)

    # -- handing it to Measure ---------------------------------------------
    def fields(self) -> List[str]:
        """The field stems the ROI is filed under, from the scope box."""
        text = self.field_edit.text().strip()
        names = [part.strip() for part in text.split(",") if part.strip()]
        return names or [ANY_FIELD]

    def roi_set(self) -> RoiSet:
        """Build a :class:`spacr.roi.RoiSet` from what has been drawn.

        :raises spacr.roi.RoiError: if nothing closed has been drawn yet.
        """
        layer = self.roi_layer(create=False)
        if layer is None:
            raise RoiError(
                "no ROI has been drawn yet. Press 'Draw ROI', click the "
                "corners of the region and double-click to close it.")
        return RoiSet.from_shapes_layer(
            layer, fields=self.fields(), mode=self.mode_combo.currentText(),
            min_overlap=float(self.overlap_spin.value()),
            invert=bool(self.invert_check.isChecked()))

    def enable(self) -> bool:
        """Register the drawn ROI as a Measure region filter.

        :returns: True when the filter was installed. A drawing mistake is
            reported in the status line rather than raised out of a click.
        """
        try:
            roi_set = self.roi_set()
            enable_roi_filter(roi_set, path=self._roi_path, verbose=False)
        except (RoiError, LayerError, OSError) as exc:
            LOG.info("could not enable the ROI filter", exc_info=True)
            self._show(str(exc), warning=True)
            return False
        self._enabled = True
        self.filter_changed.emit(True)
        self._refresh_status()
        return True

    def disable(self) -> bool:
        """Go back to measuring whole fields."""
        removed = disable_roi_filter()
        self._enabled = False
        self.filter_changed.emit(False)
        self._refresh_status()
        return removed

    # -- status -------------------------------------------------------------
    def _refresh_status(self) -> None:
        layer = self.roi_layer(create=False)
        drawn = sum(1 for s in (layer.shapes if layer else ()) if s.is_closed)
        if not self._enabled:
            self._show(f"{drawn} ROI(s) drawn · whole fields are measured "
                       f"until you press 'Measure inside ROI'")
            return
        ok, message = worker_delivery_status()
        self._show(f"{drawn} ROI(s) · {message}", warning=not ok)

    def _show(self, text: str, *, warning: bool = False) -> None:
        self.status.setObjectName("RoiStatusWarning" if warning
                                  else "RoiStatus")
        self.status.setText(text)
        # An objectName change only takes effect on a re-polish; without this
        # the warning colour arrives one message late.
        style = self.status.style()
        style.unpolish(self.status)
        style.polish(self.status)

    def closeEvent(self, event) -> None:
        """Take the pen off the canvas so it does not outlive this panel."""
        self.stop_drawing()
        super().closeEvent(event)
