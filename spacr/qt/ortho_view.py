"""``B15`` — orthogonal views and the sliders that drive them.

Three panels over one volume: the top view, the view from the side and the
view from the end, crossing at a point the user moves with a slider or a
click. :class:`spacr.layers.OrthoViews` is the model — it owns the geometry,
including the part that is silently wrong everywhere else — and this module is
the widget over it.

The z slider is labelled in world units
---------------------------------------

Not in slice numbers. "Slice 7" is a fact about the file; ``12.0 µm`` is a
fact about the sample, and it is the one a user can check against the
acquisition settings. The slider steps by the stack's own voxel size, so
dragging it moves one slice at a time and the number it shows is where the
plane actually is. On a stack with no calibration the unit is ``px`` and the
step is 1, which reads as slice numbers again — the same widget, telling the
truth in both cases.

Why the side panels are not one pixel per slice
-----------------------------------------------

Because a confocal stack is not isotropic. 0.65 µm in xy and 2 µm in z is an
ordinary spaCR stack, and a side view drawn one pixel per slice is three times
too thin — a picture that does not look broken, just slightly flat, and every
3-D shape read off it is wrong. :meth:`spacr.layers.OrthoViews.covering` gives
all three panels one world-units-per-pixel scale, so this module never divides
anything by a slice count.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QSizePolicy, QSlider, QVBoxLayout, QWidget)

from ..layers import Canvas, LabelsLayer, LayerError, LayerStack, OrthoViews
from .linked_selection import LinkedView
from .theme import register_widget_qss
from .widgets.preview_controls import FlatButton

LOG = logging.getLogger(__name__)

__all__ = [
    "OrthoPanel",
    "OrthoView",
    "ORTHO_LINK_SOURCE",
]

#: What this view calls itself on the shared selection.
ORTHO_LINK_SOURCE = "ortho_view"

#: How many world units a slider step is divided into. The slider itself is
#: integral; the world coordinate is not.
_TICKS = 1000


def _ortho_qss(palette: Dict[str, Any], opacity) -> str:
    """This view's QSS block, appended to every generated stylesheet."""
    return f"""
QWidget#OrthoView {{
    background: transparent;
}}
QFrame#OrthoPanel {{
    background: {palette["bg"]};
    border: 1px solid {palette["border_soft"]};
    border-radius: 8px;
}}
QLabel#OrthoStatus {{
    color: {palette["fg_muted"]};
}}
QLabel#OrthoPanelName {{
    color: {palette["fg_dim"]};
    font-size: 10px;
    letter-spacing: 1px;
}}
"""


register_widget_qss("OrthoView", _ortho_qss, replace=True)


# ---------------------------------------------------------------------------
# One panel
# ---------------------------------------------------------------------------

class OrthoPanel(QFrame):
    """One plane of an orthogonal view, with the crosshair drawn on it.

    Paints whatever :class:`~spacr.layers.Canvas` it is given, at that
    canvas's own resolution — the panel does not scale a pixmap, so the side
    views keep the world scale the model gave them however the widget is
    sized.

    :param name: ``'xy'``, ``'zx'`` or ``'yz'``; used for the caption and
        reported with a click.
    """

    #: The user clicked at ``(name, row, column)`` in canvas pixels.
    clicked = Signal(str, float, float)

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.setObjectName("OrthoPanel")
        self.setMinimumSize(80, 60)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._name = str(name)
        self._stack: Optional[LayerStack] = None
        self._canvas: Optional[Canvas] = None
        self._crosshair: Optional[Tuple[float, float]] = None

    @property
    def name(self) -> str:
        """Which plane this panel shows."""
        return self._name

    @property
    def canvas(self) -> Optional[Canvas]:
        """The world window being painted."""
        return self._canvas

    def show_canvas(self, stack: LayerStack, canvas: Canvas,
                    crosshair: Optional[Dict[str, float]] = None) -> None:
        """Paint ``stack`` through ``canvas``, with the crosshair at a point."""
        self._stack = stack
        self._canvas = canvas
        if crosshair is None:
            self._crosshair = None
        else:
            try:
                self._crosshair = canvas.pixel_at(crosshair)
            except KeyError:
                self._crosshair = None
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        try:
            if self._canvas is None or self._stack is None:
                painter.setPen(Qt.gray)
                painter.drawText(self.rect(), Qt.AlignCenter, "No volume")
                return
            from .widgets.live_preview import numpy_to_qpixmap
            pixmap = numpy_to_qpixmap(self._stack.render_uint8(self._canvas),
                                      normalise=False)
            painter.drawPixmap(1, 1, pixmap)
            if self._crosshair is not None:
                row, column = self._crosshair
                painter.setPen(Qt.yellow)
                painter.drawLine(1, int(row) + 1,
                                 1 + self._canvas.width, int(row) + 1)
                painter.drawLine(int(column) + 1, 1,
                                 int(column) + 1, 1 + self._canvas.height)
        except Exception:
            # A paint handler that raises takes the window with it.
            LOG.exception("Could not paint the %s panel", self._name)
        finally:
            painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or self._canvas is None:
            return
        position = event.position()
        self.clicked.emit(self._name, float(position.y()) - 1.0,
                          float(position.x()) - 1.0)


# ---------------------------------------------------------------------------
# The three-panel view
# ---------------------------------------------------------------------------

class OrthoView(LinkedView, QWidget):
    """XY, ZX and YZ over one volume, with a slider per extra dimension.

    :param stack: the :class:`~spacr.layers.LayerStack` to show. Everything in
        it is drawn on all three planes, so a labels layer over a z-stack is
        an outline in the side views too.
    :param width: the top panel's width in pixels; every other panel size
        follows from the world extents.
    :param frames: how many timepoints there are, when the volume is one frame
        of a series. Adds a ``t`` slider whose changes are announced through
        :attr:`frame_changed` rather than resolved here, because loading the
        next timepoint is the caller's job (and often a background one).
    """

    #: The crosshair moved. Carries ``{axis: world}``.
    point_changed = Signal(object)
    #: The timepoint slider moved. Carries the frame index.
    frame_changed = Signal(int)
    #: A labels layer was clicked and it knew its object key.
    object_picked = Signal(str)

    def __init__(self, stack: Optional[LayerStack] = None, parent=None, *,
                 width: int = 320, frames: int = 0,
                 axes: Tuple[str, str, str] = OrthoViews.DEFAULT_AXES):
        super().__init__(parent)
        self.setObjectName("OrthoView")
        self._stack = stack if stack is not None else LayerStack()
        self._axes = tuple(str(a) for a in axes)
        self._width = int(width)
        self._frames = max(0, int(frames))
        self._views: Optional[OrthoViews] = None
        self._sliders: Dict[str, QSlider] = {}
        self._syncing = False
        self._build()
        self.set_stack(self._stack)
        self.link_selection(ORTHO_LINK_SOURCE)

    # -- construction ------------------------------------------------------
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        grid = QGridLayout()
        grid.setSpacing(6)
        self.panels: Dict[str, OrthoPanel] = {}
        for name, (row, column) in (("xy", (0, 0)), ("yz", (0, 1)),
                                    ("zx", (1, 0))):
            holder = QWidget(self)
            column_layout = QVBoxLayout(holder)
            column_layout.setContentsMargins(0, 0, 0, 0)
            column_layout.setSpacing(2)
            caption = QLabel(name.upper(), holder)
            caption.setObjectName("OrthoPanelName")
            column_layout.addWidget(caption)
            panel = OrthoPanel(name, holder)
            panel.clicked.connect(self._on_panel_clicked)
            column_layout.addWidget(panel, 1)
            self.panels[name] = panel
            grid.addWidget(holder, row, column)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 3)
        grid.setRowStretch(1, 1)
        outer.addLayout(grid, 1)

        self.slider_box = QVBoxLayout()
        self.slider_box.setSpacing(4)
        outer.addLayout(self.slider_box)

        buttons = QHBoxLayout()
        buttons.setSpacing(4)
        self.zoom_in_button = FlatButton("＋", self, tooltip="Zoom every panel in")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button = FlatButton("－", self,
                                          tooltip="Zoom every panel out")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.fit_button = FlatButton("Fit", self,
                                     tooltip="Fit the whole volume again")
        self.fit_button.clicked.connect(self.reset_view)
        for button in (self.zoom_in_button, self.zoom_out_button,
                       self.fit_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        outer.addLayout(buttons)

        self.status = QLabel("", self)
        self.status.setObjectName("OrthoStatus")
        outer.addWidget(self.status)

    # -- the model ---------------------------------------------------------
    @property
    def stack(self) -> LayerStack:
        """The stack being shown."""
        return self._stack

    @property
    def views(self) -> Optional[OrthoViews]:
        """The three canvases, or ``None`` when the stack has no volume."""
        return self._views

    def set_stack(self, stack: LayerStack) -> None:
        """Show a different volume, rebuilding the sliders for its extents."""
        self._stack = stack
        self.reset_view()

    def reset_view(self) -> None:
        """Fit the whole volume in all three panels."""
        try:
            self._views = OrthoViews.covering(self._stack, width=self._width,
                                              axes=self._axes)
        except LayerError as exc:
            self._views = None
            self.status.setText(str(exc))
            self._build_sliders()
            self._repaint()
            return
        self._build_sliders()
        self._repaint()

    def _build_sliders(self) -> None:
        """One slider per axis the panels do not both span, plus time."""
        while self.slider_box.count():
            item = self.slider_box.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            elif item.layout() is not None:
                # The rows are nested layouts; draining the widgets and then
                # the layout keeps `set_stack` from leaving an empty row
                # behind every time it is called.
                self._drain(item.layout())
                item.layout().deleteLater()
        self._sliders = {}
        if self._views is None:
            return
        for axis in self._views.axes:
            self._add_slider(axis)
        if self._frames:
            self._add_frame_slider()

    @staticmethod
    def _drain(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _add_slider(self, axis: str) -> None:
        low, high, step = self._views.slider(axis)
        row = QHBoxLayout()
        row.setSpacing(6)
        caption = QLabel(axis, self)
        caption.setMinimumWidth(16)
        row.addWidget(caption)
        slider = QSlider(Qt.Horizontal, self)
        slider.setRange(0, _TICKS)
        slider.setValue(self._tick(axis, self._views.point.get(axis, low)))
        slider.setProperty("axis", axis)
        slider.valueChanged.connect(self._on_slider_moved)
        row.addWidget(slider, 1)
        value = QLabel("", self)
        value.setObjectName("OrthoStatus")
        value.setMinimumWidth(90)
        row.addWidget(value)
        self._sliders[axis] = slider
        slider.setProperty("readout", value)
        self.slider_box.addLayout(row)
        self._update_readout(axis)

    def _add_frame_slider(self) -> None:
        row = QHBoxLayout()
        row.setSpacing(6)
        caption = QLabel("t", self)
        caption.setMinimumWidth(16)
        row.addWidget(caption)
        slider = QSlider(Qt.Horizontal, self)
        slider.setRange(0, max(0, self._frames - 1))
        slider.valueChanged.connect(self.frame_changed.emit)
        row.addWidget(slider, 1)
        self.frame_slider = slider
        self.slider_box.addLayout(row)

    # -- the crosshair -----------------------------------------------------
    def _tick(self, axis: str, world: float) -> int:
        low, high, _step = self._views.slider(axis)
        span = high - low
        if span <= 0:
            return 0
        return int(round((float(world) - low) / span * _TICKS))

    def _world(self, axis: str, tick: int) -> float:
        low, high, _step = self._views.slider(axis)
        return low + (high - low) * (int(tick) / _TICKS)

    def _slice_grid(self, axis: str) -> Tuple[float, float]:
        """``(world of slice 0, world size of one slice)`` on ``axis``.

        The extent runs to the OUTER EDGE of the end voxels (see
        :meth:`spacr.layers.Spacing.extent`), so slice 0 sits half a voxel in
        from it. Snapping to the extent's edge instead would put every plane
        half a slice out of register — visible on nothing, wrong everywhere.
        """
        low, _high, step = self._views.slider(axis)
        return low + 0.5 * step, step

    def _snap(self, axis: str, world: float) -> float:
        """``world`` moved to the nearest voxel centre on ``axis``.

        The plane that gets drawn is a voxel plane whatever the slider says, so
        a readout of 11.3 µm for the plane at 12 µm is a number nobody can
        check against anything. Held inside the real slices too: the extent
        runs half a voxel past the last one, and a crosshair out there would
        read as one more slice than the stack has.
        """
        origin, step = self._slice_grid(axis)
        if step <= 0:
            return float(world)
        index = int(round((float(world) - origin) / step))
        index = min(max(index, 0), self._views.n_slices(axis) - 1)
        return origin + index * step

    def slice_index(self, axis: str) -> int:
        """Which slice the crosshair is on along ``axis``, counting from 0."""
        origin, step = self._slice_grid(axis)
        if step <= 0:
            return 0
        return int(round((self._views.point[axis] - origin) / step))

    def _on_slider_moved(self, tick: int) -> None:
        if self._syncing or self._views is None:
            return
        axis = self.sender().property("axis")
        self.move_to(**{axis: self._world(axis, tick)})

    def move_to(self, **coords: float) -> None:
        """Move the crosshair: ``view.move_to(z=12.0)``.

        Each coordinate is snapped to its own voxel grid and held inside the
        volume, so the crosshair is always on a plane that exists.
        """
        if self._views is None:
            return
        self._views = self._views.clamped(
            **{axis: self._snap(axis, value) for axis, value in coords.items()})
        self._sync_sliders()
        self._repaint()
        self.point_changed.emit(dict(self._views.point))

    def _sync_sliders(self) -> None:
        self._syncing = True
        try:
            for axis, slider in self._sliders.items():
                slider.setValue(self._tick(axis, self._views.point[axis]))
                self._update_readout(axis)
        finally:
            self._syncing = False

    def _update_readout(self, axis: str) -> None:
        slider = self._sliders.get(axis)
        readout = slider.property("readout") if slider is not None else None
        if readout is None:
            return
        world = self._views.point[axis]
        readout.setText(f"{world:.6g} {self._views.xy.units}"
                        f"  ·  slice {self.slice_index(axis)}")

    def _on_panel_clicked(self, name: str, row: float, column: float) -> None:
        """A click moves the crosshair AND selects whatever is under it."""
        if self._views is None:
            return
        canvas = self._views.canvases()[name]
        under = canvas.world_at(row, column)
        self.move_to(**{axis: under[axis] for axis in canvas.axes})
        layer, world, value = self._stack.pick(canvas, row, column)
        if isinstance(layer, LabelsLayer) and value:
            key = layer.object_key_at_world(world)
            if key is not None:
                layer.selected_label = int(value)
                self.object_picked.emit(key)
                self.publish_selection([key])

    # -- zoom ---------------------------------------------------------------
    def zoom_in(self) -> None:
        """Every panel one step closer, about the crosshair."""
        self._zoom(1.25)

    def zoom_out(self) -> None:
        """Every panel one step further away."""
        self._zoom(1 / 1.25)

    def _zoom(self, factor: float) -> None:
        if self._views is None:
            return
        self._views = self._views.zoomed(factor)
        self._repaint()

    def wheelEvent(self, event) -> None:
        self._zoom(1.2 if event.angleDelta().y() > 0 else 1 / 1.2)
        event.accept()

    # -- painting -----------------------------------------------------------
    def _repaint(self) -> None:
        if self._views is None:
            for panel in self.panels.values():
                panel.show_canvas(self._stack, None)
            return
        crosshair = dict(self._views.point)
        for name, canvas in self._views.canvases().items():
            self.panels[name].show_canvas(self._stack, canvas, crosshair)
        self.status.setText(
            f"{self._views.describe()} · {self._views.scale:.4g} "
            f"{self._views.xy.units}/px")

    # -- the shared selection ----------------------------------------------
    def on_linked_selection_changed(self, selection) -> None:
        """Move the crosshair onto the object another view selected."""
        if selection.keys is None or len(selection.keys) != 1:
            return
        wanted = str(selection.keys[0])
        for layer in self._stack:
            if not isinstance(layer, LabelsLayer) or layer.field is None:
                continue
            for label in layer.labels():
                if layer.field.object_key(int(label)) != wanted:
                    continue
                layer.selected_label = int(label)
                where = np.argwhere(np.asarray(layer.data) == int(label))
                if len(where):
                    centre = where.mean(axis=0)
                    self.move_to(**layer.spacing.world_map(centre))
                return

    def closeEvent(self, event) -> None:
        """Leave the shared selection when the screen goes."""
        self.unlink_selection()
        super().closeEvent(event)
