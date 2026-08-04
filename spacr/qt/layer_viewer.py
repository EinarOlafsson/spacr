"""The Qt view over :mod:`spacr.layers` — a napari-style layer viewer.

Everything that decides what a pixel ends up being lives in
:mod:`spacr.layers`, which knows nothing about Qt. This module is the part
that can only exist with a display: a canvas that paints the model's composite,
a layer list that reorders it, and the controls that set the properties the
model already knows how to honour.

The split is not tidiness. Five later features build on the layer model — ROI
shapes read by Measure, a counting points layer, a label brush with timelapse
track curation, orthogonal views with dimension sliders, and a synchronised
comparison grid — and every one of them is a change to the model with a small
widget on top. Putting the compositing rules in a ``paintEvent`` would make
all five untestable without a running ``QApplication``.

Why this canvas paints rather than reusing ``_ZoomView``
-------------------------------------------------------

:class:`spacr.qt.widgets.live_preview._ZoomView` is the right answer when the
thing being shown is a finished pixmap: it scales what it was given. Here the
zoom is a property of the world, not of the picture — zooming in means
:meth:`spacr.layers.Canvas.zoomed` and a fresh render at the new resolution,
which is how the labels layer stays crisp at 8× and how the orthogonal-view
item gets its slice sliders for free. So the canvas owns a
:class:`~spacr.layers.Canvas` and paints; the array→pixmap step still goes
through :func:`~spacr.qt.widgets.live_preview.numpy_to_qpixmap`, and file
loading through :func:`~spacr.qt.widgets.live_preview.load_preview_image`,
rather than growing a third copy of either.

Clicking an object
------------------

Clicking a labels layer picks the object under the cursor, and — if the layer
was told which field it segments — publishes it through
:mod:`spacr.qt.linked_selection`, so the same cell lights up in the UMAP, the
plate view and the annotation grid. Double-clicking asks for it to be opened.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QFrame,
                               QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QSizePolicy, QSlider,
                               QSplitter, QVBoxLayout, QWidget)

from ..layers import (Blending, COLORMAPS, Canvas, FieldKey, ImageLayer,
                      LabelsLayer, LayerError, LayerEvent, LayerStack,
                      PointsLayer, ShapesLayer, Spacing)
from .linked_selection import DEFAULT_OPEN_KIND, LinkedView, has_object_opener
from .theme import font_px, register_widget_qss
from .widgets.preview_controls import FlatButton, FlatComboBox

LOG = logging.getLogger(__name__)

__all__ = [
    "CanvasTool",
    "LayerCanvas",
    "LayerListWidget",
    "LayerViewer",
    "make_layer_viewer_screen",
    "register_companion_apps",
    "register_layer_viewer_app",
    "stack_from_paths",
    "COMPANION_APPS",
    "LAYER_VIEWER_APP_KEY",
    "LINK_SOURCE",
]

#: The app key this viewer registers under, when it is registered. Stable, and
#: referred to by :func:`register_layer_viewer_app`.
LAYER_VIEWER_APP_KEY = "layer_viewer"

#: What this view calls itself on the shared selection, so it can ignore the
#: echo of its own clicks.
LINK_SOURCE = "layer_viewer"


# ---------------------------------------------------------------------------
# Styling, through the seam rather than through theme.py
# ---------------------------------------------------------------------------

def _layer_viewer_qss(palette: Dict[str, Any], opacity) -> str:
    """The viewer's own QSS block, appended to every generated stylesheet.

    The canvas frame used to take ``palette["bg"]`` — the WINDOW colour,
    which is not a surface and which no page-opacity setting can reach, so
    the largest region on the Curate and Layer Viewer pages stayed a flat
    slab over the animated backdrop wherever the slider was. It is a page
    surface now. The *image* is unaffected: the rendered pixmap is drawn
    opaque on top of the panel in :meth:`LayerCanvas.paintEvent`, and the
    preference was never meant to reach the pictures themselves.
    """
    from .theme import pane_surface
    canvas_bg = pane_surface("surface", palette.get("theme"), opacity)
    return f"""
QWidget#LayerViewer {{
    background: transparent;
}}
QFrame#LayerCanvasFrame {{
    background: {canvas_bg};
    border: 1px solid {palette["border_soft"]};
    border-radius: 10px;
}}
QListWidget#LayerList {{
    background: {palette["surface_alt"]};
    border: 1px solid {palette["border_soft"]};
    border-radius: 10px;
    padding: 4px;
    color: {palette["fg"]};
}}
QListWidget#LayerList::item {{
    padding: 5px 6px;
    border-radius: 6px;
}}
QListWidget#LayerList::item:selected {{
    background: {palette["accent_soft"]};
    color: {palette["fg"]};
}}
QLabel#LayerViewerStatus {{
    color: {palette["fg_muted"]};
}}
QLabel#LayerViewerHeading {{
    color: {palette["fg_dim"]};
    font-size: {font_px(11)}px;
    letter-spacing: 1px;
}}
"""


register_widget_qss("LayerViewer", _layer_viewer_qss, replace=True)


# ---------------------------------------------------------------------------
# Loading, reusing the preview stack's readers
# ---------------------------------------------------------------------------

def stack_from_paths(image_path=None, labels_path=None, *,
                     spacing: Optional[Spacing] = None,
                     field: Optional[FieldKey] = None) -> LayerStack:
    """Build a stack from an image file and/or a label-mask file.

    Loading goes through
    :func:`spacr.qt.widgets.live_preview.load_preview_image`, which already
    knows to read TIFFs with ``tifffile`` so a 16-bit field keeps its bit
    depth, and channel order through
    :func:`spacr.qt.widgets.timelapse_preview.frame_channel`, which resolves
    channels-first against channels-last the same way the timelapse preview
    does. Neither is re-implemented here.

    :param spacing: the world spacing both layers share. Defaults to one
        world unit per pixel — correct for a single field, and the thing to
        override the moment a z-stack or a µm-calibrated mosaic is involved.
    """
    from .widgets.live_preview import load_preview_image

    stack = LayerStack()
    spacing = spacing or Spacing.isotropic(2, 1.0, units="px")
    if image_path is not None:
        array = load_preview_image(image_path)
        channel_axis = None
        if array.ndim == 3:
            # `frame_channel` owns the channels-first/last heuristic; asking it
            # for channel 0 tells us which axis it decided was channels.
            from .widgets.timelapse_preview import frame_channel
            first = frame_channel(array, 0)
            channel_axis = next(
                (i for i in (0, array.ndim - 1)
                 if tuple(np.delete(array.shape, i)) == first.shape), None)
        stack.add_image(array, name="image", channel_axis=channel_axis,
                        spacing=spacing)
    if labels_path is not None:
        mask = load_preview_image(labels_path)
        while mask.ndim > 2:
            mask = mask[..., 0] if mask.shape[-1] <= 8 else mask[0]
        stack.add_labels(mask.astype(np.int64), name="mask", spacing=spacing,
                         field=field, opacity=0.5)
    return stack


# ---------------------------------------------------------------------------
# Tools — what takes the canvas's mouse away from picking
# ---------------------------------------------------------------------------

class CanvasTool:
    """Something that borrows the canvas's mouse: an ROI pen, a counter.

    Two features need a click to mean something other than "select the object
    under the cursor" — drawing a polygon vertex (:mod:`spacr.qt.roi_tool`) and
    dropping a counted marker (:mod:`spacr.qt.counting_tool`) — and both need
    it in *world* coordinates, not widget pixels, so that a point placed at 8×
    zoom lands where the same point placed at 1× does.

    Every handler is given the world position the canvas has already resolved
    and returns ``True`` when it consumed the event. Returning ``False`` (the
    default for every method here) leaves the canvas doing exactly what it did
    before tools existed, which is what makes attaching one reversible.

    Subclass and override what you need; the base class is inert, so a tool
    that only wants clicks does not have to implement four no-ops.
    """

    #: What the cursor becomes while this tool is attached. ``None`` leaves it.
    cursor = None

    def press(self, view: "LayerCanvas", world: Dict[str, float],
              event: Any) -> bool:
        """A mouse button went down at ``world``. Return True to consume it."""
        return False

    def move(self, view: "LayerCanvas", world: Dict[str, float],
             event: Any) -> bool:
        """The cursor moved to ``world`` with no drag in progress."""
        return False

    def release(self, view: "LayerCanvas", world: Dict[str, float],
                event: Any) -> bool:
        """The mouse button came up at ``world`` — the end of a drag.

        What turns a drag into ONE action. A brush stroke is dozens of
        :meth:`move` calls and exactly one thing the user did, so undo has to
        take back the stroke rather than the last few pixels of it; without a
        release the tool cannot tell where one stroke ends and the next
        begins. Inert by default, like the rest of this class, so a tool that
        only wants clicks is unaffected.
        """
        return False

    def double_click(self, view: "LayerCanvas", world: Dict[str, float],
                     event: Any) -> bool:
        """A double click at ``world`` — how a polygon is closed."""
        return False

    def key(self, view: "LayerCanvas", event: Any) -> bool:
        """A key was pressed while the canvas had focus."""
        return False

    def detach(self) -> None:
        """The tool was taken off the canvas. Drop anything half-drawn."""


# ---------------------------------------------------------------------------
# The canvas
# ---------------------------------------------------------------------------

class LayerCanvas(QFrame):
    """Paints a :class:`~spacr.layers.LayerStack` through a world window.

    Wheel zooms about the cursor, dragging pans, and every change re-renders
    at the widget's own resolution rather than scaling a stale pixmap.
    """

    #: ``(layer, world, value)`` — whatever :meth:`LayerStack.pick` found.
    picked = Signal(object, object, object)
    #: The same, for a double click.
    activated = Signal(object, object, object)
    #: ``{axis: world}`` under the cursor, for a status line.
    hovered = Signal(object)
    #: The world window moved or resized.
    view_changed = Signal()

    def __init__(self, stack: Optional[LayerStack] = None, parent=None):
        super().__init__(parent)
        self.setObjectName("LayerCanvasFrame")
        self.setMinimumSize(240, 180)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._stack = stack if stack is not None else LayerStack()
        self._canvas: Optional[Canvas] = None
        self._axes = ("y", "x")
        self._depth: Dict[str, float] = {}
        self._drag: Optional[QPoint] = None
        self._tool: Optional[CanvasTool] = None
        self._stack.subscribe(self._on_layers_changed)

    # -- model ----------------------------------------------------------
    @property
    def stack(self) -> LayerStack:
        return self._stack

    def set_stack(self, stack: LayerStack) -> None:
        """Show a different stack, unsubscribing from the old one.

        The unsubscribe matters: the model holds listeners by strong
        reference, so a canvas that swapped stacks without letting go would
        keep repainting for a stack nobody is looking at.
        """
        self._stack.unsubscribe(self._on_layers_changed)
        self._stack = stack
        self._stack.subscribe(self._on_layers_changed)
        self._canvas = None
        self.update()

    def detach(self) -> None:
        """Stop listening to the model. Call from ``closeEvent``."""
        self._stack.unsubscribe(self._on_layers_changed)

    def _on_layers_changed(self, event: LayerEvent) -> None:
        if event.kind in LayerEvent.REPAINT:
            self.update()

    # -- tools ------------------------------------------------------------
    @property
    def tool(self) -> Optional[CanvasTool]:
        """The :class:`CanvasTool` currently borrowing the mouse, if any."""
        return self._tool

    def set_tool(self, tool: Optional[CanvasTool]) -> Optional[CanvasTool]:
        """Attach a tool (or ``None`` to go back to picking); returns the old one.

        Keyboard focus is granted only while a tool is attached: a tool
        typically wants Escape and Backspace, and a canvas that grabbed focus
        the rest of the time would swallow the arrow keys the surrounding
        screen uses.
        """
        previous = self._tool
        if previous is tool:
            return previous
        if previous is not None:
            previous.detach()
        self._tool = tool
        if tool is None:
            self.setFocusPolicy(Qt.NoFocus)
            self.unsetCursor()
        else:
            self.setFocusPolicy(Qt.StrongFocus)
            if tool.cursor is not None:
                self.setCursor(tool.cursor)
        self.update()
        return previous

    def _tool_world(self, event) -> Optional[Dict[str, float]]:
        """The world point under an event, or ``None`` with nothing to show."""
        canvas = self._ensure_canvas()
        if canvas is None:
            return None
        return canvas.world_at(*self._pixel(event))

    # -- the world window -----------------------------------------------
    @property
    def canvas(self) -> Optional[Canvas]:
        """The world window being shown, or ``None`` for an empty stack."""
        return self._canvas

    def set_plane(self, axes: Sequence[str],
                  depth: Optional[Dict[str, float]] = None) -> None:
        """Look at a different plane — the seam the orthogonal-view item uses."""
        self._axes = (str(axes[0]), str(axes[1]))
        self._depth = dict(depth or {})
        self._canvas = None
        self.update()
        self.view_changed.emit()

    def set_depth(self, **coords: float) -> None:
        """Move to another slice: ``canvas.set_depth(z=12.0)``."""
        self._depth.update({k: float(v) for k, v in coords.items()})
        if self._canvas is not None:
            self._canvas = self._canvas.at_depth(**self._depth)
        self.update()
        self.view_changed.emit()

    def reset_view(self) -> None:
        """Fit the whole stack into the widget."""
        self._canvas = None
        self.update()
        self.view_changed.emit()

    def _ensure_canvas(self) -> Optional[Canvas]:
        width = max(1, self.width() - 2)
        height = max(1, self.height() - 2)
        if self._canvas is None:
            if not len(self._stack):
                return None
            try:
                self._canvas = Canvas.covering(
                    self._stack, height=height, width=width,
                    axes=self._axes, depth=self._depth, margin=0.02)
            except LayerError:
                LOG.debug("Cannot fit a canvas to %s", self._axes)
                return None
        elif self._canvas.shape != (height, width):
            self._canvas = self._canvas.resized(height, width)
        return self._canvas

    # -- painting -------------------------------------------------------
    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        canvas = self._ensure_canvas()
        painter = QPainter(self)
        try:
            if canvas is None:
                painter.setPen(Qt.gray)
                painter.drawText(self.rect(), Qt.AlignCenter,
                                 "No layers yet")
                return
            from .widgets.live_preview import numpy_to_qpixmap
            pixmap = numpy_to_qpixmap(self._stack.render_uint8(canvas),
                                      normalise=False)
            painter.drawPixmap(1, 1, pixmap)
        except Exception:
            # A paint handler that raises takes the window with it, and the
            # traceback never reaches the console panel.
            LOG.exception("Could not paint the layer canvas")
        finally:
            painter.end()

    # -- interaction ----------------------------------------------------
    @staticmethod
    def _pixel(event) -> tuple:
        """Canvas pixel under an event, allowing for the 1 px frame inset."""
        position = event.position()
        return (float(position.y()) - 1.0, float(position.x()) - 1.0)

    def wheelEvent(self, event) -> None:
        canvas = self._ensure_canvas()
        if canvas is None:
            return
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self._canvas = canvas.zoomed(factor, centre=self._pixel(event))
        self.update()
        self.view_changed.emit()
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton or (
                event.button() == Qt.LeftButton
                and event.modifiers() & Qt.ShiftModifier):
            self._drag = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            return
        canvas = self._ensure_canvas()
        if canvas is None:
            return
        if self._tool is not None and self._tool.press(
                self, canvas.world_at(*self._pixel(event)), event):
            self.update()
            return
        if event.button() != Qt.LeftButton:
            return
        row, column = self._pixel(event)
        self.picked.emit(*self._stack.pick(canvas, row, column))

    def mouseDoubleClickEvent(self, event) -> None:
        canvas = self._ensure_canvas()
        if canvas is None:
            return
        if self._tool is not None and self._tool.double_click(
                self, canvas.world_at(*self._pixel(event)), event):
            self.update()
            return
        if event.button() != Qt.LeftButton:
            return
        row, column = self._pixel(event)
        self.activated.emit(*self._stack.pick(canvas, row, column))

    def keyPressEvent(self, event) -> None:
        if self._tool is not None and self._tool.key(self, event):
            self.update()
            event.accept()
            return
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        canvas = self._ensure_canvas()
        if canvas is None:
            return
        if (self._drag is None and self._tool is not None
                and self._tool.move(self, canvas.world_at(*self._pixel(event)),
                                    event)):
            self.update()
            return
        if self._drag is not None:
            position = event.position().toPoint()
            delta = position - self._drag
            self._drag = position
            self._canvas = canvas.panned(-delta.y(), -delta.x())
            self.update()
            self.view_changed.emit()
            return
        row, column = self._pixel(event)
        self.hovered.emit(canvas.world_at(row, column))

    def mouseReleaseEvent(self, event) -> None:
        if self._drag is not None:
            self._drag = None
            self.unsetCursor()
            return
        # Offered to the tool AFTER the pan check, so releasing a shift-drag
        # pan never reads as the end of a brush stroke.
        canvas = self._ensure_canvas()
        if self._tool is not None and canvas is not None and self._tool.release(
                self, canvas.world_at(*self._pixel(event)), event):
            self.update()


# ---------------------------------------------------------------------------
# The layer list
# ---------------------------------------------------------------------------

class LayerListWidget(QListWidget):
    """The stack as a list, top layer first — the order the user sees it in.

    Reversed on purpose: the model's index 0 is the bottom layer (a stack of
    acetates), while a list reads top-down, so the topmost row is the layer
    nearest the viewer. Every drag and every button here is translated back
    into a model index, so the model stays the single source of order.
    """

    def __init__(self, stack: LayerStack, parent=None):
        super().__init__(parent)
        self.setObjectName("LayerList")
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.DoubleClicked
                             | QAbstractItemView.EditKeyPressed)
        self._stack = stack
        self._syncing = False
        self._stack.subscribe(self._on_layers_changed)
        self.itemChanged.connect(self._on_item_changed)
        self.itemSelectionChanged.connect(self._on_selection_changed)
        self.model().rowsMoved.connect(self._on_rows_moved)
        self.refresh()

    def detach(self) -> None:
        """Stop listening to the model. Call from ``closeEvent``."""
        self._stack.unsubscribe(self._on_layers_changed)

    # -- model -> list ---------------------------------------------------
    def refresh(self) -> None:
        """Rebuild every row from the model."""
        self._syncing = True
        try:
            self.clear()
            for layer in reversed(list(self._stack)):
                item = QListWidgetItem(layer.name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable
                              | Qt.ItemIsEditable)
                item.setCheckState(Qt.Checked if layer.visible
                                   else Qt.Unchecked)
                item.setData(Qt.UserRole, layer.name)
                item.setToolTip(layer.describe())
                self.addItem(item)
            selected = self._stack.selected
            if selected is not None:
                row = len(self._stack) - 1 - self._stack.index(selected)
                self.setCurrentRow(row)
        finally:
            self._syncing = False

    def _on_layers_changed(self, event: LayerEvent) -> None:
        # Not while this list is the one making the change. `refresh()` clears
        # every row, and the handler that started it is still holding the
        # QListWidgetItem it was called with — using it after the rebuild is
        # a "C++ object already deleted" RuntimeError raised inside a Qt
        # signal, where no `except` in this file can reach it.
        if self._syncing:
            return
        if event.kind in ("inserted", "removed", "moved", "renamed",
                          "selected") or event.detail == "visible":
            self.refresh()

    # -- list -> model ---------------------------------------------------
    def _layer_for(self, item: QListWidgetItem):
        name = item.data(Qt.UserRole)
        try:
            return self._stack[name]
        except KeyError:
            return None

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._syncing:
            return
        layer = self._layer_for(item)
        if layer is None:
            return
        self._syncing = True
        try:
            layer.visible = item.checkState() == Qt.Checked
            text = item.text().strip()
            if text and text != layer.name:
                final = self._stack.rename(layer, text)
                item.setData(Qt.UserRole, final)
                item.setText(final)
            item.setToolTip(layer.describe())
        except LayerError:
            LOG.exception("Could not apply a layer-list edit")
            self.refresh()
        finally:
            self._syncing = False

    def _on_selection_changed(self) -> None:
        if self._syncing:
            return
        item = self.currentItem()
        if item is None:
            return
        layer = self._layer_for(item)
        if layer is not None:
            self._stack.select(layer)

    def _on_rows_moved(self, *_args) -> None:
        """A drag reordered the rows: push the new z-order into the model."""
        if self._syncing:
            return
        names = [self.item(i).data(Qt.UserRole) for i in range(self.count())]
        self._syncing = True
        try:
            # The list is top-first; the model is bottom-first.
            for position, name in enumerate(reversed(names)):
                self._stack.move(name, position)
        except LayerError:
            LOG.exception("Could not reorder the layers")
        finally:
            self._syncing = False
        self.refresh()


# ---------------------------------------------------------------------------
# The viewer
# ---------------------------------------------------------------------------

class LayerViewer(LinkedView, QWidget):
    """Canvas, layer list and per-layer controls over one
    :class:`~spacr.layers.LayerStack`."""

    #: A labels layer was clicked and it knew its object key.
    object_picked = Signal(str)

    def __init__(self, stack: Optional[LayerStack] = None, parent=None):
        super().__init__(parent)
        self.setObjectName("LayerViewer")
        self._stack = stack if stack is not None else LayerStack()
        self._build()
        self._stack.subscribe(self._on_layers_changed)
        self.link_selection(LINK_SOURCE)
        self._sync_controls()
        # Drop anywhere on this screen: the path is resolved through spaCR's
        # project layout, so the plate folder finds what this screen reads.
        from .dnd import install_for
        install_for(self, "layer_viewer")

    # -- construction ----------------------------------------------------
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        split = QSplitter(Qt.Horizontal, self)
        self.canvas = LayerCanvas(self._stack, self)
        self.canvas.picked.connect(self._on_picked)
        self.canvas.activated.connect(self._on_activated)
        self.canvas.hovered.connect(self._on_hovered)
        split.addWidget(self.canvas)

        side = QWidget(self)
        column = QVBoxLayout(side)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        heading = QLabel("LAYERS", side)
        heading.setObjectName("LayerViewerHeading")
        column.addWidget(heading)

        self.layer_list = LayerListWidget(self._stack, side)
        column.addWidget(self.layer_list, 1)

        buttons = QHBoxLayout()
        buttons.setSpacing(4)
        self.add_image_button = FlatButton(
            "＋ Image", side, tooltip="Add an image layer from a file")
        self.add_image_button.clicked.connect(self._on_add_image)
        self.add_mask_button = FlatButton(
            "＋ Mask", side, tooltip="Add a label mask layer from a file")
        self.add_mask_button.clicked.connect(self._on_add_mask)
        self.add_points_button = FlatButton(
            "＋ Points", side, tooltip="Add an empty points layer")
        self.add_points_button.clicked.connect(self._on_add_points)
        self.add_shapes_button = FlatButton(
            "＋ Shapes", side, tooltip="Add an empty shapes layer")
        self.add_shapes_button.clicked.connect(self._on_add_shapes)
        for button in (self.add_image_button, self.add_mask_button,
                       self.add_points_button, self.add_shapes_button):
            buttons.addWidget(button)
        column.addLayout(buttons)

        order = QHBoxLayout()
        order.setSpacing(4)
        self.raise_button = FlatButton("↑", side, tooltip="Move layer up")
        self.raise_button.clicked.connect(
            lambda: self._reorder(self._stack.raise_layer))
        self.lower_button = FlatButton("↓", side, tooltip="Move layer down")
        self.lower_button.clicked.connect(
            lambda: self._reorder(self._stack.lower_layer))
        self.remove_button = FlatButton("✕", side,
                                        tooltip="Remove the selected layer")
        self.remove_button.clicked.connect(self._on_remove)
        self.reset_button = FlatButton("Fit", side,
                                       tooltip="Fit every layer in the view")
        self.reset_button.clicked.connect(self.canvas.reset_view)
        for button in (self.raise_button, self.lower_button,
                       self.remove_button, self.reset_button):
            order.addWidget(button)
        column.addLayout(order)

        column.addWidget(QLabel("Opacity", side))
        self.opacity_slider = QSlider(Qt.Horizontal, side)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self._on_opacity)
        column.addWidget(self.opacity_slider)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.blending_combo = FlatComboBox(side)
        self.blending_combo.addItems(list(Blending.ALL))
        self.blending_combo.currentTextChanged.connect(self._on_blending)
        row.addWidget(self.blending_combo, 1)
        self.colormap_combo = FlatComboBox(side)
        self.colormap_combo.addItems(sorted(COLORMAPS))
        self.colormap_combo.currentTextChanged.connect(self._on_colormap)
        row.addWidget(self.colormap_combo, 1)
        column.addLayout(row)

        split.addWidget(side)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([700, 260])
        outer.addWidget(split, 1)

        self.status = QLabel("", self)
        self.status.setObjectName("LayerViewerStatus")
        outer.addWidget(self.status)

    # -- model ------------------------------------------------------------
    @property
    def stack(self) -> LayerStack:
        return self._stack

    def _on_layers_changed(self, event: LayerEvent) -> None:
        if event.kind in ("selected", "inserted", "removed"):
            self._sync_controls()

    def _sync_controls(self) -> None:
        """Point the property editors at the selected layer."""
        layer = self._stack.selected
        for widget in (self.opacity_slider, self.blending_combo,
                       self.colormap_combo, self.raise_button,
                       self.lower_button, self.remove_button):
            widget.setEnabled(layer is not None)
        self.colormap_combo.setEnabled(isinstance(layer, ImageLayer))
        if layer is None:
            self.status.setText("No layer selected")
            return
        blocked = [w.blockSignals(True) for w in
                   (self.opacity_slider, self.blending_combo,
                    self.colormap_combo)]
        try:
            self.opacity_slider.setValue(int(round(layer.opacity * 100)))
            self.blending_combo.setCurrentText(layer.blending)
            if isinstance(layer, ImageLayer):
                index = self.colormap_combo.findText(layer.colormap.name)
                if index >= 0:
                    self.colormap_combo.setCurrentIndex(index)
        finally:
            for widget, was in zip((self.opacity_slider, self.blending_combo,
                                    self.colormap_combo), blocked):
                widget.blockSignals(was)
        self.status.setText(
            f"{layer.describe()} · {layer.spacing.describe()}")

    # -- control handlers --------------------------------------------------
    def _on_opacity(self, value: int) -> None:
        layer = self._stack.selected
        if layer is not None:
            layer.opacity = value / 100.0

    def _on_blending(self, text: str) -> None:
        layer = self._stack.selected
        if layer is None or not text:
            return
        try:
            layer.blending = text
        except LayerError:
            LOG.exception("Could not set the blending mode")

    def _on_colormap(self, text: str) -> None:
        layer = self._stack.selected
        if isinstance(layer, ImageLayer) and text:
            try:
                layer.set_colormap(text)
            except LayerError:
                LOG.exception("Could not set the colormap")

    def _reorder(self, move) -> None:
        layer = self._stack.selected
        if layer is not None:
            move(layer)

    def _on_remove(self) -> None:
        layer = self._stack.selected
        if layer is not None:
            self._stack.remove(layer)

    def _on_add_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Add an image layer", "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.npy);;All files (*)")
        if path:
            self.add_image_file(path)

    def _on_add_mask(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Add a label mask layer", "",
            "Masks (*.tif *.tiff *.png *.npy);;All files (*)")
        if path:
            self.add_labels_file(path)

    def add_image_file(self, path) -> Optional[ImageLayer]:
        """Load ``path`` as an image layer. Returns it, or ``None`` on failure."""
        try:
            loaded = stack_from_paths(image_path=path,
                                      spacing=self._default_spacing())
        except Exception:
            LOG.exception("Could not load %s as an image layer", path)
            self.status.setText(f"Could not load {path}")
            return None
        layer = loaded[0]
        loaded.remove(layer)
        return self._stack.append(layer)

    def add_labels_file(self, path,
                        field: Optional[FieldKey] = None
                        ) -> Optional[LabelsLayer]:
        """Load ``path`` as a labels layer."""
        try:
            loaded = stack_from_paths(labels_path=path, field=field,
                                      spacing=self._default_spacing())
        except Exception:
            LOG.exception("Could not load %s as a labels layer", path)
            self.status.setText(f"Could not load {path}")
            return None
        layer = loaded[0]
        loaded.remove(layer)
        return self._stack.append(layer)

    def _default_spacing(self) -> Spacing:
        """The spacing a newly added 2-D layer gets.

        Taken from whatever is already in the stack, so adding a mask to a
        µm-calibrated image does not silently produce a pixel-spaced layer the
        stack would then refuse.
        """
        for layer in self._stack:
            if layer.ndim == 2:
                return layer.spacing
        return Spacing.isotropic(2, 1.0, units=self._stack.units)

    def _on_add_points(self) -> None:
        self._stack.add_points(name="points", ndim=2,
                               spacing=self._default_spacing(), size=12.0)

    def _on_add_shapes(self) -> None:
        self._stack.add_shapes(name="shapes", ndim=2,
                               spacing=self._default_spacing())

    # -- picking ----------------------------------------------------------
    def _on_picked(self, layer, world, value) -> None:
        if layer is not None:
            self._stack.select(layer)
        self.status.setText(self._describe_pick(layer, world, value))
        key = self._object_key(layer, world, value)
        if key is None:
            return
        layer.selected_label = int(value)
        self.object_picked.emit(key)
        self.publish_selection([key])

    def _on_activated(self, layer, world, value) -> None:
        key = self._object_key(layer, world, value)
        # Asked rather than caught: with nothing registered to show crops, a
        # double click should do nothing visible, not raise NoObjectOpener out
        # of a mouse handler.
        if key is None or not has_object_opener(DEFAULT_OPEN_KIND):
            return
        try:
            self.open_objects([key], reason="double-clicked in the layer viewer")
        except Exception:
            LOG.exception("Could not open %s", key)

    @staticmethod
    def _object_key(layer, world, value) -> Optional[str]:
        if not isinstance(layer, LabelsLayer) or not value:
            return None
        return layer.object_key_at_world(world)

    def _describe_pick(self, layer, world, value) -> str:
        position = " · ".join(f"{axis} {coordinate:.6g}"
                              for axis, coordinate in world.items())
        if layer is None:
            return f"{position} · nothing here"
        if isinstance(layer, LabelsLayer):
            key = layer.object_key_at_world(world)
            named = f" · {key}" if key else ""
            return f"{position} · {layer.name} label {value}{named}"
        if isinstance(layer, PointsLayer):
            return f"{position} · {layer.name} point {value}"
        if isinstance(layer, ShapesLayer):
            return f"{position} · {layer.name} shape {value}"
        return f"{position} · {layer.name}"

    def _on_hovered(self, world) -> None:
        if self._stack.selected is None:
            self.status.setText(" · ".join(
                f"{axis} {coordinate:.6g}"
                for axis, coordinate in world.items()))

    # -- the shared selection ---------------------------------------------
    def on_linked_selection_changed(self, selection) -> None:
        """Highlight the object another view selected, when we hold it."""
        if selection.keys is None or len(selection.keys) != 1:
            return
        wanted = str(selection.keys[0])
        for layer in self._stack:
            if not isinstance(layer, LabelsLayer) or layer.field is None:
                continue
            for label in layer.labels():
                if layer.field.object_key(int(label)) == wanted:
                    layer.selected_label = int(label)
                    self.status.setText(f"{layer.name} label {label} · {wanted}")
                    return

    def closeEvent(self, event) -> None:
        self.unlink_selection()
        self._stack.unsubscribe(self._on_layers_changed)
        self.canvas.detach()
        self.layer_list.detach()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def make_layer_viewer_screen(**_kwargs) -> LayerViewer:
    """Build the viewer as an app screen. The ``factory=`` for
    :func:`~spacr.qt.app.register_app`."""
    return LayerViewer()


#: Display name, one-line description and the header copy the shipped
#: ``AppScreen`` tables want, written here beside the screen so that wiring
#: the app in is one call rather than five strings invented in five files.
#: :func:`spacr.qt.app.register_app` fans them out.
APP_NAME = "Layer Viewer"
APP_DESCRIPTION = "Images, masks, points and ROIs as separate layers in one world"
APP_INTRO = (
    "One world, many layers: an image channel, the label mask over it, the "
    "points and the shapes, each with its own colormap, opacity, blending "
    "and visibility, reordered by dragging. Picking an object here selects "
    "the same object in every other open view, and vice versa.")
#: What ``spacr.cli.INTERACTIVE_ONLY`` wants: why this app has no headless
#: run, and what to do instead.
APP_CLI_NOTE = (
    "Layer Viewer is an interactive image viewer — the layer stack, the "
    "blending and the picking are the whole feature; run it in the GUI "
    "(spacr-qt). Headless, build a spacr.layers stack from Python instead.")


#: Screens that ride in on this module's registration, as
#: ``(module, function)``. See :func:`register_companion_apps`.
COMPANION_APPS = (
    ("spacr.qt.screens.image_scatter", "register"),
    ("spacr.qt.screens.lineage", "register"),
    ("spacr.qt.screens.curate", "register"),
)


def register_companion_apps() -> tuple:
    """Register the screens built on this viewer's world. Idempotent.

    ``spacr.qt.app`` holds the one import-time table of self-registering
    modules (``_SELF_REGISTERING_APPS``) and calls
    :func:`register_layer_viewer_app` out of it. The screens in
    :data:`COMPANION_APPS` grew out of this module: they either borrow the
    layer world directly (a :class:`CanvasTool` on :class:`LayerCanvas`) or
    join the same linked-selection contract this viewer joined. Registering
    them from here rather than giving each one a row in ``app.py`` keeps the
    chain one hop long and written down in a single tuple — the next screen
    adds a line to it, not a mechanism.

    One companion's failure costs that companion and nothing else, the same
    posture ``app.py`` takes towards its own table and towards plugins: an
    optional screen must never stop the window opening.

    :returns: the module names that registered without raising.
    """
    import importlib

    registered = []
    for module_name, function_name in COMPANION_APPS:
        try:
            getattr(importlib.import_module(module_name), function_name)()
        except Exception:
            LOG.exception("Could not register the app owned by %s", module_name)
        else:
            registered.append(module_name)
    return tuple(registered)


def register_layer_viewer_app(*, section: Optional[str] = None,
                              stage: Optional[str] = None,
                              key: str = LAYER_VIEWER_APP_KEY):
    """Put the viewer in the app registry, through the public seam. Idempotent.

    Called at import from the bottom of :mod:`spacr.qt.app`, which is the
    only place a registration is visible to everybody — see
    ``_SELF_REGISTERING_APPS`` there for why it cannot be called at the top
    of this module.

    Everything after ``section`` is a table this key used to need a
    hand-edit in: the screen header and blurb, the "no headless run"
    sentence, the API doc link, and the display name in nine languages.
    :func:`spacr.qt.app.register_app` distributes them; this function only
    has to know them.

    :returns: the registry row that was added, or ``None`` when the key was
        already registered. Safe to call twice: this module is reachable
        from three import paths and a duplicate key would otherwise raise.
    """
    from .app import APPS, SECTION_EXPLORE, STAGE_ALPHA, register_app
    register_companion_apps()
    if any(row[0] == key for row in APPS):
        return None
    return register_app(
        key, APP_NAME, APP_DESCRIPTION,
        # Explore, not Results & QC: "page through image layers" is the
        # example in that section's own definition. Results & QC is what a
        # finished run produced; this is asking the images a question.
        section or SECTION_EXPLORE,
        factory=make_layer_viewer_screen,
        stage=STAGE_ALPHA if stage is None else stage,
        intro=APP_INTRO,
        cli_note=APP_CLI_NOTE,
        api_module="qt/layer_viewer",
        translations=("Lagervisare", "Ebenenansicht", "Visor de capas",
                      "图层查看器", "Visualizador de camadas", "लेयर व्यूअर",
                      "레이어 뷰어", "Lagaskoðari", "Visionneuse de calques"))
