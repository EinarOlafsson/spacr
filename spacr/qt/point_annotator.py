"""Make Masks' point mode: click one point per object (item 957, alpha).

The window steps through a :func:`spacr.point_annotation.build_point_queue`
session. Each image is shown on its own, blind: the outlines a scorer will
compare the points against are not on screen. The controls are:

* left click adds a point on the clicked pixel;
* right click removes the nearest point within a few pixels;
* ``U`` flips "unsure" on the point nearest the cursor, or on the last one;
* ``Ctrl+Z`` takes back the last change;
* the wheel zooms at the cursor, and the middle button, or Shift with the
  left button, pans, as in the mask editor; ``F`` fits the image again;
* ``Enter`` marks the image done and opens the next one not done, and the
  arrow keys step back and forward without marking.

Every change is saved at once by :class:`spacr.point_annotation.PointSession`.
An overlay folder can be shown only after every image of the folder is
marked done (or from the start when the session was opened with
``hide_overlays`` off). While an overlay is on screen the points cannot be
changed, so that the outlines never steer an annotation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (QDialog, QGraphicsScene, QGraphicsView,
                               QHBoxLayout, QLabel, QPushButton, QVBoxLayout)

from ..curation_queue import mark_state, read_status
from ..point_annotation import PointSession, all_done, annotations_folder, overlays_for

LOG = logging.getLogger(__name__)

__all__ = ["PointAnnotatorDialog"]

_SURE = QColor("#22d3ee")
_UNSURE = QColor("#facc15")
_MARKER_RADIUS = 3.5
_REMOVE_SCREEN_PX = 8.0
_ZOOM_STEP = 1.25


def _display_pixels(image: np.ndarray) -> QImage:
    """Pixels as Qt shows them: 8-bit as saved, anything else stretched.

    An 8-bit grey or RGB image is shown exactly as saved, since a crop made
    for annotation already carries its display stretch. Any other image is
    stretched linearly from its 1st to its 99.8th percentile.

    :param image: grey ``(h, w)`` or colour ``(h, w, 3|4)`` pixels.
    :returns: the image Qt draws.
    """
    pixels = np.asarray(image)
    if pixels.ndim == 3 and pixels.shape[-1] == 4:
        pixels = pixels[..., :3]
    if pixels.ndim == 3 and pixels.shape[-1] not in (3,):
        pixels = pixels[..., 0]
    if pixels.dtype != np.uint8:
        values = pixels.astype(np.float32)
        lo, hi = np.percentile(values, (1.0, 99.8))
        scale = 255.0 / (hi - lo) if hi > lo else 0.0
        pixels = np.clip((values - lo) * scale, 0, 255).astype(np.uint8)
    pixels = np.ascontiguousarray(pixels)
    height, width = pixels.shape[:2]
    if pixels.ndim == 2:
        return QImage(pixels.data, width, height, width,
                      QImage.Format_Grayscale8).copy()
    return QImage(pixels.data, width, height, 3 * width,
                  QImage.Format_RGB888).copy()


class _PointView(QGraphicsView):
    """The image, its points, and the mouse: click, zoom and pan."""

    add_requested = Signal(float, float)
    remove_requested = Signal(float, float, float)

    def __init__(self, parent=None):
        """An empty view; :meth:`show_image` puts an image in it."""
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QBrush(QColor("#0b1220")))
        self._pixmap = None
        self._markers: List = []
        self._pan_from = None
        self.cursor_pixel: Optional[QPointF] = None

    def show_image(self, image: QImage) -> None:
        """Replace the image, keeping the zoom when the size is the same.

        :param image: what to show.
        """
        scene = self.scene()
        same = (self._pixmap is not None
                and self._pixmap.pixmap().size() == image.size())
        if self._pixmap is None:
            self._pixmap = scene.addPixmap(QPixmap.fromImage(image))
            self._pixmap.setZValue(0)
        else:
            self._pixmap.setPixmap(QPixmap.fromImage(image))
        scene.setSceneRect(QRectF(0, 0, image.width(), image.height()))
        if not same:
            self.fit()

    def fit(self) -> None:
        """Fit the whole image in the view."""
        if self._pixmap is not None:
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

    def show_points(self, points) -> None:
        """Draw ``points`` over the image, unsure ones in yellow.

        :param points: :class:`spacr.point_annotation.Point` objects.
        """
        scene = self.scene()
        for marker in self._markers:
            scene.removeItem(marker)
        self._markers = []
        for point in points:
            colour = _UNSURE if point.unsure else _SURE
            pen = QPen(colour, 2.0)
            pen.setCosmetic(True)
            if point.unsure:
                pen.setStyle(Qt.DashLine)
            centre_x, centre_y = point.x + 0.5, point.y + 0.5
            marker = scene.addEllipse(
                centre_x - _MARKER_RADIUS, centre_y - _MARKER_RADIUS,
                2 * _MARKER_RADIUS, 2 * _MARKER_RADIUS, pen)
            marker.setZValue(2)
            self._markers.append(marker)

    def zoom_by(self, factor: float) -> None:
        """Zoom by ``factor`` about the cursor.

        :param factor: above 1 zooms in.
        """
        self.scale(factor, factor)

    def _pixel(self, event) -> QPointF:
        """The image position under a mouse event.

        :param event: a mouse or wheel event.
        :returns: the scene position, in image pixels.
        """
        return self.mapToScene(event.position().toPoint())

    def wheelEvent(self, event) -> None:
        """Zoom at the cursor, one step per wheel notch.

        :param event: the wheel event.
        """
        notches = event.angleDelta().y() or event.angleDelta().x()
        if notches:
            self.zoom_by(_ZOOM_STEP if notches > 0 else 1.0 / _ZOOM_STEP)
        event.accept()

    def mousePressEvent(self, event) -> None:
        """Add, remove, or start a pan.

        :param event: the mouse event.
        """
        shifted = bool(event.modifiers() & Qt.ShiftModifier)
        if event.button() == Qt.MiddleButton or (
                event.button() == Qt.LeftButton and shifted):
            self._pan_from = event.position()
            event.accept()
            return
        pixel = self._pixel(event)
        if event.button() == Qt.LeftButton:
            self.add_requested.emit(float(np.floor(pixel.x())),
                                    float(np.floor(pixel.y())))
        elif event.button() == Qt.RightButton:
            scale = abs(self.transform().m11()) or 1.0
            self.remove_requested.emit(pixel.x() - 0.5, pixel.y() - 0.5,
                                       max(_REMOVE_SCREEN_PX / scale, 1.0))
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        """Pan while dragging, and remember where the cursor is.

        :param event: the mouse event.
        """
        self.cursor_pixel = self._pixel(event)
        if self._pan_from is not None:
            delta = event.position() - self._pan_from
            self._pan_from = event.position()
            horizontal = self.horizontalScrollBar()
            vertical = self.verticalScrollBar()
            horizontal.setValue(horizontal.value() - int(delta.x()))
            vertical.setValue(vertical.value() - int(delta.y()))
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        """End a pan.

        :param event: the mouse event.
        """
        self._pan_from = None
        event.accept()


class PointAnnotatorDialog(QDialog):
    """Click one point per object on each image of a point queue.

    :param queue: the session :func:`spacr.point_annotation.build_point_queue`
        built.
    :param overlays: folders of overlay images, matched to each image by
        its points stem.
    :param hide_overlays: keep the overlays hidden until every image is
        marked done. Default True.
    :param parent: the parent widget.
    """

    def __init__(self, queue, overlays: Sequence = (), hide_overlays: bool = True,
                 parent=None) -> None:
        """Build the window and open the first image not done.

        :param queue: the point queue.
        :param overlays: folders of overlay images.
        :param hide_overlays: keep overlays hidden until every image is done.
        :param parent: the parent widget.
        """
        from .i18n import tr

        super().__init__(parent)
        self.setObjectName("PointAnnotatorDialog")
        self.setWindowTitle(tr("Point mode"))
        self.queue = queue
        self.items = list(queue.items)
        self.annotations = annotations_folder(queue.folder)
        self.overlay_folders = [Path(folder) for folder in overlays]
        self.hide_overlays = bool(hide_overlays)
        self.session: Optional[PointSession] = None
        self.index = -1
        self._image: Optional[QImage] = None
        self._overlays: List[Path] = []
        self._overlay_index = -1

        layout = QVBoxLayout(self)
        self._title = QLabel(self)
        self._title.setObjectName("PointAnnotatorTitle")
        layout.addWidget(self._title)
        self.view = _PointView(self)
        self.view.add_requested.connect(self._on_add)
        self.view.remove_requested.connect(self._on_remove)
        layout.addWidget(self.view, 1)
        self._count = QLabel(self)
        self._count.setObjectName("PointAnnotatorCount")
        layout.addWidget(self._count)
        self._status = QLabel(self)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        row = QHBoxLayout()
        self._btn_previous = QPushButton(tr("Previous"), self)
        self._btn_previous.clicked.connect(lambda: self.open_index(self.index - 1))
        self._btn_next = QPushButton(tr("Next"), self)
        self._btn_next.clicked.connect(lambda: self.open_index(self.index + 1))
        self._btn_undo = QPushButton(tr("Undo"), self)
        self._btn_undo.clicked.connect(self._on_undo)
        self._btn_fit = QPushButton(tr("Fit"), self)
        self._btn_fit.clicked.connect(self.view.fit)
        self._btn_overlay = QPushButton(tr("Show overlay"), self)
        self._btn_overlay.setCheckable(True)
        self._btn_overlay.toggled.connect(self._on_overlay_toggled)
        self._btn_done = QPushButton(tr("Done, next image"), self)
        self._btn_done.setDefault(True)
        self._btn_done.clicked.connect(self.mark_done)
        for button in (self._btn_previous, self._btn_next, self._btn_undo,
                       self._btn_fit, self._btn_overlay):
            row.addWidget(button)
        row.addStretch(1)
        row.addWidget(self._btn_done)
        layout.addLayout(row)
        self.setToolTip(tr(
            "Left click adds a point, right click removes the nearest one, U "
            "marks the nearest or last point unsure, Ctrl+Z undoes, the wheel "
            "zooms and the middle button or Shift-drag pans. Every change is "
            "saved to the annotations folder beside this one."))

        for keys, slot in (
                ("U", self._on_toggle_unsure),
                (QKeySequence.Undo, self._on_undo),
                ("F", self.view.fit),
                (Qt.Key_Return, self.mark_done),
                (Qt.Key_Enter, self.mark_done),
                (Qt.Key_Right, lambda: self.open_index(self.index + 1)),
                (Qt.Key_Left, lambda: self.open_index(self.index - 1)),
                ("O", self._on_cycle_overlay),
                ("+", lambda: self.view.zoom_by(_ZOOM_STEP)),
                ("-", lambda: self.view.zoom_by(1.0 / _ZOOM_STEP))):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.activated.connect(slot)

        self.resize(900, 960)
        status = read_status(queue.folder)
        start = next((i for i, item in enumerate(self.items)
                      if not (item.stem in status
                              and status[item.stem].state == "done")), 0)
        if self.items:
            self.open_index(start)

    def _overlays_allowed(self) -> bool:
        """Whether an overlay may be shown now.

        :returns: True when overlays are not hidden, or every image is done.
        """
        return not self.hide_overlays or all_done(self.queue)

    def open_index(self, index: int) -> bool:
        """Open the image at ``index`` of the session.

        :param index: position in the session; out of range does nothing.
        :returns: whether an image was opened.
        """
        if not 0 <= index < len(self.items):
            return False
        from .mask_engine import read_image

        item = self.items[index]
        pixels = read_image(str(item.image))
        self._image = _display_pixels(pixels)
        shape = (self._image.height(), self._image.width())
        self.session = PointSession(item.image, self.annotations, shape=shape)
        self.index = index
        self._overlays = overlays_for(item.image, self.overlay_folders)
        self._overlay_index = -1
        self._btn_overlay.blockSignals(True)
        self._btn_overlay.setChecked(False)
        self._btn_overlay.blockSignals(False)
        self.view.show_image(self._image)
        self._refresh()
        return True

    def mark_done(self) -> None:
        """Save this image's points, mark it done, and open the next one not done."""
        from .i18n import tr

        if self.session is None:
            return
        self.session.save()
        item = self.items[self.index]
        mark_state(self.queue.folder, item.stem, "done",
                   n_objects=self.session.count)
        status = read_status(self.queue.folder)
        order = list(range(self.index + 1, len(self.items))) + list(range(self.index))
        waiting = [i for i in order
                   if not (self.items[i].stem in status
                           and status[self.items[i].stem].state == "done")]
        if waiting:
            self.open_index(waiting[0])
        else:
            self._refresh()
            self._status.setText(tr(
                "Every image is marked done. The overlays can be shown now; "
                "the points cannot be changed while one is on screen."))

    def _editable(self) -> bool:
        """Whether the points can be changed: not while an overlay is shown."""
        return self.session is not None and self._overlay_index < 0

    def _on_add(self, x: float, y: float) -> None:
        """Add a point where the view was clicked."""
        if self._editable() and self.session.add(x, y) is not None:
            self._refresh()

    def _on_remove(self, x: float, y: float, radius: float) -> None:
        """Remove the point nearest a right click."""
        if self._editable() and self.session.remove_nearest(x, y, radius) is not None:
            self._refresh()

    def _on_toggle_unsure(self) -> None:
        """Flip "unsure" on the point nearest the cursor, or the last one."""
        if not self._editable():
            return
        cursor = self.view.cursor_pixel
        if cursor is None:
            changed = self.session.toggle_unsure()
        else:
            scale = abs(self.view.transform().m11()) or 1.0
            changed = self.session.toggle_unsure(
                cursor.x() - 0.5, cursor.y() - 0.5,
                max(_REMOVE_SCREEN_PX / scale, 1.0))
        if changed is not None:
            self._refresh()

    def _on_undo(self) -> None:
        """Take back the last change."""
        if self._editable() and self.session.undo():
            self._refresh()

    def _on_overlay_toggled(self, checked: bool) -> None:
        """Show the first overlay, or the image again."""
        self._overlay_index = 0 if checked and self._overlays else -1
        self._show_current()

    def _on_cycle_overlay(self) -> None:
        """Step through this image's overlays and back to the image."""
        if not (self._overlays and self._overlays_allowed()):
            return
        self._overlay_index += 1
        if self._overlay_index >= len(self._overlays):
            self._overlay_index = -1
        self._btn_overlay.blockSignals(True)
        self._btn_overlay.setChecked(self._overlay_index >= 0)
        self._btn_overlay.blockSignals(False)
        self._show_current()

    def _show_current(self) -> None:
        """Show the image or the chosen overlay, with the points over it."""
        from .mask_engine import read_image

        if self._image is None:
            return
        if self._overlay_index >= 0:
            path = self._overlays[self._overlay_index]
            self.view.show_image(_display_pixels(read_image(str(path))))
        else:
            self.view.show_image(self._image)
        self._refresh()

    def _refresh(self) -> None:
        """Bring the title, count, points and buttons up to date."""
        from .i18n import tr

        if self.session is None:
            return
        item = self.items[self.index]
        status = read_status(self.queue.folder)
        done = sum(1 for each in self.items
                   if each.stem in status and status[each.stem].state == "done")
        state = (tr("done") if item.stem in status
                 and status[item.stem].state == "done" else tr("not done"))
        self._title.setText(tr("{name} ({position} of {total}, {state}); {done} of {total} done").format(
            name=item.image.name, position=self.index + 1,
            total=len(self.items), state=state, done=done))
        self._count.setText(tr("{count} points, {unsure} unsure").format(
            count=self.session.count, unsure=self.session.unsure_count))
        self.view.show_points(self.session.points)
        allowed = self._overlays_allowed() and bool(self._overlays)
        self._btn_overlay.setEnabled(allowed)
        editable = self._editable()
        self._btn_undo.setEnabled(editable)
        self._btn_previous.setEnabled(self.index > 0)
        self._btn_next.setEnabled(self.index < len(self.items) - 1)
        if self._overlay_index >= 0:
            self._status.setText(tr("Showing {overlay}. The points cannot be changed while an overlay is on screen.").format(
                overlay=self._overlays[self._overlay_index].name))
        elif not self._overlays_allowed():
            self._status.setText(tr("Blind: the overlays stay hidden until every image is marked done. Points are saved to {path}.").format(
                path=self.session.path))
        else:
            self._status.setText(tr("Points are saved to {path}.").format(
                path=self.session.path))
