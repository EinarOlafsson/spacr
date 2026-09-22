"""One image view that fits, zooms, pans -- and can be tied to another.

spaCR had a fit-on-load, wheel-zoom, drag-pan image canvas already: the QC
field browser's own ``_FieldView``. Item 473's raw-versus-enhanced window
needed the same thing twice over, and a third copy of "a QGraphicsView with
a pixmap in it" is how a codebase ends up with three that behave subtly
differently. So the field browser's view moved here unchanged, under a name
that says it is shared, and the browser goes on using it under its old one.

WHAT IS NEW HERE IS THE TIE. Two views of the same region are only worth
putting side by side if they stay in register: a curator comparing a field
with its enhanced self is looking for a difference between them, and a
difference in where the two are pointing is the one difference that is not
information. :meth:`ZoomableImageView.link_to` makes a group of views share
one zoom and one scroll position, so zooming or panning any of them moves
all of them.

The tie is on the VIEW and not on the window, because what has to match is
the transform and the two scroll bars -- the things a view owns. Views in a
group must show images of the same size for the tie to mean anything; the
compare window builds both of its pictures from one region, so they do.
"""
from __future__ import annotations

from typing import List

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView

#: How far one wheel notch zooms. The QC field browser's number, kept so
#: the two views feel the same.
ZOOM_STEP = 1.2


class ZoomableImageView(QGraphicsView):
    """Fit-on-load image canvas with wheel zoom and drag panning.

    :param parent: parent widget; ownership only.
    """

    def __init__(self, parent=None) -> None:
        """Build the view with its own scene, fitted on first load."""
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = None
        self._user_zoomed = False
        #: The other views this one moves with. See :meth:`link_to`.
        self._linked: List["ZoomableImageView"] = []
        #: True while this view is being moved BY a peer, so the move is
        #: not passed back and the two do not chase each other.
        self._following = False
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.horizontalScrollBar().valueChanged.connect(self._scrolled)
        self.verticalScrollBar().valueChanged.connect(self._scrolled)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        """Show a new image, fitted, and forget any zoom the user had applied.

        :param pixmap: the composite to show; a null one clears the view.
        """
        self._scene.clear()
        self._item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._user_zoomed = False
        self.resetTransform()
        if not pixmap.isNull():
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def clear_image(self) -> None:
        """Empty the view."""
        self._scene.clear()
        self._item = None

    def link_to(self, other: "ZoomableImageView") -> None:
        """Move with ``other``, and make ``other`` move with this one.

        Symmetric and transitive by construction: every view already in
        either group joins the other, so a group of three all follow each
        other however they were linked up.

        :param other: the view to tie this one to.
        """
        if other is self:
            return
        group = {id(self): self, id(other): other}
        for view in list(self._linked) + list(other._linked):
            group.setdefault(id(view), view)
        for view in group.values():
            view._linked = [peer for peer in group.values() if peer is not view]

    def zoom_by(self, factor: float) -> None:
        """Zoom by ``factor`` about the view's middle, and move the peers.

        The wheel zooms about the cursor (``AnchorUnderMouse``); a button
        has no cursor position to speak of, so it zooms about the middle,
        which is what a person pressing + expects to keep in view.

        :param factor: above 1 zooms in, below 1 zooms out.
        """
        anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.scale(float(factor), float(factor))
        self.setTransformationAnchor(anchor)
        self._user_zoomed = True
        self._push_to_peers()

    def fit(self) -> None:
        """Fit the whole image in the view again, and the peers with it."""
        if self._item is None:
            return
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._user_zoomed = False
        self._push_to_peers()

    def zoom_factor(self) -> float:
        """How far this view is zoomed in, as a scale factor."""
        return float(self.transform().m11())

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Zoom on the wheel, and remember that the user did.

        Once they have zoomed, a resize stops re-fitting -- a view that snapped
        back to fit every time the splitter moved would undo the inspection the
        zoom was for.

        :param event: the wheel event.
        """
        factor = ZOOM_STEP if event.angleDelta().y() > 0 else (1.0 / ZOOM_STEP)
        self.scale(factor, factor)
        self._user_zoomed = True
        self._push_to_peers()
        event.accept()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Re-fit the image, unless the user has zoomed.

        :param event: the resize event.
        """
        super().resizeEvent(event)
        if not self._user_zoomed and self._item is not None:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def _scrolled(self, _value: int) -> None:
        """A scroll bar moved: drag the peers along, unless one moved us."""
        if not self._following:
            self._push_to_peers()

    def _push_to_peers(self) -> None:
        """Copy this view's zoom and scroll position onto the linked views.

        The transform is copied rather than re-derived, so a rounding
        difference cannot accumulate into a drift between two views that
        are supposed to be showing the same pixels.
        """
        if self._following or not self._linked:
            return
        transform = self.transform()
        horizontal = self.horizontalScrollBar().value()
        vertical = self.verticalScrollBar().value()
        zoomed = self._user_zoomed
        for view in self._linked:
            if view._following:
                continue
            view._following = True
            try:
                if view.transform() != transform:
                    view.setTransform(transform)
                view._user_zoomed = zoomed
                view.horizontalScrollBar().setValue(horizontal)
                view.verticalScrollBar().setValue(vertical)
            finally:
                view._following = False
