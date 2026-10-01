"""A shared line ruler that measures image coordinates without changing pixels."""
from math import hypot, isfinite

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPen

from ..i18n import tr
from ..theme import active_palette


class ImageRuler(QObject):
    """Keep a line in image pixels and paint it through the host's transform.

    :param parent: owning image canvas or view; defaults to None.

    Hosts pass their widget-to-image mapping to :meth:`handle` and inverse
    mapping to :meth:`paint`. Zoom and pan never change the measured length.
    Pixel units are always shown. Physical units require validated
    calibration, either explicit through :meth:`set_spacing` or stated by the
    image file's own header through :meth:`calibrate_from_file`; camera
    magnification is never guessed. Right-click clears while the tool is
    active.
    """

    changed = Signal()

    def __init__(self, parent=None):
        """Start with no line, no calibration and drawing disabled."""
        super().__init__(parent)
        self.active = False
        self.start = None
        self.end = None
        self._drawing = False
        self.spacing = None
        self.unit = 'µm'

    def clear(self):
        """Remove the line without changing tool activation or calibration.

        :returns: None; emits ``changed`` even when no line was present.
        """
        self.start = self.end = None
        self._drawing = False
        self.changed.emit()

    def set_active(self, active):
        """Enable drawing; disabling preserves the finished line.

        :param active: whether unmodified left drags measure instead of edit.
        :returns: None; emits ``changed`` after updating activation.
        """
        self.active = bool(active)
        self._drawing = False
        self.changed.emit()

    def set_spacing(self, x=None, y=None, unit='µm'):
        """Set physical distance per image pixel, or clear calibration.

        :param x: positive finite distance per horizontal image pixel in ``unit``;
            defaults to None, which clears calibration and ignores ``y``.
        :param y: positive finite distance per vertical image pixel in ``unit``;
            defaults to None, which uses ``x`` for both axes.
        :param unit: physical unit label; defaults to ``'µm'``. The caller supplies
            spacing in this unit; this method does not convert units.
        :raises ValueError: when spacing is nonpositive or nonfinite.
        :returns: None; emits ``changed`` after updating calibration.
        """
        if x is None:
            self.spacing = None
        else:
            spacing = (float(x), float(x if y is None else y))
            if any(not isfinite(v) or v <= 0 for v in spacing):
                raise ValueError(tr('Pixel spacing must be finite and positive.'))
            self.spacing = spacing
        self.unit = str(unit)
        self.changed.emit()

    def calibrate_from_file(self, path, shape=None):
        """Take the pixel spacing the shown image's own file header states.

        Only a size the file states counts (OME ``PhysicalSizeX/Y``, an
        ImageJ micron calibration or centimetre resolution tags, read by
        :func:`spacr.point_spread.image_optics_metadata`); a magnification
        in the file name, an objective table or a default never calibrates
        the ruler. Calibration is cleared first, so a file that states
        nothing leaves the ruler in pixels.

        :param path: the image file being shown; None or '' only clears.
        :param shape: the displayed array's shape; defaults to None, which
            skips the check. When given, the header's (Y, X) must equal its
            first two or last two dimensions, so a resampled or cropped
            display is never measured with the file's spacing.
        :returns: the (x, y) spacing in µm now set, or None when the ruler
            stays uncalibrated.
        """
        self.set_spacing()
        if not path:
            return None
        from ...point_spread import image_optics_metadata
        try:
            stated = image_optics_metadata(path)
        except Exception:
            return None
        size = stated.get('pixel_size_um')
        if size is None:
            return None
        if shape is not None:
            header = stated.get('image_shape')
            dims = tuple(int(n) for n in shape)
            if header is None or tuple(int(n) for n in header.value) not in (dims[:2], dims[-2:]):
                return None
        try:
            y, x = size.value
            self.set_spacing(x, y, unit='µm')
        except (TypeError, ValueError):
            self.set_spacing()
            return None
        return self.spacing

    def length(self, physical=False):
        """Return the line length, or None before a line exists.

        :param physical: False (default) uses image pixels; True uses calibrated
            per-axis spacing, including different horizontal and vertical values.
        :returns: Euclidean length as a float in pixels or ``unit``; None when
            no line exists or physical length is requested without calibration.
        """
        if self.start is None or self.end is None or (physical and self.spacing is None):
            return None
        dx, dy = (self.end[i] - self.start[i] for i in range(2))
        sx, sy = self.spacing if physical else (1.0, 1.0)
        return hypot(dx * sx, dy * sy)

    def label(self):
        """Return pixel length and, only when calibrated, physical length.

        :returns: text with lengths to two decimal places; empty before a line exists.
        """
        length = self.length()
        if length is None:
            return ''
        text = tr('{length:.2f} px', length=length)
        physical = self.length(physical=True)
        if physical is not None:
            text += tr(' · {length:.2f} {unit}', length=physical, unit=self.unit)
        return text

    def handle(self, event, to_image):
        """Consume ruler mouse gestures; return False for navigation gestures.

        :param event: a mouse press, move or release from the host canvas.
        :param to_image: widget QPointF -> image (x, y), or None off-image.
        :returns: True for consumed ruler gestures; False when the host should
            handle the event. Coordinates outside the image do not move endpoints.
        """
        if not self.active:
            return False
        kind = event.type()
        if kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
            if event.button() == Qt.RightButton and not event.modifiers():
                if kind == QEvent.MouseButtonPress:
                    self.clear()
                event.accept()
                return True
        if kind == QEvent.MouseButtonPress:
            if event.button() != Qt.LeftButton or event.modifiers():
                return False
            point = to_image(event.position())
            if point is not None:
                self.start = self.end = tuple(map(float, point))
                self._drawing = True
                self.changed.emit()
            event.accept()
            return True
        if self._drawing and kind in (QEvent.MouseMove, QEvent.MouseButtonRelease):
            if kind == QEvent.MouseButtonRelease and event.button() != Qt.LeftButton:
                return False
            point = to_image(event.position())
            if point is not None:
                self.end = tuple(map(float, point))
            if kind == QEvent.MouseButtonRelease:
                self._drawing = False
            self.changed.emit()
            event.accept()
            return True
        return False

    def paint(self, painter, to_widget):
        """Draw endpoints, line and readout in widget coordinates.

        :param painter: an active painter for the host canvas/viewport.
        :param to_widget: image (x, y) -> widget QPointF or QPoint, or None.
        :returns: None; draws nothing when a line or either mapped endpoint is missing.
        """
        if self.start is None or self.end is None:
            return
        a, b = to_widget(*self.start), to_widget(*self.end)
        if a is None or b is None:
            return
        a, b = QPointF(a), QPointF(b)
        palette = active_palette()
        painter.save()
        pen = QPen(QColor(palette['accent']), 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLine(a, b)
        painter.setBrush(QColor(palette['fg']))
        painter.drawEllipse(a, 3, 3)
        painter.drawEllipse(b, 3, 3)
        text = self.label()
        metrics = painter.fontMetrics()
        width, height = metrics.horizontalAdvance(text) + 14, metrics.height() + 8
        device = painter.device()
        x = max(0, min(device.width() - width, (a.x() + b.x()) / 2 + 8))
        y = max(0, min(device.height() - height, (a.y() + b.y()) / 2 + 8))
        badge = QRectF(x, y, width, height)
        background = QColor(palette['bg'])
        background.setAlpha(210)
        painter.setPen(Qt.NoPen)
        painter.setBrush(background)
        painter.drawRoundedRect(badge, 4, 4)
        painter.setPen(QColor(palette['fg']))
        painter.drawText(badge, Qt.AlignCenter, text)
        painter.restore()
