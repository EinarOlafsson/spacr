"""Render first-run questions in a rounded, pointer-responsive card.

The accent follows the corner nearest the pointer. All visual effects are
decorative: if a blur, background, or accent cannot be painted, the card
continues to display its questions and save their answers.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

#: Corner order: top-left, top-right, bottom-right, bottom-left.
CORNERS = ("topLeft", "topRight", "bottomRight", "bottomLeft")


class SetupCard(QWidget):
    """A rounded, translucent card whose border lights at the near corner.

    :param radius: corner radius in pixels.
    :param arc: how far along each edge the accent runs, in pixels.
    """

    def __init__(self, parent: Optional[QWidget] = None, *,
                 radius: int = 18, arc: int = 90):
        super().__init__(parent)
        self._radius = int(radius)
        self._arc = int(arc)
        self._corner = 0
        #: WHERE THE ACCENT IS, as a fraction of the way round the rim
        #: clockwise from the top-left. Continuous rather than one of four
        #: corners (instruction 234): "the blue rim should flow like water
        #: towards the mouse", and water does not teleport between corners.
        self._at = 0.0
        #: Where it is heading. The gap between the two is what the easing
        #: closes, one tick at a time.
        self._towards = 0.0
        #: Laps still to run, signed: + is clockwise, - anticlockwise.
        #: WHILE THIS IS NON-ZERO THE POINTER DOES NOT STEER. A lap dragged
        #: off course by a mouse movement is not a lap, and the user cannot
        #: tell whether it went round -- which is the whole signal.
        self._laps = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(16)             # ~60fps
        self._timer.timeout.connect(self._tick)
        # MOUSE TRACKING, or `mouseMoveEvent` only fires while a button is
        # held -- which is never, on a card the user is only reading.
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_Hover, True)

    # ------------------------------------------------------------------
    def nearest_corner(self, point: QPointF) -> int:
        """Index into :data:`CORNERS` of the corner nearest ``point``.

        Ties go to the earlier corner, which only happens at the exact
        centre and is therefore never seen.
        """
        rect = QRectF(self.rect())
        points = (rect.topLeft(), rect.topRight(),
                  rect.bottomRight(), rect.bottomLeft())
        best, chosen = None, 0
        for index, corner in enumerate(points):
            dx = point.x() - corner.x()
            dy = point.y() - corner.y()
            distance = dx * dx + dy * dy
            if best is None or distance < best:
                best, chosen = distance, index
        return chosen

    def corner(self) -> str:
        """The corner the accent is currently on."""
        return CORNERS[self._corner]

    # ------------------------------------------------------------------
    def mouseMoveEvent(self, event):            # noqa: N802 - Qt naming
        self._follow(event.position())
        super().mouseMoveEvent(event)

    def event(self, event):
        # A hover move arrives even when the widget has no mouse grab, which
        # is the ordinary case here.
        try:
            from PySide6.QtCore import QEvent

            if event.type() == QEvent.Type.HoverMove:
                self._follow(event.position())
        except Exception:                        # pragma: no cover
            pass
        return super().event(event)

    def _follow(self, position) -> None:
        corner = self.nearest_corner(QPointF(position))
        if corner != self._corner:
            self._corner = corner
            # ONLY THE BORDER REPAINTS. The card is translucent over a
            # blurred backdrop, and repainting that on every mouse move is
            # the difference between a smooth accent and a stuttering one.
            self.update()

    # ------------------------------------------------------------------
    def paintEvent(self, event):                # noqa: N802 - Qt naming
        try:
            self._paint()
        except Exception:                        # pragma: no cover
            # Decoration is never load-bearing: an unpainted card is still a
            # card with working controls on it.
            pass
        super().paintEvent(event)

    # ---------------------------------------------------- where it flows to

    def perimeter_position(self, point: QPointF) -> float:
        """Where on the rim ``point`` is, as a fraction clockwise from 0.

        THE RIM IS TREATED AS A RECTANGLE, not as the rounded path it is
        drawn on. The corner arcs are a few pixels of a perimeter hundreds
        long, and paying for exact arc-length here would buy an accuracy no
        eye can see on a value that is chasing a mouse anyway.
        """
        rect = QRectF(self.rect())
        width = max(1.0, rect.width())
        height = max(1.0, rect.height())
        total = 2.0 * (width + height)
        x = min(max(float(point.x()), 0.0), width)
        y = min(max(float(point.y()), 0.0), height)
        # Which edge is nearest, and how far along it.
        gaps = ((y, x), (width - x, width + y),
                (height - y, width + height + (width - x)),
                (x, 2.0 * width + height + (height - y)))
        _gap, run = min(gaps, key=lambda pair: pair[0])
        return (run / total) % 1.0

    def flow_towards(self, point: QPointF) -> None:
        """Aim the accent at ``point``. Ignored while a circuit is running."""
        if self._laps:
            return
        self._towards = self.perimeter_position(point)
        self._start()

    def circuit(self, clockwise: bool = True) -> None:
        """Send the accent once round: clockwise, or anti- for Previous.

        THE DIRECTION IS THE MESSAGE. It tells the user which way they went
        through the slides, which is worth more than the animation -- so a
        circuit is never merged with another or shortened to catch up.
        """
        self._laps += 1.0 if clockwise else -1.0
        self._start()

    @property
    def position(self) -> float:
        """The accent's place on the rim, 0..1 clockwise from the top-left."""
        return self._at % 1.0

    @property
    def spinning(self) -> bool:
        """Whether a circuit is running."""
        return bool(self._laps)

    def _start(self) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def _tick(self) -> None:
        """One frame: run the laps down, else ease towards the pointer."""
        if self._laps:
            step = 0.03 if self._laps > 0 else -0.03
            self._at += step
            self._laps -= step
            # A LAP ENDS EXACTLY, not approximately: floating error across
            # thirty-odd frames would otherwise leave the accent a little
            # further round after every circuit, and after ten slides it
            # would be somewhere the pointer never put it.
            if abs(self._laps) < 0.031:
                # THE REMAINDER IS TRAVELLED, not undone: `_laps` is what is
                # still owed, so the last partial step ADDS it. Subtracting
                # left the accent 2% short of home on every circuit, which
                # after ten slides is a fifth of the way round from where
                # the pointer last put it.
                self._at = round(self._at + self._laps, 6)
                self._laps = 0.0
        else:
            gap = ((self._towards - self._at + 0.5) % 1.0) - 0.5
            if abs(gap) < 0.002:
                self._at = self._towards
                self._timer.stop()
            else:
                self._at += gap * 0.18      # ease, not jump: it is water
        self._corner = int((self.position + 0.125) % 1.0 * 4) % 4
        self.update()

    def _paint(self) -> None:
        from ..theme import active_palette

        palette = active_palette()
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            rect = QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)

            body = QColor(palette.get("surface", palette["bg"]))
            body.setAlpha(216)          # translucent: the blobs show through
            painter.setPen(Qt.NoPen)
            painter.setBrush(body)
            painter.drawRoundedRect(rect, self._radius, self._radius)

            edge = QColor(palette["fg"])
            edge.setAlpha(38)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(edge, 1.0))
            painter.drawRoundedRect(rect, self._radius, self._radius)

            # THE ACCENT, a run of rim centred on `position`.
            #
            # DRAWN AS A STROKE OF THE ROUNDED PATH ITSELF, clipped to a
            # length, rather than as the four hand-built corner paths below.
            # A continuous position cannot be expressed as one of four
            # corners, and `QPainterPathStroker` would give the outline of
            # the stroke rather than a segment of it -- so the segment comes
            # from `QPainterPath.pointAtPercent` along the whole rim, which
            # is the one thing Qt measures in arc length for us.
            painter.setPen(QPen(QColor(palette["accent"]), 2.0,
                                Qt.SolidLine, Qt.RoundCap))
            painter.drawPath(self._accent_path(rect))
        finally:
            painter.end()

    def _accent_path(self, rect: QRectF) -> QPainterPath:
        """The lit run of rim, centred on :attr:`position`.

        THE WHOLE RIM IS BUILT ONCE and sampled along its length, so the run
        crosses a corner without the seam four separate corner paths have --
        which is what makes it read as flowing rather than as switching.
        """
        rim = QPainterPath()
        rim.addRoundedRect(rect, self._radius, self._radius)
        total = max(1.0, rim.length())
        # The arc is a length in pixels; as a fraction of the rim it depends
        # on how big the card is, which is right -- a fixed FRACTION would
        # be a hairline on a large card and half the rim on a small one.
        span = min(0.45, max(0.04, float(self._arc) * 2.0 / total))
        start = self.position - span / 2.0

        out = QPainterPath()
        steps = 48
        for index in range(steps + 1):
            at = (start + span * index / steps) % 1.0
            point = rim.pointAtPercent(at)
            if index == 0:
                out.moveTo(point)
            else:
                out.lineTo(point)
        return out

    def _corner_path(self, rect: QRectF) -> QPainterPath:
        """The two edge runs meeting at the current corner, with its arc."""
        radius, arc = float(self._radius), float(self._arc)
        path = QPainterPath()
        name = CORNERS[self._corner]
        if name == "topLeft":
            box = QRectF(rect.left(), rect.top(), radius * 2, radius * 2)
            path.moveTo(rect.left(), rect.top() + radius + arc)
            path.lineTo(rect.left(), rect.top() + radius)
            path.arcTo(box, 180, -90)
            path.lineTo(rect.left() + radius + arc, rect.top())
        elif name == "topRight":
            box = QRectF(rect.right() - radius * 2, rect.top(),
                         radius * 2, radius * 2)
            path.moveTo(rect.right() - radius - arc, rect.top())
            path.lineTo(rect.right() - radius, rect.top())
            path.arcTo(box, 90, -90)
            path.lineTo(rect.right(), rect.top() + radius + arc)
        elif name == "bottomRight":
            box = QRectF(rect.right() - radius * 2, rect.bottom() - radius * 2,
                         radius * 2, radius * 2)
            path.moveTo(rect.right(), rect.bottom() - radius - arc)
            path.lineTo(rect.right(), rect.bottom() - radius)
            path.arcTo(box, 0, -90)
            path.lineTo(rect.right() - radius - arc, rect.bottom())
        else:
            box = QRectF(rect.left(), rect.bottom() - radius * 2,
                         radius * 2, radius * 2)
            path.moveTo(rect.left() + radius + arc, rect.bottom())
            path.lineTo(rect.left() + radius, rect.bottom())
            path.arcTo(box, 270, -90)
            path.lineTo(rect.left(), rect.bottom() - radius - arc)
        return path
